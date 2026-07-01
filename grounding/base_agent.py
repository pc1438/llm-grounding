"""
base_agent.py — Tool-use loop implementations, organized by API shape.

This file owns the loop logic for each distinct API contract. Agent files
(agents/<provider>.py) inherit the right class here and own everything
specific to their provider: client setup, model defaults, quirks.

Classes:
    BaseAgent               — ABC defining the stream() + ask() contract
    AnthropicAgent          — Anthropic Messages API (Claude)
    OpenAICompatibleAgent   — OpenAI chat.completions (Kimi, Llama)
    OpenAIResponsesAgent    — OpenAI Responses API, previous_response_id
                              chaining, parallel tool calls (GPT, Qwen)

Organized by API shape, not by provider. A new provider that speaks the
Responses API gets a new agent file inheriting OpenAIResponsesAgent —
nothing here changes. See decisions/007_agent_architecture.md.

Public interface:
    stream(question) → Generator[dict]
        Yields structured event dicts as the loop progresses.
        Consumed by run.py to produce SSE events for the UI.

    ask(question, on_progress=None) → dict
        Synchronous wrapper around stream(). Fires on_progress(str) at
        each step. Returns the final stats dict. Used by compare.py and
        programmatic callers.

Event shapes from stream():
    {"event": "tool_call",     "round": int, "search_num": int, "query": str, "params": dict}
    {"event": "search_result", "round": int, "search_num": int, "result_count": int,
                               "latency_ms": float, "sources": list, "search_uuid": str}
    {"event": "answer",        "text": str, "sources": list}
    {"event": "done",          "stats": dict}

Stats dict shape (all agents return the same structure):
    answer, sources, tool_calls, model, interface, tokens_used,
    token_breakdown {input, output, search_context},
    search_calls, api_calls, latency_ms

What does NOT belong here:
    Client instantiation, API keys, base URLs, model defaults, provider
    quirks (e.g. Qwen's you_search tool name). Those belong in the agent file.
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Generator

from search_tool import (
    INTEGRATION_INTERFACE,
    get_system_prompt,
    MAX_TOKENS,
    MAX_TOOL_ROUNDS,
    TOOL_SCHEMA,
    TOOL_SCHEMA_ANTHROPIC,
    TOOL_SCHEMA_RESPONSES,
    build_tool_log_entry,
    execute_search,
    extract_urls,
    extract_search_uuid,
    is_verbose,
    format_tool_log,
)

logger = logging.getLogger(__name__)


# ─── Shared utilities ─────────────────────────────────────────────────────────

def _empty_stats(model: str = "") -> dict:
    """Return a fresh stats dict. All loop implementations use this shape.

    Fields:
        answer           — final answer text (with [N] citation markers if applicable)
        sources          — ordered list of source URLs cited in the answer
        tool_calls       — YDC search log entries (empty for native search paths)
        model            — model ID as configured by the caller
        model_confirmed  — model ID as confirmed by the API response (may differ)
        interface        — which search integration was used: "ydc", "native_responses",
                           "native_chat", or the INTEGRATION_INTERFACE constant
        tokens_used      — total tokens (input + output)
        token_breakdown  — per-category breakdown: input, output, search_context
        search_calls     — number of search round-trips executed
        api_calls        — number of LLM API calls made
        latency_ms       — wall-clock time from first request to final answer
    """
    return {
        "answer": "",
        "sources": [],
        "tool_calls": [],
        "model": model,
        "model_confirmed": None,
        "interface": INTEGRATION_INTERFACE,
        "tokens_used": 0,
        "token_breakdown": {"input": 0, "output": 0, "search_context": 0},
        "search_calls": 0,
        "api_calls": 0,
        "latency_ms": 0.0,
    }


def _event_to_message(event: dict) -> str:
    """Convert a structured event dict to a display string for on_progress callbacks."""
    e = event.get("event")
    if e == "tool_call":
        query = event.get("query", "")
        params = event.get("params", {})
        param_str = " • ".join(f"{k}={v}" for k, v in params.items() if v)
        label = query + (" • " + param_str if param_str else "")
        return f"Search {event.get('search_num', '?')}: {label}"
    if e == "search_result":
        return f"Results received ({event.get('latency_ms', 0):.0f}ms)"
    if e == "answer":
        return "Generating answer..."
    return ""


# ─── Base class ───────────────────────────────────────────────────────────────

class BaseAgent(ABC):
    """Abstract base for all grounded LLM agents.

    Subclasses implement stream(). The ask() method is provided here and
    must not be overridden — it is the single conversion point from the
    streaming interface to the synchronous dict interface.
    """

    def __init__(self, model: str, system_prompt: str = None):
        self.model = model
        self.system_prompt = system_prompt if system_prompt is not None else get_system_prompt()

    @abstractmethod
    def stream(self, question: str) -> Generator[dict, None, None]:
        """Run the tool-use loop, yielding structured events at each step.

        Must always yield "answer" followed by "done" as the last two events,
        even on error. Callers rely on "done" to know the loop has finished.
        """
        ...

    def ask(self, question: str, on_progress=None) -> dict:
        """Synchronous wrapper around stream().

        Consumes all events, fires on_progress(str) for each step, and
        returns the stats dict from the final "done" event.

        on_progress is fire-and-forget — exceptions are silenced so they
        cannot affect the loop or the return value.
        """
        stats = None
        try:
            for event in self.stream(question):
                if on_progress:
                    msg = _event_to_message(event)
                    if msg:
                        try:
                            on_progress(msg)
                        except Exception:
                            pass
                if event["event"] == "done":
                    stats = event["stats"]
        except Exception as e:
            logger.error("stream() raised unexpectedly (model=%s): %s", self.model, e)
            if stats is None:
                stats = _empty_stats(self.model)
                stats["answer"] = ""
        return stats


# ─── Anthropic Messages API (Claude) ─────────────────────────────────────────

class AnthropicAgent(BaseAgent):
    """Tool-use loop for Anthropic's Messages API.

    Claude uses tool_use content blocks (not OpenAI function calling).
    Each round appends assistant + user messages with tool results.
    Searches execute sequentially — Claude rarely requests multiple
    tools in one round.
    """

    def __init__(self, client, model: str, system_prompt: str = None):
        super().__init__(model, system_prompt)
        self.client = client

    def stream(self, question: str) -> Generator[dict, None, None]:
        stats = _empty_stats(self.model)
        messages = [{"role": "user", "content": question}]
        t0 = time.perf_counter()
        baseline_input = 0
        search_num = 0
        response = None

        for round_num in range(MAX_TOOL_ROUNDS):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    system=self.system_prompt,
                    max_tokens=MAX_TOKENS,
                    tools=[TOOL_SCHEMA_ANTHROPIC],
                    messages=messages,
                )
            except Exception as e:
                logger.error("API call failed (model=%s, round=%d): %s", self.model, round_num, e)
                stats["answer"] = f"Error: {e}"
                stats["latency_ms"] = (time.perf_counter() - t0) * 1000
                yield {"event": "answer", "text": stats["answer"], "sources": []}
                yield {"event": "done", "stats": stats}
                return

            call_input = response.usage.input_tokens
            call_output = response.usage.output_tokens
            stats["token_breakdown"]["input"] += call_input
            stats["token_breakdown"]["output"] += call_output
            stats["api_calls"] += 1
            if round_num == 0:
                baseline_input = call_input

            if response.stop_reason != "tool_use":
                break

            tool_results = []
            for block in response.content:
                if block.type != "tool_use" or block.name != "web_search":
                    if block.type == "tool_use":
                        logger.warning("LLM called unknown tool: %s", block.name)
                    continue

                search_num += 1
                yield {
                    "event": "tool_call",
                    "round": round_num + 1,
                    "search_num": search_num,
                    "query": block.input.get("query", ""),
                    "params": {k: v for k, v in block.input.items() if k != "query"},
                }

                search_t0 = time.perf_counter()
                result = execute_search(block.input)
                elapsed_ms = (time.perf_counter() - search_t0) * 1000

                entry = build_tool_log_entry(block.input, result, elapsed_ms)
                stats["tool_calls"].append(entry)
                sources = extract_urls(result)
                stats["sources"].extend(sources)
                stats["search_calls"] += 1

                if is_verbose():
                    print(format_tool_log(entry))

                yield {
                    "event": "search_result",
                    "round": round_num + 1,
                    "search_num": search_num,
                    "result_count": entry["result_count"],
                    "latency_ms": elapsed_ms,
                    "sources": sources,
                    "search_uuid": entry.get("search_uuid", ""),
                }

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })

            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})
        else:
            logger.warning("Tool loop hit MAX_TOOL_ROUNDS=%d (model=%s)", MAX_TOOL_ROUNDS, self.model)

        for block in (response.content if response is not None else []):
            if hasattr(block, "text"):
                stats["answer"] += block.text

        total_in = stats["token_breakdown"]["input"]
        total_out = stats["token_breakdown"]["output"]
        stats["tokens_used"] = total_in + total_out
        stats["token_breakdown"]["search_context"] = max(0, total_in - baseline_input)
        stats["latency_ms"] = (time.perf_counter() - t0) * 1000

        yield {"event": "answer", "text": stats["answer"], "sources": stats["sources"]}
        yield {"event": "done", "stats": stats}


# ─── OpenAI chat.completions (Kimi, Llama, and GPT/Qwen fallback) ─────────────

class OpenAICompatibleAgent(BaseAgent):
    """Tool-use loop using OpenAI's chat.completions API.

    Used by:
      - Kimi (Moonshot): /v1/responses returns 404, confirmed 2026-06-30
      - Llama (Together AI): no native search tool
      - GPT and Qwen: when force_chat_completions=True is passed to the
        agent factory, for A/B comparison against the Responses API path

    Rebuilds full message history every round — input tokens accumulate as
    prior tool results are re-sent. Tool calls within a round execute
    sequentially. See decisions/006 for the token impact analysis.

    Args:
        max_tokens_param: "max_tokens" for most providers; "max_completion_tokens"
                          for GPT-5.x which deprecated the older param name.
        extra_body:       Provider-specific kwargs passed through to create() —
                          e.g. {"enable_thinking": False} for Qwen.
    """

    def __init__(
        self,
        client,
        model: str,
        system_prompt: str = None,
        max_tokens_param: str = "max_tokens",
        extra_body: dict | None = None,
    ):
        super().__init__(model, system_prompt)
        self.client = client
        self.max_tokens_param = max_tokens_param
        self.extra_body = extra_body

    def stream(self, question: str) -> Generator[dict, None, None]:
        stats = _empty_stats(self.model)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question},
        ]
        t0 = time.perf_counter()
        baseline_input = 0
        search_num = 0

        for round_num in range(MAX_TOOL_ROUNDS):
            try:
                create_kwargs = {
                    "model": self.model,
                    "messages": messages,
                    "tools": [TOOL_SCHEMA],
                    self.max_tokens_param: MAX_TOKENS,
                }
                if self.extra_body:
                    create_kwargs["extra_body"] = self.extra_body
                response = self.client.chat.completions.create(**create_kwargs)
            except Exception as e:
                logger.error("API call failed (model=%s, round=%d): %s", self.model, round_num, e)
                stats["answer"] = f"Error: {e}"
                stats["latency_ms"] = (time.perf_counter() - t0) * 1000
                yield {"event": "answer", "text": stats["answer"], "sources": []}
                yield {"event": "done", "stats": stats}
                return

            if response.usage:
                call_input = getattr(response.usage, "prompt_tokens", 0)
                call_output = getattr(response.usage, "completion_tokens", 0)
                stats["token_breakdown"]["input"] += call_input
                stats["token_breakdown"]["output"] += call_output
                if round_num == 0:
                    baseline_input = call_input
            stats["api_calls"] += 1

            if not response.choices:
                logger.error("API returned empty choices (model=%s, round=%d)", self.model, round_num)
                break
            choice = response.choices[0]
            if choice.finish_reason != "tool_calls" or not choice.message.tool_calls:
                break

            messages.append(choice.message)

            for tool_call in choice.message.tool_calls:
                if tool_call.function.name != "web_search":
                    logger.warning("LLM called unknown tool: %s", tool_call.function.name)
                    continue

                try:
                    args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    logger.error("Malformed tool arguments (model=%s): %s", self.model, tool_call.function.arguments[:200])
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": "Error: could not parse tool arguments",
                    })
                    continue
                search_num += 1
                yield {
                    "event": "tool_call",
                    "round": round_num + 1,
                    "search_num": search_num,
                    "query": args.get("query", ""),
                    "params": {k: v for k, v in args.items() if k != "query"},
                }

                search_t0 = time.perf_counter()
                result = execute_search(args)
                elapsed_ms = (time.perf_counter() - search_t0) * 1000

                entry = build_tool_log_entry(args, result, elapsed_ms)
                stats["tool_calls"].append(entry)
                sources = extract_urls(result)
                stats["sources"].extend(sources)
                uuid = extract_search_uuid(result)
                if uuid:
                    stats["search_uuid"] = uuid
                stats["search_calls"] += 1

                if is_verbose():
                    print(format_tool_log(entry))

                yield {
                    "event": "search_result",
                    "round": round_num + 1,
                    "search_num": search_num,
                    "result_count": entry["result_count"],
                    "latency_ms": elapsed_ms,
                    "sources": sources,
                    "search_uuid": uuid or "",
                }

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })
        else:
            logger.warning("Tool loop hit MAX_TOOL_ROUNDS=%d (model=%s)", MAX_TOOL_ROUNDS, self.model)

        stats["answer"] = response.choices[0].message.content or ""
        total_in = stats["token_breakdown"]["input"]
        total_out = stats["token_breakdown"]["output"]
        stats["tokens_used"] = total_in + total_out
        # Accumulation: prior tool results are re-sent each round as input
        stats["token_breakdown"]["search_context"] = (
            max(0, total_in - baseline_input * stats["api_calls"])
            if stats["api_calls"] > 1 else 0
        )
        stats["latency_ms"] = (time.perf_counter() - t0) * 1000

        yield {"event": "answer", "text": stats["answer"], "sources": stats["sources"]}
        yield {"event": "done", "stats": stats}


# ─── OpenAI Responses API (GPT, Qwen) ─────────────────────────────────────────

class OpenAIResponsesAgent(BaseAgent):
    """Tool-use loop using OpenAI's Responses API with stateful chaining.

    Used by:
      - GPT (OpenAI): default path
      - Qwen (DashScope): default path; tool named 'you_search' via
        TOOL_SCHEMA_RESPONSES — DashScope intercepts any function named
        'web_search' and routes it to native search instead of our function
        (confirmed empirically 2026-06-30, see decisions/006)

    Key differences from OpenAICompatibleAgent (chat.completions):
      - previous_response_id: prior context retained server-side — only new
        tool outputs sent each round, not the full message history
      - Parallel tool execution: multiple searches in one round run
        concurrently via ThreadPoolExecutor
      - Token counts reflect only new inputs per round (no accumulation)

    You.com still requires one round-trip per search — custom function tools
    cannot be executed server-side. This is why You.com always uses more
    tokens than native search regardless of which API is used.
    See decisions/006 for full token impact analysis.
    """

    def __init__(self, client, model: str, system_prompt: str = None):
        super().__init__(model, system_prompt)
        self.client = client

    def stream(self, question: str) -> Generator[dict, None, None]:
        stats = _empty_stats(self.model)
        t0 = time.perf_counter()
        previous_response_id = None
        current_input = question
        response = None
        search_num = 0

        for round_num in range(MAX_TOOL_ROUNDS):
            try:
                response = self.client.responses.create(
                    model=self.model,
                    instructions=self.system_prompt,
                    input=current_input,
                    tools=[TOOL_SCHEMA_RESPONSES],
                    previous_response_id=previous_response_id,
                )
            except Exception as e:
                logger.error("API call failed (model=%s, round=%d): %s", self.model, round_num, e)
                stats["answer"] = f"Error: {e}"
                stats["latency_ms"] = (time.perf_counter() - t0) * 1000
                yield {"event": "answer", "text": stats["answer"], "sources": []}
                yield {"event": "done", "stats": stats}
                return

            if response.usage:
                stats["token_breakdown"]["input"] += getattr(response.usage, "input_tokens", 0)
                stats["token_breakdown"]["output"] += getattr(response.usage, "output_tokens", 0)
            stats["api_calls"] += 1
            previous_response_id = response.id

            function_calls = [
                item for item in response.output
                if getattr(item, "type", None) == "function_call"
                and getattr(item, "name", None) == "you_search"
            ]

            if not function_calls:
                break

            # Assign search numbers before parallel execution (main thread, serial).
            # Parse arguments once here — reused in both the yield and _execute.
            indexed_calls = []
            for item in function_calls:
                try:
                    item_args = json.loads(item.arguments)
                except json.JSONDecodeError:
                    logger.error("Malformed tool arguments (model=%s): %s", self.model, item.arguments[:200])
                    continue
                search_num += 1
                indexed_calls.append((search_num, item, item_args))
                yield {
                    "event": "tool_call",
                    "round": round_num + 1,
                    "search_num": search_num,
                    "query": item_args.get("query", ""),
                    "params": {k: v for k, v in item_args.items() if k != "query"},
                }

            # Execute all searches in this round in parallel
            def _execute(idx_item):
                idx, item, args = idx_item  # args already parsed above
                search_t0 = time.perf_counter()
                result = execute_search(args)
                elapsed_ms = (time.perf_counter() - search_t0) * 1000
                return {
                    "search_num": idx,
                    "call_id": item.call_id,
                    "entry": build_tool_log_entry(args, result, elapsed_ms),
                    "sources": extract_urls(result),
                    "uuid": extract_search_uuid(result),
                    "elapsed_ms": elapsed_ms,
                    "output": result,
                }

            with ThreadPoolExecutor(max_workers=min(len(indexed_calls), 8)) as executor:
                results = list(executor.map(_execute, indexed_calls))

            tool_outputs = []
            for r in results:
                stats["tool_calls"].append(r["entry"])
                stats["sources"].extend(r["sources"])
                if r["uuid"]:
                    stats["search_uuid"] = r["uuid"]
                stats["search_calls"] += 1

                if is_verbose():
                    print(format_tool_log(r["entry"]))

                yield {
                    "event": "search_result",
                    "round": round_num + 1,
                    "search_num": r["search_num"],
                    "result_count": r["entry"]["result_count"],
                    "latency_ms": r["elapsed_ms"],
                    "sources": r["sources"],
                    "search_uuid": r["uuid"] or "",
                }

                tool_outputs.append({
                    "type": "function_call_output",
                    "call_id": r["call_id"],
                    "output": r["output"],
                })

            current_input = tool_outputs
        else:
            logger.warning("Tool loop hit MAX_TOOL_ROUNDS=%d (model=%s)", MAX_TOOL_ROUNDS, self.model)

        # Extract answer from Responses API output structure
        message_item = next(
            (item for item in response.output if getattr(item, "type", None) == "message"),
            None,
        ) if response else None

        if message_item:
            text_item = next(
                (c for c in getattr(message_item, "content", [])
                 if getattr(c, "type", None) == "output_text"),
                None,
            )
            stats["answer"] = text_item.text if text_item else ""

        total_in = stats["token_breakdown"]["input"]
        total_out = stats["token_breakdown"]["output"]
        stats["tokens_used"] = total_in + total_out
        # previous_response_id chaining: tokens reflect only new inputs per round,
        # no accumulation — search_context is not applicable here
        stats["token_breakdown"]["search_context"] = 0
        stats["latency_ms"] = (time.perf_counter() - t0) * 1000

        yield {"event": "answer", "text": stats["answer"], "sources": stats["sources"]}
        yield {"event": "done", "stats": stats}
