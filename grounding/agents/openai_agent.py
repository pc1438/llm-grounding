"""
openai_agent.py — OpenAI agent with You.com web search tool.

Default path: OpenAI Responses API (previous_response_id chaining, parallel
tool calls). Use force_chat_completions=True for A/B comparison against the
older chat.completions path.

    from agents.openai_agent import OpenAIAgent

    agent = OpenAIAgent()
    result = agent.ask("Compare Tesla and BYD delivery numbers this quarter")

    # Native search (no You.com):
    result = agent.run_native_search("Same question, OpenAI's built-in search")

    # Native search via chat.completions:
    result = agent.run_native_search("Same question", path="chat")

    # Force chat.completions for YDC path (A/B comparison):
    agent = OpenAIAgent(force_chat_completions=True)

Requirements:
    pip install openai
    export OPENAI_API_KEY="sk-..."
    export YDC_API_KEY="..."

API path selection (YDC):
    Responses API (default):   parallel tool calls, previous_response_id chaining.
    chat.completions fallback: full message history re-sent every round.
    See decisions/006_gpt_ydc_chat_vs_responses_api.md for full comparison.

Native search paths:
    path="responses" (default): client.responses.create() with web_search tool.
    path="chat":                chat.completions with web_search_preview tool.
    Both return the same stats dict shape.
"""

import json
import os
import time

from openai import OpenAI

from base_agent import BaseAgent, OpenAICompatibleAgent, OpenAIResponsesAgent, _empty_stats, _get_agent_default_model
from search_tool import get_system_prompt, MAX_TOKENS, MAX_TOOL_ROUNDS, NATIVE_SEARCH_BASELINE_TOKENS, TOOL_SCHEMA, inject_citations_at_positions, make_progress_reporter, make_elapsed_timer, extract_urls, execute_search

# Default model comes from pricing.json (agent_default: true for provider "openai").
# To change the default, update pricing.json — do not edit this line directly.
# Available models: "gpt-5.4" (flagship), "gpt-5.4-mini" (balanced),
#                   "gpt-5.4-nano" (fastest), "gpt-4.1" (prev gen)
DEFAULT_MODEL = _get_agent_default_model("openai", fallback="gpt-5.4")

# Per-model round caps. None = use the global MAX_TOOL_ROUNDS from search_tool.
_MODEL_MAX_ROUNDS = {
    "gpt-5.5": 6,
}


class OpenAIAgent(BaseAgent):
    """OpenAI-powered agent with You.com web search.

    Selects Responses API or chat.completions at construction time via
    force_chat_completions. Both YDC paths and native search paths are
    available as methods on this single class.

    Return value of ask() and run_native_search(): see base_agent._empty_stats()
    for the full field-by-field reference.

    Subclass this to build your own GPT-based grounded agent:

        class MyGPTAgent(OpenAIAgent):
            def __init__(self):
                super().__init__(model="gpt-5.4-mini")

            def run_native_search(self, question, **kwargs):
                # override to customise system prompt, tool config, etc.
                return super().run_native_search(question, **kwargs)
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        force_chat_completions: bool = False,
    ):
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not resolved_key:
            raise ValueError(
                "OPENAI_API_KEY not set. "
                "Pass api_key= or export OPENAI_API_KEY."
            )
        self._api_key = resolved_key
        self._client = OpenAI(api_key=resolved_key, timeout=120.0)
        cls = OpenAICompatibleAgent if force_chat_completions else OpenAIResponsesAgent
        self._impl = cls(client=self._client, model=model)
        self._max_rounds = _MODEL_MAX_ROUNDS.get(model)  # None = use global default
        super().__init__(model=model)

    def stream(self, question: str, max_rounds: int = None):
        _rounds = max_rounds if max_rounds is not None else self._max_rounds
        if _rounds is not None:
            yield from self._impl.stream(question, max_rounds=_rounds)
        else:
            yield from self._impl.stream(question)

    def run_native_search(
        self,
        question: str,
        system_prompt: str = None,
        on_progress=None,
        native_search_tool: dict = None,
        chat_native_search_tool: dict = None,
        path: str = "responses",
    ) -> dict:
        """Run a query using OpenAI's built-in web search (no You.com).

        Args:
            path: "responses" (default) — Responses API with web_search tool.
                  "chat"                — chat.completions with web_search_preview.
            native_search_tool:      Tool config for the Responses API path.
            chat_native_search_tool: Tool config for the chat.completions path.
        """
        t0 = time.perf_counter()
        try:
            if path == "chat":
                return self._run_native_chat(
                    question, system_prompt, on_progress, chat_native_search_tool
                )
            return self._run_native_responses(
                question, system_prompt, on_progress, native_search_tool
            )
        except Exception as e:
            stats = _empty_stats(self.model)
            stats["interface"] = "native_chat" if path == "chat" else "native_responses"
            stats["answer"] = f"Error: {e}"
            stats["latency_ms"] = (time.perf_counter() - t0) * 1000
            return stats

    def _run_native_responses(
        self,
        question: str,
        system_prompt: str = None,
        on_progress=None,
        native_search_tool: dict = None,
    ) -> dict:
        """Responses API path — single call, server-side search loop."""
        tool = native_search_tool or {"type": "web_search", "search_context_size": "medium"}
        prompt = system_prompt or get_system_prompt(self.model)
        stats = _empty_stats(self.model)
        stats["interface"] = "native_responses"
        _notify = make_progress_reporter("Native", on_progress)
        t0, _elapsed = make_elapsed_timer()

        _notify(f"Starting request... ({_elapsed()})")

        t_connect = time.perf_counter()
        response = self._client.responses.create(
            model=self.model,
            instructions=prompt,
            tools=[tool],
            input=question,
        )
        stats["connect_ms"] = round((time.perf_counter() - t_connect) * 1000)

        stats["model_confirmed"] = getattr(response, "model", None)
        stats["latency_ms"] = (time.perf_counter() - t0) * 1000
        stats["api_calls"] = 1

        if hasattr(response, "usage") and response.usage:
            stats["token_breakdown"]["input"] = getattr(response.usage, "input_tokens", 0)
            stats["token_breakdown"]["output"] = getattr(response.usage, "output_tokens", 0)
        stats["tokens_used"] = stats["token_breakdown"]["input"] + stats["token_breakdown"]["output"]

        source_index_map: dict[str, int] = {}

        for item in response.output:
            item_type = getattr(item, "type", "")

            if item_type == "web_search_call":
                stats["search_calls"] += 1
                search_query = getattr(item, "query", "") or getattr(item, "input", "") or ""
                _notify(f"Search {stats['search_calls']}: \"{search_query}\" ({_elapsed()})")

            elif item_type == "message":
                _notify(f"Generating answer... ({_elapsed()})")
                for content_part in getattr(item, "content", []):
                    if getattr(content_part, "type", "") == "output_text":
                        text = getattr(content_part, "text", "")
                        annotations = getattr(content_part, "annotations", []) or []
                        insertions: dict[int, list[int]] = {}
                        for ann in annotations:
                            if getattr(ann, "type", "") != "url_citation":
                                continue
                            url = getattr(ann, "url", "")
                            if not url:
                                continue
                            if url not in source_index_map:
                                source_index_map[url] = len(source_index_map) + 1
                                stats["sources"].append(url)
                            n = source_index_map[url]
                            end = getattr(ann, "end_index", None)
                            if end is not None:
                                insertions.setdefault(end, [])
                                if n not in insertions[end]:
                                    insertions[end].append(n)
                        text = inject_citations_at_positions(text, insertions)
                        stats["answer"] += text

        if stats["search_calls"] > 0 and stats["token_breakdown"]["input"] > NATIVE_SEARCH_BASELINE_TOKENS:
            stats["token_breakdown"]["search_context"] = stats["token_breakdown"]["input"] - NATIVE_SEARCH_BASELINE_TOKENS

        _notify(f"Complete — {stats['latency_ms']:.0f}ms total ({_elapsed()})")
        return stats

    def _run_native_chat(
        self,
        question: str,
        system_prompt: str = None,
        on_progress=None,
        chat_native_search_tool: dict = None,  # unused, kept for signature compat
    ) -> dict:
        """YDC via chat.completions — client-managed multi-round tool loop.

        Uses {"type": "function"} schema (valid for chat.completions).
        web_search_preview is Responses-API-only and is rejected with 400 here.
        """


        prompt = system_prompt or get_system_prompt(self.model)
        stats = _empty_stats(self.model)
        stats["interface"] = "ydc"
        _notify = make_progress_reporter("YDC/Chat", on_progress)
        t0, _elapsed = make_elapsed_timer()

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": question},
        ]
        baseline_input = 0

        for round_num in range(MAX_TOOL_ROUNDS):
            _notify(f"Round {round_num + 1} ({_elapsed()})")
            t_connect = time.perf_counter()
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=[TOOL_SCHEMA],
                max_completion_tokens=MAX_TOKENS,
            )
            if round_num == 0:
                stats["connect_ms"] = round((time.perf_counter() - t_connect) * 1000)
            stats["api_calls"] += 1

            usage = response.usage
            call_input = getattr(usage, "prompt_tokens", 0) if usage else 0
            call_output = getattr(usage, "completion_tokens", 0) if usage else 0
            stats["tokens_used"] += call_input + call_output
            stats["token_breakdown"]["input"] += call_input
            stats["token_breakdown"]["output"] += call_output
            if round_num == 0:
                baseline_input = call_input

            if round_num == 0 and response.model:
                stats["model_confirmed"] = response.model

            choice = response.choices[0]
            tool_calls = getattr(choice.message, "tool_calls", None) or []

            if not tool_calls or choice.finish_reason == "stop":
                stats["answer"] = choice.message.content or ""
                break

            messages.append({
                "role": "assistant",
                "content": choice.message.content,
                "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in tool_calls
                ],
            })

            for tc in tool_calls:
                if tc.function.name != "web_search":
                    continue
                tool_args = json.loads(tc.function.arguments or "{}")
                if len(tool_args.get("query", "")) > 200:
                    tool_args["query"] = tool_args["query"][:200]
                stats["search_calls"] += 1
                _notify(f"Search {stats['search_calls']} ({_elapsed()})")
                result = execute_search(tool_args)
                for url in extract_urls(result):
                    if url not in stats["sources"]:
                        stats["sources"].append(url)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
        else:
            stats["hit_round_limit"] = True

        stats["latency_ms"] = (time.perf_counter() - t0) * 1000
        if stats["token_breakdown"]["input"] > baseline_input:
            stats["token_breakdown"]["search_context"] = stats["token_breakdown"]["input"] - baseline_input

        _notify(f"Complete — {stats['latency_ms']:.0f}ms total ({_elapsed()})")
        return stats


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    force_cc = "--chat-completions" in args
    args = [a for a in args if a != "--chat-completions"]
    question = " ".join(args) or input("Ask something: ")
    agent = OpenAIAgent(force_chat_completions=force_cc)
    result = agent.ask(question)
    if result is None:
        print("Error: no result returned")
        sys.exit(1)
    answer = result.get("answer") or "(no answer)"
    print(answer)
    if result.get("sources"):
        print(f"\nSources: {result['sources']}")
