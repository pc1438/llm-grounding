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

import os
import time

from openai import OpenAI

from base_agent import BaseAgent, OpenAICompatibleAgent, OpenAIResponsesAgent, _empty_stats
from search_tool import get_system_prompt, MAX_TOKENS, inject_citations_at_positions, is_verbose

# Pinned model versions:
#   "gpt-5.4"        — latest flagship (recommended)
#   "gpt-5.4-mini"   — balanced cost/performance
#   "gpt-5.4-nano"   — fastest, lowest cost
#   "gpt-4.1"        — previous gen, still in API
DEFAULT_MODEL = "gpt-5.4"


class OpenAIAgent(BaseAgent):
    """OpenAI-powered agent with You.com web search.

    Selects Responses API or chat.completions at construction time via
    force_chat_completions. Both YDC paths and native search paths are
    available as methods on this single class.

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
        self._client = OpenAI(api_key=resolved_key)
        cls = OpenAICompatibleAgent if force_chat_completions else OpenAIResponsesAgent
        self._impl = cls(client=self._client, model=model)
        super().__init__(model=model)

    def stream(self, question: str):
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
        if path == "chat":
            return self._run_native_chat(
                question, system_prompt, on_progress, chat_native_search_tool
            )
        return self._run_native_responses(
            question, system_prompt, on_progress, native_search_tool
        )

    def _run_native_responses(
        self,
        question: str,
        system_prompt: str = None,
        on_progress=None,
        native_search_tool: dict = None,
    ) -> dict:
        """Responses API path — single call, server-side search loop."""
        tool = native_search_tool or {"type": "web_search", "search_context_size": "medium"}
        prompt = system_prompt or get_system_prompt()
        stats = _empty_stats(self.model)
        stats["interface"] = "native_responses"
        verbose = is_verbose()

        def _notify(msg):
            if verbose:
                print(f"  [Native] {msg}")
            if on_progress:
                on_progress(msg)

        t0 = time.perf_counter()

        def _elapsed():
            return f"{(time.perf_counter() - t0):.1f}s"

        _notify(f"Starting request... ({_elapsed()})")

        response = self._client.responses.create(
            model=self.model,
            instructions=prompt,
            tools=[tool],
            input=question,
        )

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

        if stats["search_calls"] > 0 and stats["token_breakdown"]["input"] > 200:
            stats["token_breakdown"]["search_context"] = stats["token_breakdown"]["input"] - 200

        _notify(f"Complete — {stats['latency_ms']:.0f}ms total ({_elapsed()})")
        return stats

    def _run_native_chat(
        self,
        question: str,
        system_prompt: str = None,
        on_progress=None,
        chat_native_search_tool: dict = None,
    ) -> dict:
        """chat.completions path — web_search_preview tool, single call."""
        tool = chat_native_search_tool or {"type": "web_search_preview"}
        prompt = system_prompt or get_system_prompt()
        stats = _empty_stats(self.model)
        stats["interface"] = "native_chat"
        verbose = is_verbose()

        def _notify(msg):
            if verbose:
                print(f"  [Native/Chat] {msg}")
            if on_progress:
                on_progress(msg)

        t0 = time.perf_counter()

        def _elapsed():
            return f"{(time.perf_counter() - t0):.1f}s"

        _notify(f"Starting request... ({_elapsed()})")

        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": question},
            ],
            tools=[tool],
        )

        stats["latency_ms"] = (time.perf_counter() - t0) * 1000
        stats["api_calls"] = 1

        if response.model:
            stats["model_confirmed"] = response.model

        if response.usage:
            stats["token_breakdown"]["input"] = getattr(response.usage, "prompt_tokens", 0)
            stats["token_breakdown"]["output"] = getattr(response.usage, "completion_tokens", 0)
        stats["tokens_used"] = stats["token_breakdown"]["input"] + stats["token_breakdown"]["output"]

        if response.choices:
            choice = response.choices[0]
            stats["answer"] = (choice.message.content or "")

            # Count web search tool calls reported in the response
            for tc in getattr(choice.message, "tool_calls", []) or []:
                if getattr(tc.function, "name", "") in ("web_search", "web_search_preview"):
                    stats["search_calls"] += 1
                    _notify(f"Search {stats['search_calls']} ({_elapsed()})")

        if stats["token_breakdown"]["input"] > 200:
            stats["token_breakdown"]["search_context"] = stats["token_breakdown"]["input"] - 200

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
