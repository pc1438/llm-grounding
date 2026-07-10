"""
claude_agent.py — Claude agent with You.com web search tool.

The agent decides when to search. You.com handles the retrieval.

    from agents.claude_agent import ClaudeAgent

    agent = ClaudeAgent()
    result = agent.ask("What happened in tech news this week?")

Requirements:
    pip install anthropic
    export ANTHROPIC_API_KEY="sk-ant-..."
    export YDC_API_KEY="..."

Responses API note:
    Claude uses Anthropic's Messages API, not the OpenAI Responses API.
    force_chat_completions is accepted as a parameter for API consistency but
    has no effect — this agent always uses AnthropicAgent's loop.
"""

import os
import time

from anthropic import Anthropic

from base_agent import AnthropicAgent, _empty_stats, _get_agent_default_model
from search_tool import get_system_prompt, MAX_TOKENS, NATIVE_SEARCH_BASELINE_TOKENS, inject_citations_at_positions, make_progress_reporter, make_elapsed_timer

# Default model comes from pricing.json (agent_default: true for provider "anthropic").
# To change the default, update pricing.json — do not edit this line directly.
# Available models: "claude-opus-4-8" (most capable), "claude-sonnet-4-6" (balanced),
#                   "claude-haiku-4-5-20251001" (fastest)
DEFAULT_MODEL = _get_agent_default_model("anthropic", fallback="claude-sonnet-4-6")


class ClaudeAgent(AnthropicAgent):
    """Claude-powered agent with You.com web search.

    Claude uses Anthropic's tool_use format (not OpenAI function calling),
    which is why this inherits from AnthropicAgent instead of
    OpenAICompatibleAgent.

    Return value of ask() and run_native_search(): see base_agent._empty_stats()
    for the full field-by-field reference.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        force_chat_completions: bool = False,  # accepted for API consistency, no effect
    ):
        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not resolved_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not set. "
                "Pass api_key= or export ANTHROPIC_API_KEY."
            )
        client = Anthropic(api_key=resolved_key)
        super().__init__(client=client, model=model)


    def run_native_search(
        self,
        question: str,
        system_prompt: str = None,
        on_progress=None,
        native_search_tool: dict = None,
    ) -> dict:
        """Run a query using Claude's built-in web_search tool (no You.com).

        Uses streaming so the connection stays alive during long internal searches.
        Returns the canonical stats dict (same shape as ask()).

        Args:
            native_search_tool: The tool config dict from pricing.json
                                 (e.g. {"type": "web_search_20260209", "name": "web_search"}).
                                 Defaults to the current production tool type.
        """
        tool = native_search_tool or {"type": "web_search_20260209", "name": "web_search"}
        prompt = system_prompt or get_system_prompt()
        stats = _empty_stats(self.model)
        stats["interface"] = "native_messages"
        _notify = make_progress_reporter("Native", on_progress)
        messages = [{"role": "user", "content": question}]
        t0, _elapsed = make_elapsed_timer()

        _notify(f"Starting stream... ({_elapsed()})")

        source_index_map: dict[str, int] = {}

        t_connect = time.perf_counter()
        with self.client.messages.stream(
            model=self.model,
            system=prompt,
            max_tokens=MAX_TOKENS,
            tools=[tool],
            messages=messages,
        ) as stream:
            _connect_recorded = False
            for event in stream:
                if not _connect_recorded:
                    stats["connect_ms"] = round((time.perf_counter() - t_connect) * 1000)
                    _connect_recorded = True
                etype = getattr(event, "type", "")
                if etype == "content_block_start":
                    block = getattr(event, "content_block", None)
                    if block:
                        btype = getattr(block, "type", "")
                        if btype == "web_search_tool_use":
                            query = getattr(block, "query", "") or ""
                            _notify(f"Searching: \"{query}\" ({_elapsed()})")
                        elif btype == "web_search_tool_result":
                            _notify(f"Search results received ({_elapsed()})")
                        elif btype == "text":
                            _notify(f"Generating answer... ({_elapsed()})")

            _notify(f"Stream consumed, assembling response... ({_elapsed()})")
            response = stream.get_final_message()

        stats["model_confirmed"] = getattr(response, "model", None)
        stats["token_breakdown"]["input"] = response.usage.input_tokens
        stats["token_breakdown"]["output"] = response.usage.output_tokens
        stats["tokens_used"] = response.usage.input_tokens + response.usage.output_tokens
        stats["api_calls"] = 1
        stats["latency_ms"] = (time.perf_counter() - t0) * 1000

        for block in response.content:
            if block.type == "web_search_tool_result":
                stats["search_calls"] += 1
                if hasattr(block, "content") and isinstance(block.content, list):
                    for result in block.content:
                        url = getattr(result, "url", None)
                        if url and url not in source_index_map:
                            source_index_map[url] = len(source_index_map) + 1
                            stats["sources"].append(url)
                        content_text = getattr(result, "page_content", None) or getattr(result, "text", None)
                        if content_text:
                            stats["token_breakdown"]["search_context"] += len(content_text) // 4

            elif block.type == "text" and getattr(block, "text", None):
                text = block.text
                citations = getattr(block, "citations", None) or []
                if citations:
                    insertions: dict[int, list[int]] = {}
                    for cit in citations:
                        url = getattr(cit, "url", None) or getattr(cit, "document_url", None)
                        if not url:
                            continue
                        if url not in source_index_map:
                            source_index_map[url] = len(source_index_map) + 1
                            stats["sources"].append(url)
                        n = source_index_map[url]
                        end = getattr(cit, "end_char_index", None)
                        if end is not None:
                            insertions.setdefault(end, [])
                            if n not in insertions[end]:
                                insertions[end].append(n)
                    text = inject_citations_at_positions(text, insertions)
                stats["answer"] += text

        if stats["token_breakdown"]["search_context"] == 0 and stats["token_breakdown"]["input"] > NATIVE_SEARCH_BASELINE_TOKENS:
            stats["token_breakdown"]["search_context"] = stats["token_breakdown"]["input"] - NATIVE_SEARCH_BASELINE_TOKENS

        _notify(f"Complete — {stats['latency_ms']:.0f}ms total ({_elapsed()})")
        return stats


if __name__ == "__main__":
    import sys

    question = " ".join(sys.argv[1:]) or input("Ask something: ")
    agent = ClaudeAgent()
    result = agent.ask(question)
    print(result["answer"])
    if result["sources"]:
        print(f"\nSources: {result['sources']}")
