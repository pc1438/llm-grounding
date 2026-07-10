"""
kimi_agent.py — Kimi K2.5 (Moonshot AI) agent with You.com web search tool.

Kimi's API has a $web_search tool, but it's incompatible with K2.5's
thinking mode and unavailable on OpenRouter/Together. You.com gives
consistent grounding regardless of provider or mode.

    from agents.kimi_agent import KimiAgent

    agent = KimiAgent()
    result = agent.ask("What did the Fed announce about rates this week?")

Requirements:
    pip install openai
    export MOONSHOT_API_KEY="sk-..."  (from platform.moonshot.ai)
    export YDC_API_KEY="..."

Responses API note:
    Kimi does not support the OpenAI Responses API (/v1/responses returns 404).
    Confirmed empirically + docs review 2026-06-30. See decisions/006.
    force_chat_completions is accepted as a parameter for API consistency but
    has no effect — this agent always uses chat.completions.
"""

import os
import re
import time

from openai import OpenAI

from base_agent import OpenAICompatibleAgent, _empty_stats, _get_agent_default_model
from search_tool import get_system_prompt, MAX_TOKENS, make_progress_reporter, make_elapsed_timer

# Default model comes from pricing.json (agent_default: true for provider "kimi").
# To change the default, update pricing.json — do not edit this line directly.
# Available models: "kimi-k2.5", "kimi-k2-0905-preview" (prev gen, discontinued)
DEFAULT_MODEL = _get_agent_default_model("kimi", fallback="kimi-k2.6")

# Moonshot API endpoint options:
#   api.moonshot.ai/v1   — global endpoint (recommended; used here and in run.py/compare.py)
#   api.moonshot.cn/v1   — Asia-Pacific endpoint, lower latency from CN/AP region
# Both reach the same models. We standardize on .ai/v1 so all configs point to
# the same endpoint by default; deploy with .cn/v1 only if latency is a concern.


class KimiAgent(OpenAICompatibleAgent):
    """Kimi-powered agent with You.com web search.

    Uses OpenAI-compatible function calling via Moonshot's API.
    Always uses chat.completions — Moonshot's /v1/responses returns 404.

    Return value of ask() and run_native_search(): see base_agent._empty_stats()
    for the full field-by-field reference.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        force_chat_completions: bool = False,  # accepted for API consistency, no effect
    ):
        resolved_key = api_key or os.environ.get("MOONSHOT_API_KEY", "")
        if not resolved_key:
            raise ValueError(
                "MOONSHOT_API_KEY not set. "
                "Pass api_key= or export MOONSHOT_API_KEY."
            )
        client = OpenAI(
            api_key=resolved_key,
            base_url="https://api.moonshot.ai/v1",
            timeout=120.0,
        )
        super().__init__(client=client, model=model)


    def run_native_search(
        self,
        question: str,
        system_prompt: str = None,
        on_progress=None,
    ) -> dict:
        """Run a query using Kimi's built-in $web_search tool (no You.com).

        Kimi's built-in search uses a manual tool-use loop — the API executes
        the search internally but requires the client to echo arguments back as
        the tool result to complete the agentic loop. Thinking must be disabled.
        URLs are extracted from the final answer text (not returned via API).
        """
        prompt = system_prompt or get_system_prompt()
        stats = _empty_stats(self.model)
        stats["interface"] = "native_chat"
        _notify = make_progress_reporter("Native", on_progress)
        kimi_tool = [{"type": "builtin_function", "function": {"name": "$web_search"}}]
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": question},
        ]
        t0, _elapsed = make_elapsed_timer()
        baseline_input = 0
        choice = None

        _notify(f"Starting request... ({_elapsed()})")

        try:
            from search_tool import MAX_TOOL_ROUNDS
            for round_num in range(MAX_TOOL_ROUNDS):
                _notify(f"LLM round {round_num + 1}... ({_elapsed()})")

                t_connect = time.perf_counter()
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=kimi_tool,
                    max_completion_tokens=MAX_TOKENS,
                    extra_body={"thinking": {"type": "disabled"}},
                )

                if response.usage:
                    call_input = getattr(response.usage, "prompt_tokens", 0)
                    call_output = getattr(response.usage, "completion_tokens", 0)
                else:
                    call_input = call_output = 0

                stats["token_breakdown"]["input"] += call_input
                stats["token_breakdown"]["output"] += call_output
                stats["api_calls"] += 1

                if round_num == 0:
                    stats["connect_ms"] = round((time.perf_counter() - t_connect) * 1000)
                    baseline_input = call_input
                    stats["model_confirmed"] = getattr(response, "model", None)

                choice = response.choices[0] if response.choices else None

                if not choice or choice.finish_reason != "tool_calls" or not (choice.message and choice.message.tool_calls):
                    break

                messages.append(choice.message)

                for tool_call in choice.message.tool_calls:
                    if tool_call.function.name != "$web_search":
                        continue
                    stats["search_calls"] += 1
                    raw_args = tool_call.function.arguments
                    _notify(f"Search {stats['search_calls']}: round {round_num + 1} ({_elapsed()})")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": raw_args,
                    })

            _notify(f"Generating answer... ({_elapsed()})")

            stats["latency_ms"] = (time.perf_counter() - t0) * 1000
            stats["tokens_used"] = stats["token_breakdown"]["input"] + stats["token_breakdown"]["output"]

            if stats["api_calls"] > 1:
                stats["token_breakdown"]["search_context"] = max(
                    0, stats["token_breakdown"]["input"] - baseline_input * stats["api_calls"]
                )

            stats["answer"] = choice.message.content if choice and choice.message else ""

            # Kimi does not expose source URLs via the API — extract from answer text.
            seen = set()
            for url in re.findall(r'https?://[^\s\)\]"\'<>]+', stats["answer"]):
                url = url.rstrip('.,;:')
                if url not in seen:
                    seen.add(url)
                    stats["sources"].append(url)

            _notify(f"Complete — {stats['latency_ms']:.0f}ms total ({_elapsed()})")

        except Exception as e:
            stats["answer"] = f"Error: {e}"
            stats["latency_ms"] = (time.perf_counter() - t0) * 1000

        return stats


if __name__ == "__main__":
    import sys

    question = " ".join(sys.argv[1:]) or input("Ask something: ")
    agent = KimiAgent()
    result = agent.ask(question)
    print(result["answer"])
    if result["sources"]:
        print(f"\nSources: {result['sources']}")
