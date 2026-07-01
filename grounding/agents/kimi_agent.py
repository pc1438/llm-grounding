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

from openai import OpenAI

from base_agent import OpenAICompatibleAgent

# Pinned model versions:
#   "kimi-k2.5"              — latest, 256k context (recommended)
#   "kimi-k2-0905-preview"   — previous gen, 256k context
# NOTE: moonshot-v1-* models were discontinued Jan 2026
DEFAULT_MODEL = "kimi-k2.5"

# Moonshot API endpoint options:
#   api.moonshot.ai/v1   — global endpoint (recommended; used here and in run.py/compare.py)
#   api.moonshot.cn/v1   — Asia-Pacific endpoint, lower latency from CN/AP region
# Both reach the same models. We standardize on .ai/v1 so all configs point to
# the same endpoint by default; deploy with .cn/v1 only if latency is a concern.


class KimiAgent(OpenAICompatibleAgent):
    """Kimi-powered agent with You.com web search.

    Uses OpenAI-compatible function calling via Moonshot's API.
    Always uses chat.completions — Moonshot's /v1/responses returns 404.
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
        )
        super().__init__(client=client, model=model)


if __name__ == "__main__":
    import sys

    question = " ".join(sys.argv[1:]) or input("Ask something: ")
    agent = KimiAgent()
    result = agent.ask(question)
    print(result["answer"])
    if result["sources"]:
        print(f"\nSources: {result['sources']}")
