"""
openrouter_agent.py — Generic OpenRouter agent with You.com web search and Exa native search.

OpenRouter is a unified proxy that provides OpenAI-compatible access to many
model providers under a single API key. This agent supports two search paths:

  YDC path:  Any OpenRouter model can use You.com Search via tool-use loop.

  Native Exa path (run_native_search):  OpenRouter's :online suffix appended to
    any model slug routes the request through Exa web search. This is a single
    API call — no tool-use loop. Sources come back as url_citation annotations
    on choice.message. Answer embeds markdown links inline; no [N] markers.
    Cost: $0.005/search (same as YDC).

    Note: For qwen-openrouter and kimi-openrouter, the Exa path is used as
    the native comparison path since the direct provider native search
    (DashScope web_search, Kimi $web_search) is not accessible via OpenRouter.

    from agents.openrouter_agent import OpenRouterAgent

    # Any OpenRouter model — YDC path:
    agent = OpenRouterAgent(model="meta-llama/llama-4-maverick")
    result = agent.ask("What happened in tech news this week?")

    # Exa native path (single call, no tool loop):
    result = agent.run_native_search("Who won the 2026 Olympic hockey gold?")

Requirements:
    pip install openai
    export OPENROUTER_API_KEY="sk-or-..."   (from openrouter.ai/keys)
    export YDC_API_KEY="..."  (for YDC path only)

OpenRouter model IDs:
    Models are referenced by provider/model-slug. Verify current slugs at
    openrouter.ai/models. Common ones used here:
        meta-llama/llama-4-maverick  — Meta Llama 4 Maverick
        meta-llama/llama-4-scout     — Meta Llama 4 Scout
        deepseek/deepseek-chat-v3.1  — DeepSeek V3.1
        google/gemma-3-27b-it        — Google Gemma 3 27B
        qwen/qwen3.7-max             — Alibaba Qwen3.7 Max (via OpenRouter)
        moonshotai/kimi-k2.5         — Moonshot AI Kimi K2.5 (via OpenRouter)
    OpenRouter normalises these to chat.completions — no Responses API support.
"""

import os

from openai import OpenAI

from base_agent import OpenAICompatibleAgent, _empty_stats

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterAgent(OpenAICompatibleAgent):
    """Generic OpenRouter agent with You.com web search (YDC path only).

    Accepts any OpenRouter model slug. Native search is not supported —
    use the provider's direct agent class for that.

    Return value of ask(): see base_agent._empty_stats() for field reference.
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
    ):
        if not model:
            raise ValueError("model is required — pass an OpenRouter model slug, e.g. 'qwen/qwen3.7-max'")
        resolved_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not resolved_key:
            raise ValueError(
                "OPENROUTER_API_KEY not set. "
                "Pass api_key= or export OPENROUTER_API_KEY."
            )
        client = OpenAI(
            api_key=resolved_key,
            base_url=_OPENROUTER_BASE_URL,
            timeout=120.0,
        )
        super().__init__(client=client, model=model)

    def run_native_search(
        self,
        question: str,
        system_prompt: str = None,
        on_progress=None,
        **kwargs,
    ) -> dict:
        """Run native web search via OpenRouter's :online suffix (Exa engine).

        Appends ':online' to the model slug, which routes the request through
        Exa web search. Single API call — no tool-use loop.

        Sources are returned as url_citation annotations on choice.message.
        The answer embeds markdown links inline rather than [N] markers.
        Cost: $0.005/search (openrouter.ai/docs/features/web-search).
        """
        import time
        stats = _empty_stats(self.model)
        stats["interface"] = "native_exa"

        online_model = self.model + ":online"
        t0 = time.perf_counter()
        try:
            t_connect = time.perf_counter()
            response = self.client.chat.completions.create(
                model=online_model,
                messages=[{"role": "user", "content": question}],
                max_tokens=4096,
            )
            stats["connect_ms"] = round((time.perf_counter() - t_connect) * 1000)
        except Exception as e:
            stats["answer"] = f"Error: {e}"
            stats["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            return stats

        stats["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        stats["api_calls"] = 1
        stats["search_calls"] = 1

        choice = response.choices[0]
        stats["answer"] = choice.message.content or ""
        stats["model_confirmed"] = response.model

        if response.usage:
            stats["tokens_used"] = response.usage.total_tokens
            stats["token_breakdown"]["input"] = response.usage.prompt_tokens
            stats["token_breakdown"]["output"] = response.usage.completion_tokens

        annotations = getattr(choice.message, "annotations", None) or []
        sources = []
        for ann in annotations:
            if getattr(ann, "type", None) == "url_citation":
                uc = ann.url_citation
                url = getattr(uc, "url", None)
                if url and url not in sources:
                    sources.append(url)
        stats["sources"] = sources

        return stats


if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    load_dotenv("env.txt")

    args = sys.argv[1:]
    if not args:
        print("Usage: python3 openrouter_agent.py <model-slug> <question>")
        print("  e.g. python3 openrouter_agent.py qwen/qwen3.7-max 'Latest NVIDIA earnings?'")
        sys.exit(1)
    model_slug = args[0]
    question = " ".join(args[1:]) or input("Ask something: ")
    agent = OpenRouterAgent(model=model_slug)
    result = agent.ask(question)
    print(result["answer"])
    if result["sources"]:
        print(f"\nSources: {result['sources']}")
