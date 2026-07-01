"""
qwen_agent.py — Qwen agent with You.com web search tool.

Qwen via DashScope has built-in web search, but it disappears on Hugging
Face and self-hosted infra. You.com gives consistent grounding regardless
of backend. Supports DashScope (first-party) and Hugging Face.

Default path: OpenAI Responses API (same as GPT). DashScope confirmed to
support /v1/responses as of 2026-06-30. See decisions/006 for test results.

    from agents.qwen_agent import QwenAgent

    agent = QwenAgent()                        # DashScope, Responses API (default)
    agent = QwenAgent(backend="huggingface")   # Hugging Face, chat.completions
    agent = QwenAgent(force_chat_completions=True)  # DashScope, chat.completions

    # Native search (DashScope only — requires international endpoint):
    result = agent.run_native_search("What happened at the 2026 AI Summit?")

Requirements:
    pip install openai
    export YDC_API_KEY="..."
    export DASHSCOPE_API_KEY="sk-..."    # DashScope
    export HF_TOKEN="hf_..."            # Hugging Face

Tool naming (DashScope-specific quirk):
    DashScope intercepts any function named 'web_search' and routes it to
    Qwen's native search instead of calling our custom function. We use
    'you_search' via TOOL_SCHEMA_RESPONSES to avoid this. This only applies
    to the Responses API path; chat.completions uses TOOL_SCHEMA ('web_search')
    which is fine because DashScope's chat.completions function calling does
    not have this interception behavior.
    Confirmed empirically 2026-06-30. See decisions/006.

Native search endpoint note:
    The standard DashScope domestic endpoint (dashscope.aliyuncs.com) is used
    for the YDC path. Native search (run_native_search) requires the international
    endpoint (dashscope-intl.aliyuncs.com) — the domestic and MaaS workspace
    endpoints do not support the Responses API with web_search tool.
    Both endpoints share the same DASHSCOPE_API_KEY.

Subclassing:
    QwenAgent is a proper class — subclass it to build your own Qwen-based agent:

        class MyQwenAgent(QwenAgent):
            def __init__(self):
                super().__init__(model="qwen3-max")

            def run_native_search(self, question, **kwargs):
                return super().run_native_search(question, **kwargs)
"""

import os
import time

from openai import OpenAI

from base_agent import BaseAgent, OpenAICompatibleAgent, OpenAIResponsesAgent, _empty_stats
from search_tool import get_system_prompt, MAX_TOKENS, is_verbose

BACKENDS = {
    "dashscope": {
        "api_key_env": "DASHSCOPE_API_KEY",
        # Domestic DashScope endpoint (lower latency for CN/AP region).
        # Native search uses dashscope-intl.aliyuncs.com (see run_native_search).
        # run.py uses DASHSCOPE_BASE_URL env var for per-deployment configuration.
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        # Default model for direct/programmatic use. run.py pins "qwen3.7-max"
        # in GROUNDING_MODELS for its own display/cost accounting. These are
        # intentionally separate; do NOT auto-sync.
        "default_model": "qwen-plus-2025-09-11",
        # Also: "qwen3-max", "qwen3.5-plus", "qwen3.5-flash"
        "supports_responses_api": True,
        "supports_native_search": True,
    },
    "huggingface": {
        "api_key_env": "HF_TOKEN",
        "base_url": "https://router.huggingface.co/v1",
        "default_model": "Qwen/Qwen3-8B",
        # Also: "Qwen/Qwen3.5-27B", "Qwen/Qwen3.5-4B"
        "supports_responses_api": False,
        "supports_native_search": False,
    },
}

# International endpoint for native search (Responses API with web_search tool).
# The domestic endpoint and MaaS workspace endpoint do not support this path.
_DASHSCOPE_INTL_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"


class QwenAgent(BaseAgent):
    """Qwen-powered agent with You.com web search.

    Selects Responses API or chat.completions at construction time based on
    backend and force_chat_completions. Both YDC and native search paths are
    available as methods on this single class.

    Subclass this to build your own Qwen-based grounded agent:

        class MyQwenAgent(QwenAgent):
            def __init__(self):
                super().__init__(model="qwen3-max")
    """

    def __init__(
        self,
        backend: str = "dashscope",
        model: str | None = None,
        api_key: str | None = None,
        force_chat_completions: bool = False,
    ):
        if backend not in BACKENDS:
            raise ValueError(f"Unknown backend: {backend!r}. Choose from: {list(BACKENDS)}")

        cfg = BACKENDS[backend]
        resolved_key = api_key or os.environ.get(cfg["api_key_env"], "")
        if not resolved_key:
            raise ValueError(
                f"{cfg['api_key_env']} not set. "
                f"Pass api_key= or export {cfg['api_key_env']}."
            )

        self._api_key = resolved_key
        self._backend = backend
        self._backend_cfg = cfg
        client = OpenAI(api_key=resolved_key, base_url=cfg["base_url"])
        resolved_model = model or cfg["default_model"]

        use_responses = cfg["supports_responses_api"] and not force_chat_completions
        cls = OpenAIResponsesAgent if use_responses else OpenAICompatibleAgent
        self._impl = cls(client=client, model=resolved_model)
        super().__init__(model=resolved_model)

    def stream(self, question: str):
        yield from self._impl.stream(question)

    def run_native_search(
        self,
        question: str,
        system_prompt: str = None,
        on_progress=None,
    ) -> dict:
        """Run a query using Qwen's built-in web_search tool (no You.com).

        Uses the DashScope international endpoint (dashscope-intl.aliyuncs.com)
        via the Responses API. Only available for the dashscope backend.
        The domestic DashScope endpoint and HuggingFace do not support this path.

        Source URLs are returned in item.action.sources on the web_search_call
        output item (not as url_citation annotations like OpenAI).
        """
        if not self._backend_cfg.get("supports_native_search"):
            return {
                **_empty_stats(self.model),
                "answer": "Native search not supported for this backend.",
                "interface": "not_supported",
            }

        prompt = system_prompt or get_system_prompt()
        stats = _empty_stats(self.model)
        stats["interface"] = "native_responses"
        verbose = is_verbose()

        def _notify(msg):
            if verbose:
                print(f"  [Native] {msg}")
            if on_progress:
                on_progress(msg)

        # Native search requires the international endpoint — create a separate client.
        native_client = OpenAI(api_key=self._api_key, base_url=_DASHSCOPE_INTL_BASE_URL)

        t0 = time.perf_counter()

        def _elapsed():
            return f"{(time.perf_counter() - t0):.1f}s"

        _notify(f"POST dashscope-intl · model={self.model} · tools=[web_search] ({_elapsed()})")

        response = native_client.responses.create(
            model=self.model,
            instructions=prompt,
            input=question,
            tools=[{"type": "web_search"}],
            extra_body={"enable_thinking": False},
        )

        stats["model_confirmed"] = getattr(response, "model", None)
        stats["latency_ms"] = (time.perf_counter() - t0) * 1000
        stats["api_calls"] = 1

        if hasattr(response, "usage") and response.usage:
            stats["token_breakdown"]["input"] = getattr(response.usage, "input_tokens", 0)
            stats["token_breakdown"]["output"] = getattr(response.usage, "output_tokens", 0)
        stats["tokens_used"] = stats["token_breakdown"]["input"] + stats["token_breakdown"]["output"]

        for item in response.output:
            item_type = getattr(item, "type", "")

            if item_type == "web_search_call":
                stats["search_calls"] += 1
                action = getattr(item, "action", None)
                display_query = "(internal)"
                if action:
                    raw_query = getattr(action, "query", "") or ""
                    display_query = raw_query if raw_query and raw_query.lower() != "web search" else "(internal)"
                    for src in getattr(action, "sources", []):
                        url = getattr(src, "url", "")
                        if url and url not in stats["sources"]:
                            stats["sources"].append(url)
                _notify(f"Search {stats['search_calls']}: query={display_query} · {len(stats['sources'])} sources ({_elapsed()})")

            elif item_type == "message":
                _notify(f"Generating answer... ({_elapsed()})")
                for part in getattr(item, "content", []):
                    if getattr(part, "type", "") == "output_text":
                        stats["answer"] += getattr(part, "text", "")

        if stats["search_calls"] > 0 and stats["token_breakdown"]["input"] > 200:
            stats["token_breakdown"]["search_context"] = stats["token_breakdown"]["input"] - 200

        _notify(f"Complete — {stats['latency_ms']:.0f}ms total ({_elapsed()})")
        return stats


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    force_cc = "--chat-completions" in args
    args = [a for a in args if a != "--chat-completions"]
    backend = args[0] if args and args[0] in BACKENDS else "dashscope"
    remaining = args[1:] if args and args[0] in BACKENDS else args
    question = " ".join(remaining) or input("Ask something: ")
    agent = QwenAgent(backend=backend, force_chat_completions=force_cc)
    result = agent.ask(question)
    if result is None:
        print("Error: no result returned")
        sys.exit(1)
    answer = result.get("answer") or "(no answer)"
    print(answer)
    if result.get("sources"):
        print(f"\nSources: {result['sources']}")
