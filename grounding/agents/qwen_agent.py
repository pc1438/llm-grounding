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
"""

import os
import sys

from openai import OpenAI

from base_agent import OpenAICompatibleAgent, OpenAIResponsesAgent

BACKENDS = {
    "dashscope": {
        "api_key_env": "DASHSCOPE_API_KEY",
        # Domestic DashScope endpoint (lower latency for CN/AP region).
        # compare.py uses dashscope-intl.aliyuncs.com (international endpoint,
        # used in that context because the comparison runs from a global env).
        # run.py uses DASHSCOPE_BASE_URL env var so the workspace URL can be
        # configured per deployment without code changes.
        # All three reach the same models — the difference is routing only.
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        # Default model for direct/programmatic use. run.py pins "qwen3.7-max"
        # in GROUNDING_MODELS for its own display/cost accounting. These are
        # intentionally separate; do NOT auto-sync.
        "default_model": "qwen-plus-2025-09-11",
        # Also: "qwen3-max", "qwen3.5-plus", "qwen3.5-flash"
        "supports_responses_api": True,
    },
    "huggingface": {
        "api_key_env": "HF_TOKEN",
        "base_url": "https://router.huggingface.co/v1",
        "default_model": "Qwen/Qwen3-8B",
        # Also: "Qwen/Qwen3.5-27B", "Qwen/Qwen3.5-4B"
        "supports_responses_api": False,  # HF endpoint does not support /v1/responses
    },
}


def QwenAgent(
    backend: str = "dashscope",
    model: str | None = None,
    api_key: str | None = None,
    force_chat_completions: bool = False,
):
    """Factory that returns the right agent class based on backend and force_chat_completions.

    DashScope + default:               OpenAIResponsesAgent (Responses API)
    DashScope + force_chat_completions: OpenAICompatibleAgent (chat.completions)
    HuggingFace (any):                 OpenAICompatibleAgent (chat.completions only)

    The force_chat_completions flag is set once at construction — it determines
    which loop implementation the agent uses for its lifetime, not per-call.
    """
    if backend not in BACKENDS:
        raise ValueError(f"Unknown backend: {backend!r}. Choose from: {list(BACKENDS)}")

    cfg = BACKENDS[backend]
    resolved_key = api_key or os.environ.get(cfg["api_key_env"], "")
    if not resolved_key:
        raise ValueError(
            f"{cfg['api_key_env']} not set. "
            f"Pass api_key= or export {cfg['api_key_env']}."
        )
    client = OpenAI(api_key=resolved_key, base_url=cfg["base_url"])
    resolved_model = model or cfg["default_model"]

    use_responses = cfg["supports_responses_api"] and not force_chat_completions
    cls = OpenAIResponsesAgent if use_responses else OpenAICompatibleAgent
    return cls(client=client, model=resolved_model)


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
