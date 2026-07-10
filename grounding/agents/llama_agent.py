"""
llama_agent.py — Llama 4 agent with You.com web search tool.

Supports two backends:
  1. Together AI (cloud) — easiest way to run Llama 4 with function calling
  2. Ollama (local)      — for teams running models on their own hardware

    from agents.llama_agent import LlamaAgent

    agent = LlamaAgent()                     # Together AI (default)
    agent = LlamaAgent(backend="ollama")     # Local via Ollama

Requirements:
    pip install openai
    export YDC_API_KEY="..."
    export TOGETHER_API_KEY="..."    # Together AI
    # OR: ollama pull llama4          # Ollama (local)

Responses API note:
    Together AI and Ollama do not support the OpenAI Responses API.
    force_chat_completions is accepted as a parameter for API consistency but
    has no effect — this agent always uses chat.completions.
"""

import os
import sys

from openai import OpenAI

from base_agent import OpenAICompatibleAgent, _empty_stats

BACKENDS = {
    "together": {
        "api_key_env": "TOGETHER_API_KEY",
        "base_url": "https://api.together.xyz/v1",
        # Llama 4 Maverick: MoE, 17B active / 400B total, 1M context
        "default_model": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        # Also: "meta-llama/Llama-4-Scout-17B-16E-Instruct" (10M context)
        #        "meta-llama/Llama-3.3-70B-Instruct-Turbo"  (previous gen)
    },
    "ollama": {
        "api_key_env": None,  # Ollama runs locally, no API key needed
        "base_url": "http://localhost:11434/v1",
        "default_model": "llama4",
        # Also: "llama3.3", "llama3.1"
    },
}


class LlamaAgent(OpenAICompatibleAgent):
    """Llama-powered agent with You.com web search.

    Uses OpenAI-compatible function calling via Together AI or Ollama.
    Always uses chat.completions — Together AI and Ollama don't support Responses API.

    Return value of ask() and run_native_search(): see base_agent._empty_stats()
    for the full field-by-field reference.
    """

    def __init__(
        self,
        backend: str = "together",
        model: str | None = None,
        api_key: str | None = None,
        force_chat_completions: bool = False,  # accepted for API consistency, no effect
    ):
        if backend not in BACKENDS:
            raise ValueError(f"Unknown backend: {backend!r}. Choose from: {list(BACKENDS)}")

        cfg = BACKENDS[backend]

        if backend == "ollama":
            # Ollama doesn't use API keys. The OpenAI client requires a
            # non-empty string, so we pass a placeholder.
            resolved_key = "ollama"
        else:
            resolved_key = api_key or os.environ.get(cfg["api_key_env"], "")
            if not resolved_key:
                raise ValueError(
                    f"{cfg['api_key_env']} not set. "
                    f"Pass api_key= or export {cfg['api_key_env']}."
                )

        client = OpenAI(api_key=resolved_key, base_url=cfg["base_url"], timeout=120.0)
        model = model or cfg["default_model"]
        super().__init__(client=client, model=model)


    def run_native_search(
        self,
        question: str,
        system_prompt: str = None,
        on_progress=None,
        **kwargs,
    ) -> dict:
        """Native search is not supported for Llama (Together AI / Ollama have no built-in search)."""
        stats = _empty_stats(self.model)
        stats["interface"] = "not_supported"
        stats["answer"] = "Native web search is not available for Llama via Together AI or Ollama."
        return stats


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    backend = args[0] if args and args[0] in BACKENDS else "together"
    remaining = args[1:] if args and args[0] in BACKENDS else args
    question = " ".join(remaining) or input("Ask something: ")
    agent = LlamaAgent(backend=backend)
    result = agent.ask(question)
    if result is None:
        print("Error: no result returned")
        sys.exit(1)
    answer = result.get("answer") or "(no answer)"
    print(answer)
    if result.get("sources"):
        print(f"\nSources: {result['sources']}")
