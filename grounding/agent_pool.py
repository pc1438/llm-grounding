"""
agent_pool.py — Shared agent instance cache.

Provides a single get_agent() function used by both compare.py and run.py.
Caching reuses httpx connection pools across requests, avoiding TCP+TLS
handshake overhead on every call.

Cache key: (provider, model_id, force_chat_completions)
force_chat_completions only has effect for openai and qwen providers.

Pre-warming at server startup (prewarm()) ensures all lazy module initialization
happens in the main thread before parallel requests arrive, preventing threading
deadlocks on first use.
"""

import logging
import threading

logger = logging.getLogger(__name__)

_agent_cache: dict = {}
_cache_lock = threading.Lock()


def get_agent(provider: str, model_id: str, force_chat_completions: bool = False):
    """Return a cached agent instance, creating it if needed.

    Thread-safe: uses a lock during cache population to prevent duplicate
    instantiation and module-load races under parallel requests.
    """
    key = (provider, model_id, force_chat_completions)
    if key in _agent_cache:
        return _agent_cache[key]

    with _cache_lock:
        if key in _agent_cache:
            return _agent_cache[key]

        from agents.claude_agent import ClaudeAgent
        from agents.openai_agent import OpenAIAgent
        from agents.kimi_agent import KimiAgent
        from agents.qwen_agent import QwenAgent
        from agents.openrouter_agent import OpenRouterAgent

        if provider == "anthropic":
            _agent_cache[key] = ClaudeAgent(model=model_id)
        elif provider == "openai":
            _agent_cache[key] = OpenAIAgent(model=model_id, force_chat_completions=force_chat_completions)
        elif provider == "kimi":
            _agent_cache[key] = KimiAgent(model=model_id)
        elif provider == "qwen":
            _agent_cache[key] = QwenAgent(model=model_id, force_chat_completions=force_chat_completions)
        elif provider == "openrouter":
            _agent_cache[key] = OpenRouterAgent(model=model_id)
        else:
            raise ValueError(f"Unknown provider: {provider!r} for model {model_id}")

    return _agent_cache[key]


def prewarm(models: dict) -> None:
    """Instantiate all agents at startup to avoid lazy-init races under parallel requests.

    Call once from the server main thread before serving requests.
    Skips providers whose API keys are missing — logs a warning but does not raise.

    Args:
        models: dict of {key: model_config} as loaded from pricing.json.
                Each config must have 'provider' and 'model' fields.
    """
    for key, cfg in models.items():
        provider = cfg.get("provider", "")
        model_id = cfg.get("model", "")
        if not provider or not model_id:
            continue
        try:
            get_agent(provider, model_id)
            logger.info("agent_pool: pre-warmed %s (%s)", key, model_id)
        except Exception as e:
            logger.warning("agent_pool: skipped %s — %s", key, e)
