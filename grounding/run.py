"""
run.py — Unified CLI for running any grounded LLM agent with full trace output.

This is the single entry point for the grounding codebase. It supports all five
LLM providers, shows the complete tool-use loop (what the LLM searched for, what
came back, how it synthesized the answer), and reports token/cost breakdowns.

Standalone CLI usage:
    python run.py claude "What happened in tech news today?"
    python run.py gpt    "Who won the 2026 men's hockey Olympic gold?"
    python run.py qwen   "Latest NVIDIA earnings"
    python run.py kimi   "Current S&P 500 price"
    python run.py llama  "Compare Tesla and BYD delivery numbers"

    python run.py gpt --chat-completions "query here"   # force legacy API path
    python run.py claude --verbose "query here"

Programmatic usage (for UI server or other consumers):
    from run import run_grounding, GROUNDING_MODELS

    # Generator yields events as the tool-use loop progresses
    for event in run_grounding("claude", "What happened today?"):
        print(event["event"], event.get("query", ""))

    # Force legacy chat.completions path for GPT/Qwen (A/B comparison):
    for event in run_grounding("gpt5.4", "Latest news", force_chat_completions=True):
        handle(event)

Requirements:
    pip install anthropic openai requests python-dotenv

The grounding/ directory is fully self-contained. Clone it and go.
"""

import os
import sys
from pathlib import Path
from typing import Generator

from dotenv import load_dotenv
load_dotenv("env.txt") or load_dotenv(".env")

from search_tool import (
    SYSTEM_PROMPT,      # Static base prompt — for display
    is_verbose,
    set_interface,
)
from base_agent import AnthropicAgent, OpenAICompatibleAgent, OpenAIResponsesAgent


# ─── Model configurations ──────────────────────────────────────────────────
# Single source of truth for every supported LLM. Each entry contains
# everything needed to create a client and run the tool-use loop.
#
# api_shape controls which base class is used:
#   "anthropic"       → AnthropicAgent
#   "responses"       → OpenAIResponsesAgent (default for GPT, Qwen)
#   "chat_completions"→ OpenAICompatibleAgent (Kimi, Llama, and fallback)

GROUNDING_MODELS = {
    "claude": {
        "model": "claude-sonnet-4-6",
        "provider": "anthropic",
        "api_shape": "anthropic",
        "display_name": "Claude Sonnet 4.6",
        "vendor": "Anthropic",
        "env_key": "ANTHROPIC_API_KEY",
        "base_url": None,
    },
    "gpt5.4": {
        "model": "gpt-5.4",
        "provider": "openai",
        "api_shape": "responses",
        "display_name": "GPT-5.4",
        "vendor": "OpenAI",
        "env_key": "OPENAI_API_KEY",
        "base_url": None,
        "max_tokens_param": "max_completion_tokens",  # GPT-5.x uses this instead of max_tokens
    },
    "qwen": {
        "model": "qwen3.7-max",
        "provider": "openai",
        "api_shape": "responses",
        "display_name": "Qwen3.7 Max",
        "vendor": "Alibaba (MaaS workspace)",
        "env_key": "DASHSCOPE_API_KEY",
        "base_url_env": "DASHSCOPE_BASE_URL",  # read from env so the workspace URL isn't hardcoded
        "extra_body": {"enable_thinking": False},  # only applies to chat.completions fallback
    },
    "kimi": {
        "model": "kimi-k2.6",
        "provider": "openai",
        "api_shape": "chat_completions",   # Moonshot /v1/responses returns 404, confirmed 2026-06-30
        "display_name": "Kimi K2.6",
        "vendor": "Moonshot AI",
        "env_key": "MOONSHOT_API_KEY",
        "base_url": "https://api.moonshot.ai/v1",
    },
    "llama": {
        "model": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        "provider": "openai",
        "api_shape": "chat_completions",
        "display_name": "Llama 4 Maverick",
        "vendor": "Meta (via Together AI)",
        "env_key": "TOGETHER_API_KEY",
        "base_url": "https://api.together.xyz/v1",
    },

    # ── Newer models (added 2026-06-30) ──────────────────────────────────────

    "claude-opus": {
        "model": "claude-opus-4-8",
        "provider": "anthropic",
        "api_shape": "anthropic",
        "display_name": "Claude Opus 4.8",
        "vendor": "Anthropic",
        "env_key": "ANTHROPIC_API_KEY",
        "base_url": None,
    },
    "gpt5.5": {
        "model": "gpt-5.5",
        "provider": "openai",
        "api_shape": "responses",
        "display_name": "GPT-5.5",
        "vendor": "OpenAI",
        "env_key": "OPENAI_API_KEY",
        "base_url": None,
        "max_tokens_param": "max_completion_tokens",
    },
}


# ─── Client factory ────────────────────────────────────────────────────────

LLM_TIMEOUT = 60  # seconds — max wait for any single LLM API call

def _create_client(model_config: dict):
    """Create the appropriate API client for the given model config."""
    provider = model_config["provider"]
    api_key = os.environ.get(model_config["env_key"], "")

    if not api_key:
        raise ValueError(
            f"{model_config['env_key']} not set. "
            f"Export it or add it to env.txt."
        )

    if provider == "anthropic":
        from anthropic import Anthropic
        return Anthropic(api_key=api_key, timeout=LLM_TIMEOUT)

    elif provider == "openai":
        from openai import OpenAI
        kwargs = {"api_key": api_key, "timeout": LLM_TIMEOUT}
        if model_config.get("base_url"):
            kwargs["base_url"] = model_config["base_url"]
        elif model_config.get("base_url_env"):
            url = os.environ.get(model_config["base_url_env"], "")
            if url:
                kwargs["base_url"] = url
        return OpenAI(**kwargs)

    else:
        raise ValueError(f"Unknown provider: {provider}")


def _create_agent(model_config: dict, client, force_chat_completions: bool):
    """Instantiate the right base class for the given model config.

    api_shape in GROUNDING_MODELS is the primary signal:
      "anthropic"       → AnthropicAgent
      "responses"       → OpenAIResponsesAgent (unless force_chat_completions=True)
      "chat_completions"→ OpenAICompatibleAgent always

    force_chat_completions has no effect on anthropic or chat_completions agents.
    """
    shape = model_config.get("api_shape", "chat_completions")
    model_id = model_config["model"]
    max_tokens_param = model_config.get("max_tokens_param", "max_tokens")
    extra_body = model_config.get("extra_body")

    if shape == "anthropic":
        return AnthropicAgent(client=client, model=model_id)
    elif shape == "responses" and not force_chat_completions:
        return OpenAIResponsesAgent(client=client, model=model_id)
    else:
        return OpenAICompatibleAgent(
            client=client,
            model=model_id,
            max_tokens_param=max_tokens_param,
            extra_body=extra_body,
        )


# ─── Streaming grounding runner ────────────────────────────────────────────

def run_grounding(
    model_key: str,
    question: str,
    force_chat_completions: bool = False,
) -> Generator[dict, None, None]:
    """Run a grounded query and yield events as the tool-use loop progresses.

    This is the core function. The CLI prints events to the console.
    The UI server sends them as SSE. Same logic, different presentation.

    Events yielded (in order):
        init           — model info, question
        tool_call      — LLM decided to search (query, round number, search_num)
        search_result  — You.com returned results (count, latency, sources)
        answer         — final answer text
        done           — full stats (tokens, costs, latency, sources)
        error          — something went wrong

    Args:
        force_chat_completions: Set True to use the legacy chat.completions path for
                                GPT and Qwen instead of the default Responses API.
                                Useful for A/B token comparison or debugging.
                                Has no effect on claude, kimi, or llama.
    """
    if model_key not in GROUNDING_MODELS:
        yield {"event": "error", "message": f"Unknown model: {model_key}. Available: {', '.join(GROUNDING_MODELS.keys())}"}
        return

    model_config = GROUNDING_MODELS[model_key]
    model_id = model_config["model"]

    try:
        client = _create_client(model_config)
    except ValueError as e:
        yield {"event": "error", "message": str(e)}
        return

    set_interface("direct_api")

    yield {
        "event": "init",
        "model_key": model_key,
        "model": model_id,
        "display_name": model_config["display_name"],
        "vendor": model_config["vendor"],
        "question": question,
    }

    agent = _create_agent(model_config, client, force_chat_completions)

    try:
        for event in agent.stream(question):
            etype = event["event"]
            if etype == "done":
                # Translate base_agent stats shape to the shape server.py expects
                s = event.get("stats", {})
                breakdown = s.get("token_breakdown", {})
                yield {
                    "event": "done",
                    "stats": {
                        "total_tokens": s.get("tokens_used", 0),
                        "input_tokens": breakdown.get("input", 0),
                        "output_tokens": breakdown.get("output", 0),
                        "search_context_tokens": breakdown.get("search_context", 0),
                        "api_calls": s.get("api_calls", 0),
                        "search_calls": s.get("search_calls", 0),
                        "sources": s.get("sources", []),
                        "search_uuid": s.get("search_uuid", ""),
                        "latency_ms": round(s.get("latency_ms", 0)),
                        "tool_calls": s.get("tool_calls", 0),
                        "model_confirmed": s.get("model"),
                    },
                }
            else:
                yield event
    except Exception as e:
        import logging
        logging.exception("agent.stream() raised an unexpected error")
        yield {
            "event": "done",
            "error": str(e),
            "stats": {
                "total_tokens": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "search_context_tokens": 0,
                "api_calls": 0,
                "search_calls": 0,
                "sources": [],
                "search_uuid": "",
                "latency_ms": 0,
                "tool_calls": 0,
                "model_confirmed": None,
            },
        }


# ─── CLI output ────────────────────────────────────────────────────────────

def print_grounding_trace(model_key: str, question: str, force_chat_completions: bool = False) -> None:
    """Run a grounding query and print the full trace to the console.

    This is the CLI presentation layer. It consumes the same generator
    that the UI server uses, just prints instead of streaming SSE.
    """
    for event in run_grounding(model_key, question, force_chat_completions=force_chat_completions):
        etype = event["event"]

        if etype == "error":
            print(f"Error: {event['message']}")
            return

        elif etype == "init":
            print(f"\n{'='*72}")
            print(f"GROUNDING: {event['display_name']} + You.com Search")
            print(f"{'='*72}")
            print(f"  Model:    {event['model']} ({event['vendor']})")
            print(f"  Query:    \"{event['question']}\"")
            print()

        elif etype == "tool_call":
            params_str = ""
            if event.get("params", {}):
                params_str = " | " + ", ".join(f"{k}={v}" for k, v in event.get("params", {}).items())
            print(f"  Round {event['round']}: Searching \"{event['query']}\"{params_str}")

        elif etype == "search_result":
            uuid_str = f" | uuid={event['search_uuid'][:16]}..." if event.get("search_uuid") else ""
            print(f"           → {event['result_count']} results in {event['latency_ms']}ms{uuid_str}")

        elif etype == "answer":
            print(f"\n{'─'*72}")
            print(f"ANSWER:\n")
            print(event["text"])
            if event["sources"]:
                print(f"\nSources:")
                for i, url in enumerate(event["sources"], 1):
                    print(f"  [{i}] {url}")

        elif etype == "done":
            s = event["stats"]
            print(f"\n{'─'*72}")
            print(f"STATS:")
            print(f"  Total tokens:     {s['total_tokens']:,}")
            print(f"    Input:          {s['input_tokens']:,}")
            print(f"    Output:         {s['output_tokens']:,}")
            print(f"    Search context: ~{s['search_context_tokens']:,}")
            print(f"  API calls:        {s['api_calls']}")
            print(f"  Search calls:     {s['search_calls']}")
            print(f"  Latency:          {s['latency_ms']:,}ms")
            if s.get("search_uuid"):
                print(f"  Search UUID:      {s['search_uuid']}")
            print(f"{'='*72}\n")


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = [a for a in sys.argv[1:] if a.startswith("--")]

    if "--verbose" in flags:
        os.environ["GROUNDING_VERBOSE"] = "1"
    force_cc = "--chat-completions" in flags

    model_key = args[0] if args else None
    query = " ".join(args[1:]) if len(args) > 1 else None

    if not model_key or not query:
        print("Usage: python run.py <model> \"your question here\"")
        print("       python run.py <model> --chat-completions \"your question\"")
        print("       python run.py <model> --verbose \"your question\"")
        print()
        print(f"Available models:")
        for key, cfg in GROUNDING_MODELS.items():
            print(f"  {key:8s}  {cfg['display_name']:20s}  ({cfg['vendor']})")
        print()
        print("Examples:")
        print("  python run.py claude \"What happened in tech news today?\"")
        print("  python run.py gpt    \"Who won the 2026 men's hockey Olympic gold?\"")
        print("  python run.py qwen   \"Latest NVIDIA earnings\"")
        print("  python run.py kimi   \"Current S&P 500 price\"")
        print("  python run.py llama  \"Compare Tesla and BYD delivery numbers\"")
        sys.exit(0)

    if model_key not in GROUNDING_MODELS:
        print(f"Unknown model: {model_key}")
        print(f"Available: {', '.join(GROUNDING_MODELS.keys())}")
        sys.exit(1)

    print_grounding_trace(model_key, query, force_chat_completions=force_cc)


if __name__ == "__main__":
    main()
