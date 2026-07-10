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

import json
import logging
import os
import sys
from pathlib import Path
from typing import Generator

from dotenv import load_dotenv
load_dotenv("env.txt") or load_dotenv(".env")

from search_tool import set_interface
from agent_pool import get_agent as _get_agent_pool


# ─── Model registry — loaded from pricing.json ─────────────────────────────
# pricing.json is the single source of truth for model IDs, display names, and
# vendor strings. Set in_playground: true there to include a model here —
# no code change required.
#
# pricing.json lives in comparison/ (one level up). This is the only
# cross-directory read at startup; everything else in grounding/ is self-contained.

def _load_grounding_models() -> dict:
    pricing_path = Path(__file__).parent.parent / "comparison" / "pricing.json"
    with open(pricing_path) as f:
        data = json.load(f)
    return {
        key: {
            "model":        cfg["model"],
            "display_name": cfg["display_name"],
            "vendor":       cfg.get("vendor", ""),
            "provider":     cfg["provider"],
        }
        for key, cfg in data["models"].items()
        if cfg.get("in_playground")
    }

GROUNDING_MODELS = _load_grounding_models()


def _create_agent(model_config: dict, force_chat_completions: bool):
    return _get_agent_pool(model_config["provider"], model_config["model"], force_chat_completions)


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
        agent = _create_agent(model_config, force_chat_completions)
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

    try:
        for event in agent.stream(question):
            yield event
    except Exception as e:
        logging.exception("agent.stream() raised an unexpected error")
        from base_agent import _empty_stats
        stats = _empty_stats(model_id)
        stats["answer"] = f"Error: {e}"
        yield {"event": "done", "error": str(e), "stats": stats}


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
            tb = s.get("token_breakdown", {})
            print(f"\n{'─'*72}")
            print(f"STATS:")
            print(f"  Total tokens:     {s.get('tokens_used', 0):,}")
            print(f"    Input:          {tb.get('input', 0):,}")
            print(f"    Output:         {tb.get('output', 0):,}")
            print(f"    Search context: ~{tb.get('search_context', 0):,}")
            print(f"  API calls:        {s.get('api_calls', 0)}")
            print(f"  Search calls:     {s.get('search_calls', 0)}")
            print(f"  Latency:          {s.get('latency_ms', 0):,}ms")
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
