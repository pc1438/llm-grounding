#!/usr/bin/env python3
"""
ab_gpt_search_context_size_test.py — A/B/C test: does OpenAI's search_context_size
setting affect token usage, search behavior, source count, and answer quality?

Tests three configurations (native path only, GPT-5.4):
  A) search_context_size = "low"    — minimal search context
  B) search_context_size = "medium" — default
  C) search_context_size = "high"   — maximum search context

The You.com path is not affected — this test isolates OpenAI's native search behavior.

OpenAI docs describe search_context_size as: "High level guidance for the amount of
context window space to use for the search. One of low, medium, or high."

We're looking for:
  - Token usage differences (does "high" significantly inflate input tokens?)
  - Source/citation behavior (does "high" return more url_citation annotations?)
  - Latency differences
  - Whether the current default ("medium") is the fairest config for benchmarking

Usage:
    python ab_gpt_search_context_size_test.py --verbose
    python ab_gpt_search_context_size_test.py --only A --verbose
    python ab_gpt_search_context_size_test.py --only C --verbose
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from compare import (
    MODELS,
    _create_client,
    run_native,
    is_verbose,
)

# ─── Test configurations ─────────────────────────────────────────────────────

TOOL_CONFIGS = [
    {
        "label": "A: search_context_size=low",
        "key": "A",
        "tool": {
            "type": "web_search",
            "search_context_size": "low",
        },
    },
    {
        "label": "B: search_context_size=medium (default)",
        "key": "B",
        "tool": {
            "type": "web_search",
            "search_context_size": "medium",
        },
    },
    {
        "label": "C: search_context_size=high",
        "key": "C",
        "tool": {
            "type": "web_search",
            "search_context_size": "high",
        },
    },
]

# Same query tiers as the Claude A/B/C test for comparability
QUERIES = [
    {"tier": "Simple",   "query": "What is the current price of NVIDIA stock?"},
    {"tier": "Moderate", "query": "What are the latest developments in weight loss drugs?"},
    {"tier": "Complex",  "query": "Compare the pricing, context windows, and benchmark scores of Claude 4.5, GPT-5.4, and Gemini 2.5"},
]

# Rate limit safety
COOLDOWN = 10
MAX_RETRIES = 3
RETRY_WAIT = 30


def _run_with_retry(fn, *args, **kwargs):
    for attempt in range(MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                wait = RETRY_WAIT * (1.5 ** attempt)
                print(f"      Rate limited — waiting {wait:.0f}s (retry {attempt + 1}/{MAX_RETRIES})...")
                time.sleep(wait)
            else:
                raise
    return fn(*args, **kwargs)


def main():
    verbose = "--verbose" in sys.argv
    if verbose:
        os.environ["GROUNDING_VERBOSE"] = "1"

    # --only X: run only a specific config (A, B, or C)
    only_config = None
    for i, arg in enumerate(sys.argv):
        if arg == "--only" and i + 1 < len(sys.argv):
            only_config = sys.argv[i + 1].upper()

    if only_config:
        configs = [c for c in TOOL_CONFIGS if c["key"] == only_config]
        if not configs:
            print(f"Unknown config: {only_config}. Valid: A, B, C")
            sys.exit(1)
    else:
        configs = TOOL_CONFIGS

    provider = "openai"
    model_config = MODELS[provider]
    client = _create_client(model_config)

    # Save the original tool config so we can restore it
    original_tool = model_config["native_search_tool"].copy()

    results = []
    total_calls = len(QUERIES) * len(configs)
    call_num = 0

    print(f"GPT Native Search Context Size A/B/C Test — {total_calls} total calls (native path only)")
    print(f"Model: {model_config['model']}")
    print(f"Cooldown: {COOLDOWN}s between calls")
    print(f"Configs: {', '.join(c['label'] for c in configs)}")
    print()

    for config in configs:
        # Swap the native search tool config
        model_config["native_search_tool"] = config["tool"]
        print(f"── Tool config: {config['label']} ──")
        print(f"   {json.dumps(config['tool'])}")
        print()

        for q in QUERIES:
            call_num += 1
            print(f"━━━ [{call_num}/{total_calls}] {config['label']} | {q['tier']} ━━━")
            print(f"  Query: {q['query']}")

            try:
                stats = _run_with_retry(run_native, q["query"], client, model_config)
                source_count = len(stats.get("sources", []))
                print(f"  ✓ {stats['search_calls']} searches, {stats['total_tokens']:,} tokens, "
                      f"{stats['api_calls']} LLM calls, {stats['latency_ms']:,.0f}ms, "
                      f"{source_count} sources")

                results.append({
                    "config_key": config["key"],
                    "config_label": config["label"],
                    "search_context_size": config["tool"]["search_context_size"],
                    "tier": q["tier"],
                    "query": q["query"],
                    "search_calls": stats["search_calls"],
                    "api_calls": stats["api_calls"],
                    "total_tokens": stats["total_tokens"],
                    "input_tokens": stats["input_tokens"],
                    "output_tokens": stats["output_tokens"],
                    "search_context_tokens": stats.get("search_context_tokens", 0),
                    "latency_ms": stats["latency_ms"],
                    "source_count": source_count,
                    "sources": stats.get("sources", []),
                    "answer_length": len(stats.get("answer", "")),
                })
            except Exception as e:
                print(f"  ✗ FAILED: {e}")
                results.append({
                    "config_key": config["key"], "config_label": config["label"],
                    "search_context_size": config["tool"]["search_context_size"],
                    "tier": q["tier"], "query": q["query"],
                    "error": str(e),
                })

            if call_num < total_calls:
                print(f"  ⏳ Cooldown {COOLDOWN}s...")
                time.sleep(COOLDOWN)
            print()

    # Restore original tool config
    model_config["native_search_tool"] = original_tool

    # ─── Results table ────────────────────────────────────────────────────
    print("=" * 120)
    print("GPT NATIVE SEARCH CONTEXT SIZE A/B/C TEST RESULTS")
    print("=" * 120)
    print()

    print(f"{'Tier':<10} {'Config':<42} {'Searches':>10} {'Tokens':>10} "
          f"{'Input Tok':>10} {'Output Tok':>10} {'Sources':>10} {'Latency':>10}")
    print(f"{'-'*10} {'-'*42} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    for r in sorted(results, key=lambda r: (r["tier"], r["config_key"])):
        if "error" in r:
            print(f"{r['tier']:<10} {r['config_label']:<42} {'ERROR':>10}")
            continue
        print(f"{r['tier']:<10} {r['config_label']:<42} {r['search_calls']:>10} "
              f"{r['total_tokens']:>10,} {r['input_tokens']:>10,} {r['output_tokens']:>10,} "
              f"{r['source_count']:>10} {r['latency_ms']:>9,.0f}ms")

    # ─── Delta analysis (B/medium as baseline, since it's the current default) ─
    print(f"\n{'='*120}")
    print("DELTA ANALYSIS: Config B (medium — current default) as baseline")
    print(f"{'='*120}\n")

    for tier in ["Simple", "Moderate", "Complex"]:
        rB = [r for r in results if r["config_key"] == "B" and r["tier"] == tier and "error" not in r]

        for other_key in ["A", "C"]:
            rX = [r for r in results if r["config_key"] == other_key and r["tier"] == tier and "error" not in r]
            if rB and rX:
                b, x = rB[0], rX[0]
                dt = x["total_tokens"] - b["total_tokens"]
                dl = x["latency_ms"] - b["latency_ms"]
                ds = x["search_calls"] - b["search_calls"]
                dsrc = x["source_count"] - b["source_count"]
                pct_t = (dt / b["total_tokens"] * 100) if b["total_tokens"] else 0
                sign_t = "+" if dt >= 0 else ""
                sign_l = "+" if dl >= 0 else ""
                sign_s = "+" if ds >= 0 else ""
                sign_src = "+" if dsrc >= 0 else ""
                print(f"  {tier:<10} B→{other_key}: {sign_t}{dt:,} tok ({sign_t}{pct_t:.0f}%), "
                      f"{sign_l}{dl:,.0f}ms, {sign_s}{ds} searches, {sign_src}{dsrc} sources")

        print()

    # ─── Interpretation guide ────────────────────────────────────────────
    print(f"{'='*120}")
    print("INTERPRETATION GUIDE:")
    print()
    print("  If C (high) >> B (medium) in tokens:")
    print("    → More context = more tokens = higher inference cost.")
    print("    → But if C also gets more sources/better answers, may be worth it for quality.")
    print()
    print("  If A (low) << B (medium) in tokens:")
    print("    → Low context saves tokens. Check if answer quality degrades.")
    print("    → If quality holds, 'low' could be the fairest config (cheapest native).")
    print()
    print("  If sources differ significantly:")
    print("    → search_context_size may affect how many url_citations GPT returns.")
    print("    → More sources = better citation quality scores from the judge.")
    print()
    print("  Current benchmark uses medium (default). If medium is the most balanced,")
    print("  we keep it. If high gives native an unfair advantage, stick with medium.")
    print(f"{'='*120}\n")

    # Save raw results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(__file__).parent / f"ab_gpt_search_context_size_results_{ts}.json"
    with open(out_path, "w") as f:
        json.dump({
            "test": "A/B/C GPT native search_context_size impact on tokens, sources, and latency",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "configs": {c["key"]: c["label"] for c in TOOL_CONFIGS},
            "model": MODELS[provider]["model"],
            "results": results,
        }, f, indent=2)
    print(f"Raw results saved to: {out_path}")


if __name__ == "__main__":
    main()
