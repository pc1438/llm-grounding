#!/usr/bin/env python3
"""
ab_tool_version_test.py — A/B/C test: does the native web search tool version
affect token usage, search behavior, and answer quality?

Tests three configurations (native path only):
  A) web_search_20250305 with defaults (no code execution)
  B) web_search_20260209 with defaults (code execution enabled)
  C) web_search_20260209 with allowed_callers=["direct"] (code execution disabled)

The You.com path is not affected — this test isolates the native search behavior.

Usage:
    python ab_tool_version_test.py --verbose
    python ab_tool_version_test.py --only B --verbose
    python ab_tool_version_test.py --only C --verbose
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
        "label": "A: 20250305 (defaults)",
        "key": "A",
        "tool": {
            "type": "web_search_20250305",
            "name": "web_search",
        },
    },
    {
        "label": "B: 20260209 (defaults — code execution enabled)",
        "key": "B",
        "tool": {
            "type": "web_search_20260209",
            "name": "web_search",
        },
    },
    {
        "label": "C: 20260209 (allowed_callers=[direct] — code execution disabled)",
        "key": "C",
        "tool": {
            "type": "web_search_20260209",
            "name": "web_search",
            "allowed_callers": ["direct"],
        },
    },
]

QUERIES = [
    {"tier": "Simple",   "query": "What is the current price of NVIDIA stock?"},
    {"tier": "Moderate",  "query": "What are the latest developments in weight loss drugs?"},
    {"tier": "Complex",   "query": "Compare the pricing, context windows, and benchmark scores of Claude 4.5, GPT-5.4, and Gemini 2.5"},
]

# Rate limit: Tier 2
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

    provider = "claude"
    model_config = MODELS[provider]
    client = _create_client(model_config)

    # Save the original tool config so we can restore it
    original_tool = model_config["native_search_tool"].copy()

    results = []
    total_calls = len(QUERIES) * len(configs)
    call_num = 0

    print(f"Native Tool Version A/B/C Test — {total_calls} total calls (native path only)")
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
                print(f"  ✓ {stats['search_calls']} searches, {stats['total_tokens']:,} tokens, "
                      f"{stats['api_calls']} LLM calls, {stats['latency_ms']:,.0f}ms")

                results.append({
                    "config_key": config["key"],
                    "config_label": config["label"],
                    "tool_type": config["tool"]["type"],
                    "tier": q["tier"],
                    "query": q["query"],
                    "search_calls": stats["search_calls"],
                    "api_calls": stats["api_calls"],
                    "total_tokens": stats["total_tokens"],
                    "input_tokens": stats["input_tokens"],
                    "output_tokens": stats["output_tokens"],
                    "search_context_tokens": stats.get("search_context_tokens", 0),
                    "latency_ms": stats["latency_ms"],
                    "source_count": len(stats.get("sources", [])),
                })
            except Exception as e:
                print(f"  ✗ FAILED: {e}")
                results.append({
                    "config_key": config["key"], "config_label": config["label"],
                    "tool_type": config["tool"]["type"],
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
    print("=" * 110)
    print("NATIVE TOOL VERSION A/B/C TEST RESULTS")
    print("=" * 110)
    print()

    print(f"{'Tier':<10} {'Config':<50} {'Searches':>10} {'LLM Calls':>10} "
          f"{'Tokens':>10} {'Input Tok':>10} {'Latency':>10}")
    print(f"{'-'*10} {'-'*50} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    for r in sorted(results, key=lambda r: (r["tier"], r["config_key"])):
        if "error" in r:
            print(f"{r['tier']:<10} {r['config_label']:<50} {'ERROR':>10}")
            continue
        print(f"{r['tier']:<10} {r['config_label']:<50} {r['search_calls']:>10} "
              f"{r['api_calls']:>10} {r['total_tokens']:>10,} {r['input_tokens']:>10,} "
              f"{r['latency_ms']:>9,.0f}ms")

    # ─── Delta analysis (A as baseline) ──────────────────────────────────
    print(f"\n{'='*110}")
    print("DELTA ANALYSIS: Config A (20250305) as baseline")
    print(f"{'='*110}\n")

    for tier in ["Simple", "Moderate", "Complex"]:
        rA = [r for r in results if r["config_key"] == "A" and r["tier"] == tier and "error" not in r]

        for other_key in ["B", "C"]:
            rX = [r for r in results if r["config_key"] == other_key and r["tier"] == tier and "error" not in r]
            if rA and rX:
                a, x = rA[0], rX[0]
                dt = x["total_tokens"] - a["total_tokens"]
                dl = x["latency_ms"] - a["latency_ms"]
                ds = x["search_calls"] - a["search_calls"]
                pct_t = (dt / a["total_tokens"] * 100) if a["total_tokens"] else 0
                sign_t = "+" if dt >= 0 else ""
                sign_l = "+" if dl >= 0 else ""
                sign_s = "+" if ds >= 0 else ""
                print(f"  {tier:<10} A→{other_key}: {sign_t}{dt:,} tok ({sign_t}{pct_t:.0f}%), "
                      f"{sign_l}{dl:,.0f}ms, {sign_s}{ds} searches")

        print()

    # ─── Interpretation guide ────────────────────────────────────────────
    print(f"{'='*110}")
    print("INTERPRETATION GUIDE:")
    print()
    print("  If B >> A in tokens/searches:")
    print("    → Code execution loop in 20260209 inflates native. Our benchmark used this.")
    print("    → The 65.5% cost savings headline may be partly due to code execution overhead.")
    print()
    print("  If C ≈ A:")
    print("    → 20260209 with code execution disabled behaves like 20250305.")
    print("    → Confirms code execution is the differentiator, not the search backend version.")
    print()
    print("  If C < A (fewer tokens):")
    print("    → 20260209 has a better search backend, and disabling code execution is the way to go.")
    print("    → Use C as our benchmark config (fairest to native).")
    print()
    print("  If B ≈ A:")
    print("    → Code execution doesn't materially change things. Version choice doesn't matter.")
    print(f"{'='*110}\n")

    # Save raw results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(__file__).parent / f"ab_claude_native_tool_version_results_{ts}.json"
    with open(out_path, "w") as f:
        json.dump({
            "test": "A/B/C native tool version impact on token usage and search behavior",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "configs": {c["key"]: c["label"] for c in TOOL_CONFIGS},
            "model": MODELS[provider]["model"],
            "results": results,
        }, f, indent=2)
    print(f"Raw results saved to: {out_path}")


if __name__ == "__main__":
    main()
