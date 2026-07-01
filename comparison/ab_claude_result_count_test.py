#!/usr/bin/env python3
"""
ab_count_test.py — A/B test: does changing default result count from 5 to 10
affect token usage, search behavior, and answer quality?

Runs 3 queries × 2 counts × You.com path only = 6 total calls.
Modifies the tool schema's default count before each run.

Usage:
    python ab_count_test.py --verbose
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
    run_youdotcom,
    is_verbose,
)
from search_tool import TOOL_SCHEMA, TOOL_SCHEMA_ANTHROPIC

# ─── Test configurations ─────────────────────────────────────────────────────

ALL_COUNTS = [
    ("count=5 (current)", 5),
    ("count=8", 8),
    ("count=10", 10),
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


def _set_default_count(count: int):
    """Modify the tool schema's default count in-place.

    Both TOOL_SCHEMA and TOOL_SCHEMA_ANTHROPIC reference the same
    parameters dict, so changing one changes both.
    """
    TOOL_SCHEMA["function"]["parameters"]["properties"]["count"]["default"] = count
    TOOL_SCHEMA["function"]["parameters"]["properties"]["count"]["description"] = (
        f"Number of results to return (1-20). Default {count}."
    )
    # TOOL_SCHEMA_ANTHROPIC shares input_schema with TOOL_SCHEMA's parameters,
    # but update explicitly to be safe
    TOOL_SCHEMA_ANTHROPIC["input_schema"]["properties"]["count"]["default"] = count
    TOOL_SCHEMA_ANTHROPIC["input_schema"]["properties"]["count"]["description"] = (
        f"Number of results to return (1-20). Default {count}."
    )


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

    # --only N: run only a specific count value (e.g., --only 8)
    only_count = None
    for i, arg in enumerate(sys.argv):
        if arg == "--only" and i + 1 < len(sys.argv):
            only_count = int(sys.argv[i + 1])

    if only_count is not None:
        counts = [(f"count={only_count}", only_count)]
    else:
        counts = ALL_COUNTS

    provider = "claude"
    model_config = MODELS[provider]
    client = _create_client(model_config)

    results = []
    total_calls = len(QUERIES) * len(counts)
    call_num = 0

    print(f"Count A/B Test — {total_calls} total calls (You.com path only)")
    print(f"Model: {model_config['model']}")
    print(f"Cooldown: {COOLDOWN}s between calls")
    print()

    for count_name, count_value in counts:
        _set_default_count(count_value)

        for q in QUERIES:
            call_num += 1
            print(f"━━━ [{call_num}/{total_calls}] {count_name} | {q['tier']} ━━━")
            print(f"  Query: {q['query']}")

            try:
                stats = _run_with_retry(run_youdotcom, q["query"], client, model_config)
                print(f"  ✓ {stats['search_calls']} searches, {stats['total_tokens']:,} tokens, "
                      f"{stats['api_calls']} LLM calls, {stats['latency_ms']:,.0f}ms")

                results.append({
                    "count_config": count_name,
                    "count_value": count_value,
                    "tier": q["tier"],
                    "query": q["query"],
                    "search_calls": stats["search_calls"],
                    "api_calls": stats["api_calls"],
                    "total_tokens": stats["total_tokens"],
                    "input_tokens": stats["input_tokens"],
                    "output_tokens": stats["output_tokens"],
                    "latency_ms": stats["latency_ms"],
                })
            except Exception as e:
                print(f"  ✗ FAILED: {e}")
                results.append({
                    "count_config": count_name, "count_value": count_value,
                    "tier": q["tier"], "query": q["query"],
                    "error": str(e),
                })

            if call_num < total_calls:
                print(f"  ⏳ Cooldown {COOLDOWN}s...")
                time.sleep(COOLDOWN)
            print()

    # Restore default
    _set_default_count(5)

    # ─── Results table ────────────────────────────────────────────────────
    print("=" * 100)
    print("COUNT A/B TEST RESULTS")
    print("=" * 100)
    print()

    print(f"{'Tier':<10} {'Count':<20} {'Searches':>10} {'LLM Calls':>10} "
          f"{'Tokens':>10} {'Input Tok':>10} {'Latency':>10}")
    print(f"{'-'*10} {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    for r in sorted(results, key=lambda r: (r["tier"], r["count_value"])):
        if "error" in r:
            print(f"{r['tier']:<10} {r['count_config']:<20} {'ERROR':>10}")
            continue
        print(f"{r['tier']:<10} {r['count_config']:<20} {r['search_calls']:>10} "
              f"{r['api_calls']:>10} {r['total_tokens']:>10,} {r['input_tokens']:>10,} "
              f"{r['latency_ms']:>9,.0f}ms")

    # ─── Delta analysis ──────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print("ANALYSIS: count=5 → count=10 deltas")
    print(f"{'='*100}\n")

    for tier in ["Simple", "Moderate", "Complex"]:
        r5 = [r for r in results if r["count_value"] == 5 and r["tier"] == tier and "error" not in r]
        r10 = [r for r in results if r["count_value"] == 10 and r["tier"] == tier and "error" not in r]
        if r5 and r10:
            r5, r10 = r5[0], r10[0]
            ds = r10["search_calls"] - r5["search_calls"]
            dt = r10["total_tokens"] - r5["total_tokens"]
            dl = r10["latency_ms"] - r5["latency_ms"]
            sign_s = "+" if ds >= 0 else ""
            sign_t = "+" if dt >= 0 else ""
            sign_l = "+" if dl >= 0 else ""
            print(f"  {tier:<10}  5: {r5['search_calls']} searches, {r5['total_tokens']:>8,} tok, {r5['latency_ms']:>7,.0f}ms")
            print(f"  {'':<10} 10: {r10['search_calls']} searches, {r10['total_tokens']:>8,} tok, {r10['latency_ms']:>7,.0f}ms")
            print(f"  {'':<10}  Δ: {sign_s}{ds} searches, {sign_t}{dt:,} tok, {sign_l}{dl:,.0f}ms")
            print()
        else:
            print(f"  {tier:<10}  (missing data)")

    # ─── Interpretation ──────────────────────────────────────────────────
    print(f"{'='*100}")
    print("INTERPRETATION GUIDE:")
    print()
    print("  If count=10 uses significantly more tokens but same number of searches:")
    print("    → More results per search = more context per round. Token cost goes up.")
    print("    → Only worth it if answer quality noticeably improves.")
    print()
    print("  If count=10 uses fewer searches (LLM satisfied sooner):")
    print("    → Richer results reduce the need for follow-up searches.")
    print("    → Net token impact depends on whether fewer searches offset larger results.")
    print()
    print("  If count=10 shows no meaningful difference:")
    print("    → The LLM already gets what it needs from 5 results.")
    print("    → Keep count=5 (cheaper).")
    print(f"{'='*100}\n")

    # Save raw results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(__file__).parent / f"ab_count_results_{ts}.json"
    with open(out_path, "w") as f:
        json.dump({
            "test": "A/B result count impact on token usage and search behavior",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "counts": {"A": 5, "B": 10},
            "results": results,
        }, f, indent=2)
    print(f"Raw results saved to: {out_path}")


if __name__ == "__main__":
    main()
