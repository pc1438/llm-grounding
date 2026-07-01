#!/usr/bin/env python3
"""
ab_prompt_test.py — A/B/C test: does the system prompt cause native to over-search?

Runs 3 queries × 3 prompts × 2 paths = 18 total calls.
Uses the SAME run_youdotcom() and run_native() from compare.py — no duplicate code.

Key question: If native search count drops with a minimal/no prompt but You.com
stays stable, then our prompt is a confound. If both behave similarly across
prompts, the finding is real.

Usage:
    python ab_prompt_test.py
    python ab_prompt_test.py --verbose
    python ab_prompt_test.py --only-c    # Run only Prompt C (no prompt) — 6 calls
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
    SYSTEM_PROMPT,
    _create_client,
    run_youdotcom,
    run_native,
    is_verbose,
)

# ─── The three prompts ─────────────────────────────────────────────────────────

PROMPT_A_NAME = "A (current)"
PROMPT_A = SYSTEM_PROMPT  # Reuse the exact same prompt from compare.py

PROMPT_B_NAME = "B (minimal)"
PROMPT_B = (
    "You are a helpful assistant with web search. "
    "Answer the user's question using your web search tool when you need current information. "
    "Cite your sources by number [1], [2], etc."
)

# Prompt C: true control — no system prompt at all.
# compare.py uses `SYSTEM_PROMPT if system_prompt is None else system_prompt`,
# so passing "" correctly sends an empty system prompt to the API.
PROMPT_C_NAME = "C (none)"
PROMPT_C = ""

ALL_PROMPTS = [
    (PROMPT_A_NAME, PROMPT_A),
    (PROMPT_B_NAME, PROMPT_B),
    (PROMPT_C_NAME, PROMPT_C),
]

# ─── Test queries: one per tier ───────────────────────────────────────────────

QUERIES = [
    {"tier": "Simple",   "query": "What were the final scores in yesterday's NBA games?"},
    {"tier": "Moderate",  "query": "What are the latest developments in weight loss drugs?"},
    {"tier": "Complex",   "query": "Compare the pricing, context windows, and benchmark scores of Claude 4.5, GPT-5.4, and Gemini 2.5"},
]

# Rate limit: Tier 2 (~300k input tokens/min)
COOLDOWN = 10  # seconds between calls
MAX_RETRIES = 3
RETRY_WAIT = 30

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _run_with_retry(fn, *args, **kwargs):
    """Retry on 429 rate limit errors."""
    for attempt in range(MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                wait = RETRY_WAIT * (1.5 ** attempt)
                print(f"      ⏳ Rate limited — waiting {wait:.0f}s (retry {attempt + 1}/{MAX_RETRIES})...")
                time.sleep(wait)
            else:
                raise
    return fn(*args, **kwargs)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    verbose = "--verbose" in sys.argv
    only_c = "--only-c" in sys.argv
    if verbose:
        os.environ["GROUNDING_VERBOSE"] = "1"

    provider = "claude"
    model_config = MODELS[provider]
    client = _create_client(model_config)

    # Select which prompts to run
    if only_c:
        prompts = [(PROMPT_C_NAME, PROMPT_C)]
        test_label = "Prompt C (no prompt) Test"
    else:
        prompts = ALL_PROMPTS
        test_label = "A/B/C Prompt Test"

    results = []
    total_calls = len(QUERIES) * len(prompts) * 2
    call_num = 0

    print(f"{test_label} — {total_calls} total calls")
    print(f"Model: {model_config['model']}")
    print(f"Cooldown: {COOLDOWN}s between calls (Tier 2 rate limits)")
    print()

    for prompt_name, prompt_text in prompts:
        for q in QUERIES:
            for path_name, runner in [("You.com", run_youdotcom), ("Native", run_native)]:
                call_num += 1
                print(f"━━━ [{call_num}/{total_calls}] Prompt {prompt_name} | {path_name} | {q['tier']} ━━━")
                print(f"  Query: {q['query']}")

                try:
                    stats = _run_with_retry(
                        runner,
                        q["query"], client, model_config,
                        system_prompt=prompt_text,
                    )
                    print(f"  ✓ {stats['search_calls']} searches, {stats['total_tokens']:,} tokens, "
                          f"{stats['api_calls']} LLM calls, {stats['latency_ms']:,.0f}ms")

                    results.append({
                        "prompt": prompt_name,
                        "path": path_name,
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
                        "prompt": prompt_name, "path": path_name,
                        "tier": q["tier"], "query": q["query"],
                        "error": str(e),
                    })

                # Cooldown between calls
                if call_num < total_calls:
                    print(f"  ⏳ Cooldown {COOLDOWN}s...")
                    time.sleep(COOLDOWN)
                print()

    # ─── Results table ────────────────────────────────────────────────────
    print("=" * 110)
    print(f"{test_label.upper()} RESULTS")
    print("=" * 110)
    for pname, ptext in prompts:
        label = ptext.strip()[:80] if ptext.strip() else "(empty)"
        print(f"\n{pname}: \"{label}{'...' if len(ptext.strip()) > 80 else ''}\"")
    print()

    print(f"{'Tier':<10} {'Path':<8} {'Prompt':<16} {'Searches':>10} {'LLM Calls':>10} "
          f"{'Tokens':>10} {'Input Tok':>10} {'Latency':>10}")
    print(f"{'-'*10} {'-'*8} {'-'*16} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    for r in sorted(results, key=lambda r: (r["tier"], r["path"], r["prompt"])):
        if "error" in r:
            print(f"{r['tier']:<10} {r['path']:<8} {r['prompt']:<16} {'ERROR':>10}")
            continue
        print(f"{r['tier']:<10} {r['path']:<8} {r['prompt']:<16} {r['search_calls']:>10} "
              f"{r['api_calls']:>10} {r['total_tokens']:>10,} {r['input_tokens']:>10,} "
              f"{r['latency_ms']:>9,.0f}ms")

    # ─── Delta analysis ──────────────────────────────────────────────────
    prompt_names = [p[0] for p in prompts]

    # Build all comparison pairs (A→B, A→C, B→C when all three are present)
    pairs = []
    for i in range(len(prompt_names)):
        for j in range(i + 1, len(prompt_names)):
            pairs.append((prompt_names[i], prompt_names[j]))

    for p1_name, p2_name in pairs:
        print(f"\n{'='*110}")
        print(f"ANALYSIS: {p1_name} → {p2_name} deltas")
        print(f"{'='*110}\n")

        for path_name in ["You.com", "Native"]:
            print(f"  {path_name}:")
            for tier in ["Simple", "Moderate", "Complex"]:
                r1 = [r for r in results if r["prompt"] == p1_name and r["path"] == path_name and r["tier"] == tier and "error" not in r]
                r2 = [r for r in results if r["prompt"] == p2_name and r["path"] == path_name and r["tier"] == tier and "error" not in r]
                if r1 and r2:
                    r1, r2 = r1[0], r2[0]
                    ds = r2["search_calls"] - r1["search_calls"]
                    dt = r2["total_tokens"] - r1["total_tokens"]
                    sign_s = "+" if ds >= 0 else ""
                    sign_t = "+" if dt >= 0 else ""
                    print(f"    {tier:<10}  {p1_name}: {r1['search_calls']} searches, {r1['total_tokens']:>8,} tok  →  "
                          f"{p2_name}: {r2['search_calls']} searches, {r2['total_tokens']:>8,} tok  "
                          f"(Δ {sign_s}{ds} searches, {sign_t}{dt:,} tok)")
                else:
                    print(f"    {tier:<10}  (missing data)")
            print()

    # ─── Verdict ──────────────────────────────────────────────────────────
    print(f"{'='*110}")
    print("INTERPRETATION GUIDE:")
    print()
    print("  Prompt C (no prompt) is the true control — LLM with zero instructions.")
    print()
    print("  If Native over-searches even with no prompt (C):")
    print("    → Over-searching is inherent to native search, not prompt-induced.")
    print("    → Our benchmark with Prompt A is fair as-is.")
    print()
    print("  If Native calms down with no prompt (C) but not with A:")
    print("    → The prompt is actively causing over-searching.")
    print("    → Consider using C or B for the benchmark.")
    print()
    print("  If You.com degrades with no prompt (C):")
    print("    → The prompt is helping You.com perform well.")
    print("    → Keep Prompt A — it represents real-world usage.")
    print(f"{'='*110}\n")

    # Save raw results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(__file__).parent / f"ab_prompt_results_{ts}.json"
    prompts_dict = {"A": PROMPT_A, "B": PROMPT_B, "C": PROMPT_C.strip() or "(none)"}
    with open(out_path, "w") as f:
        json.dump({
            "test": "A/B/C system prompt impact on search behavior",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prompts": prompts_dict,
            "results": results,
        }, f, indent=2)
    print(f"Raw results saved to: {out_path}")


if __name__ == "__main__":
    main()
