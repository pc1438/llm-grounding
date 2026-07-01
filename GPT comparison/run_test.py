#!/usr/bin/env python3
"""
run_test.py — Run benchmark queries through all 3 GPT + Web Search paths
and save detailed results to a timestamped JSON file.

Usage:
    python run_test.py                  # Run ALL 50 benchmark questions
    python run_test.py --tier 1         # Run only Tier 1 (Simple) questions
    python run_test.py --tier 1,2       # Run Tiers 1 and 2
    python run_test.py --ids 5,13,33,47 # Run specific question IDs
    python run_test.py --no-judge       # Skip judge evaluation (faster)
    python run_test.py --verbose        # Full logging
    python run_test.py --cooldown 10    # Seconds between queries (default 5)

Output:
    results_YYYYMMDD_HHMMSS.json in the same directory as this script.
"""

import json
import sys
import os
import time
from datetime import datetime, timezone
from pathlib import Path

# Import everything from the main comparison script
sys.path.insert(0, str(Path(__file__).parent))
from gpt_comparison import (
    MODEL,
    PRICING,
    OPENAI_API_KEY,
    YDC_API_KEY,
    ANTHROPIC_API_KEY,
    run_native,
    run_youdotcom_original,
    run_youdotcom_optimized,
    run_judge,
    compute_cost,
    _verbose,
)
from openai import OpenAI

# ─── Load benchmark questions from JSON ─────────────────────────────────────

BENCHMARK_FILE = Path(__file__).parent.parent / "comparison" / "benchmark_questions_50.json"


def load_questions(tier_filter: list[int] | None = None,
                   id_filter: list[int] | None = None) -> list[dict]:
    """Load benchmark questions, optionally filtering by tier or ID."""
    with open(BENCHMARK_FILE, "r") as f:
        data = json.load(f)

    questions = []
    for tier_block in data["tiers"]:
        tier_num = tier_block["tier"]
        tier_name = tier_block["name"]
        for q in tier_block["questions"]:
            questions.append({
                "id": q["id"],
                "tier": tier_num,
                "tier_name": tier_name,
                "domain": q["domain"],
                "query": q["query"],
            })

    # Apply filters
    if id_filter:
        questions = [q for q in questions if q["id"] in id_filter]
    elif tier_filter:
        questions = [q for q in questions if q["tier"] in tier_filter]

    return questions

LLM_TIMEOUT = 300


# ─── Per-run metrics (matches the screenshot format) ─────────────────────────

def build_run_record(stats: dict, path_type: str) -> dict:
    """Build a complete metrics record for one path run.

    Captures every field from the comparison dashboard:
    - Token usage: total, input, output, search context (est.)
    - Performance: latency, web search API calls, LLM round-trips, sources
    - Cost: total, LLM inference, search (broken out)
    """
    cost = compute_cost(stats, path_type)
    return {
        # Token usage
        "total_tokens": stats["total_tokens"],
        "input_tokens": stats["input_tokens"],
        "output_tokens": stats["output_tokens"],
        "search_context_tokens": stats["search_context_tokens"],

        # Performance
        "latency_ms": round(stats["latency_ms"], 1),
        "web_search_api_calls": stats["search_calls"],
        "llm_round_trips": stats["api_calls"],
        "sources_returned": len(stats.get("sources", [])),
        "sources": stats.get("sources", []),
        "hit_round_limit": stats["hit_round_limit"],

        # Cost breakdown
        "total_cost": cost["total_cost"],
        "llm_inference_cost": cost["llm_cost"],
        "search_cost": cost["search_cost"],

        # Answer (for judge + debugging)
        "answer_length": len(stats.get("answer", "")),
        "answer": stats.get("answer", ""),
    }


def compute_ratios(ydc_record: dict, native_record: dict) -> dict:
    """Compute You.com-to-Native ratios (matching the screenshot's 'Ratio' column)."""
    ratios = {}
    for key in ["total_tokens", "input_tokens", "search_context_tokens",
                "total_cost", "llm_inference_cost", "search_cost",
                "latency_ms", "web_search_api_calls", "llm_round_trips",
                "sources_returned"]:
        native_val = native_record.get(key, 0)
        ydc_val = ydc_record.get(key, 0)
        if native_val > 0:
            ratios[key] = round(ydc_val / native_val, 3)
        else:
            ratios[key] = None
    return ratios


# ─── Main ────────────────────────────────────────────────────────────────────

def parse_int_list(s: str) -> list[int]:
    """Parse comma-separated integers: '1,2,3' -> [1, 2, 3]."""
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main():
    args = sys.argv[1:]
    if "--verbose" in args:
        os.environ["GROUNDING_VERBOSE"] = "1"
    skip_judge = "--no-judge" in args

    # Parse --tier and --ids flags
    tier_filter = None
    id_filter = None
    cooldown = 5
    for i, a in enumerate(args):
        if a == "--tier" and i + 1 < len(args):
            tier_filter = parse_int_list(args[i + 1])
        elif a == "--ids" and i + 1 < len(args):
            id_filter = parse_int_list(args[i + 1])
        elif a == "--cooldown" and i + 1 < len(args):
            cooldown = int(args[i + 1])

    # Load questions
    QUERIES = load_questions(tier_filter=tier_filter, id_filter=id_filter)
    if not QUERIES:
        print("Error: No questions matched filters. Check --tier or --ids values.")
        sys.exit(1)

    tier_set = sorted(set(q["tier"] for q in QUERIES))

    # Validate keys
    if not OPENAI_API_KEY or OPENAI_API_KEY.startswith("sk-your-"):
        print("Error: OPENAI_API_KEY not set in gpt_comparison.py")
        sys.exit(1)
    if not YDC_API_KEY or YDC_API_KEY.startswith("your-"):
        print("Error: YDC_API_KEY not set in gpt_comparison.py")
        sys.exit(1)

    # Timestamp for output file
    run_ts = datetime.now(timezone.utc)
    ts_str = run_ts.strftime("%Y%m%d_%H%M%S")
    output_path = Path(__file__).parent / f"results_{ts_str}.json"

    client = OpenAI(api_key=OPENAI_API_KEY, timeout=LLM_TIMEOUT)

    print("=" * 90)
    print(f"GPT THREE-WAY COMPARISON — {len(QUERIES)} queries across tiers {tier_set}")
    print("=" * 90)
    filter_desc = "ALL 50 questions"
    if id_filter:
        filter_desc = f"IDs: {id_filter}"
    elif tier_filter:
        filter_desc = f"Tiers: {tier_filter}"
    print(f"  Model:   {MODEL}")
    print(f"  Filter:  {filter_desc}")
    print(f"  Judge:   {'OFF' if skip_judge else 'Claude Sonnet 4.6 (cross-model, blind)'}")
    print(f"  Output:  {output_path.name}")
    print(f"  Started: {run_ts.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print()

    # ── Results container ──
    results = {
        "run_id": ts_str,
        "model": MODEL,
        "started_at": run_ts.isoformat(),
        "pricing": PRICING,
        "skip_judge": skip_judge,
        "filter": filter_desc,
        "question_count": len(QUERIES),
        "tiers": tier_set,
        "cooldown_s": cooldown,
        "queries": [],
    }

    total_start = time.perf_counter()

    for idx, q in enumerate(QUERIES):
        print(f"━━━ [{idx+1}/{len(QUERIES)}] Tier {q['tier']}: {q['tier_name']} ━━━")
        print(f"  Domain: {q['domain']}")
        print(f"  Query:  {q['query']}")
        print()

        entry = {
            "question_id": q["id"],
            "query": q["query"],
            "tier": q["tier"],
            "tier_name": q["tier_name"],
            "domain": q["domain"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # ── Path 1: Native ──
        try:
            print("  Running Native path...")
            native_stats = run_native(q["query"], client)
            entry["native"] = build_run_record(native_stats, "native")
            print(f"  ✓ Native done — {native_stats['total_tokens']:,} tokens, "
                  f"{native_stats['search_calls']} searches, "
                  f"${entry['native']['total_cost']:.5f}, "
                  f"{native_stats['latency_ms']:.0f}ms")
        except Exception as e:
            print(f"  ✗ Native FAILED: {e}")
            entry["native"] = {"error": str(e)}
            native_stats = None

        # ── Path 2: You.com Original ──
        try:
            print("  Running You.com Original path...")
            original_stats = run_youdotcom_original(q["query"], client)
            entry["ydc_original"] = build_run_record(original_stats, "youdotcom")
            print(f"  ✓ YDC Original done — {original_stats['total_tokens']:,} tokens, "
                  f"{original_stats['search_calls']} searches, "
                  f"${entry['ydc_original']['total_cost']:.5f}, "
                  f"{original_stats['latency_ms']:.0f}ms")
        except Exception as e:
            print(f"  ✗ YDC Original FAILED: {e}")
            entry["ydc_original"] = {"error": str(e)}
            original_stats = None

        # ── Path 3: You.com Optimized ──
        try:
            print("  Running You.com Optimized path...")
            optimized_stats = run_youdotcom_optimized(q["query"], client)
            entry["ydc_optimized"] = build_run_record(optimized_stats, "youdotcom")
            print(f"  ✓ YDC Optimized done — {optimized_stats['total_tokens']:,} tokens, "
                  f"{optimized_stats['search_calls']} searches, "
                  f"${entry['ydc_optimized']['total_cost']:.5f}, "
                  f"{optimized_stats['latency_ms']:.0f}ms")
        except Exception as e:
            print(f"  ✗ YDC Optimized FAILED: {e}")
            entry["ydc_optimized"] = {"error": str(e)}
            optimized_stats = None

        # ── Ratios (matching screenshot format) ──
        if native_stats and original_stats:
            entry["ratios_original_vs_native"] = compute_ratios(
                entry["ydc_original"], entry["native"])
        if native_stats and optimized_stats:
            entry["ratios_optimized_vs_native"] = compute_ratios(
                entry["ydc_optimized"], entry["native"])
        if original_stats and optimized_stats:
            entry["ratios_optimized_vs_original"] = compute_ratios(
                entry["ydc_optimized"], entry["ydc_original"])

        # ── Judge ──
        if not skip_judge and native_stats and original_stats and optimized_stats:
            try:
                print("  Running judge (Claude Sonnet 4.6)...")
                judge_result = run_judge(
                    q["query"], native_stats, original_stats, optimized_stats)
                if judge_result.get("error"):
                    print(f"  ⚠ Judge: {judge_result['error']}")
                else:
                    verdict = judge_result.get("verdict", "?")
                    print(f"  ✓ Judge done — Verdict: {verdict}")
                entry["judge"] = judge_result
            except Exception as e:
                print(f"  ✗ Judge FAILED: {e}")
                entry["judge"] = {"error": str(e)}
        elif skip_judge:
            entry["judge"] = {"skipped": True}

        # Strip full answer text from saved output (keep length for reference)
        for path_key in ["native", "ydc_original", "ydc_optimized"]:
            if isinstance(entry.get(path_key), dict) and "answer" in entry[path_key]:
                del entry[path_key]["answer"]

        results["queries"].append(entry)

        # Save after every query (crash-safe)
        _save(results, output_path)

        # Brief cooldown between queries
        if idx < len(QUERIES) - 1:
            print(f"  ⏳ Cooling down {cooldown}s...")
            time.sleep(cooldown)
        print()

    # ── Finalize ──
    total_elapsed = time.perf_counter() - total_start
    results["completed_at"] = datetime.now(timezone.utc).isoformat()
    results["total_elapsed_s"] = round(total_elapsed, 1)

    # ── Aggregate summary ──
    results["summary"] = compute_summary(results["queries"])
    _save(results, output_path)

    # ── Print summary ──
    print_summary(results)
    print(f"\nResults saved to: {output_path}")


def _save(data: dict, path: Path):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=str)
    tmp.replace(path)


def compute_summary(queries: list[dict]) -> dict:
    """Aggregate metrics across all queries."""
    summary = {"query_count": len(queries)}

    for path_key in ["native", "ydc_original", "ydc_optimized"]:
        records = [q[path_key] for q in queries
                   if isinstance(q.get(path_key), dict) and "error" not in q[path_key]]
        if not records:
            continue
        n = len(records)
        summary[path_key] = {
            "queries_completed": n,
            # Totals
            "total_tokens": sum(r["total_tokens"] for r in records),
            "total_input_tokens": sum(r["input_tokens"] for r in records),
            "total_output_tokens": sum(r["output_tokens"] for r in records),
            "total_search_context_tokens": sum(r["search_context_tokens"] for r in records),
            "total_cost": round(sum(r["total_cost"] for r in records), 6),
            "total_llm_cost": round(sum(r["llm_inference_cost"] for r in records), 6),
            "total_search_cost": round(sum(r["search_cost"] for r in records), 6),
            "total_search_calls": sum(r["web_search_api_calls"] for r in records),
            "total_llm_round_trips": sum(r["llm_round_trips"] for r in records),
            "total_sources": sum(r["sources_returned"] for r in records),
            # Averages
            "avg_tokens": round(sum(r["total_tokens"] for r in records) / n),
            "avg_latency_ms": round(sum(r["latency_ms"] for r in records) / n),
            "avg_cost": round(sum(r["total_cost"] for r in records) / n, 6),
            "avg_search_calls": round(sum(r["web_search_api_calls"] for r in records) / n, 1),
            "avg_llm_round_trips": round(sum(r["llm_round_trips"] for r in records) / n, 1),
        }

    # Cross-path ratios
    nat = summary.get("native", {})
    for ydc_key in ["ydc_original", "ydc_optimized"]:
        ydc = summary.get(ydc_key, {})
        if nat.get("total_cost", 0) > 0 and ydc.get("total_cost", 0) > 0:
            summary[f"{ydc_key}_vs_native"] = {
                "token_ratio": round(ydc["total_tokens"] / nat["total_tokens"], 3)
                    if nat.get("total_tokens") else None,
                "cost_ratio": round(ydc["total_cost"] / nat["total_cost"], 3),
                "token_savings_pct": round((1 - ydc["total_tokens"] / nat["total_tokens"]) * 100, 1)
                    if nat.get("total_tokens") else None,
                "cost_savings_pct": round((1 - ydc["total_cost"] / nat["total_cost"]) * 100, 1),
            }

    orig = summary.get("ydc_original", {})
    opt = summary.get("ydc_optimized", {})
    if orig.get("total_cost", 0) > 0 and opt.get("total_cost", 0) > 0:
        summary["optimized_vs_original"] = {
            "token_ratio": round(opt["total_tokens"] / orig["total_tokens"], 3)
                if orig.get("total_tokens") else None,
            "cost_ratio": round(opt["total_cost"] / orig["total_cost"], 3),
            "token_savings_pct": round((1 - opt["total_tokens"] / orig["total_tokens"]) * 100, 1)
                if orig.get("total_tokens") else None,
            "cost_savings_pct": round((1 - opt["total_cost"] / orig["total_cost"]) * 100, 1),
        }

    # Judge summary
    judged = [q for q in queries
              if isinstance(q.get("judge"), dict) and "error" not in q["judge"] and not q["judge"].get("skipped")]
    if judged:
        verdicts = {"native": 0, "original": 0, "optimized": 0, "comparable": 0}
        for q in judged:
            v = q["judge"].get("verdict", "comparable")
            verdicts[v] = verdicts.get(v, 0) + 1
        summary["judge_verdicts"] = verdicts

    return summary


def print_summary(results: dict):
    """Print a human-readable summary table."""
    s = results.get("summary", {})
    if not s:
        print("No summary available.")
        return

    w = 22

    print()
    print("=" * 96)
    print("AGGREGATE SUMMARY")
    print("=" * 96)
    print(f"  Model:    {results['model']}")
    print(f"  Queries:  {s['query_count']}")
    print(f"  Duration: {results.get('total_elapsed_s', 0):.0f}s")
    print()

    # ── Per-path table ──
    nat = s.get("native", {})
    orig = s.get("ydc_original", {})
    opt = s.get("ydc_optimized", {})

    header = f"{'METRIC':<34} {'Native':>{w}} {'YDC Original':>{w}} {'YDC Optimized':>{w}}"
    print(header)
    print(f"{'-'*34} {'-'*w} {'-'*w} {'-'*w}")

    rows = [
        ("Total tokens",
         f"{nat.get('total_tokens', 0):,}",
         f"{orig.get('total_tokens', 0):,}",
         f"{opt.get('total_tokens', 0):,}"),
        ("  Input tokens",
         f"{nat.get('total_input_tokens', 0):,}",
         f"{orig.get('total_input_tokens', 0):,}",
         f"{opt.get('total_input_tokens', 0):,}"),
        ("  Output tokens",
         f"{nat.get('total_output_tokens', 0):,}",
         f"{orig.get('total_output_tokens', 0):,}",
         f"{opt.get('total_output_tokens', 0):,}"),
        ("  Search context (est.)",
         f"~{nat.get('total_search_context_tokens', 0):,}",
         f"~{orig.get('total_search_context_tokens', 0):,}",
         f"~{opt.get('total_search_context_tokens', 0):,}"),
        ("", "", "", ""),
        ("Web Search API calls",
         f"{nat.get('total_search_calls', 0)}",
         f"{orig.get('total_search_calls', 0)}",
         f"{opt.get('total_search_calls', 0)}"),
        ("LLM round-trips",
         f"{nat.get('total_llm_round_trips', 0)}",
         f"{orig.get('total_llm_round_trips', 0)}",
         f"{opt.get('total_llm_round_trips', 0)}"),
        ("Sources returned",
         f"{nat.get('total_sources', 0)}",
         f"{orig.get('total_sources', 0)}",
         f"{opt.get('total_sources', 0)}"),
        ("Avg latency",
         f"{nat.get('avg_latency_ms', 0):,.0f}ms",
         f"{orig.get('avg_latency_ms', 0):,.0f}ms",
         f"{opt.get('avg_latency_ms', 0):,.0f}ms"),
        ("", "", "", ""),
        ("Total cost (all queries)",
         f"${nat.get('total_cost', 0):.5f}",
         f"${orig.get('total_cost', 0):.5f}",
         f"${opt.get('total_cost', 0):.5f}"),
        ("  LLM inference",
         f"${nat.get('total_llm_cost', 0):.5f}",
         f"${orig.get('total_llm_cost', 0):.5f}",
         f"${opt.get('total_llm_cost', 0):.5f}"),
        ("  Search",
         f"${nat.get('total_search_cost', 0):.5f}",
         f"${orig.get('total_search_cost', 0):.5f}",
         f"${opt.get('total_search_cost', 0):.5f}"),
        ("Avg cost per query",
         f"${nat.get('avg_cost', 0):.6f}",
         f"${orig.get('avg_cost', 0):.6f}",
         f"${opt.get('avg_cost', 0):.6f}"),
    ]

    for label, v1, v2, v3 in rows:
        if not label:
            print()
            continue
        print(f"{label:<34} {v1:>{w}} {v2:>{w}} {v3:>{w}}")

    # ── Savings ──
    print(f"\n{'─'*96}")
    print("SAVINGS vs. NATIVE")
    print(f"{'─'*96}")
    for label, key in [("YDC Original", "ydc_original_vs_native"),
                        ("YDC Optimized", "ydc_optimized_vs_native")]:
        r = s.get(key, {})
        tok = r.get("token_savings_pct", 0)
        cost = r.get("cost_savings_pct", 0)
        print(f"  {label:<24} Tokens: {tok:+.1f}%    Cost: {cost:+.1f}%")

    r = s.get("optimized_vs_original", {})
    if r:
        tok = r.get("token_savings_pct", 0)
        cost = r.get("cost_savings_pct", 0)
        print(f"\n  Optimized vs Original:   Tokens: {tok:+.1f}%    Cost: {cost:+.1f}%")

    # ── Judge ──
    verdicts = s.get("judge_verdicts")
    if verdicts:
        print(f"\n{'─'*96}")
        print("JUDGE VERDICTS")
        print(f"{'─'*96}")
        print(f"  Native wins: {verdicts.get('native', 0)}  |  "
              f"YDC Original wins: {verdicts.get('original', 0)}  |  "
              f"YDC Optimized wins: {verdicts.get('optimized', 0)}  |  "
              f"Comparable: {verdicts.get('comparable', 0)}")

    print()


if __name__ == "__main__":
    main()
