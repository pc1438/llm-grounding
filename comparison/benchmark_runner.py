#!/usr/bin/env python3
"""
benchmark_runner.py — Run the 50-question benchmark suite, comparing
LLM + You.com Search vs. LLM + Native Web Search with judge evaluation.

Features:
  • Reads questions from benchmark_questions_50.json
  • Per-query sequential: You.com → Native → Judge
  • Captures search_calls, api_calls, and all cost/token metrics
  • Resume from failure: reads existing output file, skips completed queries
  • Multiple runs: if all 50 complete, appends a new run with unique ID
  • Saves results after each query (crash-safe)

Usage:
    python benchmark_runner.py                  # Run all 50, default provider=claude
    python benchmark_runner.py --provider openai
    python benchmark_runner.py --no-judge       # Skip judge evaluation
    python benchmark_runner.py --verbose        # Verbose logging
    python benchmark_runner.py --dry-run        # Show plan without running
    python benchmark_runner.py --keep-answers   # Keep full answer text in output
    python benchmark_runner.py --output results.json  # Custom output file

Requirements:
    pip install anthropic openai requests python-dotenv
"""

import argparse
import json
import os
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Ensure compare.py can be imported from same directory
sys.path.insert(0, str(Path(__file__).parent))

from compare import (
    MODELS,
    _create_client,
    _empty_stats,
    run_youdotcom,
    run_native,
    run_judge,
)

# Also need dotenv and verbose setup from compare's env loading
from compare import is_verbose, describe_native_search


# ─── Rate limit retry ─────────────────────────────────────────────────────────

MAX_RETRIES = 5
INITIAL_BACKOFF = 20  # seconds — Tier 2 has ~300k ITPM, light backoff suffices

def _run_with_retry(fn, *args, **kwargs):
    """Run a function with exponential backoff on 429 rate limit errors."""
    for attempt in range(MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                wait = INITIAL_BACKOFF * (1.5 ** attempt)
                print(f"  ⏳ Rate limited — waiting {wait:.0f}s before retry "
                      f"({attempt + 1}/{MAX_RETRIES})...")
                time.sleep(wait)
            else:
                raise
    # Final attempt — let it raise
    return fn(*args, **kwargs)


# ─── Constants ────────────────────────────────────────────────────────────────

QUESTIONS_FILE = Path(__file__).parent / "benchmark_questions_50.json"

def _default_output_path(provider: str = "claude", force_chat_completions: bool = False) -> Path:
    """Generate a timestamped output filename including model name and API path."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = MODELS[provider]["model"].replace(".", "_")  # e.g. gpt-5.4 → gpt-5_4
    # Include _chat_completions_ in filename when using the legacy path so
    # benchmark output files for the two API paths don't collide.
    suffix = "_chat_completions_" if force_chat_completions else "_"
    return Path(__file__).parent / f"benchmark_results_{model_name}{suffix}{ts}.json"


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_questions(path: Path) -> list[dict]:
    """Load benchmark questions from JSON file, flattening all tiers."""
    with open(path) as f:
        data = json.load(f)

    questions = []
    for tier in data["tiers"]:
        for q in tier["questions"]:
            questions.append({
                "id": q["id"],
                "query": q["query"],
                "domain": q["domain"],
                "tier": tier["tier"],
                "tier_name": tier["name"],
            })
    return questions


def load_existing_results(path: Path) -> dict:
    """Load existing results file, or return empty structure."""
    if not path.exists():
        return {"runs": []}
    try:
        with open(path) as f:
            data = json.load(f)
        # Ensure it has the right structure
        if "runs" not in data:
            data = {"runs": []}
        return data
    except (json.JSONDecodeError, IOError):
        print(f"  Warning: Could not parse {path}, starting fresh.")
        return {"runs": []}


def find_active_run(results: dict, run_id: str = None) -> dict | None:
    """Find an incomplete run to resume, or a specific run by ID."""
    for run in results["runs"]:
        if run_id and run["run_id"] == run_id:
            return run
        if not run_id and run["status"] == "in_progress":
            return run
    return None


def get_completed_ids(run: dict) -> set[int]:
    """Get the set of question IDs already completed in a run."""
    return {r["question_id"] for r in run.get("results", []) if r.get("status") == "completed"}


def save_results(results: dict, path: Path):
    """Atomically save results to JSON file."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2, default=str)
    tmp.replace(path)


def stats_to_record(stats: dict) -> dict:
    """Convert a stats dict to a JSON-serializable record for output."""
    return {
        "path": stats["path"],
        "model": stats["model"],
        "total_tokens": stats["total_tokens"],
        "input_tokens": stats["input_tokens"],
        "output_tokens": stats["output_tokens"],
        "search_context_tokens": stats["search_context_tokens"],
        "api_calls": stats["api_calls"],
        "search_calls": stats["search_calls"],
        "hit_round_limit": stats["hit_round_limit"],
        "source_count": len(stats.get("sources", [])),
        "sources": stats.get("sources", []),
        "latency_ms": round(stats["latency_ms"], 1),
        "answer_length": len(stats.get("answer", "")),
        # We store the answer for judge use but it's large — store it
        "answer": stats.get("answer", ""),
    }


def compute_cost(stats: dict, model_config: dict, path_type: str) -> dict:
    """Compute cost breakdown for a single path result.

    NOTE: compare.py has calculate_costs() / calculate_native_costs() covering
    similar logic but split by path type and using different output key names
    ("llm"/"search"/"total" vs "llm_cost"/"search_cost"/"total_cost"). The
    functions are intentionally separate; unify them if the key naming is ever
    aligned across both modules.
    """
    llm_cost = (
        stats["input_tokens"] * model_config["input_cost_per_m"] / 1_000_000 +
        stats["output_tokens"] * model_config["output_cost_per_m"] / 1_000_000
    )
    if path_type == "youdotcom":
        search_cost = stats.get("search_calls", 0) * (model_config.get("ydc_search_cost_per_call") or 0)
    else:
        search_cost = stats.get("search_calls", 0) * (model_config.get("native_search_cost_per_call") or 0)

    return {
        "llm_cost": round(llm_cost, 6),
        "search_cost": round(search_cost, 6),
        "total_cost": round(llm_cost + search_cost, 6),
    }


# ─── Main benchmark logic ────────────────────────────────────────────────────

def run_benchmark(
    provider: str = "claude",
    output_path: Path = None,
    skip_judge: bool = False,
    verbose: bool = False,
    dry_run: bool = False,
    keep_answers: bool = False,
    limit: int = 0,
    force_chat_completions: bool = False,
):
    """Run the full 50-question benchmark.

    Args:
        force_chat_completions: Pass to run_youdotcom() to use the legacy
                                chat.completions path for GPT/Qwen instead of
                                the default Responses API. Output filename will
                                include '_chat_completions_' suffix so results
                                from both API paths can coexist in the same dir.
                                Has no effect on claude or kimi.
                                CLI flag: --chat-completions
    """
    if output_path is None:
        output_path = _default_output_path(provider, force_chat_completions)

    if verbose:
        os.environ["GROUNDING_VERBOSE"] = "1"

    model_config = MODELS[provider]
    questions = load_questions(QUESTIONS_FILE)
    results = load_existing_results(output_path)

    # Find or create a run
    active_run = find_active_run(results)

    if active_run:
        run_id = active_run["run_id"]
        completed = get_completed_ids(active_run)
        remaining = [q for q in questions if q["id"] not in completed]
        print(f"Resuming run {run_id}")
        print(f"  {len(completed)}/{len(questions)} already completed, {len(remaining)} remaining")
    else:
        # All previous runs are complete (or no runs exist) — start a new one
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        active_run = {
            "run_id": run_id,
            "provider": provider,
            "model": model_config["model"],
            "native_search_config": describe_native_search(model_config),
            "judge_model": model_config["judge"] if not skip_judge else None,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "status": "in_progress",
            "total_questions": len(questions),
            "skip_judge": skip_judge,
            "force_chat_completions": force_chat_completions,
            "results": [],
        }
        results["runs"].append(active_run)
        remaining = questions
        completed = set()
        print(f"Starting new run {run_id}")
        print(f"  {len(questions)} questions to run")

    # Apply --limit
    if limit > 0 and len(remaining) > limit:
        remaining = remaining[:limit]
        print(f"  Limiting to first {limit} questions")

    print(f"  Provider: {provider} ({model_config['model']})")
    print(f"  Native:   {describe_native_search(model_config)}")
    print(f"  Judge:    {'OFF' if skip_judge else model_config['judge']}")
    print(f"  Output:   {output_path}")
    print()

    if dry_run:
        print("DRY RUN — would process these questions:")
        for q in remaining:
            print(f"  [{q['id']:2d}] (Tier {q['tier']}: {q['tier_name']}) [{q['domain']}] {q['query']}")
        return

    # Create API client
    ydc_key = os.environ.get("YDC_API_KEY", "")
    if not ydc_key:
        print("Error: YDC_API_KEY not set.")
        sys.exit(1)

    client = _create_client(model_config)

    # Save initial state
    save_results(results, output_path)

    total_start = time.perf_counter()
    errors = 0

    for idx, question in enumerate(remaining):
        qid = question["id"]
        qnum = len(completed) + idx + 1
        total = len(questions)

        print(f"━━━ [{qnum}/{total}] Q{qid} (Tier {question['tier']}: {question['tier_name']}) ━━━")
        print(f"  Domain: {question['domain']}")
        print(f"  Query:  {question['query']}")

        result_entry = {
            "question_id": qid,
            "query": question["query"],
            "domain": question["domain"],
            "tier": question["tier"],
            "tier_name": question["tier_name"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "pending",
        }

        # ── You.com path ──
        try:
            print(f"  Running You.com path...")
            t0 = time.perf_counter()
            ydc_stats = _run_with_retry(run_youdotcom, question["query"], client, model_config,
                                       force_chat_completions=force_chat_completions)
            elapsed = time.perf_counter() - t0
            print(f"  ✓ You.com done — {ydc_stats['total_tokens']:,} tokens, "
                  f"{ydc_stats['search_calls']} searches, {elapsed:.1f}s")

            result_entry["youdotcom"] = stats_to_record(ydc_stats)
            result_entry["youdotcom"]["cost"] = compute_cost(ydc_stats, model_config, "youdotcom")
        except Exception as e:
            print(f"  ✗ You.com FAILED: {e}")
            if verbose:
                traceback.print_exc()
            result_entry["youdotcom"] = {"error": str(e)}
            result_entry["status"] = "partial_error"
            ydc_stats = None

        # ── Native path ──
        try:
            print(f"  Running Native path...")
            t0 = time.perf_counter()
            native_stats = _run_with_retry(run_native, question["query"], client, model_config)
            elapsed = time.perf_counter() - t0
            print(f"  ✓ Native done — {native_stats['total_tokens']:,} tokens, "
                  f"{native_stats['search_calls']} searches, {elapsed:.1f}s")

            result_entry["native"] = stats_to_record(native_stats)
            result_entry["native"]["cost"] = compute_cost(native_stats, model_config, "native")
        except Exception as e:
            print(f"  ✗ Native FAILED: {e}")
            if verbose:
                traceback.print_exc()
            result_entry["native"] = {"error": str(e)}
            if result_entry["status"] != "partial_error":
                result_entry["status"] = "partial_error"
            native_stats = None

        # ── Judge ──
        if not skip_judge and ydc_stats and native_stats:
            try:
                judge_model = model_config["judge"]
                print(f"  Running judge ({judge_model})...")
                t0 = time.perf_counter()
                judge_result = _run_with_retry(
                    run_judge,
                    question["query"],
                    ydc_stats["answer"],
                    native_stats["answer"],
                    judge_model,
                    sources_ydc=ydc_stats.get("sources", []),
                    sources_native=native_stats.get("sources", []),
                )
                elapsed = time.perf_counter() - t0

                if judge_result.get("error"):
                    print(f"  ⚠ Judge error: {judge_result['error']}")
                    result_entry["judge"] = judge_result
                else:
                    verdict = judge_result.get("verdict", "?")
                    ydc_score = sum(judge_result.get("youdotcom_scores", {}).get(d, 0)
                                    for d in ["completeness", "relevance", "specificity", "citation_quality"])
                    nat_score = sum(judge_result.get("native_scores", {}).get(d, 0)
                                    for d in ["completeness", "relevance", "specificity", "citation_quality"])
                    print(f"  ✓ Judge done — You.com: {ydc_score}/20, Native: {nat_score}/20, "
                          f"Verdict: {verdict} ({elapsed:.1f}s)")
                    result_entry["judge"] = judge_result
            except Exception as e:
                print(f"  ✗ Judge FAILED: {e}")
                if verbose:
                    traceback.print_exc()
                result_entry["judge"] = {"error": str(e)}
        elif skip_judge:
            result_entry["judge"] = {"skipped": True}
        else:
            result_entry["judge"] = {"error": "Skipped — one or both paths failed"}

        # Mark status
        if result_entry["status"] == "pending":
            result_entry["status"] = "completed"
        if "error" in result_entry.get("youdotcom", {}) and "error" in result_entry.get("native", {}):
            result_entry["status"] = "failed"
            errors += 1

        # Compute per-query cost_saved_pct for the one-pager table
        ydc_cost_val = result_entry.get("youdotcom", {}).get("cost", {}).get("total_cost", 0)
        nat_cost_val = result_entry.get("native", {}).get("cost", {}).get("total_cost", 0)
        if nat_cost_val > 0:
            result_entry["cost_saved_pct"] = round((1 - ydc_cost_val / nat_cost_val) * 100, 1)
        else:
            result_entry["cost_saved_pct"] = 0

        # Compute per-query token_saved_pct
        ydc_tok = result_entry.get("youdotcom", {}).get("total_tokens", 0)
        nat_tok = result_entry.get("native", {}).get("total_tokens", 0)
        if nat_tok > 0:
            result_entry["token_saved_pct"] = round((1 - ydc_tok / nat_tok) * 100, 1)
        else:
            result_entry["token_saved_pct"] = 0

        # Strip full answers from saved output to keep file manageable
        # (keep answer_length for reference; use --keep-answers to retain)
        if not keep_answers:
            for path_key in ["youdotcom", "native"]:
                if isinstance(result_entry.get(path_key), dict) and "answer" in result_entry[path_key]:
                    del result_entry[path_key]["answer"]

        active_run["results"].append(result_entry)

        # Save after every query (crash-safe)
        save_results(results, output_path)

        # Cooldown between queries to avoid rate limits
        # Tier 2: ~300k input tokens/min → lightweight proportional wait
        total_tokens_this_query = (
            result_entry.get("youdotcom", {}).get("input_tokens", 0) +
            result_entry.get("native", {}).get("input_tokens", 0)
        )
        cooldown = max(5, (total_tokens_this_query / 300000) * 65)
        if idx < len(remaining) - 1:  # Don't wait after the last query
            print(f"  ⏳ Cooling down {cooldown:.0f}s...")
            time.sleep(cooldown)
        print()

    # ── Finalize run ──
    total_elapsed = time.perf_counter() - total_start
    completed_count = sum(1 for r in active_run["results"] if r["status"] == "completed")
    active_run["completed_at"] = datetime.now(timezone.utc).isoformat()
    active_run["status"] = "completed" if completed_count == len(questions) else "partial"
    active_run["total_elapsed_s"] = round(total_elapsed, 1)

    # ── Aggregate summary ──
    summary = compute_summary(active_run)
    active_run["summary"] = summary

    save_results(results, output_path)

    # Print final summary
    print_summary(active_run, model_config, output_path)


def compute_summary(run: dict) -> dict:
    """Compute aggregate statistics for a completed run."""
    completed = [r for r in run["results"] if r["status"] == "completed"]
    if not completed:
        return {"error": "No completed queries"}

    summary = {
        "total_queries": len(run["results"]),
        "completed": len(completed),
        "failed": sum(1 for r in run["results"] if r["status"] in ("failed", "partial_error")),
    }

    # Aggregate per path
    for path_key in ["youdotcom", "native"]:
        path_results = [r[path_key] for r in completed if isinstance(r.get(path_key), dict) and "error" not in r[path_key]]
        if not path_results:
            continue

        summary[path_key] = {
            "avg_tokens": round(sum(r["total_tokens"] for r in path_results) / len(path_results)),
            "avg_input_tokens": round(sum(r["input_tokens"] for r in path_results) / len(path_results)),
            "avg_output_tokens": round(sum(r["output_tokens"] for r in path_results) / len(path_results)),
            "avg_search_context_tokens": round(sum(r.get("search_context_tokens", 0) for r in path_results) / len(path_results)),
            "avg_search_calls": round(sum(r["search_calls"] for r in path_results) / len(path_results), 1),
            "total_search_calls": sum(r["search_calls"] for r in path_results),
            "avg_api_calls": round(sum(r["api_calls"] for r in path_results) / len(path_results), 1),
            "avg_latency_ms": round(sum(r["latency_ms"] for r in path_results) / len(path_results)),
            "avg_sources": round(sum(r["source_count"] for r in path_results) / len(path_results), 1),
            "total_cost": round(sum(r.get("cost", {}).get("total_cost", 0) for r in path_results), 4),
            "avg_cost": round(sum(r.get("cost", {}).get("total_cost", 0) for r in path_results) / len(path_results), 6),
            "total_search_cost": round(sum(r.get("cost", {}).get("search_cost", 0) for r in path_results), 4),
            "total_llm_cost": round(sum(r.get("cost", {}).get("llm_cost", 0) for r in path_results), 4),
            "hit_round_limit_count": sum(1 for r in path_results if r.get("hit_round_limit")),
        }

    # Judge aggregates
    judged = [r for r in completed if isinstance(r.get("judge"), dict) and "error" not in r["judge"] and not r["judge"].get("skipped")]
    if judged:
        dims = ["completeness", "relevance", "specificity", "citation_quality"]
        ydc_totals = {d: 0 for d in dims}
        nat_totals = {d: 0 for d in dims}
        verdicts = {"youdotcom": 0, "native": 0, "comparable": 0}

        for r in judged:
            j = r["judge"]
            for d in dims:
                ydc_totals[d] += j.get("youdotcom_scores", {}).get(d, 0)
                nat_totals[d] += j.get("native_scores", {}).get(d, 0)
            v = j.get("verdict", "comparable")
            verdicts[v] = verdicts.get(v, 0) + 1

        n = len(judged)
        summary["judge"] = {
            "evaluated_count": n,
            "youdotcom_avg_scores": {d: round(ydc_totals[d] / n, 2) for d in dims},
            "native_avg_scores": {d: round(nat_totals[d] / n, 2) for d in dims},
            "youdotcom_avg_total": round(sum(ydc_totals.values()) / n, 2),
            "native_avg_total": round(sum(nat_totals.values()) / n, 2),
            "verdicts": verdicts,
        }

    # Per-tier breakdown
    tier_groups = {}
    for r in completed:
        t = r.get("tier", 0)
        tier_groups.setdefault(t, []).append(r)

    summary["by_tier"] = {}
    for tier_num, tier_results in sorted(tier_groups.items()):
        tier_name = tier_results[0].get("tier_name", f"Tier {tier_num}")
        tier_summary = {"name": tier_name, "count": len(tier_results)}

        for path_key in ["youdotcom", "native"]:
            path_results = [r[path_key] for r in tier_results if isinstance(r.get(path_key), dict) and "error" not in r[path_key]]
            if path_results:
                tier_summary[f"{path_key}_avg_tokens"] = round(sum(r["total_tokens"] for r in path_results) / len(path_results))
                tier_summary[f"{path_key}_avg_latency_ms"] = round(sum(r["latency_ms"] for r in path_results) / len(path_results))
                tier_summary[f"{path_key}_avg_cost"] = round(sum(r.get("cost", {}).get("total_cost", 0) for r in path_results) / len(path_results), 6)

        summary["by_tier"][str(tier_num)] = tier_summary

    # Top-level headline numbers (for the one-pager)
    ydc_s = summary.get("youdotcom", {})
    nat_s = summary.get("native", {})
    if nat_s.get("total_cost", 0) > 0:
        summary["cost_savings_pct"] = round((1 - ydc_s.get("total_cost", 0) / nat_s["total_cost"]) * 100, 1)
    if nat_s.get("avg_tokens", 0) > 0:
        summary["token_savings_pct"] = round((1 - ydc_s.get("avg_tokens", 0) / nat_s["avg_tokens"]) * 100, 1)
    j = summary.get("judge", {})
    if j:
        v = j.get("verdicts", {})
        summary["quality_wins"] = f"{v.get('youdotcom', 0)}-{v.get('native', 0)}-{v.get('comparable', 0)}"

    return summary


def print_summary(run: dict, model_config: dict, output_path: Path = None):
    """Print a human-readable summary of the benchmark run."""
    s = run.get("summary", {})
    if not s or "error" in s:
        print("No summary available.")
        return

    print("=" * 78)
    print("BENCHMARK SUMMARY")
    print("=" * 78)
    print(f"  Run ID:     {run['run_id']}")
    print(f"  Model:      {run['model']}")
    print(f"  Provider:   {run['provider']}")
    print(f"  Native:     {run.get('native_search_config', 'N/A')}")
    print(f"  Questions:  {s['completed']}/{s['total_queries']} completed, {s['failed']} failed")
    print(f"  Duration:   {run.get('total_elapsed_s', 0):.0f}s")
    print()

    # ── Per-path summary ──
    w = 28
    print(f"{'METRIC':<35} {'You.com':>{w}} {'Native':>{w}}")
    print(f"{'-'*35} {'-'*w} {'-'*w}")

    ydc = s.get("youdotcom", {})
    nat = s.get("native", {})

    rows = [
        ("Avg total tokens", f"{ydc.get('avg_tokens', 0):,}", f"{nat.get('avg_tokens', 0):,}"),
        ("Avg input tokens", f"{ydc.get('avg_input_tokens', 0):,}", f"{nat.get('avg_input_tokens', 0):,}"),
        ("Avg output tokens", f"{ydc.get('avg_output_tokens', 0):,}", f"{nat.get('avg_output_tokens', 0):,}"),
        ("Avg search calls", f"{ydc.get('avg_search_calls', 0)}", f"{nat.get('avg_search_calls', 0)}"),
        ("Avg LLM round-trips", f"{ydc.get('avg_api_calls', 0)}", f"{nat.get('avg_api_calls', 0)}"),
        ("Avg sources", f"{ydc.get('avg_sources', 0)}", f"{nat.get('avg_sources', 0)}"),
        ("Avg latency", f"{ydc.get('avg_latency_ms', 0):,}ms", f"{nat.get('avg_latency_ms', 0):,}ms"),
        ("", "", ""),
        ("Total cost (all queries)", f"${ydc.get('total_cost', 0):.4f}", f"${nat.get('total_cost', 0):.4f}"),
        ("  LLM inference", f"${ydc.get('total_llm_cost', 0):.4f}", f"${nat.get('total_llm_cost', 0):.4f}"),
        ("  Search", f"${ydc.get('total_search_cost', 0):.4f}", f"${nat.get('total_search_cost', 0):.4f}"),
        ("Avg cost per query", f"${ydc.get('avg_cost', 0):.6f}", f"${nat.get('avg_cost', 0):.6f}"),
        ("Hit round limit", f"{ydc.get('hit_round_limit_count', 0)}", f"{nat.get('hit_round_limit_count', 0)}"),
    ]

    for label, ydc_val, nat_val in rows:
        if not label:
            print()
            continue
        print(f"{label:<35} {ydc_val:>{w}} {nat_val:>{w}}")

    # ── Cost savings ──
    ydc_total = ydc.get("total_cost", 0)
    nat_total = nat.get("total_cost", 0)
    if nat_total > 0:
        savings = nat_total - ydc_total
        pct = (savings / nat_total) * 100
        if savings > 0:
            print(f"\n  Total cost savings with You.com: ${savings:.4f} ({pct:.0f}% cheaper)")

    # ── Judge summary ──
    j = s.get("judge", {})
    if j:
        print(f"\n{'='*78}")
        print("QUALITY EVALUATION (judge summary)")
        print(f"{'='*78}")
        print(f"  Evaluated: {j['evaluated_count']} queries")
        print()

        dims = ["completeness", "relevance", "specificity", "citation_quality"]
        dim_labels = {"completeness": "Completeness", "relevance": "Relevance",
                      "specificity": "Specificity", "citation_quality": "Citation Quality"}

        print(f"  {'Dimension':<22} {'You.com avg':>14} {'Native avg':>14}")
        print(f"  {'-'*22} {'-'*14} {'-'*14}")
        for d in dims:
            y = j.get("youdotcom_avg_scores", {}).get(d, 0)
            n = j.get("native_avg_scores", {}).get(d, 0)
            print(f"  {dim_labels[d]:<22} {y:>11.2f}/5 {n:>11.2f}/5")

        yt = j.get("youdotcom_avg_total", 0)
        nt = j.get("native_avg_total", 0)
        print(f"  {'-'*22} {'-'*14} {'-'*14}")
        print(f"  {'Overall':<22} {yt:>10.2f}/20 {nt:>10.2f}/20")

        verdicts = j.get("verdicts", {})
        print(f"\n  Verdicts: You.com wins: {verdicts.get('youdotcom', 0)}, "
              f"Native wins: {verdicts.get('native', 0)}, "
              f"Comparable: {verdicts.get('comparable', 0)}")

    # ── Per-tier breakdown ──
    by_tier = s.get("by_tier", {})
    if by_tier:
        print(f"\n{'='*78}")
        print("PER-TIER BREAKDOWN")
        print(f"{'='*78}")
        print(f"  {'Tier':<25} {'Count':>6} {'YDC tokens':>12} {'Nat tokens':>12} {'YDC cost':>10} {'Nat cost':>10}")
        print(f"  {'-'*25} {'-'*6} {'-'*12} {'-'*12} {'-'*10} {'-'*10}")
        for tier_num in sorted(by_tier.keys()):
            t = by_tier[tier_num]
            name = f"T{tier_num}: {t['name']}"
            print(f"  {name:<25} {t['count']:>6} "
                  f"{t.get('youdotcom_avg_tokens', 0):>12,} "
                  f"{t.get('native_avg_tokens', 0):>12,} "
                  f"${t.get('youdotcom_avg_cost', 0):>8.5f} "
                  f"${t.get('native_avg_cost', 0):>8.5f}")

    print()
    if output_path:
        print(f"Results saved to: {output_path}")
    print()


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run the 50-question You.com vs Native Search benchmark"
    )
    parser.add_argument("--provider", default="claude", choices=list(MODELS.keys()),
                        help="LLM provider to benchmark (default: claude)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output JSON file path (default: timestamped file in comparison/)")
    parser.add_argument("--no-judge", action="store_true",
                        help="Skip judge evaluation")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show plan without running queries")
    parser.add_argument("--limit", type=int, default=0,
                        help="Only run first N questions (0 = all)")
    parser.add_argument("--keep-answers", action="store_true",
                        help="Keep full answer text in output file (larger file)")
    parser.add_argument("--chat-completions", action="store_true",
                        help="Force legacy chat.completions path for GPT/Qwen (default: Responses API). "
                             "Output filename will include '_chat_completions_' suffix.")

    args = parser.parse_args()
    force_cc = getattr(args, "chat_completions", False)
    output_path = args.output or _default_output_path(args.provider, force_cc)

    run_benchmark(
        provider=args.provider,
        output_path=output_path,
        skip_judge=args.no_judge,
        verbose=args.verbose,
        dry_run=args.dry_run,
        keep_answers=args.keep_answers,
        limit=args.limit,
        force_chat_completions=force_cc,
    )


if __name__ == "__main__":
    main()
