#!/usr/bin/env python3
"""
benchmark_lab.py — Experimental testbed for benchmark path comparisons.

Extends benchmark_runner.py with a working chat.completions + YDC path.
Core code files (compare.py, base_agent.py, agents/) are NOT modified here —
all experiments are self-contained in this file.

Paths:
  ydc       — YDC via Responses API (existing, via compare.run_youdotcom)
  responses — Native search via Responses API (existing, via compare.run_native)
  chat      — YDC via chat.completions (fixed here — previous "chat" path was
               sending web_search_preview which is Responses-API-only and invalid
               for chat.completions. Correct path uses function tool schema.)

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

import requests as _requests
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / "grounding/env.txt") or \
load_dotenv(Path(__file__).parent.parent / "grounding/.env")

# Ensure compare.py can be imported from same directory
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "grounding"))

print("[STARTUP] importing compare...", flush=True)
from compare import (
    MODELS,
    run_youdotcom,
    run_native,
    calculate_costs,
    is_verbose,
    describe_native_search,
)
print("[STARTUP] importing judge...", flush=True)
from judge import run_judge
print("[STARTUP] imports done", flush=True)


# ─── YDC via chat.completions ─────────────────────────────────────────────────
# This is the working implementation from GPT comparison/gpt_comparison.py.
# The existing "chat" path in benchmark_runner.py was broken — it sent
# {"type": "web_search_preview"} which is a Responses API tool type and is
# rejected by chat.completions with a 400. The correct tool type is "function".

_SEARCH_ENDPOINT = "https://ydc-index.io/v1/search"
_SEARCH_TIMEOUT = 30
_MAX_TOOL_ROUNDS = 15

_YOUDOTCOM_TOOL_SCHEMA_CHAT = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web for current information using You.com Search API. "
            "Use for real-time data, recent events, current prices, or anything "
            "that may have changed since your training cutoff."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."},
                "count": {"type": "integer", "description": "Results to return (1-20). Default 5.", "default": 5},
                "freshness": {"type": "string", "description": "Recency filter: 'day','week','month','year'."},
            },
            "required": ["query"],
        },
    },
}


def _execute_ydc_search(tool_args: dict) -> str:
    """Execute a You.com search and return formatted results."""
    key = os.environ.get("YDC_API_KEY", "")
    if not key:
        return "Error: YDC_API_KEY not set."
    query = tool_args.get("query", "")
    if not query:
        return "Error: 'query' is required but was empty."
    # Truncate long queries to avoid 414/422 errors
    if len(query) > 200:
        print(f"    [YDC] query truncated: {len(query)} → 200 chars")
        query = query[:200]
    params = {"query": query, "count": tool_args.get("count", 5)}
    if tool_args.get("freshness"):
        params["freshness"] = tool_args["freshness"]
    if tool_args.get("include_domains"):
        params["include_domains"] = tool_args["include_domains"]
    if tool_args.get("exclude_domains"):
        params["exclude_domains"] = tool_args["exclude_domains"]

    # Log full params so we can see exactly what the LLM generated
    print(f"    [YDC] params: {json.dumps(params)}")

    try:
        resp = _requests.get(
            _SEARCH_ENDPOINT,
            headers={"X-API-Key": key},
            params=params,
            timeout=_SEARCH_TIMEOUT,
        )
        if not resp.ok:
            print(f"    [YDC] HTTP {resp.status_code} — url_len={len(resp.request.url)} full_url={resp.request.url[:300]}")
        resp.raise_for_status()
    except _requests.RequestException as e:
        return f"Search failed: {e}"
    data = resp.json()
    results = data.get("results", {}).get("web", [])
    if not results:
        return f"No results found for: {query}"
    parts = []
    for i, hit in enumerate(results, 1):
        snippets = "\n".join(hit.get("snippets", []))
        parts.append(f"[{i}] {hit.get('title','')}\nURL: {hit.get('url','')}\n{snippets}")
    metadata = data.get("metadata", {})
    uuid_str = metadata.get("search_uuid", "")
    header = f"Search results for: {query} ({len(results)} results)"
    if uuid_str:
        header += f"\nSearch-UUID: {uuid_str}"
    return header + "\n\n" + "\n\n---\n\n".join(parts)


def run_youdotcom_instrumented(query: str, model_config: dict) -> dict:
    """YDC via Responses API — instrumented duplicate of base_agent.OpenAIResponsesAgent.

    Identical logic to the production path but with per-round timing, per-search
    query logging, and token counts printed so we can see exactly what GPT-5.5
    is doing on each round.

    Respects model_config["max_rounds"] to cap the tool loop (default: _MAX_TOOL_ROUNDS).
    """
    from search_tool import execute_search, extract_urls, extract_search_uuid
    from concurrent.futures import ThreadPoolExecutor
    import json as _json

    model_id = model_config["model"]
    api_key = os.environ.get("OPENAI_API_KEY", "")
    client = OpenAI(api_key=api_key, timeout=30)

    # Responses API tool schema (must use "you_search" name — "web_search" gets
    # intercepted by OpenAI and routed to its own native search)
    TOOL_SCHEMA = {
        "type": "function",
        "name": "you_search",
        "description": "Search the web using You.com Search API.",
        "parameters": {
            "type": "object",
            "properties": {
                "query":          {"type": "string"},
                "count":          {"type": "integer", "default": 5},
                "freshness":      {"type": "string"},
                "include_domains": {"type": "array", "items": {"type": "string"}},
                "exclude_domains": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["query"],
        },
    }

    from search_tool import get_system_prompt as _gsp
    stats = {
        "path": f"{model_id} + You.com (Responses API instrumented)",
        "model": model_id,
        "model_confirmed": None,
        "total_tokens": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "search_context_tokens": 0,
        "api_calls": 0,
        "search_calls": 0,
        "hit_round_limit": False,
        "sources": [],
        "search_uuid": "",
        "latency_ms": 0.0,
        "answer": "",
        "tool_calls": [],
    }

    max_rounds = model_config.get("max_rounds", _MAX_TOOL_ROUNDS)

    t0 = time.perf_counter()
    previous_response_id = None
    current_input = query
    response = None
    search_num = 0

    for round_num in range(max_rounds):
        round_ts = datetime.now(timezone.utc).isoformat()
        round_t0 = time.perf_counter()
        print(f"    [LLM] round {round_num + 1} start {round_ts}  prev_id={previous_response_id}")

        try:
            response = client.responses.create(
                model=model_id,
                instructions=_gsp(),
                input=current_input,
                tools=[TOOL_SCHEMA],
                previous_response_id=previous_response_id,
            )
        except Exception as e:
            round_elapsed = time.perf_counter() - round_t0
            print(f"    [LLM] round {round_num + 1} ERROR after {round_elapsed:.1f}s: {e}")
            stats["answer"] = f"Error: {e}"
            break

        round_elapsed = time.perf_counter() - round_t0
        round_in  = getattr(response.usage, "input_tokens",  0) if response.usage else 0
        round_out = getattr(response.usage, "output_tokens", 0) if response.usage else 0
        stats["input_tokens"]  += round_in
        stats["output_tokens"] += round_out
        stats["api_calls"] += 1
        if round_num == 0:
            stats["model_confirmed"] = getattr(response, "model", None)
        previous_response_id = response.id

        function_calls = [
            item for item in response.output
            if getattr(item, "type", None) == "function_call"
            and getattr(item, "name", None) == "you_search"
        ]
        print(f"    [LLM] round {round_num + 1} done  {round_elapsed:.1f}s — "
              f"in={round_in} out={round_out} searches_this_round={len(function_calls)}")

        if not function_calls:
            break

        # Log each search query before executing
        indexed_calls = []
        for item in function_calls:
            try:
                args = _json.loads(item.arguments)
            except Exception:
                args = {}
            search_num += 1
            q = args.get("query", "")
            # Truncation intentionally disabled — logging raw query length to observe 414/422 errors
            if len(q) > 200:
                print(f"    [YDC] search {search_num} query length={len(q)} (truncation OFF)")
            print(f"    [YDC] search {search_num} params: {_json.dumps(args)}")
            indexed_calls.append((search_num, item, args))

        def _execute(idx_item):
            idx, item, args = idx_item
            search_t0 = time.perf_counter()
            http_error = None
            try:
                result = execute_search(args)
            except Exception as e:
                result = f"Search failed: {e}"
                http_error = str(e)
            elapsed_ms = (time.perf_counter() - search_t0) * 1000
            # Detect HTTP error codes in result string
            if not http_error:
                for code in ("414", "422", "400", "429", "500"):
                    if f"HTTP {code}" in result or f"status {code}" in result:
                        http_error = f"HTTP {code}"
                        break
            urls = extract_urls(result)
            uuid = extract_search_uuid(result)
            print(f"    [YDC] search {idx} done  {elapsed_ms:.0f}ms — {len(urls)} urls{' ERROR: ' + http_error if http_error else ''}")
            return {
                "search_num": idx,
                "call_id": item.call_id,
                "output": result,
                "sources": urls,
                "uuid": uuid,
                "params": args,
                "elapsed_ms": elapsed_ms,
                "http_error": http_error,
            }

        with ThreadPoolExecutor(max_workers=min(len(indexed_calls), 8)) as executor:
            results = list(executor.map(_execute, indexed_calls))

        tool_outputs = []
        for r in results:
            stats["search_calls"] += 1
            for url in r["sources"]:
                if url not in stats["sources"]:
                    stats["sources"].append(url)
            if r["uuid"]:
                stats["search_uuid"] = r["uuid"]
            stats["tool_calls"].append({
                "search_num": r["search_num"],
                "round": round_num + 1,
                "params": r["params"],
                "query_len": len(r["params"].get("query", "")),
                "result_count": len(r["sources"]),
                "latency_ms": round(r["elapsed_ms"]),
                "http_error": r.get("http_error"),
            })
            tool_outputs.append({
                "type": "function_call_output",
                "call_id": r["call_id"],
                "output": r["output"],
            })

        current_input = tool_outputs
    else:
        stats["hit_round_limit"] = True
        print(f"    [LLM] HIT ROUND LIMIT ({max_rounds} rounds)")

    # Extract answer
    if response:
        message_item = next(
            (item for item in response.output if getattr(item, "type", None) == "message"),
            None,
        )
        if message_item:
            text_item = next(
                (c for c in getattr(message_item, "content", [])
                 if getattr(c, "type", None) == "output_text"),
                None,
            )
            if text_item:
                stats["answer"] = text_item.text

    stats["total_tokens"] = stats["input_tokens"] + stats["output_tokens"]
    stats["latency_ms"] = (time.perf_counter() - t0) * 1000
    total_elapsed = stats["latency_ms"] / 1000
    print(f"    [DONE] {total_elapsed:.1f}s total — "
          f"{stats['total_tokens']:,} tokens, {stats['search_calls']} searches, "
          f"{stats['api_calls']} rounds")
    return stats


def run_youdotcom_chat(query: str, model_config: dict) -> dict:
    """YDC via chat.completions — client-managed message history.

    Uses {"type": "function"} tool schema (valid for chat.completions).
    Contrast with the Responses API path (run_youdotcom) which uses
    {"type": "function"} in the Responses API format with previous_response_id.

    Also includes query truncation to avoid 414/422 from long LLM-generated queries.
    """
    from search_tool import get_system_prompt
    model_id = model_config["model"]
    api_key = os.environ.get("OPENAI_API_KEY", "")
    client = OpenAI(api_key=api_key, timeout=30)

    stats = {
        "path": f"{model_id} + You.com (chat.completions)",
        "model": model_id,
        "model_confirmed": None,
        "total_tokens": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "search_context_tokens": 0,
        "api_calls": 0,
        "search_calls": 0,
        "hit_round_limit": False,
        "sources": [],
        "search_uuid": "",
        "latency_ms": 0.0,
        "answer": "",
        "tool_calls": [],
    }

    messages = [
        {"role": "system", "content": get_system_prompt()},
        {"role": "user", "content": query},
    ]
    t0 = time.perf_counter()
    baseline_input = 0
    response = None

    for round_num in range(_MAX_TOOL_ROUNDS):
        round_ts = datetime.now(timezone.utc).isoformat()
        round_t0 = time.perf_counter()
        print(f"    [LLM] round {round_num+1} start {round_ts}")
        response = client.chat.completions.create(
            model=model_id,
            messages=messages,
            tools=[_YOUDOTCOM_TOOL_SCHEMA_CHAT],
            max_completion_tokens=4096,
        )
        round_elapsed = time.perf_counter() - round_t0
        print(f"    [LLM] round {round_num+1} done  {round_elapsed:.1f}s — finish_reason={response.choices[0].finish_reason}")

        usage = response.usage
        call_input = getattr(usage, "prompt_tokens", 0) if usage else 0
        call_output = getattr(usage, "completion_tokens", 0) if usage else 0
        stats["input_tokens"] += call_input
        stats["output_tokens"] += call_output
        stats["api_calls"] += 1
        if round_num == 0:
            baseline_input = call_input
            stats["model_confirmed"] = response.model

        choice = response.choices[0]
        if choice.finish_reason != "tool_calls" or not choice.message.tool_calls:
            break

        messages.append(choice.message)
        for tool_call in choice.message.tool_calls:
            if tool_call.function.name != "web_search":
                continue
            args = json.loads(tool_call.function.arguments)
            result = _execute_ydc_search(args)
            stats["search_calls"] += 1
            for line in result.split("\n"):
                if line.startswith("URL: "):
                    url = line[5:].strip()
                    if url not in stats["sources"]:
                        stats["sources"].append(url)
                elif line.startswith("Search-UUID: "):
                    stats["search_uuid"] = line[len("Search-UUID: "):].strip()
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })
    else:
        stats["hit_round_limit"] = True

    stats["latency_ms"] = (time.perf_counter() - t0) * 1000
    stats["total_tokens"] = stats["input_tokens"] + stats["output_tokens"]
    if stats["api_calls"] > 1:
        stats["search_context_tokens"] = max(0, stats["input_tokens"] - baseline_input * stats["api_calls"])
    if response and response.choices:
        stats["answer"] = response.choices[0].message.content or ""
    return stats


# ─── Rate limit retry + wall-clock timeout ────────────────────────────────────

MAX_RETRIES = 5
INITIAL_BACKOFF = 20  # seconds — Tier 2 has ~300k ITPM, light backoff suffices
PATH_TIMEOUT_S = 90   # wall-clock timeout per path attempt — catches infinite hangs

def _run_with_retry(fn, *args, **kwargs):
    """Run fn with exponential backoff on 429s and a wall-clock timeout per attempt."""
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
    for attempt in range(MAX_RETRIES):
        try:
            with ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(fn, *args, **kwargs)
                return future.result(timeout=PATH_TIMEOUT_S)
        except FuturesTimeout:
            print(f"  ⚠️  Path timed out after {PATH_TIMEOUT_S}s (attempt {attempt + 1}/{MAX_RETRIES})")
            raise RuntimeError(f"Path timed out after {PATH_TIMEOUT_S}s")
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                wait = INITIAL_BACKOFF * (1.5 ** attempt)
                print(f"  ⏳ Rate limited — waiting {wait:.0f}s before retry "
                      f"({attempt + 1}/{MAX_RETRIES})...")
                time.sleep(wait)
            else:
                raise
    # Final attempt — let it raise
    with ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(fn, *args, **kwargs)
        return future.result(timeout=PATH_TIMEOUT_S)


# ─── Constants ────────────────────────────────────────────────────────────────

QUESTIONS_FILE = Path(__file__).parent / "benchmark_questions_50.json"

VALID_PATHS = {"ydc", "responses", "chat"}
PATH_ALIASES = {"native": "responses"}  # "native" is an alias for "responses"

def _normalize_paths(paths_arg: str) -> list[str]:
    """Parse and validate the --paths argument. Returns a deduplicated list."""
    raw = [p.strip().lower() for p in paths_arg.split(",") if p.strip()]
    resolved = [PATH_ALIASES.get(p, p) for p in raw]
    seen, ordered = set(), []
    for p in resolved:
        if p not in VALID_PATHS:
            raise ValueError(f"Unknown path: {p!r}. Valid options: {', '.join(sorted(VALID_PATHS))} (or 'native' as alias for 'responses')")
        if p not in seen:
            seen.add(p)
            ordered.append(p)
    return ordered


def _default_output_path(provider: str = "claude", paths: list[str] = None) -> Path:
    """Generate a timestamped output filename including model name and paths."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = MODELS[provider]["model"].replace(".", "_")  # e.g. gpt-5.4 → gpt-5_4
    paths_tag = "_".join(paths or ["ydc", "responses"])
    return Path(__file__).parent / f"benchmark_results_{model_name}_{paths_tag}_{ts}.json"


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
        "tool_calls": stats.get("tool_calls", []),
        # We store the answer for judge use but it's large — store it
        "answer": stats.get("answer", ""),
    }


def compute_cost(stats: dict, model_config: dict, path_type: str) -> dict:
    """Compute cost breakdown for a single path result.

    Delegates to compare.py's calculate_costs() — single source of truth for
    all cost math including livecrawl overage.
    Re-keys the result to benchmark's _cost convention for output files.
    """
    c = calculate_costs(stats, model_config, path=path_type)
    return {
        "llm_cost": round(c["llm"], 6),
        "search_cost": round(c["search"], 6),
        "livecrawl_overage": round(c["livecrawl_overage"], 6),
        "total_cost": round(c["total"], 6),
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
    offset: int = 0,
    force_chat_completions: bool = False,
    paths: list[str] = None,
    fresh: bool = False,
):
    """Run the 50-question benchmark for the specified paths.

    Args:
        paths: Which search paths to benchmark. Any combination of:
               "ydc"       — LLM + You.com Search
               "responses" — LLM + native search via Responses API  (alias: "native")
               "chat"      — LLM + native search via chat.completions
               Default: ["ydc", "responses"] (backward compatible).
        fresh: Ignore any existing partial run and start clean.
        force_chat_completions: Deprecated — use paths=["ydc","chat"] instead.
                                Retained for backward compatibility.
    """
    if paths is None:
        # Backward compat: --chat-completions used to mean chat path for YDC loop,
        # not for native. Keep old behavior (just changes YDC's API shape, not paths).
        paths = ["ydc", "responses"]
    if output_path is None:
        output_path = _default_output_path(provider, paths)

    if verbose:
        os.environ["GROUNDING_VERBOSE"] = "1"

    model_config = dict(MODELS[provider])  # copy — don't mutate shared config
    # GPT-5.5 benchmark_lab override: cap rounds to control cost/latency
    _MAX_ROUNDS_OVERRIDE = {"gpt5.5": 6}
    if provider in _MAX_ROUNDS_OVERRIDE:
        model_config["max_rounds"] = _MAX_ROUNDS_OVERRIDE[provider]
    questions = load_questions(QUESTIONS_FILE)
    results = load_existing_results(output_path)

    # Find or create a run
    active_run = None if fresh else find_active_run(results)

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
            "paths": paths,
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

    # Apply --offset then --limit
    if offset > 0:
        remaining = remaining[offset:]
        print(f"  Skipping first {offset} questions (starting at Q{offset + 1})")
    if limit > 0 and len(remaining) > limit:
        remaining = remaining[:limit]
        print(f"  Limiting to {limit} questions")

    print(f"  Provider: {provider} ({model_config['model']})")
    print(f"  Paths:    {', '.join(paths)}")
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

    # Save initial state
    save_results(results, output_path)

    # Query log — one JSON line per path per query, written immediately on completion.
    # Survives crashes and is readable mid-run with: tail -f query_log_*.jsonl
    log_path = output_path.with_suffix(".jsonl")
    print(f"  Query log: {log_path.name}\n")

    def _log(entry: dict):
        with open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

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

        # ── Run each requested path ──
        path_stats = {}  # path_key → stats dict (or None on failure)

        for path_key in paths:
            ts_start = datetime.now(timezone.utc).isoformat()
            try:
                if path_key == "ydc":
                    label = "You.com (Responses API)"
                    print(f"  Running {label} path...")
                    t0 = time.perf_counter()
                    stats = _run_with_retry(run_youdotcom_instrumented, question["query"], model_config)
                    elapsed = time.perf_counter() - t0
                    cost_type = "ydc"
                elif path_key == "chat":
                    label = "You.com (chat.completions)"
                    print(f"  Running {label} path...")
                    t0 = time.perf_counter()
                    stats = _run_with_retry(run_youdotcom_chat, question["query"], model_config)
                    elapsed = time.perf_counter() - t0
                    cost_type = "ydc"
                else:
                    label = f"Native ({path_key})"
                    print(f"  Running {label} path...")
                    t0 = time.perf_counter()
                    stats = _run_with_retry(run_native, question["query"], None, model_config,
                                           native_path=path_key)
                    elapsed = time.perf_counter() - t0
                    cost_type = path_key

                ts_end = datetime.now(timezone.utc).isoformat()
                print(f"  ✓ {label} done — {stats['total_tokens']:,} tokens, "
                      f"{stats['search_calls']} searches, {elapsed:.1f}s")
                record = stats_to_record(stats)
                record["cost"] = compute_cost(stats, model_config, cost_type)
                record["ts_start"] = ts_start
                record["ts_end"] = ts_end
                record["ok"] = True
                result_entry[path_key] = record
                path_stats[path_key] = stats
                _log({"ts_start": ts_start, "ts_end": ts_end, "ok": True,
                      "question_id": qid, "domain": question["domain"],
                      "path": path_key, "model": model_config["model"],
                      "latency_ms": round(stats.get("latency_ms", 0)),
                      "search_calls": stats.get("search_calls", 0),
                      "api_calls": stats.get("api_calls", 0),
                      "total_tokens": stats.get("total_tokens", 0),
                      "error": None})

            except Exception as e:
                ts_end = datetime.now(timezone.utc).isoformat()
                print(f"  ✗ {path_key} FAILED: {e}")
                if verbose:
                    traceback.print_exc()
                result_entry[path_key] = {
                    "error": str(e),
                    "ts_start": ts_start,
                    "ts_end": ts_end,
                    "ok": False,
                }
                _log({"ts_start": ts_start, "ts_end": ts_end, "ok": False,
                      "question_id": qid, "domain": question["domain"],
                      "path": path_key, "model": model_config["model"],
                      "latency_ms": None, "search_calls": None,
                      "api_calls": None, "total_tokens": None,
                      "error": str(e)})
                if result_entry["status"] != "partial_error":
                    result_entry["status"] = "partial_error"
                path_stats[path_key] = None

        # ── Judge: ydc vs each other path ──
        ydc_stats = path_stats.get("ydc")
        judge_results = {}

        if not skip_judge and ydc_stats:
            judge_model = model_config.get("judge")
            if judge_model:
                for path_key in paths:
                    if path_key == "ydc":
                        continue
                    other_stats = path_stats.get(path_key)
                    if not other_stats:
                        judge_results[f"ydc_vs_{path_key}"] = {"error": f"{path_key} path failed — skipped"}
                        continue
                    try:
                        print(f"  Running judge ({judge_model}): ydc vs {path_key}...")
                        t0 = time.perf_counter()
                        judge_result = _run_with_retry(
                            run_judge,
                            question["query"],
                            ydc_stats["answer"],
                            other_stats["answer"],
                            judge_model,
                            sources_ydc=ydc_stats.get("sources", []),
                            sources_native=other_stats.get("sources", []),
                        )
                        elapsed = time.perf_counter() - t0
                        if judge_result.get("error"):
                            print(f"  ⚠ Judge error: {judge_result['error']}")
                        else:
                            verdict = judge_result.get("verdict", "?")
                            ydc_score = sum(judge_result.get("ydc_scores", {}).get(d, 0)
                                            for d in ["completeness", "relevance", "specificity", "citation_quality"])
                            nat_score = sum(judge_result.get("native_scores", {}).get(d, 0)
                                            for d in ["completeness", "relevance", "specificity", "citation_quality"])
                            print(f"  ✓ Judge done — YDC: {ydc_score}/20, {path_key}: {nat_score}/20, "
                                  f"Verdict: {verdict} ({elapsed:.1f}s)")
                        judge_results[f"ydc_vs_{path_key}"] = judge_result
                    except Exception as e:
                        print(f"  ✗ Judge (ydc vs {path_key}) FAILED: {e}")
                        if verbose:
                            traceback.print_exc()
                        judge_results[f"ydc_vs_{path_key}"] = {"error": str(e)}

        if skip_judge:
            result_entry["judge"] = {"skipped": True}
        elif not judge_results:
            result_entry["judge"] = {"error": "Skipped — ydc path not run or no judge configured"}
        else:
            result_entry["judge"] = judge_results

        # Mark status
        if result_entry["status"] == "pending":
            result_entry["status"] = "completed"
        if "error" in result_entry.get("ydc", {}) and "error" in result_entry.get("native", {}):
            result_entry["status"] = "failed"
            errors += 1

        # Compute per-query savings (ydc vs each other path)
        ydc_cost_val = result_entry.get("ydc", {}).get("cost", {}).get("total_cost", 0)
        ydc_tok = result_entry.get("ydc", {}).get("total_tokens", 0)
        for path_key in paths:
            if path_key == "ydc":
                continue
            nat_cost_val = result_entry.get(path_key, {}).get("cost", {}).get("total_cost", 0)
            nat_tok = result_entry.get(path_key, {}).get("total_tokens", 0)
            result_entry.setdefault("savings", {})[f"ydc_vs_{path_key}"] = {
                "cost_saved_pct": round((1 - ydc_cost_val / nat_cost_val) * 100, 1) if nat_cost_val > 0 else 0,
                "token_saved_pct": round((1 - ydc_tok / nat_tok) * 100, 1) if nat_tok > 0 else 0,
            }

        # Strip full answers from saved output to keep file manageable
        if not keep_answers:
            for path_key in paths:
                if isinstance(result_entry.get(path_key), dict) and "answer" in result_entry[path_key]:
                    del result_entry[path_key]["answer"]

        active_run["results"].append(result_entry)

        # Save after every query (crash-safe)
        save_results(results, output_path)

        # Cooldown between queries to avoid rate limits
        # Tier 2: ~300k input tokens/min → lightweight proportional wait
        total_tokens_this_query = (
            result_entry.get("ydc", {}).get("input_tokens", 0) +
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

    # Discover which paths were actually run from the result data
    all_path_keys = []
    for r in completed:
        for k in r:
            if isinstance(r[k], dict) and "total_tokens" in r[k] and k not in all_path_keys:
                all_path_keys.append(k)
    summary["paths"] = all_path_keys

    # Aggregate per path
    for path_key in all_path_keys:
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
        verdicts = {"ydc": 0, "native": 0, "comparable": 0}

        for r in judged:
            j = r["judge"]
            for d in dims:
                ydc_totals[d] += j.get("ydc_scores", {}).get(d, 0)
                nat_totals[d] += j.get("native_scores", {}).get(d, 0)
            v = j.get("verdict", "comparable")
            verdicts[v] = verdicts.get(v, 0) + 1

        n = len(judged)
        summary["judge"] = {
            "evaluated_count": n,
            "ydc_avg_scores": {d: round(ydc_totals[d] / n, 2) for d in dims},
            "native_avg_scores": {d: round(nat_totals[d] / n, 2) for d in dims},
            "ydc_avg_total": round(sum(ydc_totals.values()) / n, 2),
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

        for path_key in all_path_keys:
            path_results = [r[path_key] for r in tier_results if isinstance(r.get(path_key), dict) and "error" not in r[path_key]]
            if path_results:
                tier_summary[f"{path_key}_avg_tokens"] = round(sum(r["total_tokens"] for r in path_results) / len(path_results))
                tier_summary[f"{path_key}_avg_latency_ms"] = round(sum(r["latency_ms"] for r in path_results) / len(path_results))
                tier_summary[f"{path_key}_avg_cost"] = round(sum(r.get("cost", {}).get("total_cost", 0) for r in path_results) / len(path_results), 6)

        summary["by_tier"][str(tier_num)] = tier_summary

    # Top-level headline numbers (for the one-pager)
    ydc_s = summary.get("ydc", {})
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
    path_labels = {"ydc": "YDC (Responses)", "responses": "Native (Responses)", "chat": "YDC (Chat CC)"}
    active_paths = s.get("paths", ["ydc", "responses"])
    w = max(18, 60 // len(active_paths))

    header = f"{'METRIC':<35}"
    divider = f"{'-'*35}"
    for p in active_paths:
        label = path_labels.get(p, p)
        header += f" {label:>{w}}"
        divider += f" {'-'*w}"
    print(header)
    print(divider)

    def _pv(path_key, field, fmt="s"):
        val = s.get(path_key, {}).get(field, 0)
        if fmt == ",": return f"{val:,}"
        if fmt == "ms": return f"{val:,}ms"
        if fmt == "$4": return f"${val:.4f}"
        if fmt == "$6": return f"${val:.6f}"
        return str(val)

    metric_rows = [
        ("Avg total tokens",       "avg_tokens",           ","),
        ("Avg input tokens",       "avg_input_tokens",     ","),
        ("Avg output tokens",      "avg_output_tokens",    ","),
        ("Avg search calls",       "avg_search_calls",     "s"),
        ("Avg LLM round-trips",    "avg_api_calls",        "s"),
        ("Avg sources",            "avg_sources",          "s"),
        ("Avg latency",            "avg_latency_ms",       "ms"),
        ("", "", ""),
        ("Total cost (all queries)","total_cost",          "$4"),
        ("  LLM inference",        "total_llm_cost",       "$4"),
        ("  Search",               "total_search_cost",    "$4"),
        ("Avg cost per query",     "avg_cost",             "$6"),
        ("Hit round limit",        "hit_round_limit_count","s"),
    ]

    for label, field, fmt in metric_rows:
        if not label:
            print()
            continue
        row = f"{label:<35}"
        for p in active_paths:
            row += f" {_pv(p, field, fmt):>{w}}"
        print(row)

    # ── Cost savings (cheapest vs most expensive path) ──
    path_costs = {p: s.get(p, {}).get("total_cost", 0) for p in active_paths if s.get(p, {}).get("total_cost", 0) > 0}
    if len(path_costs) > 1:
        cheapest = min(path_costs, key=path_costs.get)
        priciest = max(path_costs, key=path_costs.get)
        savings = path_costs[priciest] - path_costs[cheapest]
        pct = (savings / path_costs[priciest]) * 100
        print(f"\n  Cheapest path: {path_labels.get(cheapest, cheapest)} (${path_costs[cheapest]:.4f})")
        print(f"  vs priciest:   {path_labels.get(priciest, priciest)} (${path_costs[priciest]:.4f}) — {pct:.0f}% more expensive")

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
            y = j.get("ydc_avg_scores", {}).get(d, 0)
            n = j.get("native_avg_scores", {}).get(d, 0)
            print(f"  {dim_labels[d]:<22} {y:>11.2f}/5 {n:>11.2f}/5")

        yt = j.get("ydc_avg_total", 0)
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
        print("PER-TIER BREAKDOWN (avg tokens / avg latency / avg cost per query)")
        print(f"{'='*78}")
        tw = 18
        tier_header = f"  {'Tier':<20} {'Count':>5}"
        for p in active_paths:
            lbl = path_labels.get(p, p)[:tw]
            tier_header += f"  {lbl:>{tw}}"
        print(tier_header)
        for tier_num in sorted(by_tier.keys()):
            t = by_tier[tier_num]
            name = f"T{tier_num}: {t['name']}"
            row = f"  {name:<20} {t['count']:>5}"
            for p in active_paths:
                tok = t.get(f"{p}_avg_tokens", 0)
                lat = t.get(f"{p}_avg_latency_ms", 0)
                cost = t.get(f"{p}_avg_cost", 0)
                cell = f"{tok:,}t/{lat/1000:.1f}s/${cost:.4f}"
                row += f"  {cell:>{tw}}"
            print(row)

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
    parser.add_argument("--offset", type=int, default=0,
                        help="Skip first N questions (0 = start from beginning)")
    parser.add_argument("--keep-answers", action="store_true",
                        help="Keep full answer text in output file (larger file)")
    parser.add_argument("--paths", default="ydc,responses",
                        help="Comma-separated paths to benchmark. Options: ydc, responses (or native), chat. "
                             "Default: ydc,responses. Example: --paths ydc,responses,chat")
    parser.add_argument("--fresh", action="store_true",
                        help="Ignore any existing partial run and start clean.")
    parser.add_argument("--chat-completions", action="store_true",
                        help="Deprecated. Use --paths ydc,chat instead. "
                             "Changes YDC loop API shape for GPT/Qwen — does not add chat.completions native path.")

    args = parser.parse_args()
    force_cc = getattr(args, "chat_completions", False)

    try:
        paths = _normalize_paths(args.paths)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    output_path = args.output or _default_output_path(args.provider, paths)

    run_benchmark(
        provider=args.provider,
        output_path=output_path,
        skip_judge=args.no_judge,
        verbose=args.verbose,
        dry_run=args.dry_run,
        keep_answers=args.keep_answers,
        limit=args.limit,
        offset=args.offset,
        force_chat_completions=force_cc,
        paths=paths,
        fresh=args.fresh,
    )


if __name__ == "__main__":
    main()
