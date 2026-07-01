#!/usr/bin/env python3
"""
gpt_comparison.py — Three-way GPT + Web Search comparison.

Runs the same query through three paths and compares cost, tokens, and quality:

  1. Native:         GPT + OpenAI's built-in web_search (Responses API, black box)
  2. You.com Original: GPT + You.com Search via chat.completions (client-side state)
  3. You.com Optimized: GPT + You.com Search via Responses API (server-side state,
                        previous_response_id — no context bloat)

Path 3 is the key addition: it uses the Responses API's previous_response_id to
chain multi-turn tool loops without re-sending the growing message history. This
avoids context bloat while keeping the rich tool schema that lets GPT control
search parameters (livecrawl, freshness, domain filters, count).

The judge is Claude (cross-model, blind A/B/C evaluation).

Usage:
    python gpt_comparison.py "What happened in tech news today?"
    python gpt_comparison.py --verbose "latest NVIDIA earnings"
    python gpt_comparison.py --no-judge "current S&P 500 price"
    python gpt_comparison.py --show-answers "Who won the 2026 NBA Finals?"

Requirements:
    pip install openai requests

    For judge evaluation (optional, skip with --no-judge):
    pip install anthropic
"""

import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from openai import OpenAI

# ─── API Keys ────────────────────────────────────────────────────────────────
# WARNING: Do not commit this file to git with real keys.
# Replace the placeholders below with your actual keys.

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
YDC_API_KEY = os.environ.get("YDC_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")  # Only needed for judge (--no-judge to skip)

# ─── Configuration ───────────────────────────────────────────────────────────

MODEL = "gpt-5.4"

PRICING = {
    "input_cost_per_m": 2.5,           # $2.50 per 1M input tokens
    "output_cost_per_m": 15.0,          # $15 per 1M output tokens
    "ydc_search_cost_per_call": 0.005,  # You.com: $5 per 1,000 queries
    "native_search_cost_per_call": 0.01,# OpenAI: $10/1k + search content at model rates
}

SEARCH_ENDPOINT = "https://ydc-index.io/v1/search"
SEARCH_TIMEOUT = 30
MAX_LIVECRAWL_CHARS = 3000
MAX_TOKENS = 4096
MAX_TOOL_ROUNDS = 15
LLM_TIMEOUT = 300


def _verbose():
    return os.environ.get("GROUNDING_VERBOSE", "0") == "1"


# ─── System prompt ───────────────────────────────────────────────────────────

def get_system_prompt() -> str:
    now = datetime.now(timezone.utc).strftime("%A, %B %d, %Y at %H:%M UTC")
    return (
        "You are a helpful research assistant with access to real-time web search. "
        "When the user asks a question that requires current information, recent "
        "events, specific data, or anything that might have changed since your "
        "training cutoff, use your web search tool to look it up. You can search "
        "multiple times if you need to compare information or dig deeper.\n\n"
        "When you use search results in your answer, cite your sources by number "
        "[1], [2], etc. If the search results don't contain what you need, say so "
        "honestly rather than guessing.\n\n"
        f"Current date and time: {now}"
    )


# ─── Tool schemas ────────────────────────────────────────────────────────────

# Rich schema — used by both You.com paths (Original and Optimized).
# The LLM can control search depth, freshness, and domain filtering per query.
YOUDOTCOM_TOOL_SCHEMA_OPENAI = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web for current information using You.com Search API. "
            "Use this tool when you need real-time data, recent events, current "
            "prices, live statistics, or any information that may have changed "
            "since your training cutoff. Returns relevant web results with "
            "titles, URLs, and content snippets."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query. Be specific and descriptive.",
                },
                "count": {
                    "type": "integer",
                    "description": "Number of results to return (1-20). Default 5.",
                    "default": 5,
                },
                "livecrawl": {
                    "type": "boolean",
                    "description": (
                        "If true, fetch full page content as markdown (more tokens, "
                        "deeper context). If false, return snippets only (fewer tokens, "
                        "cheaper inference). Use true for complex analysis, false for "
                        "quick factual lookups."
                    ),
                    "default": False,
                },
                "freshness": {
                    "type": "string",
                    "description": (
                        "Recency filter: 'day', 'week', 'month', 'year', or a date "
                        "range like '2025-01-01to2025-06-30'. Omit for no filter."
                    ),
                },
                "include_domains": {
                    "type": "string",
                    "description": "Comma-separated list of domains to restrict results to.",
                },
                "exclude_domains": {
                    "type": "string",
                    "description": "Comma-separated list of domains to exclude from results.",
                },
            },
            "required": ["query"],
        },
    },
}

# Same schema but in Responses API format (no "function" wrapper).
# Used by Path 3 (Optimized) which calls client.responses.create().
YOUDOTCOM_TOOL_SCHEMA_RESPONSES = {
    "type": "function",
    "name": "web_search",
    "description": YOUDOTCOM_TOOL_SCHEMA_OPENAI["function"]["description"],
    "parameters": YOUDOTCOM_TOOL_SCHEMA_OPENAI["function"]["parameters"],
}

# Native search tool config for OpenAI Responses API.
NATIVE_SEARCH_TOOL = {
    "type": "web_search",
    "search_context_size": "medium",
}


# ─── You.com search execution ───────────────────────────────────────────────

def execute_search(tool_args: dict) -> str:
    """Execute a You.com search and return formatted results.

    Called when the LLM invokes the web_search tool. Reads all parameters
    from the LLM's tool call — the model controls count, livecrawl,
    freshness, and domain filters.
    """
    import requests

    key = YDC_API_KEY
    if not key or key.startswith("your-"):
        return "Error: YDC_API_KEY not set. Replace the placeholder at the top of this file."

    query = tool_args.get("query", "")
    if not query:
        return "Error: 'query' is required but was empty."

    livecrawl = tool_args.get("livecrawl", False)
    params = {"query": query, "count": tool_args.get("count", 5)}
    if livecrawl:
        params["livecrawl"] = "web"
        params["livecrawl_formats"] = "markdown"
    if tool_args.get("freshness"):
        params["freshness"] = tool_args["freshness"]
    if tool_args.get("include_domains"):
        params["include_domains"] = tool_args["include_domains"]
    if tool_args.get("exclude_domains"):
        params["exclude_domains"] = tool_args["exclude_domains"]

    try:
        resp = requests.get(
            SEARCH_ENDPOINT,
            headers={"X-API-Key": key},
            params=params,
            timeout=SEARCH_TIMEOUT,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        return f"Search failed: {e}"

    data = resp.json()
    results = data.get("results", {}).get("web", [])
    if not results:
        return f"No results found for: {query}"

    parts = []
    for i, hit in enumerate(results, 1):
        title = hit.get("title", "")
        url = hit.get("url", "")
        snippets = "\n".join(hit.get("snippets", []))
        content = snippets
        if livecrawl and hit.get("contents"):
            content = (
                hit["contents"].get("markdown")
                or hit["contents"].get("html")
                or snippets
            )
            content = content[:MAX_LIVECRAWL_CHARS]
        parts.append(f"[{i}] {title}\nURL: {url}\n{content}")

    metadata = data.get("metadata", {})
    latency = metadata.get("latency", "?")
    search_uuid = metadata.get("search_uuid", "")
    header = f"Search results for: {query} ({len(results)} results, {latency}s)"
    if search_uuid:
        header += f"\nSearch-UUID: {search_uuid}"
    return header + "\n\n" + "\n\n---\n\n".join(parts)


def extract_urls(search_result: str) -> list[str]:
    return [line[5:].strip() for line in search_result.split("\n") if line.startswith("URL: ")]


def extract_search_uuid(search_result: str) -> str:
    for line in search_result.split("\n"):
        if line.startswith("Search-UUID: "):
            return line[len("Search-UUID: "):].strip()
    return ""


# ─── Stats helper ────────────────────────────────────────────────────────────

def _empty_stats(path_label: str = "") -> dict:
    return {
        "path": path_label,
        "model": MODEL,
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
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PATH 1: NATIVE — GPT + OpenAI's built-in web_search (Responses API)
# ═══════════════════════════════════════════════════════════════════════════════

def run_native(question: str, client: OpenAI, on_progress=None) -> dict:
    """GPT + OpenAI native web search. Single API call, provider-controlled search."""
    prompt = get_system_prompt()
    stats = _empty_stats(f"{MODEL} + Native Web Search")
    verbose = _verbose()

    def _notify(msg):
        if verbose:
            print(f"  [Native] {msg}")
        if on_progress:
            on_progress(msg)

    t0 = time.perf_counter()
    _notify("Starting request...")

    response = client.responses.create(
        model=MODEL,
        instructions=prompt,
        tools=[NATIVE_SEARCH_TOOL],
        input=question,
    )

    stats["latency_ms"] = (time.perf_counter() - t0) * 1000
    stats["api_calls"] = 1

    if hasattr(response, "usage") and response.usage:
        stats["input_tokens"] = getattr(response.usage, "input_tokens", 0)
        stats["output_tokens"] = getattr(response.usage, "output_tokens", 0)
    stats["total_tokens"] = stats["input_tokens"] + stats["output_tokens"]

    for item in response.output:
        item_type = getattr(item, "type", "")
        if item_type == "web_search_call":
            stats["search_calls"] += 1
            _notify(f"Search {stats['search_calls']}: \"{getattr(item, 'query', '')}\"")
        elif item_type == "message":
            for part in getattr(item, "content", []):
                if getattr(part, "type", "") == "output_text":
                    stats["answer"] += getattr(part, "text", "")
                    for ann in getattr(part, "annotations", []):
                        if getattr(ann, "type", "") == "url_citation":
                            url = getattr(ann, "url", "")
                            if url and url not in stats["sources"]:
                                stats["sources"].append(url)

    if stats["search_calls"] > 0 and stats["input_tokens"] > 200:
        stats["search_context_tokens"] = stats["input_tokens"] - 200

    _notify(f"Done — {stats['total_tokens']:,} tokens, {stats['latency_ms']:.0f}ms")
    return stats


# ═══════════════════════════════════════════════════════════════════════════════
# PATH 2: YOU.COM ORIGINAL — GPT + You.com via chat.completions (client-side state)
# ═══════════════════════════════════════════════════════════════════════════════

def run_youdotcom_original(question: str, client: OpenAI, on_progress=None) -> dict:
    """GPT + You.com Search using chat.completions with client-managed message history.

    This is the 'original' approach from compare.py. The full conversation
    (system + user + assistant + tool results) is re-sent each round, so
    context grows with each tool loop iteration.
    """
    prompt = get_system_prompt()
    stats = _empty_stats(f"{MODEL} + You.com (client-side state)")
    verbose = _verbose()

    def _notify(msg):
        if verbose:
            print(f"  [YDC-Original] {msg}")
        if on_progress:
            on_progress(msg)

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": question},
    ]
    t0 = time.perf_counter()
    baseline_input = 0

    for round_num in range(MAX_TOOL_ROUNDS):
        _notify(f"LLM round {round_num + 1}...")

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=[YOUDOTCOM_TOOL_SCHEMA_OPENAI],
            max_completion_tokens=MAX_TOKENS,
        )

        if response.usage:
            call_input = getattr(response.usage, "prompt_tokens", 0)
            call_output = getattr(response.usage, "completion_tokens", 0)
        else:
            call_input = call_output = 0

        stats["input_tokens"] += call_input
        stats["output_tokens"] += call_output
        stats["api_calls"] += 1

        if round_num == 0:
            baseline_input = call_input

        choice = response.choices[0]

        if choice.finish_reason != "tool_calls" or not choice.message.tool_calls:
            break

        messages.append(choice.message)

        for tool_call in choice.message.tool_calls:
            if tool_call.function.name != "web_search":
                continue

            args = json.loads(tool_call.function.arguments)
            _notify(f"Search {stats['search_calls'] + 1}: \"{args.get('query', '')}\"")

            search_result = execute_search(args)
            stats["search_calls"] += 1
            stats["sources"].extend(extract_urls(search_result))
            uuid = extract_search_uuid(search_result)
            if uuid:
                stats["search_uuid"] = uuid

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": search_result,
            })
    else:
        stats["hit_round_limit"] = True

    stats["latency_ms"] = (time.perf_counter() - t0) * 1000
    stats["total_tokens"] = stats["input_tokens"] + stats["output_tokens"]

    if stats["api_calls"] > 1:
        stats["search_context_tokens"] = max(0, stats["input_tokens"] - baseline_input * stats["api_calls"])

    stats["answer"] = response.choices[0].message.content or ""
    _notify(f"Done — {stats['total_tokens']:,} tokens, {stats['latency_ms']:.0f}ms")
    return stats


# ═══════════════════════════════════════════════════════════════════════════════
# PATH 3: YOU.COM OPTIMIZED — GPT + You.com via Responses API (server-side state)
# ═══════════════════════════════════════════════════════════════════════════════

def run_youdotcom_optimized(question: str, client: OpenAI, on_progress=None) -> dict:
    """GPT + You.com Search using Responses API with previous_response_id.

    This is the key new path. Instead of accumulating the full message
    history client-side and re-sending it each round (context bloat),
    it uses OpenAI's Responses API which maintains server-side state.

    Each round, the client only sends back tool call outputs + the
    previous_response_id. The server reconstructs the full context
    internally. This keeps wire traffic constant regardless of how
    many search rounds the LLM runs.

    Inspired by the YouToolSampler pattern (Approach 1), but with
    the rich 6-parameter tool schema from Approach 2 — best of both.
    """
    prompt = get_system_prompt()
    stats = _empty_stats(f"{MODEL} + You.com (server-side state)")
    verbose = _verbose()

    def _notify(msg):
        if verbose:
            print(f"  [YDC-Optimized] {msg}")
        if on_progress:
            on_progress(msg)

    previous_response_id = None
    current_input = question
    t0 = time.perf_counter()

    for round_num in range(MAX_TOOL_ROUNDS):
        _notify(f"LLM round {round_num + 1}...")

        kwargs = {
            "model": MODEL,
            "input": current_input,
            "tools": [YOUDOTCOM_TOOL_SCHEMA_RESPONSES],
            "tool_choice": "auto",
            "instructions": prompt,
        }
        if previous_response_id:
            kwargs["previous_response_id"] = previous_response_id

        response = client.responses.create(**kwargs)
        previous_response_id = response.id

        # Track tokens
        if hasattr(response, "usage") and response.usage:
            stats["input_tokens"] += getattr(response.usage, "input_tokens", 0)
            stats["output_tokens"] += getattr(response.usage, "output_tokens", 0)
        stats["api_calls"] += 1

        # Check for function calls
        function_calls = [
            item for item in response.output
            if getattr(item, "type", None) == "function_call"
            and getattr(item, "name", None) == "web_search"
        ]

        if not function_calls:
            break

        # Execute tool calls (parallel if multiple)
        def _execute_one(fc):
            args = json.loads(fc.arguments)
            _notify(f"Search {stats['search_calls'] + 1}: \"{args.get('query', '')}\"")
            search_result = execute_search(args)
            stats["search_calls"] += 1
            stats["sources"].extend(extract_urls(search_result))
            uuid = extract_search_uuid(search_result)
            if uuid:
                stats["search_uuid"] = uuid
            return {
                "type": "function_call_output",
                "call_id": fc.call_id,
                "output": search_result,
            }

        if len(function_calls) == 1:
            current_input = [_execute_one(function_calls[0])]
        else:
            with ThreadPoolExecutor(max_workers=len(function_calls)) as executor:
                futures = {executor.submit(_execute_one, fc): fc for fc in function_calls}
                current_input = [f.result() for f in as_completed(futures)]
    else:
        stats["hit_round_limit"] = True

    stats["latency_ms"] = (time.perf_counter() - t0) * 1000
    stats["total_tokens"] = stats["input_tokens"] + stats["output_tokens"]

    # Estimate search context tokens from cumulative input growth
    if stats["api_calls"] > 1 and stats["input_tokens"] > 200:
        # Rough estimate: total input minus baseline * rounds
        avg_per_round = stats["input_tokens"] / stats["api_calls"]
        stats["search_context_tokens"] = max(0, stats["input_tokens"] - int(avg_per_round))

    # Extract answer text from the final response
    for item in response.output:
        if getattr(item, "type", "") == "message":
            for part in getattr(item, "content", []):
                if getattr(part, "type", "") == "output_text":
                    stats["answer"] += getattr(part, "text", "")

    _notify(f"Done — {stats['total_tokens']:,} tokens, {stats['latency_ms']:.0f}ms")
    return stats


# ═══════════════════════════════════════════════════════════════════════════════
# JUDGE — Claude evaluates all three answers (blind, randomized positions)
# ═══════════════════════════════════════════════════════════════════════════════

JUDGE_SYSTEM_PROMPT = (
    "You are an impartial evaluation judge. You will receive a question and three "
    "answers (labeled Answer A, Answer B, and Answer C) from different search-grounded "
    "AI systems. You do NOT know which system produced which answer.\n\n"
    "IMPORTANT: You CANNOT verify factual claims. Do NOT attempt to judge whether "
    "specific facts are true or false based on your own knowledge — your training "
    "data may be outdated. Instead, evaluate the STRUCTURE and QUALITY of each "
    "answer relative to the others.\n\n"
    "Evaluate each answer on four dimensions using a 1-5 scale:\n"
    "  - Completeness: Does the answer fully address all parts of the question?\n"
    "  - Relevance: Is the information on-topic and useful for the question asked?\n"
    "  - Specificity: Does the answer provide concrete details (dates, names, numbers)?\n"
    "  - Citation Quality: Are sources cited, numbered, and traceable?\n\n"
    "Scoring rubric:\n"
    "  5 = Excellent  4 = Good  3 = Adequate  2 = Poor  1 = Very poor\n\n"
    "After scoring all three, provide a verdict: which answer is best, or are they "
    "comparable?\n\n"
    "Respond ONLY with valid JSON in this exact format:\n"
    "{\n"
    '  "answer_a": {\n'
    '    "completeness": <1-5>, "relevance": <1-5>,\n'
    '    "specificity": <1-5>, "citation_quality": <1-5>,\n'
    '    "reasoning": "<1-2 sentence justification>"\n'
    "  },\n"
    '  "answer_b": { ... same structure ... },\n'
    '  "answer_c": { ... same structure ... },\n'
    '  "verdict": "<A_best | B_best | C_best | comparable>",\n'
    '  "verdict_reasoning": "<1 sentence explaining the overall verdict>"\n'
    "}"
)


def _answer_with_sources(answer: str, sources: list[str]) -> str:
    if not sources:
        return answer
    source_block = "\n\nSources:\n" + "\n".join(
        f"[{i}] {url}" for i, url in enumerate(sources, 1)
    )
    return answer + source_block


def run_judge(
    question: str,
    stats_native: dict,
    stats_original: dict,
    stats_optimized: dict,
) -> dict:
    """Run a blind 3-way evaluation using Claude as judge.

    Randomly assigns answers to A/B/C positions to prevent position bias.
    """
    from anthropic import Anthropic

    anthropic_key = ANTHROPIC_API_KEY
    if not anthropic_key or anthropic_key.startswith("your-"):
        return {"error": "ANTHROPIC_API_KEY not set — skipping judge evaluation"}

    # Prepare answers with sources appended
    answers = {
        "native": _answer_with_sources(stats_native["answer"], stats_native.get("sources", [])),
        "original": _answer_with_sources(stats_original["answer"], stats_original.get("sources", [])),
        "optimized": _answer_with_sources(stats_optimized["answer"], stats_optimized.get("sources", [])),
    }

    # Randomize positions
    keys = list(answers.keys())
    random.shuffle(keys)
    position_map = {"a": keys[0], "b": keys[1], "c": keys[2]}

    judge_prompt = (
        f"Question: {question}\n\n"
        f"--- Answer A ---\n{answers[keys[0]]}\n\n"
        f"--- Answer B ---\n{answers[keys[1]]}\n\n"
        f"--- Answer C ---\n{answers[keys[2]]}\n\n"
        f"Evaluate all three answers. Respond with JSON only."
    )

    judge_client = Anthropic(api_key=anthropic_key, timeout=120)
    response = judge_client.messages.create(
        model="claude-sonnet-4-6",
        system=JUDGE_SYSTEM_PROMPT,
        max_tokens=1500,
        messages=[{"role": "user", "content": judge_prompt}],
    )

    raw = ""
    for block in response.content:
        if hasattr(block, "text"):
            raw += block.text

    # Parse
    try:
        clean = raw.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[1]
            clean = clean.rsplit("```", 1)[0]
        scores = json.loads(clean)
    except (json.JSONDecodeError, IndexError):
        return {"error": f"Judge returned invalid JSON: {raw[:200]}"}

    # Map positions back to path names
    result = {
        "position_map": position_map,
        "judge_model": "claude-sonnet-4-6",
    }

    for pos, path_name in position_map.items():
        pos_key = f"answer_{pos}"
        if pos_key in scores:
            result[f"{path_name}_scores"] = scores[pos_key]

    # Map verdict
    raw_verdict = scores.get("verdict", "comparable")
    if raw_verdict == "A_best":
        result["verdict"] = position_map["a"]
    elif raw_verdict == "B_best":
        result["verdict"] = position_map["b"]
    elif raw_verdict == "C_best":
        result["verdict"] = position_map["c"]
    else:
        result["verdict"] = "comparable"

    result["verdict_reasoning"] = scores.get("verdict_reasoning", "")

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT — 3-way comparison table
# ═══════════════════════════════════════════════════════════════════════════════

def compute_cost(stats: dict, path_type: str) -> dict:
    llm_cost = (
        stats["input_tokens"] * PRICING["input_cost_per_m"] / 1_000_000 +
        stats["output_tokens"] * PRICING["output_cost_per_m"] / 1_000_000
    )
    if path_type == "native":
        search_cost = stats["search_calls"] * PRICING["native_search_cost_per_call"]
    else:
        search_cost = stats["search_calls"] * PRICING["ydc_search_cost_per_call"]
    return {
        "llm_cost": round(llm_cost, 6),
        "search_cost": round(search_cost, 6),
        "total_cost": round(llm_cost + search_cost, 6),
    }


def print_comparison(
    query: str,
    native: dict,
    original: dict,
    optimized: dict,
    judge_result: dict | None = None,
    show_answers: bool = False,
) -> None:
    """Print a 3-way comparison table."""
    native_cost = compute_cost(native, "native")
    original_cost = compute_cost(original, "youdotcom")
    optimized_cost = compute_cost(optimized, "youdotcom")

    w = 24  # column width

    print("\n" + "=" * 100)
    print("THREE-WAY COMPARISON: GPT + Web Search")
    print("=" * 100)
    print(f"\nQuery:   \"{query}\"")
    print(f"Model:   {MODEL}")
    print(f"Native:  OpenAI web_search (search_context_size=medium)")
    if judge_result and not judge_result.get("error"):
        print(f"Judge:   Claude Sonnet 4.6 (blind 3-way, randomized positions)")
    print()

    header = f"{'METRIC':<32} {'Native':>{w}} {'YDC Original':>{w}} {'YDC Optimized':>{w}}"
    print(header)
    print(f"{'-'*32} {'-'*w} {'-'*w} {'-'*w}")

    rows = [
        ("Total tokens",
         f"{native['total_tokens']:,}",
         f"{original['total_tokens']:,}",
         f"{optimized['total_tokens']:,}"),
        ("  Input tokens",
         f"{native['input_tokens']:,}",
         f"{original['input_tokens']:,}",
         f"{optimized['input_tokens']:,}"),
        ("  Output tokens",
         f"{native['output_tokens']:,}",
         f"{original['output_tokens']:,}",
         f"{optimized['output_tokens']:,}"),
        ("  Search context (est.)",
         f"~{native['search_context_tokens']:,}",
         f"~{original['search_context_tokens']:,}",
         f"~{optimized['search_context_tokens']:,}"),
        ("", "", "", ""),
        ("API calls (LLM round-trips)",
         f"{native['api_calls']}",
         f"{original['api_calls']}",
         f"{optimized['api_calls']}"),
        ("Search calls",
         f"{native['search_calls']}",
         f"{original['search_calls']}",
         f"{optimized['search_calls']}"),
        ("Sources returned",
         f"{len(native['sources'])}",
         f"{len(original['sources'])}",
         f"{len(optimized['sources'])}"),
        ("Hit round limit",
         f"{'YES' if native['hit_round_limit'] else 'no'}",
         f"{'YES' if original['hit_round_limit'] else 'no'}",
         f"{'YES' if optimized['hit_round_limit'] else 'no'}"),
        ("", "", "", ""),
        ("End-to-end latency",
         f"{native['latency_ms']:,.0f}ms",
         f"{original['latency_ms']:,.0f}ms",
         f"{optimized['latency_ms']:,.0f}ms"),
        ("", "", "", ""),
        ("Estimated cost (total)",
         f"${native_cost['total_cost']:.5f}",
         f"${original_cost['total_cost']:.5f}",
         f"${optimized_cost['total_cost']:.5f}"),
        ("  LLM inference",
         f"${native_cost['llm_cost']:.5f}",
         f"${original_cost['llm_cost']:.5f}",
         f"${optimized_cost['llm_cost']:.5f}"),
        ("  Search",
         f"${native_cost['search_cost']:.5f}",
         f"${original_cost['search_cost']:.5f}",
         f"${optimized_cost['search_cost']:.5f}"),
    ]

    for label, v1, v2, v3 in rows:
        if not label:
            print()
            continue
        print(f"{label:<32} {v1:>{w}} {v2:>{w}} {v3:>{w}}")

    # ── Savings vs Native ──
    print(f"\n{'─'*100}")
    print("SAVINGS vs. NATIVE PATH")
    print(f"{'─'*100}")

    for label, stats, cost in [("You.com Original", original, original_cost),
                                ("You.com Optimized", optimized, optimized_cost)]:
        if native["total_tokens"] > 0:
            tok_save = (1 - stats["total_tokens"] / native["total_tokens"]) * 100
        else:
            tok_save = 0
        if native_cost["total_cost"] > 0:
            cost_save = (1 - cost["total_cost"] / native_cost["total_cost"]) * 100
        else:
            cost_save = 0
        print(f"  {label:<24} Tokens: {tok_save:+.0f}%    Cost: {cost_save:+.0f}%")

    # ── Optimized vs Original ──
    if original["total_tokens"] > 0:
        tok_diff = (1 - optimized["total_tokens"] / original["total_tokens"]) * 100
        print(f"\n  Optimized vs Original:   Tokens: {tok_diff:+.0f}%", end="")
        if original_cost["total_cost"] > 0:
            cost_diff = (1 - optimized_cost["total_cost"] / original_cost["total_cost"]) * 100
            print(f"    Cost: {cost_diff:+.0f}%")
        else:
            print()

    # ── Judge ──
    if judge_result:
        print(f"\n{'='*100}")
        if judge_result.get("error"):
            print(f"JUDGE: {judge_result['error']}")
        else:
            print("QUALITY EVALUATION (blind cross-model judge)")
            print(f"{'='*100}")

            dims = ["completeness", "relevance", "specificity", "citation_quality"]
            dim_labels = {
                "completeness": "Completeness",
                "relevance": "Relevance",
                "specificity": "Specificity",
                "citation_quality": "Citation Quality",
            }

            path_labels = {"native": "Native", "original": "YDC Original", "optimized": "YDC Optimized"}

            print(f"\n{'Dimension':<22} {'Native':>{16}} {'YDC Original':>{16}} {'YDC Optimized':>{16}}")
            print(f"{'-'*22} {'-'*16} {'-'*16} {'-'*16}")

            totals = {"native": 0, "original": 0, "optimized": 0}
            for dim in dims:
                vals = {}
                for path_key in ["native", "original", "optimized"]:
                    score = judge_result.get(f"{path_key}_scores", {}).get(dim, 0)
                    vals[path_key] = score
                    totals[path_key] += score
                print(f"{dim_labels[dim]:<22} "
                      f"{'█' * vals['native'] + '░' * (5 - vals['native'])} {vals['native']}/5"
                      f"       "
                      f"{'█' * vals['original'] + '░' * (5 - vals['original'])} {vals['original']}/5"
                      f"       "
                      f"{'█' * vals['optimized'] + '░' * (5 - vals['optimized'])} {vals['optimized']}/5")

            print(f"{'-'*22} {'-'*16} {'-'*16} {'-'*16}")
            print(f"{'Overall':<22} "
                  f"{'':>6}{totals['native']}/20"
                  f"{'':>8}{totals['original']}/20"
                  f"{'':>8}{totals['optimized']}/20")

            verdict = judge_result.get("verdict", "comparable")
            verdict_label = path_labels.get(verdict, verdict)
            print(f"\n  Verdict: {verdict_label} {'is best' if verdict != 'comparable' else '— all comparable'}")
            if judge_result.get("verdict_reasoning"):
                print(f"  Reason:  {judge_result['verdict_reasoning']}")

            # Per-path reasoning
            for path_key in ["native", "original", "optimized"]:
                reason = judge_result.get(f"{path_key}_scores", {}).get("reasoning", "")
                if reason:
                    print(f"  {path_labels[path_key]}: {reason}")

    # ── State management comparison ──
    print(f"\n{'─'*100}")
    print("STATE MANAGEMENT COMPARISON")
    print(f"{'─'*100}")
    print(f"  Native:        Single Responses API call — provider manages everything")
    print(f"  YDC Original:  chat.completions — full message history re-sent each round (context bloat)")
    print(f"  YDC Optimized: Responses API + previous_response_id — only tool outputs sent per round")
    print(f"{'─'*100}")

    # ── Cost footnote ──
    print(f"\n  Cost assumptions:")
    print(f"    LLM:            {MODEL} @ ${PRICING['input_cost_per_m']}/M input, ${PRICING['output_cost_per_m']}/M output")
    print(f"    You.com Search: ${PRICING['ydc_search_cost_per_call']*1000:.0f} per 1,000 queries")
    print(f"    Native Search:  ${PRICING['native_search_cost_per_call']*1000:.0f} per 1,000 searches + search content tokens at model rates")

    # ── Full answers ──
    if show_answers or _verbose():
        print(f"\n{'='*100}")
        print("FULL ANSWERS")
        print(f"{'='*100}")
        print(f"\n--- Native ---\n{native['answer']}")
        print(f"\n--- You.com Original ---\n{original['answer']}")
        print(f"\n--- You.com Optimized ---\n{optimized['answer']}")

    print()


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = [a for a in sys.argv[1:] if a.startswith("--")]

    if "--verbose" in flags:
        os.environ["GROUNDING_VERBOSE"] = "1"

    skip_judge = "--no-judge" in flags
    show_answers = "--show-answers" in flags

    query = " ".join(args) if args else None

    if not query:
        print("Usage: python gpt_comparison.py \"your question here\"")
        print("       python gpt_comparison.py --show-answers \"your question\"")
        print("       python gpt_comparison.py --no-judge \"your question\"")
        print("       python gpt_comparison.py --verbose \"your question\"")
        print()
        print("Runs three paths:")
        print("  1. Native:         GPT + OpenAI built-in web_search")
        print("  2. YDC Original:   GPT + You.com via chat.completions (client-side state)")
        print("  3. YDC Optimized:  GPT + You.com via Responses API (server-side state)")
        sys.exit(1)

    # Validate keys
    if not YDC_API_KEY or YDC_API_KEY.startswith("your-"):
        print("Error: YDC_API_KEY not set. Replace the placeholder at the top of this file.")
        sys.exit(1)
    if not OPENAI_API_KEY or OPENAI_API_KEY.startswith("sk-your-"):
        print("Error: OPENAI_API_KEY not set. Replace the placeholder at the top of this file.")
        sys.exit(1)

    client = OpenAI(api_key=OPENAI_API_KEY, timeout=LLM_TIMEOUT)

    print(f"Three-Way Comparison: {MODEL}")
    print(f"Query: \"{query}\"")
    print(f"Judge: {'OFF' if skip_judge else 'Claude Sonnet 4.6 (cross-model, blind)'}")
    print(f"Verbose: {'ON' if _verbose() else 'OFF'}")
    print()

    # ── Run all three paths ──
    print("Running Native path...")
    native_stats = run_native(query, client)
    print(f"  Done ({native_stats['total_tokens']:,} tokens, {native_stats['latency_ms']:.0f}ms)")

    print("Running You.com Original path...")
    original_stats = run_youdotcom_original(query, client)
    print(f"  Done ({original_stats['total_tokens']:,} tokens, {original_stats['latency_ms']:.0f}ms)")

    print("Running You.com Optimized path...")
    optimized_stats = run_youdotcom_optimized(query, client)
    print(f"  Done ({optimized_stats['total_tokens']:,} tokens, {optimized_stats['latency_ms']:.0f}ms)")

    # ── Judge ──
    judge_result = None
    if not skip_judge:
        print("Running judge evaluation (Claude Sonnet 4.6)...")
        judge_result = run_judge(query, native_stats, original_stats, optimized_stats)
        if judge_result.get("error"):
            print(f"  {judge_result['error']}")
        else:
            print(f"  Done")

    # ── Print comparison ──
    print_comparison(query, native_stats, original_stats, optimized_stats,
                     judge_result, show_answers)


if __name__ == "__main__":
    main()
