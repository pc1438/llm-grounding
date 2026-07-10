"""
compare.py — Side-by-side comparison: LLM + You.com Search vs. LLM + Native Web Search.

Runs the same query through both paths, compares token usage and cost, then
uses a cross-model judge to blindly evaluate answer quality on four dimensions:
completeness, relevance, specificity, and citation quality.

This proves two things:
  1. You.com as a dedicated search tool is cheaper (fewer tokens, predictable cost)
  2. The output quality is comparable or better (judge scores)

Supported providers:
  - claude  → Anthropic Claude Sonnet 4.6
  - openai  → OpenAI GPT-5.4
  - kimi    → Moonshot Kimi K2.6
  - qwen    → Alibaba Qwen3.7-max (DashScope standard endpoint)

Requirements:
    pip install anthropic openai requests python-dotenv

Run:
    python compare.py claude "What happened in tech news today?"
    python compare.py openai "Who won the 2026 Olympic hockey gold?"
    python compare.py qwen "latest AI research papers"
    python compare.py claude --verbose "current S&P 500 price"
    python compare.py openai --no-judge "latest NVIDIA earnings"

The script imports from ../grounding/ for the You.com tool-use path,
and uses each LLM provider's native web search for the comparison path.

QWEN ENDPOINT NOTE:
  The Playground tab uses the MaaS workspace endpoint (DASHSCOPE_BASE_URL env var).
  The Comparison tab uses the standard DashScope international endpoint:
    https://dashscope-intl.aliyuncs.com/compatible-mode/v1
  This distinction matters: enable_search (native web search) is NOT available on
  MaaS workspace endpoints as of 2026-06 — it requires the standard public endpoint.
  Both endpoints use the same DASHSCOPE_API_KEY.
"""

import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load env from comparison dir, then fall back to grounding dir
load_dotenv("env.txt") or load_dotenv(".env")
load_dotenv(Path(__file__).parent.parent / "grounding" / "env.txt")
load_dotenv(Path(__file__).parent.parent / "grounding" / ".env")

# Add grounding/ to path so we can import from it
sys.path.insert(0, str(Path(__file__).parent.parent / "grounding"))

from judge import run_judge

from search_tool import (
    is_verbose,
)
from agent_pool import get_agent as _get_agent_pool

def _get_agent(provider: str, model_id: str):
    return _get_agent_pool(provider, model_id)

# ─── Load model config + pricing from single source of truth ──────────────
# Edit comparison/pricing.json to update rates — this file reads it at startup.
_PRICING_FILE = Path(__file__).parent / "pricing.json"
try:
    with open(_PRICING_FILE) as _f:
        _PRICING_DATA = json.load(_f)
except Exception as _e:
    import logging as _logging
    _logging.error(f"Failed to load pricing.json: {_e}")
    _PRICING_DATA = {"models": {}}

# ─── Model configurations ─────────────────────────────────────────────────
# Loaded from comparison/pricing.json — edit that file to add models or update
# rates. All fields (native_search_tool, provider, judge, costs) live there.
# in_comparison: true entries are loaded here; playground-only models are excluded.
MODELS = {k: v for k, v in _PRICING_DATA.get("models", {}).items() if v.get("in_comparison")}



def describe_native_search(model_config: dict) -> str:
    """Return a human-readable description of the native search tool config.

    Reads the config dynamically so CLI output and UI footnotes always
    reflect whatever is set in MODELS.
    """
    tool = model_config.get("native_search_tool") or {}
    if not tool:
        return "not supported"
    tool_type = tool.get("type", "unknown")

    # Kimi builtin_function: show the function name
    if tool_type == "builtin_function":
        fn_name = tool.get("name", "unknown")
        return f"builtin_function:{fn_name} (Kimi built-in search)"

    # Qwen web_search: DashScope Responses API built-in (no search_context_size — that's OpenAI)
    if tool_type == "web_search" and not tool.get("search_context_size"):
        return "web_search (DashScope built-in, Responses API)"

    # OpenRouter Exa: :online suffix, single-call, Exa search engine
    if tool_type == "exa_online":
        return "exa_online (OpenRouter :online suffix — Exa search, single call)"

    parts = [tool_type]

    # Anthropic-specific: allowed_callers controls code execution
    callers = tool.get("allowed_callers")
    if callers:
        if callers == ["direct"]:
            parts.append("code execution disabled")
        else:
            parts.append(f"allowed_callers={callers}")
    elif "20260209" in tool_type:
        parts.append("code execution enabled (default)")

    # OpenAI-specific: search_context_size controls how much context is used
    ctx_size = tool.get("search_context_size")
    if ctx_size:
        parts.append(f"search_context_size={ctx_size}")

    return " | ".join(parts)


def calculate_costs(stats: dict, model_config: dict, path: str = "ydc") -> dict:
    """Return {"llm", "search", "livecrawl_overage", "total"} costs in USD.

    path: "ydc" (default) — You.com search path (ydc_search_cost_per_call, livecrawl).
          anything else   — native search path (native_search_cost_per_call, no livecrawl).

    Rates come from pricing.json (via model_config) — nothing is hardcoded here.
    Called by print_comparison(), server.py, and benchmark_runner.py.

    Livecrawl pricing (ydc only): first 10 URLs per call included in base $5/1k rate;
    additional URLs charged at $1/1,000 URLs, billed on count (requested, not returned).
    """
    tb = stats.get("token_breakdown", {})
    input_tokens  = tb.get("input",  stats.get("input_tokens",  0))
    output_tokens = tb.get("output", stats.get("output_tokens", 0))
    llm = (
        input_tokens  * model_config.get("input_cost_per_m",  0) / 1_000_000
        + output_tokens * model_config.get("output_cost_per_m", 0) / 1_000_000
    )
    if path == "ydc":
        search_cost_per_call = model_config.get("ydc_search_cost_per_call") or 0
        livecrawl_overage = 0.0
        for call in stats.get("tool_calls", []):
            if call.get("livecrawl"):
                count = call.get("count", 5)
                excess = max(0, count - 10)
                livecrawl_overage += excess * 0.001  # $1/1000 URLs
    else:
        search_cost_per_call = model_config.get("native_search_cost_per_call") or 0
        livecrawl_overage = 0.0

    search = stats["search_calls"] * search_cost_per_call
    total = llm + search + livecrawl_overage
    return {"llm": llm, "search": search, "livecrawl_overage": livecrawl_overage, "total": total}


# ─── Path A: LLM + You.com Search ──────────────────────────────────────────

def run_youdotcom(
    question: str,
    model_config: dict,
    on_progress=None,
) -> dict:
    """Run a query through LLM + You.com Search (tool-use approach).

    Returns _empty_stats() shape (from base_agent) plus a "path" label field.
    No translation — the agent's stats dict is returned directly.
    """
    model_id = model_config["model"]
    provider = model_config["provider"]
    agent = _get_agent(provider, model_id)
    stats = agent.ask(question, on_progress=on_progress)
    stats["path"] = f"{model_id} + You.com Search"
    return stats


# ─── Path B: LLM + Native Web Search ───────────────────────────────────────

def run_native(
    question: str,
    model_config: dict,
    on_progress=None,
    system_prompt: str = None,
    native_path: str = "responses",  # "responses" or "chat" (OpenAI only)
) -> dict:
    """Run a query through LLM + provider's native web search.

    Delegates to the provider's agent class (ClaudeAgent, OpenAIAgent, etc.)
    which owns all provider-specific native search logic. Translates the
    returned base_agent stats shape into compare.py's flat stats shape for
    downstream display and cost calculation.

    Args:
        native_path: OpenAI only — "responses" (Responses API, default) or
                     "chat" (chat.completions with web_search_preview tool).
    """
    if not model_config.get("native_search_tool"):
        from base_agent import _empty_stats as _es
        _mid = model_config.get("model", "")
        stats = _es(_mid)
        stats["not_supported"] = True
        stats["path"] = f"{_mid} + Native Web Search"
        stats["answer"] = "Native web search is not supported for this model."
        return stats

    provider = model_config["provider"]
    model_id = model_config["model"]
    agent = _get_agent(provider, model_id)

    if provider == "anthropic":
        base_stats = agent.run_native_search(
            question,
            system_prompt=system_prompt,
            on_progress=on_progress,
            native_search_tool=model_config.get("native_search_tool"),
        )
    elif provider == "openai":
        base_stats = agent.run_native_search(
            question,
            system_prompt=system_prompt,
            on_progress=on_progress,
            native_search_tool=model_config.get("native_search_tool"),
            chat_native_search_tool=model_config.get("chat_native_search_tool"),
            path=native_path,
        )
    else:
        base_stats = agent.run_native_search(
            question,
            system_prompt=system_prompt,
            on_progress=on_progress,
        )

    path_label = f"{model_id} + Native Web Search"
    if native_path == "chat":
        path_label = f"{model_id} + Native Web Search (chat.completions)"
    base_stats["path"] = path_label
    return base_stats




# ─── Comparison output ──────────────────────────────────────────────────────

def print_comparison(
    query: str,
    provider: str,
    ydc: dict,
    native: dict,
    judge_result: dict | None = None,
    show_answers: bool = False,
) -> None:
    """Print a side-by-side comparison of both runs."""
    model_config = MODELS[provider]
    w = 32

    ydc_costs = calculate_costs(ydc, model_config)
    ydc_llm, ydc_search, ydc_cost = ydc_costs["llm"], ydc_costs["search"], ydc_costs["total"]

    native_costs = calculate_costs(native, model_config, path="native")
    native_llm, native_search, native_cost = native_costs["llm"], native_costs["search"], native_costs["total"]

    print("\n" + "=" * 78)
    print("COMPARISON: You.com Search vs. Native Web Search")
    print("=" * 78)
    print(f"\nQuery:    \"{query}\"")
    print(f"Model:    {model_config['model']}")
    print(f"Native:   {describe_native_search(model_config)}")
    if judge_result and not judge_result.get("error"):
        judge_name = "GPT-5.4" if judge_result["judge_model"] == "openai" else "Claude Sonnet"
        print(f"Judge:    {judge_name} (blind evaluation, randomized position)")
    print()

    # ── Cost & Token Table ──
    print(f"{'METRIC':<30} {'You.com':>{w}} {'Native':>{w}}")
    print(f"{'-'*30} {'-'*w} {'-'*w}")

    def _tok(s, key):
        return s.get("token_breakdown", {}).get(key, 0)

    rows = [
        ("Total tokens",
         f"{ydc['tokens_used']:,}",
         f"{native['tokens_used']:,}"),
        ("  Input tokens",
         f"{_tok(ydc, 'input'):,}",
         f"{_tok(native, 'input'):,}"),
        ("  Output tokens",
         f"{_tok(ydc, 'output'):,}",
         f"{_tok(native, 'output'):,}"),
        ("  Search context (est.)",
         f"~{_tok(ydc, 'search_context'):,}",
         f"~{_tok(native, 'search_context'):,}"),
        ("", "", ""),
        ("API calls",
         f"{ydc['api_calls']}",
         f"{native['api_calls']}"),
        ("Search calls",
         f"{ydc['search_calls']}",
         f"{native['search_calls']}"),
        ("Sources returned",
         f"{len(ydc['sources'])}",
         f"{len(native['sources'])}"),
        ("", "", ""),
        ("End-to-end latency",
         f"{ydc['latency_ms']:,.0f}ms",
         f"{native['latency_ms']:,.0f}ms"),
        ("", "", ""),
        ("Estimated cost (total)",
         f"${ydc_cost:.5f}",
         f"${native_cost:.5f}"),
        ("  LLM inference",
         f"${ydc_llm:.5f}",
         f"${native_llm:.5f}"),
        ("  Search",
         f"${ydc_search:.5f}",
         f"${native_search:.5f}"),
    ]

    for label, ydc_val, native_val in rows:
        if not label:
            print()
            continue
        print(f"{label:<30} {ydc_val:>{w}} {native_val:>{w}}")

    # Savings summary
    if native["tokens_used"] > 0:
        savings = native["tokens_used"] - ydc["tokens_used"]
        savings_pct = (savings / native["tokens_used"]) * 100
        if savings > 0:
            print(f"\n  Token savings with You.com: {savings:,} tokens ({savings_pct:.0f}% fewer)")
        elif savings < 0:
            print(f"\n  You.com used {-savings:,} more tokens ({-savings_pct:.0f}% more)")

    if native_cost > 0:
        cost_savings = native_cost - ydc_cost
        cost_pct = (cost_savings / native_cost) * 100
        if cost_savings > 0:
            print(f"  Cost savings with You.com:  ${cost_savings:.5f} ({cost_pct:.0f}% cheaper per query)")

    # ── Judge Scores ──
    if judge_result:
        print(f"\n{'='*78}")
        if judge_result.get("error"):
            print(f"JUDGE: {judge_result['error']}")
        else:
            print("QUALITY EVALUATION (blind cross-model judge)")
            print(f"{'='*78}")

            ydc_s = judge_result["ydc_scores"]
            nat_s = judge_result["native_scores"]

            dimensions = ["completeness", "relevance", "specificity", "citation_quality"]
            dim_labels = {
                "completeness": "Completeness",
                "relevance": "Relevance",
                "specificity": "Specificity",
                "citation_quality": "Citation Quality",
            }

            print(f"\n{'Dimension':<25} {'You.com':>{20}} {'Native':>{20}}")
            print(f"{'-'*25} {'-'*20} {'-'*20}")

            ydc_total = 0
            nat_total = 0
            for dim in dimensions:
                y = ydc_s.get(dim, 0)
                n = nat_s.get(dim, 0)
                try:
                    y = int(y)
                except (TypeError, ValueError):
                    y = 0
                try:
                    n = int(n)
                except (TypeError, ValueError):
                    n = 0
                ydc_total += y
                nat_total += n
                label = dim_labels[dim]
                y_bar = "█" * y + "░" * (5 - y)
                n_bar = "█" * n + "░" * (5 - n)
                print(f"{label:<25} {y_bar} {y}/5          {n_bar} {n}/5")

            print(f"{'-'*25} {'-'*20} {'-'*20}")
            print(f"{'Overall':<25} {'':>11}{ydc_total}/20          {'':>5}{nat_total}/20")

            # Winner determination
            if ydc_total > nat_total:
                diff = ydc_total - nat_total
                print(f"\n  You.com path scored {diff} point(s) higher overall.")
            elif nat_total > ydc_total:
                diff = nat_total - ydc_total
                print(f"\n  Native path scored {diff} point(s) higher overall.")
            else:
                print(f"\n  Both paths scored equally.")

            # Verdict
            verdict = judge_result.get("verdict", "")
            verdict_reasoning = judge_result.get("verdict_reasoning", "")
            if verdict:
                verdict_label = {
                    "ydc": "You.com path is better",
                    "native": "Native path is better",
                    "comparable": "Both paths are comparable",
                }.get(verdict, verdict)
                print(f"\n  Verdict: {verdict_label}")
                if verdict_reasoning:
                    print(f"  Reason:  {verdict_reasoning}")

            # Judge reasoning
            ydc_reason = ydc_s.get("reasoning", "")
            nat_reason = nat_s.get("reasoning", "")
            if ydc_reason or nat_reason:
                print(f"\n  Per-answer reasoning:")
                if ydc_reason:
                    print(f"    You.com: {ydc_reason}")
                if nat_reason:
                    print(f"    Native:  {nat_reason}")

    # ── Control comparison ──
    print(f"\n{'─'*78}")
    print("Control comparison:")
    print(f"  You.com:  search_uuid={ydc.get('search_uuid', 'N/A')[:20]}...")
    print(f"            {len(ydc['sources'])} source URLs logged, domain/freshness filters available")
    print(f"  Native:   No search_uuid, no domain filters, no freshness control")
    print(f"            Search context size determined by provider, not you")
    print(f"{'─'*78}")

    # Cost footnote
    native_note = ""
    if model_config["provider"] == "anthropic":
        native_note = " + search results billed as input tokens"
    elif model_config["provider"] == "openai":
        native_note = " + search content tokens billed at model rates"

    print(f"\n  Cost assumptions:")
    print(f"    LLM:            {model_config['model']} @ ${model_config['input_cost_per_m']}/M input, ${model_config['output_cost_per_m']}/M output")
    print(f"    You.com Search: ${(model_config.get('ydc_search_cost_per_call') or 0)*1000:.0f} per 1,000 queries")
    print(f"    Native Search:  ${(model_config.get('native_search_cost_per_call') or 0)*1000:.0f} per 1,000 searches{native_note}")

    # Print full answers (--show-answers or --verbose)
    if show_answers or is_verbose():
        print(f"\n{'='*78}")
        print("FULL ANSWERS")
        print(f"{'='*78}")
        print(f"\n--- You.com Path ---\n{ydc['answer']}")
        print(f"\n--- Native Path ---\n{native['answer']}")

    print()


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = [a for a in sys.argv[1:] if a.startswith("--")]

    if "--verbose" in flags:
        os.environ["GROUNDING_VERBOSE"] = "1"

    skip_judge = "--no-judge" in flags
    show_answers = "--show-answers" in flags

    provider = args[0] if args else None
    query = " ".join(args[1:]) if len(args) > 1 else None

    if not provider or not query:
        print("Usage: python compare.py <model> \"your question here\"")
        print("       python compare.py <model> --show-answers \"your question\"")
        print("       python compare.py <model> --no-judge \"your question\"")
        print("       python compare.py <model> --verbose \"your question\"")
        print()
        print("Flags:")
        print("  --show-answers  Print full answers from both paths")
        print("  --no-judge      Skip judge evaluation (faster, no OPENAI_API_KEY needed)")
        print("  --verbose       Full logging + show answers")
        print()
        print(f"Available models: {', '.join(MODELS.keys())}")
        print()
        print("Examples:")
        print("  python compare.py claude \"Who won the 2026 men's hockey Olympic gold?\"")
        print("  python compare.py claude --show-answers \"What happened in tech news today?\"")
        print("  python compare.py claude --no-judge \"current S&P 500 price\"")
        sys.exit(1)

    if provider not in MODELS:
        print(f"Unknown model: {provider}")
        print(f"Available: {', '.join(MODELS.keys())}")
        sys.exit(1)

    model_config = MODELS[provider]

    # Validate keys
    ydc_key = os.environ.get("YDC_API_KEY", "")
    if not ydc_key:
        print("Error: YDC_API_KEY not set.")
        sys.exit(1)

    print(f"Comparison: {model_config['model']} + You.com vs. {model_config['model']} + Native Search")
    print(f"Query:      \"{query}\"")
    print(f"Judge:      {'OFF' if skip_judge else model_config['judge'] + ' (cross-model, blind)'}")
    print(f"Verbose:    {'ON' if is_verbose() else 'OFF'}")
    print()

    # Run both paths
    print("Running You.com path...")
    ydc_stats = run_youdotcom(query, model_config)
    print(f"  Done ({ydc_stats['tokens_used']:,} tokens, {ydc_stats['latency_ms']:.0f}ms)")

    print("Running Native path...")
    native_stats = run_native(query, model_config)
    print(f"  Done ({native_stats['tokens_used']:,} tokens, {native_stats['latency_ms']:.0f}ms)")

    # Run judge evaluation
    judge_result = None
    judge_model = model_config.get("judge")
    if not skip_judge and judge_model:
        print(f"Running judge evaluation ({judge_model})...")
        judge_result = run_judge(
            query, ydc_stats["answer"], native_stats["answer"], judge_model,
            sources_ydc=ydc_stats["sources"], sources_native=native_stats["sources"],
        )
        if judge_result.get("error"):
            print(f"  {judge_result['error']}")
        else:
            print(f"  Done")

    # Print comparison
    print_comparison(query, provider, ydc_stats, native_stats, judge_result, show_answers)


if __name__ == "__main__":
    main()
