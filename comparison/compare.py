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
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from dotenv import load_dotenv

# Load env from comparison dir, then fall back to grounding dir
load_dotenv("env.txt") or load_dotenv(".env")
load_dotenv(Path(__file__).parent.parent / "grounding" / "env.txt")
load_dotenv(Path(__file__).parent.parent / "grounding" / ".env")

# Add grounding/ to path so we can import from it
sys.path.insert(0, str(Path(__file__).parent.parent / "grounding"))

from anthropic import Anthropic
from openai import OpenAI

from search_tool import (
    get_system_prompt,  # Runtime prompt with current date/time
    MAX_TOKENS,         # Canonical constant — defined in search_tool.py
    MAX_TOOL_ROUNDS,    # Canonical constant — defined in search_tool.py
    TOOL_SCHEMA,
    TOOL_SCHEMA_ANTHROPIC,
    TOOL_SCHEMA_RESPONSES,
    execute_search,
    extract_urls,
    extract_search_uuid,
    is_verbose,
    set_interface,
)
from base_agent import AnthropicAgent, OpenAICompatibleAgent, OpenAIResponsesAgent
from agents.claude_agent import ClaudeAgent
from agents.openai_agent import OpenAIAgent
from agents.kimi_agent import KimiAgent
from agents.qwen_agent import QwenAgent

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

# SYSTEM_PROMPT and MAX_TOKENS imported from search_tool.py (single source of truth)


def describe_native_search(model_config: dict) -> str:
    """Return a human-readable description of the native search tool config.

    Reads the config dynamically so CLI output and UI footnotes always
    reflect whatever is set in MODELS.
    """
    tool = model_config.get("native_search_tool", {})
    tool_type = tool.get("type", "unknown")

    # Kimi builtin_function: show the function name
    if tool_type == "builtin_function":
        fn_name = tool.get("name", "unknown")
        return f"builtin_function:{fn_name} (Kimi built-in search)"


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


def calculate_costs(stats: dict, model_config: dict) -> dict:
    """Return {"llm", "search", "livecrawl_overage", "total"} costs in USD for the You.com path.

    Rates come from pricing.json (via model_config) — nothing is hardcoded here.
    Called by both print_comparison() and server.py so the formula lives once.

    Livecrawl pricing: first 10 URLs per call are included in the base $5/1k rate;
    additional URLs are charged at $1/1,000 URLs, billed on the count parameter
    (URLs requested, not returned).
    """
    llm = (
        stats["input_tokens"] * model_config["input_cost_per_m"] / 1_000_000
        + stats["output_tokens"] * model_config["output_cost_per_m"] / 1_000_000
    )
    search_cost_per_call = model_config.get("ydc_search_cost_per_call") or 0
    search = stats["search_calls"] * search_cost_per_call

    # Livecrawl overage: $1 per 1,000 URLs above the first 10 per call
    livecrawl_overage = 0.0
    for call in stats.get("tool_calls", []):
        if call.get("livecrawl"):
            count = call.get("count", 5)
            excess = max(0, count - 10)
            livecrawl_overage += excess * 0.001  # $1/1000 URLs

    total = llm + search + livecrawl_overage
    return {"llm": llm, "search": search, "livecrawl_overage": livecrawl_overage, "total": total}


def calculate_native_costs(stats: dict, model_config: dict) -> dict:
    """Return {"llm", "search", "total"} costs in USD for the native search path.

    Separate from calculate_costs because native_search_cost_per_call can be
    None (Llama has no native search) and the two paths use different config keys.
    """
    llm = (
        stats["input_tokens"] * model_config["input_cost_per_m"] / 1_000_000
        + stats["output_tokens"] * model_config["output_cost_per_m"] / 1_000_000
    )
    search_cost_per_call = model_config.get("native_search_cost_per_call") or 0
    search = stats["search_calls"] * search_cost_per_call
    return {"llm": llm, "search": search, "total": llm + search}


# ─── Judge prompt ───────────────────────────────────────────────────────────

JUDGE_SYSTEM_PROMPT = (
    "You are an impartial evaluation judge. You will receive a question and two "
    "answers (labeled Answer A and Answer B) from different search-grounded AI systems. "
    "You do NOT know which system produced which answer.\n\n"
    "IMPORTANT: You CANNOT verify factual claims. Do NOT attempt to judge whether "
    "specific facts are true or false based on your own knowledge — your training "
    "data may be outdated. Instead, evaluate the STRUCTURE and QUALITY of each "
    "answer relative to the other.\n\n"
    "Evaluate each answer on four dimensions using a 1-5 scale:\n"
    "  • Completeness — Does the answer fully address all parts of the question?\n"
    "  • Relevance — Is the information on-topic and useful for the question asked?\n"
    "  • Specificity — Does the answer provide concrete details (dates, names, numbers, "
    "    context) rather than vague or generic claims?\n"
    "  • Citation Quality — Are sources cited, numbered, and traceable? Can a reader "
    "    verify the claims by following the references?\n\n"
    "Scoring rubric:\n"
    "  5 = Excellent  4 = Good  3 = Adequate  2 = Poor  1 = Very poor\n\n"
    "After scoring both answers, provide a head-to-head verdict: which answer is "
    "better overall, or are they comparable?\n\n"
    "Respond ONLY with valid JSON in this exact format, nothing else:\n"
    "{\n"
    '  "answer_a": {\n'
    '    "completeness": <1-5>,\n'
    '    "relevance": <1-5>,\n'
    '    "specificity": <1-5>,\n'
    '    "citation_quality": <1-5>,\n'
    '    "reasoning": "<1-2 sentence justification>"\n'
    "  },\n"
    '  "answer_b": {\n'
    '    "completeness": <1-5>,\n'
    '    "relevance": <1-5>,\n'
    '    "specificity": <1-5>,\n'
    '    "citation_quality": <1-5>,\n'
    '    "reasoning": "<1-2 sentence justification>"\n'
    "  },\n"
    '  "verdict": "<A_better | B_better | comparable>",\n'
    '  "verdict_reasoning": "<1 sentence explaining the overall verdict>"\n'
    "}"
)


# ─── Token tracking helpers ─────────────────────────────────────────────────

def _empty_stats() -> dict:
    """Initialize an empty stats dict for one run."""
    return {
        "path": "",
        "model": "",
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
    }


# ─── Path A: LLM + You.com Search ──────────────────────────────────────────

def run_youdotcom(
    question: str,
    client,
    model_config: dict,
    on_progress=None,
    system_prompt: str = None,
    force_chat_completions: bool = False,
) -> dict:
    """Run a query through LLM + You.com Search (tool-use approach).

    Instantiates the right base_agent class for the model's provider and API shape,
    then calls ask() — which runs the full tool-use loop and returns the stats dict.

    API shape selection (controlled by model_config["api_shape"]):
    - anthropic      → AnthropicAgent (unaffected by force_chat_completions)
    - responses      → OpenAIResponsesAgent by default; OpenAICompatibleAgent if force_chat_completions=True
                       GPT: default path. Qwen: tool named 'you_search' — DashScope intercepts
                       'web_search' and routes to native search instead of calling our function.
    - chat_completions→ OpenAICompatibleAgent always
                       Kimi: Moonshot /v1/responses returns 404 (confirmed 2026-06-30)

    Args:
        force_chat_completions: Set True to use the legacy chat.completions path for GPT/Qwen.
                                Has no effect on anthropic or kimi.
    on_progress(msg: str) — optional callback fired on each search/round event.
    system_prompt — override the default system prompt (for A/B testing).

    See decisions/006_gpt_ydc_chat_vs_responses_api.md for full compatibility findings.
    See decisions/007_agent_architecture.md for layer responsibilities.
    """
    set_interface("direct_api")
    prompt = get_system_prompt() if system_prompt is None else system_prompt
    model_id = model_config["model"]
    provider = model_config["provider"]
    api_shape = model_config.get("api_shape", "chat_completions")
    verbose = is_verbose()

    def _notify(msg):
        if verbose:
            print(f"  [You.com] {msg}")
        if on_progress:
            on_progress(msg)

    # Instantiate the right loop class based on provider and flags
    if provider == "anthropic":
        agent = AnthropicAgent(client=client, model=model_id, system_prompt=prompt)
    elif api_shape == "responses" and not force_chat_completions:
        agent = OpenAIResponsesAgent(client=client, model=model_id, system_prompt=prompt)
    else:
        max_tokens_param = model_config.get("max_tokens_param", "max_tokens")
        extra_body = model_config.get("extra_body")
        agent = OpenAICompatibleAgent(
            client=client,
            model=model_id,
            system_prompt=prompt,
            max_tokens_param=max_tokens_param,
            extra_body=extra_body,
        )

    # Run the loop — on_progress fires at each search step
    agent_stats = agent.ask(question, on_progress=_notify)

    # Translate base_agent stats shape to compare.py's stats shape
    stats = _empty_stats()
    stats["path"] = f"{model_id} + You.com Search"
    stats["model"] = model_id
    stats["model_confirmed"] = agent_stats.get("model")
    stats["answer"] = agent_stats["answer"]
    stats["sources"] = agent_stats["sources"]
    stats["search_uuid"] = agent_stats.get("search_uuid", "")
    stats["total_tokens"] = agent_stats["tokens_used"]
    stats["input_tokens"] = agent_stats["token_breakdown"]["input"]
    stats["output_tokens"] = agent_stats["token_breakdown"]["output"]
    stats["search_context_tokens"] = agent_stats["token_breakdown"]["search_context"]
    stats["api_calls"] = agent_stats["api_calls"]
    stats["search_calls"] = agent_stats["search_calls"]
    stats["latency_ms"] = agent_stats["latency_ms"]
    stats["tool_calls"] = agent_stats.get("tool_calls", [])
    return stats


# ─── Path B: LLM + Native Web Search ───────────────────────────────────────

def run_native(
    question: str,
    client,  # retained for API compatibility — agent classes create their own clients
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
        stats = _empty_stats()
        stats["not_supported"] = True
        stats["answer"] = "Native web search is not supported for this model."
        return stats

    provider = model_config["provider"]
    model_id = model_config["model"]

    # Instantiate the provider's agent class and call run_native_search().
    # Each agent class resolves its own API key from the environment.
    if provider == "anthropic":
        agent = ClaudeAgent(model=model_id)
        base_stats = agent.run_native_search(
            question,
            system_prompt=system_prompt,
            on_progress=on_progress,
            native_search_tool=model_config.get("native_search_tool"),
        )

    elif provider == "openai":
        agent = OpenAIAgent(model=model_id)
        base_stats = agent.run_native_search(
            question,
            system_prompt=system_prompt,
            on_progress=on_progress,
            native_search_tool=model_config.get("native_search_tool"),
            chat_native_search_tool=model_config.get("chat_native_search_tool"),
            path=native_path,
        )

    elif provider == "kimi":
        agent = KimiAgent(model=model_id)
        base_stats = agent.run_native_search(
            question,
            system_prompt=system_prompt,
            on_progress=on_progress,
        )

    elif provider == "qwen":
        agent = QwenAgent(model=model_id)
        base_stats = agent.run_native_search(
            question,
            system_prompt=system_prompt,
            on_progress=on_progress,
        )

    else:
        raise ValueError(f"Unknown provider: {provider}")

    # Translate base_agent shape → compare.py flat shape
    path_label = f"{model_id} + Native Web Search"
    if native_path == "chat":
        path_label = f"{model_id} + Native Web Search (chat.completions)"

    stats = _empty_stats()
    stats["path"] = path_label
    stats["model"] = model_id
    stats["model_confirmed"] = base_stats.get("model_confirmed")
    stats["answer"] = base_stats.get("answer", "")
    stats["sources"] = base_stats.get("sources", [])
    stats["total_tokens"] = base_stats.get("tokens_used", 0)
    stats["input_tokens"] = base_stats.get("token_breakdown", {}).get("input", 0)
    stats["output_tokens"] = base_stats.get("token_breakdown", {}).get("output", 0)
    stats["search_context_tokens"] = base_stats.get("token_breakdown", {}).get("search_context", 0)
    stats["api_calls"] = base_stats.get("api_calls", 0)
    stats["search_calls"] = base_stats.get("search_calls", 0)
    stats["latency_ms"] = base_stats.get("latency_ms", 0.0)
    return stats




# ─── Judge evaluation ───────────────────────────────────────────────────────

def _answer_with_sources(answer: str, sources: list[str]) -> str:
    """Append a numbered source list to an answer for judge evaluation.

    The LLM's answer often cites [1], [2] etc. but the actual URLs are tracked
    separately in stats["sources"]. Without appending them, the judge sees
    citation markers with no traceable references — unfairly penalizing the
    You.com path on citation quality.
    """
    if not sources:
        return answer
    source_block = "\n\nSources:\n" + "\n".join(
        f"[{i}] {url}" for i, url in enumerate(sources, 1)
    )
    return answer + source_block


def run_judge(
    question: str,
    answer_ydc: str,
    answer_native: str,
    judge_model: str,
    sources_ydc: list[str] | None = None,
    sources_native: list[str] | None = None,
) -> dict:
    """Run a blind cross-model evaluation of both answers.

    Randomly assigns answers to A/B positions to prevent position bias.
    Uses a different LLM than the one being tested.

    Args:
        sources_ydc:    Optional list of URLs from the You.com path, appended
                        to the answer so the judge can evaluate citation traceability.
        sources_native: Optional list of URLs from the native path (usually already
                        inline in the answer text, but included for symmetry).

    Returns:
        dict with keys: youdotcom_scores, native_scores, judge_model, position_map
    """
    # Guard against None answers before string operations
    answer_ydc = answer_ydc or ""
    answer_native = answer_native or ""
    # Append source lists so the judge can evaluate citation traceability
    answer_ydc = _answer_with_sources(answer_ydc, sources_ydc or [])
    answer_native = _answer_with_sources(answer_native, sources_native or [])
    # Randomize position to prevent bias
    if random.random() < 0.5:
        answer_a, answer_b = answer_ydc, answer_native
        position_map = {"a": "youdotcom", "b": "native"}
    else:
        answer_a, answer_b = answer_native, answer_ydc
        position_map = {"a": "native", "b": "youdotcom"}

    judge_prompt = (
        f"Question: {question}\n\n"
        f"--- Answer A ---\n{answer_a}\n\n"
        f"--- Answer B ---\n{answer_b}\n\n"
        f"Evaluate both answers. Respond with JSON only."
    )

    if judge_model == "openai":
        # Use GPT-5.4 as judge (cross-model: GPT judges Claude)
        openai_key = os.environ.get("OPENAI_API_KEY", "")
        if not openai_key:
            return {"error": "OPENAI_API_KEY not set — skipping judge evaluation"}

        judge_client = OpenAI(api_key=openai_key)
        response = judge_client.chat.completions.create(
            model="gpt-5.4",
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": judge_prompt},
            ],
            max_completion_tokens=1024,
            temperature=0,
        )
        raw = response.choices[0].message.content

    elif judge_model == "claude":
        # Use Claude as judge
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not anthropic_key:
            return {"error": "ANTHROPIC_API_KEY not set — skipping judge evaluation"}

        judge_client = Anthropic(api_key=anthropic_key)
        response = judge_client.messages.create(
            model="claude-sonnet-4-6",
            system=JUDGE_SYSTEM_PROMPT,
            max_tokens=1024,
            messages=[{"role": "user", "content": judge_prompt}],
        )
        raw = ""
        for block in response.content:
            if hasattr(block, "text"):
                raw += block.text
    else:
        return {"error": f"Unknown judge model: {judge_model}"}

    # Parse judge response
    try:
        # Strip markdown code fences if present
        clean = raw.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[1]
            clean = clean.rsplit("```", 1)[0]
        scores = json.loads(clean)
    except (json.JSONDecodeError, IndexError):
        import logging
        logging.debug(f"Judge returned invalid JSON (first 200 chars): {raw[:200]}")
        return {"error": "Judge returned invalid JSON — check logs for details"}

    # Map positions back to paths
    ydc_key = "answer_a" if position_map["a"] == "youdotcom" else "answer_b"
    native_key = "answer_a" if position_map["a"] == "native" else "answer_b"

    # Replace A/B references in reasoning with actual path names
    a_label = "LLM + You.com" if position_map["a"] == "youdotcom" else "LLM + Native"
    b_label = "LLM + You.com" if position_map["b"] == "youdotcom" else "LLM + Native"
    def _replace_ab(text):
        if not isinstance(text, str):
            return text
        return (text
            .replace("Answer A", a_label)
            .replace("Answer B", b_label)
            .replace("answer A", a_label)
            .replace("answer B", b_label))

    for key in ["answer_a", "answer_b"]:
        if key in scores and "reasoning" in scores[key]:
            scores[key]["reasoning"] = _replace_ab(scores[key]["reasoning"])

    # Map verdict from A/B to youdotcom/native/comparable
    raw_verdict = scores.get("verdict", "comparable")
    if raw_verdict == "A_better":
        verdict = position_map["a"]  # whoever was in position A
    elif raw_verdict == "B_better":
        verdict = position_map["b"]
    else:
        verdict = "comparable"

    return {
        "youdotcom_scores": scores.get(ydc_key, {}),
        "native_scores": scores.get(native_key, {}),
        "judge_model": judge_model,
        "position_map": position_map,
        "verdict": verdict,
        "verdict_reasoning": _replace_ab(scores.get("verdict_reasoning", "")),
    }


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

    native_costs = calculate_native_costs(native, model_config)
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

    rows = [
        ("Total tokens",
         f"{ydc['total_tokens']:,}",
         f"{native['total_tokens']:,}"),
        ("  Input tokens",
         f"{ydc['input_tokens']:,}",
         f"{native['input_tokens']:,}"),
        ("  Output tokens",
         f"{ydc['output_tokens']:,}",
         f"{native['output_tokens']:,}"),
        ("  Search context (est.)",
         f"~{ydc['search_context_tokens']:,}",
         f"~{native['search_context_tokens']:,}"),
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
    if native["total_tokens"] > 0:
        savings = native["total_tokens"] - ydc["total_tokens"]
        savings_pct = (savings / native["total_tokens"]) * 100
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

            ydc_s = judge_result["youdotcom_scores"]
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
                    "youdotcom": "You.com path is better",
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


# ─── Client factory ────────────────────────────────────────────────────────

LLM_TIMEOUT = 300  # seconds — native built-in search can take minutes on complex queries

def _create_client(model_config: dict):
    """Create the appropriate API client for the given provider.

    Returns an Anthropic or OpenAI client, raising if the API key is missing.
    """
    provider = model_config["provider"]

    if provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set.")
        return Anthropic(api_key=api_key, timeout=LLM_TIMEOUT)

    elif provider == "openai":
        # Support optional base_url/api_key_env overrides for OpenAI-compatible providers (e.g. Together AI).
        api_key_env = model_config.get("api_key_env", "OPENAI_API_KEY")
        api_key = os.environ.get(api_key_env, "")
        if not api_key:
            raise ValueError(f"{api_key_env} not set.")
        kwargs = {"api_key": api_key, "timeout": LLM_TIMEOUT}
        if model_config.get("base_url"):
            kwargs["base_url"] = model_config["base_url"]
        return OpenAI(**kwargs)

    elif provider == "kimi":
        api_key = os.environ.get("MOONSHOT_API_KEY", "")
        if not api_key:
            raise ValueError("MOONSHOT_API_KEY not set.")
        return OpenAI(api_key=api_key, base_url="https://api.moonshot.ai/v1", timeout=LLM_TIMEOUT)

    elif provider == "qwen":
        api_key = os.environ.get("DASHSCOPE_API_KEY", "")
        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY not set.")
        # Standard DashScope international endpoint — required for enable_search.
        # The MaaS workspace endpoint (DASHSCOPE_BASE_URL) does NOT support enable_search.
        # Both endpoints use the same DASHSCOPE_API_KEY.
        return OpenAI(
            api_key=api_key,
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            timeout=LLM_TIMEOUT,
        )

    else:
        raise ValueError(f"Unknown provider: {provider}")


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

    client = _create_client(model_config)

    print(f"Comparison: {model_config['model']} + You.com vs. {model_config['model']} + Native Search")
    print(f"Query:      \"{query}\"")
    print(f"Judge:      {'OFF' if skip_judge else model_config['judge'] + ' (cross-model, blind)'}")
    print(f"Verbose:    {'ON' if is_verbose() else 'OFF'}")
    print()

    # Run both paths
    print("Running You.com path...")
    ydc_stats = run_youdotcom(query, client, model_config)
    print(f"  Done ({ydc_stats['total_tokens']:,} tokens, {ydc_stats['latency_ms']:.0f}ms)")

    print("Running Native path...")
    native_stats = run_native(query, client, model_config)
    print(f"  Done ({native_stats['total_tokens']:,} tokens, {native_stats['latency_ms']:.0f}ms)")

    # Run judge evaluation
    judge_result = None
    if not skip_judge:
        judge_model = model_config["judge"]
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
