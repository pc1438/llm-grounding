"""
judge.py — Blind cross-model judge for search-grounded answer evaluation.

Scores two answers (You.com path vs. native search path) across four dimensions
using a cross-model judge (GPT judges Claude, Claude judges GPT). Position is
randomized to prevent A/B ordering bias.

    from judge import run_judge

    result = run_judge(
        question="Who won the 2026 NBA Finals?",
        answer_ydc="...",
        answer_native="...",
        judge_model="openai",        # or "claude"
        sources_ydc=["https://..."],
    )
    # result: {ydc_scores, native_scores, judge_model, position_map, verdict, verdict_reasoning}

Supported judge models:
    "openai"  → GPT-5.4 via OPENAI_API_KEY
    "claude"  → Claude Sonnet 4.6 via ANTHROPIC_API_KEY

The judge creates its own API clients — it is self-contained and independent
of the grounding agent stack.
"""

import json
import logging
import os
import random
from pathlib import Path

from anthropic import Anthropic
from openai import OpenAI

def _judge_model_id(provider: str, fallback: str) -> str:
    """Read the agent_default model for a provider from pricing.json."""
    try:
        pricing_path = Path(__file__).parent / "pricing.json"
        with open(pricing_path) as f:
            data = json.load(f)
        for entry in data.get("models", {}).values():
            if entry.get("provider") == provider and entry.get("agent_default"):
                return entry["model"]
    except Exception:
        pass
    return fallback

_OPENAI_JUDGE_MODEL = _judge_model_id("openai", "gpt-5.4")
_CLAUDE_JUDGE_MODEL = _judge_model_id("anthropic", "claude-sonnet-4-6")

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


def _answer_with_sources(answer: str, sources: list) -> str:
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
    sources_ydc: list = None,
    sources_native: list = None,
) -> dict:
    """Score two answers using a blind cross-model judge.

    Args:
        question:       The original search query.
        answer_ydc:     Answer from the LLM + You.com Search path.
        answer_native:  Answer from the LLM + native web search path.
        judge_model:    "openai" (GPT-5.4) or "claude" (Claude Sonnet 4.6).
        sources_ydc:    Optional list of URLs from the You.com path, appended
                        to the answer so the judge can evaluate citation traceability.
        sources_native: Optional list of URLs from the native path (usually already
                        inline in the answer text, but included for symmetry).

    Returns:
        dict with keys: ydc_scores, native_scores, judge_model, position_map,
                        verdict ("ydc" | "native" | "comparable"), verdict_reasoning.
        On failure: dict with key "error" describing what went wrong.
    """
    answer_ydc = answer_ydc or ""
    answer_native = answer_native or ""
    answer_ydc = _answer_with_sources(answer_ydc, sources_ydc or [])
    answer_native = _answer_with_sources(answer_native, sources_native or [])

    # Randomize position to prevent A/B ordering bias
    if random.random() < 0.5:
        answer_a, answer_b = answer_ydc, answer_native
        position_map = {"a": "ydc", "b": "native"}
    else:
        answer_a, answer_b = answer_native, answer_ydc
        position_map = {"a": "native", "b": "ydc"}

    judge_prompt = (
        f"Question: {question}\n\n"
        f"--- Answer A ---\n{answer_a}\n\n"
        f"--- Answer B ---\n{answer_b}\n\n"
        f"Evaluate both answers. Respond with JSON only."
    )

    if judge_model == "openai":
        openai_key = os.environ.get("OPENAI_API_KEY", "")
        if not openai_key:
            return {"error": "OPENAI_API_KEY not set — skipping judge evaluation"}
        client = OpenAI(api_key=openai_key)
        response = client.chat.completions.create(
            model=_OPENAI_JUDGE_MODEL,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": judge_prompt},
            ],
            max_completion_tokens=1024,
            temperature=0,
        )
        raw = response.choices[0].message.content

    elif judge_model == "claude":
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not anthropic_key:
            return {"error": "ANTHROPIC_API_KEY not set — skipping judge evaluation"}
        client = Anthropic(api_key=anthropic_key)
        response = client.messages.create(
            model=_CLAUDE_JUDGE_MODEL,
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

    try:
        clean = raw.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[1]
            clean = clean.rsplit("```", 1)[0]
        scores = json.loads(clean)
    except (json.JSONDecodeError, IndexError):
        logging.debug("Judge returned invalid JSON (first 200 chars): %s", raw[:200])
        return {"error": "Judge returned invalid JSON — check logs for details"}

    ydc_key = "answer_a" if position_map["a"] == "ydc" else "answer_b"
    native_key = "answer_a" if position_map["a"] == "native" else "answer_b"

    a_label = "LLM + You.com" if position_map["a"] == "ydc" else "LLM + Native"
    b_label = "LLM + You.com" if position_map["b"] == "ydc" else "LLM + Native"

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

    raw_verdict = scores.get("verdict", "comparable")
    if raw_verdict == "A_better":
        verdict = position_map["a"]
    elif raw_verdict == "B_better":
        verdict = position_map["b"]
    else:
        verdict = "comparable"

    return {
        "ydc_scores": scores.get(ydc_key, {}),
        "native_scores": scores.get(native_key, {}),
        "judge_model": judge_model,
        "position_map": position_map,
        "verdict": verdict,
        "verdict_reasoning": _replace_ab(scores.get("verdict_reasoning", "")),
    }
