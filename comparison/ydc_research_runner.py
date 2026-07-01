"""
ydc_research_runner.py — You.com Research API runner.

The You.com Research API performs multi-step web research internally and returns
a synthesized markdown answer with citations in a single API call. No local LLM
client required — You.com handles search orchestration and synthesis.

API: POST https://api.you.com/v1/research
     X-API-Key: <YDC_API_KEY>
     body: { "input": "...", "research_effort": "deep" }
Response: { "content": "...", "content_type": "text", "sources": [...] }

research_effort levels: lite | standard | deep | exhaustive

Requires:
    pip install requests python-dotenv
    YDC_API_KEY in env.txt / .env

Run standalone:
    python ydc_research_runner.py "Who won the 2026 men's hockey Olympic gold?"
"""

import os
import sys
import time
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv("env.txt") or load_dotenv(".env")
load_dotenv(Path(__file__).parent.parent / "grounding" / "env.txt")
load_dotenv(Path(__file__).parent.parent / "grounding" / ".env")


# ─── Config ─────────────────────────────────────────────────────────────────

RESEARCH_ENDPOINT = "https://api.you.com/v1/research"
RESEARCH_TIMEOUT = 180  # 3 minutes; "deep" effort can be slow

YDC_RESEARCH_CONFIG = {
    "display_name": "You.com Research API",
    "effort": "deep",
    "cost_per_call": 0.05,  # est.; verify at you.com/platform/pricing
}


# ─── Runner ─────────────────────────────────────────────────────────────────

def run_ydc_research(question: str, api_key: str, research_effort: str = "deep", on_progress=None) -> dict:
    """Run a query through You.com Research API.

    One HTTP call — You.com handles multi-step search and synthesis internally.
    Returns a stats dict compatible with the SAC comparison module.

    Note: The Research API does not expose token counts; token fields will be 0.
    """
    headers = {
        "X-API-Key": api_key,
        "Content-Type": "application/json",
    }
    payload = {
        "input": question,
        "research_effort": research_effort,
    }

    stats = {
        "path": "You.com Research API",
        "total_tokens": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "search_context_tokens": 0,
        "tokens_estimated": True,    # API does not return token counts; output_tokens computed from answer length
        "api_calls": 1,
        "search_calls": 1,
        "hit_round_limit": False,
        "sources": [],
        "latency_ms": 0.0,
        "answer": "",
        "research_effort": research_effort,
    }

    if on_progress:
        on_progress(f"Calling You.com Research API (effort={research_effort})...")

    t0 = time.perf_counter()

    try:
        resp = requests.post(
            RESEARCH_ENDPOINT,
            json=payload,
            headers=headers,
            timeout=RESEARCH_TIMEOUT,
        )
        resp.raise_for_status()
    except requests.Timeout:
        raise RuntimeError(f"You.com Research API timed out after {RESEARCH_TIMEOUT}s")
    except requests.HTTPError as e:
        status = e.response.status_code if e.response is not None else "unknown"
        body = ""
        try:
            body = e.response.json().get("message", "") or e.response.text[:200]
        except Exception:
            pass
        raise RuntimeError(f"You.com Research API returned HTTP {status}: {body}")
    except requests.RequestException as e:
        raise RuntimeError(f"Request failed: {e}")

    stats["latency_ms"] = (time.perf_counter() - t0) * 1000
    try:
        data = resp.json()
    except (ValueError, requests.exceptions.JSONDecodeError) as e:
        raise RuntimeError(f"You.com Research API returned non-JSON response: {e}")

    # Response is nested: { "output": { "content": "...", "sources": [...] } }
    output = data.get("output", data)  # fallback to top-level if already flat

    stats["answer"] = output.get("content", "").strip()

    # Estimate output tokens from answer length (1 token ≈ 4 chars for English).
    # Input tokens are unknowable — You.com does not expose its internal search context.
    stats["output_tokens"] = len(stats["answer"]) // 4
    stats["total_tokens"] = stats["output_tokens"]

    sources: list[str] = []
    seen_urls: set[str] = set()
    for src in output.get("sources", []):
        url = src.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            sources.append(url)
    stats["sources"] = sources

    if on_progress:
        on_progress(
            f"Done: {len(sources)} sources, "
            f"{stats['latency_ms'] / 1000:.1f}s (effort={research_effort})"
        )

    return stats


# ─── CLI ────────────────────────────────────────────────────────────────────

def main():
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        print("Usage: python ydc_research_runner.py \"your question here\"")
        print(f"  Endpoint: {RESEARCH_ENDPOINT}")
        print(f"  Effort:   {YDC_RESEARCH_CONFIG['effort']}")
        sys.exit(1)

    api_key = os.environ.get("YDC_API_KEY", "")
    if not api_key:
        print("Error: YDC_API_KEY not set")
        sys.exit(1)

    print(f"Endpoint: {RESEARCH_ENDPOINT}")
    print(f"Effort:   {YDC_RESEARCH_CONFIG['effort']}")
    print(f"Query:    \"{question}\"")
    print()

    try:
        stats = run_ydc_research(
            question, api_key,
            research_effort=YDC_RESEARCH_CONFIG["effort"],
            on_progress=lambda m: print(f"  {m}"),
        )
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print()
    print(f"Answer:\n{stats['answer']}")
    print()
    print(f"Sources:  {len(stats['sources'])}")
    print(f"Latency:  {stats['latency_ms']:,.0f}ms")
    for i, url in enumerate(stats["sources"], 1):
        print(f"  [{i}] {url}")


if __name__ == "__main__":
    main()
