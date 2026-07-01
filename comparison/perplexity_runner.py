"""
perplexity_runner.py — Claude via Perplexity Agent API.

Perplexity's Agent API is a BYOLLM (bring-your-own-LLM) research platform:
  - You specify any frontier model (here: anthropic/claude-sonnet-4-6)
  - Claude runs inside Perplexity's infrastructure with access to web_search + fetch_url
  - Perplexity handles parallel search, source ranking, and result aggregation
  - Single API call returns synthesized answer + sources + token cost breakdown

Key differentiator vs. running Claude directly against You.com Search:
  - Traditional:   Claude (our infra) → serial tool calls → You.com snippets
  - Agent API:     Claude (Perplexity infra) → web_search + fetch_url → richer retrieval

Reference: https://docs.perplexity.ai/docs/agent-api/quickstart

API: POST https://api.perplexity.ai/v1/agent
     Authorization: Bearer $PERPLEXITY_API_KEY
     model: "anthropic/claude-sonnet-4-6"
     tools: web_search, fetch_url

Response cost object: usage.cost.total_cost (USD)

Requires:
    pip install requests python-dotenv
    PERPLEXITY_API_KEY in env.txt / .env

Run standalone:
    python perplexity_runner.py "Who won the 2026 men's hockey Olympic gold?"
"""

import json
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

AGENT_ENDPOINT = "https://api.perplexity.ai/v1/agent"
AGENT_TIMEOUT = 300  # Agent API can take 1-3 minutes for deep queries

PERPLEXITY_SAC_CONFIG = {
    "display_name": "Claude + Perplexity Agent API",
    "model": "anthropic/claude-sonnet-4-6",
    "sac_cost_per_call": 0.05,  # fallback est.; actual cost in usage.cost.total_cost
}

AGENT_INSTRUCTIONS = (
    "You are a research assistant. Use your web search and URL fetch tools to answer "
    "questions thoroughly and accurately.\n\n"
    "Research strategy:\n"
    "  1. Fan out: run multiple targeted search queries in parallel\n"
    "  2. Dig deeper: fetch full content from the most relevant URLs\n"
    "  3. Adapt: identify coverage gaps and run follow-up searches\n"
    "  4. Synthesize: write a comprehensive answer with citations [1], [2], etc."
)


# ─── Agent API runner ────────────────────────────────────────────────────────

def run_perplexity_sac(question: str, api_key: str, on_progress=None) -> dict:
    """Run a query through Claude inside Perplexity's Agent API with streaming.

    Uses stream=True so output items (search_results, fetch_url_results, message)
    arrive incrementally — progress messages fire in real time rather than only
    after the full response lands.

    Returns a stats dict compatible with the SAC comparison module.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": PERPLEXITY_SAC_CONFIG["model"],
        "input": question,
        "instructions": AGENT_INSTRUCTIONS,
        "tools": [
            {"type": "web_search"},
            {"type": "fetch_url"},
        ],
        "max_output_tokens": 4096,
    }

    stats = {
        "path": "Claude (Perplexity Agent API)",
        "model": PERPLEXITY_SAC_CONFIG["model"],
        "total_tokens": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "search_context_tokens": 0,
        "api_calls": 1,
        "search_calls": 0,
        "sandbox_executions": 0,
        "hit_round_limit": False,
        "sources": [],
        "latency_ms": 0.0,
        "answer": "",
        "actual_cost": None,
    }

    if on_progress:
        on_progress("Calling Perplexity Agent API (Claude + web_search + fetch_url)...")

    t0 = time.perf_counter()

    payload["stream"] = True

    try:
        resp = requests.post(
            AGENT_ENDPOINT,
            json=payload,
            headers=headers,
            timeout=AGENT_TIMEOUT,
            stream=True,
        )
        resp.raise_for_status()
    except requests.Timeout:
        raise RuntimeError(f"Perplexity Agent API timed out after {AGENT_TIMEOUT}s")
    except requests.HTTPError as e:
        status = e.response.status_code if e.response is not None else "unknown"
        body = ""
        try:
            body = e.response.json().get("message", "") or e.response.text[:200]
        except Exception:
            pass
        raise RuntimeError(f"Perplexity Agent API returned HTTP {status}: {body}")
    except requests.RequestException as e:
        raise RuntimeError(f"Request failed: {e}")

    # ── Parse named SSE stream ──
    # Each event is a pair of lines:
    #   event: <event_type>
    #   data:  <json_payload>
    #
    # Key event types:
    #   response.reasoning.search_queries  → queries Claude decided to run
    #   response.reasoning.search_results  → raw results returned
    #   response.output_item.done          → completed item (search_results or message)
    #   response.output_text.delta         → streaming answer text chunk
    #   response.completed                 → final response with usage

    answer_parts: list[str] = []
    sources: list[str] = []
    seen_urls: set[str] = set()
    current_event: str = ""

    try:
        for raw_line in resp.iter_lines():
            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line

            if line.startswith("event: "):
                current_event = line[7:].strip()
                continue

            if not line.startswith("data: "):
                continue

            payload_str = line[6:].strip()
            if payload_str == "[DONE]":
                break
            try:
                ev = json.loads(payload_str)
            except json.JSONDecodeError:
                continue

            # ── Queries Claude decided to run ──
            if current_event == "response.reasoning.search_queries":
                queries = ev.get("queries", [])
                if on_progress and queries:
                    q_str = " · ".join(f'"{q}"' for q in queries)
                    on_progress(f"Searching: {q_str}")

            # ── Search results returned ──
            elif current_event == "response.reasoning.search_results":
                results = ev.get("results", [])
                if on_progress and results:
                    on_progress(f"  → {len(results)} result(s) returned")

            # ── Output item completed (search_results or fetch_url_results) ──
            elif current_event == "response.output_item.done":
                item = ev.get("item", {})
                itype = item.get("type", "")

                if itype == "search_results":
                    stats["search_calls"] += 1
                    for r in item.get("results", []):
                        url = r.get("url", "")
                        if url and url not in seen_urls:
                            seen_urls.add(url)
                            sources.append(url)

                elif itype == "fetch_url_results":
                    fetched = item.get("results", [])
                    if on_progress and fetched:
                        on_progress(f"Fetched {len(fetched)} URL(s)")
                    for r in fetched:
                        url = r.get("url", "")
                        if url and url not in seen_urls:
                            seen_urls.add(url)
                            sources.append(url)

                elif itype == "message":
                    for part in item.get("content", []):
                        text = part.get("text", "") if isinstance(part, dict) else ""
                        if text:
                            answer_parts.append(text)
                    for ann in item.get("annotations", []):
                        url = ann.get("url", "")
                        if url and url not in seen_urls:
                            seen_urls.add(url)
                            sources.append(url)

            # ── Answer text streaming (delta chunks ignored; full text from output_item.done) ──
            elif current_event == "response.output_text.delta":
                pass

            # ── Final event with usage ──
            elif current_event == "response.completed":
                response_obj = ev.get("response", ev)
                usage = response_obj.get("usage", {})
                stats["input_tokens"] = usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0)
                stats["output_tokens"] = usage.get("output_tokens", 0) or usage.get("completion_tokens", 0)
                stats["total_tokens"] = stats["input_tokens"] + stats["output_tokens"]
                if not stats["total_tokens"]:
                    stats["total_tokens"] = usage.get("total_tokens", 0)
                cost_obj = usage.get("cost")
                if isinstance(cost_obj, dict):
                    stats["actual_cost"] = cost_obj.get("total_cost")
                elif isinstance(cost_obj, (int, float)):
                    stats["actual_cost"] = cost_obj

    except (requests.RequestException, requests.exceptions.ChunkedEncodingError) as e:
        raise RuntimeError(f"Stream interrupted: {e}")
    finally:
        stats["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    stats["answer"] = "\n\n".join(answer_parts).strip()
    stats["sources"] = sources

    if on_progress:
        on_progress(
            f"Done: {stats['total_tokens']:,} tokens, "
            f"{stats['search_calls']} search batch(es), "
            f"{len(sources)} sources, "
            f"{stats['latency_ms'] / 1000:.1f}s"
        )

    return stats


# ─── CLI ────────────────────────────────────────────────────────────────────

def main():
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        print("Usage: python perplexity_runner.py \"your question here\"")
        print(f"  Model: {PERPLEXITY_SAC_CONFIG['model']} via Perplexity Agent API")
        sys.exit(1)

    api_key = os.environ.get("PERPLEXITY_API_KEY", "")
    if not api_key:
        print("Error: PERPLEXITY_API_KEY not set")
        sys.exit(1)

    print(f"Model:  {PERPLEXITY_SAC_CONFIG['model']} via Perplexity Agent API")
    print(f"Query:  \"{question}\"")
    print()

    try:
        stats = run_perplexity_sac(
            question, api_key,
            on_progress=lambda m: print(f"  {m}"),
        )
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print()
    print(f"Answer:\n{stats['answer']}")
    print()
    print(f"Tokens:         {stats['total_tokens']:,} ({stats['input_tokens']:,} in, {stats['output_tokens']:,} out)")
    print(f"Search batches: {stats['search_calls']}")
    print(f"Sources:        {len(stats['sources'])}")
    print(f"Latency:        {stats['latency_ms']:,.0f}ms")
    if stats["actual_cost"] is not None:
        print(f"Actual cost:    ${stats['actual_cost']:.6f}")
    for i, url in enumerate(stats["sources"], 1):
        print(f"  [{i}] {url}")


if __name__ == "__main__":
    main()
