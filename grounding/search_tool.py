"""
search_tool.py — You.com Search as a callable tool for LLM agents.

SOURCE OF TRUTH for shared constants used across the entire codebase:
  - SYSTEM_PROMPT: the canonical system prompt for all LLM paths
  - MAX_TOKENS: default max output tokens per LLM call

All other modules (base_agent.py, run.py, compare.py, benchmark_runner.py)
import these from here. Do NOT redefine them elsewhere.

This module also:
  1. Defines the tool schema that gets passed to any LLM's function-calling API
  2. Implements the actual search execution when the LLM invokes the tool

The LLM decides WHEN to search and WHAT to search for.
This module handles the HOW.

Usage:
    from search_tool import TOOL_SCHEMA, SYSTEM_PROMPT, MAX_TOKENS, execute_search

    # Pass TOOL_SCHEMA to your LLM's tool/function definition
    # When the LLM calls the tool, run execute_search(tool_args)
"""

import logging
import os
import time
import requests
from typing import Optional

logger = logging.getLogger(__name__)

# ─── Configuration ───────────────────────────────────────────────────────────

SEARCH_ENDPOINT = "https://ydc-index.io/v1/search"
SEARCH_TIMEOUT_SECONDS = 30
MAX_LIVECRAWL_CHARS = 3000  # Per-result truncation to avoid blowing up context

# Integration interface identifier. Set once at startup by the entry point
# (demo.py, your FastAPI app, etc.) so tool logs always know how search was called.
# Values: "direct_api", "mcp", "langchain", "agent_skill"
INTEGRATION_INTERFACE = "direct_api"


def set_interface(interface: str) -> None:
    """Set the integration interface label for tool call logs.

    Call this once at startup. Valid values:
        direct_api    — raw HTTP to ydc-index.io/v1/search (default)
        mcp           — via You.com MCP server
        langchain     — via langchain-youdotcom
        agent_skill   — via .md agent skill file
    """
    global INTEGRATION_INTERFACE
    INTEGRATION_INTERFACE = interface


def is_verbose() -> bool:
    """Check if verbose logging is enabled.

    Toggle system-wide:
        export GROUNDING_VERBOSE=1   # on
        export GROUNDING_VERBOSE=0   # off (default)

    Or set in .env:
        GROUNDING_VERBOSE=1
    """
    return os.environ.get("GROUNDING_VERBOSE", "0") == "1"


# ─── Canonical constants (shared across the codebase) ─────────────────────────
# Other modules import these. Change them HERE, not in downstream files.

SYSTEM_PROMPT = (
    "You are a helpful research assistant with access to real-time web search. "
    "When the user asks a question that requires current information, recent "
    "events, specific data, or anything that might have changed since your "
    "training cutoff, use your web search tool to look it up. You can search "
    "multiple times if you need to compare information or dig deeper.\n\n"
    "When you use search results in your answer, cite your sources by number "
    "[1], [2], etc. If the search results don't contain what you need, say so "
    "honestly rather than guessing."
)

# TUNING LEVER: GPT-5.5 requires explicit search tool coaching that other models
# don't need (they handle include_domains/livecrawl correctly without guidance).
# If other models regress on tool usage in the future, consider promoting some of
# these rules back into SYSTEM_PROMPT, or adding per-model variants here.
_GPT_55_SEARCH_RULES = (
    "\n\nIMPORTANT — search query rules:\n"
    "- Write plain natural-language queries only. Do NOT embed Google-style operators "
    "in the query string (site:, filetype:, inurl:, intitle:, OR, AND, quoted phrases, "
    "etc.) — they are not supported and will cause the search to fail.\n"
    "- Use the include_domains parameter freely when you want results from specific "
    "sources (e.g. include_domains='reuters.com,bloomberg.com'). This is the correct "
    "and preferred way to scope a search — use it often.\n"
    "- Use exclude_domains to filter out low-quality sources.\n"
    "- Set livecrawl=true for questions requiring detailed, structured data (tables, "
    "specs, financials, statistics) where snippets alone are insufficient. Use it "
    "selectively on the most relevant sources."
)

MAX_TOKENS = 4096       # Default max output tokens per LLM call
MAX_TOOL_ROUNDS = 15    # Max LLM ↔ tool-use round-trips before we force a final answer

# Estimated baseline input tokens (system prompt + user query) for single-call native
# search paths where the model handles search internally and returns one response.
# Multi-round paths (YDC, chat.completions) measure this exactly from round 0.
# Single-call paths (Anthropic native, OpenAI Responses native) cannot measure it
# because there is no pre-search round — this constant is used as the approximation.
NATIVE_SEARCH_BASELINE_TOKENS = 300


def get_system_prompt(model: str = "") -> str:
    """Return the system prompt with the current date/time appended.

    This gives the LLM temporal context so it can interpret relative
    references like 'yesterday', 'this week', 'latest', etc. and
    construct better search queries.

    All runtime callers should use this instead of the static SYSTEM_PROMPT.
    The static constant is kept for imports that need the base text
    (e.g., A/B testing, display).

    Args:
        model: The model name. GPT-5.5 gets additional search tool coaching.
    """
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%A, %B %d, %Y at %H:%M UTC")
    extra = _GPT_55_SEARCH_RULES if "gpt-5.5" in model.lower() else ""
    return f"{SYSTEM_PROMPT}{extra}\n\nCurrent date and time: {timestamp}"


# ─── Tool schema (OpenAI-compatible format) ──────────────────────────────────
# This is the function definition the LLM sees. It works as-is with OpenAI,
# Qwen (DashScope), Kimi (Moonshot), and Llama (Together/Ollama).
# For Claude, base_agent.py converts this to Anthropic's tool format.

TOOL_SCHEMA = {
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
                    "description": (
                        "Plain natural-language search query. Be specific and descriptive. "
                        "Do NOT use Google operators (site:, filetype:, inurl:, OR, AND, etc.) — "
                        "they are unsupported and will cause errors. To restrict to specific "
                        "domains use the include_domains parameter instead."
                    ),
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
                    "description": (
                        "Comma-separated list of domains to restrict results to "
                        "(e.g. 'reuters.com,bloomberg.com'). Use this instead of "
                        "site: in the query string."
                    ),
                },
                "exclude_domains": {
                    "type": "string",
                    "description": (
                        "Comma-separated list of domains to exclude from results "
                        "(e.g. 'reddit.com,quora.com'). Use this instead of "
                        "-site: in the query string."
                    ),
                },
            },
            "required": ["query"],
        },
    },
}

# OpenAI Responses API tool format — no outer "function" wrapper.
# Tool is named "you_search" (not "web_search") so DashScope/Qwen does not intercept
# it and route to native search instead of calling our function.
TOOL_SCHEMA_RESPONSES = {
    "type": "function",
    "name": "you_search",
    "description": TOOL_SCHEMA["function"]["description"],
    "parameters": TOOL_SCHEMA["function"]["parameters"],
}

# Claude uses a different tool format. This is the converted version.
TOOL_SCHEMA_ANTHROPIC = {
    "name": "web_search",
    "description": TOOL_SCHEMA["function"]["description"],
    "input_schema": TOOL_SCHEMA["function"]["parameters"],
}


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _get_api_key(override: Optional[str] = None) -> str:
    """Resolve the You.com API key. Reads env at call time, not import time."""
    return override or os.environ.get("YDC_API_KEY", "")


def extract_urls(search_result: str) -> list[str]:
    """Pull source URLs from formatted search results.

    Used by agents to populate the 'sources' field in their response.
    """
    return [
        line[5:].strip()
        for line in search_result.split("\n")
        if line.startswith("URL: ")
    ]


def extract_search_uuid(search_result: str) -> str:
    """Pull the search_uuid from formatted search results.

    You.com returns a search_uuid per request. Since You.com has zero data
    retention (ZDR), the calling agent is responsible for logging this UUID
    if audit traceability is needed.
    """
    for line in search_result.split("\n"):
        if line.startswith("Search-UUID: "):
            return line[len("Search-UUID: "):].strip()
    return ""


def build_tool_log_entry(
    tool_args: dict,
    search_result: str,
    elapsed_ms: float = 0.0,
) -> dict:
    """Build a structured log entry for a single tool call.

    Used by agents to populate the 'tool_calls' field in their response.
    Includes search_uuid for audit traceability (caller's responsibility
    under You.com's ZDR policy).
    """
    return {
        "query": tool_args.get("query", ""),
        "livecrawl": tool_args.get("livecrawl", False),
        "count": tool_args.get("count", 5),
        "freshness": tool_args.get("freshness"),
        "include_domains": tool_args.get("include_domains"),
        "exclude_domains": tool_args.get("exclude_domains"),
        "search_uuid": extract_search_uuid(search_result),
        "interface": INTEGRATION_INTERFACE,
        "elapsed_ms": round(elapsed_ms, 1),
        "result_count": _count_results(search_result),
        "results_preview": search_result[:200],
    }


def _count_results(search_result: str) -> int:
    """Count numbered results in formatted search output."""
    return sum(1 for line in search_result.split("\n") if line.startswith("[") and "] " in line[:6])


def format_tool_log(entry: dict) -> str:
    """Format a single tool log entry for human-readable console output.

    Used by demo.py and any caller that wants to print tool call details.
    Only produces output worth showing — skips None/empty fields.
    """
    lines = [
        f"  ┌─ Tool Call: web_search",
        f"  │ Interface:   {entry.get('interface', 'unknown')}",
        f"  │ Query:       \"{entry.get('query', '')}\"",
        f"  │ Mode:        {'livecrawl' if entry.get('livecrawl') else 'snippets'}",
        f"  │ Count:       {entry.get('count', 5)} requested, {entry.get('result_count', '?')} returned",
    ]
    if entry.get("freshness"):
        lines.append(f"  │ Freshness:   {entry['freshness']}")
    if entry.get("include_domains"):
        lines.append(f"  │ Domains(+):  {entry['include_domains']}")
    if entry.get("exclude_domains"):
        lines.append(f"  │ Domains(-):  {entry['exclude_domains']}")
    if entry.get("search_uuid"):
        lines.append(f"  │ Search-UUID: {entry['search_uuid']}")
    if entry.get("elapsed_ms"):
        lines.append(f"  │ Latency:     {entry['elapsed_ms']}ms")
    lines.append(f"  └─")
    return "\n".join(lines)


# ─── Tool execution ─────────────────────────────────────────────────────────

def execute_search(
    tool_args: dict,
    api_key: Optional[str] = None,
) -> str:
    """Execute a You.com search and return formatted results.

    This is called when the LLM invokes the web_search tool.

    Args:
        tool_args: The arguments the LLM passed to the tool call.
                   Must contain 'query', optionally 'count', 'livecrawl', etc.
        api_key:   Override for YDC_API_KEY env var.

    Returns:
        A formatted string of search results, ready for the LLM to read.
    """
    key = _get_api_key(api_key)
    if not key:
        return "Error: YDC_API_KEY not set. Get one at https://you.com/platform"

    query = tool_args.get("query", "")
    if not query:
        return "Error: 'query' is required but was empty."

    try:
        count = max(1, min(20, int(tool_args.get("count", 5))))
    except (TypeError, ValueError):
        count = 5
    livecrawl = tool_args.get("livecrawl", False)
    freshness = tool_args.get("freshness")
    include_domains = tool_args.get("include_domains")

    extras = []
    if livecrawl: extras.append("livecrawl")
    if freshness: extras.append(f"freshness={freshness}")
    if include_domains: extras.append(f"include_domains={include_domains}")
    logger.info("YDC search: %r count=%d %s", query, count, " ".join(extras))
    exclude_domains = tool_args.get("exclude_domains")

    params: dict = {"query": query, "count": count}
    if livecrawl:
        params["livecrawl"] = "web"
        params["livecrawl_formats"] = "markdown"
    if freshness:
        params["freshness"] = freshness
    if include_domains:
        params["include_domains"] = include_domains
    if exclude_domains:
        params["exclude_domains"] = exclude_domains

    try:
        resp = requests.get(
            SEARCH_ENDPOINT,
            headers={"X-API-Key": key},
            params=params,
            timeout=SEARCH_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
    except requests.ConnectionError:
        logger.error("Could not connect to You.com Search API at %s", SEARCH_ENDPOINT)
        return f"Search failed: could not connect to {SEARCH_ENDPOINT}"
    except requests.Timeout:
        logger.error("You.com Search API timed out after %ds", SEARCH_TIMEOUT_SECONDS)
        return f"Search failed: request timed out after {SEARCH_TIMEOUT_SECONDS}s"
    except requests.HTTPError as e:
        status = e.response.status_code if e.response is not None else "unknown"
        body = ""
        if e.response is not None:
            try:
                body = e.response.json()
            except Exception:
                body = e.response.text[:300]
        logger.error("You.com Search API returned HTTP %s for query: %r — %s", status, query, body)
        return f"Search failed: HTTP {status}"
    except requests.RequestException as e:
        logger.error("You.com Search API request failed: %s", e)
        return "Search failed: network error"

    try:
        data = resp.json()
    except (requests.exceptions.JSONDecodeError, ValueError):
        logger.error("You.com Search API returned non-JSON response (status=%s)", resp.status_code)
        return "Search failed: unexpected response format"
    raw = data.get("results", {})
    # API shape: {"results": {"web": [...]}} is standard; guard against flat list responses
    results = raw.get("web", []) if isinstance(raw, dict) else (raw if isinstance(raw, list) else [])

    if not results:
        return f"No results found for: {query}"

    # Format results for the LLM to read
    parts = []
    for i, hit in enumerate(results, 1):
        title = hit.get("title", "")
        url = hit.get("url", "")
        snippets = "\n".join(hit.get("snippets", []))

        # If livecrawl returned full content, prefer that
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


# ─── Citation injection helpers ──────────────────────────────────────────────
# Used by native search methods on agent classes to embed [N] markers in answers.
# Shared here so agent files don't depend on compare.py.

def inject_citations_at_positions(text: str, insertions: dict) -> str:
    """Insert [N] citation markers into text at the given char end-positions.

    insertions: dict mapping char index → list of citation numbers to insert.
    Providers that return character-offset annotations (Anthropic, OpenAI) use this
    to embed inline citations before returning the answer to callers.
    """
    if not insertions:
        return text
    result_chars = []
    for i, ch in enumerate(text):
        result_chars.append(ch)
        if i + 1 in insertions:
            nums = sorted(insertions[i + 1])
            result_chars.append("".join(f"[{n}]" for n in nums))
    return "".join(result_chars)


def inject_citations(text: str, sources: list) -> str:
    """Replace [N] citation markers with markdown links to source URLs."""
    import re
    for i, url in enumerate(sources, 1):
        text = re.sub(rf'\[{i}\]', f'[[{i}]]({url})', text)
    return text


def make_progress_reporter(prefix: str, on_progress=None):
    """Return a _notify(msg) closure for agent native-search progress reporting."""
    verbose = is_verbose()
    def _notify(msg):
        if verbose:
            print(f"  [{prefix}] {msg}")
        if on_progress:
            on_progress(msg)
    return _notify


def make_elapsed_timer():
    """Return (t0, _elapsed) where _elapsed() formats seconds since t0 as a string."""
    t0 = time.perf_counter()
    def _elapsed():
        return f"{(time.perf_counter() - t0):.1f}s"
    return t0, _elapsed


# ─── Quick self-test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or input("Search query: ")
    print(f"Searching for: {query}\n")
    result = execute_search({"query": query, "count": 3})
    print(result[:800])
