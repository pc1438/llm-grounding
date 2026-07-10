# Provider Latency, Timeouts, and Endpoint Findings

**Last updated:** 2026-07-02  
**Owner:** Chak Pothina  
**Status:** Implemented — see compare.py, benchmark_runner.py, server.py

---

## Background

During runtime validation of the deduplication refactor (July 2026), two latency issues
were observed that required code changes and are documented here for future reference.

---

## 1. LLM Timeout Split: YDC vs. Native

### Problem

`compare.py` originally used a single `LLM_TIMEOUT = 300s` for all providers and paths.
gpt-5.5 on the You.com (YDC) path was observed hanging for 5+ minutes without returning
any progress. The app showed "waiting for results" with no tool calls appearing.

### Root Cause

gpt-5.5 is a reasoning model. Before deciding to call a search tool, it runs an internal
chain-of-thought that can take several minutes. The 300s timeout was designed for native
search (where the provider executes search internally and can legitimately take minutes).
Applying the same timeout to the YDC path meant the app would wait up to 5 minutes for
what should be a quick tool-call decision.

### Fix

Lowered `LLM_TIMEOUT` from 300s to 60s in `compare.py`:

```python
LLM_TIMEOUT = 60  # seconds
```

`_create_client()` is used by the YDC path only — native search agents (`ClaudeAgent`,
`OpenAIAgent`, etc.) create their own clients internally and are unaffected by this value.
No bifurcation needed; a single lower value is correct for all YDC callers.

### Implication

If gpt-5.5 (or future reasoning models) genuinely take longer than 60s to produce a
first tool call, they will time out. 60s is the right trade-off — a model that takes
more than a minute to decide what to search has a deeper problem.
Raise `LLM_TIMEOUT` if needed, but investigate why before increasing it.

---

## 2. Kimi Latency Characteristics

### Observation

Kimi's `$web_search` builtin tool executes in under 1 second server-side, but total
query latency is typically 15-20 seconds per query. The time is spent in the inter-round
LLM API calls (Moonshot's response time), not in the search itself.

### Implication

Nothing to fix in code — this is Moonshot API latency. For benchmarking purposes:
Kimi's latency numbers reflect Moonshot API speed, not search quality. Do not use
Kimi latency as a signal for search efficiency.

Note: `LLM_TIMEOUT` in compare.py applies only to the YDC client — Kimi's native search
creates its own client internally and has no externally configurable timeout here.

---

## 3. Qwen Endpoint Split: Domestic vs. International

### Background

DashScope exposes two endpoints that have **different API capabilities** — they are not
interchangeable:

| Endpoint | URL | Supports |
|---|---|---|
| Domestic | `dashscope.aliyuncs.com/compatible-mode/v1` | YDC path (chat.completions tool-use loop with `role="tool"` messages) |
| International | `dashscope-intl.aliyuncs.com/compatible-mode/v1` | Native search (Responses API with `web_search` tool) |

The domestic endpoint does NOT support the Responses API with `web_search` (as of 2026-06).
The international endpoint does NOT support `role="tool"` messages used in the chat.completions
tool-use loop — attempts fail with a 400 `InvalidParameter` error on round 2.

Both endpoints use the same `DASHSCOPE_API_KEY`.

### Current Code

`qwen_agent.py` defines both constants and documents which to use when:

```python
_DASHSCOPE_DOMESTIC_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
_DASHSCOPE_INTL_BASE_URL     = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
```

`BACKENDS["dashscope"]["base_url"]` uses the domestic endpoint for the YDC path.
`run_native_search()` always uses `_DASHSCOPE_INTL_BASE_URL`.

### Key in Use

The key in `env.txt` is a MaaS workspace key (`sk-ws-...`, ap-southeast-1 region).
It works on the international endpoint (`dashscope-intl.aliyuncs.com`).

`compare.py`'s `_create_client()` hardcodes the intl endpoint for Qwen YDC — so both
YDC and native paths use the intl endpoint and both work with this key. Confirmed
via runtime testing 2026-07-02.

`qwen_agent.py`'s `BACKENDS["dashscope"]["base_url"]` still points to the domestic
endpoint (for standalone use). If used standalone with this key, the YDC path
would 401. That path is not exercised by compare.py or server.py.

---

## Open Questions

- [ ] Does gpt-5.5 consistently take >60s on first tool call, or was the 5-minute hang
  a one-off? If consistent, raise `LLM_TIMEOUT` for reasoning models specifically.
- [ ] Will DashScope ever support `role="tool"` on the international endpoint? If yes,
  the two-client split in `run_native_search()` can be removed.
- [ ] `qwen_agent.py` standalone YDC path uses domestic endpoint — would 401 with
  the current MaaS key. Fix if standalone Qwen YDC use is needed.
