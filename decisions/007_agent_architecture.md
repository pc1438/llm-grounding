# Agent Architecture

**Last updated:** 2026-06-30
**Owner:** Chak Pothina
**Status:** Decided — implementation in progress

---

## Problem

The tool-use loop was implemented in three separate places:

1. `base_agent.py` — used by agent classes (programmatic callers)
2. `run.py` — used by the UI server and CLI (SSE streaming)
3. `compare.py` — used by the comparison CLI and benchmark runner

Same logic, three copies. Any change (e.g. switching GPT to the Responses API) had to be made in all three. This is what caused the Responses API changes to land in the wrong place initially.

---

## Decision: 4 Layers, Clear Ownership

```
search_tool.py          — You.com search execution + tool schemas
base_agent.py           — Loop implementations, organized by API shape
agents/<provider>.py    — Client setup, model defaults, provider quirks
run.py                  — SSE streaming wrapper only
compare.py              — Orchestration, judge, file output only
```

---

## Layer Responsibilities

### `search_tool.py`
Owns everything about the You.com search API:
- HTTP execution (`execute_search`)
- Tool schema definitions (`TOOL_SCHEMA`, `TOOL_SCHEMA_ANTHROPIC`, `TOOL_SCHEMA_RESPONSES`)
- Result formatting, URL extraction, log building
- Constants: `MAX_TOKENS`, `MAX_TOOL_ROUNDS`, `SYSTEM_PROMPT`

Standalone and self-contained — can be cloned and used without the rest of the app.
If the search implementation changes (SDK, MCP), only this file changes.

### `base_agent.py`
Owns the tool-use loop, organized by **API shape** — not by provider:

| Class | API shape | Providers |
|---|---|---|
| `BaseAgent` | ABC — defines `ask()` contract | all |
| `AnthropicAgent` | Anthropic Messages API | Claude |
| `OpenAICompatibleAgent` | OpenAI chat.completions | Kimi, Llama |
| `OpenAIResponsesAgent` | OpenAI Responses API + parallel tools | GPT, Qwen |

Organized by API shape because that's the stable boundary. A new provider that speaks the Responses API (e.g. a future Mistral endpoint) slots into `OpenAIResponsesAgent` with a new agent file — no changes to `base_agent.py`.

**What belongs here:** loop logic, token counting, error handling, `on_progress` callback firing.

**What does NOT belong here:** client instantiation, API keys, base URLs, model defaults, provider quirks (e.g. Qwen's `you_search` tool name). Those belong in the agent file.

### `agents/<provider>.py`
Each file is the authoritative source for one LLM provider. Someone reading `qwen_agent.py` should understand the full picture for Qwen without reading anything else.

Each file owns:
- Client instantiation (base URL, API key, SDK options)
- Default model name and available model variants
- Which base class to inherit (determines loop implementation)
- Any provider-specific overrides (tool name, extra params)
- Explanation of provider quirks in comments

Example of what lives in `qwen_agent.py` and nowhere else:
```python
# Tool must be named 'you_search' — DashScope intercepts any function named
# 'web_search' and routes it to native search instead of calling our function.
# Confirmed via empirical test 2026-06-30. See decisions/006.
TOOL_NAME_OVERRIDE = "you_search"
```

Example of what lives in `kimi_agent.py` and nowhere else:
```python
# Kimi uses chat.completions — Moonshot's /v1/responses returns 404.
# Confirmed via empirical test + docs review 2026-06-30. See decisions/006.
```

### `run.py`
Thin streaming wrapper. Its only job is:
1. Instantiate the right agent for the requested model
2. Convert `on_progress` callbacks from `ask()` into generator-yielded SSE events
3. Yield the final `done` event with stats

No loop logic. No API calls. No search execution. If the loop changes, `run.py` doesn't change.

### `compare.py`
Orchestration only:
- Runs YDC path and native path (calls `agent.ask()`)
- Runs the judge
- Scores and formats results
- Writes output files

No loop logic. Calls agent classes the same way a CLI user would.

---

## The `on_progress` Contract

`ask()` accepts an optional `on_progress` callback:

```python
def ask(self, question: str, on_progress=None) -> dict:
```

The callback is fired at each meaningful step with a plain string message:
```python
on_progress("Search 1: current gold price (1.2s)")
on_progress("Results received (340ms)")
on_progress("Generating answer... (3.1s)")
```

**`run.py`** converts these into SSE events for the UI.
**`compare.py`** uses them for CLI progress output.
**Programmatic callers** pass `None` and get the final dict only.

The callback is fire-and-forget. The agent does not wait for acknowledgement. Errors in the callback must not propagate into the agent loop.

---

## The `ask()` Return Dict

All agents return the same shape:

```python
{
    "answer":      str,           # final text response
    "sources":     list[str],     # URLs from search results
    "tool_calls":  list[dict],    # each search: query, results, latency
    "model":       str,           # model ID confirmed by API response
    "tokens_used": int,           # total input + output tokens
    "token_breakdown": {
        "input":          int,
        "output":         int,
        "search_context": int,    # tokens from accumulated search results
    },
    "search_calls":  int,
    "api_calls":     int,
    "latency_ms":    float,
}
```

`compare.py` maps this to its own stats dict for scoring and file output. The shapes are intentionally kept separate — agent dict is the programmatic API, compare stats dict includes comparison-specific fields (path, cost, judge scores).

---

## Why Not Per-Provider Loop Implementations?

The alternative — each agent file owning its full loop — was considered and rejected for one reason: `OpenAICompatibleAgent` and `OpenAIResponsesAgent` would be duplicated across multiple files (GPT + Qwen share the Responses API loop; Kimi + Llama share chat.completions). Duplication across agent files is worse than a well-named base class, because a loop change would require touching multiple agent files instead of one.

The rule of thumb: if two providers share an API shape, the loop lives in `base_agent.py`. If a provider has a unique quirk within that shape, the quirk lives in the agent file as an override.

---

## Adding a New Provider

1. Create `agents/<provider>_agent.py`
2. Inherit the right base class (`OpenAIResponsesAgent`, `OpenAICompatibleAgent`, or `AnthropicAgent`)
3. Add client setup, model defaults, any quirks
4. Add entry to `pricing.json`
5. Nothing else changes

## Adding a New API Shape (e.g. MCP, You.com SDK)

- New search execution → `search_tool.py`
- New loop pattern → new class in `base_agent.py`
- Agent files that use it → inherit the new class
- `run.py` and `compare.py` → untouched
