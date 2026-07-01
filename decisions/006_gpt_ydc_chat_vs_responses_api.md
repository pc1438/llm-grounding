# GPT + You.com: chat.completions vs. Responses API Approach

**Last updated:** 2026-06-30
**Owner:** Chak Pothina
**Status:** Under evaluation — no code change made yet

---

## Background

The current You.com tool-use path for GPT (in both `compare.py` and `run.py`) uses OpenAI's `chat.completions` API with manual message history management. An alternative approach — based on a reference implementation (`you_tool_sampler.py`) used in You.com's own eval framework — uses OpenAI's newer `responses` API with stateful chaining. The reference implementation appeared to produce better answer quality in their evaluations.

This document captures what's different between the two approaches, the provider-by-provider compatibility findings, and the decision on what to change.

---

## Provider Compatibility: Responses API

Before deciding scope, we tested whether each OpenAI-compatible provider actually supports `client.responses.create()` for a custom function tool (not native search).

### GPT (OpenAI)
Fully supported. Responses API is OpenAI's own feature. Works as expected with custom function tools, `previous_response_id` chaining, and parallel tool execution.

### Qwen (DashScope)
**Supported with one caveat.** The `/responses` endpoint exists and responds correctly — Qwen already uses it for its native search path in `compare.py`. However, if the function tool is named `web_search`, Qwen intercepts the name and routes it to its own native search instead of calling the function. Renaming the tool (e.g. `you_search`) causes Qwen to return a proper `function_call` output as expected.

Tested empirically — output from both scenarios:
- Tool named `web_search` → `output[1].type = web_search_call` (Qwen native search, not our function)
- Tool renamed to `lookup_info` → `output[1].type = function_call`, `name=lookup_info`, correct arguments returned

**Conclusion:** Qwen can use the Responses API for the You.com path, but the tool must be renamed away from `web_search`. This is a clean fix with no other impact.

### Kimi (Moonshot AI)
**Not supported.** `/v1/responses` returns a hard 404:
```
NotFoundError: 404 - {'error': 'url.not_found', 'url': '/v1/responses'}
```
Confirmed via both empirical test and documentation review. Moonshot's OpenAI compatibility is explicitly scoped to Chat Completions — their docs list chat completions, files, models, and token estimation endpoints only. No Responses API endpoint is documented or planned. Kimi stays on `chat.completions`.

### Claude (Anthropic)
Unaffected — uses the Anthropic Messages API, entirely separate path.

### Llama (Together AI)
Not evaluated — Llama has no native search tool and is lower priority. Stays on `chat.completions`.

---

## The Two Approaches

### Current: `chat.completions` with manual message history

```
Round 1: [system, user question] → LLM → tool_calls
         execute_search(args) → results
         append tool result to messages list

Round 2: [system, user question, assistant tool_call, tool result] → LLM → tool_calls
         execute_search(args) → results
         append tool result to messages list

Round N: [system, user, asst1, tool1, asst2, tool2, ..., tool(N-1)] → LLM → final answer
```

Each round resends the full accumulated conversation. Input token count grows with every search round because all prior tool results are included as new input.

Tool calls within a single round execute **sequentially** — if the LLM requests two searches at once, they run one after the other.

### New: `responses` API with `previous_response_id` chaining

```
Round 1: input=question → LLM → function_calls
         execute searches (parallel) → tool outputs

Round 2: input=[tool_output_1, tool_output_2] + previous_response_id=round1.id → LLM → function_calls
         execute searches (parallel) → tool outputs

Round N: input=[new tool outputs only] + previous_response_id=round(N-1).id → LLM → final answer
```

OpenAI/DashScope retains full conversation state server-side via the response ID. Each subsequent call only sends the **new tool outputs**, not the growing history. The model has full context of all prior rounds through the chain.

Tool calls within a single round execute **in parallel** via `ThreadPoolExecutor`.

---

## Point-by-Point Comparison

| Dimension | `chat.completions` (current) | `responses` API (new) | Assessment |
|---|---|---|---|
| **Context passing** | Full history rebuilt and sent each round | Only new tool outputs sent; history retained server-side via `previous_response_id` | New approach avoids re-encoding growing history on each round |
| **Input token growth** | Grows with each round — prior tool results re-sent as input | Each call only pays for new tool outputs | Meaningful on queries with 4+ search rounds. GPT averages 5.6 searches/query (see Decision 005) — savings should be measurable |
| **Parallel tool calls** | Sequential within a round | Concurrent via `ThreadPoolExecutor` | Reduces latency when model issues 2+ searches in one round |
| **API design intent** | General-purpose completion API | Designed for stateful tool-use loops | Responses API was built for exactly this pattern |
| **Applies to** | OpenAI, Kimi, Qwen | GPT + Qwen (with tool rename); Kimi stays on chat.completions | See compatibility findings above |
| **Token counting** | Sum `prompt_tokens` + `completion_tokens` per round | `response.usage.input_tokens` + `output_tokens` per round | Both work; Responses API per-call counts reflect only new tokens |
| **Result formatting** | Custom formatter in `search_tool.py` (title/URL/snippets blocks) | `format_results_web_and_news` from You.com SDK | **Not applicable today** — we use direct HTTP calls, not the `youdotcom` SDK. If we adopt the SDK, this formatter handles edge cases in their response structure more robustly and is worth evaluating |
| **Livecrawl** | LLM decides via tool schema args (`livecrawl=true/false`) | Reference impl forces `livecrawl="web"` + `livecrawl_formats="markdown"` on all searches | Parked as a separate variable. **Future:** check with You.com whether there is a recommended default setting or an account/API-level toggle that controls this globally — would avoid relying on LLM discretion. See [Decision 005](005_benchmark_learnings.md) Section 2 for impact data |

---

## Token Impact and Why the Native Gap Remains

### Why GPT native is so lean (16,667 avg tokens)
GPT native search uses `tools=[{"type": "web_search"}]` — a **built-in tool type** that OpenAI executes entirely server-side within a single API call. The search loop (query → fetch → synthesize) happens inside OpenAI's infrastructure. You never see the search results as tokens — you get back one response with the final answer. No round-trips, no accumulated tool result content in the context window.

### Why You.com always has round-trips regardless of API used
You.com uses a **custom function tool** (`{"type": "function", "name": "you_search", ...}`). OpenAI doesn't know what `you_search` is — it can't execute it server-side. So it returns a `function_call` output, pauses, and waits for you to execute the search and send results back. That is a round-trip. With 5.6 avg searches per query (Decision 005), that's 5.6 round-trips regardless of whether you use `chat.completions` or the Responses API.

The Responses API does **not** eliminate round-trips for custom function tools. The distinction is:
- **Built-in tool** (`web_search`, `code_interpreter`) → executed server-side, single call, no round-trips
- **Custom function tool** (`you_search`) → must round-trip to your code for every search call

### What the Responses API actually improves
The reduction comes from `previous_response_id` chaining — prior tool results are retained server-side and not re-sent as input each round. At 5.6 searches per query the accumulation is real, so token counts should decrease. But the fundamental gap with native search remains: native never puts search content in the context window at all.

Decision 005 showed GPT You.com at 28,031 avg tokens vs native 16,667. The Responses API will reduce the You.com number, but will not close the gap. The improvement is directional, not a reversal.

---

## Why the New Approach May Produce Better Answers

The reference implementation was used in a structured eval framework and showed better outcomes. Note: the reference uses the `youdotcom` SDK and its `format_results_web_and_news` formatter — neither of which applies here (we make direct HTTP calls to the You.com API and format results ourselves). Those differences are irrelevant. Two mechanisms that do apply are plausible:

1. **Cleaner context.** With `previous_response_id` chaining, the provider manages the context window server-side. There's no manual re-packing of message history, which means no risk of truncation artifacts or ordering issues from manual message assembly.

2. **Parallel searches.** When a complex query causes the model to issue 2–3 searches in a single round, running them concurrently means the model gets all results back together before deciding its next step. Sequential execution introduces artificial ordering — results from search 1 arrive before search 2 is even started, which could subtly affect how the model synthesizes.

Neither effect is guaranteed to explain a quality difference — this is directional reasoning. The right test is to run the 50-question benchmark with both approaches and compare judge scores.

---

## Files Affected (if change is made)

| File | Change |
|---|---|
| `grounding/search_tool.py` | Add `TOOL_SCHEMA_RESPONSES` constant (Responses API tool format — no outer `"function"` wrapper; tool named `you_search` to avoid Qwen native search hijack) |
| `comparison/compare.py` | Rename current `_run_youdotcom_openai` → `_run_youdotcom_chat_completions` (Kimi only); add new `_run_youdotcom_openai` using Responses API for GPT + Qwen; update dispatch |
| `grounding/run.py` | Same split: Responses API path for GPT + Qwen; keep `chat.completions` for Kimi/Llama; update dispatch |

---

## Open Questions

- [ ] Does the Responses API approach produce measurably better judge scores on the 50-question benchmark? Run benchmark before/after and compare.
- [ ] Does `previous_response_id` chaining reduce total input tokens on multi-round queries? Run same query both ways and compare token counts to quantify the reduction.
- [ ] Are there error handling differences? (e.g., what happens if a `previous_response_id` expires or the chain breaks mid-run)
- [ ] Does renaming the tool from `web_search` to `you_search` affect Qwen's search behavior or quality? (Low risk — the description is unchanged, but worth a spot-check.)
