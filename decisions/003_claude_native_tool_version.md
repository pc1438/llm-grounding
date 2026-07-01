# Decision 003: Native Web Search Tool Version

**Status:** Complete — Use Config C (`web_search_20260209` with `allowed_callers: ["direct"]`) and rerun benchmark
**Date:** April 30, 2026
**Owner:** Chak Pothina

---

## Context

Anthropic offers two versions of the native web search tool for Claude:

- **`web_search_20250305`** — Older version. Straightforward search-only behavior. Raw web content enters the context window directly.
- **`web_search_20260209`** — Newer version. Supports **dynamic filtering**: Claude writes and executes code in a sandbox to post-process search results before they enter the context window, stripping irrelevant HTML and keeping only relevant content. Anthropic's own benchmarks show ~11% accuracy improvement and ~24% fewer input tokens with dynamic filtering enabled.

Our 50-query benchmark (47 completed, results in `benchmark_results_20260429_212950.json`) was run using **`web_search_20260209` with default settings** — code execution enabled. A senior engineer flagged that the code execution loop may be inflating native's token usage and latency on complex queries, making the comparison unfairly favorable to You.com.

After the benchmark, the code was manually changed to `web_search_20250305`. We need to determine the right version to use going forward.

## The Question

**Does the native web search tool version materially affect token usage, search behavior, and cost? If so, which version gives native the fairest comparison?**

## Background: What Dynamic Filtering Actually Does

When code execution is enabled (`web_search_20260209` defaults), Claude doesn't just search — it runs a multi-step loop:

1. **Search** — Claude issues a web search query
2. **Code execution** — Claude writes a small script to post-process the results (extract pricing tables, strip navigation, filter to relevant sections)
3. **Analysis** — The filtered output enters context; Claude reasons about what it found and what's still missing
4. **Repeat** — If gaps are identified, Claude searches again with a more targeted query

This is designed to improve accuracy by reducing noise. Anthropic reports ~11% accuracy gain on BrowseComp and DeepsearchQA benchmarks, with ~24% fewer input tokens on average.

**Key cost detail:** Code execution itself is free when used with web search — no per-call charge. But you still pay for all input tokens at the LLM rate, and the cascading search behavior on complex queries can dramatically inflate token counts.

**ZDR note:** Anthropic's docs note that Zero Data Retention eligibility may differ when code execution is involved. Worth flagging for clients with strict data retention requirements.

Source: [Anthropic Web Search Tool docs](https://docs.anthropic.com/en/docs/build-with-claude/tool-use/web-search-tool), [Dynamic Filtering analysis (gend.co)](https://www.gend.co/blog/claude-web-search-dynamic-filtering)

## Both Versions Expose the Same Controls

Both versions accept identical parameters:

| Parameter | `20250305` | `20260209` |
|-----------|:----------:|:----------:|
| `allowed_callers` | ✓ | ✓ |
| `allowed_domains` | ✓ | ✓ |
| `blocked_domains` | ✓ | ✓ |
| `max_uses` | ✓ | ✓ |
| `user_location` | ✓ | ✓ |

The `allowed_callers` parameter accepts: `"direct"`, `"code_execution_20250825"`, `"code_execution_20260120"`. Setting `allowed_callers: ["direct"]` on `20260209` disables code execution while keeping the newer search backend.

## Test Design

**What we tested:** Same queries, same model (Claude Sonnet 4.6), native path only, three tool configurations.

| Config | Tool Version | `allowed_callers` | Description |
|--------|-------------|-------------------|-------------|
| A | `web_search_20250305` | defaults | Older version, no code execution |
| B | `web_search_20260209` | defaults | Newer version, code execution enabled |
| C | `web_search_20260209` | `["direct"]` | Newer version, code execution disabled |

**Test queries (one per complexity tier):**

| Tier | Query |
|------|-------|
| Simple | What is the current price of NVIDIA stock? |
| Moderate | What are the latest developments in weight loss drugs? |
| Complex | Compare the pricing, context windows, and benchmark scores of Claude 4.5, GPT-5.4, and Gemini 2.5 |

Script: `use-cases/comparison/ab_claude_native_tool_version_test.py`
Raw output: `use-cases/comparison/ab_claude_native_tool_version_results_20260430_124651.json`

## Results

Run date: April 30, 2026. Model: Claude Sonnet 4.6. Baseline: Config B (what the 50-query benchmark used).

### Raw Data

| Tier | Config | Searches | Total Tokens | Input Tokens | Latency | Sources |
|------|--------|----------|-------------|-------------|---------|---------|
| Simple | **B: 20260209 (baseline)** | **1** | **16,637** | **15,994** | **17.6s** | **10** |
| Simple | A: 20250305 | 2 | 29,904 | 29,377 | 14.2s | 20 |
| Simple | C: 20260209 direct | 2 | 29,918 | 29,377 | 14.4s | 20 |
| Moderate | **B: 20260209 (baseline)** | **2** | **28,306** | **26,593** | **37.4s** | **20** |
| Moderate | A: 20250305 | 1 | 19,113 | 17,557 | 31.3s | 10 |
| Moderate | C: 20260209 direct | 1 | 20,046 | 18,386 | 35.6s | 10 |
| Complex | **B: 20260209 (baseline)** | **19** | **387,549** | **381,141** | **174.6s** | **190** |
| Complex | A: 20250305 | 6 | 108,093 | 105,371 | 44.2s | 60 |
| Complex | C: 20260209 direct | 3 | 45,862 | 43,371 | 48.2s | 30 |

### Delta Analysis (B as baseline)

| Tier | B → A (20250305) | B → C (20260209 direct) |
|------|-------------------|-------------------------|
| Simple | +13,267 tok (+80%), –3.4s (–19%) | +13,281 tok (+80%), –3.1s (–18%) |
| Moderate | –9,193 tok (–32%), –6.1s (–16%) | –8,260 tok (–29%), –1.8s (–5%) |
| Complex | **–279,456 tok (–72%), –130.4s (–75%)** | **–341,687 tok (–88%), –126.4s (–72%)** |

### Observations

1. **Code execution helps simple queries, hurts complex ones.** On Simple, B used 16.6k tokens (1 search) vs A/C's ~30k (2 searches). Dynamic filtering stripped noise effectively. On Complex, B exploded to 387k tokens (19 searches) vs C's 46k (3 searches) — an 8.5x difference.

2. **Why code execution hurts complex queries: the cascading search loop.** With code execution enabled, Claude doesn't just filter — it *analyzes gaps*. For a multi-entity comparison, the code execution step identifies "I have Claude pricing but not Gemini benchmarks" and triggers a targeted follow-up search. Then another analysis identifies another gap. This creates a cascade: 19 searches instead of 3. Each search adds more context tokens. Without code execution, Claude gets results, reasons over them in one pass, and writes the answer with whatever it has.

3. **Why code execution helps simple queries: noise reduction.** For "NVIDIA stock price," the raw web content includes navigation, ads, related articles — a lot of noise. Code execution filters this down to just the price data. Without it, the full page content enters context (hence A/C doing 2 searches with ~30k tokens for a simple lookup). This aligns with Anthropic's ~24% token reduction claim.

4. **A ≈ C on simple and moderate, but C wins on complex.** This confirms:
   - The code execution loop (not the search backend version) is what drives the behavior difference
   - The `20260209` search backend is genuinely better than `20250305` — on complex queries, C used only 46k tokens (3 searches) vs A's 108k (6 searches). Same model, same query, no code execution for either — the newer backend returned more relevant results per search.

5. **The moderate tier is consistent.** Both A and C used ~20k tokens with 1 search vs B's 28k with 2 searches. Code execution added a marginal follow-up search but didn't cascade.

---

## Decision

**Recommended config:** `web_search_20260209` with `allowed_callers: ["direct"]` (Config C)

**Rationale:**

- **Fairest to native.** Config C gives native its best possible configuration — the newest search backend (which returned better results on complex queries) without the code execution loop that inflates tokens on complex/research queries.
- **Most defensible.** We're using Anthropic's latest tool version. We're not handicapping native with an older backend. We're simply disabling one optional feature (code execution) that demonstrably hurts native's cost/token story on the query types that matter most for our benchmark.
- **Complex and deep research queries dominate the cost story.** In our benchmark, Tiers 3 and 4 (17 of 47 queries) account for the majority of total cost. These are the queries where code execution inflated native's numbers most dramatically.
- **Simple query trade-off is acceptable.** Yes, B was better on simple queries (16.6k vs 29.9k tokens). But simple queries are cheap regardless ($0.04-$0.38/query), and the moderate/complex savings with C are far more impactful.
- **The 50-query benchmark must be rerun.** Our existing results used Config B. The headline numbers (65.5% cost savings, 67.1% token savings) were measured against a native configuration that was inflated by code execution on complex queries. We need new numbers with Config C before the one-pager is client-ready.

**Why not Config A (`20250305`)?** Config C outperformed A on the complex tier: 46k tokens vs 108k, 3 searches vs 6. The newer `20260209` search backend appears to return more relevant results per search. Using C is both fairer to native and more defensible than using an older tool version.

**Impact on benchmark:** The existing 65.5% cost savings headline will likely narrow. Native will get cheaper on complex/deep research queries. The new number will be more defensible and closer to what clients would actually see in production.

---

## Code Changes

1. `compare.py` line 71: Set to `web_search_20260209` with `allowed_callers: ["direct"]`
2. `compare.py` line 398: Update docstring to match

## Follow-up Actions

- [x] Run A/B/C test
- [x] Analyze results and update this document
- [ ] Update `compare.py` to use Config C (`web_search_20260209`, `allowed_callers: ["direct"]`)
- [ ] Rerun 50-query benchmark with Config C
- [ ] Update one-pager with new headline numbers
- [ ] Update OPEN_QUESTIONS.md items #2 and #3 — code_execution blocks now explained
- [ ] Consider noting in one-pager: "Native configured with latest tool version, code execution disabled for cost optimization"
