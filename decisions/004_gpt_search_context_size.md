# Decision 004: GPT Native Search Context Size

**Status:** Complete — Use `search_context_size: "medium"` (default)
**Date:** April 30, 2026
**Owner:** Chak Pothina

---

## Context

OpenAI's Responses API `web_search` tool accepts a `search_context_size` parameter that controls how much of the context window is allocated to search results. Options are `low`, `medium` (default), and `high`.

This is analogous to Anthropic's code execution toggle (Decision 003) — a provider-specific setting that could materially affect token usage and cost, making the comparison unfair if not configured deliberately.

Our GPT benchmark was running with no explicit `search_context_size` (defaulting to `medium`). We needed to verify whether this default is the fairest configuration.

## The Question

**Does `search_context_size` materially affect GPT's native search token usage, source count, latency, or answer quality? If so, which setting gives native the fairest comparison?**

## Test Design

**What we tested:** Same 3 queries, same model (GPT-5.4), native path only, three context size settings.

| Config | search_context_size | Description |
|--------|-------------------|-------------|
| A | `low` | Minimal search context |
| B | `medium` | Default |
| C | `high` | Maximum search context |

**Test queries (one per complexity tier):**

| Tier | Query |
|------|-------|
| Simple | What is the current price of NVIDIA stock? |
| Moderate | What are the latest developments in weight loss drugs? |
| Complex | Compare the pricing, context windows, and benchmark scores of Claude 4.5, GPT-5.4, and Gemini 2.5 |

Script: `use-cases/comparison/ab_gpt_search_context_size_test.py`
Raw output: `use-cases/comparison/ab_gpt_search_context_size_results_20260430_190947.json`

## Results

Run date: April 30, 2026. Model: GPT-5.4. Baseline: Config B (medium — current default).

### Raw Data

| Tier | Config | Searches | Total Tokens | Input Tokens | Sources | Latency |
|------|--------|----------|-------------|-------------|---------|---------|
| Simple | A: low | 1 | 5,058 | 4,918 | 0 | 4.6s |
| Simple | **B: medium (baseline)** | **1** | **5,050** | **4,918** | **0** | **3.2s** |
| Simple | C: high | 1 | 5,073 | 4,918 | 0 | 4.6s |
| Moderate | A: low | 2 | 14,095 | 12,965 | 6 | 31.4s |
| Moderate | **B: medium (baseline)** | **1** | **9,151** | **8,072** | **6** | **19.6s** |
| Moderate | C: high | 1 | 9,745 | 8,766 | 7 | 17.5s |
| Complex | A: low | 5 | 31,098 | 28,800 | 5 | 36.4s |
| Complex | **B: medium (baseline)** | **7** | **38,717** | **36,814** | **5** | **35.4s** |
| Complex | C: high | 6 | 32,946 | 30,509 | 9 | 43.0s |

### Aggregate Totals

| Config | Total Tokens | Searches | Sources | Avg Latency |
|--------|-------------|----------|---------|-------------|
| A: low | 50,251 | 8 | 11 | 24.1s |
| B: medium | 52,918 | 9 | 11 | 19.4s |
| C: high | 47,764 | 8 | 16 | 21.7s |

### Observations

1. **Simple queries: no meaningful difference.** All three configs produce ~5,050 tokens with 1 search and 0 sources. Input tokens are identical (4,918). The setting has no effect on simple lookups. GPT doesn't cite sources for simple factual answers regardless of context size.

2. **Low hurts moderate queries.** Config A triggered an extra search on the moderate query (2 vs 1) and used 54% more tokens (14k vs 9k). With less context per search, GPT wasn't satisfied and searched again — similar to Claude's cascading behavior with code execution.

3. **Complex queries: counterintuitive results.** High used *fewer* tokens than medium (33k vs 39k, -15%) with *more sources* (9 vs 5). Medium triggered the most searches (7). This suggests that with more context per search, GPT found what it needed in fewer round-trips.

4. **The differences are small compared to Claude's code execution toggle.** Claude's code execution caused an 8x token difference on complex queries (387k vs 46k). GPT's context size range is at most 1.5x (low's moderate query). The setting matters less for GPT.

5. **Zero-sources problem is structural, not configurable.** All three configs returned 0 sources on the simple query. This is GPT's behavior — it often doesn't attach `url_citation` annotations for straightforward factual answers. No `search_context_size` setting fixes this.

---

## Decision

**Recommended config:** `search_context_size: "medium"` (the default)

**Rationale:**

- **It's the default.** Customers using OpenAI's web search out of the box get medium. Using the default is the most defensible "we didn't tune anything" position.
- **Medium is balanced.** Low hurts moderate queries (extra search, +54% tokens). High gives marginal improvements on complex (more sources, fewer tokens) but with higher latency. Medium avoids the worst cases of both extremes.
- **The differences are small.** Unlike Claude's code execution toggle where the wrong config inflated tokens 8x, GPT's context size differences are modest. Total tokens across all 3 queries: low=50k, medium=53k, high=48k — within ~10% of each other.
- **Explicitly setting it is better than relying on default.** Even though medium is the default, setting it explicitly in our config makes the choice visible and auditable.

**Why not high?** High returned more sources on complex queries (9 vs 5) and used slightly fewer total tokens. However, it added latency on complex queries (+7.6s) and the source improvement is from a single query — not enough signal to justify deviating from the default. If a future 50-query benchmark shows high consistently improves citation quality, we should revisit.

**Why not low?** Low caused a cascading search on moderate queries (+54% tokens, +12s latency). It's the worst option for the query tier that represents the bulk of real-world usage.

## Code Changes

1. `compare.py`: Set `search_context_size: "medium"` explicitly in OpenAI's `native_search_tool` config
2. `compare.py`: Updated `describe_native_search()` to include `search_context_size` in output

## Follow-up Actions

- [x] Run A/B/C test
- [x] Analyze results and create this document
- [x] Update `compare.py` with explicit medium setting
- [x] Update `describe_native_search()` for OpenAI config display
- [ ] Run full 50-query GPT benchmark
- [ ] Note in one-pager: GPT native search uses default settings (search_context_size=medium)
