# Decision 002: Search Result Count (5 vs 10)

**Status:** Complete — Keep count=5
**Date:** April 30, 2026
**Owner:** Chak Pothina

---

## Context

The You.com Search tool schema includes a `count` parameter that controls how many results are returned per search call. The current default is 5. A reasonable question arose: would increasing to 10 give the LLM richer context per search, potentially reducing the number of follow-up searches and improving answer quality?

This matters for the benchmark and product positioning because result count directly affects token consumption, latency, and cost — the core metrics in our comparison against native search.

## The Question

**Does increasing the default result count from 5 to 10 improve efficiency (fewer searches, fewer tokens) or quality, or does it just inflate costs?**

## Test Design

**What we're testing:** Same queries, same model, You.com path only, with the tool schema default count changed from 5 to 10.

**Model:** Claude Sonnet 4.6 (via Anthropic API)

**Approach:** We modified the tool schema's `count` default in-place before each batch so the LLM sees the updated default in its tool definition. The LLM can still override the count per-call, but the default influences its behavior when it doesn't specify one.

**Test queries (one per complexity tier):**

| Tier | Query |
|------|-------|
| Simple | What is the current price of NVIDIA stock? |
| Moderate | What are the latest developments in weight loss drugs? |
| Complex | Compare the pricing, context windows, and benchmark scores of Claude 4.5, GPT-5.4, and Gemini 2.5 |

**Test matrix:** 3 queries × 3 count values (5, 8, 10) = 9 total API calls (You.com path only). Run in two batches: 5 and 10 first, then 8 separately using `--only 8`.

**Metrics captured:** search_calls, api_calls, total_tokens, input_tokens, output_tokens, latency_ms.

## Implementation

Script: `use-cases/comparison/ab_count_test.py`
Raw output: `use-cases/comparison/ab_count_results_20260430_091245.json` (5 and 10), `use-cases/comparison/ab_count_results_20260430_091934.json` (8)

## Results

Run date: April 30, 2026. Model: Claude Sonnet 4.6.

### Raw Data

| Tier | Count | Searches | LLM Calls | Total Tokens | Input Tokens | Latency |
|------|-------|----------|-----------|-------------|-------------|---------|
| Simple | 5 | 1 | 2 | 3,478 | 3,156 | 8.4s |
| Simple | 8 | 1 | 2 | 3,491 | 3,156 | 8.9s |
| Simple | 10 | 1 | 2 | 3,531 | 3,156 | 8.2s |
| Moderate | 5 | 1 | 2 | 6,971 | 5,850 | 26.7s |
| Moderate | 8 | **3** | 3 | **16,383** | 14,741 | **38.1s** |
| Moderate | 10 | **3** | 3 | **18,436** | 16,742 | **38.7s** |
| Complex | 5 | 8 | 4 | 77,451 | 74,089 | 84.3s |
| Complex | 8 | 10 | 5 | 60,674 | 57,965 | 61.0s |
| Complex | 10 | 10 | 5 | 54,309 | 51,269 | 57.2s |

### Delta Analysis (count=5 as baseline)

| Tier | count=5 → count=8 | count=5 → count=10 |
|------|-------------------|---------------------|
| Simple | +13 tok (+0.4%), -0s | +53 tok (+1.5%), -0.3s |
| Moderate | **+9,412 tok (+135%), +11.4s** | **+11,465 tok (+164%), +12.0s** |
| Complex | -16,777 tok (-22%), -23.3s | -23,142 tok (-30%), -27.1s |

### Observations

1. **Simple queries: no impact at any count.** All three configurations did 1 search with nearly identical tokens. The LLM gets what it needs from the first few results regardless.

2. **count=8 and count=10 behave almost identically.** Both trigger 3 searches on moderate (vs 1 for count=5), and both trigger 10 searches on complex (vs 8 for count=5). The behavioral threshold is somewhere between 5 and 8 — there is no "sweet spot" at 8 that avoids the moderate regression while capturing the complex benefit.

3. **Moderate queries: any count above 5 backfires.** Both count=8 (+135% tokens) and count=10 (+164% tokens) triggered the LLM to pursue additional angles. More diverse results per search led to more curiosity, not less. Since moderate queries are the most common tier (18 of 50 benchmark questions), this regression dominates.

4. **Complex queries: higher counts help, scaling with count.** count=8 saved 22% of tokens, count=10 saved 30%. For multi-entity comparisons, richer results per search enable faster synthesis. But this is the minority tier (11 of 50 questions).

5. **The trade-off has no middle ground.** We expected count=8 might capture the complex benefit without the moderate penalty. It doesn't — it triggers the same behavioral shift as count=10.

---

## Decision

**Recommended default:** count=5 — no change needed.

**Rationale:**

- **There is no middle ground.** We tested 5, 8, and 10. Both 8 and 10 trigger the same behavioral shift — the LLM starts doing more follow-up searches on moderate queries. The threshold is binary (≤5 vs >5), not gradual.
- **Moderate queries are the largest tier** in real-world usage and our benchmark (18 of 50 questions). A 135-164% token increase on this tier would significantly worsen the cost story.
- **Simple queries see no benefit** at any count.
- **Complex queries benefit from higher counts**, but they're the minority (11 of 50), and developers can override per-call when they need richer results.
- **The risk is asymmetric:** going above 5 makes moderate 2.3-2.6x more expensive while making complex only 22-30% cheaper. The downside outweighs the upside for the common case.
- **The tool schema exposes the count parameter.** Sophisticated users building research agents can set count=10 or count=20 for multi-entity comparisons. The default should optimize for the majority of queries.

**Impact on benchmark:** None — we proceed with count=5 as the default.

---

## Follow-up Actions

- [x] Run count=5 vs count=10 test
- [x] Run count=8 test to check for a middle ground
- [x] Analyze results and document decision
- [ ] Consider adding count guidance to documentation (e.g., "use count=10+ for complex multi-entity comparisons")
- [ ] If we run the benchmark again, could test count=10 on Tier 3-4 queries only to validate the complex-query benefit at larger n
