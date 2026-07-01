# Open Questions & Investigation Topics

Internal working document — things to investigate, experiment with, and discuss before finalizing the benchmark and one-pager.

Last updated: April 29, 2026

---

## 1. System Prompt Impact on Search Behavior

**Observation:** The current system prompt says "You can search multiple times if you need to compare information or dig deeper." This may encourage over-searching on the native path more than the You.com path.

**Questions:**
- Does removing or softening this line reduce the number of native searches?
- Does it also reduce You.com searches (and if so, does quality drop)?
- Should we test 2-3 prompt variants and measure search call counts across both paths?
- Is the prompt the reason native does 6 searches for a simple NBA scores question while You.com does 2, or would native still over-search with a minimal prompt?

**Experiment:** Run the same 5 queries with 3 different system prompts (current, minimal, no multi-search encouragement). Compare search_calls, tokens, latency, and quality for both paths.

---

## 2. Why Does Native Over-Search on Simple Queries?

**Observation:** "What were the final scores in yesterday's NBA games?" — You.com: 2 searches, 16.5s. Native: 6 searches with code_execution steps, 54s. This is a Tier 1 simple factual lookup.

**Questions:**
- Is it the system prompt (see #1)?
- Is it the quality/format of native search results — does Claude get back noisy/incomplete content and decide to keep searching?
- Is it the `code_execution` steps — is native running internal code to process results, and that processing triggers more searches?
- Can we log or capture what native actually returns as search content to compare against You.com results?
- Is this pattern consistent, or is it query-dependent? Do simple Tier 1 queries consistently over-search on native?

**What we DON'T know:** We speculated that You.com returns "more compact, structured results" but we have no direct evidence. We can see the You.com search payloads in the tool call trace, but native's search content is server-managed. We should be careful not to claim things we can't prove.

---

## 3. What Are the `code_execution` Blocks in Native Search?

**Observation:** Native search logs show repeated `code_execution` events between `web_search` events. These don't appear in the You.com path.

**Questions:**
- What is Claude's native search doing with code_execution? Is it parsing HTML? Extracting data? Deciding what to search next?
- Does this represent additional compute cost that isn't reflected in our token counts?
- Is this a feature (Claude doing deeper analysis) or overhead (Claude compensating for raw search results)?
- Should we capture code_execution counts as a metric in the benchmark?

**Action:** Add code_execution event counting to the native streaming handler in compare.py so we can quantify this.

---

## 4. Token Bloat — What's Actually in the Native Context?

**Observation:** Native uses ~33,500 avg input tokens vs ~7,300 for You.com (from 10-query benchmark). But we don't know exactly what's in those tokens.

**Questions:**
- Is native injecting full raw web pages? Snippets? Something else?
- Some native search results have encrypted/opaque content blocks. How much of the input token count is from content we can't inspect?
- Can we estimate what fraction of native input tokens is "useful" vs overhead?
- For the one-pager, can we make the "token bloat" claim more precise?

---

## 5. Latency Breakdown — Where Does Time Go?

**Observation:** Native took 54s for a simple query. ~44.5s was search/processing, ~9.5s was answer generation.

**Questions:**
- What's the per-search latency for native vs You.com? (We have this data in the logs but haven't aggregated it.)
- Is the answer generation time (post-search) actually similar between paths, with all the difference coming from the search phase?
- Is there a fixed overhead per native search round (the code_execution step)?
- For the one-pager, should we add a latency comparison? Currently we don't highlight latency.

---

## 6. Fairness of the Comparison

**Things we've controlled for:**
- Same LLM (Claude Sonnet 4.6)
- Same system prompt (identical, tool-agnostic)
- Blind judge (GPT-5.4, randomized A/B positions)
- Same questions

**Things that might not be fair:**
- You.com's tool schema (`web_search` with parameters like `freshness`, `country`) gives the LLM more control over search. Native search has no such parameters. Is this an unfair advantage?
- You.com results go through our `execute_search()` function which may format/structure results differently than raw API output. Is there processing that helps?
- The You.com path does multiple LLM round-trips (tool-use loop), while native does it all in one streaming call. The multi-round approach might give the LLM more "thinking time" between searches. Is that a confound?
- Should we disclose these differences in the one-pager, or are they inherent to the product difference?

---

## 7. Scaling to 50 Queries — What to Watch For

**Questions:**
- Will the patterns from 10 queries hold at 50? (Token savings, cost savings, quality wins.)
- Tier 3 and Tier 4 questions are complex multi-source queries. Will native's over-searching actually help on these? (More searches = more sources = better answers for complex queries?)
- Will we hit credit limits or rate limits during the run?
- What's the expected total cost for 50 queries? (Rough estimate: 50 * $0.15 avg = ~$7.50 for You.com, 50 * $0.15 avg = ~$7.50 for native, plus judge costs.)

---

## 8. One-Pager Updates After 50-Query Benchmark

**Decisions needed:**
- Do we keep the per-query table, or is 50 rows too many? Options: show top 10, show per-tier averages, show full table in appendix.
- Should we add a "number of web searches" column to the table?
- Should we add latency as a highlighted metric?
- The headline numbers (57.6% lower cost, 62.5% fewer tokens, 7/10 quality wins) will change. What if the numbers are less impressive at 50 queries?
- Should we keep it at 2 pages or expand to 3 with the larger dataset?
- Should we add per-tier breakdowns to show how the advantage scales with query complexity?

---

## 9. Reproducibility & Methodology

**Questions:**
- Answers will vary run to run (LLM non-determinism, live web content changes). How much variance should we expect?
- Should we run the benchmark multiple times and average? (Expensive but more rigorous.)
- The one-pager says "one benchmark dataset and one LLM pairing" — should we also test with GPT-5.4 as the primary LLM?
- Should we publish the benchmark code alongside the one-pager so prospects can run it themselves?

---

## Next Steps

- [ ] Run 5-query limit test, inspect output JSON for completeness
- [ ] Run full 50-query benchmark
- [ ] Analyze results, revisit open questions with data
- [ ] Design experiment for system prompt impact (#1)
- [ ] Add code_execution counting to native handler (#3)
- [ ] Update one-pager with new benchmark data
