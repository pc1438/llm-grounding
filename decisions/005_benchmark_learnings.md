# Benchmark Learnings

**Last updated:** May 1, 2026
**Owner:** Chak Pothina

---

## 1. Search Architecture Differences: Why Token Profiles Differ by LLM

This is the single most important finding from our benchmarks. Claude and GPT handle native web search in fundamentally different ways, and this explains why You.com's token advantage is large on Claude but inverts on GPT.

### Claude Native Search
Claude's built-in search (`web_search_20260209`) operates as a tool call. Search results — raw web page content — are injected directly into the conversation context window. Each search round-trip adds the previous results as input tokens to the next LLM call. With 6+ searches on complex queries, raw pages compound: by the 6th search, the LLM is sending all 5 prior page payloads as input.

Result: **76,953 avg tokens per query** (Config C, code exec off). Native returns full HTML/text, so each search adds 10–50k tokens to the context window.

### GPT Native Search
GPT's built-in search (`web_search`, `search_context_size=medium`) operates **server-side**. The search happens inside the API call — results never enter the conversation context as tool results. You pay for the answer tokens, but search content doesn't compound across round-trips.

Result: **16,667 avg tokens per query**. GPT averages 2.9 searches per query, but the context window stays lean regardless.

### You.com (Tool-Call Path)
You.com operates the same way on both LLMs: the LLM makes tool calls, You.com returns structured snippets, and those snippets enter the context window. Tokens accumulate across round-trips, but each search returns compact snippets (~1–2k tokens) instead of raw pages.

Result: **28,031 avg tokens on GPT**, **47,686 on Claude**. The difference between the two is driven by the number of searches (5.6 avg on GPT, ~5 on Claude) and context accumulation patterns.

### Why This Matters
- On Claude, You.com uses **38% fewer tokens** than native — because native's raw page injection is so bloated that even with round-trip accumulation, You.com is leaner.
- On GPT, You.com uses **68% more tokens** than native — because GPT native never puts search content in the context window at all. The round-trip accumulation that was invisible on Claude (masked by native's worse bloat) is now the dominant factor.
- **The You.com quality and cost advantage holds on both** — but the token story is model-dependent and should not be presented as universal.

---

## 2. Livecrawl Impact on GPT Benchmark

The LLM can optionally request `livecrawl=true` on You.com searches, which returns full page markdown instead of snippets. A lead engineering hypothesis was that livecrawl drives the token increase and latency on GPT.

### Findings (GPT-5.4, 50 queries)
- **12/50 queries** triggered livecrawl (heuristic: >5,000 context tokens per search call)
- **6 of 8 Deep Research queries** triggered livecrawl — heaviest concentration
- Livecrawl queries: **60,707 avg tokens, 84s avg latency**
- Snippet-only queries: **17,712 avg tokens, 52s avg latency**
- Ratio: **3.4x more tokens, 1.6x more latency** with livecrawl

### Assessment
Livecrawl is a real contributor but not the full story. Even on the 38 snippet-only queries, You.com averages 17,712 tokens vs native's 16,667 — still 6% more. The structural cause is the round-trip accumulation (5.6 searches per query, each adding to context). Livecrawl amplifies this by making each round-trip heavier.

### Action Items
- [ ] Consider disabling livecrawl for GPT benchmarks to isolate the round-trip effect
- [ ] Add per-search-call livecrawl logging to the benchmark (currently inferred from token heuristic)
- [ ] Investigate whether GPT triggers livecrawl more aggressively than Claude (tool schema difference?)

---

## 3. Code Execution Toggle (Claude-Specific)

Anthropic's `web_search_20260209` includes code execution that runs Python/JS during search. Enabled by default.

### Config B (Code Exec ON) vs Config C (Code Exec OFF)
| Metric | Config B | Config C | Impact |
|--------|----------|----------|--------|
| Avg native tokens | 156,147 | 76,953 | 2x reduction |
| Avg native cost | $0.58 | $0.29 | 2x reduction |
| Avg native latency | 101.8s | 41.7s | 2.4x faster |
| YDC token savings | 67% | 38% | Headline narrows |
| YDC cost savings | 66% | 33% | Headline narrows |
| YDC quality wins | 26/47 | 29/49 | Slightly better |

### Key Learning
Code execution inflates native token usage dramatically (up to 8x on complex queries). Disabling it is the fairest apples-to-apples comparison. But Config B represents what a customer gets out of the box, so both should be presented.

**Decision:** Use Config C for primary comparisons, show Config B as "default configuration" context. See [Decision 003](003_claude_native_tool_version.md).

---

## 4. GPT Search Context Size (GPT-Specific)

OpenAI's `search_context_size` parameter (low/medium/high) controls how much context the search uses.

### Finding
Differences are small (~10% token variance across settings). Medium is the balanced default. Low causes cascading searches on moderate queries (+54% tokens). High gives marginally better citations on complex queries but adds latency.

**Decision:** Use medium (default). See [Decision 004](004_gpt_search_context_size.md).

---

## 5. GPT Zero-Source Problem

GPT frequently returns zero `url_citation` annotations, especially on simple factual queries.

### Scale (50-query benchmark)
- **8/50 native queries** returned zero sources
- **1/50 You.com queries** returned zero sources
- Concentrated in Simple tier: native avg 0.8 sources vs You.com avg 8.4

### Impact
This is structural GPT behavior — no configuration setting fixes it. It means You.com has a significant citation quality advantage on GPT that's independent of answer quality. The blind judge scores confirm this: citation_quality is consistently higher for You.com on GPT.

---

## 6. Latency Architecture

### You.com Latency Disadvantage
You.com requires N tool-call round-trips (LLM → search API → LLM). Each round-trip incurs:
- LLM inference time (increasing with accumulated context)
- Network round-trip to You.com API
- You.com search time

Native search (both Claude and GPT) does searching server-side within the API call, avoiding the per-search network round-trip.

### Where It Hurts Most
- **GPT:** You.com 161% slower overall (60s vs 23s). GPT native is very fast server-side.
- **Claude Config C:** You.com 37.5% slower (57.3s vs 41.7s). Gap is smaller because Claude native is slower.
- **Claude Config B:** You.com 43% **faster** (58.5s vs 101.8s). Code execution loops make native slower.

### Deep Research Tier
This is where latency diverges most. You.com does 9–18 search round-trips on Deep Research queries. On Claude Config C, this tier alone causes a 75% latency gap (154s vs 88s). The other tiers are much closer.

---

## 7. Cross-Model Summary

| Metric | Claude (Config C) | GPT-5.4 |
|--------|-------------------|---------|
| YDC token savings | 38% | -68% (YDC uses more) |
| YDC cost savings | 33% | 11% |
| YDC quality wins | 29/49 (59%) | 27/50 (54%) |
| YDC avg score | 16.3/20 | 16.6/20 |
| Native avg score | 16.1/20 | 15.8/20 |
| Latency | YDC 37% slower | YDC 161% slower |
| Native sources | Comparable | 8 zero-source queries |
| YDC sources | Comparable | 31.2 avg (vs 5.0 native) |

### Universal Findings
1. **You.com wins on quality** across both LLMs (54–59% win rate)
2. **You.com wins on cost** across both LLMs (11–33% savings)
3. **You.com provides better citations** — dramatically so on GPT
4. **Token story is model-dependent** — do not present as universal
5. **Latency favors native** in both cases (except Claude Config B)

---

## 8. Prompt Does Not Cause Over-Searching

We ran A/B/C tests with three system prompt variants (standard, minimal, no prompt) on Claude. Native over-searching persisted across all variants. The behavior is inherent to built-in search, not caused by our prompt.

See [Decision 001](001_claude_system_prompt_test.md).

---

## 9. Non-Determinism

Both LLMs show significant run-to-run variance on search behavior. Example: GPT-5.4 on the NBA scores query produced 9 searches/51k tokens in one run and 1 search/3,405 tokens in the next. This is not a bug — it's inherent to LLM-controlled search.

**Implication:** Single-run benchmarks capture directional trends, not precise numbers. Per-tier averages across 50 queries are more reliable than individual query results.
