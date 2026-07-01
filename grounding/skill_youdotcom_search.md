---
name: youdotcom-web-search
description: >
  Ground LLM responses with real-time web search results from You.com Search API.
  Use this skill whenever the user asks a question that requires current information,
  fact-checking, or when the answer might be outdated in training data.
keywords:
  - web search
  - grounding
  - hallucination reduction
  - real-time data
  - fact checking
  - RAG
compatibility:
  - claude-code
  - claude-desktop
  - cursor
  - windsurf
---

# You.com Web Search — Grounding Skill

## When to Use This Skill

Activate this skill when:
- The user asks about current events, recent data, or time-sensitive information
- You need to verify a claim or fact-check before responding
- The question involves specific numbers (prices, stats, dates) that change over time
- The user explicitly asks you to search the web

## Environment Setup

```bash
export YDC_API_KEY="your-api-key"  # Get from https://you.com/platform
```

## Workflow

### Step 1: Search

Call the You.com Search API to retrieve relevant web results.

```bash
curl -G "https://ydc-index.io/v1/search" \
  -H "X-API-Key: $YDC_API_KEY" \
  --data-urlencode "query=USER_QUERY_HERE" \
  -d count=5
```

### Step 2: Choose retrieval depth

**Snippets mode** (default) — use when:
- The question is straightforward and snippet-level context is sufficient
- You want to minimize token usage and inference cost
- The target LLM has a smaller context window (8k-32k)

**Livecrawl mode** — use when:
- The question requires detailed analysis of source content
- Snippet-level context is too shallow for a good answer
- The target LLM has a large context window (100k+)

For livecrawl, add these parameters:
```
livecrawl=web&livecrawl_formats=markdown
```

### Step 3: Format context

Structure the search results as a numbered source list:

```
[Source 1] {title}
URL: {url}
{snippet or full content}

---

[Source 2] {title}
URL: {url}
{snippet or full content}
```

### Step 4: Generate grounded response

Inject the formatted context into your system prompt:

```
You are a helpful assistant. Answer using ONLY the web search results below.
Cite sources by number [1], [2], etc.
If the sources don't contain enough information, say so — do not fabricate.

{formatted_context}
```

### Step 5: Validate

- Ensure every factual claim maps to a numbered source
- Flag any claims not backed by the retrieved context
- Include source URLs for transparency

## Advanced: Domain Filtering

Restrict results to trusted sources:
```
include_domains=reuters.com,apnews.com,nature.com
```

Or exclude known low-quality sources:
```
exclude_domains=example-spam.com
```

## Advanced: Freshness Filtering

For breaking news: `freshness=day`
For recent developments: `freshness=week`
For date ranges: `freshness=2025-01-01to2025-06-30`

## Troubleshooting

| Issue | Fix |
|-------|-----|
| No results | Broaden the query, remove domain filters |
| Stale results | Add `freshness=day` or `freshness=week` |
| Shallow context | Switch to livecrawl mode |
| Too many tokens | Reduce `count`, use snippets mode, or truncate content |
