# llm-grounding

A practical toolkit for grounding LLM responses with real-time web search — and for making informed decisions about which LLM and which grounding approach is right for your use case.

Most teams building search-grounded AI systems face the same set of questions:

- Which LLM handles tool-use and grounding best for my workload?
- Should I use my LLM provider's native web search, or a dedicated search API?
- What does grounding actually cost — inference tokens, search fees, combined?
- How does grounding quality vary across providers?

This repo gives you working code to run those experiments yourself, across five LLMs, with a blind cross-model judge and full cost accounting.

## What's Here

```
llm-grounding/
├── grounding/      ← Ground any LLM with You.com Search (library + CLI)
├── comparison/     ← You.com Search vs. native web search: cost + quality
├── app/            ← Web UI: interactive trace viewer + comparison dashboard
├── decisions/      ← Decision logs: what we tested, what we found, why
└── README.md       ← You are here
```

**grounding/** is the core library. It adds real-time web search to Claude, GPT-5.4, Qwen, Kimi, and Llama via a single `agent.ask()` interface. Importable or runnable as a CLI — no UI required.

**comparison/** runs the same query through two paths — LLM + You.com Search vs. LLM + native web search — measures token usage and cost on both sides, then uses a cross-model blind judge to score answer quality. The data it produces is the basis for grounding architecture decisions.

**app/** is a thin browser UI wrapping both modules. Three tabs: About (architecture + decision context), Grounding (live tool-use trace viewer), Comparison (side-by-side cost and quality dashboard).

**decisions/** is the research log: each file documents a question we investigated, the experiment we ran, and the decision we made. Not academic — these are the actual findings that shaped the architecture here.

## Supported Models

| Key | Model | Vendor | Grounding | Comparison |
|-----|-------|--------|-----------|------------|
| `claude` | Claude Sonnet 4.6 | Anthropic | Yes | Yes |
| `gpt` | GPT-5.4 | OpenAI | Yes | Yes |
| `qwen` | Qwen Plus | Alibaba (DashScope) | Yes | Yes |
| `kimi` | Kimi K2.5 | Moonshot AI | Yes | Yes |
| `llama` | Llama 4 Maverick | Meta (via Together AI) | Yes | — |

Native web search comparison requires provider support. Claude and GPT have production web search tools. Qwen and Kimi have experimental native search that we test here. Llama has no native search.

## Quick Start

### 1. Install dependencies

```bash
cd grounding
pip install -r requirements.txt
```

### 2. Set API keys

```bash
cp grounding/.env.example grounding/env.txt
# Edit grounding/env.txt with your keys
```

```
YDC_API_KEY=your-key           # Required — you.com/platform ($100 free credits)
ANTHROPIC_API_KEY=your-key     # For Claude
OPENAI_API_KEY=your-key        # For GPT-5.4
DASHSCOPE_API_KEY=your-key     # For Qwen
MOONSHOT_API_KEY=your-key      # For Kimi
TOGETHER_API_KEY=your-key      # For Llama
```

### 3. Ground an LLM with web search

```bash
cd grounding
python run.py claude "What happened in tech news today?"
python run.py gpt    "Who won the 2026 Olympic hockey gold?"
python run.py kimi   "Latest NVIDIA earnings"
```

### 4. Compare You.com vs. native search

```bash
cd comparison
python compare.py claude "What happened in tech news today?"
python compare.py openai "Current S&P 500 price"
```

### 5. Launch the web UI

```bash
cd app
python server.py
# Open http://localhost:8080
```

## Making Grounding Architecture Decisions

The comparison tooling exists to answer a specific question: **is it better to use a dedicated search API (You.com) or your LLM provider's built-in web search?**

The tradeoffs are real:

| | Dedicated Search API (You.com) | Native Web Search |
|---|---|---|
| **Cost** | Predictable per-query fee + inference | Per-search fee + search content billed as tokens |
| **Control** | Full: query, freshness, domains, count | Provider-controlled: black box |
| **Token cost** | Higher — search results are sent as tool context | Lower — provider handles retrieval internally |
| **Auditability** | search_uuid, full source URLs, latency | Limited — provider exposes partial metadata |
| **Portability** | Same API across any LLM | Tied to each provider's implementation |

The `comparison/` module measures this empirically on your queries, with your models, at current pricing. The `decisions/` folder logs what we found.

### Choosing Between LLMs for Grounding

Beyond the search architecture decision, the LLMs themselves differ in how well they handle grounded reasoning:

- **Tool-use compliance** — does the model call search when it should, and not call it when it shouldn't?
- **Citation quality** — does the model cite sources accurately, or hallucinate citations?
- **Multi-hop reasoning** — can the model plan a sequence of searches to answer complex questions?
- **Context utilization** — does the model actually use the search results, or fall back to training data?

This repo gives you a working benchmark harness (`comparison/benchmark_runner.py`, 50 ground-truth questions) to measure these across providers on your own queries.

## Architecture

The codebase is layered: CLI modules are standalone and importable; the UI is a thin HTTP wrapper.

```
┌─────────────────────────────────────────────┐
│              app/server.py                  │  ← HTTP + SSE
└──────────┬──────────────────┬───────────────┘
           │                  │
    ┌──────▼──────┐    ┌──────▼──────┐
    │ grounding/  │    │ comparison/ │          ← CLI + library
    │   run.py    │    │  compare.py │
    └──────┬──────┘    └──────┬──────┘
           │                  │
    ┌──────▼──────────────────▼───────┐
    │        grounding/search_tool.py │         ← You.com Search API
    │        grounding/base_agent.py  │         ← tool-use loop (all providers)
    │        grounding/agents/*.py    │         ← per-provider clients
    └─────────────────────────────────┘
```

**Grounding flow:** User question → LLM decides to search → You.com returns results → LLM synthesizes a cited answer.

**Comparison flow:** Same query runs in parallel through both paths → cross-model blind judge scores completeness, relevance, specificity, citation quality → cost breakdown.

## Search Cost Reference

| Provider | Per 1,000 searches | Notes |
|----------|--------------------|-------|
| You.com Search API | $5 | Predictable; full control |
| Anthropic Web Search | $10 | + search results billed as input tokens |
| OpenAI Web Search | $10 | + search content at model input rates |
| Qwen (DashScope) | Bundled | No separate per-search fee; billed in token cost |
| Kimi (Moonshot) | Bundled | Built-in search; no separate fee |

## License

MIT
