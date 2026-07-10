# llm-grounding

A practical toolkit for grounding LLM responses with real-time web search — and for making informed decisions about which LLM and which grounding approach is right for your use case.

Most teams building search-grounded AI systems face the same set of questions:

- Which LLM handles tool-use and grounding best for my workload?
- Should I use my LLM provider's native web search, or a dedicated search API?
- What does grounding actually cost — inference tokens, search fees, combined?
- How does grounding quality vary across providers?

This repo gives you working code to run those experiments yourself, across thirteen LLMs, with a blind cross-model judge and full cost accounting.

## What's Here

```
llm-grounding/
├── grounding/      ← Ground any LLM with You.com Search (library + CLI)
├── comparison/     ← You.com Search vs. native web search: cost + quality
├── app/            ← Web UI: Playground, Compare, Multi-model, About tabs
├── decisions/      ← Decision logs: what we tested, what we found, why
└── README.md       ← You are here
```

**grounding/** is the core library. It adds real-time web search to any supported LLM via a single `agent.stream()` interface. Importable or runnable as a CLI — no UI required. `agent_pool.py` provides a shared, thread-safe agent instance cache used by both the UI server and the comparison module.

**comparison/** runs the same query through two paths — LLM + You.com Search vs. LLM + native web search — measures token usage and cost on both sides, then uses a cross-model blind judge to score answer quality. The data it produces is the basis for grounding architecture decisions.

**app/** is a thin HTTP + SSE server wrapping both modules. Four tabs:
- **About** — architecture overview, configuration reference, tunable parameters
- **Playground** — live tool-use trace viewer with full token, cost, and timing breakdown
- **Compare** — side-by-side You.com vs. native search: tokens, cost, latency, quality score
- **Multi-model** — run the same query across multiple models simultaneously and compare results

**decisions/** is the research log: each file documents a question we investigated, the experiment we ran, and the decision we made. Not academic — these are the actual findings that shaped the architecture here.

## Supported Models

| Key | Model | Vendor | Playground | Comparison |
|-----|-------|--------|------------|------------|
| `claude` | Claude Sonnet 4.6 | Anthropic | Yes | Yes |
| `claude-opus` | Claude Opus 4.8 | Anthropic | Yes | Yes |
| `gpt5.4` | GPT-5.4 | OpenAI | Yes | Yes |
| `gpt5.5` | GPT-5.5 | OpenAI | Yes | Yes |
| `qwen` | Qwen3.7 Max | Alibaba Cloud | Yes | Yes |
| `qwen-openrouter` | Qwen3.7 Max | Alibaba (via OpenRouter) | Yes | Yes |
| `kimi` | Kimi K2.6 | Moonshot AI | Yes | Yes |
| `kimi-openrouter` | Kimi K2.6 | Moonshot AI (via OpenRouter) | Yes | Yes |
| `llama` | Llama 4 Maverick | Meta (via OpenRouter) | Yes | Yes |
| `llama-scout` | Llama 4 Scout | Meta (via OpenRouter) | Yes | Yes |
| `deepseek` | DeepSeek V3.1 | DeepSeek (via OpenRouter) | Yes | Yes |
| `glm-5` | GLM-5 | Z.ai (via OpenRouter) | Yes | Yes |

Native web search comparison requires provider support. Claude and GPT have production web search tools. Qwen and Kimi have experimental native search. Llama, DeepSeek, GLM-5, and Gemma have no native search and are You.com-only.

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
OPENAI_API_KEY=your-key        # For GPT-5.4, GPT-5.5
DASHSCOPE_API_KEY=your-key     # For Qwen (direct)
MOONSHOT_API_KEY=your-key      # For Kimi (direct)
OPENROUTER_API_KEY=your-key    # For Llama, DeepSeek, GLM-5, and *-openrouter variants
```

### 3. Ground an LLM with web search

```bash
cd grounding
python run.py claude    "What happened in tech news today?"
python run.py gpt5.4   "Who won the 2026 Olympic hockey gold?"
python run.py kimi     "Latest NVIDIA earnings"
python run.py deepseek "Compare Tesla and BYD delivery numbers"
```

### 4. Compare You.com vs. native search

```bash
cd comparison
python compare.py claude  "What happened in tech news today?"
python compare.py gpt5.4  "Current S&P 500 price"
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

The codebase is layered: CLI modules are standalone and importable; the UI is a thin HTTP + SSE wrapper.

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
    │     grounding/agent_pool.py     │         ← shared agent cache (thread-safe)
    ├─────────────────────────────────┤
    │     grounding/search_tool.py    │         ← You.com Search API
    │     grounding/base_agent.py     │         ← tool-use loop (all providers)
    │     grounding/agents/*.py       │         ← per-provider clients
    └─────────────────────────────────┘
```

**Grounding flow:** User question → LLM decides to search → You.com returns results → LLM synthesizes a cited answer.

**Comparison flow:** Same query runs in parallel through both paths → cross-model blind judge scores completeness, relevance, specificity, citation quality → cost breakdown.

**Agent caching:** `agent_pool.py` maintains a shared, thread-safe cache of agent instances. Both `run.py` and `compare.py` pull from this pool — avoiding duplicate instantiation and TCP/TLS handshake overhead across parallel requests. The server pre-warms all agents at startup.

## Deploying

The dev server (`python server.py`) is fine for local use. For a shared or internet-facing deployment:

```bash
# Bind to localhost and put a reverse proxy in front
python server.py --port 8080
```

**Recommended setup:**
- **Reverse proxy:** nginx or Caddy in front — handles TLS termination and auth
- **Origin restriction:** set `ALLOWED_ORIGIN` in `server.py` to your domain (currently `http://localhost:8080`)
- **Auth:** add HTTP basic auth or SSO at the proxy layer — the app has no built-in auth
- **Process management:** run under `systemd` or `supervisord` so it restarts on crash

The server uses Python's built-in `http.server` with threading — adequate for small teams, not production-scale traffic.

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
