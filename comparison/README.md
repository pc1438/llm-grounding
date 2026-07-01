# Search Grounding: Cost & Quality Comparison

Side-by-side comparison of LLM + You.com Search vs. LLM + Native Web Search.

Proves two things:

1. **Cost:** You.com as a dedicated search tool uses fewer tokens and costs less
2. **Quality:** A blind cross-model judge confirms output quality is comparable or better

## Supported Providers

| Key | Model | Native Search | Judge |
|-----|-------|---------------|-------|
| `claude` | Claude Sonnet 4.6 | Anthropic Web Search ($10/1k + input token billing) | GPT-5.4 judges Claude |
| `openai` | GPT-5.4 | OpenAI Web Search via Responses API ($10/1k + search content tokens at model rates) | Claude judges GPT |

The judge is always a different LLM than the one being tested (cross-model) to avoid self-grading bias. Answers are randomly assigned to positions A/B so the judge can't infer which system produced which answer.

## Run

```bash
cd comparison
source ../grounding/venv/bin/activate

# Claude comparison + GPT judge
python compare.py claude "Who won the 2026 men's hockey Olympic gold?"

# GPT comparison + Claude judge
python compare.py openai "What happened in tech news today?"

# With full logging
python compare.py claude --verbose "Latest NVIDIA earnings"

# Skip judge evaluation (faster)
python compare.py openai --no-judge "current S&P 500 price"
```

## How It Works

1. **You.com path:** LLM + You.com Search tool (your code controls the search via function calling)
2. **Native path:** LLM + provider's built-in web search (black box)
3. **Judge:** A different LLM blindly evaluates both answers on four dimensions

### Flow

```
User query
    │
    ├──► You.com path                    ├──► Native path
    │    LLM + You.com Search tool       │    LLM + built-in web search
    │    (OpenAI chat.completions        │    (Anthropic tool_use /
    │     or Anthropic tool_use)         │     OpenAI Responses API)
    │                                    │
    ▼                                    ▼
Answer A                            Answer B
    │                                    │
    └────────────┬───────────────────────┘
                 │
                 ▼
         Cross-Model Judge
         (blind A/B evaluation)
                 │
                 ▼
    Scores + Verdict + Cost Comparison
```

## What It Measures

### Cost & Tokens

| Metric | You.com Path | Native Path |
|--------|-------------|-------------|
| Total tokens | Tracked per API call | Single API call |
| Search context | You control (snippets vs livecrawl) | Provider controls |
| Cost | Search + inference as separate line items | Per-token + per-search fees |
| Sources | Full URLs, search_uuid for audit | Limited visibility |

### Quality (Judge Evaluation)

| Dimension | What it measures |
|-----------|-----------------|
| Completeness | Does the answer fully address the question? |
| Relevance | Is the information on-topic and useful? |
| Specificity | Does the answer include concrete details, numbers, dates? |
| Citation Quality | Are sources cited, numbered, and traceable? |

## Search Cost Reference

| Provider | Cost per 1,000 searches | Notes |
|----------|------------------------|-------|
| You.com Search API | $5 | Predictable per-query pricing |
| Anthropic Web Search | $10 | + search results billed as input tokens |
| OpenAI Web Search | $10 | + search content tokens billed at model rates |

## Setup

Uses the same API keys as `../grounding/`. Set in `../grounding/env.txt`:

```
YDC_API_KEY=your-key
ANTHROPIC_API_KEY=your-key
OPENAI_API_KEY=your-key
```

Both keys are needed for cross-model judging (GPT judges Claude, Claude judges GPT). Use `--no-judge` to skip if you only have one provider key.

## Programmatic Usage

`compare.py` is also importable — the web UI in `../app/` consumes it as a library:

```python
from compare import MODELS, run_youdotcom, run_native, run_judge, _create_client

model_config = MODELS["claude"]
client = _create_client(model_config)

ydc_stats = run_youdotcom(query, client, model_config)
native_stats = run_native(query, client, model_config)
judge_result = run_judge(query, ydc_stats["answer"], native_stats["answer"],
                         model_config["judge"],
                         sources_ydc=ydc_stats["sources"],
                         sources_native=native_stats["sources"])
```

## Web UI

The standalone comparison server (`server.py` + `index.html`) still works for development, but the unified app in `../app/` is the primary UI. It includes the comparison tool as one of three tabs.

```bash
# Standalone (comparison only)
python server.py

# Unified app (recommended — includes grounding + about tabs)
cd ../app && python server.py
```

## Extending for Other LLMs

The `MODELS` dict in `compare.py` defines each provider. To add a new LLM, add an entry with: model ID, provider type, native search tool config, pricing, and judge assignment. You'll also need to implement the provider-specific `_run_youdotcom_<provider>()` and `_run_native_<provider>()` functions following the existing Anthropic/OpenAI patterns.
