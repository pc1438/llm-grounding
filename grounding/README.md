# LLM Grounding with You.com Web Search API

Reduce LLM hallucinations by giving your AI agent access to real-time web search. The agent decides when to search. You.com handles the retrieval. This repo provides a modular, class-based Python library that works with Claude, GPT-5.4, Qwen, Kimi, and Llama.

**What this is:** Importable Python classes you bring into your own project. Not a standalone app, not a notebook, not a REST service. You `from agents.claude_agent import ClaudeAgent` in your Flask backend, data pipeline, CLI tool, or wherever you need grounded LLM responses. The agents handle the tool-use loop internally. Your code just calls `agent.ask(question)` and gets back a grounded, cited answer.

Read the full article: [article.md](./article/article.md) | Full docs: [docs.you.com](https://docs.you.com)

## Prerequisites

- **Python 3.10+** (uses `str | None` union syntax)
- **You.com API key** — sign up at [you.com/platform](https://you.com/platform) ($100 free credits, no credit card)
- **At least one LLM provider key** — see the environment variables table below

## How It Works

```
User asks a question
       │
       ▼
┌─────────────────┐
│   LLM / Agent   │  ← "I need current data for this..."
│  (your choice)  │
└───────┬─────────┘
        │ tool call: web_search(query="...")
        ▼
┌─────────────────┐
│  You.com Search │  ← executes search, returns results
│   (search_tool) │
└───────┬─────────┘
        │ results back to agent
        ▼
┌─────────────────┐
│   LLM / Agent   │  ← reads results, cites sources
│  generates answer│
└───────┬─────────┘
        │
        ▼
  Grounded, Cited Answer
```

The LLM is the orchestrator. You.com Search is a tool it can call. The agent decides *when* to search and *what* to search for based on the user's question.

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/youdotcom-oss/grounding.git
cd grounding
pip install -r requirements.txt
```

### 2. Set your API keys

Copy the example env file and add your keys. **Never commit your env file to version control** — both `.env` and `env.txt` are in `.gitignore`.

```bash
cp .env.example env.txt
# Edit env.txt with your API keys
```

The code loads `env.txt` first, then falls back to `.env`. Use whichever you prefer. We default to `env.txt` because `.env` files are hidden on macOS and easy to miss in Finder.

You need `YDC_API_KEY` (required for all integration paths) plus the key for whichever LLM you want to use:

| Variable | Required for | Where to get it |
|----------|-------------|-----------------|
| `YDC_API_KEY` | All paths (Direct API, MCP, LangChain) | [you.com/platform](https://you.com/platform) — $100 free credits |
| `ANTHROPIC_API_KEY` | Claude agent | [console.anthropic.com](https://console.anthropic.com/) |
| `OPENAI_API_KEY` | GPT-5.4 agent, LangChain path | [platform.openai.com](https://platform.openai.com/) |
| `DASHSCOPE_API_KEY` | Qwen via DashScope | [dashscope.console.aliyun.com](https://dashscope.console.aliyun.com/) |
| `HF_TOKEN` | Qwen via Hugging Face | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |
| `MOONSHOT_API_KEY` | Kimi agent | [platform.moonshot.ai](https://platform.moonshot.ai/) |
| `TOGETHER_API_KEY` | Llama via Together AI | [api.together.ai](https://api.together.ai/) |
| `GROUNDING_VERBOSE` | Optional. Set to `1` for full tool call logging (query, search_uuid, latency, interface). Default `0`. | Set in `.env` or pass `--verbose` flag |

If a required API key is missing, the agent raises a clear `ValueError` at init time telling you which variable to set.

### 3. Use in your code

Run from the project root directory (imports resolve relative to it):

```python
from agents.claude_agent import ClaudeAgent

agent = ClaudeAgent(model="claude-sonnet-4-6")
result = agent.ask("What happened in tech news today?")

print(result["answer"])      # grounded response with citations
print(result["sources"])     # URLs the agent used
print(result["tool_calls"])  # what it searched for
```

That's the core pattern. Import an agent, call `agent.ask()`, get a grounded answer. Works the same for all five LLMs:

```python
from agents.openai_agent import OpenAIAgent        # GPT-5.4
from agents.qwen_agent import QwenAgent             # Qwen (DashScope or HF)
from agents.kimi_agent import KimiAgent             # Kimi K2.5
from agents.llama_agent import LlamaAgent           # Llama 4 (Together or Ollama)
```

### 4. Example: add to a FastAPI backend

```python
from fastapi import FastAPI
from agents.openai_agent import OpenAIAgent

app = FastAPI()
agent = OpenAIAgent(model="gpt-5.4")

@app.post("/ask")
def ask(question: str):
    result = agent.ask(question)
    return {"answer": result["answer"], "sources": result["sources"]}
```

### 5. Try the interactive demos

**Direct API** (your code runs the agent loop):

```bash
python demo.py claude             # Claude (default)
python demo.py openai             # GPT-5.4
python demo.py claude --verbose   # With full tool call logging
```

**MCP** (calls You.com's MCP server, same one IDEs use):

```bash
python mcp_demo.py                          # List available MCP tools
python mcp_demo.py search "latest AI news"  # Search via MCP
python mcp_demo.py search "query" --verbose # With full logging
```

**LangChain** (ReAct agent via langchain-youdotcom):

```bash
python langchain_agent.py                   # ReAct agent (default)
python langchain_agent.py research          # Multi-tool research agent
python langchain_agent.py rag               # RAG retriever
python langchain_agent.py --verbose         # With full logging
```

**Agent Skills**: Drop `skill_youdotcom_search.md` into your agent runtime (Claude Agent SDK, OpenAI Agents SDK, etc.). No code to run — the .md file IS the integration.

All demos support `--verbose` (or set `GROUNDING_VERBOSE=1` in `.env`) for full tool call logging: query, parameters, search_uuid, latency, and integration interface.

See **Integration Patterns** below for a comparison of when to use which path.

## Unified CLI: run.py

`run.py` is the single entry point for running any grounded LLM with full trace output. It supports all five providers, shows the complete tool-use loop, and reports token/cost breakdowns.

```bash
python run.py claude "What happened in tech news today?"
python run.py gpt    "Who won the 2026 Olympic hockey gold?"
python run.py qwen   "Latest NVIDIA earnings"
python run.py kimi   "Current S&P 500 price"
python run.py llama  "Compare Tesla and BYD delivery numbers"

python run.py claude --verbose "query here"
```

`run.py` is also importable — the web UI in `../app/` consumes it as a library:

```python
from run import run_grounding, GROUNDING_MODELS

# Generator yields events as the tool-use loop progresses
for event in run_grounding("claude", "What happened today?"):
    print(event["event"], event.get("query", ""))
```

Events: `init` → `tool_call` → `search_result` → `answer` → `done` (or `error`). The `done` event contains full stats (tokens, latency, sources).

## Project Structure

```
grounding/
├── run.py                     # Unified CLI + generator API for all 5 LLMs
├── search_tool.py             # Tool definition + You.com search execution
├── base_agent.py              # Base classes: tool-use loop for all providers
├── agents/
│   ├── __init__.py
│   ├── claude_agent.py        # Claude (Anthropic tool_use format)
│   ├── openai_agent.py        # GPT-5.4 (OpenAI function calling)
│   ├── qwen_agent.py          # Qwen (DashScope / Hugging Face)
│   ├── kimi_agent.py          # Kimi K2.5 (Moonshot)
│   └── llama_agent.py         # Llama 4 (Together AI / Ollama)
├── demo.py                    # Interactive demo — Direct API path (any LLM)
├── langchain_agent.py         # LangChain path — ReAct, multi-tool, RAG
├── mcp_demo.py                # MCP path — Python client for You.com MCP server
├── mcp_config.json            # MCP config — copy-paste for IDE setup
├── skill_youdotcom_search.md  # Agent Skills path — .md file for agent runtimes
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

**Note on imports:** This repo is designed to run from the project root (`cd grounding`). The agents use `from base_agent import ...` and `from search_tool import ...`, which resolve via Python's default path. If you need to import from elsewhere, add the project root to `PYTHONPATH` or `sys.path`.

## Two Architectures

This codebase implements **Architecture B (agent-orchestrated)**:

| | Architecture A | Architecture B (this repo) |
|---|---|---|
| **Who decides to search** | Your application code | The LLM/agent |
| **Flow** | App calls search → formats context → injects into prompt → calls LLM | User talks to LLM → LLM calls search tool → reads results → responds |
| **Pattern** | Context injection / RAG | Function calling / tool use |
| **Control** | Deterministic — every response is grounded | Autonomous — agent searches when it needs to |
| **Best for** | Pipelines where you always want grounding | Conversational agents, multi-step reasoning |

Both architectures reduce hallucinations. Architecture A force-grounds every response. Architecture B lets the agent decide, which is more natural but means the agent might sometimes skip searching when it shouldn't (solvable via system prompt tuning).

For Architecture A, see `langchain_agent.py` → `rag_retriever()`.

## Snippets vs. Livecrawl

The search tool exposes both modes to the LLM. The agent can decide which to use:

| Mode | Tokens/search | Best for |
|------|---------------|----------|
| **Snippets** (default) | ~500 | Factual lookups, smaller models, cost optimization |
| **Livecrawl** (`livecrawl=true`) | ~15,000 | Deep analysis, large context models |

~30x cost difference in inference tokens. The tool definition explains both modes to the LLM so it can make the right call.

## Integration Patterns

This repo implements all four ways to integrate You.com search. Each has its own runnable entry point, verbose logging (`--verbose` flag or `GROUNDING_VERBOSE=1`), and interface label so you always know which path is active.

| Pattern | Run it | What it does | When to use |
|---------|--------|--------------|-------------|
| **Direct API** | `python demo.py claude` | Your code runs the tool-use loop. Full control over search params, domain filtering, snippets vs. livecrawl. Logs `search_uuid` for audit. | Production backends, custom pipelines, anywhere you need full control. |
| **MCP** | `python mcp_demo.py search "query"` | Connects to You.com's MCP server (`api.you.com/mcp`). Same server your IDE uses. `mcp_config.json` has the IDE copy-paste config. | IDE integrations (Claude Desktop, Cursor, VS Code, Claude Code), or testing the MCP flow from Python. |
| **LangChain** | `python langchain_agent.py` | ReAct agent, multi-tool research agent, or RAG retriever using `langchain-youdotcom`. | LangChain/LangGraph ecosystems, RAG chains. |
| **Agent Skills** | Drop `skill_youdotcom_search.md` into your agent runtime | Structured .md instructions. The skill file IS the integration. Zero code. | Claude Agent SDK, OpenAI Agents SDK, rapid prototyping. |

**Which one should I pick?** If you're building a production backend, start with Direct API. If you're already using LangChain, use the LangChain path. If you want search in your IDE, use the MCP config. If you want zero-code prototyping, use the skill file.

## Extending This Codebase

### Adding a New LLM

If the new model uses OpenAI-compatible function calling (most do):

```python
# agents/mistral_agent.py
import os
from openai import OpenAI
from base_agent import OpenAICompatibleAgent

class MistralAgent(OpenAICompatibleAgent):
    def __init__(self, model="mistral-large-latest", api_key=None):
        resolved_key = api_key or os.environ.get("MISTRAL_API_KEY", "")
        if not resolved_key:
            raise ValueError(
                "MISTRAL_API_KEY not set. "
                "Pass api_key= or export MISTRAL_API_KEY."
            )
        client = OpenAI(
            api_key=resolved_key,
            base_url="https://api.mistral.ai/v1",
        )
        super().__init__(client=client, model=model)
```

That's it. The base class handles the entire tool-use loop. Your subclass just configures the client.

If the model has a non-OpenAI tool format, subclass `BaseAgent` directly and implement the `ask()` method (see `AnthropicAgent` in `base_agent.py` for the pattern).

### Customizing the Search Tool

Edit `search_tool.py` → `TOOL_SCHEMA` to change what the LLM sees:
- Adjust the description to change when the LLM decides to search
- Add/remove parameters to control what the LLM can configure
- Modify `execute_search()` to change result formatting

## Security Notes

- **API keys** are loaded from environment variables, never hardcoded. The `.env` file is in `.gitignore`. If you see a key in version control, rotate it immediately.
- **You.com Zero Data Retention (ZDR):** You.com does not store your search queries or results. Each response includes a `search_uuid` for correlation. Since You.com has ZDR, your agent is responsible for logging this UUID alongside queries, sources, and responses for audit traceability. The agents return structured `tool_calls` data (including `search_uuid`) and `sources` for this purpose.
- **No data egress with Ollama:** Running `LlamaAgent(backend="ollama")` keeps all LLM inference local. Only the search query leaves your infrastructure (to You.com's API).

## API Reference

- **You.com Search API:** `GET https://ydc-index.io/v1/search` — [docs](https://docs.you.com/api-reference/search)
- **You.com MCP Server:** `https://api.you.com/mcp` — [docs](https://docs.you.com/developer-resources/mcp-server)
- **LangChain package:** `pip install langchain-youdotcom` — [docs](https://docs.you.com/integrations/langchain)
- **Agent Skills:** `npx skills add youdotcom-oss/agent-skills` — [GitHub](https://github.com/youdotcom-oss/agent-skills)

## Get Your API Key

Sign up at [you.com/platform](https://you.com/platform) — $100 in free credits, no credit card required.

## License

MIT
