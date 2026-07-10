# Unified Web UI

Browser-based interface for the grounding playground, side-by-side comparison, and multi-model runs — four tabs, one server.

This is a thin HTTP + SSE wrapper around the CLI modules in `../grounding/` and `../comparison/`. All logic lives in those modules — this server just exposes them as streaming endpoints and serves the frontend.

## Run

```bash
cd app
python server.py                  # http://localhost:8080
python server.py --port 9000      # custom port
```

Then open `http://localhost:8080` in your browser.

## Tabs

### About
Architecture overview, flow diagrams, model table, search cost reference, and a live **Configuration Reference** — search loop settings, token limits, system prompt, and model-specific prompt overrides, all pulled dynamically from `/api/config` and `/api/pricing`.

### LLM + You.com Playground
Pick any of the twelve supported models, enter a query, and watch the tool-use loop step by step: initialization → search calls → results → synthesized answer. Final stats include a full token, cost, and timing breakdown (connection time, end-to-end latency, time to first activity). Events stream via SSE in real time.

### You.com vs. Native Search
Same query, two paths: LLM + You.com Search API vs. LLM + provider's built-in web search. Shows token counts, cost breakdown, latency, savings summary, and a cross-model blind judge score.

### Multi-Model You.com
Run the same query across up to five models simultaneously. Results appear as each model completes; a unified metrics table compares tokens, cost, and timing side by side.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Serves `index.html` |
| `GET` | `/api/pricing` | Full model registry from `pricing.json` |
| `GET` | `/api/config` | Search loop config: max rounds, token limits, system prompt, overrides |
| `POST` | `/api/grounding` | SSE stream — playground trace events |
| `POST` | `/api/compare` | SSE stream — comparison events |
| `POST` | `/api/multi` | SSE stream — multi-model parallel events |
| `GET` | `/api/models` | Available models for comparison tab |

### /api/grounding

Request body: `{"model": "claude", "query": "What happened today?"}`

SSE events: `init`, `tool_call`, `search_result`, `answer`, `done`, `error`

### /api/compare

Request body: `{"provider": "claude", "query": "...", "skip_judge": false}`

SSE events: `status`, `youdotcom`, `native`, `judge`, `done`, `error`

### /api/multi

Request body: `{"providers": ["claude", "gpt5.4", "kimi"], "query": "..."}`

SSE events: `model_progress`, `model_result`, `model_error`, `all_done`

## Architecture

```
app/
├── server.py      # HTTP + SSE server — pre-warms agents at startup
├── index.html     # Single-page app, 4 tabs (all JS/CSS inline, no build step)
└── README.md
```

`server.py` imports directly from `grounding/run.py` and `comparison/compare.py`. It pre-warms all agent instances at startup via `grounding/agent_pool.py` to prevent threading races under parallel multi-model requests. SSE uses HTTP/1.0 with unbuffered socket writes so events flush immediately.

## Dependencies

No additional dependencies beyond what `../grounding/requirements.txt` provides. The frontend is a single self-contained HTML file with no build step.
