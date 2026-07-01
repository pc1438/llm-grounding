# Unified Web UI

Browser-based interface combining the grounding trace viewer and cost comparison dashboard into a single three-tab application.

This is a thin HTTP wrapper around the CLI modules in `../grounding/` and `../comparison/`. All logic lives in those modules — this server just exposes them as SSE endpoints and serves the frontend.

## Run

```bash
cd app
python server.py                  # http://localhost:8080
python server.py --port 9000      # custom port
```

Then open `http://localhost:8080` in your browser.

## Tabs

### About

Architecture overview for the entire project: flow diagrams for both the grounding and comparison workflows, a table of all five supported LLMs, search cost reference, and the code structure.

### Grounding

Interactive trace viewer for the grounding tool-use loop. Pick any of the five LLMs from the dropdown, enter a query, and watch the agent work step by step: initialization → tool calls → search results → synthesized answer → final stats.

Events stream progressively via SSE — each step appears as it completes, so you see the agent's reasoning in real time.

### Comparison

Side-by-side cost and quality dashboard. Runs the same query through LLM + You.com Search and LLM + Native Web Search, then shows token counts, cost breakdowns, a savings summary, and the cross-model judge's blind evaluation.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Serves `index.html` |
| `POST` | `/api/grounding` | SSE stream of grounding trace events |
| `POST` | `/api/compare` | SSE stream of comparison events |
| `POST` | `/api/models` | JSON list of available models |

### /api/grounding

Request body: `{"model": "claude", "query": "What happened today?"}`

SSE events: `init`, `tool_call`, `search_result`, `answer`, `done`, `error`

### /api/compare

Request body: `{"provider": "claude", "query": "...", "skip_judge": false}`

SSE events: `status`, `youdotcom`, `native`, `judge`, `done`, `error`

## Architecture

```
app/
├── server.py      # HTTP server — imports from grounding/ and comparison/
├── index.html     # Single-page app with 3 tabs (all JS/CSS inline)
└── README.md
```

`server.py` adds `../grounding/` and `../comparison/` to `sys.path` and imports directly from `run.py` and `compare.py`. It uses Python's built-in `http.server` with HTTP/1.0 and unbuffered socket writes so SSE events flush immediately without chunked encoding issues.

## Dependencies

No additional dependencies beyond what `../grounding/requirements.txt` provides. The frontend is a single self-contained HTML file with no build step.
