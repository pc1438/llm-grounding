"""
server.py — Web server for the You.com Search grounding & comparison UI.

Thin HTTP layer on top of two CLI-first modules:
  - grounding/run.py   → /api/grounding  (single LLM + You.com Search trace)
  - comparison/compare.py → /api/compare  (side-by-side: You.com vs. native search)

Both endpoints use Server-Sent Events (SSE) for progressive rendering.
The frontend (index.html) has three tabs: About, Grounding, Comparison.

Run:
    python server.py                  # starts on http://localhost:8080
    python server.py --port 9000      # custom port

Then open http://localhost:8080 in your browser.

The CLI modules work standalone — this server just wraps them for the UI.
"""

import json
import logging
import logging.handlers
import os
import sys
import threading
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn
from pathlib import Path
from dotenv import load_dotenv

# ─── Path setup ────────────────────────────────────────────────────────────
# This file lives in use-cases/app/. We need to reach:
#   use-cases/grounding/  (for run.py, search_tool.py, agents/)
#   use-cases/comparison/ (for compare.py)

APP_DIR = Path(__file__).parent
USE_CASES_DIR = APP_DIR.parent
GROUNDING_DIR = USE_CASES_DIR / "grounding"
COMPARISON_DIR = USE_CASES_DIR / "comparison"
LOG_DIR = USE_CASES_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Load env first so LOG_LEVEL and API keys are available before anything else
load_dotenv(GROUNDING_DIR / "env.txt")
load_dotenv(GROUNDING_DIR / ".env")
load_dotenv(COMPARISON_DIR / "env.txt")
load_dotenv(COMPARISON_DIR / ".env")

# ─── Logging ───────────────────────────────────────────────────────────────
# Control verbosity with LOG_LEVEL in env.txt:
#   DEBUG   — full request/response detail, useful during development
#   INFO    — normal operational messages (default)
#   WARNING — only unexpected conditions
#   ERROR   — only failures; recommended for production / check-in
_log_level = getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO)
_fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
_file_handler = logging.handlers.RotatingFileHandler(
    LOG_DIR / "server.log", maxBytes=10_000_000, backupCount=3, encoding="utf-8"
)
_file_handler.setFormatter(_fmt)
_stderr_handler = logging.StreamHandler()
_stderr_handler.setFormatter(_fmt)
logging.basicConfig(level=_log_level, handlers=[_stderr_handler, _file_handler])

logger = logging.getLogger(__name__)
logger.info("=" * 60)
logger.info("SERVER START (level=%s)", logging.getLevelName(_log_level))
logger.info("=" * 60)

# Add both dirs to path so imports work
sys.path.insert(0, str(GROUNDING_DIR))
sys.path.insert(0, str(COMPARISON_DIR))

# Import from grounding/run.py
from run import GROUNDING_MODELS, run_grounding
from search_tool import MAX_TOOL_ROUNDS, MAX_TOKENS, SYSTEM_PROMPT, _GPT_55_SEARCH_RULES

# Import from comparison/compare.py
from compare import (
    MODELS,
    run_youdotcom, run_native,
    describe_native_search,
    calculate_costs,
)

# Load pricing.json directly — same source of truth compare.py uses, no cross-module dependency
_PRICING_FILE = Path(__file__).parent.parent / "comparison" / "pricing.json"
try:
    with open(_PRICING_FILE) as _f:
        _PRICING_DATA = json.load(_f)
except Exception as _e:
    logging.error("Failed to load pricing.json: %s", _e)
    _PRICING_DATA = {"models": {}}

PLAYGROUND_MODELS = {k: v for k, v in _PRICING_DATA.get("models", {}).items() if v.get("in_playground")}
from judge import run_judge
from agent_pool import prewarm as _prewarm_agents

# Pre-warm all agents in the main thread before any parallel requests arrive.
# Prevents threading deadlocks caused by lazy module initialization races.
_prewarm_agents(_PRICING_DATA.get("models", {}))


# ─── Request handler ───────────────────────────────────────────────────────

class AppHandler(SimpleHTTPRequestHandler):
    """Serves index.html and handles API requests via SSE.

    Endpoints:
        GET  /                → index.html
        POST /api/grounding   → SSE stream of grounding trace events
        POST /api/compare     → SSE stream of comparison events
    """

    # HTTP/1.0 + unbuffered wfile = SSE bytes flush immediately.
    # HTTP/1.1 without Content-Length triggers chunked encoding that
    # Python's http.server doesn't implement, causing buffering.
    protocol_version = "HTTP/1.0"

    def setup(self):
        """Override to make wfile unbuffered for SSE streaming."""
        super().setup()
        self.wfile = self.request.makefile("wb", buffering=0)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            # Serve only the known HTML file — do NOT fall through to
            # SimpleHTTPRequestHandler, which would serve any file in cwd
            # (including env.txt, .env, or other sensitive files).
            html_path = Path(__file__).parent / "index.html"
            try:
                with open(html_path, "rb") as f:
                    content = f.read()
            except FileNotFoundError:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Access-Control-Allow-Origin", self.ALLOWED_ORIGIN)
            self.end_headers()
            self.wfile.write(content)
        elif self.path == "/api/pricing":
            pricing_file = Path(__file__).parent.parent / "comparison" / "pricing.json"
            self._serve_json_file(pricing_file)
        elif self.path == "/api/config":
            self._send_json(200, {
                "max_tool_rounds": MAX_TOOL_ROUNDS,
                "max_tokens": MAX_TOKENS,
                "system_prompt": SYSTEM_PROMPT,
                "prompt_overrides": {
                    "gpt-5.5": _GPT_55_SEARCH_RULES,
                },
            })
        elif self.path == "/api/questions":
            questions_file = Path(__file__).parent.parent / "comparison" / "benchmark_questions_50.json"
            self._serve_json_file(questions_file)
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/api/grounding":
            self._handle_grounding()
        elif self.path == "/api/compare":
            self._handle_compare()
        elif self.path == "/api/multi-compare":
            self._handle_multi_compare()
        elif self.path == "/api/models":
            self._handle_models()
        else:
            self.send_error(404)

    MAX_BODY_SIZE = 10_000      # 10 KB max request body
    MAX_QUERY_LENGTH = 2_000    # 2,000 chars max query
    ALLOWED_ORIGIN = "http://localhost:8080"

    def _read_json_body(self) -> dict | None:
        """Read and parse the JSON request body. Returns None on error."""
        try:
            content_length = int(self.headers.get("Content-Length", 0))
        except ValueError:
            content_length = 0
        if content_length < 0 or content_length > self.MAX_BODY_SIZE:
            self._send_json(400, {"error": "Request body too large"})
            return None
        body = self.rfile.read(content_length).decode("utf-8")
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            self._send_json(400, {"error": "Invalid JSON"})
            return None

    def _serve_json_file(self, path: Path):
        """Open a JSON file and send its contents as a 200 response.

        Returns HTTP 500 with a generic message on any I/O or parse error.
        """
        try:
            with open(path) as f:
                data = json.load(f)
        except FileNotFoundError:
            logger.error("JSON file not found: %s", path)
            self._send_json(500, {"error": "Resource not found on server"})
            return
        except json.JSONDecodeError as e:
            logger.error("JSON parse error in %s: %s", path, e)
            self._send_json(500, {"error": "Server data error"})
            return
        except OSError as e:
            logger.error("OS error reading %s: %s", path, e)
            self._send_json(500, {"error": "Server data error"})
            return
        self._send_json(200, data)

    def _send_json(self, status: int, data: dict):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", self.ALLOWED_ORIGIN)
        self.end_headers()
        self.wfile.write(body)

    def _start_sse(self):
        """Send SSE response headers."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Access-Control-Allow-Origin", self.ALLOWED_ORIGIN)
        self.end_headers()
        self.wfile.flush()

    def _send_sse(self, event: str, data: dict):
        """Send one Server-Sent Event."""
        try:
            payload = f"event: {event}\ndata: {json.dumps(data)}\n\n"
            self.wfile.write(payload.encode("utf-8"))
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _make_sse_sender(self):
        """Return a thread-safe SSE sender for this request.

        Each endpoint that fans out to threads needs its own lock to prevent
        interleaved writes on the shared socket.
        """
        lock = threading.Lock()
        def send_safe(event, data):
            with lock:
                self._send_sse(event, data)
        return send_safe

    # ── /api/models ────────────────────────────────────────────────────────

    def _handle_models(self):
        """Return available models for both grounding and comparison."""
        self._send_json(200, {
            "grounding": {
                k: {"model": v["model"], "display_name": v["display_name"], "vendor": v["vendor"]}
                for k, v in GROUNDING_MODELS.items()
            },
            "comparison": {
                k: {"model": v["model"], "provider": v["provider"]}
                for k, v in MODELS.items()
            },
        })

    # ── /api/grounding ─────────────────────────────────────────────────────

    def _handle_grounding(self):
        """Stream grounding trace events as SSE.

        Consumes the same generator that the CLI uses (run.py),
        but sends events as SSE instead of printing to console.
        """
        params = self._read_json_body()
        if params is None:
            return

        query = params.get("query", "").strip()
        model_key = params.get("model", "claude")
        max_rounds = params.get("max_rounds", None)
        if max_rounds is not None:
            try:
                max_rounds = max(1, min(15, int(max_rounds)))
            except (TypeError, ValueError):
                max_rounds = None

        if not query:
            self._send_json(400, {"error": "Query is required"})
            return
        if len(query) > self.MAX_QUERY_LENGTH:
            self._send_json(400, {"error": f"Query too long (max {self.MAX_QUERY_LENGTH} chars)"})
            return

        if model_key not in GROUNDING_MODELS:
            self._send_json(400, {"error": f"Unknown model: {model_key}"})
            return

        ydc_key = os.environ.get("YDC_API_KEY", "")
        if not ydc_key:
            self._send_json(500, {"error": "YDC_API_KEY not configured"})
            return

        model_config = PLAYGROUND_MODELS.get(model_key, {})
        self._start_sse()

        t_start = time.perf_counter()
        first_ms = None

        for event in run_grounding(model_key, query, max_rounds=max_rounds):
            etype = event["event"]

            if first_ms is None and etype not in ("init",):
                first_ms = round((time.perf_counter() - t_start) * 1000)

            if etype == "done":
                stats = event.get("stats", {})
                overhead_ms = round((time.perf_counter() - t_start) * 1000 - stats.get("latency_ms", 0))
                costs = calculate_costs(stats, model_config)
                event = dict(event,
                    overhead_ms=max(0, overhead_ms),
                    first_ms=first_ms or 0,
                    cost=round(costs["llm"] + costs["search"], 6),
                    cost_llm=round(costs["llm"], 6),
                    cost_search=round(costs["search"], 6),
                    pricing={
                        "input_cost_per_m": model_config["input_cost_per_m"],
                        "output_cost_per_m": model_config["output_cost_per_m"],
                        "ydc_search_per_1k": model_config["ydc_search_cost_per_call"] * 1000,
                    },
                )

            self._send_sse(etype, event)

    # ── /api/compare ───────────────────────────────────────────────────────

    def _handle_compare(self):
        """Stream comparison events as SSE.

        Same logic as comparison/server.py, but now living in app/.
        """
        params = self._read_json_body()
        if params is None:
            return

        query = params.get("query", "").strip()
        provider = params.get("provider", "claude")
        skip_judge = params.get("skip_judge", False)
        max_rounds = params.get("max_rounds", None)
        if max_rounds is not None:
            try:
                max_rounds = max(1, min(15, int(max_rounds)))
            except (TypeError, ValueError):
                max_rounds = None

        if not query:
            self._send_json(400, {"error": "Query is required"})
            return
        if len(query) > self.MAX_QUERY_LENGTH:
            self._send_json(400, {"error": f"Query too long (max {self.MAX_QUERY_LENGTH} chars)"})
            return

        if provider not in MODELS:
            self._send_json(400, {"error": f"Unknown provider: {provider}"})
            return

        model_config = MODELS[provider]

        ydc_key = os.environ.get("YDC_API_KEY", "")
        if not ydc_key:
            self._send_json(500, {"error": "YDC_API_KEY not configured"})
            return

        self._start_sse()

        # ── Steps 1 & 2: Run You.com and Native paths in parallel ──
        send_sse_safe = self._make_sse_sender()

        send_sse_safe("status", {"step": "both", "message": "Running You.com + Native paths in parallel..."})

        # Shared results — each thread writes to its slot
        ydc_result = {"stats": None, "error": None, "thread_ms": None, "first_ms": None}
        native_result = {"stats": None, "error": None, "thread_ms": None, "first_ms": None}

        t_threads_start = time.perf_counter()

        def make_progress_fn(result_dict, path):
            def fn(msg):
                if result_dict["first_ms"] is None:
                    result_dict["first_ms"] = round((time.perf_counter() - t_threads_start) * 1000)
                send_sse_safe("progress", {"path": path, "message": msg})
            return fn

        def run_ydc_thread():
            t0 = time.perf_counter()
            try:
                ydc_result["stats"] = run_youdotcom(query, model_config,
                    on_progress=make_progress_fn(ydc_result, "ydc"),
                    max_rounds=max_rounds)
                ydc_result["thread_ms"] = (time.perf_counter() - t0) * 1000
            except Exception as e:
                logger.error("You.com path failed: %s", e, exc_info=True)
                ydc_result["error"] = True

        def run_native_thread():
            t0 = time.perf_counter()
            try:
                native_result["stats"] = run_native(query, model_config,
                    on_progress=make_progress_fn(native_result, "native"))
                native_result["thread_ms"] = (time.perf_counter() - t0) * 1000
            except Exception as e:
                logger.error("Native path failed: %s", e, exc_info=True)
                native_result["error"] = True

        t_ydc = threading.Thread(target=run_ydc_thread)
        t_native = threading.Thread(target=run_native_thread)
        t_ydc.start()
        t_native.start()
        t_ydc.join()
        t_native.join()

        # ── Emit results (whichever finished) ──
        if ydc_result["error"]:
            send_sse_safe("error", {"message": "You.com search failed. Please try again."})
            return

        ydc_stats = ydc_result["stats"]
        tb = ydc_stats["token_breakdown"]
        _ydc_overhead = ydc_result.get("thread_ms", ydc_stats["latency_ms"]) - ydc_stats["latency_ms"]
        logger.info("[compare/ydc] tokens=%d in=%d out=%d ctx=%d searches=%d api_calls=%d latency=%.0fms connect=%.0fms overhead=%.0fms",
            ydc_stats["tokens_used"], tb["input"], tb["output"], tb["search_context"],
            ydc_stats["search_calls"], ydc_stats["api_calls"], ydc_stats["latency_ms"],
            ydc_stats.get("connect_ms", 0), _ydc_overhead)
        _ydc = calculate_costs(ydc_stats, model_config)
        ydc_llm, ydc_search, ydc_cost = _ydc["llm"], _ydc["search"], _ydc["total"]
        logger.info("[compare/ydc] cost=llm:$%.6f search:$%.6f total:$%.6f", ydc_llm, ydc_search, ydc_cost)

        send_sse_safe("ydc", {
            "answer": ydc_stats["answer"],
            "total_tokens": ydc_stats["tokens_used"],
            "input_tokens": ydc_stats["token_breakdown"]["input"],
            "output_tokens": ydc_stats["token_breakdown"]["output"],
            "search_context_tokens": ydc_stats["token_breakdown"]["search_context"],
            "api_calls": ydc_stats["api_calls"],
            "search_calls": ydc_stats["search_calls"],
            "sources": ydc_stats["sources"],
            "search_uuid": ydc_stats.get("search_uuid", ""),
            "latency_ms": round(ydc_stats["latency_ms"]),
            "connect_ms": ydc_stats.get("connect_ms", 0),
            "overhead_ms": round(_ydc_overhead),
            "first_ms": ydc_result.get("first_ms") or 0,
            "cost": round(ydc_cost, 6),
            "cost_llm": round(ydc_llm, 6),
            "cost_search": round(ydc_search, 6),
            "model_confirmed": ydc_stats.get("model_confirmed"),
        })

        if native_result["error"]:
            send_sse_safe("error", {"message": "Native search failed. Please try again."})
            return

        native_stats = native_result["stats"]

        if native_stats.get("not_supported"):
            send_sse_safe("native_unavailable", {"message": "Native web search is not supported for this model."})
            return

        tb = native_stats["token_breakdown"]
        _native_overhead = native_result.get("thread_ms", native_stats["latency_ms"]) - native_stats["latency_ms"]
        logger.info("[compare/native] tokens=%d in=%d out=%d ctx=%d searches=%d api_calls=%d latency=%.0fms connect=%.0fms overhead=%.0fms",
            native_stats["tokens_used"], tb["input"], tb["output"], tb["search_context"],
            native_stats["search_calls"], native_stats["api_calls"], native_stats["latency_ms"],
            native_stats.get("connect_ms", 0), _native_overhead)
        _native = calculate_costs(native_stats, model_config, path="native")
        native_llm, native_search, native_cost = _native["llm"], _native["search"], _native["total"]
        logger.info("[compare/native] cost=llm:$%.6f search:$%.6f total:$%.6f", native_llm, native_search, native_cost)

        send_sse_safe("native", {
            "answer": native_stats["answer"],
            "total_tokens": native_stats["tokens_used"],
            "input_tokens": native_stats["token_breakdown"]["input"],
            "output_tokens": native_stats["token_breakdown"]["output"],
            "search_context_tokens": native_stats["token_breakdown"]["search_context"],
            "api_calls": native_stats["api_calls"],
            "search_calls": native_stats["search_calls"],
            "sources": native_stats["sources"],
            "latency_ms": round(native_stats["latency_ms"]),
            "connect_ms": native_stats.get("connect_ms", 0),
            "overhead_ms": round(_native_overhead),
            "first_ms": native_result.get("first_ms") or 0,
            "cost": round(native_cost, 6),
            "cost_llm": round(native_llm, 6),
            "cost_search": round(native_search, 6),
            "model_confirmed": native_stats.get("model_confirmed"),
        })

        # ── Step 3: Judge (sequential — needs both answers) ──
        judge_result = None
        if not skip_judge:
            send_sse_safe("status", {"step": "judge", "message": "Running judge evaluation..."})
            try:
                judge_result = run_judge(
                    query, ydc_stats["answer"], native_stats["answer"],
                    model_config["judge"],
                    sources_ydc=ydc_stats["sources"],
                    sources_native=native_stats["sources"],
                )
            except Exception as e:
                logger.error("Judge failed: %s", e, exc_info=True)
                judge_result = {"error": "Judge evaluation failed. Please try again."}

            send_sse_safe("judge", judge_result)

        # ── Done ──
        native_note = ""
        if model_config["provider"] == "anthropic":
            native_note = " + search results billed as input tokens"
        elif model_config["provider"] == "openai":
            native_note = " + search content tokens billed at model rates"
        elif model_config["provider"] == "kimi":
            native_note = " + search result tokens billed at model rates"
        elif model_config["provider"] == "qwen":
            native_note = " (search bundled in token cost — no separate per-search fee)"

        send_sse_safe("done", {
            "query": query,
            "model": model_config["model"],
            "provider": provider,
            "native_search_config": describe_native_search(model_config),
            "pricing": {
                "input_cost_per_m": model_config["input_cost_per_m"],
                "output_cost_per_m": model_config["output_cost_per_m"],
                "ydc_search_per_1k": model_config["ydc_search_cost_per_call"] * 1000,
                "native_search_per_1k": (model_config.get("native_search_cost_per_call") or 0) * 1000,
                "native_search_note": native_note,
                "pricing_source_url": model_config.get("pricing_source_url", ""),
                "native_search_source_url": model_config.get("native_search_source_url", ""),
            },
        })

    # ── /api/multi-compare ─────────────────────────────────────────────────

    def _handle_multi_compare(self):
        """Run the same query through 2-3 models in parallel, all via You.com Search API."""
        params = self._read_json_body()
        if params is None:
            return

        query = params.get("query", "").strip()
        providers = params.get("providers", [])
        max_rounds = params.get("max_rounds", None)
        if max_rounds is not None:
            try:
                max_rounds = max(1, min(15, int(max_rounds)))
            except (TypeError, ValueError):
                max_rounds = None

        if not isinstance(providers, list):
            self._send_json(400, {"error": "'providers' must be a list"})
            return

        if not query:
            self._send_json(400, {"error": "Query is required"})
            return
        if len(query) > self.MAX_QUERY_LENGTH:
            self._send_json(400, {"error": f"Query too long (max {self.MAX_QUERY_LENGTH} chars)"})
            return
        if not providers or not (2 <= len(providers) <= 5):
            self._send_json(400, {"error": "Provide 2-5 model providers"})
            return
        for p in providers:
            if p not in MODELS:
                self._send_json(400, {"error": f"Unknown provider: {p}"})
                return

        ydc_key = os.environ.get("YDC_API_KEY", "")
        if not ydc_key:
            self._send_json(500, {"error": "YDC_API_KEY not configured"})
            return

        self._start_sse()

        send_sse_safe = self._make_sse_sender()

        def run_slot(slot, provider):
            t_slot = time.perf_counter()
            _first_ms = [None]
            model_config = MODELS[provider]
            def _on_progress(msg):
                if _first_ms[0] is None:
                    _first_ms[0] = round((time.perf_counter() - t_slot) * 1000)
                send_sse_safe("model_progress", {"slot": slot, "message": msg})
            try:
                stats = run_youdotcom(query, model_config, on_progress=_on_progress, max_rounds=max_rounds)
                _slot_overhead = (time.perf_counter() - t_slot) * 1000 - stats["latency_ms"]
                logger.info("[multi/ydc] provider=%s searches=%d api_calls=%d latency=%.0fms connect=%.0fms overhead=%.0fms",
                    provider, stats["search_calls"], stats["api_calls"], stats["latency_ms"],
                    stats.get("connect_ms", 0), _slot_overhead)
                _costs = calculate_costs(stats, model_config)
                llm_cost, search_cost = _costs["llm"], _costs["search"]
                send_sse_safe("model_result", {
                    "slot": slot,
                    "provider": provider,
                    "display_name": model_config["display_name"],
                    "answer": stats["answer"],
                    "total_tokens": stats["tokens_used"],
                    "input_tokens": stats["token_breakdown"]["input"],
                    "output_tokens": stats["token_breakdown"]["output"],
                    "search_context_tokens": stats["token_breakdown"]["search_context"],
                    "api_calls": stats["api_calls"],
                    "search_calls": stats["search_calls"],
                    "sources": stats["sources"],
                    "latency_ms": round(stats["latency_ms"]),
                    "connect_ms": stats.get("connect_ms", 0),
                    "overhead_ms": round(max(0, _slot_overhead)),
                    "first_ms": _first_ms[0] or 0,
                    "cost": round(llm_cost + search_cost, 6),
                    "cost_llm": round(llm_cost, 6),
                    "cost_search": round(search_cost, 6),
                    "pricing": {
                        "input_cost_per_m": model_config["input_cost_per_m"],
                        "output_cost_per_m": model_config["output_cost_per_m"],
                        "ydc_search_per_1k": model_config["ydc_search_cost_per_call"] * 1000,
                    },
                })
            except Exception as e:
                logger.error("Model slot %s (%s) failed: %s", slot, provider, e, exc_info=True)
                send_sse_safe("model_error", {"slot": slot, "message": "Search failed. Please try again."})

        threads = [threading.Thread(target=run_slot, args=(i, p)) for i, p in enumerate(providers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        send_sse_safe("done", {"query": query})

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", self.ALLOWED_ORIGIN)
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, format, *args):
        logger.info("[http] %s %s", self.client_address[0], (format % args))


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    args = sys.argv[1:]
    port = 8080
    for i, arg in enumerate(args):
        if arg == "--port" and i + 1 < len(args):
            try:
                port = int(args[i + 1])
            except ValueError:
                print(f"Error: --port requires an integer, got: {args[i + 1]!r}")
                sys.exit(1)

    class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True

    AppHandler.ALLOWED_ORIGIN = f"http://localhost:{port}"
    server = ThreadedHTTPServer(("", port), AppHandler)
    print(f"You.com Web Search API for LLMs — Live Demo")
    print(f"Server running at http://localhost:{port}")
    print(f"Press Ctrl+C to stop\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()


if __name__ == "__main__":
    main()
