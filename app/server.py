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
import os
import sys
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn
from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# ─── Path setup ────────────────────────────────────────────────────────────
# This file lives in use-cases/app/. We need to reach:
#   use-cases/grounding/  (for run.py, search_tool.py, agents/)
#   use-cases/comparison/ (for compare.py)

APP_DIR = Path(__file__).parent
USE_CASES_DIR = APP_DIR.parent
GROUNDING_DIR = USE_CASES_DIR / "grounding"
COMPARISON_DIR = USE_CASES_DIR / "comparison"

# Load env from grounding dir (that's where env.txt lives)
load_dotenv(GROUNDING_DIR / "env.txt")
load_dotenv(GROUNDING_DIR / ".env")
load_dotenv(COMPARISON_DIR / "env.txt")
load_dotenv(COMPARISON_DIR / ".env")

# Add both dirs to path so imports work
sys.path.insert(0, str(GROUNDING_DIR))
sys.path.insert(0, str(COMPARISON_DIR))

# Import from grounding/run.py
from run import GROUNDING_MODELS, run_grounding

# Import from comparison/compare.py
from compare import (
    MODELS,
    run_youdotcom, run_native, run_judge, _create_client,
    describe_native_search,
    calculate_costs, calculate_native_costs,
)

# Import from comparison/perplexity_runner.py
from perplexity_runner import PERPLEXITY_SAC_CONFIG, run_perplexity_sac

# Import from comparison/ydc_research_runner.py
from ydc_research_runner import YDC_RESEARCH_CONFIG, run_ydc_research


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
        elif self.path == "/api/sac-compare":
            self._handle_sac_compare()
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

        self._start_sse()

        # Stream events from the grounding generator
        for event in run_grounding(model_key, query):
            self._send_sse(event["event"], event)

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

        try:
            client = _create_client(model_config)
        except ValueError as e:
            logger.error("_create_client failed for provider %s: %s", provider, e)
            self._send_json(500, {"error": "Server configuration error. Check API keys."})
            return

        self._start_sse()

        # ── Steps 1 & 2: Run You.com and Native paths in parallel ──
        send_sse_safe = self._make_sse_sender()

        send_sse_safe("status", {"step": "both", "message": "Running You.com + Native paths in parallel..."})

        # Shared results — each thread writes to its slot
        ydc_result = {"stats": None, "error": None}
        native_result = {"stats": None, "error": None}

        def run_ydc_thread():
            try:
                ydc_result["stats"] = run_youdotcom(query, client, model_config,
                    on_progress=lambda msg: send_sse_safe("progress", {"path": "youdotcom", "message": msg}))
            except Exception as e:
                logger.error("You.com path failed: %s", e, exc_info=True)
                ydc_result["error"] = True

        def run_native_thread():
            try:
                native_result["stats"] = run_native(query, client, model_config,
                    on_progress=lambda msg: send_sse_safe("progress", {"path": "native", "message": msg}))
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
        _ydc = calculate_costs(ydc_stats, model_config)
        ydc_llm, ydc_search, ydc_cost = _ydc["llm"], _ydc["search"], _ydc["total"]

        send_sse_safe("youdotcom", {
            "answer": ydc_stats["answer"],
            "total_tokens": ydc_stats["total_tokens"],
            "input_tokens": ydc_stats["input_tokens"],
            "output_tokens": ydc_stats["output_tokens"],
            "search_context_tokens": ydc_stats["search_context_tokens"],
            "api_calls": ydc_stats["api_calls"],
            "search_calls": ydc_stats["search_calls"],
            "sources": ydc_stats["sources"],
            "search_uuid": ydc_stats.get("search_uuid", ""),
            "latency_ms": round(ydc_stats["latency_ms"]),
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

        _native = calculate_native_costs(native_stats, model_config)
        native_llm, native_search, native_cost = _native["llm"], _native["search"], _native["total"]

        send_sse_safe("native", {
            "answer": native_stats["answer"],
            "total_tokens": native_stats["total_tokens"],
            "input_tokens": native_stats["input_tokens"],
            "output_tokens": native_stats["output_tokens"],
            "search_context_tokens": native_stats["search_context_tokens"],
            "api_calls": native_stats["api_calls"],
            "search_calls": native_stats["search_calls"],
            "sources": native_stats["sources"],
            "latency_ms": round(native_stats["latency_ms"]),
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

        if not isinstance(providers, list):
            self._send_json(400, {"error": "'providers' must be a list"})
            return

        if not query:
            self._send_json(400, {"error": "Query is required"})
            return
        if len(query) > self.MAX_QUERY_LENGTH:
            self._send_json(400, {"error": f"Query too long (max {self.MAX_QUERY_LENGTH} chars)"})
            return
        if not providers or not (2 <= len(providers) <= 3):
            self._send_json(400, {"error": "Provide 2-3 model providers"})
            return
        for p in providers:
            if p not in MODELS:
                self._send_json(400, {"error": f"Unknown provider: {p}"})
                return

        ydc_key = os.environ.get("YDC_API_KEY", "")
        if not ydc_key:
            self._send_json(500, {"error": "YDC_API_KEY not configured"})
            return

        # Validate all clients upfront before starting SSE
        clients = {}
        for p in providers:
            try:
                clients[p] = _create_client(MODELS[p])
            except ValueError as e:
                logger.error("_create_client failed for provider %s: %s", p, e)
                self._send_json(500, {"error": "Server configuration error. Check API keys."})
                return

        self._start_sse()

        send_sse_safe = self._make_sse_sender()

        def run_slot(slot, provider):
            model_config = MODELS[provider]
            client = clients[provider]
            try:
                stats = run_youdotcom(
                    query, client, model_config,
                    on_progress=lambda msg: send_sse_safe("model_progress", {"slot": slot, "message": msg})
                )
                _costs = calculate_costs(stats, model_config)
                llm_cost, search_cost = _costs["llm"], _costs["search"]
                send_sse_safe("model_result", {
                    "slot": slot,
                    "provider": provider,
                    "display_name": model_config["display_name"],
                    "answer": stats["answer"],
                    "total_tokens": stats["total_tokens"],
                    "input_tokens": stats["input_tokens"],
                    "output_tokens": stats["output_tokens"],
                    "search_context_tokens": stats["search_context_tokens"],
                    "api_calls": stats["api_calls"],
                    "search_calls": stats["search_calls"],
                    "sources": stats["sources"],
                    "latency_ms": round(stats["latency_ms"]),
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

    # ── /api/sac-compare ───────────────────────────────────────────────────

    def _handle_sac_compare(self):
        """Stream You.com Research API vs. Perplexity SaC comparison events as SSE.

        Both sides are single-call deep research APIs:
          - You.com Research API: one POST, multi-step search + synthesis internally
          - Perplexity SaC: Claude inside Perplexity's Agent API with SaC SDK primitives

        GPT acts as a cross-model judge to evaluate both answers.
        """
        params = self._read_json_body()
        if params is None:
            return

        query = params.get("query", "").strip()
        skip_judge = params.get("skip_judge", False)

        if not query:
            self._send_json(400, {"error": "Query is required"})
            return
        if len(query) > self.MAX_QUERY_LENGTH:
            self._send_json(400, {"error": f"Query too long (max {self.MAX_QUERY_LENGTH} chars)"})
            return

        ydc_key = os.environ.get("YDC_API_KEY", "")
        if not ydc_key:
            self._send_json(500, {"error": "YDC_API_KEY not configured"})
            return

        plx_key = os.environ.get("PERPLEXITY_API_KEY", "")
        if not plx_key:
            self._send_json(500, {"error": "PERPLEXITY_API_KEY not configured"})
            return

        self._start_sse()

        send_sse_safe = self._make_sse_sender()

        send_sse_safe("status", {"step": "youdotcom", "message": "Running You.com Research API..."})
        send_sse_safe("status", {"step": "perplexity", "message": "Running Claude + Perplexity SaC..."})

        ydc_result = {"stats": None, "error": None}
        plx_result = {"stats": None, "error": None}

        def run_ydc_thread():
            try:
                ydc_result["stats"] = run_ydc_research(
                    query, ydc_key,
                    research_effort=YDC_RESEARCH_CONFIG["effort"],
                    on_progress=lambda msg: send_sse_safe("progress", {"path": "youdotcom", "message": msg}),
                )
            except Exception as e:
                logger.error("You.com Research path failed: %s", e, exc_info=True)
                ydc_result["error"] = True

        def run_plx_thread():
            try:
                plx_result["stats"] = run_perplexity_sac(
                    query, plx_key,
                    on_progress=lambda msg: send_sse_safe("progress", {"path": "perplexity", "message": msg}),
                )
            except Exception as e:
                logger.error("Perplexity SaC path failed: %s", e, exc_info=True)
                plx_result["error"] = True

        t_ydc = threading.Thread(target=run_ydc_thread)
        t_plx = threading.Thread(target=run_plx_thread)
        t_ydc.start()
        t_plx.start()
        t_ydc.join()
        t_plx.join()

        # ── Emit You.com Research result ──
        if ydc_result["error"]:
            send_sse_safe("error", {"message": "You.com Research failed. Please try again."})
            return

        ydc_stats = ydc_result["stats"]
        if ydc_stats is None:
            send_sse_safe("error", {"message": "You.com Research returned no data. Please try again."})
            return
        # Research API bundles all costs; token counts are not exposed
        ydc_cost = YDC_RESEARCH_CONFIG["cost_per_call"]

        send_sse_safe("youdotcom", {
            "answer": ydc_stats["answer"],
            "total_tokens": ydc_stats["total_tokens"],
            "input_tokens": ydc_stats["input_tokens"],
            "output_tokens": ydc_stats["output_tokens"],
            "search_context_tokens": 0,
            "tokens_estimated": True,
            "api_calls": ydc_stats["api_calls"],
            "search_calls": ydc_stats["search_calls"],
            "sources": ydc_stats["sources"],
            "latency_ms": round(ydc_stats["latency_ms"]),
            "cost": round(ydc_cost, 6),
            "cost_llm": 0.0,
            "cost_search": round(ydc_cost, 6),
            "cost_note": "est.",
            "research_effort": ydc_stats.get("research_effort", YDC_RESEARCH_CONFIG["effort"]),
        })

        # ── Emit Perplexity result ──
        if plx_result["error"]:
            send_sse_safe("error", {"message": "Perplexity search failed. Please try again."})
            return

        plx_stats = plx_result["stats"]
        # Use actual cost from Perplexity's usage object if available; otherwise estimate
        if plx_stats.get("actual_cost") is not None:
            plx_cost = plx_stats["actual_cost"]
            plx_cost_note = "actual"
        else:
            plx_cost = plx_stats["api_calls"] * PERPLEXITY_SAC_CONFIG["sac_cost_per_call"]
            plx_cost_note = "est."

        send_sse_safe("perplexity", {
            "answer": plx_stats["answer"],
            "total_tokens": plx_stats["total_tokens"],
            "input_tokens": plx_stats["input_tokens"],
            "output_tokens": plx_stats["output_tokens"],
            "search_context_tokens": plx_stats.get("search_context_tokens", 0),
            "api_calls": plx_stats["api_calls"],
            "search_calls": plx_stats["search_calls"],
            "sandbox_executions": plx_stats.get("sandbox_executions", 0),
            "sources": plx_stats["sources"],
            "latency_ms": round(plx_stats["latency_ms"]),
            "cost": round(plx_cost, 6),
            "cost_llm": 0.0,      # Claude inference bundled into Perplexity's billing
            "cost_search": round(plx_cost, 6),
            "cost_note": plx_cost_note,
        })

        # ── Judge (GPT judges since neither side exposes a local LLM cleanly) ──
        judge_result = None
        if not skip_judge:
            send_sse_safe("status", {"step": "judge", "message": "Running judge evaluation (GPT-5.4)..."})
            try:
                judge_result = run_judge(
                    query,
                    ydc_stats["answer"],
                    plx_stats["answer"],
                    "openai",
                    sources_ydc=ydc_stats["sources"],
                    sources_native=plx_stats["sources"],
                )
            except Exception as e:
                logger.error("Judge failed (sac-compare): %s", e, exc_info=True)
                judge_result = {"error": "Judge evaluation failed. Please try again."}

            send_sse_safe("judge", judge_result)

        # ── Done ──
        send_sse_safe("done", {
            "query": query,
            "pricing": {
                "ydc_research_effort": YDC_RESEARCH_CONFIG["effort"],
                "ydc_research_per_call": YDC_RESEARCH_CONFIG["cost_per_call"],
                "ydc_research_note": "est. — You.com Research API bundles search + synthesis; verify at you.com/platform/pricing",
                "plx_sac_per_call": PERPLEXITY_SAC_CONFIG["sac_cost_per_call"],
                "plx_sac_note": "est. — Perplexity Agent API (Claude + SaC SDK); includes Claude inference; verify at docs.perplexity.ai",
            },
        })

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", self.ALLOWED_ORIGIN)
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, format, *args):
        """Cleaner logging."""
        sys.stderr.write(f"[server] {args[0]}\n")


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
