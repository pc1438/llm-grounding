"""
server.py — Web server for the comparison frontend.

Wraps the same comparison functions used by compare.py (CLI) and exposes
them as HTTP endpoints. The HTML frontend (index.html) calls these.

The /api/compare endpoint uses Server-Sent Events (SSE) to stream results
as each step completes — the frontend paints each pane the moment its
data arrives rather than waiting for everything.

Run:
    python server.py                  # starts on http://localhost:8080
    python server.py --port 9000      # custom port

Then open http://localhost:8080 in your browser.
"""

import json
import os
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn
from pathlib import Path
from dotenv import load_dotenv

# Load env
load_dotenv("env.txt") or load_dotenv(".env")
load_dotenv(Path(__file__).parent.parent / "grounding" / "env.txt")
load_dotenv(Path(__file__).parent.parent / "grounding" / ".env")

# Add grounding/ to path
sys.path.insert(0, str(Path(__file__).parent.parent / "grounding"))

from compare import MODELS, run_youdotcom, run_native, run_judge, _create_client


class ComparisonHandler(SimpleHTTPRequestHandler):
    """Serves index.html and handles /api/compare requests via SSE."""

    # Use HTTP/1.0 so each wfile.flush() actually pushes bytes to the client.
    # HTTP/1.1 + no Content-Length triggers chunked encoding which Python's
    # http.server doesn't implement, causing data to buffer until close.
    protocol_version = "HTTP/1.0"

    def setup(self):
        """Override to make wfile unbuffered for SSE streaming."""
        super().setup()
        # Replace the buffered wfile with an unbuffered socket file
        self.wfile = self.request.makefile("wb", buffering=0)

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.path = "/index.html"
            return SimpleHTTPRequestHandler.do_GET(self)
        return SimpleHTTPRequestHandler.do_GET(self)

    def do_POST(self):
        if self.path == "/api/compare":
            self._handle_compare()
        else:
            self.send_error(404)

    def _send_json(self, status: int, data: dict):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "http://localhost:8080")
        self.end_headers()
        self.wfile.write(body)

    def _send_sse(self, event: str, data: dict):
        """Send one Server-Sent Event."""
        try:
            payload = f"event: {event}\ndata: {json.dumps(data)}\n\n"
            self.wfile.write(payload.encode("utf-8"))
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass  # client disconnected

    MAX_BODY_SIZE = 10_000
    MAX_QUERY_LENGTH = 2_000

    def _handle_compare(self):
        # Read request body
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length > self.MAX_BODY_SIZE:
            self._send_json(400, {"error": "Request body too large"})
            return
        body = self.rfile.read(content_length).decode("utf-8")

        try:
            params = json.loads(body)
        except json.JSONDecodeError:
            self._send_json(400, {"error": "Invalid JSON"})
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

        # Validate keys
        ydc_key = os.environ.get("YDC_API_KEY", "")
        if not ydc_key:
            self._send_json(500, {"error": "YDC_API_KEY not configured"})
            return

        try:
            client = _create_client(model_config)
        except ValueError as e:
            self._send_json(500, {"error": str(e)})
            return

        # Start SSE stream
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")  # disable proxy buffering
        self.send_header("Access-Control-Allow-Origin", "http://localhost:8080")
        self.end_headers()
        self.wfile.flush()

        # ── Step 1: You.com path ──
        self._send_sse("status", {"step": "youdotcom", "message": "Running You.com search path..."})

        try:
            ydc_stats = run_youdotcom(query, client, model_config,
                on_progress=lambda msg: self._send_sse("progress", {"path": "youdotcom", "message": msg}))
        except Exception as e:
            self._send_sse("error", {"message": f"You.com path failed: {str(e)}"})
            return

        ydc_llm = (ydc_stats["input_tokens"] * model_config["input_cost_per_m"] / 1_000_000 +
                   ydc_stats["output_tokens"] * model_config["output_cost_per_m"] / 1_000_000)
        ydc_search = ydc_stats["search_calls"] * model_config["ydc_search_cost_per_call"]
        ydc_cost = ydc_llm + ydc_search

        self._send_sse("youdotcom", {
            "answer": ydc_stats["answer"],
            "total_tokens": ydc_stats["total_tokens"],
            "input_tokens": ydc_stats["input_tokens"],
            "output_tokens": ydc_stats["output_tokens"],
            "search_context_tokens": ydc_stats["search_context_tokens"],
            "api_calls": ydc_stats["api_calls"],
            "search_calls": ydc_stats["search_calls"],
            "sources": ydc_stats["sources"],
            "hit_round_limit": ydc_stats.get("hit_round_limit", False),
            "search_uuid": ydc_stats.get("search_uuid", ""),
            "latency_ms": round(ydc_stats["latency_ms"]),
            "cost": round(ydc_cost, 6),
            "cost_llm": round(ydc_llm, 6),
            "cost_search": round(ydc_search, 6),
        })

        # ── Step 2: Native path ──
        self._send_sse("status", {"step": "native", "message": "Running Native search path..."})

        try:
            native_stats = run_native(query, client, model_config,
                on_progress=lambda msg: self._send_sse("progress", {"path": "native", "message": msg}))
        except Exception as e:
            self._send_sse("error", {"message": f"Native path failed: {str(e)}"})
            return

        native_llm = (native_stats["input_tokens"] * model_config["input_cost_per_m"] / 1_000_000 +
                      native_stats["output_tokens"] * model_config["output_cost_per_m"] / 1_000_000)
        native_search = native_stats["search_calls"] * model_config["native_search_cost_per_call"]
        native_cost = native_llm + native_search

        self._send_sse("native", {
            "answer": native_stats["answer"],
            "total_tokens": native_stats["total_tokens"],
            "input_tokens": native_stats["input_tokens"],
            "output_tokens": native_stats["output_tokens"],
            "search_context_tokens": native_stats["search_context_tokens"],
            "api_calls": native_stats["api_calls"],
            "search_calls": native_stats["search_calls"],
            "hit_round_limit": native_stats.get("hit_round_limit", False),
            "sources": native_stats["sources"],
            "latency_ms": round(native_stats["latency_ms"]),
            "cost": round(native_cost, 6),
            "cost_llm": round(native_llm, 6),
            "cost_search": round(native_search, 6),
        })

        # ── Step 3: Judge ──
        judge_result = None
        if not skip_judge:
            self._send_sse("status", {"step": "judge", "message": "Running judge evaluation..."})
            try:
                judge_result = run_judge(
                    query, ydc_stats["answer"], native_stats["answer"],
                    model_config["judge"],
                    sources_ydc=ydc_stats["sources"],
                    sources_native=native_stats["sources"],
                )
            except Exception as e:
                judge_result = {"error": f"Judge failed: {str(e)}"}

            self._send_sse("judge", judge_result)

        # ── Done ──
        native_note = ""
        if model_config["provider"] == "anthropic":
            native_note = " + search results billed as input tokens"
        elif model_config["provider"] == "openai":
            native_note = " + search content tokens billed at model rates"

        self._send_sse("done", {
            "query": query,
            "model": model_config["model"],
            "provider": provider,
            "pricing": {
                "input_cost_per_m": model_config["input_cost_per_m"],
                "output_cost_per_m": model_config["output_cost_per_m"],
                "ydc_search_per_1k": model_config["ydc_search_cost_per_call"] * 1000,
                "native_search_per_1k": model_config["native_search_cost_per_call"] * 1000,
                "native_search_note": native_note,
            },
        })

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "http://localhost:8080")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, format, *args):
        """Cleaner logging."""
        sys.stderr.write(f"[server] {args[0]}\n")


def main():
    args = sys.argv[1:]
    port = 8080
    for i, arg in enumerate(args):
        if arg == "--port" and i + 1 < len(args):
            port = int(args[i + 1])

    class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True

    server = ThreadedHTTPServer(("", port), ComparisonHandler)
    print(f"Comparison server running at http://localhost:{port}")
    print(f"Open http://localhost:{port} in your browser")
    print(f"Press Ctrl+C to stop\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()


if __name__ == "__main__":
    main()
