"""Minimal stdlib HTTP server exposing classify() — the container entrypoint.

  POST /classify   {"text": "...", "kind": "chat|name|title"}  -> verdict JSON
  GET  /health     -> {"status": "ok"}

No web framework: stdlib http.server only, to keep the deployment image small.
"""
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from slurshield.infer import classify

_VALID_KINDS = {"name", "title", "chat"}


class _Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send(200, {"status": "ok"}) if self.path == "/health" \
            else self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/classify":
            self._send(404, {"error": "not found"})
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(n) or b"{}")
        except (ValueError, json.JSONDecodeError):
            self._send(400, {"error": "invalid JSON body"})
            return
        kind = data.get("kind")
        if kind not in _VALID_KINDS:
            self._send(400, {"error": f"kind must be one of {sorted(_VALID_KINDS)}"})
            return
        self._send(200, classify(data.get("text", ""), kind))

    def log_message(self, *args):  # silence default request logging
        pass


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    classify("warmup", "chat")  # load model/tokenizer so the first request isn't slow
    server = ThreadingHTTPServer((host, port), _Handler)
    print(f"slurshield serving on {host}:{port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main(port=int(os.environ.get("PORT", "8000")))
