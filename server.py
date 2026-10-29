"""Loopback-only HTTP server for the offline detector. Python 3.10+."""
import argparse
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from detection_service import (DocumentTooLongError, ModelManager, ModelMismatchError,
                               ModelNotReadyError, OperationConflictError, UnknownModelError)
from detector import validate_text
from model_info import MAX_WORDS

STATIC = Path(__file__).resolve().parent / "dist"
ASSETS = {"/": ("index.html", "text/html; charset=utf-8"),
          "/app.js": ("app.js", "text/javascript; charset=utf-8"),
          "/style.css": ("style.css", "text/css; charset=utf-8"),
          "/favicon.svg": ("favicon.svg", "image/svg+xml")}


def status_payload(manager):
    """Backward-compatible status body plus model-selection fields."""
    state = manager.status()
    models = {model["name"]: model for model in state["models"]}
    current = state["activeModel"] or state["requestedModel"] or "baseline"
    info = models.get(current) or models["baseline"]
    return {"app": "local-lens", "status": state["status"], "error": state["error"],
            "model": info["engineId"], "revision": info["revision"], "offline": True,
            "engine": {"id": info["engineId"], "revision": info["revision"], "offline": True,
                       "capabilities": info["capabilities"]},
            "maxWords": MAX_WORDS, "busy": state["busy"],
            "capabilities": info["capabilities"],
            "activeModel": state["activeModel"], "requestedModel": state["requestedModel"],
            "models": state["models"]}


class Handler(BaseHTTPRequestHandler):
    server_version = "LocalLens/1.0"

    def log_message(self, *_):
        pass  # Never log user text, request URLs, or results.

    def send(self, code, body, content_type="application/json; charset=utf-8"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=True).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; connect-src 'self'; script-src 'self'; style-src 'self'; img-src 'self'; font-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def allowed(self):
        port = self.server.server_port
        hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        host = self.headers.get("Host", "").lower()
        origin = self.headers.get("Origin")
        if host not in hosts or (origin and origin not in {f"http://{item}" for item in hosts}):
            self.send(403, {"success": False, "message": "Only this laptop's local app can access the detector."})
            return False
        if self.headers.get("Sec-Fetch-Site") == "cross-site":
            self.send(403, {"success": False, "message": "Cross-site requests are not accepted."})
            return False
        return True

    def do_GET(self):
        if not self.allowed():
            return
        path = urlsplit(self.path).path
        if path == "/api/status":
            self.send(200, status_payload(self.server.manager))
        elif path in ASSETS:
            filename, content_type = ASSETS[path]
            self.send(200, (STATIC / filename).read_bytes(), content_type)
        else:
            self.send(404, {"success": False, "message": "Not found."})

    def read_json(self):
        """Read a bounded JSON object; respond and return None on bad requests."""
        if self.headers.get_content_type() != "application/json":
            self.send(415, {"success": False, "message": "Send JSON with a text field."})
            return None
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 300000:
                self.send(413, {"success": False, "message": "Request is empty or too large."})
                return None
            self.connection.settimeout(10)
            payload = json.loads(self.rfile.read(length))
        except (ValueError, UnicodeError, TimeoutError):
            self.send(400, {"success": False, "message": "Send a valid JSON object."})
            return None
        if not isinstance(payload, dict):
            self.send(400, {"success": False, "message": "Send a JSON object with a text field."})
            return None
        return payload

    def do_POST(self):
        if not self.allowed():
            return
        path = urlsplit(self.path).path
        if path == "/api/stop":
            if self.headers.get("X-Local-Lens-Action") != "stop":
                self.send(403, {"success": False, "message": "Use stop.cmd to stop the local app."})
                return
            self.send(200, {"success": True})
            threading.Thread(target=self.server.shutdown, daemon=True).start()
            return
        if path == "/api/model":
            self.handle_model_activation()
            return
        if path not in ("/api/analyze", "/bff/detect"):
            self.send(404, {"success": False, "message": "Not found."})
            return
        self.handle_analysis()

    def handle_model_activation(self):
        """Activate an allowlisted model. No document text is read or stored here."""
        payload = self.read_json()
        if payload is None:
            return
        model = payload.get("model")
        if not isinstance(model, str) or not model:
            self.send(400, {"success": False, "message": "Send a JSON body with a model name."})
            return
        try:
            result = self.server.manager.activate(model)
        except UnknownModelError:
            self.send(400, {"success": False, "message": "Unknown model. Use /api/status to list the available models."})
        except OperationConflictError as exc:
            self.send(429, {"success": False, "message": str(exc)})
        else:
            self.send(202 if result == "loading" else 200,
                      {"success": True, "model": model, "status": result})

    def handle_analysis(self):
        payload = self.read_json()
        if payload is None:
            return
        model = payload.get("model")
        if model is not None and not isinstance(model, str):
            self.send(400, {"success": False, "message": "The model field must be a model name string."})
            return
        try:
            text = validate_text(payload.get("text"))
        except (ValueError, UnicodeError):
            self.send(400, {"success": False, "message": "Enter 10 or more characters of writing, up to 2,000 words and 50,000 characters, in a valid JSON text field."})
            return
        manager = self.server.manager
        try:
            data = manager.analyze(model, text)
        except UnknownModelError:
            self.send(400, {"success": False, "message": "Unknown model. Use /api/status to list the available models."})
        except ModelNotReadyError as exc:
            self.send(503, {"success": False, "message": str(exc)})
        except OperationConflictError as exc:
            self.send(429, {"success": False, "message": str(exc)})
        except ModelMismatchError as exc:
            self.send(409, {"success": False, "message": str(exc),
                            "activeModel": exc.active, "requestedModel": exc.requested})
        except DocumentTooLongError as exc:
            self.send(422, {"success": False,
                            "message": f"{exc} Shorten the text, or activate the DistilBERT "
                                       "model, which scores longer documents in windows.",
                            "textTokens": exc.text_tokens, "textTokenLimit": exc.text_token_limit,
                            "totalTokens": exc.total_tokens, "totalLimit": exc.total_limit})
        except Exception:
            self.send(500, {"success": False, "message": "Local analysis failed. Try a shorter passage or restart the app."})
        else:
            self.send(200, {"success": True, "data": data})


def main():
    parser = argparse.ArgumentParser(description="Run Local Lens entirely on your laptop.")
    parser.add_argument("--port", type=int, default=3000)
    parser.add_argument("--open", action="store_true", help="Open the app in your default browser")
    args = parser.parse_args()
    manager = ModelManager()  # Loads the baseline DistilBERT model in the background.
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    except OSError as exc:
        raise SystemExit(f"Could not start on port {args.port}. Close the existing app or use --port 3001. ({exc})")
    server.manager = manager
    print(f"Local Lens: http://localhost:{args.port}\nYour text stays on this laptop. Press Ctrl+C to stop.", flush=True)
    if args.open:
        threading.Timer(0.5, lambda: webbrowser.open(f"http://localhost:{args.port}")).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

