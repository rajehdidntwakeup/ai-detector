import http.client
import json
import sys
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from detection_service import ModelManager, _qwen_result
from server import Handler, ThreadingHTTPServer

TEXT = "Some valid text to analyze."
LONG_TEXT = " ".join(["word"] * 12)


class FakeQwenAdapter:
    engine_id = "fake/qwen-detector"
    revision = "fake-qwen-revision"
    text_token_limit = 10

    def __init__(self, max_tokens=10):
        self.max_tokens = max_tokens
        self.scored = []

    def count_text_tokens(self, text):
        return len(text.split())

    def score_text(self, text):
        self.scored.append(text)
        return 0.9


class FakeBaseline:
    def __init__(self, block=None):
        self.block = block
        self.calls = 0

    def analyze(self, text):
        self.calls += 1
        if self.block:
            self.block.wait(5)
        return {"originalContent": text, "engine": {"id": "fake/baseline"}}


def fake_registry(baseline_factory=None, qwen_factory=None):
    return {
        "baseline": {
            "label": "DistilBERT Detector", "experimental": False, "passageScores": True,
            "contextTokens": 512, "textTokenLimit": None,
            "engineId": "fake/baseline", "revision": "fake-baseline-revision",
            "capabilities": {"contextTokens": 512, "passageScores": True, "experimental": False},
            "factory": baseline_factory or (lambda: FakeBaseline()),
            "analyze": lambda engine, text, started: engine.analyze(text),
        },
        "qwen3-variable": {
            "label": "Qwen3-0.6B Detector (experimental)", "experimental": True,
            "passageScores": False, "contextTokens": 11, "textTokenLimit": 10,
            "engineId": "fake/qwen-detector", "revision": "fake-qwen-revision",
            "capabilities": {"contextTokens": 11, "textTokenLimit": 10, "passageScores": False,
                             "experimental": True},
            "factory": qwen_factory or (lambda: FakeQwenAdapter()),
            "analyze": _qwen_result,
        },
    }


def wait_for(manager, predicate, timeout=5.0):
    step = threading.Event()
    while timeout > 0:
        if predicate(manager.status()):
            return True
        step.wait(0.01)
        timeout -= 0.01
    return predicate(manager.status())


class ServerAPITests(unittest.TestCase):
    def setUp(self):
        self.manager = ModelManager(registry=fake_registry(), auto_load=None)
        self.manager.activate("baseline")
        self.assertTrue(wait_for(self.manager, lambda s: s["status"] == "ready"))
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.manager = self.manager
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.port = self.server.server_port

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def request(self, method, path, payload=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        request_headers = {"Content-Type": "application/json"}
        request_headers.update(headers or {})
        body = json.dumps(payload) if payload is not None else None
        connection.request(method, path, body=body, headers=request_headers)
        response = connection.getresponse()
        result = (response.status, response.read(), dict(response.getheaders()))
        connection.close()
        return result

    def switch_to_qwen(self):
        self.request("POST", "/api/model", {"model": "qwen3-variable"})
        self.assertTrue(wait_for(self.manager, lambda s: s["activeModel"] == "qwen3-variable"))

    def test_only_allowlisted_assets_are_served(self):
        status, body, headers = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn(b"Local Lens", body)
        self.assertIn("connect-src 'self'", headers["Content-Security-Policy"])
        self.assertEqual(headers["Cache-Control"], "no-store")
        for path in ("/../server.py", "/models/distilbert-detector/config.json", "/.git/config"):
            self.assertEqual(self.request("GET", path)[0], 404)

    def test_status_exposes_active_engine_and_model_choices(self):
        status, body, _ = self.request("GET", "/api/status")
        self.assertEqual(status, 200)
        payload = json.loads(body)
        self.assertEqual(payload["app"], "local-lens")
        self.assertEqual(payload["engine"]["id"], payload["model"])
        self.assertEqual(payload["engine"]["revision"], payload["revision"])
        self.assertEqual(payload["activeModel"], "baseline")
        self.assertEqual(payload["requestedModel"], "baseline")
        self.assertFalse(payload["busy"])
        self.assertEqual(payload["capabilities"], payload["engine"]["capabilities"])
        names = {model["name"] for model in payload["models"]}
        self.assertEqual(names, {"baseline", "qwen3-variable"})
        qwen = next(m for m in payload["models"] if m["name"] == "qwen3-variable")
        self.assertTrue(qwen["experimental"])
        self.assertFalse(qwen["passageScores"])
        self.assertEqual(qwen["textTokenLimit"], 10)

    def test_status_tracks_requested_model_while_loading(self):
        started, release = threading.Event(), threading.Event()

        def slow_qwen():
            started.set()
            release.wait(5)
            return FakeQwenAdapter()

        self.manager._registry["qwen3-variable"]["factory"] = slow_qwen
        self.assertEqual(self.request("POST", "/api/model", {"model": "qwen3-variable"})[0], 202)
        self.assertTrue(started.wait(2))
        payload = json.loads(self.request("GET", "/api/status")[1])
        self.assertEqual(payload["status"], "loading")
        self.assertEqual(payload["requestedModel"], "qwen3-variable")
        self.assertIsNone(payload["activeModel"])
        release.set()
        self.assertTrue(wait_for(self.manager, lambda s: s["status"] == "ready"))

    def test_model_activation_endpoint_contract(self):
        self.assertEqual(self.request("POST", "/api/model", {"model": "gpt-4"})[0], 400)
        self.assertEqual(self.request("POST", "/api/model", {"model": ""})[0], 400)
        self.assertEqual(self.request("POST", "/api/model", {"model": 5})[0], 400)
        self.assertEqual(self.request("POST", "/api/model", {"note": "no model field"})[0], 400)
        self.assertEqual(self.request("POST", "/api/model", {"model": "qwen3-variable"},
                                      {"Content-Type": "text/plain"})[0], 415)
        status, body, _ = self.request("POST", "/api/model", {"model": "baseline"})
        self.assertEqual(status, 200)  # already ready
        self.assertTrue(json.loads(body)["success"])

    def test_external_origins_and_dns_rebinding_hosts_are_rejected(self):
        for headers in ({"Origin": "https://example.org"}, {"Host": f"example.org:{self.port}"},
                        {"Sec-Fetch-Site": "cross-site"}):
            with self.subTest(headers=headers):
                self.assertEqual(self.request("POST", "/bff/detect", {"text": TEXT}, headers)[0], 403)
        self.assertEqual(self.request("POST", "/api/stop")[0], 403)
        self.assertEqual(self.request("POST", "/api/model", {"model": "qwen3-variable"},
                                      {"Origin": "https://example.org"})[0], 403)

    def test_validation_and_request_contract(self):
        for value in (None, 5, "short", "word " * 2001):
            with self.subTest(value=str(value)[:30]):
                self.assertEqual(self.request("POST", "/bff/detect", {"text": value})[0], 400)
        self.assertEqual(self.request("POST", "/bff/detect", {"model": 5, "text": TEXT})[0], 400)
        self.assertEqual(self.request("POST", "/bff/detect", {"model": "desklib", "text": TEXT})[0], 400)
        for path in ("/bff/detect", "/api/analyze"):
            status, body, _ = self.request("POST", path, {"text": "  Some valid text to analyze.  "})
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(body)["data"]["originalContent"], "Some valid text to analyze.")
        self.assertEqual(self.request("POST", "/bff/detect", {"text": "hello world"},
                                      {"Content-Type": "text/plain"})[0], 415)
        try:
            status, _, _ = self.request("POST", "/bff/detect", {"text": "a" * 300001})
        except ConnectionAbortedError:
            status = 413
        self.assertEqual(status, 413)

    def test_unavailable_model_returns_503(self):
        release = threading.Event()

        def slow_baseline():
            release.wait(5)
            return FakeBaseline()

        self.server.manager = ModelManager(
            registry=fake_registry(baseline_factory=slow_baseline), auto_load="baseline")
        try:
            status, body, _ = self.request("POST", "/bff/detect", {"text": TEXT})
            self.assertEqual(status, 503)
            self.assertIn("loading", json.loads(body)["message"].lower())
        finally:
            release.set()

    def test_busy_lock_returns_429_and_blocks_switching(self):
        block = threading.Event()
        self.manager._engine = FakeBaseline(block=block)  # replace the loaded engine
        first = threading.Thread(target=lambda: self.manager.analyze(None, TEXT), daemon=True)
        first.start()
        self.assertTrue(wait_for(self.manager, lambda s: s["busy"]))
        self.assertEqual(self.request("POST", "/bff/detect", {"text": TEXT})[0], 429)
        self.assertEqual(self.request("POST", "/api/model", {"model": "qwen3-variable"})[0], 429)
        block.set()
        first.join(2)
        self.assertTrue(wait_for(self.manager, lambda s: not s["busy"]))

    def test_legacy_request_after_switch_gets_explicit_mismatch(self):
        self.switch_to_qwen()
        status, body, _ = self.request("POST", "/bff/detect", {"text": TEXT})
        self.assertEqual(status, 409)
        payload = json.loads(body)
        self.assertEqual(payload["activeModel"], "qwen3-variable")
        self.assertEqual(payload["requestedModel"], "baseline")
        status, body, _ = self.request("POST", "/bff/detect", {"model": "qwen3-variable", "text": TEXT})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["data"]["passageScores"], False)
        # Returning to baseline is an explicit activation, never an implicit replay.
        self.request("POST", "/api/model", {"model": "baseline"})
        self.assertTrue(wait_for(self.manager, lambda s: s["activeModel"] == "baseline"))
        self.assertEqual(self.request("POST", "/api/analyze", {"text": TEXT})[0], 200)

    def test_oversized_qwen_input_returns_422_with_token_counts(self):
        self.switch_to_qwen()
        status, body, _ = self.request("POST", "/bff/detect",
                                       {"model": "qwen3-variable", "text": LONG_TEXT})
        self.assertEqual(status, 422)
        payload = json.loads(body)
        self.assertEqual(payload["textTokens"], 12)
        self.assertEqual(payload["textTokenLimit"], 10)
        self.assertEqual(payload["totalTokens"], 13)
        self.assertEqual(payload["totalLimit"], 11)
        self.assertIn("shorten", payload["message"].lower())
        within = " ".join(["word"] * 10)
        self.assertEqual(self.request("POST", "/bff/detect",
                                      {"model": "qwen3-variable", "text": within})[0], 200)


if __name__ == "__main__":
    unittest.main()