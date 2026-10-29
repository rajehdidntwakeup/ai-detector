"""Model-manager lifecycle and result-contract tests using fake engines (no torch)."""
import sys
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from detection_service import (DocumentTooLongError, ModelManager, ModelMismatchError,
                               ModelNotReadyError, OperationConflictError, UnknownModelError,
                               _qwen_result, default_registry)
from engines.inputs import ContextLimitError

TEXT = "Some valid text to analyze for the local detector suite."


class FakeQwenAdapter:
    engine_id = "fake/qwen-detector"
    revision = "fake-revision"
    text_token_limit = 1023

    def __init__(self, score=0.9, max_tokens=1023, limit_error=False):
        self.score = score
        self.max_tokens = max_tokens
        self.limit_error = limit_error
        self.scored = []

    def count_text_tokens(self, text):
        return len(text.split())

    def score_text(self, text):
        if self.limit_error:
            # Mirrors the real adapter: tokens/limit include the EOS readout token.
            raise ContextLimitError(7, 7)
        self.scored.append(text)
        return self.score


class FakeBaseline:
    def __init__(self, block=None):
        self.block = block
        self.calls = 0

    def analyze(self, text):
        self.calls += 1
        if self.block:
            self.block.wait()
        return {"originalContent": text, "engine": {"id": "fake/baseline"}}


def fake_registry(baseline_factory=None, qwen_factory=None):
    baseline = baseline_factory or (lambda: FakeBaseline())
    return {
        "baseline": {
            "label": "DistilBERT Detector", "experimental": False, "passageScores": True,
            "contextTokens": 512, "textTokenLimit": None,
            "factory": baseline,
            "analyze": lambda engine, text, started: engine.analyze(text),
        },
        "qwen3-variable": {
            "label": "Qwen3-0.6B Detector (experimental)", "experimental": True,
            "passageScores": False, "contextTokens": 1024, "textTokenLimit": 1023,
            "factory": qwen_factory or (lambda: FakeQwenAdapter()),
            "analyze": _qwen_result,
        },
    }


def wait_for(manager, predicate, timeout=5.0):
    event = threading.Event()

    def poll():
        while not predicate(manager.status()):
            if event.wait(0.01):
                return

    watcher = threading.Thread(target=poll, daemon=True)
    watcher.start()
    watcher.join(timeout)
    event.set()
    return predicate(manager.status())


class RegistryTests(unittest.TestCase):
    def test_default_registry_only_contains_allowlisted_keys(self):
        registry = default_registry()
        self.assertEqual(set(registry), {"baseline", "qwen3-variable"})
        self.assertEqual(registry["baseline"]["contextTokens"], 512)
        self.assertEqual(registry["qwen3-variable"]["contextTokens"], 1024)
        self.assertEqual(registry["qwen3-variable"]["textTokenLimit"], 1023)
        self.assertTrue(registry["qwen3-variable"]["experimental"])


class ManagerLifecycleTests(unittest.TestCase):
    def test_default_baseline_loads_and_scores(self):
        manager = ModelManager(registry=fake_registry())
        self.assertTrue(wait_for(manager, lambda s: s["status"] == "ready"))
        status = manager.status()
        self.assertEqual(status["activeModel"], "baseline")
        self.assertEqual(status["requestedModel"], "baseline")
        self.assertFalse(status["busy"])
        result = manager.analyze(None, TEXT)
        self.assertEqual(result["originalContent"], TEXT)

    def test_unknown_model_names_rejected(self):
        manager = ModelManager(registry=fake_registry(), auto_load=None)
        with self.assertRaises(UnknownModelError):
            manager.activate("../models/other")
        with self.assertRaises(UnknownModelError):
            manager.analyze("desklib", TEXT)

    def test_activation_is_idempotent_when_ready_or_loading(self):
        started, release = threading.Event(), threading.Event()

        def slow_qwen():
            started.set()
            release.wait(5)
            return FakeQwenAdapter()

        manager = ModelManager(registry=fake_registry(qwen_factory=slow_qwen), auto_load=None)
        self.assertEqual(manager.activate("qwen3-variable"), "loading")
        self.assertTrue(started.wait(2))
        self.assertEqual(manager.activate("qwen3-variable"), "loading")  # idempotent while loading
        release.set()
        self.assertTrue(wait_for(manager, lambda s: s["status"] == "ready"))
        self.assertEqual(manager.activate("qwen3-variable"), "ready")  # idempotent when ready

    def test_competing_switch_rejected_while_loading(self):
        started, release = threading.Event(), threading.Event()

        def slow_qwen():
            started.set()
            release.wait(5)
            return FakeQwenAdapter()

        manager = ModelManager(registry=fake_registry(qwen_factory=slow_qwen), auto_load=None)
        manager.activate("qwen3-variable")
        self.assertTrue(started.wait(2))
        with self.assertRaises(OperationConflictError):
            manager.activate("baseline")
        with self.assertRaises(ModelNotReadyError):
            manager.analyze("qwen3-variable", TEXT)
        release.set()
        self.assertTrue(wait_for(manager, lambda s: s["status"] == "ready"))

    def test_switch_to_qwen_and_back_with_mismatch_protection(self):
        manager = ModelManager(registry=fake_registry())
        self.assertTrue(wait_for(manager, lambda s: s["status"] == "ready"))
        manager.activate("qwen3-variable")
        self.assertTrue(wait_for(manager, lambda s: s["activeModel"] == "qwen3-variable"))
        with self.assertRaises(ModelMismatchError):
            manager.analyze("baseline", TEXT)  # legacy request never gets Qwen output
        result = manager.analyze("qwen3-variable", TEXT)
        self.assertEqual(result["engine"]["id"], "fake/qwen-detector")
        manager.activate("baseline")
        self.assertTrue(wait_for(manager, lambda s: s["activeModel"] == "baseline"))
        baseline_result = manager.analyze(None, TEXT)
        self.assertEqual(baseline_result["engine"]["id"], "fake/baseline")

    def test_failed_load_is_recoverable_by_retry(self):
        attempts = []

        def flaky_qwen():
            attempts.append(1)
            if len(attempts) == 1:
                raise RuntimeError("missing weights")
            return FakeQwenAdapter()

        manager = ModelManager(registry=fake_registry(qwen_factory=flaky_qwen), auto_load=None)
        manager.activate("qwen3-variable")
        self.assertTrue(wait_for(manager, lambda s: s["status"] == "error"))
        status = manager.status()
        self.assertEqual(status["requestedModel"], "qwen3-variable")
        self.assertIn("missing weights", status["error"])
        with self.assertRaises(ModelNotReadyError):
            manager.analyze("qwen3-variable", TEXT)
        manager.activate("qwen3-variable")  # explicit retry
        self.assertTrue(wait_for(manager, lambda s: s["status"] == "ready"))
        self.assertEqual(len(attempts), 2)

    def test_inference_conflicts_and_switch_locking_during_scoring(self):
        block = threading.Event()
        manager = ModelManager(
            registry=fake_registry(baseline_factory=lambda: FakeBaseline(block=block)))
        self.assertTrue(wait_for(manager, lambda s: s["status"] == "ready"))
        first = threading.Thread(target=lambda: manager.analyze(None, TEXT), daemon=True)
        first.start()
        self.assertTrue(wait_for(manager, lambda s: s["busy"]))
        with self.assertRaises(OperationConflictError):
            manager.analyze(None, TEXT)
        with self.assertRaises(OperationConflictError):
            manager.activate("qwen3-variable")
        block.set()
        first.join(2)
        self.assertTrue(wait_for(manager, lambda s: not s["busy"]))

    def test_invalid_text_rejected_before_locking(self):
        manager = ModelManager(registry=fake_registry())
        self.assertTrue(wait_for(manager, lambda s: s["status"] == "ready"))
        with self.assertRaises(ValueError):
            manager.analyze(None, "short")
        self.assertFalse(manager.status()["busy"])


class QwenResultContractTests(unittest.TestCase):
    def test_document_result_has_no_passages_and_exact_coverage(self):
        adapter = FakeQwenAdapter(score=0.8)
        result = _qwen_result(adapter, TEXT, 0.0)
        self.assertEqual(result["aiConfidence"], 0.8)
        self.assertAlmostEqual(result["humanRate"], 0.2)
        self.assertEqual(result["verdict"], "ai")
        self.assertEqual(result["passages"], [])
        self.assertFalse(result["passageScores"])
        self.assertEqual(result["segments"], [{"text": TEXT}])  # exact reconstruction
        self.assertEqual("".join(seg["text"] for seg in result["segments"]), TEXT)
        self.assertEqual(result["coverage"], {"textTokensAnalyzed": len(TEXT.split()),
                                             "textTokensTotal": len(TEXT.split()),
                                             "textTokenLimit": 1023, "truncated": False})
        self.assertEqual(result["resultSchemaVersion"], 2)
        self.assertFalse(result["scoreSemantics"]["mixedClass"])
        self.assertIsNone(result["scoreSemantics"]["referenceMapping"])
        self.assertEqual(result["engine"]["id"], "fake/qwen-detector")
        self.assertEqual(result["engine"]["revision"], "fake-revision")
        capabilities = result["engine"]["capabilities"]
        self.assertFalse(capabilities["passageScores"])
        self.assertEqual(capabilities["thresholdPolicy"], "provisional-display-bands")
        self.assertTrue(capabilities["experimental"])
        self.assertEqual(capabilities["textTokenLimit"], 1023)
        self.assertTrue(result["notices"])
        self.assertEqual(adapter.scored, [TEXT])  # single scoring pass

    def test_oversized_text_reports_exact_counts_without_scoring(self):
        adapter = FakeQwenAdapter(max_tokens=5)
        long_text = " ".join(["word"] * 12)
        with self.assertRaises(DocumentTooLongError) as caught:
            _qwen_result(adapter, long_text, 0.0)
        error = caught.exception
        self.assertEqual(error.text_tokens, 12)
        self.assertEqual(error.text_token_limit, 5)
        self.assertEqual(error.total_tokens, 13)
        self.assertEqual(error.total_limit, 6)
        self.assertEqual(adapter.scored, [])  # nothing was scored

    def test_adapter_context_limit_error_is_converted_with_text_token_counts(self):
        adapter = FakeQwenAdapter(max_tokens=10, limit_error=True)
        with self.assertRaises(DocumentTooLongError) as caught:
            _qwen_result(adapter, TEXT, 0.0)  # 6 fake tokens pass the pre-check
        error = caught.exception
        # The adapter's ContextLimitError counts the EOS readout token; strip it.
        self.assertEqual(error.text_tokens, 6)
        self.assertEqual(error.text_token_limit, 6)
        self.assertEqual(error.total_tokens, 7)
        self.assertEqual(error.total_limit, 7)


if __name__ == "__main__":
    unittest.main()
