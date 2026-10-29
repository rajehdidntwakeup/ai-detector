"""Server-facing model manager and browser result builders.

One loaded model at a time. Model switching and inference share a single
operation lock, loads run in a background thread, and only allowlisted model
names are accepted. HTTP layers pass names, never paths or repository ids.
"""
import threading
import time

from detector import Detector, leaning, validate_text
from engines import Qwen3VariableDetector
from engines.inputs import ContextLimitError
from model_info import (CONTEXT_TOKENS, LIMITATIONS, METHOD, MODEL_ID, MODEL_REVISION,
                        VERDICT_THRESHOLDS, SCORE_SEMANTICS)

# Declared limits used before a load; the loaded adapter's metadata stays authoritative.
QWEN_TEXT_TOKEN_LIMIT = 1023  # text tokens; one more EOS readout token is added at scoring time
QWEN_METHOD = ("Document-level classification; score read from the "
               "variable-position EOS token.")
QWEN_LIMITATIONS = ("English-language model; scores are not proof of authorship. "
                    "Short, edited, or non-native writing can be misclassified. "
                    "Experimental model.")


class ServiceError(Exception):
    """Base class for structured model-manager failures surfaced by the API."""


class UnknownModelError(ServiceError):
    def __init__(self, name):
        self.name = name
        super().__init__(f"Unknown model: {name!r}.")


class OperationConflictError(ServiceError):
    """A load or an analysis is already in progress."""


class ModelNotReadyError(ServiceError):
    def __init__(self, detail=None):
        self.detail = detail
        super().__init__(detail or "The selected model is still loading.")


class ModelMismatchError(ServiceError):
    def __init__(self, requested, active):
        self.requested = requested
        self.active = active
        super().__init__(f"The {requested} model is not active. Activate it first, then analyze.")


class DocumentTooLongError(ServiceError):
    def __init__(self, text_tokens, text_token_limit, total_tokens, total_limit):
        self.text_tokens = text_tokens
        self.text_token_limit = text_token_limit
        self.total_tokens = total_tokens
        self.total_limit = total_limit
        super().__init__(f"Text has {text_tokens} tokens; this model scores at most "
                         f"{text_token_limit} text tokens ({total_limit} with its readout token).")


def _baseline_capabilities():
    return {"contextTokens": CONTEXT_TOKENS, "verdictThresholds": VERDICT_THRESHOLDS,
            "method": METHOD, "limitations": LIMITATIONS,
            "passageScores": True, "experimental": False, "scoreSemantics": dict(SCORE_SEMANTICS)}


def _qwen_capabilities(text_token_limit):
    return {"contextTokens": text_token_limit + 1, "textTokenLimit": text_token_limit,
            "verdictThresholds": VERDICT_THRESHOLDS,
            "thresholdPolicy": "provisional-display-bands",
            "passageScores": False, "method": QWEN_METHOD,
            "limitations": QWEN_LIMITATIONS, "experimental": True, "scoreSemantics": dict(SCORE_SEMANTICS)}


def _qwen_result(adapter, text, started):
    """Score the complete document once and build the browser response."""
    text_tokens = adapter.count_text_tokens(text)
    limit = adapter.max_tokens
    if text_tokens > limit:
        # Never truncate or window; report the exact counts.
        raise DocumentTooLongError(text_tokens, limit, text_tokens + 1, limit + 1)
    try:
        score = adapter.score_text(text)
    except ContextLimitError as exc:
        # The adapter counts the EOS readout token; expose both views consistently.
        raise DocumentTooLongError(exc.tokens - 1, exc.limit - 1, exc.tokens, exc.limit) from exc
    word_count = len(text.split())
    notices = []
    if word_count < 80:
        notices.append("This is a short sample. Try at least 80 words for more context.")
    notices.append("This English-language model can misclassify text. Its score is not proof of authorship.")
    notices.append("Document-level analysis: passage highlights are not computed by this model.")
    notices.append("The human/AI display bands are provisional for this experimental model, not validated probabilities.")
    return {
        "resultSchemaVersion": 2,
        "scoreSemantics": dict(SCORE_SEMANTICS),
        "originalContent": text,
        "aiConfidence": score,
        "humanRate": 1 - score,
        "verdict": leaning(score),
        "wordCount": word_count,
        "passages": [],
        # Unscored segments reconstruct the exact validated text with no passage data;
        # the frontend must not read this as "zero AI passages".
        "segments": [{"text": text}],
        "passageScores": False,
        "coverage": {"textTokensAnalyzed": text_tokens, "textTokensTotal": text_tokens, "textTokenLimit": limit,
                     "truncated": False},
        "durationMs": round((time.perf_counter() - started) * 1000),
        "notices": notices,
        "engine": {"id": adapter.engine_id, "revision": adapter.revision, "offline": True,
                   "capabilities": _qwen_capabilities(limit)},
        "method": QWEN_METHOD,
    }


def _load_qwen():
    adapter = Qwen3VariableDetector()
    adapter.load()
    return adapter


def default_registry():
    """Allowlisted models. HTTP requests can only reference these keys."""
    return {
        "baseline": {
            "label": "DistilBERT Detector",
            "experimental": False,
            "passageScores": True,
            "contextTokens": CONTEXT_TOKENS,
            "textTokenLimit": None,
            "engineId": MODEL_ID,
            "revision": MODEL_REVISION,
            "capabilities": _baseline_capabilities(),
            "factory": Detector,
            # The baseline keeps its existing Detector.analyze behavior untouched.
            "analyze": lambda engine, text, started: engine.analyze(text),
        },
        "qwen3-variable": {
            "label": "Qwen3-0.6B Detector (experimental)",
            "experimental": True,
            "passageScores": False,
            "contextTokens": QWEN_TEXT_TOKEN_LIMIT + 1,
            "textTokenLimit": QWEN_TEXT_TOKEN_LIMIT,
            "engineId": "rasbt/ai-text-detector-qwen3-0.6b-variable",
            "revision": "b08d92707d5de35a0b6ccf34e73461a4ab954a33",
            "capabilities": _qwen_capabilities(QWEN_TEXT_TOKEN_LIMIT),
            "factory": _load_qwen,
            "analyze": _qwen_result,
        },
    }


class ModelManager:
    """Loads one allowlisted model at a time and serializes switching with inference.

    State (active model, requested model, status, error) is guarded by
    ``_state_lock``; loading and inference are serialized by
    ``_operation_lock``. Loads run in background threads so status requests
    stay responsive.
    """

    def __init__(self, registry=None, auto_load="baseline"):
        self._registry = registry or default_registry()
        self._state_lock = threading.Lock()
        self._operation_lock = threading.Lock()
        self._engine = None
        self._active = None
        self._requested = None
        self._status = "idle"
        self._error = None
        if auto_load is not None:
            self.activate(auto_load)

    def activate(self, name):
        """Start loading ``name`` in the background; returns "loading" or "ready".

        Repeated activation of the same ready or loading model is idempotent.
        A competing switch while loading, or any switch while a check is
        running, is rejected.
        """
        if name not in self._registry:
            raise UnknownModelError(name)
        with self._state_lock:
            if self._requested == name and self._status in ("loading", "ready"):
                return self._status
            if self._status == "loading":
                raise OperationConflictError("A model switch is already in progress.")
        if not self._operation_lock.acquire(blocking=False):
            raise OperationConflictError("A check is currently running. Try again once it completes.")
        with self._state_lock:
            self._requested = name
            self._status = "loading"
            self._error = None
            # Drop the old model reference before loading the new one so
            # Python/PyTorch can release its memory.
            self._engine = None
            self._active = None
        threading.Thread(target=self._load, args=(name,), daemon=True).start()
        return "loading"

    def _load(self, name):
        # The operation lock is already held by activate(); release it here,
        # inside the state lock, so a status snapshot never reports a ready
        # model that is still marked busy.
        entry = self._registry[name]
        try:
            engine = entry["factory"]()
        except Exception as exc:
            with self._state_lock:
                self._status = "error"
                self._error = f"The {entry['label']} could not load: {exc}"
                self._operation_lock.release()
            print(self._error, flush=True)
        else:
            with self._state_lock:
                self._engine = engine
                self._active = name
                self._status = "ready"
                self._operation_lock.release()
            print(f"{entry['label']} ready. Analysis works offline.", flush=True)

    def analyze(self, name, value):
        """Score ``value`` with ``name`` (None means baseline) and return the result.

        Never returns another model's score: an active/requested mismatch
        raises instead of scoring with whatever happens to be loaded.
        """
        if name is None:
            name = "baseline"
        if name not in self._registry:
            raise UnknownModelError(name)
        text = validate_text(value)
        with self._state_lock:
            if self._status == "loading":
                raise ModelNotReadyError("The selected model is still loading.")
        if not self._operation_lock.acquire(blocking=False):
            raise OperationConflictError("A check is already running on this laptop. Please wait and try again.")
        try:
            with self._state_lock:
                if self._status != "ready":
                    raise ModelNotReadyError(self._error or "The selected model is not ready.")
                if self._active != name:
                    raise ModelMismatchError(name, self._active)
            started = time.perf_counter()
            return self._registry[name]["analyze"](self._engine, text, started)
        finally:
            self._operation_lock.release()

    def status(self):
        """Atomic snapshot for the status endpoint; never blocks on inference."""
        with self._state_lock:
            return {
                "status": self._status,
                "error": self._error,
                "activeModel": self._active,
                "requestedModel": self._requested,
                "busy": self._operation_lock.locked(),
                "models": [self._describe(key) for key in self._registry],
            }

    def _describe(self, key):
        entry = self._registry[key]
        return {"name": key, "label": entry["label"], "experimental": entry["experimental"],
                "passageScores": entry["passageScores"], "contextTokens": entry["contextTokens"],
                "textTokenLimit": entry.get("textTokenLimit"),
                "engineId": entry.get("engineId"), "revision": entry.get("revision"),
                "capabilities": entry.get("capabilities")}
