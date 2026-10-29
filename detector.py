"""Offline inference. Model files are loaded only from the project directory."""
import os

# Set before importing any Hugging Face library. Never download during inference.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["DO_NOT_TRACK"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import math
import re
import time
from dataclasses import dataclass

from model_info import (MAX_CHARACTERS, MAX_WORDS, MODEL_DIR, MODEL_ID, MODEL_REVISION,
                        CONTEXT_TOKENS, VERDICT_THRESHOLDS, METHOD, LIMITATIONS, SCORE_SEMANTICS)
from engines.inputs import bert_window


def validate_text(value):
    if not isinstance(value, str):
        raise ValueError("Provide text as a string.")
    text = value.strip()
    if len(text) < 10:
        raise ValueError("Enter at least 10 characters to analyze.")
    if len(text) > MAX_CHARACTERS:
        raise ValueError(f"Keep the text under {MAX_CHARACTERS:,} characters.")
    if not any(char.isalpha() for char in text):
        raise ValueError("Add some written text, rather than only numbers or punctuation.")
    if len(text.split()) > MAX_WORDS:
        raise ValueError(f"This version supports up to {MAX_WORDS:,} words per check. No text was analyzed.")
    return text


@dataclass(frozen=True)
class Span:
    start: int
    end: int


def passage_spans(text, limit=80):
    """Preserve offsets; combine neighboring sentences for a bounded CPU workload."""
    ends = []
    abbreviations = {"mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "e.g.", "i.e.", "vs.", "etc."}
    for match in re.finditer(r'[.!?。！？]+[\"\u201d\u2019\')\]]*(?=\s|$)|\n\s*\n', text):
        before = text[:match.end()].rstrip()
        last_word = before.split()[-1].lower() if before.split() else ""
        if last_word in abbreviations or re.fullmatch(r"[A-Za-z]\.", last_word):
            continue
        ends.append(match.end())
    if not ends or ends[-1] != len(text):
        ends.append(len(text))
    spans = []
    start = 0
    for end in ends:
        while start < end and text[start].isspace():
            start += 1
        trimmed_end = end
        while trimmed_end > start and text[trimmed_end - 1].isspace():
            trimmed_end -= 1
        if trimmed_end > start:
            spans.append(Span(start, trimmed_end))
        start = end
    if len(spans) > limit:
        group_size = math.ceil(len(spans) / limit)
        spans = [Span(spans[i].start, spans[min(i + group_size, len(spans)) - 1].end)
                 for i in range(0, len(spans), group_size)]
    return spans


def leaning(score):
    if score >= VERDICT_THRESHOLDS["ai"]:
        return "ai"
    if score <= VERDICT_THRESHOLDS["human"]:
        return "human"
    return "uncertain"


def ai_portion(passages, scores):
    """Share of words inside AI-leaning passages: how much of the document is flagged,
    rather than how strongly. Human-leaning passages can mask AI passages in the
    token-weighted mean, the way a coverage-style report unmasks them."""
    counts = [len(passage.split()) for passage in passages]
    total = sum(counts)
    if not total:
        return 0.0
    flagged = sum(count for count, score in zip(counts, scores) if leaning(score) == "ai")
    return flagged / total


class Detector:
    def __init__(self):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if not (MODEL_DIR / "model.safetensors").is_file():
            raise RuntimeError("The local model is missing. Run setup.cmd once while connected to the internet.")
        self.torch = torch
        torch.set_num_threads(max(1, min(4, (os.cpu_count() or 2) // 2)))
        self.metadata = json.loads((MODEL_DIR / "detector-config.json").read_text(encoding="utf-8"))
        self.tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True, trust_remote_code=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(MODEL_DIR), local_files_only=True, trust_remote_code=False, use_safetensors=True,
        ).to("cpu").eval()
        self.temperature = float(self.metadata["temperature"])
        self.ai_index = int(self.metadata["label_mapping"]["ai"])
        self.max_tokens = min(512, int(self.metadata["max_length"]))
        if not math.isfinite(self.temperature) or self.temperature <= 0 or self.ai_index not in (0, 1):
            raise RuntimeError("Invalid model metadata. Run setup.cmd to restore the pinned model.")
        self.score_texts(["A short sentence warms up the local text analysis model."])

    def score_texts(self, texts):
        """Average non-overlapping token windows, weighted by their content length."""
        capacity = self.max_tokens - self.tokenizer.num_special_tokens_to_add(pair=False)
        tasks = []
        for owner, text in enumerate(texts):
            ids = self.tokenizer.encode(text, add_special_tokens=False, truncation=False, verbose=False)
            for start in range(0, max(1, len(ids)), capacity):
                chunk = ids[start:start + capacity]
                tasks.append((owner, max(1, len(chunk)), chunk))
        totals = [0.0] * len(texts)
        weights = [0] * len(texts)
        with self.torch.inference_mode():
            for start in range(0, len(tasks), 4):
                batch = tasks[start:start + 4]
                inputs = self.tokenizer.pad([bert_window(self.tokenizer, chunk) for _, _, chunk in batch], padding=True, return_tensors="pt")
                logits = self.model(**inputs).logits.float() / self.temperature
                probabilities = self.torch.softmax(logits, dim=-1)[:, self.ai_index].tolist()
                for (owner, weight, _), probability in zip(batch, probabilities):
                    if not math.isfinite(probability):
                        raise RuntimeError("The model returned an invalid score. Please retry.")
                    totals[owner] += probability * weight
                    weights[owner] += weight
        return [total / max(1, weight) for total, weight in zip(totals, weights)]

    def analyze(self, value):
        text = validate_text(value)
        started = time.perf_counter()
        spans = passage_spans(text)
        passages = [text[span.start:span.end] for span in spans]
        scores = self.score_texts([text] + passages)
        ai = scores[0]
        passage_scores = scores[1:]
        word_count = len(text.split())
        text_tokens = len(self.tokenizer.encode(text, add_special_tokens=False, truncation=False, verbose=False))
        capacity = self.max_tokens - self.tokenizer.num_special_tokens_to_add(pair=False)
        notices = []
        if word_count < 80:
            notices.append("This is a short sample. Try at least 80 words for more context.")
        notices.append("This English-language model can misclassify text. Its score is not proof of authorship.")
        return {
            "resultSchemaVersion": 2,
            "scoreSemantics": dict(SCORE_SEMANTICS),
            "passageScores": True,
            "coverage": {"textTokensAnalyzed": text_tokens, "textTokensTotal": text_tokens,
                         "documentWindows": max(1, math.ceil(text_tokens / capacity)), "truncated": False},
            "originalContent": text,
            "aiConfidence": ai,
            "humanRate": 1 - ai,
            "aiPortion": ai_portion(passages, passage_scores),
            "verdict": leaning(ai),
            "wordCount": word_count,
            "passages": [
                {"text": passage, "aiScore": score, "label": leaning(score)}
                for passage, score in zip(passages, passage_scores)
            ],
            # Python offsets are kept out of the browser: these text segments handle
            # Unicode without mixing code-point and JavaScript UTF-16 indices.
            "segments": self._segments(text, spans, passage_scores),
            "durationMs": round((time.perf_counter() - started) * 1000),
            "notices": notices,
            "engine": {"id": MODEL_ID, "revision": MODEL_REVISION, "offline": True,
                       "capabilities": {"contextTokens": CONTEXT_TOKENS,
                                        "verdictThresholds": VERDICT_THRESHOLDS,
                                        "method": METHOD, "limitations": LIMITATIONS,
                                        "passageScores": True, "scoreSemantics": dict(SCORE_SEMANTICS)}},
            "method": METHOD,
        }

    @staticmethod
    def _segments(text, spans, scores):
        output = []
        cursor = 0
        for index, (span, score) in enumerate(zip(spans, scores)):
            if span.start > cursor:
                output.append({"text": text[cursor:span.start]})
            output.append({"text": text[span.start:span.end], "passageIndex": index,
                           "aiScore": score, "label": leaning(score)})
            cursor = span.end
        if cursor < len(text):
            output.append({"text": text[cursor:]})
        return output

