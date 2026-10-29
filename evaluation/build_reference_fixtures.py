"""Build the committed synthetic turnitindetector capture fixtures.

Constructs purpose-written inputs with stable IDs and reference-shaped response
fixtures that are explicitly synthetic. Outputs live in
evaluation/fixtures/turnitindetector/ and carry no real reference observations.
Run from the repository root:

    .\\.venv\\Scripts\\python.exe -m evaluation.build_reference_fixtures
"""
import json
from pathlib import Path

OUT = Path(__file__).with_name("fixtures") / "turnitindetector"

VOCABULARY = ("the", "report", "examines", "plain", "examples", "every", "line",
              "keeps", "this", "synthetic", "sample", "steady", "reader", "notes",
              "careful", "review", "common", "words", "small", "check")


def _filler_words(count):
    return [VOCABULARY[index % len(VOCABULARY)].capitalize() if index % 10 == 0
            else VOCABULARY[index % len(VOCABULARY)] for index in range(count)]


def _repeat(sentence, times, separator=" "):
    return separator.join(sentence for _ in range(times))


def _utf16_units(text):
    """JavaScript string length: UTF-16 code units, not Python code points."""
    return len(text.encode("utf-16-le")) // 2


def _record(identifier, text, purpose, composition):
    return {
        "id": identifier,
        "observationType": "synthetic",
        "purpose": purpose,
        "composition": composition,
        "text": text,
        "utf16Units": _utf16_units(text),
        "codePoints": len(text),
        "wordCount": len(text.split()),
    }


def synthetic_inputs():
    inputs = []
    inputs.append(_record(
        "synth-utf16-9", "Detect me",
        "9 UTF-16 units: below the 10-UTF-16-unit minimum the frontend checks; expect rejection.",
        "synthetic"))
    inputs.append(_record(
        "synth-utf16-10", "Detect me?",
        "Exactly 10 UTF-16 units after trimming; boundary acceptance unknown until captured.",
        "synthetic"))
    inputs.append(_record(
        "synth-utf16-11", "Detect me!!",
        "11 UTF-16 units: one unit above the observed minimum boundary.",
        "synthetic"))
    # One surrogate pair: 9 code points, 10 UTF-16 units. Our Python validator
    # counts code points; the frontend counts UTF-16 units. This probes the gap.
    inputs.append(_record(
        "synth-utf16-surrogate", "Abc \U0001f600 ok!",
        "10 UTF-16 units but 9 code points: detects code-point versus UTF-16 counting.",
        "synthetic"))
    inputs.append(_record(
        "synth-utf16-surrogate-11", "Abc \U0001f600 ok!?",
        "One unit above the boundary while still a surrogate-pair case.",
        "synthetic"))
    inputs.append(_record(
        "synth-no-letters", "1234567890!!",
        "Passes a UTF-16 length check but contains no letters; probes the hidden letter requirement.",
        "synthetic"))
    for count in (1999, 2000, 2001):
        text = " ".join(_filler_words(count))
        inputs.append(_record(
            f"synth-words-{count}", text,
            f"Whitespace word count {count}: probes the observed 2000-word submission cutoff.",
            "synthetic"))
    # Deliberately longer editor text so a capture can record the submitted prefix.
    editor = " ".join(_filler_words(2501))
    record = _record(
        "synth-editor-2501", editor,
        "Editor text above the cutoff: capture records submitted prefix versus editor contents.",
        "synthetic")
    record["expectedSubmittedPrefixWords"] = 2000
    inputs.append(record)
    inputs.append(_record(
        "synth-repeat-sentences", _repeat("This sentence will repeat exactly three times!", 3),
        "Alignment probe: repeated identical sentences find one occurrence or scan forward in order.",
        "synthetic"))
    inputs.append(_record(
        "synth-unicode-align",
        "Mixed “quotes” and CJK 汇率和利率. Combining áccent, emoji \U0001f600, and text…",
        "Alignment probe: Unicode punctuation, CJK sentence ends, combining marks, and astral characters.",
        "synthetic"))
    inputs.append(_record(
        "synth-refs-abbrev",
        "Dr. Smith met Prof. Lee et al. at 5 p.m. (e.g. during the review). The study cites three papers, e.g. Lee (2020). "
        "Results: 4.5 ± 0.2 ms, i.e. slower. See refs: J. Doe, 2019; R. Roe, 2021.",
        "Alignment probe: abbreviations, citations, and numbers that should not end sentences.",
        "synthetic"))
    inputs.append(_record(
        "synth-composition-marked",
        "Machine drafting tends to deliver smooth, balanced, evenly hedged declarative sentences. "
        "Well, look, I dunno about that. The whole thing felt kinda sketchy to me, honestly.",
        "Composition probe: an AI-style paragraph followed by an informal human-voice paragraph; labels stay synthetic.",
        "synthetic"))
    inputs.append(_record(
        "synth-plain-baseline",
        "The quick review covers eleven steady sentences of ordinary prose. It stays plain on purpose so every "
        "capture session can compare scores against the same short body. Nothing here looks academic or styled; "
        "the wording keeps to common verbs and everyday nouns. Sessions repeat it unchanged between probes to "
        "observe score stability over time.",
        "Stable repeat control: identical text resubmitted across sessions for repeatability.",
        "synthetic"))
    return inputs


def synthetic_responses():
    """Reference-shaped envelopes constructed locally. Never evidence of server behavior."""
    def envelope(ai_confidence, human_rate, sentences=None, extra=None, keys=None):
        data = {
            "originalContent": "The quick review covers eleven steady sentences of ordinary prose. It stays plain.",
            "aiConfidence": ai_confidence,
            "humanRate": human_rate,
        }
        if sentences is not None:
            data["sentences"] = sentences
        if extra:
            data.update(extra)
        if keys is not None:
            data = {key: data[key] for key in keys if key in data}
        return {"success": True, "data": data}

    return [
        {
            "id": "resp-synth-success-full",
            "observationType": "synthetic",
            "purpose": "Complete well-formed response with sentence records for display logic tests.",
            "synthetic": True,
            "status": 200,
            "raw": envelope(
                0.83, 0.17,
                sentences=[
                    {"sentence": "The quick review covers eleven steady sentences of ordinary prose.",
                     "class_probabilities": {"ai": 0.9, "mixed": 0.05, "human": 0.05}},
                    {"sentence": "It stays plain.",
                     "class_probabilities": {"ai": 0.2, "mixed": 0.1, "human": 0.7}},
                ]),
        },
        {
            "id": "resp-synth-mixed-noncomplement",
            "observationType": "synthetic",
            "purpose": "aiConfidence + humanRate well away from 1: probes whether mixed is derived as a remainder.",
            "synthetic": True,
            "status": 200,
            "raw": envelope(0.6, 0.25),
        },
        {
            "id": "resp-synth-missing-sentences",
            "observationType": "synthetic",
            "purpose": "Absent sentences field: highlighting must be optional, not assumed.",
            "synthetic": True,
            "status": 200,
            "raw": envelope(0.4, 0.6, keys=["originalContent", "aiConfidence", "humanRate"]),
        },
        {
            "id": "resp-synth-sentence-missing-class",
            "observationType": "synthetic",
            "purpose": "Sentence class_probabilities missing a key: probes the fallback/badge behavior for absent values.",
            "synthetic": True,
            "status": 200,
            "raw": envelope(
                0.7, 0.3,
                sentences=[
                    {"sentence": "Only one class is reported here.", "class_probabilities": {"ai": 1.0}},
                    {"sentence": "This one reports none.", "class_probabilities": {}},
                    {"sentence": "This one reports nothing at all."},
                ]),
        },
        {
            "id": "resp-synth-tie-document",
            "observationType": "synthetic",
            "purpose": "Document-level equal shares: observed rule prioritizes human, then mixed, then AI.",
            "synthetic": True,
            "status": 200,
            "raw": envelope(0.5, 0.5),
        },
        {
            "id": "resp-synth-tie-sentence",
            "observationType": "synthetic",
            "purpose": "Sentence-level equal shares: observed rule prioritizes AI, then mixed, then human.",
            "synthetic": True,
            "status": 200,
            "raw": envelope(
                0.5, 0.5,
                sentences=[
                    {"sentence": "A tied sentence.", "class_probabilities": {"ai": 0.4, "mixed": 0.4, "human": 0.2}},
                    {"sentence": "An unresolved tie.", "class_probabilities": {"ai": 0.5, "mixed": 0.5}},
                    {"sentence": "Another unresolved tie.", "class_probabilities": {"ai": 0.5, "mixed": 0.5}},
                ]),
        },
        {
            "id": "resp-synth-string-number",
            "observationType": "synthetic",
            "purpose": "Scores returned as strings: probes type coercion before clamping/rounding.",
            "synthetic": True,
            "status": 200,
            "raw": {"success": True, "data": {"originalContent": "Text.", "aiConfidence": "0.83", "humanRate": "0.17"}},
        },
        {
            "id": "resp-synth-out-of-range",
            "observationType": "synthetic",
            "purpose": "Values outside [0, 1]: probes the observed clamping step.",
            "synthetic": True,
            "status": 200,
            "raw": {"success": True, "data": {"originalContent": "Text.", "aiConfidence": 1.4, "humanRate": -0.4}},
        },
        {
            "id": "resp-synth-failure-success-false",
            "observationType": "synthetic",
            "purpose": "success false with HTTP 200: local fixture for the failure state the frontend shows.",
            "synthetic": True,
            "status": 200,
            "raw": {"success": False, "data": None},
        },
        {
            "id": "resp-synth-failure-http-error",
            "observationType": "synthetic",
            "purpose": "HTTP error status: local fixture for the failure presentation.",
            "synthetic": True,
            "status": 500,
            "raw": {"success": False},
        },
        {
            "id": "resp-synth-ai-rateresult",
            "observationType": "synthetic",
            "purpose": "aiRateResult object: exercises the named-detector badge rendering (labels remain research observations).",
            "synthetic": True,
            "status": 200,
            "raw": {
                "success": True,
                "data": {
                    "originalContent": "Text.",
                    "aiConfidence": 0.66,
                    "humanRate": 0.34,
                    "aiRateResult": {"GPTZero": 61.2, "WinstonAI": 70.0},
                },
            },
        },
    ]


def synthetic_captures():
    """Capture records built from the fixtures above, all labeled synthetic."""
    inputs = {item["id"]: item for item in synthetic_inputs()}
    captures = []
    for record in synthetic_responses():
        raw = record["raw"]
        data = raw.get("data") or {}
        captures.append({
            "captureId": record["id"],
            "observationType": "synthetic",
            "capturedAt": "1970-01-01T00:00:00Z",
            "transport": "synthetic-construct",
            "source": {"url": "constructed-locally"},
            "request": {
                "endpoint": "/bff/detect",
                "submittedText": inputs["synth-plain-baseline"]["text"],
                "submittedTextSha256": None,
            },
            "responseState": {
                "status": record["status"],
                "success": raw.get("success"),
                "dataPresent": raw.get("success") is True,
                "fields": {
                    key: data[key] for key in ("aiConfidence", "humanRate", "aiRateResult") if key in data
                },
                "raw": raw,
                "returnedText": data.get("originalContent"),
                "sentences": [
                    {"text": item.get("sentence", ""),
                     "classProbabilities": item.get("class_probabilities")}
                    for item in data.get("sentences", []) or []
                ],
            },
            "protocolVersion": "1",
            "notes": record["purpose"],
        })
    return captures


def write(name, payload, indent=2):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    body = payload if indent is None else json.dumps(payload, ensure_ascii=False, indent=indent)
    path.write_text(body, encoding="utf-8")
    print(f"wrote {path}")


def main():
    write("synthetic_inputs.json", synthetic_inputs())
    write("synthetic_responses.json", synthetic_responses())
    write("synthetic_captures.jsonl",
          "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                  for record in synthetic_captures()), indent=None)


if __name__ == "__main__":
    main()