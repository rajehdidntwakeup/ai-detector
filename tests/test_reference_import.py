import json
from pathlib import Path
import tempfile
import unittest

from evaluation.import_reference import (
    check_texts,
    import_file,
    load_schema,
    scan_secrets,
    validate_record,
    write_registry,
)

FIXTURES = Path(__file__).parent.parent / "evaluation" / "fixtures" / "turnitindetector"


def minimal_record(**overrides):
    record = {
        "captureId": "synth-test-record",
        "observationType": "synthetic",
        "capturedAt": "2026-10-29T12:00:00Z",
        "transport": "synthetic-construct",
        "source": {"url": "constructed-locally"},
        "request": {
            "endpoint": "/bff/detect",
            "submittedText": "Plain synthetic wording only.",
            "submittedTextSha256": None,
        },
        "responseState": {
            "status": 200,
            "success": True,
            "dataPresent": True,
            "fields": {"aiConfidence": 0.5, "humanRate": 0.5},
            "raw": {},
            "returnedText": None,
            "sentences": [],
        },
        "protocolVersion": "1",
    }
    record.update(overrides)
    return record


class ImportValidationTests(unittest.TestCase):
    def setUp(self):
        self.schema = load_schema()

    def test_built_fixtures_import(self):
        records, report = import_file(FIXTURES / "synthetic_captures.jsonl", schema=self.schema)
        self.assertTrue(records)
        for record, row in zip(records, report):
            self.assertEqual(row["errors"], [], record["captureId"])
            self.assertEqual(record["observationType"], "synthetic")
            self.assertIn("importedAt", record)

    def test_synthetic_inputs_are_consistent(self):
        inputs = json.loads((FIXTURES / "synthetic_inputs.json").read_text(encoding="utf-8"))
        for item in inputs:
            units = len(item["text"].encode("utf-16-le")) // 2
            self.assertEqual(item["utf16Units"], units)
            self.assertEqual(item["codePoints"], len(item["text"]))
            self.assertEqual(item["wordCount"], len(item["text"].split()))
        # The surrogate-pair probe exists to differ between the two counts.
        probe = {item["id"]: item for item in inputs}["synth-utf16-surrogate"]
        self.assertEqual(probe["utf16Units"], 10)
        self.assertEqual(probe["codePoints"], 9)
        # Minimum-length boundary probes sit exactly on their named units.
        for expected in (9, 10, 11):
            boundary = {item["id"]: item for item in inputs}[f"synth-utf16-{expected}"]
            self.assertEqual(boundary["utf16Units"], expected)
            self.assertEqual(len(boundary["text"].strip()), expected)

    def test_missing_required_fields_rejected(self):
        for missing in ("captureId", "observationType", "request", "responseState", "protocolVersion"):
            record = minimal_record()
            del record[missing]
            errors = validate_record(record, self.schema)
            self.assertTrue(any(missing in error for error in errors), missing)

    def test_enum_and_const_rejected(self):
        record = minimal_record(observationType="invented-elsewhere")
        self.assertGreater(validate_record(record, self.schema), [])
        record = minimal_record()
        record["sanitization"] = {"acknowledged": False}
        self.assertTrue(validate_record(record, self.schema))

    def test_capture_id_pattern(self):
        record = minimal_record(captureId="Bad Id!")
        errors = validate_record(record, self.schema)
        self.assertTrue(any("pattern" in error for error in errors))

    def test_timestamp_timezone_required(self):
        record = minimal_record(capturedAt="2026-10-29T12:00:00")
        errors = validate_record(record, self.schema)
        self.assertTrue(any("date-time" in error for error in errors))


class IntegrityTests(unittest.TestCase):
    def setUp(self):
        self.schema = load_schema()

    def test_hash_mismatch_is_fatal(self):
        record = minimal_record()
        record["request"]["submittedText"] = "Different wording entirely."
        record["request"]["submittedTextSha256"] = "0" * 64
        with self.assertRaises(ValueError):
            check_texts(record)

    def test_unicode_counts_and_missing_hashes(self):
        from evaluation.import_reference import sha256_utf8
        record = minimal_record()
        text = "Hello 😀 world"
        record["request"].update(submittedText=text, editorText=text)
        record["responseState"]["returnedText"] = text
        check_texts(record)
        self.assertEqual(record["request"]["submittedUtf16Units"], 14)
        for key in ("submittedTextSha256", "editorTextSha256"):
            self.assertEqual(record["request"][key], sha256_utf8(text))
        self.assertEqual(record["responseState"]["returnedTextSha256"], sha256_utf8(text))

    def test_non_object_capture_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "invalid.json"
            path.write_text('[42]', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "JSON object"):
                import_file(path)

    def test_counts_and_truncation_recomputed(self):
        editor = "alpha beta gamma delta epsilon"
        record = minimal_record()
        record["request"]["submittedText"] = "alpha beta gamma"
        record["request"]["editorText"] = editor
        warnings = check_texts(record)
        self.assertEqual(record["request"]["submittedWordCount"], 3)
        self.assertEqual(record["request"]["submittedUtf16Units"], 16)
        self.assertTrue(record["request"]["truncationObserved"])
        self.assertEqual(warnings, [])

    def test_non_prefix_divergence_warns(self):
        record = minimal_record()
        record["request"]["submittedText"] = "alpha beta gamma"
        record["request"]["editorText"] = "Beta alpha gamma delta"
        warnings = check_texts(record)
        self.assertTrue(any("prefix" in warning for warning in warnings))

    def test_returned_text_difference_warns(self):
        record = minimal_record()
        record["responseState"]["returnedText"] = "Not what was submitted."
        warnings = check_texts(record)
        self.assertTrue(any("returned" in warning for warning in warnings))

    def test_secret_scan_rejects_cookies_and_auth(self):
        record = minimal_record()
        record["responseState"]["raw"] = {"headers": {"Set-Cookie": "session=1"}}
        with self.assertRaises(ValueError):
            import_file(_temp_json(record))
        self.assertTrue(scan_secrets({"a": [{"cookie": "x"}]}))
        self.assertFalse(scan_secrets({"response": {"fields": {"aiConfidence": 0.5}}}))

    def test_load_input_shapes(self):
        record = minimal_record()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "single.json"
            path.write_text(json.dumps(record), encoding="utf-8")
            self.assertEqual(len(import_file(path)[0]), 1)
            path.write_text(json.dumps([record, record | {"captureId": "synth-test-two"}]), encoding="utf-8")
            self.assertEqual(len(import_file(path)[0]), 2)
            path.write_text(json.dumps(record) + "\n" + json.dumps(record | {"captureId": "synth-test-two"}),
                            encoding="utf-8")
            self.assertEqual(len(import_file(path)[0]), 2)


def _temp_json(record):
    folder = tempfile.mkdtemp()
    path = Path(folder) / "capture.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    return str(path)


class RegistryPathTests(unittest.TestCase):
    def setUp(self):
        self.schema = load_schema()

    def test_observed_records_cannot_land_in_fixtures(self):
        record = minimal_record(observationType="observed")
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "fixtures" / "turnitindetector" / "bad.jsonl"
            with self.assertRaises(ValueError):
                write_registry([record], target)
            allowed = Path(folder) / "captures" / "local.jsonl"
            self.assertEqual(write_registry([record], allowed), allowed)
            with self.assertRaises(ValueError):
                write_registry([record], allowed)  # duplicate captureId


if __name__ == "__main__":
    unittest.main()
