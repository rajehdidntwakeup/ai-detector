import json
from pathlib import Path
import tempfile
import unittest
from evaluation.run import load_manifest, rate, summarize


class EvaluationTests(unittest.TestCase):
    def test_missing_class_is_not_zero_error(self):
        result = summarize([{"label": "ai", "verdict": "human"}])
        self.assertIsNone(result["falsePositiveRate"]["rate"])
        self.assertEqual(result["aiRecall"]["rate"], 0)
        self.assertGreater(rate(0, 40)["wilson95"][1], 0)

    def test_mixed_excluded_from_binary_metrics(self):
        result = summarize([{"label": "mixed", "verdict": "ai"}, {"label": "human", "verdict": "uncertain"}])
        self.assertEqual(result["falsePositiveRate"]["n"], 1)
        self.assertIsNone(result["aiRecall"]["rate"])
        self.assertEqual(result["uncertainRate"]["rate"], .5)

    def test_provenance_duplicates_and_split_leakage(self):
        base = {"id": "a", "text": "An original test document.", "label": "ai", "source": "test", "permission": "authored", "group": "one", "split": "development"}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "manifest.jsonl"
            for second in (base, base | {"id": "b"}, base | {"id": "b", "text": "Different document.", "split": "heldout"}, base | {"id": "b", "source": ""}):
                path.write_text(json.dumps(base)+"\n"+json.dumps(second), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_manifest(path)
            path.write_text(json.dumps(base), encoding="utf-8")
            self.assertEqual(len(load_manifest(path)), 1)

    def test_calibration_split_and_optional_provenance(self):
        record = {"id": "a", "text": "An original test document.", "label": "ai", "source": "test", "permission": "authored", "group": "one", "split": "calibration", "domain": "nursing", "generatorFamily": "gpt4"}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "manifest.jsonl"
            path.write_text(json.dumps(record), encoding="utf-8")
            loaded = load_manifest(path)
            self.assertEqual(loaded[0]["split"], "calibration")
            self.assertEqual(loaded[0]["domain"], "nursing")
