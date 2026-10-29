import copy
import hashlib
import unittest
from evaluation.calibrate import find_thresholds, leaning, validate_comparison
from evaluation.compare import implementation_hash


class CalibrationContractTests(unittest.TestCase):
    def test_impossible_target_does_not_fall_back_to_false_accusations(self):
        rows = [{"label": "human", "score": 1.0}, {"label": "ai", "score": 0.0}]
        self.assertEqual(find_thresholds(rows)["status"], "no_feasible_operating_point")
        self.assertEqual(find_thresholds(rows[:1])["status"], "insufficient_classes")

    def test_small_scores_never_produce_negative_human_thresholds(self):
        rows = [{"label": "human", "score": .001}, {"label": "ai", "score": .02}]
        t = find_thresholds(rows)
        self.assertTrue(0 <= t["human"] < t["ai"] <= 1)
        self.assertEqual(leaning(.001, t), "human")
        self.assertEqual(leaning(.02, t), "ai")

    def test_ties_and_no_recall_can_still_meet_the_fp_target(self):
        rows = [{"label": "human", "score": .5}, {"label": "ai", "score": .5}]
        t = find_thresholds(rows, 0)
        self.assertEqual(t["status"], "fitted")
        self.assertGreater(t["ai"], .5)
        self.assertEqual(t["aiRecall"], 0)
        self.assertEqual(leaning(.5, t), "uncertain")

    def test_invalid_targets_and_scores_rejected(self):
        for target in (-1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                find_thresholds([], target)
        for score in (-.01, 1.1, float("inf")):
            with self.assertRaises(ValueError):
                find_thresholds([{"label": "human", "score": score}])

    def test_calibration_rejects_stale_partial_duplicate_and_mislabeled_scores(self):
        record = {"id": "one", "text": "Exact source text.", "label": "ai", "split": "calibration", "group": "source"}
        row = {k: record[k] for k in ("id", "label", "split", "group")}
        row.update(status="ok", score=.7, sha256=hashlib.sha256(record["text"].encode()).hexdigest())
        valid = {"schemaVersion": 2, "implementationSha256": implementation_hash(), "manifestSha256": "expected",
                 "engines": [{"engine": {"id": "test"}, "rows": [row]}]}
        validate_comparison(valid, [record], "expected")
        mutations = [lambda r: r.update(schemaVersion=1), lambda r: r.update(manifestSha256="old"),
                     lambda r: r["engines"][0]["rows"].clear(),
                     lambda r: r["engines"][0]["rows"].append(row),
                     lambda r: r["engines"][0]["rows"][0].update(sha256="changed"),
                     lambda r: r["engines"][0]["rows"][0].update(label="human"),
                     lambda r: r["engines"][0]["rows"][0].update(status="unsupported_length", score=None)]
        for mutate in mutations:
            modified = copy.deepcopy(valid)
            mutate(modified)
            with self.assertRaises(ValueError):
                validate_comparison(modified, [record], "expected")
