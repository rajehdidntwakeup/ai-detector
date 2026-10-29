import unittest
from evaluation.reference_probe import analyze, display_percent, summarize


def capture(identifier='test-one', score=.8):
    return {'captureId': identifier, 'observationType': 'observed', 'capturedAt': '2026-10-29T12:00:00Z',
            'request': {'submittedText': 'A. A.'},
            'responseState': {'returnedText': 'A. A.', 'fields': {'aiConfidence': score, 'humanRate': .1},
                              'sentences': [{'text': 'A.', 'classProbabilities': {'ai': v}} for v in (.2,.4)]}}


class ProbeTests(unittest.TestCase):
    def test_probe_lineage_and_exact_repeats(self):
        from evaluation.build_reference_probes import build
        rows = build()
        self.assertEqual(len(rows), 11)
        self.assertEqual(len({r['id'] for r in rows}), 11)
        self.assertEqual(len({r['group'] for r in rows}), 1)
        self.assertEqual(rows[0]['sha256'], rows[-1]['sha256'])
        self.assertEqual(rows[0]['text'], rows[-2]['text'])
        self.assertTrue(all(r['referenceStatus'] == 'pending-manual-capture' for r in rows))

    def test_rounding_and_missing(self):
        self.assertEqual(display_percent(.985), 99)
        self.assertEqual(display_percent(1.03), 100)
        self.assertIsNone(display_percent(True))
        self.assertIsNone(display_percent(float('nan')))

    def test_repeated_alignment_and_aggregation(self):
        row = analyze(capture())
        self.assertEqual(row['alignedSentences'], 2)
        self.assertEqual(row['coveredCharacters'], 4)
        self.assertTrue(row['outsideSentenceRange'])
        self.assertAlmostEqual(row['aggregationHypotheses']['sentenceMean']['residualPoints'], 50)

    def test_missing_scores_no_aggregate(self):
        record = capture()
        record['responseState']['sentences'][0]['classProbabilities'] = None
        self.assertEqual(analyze(record)['aggregationHypotheses'], {})

    def test_repeats_and_synthetic_exclusion(self):
        synthetic = capture('synthetic', .1)
        synthetic['observationType'] = 'synthetic'
        report = summarize([capture(), capture('second', .9), synthetic])
        self.assertEqual(len(report['rows']), 2)
        self.assertAlmostEqual(report['repeatability'][0]['rangePoints'], 10)
        self.assertIsNone(summarize([capture()])['repeatability'][0]['rangePoints'])


if __name__ == '__main__':
    unittest.main()
