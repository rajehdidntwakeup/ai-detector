import unittest

from evaluation.reference_compare import compare, correlation, metrics, prepare


class ReferenceComparisonTests(unittest.TestCase):
    def test_duplicate_text_not_independent_case(self):
        base = {'observationType': 'observed', 'request': {'submittedText': 'Exact text'},
                'responseState': {'fields': {'aiConfidence': .8}}}
        cases, observations, excluded = prepare([
            base | {'captureId': 'first'}, base | {'captureId': 'repeat'},
            base | {'captureId': 'synthetic', 'observationType': 'synthetic'},
            base | {'captureId': 'invalid', 'responseState': {'fields': {'aiConfidence': True}}}])
        self.assertEqual(len(cases), 1)
        self.assertEqual(len(observations), 2)
        self.assertEqual(excluded, ['synthetic', 'invalid'])
        self.assertEqual(cases[0]['label'], 'unknown')

    def test_shared_coverage_does_not_hide_failures(self):
        results = [
            {'name': 'one', 'rows': [{'sha256': 'a', 'status': 'ok', 'score': .9},
                                    {'sha256': 'b', 'status': 'ok', 'score': .5}]},
            {'name': 'two', 'rows': [{'sha256': 'a', 'status': 'ok', 'score': .8},
                                    {'sha256': 'b', 'status': 'unsupported_length', 'score': None}]}]
        observations = [{'sha256': 'a', 'score': .7}, {'sha256': 'a', 'score': .9},
                        {'sha256': 'b', 'score': .5}]
        report = compare(results, observations, 2)
        self.assertEqual(report['commonDocuments'], 1)
        self.assertEqual(report['engines'][1]['coverage'], .5)
        self.assertEqual(report['engines'][1]['failedOrUnsupported'], 1)
        self.assertAlmostEqual(report['engines'][0]['commonSupported']['maePoints'], 10)
        self.assertAlmostEqual(report['engines'][0]['allSupported']['maePoints'], 5)

    def test_engine_failure_leaves_no_common_subset(self):
        report = compare([{'name': 'failed', 'rows': [], 'error': 'load failed'}],
                         [{'sha256': 'a', 'score': .8}], 1)
        self.assertEqual(report['commonDocuments'], 0)
        self.assertIsNone(report['engines'][0]['commonSupported']['maePoints'])

    def test_small_sample_correlation_unavailable(self):
        self.assertIsNone(metrics([(.9, .8)])['rankCorrelation'])
        self.assertIsNone(correlation([1, 1, 1], [1, 2, 3]))
        self.assertAlmostEqual(correlation([1, 2, 3], [3, 2, 1]), -1)
        self.assertAlmostEqual(correlation([1, 1, 3], [1, 1, 3]), 1)


if __name__ == '__main__':
    unittest.main()
