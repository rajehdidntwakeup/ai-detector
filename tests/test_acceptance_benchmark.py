import unittest

from evaluation.benchmark_acceptance import TOPICS, timings, workload


class WorkloadTests(unittest.TestCase):
    def test_exact_planned_word_counts_and_reproducibility(self):
        for topic in TOPICS:
            for size in (400, 2000):
                text = workload(topic, size)
                self.assertEqual(len(text.split()), size)
                self.assertEqual(text, workload(topic, size))
                self.assertEqual(text, text.strip())

    def test_p95_keeps_slow_tail(self):
        result = timings([1.0] * 14 + [12.0])
        self.assertEqual(result['count'], 15)
        self.assertEqual(result['p50Seconds'], 1.0)
        self.assertEqual(result['p95Seconds'], 12.0)


if __name__ == '__main__':
    unittest.main()
