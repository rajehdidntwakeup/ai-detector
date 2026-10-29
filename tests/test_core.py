import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from detector import Detector, ai_portion, leaning, passage_spans, validate_text


class TextContractTests(unittest.TestCase):
    def test_baseline_coverage_reports_document_tokens_across_windows(self):
        from unittest.mock import Mock
        detector = Detector.__new__(Detector)
        detector.tokenizer = Mock()
        detector.tokenizer.encode.return_value = [7] * 1021
        detector.tokenizer.num_special_tokens_to_add.return_value = 2
        detector.max_tokens = 512
        detector.score_texts = lambda texts: [0.5] * len(texts)
        result = detector.analyze("An example document. Another sentence.")
        self.assertEqual(result["coverage"], {"textTokensAnalyzed": 1021, "textTokensTotal": 1021,
                                             "documentWindows": 3, "truncated": False})
        self.assertTrue(result["passageScores"])
        self.assertFalse(result["scoreSemantics"]["mixedClass"])
        self.assertFalse(result["scoreSemantics"]["calibratedAuthorshipProbability"])
        self.assertIsNone(result["scoreSemantics"]["referenceMapping"])

    def test_limits_are_explicit_and_never_silently_truncate(self):
        for invalid in (None, 123, {}, "", " \n ", "1234567890!!!", "abcdefghi"):
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                validate_text(invalid)
        self.assertEqual(validate_text("  abcdefghij  "), "abcdefghij")
        text = "word " * 2000
        self.assertEqual(len(validate_text(text).split()), 2000)
        with self.assertRaisesRegex(ValueError, "2,000"):
            validate_text(text + "extra")
        with self.assertRaisesRegex(ValueError, "50,000"):
            validate_text("a" * 50001)

    def test_offsets_preserve_unicode_spacing_and_duplicate_sentences(self):
        text = "Dr. Smith smiled. 😀 Same sentence!\n\nSame sentence!  A final bit without punctuation"
        spans = passage_spans(text)
        self.assertEqual(text[spans[0].start:spans[0].end], "Dr. Smith smiled.")
        segments = Detector._segments(text, spans, [0.4] * len(spans))
        self.assertEqual("".join(segment["text"] for segment in segments), text)
        self.assertEqual([segment["passageIndex"] for segment in segments if "passageIndex" in segment], list(range(len(spans))))

    def test_highlight_work_is_bounded_without_losing_text(self):
        text = "A sentence. " * 600
        spans = passage_spans(text)
        self.assertLessEqual(len(spans), 80)
        segments = Detector._segments(text, spans, [0.2] * len(spans))
        self.assertEqual("".join(segment["text"] for segment in segments), text)

    def test_uncertain_range_is_not_a_third_probability(self):
        self.assertEqual(leaning(0.35), "human")
        self.assertEqual(leaning(0.35001), "uncertain")
        self.assertEqual(leaning(0.64999), "uncertain")
        self.assertEqual(leaning(0.65), "ai")

    def test_ai_portion_counts_flagged_words_not_score_strength(self):
        passages = ["one two three four", "five six", "seven"]
        # 4 of 7 words sit in AI-leaning passages; human passages do not dilute the share.
        self.assertAlmostEqual(ai_portion(passages, [0.9, 0.01, 0.99]), 5 / 7)
        self.assertAlmostEqual(ai_portion(passages, [0.01, 0.02, 0.03]), 0.0)
        self.assertAlmostEqual(ai_portion(passages, [0.7, 0.8, 0.9]), 1.0)
        self.assertEqual(ai_portion([], []), 0.0)


if __name__ == "__main__":
    unittest.main()
