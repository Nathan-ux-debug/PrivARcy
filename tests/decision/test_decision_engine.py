"""
Unit tests for the decision_engine package.

Run with:
    python -m unittest decision_engine.tests.test_decision_engine -v
"""

import unittest

from privarcy.decision import (
    DecisionEngine,
    DecisionInput,
    DecisionThresholds,
    DecisionTier,
    contains_pii,
    find_sensitive_spans,
)


class TestDecisionThresholds(unittest.TestCase):
    def test_default_thresholds_valid(self):
        thresholds = DecisionThresholds()
        self.assertLess(thresholds.t_low, thresholds.t_high)

    def test_t_low_greater_than_t_high_raises(self):
        with self.assertRaises(ValueError):
            DecisionThresholds(t_high=0.5, t_low=0.8)

    def test_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            DecisionThresholds(t_high=1.5, t_low=0.5)
        with self.assertRaises(ValueError):
            DecisionThresholds(t_high=0.9, t_low=-0.1)


class TestPIIRules(unittest.TestCase):
    def test_credit_card_number_detected(self):
        found, patterns = contains_pii("4111 1111 1111 1111")
        self.assertTrue(found)
        self.assertIn("credit_card_number", patterns)

    def test_ssn_detected(self):
        found, patterns = contains_pii("SSN: 123-45-6789")
        self.assertTrue(found)
        self.assertIn("ssn", patterns)

    def test_email_detected(self):
        found, patterns = contains_pii("contact me at alice@example.com")
        self.assertTrue(found)
        self.assertIn("email", patterns)

    def test_plain_text_no_pii(self):
        found, patterns = contains_pii("just some random menu text, no numbers here")
        self.assertFalse(found)
        self.assertEqual(patterns, [])

    def test_empty_text_no_pii(self):
        found, patterns = contains_pii("")
        self.assertFalse(found)

    def test_labeled_personal_data_returns_source_spans(self):
        text = "Name: Ada Lovelace; Date of Birth: 12/31/1990"
        spans = find_sensitive_spans(text)
        self.assertEqual(
            {span["pattern"] for span in spans},
            {"name", "date_of_birth"},
        )
        for span in spans:
            self.assertEqual(text[span["start"]:span["end"]], span["text"])

    def test_generic_document_title_is_not_sensitive(self):
        self.assertFalse(contains_pii("Birth Certificate issued by City Office")[0])


class TestDecisionEngineHighConfidence(unittest.TestCase):
    """Definition of Done: unit tests pass for HIGH confidence inputs."""

    def setUp(self):
        self.engine = DecisionEngine(DecisionThresholds(t_high=0.85, t_low=0.5))

    def test_sensitive_class_high_detection_confidence_auto_redacts(self):
        decision_input = DecisionInput(
            frame_id="f1", bbox=(0, 0, 10, 10),
            detected_class="credit_card", detection_confidence=0.95,
        )
        decision = self.engine.decide(decision_input)
        self.assertEqual(decision.tier, DecisionTier.AUTO_REDACT)
        self.assertGreaterEqual(decision.combined_confidence, 0.85)

    def test_pii_text_high_ocr_confidence_auto_redacts(self):
        decision_input = DecisionInput(
            frame_id="f1", bbox=(0, 0, 10, 10),
            detected_class="document", detection_confidence=0.3,  # class signal alone is low
            ocr_text="4111 1111 1111 1111", ocr_confidence=0.92,     # but PII text signal is high
        )
        decision = self.engine.decide(decision_input)
        self.assertEqual(decision.tier, DecisionTier.AUTO_REDACT)  # max() of the two signals wins

    def test_unknown_face_high_confidence_auto_redacts(self):
        decision_input = DecisionInput(
            frame_id="f1", bbox=(0, 0, 10, 10),
            detected_class="face", detection_confidence=0.9, is_known_face=False, face_similarity=0.1,
        )
        decision = self.engine.decide(decision_input)
        self.assertEqual(decision.tier, DecisionTier.AUTO_REDACT)


class TestDecisionEngineMediumConfidence(unittest.TestCase):
    """Definition of Done: unit tests pass for MEDIUM confidence inputs."""

    def setUp(self):
        self.engine = DecisionEngine(DecisionThresholds(t_high=0.85, t_low=0.5))

    def test_sensitive_class_medium_confidence_flags_for_review(self):
        decision_input = DecisionInput(
            frame_id="f1", bbox=(0, 0, 10, 10),
            detected_class="id_card", detection_confidence=0.65,
        )
        decision = self.engine.decide(decision_input)
        self.assertEqual(decision.tier, DecisionTier.FLAGGED)

    def test_document_class_alone_does_not_trigger_redaction(self):
        decision_input = DecisionInput(
            frame_id="f1", bbox=(0, 0, 10, 10),
            detected_class="document", detection_confidence=0.99,
        )
        decision = self.engine.decide(decision_input)
        self.assertEqual(decision.tier, DecisionTier.PASS_THROUGH)

    def test_boundary_at_t_low_is_flagged_inclusive(self):
        decision_input = DecisionInput(
            frame_id="f1", bbox=(0, 0, 10, 10),
            detected_class="credit_card", detection_confidence=0.50,  # exactly t_low
        )
        decision = self.engine.decide(decision_input)
        self.assertEqual(decision.tier, DecisionTier.FLAGGED)

    def test_boundary_just_below_t_high_is_flagged(self):
        decision_input = DecisionInput(
            frame_id="f1", bbox=(0, 0, 10, 10),
            detected_class="credit_card", detection_confidence=0.849,
        )
        decision = self.engine.decide(decision_input)
        self.assertEqual(decision.tier, DecisionTier.FLAGGED)


class TestDecisionEngineLowConfidence(unittest.TestCase):
    """Definition of Done: unit tests pass for LOW confidence inputs."""

    def setUp(self):
        self.engine = DecisionEngine(DecisionThresholds(t_high=0.85, t_low=0.5))

    def test_sensitive_class_low_confidence_passes_through(self):
        decision_input = DecisionInput(
            frame_id="f1", bbox=(0, 0, 10, 10),
            detected_class="document", detection_confidence=0.2,
        )
        decision = self.engine.decide(decision_input)
        self.assertEqual(decision.tier, DecisionTier.PASS_THROUGH)

    def test_non_sensitive_class_passes_through_regardless_of_confidence(self):
        decision_input = DecisionInput(
            frame_id="f1", bbox=(0, 0, 10, 10),
            detected_class="car", detection_confidence=0.99,  # high confidence, but not a sensitive class
        )
        decision = self.engine.decide(decision_input)
        self.assertEqual(decision.tier, DecisionTier.PASS_THROUGH)

    def test_no_signals_at_all_passes_through(self):
        decision_input = DecisionInput(frame_id="f1", bbox=(0, 0, 10, 10))
        decision = self.engine.decide(decision_input)
        self.assertEqual(decision.tier, DecisionTier.PASS_THROUGH)
        self.assertEqual(decision.combined_confidence, 0.0)

    def test_boundary_just_below_t_low_passes_through(self):
        decision_input = DecisionInput(
            frame_id="f1", bbox=(0, 0, 10, 10),
            detected_class="credit_card", detection_confidence=0.499,
        )
        decision = self.engine.decide(decision_input)
        self.assertEqual(decision.tier, DecisionTier.PASS_THROUGH)


class TestKnownFaceOverride(unittest.TestCase):
    def setUp(self):
        self.engine = DecisionEngine(DecisionThresholds(t_high=0.85, t_low=0.5))

    def test_known_face_always_passes_through(self):
        decision_input = DecisionInput(
            frame_id="f1", bbox=(0, 0, 10, 10),
            detected_class="face", detection_confidence=0.99, is_known_face=True, face_similarity=0.95,
        )
        decision = self.engine.decide(decision_input)
        self.assertEqual(decision.tier, DecisionTier.PASS_THROUGH)
        self.assertIn("enrolled person", decision.reasons[0])

    def test_known_face_override_beats_high_confidence(self):
        # Even with maximal detection confidence, an enrolled match must win.
        decision_input = DecisionInput(
            frame_id="f1", bbox=(0, 0, 10, 10),
            detected_class="face", detection_confidence=1.0, is_known_face=True, face_similarity=1.0,
        )
        decision = self.engine.decide(decision_input)
        self.assertEqual(decision.tier, DecisionTier.PASS_THROUGH)


class TestDecisionEngineBatchAndGrouping(unittest.TestCase):
    def test_decide_many_and_group_by_tier(self):
        engine = DecisionEngine(DecisionThresholds(t_high=0.85, t_low=0.5))
        inputs = [
            DecisionInput(frame_id="f1", bbox=(0, 0, 1, 1), detected_class="credit_card", detection_confidence=0.95),
            DecisionInput(frame_id="f2", bbox=(0, 0, 1, 1), detected_class="id_card", detection_confidence=0.6),
            DecisionInput(frame_id="f3", bbox=(0, 0, 1, 1), detected_class="car", detection_confidence=0.99),
        ]
        decisions = engine.decide_many(inputs)
        grouped = DecisionEngine.group_by_tier(decisions)

        self.assertEqual(len(grouped["auto_redact"]), 1)
        self.assertEqual(len(grouped["flagged"]), 1)
        self.assertEqual(len(grouped["pass_through"]), 1)

    def test_custom_sensitive_classes(self):
        engine = DecisionEngine(sensitive_classes={"license_plate"})
        decision_input = DecisionInput(
            frame_id="f1", bbox=(0, 0, 1, 1), detected_class="credit_card", detection_confidence=0.99,
        )
        # "credit_card" isn't in this engine's custom sensitive set -> should pass through.
        decision = engine.decide(decision_input)
        self.assertEqual(decision.tier, DecisionTier.PASS_THROUGH)


class TestSerialization(unittest.TestCase):
    def test_to_dict_and_to_json(self):
        import json
        engine = DecisionEngine()
        decision_input = DecisionInput(
            frame_id="clip1_042", bbox=(10, 20, 100, 200),
            detected_class="credit_card", detection_confidence=0.95,
        )
        decision = engine.decide(decision_input)
        payload = decision.to_dict()
        self.assertEqual(payload["tier"], "auto_redact")
        self.assertEqual(payload["frame_id"], "clip1_042")
        self.assertEqual(payload["bbox"], [10, 20, 100, 200])
        self.assertEqual(json.loads(decision.to_json()), payload)


if __name__ == "__main__":
    unittest.main()
