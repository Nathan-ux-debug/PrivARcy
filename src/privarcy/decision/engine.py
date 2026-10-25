"""Core rule engine: DecisionEngine.

Combines yolo_detection's class/confidence, ocr's PII-checked text, and
face_privacy's known/unknown match into one routing decision per
detected region.
"""

import logging
from typing import List, Optional, Set

from .models import Decision, DecisionInput, DecisionTier
from .rules import contains_pii
from .thresholds import DecisionThresholds

logger = logging.getLogger(__name__)

# Object classes treated as inherently sensitive by default — a detection
# of this class contributes its detection_confidence toward redaction,
# even with no OCR/face signal at all.
DEFAULT_SENSITIVE_CLASSES = {"credit_card", "id_card", "license_plate", "screen"}

# The class name(s) treated as "a face" for the purposes of the
# known/unknown face override below.
FACE_CLASSES = {"face"}


class DecisionEngine:
    """Routes each DecisionInput into AUTO_REDACT / FLAGGED / PASS_THROUGH.

    Parameters:
        thresholds: T_high / T_low gate. Defaults to DecisionThresholds().
        sensitive_classes: Object classes that count as sensitive on their
                             own (contributing detection_confidence toward
                             the combined score). Defaults to
                             DEFAULT_SENSITIVE_CLASSES.

    Combination rule: the combined confidence is the MAXIMUM of whichever
    signals actually apply to this input (class sensitivity, PII-in-text,
    unknown-face), not an average. This is a deliberately conservative
    choice for a privacy tool — if ANY single signal is highly confident
    this region is sensitive, a weak/absent signal elsewhere shouldn't
    dilute it down into "pass through."
    """

    def __init__(
        self,
        thresholds: Optional[DecisionThresholds] = None,
        sensitive_classes: Optional[Set[str]] = None,
    ) -> None:
        self.thresholds = thresholds or DecisionThresholds()
        self.sensitive_classes = sensitive_classes if sensitive_classes is not None else set(DEFAULT_SENSITIVE_CLASSES)

    def decide(self, decision_input: DecisionInput) -> Decision:
        """Route a single DecisionInput."""

        # Known-enrolled-face override: for a detection the pipeline
        # already identified as a face AND face_privacy matched it to an
        # enrolled person, always pass through — never auto-redact or
        # even flag someone who's explicitly allowlisted. This is what
        # lets face_privacy's "blur everyone except registered people"
        # behavior plug into this same generic engine.
        if decision_input.detected_class in FACE_CLASSES and decision_input.is_known_face is True:
            return Decision(
                tier=DecisionTier.PASS_THROUGH,
                combined_confidence=0.0,
                reasons=[f"face matched enrolled person (similarity {decision_input.face_similarity:.2f}) — excluded from redaction"],
                input=decision_input,
            )

        reasons: List[str] = []
        confidence_components: List[float] = []

        # Signal 1: object class sensitivity (yolo_detection)
        if decision_input.detected_class:
            if decision_input.detected_class in self.sensitive_classes:
                confidence_components.append(decision_input.detection_confidence)
                reasons.append(
                    f"class '{decision_input.detected_class}' is sensitive "
                    f"(detector confidence {decision_input.detection_confidence:.2f})"
                )
            else:
                reasons.append(f"class '{decision_input.detected_class}' is not in the sensitive class set")

        # Signal 2: unknown face (face_privacy) — only meaningful for face-class detections
        if decision_input.detected_class in FACE_CLASSES and decision_input.is_known_face is False:
            confidence_components.append(max(decision_input.detection_confidence, 1.0 - decision_input.face_similarity))
            reasons.append(f"face not recognized as enrolled (best similarity {decision_input.face_similarity:.2f})")

        # Signal 3: PII in OCR'd text
        if decision_input.ocr_text:
            has_pii, matched_patterns = contains_pii(decision_input.ocr_text)
            if has_pii:
                confidence_components.append(decision_input.ocr_confidence)
                reasons.append(f"extracted text matched PII pattern(s): {', '.join(matched_patterns)}")
            else:
                reasons.append("extracted text did not match any known PII pattern")

        combined_confidence = max(confidence_components) if confidence_components else 0.0
        if not reasons:
            reasons.append("no class, text, or face signals were provided for this detection")

        tier = self._route(combined_confidence)
        return Decision(tier=tier, combined_confidence=combined_confidence, reasons=reasons, input=decision_input)

    def decide_many(self, inputs: List[DecisionInput]) -> List[Decision]:
        """Convenience batch version of decide()."""
        return [self.decide(d) for d in inputs]

    def _route(self, combined_confidence: float) -> DecisionTier:
        if combined_confidence >= self.thresholds.t_high:
            return DecisionTier.AUTO_REDACT
        if combined_confidence >= self.thresholds.t_low:
            return DecisionTier.FLAGGED
        return DecisionTier.PASS_THROUGH

    @staticmethod
    def group_by_tier(decisions: List[Decision]) -> dict:
        """Split a list of Decisions into {tier_value: [Decision, ...]} —
        handy for handing AUTO_REDACT straight to redaction.FrameRedactor
        and FLAGGED to a human review queue."""
        grouped: dict = {tier.value: [] for tier in DecisionTier}
        for decision in decisions:
            grouped[decision.tier.value].append(decision)
        return grouped
