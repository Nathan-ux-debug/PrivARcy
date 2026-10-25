"""Data models for the decision_engine package."""

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple


class DecisionTier(str, Enum):
    AUTO_REDACT = "auto_redact"
    FLAGGED = "flagged"
    PASS_THROUGH = "pass_through"


@dataclass
class DecisionInput:
    """Everything the rule engine needs to know about one detected region,
    gathered from the upstream pipeline stages.

    Every field except frame_id/bbox is optional — not every detection
    will have gone through OCR or face matching, and the engine only
    uses the signals that are actually present.

    Attributes:
        frame_id:              Ties this decision back to a specific frame.
        bbox:                    (x_min, y_min, x_max, y_max) of the region.
        detected_class:            yolo_detection's class label, e.g.
                                    "credit_card", "document", "face", "car".
        detection_confidence:         yolo_detection's confidence for that class.
        ocr_text:                       Recognized text from `ocr`, if OCR ran
                                         on this region.
        ocr_confidence:                    `ocr`'s confidence for that text.
        is_known_face:                        For detected_class == "face" only:
                                               True if face_privacy matched this
                                               face to an enrolled person, False
                                               if it didn't, None if face matching
                                               wasn't run on this region at all.
        face_similarity:                         face_privacy's similarity score,
                                                  if is_known_face is not None.
    """

    frame_id: str
    bbox: Tuple[float, float, float, float]
    detected_class: Optional[str] = None
    detection_confidence: float = 0.0
    ocr_text: Optional[str] = None
    ocr_confidence: float = 0.0
    is_known_face: Optional[bool] = None
    face_similarity: float = 0.0


@dataclass
class Decision:
    """The routing outcome for one DecisionInput.

    Attributes:
        tier:                  Which of the 3 tiers this landed in.
        combined_confidence:      The score that was actually compared
                                    against the thresholds.
        reasons:                      Human-readable explanation of every
                                       signal that contributed — for the
                                       reviewer UI and for debugging why
                                       something was (or wasn't) flagged.
        input:                            The DecisionInput this decision
                                          was made for.
    """

    tier: DecisionTier
    combined_confidence: float
    reasons: List[str] = field(default_factory=list)
    input: Optional[DecisionInput] = None

    def to_dict(self) -> dict:
        return {
            "tier": self.tier.value,
            "combined_confidence": round(self.combined_confidence, 4),
            "reasons": self.reasons,
            "frame_id": self.input.frame_id if self.input else None,
            "bbox": list(self.input.bbox) if self.input else None,
        }

    def to_json(self, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
