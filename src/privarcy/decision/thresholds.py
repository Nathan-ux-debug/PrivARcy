"""Configurable confidence thresholds for the 3-tier routing gate."""

from dataclasses import dataclass


@dataclass
class DecisionThresholds:
    """T_high / T_low confidence gates.

    combined_confidence >= t_high              -> AUTO_REDACT
    t_low <= combined_confidence < t_high        -> FLAGGED (human review)
    combined_confidence < t_low                     -> PASS_THROUGH (ignored)

    Attributes:
        t_high: Auto-redact threshold. Default 0.85 — deliberately
                 conservative; only very confident detections skip human
                 review entirely.
        t_low:   Below this, a detection is dropped as noise rather than
                  sent to a reviewer. Default 0.50.
    """

    t_high: float = 0.85
    t_low: float = 0.50

    def __post_init__(self):
        if not (0.0 <= self.t_low <= self.t_high <= 1.0):
            raise ValueError(
                f"Thresholds must satisfy 0 <= t_low <= t_high <= 1, got t_low={self.t_low}, t_high={self.t_high}"
            )
