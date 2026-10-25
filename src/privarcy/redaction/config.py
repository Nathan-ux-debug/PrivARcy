"""Configuration for what/how to redact."""

from dataclasses import dataclass
from typing import Optional, Set

from .exceptions import RedactionError

VALID_METHODS = {"gaussian", "pixelate", "solid"}


@dataclass
class RedactionConfig:
    """Controls which detections get redacted, and how.

    Attributes:
        classes_to_redact: Set of class names to redact (e.g. {"credit_card",
                            "document"}). None means "redact every detection
                            regardless of class" — the default, since the
                            baseline model's job here is just "cover
                            whatever was flagged."
        method:              "gaussian" (blurred, still shows rough shape/color),
                              "pixelate" (blocky, common for ID/face redaction),
                              or "solid" (opaque black box — strongest guarantee
                              nothing is recoverable).
        blur_strength:        Gaussian kernel size (odd number, larger = blurrier).
                               Only used when method="gaussian".
        pixelate_blocks:        Number of blocks across the shorter side of the
                                 region when method="pixelate". Smaller = blockier.
        padding_px:               Extra pixels added around each box before
                                   redacting, so the edge of the object isn't
                                   left sharp/visible right at the box boundary.
        min_confidence:             Extra confidence floor applied here, on top of
                                     whatever threshold yolo_detection already used —
                                     lets you redact more conservatively (higher bar)
                                     than what you kept for the raw JSON output.
    """

    classes_to_redact: Optional[Set[str]] = None
    method: str = "gaussian"
    blur_strength: int = 51
    pixelate_blocks: int = 10
    padding_px: int = 4
    min_confidence: float = 0.0

    def __post_init__(self):
        if self.method not in VALID_METHODS:
            raise RedactionError(f"method must be one of {VALID_METHODS}, got '{self.method}'")
        if self.blur_strength < 1 or self.blur_strength % 2 == 0:
            raise RedactionError(f"blur_strength must be a positive odd number, got {self.blur_strength}")
        if self.pixelate_blocks < 1:
            raise RedactionError(f"pixelate_blocks must be >= 1, got {self.pixelate_blocks}")
        if self.padding_px < 0:
            raise RedactionError(f"padding_px must be >= 0, got {self.padding_px}")
        if not (0.0 <= self.min_confidence <= 1.0):
            raise RedactionError(f"min_confidence must be within [0,1], got {self.min_confidence}")
