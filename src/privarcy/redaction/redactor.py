"""Core redaction logic: FrameRedactor.

Deliberately decoupled from yolo_detection's exact Detection class — this
module accepts anything with .class_name, .confidence, and .bbox
attributes (a Python "duck typing" Protocol), so redaction doesn't need
a hard import-time dependency on yolo_detection and could just as easily
be fed detections from a different detector later.
"""

import logging
from typing import Iterable, Protocol, Tuple

import cv2
import numpy as np

from .config import RedactionConfig
from .exceptions import RedactionError

logger = logging.getLogger(__name__)


class DetectionLike(Protocol):
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]


class FrameRedactor:
    """Applies blur/pixelate/solid redaction to the regions given by a
    list of detections (e.g. from yolo_detection.YOLODetector.predict()).

    Parameters:
        config: A RedactionConfig controlling which classes get redacted
                and how. Defaults to redacting every detection with a
                Gaussian blur.
    """

    def __init__(self, config: RedactionConfig | None = None) -> None:
        self.config = config or RedactionConfig()

    def redact(self, frame: np.ndarray, detections: Iterable[DetectionLike]) -> np.ndarray:
        """Return a NEW frame with the matching detection regions redacted.

        The input frame is never modified in place — callers that still
        need the original (e.g. to save both a raw and a redacted copy)
        don't have to make a defensive copy themselves.

        Args:
            frame: BGR NumPy array (H, W, 3).
            detections: Detection objects (or anything duck-typed the
                        same way) to consider redacting.

        Raises:
            RedactionError: if frame is invalid.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise RedactionError("redact() received an empty or invalid frame array.")

        output = frame.copy()
        height, width = output.shape[:2]
        redacted_count = 0

        for det in detections:
            if self.config.classes_to_redact is not None and det.class_name not in self.config.classes_to_redact:
                continue
            if det.confidence < self.config.min_confidence:
                continue

            x1, y1, x2, y2 = self._clamp_box(det.bbox, width, height)
            if x2 <= x1 or y2 <= y1:
                continue  # degenerate box after clamping — nothing to redact

            region = output[y1:y2, x1:x2]
            output[y1:y2, x1:x2] = self._apply_method(region)
            redacted_count += 1

        if redacted_count:
            logger.debug("Redacted %d region(s) using method='%s'.", redacted_count, self.config.method)

        return output

    def _clamp_box(self, bbox: Tuple[float, float, float, float], width: int, height: int) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = bbox
        pad = self.config.padding_px
        x1 = max(0, int(round(x1)) - pad)
        y1 = max(0, int(round(y1)) - pad)
        x2 = min(width, int(round(x2)) + pad)
        y2 = min(height, int(round(y2)) + pad)
        return x1, y1, x2, y2

    def _apply_method(self, region: np.ndarray) -> np.ndarray:
        h, w = region.shape[:2]
        if h == 0 or w == 0:
            return region

        if self.config.method == "gaussian":
            # Kernel can't exceed the region's own size, and must stay odd.
            k = min(self.config.blur_strength, h if h % 2 == 1 else h - 1, w if w % 2 == 1 else w - 1)
            k = max(1, k)
            if k % 2 == 0:
                k -= 1
            k = max(1, k)
            return cv2.GaussianBlur(region, (k, k), 0)

        if self.config.method == "pixelate":
            blocks = max(1, self.config.pixelate_blocks)
            small_w = max(1, w // blocks)
            small_h = max(1, h // blocks)
            small = cv2.resize(region, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
            return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

        if self.config.method == "solid":
            return np.zeros_like(region)

        raise RedactionError(f"Unknown redaction method: {self.config.method}")  # unreachable, validated in config
