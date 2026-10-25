"""Data models for OCR output."""

import json
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class OCRResult:
    """One recognized text string, linked back to where it came from.

    Attributes:
        text:               The recognized text string.
        confidence:          Rough confidence in [0, 1] — see engine.py for
                              how this is derived from TrOCR's generation scores.
        frame_id:              Identifier for the frame this crop came from
                                (e.g. FrameMeta.frame_index, or a composite
                                video+frame key) — links this result back
                                to a specific moment in the source video.
        bbox:                    (x_min, y_min, x_max, y_max) of the region
                                  this text was read from, in the ORIGINAL
                                  frame's pixel coordinates (not the crop's).
        source_class:               Optional: what the upstream detector
                                     (e.g. yolo_detection) thought this region
                                     was (e.g. "document", "credit_card") —
                                     carried through for the rule engine.
    """

    text: str
    confidence: float
    bbox: Tuple[float, float, float, float]
    frame_id: Optional[str] = None
    source_class: Optional[str] = None

    def to_dict(self) -> dict:
        x_min, y_min, x_max, y_max = self.bbox
        return {
            "text": self.text,
            "confidence": round(float(self.confidence), 4),
            "frame_id": self.frame_id,
            "source_class": self.source_class,
            "bbox": {
                "x_min": round(float(x_min), 2), "y_min": round(float(y_min), 2),
                "x_max": round(float(x_max), 2), "y_max": round(float(y_max), 2),
            },
        }

    def to_json(self, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
