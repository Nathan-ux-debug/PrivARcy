"""Data models for detection output.

Kept separate from the detector so the output *contract* (what a
"detection" looks like) is easy to read on its own, and so downstream
consumers (an API layer, a UI, a redaction engine) can import just this
module without pulling in Ultralytics/PyTorch.
"""

import json
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class Detection:
    """A single detected object.

    Attributes:
        class_id:    Integer class index as reported by the model.
        class_name:  Human-readable label (e.g. "person", "car").
        confidence:  Detection confidence score, 0.0-1.0.
        bbox:        (x_min, y_min, x_max, y_max) in pixel coordinates,
                      relative to the original input frame.
    """

    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]

    def to_dict(self) -> dict:
        x_min, y_min, x_max, y_max = self.bbox
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(float(self.confidence), 4),
            "bbox": {
                "x_min": round(float(x_min), 2),
                "y_min": round(float(y_min), 2),
                "x_max": round(float(x_max), 2),
                "y_max": round(float(y_max), 2),
            },
        }


@dataclass
class DetectionResult:
    """All detections produced for a single frame, plus run metadata.

    Attributes:
        detections:        List of Detection objects (already
                            threshold-filtered).
        model_name:         Identifier of the model/weights used
                             (e.g. "yolov8n.pt").
        image_shape:        (height, width, channels) of the input frame.
        inference_time_ms:  Wall-clock time for the forward pass, in ms.
        device:             "cuda" or "cpu" — where inference actually ran.
        source_frame_index: Optional index tying this result back to a
                             frame from an upstream extraction pipeline
                             (e.g. FrameMeta.frame_index).
        source_video:       Optional path to the originating video file.
    """

    detections: List[Detection]
    model_name: str
    image_shape: Tuple[int, int, int]
    inference_time_ms: float
    device: str
    source_frame_index: Optional[int] = None
    source_video: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "model": self.model_name,
            "device": self.device,
            "image_shape": {
                "height": self.image_shape[0],
                "width": self.image_shape[1],
                "channels": self.image_shape[2] if len(self.image_shape) > 2 else 1,
            },
            "inference_time_ms": round(self.inference_time_ms, 2),
            "source_frame_index": self.source_frame_index,
            "source_video": self.source_video,
            "num_detections": len(self.detections),
            "detections": [d.to_dict() for d in self.detections],
        }

    def to_json(self, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
