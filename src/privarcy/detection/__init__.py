"""
yolo_detection

Wraps an Ultralytics YOLOv8 model to run inference on NumPy image frames
(e.g. frames produced by the `frame_extraction` package) and return
clean, JSON-serializable detection results.

Public API:
    YOLODetector          - loads a YOLOv8 model, runs inference, parses results
    Detection               - one bounding box + class + confidence
    DetectionResult          - all detections for one frame, plus metadata
    resolve_device            - picks "cuda" if available, else "cpu"
    ModelLoadError             - raised when the model file/weights fail to load
    InferenceError               - raised when a forward pass fails on a given frame
"""

from .exceptions import ModelLoadError, InferenceError
from .models import Detection, DetectionResult
from .device import resolve_device
from .detector import YOLODetector

__all__ = [
    "YOLODetector",
    "Detection",
    "DetectionResult",
    "resolve_device",
    "ModelLoadError",
    "InferenceError",
]

__version__ = "1.0.0"
