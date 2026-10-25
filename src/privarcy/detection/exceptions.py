"""Custom exception types for the yolo_detection package."""


class ModelLoadError(Exception):
    """Raised when the YOLOv8 model weights fail to load (missing file,
    corrupted weights, incompatible Ultralytics version, etc.)."""


class InferenceError(Exception):
    """Raised when a forward pass fails on a given frame (bad shape,
    unreadable array, backend/runtime error)."""
