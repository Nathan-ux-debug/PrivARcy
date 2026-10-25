"""
frame_extraction

Modular MP4 -> sampled-frame ingestion pipeline built on OpenCV.
Designed as the pre-processing stage ahead of a YOLOv8 object-detection stage.

Public API:
    FrameExtractor        - core class: validates a video, yields FrameMeta objects
    FrameMeta              - dataclass describing one extracted frame
    process_directory      - batch helper for a folder of uploaded videos
    VideoValidationError   - raised for missing/corrupt/unopenable videos
    FrameReadError          - raised for a mid-stream frame decode failure (strict mode)
"""

from .exceptions import VideoValidationError, FrameReadError
from .models import FrameMeta
from .extractor import FrameExtractor
from .batch import process_directory

__all__ = [
    "FrameExtractor",
    "FrameMeta",
    "process_directory",
    "VideoValidationError",
    "FrameReadError",
]

__version__ = "1.0.0"
