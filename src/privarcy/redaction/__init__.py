"""
redaction

Takes a frame plus the detections yolo_detection found in it, and paints
over the sensitive regions (blur, pixelate, or solid block) — the actual
"redact this" step referenced in the original tickets.

Pipeline position (after Tickets 1-3):
    frame_extraction -> yolo_detection -> redaction (this package) -> output frames / video

Public API:
    FrameRedactor       - applies blur/pixelate/solid redaction to one frame given detections
    RedactionConfig       - which classes to redact, method, strength, padding
    VideoWriter             - thin wrapper for writing a sequence of frames back out as an MP4
    RedactionError             - raised for invalid config or a write failure
"""

from .exceptions import RedactionError
from .config import RedactionConfig
from .redactor import FrameRedactor
from .video_writer import VideoWriter
from .live_stream import ThreadedVideoCapture

__all__ = [
    "FrameRedactor",
    "RedactionConfig",
    "VideoWriter",
    "ThreadedVideoCapture",
    "RedactionError",
]

__version__ = "1.0.0"
