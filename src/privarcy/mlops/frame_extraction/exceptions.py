"""Custom exception types for the frame_extraction package."""


class VideoValidationError(Exception):
    """Raised when a video file cannot be opened, is missing, empty, or is corrupted."""


class FrameReadError(Exception):
    """Raised when a specific frame fails to decode mid-stream (strict mode only)."""
