"""Video file validation utilities.

Kept separate from FrameExtractor so validation logic can be unit-tested,
reused (e.g. an upload endpoint validating a file before queuing a job),
or swapped out independently of the extraction/sampling logic.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2

from .exceptions import VideoValidationError

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
DEFAULT_FALLBACK_FPS = 25.0


@dataclass
class VideoInfo:
    """Basic metadata gathered while validating a video file."""

    path: Path
    source_fps: float
    total_frames: int
    duration_sec: float


def validate_video(video_path: Path) -> VideoInfo:
    """Validate that a video file exists, is non-empty, and is decodable.

    Performs three checks, from cheapest to most expensive:
      1. File exists and is non-empty on disk.
      2. OpenCV can open the container (cv2.VideoCapture.isOpened()).
      3. The first frame actually decodes (catches silently-corrupt files
         that "open" fine but contain no valid video data).

    Raises:
        VideoValidationError: if any check fails.

    Returns:
        VideoInfo with source_fps / total_frames / duration_sec populated.
    """
    if not video_path.exists():
        raise VideoValidationError(f"Video file not found: {video_path}")

    if video_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        logger.warning(
            "Extension '%s' is not in the expected set %s — attempting to open anyway.",
            video_path.suffix, SUPPORTED_EXTENSIONS,
        )

    if video_path.stat().st_size == 0:
        raise VideoValidationError(f"Video file is empty: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise VideoValidationError(
                f"OpenCV could not open video (corrupted/unsupported codec?): {video_path}"
            )

        source_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        if source_fps <= 0:
            logger.warning(
                "Source FPS reported as %.2f for '%s' — falling back to %.1f.",
                source_fps, video_path.name, DEFAULT_FALLBACK_FPS,
            )
            source_fps = DEFAULT_FALLBACK_FPS

        ok, frame = cap.read()
        if not ok or frame is None:
            raise VideoValidationError(
                f"Video opened but first frame could not be decoded: {video_path}"
            )

        duration_sec = total_frames / source_fps if total_frames else 0.0
    finally:
        cap.release()

    return VideoInfo(
        path=video_path,
        source_fps=source_fps,
        total_frames=total_frames,
        duration_sec=duration_sec,
    )
