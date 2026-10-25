"""Batch helper: run FrameExtractor over every video in a directory."""

import logging
from pathlib import Path

from .exceptions import VideoValidationError, FrameReadError
from .extractor import FrameExtractor

logger = logging.getLogger(__name__)


def process_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    target_fps: float = 1.0,
    **extractor_kwargs,
) -> dict:
    """Run FrameExtractor over every supported video file in input_dir.

    Each video gets its own subfolder under output_dir, named after its
    filename stem. A video that fails validation is skipped (logged) and
    processing continues with the rest of the batch.

    Returns:
        dict mapping {video_filename: sample_count_or_error_string}.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    summary: dict = {}

    video_files = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in FrameExtractor.SUPPORTED_EXTENSIONS
    )

    if not video_files:
        logger.warning("No supported video files found in %s", input_dir)
        return summary

    for video_path in video_files:
        video_output_dir = output_dir / video_path.stem
        try:
            extractor = FrameExtractor(
                video_path=video_path,
                output_dir=video_output_dir,
                target_fps=target_fps,
                **extractor_kwargs,
            )
            count = sum(1 for _ in extractor.extract())
            summary[video_path.name] = count
        except (VideoValidationError, FrameReadError) as exc:
            logger.error("Skipping '%s': %s", video_path.name, exc)
            summary[video_path.name] = f"ERROR: {exc}"

    return summary
