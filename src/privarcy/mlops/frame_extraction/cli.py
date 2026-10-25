"""Command-line entry point for the frame_extraction package.

Examples:
    python -m frame_extraction.cli --video uploads/clip.mp4 --output-dir frames/clip --fps 2
    python -m frame_extraction.cli --input-dir uploads --output-dir frames --fps 2
"""

import argparse
import logging
import sys

from .exceptions import VideoValidationError, FrameReadError
from .extractor import FrameExtractor
from .batch import process_directory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract frames from MP4 video(s) at a configurable sampling FPS."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=str, help="Path to a single video file.")
    source.add_argument("--input-dir", type=str, help="Directory of videos to batch process.")

    parser.add_argument("--output-dir", type=str, required=True, help="Where to write extracted frames.")
    parser.add_argument("--fps", type=float, default=1.0, help="Target sampling rate (frames/sec). Default: 1.0")
    parser.add_argument("--format", type=str, default="jpg", choices=["jpg", "jpeg", "png"], help="Output image format.")
    parser.add_argument("--jpeg-quality", type=int, default=95, help="JPEG quality 0-100 (jpg format only).")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional cap on sampled frames per video.")
    parser.add_argument("--strict", action="store_true", help="Abort on first corrupted/unreadable frame.")
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()

    extractor_kwargs = dict(
        image_format=args.format,
        jpeg_quality=args.jpeg_quality,
        max_frames=args.max_frames,
        strict=args.strict,
    )

    try:
        if args.video:
            extractor = FrameExtractor(
                video_path=args.video,
                output_dir=args.output_dir,
                target_fps=args.fps,
                **extractor_kwargs,
            )
            count = sum(1 for _ in extractor.extract())
            logger.info("Done. %d frame(s) written to %s", count, args.output_dir)
        else:
            summary = process_directory(
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                target_fps=args.fps,
                **extractor_kwargs,
            )
            logger.info("Batch summary: %s", summary)
    except (VideoValidationError, FrameReadError) as exc:
        logger.error("Extraction failed: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
