"""Command-line entry point for the redaction package.

Two modes:
  1. --video : batch mode — read a saved video file end to end, detect +
               redact every frame, write a redacted .mp4 out.
  2. --live  : real-time mode — read from a webcam or stream (RTSP/HTTP),
               detect + redact continuously, and display and/or record it.

Examples:
    # Redact a saved video file
    python -m redaction.cli --video uploads/clip.mp4 --output redacted.mp4 \\
        --classes credit_card,document --method pixelate

    # Redact a webcam feed live, on screen only (no recording)
    python -m redaction.cli --live 0 --classes person,credit_card --method gaussian

    # Redact a live feed AND save it
    python -m redaction.cli --live 0 --output redacted_live.mp4 --classes credit_card
"""

import argparse
import logging
import sys
import time

from .config import RedactionConfig
from .exceptions import RedactionError
from .redactor import FrameRedactor
from .video_writer import VideoWriter
from .live_stream import ThreadedVideoCapture

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _load_detector(model_path: str, device: str | None, conf: float):
    """Local import so redaction never hard-depends on yolo_detection
    unless you actually run the CLI — the library code (redactor.py,
    video_writer.py) works with detections from any source."""
    try:
        from yolo_detection import YOLODetector
    except ImportError as exc:
        raise RuntimeError(
            "The 'yolo_detection' package is required to run this CLI "
            "(it supplies the detections that get redacted)."
        ) from exc
    return YOLODetector(model_path=model_path, device=device, confidence_threshold=conf)


def _run_on_video(detector, redactor: FrameRedactor, video_path: str, output_path: str, fps: float | None) -> dict:
    try:
        from frame_extraction import FrameExtractor
    except ImportError as exc:
        raise RuntimeError(
            "The 'frame_extraction' package is required for --video mode."
        ) from exc

    # fps=None means "keep every native frame" — target_fps is clamped to
    # the source rate inside FrameExtractor, so a very high number achieves that.
    extractor = FrameExtractor(video_path=video_path, output_dir=None, target_fps=fps or 1_000_000)
    output_fps = extractor.source_fps / extractor.frame_interval

    writer = VideoWriter(output_path, fps=output_fps)
    total_redacted = 0
    frame_count = 0

    try:
        for frame_meta in extractor.extract():
            result = detector.predict(frame_meta.frame)
            redacted_frame = redactor.redact(frame_meta.frame, result.detections)
            writer.write_frame(redacted_frame)
            frame_count += 1
            total_redacted += sum(
                1 for d in result.detections
                if redactor.config.classes_to_redact is None or d.class_name in redactor.config.classes_to_redact
            )
    finally:
        writer.release()

    return {"frames_written": frame_count, "regions_redacted": total_redacted, "output": output_path}


def _run_live(detector, redactor: FrameRedactor, source, output_path: str | None, display: bool, max_seconds: float | None) -> dict:
    import cv2  # only needed for imshow/waitKey in live mode

    capture = ThreadedVideoCapture(source).start()
    writer: VideoWriter | None = None
    frame_count = 0
    start_time = time.time()

    logger.info("Live redaction started on source=%r. Press 'q' in the preview window to stop.", source)

    try:
        while True:
            frame = capture.read()
            if frame is None:
                time.sleep(0.01)  # capture thread hasn't produced a first frame yet
                continue

            result = detector.predict(frame)
            redacted_frame = redactor.redact(frame, result.detections)

            if output_path:
                if writer is None:
                    writer = VideoWriter(output_path, fps=30)  # display-driven fps estimate for live capture
                writer.write_frame(redacted_frame)

            if display:
                cv2.imshow("redaction (press 'q' to quit)", redacted_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_count += 1
            if max_seconds and (time.time() - start_time) >= max_seconds:
                logger.info("Reached --max-seconds=%s, stopping.", max_seconds)
                break
    finally:
        capture.stop()
        if writer is not None:
            writer.release()
        if display:
            cv2.destroyAllWindows()

    elapsed = time.time() - start_time
    return {
        "frames_processed": frame_count,
        "elapsed_sec": round(elapsed, 2),
        "avg_fps": round(frame_count / elapsed, 2) if elapsed > 0 else 0,
        "output": output_path,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Redact (blur/pixelate/black-box) detected regions in a video.")

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=str, help="Path to a saved video file to redact (batch mode).")
    source.add_argument("--live", type=str, help="Camera index (e.g. '0') or a stream URL (RTSP/HTTP) for real-time mode.")

    parser.add_argument("--output", type=str, default=None,
                         help="Output .mp4 path. Required for --video; optional for --live (omit to just preview).")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLOv8 weights path or name.")
    parser.add_argument("--device", type=str, default=None, help="'cuda', 'cpu', or omit to auto-detect.")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold.")
    parser.add_argument("--classes", type=str, default=None,
                         help="Comma-separated class names to redact. Omit to redact every detection.")
    parser.add_argument("--method", type=str, default="gaussian", choices=["gaussian", "pixelate", "solid"])
    parser.add_argument("--blur-strength", type=int, default=51, help="Gaussian kernel size (odd number).")
    parser.add_argument("--pixelate-blocks", type=int, default=10)
    parser.add_argument("--padding", type=int, default=4, help="Extra pixels around each box before redacting.")
    parser.add_argument("--fps", type=float, default=None,
                         help="[--video only] Sampling FPS. Omit to process every native frame.")
    parser.add_argument("--no-display", action="store_true", help="[--live only] Don't open a preview window.")
    parser.add_argument("--max-seconds", type=float, default=None, help="[--live only] Auto-stop after N seconds.")
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()

    if args.video and not args.output:
        logger.error("--output is required for --video mode.")
        return 1

    classes_to_redact = set(c.strip() for c in args.classes.split(",")) if args.classes else None
    config = RedactionConfig(
        classes_to_redact=classes_to_redact,
        method=args.method,
        blur_strength=args.blur_strength,
        pixelate_blocks=args.pixelate_blocks,
        padding_px=args.padding,
    )
    redactor = FrameRedactor(config)

    try:
        detector = _load_detector(args.model, args.device, args.conf)
    except RuntimeError as exc:
        logger.error(str(exc))
        return 1

    try:
        if args.video:
            summary = _run_on_video(detector, redactor, args.video, args.output, args.fps)
        else:
            live_source: int | str = int(args.live) if args.live.isdigit() else args.live
            summary = _run_live(detector, redactor, live_source, args.output, not args.no_display, args.max_seconds)
    except (RedactionError, RuntimeError) as exc:
        logger.error("Redaction failed: %s", exc)
        return 1

    logger.info("Done: %s", summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
