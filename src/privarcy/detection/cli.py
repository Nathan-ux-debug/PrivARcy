"""Command-line entry point for the yolo_detection package.

Two modes:
  1. --frames-dir : run detection on every image already on disk
                     (e.g. the output of `frame_extraction`).
  2. --video       : end-to-end — extract frames from a video with
                     `frame_extraction`, then run detection on each,
                     without writing intermediate frame images to disk.

Examples:
    python -m yolo_detection.cli --frames-dir frames/clip --output-dir detections/clip
    python -m yolo_detection.cli --video uploads/clip.mp4 --fps 2 --output-dir detections/clip \
        --model yolov8n.pt --conf 0.35
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2

from .detector import YOLODetector, DEFAULT_MODEL
from .exceptions import ModelLoadError, InferenceError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def _run_on_frames_dir(detector: YOLODetector, frames_dir: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(p for p in frames_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)

    if not image_paths:
        logger.warning("No images found in %s", frames_dir)
        return {}

    summary = {}
    for path in image_paths:
        frame = cv2.imread(str(path))
        if frame is None:
            logger.warning("Could not read image, skipping: %s", path)
            summary[path.name] = "ERROR: unreadable image"
            continue
        try:
            result = detector.predict(frame, source_video=str(path))
        except InferenceError as exc:
            logger.warning("Inference failed on %s: %s", path.name, exc)
            summary[path.name] = f"ERROR: {exc}"
            continue

        out_path = output_dir / f"{path.stem}.json"
        out_path.write_text(result.to_json())
        summary[path.name] = result.to_dict()["num_detections"]

    return summary


def _run_on_video(detector: YOLODetector, video_path: str, fps: float, output_dir: Path) -> dict:
    try:
        from frame_extraction import FrameExtractor
    except ImportError as exc:
        raise RuntimeError(
            "End-to-end --video mode requires the 'frame_extraction' package "
            "to be installed / importable alongside yolo_detection."
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    extractor = FrameExtractor(video_path=video_path, output_dir=None, target_fps=fps)

    summary = {}
    for frame_meta in extractor.extract():
        try:
            result = detector.predict(
                frame_meta.frame,
                source_frame_index=frame_meta.frame_index,
                source_video=str(video_path),
            )
        except InferenceError as exc:
            logger.warning("Inference failed on frame %d: %s", frame_meta.frame_index, exc)
            summary[f"frame{frame_meta.sample_index:06d}"] = f"ERROR: {exc}"
            continue

        out_path = output_dir / f"frame{frame_meta.sample_index:06d}.json"
        out_path.write_text(result.to_json())
        summary[out_path.name] = result.to_dict()["num_detections"]

    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run YOLOv8 object detection on frames or a video and write JSON results."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--frames-dir", type=str, help="Directory of already-extracted frame images.")
    source.add_argument("--video", type=str, help="Video file to extract frames from and detect on end-to-end.")

    parser.add_argument("--fps", type=float, default=1.0, help="Sampling FPS, only used with --video.")
    parser.add_argument("--output-dir", type=str, required=True, help="Where to write per-frame JSON results.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="YOLOv8 weights path or name.")
    parser.add_argument("--device", type=str, default=None, help="'cuda', 'cpu', or omit to auto-detect.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (0-1). Default: 0.25")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold (0-1). Default: 0.45")
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()

    try:
        detector = YOLODetector(
            model_path=args.model,
            device=args.device,
            confidence_threshold=args.conf,
            iou_threshold=args.iou,
        )
    except ModelLoadError as exc:
        logger.error("Could not load model: %s", exc)
        return 1

    try:
        if args.frames_dir:
            summary = _run_on_frames_dir(detector, Path(args.frames_dir), Path(args.output_dir))
        else:
            summary = _run_on_video(detector, args.video, args.fps, Path(args.output_dir))
    except RuntimeError as exc:
        logger.error(str(exc))
        return 1

    total = sum(v for v in summary.values() if isinstance(v, int))
    logger.info("Done. %d detection(s) across %d file(s). Results in %s", total, len(summary), args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
