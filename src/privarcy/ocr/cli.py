"""Command-line entry point for the ocr package.

Chains onto yolo_detection's JSON output: for each frame image + its
matching detections JSON, crops the regions of interest and runs TrOCR
on each, writing recognized text back out mapped to frame + bbox.

Example:
    python -m ocr.cli --frames-dir frames_out --detections-dir detections_out \\
        --output-dir ocr_out --classes document,credit_card --conf 0.3
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2

from .engine import TrOCREngine, DEFAULT_MODEL
from .exceptions import OCRError

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _run(engine: TrOCREngine, frames_dir: Path, detections_dir: Path, output_dir: Path,
          classes_filter, min_detection_confidence: float) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    detection_files = sorted(detections_dir.glob("*.json"))
    if not detection_files:
        logger.warning("No detection JSON files found in %s", detections_dir)
        return {}

    summary = {}
    for det_path in detection_files:
        stem = det_path.stem
        image_path = None
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = frames_dir / f"{stem}{ext}"
            if candidate.exists():
                image_path = candidate
                break
        if image_path is None:
            logger.warning("No matching image for %s, skipping.", det_path.name)
            continue

        frame = cv2.imread(str(image_path))
        if frame is None:
            logger.warning("Could not read image %s, skipping.", image_path)
            continue

        detection_data = json.loads(det_path.read_text())
        regions = []
        for det in detection_data.get("detections", []):
            if det.get("confidence", 0) < min_detection_confidence:
                continue
            if classes_filter and det.get("class_name") not in classes_filter:
                continue
            bbox = det["bbox"]
            regions.append({
                "bbox": (bbox["x_min"], bbox["y_min"], bbox["x_max"], bbox["y_max"]),
                "frame_id": detection_data.get("source_frame_index", stem),
                "source_class": det.get("class_name"),
            })

        results = engine.read_regions(frame, regions)
        out_path = output_dir / f"{stem}.json"
        out_path.write_text(json.dumps({"frame": stem, "ocr_results": [r.to_dict() for r in results]}, indent=2))
        summary[stem] = len(results)

    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run TrOCR on yolo_detection's cropped regions.")
    parser.add_argument("--frames-dir", type=str, required=True)
    parser.add_argument("--detections-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--classes", type=str, default=None,
                         help="Comma-separated class names to run OCR on. Omit to run on every detection.")
    parser.add_argument("--min-detection-confidence", type=float, default=0.0,
                         help="Skip detections below this confidence before cropping for OCR.")
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()
    classes_filter = set(c.strip() for c in args.classes.split(",")) if args.classes else None

    try:
        engine = TrOCREngine(model_name=args.model, device=args.device)
    except OCRError as exc:
        logger.error("Could not load TrOCR: %s", exc)
        return 1

    summary = _run(
        engine, Path(args.frames_dir), Path(args.detections_dir), Path(args.output_dir),
        classes_filter, args.min_detection_confidence,
    )
    logger.info("Done. Processed %d frame(s). Results in %s", len(summary), args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
