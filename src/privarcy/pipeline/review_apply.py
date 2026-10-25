"""Apply a reviewer's CONFIRM to an already-written output video."""

from __future__ import annotations

import os
from pathlib import Path

import cv2

from .redact_ops import redact_region


def apply_review_redactions(item: dict) -> int:
    """Re-open ``item['output_path']`` and redact every observed box of the
    confirmed detection track. Returns the number of frames modified.
    The video is replaced atomically only if every target frame was reached."""
    output_path = Path(item["output_path"])
    if not output_path.is_file():
        raise FileNotFoundError(f"Processed video not found: {output_path}")

    observations = item.get("track_observations") or [{"frame": item["frame"], "bbox": item["bbox"]}]
    boxes_by_frame: dict = {}
    for obs in observations:
        box = obs.get("redact_bbox") or obs["bbox"]
        boxes_by_frame.setdefault(int(obs["frame"]) - 1, []).append(box)

    capture = cv2.VideoCapture(str(output_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open processed video: {output_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tmp = output_path.with_name(f"{output_path.stem}.review{output_path.suffix}")
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        writer.release()
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Could not create reviewed video: {tmp}")

    applied, index = set(), 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            for box in boxes_by_frame.get(index, []):
                x1, y1, x2, y2 = (int(v) for v in box)
                frame = redact_region(frame, x1, y1, x2, y2, item.get("redaction_method", "Blur"))
                applied.add(index)
            writer.write(frame)
            index += 1
    finally:
        capture.release()
        writer.release()

    if applied != set(boxes_by_frame):
        tmp.unlink(missing_ok=True)
        raise RuntimeError("One or more reviewed frames were not present in the processed video.")
    os.replace(tmp, output_path)
    return len(applied)
