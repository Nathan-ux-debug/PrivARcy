"""Scan a video once to list the distinct people in it (the "who do you want to blur" picker).

Samples frames at a fixed interval, detects faces, links each face to the same person across samples (position first,
then appearance) and returns a ``FaceBook`` plus where each person appeared. The processing pass later asks the SAME
book who each face is, so "selected" stays selected for the whole video.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import cv2

from privarcy.pipeline.config import box_iou
from .identity import MERGE_THRESHOLD, FaceBook, FaceIdentity


@dataclass
class FaceScan:
    book: FaceBook
    total_frames: int
    stride: int
    frames_scanned: int = 0
    cancelled: bool = False
    samples: Dict[int, List[dict]] = field(default_factory=dict)      # frame_number -> [{face_id, bbox}]

    def identities(self) -> List[FaceIdentity]:
        return self.book.identities()


def scan_faces(video_path, detector, book: FaceBook, stride_seconds: float = 0.5, max_samples: int = 400,
               on_progress: Optional[Callable[[int, int], None]] = None, cancel: Optional[Callable[[], bool]] = None,
               link_iou: float = 0.3) -> FaceScan:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open the video: {video_path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) or 25.0
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        stride = max(1, int(round(fps * stride_seconds)))
        if total and total / stride > max_samples:
            stride = max(1, int(total / max_samples))
        scan = FaceScan(book=book, total_frames=total, stride=stride)
        expected = max(1, (total + stride - 1) // stride) if total else 0
        previous: List[tuple] = []                                    # (face_id, bbox) of the previous sample
        frame_number = 0
        while True:
            if cancel is not None and cancel():
                scan.cancelled = True
                break
            if frame_number % stride:
                if not capture.grab():                                  # skip frames between samples cheaply
                    break
            else:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
                current = _scan_frame(frame, frame_number, detector, book, previous, link_iou)
                scan.samples[frame_number] = [{"face_id": i, "bbox": b} for i, b in current]
                previous = current
                scan.frames_scanned += 1
                if on_progress is not None:
                    on_progress(scan.frames_scanned, expected)
            frame_number += 1
        return scan
    finally:
        capture.release()


def _scan_frame(frame, frame_number, detector, book: FaceBook, previous, link_iou) -> List[tuple]:
    used: set = set()
    result: List[tuple] = []
    for face in detector.detect(frame):
        found = book.embed(frame, face.bbox)
        if found is None:
            continue
        embedding, crop = found
        face_id = None
        best_iou = link_iou
        for prev_id, prev_box in previous:                              # same place as a moment ago -> same person
            iou = box_iou(prev_box, face.bbox)
            if prev_id not in used and iou >= best_iou:
                face_id, best_iou = prev_id, iou
        if face_id is None:                                             # otherwise by appearance (conservative)
            cand, sim = book.best_match(embedding)
            if cand is not None and cand not in used and sim >= MERGE_THRESHOLD:
                face_id = cand
        if face_id is None:
            face_id = book.create(embedding, crop, frame_number).face_id
        else:
            book.observe(face_id, embedding, crop, frame_number)
        used.add(face_id)                                               # two faces in one frame are two people
        result.append((face_id, tuple(face.bbox)))
    return result
