"""Magic Blur: click an object, blur its SHAPE, keep following it.

Detection comes from the project's YOLO model, so "an object" means something the model can detect. The shape is
refined inside the detection box with GrabCut (classic OpenCV segmentation, no extra model). Following uses the
detected class, an HSV colour signature taken when the user clicked, and position - so it also works for objects
that were on screen before the frame the user clicked.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .config import box_iou
from .plan import MagicTarget

MAX_SIDE = 160           # GrabCut works on a downscaled crop; the mask is scaled back up
MIN_CORRELATION = 0.45


def colour_signature(frame: np.ndarray, box) -> Optional[np.ndarray]:
    x1, y1, x2, y2 = (int(v) for v in box)
    crop = frame[max(0, y1):y2, max(0, x1):x2]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist


def object_mask(frame: np.ndarray, box, pad: float = 0.04, iterations: int = 3) -> np.ndarray:
    """Full-frame uint8 mask (255 inside the object's shape). Falls back to the whole box if GrabCut fails."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in box)
    px, py = int((x2 - x1) * pad), int((y2 - y1) * pad)
    x1, y1, x2, y2 = max(0, x1 - px), max(0, y1 - py), min(w, x2 + px), min(h, y2 + py)
    mask = np.zeros((h, w), np.uint8)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return mask
    crop = frame[y1:y2, x1:x2]
    scale = min(1.0, MAX_SIDE / max(crop.shape[:2]))
    small = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else crop.copy()
    gc = np.zeros(small.shape[:2], np.uint8)
    inset = max(1, int(min(small.shape[:2]) * 0.03))
    rect = (inset, inset, small.shape[1] - 2 * inset, small.shape[0] - 2 * inset)
    try:
        cv2.grabCut(small, gc, rect, np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64), iterations, cv2.GC_INIT_WITH_RECT)
        fg = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        if fg.mean() < 255 * 0.15:                                      # segmentation collapsed -> use the box
            fg[:] = 255
    except cv2.error:
        fg = np.full(small.shape[:2], 255, np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    fg = cv2.resize(fg, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST) if scale < 1 else fg
    fg = cv2.dilate(fg, np.ones((5, 5), np.uint8))                      # a little margin so edges never leak
    mask[y1:y2, x1:x2] = fg
    return mask


class MagicTracker:
    def __init__(self, targets: Sequence[MagicTarget], max_gap: int = 5) -> None:
        self.targets = list(targets)
        self.max_gap = max_gap
        self._box: Dict[int, Tuple[int, int, int, int]] = {i: tuple(t.box) for i, t in enumerate(self.targets)}
        self._seen: Dict[int, int] = {i: 0 for i in range(len(self.targets))}
        self._masks: Dict[int, np.ndarray] = {}

    def update(self, frame: np.ndarray, detections: List[dict], run: int) -> List[np.ndarray]:
        """Match targets to this run's detections; returns one full-frame mask per currently tracked target."""
        used = set()
        for i, target in enumerate(self.targets):
            best, best_score = None, 0.0
            for j, det in enumerate(detections):
                if j in used or det["name"] != target.name:
                    continue
                sig = colour_signature(frame, det["box"]) if target.signature is not None else None
                corr = cv2.compareHist(target.signature, sig, cv2.HISTCMP_CORREL) if sig is not None else 0.0
                iou = box_iou(self._box[i], det["box"])
                score = 0.6 * max(0.0, corr) + 0.4 * iou
                if target.signature is not None and corr < MIN_CORRELATION and iou < 0.3:
                    continue
                if score > best_score:
                    best, best_score = j, score
            if best is not None:
                used.add(best)
                self._box[i], self._seen[i] = tuple(detections[best]["box"]), run
                self._masks[i] = object_mask(frame, self._box[i])
            elif run - self._seen[i] > self.max_gap:
                self._masks.pop(i, None)
        return self.current_masks()

    def current_masks(self) -> List[np.ndarray]:
        return [m for m in self._masks.values() if m is not None and m.any()]
