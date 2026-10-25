"""Text localisation: find WHERE text lines are inside a region.

TrOCR recognises the text of a pre-cropped line; it does not produce text
boxes. This module supplies the missing step with a dependency-free,
classical OpenCV method (morphological gradient -> horizontal closing ->
connected components -> geometric filters -> line merge). It is fast enough
to run on every frame inside a YOLO region and works best on printed or
rendered text with reasonable contrast.

Optionally a DB text-detection ONNX model can be supplied
(``model_path=...``, via ``cv2.dnn_TextDetectionModel_DB``); if it cannot be
loaded the classical method is used. Limitations (stated plainly): very small
or heavily blurred text, curved text and low-contrast handwriting are missed.

Coordinates
-----------
``locate(image, offset=(ox, oy))`` returns boxes in the ORIGINAL frame's
coordinates: each box found inside the crop is shifted by the crop's offset.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

Box = Tuple[int, int, int, int]


@dataclass
class TextRegion:
    bbox: Box            # x1, y1, x2, y2 in ORIGINAL frame coordinates
    score: float         # 0..1 geometric/ink-density plausibility (NOT an OCR confidence)


class TextLocalizer:
    def __init__(
        self,
        min_height: int = 8,
        max_height_ratio: float = 0.35,
        min_aspect: float = 1.6,
        min_density: float = 0.18,
        working_width: int = 960,
        model_path: Optional[str] = None,
    ) -> None:
        self.min_height = min_height
        self.max_height_ratio = max_height_ratio
        self.min_aspect = min_aspect
        self.min_density = min_density
        self.working_width = working_width
        self._dnn = None
        if model_path:
            try:
                self._dnn = cv2.dnn_TextDetectionModel_DB(str(model_path))
                self._dnn.setBinaryThreshold(0.3).setPolygonThreshold(0.5).setMaxCandidates(200)
                self._dnn.setInputParams(1.0 / 255, (736, 736), (122.68, 116.67, 104.01), True)
            except Exception as exc:  # pragma: no cover - optional
                logger.warning("DB text detector could not be loaded (%s); using classical method.", exc)
                self._dnn = None

    # ------------------------------------------------------------------
    def locate(self, image: np.ndarray, offset: Tuple[int, int] = (0, 0)) -> List[TextRegion]:
        if image is None or image.size == 0:
            return []
        if self._dnn is not None:
            regions = self._locate_dnn(image, offset)
            if regions is not None:
                return regions
        return self._locate_classical(image, offset)

    # ------------------------------------------------------------------
    def _locate_dnn(self, image, offset):  # pragma: no cover - optional path
        try:
            polygons, confidences = self._dnn.detect(image)
        except Exception as exc:
            logger.warning("DB text detection failed (%s); falling back.", exc)
            return None
        out = []
        for poly, conf in zip(polygons, confidences):
            pts = np.asarray(poly).reshape(-1, 2)
            x1, y1 = pts.min(axis=0)
            x2, y2 = pts.max(axis=0)
            out.append(TextRegion((int(x1) + offset[0], int(y1) + offset[1], int(x2) + offset[0], int(y2) + offset[1]), float(conf)))
        return out

    def _locate_classical(self, image, offset) -> List[TextRegion]:
        height, width = image.shape[:2]
        scale = 1.0
        work = image
        if width > self.working_width:
            scale = self.working_width / float(width)
            work = cv2.resize(image, (self.working_width, max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
        elif width < 320:  # tiny crops: upscale so characters span enough pixels
            scale = 320.0 / width
            work = cv2.resize(image, (320, max(1, int(height * scale))), interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY) if work.ndim == 3 else work
        wh, ww = gray.shape[:2]
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        _, ink = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # Connect characters/words on a line: kernel width scales with image width.
        kernel_w = max(9, int(ww * 0.018))
        connected = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 1)))
        connected = cv2.morphologyEx(connected, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2)))

        contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes: List[Tuple[int, int, int, int, float]] = []
        min_h = max(4, int(self.min_height * scale))
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if h < min_h or h > wh * self.max_height_ratio:
                continue
            if w < h * self.min_aspect:
                continue
            roi = ink[y:y + h, x:x + w]
            density = float(np.count_nonzero(roi)) / float(w * h)
            if density < self.min_density:
                continue
            boxes.append((x, y, x + w, y + h, min(1.0, density * 2.0)))

        merged = self._merge_lines(boxes)
        regions: List[TextRegion] = []
        pad_x, pad_y = 2, 2
        for x1, y1, x2, y2, score in merged:
            fx1 = int(max(0, (x1 - pad_x) / scale))
            fy1 = int(max(0, (y1 - pad_y) / scale))
            fx2 = int(min(width, (x2 + pad_x) / scale))
            fy2 = int(min(height, (y2 + pad_y) / scale))
            if fy2 - fy1 < self.min_height:
                continue
            regions.append(TextRegion((fx1 + offset[0], fy1 + offset[1], fx2 + offset[0], fy2 + offset[1]), score))
        regions.sort(key=lambda r: (r.bbox[1], r.bbox[0]))
        return regions

    @staticmethod
    def _merge_lines(boxes):
        """Merge boxes that overlap vertically and are horizontally close,
        repeating until stable (word boxes are not visited in x order)."""
        merged = list(boxes)
        changed = True
        while changed:
            changed = False
            out: list = []
            for box in sorted(merged, key=lambda b: (b[1], b[0])):
                x1, y1, x2, y2, s = box
                for i, m in enumerate(out):
                    mx1, my1, mx2, my2, ms = m
                    v_overlap = min(y2, my2) - max(y1, my1)
                    min_h = min(y2 - y1, my2 - my1)
                    gap = max(x1 - mx2, mx1 - x2)
                    if v_overlap > 0.5 * min_h and gap < 1.5 * max(y2 - y1, my2 - my1):
                        out[i] = (min(x1, mx1), min(y1, my1), max(x2, mx2), max(y2, my2), max(s, ms))
                        changed = True
                        break
                else:
                    out.append(box)
            merged = out
        return merged
