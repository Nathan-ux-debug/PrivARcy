"""Pixel-level redaction primitives (blur / pixelate / solid mask)."""

import cv2
import numpy as np

from .config import normalize_method


def _factor(strength) -> float:
    """None keeps the original look; otherwise strength 0..1 maps to 0.5x..1.5x (0.5 == original)."""
    return 1.0 if strength is None else 0.5 + max(0.0, min(1.0, float(strength)))


def blur(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, strength=None) -> np.ndarray:
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return frame
    h, w = roi.shape[:2]
    f = _factor(strength)
    kw = max(3, int(w / 10 * f) * 2 + 1)
    kh = max(3, int(h / 10 * f) * 2 + 1)
    frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (kw, kh), 0)
    return frame


def pixelate(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, strength=None) -> np.ndarray:
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return frame
    h, w = roi.shape[:2]
    f = _factor(strength)
    small = cv2.resize(roi, (max(1, int(w / (15 * f))), max(1, int(h / (15 * f)))), interpolation=cv2.INTER_LINEAR)
    frame[y1:y2, x1:x2] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    return frame


def solid(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, strength=None) -> np.ndarray:
    frame[y1:y2, x1:x2] = 0
    return frame


def redact_region(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, method: str = "Blur", strength=None) -> np.ndarray:
    method = normalize_method(method)
    if method == "Pixelate":
        return pixelate(frame, x1, y1, x2, y2, strength)
    if method == "Solid Mask":
        return solid(frame, x1, y1, x2, y2, strength)
    return blur(frame, x1, y1, x2, y2, strength)


def redact_mask(frame: np.ndarray, mask: np.ndarray, method: str = "Blur", strength=None) -> np.ndarray:
    """Redact only the pixels where ``mask`` is non-zero (an object's shape instead of its bounding box)."""
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return frame
    x1, x2, y1, y2 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    patch = frame[y1:y2, x1:x2].copy()
    redact_region(patch, 0, 0, patch.shape[1], patch.shape[0], method, strength)
    inside = mask[y1:y2, x1:x2] > 0
    frame[y1:y2, x1:x2][inside] = patch[inside]
    return frame
