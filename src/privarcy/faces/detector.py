"""Face detection.

Uses OpenCV's built-in Haar cascade classifier — ships with opencv-python
(no model download needed), runs on CPU, and is fast enough for real-time
at the cost of being less accurate than a modern deep-learning face
detector, especially off-angle or low-light. See the README for the
upgrade path (e.g. cv2's DNN-based YuNet face detector) if accuracy here
turns out to be the bottleneck.
"""

import logging
from pathlib import Path
from typing import List

import cv2
import numpy as np

from .exceptions import FacePrivacyError
from .models import FaceBox

logger = logging.getLogger(__name__)


def opencv_diagnosis(extra: str = "") -> str:
    """Explain WHY Haar face detection is unavailable, using evidence from the
    interpreter that is actually running (not guesses)."""
    import importlib.metadata as md
    import sys

    dists = []
    for name in ("opencv-python", "opencv-contrib-python", "opencv-python-headless", "opencv-contrib-python-headless"):
        try:
            dists.append(f"{name}=={md.version(name)}")
        except md.PackageNotFoundError:
            pass
    return (
        "The imported OpenCV module does not provide CascadeClassifier. "
        f"{extra} "
        f"interpreter={sys.executable}; cv2.__file__={getattr(cv2, '__file__', None)}; "
        f"cv2.__version__={getattr(cv2, '__version__', None)}; installed OpenCV distributions={dists or 'none'}. "
        "Run `python scripts/diagnostics/diagnose_opencv.py` with the SAME interpreter for the full report. "
        "PrivARcy continues without Haar: YOLO face boxes are used for detection and face identification."
    ).replace("  ", " ")


class FaceDetector:
    """Finds face bounding boxes in a frame.

    Parameters:
        cascade_path: Path to a Haar cascade XML file. Defaults to
                       OpenCV's bundled frontal-face cascade.
        scale_factor: How much the image size is reduced at each scale
                       (smaller = more thorough but slower). Default 1.1.
        min_neighbors: How many overlapping detections are required to
                        keep a face (higher = fewer false positives, but
                        can miss real faces). Default 5.
        min_size_px: Smallest face (width, height) in pixels to detect —
                      filters out tiny false-positive blobs.
    """

    def __init__(
        self,
        cascade_path: str | Path | None = None,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size_px: tuple[int, int] = (30, 30),
    ) -> None:
        if scale_factor <= 1.0:
            raise FacePrivacyError(f"scale_factor must be > 1.0, got {scale_factor}")
        if min_neighbors < 1:
            raise FacePrivacyError(f"min_neighbors must be >= 1, got {min_neighbors}")

        if not hasattr(cv2, "CascadeClassifier"):
            raise FacePrivacyError(opencv_diagnosis())

        haar_dir = getattr(getattr(cv2, "data", None), "haarcascades", None)
        if not cascade_path and not haar_dir:
            raise FacePrivacyError(opencv_diagnosis("cv2.data.haarcascades is unavailable."))
        path = str(cascade_path) if cascade_path else haar_dir + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(path)
        if self._cascade.empty():
            raise FacePrivacyError(f"Could not load Haar cascade from '{path}'.")

        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size_px = min_size_px

    def detect(self, frame: np.ndarray) -> List[FaceBox]:
        """Return every detected face in the frame.

        Raises:
            FacePrivacyError: if frame is invalid.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise FacePrivacyError("detect() received an empty or invalid frame array.")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        # detectMultiScale3 also returns reject levels / weights, which we
        # use as a rough confidence proxy — plain detectMultiScale doesn't.
        raw = self._cascade.detectMultiScale3(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size_px,
            outputRejectLevels=True,
        )
        boxes, _reject_levels, weights = raw
        if len(boxes) == 0:
            return []

        faces: List[FaceBox] = []
        for (x, y, w, h), weight in zip(boxes, weights):
            # `weight` is an unbounded positive score, not a probability;
            # squash it into a rough [0,1] confidence proxy for consistency
            # with the rest of the pipeline (yolo_detection's confidences).
            confidence = float(min(1.0, weight / 10.0))
            faces.append(FaceBox(bbox=(float(x), float(y), float(x + w), float(y + h)), confidence=confidence))

        return faces


class YoloFaceDetector:
    """Face detector backed by the project's own YOLO model (class name 'face').

    Why this exists
    ---------------
    * Redaction already uses YOLO 'face' boxes, so face identification and
      enrolment should crop faces the SAME way. Matching HOG embeddings made
      from Haar crops against YOLO crops (or vice versa) lowers similarity.
    * It needs no ``cv2.CascadeClassifier`` at all, so a broken OpenCV install
      can no longer stop the application from starting.
    * YOLO at full resolution copes with moving / blurred / small faces much
      better than a Haar cascade.

    ``model`` is any object with Ultralytics-style ``predict`` and ``names``.
    """

    def __init__(self, model, confidence: float = 0.35, imgsz: int = 960) -> None:
        self._model = model
        self.confidence = confidence
        self.imgsz = imgsz
        names = getattr(model, "names", {}) or {}
        items = names.items() if isinstance(names, dict) else enumerate(names)
        self._face_ids = {int(i) for i, n in items if str(n).strip().lower() == "face"}

    @property
    def available(self) -> bool:
        return bool(self._face_ids)

    def detect(self, frame: np.ndarray) -> List[FaceBox]:
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise FacePrivacyError("detect() received an empty or invalid frame array.")
        if not self._face_ids:
            return []
        faces: List[FaceBox] = []
        for result in self._model.predict(source=frame, conf=self.confidence, imgsz=self.imgsz, verbose=False):
            if result.boxes is None:
                continue
            for box in result.boxes:
                if int(box.cls[0].item()) not in self._face_ids:
                    continue
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].cpu().numpy())
                faces.append(FaceBox(bbox=(x1, y1, x2, y2), confidence=float(box.conf[0].item())))
        return faces


class CompositeFaceDetector:
    """Try detectors in order; return the first non-empty result.

    Used for enrolment (portrait photos) and as the app-wide ``face_detector``.
    Detectors that failed to construct are simply not included, so the app
    degrades gracefully instead of raising at startup.
    """

    def __init__(self, detectors: list) -> None:
        self.detectors = [d for d in detectors if d is not None]
        if not self.detectors:
            raise FacePrivacyError("No face detector backend is available (no YOLO face class and no Haar cascade).")

    def detect(self, frame: np.ndarray) -> List[FaceBox]:
        for detector in self.detectors:
            faces = detector.detect(frame)
            if faces:
                return faces
        return []

    @property
    def backends(self) -> List[str]:
        return [type(d).__name__ for d in self.detectors]


def build_face_detector(yolo_model=None):
    """Create the best available detector without ever raising at startup.

    Returns (detector_or_None, notes) where notes explain what was skipped.
    """
    detectors, notes = [], []
    if yolo_model is not None:
        yolo = YoloFaceDetector(yolo_model)
        if yolo.available:
            detectors.append(yolo)
        else:
            notes.append("YOLO model has no 'face' class.")
    try:
        detectors.append(FaceDetector())
    except FacePrivacyError as exc:
        notes.append(str(exc))
    except Exception as exc:  # pragma: no cover - defensive
        notes.append(f"Haar detector unavailable: {exc}")
    if not detectors:
        return None, notes
    return CompositeFaceDetector(detectors), notes
