"""Core detection logic: YOLODetector.

Wraps an Ultralytics YOLOv8 model so the rest of the pipeline only ever
deals with plain NumPy arrays in, and Detection/DetectionResult
dataclasses out — no torch/ultralytics types leak past this module.
"""

import logging
import time
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

from privarcy.paths import model_path
from .exceptions import ModelLoadError, InferenceError
from .models import Detection, DetectionResult
from .device import resolve_device

logger = logging.getLogger(__name__)

DEFAULT_MODEL = model_path()


def _default_model_loader(model_path: str):
    """Loads an Ultralytics YOLO model from a path or a known model name.

    Isolated as a standalone function (rather than inlined in __init__)
    so tests can inject a fake loader and exercise YOLODetector's parsing
    / filtering / error-handling logic without requiring torch/ultralytics
    to be installed.
    """
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ModelLoadError(
            "The 'ultralytics' package is not installed. Install it with "
            "`pip install ultralytics` (this also installs PyTorch)."
        ) from exc

    try:
        return YOLO(model_path)
    except Exception as exc:  # Ultralytics raises assorted errors for bad/missing weights
        raise ModelLoadError(f"Failed to load YOLO weights from '{model_path}': {exc}") from exc


def _to_numpy(tensor) -> np.ndarray:
    """Best-effort conversion of a torch.Tensor (or array-like) to NumPy,
    without a hard import of torch anywhere else in the codebase."""
    if hasattr(tensor, "cpu"):
        tensor = tensor.cpu()
    if hasattr(tensor, "numpy"):
        tensor = tensor.numpy()
    return np.asarray(tensor)


class YOLODetector:
    """Loads a custom model once, then runs repeated inference on frames.

    Parameters:
        model_path: Path or name of the YOLOv8 weights (e.g. "yolov8n.pt",
                    "yolov8s.pt", or a path to custom-trained weights).
                    Ultralytics will auto-download known model names on
                    first use if not already cached locally.
        device: "cuda", "cpu", "cuda:0", or None to auto-detect (CUDA
                 used automatically if available, else CPU).
        confidence_threshold: Default minimum confidence to keep a
                                detection; overridable per predict() call.
        iou_threshold: Non-max-suppression IoU threshold passed to the model.
        _model_loader: Internal hook for dependency injection in unit
                         tests. Not intended for normal use.
    """

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL,
        device: Optional[str] = None,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        _model_loader: Optional[Callable[[str], object]] = None,
    ) -> None:
        if not (0.0 <= confidence_threshold <= 1.0):
            raise ValueError(f"confidence_threshold must be within [0,1], got {confidence_threshold}")
        if not (0.0 <= iou_threshold <= 1.0):
            raise ValueError(f"iou_threshold must be within [0,1], got {iou_threshold}")

        self.model_path = str(model_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = resolve_device(device)

        loader = _model_loader or _default_model_loader
        try:
            self._model = loader(self.model_path)
        except ModelLoadError:
            raise
        except Exception as exc:
            raise ModelLoadError(f"Failed to load YOLO weights from '{self.model_path}': {exc}") from exc

        # class_id -> class_name lookup, provided by Ultralytics on the loaded model
        self._class_names = getattr(self._model, "names", {}) or {}

        logger.info(
            "YOLODetector ready | model='%s' device='%s' conf_threshold=%.2f iou_threshold=%.2f",
            self.model_path, self.device, self.confidence_threshold, self.iou_threshold,
        )

    def predict(
        self,
        frame: np.ndarray,
        confidence_threshold: Optional[float] = None,
        source_frame_index: Optional[int] = None,
        source_video: Optional[str] = None,
    ) -> DetectionResult:
        """Run inference on a single frame and return a DetectionResult.

        Args:
            frame: BGR NumPy array (H, W, 3) — e.g. `FrameMeta.frame` from
                   the frame_extraction package.
            confidence_threshold: Overrides the instance-level default
                                    threshold for this call only.
            source_frame_index / source_video: Optional metadata carried
                    through into the result, to trace a detection back to
                    the originating video/frame in an upstream pipeline.

        Raises:
            InferenceError: if the frame is invalid or the forward pass fails.
            ValueError: if confidence_threshold is outside [0, 1].
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise InferenceError("predict() received an empty or invalid frame array.")

        threshold = self.confidence_threshold if confidence_threshold is None else confidence_threshold
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"confidence_threshold must be within [0,1], got {threshold}")

        start = time.perf_counter()
        try:
            results = self._model(
                frame,
                conf=threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False,
            )
        except Exception as exc:
            raise InferenceError(f"YOLOv8 inference failed: {exc}") from exc
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        detections = self._parse_results(results, threshold)

        return DetectionResult(
            detections=detections,
            model_name=Path(self.model_path).name,
            image_shape=tuple(frame.shape),
            inference_time_ms=elapsed_ms,
            device=self.device,
            source_frame_index=source_frame_index,
            source_video=source_video,
        )

    def _parse_results(self, results, threshold: float) -> List[Detection]:
        """Convert an Ultralytics Results object into Detection objects.

        Ultralytics returns a list of Results (one per input image); a
        single-frame call yields exactly one. Each Results.boxes exposes
        .xyxy / .conf / .cls as tensors — converted to NumPy here so no
        other module in this package needs to import torch.
        """
        if not results:
            return []

        boxes = getattr(results[0], "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = _to_numpy(boxes.xyxy)
        confs = _to_numpy(boxes.conf)
        classes = _to_numpy(boxes.cls)

        detections: List[Detection] = []
        for (x_min, y_min, x_max, y_max), conf, cls in zip(xyxy, confs, classes):
            conf = float(conf)
            if conf < threshold:
                continue  # defense-in-depth; the model's `conf` kwarg should already filter this
            class_id = int(cls)
            class_name = self._class_names.get(class_id, str(class_id))
            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=conf,
                    bbox=(float(x_min), float(y_min), float(x_max), float(y_max)),
                )
            )

        # Highest-confidence first — friendlier default ordering for API/UI consumers.
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections
