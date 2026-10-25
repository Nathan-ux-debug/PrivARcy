"""Core OCR logic: TrOCREngine.

Wraps Hugging Face's TrOCR (TrOCRProcessor + VisionEncoderDecoderModel)
so the rest of the pipeline only deals with NumPy crops in, OCRResult
out — no transformers/torch types leak past this module.
"""

import logging
from typing import Callable, List, Optional, Tuple

import numpy as np

from .exceptions import OCRError
from .models import OCRResult

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "microsoft/trocr-base-stage1"


def _default_loader(model_name: str):
    """Loads a TrOCRProcessor + VisionEncoderDecoderModel pair. Isolated
    as a standalone function so tests can inject a fake loader without
    requiring transformers/torch (and their large model download) to be
    installed."""
    try:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    except ImportError as exc:
        raise OCRError(
            "The 'transformers' package (and torch) is not installed. "
            "Install with `pip install transformers torch`."
        ) from exc

    try:
        processor = TrOCRProcessor.from_pretrained(model_name)
        model = VisionEncoderDecoderModel.from_pretrained(model_name)
    except Exception as exc:
        raise OCRError(f"Failed to load TrOCR model '{model_name}': {exc}") from exc

    return processor, model


def _resolve_device(requested: Optional[str]) -> str:
    if requested:
        return requested
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class TrOCREngine:
    """Loads TrOCR once, then runs repeated OCR on cropped regions.

    Parameters:
        model_name: Hugging Face model id. "microsoft/trocr-base-stage1"
                     (general-purpose) or "microsoft/trocr-small-printed"
                     (lighter, printed text) are common choices.
        device: "cuda", "cpu", or None to auto-detect.
        _loader: Internal hook for dependency injection in tests. Not
                  intended for normal use.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        _loader: Optional[Callable[[str], tuple]] = None,
    ) -> None:
        self.model_name = model_name
        self.device = _resolve_device(device)

        loader = _loader or _default_loader
        try:
            self.processor, self.model = loader(model_name)
        except OCRError:
            raise
        except Exception as exc:
            raise OCRError(f"Failed to load TrOCR model '{model_name}': {exc}") from exc

        if hasattr(self.model, "to"):
            try:
                self.model.to(self.device)
            except Exception:
                logger.warning("Could not move model to device='%s'; continuing on default device.", self.device)

        logger.info("TrOCREngine ready | model='%s' device='%s'", self.model_name, self.device)

    def read_region(
        self,
        frame: np.ndarray,
        bbox: Tuple[float, float, float, float],
        frame_id: Optional[str] = None,
        source_class: Optional[str] = None,
    ) -> OCRResult:
        """Crop `bbox` out of `frame` and run TrOCR on it.

        Raises:
            OCRError: if the frame/bbox is invalid, or inference fails.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise OCRError("read_region() received an empty or invalid frame array.")

        height, width = frame.shape[:2]
        x1, y1, x2, y2 = (int(round(v)) for v in bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            raise OCRError(f"bbox {bbox} is degenerate or entirely outside the frame ({width}x{height}).")

        crop = frame[y1:y2, x1:x2]

        try:
            from PIL import Image
            rgb_crop = crop[:, :, ::-1] if crop.ndim == 3 else crop  # BGR -> RGB
            pil_image = Image.fromarray(rgb_crop)
        except Exception as exc:
            raise OCRError(f"Failed to convert crop to an image for TrOCR: {exc}") from exc

        try:
            pixel_values = self.processor(images=pil_image, return_tensors="pt").pixel_values
            outputs = self.model.generate(pixel_values, output_scores=True, return_dict_in_generate=True)
            text = self.processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0]
            confidence = self._score_to_confidence(outputs)
        except Exception as exc:
            raise OCRError(f"TrOCR inference failed: {exc}") from exc

        return OCRResult(text=text.strip(), confidence=confidence, bbox=bbox, frame_id=frame_id, source_class=source_class)

    def read_regions(
        self,
        frame: np.ndarray,
        regions: List[dict],
    ) -> List[OCRResult]:
        """Batch convenience: run read_region() over several candidate
        regions from the same frame.

        Args:
            regions: List of dicts, each with at least a "bbox" key
                      (x_min, y_min, x_max, y_max), and optionally
                      "frame_id" and "source_class" — e.g. built directly
                      from yolo_detection.Detection objects:
                      [{"bbox": d.bbox, "source_class": d.class_name} for d in detections]

        Regions that fail (bad bbox, inference error) are skipped with a
        logged warning rather than aborting the whole batch — a single
        bad crop shouldn't lose every other reading from the same frame.
        """
        results: List[OCRResult] = []
        for region in regions:
            try:
                result = self.read_region(
                    frame,
                    bbox=region["bbox"],
                    frame_id=region.get("frame_id"),
                    source_class=region.get("source_class"),
                )
                results.append(result)
            except OCRError as exc:
                logger.warning("Skipping region %s: %s", region.get("bbox"), exc)
        return results

    @staticmethod
    def _score_to_confidence(outputs) -> float:
        """Derive a rough [0,1] confidence from TrOCR's per-token
        generation scores: the average of the top (chosen) token's
        softmax probability across all generated tokens. Not a
        calibrated probability of correctness, but a useful relative
        signal — a low value reliably means the model was unsure."""
        scores = getattr(outputs, "scores", None)
        if not scores:
            return 0.0

        try:
            import torch
        except ImportError:
            torch = None

        probs = []
        for step_scores in scores:
            if torch is not None and torch.is_tensor(step_scores):
                step_probs = torch.softmax(step_scores, dim=-1)
                probs.append(float(step_probs.max()))
            else:
                probs.append(
                    float(max(step_scores))
                    if hasattr(step_scores, "__iter__")
                    else float(step_scores)
                )

        return sum(probs) / len(probs) if probs else 0.0
