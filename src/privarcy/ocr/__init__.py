"""
ocr

Wraps a pre-trained TrOCR model (Hugging Face `transformers`) to read
text out of cropped bounding-box regions — the regions `yolo_detection`
already found (IDs, documents, screens) — and returns the recognized
text mapped back to its frame and bounding box.

Public API:
    TrOCREngine      - loads TrOCR, runs it on cropped regions
    OCRResult          - one recognized text string + confidence + frame/bbox linkage
    OCRError               - raised for a load or inference failure
"""

from .exceptions import OCRError
from .models import OCRResult
from .engine import TrOCREngine

__all__ = ["TrOCREngine", "OCRResult", "OCRError"]

__version__ = "1.0.0"
