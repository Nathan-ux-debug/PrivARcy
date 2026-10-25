"""Lazy, failure-tolerant access to the TrOCR engine.

The OCR engine needs ``transformers`` (and a one-time model download). Neither
is guaranteed, and model loading is slow, so the pipeline asks the provider
instead of importing TrOCR directly. If OCR cannot be loaded the pipeline keeps
working (YOLO only) and the UI reports the exact reason.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)


class OCRProvider:
    def __init__(self, model_name: str = "microsoft/trocr-base-stage1", device: Optional[str] = None, engine=None) -> None:
        self.model_name = model_name
        self.device = device
        self._engine = engine          # injectable (tests / alternative engines)
        self._error: Optional[str] = None
        self._lock = threading.Lock()
        self._attempted = engine is not None

    def get(self):
        """Return an engine with ``read_region`` or None. Never raises."""
        if self._engine is not None or self._attempted:
            return self._engine
        with self._lock:
            if self._attempted:
                return self._engine
            self._attempted = True
            try:
                from .engine import TrOCREngine

                self._engine = TrOCREngine(model_name=self.model_name, device=self.device)
                logger.info("OCR engine loaded: %s", self.model_name)
            except Exception as exc:
                self._error = str(exc)
                logger.warning("OCR unavailable: %s", exc)
        return self._engine

    @property
    def error(self) -> Optional[str]:
        return self._error

    @property
    def loaded(self) -> bool:
        return self._engine is not None

    def status(self) -> dict:
        """Cheap status that does NOT trigger a model load."""
        if self._engine is not None:
            return {"available": True, "loaded": True, "model": self.model_name, "reason": ""}
        if self._error:
            return {"available": False, "loaded": False, "model": self.model_name, "reason": self._error}
        try:
            import importlib.util

            has_tf = importlib.util.find_spec("transformers") is not None
        except Exception:
            has_tf = False
        if not has_tf:
            return {
                "available": False, "loaded": False, "model": self.model_name,
                "reason": "The 'transformers' package is not installed in this environment (pip install transformers).",
            }
        return {"available": True, "loaded": False, "model": self.model_name, "reason": "Will load on first use."}
