"""Sensitive-text stage: localise -> OCR -> PII rules -> decision.

For each region YOLO flagged as an OCR context (document, screen, id_card,
credit_card) this stage

  1. crops the region (with a small margin) and locates text LINES inside it,
  2. converts every line box back into ORIGINAL frame coordinates
     (crop offset added - see ocr.localizer),
  3. reads each *new* line with TrOCR (lines already read are re-used from a
     small IoU-matched cache, so OCR cost is paid once per line, not per frame),
  4. classifies the text with the project's PII rules through the shared
     DecisionEngine (HIGH -> auto redact, MEDIUM -> human review),
  5. returns boxes to redact. Non-sensitive text is left alone; the stage never
     redacts "all text".

Privacy of logs: recognised text lives only in local variables. Findings carry
the PII *pattern names* and the text length - never the text itself.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from privarcy.decision import DecisionEngine, DecisionInput, DecisionThresholds, DecisionTier
from privarcy.ocr.localizer import TextLocalizer
from .config import PipelineConfig, box_iou

logger = logging.getLogger(__name__)


@dataclass
class TextFinding:
    bbox: Tuple[int, int, int, int]
    tier: DecisionTier
    patterns: List[str]
    ocr_confidence: float
    source_class: str
    new: bool = True               # first time this line was seen (cache miss)
    text_length: int = 0


@dataclass
class TextStats:
    lines_found: int = 0
    lines_read: int = 0
    lines_cached: int = 0
    lines_deferred: int = 0        # found but not read this frame (OCR budget)
    uncertain: int = 0             # PII-like text read with LOW OCR confidence (not redacted on its own)
    too_small: int = 0
    ocr_errors: int = 0


class TextRedactionStage:
    def __init__(self, provider, config: PipelineConfig, localizer: Optional[TextLocalizer] = None) -> None:
        self.provider = provider
        self.config = config
        self.localizer = localizer or TextLocalizer(min_height=config.ocr_min_line_height)
        self.engine = DecisionEngine(
            thresholds=DecisionThresholds(t_high=config.auto_redact_confidence, t_low=config.review_min_confidence)
        )
        self._cache: list = []   # {"bbox", "run", "tier", "patterns", "conf", "len", "source"}
        self.stats = TextStats()

    @property
    def available(self) -> bool:
        return bool(self.config.ocr_enabled and self.provider is not None and self.provider.status()["available"])

    def expire(self, run_index: int, max_gap: int) -> None:
        self._cache = [e for e in self._cache if run_index - e["run"] <= max_gap]

    def process(
        self,
        frame: np.ndarray,
        region: Tuple[int, int, int, int],
        source_class: str,
        run_index: int,
        frame_id: str,
        stats: Optional[TextStats] = None,
    ) -> List[TextFinding]:
        stats = stats or self.stats
        engine = self.provider.get() if self.provider is not None else None
        if engine is None:
            return []

        height, width = frame.shape[:2]
        rx1, ry1, rx2, ry2 = region
        mx = int((rx2 - rx1) * 0.02)
        my = int((ry2 - ry1) * 0.02)
        cx1, cy1 = max(0, rx1 - mx), max(0, ry1 - my)
        cx2, cy2 = min(width, rx2 + mx), min(height, ry2 + my)
        if cx2 - cx1 < 16 or cy2 - cy1 < 8:
            return []

        crop = frame[cy1:cy2, cx1:cx2]
        # Boxes come back already shifted by (cx1, cy1): original-frame coordinates.
        lines = self.localizer.locate(crop, offset=(cx1, cy1))
        stats.lines_found += len(lines)

        findings: List[TextFinding] = []
        budget = self.config.ocr_max_new_lines_per_frame
        for line in lines:
            box = line.bbox
            if box[3] - box[1] < self.config.ocr_min_line_height:
                stats.too_small += 1
                continue

            entry = self._match_cache(box, source_class)
            if entry is not None:
                entry["bbox"], entry["run"] = box, run_index
                stats.lines_cached += 1
                if entry["tier"] != DecisionTier.PASS_THROUGH:
                    findings.append(self._finding(entry, new=False))
                continue

            if budget <= 0:
                stats.lines_deferred += 1
                continue
            budget -= 1

            try:
                result = engine.read_region(frame, box, frame_id=frame_id, source_class=source_class)
            except Exception as exc:
                stats.ocr_errors += 1
                logger.debug("OCR failed for a line (%s)", exc)
                continue
            stats.lines_read += 1

            decision = self.engine.decide(
                DecisionInput(
                    frame_id=frame_id, bbox=box, detected_class=None,
                    ocr_text=result.text, ocr_confidence=float(result.confidence),
                )
            )
            patterns = self._patterns_from(decision)
            tier = decision.tier
            if patterns and tier == DecisionTier.PASS_THROUGH:
                stats.uncertain += 1   # PII-shaped text, OCR too unsure to act on alone
            entry = {
                "bbox": box, "run": run_index, "tier": tier, "patterns": patterns,
                "conf": float(result.confidence), "len": len(result.text or ""), "source": source_class,
            }
            self._cache.append(entry)
            if tier != DecisionTier.PASS_THROUGH:
                findings.append(self._finding(entry, new=True))
        return findings

    # ------------------------------------------------------------------
    def _match_cache(self, box, source_class):
        best, best_iou = None, self.config.iou_threshold
        for entry in self._cache:
            iou = box_iou(entry["bbox"], box)
            if iou >= best_iou:
                best, best_iou = entry, iou
        return best

    @staticmethod
    def _patterns_from(decision) -> List[str]:
        # reasons look like "extracted text matched PII pattern(s): a, b"
        for reason in decision.reasons:
            if "PII pattern(s):" in reason:
                return [p.strip() for p in reason.split(":", 1)[1].split(",")]
        return []

    @staticmethod
    def _finding(entry, new: bool) -> TextFinding:
        x1, y1, x2, y2 = entry["bbox"]
        return TextFinding(
            bbox=(x1 - 2, y1 - 2, x2 + 2, y2 + 2), tier=entry["tier"], patterns=list(entry["patterns"]),
            ocr_confidence=entry["conf"], source_class=entry["source"], new=new, text_length=entry["len"],
        )
