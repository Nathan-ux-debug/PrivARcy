"""FrameEngine - the unified per-frame decision pipeline.

One instance holds the state of ONE stream (a recorded video or a live feed)
and is shared by VideoPipeline and LiveSession, so recorded and live
processing follow exactly the same rules:

    YOLO detect
      -> class filter (names, never ids; 'document' is an OCR region only)
      -> temporal smoothing (EMA, IoU >= 0.50) and track lifecycle
         (a track expires after 5 unmatched detection runs)
      -> DecisionEngine: HIGH >= 0.80 auto-redact | MEDIUM 0.50-0.80 review |
         LOW < 0.50 never redacted on its own (counted as an uncertainty warning)
      -> face policy (auto / selected / every) with stable per-person identities
      -> OCR text stage for document / screen / card regions
      -> redaction (applied last, on a copy-free pass, so every read - YOLO,
         face identity, OCR, review crops - sees the ORIGINAL pixels)

Differences from the previous inline loop, all privacy-motivated:
  * an auto-redacted track keeps being redacted until it expires (previously a
    one-frame detection miss exposed the object),
  * the redacted region is union(raw, smoothed) so EMA lag cannot expose a
    moving object, plus a small per-class padding,
  * recognised text is never stored; only PII pattern names are.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from privarcy.decision import DecisionEngine, DecisionInput, DecisionThresholds, DecisionTier
from .config import (
    OCR_CONTEXT_CLASSES, PipelineConfig, box_iou, clamp_box, model_class_names, normalize_method, pad_box, union_box,
)
from .magic import MagicTracker
from .plan import EditPlan
from .redact_ops import redact_mask, redact_region
from .text_stage import TextRedactionStage, TextStats

logger = logging.getLogger(__name__)


@dataclass
class FaceView:
    face_id: Optional[str]
    label: str
    bbox: Tuple[int, int, int, int]
    blurred: bool
    confidence: float


@dataclass
class FrameOutput:
    frame: np.ndarray
    records: List[dict] = field(default_factory=list)
    detections: int = 0          # detections >= review floor (kept name for compatibility)
    redactions: int = 0          # redaction operations applied to this frame
    flagged_new: int = 0
    held: int = 0                # redactions that came from a held (undetected-this-run) track
    text_redactions: int = 0
    low_confidence: Dict[str, int] = field(default_factory=dict)
    detection_ran: bool = True
    boxes: List[Tuple[int, int, int, int, str]] = field(default_factory=list)  # for overlays/tests
    faces: List["FaceView"] = field(default_factory=list)   # every face seen this frame + whether it was blurred


class FrameEngine:
    def __init__(
        self,
        model,
        config: PipelineConfig,
        source_name: str = "stream",
        face_policy=None,
        plan: Optional[EditPlan] = None,
        ocr_provider=None,
        review_dir: Optional[Path] = None,
        live: bool = False,
        text_stage: Optional[TextRedactionStage] = None,
    ) -> None:
        self.model = model
        self.config = config
        self.source_name = source_name
        self.face_policy = face_policy
        self.plan = plan
        self.magic = MagicTracker(plan.magic, config.track_max_gap) if plan is not None and plan.magic else None
        self.live = live
        self.review_dir = Path(review_dir) if review_dir else None
        if self.review_dir and config.save_review_crops and not live:
            self.review_dir.mkdir(parents=True, exist_ok=True)

        self.names = model_class_names(model)
        self.tracks: List[dict] = []
        self.review_tracks: List[dict] = []
        self.records: List[dict] = []
        self._run = 0
        self._engines: Dict[str, DecisionEngine] = {}

        use_ocr = config.ocr_enabled and (not live or config.live_ocr_enabled)
        self.text_stage = text_stage if text_stage is not None else (
            TextRedactionStage(ocr_provider, config) if (use_ocr and ocr_provider is not None) else None
        )

        # cumulative counters
        self.total_detections = 0
        self.total_redactions = 0
        self.total_flagged = 0
        self.total_held = 0
        self.total_text_redactions = 0
        self.low_confidence_totals: Dict[str, int] = {}
        self.text_stats = TextStats()

    # ------------------------------------------------------------------
    # decision helpers
    # ------------------------------------------------------------------
    def _engine_for(self, class_name: str) -> DecisionEngine:
        engine = self._engines.get(class_name)
        if engine is None:
            engine = DecisionEngine(
                thresholds=DecisionThresholds(
                    t_high=self.config.auto_threshold_for(class_name), t_low=self.config.review_min_confidence
                ),
                sensitive_classes=set(self.config.enabled_classes),
            )
            self._engines[class_name] = engine
        return engine

    def reset_decision_cache(self) -> None:
        """Call after the config's thresholds / enabled classes change."""
        self._engines.clear()

    def _class_name(self, class_id: int, result=None) -> str:
        names = getattr(result, "names", None)
        if isinstance(names, dict) and class_id in names:
            return str(names[class_id]).strip().lower()
        if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
            return str(names[class_id]).strip().lower()
        return self.names.get(class_id, f"class_{class_id}")

    # ------------------------------------------------------------------
    # tracking
    # ------------------------------------------------------------------
    def _expire_tracks(self) -> None:
        gap = self.config.track_max_gap
        self.tracks = [t for t in self.tracks if self._run - t["run"] <= gap]
        self.review_tracks = [t for t in self.review_tracks if self._run - t["run"] <= gap]
        if self.text_stage is not None:
            self.text_stage.expire(self._run, gap)

    def _smooth(self, label: str, box) -> dict:
        best, best_iou = None, self.config.iou_threshold
        for track in self.tracks:
            if track["label"] != label or track["run"] >= self._run:
                continue
            iou = box_iou(track["bbox"], box)
            if iou >= best_iou:
                best, best_iou = track, iou
        if best is None:
            track = {"label": label, "bbox": tuple(box), "run": self._run, "auto": False}
            self.tracks.append(track)
            return track
        alpha = 2.0 / (max(1, self.config.temporal_smoothing) + 1.0)
        best["bbox"] = tuple(int(round((1.0 - alpha) * p + alpha * c)) for p, c in zip(best["bbox"], box))
        best["run"] = self._run
        return best

    def _find_review_track(self, label: str, box):
        best, best_iou = None, self.config.iou_threshold
        for track in self.review_tracks:
            gap = self._run - track["run"]
            if track["label"] != label or gap <= 0 or gap > self.config.track_max_gap:
                continue
            iou = box_iou(track["bbox"], box)
            if iou >= best_iou:
                best, best_iou = track, iou
        return best

    # ------------------------------------------------------------------
    # main entry
    # ------------------------------------------------------------------
    def process(self, frame: np.ndarray, frame_number: int, fps: float = 30.0, run_detection: bool = True) -> FrameOutput:
        cfg = self.config
        height, width = frame.shape[:2]
        out = FrameOutput(frame=frame, detection_ran=run_detection)
        ops: List[Tuple[Tuple[int, int, int, int], str]] = []   # (box, label)
        ocr_candidates: List[Tuple[Tuple[int, int, int, int], str]] = []
        ocr_ready = bool(self.text_stage is not None and self.text_stage.available)

        if run_detection:
            self._run += 1
            self._expire_tracks()
            dets = self._detect(frame, width, height, out)

            for index, det in enumerate(dets):
                name, conf, box = det["name"], det["conf"], det["box"]
                if name in OCR_CONTEXT_CLASSES:
                    ocr_candidates.append((box, name))
                if name == "document" or name not in cfg.enabled_classes:
                    continue

                track = self._smooth(name, box)
                smoothed = track["bbox"]
                is_face_policy = bool(name == "face" and self.face_policy is not None)
                screen_text_only = bool(name == "screen" and cfg.screen_policy == "ocr_only" and ocr_ready)

                decision_reasons: List[str] = []
                frame_id = f"{self.source_name}:{frame_number}:{len(self.records)}"
                decision = self._engine_for(name).decide(
                    DecisionInput(frame_id=frame_id, bbox=tuple(float(v) for v in smoothed), detected_class=name,
                                  detection_confidence=conf)
                )

                record = self._new_record(name, conf, smoothed, frame_number, fps, frame_id)
                redact_box = self._redaction_box(name, box, smoothed, width, height)
                record["redact_bbox"] = list(redact_box)
                out.detections += 1

                if is_face_policy:
                    verdict = self.face_policy.verdict(frame, track, box, frame_number)
                    record.update(face_id=verdict.face_id, face_label=verdict.label)
                    if verdict.blur:
                        record["status"] = "auto_redacted"
                        track["auto"] = True
                        ops.append((redact_box, name))
                    else:
                        record["status"] = "kept_visible"
                        track["auto"] = False
                    decision_reasons = [verdict.reason]
                    out.faces.append(FaceView(verdict.face_id, verdict.label, redact_box, verdict.blur, conf))
                elif screen_text_only:
                    record["status"] = "text_only"
                elif decision.tier == DecisionTier.AUTO_REDACT:
                    record["status"] = "auto_redacted"
                    track["auto"] = True
                    ops.append((redact_box, name))
                elif decision.tier == DecisionTier.FLAGGED:
                    self._handle_review(record, name, smoothed, frame, frame_number, out)
                record["reasons"] = decision_reasons if is_face_policy else list(decision.reasons)
                self.records.append(record)
                out.records.append(record)
                out.boxes.append((*redact_box, record["status"]))

            # --- held tracks: object seen before, missed this run ---------------
            if cfg.hold_missed_detections:
                for track in self.tracks:
                    if track["auto"] and track["run"] < self._run:
                        box = pad_box(track["bbox"], cfg.box_padding.get(track["label"], 0.0), width, height)
                        ops.append((box, track["label"]))
                        out.held += 1
        else:
            for track in self.tracks:
                if track["auto"]:
                    box = pad_box(track["bbox"], cfg.box_padding.get(track["label"], 0.0), width, height)
                    ops.append((box, track["label"]))
                    out.held += 1

        # --- user edits: Magic Blur masks + Manual Blur regions --------------------
        mask_ops: List[np.ndarray] = []
        if self.magic is not None:
            if run_detection:
                all_dets = getattr(self, "_last_dets", [])
                mask_ops.extend(self.magic.update(frame, all_dets, self._run))
            else:
                mask_ops.extend(self.magic.current_masks())
        if self.plan is not None and self.plan.manual:
            seconds = frame_number / fps if fps > 0 else 0.0
            for region in self.plan.manual:
                if not region.active(seconds):
                    continue
                x1, y1, x2, y2 = region.box(width, height)
                if x2 <= x1 or y2 <= y1:
                    continue
                if region.shape == "ellipse":
                    m = np.zeros((height, width), np.uint8)
                    cv2.ellipse(m, ((x1 + x2) // 2, (y1 + y2) // 2), ((x2 - x1) // 2, (y2 - y1) // 2), 0, 0, 360, 255, -1)
                    mask_ops.append(m)
                else:
                    ops.append(((x1, y1, x2, y2), "manual"))

        # --- OCR text stage (reads ORIGINAL pixels; before any redaction) ------
        if run_detection and ocr_ready and ocr_candidates:
            self._text_stage(frame, frame_number, fps, ocr_candidates, ops, out)

        # --- apply redactions last ------------------------------------------
        method = normalize_method(cfg.redaction_method)
        strength = cfg.redaction_strength
        for (x1, y1, x2, y2), _label in ops:
            if x2 > x1 and y2 > y1:
                redact_region(out.frame, x1, y1, x2, y2, method, strength)
        for mask in mask_ops:
            redact_mask(out.frame, mask, method, strength)
        out.redactions = len(ops) + len(mask_ops)

        self.total_detections += out.detections
        self.total_redactions += out.redactions
        self.total_flagged += out.flagged_new
        self.total_held += out.held
        self.total_text_redactions += out.text_redactions
        for k, v in out.low_confidence.items():
            self.low_confidence_totals[k] = self.low_confidence_totals.get(k, 0) + v
        return out

    # ------------------------------------------------------------------
    def detect_objects(self, frame: np.ndarray) -> List[dict]:
        """Run only the detector (no tracking, no redaction). Used by the editor preview."""
        height, width = frame.shape[:2]
        return list(self._detect(frame, width, height, FrameOutput(frame=frame)))

    def _detect(self, frame, width, height, out: FrameOutput) -> List[dict]:
        cfg = self.config
        imgsz = cfg.live_imgsz if self.live else cfg.imgsz
        floor = min(cfg.low_floor, cfg.review_min_confidence, cfg.face_min_confidence)
        results = self.model.predict(source=frame, conf=floor, imgsz=imgsz, verbose=False)
        dets: List[dict] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                name = self._class_name(int(box.cls[0].item()), result)
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = clamp_box(box.xyxy[0].cpu().numpy(), width, height)
                if x2 <= x1 or y2 <= y1:
                    continue
                floor_for_class = cfg.face_min_confidence if (name == "face" and self.face_policy is not None) else cfg.review_min_confidence
                if conf < floor_for_class:
                    # LOW: never redacted on its own, but not proof of safety -> counted.
                    if name in cfg.enabled_classes:
                        out.low_confidence[name] = out.low_confidence.get(name, 0) + 1
                    continue
                dets.append({"name": name, "conf": conf, "box": (x1, y1, x2, y2)})
        self._last_dets = dets
        return dets

    def _redaction_box(self, name, raw, smoothed, width, height):
        box = union_box(raw, smoothed) if self.config.cover_raw_box else tuple(smoothed)
        return pad_box(box, self.config.box_padding.get(name, 0.0), width, height)

    def _new_record(self, name, conf, bbox, frame_number, fps, frame_id) -> dict:
        seconds = int(frame_number / fps) if fps > 0 else 0
        x1, y1, x2, y2 = (int(v) for v in bbox)
        return {
            "id": frame_id, "label": name, "category": name, "classification": name,
            "frame": frame_number,
            "time": f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}",
            "bbox": [x1, y1, x2, y2], "confidence": conf, "status": "ignored",
            "source": self.source_name, "redaction_method": normalize_method(self.config.redaction_method),
        }

    def _crop_path(self, frame_number: int) -> Path:
        return self.review_dir / f"{frame_number}_{len(self.records)}.jpg"

    def _handle_review(self, record, label, bbox, frame, frame_number, out: FrameOutput) -> None:
        """MEDIUM confidence: queue for human review, de-duplicating by track."""
        x1, y1, x2, y2 = (int(v) for v in bbox)
        track = self._find_review_track(label, (x1, y1, x2, y2))
        can_save = bool(self.review_dir and self.config.save_review_crops and not self.live)

        def save_crop() -> Optional[str]:
            if not can_save:
                return None
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                return None
            path = self._crop_path(frame_number)
            cv2.imwrite(str(path), crop)
            return str(path)

        if track is None:
            record["status"] = "pending"
            record["track_observations"] = [{"frame": frame_number, "bbox": [x1, y1, x2, y2],
                                             "redact_bbox": list(record.get("redact_bbox", [x1, y1, x2, y2]))}]
            image = save_crop()
            if image:
                record["image_path"] = image
            self.review_tracks.append({"label": label, "bbox": (x1, y1, x2, y2), "run": self._run,
                                       "record_index": len(self.records)})
            out.flagged_new += 1
            return

        review_record = self.records[track["record_index"]]
        review_record.setdefault("track_observations", []).append(
            {"frame": frame_number, "bbox": [x1, y1, x2, y2],
             "redact_bbox": list(record.get("redact_bbox", [x1, y1, x2, y2]))}
        )
        if record["confidence"] > review_record["confidence"]:
            old = review_record.get("image_path")
            image = save_crop()
            review_record.update({"frame": frame_number, "time": record["time"], "bbox": record["bbox"],
                                  "confidence": record["confidence"]})
            if image:
                review_record["image_path"] = image
                if old:
                    Path(old).unlink(missing_ok=True)
        track["bbox"], track["run"] = (x1, y1, x2, y2), self._run
        record["status"] = "tracked_duplicate"

    # ------------------------------------------------------------------
    def _text_stage(self, frame, frame_number, fps, candidates, ops, out: FrameOutput) -> None:
        stage = self.text_stage
        # OCR the largest regions first; ignore regions fully contained in an already-processed one.
        done: List[Tuple[int, int, int, int]] = []
        for region, source_class in sorted(candidates, key=lambda c: -((c[0][2] - c[0][0]) * (c[0][3] - c[0][1]))):
            if any(r[0] <= region[0] and r[1] <= region[1] and r[2] >= region[2] and r[3] >= region[3] for r in done):
                continue
            done.append(region)
            frame_id = f"{self.source_name}:{frame_number}:text"
            findings = stage.process(frame, region, source_class, self._run, frame_id, stats=self.text_stats)
            for finding in findings:
                x1, y1, x2, y2 = clamp_box(finding.bbox, frame.shape[1], frame.shape[0])
                if finding.tier == DecisionTier.AUTO_REDACT:
                    ops.append(((x1, y1, x2, y2), "sensitive_text"))
                    out.text_redactions += 1
                    if finding.new:
                        record = self._new_record("sensitive_text", finding.ocr_confidence, (x1, y1, x2, y2), frame_number, fps,
                                                  f"{self.source_name}:{frame_number}:t{len(self.records)}")
                        record.update({"status": "auto_redacted", "classification": ",".join(finding.patterns),
                                       "category": "sensitive_text", "source_class": finding.source_class,
                                       "ocr_patterns": finding.patterns, "ocr_text_length": finding.text_length,
                                       "redact_bbox": [x1, y1, x2, y2]})
                        self.records.append(record)
                        out.records.append(record)
                elif finding.tier == DecisionTier.FLAGGED and finding.new:
                    record = self._new_record("sensitive_text", finding.ocr_confidence, (x1, y1, x2, y2), frame_number, fps,
                                              f"{self.source_name}:{frame_number}:t{len(self.records)}")
                    record.update({"classification": ",".join(finding.patterns), "category": "sensitive_text",
                                   "source_class": finding.source_class, "ocr_patterns": finding.patterns,
                                   "ocr_text_length": finding.text_length, "redact_bbox": [x1, y1, x2, y2]})
                    self._handle_review(record, "sensitive_text", (x1, y1, x2, y2), frame, frame_number, out)
                    self.records.append(record)
                    out.records.append(record)

    # ------------------------------------------------------------------
    def summary(self) -> dict:
        ts = self.text_stats
        return {
            "detections": self.total_detections,
            "redactions": self.total_redactions,
            "flagged": self.total_flagged,
            "held_redactions": self.total_held,
            "text_redactions": self.total_text_redactions,
            "low_confidence_unreviewed": dict(self.low_confidence_totals),
            "ocr": {
                "enabled": bool(self.text_stage is not None),
                "available": bool(self.text_stage is not None and self.text_stage.available),
                "lines_found": ts.lines_found, "lines_read": ts.lines_read, "lines_cached": ts.lines_cached,
                "lines_deferred": ts.lines_deferred, "uncertain_lines": ts.uncertain,
                "too_small": ts.too_small, "errors": ts.ocr_errors,
            },
        }
