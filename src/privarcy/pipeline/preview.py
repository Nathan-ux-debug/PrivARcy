"""Instant editor preview.

Detecting is the slow part (one YOLO pass); deciding and drawing the blur is cheap. So the editor analyses a frame ONCE
(``PreviewEngine.analyze``) and then re-renders it every time an option changes (``render``) without running the model
again. ``render`` follows the same rules as the processing pass - same class thresholds, same face policy, same padding,
same redaction functions - so what the preview shows is what the output will contain (apart from frame-to-frame smoothing).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .config import PipelineConfig
from .frame_engine import FrameEngine
from .magic import MagicTracker, colour_signature
from .plan import EditPlan, MagicTarget
from .redact_ops import redact_mask, redact_region


@dataclass
class DetectedObject:
    index: int
    name: str
    conf: float
    box: Tuple[int, int, int, int]
    face_id: Optional[str] = None
    face_label: str = ""


@dataclass
class FrameAnalysis:
    frame_number: int
    frame: np.ndarray
    objects: List[DetectedObject] = field(default_factory=list)

    def object_at(self, x: float, y: float) -> Optional[DetectedObject]:
        """Smallest detected object containing the point (so a face inside a screen is picked before the screen)."""
        hits = [o for o in self.objects if o.box[0] <= x <= o.box[2] and o.box[1] <= y <= o.box[3]]
        return min(hits, key=lambda o: (o.box[2] - o.box[0]) * (o.box[3] - o.box[1]), default=None)


@dataclass
class Overlay:
    kind: str                         # face | object | review | region | magic
    box: Tuple[int, int, int, int]
    label: str
    state: str                        # blurred | visible | review
    face_id: Optional[str] = None


@dataclass
class RenderResult:
    image: np.ndarray
    overlays: List[Overlay]
    redactions: int


class PreviewEngine:
    def __init__(self, model, config: PipelineConfig, face_policy=None, review_dir=None) -> None:
        self.config = config
        self.face_policy = face_policy
        self.engine = FrameEngine(model, config, source_name="preview", face_policy=face_policy, review_dir=review_dir)

    def analyze(self, frame: np.ndarray, frame_number: int = 0) -> FrameAnalysis:
        analysis = FrameAnalysis(frame_number, frame.copy())
        for i, det in enumerate(self.engine.detect_objects(frame)):
            obj = DetectedObject(i, det["name"], float(det["conf"]), tuple(int(v) for v in det["box"]))
            if obj.name == "face" and self.face_policy is not None:
                obj.face_id = self.face_policy.identify(frame, obj.box, frame_number)
                ident = self.face_policy.book.get(obj.face_id) if obj.face_id else None
                obj.face_label = ident.label if ident else "Unidentified face"
            analysis.objects.append(obj)
        return analysis

    def magic_target_at(self, analysis: FrameAnalysis, x: float, y: float) -> Optional[MagicTarget]:
        """Turn a click into a Magic Blur target (None when the click is not on something the model detected)."""
        obj = analysis.object_at(x, y)
        if obj is None or obj.name == "document":
            return None
        label = obj.face_label if obj.name == "face" and obj.face_label else obj.name.replace("_", " ")
        return MagicTarget(obj.name, obj.box, colour_signature(analysis.frame, obj.box), label)

    def render(self, analysis: FrameAnalysis, plan: Optional[EditPlan] = None, seconds: Optional[float] = None) -> RenderResult:
        cfg = self.config
        frame = analysis.frame.copy()
        h, w = frame.shape[:2]
        overlays: List[Overlay] = []
        ops: List[Tuple[int, int, int, int]] = []
        masks: List[np.ndarray] = []

        for obj in analysis.objects:
            if obj.name == "document" or obj.name not in cfg.enabled_classes:
                continue
            box = self.engine._redaction_box(obj.name, obj.box, obj.box, w, h)
            if obj.name == "face" and self.face_policy is not None:
                blur, _ = self.face_policy.decide(obj.face_id)
                overlays.append(Overlay("face", box, obj.face_label or "Face", "blurred" if blur else "visible", obj.face_id))
                if blur:
                    ops.append(box)
            elif obj.conf >= cfg.auto_threshold_for(obj.name):
                ops.append(box)
                overlays.append(Overlay("object", box, obj.name.replace("_", " "), "blurred"))
            elif obj.conf >= cfg.review_min_confidence:
                overlays.append(Overlay("review", box, f"{obj.name.replace('_', ' ')} · needs review", "review"))

        if plan is not None:
            if plan.magic:
                tracker = MagicTracker(plan.magic)
                dets = [{"name": o.name, "box": o.box} for o in analysis.objects]
                if dets:
                    tracker.update(analysis.frame, dets, 1)
                masks.extend(tracker.current_masks())
                for target in plan.magic:
                    overlays.append(Overlay("magic", tuple(target.box), "Magic · " + (target.label or target.name), "blurred"))
            for region in plan.manual:
                if seconds is not None and not region.active(seconds):
                    continue
                x1, y1, x2, y2 = region.box(w, h)
                if x2 <= x1 or y2 <= y1:
                    continue
                overlays.append(Overlay("region", (x1, y1, x2, y2), region.label, "blurred"))
                if region.shape == "ellipse":
                    m = np.zeros((h, w), np.uint8)
                    cv2.ellipse(m, ((x1 + x2) // 2, (y1 + y2) // 2), ((x2 - x1) // 2, (y2 - y1) // 2), 0, 0, 360, 255, -1)
                    masks.append(m)
                else:
                    ops.append((x1, y1, x2, y2))

        for x1, y1, x2, y2 in ops:
            if x2 > x1 and y2 > y1:
                redact_region(frame, x1, y1, x2, y2, cfg.redaction_method, cfg.redaction_strength)
        for mask in masks:
            redact_mask(frame, mask, cfg.redaction_method, cfg.redaction_strength)
        return RenderResult(frame, overlays, len(ops) + len(masks))
