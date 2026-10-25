"""Shared pipeline configuration and class-name handling.

Single source of truth for:
  * which model class NAMES PrivARcy cares about,
  * the confidence policy (HIGH / MEDIUM / LOW),
  * every tunable the video pipeline and the live session share.

IMPORTANT - class mapping
-------------------------
The shipped ``yolo_detection/best.pt`` was trained with SIX classes::

    0 id_card   1 document   2 screen   3 face   4 credit_card   5 license_plate

which differs from the five-class order in the project brief
(id_card, credit_card, screen, face, license_plate). The pipeline therefore
NEVER relies on numeric class ids: it always resolves the class NAME from the
loaded model, and ``verify_class_map`` reports any mismatch. ``document`` is
not redacted as a whole region; it is only used as a *candidate region for
OCR text localisation*.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

# ----------------------------------------------------------------------------
# Class sets
# ----------------------------------------------------------------------------
INTENDED_CLASSES: Tuple[str, ...] = ("id_card", "credit_card", "screen", "face", "license_plate")
SENSITIVE_CLASSES: Set[str] = set(INTENDED_CLASSES)
# Regions in which text is localised and read (never redacted wholesale as 'document').
OCR_CONTEXT_CLASSES: Set[str] = {"document", "id_card", "credit_card", "screen"}
LEGACY_IGNORED_CLASSES: Set[str] = {"document"}

# ----------------------------------------------------------------------------
# Confidence policy (preserved from the original pipeline)
# ----------------------------------------------------------------------------
AUTO_REDACT_CONFIDENCE = 0.80   # HIGH   : >= 0.80 -> automatically redact
REVIEW_MIN_CONFIDENCE = 0.50    # MEDIUM : >= 0.50 and < 0.80 -> human review
                                # LOW    : <  0.50 -> never redacted on its own

REDACTION_METHODS = ("Blur", "Pixelate", "Solid Mask")


def normalize_method(method: Optional[str]) -> str:
    """'blur' / 'SOLID MASK' / 'solid_mask' -> canonical 'Blur' / 'Solid Mask'.

    The old default was lowercase 'blur' while the pipeline
    compared against 'Blur', so 'pixelate' / 'solid mask' silently fell back
    to blur. Always normalise here.
    """
    if not method:
        return "Blur"
    key = str(method).strip().lower().replace("_", " ").replace("-", " ")
    for canonical in REDACTION_METHODS:
        if canonical.lower() == key:
            return canonical
    if key in ("gaussian blur", "gaussian"):
        return "Blur"
    if key in ("solid", "mask", "black", "solid mask"):
        return "Solid Mask"
    return "Blur"


def model_class_names(model) -> Dict[int, str]:
    names = getattr(model, "names", None)
    if isinstance(names, dict):
        return {int(k): str(v).strip().lower() for k, v in names.items()}
    if isinstance(names, (list, tuple)):
        return {i: str(v).strip().lower() for i, v in enumerate(names)}
    return {}


def verify_class_map(names: Dict[int, str]) -> dict:
    """Compare the loaded model's classes against the intended set (by NAME)."""
    present = set(names.values())
    missing = sorted(set(INTENDED_CLASSES) - present)
    extra = sorted(present - set(INTENDED_CLASSES))
    intended_order = {n: i for i, n in enumerate(INTENDED_CLASSES)}
    order_differs = any(
        name in intended_order and intended_order[name] != idx for idx, name in names.items()
    )
    warnings: List[str] = []
    if missing:
        warnings.append(f"Model has no class(es) {missing}; those objects cannot be detected by YOLO.")
    if extra:
        warnings.append(
            f"Model has extra class(es) {extra}. 'document' is used only as an OCR region, never redacted whole."
        )
    if order_differs:
        warnings.append("Model class ORDER differs from the intended 5-class order; names are used, ids are ignored.")
    return {
        "names": dict(sorted(names.items())),
        "missing": missing,
        "extra": extra,
        "order_differs": order_differs,
        "ok": not missing,
        "warnings": warnings,
    }


@dataclass
class PipelineConfig:
    """All tunables for offline video processing and live monitoring."""

    # --- confidence policy -------------------------------------------------
    auto_redact_confidence: float = AUTO_REDACT_CONFIDENCE
    review_min_confidence: float = REVIEW_MIN_CONFIDENCE
    # Per-class overrides produced ONLY by validated, approved rule versions
    # (see privarcy.rules). Empty by default => policy unchanged.
    class_auto_redact_confidence: Dict[str, float] = field(default_factory=dict)
    # Detections between low_floor and review_min are *counted* (uncertainty
    # warning) but never redacted or reviewed on their own.
    low_floor: float = 0.25

    # --- tracking / smoothing ---------------------------------------------
    iou_threshold: float = 0.50
    track_max_gap: int = 5          # a track expires after 5 unmatched detection runs
    temporal_smoothing: int = 3     # EMA window (alpha = 2 / (n + 1))
    hold_missed_detections: bool = True   # keep redacting an auto-redacted track until it expires
    cover_raw_box: bool = True      # redact union(raw, smoothed) so EMA lag can never expose a moving object
    box_padding: Dict[str, float] = field(default_factory=lambda: {"face": 0.10, "license_plate": 0.05})

    # --- detection ---------------------------------------------------------
    imgsz: int = 1280
    live_imgsz: int = 640
    detection_interval: int = 1     # live: run YOLO every N frames (tracks are held in between)
    enabled_classes: Set[str] = field(default_factory=lambda: set(SENSITIVE_CLASSES))
    redaction_method: str = "Blur"

    # --- faces -------------------------------------------------------------
    # Faces at or above this confidence are handled by the face policy (auto / selected / every) instead of the
    # normal review band, so "blur every face" really means every detected face.
    face_min_confidence: float = 0.35
    redaction_strength: Optional[float] = None     # None = original look; 0..1 = blur / block size

    # --- OCR / sensitive text ---------------------------------------------
    ocr_enabled: bool = True
    ocr_model: str = "microsoft/trocr-base-stage1"
    ocr_max_new_lines_per_frame: int = 4
    ocr_min_line_height: int = 8
    # 'redact_all' : any detected screen is redacted (conservative default)
    # 'ocr_only'   : if OCR is available, redact only sensitive text lines on screens
    screen_policy: str = "redact_all"
    live_ocr_enabled: bool = False

    # --- storage -----------------------------------------------------------
    save_review_crops: bool = True

    def copy(self) -> "PipelineConfig":
        return copy.deepcopy(self)

    def auto_threshold_for(self, class_name: str) -> float:
        value = self.class_auto_redact_confidence.get(class_name, self.auto_redact_confidence)
        # a per-class rule can never fall to / below the review floor
        return max(self.review_min_confidence + 0.05, min(self.auto_redact_confidence, float(value)))

    def update(self, **values) -> None:
        for key, value in values.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.redaction_method = normalize_method(self.redaction_method)
        self.enabled_classes = set(self.enabled_classes)


def box_iou(a: Iterable[float], b: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1) + max(0.0, bx2 - bx1) * max(0.0, by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def clamp_box(box, width: int, height: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    return (
        max(0, min(x1, width - 1)),
        max(0, min(y1, height - 1)),
        max(0, min(x2, width)),
        max(0, min(y2, height)),
    )


def pad_box(box, fraction: float, width: int, height: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    px = (x2 - x1) * fraction
    py = (y2 - y1) * fraction
    return clamp_box((x1 - px, y1 - py, x2 + px, y2 + py), width, height)


def union_box(a, b) -> Tuple[int, int, int, int]:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))
