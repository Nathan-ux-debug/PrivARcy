"""Turn logged reviewer decisions into rule proposals - only when evidence supports it."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Dict, List

from .rules import AUTO_MAX, MIN_STEP_ABOVE_FLOOR, REVIEW_FLOOR

MIN_SAMPLES = 10          # reviewed detections in the band, per class
MIN_ACCEPT_RATE = 0.95    # share confirmed by reviewers
BAND_LOW, BAND_HIGH = REVIEW_FLOOR, AUTO_MAX


@dataclass
class Evidence:
    total_records: int = 0
    unusable_records: int = 0                       # no confidence / no class -> cannot be used
    per_class: Dict[str, dict] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Proposal:
    class_name: str
    current_threshold: float
    proposed_threshold: float
    samples: int
    accept_rate: float
    reason: str

    def summary(self) -> str:
        return f"{self.class_name}: auto-redact threshold {self.current_threshold:.2f} -> {self.proposed_threshold:.2f}"

    def to_dict(self) -> dict:
        return asdict(self)


def _usable(rec) -> bool:
    return bool(rec.detected_class) and rec.confidence and rec.confidence > 0 and rec.decision in ("accepted", "rejected")


def evidence_from_log(correction_db, limit: int = 100000) -> Evidence:
    records = correction_db.get_decisions(limit=limit)
    ev = Evidence(total_records=len(records))
    for rec in records:
        if not _usable(rec):
            ev.unusable_records += 1
            continue
        cls = rec.detected_class.split(":")[0]
        d = ev.per_class.setdefault(cls, {"in_band": [], "accepted": 0, "rejected": 0})
        d["accepted" if rec.decision == "accepted" else "rejected"] += 1
        if BAND_LOW <= rec.confidence < BAND_HIGH:
            d["in_band"].append((float(rec.confidence), rec.decision == "accepted"))
    if ev.unusable_records:
        ev.notes.append(
            f"{ev.unusable_records} of {ev.total_records} logged decisions have no confidence or class "
            "(written by an older build) and are ignored as evidence."
        )
    return ev


def propose_rules(evidence: Evidence, current: Dict[str, float]) -> List[Proposal]:
    proposals: List[Proposal] = []
    for cls, d in sorted(evidence.per_class.items()):
        band = d["in_band"]
        if len(band) < MIN_SAMPLES:
            continue
        accepted = [c for c, ok in band if ok]
        rate = len(accepted) / len(band)
        if rate < MIN_ACCEPT_RATE:
            continue
        lowest = min(accepted)
        # Propose a threshold at the lowest confidence the reviewers still confirmed, rounded DOWN to 0.05,
        # and never lower than the floor + 0.05 step.
        proposed = max(REVIEW_FLOOR + MIN_STEP_ABOVE_FLOOR, math.floor(lowest * 20) / 20.0)
        now = current.get(cls, AUTO_MAX)
        if proposed >= now - 1e-9:
            continue
        proposals.append(Proposal(
            class_name=cls, current_threshold=now, proposed_threshold=round(proposed, 2), samples=len(band),
            accept_rate=rate,
            reason=f"{len(accepted)}/{len(band)} reviewed detections between 0.50 and 0.80 were confirmed.",
        ))
    return proposals
