"""Validate a proposal against the evidence BEFORE it can be activated."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import List

from .proposals import BAND_HIGH, BAND_LOW, Evidence, Proposal
from .rules import check_invariants


@dataclass
class ValidationReport:
    passed: bool
    checks: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def validate_proposal(proposal: Proposal, evidence: Evidence, current_overrides: dict) -> ValidationReport:
    checks: List[dict] = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(ok), "detail": detail})

    problems = check_invariants({proposal.class_name: proposal.proposed_threshold})
    check("policy invariants", not problems, "; ".join(problems) or "threshold within [0.55, 0.80]; floor 0.50 untouched")
    check("only lowers threshold", proposal.proposed_threshold < proposal.current_threshold,
          f"{proposal.current_threshold:.2f} -> {proposal.proposed_threshold:.2f} (more redaction, never less)")

    band = evidence.per_class.get(proposal.class_name, {}).get("in_band", [])
    newly_auto = [(c, ok) for c, ok in band if proposal.proposed_threshold <= c < BAND_HIGH]
    false_pos = [1 for _c, ok in newly_auto if not ok]
    check("evidence size", len(band) >= 10, f"{len(band)} reviewed detections in band")
    fp_rate = (len(false_pos) / len(newly_auto)) if newly_auto else 0.0
    check("replay: newly auto-redacted items were confirmed", fp_rate <= 0.05,
          f"{len(newly_auto)} reviewed items would now be auto-redacted; {len(false_pos)} were rejected by reviewers ({fp_rate:.0%})")
    lost = [1 for c, ok in band if ok and c < BAND_LOW]
    check("replay: no confirmed item becomes un-redacted", not lost, "confirmed detections stay redacted or reviewed")
    return ValidationReport(passed=all(c["passed"] for c in checks), checks=checks)
