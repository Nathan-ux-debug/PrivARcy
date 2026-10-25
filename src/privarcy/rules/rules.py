"""Versioned rule store with audit trail and rollback.

Layout (<data dir>/rules/):
    active.json                 pointer + the active rule set
    versions/v0001.json ...     immutable history (never deleted)
    audit.jsonl                 append-only log: who/when/what/why

Version 0 is the built-in policy (no overrides). Rollback never deletes
history: it writes a NEW version that copies an older one, so the audit trail
stays linear and complete.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

AUTO_MAX = 0.80
REVIEW_FLOOR = 0.50
MIN_STEP_ABOVE_FLOOR = 0.05


@dataclass
class RuleVersion:
    version: int
    class_auto_redact_confidence: Dict[str, float] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    created_by: str = "system"
    note: str = ""
    parent: Optional[int] = None
    evidence: dict = field(default_factory=dict)
    validation: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def check_invariants(overrides: Dict[str, float]) -> List[str]:
    """Policy invariants every rule set must satisfy; returns violations."""
    problems = []
    for cls, value in overrides.items():
        if not (REVIEW_FLOOR + MIN_STEP_ABOVE_FLOOR - 1e-9 <= value <= AUTO_MAX + 1e-9):
            problems.append(f"{cls}: threshold {value:.2f} outside [{REVIEW_FLOOR + MIN_STEP_ABOVE_FLOOR:.2f}, {AUTO_MAX:.2f}]")
    return problems


class RuleStore:
    def __init__(self, directory: Path | str) -> None:
        self.dir = Path(directory)
        self.versions_dir = self.dir / "versions"
        self.audit_path = self.dir / "audit.jsonl"
        self.active_path = self.dir / "active.json"
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        if not self.active_path.exists():
            self._write_version(RuleVersion(version=0, note="Built-in policy (HIGH>=0.80 auto, MEDIUM 0.50-0.80 review)"))
            self._set_active(0)
            self._audit("init", 0, "system", "Created rule store at built-in policy.")

    # ---------------------------------------------------------------- reading
    def active_version_number(self) -> int:
        return int(json.loads(self.active_path.read_text(encoding="utf-8"))["active"])

    def get(self, version: int) -> RuleVersion:
        return RuleVersion(**json.loads((self.versions_dir / f"v{version:04d}.json").read_text(encoding="utf-8")))

    def active(self) -> RuleVersion:
        return self.get(self.active_version_number())

    def history(self) -> List[RuleVersion]:
        out = [RuleVersion(**json.loads(p.read_text(encoding="utf-8"))) for p in sorted(self.versions_dir.glob("v*.json"))]
        return out

    def audit(self, limit: int = 100) -> List[dict]:
        if not self.audit_path.exists():
            return []
        lines = self.audit_path.read_text(encoding="utf-8").splitlines()
        return [json.loads(l) for l in lines[-limit:]][::-1]

    # ---------------------------------------------------------------- writing
    def activate(self, overrides: Dict[str, float], approver: str, note: str, evidence: dict, validation: dict) -> RuleVersion:
        """Create + activate a new version. Refuses anything violating invariants
        or lacking a PASSED validation report."""
        if not validation or not validation.get("passed"):
            raise ValueError("A rule change cannot be activated without a passing validation report.")
        problems = check_invariants(overrides)
        if problems:
            raise ValueError("; ".join(problems))
        parent = self.active_version_number()
        version = RuleVersion(
            version=self._next_number(), class_auto_redact_confidence=dict(overrides), created_by=approver,
            note=note, parent=parent, evidence=evidence, validation=validation,
        )
        self._write_version(version)
        self._set_active(version.version)
        self._audit("activate", version.version, approver, note, {"overrides": overrides, "parent": parent})
        return version

    def rollback(self, to_version: int, approver: str, reason: str = "") -> RuleVersion:
        target = self.get(to_version)
        parent = self.active_version_number()
        version = RuleVersion(
            version=self._next_number(), class_auto_redact_confidence=dict(target.class_auto_redact_confidence),
            created_by=approver, note=f"Rollback to v{to_version}. {reason}".strip(), parent=parent,
        )
        self._write_version(version)
        self._set_active(version.version)
        self._audit("rollback", version.version, approver, version.note, {"restored": to_version, "from": parent})
        return version

    def reject(self, proposal_summary: str, approver: str, reason: str = "") -> None:
        self._audit("reject", self.active_version_number(), approver, f"Rejected proposal: {proposal_summary}. {reason}".strip())

    # -------------------------------------------------------------- internals
    def _next_number(self) -> int:
        return max(v.version for v in self.history()) + 1

    def _write_version(self, version: RuleVersion) -> None:
        (self.versions_dir / f"v{version.version:04d}.json").write_text(json.dumps(version.to_dict(), indent=2), encoding="utf-8")

    def _set_active(self, number: int) -> None:
        self.active_path.write_text(json.dumps({"active": number, "updated_at": time.time()}), encoding="utf-8")

    def _audit(self, action: str, version: int, actor: str, note: str, extra: Optional[dict] = None) -> None:
        entry = {"ts": time.time(), "action": action, "version": version, "actor": actor, "note": note}
        if extra:
            entry["details"] = extra
        with self.audit_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
