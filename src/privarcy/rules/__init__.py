"""Controlled correction -> rule refinement.

Three DIFFERENT things, deliberately kept apart:

  1. Correction LOGGING   - every reviewer decision is stored (data_store.CorrectionLogDB, D4).
  2. Rule REFINEMENT      - this package: evidence-based, validated, approved, versioned,
                            reversible changes to a small set of decision parameters.
  3. Model RETRAINING     - NOT done here. Corrections never change YOLO weights.

What a refinement can change: ONLY a per-class auto-redact confidence threshold,
and ONLY downwards (more redaction, less manual review) for a class whose
reviewed detections in the 0.50-0.80 band were accepted at a very high rate.
It can never raise a threshold, never disable a class and never touch the
0.50 review floor. See rules.py for the exact conditions.
"""

from .rules import RuleStore, RuleVersion
from .proposals import Evidence, Proposal, propose_rules, evidence_from_log
from .validation import ValidationReport, validate_proposal

__all__ = [
    "RuleStore", "RuleVersion", "Evidence", "Proposal", "propose_rules", "evidence_from_log",
    "ValidationReport", "validate_proposal",
]
