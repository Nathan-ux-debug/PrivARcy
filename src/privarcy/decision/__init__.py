"""
decision_engine

Combines yolo_detection's class/confidence, ocr's extracted text (PII
regex checks), and face_privacy's known/unknown match result into a
single routing decision per detected region, using configurable 3-tier
confidence gating:

    confidence >= T_high              -> AUTO_REDACT
    T_low <= confidence < T_high       -> FLAGGED   (human review queue)
    confidence < T_low                    -> PASS_THROUGH  (ignored)

Public API:
    DecisionEngine       - the rule engine: combine inputs -> route a detection
    DecisionInput           - everything the engine needs about one detection
    Decision                   - the routing result (tier + reasons)
    DecisionTier                   - AUTO_REDACT / FLAGGED / PASS_THROUGH enum
    DecisionThresholds                 - T_high / T_low configuration
"""

from .thresholds import DecisionThresholds
from .models import DecisionInput, Decision, DecisionTier
from .rules import PII_PATTERNS, contains_pii, find_sensitive_spans
from .engine import DecisionEngine

__all__ = [
    "DecisionEngine",
    "DecisionInput",
    "Decision",
    "DecisionTier",
    "DecisionThresholds",
    "PII_PATTERNS",
    "contains_pii",
    "find_sensitive_spans",
]

__version__ = "1.0.0"
