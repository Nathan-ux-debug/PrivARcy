# decision_engine

The rule engine that combines everything the other packages found —
`yolo_detection`'s class/confidence, `ocr`'s extracted text (checked for
PII), and `face_privacy`'s known/unknown match — into one routing
decision per detected region, using configurable 3-tier confidence gating.

```
yolo_detection (class + conf) ─┐
ocr (text + conf)              ├─→ decision_engine (this package) ─→ AUTO_REDACT | FLAGGED | PASS_THROUGH
face_privacy (known/unknown) ──┘
```

---

## 1. Layout

```
decision_engine/
├── __init__.py       # public API
├── thresholds.py       # DecisionThresholds — T_high / T_low config
├── models.py             # DecisionInput, Decision, DecisionTier
├── rules.py                # PII regex patterns + contains_pii()
├── engine.py                  # DecisionEngine — the actual routing logic
└── tests/
    └── test_decision_engine.py
```

## 2. The routing rule

```python
from decision_engine import DecisionEngine, DecisionInput, DecisionThresholds

engine = DecisionEngine(DecisionThresholds(t_high=0.85, t_low=0.5))

decision = engine.decide(DecisionInput(
    frame_id="clip1_042",
    bbox=(50, 50, 250, 150),
    detected_class="document", detection_confidence=0.93,
    ocr_text="CARD 4111 1111 1111 1111", ocr_confidence=0.90,
))

print(decision.tier)                 # DecisionTier.AUTO_REDACT
print(decision.combined_confidence)  # 0.93
print(decision.reasons)
# ["class 'document' is sensitive (detector confidence 0.93)",
#  "extracted text matched PII pattern(s): credit_card_number"]
```

```
combined_confidence >= T_high              → AUTO_REDACT     (redact automatically)
T_low <= combined_confidence < T_high        → FLAGGED           (human review queue)
combined_confidence < T_low                     → PASS_THROUGH        (ignored)
```

Verified against a real 4-stage chain, not just isolated unit tests: a
frame extracted from a real video → a (faked) YOLOv8 "document" detection
at 0.93 confidence → a (faked) TrOCR read of "CARD 4111 1111 1111 1111"
→ this engine correctly combined both signals into `auto_redact` with
clear, human-readable reasons for each contributing signal.

### 2.1 How signals combine

Each applicable signal contributes a confidence value; **the combined
score is the maximum of whichever signals apply**, not an average:

- **Object class** (`yolo_detection`): if `detected_class` is in the
  sensitive set (`credit_card`, `document`, `license_plate`, `id_card`
  by default — configurable), its `detection_confidence` counts.
- **PII in OCR'd text** (`ocr`): if `ocr_text` matches any pattern in
  `rules.PII_PATTERNS` (credit card numbers, SSNs, emails, phone
  numbers, passport-like IDs), `ocr_confidence` counts.
- **Unknown face** (`face_privacy`): only for `detected_class == "face"`
  — if `is_known_face is False`, this counts as a sensitive signal too.

**Why max(), not average**: this is a privacy tool. If a document is
detected with 0.95 confidence but OCR couldn't read any text at all
(`ocr_text=None`), the region should probably still get flagged —
averaging a strong class signal against a "no text found" non-signal
shouldn't dilute it down toward pass-through. A single confident signal
is enough to act on; multiple weak-but-present signals reinforcing each
other is a nice-to-have, not a requirement.

### 2.2 The known-face override

```python
DecisionInput(frame_id="f1", bbox=(...), detected_class="face",
               is_known_face=True, face_similarity=0.95)
# -> always PASS_THROUGH, regardless of any other signal
```

If `face_privacy` matched a face to an enrolled person, this engine
**always** passes it through — never flags, never auto-redacts — even if
`detection_confidence` is 1.0. This is the actual mechanism behind "blur
everyone except the people I've registered": `face_privacy` finds faces
and checks them against the registry, and this engine is what turns
"known" into a hard pass-through rather than just another weighted signal.

---

## 3. Configuring thresholds

```python
# Stricter — fewer auto-redactions, more goes to human review
strict = DecisionThresholds(t_high=0.95, t_low=0.4)

# Looser — more gets auto-redacted, less human review burden
lenient = DecisionThresholds(t_high=0.7, t_low=0.3)

engine = DecisionEngine(thresholds=strict, sensitive_classes={"credit_card", "document", "face"})
```

Invalid configurations (e.g. `t_low > t_high`, either outside `[0, 1]`)
raise `ValueError` immediately at construction — never silently produce
a nonsensical routing gate.

## 4. Batch use + grouping for downstream stages

```python
decisions = engine.decide_many([input1, input2, input3])
grouped = DecisionEngine.group_by_tier(decisions)

# grouped["auto_redact"]  -> hand straight to redaction.FrameRedactor
# grouped["flagged"]      -> push to a human review queue / data_store.CorrectionLogDB
# grouped["pass_through"] -> ignore
```

## 5. Install

Pure Python standard library — no dependencies at all.

## 6. Tests

```bash
python -m unittest decision_engine.tests.test_decision_engine -v
```

23 tests, explicitly organized around the ticket's required cases:
**high confidence** (sensitive class alone, PII text alone, unknown face
alone — each independently sufficient to auto-redact), **medium
confidence** (including the exact boundary at `t_low` and just below
`t_high`), **low confidence** (sensitive class but low confidence,
non-sensitive class regardless of confidence, no signals at all, just
below `t_low`), plus the known-face override, threshold validation,
PII pattern matching, batch/grouping, and JSON serialization.
