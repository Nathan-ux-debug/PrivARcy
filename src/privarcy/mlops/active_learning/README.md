# active_learning

Closes the loop `decision_engine`'s `FLAGGED` tier already opened. When a
human reviewer confirms a flagged detection was genuinely sensitive (via
`data_store.CorrectionLogDB`, D4), this package turns that correction
into a properly-formatted labeled training example — ready to feed
straight back into `dataset_pipeline` for a re-training round.

```
decision_engine (FLAGGED) → human review → data_store D4 (corrected)
    → active_learning (this package) → dataset_pipeline (raw input)
    → re-trained model.pt → back into yolo_detection
```

## Why this counts as active learning

Active learning's core idea is **uncertainty sampling**: don't randomly
pick what to label next, pick the examples the model is least sure
about, since those teach it the most per unit of labeling effort. Your
3-tier confidence gate already does exactly this:

```
confidence >= T_high      → AUTO_REDACT   (model is confident — skip)
T_low <= confidence < T_high → FLAGGED       (model is UNSURE — ask a human)
confidence < T_low             → PASS_THROUGH    (model is confident — skip)
```

`FLAGGED` **is** the uncertainty-sampling query strategy. This package is
the other half: taking the human's answer and turning it into new
training signal, closing the loop from "the model wasn't sure" all the
way to "the model just got better at exactly the kind of case it wasn't
sure about."

## Usage

```bash
python -m active_learning.cli \
    --corrections-db correction_log.db --frames-dir frames_out \
    --classes id_card,document,screen,face,credit_card,license_plate \
    --output-dir retrain_batch
```

This writes `retrain_batch/images/` and `retrain_batch/labels/` in
standard YOLO format. Feed that straight into `dataset_pipeline`:

```python
from dataset_pipeline import DatasetSplitter, LocalDirectorySource, ClassMap

class_map = ClassMap(names=["id_card", "document", "screen", "face", "credit_card", "license_plate"])
source = LocalDirectorySource("retrain_batch/images", "retrain_batch/labels")
DatasetSplitter(source, "retrain_split", class_map).run()
# -> retrain_split/data.yaml, ready for model.train(data=...)
```

Verified end-to-end with a real integration run: 10 simulated corrections
→ exported → split → produced a real, valid `data.yaml` — not just
isolated unit tests.

**Important:** pass the exact same `ClassMap` (same names, same order)
you used for your main annotated dataset. `CorrectionExporter` raises
`ActiveLearningError` immediately if a correction references a class
that isn't in the map you gave it — this is a deliberate hard stop,
because a silent class-id mismatch between your original dataset and
this supplemental batch would corrupt training without any visible error.

You'll typically **merge** this exported batch into your main raw
dataset folder (copy both into one `images/`+`labels/` folder) rather
than training on it alone — it's a small, targeted supplement of your
model's hardest cases, not a replacement for your main annotated set.

## Why rejected corrections aren't auto-included as labels

A `rejected` correction means a reviewer said "this flagged box was
wrong, this isn't actually a document/card/whatever." That's valuable
information, but it's incomplete: it tells you this *one region* isn't
the class the model thought it was — it says nothing about whether the
model exists correctly annotated for the rest of that image. Writing an
empty or partial label file for that frame would silently teach the
retrained model an incomplete picture of everything else in the image.

So rejected corrections are collected into `ExportStats.hard_negatives`
and written to `hard_negatives_report.json` instead — a worklist for a
human to review (and properly re-annotate, if worth including), not
something auto-converted into potentially-wrong training labels.

## A privacy note worth repeating here specifically

The images accumulating in your correction log are, by construction,
the most sensitive ones in your whole system — real credit cards, real
documents, real children's faces the model wasn't sure about. This
export step concentrates exactly that material into one retraining
batch. Treat `retrain_batch/` with the same (or greater) care as your
original raw dataset — same access controls, same retention
consideration, not just "it's a training folder like any other."

## Install

Depends on `data_store` and `dataset_pipeline` (reuses `ClassMap` from
the latter directly, so class-id numbering can never drift between the
two). No other dependencies beyond `opencv-python`.

## Tests

```bash
python -m unittest active_learning.tests.test_active_learning -v
```

8 tests covering: correct label export, exact YOLO coordinate
normalization math, multiple corrections on one frame merging into a
single label file, rejected corrections routed to hard negatives (not
labels), missing-image handling (skipped, not fatal), the substring
frame-id matching needed for `frame_extraction`'s real filename
convention, and class-map mismatch detection.
