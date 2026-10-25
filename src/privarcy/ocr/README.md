# ocr

Reads text out of the cropped regions `yolo_detection` already found
(IDs, documents, screens) using **TrOCR** (Hugging Face `transformers`),
and returns the recognized text mapped back to its frame and bounding box.

```
yolo_detection detections (bbox + class) → ocr (this package) → text + confidence,
    linked to frame_id + bbox → decision_engine (regex/PII rules)
```

---

## 1. Layout

```
ocr/
├── __init__.py       # public API
├── exceptions.py       # OCRError
├── models.py             # OCRResult — the output contract
├── engine.py               # TrOCREngine — loads TrOCR, crops, runs inference
├── cli.py                    # runs OCR over yolo_detection's JSON output
└── tests/
    └── test_ocr.py
```

## 2. How it works

```python
from ocr import TrOCREngine

engine = TrOCREngine(model_name="microsoft/trocr-base-stage1")
result = engine.read_region(frame, bbox=(120, 55, 310, 90), frame_id=42, source_class="document")
print(result.text, result.confidence)
```

1. `bbox` is clamped to the frame's actual bounds (a box that runs
   slightly outside the frame doesn't crash) — a fully-outside or
   zero-area box raises `OCRError` instead of silently returning garbage.
2. The region is cropped, converted BGR→RGB, and wrapped as a PIL image
   (what `TrOCRProcessor` expects).
3. `model.generate(..., output_scores=True, return_dict_in_generate=True)`
   runs the actual OCR, returning both the decoded text and per-token
   generation scores.
4. **Confidence** is derived from those scores: the average, across every
   generated token, of that token's own softmax probability (i.e. how
   "sure" the model was about each character/word it picked). This is
   *not* a calibrated probability of correctness — it's a relative
   signal. A low value reliably means the model struggled; a high value
   means it was confident, but confidence isn't the same as accuracy.
5. Result is an `OCRResult`: text, confidence, the *original* frame's
   bbox coordinates (not the crop's), and the `frame_id`/`source_class`
   you passed in, so it can be traced back to exactly where it came from.

`read_regions()` is the batch version — feed it every detection from a
`yolo_detection.DetectionResult` in one call; a single bad region
(degenerate bbox, inference error) is skipped with a logged warning
rather than losing every other reading from that frame.

## 3. CLI — chains directly onto yolo_detection's output

```bash
python -m ocr.cli \
    --frames-dir frames_out --detections-dir detections_out \
    --output-dir ocr_out --classes document,credit_card --conf 0.3
```

For every `detections_out/*.json` file (written by `yolo_detection.cli`),
finds the matching frame image, crops every detection whose class is in
`--classes` (omit to run OCR on every detection regardless of class) and
whose confidence clears `--min-detection-confidence`, runs TrOCR on each,
and writes one JSON per frame with all the recognized text.

## 4. Install

```bash
pip install transformers torch pillow
```

> **Note on this development environment:** `transformers` + `torch` +
> TrOCR's model weights is a multi-GB download — the same disk
> constraint already hit with `ultralytics` (7.4GB free in this
> sandbox). `TrOCREngine` accepts an injectable model loader (the same
> pattern used for `YOLODetector`), and all 12 tests run against a fake
> processor/model that mimics TrOCR's real call shape (`processor(images=...)`,
> `model.generate(..., output_scores=True, return_dict_in_generate=True)`,
> `processor.batch_decode(...)`) — covering crop/bbox validation, batch
> skip-on-error behavior, confidence derivation, and JSON serialization.
> The actual model load and a real forward pass are untested here —
> recommend a smoke test against a real image with real text once
> installed in an environment with enough disk space.

## 5. Tests

```bash
python -m unittest ocr.tests.test_ocr -v
```

12 tests covering: loader-failure wrapping, CPU device fallback,
correct text/confidence extraction, invalid-frame and degenerate/
out-of-bounds bbox handling (including the "partially outside, should
clamp not reject" case), inference-failure wrapping, batch behavior
(all-valid and skip-bad-continue), and `to_dict()`/`to_json()` shape.
