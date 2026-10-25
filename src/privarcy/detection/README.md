# yolo_detection

Wraps a pre-trained **YOLOv8** model (via [Ultralytics](https://docs.ultralytics.com/))
to run object detection on frames — including frames coming straight out
of the `frame_extraction` package (Ticket 1) — and returns clean,
JSON-serializable bounding boxes + class labels + confidence scores.

This is Ticket 2 of the pipeline:

```
MP4 upload → frame_extraction → sampled NumPy frames → yolo_detection → JSON detections → API / UI
```

---

## 1. Layout

```
yolo_detection/
├── __init__.py       # public API exports
├── exceptions.py      # ModelLoadError, InferenceError
├── models.py           # Detection & DetectionResult dataclasses (the output contract)
├── device.py             # CUDA-if-available / CPU-fallback logic, isolated & testable
├── detector.py             # YOLODetector — loads the model, runs inference, parses output
├── cli.py                   # command-line entry point (frames folder OR full video)
└── tests/
    └── test_yolo_detector.py
```

### Why split like this

| File | Responsibility | Why it's separate |
|---|---|---|
| `exceptions.py` | Two error types: model failed to load vs. a single inference call failed | Callers need to tell "the model is broken" apart from "this one frame is bad" — the first should probably stop a job, the second shouldn't |
| `models.py` | `Detection` (one box) and `DetectionResult` (all boxes for a frame + metadata), with `to_dict()` / `to_json()` | This is the **contract** the next stage (an API layer, a UI, a redaction engine) depends on. It has zero dependency on Ultralytics/PyTorch, so anything downstream can import just this file |
| `device.py` | Picks `"cuda"` if a GPU is available, else `"cpu"` | Isolated so it's unit-testable without a GPU, and so a broken/missing `torch` install degrades to CPU instead of crashing the whole pipeline |
| `detector.py` | The only place that talks to Ultralytics; loads the model once, runs `predict()`, converts tensors → NumPy → `Detection` objects | Keeping all torch/tensor handling in one file means nothing else in the codebase (or downstream) ever has to import torch or know what a `Boxes` object looks like |
| `cli.py` | Thin orchestration: reads args, calls `YOLODetector`, writes JSON files | No detection logic lives here — it only wires the pieces together |

---

## 2. How it works (explanation)

### 2.1 Loading the model

```python
detector = YOLODetector(model_path="yolov8n.pt", confidence_threshold=0.25)
```

Under the hood, `YOLODetector.__init__`:
1. Validates `confidence_threshold` and `iou_threshold` are both in `[0, 1]`.
2. Calls `resolve_device()` — if you didn't pass `device=`, it checks
   `torch.cuda.is_available()`. GPU present → `"cuda"`. No GPU, or torch
   isn't even installed → `"cpu"`, logged as a warning, never a crash.
3. Loads the weights via `ultralytics.YOLO(model_path)`. `"yolov8n.pt"`
   (nano) or `"yolov8s.pt"` (small) are downloaded automatically by
   Ultralytics on first use if not already cached; you can also point
   `model_path` at your own custom-trained `.pt` file.
4. Any failure in step 3 (missing file, corrupted weights, incompatible
   version) is caught and re-raised as `ModelLoadError` — a clear,
   package-specific exception instead of a raw Ultralytics/torch
   traceback.
5. Grabs `model.names` — the `{0: "person", 1: "car", ...}` class-id →
   class-name lookup baked into the weights.

### 2.2 Running inference on a frame

```python
result = detector.predict(frame)   # frame: NumPy array (H, W, 3), BGR
```

1. The frame is validated (not `None`, is a NumPy array, non-empty). A
   bad frame raises `InferenceError` immediately rather than letting a
   confusing shape-mismatch error bubble up from inside Ultralytics.
2. The model is called: `model(frame, conf=threshold, iou=self.iou_threshold, device=self.device, verbose=False)`.
   - `conf` is the confidence threshold — the model itself won't even
     return boxes below this. You can override it per-call:
     `detector.predict(frame, confidence_threshold=0.6)` without
     reloading the model.
   - `iou` controls Non-Max Suppression (how aggressively overlapping
     boxes for the same object get merged/removed).
3. Ultralytics returns a `Results` object. Its `.boxes` exposes
   `.xyxy` (box corners), `.conf` (scores), `.cls` (class ids) as
   PyTorch tensors. `_to_numpy()` converts each via `.cpu().numpy()`
   (falling back gracefully if it's already array-like) — this is the
   **only** place tensors are touched.
4. Each row becomes a `Detection(class_id, class_name, confidence, bbox)`.
   As a defense-in-depth check, anything below `threshold` is filtered
   again here even though the model's `conf=` kwarg should have already
   excluded it.
5. Detections are sorted highest-confidence-first (friendlier default
   for an API or UI consumer than model-output order).
6. Everything is wrapped in a `DetectionResult`, which also records
   `model_name`, `device`, `image_shape`, `inference_time_ms`, and
   optional `source_frame_index` / `source_video` for traceability back
   to the originating video.

### 2.3 Getting JSON out

```python
result.to_dict()   # plain dict
result.to_json()   # pretty-printed JSON string
```

Example shape:

```json
{
  "model": "yolov8n.pt",
  "device": "cpu",
  "image_shape": { "height": 480, "width": 640, "channels": 3 },
  "inference_time_ms": 42.7,
  "source_frame_index": 15,
  "source_video": "uploads/clip.mp4",
  "num_detections": 2,
  "detections": [
    {
      "class_id": 0,
      "class_name": "person",
      "confidence": 0.91,
      "bbox": { "x_min": 120.4, "y_min": 55.0, "x_max": 310.2, "y_max": 420.8 }
    },
    {
      "class_id": 2,
      "class_name": "car",
      "confidence": 0.77,
      "bbox": { "x_min": 400.0, "y_min": 210.5, "x_max": 610.0, "y_max": 380.0 }
    }
  ]
}
```

### 2.4 GPU / CPU behavior

- If a CUDA GPU is present and `torch` is built with CUDA support,
  inference automatically runs on it — no config needed.
- If no GPU is present, or `torch` isn't installed at all,
  `resolve_device()` falls back to `"cpu"` and logs why. The pipeline
  keeps working either way, just slower on CPU.
- You can always force it explicitly: `YOLODetector(..., device="cpu")`.

---

## 3. Install

```bash
pip install ultralytics opencv-python numpy
```

> **Note on this development environment:** `ultralytics` pulls in a
> full CUDA-enabled PyTorch install (torch + several NVIDIA packages),
> which is several GB. That download wasn't feasible in the sandbox
> this was built in (limited disk, no GPU), so `YOLODetector` was
> verified with a dependency-injected fake model that mimics
> Ultralytics' real `Results.boxes` shape (`.xyxy` / `.conf` / `.cls`
> tensors, `.names` dict) — see `tests/test_yolo_detector.py`. All
> parsing, filtering, error-handling, and JSON-formatting logic is
> exercised the same way it would be against a real model; only the
> actual `ultralytics.YOLO(...)` weight-loading + forward pass is
> mocked out. Install the real package in your target environment
> (ideally one with a GPU) to run genuine inference, and run
> `test_real_model.py` below as a smoke test.

Optional real-model smoke test once installed:

```python
from yolo_detection import YOLODetector
import cv2

detector = YOLODetector("yolov8n.pt")          # auto-downloads weights on first run
frame = cv2.imread("some_photo.jpg")
result = detector.predict(frame)
print(result.to_json())
```

---

## 4. Usage

### 4.1 Library — single frame

```python
from yolo_detection import YOLODetector

detector = YOLODetector(model_path="yolov8n.pt", confidence_threshold=0.25)
result = detector.predict(frame)          # frame = NumPy array (H, W, 3)
print(result.to_json())
```

### 4.2 Library — chained with `frame_extraction`

```python
from frame_extraction import FrameExtractor
from yolo_detection import YOLODetector

extractor = FrameExtractor("uploads/clip.mp4", target_fps=2.0)
detector = YOLODetector("yolov8n.pt", confidence_threshold=0.3)

for frame_meta in extractor.extract():
    result = detector.predict(
        frame_meta.frame,
        source_frame_index=frame_meta.frame_index,
        source_video=frame_meta.source_video,
    )
    print(frame_meta.timestamp_sec, result.to_dict()["num_detections"])
```

### 4.3 CLI

```bash
# Run on a folder of already-extracted frame images
python -m yolo_detection.cli --frames-dir frames/clip --output-dir detections/clip --conf 0.3

# End-to-end: extract frames from a video, then detect — no intermediate images written
python -m yolo_detection.cli --video uploads/clip.mp4 --fps 2 --output-dir detections/clip

# Force CPU, custom weights, looser NMS
python -m yolo_detection.cli --frames-dir frames/clip --output-dir out \
    --model yolov8s.pt --device cpu --conf 0.4 --iou 0.5
```

Each run writes one JSON file per frame into `--output-dir`.

---

## 5. Error handling

- Model fails to load (missing file, corrupted weights, incompatible
  Ultralytics version) → `ModelLoadError`, raised once at
  `YOLODetector(...)` construction — before any frames are processed.
- A single bad frame during `predict()` (empty array, wrong type) or a
  forward-pass failure (e.g. an out-of-memory error) → `InferenceError`,
  scoped to that one call so a batch/CLI run can skip it and continue.
- `confidence_threshold` / `iou_threshold` outside `[0, 1]` →
  `ValueError`, both at construction and on a per-call override.

## 6. Tests

```bash
python -m unittest yolo_detection.tests.test_yolo_detector -v
```

14 tests covering: model-load failure wrapping, invalid threshold
validation, CPU fallback when torch is unavailable, multi-detection
parsing + confidence-sort order, empty-detections handling, the
defense-in-depth confidence filter, per-call threshold overrides,
unknown class-id fallback, invalid-frame errors, forward-pass exception
wrapping, conf/iou/device pass-through to the model, and `to_dict()` /
`to_json()` shape + round-trip.
