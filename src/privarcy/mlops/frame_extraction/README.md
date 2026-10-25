# frame_extraction

MP4 ingestion and frame-sampling pipeline built on OpenCV. This is the
pre-processing stage ahead of an object-detection stage (YOLOv8, next).

## Layout

```
frame_extraction/
├── __init__.py       # public API exports
├── exceptions.py      # VideoValidationError, FrameReadError
├── models.py           # FrameMeta dataclass (frame + metadata)
├── validation.py       # file/codec validation, isolated & unit-testable
├── extractor.py         # FrameExtractor — core sampling + save + queue logic
├── batch.py               # process_directory() — batch mode over a folder
├── cli.py                  # argparse-based command-line entry point
└── tests/
    └── test_frame_extraction.py
```

## Why split like this

- **validation.py** is isolated so a file can be validated (e.g. right
  after upload, before a job is even queued) without touching extraction
  logic.
- **models.py** defines the `FrameMeta` contract that the next pipeline
  stage (YOLOv8) depends on, independent of how frames are produced.
- **extractor.py** contains the only OpenCV read-loop, so sampling-rate
  math and error handling live in one place.
- **batch.py** / **cli.py** are thin orchestration layers on top of
  `FrameExtractor` — no duplicated logic.

## Install

```bash
pip install opencv-python numpy
```

## Library usage

```python
from frame_extraction import FrameExtractor

extractor = FrameExtractor(
    video_path="uploads/clip.mp4",
    output_dir="frames/clip",
    target_fps=2.0,
    image_format="jpg",
)

for frame_meta in extractor.extract():
    print(frame_meta.frame_index, frame_meta.timestamp_sec, frame_meta.output_path)
    # frame_meta.frame -> raw NumPy array (BGR), hand this to YOLOv8
```

### Queue-based hand-off (for a threaded YOLOv8 worker)

```python
import queue
from frame_extraction import FrameExtractor

q = queue.Queue(maxsize=100)
extractor = FrameExtractor("uploads/clip.mp4", target_fps=5.0)
thread = extractor.extract_to_queue(q)  # runs in background

while True:
    frame_meta = q.get()
    if frame_meta is None:   # sentinel = extraction finished
        break
    # run_yolov8(frame_meta.frame)

thread.join()
```

### Batch mode

```python
from frame_extraction import process_directory

summary = process_directory("uploads/", "frames/", target_fps=2.0)
# {"clip1.mp4": 42, "clip2.mp4": "ERROR: ..."}
```

## CLI usage

```bash
# single file
python -m frame_extraction.cli --video uploads/clip.mp4 --output-dir frames/clip --fps 2

# batch — every video in a folder
python -m frame_extraction.cli --input-dir uploads --output-dir frames --fps 2

# options
python -m frame_extraction.cli --video clip.mp4 --output-dir out --fps 5 \
    --format png --max-frames 100 --strict
```

## Error handling

- Missing file, empty file, unopenable/corrupt container, or an
  undecodable first frame all raise `VideoValidationError` at
  construction time (before any extraction work happens).
- A mid-stream decode failure is skipped and logged by default; pass
  `strict=True` to raise `FrameReadError` instead and stop immediately.
- `process_directory()` isolates failures per file — one bad video in a
  batch doesn't stop the rest.

## Tests

```bash
python -m unittest frame_extraction.tests.test_frame_extraction -v
```

12 tests covering: validation (missing/empty/corrupt files, invalid fps),
sampling-rate correctness, timestamp monotonicity, frame shape/dtype,
fps clamping, `max_frames`, disk-write skipping, the queue interface, and
batch processing with a mixed good/bad file set.
