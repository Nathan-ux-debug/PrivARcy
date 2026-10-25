# redaction

Takes the detections `yolo_detection` produces and actually **covers**
them — blur, pixelate, or solid block — then writes the result back out
as a video, either from a saved file (batch) or a live camera/stream
feed (real-time).

This is Ticket 4 — the step nothing before it actually did:

```
frame_extraction → yolo_detection → redaction (this package) → redacted .mp4 / live preview
```

---

## 1. Layout

```
redaction/
├── __init__.py       # public API
├── exceptions.py       # RedactionError
├── config.py             # RedactionConfig — what to redact & how
├── redactor.py             # FrameRedactor — the actual pixel-painting logic
├── video_writer.py            # VideoWriter — writes frames back out as .mp4
├── live_stream.py                # ThreadedVideoCapture — low-latency webcam/RTSP reads
├── cli.py                           # --video (batch) and --live (real-time) modes
└── tests/
    └── test_redaction.py
```

### Why split like this

| File | Responsibility | Why it's separate |
|---|---|---|
| `redactor.py` | Given a frame + a list of detections, paint over the matching regions | Doesn't import `yolo_detection` or `frame_extraction` at all — it only needs objects with `.class_name` / `.confidence` / `.bbox`, so it'd work unchanged if you ever swap detectors |
| `config.py` | Which classes to redact, which method, how strong | Validated once at construction (bad method/kernel size/threshold fails immediately, not mid-video) |
| `video_writer.py` | Turns a stream of frames back into a playable `.mp4` | Isolated so codec/container concerns never leak into redaction logic |
| `live_stream.py` | Reads a camera/stream continuously in a background thread | This is what makes real-time actually work — see §3 below |

---

## 2. How it works

### 2.1 The core redact step

```python
from redaction import FrameRedactor, RedactionConfig

redactor = FrameRedactor(RedactionConfig(
    classes_to_redact={"credit_card", "document"},  # None = redact every detection
    method="pixelate",       # "gaussian" | "pixelate" | "solid"
    padding_px=4,             # extra margin around each box
))

redacted_frame = redactor.redact(frame, detection_result.detections)
```

For each detection whose class is in `classes_to_redact` (or all of them,
if that's `None`) and whose confidence clears `min_confidence`:
1. The box is padded and clamped to the frame's actual bounds (so a box
   that runs slightly outside the frame, or right at the edge, doesn't crash).
2. That rectangular region is replaced according to `method`:
   - **`gaussian`** — `cv2.GaussianBlur` with a kernel sized by
     `blur_strength` (auto-shrunk if the box itself is smaller than the
     kernel, so tiny boxes never crash).
   - **`pixelate`** — shrink the region down to a small grid
     (`pixelate_blocks` across), then scale back up with nearest-neighbor
     — the classic blocky "ID redacted" look.
   - **`solid`** — replaced with an all-black rectangle. Strongest
     guarantee nothing underneath is recoverable, at the cost of fully
     hiding the frame content there (useful for the highest-sensitivity
     items, e.g. a document photo, vs. a face where you might prefer blur).
3. The original array is never mutated — `redact()` always returns a new
   frame, so you can keep the raw version around if you need to (e.g. to
   show an operator both versions side by side).

### 2.2 Batch mode (a saved video file)

```bash
python -m redaction.cli --video uploads/clip.mp4 --output redacted.mp4 \
    --classes credit_card,document --method pixelate
```

Internally this chains `frame_extraction.FrameExtractor` (reading every
native frame by default) → `yolo_detection.YOLODetector.predict()` →
`FrameRedactor.redact()` → `VideoWriter.write_frame()`, and sets the
output `.mp4`'s fps to match the actual extraction rate so playback
speed comes out correct.

### 2.3 Live / real-time mode

```bash
python -m redaction.cli --live 0 --classes credit_card --method gaussian
```

`--live 0` opens your default webcam (any integer is a camera index; a
URL string works too, e.g. an RTSP camera feed). Each frame is detected,
redacted, and shown in a preview window; press `q` to stop. Add
`--output redacted_live.mp4` to record the redacted feed to disk at the
same time.

---

## 3. Real-time: what's actually different, and what it costs

A live feed can't be processed the same way a saved file is, for one
reason: **`cv2.VideoCapture.read()` blocks until the next frame is
ready.** If detection + redaction for one frame takes longer than the
camera's frame interval (very likely on CPU), a naive read-process-loop
falls further and further behind real time — by the time you redact
frame 100, the camera is already on frame 400.

`ThreadedVideoCapture` (in `live_stream.py`) fixes this the standard way:
a background thread continuously calls `.read()` and stores only the
**most recent** frame. The main loop always grabs whatever's newest and
silently skips whatever it couldn't get to. You trade "process every
single frame" for "always process the current moment" — which is the
right tradeoff for something like a live redaction preview, where an
old, stale frame being redacted is worse than a dropped one.

### What determines whether this is fast enough for you

- **Model size**: `yolov8n.pt` (nano) is the fastest baseline model.
  `yolov8s.pt`/`m`/`l`/`x` are progressively more accurate and slower.
  Start with nano for real-time; only move up if accuracy on your
  specific objects (credit cards, documents, plates) isn't good enough.
- **GPU vs CPU**: this is the single biggest factor. A CUDA GPU can run
  `yolov8n` well past real-time video rates. Pure CPU inference is often
  the bottleneck — `ThreadedVideoCapture` prevents *capture* lag, but it
  can't make a slow forward pass faster. If real-time on CPU isn't fast
  enough, that's a hardware/model-size problem, not a bug in this code.
- **Frame resolution**: detecting on a smaller/resized frame is
  meaningfully faster than full 1080p/4K. Consider resizing before
  `detector.predict()` if you need more headroom (not currently built
  in — a reasonable next addition if you hit this).
- **Multiple redaction targets**: `FrameRedactor` itself is cheap (basic
  OpenCV ops); the cost is almost entirely in the YOLO forward pass, not
  in painting over boxes afterward.

> **Note on this development environment:** there's no webcam, display,
> or GPU in this sandbox, so `--live` mode's actual on-screen behavior
> couldn't be run here. `ThreadedVideoCapture`'s threading/frame-delivery
> logic **is** unit-tested (with an injected fake capture backend, same
> dependency-injection pattern used for `YOLODetector`), and the full
> `frame_extraction → yolo_detection → redaction → VideoWriter` chain was
> verified end-to-end producing a real, playable output video — just not
> `--live` against a real camera. Recommend a real-camera smoke test on
> your machine before depending on it live.

---

## 4. On your specific targets (documents, credit cards, plates, faces)

- **`person`, `car`** — already in the pre-trained model's 80 COCO
  classes. No custom training needed; `redaction` works with these today.
- **`credit_card`, `document`, license plates** — not in COCO. These
  need the `dataset_pipeline` → label-on-Roboflow → fine-tune path
  before `yolo_detection` can find them at all. Once you have a custom
  `best.pt`, point `--model best.pt` at it here — nothing else changes.
- **Faces** — worth flagging explicitly: *detecting* "a face is here" is
  a normal detection task (works the same as any other class, custom
  fine-tuned or an existing face-detection model). *Recognizing whose
  face it is* — e.g. only redacting specific named children rather than
  all children — is a different, much bigger task (face recognition /
  embeddings + a reference gallery of known faces), not something a
  standard YOLO class ever does well, and it comes with real privacy/
  legal considerations (biometric data laws like BIPA/GDPR often apply
  specifically and more strictly to children's biometric data). Worth
  deciding explicitly which of these two you actually need before
  building further — they're architecturally very different.

---

## 5. Install

```bash
pip install opencv-python numpy
# yolo_detection and frame_extraction packages, for the CLI's --video/--live modes
```

`redactor.py`, `config.py`, and `video_writer.py` have zero dependency on
`yolo_detection`/`frame_extraction` — only `cli.py` imports them, and
only when you actually run it.

## 6. Tests

```bash
python -m unittest redaction.tests.test_redaction -v
```

23 tests covering: config validation (bad method, even blur kernel,
negative padding, out-of-range confidence), redaction correctness (class
filtering, confidence filtering, original frame never mutated, box
clamping at frame edges, all three methods actually changing pixels,
tiny-box edge case), `VideoWriter` (produces a genuinely decodable video,
context-manager cleanup, mismatched frame sizes handled, safe to release
with zero frames written), and `ThreadedVideoCapture` (frame delivery via
an injected fake backend, unopenable-source error handling).
