# Architecture

## Layers (dependencies point downward only)

```
app/            PyQt5 interface (pages, canvas, flyouts, workers)         knows: services
services/       BackendService  - the single facade                        knows: everything below
pipeline/       FrameEngine, VideoPipeline, JobRunner, LiveSession,        knows: detection, decision, faces,
                PreviewEngine, MagicTracker, EditPlan                       redaction, ocr, capture, storage
faces/ capture/ detection/ decision/ redaction/ ocr/ rules/ storage/       leaf packages, independent of the UI
```

The UI never touches OpenCV/YOLO directly. Anything that processes pixels lives in `pipeline/` and is testable without Qt.

## One rule set, three callers

`FrameEngine.process(frame)` is the only place that decides what is redacted in a frame
(detect -> smooth/track -> decide -> face policy -> OCR text -> user edits -> redact). It is used by:

* `VideoPipeline` - recorded video, frame by frame, resumable
* `LiveSession` - camera frames
* `PreviewEngine` - the editor: detect once (`analyze`), then `render` instantly from the cached detections
  using the same thresholds, padding, face policy and redaction functions

## Faces

`FaceBook` (identities + appearance exemplars) -> `FacePolicy` (mode + choices -> blur or not) -> engine.
`faces.scanner.scan_faces` builds the book from a video once; processing asks the same book, so a choice holds for the whole
video. See FACE_MODES.md.

## Threads (live)

```
camera -> capture thread -> [latest frame slot] -> process thread -> latest REDACTED frame -> UI timer / recording
```
The slot keeps only the newest frame; frames overwritten before processing are counted as dropped. The raw frame never
leaves these two threads. Qt only paints redacted frames.

## Data

Everything the app writes is under one folder (`privarcy.paths.data_dir()`): `outputs/`, `jobs/`, `review_frames/`,
`rules/`, `correction_log.db`, `live_measurements.json`.

## Safety invariants (each has tests)

1. A frame that fails processing live is replaced by black, never published raw.
2. Faces that cannot be identified are blurred (Auto, Every) - and also in Selected mode (cannot prove they are not selected).
3. Linking two sightings as "the same person" is strict when it could expose someone (Auto) and loose when it could miss
   someone who must stay blurred (Selected).
4. Camera passwords are never in URLs shown, logged or stored; they exist in memory only.
5. Preview shows exactly what processing will do (same rules), apart from frame-to-frame smoothing.
