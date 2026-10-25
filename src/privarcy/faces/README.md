# faces

Face detection and "which person is this".

* `detector.py` - `YoloFaceDetector` (the model's `face` class) and `FaceDetector` (Haar), combined by `build_face_detector`.
* `embedder.py` - `FaceEmbedder`: HOG appearance signature (baseline, not deep learning).
* `identity.py` - `FaceBook`: identities with exemplars and thumbnails; strict/loose thresholds explained at the top.
* `policy.py` - `FaceMode` (auto / selected / every) and `FacePolicy.decide(...)` - the single blur/keep rule.
* `scanner.py` - `scan_faces(video)`: lists the distinct people in a video.

See `docs/FACE_MODES.md`.
