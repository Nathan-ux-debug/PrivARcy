# Testing

```
python -m pytest                  # everything (Qt runs offscreen automatically in the scripts)
python -m pytest tests/faces      # one area
python -m pytest -m "not slow"    # skip the end-to-end video tests
```

## What is real and what is mocked

| Test area | Real | Mocked |
|---|---|---|
| `tests/pipeline`, `tests/faces`, `tests/capture` | engine, policy, tracking, redaction, threads, file I/O | YOLO model (scripted detections), camera device, face embedding in some logic tests |
| `tests/integration/test_e2e_faces.py` | real photo with a real face, Haar detector, HOG identities, real video written and read back | only the YOLO model (reports Haar boxes as class `face`) |
| `tests/ui` | Qt widgets, BackendService, storage, correction log, preview, processing | YOLO model, camera |

## What these tests do NOT prove

* That your trained `best.pt` detects well. Detector quality needs your own evaluation data.
* Behaviour with real cameras/RTSP hardware and real phone apps. Those paths use OpenCV; the logic around them is tested with doubles.
* Speed on your machine. Live FPS depends on CPU/GPU; the page shows measured numbers.

UI tests save screenshots to `$PRIVARCY_SHOTS` (default: temp folder) so you can look at them.
