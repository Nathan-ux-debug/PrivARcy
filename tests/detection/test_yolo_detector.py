"""
Unit tests for the yolo_detection package.

These tests do NOT require `ultralytics` or `torch` to be installed.
YOLODetector accepts a `_model_loader` injection point specifically so
its parsing / filtering / error-handling logic can be verified with a
lightweight fake model that mimics the real Ultralytics Results/Boxes
shape (.boxes.xyxy / .conf / .cls, .names).

A separate, real-weights integration check is documented in the README
for environments where torch + ultralytics are actually installed.

Run with:
    python -m unittest yolo_detection.tests.test_yolo_detector -v
"""

import unittest

import numpy as np

from privarcy.detection import YOLODetector, ModelLoadError, InferenceError


class FakeBoxes:
    """Mimics ultralytics.engine.results.Boxes closely enough for our
    parsing code: .xyxy / .conf / .cls as NumPy arrays, plus __len__."""

    def __init__(self, xyxy, conf, cls):
        self.xyxy = np.array(xyxy, dtype=float)
        self.conf = np.array(conf, dtype=float)
        self.cls = np.array(cls, dtype=float)

    def __len__(self):
        return len(self.conf)


class FakeResults:
    """Mimics a single element of the list ultralytics.YOLO(...) returns."""

    def __init__(self, boxes):
        self.boxes = boxes


class FakeYOLOModel:
    """Stand-in for `ultralytics.YOLO`. Records the last call's kwargs so
    tests can assert the detector passed conf/iou/device through
    correctly, and returns a preset list of FakeResults."""

    def __init__(self, names=None, results=None):
        self.names = names or {0: "person", 1: "car"}
        self._results = results if results is not None else [FakeResults(FakeBoxes([], [], []))]
        self.last_call_kwargs = None

    def __call__(self, frame, conf=None, iou=None, device=None, verbose=None):
        self.last_call_kwargs = {"conf": conf, "iou": iou, "device": device, "verbose": verbose}
        return self._results


def _make_detector(fake_model, **overrides) -> YOLODetector:
    kwargs = dict(model_path="fake.pt", device="cpu", confidence_threshold=0.25)
    kwargs.update(overrides)
    return YOLODetector(_model_loader=lambda path: fake_model, **kwargs)


class TestModelLoading(unittest.TestCase):
    def test_loader_exception_raises_model_load_error(self):
        def failing_loader(path):
            raise RuntimeError("corrupted weights file")

        with self.assertRaises(ModelLoadError):
            YOLODetector(model_path="bad.pt", _model_loader=failing_loader)

    def test_invalid_confidence_threshold_raises(self):
        fake_model = FakeYOLOModel()
        with self.assertRaises(ValueError):
            _make_detector(fake_model, confidence_threshold=1.5)

    def test_invalid_iou_threshold_raises(self):
        fake_model = FakeYOLOModel()
        with self.assertRaises(ValueError):
            _make_detector(fake_model, iou_threshold=-0.1)

    def test_device_defaults_to_cpu_without_torch(self):
        # torch is not installed in this environment, so auto-detect
        # (device=None) must safely fall back to "cpu" rather than crash.
        fake_model = FakeYOLOModel()
        detector = YOLODetector(model_path="fake.pt", device=None, _model_loader=lambda p: fake_model)
        self.assertEqual(detector.device, "cpu")


class TestPrediction(unittest.TestCase):
    def _frame(self):
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def test_parses_multiple_detections_and_sorts_by_confidence(self):
        boxes = FakeBoxes(
            xyxy=[[10, 20, 100, 200], [300, 300, 400, 400]],
            conf=[0.55, 0.91],
            cls=[0, 1],
        )
        fake_model = FakeYOLOModel(results=[FakeResults(boxes)])
        detector = _make_detector(fake_model)

        result = detector.predict(self._frame())

        self.assertEqual(len(result.detections), 2)
        # Highest confidence first.
        self.assertAlmostEqual(result.detections[0].confidence, 0.91)
        self.assertEqual(result.detections[0].class_name, "car")
        self.assertEqual(result.detections[1].class_name, "person")
        self.assertEqual(result.detections[1].bbox, (10.0, 20.0, 100.0, 200.0))

    def test_empty_boxes_returns_no_detections(self):
        fake_model = FakeYOLOModel(results=[FakeResults(FakeBoxes([], [], []))])
        detector = _make_detector(fake_model)
        result = detector.predict(self._frame())
        self.assertEqual(result.detections, [])

    def test_defense_in_depth_confidence_filter(self):
        # Simulates the model returning a box below the requested threshold
        # (e.g. a threshold override race) — detector must still filter it.
        boxes = FakeBoxes(xyxy=[[0, 0, 10, 10]], conf=[0.10], cls=[0])
        fake_model = FakeYOLOModel(results=[FakeResults(boxes)])
        detector = _make_detector(fake_model, confidence_threshold=0.5)
        result = detector.predict(self._frame())
        self.assertEqual(result.detections, [])

    def test_per_call_confidence_override(self):
        boxes = FakeBoxes(xyxy=[[0, 0, 10, 10]], conf=[0.30], cls=[0])
        fake_model = FakeYOLOModel(results=[FakeResults(boxes)])
        detector = _make_detector(fake_model, confidence_threshold=0.5)

        # Instance default (0.5) would exclude this; per-call override lowers it.
        result = detector.predict(self._frame(), confidence_threshold=0.2)
        self.assertEqual(len(result.detections), 1)

    def test_unknown_class_id_falls_back_to_string(self):
        boxes = FakeBoxes(xyxy=[[0, 0, 10, 10]], conf=[0.9], cls=[99])
        fake_model = FakeYOLOModel(names={0: "person"}, results=[FakeResults(boxes)])
        detector = _make_detector(fake_model)
        result = detector.predict(self._frame())
        self.assertEqual(result.detections[0].class_name, "99")

    def test_invalid_frame_raises_inference_error(self):
        fake_model = FakeYOLOModel()
        detector = _make_detector(fake_model)
        with self.assertRaises(InferenceError):
            detector.predict(None)
        with self.assertRaises(InferenceError):
            detector.predict(np.array([]))

    def test_model_forward_pass_exception_wrapped(self):
        class BrokenModel(FakeYOLOModel):
            def __call__(self, *a, **k):
                raise RuntimeError("CUDA out of memory")

        detector = _make_detector(BrokenModel())
        with self.assertRaises(InferenceError):
            detector.predict(self._frame())

    def test_out_of_range_per_call_threshold_raises(self):
        fake_model = FakeYOLOModel()
        detector = _make_detector(fake_model)
        with self.assertRaises(ValueError):
            detector.predict(self._frame(), confidence_threshold=2.0)

    def test_conf_iou_device_passed_through_to_model(self):
        fake_model = FakeYOLOModel()
        detector = _make_detector(fake_model, confidence_threshold=0.4, iou_threshold=0.6, device="cpu")
        detector.predict(self._frame())
        self.assertEqual(fake_model.last_call_kwargs["conf"], 0.4)
        self.assertEqual(fake_model.last_call_kwargs["iou"], 0.6)
        self.assertEqual(fake_model.last_call_kwargs["device"], "cpu")


class TestSerialization(unittest.TestCase):
    def test_to_dict_and_to_json_shape(self):
        boxes = FakeBoxes(xyxy=[[1, 2, 3, 4]], conf=[0.77], cls=[0])
        fake_model = FakeYOLOModel(results=[FakeResults(boxes)])
        detector = _make_detector(fake_model)

        result = detector.predict(
            np.zeros((100, 200, 3), dtype=np.uint8),
            source_frame_index=5,
            source_video="clip.mp4",
        )
        payload = result.to_dict()

        self.assertEqual(payload["model"], "fake.pt")
        self.assertEqual(payload["device"], "cpu")
        self.assertEqual(payload["image_shape"], {"height": 100, "width": 200, "channels": 3})
        self.assertEqual(payload["source_frame_index"], 5)
        self.assertEqual(payload["source_video"], "clip.mp4")
        self.assertEqual(payload["num_detections"], 1)
        self.assertEqual(payload["detections"][0]["class_name"], "person")
        self.assertEqual(payload["detections"][0]["bbox"], {"x_min": 1.0, "y_min": 2.0, "x_max": 3.0, "y_max": 4.0})

        # to_json must round-trip through the standard library parser.
        import json
        parsed = json.loads(result.to_json())
        self.assertEqual(parsed, payload)


if __name__ == "__main__":
    unittest.main()
