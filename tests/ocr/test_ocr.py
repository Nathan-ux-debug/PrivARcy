"""
Unit tests for the ocr package.

These tests do NOT require `transformers` or `torch` to be installed —
TrOCREngine accepts a `_loader` injection point specifically so its
crop/parse/error-handling logic can be verified with a lightweight fake
processor+model pair that mimics TrOCRProcessor/VisionEncoderDecoderModel's
real call shape.

Run with:
    python -m unittest ocr.tests.test_ocr -v
"""

import json
import unittest
from types import SimpleNamespace

import numpy as np

from privarcy.ocr import TrOCREngine, OCRError


class FakeProcessor:
    def __call__(self, images, return_tensors):
        return SimpleNamespace(pixel_values="pixel_values_placeholder")

    def batch_decode(self, sequences, skip_special_tokens):
        return [sequences]  # the fake model hands back the text directly as "sequences"


class FakeModel:
    def __init__(self, text="HELLO WORLD", per_token_max_probs=(0.9, 0.85, 0.95)):
        self.text = text
        self.per_token_max_probs = per_token_max_probs

    def to(self, device):
        pass  # no-op, mirrors torch.nn.Module.to()

    def generate(self, pixel_values, output_scores, return_dict_in_generate):
        # scores: one "distribution" per generated token — a flat list of
        # floats is enough here since the fallback (non-torch) confidence
        # path just takes max() of each step.
        scores = [[p] for p in self.per_token_max_probs]
        return SimpleNamespace(sequences=self.text, scores=scores)


def _fake_loader(text="HELLO WORLD", probs=(0.9, 0.85, 0.95)):
    model = FakeModel(text=text, per_token_max_probs=probs)
    return lambda model_name: (FakeProcessor(), model)


def _frame(width=200, height=150):
    return np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)


class TestModelLoading(unittest.TestCase):
    def test_loader_exception_wrapped_as_ocr_error(self):
        def failing_loader(name):
            raise RuntimeError("corrupted weights")

        with self.assertRaises(OCRError):
            TrOCREngine(model_name="fake", _loader=failing_loader)

    def test_device_defaults_to_cpu_without_torch(self):
        engine = TrOCREngine(model_name="fake", device=None, _loader=_fake_loader())
        self.assertEqual(engine.device, "cpu")  # torch isn't installed in this environment

    def test_explicit_device_respected(self):
        engine = TrOCREngine(model_name="fake", device="cpu", _loader=_fake_loader())
        self.assertEqual(engine.device, "cpu")


class TestReadRegion(unittest.TestCase):
    def test_returns_expected_text_and_confidence_range(self):
        engine = TrOCREngine(model_name="fake", _loader=_fake_loader(text="ACME BANK", probs=(0.9, 0.8, 0.95)))
        result = engine.read_region(_frame(), bbox=(10, 10, 100, 60), frame_id="f1", source_class="credit_card")
        self.assertEqual(result.text, "ACME BANK")
        self.assertTrue(0.0 <= result.confidence <= 1.0)
        self.assertAlmostEqual(result.confidence, sum((0.9, 0.8, 0.95)) / 3, places=5)
        self.assertEqual(result.frame_id, "f1")
        self.assertEqual(result.source_class, "credit_card")
        self.assertEqual(result.bbox, (10, 10, 100, 60))

    def test_invalid_frame_raises(self):
        engine = TrOCREngine(model_name="fake", _loader=_fake_loader())
        with self.assertRaises(OCRError):
            engine.read_region(None, bbox=(0, 0, 10, 10))
        with self.assertRaises(OCRError):
            engine.read_region(np.array([]), bbox=(0, 0, 10, 10))

    def test_degenerate_bbox_raises(self):
        engine = TrOCREngine(model_name="fake", _loader=_fake_loader())
        with self.assertRaises(OCRError):
            engine.read_region(_frame(), bbox=(50, 50, 50, 50))  # zero-area box

    def test_bbox_outside_frame_raises(self):
        engine = TrOCREngine(model_name="fake", _loader=_fake_loader())
        with self.assertRaises(OCRError):
            engine.read_region(_frame(width=100, height=100), bbox=(200, 200, 300, 300))

    def test_bbox_partially_outside_frame_is_clamped_not_rejected(self):
        engine = TrOCREngine(model_name="fake", _loader=_fake_loader())
        # Box extends past the right/bottom edge but still overlaps the frame.
        result = engine.read_region(_frame(width=100, height=100), bbox=(80, 80, 150, 150))
        self.assertEqual(result.text, "HELLO WORLD")

    def test_inference_failure_wrapped_as_ocr_error(self):
        class BrokenModel(FakeModel):
            def generate(self, *a, **k):
                raise RuntimeError("out of memory")

        loader = lambda name: (FakeProcessor(), BrokenModel())
        engine = TrOCREngine(model_name="fake", _loader=loader)
        with self.assertRaises(OCRError):
            engine.read_region(_frame(), bbox=(0, 0, 50, 50))


class TestReadRegions(unittest.TestCase):
    def test_batch_reads_all_regions(self):
        engine = TrOCREngine(model_name="fake", _loader=_fake_loader(text="1234 5678"))
        regions = [
            {"bbox": (0, 0, 50, 50), "frame_id": "f1", "source_class": "credit_card"},
            {"bbox": (60, 60, 100, 100), "frame_id": "f1", "source_class": "document"},
        ]
        results = engine.read_regions(_frame(width=200, height=200), regions)
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r.text == "1234 5678" for r in results))

    def test_batch_skips_bad_region_and_continues(self):
        engine = TrOCREngine(model_name="fake", _loader=_fake_loader())
        regions = [
            {"bbox": (0, 0, 50, 50)},                    # valid
            {"bbox": (500, 500, 600, 600)},               # entirely outside frame -> skipped
            {"bbox": (60, 60, 100, 100)},                # valid
        ]
        results = engine.read_regions(_frame(width=200, height=200), regions)
        self.assertEqual(len(results), 2)  # bad one excluded, good ones kept


class TestSerialization(unittest.TestCase):
    def test_to_dict_and_to_json_shape(self):
        engine = TrOCREngine(model_name="fake", _loader=_fake_loader(text="ID 998877"))
        result = engine.read_region(_frame(), bbox=(1, 2, 3, 4), frame_id=7, source_class="document")
        payload = result.to_dict()

        self.assertEqual(payload["text"], "ID 998877")
        self.assertEqual(payload["frame_id"], 7)
        self.assertEqual(payload["source_class"], "document")
        self.assertEqual(payload["bbox"], {"x_min": 1.0, "y_min": 2.0, "x_max": 3.0, "y_max": 4.0})

        parsed = json.loads(result.to_json())
        self.assertEqual(parsed, payload)


if __name__ == "__main__":
    unittest.main()
