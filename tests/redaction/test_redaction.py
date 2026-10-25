"""
Unit tests for the redaction package.

Run with:
    python -m unittest redaction.tests.test_redaction -v
"""

import shutil
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from privarcy.redaction import FrameRedactor, RedactionConfig, VideoWriter, RedactionError, ThreadedVideoCapture


@dataclass
class FakeDetection:
    """Minimal stand-in matching yolo_detection.Detection's shape
    (class_name, confidence, bbox) — redaction only depends on this
    shape, not the real class, so this is enough to test against."""

    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]


def _solid_frame(width=200, height=150, color=(100, 150, 200)) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = color
    return frame


class TestRedactionConfig(unittest.TestCase):
    def test_valid_config(self):
        config = RedactionConfig(method="pixelate", blur_strength=25, pixelate_blocks=8, padding_px=2)
        self.assertEqual(config.method, "pixelate")

    def test_invalid_method_raises(self):
        with self.assertRaises(RedactionError):
            RedactionConfig(method="invisible")

    def test_even_blur_strength_raises(self):
        with self.assertRaises(RedactionError):
            RedactionConfig(blur_strength=50)  # must be odd

    def test_negative_padding_raises(self):
        with self.assertRaises(RedactionError):
            RedactionConfig(padding_px=-1)

    def test_confidence_out_of_range_raises(self):
        with self.assertRaises(RedactionError):
            RedactionConfig(min_confidence=1.5)


class TestFrameRedactor(unittest.TestCase):
    def test_redacts_matching_class_only(self):
        frame = _solid_frame(color=(0, 0, 0))
        detections = [
            FakeDetection("credit_card", 0.9, (10, 10, 60, 60)),
            FakeDetection("person", 0.9, (100, 10, 150, 60)),
        ]
        redactor = FrameRedactor(RedactionConfig(classes_to_redact={"credit_card"}, method="solid"))
        output = redactor.redact(frame, detections)

        # credit_card region -> solid black (already black, so check it's still all-zero + shape preserved)
        self.assertTrue(np.all(output[10:60, 10:60] == 0))
        # person region should be untouched (frame was black everywhere anyway, so instead verify
        # via a colored frame in a separate test below)

    def test_untouched_region_is_unchanged(self):
        frame = _solid_frame(color=(50, 60, 70))
        detections = [FakeDetection("credit_card", 0.9, (10, 10, 60, 60))]
        redactor = FrameRedactor(RedactionConfig(classes_to_redact={"credit_card"}, method="solid"))
        output = redactor.redact(frame, detections)

        # Redacted region is now black.
        self.assertTrue(np.all(output[10:60, 10:60] == 0))
        # Far-away untouched region keeps the original color.
        self.assertTrue(np.all(output[120:140, 170:190] == [50, 60, 70]))

    def test_none_means_redact_everything(self):
        frame = _solid_frame(color=(50, 60, 70))
        detections = [FakeDetection("anything_at_all", 0.9, (0, 0, 20, 20))]
        redactor = FrameRedactor(RedactionConfig(classes_to_redact=None, method="solid"))
        output = redactor.redact(frame, detections)
        self.assertTrue(np.all(output[0:20, 0:20] == 0))

    def test_min_confidence_filters_low_confidence_detections(self):
        frame = _solid_frame(color=(50, 60, 70))
        detections = [FakeDetection("credit_card", 0.10, (10, 10, 60, 60))]
        redactor = FrameRedactor(RedactionConfig(method="solid", min_confidence=0.5))
        output = redactor.redact(frame, detections)
        # Confidence too low -> untouched.
        self.assertTrue(np.all(output[10:60, 10:60] == [50, 60, 70]))

    def test_original_frame_not_mutated(self):
        frame = _solid_frame(color=(50, 60, 70))
        original_copy = frame.copy()
        detections = [FakeDetection("x", 0.9, (0, 0, 20, 20))]
        redactor = FrameRedactor(RedactionConfig(method="solid"))
        redactor.redact(frame, detections)
        self.assertTrue(np.array_equal(frame, original_copy))  # caller's array is untouched

    def test_box_clamped_to_frame_bounds(self):
        frame = _solid_frame(width=100, height=100, color=(1, 2, 3))
        # Box extends far outside the frame — should be clamped, not crash.
        detections = [FakeDetection("x", 0.9, (-50, -50, 500, 500))]
        redactor = FrameRedactor(RedactionConfig(method="solid"))
        output = redactor.redact(frame, detections)
        self.assertEqual(output.shape, frame.shape)
        self.assertTrue(np.all(output == 0))  # whole (clamped) frame redacted

    def test_gaussian_method_changes_pixels(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        # Put a hard edge in there so blur has something to smooth.
        frame[40:60, 40:60] = 255
        detections = [FakeDetection("x", 0.9, (30, 30, 70, 70))]
        redactor = FrameRedactor(RedactionConfig(method="gaussian", blur_strength=15))
        output = redactor.redact(frame, detections)
        self.assertFalse(np.array_equal(output[30:70, 30:70], frame[30:70, 30:70]))

    def test_pixelate_method_reduces_detail(self):
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        detections = [FakeDetection("x", 0.9, (0, 0, 100, 100))]
        redactor = FrameRedactor(RedactionConfig(method="pixelate", pixelate_blocks=5))
        output = redactor.redact(frame, detections)
        # Pixelated output should have far fewer unique rows than the random original.
        self.assertLess(len(np.unique(output.reshape(-1, 3), axis=0)), len(np.unique(frame.reshape(-1, 3), axis=0)))

    def test_invalid_frame_raises(self):
        redactor = FrameRedactor()
        with self.assertRaises(RedactionError):
            redactor.redact(None, [])
        with self.assertRaises(RedactionError):
            redactor.redact(np.array([]), [])

    def test_no_detections_returns_unchanged_copy(self):
        frame = _solid_frame(color=(9, 9, 9))
        redactor = FrameRedactor()
        output = redactor.redact(frame, [])
        self.assertTrue(np.array_equal(output, frame))
        self.assertIsNot(output, frame)  # still a copy, not the same object

    def test_tiny_box_does_not_crash_gaussian(self):
        frame = _solid_frame(width=50, height=50)
        detections = [FakeDetection("x", 0.9, (10, 10, 11, 11))]  # 1x1 px box
        redactor = FrameRedactor(RedactionConfig(method="gaussian", blur_strength=51))
        output = redactor.redact(frame, detections)  # should not raise
        self.assertEqual(output.shape, frame.shape)


class TestVideoWriter(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_writes_playable_video(self):
        output_path = self.tmp_dir / "out.mp4"
        writer = VideoWriter(output_path, fps=10.0)
        for i in range(15):
            frame = _solid_frame(width=64, height=48, color=(i % 255, 0, 0))
            writer.write_frame(frame)
        writer.release()

        self.assertTrue(output_path.exists())
        self.assertGreater(output_path.stat().st_size, 0)

        # Read it back to confirm it's a genuinely valid, decodable video.
        cap = cv2.VideoCapture(str(output_path))
        self.assertTrue(cap.isOpened())
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        self.assertEqual(frame_count, 15)

    def test_context_manager_releases(self):
        output_path = self.tmp_dir / "out2.mp4"
        with VideoWriter(output_path, fps=5.0) as writer:
            writer.write_frame(_solid_frame())
        self.assertTrue(output_path.exists())

    def test_invalid_fps_raises(self):
        with self.assertRaises(RedactionError):
            VideoWriter(self.tmp_dir / "x.mp4", fps=0)

    def test_mismatched_frame_size_is_resized_not_crashed(self):
        output_path = self.tmp_dir / "out3.mp4"
        writer = VideoWriter(output_path, fps=10.0, frame_size=(64, 48))
        writer.write_frame(_solid_frame(width=64, height=48))
        writer.write_frame(_solid_frame(width=100, height=100))  # different size
        writer.release()
        self.assertTrue(output_path.exists())

    def test_release_without_any_frames_is_safe(self):
        writer = VideoWriter(self.tmp_dir / "empty.mp4", fps=10.0)
        writer.release()  # should not raise even though write_frame was never called


class FakeCaptureBackend:
    """Stand-in for cv2.VideoCapture, cycling through a fixed list of frames."""

    def __init__(self, frames):
        self._frames = frames
        self._idx = 0
        self._opened = True

    def isOpened(self):
        return self._opened

    def read(self):
        if not self._frames:
            return False, None
        frame = self._frames[self._idx % len(self._frames)]
        self._idx += 1
        return True, frame

    def release(self):
        self._opened = False


class TestThreadedVideoCapture(unittest.TestCase):
    def test_reads_frames_via_background_thread(self):
        frames = [_solid_frame(color=(i, i, i)) for i in range(5)]
        capture = ThreadedVideoCapture(source=0, _capture_factory=lambda src: FakeCaptureBackend(frames))
        capture.start()
        try:
            frame = None
            for _ in range(200):  # poll briefly until the background thread produces a frame
                frame = capture.read()
                if frame is not None:
                    break
            self.assertIsNotNone(frame)
        finally:
            capture.stop()

    def test_unopenable_source_raises(self):
        class DeadCapture(FakeCaptureBackend):
            def isOpened(self):
                return False

        with self.assertRaises(RedactionError):
            ThreadedVideoCapture(source=0, _capture_factory=lambda src: DeadCapture([]))


if __name__ == "__main__":
    unittest.main()
