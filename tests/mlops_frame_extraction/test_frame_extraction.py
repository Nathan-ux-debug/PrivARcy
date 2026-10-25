"""
Unit tests for the frame_extraction package.

Run with:
    python -m unittest frame_extraction.tests.test_frame_extraction -v
"""

import queue
import shutil
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from privarcy.mlops.frame_extraction import FrameExtractor, VideoValidationError, process_directory


def _make_synthetic_video(path: Path, num_frames=50, fps=10.0, size=(320, 240)) -> None:
    """Write a small synthetic MP4 for deterministic testing (no test fixtures needed)."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    for i in range(num_frames):
        frame = np.full((size[1], size[0], 3), (i * 5) % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()


class TestValidation(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_missing_file_raises(self):
        with self.assertRaises(VideoValidationError):
            FrameExtractor(self.tmp_dir / "does_not_exist.mp4", target_fps=1.0)

    def test_empty_file_raises(self):
        empty = self.tmp_dir / "empty.mp4"
        empty.touch()
        with self.assertRaises(VideoValidationError):
            FrameExtractor(empty, target_fps=1.0)

    def test_corrupted_file_raises(self):
        fake = self.tmp_dir / "fake.mp4"
        fake.write_text("this is not a real video file")
        with self.assertRaises(VideoValidationError):
            FrameExtractor(fake, target_fps=1.0)

    def test_zero_or_negative_fps_raises(self):
        video = self.tmp_dir / "clip.mp4"
        _make_synthetic_video(video)
        with self.assertRaises(ValueError):
            FrameExtractor(video, target_fps=0)
        with self.assertRaises(ValueError):
            FrameExtractor(video, target_fps=-1)


class TestExtraction(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.video_path = self.tmp_dir / "clip.mp4"
        _make_synthetic_video(self.video_path, num_frames=50, fps=10.0)  # 5s @ 10fps

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_sampling_rate_2fps(self):
        out_dir = self.tmp_dir / "frames"
        extractor = FrameExtractor(self.video_path, output_dir=out_dir, target_fps=2.0)
        results = list(extractor.extract())
        # 5s of video @ 2fps sampling -> 10 frames
        self.assertEqual(len(results), 10)
        self.assertEqual(len(list(out_dir.glob("*.jpg"))), 10)

    def test_timestamps_are_monotonic(self):
        extractor = FrameExtractor(self.video_path, target_fps=5.0)
        results = list(extractor.extract())
        timestamps = [r.timestamp_sec for r in results]
        self.assertEqual(timestamps, sorted(timestamps))

    def test_frame_is_numpy_array_with_expected_shape(self):
        extractor = FrameExtractor(self.video_path, target_fps=1.0)
        first = next(extractor.extract())
        self.assertIsInstance(first.frame, np.ndarray)
        self.assertEqual(first.frame.shape, (240, 320, 3))

    def test_fps_above_source_is_clamped(self):
        # Requesting 1000fps on a 10fps source should just yield every native frame.
        extractor = FrameExtractor(self.video_path, target_fps=1000.0)
        results = list(extractor.extract())
        self.assertEqual(len(results), 50)

    def test_max_frames_cap(self):
        extractor = FrameExtractor(self.video_path, target_fps=10.0, max_frames=3)
        results = list(extractor.extract())
        self.assertEqual(len(results), 3)

    def test_no_output_dir_skips_disk_write(self):
        extractor = FrameExtractor(self.video_path, output_dir=None, target_fps=2.0)
        results = list(extractor.extract())
        self.assertTrue(all(r.output_path is None for r in results))


class TestQueueInterface(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.video_path = self.tmp_dir / "clip.mp4"
        _make_synthetic_video(self.video_path, num_frames=50, fps=10.0)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_extract_to_queue_delivers_all_frames_and_sentinel(self):
        q: queue.Queue = queue.Queue()
        extractor = FrameExtractor(self.video_path, target_fps=5.0)
        thread = extractor.extract_to_queue(q)

        received = []
        while True:
            item = q.get()
            if item is None:
                break
            received.append(item)
        thread.join(timeout=5)

        self.assertEqual(len(received), 25)  # 5s @ 5fps
        self.assertTrue(all(isinstance(r.frame, np.ndarray) for r in received))


class TestBatchProcessing(unittest.TestCase):
    def setUp(self):
        self.input_dir = Path(tempfile.mkdtemp())
        self.output_dir = Path(tempfile.mkdtemp())
        _make_synthetic_video(self.input_dir / "good.mp4", num_frames=20, fps=10.0)
        (self.input_dir / "bad.mp4").write_text("corrupted")

    def tearDown(self):
        shutil.rmtree(self.input_dir, ignore_errors=True)
        shutil.rmtree(self.output_dir, ignore_errors=True)

    def test_batch_processes_good_and_skips_bad(self):
        summary = process_directory(self.input_dir, self.output_dir, target_fps=2.0)
        self.assertEqual(summary["good.mp4"], 4)  # 2s @ 2fps
        self.assertTrue(str(summary["bad.mp4"]).startswith("ERROR"))


if __name__ == "__main__":
    unittest.main()
