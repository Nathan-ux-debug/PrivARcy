"""
Unit tests for the active_learning package.

Run with:
    python -m unittest active_learning.tests.test_active_learning -v
"""

import shutil
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from privarcy.storage import CorrectionLogDB
from privarcy.mlops.dataset_pipeline import ClassMap
from privarcy.mlops.active_learning import CorrectionExporter, ActiveLearningError


def _write_fake_frame(path: Path, size=(200, 150)) -> None:
    frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    cv2.imwrite(str(path), frame)


class TestCorrectionExporter(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.frames_dir = self.tmp_dir / "frames"
        self.frames_dir.mkdir()
        self.output_dir = self.tmp_dir / "retrain_batch"
        self.class_map = ClassMap(names=["document", "credit_card", "face"])
        self.db = CorrectionLogDB(":memory:")

    def tearDown(self):
        self.db.close()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_accepted_correction_exported_as_label(self):
        _write_fake_frame(self.frames_dir / "frame_001.jpg")
        self.db.record_decision(
            frame_id="frame_001", bbox=[10, 10, 60, 60], decision="accepted",
            detected_class="document", confidence=0.7, reviewer="alice",
        )
        exporter = CorrectionExporter(self.db, self.frames_dir, self.class_map)
        stats = exporter.export(self.output_dir)

        self.assertEqual(stats.frames_exported, 1)
        self.assertEqual(stats.boxes_exported, 1)
        self.assertEqual(stats.class_counts["document"], 1)

        label_file = self.output_dir / "labels" / "frame_001.txt"
        self.assertTrue(label_file.exists())
        parts = label_file.read_text().strip().split()
        self.assertEqual(int(parts[0]), self.class_map.name_to_id["document"])  # correct class id
        self.assertTrue((self.output_dir / "images" / "frame_001.jpg").exists())

    def test_yolo_coordinates_are_normalized_correctly(self):
        _write_fake_frame(self.frames_dir / "frame_002.jpg", size=(200, 100))
        self.db.record_decision(
            frame_id="frame_002", bbox=[0, 0, 200, 100], decision="accepted",  # full-frame box
            detected_class="face", confidence=0.9,
        )
        exporter = CorrectionExporter(self.db, self.frames_dir, self.class_map)
        exporter.export(self.output_dir)

        parts = (self.output_dir / "labels" / "frame_002.txt").read_text().strip().split()
        x_center, y_center, w, h = (float(v) for v in parts[1:])
        self.assertAlmostEqual(x_center, 0.5, places=3)
        self.assertAlmostEqual(y_center, 0.5, places=3)
        self.assertAlmostEqual(w, 1.0, places=3)
        self.assertAlmostEqual(h, 1.0, places=3)

    def test_multiple_corrections_same_frame_merged_into_one_label_file(self):
        _write_fake_frame(self.frames_dir / "frame_003.jpg")
        self.db.record_decision(frame_id="frame_003", bbox=[0, 0, 50, 50], decision="accepted", detected_class="document")
        self.db.record_decision(frame_id="frame_003", bbox=[60, 60, 100, 100], decision="accepted", detected_class="face")

        exporter = CorrectionExporter(self.db, self.frames_dir, self.class_map)
        stats = exporter.export(self.output_dir)

        self.assertEqual(stats.frames_exported, 1)  # one image
        self.assertEqual(stats.boxes_exported, 2)     # two boxes in its label file
        lines = (self.output_dir / "labels" / "frame_003.txt").read_text().strip().splitlines()
        self.assertEqual(len(lines), 2)

    def test_rejected_corrections_become_hard_negatives_not_labels(self):
        _write_fake_frame(self.frames_dir / "frame_004.jpg")
        self.db.record_decision(
            frame_id="frame_004", bbox=[0, 0, 10, 10], decision="rejected",
            detected_class="document", reviewer="bob",
        )
        exporter = CorrectionExporter(self.db, self.frames_dir, self.class_map)
        stats = exporter.export(self.output_dir)

        self.assertEqual(stats.frames_exported, 0)     # nothing exported as a label
        self.assertEqual(stats.boxes_exported, 0)
        self.assertEqual(len(stats.hard_negatives), 1)
        self.assertEqual(stats.hard_negatives[0]["reviewer"], "bob")

    def test_missing_frame_image_is_skipped_not_fatal(self):
        # No frame image written on disk for this frame_id.
        self.db.record_decision(frame_id="ghost_frame", bbox=[0, 0, 10, 10], decision="accepted", detected_class="face")
        exporter = CorrectionExporter(self.db, self.frames_dir, self.class_map)
        stats = exporter.export(self.output_dir)

        self.assertEqual(stats.skipped_missing_image, 1)
        self.assertEqual(stats.frames_exported, 0)

    def test_unknown_class_raises(self):
        _write_fake_frame(self.frames_dir / "frame_005.jpg")
        self.db.record_decision(frame_id="frame_005", bbox=[0, 0, 10, 10], decision="accepted", detected_class="not_a_real_class")
        exporter = CorrectionExporter(self.db, self.frames_dir, self.class_map)
        with self.assertRaises(ActiveLearningError):
            exporter.export(self.output_dir)

    def test_substring_frame_id_matches_frame_extraction_naming(self):
        # frame_extraction names files like "clip_frame000042_t21.000s.jpg" —
        # the correction's frame_id is often just the native frame_index (42).
        _write_fake_frame(self.frames_dir / "clip_frame000042_t21.000s.jpg")
        self.db.record_decision(frame_id="42", bbox=[0, 0, 10, 10], decision="accepted", detected_class="face")

        exporter = CorrectionExporter(self.db, self.frames_dir, self.class_map)
        stats = exporter.export(self.output_dir)
        self.assertEqual(stats.frames_exported, 1)

    def test_class_map_mismatch_with_main_dataset_is_caught(self):
        # A correction references a class that isn't in the class map passed in —
        # this simulates a class-list drift between the main dataset and this export.
        _write_fake_frame(self.frames_dir / "frame_006.jpg")
        self.db.record_decision(frame_id="frame_006", bbox=[0, 0, 10, 10], decision="accepted", detected_class="license_plate")
        small_class_map = ClassMap(names=["document", "face"])  # doesn't include license_plate
        exporter = CorrectionExporter(self.db, self.frames_dir, small_class_map)
        with self.assertRaises(ActiveLearningError):
            exporter.export(self.output_dir)


if __name__ == "__main__":
    unittest.main()
