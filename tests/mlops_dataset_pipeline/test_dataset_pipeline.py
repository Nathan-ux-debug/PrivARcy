"""
Unit tests for the dataset_pipeline package.

Run with:
    python -m unittest dataset_pipeline.tests.test_dataset_pipeline -v
"""

import shutil
import tempfile
import unittest
from pathlib import Path

from privarcy.mlops.dataset_pipeline import (
    DatasetSplitter,
    LocalDirectorySource,
    RoboflowSource,
    ClassMap,
    DatasetValidationError,
    DatasetSourceError,
)
from privarcy.mlops.dataset_pipeline.validator import validate_raw_dataset


def _write_fake_image(path: Path) -> None:
    # A valid-looking JPEG isn't required for these tests — the pipeline
    # never decodes pixel data, only moves/copies files by name — so a
    # tiny placeholder is enough and keeps the test suite fast.
    path.write_bytes(b"\xff\xd8\xff\xe0FAKEJPEGDATA")


def _write_label(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def _make_raw_dataset(root: Path, num_images: int = 20, classes=("person", "car")) -> tuple[Path, Path]:
    images_dir = root / "raw_images"
    labels_dir = root / "raw_labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    for i in range(num_images):
        stem = f"img{i:03d}"
        _write_fake_image(images_dir / f"{stem}.jpg")
        class_id = i % len(classes)
        _write_label(labels_dir / f"{stem}.txt", [f"{class_id} 0.5 0.5 0.2 0.2"])

    return images_dir, labels_dir


class TestValidator(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.class_map = ClassMap(names=["person", "car"])

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_valid_dataset_passes(self):
        images_dir, labels_dir = _make_raw_dataset(self.tmp_dir, num_images=10)
        report = validate_raw_dataset(images_dir, labels_dir, self.class_map)
        self.assertEqual(len(report.valid_stems), 10)
        self.assertEqual(report.class_counts["person"], 5)
        self.assertEqual(report.class_counts["car"], 5)

    def test_missing_images_dir_raises(self):
        with self.assertRaises(DatasetValidationError):
            validate_raw_dataset(self.tmp_dir / "nope", self.tmp_dir, self.class_map)

    def test_orphan_label_strict_raises(self):
        images_dir, labels_dir = _make_raw_dataset(self.tmp_dir, num_images=5)
        _write_label(labels_dir / "ghost.txt", ["0 0.5 0.5 0.1 0.1"])
        with self.assertRaises(DatasetValidationError):
            validate_raw_dataset(images_dir, labels_dir, self.class_map, strict=True)

    def test_orphan_label_non_strict_excludes_and_continues(self):
        images_dir, labels_dir = _make_raw_dataset(self.tmp_dir, num_images=5)
        _write_label(labels_dir / "ghost.txt", ["0 0.5 0.5 0.1 0.1"])
        report = validate_raw_dataset(images_dir, labels_dir, self.class_map, strict=False)
        self.assertIn("ghost", report.orphan_labels)
        self.assertEqual(len(report.valid_stems), 5)  # ghost never counted, doesn't affect real images

    def test_out_of_range_class_id_flagged_malformed(self):
        images_dir, labels_dir = _make_raw_dataset(self.tmp_dir, num_images=3)
        _write_label(labels_dir / "img000.txt", ["9 0.5 0.5 0.1 0.1"])  # class 9 doesn't exist
        with self.assertRaises(DatasetValidationError):
            validate_raw_dataset(images_dir, labels_dir, self.class_map, strict=True)

    def test_out_of_range_class_id_reported_non_strict(self):
        images_dir, labels_dir = _make_raw_dataset(self.tmp_dir, num_images=3)
        _write_label(labels_dir / "img000.txt", ["9 0.5 0.5 0.1 0.1"])
        report = validate_raw_dataset(images_dir, labels_dir, self.class_map, strict=False)
        self.assertIn("img000.txt", report.malformed_labels)
        self.assertEqual(len(report.valid_stems), 2)

    def test_out_of_bounds_coordinates_flagged(self):
        images_dir, labels_dir = _make_raw_dataset(self.tmp_dir, num_images=3)
        _write_label(labels_dir / "img000.txt", ["0 1.5 0.5 0.1 0.1"])  # x_center > 1.0
        report = validate_raw_dataset(images_dir, labels_dir, self.class_map, strict=False)
        self.assertIn("img000.txt", report.malformed_labels)
        self.assertEqual(len(report.valid_stems), 2)  # the other 2 images are still fine

    def test_missing_label_excluded_by_default(self):
        images_dir, labels_dir = _make_raw_dataset(self.tmp_dir, num_images=5)
        (labels_dir / "img000.txt").unlink()  # remove one label -> background image
        report = validate_raw_dataset(images_dir, labels_dir, self.class_map, strict=True)
        self.assertIn("img000", report.stems_without_labels)
        self.assertEqual(len(report.valid_stems), 4)

    def test_missing_label_included_when_allowed(self):
        images_dir, labels_dir = _make_raw_dataset(self.tmp_dir, num_images=5)
        (labels_dir / "img000.txt").unlink()
        report = validate_raw_dataset(images_dir, labels_dir, self.class_map, allow_missing_labels=True)
        self.assertEqual(len(report.valid_stems), 5)


class TestDatasetSplitter(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.output_dir = self.tmp_dir / "split_out"
        self.class_map = ClassMap(names=["person", "car"])

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_split_ratios_and_counts(self):
        images_dir, labels_dir = _make_raw_dataset(self.tmp_dir, num_images=100)
        source = LocalDirectorySource(images_dir, labels_dir)
        splitter = DatasetSplitter(source, self.output_dir, self.class_map, split_ratios=(0.7, 0.15, 0.15), seed=1)
        stats = splitter.run()

        self.assertEqual(stats.split_counts.total, 100)
        self.assertEqual(stats.split_counts.train, 70)
        self.assertEqual(stats.split_counts.val, 15)
        self.assertEqual(stats.split_counts.test, 15)

    def test_output_directory_layout(self):
        images_dir, labels_dir = _make_raw_dataset(self.tmp_dir, num_images=20)
        source = LocalDirectorySource(images_dir, labels_dir)
        splitter = DatasetSplitter(source, self.output_dir, self.class_map, seed=1)
        splitter.run()

        for split in ("train", "val", "test"):
            self.assertTrue((self.output_dir / "images" / split).is_dir())
            self.assertTrue((self.output_dir / "labels" / split).is_dir())

        train_images = list((self.output_dir / "images" / "train").glob("*.jpg"))
        train_labels = list((self.output_dir / "labels" / "train").glob("*.txt"))
        self.assertEqual(len(train_images), len(train_labels))
        self.assertTrue((self.output_dir / "data.yaml").exists())

    def test_copy_mode_preserves_raw_data(self):
        images_dir, labels_dir = _make_raw_dataset(self.tmp_dir, num_images=10)
        source = LocalDirectorySource(images_dir, labels_dir)
        splitter = DatasetSplitter(source, self.output_dir, self.class_map, mode="copy", seed=1)
        splitter.run()
        self.assertEqual(len(list(images_dir.glob("*.jpg"))), 10)  # untouched

    def test_move_mode_empties_raw_data(self):
        images_dir, labels_dir = _make_raw_dataset(self.tmp_dir, num_images=10)
        source = LocalDirectorySource(images_dir, labels_dir)
        splitter = DatasetSplitter(source, self.output_dir, self.class_map, mode="move", seed=1)
        splitter.run()
        self.assertEqual(len(list(images_dir.glob("*.jpg"))), 0)  # moved out

    def test_split_is_deterministic_with_same_seed(self):
        images_dir, labels_dir = _make_raw_dataset(self.tmp_dir, num_images=30)
        source1 = LocalDirectorySource(images_dir, labels_dir)
        out1 = self.tmp_dir / "out1"
        DatasetSplitter(source1, out1, self.class_map, seed=7).run()
        train1 = sorted(p.name for p in (out1 / "images" / "train").glob("*.jpg"))

        source2 = LocalDirectorySource(images_dir, labels_dir)
        out2 = self.tmp_dir / "out2"
        DatasetSplitter(source2, out2, self.class_map, seed=7).run()
        train2 = sorted(p.name for p in (out2 / "images" / "train").glob("*.jpg"))

        self.assertEqual(train1, train2)

    def test_different_seeds_produce_different_splits(self):
        images_dir, labels_dir = _make_raw_dataset(self.tmp_dir, num_images=30)
        out1 = self.tmp_dir / "out1"
        DatasetSplitter(LocalDirectorySource(images_dir, labels_dir), out1, self.class_map, seed=1).run()
        train1 = sorted(p.name for p in (out1 / "images" / "train").glob("*.jpg"))

        out2 = self.tmp_dir / "out2"
        DatasetSplitter(LocalDirectorySource(images_dir, labels_dir), out2, self.class_map, seed=2).run()
        train2 = sorted(p.name for p in (out2 / "images" / "train").glob("*.jpg"))

        self.assertNotEqual(train1, train2)

    def test_invalid_ratios_raise(self):
        images_dir, labels_dir = _make_raw_dataset(self.tmp_dir, num_images=5)
        source = LocalDirectorySource(images_dir, labels_dir)
        with self.assertRaises(ValueError):
            DatasetSplitter(source, self.output_dir, self.class_map, split_ratios=(0.5, 0.3, 0.3))  # sums to 1.1

    def test_invalid_mode_raises(self):
        images_dir, labels_dir = _make_raw_dataset(self.tmp_dir, num_images=5)
        source = LocalDirectorySource(images_dir, labels_dir)
        with self.assertRaises(ValueError):
            DatasetSplitter(source, self.output_dir, self.class_map, mode="delete")

    def test_background_images_get_empty_label_file(self):
        images_dir, labels_dir = _make_raw_dataset(self.tmp_dir, num_images=10)
        (labels_dir / "img000.txt").unlink()
        source = LocalDirectorySource(images_dir, labels_dir)
        splitter = DatasetSplitter(source, self.output_dir, self.class_map, allow_missing_labels=True, seed=1)
        stats = splitter.run()
        self.assertEqual(stats.split_counts.total, 10)
        # The background image's label file should exist and be empty, somewhere in the split.
        all_label_files = list((self.output_dir / "labels").rglob("img000.txt"))
        self.assertEqual(len(all_label_files), 1)
        self.assertEqual(all_label_files[0].read_text(), "")


class TestYamlOutput(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_data_yaml_contents(self):
        images_dir, labels_dir = _make_raw_dataset(self.tmp_dir, num_images=20)
        output_dir = self.tmp_dir / "out"
        class_map = ClassMap(names=["person", "car"])
        DatasetSplitter(LocalDirectorySource(images_dir, labels_dir), output_dir, class_map, seed=1).run()

        content = (output_dir / "data.yaml").read_text()
        self.assertIn("train: images/train", content)
        self.assertIn("val: images/val", content)
        self.assertIn("test: images/test", content)
        self.assertIn("nc: 2", content)
        self.assertIn("0: person", content)
        self.assertIn("1: car", content)
        # Windows paths contain a colon (C:\...), which correctly gets quoted
        # in the YAML output — check the path value itself appears, rather
        # than requiring an exact unquoted "path: X" prefix match.
        self.assertIn(str(output_dir.resolve()), content)


class TestSources(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_local_source_missing_images_dir_raises(self):
        source = LocalDirectorySource(self.tmp_dir / "nope", self.tmp_dir)
        with self.assertRaises(DatasetSourceError):
            source.resolve()

    def test_local_source_resolves_existing_dirs(self):
        images_dir, labels_dir = _make_raw_dataset(self.tmp_dir, num_images=1)
        source = LocalDirectorySource(images_dir, labels_dir)
        resolved_images, resolved_labels = source.resolve()
        self.assertEqual(resolved_images, images_dir)
        self.assertEqual(resolved_labels, labels_dir)

    def test_roboflow_source_without_package_raises_clear_error(self):
        # The 'roboflow' package is intentionally not installed in this
        # test environment — confirms the missing-dependency path is
        # handled with a clear DatasetSourceError, not a raw ImportError.
        source = RoboflowSource(api_key="fake", workspace="ws", project="proj", version=1)
        with self.assertRaises(DatasetSourceError):
            source.resolve()


class TestClassMap(unittest.TestCase):
    def test_id_and_name_lookup(self):
        cm = ClassMap(names=["person", "car", "dog"])
        self.assertEqual(cm.id_to_name[1], "car")
        self.assertEqual(cm.name_to_id["dog"], 2)
        self.assertTrue(cm.is_valid_id(0))
        self.assertFalse(cm.is_valid_id(3))
        self.assertEqual(cm.num_classes, 3)

    def test_from_file(self):
        tmp_dir = Path(tempfile.mkdtemp())
        try:
            classes_file = tmp_dir / "classes.txt"
            classes_file.write_text("person\ncar\n\ndog\n")  # blank line should be skipped
            cm = ClassMap.from_file(classes_file)
            self.assertEqual(cm.names, ["person", "car", "dog"])
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
