"""Core splitting logic: DatasetSplitter.

Takes a resolved raw dataset (any DatasetSource), validates it, performs
a reproducible shuffle + split, lays files out in the directory structure
Ultralytics YOLOv8 expects, and writes data.yaml.
"""

import logging
import random
import shutil
from pathlib import Path
from typing import List, Union

from .exceptions import DatasetValidationError
from .models import ClassMap, DatasetStats, SplitCounts
from .sources import DatasetSource
from .validator import IMAGE_EXTENSIONS, validate_raw_dataset
from .yaml_writer import write_data_yaml

logger = logging.getLogger(__name__)


class DatasetSplitter:
    """Validates a raw dataset and splits it into YOLOv8's expected layout.

    Output layout under `output_dir`:
        images/train/, images/val/, images/test/
        labels/train/, labels/val/, labels/test/
        data.yaml

    Parameters:
        source:      A DatasetSource (LocalDirectorySource or RoboflowSource)
                     that resolves to a flat (images_dir, labels_dir) pair.
        output_dir:  Root folder to write the split dataset into.
        class_names: Ordered list of class names (index == YOLO class id),
                     or a pre-built ClassMap.
        split_ratios: (train, val, test) fractions — must sum to ~1.0.
                       Default: 70/15/15 as specified in the ticket.
        seed:         Random seed for the shuffle, so re-running the
                       pipeline on the same raw data reproduces the same
                       split (important for comparable evaluation runs).
        mode:          "copy" (default, keeps raw data intact) or "move"
                       (frees disk space, but consumes the raw folder).
        allow_missing_labels: Treat images with no label file as valid
                                background/negative examples instead of
                                excluding them.
        strict_validation: If True (default), any orphan or malformed
                             label file aborts the run. If False, bad
                             entries are excluded and the run proceeds.
    """

    def __init__(
        self,
        source: DatasetSource,
        output_dir: str | Path,
        class_names: Union[List[str], ClassMap],
        split_ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
        seed: int = 42,
        mode: str = "copy",
        allow_missing_labels: bool = False,
        strict_validation: bool = True,
    ) -> None:
        if mode not in ("copy", "move"):
            raise ValueError(f"mode must be 'copy' or 'move', got '{mode}'")

        total_ratio = sum(split_ratios)
        if any(r < 0 for r in split_ratios) or abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"split_ratios must be non-negative and sum to 1.0, got {split_ratios} (sum={total_ratio})")

        self.source = source
        self.output_dir = Path(output_dir)
        self.class_map = class_names if isinstance(class_names, ClassMap) else ClassMap(names=list(class_names))
        self.split_ratios = split_ratios
        self.seed = seed
        self.mode = mode
        self.allow_missing_labels = allow_missing_labels
        self.strict_validation = strict_validation

    def run(self) -> DatasetStats:
        """Resolve the source, validate, split, materialize files, write data.yaml."""
        images_dir, labels_dir = self.source.resolve()

        report = validate_raw_dataset(
            images_dir,
            labels_dir,
            self.class_map,
            allow_missing_labels=self.allow_missing_labels,
            strict=self.strict_validation,
        )

        stems = list(report.valid_stems)
        rng = random.Random(self.seed)
        rng.shuffle(stems)

        n = len(stems)
        train_ratio, val_ratio, _test_ratio = self.split_ratios
        train_end = round(n * train_ratio)
        val_end = train_end + round(n * val_ratio)

        train_stems = stems[:train_end]
        val_stems = stems[train_end:val_end]
        test_stems = stems[val_end:]

        for split_name, split_stems in (("train", train_stems), ("val", val_stems), ("test", test_stems)):
            if not split_stems and n > 0:
                logger.warning("Split '%s' ended up empty (%d total examples, ratios=%s) — "
                                "dataset may be too small for these ratios.", split_name, n, self.split_ratios)
            self._materialize_split(split_name, split_stems, images_dir, labels_dir)

        data_yaml_path = write_data_yaml(self.output_dir, self.class_map, include_test=len(test_stems) > 0)

        stats = DatasetStats(
            split_counts=SplitCounts(train=len(train_stems), val=len(val_stems), test=len(test_stems)),
            class_counts=report.class_counts,
            images_without_labels=0 if self.allow_missing_labels else len(report.stems_without_labels),
            output_dir=str(self.output_dir.resolve()),
            data_yaml_path=str(data_yaml_path),
        )

        logger.info(
            "Split complete: train=%d val=%d test=%d | classes=%s | data.yaml -> %s",
            stats.split_counts.train, stats.split_counts.val, stats.split_counts.test,
            list(stats.class_counts.keys()), stats.data_yaml_path,
        )
        return stats

    def _materialize_split(self, split_name: str, stems: List[str], images_dir: Path, labels_dir: Path) -> None:
        images_out = self.output_dir / "images" / split_name
        labels_out = self.output_dir / "labels" / split_name
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)

        transfer = shutil.move if self.mode == "move" else shutil.copy2

        for stem in stems:
            image_path = _find_image_for_stem(images_dir, stem)
            transfer(str(image_path), str(images_out / image_path.name))

            label_path = labels_dir / f"{stem}.txt"
            if label_path.exists():
                transfer(str(label_path), str(labels_out / label_path.name))
            else:
                # Background/negative example: YOLO convention is an empty label file.
                (labels_out / f"{stem}.txt").write_text("")


def _find_image_for_stem(images_dir: Path, stem: str) -> Path:
    for ext in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
        candidate = images_dir / f"{stem}{ext.upper()}"
        if candidate.exists():
            return candidate
    raise DatasetValidationError(f"Could not find image file for stem '{stem}' in {images_dir}")
