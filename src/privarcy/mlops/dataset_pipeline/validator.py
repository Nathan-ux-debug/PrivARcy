"""Validation for a raw (pre-split) image + YOLO-label dataset.

Isolated from DatasetSplitter so a dataset can be validated on its own
(e.g. right after annotation, before anyone tries to train on it) without
pulling in the split/copy logic.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from .exceptions import DatasetValidationError
from .models import ClassMap

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class ValidationReport:
    """Result of validating a raw dataset — a matched list of stems ready
    to split, plus anything that looked wrong along the way."""

    valid_stems: List[str]                       # image basenames (no ext) with a usable label
    stems_without_labels: List[str] = field(default_factory=list)   # images with no label file (background examples)
    orphan_labels: List[str] = field(default_factory=list)          # label files with no matching image
    malformed_labels: Dict[str, List[str]] = field(default_factory=dict)  # label filename -> list of issue strings
    class_counts: Dict[str, int] = field(default_factory=dict)      # class_name -> instance count


def _parse_label_file(label_path: Path, class_map: ClassMap) -> tuple[bool, List[str], Dict[str, int]]:
    """Parse one YOLO-format .txt label file.

    Each non-empty line must be: "<class_id> <x_center> <y_center> <width> <height>",
    all normalized to [0, 1]. Returns (is_valid, issues, class_counts_in_this_file).
    """
    issues: List[str] = []
    counts: Dict[str, int] = {}

    lines = [ln.strip() for ln in label_path.read_text().splitlines() if ln.strip()]
    for line_no, line in enumerate(lines, start=1):
        parts = line.split()
        if len(parts) != 5:
            issues.append(f"line {line_no}: expected 5 values (class x y w h), got {len(parts)}")
            continue

        try:
            class_id = int(float(parts[0]))
            coords = [float(p) for p in parts[1:]]
        except ValueError:
            issues.append(f"line {line_no}: could not parse numeric values")
            continue

        if not class_map.is_valid_id(class_id):
            issues.append(
                f"line {line_no}: class_id {class_id} is out of range "
                f"(expected 0-{class_map.num_classes - 1})"
            )
            continue

        if not all(0.0 <= c <= 1.0 for c in coords):
            issues.append(f"line {line_no}: bbox coordinates must be normalized to [0, 1], got {coords}")
            continue

        class_name = class_map.id_to_name[class_id]
        counts[class_name] = counts.get(class_name, 0) + 1

    return (len(issues) == 0), issues, counts


def validate_raw_dataset(
    images_dir: Path,
    labels_dir: Path,
    class_map: ClassMap,
    allow_missing_labels: bool = False,
    strict: bool = True,
) -> ValidationReport:
    """Validate that images and YOLO label files are consistent and well-formed.

    Args:
        images_dir: Folder of raw images (flat, no train/val/test subfolders yet).
        labels_dir: Folder of matching .txt label files (same basenames as images).
        class_map: The full class list — used to check class ids are in range.
        allow_missing_labels: If True, an image with no matching label file is
                               treated as a valid background/negative example
                               instead of an error.
        strict: If True, any orphan label or malformed label file raises
                DatasetValidationError. If False, problems are collected into
                the report and logged, but validation doesn't stop the run
                (useful for a first look at a messy dataset).

    Returns:
        ValidationReport with the list of usable stems and any issues found.

    Raises:
        DatasetValidationError: if images_dir/labels_dir don't exist, no
            images are found, or (in strict mode) any mismatch/malformed
            label is present.
    """
    if not images_dir.exists() or not images_dir.is_dir():
        raise DatasetValidationError(f"Images directory not found: {images_dir}")
    if not labels_dir.exists() or not labels_dir.is_dir():
        raise DatasetValidationError(f"Labels directory not found: {labels_dir}")

    image_files = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not image_files:
        raise DatasetValidationError(f"No images (extensions {IMAGE_EXTENSIONS}) found in {images_dir}")

    label_files = {p.stem: p for p in labels_dir.iterdir() if p.suffix.lower() == ".txt"}
    image_stems = {p.stem: p for p in image_files}

    report = ValidationReport(valid_stems=[])

    # Images without a matching label file.
    for stem in image_stems:
        if stem not in label_files:
            report.stems_without_labels.append(stem)

    # Labels without a matching image (orphans — always suspicious).
    for stem in label_files:
        if stem not in image_stems:
            report.orphan_labels.append(stem)

    # Parse every label that does have a matching image.
    for stem, image_path in image_stems.items():
        label_path = label_files.get(stem)
        if label_path is None:
            continue  # handled above as stems_without_labels

        is_valid, issues, counts = _parse_label_file(label_path, class_map)
        if not is_valid:
            report.malformed_labels[label_path.name] = issues
            continue

        report.valid_stems.append(stem)
        for class_name, n in counts.items():
            report.class_counts[class_name] = report.class_counts.get(class_name, 0) + n

    if allow_missing_labels:
        report.valid_stems.extend(report.stems_without_labels)

    # Report findings.
    if report.orphan_labels:
        logger.warning("%d orphan label file(s) with no matching image: %s",
                        len(report.orphan_labels), report.orphan_labels[:5])
    if report.stems_without_labels and not allow_missing_labels:
        logger.warning("%d image(s) have no matching label file and were excluded "
                        "(pass allow_missing_labels=True to include them as background examples): %s",
                        len(report.stems_without_labels), report.stems_without_labels[:5])
    if report.malformed_labels:
        logger.warning("%d label file(s) failed validation: %s",
                        len(report.malformed_labels), list(report.malformed_labels.keys())[:5])

    if strict and (report.orphan_labels or report.malformed_labels):
        raise DatasetValidationError(
            f"Dataset validation failed: {len(report.orphan_labels)} orphan label(s), "
            f"{len(report.malformed_labels)} malformed label file(s). "
            f"Pass strict=False to proceed anyway (bad entries will be excluded)."
        )

    if not report.valid_stems:
        raise DatasetValidationError("No valid image/label pairs remain after validation.")

    logger.info(
        "Validated dataset: %d usable image/label pair(s), %d class(es), %d issue(s) found.",
        len(report.valid_stems), class_map.num_classes,
        len(report.orphan_labels) + len(report.malformed_labels),
    )

    return report
