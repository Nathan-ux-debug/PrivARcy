"""Merge several SEPARATE single-class dataset exports into one unified
raw dataset with correctly-numbered classes.

This is for exactly the situation of pulling in different classes from
different places — e.g. a public Roboflow Universe `id_card` dataset,
a public `license_plate` dataset, and your own self-labeled `credit_card`
export — each downloaded as its own standalone train/valid/test export
with its own single-class numbering (almost always class_id 0, since
each was its own single-class project).

Without this step, those six exports are NOT directly usable by
DatasetSplitter: each has its own split (undermining your one
reproducible seeded split across the whole dataset) and, critically,
every one of them numbers its one class as `0` — silently merging them
as-is would make every object in every image look like the same class
to the model.

    datasets/
    ├── credit_card/    (own train/valid/test, its labels all say class 0)
    ├── document/       (own train/valid/test, its labels all say class 0)
    ├── face/           (own train/valid/test, its labels all say class 0)
    ...
        │
        ▼ merge_exports.merge_class_exports()
        │
    merged_raw/
    ├── images/   (everything, all classes, flat)
    └── labels/   (every label file's class_id REWRITTEN to its correct
                   position in the unified ClassMap)
        │
        ▼ DatasetSplitter (unchanged, works as always from here)
"""

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from .exceptions import DatasetValidationError
from .models import ClassMap
from .sources import flatten_split_export
from .validator import IMAGE_EXTENSIONS

logger = logging.getLogger(__name__)


@dataclass
class MergeStats:
    """Summary of one merge run."""

    images_per_class: Dict[str, int] = field(default_factory=dict)
    renamed_for_collision: List[str] = field(default_factory=list)
    total_images: int = 0

    def to_dict(self) -> dict:
        return {
            "images_per_class": self.images_per_class,
            "renamed_for_collision": self.renamed_for_collision,
            "total_images": self.total_images,
        }


def merge_class_exports(
    class_export_dirs: Dict[str, str | Path],
    output_dir: str | Path,
    class_map: ClassMap,
) -> MergeStats:
    """Merge N single-class Roboflow-style exports into one unified raw
    images/ + labels/ folder, with every label rewritten to the correct
    global class id.

    Args:
        class_export_dirs: {class_name: path_to_that_class's_export_root}.
                             Each export_root is expected to contain its
                             own train/valid/test/{images,labels}
                             subfolders (the standard Roboflow download
                             layout) — exactly what you get unzipping a
                             Roboflow export as-is, no pre-flattening needed.
        output_dir:            Where to write the merged images/ + labels/.
        class_map:                 The FULL, unified class list — every
                                    class_name key in class_export_dirs
                                    must appear in this map, since that's
                                    what determines the correct output
                                    class_id for each source.

    Returns:
        MergeStats with per-class image counts and any filename collisions
        that had to be renamed.

    Raises:
        DatasetValidationError: if a class_name isn't in class_map, or an
            export_dir doesn't look like a valid train/valid/test export.
    """
    output_dir = Path(output_dir)
    images_out = output_dir / "images"
    labels_out = output_dir / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    stats = MergeStats()
    seen_filenames: set[str] = set()

    for class_name, export_dir in class_export_dirs.items():
        if class_name not in class_map.name_to_id:
            raise DatasetValidationError(
                f"Class '{class_name}' isn't in the provided class_map {class_map.names}. "
                f"Every key in class_export_dirs must be one of the unified class names."
            )
        target_class_id = class_map.name_to_id[class_name]

        flat_images_dir, flat_labels_dir = flatten_split_export(Path(export_dir))
        image_files = sorted(p for p in flat_images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)

        count = 0
        for image_path in image_files:
            label_path = flat_labels_dir / f"{image_path.stem}.txt"

            out_name = image_path.name
            if out_name in seen_filenames:
                out_name = f"{class_name}_{image_path.name}"
                stats.renamed_for_collision.append(out_name)
                logger.warning("Filename collision on '%s' — renamed to '%s'.", image_path.name, out_name)
            seen_filenames.add(out_name)

            shutil.copy2(image_path, images_out / out_name)

            out_label_name = Path(out_name).stem + ".txt"
            if label_path.exists():
                remapped_lines = _remap_label_file(label_path, target_class_id)
                (labels_out / out_label_name).write_text(remapped_lines)
            else:
                # Background/negative example in the source export — preserve that as-is.
                (labels_out / out_label_name).write_text("")

            count += 1

        stats.images_per_class[class_name] = count
        stats.total_images += count
        logger.info("Merged %d image(s) from '%s' export as class_id=%d ('%s').",
                    count, export_dir, target_class_id, class_name)

    logger.info("Merge complete: %s", stats.to_dict())
    return stats


def _remap_label_file(label_path: Path, target_class_id: int) -> str:
    """Rewrite every line's class_id to target_class_id.

    Source exports here are single-class projects, so every line's
    original class_id is expected to be 0 (or otherwise irrelevant) —
    it's overwritten unconditionally, not remapped via a lookup table.
    """
    lines = [ln.strip() for ln in label_path.read_text().splitlines() if ln.strip()]
    new_lines = []
    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            logger.warning("Skipping malformed line in '%s' during remap (expected 5 values, got %d).",
                            label_path.name, len(parts))
            continue
        parts[0] = str(target_class_id)
        new_lines.append(" ".join(parts))
    return "\n".join(new_lines) + ("\n" if new_lines else "")
