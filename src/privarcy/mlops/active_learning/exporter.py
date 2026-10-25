"""Core export logic: CorrectionExporter.

Pulls confirmed (accepted) corrections out of data_store.CorrectionLogDB
and turns them into YOLO-format labeled training examples — the exact
raw images/ + labels/ input shape dataset_pipeline.DatasetSplitter expects.
"""

import logging
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Optional

import cv2

from .exceptions import ActiveLearningError
from .models import ExportStats

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


class CorrectionExporter:
    """Turns reviewer-confirmed corrections into new training data.

    Parameters:
        correction_db: A data_store.CorrectionLogDB to read from.
        frames_dir:     Folder containing the original frame images the
                          corrections' frame_ids refer to (e.g. frame_extraction's
                          output folder).
        class_map:        A dataset_pipeline.ClassMap — reused directly so the
                            class-id numbering here always matches the main
                            training dataset's numbering. Passing a different
                            class list than what the main dataset used would
                            silently corrupt the merged dataset.
    """

    def __init__(self, correction_db, frames_dir: str | Path, class_map) -> None:
        self.correction_db = correction_db
        self.frames_dir = Path(frames_dir)
        self.class_map = class_map

    def export(self, output_dir: str | Path) -> ExportStats:
        """Write accepted corrections as YOLO-format images/ + labels/,
        ready to hand to dataset_pipeline (either as its own raw input, or
        merged into your main annotated dataset before splitting).

        Rejected (false-positive) corrections are NOT auto-included as
        labels — see the class docstring in the README for why — but are
        collected into `ExportStats.hard_negatives` for manual review.
        """
        output_dir = Path(output_dir)
        images_out = output_dir / "images"
        labels_out = output_dir / "labels"
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)

        accepted = self.correction_db.get_decisions(decision="accepted", limit=1_000_000)
        rejected = self.correction_db.get_decisions(decision="rejected", limit=1_000_000)

        stats = ExportStats()
        by_frame = defaultdict(list)
        for record in accepted:
            by_frame[record.frame_id].append(record)

        for frame_id, records in by_frame.items():
            image_path = self._find_frame_image(frame_id)
            if image_path is None:
                logger.warning("No frame image found for frame_id '%s' — skipping %d correction(s).",
                                frame_id, len(records))
                stats.skipped_missing_image += len(records)
                continue

            frame = cv2.imread(str(image_path))
            if frame is None:
                logger.warning("Could not read image '%s' — skipping.", image_path)
                stats.skipped_missing_image += len(records)
                continue

            height, width = frame.shape[:2]
            label_lines = []
            for record in records:
                class_name = record.detected_class
                if class_name not in self.class_map.name_to_id:
                    raise ActiveLearningError(
                        f"Class '{class_name}' from a correction (frame '{frame_id}') isn't in the "
                        f"provided class map {self.class_map.names}. The class map passed to "
                        f"CorrectionExporter must match the one used for the main training dataset."
                    )
                class_id = self.class_map.name_to_id[class_name]
                label_lines.append(self._to_yolo_line(class_id, record.bbox, width, height))
                stats.boxes_exported += 1
                stats.class_counts[class_name] = stats.class_counts.get(class_name, 0) + 1

            shutil.copy2(image_path, images_out / image_path.name)
            (labels_out / f"{image_path.stem}.txt").write_text("\n".join(label_lines) + "\n")
            stats.frames_exported += 1

        for record in rejected:
            stats.hard_negatives.append({
                "frame_id": record.frame_id, "bbox": record.bbox,
                "detected_class": record.detected_class, "reviewer": record.reviewer,
            })

        logger.info(
            "Exported %d frame(s), %d box(es) across %d class(es). %d correction(s) skipped "
            "(no matching image). %d rejected correction(s) logged as hard negatives.",
            stats.frames_exported, stats.boxes_exported, len(stats.class_counts),
            stats.skipped_missing_image, len(stats.hard_negatives),
        )
        return stats

    def _find_frame_image(self, frame_id) -> Optional[Path]:
        stem = str(frame_id)
        for ext in IMAGE_EXTENSIONS:
            candidate = self.frames_dir / f"{stem}{ext}"
            if candidate.exists():
                return candidate

        # frame_extraction names files like "clip_frame000042_t21.000s.jpg" —
        # frame_id (e.g. the native frame_index) may only be a SUBSTRING of
        # the real filename, so fall back to a substring match.
        matches = [
            p for p in self.frames_dir.glob(f"*{stem}*")
            if p.suffix.lower() in IMAGE_EXTENSIONS
        ]
        return matches[0] if matches else None

    @staticmethod
    def _to_yolo_line(class_id: int, bbox, width: int, height: int) -> str:
        x_min, y_min, x_max, y_max = bbox
        x_center = min(max(((x_min + x_max) / 2) / width, 0.0), 1.0)
        y_center = min(max(((y_min + y_max) / 2) / height, 0.0), 1.0)
        box_w = min(max((x_max - x_min) / width, 0.0), 1.0)
        box_h = min(max((y_max - y_min) / height, 0.0), 1.0)
        return f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}"
