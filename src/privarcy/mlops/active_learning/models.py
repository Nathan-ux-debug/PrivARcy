"""Data models for the active_learning package."""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ExportStats:
    """Summary of one export run.

    Attributes:
        frames_exported:      Number of distinct frame images written out.
        boxes_exported:          Number of corrected boxes turned into
                                  YOLO label lines.
        class_counts:               class_name -> count of exported boxes.
        skipped_missing_image:          Corrections skipped because no
                                         matching frame image was found on disk.
        hard_negatives:                   Rejected (false-positive) corrections,
                                           logged separately for manual review
                                           rather than auto-included as labels
                                           (see README for why).
    """

    frames_exported: int = 0
    boxes_exported: int = 0
    class_counts: Dict[str, int] = field(default_factory=dict)
    skipped_missing_image: int = 0
    hard_negatives: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "frames_exported": self.frames_exported,
            "boxes_exported": self.boxes_exported,
            "class_counts": self.class_counts,
            "skipped_missing_image": self.skipped_missing_image,
            "hard_negatives_flagged": len(self.hard_negatives),
        }
