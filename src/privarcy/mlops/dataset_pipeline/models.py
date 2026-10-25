"""Data models shared across the dataset_pipeline package."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class ClassMap:
    """Bidirectional lookup between class id and class name.

    Built once from an ordered list of class names (index == class id,
    matching how YOLO label files reference classes by integer).
    """

    names: List[str]

    @property
    def id_to_name(self) -> Dict[int, str]:
        return {i: name for i, name in enumerate(self.names)}

    @property
    def name_to_id(self) -> Dict[str, int]:
        return {name: i for i, name in enumerate(self.names)}

    @property
    def num_classes(self) -> int:
        return len(self.names)

    def is_valid_id(self, class_id: int) -> bool:
        return 0 <= class_id < self.num_classes

    @classmethod
    def from_file(cls, path: str | Path) -> "ClassMap":
        """Load class names from a text file, one name per line."""
        path = Path(path)
        names = [line.strip() for line in path.read_text().splitlines() if line.strip()]
        return cls(names=names)


@dataclass
class SplitCounts:
    """Number of image/label pairs assigned to each split."""

    train: int
    val: int
    test: int

    @property
    def total(self) -> int:
        return self.train + self.val + self.test


@dataclass
class DatasetStats:
    """Summary returned after validating + splitting a dataset."""

    split_counts: SplitCounts
    class_counts: Dict[str, int] = field(default_factory=dict)  # class_name -> instance count (all splits)
    images_without_labels: int = 0  # background/negative images (empty or no label file)
    output_dir: str = ""
    data_yaml_path: str = ""

    def to_dict(self) -> dict:
        return {
            "split_counts": {
                "train": self.split_counts.train,
                "val": self.split_counts.val,
                "test": self.split_counts.test,
                "total": self.split_counts.total,
            },
            "class_counts": self.class_counts,
            "images_without_labels": self.images_without_labels,
            "output_dir": self.output_dir,
            "data_yaml_path": self.data_yaml_path,
        }
