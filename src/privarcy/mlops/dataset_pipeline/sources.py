"""Dataset sources.

A "source" resolves to a (images_dir, labels_dir) pair — a flat folder of
raw, un-split images and matching YOLO label files. DatasetSplitter
doesn't care which source produced them, so a teammate can point the same
pipeline at a shared Roboflow project instead of a local folder without
touching the split/validate/yaml logic at all.
"""

import logging
import shutil
from pathlib import Path
from typing import Optional, Protocol, Tuple

from .exceptions import DatasetSourceError
from .validator import IMAGE_EXTENSIONS

logger = logging.getLogger(__name__)


class DatasetSource(Protocol):
    """Anything that can resolve to a local (images_dir, labels_dir) pair."""

    def resolve(self) -> Tuple[Path, Path]: ...


class LocalDirectorySource:
    """Wraps an existing local folder of raw images + YOLO label files.

    Use this when the dataset is already on disk (e.g. exported from an
    annotation tool, or already synced by a teammate).
    """

    def __init__(self, images_dir: str | Path, labels_dir: str | Path):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)

    def resolve(self) -> Tuple[Path, Path]:
        if not self.images_dir.exists():
            raise DatasetSourceError(f"Images directory not found: {self.images_dir}")
        if not self.labels_dir.exists():
            raise DatasetSourceError(f"Labels directory not found: {self.labels_dir}")
        return self.images_dir, self.labels_dir


class RoboflowSource:
    """Downloads a dataset from Roboflow and stages it as a flat raw folder.

    Requires the `roboflow` package: `pip install roboflow`.

    Roboflow exports in YOLOv8 format already come pre-split into
    train/valid/test subfolders (each with their own images/ and labels/).
    Since this pipeline owns the train/val/test split (so ratios and the
    random seed are consistent and reproducible across the team), this
    source flattens the Roboflow export back into a single raw
    images/ + labels/ pair before handing off to DatasetSplitter.

    Parameters:
        api_key:    Roboflow API key (from your Roboflow account settings).
        workspace:  Roboflow workspace slug.
        project:    Roboflow project slug.
        version:    Dataset version number within the project.
        download_dir: Where to stage the download + flattened copy.
                       Defaults to ./roboflow_downloads/<project>_v<version>.
    """

    def __init__(
        self,
        api_key: str,
        workspace: str,
        project: str,
        version: int,
        download_dir: Optional[str | Path] = None,
    ):
        self.api_key = api_key
        self.workspace = workspace
        self.project = project
        self.version = version
        self.download_dir = Path(download_dir) if download_dir else Path(f"roboflow_downloads/{project}_v{version}")

    def resolve(self) -> Tuple[Path, Path]:
        try:
            from roboflow import Roboflow
        except ImportError as exc:
            raise DatasetSourceError(
                "The 'roboflow' package is not installed. Install it with `pip install roboflow`."
            ) from exc

        try:
            rf = Roboflow(api_key=self.api_key)
            project = rf.workspace(self.workspace).project(self.project)
            dataset = project.version(self.version).download("yolov8", location=str(self.download_dir))
        except Exception as exc:
            raise DatasetSourceError(f"Roboflow download failed: {exc}") from exc

        export_root = Path(getattr(dataset, "location", self.download_dir))
        return flatten_split_export(export_root)


def flatten_split_export(export_root: str | Path) -> Tuple[Path, Path]:
    """Gather images/labels out of a Roboflow-style train/valid/test
    export layout into a single flat staging folder this pipeline can
    re-split. Public (not just used internally by RoboflowSource) so
    merge_exports.py can reuse it for already-downloaded per-class
    exports too.
    """
    export_root = Path(export_root)
    flat_images = export_root / "_flat" / "images"
    flat_labels = export_root / "_flat" / "labels"
    flat_images.mkdir(parents=True, exist_ok=True)
    flat_labels.mkdir(parents=True, exist_ok=True)

    split_dirs = [d for d in ("train", "valid", "test") if (export_root / d).exists()]
    if not split_dirs:
        raise DatasetSourceError(
            f"Roboflow export at {export_root} doesn't have the expected "
            f"train/valid/test structure — can't flatten it."
        )

    copied = 0
    for split in split_dirs:
        images_src = export_root / split / "images"
        labels_src = export_root / split / "labels"
        if not images_src.exists():
            continue
        for img_path in images_src.iterdir():
            if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            shutil.copy2(img_path, flat_images / img_path.name)
            copied += 1
        if labels_src.exists():
            for label_path in labels_src.glob("*.txt"):
                shutil.copy2(label_path, flat_labels / label_path.name)

    logger.info("Flattened Roboflow export (%s) into %d image(s) at %s",
                ", ".join(split_dirs), copied, flat_images)
    return flat_images, flat_labels
