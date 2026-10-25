"""
dataset_pipeline

Ingests a raw folder of annotated images + YOLO-format label files,
validates them, performs a reproducible train/val/test split, lays the
result out in the directory structure Ultralytics YOLOv8 expects, and
generates the data.yaml config file training needs.

Also provides pluggable "sources" so a dataset can come from a local
directory or be pulled down via the Roboflow API — same downstream
split/validate/yaml logic either way.

Pipeline position (Ticket 3, after Tickets 1 & 2):
    frame_extraction -> (manual/external annotation) -> raw images+labels
        -> dataset_pipeline (this package) -> images/{train,val,test},
           labels/{train,val,test}, data.yaml -> Ultralytics YOLOv8 training

Public API:
    DatasetSplitter        - core class: validates + splits + writes data.yaml
    LocalDirectorySource     - wraps an existing local images/ + labels/ folder
    RoboflowSource             - downloads a dataset via the Roboflow API
    ClassMap                     - id<->name lookup for class labels
    DatasetStats                   - counts/stats returned after a run
    DatasetValidationError           - raised for malformed/mismatched data
    DatasetSourceError                 - raised when a source can't be resolved
"""

from .exceptions import DatasetValidationError, DatasetSourceError
from .models import ClassMap, DatasetStats, SplitCounts
from .sources import LocalDirectorySource, RoboflowSource, flatten_split_export
from .splitter import DatasetSplitter
from .yaml_writer import write_data_yaml
from .merge_exports import merge_class_exports, MergeStats

__all__ = [
    "DatasetSplitter",
    "LocalDirectorySource",
    "RoboflowSource",
    "flatten_split_export",
    "merge_class_exports",
    "MergeStats",
    "ClassMap",
    "DatasetStats",
    "SplitCounts",
    "write_data_yaml",
    "DatasetValidationError",
    "DatasetSourceError",
]

__version__ = "1.0.0"
