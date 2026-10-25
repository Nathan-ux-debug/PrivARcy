"""Custom exception types for the dataset_pipeline package."""


class DatasetValidationError(Exception):
    """Raised when raw images/labels are missing, mismatched, or malformed
    (orphan labels, out-of-range class ids, bad YOLO label syntax, etc.)."""


class DatasetSourceError(Exception):
    """Raised when a dataset source (local directory or Roboflow) cannot
    be resolved — missing folder, missing dependency, failed download."""
