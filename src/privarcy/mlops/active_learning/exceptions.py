"""Custom exception types for the active_learning package."""


class ActiveLearningError(Exception):
    """Raised when a correction can't be exported: missing frame image,
    unmapped class name, or an invalid bbox."""
