"""Custom exception types for the ocr package."""


class OCRError(Exception):
    """Raised when TrOCR fails to load, or fails to run on a given region crop."""
