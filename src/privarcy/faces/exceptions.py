"""Custom exception types for the face_privacy package."""


class FacePrivacyError(Exception):
    """Raised for invalid input, a detection/embedding failure, or an
    enrollment operation that can't proceed (e.g. no face found in a
    photo submitted for enrollment)."""
