"""Data models for the face_privacy package."""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class FaceBox:
    """One detected face region.

    Attributes:
        bbox:       (x_min, y_min, x_max, y_max) in pixel coordinates.
        confidence: Detection confidence. Haar cascades don't produce a
                     real probability, so this is a neighbor-count-derived
                     proxy in [0, 1] — treat it as a rough signal, not a
                     calibrated score (see detector.py).
    """

    bbox: Tuple[float, float, float, float]
    confidence: float

    @property
    def class_name(self) -> str:
        """So a FaceBox can be handed directly to redaction.FrameRedactor,
        which expects `.class_name` / `.confidence` / `.bbox` on whatever
        it's given (see redaction/redactor.py's DetectionLike protocol)."""
        return "face"


@dataclass
class MatchResult:
    """Result of comparing one detected face against the known embeddings.

    Attributes:
        face:         The FaceBox this result is for.
        person_name:  Name of the closest enrolled match, or None if
                       nothing was close enough to count as a match.
        similarity:     Cosine similarity to the closest match, in [-1, 1]
                         (in practice close to [0, 1] for these embeddings).
        is_known:         Convenience flag: person_name is not None.
    """

    face: FaceBox
    person_name: Optional[str]
    similarity: float

    @property
    def is_known(self) -> bool:
        return self.person_name is not None
