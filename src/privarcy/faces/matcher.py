"""Face matching: compare a query embedding against everyone enrolled."""

import logging
from typing import List, Optional

import numpy as np

from .exceptions import FacePrivacyError
from .models import FaceBox, MatchResult

logger = logging.getLogger(__name__)


class FaceMatcher:
    """Compares a face embedding against a set of known embeddings.

    Parameters:
        similarity_threshold: Minimum cosine similarity to count as a
                                match. Higher = stricter (fewer false
                                "known" matches, but more likely to miss
                                a real match under different lighting/
                                angle). 0.75 is a reasonable starting
                                point for these HOG embeddings — tune
                                against your own enrollment photos.
    """

    def __init__(self, similarity_threshold: float = 0.75) -> None:
        if not (-1.0 <= similarity_threshold <= 1.0):
            raise FacePrivacyError(f"similarity_threshold must be within [-1,1], got {similarity_threshold}")
        self.similarity_threshold = similarity_threshold

    def match(
        self,
        face: FaceBox,
        query_embedding: np.ndarray,
        enrolled: List[tuple[str, np.ndarray]],
    ) -> MatchResult:
        """Find the closest enrolled person to query_embedding.

        Args:
            face: The FaceBox this embedding came from (carried through
                   into the result for convenience).
            query_embedding: Embedding of the face being checked.
            enrolled: List of (person_name, embedding) pairs — typically
                       every known embedding,
                       unpacked by the caller.

        Returns:
            MatchResult with person_name=None if nothing cleared the
            similarity threshold (i.e. this is an unknown/unenrolled face).
        """
        if not enrolled:
            return MatchResult(face=face, person_name=None, similarity=0.0)

        best_name: Optional[str] = None
        best_similarity = -1.0

        for person_name, embedding in enrolled:
            similarity = self._cosine_similarity(query_embedding, embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_name = person_name

        if best_similarity >= self.similarity_threshold:
            return MatchResult(face=face, person_name=best_name, similarity=best_similarity)
        return MatchResult(face=face, person_name=None, similarity=best_similarity)

    def match_many(
        self,
        faces_with_embeddings: List[tuple[FaceBox, np.ndarray]],
        enrolled: List[tuple[str, np.ndarray]],
    ) -> List[MatchResult]:
        """Convenience batch version of match() for every face in a frame."""
        return [self.match(face, embedding, enrolled) for face, embedding in faces_with_embeddings]

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        if a.shape != b.shape:
            raise FacePrivacyError(f"Embedding shape mismatch: {a.shape} vs {b.shape} (were they made by the same FaceEmbedder config?)")
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)
