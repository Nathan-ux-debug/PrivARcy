"""Data models for records stored by data_store."""

import json
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class FaceEmbeddingRecord:
    """One enrolled face embedding (D2). A person can have multiple rows
    (multiple exemplar photos/angles) — matching checks against all of
    them and keeps the closest.

    Attributes:
        id:          Row id (None until inserted).
        person_name: Human-readable identity this embedding belongs to.
        embedding:   Feature vector (list of floats) — the actual
                      biometric-derived data. See face_privacy for how
                      this is computed.
        source:       Free-text note on where this exemplar came from
                        (e.g. "enrollment_photo_1.jpg", "webcam_frame_42").
        created_at:     ISO-8601 timestamp string.
    """

    person_name: str
    embedding: List[float]
    id: Optional[int] = None
    source: str = ""
    created_at: str = ""

    def embedding_json(self) -> str:
        return json.dumps(self.embedding)

    @staticmethod
    def embedding_from_json(text: str) -> List[float]:
        return json.loads(text)


@dataclass
class CorrectionRecord:
    """One human reviewer decision on a flagged detection (D4).

    Attributes:
        id:              Row id (None until inserted).
        frame_id:         Identifier tying this back to a specific frame
                           (e.g. frame_index or a video+frame composite key).
        bbox:              (x_min, y_min, x_max, y_max) of the reviewed detection.
        decision:            "accepted" or "rejected" — did the reviewer agree
                              the region should be redacted.
        detected_class:        What the upstream pipeline thought this was
                                (e.g. "credit_card", "face:unknown").
        confidence:               The pipeline's original confidence score for
                                   this detection, for later threshold tuning.
        context:                    Free-text/JSON blob for anything else worth
                                     keeping (reviewer notes, source video, etc.).
        reviewer:                     Identifier for who made the call.
        created_at:                     ISO-8601 timestamp string.
    """

    frame_id: str
    bbox: List[float]
    decision: str
    detected_class: str = ""
    confidence: float = 0.0
    context: str = ""
    reviewer: str = ""
    id: Optional[int] = None
    created_at: str = ""
