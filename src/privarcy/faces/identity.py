"""Face identities: "who is this face" across frames, tracks and (for offline video) the whole file.

A ``FaceBook`` stores one *identity* per distinct person seen so far: a few appearance exemplars (HOG vectors from
``FaceEmbedder``) plus a thumbnail for the picker. Tracking inside one continuous appearance is done by the
``FrameEngine`` (IoU tracks); the book is what links a person who left the frame and came back, and what lets a
second pass over the same video agree with the first one.

Safety asymmetry (read this before changing a threshold)
--------------------------------------------------------
Linking two sightings as "the same person" can fail in two ways and they are NOT equally bad:
  * false SPLIT  - one person becomes two identities.  Under ``auto`` mode the second identity has no
    "keep visible" flag, so the person is blurred.  Safe.
  * false MERGE  - two different people become one.  If that identity is "kept visible", a stranger is exposed.
    Unsafe.
So linking is deliberately conservative (``MERGE_THRESHOLD``). ``FacePolicy`` applies the opposite bias when
selected people must stay BLURRED (``selected`` mode): it accepts a looser match so a selected person is not missed.
The baseline HOG embedding is weak; a stronger embedder can be dropped in without touching callers.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

MERGE_THRESHOLD = 0.55     # strict: link two sightings only when very similar
SELECTED_THRESHOLD = 0.40   # loose: used only to avoid MISSING a person who must stay blurred
MIN_FACE_PX = 24            # smaller crops carry too little detail to identify anyone
THUMB_SIZE = 96


@dataclass
class FaceIdentity:
    face_id: str
    label: str
    exemplars: List[np.ndarray] = field(default_factory=list)
    first_frame: int = 0
    last_frame: int = 0
    sightings: int = 0
    thumbnail: Optional[np.ndarray] = None      # BGR, THUMB_SIZE square
    thumb_area: int = 0

    def summary(self) -> dict:
        return {"face_id": self.face_id, "label": self.label, "first_frame": self.first_frame,
                "last_frame": self.last_frame, "sightings": self.sightings}


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0


class FaceBook:
    def __init__(self, embedder, label_prefix: str = "Face", id_prefix: str = "F", max_exemplars: int = 6,
                 label_sep: str = " ") -> None:
        self.embedder = embedder
        self.label_prefix = label_prefix
        self.id_prefix = id_prefix
        self.label_sep = label_sep
        self.max_exemplars = max_exemplars
        self._identities: Dict[str, FaceIdentity] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ embedding
    def embed(self, frame: np.ndarray, box) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """(embedding, crop) for the face at ``box`` or None when it is too small / unreadable."""
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        if min(x2 - x1, y2 - y1) < MIN_FACE_PX:
            return None
        crop = frame[y1:y2, x1:x2]
        try:
            vec = np.asarray(self.embedder.embed(crop), dtype=np.float32)
        except Exception:
            return None
        return (vec, crop) if vec.size else None

    # ------------------------------------------------------------------ matching
    def best_match(self, embedding: np.ndarray, only: Optional[Iterable[str]] = None) -> Tuple[Optional[str], float]:
        best_id, best = None, -1.0
        with self._lock:
            pool = self._identities.values() if only is None else [self._identities[i] for i in only if i in self._identities]
            for ident in pool:
                sim = max((_cosine(embedding, ex) for ex in ident.exemplars), default=-1.0)
                if sim > best:
                    best_id, best = ident.face_id, sim
        return best_id, best

    def create(self, embedding: np.ndarray, crop: np.ndarray, frame_number: int = 0) -> FaceIdentity:
        with self._lock:
            number = len(self._identities) + 1
            ident = FaceIdentity(f"{self.id_prefix}{number:02d}", f"{self.label_prefix}{self.label_sep}{number:02d}", first_frame=frame_number)
            self._identities[ident.face_id] = ident
            self._update(ident, embedding, crop, frame_number)
            return ident

    def observe(self, face_id: str, embedding: np.ndarray, crop: np.ndarray, frame_number: int = 0) -> None:
        with self._lock:
            ident = self._identities.get(face_id)
            if ident is not None:
                self._update(ident, embedding, crop, frame_number)

    def _update(self, ident: FaceIdentity, embedding: np.ndarray, crop: np.ndarray, frame_number: int) -> None:
        ident.sightings += 1
        ident.last_frame = max(ident.last_frame, frame_number)
        if len(ident.exemplars) < self.max_exemplars:
            ident.exemplars.append(embedding)
        elif ident.sightings % 5 == 0:                       # refresh slowly so appearance can drift (angle, light)
            ident.exemplars[ident.sightings % self.max_exemplars] = embedding
        area = int(crop.shape[0] * crop.shape[1])
        if area > ident.thumb_area and crop.size:
            ident.thumb_area = area
            ident.thumbnail = cv2.resize(crop, (THUMB_SIZE, THUMB_SIZE), interpolation=cv2.INTER_AREA)

    def link_or_create(self, embedding, crop, frame_number=0, threshold=MERGE_THRESHOLD) -> FaceIdentity:
        face_id, sim = self.best_match(embedding)
        if face_id is not None and sim >= threshold:
            self.observe(face_id, embedding, crop, frame_number)
            return self._identities[face_id]
        return self.create(embedding, crop, frame_number)

    # ------------------------------------------------------------------ access
    def get(self, face_id: str) -> Optional[FaceIdentity]:
        return self._identities.get(face_id)

    def identities(self) -> List[FaceIdentity]:
        with self._lock:
            return sorted(self._identities.values(), key=lambda i: i.face_id)

    def __len__(self) -> int:
        return len(self._identities)

    def clear(self) -> None:
        with self._lock:
            self._identities.clear()
