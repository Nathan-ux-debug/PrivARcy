"""Which faces get blurred.

Three modes, one decision function used by recorded video, live camera and the editor preview:

  auto      blur every face (including people who appear later) EXCEPT identities the user chose to keep visible
  selected  blur ONLY the identities the user selected; everyone else is left alone by this rule
  every     blur every detected face, no exceptions; selections are ignored

The policy never touches pixels. It returns a ``FaceVerdict`` per face and the engine does the redaction.
A face that cannot be identified (too small, unreadable) is treated as unknown: blurred in ``auto`` / ``every``
and - fail-safe - also blurred in ``selected`` (we cannot prove it is NOT the selected person).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional, Set

import numpy as np

from .identity import MERGE_THRESHOLD, SELECTED_THRESHOLD, FaceBook


class FaceMode(str, Enum):
    AUTO = "auto"
    SELECTED = "selected"
    EVERY = "every"

    @classmethod
    def parse(cls, value) -> "FaceMode":
        try:
            return cls(str(getattr(value, "value", value)).lower())
        except ValueError:
            return cls.AUTO


MODE_TITLES = {
    FaceMode.AUTO: "Auto blur",
    FaceMode.SELECTED: "Blur selected faces",
    FaceMode.EVERY: "Blur every face",
}
MODE_HELP = {
    FaceMode.AUTO: "Everyone is blurred automatically, including people who appear later. Pick faces to keep visible.",
    FaceMode.SELECTED: "Only the faces you tick are blurred, every time they appear. Everyone else is left alone.",
    FaceMode.EVERY: "Every detected face is blurred. Individual choices are ignored.",
}


@dataclass(frozen=True)
class FaceVerdict:
    face_id: Optional[str]
    label: str
    blur: bool
    reason: str


class FacePolicy:
    def __init__(self, book: FaceBook, mode=FaceMode.AUTO, keep_visible: Iterable[str] = (),
                 selected: Iterable[str] = (), allow_new: bool = True) -> None:
        self.book = book
        self.mode = FaceMode.parse(mode)
        self.keep_visible: Set[str] = set(keep_visible)
        self.selected: Set[str] = set(selected)
        self.allow_new = allow_new
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ user controls (thread-safe)
    def set_mode(self, mode) -> None:
        with self._lock:
            self.mode = FaceMode.parse(mode)

    def set_keep_visible(self, ids: Iterable[str]) -> None:
        with self._lock:
            self.keep_visible = set(ids)

    def set_selected(self, ids: Iterable[str]) -> None:
        with self._lock:
            self.selected = set(ids)

    def toggle(self, face_id: str) -> bool:
        """Flip the choice that matters in the CURRENT mode; returns the new 'blurred' state."""
        with self._lock:
            if self.mode == FaceMode.AUTO:
                self.keep_visible.symmetric_difference_update({face_id})
            elif self.mode == FaceMode.SELECTED:
                self.selected.symmetric_difference_update({face_id})
            return self.is_blurred(face_id)

    def is_blurred(self, face_id: Optional[str]) -> bool:
        with self._lock:
            if self.mode == FaceMode.EVERY or face_id is None:
                return True
            if self.mode == FaceMode.AUTO:
                return face_id not in self.keep_visible
            return face_id in self.selected

    def snapshot(self) -> dict:
        with self._lock:
            return {"mode": self.mode.value, "keep_visible": sorted(self.keep_visible), "selected": sorted(self.selected)}

    # ------------------------------------------------------------------ identity resolution
    def identify(self, frame: np.ndarray, box, frame_number: int = 0) -> Optional[str]:
        """Resolve a face to an identity id (or None when it cannot be identified)."""
        found = self.book.embed(frame, box)
        if found is None:
            return None
        embedding, crop = found
        with self._lock:
            mode, selected = self.mode, set(self.selected)
        face_id, sim = self.book.best_match(embedding)
        if mode == FaceMode.SELECTED and selected:
            # bias to blur: a looser match against SELECTED people so one of them is not missed
            sel_id, sel_sim = self.book.best_match(embedding, only=selected)
            if sel_id is not None and sel_sim >= SELECTED_THRESHOLD and (face_id is None or sel_sim >= sim - 0.02 or sim < MERGE_THRESHOLD):
                face_id, sim = sel_id, sel_sim
        threshold = MERGE_THRESHOLD
        if face_id is not None and sim >= (SELECTED_THRESHOLD if face_id in selected and mode == FaceMode.SELECTED else threshold):
            self.book.observe(face_id, embedding, crop, frame_number)
            return face_id
        if not self.allow_new:
            return None
        return self.book.create(embedding, crop, frame_number).face_id

    def decide(self, face_id: Optional[str]) -> tuple:
        """(blur, reason) for an identity under the current mode - the single rule every code path uses."""
        with self._lock:
            mode = self.mode
        if mode == FaceMode.EVERY:
            return True, "every_face"
        if face_id is None:
            return True, "unidentified_blurred"
        blur = self.is_blurred(face_id)
        if mode == FaceMode.AUTO:
            return blur, "auto_blur" if blur else "kept_visible"
        return blur, "selected" if blur else "not_selected"

    def verdict(self, frame: np.ndarray, track: dict, box, frame_number: int = 0) -> FaceVerdict:
        """Decide one face. The result for a track is cached on the track so a person keeps one identity."""
        if "face_id" not in track:
            track["face_id"] = self.identify(frame, box, frame_number)
        face_id = track["face_id"]
        ident = self.book.get(face_id) if face_id else None
        blur, reason = self.decide(face_id)
        return FaceVerdict(face_id, ident.label if ident else "Unidentified face", blur, reason)
