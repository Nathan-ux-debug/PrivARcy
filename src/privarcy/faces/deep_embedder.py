"""Deep face embedding (ArcFace via InsightFace).

Same interface as FaceEmbedder: embed(crop) -> 1-D L2-normalised vector.
The same person gives a similar vector across angle, lighting and distance,
which is what lets a person who leaves and returns be linked again.
"""

import logging

import cv2
import numpy as np

from .exceptions import FacePrivacyError

logger = logging.getLogger(__name__)

_MIN_SIDE = 112        # upscale small crops so the model can re-find the face


class DeepFaceEmbedder:
    def __init__(self, model_name: str = "buffalo_s"):
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise FacePrivacyError("insightface is not installed (pip install insightface onnxruntime).") from exc
        self._app = FaceAnalysis(
            name=model_name,
            allowed_modules=["detection", "recognition"],
            providers=["CPUExecutionProvider"],
        )
        self._app.prepare(ctx_id=-1, det_size=(192, 192))

    def embed(self, face_crop: np.ndarray) -> np.ndarray:
        if face_crop is None or not isinstance(face_crop, np.ndarray) or face_crop.size == 0:
            raise FacePrivacyError("embed() received an empty or invalid face crop.")
        img = face_crop if face_crop.ndim == 3 else cv2.cvtColor(face_crop, cv2.COLOR_GRAY2BGR)

        h, w = img.shape[:2]
        if min(h, w) < _MIN_SIDE:
            scale = _MIN_SIDE / min(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

        # The crop is tight, so add a margin for InsightFace's own detector/aligner.
        pad = max(img.shape[:2]) // 2
        img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)

        faces = self._app.get(img)
        if not faces:
            raise FacePrivacyError("No face found in crop.")     # FaceBook treats this as unidentified -> blurred
        best = max(faces, key=lambda f: f.det_score)
        vector = np.asarray(best.normed_embedding, dtype=np.float64)
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector