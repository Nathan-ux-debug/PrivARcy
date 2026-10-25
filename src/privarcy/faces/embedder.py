"""Face embedding.

Turns a cropped face image into a fixed-length feature vector so two
faces can be compared numerically. Uses OpenCV's built-in HOG (Histogram
of Oriented Gradients) descriptor — no extra dependency or model
download, unlike a deep embedding network (FaceNet/ArcFace).

Trade-off, stated plainly: HOG captures edge/gradient structure, not a
learned notion of "identity" the way a deep face-recognition network
does. It works reasonably for a small enrolled set under fairly
consistent conditions (similar angle/lighting to the enrollment photos),
but is meaningfully less robust than a deep embedding model to pose,
lighting, and expression changes. See the README for the upgrade path.
"""

import logging

import cv2
import numpy as np

from .exceptions import FacePrivacyError

logger = logging.getLogger(__name__)

# Fixed size every face crop is resized to before computing HOG features,
# so all embeddings are the same length and directly comparable.
_FACE_SIZE = (96, 96)
_HOG = (
    cv2.HOGDescriptor(_winSize=_FACE_SIZE, _blockSize=(16, 16), _blockStride=(8, 8), _cellSize=(8, 8), _nbins=9)
    if hasattr(cv2, "HOGDescriptor")
    else None
)


def _numpy_hog(image: np.ndarray) -> np.ndarray:
    """Compute the same fixed-size HOG feature when OpenCV 5 omits HOGDescriptor."""
    gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=1)
    gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=1)
    magnitude = np.hypot(gradient_x, gradient_y)
    orientation = np.mod(np.arctan2(gradient_y, gradient_x), np.pi)
    bins = orientation * (9 / np.pi)
    cells = []
    for y in range(0, image.shape[0], 8):
        row = []
        for x in range(0, image.shape[1], 8):
            cell_magnitude = magnitude[y:y + 8, x:x + 8]
            cell_bins = bins[y:y + 8, x:x + 8]
            histogram = np.zeros(9, dtype=np.float64)
            lower = np.floor(cell_bins).astype(int) % 9
            for bin_index in range(9):
                histogram[bin_index] = cell_magnitude[lower == bin_index].sum()
            row.append(histogram)
        cells.append(row)

    features = []
    for y in range(len(cells) - 1):
        for x in range(len(cells[y]) - 1):
            block = np.concatenate((cells[y][x], cells[y][x + 1], cells[y + 1][x], cells[y + 1][x + 1]))
            block /= np.sqrt(np.dot(block, block) + 1e-6 ** 2)
            block = np.minimum(block, 0.2)
            block /= np.sqrt(np.dot(block, block) + 1e-6 ** 2)
            features.extend(block)
    return np.asarray(features, dtype=np.float64)


class FaceEmbedder:
    """Computes a HOG-based embedding vector for a cropped face image."""

    def embed(self, face_crop: np.ndarray) -> np.ndarray:
        """Return a 1-D, L2-normalized feature vector for one face crop.

        Args:
            face_crop: BGR (or grayscale) NumPy array containing just the
                        face region — e.g. frame[y1:y2, x1:x2] from a
                        FaceBox.

        Raises:
            FacePrivacyError: if face_crop is empty/invalid.
        """
        if face_crop is None or not isinstance(face_crop, np.ndarray) or face_crop.size == 0:
            raise FacePrivacyError("embed() received an empty or invalid face crop.")

        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY) if face_crop.ndim == 3 else face_crop
        resized = cv2.resize(gray, _FACE_SIZE)
        # Light histogram equalization helps normalize lighting differences
        # between the enrollment photo and a later live frame.
        equalized = cv2.equalizeHist(resized)

        vector = (
            _HOG.compute(equalized).flatten().astype(np.float64)
            if _HOG is not None
            else _numpy_hog(equalized)
        )
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector
