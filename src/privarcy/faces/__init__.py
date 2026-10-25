"""
face_privacy

Two-stage system for "blur everyone except enrolled people":
  1. Detect every face in a frame (OpenCV Haar cascade — bundled with
     opencv-python, no extra download).
  2. For each face, compute a feature embedding and compare it against
     everyone enrolled in the D2 Face Registry (data_store.FaceRegistryDB).
     A close-enough match -> known, leave alone. No match -> unknown,
     hand off to `redaction.FrameRedactor` to blur.

This is a from-scratch, dependency-light baseline (Histogram-of-Oriented-
Gradients embeddings + cosine similarity), not a deep-learning face
recognizer — see the README for accuracy expectations and the upgrade
path to a stronger model.

Public API:
    FaceDetector       - finds face bounding boxes in a frame
    FaceEmbedder          - turns a face crop into a feature vector
    FaceMatcher              - compares an embedding against enrolled people
    enroll_person               - convenience: detect + embed + store in one call
"""

from .exceptions import FacePrivacyError
from .models import FaceBox, MatchResult
from .detector import FaceDetector, YoloFaceDetector, CompositeFaceDetector, build_face_detector, opencv_diagnosis
from .embedder import FaceEmbedder
from .matcher import FaceMatcher
from .identity import FaceBook, FaceIdentity
from .policy import FaceMode, FacePolicy, FaceVerdict, MODE_HELP, MODE_TITLES

__all__ = [
    "FaceDetector",
    "YoloFaceDetector",
    "CompositeFaceDetector",
    "build_face_detector",
    "opencv_diagnosis",
    "FaceEmbedder",
    "FaceMatcher",
    "FaceBook",
    "FaceIdentity",
    "FaceMode",
    "FacePolicy",
    "FaceVerdict",
    "MODE_HELP",
    "MODE_TITLES",
    "FaceBox",
    "MatchResult",
    "FacePrivacyError",
]

__version__ = "1.0.0"
