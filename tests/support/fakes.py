"""Test doubles. Anything fake is labelled MOCK in the test names/docstrings."""

import numpy as np

REAL_NAMES = {0: "id_card", 1: "document", 2: "screen", 3: "face", 4: "credit_card", 5: "license_plate"}


class _T:
    def __init__(self, v): self.v = v
    def item(self): return self.v
    def cpu(self): return self
    def numpy(self): return np.asarray(self.v, dtype=float)
    def __getitem__(self, i): return self


class _Box:
    def __init__(self, cls_id, conf, xyxy):
        self.cls = [_T(cls_id)]
        self.conf = [_T(conf)]
        self.xyxy = [_T(list(xyxy))]


class _Result:
    def __init__(self, boxes, names):
        self.boxes = boxes
        self.names = names


class FakeYolo:
    """MOCK YOLO: returns scripted detections per predict() call.

    script: list (one entry per call) of lists of (class_name, conf, (x1,y1,x2,y2)).
    Calls past the end of the script return no detections.
    """

    def __init__(self, script, names=None):
        self.names = dict(names or REAL_NAMES)
        self._ids = {v: k for k, v in self.names.items()}
        self.script = script
        self.calls = 0

    def predict(self, source=None, conf=0.25, imgsz=640, verbose=False):
        items = self.script[self.calls] if self.calls < len(self.script) else []
        self.calls += 1
        boxes = [_Box(self._ids[n], c, b) for (n, c, b) in items if c >= conf]
        return [_Result(boxes, self.names)]


def gradient_frame(w=320, h=240):
    """Non-uniform frame so blur / mask is detectable."""
    x = np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))
    y = np.tile(np.linspace(255, 0, h, dtype=np.uint8).reshape(-1, 1), (1, w))
    noise = (np.random.RandomState(0).rand(h, w) * 60).astype(np.uint8)
    return np.dstack([x, y, noise]).astype(np.uint8)


# ---------------------------------------------------------------- face helpers (MOCK embedder, synthetic "faces")
import cv2  # noqa: E402


class ColorEmbedder:
    """MOCK embedder: a face's identity is the blue-channel level painted into it (level 50*k -> person k)."""
    def embed(self, crop):
        k = int(round(float(crop[..., 0].mean()) / 50.0))
        vec = np.zeros(8, np.float32)
        vec[min(k, 7)] = 1.0
        return vec


class AngleEmbedder:
    """MOCK embedder with controllable cosine similarity: blue level b -> unit vector at angle b degrees."""
    def embed(self, crop):
        a = np.deg2rad(float(crop[..., 0].mean()))
        return np.array([np.cos(a), np.sin(a)], np.float32)


def face_frame(faces, w=320, h=240):
    """Gradient frame with flat coloured squares. faces: list of ((x1,y1,x2,y2), person_level)."""
    frame = gradient_frame(w, h)
    for (x1, y1, x2, y2), level in faces:
        frame[y1:y2, x1:x2] = (level, 120, 200)
    return frame


def is_masked(frame, box):
    x1, y1, x2, y2 = box
    return bool((frame[y1:y2, x1:x2] == 0).all())


class FaceDet:
    """Minimal detector double with FaceBox-like results, driven by scripted boxes per call."""
    def __init__(self, script):
        self.script, self.calls = script, 0

    def detect(self, frame):
        from privarcy.faces.models import FaceBox
        items = self.script[self.calls] if self.calls < len(self.script) else []
        self.calls += 1
        return [FaceBox(tuple(b), 0.9) for b in items]


class ColorFaceDetector:
    """MOCK detector for synthetic videos: finds the flat coloured squares drawn by ``face_frame``."""
    def detect(self, frame):
        from privarcy.faces.models import FaceBox
        mask = ((abs(frame[..., 1].astype(int) - 120) < 14) & (abs(frame[..., 2].astype(int) - 200) < 14)).astype(np.uint8)  # video codecs shift pixel values a little
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        return [FaceBox((int(s[0]), int(s[1]), int(s[0] + s[2]), int(s[1] + s[3])), 0.9) for s in stats[1:] if s[4] > 400]
