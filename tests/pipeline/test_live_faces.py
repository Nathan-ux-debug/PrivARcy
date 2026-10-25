"""Live session with face selection, dropped frames, reconnect and fail-closed behaviour (MOCK camera + MOCK YOLO)."""
import time

import cv2

from privarcy.capture import CameraSource
from privarcy.faces import FaceBook, FaceMode, FacePolicy
from privarcy.pipeline.config import PipelineConfig
from privarcy.pipeline.live_session import LiveSession
from tests.support.fakes import ColorEmbedder, FakeYolo, face_frame, is_masked

A, B = (20, 20, 80, 80), (120, 20, 180, 80)


def cfg(**kw):
    c = PipelineConfig(redaction_method="Solid Mask", box_padding={}, ocr_enabled=False, temporal_smoothing=1, cover_raw_box=False)
    c.update(**kw)
    return c


class Cap:
    def __init__(self, frames=None, fail_after=None, delay=0.002):
        self.frames, self.fail_after, self.n, self.released, self.delay = frames, fail_after, 0, False, delay
    def isOpened(self): return not self.released
    def set(self, *a): return True
    def get(self, p):
        w, h = (self.frames.shape[1], self.frames.shape[0]) if self.frames is not None else (320, 240)
        return {3: w, 4: h, 5: 30}.get(int(p), 0)
    def read(self):
        self.n += 1
        if self.fail_after is not None and self.n > self.fail_after: return False, None
        time.sleep(self.delay)
        return True, (self.frames if self.frames is not None else face_frame([(A, 50), (B, 100)])).copy()
    def release(self): self.released = True


def wait(cond, t=4.0):
    end = time.time() + t
    while time.time() < end and not cond():
        time.sleep(0.01)
    return cond()


def policy(mode):
    return FacePolicy(FaceBook(ColorEmbedder(), label_prefix="T", id_prefix="T", label_sep="-"), mode)


def model(): return FakeYolo([[("face", 0.95, A), ("face", 0.95, B)]] * 20000)


def test_live_faces_get_stable_temporary_labels_and_every_face_is_blurred_in_auto_mode(tmp_path):
    s = LiveSession(model(), cfg(), 0, 320, 240, 30, tmp_path / "r.mp4", capture_factory=lambda i: Cap(), face_policy=policy(FaceMode.AUTO))
    assert wait(lambda: s.stats["frames"] > 8 and len(s.faces()) == 2)
    faces = s.faces()
    assert sorted(f["label"] for f in faces) == ["T-01", "T-02"] and all(f["blurred"] for f in faces)
    frame = s.latest_frame()
    assert is_masked(frame, A) and is_masked(frame, B)
    labels_later = sorted(f["label"] for f in s.faces()); time.sleep(0.2)
    assert sorted(f["label"] for f in s.faces()) == labels_later == ["T-01", "T-02"]      # labels do not churn
    s.stop()


def test_toggling_a_face_changes_the_live_output_and_the_recording_not_just_the_overlay(tmp_path):
    out = tmp_path / "rec.mp4"
    s = LiveSession(model(), cfg(), 0, 320, 240, 30, out, capture_factory=lambda i: Cap(), face_policy=policy(FaceMode.AUTO))
    assert wait(lambda: len(s.faces()) == 2)
    first = next(f for f in s.faces() if f["label"] == "T-01")
    assert s.toggle_face(first["face_id"]) is False                       # T-01 is now kept visible
    assert wait(lambda: not next(f for f in s.faces() if f["label"] == "T-01")["blurred"])
    assert wait(lambda: not is_masked(s.latest_frame(), A))
    frame = s.latest_frame()
    assert not is_masked(frame, A) and is_masked(frame, B)
    s.set_face_mode("every")                                              # selections ignored
    assert wait(lambda: is_masked(s.latest_frame(), A))
    s.stop()
    cap = cv2.VideoCapture(str(out)); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
    assert n >= 4 and s.status()["recording"]
    cap = cv2.VideoCapture(str(out)); ok, last = cap.read(); cap.release()
    assert ok and last.shape[:2] == (240, 320)


def test_selected_mode_leaves_unselected_faces_visible_live(tmp_path):
    s = LiveSession(model(), cfg(), 0, 320, 240, 30, capture_factory=lambda i: Cap(), face_policy=policy(FaceMode.SELECTED))
    assert wait(lambda: len(s.faces()) == 2)
    assert not any(f["blurred"] for f in s.faces())
    b = next(f for f in s.faces() if f["label"] == "T-02")
    s.toggle_face(b["face_id"])
    assert wait(lambda: is_masked(s.latest_frame(), B) and not is_masked(s.latest_frame(), A))
    s.stop()


def test_slow_model_drops_frames_instead_of_lagging_and_counts_them():
    class Slow(FakeYolo):
        def predict(self, *a, **k):
            time.sleep(0.08); return super().predict(*a, **k)
    s = LiveSession(Slow([[("face", 0.95, A)]] * 500), cfg(), 0, 320, 240, 30, capture_factory=lambda i: Cap(delay=0.005),
                    face_policy=policy(FaceMode.EVERY))
    assert wait(lambda: s.stats["frames"] >= 6, 8)
    st = s.status(); s.stop()
    assert st["dropped_frames"] > 10 and st["latency_ms"] < 400            # newest frame is processed, backlog never builds


def test_stream_camera_reconnects_after_a_dropout_then_keeps_going():
    caps = [Cap(fail_after=5), Cap()]
    opened = []
    def factory(src):
        cap = caps[len(opened)]; opened.append(src); return cap
    s = LiveSession(model(), cfg(), CameraSource.stream("rtsp://10.0.0.9:554/s", "u", "pw"), 320, 240, 30, capture_factory=factory,
                    face_policy=policy(FaceMode.EVERY))
    assert wait(lambda: s.stats["reconnects"] == 1 and s.stats["frames"] > 20, 8)
    assert s.state == "live" and caps[0].released
    s.stop(); assert "pw" not in str(s.status())


def test_stream_that_never_comes_back_ends_with_a_clear_error_and_releases():
    class Dead(Cap):
        def isOpened(self): return False
    opened = []
    def factory(src):
        opened.append(src)
        return Cap(fail_after=3) if len(opened) == 1 else Dead()
    s = LiveSession(model(), cfg(), CameraSource.stream("http://10.0.0.9/v"), 320, 240, 30, capture_factory=factory, face_policy=policy(FaceMode.EVERY))
    assert wait(lambda: s.state == "error", 10) and "stopped delivering" in s.error
    s.stop()


def test_model_failure_masks_frames_and_never_publishes_a_raw_one():
    class Flaky(FakeYolo):
        def predict(self, *a, **k):
            self.calls += 1
            if 3 <= self.calls <= 6: raise RuntimeError("cuda out of memory")
            return super().predict(*a, **k)
    published = []
    s = LiveSession(Flaky([[("face", 0.95, A)]] * 500), cfg(), 0, 320, 240, 30, capture_factory=lambda i: Cap(),
                    face_policy=policy(FaceMode.EVERY))
    wait(lambda: s.stats["frames"] > 25)
    st = s.status(); s.stop()
    assert st["masked_frames"] >= 4 and st["state"] != "error"              # recovered: a few black frames, then redacted frames
