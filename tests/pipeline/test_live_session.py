"""Live session tests use a MOCK camera (no physical webcam). They prove lifecycle,
privacy and error handling logic only - NOT real webcam behaviour or real FPS."""
import time

from privarcy.capture.sources import CameraSource
from privarcy.pipeline.config import PipelineConfig
from privarcy.pipeline.live_session import LiveSession
from tests.support.fakes import FakeYolo, gradient_frame


class FakeCap:
    def __init__(self, opened=True, fail_after=None, fps=0):
        self.opened, self.fail_after, self.n, self.released = opened, fail_after, 0, False
    def isOpened(self): return self.opened and not self.released
    def set(self, *a): return True
    def get(self, prop): return {3: 320, 4: 240, 5: 30}.get(int(prop), 0)
    def read(self):
        self.n += 1
        if self.fail_after is not None and self.n > self.fail_after:
            return False, None
        time.sleep(0.002)
        return True, gradient_frame()
    def release(self): self.released = True


def cfg(): return PipelineConfig(redaction_method="Solid Mask", box_padding={}, ocr_enabled=False)


def wait(cond, t=3.0):
    end = time.time() + t
    while time.time() < end and not cond():
        time.sleep(0.01)
    return cond()


def model(): return FakeYolo([[("face", 0.95, (50, 50, 120, 120))]] * 5000)


def test_preview_frames_are_redacted_only(tmp_path):
    cap = FakeCap()
    s = LiveSession(model(), cfg(), 0, 320, 240, 30, tmp_path / "l.mp4", capture_factory=lambda i: cap)
    assert s.is_live and wait(lambda: s.latest_frame() is not None and s.stats["frames"] > 5)
    f = s.latest_frame()
    assert (f[55:115, 55:115] == 0).all()                      # face region never shown raw
    st = s.status()
    assert st["state"] == "live" and st["fps"] > 0 and st["latency_ms"] is not None
    s.stop()
    assert cap.released and s.status()["state"] == "stopped"


def test_phone_source_uses_live_pipeline_without_creating_a_recording():
    cap = FakeCap()
    source = CameraSource(kind="phone", name="Android phone")
    session = LiveSession(model(), cfg(), source, 320, 240, 30, capture_factory=lambda _source: cap)
    assert session.is_live and wait(lambda: session.stats["frames"] > 5)
    assert session.source.kind == "phone"
    assert session.output_path is None
    session.stop()
    assert cap.released


def test_stop_is_idempotent_and_repeated_cycles_release_resources(tmp_path):
    caps = []
    for i in range(5):
        cap = FakeCap(); caps.append(cap)
        s = LiveSession(model(), cfg(), 0, 320, 240, 30, tmp_path / f"{i}.mp4", capture_factory=lambda _i, c=cap: c)
        wait(lambda: s.stats["frames"] > 2); s.stop(); s.stop()
    assert all(c.released for c in caps)


def test_camera_cannot_open_reports_error():
    s = LiveSession(model(), cfg(), 0, 320, 240, 30, capture_factory=lambda i: FakeCap(opened=False))
    assert s.state == "error" and "Could not open the camera" in s.error and not s.is_live


def test_camera_disconnect_detected():
    s = LiveSession(model(), cfg(), 0, 320, 240, 30, capture_factory=lambda i: FakeCap(fail_after=5))
    assert wait(lambda: s.state == "error", 6.0) and "stopped delivering" in s.error
    s.stop()


def test_processing_failure_fails_closed_no_raw_frame_published():
    """MOCK model that always crashes: every published frame is black (never raw) and the session finally stops."""
    class Boom(FakeYolo):
        def predict(self, *a, **k): raise RuntimeError("inference crashed")
    s = LiveSession(Boom([]), cfg(), 0, 320, 240, 30, capture_factory=lambda i: FakeCap())
    assert wait(lambda: s.latest_frame() is not None)
    assert s.latest_frame().max() == 0                          # masked, not raw
    assert wait(lambda: s.state == "error", 6.0) and "unredacted" in s.error and s.status()["masked_frames"] >= 30
    s.stop()


def test_detection_interval_holds_boxes_between_runs():
    c = cfg(); c.detection_interval = 3
    cap = FakeCap()
    s = LiveSession(FakeYolo([[("face", 0.95, (50, 50, 120, 120))]] + [[]] * 500), c, 0, 320, 240, 30, capture_factory=lambda i: cap)
    assert wait(lambda: s.stats["frames"] >= 3)
    f = s.latest_frame(); s.stop()
    assert (f[55:115, 55:115] == 0).all()
