"""Camera sources: URL validation, credential hygiene, discovery, probing. Cameras are MOCKED; logic is real."""
import time

import numpy as np
import pytest

from privarcy.capture import (CameraSource, PHONE_PRESETS, StreamURLError, build_phone_url, discover_local_cameras, mask_url,
                              open_capture, probe_source, validate_stream_url)
from privarcy.capture.security import with_credentials


@pytest.mark.parametrize("url", ["rtsp://192.168.1.20:554/stream1", "http://10.0.0.5:8080/video", "https://cam.local/mjpeg", "rtsp://cam.example.com/live"])
def test_valid_urls_are_accepted(url):
    assert validate_stream_url(url).startswith(url.split("://")[0] + "://")


@pytest.mark.parametrize("url,words", [
    ("", "Enter"), ("192.168.1.2:554/x", "must start"), ("file:///etc/passwd", "not supported"), ("ftp://x/y", "not supported"),
    ("javascript:alert(1)", "must start"), ("rtsp://user:pw@10.0.0.1/x", "Username and Password"), ("rtsp://10.0.0.1:99999/x", "valid"),
    ("http://", "host"), ("rtsp://bad host/x", "valid")])
def test_bad_urls_are_refused_with_a_helpful_message(url, words):
    with pytest.raises(StreamURLError) as e:
        validate_stream_url(url)
    assert words.lower() in str(e.value).lower()


def test_credentials_are_separate_masked_everywhere_and_encoded_only_at_open_time(caplog):
    src = CameraSource.stream("rtsp://10.0.0.9:554/s", "admin", "p@ss:w/rd")
    assert "p@ss" not in repr(src) and "p@ss" not in str(src) and "p@ss" not in src.display and "admin" not in src.display
    assert src.url == "rtsp://10.0.0.9:554/s"
    assert src.target() == "rtsp://admin:p%40ss%3Aw%2Frd@10.0.0.9:554/s"
    assert mask_url(src.target()) == "rtsp://***@10.0.0.9:554/s"
    assert mask_url("Failed rtsp://u:pw@h/x and http://a:b@c/d") == "Failed rtsp://***@h/x and http://***@c/d"
    assert with_credentials("http://h/x", None, "pw") == "http://h/x"          # a password alone is ignored


def test_phone_presets_build_valid_urls_and_explain_themselves():
    assert build_phone_url("ip_webcam", "192.168.1.50") == "http://192.168.1.50:8080/video"
    assert build_phone_url("droidcam", "192.168.1.50", 4748) == "http://192.168.1.50:4748/video"
    assert all(p["hint"] and p["title"] for p in PHONE_PRESETS.values())
    with pytest.raises(StreamURLError):
        build_phone_url("ip_webcam", "not a host!")


class Cap:
    def __init__(self, opened=True, frame=True, boom=None):
        self.opened, self.frame, self.boom, self.released = opened, frame, boom, False
    def isOpened(self): return self.opened
    def read(self):
        if self.boom: raise RuntimeError(self.boom)
        return (True, np.zeros((48, 64, 3), np.uint8)) if self.frame else (False, None)
    def get(self, p): return 25.0
    def release(self): self.released = True


def test_discovery_lists_only_cameras_that_deliver_a_frame_and_always_releases():
    caps = {0: Cap(), 1: Cap(opened=False), 2: Cap(frame=False), 3: Cap()}
    found = discover_local_cameras(4, opener=lambda i: caps[i])
    assert [f["index"] for f in found] == [0, 3] and all(c.released for c in caps.values())


def test_probe_success_and_every_failure_releases_the_camera_and_hides_the_password():
    ok = Cap(); r = probe_source(CameraSource.local(0), 3, opener=lambda s: ok)
    assert r.ok and (r.width, r.height) == (64, 48) and ok.released
    closed = Cap(opened=False)
    r = probe_source(CameraSource.stream("rtsp://10.0.0.9/s", "admin", "secret"), 3, opener=lambda s: closed)
    assert not r.ok and "secret" not in r.message and "username and password" in r.message.lower() and closed.released
    nofr = Cap(frame=False); r = probe_source(CameraSource.local(1), 3, opener=lambda s: nofr)
    assert not r.ok and "no video frame" in r.message.lower() and nofr.released
    boom = Cap(boom="cannot reach rtsp://admin:secret@10.0.0.9/s"); r = probe_source(CameraSource.local(1), 3, opener=lambda s: boom)
    assert not r.ok and "secret" not in r.message and boom.released


def test_probe_times_out_instead_of_hanging_the_app():
    class Slow(Cap):
        def isOpened(self): time.sleep(2); return True
    started = time.time()
    r = probe_source(CameraSource.local(0), timeout=0.3, opener=lambda s: Slow())
    assert not r.ok and "No answer" in r.message and time.time() - started < 1.5


def test_open_capture_uses_injected_opener_and_accepts_plain_index():
    seen = []
    open_capture(2, opener=lambda s: seen.append(s) or Cap())
    assert seen[0].kind == "local" and seen[0].index == 2
