"""UI integration tests (offscreen Qt).

REAL: Qt widgets, BackendService, storage, pipeline, correction log, rule store, face policy, preview engine.
MOCK: the YOLO model (scripted detections) and the camera (synthetic frames).
Screenshots are written to $PRIVARCY_SHOTS (default: <tmp>/screenshots) so a human can look at them.
"""
import os
import tempfile
import time
from pathlib import Path

import cv2
import pytest
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QApplication, QWidget

from tests.pipeline.test_live_faces import Cap
from tests.support.fakes import FakeYolo, gradient_frame

SHOTS = Path(os.environ.get("PRIVARCY_SHOTS", Path(tempfile.gettempdir()) / "privarcy_screenshots"))
SHOTS.mkdir(parents=True, exist_ok=True)
FACE_A, FACE_B = (40, 50, 130, 150), (300, 60, 390, 160)


def pump(app, seconds=0.0, until=None, timeout=15.0):
    end = time.time() + (timeout if until else seconds)
    while time.time() < end:
        app.processEvents()
        if until and until():
            return True
        time.sleep(0.01)
    return until() if until else True


def make_video(path, frames=40, w=480, h=320):
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
    for i in range(frames):
        f = gradient_frame(w, h)
        f[50:150, 40:130] = (60, 120, 200)        # two distinct flat "faces"
        f[60:160, 300:390] = (200, 90, 60)
        wr.write(f)
    wr.release()
    return path


@pytest.fixture(scope="module")
def qapp():
    from privarcy.app import theme
    app = QApplication.instance() or QApplication([])
    app.setStyleSheet(theme.stylesheet())
    return app


@pytest.fixture(scope="module")
def window(qapp, tmp_path_factory):
    import privarcy.app.shell as shell
    from privarcy.services.backend import BackendService

    tmp = tmp_path_factory.mktemp("proj")
    both = [("face", 0.95, FACE_A), ("face", 0.95, FACE_B), ("id_card", 0.62, (250, 220, 400, 300)), ("credit_card", 0.9, (200, 250, 300, 300))]
    model = FakeYolo([both] * 100000)
    service = BackendService(tmp / "data", model=model)
    service.update_settings(ocr_enabled=False, redaction_method="Solid Mask", detection_interval=1)

    class FakeLoader(QObject):
        loaded = pyqtSignal(object)
        failed = pyqtSignal(str)
        def start(self):
            self.loaded.emit(service)
    shell.BackendLoader = FakeLoader
    from PyQt5.QtCore import QSettings
    win = shell.MainWindow("", settings=QSettings(str(tmp / "ui.ini"), QSettings.IniFormat))
    win.resize(1360, 900)
    win.show()
    win.tmp, win.model = tmp, model
    pump(qapp, 0.3)
    yield win
    win.close()


def shot(win, name):
    QApplication.processEvents()
    win.grab().save(str(SHOTS / f"{name}.png"))


def test_starts_with_two_mode_cards_and_no_face_registry(qapp, window):
    assert window.top_stack.currentIndex() == 2
    assert "Face Registry" not in window.nav_buttons and "Face Registry" not in window.pages
    texts = [w.text() for w in window.pages["Dashboard"].findChildren(type(window.brand_name))]
    assert any("Offline" in t for t in texts) and any("Realtime" in t for t in texts)
    shot(window, "01_dashboard")


def test_all_sections_navigable(qapp, window):
    for name in ["Dashboard", "Video Processing", "Live Camera", "Review Queue", "Correction Log", "History & Outputs", "Settings"]:
        window.navigate(name)
        pump(qapp, 0.05)
        assert window.page_stack.currentWidget() is window.pages[name]
        assert window.nav_buttons[name].isChecked()


def test_sidebar_is_big_with_icons_and_status_pill(qapp, window):
    assert window.sidebar.width() >= 260
    assert all(not b.icon().isNull() and b.iconSize().width() >= 22 for name, b in window.nav_buttons.items())
    assert "OCR" in window.status_pill.text() or "Models" in window.status_pill.text()


def test_settings_hover_menu_opens_navigates_and_closes(qapp, window):
    window.navigate("Dashboard")
    window.settings_hover.show()
    pump(qapp, 0.05)
    menu = window.settings_menu
    assert menu.isVisible() and {"redaction", "classes", "thresholds", "camera", "storage", "engines", "privacy"} <= set(menu.items)
    shot(window, "02_settings_menu")
    menu.items["classes"].click()
    pump(qapp, 0.05)
    assert window.current == "Settings" and not menu.isVisible()
    window.profile_hover.show()
    assert window.profile_menu.isVisible() and {"prefs", "about", "docs", "changelog", "diag"} <= set(window.profile_menu.items)
    shot(window, "03_profile_menu")
    window.profile_hover.hide()
    assert not window.profile_menu.isVisible()


def test_hover_menu_closes_by_itself_after_the_pointer_leaves(qapp, window):
    window.settings_hover.show()
    window.settings_hover._timer.setInterval(30)
    window.settings_hover._timer.start()
    assert pump(qapp, until=lambda: not window.settings_menu.isVisible(), timeout=3)


def test_sidebar_collapse_and_expand(qapp, window):
    window.set_collapsed(True)
    pump(qapp, 0.05)
    assert window.sidebar.width() < 100 and window.nav_buttons["Dashboard"].text() == ""
    window.set_collapsed(False)
    assert window.sidebar.width() >= 260 and "Dashboard" in window.nav_buttons["Dashboard"].text()


def test_invalid_video_gives_clear_error(qapp, window, tmp_path):
    bad = tmp_path / "x.mp4"
    bad.write_bytes(b"nope")
    window.open_video(str(bad))
    page = window.pages["Video Processing"]
    assert page.banner.isVisible() and "can't be processed" in page.banner.text.text()
    assert not page.start_btn.isEnabled()


def test_editor_preview_responds_instantly_to_every_option_without_new_inference(qapp, window, tmp_path):
    src = make_video(tmp_path / "clip.mp4")
    window.open_video(str(src))
    page = window.pages["Video Processing"]
    assert pump(qapp, until=lambda: page.analysis is not None, timeout=30), "scan + analysis did not finish"
    assert len(page.policy.book.identities()) == 2 and len(page._tiles) == 2
    shot(window, "04_editor_auto")
    calls = window.model.calls
    def region_is_masked(box):
        x1, y1, x2, y2 = box
        return bool((page.preview.render(page.analysis, page.plan).image[y1:y2, x1:x2] == 0).all())
    assert region_is_masked(FACE_A) and region_is_masked(FACE_B)             # auto: everyone blurred
    first = next(iter(page._tiles))
    page._tile_clicked(first)                                                # keep Face 01 visible
    assert page.policy.decide(first)[0] is False
    visible = [b for b in (FACE_A, FACE_B) if not region_is_masked(b)]
    assert len(visible) == 1
    page.chips["credit_card"].setChecked(False)                              # chip off -> card no longer redacted
    assert window.service.config.enabled_classes.isdisjoint({"credit_card"})
    page.chips["credit_card"].setChecked(True)
    page.mode.set_current("every")                                           # every face: selection ignored
    assert region_is_masked(FACE_A) and region_is_masked(FACE_B)
    page.mode.set_current("selected")
    assert not region_is_masked(FACE_A) and not region_is_masked(FACE_B)     # nobody selected yet
    page._tile_clicked(first)
    assert sum(region_is_masked(b) for b in (FACE_A, FACE_B)) == 1
    assert window.model.calls == calls, "options must never trigger the detector again"
    assert "Blur selected" in page.summary_label.text()
    shot(window, "05_editor_selected")


def test_manual_and_magic_blur_in_the_editor(qapp, window):
    page = window.pages["Video Processing"]
    page.tabs.set_current("manual")
    page._region_drawn(10, 10, 100, 40)
    assert len(page.plan.manual) == 1 and page.stack.currentIndex() == 1
    out = page.preview.render(page.analysis, page.plan, 0.0).image
    assert (out[12:38, 12:98] == 0).all()
    page._delete_region(page.plan.manual[0])
    assert not page.plan.manual
    page.tabs.set_current("magic")
    page._canvas_clicked(5, 5)                                               # nothing there
    assert page.banner.isVisible() and not page.plan.magic
    page._canvas_clicked(*(sum(p) / 2 for p in zip(FACE_A[:2], FACE_A[2:])))
    assert len(page.plan.magic) == 1
    shot(window, "06_editor_magic")
    page.plan.magic.clear(); page._refresh_magic()
    page.tabs.set_current("auto")


def test_processing_from_the_editor_respects_the_face_choice(qapp, window):
    page = window.pages["Video Processing"]
    page.mode.set_current("selected")
    ids = list(page._tiles)
    page.policy.set_selected({ids[1]})
    page._refresh_faces()
    page.plan.manual.clear()
    page.start()
    assert not page.start_btn.isEnabled() and page.cancel_btn.isEnabled()
    assert pump(qapp, until=lambda: page.worker is None, timeout=90)
    assert page.status_pill.text() == "Completed" and page.result_card.isVisible()
    job = window.service.get_history(1)[0]
    cap = cv2.VideoCapture(job.output_path); ok, last = cap.read(); cap.release()
    masked = [bool(last[y1 + 12:y2 - 12, x1 + 12:x2 - 12].mean() < 12) for (x1, y1, x2, y2) in (FACE_A, FACE_B)]   # lossy codec: compare means
    assert masked.count(True) == 1                                           # exactly the selected person is blurred
    shot(window, "07_editor_done")


def test_dashboard_and_history_show_real_job(qapp, window):
    window.navigate("Dashboard"); pump(qapp, 0.1)
    assert "clip.mp4" in [w.text() for w in window.pages["Dashboard"].findChildren(type(window.brand_name)) if w.text() == "clip.mp4"]
    window.navigate("History & Outputs"); pump(qapp, 0.1)



def test_review_confirm_logs_and_updates_video(qapp, window):
    window.navigate("Review Queue"); pump(qapp, 0.2)
    page = window.pages["Review Queue"]
    assert len(page.items) == 1 and page.current["label"] == "id_card"
    assert page.confirm_btn.isEnabled()
    shot(window, "06_review")
    page.note.setText("looks like a real ID")
    page.confirm_btn.click()
    assert pump(qapp, until=lambda: page.worker is None and not page.items, timeout=30)
    assert "Confirmed: redacted 40" in page.banner.text.text()
    recs = window.service.list_corrections()
    assert recs[0].decision == "accepted" and recs[0].confidence > 0.6 and recs[0].bbox[2] > recs[0].bbox[0]
    assert window.service.overview()["pending_review"] == 0




def test_corrections_page_proposal_flow(qapp, window):
    svc = window.service
    for i in range(12):
        svc.correction_db.record_decision(f"s:{i}", [1, 2, 3, 4], "accepted", "credit_card", 0.62 + i * 0.01)
    window.navigate("Correction Log"); pump(qapp, 0.2)
    page = window.pages["Correction Log"]
    shot(window, "10_corrections")
    data = svc.rule_proposals()
    assert [i["proposal"].class_name for i in data["items"]] == ["credit_card"]
    v = svc.approve_proposal(data["items"][0]["proposal"], data["items"][0]["validation"])
    assert svc.config.class_auto_redact_confidence == {"credit_card": 0.6}
    svc.rollback_rules(0)
    assert svc.config.class_auto_redact_confidence == {}
    page.on_show()




@pytest.fixture
def mock_camera(monkeypatch, window):
    import privarcy.pipeline.live_session as ls
    monkeypatch.setattr(ls, "open_capture", lambda source, opener=None: Cap(frames=_face_cap_frame()))
    monkeypatch.setattr(window.service, "discover_cameras", lambda max_index=5: [{"index": 0, "label": "Mock camera"}])


class ClosedCap(Cap):
    def isOpened(self):
        return False


def _face_cap_frame():
    f = gradient_frame(480, 320)
    f[50:150, 40:130] = (60, 120, 200)
    f[60:160, 300:390] = (200, 90, 60)
    return f


def test_live_page_lifecycle_face_selection_and_recording(qapp, window, mock_camera):
    window.navigate("Live Camera")
    page = window.pages["Live Camera"]
    assert pump(qapp, until=lambda: page.camera_combo.currentData() == 0, timeout=5)
    page.start_btn.click()
    assert pump(qapp, until=lambda: page.session is not None, timeout=5)
    assert pump(qapp, until=lambda: page.session.stats["frames"] > 8 and len(page._tiles) >= 1, timeout=15)
    assert page.status_pill.text() == "Live"
    first = next(iter(page._tiles))
    assert page.session.face_policy.decide(first)[0] is True                  # auto: blurred
    page._tile_clicked(first)                                                 # keep this one visible
    assert pump(qapp, until=lambda: not page.session.face_policy.decide(first)[0], timeout=3)
    shot(window, "08_live")
    page.stop_btn.click()
    assert page.session is None and window.service.live_session is None and page.start_btn.isEnabled()
    assert window.service.last_live_recording and Path(window.service.last_live_recording).exists()


def test_live_camera_error_is_shown_and_never_leaks_the_password(qapp, window, monkeypatch):
    import privarcy.pipeline.live_session as ls
    monkeypatch.setattr(ls, "open_capture", lambda source, opener=None: ClosedCap())
    page = window.pages["Live Camera"]
    window.navigate("Live Camera")
    page.kind.set_current("network")
    page.net_url.setText("rtsp://10.0.0.9:554/stream1")
    page.net_user.setText("admin"); page.net_pass.setText("hunter2-secret")
    page.start_btn.click()
    assert pump(qapp, until=lambda: page.banner.isVisible(), timeout=8)
    shown = page.banner.text.text()
    assert "Could not open" in shown and "hunter2-secret" not in shown and page.start_btn.isEnabled()


def test_live_source_validation_messages(qapp, window):
    page = window.pages["Live Camera"]
    page.kind.set_current("network")
    for bad, words in (("", "Enter the stream address"), ("file:///etc/passwd", "not supported"), ("rtsp://u:p@h/x", "Username and Password")):
        page.net_url.setText(bad)
        page.start_btn.click()
        assert words.lower() in page.banner.text.text().lower()
    page.kind.set_current("phone")
    page.phone_host.setText("")
    page.start_btn.click()
    assert "IP address" in page.banner.text.text()


def test_test_connection_reports_result(qapp, window, monkeypatch):
    from privarcy.capture import ProbeResult
    page = window.pages["Live Camera"]
    page.kind.set_current("network")
    page.net_url.setText("http://10.0.0.5:8080/video")
    monkeypatch.setattr(window.service, "probe_camera", lambda source, timeout=8.0: ProbeResult(True, "Connected: 640x480", 640, 480))
    page.test_btn.click()
    assert pump(qapp, until=lambda: "Connected" in page.test_label.text(), timeout=5)



def test_settings_save_changes_backend_and_persist(qapp, window):
    page = window.pages["Settings"]
    window.navigate("Settings")
    page.method.setCurrentText("Pixelate")
    page.face_mode.setCurrentIndex(page.face_mode.findData("every"))
    page.class_boxes["screen"].setChecked(False)
    page.save()
    cfg = window.service.config
    assert cfg.redaction_method == "Pixelate" and "screen" not in cfg.enabled_classes and window.default_face_mode == "every"
    assert str(window.settings.value("redaction_method")) == "Pixelate"
    page.method.setCurrentText("Solid Mask"); page.class_boxes["screen"].setChecked(True)
    page.face_mode.setCurrentIndex(page.face_mode.findData("auto")); page.save()


def test_window_resizes_without_horizontal_scroll(qapp, window):
    for w, h in ((1100, 700), (1366, 768), (1920, 1040)):
        window.resize(w, h)
        for name in window.pages:
            window.navigate(name); pump(qapp, 0.03)
            bar = window.pages[name].scroll.horizontalScrollBar()
            if bar.maximum() != 0:
                col = window.pages[name].scroll.column
                wide = sorted(((c.minimumSizeHint().width(), type(c).__name__, c.objectName(), c.text()[:30] if hasattr(c, "text") and callable(c.text) else "")
                               for c in col.findChildren(QWidget)), reverse=True)[:6]
                raise AssertionError(f"{name} scrolls horizontally at {w}px by {bar.maximum()}: {wide}")
    window.resize(1360, 820)
    window.navigate("Dashboard")
    shot(window, "12_dashboard_final")

