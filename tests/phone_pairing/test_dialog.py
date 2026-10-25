"""Small UI tests for the QR pairing dialog without loading the application backend."""

from __future__ import annotations

import os
from io import BytesIO
from types import SimpleNamespace

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import qrcode
from PyQt5.QtWidgets import QApplication

import privarcy.app.pages.live as live_page
from privarcy.app.pages.live import LivePage
from privarcy.capture.phone_pairing.dialog import PhonePairingDialog


class Signal:
    def connect(self, _callback):
        pass


class Worker:
    instances = []

    def __init__(self, function, *args, **kwargs):
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.done = Signal()
        self.error = Signal()
        self.finished = Signal()
        self.instances.append(self)

    def start(self):
        pass


class PairingView:
    url = "http://192.168.1.12:8100/pair/test-token"
    fingerprint = "AA:BB:CC"
    status = "waiting_for_phone"
    seconds_remaining = 300

    @staticmethod
    def qr_png() -> bytes:
        image = qrcode.make("http://192.168.1.12:8100/pair/test-token")
        output = BytesIO()
        image.save(output, format="PNG")
        return output.getvalue()


def test_dialog_shows_pairing_url_fingerprint_and_disconnect_control():
    app = QApplication.instance() or QApplication([])
    dialog = PhonePairingDialog(PairingView())
    disconnected = []
    dialog.disconnect_requested.connect(lambda: disconnected.append(True))
    dialog.show()

    assert dialog.url.text() == PairingView.url
    assert "AA:BB:CC" in dialog.fingerprint.text()
    assert "Waiting for phone" in dialog.status.text()
    assert dialog.regenerate_button.isEnabled()
    assert dialog.disconnect_button.isEnabled()
    assert not dialog.url.isVisible()
    dialog.details_button.click()
    assert dialog.url.isVisible()

    dialog.close()
    app.processEvents()
    assert disconnected == []
    assert not dialog.isVisible()
    dialog.show()

    dialog.disconnect_button.click()
    app.processEvents()
    assert disconnected == [True]
    assert not dialog.isVisible()


def test_live_page_exposes_qr_pairing_without_backend_startup():
    qapp = QApplication.instance() or QApplication([])

    class App:
        service = SimpleNamespace(config=SimpleNamespace(live_ocr_enabled=False), lan_addresses=list)
        default_face_mode = "auto"

    page = LivePage(App())
    assert page.pair_button.text() == "Connect Phone with QR"
    assert page.pair_button.isEnabled()
    page.close()
    page.deleteLater()
    qapp.processEvents()


def test_live_page_restores_saved_phone_listener_without_showing_qr():
    qapp = QApplication.instance() or QApplication([])
    restored = PairingView()

    class App:
        service = SimpleNamespace(
            config=SimpleNamespace(live_ocr_enabled=False),
            phone_pairing=restored,
        )
        default_face_mode = "auto"

    page = LivePage(App())

    assert page._pairing is restored
    assert page.pairing_timer.isActive()
    assert page.pair_button.text() == "Show Phone QR"
    assert page.pairing_dialog is None

    page.pairing_timer.stop()
    page.close()
    page.deleteLater()
    qapp.processEvents()


def test_closing_qr_popup_keeps_pairing_and_can_reopen_the_same_dialog():
    qapp = QApplication.instance() or QApplication([])

    class App:
        service = SimpleNamespace(config=SimpleNamespace(live_ocr_enabled=False), lan_addresses=list)
        default_face_mode = "auto"

    page = LivePage(App())
    page._phone_pairing_ready(PairingView())
    dialog = page.pairing_dialog
    assert dialog is not None
    page._set_running(True)

    dialog.close()
    qapp.processEvents()
    assert page._pairing is not None
    assert page.pairing_dialog is dialog
    assert page.pair_button.isEnabled()
    assert page.pair_button.text() == "Show Phone QR"

    page.pair_button.click()
    qapp.processEvents()
    assert page.pairing_dialog is dialog
    assert dialog.isVisible()
    page.pairing_timer.stop()
    dialog.close()
    page.close()
    page.deleteLater()
    qapp.processEvents()


def test_local_camera_starts_without_stopping_or_waiting_for_phone_pairing(monkeypatch):
    qapp = QApplication.instance() or QApplication([])

    class App:
        service = SimpleNamespace(
            config=SimpleNamespace(live_ocr_enabled=False),
            start_live=lambda *args, **kwargs: None,
            stop_live=lambda: None,
            stop_phone_pairing=lambda: stopped_pairing.append(True),
            live_measurements=lambda: [],
            last_live_recording=None,
        )
        default_face_mode = "auto"

        @staticmethod
        def set_busy(_busy):
            pass

    stopped_pairing = []

    Worker.instances.clear()
    monkeypatch.setattr(live_page, "FuncWorker", Worker)
    page = LivePage(App())
    page.camera_combo.setItemText(0, "Built-in webcam")
    page.camera_combo.setItemData(0, 0)
    page._pairing = SimpleNamespace(frame_received=False)

    page.start()

    worker = Worker.instances[-1]
    assert worker.function is page.app.service.start_live
    assert worker.args[0].kind == "local"
    assert worker.args[0].index == 0
    assert page._pairing is not None

    page.session = SimpleNamespace(source=SimpleNamespace(kind="local"))
    page.stop(silent=True)
    assert page._pairing is not None
    assert not stopped_pairing

    page.close()
    page.deleteLater()
    qapp.processEvents()


def test_stopping_phone_video_disconnects_stream_but_preserves_trusted_pairing():
    qapp = QApplication.instance() or QApplication([])
    disconnected = []
    stopped_listener = []
    trusted = [{"device_id": "a" * 32, "name": "Phone", "connected": True}]

    class App:
        service = SimpleNamespace(
            config=SimpleNamespace(live_ocr_enabled=False),
            paired_phone_devices=lambda: trusted,
            disconnect_phone_devices=lambda: disconnected.append(True),
            stop_phone_pairing=lambda: stopped_listener.append(True),
            stop_live=lambda: None,
            live_measurements=lambda: [],
            last_live_recording=None,
        )
        default_face_mode = "auto"

        @staticmethod
        def set_busy(_busy):
            pass

    page = LivePage(App())
    page._pairing = PairingView()
    page.session = SimpleNamespace(source=SimpleNamespace(kind="phone"))
    page.pairing_timer.start()

    page.stop(silent=True)

    assert disconnected == [True]
    assert not stopped_listener
    assert page._pairing is not None
    assert page.pairing_timer.isActive()

    page.pairing_timer.stop()
    page.close()
    page.deleteLater()
    qapp.processEvents()


def test_phone_frames_only_start_phone_capture_when_phone_is_selected(monkeypatch):
    qapp = QApplication.instance() or QApplication([])

    class App:
        service = SimpleNamespace(
            config=SimpleNamespace(live_ocr_enabled=False),
            start_live=lambda *args, **kwargs: None,
        )
        default_face_mode = "auto"

    Worker.instances.clear()
    monkeypatch.setattr(live_page, "FuncWorker", Worker)
    page = LivePage(App())
    pairing = SimpleNamespace(
        status="streaming",
        frame_received=True,
        capture=object(),
    )
    page._pairing = pairing

    page.kind.set_current("local")
    page._poll_phone_pairing()
    assert not Worker.instances

    page.kind.set_current("phone")
    page._poll_phone_pairing()
    worker = Worker.instances[-1]
    assert worker.args[0].kind == "phone"
    assert worker.kwargs["record"] is False
    assert worker.kwargs["capture_factory"](None) is pairing.capture

    page._pairing = None
    page.start_worker = None
    page.pairing_timer.stop()
    page.close()
    page.deleteLater()
    qapp.processEvents()


def test_live_preview_expands_and_preserves_frame_aspect_ratio():
    qapp = QApplication.instance() or QApplication([])

    class App:
        service = SimpleNamespace(config=SimpleNamespace(live_ocr_enabled=False), lan_addresses=list)
        default_face_mode = "auto"

    page = LivePage(App())
    page.resize(1360, 820)
    page.show()
    qapp.processEvents()
    original_width = page.canvas.width()
    page.canvas.set_frame(np.zeros((480, 640, 3), dtype=np.uint8))
    rect = page.canvas.image_rect()

    assert page.canvas.minimumHeight() >= 400
    assert page.canvas.sizePolicy().expandingDirections()
    assert page.canvas.height() > page.canvas.minimumHeight()
    assert page.canvas.parentWidget().height() >= page.canvas.height()
    assert rect.width() / rect.height() == 4 / 3

    page.resize(1100, 700)
    qapp.processEvents()
    assert page.canvas.width() < original_width

    page.close()
    page.deleteLater()
    qapp.processEvents()
