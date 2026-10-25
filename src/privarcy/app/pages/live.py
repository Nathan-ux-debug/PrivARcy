"""Live Camera: pick a source (built-in/USB, phone, network camera), choose who to blur, watch REDACTED frames only.

Nothing slow runs here: the camera opens in a worker, inference runs in the session's own threads and this page only
paints the newest redacted frame on a timer. Camera passwords live in the password field and in memory only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from PyQt5.QtCore import QTimer, QUrl
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLineEdit,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from privarcy.capture import PHONE_PRESETS, CameraSource, StreamURLError, validate_stream_url
from privarcy.capture.phone_pairing.dialog import PAIRING_STATES, PhonePairingDialog
from privarcy.capture.security import mask_url
from privarcy.faces import MODE_HELP, FaceMode
from privarcy.pipeline.preview import Overlay

from ..canvas import FrameCanvas
from ..widgets import Banner, Card, FaceTile, Page, Pill, Segmented, StatTile, bgr_to_pixmap, button, clear_layout, label
from ..workers import FuncWorker

RESOLUTIONS = [("640 × 480", 640, 480), ("1280 × 720", 1280, 720), ("1920 × 1080", 1920, 1080)]
FACE_MODES = [("auto", "Auto"), ("selected", "Selected only"), ("every", "Every face")]

class LivePage(Page):
    def __init__(self, app) -> None:
        super().__init__("Live Camera", "Choose a camera, then choose who to blur. Only redacted frames are ever shown or recorded.")
        self.app = app
        self.scroll.column.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.session = None
        self.scan_worker: Optional[FuncWorker] = None
        self.start_worker: Optional[FuncWorker] = None
        self.probe_worker: Optional[FuncWorker] = None
        self.pairing_worker: Optional[FuncWorker] = None
        self._pairing = None
        self.pairing_dialog: Optional[PhonePairingDialog] = None
        self._tiles: Dict[str, FaceTile] = {}
        self._tile_sig = None
        self._device_refresh_tick = 0
        self.timer = QTimer(self)
        self.timer.setInterval(40)
        self.timer.timeout.connect(self._tick)
        self._status_tick = 0
        self.pairing_timer = QTimer(self)
        self.pairing_timer.setInterval(300)
        self.pairing_timer.timeout.connect(self._poll_phone_pairing)

        self.banner = Banner()
        self.body.addWidget(self.banner)
        self.body.setSpacing(12)
        cols = QHBoxLayout()
        cols.setSpacing(12)
        cols.addLayout(self._build_left(), 7)
        cols.addLayout(self._build_right(), 3)
        self.body.addLayout(cols)
        self._set_running(False)
        self._refresh_paired_devices()
        restored_pairing = getattr(self.app.service, "phone_pairing", None)
        if restored_pairing is not None:
            self._pairing = restored_pairing
            self.pair_button.setText("Show Phone QR")
            self.pairing_timer.start()

    # ------------------------------------------------------------------ layout
    def _build_left(self) -> QVBoxLayout:
        left = QVBoxLayout()
        left.setSpacing(12)
        card = Card("")
        card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas = FrameCanvas("Camera is off")
        self.canvas.setMinimumHeight(400)
        self.canvas.setMaximumHeight(560)
        self.canvas.clicked.connect(self._canvas_clicked)
        card.add(self.canvas, 1)
        row = QHBoxLayout()
        self.status_pill = Pill("Stopped", "neutral")
        row.addWidget(self.status_pill)
        self.status_text = label("Pick a source and press Start.", "Muted", wrap=True)
        row.addWidget(self.status_text, 1)
        card.add(row)
        left.addWidget(card, 1)
        grid = QGridLayout()
        grid.setSpacing(10)
        self.stats = {k: StatTile(c) for k, c in (("fps", "FPS"), ("latency", "Latency (ms)"), ("inference", "Inference / frame (ms)"),
                                                   ("dropped", "Dropped frames"), ("redactions", "Redactions"))}
        for tile in self.stats.values():
            tile.caption_label.setWordWrap(True)
        for i, tile in enumerate(self.stats.values()):
            grid.addWidget(tile, i // 2, i % 2)
        left.addLayout(grid)
        self.verify = Card("Verification on this computer")
        self.verify_text = label("", "Faint", wrap=True)
        self.verify.add(self.verify_text)
        self.open_rec_btn = button("Open last recording", "", "folder", callback=self._open_recording)
        self.open_rec_btn.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.verify.add(self.open_rec_btn)
        left.addWidget(self.verify)
        return left

    def _build_right(self) -> QVBoxLayout:
        right = QVBoxLayout()
        right.setSpacing(12)
        src = Card("Camera source")
        self.kind = Segmented([("local", "Built-in/USB"), ("phone", "Phone"), ("network", "IP camera")], "local")
        self.kind.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.kind.changed.connect(lambda k: self.source_stack.setCurrentIndex({"local": 0, "phone": 1, "network": 2}[k]))
        src.add(self.kind)
        self.source_stack = QStackedWidget()
        self.source_stack.addWidget(self._build_local())
        self.source_stack.addWidget(self._build_phone())
        self.source_stack.addWidget(self._build_network())
        src.add(self.source_stack)
        self.pair_button = button("Connect Phone with QR", "Primary", "camera", callback=self._connect_phone)
        src.add(self.pair_button)
        self.test_label = label("", "Faint", wrap=True)
        src.add(self.test_label)
        self.test_btn = button("Test connection", "", "check", callback=self.test_connection)
        self.start_btn = button("Start", "Primary", "play", callback=self.start)
        self.stop_btn = button("Stop", "Danger", "stop", "#D92D20", callback=self.stop)
        btns = QVBoxLayout()
        btns.addWidget(self.test_btn)
        run_btns = QHBoxLayout()
        for b in (self.start_btn, self.stop_btn):
            run_btns.addWidget(b)
        btns.addLayout(run_btns)
        src.add(btns)
        right.addWidget(src)

        devices = Card("Trusted phones")
        self.paired_device_combo = QComboBox()
        self.paired_device_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.paired_device_combo.setMinimumContentsLength(14)
        devices.add(self.paired_device_combo)
        self.paired_device_status = label("Pair a phone to save it for future connections.", "Faint", wrap=True)
        devices.add(self.paired_device_status)
        self.revoke_device_btn = button(
            "Revoke selected phone", "Danger", "stop", "#D92D20", callback=self._revoke_selected_phone
        )
        devices.add(self.revoke_device_btn)
        right.addWidget(devices)

        faces = Card("Faces")
        self.mode = Segmented(FACE_MODES, getattr(self.app, "default_face_mode", "auto") if getattr(self.app, "default_face_mode", "auto") in dict(FACE_MODES) else "auto")
        self.mode.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.mode.changed.connect(self._mode_changed)
        faces.add(self.mode)
        self.mode_help = label(MODE_HELP[FaceMode.parse(self.mode.current())], "Faint", wrap=True)
        faces.add(self.mode_help)
        self.tiles_grid = QGridLayout()
        self.tiles_grid.setSpacing(8)
        faces.add(self.tiles_grid)
        self.no_faces = label("Faces will appear here when the camera sees them. Click a face in the video or a tile to change it.", "Faint", wrap=True)
        faces.add(self.no_faces)
        right.addWidget(faces)

        opts = Card("Safeguards")
        self.ocr_box = QCheckBox("Read text (OCR): slow")
        self.ocr_box.setChecked(self.app.service.config.live_ocr_enabled)
        self.ocr_box.toggled.connect(lambda v: self.app.service.update_settings(live_ocr_enabled=v))
        opts.add(self.ocr_box)
        opts.add(label("If detection fails, the frame is replaced by black, never shown unredacted. Dropped frames are counted. "
                       "Network cameras reconnect automatically a few times before the session stops.", "Faint", wrap=True))
        right.addWidget(opts)
        return right

    def _build_local(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w); lay.setContentsMargins(0, 8, 0, 0); lay.setSpacing(8)
        row = QHBoxLayout()
        self.camera_combo = QComboBox(); self.camera_combo.setMinimumWidth(100); self.camera_combo.addItem("Scanning for cameras…")
        self.camera_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.camera_combo.setMinimumContentsLength(12)
        self.res_combo = QComboBox()
        for text, *_ in RESOLUTIONS:
            self.res_combo.addItem(text)
        self.rescan_btn = button("", "", "refresh", callback=self.scan)
        self.rescan_btn.setToolTip("Refresh the camera list")
        row.addWidget(self.camera_combo, 1); row.addWidget(self.rescan_btn)
        lay.addLayout(row)
        lay.addWidget(self.res_combo)
        lay.addWidget(label("If your camera is missing, close other apps that use it and allow camera access for desktop apps in "
                            "Windows Settings → Privacy & security → Camera.", "Faint", wrap=True))
        return w

    def _build_phone(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w); lay.setContentsMargins(0, 8, 0, 0); lay.setSpacing(8)
        lay.addWidget(label(
            "Use the native PrivARcy Companion app to scan the QR code and stream the phone's built-in camera.",
            "Faint", wrap=True,
        ))
        lay.addWidget(label("Existing phone camera app stream (optional)", "SectionTitle", wrap=True))
        self.phone_preset = QComboBox()
        self.phone_preset.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.phone_preset.setMinimumContentsLength(14)
        for key, info in PHONE_PRESETS.items():
            self.phone_preset.addItem(info["title"], key)
        self.phone_preset.currentIndexChanged.connect(self._phone_changed)
        lay.addWidget(self.phone_preset)
        row = QHBoxLayout()
        self.phone_host = QLineEdit(); self.phone_host.setPlaceholderText("Phone IP, e.g. 192.168.1.50")
        self.phone_port = QSpinBox(); self.phone_port.setRange(1, 65535); self.phone_port.setMaximumWidth(90)
        row.addWidget(self.phone_host, 1); row.addWidget(self.phone_port)
        lay.addLayout(row)
        self.phone_hint = label("", "Faint", wrap=True)
        lay.addWidget(self.phone_hint)
        lay.addWidget(label(
            "These fields are only for existing camera apps that provide a video stream; QR pairing does not use them.",
            "Faint", wrap=True,
        ))
        self._phone_changed()
        return w

    def _build_network(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w); lay.setContentsMargins(0, 8, 0, 0); lay.setSpacing(8)
        self.net_url = QLineEdit(); self.net_url.setPlaceholderText("rtsp://192.168.1.20:554/stream1")
        self.net_user = QLineEdit(); self.net_user.setPlaceholderText("Username (optional)")
        self.net_pass = QLineEdit(); self.net_pass.setPlaceholderText("Password (optional)"); self.net_pass.setEchoMode(QLineEdit.Password)
        for f in (self.net_url, self.net_user, self.net_pass):
            lay.addWidget(f)
        lay.addWidget(label("Works with cameras that offer an RTSP (H.264) or HTTP/MJPEG stream. Not every camera does, and automatic "
                            "camera discovery (ONVIF) is not supported. The password is kept in memory only and is never saved or logged.",
                            "Faint", wrap=True))
        return w

    def _phone_changed(self) -> None:
        key = self.phone_preset.currentData()
        info = PHONE_PRESETS.get(key)
        if info:
            self.phone_port.setValue(info["port"])
            self.phone_hint.setText(info["hint"])

    # ------------------------------------------------------------------ cameras
    def scan(self) -> None:
        if self.scan_worker is not None:
            return
        self.camera_combo.clear()
        self.camera_combo.addItem("Scanning for cameras…")
        self.rescan_btn.setEnabled(False)
        self.scan_worker = FuncWorker(self.app.service.discover_cameras)
        self.scan_worker.done.connect(self._scan_done)
        self.scan_worker.error.connect(lambda m: self._scan_done([]))
        self.scan_worker.finished.connect(self._scan_finished)
        self.scan_worker.start()

    def _scan_done(self, cams: list) -> None:
        self.camera_combo.clear()
        if not cams:
            self.camera_combo.addItem("No camera found")
            self.camera_combo.setItemData(0, None)
            return
        for cam in cams:
            self.camera_combo.addItem(cam["label"], cam["index"])

    def _scan_finished(self) -> None:
        self.scan_worker = None
        self.rescan_btn.setEnabled(self.session is None)

    def _build_source(self):
        """Returns (CameraSource, width, height) or raises ValueError with a message for the user."""
        kind = self.kind.current()
        _, w, h = RESOLUTIONS[self.res_combo.currentIndex()]
        if kind == "local":
            index = self.camera_combo.currentData()
            if index is None:
                raise ValueError("No camera selected. Plug one in and press refresh.")
            return CameraSource.local(int(index), self.camera_combo.currentText()), w, h
        try:
            if kind == "phone":
                host = self.phone_host.text().strip()
                if not host:
                    raise ValueError("Enter the phone's IP address shown in the camera app.")
                url = self.app.service.phone_stream_url(self.phone_preset.currentData(), host, self.phone_port.value())
                return CameraSource.stream(url, name=f"Phone {host}"), w, h
            url = validate_stream_url(self.net_url.text())
            return CameraSource.stream(url, self.net_user.text().strip(), self.net_pass.text()), w, h
        except StreamURLError as exc:
            raise ValueError(str(exc)) from None

    def test_connection(self) -> None:
        if self.probe_worker is not None or self.session is not None:
            return
        try:
            source, *_ = self._build_source()
        except ValueError as exc:
            self.test_label.setText(str(exc)); self.test_label.setStyleSheet("color: #D92D20;")
            return
        self.test_label.setText("Testing…"); self.test_label.setStyleSheet("")
        self.test_btn.setEnabled(False)
        self.probe_worker = FuncWorker(self.app.service.probe_camera, source)
        self.probe_worker.done.connect(self._probe_done)
        self.probe_worker.error.connect(lambda m: self._probe_done(None, m))
        self.probe_worker.finished.connect(lambda: (setattr(self, "probe_worker", None), self.test_btn.setEnabled(self.session is None)))
        self.probe_worker.start()

    def _probe_done(self, result, error: str = "") -> None:
        ok = bool(result and result.ok)
        self.test_label.setText(mask_url(error or result.message))
        self.test_label.setStyleSheet("color: #12A150;" if ok else "color: #D92D20;")

    # ------------------------------------------------------------------ run
    def start(self) -> None:
        if self.session is not None or self.start_worker is not None:
            return
        if self.kind.current() == "phone" and self._pairing is not None:
            if self._pairing.frame_received:
                self._start_phone_capture(self._pairing)
            else:
                self.banner.show_message("Start the camera in the PrivARcy Companion app to connect the phone.", "info")
            return
        try:
            source, w, h = self._build_source()
        except ValueError as exc:
            self.banner.show_message(str(exc), "warn")
            return
        self.banner.hide()
        self._set_status("Starting", "info", f"Opening {source.display}…")
        self._set_running(True)
        mode = self.mode.current() or "auto"
        self.start_worker = FuncWorker(self.app.service.start_live, source, w, h, 30.0, mode)
        self.start_worker.done.connect(self._started)
        self.start_worker.error.connect(self._start_failed)
        self.start_worker.finished.connect(lambda: setattr(self, "start_worker", None))
        self.start_worker.start()

    def _started(self, session) -> None:
        self.session = session
        if session.state == "error":
            self.banner.show_message(mask_url(session.error or "Could not start the camera."), "danger")
            self.stop(silent=True)
            return
        self.app.set_busy(True)
        self._set_status("Live", "ok", f"{session.source.display} · redacted frames only")
        if self.pairing_dialog is not None and session.source.kind == "phone":
            self.pairing_dialog.update_status("streaming")
        self._tile_sig = None
        self.timer.start()

    def _start_failed(self, message: str) -> None:
        self.banner.show_message(f"Could not start the camera: {mask_url(message)}", "danger")
        self._set_running(False)
        self._set_status("Error", "danger", "The camera did not start.")
        if self._pairing is not None:
            trusted_devices = (
                self.app.service.paired_phone_devices()
                if hasattr(self.app.service, "paired_phone_devices")
                else []
            )
            if trusted_devices and self.kind.current() == "phone":
                self.app.service.disconnect_phone_devices()
                self._set_status(
                    "Waiting for phone",
                    "info",
                    "The camera stopped. The trusted phone can reconnect without a new QR scan.",
                )
            elif not trusted_devices:
                self._end_phone_pairing()

    def stop(self, silent: bool = False) -> None:
        self.timer.stop()
        had = self.session is not None
        stopped_source = self.session.source.kind if had else None
        recording = self.app.service.stop_live() if had else None
        trusted_devices = (
            self.app.service.paired_phone_devices()
            if hasattr(self.app.service, "paired_phone_devices")
            else []
        )
        preserve_pairing = self._pairing is not None and bool(trusted_devices)
        end_pairing = (
            self._pairing is not None
            and (not had or stopped_source == "phone")
            and not preserve_pairing
        )
        if had and stopped_source == "phone" and preserve_pairing:
            self.app.service.disconnect_phone_devices()
        if end_pairing:
            self.app.service.stop_phone_pairing()
            self._pairing = None
            self.pairing_timer.stop()
        if end_pairing and self.pairing_dialog is not None:
            dialog, self.pairing_dialog = self.pairing_dialog, None
            dialog.close_silently()
        self.pair_button.setEnabled(True)
        self.pair_button.setText("Show Phone QR" if self._pairing is not None else "Connect Phone with QR")
        self.session = None
        if had:
            self.app.set_busy(False)
        self._set_running(False)
        self.canvas.set_frame(None)
        self.canvas.placeholder = "Camera is off"
        self.canvas.update()
        self._clear_tiles()
        self._refresh_paired_devices()
        if not silent:
            self._set_status("Stopped", "neutral", "Stopped." + (" The redacted recording was saved." if recording else ""))
        self._refresh_verification()

    def _set_running(self, running: bool) -> None:
        self.start_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        self.test_btn.setEnabled(not running)
        self.rescan_btn.setEnabled(not running and self.scan_worker is None)
        self.kind.setEnabled(not running)
        self.source_stack.setEnabled(not running)
        if hasattr(self, "pair_button"):
            self.pair_button.setEnabled(
                self._pairing is not None or (not running and self.pairing_worker is None)
            )

    def _set_status(self, text: str, kind: str, detail: str) -> None:
        self.status_pill.set(text, kind)
        self.status_text.setText(detail)

    def _open_recording(self) -> None:
        path = self.app.service.last_live_recording
        if path and Path(path).exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))
        else:
            self.banner.show_message("There is no recording from this run yet.", "info")

    # ------------------------------------------------------------------ live loop
    def _tick(self) -> None:
        s = self.session
        if s is None:
            return
        frame = s.latest_frame()
        faces = s.faces()
        if frame is not None:
            overlays = [Overlay("face", f["bbox"], f["label"], "blurred" if f["blurred"] else "visible", f["face_id"]) for f in faces]
            self.canvas.set_frame(frame, overlays)
        self._refresh_tiles(s)
        self._status_tick += 1
        self._device_refresh_tick += 1
        if self._device_refresh_tick % 10 == 0:
            self._refresh_paired_devices()
        if self._status_tick % 12 == 0:
            self._update_status(s)
        if s.state == "error":
            self.banner.show_message(mask_url(s.error or "The camera session ended."), "danger")
            self.stop(silent=True)
            self._set_status("Error", "danger", "The session stopped. See the message above.")

    def _update_status(self, s) -> None:
        st = s.status()
        self.stats["fps"].set_value(f"{st['fps']:.1f}")
        self.stats["latency"].set_value("—" if st["latency_ms"] is None else f"{st['latency_ms']:.0f}")
        self.stats["inference"].set_value("—" if st["inference_ms"] is None else f"{st['inference_ms']:.0f}")
        self.stats["dropped"].set_value(str(st["dropped_frames"]))
        self.stats["redactions"].set_value(str(s.stats["redacted"]))
        if st["state"] == "reconnecting":
            self._set_status("Reconnecting", "warn", "The stream dropped. Trying to reconnect…")
        elif st["state"] == "live":
            extra = f" · {st['masked_frames']} frames masked after errors" if st["masked_frames"] else ""
            self._set_status("Live", "ok", f"{st['source']} · redacted frames only{extra}")

    # ------------------------------------------------------------------ faces
    def _mode_changed(self, key: str) -> None:
        self.mode_help.setText(MODE_HELP[FaceMode.parse(key)])
        self.app.settings.setValue("default_face_mode", key)
        self.app.default_face_mode = key
        if self.session is not None:
            self.session.set_face_mode(key)
            self._tile_sig = None

    def _canvas_clicked(self, x: float, y: float) -> None:
        if self.session is None:
            return
        overlay = self.canvas.overlay_at(x, y)
        if overlay is not None and overlay.face_id and FaceMode.parse(self.mode.current()) != FaceMode.EVERY:
            self.session.toggle_face(overlay.face_id)

    def _tile_clicked(self, face_id: str) -> None:
        if self.session is not None:
            self.session.toggle_face(face_id)
            self._tile_sig = None

    def _clear_tiles(self) -> None:
        clear_layout(self.tiles_grid)
        self._tiles.clear()
        self._tile_sig = None
        self.no_faces.show()

    def _refresh_tiles(self, session) -> None:
        policy = session.face_policy
        if policy is None:
            return
        idents = policy.book.identities()
        sig = (tuple(i.face_id for i in idents), tuple(policy.decide(i.face_id)[0] for i in idents), policy.mode)
        if sig == self._tile_sig:
            return
        self._tile_sig = sig
        if [i.face_id for i in idents] != list(self._tiles):
            clear_layout(self.tiles_grid)
            self._tiles.clear()
            thumbs = session.face_thumbnails()
            for n, ident in enumerate(idents):
                tile = FaceTile(ident.face_id, ident.label, bgr_to_pixmap(thumbs[ident.face_id]) if ident.face_id in thumbs else None)
                tile.clicked.connect(self._tile_clicked)
                self._tiles[ident.face_id] = tile
                self.tiles_grid.addWidget(tile, n // 2, n % 2)
        self.no_faces.setVisible(not idents)
        mode = policy.mode
        for fid, tile in self._tiles.items():
            blur, _ = policy.decide(fid)
            if mode == FaceMode.EVERY:
                tile.set_state("blurred", "Will be blurred", False)
            elif mode == FaceMode.AUTO:
                tile.set_state("blurred" if blur else "visible", "Will be blurred" if blur else "Kept visible")
            else:
                tile.set_state("blurred" if blur else "", "Will be blurred" if blur else "Stays visible")

    # ------------------------------------------------------------------ verification
    def _refresh_verification(self) -> None:
        data = self.app.service.live_measurements()
        if data:
            m = data[-1]

            resolution = m.get("resolution", [0, 0])

            if resolution is None:
                resolution = [0, 0]
            elif isinstance(resolution, dict):
                w = resolution.get("width", 0)
                h = resolution.get("height", 0)
            else:
                if hasattr(resolution, "ravel"):
                    resolution = resolution.ravel().tolist()

                try:
                    w, h = list(resolution)[:2]
                except (TypeError, ValueError):
                    w, h = 0, 0
                    
            self.verify_text.setText(f"Last measured session: {m['when']}  ·  {m['frames']} frames at {w}×{h}  ·  {m['avg_fps']} FPS  ·  "
                                     f"{m['latency_ms']} ms latency  ·  OCR {'on' if m.get('ocr') else 'off'}")
        else:
            self.verify_text.setText("No measured session yet. Run the camera for a few seconds; real numbers are recorded here.")
        self.open_rec_btn.setEnabled(bool(self.app.service.last_live_recording))

    def _connect_phone(self) -> None:
        if self.pairing_worker is not None:
            return
        if self._pairing is not None:
            if self.pairing_dialog is not None:
                self.pairing_dialog.show()
                self.pairing_dialog.raise_()
                self.pairing_dialog.activateWindow()
            return
        if self.session is not None:
            return
        self.kind.set_current("phone")
        self.banner.hide()
        self.pair_button.setEnabled(False)
        self._set_status("Starting", "info", "Starting the secure local pairing service…")
        self.pairing_worker = FuncWorker(self.app.service.start_phone_pairing)
        self.pairing_worker.done.connect(self._phone_pairing_ready)
        self.pairing_worker.error.connect(self._phone_pairing_failed)
        self.pairing_worker.finished.connect(lambda: setattr(self, "pairing_worker", None))
        self.pairing_worker.start()

    def _phone_pairing_ready(self, pairing) -> None:
        self._pairing = pairing
        if self.pairing_dialog is None:
            self.pairing_dialog = PhonePairingDialog(pairing, self)
            self.pairing_dialog.regenerate.connect(self._regenerate_phone_pairing)
            self.pairing_dialog.disconnect_requested.connect(self._end_phone_pairing)
        else:
            self.pairing_dialog.set_pairing(pairing)
        self.pairing_dialog.show()
        self.pairing_timer.start()
        self.pair_button.setText("Show Phone QR")
        self.pair_button.setEnabled(True)
        title, kind, detail = PAIRING_STATES["waiting_for_phone"]
        self._set_status(title, kind, detail)
        self._refresh_paired_devices()

    def _phone_pairing_failed(self, message: str) -> None:
        self.pair_button.setEnabled(True)
        self.pair_button.setText("Connect Phone with QR")
        self.banner.show_message(f"Could not start phone pairing: {message}", "danger")
        self._set_status("Error", "danger", "The local phone-pairing service did not start.")

    def _regenerate_phone_pairing(self) -> None:
        if self.session is not None:
            self.stop(silent=True)
        else:
            self.app.service.stop_phone_pairing()
            self._pairing = None
        self.pairing_timer.stop()
        self._connect_phone()

    def _poll_phone_pairing(self) -> None:
        pairing = self._pairing
        if pairing is None:
            self.pairing_timer.stop()
            return
        state = pairing.status
        if self.pairing_dialog is not None:
            self.pairing_dialog.update_status(state)
        if (pairing.frame_received and self.kind.current() == "phone"
                and self.session is None and self.start_worker is None):
            self._start_phone_capture(pairing)
        if (state in ("phone_connected", "permission_required", "camera_ready")
                and (self.session is None or self.session.source.kind == "phone")):
            title, kind, detail = PAIRING_STATES[state]
            self._set_status(title, kind, detail)
        elif (state == "streaming" and self.kind.current() == "phone"
                and self.session is None and self.start_worker is None):
            self._set_status("Streaming", "ok", "Phone video is arriving; starting local redaction…")
        elif state == "waiting_for_phone":
            if self.session is not None and self.session.source.kind == "phone":
                self.stop(silent=True)
            trusted_devices = (
                self.app.service.paired_phone_devices()
                if hasattr(self.app.service, "paired_phone_devices")
                else []
            )
            if trusted_devices:
                self.pair_button.setText("Show Phone QR")
                self.pair_button.setEnabled(True)
                if self.session is None or self.session.source.kind == "phone":
                    self._set_status(
                        "Waiting for phone",
                        "info",
                        "Connect the trusted phone and tap Start Camera when ready.",
                    )
        elif state in ("connection_lost", "pairing_expired", "disconnected", "connection_error"):
            title, kind, detail = PAIRING_STATES[state]
            trusted_devices = (
                self.app.service.paired_phone_devices()
                if hasattr(self.app.service, "paired_phone_devices")
                else []
            )
            if trusted_devices:
                if self.session is not None and self.session.source.kind == "phone":
                    self.stop(silent=True)
                self.pair_button.setText("Show Phone QR")
                self.pair_button.setEnabled(True)
                if self.session is None or self.session.source.kind == "phone":
                    self._set_status(
                        "Waiting for phone",
                        "info",
                        "The trusted phone can reconnect securely; no new QR scan is required.",
                    )
                self._refresh_paired_devices()
                return
            if self.session is not None and self.session.source.kind == "phone":
                self.stop(silent=True)
            else:
                self.app.service.stop_phone_pairing()
                self._pairing = None
                self.pairing_timer.stop()
                self.pair_button.setEnabled(True)
                self.pair_button.setText("Connect Phone with QR")
            if self.pairing_dialog is not None:
                self.pairing_dialog.update_status(state)
            if state in ("connection_lost", "connection_error"):
                self.banner.show_message(detail, "danger")
            if self.session is None or self.session.source.kind == "phone":
                self._set_status(title, kind, detail)

    def _start_phone_capture(self, pairing) -> None:
        self._set_running(True)
        mode = self.mode.current() or "auto"
        source = CameraSource(kind="phone", name="Phone camera")
        _, width, height = RESOLUTIONS[self.res_combo.currentIndex()]
        self.start_worker = FuncWorker(
            self.app.service.start_live, source, width, height, 30.0, mode,
            capture_factory=lambda _source: pairing.capture, record=False,
        )
        self.start_worker.done.connect(self._started)
        self.start_worker.error.connect(self._start_failed)
        self.start_worker.finished.connect(lambda: setattr(self, "start_worker", None))
        self.start_worker.start()

    def _end_phone_pairing(self) -> None:
        if self._pairing is None:
            return
        trusted_devices = (
            self.app.service.paired_phone_devices()
            if hasattr(self.app.service, "paired_phone_devices")
            else []
        )
        if self.session is not None and self.session.source.kind == "phone":
            self.stop(silent=True)
        elif trusted_devices:
            self.app.service.disconnect_phone_devices()
        if not trusted_devices:
            self.app.service.stop_phone_pairing()
            self._pairing = None
            self.pairing_timer.stop()
        if self.pairing_dialog is not None:
            self.pairing_dialog = None
        self.pair_button.setEnabled(True)
        self.pair_button.setText(
            "Show Phone QR" if trusted_devices else "Connect Phone with QR"
        )
        if self.session is not None and self.session.source.kind != "phone":
            self._set_status("Live", "ok", f"{self.session.source.display} · redacted frames only")
        elif trusted_devices:
            self._set_status(
                "Disconnected",
                "neutral",
                "Phone disconnected. Trusted phones can reconnect without scanning a new QR code.",
            )
        else:
            self._set_status(
                "Disconnected",
                "neutral",
                "Phone disconnected. Trusted phones can reconnect without scanning a new QR code.",
            )
        self._refresh_paired_devices()

    def _refresh_paired_devices(self) -> None:
        service = self.app.service
        if not hasattr(service, "paired_phone_devices"):
            return
        devices = service.paired_phone_devices()
        selected_id = self.paired_device_combo.currentData()
        self.paired_device_combo.blockSignals(True)
        self.paired_device_combo.clear()
        for device in devices:
            state = "Connected" if device["connected"] else "Paired"
            self.paired_device_combo.addItem(
                f"{device['name']} · {state}", device["device_id"]
            )
        if selected_id:
            index = self.paired_device_combo.findData(selected_id)
            if index >= 0:
                self.paired_device_combo.setCurrentIndex(index)
        self.paired_device_combo.blockSignals(False)
        self.revoke_device_btn.setEnabled(bool(devices))
        self.paired_device_status.setText(
            f"{len(devices)} trusted phone{'s' if len(devices) != 1 else ''}. "
            "Only one phone video stream can be active at a time."
            if devices
            else "Pair a phone to save it for future connections."
        )

    def _revoke_selected_phone(self) -> None:
        device_id = self.paired_device_combo.currentData()
        if not device_id:
            return
        if self.session is not None and self.session.source.kind == "phone":
            self.stop(silent=True)
        try:
            if not self.app.service.revoke_phone_device(device_id):
                self.banner.show_message("That phone pairing was already removed.", "info")
        except (OSError, RuntimeError, ValueError) as exc:
            self.banner.show_message(f"Could not revoke the phone pairing: {exc}", "danger")
        self._refresh_paired_devices()
        self._set_status("Disconnected", "neutral", "The selected phone pairing was revoked.")

    def on_show(self) -> None:
        self._refresh_verification()
        self._refresh_paired_devices()
        if self.session is None and self.camera_combo.count() <= 1:
            self.scan()

    def shutdown(self) -> None:
        self.stop(silent=True)
        self.pairing_timer.stop()
