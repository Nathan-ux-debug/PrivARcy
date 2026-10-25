"""Video Processing: the editor.

Left: the frame with a live preview of exactly what will be redacted. Right: Auto Blur (class chips + face choices),
Manual Blur (draw regions) and Magic Blur (click an object). Every option change re-renders the preview immediately from
cached detections (see ``privarcy.pipeline.preview``) - no new inference, and the same rules as the real processing pass.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import cv2
from PyQt5.QtCore import QTimer, Qt, QUrl
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import (
    QDoubleSpinBox, QFileDialog, QFrame, QGridLayout, QHBoxLayout, QProgressBar, QPushButton, QSlider, QStackedWidget,
    QVBoxLayout, QWidget,
)

from privarcy.faces import MODE_HELP, FaceMode
from privarcy.pipeline.plan import EditPlan, ManualRegion
from ..canvas import FrameCanvas
from ..widgets import (
    Banner, Card, FaceTile, Page, Pill, Segmented, StatTile, bgr_to_pixmap, button, chip_button, clear_layout, human_bytes,
    human_duration, label,
)
from ..workers import FuncWorker, ProcessingWorker, ScanWorker

SENSITIVE = [("id_card", "ID card"), ("credit_card", "Payment card"), ("screen", "Screen"), ("license_plate", "License plate")]
FACE_MODES = [("auto", "Auto"), ("selected", "Selected only"), ("every", "Every face")]


def read_frame(path: str, index: int):
    cap = cv2.VideoCapture(path)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, index))
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()


class VideoPage(Page):
    def __init__(self, app) -> None:
        super().__init__("Edit & preview", "Choose what to hide. The preview updates as soon as you change an option, "
                                           "and the same settings are used when you process.")
        self.app = app
        self.video_path: Optional[str] = None
        self.info: Optional[dict] = None
        self.scan = None
        self.policy = None
        self.preview = None
        self.analysis = None
        self.plan = EditPlan()
        self.worker: Optional[ProcessingWorker] = None
        self.scan_worker: Optional[ScanWorker] = None
        self._analyze_worker: Optional[FuncWorker] = None
        self._cache: Dict[int, object] = {}
        self._tiles: Dict[str, FaceTile] = {}
        self._loading = False
        self._playing = False
        self._scan_ids: set = set()
        self._autoplay_pending = False
        self.frame_index = 0
        self.last_result = None
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(220)
        self._debounce.timeout.connect(self._load_frame)

        self.choose_btn = button("Choose video…", "", "folder", callback=self.choose)
        self.header_actions.addWidget(self.choose_btn)
        self.banner = Banner()
        self.body.addWidget(self.banner)

        cols = QHBoxLayout()
        cols.setSpacing(18)
        cols.addLayout(self._build_left(), 6)
        cols.addLayout(self._build_right(), 4)
        self.body.addLayout(cols)
        self.body.addWidget(self._build_run_card())
        self.result_card = Card("Result")
        self.result_body = QVBoxLayout()
        self.result_card.add(self.result_body)
        self.result_card.hide()
        self.body.addWidget(self.result_card)
        self.body.addStretch()
        self._load_options()
        self._set_controls_enabled(False)

    # ------------------------------------------------------------------ layout
    def _build_left(self) -> QVBoxLayout:
        left = QVBoxLayout()
        left.setSpacing(12)
        card = Card("")
        self.canvas = FrameCanvas("No video selected. Choose a video to begin.")
        self.canvas.setMinimumHeight(340)
        self.canvas.clicked.connect(self._canvas_clicked)
        self.canvas.rect_drawn.connect(self._region_drawn)
        card.add(self.canvas)
        row = QHBoxLayout()
        self.time_label = label("0:00 / 0:00", "Faint")
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.valueChanged.connect(self._slider_changed)
        self.play_btn = QPushButton("▶ Play")
        self.play_btn.clicked.connect(self._toggle_play)
        self.slider.sliderPressed.connect(self._stop_play)   # dragging the slider pauses
        row.addWidget(self.play_btn)
        row.addWidget(self.time_label)
        row.addWidget(self.slider, 1)
        card.add(row)
        self.file_label = label("", "Muted", wrap=True)
        card.add(self.file_label)
        self.meta_label = label("", "Faint", wrap=True)
        card.add(self.meta_label)
        left.addWidget(card)
        self.counts_label = label("", "Muted", wrap=True)
        left.addWidget(self.counts_label)
        left.addStretch()
        return left

    def _build_right(self) -> QVBoxLayout:
        right = QVBoxLayout()
        right.setSpacing(12)
        self.tabs = Segmented([("auto", "Auto Blur"), ("manual", "Manual Blur"), ("magic", "Magic Blur")], "auto")
        self.tabs.changed.connect(self._tab_changed)
        right.addWidget(self.tabs)
        self.stack = QStackedWidget()
        self.stack.addWidget(self._build_auto())
        self.stack.addWidget(self._build_manual())
        self.stack.addWidget(self._build_magic())
        right.addWidget(self.stack)

        style = Card("Blur style")
        self.method = Segmented([("Blur", "Blur"), ("Pixelate", "Pixelate"), ("Solid Mask", "Solid mask")], "Blur")
        self.method.changed.connect(self._options_changed)
        style.add(self.method)
        style.add(label("Strength", "Muted"))
        self.strength = QSlider(Qt.Horizontal)
        self.strength.setRange(0, 100)
        self.strength.setValue(50)
        self.strength.valueChanged.connect(self._options_changed)
        style.add(self.strength)
        right.addWidget(style)
        right.addStretch()
        return right

    def _build_auto(self) -> QWidget:
        host = QWidget()
        lay = QVBoxLayout(host)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)
        card = Card("Auto Blur")
        card.add(label("AI finds and blurs these for you. Pick one or several.", "Muted", wrap=True))
        card.add(label("People", "Faint"))
        self.chips: Dict[str, object] = {}
        row = QHBoxLayout()
        self.chips["face"] = chip_button("Face")
        row.addWidget(self.chips["face"])
        row.addStretch()
        card.add(row)
        card.add(label("Sensitive items", "Faint"))
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)
        for i, (key, text) in enumerate(SENSITIVE):
            self.chips[key] = chip_button(text)
            grid.addWidget(self.chips[key], i // 2, i % 2)
        self.ocr_chip = chip_button("Sensitive text · OCR")
        grid.addWidget(self.ocr_chip, 2, 0, 1, 2)
        card.add(grid)
        for chip in (*self.chips.values(), self.ocr_chip):
            chip.toggled.connect(self._options_changed)
        self.ocr_note = label("", "Faint", wrap=True)
        card.add(self.ocr_note)
        lay.addWidget(card)

        faces = Card("Faces")
        self.mode = Segmented(FACE_MODES, "auto")
        self.mode.changed.connect(self._mode_changed)
        faces.add(self.mode)
        self.mode_help = label(MODE_HELP[FaceMode.AUTO], "Faint", wrap=True)
        faces.add(self.mode_help)
        self.scan_row = QHBoxLayout()
        self.scan_label = label("", "Faint", wrap=True)
        self.scan_bar = QProgressBar()
        self.scan_bar.setRange(0, 100)
        self.scan_bar.hide()
        faces.add(self.scan_label)
        faces.add(self.scan_bar)
        self.tiles_grid = QGridLayout()
        self.tiles_grid.setSpacing(8)
        faces.add(self.tiles_grid)
        self.rescan_btn = button("Re-scan faces", "", "refresh", callback=self._start_scan)
        faces.add(self.rescan_btn)
        lay.addWidget(faces)
        return host

    def _build_manual(self) -> QWidget:
        host = QWidget()
        lay = QVBoxLayout(host)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)
        card = Card("Manual Blur")
        card.add(label("Draw a region over anything you want to hide. It stays in place for the time range you set.", "Muted", wrap=True))
        self.shape = Segmented([("rect", "Rectangle"), ("ellipse", "Ellipse")], "rect")
        card.add(self.shape)
        self.draw_btn = button("Draw new region", "Primary", "region", callback=self._toggle_draw)
        self.draw_btn.setCheckable(True)
        card.add(self.draw_btn)
        lay.addWidget(card)
        self.regions_card = Card("Regions")
        self.regions_body = QVBoxLayout()
        self.regions_card.add(self.regions_body)
        lay.addWidget(self.regions_card)
        lay.addStretch(1)
        return host

    def _build_magic(self) -> QWidget:
        host = QWidget()
        lay = QVBoxLayout(host)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)
        card = Card("Magic Blur")
        card.add(label("Click an object in the video to blur its shape. It is followed through the video by what it looks like "
                       "and where it is.", "Muted", wrap=True))
        self.magic_btn = button("Click objects in the video", "Primary", "wand", callback=self._toggle_magic)
        self.magic_btn.setCheckable(True)
        card.add(self.magic_btn)
        card.add(label("Works on things the detector knows (faces, cards, screens, plates). For anything else, use Manual Blur.", "Faint", wrap=True))
        lay.addWidget(card)
        self.magic_card = Card("Selected objects")
        self.magic_body = QVBoxLayout()
        self.magic_card.add(self.magic_body)
        lay.addWidget(self.magic_card)
        lay.addStretch(1)
        return host

    def _build_run_card(self) -> QFrame:
        box = QFrame()
        box.setObjectName("Summary")
        lay = QVBoxLayout(box)
        lay.setContentsMargins(18, 14, 18, 14)
        lay.setSpacing(10)
        top = QHBoxLayout()
        text = QVBoxLayout()
        text.setSpacing(2)
        text.addWidget(label("SUMMARY", "Faint"))
        self.summary_label = label("Choose a video to begin.", "SectionTitle", wrap=True)
        text.addWidget(self.summary_label)
        top.addLayout(text, 1)
        self.status_pill = Pill("Idle", "neutral")
        top.addWidget(self.status_pill)
        self.start_btn = button("Start processing", "Primary", "play", callback=self.start)
        self.cancel_btn = button("Cancel", "Danger", "stop", "#D92D20", callback=self.cancel)
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(False)
        top.addWidget(self.start_btn)
        top.addWidget(self.cancel_btn)
        lay.addLayout(top)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        lay.addWidget(self.progress)
        grid = QGridLayout()
        grid.setSpacing(10)
        self.tiles = {k: StatTile(c) for k, c in (("frames", "Frames"), ("detections", "Detections"), ("redactions", "Redactions"),
                                                  ("flagged", "For review"), ("fps", "Speed (fps)"), ("eta", "Time left"))}
        for i, t in enumerate(self.tiles.values()):
            grid.addWidget(t, i // 3, i % 3)
        lay.addLayout(grid)
        self.status_text = label("", "Faint")
        lay.addWidget(self.status_text)
        lay.addWidget(label("Processed locally. Nothing is uploaded.", "Faint"))
        return box

    # ------------------------------------------------------------------ options <-> config
    def _load_options(self) -> None:
        cfg = self.app.service.config
        self._loading = True
        for key, chip in self.chips.items():
            chip.setChecked(key in cfg.enabled_classes)
        ocr = self.app.service.ocr_provider.status()
        self.ocr_chip.setChecked(bool(cfg.ocr_enabled and ocr["available"]))
        self.ocr_chip.setEnabled(bool(ocr["available"]))
        self.ocr_note.setText("OCR reads text inside document, screen and card regions; only sensitive-looking text is blurred."
                              if ocr["available"] else "OCR is unavailable on this device, so sensitive text cannot be detected yet. " + ocr["reason"])
        self.ocr_note.setStyleSheet("" if ocr["available"] else "color: #C77700;")
        self.method.set_current(cfg.redaction_method, emit=False)
        self.strength.setValue(int(round((cfg.redaction_strength if cfg.redaction_strength is not None else 0.5) * 100)))
        mode = getattr(self.app, "default_face_mode", "auto")
        self.mode.set_current(mode if mode in dict(FACE_MODES) else "auto", emit=False)
        self._loading = False

    def _enabled_classes(self) -> set:
        return {k for k, c in self.chips.items() if c.isChecked()}

    def _options_changed(self, *_args) -> None:
        if self._loading:
            return
        service = self.app.service
        service.update_settings(redaction_method=self.method.current() or "Blur", enabled_classes=self._enabled_classes(),
                                ocr_enabled=self.ocr_chip.isChecked(), redaction_strength=self.strength.value() / 100.0)
        self.app.persist_settings()
        self.app.settings_changed()
        self._refresh_faces()
        self._render()

    def _mode_changed(self, key: str) -> None:
        if self.policy is not None:
            self.policy.set_mode(key)
        self.app.settings.setValue("default_face_mode", key)
        self.app.default_face_mode = key
        self.mode_help.setText(MODE_HELP[FaceMode.parse(key)])
        self._refresh_faces()
        self._render()

    # ------------------------------------------------------------------ source
    def choose(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select a video", "", "Video files (*.mp4 *.mov *.avi *.mkv *.m4v *.webm *.wmv);;All files (*)")
        if path:
            self.set_video(path)

    def set_video(self, path: str) -> None:
        if self.worker is not None:
            self.banner.show_message("A job is running. Wait for it to finish or cancel it first.", "warn")
            return
        self._stop_play()
        self.banner.hide()
        self.result_card.hide()
        info = self.app.service.validate_video(path)
        if not info.get("ok"):
            self.video_path, self.info = None, None
            self.canvas.set_frame(None)
            self.canvas.placeholder = "Cannot use this file"
            self.file_label.setText(Path(path).name)
            self.banner.show_message(f"This file can't be processed: {info.get('error')}\nTry re-exporting it as MP4 (H.264), then select it again.", "danger")
            self._set_controls_enabled(False)
            self.start_btn.setEnabled(False)
            self._set_status("Idle", "neutral", "Choose a valid video to begin.")
            return
        self.video_path, self.info = path, info
        self.plan = EditPlan()
        self._cache.clear()
        self.analysis = None
        self.scan = None
        self._scan_ids = set()
        self.file_label.setText(Path(path).name)
        self.meta_label.setText(f"{info['width']}×{info['height']}  ·  {info['fps']:.2f} fps  ·  {info['total_frames']} frames  ·  "
                                f"{human_duration(info['duration'])}  ·  {human_bytes(info['size_bytes'])}")
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(0, int(info["total_frames"]) - 1))
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self._autoplay_pending = True
        self.frame_index = 0
        self._update_time()
        first = read_frame(path, 0)
        self.canvas.set_frame(first)
        self._set_controls_enabled(True)
        for t in self.tiles.values():
            t.set_value("—")
        self.progress.setValue(0)
        self._refresh_regions()
        self._refresh_magic()
        self._set_status("Ready", "accent", "Scanning faces…")
        self._start_scan()

    # ------------------------------------------------------------------ face scan
    def _start_scan(self) -> None:
        if not self.video_path or self.scan_worker is not None:
            return
        self.policy = None
        self.preview = None
        self.analysis = None
        self._clear_tiles()
        self.scan_label.setText("Scanning the video for people…")
        self.scan_bar.setValue(0)
        self.scan_bar.show()
        self.rescan_btn.setEnabled(False)
        self.canvas.set_busy("Scanning faces…")
        self.start_btn.setEnabled(False)
        self.scan_worker = ScanWorker(self.app.service, self.video_path)
        self.scan_worker.progress.connect(lambda d, t: self.scan_bar.setValue(int(d / max(t, 1) * 100)))
        self.scan_worker.done.connect(self._scan_done)
        self.scan_worker.error.connect(self._scan_failed)
        self.scan_worker.finished.connect(self._scan_finished)
        self.scan_worker.start()

    def _scan_done(self, scan) -> None:
        self.scan = scan
        self._scan_ids = {i.face_id for i in scan.identities()}
        self._build_policy(scan.book)
        n = len(scan.identities())
        self.scan_label.setText(f"{n} {'person' if n == 1 else 'people'} found. Faces that leave and come back may be listed twice; "
                                f"in Auto mode an unrecognised face is simply blurred.")

    def _scan_failed(self, message: str) -> None:
        self.scan = None
        self._build_policy(None)
        self.scan_label.setText("")
        self.banner.show_message(f"Could not list faces ({message}). Other detections still work; faces will all be blurred.", "warn")

    def _scan_finished(self) -> None:
        self.scan_worker = None
        self.scan_bar.hide()
        self.rescan_btn.setEnabled(True)
        self.canvas.set_busy("")
        self._load_frame()

    def _build_policy(self, book) -> None:
        self.policy = self.app.service.new_face_policy(self.mode.current() or "auto", book=book)
        self.preview = self.app.service.create_preview(self.policy)
        self._refresh_faces()

    # ------------------------------------------------------------------ frames + instant preview
    def _slider_changed(self, value: int) -> None:
        self.frame_index = value
        self._update_time()
        self._debounce.start()

    def _update_time(self) -> None:
        if self.info:
            fps = self.info["fps"] or 25.0
            self.time_label.setText(f"{human_duration(self.frame_index / fps)} / {human_duration(self.info['duration'])}")

    # ------------------------------------------------------------------ playback
    def _toggle_play(self) -> None:
        if self._playing:
            self._stop_play()
        else:
            self._start_play()

    def _start_play(self) -> None:
        if not self.info or self.preview is None:
            return
        if self.frame_index >= self.slider.maximum():
            self.slider.setValue(0)
            self._debounce.stop()
        self._playing = True
        self.play_btn.setText("⏸ Pause")
        self._load_frame()

    def _stop_play(self) -> None:
        self._playing = False
        self.play_btn.setText("▶ Play")

    def _after_frame_shown(self) -> None:
        if self._playing:
            QTimer.singleShot(0, self._advance)

    def _advance(self) -> None:
        if not self._playing or not self.info:
            return
        step = max(1, round((self.info["fps"] or 25.0) / 6))   # about 6 preview frames per second of video
        nxt = self.frame_index + step
        if nxt > self.slider.maximum():
            self._stop_play()
            return
        self.slider.setValue(nxt)
        self._debounce.stop()      # skip the 220 ms scrub delay
        self._load_frame()

    # ------------------------------------------------------------------ frame analysis
    def _load_frame(self) -> None:
        """Analyse the chosen frame once (slow: runs the detector) then render instantly."""
        if not self.video_path or self.preview is None or self.scan_worker is not None:
            return
        if self.frame_index in self._cache:
            self.analysis = self._cache[self.frame_index]
            self._refresh_faces()
            self._render()
            self._after_frame_shown()
            return
        frame = read_frame(self.video_path, self.frame_index)
        if frame is None:
            self._stop_play()
            return
        self.canvas.set_frame(frame)
        self.canvas.set_busy("Analysing frame…")
        index, preview = self.frame_index, self.preview
        self._analyze_worker = FuncWorker(preview.analyze, frame, index)
        self._analyze_worker.done.connect(lambda a, i=index: self._analysis_ready(i, a))
        self._analyze_worker.error.connect(self._analysis_failed)
        self._analyze_worker.start()

    def _analysis_ready(self, index: int, analysis) -> None:
        self._cache[index] = analysis
        if len(self._cache) > 12:
            self._cache.pop(next(iter(self._cache)))
        self.canvas.set_busy("")
        if index != self.frame_index:
            return
        self.analysis = analysis
        self._refresh_faces()
        self._render()
        self.start_btn.setEnabled(self.worker is None)
        self._set_status("Ready", "accent", "Ready to process.")
        if self._autoplay_pending:
            self._autoplay_pending = False
            self._start_play()
        else:
            self._after_frame_shown()

    def _analysis_failed(self, message: str) -> None:
        self._stop_play()
        self.canvas.set_busy("")
        self.banner.show_message(f"Could not analyse this frame: {message}", "danger")

    def _seconds(self) -> float:
        fps = (self.info or {}).get("fps") or 25.0
        return self.frame_index / fps

    def _render(self) -> None:
        """Instant: no model call. Draws what the output will look like for the current options."""
        if self.preview is None or self.analysis is None:
            self._update_summary()
            return
        result = self.preview.render(self.analysis, self.plan, self._seconds())
        self.canvas.set_frame(result.image, result.overlays)
        self._update_counts(result)
        self._update_summary()

    def _update_counts(self, result) -> None:
        faces = [o for o in result.overlays if o.kind == "face"]
        blurred = sum(1 for o in faces if o.state == "blurred")
        parts = [f"{blurred} face{'s' if blurred != 1 else ''} blurred"]
        kept = sum(1 for o in faces if o.state == "visible")
        if kept:
            parts.append(f"{kept} visible")
        objs = sum(1 for o in result.overlays if o.kind == "object")
        if objs:
            parts.append(f"{objs} sensitive item{'s' if objs != 1 else ''} blurred")
        review = sum(1 for o in result.overlays if o.kind == "review")
        if review:
            parts.append(f"{review} need review")
        self.counts_label.setText("This frame: " + "  ·  ".join(parts))

    # ------------------------------------------------------------------ face tiles
    def _clear_tiles(self) -> None:
        clear_layout(self.tiles_grid)
        self._tiles.clear()
        
    def _identities(self):
        if self.policy is None:
            return []
        idents = self.policy.book.identities()
        if self._scan_ids:
            idents = [i for i in idents if i.face_id in self._scan_ids]
        return idents

    def _refresh_faces(self) -> None:
        if self.policy is None:
            return
        idents = self._identities()
        if set(self._tiles) != {i.face_id for i in idents}:
            self._clear_tiles()
            for n, ident in enumerate(idents):
                thumb = bgr_to_pixmap(ident.thumbnail) if ident.thumbnail is not None else None
                tile = FaceTile(ident.face_id, ident.label, thumb)
                tile.clicked.connect(self._tile_clicked)
                self._tiles[ident.face_id] = tile
                self.tiles_grid.addWidget(tile, n // 2, n % 2)
        mode = FaceMode.parse(self.mode.current())
        face_on = self.chips["face"].isChecked()
        for fid, tile in self._tiles.items():
            blur, _ = self.policy.decide(fid)
            if not face_on:
                tile.set_state("", "Faces are turned off", False)
            elif mode == FaceMode.EVERY:
                tile.set_state("blurred", "Will be blurred", False)
            elif mode == FaceMode.AUTO:
                tile.set_state("blurred" if blur else "visible", "Will be blurred" if blur else "Kept visible")
            else:
                tile.set_state("blurred" if blur else "", "Will be blurred" if blur else "Stays visible")

    def _tile_clicked(self, face_id: str) -> None:
        if self.policy is not None:
            self.policy.toggle(face_id)
            self._refresh_faces()
            self._render()

    # ------------------------------------------------------------------ canvas interaction
    def _canvas_clicked(self, x: float, y: float) -> None:
        tab = self.tabs.current()
        if tab == "magic" and self.preview is not None and self.analysis is not None:
            target = self.preview.magic_target_at(self.analysis, x, y)
            if target is None:
                self.banner.show_message("Nothing the detector knows is at that spot. Use Manual Blur to draw over it.", "info")
                return
            self.banner.hide()
            self.plan.magic.append(target)
            self._refresh_magic()
            self._render()
        elif tab == "auto" and self.policy is not None:
            overlay = self.canvas.overlay_at(x, y)
            if overlay is not None and overlay.kind == "face" and overlay.face_id and FaceMode.parse(self.mode.current()) != FaceMode.EVERY:
                self._tile_clicked(overlay.face_id)

    def _tab_changed(self, key: str) -> None:
        self.stack.setCurrentIndex({"auto": 0, "manual": 1, "magic": 2}[key])
        self.draw_btn.setChecked(False)
        self.magic_btn.setChecked(key == "magic")
        self.canvas.set_mode("magic" if key == "magic" else "select")

    def _toggle_draw(self) -> None:
        self.magic_btn.setChecked(False)
        self.canvas.set_mode("draw" if self.draw_btn.isChecked() else "select")

    def _toggle_magic(self) -> None:
        self.draw_btn.setChecked(False)
        self.canvas.set_mode("magic" if self.magic_btn.isChecked() else "select")

    def _region_drawn(self, x1: float, y1: float, x2: float, y2: float) -> None:
        if not self.info:
            return
        w, h = self.info["width"], self.info["height"]
        self.plan.manual.append(ManualRegion(x1 / w, y1 / h, x2 / w, y2 / h, shape=self.shape.current() or "rect", start=0.0, end=None,
                                             label=f"Region {len(self.plan.manual) + 1}"))
        self.draw_btn.setChecked(False)
        self.canvas.set_mode("select")
        self._refresh_regions()
        self._render()

    # ------------------------------------------------------------------ regions + magic lists
    def _refresh_regions(self) -> None:
        clear_layout(self.regions_body)
        if not self.plan.manual:
            self.regions_body.addWidget(label("No regions yet. Press “Draw new region”, then drag on the video.", "Faint", wrap=True))
            return
        duration = float((self.info or {}).get("duration") or 0.0)
        for i, region in enumerate(self.plan.manual):
            row = QHBoxLayout()
            row.addWidget(label(f"{region.label} · {region.shape}", "SectionTitle"), 1)
            row.addWidget(label("from", "Faint"))
            start = QDoubleSpinBox(); start.setRange(0, max(duration, 0.1)); start.setDecimals(1); start.setSuffix(" s"); start.setValue(region.start)
            end = QDoubleSpinBox(); end.setRange(0, max(duration, 0.1)); end.setDecimals(1); end.setSuffix(" s")
            end.setValue(region.end if region.end is not None else duration)
            start.valueChanged.connect(lambda v, r=region: self._set_region_time(r, start=v))
            end.valueChanged.connect(lambda v, r=region: self._set_region_time(r, end=v))
            row.addWidget(start)
            row.addWidget(label("to", "Faint"))
            row.addWidget(end)
            row.addWidget(button("Delete", "Ghost", callback=lambda r=region: self._delete_region(r)))
            wrap = QWidget(); wrap.setLayout(row)
            self.regions_body.addWidget(wrap)

    def _set_region_time(self, region, start=None, end=None) -> None:
        if start is not None:
            region.start = start
        if end is not None:
            region.end = None if end >= float((self.info or {}).get("duration") or 0.0) - 0.05 else end
        self._render()

    def _delete_region(self, region) -> None:
        self.plan.manual.remove(region)
        self._refresh_regions()
        self._render()

    def _refresh_magic(self) -> None:
        clear_layout(self.magic_body)
        if not self.plan.magic:
            self.magic_body.addWidget(label("Nothing selected yet.", "Faint"))
            return
        for target in self.plan.magic:
            row = QHBoxLayout()
            row.addWidget(label(target.label or target.name, "SectionTitle"), 1)
            row.addWidget(label("followed through the video", "Faint"))
            row.addWidget(button("Remove", "Ghost", callback=lambda t=target: self._remove_magic(t)))
            wrap = QWidget(); wrap.setLayout(row)
            self.magic_body.addWidget(wrap)

    def _remove_magic(self, target) -> None:
        self.plan.magic.remove(target)
        self._refresh_magic()
        self._render()

    # ------------------------------------------------------------------ summary
    def _update_summary(self) -> None:
        if not self.video_path:
            self.summary_label.setText("Choose a video to begin.")
            return
        mode = FaceMode.parse(self.mode.current())
        parts = []
        if self.chips["face"].isChecked():
            total = len(self._identities())
            if mode == FaceMode.EVERY:
                parts.append("Every face blurred, including new ones")
            elif mode == FaceMode.AUTO:
                kept = len(self.policy.keep_visible) if self.policy else 0
                parts.append(f"Auto blur: {max(total - kept, 0)} faces blurred, {kept} kept visible, new faces blurred")
            else:
                sel = len(self.policy.selected) if self.policy else 0
                parts.append(f"Blur selected: {sel} of {total} faces")
        names = [text for key, text in SENSITIVE if self.chips[key].isChecked()]
        if self.ocr_chip.isChecked():
            names.append("sensitive text")
        if names:
            parts.append(", ".join(names))
        if self.plan.manual:
            parts.append(f"{len(self.plan.manual)} manual region{'s' if len(self.plan.manual) != 1 else ''}")
        if self.plan.magic:
            parts.append(f"{len(self.plan.magic)} magic object{'s' if len(self.plan.magic) != 1 else ''}")
        parts.append(self.method.current() or "Blur")
        self.summary_label.setText("  ·  ".join(parts) if parts else "Nothing selected to blur.")

    # --------------------------------------------------------------------- run
    def _set_controls_enabled(self, enabled: bool) -> None:
        for w in (*self.chips.values(), self.mode, self.method, self.strength, self.tabs, self.draw_btn, self.magic_btn, self.shape,
                  self.rescan_btn, self.slider, self.play_btn, self.choose_btn):
            w.setEnabled(enabled or w is self.choose_btn)
        if enabled:
            self._load_options()

    def _set_status(self, text: str, kind: str, detail: str) -> None:
        self.status_pill.set(text, kind)
        self.status_text.setText(detail)

    def start(self) -> None:
        if not self.video_path or self.worker is not None:
            return
        self._stop_play()
        self.banner.hide()
        self.result_card.hide()
        self._options_changed()
        self.worker = ProcessingWorker(self.app.service, self.video_path, plan=self.plan, face_policy=self.policy)
        self.worker.progress.connect(self._on_progress)
        self.worker.completed.connect(self._on_done)
        self.worker.failed.connect(self._on_failed)
        self.worker.finished.connect(self._on_finished)
        self._set_controls_enabled(False)
        self.cancel_btn.setEnabled(True)
        self.start_btn.setEnabled(False)
        self._set_status("Running", "info", "Loading models and processing frames…")
        self.app.set_busy(True)
        self.worker.start()

    def cancel(self) -> None:
        if self.worker:
            self.cancel_btn.setEnabled(False)
            self._set_status("Cancelling", "warn", "Finishing the current frame…")
            self.worker.cancel()

    def _on_progress(self, p: dict) -> None:
        total = max(p["total"], 1)
        self.progress.setValue(int(p["current"] / total * 1000))
        self.tiles["frames"].set_value(f"{p['current']}/{p['total']}")
        self.tiles["detections"].set_value(str(p["detections"]))
        self.tiles["redactions"].set_value(str(p["redactions"]))
        self.tiles["flagged"].set_value(str(p["flagged"]))
        self.tiles["fps"].set_value(f"{p['fps']:.1f}")
        self.tiles["eta"].set_value(human_duration(p["eta"]))
        self.status_text.setText(f"Processing frame {p['current']} of {p['total']}")

    def _on_done(self, result) -> None:
        self.last_result = result
        if result.cancelled:
            self._set_status("Cancelled", "warn", f"Stopped at frame {result.frames}. Process the same file again to resume.")
        else:
            self._set_status("Completed", "ok", f"Finished in {human_duration(result.processing_time)} ({result.avg_fps:.1f} fps average).")
            self.progress.setValue(1000)
        self._show_result(result)

    def _on_failed(self, message: str) -> None:
        self._set_status("Failed", "danger", "Processing stopped.")
        self.banner.show_message(f"Processing failed: {message}\nNo partial file was kept. Check the video plays in another player, "
                                 "free some disk space, or try a different file.", "danger")

    def _on_finished(self) -> None:
        self.worker = None
        self._set_controls_enabled(self.video_path is not None)
        self.start_btn.setEnabled(self.video_path is not None and self.preview is not None)
        self.cancel_btn.setEnabled(False)
        self.app.set_busy(False)
        self.app.refresh_badges()

    def _show_result(self, r) -> None:
        clear_layout(self.result_body)
        self.result_card.show()
        pending = sum(1 for d in r.detections if d.get("status") == "pending")
        self.result_body.addWidget(label(f"{r.frames} frames written  ·  {r.redactions} redaction operations  ·  {pending} item(s) need review", "", wrap=True))
        summary = r.summary
        extra = []
        if summary.get("held_redactions"):
            extra.append(f"{summary['held_redactions']} redactions came from held tracks (object missed for ≤5 frames)")
        if summary.get("text_redactions"):
            extra.append(f"{summary['text_redactions']} sensitive-text redactions")
        if extra:
            self.result_body.addWidget(label("  ·  ".join(extra), "Faint", wrap=True))
        for w in r.warnings:
            b = Banner()
            b.show_message(w, "warn")
            self.result_body.addWidget(b)
        grid = QGridLayout()
        grid.setContentsMargins(0, 4, 0, 0)
        grid.setHorizontalSpacing(8)
        grid.addWidget(button("Open redacted video", "Primary", callback=lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(r.output_path))), 0, 0)
        grid.addWidget(button("Show in folder", "", "folder", callback=lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(r.output_path).parent)))), 0, 1)
        if pending:
            grid.addWidget(button(f"Review {pending} item(s)", "", "review", callback=lambda: self.app.navigate("Review Queue")), 1, 0)
        grid.setColumnStretch(2, 1)
        w = QWidget()
        w.setLayout(grid)
        self.result_body.addWidget(w)

    def on_show(self) -> None:
        if self.worker is None and self.video_path:
            self._load_options()
            self._render()

    def shutdown(self) -> None:
        self._stop_play()
        for w in (self.worker, self.scan_worker):
            if w is not None:
                w.cancel()
                w.wait(5000)