"""Settings: only options that actually change backend behaviour, plus real system status."""

from __future__ import annotations

from PyQt5.QtWidgets import QCheckBox, QComboBox, QHBoxLayout, QLineEdit, QSlider, QSpinBox, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt

from .. import theme
from ..widgets import Banner, Card, Page, Pill, StatusDot, button, clear_layout, human_bytes, label


def row(title: str, widget, hint: str = "") -> QWidget:
    w = QWidget()
    lay = QHBoxLayout(w)
    lay.setContentsMargins(0, 4, 0, 4)
    col = QVBoxLayout()
    col.setSpacing(1)
    col.addWidget(label(title))
    if hint:
        col.addWidget(label(hint, "Faint", wrap=True))
    lay.addLayout(col, 1)
    lay.addWidget(widget)
    return w


class SettingsPage(Page):
    def __init__(self, app) -> None:
        super().__init__("Settings", "Processing and privacy preferences. Changes apply to the next job or live session.")
        self.app = app
        cfg = app.service.config
        self.banner = Banner()
        self.body.addWidget(self.banner)

        # ---- policy (read only)
        pol = Card("Confidence policy")
        pol.add(label("HIGH ≥ 0.80 → redacted automatically   ·   MEDIUM 0.50–0.80 → human review   ·   LOW < 0.50 → not redacted on its own "
                      "(counted and reported as an uncertainty warning).", "Muted", wrap=True))
        pol.add(label("These thresholds are fixed project policy. Per-class changes only come from validated, approved rules (Correction Log).", "Faint", wrap=True))
        self.body.addWidget(pol)

        # ---- redaction & privacy
        priv = Card("Redaction and privacy")
        self.method = QComboBox()
        self.method.addItems(["Blur", "Pixelate", "Solid Mask"])
        priv.add(row("Default redaction style", self.method))
        self.face_mode = QComboBox()
        self.face_mode.addItem("Auto blur: everyone, you keep chosen faces visible", "auto")
        self.face_mode.addItem("Blur selected faces only", "selected")
        self.face_mode.addItem("Blur every face, no exceptions", "every")
        priv.add(row("Default face mode", self.face_mode, "Used when you open a video or the live camera. You can change it on each screen."))
        self.screen_policy = QComboBox()
        self.screen_policy.addItem("Redact the whole screen (conservative)", "redact_all")
        self.screen_policy.addItem("Redact only sensitive text on screens (needs OCR)", "ocr_only")
        priv.add(row("Screens", self.screen_policy, "Text-only mode falls back to redacting the whole screen if OCR is unavailable."))
        self.hold = QCheckBox()
        priv.add(row("Keep redacting briefly if a detection is missed", self.hold, "A redacted object stays covered for up to 5 missed frames (track lifetime) so one missed detection does not expose it."))
        self.smooth = QSlider(Qt.Horizontal)
        self.smooth.setRange(1, 15)
        self.smooth.setFixedWidth(180)
        self.smooth_val = label("")
        self.smooth.valueChanged.connect(lambda v: self.smooth_val.setText(f"{v} frames"))
        sw = QWidget()
        sl = QHBoxLayout(sw)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.addWidget(self.smooth)
        sl.addWidget(self.smooth_val)
        priv.add(row("Temporal smoothing (EMA window)", sw, "Steadies boxes between frames. Tracks match at IoU ≥ 0.50."))
        self.body.addWidget(priv)

        # ---- detection classes
        cls = Card("Detection classes")
        self.class_boxes = {}
        for key, text in (("face", "Faces"), ("id_card", "ID cards"), ("credit_card", "Payment cards"), ("screen", "Screens"),
                          ("license_plate", "License plates")):
            cb = QCheckBox()
            self.class_boxes[key] = cb
            cls.add(row(text, cb))
        cls.add(label("Sensitive text is controlled by the OCR switch below. A class that is off is neither detected nor redacted.", "Faint", wrap=True))
        self.body.addWidget(cls)
        self.sections = {"classes": cls, "redaction": priv, "thresholds": pol}

        # ---- text / OCR
        ocr = Card("Sensitive text (OCR)")
        self.ocr_enabled = QCheckBox()
        ocr.add(row("Read and redact sensitive text in videos", self.ocr_enabled))
        self.ocr_model = QLineEdit()
        self.ocr_model.setFixedWidth(260)
        ocr.add(row("TrOCR model", self.ocr_model, "Downloaded once from Hugging Face on first use. Requires the 'transformers' package."))
        self.ocr_status = label("", "Muted", wrap=True)
        ocr.add(self.ocr_status)
        self.body.addWidget(ocr)

        # ---- live
        live = Card("Live camera")
        self.interval = QSpinBox()
        self.interval.setRange(1, 10)
        live.add(row("Run detection every N frames", self.interval, "Higher = faster but boxes are held between detections. 1 = every frame."))
        self.live_ocr = QCheckBox()
        live.add(row("Read sensitive text in live video", self.live_ocr, "Slow on CPU; off by default."))
        self.body.addWidget(live)
        self.sections["camera"] = live

        row_btn = QHBoxLayout()
        self.save_btn = button("Save settings", "Primary", callback=self.save)
        row_btn.addWidget(self.save_btn)
        row_btn.addStretch()
        w = QWidget()
        w.setLayout(row_btn)
        self.body.addWidget(w)

        # ---- storage
        self.storage_card = Card("Storage")
        self.storage_box = QVBoxLayout()
        self.storage_card.add(self.storage_box)
        self.body.addWidget(self.storage_card)
        self.sections["storage"] = self.storage_card

        # ---- system status
        self.sys_card = Card("System status")
        self.sys_box = QVBoxLayout()
        self.sys_box.setSpacing(8)
        self.sys_card.add(self.sys_box)
        self.body.addWidget(self.sys_card)
        self.sections["engines"] = self.sys_card

        # ---- capabilities
        cap = Card("Capabilities and known limits")
        for text, kind, state in (
            ("Signature detection", "neutral", "Unavailable — not a model class and not in project scope; handwriting is not detected."),
            ("Moving / blurred faces", "warn", "Detected by YOLO; very small or heavily blurred faces can still be missed. Evaluate on your footage."),
            ("Recognising the same person", "warn", "HOG baseline: someone who leaves and returns may be listed as a new person. In Auto mode an unrecognised face is blurred (the safe side)."),
            ("Camera support", "warn", "USB/built-in cameras and phone or network cameras that offer RTSP or HTTP/MJPEG streams. Not every camera is compatible."),
            ("Text outside document/screen/card regions", "neutral", "Not read — OCR runs only inside regions YOLO finds."),
            ("Model retraining from corrections", "neutral", "Not performed. Corrections only inform validated rule proposals."),
        ):
            r = QHBoxLayout()
            r.addWidget(Pill("limit" if kind == "warn" else "n/a", kind))
            r.addWidget(label(f"{text}: {state}", "Muted", wrap=True), 1)
            holder = QWidget()
            holder.setLayout(r)
            cap.add(holder)
        self.body.addWidget(cap)
        self.sections["privacy"] = cap
        self.body.addStretch()
        self._load()

    # --------------------------------------------------------------- load/save
    def _load(self) -> None:
        c = self.app.service.config
        self.method.setCurrentText(c.redaction_method)
        self.face_mode.setCurrentIndex(max(0, self.face_mode.findData(getattr(self.app, "default_face_mode", "auto"))))
        for key, cb in self.class_boxes.items():
            cb.setChecked(key in c.enabled_classes)
        self.screen_policy.setCurrentIndex(max(0, self.screen_policy.findData(c.screen_policy)))
        self.hold.setChecked(c.hold_missed_detections)
        self.smooth.setValue(c.temporal_smoothing)
        self.smooth_val.setText(f"{c.temporal_smoothing} frames")
        self.ocr_enabled.setChecked(c.ocr_enabled)
        self.ocr_model.setText(c.ocr_model)
        self.interval.setValue(c.detection_interval)
        self.live_ocr.setChecked(c.live_ocr_enabled)

    def save(self) -> None:
        self.app.service.update_settings(
            redaction_method=self.method.currentText(), enabled_classes={k for k, cb in self.class_boxes.items() if cb.isChecked()},
            screen_policy=self.screen_policy.currentData(), hold_missed_detections=self.hold.isChecked(),
            temporal_smoothing=self.smooth.value(), ocr_enabled=self.ocr_enabled.isChecked(),
            ocr_model=self.ocr_model.text().strip() or "microsoft/trocr-base-stage1",
            detection_interval=self.interval.value(), live_ocr_enabled=self.live_ocr.isChecked(),
        )
        self.app.default_face_mode = self.face_mode.currentData()
        self.app.settings.setValue("default_face_mode", self.app.default_face_mode)
        self.app.persist_settings()
        self.app.settings_changed()
        self.banner.show_message("Settings saved.", "ok")
        self._status()

    def focus_section(self, key: str) -> None:
        card = self.sections.get(key)
        if card is not None:
            self.scroll.ensureWidgetVisible(card, 0, 40)

    # ------------------------------------------------------------------ status
    def on_show(self) -> None:
        self._load()
        self._status()

    def _clear_d3(self) -> None:
        st = self.app.service.storage
        freed = 0
        for job in st.list_jobs(1000):
            if job.pending_review == 0 and st.release_job_frames(job.job_id):
                freed += 1
        self.banner.show_message(f"Cleared temporary frames for {freed} finished job(s). Frames still awaiting review were kept.", "ok")
        self._status()

    def _status(self) -> None:
        svc = self.app.service
        diag = svc.diagnostics()
        ocr = diag["ocr"]
        self.ocr_status.setText(("OCR ready." if ocr["available"] else "OCR unavailable: " + ocr["reason"]))
        self.ocr_status.setStyleSheet(f"color: {theme.OK if ocr['available'] else theme.WARN};")

        clear_layout(self.storage_box)
        for k, v in diag["storage"].items():
            self.storage_box.addWidget(label(f"{k}:  {v}", "Mono"))
        usage = svc.storage.d3_usage_bytes()
        r = QHBoxLayout()
        r.addWidget(label(f"Temporary frames (D3) in use: {human_bytes(usage)}. Deleted automatically after review or {svc.storage.d3_retention_days:.0f} days.", "Muted", wrap=True), 1)
        r.addWidget(button("Clear finished", "", callback=self._clear_d3))
        w = QWidget()
        w.setLayout(r)
        self.storage_box.addWidget(w)

        clear_layout(self.sys_box)
        det, fr, cv = diag["detector"], diag["face_recognition"], diag["opencv"]
        lines = [
            ("ok" if det["loaded"] else "danger", "YOLO detector", f"classes: {', '.join(det['classes'].values())} · device: {det['device']}"),
            ("ok" if ocr["available"] else "warn", "OCR", ocr["model"] + ("" if ocr["available"] else f" — {ocr['reason']}")),
            ("ok" if fr["loaded"] else "danger", "Face detection", ", ".join(fr["backends"]) or "none"),
            ("ok" if cv["has_cascade"] else "warn", "OpenCV", f"{cv['version']}  ·  Haar cascade {'available' if cv['has_cascade'] else 'NOT available (YOLO face boxes are used instead)'}"),
            ("ok", "Rules", f"active version v{diag['rules']['active_version']}"),
        ]
        for kind, name, detail in lines:
            rr = QHBoxLayout()
            rr.addWidget(StatusDot(kind))
            rr.addWidget(label(name))
            rr.addWidget(label(detail, "Faint", wrap=True), 1)
            holder = QWidget()
            holder.setLayout(rr)
            self.sys_box.addWidget(holder)
        for w_ in det["warnings"] + fr["notes"]:
            b = Banner()
            b.show_message(w_, "warn")
            self.sys_box.addWidget(b)
