"""Dashboard: hero + upload action, quick actions, real status, recent jobs, pending review."""

from __future__ import annotations

import time

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFileDialog, QFrame, QHBoxLayout, QVBoxLayout, QWidget

from .. import theme
from ..widgets import Card, EmptyState, Page, Pill, StatusDot, button, clear_layout, hline, label


class DashboardPage(Page):
    def __init__(self, app) -> None:
        super().__init__("", "")
        self.app = app
        # Dashboard replaces the standard header with a centred hero (like the reference).
        clear_layout(self.header_row)
        self.body.removeItem(self.header_row)
        self.body.setSpacing(22)

        hero = QVBoxLayout()
        hero.setSpacing(8)
        hero.setContentsMargins(0, 26, 0, 4)
        title = label("How do you want to protect privacy?", "HeroTitle")
        title.setAlignment(Qt.AlignCenter)
        sub = label("Detect and redact faces, ID and payment cards, screens, plates and sensitive text in recorded "
                    "video or a live camera. Everything runs locally.", "HeroSub", wrap=True)
        sub.setAlignment(Qt.AlignCenter)
        sub.setFixedWidth(620)
        sub.setMinimumHeight(46)
        hero.addWidget(title)
        hero.addWidget(sub, alignment=Qt.AlignHCenter)
        self.body.addLayout(hero)

        modes = QHBoxLayout()
        modes.setSpacing(20)
        modes.addWidget(self._mode_card(
            "Offline · Recorded video", "Pick a video file, choose who to blur, preview the result, then process it into a redacted copy.",
            self._offline_body(), "Browse files", "folder", self._browse), 1)
        modes.addWidget(self._mode_card(
            "Realtime · Live camera", "Use a webcam, a phone or a network camera. Only redacted frames are shown or recorded.",
            self._realtime_body(), "Choose camera source", "camera", lambda: app.navigate("Live Camera")), 1)
        mode_host = QWidget()
        mode_host.setLayout(modes)
        mode_host.setMaximumWidth(960)
        self.body.addWidget(mode_host, alignment=Qt.AlignHCenter)
        self.setAcceptDrops(True)
        self.body.addSpacing(8)

        # lower panel: recent jobs | status
        panel = Card(padding=22)
        cols = QHBoxLayout()
        cols.setSpacing(24)
        left = QVBoxLayout()
        lh = QHBoxLayout()
        lh.addWidget(label("Recent jobs", "SectionTitle"))
        lh.addStretch()
        lh.addWidget(button("See all", "Ghost", callback=lambda: app.navigate("History & Outputs")))
        left.addLayout(lh)
        self.jobs_box = QVBoxLayout()
        self.jobs_box.setSpacing(0)
        left.addLayout(self.jobs_box)
        left.addStretch()
        cols.addLayout(left, 3)
        div = QFrame()
        div.setObjectName("VDivider")
        cols.addWidget(div)
        right = QVBoxLayout()
        right.setSpacing(10)
        right.addWidget(label("Pipeline status", "SectionTitle"))
        self.status_box = QVBoxLayout()
        self.status_box.setSpacing(9)
        right.addLayout(self.status_box)
        right.addSpacing(6)
        right.addWidget(hline())
        right.addSpacing(2)
        rh = QHBoxLayout()
        rh.addWidget(label("Pending review", "SectionTitle"))
        rh.addStretch()
        rh.addWidget(button("Open", "Ghost", callback=lambda: app.navigate("Review Queue")))
        right.addLayout(rh)
        self.review_label = label("", "Muted", wrap=True)
        right.addWidget(self.review_label)
        right.addStretch()
        cols.addLayout(right, 2)
        panel.add(cols)
        self.body.addWidget(panel)
        self.body.addStretch()

    # ------------------------------------------------------------------
    def _mode_card(self, title: str, desc: str, body: QWidget, button_text: str, icon: str, callback) -> Card:
        card = Card(padding=24)
        card.add(label(title, "SectionTitle"))
        card.add(label(desc, "Muted", wrap=True))
        card.add(body)
        card.add(button(button_text, "Primary", icon, callback=callback))
        return card

    def _offline_body(self) -> QWidget:
        zone = QFrame()
        zone.setStyleSheet(f"QFrame {{ border: 2px dashed {theme.ACCENT_BORDER}; border-radius: 12px; background: #FAF9FF; }}")
        lay = QVBoxLayout(zone)
        lay.setContentsMargins(18, 26, 18, 26)
        text = label("Drop a video here, or browse\nMP4, MOV, AVI, MKV, M4V, WEBM, WMV", "Muted")
        text.setAlignment(Qt.AlignCenter)
        text.setStyleSheet("border: none; background: transparent;")
        lay.addWidget(text)
        return zone

    def _realtime_body(self) -> QWidget:
        host = QWidget()
        row = QHBoxLayout(host)
        row.setContentsMargins(0, 8, 0, 8)
        row.setSpacing(8)
        for text, kind in (("Built-in / USB", "ok"), ("Phone", "accent"), ("IP camera", "warn")):
            row.addWidget(Pill(text, kind))
        row.addStretch()
        return host

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        urls = [u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()]
        if urls:
            self.app.open_video(urls[0])

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select a video", "",
                                              "Video files (*.mp4 *.mov *.avi *.mkv *.m4v *.webm *.wmv);;All files (*)")
        if path:
            self.app.open_video(path)

    def on_show(self) -> None:
        svc = self.app.service
        # ---- recent jobs (real manifests)
        clear_layout(self.jobs_box)
        jobs = svc.get_history(5)
        if not jobs:
            self.jobs_box.addWidget(EmptyState("video", "No jobs yet", "Process a video and it will appear here with its real results."))
        for job in jobs:
            row = QFrame()
            row.setObjectName("Row")
            rl = QHBoxLayout(row)
            rl.setContentsMargins(0, 9, 0, 9)
            col = QVBoxLayout()
            col.setSpacing(1)
            name = label(job.source_name)
            name.setStyleSheet("font-weight: 500;")
            col.addWidget(name)
            when = time.strftime("%b %d, %H:%M", time.localtime(job.created_at))
            col.addWidget(label(f"{when}  ·  {job.frames_processed} frames  ·  {job.redactions} redactions", "Faint"))
            rl.addLayout(col, 1)
            rl.addWidget(Pill(job.status.capitalize(), theme.STATUS_KIND.get(job.status, "neutral")))
            self.jobs_box.addWidget(row)

        # ---- real status from diagnostics
        diag = svc.diagnostics()
        clear_layout(self.status_box)
        det, ocr, face, red = diag["detector"], diag["ocr"], diag["face_recognition"], diag["redaction"]
        rows = [
            ("YOLO detector", "ok" if det["loaded"] else "danger", f"{len(det['classes'])} classes · {det['device']}"),
            ("OCR (TrOCR)", "ok" if ocr["available"] else "warn", ("ready" if ocr["loaded"] else "loads on first use") if ocr["available"] else "unavailable"),
            ("Face detection", "ok" if face["loaded"] else "danger", ", ".join(b.replace("FaceDetector", "").replace("Yolo", "YOLO") or "Haar" for b in face["backends"]) or "none"),
            ("Redaction", "ok", red["active"]),
        ]
        for name, kind, detail in rows:
            r = QHBoxLayout()
            r.setSpacing(8)
            r.addWidget(StatusDot(kind))
            r.addWidget(label(name))
            r.addStretch()
            r.addWidget(label(detail, "Faint"))
            w = QWidget()
            w.setLayout(r)
            self.status_box.addWidget(w)

        ov = svc.overview()
        if ov["pending_review"]:
            self.review_label.setText(f"{ov['pending_review']} detection(s) between 50% and 80% confidence are waiting for a decision.")
        else:
            self.review_label.setText("Nothing waiting. Uncertain detections from your jobs will appear here.")
