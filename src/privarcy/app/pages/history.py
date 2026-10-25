"""History & Outputs: every job from D1 manifests, with open/delete actions."""

from __future__ import annotations

import time
from pathlib import Path

from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import QFrame, QHBoxLayout, QMessageBox, QVBoxLayout

from .. import theme
from ..widgets import Banner, Card, EmptyState, Page, Pill, button, clear_layout, human_duration, label


class HistoryPage(Page):
    def __init__(self, app) -> None:
        super().__init__("History & Outputs", "Completed, cancelled and failed jobs, with access to the redacted videos.")
        self.app = app
        self.banner = Banner()
        self.body.addWidget(self.banner)
        self.card = Card(padding=8)
        self.box = QVBoxLayout()
        self.box.setSpacing(0)
        self.card.add(self.box)
        self.body.addWidget(self.card)
        self.body.addStretch()

    def on_show(self) -> None:
        clear_layout(self.box)
        jobs = self.app.service.get_history(100)
        if not jobs:
            self.box.addWidget(EmptyState("history", "No jobs yet", "Processed videos and their results will be listed here."))
            return
        for job in jobs:
            row = QFrame()
            row.setObjectName("Row")
            rl = QHBoxLayout(row)
            rl.setContentsMargins(12, 12, 12, 12)
            rl.setSpacing(12)
            col = QVBoxLayout()
            col.setSpacing(2)
            nm = label(job.source_name)
            nm.setStyleSheet("font-weight: 500;")
            col.addWidget(nm)
            when = time.strftime("%b %d %Y, %H:%M", time.localtime(job.created_at))
            meta = f"{when}  ·  {job.width}×{job.height}  ·  {job.frames_processed}/{job.total_frames} frames"
            if job.processing_time:
                meta += f"  ·  {human_duration(job.processing_time)} ({job.avg_fps:.1f} fps)"
            col.addWidget(label(meta, "Faint", wrap=True))
            counts = f"{job.detections} detections · {job.redactions} redactions"
            if job.pending_review:
                counts += f" · {job.pending_review} pending review"
            col.addWidget(label(counts, "Muted"))
            if job.error:
                col.addWidget(label(job.error, "Faint", wrap=True))
            for w in job.warnings[:2]:
                wl = label("⚠ " + w, "Faint", wrap=True)
                wl.setStyleSheet(f"color: {theme.WARN}; font-size: 12px;")
                col.addWidget(wl)
            rl.addLayout(col, 1)
            rl.addWidget(Pill(job.status.capitalize(), theme.STATUS_KIND.get(job.status, "neutral")), alignment=Qt.AlignTop)
            out = Path(job.output_path) if job.output_path else None
            exists = bool(out and out.is_file())
            b_open = button("Open", "", callback=lambda p=job.output_path: QDesktopServices.openUrl(QUrl.fromLocalFile(p)))
            b_open.setEnabled(exists and job.status in ("completed", "cancelled"))
            b_dir = button("Folder", "", "folder", callback=lambda p=job.output_path: QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(p).parent))))
            b_dir.setEnabled(exists)
            b_del = button("Delete", "Ghost", callback=lambda j=job: self._delete(j))
            for b in (b_open, b_dir, b_del):
                rl.addWidget(b, alignment=Qt.AlignTop)
            self.box.addWidget(row)

    def _delete(self, job) -> None:
        box = QMessageBox(self)
        box.setWindowTitle("Delete job")
        box.setText(f"Delete the record for “{job.source_name}”?")
        box.setInformativeText("Choose whether to also delete the redacted video. The original source video is never touched.")
        rec_only = box.addButton("Delete record only", QMessageBox.AcceptRole)
        both = box.addButton("Delete record and redacted video", QMessageBox.DestructiveRole)
        box.addButton(QMessageBox.Cancel)
        box.exec_()
        clicked = box.clickedButton()
        if clicked not in (rec_only, both):
            return
        self.app.service.storage.delete_job(job.job_id, delete_output=(clicked is both))
        self.banner.show_message("Job deleted.", "ok")
        self.on_show()
        self.app.refresh_badges()
