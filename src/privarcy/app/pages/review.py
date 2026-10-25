"""Review Queue: inspect uncertain (50–80%) detections, confirm or reject.

Confirm = redact this in the output video. Reject = false positive, leave visible.
Both are written to the correction log. Decisions do NOT retrain the detector.
"""

from __future__ import annotations

from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QComboBox, QHBoxLayout, QLineEdit, QListWidget, QListWidgetItem, QVBoxLayout, QWidget,
)

from ..widgets import Banner, Card, EmptyState, ImageLabel, Page, Pill, button, clear_layout, label
from ..workers import FuncWorker

CLASS_TEXT = {"face": "Face", "id_card": "ID card", "credit_card": "Payment card", "screen": "Screen",
              "license_plate": "License plate", "sensitive_text": "Sensitive text"}


class ReviewPage(Page):
    def __init__(self, app) -> None:
        super().__init__("Review Queue", "Uncertain detections (50–80% confidence) wait here. Confirm to redact them in the output video, "
                         "or reject false positives.")
        self.app = app
        self.items: list[dict] = []
        self.current: dict | None = None
        self.worker = None

        self.banner = Banner()
        self.body.addWidget(self.banner)
        top = QHBoxLayout()
        self.job_filter = QComboBox()
        self.job_filter.setMinimumWidth(220)
        self.job_filter.currentIndexChanged.connect(self._populate)
        top.addWidget(label("Video", "Muted"))
        top.addWidget(self.job_filter)
        top.addStretch()
        self.count_pill = Pill("0 pending", "neutral")
        top.addWidget(self.count_pill)
        self.body.addLayout(top)

        cols = QHBoxLayout()
        cols.setSpacing(18)
        self.list_card = Card("Pending", padding=14)
        self.list = QListWidget()
        self.list.setIconSize(QSize(64, 48))
        self.list.setMinimumHeight(420)
        self.list.currentRowChanged.connect(self._select)
        self.list_card.add(self.list)
        self.empty = EmptyState("review", "Nothing to review", "When a video produces detections between 50% and 80% confidence, they appear here.")
        self.list_card.add(self.empty)
        cols.addWidget(self.list_card, 4)

        self.detail = Card("Detection", padding=18)
        self.crop = ImageLabel("Select an item")
        self.crop.setMinimumHeight(260)
        self.crop.setStyleSheet("background: #0B0C0F; border-radius: 8px; color: #8C93A0;")
        self.detail.add(self.crop, 1)
        self.fields = QVBoxLayout()
        self.fields.setSpacing(4)
        self.detail.add(self.fields)
        self.note = QLineEdit()
        self.note.setPlaceholderText("Optional note for the correction log")
        self.detail.add(self.note)
        row = QHBoxLayout()
        self.confirm_btn = button("Confirm — redact", "Primary", "check", callback=lambda: self._decide("confirmed"))
        self.reject_btn = button("Reject — leave visible", "", "x", callback=lambda: self._decide("rejected"))
        self.skip_btn = button("Skip", "Ghost", callback=self._skip)
        row.addWidget(self.confirm_btn)
        row.addWidget(self.reject_btn)
        row.addStretch()
        row.addWidget(self.skip_btn)
        self.detail.add(row)
        self.detail.add(label("Decisions are saved to the correction log. They do not retrain the model; see Correction Log for "
                              "evidence-based rule proposals.", "Faint", wrap=True))
        cols.addWidget(self.detail, 6)
        self.body.addLayout(cols)
        self._set_actions(False)

    # ----------------------------------------------------------------- data
    def on_show(self) -> None:
        if self.worker is not None:
            return
        self.items = self.app.service.pending_review_items()
        jobs = sorted({(i["job_id"], i["job_name"]) for i in self.items}, key=lambda t: t[1])
        keep = self.job_filter.currentData()
        self.job_filter.blockSignals(True)
        self.job_filter.clear()
        self.job_filter.addItem("All videos", None)
        for jid, name in jobs:
            self.job_filter.addItem(name, jid)
        idx = self.job_filter.findData(keep)
        self.job_filter.setCurrentIndex(max(0, idx))
        self.job_filter.blockSignals(False)
        self._populate()

    def _visible_items(self) -> list:
        jid = self.job_filter.currentData()
        return [i for i in self.items if jid is None or i["job_id"] == jid]

    def _populate(self) -> None:
        self.list.blockSignals(True)
        self.list.clear()
        visible = self._visible_items()
        for it in visible:
            row = QListWidgetItem()
            row.setData(Qt.UserRole, it["id"])
            widget = self._row_widget(it)
            row.setSizeHint(widget.sizeHint())
            self.list.addItem(row)
            self.list.setItemWidget(row, widget)
        self.list.blockSignals(False)
        self.empty.setVisible(not visible)
        self.list.setVisible(bool(visible))
        self.count_pill.set(f"{len(self.items)} pending", "warn" if self.items else "neutral")
        if visible:
            self.list.setCurrentRow(0)
            self._select(0)
        else:
            self.current = None
            self._show_item(None)

    def _row_widget(self, it: dict) -> QWidget:
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(8, 7, 8, 7)
        thumb = QLabel_pixmap(it.get("image_path"))
        lay.addWidget(thumb)
        col = QVBoxLayout()
        col.setSpacing(1)
        col.addWidget(label(CLASS_TEXT.get(it["label"], it["label"]), ""))
        col.addWidget(label(f"Frame {it['frame']}  ·  {it['time']}", "Faint"))
        lay.addLayout(col, 1)
        lay.addWidget(Pill(f"{it['confidence']:.0%}", "warn"))
        return w

    def _select(self, row: int) -> None:
        visible = self._visible_items()
        if 0 <= row < len(visible):
            self.current = visible[row]
            self._show_item(self.current)

    def _show_item(self, it: dict | None) -> None:
        clear_layout(self.fields)
        if it is None:
            self.crop.clear_image("Select an item")
            self._set_actions(False)
            return
        pm = QPixmap(it.get("image_path") or "")
        self.crop.set_pixmap(pm if not pm.isNull() else None)
        if pm.isNull():
            self.crop.setText("Preview not available (temporary crop was removed)")
        rows = [("Predicted class", CLASS_TEXT.get(it["label"], it["label"])),
                ("Confidence", f"{it['confidence']:.1%}"),
                ("Bounding box", "x1={}, y1={}, x2={}, y2={}".format(*it["bbox"])),
                ("Frame / time", f"{it['frame']}  ·  {it['time']}"),
                ("Video", it.get("job_name", ""))]
        if it.get("ocr_patterns"):
            rows.append(("Matched pattern", ", ".join(it["ocr_patterns"])))
        if len(it.get("track_observations", [])) > 1:
            rows.append(("Tracked over", f"{len(it['track_observations'])} frames (one decision applies to all)"))
        for k, v in rows:
            r = QHBoxLayout()
            k_lbl = label(k, "Muted")
            k_lbl.setFixedWidth(120)
            r.addWidget(k_lbl)
            r.addWidget(label(v), 1)
            holder = QWidget()
            holder.setLayout(r)
            self.fields.addWidget(holder)
        self._set_actions(True)

    def _set_actions(self, on: bool) -> None:
        for b in (self.confirm_btn, self.reject_btn, self.skip_btn, self.note):
            b.setEnabled(on and self.worker is None)

    # -------------------------------------------------------------- actions
    def _skip(self) -> None:
        row = self.list.currentRow()
        if row + 1 < self.list.count():
            self.list.setCurrentRow(row + 1)

    def _decide(self, decision: str) -> None:
        if self.current is None or self.worker is not None:
            return
        item = self.current
        self._set_actions(False)
        self.banner.show_message("Saving…" if decision == "rejected" else "Applying redaction to the output video…", "info")
        self.worker = FuncWorker(self.app.service.resolve_review_item, item, decision, self.note.text().strip())
        self.worker.done.connect(lambda res, i=item, d=decision: self._decided(i, d, res))
        self.worker.error.connect(self._failed)
        self.worker.finished.connect(lambda: setattr(self, "worker", None))
        self.worker.start()

    def _decided(self, item: dict, decision: str, result: dict) -> None:
        self.note.clear()
        msg = (f"Confirmed: redacted {result['frames_redacted']} frame(s) in the output video." if decision == "confirmed"
               else "Rejected: left visible. Logged as a false positive.")
        self.banner.show_message(msg, "ok")
        self.items = [i for i in self.items if i["id"] != item["id"]]
        self._populate()
        self.app.refresh_badges()

    def _failed(self, message: str) -> None:
        self.banner.show_message(f"Could not complete this decision: {message}. The item is still pending.", "danger")
        self._set_actions(self.current is not None)


def QLabel_pixmap(path):
    from PyQt5.QtWidgets import QLabel
    lbl = QLabel()
    lbl.setFixedSize(64, 48)
    lbl.setAlignment(Qt.AlignCenter)
    lbl.setStyleSheet("background: #0B0C0F; border-radius: 6px;")
    pm = QPixmap(path or "")
    if not pm.isNull():
        lbl.setPixmap(pm.scaled(64, 48, Qt.KeepAspectRatio, Qt.SmoothTransformation))
    return lbl
