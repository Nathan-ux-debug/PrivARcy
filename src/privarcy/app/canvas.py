"""FrameCanvas: shows a (redacted) frame with clickable overlays.

Used by the editor and the live page. It only ever receives frames that are already rendered by the backend; overlays are
drawn on top (boxes, labels). Mouse positions are converted to IMAGE pixels so pages never deal with scaling.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from PyQt5.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import QSizePolicy, QWidget

from . import theme
from .widgets import bgr_to_pixmap

STATE_COLORS = {"blurred": theme.ACCENT, "visible": theme.OK, "review": theme.WARN, "selected": theme.INFO}


class FrameCanvas(QWidget):
    clicked = pyqtSignal(float, float)                 # image px
    rect_drawn = pyqtSignal(float, float, float, float)  # image px: x1, y1, x2, y2
    hovered = pyqtSignal(object)                       # overlay under the mouse or None

    def __init__(self, placeholder: str = "No video selected", parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(310, 280)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self.placeholder = placeholder
        self._pixmap: Optional[QPixmap] = None
        self._size = (0, 0)
        self.overlays: List = []
        self.mode = "select"                           # select | draw | magic
        self._hover = None
        self._drag: Optional[tuple] = None
        self.show_labels = True
        self.busy_text = ""

    # ------------------------------------------------------------------ content
    def set_frame(self, bgr: Optional[np.ndarray], overlays=()) -> None:
        if bgr is None:
            self._pixmap, self._size = None, (0, 0)
        else:
            self._pixmap = bgr_to_pixmap(bgr)
            self._size = (bgr.shape[1], bgr.shape[0])
        self.overlays = list(overlays)
        self.update()

    def set_overlays(self, overlays) -> None:
        self.overlays = list(overlays)
        self.update()

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        self.setCursor({"draw": Qt.CrossCursor, "magic": Qt.PointingHandCursor}.get(mode, Qt.ArrowCursor))
        self.update()

    def set_busy(self, text: str = "") -> None:
        self.busy_text = text
        self.update()

    # ------------------------------------------------------------------ geometry
    def image_rect(self) -> QRectF:
        if not self._pixmap:
            return QRectF(self.rect())
        w, h = self._size
        scale = min(self.width() / w, self.height() / h)
        iw, ih = w * scale, h * scale
        return QRectF((self.width() - iw) / 2, (self.height() - ih) / 2, iw, ih)

    def to_image(self, pos: QPointF) -> tuple:
        r = self.image_rect()
        w, h = self._size
        if not w:
            return (0.0, 0.0)
        return ((pos.x() - r.x()) / r.width() * w, (pos.y() - r.y()) / r.height() * h)

    def to_widget(self, box) -> QRectF:
        r = self.image_rect()
        w, h = self._size
        sx, sy = r.width() / w, r.height() / h
        return QRectF(r.x() + box[0] * sx, r.y() + box[1] * sy, (box[2] - box[0]) * sx, (box[3] - box[1]) * sy)

    def overlay_at(self, x: float, y: float):
        hits = [o for o in self.overlays if o.box[0] <= x <= o.box[2] and o.box[1] <= y <= o.box[3]]
        return min(hits, key=lambda o: (o.box[2] - o.box[0]) * (o.box[3] - o.box[1]), default=None)

    # ------------------------------------------------------------------ painting
    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(self.rect(), QColor("#0B0C0F"))
        if not self._pixmap:
            p.setPen(QColor("#8C93A0"))
            p.drawText(self.rect(), Qt.AlignCenter, self.placeholder)
            return
        r = self.image_rect()
        p.drawPixmap(r.toRect(), self._pixmap)
        font = QFont(p.font()); font.setPointSizeF(9.5); font.setBold(True); p.setFont(font)
        for o in self.overlays:
            color = QColor(STATE_COLORS.get(o.state, theme.ACCENT))
            if o.kind == "region":
                color = QColor(theme.ACCENT)
            pen = QPen(color, 3 if o is self._hover else 2)
            if o.state in ("visible", "review"):
                pen.setStyle(Qt.DashLine)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            wr = self.to_widget(o.box)
            p.drawRoundedRect(wr, 3, 3)
            if self.show_labels and o.label:
                text = o.label + {"visible": " · visible", "blurred": " · blur" if o.kind == "face" else ""}.get(o.state, "")
                fm = QFontMetrics(font)
                tag = QRectF(wr.x(), max(r.y(), wr.y() - 22), fm.horizontalAdvance(text) + 14, 20)
                p.setPen(Qt.NoPen); p.setBrush(color); p.drawRoundedRect(tag, 4, 4)
                p.setPen(QColor("white")); p.drawText(tag, Qt.AlignCenter, text)
        if self._drag:
            (x1, y1), (x2, y2) = self._drag
            p.setPen(QPen(QColor(theme.ACCENT), 2, Qt.DashLine)); p.setBrush(QColor(108, 71, 255, 40))
            p.drawRect(QRectF(QPointF(x1, y1), QPointF(x2, y2)).normalized())
        if self.busy_text:
            p.setPen(Qt.NoPen); p.setBrush(QColor(11, 12, 15, 170)); p.drawRoundedRect(QRectF(r.x() + 10, r.y() + 10, 190, 28), 8, 8)
            p.setPen(QColor("white")); p.drawText(QRectF(r.x() + 10, r.y() + 10, 190, 28), Qt.AlignCenter, self.busy_text)

    # ------------------------------------------------------------------ mouse
    def mousePressEvent(self, e) -> None:
        if not self._pixmap or e.button() != Qt.LeftButton or not self.image_rect().contains(e.pos()):
            return
        if self.mode == "draw":
            self._drag = ((e.x(), e.y()), (e.x(), e.y()))
        else:
            self.clicked.emit(*self.to_image(e.pos()))

    def mouseMoveEvent(self, e) -> None:
        if self._drag:
            self._drag = (self._drag[0], (e.x(), e.y()))
            self.update()
            return
        if self._pixmap and self.mode != "draw":
            o = self.overlay_at(*self.to_image(e.pos()))
            if o is not self._hover:
                self._hover = o
                self.hovered.emit(o)
                self.update()

    def mouseReleaseEvent(self, e) -> None:
        if self._drag:
            (ax, ay), (bx, by) = self._drag
            self._drag = None
            self.update()
            if abs(bx - ax) > 6 and abs(by - ay) > 6:
                x1, y1 = self.to_image(QPointF(min(ax, bx), min(ay, by)))
                x2, y2 = self.to_image(QPointF(max(ax, bx), max(ay, by)))
                w, h = self._size
                self.rect_drawn.emit(max(0.0, x1), max(0.0, y1), min(float(w), x2), min(float(h), y2))

    def leaveEvent(self, _e) -> None:
        if self._hover is not None:
            self._hover = None
            self.hovered.emit(None)
            self.update()
