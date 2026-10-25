"""Reusable building blocks. Keep pages free of styling details."""

from __future__ import annotations

from typing import Callable, Optional

from PyQt5.QtCore import QSize, Qt, pyqtSignal
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QButtonGroup, QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea, QSizePolicy, QVBoxLayout, QWidget,
)

from . import theme
from .icons import make_icon, make_pixmap

MAX_CONTENT_WIDTH = 1180


def label(text: str = "", name: str = "", wrap: bool = False) -> QLabel:
    lbl = QLabel(text)
    if name:
        lbl.setObjectName(name)
    lbl.setWordWrap(wrap)
    lbl.setTextInteractionFlags(Qt.NoTextInteraction)
    return lbl


def button(text: str, kind: str = "", icon: str = "", icon_color: str = "", callback: Optional[Callable] = None) -> QPushButton:
    b = QPushButton(text)
    if kind:
        b.setObjectName(kind)
    if icon:
        color = icon_color or ("#FFFFFF" if kind == "Primary" else theme.TEXT_MUTED)
        b.setIcon(make_icon(icon, color, 16))
        b.setIconSize(QSize(16, 16))
    b.setCursor(Qt.PointingHandCursor)
    if callback:
        b.clicked.connect(lambda _=False: callback())
    return b


def hline() -> QFrame:
    f = QFrame()
    f.setObjectName("Divider")
    f.setFrameShape(QFrame.NoFrame)
    return f


class Card(QFrame):
    """White surface with a thin border. ``layout`` is a QVBoxLayout."""

    def __init__(self, title: str = "", action_text: str = "", on_action: Optional[Callable] = None,
                 padding: int = 18, flat: bool = False, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("CardFlat" if flat else "Card")
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)     # never stretch taller than its content
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(padding, padding - 2, padding, padding)
        self.layout.setSpacing(10)
        if title or action_text:
            head = QHBoxLayout()
            head.setSpacing(8)
            self.title_label = label(title, "SectionTitle")
            head.addWidget(self.title_label)
            head.addStretch()
            if action_text:
                head.addWidget(button(action_text, "Ghost", callback=on_action))
            self.layout.addLayout(head)

    def add(self, widget_or_layout, stretch: int = 0):
        if isinstance(widget_or_layout, QWidget):
            self.layout.addWidget(widget_or_layout, stretch)
        else:
            self.layout.addLayout(widget_or_layout, stretch)
        return widget_or_layout


class Pill(QLabel):
    def __init__(self, text: str = "", kind: str = "neutral", parent=None) -> None:
        super().__init__(text, parent)
        self.set(text, kind)
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)

    def set(self, text: str, kind: str) -> None:
        fg, bg = theme.PILL_STYLES.get(kind, theme.PILL_STYLES["neutral"])
        self.setText(text)
        self.setStyleSheet(f"QLabel {{ color: {fg}; background: {bg}; border-radius: 9px; padding: 2px 9px; font-size: 11.5px; font-weight: 500; }}")


class StatusDot(QLabel):
    def __init__(self, kind: str = "neutral", parent=None) -> None:
        super().__init__(parent)
        self.setFixedSize(9, 9)
        self.set(kind)

    def set(self, kind: str) -> None:
        fg, _ = theme.PILL_STYLES.get(kind, theme.PILL_STYLES["neutral"])
        self.setStyleSheet(f"background: {fg}; border-radius: 4px;")


class StatTile(QFrame):
    def __init__(self, caption: str, value: str = "—", parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("CardFlat")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 11, 14, 11)
        lay.setSpacing(2)
        self.value_label = label(value, "StatValue")
        self.caption_label = label(caption, "StatLabel")
        lay.addWidget(self.value_label)
        lay.addWidget(self.caption_label)

    def set_value(self, value: str) -> None:
        self.value_label.setText(value)


class Banner(QFrame):
    """Inline message: info / ok / warn / danger. Hidden until ``show_message``."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 9, 12, 9)
        self.text = QLabel()
        self.text.setWordWrap(True)
        self.text.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lay.addWidget(self.text, 1)
        self.hide()

    def show_message(self, text: str, kind: str = "info") -> None:
        fg, bg = theme.PILL_STYLES.get(kind, theme.PILL_STYLES["info"])
        self.setStyleSheet(f"QFrame {{ background: {bg}; border: 1px solid {theme.rgba(fg, 0.28)}; border-radius: 8px; }}")
        self.text.setStyleSheet(f"color: {fg}; background: transparent; border: none;")
        self.text.setText(text)
        self.show()


class EmptyState(QWidget):
    def __init__(self, icon: str, title: str, desc: str = "", parent=None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setAlignment(Qt.AlignCenter)
        lay.setSpacing(6)
        lay.setContentsMargins(10, 24, 10, 24)
        circle = QLabel()
        circle.setFixedSize(44, 44)
        circle.setAlignment(Qt.AlignCenter)
        circle.setStyleSheet(f"background: #F1F1F5; border-radius: 22px;")
        circle.setPixmap(make_pixmap(icon, theme.TEXT_MUTED, 20))
        lay.addWidget(circle, alignment=Qt.AlignHCenter)
        t = label(title)
        t.setStyleSheet("font-weight: 600;")
        t.setAlignment(Qt.AlignCenter)
        lay.addWidget(t)
        if desc:
            d = label(desc, "Muted", wrap=True)
            d.setAlignment(Qt.AlignCenter)
            d.setMaximumWidth(360)
            lay.addWidget(d, alignment=Qt.AlignHCenter)


class ImageLabel(QLabel):
    """Shows a pixmap scaled to fit, keeping aspect ratio; resizes with the window."""

    def __init__(self, placeholder: str = "", parent=None) -> None:
        super().__init__(parent)
        self._pm: Optional[QPixmap] = None
        self._placeholder = placeholder
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(160, 90)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setObjectName("PreviewText")
        self.setText(placeholder)

    def set_pixmap(self, pm: Optional[QPixmap]) -> None:
        self._pm = pm
        if pm is None or pm.isNull():
            self._pm = None
            self.setPixmap(QPixmap())
            self.setText(self._placeholder)
        else:
            self._render()

    def clear_image(self, placeholder: Optional[str] = None) -> None:
        if placeholder is not None:
            self._placeholder = placeholder
        self.set_pixmap(None)

    def _render(self) -> None:
        if self._pm is not None:
            self.setPixmap(self._pm.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._render()


class PreviewFrame(QFrame):
    def __init__(self, placeholder: str, min_h: int = 300, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("Preview")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.image = ImageLabel(placeholder)
        self.image.setMinimumHeight(min_h)
        lay.addWidget(self.image)


class PageScroll(QScrollArea):
    """Scrollable page body with a centred max-width column (no horizontal scrolling)."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        outer = QWidget()
        outer.setObjectName("Root")
        row = QHBoxLayout(outer)
        row.setContentsMargins(32, 28, 32, 32)
        row.addStretch(1)
        self.column = QWidget()
        self.column.setMaximumWidth(MAX_CONTENT_WIDTH)
        self.column.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.body = QVBoxLayout(self.column)
        self.body.setContentsMargins(0, 0, 0, 0)
        self.body.setSpacing(18)
        row.addWidget(self.column, 100)
        row.addStretch(1)
        self.setWidget(outer)


class Page(QWidget):
    """Base page: header (title/description/actions) + scrolling body.

    Subclasses implement ``on_show`` (refresh from backend) and optionally ``on_hide``."""

    def __init__(self, title: str, description: str, parent=None) -> None:
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        self.scroll = PageScroll()
        root.addWidget(self.scroll)
        self.body = self.scroll.body
        self.header_row = QHBoxLayout()
        text = QVBoxLayout()
        text.setSpacing(3)
        text.addWidget(label(title, "PageTitle"))
        text.addWidget(label(description, "PageDesc", wrap=True))
        self.header_row.addLayout(text, 1)
        self.header_actions = QHBoxLayout()
        self.header_actions.setSpacing(8)
        self.header_row.addLayout(self.header_actions)
        self.body.addLayout(self.header_row)

    def on_show(self) -> None:  # pragma: no cover - overridden
        pass

    def on_hide(self) -> None:  # pragma: no cover - overridden
        pass

    def shutdown(self) -> None:
        pass


def clear_layout(layout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        w = item.widget()
        if w is not None:
            w.setParent(None)
            w.deleteLater()
        elif item.layout() is not None:
            clear_layout(item.layout())


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} GB"


def human_duration(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def bgr_to_pixmap(frame) -> QPixmap:
    from PyQt5.QtGui import QImage
    import numpy as np

    rgb = frame[:, :, ::-1] if frame.ndim == 3 else frame
    rgb = np.ascontiguousarray(rgb)
    h, w = rgb.shape[:2]
    fmt = QImage.Format_RGB888 if rgb.ndim == 3 else QImage.Format_Grayscale8
    image = QImage(rgb.data, w, h, rgb.strides[0], fmt).copy()   # copy: detach from numpy buffer
    return QPixmap.fromImage(image)


class Segmented(QWidget):
    """Mutually exclusive buttons (a segmented control). ``changed`` emits the selected key."""
    changed = pyqtSignal(str)

    def __init__(self, options, current: str = "", parent=None) -> None:
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self.buttons: dict = {}
        for i, (key, text) in enumerate(options):
            b = QPushButton(text)
            b.setObjectName("Seg")
            b.setCheckable(True)
            b.setCursor(Qt.PointingHandCursor)
            b.setProperty("class", "")
            style = "SegFirst" if i == 0 else ("SegLast" if i == len(options) - 1 else "")
            if style:
                b.setStyleSheet("")
                b.setObjectName("Seg")
                b.setProperty("segpos", style)
            self._group.addButton(b)
            self.buttons[key] = b
            lay.addWidget(b, 1)
            b.clicked.connect(lambda _=False, k=key: self.changed.emit(k))
        self._round_ends()
        self.set_current(current or options[0][0], emit=False)

    def _round_ends(self) -> None:
        keys = list(self.buttons)
        radius = f"border-top-left-radius:8px;border-bottom-left-radius:8px;border-left:1px solid {theme.BORDER_STRONG};"
        self.buttons[keys[0]].setStyleSheet(f"QPushButton#Seg {{ {radius} }} QPushButton#Seg:checked {{ border-left:1px solid {theme.ACCENT}; }}")
        self.buttons[keys[-1]].setStyleSheet("QPushButton#Seg { border-top-right-radius:8px; border-bottom-right-radius:8px; }")

    def current(self) -> str:
        return next((k for k, b in self.buttons.items() if b.isChecked()), "")

    def set_current(self, key: str, emit: bool = True) -> None:
        if key in self.buttons:
            self.buttons[key].setChecked(True)
            if emit:
                self.changed.emit(key)


def chip_button(text: str, checked: bool = False) -> QPushButton:
    b = QPushButton(text)
    b.setObjectName("ClassChip")
    b.setCheckable(True)
    b.setChecked(checked)
    b.setCursor(Qt.PointingHandCursor)
    return b


class FaceTile(QFrame):
    """One person in the face picker: thumbnail, name, what will happen to them. Click toggles."""
    clicked = pyqtSignal(str)

    def __init__(self, face_id: str, name: str, thumb: Optional[QPixmap], parent=None) -> None:
        super().__init__(parent)
        self.face_id = face_id
        self.setObjectName("FaceTile")
        self.setCursor(Qt.PointingHandCursor)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 8, 10, 8)
        lay.setSpacing(10)
        self.thumb = QLabel()
        self.thumb.setFixedSize(54, 54)
        self.thumb.setAlignment(Qt.AlignCenter)
        if thumb is not None:
            self.thumb.setPixmap(thumb.scaled(54, 54, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation))
        lay.addWidget(self.thumb)
        col = QVBoxLayout()
        col.setSpacing(1)
        self.name = QLabel(name)
        self.name.setStyleSheet("font-weight:600;font-size:13.5px;background:transparent;")
        self.status = QLabel("")
        self.status.setStyleSheet(f"color:{theme.TEXT_MUTED};font-size:11.5px;background:transparent;")
        col.addWidget(self.name)
        col.addWidget(self.status)
        lay.addLayout(col, 1)

    def set_state(self, state: str, text: str, enabled: bool = True) -> None:
        self.setProperty("state", state)
        self.status.setText(text)
        self.setEnabled(enabled)
        self.style().unpolish(self)
        self.style().polish(self)

    def mousePressEvent(self, e) -> None:
        if self.isEnabled() and e.button() == Qt.LeftButton:
            self.clicked.emit(self.face_id)
