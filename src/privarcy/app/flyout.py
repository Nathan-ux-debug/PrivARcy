"""Hover menus for the sidebar (Settings and profile), in the style of modern SaaS workspaces.

The menu is a child of the main window (not a separate OS window) so it behaves the same everywhere. It opens when the
pointer rests on its anchor button, stays open while the pointer is on the anchor OR the menu, and closes shortly after
the pointer leaves both. Click also toggles it, so it is usable with touch and keyboard.
"""

from __future__ import annotations

from typing import Callable, Optional

from PyQt5.QtCore import QEvent, QObject, QPoint, QTimer, Qt
from PyQt5.QtWidgets import QFrame, QLabel, QPushButton, QVBoxLayout, QWidget

from .icons import make_icon


class FlyoutMenu(QFrame):
    WIDTH = 300

    def __init__(self, parent: QWidget, title: str = "", header: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("Flyout")
        self.setFixedWidth(self.WIDTH)
        self.lay = QVBoxLayout(self)
        self.lay.setContentsMargins(10, 8, 10, 10)
        self.lay.setSpacing(2)
        if header is not None:
            self.lay.addWidget(header)
            self._divider()
        elif title:
            t = QLabel(title)
            t.setObjectName("FlyoutTitle")
            t.setAlignment(Qt.AlignCenter)
            t.setContentsMargins(0, 8, 0, 8)
            self.lay.addWidget(t)
            self._divider()
        self.items: dict = {}
        self.hide()

    def _divider(self) -> None:
        d = QFrame()
        d.setObjectName("FlyoutDivider")
        self.lay.addWidget(d)

    def section(self, text: str) -> None:
        self._divider()
        s = QLabel(text)
        s.setObjectName("FlyoutSection")
        self.lay.addWidget(s)

    def add_item(self, key: str, icon: str, text: str, callback: Callable) -> QPushButton:
        b = QPushButton("  " + text.replace("&", "&&"))
        b.setObjectName("FlyoutItem")
        b.setIcon(make_icon(icon, "#B8B8D0", 20))
        b.setIconSize(b.iconSize() * 1.0)
        b.setCursor(Qt.PointingHandCursor)
        b.clicked.connect(lambda _=False: (self.hide(), callback()))
        self.lay.addWidget(b)
        self.items[key] = b
        return b


class HoverController(QObject):
    """Opens ``menu`` next to ``anchor`` on hover; keeps it open while the pointer is on either."""

    def __init__(self, anchor: QWidget, menu: FlyoutMenu, host: QWidget, align_bottom: bool = True, delay_ms: int = 220) -> None:
        super().__init__(anchor)
        self.anchor, self.menu, self.host, self.align_bottom = anchor, menu, host, align_bottom
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(delay_ms)
        self._timer.timeout.connect(self._maybe_hide)
        anchor.installEventFilter(self)
        menu.installEventFilter(self)
        for child in menu.findChildren(QWidget):
            child.installEventFilter(self)
        anchor.setProperty("flyout_open", False)

    def eventFilter(self, obj, event) -> bool:
        t = event.type()
        if t == QEvent.Enter:
            self._timer.stop()
            if obj is self.anchor:
                self.show()
        elif t == QEvent.Leave:
            self._timer.start()
        elif t == QEvent.MouseButtonPress and obj is self.anchor:
            self.toggle()
        return False

    def show(self) -> None:
        self.menu.adjustSize()
        top_left = self.anchor.mapTo(self.host, QPoint(0, 0))
        x = top_left.x() + self.anchor.width() - 2
        y = top_left.y() + self.anchor.height() - self.menu.height() if self.align_bottom else top_left.y()
        y = max(8, min(y, self.host.height() - self.menu.height() - 8))
        self.menu.move(x, y)
        self.menu.raise_()
        self.menu.show()
        self.anchor.setProperty("flyout_open", True)

    def toggle(self) -> None:
        self.hide() if self.menu.isVisible() else self.show()

    def hide(self) -> None:
        self.menu.hide()
        self.anchor.setProperty("flyout_open", False)

    def _maybe_hide(self) -> None:
        if not (self.anchor.underMouse() or self.menu.underMouse() or any(c.underMouse() for c in self.menu.findChildren(QWidget))):
            self.hide()
