"""Application shell: big SaaS sidebar (with hover menus) + page stack + loading / error states."""

from __future__ import annotations

import json

from PyQt5.QtCore import QSettings, QSize, Qt, QUrl
from PyQt5.QtGui import QDesktopServices, QIcon, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QFrame, QHBoxLayout, QLabel, QMainWindow, QMessageBox, QProgressBar, QPushButton, QStackedWidget,
    QVBoxLayout, QWidget,
)

import privarcy
from privarcy import paths

from . import theme
from .flyout import FlyoutMenu, HoverController
from .icons import make_icon
from .widgets import button, label
from .workers import BackendLoader

NAV = [
    ("Dashboard", "dashboard"), ("Video Processing", "video"), ("Live Camera", "camera"), ("Review Queue", "review"),
    ("Correction Log", "log"), ("History & Outputs", "history"),
]
SIDEBAR_WIDTH = theme.SIDEBAR_WIDTH
ICON = 22
SETTINGS_MENU = [("redaction", "settings", "Redaction Defaults"), ("classes", "target", "Detection Classes"),
                 ("thresholds", "settings", "Confidence Thresholds"), ("camera", "camera", "Camera Defaults"),
                 ("storage", "folder", "Output & Storage"), ("engines", "cpu", "Models & Engines"), ("privacy", "lock", "Privacy")]


class MainWindow(QMainWindow):
    def __init__(self, logo_path: str = "", autoload: bool = True, settings: QSettings | None = None) -> None:
        super().__init__()
        self.setWindowTitle("PrivARcy")
        self.resize(1360, 820)
        self.setMinimumSize(980, 640)
        self.logo_path = logo_path
        if logo_path:
            pm = QPixmap(logo_path)
            if not pm.isNull():
                self.setWindowIcon(QIcon(pm))
        self.settings = settings or QSettings("PrivARcy", "Dashboard2")
        self.service = None
        self.pages: dict = {}
        self.nav_buttons: dict = {}
        self.current = "Dashboard"
        self._busy = 0
        self._loader = None

        root = QWidget()
        root.setObjectName("Root")
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        self.top_stack = QStackedWidget()
        outer.addWidget(self.top_stack)
        self.top_stack.addWidget(self._build_loading())   # 0
        self.top_stack.addWidget(self._build_error())     # 1
        self.top_stack.addWidget(self._build_app())       # 2
        if autoload:
            self.start_loading()

    # ------------------------------------------------------------ load states
    def _build_loading(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setAlignment(Qt.AlignCenter)
        lay.setSpacing(10)
        if self.logo_path:
            logo = QLabel()
            pm = QPixmap(self.logo_path)
            if not pm.isNull():
                logo.setPixmap(pm.scaled(64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                logo.setAlignment(Qt.AlignCenter)
                lay.addWidget(logo)
        t = label("Starting PrivARcy", "PageTitle")
        t.setAlignment(Qt.AlignCenter)
        lay.addWidget(t)
        self.loading_text = label("Loading the detection model and preparing storage…", "Muted")
        self.loading_text.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.loading_text)
        bar = QProgressBar()
        bar.setRange(0, 0)
        bar.setFixedWidth(260)
        lay.addWidget(bar, alignment=Qt.AlignHCenter)
        return w

    def _build_error(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setAlignment(Qt.AlignCenter)
        lay.setSpacing(12)
        t = label("PrivARcy could not start", "PageTitle")
        t.setAlignment(Qt.AlignCenter)
        lay.addWidget(t)
        self.error_text = label("", "Muted", wrap=True)
        self.error_text.setMaximumWidth(720)
        self.error_text.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lay.addWidget(self.error_text, alignment=Qt.AlignHCenter)
        row = QHBoxLayout()
        row.addStretch()
        row.addWidget(button("Try again", "Primary", callback=self.start_loading))
        row.addWidget(button("Quit", "", callback=self.close))
        row.addStretch()
        lay.addLayout(row)
        return w

    def start_loading(self) -> None:
        self.top_stack.setCurrentIndex(0)
        self._loader = BackendLoader()
        self._loader.loaded.connect(self._on_loaded)
        self._loader.failed.connect(self._on_failed)
        self._loader.start()

    def _on_failed(self, message: str) -> None:
        self.error_text.setText(message)
        self.top_stack.setCurrentIndex(1)

    def _on_loaded(self, service) -> None:
        self.service = service
        self._restore_settings()
        from .pages.dashboard import DashboardPage
        from .pages.video import VideoPage
        from .pages.live import LivePage
        from .pages.review import ReviewPage
        from .pages.corrections import CorrectionsPage
        from .pages.history import HistoryPage
        from .pages.settings import SettingsPage

        classes = {"Dashboard": DashboardPage, "Video Processing": VideoPage, "Live Camera": LivePage, "Review Queue": ReviewPage,
                   "Correction Log": CorrectionsPage, "History & Outputs": HistoryPage, "Settings": SettingsPage}
        for name, cls in classes.items():
            page = cls(self)
            self.pages[name] = page
            self.page_stack.addWidget(page)
        self.top_stack.setCurrentIndex(2)
        self.default_face_mode = str(self.settings.value("default_face_mode", "auto"))
        if self.settings.value("sidebar_collapsed", False) in (True, "true"):
            self.set_collapsed(True)
        self.navigate("Dashboard")
        self.refresh_badges()
        self.refresh_status()

    # ------------------------------------------------------------------ layout
    def _build_app(self) -> QWidget:
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.sidebar = sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(SIDEBAR_WIDTH)
        sl = QVBoxLayout(sidebar)
        sl.setContentsMargins(14, 16, 14, 14)
        sl.setSpacing(5)

        self.brand = brand = QFrame()
        brand.setObjectName("SidebarBrand")
        bl = QHBoxLayout(brand)
        bl.setContentsMargins(12, 10, 12, 10)
        bl.setSpacing(12)
        mark = QLabel()
        mark.setFixedSize(34, 34)
        mark.setAlignment(Qt.AlignCenter)
        pm = QPixmap(self.logo_path) if self.logo_path else QPixmap()
        if not pm.isNull():
            mark.setPixmap(pm.scaled(34, 34, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            mark.setPixmap(make_icon("shield", "#FFFFFF", 30).pixmap(30, 30))
        bl.addWidget(mark)
        self.brand_name = QLabel("PrivARcy")
        self.brand_name.setObjectName("BrandName")
        bl.addWidget(self.brand_name, 1)
        sl.addWidget(brand)
        sl.addSpacing(12)

        for text, icon in NAV:
            sl.addWidget(self._nav_button(text, icon))
        sl.addStretch(1)

        self.status_pill = QPushButton("  Checking models…")
        self.status_pill.setObjectName("SidebarPill")
        self.status_pill.setCursor(Qt.PointingHandCursor)
        self.status_pill.setIcon(make_icon("warn", "#FBBF24", 20))
        self.status_pill.clicked.connect(lambda: self.open_settings("engines"))
        sl.addWidget(self.status_pill)
        self.new_job_btn = QPushButton("  New job")
        self.new_job_btn.setObjectName("SidebarPrimary")
        self.new_job_btn.setIcon(make_icon("plus", "#FFFFFF", 20))
        self.new_job_btn.setCursor(Qt.PointingHandCursor)
        self.new_job_btn.clicked.connect(lambda: self.navigate("Dashboard"))
        sl.addWidget(self.new_job_btn)
        self.activity_btn = self._plain_nav("Activity", "bell", lambda: self.navigate("History & Outputs"))
        sl.addWidget(self.activity_btn)
        self.settings_btn = self._plain_nav("Settings", "settings", lambda: self.navigate("Settings"))
        self.nav_buttons["Settings"] = self.settings_btn
        self.settings_btn.setProperty("page", "Settings")
        sl.addWidget(self.settings_btn)
        self.profile_btn = QPushButton("  Local workspace")
        self.profile_btn.setObjectName("SidebarProfile")
        self.profile_btn.setIcon(make_icon("user", "#B8B8D0", 24))
        self.profile_btn.setCursor(Qt.PointingHandCursor)
        sl.addWidget(self.profile_btn)
        self.busy_label = QLabel("Local processing")
        self.busy_label.setObjectName("SidebarMeta")
        self.busy_label.setContentsMargins(12, 2, 0, 0)
        sl.addWidget(self.busy_label)

        self.page_stack = QStackedWidget()
        lay.addWidget(sidebar)
        lay.addWidget(self.page_stack, 1)
        self.page_stack.setObjectName("PageStack")
        self.page_stack.setStyleSheet("QStackedWidget#PageStack { background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #F3EEFF, stop:0.22 #FAFAFC, stop:1 #FAFAFC); }")
        self.app_widget = w
        self._build_flyouts(w)
        self.collapse_btn = QPushButton(w)
        self.collapse_btn.setObjectName("Collapse")
        self.collapse_btn.setCursor(Qt.PointingHandCursor)
        self.collapse_btn.setToolTip("Collapse or expand the sidebar")
        self.collapse_btn.clicked.connect(lambda: self.set_collapsed(not self.collapsed))
        self.collapsed = False
        self._place_collapse()
        return w

    # ------------------------------------------------------------------ hover menus
    def _build_flyouts(self, host: QWidget) -> None:
        settings_menu = FlyoutMenu(host, "App Settings")
        for key, icon, text in SETTINGS_MENU:
            settings_menu.add_item(key, icon, text, lambda k=key: self.open_settings(k))
        header = QWidget()
        hl = QHBoxLayout(header)
        hl.setContentsMargins(8, 8, 8, 8)
        hl.setSpacing(12)
        avatar = QLabel()
        avatar.setFixedSize(44, 44)
        avatar.setAlignment(Qt.AlignCenter)
        avatar.setStyleSheet("background:#2B2B3F;border-radius:22px;")
        avatar.setPixmap(make_icon("user", "#B8B8D0", 26).pixmap(26, 26))
        hl.addWidget(avatar)
        who = QVBoxLayout()
        who.setSpacing(1)
        name = QLabel("Local workspace")
        name.setObjectName("FlyoutName")
        sub = QLabel("Everything stays on this device")
        sub.setStyleSheet("font-size:12px;color:#8A8AA3;background:transparent;")
        who.addWidget(name)
        who.addWidget(sub)
        hl.addLayout(who, 1)
        profile_menu = FlyoutMenu(host, header=header)
        profile_menu.add_item("prefs", "user", "Preferences", lambda: self.open_settings("redaction"))
        profile_menu.add_item("about", "shield", "About PrivARcy", self.show_about)
        profile_menu.section("Resources")
        profile_menu.add_item("docs", "doc", "Documentation", lambda: self._open_project_file("docs"))
        profile_menu.add_item("changelog", "log", "Changelog", lambda: self._open_project_file("CHANGELOG.md"))
        profile_menu.section("Get help")
        profile_menu.add_item("diag", "bug", "Copy diagnostics for a bug report", self.copy_diagnostics)
        self.settings_menu, self.profile_menu = settings_menu, profile_menu
        self.settings_hover = HoverController(self.settings_btn, settings_menu, host)
        self.profile_hover = HoverController(self.profile_btn, profile_menu, host)

    def _plain_nav(self, text: str, icon: str, callback) -> QPushButton:
        b = QPushButton("  " + text)
        b.setObjectName("NavButton")
        b.setCheckable(True)
        b.setCursor(Qt.PointingHandCursor)
        b.setIcon(make_icon(icon, theme.SIDEBAR_TEXT, ICON))
        b.setIconSize(QSize(ICON, ICON))
        b.setProperty("icon_name", icon)
        b.setProperty("label", text)
        b.clicked.connect(lambda _=False: callback())
        return b

    def open_settings(self, section: str = "") -> None:
        self.navigate("Settings")
        page = self.pages.get("Settings")
        if page is not None and hasattr(page, "focus_section"):
            page.focus_section(section)

    def show_about(self) -> None:
        QMessageBox.information(self, "About PrivARcy", f"PrivARcy {privarcy.__version__}\n\nLocal-first privacy redaction.\n"
                                f"Data folder: {paths.data_dir()}\nModels folder: {paths.models_dir()}")

    def _open_project_file(self, name: str) -> None:
        root = paths.project_root()
        target = (root / name) if root else None
        if target is not None and target.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))
        else:
            QMessageBox.information(self, "Not found", f"{name} is not available in this installation.")

    def copy_diagnostics(self) -> None:
        if self.service is None:
            return
        info = self.service.diagnostics()
        info["version"] = privarcy.__version__
        QApplication.clipboard().setText(json.dumps(info, indent=2, default=str))
        QMessageBox.information(self, "Copied", "Diagnostics were copied to the clipboard. They contain no video content or camera passwords.")

    # ------------------------------------------------------------------ collapse
    def _place_collapse(self) -> None:
        if hasattr(self, "collapse_btn"):
            width = theme.SIDEBAR_COLLAPSED if self.collapsed else SIDEBAR_WIDTH
            self.collapse_btn.move(width - 13, 78)
            self.collapse_btn.setIcon(make_icon("chevron-right" if self.collapsed else "chevron-left", "#B8B8CE", 14))
            self.collapse_btn.raise_()

    def set_collapsed(self, collapsed: bool) -> None:
        self.collapsed = collapsed
        self.sidebar.setFixedWidth(theme.SIDEBAR_COLLAPSED if collapsed else SIDEBAR_WIDTH)
        self.brand_name.setVisible(not collapsed)
        self.busy_label.setVisible(not collapsed)
        for btn, text in [(b, b.property("label") or b.property("page")) for b in self.nav_buttons.values()] + [(self.activity_btn, "Activity")]:
            btn.setText("" if collapsed else "  " + str(text).replace("&", "&&"))
            btn.setToolTip(str(text) if collapsed else "")
        self.new_job_btn.setText("" if collapsed else "  New job")
        self.profile_btn.setText("" if collapsed else "  Local workspace")
        self.status_pill.setText("" if collapsed else self._pill_text)
        self.settings.setValue("sidebar_collapsed", collapsed)
        self._place_collapse()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._place_collapse()

    # ------------------------------------------------------------------ status pill
    _pill_text = "  Checking models…"

    def refresh_status(self) -> None:
        if self.service is None:
            return
        diag = self.service.diagnostics()
        ocr_ok, model_ok = diag["ocr"]["available"], diag["detector"]["loaded"]
        if not model_ok:
            text, ok, icon = "  Detector not loaded", False, "warn"
        elif not ocr_ok:
            text, ok, icon = "  OCR unavailable", False, "warn"
        else:
            text, ok, icon = "  Models ready", True, "check"
        self._pill_text = text
        self.status_pill.setText("" if self.collapsed else text)
        self.status_pill.setProperty("ok", ok)
        self.status_pill.setIcon(make_icon(icon, "#5BD28C" if ok else "#FBBF24", 20))
        self.status_pill.setToolTip("" if ocr_ok else diag["ocr"]["reason"])
        self.status_pill.style().unpolish(self.status_pill)
        self.status_pill.style().polish(self.status_pill)

    def _nav_button(self, text: str, icon: str) -> QPushButton:
        b = QPushButton("  " + text.replace("&", "&&"))
        b.setObjectName("NavButton")
        b.setCheckable(True)
        b.setCursor(Qt.PointingHandCursor)
        b.setIcon(make_icon(icon, theme.SIDEBAR_TEXT, ICON))
        b.setIconSize(QSize(ICON, ICON))
        b.setProperty("icon_name", icon)
        b.setProperty("page", text)
        b.setProperty("label", text)
        b.clicked.connect(lambda _=False, t=text: self.navigate(t))
        self.nav_buttons[text] = b
        return b

    # -------------------------------------------------------------- navigation
    def navigate(self, name: str) -> None:
        if self.service is None or name not in self.pages:
            return
        old = self.pages.get(self.current)
        if old is not None and self.current != name:
            old.on_hide()
        self.current = name
        self.page_stack.setCurrentWidget(self.pages[name])
        for text, b in self.nav_buttons.items():
            active = text == name
            b.setChecked(active)
            b.setIcon(make_icon(b.property("icon_name"), "#FFFFFF" if active else theme.SIDEBAR_TEXT, ICON))
        self.pages[name].on_show()
        for menu in (getattr(self, "settings_menu", None), getattr(self, "profile_menu", None)):
            if menu is not None:
                menu.hide()

    def open_video(self, path: str) -> None:
        self.navigate("Video Processing")
        self.pages["Video Processing"].set_video(path)

    def set_busy(self, busy: bool) -> None:
        self._busy += 1 if busy else -1
        self._busy = max(0, self._busy)
        self.busy_label.setText("● Working…" if self._busy else "Local processing")
        self.busy_label.setStyleSheet("color: #8C74FF;" if self._busy else "")

    def refresh_badges(self) -> None:
        if self.service is None:
            return
        n = self.service.overview()["pending_review"]
        label = "  Review Queue" + (f"     {n}" if n else "")
        self.nav_buttons["Review Queue"].setText("" if self.collapsed else label)
        self.nav_buttons["Review Queue"].setToolTip(f"Review Queue ({n} pending)" if n else "")

    def settings_changed(self) -> None:
        settings_page = self.pages.get("Settings")
        if settings_page is not None and self.current != "Settings":
            settings_page._load()

    # -------------------------------------------------------------- persistence
    _KEYS = ("redaction_method", "screen_policy", "hold_missed_detections", "temporal_smoothing",
             "ocr_enabled", "ocr_model", "detection_interval", "live_ocr_enabled")

    def persist_settings(self) -> None:
        c = self.service.config
        for k in self._KEYS:
            self.settings.setValue(k, getattr(c, k))
        self.settings.setValue("enabled_classes", sorted(c.enabled_classes))

    def _restore_settings(self) -> None:
        def b(v): return v if isinstance(v, bool) else str(v).lower() == "true"
        s, values = self.settings, {}
        casts = {"hold_missed_detections": b, "ocr_enabled": b, "live_ocr_enabled": b, "temporal_smoothing": int, "detection_interval": int}
        for k in self._KEYS:
            if s.contains(k):
                try:
                    v = s.value(k)
                    values[k] = casts.get(k, str)(v)
                except (TypeError, ValueError):
                    pass
        if s.contains("enabled_classes"):
            v = s.value("enabled_classes")
            values["enabled_classes"] = set(v if isinstance(v, list) else [v])
        if values:
            self.service.update_settings(**values)

    # ------------------------------------------------------------------- close
    def closeEvent(self, event) -> None:
        if self._busy:
            ans = QMessageBox.question(self, "Quit PrivARcy", "Work is still running. Stop it and quit?")
            if ans != QMessageBox.Yes:
                event.ignore()
                return
        for page in self.pages.values():
            try:
                page.shutdown()
            except Exception:
                pass
        if self.service is not None:
            self.service.shutdown()
        super().closeEvent(event)
