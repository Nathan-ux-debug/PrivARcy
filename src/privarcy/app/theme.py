"""PrivARcy design tokens + application stylesheet.

Derived from the supplied reference (a computer-vision workspace):
  * compact near-black sidebar, small labels, active item = violet outline
  * bright off-white workspace with a faint lavender wash at the top
  * thin 1px borders, 8-10px radii, restrained violet accent, almost no shadow
Branding, copy and layout are PrivARcy's own.
"""

SIDEBAR_BG = "#0E1013"
SIDEBAR_BORDER = "#1D2026"
SIDEBAR_TEXT = "#9BA1AC"
SIDEBAR_TEXT_ACTIVE = "#FFFFFF"
SIDEBAR_HOVER = "#171A20"

BG = "#FAFAFC"
CARD = "#FFFFFF"
BORDER = "#E6E6EC"
BORDER_STRONG = "#D4D4DD"
TEXT = "#14151A"
TEXT_MUTED = "#666B78"
TEXT_FAINT = "#9097A3"

ACCENT = "#6C47FF"
ACCENT_HOVER = "#5A35EE"
ACCENT_SOFT = "#F0ECFF"
ACCENT_BORDER = "#D9CFFF"

OK = "#12A150"
OK_SOFT = "#E7F7EE"
WARN = "#C77700"
WARN_SOFT = "#FFF4E0"
DANGER = "#D92D20"
DANGER_SOFT = "#FDECEA"
INFO = "#2F6FED"
INFO_SOFT = "#EAF1FF"

FONT = '"Inter", "Segoe UI", "Helvetica Neue", Arial, sans-serif'

PILL_STYLES = {
    "ok": (OK, OK_SOFT), "warn": (WARN, WARN_SOFT), "danger": (DANGER, DANGER_SOFT),
    "info": (INFO, INFO_SOFT), "accent": (ACCENT, ACCENT_SOFT), "neutral": (TEXT_MUTED, "#F1F1F5"),
}

STATUS_KIND = {
    "completed": "ok", "running": "info", "cancelled": "warn", "failed": "danger", "interrupted": "warn",
    "created": "neutral", "pending": "warn", "confirmed": "ok", "rejected": "neutral",
}


def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


def _check_png() -> str:
    """Write the white tick used inside checked checkboxes (QSS needs a file path)."""
    import os
    import tempfile
    from .icons import make_pixmap

    path = os.path.join(tempfile.gettempdir(), "privarcy_check.png").replace("\\", "/")
    make_pixmap("check", "#FFFFFF", 14).save(path, "PNG")
    return path


def _base_stylesheet() -> str:
    check = _check_png()
    return f"""
* {{ font-family: {FONT}; font-size: 13px; color: {TEXT}; }}
QMainWindow, QWidget#Root {{ background: {BG}; }}
QToolTip {{ background: {SIDEBAR_BG}; color: white; border: none; padding: 5px 8px; }}

/* ---------- sidebar ---------- */
QFrame#Sidebar {{ background: {SIDEBAR_BG}; border: none; }}
QFrame#Sidebar QLabel {{ color: {SIDEBAR_TEXT}; background: transparent; }}
QFrame#SidebarBrand {{ border: 1px solid {SIDEBAR_BORDER}; border-radius: 8px; background: #12151A; }}
QLabel#BrandName {{ color: white; font-weight: 600; font-size: 13px; }}
QFrame#SidebarDivider {{ background: {SIDEBAR_BORDER}; max-height: 1px; min-height: 1px; border: none; }}
QPushButton#NavButton {{
    text-align: left; padding: 7px 10px; border: 1px solid transparent; border-radius: 7px;
    color: {SIDEBAR_TEXT}; background: transparent; font-size: 12.5px;
}}
QPushButton#NavButton:hover {{ background: {SIDEBAR_HOVER}; color: white; }}
QPushButton#NavButton:checked {{ border: 1px solid {ACCENT}; color: {SIDEBAR_TEXT_ACTIVE}; background: #15131F; }}
QLabel#SidebarMeta {{ color: #6C7280; font-size: 11px; }}

/* ---------- typography ---------- */
QLabel#PageTitle {{ font-size: 22px; font-weight: 600; letter-spacing: -0.2px; }}
QLabel#PageDesc {{ color: {TEXT_MUTED}; font-size: 13px; }}
QLabel#HeroTitle {{ font-size: 28px; font-weight: 600; letter-spacing: -0.4px; }}
QLabel#HeroSub {{ color: {TEXT_MUTED}; font-size: 14px; }}
QLabel#SectionTitle {{ font-size: 14px; font-weight: 600; }}
QLabel#Muted {{ color: {TEXT_MUTED}; }}
QLabel#Faint {{ color: {TEXT_FAINT}; font-size: 12px; }}
QLabel#StatValue {{ font-size: 20px; font-weight: 600; }}
QLabel#StatLabel {{ color: {TEXT_MUTED}; font-size: 11.5px; }}
QLabel#Mono {{ font-family: "Cascadia Mono", "Consolas", monospace; font-size: 12px; color: {TEXT_MUTED}; }}

/* ---------- surfaces ---------- */
QFrame#Card {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 10px; }}
QFrame#CardFlat {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 8px; }}
QFrame#Prompt {{ background: {ACCENT_SOFT}; border: 1px solid {ACCENT_BORDER}; border-radius: 14px; }}
QFrame#PromptInner {{ background: white; border: 1px solid {ACCENT_BORDER}; border-radius: 10px; }}
QFrame#Row {{ background: transparent; border: none; border-bottom: 1px solid #F0F0F4; }}
QFrame#Divider {{ background: {BORDER}; max-height: 1px; min-height: 1px; border: none; }}
QFrame#VDivider {{ background: {BORDER}; max-width: 1px; min-width: 1px; border: none; }}
QFrame#Preview {{ background: #0B0C0F; border: 1px solid {BORDER}; border-radius: 8px; }}
QLabel#PreviewText {{ color: #8C93A0; background: transparent; }}

/* ---------- buttons ---------- */
QPushButton {{ border-radius: 8px; padding: 7px 14px; border: 1px solid {BORDER_STRONG}; background: white; color: {TEXT}; }}
QPushButton:hover {{ background: #F6F6F9; }}
QPushButton:disabled {{ color: {TEXT_FAINT}; background: #F6F6F9; border-color: {BORDER}; }}
QPushButton#Primary {{ background: {ACCENT}; border: 1px solid {ACCENT}; color: white; font-weight: 600; }}
QPushButton#Primary:hover {{ background: {ACCENT_HOVER}; border-color: {ACCENT_HOVER}; }}
QPushButton#Primary:disabled {{ background: #CFC4FF; border-color: #CFC4FF; color: white; }}
QPushButton#Danger {{ background: white; border: 1px solid #F0B7B2; color: {DANGER}; }}
QPushButton#Danger:hover {{ background: {DANGER_SOFT}; }}
QPushButton#Ghost {{ border: none; background: transparent; color: {ACCENT}; padding: 4px 6px; font-weight: 500; }}
QPushButton#Ghost:hover {{ background: {ACCENT_SOFT}; }}
QPushButton#Chip {{ border-radius: 16px; padding: 6px 14px; background: white; border: 1px solid {BORDER}; color: {TEXT}; }}
QPushButton#Chip:hover {{ border-color: {ACCENT_BORDER}; background: {ACCENT_SOFT}; }}
QPushButton#Send {{ background: {ACCENT}; border: none; border-radius: 16px; min-width: 32px; max-width: 32px; min-height: 32px; max-height: 32px; padding: 0; color: white; font-size: 15px; font-weight: 700; }}
QPushButton#Send:hover {{ background: {ACCENT_HOVER}; }}
QPushButton#Send:disabled {{ background: #CFC4FF; }}

/* ---------- inputs ---------- */
QLineEdit, QComboBox, QSpinBox, QPlainTextEdit, QTextEdit {{
    border: 1px solid {BORDER_STRONG}; border-radius: 8px; padding: 6px 9px; background: white; selection-background-color: {ACCENT_SOFT};
}}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus {{ border: 1px solid {ACCENT}; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{ border: 1px solid {BORDER}; background: white; selection-background-color: {ACCENT_SOFT}; selection-color: {TEXT}; outline: none; }}
QCheckBox, QRadioButton {{ spacing: 8px; background: transparent; }}
QCheckBox::indicator, QRadioButton::indicator {{ width: 16px; height: 16px; }}
QCheckBox::indicator {{ border: 1px solid {BORDER_STRONG}; border-radius: 4px; background: white; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; image: url({check}); }}
QRadioButton::indicator {{ width: 14px; height: 14px; border: 1px solid {BORDER_STRONG}; border-radius: 8px; background: white; }}
QRadioButton::indicator:checked {{ width: 6px; height: 6px; border: 5px solid {ACCENT}; border-radius: 8px; background: white; }}
QSlider::groove:horizontal {{ height: 4px; background: {BORDER}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
QSlider::handle:horizontal {{ width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; background: white; border: 2px solid {ACCENT}; }}
QProgressBar {{ border: none; background: #ECECF2; border-radius: 4px; max-height: 8px; min-height: 8px; text-align: center; color: transparent; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 4px; }}

/* ---------- lists / tables ---------- */
QListWidget {{ border: none; background: transparent; outline: none; }}
QListWidget::item {{ border: 1px solid transparent; border-radius: 8px; padding: 0; margin: 2px 0; }}
QListWidget::item:selected {{ background: {ACCENT_SOFT}; border: 1px solid {ACCENT_BORDER}; }}
QListWidget::item:hover:!selected {{ background: #F6F6F9; }}
QTableWidget {{ border: none; background: white; gridline-color: transparent; selection-background-color: {ACCENT_SOFT}; selection-color: {TEXT}; outline: none; }}
QTableWidget::item {{ padding: 6px 8px; border-bottom: 1px solid #F0F0F4; }}
QHeaderView::section {{ background: white; color: {TEXT_MUTED}; border: none; border-bottom: 1px solid {BORDER}; padding: 8px; font-weight: 500; font-size: 12px; text-align: left; }}

QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: #D5D5DE; border-radius: 4px; min-height: 30px; }}
QScrollBar::handle:vertical:hover {{ background: #BDBDC9; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: #D5D5DE; border-radius: 4px; min-width: 30px; }}
"""


SIDEBAR_WIDTH = 264
SIDEBAR_COLLAPSED = 78


def stylesheet() -> str:
    return _base_stylesheet() + f"""
/* ---------- big SaaS sidebar ---------- */
QFrame#SidebarBrand {{ border: 1px solid {SIDEBAR_BORDER}; border-radius: 12px; background: #151821; }}
QFrame#Sidebar QLabel#BrandName {{ color: white; font-weight: 700; font-size: 18px; }}
QPushButton#NavButton {{ text-align: left; padding: 12px 14px; border-radius: 12px; font-size: 14.5px; font-weight: 500; }}
QPushButton#NavButton:checked {{ border: 1px solid #7C5CFF; background: #1B1840; font-weight: 600; }}
QPushButton#NavButton[hovered="true"] {{ background: #1A1D26; color: white; }}
QPushButton#SidebarPrimary {{ background: {ACCENT}; border: 1px solid {ACCENT}; color: white; font-weight: 700; font-size: 14.5px; padding: 12px; border-radius: 12px; }}
QPushButton#SidebarPrimary:hover {{ background: {ACCENT_HOVER}; }}
QPushButton#SidebarPill {{ text-align: left; background: #2B2010; border: 1px solid #D69E2E; color: #FBBF24; font-weight: 600; font-size: 13.5px; padding: 11px 14px; border-radius: 12px; }}
QPushButton#SidebarPill[ok="true"] {{ background: #0F2A1B; border-color: #1F7A46; color: #5BD28C; }}
QPushButton#SidebarProfile {{ text-align: left; background: transparent; border: none; color: white; font-weight: 600; font-size: 14px; padding: 9px 10px; border-radius: 12px; }}
QPushButton#SidebarProfile:hover {{ background: #1A1D26; }}
QPushButton#Collapse {{ background: #1A1D26; border: 1px solid #2A2D38; border-radius: 13px; padding: 0; min-width: 26px; max-width: 26px; min-height: 26px; max-height: 26px; }}
QPushButton#Collapse:hover {{ background: #262A36; }}
QFrame#Flyout {{ background: #12121C; border: 1px solid #2A2A3C; border-radius: 14px; }}
QFrame#Flyout QLabel {{ background: transparent; color: #8A8AA3; }}
QLabel#FlyoutTitle {{ color: #8A8AA3; font-size: 13.5px; font-weight: 600; }}
QLabel#FlyoutName {{ color: white; font-size: 15px; font-weight: 700; }}
QLabel#FlyoutSection {{ color: #8A8AA3; font-size: 12px; padding: 8px 14px 2px 14px; }}
QPushButton#FlyoutItem {{ text-align: left; background: transparent; border: none; color: white; font-size: 14.5px; padding: 11px 14px; border-radius: 10px; }}
QPushButton#FlyoutItem:hover {{ background: #1E1E30; }}
QFrame#FlyoutDivider {{ background: #2A2A3C; max-height: 1px; min-height: 1px; border: none; }}

/* ---------- editor controls ---------- */
QPushButton#ClassChip {{ border-radius: 17px; padding: 8px 16px; background: white; border: 1.5px solid {BORDER}; color: {TEXT}; font-size: 13.5px; }}
QPushButton#ClassChip:hover {{ border-color: {ACCENT_BORDER}; }}
QPushButton#ClassChip:checked {{ background: {ACCENT_SOFT}; border: 1.5px solid {ACCENT}; color: #3A22B8; font-weight: 600; }}
QPushButton#ClassChip:disabled {{ background: #FFF8E8; border: 1.5px dashed #D69E2E; color: #7A4A00; }}
QPushButton#Seg {{ border-radius: 0; padding: 8px 10px; background: white; border: 1px solid {BORDER_STRONG}; border-left: none; color: {TEXT}; font-weight: 500; }}
QPushButton#SegFirst {{ border-top-left-radius: 8px; border-bottom-left-radius: 8px; border-left: 1px solid {BORDER_STRONG}; }}
QPushButton#SegLast {{ border-top-right-radius: 8px; border-bottom-right-radius: 8px; }}
QPushButton#Seg:checked {{ background: {ACCENT}; border-color: {ACCENT}; color: white; font-weight: 600; }}
QFrame#FaceTile {{ background: white; border: 2px solid {BORDER}; border-radius: 12px; }}
QFrame#FaceTile[state="blurred"] {{ border-color: {ACCENT}; background: {ACCENT_SOFT}; }}
QFrame#FaceTile[state="visible"] {{ border-color: {OK}; background: {OK_SOFT}; }}
QFrame#Summary {{ background: {ACCENT_SOFT}; border: 1px solid {ACCENT_BORDER}; border-radius: 12px; }}
"""
