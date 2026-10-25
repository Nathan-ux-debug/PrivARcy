"""PrivARcy desktop app entry point:  ``python -m privarcy``  or the ``privarcy`` command."""

import sys

# Import torch BEFORE PyQt5 on Windows to avoid c10.dll initialisation errors.
try:
    import torch  # noqa: F401
except Exception as exc:  # the loader screen shows the real error if the backend needs it
    print(f"[warn] torch import failed: {exc}")

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

from privarcy import paths
from privarcy.app import theme
from privarcy.app.shell import MainWindow

LOGO = paths.assets_dir() / "privarcy-main-logo.png"


def main() -> int:
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    app.setApplicationName("PrivARcy")
    app.setStyleSheet(theme.stylesheet())
    window = MainWindow(str(LOGO) if LOGO.exists() else "")
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
