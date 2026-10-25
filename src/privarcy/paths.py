"""Where things live on disk.

Source code is installed (or run) from ``src/``; everything the app *creates* goes under one data directory so
a clean checkout stays clean and backups are a single folder.

Resolution order for the data directory:
  1. ``PRIVARCY_HOME`` environment variable
  2. ``<project root>/data`` when running from a source checkout (a ``pyproject.toml`` sits next to ``src/``)
  3. ``~/PrivARcy`` when installed as a package
"""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
_CHECKOUT_ROOT = PACKAGE_DIR.parents[1]          # <root>/src/privarcy -> <root>


def project_root() -> Path | None:
    return _CHECKOUT_ROOT if (_CHECKOUT_ROOT / "pyproject.toml").is_file() else None


def data_dir() -> Path:
    env = os.environ.get("PRIVARCY_HOME")
    if env:
        return Path(env).expanduser()
    root = project_root()
    return (root / "data") if root else Path.home() / "PrivARcy"


def models_dir() -> Path:
    env = os.environ.get("PRIVARCY_MODELS")
    if env:
        return Path(env).expanduser()
    root = project_root()
    return (root / "models") if root else data_dir() / "models"


def model_path(name: str = "best.pt") -> Path:
    return models_dir() / name


def assets_dir() -> Path:
    return PACKAGE_DIR / "app" / "assets"
