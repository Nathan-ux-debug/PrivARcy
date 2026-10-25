#!/usr/bin/env bash
# macOS / Linux:  ./scripts/setup.sh && ./scripts/run.sh
set -euo pipefail
cd "$(dirname "$0")/.."
[ -d .venv ] || python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[dev]"
