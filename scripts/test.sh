#!/usr/bin/env bash
cd "$(dirname "$0")/.." && QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest "$@"
