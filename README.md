# PrivARcy

Local-first privacy redaction for **recorded video** and **live cameras**. It detects faces, ID cards, payment cards, screens,
license plates and (optionally) sensitive text, and blurs them. Everything runs on your computer; nothing is uploaded.

## What you can do

| Area | What it does |
|---|---|
| **Offline** (Video Processing) | Open a video, preview the redaction instantly, pick who to blur, process to a new file. |
| **Auto Blur** | Chips for the detector's classes (Face, ID card, Payment card, Screen, License plate, Sensitive text/OCR). Toggle a chip and the preview updates immediately. |
| **Face choices** | Three modes: **Auto** (blur everyone, keep chosen people visible), **Selected only**, **Every face**. People are tracked, so a choice holds for the whole video. See [docs/FACE_MODES.md](docs/FACE_MODES.md). |
| **Manual Blur** | Draw rectangles/ellipses over anything, with a time range. |
| **Magic Blur** | Click a detected object to blur its shape; it is followed through the video. |
| **Live Camera** | Built-in/USB cameras, QR-paired Android/iPhone phone cameras, phone camera apps and network cameras (RTSP / HTTP). Same redaction pipeline; QR-paired video is never recorded. See [docs/CAMERAS.md](docs/CAMERAS.md). |
| **Review Queue / Correction Log** | Uncertain detections (50-80 %) wait for you; every decision is logged and can propose a safer rule. |

## Install and run

Windows (PowerShell):

```powershell
.\scripts\setup.ps1 
     # creates .venv and installs everything
# copy your trained detector to models\best.pt

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\run.ps1        # or:  .\.venv\Scripts\python.exe -m privarcy
```

macOS / Linux: `./scripts/setup.sh && ./scripts/run.sh`.
Sensitive-text (OCR) is optional: `pip install -e ".[ocr]"`.

Run the tests: `.\scripts\test.ps1` (or `python -m pytest`). See [docs/TESTING.md](docs/TESTING.md).

## Project layout

```
privarcy/
├── pyproject.toml            packaging, dependencies, pytest + ruff config
├── src/privarcy/             the application package (src layout)
│   ├── app/                  PyQt5 interface: shell, pages, canvas, hover menus, theme
│   ├── services/backend.py   BackendService: the one facade the UI calls
│   ├── pipeline/             FrameEngine, video pipeline, live session, preview, magic blur, edit plan
│   ├── faces/                detectors, identities (FaceBook), face policy, video scanner
│   ├── capture/              camera sources, URL validation, discovery and connection probe
│   │   └── phone_pairing/    local HTTPS/WebRTC service, browser fallback, and QR dialog
│   ├── detection/            YOLO wrapper
│   ├── decision/  redaction/  ocr/  rules/  storage/
│   └── mlops/                dataset / active-learning / labelling tools (not used by the app)
├── tests/                    unit, integration and UI tests; phone pairing is grouped in tests/phone_pairing/
├── mobile/privarcy_companion/ native Flutter app for Android and iOS
├── models/                   trained weights (git-ignored)
├── data/                     everything the app creates: outputs, jobs, databases (git-ignored)
├── docs/  scripts/
```

Data and model locations: `PRIVARCY_HOME` (data) and `PRIVARCY_MODELS` (weights) override the defaults.

## Honest limits

* The YOLO model has six classes (`id_card`, `document`, `screen`, `face`, `credit_card`, `license_plate`). It cannot detect
  "head", "full body" or "vehicle", so those are not offered.
* Telling people apart uses a baseline HOG matcher, not deep face recognition. Someone who leaves and returns may be listed
  twice. The design is safe-by-default: an unrecognised face is blurred in Auto mode. Details in [docs/FACE_MODES.md](docs/FACE_MODES.md).
* Magic Blur works on objects the detector knows. For anything else, draw a Manual Blur region.
* Camera support depends on the camera: USB/built-in via the OS, traditional phone/network cameras via RTSP or HTTP/MJPEG, and Android/iPhone phone cameras via the native PrivARcy Companion app and local WebRTC. The existing browser pairing page remains available as a fallback. The companion app pins the local CA fingerprint encoded in the scanned QR code; users do not install a certificate manually. There is no automatic network-camera discovery (ONVIF).
* Closing the desktop QR popup only hides it; the phone stream and redaction session continue. Use **Disconnect Phone** or **Stop** to end the session.
* See [docs/STATUS.md](docs/STATUS.md) for what has and has not been verified, and on what.
