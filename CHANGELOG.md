# Changelog

## 2.0.0
**Restructure**
* Single installable package `privarcy` in a `src/` layout (was separate `Backend/` and `Frontend/` folders with path hacks).
* Runtime data moved out of the code tree into one data folder (`data/` or `PRIVARCY_HOME`); models in `models/`.
* Tests moved to `tests/` mirroring the package; `pyproject.toml` replaces ad-hoc setup text.

**Face selection**
* New three-mode face policy (Auto / Selected only / Every face) with per-person tracking, shared by recorded video, live camera and the editor preview.
* One quick scan lists the distinct people in a video; choices persist for the whole video.

**Editor**
* Auto Blur class chips, Manual Blur (regions with time range) and Magic Blur (click an object, follows it) with an instant preview that never re-runs the model.
* Blur strength setting.

**Cameras**
* Camera sources: built-in/USB, phone stream apps, RTSP/HTTP network cameras, with URL validation, credentials kept out of URLs/logs, connection test and timeouts.
* Live session rewritten: capture and inference on separate threads, newest-frame-wins (dropped frames counted), automatic reconnect for streams, fail-closed black frames on model errors.

**Interface**
* Bigger sidebar with icons, status pill, collapse, and hover menus (App Settings, profile).
* Dashboard offers Offline / Realtime.

**Removed**
* Face Registry (consent enrolment), its database, page and API, the registered-face policy, and the unused FastAPI data-store API.
