# Status

## Built and verified (automated tests)
* Face policy, identities, scanner, three modes - unit, engine and end-to-end on a real face photo.
* Manual regions (time windows, ellipse shapes) and Magic Blur (GrabCut shape, colour+position tracking).
* Instant preview: option changes re-render without calling the model (asserted by counting model calls).
* Camera layer: URL validation, credential masking, discovery, probing with timeouts, reconnect, dropped frames, fail-closed masking.
* Interface: navigation, hover menus, collapse, editor, live page, review/corrections/settings, narrow-window layout.

## Not verified here
* Your `best.pt` weights and OCR model (not available in the test environment).
* Physical USB/phone/IP cameras.
* Windows-specific behaviour (DirectShow names, high-DPI) - developed and tested on Linux with offscreen Qt.

## Known limits
* Head / full body / vehicle are not detector classes.
* Person re-identification is a baseline (see FACE_MODES.md).
* Magic Blur only for detector classes.
* No ONVIF discovery, no QR pairing.
