# Cameras

| Source | How | Needs |
|---|---|---|
| Built-in / USB | OS camera through OpenCV (DirectShow on Windows). Listed automatically; only devices that deliver a frame appear. | Camera not in use by another app; Windows camera privacy allowed for desktop apps. |
| Android/iPhone companion app | Scan the desktop QR with PrivARcy Companion, choose front or rear camera, and tap **Start Camera**. Native camera video streams directly through WebRTC into the existing live redaction pipeline. | Companion app installed; phone and PC on the same Wi-Fi; private-network firewall access. |
| Phone app | A phone app turns the phone into a network camera; PrivARcy reads its stream. Presets for IP Webcam (Android) and DroidCam give the usual address and port. | Phone and PC on the same Wi-Fi; the app running. |
| Network / IP camera | `rtsp://`, `rtsps://`, `http://`, `https://` streams (RTSP/H.264, HTTP/MJPEG). | Camera address, optional username and password. |

Not supported: automatic network-camera discovery (ONVIF), cameras that only work through their vendor's app/cloud,
and any `file://` or other scheme (refused on purpose).

## Credentials

* The address field refuses `user:pass@host`. Username and password are separate fields.
* They are merged into the URL only at the moment OpenCV opens the stream, are never shown, logged or saved, and every
  error message and label is passed through `mask_url`.

## Connecting

* Select **Built-in/USB** and press **Start** to use a laptop or USB webcam directly. This path does not start or require phone pairing.
* Select **Phone** and use **Connect Phone with QR** when using the native companion. Phone frames enter the same live redaction pipeline; changing to the local camera does not make phone frames replace the selected local source.
* **Test connection** opens the source in a worker with a timeout (8 s), reads one frame, reports size/fps and always releases it.
* Typical messages: wrong address/port, "needs a username and password", "connected but no frame", "no answer after N s".
* Streams reconnect automatically up to 3 times before the session ends; **Stop** always releases the device.
* The Live Camera preview grows with the window, fits the frame without stretching it, and keeps camera controls and redaction status alongside it.

## QR-paired Android or iPhone camera

The recommended QR flow uses the native Flutter companion in `mobile/privarcy_companion/`. Its camera uses the phone's
physical camera through native Android/iOS APIs; it does not capture the built-in Camera app or use a browser camera.
The desktop pairing service in `src/privarcy/capture/phone_pairing/` accepts the app's WebRTC offer and sends decoded
frames to the existing phone capture adapter and live redaction pipeline. The existing browser page remains a fallback.

1. Install Flutter (Dart 3.13 or later) and the platform toolchain. Start PrivARcy on Windows:

   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   .\scripts\setup.ps1
   .\scripts\run.ps1
   ```

   The setup installs the runtime dependencies from `pyproject.toml`, including the local WebRTC and QR packages.
2. Build and install the native app:

   ```powershell
   Set-Location mobile\privarcy_companion
   flutter pub get
   flutter run
   ```

   For a sideloadable Android release APK, run `flutter build apk --release`; the file is written to
   `build\app\outputs\flutter-apk\app-release.apk`. Install it with `adb install build\app\outputs\flutter-apk\app-release.apk`
   or copy it to the phone and open it. Android SDK/platform tools are required.

   For iOS, run `flutter build ios --release` on a Mac with Xcode and CocoaPods, then install through Xcode or TestFlight.
   A signed distribution build requires the appropriate Apple signing account. The generated project targets iOS 15 or later;
   Android's minimum is API 21 (Android 5.0).

3. Keep both devices on the same non-isolated Wi-Fi. In PrivARcy, open **Live Camera → Phone → Connect Phone with QR**.
   In the installed app, tap **Scan QR Code** and scan the code displayed on the PC.
4. In the app, select front or rear camera and tap **Start Camera**. Approve camera permission when asked. The app shows its
   local camera preview; the redacted video appears in PrivARcy's existing desktop preview and is saved when you stop the session.
5. Close the QR popup with its window close button to hide it without stopping the phone or redaction session. Click
   **Show Phone QR** to reopen the same pairing popup. Use **Stop Camera** in the app to stop the stream. The desktop **Stop**
   control also stops the current stream without deleting trusted phone credentials.

The QR contains the short-lived enrollment token, secure port, PC identity, and SHA-256 fingerprint of the PC's local CA.
The app downloads only that public CA over the local HTTP setup endpoint, checks its hash against the QR, then uses it as a
trust anchor for HTTPS/WSS. Certificate mismatch is rejected; the app does not bypass TLS certificate validation, and users
do not need to install a CA profile. WebRTC protects media with DTLS-SRTP. Enrollment tokens expire after five minutes and
are accepted only once; they are not written to disk. The CA private key and paired-phone credentials are protected with
the signed-in Windows user's DPAPI. The companion stores its PC credentials in Android Keystore or the iOS Keychain.
A separate HTTPS server key is generated per pairing and temporary files are deleted when the listener stops.

### Saved phone pairings

After a phone is enrolled once, its secure credential is stored on both devices. On later use, select the saved PrivARcy PC
in the companion and tap **Connect to PC**; the app discovers the PC on the local network with mDNS, authenticates with a
fresh challenge, and does not start the camera until **Start Camera** is tapped. Reconnection is bounded to five attempts.
The desktop lists trusted phones under **Trusted phones** and can revoke an individual phone; revocation invalidates its
credential immediately. The PC may trust multiple phones, but only one phone video stream can be active at a time.
Removing the companion's saved PC entry alone does not revoke the PC-side credential; use **Revoke** in the companion while
connected, or revoke the phone from the desktop.

### Network and transport details

* WebRTC carries video directly between the phone and PC. Signaling and the pairing page use HTTPS/WSS on the local network. The same-LAN path is configured without STUN or TURN and does not need public internet after dependencies are installed. mDNS discovery is local-network only.
* PrivARcy listens only on one discovered private IPv4 address and two temporary ports. It does not install firewall rules or open a public port. If Windows prompts, allow access only on **Private networks**; do not enable Public network access. A Wi-Fi guest network or client/AP isolation can block device-to-device traffic.
* The native app uses HTTP only to retrieve the public CA certificate, and validates its SHA-256 against the fingerprint embedded in the QR before trusting it. Only then does it use HTTPS/WSS with certificate validation. The desktop browser fallback still requires the HTTPS page and the browser's normal secure-context permission flow.

### Troubleshooting

* **Camera permission is missing or browser reports insecure context:** for the browser fallback, confirm the page is on `https://`, install and trust the CA, and verify its fingerprint. Never continue through a browser certificate warning. The native companion verifies the QR-pinned CA without manual certificate setup.
* **Phone cannot load the QR URL:** confirm both devices are on the same non-isolated Wi-Fi and disable guest/client isolation. A VPN or multiple network adapters can route the phone to a different interface; use the dialog's troubleshooting details and regenerate the QR after changing networks.
* **A saved PC cannot be discovered after changing Wi-Fi:** restart PrivARcy so its local listener and mDNS record bind to the current network address, then reconnect from the companion.
* **Windows Firewall blocks access:** allow PrivARcy on the private network only. The service uses temporary ports, so do not add a permanent public inbound rule.
* **App cannot connect or WebRTC does not connect:** confirm the phone and PC can reach each other on the same LAN and that private-network firewall access is allowed. The selected baseline has no TURN fallback; some isolated, routed, or restrictive Wi-Fi networks will not carry direct peer-to-peer media.
* **Permission denied or token expired:** allow camera access in phone settings and tap **Start Camera** again. A new QR is needed only for initial enrollment or after revocation; saved phones reconnect without scanning it again.
* **App is backgrounded or screen locks:** Android and iOS may suspend camera/network work. The companion intentionally stops streaming when backgrounded; reopen it, connect to the saved PC, and tap **Start Camera** again.

The automated suite exercises certificate validation, token/origin rejection, local HTTPS, WebRTC signaling, synthetic frame delivery,
QR certificate pins, and QR-popup lifecycle behavior. Companion QR parsing has Flutter unit tests. Real Android/iPhone installs,
camera permissions, camera switching, and physical-device streaming still require device testing.

## Live behaviour and safeguards

* Capture and inference run on separate threads and the newest frame wins. Dropped frames are counted and shown.
* If a frame fails to process it is replaced by black, never shown raw; after 30 failures in a row the session stops with a message.
* The recording is written from the redacted frames; the preview shows the same frames.
* Face choices work live: click a face in the video or its tile. New faces get temporary labels T-01, T-02, ...
* Live recordings, including QR-paired phone video, are written from redacted frames to the outputs folder when the session stops.
