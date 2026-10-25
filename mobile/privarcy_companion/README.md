# PrivARcy Companion

Native Flutter camera companion for PrivARcy desktop. It scans the desktop's pairing QR, shows a live phone-camera preview,
and streams directly to the PC over WebRTC. The redacted view remains in PrivARcy; the phone preview is the unredacted local
camera preview and is not presented as redacted output.

## Requirements

- Flutter with Dart 3.13.4 or later
- Android SDK/platform tools for Android builds (Android 5.0 / API 21 minimum)
- macOS with Xcode and CocoaPods for iOS builds (iOS 15 minimum)
- PrivARcy desktop and phone connected to the same non-isolated local Wi-Fi network

## Run or build

From this folder:

```powershell
flutter pub get
flutter run
```

Build Android APK:

```powershell
flutter build apk --release
adb install build\app\outputs\flutter-apk\app-release.apk
```

On macOS, build for iOS and install through Xcode:

```sh
flutter build ios --release
open ios/Runner.xcworkspace
```

An iOS physical-device build requires valid Apple signing credentials. Do not use a debug-signed Android APK for public
distribution.

## Pair and use

1. Start PrivARcy on the PC and open **Live Camera → Phone → Connect Phone with QR**.
2. In PrivARcy Companion, tap **Scan QR Code** and scan the PC's code.
3. Choose front or rear camera and tap **Start Camera**. Grant camera access when prompted.
4. The desktop preview receives frames through the existing detection/redaction pipeline. The phone displays only its own
   camera preview, not the PC's redacted output.
5. Stop from the app or the desktop. For a later session, choose the saved PC and tap **Connect to PC**, then tap
   **Start Camera**; QR scanning is only needed for initial enrollment or after the pairing is revoked.

The app downloads the public CA certificate from the private-LAN pairing endpoint and verifies its SHA-256 hash against the
value included in the QR before using that CA for HTTPS/WSS. A mismatch aborts pairing. The Android cleartext networking
permission is needed only for this public certificate bootstrap; video and signaling use TLS-protected WebSocket/WebRTC.
The QR is short-lived and single-use for enrollment. Phone credentials are stored in Android Keystore or the iOS Keychain;
the PC protects its credential store and CA private key with Windows DPAPI. Saved PCs are found over local mDNS and reconnect
using fresh challenge-response authentication. The app retries a limited number of times, and never starts the camera
automatically during reconnection. The PC can trust multiple phones, but only one video stream can be active at a time.
The desktop can revoke a phone, which invalidates its saved credential.

Keep the app in the foreground during streaming. Android and iOS can suspend camera capture and network activity when the
app is backgrounded or the screen is locked; the app stops and requires reconnection to the saved PC after returning.

## Network

Both devices must reach each other on the same LAN. Allow PrivARcy on Windows **Private networks only** if prompted. Do not
disable the firewall or allow the app on Public networks. Guest Wi-Fi, client isolation, VPNs, and some managed networks can
block direct WebRTC media; this setup has no cloud relay or TURN server.
