"""Desktop QR-pairing dialog."""

from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QDialog, QHBoxLayout, QLabel, QLineEdit, QToolButton, QVBoxLayout

from privarcy.app.widgets import button, label

PAIRING_STATES = {
    "waiting_for_phone": ("Waiting for phone", "info", "A paired phone can reconnect, or scan the QR code to add another device."),
    "phone_connected": ("Phone connected", "info", "Tap Start Camera on the phone and allow camera access."),
    "permission_required": ("Camera permission required", "warn", "Allow camera access in the phone browser."),
    "camera_ready": ("Camera ready", "info", "Starting the direct WebRTC video connection…"),
    "streaming": ("Streaming", "ok", "Video frames are arriving and will be redacted locally."),
    "connection_lost": ("Connection lost", "danger", "The phone video connection ended. Generate a fresh QR code to reconnect."),
    "pairing_expired": ("Pairing expired", "warn", "Generate a new QR code; pairing links expire after five minutes."),
    "disconnected": ("Disconnected", "neutral", "The phone camera session has ended."),
    "connection_error": ("Connection error", "danger", "Check the phone browser, Wi-Fi, and Windows Firewall."),
}


class PhonePairingDialog(QDialog):
    regenerate = pyqtSignal()
    disconnect_requested = pyqtSignal()

    def __init__(self, pairing, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Connect Phone")
        self.setMinimumWidth(480)
        self._pairing = pairing
        self._closing_silently = False
        self._disconnect_sent = False
        layout = QVBoxLayout(self)
        layout.addWidget(label("Scan with PrivARcy Companion", "SectionTitle"))
        layout.addWidget(label(
            "Open PrivARcy Companion on your phone and scan this code. Choose front or rear camera and tap Start Camera; "
            "PrivARcy redacts video locally and does not record phone-paired sessions.",
            "Muted", wrap=True,
        ))
        self.qr = QLabel()
        self.qr.setAlignment(Qt.AlignCenter)
        self.qr.setFixedSize(260, 260)
        layout.addWidget(self.qr, alignment=Qt.AlignCenter)
        self.url = QLineEdit()
        self.url.setReadOnly(True)
        self.url.setVisible(False)
        layout.addWidget(self.url)
        self.fingerprint = label("", "Faint", wrap=True)
        self.fingerprint.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
        layout.addWidget(self.fingerprint)
        self.details_button = QToolButton()
        self.details_button.setText("Troubleshooting: show connection details")
        self.details_button.setCheckable(True)
        self.details_button.toggled.connect(self.url.setVisible)
        layout.addWidget(self.details_button)
        self.status = label("", "Muted", wrap=True)
        layout.addWidget(self.status)
        self.countdown = label("", "Faint")
        layout.addWidget(self.countdown)
        buttons = QHBoxLayout()
        self.regenerate_button = button("Generate fresh QR code", "", callback=self.regenerate.emit)
        self.disconnect_button = button("Disconnect Phone", "Danger", callback=self._request_disconnect)
        buttons.addWidget(self.regenerate_button)
        buttons.addWidget(self.disconnect_button)
        layout.addLayout(buttons)
        self.set_pairing(pairing)

    def set_pairing(self, pairing) -> None:
        self._pairing = pairing
        pixmap = QPixmap()
        if not pixmap.loadFromData(pairing.qr_png(), "PNG"):
            raise RuntimeError("Could not create the phone-pairing QR code.")
        self.qr.setPixmap(pixmap.scaled(250, 250))
        self.url.setText(getattr(pairing, "qr_url", pairing.url))
        self.fingerprint.setText(
            "Certificate SHA-256 (verify before first trust):\n" + pairing.fingerprint
        )
        self.details_button.setChecked(False)
        self._disconnect_sent = False
        self.update_status()

    def update_status(self, status: str | None = None) -> None:
        state = status or self._pairing.status
        text, _, detail = PAIRING_STATES.get(state, PAIRING_STATES["connection_error"])
        self.status.setText(f"{text} — {detail}")
        self.countdown.setText(
            f"Pairing expires in {self._pairing.seconds_remaining} seconds."
            if state not in ("streaming", "disconnected", "connection_lost", "connection_error", "pairing_expired")
            else ""
        )
        self.regenerate_button.setEnabled(True)
        self.disconnect_button.setEnabled(state not in ("disconnected", "pairing_expired"))

    def _request_disconnect(self) -> None:
        self._emit_disconnect()
        self.close_silently()

    def _emit_disconnect(self) -> None:
        if not self._disconnect_sent:
            self._disconnect_sent = True
            self.disconnect_requested.emit()

    def close_silently(self) -> None:
        self.hide()

    def closeEvent(self, event) -> None:
        super().closeEvent(event)
