"""A camera source and how to open it. Credentials live only here, in memory."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import cv2

from .security import mask_url, validate_stream_url, with_credentials

# Phone apps that turn a phone into a network camera. We do not bundle or control them; the phone and the PC must be on
# the same Wi-Fi network and the app must be running. (host, port, path) are the apps' documented defaults.
PHONE_PRESETS: Dict[str, dict] = {
    "ip_webcam": {"title": "IP Webcam (Android)", "scheme": "http", "port": 8080, "path": "/video",
                  "hint": "Start the server in the app, then enter the IP address it shows."},
    "droidcam": {"title": "DroidCam (Android / iPhone)", "scheme": "http", "port": 4747, "path": "/video",
                 "hint": "Open DroidCam on the phone and use the Wi-Fi IP address and port it shows."},
    "rtsp_app": {"title": "Other app with an RTSP stream", "scheme": "rtsp", "port": 8554, "path": "/",
                 "hint": "Copy the full rtsp:// address from the app instead of using this form."},
}


def build_phone_url(preset: str, host: str, port: Optional[int] = None) -> str:
    info = PHONE_PRESETS[preset]
    return validate_stream_url(f"{info['scheme']}://{host.strip()}:{port or info['port']}{info['path']}")


@dataclass
class CameraSource:
    kind: str                                   # "local" | "stream"
    name: str = ""
    index: int = 0                              # local devices
    url: str = ""                               # stream address WITHOUT credentials
    username: Optional[str] = field(default=None, repr=False)
    password: Optional[str] = field(default=None, repr=False)

    @classmethod
    def local(cls, index: int, name: str = "") -> "CameraSource":
        return cls("local", name or f"Camera {index}", index=index)

    @classmethod
    def stream(cls, url: str, username: str = "", password: str = "", name: str = "") -> "CameraSource":
        clean = validate_stream_url(url)
        return cls("stream", name or mask_url(clean), url=clean, username=username or None, password=password or None)

    @property
    def is_stream(self) -> bool:
        return self.kind == "stream"

    @property
    def display(self) -> str:
        return self.name or (mask_url(self.url) if self.is_stream else f"Camera {self.index}")

    def __repr__(self) -> str:                                       # never print credentials
        return f"CameraSource({self.kind}, {self.display!r})"

    __str__ = __repr__

    def target(self):
        """What to hand to cv2.VideoCapture. Contains the password for streams - do not log it."""
        return with_credentials(self.url, self.username, self.password) if self.is_stream else self.index


def open_capture(source: "CameraSource | int", opener: Optional[Callable] = None):
    """Open a capture for a source. ``opener`` is injectable for tests."""
    if isinstance(source, int):
        source = CameraSource.local(source)
    if opener is not None:
        return opener(source)
    if source.is_stream:
        # prefer TCP for RTSP (UDP drops frames on Wi-Fi) and never wait forever for a dead camera
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|stimeout;5000000")
        cap = cv2.VideoCapture(source.target(), cv2.CAP_FFMPEG)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        return cap
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else 0
    return cv2.VideoCapture(source.index, backend) if backend else cv2.VideoCapture(source.index)
