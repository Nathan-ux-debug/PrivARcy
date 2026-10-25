"""Try a source without starting a session: opens it, reads one frame, reports size/fps, always releases it."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable, Optional

import cv2

from .security import mask_url, split_host_port
from .sources import CameraSource, open_capture


@dataclass
class ProbeResult:
    ok: bool
    message: str
    width: int = 0
    height: int = 0
    fps: float = 0.0


def probe_source(source: CameraSource, timeout: float = 8.0, opener: Optional[Callable] = None) -> ProbeResult:
    box: dict = {}

    def work() -> None:
        cap = None
        try:
            cap = open_capture(source, opener)
            if cap is None or not cap.isOpened():
                box["result"] = ProbeResult(False, _open_failure(source))
                return
            ok, frame = cap.read()
            if not ok or frame is None:
                box["result"] = ProbeResult(False, "Connected, but no video frame arrived. The camera may need a different "
                                                   "stream path, or someone else is already using it.")
                return
            h, w = frame.shape[:2]
            box["result"] = ProbeResult(True, f"Connected: {w}x{h}", w, h, float(cap.get(cv2.CAP_PROP_FPS) or 0.0))
        except Exception as exc:                                    # message must never contain the password
            box["result"] = ProbeResult(False, mask_url(f"Could not connect: {exc}"))
        finally:
            if cap is not None:
                cap.release()

    thread = threading.Thread(target=work, name="privarcy-probe", daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        return ProbeResult(False, f"No answer after {timeout:.0f} s. Check the address, that the camera is on and on this network, "
                                  f"and that a firewall is not blocking it.")
    return box["result"]


def _open_failure(source: CameraSource) -> str:
    if source.is_stream:
        host, port = split_host_port(source.url)
        auth = " Check the username and password." if source.username else " It may need a username and password."
        return f"Could not open {mask_url(source.url)}. Check the address and port ({host}:{port}).{auth}"
    return ("Could not open the camera. It may be in use by another application, unplugged, or blocked by "
            "Windows camera privacy settings.")
