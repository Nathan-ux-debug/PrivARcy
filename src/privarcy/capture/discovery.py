"""Finding cameras and this computer's address (for phone / network camera setup)."""

from __future__ import annotations

import importlib
import socket
import sys
from typing import Callable, List, Optional

import cv2


def discover_local_cameras(max_index: int = 5, opener: Optional[Callable[[int], object]] = None) -> List[dict]:
    """Built-in and USB cameras that open AND deliver a frame. Names come from DirectShow when pygrabber exists."""
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else 0
    names: list = []
    try:
        names = importlib.import_module("pygrabber.dshow_graph").FilterGraph().get_input_devices()
    except Exception:
        pass
    open_ = opener or (lambda i: cv2.VideoCapture(i, backend) if backend else cv2.VideoCapture(i))
    found = []
    for index in range(max_index):
        cap = None
        try:
            cap = open_(index)
            if cap is not None and cap.isOpened():
                ok, _ = cap.read()
                if ok:
                    found.append({"index": index, "label": names[index] if index < len(names) else f"Camera {index}"})
        except Exception:
            pass
        finally:
            if cap is not None:
                cap.release()
    return found


def local_ip_addresses() -> List[str]:
    """Private LAN addresses of this PC, so the user can check the phone is on the same network."""
    addresses = []
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))               # no packet is sent; it only selects the outgoing interface
            addresses.append(s.getsockname()[0])
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if ip not in addresses and not ip.startswith("127."):
                addresses.append(ip)
    except OSError:
        pass
    return addresses
