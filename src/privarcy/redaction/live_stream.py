"""Threaded video capture for real-time use.

A plain `cv2.VideoCapture().read()` call blocks until a new frame is
ready. If inference (detection + redaction) takes longer than one frame
interval — likely on CPU, or with a heavier model — reading and
processing in the same loop causes growing latency: you fall further and
further behind live video.

ThreadedVideoCapture reads frames continuously in a background thread and
always hands back the MOST RECENT frame, silently dropping any frames the
main loop couldn't keep up with. This trades "process every single frame"
for "always process the current moment" — the right tradeoff for a live
feed, where a 2-second-old frame being redacted is worse than skipping it.
"""

import logging
import threading
from typing import Callable, Optional

import cv2
import numpy as np

from .exceptions import RedactionError

logger = logging.getLogger(__name__)


class ThreadedVideoCapture:
    """Continuously reads from a camera/stream in a background thread.

    Parameters:
        source: Camera index (0 = default webcam) or a stream URL
                 (RTSP/RTMP/HTTP), anything cv2.VideoCapture accepts.
        _capture_factory: Internal hook for dependency injection in tests —
                            defaults to `cv2.VideoCapture`. Not intended
                            for normal use.
    """

    def __init__(
        self,
        source: int | str = 0,
        _capture_factory: Optional[Callable[[int | str], object]] = None,
    ) -> None:
        factory = _capture_factory or cv2.VideoCapture
        self._cap = factory(source)
        if not self._cap.isOpened():
            raise RedactionError(f"Could not open video source: {source!r}")

        self.source = source
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._frames_read = 0

    def start(self) -> "ThreadedVideoCapture":
        self._running = True
        self._thread = threading.Thread(target=self._update_loop, daemon=True)
        self._thread.start()
        logger.info("Started capture thread for source=%r", self.source)
        return self

    def _update_loop(self) -> None:
        while self._running:
            ok, frame = self._cap.read()
            if not ok or frame is None:
                continue
            with self._lock:
                self._latest_frame = frame
                self._frames_read += 1

    def read(self) -> Optional[np.ndarray]:
        """Return a copy of the most recent frame, or None if nothing has
        arrived yet (e.g. called immediately after start())."""
        with self._lock:
            return None if self._latest_frame is None else self._latest_frame.copy()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._cap.release()
        logger.info("Stopped capture for source=%r (%d frame(s) read).", self.source, self._frames_read)

    def __enter__(self) -> "ThreadedVideoCapture":
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
