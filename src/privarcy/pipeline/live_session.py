"""Live camera session.

Two threads so the interface and the camera never wait on inference:

  capture thread   reads the camera as fast as it delivers and keeps ONLY the newest frame
  process thread   takes the newest frame, runs the SAME FrameEngine as recorded video, publishes the REDACTED frame

If processing is slower than the camera, the in-between frames are dropped (and counted) instead of queueing up and
making the picture lag further and further behind. The un-redacted frame never leaves these two threads: preview and
recording only ever receive redacted frames.

Safeguards
  * camera cannot open / is unplugged        -> state "error" with a plain message (stream cameras reconnect first)
  * a frame fails to process (model error)   -> that frame is replaced by a BLACK frame (fail closed); after
                                                 ``MAX_CONSECUTIVE_PROCESS_ERRORS`` in a row the session stops
  * dropped frames                           -> counted and reported in ``status()``
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable, List, Optional

import cv2
import numpy as np

from privarcy.capture.security import mask_url
from privarcy.capture.sources import CameraSource, open_capture
from .config import PipelineConfig
from .frame_engine import FrameEngine

logger = logging.getLogger(__name__)

MAX_CONSECUTIVE_READ_FAILURES = 45
MAX_CONSECUTIVE_PROCESS_ERRORS = 30


class LiveSession:
    def __init__(
        self,
        model,
        config: PipelineConfig,
        camera_index,
        width: int,
        height: int,
        fps: float,
        output_path: Optional[str | Path] = None,
        ocr_provider=None,
        capture_factory: Optional[Callable] = None,
        face_policy=None,
        plan=None,
        reconnect_attempts: int = 3,
    ) -> None:
        self.config = config.copy()
        self.source: CameraSource = camera_index if isinstance(camera_index, CameraSource) else CameraSource.local(int(camera_index))
        self._opener_arg = camera_index
        self._factory = capture_factory
        self.requested = (width, height, fps)
        self.actual = None
        self.state = "starting"            # starting | live | reconnecting | stopped | error
        self.error: Optional[str] = None
        self.output_path = Path(output_path) if output_path else None
        self.stats = {"frames": 0, "redacted": 0, "flagged": 0, "held": 0, "low_confidence": 0,
                      "dropped": 0, "masked": 0, "reconnects": 0}
        self.audit_log: deque = deque(maxlen=100)
        self.face_policy = face_policy
        self.reconnect_attempts = reconnect_attempts

        self._latest: Optional[np.ndarray] = None
        self._faces: List[dict] = []
        self._latest_lock = threading.Lock()
        self._slot = None
        self._seq = 0
        self._cond = threading.Condition()
        self._stop = threading.Event()
        self._proc_times: deque = deque(maxlen=60)
        self._frame_stamps: deque = deque(maxlen=60)
        self._latencies: deque = deque(maxlen=60)
        self._writer = None
        self._threads: List[threading.Thread] = []

        self.engine = FrameEngine(model, self.config, source_name=f"live-{self.source.display}", face_policy=face_policy,
                                  plan=plan, ocr_provider=ocr_provider, live=True)

        self._capture = self._open()
        if self._capture is None or not self._capture.isOpened():
            self._fail(self._open_error())
            return
        self._configure(self._capture, width, height, fps)
        self.actual = (
            int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or width,
            int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or height,
            float(self._capture.get(cv2.CAP_PROP_FPS) or fps),
        )
        if self.output_path is not None:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = cv2.VideoWriter(str(self.output_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                           min(self.actual[2], 30.0) or 15.0, (self.actual[0], self.actual[1]))
            if not self._writer.isOpened():
                self._capture.release()
                self._fail("Could not create the recording file.")
                return
        self.state = "live"
        for target, name in ((self._capture_loop, "privarcy-capture"), (self._process_loop, "privarcy-process")):
            thread = threading.Thread(target=target, name=name, daemon=True)
            thread.start()
            self._threads.append(thread)

    # ------------------------------------------------------------------ capture plumbing
    def _open(self):
        try:
            if self._factory is not None:
                return self._factory(self._opener_arg)
            return open_capture(self.source)
        except Exception as exc:
            logger.error("Opening camera failed: %s", mask_url(str(exc)))
            return None

    def _open_error(self) -> str:
        if self.source.is_stream:
            return (f"Could not open the camera stream {mask_url(self.source.url)}. Check the address, the username and "
                    f"password, and that the camera is reachable from this computer.")
        return ("Could not open the camera. It may be in use by another application, unplugged, or blocked "
                "by Windows camera privacy settings.")

    def _configure(self, cap, width, height, fps) -> None:
        if self.source.is_stream and self._factory is None:
            return                                           # network streams choose their own size and rate
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)

    def _fail(self, message: str) -> None:
        self.state, self.error = "error", message
        logger.error("Live session error: %s", mask_url(message))
        with self._cond:
            self._cond.notify_all()

    @property
    def is_live(self) -> bool:
        return self.state == "live"

    def _running(self) -> bool:
        return not self._stop.is_set() and self.state in ("live", "reconnecting")

    def _reconnect(self) -> bool:
        """Stream cameras drop on Wi-Fi: try again a few times before giving up."""
        self.state = "reconnecting"
        for attempt in range(1, self.reconnect_attempts + 1):
            if self._stop.is_set():
                return False
            try:
                self._capture.release()
            except Exception:
                pass
            time.sleep(min(2.0 * attempt, 6.0) if self._factory is None else 0.05)
            cap = self._open()
            if cap is not None and cap.isOpened():
                self._capture = cap
                self.stats["reconnects"] += 1
                self.state = "live"
                return True
        return False

    def _capture_loop(self) -> None:
        try:
            self._capture_frames()
        except Exception as exc:                          # a crashed camera thread must never leave a silent, frozen session
            logger.exception("Capture thread crashed")
            self._fail(f"The camera reader stopped unexpectedly: {mask_url(str(exc))}")

    def _capture_frames(self) -> None:
        failures = 0
        while self._running():
            ok, frame = self._capture.read()
            if not ok or frame is None:
                failures += 1
                if failures >= MAX_CONSECUTIVE_READ_FAILURES:
                    if self.source.is_stream and self._reconnect():
                        failures = 0
                        continue
                    self._fail("The camera stopped delivering frames (disconnected or taken by another app).")
                    break
                time.sleep(0.02)
                continue
            failures = 0
            with self._cond:
                self._seq += 1
                self._slot = (frame, time.perf_counter(), self._seq)
                self._cond.notify()

    # ------------------------------------------------------------------ processing
    def _process_loop(self) -> None:
        last_seq, frame_number, errors = 0, 0, 0
        interval = max(1, int(self.config.detection_interval))
        try:
            while self._running():
                with self._cond:
                    while (self._slot is None or self._slot[2] == last_seq) and self._running():
                        self._cond.wait(0.1)
                    if not self._running():
                        break
                    frame, captured_at, seq = self._slot
                if last_seq:
                    self.stats["dropped"] += max(0, seq - last_seq - 1)
                last_seq = seq
                frame_number += 1
                started = time.perf_counter()
                try:
                    out_frame, out = self._process(frame, frame_number, interval)
                    errors = 0
                except Exception as exc:
                    # Fail CLOSED: a frame we could not redact is replaced by black, never shown raw.
                    errors += 1
                    self.stats["masked"] += 1
                    logger.exception("Live frame processing failed (%d in a row)", errors)
                    out_frame, out = np.zeros_like(frame), None
                    if errors >= MAX_CONSECUTIVE_PROCESS_ERRORS:
                        self._fail(f"Processing failed repeatedly, so the preview stopped to avoid showing unredacted "
                                   f"video: {mask_url(str(exc))}")
                        break
                done = time.perf_counter()
                self._proc_times.append(done - started)
                self._latencies.append(done - captured_at)
                self._frame_stamps.append(done)
                self.stats["frames"] += 1
                if out is not None:
                    self._record_stats(out)
                with self._latest_lock:
                    self._latest = out_frame
                    if out is not None:
                        self._faces = [{"face_id": f.face_id, "label": f.label, "bbox": tuple(f.bbox), "blurred": f.blurred}
                                       for f in out.faces]
                if self._writer is not None:
                    self._writer.write(out_frame)
        finally:
            if self.state in ("live", "reconnecting"):
                self.state = "stopped"

    def _process(self, frame, frame_number, interval):
        out = self.engine.process(frame, frame_number, self.actual[2] if self.actual else 30.0,
                                  run_detection=(frame_number % interval == 1 or interval == 1))
        return out.frame, out

    def _record_stats(self, out) -> None:
        self.stats["redacted"] += out.redactions
        self.stats["flagged"] += out.flagged_new
        self.stats["held"] += out.held
        self.stats["low_confidence"] += sum(out.low_confidence.values())
        for record in out.records:
            if record["status"] in ("auto_redacted", "pending"):
                self.audit_log.append({"description": f"{record['label']} {record['status'].replace('_', ' ')} "
                                                      f"({record['confidence']:.0%})", "kind": record["label"].upper()})

    # ------------------------------------------------------------------ public API
    def latest_frame(self) -> Optional[np.ndarray]:
        """The most recent REDACTED frame (a copy), or None."""
        with self._latest_lock:
            return None if self._latest is None else self._latest.copy()

    def faces(self) -> List[dict]:
        """Faces in the latest frame: id, temporary label (e.g. T-01), box, and whether it is blurred."""
        with self._latest_lock:
            return [dict(f) for f in self._faces]

    def face_thumbnails(self) -> dict:
        if self.face_policy is None:
            return {}
        return {i.face_id: i.thumbnail for i in self.face_policy.book.identities() if i.thumbnail is not None}

    def set_face_mode(self, mode) -> None:
        if self.face_policy is not None:
            self.face_policy.set_mode(mode)

    def toggle_face(self, face_id: str) -> Optional[bool]:
        return self.face_policy.toggle(face_id) if self.face_policy is not None else None

    def status(self) -> dict:
        n = len(self._frame_stamps)
        fps = 0.0
        if n >= 2:
            span = self._frame_stamps[-1] - self._frame_stamps[0]
            fps = (n - 1) / span if span > 0 else 0.0
        return {
            "state": self.state, "error": self.error, "fps": fps,
            "latency_ms": (sum(self._latencies) / len(self._latencies) * 1000.0) if self._latencies else None,
            "inference_ms": (sum(self._proc_times) / len(self._proc_times) * 1000.0) if self._proc_times else None,
            "frames": self.stats["frames"], "resolution": self.actual, "dropped_frames": self.stats["dropped"],
            "masked_frames": self.stats["masked"], "reconnects": self.stats["reconnects"],
            "source": self.source.display, "recording": str(self.output_path) if self.output_path else None,
            "target_fps_met": bool(fps >= 15.0) if n >= 10 else None,
        }

    def stop(self) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        for thread in self._threads:
            if thread.is_alive() and thread is not threading.current_thread():
                thread.join(timeout=3.0)
        try:
            if getattr(self, "_capture", None) is not None:
                self._capture.release()
        finally:
            if self._writer is not None:
                self._writer.release()
                self._writer = None
            if self.state != "error":
                self.state = "stopped"
