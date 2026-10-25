"""Background workers: nothing slow ever runs on the UI thread."""

from __future__ import annotations

import time
import traceback

from PyQt5.QtCore import QThread, pyqtSignal


class BackendLoader(QThread):
    """Builds BackendService (loads YOLO etc.) off the UI thread."""
    loaded = pyqtSignal(object)
    failed = pyqtSignal(str)

    def run(self) -> None:
        try:
            from privarcy.services.backend import BackendService
            self.loaded.emit(BackendService())
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc(limit=4)}")


class FuncWorker(QThread):
    """Run any callable in a thread; emits result or error text."""
    done = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, fn, *args, parent=None, **kwargs) -> None:
        super().__init__(parent)
        self._fn, self._args, self._kwargs = fn, args, kwargs

    def run(self) -> None:
        try:
            self.done.emit(self._fn(*self._args, **self._kwargs))
        except Exception as exc:
            self.error.emit(f"{exc}")


class ProcessingWorker(QThread):
    """Runs one video job. Progress is throttled so the UI thread is never flooded."""
    progress = pyqtSignal(dict)
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, service, video_path: str, parent=None, plan=None, face_policy=None) -> None:
        super().__init__(parent)
        self.service, self.video_path = service, video_path
        self.plan, self.face_policy = plan, face_policy
        self._state = {"current": 0, "total": 0, "detections": 0, "redactions": 0, "flagged": 0}
        self._last_emit = 0.0
        self._started = 0.0

    def _emit(self, force: bool = False) -> None:
        now = time.perf_counter()
        if not force and now - self._last_emit < 0.12:
            return
        self._last_emit = now
        elapsed = max(now - self._started, 1e-6)
        s = self._state
        fps = s["current"] / elapsed
        eta = max(0, s["total"] - s["current"]) / fps if fps > 0 else 0
        self.progress.emit({**s, "fps": fps, "eta": eta, "elapsed": elapsed})

    def run(self) -> None:
        self._started = time.perf_counter()
        st = self._state

        def on_frame(cur, total):
            st["current"], st["total"] = cur, total
            self._emit()

        def on_det(n): st["detections"] = n
        def on_flag(n): st["flagged"] = n
        def on_red(n): st["redactions"] = n

        try:
            result = self.service.run_job(self.video_path, on_frame=on_frame, on_detections=on_det,
                                          on_flagged=on_flag, on_redactions=on_red,
                                          plan=self.plan, face_policy=self.face_policy)
            self._emit(force=True)
            self.completed.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))

    def cancel(self) -> None:
        self.service.cancel_job()


class ScanWorker(QThread):
    """Lists the distinct people in a video (one quick pass) without blocking the UI."""
    progress = pyqtSignal(int, int)
    done = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, service, video_path: str, parent=None) -> None:
        super().__init__(parent)
        self.service, self.video_path = service, video_path

    def run(self) -> None:
        try:
            self.done.emit(self.service.scan_faces(self.video_path, on_progress=lambda d, t: self.progress.emit(d, t)))
        except Exception as exc:
            self.error.emit(str(exc))

    def cancel(self) -> None:
        self.service.cancel_scan()
