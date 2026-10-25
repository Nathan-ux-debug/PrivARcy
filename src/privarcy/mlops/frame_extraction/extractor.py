"""Core frame-extraction logic: FrameExtractor."""

import logging
import queue
import threading
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np

from .exceptions import VideoValidationError, FrameReadError
from .models import FrameMeta
from .validation import validate_video, SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)

MAX_CONSECUTIVE_FAILURES = 25  # guards against a hung/corrupt tail mid-stream


class FrameExtractor:
    """Ingests a single video file and yields sampled frames as NumPy arrays.

    Parameters:
        video_path:    Path to the input video file (MP4 and other
                        OpenCV-readable containers).
        output_dir:    If provided, sampled frames are written here as
                        image files. If None, frames are only
                        yielded/queued, not persisted to disk.
        target_fps:    Desired sampling rate in frames-per-second. Must be
                        > 0; clamped to the source video's native FPS.
        image_format:  Output image extension without the dot ("jpg", "png").
        jpeg_quality:  Quality (0-100), used when image_format == "jpg".
        max_frames:    Optional hard cap on the number of sampled frames.
        strict:        If True, a single failed frame read raises
                        FrameReadError and stops extraction. If False
                        (default), the bad frame is skipped and logged.
    """

    SUPPORTED_EXTENSIONS = SUPPORTED_EXTENSIONS

    def __init__(
        self,
        video_path: str | Path,
        output_dir: Optional[str | Path] = None,
        target_fps: float = 1.0,
        image_format: str = "jpg",
        jpeg_quality: int = 95,
        max_frames: Optional[int] = None,
        strict: bool = False,
    ) -> None:
        if target_fps <= 0:
            raise ValueError(f"target_fps must be > 0, got {target_fps}")

        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir) if output_dir else None
        self.target_fps = target_fps
        self.image_format = image_format.lower().lstrip(".")
        self.jpeg_quality = jpeg_quality
        self.max_frames = max_frames
        self.strict = strict

        info = validate_video(self.video_path)
        self.source_fps = info.source_fps
        self.total_frames = info.total_frames
        self.duration_sec = info.duration_sec

        effective_fps = min(self.target_fps, self.source_fps)
        self.frame_interval = max(1, round(self.source_fps / effective_fps))

        logger.info(
            "Validated '%s' | source_fps=%.2f total_frames=%d duration=%.2fs "
            "| sampling every %d native frame(s) (~%.2f fps output)",
            self.video_path.name, self.source_fps, self.total_frames, self.duration_sec,
            self.frame_interval, self.source_fps / self.frame_interval,
        )

        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Extraction
    # ------------------------------------------------------------------ #
    def extract(self) -> Iterator[FrameMeta]:
        """Generator that yields a FrameMeta object for each sampled frame.

        Frames are read sequentially and kept if their native index is a
        multiple of self.frame_interval — more robust across containers
        than seeking with CAP_PROP_POS_FRAMES.
        """
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise VideoValidationError(f"Could not (re)open video for extraction: {self.video_path}")

        frame_index = 0
        sample_index = 0
        consecutive_failures = 0

        try:
            while True:
                ok, frame = cap.read()

                if not ok or frame is None:
                    pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
                    at_expected_end = self.total_frames and pos >= self.total_frames
                    if at_expected_end or consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        break

                    consecutive_failures += 1
                    msg = f"Failed to decode frame at index {frame_index} in {self.video_path.name}"
                    if self.strict:
                        raise FrameReadError(msg)
                    logger.warning(msg + " — skipping.")
                    frame_index += 1
                    continue

                consecutive_failures = 0

                if frame_index % self.frame_interval == 0:
                    timestamp_sec = frame_index / self.source_fps
                    output_path = None

                    if self.output_dir:
                        output_path = self._save_frame(frame, sample_index, timestamp_sec)

                    yield FrameMeta(
                        frame_index=frame_index,
                        sample_index=sample_index,
                        timestamp_sec=timestamp_sec,
                        frame=frame,
                        source_video=str(self.video_path),
                        output_path=str(output_path) if output_path else None,
                    )

                    sample_index += 1
                    if self.max_frames and sample_index >= self.max_frames:
                        logger.info("Reached max_frames=%d, stopping early.", self.max_frames)
                        break

                frame_index += 1
        finally:
            cap.release()

        logger.info(
            "Finished '%s': %d frame(s) sampled from %d native frame(s).",
            self.video_path.name, sample_index, frame_index,
        )

    def _save_frame(self, frame: np.ndarray, sample_index: int, timestamp_sec: float) -> Path:
        stem = self.video_path.stem
        filename = f"{stem}_frame{sample_index:06d}_t{timestamp_sec:.3f}s.{self.image_format}"
        out_path = self.output_dir / filename

        params = []
        if self.image_format in ("jpg", "jpeg"):
            params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]

        success = cv2.imwrite(str(out_path), frame, params)
        if not success:
            raise IOError(f"Failed to write frame to disk: {out_path}")
        return out_path

    # ------------------------------------------------------------------ #
    # Queue-based interface for downstream consumers (e.g. a YOLOv8 worker)
    # ------------------------------------------------------------------ #
    def extract_to_queue(
        self,
        output_queue: "queue.Queue[Optional[FrameMeta]]",
        sentinel: Optional[FrameMeta] = None,
    ) -> threading.Thread:
        """Run extraction in a background thread, pushing each FrameMeta
        onto output_queue as it's produced. Pushes `sentinel` (default:
        None) once done so a downstream consumer knows to stop blocking
        on get().

        Returns the started Thread so the caller can .join() it.
        """

        def _worker():
            try:
                for frame_meta in self.extract():
                    output_queue.put(frame_meta)
            except (VideoValidationError, FrameReadError) as exc:
                logger.error("Extraction thread failed for %s: %s", self.video_path, exc)
            finally:
                output_queue.put(sentinel)

        thread = threading.Thread(target=_worker, name=f"extract:{self.video_path.name}", daemon=True)
        thread.start()
        return thread
