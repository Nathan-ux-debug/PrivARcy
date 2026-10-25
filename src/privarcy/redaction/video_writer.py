"""Thin wrapper around cv2.VideoWriter for writing a sequence of redacted
frames back out as a playable MP4.

Isolated so redaction logic (redactor.py) never has to know about video
containers/codecs, and so this piece could be swapped (e.g. a different
codec, or streaming output) without touching redaction logic.
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .exceptions import RedactionError

logger = logging.getLogger(__name__)


class VideoWriter:
    """Writes a sequence of same-sized BGR frames out to an MP4 file.

    Parameters:
        output_path: Destination .mp4 path. Parent folders are created
                      if needed.
        fps: Output frame rate. For a redacted video to play back at the
              correct real-world speed, this should match the *actual*
              sampling rate frames were extracted at (e.g.
              FrameExtractor.source_fps / FrameExtractor.frame_interval),
              not necessarily the source video's native fps.
        frame_size: (width, height) of every frame that will be written.
                     If omitted, it's inferred from the first frame passed
                     to write_frame().
        codec: FourCC codec string. "mp4v" is broadly compatible; "avc1"
                (H.264) gives smaller files but isn't available on every
                OpenCV build.
    """

    def __init__(
        self,
        output_path: str | Path,
        fps: float,
        frame_size: Optional[tuple[int, int]] = None,
        codec: str = "mp4v",
    ) -> None:
        if fps <= 0:
            raise RedactionError(f"fps must be > 0, got {fps}")

        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.frame_size = frame_size
        self.codec = codec
        self._writer: Optional[cv2.VideoWriter] = None
        self._frame_count = 0

    def write_frame(self, frame: np.ndarray) -> None:
        """Write one BGR frame. The writer is lazily opened on the first
        call, using that frame's dimensions if frame_size wasn't given."""
        if frame is None or frame.size == 0:
            raise RedactionError("write_frame() received an empty frame.")

        if self._writer is None:
            height, width = frame.shape[:2]
            self.frame_size = self.frame_size or (width, height)
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            self._writer = cv2.VideoWriter(str(self.output_path), fourcc, self.fps, self.frame_size)
            if not self._writer.isOpened():
                raise RedactionError(
                    f"Could not open VideoWriter for '{self.output_path}' "
                    f"(codec='{self.codec}') — try a different codec, e.g. 'avc1' or 'XVID'."
                )
            logger.info("Opened video writer: %s | fps=%.2f size=%s codec=%s",
                        self.output_path, self.fps, self.frame_size, self.codec)

        if (frame.shape[1], frame.shape[0]) != self.frame_size:
            frame = cv2.resize(frame, self.frame_size)

        self._writer.write(frame)
        self._frame_count += 1

    def release(self) -> None:
        """Finalize and close the output file. Safe to call even if no
        frames were ever written (writer was never opened)."""
        if self._writer is not None:
            self._writer.release()
            logger.info("Closed video writer: %s (%d frame(s) written).", self.output_path, self._frame_count)
        self._writer = None

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()
