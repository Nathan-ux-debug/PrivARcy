"""Data models shared across the frame_extraction package."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class FrameMeta:
    """Metadata + payload for a single extracted frame.

    Attributes:
        frame_index:   Index within the ORIGINAL (native) video stream.
        sample_index:  Index within the sampled/output sequence (0, 1, 2, ...).
        timestamp_sec: Timestamp of the frame in the source video, in seconds.
        frame:         Raw BGR NumPy array (H x W x 3), ready for a detector.
        source_video:  Path to the originating video file.
        output_path:   Path the frame was written to on disk, if saved.
    """

    frame_index: int
    sample_index: int
    timestamp_sec: float
    frame: np.ndarray = field(repr=False)
    source_video: str = ""
    output_path: Optional[str] = None
