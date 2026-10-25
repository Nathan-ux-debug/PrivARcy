"""User edits on top of automatic detection: hand-drawn regions and clicked objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class ManualRegion:
    """A region the user drew. Coordinates are 0..1 of the frame so the plan survives any resolution."""
    x1: float
    y1: float
    x2: float
    y2: float
    shape: str = "rect"                 # rect | ellipse
    start: float = 0.0                  # seconds
    end: Optional[float] = None         # seconds; None = until the end
    label: str = "Region"

    def active(self, seconds: float) -> bool:
        return seconds >= self.start and (self.end is None or seconds <= self.end)

    def box(self, width: int, height: int) -> Tuple[int, int, int, int]:
        x1, x2 = sorted((self.x1, self.x2))
        y1, y2 = sorted((self.y1, self.y2))
        return (max(0, int(x1 * width)), max(0, int(y1 * height)), min(width, int(x2 * width)), min(height, int(y2 * height)))


@dataclass
class MagicTarget:
    """An object the user clicked. It is followed through the video by class + appearance + position."""
    name: str                           # detector class, e.g. "screen"
    box: Tuple[int, int, int, int]      # where it was when clicked (pixels)
    signature: Optional[object] = None  # HSV histogram of the clicked object (numpy array)
    label: str = ""


@dataclass
class EditPlan:
    manual: List[ManualRegion] = field(default_factory=list)
    magic: List[MagicTarget] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not self.manual and not self.magic
