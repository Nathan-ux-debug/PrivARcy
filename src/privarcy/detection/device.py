"""Device selection: use CUDA if available, fall back to CPU.

Isolated so it can be unit-tested without a GPU present, and so
`detector.py` doesn't need to know *how* torch checks CUDA availability.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def resolve_device(requested: Optional[str] = None) -> str:
    """Return the device string to run inference on.

    Args:
        requested: If given ("cuda", "cpu", "cuda:0", ...), that value is
                   returned as-is — an explicit request is never overridden.
                   If None, CUDA is used when available, otherwise CPU.

    Returns:
        A device string suitable for Ultralytics' `model(..., device=...)`.

    Notes:
        This never raises: if the torch import itself fails (e.g. torch is
        not installed), it safely falls back to "cpu" rather than crashing
        the whole pipeline over a device-selection check.
    """
    if requested:
        return requested

    try:
        import torch

        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            logger.info("CUDA GPU detected: %s — using device='cuda'.", device_name)
            return "cuda"

        logger.info("No CUDA GPU detected — falling back to device='cpu'.")
        return "cpu"

    except ImportError:
        logger.warning("torch not importable — defaulting to device='cpu'.")
        return "cpu"
