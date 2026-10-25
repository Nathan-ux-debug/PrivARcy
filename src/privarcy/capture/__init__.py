"""Camera sources: built-in/USB devices, phone cameras and network (IP) cameras, all through OpenCV."""

from .discovery import discover_local_cameras, local_ip_addresses
from .phone_pairing import PhoneFrameCapture, PhonePairingIdentity, PhonePairingService
from .probe import ProbeResult, probe_source
from .security import StreamURLError, mask_url, validate_stream_url
from .sources import PHONE_PRESETS, CameraSource, build_phone_url, open_capture

__all__ = ["CameraSource", "PHONE_PRESETS", "ProbeResult", "StreamURLError", "build_phone_url",
           "PhoneFrameCapture", "PhonePairingIdentity", "PhonePairingService", "discover_local_cameras",
           "local_ip_addresses", "mask_url", "open_capture", "probe_source",
           "validate_stream_url"]
