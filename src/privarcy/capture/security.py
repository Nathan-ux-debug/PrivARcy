"""Validation and credential hygiene for network camera URLs.

Rules
-----
* Only ``rtsp://``, ``rtsps://``, ``http://`` and ``https://`` are accepted (what OpenCV/FFmpeg can read for video).
  ``file://`` and anything else is refused so the camera box cannot be used to read local files.
* A password never travels inside a URL the user typed or that we display. Credentials are separate fields,
  held in memory only, merged into the URL at the last moment, and masked in every log line and error message.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple
from urllib.parse import quote, urlsplit, urlunsplit

ALLOWED_SCHEMES = ("rtsp", "rtsps", "http", "https")
_DEFAULT_PORTS = {"rtsp": 554, "rtsps": 322, "http": 80, "https": 443}
_HOST_RE = re.compile(r"^[A-Za-z0-9]([A-Za-z0-9.\-]*[A-Za-z0-9])?$")


class StreamURLError(ValueError):
    """The address cannot be used. The message is safe to show to the user."""


def validate_stream_url(url: str) -> str:
    """Return a normalised URL without credentials, or raise ``StreamURLError`` with a helpful message."""
    url = (url or "").strip()
    if not url:
        raise StreamURLError("Enter the stream address, for example rtsp://192.168.1.20:554/stream1")
    if "://" not in url:
        raise StreamURLError("The address must start with rtsp://, http:// or https://")
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        raise StreamURLError("That address is not valid (check the host and port).") from None
    scheme = parts.scheme.lower()
    if scheme not in ALLOWED_SCHEMES:
        raise StreamURLError(f"'{scheme}://' is not supported. Use rtsp://, http:// or https://")
    if parts.username or parts.password:
        raise StreamURLError("Do not put the username or password in the address. Use the Username and Password fields.")
    host = parts.hostname or ""
    if not host or not _HOST_RE.match(host.strip("[]")) and ":" not in host:
        raise StreamURLError("The address has no valid host name or IP.")
    if port is not None and not 1 <= port <= 65535:
        raise StreamURLError("The port must be between 1 and 65535.")
    netloc = host if ":" not in host else f"[{host}]"
    if port is not None:
        netloc += f":{port}"
    return urlunsplit((scheme, netloc, parts.path or "/", parts.query, ""))


def with_credentials(url: str, username: Optional[str], password: Optional[str]) -> str:
    """Merge credentials into a validated URL for the moment of opening the stream. Never log the result."""
    if not username:
        return url
    parts = urlsplit(url)
    userinfo = quote(username, safe="")
    if password:
        userinfo += ":" + quote(password, safe="")
    return urlunsplit((parts.scheme, f"{userinfo}@{parts.netloc}", parts.path, parts.query, ""))


def mask_url(text: str) -> str:
    """Replace any ``user:pass@`` in text with ``***@`` (for logs, errors, UI labels)."""
    return re.sub(r"(?P<scheme>[a-zA-Z][a-zA-Z0-9+.\-]*://)[^/@\s]+@", r"\g<scheme>***@", text or "")


def split_host_port(url: str) -> Tuple[str, int]:
    parts = urlsplit(url)
    return parts.hostname or "", parts.port or _DEFAULT_PORTS.get(parts.scheme.lower(), 0)
