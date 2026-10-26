"""Local HTTPS/WebRTC pairing for browser-based phone cameras."""

from __future__ import annotations

import asyncio
import base64
import ctypes
import datetime
import hashlib
import hmac
import ipaddress
import json
import logging
import os
import re
import secrets
import socket
import ssl
import tempfile
import threading
import time
import uuid
from ctypes import wintypes
from pathlib import Path
from queue import Empty, Full, Queue
from urllib.parse import urlencode

import cv2
import numpy as np
import qrcode
from aiohttp import WSMsgType, web
from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamError
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID
from cryptography.hazmat.primitives.serialization import load_der_private_key
from zeroconf import ServiceInfo, Zeroconf

from ..discovery import local_ip_addresses

logger = logging.getLogger(__name__)
PAIRING_TTL_SECONDS = 300
MAX_FRAME_PIXELS = 3840 * 2160
FRAME_STALE_SECONDS = 3
MAX_PAIRED_DEVICES = 32
DEVICE_ID_PATTERN = re.compile(r"^[a-f0-9]{32}$")
DEVICE_SECRET_BYTES = 32


class PhoneFrameCapture:
    """OpenCV-style, latest-frame-only capture adapter for the live pipeline."""

    def __init__(self) -> None:
        self._frames: Queue[np.ndarray] = Queue(maxsize=1)
        self._lock = threading.Lock()
        self._released = False
        self._shape: tuple[int, int] | None = None

    def put(self, frame: np.ndarray) -> None:
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError("The phone sent a video frame with an unsupported format.")
        if frame.shape[0] * frame.shape[1] > MAX_FRAME_PIXELS:
            raise ValueError("The phone video resolution is too large. Choose a lower camera resolution.")
        with self._lock:
            if self._released:
                return
            self._shape = (frame.shape[1], frame.shape[0])
        try:
            self._frames.put_nowait(frame)
        except Full:
            try:
                self._frames.get_nowait()
            except Empty:
                pass
            self._frames.put_nowait(frame)

    def isOpened(self) -> bool:
        with self._lock:
            return not self._released

    def read(self) -> tuple[bool, np.ndarray | None]:
        with self._lock:
            if self._released:
                return False, None
        try:
            return True, self._frames.get(timeout=0.5)
        except Empty:
            return False, None

    def get(self, property_id: int) -> float:
        with self._lock:
            shape = self._shape
        if shape is None:
            return 0.0
        if property_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(shape[0])
        if property_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(shape[1])
        if property_id == cv2.CAP_PROP_FPS:
            return 30.0
        return 0.0

    def set(self, _property_id: int, _value: float) -> bool:
        return True

    def release(self) -> None:
        with self._lock:
            self._released = True


class PhonePairingIdentity:
    """Trusted local CA, with its Windows private key protected by the current user's DPAPI."""

    def __init__(self, key=None, certificate: x509.Certificate | None = None) -> None:
        if key is not None and certificate is not None:
            self._key = key
            self.certificate = certificate
            self.certificate_pem = certificate.public_bytes(serialization.Encoding.PEM)
            self.fingerprint = certificate.fingerprint(hashes.SHA256()).hex(":").upper()
            return
        now = datetime.datetime.now(datetime.timezone.utc)
        self._key = ec.generate_private_key(ec.SECP256R1())
        name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "PrivARcy Local Phone Camera CA")])
        self.certificate = (
            x509.CertificateBuilder()
            .subject_name(name)
            .issuer_name(name)
            .public_key(self._key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(minutes=1))
            .not_valid_after(now + datetime.timedelta(days=3650))
            .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
            .add_extension(x509.SubjectKeyIdentifier.from_public_key(self._key.public_key()), critical=False)
            .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(self._key.public_key()), critical=False)
            .add_extension(
                x509.KeyUsage(
                    digital_signature=True, content_commitment=False, key_encipherment=False, data_encipherment=False,
                    key_agreement=False, key_cert_sign=True, crl_sign=True, encipher_only=None, decipher_only=None,
                ),
                critical=True,
            )
            .sign(self._key, hashes.SHA256())
        )
        self.certificate_pem = self.certificate.public_bytes(serialization.Encoding.PEM)
        self.fingerprint = self.certificate.fingerprint(hashes.SHA256()).hex(":").upper()

    @classmethod
    def load_or_create(cls, directory: Path) -> "PhonePairingIdentity":
        if os.name != "nt":
            return cls()
        directory.mkdir(parents=True, exist_ok=True)
        key_path, cert_path = directory / "phone-ca.dpapi", directory / "phone-ca.crt"
        if key_path.exists() != cert_path.exists():
            raise RuntimeError(
                f"Phone-camera trust files are incomplete in {directory}. Remove that folder to create a new certificate."
            )
        if key_path.exists():
            encrypted_key = key_path.read_bytes()
            key_data = cls._dpapi_unprotect(encrypted_key)
            key = load_der_private_key(key_data, password=None)
            certificate = x509.load_pem_x509_certificate(cert_path.read_bytes())
            if certificate.not_valid_after_utc <= datetime.datetime.now(datetime.timezone.utc):
                raise RuntimeError("The saved phone-camera certificate has expired. Remove it and pair the phone again.")
            if key.public_key().public_numbers() != certificate.public_key().public_numbers():
                raise RuntimeError("The saved phone-camera certificate does not match its protected private key.")
            return cls(key, certificate)

        identity = cls()
        private_key = identity._key.private_bytes(
            serialization.Encoding.DER, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
        )
        encrypted_key = cls._dpapi_protect(private_key)
        key_tmp = key_path.with_name(f"{key_path.name}.tmp")
        cert_tmp = cert_path.with_name(f"{cert_path.name}.tmp")
        key_tmp.write_bytes(encrypted_key)
        cert_tmp.write_bytes(identity.certificate_pem)
        os.replace(key_tmp, key_path)
        os.replace(cert_tmp, cert_path)
        return identity

    @property
    def device_id(self) -> str:
        return self.certificate.fingerprint(hashes.SHA256()).hex()[:32]

    @property
    def mdns_hostname(self) -> str:
        return f"privarcy-{self.device_id}.local"

    @staticmethod
    def _dpapi_transform(data: bytes, protect: bool) -> bytes:
        if os.name != "nt":
            raise RuntimeError("Windows DPAPI is required to persist the phone-camera certificate.")

        class DataBlob(ctypes.Structure):
            _fields_ = [("size", wintypes.DWORD), ("data", ctypes.POINTER(ctypes.c_ubyte))]

        source_buffer = ctypes.create_string_buffer(data)
        source = DataBlob(len(data), ctypes.cast(source_buffer, ctypes.POINTER(ctypes.c_ubyte)))
        destination = DataBlob()
        crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.LocalFree.argtypes = [ctypes.c_void_p]
        kernel32.LocalFree.restype = ctypes.c_void_p
        if protect:
            transform = crypt32.CryptProtectData
            transform.argtypes = [
                ctypes.POINTER(DataBlob), wintypes.LPCWSTR, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(DataBlob),
            ]
            transform.restype = wintypes.BOOL
            succeeded = transform(ctypes.byref(source), "PrivARcy phone camera CA", None, None, None, 1,
                                  ctypes.byref(destination))
        else:
            transform = crypt32.CryptUnprotectData
            transform.argtypes = [
                ctypes.POINTER(DataBlob), ctypes.POINTER(wintypes.LPWSTR), ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(DataBlob),
            ]
            transform.restype = wintypes.BOOL
            description = wintypes.LPWSTR()
            succeeded = transform(ctypes.byref(source), ctypes.byref(description), None, None, None, 1,
                                  ctypes.byref(destination))
            if description:
                kernel32.LocalFree(ctypes.cast(description, ctypes.c_void_p))
        if not succeeded:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            return ctypes.string_at(destination.data, destination.size)
        finally:
            kernel32.LocalFree(destination.data)

    @classmethod
    def _dpapi_protect(cls, data: bytes) -> bytes:
        return cls._dpapi_transform(data, protect=True)

    @classmethod
    def _dpapi_unprotect(cls, data: bytes) -> bytes:
        return cls._dpapi_transform(data, protect=False)

    def server_context(
        self, addresses: tuple[str, ...], directory: str, mdns_hostname: str | None = None
    ) -> ssl.SSLContext:
        now = datetime.datetime.now(datetime.timezone.utc)
        server_key = ec.generate_private_key(ec.SECP256R1())
        server_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "PrivARcy local pairing")])
        server_cert = (
            x509.CertificateBuilder()
            .subject_name(server_name)
            .issuer_name(self.certificate.subject)
            .public_key(server_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(minutes=1))
            .not_valid_after(now + datetime.timedelta(days=2))
            .add_extension(
                x509.SubjectAlternativeName(
                    [x509.IPAddress(ipaddress.ip_address(a)) for a in addresses]
                    + ([x509.DNSName(mdns_hostname)] if mdns_hostname else [])
                ),
                critical=False,
            )
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
            .add_extension(
                x509.KeyUsage(
                    digital_signature=True, content_commitment=False, key_encipherment=False, data_encipherment=False,
                    key_agreement=False, key_cert_sign=False, crl_sign=False, encipher_only=None, decipher_only=None,
                ),
                critical=True,
            )
            .add_extension(x509.SubjectKeyIdentifier.from_public_key(server_key.public_key()), critical=False)
            .add_extension(
                x509.AuthorityKeyIdentifier.from_issuer_public_key(self._key.public_key()), critical=False
            )
            .sign(self._key, hashes.SHA256())
        )

        temp_path = Path(directory)
        cert_path, key_path = temp_path / "server.crt", temp_path / "server.key"
        cert_path.write_bytes(server_cert.public_bytes(serialization.Encoding.PEM))
        key_path.write_bytes(server_key.private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
        ))
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.load_cert_chain(str(cert_path), str(key_path))
        return context


class PhoneDeviceStore:
    """Per-installation phone credentials protected by the signed-in Windows user's DPAPI."""

    def __init__(self, directory: Path, pc_id: str) -> None:
        if os.name != "nt":
            raise RuntimeError("Persistent phone pairing requires Windows user credential protection.")
        self.directory = Path(directory)
        if not DEVICE_ID_PATTERN.fullmatch(pc_id):
            raise ValueError("The PC identity for persistent phone pairing is invalid.")
        self.pc_id = pc_id
        self.path = self.directory / "paired-devices.dpapi"
        self._lock = threading.RLock()
        self._devices: dict[str, dict[str, str]] = {}
        if self.path.exists():
            try:
                content = PhonePairingIdentity._dpapi_unprotect(self.path.read_bytes())
                decoded = json.loads(content.decode("utf-8"))
                if (
                    not isinstance(decoded, dict)
                    or decoded.get("version") != 1
                    or decoded.get("pc_id") != self.pc_id
                ):
                    raise ValueError("Unsupported paired-device data.")
                devices = decoded.get("devices")
                if not isinstance(devices, dict):
                    raise ValueError("Invalid paired-device data.")
                self._devices = self._validate_devices(devices)
            except (OSError, ValueError, UnicodeError) as exc:
                raise RuntimeError("The saved phone-pairing credentials could not be read.") from exc

    @staticmethod
    def _validate_devices(devices: dict) -> dict[str, dict[str, str]]:
        if len(devices) > MAX_PAIRED_DEVICES:
            raise ValueError("The saved phone-pairing store exceeds its device limit.")
        clean: dict[str, dict[str, str]] = {}
        for device_id, record in devices.items():
            if not isinstance(device_id, str) or not DEVICE_ID_PATTERN.fullmatch(device_id):
                raise ValueError("The saved phone-pairing store contains an invalid device ID.")
            if not isinstance(record, dict):
                raise ValueError("The saved phone-pairing store contains an invalid device record.")
            name, secret = record.get("name"), record.get("secret")
            created, last_connected = record.get("created_at"), record.get("last_connected", "")
            if (
                not isinstance(name, str) or not name or len(name) > 64
                or not isinstance(secret, str)
                or not re.fullmatch(r"[A-Za-z0-9_-]{43}=", secret)
                or not isinstance(created, str)
                or not isinstance(last_connected, str)
            ):
                raise ValueError("The saved phone-pairing store contains invalid device fields.")
            try:
                key = base64.urlsafe_b64decode(secret.encode("ascii"))
            except (ValueError, UnicodeError) as exc:
                raise ValueError("The saved phone-pairing store contains an invalid device key.") from exc
            if len(key) != DEVICE_SECRET_BYTES:
                raise ValueError("The saved phone-pairing store contains an invalid device key.")
            clean[device_id] = {
                "name": name,
                "secret": secret,
                "created_at": created,
                "last_connected": last_connected,
            }
        return clean

    def list_devices(self) -> list[dict[str, str]]:
        with self._lock:
            return [
                {"device_id": device_id, **{key: value for key, value in row.items() if key != "secret"}}
                for device_id, row in sorted(self._devices.items(), key=lambda item: item[1]["name"].casefold())
            ]

    def secret_for(self, device_id: str) -> bytes | None:
        with self._lock:
            row = self._devices.get(device_id)
            return base64.urlsafe_b64decode(row["secret"].encode("ascii")) if row else None

    def add_device(self, device_id: str, name: str, secret: bytes) -> None:
        if not DEVICE_ID_PATTERN.fullmatch(device_id):
            raise ValueError("The phone supplied an invalid device identity.")
        if not isinstance(name, str) or not isinstance(secret, bytes):
            raise ValueError("The phone supplied invalid pairing credentials.")
        clean_name = " ".join(name.split())[:64]
        if not clean_name or len(secret) != DEVICE_SECRET_BYTES:
            raise ValueError("The phone supplied invalid pairing credentials.")
        with self._lock:
            if device_id in self._devices:
                raise ValueError("This phone is already paired. Revoke it before pairing again.")
            if len(self._devices) >= MAX_PAIRED_DEVICES:
                raise ValueError(f"PrivARcy supports at most {MAX_PAIRED_DEVICES} paired phones.")
            self._devices[device_id] = {
                "name": clean_name,
                "secret": base64.urlsafe_b64encode(secret).decode("ascii"),
                "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "last_connected": "",
            }
            try:
                self._save()
            except Exception:
                del self._devices[device_id]
                raise

    def mark_connected(self, device_id: str) -> None:
        with self._lock:
            row = self._devices.get(device_id)
            if row is not None:
                previous = row["last_connected"]
                row["last_connected"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
                try:
                    self._save()
                except Exception:
                    row["last_connected"] = previous
                    raise

    def revoke(self, device_id: str) -> bool:
        with self._lock:
            removed = self._devices.pop(device_id, None)
            if removed is None:
                return False
            try:
                self._save()
            except Exception:
                self._devices[device_id] = removed
                raise
            return True

    def _save(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {"version": 1, "pc_id": self.pc_id, "devices": self._devices},
            separators=(",", ":"),
        ).encode("utf-8")
        temporary = self.path.with_name(f"{self.path.name}.tmp")
        try:
            temporary.write_bytes(PhonePairingIdentity._dpapi_protect(payload))
            os.replace(temporary, self.path)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise


class PhonePairingService:
    """One temporary local service with a single-use token and browser-trusted TLS identity."""

    def __init__(
        self,
        ttl_seconds: int = PAIRING_TTL_SECONDS,
        identity: PhonePairingIdentity | None = None,
        device_store: PhoneDeviceStore | None = None,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("Pairing lifetime must be positive.")
        addresses = []
        for value in local_ip_addresses():
            try:
                ip = ipaddress.ip_address(value)
            except ValueError:
                continue
            if ip.version == 4 and ip.is_private and not ip.is_loopback and value not in addresses:
                addresses.append(value)
        if not addresses:
            raise RuntimeError("No usable local IPv4 address was found. Connect this PC to the same Wi-Fi as the phone.")

        self.address = addresses[0]
        self.addresses = (self.address,)
        self.token = secrets.token_urlsafe(32)
        self.expires_at = time.monotonic() + ttl_seconds
        self.capture = PhoneFrameCapture()
        self._status = "waiting_for_phone"
        self._status_lock = threading.Lock()
        self._frame_seen = False
        self._last_frame_at: float | None = None
        self._used = False
        self._stopped = threading.Event()
        self._ready = threading.Event()
        self._startup_error: Exception | None = None
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._runner: web.AppRunner | None = None
        self._zeroconf: Zeroconf | None = None
        self._service_info: ServiceInfo | None = None
        self._service_registered = False
        self._connections: set[RTCPeerConnection] = set()
        self._websockets: set[web.WebSocketResponse] = set()
        self._device_websockets: dict[str, web.WebSocketResponse] = {}
        self._claimed = False
        self._active_device_ids: set[str] = set()
        self._stream_claimed = False
        self._temp_dir = tempfile.TemporaryDirectory(prefix="privarcy-phone-")
        self.identity = identity or PhonePairingIdentity()
        self.device_store = device_store
        if device_store is not None and device_store.pc_id != self.identity.device_id:
            raise ValueError("Paired phone credentials do not belong to this PC identity.")
        self.device_id = self.identity.device_id
        self.mdns_hostname = self.identity.mdns_hostname
        self._ca_pem = self.identity.certificate_pem
        self.fingerprint = self.identity.fingerprint
        tls_context = self.identity.server_context(self.addresses, self._temp_dir.name, self.mdns_hostname)
        self._http_socket = self._listen_socket(self.address)
        self._https_socket = self._listen_socket(self.address)
        self.http_port = self._http_socket.getsockname()[1]
        self.https_port = self._https_socket.getsockname()[1]
        self.url = f"http://{self.address}:{self.http_port}/pair/{self.token}"
        qr_query = urlencode({
            "ca_sha256": self.fingerprint.replace(":", "").lower(),
            "https_port": self.https_port,
            "pc_id": self.device_id,
            "mdns_host": self.mdns_hostname,
        })
        self.qr_url = f"{self.url}?{qr_query}"
        self.secure_url = f"https://{self.address}:{self.https_port}/pair/{self.token}"
        self._tls_context = tls_context
        self._allowed_hosts = {
            f"{address}:{port}" for address in self.addresses for port in (self.http_port, self.https_port)
        }
        self._allowed_hosts.update(
            f"{self.mdns_hostname}:{port}" for port in (self.http_port, self.https_port)
        )

    @staticmethod
    def _listen_socket(address: str) -> socket.socket:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((address, 0))
        sock.listen(16)
        sock.setblocking(False)
        return sock

    def qr_png(self) -> bytes:
        code = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_M, box_size=8, border=4)
        code.add_data(self.qr_url)
        code.make(fit=True)
        image = code.make_image(fill_color="black", back_color="white")
        from io import BytesIO

        output = BytesIO()
        image.save(output, format="PNG")
        return output.getvalue()

    def renew_qr(self) -> None:
        with self._status_lock:
            self.token = secrets.token_urlsafe(32)
            self.expires_at = time.monotonic() + PAIRING_TTL_SECONDS
            self._used = False
            self._claimed = False
            if self._status != "streaming":
                self._status = "waiting_for_phone"
        self.url = f"http://{self.address}:{self.http_port}/pair/{self.token}"
        query = urlencode({
            "ca_sha256": self.fingerprint.replace(":", "").lower(),
            "https_port": self.https_port,
            "pc_id": self.device_id,
            "mdns_host": self.mdns_hostname,
        })
        self.qr_url = f"{self.url}?{query}"
        self.secure_url = f"https://{self.address}:{self.https_port}/pair/{self.token}"

    def start(self) -> PhonePairingService:
        self._thread = threading.Thread(target=self._run_loop, name="privarcy-phone-pairing", daemon=True)
        self._thread.start()
        if not self._ready.wait(10):
            self.stop()
            raise RuntimeError("The local phone-pairing server did not start in time.")
        if self._startup_error is not None:
            error = self._startup_error
            self.stop()
            raise RuntimeError(f"Could not start the local phone-pairing server: {error}") from error
        return self

    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._start_servers())
            self._ready.set()
            loop.run_forever()
        except (OSError, RuntimeError, ValueError) as exc:
            self._startup_error = exc
            self._ready.set()
        finally:
            if self._runner is not None:
                loop.run_until_complete(self._runner.cleanup())
            loop.run_until_complete(asyncio.to_thread(self._close_discovery))
            loop.close()

    async def _start_servers(self) -> None:
        self._runner = web.AppRunner(self._make_app(), access_log=None)
        await self._runner.setup()
        await web.SockSite(self._runner, self._http_socket).start()
        await web.SockSite(self._runner, self._https_socket, ssl_context=self._tls_context).start()
        await asyncio.to_thread(self._register_discovery)

    def _register_discovery(self) -> None:
        self._zeroconf = Zeroconf(interfaces=list(self.addresses))
        self._service_info = ServiceInfo(
            "_privarcy._tcp.local.",
            f"PrivARcy-{self.device_id}._privarcy._tcp.local.",
            addresses=[socket.inet_aton(address) for address in self.addresses],
            port=self.https_port,
            properties={
                b"pc_id": self.device_id.encode("ascii"),
                b"ca_sha256": self.fingerprint.replace(":", "").lower().encode("ascii"),
            },
            server=f"{self.mdns_hostname}.",
        )
        self._zeroconf.register_service(self._service_info, allow_name_change=False)
        self._service_registered = True

    def _close_discovery(self) -> None:
        if self._zeroconf is None:
            return
        if self._service_info is not None and self._service_registered:
            self._zeroconf.unregister_service(self._service_info)
            self._service_registered = False
        self._zeroconf.close()
        self._zeroconf = None
        self._service_info = None

    def _make_app(self) -> web.Application:
        app = web.Application(client_max_size=65536)
        app.router.add_get("/pair/{token}", self._pair_page)
        app.router.add_get("/ca/{token}", self._download_ca)
        app.router.add_get("/signal/{token}", self._signaling)
        app.router.add_get("/device/{pc_id}/{device_id}", self._device_signaling)
        app.router.add_get("/phone_pairing.js", self._phone_script)
        app.router.add_get("/phone_pairing.css", self._phone_styles)
        return app

    def _valid_token(self, token: str, allow_used: bool = False) -> bool:
        if token != self.token or self._stopped.is_set() or time.monotonic() >= self.expires_at:
            return False
        with self._status_lock:
            return allow_used or not self._used

    def _valid_host(self, request: web.Request) -> bool:
        return request.host in self._allowed_hosts

    async def _pair_page(self, request: web.Request) -> web.Response:
        token = request.match_info["token"]
        if not self._valid_host(request) or not self._valid_token(token):
            raise web.HTTPGone(text="This pairing link is invalid, expired, or already used.")
        qr_fingerprint = request.query.get("ca_sha256", "").lower()
        expected_fingerprint = self.fingerprint.replace(":", "").lower()
        if not hmac.compare_digest(qr_fingerprint, expected_fingerprint):
            raise web.HTTPForbidden(text="The pairing link does not contain the expected certificate fingerprint.")
        template = (Path(__file__).parent / "phone_pairing.html").read_text(encoding="utf-8")
        secure_url = self.secure_url
        ca_url = f"http://{self.address}:{self.http_port}/ca/{self.token}"
        html = template.replace("__SECURE_URL__", secure_url).replace("__CA_URL__", ca_url)
        html = html.replace("__FINGERPRINT__", self.fingerprint)
        response = web.Response(text=html, content_type="text/html")
        response.headers.update({
            "Cache-Control": "no-store",
            "Content-Security-Policy": (
                f"default-src 'self'; script-src 'self'; style-src 'self'; "
                f"connect-src 'self' https://{self.address}:{self.https_port} wss:"
            ),
            "X-Content-Type-Options": "nosniff",
            "Referrer-Policy": "no-referrer",
        })
        return response

    async def _download_ca(self, request: web.Request) -> web.Response:
        token = request.match_info["token"]
        if not self._valid_host(request) or not self._valid_token(token, allow_used=True):
            raise web.HTTPGone(text="This pairing link is invalid or expired.")
        if request.query.get("platform") == "ios":
            profile_id = uuid.uuid5(uuid.NAMESPACE_URL, self.fingerprint)
            payload_id = f"com.privarcy.phone-camera.ca.{profile_id}"
            cert_content = base64.b64encode(self.identity.certificate.public_bytes(serialization.Encoding.DER))
            profile = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>PayloadContent</key><array><dict>
<key>PayloadCertificateFileName</key><string>PrivARcy-Phone-CA.cer</string>
<key>PayloadContent</key><data>{cert_content.decode("ascii")}</data>
<key>PayloadDescription</key><string>Verifies the local HTTPS camera page hosted by PrivARcy on this PC.</string>
<key>PayloadDisplayName</key><string>PrivARcy Local Phone Camera</string>
<key>PayloadIdentifier</key><string>{payload_id}.certificate</string>
<key>PayloadType</key><string>com.apple.security.root</string>
<key>PayloadUUID</key><string>{uuid.uuid5(profile_id, "certificate")}</string>
<key>PayloadVersion</key><integer>1</integer>
</dict></array>
<key>PayloadDisplayName</key><string>PrivARcy Local Phone Camera</string>
<key>PayloadIdentifier</key><string>{payload_id}</string>
<key>PayloadRemovalDisallowed</key><false/>
<key>PayloadType</key><string>Configuration</string>
<key>PayloadUUID</key><string>{profile_id}</string>
<key>PayloadVersion</key><integer>1</integer>
</dict></plist>"""
            return web.Response(
                text=profile,
                content_type="application/x-apple-aspen-config",
                headers={"Content-Disposition": 'attachment; filename="PrivARcy-Phone-CA.mobileconfig"',
                         "Cache-Control": "no-store"},
            )
        return web.Response(
            body=self._ca_pem,
            content_type="application/x-x509-ca-cert",
            headers={"Content-Disposition": 'attachment; filename="PrivARcy-Phone-CA.crt"', "Cache-Control": "no-store"},
        )

    async def _phone_script(self, request: web.Request) -> web.Response:
        if not self._valid_host(request):
            raise web.HTTPForbidden(text="Invalid pairing host.")
        script = (Path(__file__).parent / "phone_pairing.js").read_text(encoding="utf-8")
        return web.Response(
            text=script, content_type="application/javascript",
            headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
        )

    async def _phone_styles(self, request: web.Request) -> web.Response:
        if not self._valid_host(request):
            raise web.HTTPForbidden(text="Invalid pairing host.")
        styles = (Path(__file__).parent / "phone_pairing.css").read_text(encoding="utf-8")
        return web.Response(text=styles, content_type="text/css", headers={"Cache-Control": "no-store"})

    async def _signaling(self, request: web.Request) -> web.WebSocketResponse:
        token = request.match_info["token"]
        origin = request.headers.get("Origin", "")
        expected_origin = f"https://{request.host}"
        if not self._valid_host(request) or not self._valid_token(token):
            raise web.HTTPGone(text="This pairing link is invalid, expired, or already used.")
        if origin != expected_origin:
            raise web.HTTPForbidden(text="The signaling origin does not match this pairing session.")
        with self._status_lock:
            if self._used or self._claimed:
                raise web.HTTPGone(text="This pairing link has already been used.")
            self._claimed = True
            self._status = "phone_connected"

        ws = web.WebSocketResponse(max_msg_size=65536, heartbeat=20)
        pcs: list[RTCPeerConnection] = []
        owns_stream = False
        try:
            await ws.prepare(request)
            self._websockets.add(ws)
            async for message in ws:
                if message.type == WSMsgType.TEXT:
                    try:
                        payload = message.json()
                    except (ValueError, TypeError):
                        await ws.send_json({"type": "error", "message": "Invalid signaling message."})
                        continue
                    if not isinstance(payload, dict):
                        continue
                    message_type = payload.get("type")
                    if message_type == "enroll":
                        if self.device_store is None:
                            await ws.send_json({"type": "error", "message": "Persistent pairing is unavailable."})
                            continue
                        device_id = payload.get("device_id")
                        device_name = payload.get("device_name")
                        encoded_secret = payload.get("secret")
                        if (
                            not isinstance(device_id, str)
                            or not DEVICE_ID_PATTERN.fullmatch(device_id)
                            or not isinstance(device_name, str)
                            or len(device_name) > 64
                            or not isinstance(encoded_secret, str)
                            or len(encoded_secret) > 128
                        ):
                            await ws.send_json({"type": "error", "message": "Invalid phone pairing details."})
                            continue
                        try:
                            secret = base64.urlsafe_b64decode(encoded_secret.encode("ascii") + b"===")
                            self.device_store.add_device(device_id, device_name, secret)
                        except (UnicodeError, ValueError, OSError, RuntimeError) as exc:
                            await ws.send_json({"type": "error", "message": str(exc)})
                            continue
                        with self._status_lock:
                            self._used = True
                        await ws.send_json({
                            "type": "paired",
                            "device_id": device_id,
                            "pc_id": self.device_id,
                            "mdns_host": self.mdns_hostname,
                            "https_port": self.https_port,
                        })
                    elif message_type == "state":
                        state = payload.get("state")
                        if state in ("permission_required", "camera_ready", "connection_error"):
                            self._set_status(state)
                    elif message_type == "offer":
                        if pcs or not isinstance(payload.get("sdp"), str) or len(payload["sdp"]) > 60000:
                            await ws.send_json({"type": "error", "message": "Invalid or duplicate WebRTC offer."})
                            continue
                        with self._status_lock:
                            if self._stream_claimed:
                                await ws.send_json({"type": "error", "message": "Another phone is using the camera connection."})
                                continue
                            self._stream_claimed = True
                            owns_stream = True
                            self._used = True
                        pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
                        pcs.append(pc)
                        self._connections.add(pc)

                        @pc.on("track")
                        def on_track(track):
                            if track.kind == "video":
                                asyncio.create_task(self._consume_track(track, ws))

                        @pc.on("connectionstatechange")
                        async def on_connection_state_change(connection=pc):
                            if connection.connectionState == "failed":
                                self._set_status("connection_lost")

                        await pc.setRemoteDescription(RTCSessionDescription(sdp=payload["sdp"], type="offer"))
                        answer = await pc.createAnswer()
                        await pc.setLocalDescription(answer)
                        await ws.send_json({"type": "answer", "sdp": pc.localDescription.sdp})
                    elif message_type == "stop":
                        self._set_status("disconnected")
                        await ws.close()
                        break
                elif message.type in (WSMsgType.ERROR, WSMsgType.CLOSE, WSMsgType.CLOSED):
                    break
        except (ConnectionError, asyncio.TimeoutError) as exc:
            logger.info("Phone signaling connection ended: %s", exc)
        except (ValueError, TypeError, RuntimeError, OSError, web.WebSocketError):
            logger.exception("Phone signaling failed")
            self._set_status("connection_error")
        finally:
            self._websockets.discard(ws)
            for pc in pcs:
                self._connections.discard(pc)
                await pc.close()
            with self._status_lock:
                self._claimed = False
                if owns_stream:
                    self._stream_claimed = False
                if not self._used and not self._stopped.is_set() and self._status != "connection_error":
                    self._status = "waiting_for_phone"
            if self.status not in ("disconnected", "connection_error", "pairing_expired"):
                if self._used:
                    self._set_status("connection_lost")
            if self._used:
                self.capture.release()
                if self.device_store is not None and self.device_store.list_devices():
                    self.capture = PhoneFrameCapture()
                    self._frame_seen = False
                    self._last_frame_at = None
        return ws

    async def _device_signaling(self, request: web.Request) -> web.WebSocketResponse:
        device_id = request.match_info["device_id"]
        if (
            not request.secure
            or not self._valid_host(request)
            or request.match_info["pc_id"] != self.device_id
            or self.device_store is None
        ):
            raise web.HTTPForbidden(text="This paired-device endpoint is unavailable.")
        origin = request.headers.get("Origin", "")
        if origin != f"https://{request.host}":
            raise web.HTTPForbidden(text="The signaling origin does not match this connection.")
        if not DEVICE_ID_PATTERN.fullmatch(device_id):
            raise web.HTTPForbidden(text="Invalid phone identity.")
        secret = self.device_store.secret_for(device_id)
        if secret is None:
            raise web.HTTPForbidden(text="This phone is not paired or has been revoked.")

        with self._status_lock:
            if device_id in self._active_device_ids or len(self._active_device_ids) >= MAX_PAIRED_DEVICES:
                raise web.HTTPTooManyRequests(text="This phone is already connected or the connection limit was reached.")
            self._active_device_ids.add(device_id)

        ws = web.WebSocketResponse(max_msg_size=65536, heartbeat=20)
        pcs: list[RTCPeerConnection] = []
        owns_stream = False
        try:
            await ws.prepare(request)
            self._websockets.add(ws)
            server_nonce = secrets.token_urlsafe(32)
            await ws.send_json({"type": "challenge", "pc_id": self.device_id, "nonce": server_nonce})
            message = await ws.receive(timeout=10)
            if message.type != WSMsgType.TEXT:
                raise web.HTTPForbidden(text="Phone authentication was not completed.")
            try:
                auth = message.json()
            except (ValueError, TypeError):
                auth = {}
            client_nonce = auth.get("client_nonce") if isinstance(auth, dict) else None
            proof = auth.get("proof") if isinstance(auth, dict) else None
            if (
                not isinstance(client_nonce, str)
                or auth.get("type") != "auth"
                or not re.fullmatch(r"[A-Za-z0-9_-]{43}", client_nonce)
                or not isinstance(proof, str)
                or not re.fullmatch(r"[a-f0-9]{64}", proof)
            ):
                await ws.send_json({"type": "error", "message": "Invalid paired-device authentication."})
                await ws.close()
                return ws
            challenge = f"privarcy-device-auth-v1\n{self.device_id}\n{device_id}\n{server_nonce}\n{client_nonce}".encode()
            expected = hmac.new(secret, challenge, hashlib.sha256).hexdigest()
            if not hmac.compare_digest(proof, expected):
                await ws.send_json({"type": "error", "message": "Paired-device authentication failed."})
                await ws.close()
                return ws
            current_secret = self.device_store.secret_for(device_id)
            if current_secret is None or not hmac.compare_digest(current_secret, secret):
                await ws.send_json({"type": "error", "message": "This phone pairing has been revoked."})
                await ws.close()
                return ws
            self._device_websockets[device_id] = ws
            response_proof = hmac.new(secret, b"server\n" + challenge, hashlib.sha256).hexdigest()
            await ws.send_json({"type": "authenticated", "proof": response_proof})
            self.device_store.mark_connected(device_id)
            self._set_status("phone_connected")

            async for message in ws:
                if message.type != WSMsgType.TEXT:
                    if message.type in (WSMsgType.ERROR, WSMsgType.CLOSE, WSMsgType.CLOSED):
                        break
                    continue
                try:
                    payload = message.json()
                except (ValueError, TypeError):
                    await ws.send_json({"type": "error", "message": "Invalid signaling message."})
                    continue
                if not isinstance(payload, dict):
                    continue
                if payload.get("type") == "offer":
                    sdp = payload.get("sdp")
                    with self._status_lock:
                        busy = self._stream_claimed
                        if not busy and isinstance(sdp, str) and len(sdp) <= 60000:
                            self._stream_claimed = True
                            owns_stream = True
                    if busy or not isinstance(sdp, str) or len(sdp) > 60000:
                        await ws.send_json({
                            "type": "error",
                            "message": "Another phone is using the camera connection." if busy else "Invalid WebRTC offer.",
                        })
                        continue
                    self.capture = PhoneFrameCapture()
                    self._frame_seen = False
                    self._last_frame_at = None
                    pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
                    pcs.append(pc)
                    self._connections.add(pc)

                    @pc.on("track")
                    def on_track(track):
                        if track.kind == "video":
                            asyncio.create_task(self._consume_track(track, ws))

                    @pc.on("connectionstatechange")
                    async def on_connection_state_change(connection=pc):
                        if connection.connectionState == "failed":
                            self._set_status("connection_lost")

                    await pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type="offer"))
                    answer = await pc.createAnswer()
                    await pc.setLocalDescription(answer)
                    await ws.send_json({"type": "answer", "sdp": pc.localDescription.sdp})
                    self._set_status("camera_ready")
                elif payload.get("type") == "state":
                    state = payload.get("state")
                    if state in ("camera_ready", "connection_error"):
                        self._set_status(state)
                elif payload.get("type") == "stop":
                    self._set_status("disconnected")
                    break
                elif payload.get("type") == "revoke":
                    self.device_store.revoke(device_id)
                    await ws.send_json({"type": "revoked"})
                    await ws.close()
                    break
        except (asyncio.TimeoutError, ConnectionError, web.WebSocketError) as exc:
            logger.info("Paired phone signaling ended: %s", exc)
        except (ValueError, TypeError, RuntimeError, OSError):
            logger.exception("Paired phone signaling failed")
            self._set_status("connection_error")
        finally:
            self._websockets.discard(ws)
            self._device_websockets.pop(device_id, None)
            for pc in pcs:
                self._connections.discard(pc)
                await pc.close()
            with self._status_lock:
                self._active_device_ids.discard(device_id)
                if owns_stream:
                    self._stream_claimed = False
            if owns_stream:
                self.capture.release()
                self.capture = PhoneFrameCapture()
                self._frame_seen = False
                self._last_frame_at = None
            with self._status_lock:
                other_devices_connected = bool(self._active_device_ids)
            if (
                self.status != "connection_error"
                and not other_devices_connected
            ):
                self._set_status("waiting_for_phone")
        return ws

    def paired_devices(self) -> list[dict[str, str]]:
        return self.device_store.list_devices() if self.device_store is not None else []

    def revoke_device(self, device_id: str) -> bool:
        if self.device_store is None or not self.device_store.revoke(device_id):
            return False
        loop = self._loop
        if loop is not None and loop.is_running():
            future = asyncio.run_coroutine_threadsafe(self._close_device(device_id), loop)
            future.result(timeout=5)
        return True

    async def _close_device(self, device_id: str) -> None:
        ws = self._device_websockets.get(device_id)
        if ws is not None and not ws.closed:
            await ws.close()

    def disconnect_paired_devices(self) -> None:
        loop = self._loop
        if loop is not None and loop.is_running():
            future = asyncio.run_coroutine_threadsafe(self._close_paired_devices(), loop)
            future.result(timeout=5)

    async def _close_paired_devices(self) -> None:
        sockets = tuple(self._websockets)
        if sockets:
            await asyncio.gather(*(ws.close() for ws in sockets if not ws.closed))

    async def _consume_track(self, track, ws: web.WebSocketResponse) -> None:
        try:
            while not self._stopped.is_set():
                frame = await track.recv()
                image = frame.to_ndarray(format="bgr24")
                self.capture.put(image)
                with self._status_lock:
                    first_frame = not self._frame_seen
                    self._frame_seen = True
                    self._last_frame_at = time.monotonic()
                    if not self._stopped.is_set():
                        self._status = "streaming"
                if first_frame and not ws.closed:
                    await ws.send_json({"type": "state", "state": "streaming"})
        except asyncio.CancelledError:
            raise
        except (MediaStreamError, ValueError, RuntimeError, OSError) as exc:
            logger.info("Phone video track ended: %s", exc)
            if not ws.closed:
                self._set_status("connection_lost")

    def _set_status(self, status: str) -> None:
        with self._status_lock:
            if not self._stopped.is_set():
                self._status = status

    @property
    def status(self) -> str:
        if self._stopped.is_set():
            return "disconnected"
        with self._status_lock:
            if (
                time.monotonic() >= self.expires_at
                and not self._frame_seen
                and (self.device_store is None or not self.device_store.list_devices())
            ):
                return "pairing_expired"
            if (
                self._status == "streaming"
                and self._last_frame_at is not None
                and time.monotonic() - self._last_frame_at > FRAME_STALE_SECONDS
            ):
                return "connection_lost"
            return self._status

    @property
    def frame_received(self) -> bool:
        with self._status_lock:
            return self._frame_seen

    @property
    def seconds_remaining(self) -> int:
        return max(0, int(self.expires_at - time.monotonic() + 0.999))

    def stop(self) -> None:
        if self._stopped.is_set():
            return
        self._stopped.set()
        self.capture.release()
        loop = self._loop
        if loop is not None and loop.is_running():
            asyncio.run_coroutine_threadsafe(self._close_async(), loop).result(timeout=5)
            loop.call_soon_threadsafe(loop.stop)
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=5)
        for sock in (self._http_socket, self._https_socket):
            try:
                sock.close()
            except OSError:
                logger.debug("Pairing socket already closed.")
        self._temp_dir.cleanup()

    async def _close_async(self) -> None:
        for ws in tuple(self._websockets):
            await ws.close()
        for pc in tuple(self._connections):
            await pc.close()
