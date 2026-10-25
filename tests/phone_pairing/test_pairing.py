"""Tests for local pairing credentials, the capture adapter, TLS and WebRTC signaling."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import os
import plistlib
import secrets
import ssl
import time
import urllib.error
import urllib.request
from urllib.parse import parse_qs, urlsplit

import numpy as np
import pytest
from aiohttp import ClientSession, WSServerHandshakeError
from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from av import VideoFrame
from cryptography import x509
from cryptography.hazmat.primitives import hashes

from privarcy.capture.phone_pairing import (
    PhoneDeviceStore,
    PhoneFrameCapture,
    PhonePairingIdentity,
    PhonePairingService,
)


def test_pairing_token_is_random_single_use_and_expires():
    service = PhonePairingService(ttl_seconds=20)
    try:
        assert len(service.token) >= 40
        assert service._valid_token(service.token)
        assert not service._valid_token("not-the-token")
        with service._status_lock:
            service._used = True
        assert not service._valid_token(service.token)
        assert service._valid_token(service.token, allow_used=True)
        service.expires_at = time.monotonic() - 1
        assert not service._valid_token(service.token, allow_used=True)
        assert service.status == "pairing_expired"
    finally:
        service.stop()


def test_fresh_pairings_reuse_the_in_memory_ca_but_rotate_the_token():
    identity = PhonePairingIdentity()
    first = PhonePairingService(identity=identity)
    second = PhonePairingService(identity=identity)
    try:
        assert first.fingerprint == second.fingerprint == identity.fingerprint
        assert first.token != second.token
        qr = urlsplit(first.qr_url)
        query = parse_qs(qr.query)
        assert query["ca_sha256"] == [identity.fingerprint.replace(":", "").lower()]
        assert query["https_port"] == [str(first.https_port)]
    finally:
        first.stop()
        second.stop()


@pytest.mark.skipif(os.name != "nt", reason="Persistent CA storage uses Windows DPAPI.")
def test_phone_pairing_identity_is_reused_with_windows_user_protection(tmp_path):
    first = PhonePairingIdentity.load_or_create(tmp_path)
    second = PhonePairingIdentity.load_or_create(tmp_path)
    assert first.fingerprint == second.fingerprint
    assert (tmp_path / "phone-ca.dpapi").is_file()
    assert (tmp_path / "phone-ca.crt").read_bytes() == first.certificate_pem


@pytest.mark.skipif(os.name != "nt", reason="Persistent phone storage uses Windows DPAPI.")
def test_phone_device_store_is_persistent_and_bound_to_pc_identity(tmp_path):
    identity = PhonePairingIdentity.load_or_create(tmp_path)
    store = PhoneDeviceStore(tmp_path, identity.device_id)
    secret = secrets.token_bytes(32)
    device_id = secrets.token_hex(16)
    store.add_device(device_id, "My phone", secret)

    restored = PhoneDeviceStore(tmp_path, identity.device_id)
    assert restored.secret_for(device_id) == secret
    assert restored.list_devices()[0]["name"] == "My phone"
    with pytest.raises(RuntimeError, match="credentials could not be read"):
        PhoneDeviceStore(tmp_path, secrets.token_hex(16))
    assert restored.revoke(device_id)
    assert restored.secret_for(device_id) is None


class MemoryPhoneDeviceStore:
    def __init__(self, pc_id):
        self.pc_id = pc_id
        self.devices = {}

    def list_devices(self):
        return [
            {"device_id": device_id, "name": record["name"], "created_at": ""}
            for device_id, record in self.devices.items()
        ]

    def secret_for(self, device_id):
        record = self.devices.get(device_id)
        return record["secret"] if record else None

    def add_device(self, device_id, name, secret):
        if device_id in self.devices:
            raise ValueError("already paired")
        self.devices[device_id] = {"name": name, "secret": secret}

    def mark_connected(self, _device_id):
        pass

    def revoke(self, device_id):
        return self.devices.pop(device_id, None) is not None


async def _connect_paired_device(client, service, device_id, secret, context):
    origin = f"https://{service.address}:{service.https_port}"
    websocket = await client.ws_connect(
        f"wss://{service.address}:{service.https_port}/device/{service.device_id}/{device_id}",
        ssl=context,
        headers={"Origin": origin},
    )
    challenge = await asyncio.wait_for(websocket.receive_json(), timeout=5)
    client_nonce = secrets.token_urlsafe(32)
    message = (
        f"privarcy-device-auth-v1\n{service.device_id}\n{device_id}\n"
        f"{challenge['nonce']}\n{client_nonce}"
    ).encode()
    proof = hmac.new(secret, message, hashlib.sha256).hexdigest()
    await websocket.send_json(
        {"type": "auth", "client_nonce": client_nonce, "proof": proof}
    )
    authenticated = await asyncio.wait_for(websocket.receive_json(), timeout=5)
    return websocket, authenticated


def test_paired_device_authentication_is_fresh_and_streams_are_exclusive():
    identity = PhonePairingIdentity()
    store = MemoryPhoneDeviceStore(identity.device_id)
    secrets_by_device = {
        secrets.token_hex(16): secrets.token_bytes(32),
        secrets.token_hex(16): secrets.token_bytes(32),
        secrets.token_hex(16): secrets.token_bytes(32),
    }
    for index, (device_id, secret) in enumerate(secrets_by_device.items()):
        store.add_device(device_id, f"Phone {index + 1}", secret)
    service = PhonePairingService(identity=identity, device_store=store).start()

    async def exercise():
        context = ssl.create_default_context(cadata=service._ca_pem.decode("ascii"))
        async with ClientSession() as client:
            ids = list(secrets_by_device)
            peer = None
            first, first_result = await _connect_paired_device(
                client, service, ids[0], secrets_by_device[ids[0]], context
            )
            second, second_result = await _connect_paired_device(
                client, service, ids[1], secrets_by_device[ids[1]], context
            )
            assert first_result["type"] == second_result["type"] == "authenticated"
            assert len(service._active_device_ids) == 2

            peer = RTCPeerConnection(RTCConfiguration(iceServers=[]))
            peer.addTrack(SyntheticVideoTrack())
            offer = await peer.createOffer()
            await peer.setLocalDescription(offer)
            await _wait_for_ice(peer)
            await first.send_json(
                {"type": "offer", "sdp": peer.localDescription.sdp}
            )
            answer = await asyncio.wait_for(first.receive_json(), timeout=10)
            assert answer["type"] == "answer"
            await peer.setRemoteDescription(
                RTCSessionDescription(type="answer", sdp=answer["sdp"])
            )
            await asyncio.wait_for(_wait_for_frame(service), timeout=15)
            assert service.status == "streaming"

            await second.send_json({"type": "offer", "sdp": "v=0"})
            busy = await asyncio.wait_for(second.receive_json(), timeout=5)
            assert busy["type"] == "error"
            assert "Another phone" in busy["message"]

            await asyncio.to_thread(service.disconnect_paired_devices)
            await second.close()
            await peer.close()
            peer = None

            replay_id = ids[2]
            replay_secret = secrets_by_device[replay_id]
            replay_context = ssl.create_default_context(cadata=service._ca_pem.decode("ascii"))
            async with client.ws_connect(
                f"wss://{service.address}:{service.https_port}/device/{service.device_id}/{replay_id}",
                ssl=replay_context,
                headers={"Origin": f"https://{service.address}:{service.https_port}"},
            ) as original:
                original_challenge = await asyncio.wait_for(original.receive_json(), timeout=5)
                old_nonce = secrets.token_urlsafe(32)
                old_text = (
                    f"privarcy-device-auth-v1\n{service.device_id}\n{replay_id}\n"
                    f"{original_challenge['nonce']}\n{old_nonce}"
                ).encode()
                old_proof = hmac.new(replay_secret, old_text, hashlib.sha256).hexdigest()
            async with client.ws_connect(
                f"wss://{service.address}:{service.https_port}/device/{service.device_id}/{replay_id}",
                ssl=replay_context,
                headers={"Origin": f"https://{service.address}:{service.https_port}"},
            ) as replay:
                await asyncio.wait_for(replay.receive_json(), timeout=5)
                await replay.send_json(
                    {
                        "type": "auth",
                        "client_nonce": old_nonce,
                        "proof": old_proof,
                    }
                )
                rejected = await asyncio.wait_for(replay.receive_json(), timeout=5)
                assert rejected["type"] == "error"
                assert "authentication failed" in rejected["message"]

            assert await asyncio.to_thread(service.revoke_device, replay_id)
            with pytest.raises(WSServerHandshakeError) as revoked:
                await client.ws_connect(
                    f"wss://{service.address}:{service.https_port}/device/"
                    f"{service.device_id}/{replay_id}",
                    ssl=replay_context,
                    headers={
                        "Origin": f"https://{service.address}:{service.https_port}"
                    },
                )
            assert revoked.value.status == 403
            if peer is not None:
                await peer.close()
    try:
        asyncio.run(exercise())
    finally:
        service.stop()


def test_phone_stream_reports_lost_video_when_frames_go_stale():
    service = PhonePairingService()
    try:
        with service._status_lock:
            service._frame_seen = True
            service._status = "streaming"
            service._last_frame_at = time.monotonic() - 4
        assert service.status == "connection_lost"
    finally:
        service.stop()


def test_phone_capture_keeps_only_latest_frame_and_handles_resolution_changes():
    capture = PhoneFrameCapture()
    first = np.zeros((48, 64, 3), dtype=np.uint8)
    latest = np.full((72, 96, 3), 80, dtype=np.uint8)
    capture.put(first)
    capture.put(latest)
    ok, frame = capture.read()
    assert ok and frame is latest
    assert capture.get(3) == 96
    assert capture.get(4) == 72
    with pytest.raises(ValueError, match="unsupported format"):
        capture.put(np.zeros((10, 10), dtype=np.uint8))
    capture.release()
    assert not capture.isOpened()
    assert capture.read() == (False, None)


def test_pairing_http_https_and_webrtc_frame_delivery():
    service = PhonePairingService().start()

    async def exercise():
        peer = None
        try:
            response = await asyncio.to_thread(urllib.request.urlopen, service.url, timeout=5)
            page = response.read().decode("utf-8")
            assert "Install a certificate" in page
            assert service.secure_url in page
            base_url = f"http://{service.address}:{service.http_port}"
            script = await asyncio.to_thread(
                urllib.request.urlopen, f"{base_url}/phone_pairing.js", timeout=5
            )
            assert "getUserMedia" in script.read().decode("utf-8")
            styles = await asyncio.to_thread(
                urllib.request.urlopen, f"{base_url}/phone_pairing.css", timeout=5
            )
            assert ".status" in styles.read().decode("utf-8")
            ca_url = f"{base_url}/ca/{service.token}"
            ca_response = await asyncio.to_thread(urllib.request.urlopen, ca_url, timeout=5)
            downloaded_ca = x509.load_pem_x509_certificate(ca_response.read())
            assert downloaded_ca.fingerprint(hashes.SHA256()).hex(":").upper() == service.fingerprint
            profile_response = await asyncio.to_thread(
                urllib.request.urlopen, f"{ca_url}?platform=ios", timeout=5
            )
            profile = plistlib.loads(profile_response.read())
            profile_certificate = profile["PayloadContent"][0]["PayloadContent"]
            assert x509.load_der_x509_certificate(profile_certificate).fingerprint(hashes.SHA256()) == (
                downloaded_ca.fingerprint(hashes.SHA256())
            )

            trusted_context = ssl.create_default_context(cadata=service._ca_pem.decode("ascii"))
            secure_response = await asyncio.to_thread(
                urllib.request.urlopen, service.secure_url, timeout=5, context=trusted_context
            )
            assert "start camera" in secure_response.read().decode("utf-8").lower()

            async with ClientSession() as client:
                origin = f"https://{service.address}:{service.https_port}"
                async with client.ws_connect(
                    f"wss://{service.address}:{service.https_port}/signal/{service.token}",
                    ssl=trusted_context,
                    headers={"Origin": origin},
                ) as websocket:
                    await websocket.send_json({"type": "state", "state": "camera_ready"})
                    peer = RTCPeerConnection(RTCConfiguration(iceServers=[]))
                    peer.addTrack(SyntheticVideoTrack())
                    offer = await peer.createOffer()
                    await peer.setLocalDescription(offer)
                    await _wait_for_ice(peer)
                    await websocket.send_json({"type": "offer", "sdp": peer.localDescription.sdp})
                    answer = await asyncio.wait_for(websocket.receive_json(), timeout=10)
                    assert answer["type"] == "answer"
                    await peer.setRemoteDescription(RTCSessionDescription(type="answer", sdp=answer["sdp"]))

                    await asyncio.wait_for(_wait_for_frame(service), timeout=15)
                    ok, frame = await asyncio.to_thread(service.capture.read)
                    assert ok and frame is not None
                    assert frame.shape == (48, 64, 3)
                    assert service.status == "streaming"
                    assert not service._valid_token(service.token)
        finally:
            if peer is not None:
                await peer.close()

    try:
        asyncio.run(exercise())
    finally:
        service.stop()


class SyntheticVideoTrack(VideoStreamTrack):
    async def recv(self):
        pts, time_base = await self.next_timestamp()
        frame = VideoFrame.from_ndarray(np.full((48, 64, 3), (20, 100, 220), dtype=np.uint8), format="bgr24")
        frame.pts = pts
        frame.time_base = time_base
        return frame


async def _wait_for_ice(peer: RTCPeerConnection) -> None:
    deadline = asyncio.get_running_loop().time() + 5
    while peer.iceGatheringState != "complete":
        if asyncio.get_running_loop().time() >= deadline:
            raise TimeoutError("Local ICE candidate gathering timed out.")
        await asyncio.sleep(0.02)


async def _wait_for_frame(service: PhonePairingService) -> None:
    deadline = asyncio.get_running_loop().time() + 10
    while not service.frame_received:
        if asyncio.get_running_loop().time() >= deadline:
            raise TimeoutError("No phone video frame was received.")
        await asyncio.sleep(0.02)


def test_pairing_rejects_invalid_token_and_origin():
    service = PhonePairingService().start()
    try:
        request = urllib.request.Request(service.url.replace(service.token, "invalid"))
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request, timeout=5)
        assert error.value.code == 410

        async def wrong_origin():
            trusted_context = ssl.create_default_context(cadata=service._ca_pem.decode("ascii"))
            async with ClientSession() as client:
                with pytest.raises(WSServerHandshakeError) as rejected:
                    await client.ws_connect(
                        f"wss://{service.address}:{service.https_port}/signal/{service.token}",
                        ssl=trusted_context,
                        headers={"Origin": "https://untrusted.invalid"},
                    )
                assert rejected.value.status == 403

        asyncio.run(wrong_origin())
    finally:
        service.stop()


def test_signaling_disconnect_before_camera_offer_keeps_pairing_available():
    service = PhonePairingService().start()

    async def exercise():
        trusted_context = ssl.create_default_context(cadata=service._ca_pem.decode("ascii"))
        async with ClientSession() as client:
            async with client.ws_connect(
                f"wss://{service.address}:{service.https_port}/signal/{service.token}",
                ssl=trusted_context,
                headers={"Origin": f"https://{service.address}:{service.https_port}"},
            ) as websocket:
                await websocket.send_json({"type": "state", "state": "camera_ready"})
            deadline = asyncio.get_running_loop().time() + 3
            while service.status != "waiting_for_phone":
                if asyncio.get_running_loop().time() >= deadline:
                    raise TimeoutError("Pairing was not released after camera setup was cancelled.")
                await asyncio.sleep(0.02)
        assert service._valid_token(service.token)
        assert service.capture.isOpened()

    try:
        asyncio.run(exercise())
    finally:
        service.stop()
