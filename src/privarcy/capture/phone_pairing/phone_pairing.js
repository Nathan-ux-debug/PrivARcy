"use strict";

const trustPanel = document.getElementById("trust");
const cameraPanel = document.getElementById("camera");
const statusText = document.getElementById("status");
const secureLink = document.getElementById("secure-link");
const downloadLink = document.getElementById("download-ca");
const startButton = document.getElementById("start");
const stopButton = document.getElementById("stop");
const preview = document.getElementById("preview");
const cameraActive = document.getElementById("active");
const facing = document.getElementById("facing");
const fingerprint = document.getElementById("fingerprint");
const trustStatus = document.getElementById("trust-status");
const fingerprintVerified = document.getElementById("fingerprint-verified");
const secureUrl = document.body.dataset.secureUrl;
const caUrl = document.body.dataset.caUrl;

fingerprint.textContent = document.body.dataset.fingerprint;
const pairingQuery = new URLSearchParams(location.search);
const qrFingerprint = pairingQuery.get("ca_sha256") || "";
const normalizedQrFingerprint = qrFingerprint.replace(/:/g, "").toLowerCase();
const normalizedPageFingerprint = document.body.dataset.fingerprint.replace(/:/g, "").toLowerCase();
const fingerprintMatches = /^[a-f0-9]{64}$/.test(normalizedQrFingerprint)
  && normalizedQrFingerprint === normalizedPageFingerprint;
if (!fingerprintMatches) {
  trustStatus.textContent = "The certificate fingerprint in this QR code does not match PrivARcy. Do not install a certificate.";
  downloadLink.removeAttribute("href");
  secureLink.removeAttribute("href");
  downloadLink.setAttribute("aria-disabled", "true");
  secureLink.setAttribute("aria-disabled", "true");
}

const isIos = /iPad|iPhone|iPod/.test(navigator.userAgent)
  || (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1);
if (fingerprintMatches) {
  downloadLink.href = isIos ? `${caUrl}?platform=ios` : caUrl;
  trustStatus.textContent = "Before trusting the downloaded certificate, verify its SHA-256 fingerprint against the value shown in PrivARcy.";
}
downloadLink.download = isIos ? "PrivARcy-Phone-CA.mobileconfig" : "PrivARcy-Phone-CA.crt";
downloadLink.textContent = isIos ? "Download iPhone setup profile" : "Download Android certificate";
document.getElementById(isIos ? "ios-steps" : "android-steps").open = true;

function updateSecureLink() {
  if (!fingerprintMatches || !fingerprintVerified.checked) {
    secureLink.removeAttribute("href");
    secureLink.setAttribute("aria-disabled", "true");
    return;
  }
  const verifiedUrl = new URL(secureUrl);
  verifiedUrl.searchParams.set("ca_sha256", normalizedQrFingerprint);
  verifiedUrl.searchParams.set("verified", "1");
  secureLink.href = verifiedUrl.toString();
  secureLink.removeAttribute("aria-disabled");
}

fingerprintVerified.addEventListener("change", updateSecureLink);
updateSecureLink();

if (location.protocol === "https:"
    && fingerprintMatches
    && pairingQuery.get("verified") === "1") {
  trustPanel.classList.add("hidden");
  cameraPanel.classList.remove("hidden");
}

let peer = null;
let socket = null;
let localStream = null;
let stoppedByUser = false;
let pairingConsumed = false;
let disconnectTimer = null;

function setStatus(text) {
  statusText.textContent = text;
}

function cameraConstraints() {
  return {
    audio: false,
    video: {
      facingMode: { exact: facing.value },
      width: { ideal: 1280, max: 1920 },
      height: { ideal: 720, max: 1080 },
      frameRate: { ideal: 24, max: 30 }
    }
  };
}

function waitForIceGathering(connection) {
  if (connection.iceGatheringState === "complete") {
    return Promise.resolve();
  }
  return new Promise((resolve, reject) => {
    const timeout = window.setTimeout(() => {
      connection.removeEventListener("icegatheringstatechange", changed);
      reject(new Error("The local video connection timed out."));
    }, 10000);
    const changed = () => {
      if (connection.iceGatheringState === "complete") {
        window.clearTimeout(timeout);
        connection.removeEventListener("icegatheringstatechange", changed);
        resolve();
      }
    };
    connection.addEventListener("icegatheringstatechange", changed);
  });
}

function closeStreaming(sendStop) {
  stoppedByUser = true;
  if (disconnectTimer !== null) {
    window.clearTimeout(disconnectTimer);
    disconnectTimer = null;
  }
  if (sendStop && socket && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify({ type: "stop" }));
  }
  if (peer) {
    peer.close();
    peer = null;
  }
  if (socket) {
    socket.close();
    socket = null;
  }
  if (localStream) {
    localStream.getTracks().forEach((track) => track.stop());
    localStream = null;
  }
  preview.srcObject = null;
  preview.style.display = "none";
  cameraActive.classList.add("hidden");
  startButton.disabled = pairingConsumed;
  stopButton.disabled = true;
  facing.disabled = false;
}

function handleSocketMessage(event) {
  let message;
  try {
    message = JSON.parse(event.data);
  } catch (_) {
    setStatus("PrivARcy sent an invalid connection response.");
    closeStreaming(false);
    return;
  }
  if (message.type === "state" && message.state === "streaming") {
    cameraActive.textContent = "CAMERA ACTIVE · video is being received by PrivARcy";
    setStatus("Connected. Video is being processed locally.");
  } else if (message.type === "answer" && peer) {
    peer.setRemoteDescription({ type: "answer", sdp: message.sdp })
      .then(() => setStatus("Connected. Waiting for the first video frame…"))
      .catch((error) => {
        setStatus(`Could not connect: ${error.message || error.name}`);
        closeStreaming(false);
      });
  } else if (message.type === "error") {
    setStatus(message.message || "The local connection failed. Generate a fresh QR code in PrivARcy.");
    closeStreaming(false);
  }
}

function openSignalSocket() {
  const token = location.pathname.split("/").filter(Boolean).pop();
  const socketUrl = new URL(`/signal/${encodeURIComponent(token)}`, location.origin);
  socketUrl.protocol = "wss:";
  const ws = new WebSocket(socketUrl);
  socket = ws;
  ws.onmessage = handleSocketMessage;
  ws.onclose = () => {
    if (!stoppedByUser) {
      setStatus("Connection lost. Generate a fresh QR code in PrivARcy to reconnect.");
      closeStreaming(false);
    }
  };
  return new Promise((resolve, reject) => {
    const timeout = window.setTimeout(() => {
      ws.close();
      reject(new Error("PrivARcy did not respond. Check that both devices are on the same Wi-Fi."));
    }, 10000);
    ws.onopen = () => {
      window.clearTimeout(timeout);
      resolve(ws);
    };
    ws.onerror = () => {
      window.clearTimeout(timeout);
      reject(new Error("Could not reach PrivARcy. Check Wi-Fi and allow PrivARcy through the private-network firewall."));
    };
  });
}

async function beginStreaming() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    setStatus("This browser cannot access the camera securely. Check the secure connection setup or use an up-to-date browser.");
    return;
  }
  stoppedByUser = false;
  startButton.disabled = true;
  facing.disabled = true;
  setStatus("Allow camera access when your browser asks.");
  try {
    localStream = await navigator.mediaDevices.getUserMedia(cameraConstraints());
    facing.disabled = false;
    preview.srcObject = localStream;
    preview.style.display = "block";
    cameraActive.classList.remove("hidden");
    setStatus("Camera ready. Connecting to PrivARcy…");

    const ws = await openSignalSocket();
    stopButton.disabled = false;
    peer = new RTCPeerConnection({ iceServers: [] });
    localStream.getTracks().forEach((track) => peer.addTrack(track, localStream));
    peer.onconnectionstatechange = () => {
      if (!peer || stoppedByUser) {
        return;
      }
      if (peer.connectionState === "connected") {
        if (disconnectTimer !== null) {
          window.clearTimeout(disconnectTimer);
          disconnectTimer = null;
        }
        setStatus("Connected. Waiting for the first video frame…");
      } else if (peer.connectionState === "failed" || peer.connectionState === "disconnected") {
        if (disconnectTimer === null) {
          disconnectTimer = window.setTimeout(() => {
            setStatus("Connection lost. Generate a fresh QR code in PrivARcy to reconnect.");
            closeStreaming(false);
          }, peer.connectionState === "failed" ? 0 : 3000);
        }
      }
    };

    ws.send(JSON.stringify({ type: "state", state: "camera_ready" }));
    const offer = await peer.createOffer();
    await peer.setLocalDescription(offer);
    await waitForIceGathering(peer);
    if (!peer.localDescription || !peer.localDescription.sdp) {
      throw new Error("The browser could not prepare the video connection.");
    }
    ws.send(JSON.stringify({ type: "offer", sdp: peer.localDescription.sdp }));
    pairingConsumed = true;
  } catch (error) {
    const denied = error.name === "NotAllowedError" || error.name === "PermissionDeniedError";
    setStatus(denied
      ? "Camera permission was denied. Allow camera access in browser settings and tap Start Camera again."
      : (error.message || "Could not start the phone camera."));
    closeStreaming(false);
  }
}

startButton.addEventListener("click", beginStreaming);
stopButton.addEventListener("click", () => {
  setStatus("Camera stopped. Generate a fresh QR code in PrivARcy to reconnect.");
  closeStreaming(true);
});
facing.addEventListener("change", async () => {
  if (!peer || !localStream) {
    setStatus("Camera selection changed. Tap Start Camera when ready.");
    return;
  }
  let replacement = null;
  try {
    replacement = await navigator.mediaDevices.getUserMedia(cameraConstraints());
    const newTrack = replacement.getVideoTracks()[0];
    const sender = peer.getSenders().find((item) => item.track && item.track.kind === "video");
    if (!newTrack || !sender) {
      throw new Error("The selected camera could not be switched.");
    }
    const previous = localStream;
    await sender.replaceTrack(newTrack);
    localStream = replacement;
    preview.srcObject = replacement;
    previous.getTracks().forEach((track) => track.stop());
    setStatus("Camera switched. Video continues directly to PrivARcy.");
  } catch (error) {
    if (replacement) {
      replacement.getTracks().forEach((track) => track.stop());
    }
    setStatus(`Could not switch camera: ${error.message || error.name}`);
  }
});
window.addEventListener("pagehide", () => closeStreaming(true));
