import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:crypto/crypto.dart';
import 'package:flutter_webrtc/flutter_webrtc.dart';

import 'paired_pc.dart';
import 'pairing_link.dart';

class PairingClient {
  PairingClient.forQr({
    required this.link,
    required this.deviceId,
    required this.deviceName,
    required this.pairingSecret,
    required this.onStatus,
    required this.onPaired,
  }) : savedPc = null;

  PairingClient.forSaved({
    required PairedPc savedPc,
    required this.onStatus,
    required this.onPaired,
  }) : savedPc = savedPc,
       link = null,
       deviceId = savedPc.deviceId,
       deviceName = savedPc.name,
       pairingSecret = savedPc.secret;

  final PairingLink? link;
  final PairedPc? savedPc;
  final String deviceId;
  final String deviceName;
  final String pairingSecret;
  final void Function(String status) onStatus;
  final Future<void> Function(PairedPc pc) onPaired;

  WebSocket? _socket;
  HttpClient? _httpClient;
  HttpClient? _downloadClient;
  RTCPeerConnection? _peer;
  StreamSubscription<dynamic>? _socketSubscription;
  Completer<Map<String, dynamic>>? _responseCompleter;
  String? _expectedResponse;
  bool _stopped = false;
  bool _connected = false;

  bool get connected => _connected && !_stopped;

  Future<void> connect() async {
    _ensureActive();
    final pendingLink = link;
    final trustedPc = savedPc;
    if ((pendingLink == null) == (trustedPc == null)) {
      throw StateError('Choose either a new QR pairing or a saved PC.');
    }

    onStatus(
      pendingLink == null
          ? 'Finding the paired PrivARcy PC on the local network…'
          : 'Verifying the PC certificate from the pairing QR…',
    );
    final List<int> certificate;
    final Uri signalingUri;
    final String origin;
    if (pendingLink != null) {
      certificate = await _downloadPinnedCertificate(pendingLink);
      signalingUri = pendingLink.signalingUri;
      origin = pendingLink.origin;
    } else {
      final saved = trustedPc!;
      final endpoint = await PairedPcDiscovery().find(saved.pcId);
      if (endpoint == null) {
        throw const SocketException(
          'The paired PrivARcy PC is not currently discoverable on this local network.',
        );
      }
      certificate = _validateSavedCertificate(saved);
      signalingUri = Uri(
        scheme: 'wss',
        host: endpoint.host,
        port: endpoint.port,
        path: '/device/${saved.pcId}/${saved.deviceId}',
      );
      origin = 'https://${signalingUri.authority}';
    }
    _ensureActive();

    final context = SecurityContext(withTrustedRoots: false)
      ..setTrustedCertificatesBytes(certificate);
    _httpClient = HttpClient(context: context)
      ..connectionTimeout = const Duration(seconds: 8);

    if (pendingLink == null) {
      _expect('challenge');
    }
    _socket = await WebSocket.connect(
      signalingUri.toString(),
      customClient: _httpClient,
      headers: {'Origin': origin},
    ).timeout(const Duration(seconds: 10));
    _ensureActive();
    _socketSubscription = _socket!.listen(
      _handleMessage,
      onError: (Object error) {
        _failResponse(
          const SocketException(
            'The secure connection to PrivARcy was interrupted.',
          ),
        );
        if (!_stopped) {
          _connected = false;
          onStatus('Connection lost. Reconnect to your paired PC.');
        }
      },
      onDone: () {
        if (!_stopped) {
          _connected = false;
          _failResponse(
            const SocketException('PrivARcy closed the phone connection.'),
          );
          onStatus('Connection lost. Reconnect to your paired PC.');
        }
      },
      cancelOnError: true,
    );

    final PairedPc pairing;
    if (pendingLink != null) {
      pairing = await _enroll(pendingLink, certificate);
    } else {
      pairing = await _authenticateSavedPc(trustedPc!);
    }
    _ensureActive();
    _connected = true;
    await onPaired(pairing.connectedNow());
    onStatus(
      'Securely connected to ${pairing.name}. Start the camera when ready.',
    );
  }

  Future<PairedPc> _enroll(
    PairingLink pendingLink,
    List<int> certificate,
  ) async {
    final pem = utf8.decode(certificate);
    final deviceSecret = base64Url.normalize(pairingSecret);
    _expect('paired');
    _socket!.add(
      jsonEncode({
        'type': 'enroll',
        'device_id': deviceId,
        'device_name': deviceName,
        'secret': deviceSecret.replaceAll('=', ''),
      }),
    );
    final response = await _waitForResponse();
    if (response['device_id'] != deviceId ||
        response['pc_id'] != pendingLink.pcId ||
        response['mdns_host'] != pendingLink.mdnsHost ||
        response['https_port'] != pendingLink.httpsUri.port) {
      throw const FormatException(
        'The paired PC returned identity details that do not match the QR code.',
      );
    }
    return PairedPc(
      pcId: pendingLink.pcId,
      name: 'PrivARcy PC ${pendingLink.pcId.substring(0, 6)}',
      mdnsHost: pendingLink.mdnsHost,
      port: pendingLink.httpsUri.port,
      caSha256: pendingLink.caSha256,
      caPem: pem,
      deviceId: deviceId,
      secret: deviceSecret,
      lastConnected: '',
    );
  }

  Future<PairedPc> _authenticateSavedPc(PairedPc saved) async {
    final challenge = await _waitForResponse();
    if (challenge['pc_id'] != saved.pcId) {
      throw const FormatException(
        'The discovered PC identity does not match this pairing.',
      );
    }
    final serverNonce = challenge['nonce'];
    if (serverNonce is! String ||
        !RegExp(r'^[A-Za-z0-9_-]{43}$').hasMatch(serverNonce)) {
      throw const FormatException(
        'The paired PC sent an invalid authentication challenge.',
      );
    }
    final clientNonce = PairedPcStore.randomSecret();
    final secret = base64Url.decode(base64Url.normalize(saved.secret));
    final challengeText =
        'privarcy-device-auth-v1\n${saved.pcId}\n${saved.deviceId}\n$serverNonce\n$clientNonce';
    final proof = Hmac(
      sha256,
      secret,
    ).convert(utf8.encode(challengeText)).toString();
    _expect('authenticated');
    _socket!.add(
      jsonEncode({'type': 'auth', 'client_nonce': clientNonce, 'proof': proof}),
    );
    final response = await _waitForResponse();
    final expectedServerProof = Hmac(
      sha256,
      secret,
    ).convert(utf8.encode('server\n$challengeText')).toString();
    if (response['proof'] is! String ||
        !constantTimeEquals(response['proof'] as String, expectedServerProof)) {
      throw const FormatException('The paired PC could not be authenticated.');
    }
    return saved;
  }

  Future<void> startStreaming(MediaStream camera) async {
    _ensureActive();
    if (!_connected || _socket?.readyState != WebSocket.open) {
      throw const SocketException(
        'Connect to a paired PrivARcy PC before starting the camera.',
      );
    }
    if (_peer != null) {
      throw StateError('The phone camera stream is already active.');
    }

    onStatus('Starting encrypted video connection…');
    _peer = await createPeerConnection({
      'iceServers': <Map<String, String>>[],
      'sdpSemantics': 'unified-plan',
    });
    if (_stopped) {
      await _peer!.close();
      await _peer!.dispose();
      _peer = null;
      _ensureActive();
    }
    final gathered = Completer<void>();
    _peer!.onIceGatheringState = (state) {
      if (state == RTCIceGatheringState.RTCIceGatheringStateComplete &&
          !gathered.isCompleted) {
        gathered.complete();
      }
    };
    _peer!.onConnectionState = (state) {
      if (state == RTCPeerConnectionState.RTCPeerConnectionStateConnected) {
        onStatus('Connected to PrivARcy. Redaction is running on the PC.');
      } else if (state ==
          RTCPeerConnectionState.RTCPeerConnectionStateDisconnected) {
        onStatus(
          'Connection interrupted. Check that both devices are on Wi-Fi.',
        );
      } else if (state == RTCPeerConnectionState.RTCPeerConnectionStateFailed) {
        onStatus('Could not connect. Check Wi-Fi and Windows Firewall.');
      }
    };

    for (final track in camera.getVideoTracks()) {
      await _peer!.addTrack(track, camera);
    }
    _ensureActive();
    _socket!.add(jsonEncode({'type': 'state', 'state': 'camera_ready'}));
    final offer = await _peer!.createOffer();
    await _peer!.setLocalDescription(offer);
    await gathered.future.timeout(
      const Duration(seconds: 12),
      onTimeout: () =>
          throw TimeoutException('The local video connection timed out.'),
    );
    final localDescription = await _peer!.getLocalDescription();
    if (localDescription?.sdp == null || localDescription!.sdp!.isEmpty) {
      throw const FormatException(
        'The phone could not prepare a video connection.',
      );
    }

    _expect('answer');
    _socket!.add(jsonEncode({'type': 'offer', 'sdp': localDescription.sdp}));
    final answer = await _waitForResponse();
    final sdp = answer['sdp'];
    if (sdp is! String || sdp.isEmpty) {
      throw const FormatException('PrivARcy returned an invalid video answer.');
    }
    await _peer!.setRemoteDescription(RTCSessionDescription(sdp, 'answer'));
  }

  Future<void> revoke() async {
    if (!_connected || _socket?.readyState != WebSocket.open) {
      throw const SocketException(
        'Connect to the paired PC before revoking it.',
      );
    }
    _expect('revoked');
    _socket!.add(jsonEncode({'type': 'revoke'}));
    await _waitForResponse();
    await stop();
  }

  Future<List<int>> _downloadPinnedCertificate(PairingLink pairingLink) async {
    final downloadClient = HttpClient()
      ..connectionTimeout = const Duration(seconds: 8);
    _downloadClient = downloadClient;
    try {
      final request = await downloadClient
          .getUrl(pairingLink.caUri)
          .timeout(const Duration(seconds: 8));
      try {
        request.followRedirects = false;
        final response = await request.close().timeout(
          const Duration(seconds: 8),
        );
        final bytes = await response.fold<List<int>>(<int>[], (list, chunk) {
          if (list.length + chunk.length > 16384) {
            throw const FormatException(
              'The pairing certificate is unexpectedly large.',
            );
          }

          return list..addAll(chunk);
        });
        if (response.statusCode != HttpStatus.ok) {
          throw const SocketException(
            'The pairing certificate could not be downloaded.',
          );
        }
        final pem = utf8.decode(bytes);
        final encoded = pem.replaceAll(
          RegExp(r'-----BEGIN CERTIFICATE-----|-----END CERTIFICATE-----|\s'),
          '',
        );
        final der = base64.decode(encoded);
        final actualPin = sha256.convert(der).toString();
        if (actualPin != pairingLink.caSha256) {
          throw const TlsException(
            'The certificate does not match this QR code. Do not continue; generate a new code.',
          );
        }
        return bytes;
      } finally {
        request.abort();
      }
    } finally {
      downloadClient.close(force: true);
      _downloadClient = null;
    }
  }

  List<int> _validateSavedCertificate(PairedPc saved) {
    final pem = utf8.encode(saved.caPem);
    final encoded = saved.caPem.replaceAll(
      RegExp(r'-----BEGIN CERTIFICATE-----|-----END CERTIFICATE-----|\s'),
      '',
    );
    try {
      if (sha256.convert(base64.decode(encoded)).toString() != saved.caSha256) {
        throw const FormatException(
          'The saved PC certificate does not match its pairing fingerprint.',
        );
      }
    } on FormatException {
      rethrow;
    } on ArgumentError {
      throw const FormatException('The saved PC certificate is invalid.');
    }
    return pem;
  }

  void _expect(String type) {
    _expectedResponse = type;
    _responseCompleter = Completer<Map<String, dynamic>>();
  }

  Future<Map<String, dynamic>> _waitForResponse() async {
    final completer = _responseCompleter;
    if (completer == null) {
      throw StateError('No signaling response was expected.');
    }
    try {
      return await completer.future.timeout(const Duration(seconds: 12));
    } finally {
      if (identical(_responseCompleter, completer)) {
        _responseCompleter = null;
        _expectedResponse = null;
      }
    }
  }

  void _ensureActive() {
    if (_stopped) {
      throw StateError('The phone camera connection was cancelled.');
    }
  }

  void _failResponse(Object error) {
    final completer = _responseCompleter;
    if (completer != null && !completer.isCompleted) {
      completer.completeError(error);
    }
  }

  void _handleMessage(dynamic raw) {
    if (raw is! String) return;
    final dynamic decoded;
    try {
      decoded = jsonDecode(raw);
    } on FormatException {
      _failResponse(
        const FormatException('PrivARcy sent an invalid connection response.'),
      );
      return;
    }
    if (decoded is! Map<String, dynamic>) return;
    final type = decoded['type'];
    if (type == 'error') {
      final message = decoded['message'];
      _failResponse(
        SocketException(
          message is String ? message : 'The local pairing was rejected.',
        ),
      );
    } else if (type == _expectedResponse &&
        _responseCompleter != null &&
        !_responseCompleter!.isCompleted) {
      _responseCompleter!.complete(decoded);
    } else if (type == 'state' && decoded['state'] == 'streaming') {
      onStatus('Video is arriving at PrivARcy for local redaction.');
    }
  }

  Future<void> stop() async {
    final wasStopped = _stopped;
    _stopped = true;
    _connected = false;
    _downloadClient?.close(force: true);
    _downloadClient = null;
    _httpClient?.close(force: true);
    _httpClient = null;
    final socket = _socket;
    _socket = null;
    if (!wasStopped && socket?.readyState == WebSocket.open) {
      socket!.add(jsonEncode({'type': 'stop'}));
    }
    await _socketSubscription?.cancel();
    _socketSubscription = null;
    final peer = _peer;
    _peer = null;
    await peer?.close();
    await peer?.dispose();
    await socket?.close();
    _failResponse(const SocketException('The paired connection was closed.'));
    _responseCompleter = null;
    _expectedResponse = null;
  }
}

bool constantTimeEquals(String first, String second) {
  if (first.length != second.length) return false;
  var difference = 0;
  for (var index = 0; index < first.length; index++) {
    difference |= first.codeUnitAt(index) ^ second.codeUnitAt(index);
  }
  return difference == 0;
}
