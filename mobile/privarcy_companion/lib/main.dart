import 'dart:async';
import 'dart:io';
import 'dart:math';

import 'package:flutter/material.dart';
import 'package:flutter_webrtc/flutter_webrtc.dart';
import 'package:mobile_scanner/mobile_scanner.dart';

import 'paired_pc.dart';
import 'pairing_client.dart';
import 'pairing_link.dart';

void main() {
  runApp(const PrivarcyCompanionApp());
}

class PrivarcyCompanionApp extends StatelessWidget {
  const PrivarcyCompanionApp({super.key});

  @override
  Widget build(BuildContext context) {
    const navy = Color(0xFF10233F);
    return MaterialApp(
      title: 'PrivARcy Companion',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFF2867D8)),
        scaffoldBackgroundColor: const Color(0xFFF4F7FB),
        appBarTheme: const AppBarTheme(
          backgroundColor: navy,
          foregroundColor: Colors.white,
          centerTitle: false,
        ),
        useMaterial3: true,
      ),
      home: const CompanionHomePage(),
    );
  }
}

class CompanionHomePage extends StatefulWidget {
  const CompanionHomePage({super.key});

  @override
  State<CompanionHomePage> createState() => _CompanionHomePageState();
}

class _CompanionHomePageState extends State<CompanionHomePage>
    with WidgetsBindingObserver {
  final RTCVideoRenderer _renderer = RTCVideoRenderer();
  final PairedPcStore _pairingStore = PairedPcStore();
  late final Future<void> _rendererInitialization;
  PairingLink? _link;
  List<PairedPc> _pairedPcs = [];
  String? _selectedPcId;
  PairingClient? _client;
  MediaStream? _camera;
  String _status = 'Scan the QR code shown in PrivARcy on your PC.';
  bool _rendererReady = false;
  bool _starting = false;
  bool _streaming = false;
  bool _connecting = false;
  bool _useFrontCamera = false;
  bool _cancelRetry = false;
  int _attemptId = 0;
  final Random _retryRandom = Random.secure();

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _rendererInitialization = _initializeRenderer();
    unawaited(_loadPairings());
  }

  Future<void> _loadPairings() async {
    try {
      final pairings = await _pairingStore.load();
      if (!mounted) return;
      setState(() {
        _pairedPcs = pairings;
        _selectedPcId = pairings.isEmpty ? null : pairings.first.pcId;
        if (pairings.isNotEmpty) {
          _status = 'Choose a saved PrivARcy PC and connect when ready.';
        }
      });
    } catch (error) {
      if (mounted) {
        setState(
          () => _status = 'Could not read saved device pairings: $error',
        );
      }
    }
  }

  Future<void> _initializeRenderer() async {
    try {
      await _renderer.initialize();
      if (mounted) {
        setState(() => _rendererReady = true);
      }
    } catch (error) {
      if (mounted) {
        setState(() => _status = 'Camera preview could not start: $error');
      }
    }
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.paused && (_starting || _streaming)) {
      unawaited(
        _stopCamera(
          status:
              'Streaming stopped because the app went into the background. '
              'Return to the app and reconnect to the saved PC when ready.',
        ),
      );
    }
  }

  Future<void> _scanQrCode() async {
    final value = await Navigator.of(context).push<String>(
      MaterialPageRoute(builder: (_) => const PairingScannerPage()),
    );
    if (!mounted || value == null) {
      return;
    }
    try {
      final link = PairingLink.parse(value);
      await _client?.stop();
      _client = null;
      setState(() {
        _link = link;
        _selectedPcId = null;
        _status = 'QR code scanned. Choose a camera and tap Start Camera.';
      });
    } on FormatException catch (error) {
      setState(() => _status = error.message);
    }
  }

  Future<void> _startCamera() async {
    if (!_rendererReady || _starting || _streaming) {
      return;
    }
    final attemptId = ++_attemptId;
    _cancelRetry = false;
    setState(() {
      _starting = true;
      _status = 'Connecting securely before requesting camera access…';
    });

    MediaStream? camera;
    try {
      final client = await _connectSelectedPc(attemptId);
      if (!mounted || attemptId != _attemptId) return;
      setState(() => _status = 'Requesting camera permission…');
      camera = await navigator.mediaDevices.getUserMedia({
        'audio': false,
        'video': {
          'facingMode': _useFrontCamera ? 'user' : 'environment',
          'width': {'ideal': 1280, 'max': 1920},
          'height': {'ideal': 720, 'max': 1080},
          'frameRate': {'ideal': 24, 'max': 30},
        },
      });
      if (!mounted || attemptId != _attemptId) {
        await _stopTracks(camera);
        return;
      }
      _camera = camera;
      await _renderer.setSrcObject(stream: camera);
      if (!mounted || attemptId != _attemptId) return;
      await client.startStreaming(camera);
      if (!mounted || attemptId != _attemptId) {
        return;
      }
      setState(() {
        _streaming = true;
        _status = 'Connected. Video is being sent directly to PrivARcy for redaction.';
      });
    } catch (error) {
      if (camera != null && attemptId == _attemptId) {
        await _stopTracks(camera);
        await _client?.stop();
        _client = null;
      }
      if (_client?.connected != true) {
        await _client?.stop();
        _client = null;
      }
      if (attemptId == _attemptId) {
        _camera = null;
        _client = null;
        await _renderer.setSrcObject(stream: null);
      }
      if (mounted && attemptId == _attemptId) {
        setState(() {
          _streaming = false;
          _status = _friendlyError(error);
        });
      }
    } finally {
      if (mounted && attemptId == _attemptId) {
        setState(() => _starting = false);
      }
    }
  }

  Future<void> _connectSavedPc() async {
    if (_connecting) {
      _cancelRetry = true;
      ++_attemptId;
      await _client?.stop();
      _client = null;
      if (mounted) {
        setState(() {
          _connecting = false;
          _status = 'Connection attempt cancelled.';
        });
      }
      return;
    }
    if (_selectedPcId == null || _starting || _streaming) return;
    final attemptId = ++_attemptId;
    _cancelRetry = false;
    setState(() => _connecting = true);
    try {
      await _connectSelectedPc(attemptId);
    } catch (error) {
      if (mounted && attemptId == _attemptId) {
        setState(() => _status = _friendlyError(error));
      }
    } finally {
      if (mounted && attemptId == _attemptId) {
        setState(() => _connecting = false);
      }
    }
  }

  Future<PairingClient> _connectSelectedPc(int attemptId) async {
    if (_client?.connected == true) return _client!;
    final link = _link;
    if (link != null) {
      final client = PairingClient.forQr(
        link: link,
        deviceId: await _pairingStore.deviceId(),
        deviceName: Platform.isIOS ? 'iPhone or iPad' : 'Android phone',
        pairingSecret: PairedPcStore.randomSecret(),
        onStatus: _handleClientStatus,
        onPaired: _savePairedPc,
      );
      _client = client;
      await client.connect();
      if (attemptId != _attemptId || _cancelRetry) {
        await client.stop();
        throw StateError('Connection attempt cancelled.');
      }
      _link = null;
      return client;
    }

    final selectedIndex = _pairedPcs.indexWhere(
      (pc) => pc.pcId == _selectedPcId,
    );
    if (selectedIndex == -1) {
      throw const FormatException('Scan a new QR code or choose a saved PC.');
    }
    final savedPc = _pairedPcs[selectedIndex];
    Object? lastError;
    for (var attempt = 0; attempt < 5; attempt++) {
      if (attemptId != _attemptId || _cancelRetry) {
        throw StateError('Connection attempt cancelled.');
      }
      final client = PairingClient.forSaved(
        savedPc: savedPc,
        onStatus: _handleClientStatus,
        onPaired: _savePairedPc,
      );
      _client = client;
      try {
        await client.connect();
        return client;
      } catch (error) {
        await client.stop();
        _client = null;
        lastError = error;
        if (error is FormatException || attempt == 4) rethrow;
        final baseDelay = 1 << attempt;
        final jitter = 0.75 + _retryRandom.nextDouble() * 0.5;
        _setStatus(
          'PC not reachable. Retrying (${attempt + 1}/5); you can cancel.',
        );
        await Future<void>.delayed(
          Duration(milliseconds: (baseDelay * jitter * 1000).round()),
        );
      }
    }
    throw lastError ?? const SocketException('Could not reach the paired PC.');
  }

  Future<void> _savePairedPc(PairedPc pc) async {
    await _pairingStore.save(pc);
    if (!mounted) return;
    setState(() {
      _pairedPcs = [
        pc,
        ..._pairedPcs.where((existing) => existing.pcId != pc.pcId),
      ];
      _selectedPcId = pc.pcId;
    });
  }

  Future<void> _forgetSelectedPc() async {
    if (_selectedPcId == null || _starting || _streaming || _connecting) return;
    final pcId = _selectedPcId!;
    final index = _pairedPcs.indexWhere((pc) => pc.pcId == pcId);
    if (index == -1) return;
    final savedPc = _pairedPcs[index];
    PairingClient? client = _client?.connected == true ? _client : null;
    try {
      client ??= PairingClient.forSaved(
        savedPc: savedPc,
        onStatus: _handleClientStatus,
        onPaired: _savePairedPc,
      );
      if (!client.connected) await client.connect();
      await client.revoke();
      await _pairingStore.remove(pcId);
      if (mounted) {
        setState(() {
          _client = null;
          _pairedPcs.removeWhere((pc) => pc.pcId == pcId);
          _selectedPcId = _pairedPcs.isEmpty ? null : _pairedPcs.first.pcId;
          _status = 'The PC pairing was revoked on both devices.';
        });
      }
    } catch (error) {
      if (client != _client) await client?.stop();
      if (mounted) {
        setState(() {
          _status =
              'Could not revoke this pairing. Keep it saved and check that the PC is available. ${_friendlyError(error)}';
        });
      }
    }
  }

  void _setStatus(String status) {
    if (mounted) setState(() => _status = status);
  }

  void _handleClientStatus(String status) {
    _setStatus(status);
    if (status.startsWith('Connection lost') && (_streaming || _starting)) {
      unawaited(
        _stopCamera(
          status: 'Connection lost. Camera stopped; reconnect and tap Start Camera.',
        ),
      );
    }
  }

  String _friendlyError(Object error) {
    final message = error.toString();
    if (message.contains('NotAllowedError') ||
        message.contains('PermissionDenied') ||
        message.contains('permission')) {
      return 'Camera permission was denied. Allow camera access in phone settings, then try again.';
    }
    if (message.contains('timed out') || message.contains('TimeoutException')) {
      return 'PrivARcy did not respond. Check Wi-Fi and Windows Firewall, then retry or scan a fresh QR code.';
    }
    if (message.contains('certificate') || message.contains('TLS')) {
      return 'The secure certificate could not be verified. Do not continue; generate a new QR code in PrivARcy.';
    }
    if (message.contains('not currently discoverable')) {
      return 'The saved PC is not on this local network. Check that PrivARcy is open, or use QR re-pairing.';
    }
    if (message.contains('SocketException') ||
        message.contains('WebSocketException')) {
      return 'Could not reach PrivARcy. Check that both devices are on the same Wi-Fi and allow Private network access.';
    }
    return 'Could not start the phone camera: $message';
  }

  Future<void> _stopCamera({String? status}) async {
    ++_attemptId;
    final client = _client;
    final camera = _camera;
    _client = null;
    _camera = null;
    await client?.stop();
    if (camera != null) {
      await _stopTracks(camera);
    }
    if (_rendererReady) {
      await _renderer.setSrcObject(stream: null);
    }
    if (mounted) {
      setState(() {
        _starting = false;
        _connecting = false;
        _streaming = false;
        _status =
            status ??
            'Camera stopped. Reconnect to the saved PC when you are ready.';
      });
    }
  }

  Future<void> _stopTracks(MediaStream stream) async {
    for (final track in stream.getTracks()) {
      await track.stop();
    }
    await stream.dispose();
  }

  Future<void> _switchCamera() async {
    _useFrontCamera = !_useFrontCamera;
    if (_camera != null) {
      try {
        final tracks = _camera!.getVideoTracks();
        if (tracks.isEmpty || !await Helper.switchCamera(tracks.first)) {
          throw StateError('The selected camera is unavailable.');
        }
      } catch (error) {
        _useFrontCamera = !_useFrontCamera;
        if (mounted) {
          setState(() => _status = 'Could not switch camera: $error');
        }
        return;
      }
    }
    if (mounted) {
      setState(() {
        _status = _useFrontCamera
            ? 'Front camera selected.'
            : 'Rear camera selected.';
      });
    }
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    unawaited(_disposeResources());
    super.dispose();
  }

  Future<void> _disposeResources() async {
    await _stopCamera();
    await _rendererInitialization;
    await _renderer.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final active = _streaming || _starting;
    return Scaffold(
      appBar: AppBar(
        title: const Text('PrivARcy Companion'),
        actions: [
          IconButton(
            tooltip: 'Scan QR Code',
            onPressed: active || _connecting ? null : _scanQrCode,
            icon: const Icon(Icons.qr_code_scanner),
          ),
        ],
      ),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            children: [
              _StatusCard(
                status: _status,
                paired: _pairedPcs.isNotEmpty || _link != null,
              ),
              if (_pairedPcs.isNotEmpty) ...[
                const SizedBox(height: 10),
                DropdownButtonFormField<String>(
                  initialValue: _selectedPcId,
                  decoration: const InputDecoration(
                    labelText: 'Saved PrivARcy PC',
                    border: OutlineInputBorder(),
                    isDense: true,
                  ),
                  items: [
                    for (final pc in _pairedPcs)
                      DropdownMenuItem<String>(
                        value: pc.pcId,
                        child: Text(pc.name, overflow: TextOverflow.ellipsis),
                      ),
                  ],
                  onChanged: active || _connecting
                      ? null
                      : (pcId) async {
                          if (_client != null) {
                            await _client!.stop();
                            _client = null;
                          }
                          if (mounted) {
                            setState(() {
                              _selectedPcId = pcId;
                              _link = null;
                            });
                          }
                        },
                ),
                const SizedBox(height: 6),
                Row(
                  children: [
                    Expanded(
                      child: OutlinedButton.icon(
                        onPressed: active || _starting ? null : _connectSavedPc,
                        icon: Icon(
                          _connecting ? Icons.cancel_outlined : Icons.link,
                        ),
                        label: Text(_connecting ? 'Cancel' : 'Connect to PC'),
                      ),
                    ),
                    const SizedBox(width: 8),
                    TextButton(
                      onPressed: active || _connecting
                          ? null
                          : _forgetSelectedPc,
                      child: const Text('Revoke'),
                    ),
                  ],
                ),
              ],
              if (_link != null)
                Padding(
                  padding: const EdgeInsets.only(top: 8),
                  child: Text(
                    'New PC pairing: ${_link!.pcId.substring(0, 6)}. Start Camera to securely save this PC.',
                    style: const TextStyle(
                      color: Color(0xFF53657F),
                      fontSize: 12,
                    ),
                  ),
                ),
              const SizedBox(height: 12),
              const Align(
                alignment: Alignment.centerLeft,
                child: Text(
                  'Phone camera preview · redacted output appears in PrivARcy on your PC',
                  style: TextStyle(color: Color(0xFF53657F), fontSize: 12),
                ),
              ),
              const SizedBox(height: 6),
              Expanded(
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(18),
                  child: ColoredBox(
                    color: const Color(0xFF101820),
                    child: _camera != null && _rendererReady
                        ? RTCVideoView(
                            _renderer,
                            mirror: _useFrontCamera,
                            objectFit: RTCVideoViewObjectFit
                                .RTCVideoViewObjectFitCover,
                          )
                        : const Center(
                            child: Icon(
                              Icons.videocam_outlined,
                              color: Colors.white54,
                              size: 64,
                            ),
                          ),
                  ),
                ),
              ),
              const SizedBox(height: 12),
              Row(
                children: [
                  Expanded(
                    child: OutlinedButton.icon(
                      onPressed: active ? null : _scanQrCode,
                      icon: const Icon(Icons.qr_code_scanner),
                      label: const Text('Scan QR Code'),
                    ),
                  ),
                  const SizedBox(width: 10),
                  Expanded(
                    child: OutlinedButton.icon(
                      onPressed: _link == null && _selectedPcId == null
                          ? null
                          : _switchCamera,
                      icon: const Icon(Icons.cameraswitch),
                      label: Text(
                        _useFrontCamera ? 'Front camera' : 'Rear camera',
                      ),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 8),
              SizedBox(
                width: double.infinity,
                child: FilledButton.icon(
                  onPressed: _streaming || _starting
                      ? () => _stopCamera()
                      : (_link == null && _selectedPcId == null ||
                                !_rendererReady ||
                                _connecting
                            ? null
                            : _startCamera),
                  icon: Icon(active ? Icons.stop : Icons.videocam),
                  label: Text(active ? 'Stop Camera' : 'Start Camera'),
                ),
              ),
              const SizedBox(height: 6),
              const Text(
                'Keep PrivARcy open on your PC. Video is sent over your local Wi-Fi; '
                'redaction happens on the PC.',
                textAlign: TextAlign.center,
                style: TextStyle(color: Color(0xFF53657F), fontSize: 12),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _StatusCard extends StatelessWidget {
  const _StatusCard({required this.status, required this.paired});

  final String status;
  final bool paired;

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: const Color(0xFFDCE4EF)),
      ),
      child: Row(
        children: [
          Icon(
            paired ? Icons.link : Icons.link_off,
            color: paired ? const Color(0xFF23764D) : const Color(0xFF64748B),
          ),
          const SizedBox(width: 10),
          Expanded(
            child: Text(status, style: Theme.of(context).textTheme.bodyMedium),
          ),
        ],
      ),
    );
  }
}

class PairingScannerPage extends StatefulWidget {
  const PairingScannerPage({super.key});

  @override
  State<PairingScannerPage> createState() => _PairingScannerPageState();
}

class _PairingScannerPageState extends State<PairingScannerPage> {
  final MobileScannerController _scanner = MobileScannerController(
    autoStart: true,
    formats: const [BarcodeFormat.qrCode],
    detectionSpeed: DetectionSpeed.noDuplicates,
  );
  bool _handled = false;

  void _onDetect(BarcodeCapture capture) {
    if (_handled || capture.barcodes.isEmpty) {
      return;
    }
    final value = capture.barcodes.first.rawValue;
    if (value == null) {
      return;
    }
    _handled = true;
    unawaited(_scanner.stop());
    Navigator.of(context).pop(value);
  }

  @override
  void dispose() {
    unawaited(_scanner.dispose());
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Scan PrivARcy QR Code')),
      body: Stack(
        fit: StackFit.expand,
        children: [
          MobileScanner(controller: _scanner, onDetect: _onDetect),
          IgnorePointer(
            child: Center(
              child: Container(
                width: 260,
                height: 260,
                decoration: BoxDecoration(
                  border: Border.all(color: Colors.white, width: 3),
                  borderRadius: BorderRadius.circular(20),
                ),
              ),
            ),
          ),
          const Positioned(
            left: 24,
            right: 24,
            bottom: 36,
            child: Text(
              'Point your phone at the QR code shown in PrivARcy on your PC.',
              textAlign: TextAlign.center,
              style: TextStyle(
                color: Colors.white,
                fontWeight: FontWeight.w600,
              ),
            ),
          ),
        ],
      ),
    );
  }
}
