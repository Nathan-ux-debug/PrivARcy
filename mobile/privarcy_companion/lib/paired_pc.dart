import 'dart:convert';
import 'dart:math';

import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import 'package:multicast_dns/multicast_dns.dart';

class PairedPc {
  const PairedPc({
    required this.pcId,
    required this.name,
    required this.mdnsHost,
    required this.port,
    required this.caSha256,
    required this.caPem,
    required this.deviceId,
    required this.secret,
    required this.lastConnected,
  });

  final String pcId;
  final String name;
  final String mdnsHost;
  final int port;
  final String caSha256;
  final String caPem;
  final String deviceId;
  final String secret;
  final String lastConnected;

  factory PairedPc.fromJson(Map<String, dynamic> json) {
    final pcId = json['pcId'];
    final name = json['name'];
    final host = json['mdnsHost'];
    final port = json['port'];
    final fingerprint = json['caSha256'];
    final certificate = json['caPem'];
    final deviceId = json['deviceId'];
    final secret = json['secret'];
    final lastConnected = json['lastConnected'];
    if (pcId is! String ||
        !RegExp(r'^[a-f0-9]{32}$').hasMatch(pcId) ||
        name is! String ||
        name.isEmpty ||
        host is! String ||
        !RegExp(r'^privarcy-[a-f0-9]{32}\.local$').hasMatch(host) ||
        host != 'privarcy-${pcId.toLowerCase()}.local' ||
        port is! int ||
        port < 1 ||
        port > 65535 ||
        fingerprint is! String ||
        !RegExp(r'^[a-f0-9]{64}$').hasMatch(fingerprint) ||
        certificate is! String ||
        deviceId is! String ||
        !RegExp(r'^[a-f0-9]{32}$').hasMatch(deviceId) ||
        secret is! String ||
        !RegExp(r'^[A-Za-z0-9_-]{43}$').hasMatch(secret) ||
        lastConnected is! String) {
      throw const FormatException('Saved PrivARcy pairing data is invalid.');
    }
    return PairedPc(
      pcId: pcId,
      name: name,
      mdnsHost: host,
      port: port,
      caSha256: fingerprint,
      caPem: certificate,
      deviceId: deviceId,
      secret: secret,
      lastConnected: lastConnected,
    );
  }

  Map<String, Object> toJson() => {
    'pcId': pcId,
    'name': name,
    'mdnsHost': mdnsHost,
    'port': port,
    'caSha256': caSha256,
    'caPem': caPem,
    'deviceId': deviceId,
    'secret': secret,
    'lastConnected': lastConnected,
  };

  PairedPc connectedNow() => PairedPc(
    pcId: pcId,
    name: name,
    mdnsHost: mdnsHost,
    port: port,
    caSha256: caSha256,
    caPem: caPem,
    deviceId: deviceId,
    secret: secret,
    lastConnected: DateTime.now().toUtc().toIso8601String(),
  );
}

class PairedPcStore {
  PairedPcStore({FlutterSecureStorage? storage})
    : _storage =
          storage ??
          const FlutterSecureStorage(
            iOptions: IOSOptions(
              accessibility: KeychainAccessibility.unlocked_this_device,
            ),
          );

  static const _pairingsKey = 'privarcy.paired_pcs.v1';
  static const _deviceIdKey = 'privarcy.device_id.v1';
  final FlutterSecureStorage _storage;
  static final Random _random = Random.secure();

  Future<List<PairedPc>> load() async {
    final raw = await _storage.read(key: _pairingsKey);
    if (raw == null) return [];
    final decoded = jsonDecode(raw);
    if (decoded is! List) {
      throw const FormatException('Saved PrivARcy pairings are invalid.');
    }
    final pairings = decoded
        .map((entry) {
          if (entry is! Map<String, dynamic>) {
            throw const FormatException('Saved PrivARcy pairings are invalid.');
          }
          return PairedPc.fromJson(entry);
        })
        .toList(growable: true);
    final ids = pairings.map((pairing) => pairing.pcId).toSet();
    if (ids.length != pairings.length) {
      throw const FormatException(
        'Saved PrivARcy pairings contain duplicate PCs.',
      );
    }
    return pairings;
  }

  Future<String> deviceId() async {
    final existing = await _storage.read(key: _deviceIdKey);
    if (existing != null && RegExp(r'^[a-f0-9]{32}$').hasMatch(existing)) {
      return existing;
    }
    final id = _randomHex(16);
    await _storage.write(key: _deviceIdKey, value: id);
    return id;
  }

  Future<void> save(PairedPc pairing) async {
    final pairings = await load();
    final index = pairings.indexWhere((entry) => entry.pcId == pairing.pcId);
    if (index == -1) {
      pairings.add(pairing);
    } else {
      pairings[index] = pairing;
    }
    await _storage.write(
      key: _pairingsKey,
      value: jsonEncode(pairings.map((entry) => entry.toJson()).toList()),
    );
  }

  Future<void> remove(String pcId) async {
    final pairings = await load()
      ..removeWhere((entry) => entry.pcId == pcId);
    await _storage.write(
      key: _pairingsKey,
      value: jsonEncode(pairings.map((entry) => entry.toJson()).toList()),
    );
  }

  static String randomSecret() => base64Url
      .encode(List<int>.generate(32, (_) => _random.nextInt(256)))
      .replaceAll('=', '');

  static String _randomHex(int length) {
    const digits = '0123456789abcdef';
    final bytes = List<int>.generate(length, (_) => _random.nextInt(256));
    return bytes.map((byte) => digits[byte >> 4] + digits[byte & 15]).join();
  }
}

class PairedPcDiscovery {
  Future<({String host, int port})?> find(String pcId) async {
    final client = MDnsClient();
    await client.start();
    try {
      final services = await client
          .lookup<PtrResourceRecord>(
            ResourceRecordQuery.serverPointer('_privarcy._tcp.local'),
          )
          .timeout(const Duration(seconds: 5))
          .toList();
      for (final service in services) {
        final name = service.domainName;
        final records = await client
            .lookup<TxtResourceRecord>(ResourceRecordQuery.text(name))
            .timeout(const Duration(seconds: 3))
            .toList();
        if (!records.any(
          (record) =>
              record.text.split(RegExp(r'[\r\n]+')).contains('pc_id=$pcId'),
        )) {
          continue;
        }
        final endpoints = await client
            .lookup<SrvResourceRecord>(ResourceRecordQuery.service(name))
            .timeout(const Duration(seconds: 3))
            .toList();
        if (endpoints.isNotEmpty) {
          return (
            host: endpoints.first.target.replaceFirst(RegExp(r'\.$'), ''),
            port: endpoints.first.port,
          );
        }
      }
      return null;
    } finally {
      client.stop();
    }
  }
}
