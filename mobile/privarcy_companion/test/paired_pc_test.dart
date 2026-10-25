import 'package:flutter_test/flutter_test.dart';
import 'package:privarcy_companion/paired_pc.dart';

void main() {
  const pcId = '0123456789abcdef0123456789abcdef';
  final secret = List.filled(43, 'A').join();
  final pairing = <String, dynamic>{
    'pcId': pcId,
    'name': 'My PC',
    'mdnsHost': 'privarcy-$pcId.local',
    'port': 45678,
    'caSha256': 'a' * 64,
    'caPem': 'certificate',
    'deviceId': 'abcdef0123456789abcdef0123456789',
    'secret': secret,
    'lastConnected': '',
  };

  test('loads a saved PC whose discovery name matches its identity', () {
    final saved = PairedPc.fromJson(pairing);

    expect(saved.pcId, pcId);
    expect(saved.mdnsHost, 'privarcy-$pcId.local');
  });

  test('rejects a discovery name bound to a different PC identity', () {
    final otherPcId = List.filled(32, 'f').join();
    final tampered = {...pairing, 'mdnsHost': 'privarcy-$otherPcId.local'};

    expect(() => PairedPc.fromJson(tampered), throwsFormatException);
  });
}
