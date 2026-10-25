import 'package:flutter_test/flutter_test.dart';
import 'package:privarcy_companion/pairing_link.dart';

void main() {
  const token = 'A123456789012345678901234567890123456789012';
  const pin =
      '0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef';
  const pcId = '0123456789abcdef0123456789abcdef';
  const qr =
      'http://192.168.1.20:8100/pair/$token?ca_sha256=$pin&https_port=8101&pc_id=$pcId&mdns_host=privarcy-$pcId.local';

  test('parses local QR code into pinned HTTPS and WSS endpoints', () {
    final link = PairingLink.parse(qr);

    expect(link.token, token);
    expect(link.caSha256, pin);
    expect(link.pcId, pcId);
    expect(link.mdnsHost, 'privarcy-$pcId.local');
    expect(link.caUri.toString(), 'http://192.168.1.20:8100/ca/$token');
    expect(link.httpsUri.toString(), 'https://192.168.1.20:8101/pair/$token');
    expect(
      link.signalingUri.toString(),
      'wss://192.168.1.20:8101/signal/$token',
    );
    expect(link.origin, 'https://192.168.1.20:8101');
  });

  test('rejects QR values without a private address and certificate pin', () {
    expect(
      () => PairingLink.parse(qr.replaceFirst('192.168.1.20', 'example.com')),
      throwsFormatException,
    );
    expect(
      () => PairingLink.parse('http://192.168.1.20:8100/pair/$token'),
      throwsFormatException,
    );
    expect(
      () => PairingLink.parse(
        qr.replaceFirst('https_port=8101', 'https_port=70000'),
      ),
      throwsFormatException,
    );
    expect(
      () => PairingLink.parse(qr.replaceFirst('pc_id=$pcId', 'pc_id=bad')),
      throwsFormatException,
    );
  });
}
