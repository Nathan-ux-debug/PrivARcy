import 'dart:io';

class PairingLink {
  const PairingLink._({
    required this.httpUri,
    required this.httpsUri,
    required this.token,
    required this.caSha256,
    required this.pcId,
    required this.mdnsHost,
  });

  final Uri httpUri;
  final Uri httpsUri;
  final String token;
  final String caSha256;
  final String pcId;
  final String mdnsHost;

  Uri get caUri => Uri(
    scheme: httpUri.scheme,
    host: httpUri.host,
    port: httpUri.port,
    path: '/ca/$token',
  );
  Uri get signalingUri => Uri(
    scheme: 'wss',
    host: httpsUri.host,
    port: httpsUri.port,
    path: '/signal/$token',
  );

  String get origin => 'https://${httpsUri.authority}';

  static PairingLink parse(String value) {
    final uri = Uri.tryParse(value);
    if (uri == null ||
        uri.scheme != 'http' ||
        uri.userInfo.isNotEmpty ||
        uri.hasFragment ||
        uri.pathSegments.length != 2 ||
        uri.pathSegments.first != 'pair') {
      throw const FormatException('This is not a PrivARcy pairing QR code.');
    }

    final address = InternetAddress.tryParse(uri.host);
    if (address == null ||
        address.type != InternetAddressType.IPv4 ||
        !_isPrivateIpv4(address)) {
      throw const FormatException(
        'Pairing QR codes must point to a private local IPv4 address.',
      );
    }

    final token = uri.pathSegments.last;
    if (!RegExp(r'^[A-Za-z0-9_-]{40,}$').hasMatch(token)) {
      throw const FormatException('The pairing code is invalid or incomplete.');
    }

    final fingerprints = uri.queryParametersAll['ca_sha256'];
    if (fingerprints == null ||
        fingerprints.length != 1 ||
        !RegExp(r'^[a-fA-F0-9]{64}$').hasMatch(fingerprints.single)) {
      throw const FormatException(
        'The pairing code is missing its secure certificate pin.',
      );
    }

    final securePorts = uri.queryParametersAll['https_port'];
    final securePort = securePorts?.length == 1
        ? int.tryParse(securePorts!.single)
        : null;
    if (securePort == null || securePort < 1 || securePort > 65535) {
      throw const FormatException(
        'The pairing code contains an invalid secure connection port.',
      );
    }

    final pcIds = uri.queryParametersAll['pc_id'];
    final pcId = pcIds?.length == 1 ? pcIds!.single : null;
    if (pcId == null || !RegExp(r'^[a-fA-F0-9]{32}$').hasMatch(pcId)) {
      throw const FormatException(
        'The pairing code is missing the PC identity needed for persistent pairing.',
      );
    }
    final mdnsHosts = uri.queryParametersAll['mdns_host'];
    final mdnsHost = mdnsHosts?.length == 1 ? mdnsHosts!.single : null;
    if (mdnsHost == null ||
        !RegExp(r'^privarcy-[a-fA-F0-9]{32}\.local$').hasMatch(mdnsHost) ||
        mdnsHost != 'privarcy-${pcId.toLowerCase()}.local') {
      throw const FormatException(
        'The pairing code contains an invalid local PC discovery name.',
      );
    }

    return PairingLink._(
      httpUri: uri,
      httpsUri: Uri(
        scheme: 'https',
        host: uri.host,
        port: securePort,
        path: uri.path,
      ),
      token: token,
      caSha256: fingerprints.single.toLowerCase(),
      pcId: pcId.toLowerCase(),
      mdnsHost: mdnsHost.toLowerCase(),
    );
  }

  static bool _isPrivateIpv4(InternetAddress address) {
    final parts = address.address
        .split('.')
        .map(int.parse)
        .toList(growable: false);
    return parts[0] == 10 ||
        (parts[0] == 172 && parts[1] >= 16 && parts[1] <= 31) ||
        (parts[0] == 192 && parts[1] == 168);
  }
}
