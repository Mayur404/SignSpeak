"""Self-signed HTTPS support for the web UI.

Browsers only expose the microphone (getUserMedia) on secure origins:
https:// or http://localhost. When the dashboard is opened from another
machine as http://<vm-ip>:<port>, voice input is unavailable. Serving over
HTTPS with a self-signed certificate fixes that after the user accepts the
browser's one-time certificate warning.
"""
from __future__ import annotations

import datetime
import ipaddress
import shutil
import ssl
import subprocess
import tempfile
from pathlib import Path

from ..paths import DATA_DIR

CERT_DIR = DATA_DIR / "certs"
CERT_FILE = CERT_DIR / "signspeak-cert.pem"
KEY_FILE = CERT_DIR / "signspeak-key.pem"
# The 5G edge VM address the dashboard is usually opened on.
DEFAULT_EXTRA_IPS = ("172.16.0.11",)


def _san_entries(ips: list[str]) -> tuple[list[str], list[str]]:
    """Return (dns_names, ip_addresses) for the certificate's subjectAltName."""
    seen: list[str] = []
    for ip in [*ips, "127.0.0.1", *DEFAULT_EXTRA_IPS]:
        try:
            ipaddress.ip_address(ip)
        except ValueError:
            continue
        if ip not in seen:
            seen.append(ip)
    return ["localhost"], seen


def _generate_with_openssl(dns: list[str], ips: list[str]) -> bool:
    openssl = shutil.which("openssl")
    if not openssl:
        return False
    san = ",".join([f"DNS:{d}" for d in dns] + [f"IP:{i}" for i in ips])
    # Pass SAN via a config file: older OpenSSL builds lack -addext.
    config = (
        "[req]\ndistinguished_name=dn\nx509_extensions=ext\nprompt=no\n"
        "[dn]\nCN=SignSpeak\n"
        f"[ext]\nsubjectAltName={san}\nbasicConstraints=CA:FALSE\n"
        "keyUsage=digitalSignature,keyEncipherment\nextendedKeyUsage=serverAuth\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".cnf", delete=False) as cfg:
        cfg.write(config)
        cfg_path = cfg.name
    try:
        result = subprocess.run(
            [openssl, "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-sha256",
             "-days", "825", "-keyout", str(KEY_FILE), "-out", str(CERT_FILE),
             "-config", cfg_path],
            capture_output=True, text=True, timeout=60,
        )
        return result.returncode == 0 and CERT_FILE.exists() and KEY_FILE.exists()
    except (OSError, subprocess.SubprocessError):
        return False
    finally:
        Path(cfg_path).unlink(missing_ok=True)


def _generate_with_cryptography(dns: list[str], ips: list[str]) -> bool:
    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID
    except ImportError:
        return False

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "SignSpeak")])
    now = datetime.datetime.now(datetime.timezone.utc)
    alt_names = [x509.DNSName(d) for d in dns] + [x509.IPAddress(ipaddress.ip_address(i)) for i in ips]
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(days=825))
        .add_extension(x509.SubjectAlternativeName(alt_names), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
        .sign(key, hashes.SHA256())
    )
    KEY_FILE.write_bytes(key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption(),
    ))
    CERT_FILE.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    return True


def build_ssl_context(ips: list[str], cert_file: str | None = None, key_file: str | None = None) -> ssl.SSLContext:
    """Return an SSL context, generating a self-signed certificate if needed.

    ``cert_file``/``key_file`` (e.g. from WEB_SSL_CERT / WEB_SSL_KEY) take
    precedence over the auto-generated certificate in data/certs/.
    """
    if cert_file and key_file:
        cert_path, key_path = Path(cert_file), Path(key_file)
    else:
        cert_path, key_path = CERT_FILE, KEY_FILE
        if not (cert_path.exists() and key_path.exists()):
            CERT_DIR.mkdir(parents=True, exist_ok=True)
            dns, san_ips = _san_entries(ips)
            if not (_generate_with_openssl(dns, san_ips) or _generate_with_cryptography(dns, san_ips)):
                raise RuntimeError(
                    "HTTPS needs a certificate but neither the 'openssl' command nor the "
                    "Python 'cryptography' package is available. Install one "
                    "(sudo apt install openssl  or  pip install cryptography), or set "
                    "WEB_SSL_CERT and WEB_SSL_KEY to an existing certificate."
                )
            print(f"  Generated self-signed HTTPS certificate: {cert_path}")

    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(str(cert_path), str(key_path))
    return context


# ---------------------------------------------------------------------------
# One port, HTTPS only: a plain http:// request is answered with a redirect.
# ---------------------------------------------------------------------------
import re
import socket

from werkzeug.serving import ThreadedWSGIServer

_HOST_RE = re.compile(r"^[A-Za-z0-9.\-:\[\]]{1,255}$")
_TLS_HANDSHAKE = 0x16


class HttpsWithRedirectServer(ThreadedWSGIServer):
    """Threaded Werkzeug server that speaks TLS and redirects plain HTTP.

    The listening socket is not wrapped up front. Each connection's first
    byte is peeked: a TLS ClientHello (0x16) is wrapped and served normally,
    anything else is treated as plain HTTP and answered with a 301 to the
    same URL over https:// on the same port. So typing
    http://172.16.0.11:18473 lands on https://172.16.0.11:18473.
    """

    def __init__(self, host: str, port: int, app, ssl_context: ssl.SSLContext) -> None:
        super().__init__(host, port, app)
        # Set after init so Werkzeug doesn't wrap the listening socket, while
        # its request handler still reports wsgi.url_scheme = "https".
        self.ssl_context = ssl_context

    def finish_request(self, request, client_address) -> None:  # runs in the per-connection thread
        try:
            request.settimeout(15)
            first = request.recv(1, socket.MSG_PEEK)
        except OSError:
            return
        if not first:
            return
        if first[0] != _TLS_HANDSHAKE:
            self._redirect_plain_http(request)
            return
        try:
            tls_sock = self.ssl_context.wrap_socket(request, server_side=True)
        except (ssl.SSLError, OSError):
            return  # e.g. the browser rejected the self-signed cert on this attempt
        try:
            tls_sock.settimeout(None)
            super().finish_request(tls_sock, client_address)
        finally:
            try:
                tls_sock.close()
            except OSError:
                pass

    def _redirect_plain_http(self, sock) -> None:
        data = b""
        try:
            while b"\r\n\r\n" not in data and len(data) < 16384:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                data += chunk
        except OSError:
            pass
        lines = data.split(b"\r\n")
        try:
            path = lines[0].split(b" ")[1].decode("latin-1")
        except IndexError:
            path = "/"
        if not path.startswith("/") or any(c in path for c in "\r\n "):
            path = "/"
        host = ""
        for line in lines[1:]:
            if line.lower().startswith(b"host:"):
                host = line[5:].strip().decode("latin-1")
                break
        if not _HOST_RE.match(host):
            host = f"{self.server_address[0]}:{self.server_address[1]}"
        location = f"https://{host}{path}"
        body = f'<a href="{location}">{location}</a>\n'.encode()
        response = (
            "HTTP/1.1 301 Moved Permanently\r\n"
            f"Location: {location}\r\n"
            "Content-Type: text/html; charset=utf-8\r\n"
            f"Content-Length: {len(body)}\r\n"
            "Connection: close\r\n\r\n"
        ).encode("latin-1") + body
        try:
            sock.sendall(response)
        except OSError:
            pass
