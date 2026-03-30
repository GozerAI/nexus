"""Tests for SSRF URL validation hardening."""

from unittest.mock import patch

from nexus.api.routes.data import _validate_url


class TestSSRFValidation:
    """Verify SSRF protections in _validate_url."""

    def test_rejects_non_http_scheme(self):
        assert not _validate_url("ftp://example.com/file")
        assert not _validate_url("file:///etc/passwd")
        assert not _validate_url("gopher://evil.com")

    def test_rejects_empty_hostname(self):
        assert not _validate_url("http:///path")
        assert not _validate_url("")

    def test_rejects_localhost(self):
        assert not _validate_url("http://localhost/admin")
        assert not _validate_url("http://localhost:8080")

    def test_rejects_metadata_endpoint(self):
        assert not _validate_url("http://metadata.google.internal/computeMetadata")
        assert not _validate_url("http://169.254.169.254/latest/meta-data")

    @patch("nexus.api.routes.data.socket.gethostbyname_ex")
    def test_rejects_private_ip(self, mock_dns):
        mock_dns.return_value = ("internal", [], ["10.0.0.1"])
        assert not _validate_url("http://internal.corp/api")

    @patch("nexus.api.routes.data.socket.gethostbyname_ex")
    def test_rejects_loopback_ip(self, mock_dns):
        mock_dns.return_value = ("loop", [], ["127.0.0.1"])
        assert not _validate_url("http://sneaky.com")

    @patch("nexus.api.routes.data.socket.gethostbyname_ex")
    def test_rejects_link_local(self, mock_dns):
        mock_dns.return_value = ("ll", [], ["169.254.1.1"])
        assert not _validate_url("http://ll.example.com")

    @patch("nexus.api.routes.data.socket.gethostbyname_ex")
    def test_rejects_if_any_ip_private(self, mock_dns):
        """DNS rebinding: one public IP + one private → block."""
        mock_dns.return_value = ("mixed", [], ["93.184.216.34", "192.168.1.1"])
        assert not _validate_url("http://mixed.example.com")

    @patch("nexus.api.routes.data.socket.gethostbyname_ex")
    def test_allows_public_ip(self, mock_dns):
        mock_dns.return_value = ("example.com", [], ["93.184.216.34"])
        assert _validate_url("http://example.com/page")

    @patch("nexus.api.routes.data.socket.gethostbyname_ex")
    def test_allows_https(self, mock_dns):
        mock_dns.return_value = ("secure.example.com", [], ["93.184.216.34"])
        assert _validate_url("https://secure.example.com/api")

    @patch("nexus.api.routes.data.socket.gethostbyname_ex")
    def test_dns_failure_returns_false(self, mock_dns):
        import socket
        mock_dns.side_effect = socket.gaierror("Name resolution failed")
        assert not _validate_url("http://nonexistent.invalid")

    @patch("nexus.api.routes.data.socket.gethostbyname_ex")
    def test_dns_timeout_returns_false(self, mock_dns):
        import socket
        mock_dns.side_effect = socket.timeout("Timed out")
        assert not _validate_url("http://slow.example.com")

    @patch("nexus.api.routes.data.socket.gethostbyname_ex")
    def test_rejects_multicast(self, mock_dns):
        mock_dns.return_value = ("mc", [], ["224.0.0.1"])
        assert not _validate_url("http://mc.example.com")

    @patch("nexus.api.routes.data.socket.gethostbyname_ex")
    def test_rejects_reserved(self, mock_dns):
        mock_dns.return_value = ("res", [], ["240.0.0.1"])
        assert not _validate_url("http://res.example.com")
