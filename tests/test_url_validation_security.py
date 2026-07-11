# SPDX-License-Identifier: MIT
# Copyright 2025 RNA4219
"""Security tests for URL validation in collect_metrics."""

from __future__ import annotations

import pytest

from tools.perf.collect_metrics import (
    MetricsCollectionError,
    _validate_url,
)


class TestUrlValidationScheme:
    """Tests for URL scheme validation."""

    def test_allowed_http_scheme(self) -> None:
        """http:// URLs should be allowed."""
        url = "http://example.com/metrics"
        assert _validate_url(url) == url

    def test_allowed_https_scheme(self) -> None:
        """https:// URLs should be allowed."""
        url = "https://example.com/metrics"
        assert _validate_url(url) == url

    def test_blocked_file_scheme(self) -> None:
        """file:// URLs should be explicitly blocked."""
        with pytest.raises(MetricsCollectionError, match="Blocked scheme 'file'"):
            _validate_url("file:///etc/passwd", context="metrics_url")

    def test_blocked_ftp_scheme(self) -> None:
        """ftp:// URLs should be blocked."""
        with pytest.raises(MetricsCollectionError, match="Blocked scheme 'ftp'"):
            _validate_url("ftp://example.com/file", context="metrics_url")

    def test_blocked_data_scheme(self) -> None:
        """data:// URLs should be blocked."""
        with pytest.raises(MetricsCollectionError, match="Blocked scheme 'data'"):
            _validate_url("data:text/plain,hello", context="metrics_url")

    def test_blocked_gopher_scheme(self) -> None:
        """gopher:// URLs should be blocked."""
        with pytest.raises(MetricsCollectionError, match="Blocked scheme 'gopher'"):
            _validate_url("gopher://example.com", context="metrics_url")

    def test_blocked_ldap_scheme(self) -> None:
        """ldap:// URLs should be blocked."""
        with pytest.raises(MetricsCollectionError, match="Blocked scheme 'ldap'"):
            _validate_url("ldap://example.com", context="metrics_url")

    def test_missing_scheme(self) -> None:
        """URLs without scheme should be rejected."""
        with pytest.raises(MetricsCollectionError, match="Missing scheme"):
            _validate_url("example.com/metrics", context="metrics_url")

    def test_unsupported_scheme(self) -> None:
        """Unknown schemes should be rejected."""
        with pytest.raises(MetricsCollectionError, match="Unsupported scheme"):
            _validate_url("unknown://example.com", context="metrics_url")


class TestUrlValidationHostname:
    """Tests for hostname/DNS rebinding prevention."""

    def test_blocked_localhost(self) -> None:
        """localhost should be blocked."""
        with pytest.raises(MetricsCollectionError, match="Blocked hostname 'localhost'"):
            _validate_url("http://localhost:9090/metrics", context="metrics_url")

    def test_blocked_localhost_localdomain(self) -> None:
        """localhost.localdomain should be blocked."""
        with pytest.raises(MetricsCollectionError, match="Blocked hostname"):
            _validate_url("http://localhost.localdomain/metrics", context="metrics_url")

    def test_blocked_127_0_0_1(self) -> None:
        """127.0.0.1 (loopback) should be blocked."""
        with pytest.raises(MetricsCollectionError, match="Blocked loopback IP"):
            _validate_url("http://127.0.0.1:9090/metrics", context="metrics_url")

    def test_blocked_127_any(self) -> None:
        """Any 127.x.x.x (loopback range) should be blocked."""
        with pytest.raises(MetricsCollectionError, match="Blocked loopback IP"):
            _validate_url("http://127.255.255.255/metrics", context="metrics_url")

    def test_blocked_private_10_x(self) -> None:
        """10.x.x.x (private network) should be blocked."""
        with pytest.raises(MetricsCollectionError, match="Blocked private network IP"):
            _validate_url("http://10.0.0.1/metrics", context="metrics_url")

    def test_blocked_private_172_16(self) -> None:
        """172.16.x.x - 172.31.x.x (private network) should be blocked."""
        with pytest.raises(MetricsCollectionError, match="Blocked private network IP"):
            _validate_url("http://172.16.0.1/metrics", context="metrics_url")

    def test_blocked_private_192_168(self) -> None:
        """192.168.x.x (private network) should be blocked."""
        with pytest.raises(MetricsCollectionError, match="Blocked private network IP"):
            _validate_url("http://192.168.1.1/metrics", context="metrics_url")

    def test_blocked_link_local(self) -> None:
        """169.254.x.x (link-local) should be blocked."""
        with pytest.raises(MetricsCollectionError, match="Blocked (link-local|private network) IP"):
            _validate_url("http://169.254.1.1/metrics", context="metrics_url")

    def test_blocked_multicast(self) -> None:
        """Multicast addresses should be blocked."""
        with pytest.raises(MetricsCollectionError, match="Blocked multicast IP"):
            _validate_url("http://224.0.0.1/metrics", context="metrics_url")

    def test_allowed_public_ip(self) -> None:
        """Public IP addresses should be allowed."""
        url = "http://8.8.8.8/metrics"
        assert _validate_url(url) == url

    def test_allowed_public_hostname(self) -> None:
        """Public hostnames should be allowed."""
        url = "https://prometheus.example.com/metrics"
        assert _validate_url(url) == url

    def test_allowed_ip_with_port(self) -> None:
        """Public IP with port should be allowed."""
        url = "https://1.1.1.1:9090/metrics"
        assert _validate_url(url) == url

    def test_missing_hostname(self) -> None:
        """URLs without hostname should be rejected."""
        with pytest.raises(MetricsCollectionError, match="Missing hostname"):
            _validate_url("http:///metrics", context="metrics_url")


class TestUrlValidationEdgeCases:
    """Tests for edge cases in URL validation."""

    def test_ipv6_loopback_blocked(self) -> None:
        """IPv6 loopback (::1) should be blocked."""
        with pytest.raises(MetricsCollectionError, match="Blocked loopback IP"):
            _validate_url("http://[::1]:9090/metrics", context="metrics_url")

    def test_ipv6_private_blocked(self) -> None:
        """IPv6 private addresses should be blocked."""
        with pytest.raises(MetricsCollectionError, match="Blocked private network IP"):
            _validate_url("http://[fc00::1]/metrics", context="metrics_url")

    def test_ipv6_link_local_blocked(self) -> None:
        """IPv6 link-local addresses should be blocked."""
        with pytest.raises(MetricsCollectionError, match="Blocked (link-local|private network) IP"):
            _validate_url("http://[fe80::1]/metrics", context="metrics_url")

    def test_ipv6_public_allowed(self) -> None:
        """IPv6 public addresses should be allowed."""
        # Note: 2001:db8::/32 is documentation range, treated as reserved
        # Use a truly public IPv6 address
        public_url = "http://[2606:4700:4700::1111]/metrics"
        assert _validate_url(public_url) == public_url

    def test_invalid_url_format(self) -> None:
        """Invalid URL format should be rejected."""
        with pytest.raises(MetricsCollectionError, match="Invalid URL"):
            _validate_url("http://[invalid-ipv6]/metrics", context="metrics_url")

    def test_url_with_path(self) -> None:
        """URL with path should be validated correctly."""
        url = "https://prometheus.example.com/api/v1/metrics"
        assert _validate_url(url) == url

    def test_url_with_query(self) -> None:
        """URL with query string should be validated correctly."""
        url = "https://prometheus.example.com/metrics?format=prometheus"
        assert _validate_url(url) == url

    def test_context_in_error_message(self) -> None:
        """Error messages should include the context parameter."""
        with pytest.raises(MetricsCollectionError, match="pushgateway_url"):
            _validate_url("file:///etc/passwd", context="pushgateway_url")
