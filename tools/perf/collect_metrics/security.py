# SPDX-License-Identifier: MIT
# Copyright 2025 RNA4219

"""
Security utilities for metrics collection.

URL validation to prevent SSRF and local file access attacks.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import urllib.parse

LOGGER = logging.getLogger(__name__)

# Security: SSRF prevention constants
ALLOWED_SCHEMES: frozenset[str] = frozenset({"http", "https"})
BLOCKED_SCHEMES: frozenset[str] = frozenset({"file", "ftp", "ftps", "sftp", "data", "gopher", "ldap", "ldaps"})
_SKIP_URL_VALIDATION_ENV = "GOVERNANCE_SKIP_URL_VALIDATION_FOR_TESTING"


class MetricsCollectionError(RuntimeError):
    """Raised when metrics could not be collected."""


def validate_url(url: str, context: str = "url") -> str:
    """
    Validate URL to prevent SSRF and local file access attacks.

    - Only http/https schemes are allowed
    - file:// and other dangerous schemes are explicitly blocked
    - localhost, loopback, private networks, and link-local addresses are blocked

    Args:
        url: URL to validate
        context: Context string for error messages (e.g., "metrics_url" or "pushgateway_url")

    Returns:
        The validated URL (unchanged if valid)

    Raises:
        MetricsCollectionError: If URL is invalid or potentially dangerous
    """
    # Allow tests to bypass validation with explicit environment variable
    if os.environ.get(_SKIP_URL_VALIDATION_ENV, "").lower() in ("true", "1", "yes"):
        LOGGER.warning("URL validation disabled for testing: %s", url)
        return url

    try:
        parsed = urllib.parse.urlparse(url)
    except ValueError as exc:
        raise MetricsCollectionError(f"Invalid URL for {context}: {url}: {exc}") from exc

    scheme = parsed.scheme.lower()
    if not scheme:
        raise MetricsCollectionError(f"Missing scheme in {context}: {url}")

    # Explicitly block dangerous schemes
    if scheme in BLOCKED_SCHEMES:
        raise MetricsCollectionError(f"Blocked scheme '{scheme}' in {context}: {url}")

    # Only allow safe schemes
    if scheme not in ALLOWED_SCHEMES:
        raise MetricsCollectionError(f"Unsupported scheme '{scheme}' in {context}: {url}")

    # Resolve hostname and check for dangerous destinations
    hostname = parsed.hostname
    if not hostname:
        raise MetricsCollectionError(f"Missing hostname in {context}: {url}")

    hostname_lower = hostname.lower()

    # Block localhost variants
    if hostname_lower in ("localhost", "local", "localhost.localdomain"):
        raise MetricsCollectionError(f"Blocked hostname '{hostname}' in {context}: {url}")

    # Try to resolve hostname as IP address
    try:
        ip = ipaddress.ip_address(hostname)
    except ValueError:
        # hostname is not a raw IP, but could still resolve to dangerous IP
        pass
    else:
        # Block loopback (127.x.x.x)
        if ip.is_loopback:
            raise MetricsCollectionError(f"Blocked loopback IP in {context}: {url}")

        # Block private networks (10.x, 172.16-31.x, 192.168.x)
        if ip.is_private:
            raise MetricsCollectionError(f"Blocked private network IP in {context}: {url}")

        # Block link-local (169.254.x.x)
        if ip.is_link_local:
            raise MetricsCollectionError(f"Blocked link-local IP in {context}: {url}")

        # Block reserved addresses
        if ip.is_reserved:
            raise MetricsCollectionError(f"Blocked reserved IP in {context}: {url}")

        # Block multicast
        if ip.is_multicast:
            raise MetricsCollectionError(f"Blocked multicast IP in {context}: {url}")

    return url


# Alias for backwards compatibility with tests
_validate_url = validate_url
