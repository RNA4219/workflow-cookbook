from __future__ import annotations

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "security_headers: セキュリティヘッダ関連のテスト"
    )
