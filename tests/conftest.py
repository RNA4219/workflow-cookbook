from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "security_headers: セキュリティヘッダ関連のテスト"
    )
    config.addinivalue_line("markers", "posix: POSIXの権限bitが必要（Linuxで必須実行）")
    if config.getoption("--fail-on-skip"):
        config.pluginmanager.register(_RejectSkips(), "reject-skips")


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--fail-on-skip", action="store_true", help="収集・実行中のskipをCI失敗にする"
    )


class _RejectSkips:
    def __init__(self) -> None:
        self.skipped: list[str] = []

    def pytest_collectreport(self, report: pytest.CollectReport) -> None:
        if report.skipped:
            self.skipped.append(report.nodeid)

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        if report.skipped:
            self.skipped.append(report.nodeid)

    def pytest_sessionfinish(self, session: pytest.Session, exitstatus: int) -> None:
        if self.skipped:
            print("\nRequired CI tests skipped: " + ", ".join(self.skipped))
            if exitstatus == pytest.ExitCode.OK:
                session.exitstatus = pytest.ExitCode.TESTS_FAILED
