# SPDX-License-Identifier: MIT
# Copyright 2025 RNA4219

"""Smoke tests for CLI entry points."""

from __future__ import annotations

import shutil
import subprocess
import sys
import venv
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def console_scripts(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """インストール試験は作業用Python環境から隔離する。"""
    environment = tmp_path_factory.mktemp("cli-install") / "venv"
    uv = shutil.which("uv")
    if uv:
        subprocess.run(
            [uv, "venv", "--python", sys.executable, str(environment)],
            check=True,
            capture_output=True,
            text=True,
        )
    else:
        venv.EnvBuilder(with_pip=True).create(environment)
    scripts = environment / ("Scripts" if sys.platform == "win32" else "bin")
    python = scripts / ("python.exe" if sys.platform == "win32" else "python")
    command = (
        [uv, "pip", "install", "--python", str(python), "-e", str(_REPO_ROOT)]
        if uv
        else [str(python), "-m", "pip", "install", "-e", str(_REPO_ROOT)]
    )
    result = subprocess.run(command, cwd=environment.parent, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    return scripts


def _console_script(scripts: Path, name: str) -> str:
    suffix = ".exe" if sys.platform == "win32" else ""
    executable = scripts / (name + suffix)
    assert executable.is_file(), f"Missing installed entrypoint: {executable}"
    return str(executable)


class TestGovernanceGateEntrypoint:
    """Test governance gate CLI entry points."""

    def test_python_m_governance_gate_help(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "tools.ci.governance_gate", "--help"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "governance gate" in result.stdout.lower()


class TestCollectMetricsEntrypoint:
    """Test collect_metrics CLI entry points."""

    def test_python_m_collect_metrics_help(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "tools.perf.collect_metrics", "--help"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "metrics" in result.stdout.lower()


class TestCodemapUpdateEntrypoint:
    """Test codemap update CLI entry points."""

    def test_python_m_codemap_update_help(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "tools.codemap.update", "--help"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "codemap" in result.stdout.lower() or "birdseye" in result.stdout.lower()


class TestContextPackEntrypoint:
    """Test context pack CLI entry points."""

    def test_python_m_context_pack_help(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "tools.context.pack", "--help"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0


class TestFiveToolManifestEntrypoint:
    """Test five-tool manifest CLI entry points."""

    def test_python_m_five_tool_manifest_help(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "tools.ci.five_tool_manifest", "--help"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "five-tool" in result.stdout.lower()


class TestConsoleScripts:
    """Test installed console script entry points."""

    def test_wfc_governance_gate_help(self, console_scripts: Path) -> None:
        result = subprocess.run(
            [_console_script(console_scripts, "wfc-governance-gate"), "--help"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "governance gate" in result.stdout.lower()

    def test_wfc_collect_metrics_help(self, console_scripts: Path) -> None:
        result = subprocess.run(
            [_console_script(console_scripts, "wfc-collect-metrics"), "--help"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "metrics" in result.stdout.lower()

    def test_wfc_codemap_update_help(self, console_scripts: Path) -> None:
        result = subprocess.run(
            [_console_script(console_scripts, "wfc-codemap-update"), "--help"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "codemap" in result.stdout.lower() or "birdseye" in result.stdout.lower()

    def test_wfc_context_pack_help(self, console_scripts: Path) -> None:
        result = subprocess.run(
            [_console_script(console_scripts, "wfc-context-pack"), "--help"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

    def test_wfc_five_tool_manifest_help(self, console_scripts: Path) -> None:
        result = subprocess.run(
            [_console_script(console_scripts, "wfc-five-tool-manifest"), "--help"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "five-tool" in result.stdout.lower()
