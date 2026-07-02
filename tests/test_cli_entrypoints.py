# SPDX-License-Identifier: MIT
# Copyright 2025 RNA4219

"""Smoke tests for CLI entry points."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _install_editable() -> None:
    if getattr(_install_editable, "_done", False):
        return

    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", "."],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 and "No module named pip" in result.stderr:
        uv = shutil.which("uv")
        if uv is None:
            result.check_returncode()
        subprocess.run(
            [uv, "pip", "install", "--python", sys.executable, "-e", "."],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    else:
        result.check_returncode()
    setattr(_install_editable, "_done", True)


def _console_script(name: str) -> str:
    scripts_dir = Path(sys.executable).parent
    for candidate in (scripts_dir / name, scripts_dir / f"{name}.exe", scripts_dir / f"{name}.cmd"):
        if candidate.exists():
            return str(candidate)
    return name


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

    def test_wfc_governance_gate_help(self) -> None:
        _install_editable()
        result = subprocess.run(
            [_console_script("wfc-governance-gate"), "--help"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "governance gate" in result.stdout.lower()

    def test_wfc_collect_metrics_help(self) -> None:
        _install_editable()
        result = subprocess.run(
            [_console_script("wfc-collect-metrics"), "--help"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "metrics" in result.stdout.lower()

    def test_wfc_codemap_update_help(self) -> None:
        _install_editable()
        result = subprocess.run(
            [_console_script("wfc-codemap-update"), "--help"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "codemap" in result.stdout.lower() or "birdseye" in result.stdout.lower()

    def test_wfc_context_pack_help(self) -> None:
        _install_editable()
        result = subprocess.run(
            [_console_script("wfc-context-pack"), "--help"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

    def test_wfc_five_tool_manifest_help(self) -> None:
        _install_editable()
        result = subprocess.run(
            [_console_script("wfc-five-tool-manifest"), "--help"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "five-tool" in result.stdout.lower()
