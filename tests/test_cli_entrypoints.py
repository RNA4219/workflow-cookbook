# SPDX-License-Identifier: MIT
# Copyright 2025 RNA4219

"""Smoke tests for CLI entry points."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


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


class TestConsoleScripts:
    """Test installed console script entry points."""

    def test_wfc_governance_gate_help(self) -> None:
        # Build and install the package to get console scripts
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", "."],
            cwd=_REPO_ROOT,
            capture_output=True,
            check=True,
        )
        result = subprocess.run(
            ["wfc-governance-gate", "--help"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "governance gate" in result.stdout.lower()

    def test_wfc_collect_metrics_help(self) -> None:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", "."],
            cwd=_REPO_ROOT,
            capture_output=True,
            check=True,
        )
        result = subprocess.run(
            ["wfc-collect-metrics", "--help"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "metrics" in result.stdout.lower()