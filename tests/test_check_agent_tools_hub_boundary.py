"""Tests for check_agent_tools_hub_boundary.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "ci" / "check_agent_tools_hub_boundary.py"
spec = importlib.util.spec_from_file_location("check_agent_tools_hub_boundary", MODULE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError("Failed to load check_agent_tools_hub_boundary module")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

check_agent_tools_hub_boundary = module.check_agent_tools_hub_boundary


@pytest.fixture
def repo_root(tmp_path: Path) -> Path:
    return tmp_path


def _write_hub(target: Path, content: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def test_pass_workflow_cookbook_internal_content(repo_root: Path) -> None:
    """HUB with only workflow-cookbook internal content should pass."""
    hub_content = """
---
intent_id: INT-001
---

# HUB.codex.md

## Purpose

Workflow cookbook internal task routing.

## Birdseye / Codemap

Birdseye update procedures.

## Task Seed

Task seed creation procedures.

## Acceptance

Acceptance record procedures.

## CI

CI workflow procedures.

## Evidence

Evidence tracking procedures.
"""
    _write_hub(repo_root / "HUB.codex.md", hub_content)

    result = check_agent_tools_hub_boundary(repo_root)
    assert result["errors"] == []
    assert result["warnings"] == []


def test_pass_with_agent_tools_hub_reference(repo_root: Path) -> None:
    """HUB referencing agent-tools-hub for routing should pass."""
    hub_content = """
---
intent_id: INT-001
---

# HUB.codex.md

## Purpose

Workflow cookbook internal routing.
For Agent_tools overall repo selection, see [agent-tools-hub](../Agent_tools/README.md).

## Birdseye

Birdseye procedures.

## Cross-repo plugin

For cross-repo plugin configuration, refer to agent-tools-hub routing.
"""
    _write_hub(repo_root / "HUB.codex.md", hub_content)

    result = check_agent_tools_hub_boundary(repo_root)
    assert result["errors"] == []


def test_warning_routing_duplication(repo_root: Path) -> None:
    """HUB with routing table duplication should warn."""
    hub_content = """
---
intent_id: INT-001
---

# HUB.codex.md

## Repo Selection

Use this repo for docs, that repo for tasks, and another repo for testing.

This is the routing table for all repos.

## Birdseye

Birdseye procedures.
"""
    _write_hub(repo_root / "HUB.codex.md", hub_content)

    result = check_agent_tools_hub_boundary(repo_root)
    assert len(result["warnings"]) >= 1
    assert any("routing" in w.lower() for w in result["warnings"])


def test_warning_missing_agent_tools_hub_reference(repo_root: Path) -> None:
    """Cross-repo selection without agent-tools-hub reference should warn."""
    hub_content = """
---
intent_id: INT-001
---

# HUB.codex.md

## Cross-repo Guidance

For Agent_tools 全体の使い分け, choose the right repo based on task type.

## Birdseye

Birdseye procedures.
"""
    _write_hub(repo_root / "HUB.codex.md", hub_content)

    result = check_agent_tools_hub_boundary(repo_root)
    assert len(result["warnings"]) >= 1
    assert any("agent-tools-hub" in w for w in result["warnings"])


def test_warning_missing_procedure_sections(repo_root: Path) -> None:
    """HUB missing key procedure sections should warn."""
    hub_content = """
---
intent_id: INT-001
---

# HUB.codex.md

## Purpose

General information.

## Other

Other content.
"""
    _write_hub(repo_root / "HUB.codex.md", hub_content)

    result = check_agent_tools_hub_boundary(repo_root)
    assert len(result["warnings"]) >= 1
    assert any("Birdseye" in w for w in result["warnings"])


def test_pass_allowed_plugin_context(repo_root: Path) -> None:
    """Plugin/config connection context should pass."""
    hub_content = """
---
intent_id: INT-001
---

# HUB.codex.md

## Cross-repo Plugin

Plugin config for cross-repo integration.
Evidence 連携 via agent-protocols.
Acceptance 連携 to agent-taskstate.

## Task Seed

Task seed procedures.

## Birdseye

Birdseye procedures.

## Acceptance

Acceptance procedures.

## CI

CI procedures.

## Evidence

Evidence procedures.
"""
    _write_hub(repo_root / "HUB.codex.md", hub_content)

    result = check_agent_tools_hub_boundary(repo_root)
    # Should not warn about routing duplication for allowed plugin context
    routing_warnings = [w for w in result["warnings"] if "routing" in w.lower()]
    assert routing_warnings == []


def test_missing_hub_file(repo_root: Path) -> None:
    """Missing HUB.codex.md should error."""
    result = check_agent_tools_hub_boundary(repo_root)
    assert len(result["errors"]) >= 1
    assert any("not found" in e for e in result["errors"])


def test_edge_case_empty_hub_file(repo_root: Path) -> None:
    """Empty HUB.codex.md should warn about missing procedures."""
    hub_path = repo_root / "HUB.codex.md"
    hub_path.parent.mkdir(parents=True, exist_ok=True)
    hub_path.write_text("", encoding="utf-8")

    result = check_agent_tools_hub_boundary(repo_root)
    # Should warn about missing procedures
    assert len(result["warnings"]) >= 5  # All 5 procedures missing


def test_edge_case_hub_without_sections(repo_root: Path) -> None:
    """HUB without ## sections should be handled gracefully."""
    hub_content = """
---
intent_id: INT-001
---

Just a single paragraph without any section headers.
"""
    _write_hub(repo_root / "HUB.codex.md", hub_content)

    result = check_agent_tools_hub_boundary(repo_root)
    assert isinstance(result, dict)


def test_edge_case_japanese_keywords(repo_root: Path) -> None:
    """Japanese keywords (検収, 証跡) should be detected."""
    hub_content = """
---
intent_id: INT-001
---

# HUB.codex.md

## Birdseye

Birdseye procedures.

## Task Seed

Task seed procedures.

## 検収

検収記録の手順。

## CI

CI procedures.

## 証跡

証跡追跡の手順。
"""
    _write_hub(repo_root / "HUB.codex.md", hub_content)

    result = check_agent_tools_hub_boundary(repo_root)
    # Japanese keywords should satisfy Acceptance/Evidence requirements
    acceptance_warnings = [w for w in result["warnings"] if "Acceptance" in w]
    evidence_warnings = [w for w in result["warnings"] if "Evidence" in w]
    assert acceptance_warnings == []
    assert evidence_warnings == []