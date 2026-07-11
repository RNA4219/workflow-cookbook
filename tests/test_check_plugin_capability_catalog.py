# SPDX-License-Identifier: MIT
# Copyright 2025 RNA4219

"""Tests for check_plugin_capability_catalog.py."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.ci.check_plugin_capability_catalog import (
    load_catalog_capabilities,
    load_interfaces_capabilities,
    main,
    validate_plugin_capability_catalog,
)


class TestLoadCapabilities:
    """Test capability loading functions."""

    def test_load_catalog_capabilities(self, tmp_path: Path) -> None:
        catalog = tmp_path / "catalog.json"
        catalog.write_text(
            '{"capabilities": [{"id": "test.cap", "method_name": "test_method"}]}',
            encoding="utf-8",
        )
        caps = load_catalog_capabilities(catalog)
        assert caps == {"test.cap": "test_method"}

    def test_load_catalog_empty_on_missing(self, tmp_path: Path) -> None:
        caps = load_catalog_capabilities(tmp_path / "missing.json")
        assert caps == {}

    def test_load_interfaces_capabilities(self) -> None:
        caps = load_interfaces_capabilities(ROOT / "tools" / "workflow_plugins" / "interfaces.py")
        assert "task_state.sync" in caps
        assert caps["task_state.sync"] == "sync_task_acceptance"


class TestValidateCapabilityCatalog:
    """Test catalog validation."""

    def test_validate_matches_interfaces(self) -> None:
        catalog = ROOT / "examples" / "plugin-capability-catalog.sample.json"
        interfaces = ROOT / "tools" / "workflow_plugins" / "interfaces.py"
        result = validate_plugin_capability_catalog(
            catalog_path=catalog,
            interfaces_path=interfaces,
        )
        assert result.is_success

    def test_validate_detects_missing_capability(self, tmp_path: Path) -> None:
        catalog = tmp_path / "catalog.json"
        catalog.write_text(
            '{"capabilities": [{"id": "only.this", "method_name": "only_method"}]}',
            encoding="utf-8",
        )
        interfaces = ROOT / "tools" / "workflow_plugins" / "interfaces.py"
        result = validate_plugin_capability_catalog(
            catalog_path=catalog,
            interfaces_path=interfaces,
        )
        # Should have errors for missing capabilities from interfaces.py
        assert len(result.errors) > 0
        assert any("task_state.sync" in e for e in result.errors)

    def test_validate_detects_method_mismatch(self, tmp_path: Path) -> None:
        catalog = tmp_path / "catalog.json"
        catalog.write_text(
            '{"capabilities": [{"id": "task_state.sync", "method_name": "wrong_method"}]}',
            encoding="utf-8",
        )
        interfaces = ROOT / "tools" / "workflow_plugins" / "interfaces.py"
        result = validate_plugin_capability_catalog(
            catalog_path=catalog,
            interfaces_path=interfaces,
        )
        assert any("method mismatch" in e for e in result.errors)


class TestMain:
    """Test CLI entry point."""

    def test_main_returns_success_for_valid_repo(self) -> None:
        exit_code = main(())
        assert exit_code == 0