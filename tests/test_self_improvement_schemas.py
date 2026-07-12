#!/usr/bin/env python
# SPDX-License-Identifier: MIT
# Copyright 2025 RNA4219

"""Schema validation tests for self-improvement loop DTOs.

Tests that sample config files validate against their referenced schemas.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

try:
    from jsonschema import Draft202012Validator, validate
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False


_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCHEMAS_DIR = _REPO_ROOT / "schemas"
_EXAMPLES_DIR = _REPO_ROOT / "examples"


def _load_json(path: Path) -> dict[str, Any]:
    """Load JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_schema_ref(sample: dict[str, Any]) -> Path | None:
    """Resolve $schema reference to local path."""
    schema_ref = sample.get("$schema", "")
    if schema_ref.startswith("./schemas/"):
        return _SCHEMAS_DIR / schema_ref.replace("./schemas/", "")
    if schema_ref.startswith("https://workflow-cookbook.dev/schemas/"):
        schema_name = schema_ref.replace("https://workflow-cookbook.dev/schemas/", "")
        return _SCHEMAS_DIR / schema_name
    return None


@pytest.mark.skipif(not HAS_JSONSCHEMA, reason="jsonschema not installed")
class TestReflectionSummarySchema:
    """Test ReflectionSummary schema validation."""

    def test_schema_exists(self) -> None:
        """Schema file exists."""
        schema_path = _SCHEMAS_DIR / "reflection-summary.schema.json"
        assert schema_path.exists()

    def test_schema_is_valid(self) -> None:
        """Schema itself is valid JSON Schema."""
        schema = _load_json(_SCHEMAS_DIR / "reflection-summary.schema.json")
        Draft202012Validator.check_schema(schema)

    def test_sample_validates(self) -> None:
        """Sample config validates against schema."""
        sample_path = _EXAMPLES_DIR / "reflection-summary.sample.json"
        if not sample_path.exists():
            pytest.skip("Sample file not found")
        sample = _load_json(sample_path)
        schema_path = _resolve_schema_ref(sample)
        if schema_path is None or not schema_path.exists():
            pytest.skip("Schema reference not resolvable")
        schema = _load_json(schema_path)
        validate(instance=sample, schema=schema)

    def test_required_fields(self) -> None:
        """Schema requires expected fields."""
        schema = _load_json(_SCHEMAS_DIR / "reflection-summary.schema.json")
        required = schema.get("required", [])
        expected_required = [
            "session_id",
            "objective",
            "changes",
            "lessons",
            "open_questions",
            "next_actions",
            "sources",
        ]
        for field in expected_required:
            assert field in required


@pytest.mark.skipif(not HAS_JSONSCHEMA, reason="jsonschema not installed")
class TestSkillDraftRecordSchema:
    """Test SkillDraftRecord schema validation."""

    def test_schema_exists(self) -> None:
        schema_path = _SCHEMAS_DIR / "skill-draft-record.schema.json"
        assert schema_path.exists()

    def test_schema_is_valid(self) -> None:
        schema = _load_json(_SCHEMAS_DIR / "skill-draft-record.schema.json")
        Draft202012Validator.check_schema(schema)

    def test_sample_validates(self) -> None:
        sample_path = _EXAMPLES_DIR / "skill-draft-record.sample.json"
        if not sample_path.exists():
            pytest.skip("Sample file not found")
        sample = _load_json(sample_path)
        schema_path = _resolve_schema_ref(sample)
        if schema_path is None or not schema_path.exists():
            pytest.skip("Schema reference not resolvable")
        schema = _load_json(schema_path)
        validate(instance=sample, schema=schema)

    def test_review_state_enum(self) -> None:
        schema = _load_json(_SCHEMAS_DIR / "skill-draft-record.schema.json")
        props = schema.get("properties", {})
        review_state = props.get("review_state", {})
        enum_values = review_state.get("enum", [])
        assert "draft" in enum_values
        assert "review" in enum_values
        assert "approved" in enum_values
        assert "rejected" in enum_values


@pytest.mark.skipif(not HAS_JSONSCHEMA, reason="jsonschema not installed")
class TestRecallResponseSchema:
    """Test RecallResponse schema validation."""

    def test_schema_exists(self) -> None:
        schema_path = _SCHEMAS_DIR / "recall-response.schema.json"
        assert schema_path.exists()

    def test_schema_is_valid(self) -> None:
        schema = _load_json(_SCHEMAS_DIR / "recall-response.schema.json")
        Draft202012Validator.check_schema(schema)

    def test_required_fields(self) -> None:
        schema = _load_json(_SCHEMAS_DIR / "recall-response.schema.json")
        required = schema.get("required", [])
        expected_required = ["query", "summary", "hits", "stale"]
        for field in expected_required:
            assert field in required


@pytest.mark.skipif(not HAS_JSONSCHEMA, reason="jsonschema not installed")
class TestUserModelSnapshotSchema:
    """Test UserModelSnapshot schema validation."""

    def test_schema_exists(self) -> None:
        schema_path = _SCHEMAS_DIR / "user-model-snapshot.schema.json"
        assert schema_path.exists()

    def test_schema_is_valid(self) -> None:
        schema = _load_json(_SCHEMAS_DIR / "user-model-snapshot.schema.json")
        Draft202012Validator.check_schema(schema)

    def test_required_fields(self) -> None:
        schema = _load_json(_SCHEMAS_DIR / "user-model-snapshot.schema.json")
        required = schema.get("required", [])
        expected_required = [
            "user_id",
            "preferences",
            "approval_style",
            "output_conventions",
            "reviewed_at",
        ]
        for field in expected_required:
            assert field in required


@pytest.mark.skipif(not HAS_JSONSCHEMA, reason="jsonschema not installed")
class TestWorkspaceModelSnapshotSchema:
    """Test WorkspaceModelSnapshot schema validation."""

    def test_schema_exists(self) -> None:
        schema_path = _SCHEMAS_DIR / "workspace-model-snapshot.schema.json"
        assert schema_path.exists()

    def test_schema_is_valid(self) -> None:
        schema = _load_json(_SCHEMAS_DIR / "workspace-model-snapshot.schema.json")
        Draft202012Validator.check_schema(schema)

    def test_required_fields(self) -> None:
        schema = _load_json(_SCHEMAS_DIR / "workspace-model-snapshot.schema.json")
        required = schema.get("required", [])
        expected_required = ["workspace_id", "constraints", "preferred_docs", "reviewed_at"]
        for field in expected_required:
            assert field in required


@pytest.mark.skipif(not HAS_JSONSCHEMA, reason="jsonschema not installed")
class TestPeriodicNudgeSchema:
    """Test PeriodicNudge schema validation."""

    def test_schema_exists(self) -> None:
        schema_path = _SCHEMAS_DIR / "periodic-nudge.schema.json"
        assert schema_path.exists()

    def test_schema_is_valid(self) -> None:
        schema = _load_json(_SCHEMAS_DIR / "periodic-nudge.schema.json")
        Draft202012Validator.check_schema(schema)

    def test_required_fields(self) -> None:
        schema = _load_json(_SCHEMAS_DIR / "periodic-nudge.schema.json")
        required = schema.get("required", [])
        expected_required = [
            "nudge_id",
            "reason",
            "target_kind",
            "target_ref",
            "suggested_action",
            "created_at",
        ]
        for field in expected_required:
            assert field in required

    def test_blocking_default_false(self) -> None:
        schema = _load_json(_SCHEMAS_DIR / "periodic-nudge.schema.json")
        props = schema.get("properties", {})
        blocking = props.get("blocking", {})
        assert blocking.get("default") is False

    def test_gate_policy_and_evidence_type_targets_are_supported(self) -> None:
        schema = _load_json(_SCHEMAS_DIR / "periodic-nudge.schema.json")
        targets = schema["properties"]["target_kind"]["enum"]
        assert {"gate", "policy", "evidence_type"} <= set(targets)
