#!/usr/bin/env python
# SPDX-License-Identifier: MIT
# Copyright 2025 RNA4219

"""Tests for check_stale_self_improvement.py."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

# Add tools to path for import
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.ci.check_stale_self_improvement import (
    check_stale_reflections,
    check_stale_skill_drafts,
    load_json_reflection,
)


@pytest.fixture
def temp_reflections_dir(tmp_path: Path) -> Path:
    """Create temporary reflections directory."""
    reflections_dir = tmp_path / ".workflow-cache" / "reflections"
    reflections_dir.mkdir(parents=True)
    return reflections_dir


@pytest.fixture
def temp_skill_drafts_dir(tmp_path: Path) -> Path:
    """Create temporary skill drafts directory."""
    drafts_dir = tmp_path / ".workflow-cache" / "skill-drafts"
    drafts_dir.mkdir(parents=True)
    return drafts_dir


def write_reflection(path: Path, created_at: str) -> None:
    """Write a reflection file."""
    reflection = {
        "$schema": "./schemas/reflection-summary.schema.json",
        "session_id": path.stem,
        "objective": "Test reflection",
        "changes": [],
        "lessons": [],
        "open_questions": [],
        "next_actions": [],
        "sources": [],
        "created_at": created_at,
    }
    path.write_text(json.dumps(reflection, ensure_ascii=False), encoding="utf-8")


def write_skill_draft(path: Path, review_state: str, created_at: str) -> None:
    """Write a skill draft file."""
    draft = {
        "$schema": "./schemas/skill-draft-record.schema.json",
        "draft_id": path.stem,
        "source_session_id": "SESSION-TEST",
        "title": "Test skill",
        "problem": "Test problem",
        "proposed_steps": [{"step": "Test step"}],
        "review_state": review_state,
        "created_at": created_at,
    }
    path.write_text(json.dumps(draft, ensure_ascii=False), encoding="utf-8")


class TestLoadJsonReflection:
    """Test load_json_reflection function."""

    def test_valid_json(self, temp_reflections_dir: Path) -> None:
        reflection_path = temp_reflections_dir / "SESSION-TEST.json"
        write_reflection(reflection_path, "2026-05-02T12:00:00Z")
        result = load_json_reflection(reflection_path)
        assert result is not None
        assert result["session_id"] == "SESSION-TEST"

    def test_invalid_json(self, temp_reflections_dir: Path) -> None:
        reflection_path = temp_reflections_dir / "INVALID.json"
        reflection_path.write_text("not json", encoding="utf-8")
        result = load_json_reflection(reflection_path)
        assert result is None

    def test_missing_file(self, temp_reflections_dir: Path) -> None:
        reflection_path = temp_reflections_dir / "MISSING.json"
        result = load_json_reflection(reflection_path)
        assert result is None


class TestCheckStaleReflections:
    """Test check_stale_reflections function."""

    def test_fresh_reflection(self, temp_reflections_dir: Path) -> None:
        now = datetime.now()
        fresh_time = (now - timedelta(days=5)).isoformat()
        reflection_path = temp_reflections_dir / "SESSION-FRESH.json"
        write_reflection(reflection_path, fresh_time)

        nudges, warnings = check_stale_reflections(
            max_age_days=30,
            reflections_dir=temp_reflections_dir,
        )
        assert len(nudges) == 0

    def test_stale_reflection(self, temp_reflections_dir: Path) -> None:
        now = datetime.now()
        stale_time = (now - timedelta(days=45)).isoformat()
        reflection_path = temp_reflections_dir / "SESSION-STALE.json"
        write_reflection(reflection_path, stale_time)

        nudges, warnings = check_stale_reflections(
            max_age_days=30,
            reflections_dir=temp_reflections_dir,
        )
        assert len(nudges) == 1
        assert nudges[0]["target_kind"] == "reflection"
        assert nudges[0]["stale_days"] >= 45

    def test_empty_dir(self, tmp_path: Path) -> None:
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        nudges, warnings = check_stale_reflections(
            max_age_days=30,
            reflections_dir=empty_dir,
        )
        assert len(nudges) == 0
        assert len(warnings) == 0

    def test_missing_dir(self, tmp_path: Path) -> None:
        missing_dir = tmp_path / "missing"
        nudges, warnings = check_stale_reflections(
            max_age_days=30,
            reflections_dir=missing_dir,
        )
        assert len(warnings) == 1
        assert "not found" in warnings[0]


class TestCheckStaleSkillDrafts:
    """Test check_stale_skill_drafts function."""

    def test_approved_draft(self, temp_skill_drafts_dir: Path) -> None:
        now = datetime.now()
        old_time = (now - timedelta(days=30)).isoformat()
        draft_path = temp_skill_drafts_dir / "SKILL-APPROVED.json"
        write_skill_draft(draft_path, "approved", old_time)

        nudges, warnings = check_stale_skill_drafts(
            max_age_days=14,
            skill_drafts_dir=temp_skill_drafts_dir,
        )
        assert len(nudges) == 0  # approved drafts don't trigger nudges

    def test_stale_draft_in_review(self, temp_skill_drafts_dir: Path) -> None:
        now = datetime.now()
        stale_time = (now - timedelta(days=20)).isoformat()
        draft_path = temp_skill_drafts_dir / "SKILL-IN-REVIEW.json"
        write_skill_draft(draft_path, "review", stale_time)

        nudges, warnings = check_stale_skill_drafts(
            max_age_days=14,
            skill_drafts_dir=temp_skill_drafts_dir,
        )
        assert len(nudges) == 1
        assert nudges[0]["target_kind"] == "skill_draft"

    def test_fresh_draft_in_review(self, temp_skill_drafts_dir: Path) -> None:
        now = datetime.now()
        fresh_time = (now - timedelta(days=5)).isoformat()
        draft_path = temp_skill_drafts_dir / "SKILL-FRESH.json"
        write_skill_draft(draft_path, "review", fresh_time)

        nudges, warnings = check_stale_skill_drafts(
            max_age_days=14,
            skill_drafts_dir=temp_skill_drafts_dir,
        )
        assert len(nudges) == 0


class TestNudgeSchemaCompliance:
    """Test generated nudges comply with schema."""

    def test_nudge_required_fields(self, temp_reflections_dir: Path) -> None:
        now = datetime.now()
        stale_time = (now - timedelta(days=45)).isoformat()
        reflection_path = temp_reflections_dir / "SESSION-STALE.json"
        write_reflection(reflection_path, stale_time)

        nudges, _ = check_stale_reflections(
            max_age_days=30,
            reflections_dir=temp_reflections_dir,
        )
        assert len(nudges) > 0

        nudge = nudges[0]
        required_fields = [
            "nudge_id",
            "reason",
            "target_kind",
            "target_ref",
            "suggested_action",
            "created_at",
        ]
        for field in required_fields:
            assert field in nudge

    def test_blocking_is_false(self, temp_reflections_dir: Path) -> None:
        now = datetime.now()
        stale_time = (now - timedelta(days=45)).isoformat()
        reflection_path = temp_reflections_dir / "SESSION-STALE.json"
        write_reflection(reflection_path, stale_time)

        nudges, _ = check_stale_reflections(
            max_age_days=30,
            reflections_dir=temp_reflections_dir,
        )
        assert nudges[0]["blocking"] is False