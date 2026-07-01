from __future__ import annotations

from tools.ci.self_improvement_ops import (
    build_recall_response,
    export_curated_memory,
    generate_skill_draft,
)


def _reflection() -> dict:
    return {
        "session_id": "S-1",
        "task_id": "TASK-1",
        "objective": "Improve release workflow",
        "review_state": "approved",
        "lessons": [{"observation": "Release checks need a single report.", "actionable": True}],
        "next_actions": [{"action": "Generate readiness report"}],
        "sources": [{"type": "acceptance", "ref": "AC-1"}, {"type": "evidence", "ref": "EV-1"}],
    }


def test_export_curated_memory_includes_reviewed_reflections_only() -> None:
    snapshot = export_curated_memory([_reflection(), {"session_id": "S-2"}])

    assert snapshot["summary"]["included"] == 1
    assert snapshot["summary"]["skipped_unreviewed"] == 1


def test_generate_skill_draft_from_reflection() -> None:
    draft = generate_skill_draft(_reflection(), author="tester")

    assert draft["review_state"] == "draft"
    assert draft["linked_acceptance_ids"] == ["AC-1"]
    assert draft["proposed_steps"][0]["step"] == "Generate readiness report"


def test_build_recall_response_matches_terms() -> None:
    response = build_recall_response("release report", [_reflection()])

    assert response["total_hits"] == 1
    assert response["hits"][0]["source_id"] == "S-1"
