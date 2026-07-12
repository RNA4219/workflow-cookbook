from __future__ import annotations

from pathlib import Path

import yaml

from tools.ci.check_repo_responsibilities import validate_manifest

ROOT = Path(__file__).resolve().parents[1]


def test_canonical_manifest_has_one_owner_per_capability() -> None:
    payload = yaml.safe_load((ROOT / "governance/repo-responsibilities.yaml").read_text(encoding="utf-8"))
    assert validate_manifest(payload) == []


def test_duplicate_capability_is_rejected() -> None:
    payload = {
        "schema_version": "repo-responsibilities/v1",
        "capabilities": [
            {"capability": "gate", "owner_repo": "a", "canonical_path": "README.md", "responsibility": "a"},
            {"capability": "gate", "owner_repo": "b", "canonical_path": "README.md", "responsibility": "b"},
        ],
    }
    assert any("exactly one" in error for error in validate_manifest(payload))
