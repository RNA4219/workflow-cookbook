from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tools.ci.check_birdseye_freshness import evaluate_birdseye_freshness
from tools.codemap.source_freshness import source_digest, summary_digest
from tools.codemap.update import UpdateOptions, run_update
from tools.codemap.update.constants import _REPO_ROOT


def fixture(root: Path, node: str = "README.md"):
    source = root / node
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("original source", encoding="utf-8")
    caps = root / "docs/birdseye/caps/source.json"
    caps.parent.mkdir(parents=True, exist_ok=True)
    capsule = {
        "id": node,
        "role": "doc",
        "summary": "original summary",
        "deps_in": [],
        "deps_out": [],
        "generated_at": "00001",
        "source_sha256": source_digest(root, node),
    }
    capsule["review"] = {
        "source_sha256": capsule["source_sha256"],
        "summary_sha256": summary_digest(capsule),
        "reviewed_at": "2026-09-10T00:00:00Z",
    }
    caps.write_text(json.dumps(capsule), encoding="utf-8")
    index = {
        "generated_at": "00001",
        "nodes": {node: {"mtime": "00001", "caps": "docs/birdseye/caps/source.json"}},
        "edges": [],
    }
    hot = {"generated_at": "00001", "nodes": []}
    for name, content in [("index.json", index), ("hot.json", hot)]:
        (root / "docs/birdseye" / name).write_text(json.dumps(content), encoding="utf-8")
    return source, caps, capsule, index, hot


def report(root, index, hot, strict=True):
    return evaluate_birdseye_freshness(
        index_doc=index, hot_doc=hot, repo_root=root, now=datetime.now(UTC), require_reviewed=strict
    )


@pytest.mark.parametrize("node", ["docs/migration-v1.1-to-v2.md", "docs/cli-usage.md", "docs/ROLLOUT.md"])
def test_non_hot_sources_are_checked(tmp_path, node):
    source, caps, capsule, index, hot = fixture(tmp_path, node)
    assert report(tmp_path, index, hot).failures == []
    source.write_text("changed source", encoding="utf-8")
    assert any("source_sha256 changed" in e for e in report(tmp_path, index, hot).failures)
    caps.unlink()
    assert any("missing caps file" in e for e in report(tmp_path, index, hot).failures)


def test_date_or_generation_does_not_certify_changed_summary(tmp_path):
    source, caps, capsule, index, hot = fixture(tmp_path)
    capsule["summary"] = "different meaning"
    capsule["review"]["reviewed_at"] = "2026-09-11T00:00:00Z"
    capsule["generated_at"] = "00002"
    caps.write_text(json.dumps(capsule), encoding="utf-8")
    index["generated_at"] = hot["generated_at"] = "00002"
    assert any("summary review required" in e for e in report(tmp_path, index, hot).failures)


def test_legacy_is_unknown_and_strict_mode_requires_review(tmp_path):
    source, caps, capsule, index, hot = fixture(tmp_path)
    capsule.pop("source_sha256")
    capsule.pop("review")
    caps.write_text(json.dumps(capsule), encoding="utf-8")
    ordinary = report(tmp_path, index, hot, strict=False)
    assert ordinary.failures == []
    assert any("freshness unknown" in w for w in ordinary.warnings)
    assert report(tmp_path, index, hot).failures


def test_update_observes_changed_source_but_preserves_unconfirmed_review(tmp_path, monkeypatch):
    source, caps, capsule, index, hot = fixture(tmp_path)
    old_review = capsule["review"].copy()
    source.write_text("changed source", encoding="utf-8")
    monkeypatch.setattr(_REPO_ROOT, "value", tmp_path)
    run_update(UpdateOptions(targets=(caps,), emit="index+caps", radius=0))
    result = json.loads(caps.read_text(encoding="utf-8"))
    assert result["source_sha256"] == source_digest(tmp_path, "README.md")
    assert result["summary"] == capsule["summary"]
    assert result["review"] == old_review
    updated_index = json.loads((tmp_path / "docs/birdseye/index.json").read_text(encoding="utf-8"))
    updated_hot = json.loads((tmp_path / "docs/birdseye/hot.json").read_text(encoding="utf-8"))
    assert any("summary review required" in e for e in report(tmp_path, updated_index, updated_hot).failures)
