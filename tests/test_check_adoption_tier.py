from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.ci import check_adoption_tier as checker

ROOT = Path(__file__).resolve().parents[1]


def _write(path: Path, content: str = "# Doc\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _front_matter(template_version: str) -> str:
    return "\n".join(
        [
            "---",
            "intent_id: INT-TEST",
            "owner: docs-core",
            "status: active",
            "last_reviewed_at: 2026-07-01",
            "next_review_due: 2026-08-01",
            f"template_version: {template_version}",
            "---",
            "",
            "# Doc",
            "",
        ]
    )


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "tools.ci.check_adoption_tier", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_adoption_tier_detects_tier_two_and_missing_tier_three(tmp_path: Path) -> None:
    for name in ["README.md", "HUB.codex.md", "BLUEPRINT.md", "RUNBOOK.md", "GUARDRAILS.md", "EVALUATION.md"]:
        _write(tmp_path / name)

    result = _run_cli("--repo", str(tmp_path), "--json")

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["current_tier"] == 2
    assert "docs/tasks" in payload["missing_for_next_tier"]


def test_adoption_tier_check_fails_below_min_tier(tmp_path: Path) -> None:
    _write(tmp_path / "README.md")

    result = _run_cli("--repo", str(tmp_path), "--min-tier", "1", "--check")

    assert result.returncode == 1


def test_adoption_tier_detects_template_drift(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    template_root = tmp_path / "templates"
    _write(repo / "README.md")
    _write(repo / "HUB.codex.md", _front_matter("0.9.0"))
    _write(repo / "BLUEPRINT.md")
    _write(template_root / "HUB.codex.md.template", _front_matter("1.0.0"))

    result = _run_cli(
        "--repo",
        str(repo),
        "--template-root",
        str(template_root),
        "--check-drift",
        "--check",
        "--json",
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["drifted"] is True
    assert payload["drift_checks"][0]["path"] == "HUB.codex.md"



def _full_repo(repo: Path) -> None:
    for name in checker._cumulative_required_paths(2):
        _write(repo / name)
    _write(repo / "docs/tasks/TASK-fixture.md", "# A task\n")
    _write(repo / "docs/acceptance/AC-fixture.md", "# Acceptance criteria\n")
    _write(repo / "docs/birdseye/index.json", json.dumps({"nodes": {"README.md": {"role": "overview"}}}))
    _write(repo / "docs/birdseye/hot.json", json.dumps({"nodes": [{"id": "README.md"}]}))
    _write(repo / "docs/birdseye/caps/readme.json", json.dumps({"id": "README.md", "summary": "Entry point"}))


def test_tier_three_requires_valid_content(tmp_path: Path) -> None:
    _full_repo(tmp_path)
    result = _run_cli("--repo", str(tmp_path), "--min-tier", "3", "--check", "--json")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["current_tier"] == 3
    assert payload["operational_compliance"] == "not_evaluated"
    assert payload["drift_status"] == "not_checked"


def test_directories_cannot_impersonate_required_files(tmp_path: Path) -> None:
    for name in checker._cumulative_required_paths(3):
        (tmp_path / name).mkdir(parents=True, exist_ok=True)
    result = _run_cli("--repo", str(tmp_path), "--min-tier", "3", "--check", "--json")
    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["current_tier"] == -1
    assert payload["required_by_tier"]["0"][0]["reason"] == "wrong_kind"
    assert payload["invalid_for_next_tier"][0]["path"] == "README.md"


@pytest.mark.parametrize("name", checker._cumulative_required_paths(2) + [
    "docs/tasks/TASK-fixture.md", "docs/acceptance/AC-fixture.md", "docs/birdseye/index.json",
    "docs/birdseye/hot.json", "docs/birdseye/caps/readme.json",
])
def test_empty_documents_do_not_qualify_as_full(tmp_path: Path, name: str) -> None:
    _full_repo(tmp_path)
    (tmp_path / name).write_text(" \n\t", encoding="utf-8")
    result = _run_cli("--repo", str(tmp_path), "--min-tier", "3", "--check", "--json")
    assert result.returncode == 1
    assert json.loads(result.stdout)["current_tier"] < 3


@pytest.mark.parametrize("directory", ["docs/tasks", "docs/acceptance", "docs/birdseye/caps"])
def test_empty_record_directories_do_not_qualify_as_full(tmp_path: Path, directory: str) -> None:
    _full_repo(tmp_path)
    for item in (tmp_path / directory).iterdir():
        item.unlink()
    result = checker.assess_repo(tmp_path)
    assert result["current_tier"] == 2
    assert next(item for item in result["required_by_tier"]["3"] if item["path"] == directory)["reason"] == "empty"


@pytest.mark.parametrize("path,value", [
    ("docs/birdseye/index.json", "{"),
    ("docs/birdseye/index.json", "[]"),
    ("docs/birdseye/index.json", '{"nodes": {}}'),
    ("docs/birdseye/index.json", '{"nodes": {"README.md": null}}'),
    ("docs/birdseye/hot.json", '{"nodes": []}'),
    ("docs/birdseye/hot.json", '{"nodes": ["README.md"]}'),
    ("docs/birdseye/caps/readme.json", '{"id": "README.md"}'),
    ("docs/birdseye/caps/readme.json", '{"id": "README.md", "summary": " "}'),
])
def test_birdseye_json_requires_structure(tmp_path: Path, path: str, value: str) -> None:
    _full_repo(tmp_path)
    _write(tmp_path / path, value)
    result = checker.assess_repo(tmp_path)
    assert result["current_tier"] == 2
    assert any("invalid_json" in (item["reason"] or "") for item in result["required_by_tier"]["3"])


@pytest.mark.parametrize("doc_version,template_version,expected", [
    ("1.0", "1.0", "current"), ("0.9", "1.0", "drifted"),
    (None, "1.0", "unknown"), ("1.0", None, "unknown"), (None, None, "unknown"),
])
def test_drift_status_and_exit_distinguish_unknown(tmp_path: Path, doc_version: str | None, template_version: str | None, expected: str) -> None:
    repo, templates = tmp_path / "repo", tmp_path / "templates"
    _write(repo / "HUB.codex.md", _front_matter(doc_version) if doc_version else "# No version\n")
    _write(templates / "HUB.codex.md.template", _front_matter(template_version) if template_version else "# No version\n")
    result = _run_cli("--repo", str(repo), "--template-root", str(templates), "--check-drift", "--check", "--json")
    assert result.returncode == (0 if expected == "current" else 1)
    payload = json.loads(result.stdout)
    assert payload["drift_status"] == expected
    assert payload["drifted"] is (expected == "drifted")
    assert payload["drift_checks"][0]["status"] == expected


@pytest.mark.parametrize("case", ["missing_template", "no_documents", "wrong_kind", "unreadable_document", "unreadable_template"])
def test_unavailable_drift_comparison_is_not_a_pass(tmp_path: Path, case: str) -> None:
    repo, templates = tmp_path / "repo", tmp_path / "templates"
    repo.mkdir()
    templates.mkdir()
    if case != "no_documents":
        _write(repo / "HUB.codex.md", _front_matter("1.0"))
    if case != "missing_template":
        _write(templates / "HUB.codex.md.template", _front_matter("1.0"))
    if case == "wrong_kind":
        (repo / "HUB.codex.md").unlink()
        (repo / "HUB.codex.md").mkdir()
    elif case == "unreadable_document":
        (repo / "HUB.codex.md").write_bytes(b"\xff")
    elif case == "unreadable_template":
        (templates / "HUB.codex.md.template").write_bytes(b"\xff")
    result = _run_cli("--repo", str(repo), "--template-root", str(templates), "--check-drift", "--check", "--json")
    assert result.returncode == 1
    assert json.loads(result.stdout)["drift_status"] == "unknown"


def test_unknown_is_visible_in_text_and_mixed_results(tmp_path: Path) -> None:
    repo, templates = tmp_path / "repo", tmp_path / "templates"
    _write(repo / "HUB.codex.md", "# Missing version\n")
    _write(repo / "BLUEPRINT.md", _front_matter("0.9"))
    _write(templates / "HUB.codex.md.template", _front_matter("1.0"))
    _write(templates / "BLUEPRINT.md.template", _front_matter("1.0"))
    result = _run_cli("--repo", str(repo), "--template-root", str(templates), "--check-drift", "--check")
    assert result.returncode == 1
    assert "UNKNOWN HUB.codex.md" in result.stdout
    assert "DRIFTED BLUEPRINT.md" in result.stdout
    assert "Template drift: drifted" in result.stdout


def test_empty_repo_list_is_not_a_successful_gate(tmp_path: Path) -> None:
    path = tmp_path / "repos.json"
    _write(path, "[]")
    result = _run_cli("--repo-list", str(path), "--check-drift", "--check")
    assert result.returncode == 1
    assert "non-empty" in result.stderr



@pytest.mark.parametrize("raw", ["", "null", "~", "# version omitted", "null # version omitted"])
def test_null_and_comment_only_versions_remain_unknown(tmp_path: Path, raw: str) -> None:
    repo, templates = tmp_path / "repo", tmp_path / "templates"
    content = f"---\ntemplate_version: {raw}\n---\n# Doc\n"
    _write(repo / "HUB.codex.md", content)
    _write(templates / "HUB.codex.md.template", content)
    result = checker.assess_repo(repo, check_drift=True, template_root=templates)
    assert result["drift_status"] == "unknown"


def test_read_failure_and_file_instead_of_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _full_repo(tmp_path)
    original = Path.read_text
    def fail(path: Path, *args: object, **kwargs: object) -> str:
        if path.name == "README.md":
            raise PermissionError("fixture")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", fail)
    result = checker.assess_repo(tmp_path)
    assert result["current_tier"] == -1
    assert result["required_by_tier"]["0"][0]["reason"] == "unreadable"
    monkeypatch.undo()
    other = tmp_path / "other"
    _write(other / "docs/tasks", "not a directory")
    assert checker._path_status(other, "docs/tasks")["reason"] == "wrong_kind"


@pytest.mark.parametrize("extension", ["json", "JSON", "JsOn"])
@pytest.mark.parametrize("content,valid", [
    ('{"id": "README.md", "summary": "Entry point"}', True),
    ("# Not JSON", False),
    ('{"id": "README.md"}', False),
])
def test_capsule_extensions_have_identical_content_checks(tmp_path: Path, extension: str, content: str, valid: bool) -> None:
    _full_repo(tmp_path)
    capsule = tmp_path / "docs/birdseye/caps/readme.json"
    capsule.unlink()
    _write(capsule.with_name(f"readme.{extension}"), content)
    result = checker.assess_repo(tmp_path)
    assert (result["current_tier"] == 3) is valid
    status = next(item for item in result["required_by_tier"]["3"] if item["path"] == "docs/birdseye/caps")
    assert status["valid"] is valid
    if not valid:
        assert "invalid_json" in status["reason"]


def test_valid_capsule_does_not_hide_invalid_uppercase_member(tmp_path: Path) -> None:
    _full_repo(tmp_path)
    _write(tmp_path / "docs/birdseye/caps/broken.JSON", "# Not JSON")
    assert checker.assess_repo(tmp_path)["current_tier"] == 2


@pytest.mark.parametrize("directory", ["docs/tasks", "docs/acceptance"])
@pytest.mark.parametrize("extension", ["MD", "Md"])
def test_markdown_record_extensions_are_os_independent(tmp_path: Path, directory: str, extension: str) -> None:
    _full_repo(tmp_path)
    for member in (tmp_path / directory).iterdir():
        member.unlink()
    _write(tmp_path / directory / f"record.{extension}", "# A valid record\n")
    assert checker.assess_repo(tmp_path)["current_tier"] == 3


@pytest.mark.parametrize("raw,expected", [
    ('"1.0.0" # adopted version', "1.0.0"),
    ("'1.0.0' # adopted version", "1.0.0"),
    ('"1.0.0"\t# adopted version', "1.0.0"),
    ("1.0.0\t# adopted version", "1.0.0"),
    ('"1.0#release" # comment', "1.0#release"),
    ("'1.0 # release' # comment", "1.0 # release"),
    ("'1.0''release' # comment", "1.0'release"),
    ('"1.0\\\"release" # comment', '1.0"release'),
])
def test_quoted_versions_preserve_value_and_ignore_only_external_comments(tmp_path: Path, raw: str, expected: str) -> None:
    repo, templates = tmp_path / "repo", tmp_path / "templates"
    _write(repo / "HUB.codex.md", _front_matter(raw))
    _write(templates / "HUB.codex.md.template", _front_matter(json.dumps(expected)))
    result = checker.assess_repo(repo, check_drift=True, template_root=templates)
    assert result["drift_status"] == "current"
    assert result["drift_checks"][0]["document_template_version"] == expected


@pytest.mark.parametrize("raw", [
    '"1.0.0', "'1.0.0", '"1.0.0" trailing', "'1.0.0' trailing", '["1.0.0"]',
    '{version: "1.0.0"}', "|", ">", "&version 1.0.0", "*version",
])
def test_malformed_or_non_scalar_version_is_unknown(tmp_path: Path, raw: str) -> None:
    repo, templates = tmp_path / "repo", tmp_path / "templates"
    _write(repo / "HUB.codex.md", _front_matter(raw))
    _write(templates / "HUB.codex.md.template", _front_matter(raw))
    result = checker.assess_repo(repo, check_drift=True, template_root=templates)
    assert result["drift_status"] == "unknown"


@pytest.mark.parametrize("payload", [[""], [" \t"], [{"repo": ""}], [{"repo": "\n"}], [None], [{"path": "."}], [".", None]])
def test_repo_list_rejects_every_invalid_entry(tmp_path: Path, payload: object) -> None:
    path = tmp_path / "repos.json"
    _write(path, json.dumps(payload))
    result = _run_cli("--repo-list", str(path), "--check", "--json")
    assert result.returncode == 1
    assert result.stdout == ""
    assert "repo-list item" in result.stderr
    assert "Traceback" not in result.stderr



def test_json_role_requires_json_even_with_unexpected_suffix(tmp_path: Path) -> None:
    path = tmp_path / "capsule.data"
    _write(path, "# Not JSON\n")
    assert checker._content_error(path, "docs/birdseye/caps") == "invalid_json"


@pytest.mark.parametrize("front_matter", [
    '---broken\ntemplate_version: 1.0.0\n---\n',
    '---\ntemplate_version: 1.0.0\n',
    '---\ntemplate_version: 1.0.0\ntemplate_version: 1.0.0\n---\n',
    '---\nmetadata:\n  template_version: 1.0.0\n---\n',
])
def test_unavailable_top_level_version_is_unknown(tmp_path: Path, front_matter: str) -> None:
    repo, templates = tmp_path / "repo", tmp_path / "templates"
    _write(repo / "HUB.codex.md", front_matter)
    _write(templates / "HUB.codex.md.template", front_matter)
    assert checker.assess_repo(repo, check_drift=True, template_root=templates)["drift_status"] == "unknown"


def test_quoted_version_cli_remains_standalone(tmp_path: Path) -> None:
    repo, templates = tmp_path / "repo", tmp_path / "templates"
    _write(repo / "HUB.codex.md", _front_matter('"1.0.0" # adopted version'))
    _write(templates / "HUB.codex.md.template", _front_matter("1.0.0"))
    result = subprocess.run(
        [sys.executable, "-I", "-S", str(ROOT / "tools/ci/check_adoption_tier.py"),
         "--repo", str(repo), "--template-root", str(templates), "--check-drift", "--check", "--json"],
        cwd=tmp_path, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["drift_status"] == "current"



@pytest.mark.parametrize("raw", ['""', "''", '"   "', "'   '", '"\\t"', "'\t'"])
def test_quoted_blank_versions_remain_unknown(tmp_path: Path, raw: str) -> None:
    repo, templates = tmp_path / "repo", tmp_path / "templates"
    _write(repo / "HUB.codex.md", _front_matter(raw))
    _write(templates / "HUB.codex.md.template", _front_matter(raw))
    result = checker.assess_repo(repo, check_drift=True, template_root=templates)
    assert result["drift_status"] == "unknown"
    assert result["drifted"] is False
