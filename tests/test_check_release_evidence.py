from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "ci" / "check_release_evidence.py"
spec = importlib.util.spec_from_file_location("check_release_evidence", MODULE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError("Failed to load check_release_evidence module")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

validate_release_evidence = module.validate_release_evidence


def _write_release_doc(path: Path, version: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                f"# workflow-cookbook v{version}",
                "",
                "## Summary",
                "",
                "- sample",
            ]
        ),
        encoding="utf-8",
    )


def test_validate_release_evidence_pass(tmp_path: Path, monkeypatch) -> None:
    changelog = tmp_path / "CHANGELOG.md"
    releases_dir = tmp_path / "docs" / "releases"
    changelog.write_text(
        "\n".join(
            [
                "# Changelog",
                "",
                "## [Unreleased]",
                "",
                "## 1.0.0 - 2025-10-16",
            ]
        ),
        encoding="utf-8",
    )
    _write_release_doc(releases_dir / "v1.0.0.md", "1.0.0")
    monkeypatch.setattr(module, "load_git_tag_versions", lambda repo_root: {"1.0.0"})

    result = validate_release_evidence(
        repo_root=tmp_path,
        changelog_path=changelog,
        releases_dir=releases_dir,
    )

    assert result.errors == []


def test_validate_release_evidence_reports_missing_release_doc(tmp_path: Path, monkeypatch) -> None:
    changelog = tmp_path / "CHANGELOG.md"
    releases_dir = tmp_path / "docs" / "releases"
    releases_dir.mkdir(parents=True)
    changelog.write_text(
        "\n".join(
            [
                "# Changelog",
                "",
                "## [Unreleased]",
                "",
                "## 1.0.0 - 2025-10-16",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "load_git_tag_versions", lambda repo_root: {"1.0.0"})

    result = validate_release_evidence(
        repo_root=tmp_path,
        changelog_path=changelog,
        releases_dir=releases_dir,
    )

    assert "CHANGELOG version 1.0.0 is missing docs/releases/v1.0.0.md." in result.errors
    assert "Git tag v1.0.0 is missing docs/releases/v1.0.0.md." in result.errors


def test_validate_release_evidence_reports_title_mismatch(tmp_path: Path, monkeypatch) -> None:
    changelog = tmp_path / "CHANGELOG.md"
    releases_dir = tmp_path / "docs" / "releases"
    changelog.write_text(
        "\n".join(["# Changelog", "", "## [Unreleased]", "", "## 1.0.0 - 2025-10-16"]),
        encoding="utf-8",
    )
    _write_release_doc(releases_dir / "v1.0.0.md", "1.0.1")
    monkeypatch.setattr(module, "load_git_tag_versions", lambda repo_root: {"1.0.0"})

    result = validate_release_evidence(
        repo_root=tmp_path,
        changelog_path=changelog,
        releases_dir=releases_dir,
    )

    assert f"{releases_dir / 'v1.0.0.md'} title does not match filename version v1.0.0." in result.errors
