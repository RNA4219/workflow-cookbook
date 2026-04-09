"""Tests for tools.ci.check_security_posture."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "ci" / "check_security_posture.py"
spec = importlib.util.spec_from_file_location("check_security_posture", MODULE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError("Failed to load check_security_posture module")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

validate_security_posture = module.validate_security_posture
ValidationResult = module.ValidationResult


def _seed_repo(root: Path) -> None:
    (root / ".github" / "workflows").mkdir(parents=True)
    (root / "docs" / "security").mkdir(parents=True)
    (root / ".github" / "dependabot.yml").write_text(
        "\n".join(
            [
                "version: 2",
                "updates:",
                "  - package-ecosystem: github-actions",
                '    directory: "/"',
                "    schedule:",
                "      interval: weekly",
            ]
        ),
        encoding="utf-8",
    )
    (root / ".github" / "workflows" / "security.yml").write_text("name: Security\n", encoding="utf-8")
    (root / "docs" / "security" / "SAC.md").write_text(
        "SAST\nSecrets\n依存\nContainer\n", encoding="utf-8"
    )
    (root / "docs" / "security" / "Security_Review_Checklist.md").write_text(
        "Dependabot\n", encoding="utf-8"
    )
    (root / "docs" / "requirements.md").write_text("## Security posture\n", encoding="utf-8")
    (root / "docs" / "spec.md").write_text("## Security baseline\n", encoding="utf-8")
    (root / "README.md").write_text("## Security\n", encoding="utf-8")


class TestValidationResult:
    def test_is_success_with_no_errors(self) -> None:
        result = ValidationResult()
        assert result.is_success is True

    def test_is_success_with_errors(self) -> None:
        result = ValidationResult(errors=["error"])
        assert result.is_success is False

    def test_emit_outputs_errors(self, capsys) -> None:
        result = ValidationResult(
            errors=["error1"],
            warnings=["warning1"],
        )
        result.emit()
        captured = capsys.readouterr()
        assert "error1" in captured.err
        assert "warning1" in captured.err


def test_validate_security_posture_pass(tmp_path: Path) -> None:
    _seed_repo(tmp_path)
    result = validate_security_posture(repo_root=tmp_path)
    assert result.errors == []


def test_validate_security_posture_reports_missing_dependabot_settings(tmp_path: Path) -> None:
    _seed_repo(tmp_path)
    (tmp_path / ".github" / "dependabot.yml").write_text("version: 2\nupdates: []\n", encoding="utf-8")

    result = validate_security_posture(repo_root=tmp_path)

    assert ".github/dependabot.yml does not configure github-actions updates." in result.errors
    assert ".github/dependabot.yml does not enforce a weekly update schedule." in result.errors


def test_validate_security_posture_reports_missing_files(tmp_path: Path) -> None:
    result = validate_security_posture(repo_root=tmp_path)

    assert len(result.errors) > 0
    assert any("missing" in e.lower() for e in result.errors)


def test_validate_security_posture_reports_missing_sac_markers(tmp_path: Path) -> None:
    _seed_repo(tmp_path)
    (tmp_path / "docs" / "security" / "SAC.md").write_text("Missing markers\n", encoding="utf-8")

    result = validate_security_posture(repo_root=tmp_path)

    assert any("SAC.md" in e for e in result.errors)


def test_validate_security_posture_reports_missing_checklist_content(tmp_path: Path) -> None:
    _seed_repo(tmp_path)
    (tmp_path / "docs" / "security" / "Security_Review_Checklist.md").write_text("No mention of dependency alerts\n", encoding="utf-8")

    result = validate_security_posture(repo_root=tmp_path)

    assert any("Checklist" in e for e in result.errors)


def test_validate_security_posture_reports_missing_requirements_section(tmp_path: Path) -> None:
    _seed_repo(tmp_path)
    (tmp_path / "docs" / "requirements.md").write_text("No security section\n", encoding="utf-8")

    result = validate_security_posture(repo_root=tmp_path)

    assert any("requirements.md" in e for e in result.errors)


def test_validate_security_posture_reports_missing_spec_section(tmp_path: Path) -> None:
    _seed_repo(tmp_path)
    (tmp_path / "docs" / "spec.md").write_text("No security baseline\n", encoding="utf-8")

    result = validate_security_posture(repo_root=tmp_path)

    assert any("spec.md" in e for e in result.errors)


def test_validate_security_posture_reports_missing_readme_security(tmp_path: Path) -> None:
    _seed_repo(tmp_path)
    (tmp_path / "README.md").write_text("No security mention\n", encoding="utf-8")

    result = validate_security_posture(repo_root=tmp_path)

    assert any("README.md" in e for e in result.errors)


def test_validate_security_posture_warns_without_github_token(tmp_path: Path) -> None:
    _seed_repo(tmp_path)

    result = validate_security_posture(
        repo_root=tmp_path,
        github_repo="owner/repo",
        github_token=None,
    )

    assert len(result.warnings) > 0
    assert any("token" in w.lower() for w in result.warnings)