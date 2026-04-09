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


def _seed_repo(root: Path) -> None:
    (root / ".github" / "workflows" / "reusable").mkdir(parents=True)
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
    (root / ".github" / "workflows" / "reusable" / "security-ci.yml").write_text(
        "name: Security CI\n", encoding="utf-8"
    )
    (root / "docs" / "security" / "SAC.md").write_text(
        "SAST\nSecrets\n依存\nContainer\n", encoding="utf-8"
    )
    (root / "docs" / "security" / "Security_Review_Checklist.md").write_text(
        "Dependabot\n", encoding="utf-8"
    )
    (root / "docs" / "requirements.md").write_text("## Security posture\n", encoding="utf-8")
    (root / "docs" / "spec.md").write_text("## Security baseline\n", encoding="utf-8")
    (root / "README.md").write_text("## Security\n", encoding="utf-8")


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
