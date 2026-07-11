from __future__ import annotations

import json
import subprocess
from datetime import date, timedelta
from pathlib import Path

import pytest

from tools.ci import check_dependency_exceptions as exceptions
from tools.security import generate_sbom
from tools.security import visualize_dependencies as dependencies


def test_generate_sbom_parses_pins_and_writes_cyclonedx(tmp_path: Path) -> None:
    requirements = tmp_path / "requirements.txt"
    requirements.write_text(
        "# comment\nPyYAML==6.0.3\nruff>=0.15\nDemo-Pkg==1.2.3\n",
        encoding="utf-8",
    )
    output = tmp_path / "out" / "sbom.json"
    assert generate_sbom.parse_requirements(tmp_path / "missing.txt") == []
    components = generate_sbom.parse_requirements(requirements)
    assert [item["name"] for item in components] == ["PyYAML", "Demo-Pkg"]
    assert generate_sbom.generate_sbom(requirements, output, "Demo-App", "2.0") == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["bomFormat"] == "CycloneDX"
    assert payload["metadata"]["component"]["purl"] == "pkg:pypi/demo_app@2.0"
    assert len(payload["components"]) == 2
    cli_output = tmp_path / "cli.json"
    assert generate_sbom.main([
        "--requirements", str(requirements), "--output", str(cli_output),
        "--project-name", "CLI", "--project-version", "3",
    ]) == 0
    assert cli_output.exists()


def test_dependency_visualizer_parses_filters_and_formats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "requirements.txt").write_text(
        "# runtime\nPyYAML==6.0\nrequests>=2\n", encoding="utf-8"
    )
    (tmp_path / "pyproject.toml").write_text(
        "[project.optional-dependencies]\ndev = [\n  \"pytest==9\",\n  \"mypy==2\",\n]\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(dependencies, "_REPO_ROOT", tmp_path)
    direct = dependencies._get_direct_dependencies()
    assert direct == {"runtime": ["PyYAML", "requests"], "dev": ["pytest", "mypy"]}
    tree = json.dumps([
        {
            "package_name": "pyyaml", "installed_version": "6.0",
            "required_version": ">=6",
            "dependencies": [{
                "package_name": "child", "installed_version": "1",
                "required_version": "any", "dependencies": [],
            }],
        },
        {"package_name": "other", "installed_version": "1", "dependencies": []},
    ])
    assert dependencies._filter_deps("not-json", ["pyyaml"]) == []
    filtered = dependencies._filter_deps(tree, ["PyYAML"])
    assert len(filtered) == 1
    formatted = dependencies._format_dep_tree(filtered)
    assert "**pyyaml**" in formatted
    assert "child" in formatted
    markdown = dependencies._generate_markdown(direct, tree)
    assert "Runtime Dependencies" in markdown
    assert "| PyYAML | pinned |" in markdown


def test_dependency_visualizer_main_and_pipdeptree_fallbacks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def missing(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError

    monkeypatch.setattr(dependencies.subprocess, "run", missing)
    assert dependencies._run_pipdeptree() == ""

    def failed(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(1, ["pipdeptree"])

    monkeypatch.setattr(dependencies.subprocess, "run", failed)
    assert dependencies._run_pipdeptree() == ""
    monkeypatch.setattr(
        dependencies, "_get_direct_dependencies",
        lambda: {"runtime": ["PyYAML"], "dev": ["pytest"]},
    )
    monkeypatch.setattr(dependencies, "_run_pipdeptree", lambda: "")
    output = tmp_path / "deps.md"
    raw_json = tmp_path / "deps.json"
    assert dependencies.main(["--output", str(output), "--json", str(raw_json)]) == 0
    assert output.exists()
    assert raw_json.read_text(encoding="utf-8") == ""


def test_dependency_exception_parsing_and_checks(tmp_path: Path) -> None:
    assert exceptions.parse_front_matter("plain") == {}
    assert exceptions.parse_front_matter("---\nbroken") == {}
    assert exceptions.parse_front_matter("---\n: bad\n---") == {}
    future = date.today() + timedelta(days=60)
    re_eval = date.today() + timedelta(days=30)
    content = f"""---
next_review_due: {future.isoformat()}
---

### EXC-001: temporary
- **期限**: {future.isoformat()}
- **再評価日**: {re_eval.isoformat()}
"""
    parsed = exceptions.parse_exception_blocks(content)
    assert parsed[0]["id"] == "EXC-001"
    missing_ok, missing_issues = exceptions.check_exceptions(tmp_path / "missing.md")
    assert not missing_ok
    assert "not found" in missing_issues[0]
    registry = tmp_path / "exceptions.md"
    registry.write_text(content, encoding="utf-8")
    ok, issues = exceptions.check_exceptions(registry)
    assert ok
    assert issues == []
    expired = date.today() - timedelta(days=45)
    registry.write_text(
        f"""---
next_review_due: {expired.isoformat()}
---

### EXC-002: expired
- **期限**: {expired.isoformat()}
- **再評価日**: {expired.isoformat()}
""",
        encoding="utf-8",
    )
    ok, issues = exceptions.check_exceptions(registry, max_overdue_days=30)
    assert not ok
    assert any("expired" in issue for issue in issues)
    assert any("re-evaluation overdue" in issue for issue in issues)
