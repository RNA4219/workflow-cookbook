"""Tests for tools.ci.generate_evidence_report."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from textwrap import dedent

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import tools.ci.generate_evidence_report as ev_module


class TestParseFrontMatter:
    def test_parses_yaml_front_matter(self) -> None:
        content = dedent("""
            ---
            acceptance_id: AC-001
            task_id: TASK-001
            ---
            # Content
            """).strip()
        result = ev_module._parse_front_matter(content)
        assert result.get("acceptance_id") == "AC-001"

    def test_returns_empty_for_no_front_matter(self) -> None:
        result = ev_module._parse_front_matter("# No front matter")
        assert result == {}


class TestScanAcceptances:
    def test_scans_acceptance_files(self, tmp_path: Path) -> None:
        acc_dir = tmp_path / "acceptance"
        acc_dir.mkdir()
        acc_file = acc_dir / "AC-001.md"
        acc_file.write_text(dedent("""
            ---
            acceptance_id: AC-001
            task_id: TASK-001
            status: approved
            ---
            """).strip())

        acceptances = ev_module.scan_acceptances(acc_dir)
        assert len(acceptances) == 1
        assert acceptances[0].acceptance_id == "AC-001"
        assert acceptances[0].task_id == "TASK-001"

    def test_returns_empty_for_missing_dir(self, tmp_path: Path) -> None:
        acceptances = ev_module.scan_acceptances(tmp_path / "nonexistent")
        assert acceptances == []


class TestScanEvidences:
    def test_scans_evidence_json_files(self, tmp_path: Path) -> None:
        ev_dir = tmp_path / "evidence"
        ev_dir.mkdir()
        ev_file = ev_dir / "evidence.json"
        ev_file.write_text(json.dumps({
            "evidence_id": "EV-001",
            "taskSeedId": "TASK-001",
            "startTime": "2026-04-10T00:00:00Z",
            "model": "test-model",
        }))

        evidences = ev_module.scan_evidences(ev_dir)
        assert len(evidences) == 1
        assert evidences[0].evidence_id == "EV-001"
        assert evidences[0].task_id == "TASK-001"

    def test_returns_empty_for_missing_dir(self, tmp_path: Path) -> None:
        evidences = ev_module.scan_evidences(tmp_path / "nonexistent")
        assert evidences == []

    def test_skips_invalid_json(self, tmp_path: Path) -> None:
        ev_dir = tmp_path / "evidence"
        ev_dir.mkdir()
        (ev_dir / "invalid.json").write_text("not json")

        evidences = ev_module.scan_evidences(ev_dir)
        assert evidences == []

    def test_uses_file_stem_as_evidence_id_fallback(self, tmp_path: Path) -> None:
        ev_dir = tmp_path / "evidence"
        ev_dir.mkdir()
        (ev_dir / "my-evidence.json").write_text(json.dumps({
            "taskSeedId": "TASK-001",
        }))

        evidences = ev_module.scan_evidences(ev_dir)
        assert len(evidences) == 1
        assert evidences[0].evidence_id == "my-evidence"


class TestScanReleases:
    def test_scans_release_refs(self, tmp_path: Path) -> None:
        release_dir = tmp_path / "releases"
        release_dir.mkdir()
        (release_dir / "v1.md").write_text(
            dedent("""
                ---
                release_id: v1
                status: approved
                ---
                # v1
                Supported by AC-001 and EV-001.
                """).strip(),
            encoding="utf-8",
        )

        releases = ev_module.scan_releases(release_dir)

        assert releases[0].release_id == "v1"
        assert "AC-001" in releases[0].references


class TestGenerateReport:
    def test_links_by_task_id(self, tmp_path: Path) -> None:
        # Create actual files to avoid relative_to error
        acc_dir = tmp_path / "docs" / "acceptance"
        acc_dir.mkdir(parents=True)
        acc_file = acc_dir / "AC-001.md"
        acc_file.write_text("---\n---")

        ev_dir = tmp_path / ".workflow-cache"
        ev_dir.mkdir(parents=True)
        ev_file = ev_dir / "ev.json"
        ev_file.write_text("{}")

        acceptances = [
            ev_module.AcceptanceSummary(
                acceptance_id="AC-001",
                task_id="TASK-001",
                status="approved",
                file_path=acc_file,
            ),
        ]
        evidences = [
            ev_module.EvidenceSummary(
                evidence_id="EV-001",
                task_id="TASK-001",
                timestamp="2026-04-10T00:00:00Z",
                model="test-model",
                file_path=ev_file,
            ),
        ]

        original_repo_root = ev_module._REPO_ROOT
        ev_module._REPO_ROOT = tmp_path

        report = ev_module.generate_report(acceptances, evidences)

        ev_module._REPO_ROOT = original_repo_root

        assert len(report.linked) == 1
        assert report.linked[0]["acceptance_id"] == "AC-001"
        assert report.linked[0]["evidence_id"] == "EV-001"
        assert report.unlinked_acceptances == []
        assert report.unlinked_evidences == []

    def test_reports_unlinked_acceptances(self, tmp_path: Path) -> None:
        acc_dir = tmp_path / "docs" / "acceptance"
        acc_dir.mkdir(parents=True)
        acc_file = acc_dir / "AC-001.md"
        acc_file.write_text("---\n---")

        acceptances = [
            ev_module.AcceptanceSummary(
                acceptance_id="AC-001",
                task_id="TASK-001",
                status="approved",
                file_path=acc_file,
            ),
        ]
        evidences = []

        original_repo_root = ev_module._REPO_ROOT
        ev_module._REPO_ROOT = tmp_path

        report = ev_module.generate_report(acceptances, evidences)

        ev_module._REPO_ROOT = original_repo_root

        assert len(report.unlinked_acceptances) == 1
        assert report.unlinked_acceptances[0].acceptance_id == "AC-001"

    def test_reports_unlinked_evidences(self, tmp_path: Path) -> None:
        ev_dir = tmp_path / ".workflow-cache"
        ev_dir.mkdir(parents=True)
        ev_file = ev_dir / "ev.json"
        ev_file.write_text("{}")

        acceptances = []
        evidences = [
            ev_module.EvidenceSummary(
                evidence_id="EV-001",
                task_id="TASK-001",
                timestamp="2026-04-10T00:00:00Z",
                model="test-model",
                file_path=ev_file,
            ),
        ]

        original_repo_root = ev_module._REPO_ROOT
        ev_module._REPO_ROOT = tmp_path

        report = ev_module.generate_report(acceptances, evidences)

        ev_module._REPO_ROOT = original_repo_root

        assert len(report.unlinked_evidences) == 1

    def test_generates_release_readiness(self, tmp_path: Path) -> None:
        acc_file = tmp_path / "docs" / "acceptance" / "AC-001.md"
        acc_file.parent.mkdir(parents=True)
        acc_file.write_text("---\n---")
        ev_file = tmp_path / ".workflow-cache" / "ev.json"
        ev_file.parent.mkdir(parents=True)
        ev_file.write_text("{}")
        release_file = tmp_path / "docs" / "releases" / "v1.md"
        release_file.parent.mkdir(parents=True)
        release_file.write_text("# v1\nAC-001\nEV-001\n")

        original_repo_root = ev_module._REPO_ROOT
        ev_module._REPO_ROOT = tmp_path
        report = ev_module.generate_report(
            [
                ev_module.AcceptanceSummary("AC-001", "TASK-001", "approved", acc_file),
            ],
            [
                ev_module.EvidenceSummary("EV-001", "TASK-001", "", "model", ev_file),
            ],
            releases=[
                ev_module.ReleaseSummary("v1", "v1", "approved", release_file, ["AC-001", "EV-001"]),
            ],
            security={"errors": []},
            metrics={"errors": []},
        )
        ev_module._REPO_ROOT = original_repo_root

        assert report.readiness["status"] == "ready"
        assert len(report.release_links) == 2


class TestFormatMarkdownReport:
    def test_formats_report_as_markdown(self, tmp_path: Path) -> None:
        acc_dir = tmp_path / "docs" / "acceptance"
        acc_dir.mkdir(parents=True)
        acc_file = acc_dir / "AC-001.md"
        acc_file.write_text("---\n---")

        acceptances = [
            ev_module.AcceptanceSummary(
                acceptance_id="AC-001",
                task_id="TASK-001",
                status="approved",
                file_path=acc_file,
            ),
        ]

        original_repo_root = ev_module._REPO_ROOT
        ev_module._REPO_ROOT = tmp_path

        report = ev_module.generate_report(acceptances, [])

        ev_module._REPO_ROOT = original_repo_root

        markdown = ev_module.format_markdown_report(report)
        assert "# Evidence Report" in markdown
        assert "## Summary" in markdown
        assert "Acceptance Records: 1" in markdown
        assert "## Unlinked Acceptances" in markdown
