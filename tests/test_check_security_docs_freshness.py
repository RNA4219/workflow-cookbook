"""Tests for tools.ci.check_security_docs_freshness."""

from __future__ import annotations

import sys
from pathlib import Path
from textwrap import dedent

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.ci.check_security_docs_freshness import (
    _parse_front_matter,
    _count_releases_since_date,
    check_security_docs,
    main,
    DocFreshness,
)


class TestParseFrontMatter:
    def test_parses_yaml_front_matter(self) -> None:
        content = dedent("""
            ---
            last_reviewed: 2026-04-10
            next_review_due: 2026-05-10
            ---
            # Content
            """).strip()
        result = _parse_front_matter(content)
        assert result.get("last_reviewed") == "2026-04-10"
        assert result.get("next_review_due") == "2026-05-10"

    def test_returns_empty_for_no_front_matter(self) -> None:
        result = _parse_front_matter("# No front matter")
        assert result == {}

    def test_handles_quoted_values(self) -> None:
        content = dedent("""
            ---
            last_reviewed: "2026-04-10"
            next_review_due: '2026-05-10'
            ---
            """).strip()
        result = _parse_front_matter(content)
        assert result.get("last_reviewed") == "2026-04-10"
        assert result.get("next_review_due") == "2026-05-10"


class TestCountReleasesSinceDate:
    def test_counts_releases_after_date(self, tmp_path: Path) -> None:
        releases_dir = tmp_path / "releases"
        releases_dir.mkdir()
        (releases_dir / "v1.0.0.md").write_text("date: 2026-04-15\n")
        (releases_dir / "v0.9.0.md").write_text("date: 2026-03-01\n")

        count = _count_releases_since_date(releases_dir, "2026-04-01")
        assert count == 1

    def test_returns_zero_for_missing_dir(self, tmp_path: Path) -> None:
        count = _count_releases_since_date(tmp_path / "nonexistent", "2026-04-01")
        assert count == 0

    def test_handles_invalid_date(self, tmp_path: Path) -> None:
        releases_dir = tmp_path / "releases"
        releases_dir.mkdir()
        (releases_dir / "v1.0.0.md").write_text("date: invalid\n")

        count = _count_releases_since_date(releases_dir, "2026-04-01")
        assert count == 0

    def test_handles_invalid_since_date(self, tmp_path: Path) -> None:
        releases_dir = tmp_path / "releases"
        releases_dir.mkdir()
        (releases_dir / "v1.0.0.md").write_text("date: 2026-04-10\n")

        count = _count_releases_since_date(releases_dir, "invalid")
        assert count == 0


class TestCheckSecurityDocs:
    def test_checks_security_docs(self, tmp_path: Path) -> None:
        security_dir = tmp_path / "security"
        security_dir.mkdir()
        sec_file = security_dir / "policy.md"
        sec_file.write_text(dedent("""
            ---
            last_reviewed_at: 2026-04-01
            next_review_due: 2026-05-01
            ---
            # Security Policy
            """).strip())

        releases_dir = tmp_path / "releases"
        releases_dir.mkdir()

        results = check_security_docs(security_dir, releases_dir)
        assert len(results) == 1
        assert results[0].last_reviewed == "2026-04-01"
        assert results[0].next_review_due == "2026-05-01"

    def test_returns_empty_for_missing_dir(self, tmp_path: Path) -> None:
        results = check_security_docs(tmp_path / "nonexistent", tmp_path)
        assert results == []

    def test_calculates_days_overdue(self, tmp_path: Path) -> None:
        security_dir = tmp_path / "security"
        security_dir.mkdir()
        sec_file = security_dir / "policy.md"
        sec_file.write_text(dedent("""
            ---
            next_review_due: 2026-03-01
            ---
            """).strip())

        results = check_security_docs(security_dir, tmp_path)
        assert results[0].days_overdue is not None
        assert results[0].days_overdue > 0

    def test_handles_no_review_due_date(self, tmp_path: Path) -> None:
        security_dir = tmp_path / "security"
        security_dir.mkdir()
        sec_file = security_dir / "policy.md"
        sec_file.write_text("# Security Policy\n")

        results = check_security_docs(security_dir, tmp_path)
        assert results[0].next_review_due is None
        assert results[0].days_overdue is None

    def test_counts_releases_since_review(self, tmp_path: Path) -> None:
        security_dir = tmp_path / "security"
        security_dir.mkdir()
        sec_file = security_dir / "policy.md"
        sec_file.write_text(dedent("""
            ---
            last_reviewed: 2026-04-01
            ---
            """).strip())

        releases_dir = tmp_path / "releases"
        releases_dir.mkdir()
        (releases_dir / "v1.0.0.md").write_text("date: 2026-04-10\n")

        results = check_security_docs(security_dir, releases_dir)
        assert results[0].releases_since_review == 1

    def test_uses_last_reviewed_as_fallback(self, tmp_path: Path) -> None:
        security_dir = tmp_path / "security"
        security_dir.mkdir()
        sec_file = security_dir / "policy.md"
        sec_file.write_text(dedent("""
            ---
            last_reviewed: 2026-04-01
            ---
            """).strip())

        results = check_security_docs(security_dir, tmp_path)
        assert results[0].last_reviewed == "2026-04-01"


class TestMain:
    def test_main_runs_without_errors(self, tmp_path: Path) -> None:
        security_dir = tmp_path / "security"
        security_dir.mkdir()
        sec_file = security_dir / "policy.md"
        sec_file.write_text(dedent("""
            ---
            last_reviewed_at: 2026-04-01
            next_review_due: 2026-05-01
            ---
            # Security Policy
            """).strip())

        result = main([
            "--security-dir", str(security_dir),
            "--releases-dir", str(tmp_path / "releases"),
            "--max-days-overdue", "90",
        ])

        assert result == 0

    def test_main_with_check_flag(self, tmp_path: Path) -> None:
        security_dir = tmp_path / "security"
        security_dir.mkdir()
        sec_file = security_dir / "policy.md"
        sec_file.write_text(dedent("""
            ---
            next_review_due: 2020-01-01
            ---
            """).strip())

        result = main([
            "--security-dir", str(security_dir),
            "--releases-dir", str(tmp_path / "releases"),
            "--max-days-overdue", "90",
            "--check",
        ])

        # Should fail because docs are critically overdue
        assert result == 1