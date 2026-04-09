"""Tests for tools.ci.generate_slo_badges."""

from __future__ import annotations

import sys
from pathlib import Path
from textwrap import dedent
from types import SimpleNamespace

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.ci.generate_slo_badges import (
    generate_badge_markdown,
    generate_badges,
    update_readme,
    _format_failure_rate,
    _format_lead_time,
    _format_mttr,
    _get_badge_color,
    _get_failure_rate_color,
)


class TestFormatters:
    def test_format_lead_time_hours(self) -> None:
        assert _format_lead_time(4) == "4h"
        assert _format_lead_time(12) == "12h"
        assert _format_lead_time(23) == "23h"

    def test_format_lead_time_days(self) -> None:
        assert _format_lead_time(24) == "1d"
        assert _format_lead_time(48) == "2d"
        assert _format_lead_time(36) == "1.5d"

    def test_format_mttr_minutes(self) -> None:
        assert _format_mttr(15) == "15m"
        assert _format_mttr(30) == "30m"
        assert _format_mttr(45) == "45m"

    def test_format_mttr_hours(self) -> None:
        assert _format_mttr(60) == "1h"
        assert _format_mttr(120) == "2h"
        assert _format_mttr(90) == "1.5h"

    def test_format_failure_rate(self) -> None:
        assert _format_failure_rate(0.10) == "10%"
        assert _format_failure_rate(0.20) == "20%"
        assert _format_failure_rate(0.05) == "5%"

    def test_get_badge_color(self) -> None:
        assert _get_badge_color(10, (24, 48)) == "brightgreen"
        assert _get_badge_color(30, (24, 48)) == "yellow"
        assert _get_badge_color(60, (24, 48)) == "red"

    def test_get_failure_rate_color(self) -> None:
        assert _get_failure_rate_color(0.05) == "brightgreen"
        assert _get_failure_rate_color(0.15) == "yellow"
        assert _get_failure_rate_color(0.30) == "red"


class TestGenerateBadges:
    def test_generates_all_badges(self) -> None:
        policy = {
            "slo": {
                "lead_time_p95_hours": 24,
                "mttr_p95_minutes": 30,
                "change_failure_rate_max": 0.20,
            }
        }
        badges = generate_badges(policy)
        assert "lead_time" in badges
        assert "mttr" in badges
        assert "change_failure_rate" in badges
        assert "brightgreen" in badges["lead_time"]

    def test_handles_missing_slo(self) -> None:
        policy: dict = {}
        badges = generate_badges(policy)
        assert badges == {}

    def test_handles_partial_slo(self) -> None:
        policy = {"slo": {"lead_time_p95_hours": 24}}
        badges = generate_badges(policy)
        assert len(badges) == 1
        assert "lead_time" in badges


class TestGenerateBadgeMarkdown:
    def test_generates_markdown_block(self) -> None:
        badges = {
            "lead_time": "https://img.shields.io/badge/Lead%20Time%20P95-1d-brightgreen",
            "mttr": "https://img.shields.io/badge/MTTR%20P95-30m-brightgreen",
        }
        md = generate_badge_markdown(badges)
        assert "<!-- SLO-BADGES -->" in md
        assert "<!-- /SLO-BADGES -->" in md
        assert "Lead%20Time%20P95" in md


class TestUpdateReadme:
    def test_updates_existing_block(self, tmp_path: Path) -> None:
        readme = tmp_path / "README.md"
        readme.write_text(
            dedent(
                """
                # Test

                [![CI](badge.svg)]

                <!-- SLO-BADGES -->
                [old badge]
                <!-- /SLO-BADGES -->

                Some content.
                """
            ).lstrip()
        )

        badge_md = "<!-- SLO-BADGES -->\n[new badge]\n<!-- /SLO-BADGES -->"
        updated = update_readme(readme, badge_md)

        assert updated is True
        content = readme.read_text()
        assert "[new badge]" in content
        assert "[old badge]" not in content

    def test_inserts_after_ci_badge(self, tmp_path: Path) -> None:
        readme = tmp_path / "README.md"
        readme.write_text(
            dedent(
                """
                # Test

                [![CI](badge.svg)]

                Some content.
                """
            ).lstrip()
        )

        badge_md = "<!-- SLO-BADGES -->\n[new badge]\n<!-- /SLO-BADGES -->"
        updated = update_readme(readme, badge_md)

        assert updated is True
        content = readme.read_text()
        assert "[new badge]" in content

    def test_no_change_when_identical(self, tmp_path: Path) -> None:
        badge_md = "<!-- SLO-BADGES -->\n[badge]\n<!-- /SLO-BADGES -->"
        readme = tmp_path / "README.md"
        readme.write_text(f"# Test\n\n{badge_md}\n")

        updated = update_readme(readme, badge_md)
        assert updated is False