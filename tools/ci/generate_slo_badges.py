#!/usr/bin/env python
# SPDX-License-Identifier: MIT
# Copyright 2025 RNA4219

"""Generate SLO badges from governance/policy.yaml."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parents[2]
_POLICY_PATH = _REPO_ROOT / "governance" / "policy.yaml"
_README_PATH = _REPO_ROOT / "README.md"


def _parse_yaml_value(value: str) -> Any:
    """Parse a simple YAML value."""
    value = value.strip()
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1]
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _parse_simple_yaml(content: str) -> dict[str, Any]:
    """Parse a simple YAML file (handles key: value pairs and nested dicts)."""
    result: dict[str, Any] = {}
    current_path: list[str] = []

    for line in content.splitlines():
        # Skip empty lines and comments
        if not line.strip() or line.strip().startswith("#"):
            continue

        # Count indentation
        indent = len(line) - len(line.lstrip())
        indent_level = indent // 2

        # Adjust current path based on indentation
        current_path = current_path[:indent_level]

        # Parse key: value
        if ":" in line:
            key_part = line.split(":", 1)
            key = key_part[0].strip()
            value = key_part[1].strip() if len(key_part) > 1 else ""

            if value:
                # Key-value pair
                _set_nested(result, current_path + [key], _parse_yaml_value(value))
            else:
                # Nested dict start
                current_path.append(key)
                _ensure_nested(result, current_path)

    return result


def _ensure_nested(data: dict, path: list[str]) -> None:
    """Ensure nested dict exists at path."""
    current = data
    for key in path:
        if key not in current:
            current[key] = {}
        current = current[key]


def _set_nested(data: dict, path: list[str], value: Any) -> None:
    """Set a value at a nested path."""
    current = data
    for key in path[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    current[path[-1]] = value


def _load_policy(path: Path) -> dict[str, Any]:
    """Load policy.yaml."""
    if not path.exists():
        raise FileNotFoundError(f"Policy file not found: {path}")
    content = path.read_text(encoding="utf-8")
    return _parse_simple_yaml(content)


def _format_lead_time(hours: float) -> str:
    """Format lead time for display."""
    if hours >= 24:
        days = hours / 24
        return f"{days:.0f}d" if days == int(days) else f"{days:.1f}d"
    return f"{hours:.0f}h"


def _format_mttr(minutes: float) -> str:
    """Format MTTR for display."""
    if minutes >= 60:
        hours = minutes / 60
        return f"{hours:.0f}h" if hours == int(hours) else f"{hours:.1f}h"
    return f"{minutes:.0f}m"


def _format_failure_rate(rate: float) -> str:
    """Format change failure rate as percentage."""
    return f"{rate * 100:.0f}%"


def _get_badge_color(value: float, thresholds: tuple[float, float]) -> str:
    """Get badge color based on value thresholds.

    thresholds: (good_threshold, warning_threshold)
    Returns: brightgreen, yellow, or red
    """
    good, warning = thresholds
    if value <= good:
        return "brightgreen"
    if value <= warning:
        return "yellow"
    return "red"


def _get_failure_rate_color(rate: float) -> str:
    """Get color for change failure rate (lower is better)."""
    if rate <= 0.10:
        return "brightgreen"
    if rate <= 0.20:
        return "yellow"
    return "red"


def generate_badges(policy: dict[str, Any]) -> dict[str, str]:
    """Generate shields.io badge URLs from SLO values."""
    slo = policy.get("slo", {})

    badges = {}

    # Lead Time P95
    lead_time = slo.get("lead_time_p95_hours", 0)
    if lead_time:
        badges["lead_time"] = (
            f"https://img.shields.io/badge/Lead%20Time%20P95-"
            f"{_format_lead_time(lead_time)}-{_get_badge_color(lead_time, (24, 48))}"
        )

    # MTTR P95
    mttr = slo.get("mttr_p95_minutes", 0)
    if mttr:
        badges["mttr"] = (
            f"https://img.shields.io/badge/MTTR%20P95-"
            f"{_format_mttr(mttr)}-{_get_badge_color(mttr, (30, 60))}"
        )

    # Change Failure Rate
    failure_rate = slo.get("change_failure_rate_max", 0)
    if failure_rate:
        badges["change_failure_rate"] = (
            f"https://img.shields.io/badge/Change%20Failure%20Rate-"
            f"{_format_failure_rate(failure_rate)}-{_get_failure_rate_color(failure_rate)}"
        )

    return badges


def generate_badge_markdown(badges: dict[str, str]) -> str:
    """Generate markdown for SLO badges."""
    lines = ["<!-- SLO-BADGES -->"]
    for name, url in badges.items():
        lines.append(f"[![SLO: {name}]({url})]({url})")
    lines.append("<!-- /SLO-BADGES -->")
    return "\n".join(lines)


def update_readme(readme_path: Path, badge_markdown: str) -> bool:
    """Update README with new SLO badges. Returns True if updated."""
    if not readme_path.exists():
        raise FileNotFoundError(f"README not found: {readme_path}")

    content = readme_path.read_text(encoding="utf-8")

    # Check if SLO-BADGES block exists
    pattern = r"<!-- SLO-BADGES -->.*?<!-- /SLO-BADGES -->"
    match = re.search(pattern, content, re.DOTALL)

    if match:
        # Replace existing block
        new_content = re.sub(pattern, badge_markdown, content, flags=re.DOTALL)
    else:
        # Insert after CI badge line
        ci_badge_pattern = r"(\[!\[CI\].*\n)"
        ci_match = re.search(ci_badge_pattern, content)
        if ci_match:
            insert_pos = ci_match.end()
            new_content = content[:insert_pos] + "\n" + badge_markdown + "\n" + content[insert_pos:]
        else:
            # Insert after title
            title_pattern = r"(^# .*\n\n)"
            title_match = re.search(title_pattern, content, re.MULTILINE)
            if title_match:
                insert_pos = title_match.end()
                new_content = content[:insert_pos] + badge_markdown + "\n\n" + content[insert_pos:]
            else:
                new_content = badge_markdown + "\n\n" + content

    if new_content != content:
        readme_path.write_text(new_content, encoding="utf-8")
        return True
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate SLO badges from policy.yaml")
    parser.add_argument(
        "--policy",
        type=Path,
        default=_POLICY_PATH,
        help="Path to policy.yaml",
    )
    parser.add_argument(
        "--readme",
        type=Path,
        default=_README_PATH,
        help="Path to README.md",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check if README is up to date (exit 1 if needs update)",
    )
    parser.add_argument(
        "--print-markdown",
        action="store_true",
        help="Print generated markdown to stdout",
    )

    args = parser.parse_args(argv)

    try:
        policy = _load_policy(args.policy)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    badges = generate_badges(policy)
    badge_markdown = generate_badge_markdown(badges)

    if args.print_markdown:
        print(badge_markdown)
        return 0

    if args.check:
        current = args.readme.read_text(encoding="utf-8")
        if "<!-- SLO-BADGES -->" not in current:
            print("SLO badges not found in README", file=sys.stderr)
            return 1
        # Compare without whitespace differences
        current_badges = re.search(
            r"<!-- SLO-BADGES -->.*?<!-- /SLO-BADGES -->",
            current,
            re.DOTALL,
        )
        if current_badges and current_badges.group(0) == badge_markdown:
            print("SLO badges are up to date")
            return 0
        print("SLO badges need update", file=sys.stderr)
        return 1

    updated = update_readme(args.readme, badge_markdown)
    if updated:
        print(f"Updated {args.readme}")
    else:
        print("No changes needed")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())