from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Import the module for direct testing
import tools.ci.check_birdseye_freshness as bf_module


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "tools.ci.check_birdseye_freshness", *args],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )


def _write_birdseye_fixture(tmp_path: Path) -> tuple[Path, Path]:
    caps_dir = tmp_path / "docs/birdseye/caps"
    caps_dir.mkdir(parents=True)
    (caps_dir / "README.md.json").write_text("{}", encoding="utf-8")

    index_path = tmp_path / "docs/birdseye/index.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(
        json.dumps(
            {
                "generated_at": "00042",
                "nodes": {
                    "README.md": {
                        "mtime": "00001",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    hot_path = tmp_path / "docs/birdseye/hot.json"
    hot_path.write_text(
        json.dumps(
            {
                "generated_at": "00042",
                "nodes": [
                    {
                        "id": "README.md",
                        "caps": "docs/birdseye/caps/README.md.json",
                        "last_verified_at": "2026-04-10T00:00:00Z",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return index_path, hot_path


# Direct function tests for coverage
class TestLoadJson:
    def test_loads_valid_json(self, tmp_path: Path) -> None:
        json_path = tmp_path / "test.json"
        json_path.write_text('{"key": "value"}', encoding="utf-8")
        result = bf_module._load_json(json_path)
        assert result["key"] == "value"

    def test_raises_for_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(bf_module.BirdseyeFreshnessError):
            bf_module._load_json(tmp_path / "missing.json")

    def test_raises_for_invalid_json(self, tmp_path: Path) -> None:
        json_path = tmp_path / "invalid.json"
        json_path.write_text("not json", encoding="utf-8")
        with pytest.raises(bf_module.BirdseyeFreshnessError):
            bf_module._load_json(json_path)

    def test_raises_for_non_object(self, tmp_path: Path) -> None:
        json_path = tmp_path / "array.json"
        json_path.write_text("[1, 2, 3]", encoding="utf-8")
        with pytest.raises(bf_module.BirdseyeFreshnessError):
            bf_module._load_json(json_path)


class TestParseTimestamp:
    def test_parses_iso_format(self) -> None:
        result = bf_module._parse_timestamp("2026-04-10T12:00:00Z")
        assert result.year == 2026
        assert result.month == 4
        assert result.day == 10

    def test_parses_with_timezone(self) -> None:
        result = bf_module._parse_timestamp("2026-04-10T12:00:00+09:00")
        assert result.year == 2026

    def test_raises_for_invalid_format(self) -> None:
        with pytest.raises(bf_module.BirdseyeFreshnessError):
            bf_module._parse_timestamp("invalid")


class TestEvaluateBirdseyeFreshness:
    def test_validates_matching_generated_at(self, tmp_path: Path) -> None:
        caps_dir = tmp_path / "docs/birdseye/caps"
        caps_dir.mkdir(parents=True)
        (caps_dir / "README.md.json").write_text("{}", encoding="utf-8")

        index_doc = {
            "generated_at": "00042",
            "nodes": {"README.md": {"mtime": "00001"}},
        }
        hot_doc = {
            "generated_at": "00042",
            "nodes": [
                {
                    "id": "README.md",
                    "caps": "docs/birdseye/caps/README.md.json",
                    "last_verified_at": "2026-04-10T00:00:00Z",
                }
            ],
        }

        report = bf_module.evaluate_birdseye_freshness(
            index_doc=index_doc,
            hot_doc=hot_doc,
            repo_root=tmp_path,
            now=datetime.now(UTC),
        )
        assert report.failures == []

    def test_fails_for_mismatched_generated_at(self, tmp_path: Path) -> None:
        index_doc = {"generated_at": "00042", "nodes": {}}
        hot_doc = {"generated_at": "00043", "nodes": []}

        report = bf_module.evaluate_birdseye_freshness(
            index_doc=index_doc,
            hot_doc=hot_doc,
            repo_root=tmp_path,
            now=datetime.now(UTC),
        )
        assert "index.json and hot.json must share the same generated_at" in " ".join(report.failures)

    def test_fails_for_invalid_serial_format(self, tmp_path: Path) -> None:
        index_doc = {"generated_at": "invalid", "nodes": {}}
        hot_doc = {"generated_at": "invalid", "nodes": []}

        report = bf_module.evaluate_birdseye_freshness(
            index_doc=index_doc,
            hot_doc=hot_doc,
            repo_root=tmp_path,
            now=datetime.now(UTC),
        )
        assert any("5-digit serial" in f for f in report.failures)

    def test_fails_for_missing_caps_file(self, tmp_path: Path) -> None:
        index_doc = {"generated_at": "00042", "nodes": {}}
        hot_doc = {
            "generated_at": "00042",
            "nodes": [{"id": "test", "caps": "missing.json"}],
        }

        report = bf_module.evaluate_birdseye_freshness(
            index_doc=index_doc,
            hot_doc=hot_doc,
            repo_root=tmp_path,
            now=datetime.now(UTC),
        )
        assert any("missing caps" in f for f in report.failures)

    def test_warns_for_stale_verified_at(self, tmp_path: Path) -> None:
        index_doc = {"generated_at": "00042", "nodes": {}}
        hot_doc = {
            "generated_at": "00042",
            "nodes": [
                {
                    "id": "test",
                    "caps": "",
                    "last_verified_at": "2020-01-01T00:00:00Z",
                }
            ],
        }

        report = bf_module.evaluate_birdseye_freshness(
            index_doc=index_doc,
            hot_doc=hot_doc,
            repo_root=tmp_path,
            now=datetime.now(UTC),
            max_verified_age_days=90,
        )
        assert len(report.warnings) >= 1

    def test_fails_for_non_mapping_nodes(self, tmp_path: Path) -> None:
        index_doc = {"generated_at": "00042", "nodes": "invalid"}
        hot_doc = {"generated_at": "00042", "nodes": []}

        report = bf_module.evaluate_birdseye_freshness(
            index_doc=index_doc,
            hot_doc=hot_doc,
            repo_root=tmp_path,
            now=datetime.now(UTC),
        )
        assert any("nodes must be a mapping" in f for f in report.failures)


# CLI tests (kept for integration testing)
def test_birdseye_freshness_passes_for_valid_fixture(tmp_path: Path) -> None:
    index_path, hot_path = _write_birdseye_fixture(tmp_path)

    result = _run_cli(
        "--check",
        "--repo-root",
        str(tmp_path),
        "--index-path",
        str(index_path),
        "--hot-path",
        str(hot_path),
        "--max-verified-age-days",
        "365",
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["failures"] == []
    assert payload["warnings"] == []


def test_birdseye_freshness_fails_when_generated_at_mismatch(tmp_path: Path) -> None:
    index_path, hot_path = _write_birdseye_fixture(tmp_path)
    hot_doc = json.loads(hot_path.read_text(encoding="utf-8"))
    hot_doc["generated_at"] = "00043"
    hot_path.write_text(json.dumps(hot_doc), encoding="utf-8")

    result = _run_cli(
        "--check",
        "--repo-root",
        str(tmp_path),
        "--index-path",
        str(index_path),
        "--hot-path",
        str(hot_path),
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["failures"] == [
        "index.json and hot.json must share the same generated_at update cycle"
    ]


def test_birdseye_freshness_fails_when_capsule_is_missing(tmp_path: Path) -> None:
    index_path, hot_path = _write_birdseye_fixture(tmp_path)
    hot_doc = json.loads(hot_path.read_text(encoding="utf-8"))
    hot_doc["nodes"][0]["caps"] = "docs/birdseye/caps/missing.json"
    hot_path.write_text(json.dumps(hot_doc), encoding="utf-8")

    result = _run_cli(
        "--check",
        "--repo-root",
        str(tmp_path),
        "--index-path",
        str(index_path),
        "--hot-path",
        str(hot_path),
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["failures"] == [
        "hot.json: node README.md references missing caps file docs/birdseye/caps/missing.json"
    ]


def test_birdseye_freshness_warns_when_last_verified_is_stale(tmp_path: Path) -> None:
    index_path, hot_path = _write_birdseye_fixture(tmp_path)
    hot_doc = json.loads(hot_path.read_text(encoding="utf-8"))
    hot_doc["nodes"][0]["last_verified_at"] = "2025-01-01T00:00:00Z"
    hot_path.write_text(json.dumps(hot_doc), encoding="utf-8")

    result = _run_cli(
        "--check",
        "--repo-root",
        str(tmp_path),
        "--index-path",
        str(index_path),
        "--hot-path",
        str(hot_path),
        "--max-verified-age-days",
        "90",
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["failures"] == []
    assert payload["warnings"] == [
        "hot.json: node README.md last verified at 2025-01-01T00:00:00Z exceeds 90 day freshness window"
    ]
