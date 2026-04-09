from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


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
