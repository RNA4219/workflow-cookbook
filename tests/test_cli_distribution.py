from __future__ import annotations

import json
from pathlib import Path

from tools.ci import five_tool_manifest
from tools.ci.governance_gate.cli import parse_arguments
from tools.codemap.update import constants
from tools.codemap.update.cli import parse_args
from tools.perf.collect_metrics import cli as metrics_cli


def test_codemap_repo_root_defaults_to_selected_repository(tmp_path: Path) -> None:
    previous = constants._REPO_ROOT.value
    try:
        options = parse_args(
            [
                "--repo-root",
                str(tmp_path),
                "--targets",
                "docs/birdseye/index.json",
                "--dry-run",
            ]
        )
        assert constants._REPO_ROOT.get() == tmp_path.resolve()
        assert options.dry_run
    finally:
        constants._REPO_ROOT.value = previous


def test_governance_repo_root_is_explicit(tmp_path: Path) -> None:
    args = parse_arguments(["--repo-root", str(tmp_path), "--pr-body", "Intent: INT-001"])
    assert args.repo_root == tmp_path.resolve()


def test_metrics_config_precedence_uses_cli_then_cwd(
    tmp_path: Path, monkeypatch
) -> None:
    cwd_config = tmp_path / "governance" / "metrics.yaml"
    cwd_config.parent.mkdir()
    cwd_config.write_text(
        "checklist_compliance_rate: Checklist compliance (%)\n",
        encoding="utf-8",
    )
    explicit = tmp_path / "explicit.yaml"
    explicit.write_text(
        "review_latency: Review latency\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GOVERNANCE_METRICS_PATH", raising=False)
    metrics_cli.load_metric_config.cache_clear()

    assert metrics_cli.load_metric_config().keys == ("checklist_compliance_rate",)
    assert metrics_cli.load_metric_config(explicit).keys == ("review_latency",)
    metrics_cli.load_metric_config.cache_clear()


def test_five_tool_manifest_resolves_artifacts_from_repo_root(tmp_path: Path) -> None:
    artifact = tmp_path / "evidence.json"
    artifact.write_text(json.dumps({"schema_version": "1.0"}), encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "qeg_policy_hash": "sha256:test",
                "artifacts": [
                    {
                        "tool": "rand",
                        "role": "requirements",
                        "path": "evidence.json",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    manifest = five_tool_manifest.generate_manifest(config, tmp_path)

    assert manifest["artifacts"][0]["exists"] is True
    assert Path(manifest["artifacts"][0]["path"]) == artifact
