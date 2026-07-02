# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from tools.ci.five_tool_manifest import generate_manifest, main, validate_manifest


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def init_repo(path: Path) -> None:
    path.mkdir(parents=True)
    subprocess.run(["git", "-C", str(path), "init"], check=True, capture_output=True, text=True)
    (path / "README.md").write_text(f"# {path.name}\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "README.md"], check=True, capture_output=True, text=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(path),
            "-c",
            "user.email=test@example.test",
            "-c",
            "user.name=Test User",
            "commit",
            "-m",
            "init",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def base_config(tmp_path: Path) -> Path:
    for tool in ("rand", "code-to-gate", "hate", "manual-bb", "qeg"):
        init_repo(tmp_path / tool)

    write_json(
        tmp_path / "rand-artifact.json",
        {
            "schema_version": "1.0",
            "requirement_id": "rand:req-1",
            "evidence_id": "rand:ev-1",
            "gate_verdict": "go",
        },
    )
    write_json(
        tmp_path / "qeg-output.json",
        {
            "qegVersion": "1.0",
            "nodeId": "qeg:node-1",
            "policyHash": "sha256:test-policy",
            "verdict": "go",
        },
    )
    config_path = tmp_path / "five-tool-config.json"
    write_json(
        config_path,
        {
            "manifest_id": "five-tool:test",
            "run_id": "run:test",
            "qeg_policy_hash": "sha256:test-policy",
            "repos": [
                {"tool": "rand", "path": str(tmp_path / "rand")},
                {"tool": "code-to-gate", "path": str(tmp_path / "code-to-gate")},
                {"tool": "hate", "path": str(tmp_path / "hate")},
                {"tool": "manual-bb", "path": str(tmp_path / "manual-bb")},
                {"tool": "qeg", "path": str(tmp_path / "qeg")},
            ],
            "artifacts": [
                {"tool": "rand", "role": "requirements_audit_packet", "path": str(tmp_path / "rand-artifact.json")},
                {"tool": "qeg", "role": "output_record", "path": str(tmp_path / "qeg-output.json")},
            ],
        },
    )
    return config_path


def test_generate_manifest_validates_cross_repo_contract(tmp_path: Path) -> None:
    config_path = base_config(tmp_path)

    manifest = generate_manifest(config_path)
    result = validate_manifest(manifest)

    assert result.status == "ok"
    assert manifest["chain"] == ["rand", "code-to-gate", "hate", "manual-bb", "qeg"]
    assert manifest["qeg_policy_hash"] == "sha256:test-policy"
    assert {repo["tool"] for repo in manifest["repos"]} == {"rand", "code-to-gate", "hate", "manual-bb", "qeg"}


def test_validate_manifest_rejects_non_qeg_direct_gate_policy(tmp_path: Path) -> None:
    config_path = base_config(tmp_path)
    write_json(
        tmp_path / "rand-artifact.json",
        {
            "schema_version": "1.0",
            "requirement_id": "rand:req-1",
            "gate_policy": {"mode": "enforce"},
        },
    )

    result = validate_manifest(generate_manifest(config_path))

    assert result.status == "failed"
    assert result.errors == ["non-QEG artifact carries direct gate_policy: rand:requirements_audit_packet"]


def test_validate_manifest_requires_qeg_policy_hash_in_qeg_artifacts(tmp_path: Path) -> None:
    config_path = base_config(tmp_path)
    write_json(
        tmp_path / "qeg-output.json",
        {
            "qegVersion": "1.0",
            "nodeId": "qeg:node-1",
            "policyHash": "sha256:other-policy",
            "verdict": "go",
        },
    )

    result = validate_manifest(generate_manifest(config_path))

    assert result.status == "failed"
    assert result.errors == ["qeg_policy_hash is not present in QEG artifacts"]


def test_cli_generate_validate_writes_manifest(tmp_path: Path) -> None:
    config_path = base_config(tmp_path)
    manifest_path = tmp_path / "manifest.json"

    exit_code = main(["generate", "--config", str(config_path), "--out", str(manifest_path), "--validate"])

    assert exit_code == 0
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["manifest_id"] == "five-tool:test"
