# SPDX-License-Identifier: MIT
"""fixture接続検証。製品の実テスト受入は別のrunner実行で記録する。"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from defusedxml.common import DTDForbidden

from tools.ci.hate_qeg import command, expected_gate, freeze_tool, frozen_snapshot, main, run, snapshot
from tools.ci.hate_qeg_projection import file_hash, project, read_json, reconcile, write_json


def junit(path: Path, status: str = "passed", *, duplicate: bool = False) -> None:
    element = {
        "passed": "",
        "failed": '<failure message="assertion failed"/>',
        "error": '<error message="setup error"/>',
        "skipped": '<skipped message="dependency unavailable"/>',
    }[status]
    case = f'<testcase classname="tests.example" name="test_result" time="0.1">{element}</testcase>'
    path.write_text(
        f'<testsuites><testsuite name="pytest" tests="1">{case * (2 if duplicate else 1)}</testsuite></testsuites>',
        encoding="utf-8",
    )


def record(status: str = "passed") -> dict:
    return {
        "run_id": "fixture-run",
        "run_attempt": 1,
        "commit_sha": "a" * 40,
        "source_version": "0.3.0",
        "payload": {
            "canonical_test_id": "junit:tests/example.py::test_result",
            "status": status,
            "identity_components": {"suite": "pytest", "classname": "tests.example", "name": "test_result"},
        },
    }


@pytest.mark.parametrize("status", ["passed", "failed", "error", "skipped"])
def test_reconcile_preserves_all_statuses(tmp_path: Path, status: str) -> None:
    path = tmp_path / "junit.xml"
    junit(path, status)
    assert reconcile(path, [record(status)], "fixture-run", "a" * 40)[0]["payload"]["status"] == status


def test_reconcile_rejects_document_type_declarations(tmp_path: Path) -> None:
    path = tmp_path / "junit.xml"
    path.write_text('<!DOCTYPE testsuites><testsuites/>', encoding="utf-8")
    with pytest.raises(DTDForbidden):
        reconcile(path, [], "fixture-run", "a" * 40)


@pytest.mark.parametrize(
    "change", ["missing", "status", "name", "run", "revision", "attempt", "duplicate", "empty", "duplicate-xml"]
)
def test_reconcile_rejects_incomplete_or_mismatched_evidence(tmp_path: Path, change: str) -> None:
    path = tmp_path / "junit.xml"
    junit(path, duplicate=change == "duplicate-xml")
    records = [record()]
    if change == "missing":
        records = []
    elif change == "status":
        records[0]["payload"]["status"] = "failed"
    elif change == "name":
        records[0]["payload"]["identity_components"]["name"] = "different"
    elif change in {"run", "revision", "attempt"}:
        records[0][{"run": "run_id", "revision": "commit_sha", "attempt": "run_attempt"}[change]] = "different"
    elif change == "duplicate":
        records *= 2
    elif change == "empty":
        path.write_text("<testsuites/>", encoding="utf-8")
    with pytest.raises(ValueError):
        reconcile(path, records, "fixture-run", "a" * 40)


@pytest.fixture
def projection_inputs(tmp_path: Path) -> tuple[Path, dict]:
    raw = tmp_path / "raw"
    raw.mkdir()
    junit(raw / "junit.xml")
    (raw / "cobertura.xml").write_text(
        '<coverage line-rate="1"><packages><package name="sample"><classes><class name="sample" filename="sample.py"><lines><line number="1" hits="1"/></lines></class></classes></package></packages></coverage>',
        encoding="utf-8",
    )
    write_json(raw / "coverage-summary.json", {"totals": {"percent_covered": 100}})
    receipt = {
        "run_id": "fixture-run",
        "target": {
            "projectId": "workflow-cookbook",
            "buildId": "synthetic-build",
            "revision": "a" * 40,
            "environmentId": "synthetic-local",
        },
        "started_at": "2026-09-11T00:00:00Z",
        "finished_at": "2026-09-11T00:01:00Z",
        "evaluated_at": "2026-09-11T00:02:00Z",
        "pytest_exit_code": 0,
        "coverage_min": 80,
    }
    write_json(tmp_path / "receipt.json", receipt)
    write_json(tmp_path / "source-snapshot.json", {"synthetic": True})
    p0a = tmp_path / "hate/p0a"
    p0a.mkdir(parents=True)
    (p0a / "HATE-test-results.ndjson").write_text(json.dumps(record()) + "\n", encoding="utf-8")
    write_json(
        p0a / "HATE-run.json",
        {"run_id": "fixture-run", "commit_sha": "a" * 40, "payload": {"finished_at": receipt["finished_at"]}},
    )
    write_json(p0a / "precheck-decision.json", {"payload": {"decision": "eligible"}})
    write_json(tmp_path / "hate/export/qeg-bundle.json", {"synthetic": True})
    write_json(
        tmp_path / "hate/export/qeg-export-report.json",
        {
            "run_id": "fixture-run",
            "commit_sha": "a" * 40,
            "export_status": "success",
            "qeg_schema_compatibility": {"valid": True},
        },
    )
    return tmp_path, receipt


def test_projection_preserves_provenance_and_explicit_execution_policy(projection_inputs: tuple[Path, dict]) -> None:
    output, receipt = projection_inputs
    summary = project(output, receipt, scope="fixture")
    assert summary["test_cases"] == 1
    assert summary["qeg_executions"] == 2
    data = read_json(output / "gate-input.json")
    assert data["policy"]["inputContract"]["requireExecutedTests"] is True
    for artifact in data["metadata"]["inputArtifacts"]:
        assert file_hash(output / artifact["path"]) == artifact["contentHash"]
    assert data["policy"]["executionPolicy"]["target"] == receipt["target"]
    assert data["waivers"] == []


@pytest.mark.parametrize("failure", ["pytest", "coverage", "skip", "failed", "error"])
def test_projection_cannot_turn_failures_into_pass(projection_inputs: tuple[Path, dict], failure: str) -> None:
    output, receipt = projection_inputs
    if failure == "pytest":
        receipt["pytest_exit_code"] = 2
    elif failure == "coverage":
        write_json(output / "raw/coverage-summary.json", {"totals": {"percent_covered": 79.99}})
    else:
        status = "skipped" if failure == "skip" else failure
        junit(output / "raw/junit.xml", status)
        (output / "hate/p0a/HATE-test-results.ndjson").write_text(json.dumps(record(status)) + "\n", encoding="utf-8")
    project(output, receipt, scope="fixture")
    executions = [n["execution"] for n in read_json(output / "gate-input.json")["graph"]["nodes"] if "execution" in n]
    assert any(e["status"] != "pass" for e in executions)


@pytest.mark.parametrize(
    "path,field,value",
    [
        ("hate/p0a/precheck-decision.json", "payload", {"decision": "hard_dq"}),
        ("hate/export/qeg-export-report.json", "export_status", "partial"),
        ("hate/export/qeg-export-report.json", "run_id", "other"),
        ("hate/p0a/HATE-run.json", "commit_sha", "b" * 40),
    ],
)
def test_projection_rejects_bad_hate_export(
    projection_inputs: tuple[Path, dict], path: str, field: str, value: object
) -> None:
    output, receipt = projection_inputs
    data = read_json(output / path)
    data[field] = value
    write_json(output / path, data)
    with pytest.raises(ValueError):
        project(output, receipt, scope="fixture")


def test_command_preserves_nonzero_exit_and_logs(tmp_path: Path) -> None:
    code = command(
        [sys.executable, "-c", "print('observed'); raise SystemExit(3)"], tmp_path, tmp_path, "child", dict(os.environ)
    )
    assert code == 3
    assert read_json(tmp_path / "child.command.json")["exit_code"] == 3
    assert "observed" in (tmp_path / "child.stdout.log").read_text()


def test_run_refuses_existing_output_and_source_tree(tmp_path: Path) -> None:
    args = argparse.Namespace(
        repo=tmp_path / "repo", hate_root=tmp_path / "hate", qeg_root=tmp_path / "qeg", output=tmp_path / "repo/out"
    )
    with pytest.raises(ValueError, match="outside"):
        run(args)
    args.output = tmp_path
    with pytest.raises(FileExistsError):
        run(args)


def test_snapshot_detects_uncommitted_content(tmp_path: Path) -> None:
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    (tmp_path / "input.txt").write_text("before", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "input.txt"], check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.test",
            "commit",
            "-m",
            "fixture",
        ],
        check=True,
        capture_output=True,
    )
    before = snapshot(tmp_path)
    (tmp_path / "input.txt").write_text("after", encoding="utf-8")
    after = snapshot(tmp_path)
    assert before["revision"] == after["revision"]
    assert before["content_hash"] != after["content_hash"]
    assert after["dirty"] is True


def test_cli_help() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0


def test_frozen_tool_is_independent_of_later_checkout_edits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    origin = tmp_path / "origin"
    (origin / "src").mkdir(parents=True)
    (origin / "src/cli.py").write_text("original", encoding="utf-8")
    monkeypatch.setattr("tools.ci.hate_qeg.git", lambda root, *args: "a" * 40 if args[0] == "rev-parse" else "")
    locked = freeze_tool(origin, tmp_path / "frozen", ("src",))
    (origin / "src/cli.py").write_text("parallel edit", encoding="utf-8")
    assert frozen_snapshot(tmp_path / "frozen")["content_hash"] == locked["content_hash"]
    assert (tmp_path / "frozen/src/cli.py").read_text() == "original"
    (tmp_path / "frozen/src/cli.py").write_text("tamper", encoding="utf-8")
    assert frozen_snapshot(tmp_path / "frozen")["content_hash"] != locked["content_hash"]


def test_freeze_rejects_source_change_during_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import shutil

    origin = tmp_path / "origin"
    origin.mkdir()
    (origin / "cli.py").write_text("before", encoding="utf-8")
    monkeypatch.setattr("tools.ci.hate_qeg.git", lambda root, *args: "a" * 40 if args[0] == "rev-parse" else "")
    copy_file = shutil.copyfile

    def concurrent_copy(source: Path, target: Path) -> None:
        copy_file(source, target)
        source.write_text("changed during copy", encoding="utf-8")

    monkeypatch.setattr("tools.ci.hate_qeg.shutil.copyfile", concurrent_copy)
    with pytest.raises(ValueError, match="changed while copying"):
        freeze_tool(origin, tmp_path / "frozen", ("cli.py",))


def test_real_hate_qeg_clis_with_synthetic_controls(projection_inputs: tuple[Path, dict]) -> None:
    """実CLI接続のfixture試験。対象repoの実pytest証跡とは別物。"""
    workspace = Path(__file__).resolve().parents[3]
    hate = Path(os.environ.get("WFC_HATE_ROOT", str(workspace / "harness-auto-test-evidence")))
    qeg = Path(os.environ.get("WFC_QEG_ROOT", str(workspace / "quality-evidence-graph")))
    python = Path(os.environ.get("WFC_HATE_PYTHON", str(hate / ".venv/Scripts/python.exe")))
    node = Path(
        os.environ.get("WFC_NODE", str(Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "nodejs/node.exe"))
    )
    if not all(p.exists() for p in (python, node, qeg / "dist/cli.js")):
        pytest.skip("optional local HATE/QEG checkouts and built runtimes are unavailable")
    output, receipt = projection_inputs
    write_json(output / "expected-gate-verdict.json", expected_gate())
    context = {
        "provider": "generic-ci",
        "repository": "workflow-cookbook-fixture",
        "workflow": "synthetic-cli-contract",
        "job": "fixture",
        "run_id": receipt["run_id"],
        "run_attempt": 1,
        "commit_sha": receipt["target"]["revision"],
        "started_at": receipt["started_at"],
        "finished_at": receipt["finished_at"],
    }
    write_json(output / "raw/ci-context.json", context)
    env = {**os.environ, "PYTHONPATH": str(hate / "src"), "PYTHONUTF8": "1"}
    for name in list(env):
        if name.startswith(("COV_CORE_", "COVERAGE_PROCESS_START")):
            env.pop(name)
    for arguments in (
        ["p0a", "--input", str(output / "raw"), "--out", str(output / "hate/p0a")],
        ["export", "qeg", "--fixture", str(output / "hate"), "--out", str(output / "hate/export")],
    ):
        write_json(
            output / "hate/diff-risk-test.json",
            {
                "schema_version": "HATE/v1",
                "source_tool": "workflow-cookbook-fixture",
                "commit_sha": receipt["target"]["revision"],
                "changed_entities": [],
                "risks": [],
                "test_obligations": [],
            },
        )
        result = subprocess.run(
            [str(python), "-B", "-m", "hate", *arguments], cwd=hate, env=env, capture_output=True, text=True, timeout=60
        )
        assert result.returncode == 0, result.stdout + result.stderr
    project(output, receipt, scope="fixture")
    original = read_json(output / "gate-input.json")

    def gate() -> dict:
        result = subprocess.run(
            [str(node), str(qeg / "dist/cli.js"), "gate", str(output)],
            cwd=qeg,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode in {0, 2}, result.stdout + result.stderr
        return json.loads(result.stdout)

    positive = gate()
    assert positive["verdict"] == "go", positive
    validated = subprocess.run(
        [str(node), str(qeg / "dist/cli.js"), "validate", str(output)],
        cwd=qeg,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert validated.returncode == 0, validated.stdout + validated.stderr
    execution = next(n for n in original["graph"]["nodes"] if "execution" in n)
    raw_path = output / execution["execution"]["rawArtifactRef"]["path"]
    original_raw = raw_path.read_bytes()
    raw_path.write_text("{}", encoding="utf-8")
    assert gate()["verdict"] == "disqualified"
    raw_path.write_bytes(original_raw)
    for status, verdict in (("fail", "no_go"), ("skipped", "disqualified")):
        changed = copy.deepcopy(original)
        node_data = next(n for n in changed["graph"]["nodes"] if "execution" in n)
        raw = read_json(raw_path)
        raw["status"] = status
        write_json(raw_path, raw)
        node_data["execution"]["status"] = status
        node_data.pop("passed", None)
        node_data["execution"]["rawArtifactRef"]["contentHash"] = file_hash(raw_path)
        node_data["evidenceRefs"][0]["contentHash"] = file_hash(raw_path)
        write_json(output / "gate-input.json", changed)
        assert gate()["verdict"] == verdict
    raw_path.write_bytes(original_raw)
    write_json(output / "gate-input.json", original)
    raw_path.unlink()
    assert gate()["verdict"] == "disqualified"
