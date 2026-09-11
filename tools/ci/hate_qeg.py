# SPDX-License-Identifier: MIT
"""全pytestを実行し、HATEとQEGの公開CLIへ証跡を渡すローカル受入runner。"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tomllib
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tools.ci.hate_qeg_projection import file_hash, project, read_json, value_hash, write_json


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={repo.as_posix()}", "-C", str(repo), *args], text=True, encoding="utf-8"
    )  # nosec B607 -- fixed git command


def snapshot(repo: Path, *, runtime: tuple[str, ...] = ()) -> dict[str, Any]:
    names = set(git(repo, "ls-files", "--cached", "--others", "--exclude-standard", "-z").split("\0")) - {""}
    for folder in runtime:
        names.update(
            p.relative_to(repo).as_posix()
            for p in (repo / folder).rglob("*")
            if p.is_file() and "__pycache__" not in p.parts
        )
    files = {name: file_hash(repo / name) if (repo / name).is_file() else None for name in sorted(names)}
    return {
        "root": str(repo),
        "revision": git(repo, "rev-parse", "HEAD").strip(),
        "dirty": bool(git(repo, "status", "--porcelain")),
        "files": files,
        "content_hash": value_hash(files),
    }


def frozen_snapshot(root: Path) -> dict[str, Any]:
    files = {
        p.relative_to(root).as_posix(): file_hash(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts
    }
    return {"root": str(root), "files": files, "content_hash": value_hash(files)}


def freeze_tool(origin: Path, destination: Path, entries: tuple[str, ...]) -> dict[str, Any]:
    """並行作業中のcheckoutから実行中に別versionを読むことを防ぐ。"""
    destination.mkdir(parents=True, exist_ok=False)
    provenance = {
        "origin": str(origin),
        "revision": git(origin, "rev-parse", "HEAD").strip(),
        "dirty": bool(git(origin, "status", "--porcelain")),
    }
    files: dict[str, str] = {}
    for entry in entries:
        source = origin / entry
        if not source.exists():
            raise ValueError(f"missing tool runtime input: {source}")
        for path in sorted(source.rglob("*")) if source.is_dir() else [source]:
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            if not path.resolve().is_relative_to(origin):
                raise ValueError(f"tool input escapes source root: {path}")
            name = path.relative_to(origin).as_posix()
            files[name] = file_hash(path)
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    copied = frozen_snapshot(destination)
    if copied["files"] != files or any(file_hash(origin / name) != digest for name, digest in files.items()):
        raise ValueError("tool source changed while copying; use a new output directory and retry")
    return {**provenance, **copied}


def expected_gate() -> dict[str, Any]:
    return {
        "fixture": "local-automated-test-acceptance",
        "description": "Predeclared oracle: complete successful tests and coverage threshold yield Go; this is a real test run, not fixture execution.",
        "expectedVerdict": "go",
        "expectedDisqualifications": [],
        "expectedBlockers": [],
        "expectedResidualRisks": [],
        "expectedHumanReview": [],
        "expectedExitCode": 0,
        "contractRef": "docs/contracts/hate-qeg-test-gate.md",
    }


def command(args: list[str], cwd: Path, output: Path, name: str, env: dict[str, str], timeout: int = 900) -> int:
    started = now()
    with (
        (output / f"{name}.stdout.log").open("w", encoding="utf-8") as stdout,
        (output / f"{name}.stderr.log").open("w", encoding="utf-8") as stderr,
    ):
        try:
            code = subprocess.run(
                args, cwd=cwd, env=env, stdout=stdout, stderr=stderr, timeout=timeout, check=False
            ).returncode
        except subprocess.TimeoutExpired:
            code = 124
    write_json(
        output / f"{name}.command.json",
        {"args": args, "cwd": str(cwd), "started_at": started, "finished_at": now(), "exit_code": code},
    )
    print(f"{name}: exit {code}", flush=True)
    return code


def run(args: argparse.Namespace) -> int:
    repo = args.repo.resolve()
    output = args.output.resolve()
    hate = args.hate_root.resolve()
    qeg = args.qeg_root.resolve()
    if any(output.is_relative_to(root) for root in (repo, hate, qeg)):
        raise ValueError("output must be outside the source repositories")
    output.mkdir(parents=True, exist_ok=False)
    try:
        return execute(args, repo, output, hate, qeg)
    except (ValueError, KeyError, OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        write_json(output / "result.json", {"status": "blocked", "error": str(exc), "release_approval": None})
        print(str(exc), file=sys.stderr)
        return 1


def execute(args: argparse.Namespace, repo: Path, output: Path, hate: Path, qeg: Path) -> int:
    env = {
        **os.environ,
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
        "COVERAGE_FILE": str(output / ".coverage"),
        "UV_OFFLINE": "true",
    }
    # 呼出元のcoverageが他repoのcollectorを計測しない。pytest側で明示的にcovを開始する。
    for name in list(env):
        if name.startswith(("COV_CORE_", "COVERAGE_PROCESS_START")) or name == "PYTEST_ADDOPTS":
            env.pop(name)
    hate_lock = freeze_tool(hate, output / "toolchain/hate", ("src", "schemas", "pyproject.toml"))
    qeg_lock = freeze_tool(qeg, output / "toolchain/qeg", ("dist", "src", "schemas", "package.json", "node_modules"))
    hate, qeg = Path(hate_lock["root"]), Path(qeg_lock["root"])
    snapshots = {"target": snapshot(repo), "hate": hate_lock, "qeg": qeg_lock}
    write_json(output / "source-snapshot.json", snapshots)
    write_json(output / "expected-gate-verdict.json", expected_gate())
    run_id = "hate:wfc-" + uuid.uuid4().hex
    env.update(
        {
            "WFC_HATE_ROOT": str(hate),
            "WFC_QEG_ROOT": str(qeg),
            "WFC_HATE_PYTHON": str(args.hate_python),
            "WFC_NODE": str(args.node),
        }
    )
    target = {
        "projectId": "workflow-cookbook",
        "buildId": snapshots["target"]["content_hash"],
        "revision": snapshots["target"]["revision"],
        "environmentId": "local-" + sys.platform + "-" + args.python.name,
    }
    raw = output / "raw"
    raw.mkdir()
    test_args = [
        str(args.python),
        "-B",
        "-m",
        "pytest",
        "tests",
        "-q",
        "--junitxml=" + str(raw / "junit.xml"),
        "--cov=tools",
        "--cov=security_headers",
        "--cov-report=term",
        "--cov-report=json:" + str(raw / "coverage-summary.json"),
        "--cov-report=xml:" + str(raw / "cobertura.xml"),
        "--cov-fail-under=80",
    ]
    started = now()
    pytest_code = command(test_args, repo, output, "pytest", env)
    finished = now()
    receipt = {
        "run_id": run_id,
        "target": target,
        "started_at": started,
        "finished_at": finished,
        "evaluated_at": now(),
        "pytest_exit_code": pytest_code,
        "coverage_min": 80,
        "command": test_args,
        "source_snapshot_hash": file_hash(output / "source-snapshot.json"),
        "execution_scope": "local pytest; includes unit/CLI/isolated integration tests; not a production service run",
    }
    write_json(output / "receipt.json", receipt)
    context = {
        "provider": "generic-ci",
        "repository": "workflow-cookbook",
        "workflow": "local-hate-qeg",
        "job": "pytest",
        "event_name": "local",
        "run_id": run_id,
        "run_attempt": 1,
        "commit_sha": target["revision"],
        "started_at": started,
        "finished_at": finished,
    }
    write_json(raw / "ci-context.json", context)
    hate_version = tomllib.loads((hate / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    hate_cmd = [str(args.hate_python), "-B", "-m", "hate"]
    hate_env = {**env, "PYTHONPATH": str(hate / "src")}
    codes = {"pytest": pytest_code}
    codes["hate_p0a"] = command(
        [*hate_cmd, "p0a", "--input", str(raw), "--out", str(output / "hate/p0a"), "--source-version", hate_version],
        hate,
        output,
        "hate-p0a",
        hate_env,
    )
    if codes["hate_p0a"] != 0:
        raise ValueError("HATE p0a failed; see hate-p0a logs and precheck-decision.json")
    # 既存suiteの回帰受入。未実施のCTG解析やdiff-risk mappingを捏造しない。
    write_json(
        output / "hate/diff-risk-test.json",
        {
            "schema_version": "HATE/v1",
            "source_tool": "workflow-cookbook",
            "commit_sha": target["revision"],
            "changed_entities": [],
            "risks": [],
            "test_obligations": [],
        },
    )
    codes["hate_export"] = command(
        [*hate_cmd, "export", "qeg", "--fixture", str(output / "hate"), "--out", str(output / "hate/export")],
        hate,
        output,
        "hate-export",
        hate_env,
    )
    if codes["hate_export"] != 0:
        raise ValueError("HATE export failed; see hate-export logs")
    receipt["evaluated_at"] = now()
    write_json(output / "receipt.json", receipt)
    summary = project(output, receipt)
    qeg_cmd = [str(args.node), str(qeg / "dist/cli.js")]
    for action in ("validate", "gate", "record"):
        codes["qeg_" + action] = command([*qeg_cmd, action, str(output)], qeg, output, "qeg-" + action, env)
    codes["qeg_outputs_read"] = command(
        [*qeg_cmd, "outputs", "read", str(output)], qeg, output, "qeg-outputs-read", env
    )
    after = {
        "target": snapshot(repo),
        "hate": {**hate_lock, **frozen_snapshot(hate)},
        "qeg": {**qeg_lock, **frozen_snapshot(qeg)},
    }
    changed = [name for name in snapshots if snapshots[name] != after[name]]
    write_json(output / "source-verification.json", {"unchanged": not changed, "changed": changed, "after": after})
    gate = read_json(output / "qeg-gate.stdout.log")
    okay = not changed and all(value == 0 for value in codes.values()) and gate.get("verdict") == "go"
    result = {
        "status": "passed" if okay else "failed",
        **summary,
        "qeg_verdict": gate.get("verdict"),
        "commands": codes,
        "source_unchanged": not changed,
        "release_approval": None,
        "not_evaluated": [
            "RanD",
            "Code-to-gate",
            "manual-bb",
            "model performance",
            "production deployment",
            "release approval",
        ],
    }
    write_json(output / "result.json", result)
    write_json(
        output / "artifact-hashes.json",
        {
            p.relative_to(output).as_posix(): file_hash(p)
            for p in sorted(output.rglob("*"))
            if p.is_file() and not p.name.startswith(".coverage") and p.name != "artifact-hashes.json"
        },
    )
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if okay else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--hate-root", type=Path, required=True)
    parser.add_argument("--hate-python", type=Path, required=True)
    parser.add_argument("--qeg-root", type=Path, required=True)
    parser.add_argument("--node", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
