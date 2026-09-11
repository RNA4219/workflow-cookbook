from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import jsonschema

from tools.supervision.workspace_coordinator import WorkspaceCoordinator, deterministic_job_key

SCRIPT = Path(__file__).parents[2] / "tools" / "supervision" / "workspace_coordinator.py"


def coordinator(tmp_path: Path, workspace: Path, *, clock=None) -> WorkspaceCoordinator:
    return WorkspaceCoordinator(workspace, state_root=tmp_path / "state", clock=clock or __import__("time").time)


def test_parallel_reads_and_overlapping_write_conflict(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    service = coordinator(tmp_path, workspace)
    first = service.acquire(mode="read", owner="a", task_id="t1", paths=["src"], wip_key=None)
    second = service.acquire(mode="read", owner="b", task_id="t2", paths=["src"], wip_key=None)
    blocked = service.acquire(mode="write", owner="c", task_id="t3", paths=["src"], wip_key=None)

    assert first["acquired"] is True
    assert second["acquired"] is True
    assert blocked["acquired"] is False
    assert blocked["reason"] == "workspace_busy"


def test_disjoint_write_scopes_are_allowed_but_parent_scope_conflicts(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    service = coordinator(tmp_path, workspace)
    left = service.acquire(mode="write", owner="a", task_id="t1", paths=["src/a"], wip_key=None)
    right = service.acquire(mode="write", owner="b", task_id="t2", paths=["src/b"], wip_key=None)
    parent = service.acquire(mode="write", owner="c", task_id="t3", paths=["src"], wip_key=None)

    assert left["acquired"] is True
    assert right["acquired"] is True
    assert parent["reason"] == "workspace_busy"


def test_global_wip_serializes_io_across_workspaces(tmp_path: Path) -> None:
    first_workspace = tmp_path / "one"
    second_workspace = tmp_path / "two"
    first_workspace.mkdir()
    second_workspace.mkdir()
    first = coordinator(tmp_path, first_workspace)
    second = coordinator(tmp_path, second_workspace)

    acquired = first.acquire(mode="read", owner="a", task_id="t1", wip_key="workspace_io_global")
    blocked = second.acquire(mode="read", owner="b", task_id="t2", wip_key="workspace_io_global")

    assert acquired["acquired"] is True
    assert blocked["reason"] == "wip_busy"


def test_heartbeat_token_expiry_and_fence_token(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    now = [100.0]
    service = coordinator(tmp_path, workspace, clock=lambda: now[0])
    acquired = service.acquire(mode="write", owner="a", task_id="t1", ttl_seconds=5, wip_key=None)
    lease = acquired["lease"]

    assert service.heartbeat(lease["lease_id"], "wrong")["reason"] == "token_mismatch"
    now[0] = 106.0
    assert service.heartbeat(lease["lease_id"], acquired["lease_token"])["reason"] == "lease_not_active"
    replacement = service.acquire(mode="write", owner="b", task_id="t2", wip_key=None)
    assert replacement["acquired"] is True
    assert replacement["lease"]["fence_token"] > lease["fence_token"]


def test_terminal_job_is_reused_and_token_is_not_persisted(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    service = coordinator(tmp_path, workspace)
    acquired = service.acquire(mode="write", owner="a", task_id="t1", job_key="job:exact", wip_key=None)
    finished = service.finish(
        acquired["lease"]["lease_id"],
        acquired["lease_token"],
        outcome="succeeded",
        job_key="job:exact",
        result_refs=["artifact/result.json"],
    )
    reused = service.acquire(mode="write", owner="b", task_id="t2", job_key="job:exact", wip_key=None)

    assert finished["ok"] is True
    assert reused["reason"] == "job_already_terminal"
    assert reused["job"]["result_refs"] == ["artifact/result.json"]
    assert reused["job"]["reuse_requires_validation"] is True
    with sqlite3.connect(service.db_path) as connection:
        token_sha = connection.execute("SELECT token_sha256 FROM leases").fetchone()[0]
    assert token_sha != acquired["lease_token"]
    assert acquired["lease_token"] not in service.db_path.read_bytes().decode("latin1")


def test_heartbeat_release_and_invalid_job_reuse(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    service = coordinator(tmp_path, workspace)
    released_lease = service.acquire(mode="write", owner="a", task_id="t1", job_key="job:released", wip_key=None)

    heartbeat = service.heartbeat(released_lease["lease"]["lease_id"], released_lease["lease_token"])
    released = service.release(released_lease["lease"]["lease_id"], released_lease["lease_token"])
    reacquired = service.acquire(mode="write", owner="b", task_id="t2", job_key="job:invalid", wip_key=None)
    invalid = service.finish(
        reacquired["lease"]["lease_id"],
        reacquired["lease_token"],
        outcome="invalid",
        job_key="job:invalid",
    )
    reused = service.acquire(mode="write", owner="c", task_id="t3", job_key="job:invalid", wip_key=None)

    assert heartbeat["ok"] is True
    assert released["ok"] is True
    assert invalid["job"]["status"] == "invalid"
    assert reused["reason"] == "job_already_terminal"
    assert reused["job"]["status"] == "invalid"


def test_deterministic_job_key_ignores_input_hash_order(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    params = {
        "workspace": workspace,
        "scope": "evaluation",
        "role": "Evaluator",
        "stage": "G200",
        "source_revision": "abc123",
        "command_identity": "runner-v1",
        "output_root": "artifacts/run-1",
    }
    first = deterministic_job_key(input_hashes=["b", "a"], **params)
    second = deterministic_job_key(input_hashes=["a", "b", "a"], **params)
    assert first["job_key"] == second["job_key"]
    assert first["job_key"].startswith("sha256:")


def test_two_processes_cannot_acquire_global_writer_together(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    state_root = tmp_path / "state"
    base = [
        sys.executable,
        str(SCRIPT),
        "acquire",
        "--workspace",
        str(workspace),
        "--state-root",
        str(state_root),
        "--mode",
        "write",
        "--task-id",
        "race",
    ]
    first = subprocess.Popen(base + ["--owner", "one"], stdout=subprocess.PIPE, text=True)
    second = subprocess.Popen(base + ["--owner", "two"], stdout=subprocess.PIPE, text=True)
    outputs = [json.loads(first.communicate(timeout=10)[0]), json.loads(second.communicate(timeout=10)[0])]
    assert sum(item.get("acquired") is True for item in outputs) == 1
    assert sum(item.get("reason") in {"wip_busy", "workspace_busy"} for item in outputs) == 1
    assert sorted([first.returncode, second.returncode]) == [0, 2]


def test_probe_has_hard_timeout(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "probe",
            "--workspace",
            str(workspace),
            "--state-root",
            str(tmp_path / "state"),
            "--timeout-ms",
            "20",
            "--probe-delay-ms",
            "200",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=5,
    )
    assert completed.returncode == 4
    assert json.loads(completed.stdout)["reason"] == "probe_timeout"


def test_cli_lifecycle_returns_json_and_reuses_terminal_job(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    state_root = tmp_path / "state"

    def run(*arguments: str, expected_code: int = 0) -> dict[str, object]:
        completed = subprocess.run(
            [sys.executable, str(SCRIPT), *arguments],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        assert completed.returncode == expected_code, completed.stderr
        return json.loads(completed.stdout)

    job = run(
        "job-key",
        "--workspace",
        str(workspace),
        "--scope",
        "smoke",
        "--role",
        "Supervisor",
        "--stage",
        "lifecycle",
        "--source-revision",
        "local",
        "--input-hash",
        "sha256:smoke",
        "--command-identity",
        "cli-smoke-v1",
        "--output-root",
        "artifacts/smoke-unused",
    )
    acquired = run(
        "acquire",
        "--workspace",
        str(workspace),
        "--state-root",
        str(state_root),
        "--mode",
        "write",
        "--owner",
        "supervisor",
        "--task-id",
        "smoke",
        "--job-key",
        str(job["job_key"]),
    )
    lease = acquired["lease"]
    assert isinstance(lease, dict)
    heartbeat = run(
        "heartbeat",
        "--workspace",
        str(workspace),
        "--state-root",
        str(state_root),
        "--lease-id",
        str(lease["lease_id"]),
        "--lease-token",
        str(acquired["lease_token"]),
    )
    finished = run(
        "finish",
        "--workspace",
        str(workspace),
        "--state-root",
        str(state_root),
        "--lease-id",
        str(lease["lease_id"]),
        "--lease-token",
        str(acquired["lease_token"]),
        "--job-key",
        str(job["job_key"]),
        "--outcome",
        "succeeded",
        "--result-ref",
        "smoke:pass",
    )
    reused = run(
        "acquire",
        "--workspace",
        str(workspace),
        "--state-root",
        str(state_root),
        "--mode",
        "write",
        "--owner",
        "supervisor-2",
        "--task-id",
        "smoke-reuse",
        "--job-key",
        str(job["job_key"]),
    )

    assert acquired["acquired"] is True
    assert heartbeat["ok"] is True
    assert finished["ok"] is True
    assert reused["reason"] == "job_already_terminal"
    assert isinstance(reused["job"], dict)
    assert reused["job"]["status"] == "succeeded"


def test_public_lease_and_job_match_schemas(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    service = coordinator(tmp_path, workspace)
    acquired = service.acquire(mode="write", owner="a", task_id="t1", job_key="schema-job", wip_key=None)
    root = Path(__file__).parents[2]
    lease_schema = json.loads((root / "schemas" / "workspace-lease.schema.json").read_text(encoding="utf-8"))
    jsonschema.validate(acquired["lease"], lease_schema)
    finished = service.finish(
        acquired["lease"]["lease_id"],
        acquired["lease_token"],
        outcome="invalid",
        job_key="schema-job",
    )
    job_schema = json.loads((root / "schemas" / "job-ledger.schema.json").read_text(encoding="utf-8"))
    jsonschema.validate(finished["job"], job_schema)
