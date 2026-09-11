import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from tools.evaluation.workflow import digest
from tools.supervision.workspace_coordinator import WorkspaceCoordinator
from tools.workflow_plugins.checkpoint import TaskstateCLI, WorkflowCheckpoint, main


class MemoryStore:
    def __init__(self):
        self.state = {
            "task_id": "task",
            "revision": 1,
            "current_step": "before",
            "context_policy": {"unrelated": "keep"},
            "constraints": ["preserve"],
        }
        self.conflict = False

    def get(self, task_id):
        return deepcopy(self.state)

    def patch(self, task_id, revision, payload):
        if self.conflict or revision != self.state["revision"]:
            raise RuntimeError("revision mismatch")
        self.state.update(deepcopy(payload))
        self.state["revision"] += 1
        return deepcopy(self.state)


def artifact(root, name="output.txt"):
    path = root / name
    if not path.exists():
        path.write_text(name + " 日本語", encoding="utf-8")
    return {"path": name, "sha256": digest(path)}


@pytest.fixture
def setup(tmp_path):
    clock = [100.0]
    coordinator = WorkspaceCoordinator(tmp_path, state_root=tmp_path / "coord", clock=lambda: clock[0])
    plan = {
        "schema_version": "1.0",
        "task_id": "task",
        "run_id": "run",
        "policy_version": "fixture-v1",
        "steps": ["one", "two"],
        "inputs": [artifact(tmp_path, "input.txt")],
    }
    store = MemoryStore()
    checkpoint = WorkflowCheckpoint(store, coordinator, plan)
    checkpoint.grant = coordinator.acquire(mode="write", owner="fixture", task_id="task", job_key=checkpoint.job_key)
    return checkpoint, store, coordinator, clock


def finish_steps(checkpoint):
    checkpoint.start()
    for step in checkpoint.steps:
        checkpoint.transition("begin", step)
        checkpoint.transition("complete", step, artifacts=[artifact(checkpoint.coordinator.workspace, step + ".txt")])


@pytest.mark.parametrize(
    "change,message",
    [
        ("task", "state task mismatch"),
        ("policy", "context_policy object"),
        ("checkpoint", "invalid checkpoint"),
        ("steps", "step order"),
        ("status", "step status"),
        ("uncommitted", "unfinished step"),
    ],
)
def test_malformed_persisted_state_is_rejected_without_writes(setup, change, message):
    checkpoint, store, coordinator, _ = setup
    checkpoint.start()
    saved = store.state["context_policy"]["workflow_checkpoint"]
    if change == "task":
        store.state["task_id"] = "different"
    elif change == "policy":
        store.state["context_policy"] = []
    elif change == "checkpoint":
        saved["schema_version"] = "2"
    elif change == "steps":
        saved["steps"].reverse()
    elif change == "status":
        saved["steps"][0]["status"] = "failed"
    else:
        saved["steps"][0]["artifacts"] = [artifact(coordinator.workspace)]
    before = deepcopy(store.state)
    with pytest.raises(ValueError, match=message):
        checkpoint.status()
    assert store.state == before


@pytest.mark.parametrize("state,resolution", [("pending", "completed"), ("running", "unknown")])
def test_reconciliation_requires_an_in_doubt_step_and_valid_decision(setup, state, resolution):
    checkpoint, store, coordinator, _ = setup
    checkpoint.start()
    if state == "running":
        checkpoint.transition("begin", "one")
    before = deepcopy(store.state)
    with pytest.raises(ValueError, match="reconcile requires"):
        checkpoint.transition(
            "reconcile", "one", resolution=resolution, record=artifact(coordinator.workspace, "decision.md")
        )
    assert store.state == before


def test_lease_expiring_between_validation_and_finish_cannot_report_success(setup, monkeypatch):
    checkpoint, store, coordinator, clock = setup
    finish_steps(checkpoint)
    before = deepcopy(store.state)
    real_finish = coordinator.finish

    def expire_then_finish(*args, **kwargs):
        clock[0] += 301
        return real_finish(*args, **kwargs)

    monkeypatch.setattr(coordinator, "finish", expire_then_finish)
    with pytest.raises(ValueError, match="finalize rejected: lease_not_active"):
        checkpoint.finalize()
    assert store.state == before
    assert coordinator.status()["jobs"][0]["status"] == "expired"
    checkpoint.grant = coordinator.acquire(mode="write", owner="resumer", task_id="task", job_key=checkpoint.job_key)
    monkeypatch.setattr(coordinator, "finish", real_finish)
    assert checkpoint.finalize()["job"]["status"] == "succeeded"


def test_cli_can_persist_reconcile_complete_and_finalize_across_invocations(tmp_path, monkeypatch, capsys):
    from tools.workflow_plugins import checkpoint as module

    store = MemoryStore()
    coordinator = WorkspaceCoordinator(tmp_path, state_root=tmp_path / "coord")
    plan = {
        "schema_version": "1.0",
        "task_id": "task",
        "run_id": "run",
        "policy_version": "fixture",
        "steps": ["one"],
        "inputs": [artifact(tmp_path, "input.txt")],
    }
    session = WorkflowCheckpoint(store, coordinator, plan)
    grant = coordinator.acquire(mode="write", owner="fixture", task_id="task", job_key=session.job_key)
    payloads = {
        "plan.json": plan,
        "client.json": {},
        "grant.json": grant,
        "artifacts.json": {"artifacts": [artifact(tmp_path)]},
        "record.json": artifact(tmp_path, "decision.md"),
    }
    for name, payload in payloads.items():
        (tmp_path / name).write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(module, "TaskstateCLI", lambda **kwargs: store)
    common = [
        "--plan",
        str(tmp_path / "plan.json"),
        "--state-client",
        str(tmp_path / "client.json"),
        "--workspace",
        str(tmp_path),
        "--coordinator-root",
        str(tmp_path / "coord"),
        "--grant",
        str(tmp_path / "grant.json"),
    ]

    def invoke(action, *arguments, code=0):
        assert main([action, *common, *arguments]) == code
        return json.loads(capsys.readouterr().out)

    assert invoke("start")["status"] == "ready"
    assert invoke("begin", "--step", "one")["running"] == ["one"]
    assert invoke("status", code=2)["status"] == "needs_reconciliation"
    reconciled = invoke(
        "reconcile", "--step", "one", "--resolution", "pending", "--record", str(tmp_path / "record.json")
    )
    assert reconciled["checkpoint"]["steps"][0]["reconciliations"][0]["resolution"] == "pending"
    invoke("begin", "--step", "one")
    assert invoke("complete", "--step", "one", "--artifacts", str(tmp_path / "artifacts.json"))["status"] == "completed"
    assert invoke("finalize")["job"]["status"] == "succeeded"


def test_persistent_identity_idempotent_start_and_complete(setup):
    checkpoint, store, coordinator, _ = setup
    assert checkpoint.status()["status"] == "not_started"
    assert checkpoint.start()["remaining"] == ["one", "two"]
    before = store.state["revision"]
    checkpoint.start()
    assert store.state["revision"] == before
    checkpoint.transition("begin", "one")
    artifacts = [artifact(coordinator.workspace)]
    checkpoint.transition("complete", "one", artifacts=artifacts)
    before = store.state["revision"]
    checkpoint.transition("complete", "one", artifacts=artifacts)
    assert store.state["revision"] == before
    restored = WorkflowCheckpoint(
        store,
        WorkspaceCoordinator(coordinator.workspace, state_root=coordinator.state_dir, clock=coordinator.clock),
        checkpoint.plan,
    )
    assert restored.status()["remaining"] == ["two"]
    assert store.state["context_policy"]["unrelated"] == "keep"
    assert store.state["constraints"] == ["preserve"]
    assert "lease_token" not in json.dumps(store.state)


def test_interruption_blocks_automatic_rerun_until_recorded_resolution(setup):
    checkpoint, store, coordinator, _ = setup
    checkpoint.start()
    checkpoint.transition("begin", "one")
    restored = WorkflowCheckpoint(store, coordinator, checkpoint.plan, grant=checkpoint.grant)
    assert restored.status()["status"] == "needs_reconciliation"
    with pytest.raises(ValueError, match="reconciliation"):
        restored.transition("begin", "one")
    record = artifact(coordinator.workspace, "decision.md")
    restored.transition("reconcile", "one", resolution="pending", record=record)
    assert restored.status()["status"] == "ready"
    restored.transition("begin", "one")
    restored.transition(
        "reconcile", "one", resolution="completed", record=record, artifacts=[artifact(coordinator.workspace)]
    )
    assert restored.status()["remaining"] == ["two"]
    decisions = restored.status()["checkpoint"]["steps"][0]["reconciliations"]
    assert [d["resolution"] for d in decisions] == ["pending", "completed"]


def test_reacquire_after_checkpoint_before_finish_and_terminal_reuse(setup):
    checkpoint, store, coordinator, clock = setup
    finish_steps(checkpoint)
    clock[0] += 400
    restored = WorkflowCheckpoint(store, coordinator, checkpoint.plan)
    restored.grant = coordinator.acquire(mode="write", owner="resumer", task_id="task", job_key=restored.job_key)
    assert restored.grant["acquired"]
    assert restored.finalize()["job"]["status"] == "succeeded"
    assert restored.finalize()["artifacts_verified"]
    (coordinator.workspace / "one.txt").write_text("modified")
    with pytest.raises(ValueError, match="hash"):
        restored.finalize()


def test_invalid_terminal_job_is_not_promoted(setup):
    checkpoint, _, coordinator, _ = setup
    finish_steps(checkpoint)
    coordinator.finish(
        checkpoint.grant["lease"]["lease_id"],
        checkpoint.grant["lease_token"],
        outcome="invalid",
        job_key=checkpoint.job_key,
    )
    with pytest.raises(ValueError, match="invalid job"):
        checkpoint.finalize()


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", "2"),
        ("steps", []),
        ("steps", ["one", "one"]),
        ("steps", [""]),
        ("inputs", []),
        ("task_id", ""),
        ("policy_version", None),
    ],
)
def test_invalid_plans_fail(setup, field, value):
    checkpoint, store, coordinator, _ = setup
    plan = deepcopy(checkpoint.plan)
    plan[field] = value
    with pytest.raises(ValueError):
        WorkflowCheckpoint(store, coordinator, plan)


@pytest.mark.parametrize("change", ["input", "policy", "output", "record"])
def test_stale_inputs_outputs_policy_and_reconciliation_records_fail(setup, change):
    checkpoint, store, coordinator, _ = setup
    checkpoint.start()
    checkpoint.transition("begin", "one")
    checkpoint.transition(
        "reconcile",
        "one",
        resolution="completed",
        record=artifact(coordinator.workspace, "record.md"),
        artifacts=[artifact(coordinator.workspace)],
    )
    if change == "policy":
        plan = deepcopy(checkpoint.plan)
        plan["policy_version"] = "v2"
        checkpoint = WorkflowCheckpoint(store, coordinator, plan)
    else:
        path = {"input": "input.txt", "output": "output.txt", "record": "record.md"}[change]
        (coordinator.workspace / path).write_text("changed")
    with pytest.raises(ValueError):
        checkpoint.status()


@pytest.mark.parametrize("change", ["expired", "wrong_task", "wrong_job", "read", "scope", "token", "fence", "absent"])
def test_invalid_lease_cannot_save(setup, change):
    checkpoint, store, coordinator, clock = setup
    if change == "expired":
        clock[0] += 400
    elif change in ("wrong_task", "wrong_job", "read", "scope"):
        coordinator.release(checkpoint.grant["lease"]["lease_id"], checkpoint.grant["lease_token"])
        checkpoint.grant = coordinator.acquire(
            mode="read" if change == "read" else "write",
            owner="other",
            task_id="other" if change == "wrong_task" else "task",
            job_key="other" if change == "wrong_job" else checkpoint.job_key,
            paths=["part"] if change == "scope" else None,
        )
    elif change == "token":
        checkpoint.grant["lease_token"] = "wrong"
    elif change == "fence":
        checkpoint.grant["lease"]["fence_token"] += 1
    else:
        checkpoint.grant = {}
    with pytest.raises(ValueError):
        checkpoint.start()
    assert "workflow_checkpoint" not in store.state["context_policy"]


def test_cas_conflict_preserves_original_state(setup):
    checkpoint, store, _, _ = setup
    checkpoint.start()
    before = deepcopy(store.state)
    store.conflict = True
    with pytest.raises(RuntimeError, match="revision"):
        checkpoint.transition("begin", "one")
    assert store.state == before


@pytest.mark.parametrize("action,step", [("begin", "two"), ("complete", "one"), ("bad", "one"), ("begin", "missing")])
def test_order_and_transition_guards(setup, action, step):
    checkpoint, _, _, _ = setup
    checkpoint.start()
    with pytest.raises(ValueError):
        checkpoint.transition(action, step)


def test_missing_start_and_incomplete_finalize(setup):
    checkpoint, _, _, _ = setup
    with pytest.raises(ValueError, match="not started"):
        checkpoint.transition("begin", "one")
    with pytest.raises(ValueError, match="all steps"):
        checkpoint.finalize()


def test_completed_artifact_set_cannot_be_replaced(setup):
    checkpoint, _, coordinator, _ = setup
    finish_steps(checkpoint)
    with pytest.raises(ValueError, match="artifact set changed"):
        checkpoint.transition("complete", "one", artifacts=[artifact(coordinator.workspace, "other.txt")])


@pytest.mark.parametrize("kind", ["absolute", "outside", "duplicate", "missing", "bad_hash", "non_object"])
def test_invalid_artifacts(setup, kind):
    checkpoint, _, coordinator, _ = setup
    checkpoint.start()
    checkpoint.transition("begin", "one")
    item = artifact(coordinator.workspace)
    values = [item]
    if kind == "absolute":
        item["path"] = str(coordinator.workspace / item["path"])
    elif kind == "outside":
        outside = coordinator.workspace.parent / "outside.txt"
        outside.write_text("outside")
        item.update(path="../outside.txt", sha256=digest(outside))
    elif kind == "duplicate":
        values.append(dict(item))
    elif kind == "missing":
        item["path"] = "absent.txt"
    elif kind == "non_object":
        values = ["bad"]
    else:
        item["sha256"] = "sha256:" + "0" * 64
    with pytest.raises((ValueError, OSError)):
        checkpoint.transition("complete", "one", artifacts=values)


def test_corrupt_state_is_rejected(setup):
    checkpoint, store, _, _ = setup
    checkpoint.start()
    store.state["context_policy"]["workflow_checkpoint"]["steps"][1]["status"] = "running"
    with pytest.raises(ValueError, match="out-of-order"):
        checkpoint.status()


@pytest.mark.parametrize(
    "payload,code",
    [
        ({"ok": False, "error": {"code": "conflict"}}, 1),
        ({"ok": True, "data": []}, 0),
        ([], 0),
        ({"ok": True, "data": {}}, 3),
    ],
)
def test_cli_envelope_errors(tmp_path, monkeypatch, payload, code):
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: subprocess.CompletedProcess([], code, json.dumps(payload), "")
    )
    client = TaskstateCLI(["python"], cwd=tmp_path, db=tmp_path / "db")
    with pytest.raises((ValueError, RuntimeError)):
        client.get("task")


def test_cli_timeout_is_unknown_outcome(tmp_path, monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired([], 1)

    monkeypatch.setattr(subprocess, "run", timeout)
    client = TaskstateCLI(["python"], cwd=tmp_path, db=tmp_path / "db")
    with pytest.raises(RuntimeError, match="outcome unknown"):
        client.patch("task", 1, {"current_step": "x"})


def test_cli_argv_json_file_and_cleanup(tmp_path, monkeypatch):
    captured = []

    def execute(argv, **kwargs):
        path = Path(argv[-1])
        captured.append(path)
        assert json.loads(path.read_text(encoding="utf-8"))["current_step"] == "日本語"
        assert argv[:2] == ["trusted", "--db"]
        assert "--expected-revision" in argv and not kwargs.get("shell")
        assert kwargs["env"]["PYTHONIOENCODING"] == "utf-8"
        return subprocess.CompletedProcess(argv, 0, '{"ok": true, "data": {"revision": 2}}', "")

    monkeypatch.setattr(subprocess, "run", execute)
    client = TaskstateCLI(["trusted"], cwd=tmp_path, db=tmp_path / "db")
    assert client.patch("task", 1, {"current_step": "日本語"})["revision"] == 2
    assert not captured[0].exists()


@pytest.mark.parametrize("command", ["bad string", [], [""]])
def test_client_requires_argv(tmp_path, command):
    with pytest.raises(ValueError):
        TaskstateCLI(command, cwd=tmp_path, db=tmp_path / "db")


def test_command_line_status_and_error(setup, tmp_path, monkeypatch, capsys):
    checkpoint, store, _, _ = setup
    monkeypatch.setattr("tools.workflow_plugins.checkpoint.TaskstateCLI", lambda **kwargs: store)
    config = tmp_path / "client.json"
    config.write_text("{}")
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(checkpoint.plan))
    args = [
        "status",
        "--plan",
        str(plan),
        "--state-client",
        str(config),
        "--workspace",
        str(tmp_path),
        "--coordinator-root",
        str(tmp_path / "coord"),
    ]
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "not_started"
    checkpoint.start()
    checkpoint.transition("begin", "one")
    assert main(args) == 2
    capsys.readouterr()
    plan.write_text("{}")
    assert main(args) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "invalid"


def test_real_taskstate_cli_reopen_and_cas(tmp_path):
    sibling_src = Path(__file__).resolve().parents[2] / "agent-taskstate" / "src"
    if not (sibling_src / "agent_taskstate").is_dir():
        pytest.skip("optional sibling agent-taskstate integration")
    client = TaskstateCLI(
        [sys.executable, "-B", "-m", "agent_taskstate.cli"],
        cwd=sibling_src,
        db=tmp_path / "taskstate.sqlite3",
    )
    client._call(["init"])
    client._call(
        [
            "task",
            "create",
            "--json",
            json.dumps({"id": "task", "kind": "feature", "title": "fixture", "goal": "resume fixture"}),
        ]
    )
    client._call(
        [
            "state",
            "put",
            "--task",
            "task",
            "--json",
            json.dumps({"current_step": "初期状態", "context_policy": {"keep": True}}),
        ]
    )
    coordinator = WorkspaceCoordinator(tmp_path, state_root=tmp_path / "coord")
    assert client.get("task")["current_step"] == "初期状態"
    plan = {
        "schema_version": "1.0",
        "task_id": "task",
        "run_id": "run",
        "policy_version": "fixture",
        "steps": ["one"],
        "inputs": [artifact(tmp_path, "input.txt")],
    }
    checkpoint = WorkflowCheckpoint(client, coordinator, plan)
    checkpoint.grant = coordinator.acquire(mode="write", owner="fixture", task_id="task", job_key=checkpoint.job_key)
    finish_steps(checkpoint)
    restored_client = TaskstateCLI(client.command, cwd=sibling_src, db=tmp_path / "taskstate.sqlite3")
    restored = WorkflowCheckpoint(restored_client, coordinator, plan, grant=checkpoint.grant)
    assert restored.status()["status"] == "completed"
    assert restored_client.get("task")["context_policy"]["keep"] is True
    with pytest.raises(RuntimeError, match="revision"):
        restored_client.patch("task", 1, {"current_step": "stale"})
    assert restored.finalize()["job"]["status"] == "succeeded"
