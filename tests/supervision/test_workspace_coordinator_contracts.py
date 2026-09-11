"""共有workspaceの状態遷移・永続化・CLI契約を一時DBで検証する。"""

import json
import os
import sqlite3

import pytest

from tools.supervision import workspace_coordinator as coordinator


@pytest.mark.parametrize("failure", ["busy_once", "busy_timeout", "non_busy", "unexpected"])
def test_connection_initialization_retries_only_busy_within_deadline(tmp_path, monkeypatch, failure):
    workspace = tmp_path / "repo"
    workspace.mkdir()
    connect = sqlite3.connect
    created = []
    closed = []
    error = ValueError("invalid configuration") if failure == "unexpected" else sqlite3.OperationalError("fixture")
    if isinstance(error, sqlite3.OperationalError):
        error.sqlite_errorcode = sqlite3.SQLITE_ERROR if failure == "non_busy" else sqlite3.SQLITE_BUSY

    class FirstConnection(sqlite3.Connection):
        def execute(self, sql, *args, **kwargs):
            if sql == "PRAGMA journal_mode=WAL":
                raise error
            return super().execute(sql, *args, **kwargs)

        def close(self):
            closed.append(self)
            return super().close()

    def open_connection(*args, **kwargs):
        if not created:
            kwargs["factory"] = FirstConnection
        result = connect(*args, **kwargs)
        created.append(result)
        return result

    monkeypatch.setattr(coordinator.sqlite3, "connect", open_connection)
    options = {"state_root": tmp_path / "state", "timeout_ms": 0 if failure == "busy_timeout" else 1000}
    if failure == "busy_once":
        service = coordinator.WorkspaceCoordinator(workspace, **options)
        assert len(created) == 2
        lease = service.acquire(mode="write", owner="a", task_id="t", job_key=None, wip_key=None)
        assert lease["acquired"] is True
    else:
        with pytest.raises(type(error)) as caught:
            coordinator.WorkspaceCoordinator(workspace, **options)
        assert caught.value is error
        assert len(created) == 1
    assert closed == [created[0]]
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        created[0].execute("SELECT 1")


@pytest.fixture
def workspace(tmp_path):
    path = tmp_path / "repo"
    path.mkdir()
    return path


@pytest.fixture
def service(workspace, tmp_path):
    clock = [100.0]
    instance = coordinator.WorkspaceCoordinator(workspace, state_root=tmp_path / "state", clock=lambda: clock[0])
    return instance, clock


@pytest.fixture
def cli(workspace, tmp_path, capsys):
    def invoke(command, *arguments, expected_code=0):
        code = coordinator.main(
            [
                command,
                "--workspace",
                str(workspace),
                "--state-root",
                str(tmp_path / "state"),
                *arguments,
            ]
        )
        captured = capsys.readouterr()
        assert code == expected_code, captured
        assert captured.err == ""
        return json.loads(captured.out)

    return invoke


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"mode": "append"}, "mode must be"),
        ({"owner": ""}, "owner and task_id"),
        ({"task_id": ""}, "owner and task_id"),
        ({"ttl_seconds": 0}, "ttl_seconds must be positive"),
        ({"ttl_seconds": -1}, "ttl_seconds must be positive"),
        ({"paths": ["../outside"]}, "scope escapes workspace"),
    ],
)
def test_invalid_acquisition_leaves_no_lease_job_or_event(service, overrides, message):
    instance, _ = service
    arguments = {"mode": "write", "owner": "one", "task_id": "task", "job_key": "job"}
    with pytest.raises(ValueError, match=message):
        instance.acquire(**(arguments | overrides))
    state = instance.status(include_events=10)
    assert state["active_leases"] == []
    assert state["jobs"] == []
    assert state["events"] == []


def test_workspace_must_be_directory(tmp_path):
    source = tmp_path / "file.txt"
    source.write_text("fixture", encoding="utf-8")
    with pytest.raises(ValueError, match="not a directory"):
        coordinator.WorkspaceCoordinator(source, state_root=tmp_path / "state")
    assert not (tmp_path / "state").exists()


def test_environment_selects_persistent_state_root(workspace, tmp_path, monkeypatch):
    state_root = tmp_path / "configured-state"
    monkeypatch.setenv("WORKFLOW_SUPERVISOR_STATE_ROOT", str(state_root))
    instance = coordinator.WorkspaceCoordinator(workspace)
    lease = instance.acquire(mode="read", owner="one", task_id="task")
    reopened = coordinator.WorkspaceCoordinator(workspace)
    assert reopened.db_path == state_root / "coordinator.sqlite3"
    assert reopened.status()["active_leases"][0]["lease_id"] == lease["lease"]["lease_id"]


def test_equivalent_scopes_are_deduplicated_and_sibling_prefix_is_disjoint(service, workspace):
    instance, _ = service
    acquired = instance.acquire(
        mode="write",
        owner="one",
        task_id="task",
        paths=["src", "./src", str(workspace / "src")],
    )
    assert acquired["lease"]["paths"] == [os.path.normcase(str(workspace / "src"))]
    sibling = instance.acquire(mode="write", owner="two", task_id="other", paths=["src-other"])
    assert sibling["acquired"] is True
    child = instance.acquire(mode="read", owner="three", task_id="child", paths=["src/nested"])
    assert child["reason"] == "workspace_busy"


@pytest.mark.parametrize("previous_state", ["released", "expired"])
def test_active_job_is_deduplicated_then_retried_after_release_or_expiry(service, previous_state):
    instance, clock = service
    first = instance.acquire(mode="read", owner="one", task_id="task", job_key="same", ttl_seconds=5)
    duplicate = instance.acquire(mode="read", owner="two", task_id="retry", job_key="same")
    assert duplicate["reason"] == "job_active"
    assert len(instance.status()["active_leases"]) == 1
    if previous_state == "released":
        assert instance.release(first["lease"]["lease_id"], first["lease_token"])["ok"]
    else:
        clock[0] = 105.0
    assert instance.status()["jobs"][0]["status"] == previous_state
    retry = instance.acquire(mode="read", owner="two", task_id="retry", job_key="same")
    assert retry["acquired"] is True
    assert retry["lease"]["fence_token"] > first["lease"]["fence_token"]
    jobs = instance.status()["jobs"]
    assert len(jobs) == 1
    assert jobs[0]["status"] == "active"
    assert jobs[0]["owner"] == "two"
    assert jobs[0]["task_id"] == "retry"
    assert jobs[0]["completed_at"] is None
    assert jobs[0]["result_refs"] == []


def test_heartbeat_extends_deadline_and_expires_at_exact_boundary(service):
    instance, clock = service
    acquired = instance.acquire(mode="read", owner="one", task_id="task", job_key="job", ttl_seconds=5)
    clock[0] = 104.0
    heartbeat = instance.heartbeat(acquired["lease"]["lease_id"], acquired["lease_token"], ttl_seconds=10)
    assert heartbeat["expires_at"] == 114.0
    clock[0] = 105.0
    assert len(instance.status()["active_leases"]) == 1
    clock[0] = 114.0
    state = instance.status(include_events=10)
    assert state["active_leases"] == []
    assert state["jobs"][0]["status"] == "expired"
    assert [event["event_type"] for event in state["events"]] == [
        "lease.acquired",
        "lease.heartbeat",
        "lease.expired",
    ]


def test_invalid_heartbeat_ttl_does_not_extend_lease(service):
    instance, _ = service
    acquired = instance.acquire(mode="read", owner="one", task_id="task", ttl_seconds=5)
    with pytest.raises(ValueError, match="ttl_seconds must be positive"):
        instance.heartbeat(acquired["lease"]["lease_id"], acquired["lease_token"], ttl_seconds=0)
    assert instance.status()["active_leases"][0]["expires_at"] == 105.0


@pytest.mark.parametrize("lease_job_key", ["expected", None])
def test_finish_requires_matching_job_and_terminal_outcome(service, lease_job_key):
    instance, _ = service
    acquired = instance.acquire(mode="write", owner="one", task_id="task", job_key=lease_job_key)
    lease_id, token = acquired["lease"]["lease_id"], acquired["lease_token"]
    with pytest.raises(ValueError, match="outcome must be"):
        instance.finish(lease_id, token, outcome="active", job_key="expected")
    assert instance.finish(lease_id, token, outcome="succeeded", job_key="other") == {
        "ok": False,
        "reason": "job_key_mismatch",
    }
    state = instance.status(include_events=10)
    assert len(state["active_leases"]) == 1
    assert [job["status"] for job in state["jobs"]] == (["active"] if lease_job_key else [])
    assert [event["event_type"] for event in state["events"]] == ["lease.acquired"]


@pytest.mark.parametrize("method", ["release", "finish"])
@pytest.mark.parametrize("condition,reason", [("wrong_token", "token_mismatch"), ("expired", "lease_not_active")])
def test_rejected_completion_cannot_record_success(service, method, condition, reason):
    instance, clock = service
    acquired = instance.acquire(mode="write", owner="one", task_id="task", job_key="job", ttl_seconds=5)
    token = acquired["lease_token"]
    if condition == "expired":
        clock[0] = 105.0
    else:
        token = "incorrect"
    extra = {"outcome": "succeeded", "job_key": "job"} if method == "finish" else {}
    assert getattr(instance, method)(acquired["lease"]["lease_id"], token, **extra) == {
        "ok": False,
        "reason": reason,
    }
    state = instance.status()
    assert state["jobs"][0]["status"] == ("expired" if condition == "expired" else "active")
    assert state["jobs"][0]["completed_at"] is None


@pytest.mark.parametrize("outcome", ["succeeded", "invalid"])
def test_terminal_result_survives_reopen_and_matching_acquisition(service, outcome):
    instance, _ = service
    first = instance.acquire(mode="write", owner="one", task_id="task", job_key="job")
    finished = instance.finish(
        first["lease"]["lease_id"],
        first["lease_token"],
        outcome=outcome,
        job_key="job",
        result_refs=["fixture:result"],
    )
    reopened = coordinator.WorkspaceCoordinator(instance.workspace, state_root=instance.state_dir, clock=instance.clock)
    result = reopened.acquire(mode="write", owner="two", task_id="later", job_key="job")
    assert result["acquired"] is False
    assert result["reason"] == "job_already_terminal"
    assert result["job"] == finished["job"]
    assert result["job"]["reuse_requires_validation"] is (outcome == "succeeded")
    assert reopened.status()["active_leases"] == []
    assert reopened.status()["jobs"] == [finished["job"]]


def test_status_limits_events_in_chronological_order_and_isolates_workspaces(service, tmp_path):
    instance, clock = service
    acquired = instance.acquire(mode="read", owner="one", task_id="task")
    clock[0] = 101.0
    instance.heartbeat(acquired["lease"]["lease_id"], acquired["lease_token"])
    other_path = tmp_path / "other"
    other_path.mkdir()
    other = coordinator.WorkspaceCoordinator(other_path, state_root=instance.state_dir, clock=instance.clock)
    other.acquire(mode="read", owner="other", task_id="unrelated")
    clock[0] = 102.0
    instance.release(acquired["lease"]["lease_id"], acquired["lease_token"])
    assert instance.status()["events"] == []
    events = instance.status(include_events=2)["events"]
    assert [event["event_type"] for event in events] == ["lease.heartbeat", "lease.released"]
    assert [event["observed_at"] for event in events] == [101.0, 102.0]
    assert events[0]["sequence"] < events[1]["sequence"]
    assert len(other.status(include_events=10)["events"]) == 1


@pytest.mark.parametrize("command", ["heartbeat", "release", "finish"])
def test_cli_inactive_lease_returns_exit_three(cli, command):
    arguments = ["--lease-id", "missing", "--lease-token", "unused"]
    if command == "finish":
        arguments += ["--job-key", "job", "--outcome", "succeeded"]
    assert cli(command, *arguments, expected_code=3) == {"ok": False, "reason": "lease_not_active"}


def test_cli_release_status_and_nonpositive_event_limit(cli):
    acquired = cli("acquire", "--mode", "read", "--owner", "one", "--task-id", "task")
    assert acquired["acquired"] is True
    released = cli("release", "--lease-id", acquired["lease"]["lease_id"], "--lease-token", acquired["lease_token"])
    assert released["ok"] is True
    state = cli("status", "--events", "1")
    assert state["active_leases"] == []
    assert [event["event_type"] for event in state["events"]] == ["lease.released"]
    assert cli("status", "--events", "-1")["events"] == []


def test_cli_invalid_ttl_returns_structured_error_without_acquiring(cli):
    report = cli(
        "acquire",
        "--mode",
        "write",
        "--owner",
        "one",
        "--task-id",
        "task",
        "--ttl-seconds",
        "0",
        expected_code=4,
    )
    assert report["ok"] is False
    assert report["reason"] == "error"
    assert "ttl_seconds must be positive" in report["message"]
    assert cli("status")["active_leases"] == []


@pytest.mark.parametrize("command", ["status", "probe"])
def test_cli_missing_workspace_reports_error(tmp_path, capsys, command):
    code = coordinator.main(
        [
            command,
            "--workspace",
            str(tmp_path / "missing"),
            "--state-root",
            str(tmp_path / "state"),
            "--timeout-ms",
            "10000",
        ]
    )
    assert code == 4
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is False
    assert report["reason"] == ("probe_failed" if command == "probe" else "error")


def test_probe_reads_persistent_state_through_real_child_process(cli):
    acquired = cli("acquire", "--mode", "read", "--owner", "one", "--task-id", "task")
    report = cli("probe", "--timeout-ms", "10000")
    assert report["ok"] is True
    assert report["latency_ms"] >= 0
    assert [lease["lease_id"] for lease in report["child"]["active_leases"]] == [acquired["lease"]["lease_id"]]


def test_probe_child_applies_requested_delay(cli, monkeypatch):
    delays = []
    monkeypatch.setattr(coordinator.time, "sleep", delays.append)
    report = cli("_probe-child", "--probe-delay-ms", "25")
    assert report["active_leases"] == []
    assert delays == [0.025]


@pytest.mark.parametrize(
    "field,value",
    [
        ("scope", "other"),
        ("role", "Reviewer"),
        ("stage", "postrun"),
        ("source_revision", "def456"),
        ("input_hashes", ["changed"]),
        ("command_identity", "runner-v2"),
        ("output_root", "artifacts/other"),
    ],
)
def test_job_key_changes_with_each_execution_identity_field(workspace, field, value):
    parameters = {
        "workspace": workspace,
        "scope": "unit-fixture",
        "role": "Tester",
        "stage": "test",
        "source_revision": "abc123",
        "input_hashes": ["fixture"],
        "command_identity": "runner-v1",
        "output_root": "artifacts/result",
    }
    original = coordinator.deterministic_job_key(**parameters)
    changed = coordinator.deterministic_job_key(**(parameters | {field: value}))
    assert original["job_key"] != changed["job_key"]
