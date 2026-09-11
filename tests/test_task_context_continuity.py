import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from tools.context.taskstate import assemble, main
from tools.evaluation.workflow import digest
from tools.workflow_plugins.checkpoint import TaskstateCLI


class Store:
    def __init__(self, bundle):
        self.bundle = bundle
        self.calls = 0
        self.change = None

    def build_context(self, task_id):
        self.calls += 1
        if self.change and self.calls == 2:
            self.change(self.bundle)
        return deepcopy(self.bundle)


@pytest.fixture
def setup(tmp_path):
    doc = tmp_path / "policy.md"
    doc.write_text("# 作業の正本\n目的を維持し、承認なしに公開しない。\n" + "必須根拠\n" * 100, encoding="utf-8")
    bundle = {
        "id": "bundle-1",
        "task_id": "task",
        "generator_version": "1.1.0",
        "generated_at": "2026-09-11T00:00:00Z",
        "raw_included": False,
        "source_refs": ["agent-taskstate:task:local:task"],
        "diagnostics": {"partial_bundle": False, "missing_refs": [], "unsupported_refs": []},
        "state_snapshot": {
            "task": {"id": "task", "goal": "ドリフトを防いで改修する", "status": "in_progress"},
            "task_state": {
                "task_id": "task",
                "revision": 7,
                "current_step": "検証",
                "current_summary": "必読を保持する",
                "constraints": ["公開は未承認", "根拠を削らない"],
                "done_when": ["必要状態の欠落ゼロ"],
                "artifact_refs": [],
                "evidence_refs": [],
                "confidence": "medium",
                "context_policy": {
                    "workflow_checkpoint": {"status": "running"},
                    "workflow_context": {"required_documents": [{"path": "policy.md", "sha256": digest(doc)}]},
                },
            },
            "decisions": [
                {
                    "id": "d1",
                    "task_id": "task",
                    "summary": "削減率を主指標にしない",
                    "rationale": "ドリフト防止が優先",
                    "status": "accepted",
                },
                {"id": "d2", "task_id": "task", "summary": "公開する案", "status": "proposed"},
            ],
            "open_questions": [{"id": "q1", "task_id": "task", "question": "実利用の成功率は未測定", "status": "open"}],
            "runs": [{"id": "r1", "task_id": "task", "summary": "過去pilotはinvalid_control"}],
            "resolved_summaries": [],
            "selected_raw": {},
            "purpose": "continue_work",
            "rebuild_level": "L2",
        },
    }
    return tmp_path, Store(bundle)


def run(setup, **kwargs):
    root, store = setup
    return assemble(
        root, "作業 検証", kwargs.pop("budget_bytes", 30000), store=store, task_id="task", expected_revision=7, **kwargs
    )


def test_preserves_entire_snapshot_and_complete_required_source(setup):
    root, store = setup
    before = deepcopy(store.bundle["state_snapshot"])
    result = run(setup)
    protected = result["context"].split("\n## agent-taskstate recovery snapshot\n")[1].split("\n## policy.md:1\n")[0]
    assert json.loads(protected)["state_snapshot"] == before
    assert (root / "policy.md").read_bytes().decode("utf-8") in result["context"]
    assert result["ready_for_model"] and result["status"] == "ready"
    assert result["context_bytes"] == len(result["context"].encode("utf-8"))
    assert store.bundle["state_snapshot"] == before and store.calls == 2


def test_required_boundary_retains_every_byte_and_does_not_trim(setup):
    full = run(setup)
    exact = run(setup, budget_bytes=full["required_bytes"])
    assert exact["ready_for_model"] and exact["context"] == full["context"]
    insufficient = run(setup, budget_bytes=full["required_bytes"] - 1)
    assert insufficient["status"] == "insufficient_budget" and insufficient["missing_bytes"] == 1
    assert insufficient["context"] == "" and not insufficient["ready_for_model"]


def test_long_supplement_cannot_evict_state_or_required_source(setup):
    root, _ = setup
    mandatory = run(setup)["context"]
    (root / "large.md").write_text("検証 作業\n" * 5000, encoding="utf-8")
    result = run(setup, budget_bytes=len(mandatory.encode("utf-8")) + 120)
    assert result["context"].startswith(mandatory)
    assert result["context_bytes"] <= result["budget_bytes"]
    assert result["supplemental"]["selected"][0]["excerpt"]


@pytest.mark.parametrize("field", ["goal", "constraint", "decision", "question", "revision"])
def test_concurrent_semantic_state_change_stops_context_delivery(setup, field):
    _, store = setup

    def change(bundle):
        snapshot = bundle["state_snapshot"]
        if field == "goal":
            snapshot["task"]["goal"] = "別の目的"
        elif field == "constraint":
            snapshot["task_state"]["constraints"].append("新しいユーザー指示")
        elif field == "decision":
            snapshot["decisions"][0]["status"] = "rejected"
        elif field == "question":
            snapshot["open_questions"].clear()
        else:
            snapshot["task_state"]["revision"] += 1

    store.change = change
    result = run(setup)
    assert result["status"] == "state_changed" and result["context"] == ""
    assert not result["ready_for_model"]


@pytest.mark.parametrize(
    "diagnostic,value",
    [
        ("partial_bundle", True),
        ("missing_refs", ["memx:evidence:local:missing"]),
        ("unsupported_refs", ["tracker:issue:jira:X-1"]),
    ],
)
def test_unresolved_sources_cannot_be_hidden_by_complete_local_documents(setup, diagnostic, value):
    setup[1].bundle["diagnostics"][diagnostic] = value
    result = run(setup)
    assert result["status"] == "needs_evidence" and not result["ready_for_model"]
    assert result["context"] == ""


@pytest.mark.parametrize(
    "damage",
    [
        "bundle_task",
        "snapshot_task",
        "revision",
        "goal",
        "done_when",
        "missing_decisions",
        "cross_task",
        "missing_raw",
        "provenance",
    ],
)
def test_incomplete_or_wrong_identity_is_not_a_usable_context(setup, damage):
    bundle = setup[1].bundle
    snapshot = bundle["state_snapshot"]
    if damage == "bundle_task":
        bundle["task_id"] = "other"
    elif damage == "snapshot_task":
        snapshot["task"]["id"] = "other"
    elif damage == "revision":
        snapshot["task_state"]["revision"] = 6
    elif damage == "goal":
        snapshot["task"]["goal"] = ""
    elif damage == "done_when":
        snapshot["task_state"]["done_when"] = []
    elif damage == "missing_decisions":
        snapshot.pop("decisions")
    elif damage == "cross_task":
        snapshot["decisions"][0]["task_id"] = "other"
    elif damage == "missing_raw":
        snapshot.pop("selected_raw")
    else:
        bundle.pop("diagnostics")
    with pytest.raises(ValueError):
        run(setup)


@pytest.mark.parametrize("damage", ["missing", "hash", "outside", "duplicate", "during_io"])
def test_required_sources_cannot_be_removed_replaced_or_substituted(setup, damage):
    root, store = setup
    entries = store.bundle["state_snapshot"]["task_state"]["context_policy"]["workflow_context"]["required_documents"]
    if damage == "missing":
        (root / "policy.md").unlink()
    elif damage == "hash":
        (root / "policy.md").write_text("変わった原文", encoding="utf-8")
    elif damage == "outside":
        entries[0]["path"] = "../outside.md"
    elif damage == "duplicate":
        entries.append(deepcopy(entries[0]))
    else:
        store.change = lambda _: (root / "policy.md").write_text("取得中に更新", encoding="utf-8")
    with pytest.raises((OSError, ValueError)):
        run(setup)


def test_state_only_requires_explicit_choice_and_never_bypasses_required_sources(setup):
    root, store = setup
    store.bundle["state_snapshot"]["task_state"]["context_policy"]["workflow_context"]["required_documents"] = []
    result = run(setup, scope_paths=[])
    assert result["status"] == "needs_evidence"
    state_only = run(setup, state_only=True)
    assert state_only["ready_for_model"] and "ドリフトを防いで改修する" in state_only["context"]
    assert "作業の正本" not in state_only["context"]


def test_cli_reports_ready_budget_failure_and_invalid_input(setup, monkeypatch, capsys):
    root, store = setup
    config = root / "client.json"
    config.write_text("{}", encoding="utf-8")
    monkeypatch.setattr("tools.context.taskstate.TaskstateCLI", lambda **_: store)
    args = [
        "--state-client",
        str(config),
        "--task-id",
        "task",
        "--expected-revision",
        "7",
        "--repo-root",
        str(root),
        "--query",
        "作業",
        "--budget-bytes",
    ]
    assert main(args + ["30000"]) == 0
    assert json.loads(capsys.readouterr().out)["ready_for_model"]
    assert main(args + ["1"]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "insufficient_budget"
    store.bundle["state_snapshot"]["task_state"]["revision"] = 8
    assert main(args + ["30000"]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "invalid"


def test_real_taskstate_cli_reopen_keeps_goal_decisions_questions_and_required_source(tmp_path):
    sibling_src = Path(__file__).resolve().parents[2] / "agent-taskstate" / "src"
    if not (sibling_src / "agent_taskstate").is_dir():
        pytest.skip("optional sibling agent-taskstate integration")
    client = TaskstateCLI(
        [sys.executable, "-B", "-m", "agent_taskstate.cli"], cwd=sibling_src, db=tmp_path / "taskstate.sqlite3"
    )
    client._call(["init"])
    client._call(
        [
            "task",
            "create",
            "--json",
            json.dumps({"id": "task", "kind": "feature", "title": "継続性", "goal": "目的を維持する"}),
        ]
    )
    source = tmp_path / "acceptance.md"
    source.write_text("# 受入\n目的・制約・判断根拠を保持する。", encoding="utf-8")
    state = {
        "current_step": "検証",
        "constraints": ["削減を主目標にしない"],
        "done_when": ["必須状態を復元できる"],
        "context_policy": {
            "workflow_context": {"required_documents": [{"path": "acceptance.md", "sha256": digest(source)}]}
        },
    }
    client._call(["state", "put", "--task", "task", "--json", json.dumps(state)])
    client._call(
        [
            "decision",
            "add",
            "--task",
            "task",
            "--json",
            json.dumps({"summary": "ドリフト防止を優先", "rationale": "ユーザーの判断", "status": "accepted"}),
        ]
    )
    client._call(
        [
            "question",
            "add",
            "--task",
            "task",
            "--json",
            json.dumps({"question": "実モデル長期効果は未確認", "priority": "high"}),
        ]
    )
    before = client.get("task")
    first = assemble(
        tmp_path, "受入", 50000, store=client, task_id="task", expected_revision=1, scope_paths=["acceptance.md"]
    )
    reopened = TaskstateCLI(client.command, cwd=sibling_src, db=tmp_path / "taskstate.sqlite3")
    second = assemble(
        tmp_path, "受入", 50000, store=reopened, task_id="task", expected_revision=1, scope_paths=["acceptance.md"]
    )
    assert first["ready_for_model"] and second["ready_for_model"]
    assert first["snapshot_sha256"] == second["snapshot_sha256"]
    assert first["bundle_id"] != second["bundle_id"]
    for phrase in (
        "目的を維持する",
        "削減を主目標にしない",
        "ドリフト防止を優先",
        "実モデル長期効果は未確認",
        "目的・制約・判断根拠を保持する",
    ):
        assert phrase in second["context"]
    assert reopened.get("task") == before
    reopened.patch("task", 1, {"current_step": "ユーザー変更を反映", "constraints": ["新しい制約を守る"]})
    with pytest.raises(ValueError, match="revision"):
        assemble(tmp_path, "受入", 50000, store=reopened, task_id="task", expected_revision=1)
    refreshed = assemble(
        tmp_path, "受入", 50000, store=reopened, task_id="task", expected_revision=2, scope_paths=["acceptance.md"]
    )
    assert refreshed["ready_for_model"] and "新しい制約を守る" in refreshed["context"]
    assert refreshed["snapshot_sha256"] != first["snapshot_sha256"]
