from __future__ import annotations

import json
import time
from pathlib import Path

from tools.workflow_plugins.errors import WorkflowPluginCapabilityError
from tools.workflow_plugins.errors import WorkflowPluginTimeoutError
from tools.workflow_plugins.runtime import PluginPolicy, WorkflowPluginRuntime


def _write_plugin_package(root: Path, *, capabilities: list[str], package_name: str = "runtime_plugin") -> Path:
    package_dir = root / package_name
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    methods: list[str] = []
    if "docs.resolve" in capabilities:
        methods.append(
            """
    def resolve_docs(self, *, repo_root, task_id, intent_id=None):
        return {"required": [], "recommended": [], "errors": [], "warnings": []}
""".rstrip()
        )
    if "docs.ack" in capabilities:
        methods.append(
            """
    def ack_docs(self, *, repo_root, task_id, doc_ids, reader):
        return {"receipts": []}
""".rstrip()
        )
    if "acceptance.index" in capabilities:
        methods.append(
            """
    def build_acceptance_index(self, *, repo_root):
        return {"markdown": "# Index\\n", "rows": []}
""".rstrip()
        )
    (package_dir / "plugin.py").write_text(
        f"""
class Plugin:
    capabilities = {capabilities!r}
{''.join(methods)}


def create_plugin():
    return Plugin()
""".strip(),
        encoding="utf-8",
    )
    config_path = root / "plugins.json"
    config_path.write_text(
        json.dumps(
            {
                "workflow_plugins": [
                    {
                        "factory": f"{package_name}.plugin:create_plugin",
                        "python_paths": ["."],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return config_path


def test_workflow_plugin_runtime_filters_capability(tmp_path: Path) -> None:
    config_path = _write_plugin_package(tmp_path, capabilities=["docs.resolve", "docs.ack"])

    runtime = WorkflowPluginRuntime.from_config(config_path)

    plugins = list(runtime.iter_capability("docs.resolve"))
    assert len(plugins) == 1
    assert "docs.resolve" in plugins[0].capabilities


def test_workflow_plugin_runtime_first_capability_raises_for_missing(tmp_path: Path) -> None:
    config_path = _write_plugin_package(tmp_path, capabilities=["acceptance.index"], package_name="missing_plugin")

    runtime = WorkflowPluginRuntime.from_config(config_path)

    try:
        runtime.first_capability("docs.stale_check")
    except WorkflowPluginCapabilityError as exc:
        assert "docs.stale_check" in str(exc)
    else:
        raise AssertionError("WorkflowPluginCapabilityError was not raised")


def test_workflow_plugin_runtime_invoke_first_calls_capability_method(tmp_path: Path) -> None:
    config_path = _write_plugin_package(tmp_path, capabilities=["docs.resolve"], package_name="invoke_plugin")

    runtime = WorkflowPluginRuntime.from_config(config_path)

    result = runtime.invoke_first("docs.resolve", repo_root=tmp_path, task_id="20260410-01")

    assert result["required"] == []


def test_workflow_plugin_runtime_invoke_all_calls_each_plugin(tmp_path: Path) -> None:
    _write_plugin_package(tmp_path / "first", capabilities=["docs.resolve"], package_name="first_plugin")
    second_root = tmp_path / "second"
    second_root.mkdir()
    _write_plugin_package(second_root, capabilities=["docs.resolve"], package_name="second_plugin")
    config_path = tmp_path / "plugins.json"
    config_path.write_text(
        json.dumps(
            {
                "workflow_plugins": [
                    {"factory": "first_plugin.plugin:create_plugin", "python_paths": ["first"]},
                    {"factory": "second_plugin.plugin:create_plugin", "python_paths": ["second"]},
                ]
            }
        ),
        encoding="utf-8",
    )

    runtime = WorkflowPluginRuntime.from_config(config_path)

    results = runtime.invoke_all("docs.resolve", repo_root=tmp_path, task_id="20260410-01")

    assert len(results) == 2


class _SlowPlugin:
    capabilities = ["docs.resolve"]

    def resolve_docs(self, *, repo_root: Path, task_id: str, intent_id: str | None = None) -> dict[str, object]:
        time.sleep(0.2)
        return {"required": [], "recommended": [], "errors": [], "warnings": []}


class _TracingPlugin:
    capabilities = ["docs.resolve"]

    def resolve_docs(self, *, repo_root: Path, task_id: str, intent_id: str | None = None) -> dict[str, object]:
        return {"required": [{"doc_id": "README.md"}], "recommended": [], "errors": [], "warnings": []}


def test_workflow_plugin_runtime_thread_timeout_returns_promptly(tmp_path: Path) -> None:
    runtime = WorkflowPluginRuntime(
        [_SlowPlugin()],
        default_policy=PluginPolicy(timeout_seconds=0.01, trace_enabled=True),
    )

    started = time.perf_counter()
    try:
        runtime.invoke_first("docs.resolve", repo_root=tmp_path, task_id="20260410-01")
    except WorkflowPluginTimeoutError:
        pass
    else:
        raise AssertionError("WorkflowPluginTimeoutError was not raised")

    assert time.perf_counter() - started < 0.15
    assert runtime.traces[0].timed_out is True
    assert runtime.traces[0].isolation_mode == "thread"


def test_workflow_plugin_runtime_can_export_trace_json(tmp_path: Path) -> None:
    trace_path = tmp_path / "traces" / "workflow-plugin-traces.json"
    runtime = WorkflowPluginRuntime(
        [_TracingPlugin()],
        default_policy=PluginPolicy(timeout_seconds=1.0, trace_enabled=True),
    )

    result = runtime.invoke_first("docs.resolve", repo_root=tmp_path, task_id="20260410-01")
    runtime.write_traces_json(trace_path)

    assert result["required"][0]["doc_id"] == "README.md"
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    assert payload[0]["success"] is True
    assert payload[0]["result_summary"] == "dict(keys=['errors', 'recommended', 'required', 'warnings'])"


def test_workflow_plugin_runtime_can_export_trace_evidence_jsonl(tmp_path: Path) -> None:
    evidence_path = tmp_path / "evidence" / "plugin-traces.jsonl"
    runtime = WorkflowPluginRuntime(
        [_TracingPlugin()],
        default_policy=PluginPolicy(timeout_seconds=1.0, trace_enabled=True),
    )

    runtime.invoke_first("docs.resolve", repo_root=tmp_path, task_id="20260410-01")
    runtime.write_trace_evidence_jsonl(
        evidence_path,
        task_seed_id="TS-001",
        base_commit="abc1234",
        head_commit="def5678",
        actor="codex",
    )

    payload = json.loads(evidence_path.read_text(encoding="utf-8").splitlines()[0])
    assert payload["kind"] == "Evidence"
    assert payload["id"] == "EV-WORKFLOW-PLUGIN-001"
    assert payload["taskSeedId"] == "TS-001"
    assert payload["tools"] == ["WorkflowPluginRuntime", "docs.resolve"]
    assert payload["policyVerdict"] == "approved"
    assert payload["model"]["name"] == "workflow-plugin-runtime"
