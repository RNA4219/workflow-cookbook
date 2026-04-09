from __future__ import annotations

import json
from pathlib import Path

from tools.workflow_plugins.errors import WorkflowPluginCapabilityError
from tools.workflow_plugins.runtime import WorkflowPluginRuntime


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
