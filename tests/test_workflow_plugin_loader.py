from __future__ import annotations

import json
from pathlib import Path

from tools.workflow_plugins.plugin_config import load_workflow_plugin_specs
from tools.workflow_plugins.plugin_loader import (
    WorkflowPluginCapabilityError,
    WorkflowPluginLoadError,
    instantiate_workflow_plugins,
)


def _write_plugin_package(root: Path) -> None:
    package_dir = root / "fake_workflow_plugin"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "plugin.py").write_text(
        """
class Plugin:
    capabilities = ["task_state.sync"]

    def sync_task_acceptance(self, *, repo_root):
        return {"tasks": [], "acceptances": [], "errors": [], "warnings": []}


def create_plugin():
    return Plugin()
""".strip(),
        encoding="utf-8",
    )


def test_load_workflow_plugin_specs_and_instantiate(tmp_path: Path) -> None:
    _write_plugin_package(tmp_path)
    config_path = tmp_path / "plugins.json"
    config_path.write_text(
        json.dumps(
            {
                "workflow_plugins": [
                    {
                        "factory": "fake_workflow_plugin.plugin:create_plugin",
                        "python_paths": ["."],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    specs = load_workflow_plugin_specs(config_path)
    plugins = instantiate_workflow_plugins(specs, base_path=config_path.parent)

    assert len(plugins) == 1
    assert "task_state.sync" in plugins[0].capabilities


def test_instantiate_workflow_plugin_requires_capabilities(tmp_path: Path) -> None:
    package_dir = tmp_path / "bad_plugin"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "plugin.py").write_text(
        """
class Plugin:
    pass


def create_plugin():
    return Plugin()
""".strip(),
        encoding="utf-8",
    )

    config_path = tmp_path / "plugins.json"
    config_path.write_text(
        json.dumps(
            {
                "workflow_plugins": [
                    {
                        "factory": "bad_plugin.plugin:create_plugin",
                        "python_paths": ["."],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    specs = load_workflow_plugin_specs(config_path)
    try:
        instantiate_workflow_plugins(specs, base_path=config_path.parent)
    except WorkflowPluginLoadError as exc:
        assert "capabilities" in str(exc)
    else:
        raise AssertionError("WorkflowPluginLoadError was not raised")


def test_instantiate_workflow_plugin_requires_capability_method(tmp_path: Path) -> None:
    package_dir = tmp_path / "missing_method_plugin"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "plugin.py").write_text(
        """
class Plugin:
    capabilities = ["docs.resolve"]


def create_plugin():
    return Plugin()
""".strip(),
        encoding="utf-8",
    )

    config_path = tmp_path / "plugins.json"
    config_path.write_text(
        json.dumps(
            {
                "workflow_plugins": [
                    {
                        "factory": "missing_method_plugin.plugin:create_plugin",
                        "python_paths": ["."],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    specs = load_workflow_plugin_specs(config_path)
    try:
        instantiate_workflow_plugins(specs, base_path=config_path.parent)
    except WorkflowPluginCapabilityError as exc:
        assert "docs.resolve" in str(exc)
    else:
        raise AssertionError("WorkflowPluginCapabilityError was not raised")
