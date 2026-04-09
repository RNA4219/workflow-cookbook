from __future__ import annotations

import json
from pathlib import Path

from tools.workflow_plugins import validate_workflow_plugin_config


def _write_plugin(root: Path, *, capabilities: list[str]) -> Path:
    package_dir = root / "validate_plugin"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "plugin.py").write_text(
        f"""
class Plugin:
    capabilities = {capabilities!r}

    def sync_task_acceptance(self, *, repo_root):
        return {{"tasks": [], "acceptances": [], "errors": [], "warnings": []}}


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
                        "factory": "validate_plugin.plugin:create_plugin",
                        "python_paths": ["."],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return config_path


def test_validate_workflow_plugin_config_shape_only(tmp_path: Path, capsys) -> None:
    config_path = _write_plugin(tmp_path, capabilities=["task_state.sync"])

    result = validate_workflow_plugin_config.main(["--plugin-config", str(config_path), "--emit-json"])

    captured = capsys.readouterr()
    assert result == 0
    payload = json.loads(captured.out)
    assert payload["valid"] is True
    assert payload["specs"][0]["factory"] == "validate_plugin.plugin:create_plugin"


def test_validate_workflow_plugin_config_instantiates_plugin(tmp_path: Path, capsys) -> None:
    config_path = _write_plugin(tmp_path, capabilities=["task_state.sync"])

    result = validate_workflow_plugin_config.main(
        ["--plugin-config", str(config_path), "--instantiate", "--emit-json"]
    )

    captured = capsys.readouterr()
    assert result == 0
    payload = json.loads(captured.out)
    assert payload["plugins"][0]["capabilities"] == ["task_state.sync"]
