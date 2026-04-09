from __future__ import annotations

import json
from pathlib import Path

from tools.ci import check_task_acceptance_sync


def _write_plugin(root: Path, *, report: dict, package_name: str) -> Path:
    package_dir = root / package_name
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "plugin.py").write_text(
        f"""
class Plugin:
    capabilities = ["task_state.sync", "acceptance.index"]

    def sync_task_acceptance(self, *, repo_root):
        return {report!r}

    def build_acceptance_index(self, *, repo_root):
        return {{"markdown": "# Index\\n"}}


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


def test_check_task_acceptance_sync_passes(tmp_path: Path, capsys) -> None:
    config_path = _write_plugin(
        tmp_path,
        report={
            "tasks": [{"task_id": "20260410-01", "status": "done", "acceptance_ids": ["AC-20260410-01"]}],
            "acceptances": [{"acceptance_id": "AC-20260410-01"}],
            "errors": [],
            "warnings": [],
        },
        package_name="sync_plugin_pass",
    )

    result = check_task_acceptance_sync.main(["--plugin-config", str(config_path)])

    captured = capsys.readouterr()
    assert result == 0
    assert "synchronized" in captured.out


def test_check_task_acceptance_sync_fails_for_done_task_without_acceptance(
    tmp_path: Path, capsys
) -> None:
    config_path = _write_plugin(
        tmp_path,
        report={
            "tasks": [{"task_id": "20260410-01", "status": "done", "acceptance_ids": []}],
            "acceptances": [],
            "errors": [],
            "warnings": [],
        },
        package_name="sync_plugin_fail",
    )

    result = check_task_acceptance_sync.main(["--plugin-config", str(config_path)])

    captured = capsys.readouterr()
    assert result == 1
    assert "does not have an acceptance record" in captured.err
