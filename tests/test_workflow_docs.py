from __future__ import annotations

import json
from pathlib import Path

from tools.context import workflow_docs


def _write_docs_plugin(root: Path) -> Path:
    package_dir = root / "docs_plugin"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "plugin.py").write_text(
        """
class Plugin:
    capabilities = ["docs.resolve", "docs.ack", "docs.stale_check"]

    def resolve_docs(self, *, repo_root, task_id, intent_id=None):
        return {"required": [{"doc_id": "README.md", "version": "v1"}], "recommended": []}

    def ack_docs(self, *, repo_root, task_id, doc_ids, reader):
        return {"receipts": [{"task_id": task_id, "doc_ids": doc_ids, "reader": reader}]}

    def stale_check(self, *, repo_root, task_id):
        return {"task_id": task_id, "stale": [{"doc_id": "README.md"}]}


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
                        "factory": "docs_plugin.plugin:create_plugin",
                        "python_paths": ["."],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return config_path


def test_workflow_docs_resolve(tmp_path: Path, capsys) -> None:
    config_path = _write_docs_plugin(tmp_path)
    result = workflow_docs.main(
        ["--plugin-config", str(config_path), "resolve", "--task-id", "20260410-01"]
    )
    captured = capsys.readouterr()
    assert result == 0
    payload = json.loads(captured.out)
    assert payload["required"][0]["doc_id"] == "README.md"


def test_workflow_docs_stale_check_mode_returns_nonzero(tmp_path: Path, capsys) -> None:
    config_path = _write_docs_plugin(tmp_path)
    result = workflow_docs.main(
        ["--plugin-config", str(config_path), "stale", "--task-id", "20260410-01", "--check"]
    )
    captured = capsys.readouterr()
    assert result == 1
    payload = json.loads(captured.out)
    assert payload["stale"][0]["doc_id"] == "README.md"
