from __future__ import annotations

import json
from pathlib import Path

from tools.workflow_plugins.plugin_config import load_workflow_plugin_specs_from_path
from tools.workflow_plugins.runtime import WorkflowPluginRuntime


def _write_file(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def _write_fake_agent_taskstate_repo(root: Path) -> None:
    _write_file(root / "agent_taskstate_workflow_plugin" / "__init__.py", "")
    _write_file(
        root / "agent_taskstate_workflow_plugin" / "plugin.py",
        """
from dataclasses import dataclass

@dataclass(frozen=True)
class SyncResult:
    tasks: list
    acceptances: list
    errors: list
    warnings: list

@dataclass(frozen=True)
class IndexResult:
    markdown: str
    rows: list

class Plugin:
    capabilities = ("task_state.sync", "acceptance.index")

    def sync_task_acceptance(self, *, repo_root):
        return SyncResult(
            tasks=[{"task_id": "20260410-01", "status": "done", "acceptance_ids": ["AC-20260410-01"]}],
            acceptances=[{"acceptance_id": "AC-20260410-01"}],
            errors=[],
            warnings=[],
        )

    def build_acceptance_index(self, *, repo_root):
        return IndexResult(markdown="# Acceptance Index\\n", rows=[])

def create_plugin(**kwargs):
    return Plugin()
""",
    )


def _write_fake_memx_repo(root: Path) -> None:
    _write_file(root / "memx_resolver_workflow_plugin" / "__init__.py", "")
    _write_file(
        root / "memx_resolver_workflow_plugin" / "plugin.py",
        """
from dataclasses import dataclass

@dataclass(frozen=True)
class ResolveResult:
    required: list
    recommended: list
    errors: list
    warnings: list

class Plugin:
    capabilities = ("docs.resolve", "docs.ack", "docs.stale_check")

    def resolve_docs(self, *, repo_root, task_id, intent_id=None):
        return ResolveResult(
            required=[{"doc_id": "README.md", "version": "v1"}],
            recommended=[],
            errors=[],
            warnings=[],
        )

    def ack_docs(self, *, repo_root, task_id, doc_ids, reader):
        return {"receipts": [{"task_id": task_id, "doc_ids": doc_ids, "reader": reader}]}

    def stale_check(self, *, repo_root, task_id):
        return {"task_id": task_id, "stale": []}

def create_plugin(**kwargs):
    return Plugin()
""",
    )


def test_sample_plugin_config_can_drive_runtime_with_fake_sibling_repos(tmp_path: Path) -> None:
    sample_path = Path(__file__).resolve().parents[1] / "examples" / "workflow_plugins.cross_repo.sample.json"
    payload = json.loads(sample_path.read_text(encoding="utf-8"))

    fake_root = tmp_path / "workspace"
    cookbook_root = fake_root / "workflow-cookbook"
    agent_taskstate_root = fake_root / "agent-taskstate"
    memx_root = fake_root / "memx-resolver"
    (cookbook_root / "examples").mkdir(parents=True)
    _write_fake_agent_taskstate_repo(agent_taskstate_root)
    _write_fake_memx_repo(memx_root)

    config_path = cookbook_root / "examples" / "workflow_plugins.cross_repo.sample.json"
    config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    specs = load_workflow_plugin_specs_from_path(config_path)
    assert len(specs) == 2

    runtime = WorkflowPluginRuntime.from_config(config_path)
    sync_result = runtime.invoke_first("task_state.sync", repo_root=cookbook_root)
    resolve_result = runtime.invoke_first("docs.resolve", repo_root=cookbook_root, task_id="20260410-01")

    assert getattr(sync_result, "tasks")[0]["task_id"] == "20260410-01"
    assert getattr(resolve_result, "required")[0]["doc_id"] == "README.md"
