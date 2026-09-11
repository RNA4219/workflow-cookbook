"""目的・制約・判断根拠を保持してagent-taskstateから文脈を再構成する。"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any, Protocol

from tools.context.progressive import document_paths, retrieve, within
from tools.evaluation.workflow import digest, integer, read_object, text
from tools.workflow_plugins.checkpoint import TaskstateCLI


class ContextStore(Protocol):
    def build_context(self, task_id: str) -> dict[str, Any]: ...


def encoded(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def snapshot_hash(snapshot: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(encoded(snapshot).encode("utf-8")).hexdigest()


def checked_snapshot(bundle: Mapping[str, Any], task_id: str, revision: int) -> dict[str, Any]:
    if not isinstance(bundle, Mapping):
        raise ValueError("bundle object required")
    if bundle.get("task_id") != task_id:
        raise ValueError("bundle task mismatch")
    for field in ("id", "generator_version", "generated_at"):
        text(bundle.get(field), "bundle " + field)
    if not isinstance(bundle.get("raw_included"), bool):
        raise ValueError("bundle raw_included flag required")
    snapshot = bundle.get("state_snapshot")
    if not isinstance(snapshot, dict):
        raise ValueError("complete state_snapshot required")
    task, state = snapshot.get("task"), snapshot.get("task_state")
    if not isinstance(task, dict) or not isinstance(state, dict):
        raise ValueError("snapshot task and task_state required")
    if task.get("id") != task_id or state.get("task_id") != task_id:
        raise ValueError("snapshot task mismatch")
    if integer(state.get("revision"), "state revision", minimum=1) != revision:
        raise ValueError("state revision mismatch; reload current task")
    text(task.get("goal"), "task goal")
    text(state.get("current_step"), "current_step")
    for field in ("constraints", "done_when", "artifact_refs", "evidence_refs"):
        value = state.get(field)
        if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
            raise ValueError("state " + field + " must be a text array")
    if not state["done_when"]:
        raise ValueError("nonempty done_when required")
    if not isinstance(state.get("context_policy"), dict) or "current_summary" not in state or "confidence" not in state:
        raise ValueError("complete task_state required")
    for field in ("decisions", "open_questions", "runs", "resolved_summaries"):
        if not isinstance(snapshot.get(field), list) or any(not isinstance(item, dict) for item in snapshot[field]):
            raise ValueError("snapshot " + field + " must be an object array")
    for field in ("decisions", "open_questions", "runs"):
        if any(item.get("task_id") != task_id for item in snapshot[field]):
            raise ValueError("snapshot contains another task's records")
    if not isinstance(snapshot.get("selected_raw"), dict):
        raise ValueError("selected_raw object required")
    if not isinstance(bundle.get("source_refs"), list) or not isinstance(bundle.get("diagnostics"), dict):
        raise ValueError("bundle provenance required")
    diagnostics = bundle["diagnostics"]
    if not isinstance(diagnostics.get("partial_bundle"), bool) or any(
        not isinstance(diagnostics.get(field), list) for field in ("missing_refs", "unsupported_refs")
    ):
        raise ValueError("complete resolver diagnostics required")
    return deepcopy(snapshot)


def required_documents(root: Path, snapshot: Mapping[str, Any]) -> list[dict[str, str]]:
    policy = snapshot["task_state"]["context_policy"].get("workflow_context", {})
    if not isinstance(policy, dict) or not isinstance(policy.get("required_documents", []), list):
        raise ValueError("workflow_context.required_documents array required")
    documents: list[dict[str, str]] = []
    seen: set[str] = set()
    for entry in policy.get("required_documents", []):
        if not isinstance(entry, dict):
            raise ValueError("required document must be an object")
        path = within(root, text(entry.get("path"), "required path"))
        relative = path.relative_to(root).as_posix()
        if relative in seen:
            raise ValueError("duplicate required document")
        seen.add(relative)
        raw = path.read_bytes()
        actual = "sha256:" + hashlib.sha256(raw).hexdigest()
        if entry.get("sha256") != actual:
            raise ValueError("required document hash mismatch: " + relative)
        documents.append({"path": relative, "sha256": actual, "context": f"\n## {relative}:1\n" + raw.decode("utf-8")})
    return documents


def assemble(
    repo_root: Path,
    query: str,
    budget_bytes: int,
    *,
    store: ContextStore,
    task_id: str,
    expected_revision: int,
    scope_paths: Sequence[str] = (".",),
    state_only: bool = False,
    target_documents: int = 3,
) -> dict[str, Any]:
    root = repo_root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("repo_root must be a directory")
    text(task_id, "task_id")
    text(query, "query")
    integer(expected_revision, "expected_revision", minimum=1)
    integer(budget_bytes, "budget_bytes", minimum=1)
    integer(target_documents, "target_documents", minimum=1)
    if not isinstance(state_only, bool):
        raise ValueError("state_only must be boolean")
    bundle = store.build_context(task_id)
    snapshot = checked_snapshot(bundle, task_id, expected_revision)
    provenance = {
        key: deepcopy(bundle[key])
        for key in ("id", "task_id", "generator_version", "generated_at", "raw_included", "source_refs", "diagnostics")
    }
    protected = (
        "\n## agent-taskstate recovery snapshot\n" + encoded({"bundle": provenance, "state_snapshot": snapshot}) + "\n"
    )
    documents = required_documents(root, snapshot)
    mandatory = protected + "".join(document["context"] for document in documents)
    mandatory_bytes = len(mandatory.encode("utf-8"))
    result: dict[str, Any] = {
        "schema_version": "1.0",
        "task_id": task_id,
        "state_revision": expected_revision,
        "bundle_id": bundle["id"],
        "snapshot_sha256": snapshot_hash(snapshot),
        "budget_bytes": budget_bytes,
        "required_bytes": mandatory_bytes,
        "protected_state_bytes": len(protected.encode("utf-8")),
        "required_documents": [{k: document[k] for k in ("path", "sha256")} for document in documents],
        "context": "",
        "context_bytes": 0,
        "ready_for_model": False,
        "source_refs": provenance["source_refs"],
        "diagnostics": provenance["diagnostics"],
        "state_only": state_only,
        "supplemental": None,
    }
    if mandatory_bytes > budget_bytes:
        return {**result, "status": "insufficient_budget", "missing_bytes": mandatory_bytes - budget_bytes}
    required_paths = {document["path"] for document in documents}
    extra: dict[str, Any] | None = None
    remaining = budget_bytes - mandatory_bytes
    if not state_only and remaining:
        scopes = [path for path in document_paths(root, scope_paths) if path not in required_paths]
        if scopes:
            extra = retrieve(root, query, remaining, scope_paths=scopes, target_documents=target_documents)
    if extra is not None:
        result["supplemental"] = {key: value for key, value in extra.items() if key != "context"}
    # Rebuild after I/O: goal/decision/question changes need not increment state.revision.
    after = store.build_context(task_id)
    if not isinstance(after, Mapping):
        raise ValueError("bundle object required")
    after_snapshot = after.get("state_snapshot")
    if not isinstance(after_snapshot, dict) or snapshot_hash(after_snapshot) != result["snapshot_sha256"]:
        return {**result, "status": "state_changed"}
    checked_snapshot(after, task_id, expected_revision)
    result["verified_bundle_id"] = after["id"]
    for document in documents:
        if digest(root / document["path"]) != document["sha256"]:
            raise ValueError("required document changed during retrieval: " + document["path"])
    diagnostics = [bundle["diagnostics"], after["diagnostics"]]
    unresolved = any(
        value.get("partial_bundle") or value.get("missing_refs") or value.get("unsupported_refs")
        for value in diagnostics
    )
    if unresolved or (not state_only and not documents and (extra is None or not extra["selected"])):
        return {**result, "status": "needs_evidence"}
    context = mandatory + (extra["context"] if extra is not None else "")
    result.update(
        {
            "status": "ready",
            "ready_for_model": True,
            "context": context,
            "context_bytes": len(context.encode("utf-8")),
            "context_sha256": "sha256:" + hashlib.sha256(context.encode("utf-8")).hexdigest(),
        }
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-client", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--expected-revision", type=int, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--budget-bytes", type=int, required=True)
    parser.add_argument("--scope", action="append")
    parser.add_argument("--state-only", action="store_true")
    parser.add_argument("--target-documents", type=int, default=3)
    args = parser.parse_args(argv)
    try:
        store = TaskstateCLI(**read_object(args.state_client))
        result = assemble(
            args.repo_root,
            args.query,
            args.budget_bytes,
            store=store,
            task_id=args.task_id,
            expected_revision=args.expected_revision,
            scope_paths=args.scope or ["."],
            state_only=args.state_only,
            target_documents=args.target_documents,
        )
        code = 0 if result["ready_for_model"] else 2
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        result, code = {"status": "invalid", "ready_for_model": False, "context": "", "error": str(exc)}, 1
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
