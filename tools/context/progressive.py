"""原文の鮮度と実byte予算に基づく段階取得。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from tools.codemap.source_freshness import review_matches
from tools.evaluation.workflow import integer, text
from tools.workflow_plugins.interfaces import coerce_docs_resolve_result
from tools.workflow_plugins.runtime import WorkflowPluginRuntime

IGNORED = {".git", ".venv", "venv", "node_modules", "__pycache__", ".workflow-cache", "birdseye"}


def within(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"path outside repo: {value}")
    return path


def document_paths(root: Path, scopes: Sequence[str]) -> list[str]:
    found: set[str] = set()
    for scope in scopes:
        path = within(root, scope)
        if path.is_file():
            found.add(path.relative_to(root).as_posix())
        elif path.is_dir():
            for directory, folders, files in os.walk(path, followlinks=False):
                folders[:] = sorted(name for name in folders if name not in IGNORED)
                for name in sorted(files):
                    if name.endswith(".md"):
                        candidate = (Path(directory) / name).resolve()
                        if candidate.is_relative_to(root):
                            found.add(candidate.relative_to(root).as_posix())
        else:
            raise ValueError(f"scope does not exist: {scope}")
    return sorted(found)


def retrieve(
    repo_root: Path,
    query: str,
    budget_bytes: int,
    *,
    max_hops: int = 2,
    target_documents: int = 3,
    scope_paths: Sequence[str] = (".",),
    required_paths: Sequence[str] = (),
    runtime: WorkflowPluginRuntime | None = None,
    task_id: str | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    root = repo_root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("repo_root must be a directory")
    text(query, "query")
    integer(budget_bytes, "budget_bytes", minimum=1)
    integer(target_documents, "target_documents", minimum=1)
    if isinstance(max_hops, bool) or max_hops not in (0, 1, 2):
        raise ValueError("max_hops must be 0, 1 or 2")
    words = re.findall(r"[^\W_]+", query.casefold())
    if not words:
        raise ValueError("query needs searchable words")
    warnings: list[str] = []
    io_bytes = 0
    bodies: dict[str, bytes] = {}

    def read(path: Path) -> bytes:
        nonlocal io_bytes
        data = path.read_bytes()
        io_bytes += len(data)
        return data

    def body(path: str) -> bytes:
        if path not in bodies:
            bodies[path] = read(within(root, path))
        return bodies[path]

    def score(value: str) -> int:
        return sum(word in value.casefold() for word in words)

    required = list(required_paths)
    if runtime is not None:
        text(task_id, "task_id")
        resolved = coerce_docs_resolve_result(
            runtime.invoke_first(
                "docs.resolve",
                repo_root=root,
                task_id=task_id,
                intent_id=None,
            )
        )
        if resolved.errors:
            raise ValueError("docs.resolve: " + "; ".join(resolved.errors))
        warnings.extend(resolved.warnings)
        required.extend(text(entry.get("path"), "required path") for entry in resolved.required)
    required = list(dict.fromkeys(within(root, path).relative_to(root).as_posix() for path in required))
    candidates = document_paths(root, scope_paths)
    candidates = sorted(set(candidates) | set(required))
    chunks: list[str] = []
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    consumed = 0
    budget_limited = False

    def result(status: str, **extra: Any) -> dict[str, Any]:
        context = "".join(chunks)
        return {
            "schema_version": "1.0",
            "status": status,
            "query": query,
            "context": context,
            "selected": selected,
            "context_bytes": len(context.encode("utf-8")),
            "budget_bytes": budget_bytes,
            "source_io_bytes": io_bytes,
            "source_io_scope": "retriever_only",
            "warnings": warnings,
            "elapsed_ms": (time.perf_counter() - started) * 1000,
            "token_count": None,
            **extra,
        }

    required_bytes = sum(len(f"\n## {path}:1\n".encode("utf-8")) + len(body(path)) for path in required)
    if required_bytes > budget_bytes:
        return result("insufficient_budget", required_bytes=required_bytes)

    def add(path: str, via: str, hop: int, *, mandatory: bool = False) -> bool:
        nonlocal consumed, budget_limited
        if path in selected_ids:
            return False
        raw = body(path)
        content = raw.decode("utf-8")
        line = 1
        excerpt = False
        prefix = f"\n## {path}:1\n"
        remaining = budget_bytes - consumed
        if len(prefix.encode("utf-8")) + len(raw) > remaining and not mandatory:
            lines = content.splitlines(keepends=True)
            offset = next((i for i, value in enumerate(lines) if score(value)), 0)
            line = offset + 1
            prefix = f"\n## {path}:{line}\n"
            allowance = remaining - len(prefix.encode("utf-8"))
            if allowance <= 0:
                budget_limited = True
                return False
            content = "".join(lines[offset:]).encode("utf-8")[:allowance].decode("utf-8", errors="ignore")
            if not content:
                budget_limited = True
                return False
            excerpt = True
            budget_limited = True
        rendered = prefix + content
        consumed += len(rendered.encode("utf-8"))
        chunks.append(rendered)
        selected_ids.add(path)
        selected.append(
            {
                "path": path,
                "source_sha256": hashlib.sha256(raw).hexdigest(),
                "via": via,
                "hop": hop,
                "start_line": line,
                "excerpt": excerpt,
            }
        )
        return True

    for path in required:
        add(path, "required", 0, mandatory=True)

    caps: dict[str, dict[str, Any]] = {}
    index_path = root / "docs/birdseye/index.json"
    if len(selected) < target_documents:
        try:
            index = json.loads(read(index_path))
            if not isinstance(index, dict) or not isinstance(index.get("nodes"), dict):
                raise ValueError("index must have file-map nodes")
            for path, entry in index["nodes"].items():
                if path not in candidates or not isinstance(entry, dict):
                    continue
                try:
                    capsule = json.loads(read(within(root, text(entry.get("caps"), "caps path"))))
                    if not isinstance(capsule, dict) or capsule.get("id") != path:
                        raise ValueError("caps id mismatch")
                    caps[path] = capsule
                except (OSError, ValueError, TypeError) as exc:
                    warnings.append(f"caps unavailable: {path}: {exc}")
        except (OSError, ValueError, TypeError) as exc:
            warnings.append(f"index unavailable; using source search: {exc}")

    checked: dict[str, bool] = {}

    def fresh(path: str) -> bool:
        if path not in checked:
            capsule = caps[path]
            current = hashlib.sha256(body(path)).hexdigest()
            checked[path] = capsule.get("source_sha256") == current and review_matches(capsule, current)
            if not checked[path]:
                warnings.append(f"stale or unreviewed capsule: {path}")
        return checked[path]

    ranked = sorted(
        caps,
        key=lambda path: (-score(path + " " + json.dumps(caps[path].get("summary", ""), ensure_ascii=False)), path),
    )
    frontier: list[str] = []
    for path in ranked:
        if len(selected) >= target_documents:
            break
        if score(path + " " + json.dumps(caps[path].get("summary", ""), ensure_ascii=False)) and fresh(path):
            if add(path, "birdseye", 0):
                frontier.append(path)
    visited = set(frontier)
    for hop in range(1, max_hops + 1):
        next_frontier = []
        for parent in frontier:
            dependencies = caps[parent].get("deps_out", [])
            if not isinstance(dependencies, list):
                warnings.append(f"invalid dependencies: {parent}")
                continue
            for path in dependencies:
                if not isinstance(path, str) or path in visited or path not in caps:
                    continue
                visited.add(path)
                if len(selected) < target_documents and fresh(path) and add(path, "dependency", hop):
                    next_frontier.append(path)
        frontier = next_frontier
        if not frontier or len(selected) >= target_documents:
            break

    if len(selected) < target_documents:
        ranked_sources = sorted(
            ((score(path + " " + body(path).decode("utf-8")), path) for path in candidates if path not in selected_ids),
            key=lambda item: (-item[0], item[1]),
        )
        for relevance, path in ranked_sources:
            if len(selected) >= target_documents:
                break
            if relevance:
                add(path, "source_search", 0)
    status = (
        "not_found"
        if not selected
        else ("partial" if budget_limited or len(selected) < target_documents else "selected")
    )
    return result(status)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--budget-bytes", type=int, required=True)
    parser.add_argument("--max-hops", type=int, default=2)
    parser.add_argument("--target-documents", type=int, default=3)
    parser.add_argument("--scope", action="append")
    parser.add_argument("--required", action="append", default=[])
    parser.add_argument("--plugin-config", type=Path)
    parser.add_argument("--task-id")
    args = parser.parse_args(argv)
    try:
        runtime = WorkflowPluginRuntime.from_config(args.plugin_config) if args.plugin_config else None
        output = retrieve(
            args.repo_root,
            args.query,
            args.budget_bytes,
            max_hops=args.max_hops,
            target_documents=args.target_documents,
            scope_paths=args.scope or ["."],
            required_paths=args.required,
            runtime=runtime,
            task_id=args.task_id,
        )
        code = 2 if output["status"] == "insufficient_budget" else 0
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        output, code = {"status": "invalid", "error": str(exc)}, 1
    print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
