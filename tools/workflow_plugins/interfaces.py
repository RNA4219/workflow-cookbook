# SPDX-License-Identifier: MIT
# Copyright 2025 RNA4219

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable


CAPABILITY_METHOD_NAMES: dict[str, str] = {
    "task_state.sync": "sync_task_acceptance",
    "acceptance.index": "build_acceptance_index",
    "docs.resolve": "resolve_docs",
    "docs.ack": "ack_docs",
    "docs.stale_check": "stale_check",
}


@dataclass(frozen=True)
class TaskAcceptanceSyncReport:
    tasks: list[dict[str, Any]]
    acceptances: list[dict[str, Any]]
    errors: list[str]
    warnings: list[str]


@dataclass(frozen=True)
class AcceptanceIndexResult:
    markdown: str
    rows: list[dict[str, Any]]


@dataclass(frozen=True)
class DocsResolveResult:
    required: list[dict[str, Any]]
    recommended: list[dict[str, Any]]
    errors: list[str]
    warnings: list[str]


@dataclass(frozen=True)
class DocsAckResult:
    receipts: list[dict[str, Any]]


@dataclass(frozen=True)
class DocsStaleResult:
    task_id: str
    stale: list[dict[str, Any]]


@runtime_checkable
class WorkflowPluginProtocol(Protocol):
    capabilities: Sequence[str]


@runtime_checkable
class TaskStateSyncPluginProtocol(WorkflowPluginProtocol, Protocol):
    def sync_task_acceptance(self, *, repo_root: Path) -> TaskAcceptanceSyncReport | Mapping[str, Any]:
        ...


@runtime_checkable
class AcceptanceIndexPluginProtocol(WorkflowPluginProtocol, Protocol):
    def build_acceptance_index(self, *, repo_root: Path) -> AcceptanceIndexResult | Mapping[str, Any]:
        ...


@runtime_checkable
class DocsResolvePluginProtocol(WorkflowPluginProtocol, Protocol):
    def resolve_docs(
        self,
        *,
        repo_root: Path,
        task_id: str,
        intent_id: str | None = None,
    ) -> DocsResolveResult | Mapping[str, Any]:
        ...


@runtime_checkable
class DocsAckPluginProtocol(WorkflowPluginProtocol, Protocol):
    def ack_docs(
        self,
        *,
        repo_root: Path,
        task_id: str,
        doc_ids: list[str],
        reader: str,
    ) -> DocsAckResult | Mapping[str, Any]:
        ...


@runtime_checkable
class DocsStalePluginProtocol(WorkflowPluginProtocol, Protocol):
    def stale_check(self, *, repo_root: Path, task_id: str) -> DocsStaleResult | Mapping[str, Any]:
        ...


def as_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    return value


def _mapping(value: Any, *, kind: str) -> dict[str, Any]:
    payload = as_jsonable(value)
    if not isinstance(payload, Mapping):
        raise TypeError(f"{kind} payload must be a mapping")
    return dict(payload)


def coerce_task_acceptance_sync_report(value: Any) -> TaskAcceptanceSyncReport:
    payload = _mapping(value, kind="task_state.sync")
    return TaskAcceptanceSyncReport(
        tasks=list(payload.get("tasks", [])),
        acceptances=list(payload.get("acceptances", [])),
        errors=[str(item) for item in payload.get("errors", [])],
        warnings=[str(item) for item in payload.get("warnings", [])],
    )


def coerce_acceptance_index_result(value: Any) -> AcceptanceIndexResult:
    payload = _mapping(value, kind="acceptance.index")
    return AcceptanceIndexResult(
        markdown=str(payload.get("markdown", "")),
        rows=list(payload.get("rows", [])),
    )


def coerce_docs_resolve_result(value: Any) -> DocsResolveResult:
    payload = _mapping(value, kind="docs.resolve")
    return DocsResolveResult(
        required=list(payload.get("required", [])),
        recommended=list(payload.get("recommended", [])),
        errors=[str(item) for item in payload.get("errors", [])],
        warnings=[str(item) for item in payload.get("warnings", [])],
    )


def coerce_docs_ack_result(value: Any) -> DocsAckResult:
    payload = _mapping(value, kind="docs.ack")
    return DocsAckResult(receipts=list(payload.get("receipts", [])))


def coerce_docs_stale_result(value: Any) -> DocsStaleResult:
    payload = _mapping(value, kind="docs.stale_check")
    return DocsStaleResult(
        task_id=str(payload.get("task_id", "")),
        stale=list(payload.get("stale", [])),
    )


CAPABILITY_COERCERS: dict[str, Callable[[Any], Any]] = {
    "task_state.sync": coerce_task_acceptance_sync_report,
    "acceptance.index": coerce_acceptance_index_result,
    "docs.resolve": coerce_docs_resolve_result,
    "docs.ack": coerce_docs_ack_result,
    "docs.stale_check": coerce_docs_stale_result,
}
