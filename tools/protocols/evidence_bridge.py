from __future__ import annotations

import hashlib
import json
import platform
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from tools.perf.structured_logger import InferenceLogPlugin, InferenceLogRecord

JsonMapping = Mapping[str, Any]

_LOCKFILE_CANDIDATES = (
    "uv.lock",
    "poetry.lock",
    "Pipfile.lock",
    "requirements.lock",
    "requirements.txt",
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "go.sum",
    "Cargo.lock",
)
_STALE_CLASSIFICATIONS = {"fresh", "soft_stale", "hard_stale"}
_MERGE_STATUSES = {
    "not_applicable",
    "not_attempted",
    "merged",
    "manual_resolution_required",
}
_POLICY_VERDICTS = {"approved", "rejected", "manual_review_required"}
_APPROVAL_ROLES = {"project_lead", "security_reviewer", "release_manager", "admin"}
_APPROVAL_DECISIONS = {"approved", "rejected"}


class AgentProtocolEvidenceError(ValueError):
    pass


class InferenceContextExtractor(Protocol):
    def extract(self, record: InferenceLogRecord) -> JsonMapping | None:
        ...


class EnvironmentResolver(Protocol):
    def resolve(self) -> JsonMapping:
        ...


class EvidenceWriter(Protocol):
    def write(self, evidence: JsonMapping) -> None:
        ...


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


def _stable_json(value: Any) -> str:
    return json.dumps(_to_jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return f"sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


def _sha256_json(value: Any) -> str:
    return _sha256_text(_stable_json(value))


def _normalize_timestamp(value: str | datetime | None, *, default: str | None = None) -> str:
    if value is None:
        if default is None:
            value = datetime.now(timezone.utc)
        else:
            return default
    if isinstance(value, datetime):
        parsed = value
    else:
        candidate = value.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(candidate)
        except ValueError as exc:
            raise AgentProtocolEvidenceError(f"Invalid timestamp: {value}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)
    return parsed.isoformat()


def _require_mapping(value: Any, *, field_name: str) -> JsonMapping:
    if not isinstance(value, Mapping):
        raise AgentProtocolEvidenceError(f"{field_name} must be a mapping")
    return value


def _require_text(mapping: JsonMapping, key: str, *, field_name: str | None = None) -> str:
    value = mapping.get(key)
    label = field_name or key
    if not isinstance(value, str) or not value.strip():
        raise AgentProtocolEvidenceError(f"{label} is required")
    return value.strip()


def _optional_text(mapping: JsonMapping, key: str, default: str) -> str:
    value = mapping.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return default


def _normalize_tools(value: Any) -> list[str]:
    if value is None:
        return ["StructuredLogger"]
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise AgentProtocolEvidenceError("agent_protocol.tools must be a sequence of strings")
    normalized = [str(item).strip() for item in value if str(item).strip()]
    if not normalized:
        raise AgentProtocolEvidenceError("agent_protocol.tools must contain at least one tool")
    return normalized


def _normalize_stale_status(value: Any, *, default_timestamp: str) -> dict[str, str]:
    if value is None:
        return {
            "classification": "fresh",
            "evaluatedAt": default_timestamp,
        }
    mapping = _require_mapping(value, field_name="agent_protocol.stale_status")
    classification = _require_text(mapping, "classification", field_name="stale_status.classification")
    if classification not in _STALE_CLASSIFICATIONS:
        raise AgentProtocolEvidenceError("stale_status.classification is invalid")
    stale_status = {
        "classification": classification,
        "evaluatedAt": _normalize_timestamp(mapping.get("evaluated_at"), default=default_timestamp),
    }
    reason = mapping.get("reason")
    if isinstance(reason, str) and reason.strip():
        stale_status["reason"] = reason.strip()
    return stale_status


def _normalize_merge_result(value: Any) -> dict[str, str]:
    if value is None:
        return {"status": "not_applicable"}
    mapping = _require_mapping(value, field_name="agent_protocol.merge_result")
    status = _require_text(mapping, "status", field_name="merge_result.status")
    if status not in _MERGE_STATUSES:
        raise AgentProtocolEvidenceError("merge_result.status is invalid")
    merge_result: dict[str, str] = {"status": status}
    merged_at = mapping.get("merged_at")
    if merged_at is not None:
        merge_result["mergedAt"] = _normalize_timestamp(merged_at)
    for source_key, target_key in (("strategy", "strategy"), ("reason", "reason")):
        candidate = mapping.get(source_key)
        if isinstance(candidate, str) and candidate.strip():
            merge_result[target_key] = candidate.strip()
    return merge_result


def _normalize_approvals(value: Any) -> list[dict[str, str]] | None:
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise AgentProtocolEvidenceError("agent_protocol.approvals_snapshot must be a sequence")
    approvals: list[dict[str, str]] = []
    for index, item in enumerate(value):
        mapping = _require_mapping(item, field_name=f"agent_protocol.approvals_snapshot[{index}]")
        role = _require_text(mapping, "role", field_name="approvals_snapshot.role")
        if role not in _APPROVAL_ROLES:
            raise AgentProtocolEvidenceError("approvals_snapshot.role is invalid")
        decision = _require_text(mapping, "decision", field_name="approvals_snapshot.decision")
        if decision not in _APPROVAL_DECISIONS:
            raise AgentProtocolEvidenceError("approvals_snapshot.decision is invalid")
        approval = {
            "role": role,
            "actorId": _require_text(mapping, "actor_id", field_name="approvals_snapshot.actor_id"),
            "decision": decision,
            "decidedAt": _normalize_timestamp(mapping.get("decided_at")),
        }
        reason = mapping.get("reason")
        if isinstance(reason, str) and reason.strip():
            approval["reason"] = reason.strip()
        approvals.append(approval)
    if not approvals:
        raise AgentProtocolEvidenceError("agent_protocol.approvals_snapshot must not be empty")
    return approvals


def _detect_lockfile(repo_root: Path, candidates: Sequence[str]) -> Path | None:
    for name in candidates:
        candidate = repo_root / name
        if candidate.is_file():
            return candidate
    return None


class AgentProtocolContextExtractor:
    def __init__(self, *, context_key: str = "agent_protocol") -> None:
        self._context_key = context_key

    def extract(self, record: InferenceLogRecord) -> JsonMapping | None:
        extra = record.extra
        if extra is None:
            return None
        if not isinstance(extra, Mapping):
            raise AgentProtocolEvidenceError("record.extra must be a mapping")
        context = extra.get(self._context_key)
        if context is None:
            return None
        return _require_mapping(context, field_name=f"record.extra.{self._context_key}")


class RepositoryEnvironmentResolver:
    def __init__(
        self,
        *,
        repo_root: str | Path,
        lockfile_candidates: Sequence[str] = _LOCKFILE_CANDIDATES,
    ) -> None:
        self._repo_root = Path(repo_root).expanduser()
        self._lockfile_candidates = tuple(lockfile_candidates)

    def resolve(self) -> JsonMapping:
        return {
            "os": _detect_os(),
            "runtime": _detect_runtime(),
            "containerImageDigest": "uncontainerized",
            "lockfileHash": self._lockfile_hash(),
        }

    def _lockfile_hash(self) -> str:
        lockfile = _detect_lockfile(self._repo_root, self._lockfile_candidates)
        if lockfile is None:
            return _sha256_text("missing-lockfile")
        return _sha256_text(lockfile.read_text(encoding="utf-8"))


class JsonLinesEvidenceWriter:
    def __init__(self, *, path: str | Path) -> None:
        self._path = Path(path).expanduser()

    def write(self, evidence: JsonMapping) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(evidence), ensure_ascii=False) + "\n")


class AgentProtocolEvidenceMapper:
    def __init__(
        self,
        *,
        context_extractor: InferenceContextExtractor | None = None,
        environment_resolver: EnvironmentResolver,
    ) -> None:
        self._context_extractor = context_extractor or AgentProtocolContextExtractor()
        self._environment_resolver = environment_resolver

    def map_record(self, record: InferenceLogRecord) -> dict[str, Any] | None:
        context = self._context_extractor.extract(record)
        if context is None:
            return None

        timestamp = _normalize_timestamp(record.timestamp)
        start_time = _normalize_timestamp(context.get("start_time"), default=timestamp)
        model_name = record.model.strip() if isinstance(record.model, str) and record.model.strip() else None
        if model_name is None:
            model_name = _optional_text(context, "model_name", "")
        if not model_name:
            raise AgentProtocolEvidenceError("model is required to emit Evidence")

        parameters_hash = context.get("parameters_hash")
        if isinstance(parameters_hash, str) and parameters_hash.strip():
            normalized_parameters_hash = parameters_hash.strip()
        else:
            normalized_parameters_hash = _sha256_json(context.get("parameters", {}))

        diff_hash = context.get("diff_hash")
        if isinstance(diff_hash, str) and diff_hash.strip():
            normalized_diff_hash = diff_hash.strip()
        else:
            normalized_diff_hash = _sha256_json(context.get("diff", ""))

        resolved_environment = dict(self._environment_resolver.resolve())
        environment_overrides = context.get("environment")
        if environment_overrides is not None:
            resolved_environment.update(
                _require_mapping(environment_overrides, field_name="agent_protocol.environment")
            )

        evidence: dict[str, Any] = {
            "schemaVersion": "1.0.0",
            "id": _require_text(context, "evidence_id", field_name="agent_protocol.evidence_id"),
            "kind": "Evidence",
            "state": "Published",
            "version": 1,
            "createdAt": timestamp,
            "updatedAt": timestamp,
            "taskSeedId": _require_text(context, "task_seed_id", field_name="agent_protocol.task_seed_id"),
            "baseCommit": _require_text(context, "base_commit", field_name="agent_protocol.base_commit"),
            "headCommit": _require_text(context, "head_commit", field_name="agent_protocol.head_commit"),
            "inputHash": _sha256_json(record.prompt if record.prompt is not None else {}),
            "outputHash": _sha256_json(record.response if record.response is not None else {}),
            "model": {
                "name": model_name,
                "version": _optional_text(context, "model_version", "unknown"),
                "parametersHash": normalized_parameters_hash,
            },
            "tools": _normalize_tools(context.get("tools")),
            "environment": {
                "os": _optional_text(resolved_environment, "os", _detect_os()),
                "runtime": _optional_text(resolved_environment, "runtime", _detect_runtime()),
                "containerImageDigest": _optional_text(
                    resolved_environment,
                    "containerImageDigest",
                    "uncontainerized",
                ),
                "lockfileHash": _optional_text(
                    resolved_environment,
                    "lockfileHash",
                    _sha256_text("missing-lockfile"),
                ),
            },
            "staleStatus": _normalize_stale_status(context.get("stale_status"), default_timestamp=timestamp),
            "mergeResult": _normalize_merge_result(context.get("merge_result")),
            "startTime": start_time,
            "endTime": timestamp,
            "actor": _require_text(context, "actor", field_name="agent_protocol.actor"),
            "policyVerdict": _optional_text(context, "policy_verdict", "manual_review_required"),
            "diffHash": normalized_diff_hash,
        }
        if evidence["policyVerdict"] not in _POLICY_VERDICTS:
            raise AgentProtocolEvidenceError("policy_verdict is invalid")
        approvals = _normalize_approvals(context.get("approvals_snapshot"))
        if approvals is not None:
            evidence["approvalsSnapshot"] = approvals
        return evidence


class AgentProtocolEvidencePlugin(InferenceLogPlugin):
    def __init__(
        self,
        *,
        writer: EvidenceWriter,
        mapper: AgentProtocolEvidenceMapper,
    ) -> None:
        self._writer = writer
        self._mapper = mapper

    def handle_inference(self, record: InferenceLogRecord) -> None:
        evidence = self._mapper.map_record(record)
        if evidence is None:
            return
        self._writer.write(evidence)


class AgentProtocolEvidenceBridge:
    def __init__(
        self,
        *,
        repo_root: str | Path,
        context_extractor: InferenceContextExtractor | None = None,
        environment_resolver: EnvironmentResolver | None = None,
    ) -> None:
        self._mapper = AgentProtocolEvidenceMapper(
            context_extractor=context_extractor,
            environment_resolver=environment_resolver
            or RepositoryEnvironmentResolver(repo_root=repo_root),
        )

    def build_inference_evidence(self, record: InferenceLogRecord) -> dict[str, Any] | None:
        return self._mapper.map_record(record)


class AgentProtocolEvidenceFileSink:
    def __init__(
        self,
        *,
        path: str | Path,
        repo_root: str | Path,
        context_extractor: InferenceContextExtractor | None = None,
        environment_resolver: EnvironmentResolver | None = None,
        bridge: AgentProtocolEvidenceBridge | None = None,
    ) -> None:
        if bridge is None:
            bridge = AgentProtocolEvidenceBridge(
                repo_root=repo_root,
                context_extractor=context_extractor,
                environment_resolver=environment_resolver,
            )
        self._plugin = AgentProtocolEvidencePlugin(
            writer=JsonLinesEvidenceWriter(path=path),
            mapper=bridge._mapper,
        )

    def emit_inference_evidence(self, record: InferenceLogRecord) -> None:
        self._plugin.handle_inference(record)

    def handle_inference(self, record: InferenceLogRecord) -> None:
        self._plugin.handle_inference(record)


def create_agent_protocol_evidence_plugin(
    *,
    path: str | Path,
    repo_root: str | Path,
    context_key: str = "agent_protocol",
) -> AgentProtocolEvidencePlugin:
    return AgentProtocolEvidencePlugin(
        writer=JsonLinesEvidenceWriter(path=path),
        mapper=AgentProtocolEvidenceMapper(
            context_extractor=AgentProtocolContextExtractor(context_key=context_key),
            environment_resolver=RepositoryEnvironmentResolver(repo_root=repo_root),
        ),
    )


def _detect_os() -> str:
    return f"{platform.system()} {platform.release()}".strip()


def _detect_runtime() -> str:
    return f"Python {platform.python_version()}"


__all__ = [
    "AgentProtocolContextExtractor",
    "AgentProtocolEvidenceBridge",
    "AgentProtocolEvidenceError",
    "AgentProtocolEvidenceFileSink",
    "AgentProtocolEvidenceMapper",
    "AgentProtocolEvidencePlugin",
    "EvidenceWriter",
    "InferenceContextExtractor",
    "JsonLinesEvidenceWriter",
    "RepositoryEnvironmentResolver",
    "create_agent_protocol_evidence_plugin",
]
