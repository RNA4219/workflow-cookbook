#!/usr/bin/env python3
"""SQLite/WAL coordinator for multi-agent work in one local workspace."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import sqlite3
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Sequence

SCHEMA_VERSION = "workspace-supervisor-v1"
DEFAULT_TTL_SECONDS = 300
DEFAULT_TIMEOUT_MS = 5_000
TERMINAL_JOB_STATES = {"succeeded", "invalid"}


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _default_state_root() -> Path:
    configured = os.environ.get("WORKFLOW_SUPERVISOR_STATE_ROOT")
    return Path(configured).expanduser() if configured else Path.home() / ".workflow-cookbook" / "supervision"


def deterministic_job_key(
    *,
    workspace: str | Path,
    scope: str,
    role: str,
    stage: str,
    source_revision: str,
    input_hashes: Sequence[str],
    command_identity: str,
    output_root: str,
) -> dict[str, Any]:
    workspace_path = Path(workspace).expanduser().resolve(strict=True)
    payload = {
        "schema_version": "workspace-job-key-v1",
        "workspace": os.path.normcase(str(workspace_path)),
        "scope": scope,
        "role": role,
        "stage": stage,
        "source_revision": source_revision,
        "input_hashes": sorted(set(input_hashes)),
        "command_identity": command_identity,
        "output_root": os.path.normcase(str((workspace_path / output_root).resolve(strict=False))),
    }
    return {"job_key": f"sha256:{_sha256(_json(payload))}", "payload": payload}


class WorkspaceCoordinator:
    """Durable shared-read/exclusive-write leases with terminal job dedupe."""

    def __init__(
        self,
        workspace: str | Path,
        *,
        state_root: str | Path | None = None,
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
        clock: Callable[[], float] = time.time,
    ) -> None:
        workspace_path = Path(workspace).expanduser().resolve(strict=True)
        if not workspace_path.is_dir():
            raise ValueError(f"workspace is not a directory: {workspace_path}")
        self.workspace = workspace_path
        self.workspace_key = os.path.normcase(str(workspace_path))
        self.workspace_id = _sha256(self.workspace_key)
        root = Path(state_root).expanduser() if state_root else _default_state_root()
        self.state_dir = root
        self.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.db_path = self.state_dir / "coordinator.sqlite3"
        self.timeout_ms = timeout_ms
        self.clock = clock
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        deadline = time.monotonic() + max(self.timeout_ms, 0) / 1_000
        while True:
            connection = sqlite3.connect(
                self.db_path, timeout=max(deadline - time.monotonic(), 0), isolation_level=None
            )
            try:
                connection.row_factory = sqlite3.Row
                # Changing journal mode during concurrent first opens can return
                # SQLITE_BUSY without invoking SQLite's normal busy handler.
                connection.execute("PRAGMA journal_mode=WAL")
                connection.execute("PRAGMA synchronous=FULL")
                connection.execute("PRAGMA foreign_keys=ON")
                connection.execute(f"PRAGMA busy_timeout={self.timeout_ms}")
                return connection
            except sqlite3.OperationalError as exc:
                connection.close()
                remaining = deadline - time.monotonic()
                code = getattr(exc, "sqlite_errorcode", 0)
                if code & 0xFF != sqlite3.SQLITE_BUSY or remaining <= 0:
                    raise
                time.sleep(min(0.01, remaining))
            except BaseException:
                connection.close()
                raise

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS leases (
                    lease_id TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL,
                    mode TEXT NOT NULL CHECK (mode IN ('read', 'write')),
                    paths_json TEXT NOT NULL,
                    owner TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    job_key TEXT,
                    wip_key TEXT,
                    token_sha256 TEXT NOT NULL,
                    acquired_at REAL NOT NULL,
                    heartbeat_at REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    fence_token INTEGER NOT NULL UNIQUE,
                    state TEXT NOT NULL CHECK (state IN ('active', 'released', 'expired', 'completed', 'invalid'))
                );
                CREATE INDEX IF NOT EXISTS leases_active_idx ON leases(workspace_id, state, expires_at);
                CREATE INDEX IF NOT EXISTS leases_wip_idx ON leases(workspace_id, wip_key, state);
                CREATE TABLE IF NOT EXISTS jobs (
                    job_key TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    owner TEXT NOT NULL,
                    lease_id TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('active', 'released', 'expired', 'succeeded', 'invalid')),
                    created_at REAL NOT NULL,
                    completed_at REAL,
                    result_refs_json TEXT NOT NULL DEFAULT '[]'
                );
                CREATE TABLE IF NOT EXISTS events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    workspace_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    observed_at REAL NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS counters (
                    counter_key TEXT PRIMARY KEY,
                    value INTEGER NOT NULL
                );
                """
            )
        try:
            os.chmod(self.db_path, 0o600)
        except OSError:
            pass

    def _canonical_scopes(self, paths: Sequence[str] | None) -> list[str]:
        scopes = paths or ["."]
        canonical: list[str] = []
        for item in scopes:
            candidate = Path(item)
            resolved = (
                candidate.resolve(strict=False)
                if candidate.is_absolute()
                else (self.workspace / candidate).resolve(strict=False)
            )
            try:
                resolved.relative_to(self.workspace)
            except ValueError as exc:
                raise ValueError(f"scope escapes workspace: {item}") from exc
            normalized = os.path.normcase(str(resolved))
            if normalized not in canonical:
                canonical.append(normalized)
        return sorted(canonical)

    @staticmethod
    def _paths_overlap(left: str, right: str) -> bool:
        try:
            common = os.path.commonpath([left, right])
        except ValueError:
            return False
        return common == left or common == right

    @classmethod
    def _scope_sets_overlap(cls, left: Sequence[str], right: Sequence[str]) -> bool:
        return any(cls._paths_overlap(a, b) for a in left for b in right)

    def _event(
        self,
        connection: sqlite3.Connection,
        event_type: str,
        payload: dict[str, Any],
        *,
        workspace_id: str | None = None,
    ) -> None:
        connection.execute(
            "INSERT INTO events(workspace_id,event_type,observed_at,payload_json) VALUES(?,?,?,?)",
            (workspace_id or self.workspace_id, event_type, self.clock(), _json(payload)),
        )

    def _expire_stale(self, connection: sqlite3.Connection) -> None:
        now = self.clock()
        stale = connection.execute(
            "SELECT lease_id,workspace_id,job_key,owner,task_id FROM leases WHERE state='active' AND expires_at<=?",
            (now,),
        ).fetchall()
        for row in stale:
            connection.execute("UPDATE leases SET state='expired' WHERE lease_id=?", (row["lease_id"],))
            if row["job_key"]:
                connection.execute(
                    "UPDATE jobs SET status='expired' WHERE job_key=? AND status='active'",
                    (row["job_key"],),
                )
            self._event(connection, "lease.expired", dict(row), workspace_id=row["workspace_id"])

    def acquire(
        self,
        *,
        mode: str,
        owner: str,
        task_id: str,
        paths: Sequence[str] | None = None,
        job_key: str | None = None,
        wip_key: str | None = None,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
    ) -> dict[str, Any]:
        if mode not in {"read", "write"}:
            raise ValueError("mode must be read or write")
        if not owner or not task_id:
            raise ValueError("owner and task_id are required")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        scopes = self._canonical_scopes(paths)
        token = secrets.token_hex(32)
        lease_id = str(uuid.uuid4())
        now = self.clock()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._expire_stale(connection)
            if job_key:
                job = connection.execute("SELECT * FROM jobs WHERE job_key=?", (job_key,)).fetchone()
                if job and job["status"] in TERMINAL_JOB_STATES:
                    self._event(connection, "job.reused", {"job_key": job_key, "status": job["status"], "owner": owner})
                    connection.commit()
                    return {"acquired": False, "reason": "job_already_terminal", "job": self._public_job(job)}
                if job and job["status"] == "active":
                    connection.rollback()
                    return {"acquired": False, "reason": "job_active", "job": self._public_job(job)}
            active = connection.execute("SELECT * FROM leases WHERE state='active' ORDER BY acquired_at").fetchall()
            if wip_key:
                conflict = next((row for row in active if row["wip_key"] == wip_key), None)
                if conflict:
                    connection.rollback()
                    return {"acquired": False, "reason": "wip_busy", "conflict": self._public_lease(conflict)}
            for row in active:
                existing_scopes = json.loads(row["paths_json"])
                if not self._scope_sets_overlap(scopes, existing_scopes):
                    continue
                if mode == "write" or row["mode"] == "write":
                    connection.rollback()
                    return {"acquired": False, "reason": "workspace_busy", "conflict": self._public_lease(row)}
            counter = connection.execute("SELECT value FROM counters WHERE counter_key='workspace_io_fence'").fetchone()
            fence_token = int(counter["value"]) + 1 if counter else 1
            connection.execute(
                "INSERT INTO counters(counter_key,value) VALUES('workspace_io_fence',?) "
                "ON CONFLICT(counter_key) DO UPDATE SET value=excluded.value",
                (fence_token,),
            )
            connection.execute(
                "INSERT INTO leases VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    lease_id,
                    self.workspace_id,
                    mode,
                    _json(scopes),
                    owner,
                    task_id,
                    job_key,
                    wip_key,
                    _sha256(token),
                    now,
                    now,
                    now + ttl_seconds,
                    fence_token,
                    "active",
                ),
            )
            if job_key:
                connection.execute(
                    "INSERT INTO jobs(job_key,workspace_id,task_id,owner,lease_id,status,created_at,result_refs_json) "
                    "VALUES(?,?,?,?,?,'active',?,'[]') "
                    "ON CONFLICT(job_key) DO UPDATE SET workspace_id=excluded.workspace_id,task_id=excluded.task_id,"
                    "owner=excluded.owner,lease_id=excluded.lease_id,status='active',created_at=excluded.created_at,"
                    "completed_at=NULL,result_refs_json='[]' WHERE jobs.status IN ('released','expired')",
                    (job_key, self.workspace_id, task_id, owner, lease_id, now),
                )
            self._event(
                connection,
                "lease.acquired",
                {
                    "lease_id": lease_id,
                    "fence_token": fence_token,
                    "mode": mode,
                    "owner": owner,
                    "task_id": task_id,
                    "job_key": job_key,
                    "wip_key": wip_key,
                    "paths": scopes,
                },
            )
            connection.commit()
        return {
            "acquired": True,
            "lease": {
                "lease_id": lease_id,
                "workspace_id": self.workspace_id,
                "mode": mode,
                "owner": owner,
                "task_id": task_id,
                "job_key": job_key,
                "wip_key": wip_key,
                "paths": scopes,
                "acquired_at": now,
                "heartbeat_at": now,
                "expires_at": now + ttl_seconds,
                "fence_token": fence_token,
                "state": "active",
            },
            "lease_token": token,
        }

    def heartbeat(self, lease_id: str, lease_token: str, ttl_seconds: int = DEFAULT_TTL_SECONDS) -> dict[str, Any]:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        now = self.clock()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            checked = self._checked_lease(connection, lease_id, lease_token)
            if "lease" not in checked:
                connection.rollback()
                return checked
            connection.execute(
                "UPDATE leases SET heartbeat_at=?,expires_at=? WHERE lease_id=?",
                (now, now + ttl_seconds, lease_id),
            )
            self._event(connection, "lease.heartbeat", {"lease_id": lease_id, "expires_at": now + ttl_seconds})
            connection.commit()
        return {"ok": True, "lease_id": lease_id, "heartbeat_at": now, "expires_at": now + ttl_seconds}

    def release(self, lease_id: str, lease_token: str) -> dict[str, Any]:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            checked = self._checked_lease(connection, lease_id, lease_token)
            if "lease" not in checked:
                connection.rollback()
                return checked
            lease = checked["lease"]
            connection.execute("UPDATE leases SET state='released' WHERE lease_id=?", (lease_id,))
            if lease["job_key"]:
                connection.execute(
                    "UPDATE jobs SET status='released' WHERE job_key=? AND status='active'",
                    (lease["job_key"],),
                )
            self._event(connection, "lease.released", {"lease_id": lease_id, "owner": lease["owner"]})
            connection.commit()
        return {"ok": True, "lease": self._public_lease(lease)}

    def finish(
        self,
        lease_id: str,
        lease_token: str,
        *,
        outcome: str,
        job_key: str,
        result_refs: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        if outcome not in TERMINAL_JOB_STATES:
            raise ValueError("outcome must be succeeded or invalid")
        lease_outcome = "completed" if outcome == "succeeded" else "invalid"
        now = self.clock()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            checked = self._checked_lease(connection, lease_id, lease_token)
            if "lease" not in checked:
                connection.rollback()
                return checked
            lease = checked["lease"]
            if lease["job_key"] and lease["job_key"] != job_key:
                connection.rollback()
                return {"ok": False, "reason": "job_key_mismatch"}
            job = connection.execute("SELECT * FROM jobs WHERE job_key=?", (job_key,)).fetchone()
            if job and job["status"] in TERMINAL_JOB_STATES:
                connection.execute("UPDATE leases SET state=? WHERE lease_id=?", (lease_outcome, lease_id))
                self._event(
                    connection, "job.reused", {"job_key": job_key, "status": job["status"], "lease_id": lease_id}
                )
                connection.commit()
                return {"ok": True, "reused": True, "job": self._public_job(job)}
            refs = list(result_refs or [])
            connection.execute("UPDATE leases SET state=? WHERE lease_id=?", (lease_outcome, lease_id))
            connection.execute(
                "INSERT INTO jobs(job_key,workspace_id,task_id,owner,lease_id,status,created_at,completed_at,result_refs_json) "
                "VALUES(?,?,?,?,?,?,?,?,?) ON CONFLICT(job_key) DO UPDATE SET status=excluded.status,"
                "lease_id=excluded.lease_id,completed_at=excluded.completed_at,result_refs_json=excluded.result_refs_json",
                (
                    job_key,
                    self.workspace_id,
                    lease["task_id"],
                    lease["owner"],
                    lease_id,
                    outcome,
                    lease["acquired_at"],
                    now,
                    _json(refs),
                ),
            )
            self._event(
                connection,
                "job.finished",
                {"job_key": job_key, "outcome": outcome, "lease_id": lease_id, "result_refs": refs},
            )
            connection.commit()
            job = connection.execute("SELECT * FROM jobs WHERE job_key=?", (job_key,)).fetchone()
        return {"ok": True, "reused": False, "job": self._public_job(job)}

    def status(self, *, include_events: int = 0) -> dict[str, Any]:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._expire_stale(connection)
            leases = connection.execute(
                "SELECT * FROM leases WHERE workspace_id=? AND state='active' ORDER BY acquired_at",
                (self.workspace_id,),
            ).fetchall()
            jobs = connection.execute(
                "SELECT * FROM jobs WHERE workspace_id=? ORDER BY created_at DESC",
                (self.workspace_id,),
            ).fetchall()
            events: list[dict[str, Any]] = []
            if include_events > 0:
                rows = connection.execute(
                    "SELECT * FROM events WHERE workspace_id=? ORDER BY sequence DESC LIMIT ?",
                    (self.workspace_id, include_events),
                ).fetchall()
                events = [self._public_event(row) for row in reversed(rows)]
            connection.commit()
        return {
            "schema_version": SCHEMA_VERSION,
            "workspace_id": self.workspace_id,
            "workspace_path": str(self.workspace),
            "database_path": str(self.db_path),
            "journal_mode": "wal",
            "active_leases": [self._public_lease(row) for row in leases],
            "jobs": [self._public_job(row) for row in jobs],
            "events": events,
        }

    def _checked_lease(self, connection: sqlite3.Connection, lease_id: str, lease_token: str) -> dict[str, Any]:
        self._expire_stale(connection)
        lease = connection.execute("SELECT * FROM leases WHERE lease_id=?", (lease_id,)).fetchone()
        if not lease or lease["state"] != "active":
            return {"ok": False, "reason": "lease_not_active"}
        if not secrets.compare_digest(lease["token_sha256"], _sha256(lease_token)):
            return {"ok": False, "reason": "token_mismatch"}
        return {"ok": True, "lease": lease}

    @staticmethod
    def _public_lease(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "lease_id": row["lease_id"],
            "workspace_id": row["workspace_id"],
            "mode": row["mode"],
            "paths": json.loads(row["paths_json"]),
            "owner": row["owner"],
            "task_id": row["task_id"],
            "job_key": row["job_key"],
            "wip_key": row["wip_key"],
            "acquired_at": row["acquired_at"],
            "heartbeat_at": row["heartbeat_at"],
            "expires_at": row["expires_at"],
            "fence_token": row["fence_token"],
            "state": row["state"],
        }

    @staticmethod
    def _public_job(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "job_key": row["job_key"],
            "task_id": row["task_id"],
            "owner": row["owner"],
            "lease_id": row["lease_id"],
            "status": row["status"],
            "created_at": row["created_at"],
            "completed_at": row["completed_at"],
            "result_refs": json.loads(row["result_refs_json"]),
            "reuse_requires_validation": row["status"] == "succeeded",
        }

    @staticmethod
    def _public_event(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "sequence": row["sequence"],
            "event_type": row["event_type"],
            "observed_at": row["observed_at"],
            "payload": json.loads(row["payload_json"]),
        }


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--state-root")
    parser.add_argument("--timeout-ms", type=int, default=DEFAULT_TIMEOUT_MS)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    acquire = sub.add_parser("acquire")
    _add_common(acquire)
    acquire.add_argument("--mode", choices=("read", "write"), required=True)
    acquire.add_argument("--owner", required=True)
    acquire.add_argument("--task-id", required=True)
    acquire.add_argument("--path", action="append", dest="paths")
    acquire.add_argument("--job-key")
    acquire.add_argument("--wip-key", help="Serialize jobs sharing this optional resource key")
    acquire.add_argument("--ttl-seconds", type=int, default=DEFAULT_TTL_SECONDS)
    for name in ("heartbeat", "release"):
        target = sub.add_parser(name)
        _add_common(target)
        target.add_argument("--lease-id", required=True)
        target.add_argument("--lease-token", required=True)
        if name == "heartbeat":
            target.add_argument("--ttl-seconds", type=int, default=DEFAULT_TTL_SECONDS)
    finish = sub.add_parser("finish")
    _add_common(finish)
    finish.add_argument("--lease-id", required=True)
    finish.add_argument("--lease-token", required=True)
    finish.add_argument("--job-key", required=True)
    finish.add_argument("--outcome", choices=sorted(TERMINAL_JOB_STATES), required=True)
    finish.add_argument("--result-ref", action="append", default=[])
    status = sub.add_parser("status")
    _add_common(status)
    status.add_argument("--events", type=int, default=0)
    probe = sub.add_parser("probe")
    _add_common(probe)
    probe.add_argument("--probe-delay-ms", type=int, default=0, help=argparse.SUPPRESS)
    job_key = sub.add_parser("job-key")
    job_key.add_argument("--workspace", required=True)
    job_key.add_argument("--scope", required=True)
    job_key.add_argument("--role", required=True)
    job_key.add_argument("--stage", required=True)
    job_key.add_argument("--source-revision", required=True)
    job_key.add_argument("--input-hash", action="append", default=[])
    job_key.add_argument("--command-identity", required=True)
    job_key.add_argument("--output-root", required=True)
    child = sub.add_parser("_probe-child")
    _add_common(child)
    child.add_argument("--probe-delay-ms", type=int, default=0)
    return parser


def _coordinator(args: argparse.Namespace) -> WorkspaceCoordinator:
    return WorkspaceCoordinator(args.workspace, state_root=args.state_root, timeout_ms=args.timeout_ms)


def _probe(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "_probe-child",
        "--workspace",
        args.workspace,
        "--timeout-ms",
        str(args.timeout_ms),
        "--probe-delay-ms",
        str(args.probe_delay_ms),
    ]
    if args.state_root:
        command.extend(["--state-root", args.state_root])
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=args.timeout_ms / 1_000, check=False
        )
    except subprocess.TimeoutExpired:
        return 4, {"ok": False, "reason": "probe_timeout", "timeout_ms": args.timeout_ms}
    latency = round((time.monotonic() - started) * 1_000, 3)
    if completed.returncode != 0:
        return 4, {"ok": False, "reason": "probe_failed", "stderr": completed.stderr[-500:]}
    return 0, {"ok": True, "latency_ms": latency, "child": json.loads(completed.stdout)}


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "job-key":
            result = deterministic_job_key(
                workspace=args.workspace,
                scope=args.scope,
                role=args.role,
                stage=args.stage,
                source_revision=args.source_revision,
                input_hashes=args.input_hash,
                command_identity=args.command_identity,
                output_root=args.output_root,
            )
            code = 0
        elif args.command == "probe":
            code, result = _probe(args)
        else:
            coordinator = _coordinator(args)
            if args.command == "acquire":
                result = coordinator.acquire(
                    mode=args.mode,
                    owner=args.owner,
                    task_id=args.task_id,
                    paths=args.paths,
                    job_key=args.job_key,
                    wip_key=args.wip_key,
                    ttl_seconds=args.ttl_seconds,
                )
                code = 0 if result.get("acquired") or result.get("reason") == "job_already_terminal" else 2
            elif args.command == "heartbeat":
                result = coordinator.heartbeat(args.lease_id, args.lease_token, args.ttl_seconds)
                code = 0 if result.get("ok") else 3
            elif args.command == "release":
                result = coordinator.release(args.lease_id, args.lease_token)
                code = 0 if result.get("ok") else 3
            elif args.command == "finish":
                result = coordinator.finish(
                    args.lease_id,
                    args.lease_token,
                    outcome=args.outcome,
                    job_key=args.job_key,
                    result_refs=args.result_ref,
                )
                code = 0 if result.get("ok") else 3
            elif args.command == "status":
                result = coordinator.status(include_events=max(args.events, 0))
                code = 0
            elif args.command == "_probe-child":
                if args.probe_delay_ms:
                    time.sleep(args.probe_delay_ms / 1_000)
                result = coordinator.status()
                code = 0
            else:
                raise ValueError(f"unsupported command: {args.command}")
    except (OSError, sqlite3.Error, ValueError) as exc:
        result = {"ok": False, "reason": "error", "message": str(exc)}
        code = 4
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
