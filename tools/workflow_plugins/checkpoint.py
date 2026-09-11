"""既存taskstate CASとcoordinatorを接続する永続checkpoint。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any, Protocol

from tools.evaluation.workflow import digest, integer, read_object, text
from tools.supervision.workspace_coordinator import WorkspaceCoordinator, deterministic_job_key


class StateStore(Protocol):
    def get(self, task_id: str) -> dict[str, Any]: ...
    def patch(self, task_id: str, revision: int, payload: dict[str, Any]) -> dict[str, Any]: ...


class TaskstateCLI:
    """Configured trusted CLI. Never uses shell or replaces the state database."""

    def __init__(self, command: Sequence[str], *, cwd: str | Path, db: str | Path, timeout: int = 30) -> None:
        if isinstance(command, str) or not command:
            raise ValueError("command must be a nonempty argv array")
        self.command = [text(value, "command argument") for value in command]
        self.cwd = Path(cwd).resolve(strict=True)
        self.db = str(Path(db).resolve())
        self.timeout = integer(timeout, "timeout", minimum=1)

    def _call(self, args: list[str]) -> dict[str, Any]:
        environment = os.environ.copy()
        environment["PYTHONIOENCODING"] = "utf-8"
        try:
            result = subprocess.run(
                [*self.command, "--db", self.db, *args],
                cwd=self.cwd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                env=environment,
                timeout=self.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("taskstate response timeout; update outcome unknown, re-read state") from exc
        payload = json.loads(result.stdout)
        if not isinstance(payload, dict) or payload.get("ok") is not True or result.returncode:
            raise RuntimeError(
                f"taskstate rejected request: {payload.get('error') if isinstance(payload, dict) else 'invalid envelope'}"
            )
        if not isinstance(payload.get("data"), dict):
            raise ValueError("taskstate data must be an object")
        return dict(payload["data"])

    def get(self, task_id: str) -> dict[str, Any]:
        return self._call(["state", "get", "--task", task_id])

    def build_context(self, task_id: str) -> dict[str, Any]:
        """Append a complete recovery bundle through the existing public CLI."""
        return self._call(["context", "build", "--task", task_id, "--reason", "recovery", "--rebuild-level", "L2"])

    def patch(self, task_id: str, revision: int, payload: dict[str, Any]) -> dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="workflow-state-") as directory:
            path = Path(directory) / "patch.json"
            path.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding="utf-8")
            return self._call(
                ["state", "patch", "--task", task_id, "--expected-revision", str(revision), "--file", str(path)]
            )


class WorkflowCheckpoint:
    def __init__(
        self,
        store: StateStore,
        coordinator: WorkspaceCoordinator,
        plan: Mapping[str, Any],
        *,
        grant: Mapping[str, Any] | None = None,
    ) -> None:
        self.store, self.coordinator = store, coordinator
        self.plan = deepcopy(dict(plan))
        if self.plan.get("schema_version") != "1.0":
            raise ValueError("plan schema_version 1.0 required")
        self.task_id = text(plan.get("task_id"), "task_id")
        self.run_id = text(plan.get("run_id"), "run_id")
        policy = text(plan.get("policy_version"), "policy_version")
        steps = plan.get("steps")
        if not isinstance(steps, list) or not steps:
            raise ValueError("nonempty ordered steps required")
        self.steps = [text(step, "step") for step in steps]
        if len(set(self.steps)) != len(self.steps):
            raise ValueError("duplicate step")
        self._artifacts(plan.get("inputs"))
        self.plan_hash = (
            "sha256:"
            + hashlib.sha256(
                json.dumps(
                    self.plan, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
                ).encode()
            ).hexdigest()
        )
        self.job_key = str(
            deterministic_job_key(
                workspace=coordinator.workspace,
                scope=self.task_id,
                role="workflow",
                stage="checkpoint",
                source_revision=policy,
                input_hashes=[self.plan_hash],
                command_identity="workflow-checkpoint-v1",
                output_root=".",
            )["job_key"]
        )
        self.grant = dict(grant or {})

    def _artifacts(self, values: object, *, nonempty: bool = True) -> list[dict[str, str]]:
        if not isinstance(values, list) or (nonempty and not values):
            raise ValueError("nonempty artifact array required")
        artifacts: list[dict[str, str]] = []
        seen: set[Path] = set()
        for value in values:
            if not isinstance(value, dict):
                raise ValueError("artifact object required")
            relative = Path(text(value.get("path"), "artifact path"))
            if relative.is_absolute():
                raise ValueError("artifact path must be workspace relative")
            path = (self.coordinator.workspace / relative).resolve(strict=True)
            if not path.is_relative_to(self.coordinator.workspace) or not path.is_file():
                raise ValueError("artifact outside workspace or not a file")
            expected = text(value.get("sha256"), "artifact sha256")
            if not re.fullmatch(r"sha256:[0-9a-f]{64}", expected) or digest(path) != expected:
                raise ValueError("artifact hash mismatch")
            if path in seen:
                raise ValueError("duplicate artifact")
            seen.add(path)
            artifacts.append({"path": path.relative_to(self.coordinator.workspace).as_posix(), "sha256": expected})
        return artifacts

    def _load(self) -> tuple[dict[str, Any], dict[str, Any] | None]:
        self._artifacts(self.plan["inputs"])
        state = self.store.get(self.task_id)
        if state.get("task_id") != self.task_id:
            raise ValueError("state task mismatch")
        integer(state.get("revision"), "state revision", minimum=1)
        policy = state.get("context_policy")
        if not isinstance(policy, dict):
            raise ValueError("context_policy object required")
        checkpoint = policy.get("workflow_checkpoint")
        if checkpoint is None:
            return state, None
        if not isinstance(checkpoint, dict) or checkpoint.get("schema_version") != "1.0":
            raise ValueError("invalid checkpoint")
        if (checkpoint.get("plan_sha256"), checkpoint.get("job_key"), checkpoint.get("run_id")) != (
            self.plan_hash,
            self.job_key,
            self.run_id,
        ):
            raise ValueError("checkpoint identity changed")
        rows = checkpoint.get("steps")
        if not isinstance(rows, list) or [row.get("step_id") for row in rows if isinstance(row, dict)] != self.steps:
            raise ValueError("checkpoint step order changed")
        waiting = False
        for row in rows:
            status = row.get("status")
            if status not in ("pending", "running", "completed"):
                raise ValueError("invalid checkpoint step status")
            if waiting and status != "pending":
                raise ValueError("checkpoint has out-of-order execution")
            if status == "completed":
                self._artifacts(row.get("artifacts"))
            else:
                waiting = True
                if row.get("artifacts"):
                    raise ValueError("unfinished step cannot have committed artifacts")
            for decision in row.get("reconciliations", []):
                self._artifacts([decision["record"]])
        return state, deepcopy(checkpoint)

    def _guard(self) -> dict[str, Any]:
        lease = self.grant.get("lease")
        if not isinstance(lease, dict):
            raise ValueError("write lease grant required")
        lease_id = text(lease.get("lease_id"), "lease_id")
        token = text(self.grant.get("lease_token"), "lease token")
        active = next((row for row in self.coordinator.status()["active_leases"] if row["lease_id"] == lease_id), None)
        if not active or (
            active["task_id"],
            active["job_key"],
            active["mode"],
            active["paths"],
            active["fence_token"],
        ) != (
            self.task_id,
            self.job_key,
            "write",
            [os.path.normcase(str(self.coordinator.workspace))],
            lease.get("fence_token"),
        ):
            raise ValueError("lease expired or task/job/scope/fence mismatch")
        if self.coordinator.heartbeat(lease_id, token).get("ok") is not True:
            raise ValueError("lease token rejected")
        return dict(active)

    def _save(self, state: dict[str, Any], checkpoint: dict[str, Any]) -> dict[str, Any]:
        lease = self._guard()
        policy = deepcopy(state["context_policy"])
        checkpoint["last_fence_token"] = lease["fence_token"]
        policy["workflow_checkpoint"] = checkpoint
        pending = next((row["step_id"] for row in checkpoint["steps"] if row["status"] != "completed"), None)
        return self.store.patch(
            self.task_id,
            state["revision"],
            {
                "context_policy": policy,
                "current_step": pending or "workflow:completed",
            },
        )

    def start(self) -> dict[str, Any]:
        state, checkpoint = self._load()
        if checkpoint is None:
            checkpoint = {
                "schema_version": "1.0",
                "plan_sha256": self.plan_hash,
                "run_id": self.run_id,
                "job_key": self.job_key,
                "steps": [
                    {"step_id": step, "status": "pending", "artifacts": [], "reconciliations": []}
                    for step in self.steps
                ],
            }
            self._save(state, checkpoint)
        return self.status()

    def status(self) -> dict[str, Any]:
        state, checkpoint = self._load()
        if checkpoint is None:
            return {"status": "not_started", "job_key": self.job_key, "revision": state["revision"]}
        rows = checkpoint["steps"]
        running = [row["step_id"] for row in rows if row["status"] == "running"]
        remaining = [row["step_id"] for row in rows if row["status"] != "completed"]
        return {
            "status": "needs_reconciliation" if running else "ready" if remaining else "completed",
            "job_key": self.job_key,
            "task_id": self.task_id,
            "run_id": self.run_id,
            "revision": state["revision"],
            "running": running,
            "remaining": remaining,
            "checkpoint": checkpoint,
        }

    def transition(
        self,
        action: str,
        step_id: str,
        *,
        artifacts: object = None,
        resolution: str | None = None,
        record: object = None,
    ) -> dict[str, Any]:
        state, checkpoint = self._load()
        if checkpoint is None:
            raise ValueError("checkpoint not started")
        row = next((row for row in checkpoint["steps"] if row["step_id"] == step_id), None)
        if row is None:
            raise ValueError("unknown step")
        if action == "complete" and row["status"] == "completed":
            if self._artifacts(artifacts) != row["artifacts"]:
                raise ValueError("completed artifact set changed")
            return self.status()
        first = next((row["step_id"] for row in checkpoint["steps"] if row["status"] != "completed"), None)
        if first != step_id:
            raise ValueError("step is not next")
        if action == "begin":
            if row["status"] != "pending":
                raise ValueError("in-doubt step requires reconciliation")
            row["status"] = "running"
        elif action == "complete":
            if row["status"] != "running":
                raise ValueError("begin is required before complete")
            row.update(status="completed", artifacts=self._artifacts(artifacts))
        elif action == "reconcile":
            if row["status"] != "running" or resolution not in ("pending", "completed"):
                raise ValueError("reconcile requires running step and pending/completed resolution")
            decision = self._artifacts([record])[0]
            row["artifacts"] = self._artifacts(artifacts) if resolution == "completed" else []
            row["status"] = resolution
            row["reconciliations"].append({"resolution": resolution, "record": decision})
        else:
            raise ValueError("unknown action")
        self._save(state, checkpoint)
        return self.status()

    def finalize(self) -> dict[str, Any]:
        result = self.status()
        if result["status"] != "completed":
            raise ValueError("all steps must be completed before finalize")
        job = next((row for row in self.coordinator.status()["jobs"] if row["job_key"] == self.job_key), None)
        if job and job["status"] in ("succeeded", "invalid"):
            if job["status"] != "succeeded":
                raise ValueError("invalid job cannot be reused")
            return {"ok": True, "reused": True, "job": job, "artifacts_verified": True}
        lease = self._guard()
        result = self.coordinator.finish(
            lease["lease_id"],
            self.grant["lease_token"],
            outcome="succeeded",
            job_key=self.job_key,
            result_refs=[artifact["path"] for row in result["checkpoint"]["steps"] for artifact in row["artifacts"]],
        )
        if result.get("ok") is not True:
            raise ValueError(f"finalize rejected: {result.get('reason')}")
        return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "status", "begin", "complete", "reconcile", "finalize"))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument(
        "--state-client", type=Path, required=True, help="JSON: command argv, cwd, db, optional timeout"
    )
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--coordinator-root", type=Path, required=True)
    parser.add_argument("--grant", type=Path)
    parser.add_argument("--step")
    parser.add_argument("--artifacts", type=Path, help="JSON object with artifacts array")
    parser.add_argument("--resolution", choices=("pending", "completed"))
    parser.add_argument("--record", type=Path, help="JSON object with path/sha256 of decision record")
    args = parser.parse_args(argv)
    try:
        checkpoint = WorkflowCheckpoint(
            TaskstateCLI(**read_object(args.state_client)),
            WorkspaceCoordinator(args.workspace, state_root=args.coordinator_root),
            read_object(args.plan),
            grant=read_object(args.grant) if args.grant else None,
        )
        if args.action in ("start", "status", "finalize"):
            result = getattr(checkpoint, args.action)()
        else:
            result = checkpoint.transition(
                args.action,
                text(args.step, "step"),
                artifacts=read_object(args.artifacts).get("artifacts") if args.artifacts else None,
                resolution=args.resolution,
                record=read_object(args.record) if args.record else None,
            )
        code = 2 if args.action in ("status", "start") and result.get("status") == "needs_reconciliation" else 0
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        result, code = {"status": "invalid", "error": str(exc)}, 1
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
