# SPDX-License-Identifier: MIT
# Copyright 2025 RNA4219

"""Data types used by the workflow plugin runtime."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class PluginPolicy:
    """Policy for plugin execution."""

    timeout_seconds: float = 30.0
    retry_count: int = 0
    retry_delay_seconds: float = 1.0
    continue_on_error: bool = False
    trace_enabled: bool = False
    isolation_mode: str = "thread"


@dataclass(frozen=True)
class RunContext:
    task_id: str
    run_id: str

    def __post_init__(self) -> None:
        if any(not isinstance(value, str) or not value.strip() for value in (self.task_id, self.run_id)):
            raise ValueError("task_id and run_id are required")


@dataclass
class PluginTrace:
    """Trace record for a plugin invocation."""

    plugin_name: str
    capability: str
    method_name: str
    start_time: float
    end_time: float | None = None
    success: bool = True
    error: str | None = None
    result_summary: str | None = None
    attempt: int = 1
    timeout_seconds: float | None = None
    isolation_mode: str = "thread"
    timed_out: bool = False
    task_id: str | None = None
    run_id: str | None = None
    invocation_id: str | None = None
    span_id: str | None = None

    @property
    def duration_seconds(self) -> float | None:
        if self.end_time is None:
            return None
        return self.end_time - self.start_time

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for field in ("task_id", "run_id", "invocation_id", "span_id"):
            if payload[field] is None:
                del payload[field]
        payload["duration_seconds"] = self.duration_seconds
        return payload


@dataclass
class InvocationResult:
    """Result of a plugin invocation with tracing info."""

    result: Any
    trace: PluginTrace
    error: Exception | None = None
