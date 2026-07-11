# SPDX-License-Identifier: MIT
# Copyright 2025 RNA4219

"""
Types and data classes for Birdseye update.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from .session import BirdseyeUpdateSession


class TargetResolutionError(RuntimeError):
    """Raised when Birdseye targets cannot be resolved."""


class DiffResolver(Protocol):
    def resolve(self, reference: str) -> tuple[Path, ...]:
        ...


CapsuleEntry = tuple[Path, dict[str, Any], str]
CapsuleState = dict[str, CapsuleEntry]
Graph = dict[str, list[str]]


@dataclass(frozen=True)
class UpdateOptions:
    targets: tuple[Path, ...]
    emit: str
    dry_run: bool = False
    since: str | None = None
    diff_resolver: DiffResolver | None = None
    radius: int = 2

    def resolve_targets(self) -> tuple[Path, ...]:
        from .session import _default_birdseye_targets, _normalise_target
        resolved = [_normalise_target(path) for path in self.targets]
        if self.since:
            if self.diff_resolver is None:
                raise TargetResolutionError("Diff resolver is required when --since is used")
            import subprocess
            try:
                derived = self.diff_resolver.resolve(self.since)
            except subprocess.CalledProcessError as exc:
                raise TargetResolutionError(
                    f"Failed to resolve git diff for --since: {exc}"
                ) from exc
            resolved.extend(_normalise_target(path) for path in derived)
        if not resolved:
            resolved.extend(_default_birdseye_targets())
        unique_targets = tuple(dict.fromkeys(resolved))
        if not unique_targets:
            raise TargetResolutionError("Specify --targets, --since, or both")
        return unique_targets


@dataclass(frozen=True)
class UpdateReport:
    generated_at: str
    planned_writes: tuple[Path, ...]
    performed_writes: tuple[Path, ...]


@dataclass(frozen=True)
class PlannedWrite:
    path: Path
    content: str
    original: str


@dataclass(frozen=True)
class BirdseyePlan:
    generated_at: str
    writes: tuple[PlannedWrite, ...]


@dataclass(frozen=True)
class BirdseyeRootPlan:
    loads: tuple[str, ...]
    writes: tuple[PlannedWrite, ...]
    remembered: str | None = None

    def apply(self, session: BirdseyeUpdateSession) -> None:
        session._writes.extend(self.writes)
        if self.remembered is not None:
            session._remember_generated(self.remembered)