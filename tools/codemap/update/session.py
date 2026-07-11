# SPDX-License-Identifier: MIT
# Copyright 2025 RNA4219

"""
Session management for Birdseye update.

Provides timestamp handling, serial allocation, and update session.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .constants import _REPO_ROOT
from .serial import _SerialAllocator
from .types import (
    BirdseyePlan,
    PlannedWrite,
    UpdateOptions,
    UpdateReport,
)


def utc_now() -> datetime:
    return datetime.now(UTC)


def _format_timestamp(moment: datetime) -> str:
    return moment.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _resolve_root(target: Path) -> Path:
    resolved = target.resolve()
    if resolved.is_dir():
        if resolved.name == "caps":
            return resolved.parent.resolve()
        return resolved
    parent = resolved.parent
    if parent.name == "caps":
        return parent.parent.resolve()
    return parent.resolve()


def _load_json(path: Path) -> tuple[Any, str]:
    raw = path.read_text(encoding="utf-8")
    import json
    return json.loads(raw), raw


def _dump_json(data: Any) -> str:
    import json
    return json.dumps(data, ensure_ascii=False, indent=2) + "\n"


def _sorted_unique(items: Sequence[str]) -> list[str]:
    return sorted(dict.fromkeys(items))


def _finalise(paths: set[Path]) -> tuple[Path, ...]:
    return tuple(sorted(paths, key=lambda candidate: candidate.as_posix()))


def _group_targets(targets: Iterable[Path]) -> dict[Path, list[Path]]:
    grouped: dict[Path, list[Path]] = {}
    for target in targets:
        normalised = _normalise_target(target)
        root = _resolve_root(normalised)
        grouped.setdefault(root, []).append(normalised)
    return grouped


def _default_birdseye_targets() -> tuple[Path, ...]:
    birdseye_root = _REPO_ROOT.get() / "docs" / "birdseye"
    candidates = (
        birdseye_root / "index.json",
        birdseye_root / "hot.json",
        birdseye_root / "caps",
    )
    fallback: list[Path] = []
    for candidate in candidates:
        if candidate.is_file() or candidate.is_dir():
            fallback.append(candidate.resolve())
    return tuple(fallback)


def _normalise_target(target: Path) -> Path:
    if target.is_absolute():
        return target.resolve()
    return (_REPO_ROOT.get() / target).resolve()



class BirdseyeUpdateSession:
    def __init__(self, *, options: UpdateOptions, timestamp: str | None = None) -> None:
        self.options = options
        self.emit_index = options.emit in {"index", "index+caps"}
        self.emit_caps = options.emit in {"caps", "index+caps"}
        self.timestamp = timestamp or _format_timestamp(utc_now())
        self.serial_allocator = _SerialAllocator()
        from .graph import BirdseyeFocusResolver
        self.focus_resolver = BirdseyeFocusResolver(radius=options.radius)
        self._generated_at: str | None = None
        self._writes: list[PlannedWrite] = []

    def plan(self) -> BirdseyePlan:
        resolved_targets = self.options.resolve_targets()
        grouped = _group_targets(resolved_targets)
        for root, root_targets in grouped.items():
            self._plan_for_root(root, root_targets)
        return BirdseyePlan(
            generated_at=self._generated_at or self.timestamp,
            writes=tuple(self._writes),
        )

    def execute(self, plan: BirdseyePlan) -> UpdateReport:
        planned_paths = {write.path for write in plan.writes}
        performed: set[Path] = set()
        if not self.options.dry_run:
            for write in plan.writes:
                write.path.parent.mkdir(parents=True, exist_ok=True)
                write.path.write_text(write.content, encoding='utf-8')
                performed.add(write.path)
        return UpdateReport(
            generated_at=plan.generated_at,
            planned_writes=_finalise(planned_paths),
            performed_writes=_finalise(performed),
        )

    def _plan_for_root(self, root: Path, root_targets: Sequence[Path]) -> None:
        from .capsule import BirdseyeRootBuilder
        builder = BirdseyeRootBuilder(
            root=root,
            root_targets=root_targets,
            emit_index=self.emit_index,
            emit_caps=self.emit_caps,
            timestamp=self.timestamp,
            serial_allocator=self.serial_allocator,
            focus_resolver=self.focus_resolver,
        )
        plan = builder.build()
        plan.apply(self)


    def _remember_generated(self, value: str) -> None:
        if self._generated_at is None:
            self._generated_at = value


def run_update(options: UpdateOptions) -> UpdateReport:
    timestamp = _format_timestamp(utc_now())
    session = BirdseyeUpdateSession(options=options, timestamp=timestamp)
    plan = session.plan()
    return session.execute(plan)