# SPDX-License-Identifier: MIT
# Copyright 2025 RNA4219

"""
Session management for Birdseye update.

Provides timestamp handling, serial allocation, and update session.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .types import (
    CapsuleState,
    UpdateOptions,
    BirdseyePlan,
    PlannedWrite,
    BirdseyeRootPlan,
)
from .constants import _REPO_ROOT


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


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


_SERIAL_PATTERN = re.compile(r"\d{05}")


def _coerce_serial(candidate: Any) -> int | None:
    if isinstance(candidate, str) and _SERIAL_PATTERN.fullmatch(candidate):
        return int(candidate)
    return None


class _SerialAllocator:
    __slots__ = ("max_serial", "_next_serial")

    def __init__(self) -> None:
        self.max_serial = 0
        self._next_serial: int | None = None

    def observe(self, candidate: Any) -> None:
        value = _coerce_serial(candidate)
        if value is not None and value > self.max_serial:
            self.max_serial = value

    def allocate(self, existing: Any) -> str:
        candidate = _coerce_serial(existing)
        if candidate is not None and candidate > self.max_serial:
            self.max_serial = candidate
        if self._next_serial is None:
            self._next_serial = self.max_serial + 1 if self.max_serial else 1
        target = self._next_serial
        if target > self.max_serial:
            self.max_serial = target
        return f"{target:05d}"


def next_generated_at(existing: Any, fallback: str, *, allocator: _SerialAllocator) -> str:
    del fallback
    return allocator.allocate(existing)


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

    def execute(self, plan: BirdseyePlan) -> "UpdateReport":
        from .types import UpdateReport
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

    def _resolve_focus_nodes(
        self,
        root_targets: Sequence[Path],
        root: Path,
        graph_out: Mapping[str, Sequence[str]],
        graph_in: Mapping[str, Sequence[str]],
        caps_state: CapsuleState,
        cap_path_lookup: Mapping[Path, str],
        available_caps: Mapping[str, Path],
    ) -> set[str]:
        return self.focus_resolver.resolve(
            root_targets=root_targets,
            root=root,
            graph_out=graph_out,
            graph_in=graph_in,
            caps_state=caps_state,
            cap_path_lookup=cap_path_lookup,
            available_caps=available_caps,
        )

    def _plan_hot(
        self,
        hot_path: Path,
        hot_data: dict[str, Any],
        hot_original: str,
    ) -> None:
        new_generated = next_generated_at(
            hot_data.get("generated_at"),
            self.timestamp,
            allocator=self.serial_allocator,
        )
        if hot_data.get("generated_at") != new_generated:
            hot_data["generated_at"] = new_generated
            self._remember_generated(new_generated)
        serialized = _dump_json(hot_data)
        if serialized != hot_original:
            self._writes.append(
                PlannedWrite(path=hot_path, content=serialized, original=hot_original)
            )

    def _plan_capsule(
        self,
        cap_id: str,
        capsule: tuple[Path, dict[str, Any], str],
        graph_out: Mapping[str, Sequence[str]],
        graph_in: Mapping[str, Sequence[str]],
    ) -> None:
        cap_path, cap_data, cap_original = capsule
        expected_out = _sorted_unique(graph_out.get(cap_id, []))
        expected_in = _sorted_unique(graph_in.get(cap_id, []))
        updated = False
        if cap_data.get("deps_out") != expected_out:
            cap_data["deps_out"] = expected_out
            updated = True
        if cap_data.get("deps_in") != expected_in:
            cap_data["deps_in"] = expected_in
            updated = True
        new_generated = next_generated_at(
            cap_data.get("generated_at"),
            self.timestamp,
            allocator=self.serial_allocator,
        )
        if cap_data.get("generated_at") != new_generated:
            cap_data["generated_at"] = new_generated
            updated = True
            self._remember_generated(new_generated)
        if updated:
            serialized = _dump_json(cap_data)
            if serialized != cap_original:
                self._writes.append(
                    PlannedWrite(path=cap_path, content=serialized, original=cap_original)
                )

    def _remember_generated(self, value: str) -> None:
        if self._generated_at is None:
            self._generated_at = value


def run_update(options: UpdateOptions) -> "UpdateReport":
    timestamp = _format_timestamp(utc_now())
    session = BirdseyeUpdateSession(options=options, timestamp=timestamp)
    plan = session.plan()
    return session.execute(plan)