# SPDX-License-Identifier: MIT
# Copyright 2025 RNA4219

"""
Capsule building and planning for Birdseye.

Generates capsule content and plans updates.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from .types import (
    CapsuleEntry,
    CapsuleState,
    PlannedWrite,
    BirdseyeRootPlan,
    UpdateOptions,
)
from .constants import _REPO_ROOT, _BIRDSEYE_REGENERATE_COMMAND
from .graph import build_graph, BirdseyeFocusResolver
from .session import _SerialAllocator, next_generated_at, _sorted_unique


def load_json(path: Path) -> tuple[Any, str]:
    raw = path.read_text(encoding="utf-8")
    return json.loads(raw), raw


def dump_json(data: Any) -> str:
    import json
    return json.dumps(data, ensure_ascii=False, indent=2) + "\n"


class BirdseyeRootBuilder:
    """Build update plans for a Birdseye root."""

    def __init__(
        self,
        *,
        root: Path,
        root_targets: Sequence[Path],
        emit_index: bool,
        emit_caps: bool,
        timestamp: str,
        serial_allocator: _SerialAllocator,
        focus_resolver: BirdseyeFocusResolver,
    ) -> None:
        self.root = root
        self.root_targets = root_targets
        self.emit_index = emit_index
        self.emit_caps = emit_caps
        self.timestamp = timestamp
        self.serial_allocator = serial_allocator
        self.index_path = root / "index.json"
        self.hot_path = root / "hot.json"
        self.caps_dir = root / "caps"
        self._loads: list[str] = []
        self._writes: list[PlannedWrite] = []
        self._focus_resolver = focus_resolver
        self._first_generated: str | None = None

    def build(self) -> BirdseyeRootPlan:
        self._validate()
        index_data, index_original = self._load_index()
        graph_out, graph_in = build_graph(index_data)

        hot_data: dict[str, Any] | None = None
        hot_original: str | None = None
        if self.emit_index or self.emit_caps:
            hot_data, hot_original = self._load_hot()

        caps_state: CapsuleState = {}
        cap_path_lookup: dict[Path, str] = {}
        available_caps: dict[str, Path] = {}
        if self.emit_caps:
            caps_state, cap_path_lookup, available_caps = self._load_capsules(index_data)

        if self.emit_index:
            self._plan_index(index_data, index_original)
            if hot_data is not None and hot_original is not None:
                self._plan_hot(hot_data, hot_original)

        if self.emit_caps and available_caps:
            focus_nodes = self._resolve_focus_nodes(
                graph_out,
                graph_in,
                caps_state,
                cap_path_lookup,
                available_caps,
            )
            self._ensure_placeholders(focus_nodes, caps_state, available_caps)
            for cap_id in sorted(focus_nodes):
                self._plan_capsule(cap_id, caps_state[cap_id], graph_out, graph_in)

        return BirdseyeRootPlan(
            loads=tuple(self._loads),
            writes=tuple(self._writes),
            remembered=self._first_generated,
        )

    def _validate(self) -> None:
        if not self.index_path.is_file():
            raise FileNotFoundError(self.index_path)
        if self.emit_index and not self.hot_path.exists():
            raise FileNotFoundError(
                f"{self.hot_path} is missing. Regenerate via: {_BIRDSEYE_REGENERATE_COMMAND}"
            )
        if self.emit_caps and not self.caps_dir.is_dir():
            raise FileNotFoundError(self.caps_dir)

    def _load_index(self) -> tuple[dict[str, Any], str]:
        self._loads.append("index")
        loaded_index, index_original = load_json(self.index_path)
        index_data = loaded_index if isinstance(loaded_index, dict) else {}
        self.serial_allocator.observe(index_data.get("generated_at"))
        return index_data, index_original

    def _load_hot(self) -> tuple[dict[str, Any] | None, str | None]:
        self._loads.append("hot")
        if not self.hot_path.exists():
            return None, None
        loaded_hot, hot_original = load_json(self.hot_path)
        if not isinstance(loaded_hot, dict):
            return None, None
        self.serial_allocator.observe(loaded_hot.get("generated_at"))
        return loaded_hot, hot_original

    def _load_capsules(
        self, index_data: Mapping[str, Any]
    ) -> tuple[CapsuleState, dict[Path, str], dict[str, Path]]:
        self._loads.append("caps")
        caps_state: CapsuleState = {}
        cap_path_lookup: dict[Path, str] = {}
        available_caps: dict[str, Path] = {}
        raw_nodes = index_data.get("nodes", {})
        if isinstance(raw_nodes, Mapping):
            for node_id, node_payload in raw_nodes.items():
                if not isinstance(node_id, str) or not isinstance(node_payload, Mapping):
                    continue
                caps_ref = node_payload.get("caps")
                if not isinstance(caps_ref, str) or not caps_ref:
                    continue
                candidate_path = Path(caps_ref)
                if candidate_path.is_absolute():
                    resolved_candidate = candidate_path.resolve()
                else:
                    resolved_candidate = (_REPO_ROOT.get() / candidate_path).resolve()
                available_caps.setdefault(node_id, resolved_candidate)
                cap_path_lookup.setdefault(resolved_candidate, node_id)
        for cap_path in sorted(self.caps_dir.glob("*.json")):
            if not cap_path.is_file():
                continue
            cap_data, cap_original = load_json(cap_path)
            if not isinstance(cap_data, dict):
                continue
            cap_id = cap_data.get("id")
            if not isinstance(cap_id, str):
                continue
            cap_path_resolved = cap_path.resolve()
            caps_state[cap_id] = (cap_path_resolved, cap_data, cap_original)
            cap_path_lookup[cap_path_resolved] = cap_id
            available_caps.setdefault(cap_id, cap_path_resolved)
            self.serial_allocator.observe(cap_data.get("generated_at"))
        return caps_state, cap_path_lookup, available_caps

    def _plan_index(self, index_data: dict[str, Any], index_original: str) -> None:
        new_generated = next_generated_at(
            index_data.get("generated_at"),
            self.timestamp,
            allocator=self.serial_allocator,
        )
        if index_data.get("generated_at") != new_generated:
            index_data["generated_at"] = new_generated
            self._remember_generated(new_generated)
        serialized = dump_json(index_data)
        if serialized != index_original:
            self._writes.append(
                PlannedWrite(path=self.index_path, content=serialized, original=index_original)
            )

    def _plan_hot(self, hot_data: dict[str, Any], hot_original: str) -> None:
        new_generated = next_generated_at(
            hot_data.get("generated_at"),
            self.timestamp,
            allocator=self.serial_allocator,
        )
        if hot_data.get("generated_at") != new_generated:
            hot_data["generated_at"] = new_generated
            self._remember_generated(new_generated)
        serialized = dump_json(hot_data)
        if serialized != hot_original:
            self._writes.append(
                PlannedWrite(path=self.hot_path, content=serialized, original=hot_original)
            )

    def _resolve_focus_nodes(
        self,
        graph_out: Mapping[str, Sequence[str]],
        graph_in: Mapping[str, Sequence[str]],
        caps_state: CapsuleState,
        cap_path_lookup: Mapping[Path, str],
        available_caps: Mapping[str, Path],
    ) -> set[str]:
        return self._focus_resolver.resolve(
            root_targets=self.root_targets,
            root=self.root,
            graph_out=graph_out,
            graph_in=graph_in,
            caps_state=caps_state,
            cap_path_lookup=cap_path_lookup,
            available_caps=available_caps,
        )

    def _ensure_placeholders(
        self,
        focus_nodes: Iterable[str],
        caps_state: CapsuleState,
        available_caps: Mapping[str, Path],
    ) -> None:
        for cap_id in focus_nodes:
            if cap_id in caps_state:
                continue
            cap_path = available_caps.get(cap_id)
            if cap_path is None:
                continue
            placeholder_data: dict[str, Any] = {
                "id": cap_id,
                "role": "doc",
                "public_api": [],
                "summary": cap_id,
                "deps_out": [],
                "deps_in": [],
                "risks": [],
                "tests": [],
            }
            caps_state[cap_id] = (cap_path, placeholder_data, "")

    def _plan_capsule(
        self,
        cap_id: str,
        capsule: CapsuleEntry,
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
            serialized = dump_json(cap_data)
            if serialized != cap_original:
                self._writes.append(
                    PlannedWrite(path=cap_path, content=serialized, original=cap_original)
                )

    def _remember_generated(self, value: str) -> None:
        if self._first_generated is None:
            self._first_generated = value