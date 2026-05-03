# SPDX-License-Identifier: MIT
# Copyright 2025 RNA4219

"""
Graph operations for Birdseye.

Builds dependency graphs and resolves focus nodes.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Mapping, Sequence

from .types import CapsuleState, Graph


def build_graph(index_data: Mapping[str, Any]) -> tuple[Graph, Graph]:
    """Build outgoing and incoming dependency graphs from index data."""
    raw_nodes = index_data.get("nodes", {})
    if not isinstance(raw_nodes, Mapping):
        raw_nodes = {}
    graph_out: Graph = {node: [] for node in raw_nodes if isinstance(node, str)}
    graph_in: Graph = {node: [] for node in raw_nodes if isinstance(node, str)}
    for raw_edge in index_data.get("edges", []):
        if not isinstance(raw_edge, Sequence) or len(raw_edge) != 2:
            continue
        source, destination = raw_edge
        if not isinstance(source, str) or not isinstance(destination, str):
            continue
        graph_out.setdefault(source, []).append(destination)
        graph_in.setdefault(destination, []).append(source)
        graph_out.setdefault(destination, graph_out.get(destination, []))
        graph_in.setdefault(source, graph_in.get(source, []))
    for values in graph_out.values():
        values.sort()
    for values in graph_in.values():
        values.sort()
    return graph_out, graph_in


class BirdseyeFocusResolver:
    """Resolve focused nodes within radius from targets."""

    def __init__(self, *, radius: int = 2) -> None:
        self.radius = radius

    def resolve(
        self,
        root_targets: Sequence[Path],
        root: Path,
        graph_out: Mapping[str, Sequence[str]],
        graph_in: Mapping[str, Sequence[str]],
        caps_state: CapsuleState,
        cap_path_lookup: Mapping[Path, str],
        available_caps: Mapping[str, Path],
    ) -> set[str]:
        combined_caps = self._merge_capsules(caps_state, available_caps)
        if not combined_caps:
            return set()
        if self._targets_include_root(root_targets, root):
            return set(combined_caps)
        focus_nodes = self._initial_focus_nodes(root_targets, cap_path_lookup)
        if not focus_nodes:
            focus_nodes = set(caps_state) or set(combined_caps)
        seen = self._expand_focus(focus_nodes, graph_out, graph_in)
        return {node for node in seen if node in combined_caps}

    def _merge_capsules(
        self,
        caps_state: CapsuleState,
        available_caps: Mapping[str, Path],
    ) -> dict[str, Path]:
        combined: dict[str, Path] = dict(available_caps)
        for cap_id, (cap_path, _cap_data, _cap_original) in caps_state.items():
            combined.setdefault(cap_id, cap_path)
        return combined

    def _targets_include_root(
        self, root_targets: Sequence[Path], root: Path
    ) -> bool:
        root_resolved = root.resolve()
        index_resolved = (root / "index.json").resolve()
        caps_dir_resolved = (root / "caps").resolve()
        hot_resolved = (root / "hot.json").resolve()
        special_roots = {
            root_resolved,
            index_resolved,
            caps_dir_resolved,
            hot_resolved,
        }
        for candidate in root_targets:
            resolved = candidate.resolve()
            if resolved in special_roots:
                return True
        return False

    def _initial_focus_nodes(
        self,
        root_targets: Sequence[Path],
        cap_path_lookup: Mapping[Path, str],
    ) -> set[str]:
        seeds: set[str] = set()
        for candidate in root_targets:
            resolved = candidate.resolve()
            cap_id = cap_path_lookup.get(resolved)
            if cap_id:
                seeds.add(cap_id)
        return seeds

    def _expand_focus(
        self,
        focus_nodes: set[str],
        graph_out: Mapping[str, Sequence[str]],
        graph_in: Mapping[str, Sequence[str]],
    ) -> set[str]:
        if self.radius < 0:
            return set()
        seen: set[str] = set()
        queue: deque[tuple[str, int]] = deque((node, 0) for node in focus_nodes)
        empty: Sequence[str] = ()
        while queue:
            node, distance = queue.popleft()
            if node in seen or distance > self.radius:
                continue
            seen.add(node)
            if distance == self.radius:
                continue
            for neighbour in graph_out.get(node, empty):
                if neighbour not in seen:
                    queue.append((neighbour, distance + 1))
            for neighbour in graph_in.get(node, empty):
                if neighbour not in seen:
                    queue.append((neighbour, distance + 1))
        return seen