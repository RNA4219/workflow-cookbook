# SPDX-License-Identifier: MIT
# Copyright 2025 RNA4219

"""Graph normalization and base signal calculation for context pack."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import cast

from .config import _as_mapping, _config_float, _config_int
from .types import BaseSignals, GraphEdge, GraphNode, GraphView, IntentProfile


def _token_set(*parts: str) -> set[str]:
    tokens: set[str] = set()
    for part in parts:
        for piece in part.replace("/", " ").replace("-", " ").split():
            clean = "".join(ch for ch in piece.lower() if ch.isalnum())
            if clean:
                tokens.add(clean)
    return tokens


def _intent_profile(intent: str, halflife: int) -> dict[str, object]:
    tokens = _token_set(intent)
    role = next((role for role in ["impl", "ops", "risk", "spec"] if role in tokens), None)
    keywords = sorted(tokens - {"int", "intent"})
    return {"keywords": keywords, "role": role, "halflife": halflife}


def _recency_score(iso_ts: str, halflife: int) -> float:
    modified_at = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
    age = max(0.0, (datetime.now(UTC) - modified_at).total_seconds() / 86400)
    return math.exp(-age / max(halflife, 1))


def _hub_scores(nodes: Sequence[GraphNode], edges: Sequence[GraphEdge]) -> dict[str, float]:
    degree: Counter[str] = Counter()
    for edge in edges:
        source = edge.get("src")
        if isinstance(source, str):
            degree[source] += 1
    max_degree = max(degree.values(), default=1)
    scores: dict[str, float] = {}
    for node in nodes:
        node_id = node.get("id")
        if not isinstance(node_id, str):
            continue
        out_degree = degree.get(node_id, 0)
        scores[node_id] = math.log1p(out_degree) / math.log1p(max_degree) if max_degree else 0.0
    return scores


def _role_weight(node_role: str | None, intent_role: str | None) -> float:
    if not intent_role:
        return 0.4
    if node_role == intent_role:
        return 0.6
    return 0.2


def _intent_hit(keywords: Sequence[str], node_tokens: set[str]) -> float:
    if not keywords:
        return 0.0
    hits = sum(1 for keyword in keywords if keyword in node_tokens)
    return hits / len(keywords)


def _diff_hit(path: str, diff_paths: Sequence[str]) -> float:
    if not diff_paths:
        return 0.0
    if path in diff_paths:
        return 1.0
    directories = {diff_path.rsplit("/", 1)[0] for diff_path in diff_paths if "/" in diff_path}
    if any(path.startswith(directory + "/") or path == directory for directory in directories):
        return 0.7
    domains = {diff_path.split("/", 1)[0] for diff_path in diff_paths}
    if any(path.startswith(domain) for domain in domains):
        return 0.4
    return 0.0


def _token_budget(node: Mapping[str, object]) -> int:
    estimate = node.get("token_estimate")
    if isinstance(estimate, (int, float)):
        return int(estimate)
    heading_value = node.get("heading")
    heading = heading_value if isinstance(heading_value, str) else ""
    return max(32, len(heading.split()) * 10)


class GraphViewBuilder:
    def __init__(
        self,
        *,
        graph: Mapping[str, object],
        intent: str,
        diff_paths: Sequence[str],
        config: Mapping[str, object],
    ) -> None:
        self._graph = graph
        self._intent = intent
        self._diff_paths = [str(path) for path in diff_paths]
        self._config = _as_mapping(config)

    def normalize_nodes(self) -> list[GraphNode]:
        nodes_raw = self._graph.get("nodes", [])
        if not isinstance(nodes_raw, list):
            return []
        return [cast("GraphNode", node) for node in nodes_raw]

    def normalize_edges(self) -> list[GraphEdge]:
        edges_raw = self._graph.get("edges", [])
        if not isinstance(edges_raw, list):
            return []
        return [cast("GraphEdge", edge) for edge in edges_raw]

    def build_intent_profile(self) -> IntentProfile:
        halflife = _config_int(self._config, "recency_halflife_days", 45)
        raw_intent = _intent_profile(self._intent, halflife)
        keywords_value = raw_intent.get("keywords", [])
        keywords = [str(item) for item in keywords_value] if isinstance(keywords_value, list) else []
        role_value = raw_intent.get("role")
        role = role_value if isinstance(role_value, str) else None
        return IntentProfile(keywords=keywords, role=role, halflife=halflife)

    def build_adjacency(
        self, edges: Sequence[GraphEdge]
    ) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
        adjacency: dict[str, list[str]] = {}
        reverse_adjacency: dict[str, list[str]] = {}
        for edge in edges:
            source = edge.get("src")
            destination = edge.get("dst")
            if isinstance(source, str) and isinstance(destination, str):
                adjacency.setdefault(source, []).append(destination)
                reverse_adjacency.setdefault(destination, []).append(source)
        return adjacency, reverse_adjacency

    def compute_base_signals(
        self,
        nodes: Sequence[GraphNode],
        edges: Sequence[GraphEdge],
        intent_profile: IntentProfile,
    ) -> tuple[dict[str, BaseSignals], dict[str, float], list[str]]:
        hub_scores = _hub_scores(nodes, edges)
        weights_map = _as_mapping(self._config.get("weights"))
        weight_intent = _config_float(weights_map, "intent", 0.0)
        weight_diff = _config_float(weights_map, "diff", 0.0)
        weight_recency = _config_float(weights_map, "recency", 0.0)
        weight_hub = _config_float(weights_map, "hub", 0.0)
        weight_role = _config_float(weights_map, "role", 0.0)
        base_signals: dict[str, BaseSignals] = {}
        base_scores: dict[str, float] = {}
        hits: list[str] = []
        for node in nodes:
            node_id = node.get("id")
            if not isinstance(node_id, str):
                continue
            heading = node.get("heading") or ""
            path = node.get("path") or ""
            tokens = _token_set(str(heading), str(path))
            intent_hit = _intent_hit(intent_profile.keywords, tokens)
            diff_hit = _diff_hit(str(path), self._diff_paths)
            mtime_value = node.get("mtime")
            modified_at = mtime_value if isinstance(mtime_value, str) else datetime.now(UTC).isoformat()
            recency = _recency_score(modified_at, intent_profile.halflife)
            hub = hub_scores.get(node_id, 0.0)
            role_value = node.get("role")
            role_score = _role_weight(
                role_value if isinstance(role_value, str) else None,
                intent_profile.role,
            )
            signals = BaseSignals(intent_hit, diff_hit, recency, hub, role_score)
            base_signals[node_id] = signals
            if intent_hit > 0 or diff_hit > 0:
                hits.append(node_id)
            base_scores[node_id] = (
                signals.intent * weight_intent
                + signals.diff * weight_diff
                + signals.recency * weight_recency
                + signals.hub * weight_hub
                + signals.role * weight_role
            )
        return base_signals, base_scores, hits

    def build(self) -> GraphView:
        nodes = self.normalize_nodes()
        edges = self.normalize_edges()
        intent_profile = self.build_intent_profile()
        adjacency, reverse_adjacency = self.build_adjacency(edges)
        base_signals, base_scores, hits = self.compute_base_signals(nodes, edges, intent_profile)
        return GraphView(
            nodes=nodes,
            edges=edges,
            intent_profile=intent_profile,
            adjacency=adjacency,
            reverse_adjacency=reverse_adjacency,
            base_signals=base_signals,
            base_scores=base_scores,
            hits=hits,
        )
