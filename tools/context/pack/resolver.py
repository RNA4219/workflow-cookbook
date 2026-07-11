# SPDX-License-Identifier: MIT
# Copyright 2025 RNA4219

"""
Graph resolution and candidate selection for context pack.
"""

from __future__ import annotations

import math
from collections import Counter, deque
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from typing import cast

from .types import (
    BaseSignals,
    CandidateRanking,
    GraphEdge,
    GraphNode,
    GraphView,
    IntentProfile,
)


def _as_mapping(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return cast("Mapping[str, object]", value)
    return {}


def _config_int(config: Mapping[str, object], key: str, default: int) -> int:
    value = config.get(key, default)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return default
    return default


def _config_float(config: Mapping[str, object], key: str, default: float) -> float:
    value = config.get(key, default)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


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
    role = next((r for r in ["impl", "ops", "risk", "spec"] if r in tokens), None)
    keywords = sorted(tokens - {"int", "intent"})
    return {"keywords": keywords, "role": role, "halflife": halflife}


def _recency_score(iso_ts: str, halflife: int) -> float:
    mt = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
    age = max(0.0, (datetime.now(UTC) - mt).total_seconds() / 86400)
    return math.exp(-age / max(halflife, 1))


def _hub_scores(nodes: Sequence[GraphNode], edges: Sequence[GraphEdge]) -> dict[str, float]:
    degree: Counter[str] = Counter()
    for edge in edges:
        src = edge.get("src")
        if isinstance(src, str):
            degree[src] += 1
    max_deg = max(degree.values(), default=1)
    result: dict[str, float] = {}
    for node in nodes:
        node_id = node.get("id")
        if not isinstance(node_id, str):
            continue
        out_deg = degree.get(node_id, 0)
        result[node_id] = math.log1p(out_deg) / math.log1p(max_deg) if max_deg else 0.0
    return result


def _role_weight(node_role: str | None, intent_role: str | None) -> float:
    if not intent_role:
        return 0.4
    if node_role == intent_role:
        return 0.6
    return 0.2


def _intent_hit(keywords: Sequence[str], node_tokens: set[str]) -> float:
    if not keywords:
        return 0.0
    hits = sum(1 for kw in keywords if kw in node_tokens)
    return hits / len(keywords)


def _diff_hit(path: str, diff_paths: Sequence[str]) -> float:
    if not diff_paths:
        return 0.0
    if path in diff_paths:
        return 1.0
    directories = {d.rsplit("/", 1)[0] for d in diff_paths if "/" in d}
    if any(path.startswith(dir_path + "/") or path == dir_path for dir_path in directories):
        return 0.7
    domains = {d.split("/", 1)[0] for d in diff_paths}
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


def _candidate_ids(
    hits: Iterable[str],
    adjacency: dict[str, list[str]],
    reverse_adj: dict[str, list[str]],
    max_hops: int,
) -> set[str]:
    seen: set[str] = set()
    queue = deque((node, 0) for node in hits)
    while queue:
        node, dist = queue.popleft()
        if node in seen or dist > max_hops:
            continue
        seen.add(node)
        for nxt in adjacency.get(node, []):
            queue.append((nxt, dist + 1))
        for nxt in reverse_adj.get(node, []):
            queue.append((nxt, dist + 1))
    return seen


def personalize_scores(
    nodes: Sequence[GraphNode],
    edges: Sequence[GraphEdge],
    base_scores: Mapping[str, float],
    lam: float,
    iters: int,
    tol: float,
) -> dict[str, float]:
    indexed_nodes: list[GraphNode] = []
    node_ids: list[str] = []
    for node in nodes:
        node_id = node.get("id")
        if isinstance(node_id, str):
            indexed_nodes.append(node)
            node_ids.append(node_id)
    n = len(indexed_nodes)
    if n == 0:
        return {}
    id_to_index = {node_id: idx for idx, node_id in enumerate(node_ids)}
    adjacency: list[list[int]] = [[] for _ in range(n)]
    outdeg = [0] * n
    for edge in edges:
        src = edge.get("src")
        dst = edge.get("dst")
        if not (isinstance(src, str) and isinstance(dst, str)):
            continue
        src_idx = id_to_index.get(src)
        dst_idx = id_to_index.get(dst)
        if src_idx is None or dst_idx is None:
            continue
        adjacency[src_idx].append(dst_idx)
        outdeg[src_idx] += 1
    teleport = [max(base_scores.get(node_ids[i], 0.0), 0.0) for i in range(n)]
    total = sum(teleport) or 1.0
    teleport = [value / total for value in teleport]
    state = [1.0 / n] * n
    for _ in range(iters):
        updated = [(1 - lam) * teleport[i] for i in range(n)]
        for i, row in enumerate(adjacency):
            share = lam * state[i] / (outdeg[i] or n)
            if outdeg[i] == 0:
                for j in range(n):
                    updated[j] += share
            else:
                for j in row:
                    updated[j] += share
        delta = sum(abs(updated[i] - state[i]) for i in range(n))
        state = updated
        if delta < tol:
            break
    normaliser = sum(state) or 1.0
    return {node_ids[i]: state[i] / normaliser for i in range(n)}


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
        keywords = (
            [str(item) for item in keywords_value]
            if isinstance(keywords_value, list)
            else []
        )
        role_value = raw_intent.get("role")
        role = role_value if isinstance(role_value, str) else None
        return IntentProfile(keywords=keywords, role=role, halflife=halflife)

    def build_adjacency(
        self, edges: Sequence[GraphEdge]
    ) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
        adjacency: dict[str, list[str]] = {}
        reverse_adj: dict[str, list[str]] = {}
        for edge in edges:
            src = edge.get("src")
            dst = edge.get("dst")
            if isinstance(src, str) and isinstance(dst, str):
                adjacency.setdefault(src, []).append(dst)
                reverse_adj.setdefault(dst, []).append(src)
        return adjacency, reverse_adj

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
            mtime = (
                mtime_value
                if isinstance(mtime_value, str)
                else datetime.now(UTC).isoformat()
            )
            recency = _recency_score(mtime, intent_profile.halflife)
            hub = hub_scores.get(node_id, 0.0)
            role_value_node = node.get("role")
            role_score = _role_weight(
                role_value_node if isinstance(role_value_node, str) else None,
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
        adjacency, reverse_adj = self.build_adjacency(edges)
        base_signals, base_scores, hits = self.compute_base_signals(
            nodes, edges, intent_profile
        )
        return GraphView(
            nodes=nodes,
            edges=edges,
            intent_profile=intent_profile,
            adjacency=adjacency,
            reverse_adjacency=reverse_adj,
            base_signals=base_signals,
            base_scores=base_scores,
            hits=hits,
        )


class CandidateSelector:
    view: GraphView
    config: Mapping[str, object]

    def __init__(self, view: GraphView, config: Mapping[str, object]) -> None:
        self.view = view
        self.config = config

    def select(self) -> list[GraphNode]:
        candidate_ids = self._collect_candidate_ids()
        sorted_candidates = sorted(
            candidate_ids,
            key=lambda node_id: self.view.base_scores.get(node_id, 0.0),
            reverse=True,
        )
        limits = _as_mapping(self.config.get("limits"))
        ncand = _config_int(limits, "ncand", 2000)
        selected_ids = set(sorted_candidates[:ncand])
        candidate_nodes = self._filter_nodes(selected_ids if selected_ids else None)
        if not candidate_nodes:
            candidate_nodes = self._filter_nodes(None)
        return candidate_nodes

    def _collect_candidate_ids(self) -> set[str]:
        if self.view.hits:
            return _candidate_ids(
                self.view.hits,
                self.view.adjacency,
                self.view.reverse_adjacency,
                2,
            )
        return set(self.view.base_scores.keys())

    def _filter_nodes(self, allowed: set[str] | None) -> list[GraphNode]:
        result: list[GraphNode] = []
        for node in self.view.nodes:
            node_id = node.get("id")
            if not isinstance(node_id, str):
                continue
            if allowed is None or node_id in allowed:
                result.append(node)
        return result


class PPRRanker:
    view: GraphView
    config: Mapping[str, object]
    candidate_nodes: Sequence[GraphNode]

    def __init__(
        self,
        view: GraphView,
        config: Mapping[str, object],
        candidate_nodes: Sequence[GraphNode],
    ) -> None:
        self.view = view
        self.config = config
        self.candidate_nodes = candidate_nodes

    def rank(self) -> CandidateRanking:
        candidates = list(self.candidate_nodes)
        pagerank_cfg = _as_mapping(self.config.get("pagerank"))
        limits = _as_mapping(self.config.get("limits"))
        ppr_scores = personalize_scores(
            candidates,
            self.view.edges,
            self.view.base_scores,
            _config_float(pagerank_cfg, "lambda", 0.85),
            _config_int(limits, "iters", 50),
            _config_float(limits, "tol", 1e-6),
        )
        theta = _config_float(pagerank_cfg, "theta", 0.6)
        scores: dict[str, float] = {}
        for node in candidates:
            node_id = node.get("id")
            if not isinstance(node_id, str):
                continue
            base_score = self.view.base_scores.get(node_id, 0.0)
            ppr_score = ppr_scores.get(node_id, 0.0)
            scores[node_id] = theta * ppr_score + (1 - theta) * base_score

        def node_score(node: GraphNode) -> float:
            node_id = node.get("id")
            if isinstance(node_id, str):
                return scores.get(node_id, 0.0)
            return 0.0

        ranked_nodes = sorted(candidates, key=node_score, reverse=True)
        if not ranked_nodes:
            ranked_nodes = candidates
        return CandidateRanking(
            candidate_nodes=candidates,
            ppr_scores=ppr_scores,
            scores=scores,
            ranked_nodes=ranked_nodes,
        )


def build_graph_view(
    graph: Mapping[str, object],
    intent: str,
    diff_paths: Sequence[str],
    config: Mapping[str, object],
) -> GraphView:
    builder = GraphViewBuilder(
        graph=graph,
        intent=intent,
        diff_paths=diff_paths,
        config=config,
    )
    return builder.build()


def score_candidates(
    view: GraphView,
    config: Mapping[str, object],
) -> CandidateRanking:
    candidate_nodes = CandidateSelector(view=view, config=config).select()
    return PPRRanker(
        view=view,
        config=config,
        candidate_nodes=candidate_nodes,
    ).rank()