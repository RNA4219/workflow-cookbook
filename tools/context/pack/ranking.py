# SPDX-License-Identifier: MIT
# Copyright 2025 RNA4219

"""Candidate discovery and personalized PageRank for context pack."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping, Sequence

from .config import _as_mapping, _config_float, _config_int
from .types import CandidateRanking, GraphEdge, GraphNode, GraphView


def _candidate_ids(
    hits: Iterable[str],
    adjacency: dict[str, list[str]],
    reverse_adj: dict[str, list[str]],
    max_hops: int,
) -> set[str]:
    seen: set[str] = set()
    queue = deque((node, 0) for node in hits)
    while queue:
        node, distance = queue.popleft()
        if node in seen or distance > max_hops:
            continue
        seen.add(node)
        for neighbour in adjacency.get(node, []):
            queue.append((neighbour, distance + 1))
        for neighbour in reverse_adj.get(node, []):
            queue.append((neighbour, distance + 1))
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
    count = len(indexed_nodes)
    if count == 0:
        return {}
    id_to_index = {node_id: index for index, node_id in enumerate(node_ids)}
    adjacency: list[list[int]] = [[] for _ in range(count)]
    out_degrees = [0] * count
    for edge in edges:
        source = edge.get("src")
        destination = edge.get("dst")
        if not (isinstance(source, str) and isinstance(destination, str)):
            continue
        source_index = id_to_index.get(source)
        destination_index = id_to_index.get(destination)
        if source_index is None or destination_index is None:
            continue
        adjacency[source_index].append(destination_index)
        out_degrees[source_index] += 1
    teleport = [max(base_scores.get(node_ids[index], 0.0), 0.0) for index in range(count)]
    total = sum(teleport) or 1.0
    teleport = [value / total for value in teleport]
    state = [1.0 / count] * count
    for _ in range(iters):
        updated = [(1 - lam) * teleport[index] for index in range(count)]
        for index, row in enumerate(adjacency):
            share = lam * state[index] / (out_degrees[index] or count)
            if out_degrees[index] == 0:
                for target_index in range(count):
                    updated[target_index] += share
            else:
                for target_index in row:
                    updated[target_index] += share
        delta = sum(abs(updated[index] - state[index]) for index in range(count))
        state = updated
        if delta < tol:
            break
    normaliser = sum(state) or 1.0
    return {node_ids[index]: state[index] / normaliser for index in range(count)}

class CandidateSelector:
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
        candidate_limit = _config_int(limits, "ncand", 2000)
        selected_ids = set(sorted_candidates[:candidate_limit])
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
