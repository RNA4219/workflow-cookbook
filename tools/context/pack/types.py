# SPDX-License-Identifier: MIT
# Copyright 2025 RNA4219

"""
Types and data classes for context pack.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import (
    TypedDict,
)


class GraphNode(TypedDict, total=False):
    id: str
    path: str
    heading: str
    depth: int
    mtime: str
    token_estimate: int
    role: str


class GraphEdge(TypedDict, total=False):
    src: str
    dst: str
    type: str


ConfigDict = dict[str, object]


class SectionWhy(TypedDict):
    intent: float
    diff: float
    recency: float
    hub: float
    role: float
    ppr: float
    score: float


class SectionEntry(TypedDict):
    id: str
    tok: int
    filters: list[str]
    why: SectionWhy


class PackMetrics(TypedDict):
    token_in: int
    token_src: int
    dup_rate: float
    ppr_entropy: float
    diversity_penalty: float


class PackOutput(TypedDict):
    intent: str
    budget: str
    sections: list[SectionEntry]
    metrics: PackMetrics


DEFAULT_CONFIG: ConfigDict = {
    "pagerank": {"lambda": 0.85, "theta": 0.6},
    "weights": {"intent": 0.40, "diff": 0.25, "recency": 0.20, "hub": 0.10, "role": 0.05},
    "recency_halflife_days": 45,
    "diversity": {"mu_file": 0.15, "mu_role": 0.10},
    "limits": {"ncand": 2000, "iters": 50, "tol": 1e-6},
}


@dataclass(frozen=True)
class IntentProfile:
    keywords: list[str]
    role: str | None
    halflife: int


@dataclass(frozen=True)
class BaseSignals:
    intent: float
    diff: float
    recency: float
    hub: float
    role: float


@dataclass
class GraphView:
    nodes: Sequence[GraphNode]
    edges: Sequence[GraphEdge]
    intent_profile: IntentProfile
    adjacency: dict[str, list[str]]
    reverse_adjacency: dict[str, list[str]]
    base_signals: dict[str, BaseSignals]
    base_scores: dict[str, float]
    hits: list[str]


@dataclass
class CandidateRanking:
    candidate_nodes: list[GraphNode]
    ppr_scores: dict[str, float]
    scores: dict[str, float]
    ranked_nodes: list[GraphNode]


@dataclass(frozen=True)
class SectionSelection:
    sections: list[SectionEntry]
    token_in: int
    penalties: list[float]


@dataclass
class AssemblyResult:
    sections: list[SectionEntry]
    metrics: PackMetrics


@dataclass(frozen=True)
class ContextPackPlan:
    intent: str
    budget_tokens: int
    diff_paths: Sequence[str]
    graph: Mapping[str, object]
    config: Mapping[str, object]
    view: GraphView
    ranking: CandidateRanking
    assembly: AssemblyResult

    @property
    def target_candidates(self) -> list[str]:
        candidates: list[str] = []
        for node in self.ranking.ranked_nodes:
            node_id = node.get("id")
            if isinstance(node_id, str):
                candidates.append(node_id)
        return candidates

    @property
    def budget_remaining(self) -> int:
        consumed = int(self.assembly.metrics["token_in"])
        remaining = self.budget_tokens - consumed
        return remaining if remaining >= 0 else 0