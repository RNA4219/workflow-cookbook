# SPDX-License-Identifier: MIT
# Copyright 2025 RNA4219

"""Compatibility facade for context pack graph resolution and ranking."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .config import _as_mapping, _config_float, _config_int
from .ranking import CandidateSelector, PPRRanker, _candidate_ids, personalize_scores
from .signals import (
    GraphViewBuilder,
    _diff_hit,
    _hub_scores,
    _intent_hit,
    _intent_profile,
    _recency_score,
    _role_weight,
    _token_budget,
    _token_set,
)
from .types import CandidateRanking, GraphView

__all__ = [
    "CandidateSelector",
    "GraphViewBuilder",
    "PPRRanker",
    "_as_mapping",
    "_candidate_ids",
    "_config_float",
    "_config_int",
    "_diff_hit",
    "_hub_scores",
    "_intent_hit",
    "_intent_profile",
    "_recency_score",
    "_role_weight",
    "_token_budget",
    "_token_set",
    "build_graph_view",
    "personalize_scores",
    "score_candidates",
]


def build_graph_view(
    graph: Mapping[str, object],
    intent: str,
    diff_paths: Sequence[str],
    config: Mapping[str, object],
) -> GraphView:
    return GraphViewBuilder(
        graph=graph,
        intent=intent,
        diff_paths=diff_paths,
        config=config,
    ).build()


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
