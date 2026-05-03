# SPDX-License-Identifier: MIT
# Copyright 2025 RNA4219

"""
Context pack package.

Generates context packs with PPR scoring.
"""

from __future__ import annotations

from .types import (
    GraphNode,
    GraphEdge,
    ConfigDict,
    SectionWhy,
    SectionEntry,
    PackMetrics,
    PackOutput,
    DEFAULT_CONFIG,
    IntentProfile,
    BaseSignals,
    GraphView,
    CandidateRanking,
    SectionSelection,
    AssemblyResult,
    ContextPackPlan,
)
from .resolver import (
    build_graph_view,
    score_candidates,
    personalize_scores,
    GraphViewBuilder,
    CandidateSelector,
    PPRRanker,
    _recency_score,
    _token_budget,
    _intent_hit,
    _diff_hit,
    _hub_scores,
    _role_weight,
    _candidate_ids,
    _token_set,
    _config_int,
    _config_float,
    _as_mapping,
)
from .compression import (
    assemble_sections,
    SectionSelector,
    PackMetricsBuilder,
    _diversity_penalty,
)
from .cli import (
    load_config,
    pack_graph,
    ContextPackPlanner,
    ContextPackExecutor,
    main,
)

__all__ = [
    "GraphNode",
    "GraphEdge",
    "ConfigDict",
    "SectionWhy",
    "SectionEntry",
    "PackMetrics",
    "PackOutput",
    "DEFAULT_CONFIG",
    "IntentProfile",
    "BaseSignals",
    "GraphView",
    "CandidateRanking",
    "SectionSelection",
    "AssemblyResult",
    "ContextPackPlan",
    "build_graph_view",
    "score_candidates",
    "personalize_scores",
    "assemble_sections",
    "GraphViewBuilder",
    "CandidateSelector",
    "PPRRanker",
    "SectionSelector",
    "PackMetricsBuilder",
    "load_config",
    "pack_graph",
    "ContextPackPlanner",
    "ContextPackExecutor",
    "main",
    # Internal exports for tests
    "_recency_score",
    "_token_budget",
    "_diversity_penalty",
    "_intent_hit",
    "_diff_hit",
    "_hub_scores",
    "_role_weight",
    "_candidate_ids",
    "_token_set",
    "_config_int",
    "_config_float",
    "_as_mapping",
]