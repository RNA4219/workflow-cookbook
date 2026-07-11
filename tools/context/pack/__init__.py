# SPDX-License-Identifier: MIT
# Copyright 2025 RNA4219

"""
Context pack package.

Generates context packs with PPR scoring.
"""

from __future__ import annotations

from .cli import (
    ContextPackExecutor,
    ContextPackPlanner,
    load_config,
    main,
    pack_graph,
)
from .compression import (
    PackMetricsBuilder,
    SectionSelector,
    _diversity_penalty,
    assemble_sections,
)
from .resolver import (
    CandidateSelector,
    GraphViewBuilder,
    PPRRanker,
    _as_mapping,
    _candidate_ids,
    _config_float,
    _config_int,
    _diff_hit,
    _hub_scores,
    _intent_hit,
    _recency_score,
    _role_weight,
    _token_budget,
    _token_set,
    build_graph_view,
    personalize_scores,
    score_candidates,
)
from .types import (
    DEFAULT_CONFIG,
    AssemblyResult,
    BaseSignals,
    CandidateRanking,
    ConfigDict,
    ContextPackPlan,
    GraphEdge,
    GraphNode,
    GraphView,
    IntentProfile,
    PackMetrics,
    PackOutput,
    SectionEntry,
    SectionSelection,
    SectionWhy,
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