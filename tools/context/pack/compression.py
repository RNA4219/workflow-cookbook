# SPDX-License-Identifier: MIT
# Copyright 2025 RNA4219

"""
Section selection and metrics for context pack compression.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Mapping, Sequence

from .types import (
    GraphNode,
    GraphView,
    CandidateRanking,
    SectionEntry,
    SectionSelection,
    PackMetrics,
    AssemblyResult,
)
from .resolver import _as_mapping, _config_float, _token_budget


def _diversity_penalty(
    path: str,
    role: str | None,
    path_counter: Counter[str],
    role_counter: Counter[str],
    total_selected: int,
    mu_file: float,
    mu_role: float,
) -> float:
    if total_selected == 0:
        return 1.0
    file_ratio = path_counter[path] / total_selected if total_selected else 0.0
    role_ratio = role_counter[role or "unknown"] / total_selected if total_selected else 0.0
    penalty = 1.0 - (mu_file * file_ratio + mu_role * role_ratio)
    return max(0.1, penalty)


class SectionSelector:
    def __init__(
        self,
        *,
        view: GraphView,
        ranking: CandidateRanking,
        budget_tokens: int,
        config: Mapping[str, object],
    ) -> None:
        self._view = view
        self._ranking = ranking
        self._budget_tokens = budget_tokens
        diversity_cfg = _as_mapping(config.get("diversity"))
        self._mu_file = _config_float(diversity_cfg, "mu_file", 0.15)
        self._mu_role = _config_float(diversity_cfg, "mu_role", 0.10)
        self._path_counter: Counter[str] = Counter()
        self._role_counter: Counter[str] = Counter()

    def select(self) -> SectionSelection:
        sections: List[SectionEntry] = []
        penalties: List[float] = []
        token_in = 0
        for node in self._ranking.ranked_nodes:
            node_id = node.get("id")
            if not isinstance(node_id, str):
                continue
            tokens = _token_budget(node)
            if token_in + tokens > self._budget_tokens:
                continue
            path_value = node.get("path")
            path_str = path_value if isinstance(path_value, str) else ""
            role_value = node.get("role")
            role_str = role_value if isinstance(role_value, str) else None
            penalty = _diversity_penalty(
                path_str,
                role_str,
                self._path_counter,
                self._role_counter,
                len(sections),
                self._mu_file,
                self._mu_role,
            )
            penalties.append(penalty)
            signals = self._view.base_signals[node_id]
            adjusted = self._ranking.scores.get(node_id, 0.0) * penalty
            section: SectionEntry = {
                "id": node_id,
                "tok": tokens,
                "filters": ["lossless", "pointer", "role_extract"],
                "why": {
                    "intent": signals.intent,
                    "diff": signals.diff,
                    "recency": signals.recency,
                    "hub": signals.hub,
                    "role": signals.role,
                    "ppr": self._ranking.ppr_scores.get(node_id, 0.0),
                    "score": adjusted,
                },
            }
            sections.append(section)
            token_in += tokens
            self._path_counter[path_str] += 1
            self._role_counter[role_str or "unknown"] += 1
        return SectionSelection(sections=sections, token_in=token_in, penalties=penalties)


class PackMetricsBuilder:
    def __init__(self, *, ranking: CandidateRanking, selection: SectionSelection) -> None:
        self._ranking = ranking
        self._selection = selection

    def build(self) -> PackMetrics:
        token_src = sum(_token_budget(node) for node in self._ranking.ranked_nodes)
        penalty_terms = [1 - value for value in self._selection.penalties]
        count = len(self._selection.penalties)
        diversity_penalty = sum(penalty_terms) / count if count else 0.0
        sections = self._selection.sections
        dup_rate = 0.0
        if sections:
            unique_paths = {str(section["id"]).split("#", 1)[0] for section in sections}
            dup_rate = 1.0 - len(unique_paths) / len(sections)
        ppr_values: List[float] = []
        for node in self._ranking.candidate_nodes:
            node_id = node.get("id")
            if isinstance(node_id, str):
                ppr_values.append(self._ranking.ppr_scores.get(node_id, 0.0))
        normaliser = sum(ppr_values) or 1.0
        entropy = -sum(
            (value / normaliser) * math.log(value / normaliser)
            for value in ppr_values
            if value > 0
        )
        return {
            "token_in": self._selection.token_in,
            "token_src": token_src,
            "dup_rate": dup_rate,
            "ppr_entropy": entropy,
            "diversity_penalty": diversity_penalty,
        }


def assemble_sections(
    view: GraphView,
    ranking: CandidateRanking,
    budget_tokens: int,
    config: Mapping[str, object],
) -> AssemblyResult:
    selector = SectionSelector(
        view=view,
        ranking=ranking,
        budget_tokens=budget_tokens,
        config=config,
    )
    selection = selector.select()
    metrics = PackMetricsBuilder(ranking=ranking, selection=selection).build()
    return AssemblyResult(sections=selection.sections, metrics=metrics)