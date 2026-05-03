# SPDX-License-Identifier: MIT
# Copyright 2025 RNA4219

"""
CLI entry point for context pack.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, Mapping, MutableMapping, Sequence

from .types import (
    ConfigDict,
    DEFAULT_CONFIG,
    PackOutput,
    ContextPackPlan,
)
from .resolver import build_graph_view, score_candidates
from .compression import assemble_sections


def _parse_simple_yaml(text: str) -> ConfigDict:
    root: Dict[str, object] = {}
    stack: list[tuple[int, MutableMapping[str, object]]] = [(0, root)]
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        while stack and indent < stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        key, _, rest = line.partition(":")
        key = key.strip()
        value = rest.strip()
        if not value:
            node: MutableMapping[str, object] = {}
            parent[key] = node
            stack.append((indent + 2, node))
        else:
            parent[key] = _coerce_scalar(value)
    return root


def _coerce_scalar(value: str):
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if any(ch in value for ch in ".eE"):
            return float(value)
        return int(value)
    except ValueError:
        return value


def load_config(path: Path | None = None) -> ConfigDict:
    target = path or Path("tools/context/config.yaml")
    if not target.exists():
        return json.loads(json.dumps(DEFAULT_CONFIG))
    return _parse_simple_yaml(target.read_text())


class ContextPackPlanner:
    def __init__(
        self,
        *,
        config_loader: Callable[[Path | None], Mapping[str, object]] | None = None,
    ) -> None:
        self._config_loader = config_loader or load_config
        self.config: Mapping[str, object] | None = None
        self.graph: Mapping[str, object] | None = None
        self.view: "GraphView | None" = None
        self.ranking: "CandidateRanking | None" = None
        self.assembly: "AssemblyResult | None" = None

    def build_plan(
        self,
        *,
        graph_path: Path,
        intent: str,
        budget_tokens: int,
        diff_paths: Sequence[str],
        config: Mapping[str, object] | None = None,
    ) -> ContextPackPlan:
        from .types import GraphView, CandidateRanking, AssemblyResult
        config_map = config if config is not None else self._config_loader(None)
        graph = json.loads(graph_path.read_text())
        view = build_graph_view(graph, intent, diff_paths, config_map)
        ranking = score_candidates(view, config_map)
        assembly = assemble_sections(view, ranking, budget_tokens, config_map)
        self.config = config_map
        self.graph = graph
        self.view = view
        self.ranking = ranking
        self.assembly = assembly
        return ContextPackPlan(
            intent=intent,
            budget_tokens=budget_tokens,
            diff_paths=tuple(diff_paths),
            graph=graph,
            config=config_map,
            view=view,
            ranking=ranking,
            assembly=assembly,
        )


class ContextPackExecutor:
    def __init__(self, plan: ContextPackPlan) -> None:
        self.plan = plan
        self.output: PackOutput | None = None

    def pack(self) -> PackOutput:
        result: PackOutput = {
            "intent": self.plan.intent,
            "budget": str(self.plan.budget_tokens),
            "sections": self.plan.assembly.sections,
            "metrics": self.plan.assembly.metrics,
        }
        self.output = result
        return result


def pack_graph(
    graph_path: Path,
    intent: str,
    budget_tokens: int,
    diff_paths: Sequence[str],
    config: Mapping[str, object] | None = None,
) -> PackOutput:
    planner = ContextPackPlanner()
    plan = planner.build_plan(
        graph_path=graph_path,
        intent=intent,
        budget_tokens=budget_tokens,
        diff_paths=diff_paths,
        config=config,
    )
    executor = ContextPackExecutor(plan)
    return executor.pack()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate context packs with PPR scoring")
    parser.add_argument("--graph", type=Path, default=Path("reports/context/graph.json"))
    parser.add_argument("--intent", required=True)
    parser.add_argument("--budget", type=int, default=2000)
    parser.add_argument("--diff", type=Path, nargs="*", default=[])
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("reports/context/pack.json"))
    args = parser.parse_args()
    diff_paths = []
    for diff in args.diff:
        diff_paths.extend(diff.read_text().splitlines())
    config = load_config(args.config) if args.config else None
    planner = ContextPackPlanner()
    plan = planner.build_plan(
        graph_path=args.graph,
        intent=args.intent,
        budget_tokens=args.budget,
        diff_paths=diff_paths,
        config=config,
    )
    executor = ContextPackExecutor(plan)
    result = executor.pack()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()