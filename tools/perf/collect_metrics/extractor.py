# SPDX-License-Identifier: MIT
# Copyright 2025 RNA4219

"""
Metric extraction core classes.

Defines MetricDefinition, MetricDefinitionRegistry, and MetricExtractor.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Mapping, MutableMapping, Sequence

from .helpers import OVERWRITE_KEYS
from .rules import NumericRule, StructuredRule
from .security import MetricsCollectionError


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class MetricDefinition:
    """Definition of a metric with extraction rules."""
    key: str
    structured_rules: tuple[StructuredRule, ...]
    numeric_rules: tuple[NumericRule, ...]
    required: bool = True


class MetricDefinitionRegistry:
    """Registry for metric definitions."""

    def __init__(self) -> None:
        self._definitions: dict[str, MetricDefinition] = {}
        self._order: list[str] = []

    def register(self, definition: MetricDefinition) -> None:
        if definition.key in self._definitions:
            self._definitions[definition.key] = definition
            return
        self._order.append(definition.key)
        self._definitions[definition.key] = definition

    def get(self, key: str) -> MetricDefinition | None:
        return self._definitions.get(key)

    def definitions(self) -> tuple[MetricDefinition, ...]:
        return tuple(self._definitions[key] for key in self._order)


class MetricExtractor:
    """Extract metrics from multiple sources using definitions."""

    def __init__(
        self,
        definitions: Sequence[MetricDefinition],
        *,
        percentage_keys: Sequence[str],
    ) -> None:
        self._definitions = {definition.key: definition for definition in definitions}
        self._ordered_keys = tuple(definition.key for definition in definitions)
        self._ordered_definitions = tuple(definitions)
        self._percentage_keys = tuple(percentage_keys)
        self._key_set = frozenset(self._ordered_keys)

    def capture_structured(
        self,
        source: Mapping[str, object],
        target: MutableMapping[str, float],
        *,
        overwrite: bool = False,
    ) -> None:
        for definition in self._ordered_definitions:
            existing = definition.key in target
            for rule in definition.structured_rules:
                if existing and not (overwrite or rule.overwrite):
                    continue
                value = rule.extract(definition.key, source)
                if value is not None:
                    target[definition.key] = value
                    existing = True
                    break

    def capture_numeric(
        self,
        source: Mapping[str, float],
        target: MutableMapping[str, float],
    ) -> None:
        for definition in self._ordered_definitions:
            if definition.key in target:
                continue
            for rule in definition.numeric_rules:
                value = rule.extract(definition.key, source)
                if value is not None:
                    target[definition.key] = value
                    break

    def merge(self, sources: Sequence[Mapping[str, float]]) -> dict[str, float]:
        combined: dict[str, float] = {}
        reported_unexpected: set[str] = set()
        for mapping in sources:
            unexpected = [
                key for key in mapping if key not in self._key_set and key not in reported_unexpected
            ]
            if unexpected:
                reported_unexpected.update(unexpected)
                LOGGER.warning(
                    "Ignoring metrics not defined in governance/metrics.yaml: %s",
                    ", ".join(sorted(unexpected)),
                )
            for definition in self._ordered_definitions:
                key = definition.key
                if key in mapping and key not in combined:
                    combined[key] = mapping[key]
            for key in OVERWRITE_KEYS:
                if key in mapping:
                    combined[key] = mapping[key]
        missing_required = [
            definition.key
            for definition in self._ordered_definitions
            if definition.required and definition.key not in combined
        ]
        if missing_required:
            raise MetricsCollectionError("Missing metrics: " + ", ".join(missing_required))
        metrics = {
            definition.key: combined[definition.key]
            for definition in self._ordered_definitions
            if definition.key in combined
        }
        for key in self._percentage_keys:
            if key in metrics:
                metrics[key] *= 100.0
        return metrics

    def percentage_keys(self) -> tuple[str, ...]:
        return self._percentage_keys


__all__ = [
    "MetricDefinition",
    "MetricDefinitionRegistry",
    "MetricExtractor",
]