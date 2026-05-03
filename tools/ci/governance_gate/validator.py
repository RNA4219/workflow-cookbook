# SPDX-License-Identifier: MIT
# Copyright 2025 RNA4219

"""
Validation context and outcome handling.

Defines ValidationContext, ValidationOutcome, and PRBodyValidator.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Callable, Iterator, Sequence, TextIO, Tuple

from .resolver import collect_recent_category_hints
from .rules import DEFAULT_VALIDATION_RULES, ValidationRule


@dataclass
class ValidationOutcome:
    """Aggregated validation errors and warnings."""
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    _messages: list[Tuple[str, str]] = field(default_factory=list, repr=False)

    @property
    def is_success(self) -> bool:
        return not self.errors

    def add_error(self, message: str) -> None:
        self.errors.append(message)
        self._messages.append(("error", message))

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)
        self._messages.append(("warning", message))

    def iter_messages(self) -> Iterator[Tuple[str, str]]:
        return iter(self._messages)

    def emit(self, *, stream: TextIO = sys.stderr) -> None:
        for _, message in self._messages:
            print(message, file=stream)


@dataclass
class ValidationContext:
    """Context for PR body validation."""
    body: str
    category_hints: Sequence[str] | None = None
    hint_resolver: Callable[[], Sequence[str] | None] | None = None
    intent_present: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self.body = self.body or ""
        self._resolved_hints: list[str] | None = None

    @property
    def normalized_body(self) -> str:
        return self.body

    def resolve_category_hints(self) -> list[str]:
        if self._resolved_hints is None:
            if self.category_hints is not None:
                hints: Sequence[str] | None = self.category_hints
            else:
                resolver = self.hint_resolver or collect_recent_category_hints
                hints = resolver() or []
            self._resolved_hints = [hint for hint in hints if hint]
        return list(self._resolved_hints)


class PRBodyValidator:
    """Validates PR body against configured rules."""

    def __init__(self, rules: Sequence[ValidationRule] | None = None) -> None:
        if rules is None:
            rules = DEFAULT_VALIDATION_RULES
        self._rules = list(rules)

    def validate(self, context: ValidationContext) -> ValidationOutcome:
        outcome = ValidationOutcome()
        for rule in self._rules:
            rule.evaluate(context, outcome)
        return outcome


def collect_validation_outcome(
    body: str | None,
    *,
    category_hints: Sequence[str] | None = None,
    hint_resolver: Callable[[], Sequence[str] | None] | None = None,
) -> ValidationOutcome:
    """Run validation and return outcome without emitting."""
    validator = PRBodyValidator()
    context = ValidationContext(
        body=body or "",
        category_hints=category_hints,
        hint_resolver=hint_resolver,
    )
    return validator.validate(context)


def validate_pr_body(
    body: str | None,
    *,
    category_hints: Sequence[str] | None = None,
    hint_resolver: Callable[[], Sequence[str] | None] | None = None,
) -> bool:
    """Run validation, emit messages, and return success status."""
    outcome = collect_validation_outcome(
        body,
        category_hints=category_hints,
        hint_resolver=hint_resolver,
    )
    outcome.emit(stream=sys.stderr)
    return outcome.is_success


__all__ = [
    "ValidationOutcome",
    "ValidationContext",
    "PRBodyValidator",
    "collect_validation_outcome",
    "validate_pr_body",
]