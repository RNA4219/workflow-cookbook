# SPDX-License-Identifier: MIT
# Copyright 2025 RNA4219

"""
Validation rules for PR body governance checks.

Defines regex patterns and rule classes for governance gate validation.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .validator import ValidationContext, ValidationOutcome


INTENT_PATTERN = re.compile(
    r"Intent\s*[：:]\s*INT-[0-9A-Z]+(?:-[0-9A-Z]+)*",
    re.IGNORECASE,
)
INTENT_CATEGORY_PATTERN = re.compile(r"INT-(\d{3,6})-([A-Z]{2,10})-", re.IGNORECASE)
INTENT_ID_PATTERN = re.compile(r"(INT-\d{3,6})", re.IGNORECASE)
EVALUATION_HEADING_PATTERN = re.compile(
    r"^#{2,6}\s*EVALUATION\b",
    re.IGNORECASE | re.MULTILINE,
)
EVALUATION_ANCHOR_PATTERN = re.compile(
    r"(?:EVALUATION\.md)?#acceptance-criteria",
    re.IGNORECASE,
)
ACCEPTANCE_RECORD_PATTERN = re.compile(
    r"docs/acceptance/AC-\d{8}-\d{2}\.md|AC-\d{8}-\d{2}\.md",
    re.IGNORECASE,
)
PRIORITY_PATTERN = re.compile(r"Priority\s*Score\s*:\s*\d+(?:\.\d+)?", re.IGNORECASE)

DOCS_MATRIX_PATTERN = {
    "REQUIREMENTS": re.compile(
        r"REQUIREMENTS:\s*present\?\s*\[(?P<yes>[xX ])\]\s*yes\s*/\s*\[(?P<later>[xX ])\]\s*later",
        re.IGNORECASE,
    ),
    "SPEC": re.compile(
        r"SPEC:\s*present\?\s*\[(?P<yes>[xX ])\]\s*yes\s*/\s*\[(?P<later>[xX ])\]\s*later",
        re.IGNORECASE,
    ),
    "DESIGN": re.compile(
        r"DESIGN:\s*present\?\s*\[(?P<yes>[xX ])\]\s*yes\s*/\s*\[(?P<later>[xX ])\]\s*later",
        re.IGNORECASE,
    ),
}

ALLOWED_INTENT_CATEGORIES = {
    "OPS",
    "SEC",
    "PLAT",
    "APP",
    "QA",
    "DOCS",
}


class ValidationRule:
    """Base class for validation rules."""

    def evaluate(self, context: ValidationContext, outcome: ValidationOutcome) -> None:
        raise NotImplementedError()


class IntentPresenceRule(ValidationRule):
    """Validates that PR body contains an Intent reference."""

    def evaluate(self, context: ValidationContext, outcome: ValidationOutcome) -> None:
        has_intent = bool(INTENT_PATTERN.search(context.normalized_body))
        context.intent_present = has_intent
        if not has_intent:
            outcome.add_error("PR body must include 'Intent: INT-xxx'")


class IntentCategoryRule(ValidationRule):
    """Validates that Intent has a valid category."""

    def evaluate(self, context: ValidationContext, outcome: ValidationOutcome) -> None:
        if not context.intent_present:
            return

        normalized_body = context.normalized_body
        category_matches = list(INTENT_CATEGORY_PATTERN.findall(normalized_body))
        if category_matches:
            for _, raw_category in category_matches:
                category = raw_category.upper()
                if category not in ALLOWED_INTENT_CATEGORIES:
                    allowed = ", ".join(sorted(ALLOWED_INTENT_CATEGORIES))
                    outcome.add_error(
                        f"Intent category '{category}' is not allowed. Allowed categories: {allowed}."
                    )
            return

        base_ids = {match.upper() for match in INTENT_ID_PATTERN.findall(normalized_body)}
        intent_reference = ", ".join(sorted(base_ids)) or "INT-???"
        hints = context.resolve_category_hints()
        if hints:
            suggestion = ", ".join(hints)
            message = (
                "No intent category pattern (INT-###-CAT-) detected for"
                f" {intent_reference}. Consider categories: {suggestion}."
            )
        else:
            message = (
                "No intent category pattern (INT-###-CAT-) detected and unable"
                " to infer category from recent changes."
            )
        outcome.add_warning(message)


class EvaluationReferenceRule(ValidationRule):
    """Validates that PR references an EVALUATION anchor."""

    def evaluate(self, context: ValidationContext, outcome: ValidationOutcome) -> None:
        normalized_body = context.normalized_body
        has_evaluation_heading = bool(EVALUATION_HEADING_PATTERN.search(normalized_body))
        has_evaluation_anchor = bool(EVALUATION_ANCHOR_PATTERN.search(normalized_body))
        if not (has_evaluation_heading or has_evaluation_anchor):
            outcome.add_error("PR must reference EVALUATION (acceptance) anchor")


class PriorityScoreRule(ValidationRule):
    """Warns if PR lacks a Priority Score."""

    def evaluate(self, context: ValidationContext, outcome: ValidationOutcome) -> None:
        if not PRIORITY_PATTERN.search(context.normalized_body):
            outcome.add_warning(
                "Consider adding 'Priority Score: <number>' based on prioritization.yaml"
            )


class AcceptanceRecordRule(ValidationRule):
    """Validates that PR references an acceptance record."""

    def evaluate(self, context: ValidationContext, outcome: ValidationOutcome) -> None:
        if not ACCEPTANCE_RECORD_PATTERN.search(context.normalized_body):
            outcome.add_error(
                "PR must reference an acceptance record under docs/acceptance/ (AC-YYYYMMDD-xx.md)"
            )


class DocsMatrixRule(ValidationRule):
    """Validates Docs matrix selections are properly marked."""

    def evaluate(self, context: ValidationContext, outcome: ValidationOutcome) -> None:
        for label, pattern in DOCS_MATRIX_PATTERN.items():
            match = pattern.search(context.normalized_body)
            if match is None:
                outcome.add_error(
                    f"PR must include Docs matrix selection for {label} using '[x] yes' or '[x] later'."
                )
                continue
            yes_marked = match.group("yes").strip().lower() == "x"
            later_marked = match.group("later").strip().lower() == "x"
            if yes_marked == later_marked:
                outcome.add_error(
                    f"Docs matrix for {label} must select exactly one of yes/later."
                )


DEFAULT_VALIDATION_RULES: tuple[ValidationRule, ...] = (
    IntentPresenceRule(),
    IntentCategoryRule(),
    EvaluationReferenceRule(),
    AcceptanceRecordRule(),
    DocsMatrixRule(),
    PriorityScoreRule(),
)


__all__ = [
    "ValidationRule",
    "IntentPresenceRule",
    "IntentCategoryRule",
    "EvaluationReferenceRule",
    "PriorityScoreRule",
    "AcceptanceRecordRule",
    "DocsMatrixRule",
    "DEFAULT_VALIDATION_RULES",
    "INTENT_PATTERN",
    "INTENT_CATEGORY_PATTERN",
    "INTENT_ID_PATTERN",
    "EVALUATION_HEADING_PATTERN",
    "EVALUATION_ANCHOR_PATTERN",
    "ACCEPTANCE_RECORD_PATTERN",
    "PRIORITY_PATTERN",
    "DOCS_MATRIX_PATTERN",
    "ALLOWED_INTENT_CATEGORIES",
]