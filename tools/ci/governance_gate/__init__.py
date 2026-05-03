# SPDX-License-Identifier: MIT
# Copyright 2025 RNA4219

"""
Governance gate validation package.

Validates PR bodies against governance requirements.
"""

from __future__ import annotations

from .resolver import (
    ResolutionResult,
    PRBodyResolver,
    resolve_pr_body,
    CategoryHintResolver,
    collect_recent_category_hints,
    get_changed_paths,
    infer_categories_from_paths,
    PATH_CATEGORY_HINTS,
)
from .rules import (
    ValidationRule,
    IntentPresenceRule,
    IntentCategoryRule,
    EvaluationReferenceRule,
    PriorityScoreRule,
    AcceptanceRecordRule,
    DocsMatrixRule,
    DEFAULT_VALIDATION_RULES,
    INTENT_PATTERN,
    INTENT_CATEGORY_PATTERN,
    INTENT_ID_PATTERN,
    EVALUATION_HEADING_PATTERN,
    EVALUATION_ANCHOR_PATTERN,
    ACCEPTANCE_RECORD_PATTERN,
    PRIORITY_PATTERN,
    DOCS_MATRIX_PATTERN,
    ALLOWED_INTENT_CATEGORIES,
)
from .validator import (
    ValidationOutcome,
    ValidationContext,
    PRBodyValidator,
    collect_validation_outcome,
    validate_pr_body,
)
from .cli import main, parse_arguments


__all__ = [
    "ResolutionResult",
    "PRBodyResolver",
    "resolve_pr_body",
    "CategoryHintResolver",
    "collect_recent_category_hints",
    "get_changed_paths",
    "infer_categories_from_paths",
    "PATH_CATEGORY_HINTS",
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
    "ValidationOutcome",
    "ValidationContext",
    "PRBodyValidator",
    "collect_validation_outcome",
    "validate_pr_body",
    "main",
    "parse_arguments",
]