# SPDX-License-Identifier: MIT
# Copyright 2025 RNA4219

"""
Governance gate validation package.

Validates PR bodies against governance requirements.
"""

from __future__ import annotations

from .cli import main, parse_arguments
from .resolver import (
    PATH_CATEGORY_HINTS,
    CategoryHintResolver,
    PRBodyResolver,
    ResolutionResult,
    collect_recent_category_hints,
    get_changed_paths,
    infer_categories_from_paths,
    resolve_pr_body,
)
from .rules import (
    ACCEPTANCE_RECORD_PATTERN,
    ALLOWED_INTENT_CATEGORIES,
    DEFAULT_VALIDATION_RULES,
    DOCS_MATRIX_PATTERN,
    EVALUATION_ANCHOR_PATTERN,
    EVALUATION_HEADING_PATTERN,
    INTENT_CATEGORY_PATTERN,
    INTENT_ID_PATTERN,
    INTENT_PATTERN,
    PRIORITY_PATTERN,
    AcceptanceRecordRule,
    DocsMatrixRule,
    EvaluationReferenceRule,
    IntentCategoryRule,
    IntentPresenceRule,
    PriorityScoreRule,
    ValidationRule,
)
from .validator import (
    PRBodyValidator,
    ValidationContext,
    ValidationOutcome,
    collect_validation_outcome,
    validate_pr_body,
)

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