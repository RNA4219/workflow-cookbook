#!/usr/bin/env python
# SPDX-License-Identifier: MIT
# Copyright 2025 RNA4219

"""Compatibility entrypoint for the docs review due checker."""

from tools.ci.docs_review_due.artifacts import (
    _build_nudges,
    _owner_summary,
    _render_task_seed,
    _review_update_plan,
    _write_json,
)
from tools.ci.docs_review_due.cli import _REPO_ROOT, _parse_today, main
from tools.ci.docs_review_due.models import DocReviewStatus
from tools.ci.docs_review_due.scanner import (
    _EXCLUDED_DIRS,
    _EXCLUDED_FILES,
    _TERMINAL_STATUSES,
    _categorize,
    _parse_front_matter,
    _scan_docs,
)

__all__ = [
    "DocReviewStatus",
    "_EXCLUDED_DIRS",
    "_EXCLUDED_FILES",
    "_TERMINAL_STATUSES",
    "_build_nudges",
    "_categorize",
    "_owner_summary",
    "_parse_front_matter",
    "_parse_today",
    "_render_task_seed",
    "_REPO_ROOT",
    "_review_update_plan",
    "_scan_docs",
    "_write_json",
    "main",
]

if __name__ == "__main__":
    raise SystemExit(main())
