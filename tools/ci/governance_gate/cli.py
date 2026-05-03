# SPDX-License-Identifier: MIT
# Copyright 2025 RNA4219

"""
CLI entry point for governance gate checks.

Argument parsing and main execution flow.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Sequence

from .resolver import PRBodyResolver
from .validator import collect_validation_outcome


def parse_arguments(argv: Sequence[str]) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run governance gate checks")
    parser.add_argument("--pr-body", help="PR本文を直接指定")
    parser.add_argument(
        "--pr-body-path",
        type=Path,
        help="PR本文が含まれるファイルパスを指定",
    )
    return parser.parse_args(list(argv))


def main(
    argv: Sequence[str] | None = None,
    *,
    category_hints: Sequence[str] | None = None,
    hint_resolver: Callable[[], Sequence[str] | None] | None = None,
) -> int:
    """Run governance gate validation and return exit code."""
    args = parse_arguments(argv or ())
    resolver = PRBodyResolver()
    resolution = resolver.resolve(
        cli_body=args.pr_body,
        cli_body_path=args.pr_body_path,
    )
    if not resolution.is_success:
        resolution.emit_errors(stream=sys.stderr)
        return 1
    body = resolution.body
    assert body is not None

    outcome = collect_validation_outcome(
        body,
        category_hints=category_hints,
        hint_resolver=hint_resolver,
    )
    outcome.emit(stream=sys.stderr)
    if not outcome.is_success:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main(tuple(sys.argv[1:])))