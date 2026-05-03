# SPDX-License-Identifier: MIT
# Copyright 2025 RNA4219

"""
CLI entry point for Birdseye update.

Provides argument parsing and main execution.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Iterable

from .types import UpdateOptions, TargetResolutionError
from .diff import GitDiffResolver
from .session import run_update, _normalise_target


def parse_args(argv: Iterable[str] | None = None) -> UpdateOptions:
    parser = argparse.ArgumentParser(
        description="Regenerate Birdseye index and capsules.",
    )
    parser.add_argument(
        "--targets",
        type=str,
        help="Comma-separated list of Birdseye resources to analyse.",
    )
    parser.add_argument(
        "--since",
        type=str,
        nargs="?",
        const="main",
        help="Derive targets from git diff since the given reference (default: main).",
    )
    parser.add_argument(
        "--emit",
        type=str,
        choices=("index", "caps", "index+caps"),
        default="index+caps",
        help="Select which artefacts to write.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute updates without writing to disk.",
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=2,
        help="Traverse the Birdseye graph up to this many hops from each target.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.radius < 0:
        parser.error("--radius must be 0 or greater")
    target_paths: list[Path] = []
    if args.targets:
        target_paths.extend(
            Path(value.strip()) for value in args.targets.split(",") if value.strip()
        )
    if not target_paths and not args.since:
        parser.error("Specify --targets, --since, or both")
    normalised_targets = tuple(
        dict.fromkeys(_normalise_target(path) for path in target_paths)
    )
    return UpdateOptions(
        targets=normalised_targets,
        emit=args.emit,
        dry_run=args.dry_run,
        since=args.since,
        radius=args.radius,
    )


def ensure_python_version() -> None:
    if sys.version_info < (3, 11):
        print("[ERROR] Python 3.11 or newer is required.")
        raise SystemExit(1)


def main(argv: Iterable[str] | None = None) -> int:
    ensure_python_version()
    options = parse_args(argv)
    options = replace(options, diff_resolver=GitDiffResolver())
    try:
        run_update(options)
    except TargetResolutionError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())