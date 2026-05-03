# SPDX-License-Identifier: MIT
# Copyright 2025 RNA4219

"""
Shared constants for Birdseye update package.
"""

from __future__ import annotations

from pathlib import Path


# Mutable container for repo root (allows monkeypatching)
class _RepoRootContainer:
    __slots__ = ("value",)

    def __init__(self) -> None:
        self.value = Path(__file__).resolve().parents[3]

    def __call__(self) -> Path:
        return self.value

    def get(self) -> Path:
        return self.value


_REPO_ROOT = _RepoRootContainer()

_BIRDSEYE_REGENERATE_COMMAND = (
    "python tools/codemap/update.py --targets "
    "docs/birdseye/index.json,docs/birdseye/hot.json --emit index+caps"
)