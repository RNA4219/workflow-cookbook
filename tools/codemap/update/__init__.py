# SPDX-License-Identifier: MIT
# Copyright 2025 RNA4219

"""
Birdseye update package.

Regenerates Birdseye index and capsules.
"""

from __future__ import annotations

import subprocess
import sys

from . import constants
from . import cli
from .constants import _REPO_ROOT, _BIRDSEYE_REGENERATE_COMMAND
from .types import (
    CapsuleEntry,
    CapsuleState,
    Graph,
    TargetResolutionError,
    DiffResolver,
    UpdateOptions,
    UpdateReport,
    PlannedWrite,
    BirdseyePlan,
    BirdseyeRootPlan,
)
from .diff import (
    GitDiffParser,
    GitDiffResolver,
    derive_targets_from_since,
)
from .graph import (
    build_graph,
    BirdseyeFocusResolver,
)
from .capsule import (
    BirdseyeRootBuilder,
    load_json,
    dump_json,
)
from .session import (
    BirdseyeUpdateSession,
    run_update,
    utc_now,
    _default_birdseye_targets,
    _normalise_target,
    _group_targets,
    _load_json,
    _dump_json,
    _sorted_unique,
    _SerialAllocator,
    next_generated_at,
)
from .cli import (
    parse_args,
    ensure_python_version,
    main,
)

# Internal exports for tests
_derive_targets_from_since = derive_targets_from_since
_next_generated_at = next_generated_at

__all__ = [
    "CapsuleEntry",
    "CapsuleState",
    "Graph",
    "TargetResolutionError",
    "DiffResolver",
    "UpdateOptions",
    "UpdateReport",
    "PlannedWrite",
    "BirdseyePlan",
    "BirdseyeRootPlan",
    "GitDiffParser",
    "GitDiffResolver",
    "derive_targets_from_since",
    "build_graph",
    "BirdseyeFocusResolver",
    "BirdseyeRootBuilder",
    "load_json",
    "dump_json",
    "BirdseyeUpdateSession",
    "run_update",
    "utc_now",
    "parse_args",
    "ensure_python_version",
    "main",
    "constants",
    "cli",
    # Internal exports for tests
    "subprocess",
    "sys",
    "_REPO_ROOT",
    "_BIRDSEYE_REGENERATE_COMMAND",
    "_derive_targets_from_since",
    "_default_birdseye_targets",
    "_normalise_target",
    "_group_targets",
    "_load_json",
    "_dump_json",
    "_sorted_unique",
    "_SerialAllocator",
    "next_generated_at",
    "_next_generated_at",
]