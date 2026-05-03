# SPDX-License-Identifier: MIT
# Copyright 2025 RNA4219

"""
PR body resolution utilities.

Resolves PR body content from CLI args, environment variables, or GitHub event payload.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, List, Sequence, TextIO


_REPO_ROOT = Path(__file__).resolve().parents[3]


def get_changed_paths(refspec: str) -> List[str]:
    """Get list of changed file paths from git diff."""
    result = subprocess.run(  # nosec B603,B607  # git command with fixed args, no shell=True, refspec validated by caller
        ["git", "diff", "--name-only", refspec],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=_REPO_ROOT,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def read_event_body(event_path: Path) -> str | None:
    """Read PR body from GitHub event payload file."""
    if not event_path.exists():
        return None
    payload = json.loads(event_path.read_text(encoding="utf-8"))
    pull_request = payload.get("pull_request")
    if not isinstance(pull_request, dict):
        return None
    body = pull_request.get("body")
    if body is None:
        return None
    if not isinstance(body, str):
        return None
    return body


def read_pr_body_from_path(path: Path) -> str | None:
    """Read PR body from a file path."""
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


@dataclass
class ResolutionResult:
    """Result of PR body resolution attempt."""
    body: str | None
    errors: list[str] = field(default_factory=list)

    @property
    def is_success(self) -> bool:
        return self.body is not None

    @property
    def combined_error_message(self) -> str:
        return "\n".join(self.errors)

    def emit_errors(self, *, stream: TextIO | None = None) -> None:
        if not self.errors:
            return
        target = sys.stderr if stream is None else stream
        print(self.combined_error_message, file=target)


class PRBodyResolver:
    """Resolves PR body from multiple sources with precedence."""

    def __init__(
        self,
        *,
        env_getter: Callable[[str], str | None] | None = None,
        path_reader: Callable[[Path], str | None] | None = None,
        event_reader: Callable[[Path], str | None] | None = None,
    ) -> None:
        self._env_getter = env_getter or os.environ.get
        self._path_reader = path_reader or read_pr_body_from_path
        self._event_reader = event_reader or read_event_body

    def resolve(
        self,
        *,
        cli_body: str | None = None,
        cli_body_path: Path | None = None,
    ) -> ResolutionResult:
        if cli_body is not None:
            return ResolutionResult(body=cli_body)

        errors: list[str] = []

        if cli_body_path is not None:
            body_from_cli_path = self._path_reader(cli_body_path)
            if body_from_cli_path is not None:
                return ResolutionResult(body=body_from_cli_path)
            errors.append(f"PR body file not found: {cli_body_path}")

        direct_body = self._env_getter("PR_BODY")
        if direct_body is not None:
            return ResolutionResult(body=direct_body)

        env_body_path_value = self._env_getter("PR_BODY_PATH")
        if env_body_path_value:
            env_body_path = Path(env_body_path_value)
            body_from_env_path = self._path_reader(env_body_path)
            if body_from_env_path is not None:
                return ResolutionResult(body=body_from_env_path)
            errors.append(f"PR body file not found: {env_body_path}")

        event_path_value = self._env_getter("GITHUB_EVENT_PATH")
        if event_path_value:
            body_from_event = self._event_reader(Path(event_path_value))
            if body_from_event is not None:
                return ResolutionResult(body=body_from_event)

        errors.append("PR body data is unavailable. Set PR_BODY or GITHUB_EVENT_PATH.")
        return ResolutionResult(body=None, errors=errors)


def resolve_pr_body(
    *, cli_body: str | None = None, cli_body_path: Path | None = None
) -> str | None:
    """Convenience function to resolve PR body using default resolver."""
    resolver = PRBodyResolver()
    result = resolver.resolve(cli_body=cli_body, cli_body_path=cli_body_path)
    if not result.is_success:
        result.emit_errors()
        return None
    return result.body


def infer_categories_from_paths(paths: Iterable[str]) -> List[str]:
    """Infer intent category hints from changed file paths."""
    import re
    suggestions: list[str] = []
    for path in paths:
        normalized_path = path.strip().lstrip("./")
        if not normalized_path:
            continue
        segments = re.split(r"[/_.-]+", normalized_path)
        for segment in segments:
            hint = PATH_CATEGORY_HINTS.get(segment.lower())
            if hint and hint not in suggestions:
                suggestions.append(hint)
    return suggestions


PATH_CATEGORY_HINTS = {
    "ops": "OPS",
    "runbook": "OPS",
    "security": "SEC",
    "sec": "SEC",
    "platform": "PLAT",
    "infra": "PLAT",
    "app": "APP",
    "frontend": "APP",
    "qa": "QA",
    "test": "QA",
    "docs": "DOCS",
    "documentation": "DOCS",
}


class CategoryHintResolver:
    """Resolves category hints from recent git changes."""

    def __init__(
        self,
        *,
        env_getter: Callable[[str], str | None] | None = None,
        changed_paths_provider: Callable[[str], Sequence[str]] | None = None,
        fallback_refspec: str = "HEAD^..HEAD",
    ) -> None:
        self._env_getter = env_getter or os.environ.get
        self._changed_paths_provider = changed_paths_provider or get_changed_paths
        self._fallback_refspec = fallback_refspec

    def resolve(
        self,
        *,
        base_ref: str | None = None,
        head_ref: str = "HEAD",
        fallback_refspec: str | None = None,
    ) -> List[str]:
        resolved_base = (base_ref or self._env_getter("GITHUB_BASE_REF") or "").strip()
        effective_fallback = fallback_refspec or self._fallback_refspec

        refspec_candidates: list[str] = []
        if resolved_base:
            base_spec = resolved_base
            if not base_spec.startswith("origin/"):
                base_spec = f"origin/{base_spec}"
            refspec_candidates.append(f"{base_spec}...{head_ref}")
        refspec_candidates.append(effective_fallback)

        last_index = len(refspec_candidates) - 1
        for index, refspec in enumerate(refspec_candidates):
            try:
                changed_paths = self._changed_paths_provider(refspec)
            except subprocess.CalledProcessError:
                continue
            hints = infer_categories_from_paths(changed_paths)
            if hints or index == last_index:
                return hints

        return []


def collect_recent_category_hints(
    *,
    base_ref: str | None = None,
    head_ref: str = "HEAD",
    fallback_refspec: str = "HEAD^..HEAD",
) -> List[str]:
    """Convenience function to collect category hints using default resolver."""
    resolver = CategoryHintResolver()
    return resolver.resolve(
        base_ref=base_ref,
        head_ref=head_ref,
        fallback_refspec=fallback_refspec,
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
]