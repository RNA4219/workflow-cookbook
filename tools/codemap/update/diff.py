# SPDX-License-Identifier: MIT
# Copyright 2025 RNA4219

"""
Diff resolution for Birdseye update.

Parses git diff output and derives target paths.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Protocol

from .constants import _REPO_ROOT


def _format_diff_reference(reference: str) -> str:
    stripped = reference.strip()
    if ".." in stripped:
        return stripped
    return f"{stripped}...HEAD"


class GitDiffParser:
    def __init__(self, *, repo_root: Path | None = None) -> None:
        self._repo_root = repo_root or _REPO_ROOT.get()

    def parse(self, payload: str) -> tuple[Path, ...]:
        diff_entries: list[str] = []
        for raw_line in payload.splitlines():
            stripped = raw_line.strip()
            if not stripped:
                continue
            columns = [segment.strip() for segment in stripped.split("\t")]
            if not columns:
                continue
            status = columns[0]
            if not status:
                continue
            kind = status[0].upper()
            payloads: list[str]
            if kind in {"R", "C"} and len(columns) >= 3:
                payloads = columns[1:3]
            elif kind in {"A", "M", "D", "T", "U"} and len(columns) >= 2:
                payloads = [columns[1]]
            else:
                continue
            for value in payloads:
                candidate = value.replace("\\", "/").strip()
                if not candidate:
                    continue
                diff_entries.append(candidate)
        derived = derive_targets_from_since(diff_entries, repo_root=self._repo_root)
        return tuple(
            path
            for path in derived
            if len(path.parts) >= 2 and path.parts[0] == "docs" and path.parts[1] == "birdseye"
        )


class _GitDiffRunner(Protocol):
    def __call__(self, args: Sequence[str]) -> subprocess.CompletedProcess[str]:
        ...


def _run_git_diff(args: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        capture_output=True,
        text=True,
        check=True,
        cwd=_REPO_ROOT.get(),
    )


class GitDiffResolver:
    def __init__(
        self,
        *,
        parser: GitDiffParser | None = None,
        runner: _GitDiffRunner | None = None,
    ) -> None:
        self._parser = parser or GitDiffParser()
        self._runner = runner or _run_git_diff

    def resolve(self, reference: str) -> tuple[Path, ...]:
        diff_reference = _format_diff_reference(reference)
        command = [
            "git",
            "diff",
            "--name-status",
            "--find-renames",
            "--find-copies",
            diff_reference,
        ]
        result = self._runner(command)
        return self._parser.parse(result.stdout)


def derive_targets_from_since(
    diff_paths: Iterable[str | Path], *, repo_root: Path | None = None
) -> tuple[Path, ...]:
    base_root = repo_root or _REPO_ROOT.get()
    derived: list[Path] = []
    seen: set[Path] = set()

    def _add(path: Path) -> None:
        if path not in seen:
            seen.add(path)
            derived.append(path)

    for path in diff_paths:
        if isinstance(path, Path):
            raw_entry = path.as_posix()
        else:
            raw_entry = path
        raw_entry = raw_entry.replace("\\", "/").strip()
        if not raw_entry:
            continue
        segments: list[str] = []
        for marker in (" -> ", " => "):
            if marker in raw_entry:
                left, right = raw_entry.split(marker, 1)
                segments.extend([left.strip(), right.strip()])
                break
        if not segments:
            segments.append(raw_entry)
        for segment in segments:
            candidate = segment
            if not candidate:
                continue
            while candidate.startswith("./"):
                candidate = candidate[2:]
            candidate = candidate.rstrip("/")
            if not candidate:
                continue
            normalised = Path(candidate)
            if normalised.is_absolute():
                try:
                    normalised = normalised.relative_to(base_root)
                except ValueError:
                    continue
            if normalised.parts[:2] == ("docs", "birdseye"):
                _add(normalised)
                continue
            candidate_slug = normalised.as_posix() if normalised.parts else candidate
            capsule_slug = candidate_slug.replace("/", ".")
            capsule_path = Path("docs/birdseye/caps") / f"{capsule_slug}.json"
            if (base_root / capsule_path).is_file():
                _add(capsule_path)
    return tuple(derived)