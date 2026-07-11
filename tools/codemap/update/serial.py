# SPDX-License-Identifier: MIT
# Copyright 2025 RNA4219

"""Birdseye generated-at serial allocation."""

from __future__ import annotations

import re
from typing import Any

_SERIAL_PATTERN = re.compile(r"\d{05}")


def _coerce_serial(candidate: Any) -> int | None:
    if isinstance(candidate, str) and _SERIAL_PATTERN.fullmatch(candidate):
        return int(candidate)
    return None


class _SerialAllocator:
    __slots__ = ("max_serial", "_next_serial")

    def __init__(self) -> None:
        self.max_serial = 0
        self._next_serial: int | None = None

    def observe(self, candidate: Any) -> None:
        value = _coerce_serial(candidate)
        if value is not None and value > self.max_serial:
            self.max_serial = value

    def allocate(self, existing: Any) -> str:
        candidate = _coerce_serial(existing)
        if candidate is not None and candidate > self.max_serial:
            self.max_serial = candidate
        if self._next_serial is None:
            self._next_serial = self.max_serial + 1 if self.max_serial else 1
        target = self._next_serial
        if target > self.max_serial:
            self.max_serial = target
        return f"{target:05d}"


def next_generated_at(existing: Any, fallback: str, *, allocator: _SerialAllocator) -> str:
    del fallback
    return allocator.allocate(existing)
