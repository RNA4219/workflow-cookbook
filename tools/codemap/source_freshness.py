# SPDX-License-Identifier: MIT
"""原文と要約の確認記録。生成世代と内容確認を区別する。"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SUMMARY_FIELDS = ("id", "role", "summary", "public_api", "risks", "tests", "deps_in", "deps_out")


def source_digest(repo_root: Path, node_id: str) -> str | None:
    root = repo_root.resolve()
    path = (root / node_id).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        return None
    # 自身の出力をhashする循環は作らない。
    if path.is_relative_to(root / "docs/birdseye") and path.suffix == ".json":
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary_digest(capsule: Mapping[str, Any]) -> str:
    content = {key: capsule.get(key) for key in SUMMARY_FIELDS}
    serialized = json.dumps(content, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def review_matches(capsule: Mapping[str, Any], source_hash: str) -> bool:
    review = capsule.get("review")
    return (
        isinstance(review, Mapping)
        and review.get("source_sha256") == source_hash
        and review.get("summary_sha256") == summary_digest(capsule)
        and isinstance(review.get("reviewed_at"), str)
        and bool(review["reviewed_at"].strip())
    )
