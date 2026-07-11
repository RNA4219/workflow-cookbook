from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class DocReviewStatus:
    file_path: Path
    rel_path: str
    owner: str | None
    status: str | None
    last_reviewed: str | None
    next_review_due: str | None
    days_until_review: int | None
    days_overdue: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "file": self.rel_path,
            "owner": self.owner,
            "status": self.status,
            "last_reviewed": self.last_reviewed,
            "next_review_due": self.next_review_due,
            "days_until_review": self.days_until_review,
            "days_overdue": self.days_overdue,
        }
