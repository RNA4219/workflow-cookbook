# SPDX-License-Identifier: MIT
# Copyright 2025 RNA4219

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class ValidationResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_success(self) -> bool:
        return not self.errors

    def emit(self) -> None:
        for message in self.errors:
            print(message, file=sys.stderr)
        for message in self.warnings:
            print(message, file=sys.stderr)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _fetch_repo_security(repo: str, token: str) -> dict:
    request = urllib.request.Request(
        f"https://api.github.com/repos/{repo}",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def _vulnerability_alerts_enabled(repo: str, token: str) -> bool:
    request = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/vulnerability-alerts",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return response.status == 204
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return False
        raise


def validate_security_posture(
    *,
    repo_root: Path,
    github_repo: str | None = None,
    github_token: str | None = None,
) -> ValidationResult:
    result = ValidationResult()

    dependabot_path = repo_root / ".github" / "dependabot.yml"
    security_workflow_path = repo_root / ".github" / "workflows" / "security.yml"
    sac_path = repo_root / "docs" / "security" / "SAC.md"
    checklist_path = repo_root / "docs" / "security" / "Security_Review_Checklist.md"
    requirements_path = repo_root / "docs" / "requirements.md"
    spec_path = repo_root / "docs" / "spec.md"
    readme_path = repo_root / "README.md"

    required_paths = (
        dependabot_path,
        security_workflow_path,
        sac_path,
        checklist_path,
        requirements_path,
        spec_path,
        readme_path,
    )
    for path in required_paths:
        if not path.exists():
            result.errors.append(f"Required security asset is missing: {path}")

    if result.errors:
        return result

    dependabot_text = _read_text(dependabot_path)
    if "package-ecosystem: github-actions" not in dependabot_text:
        result.errors.append(".github/dependabot.yml does not configure github-actions updates.")
    if "interval: weekly" not in dependabot_text:
        result.errors.append(".github/dependabot.yml does not enforce a weekly update schedule.")

    sac_text = _read_text(sac_path)
    for marker in ("SAST", "Secrets", "依存", "Container"):
        if marker not in sac_text and marker.lower() not in sac_text.lower():
            result.errors.append(f"docs/security/SAC.md does not mention the '{marker}' gate.")

    checklist_text = _read_text(checklist_path)
    if "Dependabot" not in checklist_text and "dependabot" not in checklist_text:
        result.errors.append(
            "docs/security/Security_Review_Checklist.md does not mention Dependabot or dependency alert operations."
        )

    requirements_text = _read_text(requirements_path)
    spec_text = _read_text(spec_path)
    readme_text = _read_text(readme_path)
    if "Security posture" not in requirements_text and "セキュリティ" not in requirements_text:
        result.errors.append("docs/requirements.md does not contain an explicit security posture section.")
    if "Security baseline" not in spec_text and "セキュリティ" not in spec_text:
        result.errors.append("docs/spec.md does not contain an explicit security baseline section.")
    if "Security" not in readme_text:
        result.errors.append("README.md does not advertise the security baseline entry point.")

    if github_repo:
        if not github_token:
            result.warnings.append("GitHub security settings were not checked because no token was provided.")
            return result
        try:
            repo_payload = _fetch_repo_security(github_repo, github_token)
            security = repo_payload.get("security_and_analysis", {})
            dependabot_updates = security.get("dependabot_security_updates", {}).get("status")
            secret_scanning = security.get("secret_scanning", {}).get("status")
            push_protection = security.get("secret_scanning_push_protection", {}).get("status")
            if dependabot_updates != "enabled":
                result.errors.append("GitHub Dependabot security updates are not enabled.")
            if secret_scanning != "enabled":
                result.errors.append("GitHub secret scanning is not enabled.")
            if push_protection != "enabled":
                result.errors.append("GitHub secret scanning push protection is not enabled.")
            if not _vulnerability_alerts_enabled(github_repo, github_token):
                result.errors.append("GitHub vulnerability alerts are not enabled.")
        except urllib.error.HTTPError as exc:
            result.errors.append(f"Failed to fetch GitHub security posture: HTTP {exc.code}")

    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate security posture docs, workflows, and optional GitHub settings.")
    parser.add_argument("--check", action="store_true", help="Exit non-zero when validation fails.")
    parser.add_argument("--repo-root", type=Path, default=_REPO_ROOT, help="Repository root.")
    parser.add_argument("--github-repo", help="Repository in owner/name form for remote posture checks.")
    parser.add_argument("--token-env", default="GITHUB_TOKEN", help="Environment variable that contains a GitHub token.")
    args = parser.parse_args(argv)

    token = os.environ.get(args.token_env) if args.github_repo else None
    result = validate_security_posture(
        repo_root=args.repo_root,
        github_repo=args.github_repo,
        github_token=token,
    )
    result.emit()
    if result.is_success:
        print("Security posture matches docs, workflows, and GitHub settings.")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
