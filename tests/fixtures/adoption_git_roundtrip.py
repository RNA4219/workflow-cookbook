# SPDX-License-Identifier: MIT
"""WindowsでコピーをGitへ登録し、そのbundleを別OSで検査する回帰fixture。"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
COPY_ROOT = "workflow-cookbook"
SCRIPT = f"{COPY_ROOT}/upstream/run.sh"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def environment(*, offline: bool = False) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env.pop("PYTHONPATH", None)
    env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull,
               PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1")
    if offline:
        env["PATH"] = ""
    return env


def git(repo: Path, *args: str) -> str:
    executable = shutil.which("git")
    require(executable is not None, "Gitが必要です。")
    result = subprocess.run(
        [str(executable), "-c", f"safe.directory={repo}", "-c", "init.templateDir=",
         "-c", f"core.hooksPath={repo / '.disabled-hooks'}", "-c", "commit.gpgSign=false",
         "-C", str(repo), *args],
        env=environment(), capture_output=True, text=True, encoding="utf-8", check=False,
    )
    require(result.returncode == 0, f"Git {args!r}: {result.stderr}")
    return result.stdout.strip()


def init(repo: Path, *, autocrlf: bool) -> None:
    repo.mkdir()
    git(repo, "init", "--initial-branch=main")
    for name, value in (("user.name", "Adoption regression fixture"),
                        ("user.email", "fixture@example.invalid"),
                        ("core.autocrlf", str(autocrlf).lower()), ("core.filemode", "false")):
        git(repo, "config", "--local", name, value)


def cli(script: Path, repo: Path, *args: str, expected: int = 0, offline: bool = False) -> dict[str, Any]:
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-B", *(["-X", "utf8"] if os.name == "nt" else []),
         str(script), "--repo", str(repo), *args],
        cwd=repo.parent, env=environment(offline=offline),
        capture_output=True, text=True, encoding="utf-8", check=False,
    )
    require(result.returncode == expected, f"CLI exit {result.returncode}: {result.stdout}\n{result.stderr}")
    report = json.loads(result.stdout)
    require(isinstance(report, dict), "CLIの出力はJSON objectである必要があります。")
    require(report.get("operational_compliance") == "not_evaluated", "運用準拠は判定しません。")
    return {"exit_code": result.returncode, "report": report}


def index_mode(repo: Path) -> str:
    entries = git(repo, "ls-files", "--stage", "--", SCRIPT).splitlines()
    require(len(entries) == 1, f"run.shのindex entryが1件ではありません: {entries}")
    metadata, path = entries[0].split("\t", 1)
    mode, _oid, stage = metadata.split()
    require(path == SCRIPT and stage == "0", "run.shのindex entryが競合しています。")
    return mode


def create(output: Path, report: dict[str, Any]) -> None:
    require(os.name == "nt", "createはWindows上で実行してOS差を実測してください。")
    # COREの収集だけimportし、検査対象CLIは独立Pythonで実行する。
    sys.path.insert(0, str(ROOT))
    from tools.adoption import CORE

    source, target = output / "source", output / "target"
    init(source, autocrlf=False)
    for name in CORE:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"# Fixture {name}\n".encode())
    (source / "run.sh").write_bytes(b"#!/bin/sh\nexit 0\n")
    git(source, "add", ".")
    git(source, "update-index", "--chmod=+x", "--", "run.sh")
    git(source, "commit", "-m", "Executable source fixture")
    report["source_mode"] = git(source, "ls-tree", "HEAD", "--", "run.sh").split()[0]
    require(report["source_mode"] == "100755", "sourceの実行属性がありません。")

    init(target, autocrlf=True)
    report["copy"] = cli(ROOT / "tools/adoption.py", target, "--source", str(source))
    verifier = target / COPY_ROOT / "verify.py"
    report["offline_check"] = cli(verifier, target, "--check", offline=True)
    manifest = json.loads((target / COPY_ROOT / "manifest.json").read_text(encoding="utf-8"))
    require(manifest["files"]["upstream/run.sh"]["mode"] == "100755", "manifestが実行属性を失いました。")
    git(target, "add", ".")
    report["initial_index_mode"] = index_mode(target)
    require(report["initial_index_mode"] == "100644", "通常git addでのWindows実行属性欠落を再現できません。")
    before = (target / ".git/index").read_bytes()
    report["initial_git_mode_check"] = cli(verifier, target, "--check-git-modes", expected=1)
    require(report["initial_git_mode_check"]["report"]["git_modes"] == "invalid", "属性欠落が検出されません。")
    require(bool(report["initial_git_mode_check"]["report"].get("mode_issues")), "欠落の詳細がありません。")
    require((target / ".git/index").read_bytes() == before, "検査がGit indexを書き換えました。")

    # fixtureの利用者が明示的に登録する。コピーCLIはGit indexを変更しない。
    git(target, "update-index", "--chmod=+x", "--", SCRIPT)
    report["fixed_index_mode"] = index_mode(target)
    require(report["fixed_index_mode"] == "100755", "明示した実行属性が登録されません。")
    before = (target / ".git/index").read_bytes()
    report["fixed_git_mode_check"] = cli(verifier, target, "--check-git-modes")
    require(report["fixed_git_mode_check"]["report"]["git_modes"] == "verified", "実行属性が検証されません。")
    require((target / ".git/index").read_bytes() == before, "検査がGit indexを書き換えました。")
    git(target, "commit", "-m", "Copied workflow with explicit executable mode")
    report["target_commit"] = git(target, "rev-parse", "HEAD")
    report["tree_mode"] = git(target, "ls-tree", "HEAD", "--", SCRIPT).split()[0]
    require(report["tree_mode"] == "100755", "commitで実行属性が失われました。")
    bundle = output / "adoption.bundle"
    git(target, "bundle", "create", str(bundle), "--all")
    git(target, "bundle", "verify", str(bundle))
    report["bundle"] = str(bundle)


def verify(bundle: Path, output: Path, report: dict[str, Any], *, require_posix: bool) -> None:
    require(bundle.is_file(), f"bundleがありません: {bundle}")
    require(not require_posix or os.name == "posix", "この検査はPOSIX runnerを必要とします。")
    clone = output / "clone"
    git(output, "-c", "core.autocrlf=false", "clone", str(bundle), str(clone))
    report["target_commit"] = git(clone, "rev-parse", "HEAD")
    report["tree_mode"] = git(clone, "ls-tree", "HEAD", "--", SCRIPT).split()[0]
    require(report["tree_mode"] == "100755", "bundleのGit treeに実行属性がありません。")
    report["offline_check"] = cli(clone / COPY_ROOT / "verify.py", clone, "--check", offline=True)
    require(report["offline_check"]["report"]["status"] == "verified", "clone後の単独検査に失敗しました。")
    mode = (clone / SCRIPT).stat().st_mode
    report["filesystem_mode"] = oct(stat.S_IMODE(mode))
    report["posix_execution_checked"] = os.name == "posix"
    if os.name == "posix":
        require(bool(mode & 0o111), "Linux checkout後の実行属性がありません。")
        completed = subprocess.run(
            [str(clone / SCRIPT)], cwd=output, env=environment(offline=True),
            capture_output=True, text=True, encoding="utf-8", check=False,
        )
        require(completed.returncode == 0, f"直接実行に失敗: {completed.stderr}")
        report["script_exit_code"] = completed.returncode
    require(git(clone, "status", "--porcelain") == "", "単独検査がcheckoutを変更しました。")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    creator = modes.add_parser("create")
    creator.add_argument("--output", type=Path, required=True)
    verifier = modes.add_parser("verify")
    verifier.add_argument("--bundle", type=Path, required=True)
    verifier.add_argument("--output", type=Path, required=True)
    verifier.add_argument("--require-posix", action="store_true")
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    # 既存ディレクトリへ書かず、fixtureの掃除や既存repoの更新も行わない。
    output.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {"mode": args.mode, "os": os.name, "status": "running"}
    try:
        if args.mode == "create":
            create(output, report)
        else:
            verify(args.bundle.expanduser().resolve(), output, report, require_posix=args.require_posix)
        report["status"] = "passed"
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
    serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    (output / "results.json").write_text(serialized, encoding="utf-8", newline="\n")
    print(serialized, end="")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())