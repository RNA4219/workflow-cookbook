# SPDX-License-Identifier: MIT
"""固定コミットのWorkflow Cookbookを一括コピーし、整合性を検査する。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DIRECTORY = "workflow-cookbook"
START = b"<!-- workflow-cookbook:full:v1 -->"
END = b"<!-- /workflow-cookbook:full:v1 -->"
BLOCK = (
    START.decode()
    + "\n## Workflow Cookbook\n\n"
    + "共有ワークフローは [導入契約](workflow-cookbook/ADOPTION.md) と\n"
    + "[正本HUB](workflow-cookbook/upstream/HUB.codex.md) に従う。\n"
    + "開始時に導入契約を読み、対象作業に必要な原文・実行手順・検証契約を参照する。\n"
    + "上位指示、ユーザーの作業許可、導入先固有の指示を確認し、衝突は根拠付きで扱う。\n"
    + "コピー元の実績を導入先の検証結果として扱わない。\n"
    + END.decode()
    + "\n"
).encode("utf-8")
CORE = (
    "README.md", "HUB.codex.md", "BLUEPRINT.md", "RUNBOOK.md", "GUARDRAILS.md",
    "EVALUATION.md", "CHECKLISTS.md", "TASK.codex.md", "LICENSE", "pyproject.toml",
    "docs/CONTRACTS.md", "docs/birdseye/index.json", "docs/birdseye/hot.json",
    "docs/TASKS.md", "docs/ci-config.md", "docs/workflow-evolution.md", "docs/acceptance/README.md",
    "docs/contracts/task-context-continuity.md", "docs/contracts/evaluation-identity-contract.md",
)
RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


@dataclass(frozen=True)
class Snapshot:
    commit: str
    tree: str
    files: dict[str, bytes]
    modes: dict[str, str]


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _path(name: str) -> str:
    parts = name.split("/")
    if any(
        part in ("", ".", "..") or part.lower() == ".git" or part.endswith((".", " "))
        or part.split(".")[0].upper() in RESERVED
        or re.search(r'[\\<>:"|?*\x00-\x1f]', part)
        for part in parts
    ):
        raise ValueError(f"非対応の相対パス: {name!r}")
    return name


def _linked(path: Path) -> bool:
    info = path.lstat()
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, "st_file_attributes", 0) & 0x400)


def _regular(path: Path) -> bytes:
    if _linked(path) or not path.is_file():
        raise ValueError(f"通常ファイルではありません: {path}")
    return path.read_bytes()


def _git(source: Path, *args: str, data: bytes | None = None) -> bytes:
    executable = shutil.which("git")
    if executable is None:
        raise ValueError("コピーにはGitが必要です。")
    result = subprocess.run(
        [executable, "-c", f"safe.directory={source}", "-C", str(source), *args],
        input=data, capture_output=True, check=False,
    )
    if result.returncode:
        raise ValueError(f"Gitに失敗しました: {result.stderr.decode('utf-8', errors='replace').strip()}")
    return result.stdout


def snapshot(source: Path, ref: str) -> Snapshot:
    source = source.resolve()
    top = Path(_git(source, "rev-parse", "--show-toplevel").decode("utf-8").strip()).resolve()
    if top != source:
        raise ValueError("--sourceはWorkflow CookbookのGitルートを指定してください。")
    commit = _git(source, "rev-parse", "--verify", "--end-of-options", f"{ref}^{{commit}}").decode().strip()
    tree = _git(source, "rev-parse", f"{commit}^{{tree}}").decode().strip()
    entries: dict[str, tuple[str, str]] = {}
    folded: set[str] = set()
    for entry in _git(source, "ls-tree", "-rz", "--full-tree", commit).split(b"\0"):
        if not entry:
            continue
        header, raw_name = entry.split(b"\t", 1)
        mode, kind, oid = header.decode().split()
        name = _path(raw_name.decode("utf-8"))
        if mode not in ("100644", "100755") or kind != "blob":
            raise ValueError(f"link/submodule等はコピーできません: {name}")
        if name.casefold() in folded:
            raise ValueError(f"大文字小文字で衝突するパス: {name}")
        folded.add(name.casefold())
        entries[name] = (mode, oid)
    if not set(CORE) <= entries.keys():
        raise ValueError("Workflow Cookbookの必須ファイルがないコミットです。")
    ids = list(dict.fromkeys(oid for _, oid in entries.values()))
    output = _git(source, "cat-file", "--batch", data=("\n".join(ids) + "\n").encode())
    blobs: dict[str, bytes] = {}
    position = 0
    for oid in ids:
        boundary = output.index(b"\n", position)
        returned, kind, raw_size = output[position:boundary].decode().split()
        size = int(raw_size)
        position = boundary + 1
        blob = output[position:position + size]
        if returned != oid or kind != "blob" or len(blob) != size or output[position + size:position + size + 1] != b"\n":
            raise ValueError("Git blobの応答が不完全です。")
        blobs[oid] = blob
        position += size + 1
    return Snapshot(commit, tree, {name: blobs[oid] for name, (_, oid) in entries.items()},
                    {name: mode for name, (mode, _) in entries.items()})


def adoption_text(commit: str) -> bytes:
    return f"""# Workflow Cookbookの適用契約

コピー元コミット: {commit}

## 使い始める

1. 導入先のREADME、AGENTS、既存仕様を読み、目的・制約・完了条件を確認する。
2. [共有HUB](upstream/HUB.codex.md)、[Guardrails](upstream/GUARDRAILS.md)、
   [Blueprint](upstream/BLUEPRINT.md)、[Runbook](upstream/RUNBOOK.md)、
   [Evaluation](upstream/EVALUATION.md)から対象作業の原文へ進む。
3. [Task Seed](upstream/TASK.codex.md)、[タスク運用](upstream/docs/TASKS.md)、
   [Acceptance](upstream/docs/acceptance/)、[チェックリスト](upstream/CHECKLISTS.md)に従い、
   導入先自身の要求・変更・検証結果を導入先の文書へ記録する。
4. 参照の索引は[Birdseye](upstream/docs/birdseye/index.json)を使う。
   この索引の対象はコピーされた上流資料。導入先コードは通常検索または導入先の索引で調べる。
   鮮度不明・未登録・不適合なら原文へ戻り、生成済みを確認済みとしない。

## 文脈を維持する

[文脈継続契約](upstream/docs/contracts/task-context-continuity.md)、
[取得・観測・再開](upstream/docs/workflow-evolution.md)を参照する。
目的・制約・進捗・決定・未解決事項を保持し、ユーザーの変更を反映する。
agent-taskstateは接続できるときに既存のstate、decision、question、checkpointを使う。
履歴は保存して必要時に参照する。全履歴を毎回プロンプトへ復元する義務は追加しない。
コピー元にその義務を今後の必須改修と読める記載がある場合も、この導入契約では採用しない。
固定の100行/2ファイル制限、ツール呼び出しの本文JSONへの重複出力は要求しない。

## 検証と外部接続

[契約一覧](upstream/docs/CONTRACTS.md)、[CI構成](upstream/docs/ci-config.md)、
[評価identity](upstream/docs/contracts/evaluation-identity-contract.md)を参照する。
HATE/QEG等の接続はコピーしたtools、schemas、examplesと関連契約から構成し、
外部ツール本体・認証・環境パス・測定集合を導入先で確認する。
比較評価ではpreflight/postrunを通し、通常の単体試験をモデル性能測定として報告しない。
CIやブランチ保護、owner、KPI、承認条件は導入先の契約に合わせる。
ユーザーの許可と適用条件を確認せず、一律の追加承認ルールを作らない。
対応runnerのないツールを実行したことにしない。

## コピーの範囲と残る作業

upstreamにはGitの固定コミットにある全ファイルをそのまま収録した。
コードと文書の相対パスを維持するため、上流の過去タスク・評価資料も含む。
これらは上流の記録であり、導入先の完了タスク・実測値・合格証跡には流用しない。
上位指示、ユーザーが許可した作業、導入先固有の指示を確認して共有契約を適用する。
相互に両立しない条件、適用対象外、未接続の外部ツールは理由を記録する。
導入先の仕様・Task/Acceptance・CI・外部接続・実際の受入試験は、対象作業で整える。

upstreamは版を固定した参照領域として保持する。生成物・仮想環境・作業結果は領域外へ置き、
Pythonは-BまたはPYTHONDONTWRITEBYTECODE=1で実行してキャッシュの追加を避ける。
コード実行が必要な場合はupstreamを実行ルートとし、Python依存はpyproject.tomlを参照する。
相対入力の基準を確認し、--repo/--repo-root等に導入先のパスを指定する。
upstreamの.git履歴は収録していない。Git履歴を必要とする操作は導入先のGitルートを指定する。

コピー先の単独検査: python -B workflow-cookbook/verify.py --repo . --check
導入先ルートで実行でき、元checkout・Git・追加Python依存は不要。
公開CLIからも wfc-copy --repo <導入先> --check で同じ検査ができる。
verify.pyは導入時のCLIコードをそのまま同梱し、manifestにhashを記録する。
コピー成功は運用準拠・テスト合格の認定ではない。運用状態はnot_evaluatedのままとする。
""".encode("utf-8")


def _inventory(root: Path) -> dict[str, bytes]:
    if _linked(root) or not root.is_dir():
        raise ValueError(f"コピー領域が通常ディレクトリではありません: {root}")
    result: dict[str, bytes] = {}
    for parent, dirs, files in os.walk(root, followlinks=False):
        for name in dirs:
            if _linked(Path(parent) / name):
                raise ValueError(f"コピー領域にリンクがあります: {name}")
        for name in files:
            path = Path(parent) / name
            result[path.relative_to(root).as_posix()] = _regular(path)
    return result


def _report(status: str, manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": status, "source_commit": manifest["source_commit"], "source_tree": manifest["source_tree"],
        "upstream_files": sum(name.startswith("upstream/") for name in manifest["files"]),
        "operational_compliance": "not_evaluated",
    }


def verify(repo: Path) -> dict[str, Any]:
    root = repo / DIRECTORY
    files = _inventory(root)
    manifest = json.loads(files.pop("manifest.json"))
    if not isinstance(manifest, dict) or manifest.get("format_version") != 1:
        raise ValueError("非対応のmanifestです。")
    for field in ("source_commit", "source_tree"):
        if not isinstance(manifest.get(field), str) or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", manifest[field]):
            raise ValueError(f"不正な{field}です。")
    expected = manifest.get("files")
    if not isinstance(expected, dict) or not {"ADOPTION.md", "verify.py"} <= expected.keys():
        raise ValueError("manifestのファイル一覧が不正です。")
    if set(expected) != set(files) or not {f"upstream/{name}" for name in CORE} <= expected.keys():
        raise ValueError("コピー領域のファイルが追加・欠落しています。")
    for name, details in expected.items():
        _path(name)
        if not isinstance(details, dict) or details.get("mode") not in ("100644", "100755"):
            raise ValueError(f"不正なファイル情報: {name}")
        if details.get("sha256") != _digest(files[name]):
            raise ValueError(f"コピーしたファイルが改変されています: {name}")
        if os.name != "nt" and bool((root / name).stat().st_mode & 0o111) != (details["mode"] == "100755"):
            raise ValueError(f"実行属性が変化しています: {name}")
    if files["ADOPTION.md"] != adoption_text(manifest["source_commit"]):
        raise ValueError("導入契約が一致しません。")
    agents = _regular(repo / "AGENTS.md")
    if agents.count(START) != 1 or agents.count(END) != 1 or agents.count(BLOCK) != 1:
        raise ValueError("AGENTS.mdの管理ブロックが欠落・改変・重複しています。")
    return manifest


def _target(repo: Path) -> Path:
    target = Path(os.path.abspath(repo.expanduser()))
    if not target.is_dir():
        raise ValueError("--repoには既存の導入先ディレクトリを指定してください。")
    for component in (target, *target.parents):
        if _linked(component):
            raise ValueError(f"リンク経由の導入先は使えません: {component}")
    return target



def _append_agents(path: Path, original: bytes | None, addition: bytes) -> None:
    # O_APPENDで既存部分を置換しない。新規作成はO_EXCLで競合を検出する。
    flags = os.O_RDWR | os.O_APPEND | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    if original is None:
        flags |= os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o666)
    with os.fdopen(descriptor, "r+b", buffering=0) as handle:
        opened = os.fstat(handle.fileno())
        if not stat.S_ISREG(opened.st_mode) or _linked(path) or not os.path.samestat(opened, path.lstat()):
            raise ValueError("AGENTS.mdの実体が変化しました。")
        prefix = original or b""
        if handle.read() != prefix:
            raise ValueError("追記前にAGENTS.mdが更新されました。")
        # 比較後に他の編集が入っても、追記はその既存バイト列を上書きしない。
        if os.write(handle.fileno(), addition) != len(addition):
            raise OSError("AGENTS.mdへの追記が不完全です。")
        handle.seek(0)
        if handle.read() != prefix + addition or not os.path.samestat(opened, path.lstat()):
            raise ValueError("追記中にAGENTS.mdが更新されました。変更を保持します。")


def copy_workflow(repo: Path, source: Path, ref: str = "HEAD", *, dry_run: bool = False) -> dict[str, Any]:
    repo = _target(repo)
    source = source.expanduser().resolve()
    if source == repo or source in repo.parents or repo in source.parents:
        raise ValueError("コピー元と導入先を包含関係のない場所へ分離してください。")
    snap = snapshot(source, ref)
    root = repo / DIRECTORY
    if root.exists() or root.is_symlink():
        manifest = verify(repo)
        if manifest["source_commit"] != snap.commit or manifest["source_tree"] != snap.tree:
            raise ValueError("別版のコピーが存在します。独自差分を確認してから別領域へ導入してください。")
        expected = {f"upstream/{name}": content for name, content in snap.files.items()}
        expected["ADOPTION.md"] = adoption_text(snap.commit)
        expected["verify.py"] = Path(__file__).read_bytes()
        if {name: info["sha256"] for name, info in manifest["files"].items()} != {
            name: _digest(content) for name, content in expected.items()
        }:
            raise ValueError("コピー元の固定コミットまたは導入ツールと内容が一致しません。")
        return _report("unchanged", manifest)
    agents_path = repo / "AGENTS.md"
    original = _regular(agents_path) if agents_path.exists() or agents_path.is_symlink() else None
    if original is not None:
        original.decode("utf-8-sig")
        if START in original or END in original:
            raise ValueError("AGENTS.mdに既存または不完全な管理ブロックがあります。")
    payload = {f"upstream/{name}": content for name, content in snap.files.items()}
    payload["ADOPTION.md"] = adoption_text(snap.commit)
    payload["verify.py"] = Path(__file__).read_bytes()
    manifest = {
        "format_version": 1, "source_commit": snap.commit, "source_tree": snap.tree,
        "files": {name: {"sha256": _digest(content), "mode": (snap.modes[name.removeprefix("upstream/")] if name.startswith("upstream/") else "100644")}
                  for name, content in payload.items()},
    }
    if dry_run:
        return _report("planned", manifest)
    prefix = original or b""
    separator = b"" if not prefix else (b"\n" if prefix.endswith(b"\n") else b"\n\n")
    addition = separator + BLOCK
    with tempfile.TemporaryDirectory(prefix=".wfc-copy-", dir=repo) as temporary:
        stage = Path(temporary) / DIRECTORY
        stage.mkdir()
        for name, content in payload.items():
            destination = stage / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
            destination.chmod(0o755 if manifest["files"][name]["mode"] == "100755" else 0o644)
        (stage / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        # 全候補を準備後に既存状態を再確認する。
        current = _regular(agents_path) if agents_path.exists() or agents_path.is_symlink() else None
        if current != original or root.exists() or root.is_symlink():
            raise ValueError("準備中に導入先が変化しました。")
        stage.rename(root)
        try:
            _append_agents(agents_path, original, addition)
            verify(repo)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            # 公開後の削除・全体復元も並行編集を失い得るため、両方を残して要確認とする。
            raise ValueError(f"公開後の検査に失敗しました。コピー領域とAGENTS.mdを保持します: {exc}") from exc
    return _report("copied", manifest)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True, help="既存の導入先ディレクトリ")
    parser.add_argument("--source", type=Path, default=Path.cwd(), help="コピー元checkout（既定: cwd）")
    parser.add_argument("--ref", default="HEAD", help="固定するGit revision（既定: HEAD）")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--dry-run", action="store_true", help="書き込まず導入計画を表示")
    group.add_argument("--check", action="store_true", help="コピー元への接続なしで整合性検査")
    args = parser.parse_args(argv)
    try:
        if args.check:
            result = _report("verified", verify(_target(args.repo)))
        else:
            result = copy_workflow(args.repo, args.source, args.ref, dry_run=args.dry_run)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "error", "error": str(exc), "operational_compliance": "not_evaluated"}, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
