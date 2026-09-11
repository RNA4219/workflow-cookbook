# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tools import adoption

ROOT = Path(__file__).resolve().parents[1]


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={repo}", "-C", str(repo), *args],
        capture_output=True, check=True, encoding="utf-8",
    )
    return result.stdout.strip()


@pytest.fixture()
def source(tmp_path: Path) -> Path:
    repo = tmp_path / "source"
    repo.mkdir()
    git(repo, "init")
    git(repo, "config", "user.email", "fixture@example.invalid")
    git(repo, "config", "user.name", "Fixture")
    git(repo, "config", "core.autocrlf", "false")
    for name in adoption.CORE:
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"# Original {name}\n".encode())
    (repo / "日本語.bin").write_bytes(bytes(range(256)))
    (repo / "run.sh").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    (repo / ".gitattributes").write_text("日本語.bin export-ignore\n", encoding="utf-8")
    git(repo, "add", ".")
    git(repo, "update-index", "--chmod=+x", "run.sh")
    git(repo, "commit", "-m", "fixture")
    # ローカルの変更や未追跡データをコピーへ混ぜない。
    (repo / "HUB.codex.md").write_text("uncommitted change", encoding="utf-8")
    (repo / "private-local.txt").write_text("local only", encoding="utf-8")
    return repo


@pytest.fixture()
def target(tmp_path: Path) -> Path:
    target = tmp_path / "日本語 target"
    target.mkdir()
    return target


def cli(target: Path, *args: str, cwd: Path = ROOT) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    return subprocess.run(
        [sys.executable, "-B", "-m", "tools.adoption", "--repo", str(target), *args],
        cwd=cwd, env=env, capture_output=True, text=True, encoding="utf-8", check=False,
    )


def test_real_git_snapshot_ignores_working_changes_and_retains_all_blobs(source: Path) -> None:
    snap = adoption.snapshot(source, "HEAD")
    assert snap.commit == git(source, "rev-parse", "HEAD")
    assert snap.tree == git(source, "rev-parse", "HEAD^{tree}")
    assert snap.files["HUB.codex.md"] == b"# Original HUB.codex.md\n"
    assert "private-local.txt" not in snap.files
    assert snap.files["日本語.bin"] == bytes(range(256))
    assert snap.modes["run.sh"] == "100755"
    assert set(snap.files) == set(git(source, "-c", "core.quotePath=false", "ls-files").splitlines())


@pytest.mark.parametrize("prefix", [None, b"", b"# Local\r\n", b"# No final newline", "\ufeff# 日本語\n".encode()])
def test_copy_preserves_agents_and_repeated_run_is_byte_identical(source: Path, target: Path, prefix: bytes | None) -> None:
    agents = target / "AGENTS.md"
    if prefix is not None:
        agents.write_bytes(prefix)
    (target / "README.md").write_bytes(b"project readme")
    result = adoption.copy_workflow(target, source)
    assert result["status"] == "copied"
    assert result["operational_compliance"] == "not_evaluated"
    assert agents.read_bytes().startswith(prefix or b"")
    assert agents.read_bytes().count(adoption.BLOCK) == 1
    before = {p.relative_to(target): (p.read_bytes(), p.stat().st_mtime_ns) for p in target.rglob("*") if p.is_file()}
    assert adoption.copy_workflow(target, source)["status"] == "unchanged"
    assert {p.relative_to(target): (p.read_bytes(), p.stat().st_mtime_ns) for p in target.rglob("*") if p.is_file()} == before
    manifest = adoption.verify(target)
    assert manifest["source_commit"] == result["source_commit"]
    assert (target / "README.md").read_bytes() == b"project readme"
    assert result["upstream_files"] == len(adoption.CORE) + 3


def test_dry_run_is_read_only(source: Path, target: Path) -> None:
    result = adoption.copy_workflow(target, source, dry_run=True)
    assert result["status"] == "planned"
    assert list(target.iterdir()) == []


def test_cli_apply_check_and_error_are_json(source: Path, target: Path) -> None:
    copied = cli(target, "--source", str(source))
    assert copied.returncode == 0, copied.stdout + copied.stderr
    assert json.loads(copied.stdout)["status"] == "copied"
    checked = cli(target, "--check", "--source", str(target / "absent"))
    assert checked.returncode == 0
    assert json.loads(checked.stdout)["status"] == "verified"
    (target / "workflow-cookbook/upstream/LICENSE").unlink()
    failed = cli(target, "--check")
    assert failed.returncode == 1
    assert json.loads(failed.stdout)["operational_compliance"] == "not_evaluated"


@pytest.mark.parametrize("change", ["modify", "delete", "add", "agents", "duplicate", "manifest", "file_info", "commit", "format"])
def test_check_detects_damage(source: Path, target: Path, change: str) -> None:
    adoption.copy_workflow(target, source)
    root = target / adoption.DIRECTORY
    path = root / "upstream/README.md"
    if change == "modify":
        path.write_bytes(b"modified")
    elif change == "delete":
        path.unlink()
    elif change == "add":
        (root / "extra.txt").write_bytes(b"added")
    elif change == "agents":
        (target / "AGENTS.md").write_bytes(b"# Local only\n")
    elif change == "duplicate":
        with (target / "AGENTS.md").open("ab") as handle:
            handle.write(adoption.BLOCK)
    elif change == "manifest":
        (root / "manifest.json").write_text("[]", encoding="utf-8")
    else:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if change == "file_info":
            manifest["files"]["upstream/README.md"] = "invalid"
        elif change == "commit":
            manifest["source_commit"] = "invalid"
        else:
            manifest["format_version"] = 99
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert adoption.main(["--repo", str(target), "--check"]) == 1
    with pytest.raises((ValueError, TypeError, KeyError)):
        adoption.copy_workflow(target, source)


def test_local_agents_edits_outside_managed_block_are_allowed(source: Path, target: Path) -> None:
    adoption.copy_workflow(target, source)
    agents = target / "AGENTS.md"
    agents.write_bytes(b"# Updated project policy\n\n" + agents.read_bytes() + b"\nLocal note\n")
    assert adoption.copy_workflow(target, source)["status"] == "unchanged"


@pytest.mark.parametrize("kind", ["directory", "file", "partial_block", "non_utf8"])
def test_conflict_does_not_change_existing_files(source: Path, target: Path, kind: str) -> None:
    if kind == "directory":
        (target / adoption.DIRECTORY).mkdir()
    elif kind == "file":
        (target / adoption.DIRECTORY).write_bytes(b"existing")
    elif kind == "partial_block":
        (target / "AGENTS.md").write_bytes(adoption.START)
    else:
        (target / "AGENTS.md").write_bytes(b"\xff")
    before = {p.relative_to(target): p.read_bytes() for p in target.rglob("*") if p.is_file()}
    assert adoption.main(["--repo", str(target), "--source", str(source)]) == 1
    assert {p.relative_to(target): p.read_bytes() for p in target.rglob("*") if p.is_file()} == before


def test_different_commit_fails_without_overwrite(source: Path, target: Path) -> None:
    adoption.copy_workflow(target, source)
    old = (target / adoption.DIRECTORY / "manifest.json").read_bytes()
    git(source, "add", "HUB.codex.md")
    git(source, "commit", "-m", "update")
    with pytest.raises(ValueError, match="別版"):
        adoption.copy_workflow(target, source)
    assert (target / adoption.DIRECTORY / "manifest.json").read_bytes() == old


@pytest.mark.parametrize("relation", ["same", "nested_target", "parent_target", "missing_target"])
def test_rejects_overlapping_or_missing_target(source: Path, relation: str) -> None:
    target = source
    if relation == "nested_target":
        target = source / "nested"
        target.mkdir()
    elif relation == "parent_target":
        target = source.parent
    elif relation == "missing_target":
        target = source.parent / "missing"
    with pytest.raises(ValueError):
        adoption.copy_workflow(target, source)
    assert not (target / adoption.DIRECTORY).exists()


@pytest.mark.parametrize("name", ["../file", "/absolute", "C:drive", "a\\b", "a//b", "a/./b", ".git/config", "NUL.txt", "file.", "file "])
def test_rejects_nonportable_source_names(name: str) -> None:
    with pytest.raises(ValueError):
        adoption._path(name)


def test_source_and_revision_require_real_complete_git_root(source: Path) -> None:
    with pytest.raises(ValueError, match="Gitルート"):
        adoption.snapshot(source / "docs", "HEAD")
    with pytest.raises(ValueError, match="Git"):
        adoption.snapshot(source, "--help")
    git(source, "rm", "LICENSE")
    git(source, "commit", "-m", "remove core")
    with pytest.raises(ValueError, match="必須"):
        adoption.snapshot(source, "HEAD")


def test_git_is_not_required_for_offline_check(source: Path, target: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    adoption.copy_workflow(target, source)
    monkeypatch.setattr(shutil, "which", lambda _: None)
    assert adoption.main(["--repo", str(target), "--check"]) == 0
    with pytest.raises(ValueError, match="Gitが必要"):
        adoption.snapshot(source, "HEAD")


def test_git_symlink_entry_is_rejected_without_materializing_it(source: Path) -> None:
    oid = git(source, "hash-object", "README.md")
    git(source, "update-index", "--add", "--cacheinfo", f"120000,{oid},reference-link")
    git(source, "commit", "-m", "link mode")
    with pytest.raises(ValueError, match="link/submodule"):
        adoption.snapshot(source, "HEAD")


def test_git_case_collisions_are_rejected(source: Path) -> None:
    oid = git(source, "hash-object", "README.md")
    git(source, "update-index", "--add", "--cacheinfo", f"100644,{oid},readme.md")
    git(source, "commit", "-m", "case collision")
    with pytest.raises(ValueError, match="衝突"):
        adoption.snapshot(source, "HEAD")


@pytest.mark.parametrize("which", ["target", "agents", "managed_dir", "managed_file"])
def test_reparse_points_are_refused(source: Path, target: Path, monkeypatch: pytest.MonkeyPatch, which: str) -> None:
    adoption.copy_workflow(target, source)
    paths = {
        "target": target, "agents": target / "AGENTS.md",
        "managed_dir": target / adoption.DIRECTORY / "upstream/docs",
        "managed_file": target / adoption.DIRECTORY / "upstream/README.md",
    }
    actual = adoption._linked
    monkeypatch.setattr(adoption, "_linked", lambda path: path == paths[which] or actual(path))
    assert adoption.main(["--repo", str(target), "--check"]) == 1


@pytest.mark.parametrize("existing", [True, False])
def test_post_install_failure_preserves_published_files(source: Path, target: Path, monkeypatch: pytest.MonkeyPatch, existing: bool) -> None:
    if existing:
        (target / "AGENTS.md").write_bytes(b"Original")
    def fail(_: Path) -> dict:
        raise ValueError("verification failure")
    monkeypatch.setattr(adoption, "verify", fail)
    with pytest.raises(ValueError, match="verification"):
        adoption.copy_workflow(target, source)
    assert (target / adoption.DIRECTORY).is_dir()
    assert not list(target.glob(".wfc-copy-*"))
    assert adoption.BLOCK in (target / "AGENTS.md").read_bytes()
    if existing:
        assert (target / "AGENTS.md").read_bytes().startswith(b"Original")


def test_agents_append_failure_keeps_installed_tree(source: Path, target: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (target / "AGENTS.md").write_bytes(b"Original")
    def fail(*_: object) -> None:
        raise OSError("append failure")
    monkeypatch.setattr(adoption, "_append_agents", fail)
    with pytest.raises(ValueError, match="append failure"):
        adoption.copy_workflow(target, source)
    assert (target / "AGENTS.md").read_bytes() == b"Original"
    assert (target / adoption.DIRECTORY).exists()


def test_concurrent_preparation_edit_is_preserved(source: Path, target: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (target / "AGENTS.md").write_bytes(b"Original")
    original = Path.write_bytes
    def write(path: Path, data: bytes) -> int:
        result = original(path, data)
        if path.name == "verify.py" and path.parent.parent.name.startswith(".wfc-copy-"):
            original(target / "AGENTS.md", b"Concurrent edit")
        return result
    monkeypatch.setattr(Path, "write_bytes", write)
    with pytest.raises(ValueError, match="準備中"):
        adoption.copy_workflow(target, source)
    assert (target / "AGENTS.md").read_bytes() == b"Concurrent edit"
    assert not (target / adoption.DIRECTORY).exists()


@pytest.mark.parametrize("edited", ["agents", "upstream"])
def test_rollback_keeps_post_install_user_edits(source: Path, target: Path, monkeypatch: pytest.MonkeyPatch, edited: str) -> None:
    changed = target / ("AGENTS.md" if edited == "agents" else "workflow-cookbook/upstream/README.md")
    def fail(_: Path) -> dict:
        changed.write_bytes(b"Concurrent edit")
        raise ValueError("changed during verification")
    monkeypatch.setattr(adoption, "verify", fail)
    with pytest.raises(ValueError, match="changed"):
        adoption.copy_workflow(target, source)
    assert changed.read_bytes() == b"Concurrent edit"
    assert (target / adoption.DIRECTORY).exists()


def test_manifest_cannot_redefine_adoption_contract(source: Path, target: Path) -> None:
    adoption.copy_workflow(target, source)
    root = target / adoption.DIRECTORY
    (root / "ADOPTION.md").write_bytes(b"changed policy")
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["files"]["ADOPTION.md"]["sha256"] = hashlib.sha256(b"changed policy").hexdigest()
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="導入契約"):
        adoption.verify(target)


def test_reference_instructions_do_not_claim_operational_pass() -> None:
    text = adoption.adoption_text("a" * 40).decode()
    for expected in ("HATE/QEG", "agent-taskstate", "全履歴を毎回", "一律の追加承認", "not_evaluated", "導入先の完了タスク"):
        assert expected in text


def test_bundled_verifier_runs_without_checkout_git_or_site_packages(source: Path, target: Path) -> None:
    adoption.copy_workflow(target, source)
    env = os.environ.copy()
    env["PATH"] = ""
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-B", str(target / "workflow-cookbook/verify.py"),
         "--repo", str(target), "--check"],
        cwd=target, env=env, capture_output=True, encoding="utf-8", check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["status"] == "verified"


def test_preparation_write_failure_leaves_no_partial_install(source: Path, target: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write = Path.write_bytes
    def fail(path: Path, data: bytes) -> int:
        if path.name == "LICENSE":
            raise OSError("write failure")
        return write(path, data)
    monkeypatch.setattr(Path, "write_bytes", fail)
    with pytest.raises(OSError, match="write failure"):
        adoption.copy_workflow(target, source)
    assert list(target.iterdir()) == []


@pytest.mark.parametrize("edit", ["replace_contents", "append"])
def test_edit_between_agents_comparison_and_write_is_not_lost(source: Path, target: Path, monkeypatch: pytest.MonkeyPatch, edit: str) -> None:
    agents = target / "AGENTS.md"
    agents.write_bytes(b"Original")
    write = os.write
    concurrent = b"Concurrent edit" if edit == "replace_contents" else b"Original plus local note"
    def racing_write(fd: int, content: bytes) -> int:
        if os.path.samestat(os.fstat(fd), agents.stat()):
            agents.write_bytes(concurrent)
        return write(fd, content)
    monkeypatch.setattr(os, "write", racing_write)
    with pytest.raises(ValueError, match="追記中"):
        adoption.copy_workflow(target, source)
    assert agents.read_bytes() == concurrent + b"\n\n" + adoption.BLOCK
    assert (target / adoption.DIRECTORY).is_dir()


def test_new_agents_creation_collision_keeps_user_file(source: Path, target: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    agents = target / "AGENTS.md"
    original_open = os.open
    def racing_open(path: object, flags: int, mode: int = 0o777, **kwargs: object) -> int:
        if path == agents:
            agents.write_bytes(b"Created by user")
        return original_open(path, flags, mode, **kwargs)
    monkeypatch.setattr(os, "open", racing_open)
    with pytest.raises(ValueError, match="公開後"):
        adoption.copy_workflow(target, source)
    assert agents.read_bytes() == b"Created by user"


def test_incomplete_append_is_reported_without_truncating_user_bytes(source: Path, target: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    agents = target / "AGENTS.md"
    agents.write_bytes(b"Original")
    write = os.write
    def short_write(fd: int, data: bytes) -> int:
        if os.path.samestat(os.fstat(fd), agents.stat()):
            return write(fd, data[:3])
        return write(fd, data)
    monkeypatch.setattr(os, "write", short_write)
    with pytest.raises(ValueError, match="不完全"):
        adoption.copy_workflow(target, source)
    assert agents.read_bytes().startswith(b"Original")
    assert (target / adoption.DIRECTORY).is_dir()


def test_append_refuses_changed_prefix_and_identity(target: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    agents = target / "AGENTS.md"
    agents.write_bytes(b"Changed")
    with pytest.raises(ValueError, match="追記前"):
        adoption._append_agents(agents, b"Original", adoption.BLOCK)
    monkeypatch.setattr(os.path, "samestat", lambda *_: False)
    with pytest.raises(ValueError, match="実体"):
        adoption._append_agents(agents, b"Changed", adoption.BLOCK)
    assert agents.read_bytes() == b"Changed"


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits are checked on Linux CI")
def test_new_agents_is_not_group_or_world_accessible_without_umask(target: Path) -> None:
    agents = target / "AGENTS.md"
    previous = os.umask(0)
    try:
        adoption._append_agents(agents, None, adoption.BLOCK)
    finally:
        os.umask(previous)
    assert agents.stat().st_mode & 0o077 == 0
    assert agents.read_bytes() == adoption.BLOCK
