from __future__ import annotations

import json
import re
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Iterable, Mapping, Sequence

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.codemap import update


def test_ensure_python_version_exits(monkeypatch, capsys):
    monkeypatch.setattr(update, "sys", SimpleNamespace(version_info=(3, 10, 0)))

    with pytest.raises(SystemExit) as excinfo:
        update.ensure_python_version()

    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert "Python 3.11 or newer is required" in captured.out


def test_since_command_resolves_capsules_for_non_birdseye_diff(monkeypatch):
    options = update.parse_args(["--since", "--emit", "index+caps"])

    def fake_run(args, capture_output, text, check, cwd):
        assert args == [
            "git",
            "diff",
            "--name-status",
            "--find-renames",
            "main...HEAD",
        ]
        assert cwd == update._REPO_ROOT
        return SimpleNamespace(stdout="M\tREADME.md\n")

    monkeypatch.setattr(update.subprocess, "run", fake_run)
    options = replace(options, diff_resolver=update.GitDiffResolver())

    resolved = options.resolve_targets()

    expected = update._REPO_ROOT / "docs" / "birdseye" / "caps" / "README.md.json"
    assert expected in resolved


def test_since_command_falls_back_to_default_targets(monkeypatch):
    options = update.parse_args(["--since"])

    options = replace(options, diff_resolver=SimpleNamespace(resolve=lambda _: ()))

    resolved = options.resolve_targets()

    expected_index = (update._REPO_ROOT / "docs" / "birdseye" / "index.json").resolve()
    expected_hot = (update._REPO_ROOT / "docs" / "birdseye" / "hot.json").resolve()
    expected_caps = (update._REPO_ROOT / "docs" / "birdseye" / "caps").resolve()

    assert resolved == (expected_index, expected_hot, expected_caps)


def test_git_diff_resolver_parses_rename_status(monkeypatch):
    captured = {}

    def fake_run(args, capture_output, text, check, cwd):
        captured["args"] = args
        assert capture_output is True
        assert text is True
        assert check is True
        captured["cwd"] = cwd
        return SimpleNamespace(stdout="R100\tREADME.md\tRUNBOOK.md\n")

    monkeypatch.setattr(update.subprocess, "run", fake_run)

    resolver = update.GitDiffResolver()
    resolved = resolver.resolve("main")

    assert captured["args"] == [
        "git",
        "diff",
        "--name-status",
        "--find-renames",
        "main...HEAD",
    ]
    assert captured["cwd"] == update._REPO_ROOT
    assert resolved == (
        Path("docs/birdseye/caps/README.md.json"),
        Path("docs/birdseye/caps/RUNBOOK.md.json"),
    )


def test_git_diff_resolver_parses_copy_status(monkeypatch):
    captured = {}

    def fake_run(args, capture_output, text, check, cwd):
        captured.update(
            {
                "args": args,
                "capture_output": capture_output,
                "text": text,
                "check": check,
                "cwd": cwd,
            }
        )
        return SimpleNamespace(stdout="C100\tREADME.md\tRUNBOOK.md\n")

    monkeypatch.setattr(update.subprocess, "run", fake_run)

    resolver = update.GitDiffResolver()
    resolved = resolver.resolve("main")

    assert captured["args"] == [
        "git",
        "diff",
        "--name-status",
        "--find-renames",
        "--find-copies",
        "main...HEAD",
    ]
    assert captured["capture_output"] is True
    assert captured["text"] is True
    assert captured["check"] is True
    assert captured["cwd"] == update._REPO_ROOT
    assert resolved == (
        Path("docs/birdseye/caps/README.md.json"),
        Path("docs/birdseye/caps/RUNBOOK.md.json"),
    )


def test_git_diff_resolver_parses_type_change(monkeypatch):
    def fake_run(args, capture_output, text, check, cwd):
        return SimpleNamespace(stdout="T\tREADME.md\n")

    monkeypatch.setattr(update.subprocess, "run", fake_run)

    resolver = update.GitDiffResolver()
    resolved = resolver.resolve("main")

    assert resolved == (Path("docs/birdseye/caps/README.md.json"),)


def test_git_diff_resolver_uses_repo_root(monkeypatch):
    captured = {}

    def fake_run(args, capture_output, text, check, cwd):
        captured.update(
            {
                "args": args,
                "capture_output": capture_output,
                "text": text,
                "check": check,
                "cwd": cwd,
            }
        )
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(update.subprocess, "run", fake_run)

    resolver = update.GitDiffResolver()
    resolver.resolve("feature")

    assert captured["args"] == [
        "git",
        "diff",
        "--name-status",
        "--find-renames",
        "feature...HEAD",
    ]
    assert captured["capture_output"] is True
    assert captured["text"] is True
    assert captured["check"] is True
    assert captured["cwd"] == update._REPO_ROOT


def test_derive_targets_from_since_accepts_absolute_paths():
    repo_root = Path(__file__).resolve().parents[1]
    absolute_readme = (repo_root / "README.md").resolve()

    derived = update._derive_targets_from_since((absolute_readme,), repo_root=repo_root)

    assert derived == (Path("docs/birdseye/caps/README.md.json"),)


def test_derive_targets_from_since_handles_rename_notation():
    derived = update._derive_targets_from_since(("README.md -> RUNBOOK.md",))

    assert derived == (
        Path("docs/birdseye/caps/README.md.json"),
        Path("docs/birdseye/caps/RUNBOOK.md.json"),
    )


def test_derive_targets_from_since_deduplicates_old_and_new_paths():
    derived = update._derive_targets_from_since(("README.md -> README.md",))

    assert derived == (Path("docs/birdseye/caps/README.md.json"),)


def _write_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _caps_payload(cap_id: str, *, deps_out=None, deps_in=None) -> dict[str, object]:
    return {
        "id": cap_id,
        "role": "doc",
        "public_api": [],
        "summary": cap_id,
        "deps_out": list(deps_out or []),
        "deps_in": list(deps_in or []),
        "risks": [],
        "tests": [],
    }


def _next_serial(serial: str) -> str:
    match = re.fullmatch(r"\d{5}", serial)
    assert match, f"serial must be 5 digits, got {serial!r}"
    return f"{int(serial) + 1:05d}"


def test_next_generated_at_handles_non_serial_inputs():
    allocator = update._SerialAllocator()

    generated = update._next_generated_at(None, "2025-01-01T00:00:00Z", allocator=allocator)
    assert generated == "00001"

    follow_up = update._next_generated_at(
        "2024-12-31T23:59:59Z",
        "2025-01-01T00:00:00Z",
        allocator=allocator,
    )
    assert follow_up == "00001"

    seeded_allocator = update._SerialAllocator()
    seeded_allocator.observe("00041")

    advanced = update._next_generated_at(
        "unexpected-value",
        "2025-01-01T00:00:00Z",
        allocator=seeded_allocator,
    )
    assert advanced == "00042"


_HOT_INDEX_SNAPSHOT = "docs/birdseye/index.json"
_HOT_REFRESH_COMMAND = (
    "python tools/codemap/update.py --targets docs/birdseye/index.json,docs/birdseye/hot.json --emit index+caps"
)
_HOT_CURATION_NOTES = "Birdseye ホットノードのサンプルノート"


_HOT_HOTLIST_METADATA = {
    "index_snapshot": _HOT_INDEX_SNAPSHOT,
    "refresh_command": _HOT_REFRESH_COMMAND,
    "curation_notes": _HOT_CURATION_NOTES,
}


_HOT_NODES_FIXTURE: Sequence[dict[str, object]] = (
    {
        "id": "README.md",
        "role": "bootstrap",
        "reason": "Birdseye エントリーポイントのサンプル",
        "caps": "docs/birdseye/caps/README.md.json",
        "edges": ["GUARDRAILS.md", "docs/birdseye/index.json"],
        "last_verified_at": "2024-01-01T00:00:00Z",
        "refresh_command": _HOT_REFRESH_COMMAND,
        "index_snapshot": _HOT_INDEX_SNAPSHOT,
        "curation_notes": _HOT_CURATION_NOTES,
    },
    {
        "id": "GUARDRAILS.md",
        "role": "policy",
        "reason": "運用ガイドラインのサンプル",
        "caps": "docs/birdseye/caps/GUARDRAILS.md.json",
        "edges": ["README.md", "docs/birdseye/index.json"],
        "last_verified_at": "2024-01-01T00:00:00Z",
        "refresh_command": _HOT_REFRESH_COMMAND,
        "index_snapshot": _HOT_INDEX_SNAPSHOT,
        "curation_notes": _HOT_CURATION_NOTES,
    },
    {
        "id": "docs/birdseye/index.json",
        "role": "index",
        "reason": "Birdseye インデックスのホットビュー",
        "caps": "docs/birdseye/caps/docs.birdseye.index.json.json",
        "edges": ["README.md", "GUARDRAILS.md"],
        "last_verified_at": "2024-01-01T00:00:00Z",
        "refresh_command": _HOT_REFRESH_COMMAND,
        "index_snapshot": _HOT_INDEX_SNAPSHOT,
        "curation_notes": _HOT_CURATION_NOTES,
    },
)


_HOT_NODE_IDS: tuple[str, ...] = tuple(node["id"] for node in _HOT_NODES_FIXTURE)


def test_hot_refresh_command_matches_documentation():
    repo_root = Path(__file__).resolve().parents[1]
    hot_doc = json.loads((repo_root / "docs" / "birdseye" / "hot.json").read_text(encoding="utf-8"))

    assert hot_doc["refresh_command"] == _HOT_REFRESH_COMMAND
    for node in hot_doc.get("nodes", []):
        if "refresh_command" in node:
            assert node["refresh_command"] == _HOT_REFRESH_COMMAND


def test_hot_nodes_caps_refer_to_existing_capsules():
    repo_root = Path(__file__).resolve().parents[1]
    hot_doc = json.loads((repo_root / "docs" / "birdseye" / "hot.json").read_text(encoding="utf-8"))

    birdseye_index_node = next(
        (node for node in hot_doc.get("nodes", []) if node.get("id") == "docs/birdseye/index.json"),
        None,
    )
    assert birdseye_index_node is not None
    assert (
        birdseye_index_node.get("caps") == "docs/birdseye/caps/docs.birdseye.index.json.json"
    )

    for node in hot_doc.get("nodes", []):
        caps_ref = node.get("caps")
        if caps_ref is None:
            continue

        assert isinstance(caps_ref, str), f"caps must be str or null, got {type(caps_ref)!r}"

        caps_path = repo_root / caps_ref
        assert caps_path.is_file(), f"Capsule missing for node {node['id']}: {caps_ref}"

        capsule_payload = json.loads(caps_path.read_text(encoding="utf-8"))
        assert (
            capsule_payload.get("id") == node["id"]
        ), f"Capsule id mismatch for {node['id']}: {capsule_payload.get('id')}"


def test_index_contains_docs_birdseye_node_with_bidirectional_edges():
    repo_root = Path(__file__).resolve().parents[1]
    index_doc = json.loads((repo_root / "docs" / "birdseye" / "index.json").read_text(encoding="utf-8"))

    nodes = index_doc.get("nodes", {})
    assert "docs/BIRDSEYE.md" in nodes

    capsule_path = nodes["docs/BIRDSEYE.md"].get("caps")
    assert capsule_path == "docs/birdseye/caps/docs.BIRDSEYE.md.json"

    edges = {
        tuple(edge)
        for edge in index_doc.get("edges", [])
        if isinstance(edge, list) and len(edge) == 2
    }

    neighbours = ("README.md", "GUARDRAILS.md", "tools/codemap/README.md")
    for neighbour in neighbours:
        assert ("docs/BIRDSEYE.md", neighbour) in edges
        assert (neighbour, "docs/BIRDSEYE.md") in edges


def test_index_mtime_is_serial_numbered():
    repo_root = Path(__file__).resolve().parents[1]
    index_doc = json.loads((repo_root / "docs" / "birdseye" / "index.json").read_text(encoding="utf-8"))

    mtimes = [node.get("mtime") for node in index_doc.get("nodes", {}).values()]

    assert mtimes, "mtime entries should exist"
    assert all(isinstance(mtime, str) and re.fullmatch(r"\d{5}", mtime) for mtime in mtimes)

    serials = sorted(int(mtime) for mtime in mtimes)
    assert serials == list(range(1, len(mtimes) + 1))


def _prepare_birdseye(
    tmp_path: Path,
    *,
    edges: Iterable[Iterable[str]],
    caps_payloads: Mapping[str, Mapping[str, object]],
    hot_entries: Sequence[str] | None,
    root: Path | None = None,
) -> tuple[Path, Path, Path, dict[str, Path]]:
    root = Path(root) if root is not None else tmp_path / "birdseye"
    caps_dir = root / "caps"
    caps_dir.mkdir(parents=True)
    nodes = {
        cap_id: {"role": payload.get("role", "doc"), "caps": f"docs/birdseye/caps/{cap_id}.json"}
        for cap_id, payload in caps_payloads.items()
    }
    edge_pairs = [list(pair) for pair in edges]
    index_path = root / "index.json"
    _write_json(index_path, {"generated_at": "00001", "nodes": nodes, "edges": edge_pairs})
    cap_paths = {}
    for cap_id, payload in caps_payloads.items():
        cap_path = caps_dir / f"{cap_id}.json"
        cap_paths[cap_id] = cap_path
        _write_json(cap_path, payload)
    hot_path = root / "hot.json"
    defaults_lookup = {entry["id"]: entry for entry in _HOT_NODES_FIXTURE}
    selected_ids = list(hot_entries) if hot_entries is not None else list(_HOT_NODE_IDS)
    neighbours: dict[str, set[str]] = {}
    for pair in edge_pairs:
        if len(pair) != 2:
            continue
        source, destination = pair
        if not isinstance(source, str) or not isinstance(destination, str):
            continue
        neighbours.setdefault(source, set()).add(destination)
        neighbours.setdefault(destination, set()).add(source)
    serialized_nodes = []
    for entry_id in selected_ids:
        defaults = defaults_lookup.get(entry_id, {})
        index_node = nodes.get(entry_id, {})
        resolved_edges = (
            list(defaults["edges"]) if "edges" in defaults else sorted(neighbours.get(entry_id, set()))
        )
        if "caps" in defaults:
            resolved_caps = defaults["caps"]
        else:
            resolved_caps = index_node.get("caps") or f"docs/birdseye/caps/{entry_id}.json"
        payload = {
            "id": entry_id,
            "role": defaults.get("role", index_node.get("role", "doc")),
            "edges": resolved_edges,
            "caps": resolved_caps,
            "last_verified_at": defaults.get("last_verified_at", "2024-01-01T00:00:00Z"),
        }
        if "reason" in defaults:
            payload["reason"] = defaults["reason"]
        if "refresh_command" in defaults:
            payload["refresh_command"] = defaults["refresh_command"]
        if "index_snapshot" in defaults:
            payload["index_snapshot"] = defaults["index_snapshot"]
        else:
            payload["index_snapshot"] = _HOT_INDEX_SNAPSHOT
        if "curation_notes" in defaults:
            payload["curation_notes"] = defaults["curation_notes"]
        else:
            payload["curation_notes"] = _HOT_CURATION_NOTES
        serialized_nodes.append(payload)
    _write_json(
        hot_path,
        {
            "generated_at": "00001",
            **_HOT_HOTLIST_METADATA,
            "nodes": serialized_nodes,
        },
    )
    return root, index_path, hot_path, cap_paths


def test_run_update_with_since_updates_capsules_within_two_hops(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(update, "_REPO_ROOT", tmp_path)

    caps_payloads = {
        "alpha.md": _caps_payload("alpha.md", deps_out=["stale"], deps_in=["stale"]),
        "beta.md": _caps_payload("beta.md", deps_out=["stale"], deps_in=["stale"]),
        "gamma.md": _caps_payload("gamma.md", deps_out=["stale"], deps_in=["stale"]),
        "delta.md": _caps_payload("delta.md", deps_out=["stale"], deps_in=["stale"]),
    }
    _, _, _, cap_paths = _prepare_birdseye(
        tmp_path,
        edges=[
            ["alpha.md", "beta.md"],
            ["beta.md", "alpha.md"],
            ["beta.md", "gamma.md"],
            ["gamma.md", "beta.md"],
            ["gamma.md", "delta.md"],
            ["delta.md", "gamma.md"],
        ],
        caps_payloads=caps_payloads,
        hot_entries=_HOT_NODE_IDS,
        root=tmp_path / "docs" / "birdseye",
    )

    options = update.UpdateOptions(
        targets=(),
        emit="caps",
        dry_run=False,
        since="main",
        diff_resolver=SimpleNamespace(
            resolve=lambda _: (Path("docs/birdseye/caps/alpha.md.json"),)
        ),
    )

    report = update.run_update(options)

    expected_caps = {cap_paths[name] for name in ("alpha.md", "beta.md", "gamma.md")}

    assert set(report.planned_writes) == expected_caps
    assert set(report.performed_writes) == expected_caps

    delta_payload = json.loads(cap_paths["delta.md"].read_text(encoding="utf-8"))
    assert delta_payload["deps_out"] == ["stale"]
    assert delta_payload["deps_in"] == ["stale"]


def test_run_update_with_since_handles_git_rename(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(update, "_REPO_ROOT", tmp_path)

    caps_payloads = {
        "README.md": _caps_payload("README.md"),
        "RUNBOOK.md": _caps_payload("RUNBOOK.md"),
    }
    _, _, _, cap_paths = _prepare_birdseye(
        tmp_path,
        edges=[["README.md", "RUNBOOK.md"], ["RUNBOOK.md", "README.md"]],
        caps_payloads=caps_payloads,
        hot_entries=("README.md", "RUNBOOK.md"),
        root=tmp_path / "docs" / "birdseye",
    )

    def fake_run(args, capture_output, text, check, cwd):
        assert args == [
            "git",
            "diff",
            "--name-status",
            "--find-renames",
            "main...HEAD",
        ]
        assert cwd == tmp_path
        return SimpleNamespace(stdout="R100\tREADME.md\tRUNBOOK.md\n")

    monkeypatch.setattr(update.subprocess, "run", fake_run)

    options = update.UpdateOptions(
        targets=(),
        emit="caps",
        dry_run=False,
        since="main",
        diff_resolver=update.GitDiffResolver(),
    )

    report = update.run_update(options)

    expected_caps = {cap_paths[name] for name in ("README.md", "RUNBOOK.md")}

    assert set(report.planned_writes) == expected_caps
    assert set(report.performed_writes) == expected_caps


def test_run_update_resolves_targets_from_repo_root_outside_cwd(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()

    caps_payloads = {"README.md": _caps_payload("README.md")}
    _root, index_path, hot_path, _ = _prepare_birdseye(
        repo_root,
        edges=[],
        caps_payloads=caps_payloads,
        hot_entries=("README.md",),
        root=repo_root / "docs" / "birdseye",
    )

    monkeypatch.setattr(update, "_REPO_ROOT", repo_root)
    monkeypatch.chdir(outside_dir)

    options = update.parse_args(["--targets", "docs/birdseye/index.json", "--emit", "index"])

    assert options.targets == (index_path,)
    resolved = options.resolve_targets()
    assert resolved == (index_path,)

    report = update.run_update(options)

    expected_writes = {index_path, hot_path}
    assert set(report.planned_writes) == expected_writes
    assert set(report.performed_writes) == expected_writes

    refreshed_index = json.loads(index_path.read_text(encoding="utf-8"))
    refreshed_hot = json.loads(hot_path.read_text(encoding="utf-8"))

    assert refreshed_index["generated_at"] == "00002"
    assert refreshed_hot["generated_at"] == "00002"


@pytest.mark.parametrize("dry_run", [False, True])
def test_run_update_refreshes_metadata_and_dependencies(tmp_path, monkeypatch, dry_run):
    caps_payloads = {
        "alpha.md": _caps_payload("alpha.md", deps_in=["obsolete"]),
        "beta.md": _caps_payload("beta.md", deps_out=["stale"]),
    }
    root, index_path, hot_path, cap_paths = _prepare_birdseye(
        tmp_path,
        edges=[["alpha.md", "beta.md"], ["beta.md", "alpha.md"]],
        caps_payloads=caps_payloads,
        hot_entries=_HOT_NODE_IDS,
    )

    baseline_index = json.loads(index_path.read_text(encoding="utf-8"))
    base_serial = baseline_index["generated_at"]
    expected_serial = _next_serial(base_serial)

    frozen_now = datetime(2025, 1, 1, 9, 30, tzinfo=timezone.utc)
    monkeypatch.setattr(update, "utc_now", lambda: frozen_now)

    snapshots = None
    if dry_run:
        snapshots = {
            path: path.read_text(encoding="utf-8")
            for path in (index_path, hot_path, *cap_paths.values())
        }

    report = update.run_update(
        update.UpdateOptions(targets=(root,), emit="index+caps", dry_run=dry_run)
    )

    assert re.fullmatch(r"\d{5}", report.generated_at)
    assert report.generated_at == expected_serial
    assert set(report.planned_writes) == {index_path, hot_path, *cap_paths.values()}

    if dry_run:
        assert report.performed_writes == ()
        assert snapshots is not None
        for path, before in snapshots.items():
            assert path.read_text(encoding="utf-8") == before
        return

    assert set(report.performed_writes) == {index_path, hot_path, *cap_paths.values()}

    refreshed_index = json.loads(index_path.read_text(encoding="utf-8"))
    assert refreshed_index["generated_at"] == expected_serial

    refreshed_alpha = json.loads(cap_paths["alpha.md"].read_text(encoding="utf-8"))
    assert refreshed_alpha["deps_out"] == ["beta.md"]
    assert refreshed_alpha["deps_in"] == ["beta.md"]

    refreshed_beta = json.loads(cap_paths["beta.md"].read_text(encoding="utf-8"))
    assert refreshed_beta["deps_out"] == ["alpha.md"]
    assert refreshed_beta["deps_in"] == ["alpha.md"]

    refreshed_hot = json.loads(hot_path.read_text(encoding="utf-8"))
    assert refreshed_hot["generated_at"] == expected_serial
    expected_hot_nodes = [dict(node) for node in _HOT_NODES_FIXTURE]
    assert refreshed_hot["index_snapshot"] == _HOT_INDEX_SNAPSHOT
    assert refreshed_hot["refresh_command"] == _HOT_REFRESH_COMMAND
    assert refreshed_hot["curation_notes"] == _HOT_CURATION_NOTES
    assert refreshed_hot["nodes"] == expected_hot_nodes
    for node in refreshed_hot["nodes"]:
        assert node["refresh_command"] == _HOT_REFRESH_COMMAND
        assert node["index_snapshot"] == _HOT_INDEX_SNAPSHOT
        assert node["curation_notes"] == _HOT_CURATION_NOTES
    assert len(refreshed_hot["nodes"]) == len(expected_hot_nodes)
    assert refreshed_hot["nodes"][0]["edges"] == expected_hot_nodes[0]["edges"]
    assert refreshed_hot["nodes"][0]["caps"] == expected_hot_nodes[0]["caps"]
    assert refreshed_hot["nodes"][0]["role"] == expected_hot_nodes[0]["role"]
    assert (
        refreshed_hot["nodes"][0]["last_verified_at"]
        == expected_hot_nodes[0]["last_verified_at"]
    )
    hot_caps_lookup = {node["id"]: node["caps"] for node in refreshed_hot["nodes"]}
    assert (
        hot_caps_lookup["docs/birdseye/index.json"]
        == "docs/birdseye/caps/docs.birdseye.index.json.json"
    )


def test_run_update_requires_hot_json_when_emitting_index(tmp_path, monkeypatch):
    caps_payloads = {
        "alpha.md": _caps_payload("alpha.md"),
        "beta.md": _caps_payload("beta.md"),
    }
    root, _, hot_path, _ = _prepare_birdseye(
        tmp_path,
        edges=[["alpha.md", "beta.md"]],
        caps_payloads=caps_payloads,
        hot_entries=_HOT_NODE_IDS,
    )

    hot_path.unlink()

    monkeypatch.setattr(update, "_REPO_ROOT", tmp_path)

    with pytest.raises(FileNotFoundError) as excinfo:
        update.run_update(update.UpdateOptions(targets=(root,), emit="index+caps"))

    message = str(excinfo.value)
    assert str(hot_path) in message
    assert _HOT_REFRESH_COMMAND in message


def test_run_update_preserves_hot_nodes_structure(tmp_path, monkeypatch):
    caps_payloads = {
        "alpha.md": _caps_payload("alpha.md", deps_in=["obsolete"]),
        "beta.md": _caps_payload("beta.md", deps_out=["stale"]),
    }
    root, _, hot_path, _ = _prepare_birdseye(
        tmp_path,
        edges=[["alpha.md", "beta.md"], ["beta.md", "alpha.md"]],
        caps_payloads=caps_payloads,
        hot_entries=_HOT_NODE_IDS,
    )

    baseline_hot = json.loads(hot_path.read_text(encoding="utf-8"))
    base_serial = baseline_hot["generated_at"]
    expected_serial = _next_serial(base_serial)
    assert baseline_hot["index_snapshot"] == _HOT_INDEX_SNAPSHOT
    assert baseline_hot["refresh_command"] == _HOT_REFRESH_COMMAND
    assert baseline_hot["curation_notes"] == _HOT_CURATION_NOTES
    assert (
        {
            node["id"]: node["caps"]
            for node in baseline_hot["nodes"]
            if "caps" in node and "id" in node
        }["docs/birdseye/index.json"]
        == "docs/birdseye/caps/docs.birdseye.index.json.json"
    )
    for node in baseline_hot["nodes"]:
        assert node["refresh_command"] == _HOT_REFRESH_COMMAND
        assert node["index_snapshot"] == _HOT_INDEX_SNAPSHOT
        assert node["curation_notes"] == _HOT_CURATION_NOTES

    frozen_now = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(update, "utc_now", lambda: frozen_now)

    report = update.run_update(
        update.UpdateOptions(targets=(root,), emit="index+caps", dry_run=False)
    )

    assert re.fullmatch(r"\d{5}", report.generated_at)
    assert report.generated_at == expected_serial

    refreshed_hot = json.loads(hot_path.read_text(encoding="utf-8"))
    assert refreshed_hot["generated_at"] == expected_serial
    assert refreshed_hot["index_snapshot"] == _HOT_INDEX_SNAPSHOT
    assert refreshed_hot["refresh_command"] == _HOT_REFRESH_COMMAND
    assert refreshed_hot["curation_notes"] == _HOT_CURATION_NOTES
    assert refreshed_hot["nodes"] == baseline_hot["nodes"]
    assert (
        {
            node["id"]: node["caps"]
            for node in refreshed_hot["nodes"]
            if "caps" in node and "id" in node
        }["docs/birdseye/index.json"]
        == "docs/birdseye/caps/docs.birdseye.index.json.json"
    )
    for node in refreshed_hot["nodes"]:
        assert node["refresh_command"] == _HOT_REFRESH_COMMAND
        assert node["index_snapshot"] == _HOT_INDEX_SNAPSHOT
        assert node["curation_notes"] == _HOT_CURATION_NOTES
    assert len(refreshed_hot["nodes"]) == len(_HOT_NODES_FIXTURE)
    assert all(
        node["last_verified_at"] == "2024-01-01T00:00:00Z"
        for node in refreshed_hot["nodes"]
    )
    assert refreshed_hot["nodes"][0]["edges"] == baseline_hot["nodes"][0]["edges"]
    assert refreshed_hot["nodes"][0]["caps"] == baseline_hot["nodes"][0]["caps"]
    assert refreshed_hot["nodes"][0]["role"] == baseline_hot["nodes"][0]["role"]


def test_run_update_limits_caps_to_two_hop_scope(tmp_path, monkeypatch):
    caps_payloads = {
        cap_id: _caps_payload(cap_id, deps_out=["stale"], deps_in=["old"])
        for cap_id in ("alpha.md", "beta.md", "gamma.md", "delta.md", "epsilon.md")
    }
    root, index_path, hot_path, cap_paths = _prepare_birdseye(
        tmp_path,
        edges=[
            ["alpha.md", "beta.md"],
            ["beta.md", "gamma.md"],
            ["gamma.md", "delta.md"],
            ["delta.md", "epsilon.md"],
        ],
        caps_payloads=caps_payloads,
        hot_entries=[],
    )

    baseline_hot = json.loads(hot_path.read_text(encoding="utf-8"))
    assert baseline_hot["nodes"] == []

    frozen_now = datetime(2025, 1, 2, tzinfo=timezone.utc)
    monkeypatch.setattr(update, "utc_now", lambda: frozen_now)

    report = update.run_update(
        update.UpdateOptions(targets=(cap_paths["beta.md"],), emit="caps", dry_run=False)
    )

    expected_caps = {
        cap_paths[cap_id]
        for cap_id in ("alpha.md", "beta.md", "gamma.md", "delta.md")
    }
    assert set(report.planned_writes) == expected_caps
    assert set(report.performed_writes) == expected_caps

    expected_deps = {
        "alpha.md": (["beta.md"], []),
        "beta.md": (["gamma.md"], ["alpha.md"]),
        "gamma.md": (["delta.md"], ["beta.md"]),
        "delta.md": (["epsilon.md"], ["gamma.md"]),
    }
    for cap_id, (deps_out, deps_in) in expected_deps.items():
        refreshed = json.loads(cap_paths[cap_id].read_text(encoding="utf-8"))
        assert refreshed["deps_out"] == deps_out
        assert refreshed["deps_in"] == deps_in


def test_run_update_refreshes_caps_generated_at(tmp_path, monkeypatch):
    edges = [
        ["alpha.md", "beta.md"],
        ["beta.md", "gamma.md"],
        ["gamma.md", "delta.md"],
        ["delta.md", "epsilon.md"],
    ]
    expected_deps = {
        "alpha.md": (["beta.md"], []),
        "beta.md": (["gamma.md"], ["alpha.md"]),
        "gamma.md": (["delta.md"], ["beta.md"]),
        "delta.md": (["epsilon.md"], ["gamma.md"]),
        "epsilon.md": ([], ["delta.md"]),
    }
    base_serial = "00009"
    caps_payloads = {}
    for cap_id, (deps_out, deps_in) in expected_deps.items():
        payload = _caps_payload(cap_id, deps_out=deps_out, deps_in=deps_in)
        payload["generated_at"] = base_serial
        caps_payloads[cap_id] = payload

    root, _, hot_path, cap_paths = _prepare_birdseye(
        tmp_path,
        edges=edges,
        caps_payloads=caps_payloads,
        hot_entries=[],
    )

    assert json.loads(hot_path.read_text(encoding="utf-8"))["nodes"] == []

    frozen_now = datetime(2025, 1, 5, tzinfo=timezone.utc)
    monkeypatch.setattr(update, "utc_now", lambda: frozen_now)

    expected_serial = _next_serial(base_serial)
    report = update.run_update(
        update.UpdateOptions(targets=(cap_paths["beta.md"],), emit="caps", dry_run=False)
    )

    expected_caps = {
        cap_paths[cap_id]
        for cap_id in ("alpha.md", "beta.md", "gamma.md", "delta.md")
    }

    assert re.fullmatch(r"\d{5}", report.generated_at)
    assert report.generated_at == expected_serial
    assert set(report.planned_writes) == expected_caps
    assert set(report.performed_writes) == expected_caps

    for cap_id in ("alpha.md", "beta.md", "gamma.md", "delta.md"):
        refreshed = json.loads(cap_paths[cap_id].read_text(encoding="utf-8"))
        assert re.fullmatch(r"\d{5}", refreshed["generated_at"])
        assert refreshed["generated_at"] == expected_serial
        expected_out, expected_in = expected_deps[cap_id]
        assert refreshed["deps_out"] == expected_out
        assert refreshed["deps_in"] == expected_in

    untouched = json.loads(cap_paths["epsilon.md"].read_text(encoding="utf-8"))
    assert untouched["generated_at"] == base_serial
    expected_out, expected_in = expected_deps["epsilon.md"]
    assert untouched["deps_out"] == expected_out
    assert untouched["deps_in"] == expected_in


def test_run_update_recovers_non_serial_generated_at(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(update, "_REPO_ROOT", tmp_path)

    caps_payloads = {
        "alpha.md": {**_caps_payload("alpha.md"), "generated_at": "00007"},
        "beta.md": {**_caps_payload("beta.md"), "generated_at": "2024-01-01T00:00:00Z"},
    }

    _root, index_path, hot_path, cap_paths = _prepare_birdseye(
        tmp_path,
        edges=[["alpha.md", "beta.md"], ["beta.md", "alpha.md"]],
        caps_payloads=caps_payloads,
        hot_entries=("alpha.md", "beta.md"),
        root=tmp_path / "docs" / "birdseye",
    )

    index_payload = json.loads(index_path.read_text(encoding="utf-8"))
    index_payload["generated_at"] = None
    _write_json(index_path, index_payload)

    hot_payload = json.loads(hot_path.read_text(encoding="utf-8"))
    hot_payload["generated_at"] = "2023-12-31T23:59:59Z"
    _write_json(hot_path, hot_payload)

    report = update.run_update(
        update.UpdateOptions(targets=(cap_paths["beta.md"],), emit="index+caps", dry_run=False)
    )

    expected_serial = "00008"

    assert report.generated_at == expected_serial

    refreshed_index = json.loads(index_path.read_text(encoding="utf-8"))
    assert refreshed_index["generated_at"] == expected_serial

    refreshed_hot = json.loads(hot_path.read_text(encoding="utf-8"))
    assert refreshed_hot["generated_at"] == expected_serial

    refreshed_beta = json.loads(cap_paths["beta.md"].read_text(encoding="utf-8"))
    assert refreshed_beta["generated_at"] == expected_serial

    updated_alpha = json.loads(cap_paths["alpha.md"].read_text(encoding="utf-8"))
    assert updated_alpha["generated_at"] == expected_serial


def test_run_update_accepts_caps_directory_target(tmp_path, monkeypatch):
    caps_payloads = {
        cap_id: _caps_payload(cap_id)
        for cap_id in ("alpha.md", "beta.md", "gamma.md", "delta.md", "epsilon.md")
    }
    root_base = tmp_path / "docs" / "birdseye"
    root, index_path, hot_path, cap_paths = _prepare_birdseye(
        tmp_path,
        edges=[
            ["alpha.md", "beta.md"],
            ["beta.md", "gamma.md"],
            ["gamma.md", "delta.md"],
            ["delta.md", "epsilon.md"],
        ],
        caps_payloads=caps_payloads,
        hot_entries=_HOT_NODE_IDS,
        root=root_base,
    )

    monkeypatch.setattr(update, "_REPO_ROOT", tmp_path)
    baseline_index = json.loads(index_path.read_text(encoding="utf-8"))
    base_serial = baseline_index["generated_at"]
    expected_serial = _next_serial(base_serial)

    monkeypatch.chdir(tmp_path)

    frozen_now = datetime(2025, 1, 4, tzinfo=timezone.utc)
    monkeypatch.setattr(update, "utc_now", lambda: frozen_now)

    report = update.run_update(
        update.UpdateOptions(targets=(Path("docs/birdseye/caps"),), emit="index+caps", dry_run=False)
    )

    expected_caps = {cap_paths[cap_id] for cap_id in cap_paths}
    expected_writes = expected_caps | {index_path, hot_path}

    assert re.fullmatch(r"\d{5}", report.generated_at)
    assert report.generated_at == expected_serial
    assert set(report.planned_writes) == expected_writes
    assert set(report.performed_writes) == expected_writes

    refreshed_index = json.loads(index_path.read_text(encoding="utf-8"))
    assert refreshed_index["generated_at"] == expected_serial

    refreshed_hot = json.loads(hot_path.read_text(encoding="utf-8"))
    assert refreshed_hot["generated_at"] == expected_serial
    assert len(refreshed_hot["nodes"]) == len(_HOT_NODE_IDS)

    expected_deps = {
        "alpha.md": (["beta.md"], []),
        "beta.md": (["gamma.md"], ["alpha.md"]),
        "gamma.md": (["delta.md"], ["beta.md"]),
        "delta.md": (["epsilon.md"], ["gamma.md"]),
        "epsilon.md": ([], ["delta.md"]),
    }
    for cap_id, (deps_out, deps_in) in expected_deps.items():
        refreshed = json.loads(cap_paths[cap_id].read_text(encoding="utf-8"))
        assert refreshed["deps_out"] == deps_out
        assert refreshed["deps_in"] == deps_in


def test_parse_args_returns_since_without_resolving(monkeypatch):
    monkeypatch.setattr(
        update.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("should not be called")),
    )

    options = update.parse_args(["--since", "feature"])

    assert options.targets == ()
    assert options.since == "feature"


def test_parse_args_supports_since_and_limits_scope(tmp_path, monkeypatch):
    caps_payloads = {
        cap_id: _caps_payload(cap_id, deps_out=["stale"], deps_in=["old"])
        for cap_id in (
            "alpha.md",
            "beta.md",
            "gamma.md",
            "delta.md",
            "epsilon.md",
            "zeta.md",
        )
    }
    root_base = tmp_path / "docs" / "birdseye"
    root, _, _, cap_paths = _prepare_birdseye(
        tmp_path,
        edges=[
            ["alpha.md", "beta.md"],
            ["beta.md", "gamma.md"],
            ["gamma.md", "delta.md"],
            ["delta.md", "epsilon.md"],
            ["epsilon.md", "zeta.md"],
        ],
        caps_payloads=caps_payloads,
        hot_entries=_HOT_NODE_IDS,
        root=root_base,
    )

    frozen_now = datetime(2025, 1, 3, tzinfo=timezone.utc)
    monkeypatch.setattr(update, "utc_now", lambda: frozen_now)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(update, "_REPO_ROOT", tmp_path)

    class FakeResolver:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def resolve(self, reference: str) -> tuple[Path, ...]:
            self.calls.append(reference)
            return (
                Path("docs/birdseye/caps/delta.md.json"),
            )

    options = update.parse_args(["--emit", "caps", "--since"])

    resolver = FakeResolver()
    options = replace(options, diff_resolver=resolver)

    assert resolver.calls == []

    report = update.run_update(options)

    assert resolver.calls == ["main"]

    expected_caps = {
        cap_paths[cap_id]
        for cap_id in ("beta.md", "gamma.md", "delta.md", "epsilon.md", "zeta.md")
    }
    assert set(report.planned_writes) == expected_caps
    assert set(report.performed_writes) == expected_caps

    untouched = json.loads(cap_paths["alpha.md"].read_text(encoding="utf-8"))
    assert untouched["deps_out"] == ["stale"]
    assert untouched["deps_in"] == ["old"]


def test_git_diff_resolver_filters_paths(monkeypatch):
    captured = {}

    def fake_run(args, *, capture_output, text, check, cwd):
        captured.update(
            {
                "args": args,
                "cwd": cwd,
            }
        )
        return SimpleNamespace(
            stdout=(
                "M\tdocs/birdseye/caps/delta.md.json\n"
                "M\tREADME.md\n"
                "M\tdocs/birdseye/index.json\n"
            )
        )

    monkeypatch.setattr(update.subprocess, "run", fake_run)

    resolver = update.GitDiffResolver()

    result = resolver.resolve("develop")

    assert captured["args"] == [
        "git",
        "diff",
        "--name-status",
        "--find-renames",
        "develop...HEAD",
    ]
    assert captured["cwd"] == update._REPO_ROOT
    assert result == (
        Path("docs/birdseye/caps/delta.md.json"),
        Path("docs/birdseye/caps/README.md.json"),
        Path("docs/birdseye/index.json"),
    )


def test_group_targets_resolves_caps_paths(tmp_path):
    root = tmp_path / "docs" / "birdseye"
    caps_dir = root / "caps"
    caps_dir.mkdir(parents=True)
    index_path = root / "index.json"
    hot_path = root / "hot.json"
    cap_file = caps_dir / "alpha.json"
    for path in (index_path, hot_path, cap_file):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")

    grouped = update._group_targets((root, index_path, caps_dir, cap_file))

    assert set(grouped) == {root}
    assert grouped[root] == [root, index_path, caps_dir, cap_file]


def test_resolve_focus_nodes_includes_two_hop_neighbours(tmp_path):
    root = tmp_path / "docs" / "birdseye"
    caps_dir = root / "caps"
    caps_dir.mkdir(parents=True)
    cap_alpha = caps_dir / "alpha.json"
    cap_beta = caps_dir / "beta.json"
    cap_gamma = caps_dir / "gamma.json"
    cap_delta = caps_dir / "delta.json"
    cap_paths = {
        "alpha": cap_alpha,
        "beta": cap_beta,
        "gamma": cap_gamma,
        "delta": cap_delta,
    }
    for path in cap_paths.values():
        path.write_text("{}", encoding="utf-8")

    caps_state = {
        cap_id: (cap_path, {"id": cap_id}, "{}")
        for cap_id, cap_path in cap_paths.items()
    }
    cap_lookup = {cap_path.resolve(): cap_id for cap_id, cap_path in cap_paths.items()}
    graph_out = {"alpha": ["beta"], "beta": ["gamma"], "gamma": ["delta"], "delta": []}
    graph_in = {"beta": ["alpha"], "gamma": ["beta"], "delta": ["gamma"], "alpha": []}

    focus = update._resolve_focus_nodes(
        (cap_beta,),
        root,
        graph_out,
        graph_in,
        caps_state,
        cap_lookup,
    )

    assert focus == {"alpha", "beta", "gamma", "delta"}


def test_refresh_hot_updates_serial_and_preserves_metadata(tmp_path):
    hot_path = tmp_path / "hot.json"
    _write_json(
        hot_path,
        {
            "generated_at": "00042",
            **_HOT_HOTLIST_METADATA,
            "nodes": list(_HOT_NODES_FIXTURE),
        },
    )

    planned: set[Path] = set()
    performed: set[Path] = set()
    timestamp = "2025-01-02T00:00:00Z"
    hot_original = hot_path.read_text(encoding="utf-8")
    hot_payload = json.loads(hot_original)
    allocator = update._SerialAllocator()
    allocator.observe("00042")
    update._refresh_hot(
        hot_path,
        hot_payload,
        hot_original,
        timestamp=timestamp,
        dry_run=False,
        planned=planned,
        performed=performed,
        remember_generated=lambda _: None,
        allocator=allocator,
    )

    assert planned == {hot_path}
    assert performed == {hot_path}
    refreshed = json.loads(hot_path.read_text(encoding="utf-8"))
    assert refreshed["generated_at"] == "00043"
    assert refreshed["index_snapshot"] == _HOT_INDEX_SNAPSHOT
    assert refreshed["refresh_command"] == _HOT_REFRESH_COMMAND
    assert refreshed["curation_notes"] == _HOT_CURATION_NOTES
    for node in refreshed["nodes"]:
        assert node["refresh_command"] == _HOT_REFRESH_COMMAND
        assert node["index_snapshot"] == _HOT_INDEX_SNAPSHOT
        assert node["curation_notes"] == _HOT_CURATION_NOTES


def test_refresh_capsule_updates_dependencies_and_serial(tmp_path):
    cap_path = tmp_path / "caps" / "alpha.json"
    cap_path.parent.mkdir(parents=True)
    _write_json(
        cap_path,
        _caps_payload(
            "alpha",
            deps_out=["stale"],
            deps_in=["obsolete"],
        )
        | {"generated_at": "00010"},
    )
    cap_original = cap_path.read_text(encoding="utf-8")
    caps_state = {"alpha": (cap_path, json.loads(cap_original), cap_original)}
    planned: set[Path] = set()
    performed: set[Path] = set()
    allocator = update._SerialAllocator()
    allocator.observe("00010")

    update._refresh_capsule(
        "alpha",
        caps_state["alpha"],
        {"alpha": ["beta"]},
        {"alpha": ["gamma"]},
        "2025-01-02T00:00:00Z",
        dry_run=False,
        planned=planned,
        performed=performed,
        remember_generated=lambda _: None,
        allocator=allocator,
    )

    assert planned == {cap_path}
    assert performed == {cap_path}
    refreshed = json.loads(cap_path.read_text(encoding="utf-8"))
    assert refreshed["deps_out"] == ["beta"]
    assert refreshed["deps_in"] == ["gamma"]
    assert refreshed["generated_at"] == "00011"
