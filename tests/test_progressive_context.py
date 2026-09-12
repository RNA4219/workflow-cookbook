import hashlib
import json
import os
import sys

import pytest

from tools.codemap.source_freshness import summary_digest
from tools.context.progressive import main, retrieve
from tools.workflow_plugins.runtime import PluginPolicy, WorkflowPluginRuntime


@pytest.fixture
def isolated_memx_imports(monkeypatch):
    prefix = "memx_resolver_workflow_plugin"
    original = {name: value for name, value in sys.modules.items() if name == prefix or name.startswith(prefix + ".")}
    monkeypatch.setattr(sys, "path", list(sys.path))
    for name in original:
        del sys.modules[name]
    try:
        yield
    finally:
        for name in list(sys.modules):
            if name == prefix or name.startswith(prefix + "."):
                del sys.modules[name]
        sys.modules.update(original)


def test_real_memx_cache_invalidation_and_run_trace(tmp_path, isolated_memx_imports):
    import time
    from pathlib import Path

    from tools.workflow_plugins.run_report import build_run_report
    from tools.workflow_plugins.runtime import RunContext

    sibling = Path(os.environ.get("WFC_MEMX_ROOT", str(Path(__file__).resolve().parents[2] / "memx-resolver")))
    if not (sibling / "memx_resolver_workflow_plugin").is_dir():
        pytest.skip("optional sibling memx integration")
    tasks = tmp_path / "docs/tasks"
    tasks.mkdir(parents=True)
    (tasks / "task.md").write_text("---\ntask_id: fixture\n---\n# Task\n", encoding="utf-8")
    readme = tmp_path / "README.md"
    readme.write_text("# target old\n", encoding="utf-8")
    config = tmp_path / "plugins.json"
    config.write_text(
        json.dumps(
            {
                "workflow_plugins": [
                    {
                        "factory": "memx_resolver_workflow_plugin.plugin:create_plugin",
                        "python_paths": [str(sibling)],
                        "options": {},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    started = time.time()
    runtime = WorkflowPluginRuntime.from_config(config, run_context=RunContext("fixture", "run"))
    first = retrieve(tmp_path, "target", 10000, runtime=runtime, task_id="fixture")
    cache = tmp_path / ".workflow-cache/memx-doc-resolve.json"
    initial_cache = json.loads(cache.read_text(encoding="utf-8"))
    assert "target old" in first["context"]
    again = retrieve(tmp_path, "target", 10000, runtime=runtime, task_id="fixture")
    assert again["context"] == first["context"]
    assert json.loads(cache.read_text(encoding="utf-8")) == initial_cache
    readme.write_text("# target revised\n", encoding="utf-8")
    changed = retrieve(tmp_path, "target", 10000, runtime=runtime, task_id="fixture")
    assert "target revised" in changed["context"] and "target old" not in changed["context"]
    assert json.loads(cache.read_text(encoding="utf-8")) != initial_cache
    assert not (tmp_path / ".workflow-cache/memx-doc-receipts.json").exists()
    outcome = {
        "schema_version": "1.0",
        "task_id": "fixture",
        "run_id": "run",
        "acceptance_id": "fixture-check",
        "accepted": False,
        "started_at": started,
        "finished_at": time.time(),
        "input_tokens": None,
        "output_tokens": None,
        "cost": None,
    }
    report = build_run_report(runtime.traces, outcome)
    assert report["invocations"] == 3 and report["accepted"] is False
    assert report["capabilities"]["docs.resolve"]["attempts"] == 3


def make_index(root, records):
    nodes = {}
    directory = root / "docs/birdseye"
    directory.mkdir(parents=True)
    for name, summary, dependencies in records:
        raw = (root / name).read_bytes()
        source_hash = hashlib.sha256(raw).hexdigest()
        capsule = {"id": name, "summary": summary, "deps_out": dependencies, "source_sha256": source_hash}
        capsule["review"] = {
            "source_sha256": source_hash,
            "summary_sha256": summary_digest(capsule),
            "reviewed_at": "2026-09-10",
        }
        path = directory / (name.replace("/", "_") + ".json")
        path.write_text(json.dumps(capsule), encoding="utf-8")
        nodes[name] = {"caps": path.relative_to(root).as_posix()}
    (directory / "index.json").write_text(json.dumps({"nodes": nodes}), encoding="utf-8")


@pytest.fixture
def documents(tmp_path):
    for name, content in {"a.md": "target source", "b.md": "dependency one", "c.md": "dependency two"}.items():
        (tmp_path / name).write_text(content, encoding="utf-8")
    make_index(tmp_path, [("a.md", "target", ["b.md"]), ("b.md", "other", ["c.md"]), ("c.md", "other", ["a.md"])])
    return tmp_path


@pytest.mark.parametrize("hops,expected", [(0, ["a.md"]), (1, ["a.md", "b.md"]), (2, ["a.md", "b.md", "c.md"])])
def test_dependency_expansion_is_bounded_and_cycles_do_not_duplicate(documents, hops, expected):
    result = retrieve(documents, "target", 1000, max_hops=hops)
    assert [item["path"] for item in result["selected"]] == expected
    assert len({item["path"] for item in result["selected"]}) == len(expected)
    assert result["context_bytes"] <= 1000
    assert result["token_count"] is None


@pytest.mark.parametrize("condition", ["missing", "invalid", "shape", "stale", "unreviewed"])
def test_unusable_index_falls_back_to_current_source(documents, condition):
    index = documents / "docs/birdseye/index.json"
    if condition == "missing":
        index.unlink()
    elif condition == "invalid":
        index.write_text("{", encoding="utf-8")
    elif condition == "shape":
        index.write_text("[]", encoding="utf-8")
    elif condition == "stale":
        (documents / "a.md").write_text("target changed", encoding="utf-8")
    else:
        path = documents / "docs/birdseye/a.md.json"
        capsule = json.loads(path.read_text(encoding="utf-8"))
        capsule.pop("review")
        path.write_text(json.dumps(capsule), encoding="utf-8")
    result = retrieve(documents, "target", 1000, target_documents=1)
    assert result["selected"][0]["via"] == "source_search"
    assert "target" in result["context"]
    assert result["warnings"]


def test_single_file_scope_ignores_other_matching_documents(documents):
    result = retrieve(documents, "target", 1000, scope_paths=["a.md"], target_documents=1)
    assert [entry["path"] for entry in result["selected"]] == ["a.md"]
    assert "dependency one" not in result["context"]


def test_missing_scope_and_file_as_repo_are_explicit_errors(documents):
    with pytest.raises(ValueError, match="scope does not exist"):
        retrieve(documents, "target", 1000, scope_paths=["absent"])
    with pytest.raises(ValueError, match="repo_root must be a directory"):
        retrieve(documents / "a.md", "target", 1000)


def test_discovery_skips_file_resolving_outside_repo(documents, monkeypatch):
    from pathlib import Path

    outside = documents.parent / (documents.name + "-external.md")
    outside.write_text("target external fixture", encoding="utf-8")
    alias = documents / "external.md"
    alias.write_text("reparse point fixture", encoding="utf-8")
    original_resolve = Path.resolve

    def resolve(path, *args, **kwargs):
        # Windowsのsymlink作成権限を要求せず、OSのreparse point解決結果を模擬する。
        return outside if path == alias else original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolve)
    result = retrieve(documents, "target", 1000, target_documents=10)
    assert [entry["path"] for entry in result["selected"]] == ["a.md", "b.md", "c.md"]
    assert "external fixture" not in result["context"]


@pytest.mark.parametrize("budget", [1, len("\n## a.md:1\n".encode()) + 1])
def test_budget_cannot_admit_empty_heading_or_partial_utf8_character(tmp_path, budget):
    (tmp_path / "a.md").write_text("検索" * 20, encoding="utf-8")
    result = retrieve(tmp_path, "検索", budget, target_documents=1)
    assert result["status"] == "not_found"
    assert result["selected"] == [] and result["context"] == ""
    assert result["context_bytes"] == 0


@pytest.mark.parametrize("damage", ["invalid_json", "wrong_id", "wrong_shape", "missing_caps", "invalid_entry"])
def test_broken_capsules_fall_back_to_verifiable_source(documents, damage):
    path = documents / "docs/birdseye/a.md.json"
    if damage == "invalid_json":
        path.write_text("{", encoding="utf-8")
    elif damage in ("wrong_id", "wrong_shape"):
        payload = [] if damage == "wrong_shape" else {"id": "another.md"}
        path.write_text(json.dumps(payload), encoding="utf-8")
    else:
        index_path = documents / "docs/birdseye/index.json"
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        payload["nodes"]["a.md"] = {} if damage == "missing_caps" else None
        index_path.write_text(json.dumps(payload), encoding="utf-8")
    result = retrieve(documents, "target", 1000, target_documents=1)
    assert [entry["path"] for entry in result["selected"]] == ["a.md"]
    assert result["selected"][0]["via"] == "source_search"
    assert "target source" in result["context"]
    if damage != "invalid_entry":
        assert any("caps unavailable: a.md" in warning for warning in result["warnings"])


def test_dependency_shape_error_does_not_prevent_source_fallback(tmp_path):
    (tmp_path / "a.md").write_text("target", encoding="utf-8")
    (tmp_path / "b.md").write_text("fallbackword", encoding="utf-8")
    make_index(tmp_path, [("a.md", "target", "not-an-array"), ("b.md", "unrelated", [])])
    result = retrieve(tmp_path, "target fallbackword", 1000, target_documents=2)
    assert [entry["via"] for entry in result["selected"]] == ["birdseye", "source_search"]
    assert result["warnings"] == ["invalid dependencies: a.md"]


def test_required_and_multiple_seed_paths_are_not_duplicated(tmp_path):
    for name in ("a.md", "b.md", "c.md"):
        (tmp_path / name).write_text("target " + name, encoding="utf-8")
    make_index(
        tmp_path,
        [
            ("a.md", "target", ["b.md"]),
            ("b.md", "target", ["a.md", None, "missing.md"]),
            ("c.md", "other", []),
        ],
    )
    result = retrieve(tmp_path, "target", 1000, target_documents=5, required_paths=["a.md"])
    paths = [entry["path"] for entry in result["selected"]]
    assert paths == ["a.md", "b.md", "c.md"]
    assert result["context"].count("## a.md:1") == 1


def test_seed_selection_stops_at_target(tmp_path):
    for name in ("a.md", "b.md"):
        (tmp_path / name).write_text("target " + name, encoding="utf-8")
    make_index(tmp_path, [("a.md", "target", []), ("b.md", "target", [])])
    result = retrieve(tmp_path, "target", 1000, target_documents=1)
    assert [entry["path"] for entry in result["selected"]] == ["a.md"]


def test_dependencies_beyond_target_and_repeated_edges_do_not_add_documents(tmp_path):
    for name in ("a.md", "b.md", "c.md"):
        (tmp_path / name).write_text(name, encoding="utf-8")
    make_index(
        tmp_path, [("a.md", "target", ["a.md", "b.md", "b.md", "c.md"]), ("b.md", "other", []), ("c.md", "other", [])]
    )
    result = retrieve(tmp_path, "target", 1000, target_documents=2)
    assert [entry["path"] for entry in result["selected"]] == ["a.md", "b.md"]
    assert "c.md" not in result["context"]


def test_cli_invalid_query_is_json_error(documents, capsys):
    assert main(["--repo-root", str(documents), "--query", "!!!", "--budget-bytes", "100"]) == 1
    error = json.loads(capsys.readouterr().out)
    assert error["status"] == "invalid" and "searchable words" in error["error"]


def test_unindexed_document_is_found_and_unrelated_file_is_not_selected(documents):
    (documents / "new.md").write_text("migration v2 procedure", encoding="utf-8")
    result = retrieve(documents, "migration", 1000, target_documents=1)
    assert [item["path"] for item in result["selected"]] == ["new.md"]
    assert result["selected"][0]["via"] == "source_search"


def test_required_document_budget_has_exact_byte_boundary(documents):
    content = (documents / "a.md").read_text(encoding="utf-8")
    exact = len(("\n## a.md:1\n" + content).encode("utf-8"))
    passed = retrieve(documents, "target", exact, required_paths=["a.md"], target_documents=1)
    assert passed["context_bytes"] == exact
    assert passed["selected"][0]["via"] == "required"
    blocked = retrieve(documents, "target", exact - 1, required_paths=["a.md"])
    assert blocked["status"] == "insufficient_budget"
    assert blocked["context"] == ""
    assert blocked["required_bytes"] == exact


def test_unicode_excerpt_respects_actual_context_bytes(tmp_path):
    (tmp_path / "日本語.md").write_text("前置き\n" + "検索対象です" * 100, encoding="utf-8")
    result = retrieve(tmp_path, "検索", 70, target_documents=1)
    assert 0 < result["context_bytes"] <= 70
    assert len(result["context"].encode("utf-8")) == result["context_bytes"]
    assert result["selected"][0]["start_line"] == 2
    assert result["selected"][0]["excerpt"] is True
    assert result["status"] == "partial"
    assert "�" not in result["context"]


def test_no_matching_query_has_no_false_success(documents):
    result = retrieve(documents, "nonexistent", 1000)
    assert result["status"] == "not_found"
    assert result["context"] == ""
    assert result["selected"] == []


@pytest.mark.parametrize(
    "field,value", [("budget_bytes", 0), ("budget_bytes", True), ("max_hops", 3), ("query", "!!!")]
)
def test_invalid_inputs_are_rejected(documents, field, value):
    arguments = {"query": "target", "budget_bytes": 1000} | {field: value}
    with pytest.raises(ValueError):
        retrieve(documents, **arguments)


def test_outside_scope_is_rejected(documents):
    with pytest.raises(ValueError, match="outside repo"):
        retrieve(documents, "target", 1000, scope_paths=[".."])


def test_plugin_required_documents_take_priority_and_no_ack_is_sent(documents):
    class Plugin:
        capabilities = ("docs.resolve",)
        calls = 0

        def resolve_docs(self, **kwargs):
            self.calls += 1
            assert kwargs["task_id"] == "fixture"
            return {"required": [{"path": "b.md"}], "recommended": [], "errors": [], "warnings": ["fixture"]}

    plugin = Plugin()
    runtime = WorkflowPluginRuntime([plugin], default_policy=PluginPolicy(isolation_mode="inline"))
    result = retrieve(documents, "target", 1000, runtime=runtime, task_id="fixture", target_documents=1)
    assert plugin.calls == 1
    assert result["selected"][0]["path"] == "b.md"
    assert result["warnings"] == ["fixture"]


def test_plugin_errors_do_not_silently_drop_required_policy(documents):
    class Plugin:
        capabilities = ("docs.resolve",)

        def resolve_docs(self, **kwargs):
            return {"required": [], "recommended": [], "errors": ["missing policy"], "warnings": []}

    runtime = WorkflowPluginRuntime([Plugin()], default_policy=PluginPolicy(isolation_mode="inline"))
    with pytest.raises(ValueError, match="missing policy"):
        retrieve(documents, "target", 1000, runtime=runtime, task_id="fixture")


def test_cli_reports_json_and_budget_failure(documents, capsys):
    args = ["--repo-root", str(documents), "--query", "target", "--budget-bytes", "1", "--required", "a.md"]
    assert main(args) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "insufficient_budget"
    assert main(["--repo-root", str(documents), "--query", "target", "--budget-bytes", "1000"]) == 0
    assert json.loads(capsys.readouterr().out)["context_bytes"] > 0
