# SPDX-License-Identifier: MIT
"""実JUnitとHATE正規化結果を照合し、QEGのnative実行契約へ投影する。"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from defusedxml import ElementTree as ET


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def file_hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def value_hash(value: Any) -> str:
    return (
        "sha256:"
        + hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    )


def reconcile(junit: Path, records: list[dict[str, Any]], run_id: str, revision: str) -> list[dict[str, Any]]:
    """件数だけでなくsuite/class/name/statusを全件照合する。"""
    root = ET.parse(junit, forbid_dtd=True).getroot()
    suites = [root] if root.tag == "testsuite" else root.findall(".//testsuite")
    observed = []
    for suite in suites:
        for case in suite.findall("testcase"):
            status = next(
                (
                    s
                    for tag, s in (("error", "error"), ("failure", "failed"), ("skipped", "skipped"))
                    if case.find(tag) is not None
                ),
                "passed",
            )
            observed.append((suite.get("name", "junit"), case.get("classname", ""), case.get("name"), status))
    normalized = []
    ids = set()
    for record in records:
        if record.get("run_id") != run_id or record.get("commit_sha") != revision or record.get("run_attempt") != 1:
            raise ValueError("HATE run/revision/attempt mismatch")
        payload = record["payload"]
        identity = payload["identity_components"]
        normalized.append((identity["suite"], identity["classname"], identity["name"], payload["status"]))
        canonical = payload["canonical_test_id"]
        if canonical in ids:
            raise ValueError("duplicate HATE test identity")
        ids.add(canonical)
    if not observed or len(set(row[:3] for row in observed)) != len(observed):
        raise ValueError("empty or duplicate JUnit cases")
    if Counter(observed) != Counter(normalized):
        raise ValueError("JUnit/HATE cases or statuses do not match")
    return records


def project(output: Path, receipt: dict[str, Any], *, scope: str = "real_environment") -> dict[str, Any]:
    """output配下の原本だけを読み、同じdirectory内にQEG入力を生成する。"""
    revision = receipt["target"]["revision"]
    run_id = receipt["run_id"]
    records = [
        json.loads(line)
        for line in (output / "hate/p0a/HATE-test-results.ndjson").read_text(encoding="utf-8").splitlines()
        if line
    ]
    reconcile(output / "raw/junit.xml", records, run_id, revision)
    run = read_json(output / "hate/p0a/HATE-run.json")
    if (run["run_id"], run["commit_sha"], run["payload"]["finished_at"]) != (run_id, revision, receipt["finished_at"]):
        raise ValueError("HATE run receipt mismatch")
    precheck = read_json(output / "hate/p0a/precheck-decision.json")["payload"]
    report = read_json(output / "hate/export/qeg-export-report.json")
    if (
        precheck["decision"] != "eligible"
        or report["export_status"] != "success"
        or not report["qeg_schema_compatibility"]["valid"]
    ):
        raise ValueError("HATE evidence is incomplete or ineligible")
    if report["run_id"] != run_id or report["commit_sha"] != revision:
        raise ValueError("HATE export identity mismatch")
    coverage = read_json(output / "raw/coverage-summary.json")["totals"]
    inputs = [
        ("raw/junit.xml", "junit", "execution_evidence"),
        ("raw/cobertura.xml", "cobertura", "coverage"),
        ("raw/coverage-summary.json", "coverage.py", "coverage"),
        ("hate/p0a/HATE-test-results.ndjson", "hate", "test_results"),
        ("hate/p0a/HATE-run.json", "hate", "run"),
        ("hate/p0a/precheck-decision.json", "hate", "precheck"),
        ("hate/export/qeg-bundle.json", "hate", "qeg_bundle"),
        ("hate/export/qeg-export-report.json", "hate", "export_report"),
        ("receipt.json", "qeg-native", "test_model"),
        ("source-snapshot.json", "qeg-native", "source_snapshot"),
    ]
    artifacts = [
        {
            "id": f"qeg:input-{i}",
            "path": path,
            "adapter": adapter,
            "kind": kind,
            "contentHash": file_hash(output / path),
            "revision": revision,
        }
        for i, (path, adapter, kind) in enumerate(inputs)
    ]
    sources = [
        {
            "id": "qeg:source-receipt",
            "path": "receipt.json",
            "revision": revision,
        }
    ]
    trace = {
        "sourceRefs": sources,
        "assumptions": [
            "Local pytest execution; individual tests may use mocks. No production service or release approval evaluated."
        ],
        "confidence": "high",
    }
    target = receipt["target"]
    write_json(output / "build-binding.json", {"bindingVersion": "qeg-build/v1", "target": target})
    policy = {
        "policyId": "qeg:workflow-automated-test-acceptance-v1",
        "profile": "standard",
        "effectiveDate": receipt["started_at"],
        "approver": "local-technical-test-policy-no-release-approval",
        "sourceRefs": sources,
        "dqScope": [f"DQ-{i:02}" for i in range(1, 22)],
        "exitCodePolicy": {"go": 0, "conditional_go": 2, "no_go": 2, "disqualified": 2},
        "inputContract": {
            "mode": "native_graph",
            "requireExecutedTests": True,
            "sourceRefs": sources,
            "requiredArtifacts": [
                {"adapter": adapter, "kind": kind} for adapter, kind in sorted({(a, k) for _, a, k in inputs})
            ],
            "evaluationScope": {
                "kind": scope,
                "target": "workflow-cookbook local automated tests",
                "notEvaluated": [
                    "RanD requirements audit",
                    "Code-to-gate analysis",
                    "Manual black-box QA",
                    "Model performance comparison",
                    "Production deployment",
                    "Release approval",
                ],
            },
        },
        "executionPolicy": {
            "target": target,
            "maxEvidenceAgeHours": 24,
            "sourceRefs": sources,
            "buildBindingRef": {
                "id": "qeg:build",
                "path": "build-binding.json",
                "contentHash": file_hash(output / "build-binding.json"),
                "revision": revision,
            },
        },
    }
    policy["policyHash"] = value_hash(policy)
    write_json(output / "policy.json", policy)
    metadata = {
        "qegVersion": "0.2",
        "runId": "qeg:" + run_id,
        "createdAt": receipt["evaluated_at"],
        "headRef": revision,
        "profile": "standard",
        "policyId": policy["policyId"],
        "policyHash": policy["policyHash"],
        "inputArtifacts": artifacts,
        "requiredConnectorStatus": {adapter: "success" for _, adapter, _ in inputs},
    }
    requirement_id = "qeg:requirement-local-regression"
    nodes: list[dict[str, Any]] = [
        {
            "id": requirement_id,
            "kind": "requirement",
            "title": "All collected regression tests pass and measured coverage meets the configured threshold",
            "sourceArtifactIds": ["qeg:input-8"],
            "traceability": trace,
            "acceptanceCriteriaIds": [],
        }
    ]
    edges = []
    test_ids = []
    items = [
        (
            r["payload"]["canonical_test_id"],
            {"passed": "pass", "failed": "fail", "error": "fail", "skipped": "skipped"}[r["payload"]["status"]],
            r["source_version"],
        )
        for r in records
    ]
    # pytestのcoverage plugin failure / collection errorをcaseのpassで相殺しない。
    items.append(
        (
            "pytest-process-and-coverage",
            "pass"
            if receipt["pytest_exit_code"] == 0 and coverage["percent_covered"] >= receipt["coverage_min"]
            else "fail",
            "workflow-hate-qeg/v1",
        )
    )
    for index, (canonical, status, version) in enumerate(items):
        test_id = "qeg:test-" + hashlib.sha256(canonical.encode()).hexdigest()
        evidence_id = f"qeg:execution-{index}"
        identity = {
            "producer": "qeg-native",
            "projectId": target["projectId"],
            "featureId": "pytest-regression",
            "caseId": canonical,
        }
        raw = {
            "executionVersion": "qeg-execution/v1",
            "testId": test_id,
            "identity": identity,
            "producerVersion": "workflow-hate-qeg/v1+HATE/" + version,
            "target": target,
            "runId": run_id,
            "completedAt": receipt["finished_at"],
            "status": status,
            "executionMode": "real",
        }
        raw_path = f"execution/{index}.json"
        write_json(output / raw_path, raw)
        ref = {
            "id": f"qeg:raw-{index}",
            "path": raw_path,
            "contentHash": file_hash(output / raw_path),
            "revision": revision,
        }
        node_trace = {
            **trace,
            "sourceRefs": [
                *sources,
                {
                    "id": f"qeg:hate-record-{index}",
                    "path": "hate/p0a/HATE-test-results.ndjson" if index < len(records) else "receipt.json",
                    "label": canonical,
                },
            ],
        }
        nodes.append(
            {
                "id": test_id,
                "kind": "test",
                "title": canonical,
                "sourceArtifactIds": ["qeg:input-3", "qeg:input-8"],
                "traceability": node_trace,
                "testExecutionMode": "real",
                "existing": True,
                "layer": "unit",
                "executionIdentity": identity,
            }
        )
        nodes.append(
            {
                "id": evidence_id,
                "kind": "execution_evidence",
                "title": canonical + " execution",
                "sourceArtifactIds": ["qeg:input-3", "qeg:input-8"],
                "traceability": node_trace,
                "execution": {**raw, "rawArtifactRef": ref},
                **({"passed": status == "pass"} if status in {"pass", "fail"} else {}),
                "evidenceRefs": [{**ref, "evidenceKind": "test_result", "capturedAt": receipt["finished_at"]}],
            }
        )
        edges.append(
            {
                "id": f"qeg:edge-{index}",
                "kind": "evidenced_by",
                "from": test_id,
                "to": evidence_id,
                "traceability": node_trace,
            }
        )
        test_ids.append(test_id)
    graph = {
        "metadata": metadata,
        "nodes": nodes,
        "edges": edges,
        "completeness": {"score": 1, "partial": False, "parserFailures": [], "unsupportedClaims": []},
    }
    obligation = {
        "id": "qeg:obligation-regression",
        "requirementIds": [requirement_id],
        "riskIds": [],
        "failureModeIds": [],
        "changedCodeIds": [],
        "priority": "P1",
        "riskPriorityIndex": 0.5,
        "gateRelevance": "blocking",
        "traceability": trace,
    }
    placement = {
        "id": "qeg:placement-regression",
        "kind": "test_placement",
        "title": "Existing pytest regression suite",
        "traceability": trace,
        "sourceArtifactIds": ["qeg:input-8"],
        "obligationId": obligation["id"],
        "primaryLayer": "unit",
        "disposition": "reuse",
        "gateRelevance": "blocking",
        "candidateScores": [],
        "selectedTestIds": test_ids,
    }
    result = {
        "metadata": metadata,
        "graph": graph,
        "policy": policy,
        "waivers": [],
        "placementPlan": {"metadata": metadata, "obligations": [obligation], "placements": [placement]},
    }
    write_json(output / "gate-input.json", result)
    return {
        "test_cases": len(records),
        "qeg_executions": len(items),
        "statuses": dict(Counter(r["payload"]["status"] for r in records)),
        "coverage": coverage,
        "policy_hash": policy["policyHash"],
    }
