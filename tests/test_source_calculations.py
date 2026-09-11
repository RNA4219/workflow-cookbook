"""整数式と原文からの計算。既知benchmarkの正解をfixtureへ転記しない。"""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from tools.calculation.exact import evaluate
from tools.calculation.sources import derive, main, render


def source(content):
    return {"content": content, "sha256": "sha256:" + hashlib.sha256(content.encode()).hexdigest()}


def fixture():
    documents = {
        "capacity.md": source("# Capacity\ncapacity=103\nbuffer=4\n"),
        "choices.md": source("# Resources\nsku | hours | memo\n--- | ---: | ---\nZ | 10 | alpha\nA | 11 | beta\n"),
    }
    recipe = {"schema_version": "1.0", "expression": "(capacity-buffer)//hours",
              "scalars": {"capacity": {"path": "capacity.md", "key": "capacity"},
                          "buffer": {"path": "capacity.md", "key": "buffer"}},
              "table": {"path": "choices.md", "row_key": "sku", "bindings": {"hours": "hours"}}}
    return documents, recipe


@pytest.mark.parametrize(("expression", "bindings", "expected"), [
    ("(available-reserve)//cost", {"available": 103, "reserve": 4, "cost": 10}, 9),
    ("(available-reserve)//cost", {"available": 104, "reserve": 4, "cost": 10}, 10),
    ("(available-reserve)//cost", {"available": 105, "reserve": 4, "cost": 10}, 10),
    ("-7//3", {}, -3), ("7//-3", {}, -3), ("-7//-3", {}, 2),
    ("0//7", {}, 0), ("(a*3 + +b) % 7", {"a": 5, "b": 4}, 5), ("42", {}, 42),
])
def test_exact_arithmetic(expression, bindings, expected):
    result = evaluate(expression, bindings)
    assert result["result"] == expected
    assert result["expression"] == expression and result["bindings"] == bindings


def test_quotient_identity_on_generated_numbers():
    rng = random.Random(21974)
    for _ in range(250):
        numerator, denominator = rng.randrange(-10**30, 10**30), rng.randrange(1, 10**15)
        result = evaluate("n//d", {"n": numerator, "d": denominator})
        step = result["trace"][-1]
        assert step["result"] * denominator + step["remainder"] == numerator
        assert 0 <= step["remainder"] < denominator
        assert step["result"] * denominator <= numerator < (step["result"] + 1) * denominator


@pytest.mark.parametrize(("expression", "bindings", "match"), [
    ("", {}, "nonempty"), ("1" * 513, {}, "512 characters"), (None, {}, "nonempty"),
    ("1+", {}, "invalid"), ("x", {"bad-name": 1}, "named"), ("x", [], "named"),
    ("x", {"x": True}, "integer"), ("x", {"x": "1"}, "integer"),
    ("x", {"x": 1 << 256}, "256 bits"), ("1.5", {}, "integer"), ("True", {}, "integer"),
    ("unknown", {}, "unbound"), ("1", {"unused": 2}, "unused"),
    ("a/a", {"a": 2}, "unsupported"), ("2**3", {}, "unsupported"),
    ("abs(1)", {}, "unsupported"), ("[1]", {}, "unsupported"),
    ("1//0", {}, "division by zero"), ("1%0", {}, "modulo by zero"),
    ("x*x", {"x": 1 << 200}, "256 bits"),
    ("+" * 18 + "1", {}, "depth 16"), ("+".join("x" for _ in range(40)), {"x": 1}, "64 AST"),
])
def test_invalid_integer_expressions(expression, bindings, match):
    with pytest.raises(ValueError, match=match):
        evaluate(expression, bindings)


def test_all_rows_provenance_and_original_inputs_unchanged():
    documents, recipe = fixture()
    original = deepcopy((documents, recipe))
    result = derive(documents, recipe)
    assert [(row["row_id"], row["result"]) for row in result["rows"]] == [("Z", 9), ("A", 9)]
    assert result["rows"][0]["binding_sources"] == {
        "capacity": {"path": "capacity.md", "line": 2, "key": "capacity"},
        "buffer": {"path": "capacity.md", "line": 3, "key": "buffer"},
        "hours": {"path": "choices.md", "line": 4, "column": "hours"},
    }
    assert result["rows"][0]["trace"] == [
        {"operator": "-", "left": 103, "right": 4, "result": 99},
        {"operator": "//", "left": 99, "right": 10, "result": 9, "remainder": 9},
    ]
    assert result["sources"] == [{"path": p, "sha256": documents[p]["sha256"]} for p in documents]
    assert (documents, recipe) == original
    shown = render(result)
    assert "Z | 103 | 4 | 10 | 9" in shown and "A | 103 | 4 | 11 | 9" in shown
    assert "capacity.md" in shown and "choices.md" in shown


def test_source_change_and_reordered_rows_recompute_every_alternative():
    documents, recipe = fixture()
    documents["capacity.md"] = source("capacity=205\r\nbuffer=5\r\n")
    documents["choices.md"] = source("| memo | hours | sku |\r\n| --- | --- | --- |\r\n| beta | 25 | A |\r\n| alpha | 5 | Z |\r\n")
    result = derive(documents, recipe)
    assert [(row["row_id"], row["result"]) for row in result["rows"]] == [("A", 8), ("Z", 40)]


@pytest.mark.parametrize(("change", "match"), [
    ("hash", "hash mismatch"), ("missing", "source missing"), ("empty", "nonempty"),
    ("repeat_scalar", "exactly once"), ("missing_scalar", "exactly once"), ("decimal", "integer"),
    ("repeat_table", "exactly one"), ("missing_table", "exactly one"), ("empty_table", "nonempty table"),
    ("duplicate_row", "duplicate table"), ("ragged_row", "column count"), ("bad_cell", "integer"),
    ("zero", "division by zero"), ("escaped", "escaped table"), ("duplicate_header", "ambiguous"),
    ("short_divider", "ambiguous"), ("too_many_rows", "512 rows"), ("blank_row_key", "row identity"),
])
def test_source_failures_are_not_silently_ignored(change, match):
    documents, recipe = fixture()
    content = documents["choices.md"]["content"]
    if change == "hash":
        documents["choices.md"]["content"] += "changed"
    elif change == "missing":
        del documents["choices.md"]
    elif change == "empty":
        documents["choices.md"] = source("")
    elif change in ("repeat_scalar", "missing_scalar", "decimal"):
        documents["capacity.md"] = source({"repeat_scalar": "capacity=1\ncapacity=2\nbuffer=0\n",
                                            "missing_scalar": "capacity=1\n", "decimal": "capacity=1.2\nbuffer=0\n"}[change])
    else:
        changed = {
            "repeat_table": content + "\n" + content,
            "missing_table": "sku | other\n--- | ---\nX | 1\n",
            "empty_table": "sku | hours\n--- | ---\n",
            "duplicate_row": content.replace("A | 11", "Z | 11"),
            "ragged_row": content.replace("A | 11 | beta", "A | 11"),
            "bad_cell": content.replace("A | 11", "A | text"),
            "zero": content.replace("A | 11", "A | 0"),
            "escaped": content.replace("beta", "b\\|eta"),
            "duplicate_header": "sku | hours | hours\n--- | --- | ---\nX | 1 | 2\n",
            "short_divider": "sku | hours\n--- | --- | ---\nX | 1\n",
            "too_many_rows": "sku | hours\n--- | ---\n" + "".join(f"R{i} | 1\n" for i in range(513)),
            "blank_row_key": "| sku | hours | memo |\n| --- | --- | --- |\n| | 11 | beta |\n",
        }[change]
        documents["choices.md"] = source(changed)
    with pytest.raises(ValueError, match=match):
        derive(documents, recipe)


@pytest.mark.parametrize("mutation", ["version", "no_scalar_mapping", "no_table_mapping", "no_bindings", "duplicate_name", "bad_name", "bad_scalar_spec", "empty_column"])
def test_invalid_recipes(mutation):
    documents, recipe = fixture()
    if mutation == "version":
        recipe["schema_version"] = "9"
    elif mutation == "no_scalar_mapping":
        recipe["scalars"] = []
    elif mutation == "no_table_mapping":
        recipe["table"] = []
    elif mutation == "no_bindings":
        recipe["table"]["bindings"] = {}
    elif mutation == "duplicate_name":
        recipe["table"]["bindings"]["buffer"] = "hours"
    elif mutation == "bad_name":
        recipe["table"]["bindings"]["not a name"] = "hours"
    elif mutation == "bad_scalar_spec":
        recipe["scalars"]["capacity"] = None
    else:
        recipe["table"]["bindings"]["hours"] = ""
    with pytest.raises(ValueError):
        derive(documents, recipe)


def test_multiple_table_variables_and_no_scalar_values():
    documents = {"jobs.md": source("job | duration | workers\n--- | --- | ---\nQA | 3 | 7\n\nend\n")}
    recipe = {"schema_version": "1.0", "expression": "duration*workers", "scalars": {},
              "table": {"path": "jobs.md", "row_key": "job", "bindings": {"duration": "duration", "workers": "workers"}}}
    assert derive(documents, recipe)["rows"][0]["result"] == 21


def test_cli_success(tmp_path, capsys):
    documents, recipe = fixture()
    request = tmp_path / "input.json"
    request.write_text(json.dumps({"documents": documents, "recipe": recipe}), encoding="utf-8")
    assert main(["--input", str(request)]) == 0
    assert json.loads(capsys.readouterr().out)["rows"][1]["result"] == 9
    proc = subprocess.run([sys.executable, "-B", "-m", "tools.calculation.sources", "--input", str(request)],
                          cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, encoding="utf-8")
    assert proc.returncode == 0
    assert json.loads(proc.stdout)["rows"][0]["result"] == 9


@pytest.mark.parametrize("content", ["null", "{}", "{invalid}", "[]"])
def test_cli_invalid_json(tmp_path, capsys, content):
    request = tmp_path / "invalid.json"
    request.write_text(content, encoding="utf-8")
    assert main(["--input", str(request)]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "invalid"


def test_cli_missing_input(tmp_path, capsys):
    assert main(["--input", str(tmp_path / "absent.json")]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "invalid"
