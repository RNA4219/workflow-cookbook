"""hash付き原文のscalarと表の全行から、同じ式の計算結果を作る。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tools.calculation.exact import IDENTIFIER, evaluate, integer


def text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("nonempty text required: " + label)
    return value


def numeric(value: str) -> int:
    if not re.fullmatch(r"[+-]?[0-9]{1,78}", value.strip()):
        raise ValueError("integer source value required")
    return integer(int(value))


def cells(line: str) -> list[str]:
    if "\\|" in line:
        raise ValueError("escaped table delimiters are unsupported")
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [part.strip() for part in line.split("|")]


def derive(documents: Mapping[str, Mapping[str, str]], recipe: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(documents, Mapping) or not isinstance(recipe, Mapping) or recipe.get("schema_version") != "1.0":
        raise ValueError("documents mapping and recipe version 1.0 required")
    sources: dict[str, dict[str, Any]] = {}

    def document(path: object) -> dict[str, Any]:
        name = text(path, "source path")
        if name not in documents or not isinstance(documents[name], Mapping):
            raise ValueError("source missing: " + name)
        if name not in sources:
            entry = documents[name]
            content = text(entry.get("content"), "source content")
            digest = "sha256:" + hashlib.sha256(content.encode("utf-8")).hexdigest()
            if entry.get("sha256") != digest:
                raise ValueError("source hash mismatch: " + name)
            sources[name] = {"path": name, "sha256": digest, "lines": content.splitlines()}
        return sources[name]

    scalar_specs = recipe.get("scalars")
    table = recipe.get("table")
    if not isinstance(scalar_specs, Mapping) or not isinstance(table, Mapping):
        raise ValueError("scalar and table specifications required")
    table_specs = table.get("bindings")
    if not isinstance(table_specs, Mapping) or not table_specs:
        raise ValueError("table bindings required")
    names = [*scalar_specs, *table_specs]
    if len(names) != len(set(names)) or any(not isinstance(n, str) or not IDENTIFIER.fullmatch(n) for n in names):
        raise ValueError("unique variable names required")
    scalars: dict[str, int] = {}
    scalar_refs: dict[str, dict[str, Any]] = {}
    for name, spec in scalar_specs.items():
        if not isinstance(spec, Mapping):
            raise ValueError("scalar specification required")
        source = document(spec.get("path"))
        key = text(spec.get("key"), "scalar key")
        pattern = re.compile(r"^\s*" + re.escape(key) + r"\s*=\s*(.*?)\s*$")
        matches = [(line_number, match.group(1)) for line_number, line in enumerate(source["lines"], 1)
                   if (match := pattern.fullmatch(line)) is not None]
        if len(matches) != 1:
            raise ValueError("scalar must occur exactly once: " + key)
        line_number, value = matches[0]
        scalars[name] = numeric(value)
        scalar_refs[name] = {"path": source["path"], "line": line_number, "key": key}
    source = document(table.get("path"))
    row_key = text(table.get("row_key"), "table row key")
    columns = {name: text(column, "table column") for name, column in table_specs.items()}
    required = {row_key, *columns.values()}
    tables = []
    lines = source["lines"]
    for index in range(len(lines) - 1):
        if "|" not in lines[index] or "|" not in lines[index + 1]:
            continue
        header, divider = cells(lines[index]), cells(lines[index + 1])
        if not set(header).issuperset(required) or not all(re.fullmatch(r":?-{3,}:?", part) for part in divider):
            continue
        if len(header) != len(set(header)) or len(header) != len(divider):
            raise ValueError("ambiguous table header")
        tables.append((index, header))
    if len(tables) != 1:
        raise ValueError("exactly one matching table required")
    index, header = tables[0]
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for offset in range(index + 2, len(lines)):
        if not lines[offset].strip() or "|" not in lines[offset]:
            break
        values = cells(lines[offset])
        if len(values) != len(header):
            raise ValueError("table row column count mismatch")
        row = dict(zip(header, values, strict=True))
        identity = text(row[row_key], "row identity")
        if identity in seen:
            raise ValueError("duplicate table row identity")
        seen.add(identity)
        if len(seen) > 512:
            raise ValueError("table exceeds 512 rows")
        bindings = {**scalars, **{name: numeric(row[column]) for name, column in columns.items()}}
        calculated = evaluate(text(recipe.get("expression"), "expression"), bindings)
        refs = {**scalar_refs, **{name: {"path": source["path"], "line": offset + 1, "column": column}
                                  for name, column in columns.items()}}
        rows.append({"row_id": identity, **calculated, "binding_sources": refs})
    if not rows:
        raise ValueError("nonempty table required")
    return {"schema_version": "1.0", "expression": recipe["expression"], "row_key": row_key,
            "sources": [{k: v for k, v in item.items() if k != "lines"} for item in sources.values()], "rows": rows}


def render(receipt: Mapping[str, Any]) -> str:
    """deriveで生成した全行を、元資料への参照付きで表示する。"""
    names = list(receipt["rows"][0]["bindings"])
    header = [receipt["row_key"], *names, "厳密な計算結果"]
    lines = ["\n## 原文からの計算ツール結果", "", "式: `" + receipt["expression"] + "`。全行を同じ式で計算。",
             "原文: " + ", ".join(source["path"] for source in receipt["sources"]),
             "この表は補助計算です。採用する行は現在の決定から選び、引用は原文のpathを使ってください。", "",
             " | ".join(header), " | ".join("---" for _ in header)]
    for row in receipt["rows"]:
        lines.append(" | ".join(str(value) for value in [row["row_id"], *(row["bindings"][name] for name in names), row["result"]]))
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        request = json.loads(args.input.read_text(encoding="utf-8"))
        if not isinstance(request, dict):
            raise ValueError("request object required")
        documents, recipe = request.get("documents"), request.get("recipe")
        if not isinstance(documents, Mapping) or not isinstance(recipe, Mapping):
            raise ValueError("documents and recipe mappings required")
        result = derive(documents, recipe)
    except (OSError, ValueError, TypeError) as exc:
        print(json.dumps({"status": "invalid", "error": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
