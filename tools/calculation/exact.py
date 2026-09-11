"""小さな整数式を厳密に評価し、演算の証跡を返す。"""

from __future__ import annotations

import ast
import re
from collections.abc import Mapping
from typing import Any

IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z_0-9]*\Z")


def integer(value: object) -> int:
    if type(value) is not int or value.bit_length() > 256:
        raise ValueError("integer of at most 256 bits required")
    return value


def evaluate(expression: str, bindings: Mapping[str, int]) -> dict[str, Any]:
    if not isinstance(expression, str) or not expression.strip() or len(expression) > 512:
        raise ValueError("nonempty expression of at most 512 characters required")
    if not isinstance(bindings, Mapping) or any(
        not isinstance(name, str) or not IDENTIFIER.fullmatch(name) for name in bindings
    ):
        raise ValueError("named integer bindings required")
    values = {name: integer(value) for name, value in bindings.items()}
    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, ValueError) as exc:
        raise ValueError("invalid integer expression") from exc
    if sum(1 for _ in ast.walk(tree)) > 64:
        raise ValueError("expression exceeds 64 AST nodes")
    used: set[str] = set()
    trace: list[dict[str, Any]] = []

    def visit(node: ast.AST, depth: int = 0) -> int:
        if depth > 16:
            raise ValueError("expression exceeds depth 16")
        if isinstance(node, ast.Constant):
            return integer(node.value)
        if isinstance(node, ast.Name):
            if node.id not in values:
                raise ValueError("unbound variable: " + node.id)
            used.add(node.id)
            return values[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand, depth + 1)
            result = integer(value if isinstance(node.op, ast.UAdd) else -value)
            trace.append({"operator": "unary+" if isinstance(node.op, ast.UAdd) else "unary-", "operand": value, "result": result})
            return result
        if not isinstance(node, ast.BinOp) or not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod)):
            raise ValueError("unsupported integer operation")
        left, right = visit(node.left, depth + 1), visit(node.right, depth + 1)
        if isinstance(node.op, ast.Add):
            operator, result = "+", left + right
        elif isinstance(node.op, ast.Sub):
            operator, result = "-", left - right
        elif isinstance(node.op, ast.Mult):
            operator, result = "*", left * right
        elif isinstance(node.op, ast.FloorDiv):
            if not right:
                raise ValueError("division by zero")
            operator, result = "//", left // right
        else:
            if not right:
                raise ValueError("modulo by zero")
            operator, result = "%", left % right
        result = integer(result)
        step = {"operator": operator, "left": left, "right": right, "result": result}
        if operator == "//":
            step["remainder"] = left % right
        trace.append(step)
        return result

    value = visit(tree.body)
    if used != set(values):
        raise ValueError("unused bindings: " + ", ".join(sorted(set(values) - used)))
    return {"expression": expression, "bindings": values, "result": value, "trace": trace}
