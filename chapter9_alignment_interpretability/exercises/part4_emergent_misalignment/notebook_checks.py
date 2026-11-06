"""Structural notebook checks; scientific correctness belongs to numerical tests."""

from __future__ import annotations

import ast


def unfinished_definitions(source: str | ast.AST) -> list[str]:
    """Name functions with empty bodies or explicit unimplemented exceptions.

    Pass parsed code, never a notebook's Markdown. A mentioned exception in a
    comment or docstring is not an unfinished implementation. Methods and
    nested functions are checked individually, while normal argument-validation
    exceptions are permitted.
    """
    tree = ast.parse(source) if isinstance(source, str) else source
    unfinished: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = [statement for statement in node.body if not (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
            and (isinstance(statement.value.value, str) or statement.value.value is Ellipsis)
        )]
        empty = not body or all(isinstance(statement, ast.Pass) for statement in body)
        explicit = False
        for statement in ast.walk(node):
            if not isinstance(statement, ast.Raise):
                continue
            exception = statement.exc.func if isinstance(statement.exc, ast.Call) else statement.exc
            name = exception.id if isinstance(exception, ast.Name) else getattr(exception, "attr", None)
            explicit |= name == "NotImplementedError"
        if empty or explicit:
            unfinished.append(node.name)
    return unfinished
