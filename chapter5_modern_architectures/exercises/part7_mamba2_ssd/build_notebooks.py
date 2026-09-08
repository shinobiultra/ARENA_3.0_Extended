"""Render the Mamba-2 lesson, paired notebooks, and instruction page.

Edit prose and experiment cells in lesson.md and implementations in solutions.py.
Definitions are inserted as readable Python into every learner-facing surface.
No notebook executes a reference implementation on the learner's behalf.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import nbformat

SECTION = Path(__file__).resolve().parent
STEM = "5.7_Mamba-2_and_SSD"
PAGE = SECTION.parents[1] / "instructions/pages/07_[5.7]_Mamba-2_and_Structured_State-Space_Duality.md"
CELL = re.compile(r"^<!-- cell: (markdown|code)(?:; exercises: (.*?))? -->\n", re.M)
DEFINITION = re.compile(r"\{\{definition:(\w+)\}\}")


def definitions() -> dict[str, str]:
    source = (SECTION / "solutions.py").read_text()
    lines = source.splitlines(keepends=True)
    result = {}
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            start = min([node.lineno] + [item.lineno for item in node.decorator_list]) - 1
            result[node.name] = "".join(lines[start:node.end_lineno]).rstrip()
    return result


def exercise_source(source: str, names: set[str]) -> str:
    """Keep complete signatures and scaffolding while hiding assigned bodies."""
    lines = source.splitlines(keepends=True)
    replacements = []
    for node in ast.parse(source).body:
        if not isinstance(node, (ast.FunctionDef, ast.ClassDef)) or node.name not in names:
            continue
        methods = node.body if isinstance(node, ast.ClassDef) else [node]
        for method in methods:
            if not isinstance(method, ast.FunctionDef):
                continue
            body = method.body
            if isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                body = body[1:]  # Preserve the function's teaching docstring.
            if not body:
                continue
            indent = " " * (method.col_offset + 4)
            replacements.append((body[0].lineno - 1, body[-1].end_lineno,
                                 indent + "raise NotImplementedError()\n"))
    for start, end, replacement in sorted(replacements, reverse=True):
        lines[start:end] = [replacement]
    return "".join(lines).rstrip()


def render() -> tuple[nbformat.NotebookNode, nbformat.NotebookNode, str]:
    lesson = (SECTION / "lesson.md").read_text()
    matches = list(CELL.finditer(lesson))
    implementations = definitions()
    exercise_cells, solution_cells, page = [], [], []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(lesson)
        source = lesson[match.end():end].strip()
        source = DEFINITION.sub(lambda item: implementations[item[1]], source)
        cell_id = f"mamba2-{index:02d}"
        if match[1] == "markdown":
            for code in re.findall(r"```python\n(.*?)```", source, re.S):
                ast.parse(code)  # Reject broken copy-paste solutions at build time.
            exercise_cells.append(nbformat.v4.new_markdown_cell(source, id=cell_id))
            solution_cells.append(nbformat.v4.new_markdown_cell(source, id=cell_id))
            page.append(source)
        else:
            ast.parse(source)
            names = {name for name in (match[2] or "").split(",") if name}
            student = exercise_source(source, names)
            ast.parse(student)
            metadata = {"exercise_functions": sorted(names)} if names else {}
            exercise_cells.append(nbformat.v4.new_code_cell(student, id=cell_id, metadata=metadata))
            solution_cells.append(nbformat.v4.new_code_cell(source, id=cell_id, metadata=metadata))
            page.append(f"```python\n{student}\n```")
    metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    return (nbformat.v4.new_notebook(cells=exercise_cells, metadata=metadata),
            nbformat.v4.new_notebook(cells=solution_cells, metadata=metadata),
            "\n\n".join(page) + "\n")


def main() -> None:
    learner, solved, page = render()
    solution_path = SECTION / f"{STEM}_solutions.ipynb"
    if solution_path.exists():
        old = nbformat.read(solution_path, as_version=4)
        # An unchanged prefix retains outputs. Any changed code invalidates all
        # later outputs because those cells may depend on its runtime state.
        valid = True
        for new, previous in zip(solved.cells, old.cells):
            if new.cell_type == "code":
                valid = valid and previous.cell_type == "code" and new.source == previous.source
                if valid:
                    new.outputs = previous.outputs
                    new.execution_count = previous.execution_count
    nbformat.write(learner, SECTION / f"{STEM}_exercises.ipynb")
    nbformat.write(solved, solution_path)
    PAGE.write_text(page)


if __name__ == "__main__":
    main()
