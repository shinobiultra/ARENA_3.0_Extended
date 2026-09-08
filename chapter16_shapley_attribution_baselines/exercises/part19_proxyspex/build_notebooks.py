"""Build, execute and render unpublished 16.19 drafts from their own visible answers."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chapter16_shapley_attribution_baselines.exercises.part19_proxyspex.lesson_cells import CELLS, TEST_CALLS


STEM = "16.19_Explain_the_Proxy_Check_the_Model"
TOKEN = re.compile(r"<!-- (code|exercise|solution): ([\w,]+) -->")
FROZEN_METHOD_SHA256 = "c1eecc731bde4fefd7c0d30f07ac22196d96fe3560f389b9f8561bb9f5bc5a01"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def function_sources():
    source = (HERE / "proxy_methods.py").read_text()
    if hashlib.sha256(source.encode()).hexdigest() != FROZEN_METHOD_SHA256:
        raise ValueError("Frozen method source changed; review the scientific freeze before rebuilding.")
    return {node.name: ast.get_source_segment(source, node) for node in ast.parse(source).body
            if isinstance(node, ast.FunctionDef)}


def _stub(source):
    node = ast.parse(source).body[0]
    first = node.body[0]
    doc = first if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str) else None
    node.body = ([doc] if doc else []) + [ast.Raise(
        exc=ast.Call(func=ast.Name(id="NotImplementedError", ctx=ast.Load()), args=[], keywords=[]), cause=None)]
    return ast.unparse(ast.fix_missing_locations(node))


def _cell(kind, source, **arena):
    cell = {"cell_type": kind, "id": hashlib.sha256((kind+"\n"+source).encode()).hexdigest()[:12],
            "metadata": {"arena": arena} if arena else {}, "source": source}
    if kind == "code":
        cell.update(execution_count=None, outputs=[])
    return cell


def build_notebooks():
    text = (HERE / "lesson.md").read_text()
    sources = function_sources()
    student, solved, position = [], [], 0
    for token in TOKEN.finditer(text):
        prose = text[position:token.start()].strip()
        if prose:
            student.append(_cell("markdown", prose)); solved.append(_cell("markdown", prose))
        kind, key = token.groups()
        if kind == "code":
            cell = _cell("code", CELLS[key].strip(), calling_cell=key)
            student.append(copy.deepcopy(cell)); solved.append(cell)
        elif kind == "exercise":
            names = key.split(",")
            checks = "\n".join(TEST_CALLS[name] for name in names)
            answer = "\n\n".join(sources[name] for name in names)
            stub = "\n\n".join(_stub(sources[name]) for name in names)
            identifier = hashlib.sha256(("exercise:"+key).encode()).hexdigest()[:12]
            exercise = _cell("code", stub+"\n\n"+checks, exercise_functions=names)
            solution = _cell("code", answer+"\n\n"+checks, exercise_functions=names)
            exercise["id"] = solution["id"] = identifier
            student.append(exercise); solved.append(solution)
        else:
            names = key.split(",")
            answer = "\n\n".join(sources[name] for name in names)
            student.append(_cell("markdown", "<details>\n<summary>Solution</summary>\n\n```python\n"+answer+
                                 "\n```\n\n</details>", solution_for=names))
        position = token.end()
    tail = text[position:].strip()
    if tail:
        student.append(_cell("markdown", tail)); solved.append(_cell("markdown", tail))
    metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python", "version": sys.version.split()[0]},
                "arena": {"section": "16.19", "status": "analytic_draft_native_unexecuted",
                          "original_chapters_modified": False, "method_sha256": FROZEN_METHOD_SHA256}}
    make = lambda cells: {"nbformat": 4, "nbformat_minor": 5, "metadata": copy.deepcopy(metadata), "cells": cells}
    return make(student), make(solved)


def complete_from_own_answers(student):
    notebook = copy.deepcopy(student)
    answers = {}
    for cell in notebook["cells"]:
        names = cell.get("metadata", {}).get("arena", {}).get("solution_for")
        if names:
            answers[tuple(names)] = re.search(r"```python\n(.*?)\n```", cell["source"], re.S).group(1)
    completed = []
    for cell in notebook["cells"]:
        info = cell.get("metadata", {}).get("arena", {})
        if info.get("solution_for"):
            continue
        if info.get("exercise_functions"):
            names = info["exercise_functions"]
            cell["source"] = answers[tuple(names)]+"\n\n"+"\n".join(TEST_CALLS[name] for name in names)
        completed.append(cell)
    notebook["cells"] = completed
    _, solved = build_notebooks()
    assert [(c["cell_type"], c["source"]) for c in completed] == [(c["cell_type"], c["source"]) for c in solved["cells"]]
    return notebook


def write_drafts(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    student, solved = build_notebooks()
    complete_from_own_answers(student)
    for variant, notebook in (("exercises", student), ("solutions", solved)):
        (output / f"DRAFT_{STEM}_{variant}.ipynb").write_text(json.dumps(notebook, indent=1, ensure_ascii=False)+"\n")
    return student, solved


def execute_analytic(output):
    """Execute every draft cell, including explicit unexecuted-native markers."""
    import nbformat
    from nbclient import NotebookClient

    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new empty draft execution directory to preserve previous outcomes.")
    output.mkdir(parents=True, exist_ok=True)
    student, _ = write_drafts(output)
    notebook = nbformat.reads(json.dumps(complete_from_own_answers(student)), as_version=4)
    sources = [HERE/name for name in ("proxy_methods.py", "proxy_methods_tests.py", "proxy_learner_tests.py",
                                      "proxy_plots.py", "lesson.md", "lesson_cells.py", "build_notebooks.py")]
    before = {path.name: digest(path) for path in sources}
    for path in sources:
        destination = output / "source_snapshot" / path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    environment = {**os.environ, "PATH": str(Path(sys.executable).parent)+os.pathsep+os.environ.get("PATH", ""),
                   "ARENA_PROXYSPEX_OUTPUT_DIRECTORY": str(output / "figures"),
                   "OPENBLAS_NUM_THREADS": "4", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}
    record = {"status": "RUNNING_ANALYTIC_DRAFT", "completed_from_own_adjacent_answers": True,
              "source_inputs": before, "method_sha256": FROZEN_METHOD_SHA256,
              "native_execution": "UNEXECUTED", "model_queries": 0}
    start = time.perf_counter()
    error = None
    try:
        NotebookClient(notebook, timeout=600, kernel_name="python3",
                       resources={"metadata": {"path": str(ROOT)}}).execute(env=environment)
        record["status"] = "ANALYTIC_PASS_NATIVE_UNEXECUTED"
    except BaseException as failure:
        error = failure
        record.update(status="FAIL", failure=repr(failure))
    finally:
        completed_path = output / "completed_learner.ipynb"
        nbformat.write(notebook, completed_path)
        after = {path.name: digest(path) for path in sources}
        record.update(elapsed_seconds=time.perf_counter()-start,
                      changed_sources=[name for name in before if before[name] != after[name]],
                      cells=len(notebook.cells), code_cells=sum(c.cell_type=="code" for c in notebook.cells),
                      executed_code_cells=sum(c.cell_type=="code" and c.execution_count is not None for c in notebook.cells),
                      embedded_figures=sum("image/png" in out.get("data", {}) for c in notebook.cells if c.cell_type=="code" for out in c.outputs),
                      completed_learner_sha256=digest(completed_path),
                      figures={path.name: digest(path) for path in sorted((output / "figures").glob("*")) if path.is_file()})
        if record["changed_sources"] or (not error and record["embedded_figures"] != 9):
            record.update(status="FAIL", failure="Source changed or nine analytic figures were not rendered.")
        (output / "execution.json").write_text(json.dumps(record, indent=2)+"\n")
        print(json.dumps({key:value for key,value in record.items() if key not in {"source_inputs", "figures"}}, indent=2), flush=True)
    if error:
        raise error
    if record["status"] != "ANALYTIC_PASS_NATIVE_UNEXECUTED":
        raise AssertionError(record["failure"])
    return record


def render_review(output):
    """Render actual executed demonstrations while preserving the exercise stubs."""
    import nbformat
    from nbconvert import HTMLExporter

    output = Path(output).resolve()
    record = json.loads((output / "execution.json").read_text())
    if record["status"] != "ANALYTIC_PASS_NATIVE_UNEXECUTED":
        raise ValueError("A passing analytic execution is required before rendering.")
    for name, expected in record["source_inputs"].items():
        if digest(HERE / name) != expected:
            raise ValueError("Source changed after execution: "+name)
    student, solved = build_notebooks()
    executed = nbformat.read(output / "completed_learner.ipynb", as_version=4)
    assert [(c.cell_type, c.source) for c in executed.cells] == [(c["cell_type"], c["source"]) for c in solved["cells"]]
    by_id = {cell.id: cell for cell in executed.cells}
    for cell in student["cells"]:
        if cell["cell_type"] == "code" and not cell.get("metadata", {}).get("arena", {}).get("exercise_functions"):
            observed = by_id[cell["id"]]
            assert cell["source"] == observed.source
            cell["outputs"] = copy.deepcopy(observed.outputs)
            cell["execution_count"] = observed.execution_count
    review = output / "html_review"
    review.mkdir(exist_ok=False)
    hashes = {}
    for name, notebook in (("exercises", nbformat.reads(json.dumps(student), as_version=4)), ("solutions", executed)):
        nbformat.write(notebook, review / f"DRAFT_{STEM}_{name}.ipynb")
        html, _ = HTMLExporter().from_notebook_node(notebook)
        path = review / f"{name}.html"
        path.write_text(html)
        hashes[path.name] = digest(path)
    (review / "render.json").write_text(json.dumps({"status": "RENDERED_REQUIRES_VISUAL_REVIEW", "html": hashes,
        "exercise_stubs_preserved": True, "recorded_demonstrations_only": True,
        "native_execution": "UNEXECUTED", "independent_review": False}, indent=2)+"\n")
    return hashes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft-output", type=Path, required=True)
    parser.add_argument("--execute-analytic", action="store_true")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    if args.execute_analytic:
        execute_analytic(args.draft_output)
    else:
        write_drafts(args.draft_output)
    if args.render:
        render_review(args.draft_output)
    print(args.draft_output.resolve(), flush=True)


if __name__ == "__main__":
    main()
