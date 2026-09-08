"""Build unpublished lessons with visible methods and portable native replay.

No default writes to canonical notebook/page locations. Native cells refit from
the portable score fixture and never start a model or a cold query experiment.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import html
import json
import os
from pathlib import Path
import re
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from chapter16_shapley_attribution_baselines.exercises.part18_spex.lesson_cells import CELLS, EXERCISES

STEM = "16.18_Find_the_Interactions_Without_Trying_Every_Subset"
TOKEN = re.compile(r"<!-- (code|exercise|solution): ([\w,]+) -->")


def function_sources():
    """Include decorators so the visible own answer is the actual function."""
    output = {}
    for filename in ("spex_methods.py", "native_data.py"):
        text = (HERE / filename).read_text()
        lines = text.splitlines()
        for node in ast.parse(text).body:
            if isinstance(node, ast.FunctionDef):
                first = min([node.lineno] + [d.lineno for d in node.decorator_list])
                output[node.name] = "\n".join(lines[first-1:node.end_lineno])
    return output


def _stub(source):
    node = ast.parse(source).body[0]
    has_doc = isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str)
    doc = [node.body[0]] if has_doc else []
    node.body = doc + [ast.Raise(exc=ast.Call(func=ast.Name(id="NotImplementedError", ctx=ast.Load()),
                                             args=[ast.Constant(value=f"Implement {node.name}")], keywords=[]), cause=None)]
    return ast.unparse(ast.fix_missing_locations(node))


def _cell(kind, source, **arena):
    identifier = hashlib.sha256((kind + "\n" + source).encode()).hexdigest()[:12]
    cell = {"cell_type": kind, "id": identifier, "metadata": {"arena": arena} if arena else {}, "source": source}
    if kind == "code":
        cell.update(execution_count=None, outputs=[])
    return cell


def _prose(source):
    """Keep details content inside its actual HTML element in nbconvert too.

    Blank-line-separated raw HTML plus Markdown was auto-closed by this
    environment's renderer. Render the inner Markdown explicitly and emit a
    single HTML block; normal notebook prose remains Markdown.
    """
    from nbconvert.filters import markdown2html
    def replace(match):
        summary, body = match.groups()
        rendered = str(markdown2html(body.strip())).replace("\n", " ")
        return f"<details><summary>{summary}</summary><div>{rendered}</div></details>"
    return re.sub(r"<details>\s*<summary>(.*?)</summary>(.*?)</details>", replace, source, flags=re.S)


def _solution(answer):
    # Entities keep logical code newlines without blank lines breaking the raw
    # HTML block. complete_from_own_answers decodes this same visible answer.
    escaped = html.escape(answer).replace("\n", "&#10;")
    return f'<details><summary>Solution</summary><div class="highlight"><pre><code>{escaped}</code></pre></div></details>'


def _exercise_test(key):
    return EXERCISES[key]["tests"] + f'\nprint("Exercise {key}: all immediate tests passed!")'


def build_notebooks():
    text = (HERE / "lesson.md").read_text()
    sources = function_sources()
    student, solved, position = [], [], 0
    found_exercises, found_solutions = [], []
    for token in TOKEN.finditer(text):
        prose = text[position:token.start()].strip()
        if prose:
            student.append(_cell("markdown", _prose(prose))); solved.append(_cell("markdown", _prose(prose)))
        kind, key = token.groups()
        if kind == "code":
            cell = _cell("code", CELLS[key].strip(), calling_cell=key,
                         native_execution=key.startswith("native_"))
            student.append(copy.deepcopy(cell)); solved.append(cell)
        else:
            functions = EXERCISES[key]["functions"]
            answer = "\n\n".join(sources[name] for name in functions)
            if kind == "exercise":
                found_exercises.append(key)
                sc = _cell("code", "\n\n".join(_stub(sources[name]) for name in functions) + "\n\n" + _exercise_test(key),
                           exercise_key=key, exercise_functions=list(functions))
                ac = _cell("code", answer + "\n\n" + _exercise_test(key), exercise_key=key, exercise_functions=list(functions))
                sc["id"] = ac["id"] = hashlib.sha256(("exercise:" + key).encode()).hexdigest()[:12]
                student.append(sc); solved.append(ac)
            else:
                found_solutions.append(key)
                student.append(_cell("markdown", _solution(answer), solution_for=key))
        position = token.end()
    tail = text[position:].strip()
    if tail:
        student.append(_cell("markdown", _prose(tail))); solved.append(_cell("markdown", _prose(tail)))
    assert found_exercises == found_solutions == list(EXERCISES), "Every exercise needs its adjacent own solution"
    metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python", "version": "3.14"},
                "arena": {"section": "16.18", "status": "unpublished_integration_draft",
                          "native_status": "PORTABLE_REPLAY_REQUIRED", "original_chapters_modified": False,
                          "exercise_groups": len(EXERCISES), "student_functions": sum(len(r["functions"]) for r in EXERCISES.values())}}
    def make(cells):
        return {"nbformat": 4, "nbformat_minor": 5, "metadata": copy.deepcopy(metadata), "cells": cells}
    return make(student), make(solved)


def complete_from_own_answers(student):
    """Fill stubs only from the student's adjacent solution dropdowns."""
    notebook = copy.deepcopy(student)
    answers = {}
    for cell in notebook["cells"]:
        key = cell.get("metadata", {}).get("arena", {}).get("solution_for")
        if key:
            answers[key] = html.unescape(re.search(r"<code>(.*?)</code>", cell["source"], re.S).group(1))
    filled = []
    for cell in notebook["cells"]:
        info = cell.get("metadata", {}).get("arena", {})
        if info.get("solution_for"):
            continue
        if info.get("exercise_key"):
            key = info["exercise_key"]
            cell["source"] = answers[key] + "\n\n" + _exercise_test(key)
        filled.append(cell)
    notebook["cells"] = filled
    _, solved = build_notebooks()
    assert [(c["cell_type"], c["source"]) for c in filled] == [(c["cell_type"], c["source"]) for c in solved["cells"]]
    return notebook


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def execute_own_answers(student, output_dir, *, native_fixture=None):
    """Use a real kernel bound to this Python, with source/outputs retained."""
    import nbformat
    from nbclient import NotebookClient
    output_dir = Path(output_dir).resolve()
    notebook = nbformat.from_dict(complete_from_own_answers(student))
    kernel_name = "arena-spex-analytic"
    jupyter_root = output_dir / "jupyter"
    kernel_dir = jupyter_root / "kernels" / kernel_name
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / "kernel.json").write_text(json.dumps({"argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "ARENA SPEX analytic draft", "language": "python"}))
    old = {key: os.environ.get(key) for key in ("JUPYTER_PATH", "ARENA_SPEX_OUTPUT_DIRECTORY", "MPLBACKEND",
           "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "ARENA_NATIVE_QUERY_FIXTURE")}
    os.environ["JUPYTER_PATH"] = str(jupyter_root) + (os.pathsep + old["JUPYTER_PATH"] if old["JUPYTER_PATH"] else "")
    os.environ["ARENA_SPEX_OUTPUT_DIRECTORY"] = str(output_dir / "figures")
    os.environ["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"
    os.environ["OPENBLAS_NUM_THREADS"] = os.environ["OMP_NUM_THREADS"] = "2"
    if native_fixture is not None:
        os.environ["ARENA_NATIVE_QUERY_FIXTURE"] = str(Path(native_fixture).resolve())
    started = time.time()
    sources = ["spex_methods.py", "spex_coding.py", "spex_methods_tests.py", "native_data.py",
               "native_data_tests.py", "native_learner.py", "native_learner_tests.py", "native_query_plan.py",
               "native_study.py", "lesson.md", "lesson_cells.py", "lesson_tests.py", "spex_plots.py", "build_notebooks.py"]
    source_dir = output_dir / "sources"
    source_dir.mkdir(exist_ok=True)
    for name in sources:
        (source_dir / name).write_bytes((HERE / name).read_bytes())
    before_hashes = {name: _sha(source_dir / name) for name in sources}
    status, failure = "passed", None
    path = output_dir / f"DRAFT_{STEM}_own_answer_executed.ipynb"
    try:
        NotebookClient(notebook, kernel_name=kernel_name, timeout=900,
                       resources={"metadata": {"path": str(ROOT)}}, record_timing=True).execute()
    except Exception as exc:
        status, failure = "failed", repr(exc)
        raise
    finally:
        nbformat.write(notebook, path)
        after_hashes = {name: _sha(HERE / name) for name in sources}
        if after_hashes != before_hashes and status == "passed":
            status, failure = "source_changed", "A source changed during execution; compare the retained before/after hashes."
        binding_path = output_dir / "native_replay_binding.json"
        native_binding = json.loads(binding_path.read_text()) if binding_path.exists() else {}
        replay_directories = [output_dir / "figures" / name for name in ("native_default_replay", "native_play_replay")]
        replay_complete = all((path / "hashes.json").exists() for path in replay_directories)
        receipt = {"status": status, "scope": "own-answer analytic and portable native replay execution",
                   "native_status": "PORTABLE_REPLAY_COMPLETE" if replay_complete and status == "passed" else "INCOMPLETE",
                   "python": sys.executable, "elapsed_seconds": time.time() - started,
                   "source_sha256": before_hashes, "source_sha256_after": after_hashes,
                   "sources_unchanged": before_hashes == after_hashes,
                   "executed_notebook": str(path), "notebook_sha256": _sha(path),
                   "executed_code_cells": sum(c["cell_type"] == "code" and c.get("execution_count") is not None for c in notebook.cells),
                   "total_code_cells": sum(c["cell_type"] == "code" for c in notebook.cells),
                   "figures": {p.name: _sha(p) for p in sorted((output_dir / "figures").glob("*.png"))},
                   "failure": failure,
                   "native_execution_sha256": native_binding.get("native_execution_sha256"),
                   "fresh_reproduction_sha256": native_binding.get("fresh_reproduction_sha256"),
                   "portable_replay": {"status": "PASS" if replay_complete and status == "passed" else "INCOMPLETE",
                                       "fixture_sha256": native_binding.get("fixture_sha256"),
                                       "no_full_study_cache_required": True,
                                       "new_model_queries": 0,
                                       "replay_records": [str(path) for path in replay_directories]}}
        (output_dir / "own_answer_execution.json").write_text(json.dumps(receipt, indent=2) + "\n")
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return notebook, receipt


def render_html(notebook, path):
    import nbformat
    from nbconvert import HTMLExporter
    exporter = HTMLExporter(template_name="lab")
    exporter.embed_images = True
    body, _ = exporter.from_notebook_node(nbformat.from_dict(notebook))
    Path(path).write_text(body)
    return Path(path)


def render_instruction_markdown(student, executed, path):
    """Use the student's actual prose/stubs/solutions with executed PNG results.

    This is a publication input only; the builder never writes the canonical
    instruction page. The publisher installs these exact bytes and named assets.
    """
    executed_by_key = {cell.get("metadata", {}).get("arena", {}).get("calling_cell"): cell
                       for cell in executed["cells"] if cell["cell_type"] == "code"}
    blocks = []
    for cell in student["cells"]:
        if cell["cell_type"] == "markdown":
            blocks.append(cell["source"])
            continue
        blocks.append("```python\n" + cell["source"] + "\n```")
        key = cell.get("metadata", {}).get("arena", {}).get("calling_cell")
        completed = executed_by_key.get(key)
        if completed is None:
            continue
        figures = re.findall(r'show_figure\([\s\S]*?,\s*"([0-9][\w]+)"\s*\)', cell["source"])
        for name in figures:
            blocks.append(f"![{name.replace('_', ' ')}](../assets/1618_{name}.png)")
        for output in completed.get("outputs", []):
            if output.get("output_type") == "stream":
                blocks.append("```text\n" + output["text"].rstrip() + "\n```")
            elif "text/html" in output.get("data", {}):
                blocks.append(output["data"]["text/html"])
    Path(path).write_text("\n\n".join(blocks) + "\n")
    return Path(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft-output", type=Path, required=True)
    parser.add_argument("--execute-own-answers", action="store_true")
    parser.add_argument("--render-html", action="store_true")
    parser.add_argument("--native-fixture", type=Path)
    args = parser.parse_args()
    output = args.draft_output.resolve()
    if output == HERE or HERE in output.parents:
        raise ValueError("Draft artifacts must be written outside the section's publication directory")
    output.mkdir(parents=True, exist_ok=True)
    student, solved = build_notebooks()
    complete_from_own_answers(student)
    for name, notebook in (("exercises", student), ("solutions", solved)):
        path = output / f"DRAFT_{STEM}_{name}.ipynb"
        path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
        if args.render_html:
            render_html(notebook, output / f"DRAFT_{STEM}_{name}.html")
        print(path, flush=True)
    if args.execute_own_answers:
        notebook, receipt = execute_own_answers(student, output, native_fixture=args.native_fixture)
        if args.render_html:
            render_html(notebook, output / f"DRAFT_{STEM}_own_answer_executed.html")
        render_instruction_markdown(student, notebook, output / f"DRAFT_{STEM}_instructions.md")
        print(json.dumps({key: receipt[key] for key in ("status", "native_status", "executed_code_cells", "total_code_cells", "elapsed_seconds")}), flush=True)


if __name__ == "__main__":
    main()
