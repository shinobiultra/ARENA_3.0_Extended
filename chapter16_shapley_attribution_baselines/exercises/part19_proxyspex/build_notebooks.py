"""Build, execute and render unpublished 16.19 drafts from their own visible answers."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import textwrap
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
                "arena": {"section": "16.19", "status": "draft_requires_own_answer_execution_and_review",
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


def native_replay_receipt(output, fixture_hash):
    """Check provenance and numerical reproduction, without imposing a model-quality threshold."""
    output = Path(output)
    binding = json.loads((output / "native_replay_binding.json").read_text())
    if binding["fixture_sha256"] != fixture_hash:
        raise ValueError("The notebook loaded another native fixture.")
    receipts = [json.loads((output / name).read_text()) for name in ("native_default_replay.json", "native_play_replay.json")]
    max_delta, max_prediction_delta, pointwise_checks = 0., 0., 0
    expected_cases = ((0, 0, 1824), (0, 0, 456))
    expected_stages = {"proxy", "extracted", "truncated", "refined", "constant", "additive"}
    for replay, expected_case in zip(receipts, expected_cases, strict=True):
        observed_case = tuple(replay.get(key) for key in ("prompt_index", "seed", "budget"))
        if replay.get("method") != "proxyspex" or observed_case != expected_case:
            raise ValueError("The receipt does not cover the two declared native replay cases.")
        if (replay["status"] != "COMPLETE" or replay["new_model_queries"] != 0
                or replay["new_physical_model_rows"] != 0 or replay["audit_labels_received_by_fit"]):
            raise ValueError("The learner replay did not preserve the zero-query training/audit boundary.")
        if len(replay["comparisons"]) != 2 or {row.get("condition") for row in replay["comparisons"]} != {"real", "shuffled"}:
            raise ValueError("Require both real and shuffled pointwise comparisons.")
        for comparison in replay["comparisons"]:
            pointwise = comparison.get("prediction_allclose", {})
            deltas = comparison.get("maximum_prediction_absolute_delta", {})
            if (set(pointwise) != expected_stages or set(deltas) != expected_stages
                    or set(comparison["student_metrics"]) != expected_stages
                    or comparison.get("prediction_tolerance") != {"atol": 1e-8, "rtol": 1e-7}
                    or any(value is not True for value in pointwise.values())
                    or any(not math.isfinite(value) or value < 0 for value in deltas.values())):
                raise ValueError("Require passing pointwise prediction checks for every stage and baseline at the declared tolerance.")
            max_prediction_delta = max(max_prediction_delta, *deltas.values())
            pointwise_checks += len(pointwise)
            for stage, metrics in comparison["student_metrics"].items():
                for name, actual in metrics.items():
                    expected = comparison["recorded_metrics"][stage][name]
                    if actual is None or expected is None:
                        if actual != expected:
                            raise ValueError("Undefined audit metrics differ from the saved result.")
                        continue
                    max_delta = max(max_delta, abs(actual - expected))
                    if not math.isclose(actual, expected, rel_tol=1e-7, abs_tol=1e-8):
                        raise ValueError("The student's reconstructed predictions differ from the recorded calculation.")
    return {"native_execution_sha256": binding["native_execution_sha256"],
            "fresh_reproduction_sha256": binding["fresh_reproduction_sha256"],
            "native_replay_binding_sha256": digest(output / "native_replay_binding.json"),
            "replay_receipts": {name: digest(output / name) for name in ("native_default_replay.json", "native_play_replay.json")},
            "maximum_reproduced_metric_absolute_delta": max_delta,
            "maximum_reproduced_prediction_absolute_delta": max_prediction_delta,
            "pointwise_prediction_checks": pointwise_checks,
            "prediction_tolerance": {"atol": 1e-8, "rtol": 1e-7}}


def execute_learner(output, *, native=False, fixture=None):
    """Execute own answers; native mode refits saved responses and never queries GPT-2."""
    import nbformat
    from nbclient import NotebookClient

    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new empty draft execution directory to preserve previous outcomes.")
    output.mkdir(parents=True, exist_ok=True)
    student, _ = write_drafts(output)
    notebook = nbformat.reads(json.dumps(complete_from_own_answers(student)), as_version=4)
    if not native:
        for cell in notebook.cells:
            if cell.metadata.get("arena", {}).get("calling_cell") in {"native_study", "native_references", "native_play"}:
                cell.metadata["tags"] = ["skip-execution"]
    sources = [HERE/name for name in ("proxy_methods.py", "proxy_methods_tests.py", "proxy_learner_tests.py",
                                      "proxy_plots.py", "lesson.md", "lesson_cells.py", "build_notebooks.py")]
    if native:
        sources.append(HERE.parent / "part18_spex/native_learner.py")
    before = {str(path.relative_to(ROOT)): digest(path) for path in sources}
    for path in sources:
        destination = output / "source_snapshot" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    environment = {**os.environ, "PATH": str(Path(sys.executable).parent)+os.pathsep+os.environ.get("PATH", ""),
                   "ARENA_PROXYSPEX_OUTPUT_DIRECTORY": str(output / "figures"),
                   "ARENA_LEARNER_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
    if fixture is not None:
        environment["ARENA_NATIVE_QUERY_FIXTURE"] = str(Path(fixture).resolve())
    fixture_path = Path(environment.get("ARENA_NATIVE_QUERY_FIXTURE", str(HERE.parent / "part18_spex/expected_outputs/native_query_fixture.json.gz")))
    record = {"status": "RUNNING_LEARNER_DRAFT", "completed_from_own_adjacent_answers": True,
              "source_inputs": before, "method_sha256": FROZEN_METHOD_SHA256,
              "native_execution": "REPLAY_SAVED_RESPONSES" if native else "UNEXECUTED_ANALYTIC_ONLY",
              "model_queries": 0, "scope": "complete_learner" if native else "analytic_only"}
    if native:
        record["portable_replay"] = {"status": "RUNNING", "fixture_sha256": digest(fixture_path),
                                     "no_full_study_cache_required": True}
    start = time.perf_counter()
    error = None
    try:
        NotebookClient(notebook, timeout=600, kernel_name="python3",
                       resources={"metadata": {"path": str(ROOT)}}).execute(env=environment)
        record["status"] = "passed" if native else "ANALYTIC_PASS_NATIVE_UNEXECUTED"
        if native:
            record.update(native_replay_receipt(output, record["portable_replay"]["fixture_sha256"]))
            record["portable_replay"]["status"] = "PASS"
    except BaseException as failure:
        error = failure
        record.update(status="FAIL", failure=repr(failure))
    finally:
        completed_path = output / "completed_learner.ipynb"
        nbformat.write(notebook, completed_path)
        after = {str(path.relative_to(ROOT)): digest(path) for path in sources}
        record.update(elapsed_seconds=time.perf_counter()-start, sources_unchanged=before == after,
                      changed_sources=[name for name in before if before[name] != after[name]],
                      cells=len(notebook.cells), code_cells=sum(c.cell_type=="code" for c in notebook.cells),
                      executed_code_cells=sum(c.cell_type=="code" and c.execution_count is not None for c in notebook.cells),
                      embedded_figures=sum("image/png" in out.get("data", {}) for c in notebook.cells if c.cell_type=="code" for out in c.outputs),
                      completed_learner_sha256=digest(completed_path),
                      figures={path.name: digest(path) for path in sorted((output / "figures").glob("*.png"))},
                      figure_files={path.name: digest(path) for path in sorted((output / "figures").glob("*")) if path.is_file()})
        record.update(total_code_cells=record["code_cells"], executed_notebook=str(completed_path),
                      notebook_sha256=record["completed_learner_sha256"])
        expected_figures = 13 if native else 9
        skipped = 0 if native else 3
        if record["changed_sources"] or (not error and (record["embedded_figures"] != expected_figures
                or record["executed_code_cells"] != record["code_cells"] - skipped)):
            record.update(status="FAIL", failure="Source changed or expected code cells/figures were not executed.")
        (output / "execution.json").write_text(json.dumps(record, indent=2)+"\n")
        print(json.dumps({key:value for key,value in record.items() if key not in {"source_inputs", "figures"}}, indent=2), flush=True)
    if error:
        raise error
    if record["status"] not in {"passed", "ANALYTIC_PASS_NATIVE_UNEXECUTED"}:
        raise AssertionError(record["failure"])
    return record


def render_review(output):
    """Render actual executed demonstrations while preserving the exercise stubs."""
    import nbformat
    from nbconvert import HTMLExporter

    output = Path(output).resolve()
    record = json.loads((output / "execution.json").read_text())
    if record["status"] not in {"passed", "ANALYTIC_PASS_NATIVE_UNEXECUTED"}:
        raise ValueError("A passing own-answer execution is required before rendering.")
    for name, expected in record["source_inputs"].items():
        if digest(ROOT / name) != expected:
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
    page = export_page(output, student)
    (review / "render.json").write_text(json.dumps({"status": "RENDERED_REQUIRES_VISUAL_REVIEW", "html": hashes,
        "exercise_stubs_preserved": True, "recorded_demonstrations_only": True,
        "native_execution": record["native_execution"], "independent_review": False, "page": page}, indent=2)+"\n")
    return hashes


def export_page(output, notebook):
    """Export the actual exercised page and byte-identical saved figures for later staging."""
    output = Path(output)
    directory = output / "publication_inputs"
    directory.mkdir(exist_ok=False)
    assets = directory / "assets"
    assets.mkdir()
    figures = sorted((output / "figures").glob("*.png"))
    figure_index, sections, hashes = 0, [], {}
    for cell in notebook["cells"]:
        if cell["cell_type"] == "markdown":
            sections.append(cell["source"])
            continue
        sections.append("```python\n" + cell["source"] + "\n```")
        for displayed in cell.get("outputs", []):
            data = displayed.get("data", {})
            if "image/png" in data:
                if figure_index >= len(figures):
                    raise ValueError("A displayed figure has no saved scientific PNG.")
                source = figures[figure_index]
                figure_index += 1
                destination = assets / ("1619_" + source.name)
                shutil.copy2(source, destination)
                hashes[destination.name] = digest(destination)
                if hashes[destination.name] != digest(source):
                    raise ValueError("Figure bytes changed during page export.")
                caption = source.stem.split("_", 1)[1].replace("_", " ")
                sections.append(f"![Actual learner result: {caption}](../assets/{destination.name})")
            elif "text/markdown" in data:
                sections.append(data["text/markdown"])
            else:
                text = displayed.get("text", data.get("text/plain", ""))
                if text.strip():
                    wrapped = "\n".join(textwrap.fill(line, width=140, replace_whitespace=False) for line in text.splitlines())
                    sections.append("```text\n" + wrapped + "\n```")
    if figure_index != len(figures):
        raise ValueError("A saved figure was not displayed in the learner notebook.")
    page = directory / "19_[16.19]_Explain_the_Proxy_Check_the_Model.md"
    page.write_text("\n\n".join(sections) + "\n")
    return {"path": str(page), "sha256": digest(page), "assets": hashes,
            "status": "UNPUBLISHED_EXACT_LEARNER_INPUTS"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft-output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute-analytic", action="store_true")
    mode.add_argument("--execute-native", action="store_true")
    parser.add_argument("--native-fixture", type=Path)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    if args.execute_analytic or args.execute_native:
        execute_learner(args.draft_output, native=args.execute_native, fixture=args.native_fixture)
    else:
        write_drafts(args.draft_output)
    if args.render:
        render_review(args.draft_output)
    print(args.draft_output.resolve(), flush=True)


if __name__ == "__main__":
    main()
