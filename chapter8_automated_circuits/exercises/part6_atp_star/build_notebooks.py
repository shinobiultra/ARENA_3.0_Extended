"""Render the readable AtP* lesson and canonical method definitions together."""
from __future__ import annotations

import ast
from copy import deepcopy
from pathlib import Path
import re
import sys

import nbformat

SECTION = Path(__file__).resolve().parent
ROOT = SECTION.parents[2]
PAGE = ROOT / "chapter8_automated_circuits/instructions/pages/06_[8.6]_AtP-Star_from_Scratch.md"
SOURCE = (SECTION / "solutions.py").read_text()
SOURCE_LINES = SOURCE.splitlines(keepends=True)
DEFINITIONS = {
    node.name: node for node in ast.parse(SOURCE).body
    if isinstance(node, (ast.FunctionDef, ast.ClassDef))
}


def definition(name: str, *, stub: bool = False) -> str:
    node = DEFINITIONS[name]
    if stub:
        node = deepcopy(node)
        doc = node.body[:1] if ast.get_docstring(node) is not None else []
        node.body = doc + [ast.Raise(
            exc=ast.Call(func=ast.Name(id="NotImplementedError", ctx=ast.Load()),
                         args=[ast.Constant(value="Implement this method, then run the checks below.")], keywords=[]),
            cause=None,
        )]
        return ast.unparse(ast.fix_missing_locations(node))
    start = min([node.lineno, *[item.lineno for item in node.decorator_list]]) - 1
    return "".join(SOURCE_LINES[start:node.end_lineno]).rstrip()


def definitions(names: str, *, stub: bool = False) -> str:
    return "\n\n\n".join(definition(name.strip(), stub=stub) for name in names.split(","))


def build_cells(*, solved: bool):
    cells = []
    lesson = (SECTION / "lesson.md").read_text()
    chunks = re.split(r"<!-- CELL: (markdown|code) -->\n", lesson)
    for kind, source in zip(chunks[1::2], chunks[2::2], strict=True):
        source = source.strip()
        if kind == "markdown":
            source = re.sub(r"\{\{solution:([^}]+)\}\}",
                            lambda match: "```python\n" + definitions(match[1]) + "\n```", source)
            for block in re.findall(r"```python\n(.*?)```", source, re.S):
                ast.parse(block)
            cell = nbformat.v4.new_markdown_cell(source)
        else:
            source = re.sub(r"^# (EXERCISE|PROVIDED): (.+)$",
                            lambda match: definitions(match[2], stub=match[1] == "EXERCISE" and not solved),
                            source, flags=re.M)
            ast.parse(source)
            cell = nbformat.v4.new_code_cell(source)
        cell.id = f"atp-star-{len(cells):03d}"
        cells.append(cell)
    return cells


def main():
    for solved, suffix in ((False, "exercises"), (True, "solutions")):
        notebook = nbformat.v4.new_notebook(cells=build_cells(solved=solved), metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        })
        nbformat.write(notebook, SECTION / f"8.6_AtP-Star_from_Scratch_{suffix}.ipynb")
    page_parts = [cell.source if cell.cell_type == "markdown" else "```python\n" + cell.source + "\n```"
                  for cell in build_cells(solved=False)]
    PAGE.write_text("\n\n".join(page_parts).replace("../../instructions/assets/", "../assets/") + "\n")


def finalize_report():
    """Record actual CPU checks; a CPU report is never an accepted CUDA release."""
    import json
    import subprocess
    from datetime import datetime, timezone
    import yaml
    from jsonschema import Draft202012Validator

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from arena_ext.verification import build_report_input_manifest, current_git_commit
    from chapter8_automated_circuits.exercises.part6_atp_star import solutions

    subprocess.run([sys.executable, "-m", "pytest", "-q", str(SECTION / "tests.py"),
                    "-k", "not candidate_report_requires_cuda_and_current_inputs"], cwd=ROOT, check=True)
    solved = nbformat.read(SECTION / "8.6_AtP-Star_from_Scratch_solutions.ipynb", as_version=4)
    code_cells = [cell for cell in solved.cells if cell.cell_type == "code"]
    if not all(cell.execution_count is not None for cell in code_cells):
        raise RuntimeError("Execute every completed learner cell before finalizing CPU evidence.")
    if any(output.output_type == "error" for cell in code_cells for output in cell.outputs):
        raise RuntimeError("The completed learner contains an execution error.")
    toy = solutions.run_toy_study()
    real = json.loads((SECTION / "expected_outputs/reference_metrics.json").read_text())
    if not toy["accepted"] or not real["accepted"] or real["device"] != "cpu":
        raise RuntimeError("The scoped numerical experiment must pass on the declared CPU path.")
    lock = yaml.safe_load((SECTION / "artifacts.lock.yml").read_text())
    report = json.loads((SECTION / "verification_report.json").read_text())
    for key in ("claim_scope", "models", "datasets", "evidence_level", "safety_notes"):
        report[key] = lock[key]
    report.update({
        "date_run": datetime.now(timezone.utc).isoformat(), "git_commit": current_git_commit(ROOT),
        "tests_passed": True, "metrics": {"toy": toy, "real_cpu": real},
        "cpu_evidence": {"status": "passed", "solution_notebook_executed": True,
                         "code_cells": len(code_cells), "error_outputs": 0},
        "accepted": False,
    })
    report["gpu_evidence"]["status"] = "not_run_pending_independent_review"
    report["gpu_evidence"]["uses_cuda"] = False
    report["known_failures"] = [
        "Independent review and strict CUDA evidence remain pending.",
        "The paper's subset-sampling false-negative diagnostic is not implemented.",
        "The same 24 grammar pairs are used for screening and verification; no held-out circuit generalization is established.",
        "Only final-position Q/K sites are included, not a complete circuit.",
    ]
    inputs = sorted(path for path in SECTION.rglob("*") if path.is_file()
                    and "__pycache__" not in path.parts and path.name != "verification_report.json")
    inputs += [PAGE, ROOT / "chapter8_automated_circuits/instructions/assets/atp_star_toy_signature.png",
               ROOT / "chapter8_automated_circuits/instructions/assets/atp_star_transformerlens_signature.png"]
    report["report_inputs"] = build_report_input_manifest(ROOT, inputs)
    schema = json.loads((SECTION / "verification_report.schema.json").read_text())
    Draft202012Validator(schema).validate(report)
    (SECTION / "verification_report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    if "--finalize" in sys.argv:
        finalize_report()
    else:
        main()
