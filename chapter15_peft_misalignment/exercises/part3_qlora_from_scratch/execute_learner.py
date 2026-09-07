"""Execute the real learner, completed solely from its own Solution dropdowns.

Run from the repository root with ``uv run python <this file>``. CUDA is used
when available; set ARENA_QLORA_CPU_ONLY=1 for the explicitly limited CPU path.
The completed temporary notebook is retained on failure for diagnosis.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from chapter15_peft_misalignment.exercises.part3_qlora_from_scratch.build_notebooks import complete_learner

HERE = Path(__file__).resolve().parent


def main() -> None:
    learner_path = HERE / "15.3_QLoRA_from_Scratch_exercises.ipynb"
    solution_path = HERE / "15.3_QLoRA_from_Scratch_solutions.ipynb"
    completed = nbformat.reads(json.dumps(complete_learner(json.loads(learner_path.read_text()))), as_version=4)
    solved = nbformat.read(solution_path, as_version=4)
    for actual, expected in zip(completed.cells, solved.cells, strict=True):
        if actual.cell_type == "code":
            assert ast.dump(ast.parse(actual.source)) == ast.dump(ast.parse(expected.source))

    def progress(cell: dict, cell_index: int, **_: object) -> None:
        if cell.cell_type == "code":
            print(f"Executing cell {cell_index}: {cell.source.splitlines()[0]}", flush=True)

    client = NotebookClient(completed, timeout=1800, kernel_name="python3",
                            resources={"metadata": {"path": str(ROOT)}}, on_cell_start=progress)
    try:
        with client.setup_kernel():
            for index, cell in enumerate(completed.cells):
                client.execute_cell(cell, index)
            capture = nbformat.v4.new_code_cell(
                "Path('/tmp/arena-qlora-learner-metrics.json').write_text(json.dumps("
                "{'cpu': cpu_result, 'cuda': globals().get('cuda_result'), "
                "'precision_diagnostic': precision_diagnostic}, indent=2))"
            )
            completed.cells.append(capture)
            client.execute_cell(capture, len(completed.cells) - 1)
            completed.cells.pop()
    finally:
        nbformat.write(completed, "/tmp/qlora_completed_learner.ipynb")
    for actual, expected in zip(completed.cells, solved.cells, strict=True):
        if actual.cell_type == "code":
            expected.outputs = actual.outputs
            expected.execution_count = actual.execution_count
    nbformat.write(solved, solution_path)
    print("Saved completed learner outputs and /tmp/arena-qlora-learner-metrics.json", flush=True)


if __name__ == "__main__":
    main()
