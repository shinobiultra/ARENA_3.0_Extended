"""Complete 12.1 from its own dropdowns and execute every cell in a fresh kernel.

Use ARENA_CLIP_CPU_ONLY=1 to explicitly skip pretrained checkpoints. Both tiny
objectives and all three seeds still run. The default uses CUDA when available.
"""
from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from chapter12_vlm_interpretability.exercises.part1_clip_siglip_vlm_controls.build_notebooks import (
    HERE, STEM, complete_learner,
)


def main() -> None:
    learner = json.loads((HERE / f"{STEM}_exercises.ipynb").read_text())
    completed = nbformat.reads(json.dumps(complete_learner(learner)), as_version=4)
    solved = nbformat.read(HERE / f"{STEM}_solutions.ipynb", as_version=4)
    for actual, expected in zip(completed.cells, solved.cells, strict=True):
        if actual.cell_type == "code":
            assert ast.dump(ast.parse(actual.source)) == ast.dump(ast.parse(expected.source))

    def progress(cell, cell_index, **_):
        if cell.cell_type == "code":
            print(f"Executing cell {cell_index}: {cell.source.splitlines()[0]}", flush=True)

    client = NotebookClient(completed, timeout=1800, kernel_name="python3",
                            resources={"metadata": {"path": str(ROOT)}}, on_cell_start=progress)
    try:
        client.execute()
    finally:
        nbformat.write(completed, "/tmp/clip_completed_learner.ipynb")
    for actual, expected in zip(completed.cells, solved.cells, strict=True):
        if actual.cell_type == "code":
            expected.outputs = actual.outputs
            expected.execution_count = actual.execution_count
    nbformat.write(solved, HERE / f"{STEM}_solutions.ipynb")
    manifest = {
        "completed_from_own_answers": True,
        "code_cells_executed": sum(cell.cell_type == "code" for cell in completed.cells),
        "files": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (
            HERE / "lesson.md", HERE / "pixel_clip.py", HERE / "pixel_tests.py",
            HERE / "typography.py", HERE / f"{STEM}_exercises.ipynb",
            HERE / f"{STEM}_solutions.ipynb",
        )},
    }
    (HERE / "expected_outputs" / "learner_execution.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved {manifest['code_cells_executed']} executed cells and source hashes", flush=True)


if __name__ == "__main__":
    main()
