"""Execute the readable 9.1 learner, completed solely from its own answers.

Partial scopes are explicitly recorded and never publish release artifacts.
Every full execution reruns the actual development and frozen final experiments.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from chapter9_alignment_interpretability.exercises.part1_refusal_directions_safe_steering.build_notebooks import (
    HERE, STEM, complete_learner, lesson_cells,
)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ProgressNotebookClient(NotebookClient):
    """Expose the lesson's own progress prints during long generation cells."""
    def process_message(self, msg, cell, cell_index):
        if msg["msg_type"] == "stream":
            print(msg["content"].get("text", ""), end="", flush=True)
        return super().process_message(msg, cell, cell_index)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("numerical", "full"), default="full")
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    output = args.output_directory.resolve()
    if args.publish and args.scope != "full":
        raise ValueError("Only a successful full learner execution may publish outputs")
    output.mkdir(parents=True, exist_ok=False)
    figures, runs = output / "figures", output / "experiments"
    figures.mkdir()
    tracked_sources = [HERE / "lesson.md", *sorted(HERE.glob("*.py"))]
    source_hashes = {str(path.relative_to(HERE)): file_hash(path) for path in tracked_sources}
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for path in tracked_sources:
        (snapshot / path.name).write_bytes(path.read_bytes())
    # Preserve the actual executed lesson even if later work edits the draft.
    (output / "lesson_snapshot.md").write_bytes((HERE / "lesson.md").read_bytes())
    learner = json.loads((HERE / f"{STEM}_exercises.ipynb").read_text())
    learner["cells"] = lesson_cells()
    completed = nbformat.reads(json.dumps(complete_learner(learner)), as_version=4)
    original_code = [cell.source for cell in completed.cells if cell.cell_type == "code"]
    # Only output paths are overridden; the method and experiment bodies stay intact.
    completed.cells[1].source += f"\nassets_dir = Path({str(figures)!r})\n"

    def progress(cell, cell_index, **_):
        if cell.cell_type == "code":
            print(f"Executing learner cell {cell_index}: {cell.source.splitlines()[0]}", flush=True)

    environment = {**os.environ, "ARENA_REFUSAL_OUTPUT_DIRECTORY": str(runs),
                   "ARENA_REFUSAL_NUMERICAL_ONLY": "1" if args.scope == "numerical" else "0"}
    client = ProgressNotebookClient(completed, timeout=3600, kernel_name="python3",
                            resources={"metadata": {"path": str(ROOT)}}, on_cell_start=progress)
    manifest = {
        "status": "RUNNING", "scope": args.scope, "completed_from_own_answers": True,
        "fresh_experiments_required": True, "input_files": source_hashes,
        "output_directory_override": str(output), "full_release_validation": args.scope == "full",
    }
    started = time.perf_counter()
    try:
        client.execute(env=environment)
        manifest["status"] = "PASS"
    except BaseException as error:
        manifest.update(status="FAIL", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        nbformat.write(completed, output / "completed_learner.ipynb")
        current_names = {"lesson.md", *(path.name for path in HERE.glob("*.py"))}
        changed_sources = [name for name, digest in source_hashes.items()
                           if not (HERE / name).exists() or file_hash(HERE / name) != digest]
        changed_sources.extend(sorted(current_names - source_hashes.keys()))
        manifest.update(
            elapsed_seconds=time.perf_counter() - started,
            code_cells_executed=sum(cell.cell_type == "code" and cell.execution_count is not None
                                    for cell in completed.cells),
            figures={path.name: file_hash(path) for path in sorted(figures.glob("*.png"))},
            sources_changed_during_run=changed_sources,
        )
        (output / "execution.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.publish:
        if manifest["sources_changed_during_run"]:
            raise ValueError("Source changed during execution; do not publish mixed-source evidence")
        published_learner = nbformat.read(HERE / f"{STEM}_exercises.ipynb", as_version=4)
        canonical_cells = [(cell["cell_type"], "".join(cell["source"])) for cell in learner["cells"]]
        if [(cell.cell_type, cell.source) for cell in published_learner.cells] != canonical_cells:
            raise ValueError("The published learner differs from the exact executed lesson")
        solved = nbformat.read(HERE / f"{STEM}_solutions.ipynb", as_version=4)
        expected_code = [cell.source for cell in solved.cells if cell.cell_type == "code"]
        if len(solved.cells) != len(completed.cells) or len(expected_code) != len(original_code):
            raise ValueError("Build the complete paired notebooks before publication")
        for actual, expected in zip(original_code, expected_code, strict=True):
            if ast.dump(ast.parse(actual)) != ast.dump(ast.parse(expected)):
                raise ValueError("Published solutions differ from the executed learner answers")
        evidence = runs / "completed_refusal_study.json.gz"
        if not evidence.exists():
            raise ValueError("The full learner must save its raw experiments before publication")
        import gzip
        from chapter9_alignment_interpretability.exercises.part1_refusal_directions_safe_steering.direction_evidence import (
            REQUIRED_FIGURES, validate_refusal_measurements,
        )
        if set(manifest["figures"]) != REQUIRED_FIGURES:
            raise ValueError("Generate all six actual learner figures before publication")
        manifest["measurement_audit"] = validate_refusal_measurements(json.loads(gzip.decompress(evidence.read_bytes())))
        for actual, expected in zip(completed.cells, solved.cells, strict=True):
            if actual.cell_type == "code":
                expected.outputs, expected.execution_count = actual.outputs, actual.execution_count
        nbformat.write(solved, HERE / f"{STEM}_solutions.ipynb")
        for path in figures.glob("*.png"):
            (HERE.parents[1] / "instructions" / "assets" / path.name).write_bytes(path.read_bytes())
        target = HERE / "expected_outputs" / "completed_refusal_study.json.gz"
        target.write_bytes(evidence.read_bytes())
        manifest["published_evidence_sha256"] = file_hash(target)
        manifest["published_notebook_sha256"] = file_hash(HERE / f"{STEM}_solutions.ipynb")
        manifest["published_exercises_sha256"] = file_hash(HERE / f"{STEM}_exercises.ipynb")
        (HERE / "expected_outputs" / "learner_execution.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: manifest[key] for key in ("status", "scope", "code_cells_executed", "elapsed_seconds")}), flush=True)


if __name__ == "__main__":
    main()
