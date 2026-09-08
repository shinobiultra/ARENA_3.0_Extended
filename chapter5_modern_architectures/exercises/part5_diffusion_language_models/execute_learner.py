"""Execute the readable 5.5 learner, completed solely from its own answers.

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
from chapter5_modern_architectures.exercises.part5_diffusion_language_models.build_notebooks import (
    HERE, STEM, complete_learner, lesson_cells,
)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# Exact learner dependencies. Independent review/report code is not executed
# by the notebook and is deliberately outside this input inventory.
LEARNER_INPUTS = (
    "lesson.md", "build_notebooks.py", "execute_learner.py", "tests.py", "solutions.py",
    "text_data.py", "text_diffusion_methods.py", "text_diffusion_tests.py",
    "text_model.py", "text_model_tests.py", "text_training.py", "text_training_tests.py",
    "text_evaluation.py", "text_evaluation_tests.py", "text_plotting.py",
    "artifacts/diffusiongemma_vllm_probe.json",
)
REQUIRED_FIGURES = {
    "diffusion_copy_controls.png", "diffusion_copy_trajectory.png",
    "diffusion_text_training.png", "diffusion_text_task_controls.png",
    "diffusion_text_budget_quality.png", "diffusion_text_cost_quality.png",
    "diffusion_text_trajectory.png", "diffusion_text_entropy_commitment.png",
    "diffusion_text_story_gallery.png", "diffusion_text_schedule_play.png",
    "diffusion_text_arithmetic_gallery.png", "diffusion_text_temperature_blocks.png",
}


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
    args = parser.parse_args()
    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=False)
    figures, runs = output / "figures", output / "experiments"
    figures.mkdir()
    tracked_sources = [HERE / name for name in LEARNER_INPUTS]
    source_hashes = {str(path.relative_to(HERE)): file_hash(path) for path in tracked_sources}
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for path in tracked_sources:
        target = snapshot / path.relative_to(HERE)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
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

    environment = {**os.environ, "ARENA_DIFFUSION_OUTPUT_DIRECTORY": str(runs),
                   "ARENA_DIFFUSION_NUMERICAL_ONLY": "1" if args.scope == "numerical" else "0"}
    client = ProgressNotebookClient(completed, timeout=3600, kernel_name="python3",
                            resources={"metadata": {"path": str(ROOT)}}, on_cell_start=progress)
    manifest = {
        "status": "RUNNING", "scope": args.scope, "completed_from_own_answers": True,
        "fresh_experiments_required": True, "input_files": source_hashes,
        "output_directory_override": str(output), "full_learner_execution": args.scope == "full",
        "full_release_validation": False,
        "publication_status": "requires separate scientific validation and fresh CUDA check",
    }
    started = time.perf_counter()
    try:
        client.execute(env=environment)
        if args.scope == "full":
            if not (runs / "completed_text_study.json.gz").is_file():
                raise ValueError("The full learner did not save its actual text experiment")
            if {path.name for path in figures.glob("*.png")} != REQUIRED_FIGURES:
                raise ValueError("The full learner must generate all twelve actual figures")
        manifest["status"] = "PASS"
    except BaseException as error:
        manifest.update(status="FAIL", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        nbformat.write(completed, output / "completed_learner.ipynb")
        changed_sources = [name for name, digest in source_hashes.items()
                           if not (HERE / name).exists() or file_hash(HERE / name) != digest]
        manifest.update(
            elapsed_seconds=time.perf_counter() - started,
            code_cells_executed=sum(cell.cell_type == "code" and cell.execution_count is not None
                                    for cell in completed.cells),
            figures={path.name: file_hash(path) for path in sorted(figures.glob("*.png"))},
            sources_changed_during_run=changed_sources,
        )
        (output / "execution.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: manifest[key] for key in ("status", "scope", "code_cells_executed", "elapsed_seconds")}), flush=True)


if __name__ == "__main__":
    main()
