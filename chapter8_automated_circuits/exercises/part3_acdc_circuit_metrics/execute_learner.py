"""Execute the readable 8.3 learner, completed solely from its own answers.

Partial scopes are explicitly recorded and never publish release artifacts.
Real executions start fresh searches in their own output directory. An explicit
resume option can continue checkpoints from this same learner's interrupted
execution, after validating ownership, unchanged methods and live circuit loss.
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
from chapter8_automated_circuits.exercises.part3_acdc_circuit_metrics.build_notebooks import (
    HERE, STEM, complete_learner, lesson_cells,
)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_owned_resume(prior: Path, output: Path, current_sources: dict) -> dict:
    prior = prior.resolve()
    record_path = prior / "execution.json"
    record = json.loads(record_path.read_text())
    if (record.get("status") != "FAIL" or record.get("scope") != "full"
            or record.get("completed_from_own_answers") is not True
            or record.get("fresh_searches_required") is not True
            or record.get("sources_changed_during_run")
            or record.get("output_directory_override") != str(prior)):
        raise ValueError("Resume requires this learner's unchanged, interrupted full execution")
    previous_notebook = json.loads((prior / "completed_learner.ipynb").read_text())
    errors = [result["ename"] for cell in previous_notebook["cells"] for result in cell.get("outputs", [])
              if result.get("output_type") == "error"]
    if errors != ["KeyboardInterrupt"]:
        raise ValueError("Only an interrupted kernel may resume; method failures require a fresh run")
    # These files change only to add durable execution/checkpoint orchestration.
    # The lesson, graph, interventions, metrics, all other tests and paper code
    # must remain exact. Each checkpoint independently binds actual method ASTs.
    infrastructure = {"execute_learner.py", "edge_experiments.py", "edge_experiment_tests.py"}
    previous_sources = record["input_files"]
    if set(previous_sources) != set(current_sources):
        raise ValueError("Owned continuation requires the same complete section source set")
    changed = [name for name, digest in previous_sources.items() if current_sources[name] != digest]
    if set(changed) - infrastructure:
        raise ValueError("A learner or method source changed; start fresh searches")
    if file_hash(prior / "lesson_snapshot.md") != previous_sources["lesson.md"]:
        raise ValueError("The interrupted learner snapshot changed")
    snapshots = output / "owned_checkpoint_snapshots"
    snapshots.mkdir()
    inventory = {"prior_execution_sha256": file_hash(record_path), "prior_directory": str(prior),
                 "prior_notebook_sha256": file_hash(prior / "completed_learner.ipynb"),
                 "infrastructure_changes": changed, "prior_source_sha256": previous_sources, "checkpoints": []}
    for index, path in enumerate(sorted((prior / "searches").rglob("checkpoint.json"))):
        if not path.resolve().is_relative_to(prior / "searches"):
            raise ValueError("A checkpoint points outside its owned learner output")
        context = json.loads(path.with_name("context.json").read_text())
        target = snapshots / f"checkpoint-{index}.json"
        target.write_bytes(path.read_bytes())
        inventory["checkpoints"].append({"context": context, "prior_checkpoint": str(path),
            "checkpoint_sha256": file_hash(target), "snapshot_path": str(target)})
    if not inventory["checkpoints"]:
        raise ValueError("The interrupted learner has no owned checkpoints")
    (output / "owned_resume_inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    return inventory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("numerical", "greater-than", "full"), default="full")
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--resume-interrupted-run", type=Path)
    args = parser.parse_args()
    output = args.output_directory.resolve()
    if args.publish and args.scope != "full":
        raise ValueError("Only a successful full learner execution may publish outputs")
    output.mkdir(parents=True, exist_ok=False)
    figures, runs = output / "figures", output / "searches"
    figures.mkdir()
    tracked_sources = [HERE / "lesson.md", *sorted(HERE.glob("*.py"))]
    source_hashes = {str(path.relative_to(HERE)): file_hash(path) for path in tracked_sources}
    if args.resume_interrupted_run and args.scope != "full":
        raise ValueError("Owned interrupted continuation is a full-learner operation")
    owned_resume = (prepare_owned_resume(args.resume_interrupted_run, output, source_hashes)
                    if args.resume_interrupted_run else None)
    # Preserve the actual executed lesson even if later work edits the draft.
    (output / "lesson_snapshot.md").write_bytes((HERE / "lesson.md").read_bytes())
    learner = json.loads((HERE / f"{STEM}_exercises.ipynb").read_text())
    learner["cells"] = lesson_cells()
    if args.scope == "greater-than":
        end = next(index for index, cell in enumerate(learner["cells"])
                   if cell["cell_type"] == "markdown" and
                   "### Four thresholds, four fresh searches" in "".join(cell["source"]))
        learner["cells"] = learner["cells"][:end]
    completed = nbformat.reads(json.dumps(complete_learner(learner)), as_version=4)
    original_code = [cell.source for cell in completed.cells if cell.cell_type == "code"]
    # Only output paths are overridden; the method and experiment bodies stay intact.
    completed.cells[1].source += f"\nassets_dir = Path({str(figures)!r})\n"
    for cell in completed.cells:
        if cell.cell_type == "code" and 'edge_run_directory = section_dir / "expected_outputs"' in cell.source:
            cell.source = cell.source.replace(
                'edge_run_directory = section_dir / "expected_outputs" / "learner_edge_runs"',
                f"edge_run_directory = Path({str(runs)!r})",
            )

    def progress(cell, cell_index, **_):
        if cell.cell_type == "code":
            print(f"Executing learner cell {cell_index}: {cell.source.splitlines()[0]}", flush=True)

    environment = {**os.environ, "ARENA_ACDC_FRESH_SEARCH": "1",
                   "ARENA_ACDC_NUMERICAL_ONLY": "1" if args.scope == "numerical" else "0"}
    environment.pop("ARENA_ACDC_OWNED_RESUME_INVENTORY", None)
    if owned_resume:
        environment["ARENA_ACDC_OWNED_RESUME_INVENTORY"] = str(output / "owned_resume_inventory.json")
    client = NotebookClient(completed, timeout=14400, kernel_name="python3",
                            resources={"metadata": {"path": str(ROOT)}}, on_cell_start=progress)
    manifest = {
        "status": "RUNNING", "scope": args.scope, "completed_from_own_answers": True,
        "fresh_searches_required": True, "input_files": source_hashes,
        "output_directory_override": str(output), "full_release_validation": args.scope == "full",
        "search_origin": "owned_interrupted_continuation" if owned_resume else "fresh_in_this_kernel",
        "owned_resume": owned_resume,
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
        evidence = runs / "completed_edge_study.json.gz"
        if not evidence.exists():
            raise ValueError("The full learner must save its raw experiments before publication")
        for actual, expected in zip(completed.cells, solved.cells, strict=True):
            if actual.cell_type == "code":
                expected.outputs, expected.execution_count = actual.outputs, actual.execution_count
        nbformat.write(solved, HERE / f"{STEM}_solutions.ipynb")
        for path in figures.glob("*.png"):
            (HERE.parents[1] / "instructions" / "assets" / path.name).write_bytes(path.read_bytes())
        target = HERE / "expected_outputs" / "completed_edge_study.json.gz"
        target.write_bytes(evidence.read_bytes())
        manifest["published_evidence_sha256"] = file_hash(target)
        manifest["published_notebook_sha256"] = file_hash(HERE / f"{STEM}_solutions.ipynb")
        manifest["published_exercises_sha256"] = file_hash(HERE / f"{STEM}_exercises.ipynb")
        (HERE / "expected_outputs" / "learner_execution.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: manifest[key] for key in ("status", "scope", "code_cells_executed", "elapsed_seconds")}), flush=True)


if __name__ == "__main__":
    main()
