"""Execute this draft from its own exercise answers in a fresh CPU kernel."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import nbformat
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from chapter16_shapley_attribution_baselines.exercises.part9_influence_and_tracin.build_notebooks import complete_learner, lesson_cells

SOURCE_NAMES = [
    "lesson.md", "attribution_methods.py", "attribution_methods_tests.py",
    "attribution_data.py", "attribution_data_tests.py", "attribution_study.py",
    "attribution_study_tests.py", "attribution_reference.py", "attribution_play.py",
    "data_manifest.json", "native_protocol.md", "build_notebooks.py", "execute_learner.py",
]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def execute(directory: Path):
    directory.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for name in SOURCE_NAMES:
        path = HERE / name
        relative = str(path.relative_to(ROOT))
        snapshot = directory / "source_snapshot" / relative
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(path.read_bytes())
        hashes[relative] = digest(path)
    state = {"status": "RUNNING", "scope": "full_real_digit_learner", "native_model_executed": False,
             "source_inputs": hashes, "completed_from_own_answers": True}
    started = time.perf_counter()
    notebook = None
    def save():
        state["elapsed_seconds"] = time.perf_counter() - started
        (directory / "execution.json").write_text(json.dumps(state, indent=2) + "\n")
    def finished(cell, cell_index, **kwargs):
        state["last_completed_cell"] = cell_index
        save()
    save()
    try:
        draft = {"cells": lesson_cells(), "metadata": {"kernelspec": {
            "name": "python3", "display_name": "Python 3", "language": "python"}}, "nbformat": 4, "nbformat_minor": 5}
        notebook = nbformat.reads(json.dumps(complete_learner(draft)), as_version=4)
        NotebookClient(notebook, timeout=1800, kernel_name="python3", on_cell_executed=finished,
                       resources={"metadata": {"path": str(ROOT)}}).execute(
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
                 "ARENA_ATTRIBUTION_OUTPUT_DIRECTORY": str(directory / "figures")})
        summary_path = directory / "figures/native_summary.json"
        summary = json.loads(summary_path.read_text())
        for key, expected in {
            "reference_refits": 930, "sgd_paths": 6, "sgd_updates": 30720,
            "train_count": 128, "ranking_query_count": 48, "final_query_count": 64,
            "policy_selection_queries": "ranking_only",
            "final_evaluated_after_both_reference_conditions_complete": True,
        }.items():
            if summary.get(key) != expected:
                raise ValueError(f"Full learner contract failed: {key}.")
        if len(summary["independent_scipy_checks"]) != 10:
            raise ValueError("The independent native reference checks were not completed.")
        if any(value["max_refit_gradient"] > 1e-10 for value in summary["conditions"].values()):
            raise ValueError("A reference refit did not converge.")
        native_directory = Path(summary["native_directory"])
        native_directory.relative_to(directory)
        tensor_paths = sorted(native_directory.rglob("*.pt"))
        if len(tensor_paths) != 15:
            raise ValueError("The full native run must retain fourteen study tensors and the query play.")
        state.update(native_model_executed=True, native_summary_sha256=digest(summary_path),
                     native_summary=summary, native_directory=str(native_directory),
                     tensor_artifacts={str(p.relative_to(directory)): {
                         "sha256": digest(p), "size_bytes": p.stat().st_size,
                     } for p in tensor_paths})
        state.update(status="PASS", cells=len(notebook.cells), code_cells=sum(c.cell_type == "code" for c in notebook.cells),
                     embedded_figures=sum("image/png" in o.get("data", {}) for c in notebook.cells for o in c.get("outputs", [])),
                     figures={p.name: digest(p) for p in sorted((directory / "figures").glob("*.png"))})
    except BaseException as error:
        state.update(status="FAIL", error=str(error))
        raise
    finally:
        state["changed_sources"] = [name for name, checksum in hashes.items() if digest(ROOT / name) != checksum]
        if state["changed_sources"]:
            state["status"] = "FAIL"
        if notebook is not None:
            nbformat.write(notebook, directory / "completed_learner.ipynb")
            state["completed_learner_sha256"] = digest(directory / "completed_learner.ipynb")
        save()
    if state["status"] != "PASS":
        raise ValueError("The learner inputs changed during execution.")
    print(json.dumps({k: v for k, v in state.items()
                      if k not in ("source_inputs", "figures", "tensor_artifacts", "native_summary")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path, required=True)
    execute(parser.parse_args().output_directory.resolve())
