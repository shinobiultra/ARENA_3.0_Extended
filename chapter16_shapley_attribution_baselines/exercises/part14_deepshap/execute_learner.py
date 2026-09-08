"""Execute the complete DeepSHAP own-answer lesson in a fresh CPU kernel."""

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
from chapter16_shapley_attribution_baselines.exercises.part14_deepshap.run_native import (
    DETERMINISTIC_ARTIFACTS,
    TIMING_ARTIFACT,
    source_paths,
)

EXPECTED_FIGURES = tuple(
    f"{stem}.png"
    for stem in (
        "01_rescale_limits",
        "02_near_zero_epsilon",
        "03_affine_controls",
        "04_background_average",
        "05_threshold_complete_not_exact",
        "06_cancellation_implementation_dependence",
        "07_replacement_fixture",
        "08_performance_controls",
        "09_signed_parity_all_rows",
        "10_error_heatmaps_all_rows",
        "11_fixed_waterfall",
        "12_fixed_attribution_comparison",
        "13_completeness_vs_exact",
        "14_background_sensitivity",
        "15_replacement_curves",
        "16_threshold_play",
        "17_real_recipe_play",
    )
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def learner_source_paths() -> list[Path]:
    """Sources whose exact bytes are bound to an own-answer execution."""
    paths = source_paths() + [
        HERE / name
        for name in (
            "lesson.md",
            "build_notebooks.py",
            "execute_learner.py",
            "deepshap_contract_tests.py",
            "reproduce_native.py",
        )
    ]
    return list(dict.fromkeys(paths))


def _write_state(directory: Path, state: dict) -> None:
    (directory / "execution.json").write_text(json.dumps(state, indent=2) + "\n")


def execute(directory: Path) -> dict:
    """Complete every answer, execute every cell, and inventory native/visual evidence."""
    from chapter16_shapley_attribution_baselines.exercises.part14_deepshap.build_notebooks import (
        complete_learner,
        lesson_cells,
    )

    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    paths = learner_source_paths()
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        state = {"status": "FAIL", "error": "Required learner sources missing: " + ", ".join(missing)}
        _write_state(directory, state)
        raise FileNotFoundError(state["error"])
    hashes: dict[str, str] = {}
    for path in paths:
        relative = str(path.relative_to(ROOT))
        snapshot = directory / "source_snapshot" / relative
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(path.read_bytes())
        hashes[relative] = digest(path)
    state = {
        "status": "RUNNING",
        "scope": "complete_deepshap_own_answer_learner",
        "source_inputs": hashes,
        "completed_from_own_answers": True,
        "native_model_executed": False,
        "execution_device": "cpu",
    }
    started = time.perf_counter()
    notebook = None

    def save() -> None:
        state["elapsed_seconds"] = time.perf_counter() - started
        _write_state(directory, state)

    def finished(cell, cell_index, **kwargs) -> None:
        state["last_completed_cell"] = cell_index
        save()

    save()
    try:
        draft = {
            "cells": lesson_cells(),
            "metadata": {
                "kernelspec": {
                    "name": "python3",
                    "display_name": "Python 3",
                    "language": "python",
                }
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        notebook = nbformat.reads(json.dumps(complete_learner(draft)), as_version=4)
        NotebookClient(
            notebook,
            timeout=3600,
            kernel_name="python3",
            on_cell_executed=finished,
            resources={"metadata": {"path": str(ROOT)}},
        ).execute(
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": "",
                "OMP_NUM_THREADS": "4",
                "MKL_NUM_THREADS": "4",
                "OPENBLAS_NUM_THREADS": "4",
                "ARENA_DEEPSHAP_OUTPUT_DIRECTORY": str(directory / "figures"),
            }
        )
        figure_directory = directory / "figures"
        summary_path = figure_directory / "native_summary.json"
        if not summary_path.is_file():
            raise ValueError("The complete lesson did not write native_summary.json.")
        summary = json.loads(summary_path.read_text())
        required = {
            "model_fits": 7,
            "mlp_fits": 6,
            "ridge_fits": 1,
            "clean_fits": 3,
            "shuffled_fits": 3,
            "optimizer_updates": 4800,
            "train_rows": 758,
            "explanation_row_count": 153,
            "heldout_row_count": 119,
            "rows_explained_per_model": 272,
            "model_row_explanations": 1904,
            "coalitions": 256,
            "exact_models": 7,
            "exact_model_rows": 1904,
            "coalition_values": 487424,
            "primary_hybrid_predictions": 31195136,
            "attribution_coordinates_per_method": 15232,
            "background_rows": 64,
            "background_variants": 5,
            "replacement_models": 7,
            "random_order_controls": 100,
            "pair_model_rows": 121856,
            "pair_local_values": 974848,
            "audit_used_for_selection": False,
        }
        for key, expected in required.items():
            if summary.get(key) != expected:
                raise ValueError(f"Complete DeepSHAP learner contract failed: {key}.")
        native = Path(summary["native_directory"]).resolve()
        native.relative_to(directory)
        expected_artifacts = set(DETERMINISTIC_ARTIFACTS) | {TIMING_ARTIFACT}
        artifacts = sorted(native.glob("*.pt"))
        if {path.name for path in artifacts} != expected_artifacts:
            raise ValueError("The complete native DeepSHAP payload set is missing or has extras.")
        boundary = json.loads((native / "before_audit.json").read_text())
        if boundary.get("audit_loaded") is not False:
            raise ValueError("The explanation ranking boundary did not precede audit loading.")
        if digest(native / "prepared.pt") != boundary.get("prepared_sha256"):
            raise ValueError("Prepared rankings changed after the audit boundary.")
        figures = sorted(figure_directory.glob("*.png"))
        embedded = sum(
            "image/png" in output.get("data", {})
            for cell in notebook.cells
            for output in cell.get("outputs", [])
        )
        if {path.name for path in figures} != set(EXPECTED_FIGURES):
            raise ValueError("The complete learner must generate the 17 frozen DeepSHAP figures.")
        if embedded != len(figures):
            raise ValueError("Every generated figure must be visibly embedded exactly once.")
        state.update(
            status="PASS",
            native_model_executed=True,
            native_summary=summary,
            native_summary_sha256=digest(summary_path),
            native_directory=str(native),
            native_artifacts={
                str(path.relative_to(directory)): {
                    "sha256": digest(path),
                    "size_bytes": path.stat().st_size,
                }
                for path in artifacts
            },
            cells=len(notebook.cells),
            code_cells=sum(cell.cell_type == "code" for cell in notebook.cells),
            embedded_figures=embedded,
            figures={path.name: digest(path) for path in figures},
        )
    except BaseException as error:
        state.update(status="FAIL", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        state["changed_sources"] = [
            name
            for name, checksum in hashes.items()
            if not (ROOT / name).is_file() or digest(ROOT / name) != checksum
        ]
        if state["changed_sources"]:
            state["status"] = "FAIL"
        if notebook is not None:
            completed = directory / "completed_learner.ipynb"
            nbformat.write(notebook, completed)
            state["completed_learner_sha256"] = digest(completed)
        save()
    if state["status"] != "PASS":
        raise ValueError("Learner sources changed during execution.")
    print(
        json.dumps(
            {
                key: value
                for key, value in state.items()
                if key not in {"source_inputs", "native_artifacts", "native_summary", "figures"}
            },
            indent=2,
        )
    )
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path, required=True)
    execute(parser.parse_args().output_directory)
