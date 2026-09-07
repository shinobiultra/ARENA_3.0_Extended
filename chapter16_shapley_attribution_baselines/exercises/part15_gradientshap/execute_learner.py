"""Execute the complete GradientSHAP own-answer lesson in a fresh CPU kernel."""

from __future__ import annotations

import argparse
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
from chapter16_shapley_attribution_baselines.exercises.part15_gradientshap.run_native import (
    DETERMINISTIC_ARTIFACTS,
    EXTERNAL_DETERMINISTIC_ARTIFACT,
    REPLICATE_ELEMENTS,
    REPLICATE_SHAPE,
    TIMING_ARTIFACT,
    digest,
    manifest_path,
    source_paths,
)


EXPECTED_FIGURES = tuple(
    f"{stem}.png"
    for stem in (
        "01_schedule_prefix",
        "02_path_interpolation",
        "03_gradient_controls",
        "04_relu_samples",
        "05_mc_convergence",
        "06_estimator_controls",
        "07_quadrature_doubling",
        "08_background_average",
        "09_gaussian_target_shift",
        "10_path_vs_coalition",
        "11_kink_representation",
        "12_replacement_fixture",
        "13_threshold_play",
        "14_native_overview",
        "15_allrow_method_map",
        "16_fixed_waterfall",
        "17_background_and_noise",
        "18_replicates_and_distance",
        "19_replacement_curves",
        "20_real_recipe_play",
    )
)


def learner_source_paths() -> list[Path]:
    """Bind scientific, lesson, execution, and reproduction sources by exact bytes."""
    paths = source_paths() + [
        HERE / name
        for name in (
            "lesson.md",
            "build_notebooks.py",
            "execute_learner.py",
            "gradientshap_contracts.py",
            "reproduce_native.py",
            "publish_study.py",
        )
    ]
    return list(dict.fromkeys(paths))


def _write_state(directory: Path, state: dict) -> None:
    (directory / "execution.json").write_text(json.dumps(state, indent=2) + "\n")


def _validate_native_summary(summary: dict) -> None:
    exact = {
        "study": "16.15_gradientshap_expected_gradients",
        "phase": "evaluated_after_audit",
        "model_keys": [
            "clean_0",
            "clean_1",
            "clean_2",
            "shuffled_0",
            "shuffled_1",
            "shuffled_2",
            "ridge",
        ],
        "model_fits": 7,
        "optimizer_updates": 4800,
        "train_rows": 758,
        "explanation_rows": 153,
        "audit_rows": 119,
        "all_model_rows": 1904,
        "nested_budgets": [8, 32, 128, 512, 2048],
        "quadrature_nodes": [256, 512],
        "background_variants": 5,
        "noise_variants": 3,
        "sampling_replicates": 64,
        "replicate_sample_shape": list(REPLICATE_SHAPE),
        "replicate_sample_dtype": "float64",
        "replicate_sample_coordinates": REPLICATE_ELEMENTS,
        "primary_sample_coordinates": 31_195_136,
        "audit_used_for_selection": False,
        "reference_status": "PASS",
        "deterministic_artifacts": list(DETERMINISTIC_ARTIFACTS),
    }
    for key, expected in exact.items():
        if summary.get(key) != expected:
            raise ValueError(
                f"Complete GradientSHAP learner contract failed at {key}: "
                f"{summary.get(key)!r} != {expected!r}."
            )
    if summary.get("manifest_array_count", 0) <= 0:
        raise ValueError("Recursive artifact manifests did not retain any numerical arrays.")
    if summary.get("manifest_array_elements", 0) < REPLICATE_ELEMENTS:
        raise ValueError("Recursive manifests omit part of the full replicate contribution array.")


def _native_inventory(native: Path) -> dict[str, dict]:
    expected_pt = set(DETERMINISTIC_ARTIFACTS) | {TIMING_ARTIFACT}
    if {path.name for path in native.glob("*.pt")} != expected_pt:
        raise ValueError("The complete native GradientSHAP payload set is missing or has extras.")
    required_metadata = {
        "before_audit.json",
        "summary.json",
        "artifact_manifest.json",
        *(manifest_path(Path(name)).name for name in expected_pt),
    }
    missing = [name for name in required_metadata if not (native / name).is_file()]
    if missing:
        raise ValueError("Native evidence metadata is incomplete: " + ", ".join(sorted(missing)))
    records: dict[str, dict] = {}
    for path in sorted(
        [*native.glob("*.pt"), *native.glob("*.manifest.json")]
        + [native / "before_audit.json", native / "summary.json", native / "artifact_manifest.json"]
    ):
        records[str(path.name)] = {"sha256": digest(path), "size_bytes": path.stat().st_size}
    return records


def execute(directory: Path) -> dict:
    """Complete every answer, execute every cell, and inventory native/visual evidence."""
    from chapter16_shapley_attribution_baselines.exercises.part15_gradientshap.build_notebooks import (
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
        "scope": "complete_gradientshap_own_answer_learner",
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
            timeout=21_600,
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
                "ARENA_GRADIENTSHAP_OUTPUT_DIRECTORY": str(directory / "figures"),
                "ARENA_GRADIENTSHAP_FIGURE_DIR": str(directory / "figures"),
                "ARENA_GRADIENTSHAP_SAVE_FIGURES": "1",
            }
        )
        figure_directory = directory / "figures"
        summary_path = figure_directory / "native_summary.json"
        if not summary_path.is_file():
            raise ValueError("The complete lesson did not write native_summary.json.")
        summary = json.loads(summary_path.read_text())
        _validate_native_summary(summary)
        native = Path(summary["native_directory"]).resolve()
        native.relative_to(directory)
        artifacts = _native_inventory(native)
        boundary = json.loads((native / "before_audit.json").read_text())
        if boundary.get("phase") != "prepared_before_audit" or boundary.get("audit_loaded") is not False:
            raise ValueError("The fitted/prepared seal did not precede the literal audit load.")
        if boundary.get("state_hash_count") != 21:
            raise ValueError("The preparation boundary does not bind all 21 model state hashes.")
        figures = sorted(figure_directory.glob("*.png"))
        embedded = sum(
            "image/png" in output.get("data", {})
            for cell in notebook.cells
            for output in cell.get("outputs", [])
        )
        if {path.name for path in figures} != set(EXPECTED_FIGURES):
            raise ValueError("The complete learner must generate the 20 frozen GradientSHAP figures.")
        if embedded != len(figures):
            raise ValueError("Every generated figure must be visibly embedded exactly once.")
        state.update(
            status="PASS",
            native_model_executed=True,
            native_summary=summary,
            native_summary_sha256=digest(summary_path),
            native_directory=str(native),
            native_artifacts=artifacts,
            external_replicate_artifact={
                "relative_path": str((native / EXTERNAL_DETERMINISTIC_ARTIFACT).relative_to(directory)),
                **artifacts[EXTERNAL_DETERMINISTIC_ARTIFACT],
                "manifest": manifest_path(Path(EXTERNAL_DETERMINISTIC_ARTIFACT)).name,
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
