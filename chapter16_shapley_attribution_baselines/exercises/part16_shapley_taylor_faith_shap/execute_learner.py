"""Execute the complete 16.16 own-answer lesson in a fresh CPU kernel."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import nbformat
from nbclient import NotebookClient


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from chapter16_shapley_attribution_baselines.exercises.part16_shapley_taylor_faith_shap.run_native import (
    BACKGROUND_SIZES,
    DETERMINISTIC_ARTIFACTS,
    EXTERNAL_DETERMINISTIC_ARTIFACTS,
    FEATURES,
    MODEL_KEYS,
    PAIRS,
    SPLIT_ROWS,
    TIMING_ARTIFACT,
    digest,
    manifest_path,
    source_paths,
)


EXPECTED_FIGURES = tuple(
    f"{stem}.png"
    for stem in (
        "01_exact_allocation_ledger",
        "02_why_faith_moves_singletons",
        "10_dividend_play",
        "03_native_game_and_controls",
        "04_fixed_row_interaction_graph",
        "05_allrow_pair_heatmaps",
        "06_heldout_pair_order_capture",
        "07_order_two_reconstruction",
        "08_background_changes_the_game",
        "09_contextual_pair_interventions",
        "11_donor_context_play",
    )
)


def learner_source_paths() -> list[Path]:
    """Bind scientific, lesson, execution, reproduction, and publisher sources by bytes."""

    paths = source_paths() + [
        HERE / name
        for name in (
            "lesson.md",
            "build_notebooks.py",
            "execute_learner.py",
            "interaction_contracts.py",
            "reproduce_native.py",
            "publish_study.py",
        )
    ]
    return list(dict.fromkeys(paths))


def _write_state(directory: Path, state: Mapping[str, Any]) -> None:
    (directory / "execution.json").write_text(json.dumps(state, indent=2) + "\n")


def _validate_native_summary(summary: Mapping[str, Any]) -> None:
    exact = {
        "study": "16.16_shapley_taylor_faith_shap",
        "phase": "evaluated_after_audit",
        "model_keys": list(MODEL_KEYS),
        "model_fits": len(MODEL_KEYS),
        "known_model_state_hashes": 21,
        "train_rows": 758,
        "explanation_rows": SPLIT_ROWS["explain"],
        "audit_rows": SPLIT_ROWS["audit"],
        "all_rows": sum(SPLIT_ROWS.values()),
        "coalitions": 2**FEATURES,
        "pairs": PAIRS,
        "method_keys": ["sii", "stii", "fsii"],
        "backgrounds": list(BACKGROUND_SIZES),
        "audit_used_for_selection": False,
        "reference_status": "PASS",
        "deterministic_artifacts": list(DETERMINISTIC_ARTIFACTS),
        "scientific_results_are_descriptive": True,
    }
    for key, expected in exact.items():
        if summary.get(key) != expected:
            raise ValueError(
                f"Complete 16.16 learner contract failed at {key}: "
                f"{summary.get(key)!r} != {expected!r}."
            )
    explain = summary.get("primary_explain_hybrid", {})
    audit = summary.get("primary_audit_hybrid", {})
    if explain.get("shape") != [7, 153, 256, 64] or explain.get("dtype") != "float64":
        raise ValueError("The complete primary explanation hybrid tensor is missing.")
    if audit.get("shape") != [7, 119, 256, 64] or audit.get("dtype") != "float64":
        raise ValueError("The complete primary audit hybrid tensor is missing.")
    if summary.get("manifest_array_count", 0) <= 0:
        raise ValueError("Recursive artifact manifests retained no numerical arrays.")


def _native_inventory(native: Path) -> dict[str, dict[str, Any]]:
    expected_pt = set(DETERMINISTIC_ARTIFACTS) | {TIMING_ARTIFACT}
    if {path.name for path in native.glob("*.pt")} != expected_pt:
        raise ValueError("The complete native 16.16 payload set is missing or has extras.")
    metadata = {
        "before_audit.json",
        "summary.json",
        "artifact_manifest.json",
        *(manifest_path(Path(name)).name for name in expected_pt),
    }
    missing = [name for name in metadata if not (native / name).is_file()]
    if missing:
        raise ValueError("Native evidence metadata is incomplete: " + ", ".join(sorted(missing)))
    return {
        path.name: {"sha256": digest(path), "size_bytes": path.stat().st_size}
        for path in sorted(
            [*native.glob("*.pt"), *native.glob("*.manifest.json")]
            + [native / "before_audit.json", native / "summary.json", native / "artifact_manifest.json"]
        )
    }


def execute(directory: Path) -> dict[str, Any]:
    """Complete every answer, execute every cell, and inventory actual native evidence."""

    from chapter16_shapley_attribution_baselines.exercises.part16_shapley_taylor_faith_shap.build_notebooks import (
        complete_learner,
        lesson_cells,
        validate_answer_parity,
        validate_exercise_surface,
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
    state: dict[str, Any] = {
        "status": "RUNNING",
        "scope": "complete_16_16_own_answer_learner",
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
        completed = complete_learner(draft)
        validate_answer_parity(completed, required=True)
        validate_exercise_surface((HERE / "lesson.md").read_text())
        notebook = nbformat.reads(json.dumps(completed), as_version=4)
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
                "ARENA_INTERACTION_OUTPUT_DIRECTORY": str(directory / "figures"),
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
        if (
            boundary.get("phase") != "prepared_before_audit"
            or boundary.get("audit_loaded") is not False
            or boundary.get("state_hash_count") != 21
        ):
            raise ValueError("The physical explanation seal did not precede literal audit load.")
        figures = sorted(figure_directory.glob("*.png"))
        embedded = sum(
            "image/png" in output.get("data", {})
            for cell in notebook.cells
            for output in cell.get("outputs", [])
        )
        if {path.name for path in figures} != set(EXPECTED_FIGURES):
            raise ValueError("The complete learner must generate all 11 frozen figures.")
        if embedded != len(figures):
            raise ValueError("Every generated figure must be visibly embedded exactly once.")
        state.update(
            status="PASS",
            native_model_executed=True,
            native_summary=summary,
            native_summary_sha256=digest(summary_path),
            native_directory=str(native),
            native_artifacts=artifacts,
            external_hybrid_artifacts={
                name: {
                    "relative_path": str((native / name).relative_to(directory)),
                    **artifacts[name],
                    "manifest": manifest_path(Path(name)).name,
                }
                for name in EXTERNAL_DETERMINISTIC_ARTIFACTS
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
            completed_path = directory / "completed_learner.ipynb"
            nbformat.write(notebook, completed_path)
            state["completed_learner_sha256"] = digest(completed_path)
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
