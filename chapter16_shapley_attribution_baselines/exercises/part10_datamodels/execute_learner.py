"""Execute this section from its own exercise answers in a fresh CPU kernel."""

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
from chapter16_shapley_attribution_baselines.exercises.part10_datamodels.build_notebooks import (
    complete_learner,
    lesson_cells,
)
from chapter16_shapley_attribution_baselines.exercises.part10_datamodels.run_native import (
    source_paths,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def learner_source_paths():
    paths = source_paths() + [
        HERE / name
        for name in (
            "lesson.md",
            "build_notebooks.py",
            "execute_learner.py",
            "datamodel_contract_tests.py",
            "reproduce_native.py",
        )
    ]
    paths.append(HERE.parent / "part9_influence_and_tracin/reproduce_native.py")
    return paths


def execute(directory: Path):
    directory.mkdir(parents=True, exist_ok=False)
    paths = learner_source_paths()
    hashes = {}
    for path in paths:
        relative = str(path.relative_to(ROOT))
        snapshot = directory / "source_snapshot" / relative
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(path.read_bytes())
        hashes[relative] = digest(path)
    state = dict(
        status="RUNNING",
        scope="complete_datamodels_learner",
        source_inputs=hashes,
        completed_from_own_answers=True,
        native_model_executed=False,
    )
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
        draft = dict(
            cells=lesson_cells(),
            metadata={
                "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}
            },
            nbformat=4,
            nbformat_minor=5,
        )
        notebook = nbformat.reads(json.dumps(complete_learner(draft)), as_version=4)
        NotebookClient(
            notebook,
            timeout=600,
            kernel_name="python3",
            on_cell_executed=finished,
            resources={"metadata": {"path": str(ROOT)}},
        ).execute(
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": "",
                "OMP_NUM_THREADS": "4",
                "MKL_NUM_THREADS": "4",
                "ARENA_DATAMODELS_OUTPUT_DIRECTORY": str(directory / "figures"),
            }
        )
        summary_path = directory / "figures/native_summary.json"
        summary = json.loads(summary_path.read_text())
        for key, expected in dict(
            classifier_refits=10753,
            separate_refits=6,
            independent_scipy_optima=10,
            independent_lasso_fits=12,
            play_refits=18,
            train_images=128,
            query_images=64,
            heldout_subsets_per_fraction=256,
            selection_used_test_responses=False,
        ).items():
            if summary.get(key) != expected:
                raise ValueError(f"Full learner contract failed: {key}.")
        if summary["max_classifier_gradient"] > 1e-10:
            raise ValueError("A native classifier did not converge.")
        native = Path(summary["native_directory"])
        native.relative_to(directory)
        tensors = sorted(native.glob("*.pt"))
        if {p.name for p in tensors} != {"prepared.pt", "evaluated.pt", "play.pt"}:
            raise ValueError("The complete native payloads are missing.")
        state.update(
            native_model_executed=True,
            native_summary=summary,
            native_summary_sha256=digest(summary_path),
            native_directory=str(native),
            tensor_artifacts={
                str(p.relative_to(directory)): dict(sha256=digest(p), size_bytes=p.stat().st_size)
                for p in tensors
            },
        )
        state.update(
            status="PASS",
            cells=len(notebook.cells),
            code_cells=sum(c.cell_type == "code" for c in notebook.cells),
            embedded_figures=sum(
                "image/png" in o.get("data", {})
                for c in notebook.cells
                for o in c.get("outputs", [])
            ),
            figures={p.name: digest(p) for p in sorted((directory / "figures").glob("*.png"))},
        )
    except BaseException as error:
        state.update(status="FAIL", error=str(error))
        raise
    finally:
        state["changed_sources"] = [
            name for name, checksum in hashes.items() if digest(ROOT / name) != checksum
        ]
        if state["changed_sources"]:
            state["status"] = "FAIL"
        if notebook is not None:
            nbformat.write(notebook, directory / "completed_learner.ipynb")
            state["completed_learner_sha256"] = digest(directory / "completed_learner.ipynb")
        save()
    if state["status"] != "PASS":
        raise ValueError("The learner source changed during execution.")
    print(
        json.dumps(
            {
                key: value
                for key, value in state.items()
                if key not in ("source_inputs", "figures", "tensor_artifacts", "native_summary")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path, required=True)
    execute(parser.parse_args().output_directory.resolve())
