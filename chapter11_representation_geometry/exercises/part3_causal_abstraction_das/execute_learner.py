"""Execute the complete developing learner from its own answer dropdowns.

This runs the exact model, native GPT-2 development and frozen final evaluation.
Publication still requires separate fresh native reproduction and visual review.
"""

from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from chapter11_representation_geometry.exercises.part3_causal_abstraction_das.build_notebooks import (
    HERE,
    lesson_cells,
    complete_learner,
)


def execute(output: Path):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    names = [
        "lesson.md",
        "build_notebooks.py",
        "execute_learner.py",
        "das_methods.py",
        "das_methods_tests.py",
        "das_organism.py",
        "das_training_methods.py",
        "das_training_tests.py",
        "das_native.py",
        "das_native_tests.py",
        "das_ioi_data.py",
        "das_protocol.py",
        "native_protocol.md",
        "run_das_screen.py",
        "run_native_development.py",
    ]
    digests = {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in names}
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for name in names:
        (snapshot / name).write_bytes((HERE / name).read_bytes())
    notebook = None
    started = time.perf_counter()
    state = {
        "status": "RUNNING",
        "source_inputs": digests,
        "scope": "exact_model_and_pretrained_GPT2",
        "pretrained_model_executed": False,
        "full_release_validation": False,
        "completed_from_own_answers": True,
    }

    def save():
        state["elapsed_seconds"] = time.perf_counter() - started
        (output / "execution.json").write_text(json.dumps(state, indent=2) + "\n")

    save()

    def finished(cell, cell_index, **kwargs):
        state["last_completed_cell"] = cell_index
        save()

    try:
        notebook = complete_learner(
            {
                "cells": lesson_cells(),
                "metadata": {
                    "kernelspec": {
                        "display_name": "Python 3",
                        "language": "python",
                        "name": "python3",
                    }
                },
                "nbformat": 4,
                "nbformat_minor": 5,
            }
        )
        notebook = nbformat.reads(json.dumps(notebook), as_version=4)
        NotebookClient(
            notebook,
            timeout=900,
            kernel_name="python3",
            on_cell_executed=finished,
            resources={"metadata": {"path": str(ROOT)}},
        ).execute(env={**os.environ, "ARENA_DAS_OUTPUT_DIRECTORY": str(output / "figures")})
        state.update(
            status="PASS",
            code_cells=sum(c.cell_type == "code" for c in notebook.cells),
            embedded_figures=sum(
                "image/png" in o.get("data", {})
                for c in notebook.cells
                for o in c.get("outputs", [])
            ),
            figures=sorted(p.name for p in (output / "figures").glob("*.png")),
        )
        native = json.loads((output / "figures" / "native_summary.json").read_text())
        if not native["weights_unchanged"] or not native["model_gradients_absent"]:
            raise ValueError("The pretrained model must remain frozen throughout the learner.")
        state.update(
            pretrained_model_executed=True,
            native_summary=native,
            native_tensor_sha256=hashlib.sha256(
                (output / "figures" / "actual_native_tensors.pt").read_bytes()
            ).hexdigest(),
        )
    except BaseException as error:
        state.update(status="FAIL", error=str(error))
        raise
    finally:
        state["elapsed_seconds"] = time.perf_counter() - started
        state["changed_sources"] = [
            name
            for name, digest in digests.items()
            if hashlib.sha256((HERE / name).read_bytes()).hexdigest() != digest
        ]
        if state["changed_sources"]:
            state["status"] = "FAIL"
        if notebook is not None:
            nbformat.write(notebook, output / "completed_learner.ipynb")
        save()
    if state["status"] != "PASS":
        raise ValueError("Draft sources changed during execution.")
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path, required=True)
    execute(parser.parse_args().output_directory)
