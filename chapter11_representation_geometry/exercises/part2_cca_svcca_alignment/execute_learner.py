"""Execute the alignment draft using its own solution dropdowns on CPU."""

from __future__ import annotations
import argparse, hashlib, json, sys, time
from pathlib import Path
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from chapter11_representation_geometry.exercises.part2_cca_svcca_alignment.build_notebooks import (
    HERE,
    lesson_cells,
    complete_learner,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--scope", choices=["foundations", "full"], default="full")
    args = parser.parse_args()
    run = args.output_directory.resolve()
    run.mkdir(parents=True, exist_ok=False)
    names = [
        "lesson.md",
        "alignment_methods.py",
        "alignment_methods_tests.py",
        "alignment_ioi_data.py",
        "alignment_model.py",
        "alignment_study.py",
        "run_alignment_development.py",
        "alignment_data.py",
        "alignment_protocol.json",
        "build_notebooks.py",
        "execute_learner.py",
    ]
    inputs = {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in names}
    snapshot = run / "source_snapshot"
    snapshot.mkdir()
    for name in names:
        (snapshot / name).write_bytes((HERE / name).read_bytes())
    cells = lesson_cells()
    if args.scope == "foundations":
        stop = next(
            i
            for i, c in enumerate(cells)
            if "".join(c["source"]).startswith("# 3️⃣ Test a real intervention")
        )
        cells = cells[:stop]
    notebook = complete_learner(
        {
            "cells": cells,
            "metadata": {
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }
    )
    notebook["cells"][1]["source"].append(
        f"\nassets_dir = Path({str(run / 'figures')!r})\nassets_dir.mkdir(exist_ok=True)\n"
    )
    notebook = nbformat.reads(json.dumps(notebook), as_version=4)
    state = {
        "status": "RUNNING",
        "scope": args.scope,
        "full_release_validation": False,
        "completed_from_own_answers": True,
        "source_inputs": inputs,
    }
    started = time.time()

    def save():
        (run / "execution.json").write_text(json.dumps(state, indent=2) + "\n")

    save()

    def completed(cell, cell_index, **kwargs):
        state["last_completed_cell"] = cell_index
        nbformat.write(notebook, run / "completed_learner.ipynb")
        save()

    try:
        NotebookClient(
            notebook,
            kernel_name="python3",
            timeout=3600,
            on_cell_executed=completed,
            resources={"metadata": {"path": str(ROOT)}},
        ).execute()
        state.update(
            status="PASS",
            code_cells=sum(c.cell_type == "code" for c in notebook.cells),
            figures=sorted(p.name for p in (run / "figures").glob("*.png")),
        )
    except BaseException as error:
        state.update(status="FAIL", error=str(error))
        raise
    finally:
        state.update(
            elapsed_seconds=time.time() - started,
            changed_sources=[
                name
                for name, digest in inputs.items()
                if hashlib.sha256((HERE / name).read_bytes()).hexdigest() != digest
            ],
        )
        if state["changed_sources"]:
            state["status"] = "FAIL"
        nbformat.write(notebook, run / "completed_learner.ipynb")
        save()
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
