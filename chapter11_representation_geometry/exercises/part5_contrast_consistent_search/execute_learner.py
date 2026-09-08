"""Execute the current CCS draft from its own answers in a fresh notebook kernel."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from chapter11_representation_geometry.exercises.part5_contrast_consistent_search.build_notebooks import (
    complete_learner,
    lesson_cells,
)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def execute(directory):
    directory.mkdir(parents=True, exist_ok=False)
    names = ["lesson.md", "ccs_methods.py", "ccs_methods_tests.py", "reference_parity.py",
             "build_notebooks.py", "execute_learner.py", "ccs_native_methods.py",
             "ccs_native_methods_tests.py", "ccs_native.py", "ccs_data.py", "ccs_study.py", "native_protocol.md"]
    paths = [HERE / name for name in names]
    paths.append(ROOT / "chapter11_representation_geometry/exercises/part4_concept_erasure/erasure_methods.py")
    hashes = {}
    for path in paths:
        relative = str(path.relative_to(ROOT))
        snapshot = directory / "source_snapshot" / relative
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(path.read_bytes())
        hashes[relative] = digest(path)
    started = time.perf_counter()
    state = {"status": "RUNNING", "source_inputs": hashes,
             "completed_from_own_answers": True, "native_model_executed": False}
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
            "name": "python3", "display_name": "Python 3", "language": "python"}},
            "nbformat": 4, "nbformat_minor": 5}
        notebook = nbformat.reads(json.dumps(complete_learner(draft)), as_version=4)
        NotebookClient(notebook, timeout=1800, kernel_name="python3", on_cell_executed=finished,
                       resources={"metadata": {"path": str(ROOT)}}).execute(
            env={**os.environ, "ARENA_CCS_OUTPUT_DIRECTORY": str(directory / "figures")})
        state.update(status="PASS", cells=len(notebook.cells),
                     code_cells=sum(cell.cell_type == "code" for cell in notebook.cells),
                     embedded_figures=sum("image/png" in output.get("data", {})
                                          for cell in notebook.cells for output in cell.get("outputs", [])),
                     figures={p.name: digest(p) for p in sorted((directory / "figures").glob("*.png"))})
        summary_path = directory / "figures/native_summary.json"
        summary = json.loads(summary_path.read_text())
        if summary["status"] != "PASS" or not summary["weights_unchanged"] or not summary["model_gradients_absent"]:
            raise ValueError("The complete frozen native experiment did not pass.")
        if summary["case_count"] != 18 or summary["final_reporter_scores"] != 98304:
            raise ValueError("Every prescribed case and final reporter score must remain in the learner.")
        if not summary["final_started_only_after_all_fits"] or summary["final_used_for_selection"]:
            raise ValueError("Final data must not select or fit the reporters.")
        state.update(native_model_executed=True, native_summary_sha256=digest(summary_path),
                     native_case_count=summary["case_count"], final_reporter_scores=summary["final_reporter_scores"],
                     native_artifacts={str(p.relative_to(directory / "figures")): {"sha256": digest(p), "size_bytes": p.stat().st_size}
                                       for p in sorted((directory / "figures/native").glob("*.pt"))})
    except BaseException as error:
        state.update(status="FAIL", error=str(error))
        raise
    finally:
        state["changed_sources"] = [name for name, checksum in hashes.items() if digest(ROOT / name) != checksum]
        if state["changed_sources"]:
            state["status"] = "FAIL"
        if notebook is not None:
            nbformat.write(notebook, directory / "completed_learner.ipynb")
        save()
    if state["status"] != "PASS":
        raise ValueError("Learner source changed during execution.")
    print(json.dumps({key: value for key, value in state.items() if key not in {"source_inputs", "figures"}}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path, required=True)
    execute(parser.parse_args().output_directory.resolve())
