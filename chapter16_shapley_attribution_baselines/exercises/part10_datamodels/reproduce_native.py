"""Rebuild all native fits independently and compare every learner result."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from chapter16_shapley_attribution_baselines.exercises.part10_datamodels.run_native import run
from chapter16_shapley_attribution_baselines.exercises.part9_influence_and_tracin.reproduce_native import (
    compare,
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reproduce(learner_directory: Path, output_directory: Path):
    learner = json.loads((learner_directory / "execution.json").read_text())
    if learner.get("status") != "PASS" or not learner.get("native_model_executed"):
        raise ValueError("Require a complete native own-answer learner before reproduction.")
    output_directory.mkdir(parents=True, exist_ok=False)
    inputs = dict(learner["source_inputs"])
    inputs[str(Path(__file__).relative_to(ROOT))] = digest(Path(__file__))
    comparison_source = HERE.parent / "part9_influence_and_tracin/reproduce_native.py"
    inputs[str(comparison_source.relative_to(ROOT))] = digest(comparison_source)
    for name, expected in inputs.items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Own learner source changed before reproduction: {name}")
        target = output_directory / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    for name, evidence in learner["tensor_artifacts"].items():
        path = learner_directory / name
        if digest(path) != evidence["sha256"] or path.stat().st_size != evidence["size_bytes"]:
            raise ValueError(f"Own learner tensor changed: {name}")
    state = dict(
        status="RUNNING",
        learner_directory=str(learner_directory),
        learner_execution_sha256=digest(learner_directory / "execution.json"),
        source_inputs=inputs,
        native_model_retrained=False,
        comparisons=dict(tensors=0, tensor_elements=0, scalars=0, max_absolute_error=0.0),
    )
    start = time.perf_counter()
    try:
        # This call reloads the pinned source archive and fits every model anew.
        # No learner weights, coefficients, responses or masks are supplied.
        native_state = run(output_directory / "fresh_run")
        state["native_model_retrained"] = True
        original = Path(learner["native_directory"])
        fresh = output_directory / "fresh_run/native"
        if {p.name for p in original.glob("*.pt")} != {"prepared.pt", "evaluated.pt", "play.pt"}:
            raise ValueError("The expected complete learner tensor set is missing.")
        for name in ("prepared.pt", "evaluated.pt", "play.pt"):
            compare(
                torch.load(fresh / name, weights_only=True),
                torch.load(original / name, weights_only=True),
                name,
                state["comparisons"],
            )
        # The independent numerical references must also agree, including model IDs.
        compare(
            json.loads((output_directory / "fresh_run/reference_checks.json").read_text()),
            json.loads((learner_directory / "figures/reference_checks.json").read_text()),
            "independent_references",
            state["comparisons"],
        )
        state.update(
            status="PASS",
            native_execution=native_state,
            artifacts={
                str(p.relative_to(output_directory)): dict(
                    sha256=digest(p), size_bytes=p.stat().st_size
                )
                for p in fresh.glob("*.pt")
            },
        )
    except BaseException as error:
        state.update(status="FAIL", error=str(error))
        raise
    finally:
        state["changed_sources"] = [
            name for name, expected in inputs.items() if digest(ROOT / name) != expected
        ]
        if state["changed_sources"]:
            state["status"] = "FAIL"
        state["elapsed_seconds"] = time.perf_counter() - start
        (output_directory / "reproduction.json").write_text(json.dumps(state, indent=2) + "\n")
    if state["status"] != "PASS":
        raise ValueError("A source changed during fresh reproduction.")
    print(
        json.dumps(
            {k: v for k, v in state.items() if k not in ("source_inputs", "artifacts")}, indent=2
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--learner-directory", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    args = parser.parse_args()
    reproduce(args.learner_directory.resolve(), args.output_directory.resolve())
