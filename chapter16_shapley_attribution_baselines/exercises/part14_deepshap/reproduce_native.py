"""Freshly refit DeepSHAP and compare every deterministic learner result."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from chapter16_shapley_attribution_baselines.exercises.part14_deepshap.run_native import (
    DETERMINISTIC_ARTIFACTS,
    TIMING_ARTIFACT,
    load_payload,
    run,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _numbers_equal(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float) and math.isnan(expected):
        return isinstance(actual, float) and math.isnan(actual)
    return actual == expected


def compare(actual: Any, expected: Any, path: str, counts: dict[str, Any]) -> None:
    """Recursively require equal types, keys, shapes, dtypes, and every array value."""
    if isinstance(expected, np.ndarray):
        if not isinstance(actual, np.ndarray):
            raise ValueError(f"Array type mismatch: {path}")
        if actual.dtype != expected.dtype or actual.shape != expected.shape:
            raise ValueError(f"Array metadata mismatch: {path}")
        if not np.array_equal(actual, expected, equal_nan=True):
            difference = "nonnumeric"
            if np.issubdtype(expected.dtype, np.number):
                difference = float(np.nanmax(np.abs(actual - expected)))
            raise AssertionError(f"Array values differ at {path}; max difference={difference}")
        counts["arrays"] += 1
        counts["array_elements"] += expected.size
        if expected.size and np.issubdtype(expected.dtype, np.number):
            difference = float(np.nanmax(np.abs(actual - expected)))
            counts["max_absolute_error"] = max(counts["max_absolute_error"], difference)
        return
    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor):
            raise ValueError(f"Tensor type mismatch: {path}")
        if actual.dtype != expected.dtype or actual.shape != expected.shape:
            raise ValueError(f"Tensor metadata mismatch: {path}")
        if not torch.equal(actual, expected):
            raise AssertionError(f"Tensor values differ at {path}")
        counts["arrays"] += 1
        counts["array_elements"] += expected.numel()
        if expected.numel() and (expected.is_floating_point() or expected.is_complex()):
            difference = float((actual - expected).abs().max())
            counts["max_absolute_error"] = max(counts["max_absolute_error"], difference)
        return
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or list(actual) != list(expected):
            raise ValueError(f"Dictionary keys/order mismatch: {path}")
        for key in expected:
            compare(actual[key], expected[key], f"{path}/{key}", counts)
        return
    if isinstance(expected, (list, tuple)):
        if type(actual) is not type(expected) or len(actual) != len(expected):
            raise ValueError(f"Sequence mismatch: {path}")
        for index, item in enumerate(expected):
            compare(actual[index], item, f"{path}/{index}", counts)
        return
    if type(actual) is not type(expected) or not _numbers_equal(actual, expected):
        raise ValueError(f"Scalar mismatch: {path}: {actual!r} != {expected!r}")
    counts["scalars"] += 1


def _is_measured_timing_leaf(path: str) -> bool:
    leaf = path.rsplit("/", 1)[-1].lower()
    return any(token in leaf for token in ("second", "timing", "elapsed", "median"))


def compare_timing(actual: Any, expected: Any, path: str, counts: dict[str, Any]) -> None:
    """Compare timing structure/inputs while treating measured durations as descriptive."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or list(actual) != list(expected):
            raise ValueError(f"Timing dictionary mismatch: {path}")
        for key in expected:
            compare_timing(actual[key], expected[key], f"{path}/{key}", counts)
        return
    if isinstance(expected, (list, tuple)):
        if type(actual) is not type(expected) or len(actual) != len(expected):
            raise ValueError(f"Timing sequence mismatch: {path}")
        for index, item in enumerate(expected):
            compare_timing(actual[index], item, f"{path}/{index}", counts)
        return
    if _is_measured_timing_leaf(path):
        if isinstance(expected, np.ndarray):
            if not isinstance(actual, np.ndarray) or actual.dtype != expected.dtype or actual.shape != expected.shape:
                raise ValueError(f"Timing array metadata mismatch: {path}")
            if not np.isfinite(actual).all() or np.any(actual < 0):
                raise ValueError(f"Invalid measured timings: {path}")
            counts["measured_arrays"] += 1
            counts["measured_elements"] += actual.size
            return
        if isinstance(expected, (float, np.floating)):
            if not isinstance(actual, type(expected)) or not np.isfinite(actual) or actual < 0:
                raise ValueError(f"Invalid measured timing: {path}")
            counts["measured_scalars"] += 1
            return
    compare(actual, expected, path, counts["deterministic"])


def reproduce(learner_directory: Path, output_directory: Path) -> dict[str, Any]:
    learner_directory = Path(learner_directory).resolve()
    output_directory = Path(output_directory).resolve()
    learner = json.loads((learner_directory / "execution.json").read_text())
    if learner.get("status") != "PASS" or learner.get("native_model_executed") is not True:
        raise ValueError("Require a passing complete own-answer learner execution.")
    output_directory.mkdir(parents=True, exist_ok=False)
    inputs = dict(learner["source_inputs"])
    inputs[str(Path(__file__).relative_to(ROOT))] = digest(Path(__file__))
    for name, checksum in inputs.items():
        source = ROOT / name
        if not source.is_file() or digest(source) != checksum:
            raise ValueError(f"Own-answer source changed before reproduction: {name}")
        snapshot = output_directory / "source_snapshot" / name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(source.read_bytes())
    for name, record in learner["native_artifacts"].items():
        artifact = learner_directory / name
        if (
            not artifact.is_file()
            or artifact.stat().st_size != record["size_bytes"]
            or digest(artifact) != record["sha256"]
        ):
            raise ValueError(f"Learner artifact changed or is missing: {name}")

    state: dict[str, Any] = {
        "status": "RUNNING",
        "learner_directory": str(learner_directory),
        "learner_execution_sha256": digest(learner_directory / "execution.json"),
        "source_inputs": inputs,
        "native_model_retrained": False,
        "cached_model_inputs_used": False,
        "comparisons": {
            "arrays": 0,
            "array_elements": 0,
            "scalars": 0,
            "max_absolute_error": 0.0,
        },
        "timing_comparison": {
            "measured_arrays": 0,
            "measured_elements": 0,
            "measured_scalars": 0,
            "deterministic": {
                "arrays": 0,
                "array_elements": 0,
                "scalars": 0,
                "max_absolute_error": 0.0,
            },
        },
    }
    started = time.perf_counter()
    try:
        # No learner payload is passed to run: source data is loaded and all seven models refit first.
        fresh_state = run(output_directory / "fresh_run")
        state["native_model_retrained"] = fresh_state["summary"].get("model_fits") == 7
        original = Path(learner["native_directory"])
        fresh = output_directory / "fresh_run"
        expected_names = set(DETERMINISTIC_ARTIFACTS) | {TIMING_ARTIFACT}
        if {path.name for path in original.glob("*.pt")} != expected_names:
            raise ValueError("The complete learner native artifact set is missing or has extras.")
        if {path.name for path in fresh.glob("*.pt")} != expected_names:
            raise ValueError("The complete fresh native artifact set is missing or has extras.")
        for name in DETERMINISTIC_ARTIFACTS:
            compare(
                load_payload(fresh / name),
                load_payload(original / name),
                name,
                state["comparisons"],
            )
        compare_timing(
            load_payload(fresh / TIMING_ARTIFACT),
            load_payload(original / TIMING_ARTIFACT),
            TIMING_ARTIFACT,
            state["timing_comparison"],
        )
        if not state["native_model_retrained"]:
            raise ValueError("Fresh run did not complete all seven native model fits.")
        state.update(
            status="PASS",
            native_execution=fresh_state,
            artifacts={
                str(path.relative_to(output_directory)): {
                    "sha256": digest(path),
                    "size_bytes": path.stat().st_size,
                }
                for path in sorted(fresh.glob("*.pt"))
            },
        )
    except BaseException as error:
        state.update(status="FAIL", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        state["elapsed_seconds"] = time.perf_counter() - started
        state["changed_sources"] = [
            name
            for name, checksum in inputs.items()
            if not (ROOT / name).is_file() or digest(ROOT / name) != checksum
        ]
        if state["changed_sources"]:
            state["status"] = "FAIL"
        (output_directory / "reproduction.json").write_text(json.dumps(state, indent=2) + "\n")
    if state["status"] != "PASS":
        raise ValueError("A source changed during fresh reproduction.")
    print(json.dumps({"status": "PASS", "comparisons": state["comparisons"]}, indent=2))
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--learner-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    reproduce(arguments.learner_directory, arguments.output_directory)
