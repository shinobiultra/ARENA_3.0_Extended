"""Freshly rerun GradientSHAP and compare every deterministic payload identity."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from chapter16_shapley_attribution_baselines.exercises.part15_gradientshap.run_native import (
    DETERMINISTIC_ARTIFACTS,
    EXTERNAL_DETERMINISTIC_ARTIFACT,
    TIMING_ARTIFACT,
    digest,
    _array_identity,
    manifest_path,
)


def _load_manifest(directory: Path, name: str) -> dict[str, Any]:
    path = manifest_path(directory / name)
    if not path.is_file():
        raise FileNotFoundError(f"Missing recursive payload manifest: {path}")
    manifest = json.loads(path.read_text())
    artifact = directory / name
    if manifest.get("artifact") != name:
        raise ValueError(f"Manifest names the wrong artifact: {path}")
    if not artifact.is_file():
        raise FileNotFoundError(f"Missing native artifact: {artifact}")
    if manifest.get("size_bytes") != artifact.stat().st_size:
        raise ValueError(f"Native artifact size changed: {artifact}")
    if manifest.get("artifact_sha256") != digest(artifact):
        raise ValueError(f"Native artifact bytes changed: {artifact}")
    return manifest


def validate_artifact_directory(directory: Path) -> dict[str, dict[str, Any]]:
    """Validate the full bounded artifact set and its combined inventory."""
    directory = Path(directory).resolve()
    expected = set(DETERMINISTIC_ARTIFACTS) | {TIMING_ARTIFACT}
    if {path.name for path in directory.glob("*.pt")} != expected:
        raise ValueError("Native payload set is missing a required artifact or has an extra.")
    manifests = {name: _load_manifest(directory, name) for name in sorted(expected)}
    inventory_path = directory / "artifact_manifest.json"
    if not inventory_path.is_file():
        raise FileNotFoundError("The combined native artifact manifest is missing.")
    inventory = json.loads(inventory_path.read_text())
    if set(inventory.get("deterministic", {})) != set(DETERMINISTIC_ARTIFACTS):
        raise ValueError("The combined manifest does not bind every deterministic payload.")
    for name in DETERMINISTIC_ARTIFACTS:
        record = inventory["deterministic"][name]
        manifest_file = manifest_path(directory / name)
        if record.get("artifact_sha256") != manifests[name]["artifact_sha256"]:
            raise ValueError(f"Combined artifact hash mismatch: {name}")
        if record.get("manifest_sha256") != digest(manifest_file):
            raise ValueError(f"Combined recursive manifest hash mismatch: {name}")
        expected_external = name == EXTERNAL_DETERMINISTIC_ARTIFACT
        if record.get("external_large_payload") is not expected_external:
            raise ValueError(f"External-payload designation mismatch: {name}")

    # The persistent large artifact is not trusted merely because its serialized bytes match a
    # previously written checksum. Read it back, recompute its array value identity in bounded
    # memoryview chunks, and link that identity to the descriptor stored inside evaluated.pt.
    from chapter16_shapley_attribution_baselines.exercises.part15_gradientshap.run_native import (
        REPLICATE_ELEMENTS,
        REPLICATE_SHAPE,
        load_payload,
    )

    replicate = load_payload(directory / EXTERNAL_DETERMINISTIC_ARTIFACT)
    identity = _array_identity(replicate)
    if identity.get("shape") != list(REPLICATE_SHAPE):
        raise ValueError("External replicate payload has the wrong read-back shape.")
    if identity.get("dtype") != "float64" or identity.get("elements") != REPLICATE_ELEMENTS:
        raise ValueError("External replicate payload has the wrong read-back dtype or size.")
    root_record = manifests[EXTERNAL_DETERMINISTIC_ARTIFACT].get("records", [{}])[0]
    for key in ("kind", "shape", "dtype", "elements", "value_sha256"):
        if root_record.get(key) != identity.get(key):
            raise ValueError(f"External replicate recursive manifest mismatch: {key}")
    del replicate
    gc.collect()

    evaluated = load_payload(directory / "evaluated.pt")
    descriptor = evaluated.get("sampling_replicates", {}).get("samples", {})
    expected_descriptor = {
        "storage": "external_pt",
        "artifact": EXTERNAL_DETERMINISTIC_ARTIFACT,
        "manifest": manifest_path(Path(EXTERNAL_DETERMINISTIC_ARTIFACT)).name,
        "payload_path": "/sampling_replicates/samples",
        **identity,
        "artifact_sha256": manifests[EXTERNAL_DETERMINISTIC_ARTIFACT]["artifact_sha256"],
        "artifact_size_bytes": manifests[EXTERNAL_DETERMINISTIC_ARTIFACT]["size_bytes"],
    }
    if descriptor != expected_descriptor:
        raise ValueError("evaluated.pt does not exactly bind the external replicate payload.")
    del evaluated
    gc.collect()
    return manifests


def _manifest_identity(manifest: dict[str, Any]) -> dict[str, Any]:
    """Return the deterministic value identity, excluding serialization bytes."""
    return {
        "schema_version": manifest.get("schema_version"),
        "artifact": manifest.get("artifact"),
        "records": manifest.get("records"),
        "counts": manifest.get("counts"),
    }


def compare_manifests(
    actual: dict[str, Any],
    expected: dict[str, Any],
    name: str,
    counts: dict[str, Any],
) -> None:
    """Require exact type/structure/shape/dtype/value hashes for all payload leaves."""
    if _manifest_identity(actual) != _manifest_identity(expected):
        actual_records = actual.get("records", [])
        expected_records = expected.get("records", [])
        for index, (left, right) in enumerate(zip(actual_records, expected_records)):
            if left != right:
                raise AssertionError(
                    f"Deterministic payload differs at {name} record {index}: "
                    f"{left.get('path')} != {right.get('path')}"
                )
        raise AssertionError(f"Deterministic payload manifest differs: {name}")
    item_counts = expected["counts"]
    counts["arrays"] += item_counts["arrays"]
    counts["array_elements"] += item_counts["array_elements"]
    counts["records"] += item_counts["records"]
    counts["scalars"] += sum(
        record.get("kind")
        in {"none", "bool", "int", "float", "str", "bytes", "numpy.scalar"}
        for record in expected["records"]
    )


def _validate_timing(value: Any, path: str = "timings") -> dict[str, int]:
    counts = {"measured_scalars": 0, "containers": 0}
    if isinstance(value, dict):
        counts["containers"] += 1
        for key, item in value.items():
            child = _validate_timing(item, f"{path}/{key}")
            counts = {name: counts[name] + child[name] for name in counts}
        return counts
    if isinstance(value, (list, tuple)):
        counts["containers"] += 1
        for index, item in enumerate(value):
            child = _validate_timing(item, f"{path}/{index}")
            counts = {name: counts[name] + child[name] for name in counts}
        return counts
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, (bool, np.bool_)
    ):
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"Invalid measured timing at {path}.")
        counts["measured_scalars"] += 1
        return counts
    raise TypeError(f"Unexpected timing metadata at {path}: {type(value).__name__}")


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

    original = Path(learner["native_directory"]).resolve()
    original.relative_to(learner_directory)
    original_manifests = validate_artifact_directory(original)
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
            "records": 0,
            "scalars": 0,
            "max_absolute_error": 0.0,
            "comparison": "exact_recursive_value_sha256",
        },
    }
    started = time.perf_counter()
    try:
        # No learner payload enters the child. It loads source data and refits all seven predictors
        # in a distinct CPU process, as required by the frozen reproduction protocol.
        fresh_directory = output_directory / "fresh_run"
        process = subprocess.run(
            [
                sys.executable,
                str(HERE / "run_native.py"),
                "--output-directory",
                str(fresh_directory),
            ],
            cwd=ROOT,
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": "",
                "OMP_NUM_THREADS": "4",
                "MKL_NUM_THREADS": "4",
                "OPENBLAS_NUM_THREADS": "4",
            },
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        (output_directory / "fresh_run.log").write_text(process.stdout)
        if process.returncode != 0:
            tail = "\n".join(process.stdout.splitlines()[-40:])
            raise RuntimeError(f"Fresh native child process failed:\n{tail}")
        fresh_state = json.loads((fresh_directory / "run.json").read_text())
        if fresh_state.get("status") != "PASS":
            raise ValueError("Fresh native child process did not record PASS.")
        state["native_model_retrained"] = fresh_state["summary"].get("model_fits") == 7
        fresh = fresh_directory
        fresh_manifests = validate_artifact_directory(fresh)
        for name in DETERMINISTIC_ARTIFACTS:
            compare_manifests(
                fresh_manifests[name],
                original_manifests[name],
                name,
                state["comparisons"],
            )
        from chapter16_shapley_attribution_baselines.exercises.part15_gradientshap.run_native import (
            load_payload,
        )

        state["timing_comparison"] = {
            "original": _validate_timing(load_payload(original / TIMING_ARTIFACT)),
            "fresh": _validate_timing(load_payload(fresh / TIMING_ARTIFACT)),
            "durations_are_descriptive": True,
        }
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
