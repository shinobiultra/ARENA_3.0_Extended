"""Freshly rerun 16.16 in another CPU process and compare every deterministic leaf."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from chapter16_shapley_attribution_baselines.exercises.part16_shapley_taylor_faith_shap.run_native import (
    BACKGROUND_SIZES,
    DETERMINISTIC_ARTIFACTS,
    EXTERNAL_DETERMINISTIC_ARTIFACTS,
    MODEL_KEYS,
    TIMING_ARTIFACT,
    _array_identity,
    _expected_hybrid_shape,
    _main_hybrid_path,
    _primary_alias_path,
    _walk_path,
    digest,
    hybrid_artifact_name,
    load_payload,
    manifest_path,
)


def _load_manifest(directory: Path, name: str) -> dict[str, Any]:
    path = manifest_path(directory / name)
    if not path.is_file():
        raise FileNotFoundError(f"Missing recursive payload manifest: {path}")
    manifest = json.loads(path.read_text())
    artifact = directory / name
    if manifest.get("artifact") != name or not artifact.is_file():
        raise ValueError(f"Manifest does not bind the required artifact: {name}")
    if manifest.get("size_bytes") != artifact.stat().st_size:
        raise ValueError(f"Native artifact size changed: {artifact}")
    if manifest.get("artifact_sha256") != digest(artifact):
        raise ValueError(f"Native artifact bytes changed: {artifact}")
    return manifest


def _expected_descriptor(
    value: Any,
    manifest: Mapping[str, Any],
    logical_path: Sequence[str],
    *,
    view: bool = False,
) -> dict[str, Any]:
    identity = _array_identity(value)
    descriptor = {
        "kind": "external_tensor_view" if view else "external_tensor",
        "artifact": manifest["artifact"],
        "manifest": manifest_path(Path(manifest["artifact"])).name,
        "payload_path": "/" + "/".join(logical_path),
        "array_kind": identity.pop("kind"),
        **identity,
        "artifact_sha256": manifest["artifact_sha256"],
        "size_bytes": manifest["size_bytes"],
    }
    if view:
        descriptor["selector"] = {"axis": 0, "index": 0}
        descriptor["target_payload_path"] = (
            "/primary64/hybrid_prediction"
            if logical_path[0] == "background_variants"
            else "/primary64_audit/hybrid_prediction"
        )
    return descriptor


def _validate_external_descriptors(
    directory: Path,
    manifests: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Read every physical shard and exactly link it to both persisted phase payloads."""

    prepared = load_payload(directory / "prepared.pt")
    evaluated = load_payload(directory / "evaluated.pt")
    identities: dict[str, dict[str, Any]] = {}
    for split, payload in (("explain", prepared), ("audit", evaluated)):
        for background in BACKGROUND_SIZES:
            name = hybrid_artifact_name(split, background)
            value = load_payload(directory / name)
            identity = _array_identity(value)
            if identity["shape"] != list(_expected_hybrid_shape(split, background)):
                raise ValueError(f"Read-back shard shape changed: {name}")
            if identity["dtype"] != "float64":
                raise ValueError(f"Read-back shard dtype changed: {name}")
            root_record = manifests[name].get("records", [{}])[0]
            for key in ("kind", "shape", "dtype", "elements", "value_sha256"):
                if root_record.get(key) != identity.get(key):
                    raise ValueError(f"Read-back shard recursive identity differs: {name}/{key}")
            path = _main_hybrid_path(split, background)
            if _walk_path(payload, path) != _expected_descriptor(
                value, manifests[name], path
            ):
                raise ValueError(f"Persisted payload does not bind external shard: {name}")
            if background == "primary64":
                alias_path = _primary_alias_path(split)
                view = value[0]
                if _walk_path(payload, alias_path) != _expected_descriptor(
                    view, manifests[name], alias_path, view=True
                ):
                    raise ValueError(f"Primary clean_0 alias does not bind first model: {split}")
            identities[name] = identity
            del value
            gc.collect()
    for background in BACKGROUND_SIZES:
        if background == "primary64":
            explain_descriptor = _walk_path(prepared, _primary_alias_path("explain"))
            audit_descriptor = _walk_path(evaluated, _primary_alias_path("audit"))
            logical = _walk_path(
                evaluated,
                ("all_rows", "background_variants", "primary64", "hybrid_prediction"),
            )
        else:
            explain_descriptor = _walk_path(prepared, _main_hybrid_path("explain", background))
            audit_descriptor = _walk_path(evaluated, _main_hybrid_path("audit", background))
            logical = _walk_path(
                evaluated,
                ("all_rows", "background_variants", background, "hybrid_prediction"),
            )
        if logical.get("explain") != explain_descriptor or logical.get("audit") != audit_descriptor:
            raise ValueError(f"Logical all-row descriptors changed: {background}")
    primary_logical = _walk_path(evaluated, ("all_rows", "primary64", "hybrid_prediction"))
    primary_explain = _walk_path(prepared, _main_hybrid_path("explain", "primary64"))
    primary_audit = _walk_path(evaluated, _main_hybrid_path("audit", "primary64"))
    if primary_logical.get("explain") != primary_explain or primary_logical.get("audit") != primary_audit:
        raise ValueError("Logical primary all-row descriptors changed.")
    del prepared, evaluated
    gc.collect()
    return identities


def validate_artifact_directory(directory: Path) -> dict[str, dict[str, Any]]:
    """Validate every physical byte, recursive value identity, descriptor, and seal."""

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
        if record.get("artifact_sha256") != manifests[name]["artifact_sha256"]:
            raise ValueError(f"Combined artifact hash mismatch: {name}")
        if record.get("manifest_sha256") != digest(manifest_path(directory / name)):
            raise ValueError(f"Combined recursive manifest hash mismatch: {name}")
        if record.get("external_large_payload") is not (
            name in EXTERNAL_DETERMINISTIC_ARTIFACTS
        ):
            raise ValueError(f"External-payload designation mismatch: {name}")
    timing = inventory.get("timing", {})
    if (
        timing.get("artifact") != TIMING_ARTIFACT
        or timing.get("artifact_sha256") != manifests[TIMING_ARTIFACT]["artifact_sha256"]
        or timing.get("manifest") != manifest_path(Path(TIMING_ARTIFACT)).name
        or timing.get("manifest_sha256") != digest(manifest_path(directory / TIMING_ARTIFACT))
    ):
        raise ValueError("Combined timing artifact binding changed.")
    _validate_external_descriptors(directory, manifests)

    boundary_path = directory / "before_audit.json"
    if not boundary_path.is_file():
        raise FileNotFoundError("The physical pre-audit seal is missing.")
    boundary = json.loads(boundary_path.read_text())
    expected_explain = {
        "fitted.pt",
        "prepared.pt",
        *(hybrid_artifact_name("explain", name) for name in BACKGROUND_SIZES),
    }
    if (
        boundary.get("phase") != "prepared_before_audit"
        or boundary.get("audit_loaded") is not False
        or boundary.get("state_hash_count") != 21
        or set(boundary.get("artifacts", {})) != expected_explain
    ):
        raise ValueError("The pre-audit seal is incomplete.")
    for name, record in boundary["artifacts"].items():
        if record.get("sha256") != digest(directory / name):
            raise ValueError(f"Sealed artifact bytes changed: {name}")
        if record.get("manifest_sha256") != digest(manifest_path(directory / name)):
            raise ValueError(f"Sealed recursive manifest changed: {name}")
    _validate_timing(load_payload(directory / TIMING_ARTIFACT))
    return manifests


def _manifest_identity(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": manifest.get("schema_version"),
        "artifact": manifest.get("artifact"),
        "records": manifest.get("records"),
        "counts": manifest.get("counts"),
    }


def compare_manifests(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    name: str,
    counts: dict[str, Any],
) -> None:
    """Require exact structure, dtype, shape, scalar, and array value hashes."""

    if _manifest_identity(actual) != _manifest_identity(expected):
        left, right = actual.get("records", []), expected.get("records", [])
        for index, (actual_record, expected_record) in enumerate(zip(left, right)):
            if actual_record != expected_record:
                raise AssertionError(
                    f"Deterministic payload differs at {name} record {index}: "
                    f"{actual_record.get('path')} != {expected_record.get('path')}"
                )
        raise AssertionError(f"Deterministic payload manifest differs: {name}")
    item_counts = expected["counts"]
    counts["arrays"] += item_counts["arrays"]
    counts["array_elements"] += item_counts["array_elements"]
    counts["records"] += item_counts["records"]
    counts["scalars"] += sum(
        record.get("kind") in {"none", "bool", "int", "float", "str", "bytes", "numpy.scalar"}
        for record in expected["records"]
    )


def _validate_timing(value: Any, path: str = "timings") -> dict[str, int]:
    counts = {"measured_scalars": 0, "containers": 0}
    if isinstance(value, Mapping):
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
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"Invalid measured timing at {path}.")
        counts["measured_scalars"] += 1
        return counts
    raise TypeError(f"Unexpected timing metadata at {path}: {type(value).__name__}")


def reproduce(learner_directory: Path, output_directory: Path) -> dict[str, Any]:
    """Run a source-only child refit and compare all deterministic payloads exactly."""

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
        "child_received_learner_payloads": False,
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
        if fresh_state.get("cached_model_inputs_used") is not False:
            raise ValueError("Fresh reproduction used a cached learner model.")
        state["native_model_retrained"] = fresh_state.get("summary", {}).get("model_fits") == len(
            MODEL_KEYS
        )
        fresh_manifests = validate_artifact_directory(fresh_directory)
        for name in DETERMINISTIC_ARTIFACTS:
            compare_manifests(
                fresh_manifests[name], original_manifests[name], name, state["comparisons"]
            )
        state["timing_comparison"] = {
            "original": _validate_timing(load_payload(original / TIMING_ARTIFACT)),
            "fresh": _validate_timing(load_payload(fresh_directory / TIMING_ARTIFACT)),
            "durations_are_descriptive": True,
        }
        if not state["native_model_retrained"]:
            raise ValueError("Fresh run did not refit all seven predictors.")
        state.update(
            status="PASS",
            native_execution=fresh_state,
            artifacts={
                str(path.relative_to(output_directory)): {
                    "sha256": digest(path),
                    "size_bytes": path.stat().st_size,
                }
                for path in sorted(fresh_directory.glob("*.pt"))
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
