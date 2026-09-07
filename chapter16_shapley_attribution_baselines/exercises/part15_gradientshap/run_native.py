"""Run the frozen GradientSHAP study and retain complete, hash-bound evidence."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Iterator

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

PACKAGE = "chapter16_shapley_attribution_baselines.exercises.part15_gradientshap"
DEEP_PACKAGE = "chapter16_shapley_attribution_baselines.exercises.part14_deepshap"
SMALL_DETERMINISTIC_ARTIFACTS = (
    "fitted.pt",
    "prepared.pt",
    "evaluated.pt",
    "reference.pt",
    "diagnostics.pt",
    "play.pt",
)
EXTERNAL_DETERMINISTIC_ARTIFACT = "replicate_samples.pt"
DETERMINISTIC_ARTIFACTS = SMALL_DETERMINISTIC_ARTIFACTS + (
    EXTERNAL_DETERMINISTIC_ARTIFACT,
)
TIMING_ARTIFACT = "timings.pt"
REPLICATE_SHAPE = (64, 3, 64, 2048, 8)
REPLICATE_ELEMENTS = math.prod(REPLICATE_SHAPE)
BUDGETS = (8, 32, 128, 512, 2048)
REPLICATE_MODELS = ("clean_0", "shuffled_0", "ridge")
HASH_CHUNK_BYTES = 64 * 1024 * 1024


def digest(path: Path) -> str:
    """Stream a file SHA256 without materializing a large payload as bytes."""
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(HASH_CHUNK_BYTES):
            result.update(chunk)
    return result.hexdigest()


def source_paths() -> list[Path]:
    """Return every source that determines a standalone native result."""
    local = [
        HERE / name
        for name in (
            "gradientshap_data.py",
            "gradientshap_methods.py",
            "gradientshap_methods_tests.py",
            "gradientshap_study.py",
            "gradientshap_study_tests.py",
            "gradientshap_reference.py",
            "gradientshap_reference_tests.py",
            "gradientshap_diagnostics.py",
            "gradientshap_plots.py",
            "protocol.md",
            "protocol_freeze.json",
            "data_manifest.json",
            "run_native.py",
        )
    ]
    external = [
        ROOT
        / "chapter16_shapley_attribution_baselines/exercises/part14_deepshap/deepshap_methods.py"
    ]
    return local + external


def validate_freeze() -> dict[str, Any]:
    """Require a zero-native-run freeze over protocol, manifest, and data loader."""
    freeze_path = HERE / "protocol_freeze.json"
    if not freeze_path.is_file():
        raise FileNotFoundError("GradientSHAP cannot run before protocol_freeze.json exists.")
    freeze = json.loads(freeze_path.read_text())
    expected = freeze.get("sha256", {})
    required = {"protocol.md", "data_manifest.json", "gradientshap_data.py"}
    if set(expected) != required:
        raise ValueError("The GradientSHAP freeze must pin protocol, manifest, and data loader.")
    for name, checksum in expected.items():
        if not isinstance(checksum, str) or len(checksum) != 64:
            raise ValueError(f"Invalid frozen SHA256 for {name}.")
        if digest(HERE / name) != checksum:
            raise ValueError(f"Frozen source changed: {name}")
    if freeze.get("native_fits_before_freeze") != 0:
        raise ValueError("The protocol was not frozen before native fitting.")
    if freeze.get("native_attribution_runs_before_freeze") != 0:
        raise ValueError("The protocol was not frozen before native attribution.")
    if not freeze.get("frozen_at_utc"):
        raise ValueError("The protocol freeze needs a nonempty UTC timestamp.")
    return freeze


def _array(value: Any) -> tuple[np.ndarray, str]:
    if isinstance(value, np.ndarray):
        array = value
        kind = "numpy.ndarray"
    elif isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise ValueError("Scientific payload tensors must be on CPU before persistence.")
        array = value.detach().numpy()
        kind = "torch.Tensor"
    else:
        raise TypeError(f"Expected NumPy array or Torch tensor, found {type(value).__name__}.")
    if array.dtype.hasobject:
        raise TypeError("Object arrays cannot enter deterministic evidence.")
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)
    return array, kind


def _array_value_sha256(value: Any) -> str:
    """Hash contiguous array bytes in bounded views, including multi-gigabyte arrays."""
    array, _ = _array(value)
    byte_view = memoryview(array).cast("B")
    result = hashlib.sha256()
    for start in range(0, len(byte_view), HASH_CHUNK_BYTES):
        result.update(byte_view[start : start + HASH_CHUNK_BYTES])
    return result.hexdigest()


def _path(parent: str, item: Any) -> str:
    token = str(item).replace("~", "~0").replace("/", "~1")
    return f"{parent}/{token}" if parent else f"/{token}"


def _scalar_record(value: Any) -> dict[str, Any]:
    if isinstance(value, np.generic):
        return {
            "kind": "numpy.scalar",
            "dtype": str(value.dtype),
            "value": _scalar_record(value.item()),
        }
    if value is None:
        return {"kind": "none", "value": None}
    if isinstance(value, bool):
        return {"kind": "bool", "value": value}
    if isinstance(value, int):
        return {"kind": "int", "value": str(value)}
    if isinstance(value, float):
        return {"kind": "float", "value_hex": value.hex()}
    if isinstance(value, str):
        return {"kind": "str", "value": value}
    if isinstance(value, bytes):
        return {
            "kind": "bytes",
            "size_bytes": len(value),
            "value_sha256": hashlib.sha256(value).hexdigest(),
        }
    raise TypeError(f"Unsupported deterministic payload leaf: {type(value).__name__}")


def _manifest_records(value: Any, path: str = "") -> Iterator[dict[str, Any]]:
    if isinstance(value, (np.ndarray, torch.Tensor)):
        array, kind = _array(value)
        yield {
            "path": path or "/",
            "kind": kind,
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "elements": int(array.size),
            "value_sha256": _array_value_sha256(value),
        }
        return
    if isinstance(value, dict):
        keys = list(value)
        if not all(isinstance(key, (str, int)) and not isinstance(key, bool) for key in keys):
            raise TypeError("Deterministic payload dictionaries require string or integer keys.")
        yield {
            "path": path or "/",
            "kind": "dict",
            "keys": [
                {"kind": "str", "value": key}
                if isinstance(key, str)
                else {"kind": "int", "value": str(key)}
                for key in keys
            ],
        }
        for key in keys:
            yield from _manifest_records(value[key], _path(path, key))
        return
    if isinstance(value, (list, tuple)):
        yield {
            "path": path or "/",
            "kind": "tuple" if isinstance(value, tuple) else "list",
            "length": len(value),
        }
        for index, item in enumerate(value):
            yield from _manifest_records(item, _path(path, index))
        return
    yield {"path": path or "/", **_scalar_record(value)}


def payload_manifest(value: Any, artifact: Path) -> dict[str, Any]:
    """Describe every payload leaf and bind it to the serialized artifact bytes."""
    records = list(_manifest_records(value))
    array_records = [record for record in records if "elements" in record]
    return {
        "schema_version": 1,
        "artifact": Path(artifact).name,
        "artifact_sha256": digest(artifact),
        "size_bytes": Path(artifact).stat().st_size,
        "records": records,
        "counts": {
            "arrays": len(array_records),
            "array_elements": sum(record["elements"] for record in array_records),
            "records": len(records),
        },
    }


def manifest_path(artifact: Path) -> Path:
    return Path(artifact).with_suffix(".manifest.json")


def _write_manifest(path: Path, manifest: dict[str, Any]) -> Path:
    path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return path


def _reject_live_modules(value: Any, path: str = "/") -> None:
    if isinstance(value, torch.nn.Module):
        raise TypeError(f"Payload {path} contains a live module; persist complete states instead.")
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_live_modules(item, _path(path.rstrip("/"), key))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_live_modules(item, _path(path.rstrip("/"), index))


def save_payload(directory: Path, name: str, value: Any) -> tuple[Path, dict[str, Any]]:
    """Direct-save a course-owned payload, then write its recursive identity."""
    if Path(name).name != name or not name.endswith(".pt"):
        raise ValueError("Payload names must be simple .pt filenames.")
    _reject_live_modules(value)
    path = Path(directory) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(value, path)
    manifest = payload_manifest(value, path)
    _write_manifest(manifest_path(path), manifest)
    return path, manifest


def load_payload(path: Path) -> Any:
    """Load course-owned NumPy/Torch evidence on CPU."""
    return torch.load(path, weights_only=False, map_location="cpu")


def _array_identity(value: Any) -> dict[str, Any]:
    array, kind = _array(value)
    return {
        "kind": kind,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "elements": int(array.size),
        "value_sha256": _array_value_sha256(value),
    }


def _external_reference(samples: Any, manifest: dict[str, Any]) -> dict[str, Any]:
    identity = _array_identity(samples)
    return {
        "storage": "external_pt",
        "artifact": EXTERNAL_DETERMINISTIC_ARTIFACT,
        "manifest": manifest_path(Path(EXTERNAL_DETERMINISTIC_ARTIFACT)).name,
        "payload_path": "/sampling_replicates/samples",
        **identity,
        "artifact_sha256": manifest["artifact_sha256"],
        "artifact_size_bytes": manifest["size_bytes"],
    }


def _persisted_evaluated(evaluated: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    """Shallow-copy only the route to the external sample leaf; do not copy arrays."""
    result = dict(evaluated)
    replicates = dict(result["sampling_replicates"])
    replicates["samples"] = reference
    result["sampling_replicates"] = replicates
    return result


def _require_array(value: Any, shape: tuple[int, ...], dtype: np.dtype, name: str) -> None:
    array, _ = _array(value)
    if array.shape != shape or array.dtype != dtype:
        raise ValueError(f"{name} must have shape {shape} and dtype {dtype}.")


def _validate_replicate_branch(replicates: dict[str, Any]) -> None:
    """Require every declared schedule, split, estimate, and diagnostic array before save."""
    required = {
        "source_ids",
        "row_split",
        "split_labels",
        "models",
        "model_labels",
        "budgets",
        "replicate_seeds",
        "baseline_index",
        "alpha",
        "samples",
        "phi_prefix",
        "se_prefix",
        "q512",
        "q512_error",
        "coverage_95",
        "coverage_rate",
    }
    if not isinstance(replicates, dict) or not required <= set(replicates):
        raise ValueError("Sampling-replicate evidence is missing required complete branches.")
    r, m, n, k, d = REPLICATE_SHAPE
    p = len(BUDGETS)
    if tuple(replicates["models"]) != REPLICATE_MODELS:
        raise ValueError("Sampling-replicate model order changed.")
    if tuple(replicates["split_labels"]) != ("explanation", "audit"):
        raise ValueError("Sampling-replicate split labels changed.")
    _require_array(replicates["source_ids"], (n,), np.dtype("int64"), "replicate source_ids")
    _require_array(replicates["row_split"], (n,), np.dtype("int64"), "replicate row_split")
    split = _array(replicates["row_split"])[0]
    if not np.array_equal(split, np.r_[np.zeros(n // 2), np.ones(n - n // 2)]):
        raise ValueError("Replicate rows must retain explanation then audit split identity.")
    _require_array(replicates["budgets"], (p,), np.dtype("int64"), "replicate budgets")
    if not np.array_equal(_array(replicates["budgets"])[0], np.asarray(BUDGETS)):
        raise ValueError("Replicate budgets changed from the frozen nested prefixes.")
    _require_array(replicates["replicate_seeds"], (r,), np.dtype("uint64"), "replicate seeds")
    _require_array(replicates["baseline_index"], (r, n, k), np.dtype("int64"), "replicate donors")
    _require_array(replicates["alpha"], (r, n, k), np.dtype("float64"), "replicate alphas")
    _require_array(replicates["samples"], REPLICATE_SHAPE, np.dtype("float64"), "replicate samples")
    prefix_shape = (r, m, p, n, d)
    _require_array(replicates["phi_prefix"], prefix_shape, np.dtype("float64"), "replicate means")
    _require_array(replicates["se_prefix"], prefix_shape, np.dtype("float64"), "replicate SE")
    _require_array(replicates["q512"], (m, n, d), np.dtype("float64"), "replicate q512")
    _require_array(replicates["q512_error"], prefix_shape, np.dtype("float64"), "replicate errors")
    _require_array(replicates["coverage_95"], prefix_shape, np.dtype("bool"), "replicate coverage")
    _require_array(replicates["coverage_rate"], (m, p), np.dtype("float64"), "coverage rates")


def _primary_background_ids(prepared: dict[str, Any]) -> np.ndarray:
    backgrounds = prepared.get("backgrounds", {})
    primary = backgrounds.get("primary64", {})
    for key in ("source_ids", "ids"):
        if key in primary:
            return np.asarray(primary[key], dtype=np.int64)
    if "background_ids" in prepared:
        return np.asarray(prepared["background_ids"], dtype=np.int64)
    raise ValueError("Prepared payload must retain primary64 source IDs.")


def _require_reproduction_gate(prepared: dict[str, Any]) -> None:
    reproduction = prepared.get("reproduction")
    if not isinstance(reproduction, dict) or reproduction.get("status") != "PASS":
        raise ValueError("Prepared attribution requires a passing 16.14 state reproduction gate.")
    expected = reproduction.get("expected_model_state_hashes", {})
    actual = reproduction.get("model_state_hashes", {})
    state_hash_count = sum(
        len(states) for states in expected.values() if isinstance(states, dict)
    ) if isinstance(expected, dict) else 0
    if state_hash_count != 21 or expected != actual:
        raise ValueError("All 21 initial, standardized, and raw model hashes must match exactly.")
    mismatches = reproduction.get("mismatches")
    if not isinstance(mismatches, (list, tuple, dict)) or len(mismatches) != 0:
        raise ValueError("A model-state mismatch must stop before GradientSHAP attribution.")


def write_before_audit(directory: Path, prepared: dict[str, Any]) -> dict[str, Any]:
    """Seal fitted and prepared bytes before the literal audit load."""
    directory = Path(directory)
    if prepared.get("phase") != "prepared_before_audit":
        raise ValueError("Prepared payload must declare phase='prepared_before_audit'.")
    if prepared.get("audit_used_for_selection") is not False:
        raise ValueError("Prepared rankings must explicitly exclude the audit split.")
    _require_reproduction_gate(prepared)
    files = [directory / "fitted.pt", directory / "prepared.pt"]
    manifests = [manifest_path(path) for path in files]
    if not all(path.is_file() for path in files + manifests):
        raise ValueError("Save fitted/prepared payloads and manifests before sealing audit.")
    explain_ids = np.asarray(prepared["explain_ids"], dtype=np.int64)
    background_ids = _primary_background_ids(prepared)
    boundary = {
        "schema_version": 1,
        "phase": "prepared_before_audit",
        "audit_loaded": False,
        "fitted_sha256": digest(files[0]),
        "fitted_manifest_sha256": digest(manifests[0]),
        "prepared_sha256": digest(files[1]),
        "prepared_manifest_sha256": digest(manifests[1]),
        "explain_ids": _array_identity(explain_ids),
        "background_ids": _array_identity(background_ids),
        "state_hash_count": 21,
    }
    (directory / "before_audit.json").write_text(json.dumps(boundary, indent=2) + "\n")
    return boundary


def persist_prepared(directory: Path, fitted: dict, prepared: dict) -> dict[str, Any]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    save_payload(directory, "fitted.pt", fitted)
    save_payload(directory, "prepared.pt", prepared)
    return write_before_audit(directory, prepared)


def _sample_count(blocks: dict[str, Any]) -> int:
    total = 0
    for item in blocks.values():
        mc = item.get("mc", item).get("samples") if isinstance(item, dict) else None
        if isinstance(mc, (np.ndarray, torch.Tensor)):
            total += int(mc.numel() if isinstance(mc, torch.Tensor) else mc.size)
    return total


def summarize(
    fitted: dict[str, Any],
    prepared: dict[str, Any],
    evaluated: dict[str, Any],
    reference: dict[str, Any],
    manifests: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build an index from actual 16.15 payload shapes, without outcome claims."""
    models = fitted["models"]
    replicate = evaluated["sampling_replicates"]["samples"]
    identity = _array_identity(replicate)
    return {
        "study": "16.15_gradientshap_expected_gradients",
        "phase": evaluated.get("phase"),
        "model_keys": list(models),
        "model_fits": len(models),
        "optimizer_updates": sum(
            len(item.get("loss_trace", ()))
            for item in models.values()
            if item.get("kind") != "ridge"
        ),
        "train_rows": len(fitted["train_ids"]),
        "explanation_rows": len(prepared["explain_ids"]),
        "audit_rows": len(evaluated["audit_ids"]),
        "all_model_rows": len(models)
        * (len(prepared["explain_ids"]) + len(evaluated["audit_ids"])),
        "nested_budgets": list(BUDGETS),
        "quadrature_nodes": [256, 512],
        "background_variants": len(evaluated.get("background_variants", {})),
        "noise_variants": sum(
            isinstance(name, str) and name.startswith("sigma_")
            for name in evaluated.get("noise_variants", {})
        ),
        "sampling_replicates": int(identity["shape"][0]),
        "replicate_sample_shape": identity["shape"],
        "replicate_sample_dtype": identity["dtype"],
        "replicate_sample_coordinates": identity["elements"],
        "replicate_sample_sha256": identity["value_sha256"],
        "primary_sample_coordinates": _sample_count(prepared.get("primary", {}))
        + _sample_count(evaluated.get("audit_primary", {})),
        "audit_used_for_selection": prepared["audit_used_for_selection"],
        "reference_status": reference.get("status"),
        "deterministic_artifacts": list(DETERMINISTIC_ARTIFACTS),
        "manifest_array_count": sum(item["counts"]["arrays"] for item in manifests.values()),
        "manifest_array_elements": sum(
            item["counts"]["array_elements"] for item in manifests.values()
        ),
    }


def _verify_boundary(directory: Path) -> dict[str, Any]:
    boundary_path = directory / "before_audit.json"
    if not boundary_path.is_file():
        raise ValueError("Prepared evidence must be sealed before audit evaluation.")
    boundary = json.loads(boundary_path.read_text())
    if boundary.get("audit_loaded") is not False:
        raise ValueError("The recorded preparation boundary is invalid.")
    for name in ("fitted", "prepared"):
        artifact = directory / f"{name}.pt"
        manifest = manifest_path(artifact)
        if digest(artifact) != boundary.get(f"{name}_sha256"):
            raise ValueError(f"{name.title()} evidence changed after the audit boundary.")
        if digest(manifest) != boundary.get(f"{name}_manifest_sha256"):
            raise ValueError(f"{name.title()} manifest changed after the audit boundary.")
    return boundary


def persist_completed(
    directory: Path,
    fitted: dict[str, Any],
    prepared: dict[str, Any],
    evaluated: dict[str, Any],
    diagnostics: dict[str, Any],
    reference: dict[str, Any],
    play: dict[str, Any],
    timings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist all results and externalize only the complete replicate sample tensor."""
    directory = Path(directory)
    _verify_boundary(directory)
    if evaluated.get("phase") != "evaluated_after_audit":
        raise ValueError("Evaluated payload must declare phase='evaluated_after_audit'.")
    if evaluated.get("audit_used_for_selection") is not False:
        raise ValueError("Evaluation must preserve audit_used_for_selection=False.")
    replicates = evaluated.get("sampling_replicates", {})
    _validate_replicate_branch(replicates)
    samples = replicates.get("samples")
    array, _ = _array(samples)
    if array.shape != REPLICATE_SHAPE or array.dtype != np.dtype("float64"):
        raise ValueError(
            f"Replicate samples must be CPU float64 with exact shape {REPLICATE_SHAPE}."
        )

    _, replicate_manifest = save_payload(directory, EXTERNAL_DETERMINISTIC_ARTIFACT, samples)
    external = _external_reference(samples, replicate_manifest)
    persisted_evaluation = _persisted_evaluated(evaluated, external)
    manifests = {
        "fitted.pt": json.loads(manifest_path(directory / "fitted.pt").read_text()),
        "prepared.pt": json.loads(manifest_path(directory / "prepared.pt").read_text()),
        EXTERNAL_DETERMINISTIC_ARTIFACT: replicate_manifest,
    }
    for name, value in (
        ("evaluated.pt", persisted_evaluation),
        ("reference.pt", reference),
        ("diagnostics.pt", diagnostics),
        ("play.pt", play),
    ):
        _, manifests[name] = save_payload(directory, name, value)
    save_payload(directory, TIMING_ARTIFACT, timings or {})
    summary = summarize(fitted, prepared, evaluated, reference, manifests)
    if summary["replicate_sample_coordinates"] != REPLICATE_ELEMENTS:
        raise ValueError("The full declared sampling-replicate array was not retained.")
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    inventory = {
        "schema_version": 1,
        "deterministic": {
            name: {
                "artifact_sha256": manifests[name]["artifact_sha256"],
                "size_bytes": manifests[name]["size_bytes"],
                "manifest": manifest_path(Path(name)).name,
                "manifest_sha256": digest(manifest_path(directory / name)),
                "external_large_payload": name == EXTERNAL_DETERMINISTIC_ARTIFACT,
            }
            for name in DETERMINISTIC_ARTIFACTS
        },
        "timing": {
            "artifact": TIMING_ARTIFACT,
            "artifact_sha256": digest(directory / TIMING_ARTIFACT),
            "manifest": manifest_path(Path(TIMING_ARTIFACT)).name,
            "manifest_sha256": digest(manifest_path(directory / TIMING_ARTIFACT)),
        },
    }
    (directory / "artifact_manifest.json").write_text(json.dumps(inventory, indent=2) + "\n")
    present = {path.name for path in directory.glob("*.pt")}
    expected = set(DETERMINISTIC_ARTIFACTS) | {TIMING_ARTIFACT}
    if present != expected:
        raise ValueError(f"Native artifact set mismatch: {sorted(present)}")
    return summary


def write_notebook_summary(output_root: Path, native_directory: Path, summary: dict) -> Path:
    output_root = Path(output_root).resolve()
    native_directory = Path(native_directory).resolve()
    native_directory.relative_to(output_root)
    payload = dict(summary, native_directory=str(native_directory))
    path = output_root / "native_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def _raw_models(fitted: dict[str, Any], study: Any) -> dict[str, torch.nn.Module]:
    return {
        key: study._model_from_state(record)
        for key, record in fitted["models"].items()
    }


def _reference_inputs(
    fitted: dict[str, Any],
    prepared: dict[str, Any],
    evaluated: dict[str, Any],
    study: Any,
) -> tuple[
    dict[str, torch.nn.Module],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    models = _raw_models(fitted, study)
    queries = evaluated["all_row_x_raw"]
    source_ids = evaluated["query_source_ids"]
    if not isinstance(queries, torch.Tensor) or queries.device.type != "cpu":
        raise TypeError("Reference queries must remain the retained CPU tensor.")
    if queries.dtype != torch.float64:
        raise TypeError("Reference queries must remain float64.")
    if not isinstance(source_ids, torch.Tensor) or source_ids.dtype != torch.int64:
        raise TypeError("Reference source IDs must remain the retained int64 tensor.")
    primary = prepared["backgrounds"]["primary64"]
    for key in ("x_raw", "x", "rows", "raw_x"):
        if key in primary:
            background = primary[key]
            if not isinstance(background, torch.Tensor) or background.device.type != "cpu":
                raise TypeError("Reference background must remain the retained CPU tensor.")
            if background.dtype != torch.float64:
                raise TypeError("Reference background must remain float64.")
            return models, queries, background, source_ids
    raise ValueError("Prepared primary64 background must retain its raw input rows.")


def run(
    directory: Path,
    *,
    methods: Any = None,
    study: Any = None,
    diagnostics_module: Any = None,
    reference_module: Any = None,
    deep_methods: Any = None,
) -> dict[str, Any]:
    """Refit seven known predictors and execute the full frozen study on CPU."""
    validate_freeze()
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    paths = source_paths()
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required native sources are missing: " + ", ".join(missing))
    source_hashes = {str(path.relative_to(ROOT)): digest(path) for path in paths}
    for name in source_hashes:
        snapshot = directory / "source_snapshot" / name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes((ROOT / name).read_bytes())

    torch.set_num_threads(4)
    data = importlib.import_module(f"{PACKAGE}.gradientshap_data")
    methods = methods or importlib.import_module(f"{PACKAGE}.gradientshap_methods")
    study = study or importlib.import_module(f"{PACKAGE}.gradientshap_study")
    diagnostics_module = diagnostics_module or importlib.import_module(
        f"{PACKAGE}.gradientshap_diagnostics"
    )
    reference_module = reference_module or importlib.import_module(
        f"{PACKAGE}.gradientshap_reference"
    )
    deep_methods = deep_methods or importlib.import_module(f"{DEEP_PACKAGE}.deepshap_methods")
    state: dict[str, Any] = {
        "status": "RUNNING",
        "source_inputs": source_hashes,
        "new_training": True,
        "cached_model_reuse": False,
        "execution_device": "cpu",
        "model_fits_completed": 0,
        "native_attribution_started": False,
    }
    started = time.perf_counter()
    phase_times: dict[str, float] = {}
    try:
        train = data.load_split("train")
        explain = data.load_split("explain")
        phase = time.perf_counter()
        fitted = study.fit_models(train)
        phase_times["fit_seconds"] = time.perf_counter() - phase
        state["model_fits_completed"] = len(fitted["models"])
        reproduction = study.verify_16_14_reproduction(fitted, data.load_manifest())
        if reproduction.get("status") != "PASS":
            raise ValueError("Known 16.14 model-state reproduction failed before attribution.")
        # fitted.pt owns the self-contained 21-state comparison and known-section provenance.
        # Keep the validator pure by attaching its returned record in the runner.
        fitted = dict(fitted)
        fitted["reproduction"] = reproduction
        state["native_attribution_started"] = True
        phase = time.perf_counter()
        prepared = study.prepare_study(
            fitted,
            train,
            explain,
            methods=methods,
            deep_methods=deep_methods,
        )
        phase_times["prepare_seconds"] = time.perf_counter() - phase
        _require_reproduction_gate(prepared)
        persist_prepared(directory, fitted, prepared)

        # This literal load occurs only after fitted.pt, prepared.pt, and before_audit.json exist.
        audit = data.load_split("audit")
        phase = time.perf_counter()
        evaluated = study.evaluate_study(
            fitted,
            prepared,
            train,
            explain,
            audit,
            methods=methods,
            deep_methods=deep_methods,
        )
        phase_times["evaluate_seconds"] = time.perf_counter() - phase
        models, queries, background, source_ids = _reference_inputs(
            fitted, prepared, evaluated, study
        )
        phase = time.perf_counter()
        reference = reference_module.validate_reference(
            models,
            queries,
            background,
            source_ids,
            methods,
            samples=128,
        )
        phase_times["reference_seconds"] = time.perf_counter() - phase
        if reference.get("status") != "PASS":
            raise ValueError("Pinned SHAP/Captum same-draw reference validation failed.")
        diagnostics = diagnostics_module.run_diagnostics(methods=methods)
        play = study.native_play(
            fitted,
            prepared,
            evaluated,
            train,
            explain,
            audit,
            methods=methods,
            deep_methods=deep_methods,
        )
        phase_times["total_seconds"] = time.perf_counter() - started
        summary = persist_completed(
            directory,
            fitted,
            prepared,
            evaluated,
            diagnostics,
            reference,
            play,
            timings={"phase_seconds": phase_times},
        )
        state.update(status="PASS", summary=summary)
    except BaseException as error:
        state.update(status="FAIL", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        state["elapsed_seconds"] = time.perf_counter() - started
        state["changed_sources"] = [
            name
            for name, checksum in source_hashes.items()
            if not (ROOT / name).is_file() or digest(ROOT / name) != checksum
        ]
        if state["changed_sources"]:
            state["status"] = "FAIL"
        (directory / "run.json").write_text(json.dumps(state, indent=2) + "\n")
    if state["status"] != "PASS":
        raise ValueError("Sources changed during native execution.")
    print(json.dumps({"status": "PASS", "summary": state["summary"]}, indent=2))
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path, required=True)
    run(parser.parse_args().output_directory)
