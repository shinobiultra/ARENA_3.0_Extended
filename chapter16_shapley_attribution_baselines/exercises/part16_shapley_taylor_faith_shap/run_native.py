"""Run the frozen 16.16 interaction study and retain complete, hash-bound evidence.

This module deliberately contains orchestration and persistence only.  The scientific
implementation remains visible in ``interaction_methods.py`` and ``interaction_study.py`` and is
inserted into the learner notebook. ``run`` refuses to fit a model until a reviewed
zero-run protocol freeze exists and still matches its pinned inputs. A development-only
freeze records root resolution of the independent findings without claiming independent approval.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

PACKAGE = "chapter16_shapley_attribution_baselines.exercises.part16_shapley_taylor_faith_shap"
MODEL_KEYS = (
    "clean_0",
    "clean_1",
    "clean_2",
    "shuffled_0",
    "shuffled_1",
    "shuffled_2",
    "ridge",
)
BACKGROUND_SIZES = {
    "primary64": 64,
    "nested16": 16,
    "nested128": 128,
    "independent64_a": 64,
    "independent64_b": 64,
}
SPLIT_ROWS = {"explain": 153, "audit": 119}
COALITIONS = 256
FEATURES = 8
PAIRS = math.comb(FEATURES, 2)
METHOD_KEYS = ("sii", "stii", "fsii")
SMALL_DETERMINISTIC_ARTIFACTS = (
    "fitted.pt",
    "prepared.pt",
    "evaluated.pt",
    "reference.pt",
    "diagnostics.pt",
    "play.pt",
)
TIMING_ARTIFACT = "timings.pt"
HASH_CHUNK_BYTES = 64 * 1024 * 1024
SHARD_SCHEMA_VERSION = 1


def hybrid_artifact_name(split: str, background: str) -> str:
    """Return the frozen physical filename for one complete raw donor-prediction shard."""

    if split not in SPLIT_ROWS:
        raise ValueError(f"Unknown interaction split: {split}")
    if background not in BACKGROUND_SIZES:
        raise ValueError(f"Unknown interaction background: {background}")
    prefix = "primary" if background == "primary64" else background
    return f"{prefix}_{split}_hybrid.pt"


EXTERNAL_DETERMINISTIC_ARTIFACTS = tuple(
    hybrid_artifact_name(split, background)
    for split in ("explain", "audit")
    for background in BACKGROUND_SIZES
)
DETERMINISTIC_ARTIFACTS = SMALL_DETERMINISTIC_ARTIFACTS + EXTERNAL_DETERMINISTIC_ARTIFACTS


def digest(path: Path) -> str:
    """Stream a file SHA256 without materializing a large payload as bytes."""

    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(HASH_CHUNK_BYTES):
            result.update(chunk)
    return result.hexdigest()


def source_paths() -> list[Path]:
    """Return the atomic source set that determines a standalone native result."""

    paths = [
        HERE / name
        for name in (
            "interaction_data.py",
            "interaction_methods.py",
            "interaction_methods_tests.py",
            "interaction_study.py",
            "interaction_study_tests.py",
            "interaction_reference.py",
            "interaction_reference_tests.py",
            "interaction_diagnostics.py",
            "interaction_plots.py",
            "interaction_plots_tests.py",
            "protocol.md",
            "protocol_freeze.json",
            "pre_run_resolution.json",
            "data_manifest.json",
            "run_native.py",
        )
    ]
    amendments = HERE / "implementation_amendments.json"
    if amendments.is_file():
        paths.append(amendments)
    return paths


def validate_freeze() -> dict[str, Any]:
    """Require a zero-native-run freeze over protocol, manifest, and exact data loader."""

    freeze_path = HERE / "protocol_freeze.json"
    if not freeze_path.is_file():
        raise FileNotFoundError(
            "16.16 cannot fit or evaluate before protocol_freeze.json exists."
        )
    freeze = json.loads(freeze_path.read_text())
    expected = freeze.get("sha256", {})
    required = {"protocol.md", "data_manifest.json", "interaction_data.py"}
    if set(expected) != required:
        raise ValueError("The 16.16 freeze must pin protocol, manifest, and data loader.")
    for name, checksum in expected.items():
        if not isinstance(checksum, str) or len(checksum) != 64:
            raise ValueError(f"Invalid frozen SHA256 for {name}.")
        if digest(HERE / name) != checksum:
            raise ValueError(f"Frozen source changed: {name}")
    if freeze.get("native_fits_before_freeze") != 0:
        raise ValueError("The protocol was not frozen before native fitting.")
    if freeze.get("native_game_runs_before_freeze") != 0:
        raise ValueError("The protocol was not frozen before native game evaluation.")
    if freeze.get("native_attribution_runs_before_freeze") != 0:
        raise ValueError("The protocol was not frozen before native interaction attribution.")
    if freeze.get("audit_loads_before_freeze") != 0:
        raise ValueError("The audit split was loaded before the protocol freeze.")
    if freeze.get("independent_approval") is not True:
        if freeze.get("independent_approval") is not False or freeze.get("execution_scope") != "development_only":
            raise ValueError("The zero-run freeze needs independent approval or an explicit development-only resolution.")
        resolution_path = HERE / "pre_run_resolution.json"
        if not resolution_path.is_file() or digest(resolution_path) != freeze.get("pre_run_resolution_sha256"):
            raise ValueError("The development freeze needs its exact pre-run resolution record.")
        resolution = json.loads(resolution_path.read_text())
        findings = resolution.get("findings", [])
        if (
            resolution.get("reviewer") != "root"
            or resolution.get("independent_recheck") != "pending"
            or resolution.get("publication_authorized") is not False
            or len(findings) != 6
            or {item.get("id") for item in findings} != set(range(1, 7))
            or any(item.get("status") != "RESOLVED_ROOT_CHECK" for item in findings)
        ):
            raise ValueError("Development execution requires all six findings resolved and publication still blocked.")
        required_sources = {path.name for path in source_paths()} - {
            "protocol_freeze.json", "pre_run_resolution.json", "implementation_amendments.json"
        }
        source_hashes = dict(resolution.get("source_sha256", {}))
        if not required_sources <= set(source_hashes):
            raise ValueError("The pre-run resolution does not bind every scientific source.")
        amendments_path = HERE / "implementation_amendments.json"
        if amendments_path.is_file():
            amendments = json.loads(amendments_path.read_text())
            if (
                amendments.get("protocol_freeze_sha256") != digest(freeze_path)
                or amendments.get("independent_approval") is not False
                or amendments.get("publication_authorized") is not False
            ):
                raise ValueError("Implementation amendments must preserve the original development freeze and pending review.")
            for entry in amendments.get("entries", []):
                if (
                    entry.get("classification") != "implementation_correction"
                    or not entry.get("reason")
                    or not entry.get("recorded_at_utc")
                    or entry.get("scientific_design_changed") is not False
                    or not entry.get("evidence_files")
                ):
                    raise ValueError("Every implementation correction needs dated evidence and an unchanged scientific design.")
                for path_text, evidence in entry["evidence_files"].items():
                    evidence_path = Path(path_text)
                    if (
                        not evidence_path.is_file()
                        or evidence_path.stat().st_size != evidence.get("size_bytes")
                        or digest(evidence_path) != evidence.get("sha256")
                    ):
                        raise ValueError(f"Implementation correction evidence changed: {path_text}")
                for name, change in entry.get("source_changes", {}).items():
                    if name in required or name not in source_hashes or change.get("before") != source_hashes[name]:
                        raise ValueError(f"Invalid implementation amendment chain: {name}")
                    source_hashes[name] = change.get("after")
        for name, checksum in source_hashes.items():
            if Path(name).name != name or digest(HERE / name) != checksum:
                raise ValueError(f"Resolved pre-run source changed: {name}")
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
    """Hash array bytes in bounded memoryview chunks, including large physical shards."""

    array, _ = _array(value)
    view = memoryview(array).cast("B")
    result = hashlib.sha256()
    for start in range(0, len(view), HASH_CHUNK_BYTES):
        result.update(view[start : start + HASH_CHUNK_BYTES])
    return result.hexdigest()


def _array_identity(value: Any) -> dict[str, Any]:
    array, kind = _array(value)
    return {
        "kind": kind,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "elements": int(array.size),
        "value_sha256": _array_value_sha256(value),
    }


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
        yield {"path": path or "/", **_array_identity(value)}
        return
    if isinstance(value, Mapping):
        keys = list(value)
        if not all(isinstance(key, (str, int)) and not isinstance(key, bool) for key in keys):
            raise TypeError("Deterministic dictionaries require string or integer keys.")
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
    """Describe every payload leaf and bind it to serialized artifact bytes."""

    records = list(_manifest_records(value))
    arrays = [record for record in records if "elements" in record]
    return {
        "schema_version": 1,
        "artifact": Path(artifact).name,
        "artifact_sha256": digest(artifact),
        "size_bytes": Path(artifact).stat().st_size,
        "records": records,
        "counts": {
            "arrays": len(arrays),
            "array_elements": sum(record["elements"] for record in arrays),
            "records": len(records),
        },
    }


def manifest_path(artifact: Path) -> Path:
    return Path(artifact).with_suffix(".manifest.json")


def _write_json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    return path


def _reject_live_modules(value: Any, path: str = "/") -> None:
    if isinstance(value, torch.nn.Module):
        raise TypeError(f"Payload {path} contains a live module; persist complete states instead.")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_live_modules(item, _path(path.rstrip("/"), key))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_live_modules(item, _path(path.rstrip("/"), index))


def _reject_measured_timings(value: Any, path: str = "/") -> None:
    """Keep wall-clock values out of exact fresh-reproduction payloads."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            label = str(key).lower()
            if label in {"timing", "timings"} or label.endswith("_seconds"):
                raise ValueError(f"Measured timing must live only in {TIMING_ARTIFACT}: {_path(path.rstrip('/'), key)}")
            _reject_measured_timings(item, _path(path.rstrip("/"), key))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_measured_timings(item, _path(path.rstrip("/"), index))


def split_measured_timings(
    value: Any, path: str = ""
) -> tuple[Any, dict[str, Any]]:
    """Remove timing leaves without copying arrays and return them by logical path."""

    timings: dict[str, Any] = {}
    if isinstance(value, Mapping):
        clean = {}
        for key, item in value.items():
            child_path = _path(path, key)
            label = str(key).lower()
            if label in {"timing", "timings"} or label.endswith("_seconds"):
                timings[child_path] = item
                continue
            child, found = split_measured_timings(item, child_path)
            clean[key] = child
            timings.update(found)
        return clean, timings
    if isinstance(value, list):
        clean_list = []
        for index, item in enumerate(value):
            child, found = split_measured_timings(item, _path(path, index))
            clean_list.append(child)
            timings.update(found)
        return clean_list, timings
    if isinstance(value, tuple):
        clean_tuple = []
        for index, item in enumerate(value):
            child, found = split_measured_timings(item, _path(path, index))
            clean_tuple.append(child)
            timings.update(found)
        return tuple(clean_tuple), timings
    return value, timings


def save_payload(directory: Path, name: str, value: Any) -> tuple[Path, dict[str, Any]]:
    """Direct-save a course-owned payload, then write its recursive identity."""

    if Path(name).name != name or not name.endswith(".pt"):
        raise ValueError("Payload names must be simple .pt filenames.")
    _reject_live_modules(value)
    if name != TIMING_ARTIFACT:
        _reject_measured_timings(value)
    path = Path(directory) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(value, path)
    manifest = payload_manifest(value, path)
    _write_json(manifest_path(path), manifest)
    return path, manifest


def load_payload(path: Path) -> Any:
    """Load course-owned NumPy/Torch evidence on CPU."""

    return torch.load(path, weights_only=False, map_location="cpu")


def _walk_path(value: Mapping[str, Any], path: Sequence[str]) -> Any:
    item: Any = value
    for key in path:
        if not isinstance(item, Mapping) or key not in item:
            raise KeyError("Missing payload path /" + "/".join(path))
        item = item[key]
    return item


def _replace_paths(value: Mapping[str, Any], replacements: Mapping[tuple[str, ...], Any]) -> dict:
    """Copy only mappings on replaced routes; numerical leaves are never copied."""

    result = dict(value)
    grouped: dict[str, dict[tuple[str, ...], Any]] = {}
    for path, replacement in replacements.items():
        if not path:
            raise ValueError("Cannot replace a payload root.")
        grouped.setdefault(path[0], {})[path[1:]] = replacement
    for key, children in grouped.items():
        if () in children:
            if len(children) != 1:
                raise ValueError(f"Overlapping payload replacements at {key}.")
            result[key] = children[()]
        else:
            original = result.get(key)
            if not isinstance(original, Mapping):
                raise KeyError(f"Missing mapping on replacement route: {key}")
            result[key] = _replace_paths(original, children)
    return result


def _main_hybrid_path(split: str, background: str) -> tuple[str, ...]:
    root = "primary64" if split == "explain" else "primary64_audit"
    if background == "primary64":
        return (root, "hybrid_prediction")
    variants = "background_variants" if split == "explain" else "background_variants_audit"
    return (variants, background, "hybrid_prediction")


def _primary_alias_path(split: str) -> tuple[str, ...]:
    variants = "background_variants" if split == "explain" else "background_variants_audit"
    return (variants, "primary64", "hybrid_prediction")


def _expected_hybrid_shape(split: str, background: str) -> tuple[int, ...]:
    rows, donors = SPLIT_ROWS[split], BACKGROUND_SIZES[background]
    if background == "primary64":
        return (len(MODEL_KEYS), rows, COALITIONS, donors)
    return (rows, COALITIONS, donors)


def _require_array(value: Any, shape: tuple[int, ...], dtype: np.dtype, name: str) -> None:
    array, _ = _array(value)
    if array.shape != shape or array.dtype != dtype:
        raise ValueError(f"{name} must have shape {shape} and dtype {dtype}.")


def _external_reference(
    value: Any,
    manifest: Mapping[str, Any],
    logical_path: Sequence[str],
) -> dict[str, Any]:
    identity = _array_identity(value)
    return {
        "kind": "external_tensor",
        "artifact": manifest["artifact"],
        "manifest": manifest_path(Path(manifest["artifact"])).name,
        "payload_path": "/" + "/".join(logical_path),
        "array_kind": identity.pop("kind"),
        **identity,
        "artifact_sha256": manifest["artifact_sha256"],
        "size_bytes": manifest["size_bytes"],
    }


def _external_view_reference(
    view: Any,
    manifest: Mapping[str, Any],
    logical_path: Sequence[str],
) -> dict[str, Any]:
    identity = _array_identity(view)
    target = (
        ("primary64", "hybrid_prediction")
        if logical_path[0] == "background_variants"
        else ("primary64_audit", "hybrid_prediction")
    )
    return {
        "kind": "external_tensor_view",
        "artifact": manifest["artifact"],
        "manifest": manifest_path(Path(manifest["artifact"])).name,
        "payload_path": "/" + "/".join(logical_path),
        "target_payload_path": "/" + "/".join(target),
        "selector": {"axis": 0, "index": 0},
        "array_kind": identity.pop("kind"),
        **identity,
        "artifact_sha256": manifest["artifact_sha256"],
        "size_bytes": manifest["size_bytes"],
    }


def restore_external_tensor(directory: Path, descriptor: Mapping[str, Any]) -> torch.Tensor:
    """Read and rehash one declared external tensor or model-axis view."""

    required = {
        "kind",
        "artifact",
        "manifest",
        "payload_path",
        "array_kind",
        "shape",
        "dtype",
        "elements",
        "value_sha256",
        "artifact_sha256",
        "size_bytes",
    }
    if not isinstance(descriptor, Mapping) or not required <= set(descriptor):
        raise ValueError("External tensor descriptor is incomplete.")
    if descriptor["kind"] not in {"external_tensor", "external_tensor_view"}:
        raise ValueError("External tensor descriptor has an unknown kind.")
    artifact = Path(directory) / str(descriptor["artifact"])
    if artifact.name != descriptor["artifact"] or not artifact.is_file():
        raise FileNotFoundError("External tensor artifact is missing.")
    if artifact.stat().st_size != descriptor["size_bytes"] or digest(artifact) != descriptor["artifact_sha256"]:
        raise ValueError("External tensor artifact bytes changed.")
    recursive = manifest_path(artifact)
    if recursive.name != descriptor["manifest"] or not recursive.is_file():
        raise ValueError("External tensor recursive manifest is missing.")
    manifest = json.loads(recursive.read_text())
    if (
        manifest.get("artifact") != artifact.name
        or manifest.get("artifact_sha256") != descriptor["artifact_sha256"]
        or manifest.get("size_bytes") != descriptor["size_bytes"]
    ):
        raise ValueError("External tensor recursive manifest changed.")
    value = load_payload(artifact)
    full_identity = _array_identity(value)
    root_record = manifest.get("records", [{}])[0]
    if any(root_record.get(key) != full_identity[key] for key in full_identity):
        raise ValueError("External tensor root manifest differs from read-back value.")
    if descriptor["kind"] == "external_tensor_view":
        if descriptor.get("selector") != {"axis": 0, "index": 0}:
            raise ValueError("Only the frozen clean_0 model-axis view is supported.")
        targets = {
            "primary_explain_hybrid.pt": "/primary64/hybrid_prediction",
            "primary_audit_hybrid.pt": "/primary64_audit/hybrid_prediction",
        }
        expected_target = targets.get(descriptor["artifact"])
        if expected_target is None or descriptor.get("target_payload_path") != expected_target:
            raise ValueError("External tensor view targets the wrong primary shard.")
        value = value.select(0, 0)
    identity = _array_identity(value)
    expected = {
        "kind": descriptor["array_kind"],
        "shape": descriptor["shape"],
        "dtype": descriptor["dtype"],
        "elements": descriptor["elements"],
        "value_sha256": descriptor["value_sha256"],
    }
    if identity != expected:
        raise ValueError("External tensor value identity changed on read-back.")
    return value


def iter_logical_hybrid_parts(
    directory: Path, logical: Mapping[str, Any]
) -> Iterator[tuple[str, torch.Tensor]]:
    """Restore explanation then audit tensors without concatenating or copying them."""

    if logical.get("storage") != "split_external_shards":
        raise ValueError("Logical hybrid mapping must declare split_external_shards.")
    for split in ("explain", "audit"):
        descriptor = logical.get(split)
        if not isinstance(descriptor, Mapping):
            raise ValueError(f"Logical hybrid mapping is missing {split} descriptor.")
        yield split, restore_external_tensor(directory, descriptor)


def _persist_phase_hybrids(
    directory: Path,
    payload: Mapping[str, Any],
    split: str,
) -> tuple[dict, dict[str, dict[str, Any]]]:
    """Save every full raw hybrid leaf once and return a descriptor-only disk payload."""

    replacements: dict[tuple[str, ...], Any] = {}
    manifests: dict[str, dict[str, Any]] = {}
    for background in BACKGROUND_SIZES:
        path = _main_hybrid_path(split, background)
        value = _walk_path(payload, path)
        if isinstance(value, torch.Tensor) and not value.is_contiguous():
            raise ValueError("Physical hybrid shards must be contiguous before persistence.")
        _require_array(
            value,
            _expected_hybrid_shape(split, background),
            np.dtype("float64"),
            "/" + "/".join(path),
        )
        name = hybrid_artifact_name(split, background)
        _, manifest = save_payload(directory, name, value)
        manifests[name] = manifest
        replacements[path] = _external_reference(value, manifest, path)
        if background == "primary64":
            alias_path = _primary_alias_path(split)
            alias = _walk_path(payload, alias_path)
            main_array, _ = _array(value)
            alias_array, _ = _array(alias)
            if alias_array.shape != main_array[0].shape or not np.array_equal(
                alias_array, main_array[0]
            ):
                raise ValueError("The primary64 clean_0 variant must alias the first model shard.")
            replacements[alias_path] = _external_view_reference(alias, manifest, alias_path)
    return _replace_paths(payload, replacements), manifests


def _replace_logical_all_rows(
    evaluated: Mapping[str, Any],
    persisted_prepared: Mapping[str, Any],
) -> dict:
    """Point logical all-row split leaves at existing descriptors without concatenation."""

    replacements: dict[tuple[str, ...], Any] = {}
    for background in BACKGROUND_SIZES:
        explain_path = (
            _main_hybrid_path("explain", background)
            if background != "primary64"
            else _primary_alias_path("explain")
        )
        audit_path = (
            _main_hybrid_path("audit", background)
            if background != "primary64"
            else _primary_alias_path("audit")
        )
        if background == "primary64":
            logical_root = ("all_rows", "background_variants", "primary64", "hybrid_prediction")
        else:
            logical_root = ("all_rows", "background_variants", background, "hybrid_prediction")
        replacements[logical_root + ("explain",)] = _walk_path(
            persisted_prepared, explain_path
        )
        replacements[logical_root + ("audit",)] = _walk_path(evaluated, audit_path)

    primary_logical = ("all_rows", "primary64", "hybrid_prediction")
    replacements[primary_logical + ("explain",)] = _walk_path(
        persisted_prepared, _main_hybrid_path("explain", "primary64")
    )
    replacements[primary_logical + ("audit",)] = _walk_path(
        evaluated, _main_hybrid_path("audit", "primary64")
    )
    return _replace_paths(evaluated, replacements)


def _state_hash_gate(record: Mapping[str, Any]) -> int:
    if record.get("status") != "PASS":
        raise ValueError("All known model-state identities must pass before game evaluation.")
    expected = record.get("expected_model_state_hashes", {})
    actual = record.get("model_state_hashes", {})
    if tuple(expected) != MODEL_KEYS or expected != actual:
        raise ValueError("Known model-state identities differ from the frozen 16.14 evidence.")
    count = sum(len(states) for states in expected.values() if isinstance(states, Mapping))
    mismatches = record.get("mismatches")
    if count != 21 or not isinstance(mismatches, (list, tuple, dict)) or len(mismatches):
        raise ValueError("Exactly 21 matching state identities are required before any game.")
    return count


def write_before_audit(
    directory: Path,
    fitted: Mapping[str, Any],
    prepared: Mapping[str, Any],
) -> dict[str, Any]:
    """Seal fitted, prepared, and every explanation shard before literal audit access."""

    directory = Path(directory)
    if prepared.get("phase") != "prepared_before_audit":
        raise ValueError("Prepared payload must declare phase='prepared_before_audit'.")
    if prepared.get("audit_used_for_selection") is not False:
        raise ValueError("Prepared decisions must explicitly exclude the audit split.")
    reproduction = fitted.get("reproduction", prepared.get("reproduction"))
    if not isinstance(reproduction, Mapping):
        raise ValueError("Fitted evidence must bind the known-model reproduction gate.")
    state_count = _state_hash_gate(reproduction)
    names = ["fitted.pt", "prepared.pt"] + [
        hybrid_artifact_name("explain", background) for background in BACKGROUND_SIZES
    ]
    files = [directory / name for name in names]
    manifests = [manifest_path(path) for path in files]
    if not all(path.is_file() for path in files + manifests):
        raise ValueError("Fitted, prepared, and all explanation shards must exist before audit.")
    boundary = {
        "schema_version": 1,
        "phase": "prepared_before_audit",
        "audit_loaded": False,
        "audit_used_for_selection": False,
        "state_hash_count": state_count,
        "explain_source_ids": _array_identity(prepared["explain_source_ids"]),
        "artifacts": {
            path.name: {
                "sha256": digest(path),
                "size_bytes": path.stat().st_size,
                "manifest": manifest_path(path).name,
                "manifest_sha256": digest(manifest_path(path)),
            }
            for path in files
        },
    }
    _write_json(directory / "before_audit.json", boundary)
    return boundary


def persist_prepared(
    directory: Path,
    fitted: Mapping[str, Any],
    prepared: Mapping[str, Any],
) -> dict[str, Any]:
    """Persist the complete explanation phase and create its physical audit boundary."""

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if "reproduction" not in fitted:
        fitted_for_disk = dict(fitted)
        fitted_for_disk["reproduction"] = prepared.get("reproduction")
    else:
        fitted_for_disk = dict(fitted)
        if fitted_for_disk["reproduction"] != prepared.get("reproduction"):
            raise ValueError("Fitted and prepared reproduction records differ.")
    _state_hash_gate(fitted_for_disk["reproduction"])
    deterministic_prepared, _ = split_measured_timings(prepared)
    persisted_prepared, _ = _persist_phase_hybrids(
        directory, deterministic_prepared, "explain"
    )
    save_payload(directory, "fitted.pt", fitted_for_disk)
    save_payload(directory, "prepared.pt", persisted_prepared)
    return write_before_audit(directory, fitted_for_disk, prepared)


def _verify_boundary(directory: Path) -> dict[str, Any]:
    path = Path(directory) / "before_audit.json"
    if not path.is_file():
        raise ValueError("Prepared evidence must be sealed before the audit split is loaded.")
    boundary = json.loads(path.read_text())
    if boundary.get("audit_loaded") is not False or boundary.get("state_hash_count") != 21:
        raise ValueError("The recorded preparation boundary is invalid.")
    for name, record in boundary.get("artifacts", {}).items():
        artifact = Path(directory) / name
        recursive = manifest_path(artifact)
        if digest(artifact) != record.get("sha256"):
            raise ValueError(f"Pre-audit artifact changed: {name}")
        if digest(recursive) != record.get("manifest_sha256"):
            raise ValueError(f"Pre-audit recursive manifest changed: {name}")
    return boundary


def _inventory_record(directory: Path, name: str, manifest: Mapping[str, Any]) -> dict:
    return {
        "artifact_sha256": manifest["artifact_sha256"],
        "size_bytes": manifest["size_bytes"],
        "manifest": manifest_path(Path(name)).name,
        "manifest_sha256": digest(manifest_path(Path(directory) / name)),
        "external_large_payload": name in EXTERNAL_DETERMINISTIC_ARTIFACTS,
    }


def summarize(
    fitted: Mapping[str, Any],
    prepared: Mapping[str, Any],
    evaluated: Mapping[str, Any],
    reference: Mapping[str, Any],
    manifests: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Build descriptive counts from retained tensors without scientific outcome gates."""

    primary_explain = _walk_path(prepared, _main_hybrid_path("explain", "primary64"))
    primary_audit = _walk_path(evaluated, _main_hybrid_path("audit", "primary64"))
    return {
        "study": "16.16_shapley_taylor_faith_shap",
        "phase": evaluated.get("phase"),
        "model_keys": list(fitted.get("models", {})),
        "model_fits": len(fitted.get("models", {})),
        "known_model_state_hashes": 21,
        "train_rows": int(len(fitted["train_ids"])),
        "explanation_rows": int(len(prepared["explain_source_ids"])),
        "audit_rows": int(len(evaluated["audit_source_ids"])),
        "all_rows": int(len(prepared["explain_source_ids"]) + len(evaluated["audit_source_ids"])),
        "coalitions": COALITIONS,
        "pairs": PAIRS,
        "method_keys": list(METHOD_KEYS),
        "backgrounds": list(BACKGROUND_SIZES),
        "primary_explain_hybrid": _array_identity(primary_explain),
        "primary_audit_hybrid": _array_identity(primary_audit),
        "full_hybrid_coordinates": sum(
            manifest["counts"]["array_elements"]
            for name, manifest in manifests.items()
            if name in EXTERNAL_DETERMINISTIC_ARTIFACTS
        ),
        "audit_used_for_selection": evaluated.get("audit_used_for_selection"),
        "reference_status": reference.get("status"),
        "deterministic_artifacts": list(DETERMINISTIC_ARTIFACTS),
        "manifest_array_count": sum(item["counts"]["arrays"] for item in manifests.values()),
        "manifest_array_elements": sum(
            item["counts"]["array_elements"] for item in manifests.values()
        ),
        "scientific_results_are_descriptive": True,
    }


def persist_completed(
    directory: Path,
    fitted: Mapping[str, Any],
    prepared: Mapping[str, Any],
    evaluated: Mapping[str, Any],
    reference: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    play: Mapping[str, Any],
    timings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist all results after rechecking the immutable explanation boundary."""

    directory = Path(directory)
    _verify_boundary(directory)
    if evaluated.get("phase") != "evaluated_after_audit":
        raise ValueError("Evaluated payload must declare phase='evaluated_after_audit'.")
    if evaluated.get("audit_used_for_selection") is not False:
        raise ValueError("Evaluation must preserve audit_used_for_selection=False.")
    deterministic_reference, reference_timings = split_measured_timings(reference)
    _, prepared_timings = split_measured_timings(prepared)
    persisted_evaluated, audit_manifests = _persist_phase_hybrids(
        directory, evaluated, "audit"
    )
    persisted_prepared = load_payload(directory / "prepared.pt")
    persisted_evaluated = _replace_logical_all_rows(
        persisted_evaluated, persisted_prepared
    )
    manifests = {
        "fitted.pt": json.loads(manifest_path(directory / "fitted.pt").read_text()),
        "prepared.pt": json.loads(manifest_path(directory / "prepared.pt").read_text()),
        **{
            hybrid_artifact_name("explain", background): json.loads(
                manifest_path(directory / hybrid_artifact_name("explain", background)).read_text()
            )
            for background in BACKGROUND_SIZES
        },
        **audit_manifests,
    }
    for name, value in (
        ("evaluated.pt", persisted_evaluated),
        ("reference.pt", deterministic_reference),
        ("diagnostics.pt", diagnostics),
        ("play.pt", play),
    ):
        _, manifests[name] = save_payload(directory, name, value)
    timing_payload = dict(timings or {})
    if "prepared_reference" in timing_payload or "reference" in timing_payload:
        raise ValueError("persist_completed owns the prepared/reference timing branches.")
    timing_payload["prepared_reference"] = prepared_timings
    timing_payload["reference"] = reference_timings
    save_payload(directory, TIMING_ARTIFACT, timing_payload)
    summary = summarize(fitted, prepared, evaluated, deterministic_reference, manifests)
    if summary["model_fits"] != len(MODEL_KEYS):
        raise ValueError("Exactly seven freshly fitted known predictors are required.")
    _write_json(directory / "summary.json", summary)
    inventory = {
        "schema_version": SHARD_SCHEMA_VERSION,
        "deterministic": {
            name: _inventory_record(directory, name, manifests[name])
            for name in DETERMINISTIC_ARTIFACTS
        },
        "timing": {
            "artifact": TIMING_ARTIFACT,
            "artifact_sha256": digest(directory / TIMING_ARTIFACT),
            "manifest": manifest_path(Path(TIMING_ARTIFACT)).name,
            "manifest_sha256": digest(manifest_path(directory / TIMING_ARTIFACT)),
        },
    }
    _write_json(directory / "artifact_manifest.json", inventory)
    expected = set(DETERMINISTIC_ARTIFACTS) | {TIMING_ARTIFACT}
    if {path.name for path in directory.glob("*.pt")} != expected:
        raise ValueError("Native artifact set is incomplete or contains an undeclared payload.")
    return summary


def write_notebook_summary(output_root: Path, native_directory: Path, summary: Mapping[str, Any]) -> Path:
    output_root = Path(output_root).resolve()
    native_directory = Path(native_directory).resolve()
    native_directory.relative_to(output_root)
    return _write_json(
        output_root / "native_summary.json",
        {**summary, "native_directory": str(native_directory)},
    )


def run(
    directory: Path,
    *,
    data_module: Any = None,
    methods: Any = None,
    study: Any = None,
    diagnostics_module: Any = None,
    reference_module: Any = None,
) -> dict[str, Any]:
    """Refit seven models and execute the full frozen study on CPU.

    The explicit sequence below is also the execution contract used by the learner.  In
    particular, ``load_split('audit')`` is textually and dynamically after ``persist_prepared``.
    """

    validate_freeze()
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    paths = source_paths()
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required native sources are missing: " + ", ".join(missing))
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in paths}
    for name in hashes:
        snapshot = directory / "source_snapshot" / name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes((ROOT / name).read_bytes())

    torch.set_num_threads(4)
    data_module = data_module or importlib.import_module(f"{PACKAGE}.interaction_data")
    methods = methods or importlib.import_module(f"{PACKAGE}.interaction_methods")
    study = study or importlib.import_module(f"{PACKAGE}.interaction_study")
    diagnostics_module = diagnostics_module or importlib.import_module(
        f"{PACKAGE}.interaction_diagnostics"
    )
    reference_module = reference_module or importlib.import_module(
        f"{PACKAGE}.interaction_reference"
    )
    state: dict[str, Any] = {
        "status": "RUNNING",
        "source_inputs": hashes,
        "new_training": True,
        "cached_model_inputs_used": False,
        "execution_device": "cpu",
        "model_fits_completed": 0,
        "native_game_started": False,
        "audit_loaded": False,
    }
    phase_times: dict[str, float] = {}
    started = time.perf_counter()
    try:
        train = data_module.load_split("train")
        explain = data_module.load_split("explain")
        phase = time.perf_counter()
        fitted = study.fit_known_models(train)
        phase_times["fit_seconds"] = time.perf_counter() - phase
        state["model_fits_completed"] = len(fitted.get("models", {}))
        reproduction = study.verify_known_reproduction(fitted, data_module.load_manifest())
        _state_hash_gate(reproduction)
        fitted = {**fitted, "reproduction": reproduction}

        state["native_game_started"] = True
        phase = time.perf_counter()
        prepared = study.prepare_study(
            fitted,
            train,
            explain,
            methods=methods,
            reference=reference_module,
            manifest=data_module.load_manifest(),
        )
        phase_times["prepare_seconds"] = time.perf_counter() - phase
        persist_prepared(directory, fitted, prepared)

        # Literal audit boundary: no helper above can receive or load the audit split.
        audit = data_module.load_split("audit")
        state["audit_loaded"] = True
        phase = time.perf_counter()
        evaluated = study.evaluate_study(
            fitted,
            prepared,
            audit,
            methods=methods,
            manifest=data_module.load_manifest(),
        )
        phase_times["evaluate_seconds"] = time.perf_counter() - phase
        reference = reference_module.validate_study_reference(prepared, evaluated)
        diagnostics = diagnostics_module.interaction_diagnostics(prepared, evaluated, reference)
        play = study.native_play(evaluated, methods=methods)
        summary = persist_completed(
            directory,
            fitted,
            prepared,
            evaluated,
            reference,
            diagnostics,
            play,
            timings={
                **phase_times,
                "total_seconds": time.perf_counter() - started,
            },
        )
        state.update(status="PASS", summary=summary)
    except BaseException as error:
        state.update(status="FAIL", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        state["elapsed_seconds"] = time.perf_counter() - started
        state["changed_sources"] = [
            name
            for name, checksum in hashes.items()
            if not (ROOT / name).is_file() or digest(ROOT / name) != checksum
        ]
        if state["changed_sources"]:
            state["status"] = "FAIL"
        _write_json(directory / "run.json", state)
    if state["status"] != "PASS":
        raise ValueError("A native source changed during execution.")
    print(json.dumps({"status": "PASS", "summary": state["summary"]}, indent=2))
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path, required=True)
    run(parser.parse_args().output_directory)
