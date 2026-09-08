"""Run the frozen TreeSHAP study from fresh CPU fits and retain complete evidence."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

PACKAGE = "chapter16_shapley_attribution_baselines.exercises.part13_treeshap"
DETERMINISTIC_ARTIFACTS = (
    "fitted.pt",
    "prepared.pt",
    "evaluated.pt",
    "diagnostics.pt",
    "reference.pt",
    "play.pt",
)
TIMING_ARTIFACT = "timings.pt"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_paths() -> list[Path]:
    """Every source that determines a standalone native result."""
    return [
        HERE / name
        for name in (
            "treeshap_data.py",
            "treeshap_methods.py",
            "treeshap_methods_tests.py",
            "treeshap_study.py",
            "treeshap_reference.py",
            "treeshap_diagnostics.py",
            "treeshap_plots.py",
            "author_reference.json",
            "reference_boundary_addendum.md",
            "protocol.md",
            "protocol_freeze.json",
            "data_manifest.json",
            "run_native.py",
        )
    ]


def validate_freeze() -> dict[str, Any]:
    """Require all three files named by the pre-fit freeze to remain byte-identical."""
    freeze = json.loads((HERE / "protocol_freeze.json").read_text())
    expected = freeze.get("sha256", {})
    if set(expected) != {"protocol.md", "data_manifest.json", "treeshap_data.py"}:
        raise ValueError("The TreeSHAP freeze must pin protocol, manifest, and data loader.")
    for name, checksum in expected.items():
        if digest(HERE / name) != checksum:
            raise ValueError(f"Frozen source changed: {name}")
    if freeze.get("native_fits_before_freeze") != 0:
        raise ValueError("The protocol was not frozen before native fitting.")
    return freeze


def _without_sklearn(value: Any) -> Any:
    """Copy a nested fitted payload while removing live sklearn estimators."""
    if isinstance(value, dict):
        return {key: _without_sklearn(item) for key, item in value.items() if key != "sklearn"}
    if isinstance(value, list):
        return [_without_sklearn(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_without_sklearn(item) for item in value)
    if isinstance(value, np.ndarray):
        return value.copy()
    return value


def serializable_fitted(fitted: dict) -> dict:
    """Return fitted numerical/tree evidence without live sklearn estimators."""
    return _without_sklearn(fitted)


def save_payload(directory: Path, name: str, value: Any) -> Path:
    """Persist a named course-owned payload; fitted.pt is stripped of sklearn objects."""
    if Path(name).name != name or not name.endswith(".pt"):
        raise ValueError("Payload names must be simple .pt filenames.")
    path = Path(directory) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if name == "fitted.pt":
        value = serializable_fitted(value)
    torch.save(value, path)
    return path


def load_payload(path: Path) -> Any:
    """Load course-owned NumPy payloads; they deliberately use pickle-backed torch.save."""
    return torch.load(path, weights_only=False, map_location="cpu")


def write_before_audit(directory: Path, prepared: dict) -> dict[str, Any]:
    """Seal already-saved fitted/prepared payloads before the audit split is loaded."""
    directory = Path(directory)
    if prepared.get("audit_used_for_selection") is not False:
        raise ValueError("Prepared rankings must explicitly exclude the audit split.")
    fitted_path = directory / "fitted.pt"
    prepared_path = directory / "prepared.pt"
    if not fitted_path.is_file() or not prepared_path.is_file():
        raise ValueError("Save fitted.pt and prepared.pt before sealing the audit boundary.")
    boundary = {
        "phase": "prepared_before_audit",
        "audit_loaded": False,
        "fitted_sha256": digest(fitted_path),
        "prepared_sha256": digest(prepared_path),
        "explain_ids_sha256": hashlib.sha256(
            np.asarray(prepared["explain_ids"]).tobytes()
        ).hexdigest(),
        "background_ids_sha256": hashlib.sha256(
            np.asarray(prepared["background_ids"]).tobytes()
        ).hexdigest(),
    }
    (directory / "before_audit.json").write_text(json.dumps(boundary, indent=2) + "\n")
    return boundary


def persist_prepared(directory: Path, fitted: dict, prepared: dict) -> dict[str, Any]:
    """Save fresh fits and rankings, then seal the pre-audit boundary."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    save_payload(directory, "fitted.pt", fitted)
    save_payload(directory, "prepared.pt", prepared)
    return write_before_audit(directory, prepared)


def _reference_scalars(value: Any, prefix: str = "") -> dict[str, int | float | bool | str]:
    """Flatten scalar reference outcomes for the JSON summary; arrays remain in reference.pt."""
    result: dict[str, int | float | bool | str] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            result.update(_reference_scalars(item, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, (bool, str, int, float, np.bool_, np.integer, np.floating)):
        result[prefix] = value.item() if isinstance(value, np.generic) else value
    return result


def summarize(
    fitted: dict,
    prepared: dict,
    evaluated: dict,
    diagnostics: dict,
    reference: dict,
) -> dict[str, Any]:
    models = fitted["models"]
    component_trees = sum(len(item["ensemble"]["trees"]) for item in models.values())
    exact_keys = [key for key, item in prepared["explanation"].items() if "game_path" in item]
    clean = {key: value for key, value in evaluated["performance"].items() if "_clean_" in key}
    shuffled = {
        key: value for key, value in evaluated["performance"].items() if "_shuffled_" in key
    }
    return {
        "model_fits": len(models),
        "clean_fits": len(clean),
        "shuffled_fits": len(shuffled),
        "component_trees": component_trees,
        "model_families": ["tree", "forest", "boost"],
        "training_seeds": [161310, 161311, 161312],
        "train_rows": len(fitted["train_ids"]),
        "explanation_row_count": len(prepared["explain_ids"]),
        "heldout_row_count": len(evaluated["audit_ids"]),
        "rows_explained_per_model": len(prepared["explain_ids"]) + len(evaluated["audit_ids"]),
        "model_row_explanations": len(models)
        * (len(prepared["explain_ids"]) + len(evaluated["audit_ids"])),
        "coalitions": 256,
        "exact_models": len(exact_keys),
        "exact_games": 2,
        "background_rows": len(prepared["background_ids"]),
        "background_variants": len(evaluated["backgrounds"]),
        "deletion_models": len(evaluated["deletion"]),
        "saabas_models": len(models),
        "random_order_controls": 100,
        "audit_used_for_selection": prepared["audit_used_for_selection"],
        "audit_mse_clean": clean,
        "audit_mse_shuffled": shuffled,
        "training_mean_audit_mse": evaluated["mean_baseline_mse"],
        "diagnostic_groups": sorted(diagnostics),
        "reference_scalars": _reference_scalars(reference),
    }


def persist_completed(
    directory: Path,
    fitted: dict,
    prepared: dict,
    evaluated: dict,
    diagnostics: dict,
    reference: dict,
    play: dict,
) -> dict[str, Any]:
    """Persist post-audit evidence, retaining timing in its own nondeterministic artifact."""
    directory = Path(directory)
    boundary_path = directory / "before_audit.json"
    if not boundary_path.exists():
        raise ValueError("Prepared evidence must be persisted before audit evaluation.")
    boundary = json.loads(boundary_path.read_text())
    if boundary.get("audit_loaded") is not False:
        raise ValueError("The recorded preparation boundary is invalid.")
    if digest(directory / "prepared.pt") != boundary.get("prepared_sha256"):
        raise ValueError("Prepared evidence changed after the audit boundary.")
    if diagnostics.keys() != {"mechanism", "timing"}:
        raise ValueError("Diagnostics must retain separate mechanism and timing branches.")
    save_payload(directory, "evaluated.pt", evaluated)
    save_payload(directory, "diagnostics.pt", diagnostics["mechanism"])
    save_payload(directory, TIMING_ARTIFACT, diagnostics["timing"])
    save_payload(directory, "reference.pt", reference)
    save_payload(directory, "play.pt", play)
    summary = summarize(fitted, prepared, evaluated, diagnostics["mechanism"], reference)
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    expected = set(DETERMINISTIC_ARTIFACTS) | {TIMING_ARTIFACT}
    present = {path.name for path in directory.glob("*.pt")}
    if present != expected:
        raise ValueError(f"Native artifact set mismatch: {sorted(present)}")
    return summary


def write_notebook_summary(output_root: Path, native_directory: Path, summary: dict) -> Path:
    """Write the small JSON index consumed by the fresh-kernel executor."""
    output_root = Path(output_root).resolve()
    native_directory = Path(native_directory).resolve()
    native_directory.relative_to(output_root)
    payload = dict(summary, native_directory=str(native_directory))
    path = output_root / "native_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def run(
    directory: Path,
    *,
    methods=None,
    study=None,
    diagnostics_module=None,
    reference_module=None,
) -> dict[str, Any]:
    """Refit all 18 models and execute the full frozen protocol on CPU."""
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

    data = importlib.import_module(f"{PACKAGE}.treeshap_data")
    methods = methods or importlib.import_module(f"{PACKAGE}.treeshap_methods")
    study = study or importlib.import_module(f"{PACKAGE}.treeshap_study")
    diagnostics_module = diagnostics_module or importlib.import_module(
        f"{PACKAGE}.treeshap_diagnostics"
    )
    reference_module = reference_module or importlib.import_module(f"{PACKAGE}.treeshap_reference")
    state: dict[str, Any] = {
        "status": "RUNNING",
        "source_inputs": source_hashes,
        "new_training": True,
        "cached_model_reuse": False,
        "execution_device": "cpu",
        "model_fits_completed": 0,
    }
    started = time.perf_counter()
    try:
        train = data.load_split("train")
        explain = data.load_split("explain")
        fitted = study.fit_models(train)
        state["model_fits_completed"] = len(fitted["models"])
        prepared = study.prepare_study(fitted, train, explain)
        persist_prepared(directory, fitted, prepared)
        audit = data.load_split("audit")
        evaluated = study.evaluate_study(fitted, prepared, train, explain, audit)
        reference = reference_module.check_reference(
            fitted, prepared, evaluated, explain, audit
        )
        diagnostics = diagnostics_module.run_diagnostics(methods=methods, include_timing=True)
        play = study.native_play(fitted, train, audit)
        summary = persist_completed(
            directory, fitted, prepared, evaluated, diagnostics, reference, play
        )
        state.update(status="PASS", summary=summary)
    except BaseException as error:
        state.update(status="FAIL", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        state["elapsed_seconds"] = time.perf_counter() - started
        state["changed_sources"] = [
            name for name, checksum in source_hashes.items() if digest(ROOT / name) != checksum
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
