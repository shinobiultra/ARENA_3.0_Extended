"""Reference answers and the fresh-CPU Expected Gradients report entry point."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
import sys
import tempfile
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from chapter16_shapley_attribution_baselines.exercises.part15_gradientshap.gradientshap_methods import *
from chapter16_shapley_attribution_baselines.exercises.part15_gradientshap import (
    gradientshap_methods_tests,
)


def run_smoke_test(cpu: bool = True) -> dict:
    """Run the nine independent immediate learner tests without fitting a model."""
    names = []
    for name, function in vars(gradientshap_methods_tests).items():
        if name.startswith("test_") and inspect.isfunction(function):
            function()
            names.append(name)
    return {
        "tests_passed": True,
        "accepted": True,
        "test_count": len(names),
        "student_method_exercises": 9,
        "claim_scope": "Analytic tests of explicit Expected Gradients schedules, paths, gradients, "
        "ordinary Monte Carlo SE, approximate quadrature, noisy targets and replacement curves.",
    }


def _array(value: Any) -> np.ndarray:
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def _shape(value: Any, expected: tuple[int, ...], name: str) -> np.ndarray:
    array = _array(value)
    if array.shape != expected:
        raise ValueError(f"Incomplete {name}: expected {expected}, got {array.shape}.")
    return array


def _study_metrics(fitted: dict, prepared: dict, evaluated: dict, manifest: dict) -> dict:
    """Measure coverage and descriptive errors from complete retained arrays."""
    from chapter16_shapley_attribution_baselines.exercises.part15_gradientshap.gradientshap_study import (
        MODEL_KEYS,
        verify_16_14_reproduction,
    )

    torch.set_num_threads(4)
    reproduced = verify_16_14_reproduction(fitted, manifest)
    models = fitted["models"]
    if tuple(evaluated["all_rows"]) != MODEL_KEYS:
        raise ValueError("Every frozen model must retain its full all-row result.")
    primary_coordinates = coalition_values = hybrid_predictions = 0
    q_coordinates = {256: 0, 512: 0}
    descriptive = {"eg_exact_gap_max": 0.0, "quadrature_drift_max": 0.0, "mc_residual_max": 0.0}
    for key, block in evaluated["all_rows"].items():
        primary_coordinates += _shape(block["mc"]["samples"], (272, 2048, 8), key).size
        phi = _shape(block["mc"]["phi_prefix"], (5, 272, 8), f"{key}.prefix")
        _shape(block["mc"]["se_prefix"], (5, 272, 8), f"{key}.se")
        base = block["baselines"]
        coalition_values += _shape(base["coalition_values"], (272, 256), key).size
        hybrid_predictions += _shape(base["hybrid_prediction"], (272, 256, 64), key).size
        exact = _shape(base["phi_exact"], (272, 8), f"{key}.exact")
        for nodes in q_coordinates:
            q_coordinates[nodes] += _shape(
                block["quadrature"][f"pair_{nodes}"], (272, 64, 8), f"{key}.q{nodes}"
            ).size
        descriptive["eg_exact_gap_max"] = max(
            descriptive["eg_exact_gap_max"], float(np.abs(phi[-1] - exact).max())
        )
        drift = _array(block["quadrature"]["drift"])
        descriptive["quadrature_drift_max"] = max(
            descriptive["quadrature_drift_max"], float(np.abs(drift).max())
        )
        residual = _array(block["mc"]["completeness_residual_prefix"])
        descriptive["mc_residual_max"] = max(
            descriptive["mc_residual_max"], float(np.abs(residual).max())
        )
    replicas = evaluated["sampling_replicates"]
    external = replicas["samples"]
    if external.get("storage") != "external_pt" or external.get("shape") != [64, 3, 64, 2048, 8]:
        raise ValueError("The complete replicate sample tensor must be externally bound.")
    _shape(replicas["baseline_index"], (64, 64, 2048), "replicate donors")
    _shape(replicas["alpha"], (64, 64, 2048), "replicate alpha")
    seeds = _array(replicas["replicate_seeds"]).tolist()
    if seeds != manifest["sampling_design_candidate"]["replicate_seeds"]:
        raise ValueError("Replicate seeds differ from the frozen manifest.")
    for name, block in evaluated["background_variants"].items():
        donors = len(block["background_source_ids"])
        _shape(block["mc"]["samples"], (272, 2048, 8), f"{name}.samples")
        _shape(block["baselines"]["hybrid_prediction"], (272, 256, donors), f"{name}.hybrids")
        for nodes in (256, 512):
            _shape(block["quadrature"][f"pair_{nodes}"], (272, donors, 8), f"{name}.q{nodes}")
    for name, block in evaluated["noise_variants"].items():
        if name.startswith("sigma_"):
            _shape(block["samples"], (272, 2048, 8), f"{name}.samples")
            _shape(block["noisy_prediction_samples"], (272, 2048), f"{name}.predictions")
    random_curves = evaluated["replacement_curves"]["random"]
    _shape(random_curves["insertion"], (7, 119, 100, 9), "random insertion")
    _shape(random_curves["deletion"], (7, 119, 100, 9), "random deletion")
    distances = evaluated["distance_diagnostic"]
    return {
        "study": "16.15_gradientshap_expected_gradients",
        "model_fits": len(models),
        "fresh_model_fits": len(models),
        "model_keys": list(models),
        "mlp_fits": sum(item["kind"] == "mlp" for item in models.values()),
        "shuffled_fits": sum(item["control"] == "shuffled" for item in models.values()),
        "optimizer_updates": sum(len(v["loss_trace"]) for v in models.values() if v["kind"] == "mlp"),
        "known_model_state_hashes_matched": sum(len(v) for v in reproduced["model_state_hashes"].values()),
        "train_row_count": len(fitted["train_ids"]),
        "explanation_row_count": len(prepared["explain_ids"]),
        "heldout_row_count": len(evaluated["audit_ids"]),
        "cross_split_recipe_overlap": manifest["cross_split_recipe_overlap"],
        "final_used_for_selection": evaluated["audit_used_for_selection"],
        "model_row_explanations": sum(len(v["mc"]["source_ids"]) for v in evaluated["all_rows"].values()),
        "primary_sample_coordinates": int(primary_coordinates),
        "model_coalition_values": int(coalition_values),
        "primary_hybrid_predictions": int(hybrid_predictions),
        "quadrature_pair_coordinates": {str(k): int(v) for k, v in q_coordinates.items()},
        "nested_budgets": _array(replicas["budgets"]).tolist(),
        "replicate_sample_coordinates": external["elements"],
        "replicate_sample_shape": external["shape"],
        "replicate_sample_sha256": external["value_sha256"],
        "replicate_file_sha256": external["artifact_sha256"],
        "replicate_file_size_bytes": external["artifact_size_bytes"],
        "replicate_seeds_retained": len(seeds),
        "background_variants": len(evaluated["background_variants"]),
        "noise_variants": sum(name.startswith("sigma_") for name in evaluated["noise_variants"]),
        "random_order_controls": len(prepared["random_orders"]),
        "replacement_models": len(evaluated["replacement_curves"]) - 1,
        "path_distance_points": int(_array(distances["path"]["distance"]).size),
        "coalition_distance_points": int(_array(distances["coalition"]["distance"]).size),
        "quadrature_is_approximate": True,
        "finite_mc_completeness_is_acceptance_gate": False,
        "eg_is_exact_feature_shapley": False,
        "positive_noise_changes_target": True,
        "whole_paper_replication": False,
        **descriptive,
    }


def _parity_stats(result: dict) -> tuple[float, float, float]:
    """Recheck retained author vectors against the actual sampled contributions."""
    author = _array(result["author_phi"])
    student = _array(result["student_phi"])
    contributions = _array(result["student_contributions"])
    if author.shape != student.shape or contributions.shape[::2] != author.shape:
        raise ValueError("Reference vectors and retained samples have incompatible shapes.")
    if not np.allclose(student, contributions.mean(1), atol=1e-10, rtol=1e-10):
        raise ValueError("Retained student vector is not the mean of its contributions.")
    # Match the adapter's allclose(author, student) convention exactly.
    error = np.abs(author - student)
    bounds = 1e-10 + 1e-10 * np.abs(student)
    scaled = error / bounds
    if not np.isfinite(scaled).all() or float(scaled.max()) > 1.0:
        raise ValueError("Retained same-draw reference vectors fail the frozen tolerance.")
    return float(error.max()), float(scaled.max()), float(bounds.max())


def _reference_metrics(reference: dict) -> dict:
    from chapter16_shapley_attribution_baselines.exercises.part15_gradientshap.gradientshap_study import MODEL_KEYS

    totals, errors, scaled_errors, bounds = {}, {}, {}, {}
    for name, package in reference["packages"].items():
        if tuple(package["models"]) != MODEL_KEYS:
            raise ValueError("Every author package must retain all seven frozen models.")
        rows = coordinates = 0
        results = []
        for key, item in package["models"].items():
            phi = _shape(item["author_phi"], (272, 8), f"{name}.{key}.author")
            _shape(item["student_contributions"], (272, 128, 8), f"{name}.{key}.samples")
            results.append(_parity_stats(item))
            rows += phi.shape[0]
            coordinates += phi.size
        totals[name] = (rows, coordinates)
        errors[name], scaled_errors[name], bounds[name] = map(max, zip(*results))
    if set(totals) != {"shap", "captum"} or totals["shap"] != totals["captum"]:
        raise ValueError("Both packages must retain equal complete coverage.")
    positive = reference["positive_noise"]
    for package in positive["packages"].values():
        _parity_stats(package["same_draw"])
        if not np.any(_array(package["same_draw"]["draws"]["noise"]) != 0):
            raise ValueError("The positive-noise reference must actually retain noise.")
    return {
        "author_reference_passed": reference["status"] == "PASS" and all(reference["passes"].values()),
        "author_reference_versions": reference["versions"],
        "author_samples_per_package": _array(reference["packages"]["shap"]["models"]["clean_0"]["student_contributions"]).shape[1],
        "author_model_rows_per_package": totals["shap"][0],
        "author_total_vectors": sum(value[0] for value in totals.values()),
        "author_total_coordinates": sum(value[1] for value in totals.values()),
        "author_reference_atol": reference["tolerances"]["same_draw_atol"],
        "author_reference_rtol": reference["tolerances"]["same_draw_rtol"],
        "author_shap_max_error": errors["shap"],
        "author_captum_max_error": errors["captum"],
        "author_shap_scaled_error": scaled_errors["shap"],
        "author_captum_scaled_error": scaled_errors["captum"],
        "author_shap_tolerance_bound": bounds["shap"],
        "author_captum_tolerance_bound": bounds["captum"],
        "package_draw_tables_distinct": reference["passes"]["package_draw_tables_distinct"],
        "positive_noise_reference_passed": positive["status"] == "PASS",
    }


def run_gpu_test(max_vram_gb: float = 0.0) -> dict:
    """Legacy entry name: refit all seven predictors on CPU and verify every result."""
    from chapter16_shapley_attribution_baselines.exercises.part15_gradientshap.build_notebooks import STEM
    from chapter16_shapley_attribution_baselines.exercises.part15_gradientshap.publish_study import (
        PAGE,
        published_asset_name,
        validate_fresh,
    )
    from chapter16_shapley_attribution_baselines.exercises.part15_gradientshap.reproduce_native import reproduce
    from chapter16_shapley_attribution_baselines.exercises.part15_gradientshap.run_native import digest, load_payload

    here = Path(__file__).parent
    evidence_path = here / "expected_outputs/learner_execution.json"
    if not evidence_path.is_file():
        raise FileNotFoundError("Publish the executed, reproduced and visually reviewed learner first.")
    published = json.loads(evidence_path.read_text())
    for path, field in (
        (here / f"{STEM}_exercises.ipynb", "published_exercises_sha256"),
        (here / f"{STEM}_solutions.ipynb", "published_solutions_sha256"),
        (PAGE, "published_page_sha256"),
    ):
        if not path.is_file() or digest(path) != published[field]:
            raise ValueError(f"Published learner changed: {path}")
    assets = PAGE.parent.parent / "assets"
    for name, checksum in published["published_figures"].items():
        if published_asset_name(name.removeprefix("gradientshap_")) != name:
            raise ValueError(f"Published figure has an invalid namespace: {name}")
        if not (assets / name).is_file() or digest(assets / name) != checksum:
            raise ValueError(f"Published figure changed: {name}")
    learner = Path(published["learner_directory"])
    cache = learner.parent / "report_reproductions"
    cache.mkdir(exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="report-", dir=cache)) / "fresh_cpu"
    reproduce(learner, output)
    execution, fresh = validate_fresh(learner, output)
    fresh_path = here / "expected_outputs/fresh_reproduction.json"
    fresh_path.write_bytes((output / "reproduction.json").read_bytes())
    native_directory = output / "fresh_run"
    fitted = load_payload(native_directory / "fitted.pt")
    prepared = load_payload(native_directory / "prepared.pt")
    evaluated = load_payload(native_directory / "evaluated.pt")
    reference = load_payload(native_directory / "reference.pt")
    manifest = json.loads((here / "data_manifest.json").read_text())
    original_native = Path(execution["native_directory"])
    original_fitted = load_payload(original_native / "fitted.pt")
    original_summary = json.loads((learner / "figures/native_summary.json").read_text())
    if len(original_fitted["models"]) != original_summary["model_fits"]:
        raise ValueError("Original learner model count disagrees with its retained fitted payload.")
    compared = fresh["comparisons"]
    return {
        "cuda_available": False,
        "device": "cpu",
        "peak_vram_gb": 0.0,
        "within_vram_budget": max_vram_gb >= 0.0,
        "actual_learner_passed": execution["status"] == "PASS",
        "own_answer_code_cells": execution["code_cells"],
        "embedded_figures": execution["embedded_figures"],
        "native_model_retrained": fresh["native_model_retrained"],
        "native_model_retrained_in_this_report": (
            fresh["native_model_retrained"] is True
            and fresh["cached_model_inputs_used"] is False
            and fresh["native_execution"]["summary"]["model_fits"] == len(fitted["models"])
        ),
        "native_evidence_reused": fresh["cached_model_inputs_used"],
        "fresh_reproduction_passed": fresh["status"] == "PASS",
        "fresh_compared_arrays": compared["arrays"],
        "fresh_compared_elements": compared["array_elements"],
        "fresh_max_error": compared["max_absolute_error"],
        "fresh_comparison": compared["comparison"],
        "fresh_output_directory": str(output),
        "fresh_elapsed_seconds": fresh["elapsed_seconds"],
        "learner_evidence_sha256": digest(evidence_path),
        "fresh_evidence_sha256": digest(fresh_path),
        **_study_metrics(fitted, prepared, evaluated, manifest),
        **_reference_metrics(reference),
        "model_fits": len(original_fitted["models"]),
    }


def run_full_experiment(max_vram_gb: float = 0.0) -> dict:
    return {"numerical": run_smoke_test(), "fresh_cpu_model": run_gpu_test(max_vram_gb)}
