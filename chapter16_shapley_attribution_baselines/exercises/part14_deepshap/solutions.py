"""Reference answers and the published fresh-CPU DeepSHAP evidence contract."""

from __future__ import annotations

import inspect
import json
import math
from pathlib import Path
import sys
import tempfile
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from chapter16_shapley_attribution_baselines.exercises.part14_deepshap.deepshap_methods import *
from chapter16_shapley_attribution_baselines.exercises.part14_deepshap import (
    deepshap_methods_tests,
)


def run_smoke_test(cpu: bool = True) -> dict:
    """Run all immediate independent method and provided-helper checks."""
    names = []
    for name, function in vars(deepshap_methods_tests).items():
        if name.startswith("test_") and inspect.isfunction(function):
            function()
            names.append(name)
    return {
        "tests_passed": True,
        "accepted": True,
        "test_count": len(names),
        "student_method_exercises": 9,
        "claim_scope": (
            "Rescale propagation, intact-donor games, exact contraction, RevealCancel groups, "
            "replacement curves, and the two provided helper contracts on declared CPU float64 "
            "Sequential models."
        ),
    }


def _array(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _max_abs(value: Any) -> float:
    array = np.abs(_array(value).astype(np.float64, copy=False))
    return float(np.max(array, initial=0.0))


def _numeric_leaves(value: Any, prefix: str = "") -> list[tuple[str, float]]:
    leaves: list[tuple[str, float]] = []
    if isinstance(value, dict):
        for name, item in value.items():
            path = f"{prefix}.{name}" if prefix else str(name)
            leaves.extend(_numeric_leaves(item, path))
    elif isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, (bool, np.bool_)
    ):
        number = float(value)
        if math.isfinite(number):
            leaves.append((prefix, number))
    elif isinstance(value, (np.ndarray, torch.Tensor)) and _array(value).size == 1:
        number = float(_array(value).reshape(-1)[0])
        if math.isfinite(number):
            leaves.append((prefix, number))
    return leaves


def _study_errors(prepared: dict, evaluated: dict) -> dict[str, float]:
    local_accuracy = 0.0
    local_accuracy_passed = True
    exact_efficiency = 0.0
    approximation_gap = 0.0
    ridge_exact = 0.0
    for block in (prepared["models"], evaluated["models"]):
        for key, item in block.items():
            prediction = _array(item["prediction"])
            base = float(_array(item["base"]))
            deep = _array(item["phi_deep"])
            exact = _array(item["phi_exact"])
            game = _array(item["game"])
            reconstructed = base + deep.sum(1)
            local_accuracy = max(local_accuracy, _max_abs(reconstructed - prediction))
            local_accuracy_passed &= bool(
                np.allclose(reconstructed, prediction, atol=1.0e-7, rtol=1.0e-8)
            )
            exact_efficiency = max(
                exact_efficiency,
                _max_abs(game[:, 0] + exact.sum(1) - game[:, -1]),
            )
            approximation_gap = max(approximation_gap, _max_abs(deep - exact))
            if key == "ridge":
                ridge_exact = max(ridge_exact, _max_abs(deep - exact))
    return {
        "real_local_accuracy_passed": local_accuracy_passed,
        "real_local_accuracy_max_error": local_accuracy,
        "exact_efficiency_max_error": exact_efficiency,
        "deepshap_exact_gap_max": approximation_gap,
        "ridge_exact_max_error": ridge_exact,
    }


def _reference_metrics(reference: dict) -> dict[str, Any]:
    counts = reference["counts"]
    passes = reference["passes"]
    errors = [number for _, number in _numeric_leaves(reference.get("max_errors", {}))]
    return {
        "author_reference_passed": reference.get("status") == "PASS"
        and bool(passes)
        and all(value is True for value in passes.values()),
        "author_reference_model_rows": counts["model_rows"],
        "author_reference_local_values": counts["local_values"],
        "author_pair_rows_checked": counts["pair_model_rows"],
        "author_pair_local_values_checked": counts["pair_local_values"],
        "author_reference_max_error": max(errors, default=0.0),
        "author_reference_atol": 1.0e-7,
        "author_reference_rtol": 1.0e-8,
    }


def run_gpu_test(max_vram_gb: float = 0.0) -> dict:
    """Legacy report entry point; refit every DeepSHAP model on CPU without cached weights."""
    from chapter16_shapley_attribution_baselines.exercises.part14_deepshap.build_notebooks import STEM
    from chapter16_shapley_attribution_baselines.exercises.part14_deepshap.execute_learner import (
        digest,
    )
    from chapter16_shapley_attribution_baselines.exercises.part14_deepshap.publish_study import (
        PAGE,
        published_asset_name,
        validate_fresh,
    )
    from chapter16_shapley_attribution_baselines.exercises.part14_deepshap.reproduce_native import (
        reproduce,
    )
    from chapter16_shapley_attribution_baselines.exercises.part14_deepshap.run_native import (
        load_payload,
    )

    here = Path(__file__).parent
    evidence_path = here / "expected_outputs/learner_execution.json"
    if not evidence_path.is_file():
        raise FileNotFoundError("Publish an executed, reproduced, visually reviewed learner first.")
    published = json.loads(evidence_path.read_text())
    for path, key in (
        (here / f"{STEM}_exercises.ipynb", "published_exercises_sha256"),
        (here / f"{STEM}_solutions.ipynb", "published_solutions_sha256"),
        (PAGE, "published_page_sha256"),
    ):
        if not path.is_file() or digest(path) != published[key]:
            raise ValueError("Published learner changed: " + str(path))
    assets = PAGE.parent.parent / "assets"
    for name, checksum in published["published_figures"].items():
        if published_asset_name(name.removeprefix("deepshap_")) != name:
            raise ValueError("Published DeepSHAP figure is not namespaced: " + name)
        if not (assets / name).is_file() or digest(assets / name) != checksum:
            raise ValueError("Published figure changed: " + name)

    learner = Path(published["learner_directory"])
    cache = learner.parent / "report_reproductions"
    cache.mkdir(exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="report-", dir=cache)) / "fresh_cpu"
    reproduce(learner, output)
    execution, fresh = validate_fresh(learner, output)
    fresh_path = here / "expected_outputs/fresh_reproduction.json"
    fresh_path.write_bytes((output / "reproduction.json").read_bytes())

    native = fresh["native_execution"]["summary"]
    native_directory = output / "fresh_run"
    fitted = load_payload(native_directory / "fitted.pt")
    prepared = load_payload(native_directory / "prepared.pt")
    evaluated = load_payload(native_directory / "evaluated.pt")
    diagnostics = load_payload(native_directory / "diagnostics.pt")
    reference = load_payload(native_directory / "reference.pt")
    manifest = json.loads((here / "data_manifest.json").read_text())
    comparisons = fresh["comparisons"]
    study_errors = _study_errors(prepared, evaluated)
    folding_error = max(
        float(item["folding_max_abs_error"]) for item in fitted["models"].values()
    )

    metrics = {
        "cuda_available": False,
        "device": "cpu",
        "peak_vram_gb": 0.0,
        "within_vram_budget": max_vram_gb >= 0.0,
        "actual_learner_passed": True,
        "own_answer_code_cells": execution["code_cells"],
        "embedded_figures": execution["embedded_figures"],
        "native_model_retrained": True,
        "native_model_retrained_in_this_report": True,
        "native_evidence_reused": False,
        "fresh_reproduction_passed": True,
        "fresh_compared_arrays": comparisons["arrays"],
        "fresh_compared_elements": comparisons["array_elements"],
        "fresh_max_error": comparisons["max_absolute_error"],
        "fresh_output_directory": str(output),
        "fresh_elapsed_seconds": fresh["elapsed_seconds"],
        "learner_evidence_sha256": digest(evidence_path),
        "fresh_evidence_sha256": digest(fresh_path),
        "model_fits": native["model_fits"],
        "fresh_model_fits": native["model_fits"],
        "all_model_fits_retained": True,
        "mlp_fits": native["mlp_fits"],
        "neural_training_runs": native["mlp_fits"],
        "ridge_fits": native["ridge_fits"],
        "clean_fits": native["clean_fits"],
        "shuffled_fits": native["shuffled_fits"],
        "shuffled_models_evaluated": native["shuffled_fits"],
        "optimizer_updates": native["optimizer_updates"],
        "model_keys": native["model_keys"],
        "training_seeds": native["training_seeds"],
        "all_training_seeds_retained": True,
        "shuffled_training_targets": True,
        "train_row_count": native["train_rows"],
        "train_group_count": manifest["group_sizes"][0],
        "explanation_group_count": manifest["group_sizes"][1],
        "heldout_unit": "recipe_group",
        "heldout_group_count": manifest["group_sizes"][2],
        "heldout_row_count": native["heldout_row_count"],
        "explanation_row_count": native["explanation_row_count"],
        "rows_explained_per_model": native["rows_explained_per_model"],
        "model_row_explanations": native["model_row_explanations"],
        "final_used_for_selection": native["audit_used_for_selection"],
        "cross_split_recipe_overlap": manifest["cross_split_recipe_overlap"],
        "exact_models": native["exact_models"],
        "exact_coalitions": native["coalitions"],
        "exact_model_rows": native["exact_model_rows"],
        "model_coalition_values": native["coalition_values"],
        "primary_hybrid_predictions": native["primary_hybrid_predictions"],
        "attribution_coordinates_per_method": native["attribution_coordinates_per_method"],
        "pair_model_rows": native["pair_model_rows"],
        "pair_local_values": native["pair_local_values"],
        "background_rows": native["background_rows"],
        "real_local_accuracy_atol": 1.0e-7,
        "real_local_accuracy_rtol": 1.0e-8,
        "deepshap_is_exact_shap": False,
        "deepshap_approximation_disclosed": True,
        "background_variants": native["background_variants"],
        "background_rows_compared": native["rows_explained_per_model"],
        "alternate_backgrounds_compared": True,
        "replacement_models": native["replacement_models"],
        "replacement_audit_rows": native["heldout_row_count"],
        "random_order_controls": native["random_order_controls"],
        "gradient_baseline_model_rows": native["model_row_explanations"],
        "training_mean_baseline": True,
        "folding_max_error": folding_error,
        "whole_paper_replication": False,
        **study_errors,
        **_reference_metrics(reference),
    }
    nearzero = diagnostics["near_zero"]
    cancellation_errors = [
        float(value[name])
        for value in diagnostics["cancellation"]["models"].values()
        for name in (
            "binary_max_abs_error",
            "deep_max_abs_error",
            "exact_max_abs_error",
            "reveal_cancel_max_abs_error",
        )
    ]
    metrics.update(
        nearzero_primary_local_accuracy_passed=False,
        nearzero_primary_failure_disclosed=True,
        nearzero_primary_residual=_max_abs(nearzero["primary_completeness_residual"]),
        nearzero_epsilon_zero_max_error=_max_abs(
            nearzero["exact_secant_completeness_residual"]
        ),
        threshold_exact_max_error=float(diagnostics["threshold"]["fixture_max_abs_error"]),
        min_exact_max_error=max(cancellation_errors, default=0.0),
        dummy_feature_max_abs=float(diagnostics["affine_controls"]["dummy_feature_max_abs"]),
        unsupported_operator_rejected=bool(
            reference["passes"]["reference_model_boundaries"]
        ),
    )
    return metrics


def run_full_experiment(max_vram_gb: float = 0.0) -> dict:
    return {"numerical": run_smoke_test(), "fresh_cpu_model": run_gpu_test(max_vram_gb)}
