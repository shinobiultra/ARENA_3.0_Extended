"""Reference answers and the fresh-CPU Shapley interaction report entry point.

The teaching implementation is inserted into the notebooks. This file measures
the retained evidence for the repository report; it does not supply the learner's
scientific outputs or grade which interaction definition wins.
"""

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
from chapter16_shapley_attribution_baselines.exercises.part16_shapley_taylor_faith_shap.interaction_methods import *
from chapter16_shapley_attribution_baselines.exercises.part16_shapley_taylor_faith_shap import interaction_methods_tests


def run_smoke_test(cpu: bool = True) -> dict:
    """Run immediate semantic checks without loading data or fitting a model."""
    names = []
    for name, function in vars(interaction_methods_tests).items():
        if name.startswith("test_") and inspect.isfunction(function):
            function()
            names.append(name)
    return {
        "tests_passed": True,
        "accepted": True,
        "test_count": len(names),
        "student_core_functions": 10,
        "student_method_exercises": 8,
        "claim_scope": "Analytic checks of Mobius dividends, Shapley-Taylor, exact constrained "
        "Faith-Shap and its closed form, reconstruction, complete donor games and context derivatives.",
    }


def _array(value: Any) -> np.ndarray:
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def _shape(value: Any, expected: tuple[int, ...], name: str) -> np.ndarray:
    array = _array(value)
    if array.shape != expected:
        raise ValueError(f"Incomplete {name}: expected {expected}, got {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError(f"Nonfinite retained values in {name}.")
    return array


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"Invalid SHA256 for {name}.")
    return value


def _study_metrics(fitted: dict, prepared: dict, evaluated: dict, manifest: dict) -> dict:
    """Measure full study coverage from arrays; scientific disagreements remain descriptive."""
    from chapter16_shapley_attribution_baselines.exercises.part16_shapley_taylor_faith_shap.interaction_study import (
        BACKGROUND_SPECS, MODEL_KEYS, METHOD_KEYS, verify_known_reproduction,
    )

    reproduced = verify_known_reproduction(fitted, manifest)
    if tuple(evaluated["model_keys"]) != MODEL_KEYS or tuple(evaluated["method_keys"]) != METHOD_KEYS:
        raise ValueError("Retained model/method axes differ from the frozen order.")
    for name, ids, count in (
        ("train", fitted["train_ids"], 758),
        ("explain", prepared["explain_source_ids"], 153),
        ("audit", evaluated["audit_source_ids"], 119),
    ):
        observed = _shape(ids, (count,), f"{name} IDs")
        if observed.tolist() != manifest["splits"][name]:
            raise ValueError(f"Retained {name} identities differ from the manifest.")
    if prepared["audit_used_for_selection"] is not False or evaluated["audit_used_for_selection"] is not False:
        raise ValueError("Audit outcomes changed a prepared selection.")
    source_ids = _shape(evaluated["source_ids"], (272,), "all query IDs")
    if source_ids.tolist() != manifest["splits"]["explain"] + manifest["splits"]["audit"]:
        raise ValueError("All-row order must preserve explanation then audit.")

    primary = evaluated["all_rows"]["primary64"]
    values = _shape(primary["coalition_values"], (7, 272, 256), "primary games")
    for name in ("mobius", "stii_coefficients", "fsii_coefficients", "fsii_closed_form"):
        _shape(primary[name], values.shape, name)
    _shape(primary["ordinary_shapley"], (7, 272, 8), "ordinary Shapley")
    _shape(primary["sii_pairs"], (7, 272, 28), "SII pairs")
    _shape(primary["pair_values"], (7, 272, 3, 28), "all method pairs")
    contexts = _shape(primary["context_synergy"], (7, 272, 28, 64), "all pair contexts")
    reconstruction = _shape(primary["reconstruction"], (2, 7, 272, 256), "reconstruction")
    residual = _shape(primary["residual"], reconstruction.shape, "reconstruction residual")
    if not np.array_equal(residual, values[None] - reconstruction):
        raise ValueError("Retained reconstruction residuals do not match their arrays.")
    kernel_mse = _shape(primary["kernel_mse"], (2, 7, 272), "weighted errors")
    _shape(primary["rmse_by_size"], (2, 7, 272, 9), "errors by coalition size")
    efficiency = _shape(primary["efficiency_residual"], (2, 7, 272), "efficiency")
    if tuple(evaluated["all_rows"]["background_variants"]) != tuple(BACKGROUND_SPECS):
        raise ValueError("All five frozen backgrounds must be retained in order.")
    alternate_games = 0
    for name, block in evaluated["all_rows"]["background_variants"].items():
        game = _shape(block["coalition_values"], (272, 256), f"{name} games")
        _shape(block["pair_values"], (272, 3, 28), f"{name} pairs")
        _shape(block["context_synergy"], (272, 28, 64), f"{name} contexts")
        if name == "primary64":
            if not np.array_equal(game, values[0]):
                raise ValueError("Primary background alias differs from the clean_0 game.")
        else:
            alternate_games += len(game)
    direct = evaluated["direct_full_context"]
    direct_four = _shape(direct["four_values"], (7, 272, 28, 4), "direct four-forward values")
    direct_synergy = _shape(direct["synergy"], (7, 272, 28), "direct synergy")
    if not np.array_equal(direct_synergy, direct_four[..., 3] - direct_four[..., 1] - direct_four[..., 2] + direct_four[..., 0]):
        raise ValueError("Direct four-forward differences do not match their retained values.")

    order = _shape(prepared["rankings"]["pair_order"], (7, 3, 28), "frozen pair orders")
    if not np.array_equal(np.sort(order, axis=-1), np.broadcast_to(np.arange(28), order.shape)):
        raise ValueError("Every explanation pair order must be a full permutation.")
    random_order = _shape(prepared["random_pair_orders"]["orders"], (256, 28), "random pair orders")
    if not np.array_equal(np.sort(random_order, axis=-1), np.broadcast_to(np.arange(28), random_order.shape)):
        raise ValueError("Every random pair order must be a full permutation.")
    capture = evaluated["pair_capture"]
    _shape(capture["capture_curves"], (7, 3, 29), "audit capture")
    _shape(capture["random_capture_curves"], (7, 256, 29), "random audit capture")
    _shape(capture["zero_denominator"], (7,), "capture zero flags")
    null = evaluated["random_controls"]["within_size"]
    permutations = _shape(null["permutation"], (128, 119, 256), "within-size permutations")
    _shape(null["pair_values"], (128, 119, 3, 28), "within-size pair values")
    seeds = null["derived_seeds_hex"]
    if len(seeds) != 128 or any(len(row) != 119 for row in seeds):
        raise ValueError("All within-size private seeds must be retained.")
    support = evaluated["support"]
    _shape(support["nearest_train_distance"], (272, 256, 8), "support distances")
    nearest = _shape(support["nearest_train_source_id"], (272, 256, 8), "nearest source IDs")
    if not np.isin(nearest, manifest["splits"]["train"]).all():
        raise ValueError("A nearest source ID is outside the training split.")
    models = fitted["models"]
    return {
        "study": "16.16_shapley_taylor_faith_shap",
        "model_fits": len(models),
        "fresh_model_fits": len(models),
        "model_keys": list(models),
        "mlp_fits": sum(item["kind"] == "mlp" for item in models.values()),
        "shuffled_fits": sum(item["control"] == "shuffled" for item in models.values()),
        "optimizer_updates": sum(len(item["loss_trace"]) for item in models.values() if item["kind"] == "mlp"),
        "known_model_state_hashes_matched": sum(len(v) for v in reproduced["model_state_hashes"].values()),
        "train_row_count": len(fitted["train_ids"]),
        "explanation_row_count": len(prepared["explain_source_ids"]),
        "heldout_row_count": len(evaluated["audit_source_ids"]),
        "cross_split_recipe_overlap": manifest["cross_split_recipe_overlap"],
        "final_used_for_selection": evaluated["audit_used_for_selection"],
        "model_row_explanations": int(np.prod(values.shape[:-1])),
        "model_coalition_values": values.size,
        "unique_empirical_games": int(np.prod(values.shape[:-1])) + alternate_games,
        "background_variants": len(evaluated["all_rows"]["background_variants"]),
        "primary_context_values": contexts.size,
        "direct_four_forward_values": direct_four.size,
        "within_size_null_replicates": permutations.shape[0],
        "within_size_permutation_elements": permutations.size,
        "within_size_seeds_retained": sum(map(len, seeds)),
        "random_pair_order_controls": len(random_order),
        "support_distance_points": nearest.size,
        "independent_model_replication": False,
        "whole_paper_replication": False,
        "observational_interactions_are_causal": False,
        "sii_joint_efficiency_claimed": False,
        "method_superiority_is_acceptance_gate": False,
        "background_changes_estimand": True,
        "faith_kernel_mse_max": float(kernel_mse[1].max()),
        "stii_kernel_mse_max": float(kernel_mse[0].max()),
        "efficient_decomposition_residual_max": float(np.abs(efficiency).max()),
        "stii_faith_coordinate_difference_max": float(np.abs(_array(primary["stii_coefficients"]) - _array(primary["fsii_coefficients"])).max()),
        "direct_vs_table_context_difference_max": float(np.abs(_array(direct["lookup_difference"])).max()),
    }


def _reference_metrics(reference: dict, evaluated: dict) -> dict:
    """Recheck all retained reference vectors and their frozen, per-game bounds."""
    from chapter16_shapley_attribution_baselines.exercises.part16_shapley_taylor_faith_shap.interaction_reference import (
        _array_sha256, validate_reference_payload,
    )

    all_rows = evaluated["all_rows"]
    blocks = [all_rows["primary64"]] + [
        all_rows["background_variants"][name]
        for name in ("nested16", "nested128", "independent64_a", "independent64_b")
    ]

    def joined(key: str) -> np.ndarray:
        arrays = [_array(block[key]) for block in blocks]
        return np.concatenate([array.reshape(-1, array.shape[-1]) for array in arrays])

    values = _shape(joined("coalition_values"), (2992, 256), "reference source games")
    if reference["identity_sha256"]["game_values"] != _array_sha256(values):
        raise ValueError("Reference games differ from the actual retained empirical games.")
    if reference["versions"] != {"shapiq": "1.5.2"}:
        raise ValueError("The pinned package version changed.")
    if reference["scope"]["n_players"] != 8 or reference["scope"]["order"] != 2:
        raise ValueError("Empirical references must use eight features at order two.")
    terms = np.array([mask for mask in range(256) if mask.bit_count() <= 2], dtype=np.int64)
    pairs = np.array([(1 << left) | (1 << right) for left in range(8) for right in range(left + 1, 8)], dtype=np.int64)
    for name, expected in (("coalition", np.arange(256)), ("package_terms", terms), ("package_sii_terms", terms), ("pairs", pairs)):
        if not np.array_equal(_array(reference["masks"][name]), expected):
            raise ValueError(f"Reference {name} axis differs from the frozen mask order.")
    student = reference["student"]
    for name, key in (("stii", "stii_coefficients"), ("fsii", "fsii_coefficients"), ("sii_pairs", "sii_pairs")):
        actual = joined(key)
        if not np.array_equal(_shape(student[name], actual.shape, f"reference student {name}"), actual):
            raise ValueError(f"Reference student {name} differs from the learner's retained values.")
    independent = reference["independent"]
    stii = _shape(independent["stii"], (2992, 256), "independent STII")
    fsii = _shape(independent["fsii_closed_form"], (2992, 256), "independent FSII")
    kkt = _shape(independent["fsii_kkt"], (2992, 256), "independent KKT")
    sii = _shape(independent["sii_pairs"], (2992, 28), "independent SII")
    package = reference["packages"]["shapiq"]
    package_st = _shape(package["stii"], (2992, 37), "package STII")
    package_fs = _shape(package["fsii"], (2992, 37), "package FSII")
    package_sii = _shape(package["sii"], (2992, 37), "package raw SII")
    pair_positions = np.array([int(np.flatnonzero(terms == mask)[0]) for mask in pairs])
    errors = {
        "student_stii_minus_independent": _array(student["stii"]) - stii,
        "student_fsii_minus_independent_closed": _array(student["fsii"]) - fsii,
        "student_sii_pairs_minus_independent": _array(student["sii_pairs"]) - sii,
        "independent_kkt_minus_closed_fsii": kkt - fsii,
        "shapiq_stii_minus_independent": package_st - stii[:, terms],
        "shapiq_fsii_minus_independent": package_fs - fsii[:, terms],
        "shapiq_sii_pairs_minus_independent": package_sii[:, pair_positions] - sii,
    }
    centered_total = values[:, -1] - values[:, 0]
    coefficient_arrays = {
        "student_stii": _array(student["stii"]),
        "student_fsii": _array(student["fsii"]),
        "independent_stii": stii,
        "independent_fsii_closed": fsii,
        "independent_fsii_kkt": kkt,
    }
    scale = np.maximum(1.0, np.max(np.abs(values - values[:, :1]), axis=1))
    above_order = np.array([mask.bit_count() > 2 for mask in range(256)])
    for name, coefficients in coefficient_arrays.items():
        if np.any(coefficients[:, above_order] != 0):
            raise ValueError(f"{name} retained nonzero coefficients above order two.")
        if np.any(np.abs(coefficients[:, 0] - values[:, 0]) > 1e-10 + 1e-12 * scale):
            raise ValueError(f"{name} violates the intercept constraint's numerical bound.")
    efficiency = {name: coefficients[:, 1:].sum(1) - centered_total for name, coefficients in coefficient_arrays.items()}
    # Match the adapter's full-axis reduction order, including structural zeros.
    for name, coefficients in (("shapiq_stii", package_st), ("shapiq_fsii", package_fs)):
        expanded = np.zeros_like(values)
        expanded[:, terms] = coefficients
        efficiency[name] = expanded[:, 1:].sum(1) - centered_total
    maxima: dict[str, float] = {}
    for branch_name, calculated in (("errors", errors), ("efficiency_errors", efficiency)):
        if set(reference[branch_name]) != set(calculated):
            raise ValueError(f"Reference {branch_name} coverage changed.")
        for name, error in calculated.items():
            retained = _shape(reference[branch_name][name], error.shape, name)
            if not np.array_equal(retained, error):
                raise ValueError(f"Retained reference residual is inconsistent: {name}.")
            is_package_fs = name in {"shapiq_fsii", "shapiq_fsii_minus_independent"}
            floor = 1e-8 if is_package_fs else 1e-10
            factor = (3e-8 if branch_name == "efficiency_errors" else 5e-9) if is_package_fs else 1e-12
            bounds = floor + factor * scale
            ratios = np.abs(error.reshape(2992, -1)) / bounds[:, None]
            maximum = float(ratios.max())
            if not np.isfinite(maximum) or maximum > 1:
                raise ValueError(f"Reference {name} exceeds its pre-native numerical bound.")
            maxima[f"{branch_name}.{name}"] = maximum
    validated = validate_reference_payload(reference, expected_game_count=len(values))
    measured = {
        "reference_games": len(values),
        "reference_student_coordinates": sum(_array(v).size for v in student.values()),
        "reference_raw_package_coordinates": sum(_array(v).size for v in package.values()),
        "reference_relevant_package_coordinates": package_st.size + package_fs.size + sii.size,
        "reference_package_index_calls": len(values) * 3,
        "reference_versions": reference["versions"],
        "reference_max_error_to_bound_ratio": max(maxima.values()),
        "reference_error_to_bound_ratios": maxima,
        "reference_passed": reference["status"] == "PASS" and all(reference["passes"].values()),
        "reference_full_direct_package_scope": reference["scope"]["direct_package_call_per_game"] is True,
        "reference_source_files_rehashed": validated["sources_rehashed"],
    }
    return measured


def _hybrid_metrics(directory: Path, prepared: dict, evaluated: dict, manifest: dict) -> dict:
    """Open all ten actual donor shards and validate the descriptor/file/value bindings."""
    from chapter16_shapley_attribution_baselines.exercises.part16_shapley_taylor_faith_shap.run_native import restore_external_tensor

    artifacts = set()
    total = primary = 0
    file_hashes = {}
    for record in manifest["physical_hybrid_shards"]["shards"]:
        node = prepared if record["stage"] == "prepared_before_audit" else evaluated
        for key in record["payload_path"].split("."):
            node = node[key]
        if node["artifact"] != record["artifact"] or node["artifact"] in artifacts:
            raise ValueError("A physical hybrid shard is aliased, duplicated or misnamed.")
        artifacts.add(node["artifact"])
        array = _shape(restore_external_tensor(directory, node), tuple(record["shape"]), node["artifact"])
        if array.dtype != np.dtype("float64") or array.size != record["elements"]:
            raise ValueError("Hybrid shard dtype/count differs from the frozen manifest.")
        total += array.size
        if node["artifact"].startswith("primary_"):
            primary += array.size
        file_hashes[node["artifact"]] = _sha256(node["artifact_sha256"], node["artifact"])
        del array
    return {
        "physical_hybrid_shards": len(artifacts),
        "primary_hybrid_predictions": int(primary),
        "full_hybrid_predictions": int(total),
        "physical_hybrid_file_sha256": file_hashes,
        "physical_hybrid_values_rehashed": True,
    }


def _exact_metrics(prepared: dict) -> dict:
    """Revalidate every fixed, random and full-order case from its retained vectors."""
    from chapter16_shapley_attribution_baselines.exercises.part16_shapley_taylor_faith_shap.interaction_reference import validate_exact_reference_suite

    validated = validate_exact_reference_suite(prepared)
    return {
        "exact_fixed_games": validated["fixed_unique_games"],
        "exact_random_games": validated["random_unique_games"],
        "exact_random_game_order_cases": validated["random_game_order_cases"],
        "exact_boundary_order_n_cases": validated["boundary_order_n_cases"],
        "exact_package_game_instances": validated["direct_shapiq_exact_computers"],
        "exact_package_index_calls": validated["direct_shapiq_index_calls"],
        "exact_reference_coordinates": validated["student_full_axis_coordinates"],
        "exact_reference_passed": validated["status"] == "PASS",
    }


def run_gpu_test(max_vram_gb: float = 0.0) -> dict:
    """Legacy report API name: perform another complete fresh CPU fit and reproduction."""
    from chapter16_shapley_attribution_baselines.exercises.part16_shapley_taylor_faith_shap.build_notebooks import STEM
    from chapter16_shapley_attribution_baselines.exercises.part16_shapley_taylor_faith_shap.publish_study import (
        PAGE, published_asset_name, validate_fresh,
    )
    from chapter16_shapley_attribution_baselines.exercises.part16_shapley_taylor_faith_shap.reproduce_native import reproduce
    from chapter16_shapley_attribution_baselines.exercises.part16_shapley_taylor_faith_shap.run_native import digest, load_payload

    here = Path(__file__).parent
    evidence_path = here / "expected_outputs/learner_execution.json"
    if not evidence_path.is_file():
        raise FileNotFoundError("Publish the fully executed, reproduced and visually reviewed learner first.")
    published = json.loads(evidence_path.read_text())
    for path, field in (
        (here / f"{STEM}_exercises.ipynb", "published_exercises_sha256"),
        (here / f"{STEM}_solutions.ipynb", "published_solutions_sha256"),
        (PAGE, "published_page_sha256"),
    ):
        if not path.is_file() or digest(path) != published[field]:
            raise ValueError(f"Published learner changed: {path}.")
    assets = PAGE.parent.parent / "assets"
    for name, checksum in published["published_figures"].items():
        if published_asset_name(name.removeprefix("shapley_interactions_")) != name:
            raise ValueError(f"Invalid published figure namespace: {name}.")
        if not (assets / name).is_file() or digest(assets / name) != checksum:
            raise ValueError(f"Published figure changed: {name}.")
    learner = Path(published["learner_directory"])
    cache = learner.parent / "report_reproductions"
    cache.mkdir(exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="report-", dir=cache)) / "fresh_cpu"
    reproduce(learner, output)
    execution, fresh = validate_fresh(learner, output)
    fresh_path = here / "expected_outputs/fresh_reproduction.json"
    fresh_path.write_bytes((output / "reproduction.json").read_bytes())
    native = output / "fresh_run"
    fitted = load_payload(native / "fitted.pt")
    prepared = load_payload(native / "prepared.pt")
    evaluated = load_payload(native / "evaluated.pt")
    reference = load_payload(native / "reference.pt")
    manifest = json.loads((here / "data_manifest.json").read_text())
    original_fitted = load_payload(Path(execution["native_directory"]) / "fitted.pt")
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
            and fresh["child_received_learner_payloads"] is False
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
        **_reference_metrics(reference, evaluated),
        **_hybrid_metrics(native, prepared, evaluated, manifest),
        **_exact_metrics(prepared),
        "model_fits": len(original_fitted["models"]),
    }


def run_full_experiment(max_vram_gb: float = 0.0) -> dict:
    return {"numerical": run_smoke_test(), "fresh_cpu_model": run_gpu_test(max_vram_gb)}
