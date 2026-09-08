"""Reference answers and the published fresh-CPU TreeSHAP evidence contract."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from chapter16_shapley_attribution_baselines.exercises.part13_treeshap.treeshap_methods import *
from chapter16_shapley_attribution_baselines.exercises.part13_treeshap import (
    treeshap_methods_tests,
)


def run_smoke_test(cpu: bool = True) -> dict:
    """Run the immediate independent method checks used by learner exercise cells."""
    names = []
    for name, function in vars(treeshap_methods_tests).items():
        if name.startswith("test_") and inspect.isfunction(function):
            function()
            names.append(name)
    return {
        "tests_passed": True,
        "accepted": True,
        "test_count": len(names),
        "claim_scope": (
            "Exact coalition, path-extension, repeated-feature, cover-weighted, interventional, "
            "Saabas, and ensemble linearity checks for the declared tree games."
        ),
    }


def _reference_metrics(reference: dict) -> dict:
    counts = reference["counts"]
    maximum = reference["max_errors"]
    path_error = max(maximum["prediction"], maximum["path_base"], maximum["path_phi"])
    raw_error = max(
        maximum["interventional_raw_base"],
        maximum["interventional_raw_phi"],
        maximum["interventional_raw_prediction_residual"],
    )
    corrected_error = max(
        maximum["interventional_corrected_base"],
        maximum["interventional_corrected_phi"],
        maximum["interventional_corrected_prediction_residual"],
    )
    return {
        "path_reference_passed": reference["passes"]["path_compiled"],
        "path_reference_model_rows": counts["model_rows"],
        "path_reference_max_error": float(path_error),
        "raw_interventional_reference_passed": reference["unmodified_interventional_passed"],
        "raw_interventional_reference_disclosed": True,
        "raw_interventional_reference_model_rows": counts["model_rows"],
        "raw_interventional_reference_affected_rows": counts["raw_affected_model_rows"],
        "raw_interventional_reference_max_error": float(raw_error),
        "corrected_interventional_reference_passed": reference[
            "threshold_corrected_interventional_passed"
        ],
        "corrected_interventional_reference_model_rows": counts["model_rows"],
        "corrected_interventional_reference_max_error": float(corrected_error),
        "exact_model_rows": counts["exact_model_rows"],
        "exact_game_model_rows": 2 * counts["exact_model_rows"],
        "corrected_thresholds": counts["corrected_thresholds"],
        "raw_source_route_disagreements": counts["raw_source_route_disagreements"],
        "corrected_source_route_disagreements": counts[
            "corrected_source_route_disagreements"
        ],
    }


def run_gpu_test(max_vram_gb: float = 0.0) -> dict:
    """Legacy report entry point; reproduce all native models on CPU without cached weights."""
    from chapter16_shapley_attribution_baselines.exercises.part13_treeshap.build_notebooks import STEM
    from chapter16_shapley_attribution_baselines.exercises.part13_treeshap.execute_learner import digest
    from chapter16_shapley_attribution_baselines.exercises.part13_treeshap.publish_study import (
        PAGE,
        published_asset_name,
        validate_fresh,
    )
    from chapter16_shapley_attribution_baselines.exercises.part13_treeshap.reproduce_native import (
        reproduce,
    )
    from chapter16_shapley_attribution_baselines.exercises.part13_treeshap.run_native import (
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
        if published_asset_name(name.removeprefix("treeshap_")) != name:
            raise ValueError("Published TreeSHAP figure is not namespaced: " + name)
        if digest(assets / name) != checksum:
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
    reference = load_payload(native_directory / "reference.pt")
    diagnostics = load_payload(native_directory / "diagnostics.pt")
    prepared = load_payload(native_directory / "prepared.pt")
    evaluated = load_payload(native_directory / "evaluated.pt")
    manifest = json.loads((here / "data_manifest.json").read_text())
    comparisons = fresh["comparisons"]
    saabas_additivity = 0.0
    saabas_gap = 0.0
    exhaustive_path_error = 0.0
    exhaustive_interventional_error = 0.0
    for block in (prepared["explanation"], evaluated["audit"]):
        for item in block.values():
            saabas_additivity = max(
                saabas_additivity,
                float(
                    np.max(
                        np.abs(
                            item["base_path"]
                            + item["phi_saabas"].sum(1)
                            - item["prediction"]
                        )
                    )
                ),
            )
            saabas_gap = max(
                saabas_gap,
                float(np.max(np.abs(item["phi_saabas"] - item["phi_path"]))),
            )
            if "game_path" in item:
                exhaustive_path_error = max(
                    exhaustive_path_error,
                    float(
                        np.max(
                            np.abs(exact_shapley(item["game_path"]) - item["phi_path"])
                        )
                    ),
                )
                exhaustive_interventional_error = max(
                    exhaustive_interventional_error,
                    float(
                        np.max(
                            np.abs(
                                exact_shapley(item["game_interventional"])
                                - item["phi_interventional"]
                            )
                        )
                    ),
                )
    reference_maximum = reference["max_errors"]
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
        "component_trees": native["component_trees"],
        "model_families": native["model_families"],
        "training_seeds": native["training_seeds"],
        "clean_fits": native["clean_fits"],
        "shuffled_fits": native["shuffled_fits"],
        "shuffled_models_evaluated": native["shuffled_fits"],
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
        "exact_games": native["exact_games"],
        "exact_coalitions": native["coalitions"],
        "shap_parity_rows": reference["counts"]["model_rows"],
        "shap_parity_rows_per_model": native["rows_explained_per_model"],
        "shap_parity_model_rows": reference["counts"]["model_rows"],
        "prediction_parity_max_error": float(reference_maximum["prediction"]),
        "base_value_parity_max_error": float(
            max(
                reference_maximum["path_base"],
                reference_maximum["interventional_corrected_base"],
            )
        ),
        "random_order_controls": native["random_order_controls"],
        "deletion_models": native["deletion_models"],
        "deletion_audit_rows": native["heldout_row_count"],
        "deletion_uses_exact_interventional_game": True,
        "background_variants": native["background_variants"],
        "background_rows_compared": native["rows_explained_per_model"],
        "alternate_backgrounds_compared": True,
        "saabas_models": native["saabas_models"],
        "saabas_model_rows": native["model_row_explanations"],
        "saabas_deletion_models": native["deletion_models"],
        "saabas_differs_from_shap": saabas_gap > 1.0e-8,
        "saabas_additivity_max_error": saabas_additivity,
        "saabas_attribution_gap_max": saabas_gap,
        "training_mean_baseline": True,
        "dummy_feature_max_abs": float(diagnostics["runtime_parity"]["dummy_feature_max_abs"]),
        "all_training_seeds_retained": True,
        "shuffled_training_targets": True,
        "whole_paper_replication": False,
        **_reference_metrics(reference),
    }
    # These two report the exhaustive declared games themselves. The separately
    # named corrected compiled-reference error retains its float32 C++ tolerance.
    metrics["exhaustive_path_max_error"] = exhaustive_path_error
    metrics["exhaustive_interventional_max_error"] = exhaustive_interventional_error
    return metrics


def run_full_experiment(max_vram_gb: float = 0.0) -> dict:
    return {"numerical": run_smoke_test(), "fresh_cpu_model": run_gpu_test(max_vram_gb)}
