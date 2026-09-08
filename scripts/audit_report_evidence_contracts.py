"""Audit verification reports for non-placeholder evidence contracts."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_extension_verification_reports import extension_records, expected_metric_failures
from arena_ext.verification import verify_report_input_manifest


PLACEHOLDER_EVIDENCE_LEVELS = {
    "notebook_contract",
    "placeholder_only",
}
PLACEHOLDER_CLAIM_TERMS = (
    "placeholder",
    "starter report",
)


def _has_mapping(value: Any) -> bool:
    return isinstance(value, dict) and bool(value)


def _has_nonempty_list(value: Any) -> bool:
    return isinstance(value, list) and bool(value)


def _has_negative_control(controls: list[Any]) -> bool:
    text = " ".join(str(control).lower() for control in controls)
    return any(term in text for term in ("negative", "random", "shuffle", "permut", "control"))


def _bounded_number(value: Any, lower: float, upper: float) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and lower <= value <= upper
    )


def _has_fitted_tree_regression_evidence(measured: dict[str, Any]) -> bool:
    """Require the complete TreeSHAP fit, parity and control study.

    This is intentionally separate from the SAGE global-regression contract:
    tree estimators have no neural optimizer updates or sampled SAGE paths.
    Counts bind parity to every prescribed model and row, while the error fields
    require full attribution-vector checks rather than additivity alone.
    """
    return (
        measured.get("model_fits") == 18
        and measured.get("fresh_model_fits") == 18
        and measured.get("component_trees") == 486
        and measured.get("model_families") == ["tree", "forest", "boost"]
        and measured.get("training_seeds") == [161310, 161311, 161312]
        and measured.get("clean_fits") == 9
        and measured.get("shuffled_fits") == 9
        and measured.get("all_model_fits_retained") is True
        and measured.get("all_training_seeds_retained") is True
        and measured.get("shuffled_training_targets") is True
        and measured.get("shuffled_models_evaluated") == 9
        and measured.get("heldout_unit") == "recipe_group"
        and measured.get("train_row_count") == 758
        and measured.get("explanation_row_count") == 153
        and measured.get("heldout_row_count") == 119
        and measured.get("train_group_count") == 298
        and measured.get("explanation_group_count") == 64
        and measured.get("heldout_group_count") == 65
        and measured.get("cross_split_recipe_overlap") == 0
        and measured.get("final_used_for_selection") is False
        and measured.get("rows_explained_per_model") == 272
        and measured.get("model_row_explanations") == 4896
        and measured.get("shap_parity_rows_per_model") == 272
        and measured.get("shap_parity_model_rows") == 4896
        and measured.get("exact_models") == 3
        and measured.get("exact_games") == 2
        and measured.get("exact_coalitions") == 256
        and measured.get("exact_model_rows") == 816
        and measured.get("exact_game_model_rows") == 1632
        and measured.get("path_reference_passed") is True
        and measured.get("path_reference_model_rows") == 4896
        and _bounded_number(measured.get("path_reference_max_error"), 0, 1e-8)
        # Preserve the installed SHAP 0.49.1 threshold32 failure as evidence.
        # The corrected C++ comparison is a separate named reference route.
        and measured.get("raw_interventional_reference_passed") is False
        and measured.get("raw_interventional_reference_disclosed") is True
        and measured.get("raw_interventional_reference_model_rows") == 4896
        and _bounded_number(
            measured.get("raw_interventional_reference_affected_rows"), 2, 4896
        )
        and _bounded_number(measured.get("raw_interventional_reference_max_error"), 1e-3, 10.0)
        and measured.get("corrected_interventional_reference_passed") is True
        and measured.get("corrected_interventional_reference_model_rows") == 4896
        and _bounded_number(measured.get("corrected_interventional_reference_max_error"), 0, 2e-5)
        and measured.get("corrected_thresholds") == 941
        and measured.get("raw_source_route_disagreements") == 217
        and measured.get("corrected_source_route_disagreements") == 0
        and _bounded_number(measured.get("exhaustive_path_max_error"), 0, 1e-8)
        and _bounded_number(measured.get("exhaustive_interventional_max_error"), 0, 1e-8)
        and _bounded_number(measured.get("prediction_parity_max_error"), 0, 1e-8)
        and _bounded_number(measured.get("base_value_parity_max_error"), 0, 2e-5)
        and measured.get("deletion_models") == 3
        and measured.get("deletion_audit_rows") == 119
        and measured.get("deletion_uses_exact_interventional_game") is True
        and measured.get("random_order_controls") == 100
        and measured.get("background_variants") == 4
        and measured.get("background_rows_compared") == 272
        and measured.get("alternate_backgrounds_compared") is True
        and measured.get("saabas_model_rows") == 4896
        and measured.get("saabas_deletion_models") == 3
        and measured.get("saabas_differs_from_shap") is True
        and _bounded_number(measured.get("saabas_additivity_max_error"), 0, 1e-8)
        and _bounded_number(measured.get("saabas_attribution_gap_max"), 1e-8, float("inf"))
        and measured.get("training_mean_baseline") is True
        and _bounded_number(measured.get("dummy_feature_max_abs"), 0, 1e-10)
        and measured.get("embedded_figures", 0) >= 16
        and measured.get("fresh_compared_elements", 0) >= 1_055_546
    )


def _has_deepshap_neural_evidence(measured: dict[str, Any]) -> bool:
    """Require the full neural DeepSHAP approximation and exact-game study."""
    return (
        measured.get("model_fits") == 7
        and measured.get("fresh_model_fits") == 7
        and measured.get("mlp_fits") == 6
        and measured.get("neural_training_runs") == 6
        and measured.get("ridge_fits") == 1
        and measured.get("clean_fits") == 3
        and measured.get("shuffled_fits") == 3
        and measured.get("shuffled_models_evaluated") == 3
        and measured.get("optimizer_updates") == 4800
        and measured.get("model_keys")
        == [
            "clean_0",
            "clean_1",
            "clean_2",
            "shuffled_0",
            "shuffled_1",
            "shuffled_2",
            "ridge",
        ]
        and measured.get("training_seeds") == [161410, 161411, 161412]
        and measured.get("all_model_fits_retained") is True
        and measured.get("all_training_seeds_retained") is True
        and measured.get("shuffled_training_targets") is True
        and measured.get("heldout_unit") == "recipe_group"
        and measured.get("train_row_count") == 758
        and measured.get("explanation_row_count") == 153
        and measured.get("heldout_row_count") == 119
        and measured.get("train_group_count") == 298
        and measured.get("explanation_group_count") == 64
        and measured.get("heldout_group_count") == 65
        and measured.get("cross_split_recipe_overlap") == 0
        and measured.get("final_used_for_selection") is False
        and measured.get("rows_explained_per_model") == 272
        and measured.get("model_row_explanations") == 1904
        and measured.get("exact_models") == 7
        and measured.get("exact_coalitions") == 256
        and measured.get("exact_model_rows") == 1904
        and measured.get("model_coalition_values") == 487424
        and measured.get("primary_hybrid_predictions") == 31195136
        and measured.get("attribution_coordinates_per_method") == 15232
        and measured.get("background_rows") == 64
        and measured.get("pair_model_rows") == 121856
        and measured.get("pair_local_values") == 974848
        and _bounded_number(measured.get("folding_max_error"), 0, 1e-10)
        and measured.get("author_reference_passed") is True
        and measured.get("author_reference_model_rows") == 1904
        and measured.get("author_reference_local_values") == 15232
        and measured.get("author_pair_rows_checked") == 896
        and measured.get("author_pair_local_values_checked") == 7168
        and measured.get("author_reference_atol") == 1e-7
        and measured.get("author_reference_rtol") == 1e-8
        and _bounded_number(measured.get("author_reference_max_error"), 0, 1e-7)
        and measured.get("real_local_accuracy_passed") is True
        and measured.get("real_local_accuracy_atol") == 1e-7
        and measured.get("real_local_accuracy_rtol") == 1e-8
        and _bounded_number(measured.get("real_local_accuracy_max_error"), 0, 1e-7)
        and _bounded_number(measured.get("exact_efficiency_max_error"), 0, 1e-10)
        and measured.get("deepshap_is_exact_shap") is False
        and measured.get("deepshap_approximation_disclosed") is True
        and _bounded_number(measured.get("deepshap_exact_gap_max"), 1e-10, float("inf"))
        and _bounded_number(measured.get("ridge_exact_max_error"), 0, 1e-10)
        and measured.get("nearzero_primary_local_accuracy_passed") is False
        and measured.get("nearzero_primary_failure_disclosed") is True
        and _bounded_number(measured.get("nearzero_primary_residual"), 3.999e-7, 4.001e-7)
        and _bounded_number(measured.get("nearzero_epsilon_zero_max_error"), 0, 1e-10)
        and _bounded_number(measured.get("threshold_exact_max_error"), 0, 1e-10)
        and _bounded_number(measured.get("min_exact_max_error"), 0, 1e-10)
        and measured.get("background_variants") == 5
        and measured.get("background_rows_compared") == 272
        and measured.get("alternate_backgrounds_compared") is True
        and measured.get("replacement_models") == 7
        and measured.get("replacement_audit_rows") == 119
        and measured.get("random_order_controls") == 100
        and measured.get("gradient_baseline_model_rows") == 1904
        and measured.get("training_mean_baseline") is True
        and _bounded_number(measured.get("dummy_feature_max_abs"), 0, 1e-10)
        and measured.get("unsupported_operator_rejected") is True
        and measured.get("whole_paper_replication") is False
        and measured.get("embedded_figures", 0) >= 17
        and measured.get("fresh_compared_arrays", 0) >= 842
        and measured.get("fresh_compared_elements", 0) >= 5_887_125
    )


def verification_report_records() -> list[dict[str, Path]]:
    records = [
        {
            "lock_path": record["lock_path"],
            "report_path": record["report_path"],
        }
        for record in extension_records()
    ]
    for report_path in sorted((ROOT / "docs/evidence").glob("*/verification_report.json")):
        lock_path = report_path.with_name("artifacts.lock.yml")
        if lock_path.exists():
            records.append({"lock_path": lock_path, "report_path": report_path})
    return records


def report_freshness_blockers(report: dict[str, Any]) -> list[str]:
    notebook_id = str(report.get("notebook_id", "<unknown>"))
    return [
        f"{notebook_id}: {blocker}"
        for blocker in verify_report_input_manifest(ROOT, report.get("report_inputs", {}))
    ]


def _gradientshap_evidence_hashes_match(report: dict[str, Any], measured: dict[str, Any]) -> bool:
    """Bind the learner/fresh evidence hashes to the report's verified input manifest."""
    manifest = report.get("report_inputs", {})
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), list):
        return False
    files = manifest["files"]
    prefix = "chapter16_shapley_attribution_baselines/exercises/part15_gradientshap/expected_outputs/"
    names = {"learner_evidence_sha256": "learner_execution.json", "fresh_evidence_sha256": "fresh_reproduction.json"}
    for field, name in names.items():
        checksum = measured.get(field)
        matches = [entry for entry in files if isinstance(entry, dict) and entry.get("path") == prefix + name]
        if not isinstance(checksum, str) or len(checksum) != 64:
            return False
        if any(character not in "0123456789abcdef" for character in checksum):
            return False
        if len(matches) != 1 or matches[0].get("sha256") != checksum:
            return False
    return True


def _has_gradientshap_evidence(measured: dict[str, Any]) -> bool:
    """Require complete Expected Gradients scope without grading scientific outcomes.

    These counts distinguish full retained draws, pair quadrature and same-draw
    author vectors from a smoke test. Monte Carlo error, quadrature drift,
    cross-game disagreement and replacement superiority are not acceptance gates.
    """
    exact = {
        "study": "16.15_gradientshap_expected_gradients",
        "model_fits": 7,
        "fresh_model_fits": 7,
        "mlp_fits": 6,
        "shuffled_fits": 3,
        "optimizer_updates": 4800,
        "known_model_state_hashes_matched": 21,
        "train_row_count": 758,
        "explanation_row_count": 153,
        "heldout_row_count": 119,
        "cross_split_recipe_overlap": 0,
        "model_row_explanations": 1904,
        "primary_sample_coordinates": 31195136,
        "model_coalition_values": 487424,
        "primary_hybrid_predictions": 31195136,
        "quadrature_pair_coordinates": {"256": 974848, "512": 974848},
        "nested_budgets": [8, 32, 128, 512, 2048],
        "replicate_sample_coordinates": 201326592,
        "replicate_sample_shape": [64, 3, 64, 2048, 8],
        "replicate_seeds_retained": 64,
        "background_variants": 5,
        "noise_variants": 3,
        "random_order_controls": 100,
        "replacement_models": 7,
        "path_distance_points": 34816,
        "coalition_distance_points": 557056,
        "author_reference_versions": {"shap": "0.49.1", "captum": "0.9.0"},
        "author_samples_per_package": 128,
        "author_model_rows_per_package": 1904,
        "author_total_vectors": 3808,
        "author_total_coordinates": 30464,
        "author_reference_atol": 1e-10,
        "author_reference_rtol": 1e-10,
        "fresh_comparison": "exact_recursive_value_sha256",
    }
    required_true = (
        "quadrature_is_approximate", "positive_noise_changes_target",
        "author_reference_passed", "positive_noise_reference_passed",
        "package_draw_tables_distinct",
    )
    required_false = (
        "final_used_for_selection", "finite_mc_completeness_is_acceptance_gate",
        "eg_is_exact_feature_shapley", "whole_paper_replication",
    )
    hashes = ("replicate_sample_sha256", "replicate_file_sha256")
    return (
        all(measured.get(key) == value for key, value in exact.items())
        and all(measured.get(key) is True for key in required_true)
        and all(measured.get(key) is False for key in required_false)
        and all(
            isinstance(measured.get(key), str)
            and len(measured[key]) == 64
            and all(character in "0123456789abcdef" for character in measured[key])
            for key in hashes
        )
        and _bounded_number(measured.get("replicate_file_size_bytes"), 1610612736, float("inf"))
        and _bounded_number(measured.get("fresh_compared_elements"), 263716864, float("inf"))
        and _bounded_number(measured.get("author_shap_max_error"), 0, float("inf"))
        and _bounded_number(measured.get("author_captum_max_error"), 0, float("inf"))
        and all(
            _bounded_number(measured.get(f"author_{name}_scaled_error"), 0, 1)
            and _bounded_number(measured.get(f"author_{name}_tolerance_bound"), 1e-10, float("inf"))
            and measured[f"author_{name}_max_error"] <= measured[f"author_{name}_tolerance_bound"]
            for name in ("shap", "captum")
        )
    )


def _has_shapley_interaction_evidence(report: dict[str, Any], measured: dict[str, Any]) -> bool:
    """Require complete empirical games and raw reference coverage for 16.16.

    Numerical reference bounds validate implementations. STII/Faith allocation
    disagreement, reconstruction error and ranking superiority remain outcomes.
    """
    exact = {
        "study": "16.16_shapley_taylor_faith_shap",
        "model_fits": 7, "fresh_model_fits": 7, "mlp_fits": 6,
        "shuffled_fits": 3, "optimizer_updates": 4800,
        "known_model_state_hashes_matched": 21,
        "train_row_count": 758, "explanation_row_count": 153,
        "heldout_row_count": 119, "cross_split_recipe_overlap": 0,
        "model_row_explanations": 1904, "model_coalition_values": 487424,
        "unique_empirical_games": 2992, "background_variants": 5,
        "primary_context_values": 3411968, "direct_four_forward_values": 213248,
        "within_size_null_replicates": 128, "within_size_permutation_elements": 3899392,
        "within_size_seeds_retained": 15232, "random_pair_order_controls": 256,
        "support_distance_points": 557056,
        "reference_games": 2992, "reference_student_coordinates": 1615680,
        "reference_raw_package_coordinates": 332112,
        "reference_relevant_package_coordinates": 305184,
        "reference_package_index_calls": 8976, "reference_versions": {"shapiq": "1.5.2"},
        "physical_hybrid_shards": 10, "primary_hybrid_predictions": 31195136,
        "full_hybrid_predictions": 50135040,
        "exact_fixed_games": 8, "exact_random_games": 15,
        "exact_random_game_order_cases": 45, "exact_boundary_order_n_cases": 5,
        "exact_package_game_instances": 58, "exact_package_index_calls": 174,
        "fresh_comparison": "exact_recursive_value_sha256",
    }
    true_flags = (
        "background_changes_estimand", "reference_passed", "reference_full_direct_package_scope",
        "physical_hybrid_values_rehashed", "exact_reference_passed",
    )
    false_flags = (
        "final_used_for_selection", "independent_model_replication", "whole_paper_replication",
        "observational_interactions_are_causal", "sii_joint_efficiency_claimed",
        "method_superiority_is_acceptance_gate",
    )
    files = report.get("report_inputs", {}).get("files", [])
    if not isinstance(files, list):
        return False
    prefix = "chapter16_shapley_attribution_baselines/exercises/part16_shapley_taylor_faith_shap/expected_outputs/"
    for key, name in (("learner_evidence_sha256", "learner_execution.json"), ("fresh_evidence_sha256", "fresh_reproduction.json")):
        checksum = measured.get(key)
        matches = [item for item in files if isinstance(item, dict) and item.get("path") == prefix + name]
        if not isinstance(checksum, str) or len(checksum) != 64 or any(c not in "0123456789abcdef" for c in checksum):
            return False
        if len(matches) != 1 or matches[0].get("sha256") != checksum:
            return False
    hashes = measured.get("physical_hybrid_file_sha256", {})
    names = {
        f"{background}_{split}_hybrid.pt"
        for split in ("explain", "audit")
        for background in ("primary", "nested16", "nested128", "independent64_a", "independent64_b")
    }
    return (
        all(measured.get(key) == value for key, value in exact.items())
        and all(measured.get(key) is True for key in true_flags)
        and all(measured.get(key) is False for key in false_flags)
        and _bounded_number(measured.get("fresh_compared_elements"), 50135040, float("inf"))
        and _bounded_number(measured.get("reference_max_error_to_bound_ratio"), 0, 1)
        and isinstance(hashes, dict) and set(hashes) == names
        and all(isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value) for value in hashes.values())
    )


def has_native_cpu_model_evidence(lock: dict[str, Any], report: dict[str, Any]) -> bool:
    """Accept an explicitly CPU-native study with own-answer and fresh-model evidence.

    CPU budget placeholders remain invalid. Existing GPU-declared contracts keep
    their CUDA requirement; model execution device is separate from host hardware.
    Input manifests bind the complete learner/fresh artifacts to this report.
    """
    if lock.get("execution_device") != "cpu" or lock.get("exercise_metadata", {}).get("REQUIRES_GPU") is not False:
        return False
    evidence = report.get("metrics", {}).get("gpu_evidence", {})
    measured = report.get("metrics", {}).get("gpu_test", {})
    pretrained = (evidence.get("category") == "cpu_or_budget_metric" and evidence.get("uses_cuda") is False
            and measured.get("device") == "cpu" and measured.get("native_model_reloaded") is True
            and measured.get("actual_learner_passed") is True and measured.get("fresh_reproduction_passed") is True
            and measured.get("own_answer_code_cells", 0) > 0 and measured.get("fresh_compared_examples", 0) > 0
            and measured.get("fresh_condition_batches", 0) > 0 and measured.get("fresh_top_token_mismatches") == 0
            and isinstance(measured.get("fresh_max_logit_error"), (int, float))
            and 0 <= measured["fresh_max_logit_error"] <= 1e-5
            and bool(measured.get("learner_evidence_sha256")) and bool(measured.get("fresh_evidence_sha256")))
    # A from-scratch classifier has real refits and prediction losses, rather
    # than a pretrained vocabulary. Keep that evidence distinct from token
    # metrics, and require fresh numerical reproduction and an explicit held-out unit.
    retrained = (
        evidence.get("category") == "cpu_or_budget_metric" and evidence.get("uses_cuda") is False
        and measured.get("device") == "cpu"
        and measured.get("native_model_retrained") is True
        and measured.get("native_model_retrained_in_this_report") is True
        and measured.get("native_evidence_reused") is False
        and measured.get("actual_learner_passed") is True
        and measured.get("fresh_reproduction_passed") is True
        and measured.get("own_answer_code_cells", 0) > 0
        and measured.get("reference_refits", 0) > 0
        and measured.get("fresh_compared_tensors", 0) > 0
        and measured.get("fresh_compared_elements", 0) > 0
        and isinstance(measured.get("fresh_max_error"), (int, float))
        and 0 <= measured["fresh_max_error"] <= 1e-10
        and bool(measured.get("learner_evidence_sha256"))
        and bool(measured.get("fresh_evidence_sha256"))
    )
    heldout_queries = (
        measured.get("final_query_count", 0) >= 20
        and measured.get("final_used_for_selection") is False
        and measured.get("fresh_compared_elements", 0) >= measured.get("final_query_count", 0)
    )
    # A datamodel's target images supply regression responses. Generalization is
    # tested on unseen training subsets, not on images unseen by the datamodel.
    heldout_subsets = (
        measured.get("heldout_unit") == "training_subset_model"
        and measured.get("heldout_model_count", 0) >= 20
        and measured.get("test_responses_used_for_selection") is False
        and measured.get("query_image_count", 0) >= 20
        and measured.get("query_images_used_as_regression_targets") is True
        and measured.get("query_images_excluded_from_classifier_training") is True
        and measured.get("independent_lasso_checks", 0) > 0
        and measured.get("reference_refits", 0) >= measured.get("heldout_model_count", 0)
        and measured.get("fresh_compared_elements", 0) >=
            measured.get("heldout_model_count", 0) * measured.get("query_image_count", 0)
    )
    optimized_classifiers = measured.get("independent_scipy_checks", 0) > 0 and (heldout_queries or heldout_subsets)
    # Training-dynamics studies validate an SGD path with an independent optimizer
    # loop, rather than pretending to solve a convex optimum with SciPy.
    training_dynamics = (
        measured.get("heldout_unit") == "evaluation_image"
        and measured.get("heldout_image_count", 0) >= 20
        and measured.get("heldout_split_count", 0) >= 2
        and measured.get("final_used_for_selection") is False
        and measured.get("heldout_images_excluded_from_training") is True
        and measured.get("independent_optimizer_checks", 0) >= 2
        and measured.get("independent_numpy_checks", 0) > 0
        and measured.get("separate_training_checks", 0) >= 2
        and measured.get("mapping_runs", 0) >= 5
        and measured.get("selection_refits", 0) >= 20
        and measured.get("reference_refits", 0) == measured.get("mapping_runs", 0) + measured.get("selection_refits", 0)
        and measured.get("optimizer_updates", 0) > measured.get("reference_refits", 0)
        and measured.get("both_logging_conventions_recorded") is True
        and measured.get("all_retraining_seeds_retained") is True
        and measured.get("class_matched_controls") is True
        and measured.get("equal_update_control") is True
        and measured.get("fresh_compared_elements", 0) >= measured.get("heldout_image_count", 0) * measured.get("reference_refits", 0)
    )
    global_regression = (
        measured.get("heldout_unit") == "recipe_group"
        and measured.get("heldout_group_count", 0) >= 20
        and measured.get("heldout_row_count", 0) >= measured.get("heldout_group_count", 0)
        and measured.get("explanation_row_count", 0) >= 20
        and measured.get("final_used_for_selection") is False
        and measured.get("cross_split_recipe_overlap") == 0
        and measured.get("all_training_seeds_retained") is True
        and measured.get("shuffled_training_targets") is True
        and measured.get("neural_training_runs", 0) >= 6
        and measured.get("optimizer_updates", 0) > measured.get("reference_refits", 0)
        and measured.get("random_order_controls", 0) >= 20
        and measured.get("exact_coalitions", 0) >= 16
        and measured.get("direct_sampled_paths", 0) >= 100
        and measured.get("independent_permutations", 0) >= 24
        and measured.get("independent_weighted_regression_coalitions", 0) >= 14
        and measured.get("independent_numpy_predictions", 0) >= 20
        and measured.get("author_imputation_comparisons", 0) >= 20
        and isinstance(measured.get("local_global_max_error"), (int, float))
        and 0 <= measured["local_global_max_error"] <= 1e-8
        and isinstance(measured.get("direct_path_max_error"), (int, float))
        and 0 <= measured["direct_path_max_error"] <= 1e-8
        and measured.get("ignored_feature_importance") == 0
    )
    fitted_tree_regression = (
        evidence.get("category") == "cpu_or_budget_metric"
        and evidence.get("uses_cuda") is False
        and measured.get("device") == "cpu"
        and measured.get("native_model_retrained") is True
        and measured.get("native_model_retrained_in_this_report") is True
        and measured.get("native_evidence_reused") is False
        and measured.get("actual_learner_passed") is True
        and measured.get("fresh_reproduction_passed") is True
        and measured.get("own_answer_code_cells", 0) > 0
        and measured.get("fresh_compared_arrays", 0) > 0
        and measured.get("fresh_compared_elements", 0) > 0
        and _bounded_number(measured.get("fresh_max_error"), 0, 1e-10)
        and bool(measured.get("learner_evidence_sha256"))
        and bool(measured.get("fresh_evidence_sha256"))
        and _has_fitted_tree_regression_evidence(measured)
    )
    deepshap_neural = (
        evidence.get("category") == "cpu_or_budget_metric"
        and evidence.get("uses_cuda") is False
        and measured.get("device") == "cpu"
        and measured.get("native_model_retrained") is True
        and measured.get("native_model_retrained_in_this_report") is True
        and measured.get("native_evidence_reused") is False
        and measured.get("actual_learner_passed") is True
        and measured.get("fresh_reproduction_passed") is True
        and measured.get("own_answer_code_cells", 0) > 0
        and _bounded_number(measured.get("fresh_max_error"), 0, 1e-10)
        and bool(measured.get("learner_evidence_sha256"))
        and bool(measured.get("fresh_evidence_sha256"))
        and _has_deepshap_neural_evidence(measured)
    )
    gradientshap_native = (
        evidence.get("category") == "cpu_or_budget_metric"
        and evidence.get("uses_cuda") is False
        and measured.get("device") == "cpu"
        and measured.get("native_model_retrained") is True
        and measured.get("native_model_retrained_in_this_report") is True
        and measured.get("native_evidence_reused") is False
        and measured.get("actual_learner_passed") is True
        and measured.get("fresh_reproduction_passed") is True
        and measured.get("own_answer_code_cells", 0) > 0
        and measured.get("embedded_figures") == 20
        and _bounded_number(measured.get("fresh_max_error"), 0, 0)
        and bool(measured.get("learner_evidence_sha256"))
        and bool(measured.get("fresh_evidence_sha256"))
        and _has_gradientshap_evidence(measured)
        and _gradientshap_evidence_hashes_match(report, measured)
    )
    shapley_interactions_native = (
        evidence.get("category") == "cpu_or_budget_metric"
        and evidence.get("uses_cuda") is False
        and measured.get("device") == "cpu"
        and measured.get("native_model_retrained") is True
        and measured.get("native_model_retrained_in_this_report") is True
        and measured.get("native_evidence_reused") is False
        and measured.get("actual_learner_passed") is True
        and measured.get("fresh_reproduction_passed") is True
        and measured.get("own_answer_code_cells", 0) > 0
        and measured.get("embedded_figures") == 11
        and _bounded_number(measured.get("fresh_max_error"), 0, 0)
        and _has_shapley_interaction_evidence(report, measured)
    )
    from scripts.shapiq_learner_contract import has_native_evidence

    shapiq_native = (
        evidence.get("category") == "cpu_or_budget_metric"
        and evidence.get("uses_cuda") is False
        and has_native_evidence(report)
    )
    from chapter16_shapley_attribution_baselines.exercises.part18_spex.native_report import (
        has_native_evidence as has_sparse_native_evidence,
    )

    sparse_interactions_native = (
        evidence.get("category") == "cpu_or_budget_metric"
        and evidence.get("uses_cuda") is False
        and has_sparse_native_evidence(report)
    )
    return (
        pretrained
        or (retrained and (optimized_classifiers or training_dynamics or global_regression))
        or fitted_tree_regression
        or deepshap_neural
        or gradientshap_native
        or shapley_interactions_native
        or shapiq_native
        or sparse_interactions_native
    )


def report_contract_blockers(lock: dict[str, Any], report: dict[str, Any]) -> list[str]:
    notebook_id = str(lock.get("notebook_id", report.get("notebook_id", "<unknown>")))
    blockers: list[str] = []

    if report.get("notebook_id") != lock.get("notebook_id"):
        blockers.append(f"{notebook_id}: report notebook_id does not match lockfile")
    if report.get("gt_tier") != lock.get("gt_tier"):
        blockers.append(f"{notebook_id}: report gt_tier does not match lockfile")
    if report.get("evidence_level") != lock.get("evidence_level"):
        blockers.append(f"{notebook_id}: report evidence_level does not match lockfile")
    if report.get("claim_scope") != lock.get("claim_scope"):
        blockers.append(f"{notebook_id}: report claim_scope does not match lockfile")

    evidence_level = str(report.get("evidence_level", ""))
    claim_scope = str(report.get("claim_scope", ""))
    if evidence_level in PLACEHOLDER_EVIDENCE_LEVELS:
        blockers.append(f"{notebook_id}: placeholder evidence_level={evidence_level}")
    if len(claim_scope) < 80:
        blockers.append(f"{notebook_id}: claim_scope is too thin")
    lower_claim_scope = claim_scope.lower()
    for term in PLACEHOLDER_CLAIM_TERMS:
        if term in lower_claim_scope:
            blockers.append(f"{notebook_id}: claim_scope contains placeholder term {term!r}")

    if report.get("tests_passed") is not True or report.get("accepted") is not True:
        blockers.append(f"{notebook_id}: report is not accepted")
    if report.get("known_failures") not in ([], None):
        blockers.append(f"{notebook_id}: accepted report has known_failures")

    metrics = report.get("metrics", {})
    evidence = metrics.get("gpu_evidence", {}) if isinstance(metrics, dict) else {}
    if evidence.get("category") != "cuda_section_metric" and not has_native_cpu_model_evidence(lock, report):
        blockers.append(f"{notebook_id}: gpu_evidence.category={evidence.get('category')!r}")
    if evidence.get("uses_cuda") is not True and not has_native_cpu_model_evidence(lock, report):
        blockers.append(f"{notebook_id}: gpu_evidence.uses_cuda is not true")
    if evidence.get("placeholder_only") is True:
        blockers.append(f"{notebook_id}: gpu_evidence is placeholder_only")
    if not evidence.get("section_specific_metric_keys"):
        blockers.append(f"{notebook_id}: gpu_evidence lacks section-specific metric keys")

    controls = lock.get("controls", [])
    if not _has_nonempty_list(controls):
        blockers.append(f"{notebook_id}: lockfile controls are empty")
    elif not _has_negative_control(controls):
        blockers.append(f"{notebook_id}: lockfile controls lack a negative/random control")

    baselines = report.get("baselines")
    if not _has_mapping(baselines):
        blockers.append(f"{notebook_id}: baselines are empty")
    else:
        if not baselines.get("declared_controls"):
            blockers.append(f"{notebook_id}: baselines.declared_controls is empty")
        if not baselines.get("expected_metrics"):
            blockers.append(f"{notebook_id}: baselines.expected_metrics is empty")

    negative_controls = report.get("negative_controls")
    if not _has_mapping(negative_controls):
        blockers.append(f"{notebook_id}: negative_controls are empty")
    elif not negative_controls.get("declared_controls"):
        blockers.append(f"{notebook_id}: negative_controls.declared_controls is empty")

    if not _has_mapping(report.get("ood_tests")):
        blockers.append(f"{notebook_id}: ood_tests are empty")

    if not _has_nonempty_list(report.get("models")):
        blockers.append(f"{notebook_id}: models are empty")
    if not _has_nonempty_list(report.get("datasets")):
        blockers.append(f"{notebook_id}: datasets are empty")
    if not _has_nonempty_list(report.get("safety_notes")):
        blockers.append(f"{notebook_id}: safety_notes are empty")

    blockers.extend(f"{notebook_id}: {failure}" for failure in expected_metric_failures(
        lock.get("expected_metrics", {}),
        report.get("metrics", {}),
    ))
    blockers.extend(report_freshness_blockers(report))
    return blockers


def report_evidence_blockers() -> list[str]:
    blockers: list[str] = []
    for record in verification_report_records():
        lock = yaml.safe_load(record["lock_path"].read_text())
        report = json.loads(record["report_path"].read_text())
        blockers.extend(report_contract_blockers(lock, report))
    return blockers


def main() -> None:
    records = verification_report_records()
    blockers = report_evidence_blockers()
    print(f"reports_checked={len(records)}")
    print(f"report_evidence_blockers={len(blockers)}")
    if blockers:
        print("REPORT_EVIDENCE_CONTRACTS=FAIL")
        for blocker in blockers:
            print(f"- {blocker}")
        raise SystemExit(1)
    print("REPORT_EVIDENCE_CONTRACTS=PASS")


if __name__ == "__main__":
    main()
