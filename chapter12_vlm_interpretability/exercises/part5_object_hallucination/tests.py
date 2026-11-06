"""Lazy bridge to untouched historical learner assertions; no experiment runs at import."""
from importlib import import_module

def _legacy():
    return import_module("chapter12_vlm_interpretability.exercises.part2_object_hallucination_arbitration.tests")

def test_build_arbitration_benchmark(*args, **kwargs):
    return _legacy().test_build_arbitration_benchmark(*args, **kwargs)

def test_encode_prompt_contains_query_and_prior_not_answer(*args, **kwargs):
    return _legacy().test_encode_prompt_contains_query_and_prior_not_answer(*args, **kwargs)

def test_counterfactual_changes_only_pixels_and_evaluation_label(*args, **kwargs):
    return _legacy().test_counterfactual_changes_only_pixels_and_evaluation_label(*args, **kwargs)

def test_model_inference_is_independent_of_example_label_metadata(*args, **kwargs):
    return _legacy().test_model_inference_is_independent_of_example_label_metadata(*args, **kwargs)

def test_pixel_prompt_organism_recognizes_full_factorial_and_wrong_shape(*args, **kwargs):
    return _legacy().test_pixel_prompt_organism_recognizes_full_factorial_and_wrong_shape(*args, **kwargs)

def test_bbox_to_patch_indices(*args, **kwargs):
    return _legacy().test_bbox_to_patch_indices(*args, **kwargs)

def test_matched_controls_are_true_background_equal_distinct_and_nonoverlapping(*args, **kwargs):
    return _legacy().test_matched_controls_are_true_background_equal_distinct_and_nonoverlapping(*args, **kwargs)

def test_cache_has_pixel_and_prompt_routing_stages(*args, **kwargs):
    return _legacy().test_cache_has_pixel_and_prompt_routing_stages(*args, **kwargs)

def test_binary_metrics_are_semantic(*args, **kwargs):
    return _legacy().test_binary_metrics_are_semantic(*args, **kwargs)

def test_replace_token_activations_is_exact_and_stub_sensitive(*args, **kwargs):
    return _legacy().test_replace_token_activations_is_exact_and_stub_sensitive(*args, **kwargs)

def test_counterfactual_patch_sweep_is_causal_and_controlled(*args, **kwargs):
    return _legacy().test_counterfactual_patch_sweep_is_causal_and_controlled(*args, **kwargs)

def test_validation_selects_layer_and_test_remains_untouched(*args, **kwargs):
    return _legacy().test_validation_selects_layer_and_test_remains_untouched(*args, **kwargs)

def test_parameterized_playground_changes_strength_and_region(*args, **kwargs):
    return _legacy().test_parameterized_playground_changes_strength_and_region(*args, **kwargs)

def test_toy_signature_and_smoke_fixture(*args, **kwargs):
    return _legacy().test_toy_signature_and_smoke_fixture(*args, **kwargs)

def test_vlm_geometry_controls_exclude_object_and_distractor(*args, **kwargs):
    return _legacy().test_vlm_geometry_controls_exclude_object_and_distractor(*args, **kwargs)

def test_real_acceptance_requires_behavior_and_directional_controls(*args, **kwargs):
    return _legacy().test_real_acceptance_requires_behavior_and_directional_controls(*args, **kwargs)

def test_real_acceptance_rejects_wrong_identity_nonfinite_and_impossible_rates(*args, **kwargs):
    return _legacy().test_real_acceptance_rejects_wrong_identity_nonfinite_and_impossible_rates(*args, **kwargs)

def test_real_acceptance_recomputes_raw_rows_and_denominators(*args, **kwargs):
    return _legacy().test_real_acceptance_recomputes_raw_rows_and_denominators(*args, **kwargs)

def test_capture_module_output_removes_hook(*args, **kwargs):
    return _legacy().test_capture_module_output_removes_hook(*args, **kwargs)

def test_exercise_notebook_has_arena_surface_and_immediate_stub_tests(*args, **kwargs):
    return _legacy().test_exercise_notebook_has_arena_surface_and_immediate_stub_tests(*args, **kwargs)

def test_solution_notebook_keeps_taught_causal_methods_inline_and_is_executed(*args, **kwargs):
    return _legacy().test_solution_notebook_keeps_taught_causal_methods_inline_and_is_executed(*args, **kwargs)

def test_instruction_page_places_images_beside_prose_and_uses_honest_number(*args, **kwargs):
    return _legacy().test_instruction_page_places_images_beside_prose_and_uses_honest_number(*args, **kwargs)

def test_local_report_schema_and_manifest_are_strict_and_fresh(*args, **kwargs):
    return _legacy().test_local_report_schema_and_manifest_are_strict_and_fresh(*args, **kwargs)

def test_real_path_is_pinned_cuda_only_validation_selected_and_has_no_invalid_equality_gate(*args, **kwargs):
    return _legacy().test_real_path_is_pinned_cuda_only_validation_selected_and_has_no_invalid_equality_gate(*args, **kwargs)

def test_dropdowns_complete_learner_without_missing_helpers(*args, **kwargs):
    return _legacy().test_dropdowns_complete_learner_without_missing_helpers(*args, **kwargs)
