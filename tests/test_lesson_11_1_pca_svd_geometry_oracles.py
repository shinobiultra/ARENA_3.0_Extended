"""11.1 PCA, SVD and geometry: tiny CPU oracles run against the lesson solutions."""
from __future__ import annotations

import importlib

import pytest

MODULE = "chapter11_representation_geometry.exercises.part1_pca_svd_geometry_controls"


@pytest.fixture(scope="module")
def methods():
    return importlib.import_module(MODULE + ".solutions")


ORACLES = [
    "test_pca_fits_only_training_and_preserves_total_variance",
    "test_ridge_has_unpenalized_intercept_and_known_shrinkage",
    "test_template_centering_is_explicitly_transductive",
    "test_angles_wrap_around_and_zero_abstains",
    "test_calendar_order_null_is_distinct_from_class_identification",
    "test_neighborhood_ranks_match_independent_library_and_duality",
    "test_procrustes_recovers_rotation_reflection_and_translation",
    "test_minimum_norm_coordinate_edit_matches_constraint_solution",
    "test_donor_subspace_does_not_copy_complement",
    "test_exact_ring_plane_changes_answer_and_preserves_nuisance",
    "test_semantic_positions_respect_both_padding_sides_and_cleanup_hooks",
    "test_mean_ablation_and_literal_author_coordinates_answer_different_questions",
    "test_native_capture_preserves_baseline_and_cleans_up_on_failure",
    "test_token_offsets_do_not_assume_identical_prompt_lengths",
    "test_restricted_calendar_success_does_not_hide_full_vocabulary_failure",
    "test_least_squares_repeated_fits_and_rank_deficiency",
]


@pytest.mark.parametrize("oracle", ORACLES)
def test_generated_visible_methods_against_independent_cpu_oracles(methods, oracle):
    tests = importlib.import_module(MODULE + ".geometry_methods_tests")
    torch = importlib.import_module("torch")
    previous_dtype = torch.get_default_dtype()
    try:
        # The preserved native-hook oracle explicitly creates float32 model outputs.
        # Reproduce its historical float32 default even after another test changes it.
        torch.set_default_dtype(torch.float32)
        getattr(tests, oracle)(methods)
    finally:
        torch.set_default_dtype(previous_dtype)
