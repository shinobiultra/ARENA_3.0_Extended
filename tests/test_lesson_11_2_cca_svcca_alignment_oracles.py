"""11.2 CCA, SVCCA and alignment: tiny CPU oracles run against the lesson solutions."""
from __future__ import annotations

import importlib

import pytest

MODULE = "chapter11_representation_geometry.exercises.part2_cca_svcca_alignment"


@pytest.fixture(scope="module")
def methods():
    return importlib.import_module(MODULE + ".solutions")


ORACLES = [
    "test_whitening_covariance_rank_and_training_only_fit",
    "test_canonical_correlations_match_known_oracle_under_affine_maps",
    "test_heldout_direction_is_fixed_and_negative_correlation_is_visible",
    "test_independent_high_width_views_overfit_training_and_fail_heldout",
    "test_svcca_variance_cutoff_can_remove_a_low_variance_feature",
    "test_procrustes_preserves_lengths_and_cannot_undo_anisotropic_scaling",
    "test_canonical_transport_equals_exact_linear_map_and_zero_rank_noop",
    "test_ridge_transport_matches_primal_oracle_and_unpenalized_intercept",
    "test_distribution_kl_uses_full_vocabulary_and_correct_orientation",
]


@pytest.mark.parametrize("oracle", ORACLES)
def test_generated_visible_methods_against_independent_cpu_oracles(methods, oracle):
    tests = importlib.import_module(MODULE + ".alignment_methods_tests")
    torch = importlib.import_module("torch")
    previous_dtype = torch.get_default_dtype()
    try:
        # Reproduce the historical process default and restore ambient test state.
        # Most mathematical fixtures explicitly request float64 themselves.
        torch.set_default_dtype(torch.float32)
        getattr(tests, oracle)(methods)
    finally:
        torch.set_default_dtype(previous_dtype)
