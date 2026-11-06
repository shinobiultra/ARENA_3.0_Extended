"""11.4 Concept erasure: tiny CPU oracles run against the lesson solutions."""
from __future__ import annotations

import importlib

import pytest

MODULE = "chapter11_representation_geometry.exercises.part4_concept_erasure"


@pytest.fixture(scope="module")
def methods():
    return importlib.import_module(MODULE + ".solutions")


ORACLES = [
    "test_centered_moments", "test_psd_support_and_rejection", "test_four_point_optimal_edit",
    "test_affine_mean_preservation_and_identity", "test_independent_kkt_optimality",
    "test_singular_covariance_and_randomized_invariants", "test_probe_removal_requires_refitting",
    "test_nonlinear_information_and_loss_boundary", "test_random_control_matches_each_edit_norm",
    "test_actual_vocabulary_outcome_metrics",
]


@pytest.mark.parametrize("oracle", ORACLES)
def test_generated_visible_methods_against_independent_cpu_oracles(methods, oracle):
    tests = importlib.import_module(MODULE + ".erasure_methods_tests")
    torch = importlib.import_module("torch")
    previous_dtype = torch.get_default_dtype()
    try:
        # The preserved native-hook oracle explicitly creates float32 model outputs.
        # Reproduce its historical float32 default even after another test changes it.
        torch.set_default_dtype(torch.float32)
        getattr(tests, oracle)(methods)
    finally:
        torch.set_default_dtype(previous_dtype)


def test_predeclared_splits_without_model_execution():
    tests = importlib.import_module(MODULE + ".erasure_methods_tests")
    tests.test_predeclared_data_splits()
