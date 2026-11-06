"""11.3 Causal abstraction and DAS: tiny CPU oracles run against the lesson solutions."""
from __future__ import annotations

import importlib

import pytest

MODULE = "chapter11_representation_geometry.exercises.part3_causal_abstraction_das"


@pytest.fixture(scope="module")
def methods():
    return importlib.import_module(MODULE + ".solutions")


ORACLES = [
    "test_high_level_mechanisms",
    "test_projector_matches_explicit_rotate_swap_unrotate",
    "test_orthonormal_basis_limits_and_invalid_columns",
    "test_disjoint_interventions_commute_and_same_variable_overwrites",
    "test_qr_and_independent_finite_difference_gradient",
    "test_full_vocabulary_loss_and_empty_iia",
    "test_predictive_unused_copy_fails_causal_interchange",
    "test_alignment_optimization_changes_subspace_and_freezes_model",
]


@pytest.mark.parametrize("oracle", ORACLES)
def test_generated_visible_methods_against_independent_cpu_oracles(methods, oracle):
    tests = importlib.import_module(MODULE + ".das_methods_tests")
    torch = importlib.import_module("torch")
    previous_dtype, previous_threads = torch.get_default_dtype(), torch.get_num_threads()
    try:
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(2)
        getattr(tests, oracle)(methods)
    finally:
        torch.set_default_dtype(previous_dtype)
        torch.set_num_threads(previous_threads)


@pytest.mark.parametrize("oracle,method", [("test_role_probe", "fit_role_probe"), ("test_minibatch_alignment", "fit_minibatch_alignment")])
def test_generated_training_methods_against_independent_oracles(methods, oracle, method):
    tests = importlib.import_module(MODULE + ".das_training_tests")
    torch = importlib.import_module("torch")
    previous_dtype, previous_threads = torch.get_default_dtype(), torch.get_num_threads()
    try:
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(2)
        getattr(tests, oracle)(getattr(methods, method))
    finally:
        torch.set_default_dtype(previous_dtype)
        torch.set_num_threads(previous_threads)


def test_frozen_ioi_data_contract_and_visible_role_targets(methods):
    torch = importlib.import_module("torch")
    checks = importlib.import_module(MODULE + ".das_training_tests")
    checks.test_ioi_intervention_contract()
    names = torch.tensor([[11, 29], [11, 29], [37, 53], [37, 53]])
    roles = torch.tensor([0, 1, 0, 1])
    assert methods.role_interchange_targets(names, roles).tolist() == [29, 11, 53, 37]
    assert torch.equal(names, torch.tensor([[11, 29], [11, 29], [37, 53], [37, 53]]))
