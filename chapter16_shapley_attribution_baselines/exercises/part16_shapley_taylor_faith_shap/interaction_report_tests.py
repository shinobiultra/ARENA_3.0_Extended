"""Synthetic-only regression checks for the retained-reference report frontend.

The fixture contains seeded Gaussian cooperative games, not Concrete model
outputs. It never reads a dataset, fits a predictor or accesses an audit split.
"""

import copy

import numpy as np
import pytest
import torch

from chapter16_shapley_attribution_baselines.exercises.part16_shapley_taylor_faith_shap import (
    interaction_methods as methods,
    interaction_reference as reference,
    solutions,
)
from chapter16_shapley_attribution_baselines.exercises.part16_shapley_taylor_faith_shap.interaction_study import pairwise_sii


@pytest.fixture(scope="module")
def synthetic_reference():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(4)
    try:
        values = torch.from_numpy(np.random.default_rng(161616).normal(size=(2992, 256)))
        mobius = methods.mobius_transform(values, n_players=8)
        stii = methods.shapley_taylor_interactions(mobius, n_players=8, order=2)
        fsii = methods.faith_shap_interactions(values, n_players=8, max_order=2)
        sii = pairwise_sii(values, n_players=8)
        result = reference.reference_check(values, stii, fsii, sii, n_players=8, order=2, expected_game_count=2992)
        arrays = {"coalition_values": values, "stii_coefficients": stii, "fsii_coefficients": fsii, "sii_pairs": sii}
        primary = {name: array[:1904].reshape(7, 272, -1) for name, array in arrays.items()}
        variants = {}
        for index, name in enumerate(("nested16", "nested128", "independent64_a", "independent64_b")):
            variants[name] = {key: array[1904 + index * 272 : 1904 + (index + 1) * 272] for key, array in arrays.items()}
        evaluated = {"all_rows": {"primary64": primary, "background_variants": variants}}
        yield result, evaluated
    finally:
        torch.set_num_threads(previous_threads)


def test_complete_reference_metrics_are_derived_from_actual_arrays(synthetic_reference):
    reference_result, evaluated = synthetic_reference
    measured = solutions._reference_metrics(reference_result, evaluated)
    assert measured["reference_passed"] is True
    assert measured["reference_games"] == 2992
    assert measured["reference_student_coordinates"] == 1615680
    assert measured["reference_raw_package_coordinates"] == 332112
    assert measured["reference_relevant_package_coordinates"] == 305184
    assert measured["reference_package_index_calls"] == 8976
    assert measured["reference_max_error_to_bound_ratio"] <= 1


def test_report_rejects_truncated_package_vector_even_with_pass_labels(synthetic_reference):
    result, evaluated = synthetic_reference
    changed = copy.deepcopy(result)
    changed["packages"]["shapiq"]["fsii"] = changed["packages"]["shapiq"]["fsii"][:-1]
    with pytest.raises(ValueError, match="package FSII"):
        solutions._reference_metrics(changed, evaluated)


def test_report_recomputes_reference_residuals(synthetic_reference):
    result, evaluated = synthetic_reference
    changed = copy.deepcopy(result)
    changed["errors"]["shapiq_fsii_minus_independent"][0, 0] += 0.5
    with pytest.raises(ValueError, match="residual is inconsistent"):
        solutions._reference_metrics(changed, evaluated)


def test_report_binds_reference_student_to_retained_game_results(synthetic_reference):
    result, evaluated = synthetic_reference
    changed = copy.deepcopy(result)
    changed["student"]["stii"][0, 3] += 1
    with pytest.raises(ValueError, match="differs from the learner"):
        solutions._reference_metrics(changed, evaluated)


def test_report_rejects_package_axis_permutation(synthetic_reference):
    result, evaluated = synthetic_reference
    changed = copy.deepcopy(result)
    changed["masks"]["package_terms"] = changed["masks"]["package_terms"].flip(0)
    with pytest.raises(ValueError, match="mask order"):
        solutions._reference_metrics(changed, evaluated)


def test_report_checks_numerical_bound_independently_of_pass_metadata(synthetic_reference):
    result, evaluated = synthetic_reference
    changed = copy.deepcopy(result)
    changed["packages"]["shapiq"]["stii"][0, 3] += 1
    mask = int(changed["masks"]["package_terms"][3])
    changed["errors"]["shapiq_stii_minus_independent"][0, 3] = (
        changed["packages"]["shapiq"]["stii"][0, 3]
        - changed["independent"]["stii"][0, mask]
    )
    with pytest.raises(ValueError, match="pre-native numerical bound"):
        solutions._reference_metrics(changed, evaluated)
