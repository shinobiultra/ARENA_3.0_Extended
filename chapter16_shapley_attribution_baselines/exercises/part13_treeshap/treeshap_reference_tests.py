"""Independent reference, identity, mutation, and threshold-boundary tests."""

from __future__ import annotations

import copy
import inspect
import itertools

import numpy as np
import pytest
import shap
from sklearn.tree import DecisionTreeRegressor

from . import treeshap_reference as reference


def _identity_fixture():
    data, ids = reference._frozen_identity()
    order = reference._background_indices(ids["train"])
    background_ids = ids["train"][order]
    fitted = {"train_ids": ids["train"].copy()}
    prepared = {
        "explain_ids": ids["explain"].copy(),
        "background_ids": background_ids.copy(),
        "background": data[background_ids, :8].copy(),
    }
    evaluated = {
        "audit_ids": ids["audit"].copy(),
        "audit_targets": data[ids["audit"], 8].copy(),
    }
    explain = {
        "ids": ids["explain"].copy(),
        "x": data[ids["explain"], :8].copy(),
        "y": data[ids["explain"], 8].copy(),
    }
    audit = {
        "ids": ids["audit"].copy(),
        "x": data[ids["audit"], :8].copy(),
        "y": data[ids["audit"], 8].copy(),
    }
    return fitted, prepared, evaluated, explain, audit


def _coalition_masks(d):
    return np.array(
        [[bool(mask & (1 << feature)) for feature in range(d)] for mask in range(2**d)]
    )


def _path_game(model, rows):
    """Small independent recursive definition using sklearn's stored covers."""
    tree = model.tree_
    masks = _coalition_masks(rows.shape[1])

    def value(node, row, mask):
        left, right = tree.children_left[node], tree.children_right[node]
        if left == -1:
            return tree.value[node, 0, 0]
        feature = tree.feature[node]
        if mask[feature]:
            child = left if row[feature] <= tree.threshold[node] else right
            return value(child, row, mask)
        left_weight = tree.weighted_n_node_samples[left] / tree.weighted_n_node_samples[node]
        return left_weight * value(left, row, mask) + (1 - left_weight) * value(
            right, row, mask
        )

    return np.array([[value(0, row, mask) for mask in masks] for row in rows])


def test_reference_sources_and_compiled_extension_are_pinned():
    environment = reference._reference_environment()
    assert environment["versions"]["shap"] == "0.49.1"
    assert environment["boundary_addendum_sha256"] == reference.ADDENDUM_SHA256
    assert environment["cached_source_sha256"]["cext/tree_shap.h"]
    assert environment["cext_module"].endswith(('.so', '.pyd'))


def test_permutation_oracle_is_exact_signed_and_batched():
    games = np.array(
        [
            [1.5, 2.8, 2.5, 3.0],
            [0.0, -2.0, 5.0, 11.0],
            [3.0, 3.0, 3.0, 7.0],
        ]
    )
    phi, count = reference.permutation_reference(games)
    assert count == 2
    np.testing.assert_allclose(phi, reference.subset_reference(games), atol=1e-14)
    np.testing.assert_allclose(phi.sum(axis=1), games[:, -1] - games[:, 0], atol=1e-14)


def test_same_total_wrong_feature_allocation_is_rejected():
    exact = np.array([[3.0, -1.0, 2.0]])
    wrong = np.array([[2.0, 0.0, 2.0]])
    assert exact.sum() == wrong.sum()
    with pytest.raises(AssertionError, match="full-value parity"):
        reference._require_close("mutated local vector", wrong, exact)


@pytest.mark.parametrize("mutation", ["background_value", "background_id_order", "row_order"])
def test_frozen_identity_refuses_wrong_background_or_row_order(mutation):
    fitted, prepared, evaluated, explain, audit = _identity_fixture()
    if mutation == "background_value":
        prepared["background"][0] = prepared["background"][1]
    elif mutation == "background_id_order":
        prepared["background_ids"][[0, 1]] = prepared["background_ids"][[1, 0]]
    else:
        explain["ids"][[0, 1]] = explain["ids"][[1, 0]]
    with pytest.raises(ValueError, match="background|row identity/order"):
        reference._validate_identity(fitted, prepared, evaluated, explain, audit)


def test_all_64_float32_background_rows_are_accepted_exactly():
    fixture = _identity_fixture()
    all_x, explain_x, audit_x, background, ids = reference._validate_identity(*fixture)
    assert all_x.shape == (1030, 8)
    assert explain_x.shape == (153, 8)
    assert audit_x.shape == (119, 8)
    assert background.shape == (64, 8)
    assert background.dtype == np.float32
    assert len(np.unique(ids["train"])) == 758


def test_floor_float32_is_route_equivalent_for_positive_and_negative_thresholds():
    lower = np.array([np.float32(10.1), np.float32(-10.1)], dtype=np.float32)
    upper = np.nextafter(lower, np.float32(np.inf))
    original = lower.astype(np.float64) + 0.75 * (
        upper.astype(np.float64) - lower.astype(np.float64)
    )
    nearest, corrected, rounded_up = reference._floor_float32_thresholds(original)
    assert rounded_up.all()
    np.testing.assert_array_equal(corrected, lower)
    assert np.all(corrected.astype(np.float64) <= original)
    assert np.all(np.nextafter(corrected, np.float32(np.inf)).astype(np.float64) > original)
    neighbors = np.stack(
        [np.nextafter(corrected, np.float32(-np.inf)), corrected, np.nextafter(corrected, np.float32(np.inf))],
        axis=1,
    )
    expected = np.tile([True, True, False], (2, 1))
    np.testing.assert_array_equal(neighbors.astype(np.float64) <= original[:, None], expected)
    np.testing.assert_array_equal(neighbors <= corrected[:, None], expected)
    assert np.all(nearest > corrected)


def test_sklearn_and_path_reference_use_less_equal_at_float32_boundary():
    x_train = np.array([[0.0], [0.0], [2.0], [2.0]])
    model = DecisionTreeRegressor(max_depth=1, random_state=0).fit(
        x_train, np.array([10.0, 10.0, 30.0, 30.0])
    )
    model.tree_.threshold[0] = 1.0
    at = np.float32(1.0)
    rows = np.array(
        [[np.nextafter(at, np.float32(-np.inf))], [at], [np.nextafter(at, np.float32(np.inf))]],
        dtype=np.float32,
    )
    prediction = model.predict(rows)
    np.testing.assert_array_equal(prediction, [10.0, 10.0, 30.0])
    explainer = shap.TreeExplainer(
        model, model_output="raw", feature_perturbation="tree_path_dependent"
    )
    phi = explainer.shap_values(rows, approximate=False, check_additivity=True)
    base = np.asarray(explainer.expected_value).item()
    np.testing.assert_allclose(base + phi.sum(axis=1), prediction)


def test_repeated_feature_is_one_player_for_both_games():
    x0 = np.repeat(np.arange(8, dtype=np.float32), 2)
    x_train = np.column_stack([x0, np.tile([0.0, 1.0], 8)]).astype(np.float32)
    y = np.array([0.0, 0.0, 1.0, 1.0, 7.0, 7.0, 8.0, 8.0,
                  20.0, 20.0, 21.0, 21.0, 40.0, 40.0, 41.0, 41.0])
    model = DecisionTreeRegressor(max_depth=3, random_state=3).fit(x_train, y)
    internal_features = model.tree_.feature[model.tree_.children_left >= 0]
    assert len(internal_features) >= 2
    np.testing.assert_array_equal(np.unique(internal_features), [0])
    rows = np.array([[2.0, 999.0], [7.0, -999.0]], dtype=np.float32)

    path_game = _path_game(model, rows)
    expected_path, permutation_count = reference.permutation_reference(path_game)
    assert permutation_count == 2
    path = shap.TreeExplainer(
        model, model_output="raw", feature_perturbation="tree_path_dependent"
    )
    np.testing.assert_allclose(
        path.shap_values(rows, approximate=False, check_additivity=True), expected_path, atol=1e-12
    )
    np.testing.assert_array_equal(expected_path[:, 1], 0.0)

    masks = _coalition_masks(2)
    game = np.array([
        [model.predict(np.where(mask, row, donor)[None])[0] for mask in masks]
        for row in rows for donor in x_train
    ]).reshape(len(rows), len(x_train), len(masks)).mean(axis=1)
    expected_interventional, _ = reference.permutation_reference(game)
    interventional = shap.TreeExplainer(
        model,
        data=x_train,
        model_output="raw",
        feature_perturbation="interventional",
    )
    np.testing.assert_allclose(
        interventional.shap_values(rows, approximate=False, check_additivity=True),
        expected_interventional,
        atol=1e-12,
    )
    np.testing.assert_array_equal(expected_interventional[:, 1], 0.0)


def test_threshold_copy_fixes_round_up_without_mutating_sklearn_model():
    model = DecisionTreeRegressor(max_depth=1, random_state=0).fit(
        np.array([[0.0], [0.0], [2.0], [2.0]]),
        np.array([10.0, 10.0, 30.0, 30.0]),
    )
    floor = np.float32(1.0)
    ceiling = np.nextafter(floor, np.float32(np.inf))
    model.tree_.threshold[0] = float(floor) + 0.75 * (float(ceiling) - float(floor))
    rows = np.array([[floor], [ceiling]], dtype=np.float32)
    original_hash = reference._model_hash(model)
    raw = shap.TreeExplainer(
        model,
        data=np.array([[0.0], [2.0]], dtype=np.float32),
        model_output="raw",
        feature_perturbation="interventional",
    )
    corrected, nearest, audit = reference._threshold_copies(raw, model, rows)
    assert audit["internal_threshold_count"] == 1
    assert audit["rounded_up_threshold_count"] == 1
    np.testing.assert_array_equal(audit["corrected_neighbor_routes"], [[True, True, False]])
    np.testing.assert_array_equal(audit["raw_neighbor_routes"], [[True, True, True]])
    np.testing.assert_array_equal(corrected.model.predict(rows), model.predict(rows))
    assert nearest.model.predict(rows)[1] != model.predict(rows)[1]
    assert reference._model_hash(model) == original_hash


def test_reference_call_is_explicit_and_checks_complete_vectors():
    source = inspect.getsource(reference._compiled_reference)
    assert 'feature_perturbation="tree_path_dependent"' in source
    assert 'feature_perturbation="interventional"' in source
    assert "approximate=False" in source
    assert "shap.sample" not in source
    checker = inspect.getsource(reference.check_reference)
    assert "subset_reference" in checker
    assert "phi_interventional_raw" in checker
    assert "phi_interventional_corrected" in checker
    assert "_require_close" in checker
