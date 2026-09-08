"""Independent reference, identity, algebra, and boundary tests for DeepSHAP."""

from __future__ import annotations

import copy
import inspect

import numpy as np
import pytest
import torch as t

from . import deepshap_reference as reference


def _identity_fixture():
    data, ids = reference._frozen_identity()
    positions = reference._background_indices(ids["train"])
    background_ids = ids["train"][positions]
    background = data[background_ids, :8].copy()
    fitted = {
        "train_ids": ids["train"].copy(),
        "train_x_raw": data[ids["train"], :8].copy(),
        "train_y_raw": data[ids["train"], 8].copy(),
    }
    prepared = {
        "explain_ids": ids["explain"].copy(),
        "explain_x_raw": data[ids["explain"], :8].copy(),
        "explain_targets": data[ids["explain"], 8].copy(),
        "background_ids": background_ids.copy(),
        "background": background.copy(),
        "background_indices": {"primary64": positions.copy()},
    }
    evaluated = {
        "audit_ids": ids["audit"].copy(),
        "audit_targets": data[ids["audit"], 8].copy(),
        "combined_explanation_audit_ids": np.concatenate(
            [ids["explain"], ids["audit"]]
        ),
        "combined_split": np.concatenate(
            [np.zeros(153, dtype=np.int64), np.ones(119, dtype=np.int64)]
        ),
        "background_variants": {
            "primary64": {
                "background_ids": background_ids.copy(),
                "background": background.copy(),
                "row_ids": np.concatenate([ids["explain"], ids["audit"]]),
                "row_split": np.concatenate(
                    [np.zeros(153, dtype=np.int64), np.ones(119, dtype=np.int64)]
                ),
            }
        },
    }
    rows = {
        name: {
            "ids": split_ids.copy(),
            "x": data[split_ids, :8].copy(),
            "y": data[split_ids, 8].copy(),
        }
        for name, split_ids in ids.items()
    }
    return fitted, prepared, evaluated, rows["train"], rows["explain"], rows["audit"]


def _affine_model() -> t.nn.Sequential:
    model = t.nn.Sequential(t.nn.Linear(2, 1, dtype=t.float64))
    with t.no_grad():
        model[0].weight.copy_(t.tensor([[2.0, -3.0]], dtype=t.float64))
        model[0].bias.copy_(t.tensor([5.0], dtype=t.float64))
    model.eval()
    return model


def test_reference_sources_and_installed_deep_explainer_are_pinned():
    environment = reference._reference_environment()
    assert environment["versions"] == {
        "numpy": "1.26.4",
        "torch": "2.12.1+cu132",
        "scikit_learn": "1.9.0",
        "shap": "0.49.1",
    }
    assert len(environment["installed_source_sha256"]) == 3
    assert len(environment["paper_pdf_sha256"]) == 3
    assert environment["historical_deeplift_commit"] == "0201a218965a263b9dd353099feacbb6f6db0051"


def test_permutation_oracle_matches_independent_subset_weights_for_signed_games():
    games = np.array(
        [
            [1.5, 2.8, 2.5, 3.0],
            [0.0, -2.0, 5.0, 11.0],
            [3.0, 3.0, 3.0, 7.0],
        ],
        dtype=np.float64,
    )
    permutation, count = reference.permutation_reference(games)
    subset = reference.subset_reference(games)
    assert count == 2
    np.testing.assert_allclose(permutation, subset, atol=1e-14, rtol=1e-14)
    np.testing.assert_allclose(permutation.sum(1), games[:, -1] - games[:, 0], atol=1e-14)


def test_ridge_closed_form_uses_every_background_row_and_raw_units():
    x = np.array([[4.0, 2.0], [-1.0, 3.0]])
    background = np.array([[1.0, -1.0], [2.0, 0.0]])
    result = reference.ridge_closed_form([[2.0, -3.0]], [5.0], x, background)
    assert result["base"] == 9.5
    np.testing.assert_array_equal(result["prediction"], [7.0, -6.0])
    np.testing.assert_array_equal(result["phi"][0], [5.0, -7.5])
    np.testing.assert_allclose(
        result["base"] + result["phi"].sum(1), result["prediction"], atol=1e-14
    )


def test_same_total_wrong_feature_allocation_is_rejected():
    exact = np.array([[3.0, -1.0, 2.0]])
    wrong = np.array([[2.0, 0.0, 2.0]])
    assert exact.sum() == wrong.sum()
    with pytest.raises(AssertionError, match="full-value parity"):
        reference._require_close("mutated local vector", wrong, exact)


@pytest.mark.parametrize(
    "mutation",
    ("background_value", "background_id_order", "explain_row_order", "audit_row_order"),
)
def test_identity_refuses_wrong_background_or_row_order(mutation):
    fixture = list(_identity_fixture())
    fitted, prepared, evaluated, train, explain, audit = fixture
    if mutation == "background_value":
        prepared["background"][0] = prepared["background"][1]
    elif mutation == "background_id_order":
        prepared["background_ids"][[0, 1]] = prepared["background_ids"][[1, 0]]
    elif mutation == "explain_row_order":
        explain["ids"][[0, 1]] = explain["ids"][[1, 0]]
    else:
        evaluated["audit_ids"][[0, 1]] = evaluated["audit_ids"][[1, 0]]
    with pytest.raises(ValueError, match="background|row identity/order|row identity"):
        reference._validate_identity(fitted, prepared, evaluated, train, explain, audit)


def test_identity_accepts_exactly_all_64_float64_primary_donors():
    identity = reference._validate_identity(*_identity_fixture())
    assert identity["background"].shape == (64, 8)
    assert identity["background"].dtype == np.float64
    assert identity["background_ids"].shape == (64,)
    assert len(np.unique(identity["background_ids"])) == 64
    assert identity["explain_x"].shape == (153, 8)
    assert identity["audit_x"].shape == (119, 8)


def test_author_call_keeps_all_background_rows_and_model_state():
    model = _affine_model()
    background = np.column_stack(
        [np.arange(64, dtype=np.float64), -np.arange(64, dtype=np.float64)]
    )
    x = np.array([[4.0, 2.0], [3.0, -1.0]], dtype=np.float64)
    state_before = reference._model_hash(model)
    result = reference._author_deep(model, x, background)
    analytic = reference.ridge_closed_form([[2.0, -3.0]], [5.0], x, background)
    np.testing.assert_allclose(result["phi"], analytic["phi"], atol=1e-12, rtol=1e-12)
    assert result["base"] == analytic["base"]
    assert reference._model_hash(model) == state_before
    assert model.training is False
    source = inspect.getsource(reference._author_deep)
    assert "check_additivity=False" in source
    assert "shap.sample" not in source


def test_bounded_single_donor_author_values_average_to_full_background_value():
    model = _affine_model()
    x = np.array([[4.0, 2.0]], dtype=np.float64)
    background = np.array([[1.0, -1.0], [2.0, 0.0], [-2.0, 3.0]], dtype=np.float64)
    full = reference._author_deep(model, x, background)
    bounded = reference._bounded_pair_author(model, x, background)
    np.testing.assert_allclose(bounded["phi"].mean(0), full["phi"][0], atol=1e-14)
    np.testing.assert_allclose(bounded["base"].mean(), full["base"], atol=1e-14)
    np.testing.assert_allclose(
        bounded["base"] + bounded["phi"].sum(1), bounded["prediction"], atol=1e-14
    )


def test_known_threshold_min_equivalence_and_nearzero_failure_are_disclosed():
    diagnostic = reference.known_diagnostics()
    np.testing.assert_allclose(diagnostic["threshold"]["exact"], [[0.5, 0.5]], atol=1e-14)
    np.testing.assert_allclose(
        diagnostic["threshold"]["deep_explainer"], [[2 / 3, 1 / 3]], atol=1e-14
    )
    np.testing.assert_array_equal(diagnostic["threshold"]["single_feature_necessity"], [[1, 1]])
    np.testing.assert_array_equal(diagnostic["minimum"]["minimum_outputs"], [0, 0, 0, 1])
    np.testing.assert_array_equal(
        diagnostic["minimum"]["minimum_outputs"], diagnostic["minimum"]["mirrored_outputs"]
    )
    np.testing.assert_array_equal(diagnostic["minimum"]["deep_explainer"], [[0, 1]])
    np.testing.assert_array_equal(
        diagnostic["minimum"]["mirrored_deep_explainer"], [[1, 0]]
    )
    assert diagnostic["nearzero"]["conservation_passed_at_primary_tolerance"] is False
    assert diagnostic["nearzero"]["expected_failure_disclosed"] is True
    np.testing.assert_array_equal(diagnostic["nearzero"]["deep_explainer"], [[0]])
    np.testing.assert_array_equal(diagnostic["nearzero"]["epsilon_zero_secant"], [[4e-7]])


def test_constant_dummy_and_multiple_background_controls_are_exact():
    diagnostic = reference.known_diagnostics()
    np.testing.assert_array_equal(diagnostic["constant"]["deep_explainer"], [[0, 0]])
    np.testing.assert_array_equal(diagnostic["dummy"]["deep_explainer"][:, 2], [0])
    np.testing.assert_array_equal(diagnostic["dummy"]["exact"][:, 2], [0])
    np.testing.assert_allclose(
        diagnostic["two_background_relu"]["deep_explainer"], [[1.5, 1.5]], atol=1e-14
    )
    np.testing.assert_allclose(
        diagnostic["two_background_relu"]["exact"], [[1.5, 1.5]], atol=1e-14
    )


def test_reference_model_boundary_rejects_unsupported_shared_inplace_and_subclasses():
    class SequentialSubclass(t.nn.Sequential):
        pass

    class LinearSubclass(t.nn.Linear):
        pass

    invalid = [
        t.nn.Sequential(t.nn.Linear(2, 2, dtype=t.float64), t.nn.Sigmoid(), t.nn.Linear(2, 1, dtype=t.float64)),
        t.nn.Sequential(t.nn.Linear(2, 2, dtype=t.float64), t.nn.ReLU(inplace=True), t.nn.Linear(2, 1, dtype=t.float64)),
        SequentialSubclass(t.nn.Linear(2, 1, dtype=t.float64)),
        t.nn.Sequential(LinearSubclass(2, 1, dtype=t.float64)),
    ]
    shared = t.nn.ReLU()
    invalid.append(
        t.nn.Sequential(
            t.nn.Linear(2, 2, dtype=t.float64),
            shared,
            t.nn.Linear(2, 2, dtype=t.float64),
            shared,
            t.nn.Linear(2, 1, dtype=t.float64),
        )
    )
    for model in invalid:
        with pytest.raises((TypeError, ValueError)):
            reference._validate_reference_model(model)


def test_reference_does_not_import_learner_attribution_or_study_core():
    source = inspect.getsource(reference)
    assert "import deepshap_methods" not in source
    assert "import deepshap_study" not in source
    checker = inspect.getsource(reference.validate_reference)
    assert "subset_reference" in checker
    assert "_author_deep" in checker
    assert "_bounded_pair_author" in checker
    assert "ridge_closed_form" in checker


def test_block_requires_complete_finite_pair_tensor():
    rows = 2
    template = {
        "prediction": np.zeros(rows),
        "base": np.array(0.0),
        "phi_deep": np.zeros((rows, 8)),
        "phi_exact": np.zeros((rows, 8)),
        "phi_gradient": np.zeros((rows, 8)),
        "game": np.zeros((rows, 256)),
        "pair_deep": np.zeros((rows, 64, 8)),
    }
    block = {key: copy.deepcopy(template) for key in reference.MODEL_KEYS}
    accepted = reference._validate_block("fixture", block, rows)
    assert accepted["ridge"]["pair_deep"].shape == (2, 64, 8)
    block["clean_0"]["pair_deep"][1, 63, 7] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        reference._validate_block("fixture", block, rows)
