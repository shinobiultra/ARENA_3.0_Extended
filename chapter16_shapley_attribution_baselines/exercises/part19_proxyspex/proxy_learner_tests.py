"""Immediate checks for grouped notebook exercises and plot input contracts."""

import numpy as np
import pytest

from chapter16_shapley_attribution_baselines.exercises.part19_proxyspex import proxy_methods as answers
from chapter16_shapley_attribution_baselines.exercises.part19_proxyspex.proxy_methods_tests import cube, direct_character


def test_predict_spectrum(function=answers.predict_spectrum):
    masks = cube(4)
    spectrum = {(): .7, (0,): -.3, (1, 3): 1.2}
    expected = .7 - .3 * direct_character(masks, (0,)) + 1.2 * direct_character(masks, (1, 3))
    np.testing.assert_allclose(function(masks, spectrum), expected, atol=1e-13)
    np.testing.assert_array_equal(function(masks, {}), np.zeros(len(masks)))
    print("predict_spectrum passed: signed reconstruction and an empty spectrum.")


def test_predict_boosted_trees(function=answers.predict_boosted_trees):
    masks = cube(4)
    values = .3 + masks[:, 0] + 2 * masks[:, 0] * masks[:, 1]
    model = answers.fit_boosted_trees(masks, values, max_depth=2, n_estimators=4, min_samples_leaf=1)
    expected = np.full(len(masks), model["initial_value"])
    for tree in model["trees"]:
        expected += model["learning_rate"] * tree.predict(masks)
    np.testing.assert_allclose(function(model, masks), expected, atol=1e-13)
    with pytest.raises(ValueError):
        function(model, cube(3))
    print("predict_boosted_trees passed: all shrinkage-weighted trees plus the initial mean.")


def test_fit_proxy(function=answers.fit_proxy):
    masks = cube(5)
    values = np.arange(len(masks)) / 100 + .4 * direct_character(masks, (1, 3))
    candidates = ({"max_depth": 1, "n_estimators": 3, "min_samples_leaf": 1},
                  {"max_depth": 2, "n_estimators": 5, "min_samples_leaf": 1})
    namespace = function.__globals__
    original = namespace["fit_boosted_trees"]
    observations, models = [], []

    def observe(rows, responses, **parameters):
        observations.append((rows.copy(), responses.copy()))
        model = original(rows, responses, **parameters)
        models.append(model)
        return model

    namespace["fit_boosted_trees"] = observe
    try:
        result = function(masks, values, fold_seed=21, candidates=candidates, ridge_alphas=(.01, 1.0))
    finally:
        namespace["fit_boosted_trees"] = original
    assert len(observations) == 7
    for candidate in range(2):
        for fold, record in enumerate(result["folds"]):
            rows, responses = observations[candidate * 3 + fold]
            np.testing.assert_array_equal(rows, masks[record["train_indices"]])
            np.testing.assert_array_equal(responses, values[record["train_indices"]])
    selected = result["proxy_cv"]["selected_index"]
    for fold in range(3):
        spectrum = namespace["ensemble_to_fourier"](models[selected * 3 + fold])
        assert result["ridge_cv"]["fold_supports"][fold] == namespace["select_energy_support"](spectrum)["supports"]
    assert result["accounting"]["residual_tree_fits"] == 24 + candidates[selected]["n_estimators"]
    assert result["accounting"]["ridge_candidate_fits"] == 7
    assert result["accounting"]["oracle_calls_by_methods"] == 0
    expected = np.full(len(masks), result["model"]["initial_value"])
    for tree in result["model"]["trees"]:
        expected += result["model"]["learning_rate"] * tree.predict(masks)
    np.testing.assert_allclose(namespace["predict_spectrum"](masks, result["full_spectrum"]), expected, atol=1e-12)
    with pytest.raises(TypeError, match="audit_values"):
        function(masks, values, fold_seed=21, audit_values=np.ones(30))
    print("fit_proxy passed: observed fold isolation, fold-specific supports, exact fit counts and no audit argument.")


def test_native_plots_reject_unexecuted_payload():
    from chapter16_shapley_attribution_baselines.exercises.part19_proxyspex import proxy_plots
    with pytest.raises(ValueError, match="unexecuted"):
        proxy_plots.plot_native_stages(None, prompt_index=0, seed=0)
    with pytest.raises(ValueError, match="unexecuted"):
        proxy_plots.plot_native_spectrum({"results": []}, prompt_index=0, seed=0, budget=456)
