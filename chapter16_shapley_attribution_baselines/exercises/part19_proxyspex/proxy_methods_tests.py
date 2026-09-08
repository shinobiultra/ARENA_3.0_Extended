"""Independent analytic oracles and immediate feedback for student functions.

Run from repository root with `.venv/bin/python -m pytest <this file> -q`.
Tests accept a function argument so the notebook can check student definitions.
"""

import inspect
from itertools import combinations, product
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.tree import DecisionTreeRegressor

from chapter16_shapley_attribution_baselines.exercises.part19_proxyspex import proxy_methods as answers


def cube(n):
    return np.array(list(product((0, 1), repeat=n)), dtype=np.uint8)


def direct_character(masks, support):
    return np.array([(-1.0) ** sum(int(row[i]) for i in support) for row in masks])


def direct_fourier(masks, values):
    return {support: float(np.mean(values * direct_character(masks, support)))
            for degree in range(masks.shape[1] + 1)
            for support in combinations(range(masks.shape[1]), degree)}


def assert_spectrum(actual, expected, atol=1e-12):
    for support in set(actual) | set(expected):
        assert abs(actual.get(support, 0) - expected.get(support, 0)) <= atol, support


def explicit_tree(specification, n_features):
    """Independent tree fixture: (feature, threshold, left, right) or scalar leaf."""
    left, right, feature, threshold, values = [], [], [], [], []

    def add(spec):
        node = len(left)
        left.append(-1); right.append(-1); feature.append(-2); threshold.append(-2.0); values.append(0.0)
        if isinstance(spec, tuple):
            feature[node], threshold[node] = spec[:2]
            left[node], right[node] = add(spec[2]), add(spec[3])
        else:
            values[node] = float(spec)
        return node

    add(specification)
    return SimpleNamespace(n_features_in_=n_features, tree_=SimpleNamespace(
        children_left=np.array(left), children_right=np.array(right), feature=np.array(feature),
        threshold=np.array(threshold), value=np.array(values)[:, None, None]))


def traverse_tree(tree, masks):
    """Predict by routing rows; no Fourier operations used in this oracle."""
    result = []
    for row in masks:
        node = 0
        while tree.tree_.children_left[node] != -1:
            node = (tree.tree_.children_left[node] if row[tree.tree_.feature[node]] <= tree.tree_.threshold[node]
                    else tree.tree_.children_right[node])
        result.append(float(tree.tree_.value[node, 0, 0]))
    return np.array(result)


def test_character_matrix(function=answers.character_matrix):
    for n in range(1, 6):
        masks = cube(n)
        supports = [support for degree in range(n + 1) for support in combinations(range(n), degree)]
        actual = function(masks, supports)
        expected = np.column_stack([direct_character(masks, support) for support in supports])
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(actual.T @ actual, len(masks) * np.eye(len(supports)))
    assert function(cube(2), []).shape == (4, 0)
    for masks in ([[0.2, 1]], [[-1, 1]], [[np.nan, 1]], [], [0, 1]):
        with pytest.raises(ValueError):
            function(masks, [()])
    for supports in ([(1, 0)], [(0, 0)], [(2,)], [(True,)], [(-1,)]):
        with pytest.raises(ValueError):
            function(cube(2), supports)
    print("character_matrix passed: every Boolean character through five features, signs and orthogonality.")


def test_path_indicator_fourier(function=answers.path_indicator_fourier):
    masks = cube(3)
    for length in range(5):
        for path in product(tuple(product(range(3), (0, 1))), repeat=length):
            expected = np.array([all(row[i] == bit for i, bit in path) for row in masks], float)
            assert_spectrum(function(path), direct_fourier(masks, expected))
    assert function([(0, 0), (0, 1)]) == {}
    assert function([(2, 1)] * 4) == {(): 0.5, (2,): -0.5}
    with pytest.raises(ValueError):
        function([(0, 0.5)])
    print("path_indicator_fourier passed: repeated, contradictory and empty paths obey XOR multiplication.")


def test_tree_to_fourier(function=answers.tree_to_fourier):
    # The second split repeats feature 0; one large-valued leaf is unreachable.
    repeated = explicit_tree((0, 0.5, (0, 0.5, 2.0, 999.0), (1, 0.5, -3.0, 5.0)), 2)
    for tree in (repeated, explicit_tree(4.5, 2),
                 explicit_tree((0, -0.1, 800.0, -2.0), 2),
                 explicit_tree((1, 1.0, 3.0, 900.0), 2)):
        masks = cube(2)
        assert_spectrum(function(tree), direct_fourier(masks, traverse_tree(tree, masks)))
    for n in range(1, 6):
        masks = cube(n)
        rng = np.random.default_rng(100 + n)
        tree = DecisionTreeRegressor(max_depth=n, random_state=4).fit(masks, rng.normal(size=len(masks)))
        assert_spectrum(function(tree), direct_fourier(masks, tree.predict(masks)))
    print("tree_to_fourier passed: real fitted trees, constant leaves, repeated splits and unreachable branches.")


def test_known_signed_parity_tree(function=answers.tree_to_fourier):
    def parity_tree(remaining, bits):
        if not remaining:
            return -1.25 * (-1) ** sum(bits)
        return (remaining[0], 0.5, parity_tree(remaining[1:], bits + [0]),
                parity_tree(remaining[1:], bits + [1]))
    tree = explicit_tree(parity_tree([2, 0, 1], []), 4)
    masks = cube(4)
    assert_spectrum(function(tree), {(0, 1, 2): -1.25})
    np.testing.assert_array_equal(traverse_tree(tree, masks), -1.25 * direct_character(masks, (0, 1, 2)))
    print("known parity tree passed: negative third-order amplitude, with exact independent row routing.")


def test_fit_boosted_trees(function=answers.fit_boosted_trees):
    masks = cube(5)
    values = 0.3 + masks[:, 0] - 0.7 * masks[:, 3] + 1.2 * masks[:, 1] * masks[:, 4]
    model = function(masks, values, max_depth=2, n_estimators=7, learning_rate=0.3,
                     min_samples_leaf=1, seed=31)
    prediction = np.full(len(values), values.mean())
    assert model["initial_value"] == values.mean()
    assert len(model["trees"]) == 7
    for step, actual_tree in enumerate(model["trees"]):
        independent = DecisionTreeRegressor(max_depth=2, min_samples_leaf=1,
                                           criterion="squared_error", random_state=31 + step)
        independent.fit(masks, values - prediction)
        np.testing.assert_allclose(actual_tree.predict(masks), independent.predict(masks), atol=1e-13)
        prediction += 0.3 * independent.predict(masks)
        assert model["train_mse"][step + 1] == pytest.approx(np.mean((values - prediction) ** 2))
    assert np.max(np.diff(model["train_mse"])) <= 1e-12
    np.testing.assert_allclose(answers.predict_boosted_trees(model, masks), prediction)
    assert model["train_mse"][-1] < model["train_mse"][0]
    for keyword, bad in (("n_estimators", 0), ("max_depth", True), ("learning_rate", 2), ("min_samples_leaf", 0)):
        arguments = dict(max_depth=2, n_estimators=7, learning_rate=0.3, min_samples_leaf=1)
        arguments[keyword] = bad
        with pytest.raises(ValueError):
            function(masks, values, **arguments)
    print("fit_boosted_trees passed: all seven genuine residual fits match independent CART predictions.")


def test_ensemble_to_fourier(function=answers.ensemble_to_fourier):
    masks = cube(6)
    games = (np.full(len(masks), -2.0),
             0.3 + masks[:, 0] - 0.7 * masks[:, 3],
             -1.25 * direct_character(masks, (1, 3, 5)),
             0.3 + masks[:, 0] - 0.7 * masks[:, 3] + 1.2 * masks[:, 1] * masks[:, 4]
             - 2.4 * masks[:, 0] * masks[:, 2] * masks[:, 5])
    for values in games:
        model = answers.fit_boosted_trees(masks, values, max_depth=3, n_estimators=25,
                                         min_samples_leaf=1, seed=2)
        prediction = sum((model["learning_rate"] * tree.predict(masks) for tree in model["trees"]),
                         np.full(len(masks), model["initial_value"]))
        assert_spectrum(function(model), direct_fourier(masks, prediction))
        np.testing.assert_allclose(answers.predict_spectrum(masks, function(model)), prediction, atol=1e-12)
    print("ensemble_to_fourier passed: exact additive, parity, polynomial and constant boosted-tree spectra.")


def test_greedy_proxy_failure_is_separate_from_extraction():
    masks = cube(10)
    values = -1.25 * direct_character(masks, (1, 3, 5))
    # Correct depth is available, but these few greedy trees waste zero-gain
    # early splits. This intentionally bounded analytic control is not native tuning.
    model = answers.fit_boosted_trees(masks, values, max_depth=3, n_estimators=3,
                                     min_samples_leaf=1, seed=1)
    proxy = answers.predict_boosted_trees(model, masks)
    spectrum = answers.ensemble_to_fourier(model)
    extracted = answers.predict_spectrum(masks, spectrum)
    assert np.sqrt(np.mean((values - proxy) ** 2)) > 1.0
    np.testing.assert_allclose(extracted, proxy, atol=1e-13)


def test_shuffled_responses_and_wrong_depth_fail_visibly():
    masks = cube(6)
    values = 0.3 + masks[:, 0] + 2.0 * masks[:, 0] * masks[:, 1] * masks[:, 2]
    ordinary = answers.fit_boosted_trees(masks, values, max_depth=3, n_estimators=32,
                                        min_samples_leaf=1, seed=5)
    shallow = answers.fit_boosted_trees(masks, values, max_depth=1, n_estimators=32,
                                       min_samples_leaf=1, seed=5)
    shuffled = answers.fit_boosted_trees(masks, np.random.default_rng(84).permutation(values),
                                        max_depth=3, n_estimators=32, min_samples_leaf=1, seed=5)
    errors = [np.mean((answers.predict_boosted_trees(model, masks) - values) ** 2)
              for model in (ordinary, shallow, shuffled)]
    assert errors[0] < errors[1] < errors[2]
    assert errors[2] > errors[0] * 100
    for model in (ordinary, shallow, shuffled):
        np.testing.assert_allclose(answers.predict_spectrum(masks, answers.ensemble_to_fourier(model)),
                                   answers.predict_boosted_trees(model, masks), atol=1e-12)


def test_select_energy_support(function=answers.select_energy_support):
    cases = [({}, 0.95, {(): 0.0}), ({(): 0.0}, 0.95, {(): 0.0}),
             ({(): 3.0}, 0.95, {(): 3.0}), ({(0, 1): -2.0}, 0.95, {(): 0.0, (0, 1): -2.0}),
             ({(): 0.1, (0,): 3.0, (1,): -1.0}, 0.9, {(): 0.1, (0,): 3.0}),
             ({(): 0.1, (0,): 3.0, (1,): -1.0}, 0.95, {(): 0.1, (0,): 3.0, (1,): -1.0}),
             ({(1,): -2.0, (0,): 2.0, (2,): 0.0}, 0.5, {(): 0.0, (0,): 2.0})]
    for spectrum, fraction, expected in cases:
        result = function(spectrum, fraction)
        assert result["spectrum"] == expected
        assert result["retained_energy_fraction"] >= fraction
    for scale in (1e-200, 1.0, 1e200):
        spectrum = {(): scale / 100, (0,): 3 * scale, (1,): -scale}
        assert function(spectrum, 0.9)["supports"] == ((), (0,))
        assert function(spectrum, 0)["supports"] == ((),)
        assert function(spectrum, 1)["supports"] == ((), (0,), (1,))
    rng = np.random.default_rng(17)
    for _ in range(100):
        coefficients = rng.normal(size=8)
        spectrum = {(i,): value for i, value in enumerate(coefficients)}
        fraction = rng.uniform()
        result = function(spectrum, fraction)
        ranked = sorted(coefficients ** 2, reverse=True)
        count = result["selected_nonconstant_count"]
        assert sum(ranked[:count]) >= fraction * sum(ranked) - 1e-14
        assert count == 0 or sum(ranked[:count - 1]) < fraction * sum(ranked)
    for fraction in (-0.1, 1.01, np.nan):
        with pytest.raises(ValueError):
            function({(): 1.0}, fraction)
    print("select_energy_support passed: shortest >= prefix, intercept, ties, empty/one-term spectra and scaling.")


def test_author_strict_energy_failure(function=answers.author_strict_energy_selection):
    assert function({(0, 1, 2): -1.25}) == {}
    # The crossing term is dropped, and an intercept below the threshold vanishes.
    spectrum = {(): 0.1, (0,): 3.0, (1,): -1.0}
    assert function(spectrum) == {(0,): 3.0}
    assert answers.select_energy_support(spectrum)["retained_energy_fraction"] == 1.0
    # Boundary ties are all discarded by the author rule.
    assert function({(0,): 2.0, (1,): -2.0}) == {}
    with pytest.raises(ValueError, match="argmin"):
        function({})
    print("author selection diagnostic passed: preserved empty parity, omitted intercept and strict-boundary failures.")


def test_grouped_mask_folds(function=answers.grouped_mask_folds):
    masks = np.repeat(cube(4), np.arange(1, 17), axis=0)
    first = function(masks, fold_seed=17, folds=3)
    second = function(masks, fold_seed=17, folds=3)
    seen = []
    for a, b in zip(first, second, strict=True):
        train, validation = a["train_indices"], a["validation_indices"]
        np.testing.assert_array_equal(train, b["train_indices"])
        assert not ({tuple(row) for row in masks[train]} & {tuple(row) for row in masks[validation]})
        np.testing.assert_array_equal(np.sort(np.concatenate((train, validation))), np.arange(len(masks)))
        seen.extend(validation.tolist())
    assert sorted(seen) == list(range(len(masks)))
    assert not np.array_equal(first[0]["validation_indices"], function(masks, fold_seed=18)[0]["validation_indices"])
    with pytest.raises(ValueError):
        function(np.zeros((10, 2)), fold_seed=0)
    print("grouped_mask_folds passed: duplicate-safe, seeded, complete validation coverage.")


def test_ridge_path(function=answers.ridge_path):
    rng = np.random.default_rng(42)
    features = rng.normal(size=(14, 5))
    features[:, 0] = 1.0
    features[:, 4] = features[:, 3]  # deliberately rank deficient
    values = rng.normal(size=14)
    alphas = (1e-3, 0.1, 10.0)
    actual = function(features, values, alphas)
    expected = np.array([np.linalg.solve(features.T @ features + alpha * np.eye(5), features.T @ values)
                         for alpha in alphas])
    np.testing.assert_allclose(actual, expected, atol=1e-11)
    assert function(np.ones((3, 1)), np.ones(3) * 2, [1.0])[0, 0] == pytest.approx(1.5)
    for alphas in ([], [0], [-1], [np.nan]):
        with pytest.raises(ValueError):
            function(features, values, alphas)
    print("ridge_path passed: independent normal equations, rank deficiency and regularized intercept.")


def test_refine_coefficients(function=answers.refine_coefficients):
    masks = cube(4)
    values = 0.7 - 1.2 * direct_character(masks, (0, 2)) + 0.3 * direct_character(masks, (1,))
    folds = answers.grouped_mask_folds(masks, fold_seed=13)
    full = {(): 0.3, (0, 2): -1.0, (1,): 0.1}
    fold_spectra = [{(): 0.5, (0, 2): 0.4}, {(): 0.5, (1,): -1.0}, {(): 0.5}]
    alphas = (0.01, 1.0, 10.0)
    result = function(masks, values, full, fold_records=folds, fold_spectra=fold_spectra,
                      energy_fraction=1, alphas=alphas)
    sum_errors = np.zeros(len(alphas))
    for record, spectrum in zip(folds, fold_spectra, strict=True):
        train, validation = record["train_indices"], record["validation_indices"]
        design = np.column_stack([direct_character(masks, support) for support in spectrum])
        for index, alpha in enumerate(alphas):
            coefficients = np.linalg.solve(design[train].T @ design[train] + alpha * np.eye(len(spectrum)),
                                           design[train].T @ values[train])
            sum_errors[index] += np.sum((design[validation] @ coefficients - values[validation]) ** 2)
    np.testing.assert_allclose(result["cv"]["validation_mse"], sum_errors / len(masks), atol=1e-13)
    assert result["cv"]["selected_alpha"] == alphas[int(np.argmin(sum_errors))]
    assert result["cv"]["fold_supports"] == [tuple(spectrum) for spectrum in fold_spectra]
    print("refine_coefficients passed: fold-specific supports and training-only normal-equation CV agree.")


def test_fit_proxy_no_audit_or_fold_support_leakage(monkeypatch):
    masks = cube(5)
    values = np.arange(len(masks)) / 100 + 0.4 * direct_character(masks, (1, 3))
    candidates = ({"max_depth": 1, "n_estimators": 3, "min_samples_leaf": 1},
                  {"max_depth": 2, "n_estimators": 5, "min_samples_leaf": 1})
    original_fit = answers.fit_boosted_trees
    fit_inputs, fit_models = [], []

    def observed_fit(rows, responses, **parameters):
        fit_inputs.append((rows.copy(), responses.copy()))
        model = original_fit(rows, responses, **parameters)
        fit_models.append(model)
        return model

    monkeypatch.setattr(answers, "fit_boosted_trees", observed_fit)
    result = answers.fit_proxy(masks, values, fold_seed=21, candidates=candidates,
                               folds=3, ridge_alphas=(0.01, 1.0))
    assert len(fit_inputs) == 7
    for candidate in range(2):
        for fold, record in enumerate(result["folds"]):
            rows, responses = fit_inputs[candidate * 3 + fold]
            np.testing.assert_array_equal(rows, masks[record["train_indices"]])
            np.testing.assert_array_equal(responses, values[record["train_indices"]])
    selected = result["proxy_cv"]["selected_index"]
    for fold in range(3):
        spectrum = answers.ensemble_to_fourier(fit_models[selected * 3 + fold])
        expected_supports = answers.select_energy_support(spectrum)["supports"]
        assert result["ridge_cv"]["fold_supports"][fold] == expected_supports
    assert result["accounting"]["residual_tree_fits"] == 3 * (3 + 5) + candidates[selected]["n_estimators"]
    assert result["accounting"]["ridge_candidate_fits"] == 7
    assert result["accounting"]["oracle_calls_by_methods"] == 0
    assert result["accounting"]["unique_input_masks"] == result["accounting"]["input_rows"] == 32
    assert "audit" not in " ".join(inspect.signature(answers.fit_proxy).parameters)
    with pytest.raises(TypeError, match="audit_values"):
        answers.fit_proxy(masks, values, fold_seed=21, audit_values=np.full(100, -999.0))
    np.testing.assert_allclose(answers.predict_spectrum(masks, result["full_spectrum"]),
                               answers.predict_boosted_trees(result["model"], masks), atol=1e-12)


def test_stage_errors(function=answers.stage_errors):
    truth = np.array([0, 1, -1], float)
    proxy, full, truncated, refined = truth + 2, truth + 2, truth + 3, truth + 5
    result = function(truth, proxy, full, truncated, refined)
    assert result == {"audit_rows": 3, "model_to_proxy_rmse": 2.0, "proxy_to_extraction_rmse": 0.0,
                      "extraction_to_truncation_rmse": 1.0, "truncation_to_refinement_rmse": 2.0,
                      "model_to_extraction_rmse": 2.0, "model_to_truncation_rmse": 3.0,
                      "model_to_refinement_rmse": 5.0}
    with pytest.raises(ValueError):
        function(truth, proxy, full[:-1], truncated, refined)
    print("stage_errors passed: extraction fidelity and worsening refinement remain separate.")


def test_prospective_defaults_are_explicit():
    assert [(p["max_depth"], p["n_estimators"], p["learning_rate"], p["min_samples_leaf"])
            for p in answers.PROXY_CANDIDATES] == [(1, 128, 0.1, 2), (1, 512, 0.1, 2),
                                                  (3, 128, 0.1, 2), (3, 512, 0.1, 2)]
    assert answers.RIDGE_ALPHAS == (1e-6, 1e-3, 0.1, 1.0, 10.0)


def test_refinement_rejects_overlapping_or_incomplete_folds():
    masks, values = cube(3), np.arange(8, dtype=float)
    valid = answers.grouped_mask_folds(masks, fold_seed=0)
    broken = [{"train_indices": np.arange(7), "validation_indices": np.array([0])}, *valid[1:]]
    with pytest.raises(ValueError, match="partition"):
        answers.refine_coefficients(masks, values, {(): 1}, fold_records=broken,
                                    fold_spectra=[{(): 1}] * 3)
    with pytest.raises(ValueError, match="cover"):
        answers.refine_coefficients(masks, values, {(): 1}, fold_records=valid[:2],
                                    fold_spectra=[{(): 1}] * 2)
    duplicate_masks = np.repeat(cube(2), 2, axis=0)
    crossing = [{"train_indices": np.arange(0, 8, 2), "validation_indices": np.arange(1, 8, 2)}]
    with pytest.raises(ValueError, match="duplicate"):
        answers.refine_coefficients(duplicate_masks, values, {(): 1}, fold_records=crossing,
                                    fold_spectra=[{(): 1}])
