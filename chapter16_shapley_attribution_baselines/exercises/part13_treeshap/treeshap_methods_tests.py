"""Immediate algorithm checks with hand-computable and independent subset oracles."""

import math
import itertools
import numpy as np
import pytest

from . import treeshap_methods as methods


def repeated_tree():
    return {
        "left": np.array([1, -1, 3, -1, -1]),
        "right": np.array([2, -1, 4, -1, -1]),
        "feature": np.array([0, -1, 0, -1, -1]),
        "threshold": np.array([1.0, 0, 2.0, 0, 0]),
        "value": np.array([20.0, 10.0, 80 / 3, 20.0, 30.0]),
        "cover": np.array([100.0, 40.0, 60.0, 20.0, 40.0]),
    }


def binary_tree(order=(0, 1), joint=None):
    """Two equivalent trees for x0 + 2*x1, with supplied population counts."""
    joint = np.array([[9.0, 1.0], [1.0, 9.0]]) if joint is None else np.asarray(joint)
    fields = {key: [] for key in ("left", "right", "feature", "threshold", "value", "cover")}
    examples = np.array(list(itertools.product([0, 1], repeat=2)))
    counts = joint.ravel()

    def add(depth, keep):
        node = len(fields["left"])
        for key in fields:
            fields[key].append(-1)
        fields["cover"][node] = counts[keep].sum()
        fields["value"][node] = np.average(examples[keep] @ [1.0, 2.0], weights=counts[keep])
        fields["threshold"][node] = 0.5
        if depth < 2:
            feature = order[depth]
            fields["feature"][node] = feature
            fields["left"][node] = add(depth + 1, keep[examples[keep, feature] == 0])
            fields["right"][node] = add(depth + 1, keep[examples[keep, feature] == 1])
        return node

    add(0, np.arange(4))
    return {
        key: np.array(value, dtype=int if key in ("left", "right", "feature") else float)
        for key, value in fields.items()
    }


def enumerated_weights(zeros, ones):
    """Independent definition: enumerate subsets of a unique-feature leaf path."""
    n = len(zeros)
    answer = np.zeros(n + 1)
    for flags in itertools.product([False, True], repeat=n):
        size = sum(flags)
        product = math.prod(ones[j] if flag else zeros[j] for j, flag in enumerate(flags))
        answer[size] += (
            product * math.factorial(size) * math.factorial(n - size) / math.factorial(n + 1)
        )
    return answer


def test_tree_predict(fn=methods.tree_predict):
    tree = repeated_tree()
    x = np.array([[0.0], [1.0], [1.5], [2.0], [3.0]])
    assert np.array_equal(fn(tree, x), [10, 10, 20, 20, 30]), (
        "tree_predict must follow every repeated-feature branch and return its exact leaf value."
    )
    # Sklearn casts inputs to float32 before applying its float64 threshold.
    tree["threshold"][0] = 1.00000001
    np.testing.assert_array_equal(fn(tree, np.array([[1.00000002]])), [10])
    tree["threshold"][0] = 10.099999904632568
    tree["threshold"][2] = 20.0
    np.testing.assert_array_equal(fn(tree, np.array([[10.1]])), [20])
    print("tree_predict: repeated thresholds and float32 boundary passed")


def test_tree_path_game(fn=methods.tree_path_game):
    masks = methods.coalition_masks(2)
    actual = fn(binary_tree(), np.array([[1, 1]]), masks)
    np.testing.assert_allclose(actual, [[1.5, 2.8, 2.5, 3.0]])
    reverse = fn(binary_tree((1, 0)), np.array([[1, 1]]), masks)
    np.testing.assert_allclose(reverse, [[1.5, 2.0, 2.9, 3.0]])
    np.testing.assert_allclose(
        fn(repeated_tree(), np.array([[3, 999]]), masks), [[20, 30, 20, 30]]
    )
    boundary = repeated_tree()
    boundary["threshold"][0] = 10.099999904632568
    boundary["threshold"][2] = 20.0
    np.testing.assert_allclose(fn(boundary, np.array([[10.1, 0]]), masks), [[20, 20, 20, 20]])
    print("tree_path_game: cover weighting and repeated-feature membership passed")


def test_exact_shapley(fn=methods.exact_shapley):
    games = np.array([[1.5, 2.8, 2.5, 3], [1.5, 2, 2.9, 3], [0, 0, 0, 1]])
    np.testing.assert_allclose(fn(games), [[0.9, 0.6], [0.3, 1.2], [0.5, 0.5]])
    rng = np.random.default_rng(131)
    values = rng.normal(size=16)
    by_order = []
    for order in itertools.permutations(range(4)):
        current, contribution = 0, np.zeros(4)
        for j in order:
            added = current | (1 << j)
            contribution[j] = values[added] - values[current]
            current = added
        by_order.append(contribution)
    np.testing.assert_allclose(fn(values), np.mean(by_order, axis=0), atol=1e-12)
    print("exact_shapley: signed, interaction and all-ordering checks passed")


def test_extend_path(fn=methods.extend_path):
    zeros = np.array([[0.2, 0.7, 0.5], [0.0, 0.4, 1.0], [0.6, 0.2, 0.3]])
    ones = np.array([[1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]])
    weights = np.ones((3, 1))
    for k in range(3):
        previous = weights
        snapshot = weights.copy()
        weights = fn(weights, zeros[:, k], ones[:, k])
        np.testing.assert_allclose(
            weights, [enumerated_weights(z[: k + 1], o[: k + 1]) for z, o in zip(zeros, ones)]
        )
        np.testing.assert_array_equal(
            previous,
            snapshot,
            err_msg="extend_path must not mutate its input cardinality weights.",
        )
        assert not np.shares_memory(weights, previous), (
            "extend_path must return fresh cardinality weights without overwriting prior state."
        )
    print("extend_path: all subsets and cardinality weights passed")


def test_unwind_path(fn=methods.unwind_path):
    zeros = np.array([[0.2, 0.7, 0.5], [0.0, 0.4, 1.0], [0.6, 0.2, 0.3]])
    ones = np.array([[1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]])
    weights = np.array([enumerated_weights(z, o) for z, o in zip(zeros, ones)])
    original = weights.copy()
    for k in range(3):
        result = fn(weights, zeros[:, k], ones[:, k])
        expected = [
            enumerated_weights(np.delete(z, k), np.delete(o, k)) for z, o in zip(zeros, ones)
        ]
        np.testing.assert_allclose(result, expected, atol=1e-14)
        np.testing.assert_array_equal(weights, original)
    with pytest.raises(ValueError, match="dead path"):
        fn(np.zeros((1, 2)), np.zeros(1), np.zeros(1))
    print("unwind_path: arbitrary removal, zero fractions and no mutation passed")


def test_tree_shap(fn=methods.tree_shap):
    rng = np.random.default_rng(132)
    x = rng.uniform(-1, 4, (17, 3))
    masks = methods.coalition_masks(3)
    for tree in (repeated_tree(), binary_tree(), binary_tree((1, 0))):
        expected = methods.exact_shapley(methods.tree_path_game(tree, x, masks))
        np.testing.assert_allclose(fn(tree, x), expected, atol=1e-12)
        donor = rng.uniform(-1, 4, x.shape)
        paired_games = np.array(
            [
                methods.tree_predict(tree, np.where(masks, row, background))
                for row, background in zip(x, donor)
            ]
        )
        np.testing.assert_allclose(
            fn(tree, x, donor), methods.exact_shapley(paired_games), atol=1e-12
        )
        np.testing.assert_array_equal(fn(tree, x)[:, 2], 0)
    constant = {key: value[:1].copy() for key, value in repeated_tree().items()}
    constant["left"][0] = constant["right"][0] = -1
    np.testing.assert_array_equal(fn(constant, x), np.zeros_like(x))
    # A float32 query must compare against the original float64 threshold.
    # NumPy 1.26 otherwise downcasts an array-versus-scalar threshold comparison.
    boundary = repeated_tree()
    boundary["threshold"][0] = 10.099999904632568
    boundary["threshold"][2] = 20.0
    query = np.array([[10.1, 0.0]])
    np.testing.assert_allclose(fn(boundary, query), [[0.0, 0.0]], atol=1e-12)
    np.testing.assert_allclose(
        methods.tree_path_game(boundary, query, methods.coalition_masks(2)),
        [[20.0, 20.0, 20.0, 20.0]],
    )
    print("tree_shap: repeated features, dead donor paths, constants and dummies passed")


def test_ensemble_shap(fn=methods.ensemble_shap):
    ensemble = {
        "trees": [binary_tree(), repeated_tree()],
        "weights": np.array([0.4, -0.2]),
        "offset": 7.0,
    }
    x = np.array([[1.0, 1.0], [3.0, 0.0]])
    background = np.array([[0.0, 0.0], [1.0, 1.0], [3.0, 0.0]])
    predict = lambda rows: 7 + sum(
        w * methods.tree_predict(tree, rows)
        for tree, w in zip(ensemble["trees"], ensemble["weights"])
    )
    for bg in (None, background):
        base, phi = fn(ensemble, x, bg)
        if bg is None:
            game = 7 + sum(
                w * methods.tree_path_game(tree, x, methods.coalition_masks(2))
                for tree, w in zip(ensemble["trees"], ensemble["weights"])
            )
        else:
            game = methods.interventional_game(predict, x, bg, methods.coalition_masks(2))
        np.testing.assert_allclose(phi, methods.exact_shapley(game), atol=1e-12)
        np.testing.assert_allclose(base, game[:, 0])
        np.testing.assert_allclose(base + phi.sum(1), predict(x))
    print("ensemble_shap: signed tree weights, offsets and donor averaging passed")


def test_interventional_game(fn=methods.interventional_game):
    predict = lambda x: x[:, 0] + x[:, 0] * x[:, 1]
    x = np.array([[2.0, 3.0], [4.0, 5.0]])
    background = np.array([[0.0, 0.0], [1.0, 2.0], [3.0, 1.0]])
    masks = methods.coalition_masks(2)
    result = fn(predict, x, background, masks)
    expected = np.array(
        [
            [
                np.mean([predict(np.where(mask, row, donor)[None])[0] for donor in background])
                for mask in masks
            ]
            for row in x
        ]
    )
    np.testing.assert_allclose(result, expected)
    assert result[0, 0] != predict(background.mean(0)[None])[0], (
        "Interventional values must average predictions over intact donors; for this nonlinear "
        "model, prediction at the mean donor is a different game."
    )
    print("interventional_game: intact donors, nonlinear averaging and endpoints passed")


def test_saabas(fn=methods.saabas):
    x = np.array([[1.0, 1.0]])
    base, phi = fn(binary_tree(), x)
    np.testing.assert_allclose(base, [1.5])
    np.testing.assert_allclose(phi, [[1.3, 0.2]])
    np.testing.assert_allclose(base + phi.sum(1), [3])
    assert not np.allclose(phi, [[0.9, 0.6]]), (
        "Saabas must retain its order-dependent allocation instead of collapsing to the exact "
        "Shapley allocation on this correlated tree."
    )
    base, phi = fn(repeated_tree(), np.array([[3.0, 0.0]]))
    np.testing.assert_allclose(phi, [[10.0, 0.0]])
    print("saabas: efficient allocation that differs from Shapley passed")
