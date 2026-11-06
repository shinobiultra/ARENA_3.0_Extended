"""Small immediate checks, with independently calculated expected answers."""

import itertools
import math
import numpy as np
import torch as t
from chapter16_shapley_attribution_baselines.exercises.part12_sage_global_importance import (
    sage_methods as methods,
)


def test_coalition_masks(fn=methods.coalition_masks):
    expected = t.tensor([[False, False], [True, False], [False, True], [True, True]])
    assert t.equal(
        fn(2), expected
    ), "Feature 0 is the least significant bit; include empty and full."
    assert fn(4).sum(0).tolist() == [8] * 4, "Each feature must appear in exactly half of all coalitions."
    print("All tests in test_coalition_masks passed!")


def test_marginal_predict(fn=methods.marginal_predict):
    x = t.tensor([[3.0, 5.0], [2.0, 7.0]], dtype=t.float64)
    background = t.tensor([[0.0, 1.0], [2.0, 3.0]], dtype=t.float64)
    model = lambda z: z[:, 0].square() + z[:, 1]
    masks = t.tensor([[False, True], [True, False]])
    t.testing.assert_close(
        fn(model, x, background, masks),
        t.tensor([7.0, 6.0], dtype=t.float64),
        msg="Average model predictions, not inputs: E[X²] is not E[X]².",
    )
    both = fn(model, x, background, t.ones_like(masks))
    t.testing.assert_close(both, model(x))
    empty = fn(model, x, background, t.zeros_like(masks))
    t.testing.assert_close(empty, t.full((2,), 4.0, dtype=t.float64))
    # Preserve within-background dependence: both missing coordinates share a donor row.
    product = lambda z: z[:, 0] * z[:, 1]
    t.testing.assert_close(
        fn(product, x, background, t.zeros_like(masks)), t.full((2,), 3.0, dtype=t.float64)
    )
    print("All tests in test_marginal_predict passed!")


def test_loss_game(fn=methods.loss_game):
    p = t.tensor([[2.0, 1.0, 3.0, 0.0], [2.0, 4.0, 1.0, 3.0]], dtype=t.float64)
    y = t.tensor([0.0, 4.0], dtype=t.float64)
    expected = np.array([[0.0, 3.0, -5.0, 4.0], [0.0, 4.0, -5.0, 3.0]])
    np.testing.assert_allclose(
        fn(p, y),
        expected,
        err_msg="Improvement is empty loss minus restricted loss; harmful features can be negative.",
    )
    print("All tests in test_loss_game passed!")


def test_exact_shapley(fn=methods.exact_shapley):
    rng = np.random.default_rng(16012)
    games = rng.normal(size=(3, 16))
    expected = np.zeros((3, 4))
    for order in itertools.permutations(range(4)):
        mask = 0
        for j in order:
            added = mask + (1 << j)
            expected[:, j] += (games[:, added] - games[:, mask]) / math.factorial(4)
            mask = added
    actual = fn(t.from_numpy(games))
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    np.testing.assert_allclose(
        actual.sum(1),
        games[:, -1] - games[:, 0],
        atol=1e-12,
        err_msg="Efficiency compares full with empty, even when empty is nonzero.",
    )
    # XOR has no singleton predictive power, yet both features share the full improvement.
    t.testing.assert_close(
        fn(t.tensor([0.0, 0.0, 0.0, 1.0], dtype=t.float64)), t.tensor([0.5, 0.5], dtype=t.float64)
    )
    print("All tests in test_exact_shapley passed!")


def test_gaussian_game(fn=methods.gaussian_game):
    beta = t.tensor([1.0, 0.0], dtype=t.float64)
    for rho in [0.0, 0.6, 1.0]:
        cov = t.tensor([[1.0, rho], [rho, 1.0]], dtype=t.float64)
        result = fn(cov, beta, beta, True, 0.25)
        t.testing.assert_close(
            result["game"], t.tensor([0.0, 1.0, rho * rho, 1.0], dtype=t.float64)
        )
        t.testing.assert_close(
            result["risk"], t.tensor([1.25, 0.25, 1.25 - rho * rho, 0.25], dtype=t.float64)
        )
        marginal = fn(cov, beta, beta, False)
        t.testing.assert_close(
            marginal["game"],
            t.tensor([0.0, 1.0, 0.0, 1.0], dtype=t.float64),
            msg="Marginal restriction ignores the unused feature even when it is informative.",
        )
    wrong = fn(t.eye(2, dtype=t.float64), beta, t.tensor([1.0, 2.0], dtype=t.float64))
    t.testing.assert_close(wrong["game"], t.tensor([0.0, 1.0, -4.0, -3.0], dtype=t.float64))
    print("All tests in test_gaussian_game passed!")


def test_path_contributions(fn=methods.path_contributions):
    x = t.tensor([[1.0, 2.0], [1.0, 2.0]], dtype=t.float64)
    y = t.tensor([3.0, 3.0], dtype=t.float64)
    restricted = lambda z, mask: (z * mask).sum(1)
    order = t.tensor([[0, 1], [1, 0]])
    t.testing.assert_close(
        fn(restricted, x, y, order),
        t.tensor([[5.0, 4.0], [1.0, 8.0]], dtype=t.float64),
        msg="Return scores by feature index, not by the step when it entered.",
    )
    try:
        fn(restricted, x, y, t.tensor([[0, 0], [1, 0]]))
    except ValueError:
        pass
    else:
        raise AssertionError("Repeated features are not a permutation.")
    print("All tests in test_path_contributions passed!")


def test_permutation_sage(fn=methods.permutation_sage):
    x = t.tensor([[1.0, 2.0], [3.0, -1.0]], dtype=t.float64)
    y = t.tensor([3.0, 2.0], dtype=t.float64)
    model = lambda z: z[:, 0] + z[:, 1]
    background = t.zeros((2, 2), dtype=t.float64)
    rows = t.tensor([0, 0, 1, 1])
    orders = t.tensor([[0, 1], [1, 0], [0, 1], [1, 0]])
    expected = t.tensor([[5.0, 4.0], [1.0, 8.0], [3.0, 1.0], [9.0, -5.0]], dtype=t.float64)
    for batch in [1, 3, 9]:
        t.testing.assert_close(fn(model, x, y, background, rows, orders, batch), expected)
    print("All tests in test_permutation_sage passed!")


def test_summarize_paths(fn=methods.summarize_paths):
    x = t.tensor([[1.0, 3.0], [3.0, -1.0], [5.0, 4.0], [7.0, 2.0]], dtype=t.float64)
    result = fn(x, [2, 4])
    np.testing.assert_allclose(result["mean"], np.stack([x.numpy()[:n].mean(0) for n in [2, 4]]))
    np.testing.assert_allclose(
        result["se"],
        np.stack([x.numpy()[:n].std(0, ddof=1) / np.sqrt(n) for n in [2, 4]]),
        err_msg="Use the SD of independent paths divided by sqrt(n), not the SD itself.",
    )
    print("All tests in test_summarize_paths passed!")


def test_prefix_values(fn=methods.prefix_values):
    game = t.tensor([0.0, 2.0, -3.0, -1.0, 4.0, 6.0, 1.0, 3.0], dtype=t.float64)
    order = t.tensor([[2, 0, 1], [1, 2, 0]])
    t.testing.assert_close(
        fn(game, order), t.tensor([[0.0, 4.0, 6.0, 3.0], [0.0, -3.0, 1.0, 3.0]], dtype=t.float64)
    )
    print("All tests in test_prefix_values passed!")
