"""Independent numerical oracles, usable immediately after each exercise."""

from itertools import combinations

import numpy as np
import pytest
import torch
from sklearn.linear_model import Lasso

from . import datamodel_methods as reference

DTYPE = torch.float64


def test_sample_subsets(fn=reference.sample_subsets):
    masks = fn(6, 3, 4000, 91)
    assert (
        masks.shape == (4000, 6) and masks.dtype == torch.bool
    ), "Return one boolean row per sampled model."
    assert (
        masks.sum(1) == 3
    ).all(), "Every row must include exactly three items; Bernoulli sampling changes the task."
    assert torch.equal(
        masks, fn(6, 3, 4000, 91)
    ), "A repeated seed must reproduce the identical masks."
    assert not torch.equal(masks, fn(6, 3, 4000, 92)), "The seed should affect the sampled subsets."
    possible = {tuple(x) for x in combinations(range(6), 3)}
    observed = {tuple(torch.nonzero(row).flatten().tolist()) for row in masks}
    assert (
        observed == possible
    ), "The sampler must reach every possible subset, not only cyclic windows."
    # All twenty possible subsets should occur near 1/20, not only cyclic windows.
    counts = [
        sum(tuple(torch.nonzero(row).flatten().tolist()) == item for row in masks)
        for item in sorted(possible)
    ]
    assert (
        min(counts) > 140 and max(counts) < 270
    ), "Subset frequencies are inconsistent with uniform independent permutations."
    print("All tests in test_sample_subsets passed!")


def test_correct_class_margins(fn=reference.correct_class_margins):
    logits = torch.tensor(
        [[[2.0, 9.0, 5.0], [1.0, 0.0, -2.0]], [[3.0, 3.0, 1.0], [-4.0, -8.0, -3.0]]], dtype=DTYPE
    )
    labels = torch.tensor([1, 0])
    before = logits.clone()
    torch.testing.assert_close(
        fn(logits, labels), torch.tensor([[4.0, 1.0], [0.0, -1.0]], dtype=DTYPE)
    )
    torch.testing.assert_close(fn(logits + 1000, labels), fn(logits, labels))
    torch.testing.assert_close(logits, before)
    torch.testing.assert_close(fn(logits[0], labels), torch.tensor([4.0, 1.0], dtype=DTYPE))
    print("All tests in test_correct_class_margins passed!")


def test_soft_threshold(fn=reference.soft_threshold):
    x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=DTYPE)
    expected = torch.tensor([-2.0, 0.0, 0.0, 0.0, 2.0], dtype=DTYPE)
    torch.testing.assert_close(fn(x, 1.0), expected)
    torch.testing.assert_close(fn(x, 0.0), x)
    # Independently minimize each scalar proximal objective on a fine grid.
    grid = torch.linspace(-5, 5, 20001, dtype=DTYPE)
    objectives = (grid[:, None] - x).square() / 2 + grid[:, None].abs()
    grid_solution = grid[objectives.argmin(0)]
    torch.testing.assert_close(fn(x, 1.0), grid_solution, atol=5e-4, rtol=0)
    with pytest.raises(ValueError):
        fn(x, -1.0)
    print("All tests in test_soft_threshold passed!")


def test_lasso_objective(fn=reference.lasso_objective):
    x = torch.tensor([[0.0, 1.0], [1.0, 1.0]], dtype=DTYPE)
    y = torch.tensor([[1.0, 3.0], [2.0, 4.0]], dtype=DTYPE)
    w = torch.tensor([[1.0, -2.0], [0.0, 1.0]], dtype=DTYPE)
    bias = torch.tensor([1.0, 2.0], dtype=DTYPE)
    penalty = torch.tensor([0.2, 0.3], dtype=DTYPE)
    torch.testing.assert_close(fn(x, y, w, bias, penalty), torch.tensor([0.2, 3.15], dtype=DTYPE))
    # Large intercepts must not incur an L1 penalty.
    torch.testing.assert_close(fn(x, y + 80, w, bias + 80, penalty), fn(x, y, w, bias, penalty))
    print("All tests in test_lasso_objective passed!")


def test_lasso_kkt(fn=reference.lasso_kkt):
    gram = torch.eye(3, dtype=DTYPE)
    cross = torch.tensor([[3.0, 1.0], [-0.2, -4.0], [-2.0, 0.5]], dtype=DTYPE)
    penalty = torch.tensor([1.0, 0.5], dtype=DTYPE)
    exact = torch.tensor([[2.0, 0.5], [0.0, -3.5], [-1.0, 0.0]], dtype=DTYPE)
    torch.testing.assert_close(fn(gram, cross, exact, penalty), torch.zeros(2, dtype=DTYPE))
    wrong = exact.clone()
    wrong[1, 0] = 0.1
    wrong[0, 1] = 0
    torch.testing.assert_close(
        fn(gram, cross, wrong, penalty), torch.tensor([1.3, 0.5], dtype=DTYPE)
    )
    print("All tests in test_lasso_kkt passed!")


def test_fit_lasso(fn=reference.fit_lasso):
    rng = np.random.default_rng(816)
    x = torch.from_numpy(rng.normal(size=(240, 9)))
    true = torch.zeros(9, 3, dtype=DTYPE)
    true[[0, 2, 6], :] = torch.tensor(
        [[1.0, 0.0, -0.5], [0.0, 2.0, 0.2], [-0.4, 0.0, 0.0]], dtype=DTYPE
    )
    y = x @ true + torch.tensor([2.0, -1.0, 4.0]) + torch.from_numpy(rng.normal(0, 0.05, (240, 3)))
    penalty = torch.tensor([0.02, 0.08, 0.05], dtype=DTYPE)
    fit = fn(x, y, penalty)
    assert (
        fit["kkt"] <= 1e-8
    ).all(), "Check stationarity for every target before returning an optimum."
    for q in range(3):
        independent = Lasso(alpha=penalty[q].item(), fit_intercept=True, tol=1e-13, max_iter=100000)
        independent.fit(x.numpy(), y[:, q].numpy())
        np.testing.assert_allclose(
            fit["weights"][:, q].numpy(), independent.coef_, atol=2e-7, rtol=2e-7
        )
        np.testing.assert_allclose(fit["bias"][q].item(), independent.intercept_, atol=2e-7, rtol=0)
    # Rank-deficient masks have a coefficient-offset ambiguity. Compare the
    # predicted function, objective and stationarity, not a chosen representative.
    fixed = torch.zeros(70, 8, dtype=DTYPE)
    for row, chosen in enumerate(combinations(range(8), 4)):
        fixed[row, list(chosen)] = 1
    response = (
        fixed @ torch.arange(8, dtype=DTYPE)[:, None] / 10 + 0.03 * fixed[:, :1] * fixed[:, 1:2]
    )
    p = torch.tensor([0.003], dtype=DTYPE)
    singular = fn(fixed, response, p)
    independent = Lasso(alpha=0.003, tol=1e-13, max_iter=100000).fit(
        fixed.numpy(), response[:, 0].numpy()
    )
    prediction = fixed @ singular["weights"] + singular["bias"]
    np.testing.assert_allclose(
        prediction[:, 0].numpy(), independent.predict(fixed.numpy()), atol=3e-7, rtol=0
    )
    actual_objective = (
        (prediction - response).square().mean() / 2 + p[0] * singular["weights"].abs().sum()
    ).item()
    expected_objective = (
        np.mean((independent.predict(fixed.numpy()) - response[:, 0].numpy()) ** 2) / 2
        + 0.003 * np.abs(independent.coef_).sum()
    )
    assert (
        abs(actual_objective - expected_objective) < 1e-9
    ), "The rank-deficient fit must match the independently minimized objective."
    print("All tests in test_fit_lasso passed!")


def test_lasso_failure_and_constant_design():
    x = torch.tensor([[1.0, 2.0], [2.0, 3.0], [4.0, 0.0], [5.0, 3.0]], dtype=DTYPE)
    y = torch.tensor([[1.0], [4.0], [0.0], [3.0]], dtype=DTYPE)
    with pytest.raises(RuntimeError, match="did not converge"):
        reference.fit_lasso(x, y, torch.tensor([0.001], dtype=DTYPE), max_steps=1)
    fit = reference.fit_lasso(torch.ones_like(x), y, torch.tensor([0.1], dtype=DTYPE))
    torch.testing.assert_close(fit["weights"], torch.zeros(2, 1, dtype=DTYPE))
    torch.testing.assert_close(fit["bias"], y.mean(0))
    print("All tests in test_lasso_failure_and_constant_design passed!")


def test_fit_datamodel_path(fn=reference.fit_datamodel_path):
    rng = torch.Generator().manual_seed(513)
    x = torch.randn(100, 7, generator=rng, dtype=DTYPE)
    y = torch.stack([x[:, 1] - x[:, 4], torch.full((100,), 3.0, dtype=DTYPE)], dim=1)
    xv = torch.randn(25, 7, generator=rng, dtype=DTYPE) + 2
    yv = torch.stack([xv[:, 1] - xv[:, 4], torch.full((25,), 3.0, dtype=DTYPE)], dim=1)
    fractions = torch.tensor([1.0, 0.1, 0.01], dtype=DTYPE)
    result = fn(x, y, xv, yv, fractions)
    assert result["selected"].tolist() == [
        2,
        0,
    ], "Choose the smallest validation error per target, with the first path index on a tie."
    torch.testing.assert_close(
        result["weights"][0], torch.zeros_like(result["weights"][0]), atol=1e-14, rtol=0
    )
    for q in range(2):
        k = result["selected"][q]
        torch.testing.assert_close(result["selected_weights"][:, q], result["weights"][k, :, q])
        torch.testing.assert_close(result["selected_bias"][q], result["bias"][k, q])
    # Altering validation outputs may change selection; it must never change
    # fitted coefficients, training centering or the available path.
    changed = fn(x, y, xv, yv + 20, fractions)
    torch.testing.assert_close(changed["weights"], result["weights"], atol=0, rtol=0)
    torch.testing.assert_close(changed["bias"], result["bias"], atol=0, rtol=0)
    print("All tests in test_fit_datamodel_path passed!")


def test_select_supportive(fn=reference.select_supportive):
    scores = torch.tensor([-1.0, 2.0, 2.0, 0.0, 4.0], dtype=DTYPE)
    assert fn(scores, 2).tolist() == [
        1,
        4,
    ], "Select largest scores and keep the earlier source index when scores tie."
    assert fn(scores, 0).numel() == 0, "A zero deletion budget must select no images."
    assert torch.equal(
        fn(scores + 81, 2), fn(scores, 2)
    ), "A common coefficient offset changes magnitudes, not ranking."
    print("All tests in test_select_supportive passed!")


def test_matched_random_removals(fn=reference.matched_random_removals):
    labels = torch.tensor([0, 0, 1, 1, 1, 2, 2, 2])
    selected = torch.tensor([0, 3, 4, 6])
    controls = fn(labels, selected, 64, 617)
    assert controls.shape == (
        64,
        4,
    ), "Return a separate four-image deletion for each of the 64 controls."
    for row in controls:
        assert row.unique().numel() == 4, "Sample without replacement inside each deletion set."
        assert torch.bincount(labels[row], minlength=3).tolist() == [
            1,
            2,
            1,
        ], "Match removed class counts, not only the total budget."
    assert torch.equal(
        controls, fn(labels, selected, 64, 617)
    ), "The control seed must reproduce the same random deletions."
    assert (
        controls.unique(dim=0).shape[0] > 10
    ), "Draw each replicate independently rather than repeating one random set."
    print("All tests in test_matched_random_removals passed!")


def test_centered_weight_similarity(fn=reference.centered_weight_similarity):
    weights = torch.tensor(
        [[1.0, 4.0, -1.0, 0.0], [2.0, 8.0, -2.0, 0.0], [3.0, 12.0, -3.0, 0.0]], dtype=DTYPE
    )
    result = fn(weights)
    expected = torch.tensor(
        [
            [1.0, 1.0, -1.0, 0.0],
            [1.0, 1.0, -1.0, 0.0],
            [-1.0, -1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=DTYPE,
    )
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(fn(weights + torch.tensor([10.0, -3.0, 5.0, 41.0])), expected)
    print("All tests in test_centered_weight_similarity passed!")


def test_fixed_cardinality_gauge():
    masks = reference.sample_subsets(10, 4, 32, 451).double()
    weights = torch.arange(10, dtype=DTYPE) / 8
    bias, offset = 0.3, 2.5
    torch.testing.assert_close(
        masks @ weights + bias, masks @ (weights + offset) + bias - 4 * offset
    )
    smaller = masks.clone()
    smaller[:, 0] = 0
    actual_shift = smaller @ (weights + offset) + bias - 4 * offset - (smaller @ weights + bias)
    torch.testing.assert_close(actual_shift, (smaller.sum(1) - 4) * offset)
    print("All tests in test_fixed_cardinality_gauge passed!")
