"""Independent oracles and counterexamples, also callable with student functions."""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from scipy.optimize import root
from scipy.special import expit

from chapter16_shapley_attribution_baselines.exercises.part9_influence_and_tracin import attribution_methods as answers

DTYPE = torch.float64


def _fixture():
    x = torch.tensor([[1., 0., 1.], [0., 2., 1.], [-1., 1., 1.], [2., -1., 1.], [.3, .8, 1.]], dtype=DTYPE)
    y = torch.tensor([1., 0., 0., 1., 1.], dtype=DTYPE)
    return x, y


def _scipy_fit(x, y, weights, l2):
    """Independent optimizer and NumPy derivatives, with an explicit residual check."""
    a, target = x.numpy(), y.numpy()
    mass = np.asarray(weights, dtype=float)
    mass = mass / mass.sum()
    def gradient(theta):
        return a.T @ (mass * (expit(a @ theta) - target)) + l2 * theta
    def hessian(theta):
        p = expit(a @ theta)
        return a.T @ ((mass * p * (1 - p))[:, None] * a) + l2 * np.eye(a.shape[1])
    fitted = root(gradient, np.zeros(a.shape[1]), jac=hessian, tol=1e-11)
    assert np.max(np.abs(gradient(fitted.x))) < 1e-11, "The independent root oracle did not reach stationarity."
    return torch.from_numpy(fitted.x)


def test_logistic_losses(candidate=None):
    candidate = candidate or answers.logistic_losses
    x, y = _fixture()
    theta = torch.tensor([.7, -.4, .2], dtype=DTYPE)
    expected = F.binary_cross_entropy_with_logits(x @ theta, y, reduction="none") + .1 * theta.square().sum()
    torch.testing.assert_close(candidate(theta, x, y, .2), expected, msg="Return one loss per example and include the regularizer once in each.")
    extreme_x = torch.tensor([[1000.], [-1000.]], dtype=DTYPE)
    result = candidate(torch.ones(1, dtype=DTYPE), extreme_x, torch.tensor([1., 0.], dtype=DTYPE))
    assert torch.isfinite(result).all() and result.max() < 1e-8, "Use a numerically stable log-loss for confident correct predictions."
    print("All tests in test_logistic_losses passed!")


def test_logistic_gradients(candidate=None):
    candidate = candidate or answers.logistic_gradients
    x, y = _fixture()
    theta = torch.tensor([.7, -.4, .2], dtype=DTYPE, requires_grad=True)
    losses = F.binary_cross_entropy_with_logits(x @ theta, y, reduction="none") + .1 * theta.square().sum()
    expected = torch.stack([torch.autograd.grad(loss, theta, retain_graph=True)[0] for loss in losses])
    torch.testing.assert_close(candidate(theta.detach(), x, y, .2), expected, msg="Each row is a parameter gradient, including that row's regularizer.")
    torch.testing.assert_close(candidate(theta.detach(), x, y, .2).mean(0), torch.autograd.grad(losses.mean(), theta)[0])
    print("All tests in test_logistic_gradients passed!")


def test_weighted_objective(candidate=None):
    candidate = candidate or answers.weighted_objective
    x, y = _fixture()
    theta = torch.tensor([[.7, -.4, .2], [-.3, .1, -.2]], dtype=DTYPE)
    weights = torch.tensor([[0., 1., 1., 2., .5], [3., 0., 1., 1., 1.]], dtype=DTYPE)
    expected = []
    for t, w in zip(theta, weights):
        loss = F.binary_cross_entropy_with_logits(x @ t, y, reduction="none")
        expected.append((loss * w).sum() / w.sum() + .1 * t.square().sum())
    torch.testing.assert_close(candidate(theta, x, y, weights, .2), torch.stack(expected))
    torch.testing.assert_close(candidate(theta, x, y, weights * 9, .2), torch.stack(expected), msg="Rescaling all example weights must not change relative regularization.")
    with pytest.raises(ValueError):
        candidate(theta, x, y, torch.zeros_like(weights), .2)
    print("All tests in test_weighted_objective passed!")


def test_objective_derivatives(candidate=None):
    candidate = candidate or answers.objective_derivatives
    x, y = _fixture()
    theta = torch.tensor([[.7, -.4, .2], [-.3, .1, -.2]], dtype=DTYPE)
    weights = torch.tensor([[0., 1., 1., 2., .5], [3., 0., 1., 1., 1.]], dtype=DTYPE)
    gradients, hessians = candidate(theta, x, y, weights, .2)
    for index, (t, w) in enumerate(zip(theta, weights)):
        def objective(v):
            return (F.binary_cross_entropy_with_logits(x @ v, y, reduction="none") * w).sum() / w.sum() + .1 * v.square().sum()
        g = torch.autograd.functional.jacobian(objective, t)
        h = torch.autograd.functional.hessian(objective, t)
        torch.testing.assert_close(gradients[index], g, msg="Do not average one fit's weights with those of another fit.")
        torch.testing.assert_close(hessians[index], h, msg="The regularizer contributes to every diagonal, including the bias coordinate.")
    print("All tests in test_objective_derivatives passed!")


def test_logistic_hvp(candidate=None):
    candidate = candidate or answers.logistic_hvp
    x, y = _fixture()
    theta = torch.tensor([.7, -.4, .2], dtype=DTYPE, requires_grad=True)
    weights = torch.tensor([0., 1., 1., 2., .5], dtype=DTYPE)
    vector = torch.tensor([2., -1., .3], dtype=DTYPE)
    loss = (F.binary_cross_entropy_with_logits(x @ theta, y, reduction="none") * weights).sum() / weights.sum() + .1 * theta.square().sum()
    gradient = torch.autograd.grad(loss, theta, create_graph=True)[0]
    expected = torch.autograd.grad(gradient @ vector, theta)[0]
    torch.testing.assert_close(candidate(theta.detach(), x, y, weights, vector, .2), expected, msg="An HVP is a directional derivative of the objective gradient, not a per-example Hessian average with the wrong denominator.")
    print("All tests in test_logistic_hvp passed!")


def test_conjugate_gradient(candidate=None):
    candidate = candidate or answers.conjugate_gradient
    matrix = torch.tensor([[4., 1., -.5], [1., 3., .2], [-.5, .2, 2.]], dtype=DTYPE)
    rhs = torch.tensor([1., 2., -3.], dtype=DTYPE)
    solution, history = candidate(lambda v: matrix @ v, rhs)
    torch.testing.assert_close(solution, torch.linalg.solve(matrix, rhs), rtol=1e-9, atol=1e-11)
    torch.testing.assert_close(history[-1], torch.linalg.vector_norm(matrix @ solution - rhs), rtol=1e-6, atol=1e-12)
    zero, norms = candidate(lambda v: matrix @ v, torch.zeros_like(rhs))
    assert torch.equal(zero, torch.zeros_like(rhs)) and len(norms) == 1, "A zero right-hand side should finish before dividing by its residual norm."
    with pytest.raises(ValueError, match="curvature|SPD"):
        candidate(lambda v: -v, rhs)
    with pytest.raises(RuntimeError):
        candidate(lambda v: matrix @ v, rhs, max_steps=1)
    print("All tests in test_conjugate_gradient passed!")


def test_upweight_influence(candidate=None):
    candidate = candidate or answers.upweight_influence
    h = torch.tensor([[3., .4], [.4, 1.]], dtype=DTYPE)
    center = torch.tensor([.3, -.8], dtype=DTYPE)
    point_centers = torch.tensor([[2., -1.], [-.5, .2], [.1, .9]], dtype=DTYPE)
    query_centers = torch.tensor([[1., 1.], [-2., .5]], dtype=DTYPE)
    point_hessian = torch.tensor([[1., .2], [.2, .8]], dtype=DTYPE)
    train_gradients = (center - point_centers) @ point_hessian
    query_gradients = center - query_centers
    predicted = candidate(h, train_gradients, query_gradients)
    eps = 1e-5
    finite_differences = []
    for target in point_centers:
        losses = []
        for sign in (-1, 1):
            value = sign * eps
            optimum = torch.linalg.solve(h + value * point_hessian, h @ center + value * point_hessian @ target)
            losses.append(.5 * (optimum - query_centers).square().sum(1))
        finite_differences.append((losses[1] - losses[0]) / (2 * eps))
    torch.testing.assert_close(predicted, torch.stack(finite_differences), rtol=1e-8, atol=1e-9, msg="Upweight influence is a signed loss derivative; positive means the query loss increases.")
    with pytest.raises((ValueError, torch.linalg.LinAlgError)):
        candidate(-torch.eye(2, dtype=DTYPE), train_gradients, query_gradients)
    print("All tests in test_upweight_influence passed!")


def test_tracin_checkpoint_scores(candidate=None):
    candidate = candidate or answers.tracin_checkpoint_scores
    train = torch.tensor([[[1., 2.], [-1., 0.], [0., 1.]], [[2., 0.], [0., -3.], [1., -1.]]], dtype=DTYPE)
    query = torch.tensor([[[3., -1.], [-2., 2.]], [[1., 1.], [2., -1.]]], dtype=DTYPE)
    rates = torch.tensor([.2, .5], dtype=DTYPE)
    expected = torch.tensor([[1.2, 2.4], [-2.1, 1.9], [-.2, 1.9]], dtype=DTYPE)
    torch.testing.assert_close(candidate(train, query, rates), expected, msg="Apply the learning rate inside the sum and keep distinct train/query axes.")
    torch.testing.assert_close(candidate(train.flip(0), query.flip(0), rates.flip(0)), expected)
    torch.testing.assert_close(candidate(query, train, rates).T, expected)
    with pytest.raises(ValueError):
        candidate(train, query, torch.tensor([-.1, .2], dtype=DTYPE))
    print("All tests in test_tracin_checkpoint_scores passed!")


def test_ghost_linear_products(candidate=None):
    candidate = candidate or answers.ghost_linear_products
    generator = torch.Generator().manual_seed(205)
    inputs = torch.randn(5, 4, generator=generator, dtype=DTYPE)
    targets = torch.randn(5, 3, generator=generator, dtype=DTYPE)
    weight = torch.randn(3, 4, generator=generator, dtype=DTYPE, requires_grad=True)
    bias = torch.randn(3, generator=generator, dtype=DTYPE, requires_grad=True)
    preactivations = inputs @ weight.T + bias
    losses = (torch.tanh(preactivations) - targets).square().sum(1)
    deltas = torch.autograd.grad(losses.sum(), preactivations, retain_graph=True)[0]
    for include_bias in (False, True):
        gradients = []
        for loss in losses:
            w_gradient, b_gradient = torch.autograd.grad(loss, (weight, bias), retain_graph=True)
            gradients.append(torch.cat([w_gradient.flatten(), b_gradient]) if include_bias else w_gradient.flatten())
        gradients = torch.stack(gradients)
        expected = gradients[:3] @ gradients[3:].T
        actual = candidate(inputs[:3], deltas[:3], inputs[3:], deltas[3:], bias=include_bias)
        torch.testing.assert_close(actual, expected, msg="The two factor dot products multiply; include the bias dot product only when the layer has a bias.")
    print("All tests in test_ghost_linear_products passed!")


def test_fit_weighted_logistic(candidate=None):
    candidate = candidate or answers.fit_weighted_logistic
    x, y = _fixture()
    weights = torch.tensor([[1., 1., 1., 1., 1.], [1., 0., 1., 1., 1.], [2., .2, 1., 0., 3.]], dtype=DTYPE)
    result = candidate(x, y, weights, l2=.2)
    expected = torch.stack([_scipy_fit(x, y, w.numpy(), .2) for w in weights])
    torch.testing.assert_close(result["theta"], expected, rtol=1e-8, atol=1e-9, msg="Every weighted problem must converge independently to its own regularized optimum.")
    assert result["gradient_norms"][-1].max() <= 1e-10, "Iteration count is not evidence that a refit converged."
    assert torch.equal(result["states"][0], torch.zeros_like(expected)), "Start each counterfactual independently from zero."
    with pytest.raises(RuntimeError):
        candidate(x, y, weights, l2=.2, max_steps=1)
    print("All tests in test_fit_weighted_logistic passed!")


def test_regularized_removal_normalization():
    x, y = _fixture()
    n = len(x)
    l2 = .3
    weights = np.ones(n)
    optimum = _scipy_fit(x, y, weights, l2)
    _, hessian = answers.objective_derivatives(optimum[None], x, y, torch.ones(1, n, dtype=DTYPE), l2)
    query_x = torch.tensor([[.2, 1.4, 1.], [1.2, -.8, 1.]], dtype=DTYPE)
    query_y = torch.tensor([0., 1.], dtype=DTYPE)
    raw = answers.logistic_gradients(optimum, x, y)
    full = answers.logistic_gradients(optimum, x, y, l2)
    query_grad = answers.logistic_gradients(optimum, query_x, query_y)
    correct = answers.upweight_influence(hessian[0], full, query_grad)
    wrong = answers.upweight_influence(hessian[0], raw, query_grad)
    finite = []
    eps = 1e-5
    for index in range(n):
        losses = []
        for sign in (-1, 1):
            w = weights.copy()
            w[index] += sign * n * eps
            refitted = _scipy_fit(x, y, w, l2)
            losses.append(F.binary_cross_entropy_with_logits(query_x @ refitted, query_y, reduction="none"))
        finite.append((losses[1] - losses[0]) / (2 * eps))
    finite = torch.stack(finite)
    torch.testing.assert_close(correct, finite, rtol=1e-7, atol=1e-8)
    assert (wrong - finite).abs().max() > .01, "This control must expose the different derivative obtained by leaving the regularizer out of the upweighted row."
    print("All tests in test_regularized_removal_normalization passed!")


def test_sgd_trace(candidate=None):
    candidate = candidate or answers.sgd_trace
    x, y = _fixture()
    orders = torch.tensor([[2, 0, 4, 1, 3], [3, 1, 0, 4, 2]])
    epoch_rates = torch.tensor([.07, .02], dtype=DTYPE)
    trace = candidate(x, y, orders, epoch_rates, .2)
    expected = [torch.zeros(x.shape[1], dtype=DTYPE)]
    for epoch, order in enumerate(orders):
        for index in order:
            theta = expected[-1].clone().requires_grad_(True)
            loss = F.binary_cross_entropy_with_logits(x[index] @ theta, y[index]) + .1 * theta.square().sum()
            gradient = torch.autograd.grad(loss, theta)[0]
            expected.append((theta - epoch_rates[epoch] * gradient).detach())
    torch.testing.assert_close(trace["states"], torch.stack(expected), msg="Store the actual pre/post states, including the zero initialization.")
    assert torch.equal(trace["used_ids"], orders.flatten()), "The path attribution must use the exact training visitation order."
    with pytest.raises(ValueError):
        candidate(x, y, torch.zeros_like(orders), epoch_rates, .2)
    print("All tests in test_sgd_trace passed!")


def test_trace_loss_reductions(candidate=None):
    candidate = candidate or answers.trace_loss_reductions
    x, y = _fixture()
    orders = torch.tensor([[2, 0, 4, 1, 3], [3, 1, 0, 4, 2]])
    epoch_rates = torch.tensor([.07, .02], dtype=DTYPE)
    trace = answers.sgd_trace(x, y, orders, epoch_rates, .2)
    qx, qy = x[:3].flip(0), 1 - y[:3].flip(0)
    result = candidate(trace["states"], trace["used_ids"], trace["rates"], x, y, qx, qy, .2, chunk_size=3)
    initial_loss = F.binary_cross_entropy_with_logits(qx @ trace["states"][0], qy, reduction="none")
    final_loss = F.binary_cross_entropy_with_logits(qx @ trace["states"][-1], qy, reduction="none")
    torch.testing.assert_close(result["exact"].sum(0), initial_loss - final_loss, msg="Every query's exact allocations must telescope to its observed total loss reduction.")
    for step, index in enumerate(trace["used_ids"]):
        theta = trace["states"][step].clone().requires_grad_(True)
        train_loss = F.binary_cross_entropy_with_logits(x[index] @ theta, y[index]) + .1 * theta.square().sum()
        train_gradient = torch.autograd.grad(train_loss, theta, retain_graph=True)[0]
        query_losses = F.binary_cross_entropy_with_logits(qx @ theta, qy, reduction="none")
        query_gradients = torch.stack([torch.autograd.grad(loss, theta, retain_graph=True)[0] for loss in query_losses])
        expected = trace["rates"][step] * query_gradients @ train_gradient
        torch.testing.assert_close(result["first_order_steps"][step], expected)
    all_at_once = candidate(trace["states"], trace["used_ids"], trace["rates"], x, y, qx, qy, .2, chunk_size=100)
    for key in result:
        torch.testing.assert_close(result[key], all_at_once[key], msg="Chunking must not alter row attribution or the last partial chunk.")
    assert (result["first_order"] - result["raw_only"]).abs().max() > 1e-5, "The regularizer omission control must differ from the update actually executed."
    wrong_states = trace["states"].clone()
    wrong_states[3, 0] += .1
    with pytest.raises(ValueError, match="SGD"):
        candidate(wrong_states, trace["used_ids"], trace["rates"], x, y, qx, qy, .2)
    print("All tests in test_trace_loss_reductions passed!")


def test_class_matched_removals(candidate=None):
    candidate = candidate or answers.class_matched_removals
    scores = torch.tensor([1., -2., -2., 5., -10., 3.], dtype=DTYPE)
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    assert candidate(scores, labels, 2).tolist() == [1, 4], "Choose the lowest score in each class; preserve source-row order when scores tie."
    assert candidate(scores, labels, 4).tolist() == [1, 2, 4, 5], "Every policy must remove the same number of observed examples from each class."
    assert candidate(scores, labels, 0).numel() == 0, "A zero budget must leave the training set intact."
    with pytest.raises(ValueError):
        candidate(scores, labels, 3)
    with pytest.raises(ValueError):
        candidate(scores, labels, 8)
    print("All tests in test_class_matched_removals passed!")
