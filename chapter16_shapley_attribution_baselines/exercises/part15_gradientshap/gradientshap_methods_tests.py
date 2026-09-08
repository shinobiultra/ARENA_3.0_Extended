"""Immediate, hand-derived tests for the nine Expected Gradients exercises."""

from __future__ import annotations

import math
import random

import numpy as np
import pytest
import torch as t
from torch import Tensor, nn

from . import gradientshap_methods as methods


def _linear(weight, bias=0.0):
    weight = t.tensor(weight, dtype=t.float64)
    if weight.ndim == 1:
        weight = weight[None, :]
    layer = nn.Linear(weight.shape[1], weight.shape[0], dtype=t.float64)
    with t.no_grad():
        layer.weight.copy_(weight)
        layer.bias.copy_(t.as_tensor(bias, dtype=t.float64).expand(weight.shape[0]))
    return layer


def _relu_model():
    return nn.Sequential(_linear([1.0]), nn.ReLU()).eval()


def _threshold_model():
    return nn.Sequential(_linear([2.0, 1.0], -2.0), nn.ReLU()).eval()


def _assert_unchanged(actual, expected, message):
    t.testing.assert_close(actual, expected, rtol=0, atol=0, msg=message)


def _assert_fresh(result, source, message):
    assert result.untyped_storage().data_ptr() != source.untyped_storage().data_ptr(), message


def _assert_numpy_state(actual, expected, message):
    same = (
        actual[0] == expected[0]
        and np.array_equal(actual[1], expected[1])
        and actual[2:] == expected[2:]
    )
    assert same, message


def test_sample_path_schedule(fn=methods.sample_path_schedule):
    source_ids = t.tensor([9, 2, 17], dtype=t.int64)
    source_before = source_ids.clone()
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = t.random.get_rng_state().clone()

    indices, alpha = fn(source_ids, background_rows=5, samples=6, seed=161510)
    expected_indices = t.tensor(
        [[1, 3, 0, 4, 2, 4], [0, 2, 3, 2, 0, 4], [2, 1, 1, 2, 4, 2]],
        dtype=t.int64,
    )
    expected_alpha = t.tensor(
        [
            [
                0.07504434827178863,
                0.021828046764220677,
                0.5750343028040587,
                0.47640116681752054,
                0.9277345846161994,
                0.32361717942808776,
            ],
            [
                0.7739168747191452,
                0.08990092032987362,
                0.7775792643778291,
                0.20668991838473294,
                0.708533097045279,
                0.19143075643643404,
            ],
            [
                0.3395033910267623,
                0.8732231488345139,
                0.5897005498345314,
                0.7660039848483702,
                0.4547006261027976,
                0.27734238133476885,
            ],
        ],
        dtype=t.float64,
    )
    assert t.equal(indices, expected_indices), (
        "Use the declared per-source donor stream; changing row batching or alpha draws "
        "must not shift donor identities."
    )
    assert t.equal(alpha, expected_alpha), (
        "Use the exact independent per-source alpha stream declared by the lesson."
    )
    assert indices.shape == alpha.shape == (3, 6), (
        "Return one donor and one alpha for every source row and Monte Carlo sample."
    )
    assert indices.dtype == t.int64 and alpha.dtype == t.float64, (
        "Schedules must use CPU int64 donor indices and CPU float64 alphas."
    )

    short_indices, short_alpha = fn(source_ids, 5, 3, 161510)
    assert t.equal(short_indices, indices[:, :3]), (
        "A smaller sample budget must be a literal donor-prefix of a larger budget."
    )
    t.testing.assert_close(
        short_alpha,
        alpha[:, :3],
        rtol=0,
        atol=0,
        msg="A smaller sample budget must be a literal alpha-prefix of a larger budget.",
    )
    reordered_ids = t.tensor([17, 9], dtype=t.int64)
    reordered_indices, reordered_alpha = fn(reordered_ids, 5, 6, 161510)
    assert t.equal(reordered_indices, indices[[2, 0]]), (
        "Each source ID must name its own donor stream, independent of row order."
    )
    t.testing.assert_close(
        reordered_alpha,
        alpha[[2, 0]],
        rtol=0,
        atol=0,
        msg="Each source ID must name its own alpha stream, independent of row order.",
    )
    one_indices, one_alpha = fn(source_ids[1:2].clone(), 5, 6, 161510)
    assert t.equal(one_indices[0], indices[1]), (
        "Sampling a row in its own batch must preserve its complete donor schedule."
    )
    t.testing.assert_close(
        one_alpha[0],
        alpha[1],
        rtol=0,
        atol=0,
        msg="Sampling a row in its own batch must preserve its complete alpha schedule.",
    )
    _assert_unchanged(source_ids, source_before, "Schedule generation must not mutate source IDs.")
    _assert_fresh(indices, source_ids, "A schedule must not alias its source-ID tensor.")
    assert random.getstate() == python_state, (
        "Schedule generation must not advance Python's global random stream."
    )
    _assert_numpy_state(
        np.random.get_state(), numpy_state, "Schedule generation must not advance global NumPy RNG."
    )
    assert t.equal(t.random.get_rng_state(), torch_state), (
        "Schedule generation must not advance Torch's global RNG."
    )

    with pytest.raises(ValueError, match="unique"):
        fn(t.tensor([3, 3], dtype=t.int64), 5, 2, 0)
    with pytest.raises(ValueError, match="nonnegative"):
        fn(t.tensor([-1], dtype=t.int64), 5, 2, 0)
    with pytest.raises(ValueError, match="positive integer"):
        fn(t.tensor([1], dtype=t.int64), 0, 2, 0)
    with pytest.raises(ValueError, match=r"2\*\*64"):
        fn(t.tensor([1], dtype=t.int64), 5, 2, -1)
    print("All tests in test_sample_path_schedule passed!")


def test_interpolate_paths(fn=methods.interpolate_paths):
    x = t.tensor([[2.0, 4.0]], dtype=t.float64)
    background = t.tensor([[0.0, 0.0], [1.0, 2.0]], dtype=t.float64)
    indices = t.tensor([[0, 1]], dtype=t.int64)
    alpha = t.tensor([[0.0, 0.5]], dtype=t.float64)
    snapshots = x.clone(), background.clone(), indices.clone(), alpha.clone()

    points, delta = fn(x, background, indices, alpha)
    t.testing.assert_close(
        points,
        t.tensor([[[0.0, 0.0], [1.5, 3.0]]], dtype=t.float64),
        msg="Interpolate from the selected donor at alpha=0 toward the query at alpha=1.",
    )
    t.testing.assert_close(
        delta,
        t.tensor([[[2.0, 4.0], [1.0, 2.0]]], dtype=t.float64),
        msg=(
            "Each multiplier delta must use the same complete query and selected donor "
            "as its path."
        ),
    )

    noise = t.tensor([[[1.0, -1.0], [-1.0, 2.0]]], dtype=t.float64)
    noise_before = noise.clone()
    noisy_points, noisy_delta = fn(x, background, indices, alpha, noise)
    t.testing.assert_close(
        noisy_points,
        t.tensor([[[0.0, 0.0], [1.0, 4.0]]], dtype=t.float64),
        msg="Construct every noisy path from its donor to its own perturbed query.",
    )
    t.testing.assert_close(
        noisy_delta,
        t.tensor([[[3.0, 3.0], [0.0, 4.0]]], dtype=t.float64),
        msg="Use x+noise minus donor in the noisy multiplier, not the clean delta.",
    )
    for actual, expected, name in zip(
        (x, background, indices, alpha),
        snapshots,
        ("x", "background", "indices", "alpha"),
        strict=True,
    ):
        _assert_unchanged(actual, expected, f"interpolate_paths must not mutate {name}.")
    _assert_unchanged(noise, noise_before, "interpolate_paths must not mutate noise.")
    _assert_fresh(points, x, "Path points must be new values rather than aliases of x.")
    _assert_fresh(delta, background, "Path deltas must not alias the background.")

    with pytest.raises(ValueError, match="closed interval"):
        fn(x, background, indices, t.tensor([[0.0, 1.01]], dtype=t.float64))
    with pytest.raises(ValueError, match="supplied background"):
        fn(x, background, t.tensor([[0, 2]], dtype=t.int64), alpha)
    with pytest.raises(ValueError, match="noise must have shape"):
        fn(x, background, indices, alpha, t.zeros((1, 1, 2), dtype=t.float64))
    with pytest.raises(ValueError, match="same feature"):
        fn(x, t.zeros((2, 3), dtype=t.float64), indices, alpha)
    print("All tests in test_interpolate_paths passed!")


def test_input_gradients(fn=methods.input_gradients):
    class Quadratic(nn.Module):
        def forward(self, x):
            return x[:, 0] ** 2 + 3 * x[:, 1]

    points = t.tensor(
        [[[1.0, 2.0], [-2.0, 0.0]], [[0.5, -1.0], [3.0, 4.0]]],
        dtype=t.float64,
    )
    points_before = points.clone()
    expected = t.tensor(
        [[[2.0, 3.0], [-4.0, 3.0]], [[1.0, 3.0], [6.0, 3.0]]],
        dtype=t.float64,
    )
    actual = fn(Quadratic(), points, forward_batch=1)
    t.testing.assert_close(
        actual,
        expected,
        msg="Differentiate the requested scalar output at every independent path point.",
    )
    t.testing.assert_close(
        fn(Quadratic(), points, forward_batch=20),
        expected,
        msg="A row-independent scalar model must give the same gradients under different batches.",
    )

    class Constant(nn.Module):
        def forward(self, x):
            return t.full((len(x), 1), 7.0, dtype=t.float64)

    t.testing.assert_close(
        fn(Constant(), points),
        t.zeros_like(points),
        rtol=0,
        atol=0,
        msg="A scalar function constant with respect to its input has zero input gradient.",
    )

    linear = _linear([2.0, -1.0])
    linear.train()
    linear.weight.grad = t.tensor([[7.0, 8.0]], dtype=t.float64)
    linear.bias.grad = t.tensor([9.0], dtype=t.float64)
    weight_grad = linear.weight.grad
    bias_grad = linear.bias.grad
    fn(linear, points)
    assert linear.training, "input_gradients must preserve the caller's model mode."
    assert linear.weight.grad is weight_grad and linear.bias.grad is bias_grad, (
        "input_gradients must preserve the identity of existing parameter-gradient buffers."
    )
    t.testing.assert_close(
        linear.weight.grad,
        t.tensor([[7.0, 8.0]], dtype=t.float64),
        rtol=0,
        atol=0,
        msg="Input differentiation must not accumulate or overwrite weight gradients.",
    )
    t.testing.assert_close(
        linear.bias.grad,
        t.tensor([9.0], dtype=t.float64),
        rtol=0,
        atol=0,
        msg="Input differentiation must not accumulate or overwrite bias gradients.",
    )

    class StateConsumer(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("calls", t.tensor(0.0, dtype=t.float64))

        def forward(self, x):
            self.calls.add_(1)
            self.eval()
            random.random()
            np.random.random()
            t.rand(1)
            t.set_default_dtype(t.float32)
            return x.sum(dim=1)

    consumer = StateConsumer().train()
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = t.random.get_rng_state().clone()
    default_dtype = t.get_default_dtype()
    fn(consumer, points, forward_batch=2)
    assert consumer.training, "Restore model mode even when a forward method changes it."
    assert consumer.calls.item() == 0, "Restore mutable model buffers after gradient evaluation."
    assert random.getstate() == python_state, "Restore Python's global RNG after model evaluation."
    _assert_numpy_state(
        np.random.get_state(), numpy_state, "Restore NumPy's global RNG after model evaluation."
    )
    assert t.equal(t.random.get_rng_state(), torch_state), (
        "Restore Torch's global RNG after model evaluation."
    )
    assert t.get_default_dtype() == default_dtype, (
        "Restore Torch's global default dtype after model evaluation."
    )
    _assert_unchanged(points, points_before, "input_gradients must not mutate path points.")
    _assert_fresh(actual, points, "Returned gradients must not alias path points.")

    class VectorOutput(nn.Module):
        def forward(self, x):
            return x

    with pytest.raises(ValueError, match="one scalar output"):
        fn(VectorOutput(), points)
    with pytest.raises(TypeError, match="CPU float64"):
        fn(nn.Linear(2, 1, dtype=t.float32), points)
    with pytest.raises(ValueError, match="positive integer"):
        fn(Quadratic(), points, forward_batch=0)
    print("All tests in test_input_gradients passed!")


def test_expected_gradient_samples(fn=methods.expected_gradient_samples):
    x = t.tensor([[1.0]], dtype=t.float64)
    background = t.tensor([[-1.0]], dtype=t.float64)
    indices = t.tensor([[0, 0]], dtype=t.int64)
    alpha = t.tensor([[0.25, 0.75]], dtype=t.float64)
    snapshots = x.clone(), background.clone(), indices.clone(), alpha.clone()
    relu_samples = fn(_relu_model(), x, background, indices, alpha)
    t.testing.assert_close(
        relu_samples,
        t.tensor([[[0.0], [2.0]]], dtype=t.float64),
        rtol=0,
        atol=0,
        msg="A ReLU path from -1 to 1 contributes zero before its gate and two after it.",
    )

    class Square(nn.Module):
        def forward(self, z):
            return z[:, 0] ** 2

    square_samples = fn(
        Square(),
        x,
        t.tensor([[0.0]], dtype=t.float64),
        indices,
        alpha,
    )
    t.testing.assert_close(
        square_samples,
        t.tensor([[[0.5], [1.5]]], dtype=t.float64),
        msg="For f(z)=z^2 from zero to one, each path draw contributes 2*alpha.",
    )

    affine = _linear([2.0, -3.0], 0.5)
    affine_x = t.tensor([[2.0, 1.0]], dtype=t.float64)
    affine_background = t.tensor([[0.0, 0.0], [1.0, 2.0]], dtype=t.float64)
    affine_samples = fn(
        affine,
        affine_x,
        affine_background,
        t.tensor([[0, 1]], dtype=t.int64),
        t.tensor([[0.1, 0.9]], dtype=t.float64),
    )
    t.testing.assert_close(
        affine_samples,
        t.tensor([[[4.0, -3.0], [2.0, 3.0]]], dtype=t.float64),
        msg="An affine path contribution is weight times query-minus-donor, independent of alpha.",
    )
    for actual, expected, name in zip(
        (x, background, indices, alpha),
        snapshots,
        ("x", "background", "indices", "alpha"),
        strict=True,
    ):
        _assert_unchanged(actual, expected, f"expected_gradient_samples must not mutate {name}.")
    _assert_fresh(relu_samples, x, "Per-draw contributions must not alias query inputs.")
    print("All tests in test_expected_gradient_samples passed!")


def test_summarize_expected_gradients(fn=methods.summarize_expected_gradients):
    samples = t.tensor(
        [[[[0.0, 2.0], [2.0, 0.0], [0.0, 2.0], [2.0, 0.0]]]],
        dtype=t.float64,
    ).squeeze(0)
    samples_before = samples.clone()
    means, standard_errors = fn(samples, (2, 4))
    t.testing.assert_close(
        means,
        t.ones((2, 1, 2), dtype=t.float64),
        msg="Each reported estimate must average a literal prefix of the retained draws.",
    )
    expected_se = t.tensor(
        [[[1.0, 1.0]], [[1 / math.sqrt(3), 1 / math.sqrt(3)]]],
        dtype=t.float64,
    )
    t.testing.assert_close(
        standard_errors,
        expected_se,
        msg=(
            "Use the square root of unbiased sample variance divided by the square root "
            "of prefix size."
        ),
    )
    shap_proxy = samples.var(dim=1, correction=0) / math.sqrt(samples.shape[1])
    assert not t.allclose(standard_errors[-1], shap_proxy), (
        "Ordinary estimator SE must stay distinct from SHAP 0.49.1's variance/sqrt(K) proxy."
    )
    _assert_unchanged(samples, samples_before, "Summarization must not mutate retained samples.")
    _assert_fresh(means, samples, "Prefix means must not alias the retained contribution tensor.")

    with pytest.raises(ValueError, match="between 2"):
        fn(samples, (1,))
    with pytest.raises(ValueError, match="strictly increasing"):
        fn(samples, (4, 2))
    with pytest.raises(ValueError, match="strictly increasing"):
        fn(samples, (2, 2))
    with pytest.raises(TypeError, match="nonempty tuple"):
        fn(samples, [2, 4])
    print("All tests in test_summarize_expected_gradients passed!")


def test_expected_gradients(fn=methods.expected_gradients):
    model = _linear([2.0, -3.0], 0.5)
    x = t.tensor([[2.0, 1.0]], dtype=t.float64)
    background = t.tensor([[0.0, 0.0], [1.0, 2.0]], dtype=t.float64)
    indices = t.tensor([[0, 1, 0, 1]], dtype=t.int64)
    alpha = t.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=t.float64)
    snapshots = x.clone(), background.clone(), indices.clone(), alpha.clone()
    base, phi, standard_error = fn(model, x, background, indices, alpha)
    t.testing.assert_close(
        base,
        t.tensor(-1.5, dtype=t.float64),
        msg="The base must average model outputs over every supplied background row.",
    )
    t.testing.assert_close(
        phi,
        t.tensor([[3.0, 0.0]], dtype=t.float64),
        msg="Average the retained path contributions, not donors or gradients separately.",
    )
    t.testing.assert_close(
        standard_error,
        t.tensor([[1 / math.sqrt(3), math.sqrt(3)]], dtype=t.float64),
        msg="Return the ordinary coordinatewise Monte Carlo standard error.",
    )
    prediction = model(x)[0, 0]
    t.testing.assert_close(
        base + phi.sum(),
        prediction,
        msg=(
            "The balanced affine fixture must reconstruct prediction minus the "
            "full-background base."
        ),
    )

    only_first = t.zeros((1, 4), dtype=t.int64)
    base_again, phi_first, _ = fn(model, x, background, only_first, alpha)
    t.testing.assert_close(
        base_again,
        base,
        rtol=0,
        atol=0,
        msg=(
            "The declared base uses the full background even when a finite schedule "
            "omits a donor."
        ),
    )
    t.testing.assert_close(
        phi_first,
        t.tensor([[4.0, -3.0]], dtype=t.float64),
        msg=(
            "Finite donor frequencies may move the attribution away from the "
            "full-background target."
        ),
    )
    for actual, expected, name in zip(
        (x, background, indices, alpha),
        snapshots,
        ("x", "background", "indices", "alpha"),
        strict=True,
    ):
        _assert_unchanged(actual, expected, f"expected_gradients must not mutate {name}.")
    _assert_fresh(phi, x, "Expected Gradients values must not alias query inputs.")
    with pytest.raises(ValueError, match="at least two samples"):
        fn(
            model,
            x,
            background,
            t.tensor([[0]], dtype=t.int64),
            t.tensor([[0.5]], dtype=t.float64),
        )
    print("All tests in test_expected_gradients passed!")


def test_gauss_legendre_pairs(fn=methods.gauss_legendre_pairs):
    affine = _linear([2.0, -3.0], 0.5)
    x = t.tensor([[2.0, 1.0], [1.0, -1.0]], dtype=t.float64)
    background = t.tensor([[0.0, 0.0], [1.0, 2.0]], dtype=t.float64)
    x_before, background_before = x.clone(), background.clone()
    affine_pairs = fn(affine, x, background, n_nodes=1, row_batch=1, forward_batch=2)
    expected_affine = t.tensor(
        [
            [[4.0, -3.0], [2.0, 3.0]],
            [[2.0, 3.0], [0.0, 9.0]],
        ],
        dtype=t.float64,
    )
    t.testing.assert_close(
        affine_pairs,
        expected_affine,
        msg="One midpoint node integrates every affine path exactly for each donor.",
    )

    class Square(nn.Module):
        def forward(self, z):
            return z[:, 0] ** 2

    square_pairs = fn(
        Square(),
        t.tensor([[1.0]], dtype=t.float64),
        t.tensor([[0.0], [-1.0]], dtype=t.float64),
        n_nodes=1,
    )
    t.testing.assert_close(
        square_pairs,
        t.tensor([[[1.0], [0.0]]], dtype=t.float64),
        msg="Midpoint quadrature exactly integrates the linear path gradient of a quadratic.",
    )

    threshold_pairs = fn(
        _threshold_model(),
        t.tensor([[1.0, 1.0]], dtype=t.float64),
        t.zeros((1, 2), dtype=t.float64),
        n_nodes=2,
    )
    t.testing.assert_close(
        threshold_pairs,
        t.tensor([[[1.0, 0.5]]], dtype=t.float64),
        msg=(
            "Two Gauss-Legendre nodes place one point above the alpha=2/3 gate; retain "
            "this finite approximation rather than replacing it by exact Shapley."
        ),
    )
    assert not t.allclose(
        threshold_pairs[0, 0], t.tensor([2 / 3, 1 / 3], dtype=t.float64)
    ), "Finite quadrature across a ReLU jump must remain labeled approximate."
    t.testing.assert_close(
        fn(affine, x, background, n_nodes=3, row_batch=2, forward_batch=100),
        expected_affine,
        msg="Row and forward batching must not change a row-independent affine integral.",
    )
    _assert_unchanged(x, x_before, "gauss_legendre_pairs must not mutate queries.")
    _assert_unchanged(background, background_before, "gauss_legendre_pairs must not mutate donors.")
    _assert_fresh(affine_pairs, x, "Pairwise quadrature values must not alias x.")
    with pytest.raises(ValueError, match="positive integer"):
        fn(affine, x, background, n_nodes=0)
    with pytest.raises(ValueError, match="positive integer"):
        fn(affine, x, background, n_nodes=2, row_batch=0)
    print("All tests in test_gauss_legendre_pairs passed!")


def test_noisy_expected_gradient_samples(fn=methods.noisy_expected_gradient_samples):
    x = t.tensor([[0.0]], dtype=t.float64)
    background = t.tensor([[-1.0]], dtype=t.float64)
    indices = t.tensor([[0, 0]], dtype=t.int64)
    alpha = t.tensor([[0.25, 0.75]], dtype=t.float64)
    zero_noise = t.zeros((1, 2, 1), dtype=t.float64)
    clean = methods.expected_gradient_samples(_relu_model(), x, background, indices, alpha)
    zero = fn(_relu_model(), x, background, indices, alpha, zero_noise)
    assert t.equal(zero, clean), (
        "An explicit zero-noise schedule must reproduce clean Expected Gradients exactly."
    )

    noise = t.ones((1, 2, 1), dtype=t.float64)
    snapshots = x.clone(), background.clone(), indices.clone(), alpha.clone(), noise.clone()
    noisy = fn(_relu_model(), x, background, indices, alpha, noise)
    t.testing.assert_close(
        noisy,
        t.tensor([[[0.0], [2.0]]], dtype=t.float64),
        rtol=0,
        atol=0,
        msg=(
            "For x+noise=1 and donor -1, use the perturbed query in both the sampled "
            "path point and the two-unit multiplier delta."
        ),
    )
    assert noisy.mean().item() == 1.0 and clean.mean().item() == 0.0, (
        "Nonzero query noise can change the explanation target, not merely its variance."
    )
    for actual, expected, name in zip(
        (x, background, indices, alpha, noise),
        snapshots,
        ("x", "background", "indices", "alpha", "noise"),
        strict=True,
    ):
        _assert_unchanged(
            actual,
            expected,
            f"noisy_expected_gradient_samples must not mutate {name}.",
        )
    _assert_fresh(noisy, noise, "Noisy path contributions must not alias the supplied noise.")
    print("All tests in test_noisy_expected_gradient_samples passed!")


def test_ranked_coalition_curves(fn=methods.ranked_coalition_curves):
    values = t.tensor(
        [
            [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 3.0],
            [5.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 1.0],
            [0.0, -1.0, 2.0, -3.0, 4.0, -5.0, 6.0, 0.0],
        ],
        dtype=t.float64,
    )
    phi = t.tensor(
        [[2.0, 2.0, -1.0], [1.0, -2.0, -2.0], [0.0, 0.0, 0.0]],
        dtype=t.float64,
    )
    values_before, phi_before = values.clone(), phi.clone()
    result = fn(values, phi)
    expected_orders = t.tensor([[0, 1, 2], [1, 2, 0], [0, 1, 2]], dtype=t.int64)
    expected_insertion_masks = t.tensor(
        [[0, 1, 3, 7], [0, 2, 6, 7], [0, 1, 3, 7]], dtype=t.int64
    )
    expected_deletion_masks = t.tensor(
        [[7, 6, 4, 0], [7, 5, 1, 0], [7, 6, 4, 0]], dtype=t.int64
    )
    assert set(result) == {
        "orders",
        "insertion_masks",
        "deletion_masks",
        "insertion",
        "deletion",
    }, "Expose the exact orders, masks, and raw game values used by both prefix curves."
    assert t.equal(result["orders"], expected_orders), (
        "Rank descending sign(full-empty)*phi; use positive direction at zero and stable ties."
    )
    assert t.equal(result["insertion_masks"], expected_insertion_masks), (
        "Insertion must start empty, add each ranked bit once, and end full."
    )
    assert t.equal(result["deletion_masks"], expected_deletion_masks), (
        "Deletion must start full, remove the same ranked bits, and end empty."
    )
    t.testing.assert_close(
        result["insertion"],
        values.gather(1, expected_insertion_masks),
        msg="Insertion must expose raw coalition values at its exact masks.",
    )
    t.testing.assert_close(
        result["deletion"],
        values.gather(1, expected_deletion_masks),
        msg="Deletion must expose raw coalition values at its exact masks.",
    )
    assert result["insertion"][0].tolist() == [0.0, 10.0, 30.0, 3.0], (
        "Keep nonmonotone computational interventions instead of forcing attractive curves."
    )

    supplied = t.tensor([[2, 1, 0], [0, 1, 2], [1, 0, 2]], dtype=t.int64)
    supplied_before = supplied.clone()
    controlled = fn(values, phi, supplied)
    assert t.equal(controlled["orders"], supplied), (
        "Use a frozen supplied order unchanged rather than reranking audit rows."
    )
    _assert_unchanged(supplied, supplied_before, "Explicit feature orders must not be mutated.")
    _assert_fresh(controlled["orders"], supplied, "Returned orders must not alias supplied orders.")
    _assert_unchanged(values, values_before, "ranked_coalition_curves must not mutate values.")
    _assert_unchanged(phi, phi_before, "ranked_coalition_curves must not mutate phi.")
    with pytest.raises(ValueError, match="exactly once"):
        fn(values, phi, t.zeros((3, 3), dtype=t.int64))
    with pytest.raises(ValueError, match=r"2\*\*D"):
        fn(values[:, :7], phi)
    print("All tests in test_ranked_coalition_curves passed!")
