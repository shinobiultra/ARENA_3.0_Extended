"""Immediate DeepSHAP checks with small, independently computed numerical oracles."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
import torch as t
from torch import nn

from . import deepshap_methods as methods


def _linear(weight, bias=0.0):
    weight = t.tensor(weight, dtype=t.float64)
    if weight.ndim == 1:
        weight = weight[None, :]
    layer = nn.Linear(weight.shape[1], weight.shape[0], dtype=t.float64)
    with t.no_grad():
        layer.weight.copy_(weight)
        layer.bias.copy_(t.as_tensor(bias, dtype=t.float64).expand(weight.shape[0]))
    return layer


def _threshold_model():
    return nn.Sequential(_linear([2.0, 1.0], -2.0), nn.ReLU(), _linear([1.0])).eval()


def _minimum_model():
    # On nonnegative inputs: ReLU(x1) - ReLU(x1-x0) = min(x0,x1).
    return nn.Sequential(
        _linear([[0.0, 1.0], [-1.0, 1.0]]),
        nn.ReLU(),
        _linear([1.0, -1.0]),
    ).eval()


def _relu_model():
    return nn.Sequential(_linear([1.0]), nn.ReLU(), _linear([1.0])).eval()


def _assert_unchanged(actual, expected, message):
    t.testing.assert_close(actual, expected, rtol=0, atol=0, msg=message)


def _assert_fresh(result, source, message):
    assert result.untyped_storage().data_ptr() != source.untyped_storage().data_ptr(), message


def test_paired_forward(fn=methods.paired_forward):
    first = _linear([[1.0, -2.0], [0.5, 1.0]], [0.5, -1.0])
    model = nn.Sequential(first, nn.ReLU(), _linear([2.0, -1.0], 0.25)).eval()
    x = t.tensor([[2.0, 1.0], [-1.0, 3.0]], dtype=t.float64)
    background = t.tensor([[0.0, 0.0], [1.0, -1.0]], dtype=t.float64)
    x_before, b_before = x.clone(), background.clone()

    x_states, b_states = fn(model, x, background)
    expected_x = [
        x,
        t.tensor([[0.5, 1.0], [-6.5, 1.5]], dtype=t.float64),
        t.tensor([[0.5, 1.0], [0.0, 1.5]], dtype=t.float64),
        t.tensor([[0.25], [-1.25]], dtype=t.float64),
    ]
    expected_b = [
        background,
        t.tensor([[0.5, -1.0], [3.5, -1.5]], dtype=t.float64),
        t.tensor([[0.5, 0.0], [3.5, 0.0]], dtype=t.float64),
        t.tensor([[1.25], [7.25]], dtype=t.float64),
    ]
    assert len(x_states) == len(b_states) == 4, (
        "Return the input followed by every layer output for both forward passes."
    )
    for actual, expected in zip(x_states, expected_x, strict=True):
        t.testing.assert_close(actual, expected)
    for actual, expected in zip(b_states, expected_b, strict=True):
        t.testing.assert_close(actual, expected)
    _assert_unchanged(x, x_before, "paired_forward must not overwrite query inputs.")
    _assert_unchanged(background, b_before, "paired_forward must not overwrite background rows.")
    _assert_fresh(x_states[0], x, "Saved query states must not alias mutable learner inputs.")
    _assert_fresh(b_states[0], background, "Saved reference states must not alias learner inputs.")

    class LinearSubclass(nn.Linear):
        pass

    class SequentialSubclass(nn.Sequential):
        pass

    invalid = [
        (
            nn.Sequential(_linear([1.0]), nn.Sigmoid(), _linear([1.0])).eval(),
            TypeError,
            "Unsupported module Sigmoid",
        ),
        (
            nn.Sequential(_linear([1.0]), nn.ReLU(inplace=True), _linear([1.0])).eval(),
            ValueError,
            "In-place ReLU",
        ),
        (
            nn.Sequential(LinearSubclass(1, 1, dtype=t.float64)).eval(),
            TypeError,
            "Unsupported module LinearSubclass",
        ),
        (
            SequentialSubclass(_linear([1.0])).eval(),
            TypeError,
            "exact nn.Sequential",
        ),
    ]
    shared = nn.ReLU()
    invalid.append(
        (
            nn.Sequential(_linear([1.0]), shared, shared, _linear([1.0])).eval(),
            ValueError,
            "distinct module",
        )
    )
    for bad_model, error_type, message in invalid:
        with pytest.raises(error_type, match=message):
            fn(bad_model, t.zeros((1, 1), dtype=t.float64), t.zeros((1, 1), dtype=t.float64))
    with pytest.raises(ValueError, match="finite"):
        fn(model, t.tensor([[float("nan"), 0.0]], dtype=t.float64), background)
    with pytest.raises(ValueError, match="expects 3 values"):
        fn(
            nn.Sequential(_linear([1.0, 2.0, 3.0])).eval(),
            t.zeros((1, 2), dtype=t.float64),
            t.zeros((1, 2), dtype=t.float64),
        )
    print("All tests in test_paired_forward passed!")


def test_rescale_slope(fn=methods.rescale_slope):
    zx = t.tensor([2.0, -2.0, 2.0, 0.0, 1.0, 4e-7], dtype=t.float64)
    zb = t.tensor([-1.0, -1.0, 1.0, 0.0, 1.0 + 1e-8, -4e-7], dtype=t.float64)
    zx_before, zb_before = zx.clone(), zb.clone()
    actual = fn(zx, zb)
    t.testing.assert_close(
        actual,
        t.tensor([2 / 3, 0.0, 1.0, 0.0, 1.0, 0.0], dtype=t.float64),
        msg="Use a ReLU secant except in the explicit small-denominator reference-gradient branch.",
    )
    t.testing.assert_close(
        fn(zx[-1:], zb[-1:], epsilon=0),
        t.tensor([0.5], dtype=t.float64),
        msg="epsilon=0 must retain every nonzero secant, including a tiny ReLU crossing.",
    )
    grid = fn(zx[:2, None], zb[:3][None, :])
    assert grid.shape == (2, 3), "Actual and reference activations must broadcast to a pair grid."
    _assert_unchanged(zx, zx_before, "rescale_slope must not mutate actual activations.")
    _assert_unchanged(zb, zb_before, "rescale_slope must not mutate reference activations.")
    _assert_fresh(actual, zx, "Rescale slopes must be newly computed values, not an input alias.")
    with pytest.raises(ValueError, match="nonnegative"):
        fn(zx, zb, epsilon=-1)
    with pytest.raises(ValueError, match="finite"):
        fn(t.tensor([float("inf")], dtype=t.float64), t.zeros(1, dtype=t.float64))
    print("All tests in test_rescale_slope passed!")


def test_rescale_multipliers(fn=methods.rescale_multipliers):
    x = t.tensor([[1.0, 1.0], [2.0, 1.0]], dtype=t.float64)
    background = t.tensor([[0.0, 0.0], [1.0, -1.0]], dtype=t.float64)
    x_before, b_before = x.clone(), background.clone()
    actual = fn(_threshold_model(), x, background)
    expected = t.tensor(
        [[[2 / 3, 1 / 3], [1.0, 0.5]], [[1.2, 0.6], [1.5, 0.75]]],
        dtype=t.float64,
    )
    t.testing.assert_close(
        actual,
        expected,
        msg="Propagate each query/reference ReLU secant through both Linear weight matrices.",
    )
    affine = nn.Sequential(_linear([2.0, -3.0], 5.0)).eval()
    t.testing.assert_close(
        fn(affine, x, background),
        t.tensor([2.0, -3.0], dtype=t.float64).expand(2, 2, 2),
    )
    _assert_unchanged(x, x_before, "rescale_multipliers must not mutate x.")
    _assert_unchanged(background, b_before, "rescale_multipliers must not mutate background.")
    _assert_fresh(actual, x, "Input multipliers must not alias x.")
    shared = nn.ReLU()
    with pytest.raises(ValueError, match="shared modules|distinct module"):
        fn(
            nn.Sequential(_linear([1.0]), shared, shared, _linear([1.0])).eval(),
            t.ones((1, 1), dtype=t.float64),
            t.zeros((1, 1), dtype=t.float64),
        )
    with pytest.raises(ValueError, match="nonnegative"):
        fn(affine, x, background, epsilon=-1)
    print("All tests in test_rescale_multipliers passed!")


def test_single_reference(fn=methods.single_reference):
    affine = nn.Sequential(_linear([2.0, -3.0], 5.0)).eval()
    x = t.tensor([[4.0, 2.0]], dtype=t.float64)
    reference = t.tensor([1.0, -1.0], dtype=t.float64)
    t.testing.assert_close(
        fn(affine, x, reference),
        t.tensor([[6.0, -9.0]], dtype=t.float64),
        msg="A Linear layer allocates weight times the input-reference difference.",
    )
    threshold = fn(
        _threshold_model(),
        t.tensor([[1.0, 1.0]], dtype=t.float64),
        t.zeros(2, dtype=t.float64),
    )
    t.testing.assert_close(threshold, t.tensor([[2 / 3, 1 / 3]], dtype=t.float64))
    minimum = fn(
        _minimum_model(),
        t.tensor([[2.0, 1.0]], dtype=t.float64),
        t.zeros(2, dtype=t.float64),
    )
    t.testing.assert_close(
        minimum,
        t.tensor([[0.0, 1.0]], dtype=t.float64),
        msg=(
            "Rescale follows this minimum circuit's composition; it need not split exact "
            "feature credit."
        ),
    )
    tiny_x = t.tensor([[4e-7]], dtype=t.float64)
    tiny_b = t.tensor([-4e-7], dtype=t.float64)
    t.testing.assert_close(fn(_relu_model(), tiny_x, tiny_b), t.zeros_like(tiny_x))
    t.testing.assert_close(
        fn(_relu_model(), tiny_x, tiny_b, epsilon=0),
        t.tensor([[4e-7]], dtype=t.float64),
        msg="The exact nonzero secant restores the tiny crossing output difference.",
    )
    with pytest.raises(ValueError, match="1 dimensions"):
        fn(affine, x, reference[None])
    print("All tests in test_single_reference passed!")


def test_background_deepshap(fn=methods.background_deepshap):
    model = _relu_model()
    x = t.tensor([[2.0]], dtype=t.float64)
    background = t.tensor([[-2.0], [1.0]], dtype=t.float64)
    x_before, b_before = x.clone(), background.clone()
    base, phi = fn(model, x, background)
    t.testing.assert_close(base, t.tensor(0.5, dtype=t.float64))
    t.testing.assert_close(
        phi,
        t.tensor([[1.5]], dtype=t.float64),
        msg="Average the complete values 2 and 1; do not explain from the mean reference -0.5.",
    )
    mean_reference_value = 2.0
    assert float(phi[0, 0]) != mean_reference_value, (
        "For nonlinear models, an attribution averaged over references is not an "
        "attribution from their mean."
    )
    reversed_base, reversed_phi = fn(model, x, background.flip(0))
    t.testing.assert_close(reversed_base, base)
    t.testing.assert_close(reversed_phi, phi)
    affine = nn.Sequential(_linear([2.0, -3.0], 5.0)).eval()
    affine_x = t.tensor([[4.0, 2.0]], dtype=t.float64)
    affine_b = t.tensor([[1.0, -1.0], [2.0, 0.0]], dtype=t.float64)
    affine_base, affine_phi = fn(affine, affine_x, affine_b)
    t.testing.assert_close(affine_base, t.tensor(9.5, dtype=t.float64))
    t.testing.assert_close(affine_phi, t.tensor([[5.0, -7.5]], dtype=t.float64))
    _assert_unchanged(x, x_before, "background_deepshap must not mutate x.")
    _assert_unchanged(background, b_before, "background_deepshap must not mutate background.")
    _assert_fresh(phi, x, "DeepSHAP values must not alias the query tensor.")
    print("All tests in test_background_deepshap passed!")


def test_coalition_values(fn=methods.coalition_values):
    model = _minimum_model()
    x = t.tensor([[2.0, 2.0], [1.0, 2.0]], dtype=t.float64)
    background = t.tensor([[0.0, 2.0], [2.0, 0.0]], dtype=t.float64)
    masks = t.tensor([[False, False], [True, False], [False, True], [True, True]])
    snapshots = x.clone(), background.clone(), masks.clone()
    expected = t.tensor([[0.0, 1.0, 1.0, 2.0], [0.0, 0.5, 1.0, 1.0]], dtype=t.float64)
    for row_batch, forward_batch in [(1, 1), (8, 65_536)]:
        t.testing.assert_close(
            fn(model, x, background, masks, row_batch, forward_batch),
            expected,
            msg=(
                "Each missing block must come from one intact donor before predictions are "
                "averaged."
            ),
        )
    mean_donor_empty = 1.0
    assert float(expected[0, 0]) != mean_donor_empty, (
        "The empty game averages min(0,2) and min(2,0); min of the mean donor is a different game."
    )
    _assert_unchanged(x, snapshots[0], "coalition_values must not mutate x.")
    _assert_unchanged(background, snapshots[1], "coalition_values must not mutate background.")
    assert t.equal(masks, snapshots[2]), "coalition_values must not mutate masks."
    result = fn(model, x, background, masks)
    _assert_fresh(result, x, "The coalition table must not alias x.")
    with pytest.raises(TypeError, match="Boolean"):
        fn(model, x, background, masks.long())

    class LinearSubclass(nn.Linear):
        pass

    with pytest.raises(TypeError, match="LinearSubclass"):
        fn(
            nn.Sequential(LinearSubclass(2, 1, dtype=t.float64)).eval(),
            x,
            background,
            masks,
        )
    print("All tests in test_coalition_values passed!")


def test_exact_shapley(fn=methods.exact_shapley):
    rng = np.random.default_rng(161414)
    games = rng.normal(size=(2, 3, 16))
    expected = np.zeros((2, 3, 4))
    for order in itertools.permutations(range(4)):
        subset = 0
        for feature in order:
            added = subset | (1 << feature)
            expected[..., feature] += (games[..., added] - games[..., subset]) / math.factorial(4)
            subset = added
    values = t.from_numpy(games)
    snapshot = values.clone()
    actual = fn(values)
    np.testing.assert_allclose(
        actual.numpy(),
        expected,
        atol=1e-12,
        err_msg="Exact Shapley must average each feature's marginal over every feature ordering.",
    )
    np.testing.assert_allclose(actual.sum(-1), games[..., -1] - games[..., 0], atol=1e-12)
    t.testing.assert_close(
        fn(t.tensor([0.0, 0.0, 0.0, 1.0], dtype=t.float64)),
        t.tensor([0.5, 0.5], dtype=t.float64),
    )
    _assert_unchanged(values, snapshot, "exact_shapley must not mutate the coalition table.")
    _assert_fresh(actual, values, "Exact Shapley values must not alias the coalition table.")
    with pytest.raises(ValueError, match=r"2\*\*D"):
        fn(t.zeros(3, dtype=t.float64))
    with pytest.raises(ValueError, match="coalition axis"):
        fn(t.tensor(0.0, dtype=t.float64))
    print("All tests in test_exact_shapley passed!")


def test_reveal_cancel_groups(fn=methods.reveal_cancel_groups):
    reference = t.tensor([0.0], dtype=t.float64)
    positive = t.tensor([2.0], dtype=t.float64)
    negative = t.tensor([-1.0], dtype=t.float64)
    snapshots = reference.clone(), positive.clone(), negative.clone()
    positive_value, negative_value = fn(t.relu, reference, positive, negative)
    t.testing.assert_close(positive_value, t.tensor([1.5], dtype=t.float64))
    t.testing.assert_close(negative_value, t.tensor([-0.5], dtype=t.float64))
    t.testing.assert_close(
        t.stack((2 - positive_value[0], -negative_value[0])),
        t.tensor([0.5, 0.5], dtype=t.float64),
        msg="For x1-ReLU(x1-x0), subtract grouped ReLU credit from the direct x1 path.",
    )
    for actual, expected, name in zip(
        (reference, positive, negative),
        snapshots,
        ("reference", "positive", "negative"),
        strict=True,
    ):
        _assert_unchanged(actual, expected, f"RevealCancel must not mutate {name}.")
    _assert_fresh(positive_value, positive, "RevealCancel must return fresh positive-group values.")
    _assert_fresh(negative_value, negative, "RevealCancel must return fresh negative-group values.")
    broadcast_positive, broadcast_negative = fn(
        t.relu,
        t.zeros((2, 1), dtype=t.float64),
        t.tensor([[2.0, 1.0]], dtype=t.float64),
        t.tensor(-1.0, dtype=t.float64),
    )
    assert broadcast_positive.shape == broadcast_negative.shape == (2, 2), (
        "Broadcast reference, positive and negative changes to the same output shape."
    )
    with pytest.raises(ValueError, match="nonnegative"):
        fn(t.relu, reference, -positive, negative)
    print("All tests in test_reveal_cancel_groups passed!")


def test_ranked_replacement_curves(fn=methods.ranked_replacement_curves):
    values = t.tensor(
        [
            [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 3.0],
            [5.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 1.0],
            [0.0, -1.0, 2.0, -3.0, 4.0, -5.0, 6.0, 0.0],
        ],
        dtype=t.float64,
    )
    phi = t.tensor([[2.0, 2.0, -1.0], [1.0, -2.0, -2.0], [0.0, 0.0, 0.0]], dtype=t.float64)
    values_before, phi_before = values.clone(), phi.clone()
    result = fn(values, phi)
    expected_orders = t.tensor([[0, 1, 2], [1, 2, 0], [0, 1, 2]])
    expected_insertion_masks = t.tensor([[0, 1, 3, 7], [0, 2, 6, 7], [0, 1, 3, 7]])
    expected_deletion_masks = t.tensor([[7, 6, 4, 0], [7, 5, 1, 0], [7, 6, 4, 0]])
    assert t.equal(result["orders"], expected_orders), (
        "Rank descending sign(full-empty)*phi; treat zero direction as positive and break "
        "ties by feature index."
    )
    assert t.equal(result["insertion_masks"], expected_insertion_masks), (
        "Insertion starts empty and sets each ranked feature bit once, ending at the full mask."
    )
    assert t.equal(result["deletion_masks"], expected_deletion_masks), (
        "Deletion starts full and clears the same ranked feature bits, ending at the empty mask."
    )
    t.testing.assert_close(result["insertion"], values.gather(1, expected_insertion_masks))
    t.testing.assert_close(result["deletion"], values.gather(1, expected_deletion_masks))
    assert result["insertion"][0].tolist() == [0.0, 10.0, 30.0, 3.0], (
        "Return raw game values in prefix order; never force a replacement curve to be monotone."
    )
    supplied = t.tensor([[2, 1, 0], [0, 1, 2], [1, 0, 2]])
    supplied_before = supplied.clone()
    controlled = fn(values, phi, supplied)
    assert t.equal(controlled["orders"], supplied), (
        "Use explicitly supplied orders unchanged; do not rerank the controlled experiment."
    )
    _assert_unchanged(supplied, supplied_before, "Explicit orders must not be mutated.")
    _assert_fresh(controlled["orders"], supplied, "Returned orders must not alias supplied orders.")
    _assert_unchanged(values, values_before, "ranked_replacement_curves must not mutate values.")
    _assert_unchanged(phi, phi_before, "ranked_replacement_curves must not mutate phi.")
    with pytest.raises(ValueError, match="exactly once"):
        fn(values, phi, t.zeros((3, 3), dtype=t.int64))
    print("All tests in test_ranked_replacement_curves passed!")


def test_coalition_masks(fn=methods.coalition_masks):
    expected = t.tensor([[False, False], [True, False], [False, True], [True, True]])
    actual = fn(2)
    assert t.equal(actual, expected), (
        "Feature zero is the least-significant bit; include empty and full coalitions."
    )
    assert actual.dtype == t.bool and actual.device.type == "cpu", (
        "Return CPU Boolean masks so each column directly selects query versus donor values."
    )
    assert fn(4).sum(0).tolist() == [8] * 4, (
        "Across all 16 coalitions, each of the four features appears in exactly half."
    )
    _assert_fresh(actual, fn(2), "Each coalition_masks call must return an independent tensor.")
    with pytest.raises(ValueError, match="positive integer"):
        fn(True)
    print("All tests in test_coalition_masks passed!")


def test_gradient_delta(fn=methods.gradient_delta):
    model = _threshold_model()
    x = t.tensor([[1.0, 1.0]], dtype=t.float64)
    background = t.zeros((2, 2), dtype=t.float64)
    x_before, b_before = x.clone(), background.clone()
    for parameter in model.parameters():
        parameter.grad = t.full_like(parameter, 7.0)
    gradients_before = [parameter.grad.clone() for parameter in model.parameters()]
    actual = fn(model, x, background)
    t.testing.assert_close(
        actual,
        t.tensor([[2.0, 1.0]], dtype=t.float64),
        msg="gradient_delta uses the ordinary endpoint gradient at x times x-mean(background).",
    )
    _assert_unchanged(x, x_before, "gradient_delta must not mutate x or enable gradients on it.")
    _assert_unchanged(background, b_before, "gradient_delta must not mutate background.")
    assert not x.requires_grad and x.grad is None, (
        "Take gradients on a separate query tensor; preserve the learner's input gradient state."
    )
    for parameter, expected in zip(model.parameters(), gradients_before, strict=True):
        _assert_unchanged(
            parameter.grad,
            expected,
            "gradient_delta must not accumulate parameter gradients.",
        )
    _assert_fresh(actual, x, "gradient_delta must return a fresh attribution tensor.")
    constant = nn.Sequential(_linear([0.0, 0.0], 3.0)).eval()
    t.testing.assert_close(fn(constant, x, background), t.zeros_like(x))
    print("All tests in test_gradient_delta passed!")
