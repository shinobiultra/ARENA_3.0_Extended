"""Independent tiny oracles for the visible In-Run Data Shapley exercises."""
import itertools
import math

import pytest
import torch
import torch.nn.functional as F

from chapter16_shapley_attribution_baselines.exercises.part7_data_shapley_in_one_training_run import inrun_methods as ref


def _case():
    params = ref.init_mlp(3, 2, 41)
    x = torch.tensor([[1., 0., 1.], [1., 0., 1.], [-1., 1., 0.], [0., 2., 1.]], dtype=torch.float64)
    y = torch.tensor([1., 1., 0., 1.], dtype=torch.float64)
    return params, x, y


def _autograd_rows(params, x, y):
    model = torch.nn.Sequential(torch.nn.Linear(params[0].shape[0], params[0].shape[1]),
                                torch.nn.Tanh(), torch.nn.Linear(params[2].shape[0], 1)).double()
    with torch.no_grad():
        model[0].weight.copy_(params[0].T)
        model[0].bias.copy_(params[1])
        model[2].weight.copy_(params[2].T)
        model[2].bias.copy_(params[3])
    rows = []
    for example, label in zip(x, y):
        loss = F.binary_cross_entropy_with_logits(model(example).squeeze(), label)
        gradients = torch.autograd.grad(loss, tuple(model.parameters()))
        rows.append(torch.cat([g.T.flatten() if g.ndim == 2 else g.flatten() for g in gradients]))
    return torch.stack(rows)


def check_native_gradient_identity(params, x, y, factors, ghost):
    left = _autograd_rows(params, x[:4], y[:4])
    right = _autograd_rows(params, x[4:7], y[4:7])
    observed = ghost(factors(params, x[:4], y[:4]), factors(params, x[4:7], y[4:7]))
    torch.testing.assert_close(observed, left @ right.T, atol=1e-12, rtol=1e-10)
    print("Native classifier ghost similarities match independent autograd.")


def test_forward_and_factors(forward=ref.mlp_forward, factors=ref.mlp_factors,
                             materialize=ref.per_example_gradients):
    params, x, y = _case()
    logits, h = forward(params, x)
    torch.testing.assert_close(h, torch.tanh(x @ params[0] + params[1]))
    torch.testing.assert_close(logits, (h @ params[2] + params[3]).flatten())
    torch.testing.assert_close(materialize(factors(params, x, y)), _autograd_rows(params, x, y), atol=1e-12, rtol=1e-12)
    print("Forward pass and every per-example gradient match independent autograd.")


def test_ghost_products(ghost=ref.ghost_dot_products):
    params, x, y = _case()
    left, right = ref.mlp_factors(params, x, y), ref.mlp_factors(params, x[:2] * 0.7, 1-y[:2])
    expected = _autograd_rows(params, x, y) @ _autograd_rows(params, x[:2]*0.7, 1-y[:2]).T
    torch.testing.assert_close(ghost(left, right), expected, atol=1e-12, rtol=1e-12)
    # Inputs zero: nonzero bias gradients must survive.
    a, d = torch.zeros(2, 3, dtype=torch.float64), torch.tensor([[2.], [-3.]], dtype=torch.float64)
    torch.testing.assert_close(ghost(((a, d),), ((a, d),)), d @ d.T)
    with pytest.raises(ValueError):
        ghost(left, right[:1])
    print("Ghost products match full parameter gradients, including bias-only examples.")


def test_fixed_denominator(step=ref.sgd_step):
    params, x, y = _case()
    factors = ref.mlp_factors(params, x, y)
    rows = _autograd_rows(params, x, y)
    keep = torch.tensor([True, False, True, False])
    actual = torch.cat([v.flatten() for v in step(params, factors, 0.2, 4, keep)])
    expected = torch.cat([v.flatten() for v in params]) - 0.2 / 4 * rows[keep].sum(0)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    unchanged = step(params, factors, 0.2, 4, torch.zeros(4, dtype=torch.bool))
    for a, b in zip(params, unchanged):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    print("Subset updates keep the original denominator; empty updates do nothing.")


def test_local_game(game=ref.local_coalition_values):
    params, x, y = _case()
    before_params = [p.clone() for p in params]
    values = game(params, x, y, x, y, 0.1)
    rows = _autograd_rows(params, x, y)
    flat = torch.cat([p.flatten() for p in params])
    def loss(vector):
        logits = (torch.tanh(x @ vector[:6].reshape(3, 2) + vector[6:8]) @ vector[8:10] + vector[10])
        return F.binary_cross_entropy_with_logits(logits, y)
    expected = []
    for mask in range(16):
        ids = [i for i in range(4) if mask & (1 << i)]
        gradient = rows[ids].sum(0) if ids else torch.zeros_like(flat)
        expected.append(loss(flat) - loss(flat - 0.1 / 4 * gradient))
    torch.testing.assert_close(values, torch.stack(expected), atol=1e-12, rtol=1e-12)
    for a, b in zip(params, before_params):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    print("All sixteen local coalitions match an independent flat-parameter update.")


def test_batch_shapley(shapley=ref.batch_shapley):
    generator = torch.Generator().manual_seed(9)
    values = torch.randn(16, generator=generator, dtype=torch.float64)
    oracle = torch.zeros(4, dtype=torch.float64)
    for order in itertools.permutations(range(4)):
        mask = 0
        for player in order:
            updated = mask | (1 << player)
            oracle[player] += (values[updated] - values[mask]) / math.factorial(4)
            mask = updated
    torch.testing.assert_close(shapley(values), oracle, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(shapley(values).sum(), values[-1]-values[0], atol=1e-12, rtol=1e-12)
    with pytest.raises(ValueError):
        shapley(values[:7])
    print("Weighted subsets match all 24 permutations and conserve total utility.")


def test_taylor_allocations(first=ref.first_order_scores, second=ref.second_order_scores):
    gradients = torch.tensor([[1., 2.], [-1., 0.], [0., 1.]], dtype=torch.float64)
    gradient = torch.tensor([2., -1.], dtype=torch.float64)
    hessian = torch.tensor([[3., .5], [.5, 2.]], dtype=torch.float64)
    scale = 0.3 / 3
    values = []
    for mask in range(8):
        ids = [i for i in range(3) if mask & (1 << i)]
        delta = -scale * (gradients[ids].sum(0) if ids else torch.zeros(2, dtype=torch.float64))
        values.append(-gradient @ delta - 0.5 * delta @ hessian @ delta)
    torch.testing.assert_close(second(gradients, gradient, hessian, 0.3, 3), ref.batch_shapley(torch.stack(values)), atol=1e-12, rtol=1e-12)
    similarities = (gradients @ gradient)[:, None].repeat(1, 7)
    torch.testing.assert_close(first(similarities, 0.3, 3), scale * gradients @ gradient)
    print("First-order scaling and quadratic self/cross interactions have exact Shapley allocations.")


def test_accounting_trajectory(accountant=ref.run_accountant):
    params, x, y = _case()
    schedule = ref.make_schedule(4, 2, 3, 8)
    result = accountant(params, x, y, x, y, schedule, 0.1, exact=True)
    torch.testing.assert_close(result['exact_allocations'].sum(1), result['losses'][:-1]-result['losses'][1:], atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(result['exact_scores'].sum(), result['losses'][0]-result['losses'][-1], atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(result['scores'], result['allocations'].sum(0))
    removed = accountant(params, x, y, x, y, schedule, 0.1, removed=(0, 1, 2, 3))
    for actual, initial in zip(removed['params'], params):
        torch.testing.assert_close(actual, initial, atol=0, rtol=0)
    torch.testing.assert_close(removed['scores'], torch.zeros(4, dtype=torch.float64), atol=0, rtol=0)
    print("Exact per-step allocations telescope; repeated visits accumulate; masked training is inert.")
