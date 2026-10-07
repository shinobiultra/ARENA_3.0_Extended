"""Tests for [16.9] Data Shapley in One Training Run.

Wherever possible these compare against independent oracles: autograd, brute-force averages over all
permutations, or scipy. None of them import the solutions file.
"""

import itertools
import math
from typing import Callable

import torch as t
import torch.nn.functional as F

DTYPE = t.float64


# ---------------------------------------------------------------------------------------------------------------
# Oracles
# ---------------------------------------------------------------------------------------------------------------


def _permutation_shapley(utility: Callable, n: int) -> t.Tensor:
    """Shapley values as the average marginal contribution over all n! orderings (independent of any formula)."""
    values = t.zeros(n, dtype=DTYPE)
    for order in itertools.permutations(range(n)):
        coalition, previous = frozenset(), utility(frozenset())
        for i in order:
            coalition = coalition | {i}
            current = utility(coalition)
            values[i] += current - previous
            previous = current
    return values / math.factorial(n)


def _random_game(n: int, seed: int) -> Callable:
    """A random game with v(empty) = 0 and plenty of interactions."""
    gen = t.Generator().manual_seed(seed)
    table = t.randn(2**n, generator=gen, dtype=DTYPE)
    table[0] = 0.0
    return lambda S: table[sum(1 << i for i in S)].item()


def _autograd_forward(leaves: list[t.Tensor], x: t.Tensor) -> t.Tensor:
    h = x
    n_layers = len(leaves) // 2
    for k in range(n_layers):
        h = h @ leaves[2 * k] + leaves[2 * k + 1]
        if k < n_layers - 1:
            h = t.tanh(h)
    return h.squeeze(-1)


def _flat(params) -> t.Tensor:
    return t.cat([p.flatten() for pair in params for p in pair])


def _autograd_per_example_grads(params, x: t.Tensor, y: t.Tensor) -> t.Tensor:
    """One autograd backward pass per example, flattened as (W.flatten(), b) per layer."""
    rows = []
    for xi, yi in zip(x, y):
        leaves = [p.detach().clone().requires_grad_(True) for pair in params for p in pair]
        loss = F.binary_cross_entropy_with_logits(_autograd_forward(leaves, xi[None]), yi[None])
        rows.append(t.cat([g.flatten() for g in t.autograd.grad(loss, leaves)]))
    return t.stack(rows)


def _autograd_weighted_step(params, x, y, lr, denominator, weights):
    leaves = [p.detach().clone().requires_grad_(True) for pair in params for p in pair]
    losses = F.binary_cross_entropy_with_logits(_autograd_forward(leaves, x), y, reduction="none")
    grads = t.autograd.grad((weights * losses).sum() / denominator, leaves)
    new = [(p - lr * g).detach() for p, g in zip(leaves, grads)]
    return [(new[2 * k], new[2 * k + 1]) for k in range(len(params))]


def _val_loss(params, x, y) -> float:
    leaves = [p for pair in params for p in pair]
    return F.binary_cross_entropy_with_logits(_autograd_forward(leaves, x), y).item()


def _small_problem(seed: int = 0, n: int = 6, d: int = 5):
    gen = t.Generator().manual_seed(seed)
    x = t.randn(n, d, generator=gen, dtype=DTYPE)
    y = (t.rand(n, generator=gen, dtype=DTYPE) > 0.5).to(DTYPE)
    return x, y


# ---------------------------------------------------------------------------------------------------------------
# Section 1
# ---------------------------------------------------------------------------------------------------------------


def _logreg_oracle(x, y, coalition, steps, lr):
    """Full-batch GD with autograd on sum_{i in S} BCE_i / n."""
    n = len(x)
    w = t.zeros(x.shape[1], dtype=DTYPE, requires_grad=True)
    b = t.zeros((), dtype=DTYPE, requires_grad=True)
    mask = t.zeros(n, dtype=DTYPE)
    mask[list(coalition)] = 1.0
    for _ in range(steps):
        loss = (mask * F.binary_cross_entropy_with_logits(x @ w + b, y, reduction="none")).sum() / n
        gw, gb = t.autograd.grad(loss, [w, b])
        with t.no_grad():
            w -= lr * gw
            b -= lr * gb
    return w.detach(), b.detach()


def test_train_on_subset(train_on_subset: Callable):
    x, y = _small_problem(1)
    w, b = train_on_subset(x, y, frozenset(), steps=5, lr=0.3)
    assert w.shape == (x.shape[1],) and b.shape == (), f"Expected w of shape {(x.shape[1],)} and scalar b, got {tuple(w.shape)}, {tuple(b.shape)}"
    assert w.abs().max() == 0 and b.abs() == 0, "Training on the empty coalition should leave the parameters at zero"
    for coalition in [frozenset({2}), frozenset({0, 3, 5}), frozenset(range(6))]:
        w, b = train_on_subset(x, y, coalition, steps=7, lr=0.3)
        w_ref, b_ref = _logreg_oracle(x, y, coalition, 7, 0.3)
        t.testing.assert_close(w, w_ref, msg=f"Wrong weights for coalition {set(coalition)}. Are you dividing by n = len(x) (not |S|)?")
        t.testing.assert_close(b, b_ref, msg=f"Wrong bias for coalition {set(coalition)}")
    print("All tests in `test_train_on_subset` passed!")


def test_subset_utility(subset_utility: Callable):
    x, y = _small_problem(2)
    vx, vy = _small_problem(3)
    assert abs(subset_utility(frozenset(), x, y, vx, vy, steps=6, lr=0.5)) < 1e-12, "v(empty) should be 0"
    for coalition in [frozenset({1}), frozenset({0, 2, 4})]:
        w, b = _logreg_oracle(x, y, coalition, 6, 0.5)
        expected = math.log(2) - F.binary_cross_entropy_with_logits(vx @ w + b, vy).item()
        result = subset_utility(coalition, x, y, vx, vy, steps=6, lr=0.5)
        assert isinstance(result, float), "subset_utility should return a Python float"
        assert abs(result - expected) < 1e-10, f"v({set(coalition)}): expected {expected:.6f}, got {result:.6f}"
    print("All tests in `test_subset_utility` passed!")


def test_exact_data_shapley(exact_data_shapley: Callable):
    for n, seed in [(3, 0), (5, 1)]:
        game = _random_game(n, seed)
        calls = []
        values, table = exact_data_shapley(lambda S: calls.append(S) or game(S), n)
        assert values.shape == (n,), f"Expected values of shape ({n},), got {tuple(values.shape)}"
        assert len(table) == 2**n, f"The table should contain all {2**n} coalitions, got {len(table)}"
        assert len(calls) == 2**n, f"Each of the {2**n} coalitions should be evaluated exactly once, got {len(calls)} calls"
        expected = _permutation_shapley(game, n)
        t.testing.assert_close(values.to(DTYPE), expected, msg="Values don't match the average over all orderings")
    # Symmetry and null player on a structured game: players 0 and 1 are interchangeable, player 3 never matters
    game = lambda S: float(len(S & {0, 1, 2}) ** 2)
    values, _ = exact_data_shapley(game, 4)
    assert abs(values[0] - values[1]) < 1e-12 and abs(values[3]) < 1e-12, "Symmetry or null-player property violated"
    print("All tests in `test_exact_data_shapley` passed!")


def test_leave_one_out(leave_one_out: Callable):
    game = _random_game(4, 2)
    everyone = frozenset(range(4))
    expected = t.tensor([game(everyone) - game(everyone - {i}) for i in range(4)], dtype=DTYPE)
    t.testing.assert_close(leave_one_out(game, 4).to(DTYPE), expected)
    print("All tests in `test_leave_one_out` passed!")


# ---------------------------------------------------------------------------------------------------------------
# Section 2
# ---------------------------------------------------------------------------------------------------------------


def _mc_oracle(game, n, M, seed, tolerance=None):
    gen = t.Generator().manual_seed(seed)
    totals = t.zeros(n, dtype=DTYPE)
    v_empty, v_full = game(frozenset()), game(frozenset(range(n)))
    for _ in range(M):
        perm = t.randperm(n, generator=gen).tolist()
        coalition, previous = frozenset(), v_empty
        for i in perm:
            if tolerance is not None and abs(v_full - previous) < tolerance:
                break
            coalition = coalition | {i}
            current = game(coalition)
            totals[i] += current - previous
            previous = current
    return totals / M


class _Counter:
    def __init__(self, fn):
        self.fn, self.calls = fn, 0

    def __call__(self, S):
        self.calls += 1
        return self.fn(S)


def test_mc_data_shapley(mc_data_shapley: Callable):
    weights = t.tensor([0.5, -1.0, 2.0, 0.25, 0.0], dtype=DTYPE)
    additive = lambda S: weights[list(S)].sum().item()
    counted = _Counter(additive)
    result = mc_data_shapley(counted, 5, 3, seed=0)
    assert result.shape == (5,), f"Expected shape (5,), got {tuple(result.shape)}"
    t.testing.assert_close(result.to(DTYPE), weights, msg="For an additive game, every ordering gives the exact Shapley values")
    assert counted.calls == 1 + 5 * 3, f"Expected 1 + n*M = 16 utility calls, got {counted.calls}"
    game = _random_game(6, 3)
    for seed in [0, 7]:
        t.testing.assert_close(
            mc_data_shapley(game, 6, 10, seed=seed).to(DTYPE), _mc_oracle(game, 6, 10, seed),
            msg="Doesn't match the reference estimator. Are you using one generator and t.randperm(n, generator=gen)?",
        )
    # With many orderings, it should be close to the exact values
    error = (mc_data_shapley(game, 6, 2000, seed=1).to(DTYPE) - _permutation_shapley(game, 6)).abs().max()
    assert error < 0.1, f"With 2000 orderings the estimate should be close to exact; max error {error:.3f}"
    print("All tests in `test_mc_data_shapley` passed!")


def test_tmc_data_shapley(tmc_data_shapley: Callable, mc_data_shapley: Callable):
    game = _random_game(6, 4)
    counted = _Counter(game)
    result = tmc_data_shapley(counted, 6, 8, tolerance=0.0, seed=2)
    t.testing.assert_close(result.to(DTYPE), mc_data_shapley(game, 6, 8, seed=2).to(DTYPE), msg="tolerance=0 should reproduce plain MC")
    assert counted.calls == 2 + 6 * 8, f"With tolerance 0, expected 2 + n*M = 50 calls, got {counted.calls}"
    counted = _Counter(game)
    result = tmc_data_shapley(counted, 6, 8, tolerance=1e9, seed=2)
    assert result.abs().max() == 0 and counted.calls == 2, "With a huge tolerance, every ordering truncates immediately"
    for tolerance in [0.3, 1.0]:
        t.testing.assert_close(
            tmc_data_shapley(game, 6, 20, tolerance=tolerance, seed=5).to(DTYPE), _mc_oracle(game, 6, 20, 5, tolerance),
            msg=f"Doesn't match the reference truncated estimator at tolerance {tolerance}",
        )
    print("All tests in `test_tmc_data_shapley` passed!")


# ---------------------------------------------------------------------------------------------------------------
# Section 3
# ---------------------------------------------------------------------------------------------------------------


def test_mlp_forward(mlp_forward: Callable, init_mlp: Callable):
    x, _ = _small_problem(5, n=4, d=6)
    for sizes in [[6, 1], [6, 5, 1], [6, 4, 3, 1]]:
        params = init_mlp(sizes, seed=1)
        for _, b in params:
            b.normal_()
        logits, layer_inputs = mlp_forward(params, x)
        assert logits.shape == (4,), f"Logits should have shape (batch,), got {tuple(logits.shape)}"
        assert len(layer_inputs) == len(params), "There should be one layer input per layer"
        t.testing.assert_close(layer_inputs[0], x, msg="The first layer's input should be x itself")
        layers = []
        for k, (W, b) in enumerate(params):
            linear = t.nn.Linear(W.shape[0], W.shape[1]).to(DTYPE)
            with t.no_grad():
                linear.weight.copy_(W.T)
                linear.bias.copy_(b)
            layers += [linear] + ([t.nn.Tanh()] if k < len(params) - 1 else [])
        t.testing.assert_close(logits, t.nn.Sequential(*layers)(x).squeeze(-1).detach())
        h = x
        for k in range(len(params) - 1):
            h = layers[2 * k + 1](layers[2 * k](h)).detach()
            t.testing.assert_close(layer_inputs[k + 1], h, msg=f"layer_inputs[{k + 1}] should be tanh of layer {k}'s output")
    print("All tests in `test_mlp_forward` passed!")


def test_backprop_factors(backprop_factors: Callable, per_example_grads: Callable, mlp_forward: Callable, init_mlp: Callable):
    x, y = _small_problem(6, n=5, d=6)
    for sizes in [[6, 1], [6, 4, 1], [6, 4, 3, 1]]:
        params = init_mlp(sizes, seed=2)
        factors = backprop_factors(params, x, y)
        assert len(factors) == len(params), "There should be one (a, d) pair per layer"
        for k, ((a, d), (W, _)) in enumerate(zip(factors, params)):
            assert a.shape == (5, W.shape[0]) and d.shape == (5, W.shape[1]), (
                f"Layer {k}: expected a of shape {(5, W.shape[0])} and d of shape {(5, W.shape[1])}, got {tuple(a.shape)}, {tuple(d.shape)}"
            )
        t.testing.assert_close(
            factors[-1][1].squeeze(-1), t.sigmoid(mlp_forward(params, x)[0]) - y,
            msg="The last layer's output gradient should be sigmoid(logit) - y (not divided by the batch size)",
        )
        t.testing.assert_close(
            per_example_grads(factors), _autograd_per_example_grads(params, x, y),
            msg=f"Per-example gradients don't match autograd for sizes {sizes}",
        )
    print("All tests in `test_backprop_factors` passed!")


def test_ghost_dot_products(ghost_dot_products: Callable, backprop_factors: Callable, init_mlp: Callable):
    x, y = _small_problem(7, n=5, d=6)
    x2, y2 = _small_problem(8, n=3, d=6)
    for sizes in [[6, 1], [6, 4, 1], [6, 4, 3, 1]]:
        params = init_mlp(sizes, seed=3)
        result = ghost_dot_products(backprop_factors(params, x, y), backprop_factors(params, x2, y2))
        assert result.shape == (5, 3), f"Expected shape (5, 3), got {tuple(result.shape)}"
        expected = _autograd_per_example_grads(params, x, y) @ _autograd_per_example_grads(params, x2, y2).T
        t.testing.assert_close(result, expected, msg=f"Ghost dot products don't match autograd for sizes {sizes}")
    # An all-zero input still has a bias gradient
    a, d = t.zeros(2, 3, dtype=DTYPE), t.tensor([[2.0], [-3.0]], dtype=DTYPE)
    t.testing.assert_close(ghost_dot_products([(a, d)], [(a, d)]), d @ d.T, msg="Did you forget the +1 for the bias?")
    print("All tests in `test_ghost_dot_products` passed!")


def test_sgd_step(sgd_step: Callable, backprop_factors: Callable, init_mlp: Callable):
    x, y = _small_problem(9, n=4, d=6)
    params = init_mlp([6, 4, 1], seed=4)
    before = [(W.clone(), b.clone()) for W, b in params]
    factors = backprop_factors(params, x, y)
    for weights in [None, t.tensor([1.0, 0.0, 1.0, 0.0], dtype=DTYPE)]:
        result = sgd_step(params, factors, 0.3, 4, weights)
        expected = _autograd_weighted_step(params, x, y, 0.3, 4, t.ones(4, dtype=DTYPE) if weights is None else weights)
        t.testing.assert_close(_flat(result), _flat(expected), msg=f"Wrong update with weights={weights}")
    unchanged = sgd_step(params, factors, 0.3, 4, t.zeros(4, dtype=DTYPE))
    t.testing.assert_close(_flat(unchanged), _flat(before), msg="All-zero weights should leave the parameters unchanged")
    t.testing.assert_close(_flat(params), _flat(before), msg="sgd_step modified its input parameters in place")
    print("All tests in `test_sgd_step` passed!")


def test_sgd_train(sgd_train: Callable, init_mlp: Callable):
    x, y = _small_problem(10, n=6, d=6)
    params = init_mlp([6, 3, 1], seed=5)
    schedule = [t.tensor([0, 1, 2]), t.tensor([3, 4, 5]), t.tensor([5, 0, 2]), t.tensor([1, 3, 4])]
    keep = t.tensor([1.0, 0.0, 1.0, 1.0, 0.0, 1.0], dtype=DTYPE)
    for k in [None, keep]:
        expected = params
        for batch in schedule:
            w = t.ones(len(batch), dtype=DTYPE) if k is None else k[batch]
            expected = _autograd_weighted_step(expected, x[batch], y[batch], 0.2, len(batch), w)
        t.testing.assert_close(_flat(sgd_train(params, x, y, schedule, 0.2, keep=k)), _flat(expected),
                               msg=f"Training doesn't match autograd SGD (keep={k})")
    unchanged = sgd_train(params, x, y, schedule, 0.2, keep=t.zeros(6, dtype=DTYPE))
    t.testing.assert_close(_flat(unchanged), _flat(params), msg="Switching every example off should leave the parameters unchanged")
    print("All tests in `test_sgd_train` passed!")


# ---------------------------------------------------------------------------------------------------------------
# Section 4
# ---------------------------------------------------------------------------------------------------------------


def test_local_game_values(local_game_values: Callable, backprop_factors: Callable, init_mlp: Callable):
    x, y = _small_problem(11, n=3, d=6)
    vx, vy = _small_problem(12, n=7, d=6)
    params = init_mlp([6, 4, 1], seed=6)
    before = _flat(params).clone()
    values = local_game_values(params, backprop_factors(params, x, y), vx, vy, 0.5, 3)
    assert values.shape == (8,), f"Expected 2**3 = 8 values, got shape {tuple(values.shape)}"
    loss_before = _val_loss(params, vx, vy)
    for mask in range(8):
        weights = t.tensor([float((mask >> i) & 1) for i in range(3)], dtype=DTYPE)
        expected = loss_before - _val_loss(_autograd_weighted_step(params, x, y, 0.5, 3, weights), vx, vy)
        assert abs(values[mask].item() - expected) < 1e-10, f"Wrong value for bitmask {mask:03b}: expected {expected:.6f}, got {values[mask]:.6f}"
    t.testing.assert_close(_flat(params), before, msg="local_game_values modified the parameters")
    print("All tests in `test_local_game_values` passed!")


def test_first_order_credit(first_order_credit: Callable):
    dots = t.tensor([[1.0, 3.0], [-2.0, 0.0], [0.5, 0.5]], dtype=DTYPE)
    t.testing.assert_close(first_order_credit(dots, 0.6, 3), 0.6 / 3 * t.tensor([2.0, -1.0, 0.5], dtype=DTYPE))
    print("All tests in `test_first_order_credit` passed!")


def test_ghost_vector_products(ghost_vector_products: Callable, backprop_factors: Callable, init_mlp: Callable):
    x, y = _small_problem(13, n=5, d=6)
    params = init_mlp([6, 4, 1], seed=7)
    gen = t.Generator().manual_seed(0)
    vector = [(t.randn(*W.shape, generator=gen, dtype=DTYPE), t.randn(*b.shape, generator=gen, dtype=DTYPE)) for W, b in params]
    expected = _autograd_per_example_grads(params, x, y) @ _flat(vector)
    t.testing.assert_close(ghost_vector_products(backprop_factors(params, x, y), vector), expected)
    print("All tests in `test_ghost_vector_products` passed!")


def test_second_order_credit(second_order_credit: Callable, ghost_vector_products: Callable):
    # A one-layer network whose per-example gradients we can write down, and a QUADRATIC validation loss
    # L(theta) = 0.5 (theta - target)^T H (theta - target), for which the second-order expansion is exact.
    gen = t.Generator().manual_seed(1)
    B, d_in = 4, 3
    a, d = t.randn(B, d_in, generator=gen, dtype=DTYPE), t.randn(B, 1, generator=gen, dtype=DTYPE)
    grads = t.cat([(a[:, :, None] * d[:, None, :]).flatten(1), d], dim=1)  # (B, d_in + 1)
    M = t.randn(d_in + 1, d_in + 1, generator=gen, dtype=DTYPE)
    H = M @ M.T + 0.5 * t.eye(d_in + 1, dtype=DTYPE)
    theta, target = t.randn(d_in + 1, generator=gen, dtype=DTYPE), t.randn(d_in + 1, generator=gen, dtype=DTYPE)
    params = [(theta[:d_in, None].clone(), theta[d_in:].clone())]
    loss = lambda th: 0.5 * (th - target) @ H @ (th - target)
    g_val = H @ (theta - target)
    lr, denominator = 0.7, B

    def hvp_fn(params, val_x, val_y, vector):
        out = H @ t.cat([vector[0][0].flatten(), vector[0][1]])
        return [(out[:d_in, None], out[d_in:])]

    def game(S):
        step = lr / denominator * grads[list(S)].sum(0) if S else t.zeros(d_in + 1, dtype=DTYPE)
        return (loss(theta) - loss(theta - step)).item()

    first = lr / denominator * grads @ g_val
    result = second_order_credit(params, [(a, d)], None, None, lr, denominator, first, hvp_fn=hvp_fn)
    t.testing.assert_close(result, _permutation_shapley(game, B), msg="For a quadratic loss, second-order credit should be exactly the Shapley value")
    print("All tests in `test_second_order_credit` passed!")


def test_run_in_run_shapley(run_in_run_shapley: Callable, sgd_train: Callable, init_mlp: Callable):
    x, y = _small_problem(14, n=6, d=5)
    vx, vy = _small_problem(15, n=4, d=5)
    params = init_mlp([5, 3, 1], seed=8)
    schedule = [t.tensor([0, 1, 2]), t.tensor([3, 4, 5]), t.tensor([4, 2, 0]), t.tensor([1, 5, 3])]
    lr = 0.4
    result = run_in_run_shapley(params, x, y, vx, vy, schedule, lr, exact_local=True)
    T, n = len(schedule), len(x)
    assert result["losses"].shape == (T + 1,), f"losses should have shape (T + 1,) = ({T + 1},)"
    for key in ["first", "second", "exact"]:
        assert result[key].shape == (n,) and result[f"{key}_per_step"].shape == (T, n), f"Wrong shape for {key!r}"
        t.testing.assert_close(result[key], result[f"{key}_per_step"].sum(0), msg=f"{key!r} should be the sum over steps of {key}_per_step")
        for step, batch in enumerate(schedule):
            outside = t.ones(n, dtype=t.bool)
            outside[batch] = False
            assert result[f"{key}_per_step"][step, outside].abs().max() == 0, "Examples not in a step's batch should get zero credit there"
    t.testing.assert_close(_flat(result["params"]), _flat(sgd_train(params, x, y, schedule, lr)),
                           msg="The accountant should train exactly like sgd_train")
    losses = result["losses"]
    t.testing.assert_close(losses[0].item(), _val_loss(params, vx, vy))
    t.testing.assert_close(result["exact_per_step"].sum(1), losses[:-1] - losses[1:],
                           msg="Exact local credit of each step should add up to that step's improvement in validation loss")
    # First-order credit at step 0 against autograd
    g_val = _autograd_per_example_grads(params, vx, vy).mean(0)
    expected = lr / 3 * _autograd_per_example_grads(params, x[schedule[0]], y[schedule[0]]) @ g_val
    t.testing.assert_close(result["first_per_step"][0, schedule[0]], expected, msg="Wrong first-order credit at step 0")
    no_exact = run_in_run_shapley(params, x, y, vx, vy, schedule, lr)
    assert no_exact["exact"] is None and no_exact["exact_per_step"] is None, "With exact_local=False, 'exact' should be None"
    t.testing.assert_close(no_exact["second"], result["second"])
    print("All tests in `test_run_in_run_shapley` passed!")


# ---------------------------------------------------------------------------------------------------------------
# Section 5
# ---------------------------------------------------------------------------------------------------------------


def test_mc_from_prefix_utilities(mc_from_prefix_utilities: Callable):
    n, M = 5, 30
    game = _random_game(n, 6)
    gen = t.Generator().manual_seed(3)
    orderings = t.stack([t.randperm(n, generator=gen) for _ in range(M)])
    prefix = t.tensor([[game(frozenset(o[:k].tolist())) for k in range(n + 1)] for o in orderings], dtype=DTYPE)
    v_full = game(frozenset(range(n)))
    for tolerance in [0.0, 0.5, 1.5]:
        expected, expected_trainings = t.zeros(n, dtype=DTYPE), 0
        for o in orderings.tolist():  # sequential reference
            coalition, previous = frozenset(), 0.0
            for i in o:
                if tolerance > 0 and abs(v_full - previous) < tolerance:
                    break
                coalition = coalition | {i}
                current = game(coalition)
                expected_trainings += 1
                expected[i] += current - previous
                previous = current
        values, n_trainings = mc_from_prefix_utilities(orderings, prefix, tolerance)
        t.testing.assert_close(values.to(DTYPE), expected / M, msg=f"Wrong values at tolerance {tolerance}")
        assert n_trainings == expected_trainings, f"At tolerance {tolerance}, expected {expected_trainings} trainings, got {n_trainings}"
    print("All tests in `test_mc_from_prefix_utilities` passed!")


def test_spearman_correlation(spearman_correlation: Callable):
    from scipy.stats import spearmanr

    gen = t.Generator().manual_seed(4)
    x = t.randn(50, generator=gen, dtype=DTYPE)
    y = x + 2 * t.randn(50, generator=gen, dtype=DTYPE)
    assert abs(spearman_correlation(x, y) - spearmanr(x.numpy(), y.numpy()).statistic) < 1e-10
    assert abs(spearman_correlation(x, x.exp()) - 1.0) < 1e-12, "A monotone transformation should have correlation 1"
    assert abs(spearman_correlation(x, -x) + 1.0) < 1e-12
    print("All tests in `test_spearman_correlation` passed!")


def test_removal_keep_masks(removal_keep_masks: Callable):
    scores = t.tensor([0.3, -1.0, 2.0, 0.1, -0.5], dtype=DTYPE)
    lowest = removal_keep_masks(scores, [0, 1, 3])
    expected = t.tensor([[1, 1, 1, 1, 1], [1, 0, 1, 1, 1], [1, 0, 1, 0, 0]], dtype=DTYPE)
    t.testing.assert_close(lowest.to(DTYPE), expected)
    highest = removal_keep_masks(scores, [2], lowest=False)
    t.testing.assert_close(highest.to(DTYPE), t.tensor([[0, 1, 0, 1, 1]], dtype=DTYPE))
    print("All tests in `test_removal_keep_masks` passed!")
