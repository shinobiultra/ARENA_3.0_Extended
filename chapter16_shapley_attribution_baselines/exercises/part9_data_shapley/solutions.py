# %%


import itertools
import math
import sys
import time
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as t
import torch.nn.functional as F
from IPython.display import display
from jaxtyping import Float, Int
from torch import Tensor
from tqdm.auto import tqdm

# Everything in this section is small enough to run on a CPU. We use float64 so that the identities we check
# (Shapley efficiency, ghost dot products vs autograd, telescoping sums) hold to ~12 decimal places, and a single
# thread because our tensors are tiny (multi-threading only adds overhead, especially on a shared machine).
t.set_default_dtype(t.float64)
t.set_num_threads(1)

# Make sure exercises are in the path
chapter = "chapter16_shapley_attribution_baselines"
section = "part9_data_shapley"
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part9_data_shapley.tests as tests
import part9_data_shapley.utils as utils

Coalition = frozenset[int]  # a set of training-example indices
Params = list[tuple[Tensor, Tensor]]  # one (weight, bias) pair per layer

MAIN = __name__ == "__main__"

# %%

TOY_X = t.tensor([[2.0, 1.0], [2.0, 1.0], [1.5, 1.5], [2.0, -0.5], [-2.0, -1.0], [-1.5, -1.5], [2.0, 1.0], [0.2, 2.5]])
TOY_Y = t.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0])
TOY_NAMES = [
    "duplicate A", "duplicate A (copy)", "positive, diagonal", "positive, low",
    "negative", "negative, diagonal", "MISLABELLED copy of A", "rare positive",
]
TOY_VAL_X = t.tensor([[2.0, 1.0], [1.5, 1.2], [2.0, -0.5], [1.0, 2.0], [-2.0, -1.0], [-1.5, -1.2], [-2.0, 0.5], [-1.0, -2.0]])
TOY_VAL_Y = t.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])

# The learning algorithm: zero initialisation, 40 steps of full-batch gradient descent with learning rate 0.2
TOY_STEPS, TOY_LR = 40, 0.2

# %%

if MAIN:
    utils.plot_toy_problem(TOY_X, TOY_Y, TOY_VAL_X, TOY_VAL_Y, TOY_NAMES)

# %%

def train_on_subset(
    x: Float[Tensor, "n d"], y: Float[Tensor, " n"], coalition: Coalition, steps: int = TOY_STEPS, lr: float = TOY_LR
) -> tuple[Float[Tensor, " d"], Float[Tensor, ""]]:
    """
    Trains logistic regression from zero init with full-batch GD on the examples in `coalition`, using the
    fixed denominator n = len(x). Returns (w, b).
    """
    n = len(x)
    mask = t.zeros(n)
    mask[list(coalition)] = 1.0
    w, b = t.zeros(x.shape[1]), t.zeros(())
    for _ in range(steps):
        errors = (t.sigmoid(x @ w + b) - y) * mask
        w = w - lr * x.T @ errors / n
        b = b - lr * errors.sum() / n
    return w, b


def subset_utility(
    coalition: Coalition,
    x: Float[Tensor, "n d"],
    y: Float[Tensor, " n"],
    val_x: Float[Tensor, "m d"],
    val_y: Float[Tensor, " m"],
    steps: int = TOY_STEPS,
    lr: float = TOY_LR,
) -> float:
    """Returns v(S): validation loss of the untrained model minus validation loss after training on S."""
    w, b = train_on_subset(x, y, coalition, steps, lr)
    loss_before = F.binary_cross_entropy_with_logits(t.zeros(len(val_y)), val_y)
    loss_after = F.binary_cross_entropy_with_logits(val_x @ w + b, val_y)
    return (loss_before - loss_after).item()


if MAIN:
    tests.test_train_on_subset(train_on_subset)
    tests.test_subset_utility(subset_utility)

# %%

def toy_v(coalition: Coalition, y: Tensor = TOY_Y) -> float:
    """The toy game: train on `coalition` of the 8 toy points, return the validation loss improvement."""
    return subset_utility(coalition, TOY_X, y, TOY_VAL_X, TOY_VAL_Y)

# %%

if MAIN:
    for coalition in [(), (0,), (6,), (0, 6), (0, 1), (0, 1, 6), tuple(range(8))]:
        print(f"v({set(coalition) or '{}'}) = {toy_v(frozenset(coalition)):+.4f}")

# %%

def exact_data_shapley(utility_fn: Callable[[Coalition], float], n: int) -> tuple[Float[Tensor, " n"], dict[Coalition, float]]:
    """
    Computes exact Shapley values by evaluating `utility_fn` on all 2^n coalitions.
    Returns (values, table) where table maps every coalition to its utility.
    """
    table = {
        frozenset(c): utility_fn(frozenset(c)) for size in range(n + 1) for c in itertools.combinations(range(n), size)
    }
    values = t.zeros(n)
    for i in range(n):
        others = [j for j in range(n) if j != i]
        for size in range(n):
            weight = 1 / (n * math.comb(n - 1, size))
            for c in itertools.combinations(others, size):
                coalition = frozenset(c)
                values[i] += weight * (table[coalition | {i}] - table[coalition])
    return values, table


if MAIN:
    tests.test_exact_data_shapley(exact_data_shapley)

# %%

if MAIN:
    t0 = time.time()
    toy_exact, toy_table = exact_data_shapley(toy_v, 8)
    full = frozenset(range(8))
    print(f"Trained {len(toy_table)} models in {time.time() - t0:.1f}s\n")
    for i, name in enumerate(TOY_NAMES):
        print(f"{i}  {name:<24} {toy_exact[i]:+.4f}")
    print(f"\nsum of values = {toy_exact.sum():.4f},   v(full dataset) = {toy_table[full]:.4f}")

# %%

def leave_one_out(utility_fn: Callable[[Coalition], float], n: int) -> Float[Tensor, " n"]:
    """Returns v(D) - v(D minus i) for every i."""
    everyone = frozenset(range(n))
    v_full = utility_fn(everyone)
    return t.tensor([v_full - utility_fn(everyone - {i}) for i in range(n)])


if MAIN:
    tests.test_leave_one_out(leave_one_out)

# %%

if MAIN:
    toy_loo = leave_one_out(lambda S: toy_table[S], 8)
    
    swapped_y = TOY_Y[[6, 1, 2, 3, 4, 5, 0, 7]]  # swap the labels of points 0 and 6
    toy_exact_swapped, _ = exact_data_shapley(lambda S: toy_v(S, swapped_y), 8)
    
    utils.plot_value_bars(
        {
            "exact Data Shapley": toy_exact,
            "leave-one-out": toy_loo,
            "exact Data Shapley, labels of 0 and 6 swapped (control)": toy_exact_swapped,
        },
        [f"{i}: {name}" for i, name in enumerate(TOY_NAMES)],
        "Toy problem: exact Data Shapley (256 trainings) vs leave-one-out (9 trainings)",
        colors=[utils.COLORS["data_shapley"], utils.COLORS["loo"], utils.COLORS["control"]],
        highlight=[6],
    )
    print("LOO:", [round(v, 4) for v in toy_loo.tolist()])

# %%

if MAIN:
    try_x, try_y = TOY_X.clone(), TOY_Y.clone()
    try_y[7] = 0.0  # flip the label of the rare positive
    
    try_values, try_table = exact_data_shapley(lambda S: subset_utility(S, try_x, try_y, TOY_VAL_X, TOY_VAL_Y), 8)
    try_loo = leave_one_out(lambda S: try_table[S], 8)
    display(
        pd.DataFrame(
            {"original Data Shapley": toy_exact, "new Data Shapley": try_values, "new LOO": try_loo},
            index=[f"{i}: {name}" for i, name in enumerate(TOY_NAMES)],
        ).round(4)
    )

# %%

class CountingUtility:
    """Wraps a utility function and counts how many times it's called (i.e. how many models would be trained)."""

    def __init__(self, utility_fn: Callable[[Coalition], float]):
        self.utility_fn = utility_fn
        self.calls = 0

    def __call__(self, coalition: Coalition) -> float:
        self.calls += 1
        return self.utility_fn(coalition)

# %%

def mc_data_shapley(utility_fn: Callable[[Coalition], float], n: int, n_permutations: int, seed: int = 0) -> Float[Tensor, " n"]:
    """Monte Carlo (permutation sampling) estimate of the Shapley values of `utility_fn`."""
    gen = t.Generator().manual_seed(seed)
    totals = t.zeros(n)
    v_empty = utility_fn(frozenset())
    for _ in range(n_permutations):
        coalition, previous = frozenset(), v_empty
        for i in t.randperm(n, generator=gen).tolist():
            coalition = coalition | {i}
            current = utility_fn(coalition)
            totals[i] += current - previous
            previous = current
    return totals / n_permutations


if MAIN:
    tests.test_mc_data_shapley(mc_data_shapley)

# %%

def tmc_data_shapley(
    utility_fn: Callable[[Coalition], float], n: int, n_permutations: int, tolerance: float, seed: int = 0
) -> Float[Tensor, " n"]:
    """Truncated Monte Carlo estimate of the Shapley values of `utility_fn`."""
    gen = t.Generator().manual_seed(seed)
    totals = t.zeros(n)
    v_empty = utility_fn(frozenset())
    v_full = utility_fn(frozenset(range(n)))
    for _ in range(n_permutations):
        coalition, previous = frozenset(), v_empty
        for i in t.randperm(n, generator=gen).tolist():
            if abs(v_full - previous) < tolerance:
                break  # every remaining example in this ordering gets zero
            coalition = coalition | {i}
            current = utility_fn(coalition)
            totals[i] += current - previous
            previous = current
    return totals / n_permutations


if MAIN:
    tests.test_tmc_data_shapley(tmc_data_shapley, mc_data_shapley)

# %%

if MAIN:
    BUDGETS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    N_SEEDS = 20
    toy_lookup = lambda S: toy_table[S]
    
    estimators = {
        "Monte Carlo": lambda u, M, seed: mc_data_shapley(u, 8, M, seed),
        "truncated MC, tolerance 0.01": lambda u, M, seed: tmc_data_shapley(u, 8, M, 0.01, seed),
        "truncated MC, tolerance 0.05": lambda u, M, seed: tmc_data_shapley(u, 8, M, 0.05, seed),
    }
    mc_curves, mc_rows = {}, []
    for name, estimator in estimators.items():
        calls, errors = np.zeros((len(BUDGETS), N_SEEDS)), np.zeros((len(BUDGETS), N_SEEDS))
        for b, budget in enumerate(BUDGETS):
            for seed in range(N_SEEDS):
                counted = CountingUtility(toy_lookup)
                estimate = estimator(counted, budget, seed)
                calls[b, seed], errors[b, seed] = counted.calls, (estimate - toy_exact).abs().max()
            mc_rows.append({"estimator": name, "orderings": budget, "mean trainings": calls[b].mean(), "mean max error": errors[b].mean()})
        mc_curves[name] = (calls.mean(1), errors.mean(1), np.quantile(errors, 0.1, axis=1), np.quantile(errors, 0.9, axis=1))
    
    loo_error = (toy_loo - toy_exact).abs().max().item()
    utils.plot_mc_convergence(
        mc_curves, exact_cost=256, loo_point=(9, loo_error),
        title="Toy problem: Monte Carlo Data Shapley error vs cost (mean and 10-90% range over 20 seeds)",
    )
    mc_table = pd.DataFrame(mc_rows).pivot(index="orderings", columns="estimator", values=["mean trainings", "mean max error"])
    display(mc_table.loc[[4, 16, 32, 64, 256]].round(3))

# %%

if MAIN:
    splits = utils.load_digit_splits(seed=0)
    train_x, clean_train_y = splits["train"]["x"], splits["train"]["y"]
    val_x, val_y = splits["val"]["x"], splits["val"]["y"]
    test_x, test_y = splits["test"]["x"], splits["test"]["y"]
    N_TRAIN = len(train_x)
    
    # Flip 6 labels of each class, chosen at random
    flip_gen = t.Generator().manual_seed(0)
    flipped = t.zeros(N_TRAIN, dtype=t.bool)
    for label in [0.0, 1.0]:
        candidates = (clean_train_y == label).nonzero().flatten()
        flipped[candidates[t.randperm(len(candidates), generator=flip_gen)[:6]]] = True
    train_y = t.where(flipped, 1 - clean_train_y, clean_train_y)
    
    print(f"train {tuple(train_x.shape)}, val {tuple(val_x.shape)}, test {tuple(test_x.shape)}")
    print(f"flipped training labels: {flipped.nonzero().flatten().tolist()}")

# %%

if MAIN:
    shown = t.cat([flipped.nonzero().flatten(), (~flipped).nonzero().flatten()[t.randperm(116, generator=t.Generator().manual_seed(1))[:12]]])
    utils.plot_digit_gallery(
        splits["train"]["images"][shown],
        [f"#{i}: labelled '{1 + 6 * int(train_y[i])}'" for i in shown.tolist()],
        flipped[shown].tolist(),
        "Training images and the labels we train on (red border = flipped label)",
        ncols=12,
    )

# %%

def init_mlp(sizes: list[int], seed: int) -> Params:
    """Gaussian init with variance 1/d_in for the weights, zeros for the biases."""
    gen = t.Generator().manual_seed(seed)
    return [
        (t.randn(d_in, d_out, generator=gen) / math.sqrt(d_in), t.zeros(d_out)) for d_in, d_out in zip(sizes[:-1], sizes[1:])
    ]

# %%

def mlp_forward(params: Params, x: Float[Tensor, "batch d_in"]) -> tuple[Float[Tensor, " batch"], list[Tensor]]:
    """Returns (logits, layer_inputs), where layer_inputs[k] is the input to layer k."""
    layer_inputs = []
    h = x
    for k, (W, b) in enumerate(params):
        layer_inputs.append(h)
        h = h @ W + b
        if k < len(params) - 1:
            h = t.tanh(h)
    return h.squeeze(-1), layer_inputs


if MAIN:
    tests.test_mlp_forward(mlp_forward, init_mlp)

# %%

def mlp_loss(params: Params, x: Float[Tensor, "batch d_in"], y: Float[Tensor, " batch"]) -> Float[Tensor, ""]:
    """Mean binary cross-entropy of the MLP on (x, y)."""
    return F.binary_cross_entropy_with_logits(mlp_forward(params, x)[0], y)

# %%

def backprop_factors(params: Params, x: Float[Tensor, "batch d_in"], y: Float[Tensor, " batch"]) -> list[tuple[Tensor, Tensor]]:
    """
    Returns [(a_0, d_0), ..., (a_{L-1}, d_{L-1})]: each layer's input and the gradient of each example's own
    (unreduced) BCE loss with respect to that layer's pre-activation.
    """
    logits, layer_inputs = mlp_forward(params, x)
    d = (t.sigmoid(logits) - y)[:, None]
    factors = []
    for k in reversed(range(len(params))):
        factors.append((layer_inputs[k], d))
        if k > 0:
            d = (d @ params[k][0].T) * (1 - layer_inputs[k] ** 2)
    return factors[::-1]

# %%

def per_example_grads(factors: list[tuple[Tensor, Tensor]]) -> Float[Tensor, "batch n_params"]:
    """Materialises every example's full gradient vector from the factors."""
    blocks = []
    for a, d in factors:
        blocks.append((a[:, :, None] * d[:, None, :]).flatten(1))  # per-example outer products a_i^T d_i
        blocks.append(d)
    return t.cat(blocks, dim=1)


if MAIN:
    tests.test_backprop_factors(backprop_factors, per_example_grads, mlp_forward, init_mlp)

# %%

def ghost_dot_products(
    factors_1: list[tuple[Tensor, Tensor]], factors_2: list[tuple[Tensor, Tensor]]
) -> Float[Tensor, "batch_1 batch_2"]:
    """Returns the matrix of dot products g_i . g_j between per-example gradients, without materialising them."""
    if len(factors_1) != len(factors_2):
        raise ValueError("Gradient factors must describe the same number of layers.")
    total = 0.0
    for (a_1, d_1), (a_2, d_2) in zip(factors_1, factors_2):
        total = total + (a_1 @ a_2.T + 1) * (d_1 @ d_2.T)
    return total


if MAIN:
    tests.test_ghost_dot_products(ghost_dot_products, backprop_factors, init_mlp)

# %%

def sgd_step(
    params: Params, factors: list[tuple[Tensor, Tensor]], lr: float, denominator: int, weights: Tensor | None = None
) -> Params:
    """One SGD step on sum_i weights_i * loss_i / denominator, computed from the gradient factors."""
    new_params = []
    for (W, b), (a, d) in zip(params, factors):
        if weights is not None:
            d = d * weights[:, None]
        new_params.append((W - lr * a.T @ d / denominator, b - lr * d.sum(0) / denominator))
    return new_params


def sgd_train(
    params: Params, x: Tensor, y: Tensor, schedule: list[Tensor], lr: float, keep: Tensor | None = None
) -> Params:
    """Runs SGD over the minibatch schedule. Examples with keep[i] == 0 never contribute to an update."""
    for batch in schedule:
        factors = backprop_factors(params, x[batch], y[batch])
        params = sgd_step(params, factors, lr, len(batch), None if keep is None else keep[batch])
    return params


if MAIN:
    tests.test_sgd_step(sgd_step, backprop_factors, init_mlp)
    tests.test_sgd_train(sgd_train, init_mlp)

# %%

SIZES = [64, 16, 1]
LR, BATCH_SIZE, EPOCHS = 0.1, 4, 4
SEEDS = [0, 1, 2]


def make_schedule(n: int, batch_size: int, epochs: int, seed: int) -> list[Int[Tensor, " batch"]]:
    """A fixed list of minibatches: each epoch is a fresh random permutation of range(n), cut into batches."""
    gen = t.Generator().manual_seed(seed)
    return [batch for _ in range(epochs) for batch in t.randperm(n, generator=gen).reshape(-1, batch_size)]


@t.no_grad()
def evaluate(params: Params, x: Tensor, y: Tensor) -> tuple[float, float]:
    """Returns (mean BCE loss, accuracy)."""
    logits = mlp_forward(params, x)[0]
    return F.binary_cross_entropy_with_logits(logits, y).item(), ((logits > 0).double() == y).double().mean().item()

# %%

if MAIN:
    params0 = init_mlp(SIZES, seed=0)
    schedule0 = make_schedule(N_TRAIN, BATCH_SIZE, EPOCHS, seed=0)
    trained = sgd_train(params0, train_x, train_y, schedule0, LR)
    trained_without_flips = sgd_train(params0, train_x, train_y, schedule0, LR, keep=(~flipped).double())
    
    for name, p in [("all 128 images", trained), ("flipped images switched off", trained_without_flips)]:
        (val_loss, val_acc), (test_loss, test_acc) = evaluate(p, val_x, val_y), evaluate(p, test_x, test_y)
        print(f"{name:<28}  val loss {val_loss:.4f} (acc {val_acc:.1%})   test loss {test_loss:.4f} (acc {test_acc:.1%})")

# %%

if MAIN:
    train_digit = splits["train"]["digit"]
    row_order = sorted(range(N_TRAIN), key=lambda i: (train_digit[i].item(), flipped[i].item()))
    col_order = val_y.argsort(stable=True)
    group_sizes = [((train_digit == d) & (flipped == f)).sum().item() for d in [1, 7] for f in [False, True]]
    row_boundaries = np.cumsum(group_sizes)[:-1].tolist()
    
    similarity = {}
    for name, p in [("at initialisation", params0), ("after training (step 128)", trained)]:
        sims = ghost_dot_products(backprop_factors(p, train_x, train_y), backprop_factors(p, val_x, val_y))
        similarity[name] = sims[row_order][:, col_order]
    
    utils.plot_similarity_heatmaps(
        similarity,
        row_boundaries=row_boundaries,
        row_names=["1s, label '1'", "1s, FLIPPED to '7'", "7s, label '7'", "7s, FLIPPED to '1'"],
        col_boundaries=[24],
        col_names=["validation 1s", "validation 7s"],
        suptitle="Per-example gradient dot products between training and validation images (blue = training on it helps)",
    )
    for name, sims in similarity.items():
        rows = sims.mean(1)
        is_flipped = flipped[row_order]
        print(f"{name:<26} mean row: clean {rows[~is_flipped].mean():+.3f}, flipped {rows[is_flipped].mean():+.3f}")

# %%

if MAIN:
    timing_params, timing = [], {"materialise per-example gradients": [], "ghost dot product": []}
    for width in [16, 64, 256, 1024, 2048]:
        p = init_mlp([64, width, 1], seed=0)
        factors_train, factors_val = backprop_factors(p, train_x[:32], train_y[:32]), backprop_factors(p, val_x, val_y)
        timing_params.append(sum(W.numel() + b.numel() for W, b in p))
        for name, fn in [
            ("materialise per-example gradients", lambda: per_example_grads(factors_train) @ per_example_grads(factors_val).T),
            ("ghost dot product", lambda: ghost_dot_products(factors_train, factors_val)),
        ]:
            times = []
            for _ in range(5):
                start = time.perf_counter()
                result = fn()
                times.append(time.perf_counter() - start)
            timing[name].append(min(times))
        t.testing.assert_close(per_example_grads(factors_train) @ per_example_grads(factors_val).T, ghost_dot_products(factors_train, factors_val))
    
    utils.plot_timing(timing_params, timing, "Cost of a 32 x 48 matrix of gradient dot products (CPU, float64)")
    for n_p, slow, fast in zip(timing_params, *timing.values()):
        print(f"{n_p:>8} params: materialised {slow * 1e3:7.2f} ms, ghost {fast * 1e3:6.3f} ms  ({slow / fast:5.0f}x)")

# %%

def local_game_values(
    params: Params, factors: list[tuple[Tensor, Tensor]], val_x: Tensor, val_y: Tensor, lr: float, denominator: int
) -> Float[Tensor, " two_to_the_B"]:
    """U_t(S) for every coalition S of the minibatch, indexed by bitmask."""
    batch_size = factors[0][0].shape[0]
    loss_before = mlp_loss(params, val_x, val_y)
    values = []
    for mask in range(2**batch_size):
        weights = t.tensor([float((mask >> i) & 1) for i in range(batch_size)])
        values.append(loss_before - mlp_loss(sgd_step(params, factors, lr, denominator, weights), val_x, val_y))
    return t.stack(values)


if MAIN:
    tests.test_local_game_values(local_game_values, backprop_factors, init_mlp)

# %%

def exact_local_shapley(values: Float[Tensor, " two_to_the_B"]) -> Float[Tensor, " B"]:
    """Exact Shapley values of a bitmask-indexed game, using exact_data_shapley from section 1."""
    n_coalitions = values.numel()
    if values.ndim != 1 or n_coalitions < 2 or n_coalitions & (n_coalitions - 1):
        raise ValueError("Expected a one-dimensional complete power-of-two coalition table.")
    batch_size = values.numel().bit_length() - 1
    shapley, _ = exact_data_shapley(lambda S: values[sum(1 << i for i in S)].item(), batch_size)
    return shapley

# %%

def first_order_credit(train_val_dots: Float[Tensor, "B n_val"], lr: float, denominator: int) -> Float[Tensor, " B"]:
    """First-order In-Run credit (lr / denominator) * g_i . g_val, where g_val is the mean validation gradient."""
    return lr / denominator * train_val_dots.mean(dim=1)


if MAIN:
    tests.test_first_order_credit(first_order_credit)

# %%

def val_hvp(params: Params, val_x: Tensor, val_y: Tensor, vector: Params) -> Params:
    """Returns H_val @ vector, where H_val is the Hessian of the mean validation loss at `params`."""
    leaves = [p.detach().clone().requires_grad_(True) for pair in params for p in pair]
    pairs = [(leaves[2 * k], leaves[2 * k + 1]) for k in range(len(params))]
    grads = t.autograd.grad(mlp_loss(pairs, val_x, val_y), leaves, create_graph=True)
    grad_dot_vector = sum((g * v).sum() for g, v in zip(grads, [v for pair in vector for v in pair]))
    hvp = t.autograd.grad(grad_dot_vector, leaves)
    return [(hvp[2 * k].detach(), hvp[2 * k + 1].detach()) for k in range(len(params))]

# %%

def ghost_vector_products(factors: list[tuple[Tensor, Tensor]], vector: Params) -> Float[Tensor, " B"]:
    """Returns g_i . vector for every example i, without materialising g_i."""
    total = 0.0
    for (a, d), (U, u) in zip(factors, vector):
        total = total + ((a @ U) * d).sum(dim=-1) + d @ u
    return total


if MAIN:
    tests.test_ghost_vector_products(ghost_vector_products, backprop_factors, init_mlp)

# %%

def second_order_credit(
    params: Params,
    factors: list[tuple[Tensor, Tensor]],
    val_x: Tensor,
    val_y: Tensor,
    lr: float,
    denominator: int,
    first_order: Float[Tensor, " B"],
    hvp_fn: Callable = val_hvp,
) -> Float[Tensor, " B"]:
    """Second-order In-Run credit: first order minus (lr/denominator)^2 / 2 * g_i^T H_val sum_j g_j."""
    batch_gradient = [(a.T @ d, d.sum(0)) for a, d in factors]
    hvp = hvp_fn(params, val_x, val_y, batch_gradient)
    return first_order - 0.5 * (lr / denominator) ** 2 * ghost_vector_products(factors, hvp)


if MAIN:
    tests.test_second_order_credit(second_order_credit, ghost_vector_products)

# %%

if MAIN:
    batch = schedule0[0]
    factors = backprop_factors(params0, train_x[batch], train_y[batch])
    val_factors = backprop_factors(params0, val_x, val_y)
    rows = []
    for lr in [LR, 10 * LR]:
        exact_local = exact_local_shapley(local_game_values(params0, factors, val_x, val_y, lr, BATCH_SIZE))
        first = first_order_credit(ghost_dot_products(factors, val_factors), lr, BATCH_SIZE)
        second = second_order_credit(params0, factors, val_x, val_y, lr, BATCH_SIZE, first)
        for name, values in [("exact local", exact_local), ("first order", first), ("second order", second)]:
            rows.append({"lr": lr, "credit": name, **{f"#{i}": v for i, v in zip(batch.tolist(), values.tolist())}, "sum": values.sum().item()})
    display(pd.DataFrame(rows).set_index(["lr", "credit"]).round(4))

# %%

def run_in_run_shapley(
    params: Params,
    x: Tensor,
    y: Tensor,
    val_x: Tensor,
    val_y: Tensor,
    schedule: list[Tensor],
    lr: float,
    exact_local: bool = False,
) -> dict:
    """Trains with SGD over `schedule`, accumulating first-order, second-order (and optionally exact local) credit."""
    n, n_steps = len(x), len(schedule)
    per_step = {key: t.zeros(n_steps, n) for key in ["first", "second", "exact"]}
    losses = [mlp_loss(params, val_x, val_y).item()]
    for step, batch in enumerate(schedule):
        factors = backprop_factors(params, x[batch], y[batch])
        val_factors = backprop_factors(params, val_x, val_y)
        first = first_order_credit(ghost_dot_products(factors, val_factors), lr, len(batch))
        second = second_order_credit(params, factors, val_x, val_y, lr, len(batch), first)
        per_step["first"][step].index_add_(0, batch, first)
        per_step["second"][step].index_add_(0, batch, second)
        if exact_local:
            exact = exact_local_shapley(local_game_values(params, factors, val_x, val_y, lr, len(batch)))
            per_step["exact"][step].index_add_(0, batch, exact)
        params = sgd_step(params, factors, lr, len(batch))
        losses.append(mlp_loss(params, val_x, val_y).item())
    result = {"params": params, "losses": t.tensor(losses)}
    for key in ["first", "second", "exact"]:
        has_values = key != "exact" or exact_local
        result[key] = per_step[key].sum(0) if has_values else None
        result[f"{key}_per_step"] = per_step[key] if has_values else None
    return result


if MAIN:
    tests.test_run_in_run_shapley(run_in_run_shapley, sgd_train, init_mlp)

# %%

if MAIN:
    toy_params0 = [(t.zeros(2, 1), t.zeros(1))]
    toy_schedule = [t.arange(8)] * TOY_STEPS
    toy_run = run_in_run_shapley(toy_params0, TOY_X, TOY_Y, TOY_VAL_X, TOY_VAL_Y, toy_schedule, TOY_LR, exact_local=True)
    
    w_ref, b_ref = train_on_subset(TOY_X, TOY_Y, frozenset(range(8)))
    t.testing.assert_close(toy_run["params"][0][0].squeeze(-1), w_ref)
    t.testing.assert_close(toy_run["params"][0][1].squeeze(-1), b_ref)
    
    print(f"actual improvement L(theta_0) - L(theta_T) = {toy_run['losses'][0] - toy_run['losses'][-1]:.4f}  (= v(D) = {toy_table[full]:.4f})")
    for name, key in [("exact local", "exact"), ("first order", "first"), ("second order", "second")]:
        values = toy_run[key]
        r = np.corrcoef(values, toy_exact)[0, 1]
        print(f"In-Run, {name:<13} total {values.sum():.4f}   correlation with exact Data Shapley {r:.3f}")
    display(
        pd.DataFrame(
            {"exact Data Shapley": toy_exact, "In-Run, exact local": toy_run["exact"], "In-Run, first order": toy_run["first"],
             "In-Run, second order": toy_run["second"], "leave-one-out": toy_loo},
            index=[f"{i}: {name}" for i, name in enumerate(TOY_NAMES)],
        ).round(4)
    )
    
    utils.plot_value_bars(
        {
            "exact Data Shapley (256 trainings)": toy_exact,
            "In-Run, exact local games (1 run)": toy_run["exact"],
            "In-Run, first order (1 run)": toy_run["first"],
            "In-Run, second order (1 run)": toy_run["second"],
            "leave-one-out (9 trainings)": toy_loo,
        },
        [f"{i}: {name}" for i, name in enumerate(TOY_NAMES)],
        "Toy problem: In-Run Data Shapley vs exact Data Shapley",
        colors=[utils.COLORS[k] for k in ["data_shapley", "exact_local", "first", "second", "loo"]],
        highlight=[6],
    )

# %%

if MAIN:
    in_run, in_run_shuffled = {}, {}
    t0 = time.time()
    for seed in tqdm(SEEDS):
        p0, schedule = init_mlp(SIZES, seed), make_schedule(N_TRAIN, BATCH_SIZE, EPOCHS, seed)
        in_run[seed] = run_in_run_shapley(p0, train_x, train_y, val_x, val_y, schedule, LR, exact_local=True)
        shuffled_val_y = val_y[t.randperm(len(val_y), generator=t.Generator().manual_seed(100 + seed))]
        in_run_shuffled[seed] = run_in_run_shapley(p0, train_x, train_y, val_x, shuffled_val_y, schedule, LR)
    print(f"6 accounted training runs in {time.time() - t0:.1f}s")
    
    run = in_run[0]
    t.testing.assert_close(run["params"][0][0], trained[0][0])  # the accountant didn't change the training run
    actual = run["losses"][0] - run["losses"][-1]
    print(f"\nseed 0: validation loss {run['losses'][0]:.4f} -> {run['losses'][-1]:.4f}, actual improvement {actual:.4f}")
    for name in ["exact", "first", "second"]:
        print(f"  total credit, {name:<7}: {run[name].sum():.4f}  ({run[name].sum() / actual:.2f}x the actual improvement)")
    control = in_run_shuffled[0]
    print(f"control: loss on the shuffled-label validation set goes {control['losses'][0]:.4f} -> {control['losses'][-1]:.4f}")

# %%

if MAIN:
    actual_curve = (run["losses"][0] - run["losses"]).numpy()
    utils.plot_accounting(
        {
            "actual improvement in validation loss": actual_curve,
            "In-Run, exact local games": np.concatenate([[0], run["exact_per_step"].sum(1).cumsum(0).numpy()]),
            "In-Run, first order": np.concatenate([[0], run["first_per_step"].sum(1).cumsum(0).numpy()]),
            "In-Run, second order": np.concatenate([[0], run["second_per_step"].sum(1).cumsum(0).numpy()]),
        },
        {"In-Run, first order": (run["exact"], run["first"]), "In-Run, second order": (run["exact"], run["second"])},
        flipped,
        "Seed 0: accounting for one SGD run on the digits (128 steps, minibatches of 4)",
    )
    for name in ["first", "second"]:
        print(f"correlation of {name}-order with exact local credit: {np.corrcoef(run[name], run['exact'])[0, 1]:.4f}")

# %%

if MAIN:
    utils.plot_credit_trajectories(
        run["second_per_step"].cumsum(0), flipped,
        "Seed 0: cumulative second-order In-Run credit of every training image",
    )
    epoch_credit = run["second_per_step"].reshape(EPOCHS, -1, N_TRAIN).sum(1)  # (epoch, example)
    for epoch in range(EPOCHS):
        print(f"epoch {epoch}: total credit to clean examples {epoch_credit[epoch, ~flipped].sum():+.4f}, to flipped {epoch_credit[epoch, flipped].sum():+.4f}")

# %%

def count_flipped_in_lowest(scores: Tensor, k: int = 12) -> int:
    """How many of the k lowest-scoring training examples have flipped labels?"""
    return flipped[scores.argsort()[:k]].sum().item()

# %%

if MAIN:
    detection = pd.DataFrame(
        {
            seed: {
                "In-Run, exact local": count_flipped_in_lowest(in_run[seed]["exact"]),
                "In-Run, first order": count_flipped_in_lowest(in_run[seed]["first"]),
                "In-Run, second order": count_flipped_in_lowest(in_run[seed]["second"]),
                "control: second order, shuffled validation labels": count_flipped_in_lowest(in_run_shuffled[seed]["second"]),
                "control: random ranking (expected)": 12 * 12 / 128,
            }
            for seed in SEEDS
        }
    ).rename(columns=lambda s: f"seed {s}")
    display(detection.round(1))

# %%

if MAIN:
    TRY_LRS = [0.003, 0.01, 0.03, 0.1, 0.3]
    ratios = {"In-Run, first order": [], "In-Run, second order": [], "In-Run, exact local games": []}
    for lr in TRY_LRS:
        result = run_in_run_shapley(params0, train_x, train_y, val_x, val_y, schedule0, lr, exact_local=True)
        improvement = (result["losses"][0] - result["losses"][-1]).item()
        for name, key in [("In-Run, first order", "first"), ("In-Run, second order", "second"), ("In-Run, exact local games", "exact")]:
            ratios[name].append(result[key].sum().item() / improvement)
        print(f"lr {lr:<6} improvement {improvement:.4f}  first/actual {ratios['In-Run, first order'][-1]:.3f}  second/actual {ratios['In-Run, second order'][-1]:.3f}")
    utils.plot_lr_balance(TRY_LRS, ratios, "Seed 0: total credit allocated / actual improvement, by learning rate")

# %%

@t.no_grad()
def train_many(params: Params, x: Tensor, y: Tensor, schedule: list[Tensor], lr: float, keep: Float[Tensor, "K n"]) -> Params:
    """
    Trains K copies of the MLP at once, model k using only the examples with keep[k, i] == 1 (fixed denominator).
    Equivalent to [sgd_train(params, x, y, schedule, lr, keep[k]) for k in range(K)], but much faster.
    Returns params with a leading K dimension: W of shape (K, d_in, d_out), b of shape (K, d_out).
    """
    K = keep.shape[0]
    params = [(W.expand(K, *W.shape).clone(), b.expand(K, *b.shape).clone()) for W, b in params]
    for batch in schedule:
        xb, yb, batch_size = x[batch], y[batch], len(batch)
        layer_inputs, h = [], xb.expand(K, *xb.shape)
        for k, (W, b) in enumerate(params):  # forward pass: h has shape (K, batch, d)
            layer_inputs.append(h)
            h = h @ W + b[:, None, :]
            if k < len(params) - 1:
                h = t.tanh(h)
        d = (t.sigmoid(h) - yb[:, None]) * keep[:, batch, None]  # output gradient, masked per model
        new_params = [None] * len(params)
        for k in reversed(range(len(params))):  # backward pass + SGD update
            W, b = params[k]
            a = layer_inputs[k]
            new_params[k] = (W - lr * a.mT @ d / batch_size, b - lr * d.sum(1) / batch_size)
            if k > 0:
                d = (d @ W.mT) * (1 - a**2)
        params = new_params
    return params


@t.no_grad()
def many_losses(params: Params, x: Tensor, y: Tensor) -> Float[Tensor, " K"]:
    """Mean BCE loss of each of the K models in a batch of models."""
    h = x.expand(params[0][0].shape[0], *x.shape)
    for k, (W, b) in enumerate(params):
        h = h @ W + b[:, None, :]
        if k < len(params) - 1:
            h = t.tanh(h)
    return F.binary_cross_entropy_with_logits(h.squeeze(-1), y.expand(h.shape[0], -1), reduction="none").mean(-1)

# %%

if MAIN:
    check_keep = (t.rand(3, N_TRAIN, generator=t.Generator().manual_seed(0)) > 0.3).double()
    many = train_many(params0, train_x, train_y, schedule0, LR, check_keep)
    for k in range(3):
        single = sgd_train(params0, train_x, train_y, schedule0, LR, keep=check_keep[k])
        for (W_many, b_many), (W, b) in zip(many, single):
            t.testing.assert_close(W_many[k], W)
            t.testing.assert_close(b_many[k], b)
    print("train_many agrees with sgd_train")

# %%

def mc_from_prefix_utilities(
    orderings: Int[Tensor, "M n"], prefix_utilities: Float[Tensor, "M n_plus_1"], tolerance: float = 0.0
) -> tuple[Float[Tensor, " n"], int]:
    """Returns the (truncated) Monte Carlo Shapley estimate and the number of trainings it needed."""
    M, n = orderings.shape
    marginals = prefix_utilities[:, 1:] - prefix_utilities[:, :-1]  # (M, n), by position in the ordering
    if tolerance > 0:
        v_full = prefix_utilities[:, -1:]
        close = (v_full - prefix_utilities[:, :-1]).abs() < tolerance  # truncate before adding position j?
        truncated = close.cumsum(dim=1) > 0
        marginals = marginals.masked_fill(truncated, 0.0)
        n_trainings = int((~truncated).sum())
    else:
        n_trainings = M * n
    totals = t.zeros(n).index_add_(0, orderings.flatten(), marginals.flatten())
    return totals / M, n_trainings


if MAIN:
    tests.test_mc_from_prefix_utilities(mc_from_prefix_utilities)

# %%

if MAIN:
    N_ORDERINGS = 200
    mc_results, loo_results = {}, {}
    t0 = time.time()
    for seed in SEEDS:
        p0, schedule = init_mlp(SIZES, seed), make_schedule(N_TRAIN, BATCH_SIZE, EPOCHS, seed)
        loss_untrained = mlp_loss(p0, val_x, val_y)
        gen = t.Generator().manual_seed(1000 + seed)
        orderings = t.stack([t.randperm(N_TRAIN, generator=gen) for _ in range(N_ORDERINGS)])
        prefix_utilities = []
        for ordering in tqdm(orderings, desc=f"seed {seed}"):
            # keep[k, i] = 1 if example i is among the first k of the ordering
            position = t.empty(N_TRAIN, dtype=t.long)
            position[ordering] = t.arange(N_TRAIN)
            keep = (position[None, :] < t.arange(N_TRAIN + 1)[:, None]).double()
            prefix_utilities.append(loss_untrained - many_losses(train_many(p0, train_x, train_y, schedule, LR, keep), val_x, val_y))
        prefix_utilities = t.stack(prefix_utilities)
        mc_results[seed] = dict(orderings=orderings, prefix_utilities=prefix_utilities)
        mc_results[seed]["values"], _ = mc_from_prefix_utilities(orderings, prefix_utilities)
        # Leave-one-out: the full dataset, and the full dataset minus each example
        keep = t.ones(N_TRAIN + 1, N_TRAIN)
        keep[t.arange(1, N_TRAIN + 1), t.arange(N_TRAIN)] = 0.0
        utilities = loss_untrained - many_losses(train_many(p0, train_x, train_y, schedule, LR, keep), val_x, val_y)
        loo_results[seed] = utilities[0] - utilities[1:]
    print(f"{3 * N_ORDERINGS * N_TRAIN:,} + {3 * (N_TRAIN + 1)} trainings in {time.time() - t0:.0f}s")
    
    mc0 = mc_results[0]
    v_full = mc0["prefix_utilities"][0, -1].item()
    print(f"\nseed 0: v(D) = {v_full:.4f} (the In-Run accountant's total was {in_run[0]['exact'].sum():.4f}),  sum of MC values = {mc0['values'].sum():.4f}")
    half_1, _ = mc_from_prefix_utilities(mc0["orderings"][:100], mc0["prefix_utilities"][:100])
    half_2, _ = mc_from_prefix_utilities(mc0["orderings"][100:], mc0["prefix_utilities"][100:])
    print(f"split-half correlation of the MC estimate (100 vs 100 orderings): {np.corrcoef(half_1, half_2)[0, 1]:.3f}")
    for tolerance in [0.01, 0.02, 0.05]:
        values, n_trainings = mc_from_prefix_utilities(mc0["orderings"], mc0["prefix_utilities"], tolerance)
        print(f"truncated MC, tolerance {tolerance}: {1 - n_trainings / (N_ORDERINGS * N_TRAIN):.0%} of trainings saved, "
              f"max |change| in values {(values - mc0['values']).abs().max():.4f} (values range {mc0['values'].min():.4f} to {mc0['values'].max():.4f})")

# %%

def spearman_correlation(x: Float[Tensor, " n"], y: Float[Tensor, " n"]) -> float:
    """Spearman rank correlation (no ties)."""
    rank_x, rank_y = x.argsort().argsort().double(), y.argsort().argsort().double()
    rank_x, rank_y = rank_x - rank_x.mean(), rank_y - rank_y.mean()
    return ((rank_x @ rank_y) / (rank_x.norm() * rank_y.norm())).item()


if MAIN:
    tests.test_spearman_correlation(spearman_correlation)

# %%

if MAIN:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, (name, scores) in zip(
        axes,
        [("In-Run, first order", in_run[0]["first"]), ("In-Run, second order", in_run[0]["second"]), ("leave-one-out", loo_results[0])],
    ):
        rho = spearman_correlation(mc_results[0]["values"], scores)
        utils.plot_method_scatter(
            mc_results[0]["values"], scores, flipped, "Monte Carlo Data Shapley (25,600 trainings)", name,
            f"{name} vs Monte Carlo Data Shapley (Spearman {rho:.2f})", ax=ax,
        )
    fig.suptitle("Seed 0: every training image, scored by an expensive and a cheap method", fontsize=11)
    plt.tight_layout()
    plt.show()
    
    comparison = pd.DataFrame(
        {
            seed: {
                **{f"Spearman with MC Data Shapley: {name}": spearman_correlation(mc_results[seed]["values"], scores)
                   for name, scores in [("In-Run 1st order", in_run[seed]["first"]), ("In-Run 2nd order", in_run[seed]["second"]),
                                        ("LOO", loo_results[seed]), ("shuffled-label control", in_run_shuffled[seed]["second"])]},
                **{f"flipped in lowest 12: {name}": count_flipped_in_lowest(scores)
                   for name, scores in [("MC Data Shapley", mc_results[seed]["values"]), ("LOO", loo_results[seed]),
                                        ("In-Run 2nd order", in_run[seed]["second"]), ("In-Run 1st order", in_run[seed]["first"])]},
                # The same comparison restricted to the 116 clean-label examples, plus the MC estimate's own noise ceiling
                **{f"clean only, Spearman with MC: {name}": spearman_correlation(mc_results[seed]["values"][~flipped], scores[~flipped])
                   for name, scores in [("In-Run 1st order", in_run[seed]["first"]), ("In-Run 2nd order", in_run[seed]["second"]),
                                        ("LOO", loo_results[seed])]},
                "clean only, MC split-half (100 vs 100 orderings)": spearman_correlation(
                    *[mc_from_prefix_utilities(mc_results[seed]["orderings"][half], mc_results[seed]["prefix_utilities"][half])[0][~flipped]
                      for half in [slice(0, 100), slice(100, 200)]]
                ),
            }
            for seed in SEEDS
        }
    ).rename(columns=lambda s: f"seed {s}")
    with pd.option_context("display.max_colwidth", 60, "display.width", 200):
        display(comparison.round(2))

# %%

def removal_keep_masks(scores: Float[Tensor, " n"], ks: list[int], lowest: bool = True) -> Float[Tensor, "n_ks n"]:
    """Keep masks that remove the k lowest-scoring (or highest-scoring) examples, for each k in ks."""
    order = scores.argsort(descending=not lowest)
    keep = t.ones(len(ks), len(scores))
    for row, k in enumerate(ks):
        keep[row, order[:k]] = 0.0
    return keep


if MAIN:
    tests.test_removal_keep_masks(removal_keep_masks)

# %%

if MAIN:
    KS = [0, 4, 8, 12, 16, 20, 24, 32]
    N_RANDOM = 10
    methods = {
        "In-Run, first order": lambda seed: in_run[seed]["first"],
        "In-Run, second order": lambda seed: in_run[seed]["second"],
        "Monte Carlo Data Shapley": lambda seed: mc_results[seed]["values"],
        "leave-one-out": lambda seed: loo_results[seed],
        "In-Run, shuffled validation labels (control)": lambda seed: in_run_shuffled[seed]["second"],
    }
    deletion, oracle = {}, {}
    for seed in SEEDS:
        p0, schedule = init_mlp(SIZES, seed), make_schedule(N_TRAIN, BATCH_SIZE, EPOCHS, seed)
        names, masks = [], []
        for name, get_scores in methods.items():
            for which in ["lowest", "highest"]:
                names.append(f"{name}|{which}")
                masks.append(removal_keep_masks(get_scores(seed), KS, lowest=(which == "lowest")))
        gen = t.Generator().manual_seed(2000 + seed)
        for draw in range(N_RANDOM):
            names.append(f"random|{draw}")
            masks.append(removal_keep_masks(t.rand(N_TRAIN, generator=gen), KS))
        names.append("oracle")
        masks.append((~flipped).double()[None])
        test_losses = many_losses(train_many(p0, train_x, train_y, schedule, LR, t.cat(masks)), test_x, test_y)
        full_loss = test_losses[0].item()  # row 0 is k = 0: nothing removed
        changes = (test_losses - full_loss).split([len(m) for m in masks])
        deletion[seed] = {name: change.numpy() for name, change in zip(names, changes) if not name.startswith("random")}
        deletion[seed]["random"] = np.stack([change.numpy() for name, change in zip(names, changes) if name.startswith("random")])
        oracle[seed] = deletion[seed].pop("oracle")[0]
    
    utils.plot_deletion_curves(
        KS, deletion, oracle, "Delete the k lowest- or highest-credit training images, retrain, measure test loss (3 seeds)"
    )

# %%

if MAIN:
    k_index = KS.index(12)
    rows = {}
    for seed in SEEDS:
        column = {f"remove lowest: {name}": deletion[seed][f"{name}|lowest"][k_index] for name in methods}
        column.update({f"remove highest: {name}": deletion[seed][f"{name}|highest"][k_index] for name in methods})
        random_12 = deletion[seed]["random"][:, k_index]
        column.update({"remove 12 random: mean": random_12.mean(), "remove 12 random: min": random_12.min(), "remove 12 random: max": random_12.max()})
        column["remove the 12 flipped (oracle)"] = oracle[seed]
        rows[f"seed {seed}"] = column
    display(pd.DataFrame(rows).round(4))

# %%

if MAIN:
    scores = in_run[0]["second"]
    order = scores.argsort()
    shown = t.cat([order[:8], order.flip(0)[:8]])
    utils.plot_digit_gallery(
        splits["train"]["images"][shown],
        [f"#{i} '{1 + 6 * int(train_y[i])}'  {scores[i]:+.4f}" for i in shown.tolist()],
        flipped[shown].tolist(),
        "Seed 0, second-order In-Run credit: 8 lowest (top row) and 8 highest (bottom row)",
        ncols=8,
    )

# %%

if MAIN:
    def timed(fn: Callable) -> float:
        start = time.perf_counter()
        fn()
        return time.perf_counter() - start
    
    
    p0 = init_mlp(SIZES, 0)
    one_ordering_keep = (t.arange(N_TRAIN)[None, :] < t.arange(N_TRAIN + 1)[:, None]).double()
    costs = pd.DataFrame(
        [
            {"method": "plain SGD training run", "models trained": 1,
             "seconds": timed(lambda: sgd_train(p0, train_x, train_y, schedule0, LR))},
            {"method": "In-Run, first + second order", "models trained": 1,
             "seconds": timed(lambda: run_in_run_shapley(p0, train_x, train_y, val_x, val_y, schedule0, LR))},
            {"method": "In-Run, + exact local games", "models trained": 1,
             "seconds": timed(lambda: run_in_run_shapley(p0, train_x, train_y, val_x, val_y, schedule0, LR, exact_local=True))},
            {"method": "leave-one-out", "models trained": N_TRAIN + 1,
             "seconds": timed(lambda: train_many(p0, train_x, train_y, schedule0, LR, t.ones(N_TRAIN + 1, N_TRAIN)))},
            {"method": f"Monte Carlo Data Shapley ({N_ORDERINGS} orderings)", "models trained": N_ORDERINGS * N_TRAIN,
             "seconds": N_ORDERINGS * timed(lambda: train_many(p0, train_x, train_y, schedule0, LR, one_ordering_keep))},
        ]
    ).set_index("method")
    display(costs.round(2))

# %%

if MAIN:
    seed0_scores = {"In-Run 1st": in_run[0]["first"], "In-Run 2nd": in_run[0]["second"], "MC Data Shapley": mc_results[0]["values"]}
    TRY_EXAMPLES = [in_run[0]["second"].argmin().item(), in_run[0]["second"].argmax().item(), flipped.nonzero()[0].item()]
    
    p0 = init_mlp(SIZES, 0)
    full_val_loss = mlp_loss(sgd_train(p0, train_x, train_y, schedule0, LR), val_x, val_y).item()
    for i in TRY_EXAMPLES:
        keep = t.ones(N_TRAIN)
        keep[i] = 0.0
        without_i = mlp_loss(sgd_train(p0, train_x, train_y, schedule0, LR, keep=keep), val_x, val_y).item()
        predictions = ", ".join(f"{name} {s[i]:+.4f}" for name, s in seed0_scores.items())
        print(f"#{i:<3} (flipped={bool(flipped[i])!s:<5}) actual effect of keeping it (LOO) {without_i - full_val_loss:+.4f} | {predictions}")

# %%
