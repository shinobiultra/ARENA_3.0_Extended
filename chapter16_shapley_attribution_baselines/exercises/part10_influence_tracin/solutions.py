# %%


MAIN = __name__ == "__main__"

# %%

from pathlib import Path
from collections.abc import Callable
import math
import os
import sys

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
import matplotlib.pyplot as plt
from IPython import get_ipython

if MAIN:
    if get_ipython() is not None:
        get_ipython().run_line_magic("matplotlib", "inline")

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from chapter16_shapley_attribution_baselines.exercises.part10_influence_tracin import tests
from chapter16_shapley_attribution_baselines.exercises.part9_influence_and_tracin.attribution_data import (
    load_study_data,
    data_manifest,
)

DTYPE = torch.float64
if MAIN:
    torch.set_num_threads(4)
if MAIN:
    OUTPUT = Path(
        os.environ.get("ARENA_ATTRIBUTION_OUTPUT_DIRECTORY", "/tmp/arena16_9_influence_tracin")
    )
if MAIN:
    OUTPUT.mkdir(parents=True, exist_ok=True)
if MAIN:
    plt.rcParams.update(
        {
            "figure.figsize": (10, 4),
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 11,
        }
    )


def show_figure(figure, name, **layout):
    figure.tight_layout(**layout)
    figure.savefig(OUTPUT / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.show()

# %%

if MAIN:
    data = load_study_data()
import json

if MAIN:
    manifest_path = (
        ROOT
        / "chapter16_shapley_attribution_baselines/exercises/part9_influence_and_tracin/data_manifest.json"
    )
if MAIN:
    assert data_manifest(data) == json.loads(manifest_path.read_text())
if MAIN:
    train = data["train"]
if MAIN:
    fig, axes = plt.subplots(1, 4, figsize=(10, 3))
if MAIN:
    for ax, (digit, flipped) in zip(axes, [(1, False), (1, True), (7, False), (7, True)]):
        index = (
            torch.nonzero((train["digit"] == digit) & (train["flipped"] == flipped)).flatten()[0].item()
        )
        ax.imshow(train["images"][index], cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
        observed = 7 if train["corrupted_y"][index] else 1
        ax.set_title(f"Source digit: {digit}\nTraining label: {observed}", fontsize=11)
        ax.set_xlabel(train["ids"][index], fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
if MAIN:
    show_figure(fig, "attribution_training_images")

# %%

def logistic_losses(theta: Tensor, x: Tensor, y: Tensor, l2: float = 0.0) -> Tensor:
    """Unreduced per-example losses, optionally including R(theta) in each row."""
    if theta.ndim != 1 or x.ndim != 2 or x.shape[1] != theta.numel() or y.shape != (len(x),):
        raise ValueError("Expected theta [parameters], x [examples, parameters], y [examples].")
    if not ((y == 0) | (y == 1)).all() or not math.isfinite(l2) or l2 < 0:
        raise ValueError("Use binary labels and a finite nonnegative regularizer.")
    logits = x @ theta
    return F.softplus(logits) - y * logits + 0.5 * l2 * theta.square().sum()


def logistic_gradients(theta: Tensor, x: Tensor, y: Tensor, l2: float = 0.0) -> Tensor:
    """Per-example parameter gradients; their mean is the objective gradient."""
    logistic_losses(theta, x, y, l2)  # Validate the common public contract.
    residual = torch.sigmoid(x @ theta) - y
    return residual[:, None] * x + l2 * theta[None, :]


if MAIN:
    tests.test_logistic_losses(logistic_losses)
if MAIN:
    tests.test_logistic_gradients(logistic_gradients)

# %%

def weighted_objective(theta: Tensor, x: Tensor, y: Tensor, weights: Tensor, l2: float) -> Tensor:
    """One independent normalized objective per row of theta [fits, parameters]."""
    if theta.ndim != 2 or x.ndim != 2 or theta.shape[1] != x.shape[1]:
        raise ValueError("Expected theta [fits, parameters] and x [examples, parameters].")
    if weights.shape != (len(theta), len(x)) or y.shape != (len(x),):
        raise ValueError("Expected weights [fits, examples] and y [examples].")
    if not torch.isfinite(weights).all() or (weights < 0).any() or (weights.sum(1) <= 0).any():
        raise ValueError("Every fit needs finite nonnegative weights with positive total mass.")
    if not ((y == 0) | (y == 1)).all() or not math.isfinite(l2) or l2 < 0:
        raise ValueError("Use binary targets and a finite nonnegative regularizer.")
    probabilities = weights / weights.sum(1, keepdim=True)
    logits = theta @ x.T
    return ((F.softplus(logits) - y[None, :] * logits) * probabilities).sum(
        1
    ) + 0.5 * l2 * theta.square().sum(1)


def objective_derivatives(
    theta: Tensor, x: Tensor, y: Tensor, weights: Tensor, l2: float
) -> tuple[Tensor, Tensor]:
    """Gradient and dense Hessian of each independently weighted objective."""
    weighted_objective(theta, x, y, weights, l2)
    mass = weights / weights.sum(1, keepdim=True)
    probabilities = torch.sigmoid(theta @ x.T)
    gradient = ((probabilities - y[None, :]) * mass) @ x + l2 * theta
    curvature = probabilities * (1 - probabilities) * mass
    hessian = torch.einsum("np,bn,nq->bpq", x, curvature, x)
    hessian = hessian + l2 * torch.eye(x.shape[1], dtype=x.dtype, device=x.device)
    return gradient, hessian


if MAIN:
    tests.test_weighted_objective(weighted_objective)
if MAIN:
    tests.test_objective_derivatives(objective_derivatives)

# %%

def logistic_hvp(
    theta: Tensor, x: Tensor, y: Tensor, weights: Tensor, vector: Tensor, l2: float
) -> Tensor:
    """An objective Hessian-vector product without materializing a Hessian."""
    if theta.ndim != 1 or vector.shape != theta.shape or weights.shape != (len(x),):
        raise ValueError(
            "Use one parameter vector, one matching direction and one example-weight vector."
        )
    weighted_objective(theta[None], x, y, weights[None], l2)
    probabilities = torch.sigmoid(x @ theta)
    curvature = weights / weights.sum() * probabilities * (1 - probabilities)
    return x.T @ (curvature * (x @ vector)) + l2 * vector


if MAIN:
    tests.test_logistic_hvp(logistic_hvp)

# %%

def conjugate_gradient(
    matvec: Callable[[Tensor], Tensor],
    rhs: Tensor,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    max_steps: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Solve an SPD system, returning the actual residual norms at every step."""
    if rhs.ndim != 1 or not torch.isfinite(rhs).all():
        raise ValueError("The right-hand side must be one finite vector.")
    if not all(math.isfinite(t) and t >= 0 for t in (rtol, atol)):
        raise ValueError("Residual tolerances must be finite and nonnegative.")
    max_steps = 2 * rhs.numel() if max_steps is None else max_steps
    if max_steps < 1:
        raise ValueError("Allow at least one CG step.")
    answer = torch.zeros_like(rhs)
    residual = rhs - matvec(answer)
    direction = residual.clone()
    squared_norm = residual @ residual
    threshold = atol + rtol * torch.linalg.vector_norm(rhs)
    history = [torch.linalg.vector_norm(residual)]
    if history[-1] <= threshold:
        return answer, torch.stack(history)
    for _ in range(max_steps):
        product = matvec(direction)
        curvature = direction @ product
        if not torch.isfinite(curvature) or curvature <= 0:
            raise ValueError("CG encountered nonpositive curvature; its SPD assumption failed.")
        alpha = squared_norm / curvature
        answer = answer + alpha * direction
        residual = residual - alpha * product
        # A recurrence residual can drift: success is checked against A @ answer.
        actual_norm = torch.linalg.vector_norm(rhs - matvec(answer))
        history.append(actual_norm)
        if actual_norm <= threshold:
            return answer, torch.stack(history)
        next_squared_norm = residual @ residual
        direction = residual + (next_squared_norm / squared_norm) * direction
        squared_norm = next_squared_norm
    raise RuntimeError(
        f"CG did not converge: residual {float(history[-1]):.3g} exceeds {float(threshold):.3g}."
    )


if MAIN:
    tests.test_conjugate_gradient(conjugate_gradient)

# %%

if MAIN:
    rotation = torch.tensor([[0.8, -0.6], [0.6, 0.8]], dtype=DTYPE)
if MAIN:
    H = rotation @ torch.diag(torch.tensor([0.4, 4.0], dtype=DTYPE)) @ rotation.T
if MAIN:
    g = torch.tensor([1.0, 0.5], dtype=DTYPE)
if MAIN:
    response, residuals = conjugate_gradient(lambda v: H @ v, g)
if MAIN:
    u, v = np.meshgrid(np.linspace(-3, 3, 150), np.linspace(-3, 3, 150))
if MAIN:
    points = torch.tensor(np.stack([u, v], -1), dtype=DTYPE)
if MAIN:
    energy = 0.5 * torch.einsum("...p,pq,...q->...", points, H, points)
if MAIN:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
if MAIN:
    axes[0].contour(u, v, energy.numpy(), levels=[0.25, 0.5, 1, 2, 4, 8], colors=".7")
if MAIN:
    for vector, label, color in [
        (g, "Identity-curvature control", "#d95f02"),
        (response, "Hessian-aware response", "#1f77b4"),
    ]:
        axes[0].arrow(
            0, 0, *vector.tolist(), width=0.025, color=color, length_includes_head=True, label=label
        )
if MAIN:
    axes[0].set(
        xlabel="Parameter 1",
        ylabel="Parameter 2",
        aspect="equal",
        title="Same gradient, different geometry",
    )
if MAIN:
    axes[0].legend(fontsize=8)
if MAIN:
    axes[1].semilogy(
        range(len(residuals)), residuals.clamp_min(1e-16), "o-", label="CG actual residual"
    )
if MAIN:
    axes[1].axhline(
        float(torch.linalg.vector_norm(g)), color="#d95f02", ls="--", label="Zero-solution control"
    )
if MAIN:
    axes[1].set(xlabel="CG step", ylabel="Residual norm", xticks=range(len(residuals)))
if MAIN:
    axes[1].legend(fontsize=9)
if MAIN:
    show_figure(fig, "attribution_curvature_solve")

# %%

def fit_weighted_logistic(
    x: Tensor,
    y: Tensor,
    weights: Tensor,
    l2: float = 0.02,
    tolerance: float = 1e-10,
    max_steps: int = 60,
) -> dict[str, Tensor]:
    """Independently refit every row of weights from zero using Newton/Armijo.

    Batching shares matrix operations, not fitted parameters or loss reductions.
    Each fit keeps its own line search and convergence decision. No failed fit is
    silently accepted. Regularization stays fixed after normalization of weights.
    """
    if weights.ndim != 2 or l2 <= 0 or not math.isfinite(l2):
        raise ValueError("Use weights [fits, examples] and a strictly positive regularizer.")
    if tolerance <= 0 or not math.isfinite(tolerance) or max_steps < 1:
        raise ValueError("Use a positive residual tolerance and iteration budget.")
    theta = torch.zeros(weights.shape[0], x.shape[1], dtype=x.dtype, device=x.device)
    states, residuals = [], []
    for _ in range(max_steps + 1):
        loss = weighted_objective(theta, x, y, weights, l2)
        gradient, hessian = objective_derivatives(theta, x, y, weights, l2)
        norms = torch.linalg.vector_norm(gradient, ord=float("inf"), dim=1)
        states.append(theta.clone())
        residuals.append(norms)
        active = norms > tolerance
        if not active.any():
            return {
                "theta": theta,
                "states": torch.stack(states),
                "gradient_norms": torch.stack(residuals),
                "objective": loss,
            }
        if len(states) > max_steps:
            break
        step = torch.linalg.solve(hessian, gradient.unsqueeze(-1)).squeeze(-1)
        step[~active] = 0
        decrement = (gradient * step).sum(1)
        rates = torch.ones(len(theta), dtype=x.dtype, device=x.device)
        accepted = ~active
        proposed = theta.clone()
        for _ in range(30):
            trial = theta - rates[:, None] * step
            trial_loss = weighted_objective(trial, x, y, weights, l2)
            # Near the optimum, the predicted decrease can be below one ulp.
            # Allow objective roundoff while still requiring the final gradient
            # tolerance; otherwise a vanishing step can stall a correct refit.
            roundoff = 8 * torch.finfo(x.dtype).eps * loss.abs().clamp_min(1)
            sufficient = trial_loss <= loss - 1e-4 * rates * decrement + roundoff
            newly_accepted = sufficient & ~accepted
            proposed[newly_accepted] = trial[newly_accepted]
            accepted |= sufficient
            if accepted.all():
                break
            rates[~accepted] *= 0.5
        if not accepted.all():
            raise RuntimeError(
                "Newton line search failed; retain the failure instead of accepting an unfinished refit."
            )
        theta = proposed
    raise RuntimeError(
        f"Newton did not converge for every fit; largest gradient is {float(norms.max()):.3g}."
    )


if MAIN:
    tests.test_fit_weighted_logistic(fit_weighted_logistic)

# %%

def upweight_influence(
    hessian: Tensor, train_gradients: Tensor, query_gradients: Tensor, damping: float = 0.0
) -> Tensor:
    """d(query loss)/d epsilon for F + epsilon * one regularized training loss."""
    p = hessian.shape[0]
    if hessian.shape != (p, p) or train_gradients.ndim != 2 or query_gradients.ndim != 2:
        raise ValueError("Use a square Hessian and two example-by-parameter gradient matrices.")
    if train_gradients.shape[1] != p or query_gradients.shape[1] != p:
        raise ValueError("Every gradient must refer to the same parameter coordinates.")
    if not math.isfinite(damping) or damping < 0:
        raise ValueError("Damping must be finite and nonnegative.")
    if not torch.allclose(hessian, hessian.T, rtol=1e-10, atol=1e-12):
        raise ValueError("The objective Hessian must be symmetric.")
    system = hessian + damping * torch.eye(p, dtype=hessian.dtype, device=hessian.device)
    torch.linalg.cholesky(system)  # Fail explicitly if the local SPD premise fails.
    response = torch.linalg.solve(system, query_gradients.T)
    return -train_gradients @ response


if MAIN:
    tests.test_upweight_influence(upweight_influence)

# %%

if MAIN:
    small_x = torch.tensor(
        [[1.0, 0.0, 1.0], [0.0, 2.0, 1.0], [-1.0, 1.0, 1.0], [2.0, -1.0, 1.0], [0.3, 0.8, 1.0]],
        dtype=DTYPE,
    )
if MAIN:
    small_y = torch.tensor([1.0, 0.0, 0.0, 1.0, 1.0], dtype=DTYPE)
if MAIN:
    n = len(small_x)
if MAIN:
    l2 = 0.3
if MAIN:
    base = fit_weighted_logistic(small_x, small_y, torch.ones(1, n, dtype=DTYPE), l2)
if MAIN:
    theta = base["theta"][0]
if MAIN:
    _, hs = objective_derivatives(theta[None], small_x, small_y, torch.ones(1, n, dtype=DTYPE), l2)
if MAIN:
    qx = torch.tensor([[0.2, 1.4, 1.0]], dtype=DTYPE)
if MAIN:
    qy = torch.zeros(1, dtype=DTYPE)
if MAIN:
    g_query = logistic_gradients(theta, qx, qy)
if MAIN:
    g_full = logistic_gradients(theta, small_x, small_y, l2)
if MAIN:
    g_raw = logistic_gradients(theta, small_x, small_y)
if MAIN:
    influence = upweight_influence(hs[0], g_full, g_query)[:, 0]
if MAIN:
    wrong_normalization = upweight_influence(hs[0], g_raw, g_query)[:, 0]
if MAIN:
    strengths = torch.linspace(-1, 1, 25, dtype=DTYPE)
if MAIN:
    example = 1
if MAIN:
    weights = torch.ones(len(strengths), n, dtype=DTYPE)
if MAIN:
    weights[:, example] += strengths
if MAIN:
    refits = fit_weighted_logistic(small_x, small_y, weights, l2)
if MAIN:
    actual_loss = F.softplus(refits["theta"] @ qx.T).flatten()
if MAIN:
    base_loss = logistic_losses(theta, qx, qy)[0]
if MAIN:
    actual_change = actual_loss - base_loss
if MAIN:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
if MAIN:
    axes[0].plot(strengths, actual_change, color="black", label="Independently refitted")
if MAIN:
    axes[0].plot(
        strengths, strengths / n * influence[example], "--", label="Matched-objective derivative"
    )
if MAIN:
    axes[0].plot(
        strengths,
        strengths / n * wrong_normalization[example],
        ":",
        color="#d95f02",
        label="Data-gradient-only control",
    )
if MAIN:
    axes[0].axhline(0, color=".6", lw=0.8)
if MAIN:
    axes[0].axvline(-1, color=".6", lw=0.8)
if MAIN:
    axes[0].set(
        xlabel="Change in one row's relative weight",
        ylabel="Change in query loss",
        title="-1 removes the row",
    )
if MAIN:
    axes[0].legend(fontsize=8)
if MAIN:
    probe_strengths = torch.logspace(-3, 0, 7, dtype=DTYPE)
if MAIN:
    probe_weights = torch.ones(2 * len(probe_strengths), n, dtype=DTYPE)
if MAIN:
    probe_weights[: len(probe_strengths), example] -= probe_strengths
if MAIN:
    probe_weights[len(probe_strengths) :, example] += probe_strengths
if MAIN:
    probe_refits = fit_weighted_logistic(small_x, small_y, probe_weights, l2)
if MAIN:
    probe_losses = F.softplus(probe_refits["theta"] @ qx.T).flatten()
if MAIN:
    measured_derivative = (
        probe_losses[len(probe_strengths) :] - probe_losses[: len(probe_strengths)]
    ) / (2 * probe_strengths / n)
if MAIN:
    for label, value in [
        ("Correct derivative", influence[example]),
        ("Wrong normalization", wrong_normalization[example]),
    ]:
        axes[1].loglog(
            probe_strengths, (measured_derivative - value).abs().clamp_min(1e-12), "o-", label=label
        )
if MAIN:
    axes[1].set(
        xlabel="Magnitude of relative-weight change",
        ylabel="Error in upweight derivative",
        title="Zoom in on the local claim",
    )
if MAIN:
    axes[1].legend(fontsize=8)
if MAIN:
    show_figure(fig, "attribution_weight_perturbation")
if MAIN:
    tests.test_regularized_removal_normalization()

# %%

def sgd_trace(
    x: Tensor, y: Tensor, orders: Tensor, epoch_rates: Tensor, l2: float
) -> dict[str, Tensor]:
    """Record the actual single-example optimization path, starting at zero."""
    if orders.ndim != 2 or orders.shape[1] != len(x) or epoch_rates.shape != (len(orders),):
        raise ValueError("Provide one permutation and one learning rate for every epoch.")
    expected = torch.arange(len(x), device=orders.device).expand_as(orders)
    if not torch.equal(orders.sort(dim=1).values, expected):
        raise ValueError("Each epoch must visit every training example exactly once.")
    if not torch.isfinite(epoch_rates).all() or (epoch_rates <= 0).any():
        raise ValueError("Epoch learning rates must be finite and strictly positive.")
    used_ids = orders.flatten()
    rates = epoch_rates.repeat_interleave(len(x))
    states = torch.zeros(len(used_ids) + 1, x.shape[1], dtype=x.dtype, device=x.device)
    with torch.no_grad():
        for t, index in enumerate(used_ids.tolist()):
            gradient = logistic_gradients(
                states[t], x[index : index + 1], y[index : index + 1], l2
            )[0]
            states[t + 1] = states[t] - rates[t] * gradient
    return {"states": states, "used_ids": used_ids, "rates": rates}


if MAIN:
    tests.test_sgd_trace(sgd_trace)

# %%

def trace_loss_reductions(
    states: Tensor,
    used_ids: Tensor,
    rates: Tensor,
    train_x: Tensor,
    train_y: Tensor,
    query_x: Tensor,
    query_y: Tensor,
    l2: float,
    chunk_size: int = 128,
) -> dict[str, Tensor]:
    """Actual loss accounting and its Taylor approximation along recorded SGD.

    The exact allocation is well-defined here because each update uses one row.
    The calculation includes that update's regularizer contribution. The raw-only
    control deliberately omits it. No tensor of per-step per-query full gradients
    is needed: a logistic query gradient is its residual times its input vector.
    """
    steps = len(used_ids)
    if states.ndim != 2 or len(states) != steps + 1 or rates.shape != (steps,):
        raise ValueError("Every update needs a before/after state, training ID and learning rate.")
    if used_ids.dtype != torch.int64 or (used_ids < 0).any() or (used_ids >= len(train_x)).any():
        raise ValueError("Recorded training IDs must be valid integer row indices.")
    if chunk_size < 1 or not torch.isfinite(rates).all() or (rates <= 0).any():
        raise ValueError("Use a positive chunk size and positive finite learning rates.")
    exact_steps, first_order_steps, raw_steps = [], [], []
    for start in range(0, steps, chunk_size):
        stop = min(start + chunk_size, steps)
        before, after = states[start:stop], states[start + 1 : stop + 1]
        x = train_x[used_ids[start:stop]]
        y = train_y[used_ids[start:stop]]
        raw_gradient = (torch.sigmoid((before * x).sum(1)) - y)[:, None] * x
        gradient = raw_gradient + l2 * before
        if not torch.allclose(
            after - before, -rates[start:stop, None] * gradient, rtol=1e-9, atol=1e-12
        ):
            raise ValueError("Recorded states do not follow the stated single-example SGD update.")
        before_logits, after_logits = before @ query_x.T, after @ query_x.T
        before_loss = F.softplus(before_logits) - query_y[None, :] * before_logits
        after_loss = F.softplus(after_logits) - query_y[None, :] * after_logits
        query_residual = torch.sigmoid(before_logits) - query_y[None, :]
        exact_steps.append(before_loss - after_loss)
        first_order_steps.append(rates[start:stop, None] * query_residual * (gradient @ query_x.T))
        raw_steps.append(rates[start:stop, None] * query_residual * (raw_gradient @ query_x.T))
    exact_steps = torch.cat(exact_steps)
    first_order_steps = torch.cat(first_order_steps)
    raw_steps = torch.cat(raw_steps)
    shape = (len(train_x), len(query_x))

    def collect(values):
        return torch.zeros(shape, dtype=states.dtype, device=states.device).index_add_(
            0, used_ids, values
        )

    return {
        "exact": collect(exact_steps),
        "first_order": collect(first_order_steps),
        "raw_only": collect(raw_steps),
        "exact_steps": exact_steps,
        "first_order_steps": first_order_steps,
        "raw_only_steps": raw_steps,
    }


if MAIN:
    tests.test_trace_loss_reductions(trace_loss_reductions)

# %%

if MAIN:
    orders = torch.stack(
        [torch.randperm(n, generator=torch.Generator().manual_seed(100 + epoch)) for epoch in range(15)]
    )
if MAIN:
    epoch_rates = torch.cat([torch.full((8,), 0.12, dtype=DTYPE), torch.full((7,), 0.04, dtype=DTYPE)])
if MAIN:
    trace = sgd_trace(small_x, small_y, orders, epoch_rates, 0.3)
if MAIN:
    path_scores = trace_loss_reductions(
        trace["states"], trace["used_ids"], trace["rates"], small_x, small_y, qx, qy, 0.3
    )
if MAIN:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
if MAIN:
    for key, label, style in [
        ("exact_steps", "Observed loss reduction", "-"),
        ("first_order_steps", "First-order full update", "--"),
        ("raw_only_steps", "Omit regularizer control", ":"),
    ]:
        axes[0].plot(torch.cumsum(path_scores[key][:, 0], 0), style, label=label)
if MAIN:
    axes[0].set(xlabel="Training update", ylabel="Cumulative query-loss reduction")
if MAIN:
    axes[0].legend(fontsize=8)
if MAIN:
    pos = np.arange(n)
if MAIN:
    for offset, key, label in [
        (-0.25, "exact", "Exact path"),
        (0, "first_order", "Taylor"),
        (0.25, "raw_only", "Raw-only control"),
    ]:
        axes[1].bar(pos + offset, path_scores[key][:, 0], width=0.25, label=label)
if MAIN:
    axes[1].axhline(0, color="black", lw=0.7)
if MAIN:
    axes[1].set(xlabel="Training row", ylabel="Assigned query-loss reduction", xticks=pos)
if MAIN:
    axes[1].legend(fontsize=8)
if MAIN:
    show_figure(fig, "attribution_exact_training_path")

# %%

def tracin_checkpoint_scores(
    train_gradients: Tensor, query_gradients: Tensor, rates: Tensor
) -> Tensor:
    """Weighted gradient products; positive means first-order loss reduction.

    The input is [checkpoints, examples, parameters]. Rates are actual checkpoint
    learning-rate weights. A checkpoint subset covers only its selected visits;
    this function does not rescale scores to pretend they sum to observed loss.
    """
    if train_gradients.ndim != 3 or query_gradients.ndim != 3:
        raise ValueError("Keep checkpoint, example and parameter axes distinct.")
    k, _, p = train_gradients.shape
    if (
        query_gradients.shape[0] != k
        or query_gradients.shape[2] != p
        or rates.shape != (k,)
        or k == 0
    ):
        raise ValueError("Checkpoint counts, parameter coordinates and rate counts must match.")
    if not torch.isfinite(rates).all() or (rates < 0).any():
        raise ValueError("Use finite nonnegative learning-rate weights.")
    return torch.einsum("knp,kqp,k->nq", train_gradients, query_gradients, rates)


if MAIN:
    tests.test_tracin_checkpoint_scores(tracin_checkpoint_scores)

# %%

if MAIN:
    checkpoint_theta = trace["states"][n::n]
if MAIN:
    train_gradients = torch.stack(
        [logistic_gradients(t, small_x, small_y, 0.3) for t in checkpoint_theta]
    )
if MAIN:
    query_gradients = torch.stack([logistic_gradients(t, qx, qy) for t in checkpoint_theta])
if MAIN:
    choices = {
        "All epochs": slice(None),
        "Early epochs": slice(0, 8),
        "Late epochs": slice(8, None),
        "Last epoch": slice(-1, None),
    }
if MAIN:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
if MAIN:
    for label, selected in choices.items():
        scores = tracin_checkpoint_scores(
            train_gradients[selected], query_gradients[selected], epoch_rates[selected]
        )[:, 0]
        axes[0].plot(pos, scores, "o-", label=label)
        axes[1].scatter(path_scores["exact"][:, 0], scores, label=label)
if MAIN:
    axes[0].axhline(0, color=".5", lw=0.8)
if MAIN:
    axes[0].set(xlabel="Training row", ylabel="Checkpoint score", xticks=pos)
if MAIN:
    axes[0].legend(fontsize=8)
if MAIN:
    axes[1].axhline(0, color=".7", lw=0.8)
if MAIN:
    axes[1].axvline(0, color=".7", lw=0.8)
if MAIN:
    axes[1].set(xlabel="Exact observed path contribution", ylabel="Checkpoint score")
if MAIN:
    axes[1].legend(fontsize=8)
if MAIN:
    show_figure(fig, "attribution_checkpoint_choices")

# %%

def ghost_linear_products(
    left_inputs: Tensor,
    left_deltas: Tensor,
    right_inputs: Tensor,
    right_deltas: Tensor,
    bias: bool = True,
) -> Tensor:
    """Exact gradient products for one affine layer with one invocation per example.

    Deltas differentiate scalar losses with respect to preactivations. This is not
    a formula for tied parameters used at several token positions; those require
    all pairs of positions. Shared regularizers are also separate gradient terms.
    """
    arrays = (left_inputs, left_deltas, right_inputs, right_deltas)
    if any(a.ndim != 2 for a in arrays):
        raise ValueError("Inputs and preactivation derivatives must be two-dimensional.")
    if len(left_inputs) != len(left_deltas) or len(right_inputs) != len(right_deltas):
        raise ValueError("Every example needs its own input and preactivation derivative.")
    if (
        left_inputs.shape[1] != right_inputs.shape[1]
        or left_deltas.shape[1] != right_deltas.shape[1]
    ):
        raise ValueError("Both sides must describe the same affine layer.")
    return (left_inputs @ right_inputs.T + int(bias)) * (left_deltas @ right_deltas.T)


if MAIN:
    tests.test_ghost_linear_products(ghost_linear_products)

# %%

if MAIN:
    learning_rate_multiplier = 4.0
if MAIN:
    query_row = 2
if MAIN:
    query_label = 1.0
if MAIN:
    play_x = small_x[query_row : query_row + 1]
if MAIN:
    play_y = torch.tensor([query_label], dtype=DTYPE)
if MAIN:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
if MAIN:
    for multiplier, color in [(1.0, "#1f77b4"), (learning_rate_multiplier, "#d95f02")]:
        changed_trace = sgd_trace(small_x, small_y, orders, epoch_rates * multiplier, 0.3)
        changed_scores = trace_loss_reductions(
            changed_trace["states"],
            changed_trace["used_ids"],
            changed_trace["rates"],
            small_x,
            small_y,
            play_x,
            play_y,
            0.3,
        )
        exact = changed_scores["exact_steps"][:, 0].cumsum(0)
        linear = changed_scores["first_order_steps"][:, 0].cumsum(0)
        axes[0].plot(exact, color=color, label=f"Exact, rate × {multiplier:g}")
        axes[0].plot(linear, "--", color=color, label=f"Taylor, rate × {multiplier:g}")
        axes[1].plot((exact - linear).abs(), color=color, label=f"Rate × {multiplier:g}")
if MAIN:
    axes[0].set(xlabel="Training update", ylabel="Cumulative query-loss reduction")
if MAIN:
    axes[1].set(xlabel="Training update", ylabel="Absolute cumulative Taylor error")
if MAIN:
    for ax in axes:
        ax.legend(fontsize=8)
if MAIN:
    show_figure(fig, "attribution_step_size_play")

# %%

import json
import tempfile
from types import SimpleNamespace
from scipy.stats import rankdata

L2 = 0.02
ORDER_SEEDS = (0, 1, 2)
RANDOM_SEEDS = tuple(range(1000, 1032))
BUDGETS = (2, 4, 8, 16, 32)
DAMPING = (0.0, 0.01, 0.1, 1.0)
STRENGTHS = (-1.0, -0.5, -0.1, -0.01, 0.01, 0.1, 0.5, 1.0)
CHECKPOINT_CHOICES = {
    "all": slice(None),
    "early": slice(0, 20),
    "late": slice(20, None),
    "last": slice(-1, None),
}

# %%

def class_matched_removals(helpfulness: Tensor, labels: Tensor, budget: int) -> Tensor:
    """Select low helpfulness, with equal observed-class counts and stable ties."""
    if (
        helpfulness.ndim != 1
        or labels.shape != helpfulness.shape
        or not torch.isfinite(helpfulness).all()
    ):
        raise ValueError(
            "Provide one finite helpfulness score and observed label per training row."
        )
    if not ((labels == 0) | (labels == 1)).all() or budget < 0 or budget % 2:
        raise ValueError("Binary class matching requires an even nonnegative deletion budget.")
    selected = []
    for label in (0, 1):
        ids = torch.nonzero(labels == label).flatten()
        if len(ids) < budget // 2:
            raise ValueError("The budget exceeds the available rows in an observed class.")
        order = torch.argsort(helpfulness[ids], stable=True)
        selected.append(ids[order[: budget // 2]])
    return torch.cat(selected).sort().values


if MAIN:
    tests.test_class_matched_removals(class_matched_removals)

# %%

def query_outputs(theta, queries):
    """Actual classifier probabilities and unreduced query losses for each fit."""
    logits = theta @ queries["x"].T
    losses = F.softplus(logits) - queries["y"][None, :] * logits
    return {
        "logits": logits,
        "probabilities": torch.sigmoid(logits),
        "losses": losses,
        "correct": (logits >= 0) == queries["y"][None, :].bool(),
    }


def signed_pixel_similarity(train, train_y, queries):
    """A labelled image-retrieval control, without model loss or curvature."""
    pixels = F.normalize(train["x"][:, :-1], dim=1)
    query_pixels = F.normalize(queries["x"][:, :-1], dim=1)
    agreement = (2 * train_y[:, None] - 1) * (2 * queries["y"][None, :] - 1)
    return (pixels @ query_pixels.T) * agreement

# %%

def checkpoint_analysis(trace, train, train_y, queries, m):
    n = len(train_y)
    checkpoints = trace["states"][n::n]
    rates = trace["rates"][n - 1 :: n]
    train_gradients = torch.stack(
        [m.logistic_gradients(theta, train["x"], train_y, L2) for theta in checkpoints]
    )
    query_gradients = torch.stack(
        [m.logistic_gradients(theta, queries["x"], queries["y"]) for theta in checkpoints]
    )
    scores = {}
    for name, selected in CHECKPOINT_CHOICES.items():
        scores[name] = m.tracin_checkpoint_scores(
            train_gradients[selected], query_gradients[selected], rates[selected]
        )
    ghost_products = []
    for theta, query_gradient in zip(checkpoints, query_gradients):
        train_delta = torch.sigmoid(train["x"] @ theta) - train_y
        query_delta = torch.sigmoid(queries["x"] @ theta) - queries["y"]
        raw = m.ghost_linear_products(
            train["x"][:, :-1],
            train_delta[:, None],
            queries["x"][:, :-1],
            query_delta[:, None],
            bias=True,
        )
        # Training rows include the regularizer; query prediction loss does not.
        regularizer_product = (L2 * theta @ query_gradient.T)[None, :]
        ghost_products.append(raw + regularizer_product)
    ghost_products = torch.stack(ghost_products)
    dense_products = torch.einsum("knp,kqp->knq", train_gradients, query_gradients)
    ghost_error = float((ghost_products - dense_products).abs().max())
    if ghost_error > 1e-10:
        raise ValueError(
            f"Factored gradient products differ from complete parameter gradients: {ghost_error}."
        )
    ghost_scores = (ghost_products * rates[:, None, None]).sum(0)
    torch.testing.assert_close(ghost_scores, scores["all"], rtol=1e-10, atol=1e-11)
    return {
        "scores": scores,
        "ghost_all_scores": ghost_scores,
        "ghost_max_error": ghost_error,
        "checkpoints": checkpoints,
        "checkpoint_rates": rates,
    }

# %%

def score_queries(fits, train, train_y, queries, m):
    """Compute all measurements at already fitted parameters; never refit here."""
    n = len(train_y)
    theta = fits["base"]["theta"][0]
    unit_weights = torch.ones(1, n, dtype=theta.dtype)
    _, hessian = m.objective_derivatives(theta[None], train["x"], train_y, unit_weights, L2)
    train_gradient = m.logistic_gradients(theta, train["x"], train_y, L2)
    raw_train_gradient = m.logistic_gradients(theta, train["x"], train_y)
    query_gradient = m.logistic_gradients(theta, queries["x"], queries["y"])
    baseline = query_outputs(theta[None], queries)
    loo = query_outputs(fits["loo"]["theta"], queries)
    perturbations = query_outputs(fits["perturbations"]["theta"], queries)
    removal_predictions = torch.stack(
        [
            -m.upweight_influence(hessian[0], train_gradient, query_gradient, damping) / n
            for damping in DAMPING
        ]
    )
    wrong_normalization = -m.upweight_influence(hessian[0], raw_train_gradient, query_gradient) / n
    convergence = []
    last = len(fits["base"]["states"]) - 1
    state_indices = [0, min(1, last), min(3, last), last]
    for index in state_indices:
        current = fits["base"]["states"][index, 0]
        _, h = m.objective_derivatives(current[None], train["x"], train_y, unit_weights, L2)
        g_train = m.logistic_gradients(current, train["x"], train_y, L2)
        g_query = m.logistic_gradients(current, queries["x"], queries["y"])
        convergence.append(m.upweight_influence(h[0], g_train, g_query))
    losses = perturbations["losses"].reshape(len(fits["probe_indices"]), len(STRENGTHS), -1)
    finite_derivative = (losses[:, 4] - losses[:, 3]) / (2 * 0.01 / n)
    paths = []
    for trace in fits["paths"]:
        accounting = m.trace_loss_reductions(
            trace["states"],
            trace["used_ids"],
            trace["rates"],
            train["x"],
            train_y,
            queries["x"],
            queries["y"],
            L2,
        )
        endpoint = query_outputs(trace["states"][[0, -1]], queries)
        actual_change = endpoint["losses"][0] - endpoint["losses"][1]
        torch.testing.assert_close(
            accounting["exact"].sum(0), actual_change, rtol=1e-10, atol=1e-11
        )
        checkpoints = checkpoint_analysis(trace, train, train_y, queries, m)
        paths.append({"accounting": accounting, "checkpoints": checkpoints, "endpoints": endpoint})
    return {
        "baseline": baseline,
        "loo": loo,
        "loo_loss_changes": loo["losses"] - baseline["losses"],
        "perturbations": perturbations,
        "finite_derivative": finite_derivative,
        "removal_predictions": removal_predictions,
        "wrong_normalization": wrong_normalization,
        "convergence_upweight": torch.stack(convergence),
        "convergence_state_indices": state_indices,
        "gradient_similarity": train_gradient @ query_gradient.T,
        "signed_pixel_similarity": signed_pixel_similarity(train, train_y, queries),
        "paths": paths,
        "query_ids": queries["ids"],
    }

# %%

def fit_condition(train, ranking, probe_indices, train_y, m, directory, progress):
    n, p = train["x"].shape
    dtype = train["x"].dtype
    unit_weights = torch.ones(1, n, dtype=dtype)
    fits = {"base": m.fit_weighted_logistic(train["x"], train_y, unit_weights, L2)}
    progress("Converged full-data model")
    loo_weights = torch.ones(n, n, dtype=dtype) - torch.eye(n, dtype=dtype)
    fits["loo"] = m.fit_weighted_logistic(train["x"], train_y, loo_weights, L2)
    perturbation_weights = torch.ones(len(probe_indices) * len(STRENGTHS), n, dtype=dtype)
    for row, (index, strength) in enumerate((i, s) for i in probe_indices for s in STRENGTHS):
        perturbation_weights[row, index] += strength
    fits["perturbations"] = m.fit_weighted_logistic(train["x"], train_y, perturbation_weights, L2)
    fits["probe_indices"] = list(probe_indices)
    fits["perturbation_strengths"] = torch.tensor(STRENGTHS, dtype=dtype)
    torch.save(fits, directory / "reference_fits.pt")
    progress("Refitted all 128 leave-one-out and 96 weight-perturbation models")
    individual_errors = []
    for index in (0, 1, 2):
        individual = m.fit_weighted_logistic(
            train["x"], train_y, loo_weights[index : index + 1], L2
        )
        error = float((individual["theta"][0] - fits["loo"]["theta"][index]).abs().max())
        if error > 1e-8:
            raise ValueError(
                f"Batched and separate refits differ for deletion row {index}: {error}."
            )
        individual_errors.append(error)
    fits["individual_refit_errors"] = torch.tensor(individual_errors, dtype=dtype)
    fits["paths"] = []
    epoch_rates = torch.cat(
        [torch.full((20,), 0.05, dtype=dtype), torch.full((20,), 0.01, dtype=dtype)]
    )
    for seed in ORDER_SEEDS:
        generator = torch.Generator().manual_seed(seed)
        orders = torch.stack([torch.randperm(n, generator=generator) for _ in range(40)])
        trace = m.sgd_trace(train["x"], train_y, orders, epoch_rates, L2)
        end = trace["states"][-1]
        gradient = m.logistic_gradients(end, train["x"], train_y, L2).mean(0)
        trace["seed"] = seed
        trace["final_gradient_max"] = gradient.abs().max()
        trace["distance_to_converged"] = torch.linalg.vector_norm(end - fits["base"]["theta"][0])
        fits["paths"].append(trace)
        torch.save(trace, directory / f"sgd_seed_{seed}.pt")
        progress(f"Recorded all 5120 actual SGD updates for order seed {seed}")
    measured = score_queries(fits, train, train_y, ranking, m)
    rankers = {
        "influence": measured["removal_predictions"][0].mean(1),
        "gradient": measured["gradient_similarity"].mean(1),
        "pixels": measured["signed_pixel_similarity"].mean(1),
        "ranking_loo": measured["loo_loss_changes"].mean(1),
    }
    for seed, path in zip(ORDER_SEEDS, measured["paths"]):
        for name, scores in path["checkpoints"]["scores"].items():
            rankers[f"tracin_seed{seed}_{name}"] = scores.mean(1)
    for seed in RANDOM_SEEDS:
        rankers[f"random_{seed}"] = torch.rand(
            n, generator=torch.Generator().manual_seed(seed), dtype=dtype
        )
    policies = []
    deletion_weights = []
    for name, scores in rankers.items():
        for budget in BUDGETS:
            removed = m.class_matched_removals(scores, train_y, budget)
            weights = torch.ones(n, dtype=dtype)
            weights[removed] = 0
            deletion_weights.append(weights)
            policies.append(
                {
                    "ranker": name,
                    "budget": budget,
                    "removed": removed,
                    "removed_source_ids": [train["ids"][i] for i in removed.tolist()],
                }
            )
    if len(rankers) != 48 or len(policies) != 240:
        raise ValueError("Every prescribed ranker and deletion budget must remain in the study.")
    deletion_weights = torch.stack(deletion_weights)
    fits["deletions"] = m.fit_weighted_logistic(train["x"], train_y, deletion_weights, L2)
    fits["policies"], fits["rankers"] = policies, rankers
    fits["deletion_weights"] = deletion_weights
    measured["deletions"] = query_outputs(fits["deletions"]["theta"], ranking)
    cg_checks = []
    theta = fits["base"]["theta"][0]
    _, hessian = m.objective_derivatives(theta[None], train["x"], train_y, unit_weights, L2)
    for index in (0, 1):
        rhs = m.logistic_gradients(
            theta, ranking["x"][index : index + 1], ranking["y"][index : index + 1]
        )[0]
        direction, history = m.conjugate_gradient(
            lambda v: m.logistic_hvp(theta, train["x"], train_y, unit_weights[0], v, L2),
            rhs,
        )
        torch.testing.assert_close(
            direction, torch.linalg.solve(hessian[0], rhs), rtol=1e-8, atol=1e-9
        )
        cg_checks.append({"query_index": index, "direction": direction, "residuals": history})
    fits["cg_checks"] = cg_checks
    fits["train_y"] = train_y.clone()
    fits["stage"] = "FITS_AND_RANKING_COMPLETE"
    torch.save(fits, directory / "complete_fits.pt")
    torch.save(measured, directory / "ranking_measurements.pt")
    progress("All 48 policies and 240 deletion refits complete; final queries remain unevaluated")
    return fits, measured


def prepare_study(data, m, directory, progress=lambda message: None):
    """Fit every condition and choose policies using ranking queries only."""
    train, ranking = data["train"], data["ranking"]
    fitted, measured = {}, {}
    for name, labels in [("clean", train["y"]), ("corrupted", train["corrupted_y"])]:
        condition_directory = directory / name
        condition_directory.mkdir(parents=True, exist_ok=False)

        def report(message):
            progress(f"{name}: {message}")

        fitted[name], measured[name] = fit_condition(
            train,
            ranking,
            data["upweight_probe_indices"],
            labels,
            m,
            condition_directory,
            report,
        )
    return fitted, measured

# %%

if MAIN:
    method_names = [
        "logistic_losses",
        "logistic_gradients",
        "objective_derivatives",
        "logistic_hvp",
        "conjugate_gradient",
        "upweight_influence",
        "tracin_checkpoint_scores",
        "ghost_linear_products",
        "fit_weighted_logistic",
        "sgd_trace",
        "trace_loss_reductions",
        "class_matched_removals",
    ]
if MAIN:
    m = SimpleNamespace(**{name: globals()[name] for name in method_names})
if MAIN:
    NATIVE_DIR = Path(tempfile.mkdtemp(prefix="native-", dir=OUTPUT))
if MAIN:
    fitted, ranking_results = prepare_study(data, m, NATIVE_DIR, progress=print)

# %%

from chapter16_shapley_attribution_baselines.exercises.part9_influence_and_tracin.attribution_reference import (
    scipy_logistic_optimum,
)

if MAIN:
    native_reference_checks = []
if MAIN:
    for condition, fits in fitted.items():
        n = len(data["train"]["y"])
        cases = [("base", torch.ones(n, dtype=DTYPE), fits["base"]["theta"][0])]
        for index in (0, 1, 2):
            weights = torch.ones(n, dtype=DTYPE)
            weights[index] = 0
            cases.append((f"loo_{index}", weights, fits["loo"]["theta"][index]))
        weights = torch.ones(n, dtype=DTYPE)
        weights[data["upweight_probe_indices"][0]] = 0.5
        cases.append(("half_weight", weights, fits["perturbations"]["theta"][1]))
        for name, weights, actual in cases:
            reference, residual = scipy_logistic_optimum(
                data["train"]["x"], fits["train_y"], weights, L2
            )
            error = float((actual - reference).abs().max())
            assert error < 1e-8, (condition, name, error)
            native_reference_checks.append(
                {"condition": condition, "case": name, "parameter_error": error, "residual": residual}
            )
if MAIN:
    print("All ten independent digit-refit checks passed.")

# %%

if MAIN:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
if MAIN:
    for row, condition in enumerate(["clean", "corrupted"]):
        fits = fitted[condition]
        for seed, (trace, result) in enumerate(zip(fits["paths"], ranking_results[condition]["paths"])):
            states = trace["states"][::128]
            objective = weighted_objective(
                states,
                data["train"]["x"],
                fits["train_y"],
                torch.ones(len(states), 128, dtype=DTYPE),
                L2,
            )
            ranking_loss = query_outputs(states, data["ranking"])["losses"].mean(1)
            axes[row, 0].plot(range(41), objective, label=f"Order seed {seed}")
            axes[row, 1].plot(range(41), ranking_loss, label=f"Order seed {seed}")
        references = [
            float(fits["base"]["objective"][0]),
            float(ranking_results[condition]["baseline"]["losses"].mean()),
        ]
        for column in (0, 1):
            axes[row, column].axhline(
                references[column], color="black", ls="--", label="Converged model"
            )
            axes[row, column].axhline(math.log(2), color=".5", ls=":", label="Untrained")
            axes[row, column].set(
                xlabel="Epoch", ylabel="Regularized objective" if column == 0 else "Ranking-query loss"
            )
            axes[row, column].set_title(f"{condition.capitalize()} training labels")
            axes[row, column].legend(fontsize=8)
if MAIN:
    show_figure(fig, "attribution_native_training")

# %%

if MAIN:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
if MAIN:
    for row, condition in enumerate(["clean", "corrupted"]):
        for offset, key, label in [
            (-0.25, "exact", "Observed"),
            (0, "first_order", "Taylor"),
            (0.25, "raw_only", "Omit regularizer"),
        ]:
            totals = [
                float(path["accounting"][key].sum(0).mean())
                for path in ranking_results[condition]["paths"]
            ]
            axes[row, 0].bar(np.arange(3) + offset, totals, width=0.25, label=label)
        for seed, path in enumerate(ranking_results[condition]["paths"]):
            exact = path["accounting"]["exact_steps"].mean(1).cumsum(0)
            taylor = path["accounting"]["first_order_steps"].mean(1).cumsum(0)
            axes[row, 1].plot(
                torch.arange(1, len(exact) + 1) / 128,
                (exact - taylor).abs(),
                label=f"Order seed {seed}",
            )
        axes[row, 0].axhline(0, color="black", lw=0.7)
        axes[row, 0].set(
            xlabel="Order seed", xticks=[0, 1, 2], ylabel="Total ranking-query loss reduction"
        )
        axes[row, 1].set(xlabel="Epoch", ylabel="Absolute accumulated Taylor error")
        for column in (0, 1):
            axes[row, column].set_title(condition.capitalize())
            axes[row, column].legend(fontsize=8)
if MAIN:
    show_figure(fig, "attribution_native_path_accounting")

# %%

def evaluate_final(fitted, train, final_queries, m, directory):
    """Evaluate already frozen fits and policies, after both conditions finish."""
    if set(fitted) != {"clean", "corrupted"}:
        raise ValueError("Both conditions must finish before final evaluation.")
    if any(value.get("stage") != "FITS_AND_RANKING_COMPLETE" for value in fitted.values()):
        raise ValueError(
            "Final evaluation cannot run before every fit and deletion policy is frozen."
        )
    final = {}
    for name, fits in fitted.items():
        final[name] = score_queries(fits, train, fits["train_y"], final_queries, m)
        final[name]["deletions"] = query_outputs(fits["deletions"]["theta"], final_queries)
        torch.save(final[name], directory / name / "final_measurements.pt")
    return final

# %%

if MAIN:
    final_results = evaluate_final(fitted, data["train"], data["final"], m, NATIVE_DIR)
if MAIN:
    query_data = data["final"]
if MAIN:
    results = final_results
if MAIN:
    print("All frozen conditions evaluated on the 64 final queries.")

# %%

if MAIN:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
if MAIN:
    for row, condition in enumerate(["clean", "corrupted"]):
        measured = results[condition]
        actual = measured["loo_loss_changes"]
        predicted = measured["removal_predictions"][0]
        wrong_sign = (actual > 0) != (predicted > 0)
        axes[row, 0].scatter(
            actual[~wrong_sign], predicted[~wrong_sign], s=5, alpha=0.18, rasterized=True
        )
        axes[row, 0].scatter(
            actual[wrong_sign],
            predicted[wrong_sign],
            s=9,
            color="#d73027",
            alpha=0.6,
            label=f"Wrong sign: {int(wrong_sign.sum())}/{actual.numel()}",
        )
        low, high = float(torch.minimum(actual.min(), predicted.min())), float(
            torch.maximum(actual.max(), predicted.max())
        )
        axes[row, 0].plot([low, high], [low, high], color="black", lw=1)
        axes[row, 0].set_ylabel("Predicted removal loss change")
        axes[row, 0].legend(fontsize=8)
        for seed, path in enumerate(measured["paths"]):
            axes[row, 1].scatter(
                actual,
                path["checkpoints"]["scores"]["all"],
                s=4,
                alpha=0.12,
                label=f"Order {seed}",
                rasterized=True,
            )
        axes[row, 1].set_ylabel("All-checkpoint TracIn score")
        axes[row, 1].legend(fontsize=8, markerscale=3)
        axes[row, 2].scatter(
            actual,
            measured["signed_pixel_similarity"],
            s=4,
            alpha=0.12,
            color="#35978f",
            rasterized=True,
        )
        axes[row, 2].set_ylabel("Signed pixel cosine")
        for column, label in enumerate(["Influence", "TracInCP", "Pixel control"]):
            axes[row, column].axhline(0, color=".6", lw=0.6)
            axes[row, column].axvline(0, color=".6", lw=0.6)
            axes[row, column].set_xlabel("Actual removal loss change")
            axes[row, column].set_title(f"{condition.capitalize()}: {label}")
if MAIN:
    show_figure(fig, "attribution_final_loo_comparison")

# %%

def per_query_spearman(actual, scores):
    a = rankdata(actual.numpy(), axis=0)
    b = rankdata(scores.numpy(), axis=0)
    a, b = a - a.mean(0), b - b.mean(0)
    denominator = np.sqrt((a * a).sum(0) * (b * b).sum(0))
    return np.divide(
        (a * b).sum(0), denominator, out=np.full_like(denominator, np.nan), where=denominator > 0
    )


if MAIN:
    fig, axes = plt.subplots(1, 2, figsize=(15, 9), sharey=True)
if MAIN:
    for ax, condition in zip(axes, ["clean", "corrupted"]):
        measured = results[condition]
        methods_to_compare = [
            (f"Influence: damping {value:g}", measured["removal_predictions"][index])
            for index, value in enumerate(DAMPING)
        ]
        methods_to_compare += [
            ("Wrong normalization", measured["wrong_normalization"]),
            ("Gradient", measured["gradient_similarity"]),
            ("Pixels", measured["signed_pixel_similarity"]),
        ]
        for seed, path in enumerate(measured["paths"]):
            methods_to_compare.append((f"Exact path: order {seed}", path["accounting"]["exact"]))
            for name, scores in path["checkpoints"]["scores"].items():
                methods_to_compare.append((f"CP {name}: order {seed}", scores))
        correlations = np.stack(
            [
                per_query_spearman(measured["loo_loss_changes"], scores)
                for _, scores in methods_to_compare
            ]
        )
        im = ax.imshow(np.ma.masked_invalid(correlations), aspect="auto", vmin=-1, vmax=1, cmap="RdBu")
        ax.set(
            xlabel="Final query index (all 64)",
            title=f"{condition.capitalize()} training labels",
            yticks=range(len(methods_to_compare)),
        )
        ax.set_yticklabels([name for name, _ in methods_to_compare], fontsize=9)
if MAIN:
    fig.subplots_adjust(left=0.19, right=0.85, wspace=0.15)
if MAIN:
    colorbar_axis = fig.add_axes([0.88, 0.17, 0.018, 0.64])
if MAIN:
    fig.colorbar(im, cax=colorbar_axis, label="Spearman correlation with actual leave-one-out effects")
if MAIN:
    fig.savefig(OUTPUT / "attribution_final_query_agreement.png", dpi=150, bbox_inches="tight")
if MAIN:
    plt.show()

# %%

if MAIN:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
if MAIN:
    for row, condition in enumerate(["clean", "corrupted"]):
        fits, measured = fitted[condition], results[condition]
        probes = fits["probe_indices"]
        finite = measured["finite_derivative"]
        unfinished = (measured["convergence_upweight"][:, probes] - finite[None]).abs().flatten(1)
        damped = (
            (-len(train["x"]) * measured["removal_predictions"][:, probes] - finite[None])
            .abs()
            .flatten(1)
        )
        wrong = (-len(train["x"]) * measured["wrong_normalization"][probes] - finite).abs().flatten()
        for ax, errors in zip(axes[row], [list(unfinished), [*damped, wrong]]):
            for column, error in enumerate(errors):
                offset = np.linspace(-0.16, 0.16, len(error))
                ax.scatter(column + offset, error, s=3, alpha=0.14)
                ax.plot(
                    [column - 0.23, column + 0.23], [float(error.median())] * 2, color="black", lw=2
                )
            ax.set_yscale("symlog", linthresh=1e-8)
            ax.set_ylabel("Absolute derivative error")
            ax.grid(axis="y", alpha=0.15)
        indices = measured["convergence_state_indices"]
        axes[row, 0].set_xticks(
            range(4),
            [f"Step {i}\n|grad|max={float(fits['base']['gradient_norms'][i, 0]):.1e}" for i in indices],
            fontsize=8,
        )
        axes[row, 0].set_title(f"{condition.capitalize()}: unfinished optimization")
        axes[row, 1].set_xticks(
            range(5), [f"Damping\n{v:g}" for v in DAMPING] + ["Wrong\nnormalization"], fontsize=9
        )
        axes[row, 1].set_title(f"{condition.capitalize()}: converged, different inverses")
if MAIN:
    show_figure(fig, "attribution_final_derivative_controls")

# %%

if MAIN:
    for condition in ["clean", "corrupted"]:
        fits, measured = fitted[condition], results[condition]
        actual = measured["perturbations"]["losses"].reshape(12, 8, -1).mean(2)
        actual -= measured["baseline"]["losses"].mean()
        fig, axes = plt.subplots(3, 4, figsize=(14, 10))
        for panel, (ax, index) in enumerate(zip(axes.flat, fits["probe_indices"])):
            # -s * predicted deletion = epsilon * predicted upweight derivative.
            predicted = -np.array(STRENGTHS) * float(measured["removal_predictions"][0, index].mean())
            wrong = -np.array(STRENGTHS) * float(measured["wrong_normalization"][index].mean())
            ax.plot(STRENGTHS, actual[panel], "o-", color="#2166ac", label="Independent refits", ms=4)
            ax.plot(STRENGTHS, predicted, color="#b2182b", label="Local influence")
            ax.plot(STRENGTHS, wrong, "--", color=".5", label="Wrong normalization")
            ax.axhline(0, color=".6", lw=0.6)
            ax.axvline(0, color=".6", lw=0.6)
            true_digit = int(train["digit"][index])
            observed_digit = 7 if fits["train_y"][index] else 1
            ax.set_title(
                f"{train['ids'][index]}\nSource {true_digit}; trained as {observed_digit}", fontsize=9
            )
            ax.set_xlabel("Relative weight change s", fontsize=9)
            ax.set_ylabel("Mean query loss change", fontsize=9)
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.01))
        fig.suptitle(f"{condition.capitalize()} labels: all prescribed weight perturbations", y=1.04)
        show_figure(fig, f"attribution_final_perturbations_{condition}")

# %%

if MAIN:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
if MAIN:
    colors = {
        "influence": "#b2182b",
        "ranking_loo": "#2166ac",
        "gradient": "#e08214",
        "pixels": "#1b7837",
    }
if MAIN:
    line_styles = {"all": "-", "early": "--", "late": ":", "last": "-."}
if MAIN:
    for row, condition in enumerate(["clean", "corrupted"]):
        fits, measured = fitted[condition], results[condition]
        baseline_loss = float(measured["baseline"]["losses"].mean())
        baseline_accuracy = float(measured["baseline"]["correct"].double().mean())
        names = list(fits["rankers"])
        # Draw random and checkpoint curves first so primary curves remain visible.
        names.sort(
            key=lambda name: 0 if name.startswith("random") else (1 if name.startswith("tracin") else 2)
        )
        for name in names:
            indices = [i for i, policy in enumerate(fits["policies"]) if policy["ranker"] == name]
            loss = [0.0, *(measured["deletions"]["losses"][indices].mean(1) - baseline_loss).tolist()]
            accuracy = [
                baseline_accuracy,
                *measured["deletions"]["correct"][indices].double().mean(1).tolist(),
            ]
            if name.startswith("random"):
                style = dict(
                    color=".65",
                    alpha=0.3,
                    lw=0.7,
                    label="32 random rankings" if name == "random_1000" else None,
                )
            elif name.startswith("tracin"):
                choice = name.rsplit("_", 1)[1]
                style = dict(
                    color="#762a83",
                    alpha=0.5,
                    lw=1,
                    ls=line_styles[choice],
                    label=f"CP {choice} (3 orders)" if "seed0" in name else None,
                )
            else:
                style = dict(
                    color=colors[name],
                    lw=2.2,
                    marker="o",
                    ms=3,
                    label={"ranking_loo": "Ranking LOO", "pixels": "Signed pixels"}.get(
                        name, name.capitalize()
                    ),
                )
            for ax, values in zip(axes[row], [loss, accuracy]):
                ax.plot([0, *BUDGETS], values, **style)
        for ax in axes[row]:
            ax.set(xlabel="Training examples removed", xticks=[0, *BUDGETS])
            ax.grid(alpha=0.15)
        axes[row, 0].axhline(0, color="black", lw=0.7)
        axes[row, 1].axhline(baseline_accuracy, color="black", lw=0.7, ls="--")
        axes[row, 0].set(title=f"{condition.capitalize()}: loss", ylabel="Mean final loss change")
        axes[row, 1].set(
            title=f"{condition.capitalize()}: accuracy",
            ylabel="Correct final queries",
            ylim=(0.94, 1.005),
        )
        axes[row, 1].set_yticks(
            [i / 64 for i in (61, 62, 63, 64)], [f"{i}/64" for i in (61, 62, 63, 64)]
        )
if MAIN:
    handles, labels = axes[0, 0].get_legend_handles_labels()
if MAIN:
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, 1.09))
if MAIN:
    show_figure(fig, "attribution_final_deletion_curves")

# %%

if MAIN:
    gallery_queries = [
        i
        for digit in (1, 7)
        for i in torch.nonzero(query_data["digit"] == digit).flatten().tolist()[:2]
    ]
if MAIN:
    for condition in ["clean", "corrupted"]:
        fits, measured = fitted[condition], results[condition]
        methods_for_gallery = [
            ("Influence", measured["removal_predictions"][0]),
            ("CP order 0", measured["paths"][0]["checkpoints"]["scores"]["all"]),
            ("Signed pixels", measured["signed_pixel_similarity"]),
        ]
        fig, axes = plt.subplots(4, 7, figsize=(16, 12))
        for row, query_index in enumerate(gallery_queries):
            axes[row, 0].imshow(query_data["images"][query_index], cmap="gray_r", vmin=0, vmax=1)
            probability = float(measured["baseline"]["probabilities"][0, query_index])
            axes[row, 0].set_title(
                f"Query: digit {int(query_data['digit'][query_index])}\nP(7) = {probability:.3f}",
                fontsize=10,
            )
            axes[row, 0].set_xlabel(query_data["ids"][query_index], fontsize=8)
            for method_index, (name, scores) in enumerate(methods_for_gallery):
                for side, (label, choose) in enumerate(
                    [("Highest", torch.argmax), ("Lowest", torch.argmin)]
                ):
                    index = int(choose(scores[:, query_index]))
                    ax = axes[row, 1 + 2 * method_index + side]
                    ax.imshow(train["images"][index], cmap="gray_r", vmin=0, vmax=1)
                    true_digit, observed = int(train["digit"][index]), (
                        7 if fits["train_y"][index] else 1
                    )
                    actual = float(measured["loo_loss_changes"][index, query_index])
                    ax.set_title(
                        f"{name}: {label}\nSource {true_digit}; trained as {observed}", fontsize=8
                    )
                    ax.set_xlabel(
                        f"{train['ids'][index]}\nScore {float(scores[index, query_index]):+.2e}\nActual removal {actual:+.2e}",
                        fontsize=8,
                    )
        for ax in axes.flat:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(f"{condition.capitalize()} training labels: fixed final-query gallery", y=1.01)
        show_figure(fig, f"attribution_final_gallery_{condition}", h_pad=4)

# %%

if MAIN:
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
if MAIN:
    ghost_control_errors = {}
if MAIN:
    for row, condition in enumerate(["clean", "corrupted"]):
        ghost_control_errors[condition] = []
        for seed, path in enumerate(results[condition]["paths"]):
            cp = path["checkpoints"]
            dense, factored = cp["scores"]["all"], cp["ghost_all_scores"]
            shuffled = torch.zeros_like(dense)
            for theta, rate in zip(cp["checkpoints"], cp["checkpoint_rates"]):
                train_delta = torch.sigmoid(train["x"] @ theta) - fitted[condition]["train_y"]
                query_delta = torch.sigmoid(query_data["x"] @ theta) - query_data["y"]
                raw = ghost_linear_products(
                    train["x"][:, :-1],
                    train_delta.roll(1)[:, None],
                    query_data["x"][:, :-1],
                    query_delta[:, None],
                )
                query_gradient = logistic_gradients(theta, query_data["x"], query_data["y"])
                shuffled += rate * (raw + (L2 * theta @ query_gradient.T)[None, :])
            ax = axes[row, seed]
            ax.scatter(
                dense,
                shuffled,
                s=3,
                alpha=0.1,
                color="#d95f02",
                label="Rolled residual control",
                rasterized=True,
            )
            ax.scatter(
                dense,
                factored,
                s=3,
                alpha=0.12,
                color="#1b9e77",
                label="Factored product",
                rasterized=True,
            )
            ax.plot(
                [float(dense.min()), float(dense.max())],
                [float(dense.min()), float(dense.max())],
                color="black",
                lw=0.8,
            )
            error = float((factored - dense).abs().max())
            control_error = float((shuffled - dense).abs().mean())
            ghost_control_errors[condition].append(control_error)
            ax.set(
                title=f"{condition.capitalize()}: order {seed}\nExact max error {error:.1e}",
                xlabel="Complete-gradient CP score",
                ylabel="Factored or control score",
            )
            ax.legend(fontsize=8, markerscale=3)
if MAIN:
    show_figure(fig, "attribution_native_ghost_products")

# %%

PLAY_CONDITION = "corrupted"  # "clean" or "corrupted"
PLAY_QUERY = 0  # 0 ... 47 in the ranking split
PLAY_BUDGET = 4  # an even number between 2 and 32

# %%

def query_deletion_play(train, query, fits, m, budget=4, l2=0.02):
    """Choose by this query, then refit the chosen set and 32 matched controls.

    This is an exploratory query-specific intervention, not a held-out policy
    evaluation. It does not change any reference-study fits or deletion policies.
    """
    n = len(train["x"])
    theta, labels = fits["base"]["theta"][0], fits["train_y"]
    _, hessian = m.objective_derivatives(
        theta[None],
        train["x"],
        labels,
        torch.ones(1, n, dtype=theta.dtype),
        l2,
    )
    scores = (
        -m.upweight_influence(
            hessian[0],
            m.logistic_gradients(theta, train["x"], labels, l2),
            m.logistic_gradients(theta, query["x"], query["y"]),
        )[:, 0]
        / n
    )
    selected = [m.class_matched_removals(scores, labels, budget)]
    for seed in range(1000, 1032):
        random_scores = torch.rand(
            n, generator=torch.Generator().manual_seed(seed), dtype=theta.dtype
        )
        selected.append(m.class_matched_removals(random_scores, labels, budget))
    weights = torch.ones(33, n, dtype=theta.dtype)
    for row, removed in enumerate(selected):
        weights[row, removed] = 0
    refits = m.fit_weighted_logistic(train["x"], labels, weights, l2)
    all_theta = torch.cat([theta[None], refits["theta"]])
    logits = all_theta @ query["x"].T
    losses = torch.nn.functional.softplus(logits) - query["y"][None, :] * logits
    return {
        "budget": budget,
        "query_ids": query["ids"],
        "scores": scores,
        "selected": selected,
        "weights": weights,
        "refits": refits,
        "probabilities": torch.sigmoid(logits),
        "losses": losses,
        "loss_changes": losses[1:] - losses[:1],
    }

# %%

if MAIN:
    assert PLAY_CONDITION in fitted and 0 <= PLAY_QUERY < 48
if MAIN:
    assert PLAY_BUDGET in range(2, 33, 2)
if MAIN:
    play_query = {
        "x": data["ranking"]["x"][PLAY_QUERY : PLAY_QUERY + 1],
        "y": data["ranking"]["y"][PLAY_QUERY : PLAY_QUERY + 1],
        "ids": [data["ranking"]["ids"][PLAY_QUERY]],
    }
if MAIN:
    play_result = query_deletion_play(train, play_query, fitted[PLAY_CONDITION], m, budget=PLAY_BUDGET)
if MAIN:
    torch.save(play_result, NATIVE_DIR / "query_play.pt")
if MAIN:
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), gridspec_kw={"width_ratios": [1, 1, 1, 1.8]})
if MAIN:
    axes[0].imshow(data["ranking"]["images"][PLAY_QUERY], cmap="gray_r", vmin=0, vmax=1)
if MAIN:
    axes[0].set_title(f"Query: digit {int(data['ranking']['digit'][PLAY_QUERY])}")
if MAIN:
    axes[0].set_xlabel(play_query["ids"][0], fontsize=8)
if MAIN:
    for ax, (label, choose) in zip(
        axes[1:3], [("Highest influence", torch.argmax), ("Lowest influence", torch.argmin)]
    ):
        index = int(choose(play_result["scores"]))
        ax.imshow(train["images"][index], cmap="gray_r", vmin=0, vmax=1)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel(
            f"{train['ids'][index]}\nSource {int(train['digit'][index])}; trained as {7 if fitted[PLAY_CONDITION]['train_y'][index] else 1}",
            fontsize=8,
        )
if MAIN:
    for ax in axes[:3]:
        ax.set_xticks([])
        ax.set_yticks([])
if MAIN:
    changes = play_result["loss_changes"][:, 0]
if MAIN:
    axes[3].scatter(
        np.linspace(0.8, 1.2, 32), changes[1:], color=".5", s=14, alpha=0.6, label="Each random refit"
    )
if MAIN:
    axes[3].scatter([0], changes[:1], color="#b2182b", s=60, marker="D", label="Influence selection")
if MAIN:
    axes[3].axhline(0, color="black", lw=0.8)
if MAIN:
    axes[3].set(
        xticks=[0, 1],
        xticklabels=["Influence", "32 random"],
        ylabel="Query loss change after refitting",
        title=f"Remove {PLAY_BUDGET} from {PLAY_CONDITION} labels",
    )
if MAIN:
    show_figure(fig, "attribution_native_query_play")
if MAIN:
    print(
        f"Query P(7): {float(play_result['probabilities'][0, 0]):.3f} before; {float(play_result['probabilities'][1, 0]):.3f} after selected removal."
    )

# %%

# Save supporting run metadata; the figures and actual interventions above
# are the learning results. Tensor artifacts retain every prescribed case.
if MAIN:
    native_summary = {
        "scope": "real_digit_influence_and_tracin",
        "native_model_executed": True,
        "native_directory": str(NATIVE_DIR),
        "device": "cpu",
        "dtype": "float64",
        "train_count": 128,
        "ranking_query_count": 48,
        "final_query_count": 64,
        "reference_refits": sum(
            sum(len(value[key]["theta"]) for key in ("base", "loo", "perturbations", "deletions"))
            for value in fitted.values()
        ),
        "sgd_paths": sum(len(value["paths"]) for value in fitted.values()),
        "sgd_updates": sum(
            len(path["used_ids"]) for value in fitted.values() for path in value["paths"]
        ),
        "policy_selection_queries": "ranking_only",
        "final_evaluated_after_both_reference_conditions_complete": True,
        "independent_scipy_checks": native_reference_checks,
        "separate_refit_checks": 6,
        "exploratory_play": {
            "condition": PLAY_CONDITION,
            "query_index": PLAY_QUERY,
            "budget": PLAY_BUDGET,
            "refits": len(play_result["refits"]["theta"]),
            "selected_loss_change": float(play_result["loss_changes"][0, 0]),
        },
        "conditions": {},
    }
if MAIN:
    for condition in ["clean", "corrupted"]:
        fits, measured = fitted[condition], results[condition]
        actual, prediction = measured["loo_loss_changes"], measured["removal_predictions"][0]
        native_summary["conditions"][condition] = {
            "final_correct": int(measured["baseline"]["correct"].sum()),
            "final_loss": float(measured["baseline"]["losses"].mean()),
            "influence_wrong_sign": int(((actual > 0) != (prediction > 0)).sum()),
            "train_query_pairs": actual.numel(),
            "per_query_influence_spearman": per_query_spearman(actual, prediction).tolist(),
            "max_refit_gradient": max(
                float(fits[key]["gradient_norms"][-1].max())
                for key in ("base", "loo", "perturbations", "deletions")
            ),
            "max_ghost_error": max(
                path["checkpoints"]["ghost_max_error"] for path in measured["paths"]
            ),
            "rolled_residual_control_mean_errors": ghost_control_errors[condition],
            "rankers": len(fits["rankers"]),
            "deletion_policies": len(fits["policies"]),
            "gallery_query_ids": [query_data["ids"][i] for i in gallery_queries],
        }
if MAIN:
    (OUTPUT / "native_summary.json").write_text(
        json.dumps(native_summary, indent=2, allow_nan=False) + "\n"
    )
if MAIN:
    print("Saved the full study and editable-query results.")

# %%
