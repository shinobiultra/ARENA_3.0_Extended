"""Small trainable methods to expose as native-model learner exercises."""

from __future__ import annotations

from collections.abc import Callable
import torch as t

from .das_methods import orthonormal_columns, subspace_interchange, intervention_loss


def fit_minibatch_alignment(
    base: t.Tensor,
    source: t.Tensor,
    targets: t.Tensor,
    continue_model: Callable[[t.Tensor, t.Tensor], t.Tensor],
    *,
    rank: int,
    steps: int = 120,
    batch_size: int = 16,
    learning_rate: float = 0.01,
    seed: int = 0,
) -> dict:
    """DAS with recipient-context-aware minibatches and a thin QR basis.

    continue_model(pair_indices, replacements) resumes the frozen recipient
    computation. The only optimizer parameter is raw, a width × rank matrix.
    All pairs are visited once per shuffled epoch. Save the final iterate;
    training losses below are minibatch losses, not held-out measurements.
    """
    if base.ndim != 2 or source.shape != base.shape or targets.shape != (len(base),):
        raise ValueError("Use paired activation matrices and one target per pair.")
    if not 1 <= rank <= base.shape[1] or min(steps, batch_size) < 1 or learning_rate <= 0:
        raise ValueError("Rank, training steps, batch size and learning rate must be positive.")
    base, source = base.detach(), source.detach()
    generator = t.Generator(device=base.device).manual_seed(seed)
    raw = t.nn.Parameter(
        t.randn(base.shape[1], rank, dtype=base.dtype, device=base.device, generator=generator)
        / base.shape[1] ** 0.5
    )
    optimizer = t.optim.Adam([raw], lr=learning_rate)
    history, order, offset = [], t.empty(0, dtype=t.long), 0
    for step in range(steps):
        if offset >= len(order):
            order = t.randperm(len(base), generator=generator, device=base.device)
            offset = 0
        indices = order[offset : offset + batch_size]
        offset += batch_size
        basis = orthonormal_columns(raw)
        replacement = subspace_interchange(base[indices], source[indices], basis)
        logits = continue_model(indices, replacement)
        loss = intervention_loss(logits, targets[indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if not t.isfinite(loss) or raw.grad is None or not t.isfinite(raw.grad).all():
            raise ValueError("A finite intervention gradient must reach the alignment.")
        history.append(
            dict(
                step=step,
                loss=float(loss.detach()),
                gradient_norm=float(raw.grad.norm()),
                batch_size=len(indices),
                iia=float((logits.detach().argmax(-1) == targets[indices]).float().mean()),
            )
        )
        optimizer.step()
    return dict(basis=orthonormal_columns(raw).detach().clone(), history=history)


def fit_role_probe(activations: t.Tensor, roles: t.Tensor, ridge: float = 0.1) -> dict:
    """Observational ridge probe; its coefficient defines a rank-one edit.

    Fit only training activations to predict R in {-1, +1}. Solving in sample
    space avoids a width × width inverse. This is a predictive baseline, with
    no intervention loss. It need not identify a causally effective subspace.
    """
    if activations.ndim != 2 or roles.shape != (len(activations),) or ridge <= 0:
        raise ValueError("Use one binary role label per activation and positive ridge.")
    if not t.all((roles == 0) | (roles == 1)):
        raise ValueError("Role labels must be zero or one.")
    x_mean = activations.mean(0)
    target = roles.to(activations) * 2 - 1
    y_mean = target.mean()
    centered = activations - x_mean
    gram = centered @ centered.T
    dual = t.linalg.solve(
        gram
        + ridge
        * len(activations)
        * t.eye(len(activations), dtype=activations.dtype, device=activations.device),
        target - y_mean,
    )
    coefficient = centered.T @ dual
    if coefficient.norm() < 1e-8:
        raise ValueError("The fitted probe has no direction.")
    return dict(
        basis=(coefficient / coefficient.norm())[:, None],
        coefficient=coefficient,
        x_mean=x_mean,
        y_mean=y_mean,
    )
