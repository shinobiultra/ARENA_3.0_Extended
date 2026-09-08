"""Visible mathematical methods for the developing causal-alignment lesson.

The real-model lesson is still under construction. These functions are small
enough to implement in the notebook, with independent tests for each operation.
"""

from __future__ import annotations

from collections.abc import Callable

import torch as t


def high_level_interchange(
    base: t.Tensor, source: t.Tensor, variables: tuple[int, ...]
) -> t.Tensor:
    """Replace named binary variables, preserving the rest of each base row.

    Column 0 is a routing decision; column 1 is an independent side decision.
    This returns the intervened variable setting, before evaluating the output.
    """
    if base.ndim != 2 or base.shape[1] != 2 or source.shape != base.shape:
        raise ValueError("Expected paired [examples, two binary variables] arrays.")
    if not t.all((base == 0) | (base == 1)) or not t.all((source == 0) | (source == 1)):
        raise ValueError("The high-level variables must be binary.")
    if len(set(variables)) != len(variables) or any(i not in (0, 1) for i in variables):
        raise ValueError("Choose distinct variables from (0, 1).")
    result = base.clone()
    result[:, list(variables)] = source[:, list(variables)]
    return result


def high_level_output(variables: t.Tensor) -> t.Tensor:
    """Joint output class for routing and side decisions: 2 * routing + side."""
    return (2 * variables[:, 0] + variables[:, 1]).long()


def orthonormal_columns(raw: t.Tensor) -> t.Tensor:
    """Differentiable reduced QR, with a defined sign convention.

    Only the column span is used by an intervention. QR signs do not change its
    projector. The sign convention makes saved bases easier to compare; QR is
    differentiable only away from rank loss. A zero-column basis is allowed.
    """
    if raw.ndim != 2 or raw.shape[1] > raw.shape[0] or not raw.is_floating_point():
        raise ValueError("Use a floating [width, rank] matrix, with rank <= width.")
    if not t.isfinite(raw).all():
        raise ValueError("The alignment parameters must be finite.")
    if raw.shape[1] == 0:
        return raw.clone()
    if int(t.linalg.matrix_rank(raw.detach())) != raw.shape[1]:
        raise ValueError("QR alignment requires linearly independent columns.")
    q, r = t.linalg.qr(raw, mode="reduced")
    signs = t.where(r.diagonal() < 0, -1.0, 1.0)
    return q * signs


def subspace_interchange(base: t.Tensor, source: t.Tensor, basis: t.Tensor) -> t.Tensor:
    """Copy source coordinates in a column-orthonormal subspace.

    For row vectors, h' = h + (s - h) Q Q.T. All base coordinates in the
    orthogonal complement are preserved. This is equivalent to rotate, replace
    the selected coordinates, then unrotate, without building a width^2 matrix.
    """
    if base.ndim != 2 or source.shape != base.shape:
        raise ValueError("Expected paired [examples, width] activations.")
    if basis.ndim != 2 or basis.shape[0] != base.shape[1]:
        raise ValueError("The basis must have shape [width, rank].")
    gram = basis.T @ basis
    identity = t.eye(basis.shape[1], dtype=basis.dtype, device=basis.device)
    if not t.allclose(gram, identity, atol=1e-5, rtol=1e-5):
        raise ValueError("Subspace interchange needs orthonormal columns.")
    return base + ((source - base) @ basis) @ basis.T


def intervention_loss(logits: t.Tensor, target: t.Tensor) -> t.Tensor:
    """Mean full-output-space cross entropy against high-level counterfactuals."""
    if logits.ndim != 2 or target.shape != (logits.shape[0],) or not len(target):
        raise ValueError("Use nonempty [examples, classes] logits and one target per row.")
    if target.dtype != t.long:
        raise ValueError("Targets must be integer class indices (torch.long).")
    if t.any(target < 0) or t.any(target >= logits.shape[1]):
        raise ValueError("A target is outside the output vocabulary.")
    log_normalizer = t.logsumexp(logits, dim=-1)
    target_logit = logits.gather(1, target[:, None]).squeeze(1)
    return (log_normalizer - target_logit).mean()


def interchange_accuracy(logits: t.Tensor, target: t.Tensor, mask: t.Tensor | None = None) -> float:
    """Output agreement on a specified intervention distribution.

    An empty subgroup is undefined (NaN), not perfect agreement. Report both
    all-pair IIA and changed-variable IIA: identity pairs can inflate an average.
    """
    if logits.ndim != 2 or target.shape != (len(logits),):
        raise ValueError("Use one target for each logit row.")
    if mask is None:
        mask = t.ones_like(target, dtype=t.bool)
    if mask.shape != target.shape or mask.dtype != t.bool:
        raise ValueError("The subgroup mask must be a boolean vector of matching shape.")
    if not mask.any():
        return float("nan")
    return float((logits.argmax(-1)[mask] == target[mask]).double().mean())


def fit_alignment(
    base: t.Tensor,
    source: t.Tensor,
    targets: t.Tensor,
    continue_model: Callable[[t.Tensor], t.Tensor],
    *,
    rank: int,
    steps: int = 200,
    learning_rate: float = 0.03,
    seed: int = 0,
) -> dict:
    """Optimize only a subspace using frozen activations and a frozen model tail.

    This small full-batch implementation is used by the exact organism. For a
    transformer, the continuation also needs each base prompt's untouched token
    positions. The caller must freeze model weights but retain autograd through
    the tail; inference_mode/no_grad around that tail would break DAS training.
    The returned alignment is the final step, not a test-selected checkpoint.
    """
    if not 1 <= rank <= base.shape[-1] or steps < 1 or learning_rate <= 0:
        raise ValueError("Use a feasible positive rank, step count and learning rate.")
    base, source = base.detach(), source.detach()
    generator = t.Generator(device=base.device).manual_seed(seed)
    raw = t.nn.Parameter(
        t.randn(base.shape[1], rank, dtype=base.dtype, device=base.device, generator=generator)
    )
    optimizer = t.optim.Adam([raw], lr=learning_rate)
    history = []
    for step in range(steps + 1):
        q = orthonormal_columns(raw)
        patched = subspace_interchange(base, source, q)
        logits = continue_model(patched)
        loss = intervention_loss(logits, targets)
        if not t.isfinite(loss):
            raise ValueError("The intervention objective became nonfinite.")
        history.append(
            {
                "step": step,
                "loss": float(loss.detach()),
                "iia": interchange_accuracy(logits.detach(), targets),
            }
        )
        if step == steps:
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if raw.grad is None or not t.isfinite(raw.grad).all():
            raise ValueError("A finite gradient must reach the alignment parameters.")
        optimizer.step()
    return {"basis": q.detach().clone(), "history": history}
