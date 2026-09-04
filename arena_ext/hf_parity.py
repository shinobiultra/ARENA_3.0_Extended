"""Parity metrics for custom implementations versus reference models."""

from __future__ import annotations

from dataclasses import dataclass

import torch as t
import torch.nn.functional as F


@dataclass(frozen=True)
class ParityReport:
    """Logit-level parity report for notebook verification blocks."""

    max_abs_diff: float
    mse: float
    kl_divergence: float
    topk_agreement: float

    def passed(
        self,
        *,
        max_abs_diff: float | None = None,
        mse: float | None = None,
        kl_divergence: float | None = None,
        topk_agreement: float | None = None,
    ) -> bool:
        if (
            max_abs_diff is None
            and mse is None
            and kl_divergence is None
            and topk_agreement is None
        ):
            raise ValueError("At least one explicit parity tolerance is required.")
        checks = []
        if max_abs_diff is not None:
            checks.append(self.max_abs_diff <= max_abs_diff)
        if mse is not None:
            checks.append(self.mse <= mse)
        if kl_divergence is not None:
            checks.append(self.kl_divergence <= kl_divergence)
        if topk_agreement is not None:
            checks.append(self.topk_agreement >= topk_agreement)
        return all(checks)


def _validate_logits(custom_logits: t.Tensor, reference_logits: t.Tensor) -> None:
    if custom_logits.shape != reference_logits.shape:
        raise ValueError(
            "custom_logits and reference_logits must have the same shape; "
            f"got {tuple(custom_logits.shape)} and {tuple(reference_logits.shape)}."
        )
    if custom_logits.ndim < 2:
        raise ValueError("Expected logits with at least batch and vocab dimensions.")


def topk_agreement(custom_logits: t.Tensor, reference_logits: t.Tensor, k: int = 10) -> float:
    """Return the fraction of positions with identical top-k token sets."""

    _validate_logits(custom_logits, reference_logits)
    if k <= 0:
        raise ValueError("k must be positive.")
    if k > custom_logits.shape[-1]:
        raise ValueError("k cannot exceed vocab size.")

    custom_topk = custom_logits.topk(k, dim=-1).indices.sort(dim=-1).values
    reference_topk = reference_logits.topk(k, dim=-1).indices.sort(dim=-1).values
    return custom_topk.eq(reference_topk).all(dim=-1).float().mean().item()


def mean_kl_divergence(custom_logits: t.Tensor, reference_logits: t.Tensor) -> float:
    """Compute mean KL(reference || custom) over all non-vocab positions."""

    _validate_logits(custom_logits, reference_logits)
    reference_log_probs = F.log_softmax(reference_logits.float(), dim=-1)
    custom_log_probs = F.log_softmax(custom_logits.float(), dim=-1)
    reference_probs = reference_log_probs.exp()
    kl_by_position = (reference_probs * (reference_log_probs - custom_log_probs)).sum(dim=-1)
    return kl_by_position.mean().item()


def compare_logits(
    custom_logits: t.Tensor,
    reference_logits: t.Tensor,
    k: int = 10,
) -> ParityReport:
    """Compute standard HF-parity metrics for two logit tensors."""

    _validate_logits(custom_logits, reference_logits)
    diff = custom_logits.float() - reference_logits.float()
    return ParityReport(
        max_abs_diff=diff.abs().max().item(),
        mse=diff.pow(2).mean().item(),
        kl_divergence=mean_kl_divergence(custom_logits, reference_logits),
        topk_agreement=topk_agreement(custom_logits, reference_logits, k=k),
    )


def deterministic_generation_equal(
    custom_tokens: t.Tensor | list[int],
    reference_tokens: t.Tensor | list[int],
) -> bool:
    """Check exact greedy-generation equality for fixed prompt/seed outputs."""

    custom = t.as_tensor(custom_tokens)
    reference = t.as_tensor(reference_tokens)
    return custom.shape == reference.shape and bool(custom.eq(reference).all().item())
