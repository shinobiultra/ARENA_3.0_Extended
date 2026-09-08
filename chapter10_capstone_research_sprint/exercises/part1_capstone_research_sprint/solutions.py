# %%
"""Reference implementation for [10.1] Capstone Research Sprint.

The section studies one exact model organism. A parity feature is mixed through
an eight-dimensional activation space, then read out by a known causal
direction. Learners recover that direction from train templates and test it on
held-out templates with counterfactual activation patching.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch as t
from torch import Tensor


MAIN = __name__ == "__main__"
D_MODEL = 8
ROTATION_SEED = 7
TRAIN_TEMPLATES = tuple(range(12))
HELDOUT_TEMPLATES = tuple(range(12, 20))


# %%
@dataclass(frozen=True)
class ParityBatch:
    """Inputs, exact latent features, hidden activations, and XOR labels."""

    bits: Tensor
    template_ids: Tensor
    latent_features: Tensor
    activations: Tensor
    labels: Tensor


def make_rotation(seed: int = ROTATION_SEED, device: str | t.device = "cpu") -> Tensor:
    """Create the fixed orthogonal mixing matrix on CPU, then move it to device."""

    generator = t.Generator(device="cpu").manual_seed(seed)
    raw = t.randn(D_MODEL, D_MODEL, generator=generator, dtype=t.float64)
    q, r = t.linalg.qr(raw)
    signs = t.sign(t.diag(r))
    signs[signs == 0] = 1
    return (q * signs.unsqueeze(0)).to(device)


def template_nuisance(template_ids: Tensor) -> Tensor:
    """Five deterministic template features that never determine the label."""

    ids = template_ids.to(dtype=t.float64)
    scaled = (ids - 9.5) / 9.5
    alternating = t.where(template_ids.remainder(2) == 0, 1.0, -1.0).to(ids)
    return t.stack(
        [ids.sin(), ids.cos(), scaled, alternating, (2 * ids + 1).sin()],
        dim=-1,
    )


def make_parity_batch(template_ids: Iterable[int], rotation: Tensor) -> ParityBatch:
    """Enumerate all four bit pairs for each template and run the exact encoder."""

    template_ids = tuple(int(template_id) for template_id in template_ids)
    if not template_ids:
        raise ValueError("template_ids must contain at least one template")
    device = rotation.device
    bit_block = t.tensor(
        [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]],
        dtype=t.float64,
        device=device,
    )
    bits = bit_block.repeat(len(template_ids), 1)
    ids = t.tensor(template_ids, dtype=t.long, device=device).repeat_interleave(4)
    parity = bits[:, 0] * bits[:, 1]
    nuisance = template_nuisance(ids)
    latent_features = t.cat([bits, parity[:, None], nuisance], dim=1)
    activations = latent_features @ rotation.T
    labels = (parity > 0).long()
    return ParityBatch(bits, ids, latent_features, activations, labels)


def predict_from_direction(activations: Tensor, direction: Tensor) -> Tensor:
    """Apply the model organism's binary linear readout convention."""

    scores = activations @ direction
    scores = t.where(scores.abs() < 1e-10, t.zeros_like(scores), scores)
    return (scores >= 0).long()


# %%
def fit_ridge_direction(features: Tensor, labels: Tensor, ridge: float = 1e-3) -> Tensor:
    """Fit a linear direction to signed binary labels with a closed-form ridge solve."""

    if features.ndim != 2 or labels.shape != (features.shape[0],):
        raise ValueError("expected features [batch, d] and labels [batch]")
    if ridge < 0:
        raise ValueError("ridge must be non-negative")
    target = labels.to(features.dtype) * 2 - 1
    identity = t.eye(features.shape[1], dtype=features.dtype, device=features.device)
    return t.linalg.solve(features.T @ features + ridge * identity, features.T @ target)


def direction_accuracy(features: Tensor, labels: Tensor, direction: Tensor) -> float:
    """Return binary classification accuracy for a direction."""

    return float((predict_from_direction(features, direction) == labels).double().mean())


def shuffled_label_baseline(
    train_features: Tensor,
    train_labels: Tensor,
    test_features: Tensor,
    test_labels: Tensor,
    *,
    n_shuffles: int = 128,
    seed: int = 11,
) -> Tensor:
    """Fit probes to permuted train labels and return held-out accuracies."""

    generator = t.Generator(device="cpu").manual_seed(seed)
    scores = []
    for _ in range(n_shuffles):
        permutation = t.randperm(len(train_labels), generator=generator).to(train_labels.device)
        direction = fit_ridge_direction(train_features, train_labels[permutation])
        scores.append(direction_accuracy(test_features, test_labels, direction))
    return t.tensor(scores, dtype=t.float64)


# %%
def paired_bootstrap_delta_ci(
    method_correct: Tensor,
    baseline_correct: Tensor,
    *,
    n_resamples: int = 2_000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap a paired accuracy difference and return estimate, 95% interval."""

    if method_correct.shape != baseline_correct.shape or method_correct.ndim != 1:
        raise ValueError("correctness tensors must be one-dimensional and equally sized")
    delta = method_correct.to(t.float64) - baseline_correct.to(t.float64)
    generator = t.Generator(device="cpu").manual_seed(seed)
    sample_indices = t.randint(
        len(delta),
        (n_resamples, len(delta)),
        generator=generator,
        device="cpu",
    ).to(delta.device)
    bootstrap = delta[sample_indices].mean(dim=1).cpu()
    low, high = t.quantile(bootstrap, t.tensor([0.025, 0.975], dtype=t.float64))
    return float(delta.mean()), float(low), float(high)


# %%
def counterfactual_donor_indices(batch: ParityBatch) -> Tensor:
    """Pair each example with the same template and first bit, but flipped second bit."""

    indices = []
    for row in range(len(batch.labels)):
        matches = (
            (batch.template_ids == batch.template_ids[row])
            & (batch.bits[:, 0] == batch.bits[row, 0])
            & (batch.bits[:, 1] == -batch.bits[row, 1])
        )
        found = t.nonzero(matches, as_tuple=False).flatten()
        if len(found) != 1:
            raise RuntimeError("each recipient must have exactly one counterfactual donor")
        indices.append(found[0])
    return t.stack(indices)


def patch_along_direction(
    recipient_activations: Tensor,
    donor_activations: Tensor,
    direction: Tensor,
) -> Tensor:
    """Replace the recipient's projection along direction with the donor's projection."""

    unit = direction / direction.norm()
    coefficient_delta = (donor_activations - recipient_activations) @ unit
    return recipient_activations + coefficient_delta[:, None] * unit


def patch_target_accuracy(
    batch: ParityBatch,
    donor_indices: Tensor,
    patch_direction: Tensor,
    readout_direction: Tensor,
) -> float:
    """Measure whether a patch makes the model match the donor's answer."""

    patched = patch_along_direction(
        batch.activations,
        batch.activations[donor_indices],
        patch_direction,
    )
    predictions = predict_from_direction(patched, readout_direction)
    return float((predictions == batch.labels[donor_indices]).double().mean())


def random_direction_controls(
    batch: ParityBatch,
    donor_indices: Tensor,
    readout_direction: Tensor,
    *,
    n_directions: int = 256,
    seed: int = 13,
) -> tuple[Tensor, Tensor]:
    """Return patch scores and exact-direction alignments for isotropic controls."""

    generator = t.Generator(device="cpu").manual_seed(seed)
    directions = t.randn(
        n_directions,
        batch.activations.shape[1],
        generator=generator,
        dtype=t.float64,
    )
    directions = directions / directions.norm(dim=1, keepdim=True)
    directions = directions.to(batch.activations.device)
    exact_unit = readout_direction / readout_direction.norm()
    scores = t.tensor(
        [
            patch_target_accuracy(batch, donor_indices, direction, readout_direction)
            for direction in directions
        ],
        dtype=t.float64,
    )
    alignments = (directions @ exact_unit).abs().cpu()
    return scores, alignments


# %%
def noise_sweep_accuracy(
    activations: Tensor,
    labels: Tensor,
    direction: Tensor,
    sigmas: Tensor,
    *,
    repeats: int = 128,
    seed: int = 123,
) -> Tensor:
    """Measure probe accuracy as isotropic activation noise increases."""

    generator = t.Generator(device="cpu").manual_seed(seed)
    expanded = activations.repeat_interleave(repeats, dim=0)
    expanded_labels = labels.repeat_interleave(repeats)
    standard_noise = t.randn(
        expanded.shape,
        generator=generator,
        dtype=activations.dtype,
        device="cpu",
    ).to(activations.device)
    accuracies = []
    for sigma in sigmas:
        noisy = expanded + float(sigma) * standard_noise
        accuracies.append(direction_accuracy(noisy, expanded_labels, direction))
    return t.tensor(accuracies, dtype=t.float64)


# %%
def run_study(device: str | t.device = "cpu") -> dict[str, object]:
    """Run the complete preregistered parity-direction study."""

    rotation = make_rotation(device=device)
    exact_direction = rotation[:, 2]
    train = make_parity_batch(TRAIN_TEMPLATES, rotation)
    heldout = make_parity_batch(HELDOUT_TEMPLATES, rotation)

    learned_direction = fit_ridge_direction(train.activations, train.labels)
    raw_direction = fit_ridge_direction(train.bits, train.labels)
    template_direction = fit_ridge_direction(train.latent_features[:, 3:], train.labels)

    heldout_predictions = predict_from_direction(heldout.activations, learned_direction)
    raw_predictions = predict_from_direction(heldout.bits, raw_direction)
    heldout_accuracy = float((heldout_predictions == heldout.labels).double().mean())
    raw_accuracy = float((raw_predictions == heldout.labels).double().mean())
    template_accuracy = direction_accuracy(
        heldout.latent_features[:, 3:], heldout.labels, template_direction
    )
    shuffle_scores = shuffled_label_baseline(
        train.activations,
        train.labels,
        heldout.activations,
        heldout.labels,
    )

    learned_unit = learned_direction / learned_direction.norm()
    exact_unit = exact_direction / exact_direction.norm()
    direction_cosine = float((learned_unit @ exact_unit).abs())
    bootstrap_delta, bootstrap_low, bootstrap_high = paired_bootstrap_delta_ci(
        heldout_predictions == heldout.labels,
        raw_predictions == heldout.labels,
    )

    donor_indices = counterfactual_donor_indices(heldout)
    learned_patch_accuracy = patch_target_accuracy(
        heldout, donor_indices, learned_direction, exact_direction
    )
    exact_patch_accuracy = patch_target_accuracy(
        heldout, donor_indices, exact_direction, exact_direction
    )
    random_patch_scores, random_alignments = random_direction_controls(
        heldout, donor_indices, exact_direction
    )

    ablated = heldout.activations - (heldout.activations @ learned_unit)[:, None] * learned_unit
    ablation_accuracy = direction_accuracy(ablated, heldout.labels, exact_direction)
    sigmas = t.tensor([0.0, 0.25, 0.5, 1.0, 2.0, 3.0], dtype=t.float64)
    noise_accuracies = noise_sweep_accuracy(
        heldout.activations,
        heldout.labels,
        learned_direction,
        sigmas,
    )

    criteria = {
        "heldout_accuracy_at_least_0_95": heldout_accuracy >= 0.95,
        "direction_cosine_at_least_0_98": direction_cosine >= 0.98,
        "raw_and_template_baselines_at_most_0_60": max(raw_accuracy, template_accuracy) <= 0.60,
        "patch_target_accuracy_at_least_0_90": learned_patch_accuracy >= 0.90,
        "patch_beats_random_mean_by_0_70": (
            learned_patch_accuracy - float(random_patch_scores.mean()) >= 0.70
        ),
    }
    return {
        "model_family": "exact_rotated_parity_model",
        "dataset": "balanced_xor_with_heldout_template_nuisance_v1",
        "d_model": D_MODEL,
        "train_template_count": len(TRAIN_TEMPLATES),
        "heldout_template_count": len(HELDOUT_TEMPLATES),
        "train_example_count": len(train.labels),
        "heldout_example_count": len(heldout.labels),
        "heldout_accuracy": heldout_accuracy,
        "raw_bits_accuracy": raw_accuracy,
        "template_only_accuracy": template_accuracy,
        "label_shuffle_accuracy_mean": float(shuffle_scores.mean()),
        "label_shuffle_accuracy_p95": float(t.quantile(shuffle_scores, 0.95)),
        "direction_cosine": direction_cosine,
        "paired_accuracy_delta": bootstrap_delta,
        "paired_accuracy_delta_ci_low": bootstrap_low,
        "paired_accuracy_delta_ci_high": bootstrap_high,
        "ablation_accuracy": ablation_accuracy,
        "ablation_drop": heldout_accuracy - ablation_accuracy,
        "exact_patch_target_accuracy": exact_patch_accuracy,
        "learned_patch_target_accuracy": learned_patch_accuracy,
        "random_patch_target_accuracy_mean": float(random_patch_scores.mean()),
        "random_patch_target_accuracy_median": float(random_patch_scores.median()),
        "random_patch_target_accuracy_p95": float(t.quantile(random_patch_scores, 0.95)),
        "best_random_patch_target_accuracy": float(random_patch_scores.max()),
        "best_random_direction_alignment": float(
            random_alignments[random_patch_scores.argmax()]
        ),
        "noise_sigmas": [float(value) for value in sigmas],
        "noise_accuracies": [float(value) for value in noise_accuracies],
        "noise_accuracy_at_sigma_1": float(noise_accuracies[3]),
        "preregistered_criteria": criteria,
        "contract_passed": all(criteria.values()),
        "tests_passed": all(criteria.values()),
        "accepted": all(criteria.values()),
    }


def run_smoke_test(cpu: bool = True) -> dict[str, object]:
    """Run the complete small study on CPU for notebook and CI checks."""

    _ = cpu
    return run_study(device="cpu")


def run_gpu_test(max_vram_gb: float = 24.0) -> dict[str, object]:
    """Repeat the exact study on CUDA for the serialized extension report."""

    if not t.cuda.is_available():
        raise RuntimeError("CUDA is required for the 10.1 GPU verification path.")
    t.cuda.reset_peak_memory_stats()
    result = run_study(device="cuda")
    t.cuda.synchronize()
    peak_vram_gb = t.cuda.max_memory_allocated() / 1024**3
    within_budget = peak_vram_gb <= max_vram_gb
    return {
        **result,
        "cuda_available": True,
        "device": t.cuda.get_device_name(0),
        "scientific_result_executed": True,
        "exact_ground_truth_checked": result["direction_cosine"] >= 0.98,
        "live_intervention_executed": result["learned_patch_target_accuracy"] >= 0.90,
        "peak_vram_gb": peak_vram_gb,
        "within_vram_budget": within_budget,
        "max_allowed_gpu_gb": max_vram_gb,
        "preflight_passed": bool(result["contract_passed"] and within_budget),
        "full_path": (
            "Recover the exact distributed XOR direction, evaluate held-out templates, "
            "run causal patches, compare isotropic random directions, and stress-test noise."
        ),
    }


def run_full_experiment(max_vram_gb: float = 24.0) -> dict[str, object]:
    """Run the serialized CUDA verification path."""

    return run_gpu_test(max_vram_gb=max_vram_gb)


if MAIN:
    metrics = run_smoke_test(cpu=True)
    for key in (
        "heldout_accuracy",
        "raw_bits_accuracy",
        "direction_cosine",
        "learned_patch_target_accuracy",
        "random_patch_target_accuracy_mean",
        "contract_passed",
    ):
        print(f"{key}: {metrics[key]}")
