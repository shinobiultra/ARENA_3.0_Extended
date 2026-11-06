# %%


MAIN = __name__ == "__main__"

# %%

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch as t
from IPython import get_ipython

if MAIN:
    if get_ipython() is not None:
        get_ipython().run_line_magic("matplotlib", "inline")

root_dir = Path(__file__).resolve().parents[3]
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))
from chapter11_representation_geometry.exercises.part1_pca_svd_geometry_controls import (
    geometry_methods_tests as tests,
)

assets_dir = Path(os.environ.get("ARENA_GEOMETRY_OUTPUT_DIRECTORY", "geometry_outputs"))
if MAIN:
    assets_dir.mkdir(parents=True, exist_ok=True)
if MAIN:
    t.set_num_threads(4)
if MAIN:
    plt.style.use("seaborn-v0_8-whitegrid")

# %%

def fit_pca(x: t.Tensor, components: int):
    """Return training mean, orthonormal column basis and total-variance fractions."""
    if x.ndim != 2 or x.shape[0] < 2 or not 1 <= components <= min(x.shape):
        raise ValueError("PCA needs at least two rows and a feasible component count.")
    mean = x.mean(0)
    _, singular, vh = t.linalg.svd(x - mean, full_matrices=False)
    energy = singular.square()
    if energy.sum() <= 0:
        raise ValueError("Constant observations have no explained-variance ratio.")
    return mean, vh[:components].T, energy[:components] / energy.sum()
if MAIN:


    tests.test_pca_fits_only_training_and_preserves_total_variance(SimpleNamespace(**globals()))
    print("All tests passed!")

# %%

def circular_coordinates(labels: t.Tensor, period: int):
    """Monday and January have phase zero; adjacent classes differ by 2π/period."""
    if period < 3:
        raise ValueError('A circular family needs at least three distinct classes.')
    angle = labels * (2 * t.pi / period)
    return t.stack((angle.cos(), angle.sin()), -1)


def angular_predictions(coordinates: t.Tensor, period: int):
    """Nearest unit-circle class by angle; an exactly zero vector abstains as -1."""
    if coordinates.shape[-1] != 2 or period < 3:
        raise ValueError('Use two coordinates and period >= 3.')
    angle = t.atan2(coordinates[..., 1], coordinates[..., 0])
    prediction = t.round(angle * period / (2 * t.pi)).long().remainder(period)
    return t.where(coordinates.square().sum(-1) > 0, prediction, -1)


def center_template_grid(grid: t.Tensor):
    """Subtract each template's complete class-balanced batch mean.

This uses all examples in that batch. On a held-out template it is an explicit
transductive operation, not a preprocessing transform fitted on training data.
"""
    if grid.ndim != 3 or grid.shape[1] < 2:
        raise ValueError('Expected [templates, classes, width] with at least two classes.')
    return grid - grid.mean(1, keepdim=True)
if MAIN:


    tests.test_angles_wrap_around_and_zero_abstains(SimpleNamespace(**globals()))
    tests.test_template_centering_is_explicitly_transductive(SimpleNamespace(**globals()))
    print("All tests passed!")

# %%

if MAIN:
    generator = t.Generator().manual_seed(1101)
    rotation = t.linalg.qr(t.randn(8, 8, generator=generator, dtype=t.float64)).Q
    calendar = circular_coordinates(t.arange(7, dtype=t.float64), 7)
    residual = t.cat((calendar, t.zeros(7, 6, dtype=t.float64)), dim=1) @ rotation.T
    offsets = (
        t.tensor(
            [
                [0.0, 0.0, 7.0, -4.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -5.0, 8.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 11.0, 6.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=t.float64,
        )
        @ rotation.T
    )
    grid = residual[None] + offsets[:, None]
    centered = center_template_grid(grid)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax, values, title in zip(
        axes,
        [grid, centered],
        ["Template offsets dominate", "Balanced template centering"],
        strict=True,
    ):
        mean, basis, variance = fit_pca(values[:2].flatten(0, 1), 2)
        projected = (values - mean) @ basis
        for i, marker in enumerate(["o", "s", "X"]):
            ax.scatter(
                projected[i, :, 0],
                projected[i, :, 1],
                c=np.arange(7),
                cmap="tab10",
                vmin=0,
                vmax=9,
                marker=marker,
                s=65,
                edgecolor="black",
                label="held-out batch" if i == 2 else f"train template {i}",
            )
        ax.set(title=title, xlabel=f"PC1 ({variance[0]:.0%})", ylabel=f"PC2 ({variance[1]:.0%})")
        ax.legend(fontsize=8)
    axes[1].set_aspect("equal", adjustable="box")
    fig.savefig(assets_dir / "geometry_template_offsets.png", dpi=160)
    plt.show()

# %%

def fit_ridge_coordinates(x: t.Tensor, targets: t.Tensor, ridge: float = 0.0):
    """Affine least squares; intercept is not penalized, penalty is ridge * ||W||²."""
    if x.ndim != 2 or targets.ndim != 2 or x.shape[0] != targets.shape[0] or ridge < 0:
        raise ValueError("Use paired observation matrices and nonnegative ridge.")
    mean, bias = x.mean(0), targets.mean(0)
    centered, values = x - mean, targets - bias
    if ridge == 0:
        weights = t.linalg.lstsq(
            centered, values, driver="gelsd" if x.device.type == "cpu" else "gels"
        ).solution
    else:
        weights = t.linalg.solve(
            centered.T @ centered + ridge * t.eye(x.shape[1], device=x.device, dtype=x.dtype),
            centered.T @ values,
        )
    return mean, weights, bias


def coordinate_r2(targets: t.Tensor, predictions: t.Tensor):
    """Joint coordinate R², relative to the observed target mean; may be negative."""
    if targets.shape != predictions.shape:
        raise ValueError("Prediction and target inventories must match.")
    denominator = (targets - targets.mean(0)).square().sum()
    if denominator <= 0:
        raise ValueError("R² is undefined when targets have no variation.")
    return 1 - (targets - predictions).square().sum() / denominator
if MAIN:


    tests.test_ridge_has_unpenalized_intercept_and_known_shrinkage(SimpleNamespace(**globals()))
    tests.test_least_squares_repeated_fits_and_rank_deficiency(SimpleNamespace(**globals()))
    print("All tests passed!")

# %%

def knn_predictions(train: t.Tensor, labels: t.Tensor, test: t.Tensor, k: int = 3):
    """Euclidean kNN; deterministic nearest-row and lowest-label tie breaking."""
    if not 1 <= k <= len(train) or labels.shape != (len(train),):
        raise ValueError('Use one label per training row and a valid neighbor count.')
    classes, encoded = labels.unique(sorted=True, return_inverse=True)
    neighbors = t.cdist(test, train).argsort(dim=-1, stable=True)[:, :k]
    votes = t.nn.functional.one_hot(encoded[neighbors], len(classes)).sum(1)
    return classes[votes.argmax(-1)]


def cyclic_distance_agreement(class_points: t.Tensor):
    """Correlation of pair distances with chronological circle chord distances.

Class identities, not individual prompt rows, are the permutation unit. A high
score can support periodic order; ID classification alone cannot.
"""
    count = class_points.shape[0]
    if count < 4:
        raise ValueError('Use at least four ordered class centroids.')
    target = circular_coordinates(t.arange(count, device=class_points.device, dtype=class_points.dtype), count)
    observed, expected = t.pdist(class_points), t.pdist(target)
    a, b = observed - observed.mean(), expected - expected.mean()
    denominator = a.norm() * b.norm()
    if denominator <= 0:
        raise ValueError('Distance agreement is undefined for a constant distance pattern.')
    return (a @ b) / denominator


def cyclic_permutation_test(class_points: t.Tensor, permutations: t.Tensor):
    """One-sided Monte Carlo null including the observed arrangement via +1."""
    count = class_points.shape[0]
    if permutations.ndim != 2 or permutations.shape[1] != count or not len(permutations):
        raise ValueError('Each null must permute every class exactly once.')
    expected = t.arange(count, device=permutations.device).expand_as(permutations)
    if not t.equal(permutations.sort(-1).values, expected):
        raise ValueError('Each null must permute every class exactly once.')
    observed = cyclic_distance_agreement(class_points)
    null = t.stack([cyclic_distance_agreement(class_points[order]) for order in permutations])
    probability = (1 + (null >= observed - 1e-7).sum()) / (1 + len(null))
    return observed, null, probability
if MAIN:


    tests.test_calendar_order_null_is_distinct_from_class_identification(SimpleNamespace(**globals()))
    print("All tests passed!")

# %%

if MAIN:
    orders = t.stack([t.randperm(7, generator=generator) for _ in range(256)])
    observed, null, probability = cyclic_permutation_test(residual, orders)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].scatter(calendar[:, 0], calendar[:, 1], c=np.arange(7), cmap="tab10", vmin=0, vmax=9, s=80)
    for i, day in enumerate(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]):
        axes[0].annotate(day, calendar[i], xytext=(6, 5), textcoords="offset points")
    axes[0].set(
        title="Known chronological order",
        aspect="equal",
        xlabel="cosine coordinate",
        ylabel="sine coordinate",
    )
    axes[1].hist(null.numpy(), bins=20, color="#8a9eb2", label="Class-order permutations")
    axes[1].axvline(float(observed), color="#bf4c28", linewidth=2, label="Chronological order")
    axes[1].set(
        title=f"Identity survives every permutation; order does not\np = {float(probability):.4f}",
        xlabel="Correlation with chronological chord distances",
        ylabel="Null arrangements",
    )
    axes[1].legend(fontsize=8)
    fig.savefig(assets_dir / "geometry_calendar_order_null.png", dpi=160)
    plt.show()

# %%

def neighborhood_scores(high: t.Tensor, low: t.Tensor, k: int = 5):
    """Trustworthiness (intrusions) and continuity (missing true neighbors)."""
    count = len(high)
    if len(low) != count or not 1 <= k < count / 2:
        raise ValueError("Use aligned rows and 1 <= k < n/2.")

    def ranks(points):
        distances = t.cdist(points, points)
        distances.fill_diagonal_(float("inf"))
        order = distances.argsort(dim=-1, stable=True)
        rank = t.empty_like(order)
        rank.scatter_(1, order, t.arange(1, count + 1, device=points.device).expand(count, -1))
        return rank

    original, projected = ranks(high), ranks(low)
    intrusion = t.where(projected <= k, (original - k).clamp_min(0), 0).sum()
    missing = t.where(original <= k, (projected - k).clamp_min(0), 0).sum()
    factor = 2 / (count * k * (2 * count - 3 * k - 1))
    return 1 - factor * intrusion, 1 - factor * missing
if MAIN:


    tests.test_neighborhood_ranks_match_independent_library_and_duality(SimpleNamespace(**globals()))
    print("All tests passed!")

# %%

def fit_procrustes(source: t.Tensor, target: t.Tensor):
    """Fit centered row-vector orthogonal alignment; reflections are permitted."""
    if source.shape != target.shape or source.ndim != 2:
        raise ValueError("Align the same observations in equal-dimensional spaces.")
    source_mean, target_mean = source.mean(0), target.mean(0)
    u, _, vh = t.linalg.svd((source - source_mean).T @ (target - target_mean))
    return source_mean, target_mean, u @ vh
if MAIN:


    tests.test_procrustes_recovers_rotation_reflection_and_translation(SimpleNamespace(**globals()))
    print("All tests passed!")

# %%

def replace_coordinates(activations: t.Tensor, mean: t.Tensor, basis: t.Tensor,
                        weights: t.Tensor, bias: t.Tensor, target: t.Tensor):
    """Minimum-L2 change satisfying the fitted PCA-coordinate probe.

The PCA basis must have orthonormal columns, and the probe must have full
column rank. Nothing outside the probe's column space is changed. Satisfying
this equation alone does not establish the model uses these coordinates.
"""
    if basis.ndim != 2 or weights.ndim != 2 or basis.shape[1] != weights.shape[0]:
        raise ValueError('Basis [width,k] and weights [k,coordinates] must align.')
    if not t.allclose(basis.T @ basis, t.eye(basis.shape[1], device=basis.device, dtype=basis.dtype), atol=1e-5, rtol=1e-5):
        raise ValueError('Minimum-norm coordinates require an orthonormal PCA basis.')
    if t.linalg.matrix_rank(weights) != weights.shape[1]:
        raise ValueError('Every requested coordinate must be independently controllable.')
    prediction = (activations - mean) @ basis @ weights + bias
    correction = (target - prediction) @ t.linalg.pinv(weights) @ basis.T
    return activations + correction


def replace_subspace(activations: t.Tensor, donor: t.Tensor, basis: t.Tensor):
    """Copy only donor coordinates in an orthonormal column basis."""
    if not t.allclose(basis.T @ basis, t.eye(basis.shape[1], device=basis.device, dtype=basis.dtype), atol=1e-5, rtol=1e-5):
        raise ValueError('A donor subspace needs an orthonormal column basis.')
    return activations + (donor - activations) @ basis @ basis.T
if MAIN:


    tests.test_minimum_norm_coordinate_edit_matches_constraint_solution(SimpleNamespace(**globals()))
    tests.test_donor_subspace_does_not_copy_complement(SimpleNamespace(**globals()))
    tests.test_exact_ring_plane_changes_answer_and_preserves_nuisance(SimpleNamespace(**globals()))
    print("All tests passed!")

# %%

if MAIN:
    mean, basis, _ = fit_pca(residual, 2)
    probe_mean, weights, bias = fit_ridge_coordinates((residual - mean) @ basis, calendar)
    bias = bias - probe_mean @ weights
    receiver = residual + 19 * rotation[:, 2]
    desired = circular_coordinates((t.arange(7, dtype=t.float64) + 3) % 7, 7)
    patched = replace_coordinates(receiver, mean, basis, weights, bias, desired)
    wrong_plane = replace_subspace(receiver, residual.roll(-3, 0), rotation[:, 2:4])


    # This exact model reads calendar phase in one head and an unrelated scalar in another.
    def calendar_logits(h):
        return (h @ rotation[:, :2]) @ calendar.T


    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    for ax, h, title in zip(
        axes,
        [receiver, patched, wrong_plane],
        ["Original residual", "Discovered calendar plane: +3 days", "Orthogonal rank-two control"],
        strict=True,
    ):
        image = ax.imshow(calendar_logits(h).numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
        ax.grid(False)
        ax.set(
            title=title, xlabel="Answer class", ylabel="Original day", xticks=range(7), yticks=range(7)
        )
        for i, answer in enumerate(calendar_logits(h).argmax(-1)):
            ax.text(int(answer), i, "×", ha="center", va="center", fontsize=16, color="white")
    fig.colorbar(image, ax=axes, shrink=0.8, label="Exact-model answer logit")
    fig.savefig(assets_dir / "geometry_plane_intervention.png", dpi=160)
    plt.show()
    print(
        "Unrelated coordinate after the calendar edit:",
        (patched @ rotation[:, 2]).round(decimals=6).tolist(),
    )

# %%

if MAIN:
    PLAY_ANGLE_DEGREES = 155
    PLAY_RADIUS = 1.0
    phase = t.tensor(PLAY_ANGLE_DEGREES * np.pi / 180, dtype=t.float64)
    target = PLAY_RADIUS * t.stack((phase.cos(), phase.sin()))
    played = replace_coordinates(receiver[:1], mean, basis, weights, bias, target)
    probabilities = calendar_logits(played).softmax(-1)[0]
    plt.figure(figsize=(8, 3.3))
    positions = np.arange(7)
    plt.bar(
        positions - 0.22,
        calendar_logits(receiver[:1]).softmax(-1)[0].numpy(),
        width=0.22,
        label="Original",
        color="#8a9eb2",
    )
    plt.bar(positions, probabilities.numpy(), width=0.22, label="Coordinate edit", color="#bf4c28")
    plt.bar(
        positions + 0.22,
        calendar_logits(wrong_plane[:1]).softmax(-1)[0].numpy(),
        width=0.22,
        label="Orthogonal-plane control",
        color="#487d9b",
    )
    plt.legend(fontsize=8)
    plt.xticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    plt.ylabel("Exact-model answer probability")
    plt.title(
        f"Angle {PLAY_ANGLE_DEGREES}° | radius {PLAY_RADIUS} | unrelated coordinate {float(played @ rotation[:, 2]):.2f}"
    )
    plt.tight_layout()
    plt.savefig(assets_dir / "geometry_angle_play.png", dpi=160)
    plt.show()

# %%

def gather_token_positions(hidden: t.Tensor, positions: t.Tensor, attention_mask: t.Tensor):
    """Read one explicitly aligned real token per row; reject padding positions."""
    batch, length = hidden.shape[:2]
    if positions.shape != (batch,) or attention_mask.shape != (batch, length):
        raise ValueError('Supply one position per row and the matching padding mask.')
    if (positions < 0).any() or (positions >= length).any():
        raise ValueError('Selected token position is outside the sequence.')
    row = t.arange(batch, device=hidden.device)
    if not attention_mask[row, positions].bool().all():
        raise ValueError('Never treat padding as a semantic token.')
    return hidden[row, positions]


def forward_with_residual_replacement(model, block, encoded, positions, replacement):
    """Replace the input residual to one native decoder block for one forward pass.

`positions` names the semantic token independently in every padded row. The
hook always gets removed, even if the model raises; caller controls inference
and model mode. Returning logits keeps scoring choices visible to the learner.
"""
    def hook(module, args):
        hidden = args[0]
        current = gather_token_positions(hidden, positions, encoded['attention_mask'])
        if current.shape != replacement.shape:
            raise ValueError('Replacement must match [batch, residual width].')
        modified = hidden.clone()
        modified[t.arange(len(hidden), device=hidden.device), positions] = replacement.to(hidden)
        return (modified, *args[1:])
    handle = block.register_forward_pre_hook(hook)
    try:
        return model(**encoded, use_cache=False).logits
    finally:
        handle.remove()
if MAIN:


    tests.test_semantic_positions_respect_both_padding_sides_and_cleanup_hooks(SimpleNamespace(**globals()))
    print("All tests passed!")

# %%

def mean_ablate_coordinates(mean: t.Tensor, basis: t.Tensor, weights: t.Tensor,
                           bias: t.Tensor, target: t.Tensor):
    """Set fitted coordinates; replace the entire complement by its training mean.

Unlike replace_coordinates, this deliberately discards receiver information
outside the fitted plane. Compare against mean ablation alone and a pure edit.
"""
    receiver = mean.expand(*target.shape[:-1], mean.shape[-1])
    return replace_coordinates(receiver, mean, basis, weights, bias, target)


def author_raw_circle_replacement(mean: t.Tensor, basis: t.Tensor,
                                  weights: t.Tensor, target: t.Tensor):
    """Literal row-vector QR formula in the paper authors' intervention code.

Reference: JoshEngels/MultiDimensionalFeatures, commit ff8c7eee8f13d78805f590ced6f3fc2227dd1daa,
intervention/circle_finding_utils.py: patch_in_subspace_activations (MIT).
This sets raw plane coordinates and mean-ablates the complement. It is not
the affine centered-coordinate constraint used in mean_ablate_coordinates.
"""
    if t.linalg.matrix_rank(weights) != weights.shape[1]:
        raise ValueError('The circular probe needs independent coordinate columns.')
    q, r = t.linalg.qr(weights, mode='reduced')
    plane = basis @ q
    return mean + target @ t.linalg.inv(r) @ plane.T - mean @ plane @ plane.T
if MAIN:


    tests.test_mean_ablation_and_literal_author_coordinates_answer_different_questions(SimpleNamespace(**globals()))
    print("All tests passed!")

# %%

def capture_residual_positions(model, boundaries, encoded, positions):
    """Capture one semantic position at each supplied native residual boundary."""
    saved = {}
    handles = []

    def make_hook(index):
        def hook(module, args):
            saved[index] = (
                gather_token_positions(args[0], positions, encoded["attention_mask"])
                .detach()
                .float()
                .cpu()
            )

        return hook

    try:
        for index, boundary in enumerate(boundaries):
            handles.append(boundary.register_forward_pre_hook(make_hook(index)))
        logits = model(**encoded, use_cache=False).logits
        if set(saved) != set(range(len(boundaries))):
            raise ValueError("Not every requested residual boundary ran.")
        return logits, t.stack([saved[i] for i in range(len(boundaries))], dim=1)
    finally:
        for handle in handles:
            handle.remove()
if MAIN:


    tests.test_native_capture_preserves_baseline_and_cleans_up_on_failure(SimpleNamespace(**globals()))
    print("All tests passed!")

# %%

def positions_from_spans(offset_mapping: t.Tensor, attention_mask: t.Tensor, spans: t.Tensor):
    """Locate a whole, single-token semantic span using actual tokenizer offsets."""
    left, right = offset_mapping.unbind(-1)
    overlaps = (left < spans[:, 1, None]) & (right > spans[:, 0, None]) & attention_mask.bool()
    if not overlaps.sum(-1).eq(1).all():
        raise ValueError("Each semantic word must overlap exactly one real token.")
    positions = overlaps.long().argmax(-1)
    row = t.arange(len(spans), device=spans.device)
    if ((left[row, positions] > spans[:, 0]) | (right[row, positions] < spans[:, 1])).any():
        raise ValueError("The selected token does not contain the whole semantic word.")
    return positions


def score_calendar_logits(
    logits: t.Tensor, original_ids: t.Tensor, desired_ids: t.Tensor, calendar_ids: t.Tensor
):
    """Keep full-vocabulary correctness separate from the restricted-choice task."""
    if (
        logits.ndim != 2
        or original_ids.shape != (len(logits),)
        or desired_ids.shape != (len(logits),)
    ):
        raise ValueError("Every model row needs an original and a desired answer token.")
    if not t.isfinite(logits).all():
        raise ValueError("Nonfinite model logits cannot support an answer claim.")
    row = t.arange(len(logits), device=logits.device)
    top = logits.argmax(-1)
    restricted = calendar_ids[logits[:, calendar_ids].argmax(-1)]
    return {
        "top_token_ids": top,
        "restricted_token_ids": restricted,
        "original_full_correct": top.eq(original_ids),
        "desired_full_correct": top.eq(desired_ids),
        "original_restricted_correct": restricted.eq(original_ids),
        "desired_restricted_correct": restricted.eq(desired_ids),
        "desired_minus_original_logit": logits[row, desired_ids] - logits[row, original_ids],
    }
if MAIN:


    tests.test_token_offsets_do_not_assume_identical_prompt_lengths(SimpleNamespace(**globals()))
    tests.test_restricted_calendar_success_does_not_hide_full_vocabulary_failure(
        SimpleNamespace(**globals())
    )
    print("All tests passed!")

# %%

import gzip
import hashlib
import json
from chapter11_representation_geometry.exercises.part1_pca_svd_geometry_controls.geometry_data import (
    NAMES,
    calendar_rows,
    paired_donor_indices,
    load_protocol,
)

if MAIN:
    protocol = load_protocol()
    RUN_REAL_MODELS = os.environ.get("ARENA_GEOMETRY_SCOPE", "full") == "full"
    print(
        "Training/development-selected layers:",
        {family: entry["layer_input"] for family, entry in protocol["selected"].items()},
    )
    print(
        "Final prompts:",
        protocol["final_base_rows"],
        "| counterfactual pairs:",
        protocol["all_donor_pairs"],
    )
    print("Random-control draws:", protocol["null_draws"])

# %%

def encode_calendar_batch(tokenizer, rows, device):
    encoded = tokenizer(
        [row["prompt"] for row in rows],
        padding=True,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offsets = encoded.pop("offset_mapping")
    positions = positions_from_spans(
        offsets, encoded["attention_mask"], t.tensor([row["source_span"] for row in rows])
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    if not encoded["attention_mask"][:, -1].bool().all():
        raise ValueError(
            "Use left padding so the last model position is an actual prediction position."
        )
    return encoded, positions.to(device)

# %%

@t.inference_mode()
def capture_calendar_rows(model, tokenizer, rows, calendar_ids, batch_size=32):
    """All residual boundaries, reduced answer logits and full-vocabulary winners."""
    device = next(model.parameters()).device
    boundaries = [*model.model.layers, model.model.norm]
    hidden, logits, top = [], [], []
    for start in range(0, len(rows), batch_size):
        encoded, positions = encode_calendar_batch(
            tokenizer, rows[start : start + batch_size], device
        )
        out, acts = capture_residual_positions(model, boundaries, encoded, positions)
        last = out[:, -1].float()
        hidden.append(acts)
        logits.append(last[:, calendar_ids].cpu())
        top.append(last.argmax(-1).cpu())
    return {
        "activations": t.cat(hidden),
        "calendar_logits": t.cat(logits),
        "top_token_ids": t.cat(top),
    }

# %%

@t.inference_mode()
def measure_calendar_replacements(
    model, tokenizer, rows, calendar_ids, layer, replacements, batch_size=32
):
    """Same ordered batches as capture; no outcome-dependent example filtering."""
    device = next(model.parameters()).device
    boundary = model.model.layers[layer] if layer < len(model.model.layers) else model.model.norm
    logits, top = [], []
    for start in range(0, len(rows), batch_size):
        encoded, positions = encode_calendar_batch(
            tokenizer, rows[start : start + batch_size], device
        )
        out = forward_with_residual_replacement(
            model, boundary, encoded, positions, replacements[start : start + batch_size]
        )
        last = out[:, -1].float()
        logits.append(last[:, calendar_ids].cpu())
        top.append(last.argmax(-1).cpu())
    return {"calendar_logits": t.cat(logits), "top_token_ids": t.cat(top)}

# %%

def calendar_outcome_rows(rows, measured, calendar_ids, tokenizer, shift, condition, draw=None):
    """Retain all calendar logits and full-vocabulary winners for independent scoring."""
    count = len(calendar_ids)
    logits = measured["calendar_logits"]
    top = measured["top_token_ids"]
    if logits.shape != (len(rows), count) or top.shape != (len(rows),):
        raise ValueError("Measured predictions must contain every requested row exactly once.")
    original = t.tensor([row["answer"] for row in rows])
    desired = (original + shift) % count
    # The pure tensor scorer handles both vocabularies. The compact archive retains
    # full-vocabulary winners separately, so a calendar-only argmax cannot hide them.
    scores = score_calendar_logits(logits, original, desired, t.arange(count))
    result = []
    for i, row in enumerate(rows):
        result.append(
            {
                "row_id": row["id"],
                "family": row["family"],
                "template_index": row["template_index"],
                "start": row["start"],
                "offset": row["offset"],
                "shift": shift,
                "condition": condition,
                "draw": draw,
                "original_answer": int(original[i]),
                "desired_answer": int(desired[i]),
                "restricted_answer": int(scores["restricted_token_ids"][i]),
                "top_token_id": int(top[i]),
                "top_token_text": tokenizer.decode(int(top[i])),
                "calendar_logits": logits[i].tolist(),
                "desired_minus_original_logit": float(scores["desired_minus_original_logit"][i]),
                "original_full_correct": bool(top[i] == calendar_ids[original[i]]),
                "desired_full_correct": bool(top[i] == calendar_ids[desired[i]]),
            }
        )
    return result

# %%

def fit_calendar_probe(activations, labels, period, components=5):
    mean, basis, variance = fit_pca(activations, components)
    coordinates = circular_coordinates(labels.to(activations.dtype), period)
    zmean, weights, bias = fit_ridge_coordinates((activations - mean) @ basis, coordinates)
    return {
        "mean": mean,
        "basis": basis,
        "weights": weights,
        "bias": bias - zmean @ weights,
        "variance": variance,
    }

# %%

def development_calendar_sweep(
    model, tokenizer, train_rows, dev_rows, train_capture, dev_capture, calendar_ids, batch_size=32
):
    """Recompute every development boundary; the published final choice stays frozen."""
    count = len(calendar_ids)
    train_labels = t.tensor([row["start"] for row in train_rows])
    dev_labels = t.tensor([row["start"] for row in dev_rows])
    targets = circular_coordinates(((dev_labels + 2) % count).float(), count)
    donors = t.tensor(paired_donor_indices(dev_rows, 2, count))
    original = t.tensor([row["answer"] for row in dev_rows])
    desired = (original + 2) % count
    indices = t.arange(len(dev_rows))
    baseline = dev_capture["calendar_logits"]
    baseline_margin = baseline[indices, desired] - baseline[indices, original]
    curves = []
    outcomes = []
    for boundary in range(train_capture["activations"].shape[1]):
        probe = fit_calendar_probe(
            train_capture["activations"][:, boundary], train_labels, count, 5
        )
        mean, basis, w, bias = (probe[key] for key in ("mean", "basis", "weights", "bias"))
        receiver = dev_capture["activations"][:, boundary]
        raw = author_raw_circle_replacement(mean, basis, w, targets)
        random = t.randn(receiver.shape, generator=t.Generator().manual_seed(1101 + boundary))
        random = (
            random / random.norm(dim=-1, keepdim=True) * (raw - receiver).norm(dim=-1, keepdim=True)
        )
        changes = {
            "pure_coordinate": replace_coordinates(receiver, mean, basis, w, bias, targets),
            "centered_mean_ablated": mean_ablate_coordinates(mean, basis, w, bias, targets),
            "author_raw_mean_ablated": raw,
            "full_donor": receiver[donors],
            "mean_only": mean.expand_as(receiver),
            "equal_norm_random_edit": receiver + random,
        }
        point = {"boundary": boundary, "conditions": {}}
        for name, value in changes.items():
            measured = measure_calendar_replacements(
                model, tokenizer, dev_rows, calendar_ids, boundary, value, batch_size
            )
            saved = calendar_outcome_rows(dev_rows, measured, calendar_ids, tokenizer, 2, name)
            for row in saved:
                row["layer_input"] = boundary
            outcomes.extend(saved)
            logits = measured["calendar_logits"]
            margin = logits[indices, desired] - logits[indices, original]
            point["conditions"][name] = {
                "desired_choice_accuracy": float(logits.argmax(-1).eq(desired).float().mean()),
                "mean_logit_effect": float((margin - baseline_margin).mean()),
                "desired_full_accuracy": sum(row["desired_full_correct"] for row in saved)
                / len(saved),
            }
        curves.append(point)
        if boundary % 8 == 0:
            print(dev_rows[0]["family"], "development boundary", boundary, "complete", flush=True)
    return curves, outcomes

# %%

def run_calendar_study(model, tokenizer, protocol, output_directory):
    """Execute the frozen train/development/final experiment, keeping every control."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=False)
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    batch_size = protocol["batch_size"]
    result = {
        "protocol": protocol,
        "tasks": {},
        "outcomes": [],
        "model_forward_rows": 0,
        "nulls": [],
    }
    tensors = {}

    def checkpoint():
        payload = json.dumps(result, separators=(",", ":")).encode()
        (output / "measured_study.json.gz").write_bytes(gzip.compress(payload, mtime=0))

    for family_index, (family, names) in enumerate(NAMES.items()):
        count = len(names)
        tokens = [tokenizer.encode(name, add_special_tokens=False) for name in names]
        if not all(len(ids) == 1 for ids in tokens):
            raise ValueError("Every answer must be one actual model token.")
        choice_ids = t.tensor([ids[0] for ids in tokens])
        if len(choice_ids.unique()) != count:
            raise ValueError("Calendar classes must have distinct answer tokens.")
        rows = {
            split: calendar_rows(family, split, protocol)
            for split in ("train", "development", "final")
        }
        prompt_inventory = [row["prompt"] for split in rows.values() for row in split]
        if len(set(prompt_inventory)) != len(prompt_inventory):
            raise ValueError("Prompt splits overlap.")
        captured = {
            split: capture_calendar_rows(model, tokenizer, rows[split], choice_ids, batch_size)
            for split in ("train", "development")
        }
        development_curves, development_outcomes = development_calendar_sweep(
            model,
            tokenizer,
            rows["train"],
            rows["development"],
            captured["train"],
            captured["development"],
            choice_ids,
            batch_size,
        )
        # These fresh curves remain visible; they do not change the already-frozen layer choice.
        captured["final"] = capture_calendar_rows(
            model, tokenizer, rows["final"], choice_ids, batch_size
        )
        result["outcomes"].extend(development_outcomes)
        result["model_forward_rows"] += sum(len(items) for items in rows.values()) + len(
            development_outcomes
        )
        train_labels = t.tensor([r["start"] for r in rows["train"]])
        final_labels = t.tensor([r["start"] for r in rows["final"]])
        layer = protocol["selected"][family]["layer_input"]
        probe = fit_calendar_probe(
            captured["train"]["activations"][:, layer], train_labels, count, 5
        )
        mean, basis, w, bias = (probe[key] for key in ("mean", "basis", "weights", "bias"))
        final = captured["final"]["activations"][:, layer]
        final_targets = circular_coordinates(final_labels.float(), count)
        prediction = (final - mean) @ basis @ w + bias
        curves = []
        permutations = t.stack(
            [
                t.randperm(
                    count,
                    generator=t.Generator().manual_seed(
                        protocol["circular_order_null_seed"] + family_index * 1000 + draw
                    ),
                )
                for draw in range(protocol["circular_order_null_draws"])
            ]
        )
        for boundary in range(captured["train"]["activations"].shape[1]):
            train_acts = captured["train"]["activations"][:, boundary]
            other = fit_calendar_probe(train_acts, train_labels, count, 5)
            pred = (captured["final"]["activations"][:, boundary] - other["mean"]) @ other[
                "basis"
            ] @ other["weights"] + other["bias"]
            centroids = t.stack([train_acts[train_labels.eq(i)].mean(0) for i in range(count)])
            xy = (centroids - other["mean"]) @ other["basis"][:, :2]
            actual, null, p = cyclic_permutation_test(xy, permutations)
            curves.append(
                {
                    "boundary": boundary,
                    "final_coordinate_r2": float(coordinate_r2(final_targets, pred)),
                    "final_angular_accuracy": float(
                        angular_predictions(pred, count).eq(final_labels).float().mean()
                    ),
                    "training_pca2_order_agreement": float(actual),
                    "order_null": null.tolist(),
                    "order_p": float(p),
                    "training_variance": other["variance"].tolist(),
                }
            )
        leave_out = []
        for heldout_class in range(count):
            keep = train_labels.ne(heldout_class)
            target_rows = final_labels.eq(heldout_class)
            other = fit_calendar_probe(
                captured["train"]["activations"][keep, layer], train_labels[keep], count, 5
            )
            pred = (final[target_rows] - other["mean"]) @ other["basis"] @ other["weights"] + other[
                "bias"
            ]
            leave_out.append(
                {
                    "class": heldout_class,
                    "rows": int(target_rows.sum()),
                    "predicted_coordinates": pred.tolist(),
                    "angular_correct": int(
                        angular_predictions(pred, count).eq(heldout_class).sum()
                    ),
                    "mean_coordinate_squared_error": float(
                        (pred - final_targets[target_rows]).square().sum(-1).mean()
                    ),
                }
            )
        task = {
            "names": list(names),
            "calendar_token_ids": choice_ids.tolist(),
            "rows": rows,
            "layer_input": layer,
            "layer_sweep": curves,
            "leave_one_class_out": leave_out,
            "final_coordinate_r2": float(coordinate_r2(final_targets, prediction)),
            "final_coordinate_predictions": prediction.tolist(),
            "final_angular_accuracy": float(
                angular_predictions(prediction, count).eq(final_labels).float().mean()
            ),
            "training_pca_variance": probe["variance"].tolist(),
            "development_interventions": development_curves,
        }
        result["tasks"][family] = task
        shared = {"baseline": captured["final"]}
        for name, value in [("no_op", final), ("mean_only", mean.expand_as(final))]:
            shared[name] = measure_calendar_replacements(
                model, tokenizer, rows["final"], choice_ids, layer, value, batch_size
            )
            result["model_forward_rows"] += len(final)
        if not t.equal(
            shared["baseline"]["calendar_logits"], shared["no_op"]["calendar_logits"]
        ) or not t.equal(shared["baseline"]["top_token_ids"], shared["no_op"]["top_token_ids"]):
            raise ValueError("No-op residual replacement changed the actual baseline.")
        task["no_op_max_logit_error"] = 0.0
        for split in ("train", "development"):
            result["outcomes"].extend(
                calendar_outcome_rows(
                    rows[split], captured[split], choice_ids, tokenizer, 2, split + "_baseline"
                )
            )
        primary = None
        for shift in range(1, count):
            donors = t.tensor(paired_donor_indices(rows["final"], shift, count))
            target = circular_coordinates(((final_labels + shift) % count).float(), count)
            replacements = {
                "pure_coordinate": replace_coordinates(final, mean, basis, w, bias, target),
                "centered_mean_ablated": mean_ablate_coordinates(mean, basis, w, bias, target),
                "author_raw_mean_ablated": author_raw_circle_replacement(mean, basis, w, target),
                "full_donor": final[donors],
                "pca5_donor": replace_subspace(final, final[donors], basis),
            }
            for name, value in replacements.items():
                measured = measure_calendar_replacements(
                    model, tokenizer, rows["final"], choice_ids, layer, value, batch_size
                )
                result["model_forward_rows"] += len(final)
                result["outcomes"].extend(
                    calendar_outcome_rows(
                        rows["final"], measured, choice_ids, tokenizer, shift, name
                    )
                )
            for name, measured in shared.items():
                result["outcomes"].extend(
                    calendar_outcome_rows(
                        rows["final"], measured, choice_ids, tokenizer, shift, name
                    )
                )
            if shift == protocol["primary_control_donor_shift"]:
                primary = {
                    "donors": donors,
                    "target": target,
                    "delta_norm": (replacements["author_raw_mean_ablated"] - final).norm(
                        dim=-1, keepdim=True
                    ),
                }
            checkpoint()
            print(family, "all-donor shift", shift, "complete", flush=True)
        shift = protocol["primary_control_donor_shift"]
        for draw in range(protocol["null_draws"]):
            seed = protocol["null_seed"] + family_index * 10000 + draw
            random_plane = t.linalg.qr(
                t.randn(final.shape[1], 2, generator=t.Generator().manual_seed(seed)),
                mode="reduced",
            ).Q
            noise = t.randn(final.shape, generator=t.Generator().manual_seed(seed + 1000))
            noise = noise / noise.norm(dim=-1, keepdim=True) * primary["delta_norm"]
            permutation = t.randperm(count, generator=t.Generator().manual_seed(seed + 2000))
            perm_targets = circular_coordinates(permutation[train_labels].float(), count)
            zmean, perm_w, perm_b = fit_ridge_coordinates(
                (captured["train"]["activations"][:, layer] - mean) @ basis, perm_targets
            )
            desired_perm = circular_coordinates(
                permutation[(final_labels + shift) % count].float(), count
            )
            changes = {
                "same_rank_random_donor_plane": replace_subspace(
                    final, final[primary["donors"]], random_plane
                ),
                "equal_norm_random_edit": final + noise,
                "class_order_permuted_circular_probe": author_raw_circle_replacement(
                    mean, basis, perm_w, desired_perm
                ),
            }
            result["nulls"].append(
                {
                    "family": family,
                    "draw": draw,
                    "seed": seed,
                    "class_order_permutation": permutation.tolist(),
                    "random_plane_sha256": hashlib.sha256(
                        random_plane.numpy().tobytes()
                    ).hexdigest(),
                }
            )
            for name, value in changes.items():
                measured = measure_calendar_replacements(
                    model, tokenizer, rows["final"], choice_ids, layer, value, batch_size
                )
                result["model_forward_rows"] += len(final)
                result["outcomes"].extend(
                    calendar_outcome_rows(
                        rows["final"], measured, choice_ids, tokenizer, shift, name, draw
                    )
                )
            if draw % 8 == 7:
                checkpoint()
                print(family, "null draw", draw + 1, "complete", flush=True)
        tensors[family] = {
            "captured": captured,
            "probe": probe,
            "rows": rows,
            "calendar_token_ids": choice_ids,
        }
        t.save(tensors, output / "actual_activations_and_probes.pt")
        checkpoint()
    result["complete"] = True
    result["retained_outcomes"] = len(result["outcomes"])
    result["final_base_rows"] = sum(len(task["rows"]["final"]) for task in result["tasks"].values())
    result["all_donor_pairs"] = sum(
        len(task["rows"]["final"]) * (len(task["names"]) - 1) for task in result["tasks"].values()
    )
    if (
        result["final_base_rows"] != protocol["final_base_rows"]
        or result["all_donor_pairs"] != protocol["all_donor_pairs"]
    ):
        raise ValueError("The frozen data inventory did not execute completely.")
    checkpoint()
    return result, tensors

# %%

if MAIN:
    if RUN_REAL_MODELS:
        import time
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_cache = os.environ.get("ARENA_GEOMETRY_MODEL_CACHE")
        tokenizer = AutoTokenizer.from_pretrained(
            protocol["model_id"], revision=protocol["revision"], cache_dir=model_cache
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model = (
            AutoModelForCausalLM.from_pretrained(
                protocol["model_id"],
                revision=protocol["revision"],
                dtype=t.bfloat16,
                attn_implementation=protocol["attention"],
                cache_dir=model_cache,
            )
            .to("cuda")
            .eval()
        )
        t.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        study_result, study_tensors = run_calendar_study(
            model, tokenizer, protocol, assets_dir.parent / "experiments"
        )
        print(f"Actual study runtime: {time.perf_counter() - started:.1f} s")
        print("Retained outcomes:", study_result["retained_outcomes"])
    else:
        print(
            "Numerical foundation mode: pretrained-model fitting, capture and interventions are skipped."
        )

# %%

if MAIN:
    if RUN_REAL_MODELS:
        import pandas as pd
        from IPython.display import display
        from chapter11_representation_geometry.exercises.part1_pca_svd_geometry_controls import (
            geometry_plotting as plots,
        )

        outcome_frame = pd.DataFrame(study_result["outcomes"])
        outcome_frame["desired_correct"] = outcome_frame.restricted_answer.eq(
            outcome_frame.desired_answer
        )
        outcome_frame["original_correct"] = outcome_frame.restricted_answer.eq(
            outcome_frame.original_answer
        )
        final_frame = outcome_frame[outcome_frame.row_id.str.startswith("final/")].copy()
        baseline_frame = outcome_frame[
            outcome_frame.condition.isin(["train_baseline", "development_baseline", "baseline"])
            & outcome_frame["shift"].eq(protocol["primary_control_donor_shift"])
        ].copy()
        baseline_frame["split"] = baseline_frame.row_id.str.split("/").str[0]
        display(
            baseline_frame.groupby(["family", "split"])
            .agg(
                prompts=("row_id", "size"),
                calendar_accuracy=("original_correct", "mean"),
                full_vocabulary_accuracy=("original_full_correct", "mean"),
            )
            .round(3)
        )

# %%

if MAIN:
    if RUN_REAL_MODELS:
        projections = {}
        for family, task in study_tensors.items():
            layer = protocol["selected"][family]["layer_input"]
            probe = task["probe"]
            train = task["captured"]["train"]["activations"][:, layer]
            final = task["captured"]["final"]["activations"][:, layer]
            train_z = (train - probe["mean"]) @ probe["basis"]
            final_z = (final - probe["mean"]) @ probe["basis"]
            projections[family] = {
                "names": NAMES[family],
                "train_pca": train_z[:, :2].numpy(),
                "final_pca": final_z[:, :2].numpy(),
                "train_probe": (train_z @ probe["weights"] + probe["bias"]).numpy(),
                "final_probe": (final_z @ probe["weights"] + probe["bias"]).numpy(),
                "train_labels": [r["start"] for r in task["rows"]["train"]],
                "final_labels": [r["start"] for r in task["rows"]["final"]],
            }
        plots.plot_calendar_projections(projections, assets_dir)
        plots.plot_calendar_layers(study_result["tasks"], assets_dir)
        display(
            pd.DataFrame(
                [
                    {
                        "family": family,
                        "new_template_R2": task["final_coordinate_r2"],
                        "new_template_angle_accuracy": task["final_angular_accuracy"],
                        "unseen_class_angle_accuracy": sum(
                            row["angular_correct"] for row in task["leave_one_class_out"]
                        )
                        / sum(row["rows"] for row in task["leave_one_class_out"]),
                    }
                    for family, task in study_result["tasks"].items()
                ]
            ).round(3)
        )

# %%

if MAIN:
    if RUN_REAL_MODELS:
        primary = final_frame[final_frame["shift"].eq(protocol["primary_control_donor_shift"])]
        aggregate = dict(
            desired_accuracy=("desired_correct", "mean"),
            desired_full_accuracy=("desired_full_correct", "mean"),
            mean_margin=("desired_minus_original_logit", "mean"),
            prompts=("row_id", "size"),
        )
        main_summary = (
            primary[primary.draw.isna()].groupby(["family", "condition"]).agg(**aggregate).reset_index()
        )
        null_summary = (
            primary[primary.draw.notna()]
            .groupby(["family", "condition", "draw"])
            .agg(**aggregate)
            .reset_index()
        )
        plots.plot_calendar_final_controls(main_summary, null_summary, assets_dir)
        plots.plot_calendar_all_donors(
            final_frame[final_frame.draw.isna()], study_result["tasks"], assets_dir
        )
        plots.plot_calendar_examples(final_frame, study_result["tasks"], assets_dir)
        display(
            main_summary[
                ["family", "condition", "prompts", "desired_accuracy", "desired_full_accuracy"]
            ].round(3)
        )

# %%

def run_geometry_projection_sweep(study_tensors, protocol, output_directory):
    """Inductive UMAP and explicitly transductive t-SNE, including every null fit."""
    import umap
    from sklearn.manifold import TSNE

    records = []
    references = {}
    for family_index, (family, task) in enumerate(study_tensors.items()):
        layer = protocol["selected"][family]["layer_input"]
        actual_train = task["captured"]["train"]["activations"][:, layer]
        actual_final = task["captured"]["final"]["activations"][:, layer]
        ytrain = t.tensor([row["start"] for row in task["rows"]["train"]])
        yfinal = t.tensor([row["start"] for row in task["rows"]["final"]])
        mean = actual_train.mean(0)
        noise = t.randn(
            len(actual_train) + len(actual_final),
            actual_train.shape[-1],
            generator=t.Generator().manual_seed(3300 + family_index),
        )
        actual = t.cat([actual_train, actual_final]) - mean
        noise = noise / noise.norm(dim=-1, keepdim=True) * actual.norm(dim=-1, keepdim=True)
        for control, train, final in [
            ("real", actual_train, actual_final),
            ("independent_noise", noise[: len(actual_train)], noise[len(actual_train) :]),
        ]:
            center, axes, _ = fit_pca(train, min(20, len(train) - 1))
            train_input = ((train - center) @ axes).numpy()
            final_input = ((final - center) @ axes).numpy()
            high = t.cat([train, final])
            for reducer in ("umap", "tsne"):
                settings = (
                    protocol["umap_settings"]
                    if reducer == "umap"
                    else [{"perplexity": p} for p in protocol["tsne_perplexities"]]
                )
                for setting_index, setting in enumerate(settings):
                    for seed in protocol["reducer_seeds"]:
                        if reducer == "umap":
                            method = umap.UMAP(
                                n_components=2,
                                **setting,
                                metric="euclidean",
                                random_state=seed,
                                transform_seed=seed,
                                n_jobs=1,
                            )
                            train_low = t.tensor(method.fit_transform(train_input), dtype=t.float32)
                            final_low = t.tensor(method.transform(final_input), dtype=t.float32)
                            scope = "inductive: fit training only, transform final"
                        else:
                            joined = t.cat([t.tensor(train_input), t.tensor(final_input)]).numpy()
                            method = TSNE(
                                n_components=2,
                                **setting,
                                init=protocol["tsne_init"],
                                learning_rate="auto",
                                max_iter=protocol["tsne_iterations"],
                                random_state=seed,
                                metric="euclidean",
                                n_jobs=1,
                            )
                            low = t.tensor(method.fit_transform(joined), dtype=t.float32)
                            train_low, final_low = low[: len(train)], low[len(train) :]
                            scope = "transductive: unlabeled final points included in fitting"
                        prediction = knn_predictions(train_low, ytrain, final_low, 3)
                        shuffle = t.randperm(
                            len(ytrain),
                            generator=t.Generator().manual_seed(
                                4400 + family_index * 100 + seed * 10 + setting_index
                            ),
                        )
                        shuffled_prediction = knn_predictions(
                            train_low, ytrain[shuffle], final_low, 3
                        )
                        trust, continuity = neighborhood_scores(
                            high, t.cat([train_low, final_low]), 5
                        )
                        key = (family, control, reducer, setting_index)
                        if key not in references:
                            references[key] = (train_low.clone(), final_low.clone())
                        ref_train, ref_final = references[key]
                        source_mean, target_mean, rotation = fit_procrustes(train_low, ref_train)
                        aligned = (final_low - source_mean) @ rotation + target_mean
                        normalized_error = float(
                            (aligned - ref_final).square().mean().sqrt()
                            / ref_final.std().clamp_min(1e-8)
                        )
                        records.append(
                            {
                                "family": family,
                                "control": control,
                                "reducer": reducer,
                                "scope": scope,
                                "setting": setting,
                                "setting_index": setting_index,
                                "seed": seed,
                                "train_projection": train_low.tolist(),
                                "final_projection": final_low.tolist(),
                                "final_knn_accuracy": float(prediction.eq(yfinal).float().mean()),
                                "permuted_training_labels_accuracy": float(
                                    shuffled_prediction.eq(yfinal).float().mean()
                                ),
                                "trustworthiness": float(trust),
                                "continuity": float(continuity),
                                "procrustes_final_error_relative_to_seed0": normalized_error,
                                "final_predictions": prediction.tolist(),
                                "permuted_training_label_predictions": shuffled_prediction.tolist(),
                            }
                        )
                    print(
                        family, control, reducer, "setting", setting_index, "complete", flush=True
                    )
    path = Path(output_directory) / "projection_study.json.gz"
    path.write_bytes(gzip.compress(json.dumps(records, separators=(",", ":")).encode(), mtime=0))
    return records

# %%

if MAIN:
    if RUN_REAL_MODELS:
        projection_records = run_geometry_projection_sweep(
            study_tensors, protocol, assets_dir.parent / "experiments"
        )
        assert len(projection_records) == 120, (
            "Keep all families, controls, reducers, settings and seeds."
        )
        plots.plot_projection_sweep(projection_records, assets_dir)

# %%

if MAIN:
    projection_seed = 0
    projection_setting = 1
    if RUN_REAL_MODELS:
        plots.plot_projection_gallery(
            projection_records,
            study_tensors,
            assets_dir,
            seed=projection_seed,
            setting_index=projection_setting,
        )

# %%

def run_calendar_angle_play(model, tokenizer, study_tensors, protocol, output_directory):
    """A fixed source per family; sweep continuous coordinates with paired controls."""
    records = []
    for family_index, (family, task) in enumerate(study_tensors.items()):
        row = task["rows"]["final"][0]
        count = len(NAMES[family])
        probe = task["probe"]
        mean, basis, w, bias = (probe[key] for key in ("mean", "basis", "weights", "bias"))
        layer = protocol["selected"][family]["layer_input"]
        receiver = task["captured"]["final"]["activations"][0, layer]
        angles = (
            t.arange(count * protocol["angle_play"]["steps_per_class"])
            * 2
            * t.pi
            / (count * protocol["angle_play"]["steps_per_class"])
        )
        radii = t.tensor(protocol["angle_play"]["radii"])
        radius_grid, angle_grid = t.meshgrid(radii, angles, indexing="ij")
        target = radius_grid.flatten()[:, None] * t.stack(
            [angle_grid.flatten().cos(), angle_grid.flatten().sin()], -1
        )
        receivers = receiver.expand(len(target), -1)
        raw = author_raw_circle_replacement(mean, basis, w, target)
        # One fixed direction across the grid: only its matched magnitude changes.
        direction = t.randn(
            receiver.shape, generator=t.Generator().manual_seed(5500 + family_index)
        )
        direction = direction / direction.norm()
        replacements = {
            "pure_coordinate": replace_coordinates(receivers, mean, basis, w, bias, target),
            "author_raw_mean_ablated": raw,
            "equal_norm_random_edit": receivers
            + (raw - receivers).norm(dim=-1, keepdim=True) * direction,
        }
        shared = {}
        for name, value in [("baseline", receiver[None]), ("mean_only", mean[None])]:
            measured = measure_calendar_replacements(
                model,
                tokenizer,
                [row],
                task["calendar_token_ids"],
                layer,
                value,
                protocol["batch_size"],
            )
            shared[name] = {
                "calendar_logits": measured["calendar_logits"][0].tolist(),
                "top_token_id": int(measured["top_token_ids"][0]),
                "top_token_text": tokenizer.decode(int(measured["top_token_ids"][0])),
            }
        for condition, value in replacements.items():
            measured = measure_calendar_replacements(
                model,
                tokenizer,
                [row] * len(target),
                task["calendar_token_ids"],
                layer,
                value,
                protocol["batch_size"],
            )
            records.append(
                {
                    "family": family,
                    "row": row,
                    "condition": condition,
                    "layer_input": layer,
                    "radii": radii.tolist(),
                    "angles": angles.tolist(),
                    "shared": shared,
                    "calendar_logits": measured["calendar_logits"].tolist(),
                    "top_token_ids": measured["top_token_ids"].tolist(),
                    "top_token_text": [
                        tokenizer.decode(int(token)) for token in measured["top_token_ids"]
                    ],
                }
            )
        print(family, "continuous angle/radius play complete", flush=True)
    (Path(output_directory) / "angle_play.json.gz").write_bytes(
        gzip.compress(json.dumps(records, separators=(",", ":")).encode(), mtime=0)
    )
    return records

# %%

if MAIN:
    if RUN_REAL_MODELS:
        angle_records = run_calendar_angle_play(
            model, tokenizer, study_tensors, protocol, assets_dir.parent / "experiments"
        )
        plots.plot_angle_play(angle_records, assets_dir)

# %%

if MAIN:
    play_family = "weekday"  # 'weekday' or 'month'
    play_start = 0  # Monday / January
    play_offset = 1
    play_shift = 2
    play_radius = 1.0
    if RUN_REAL_MODELS:
        task = study_tensors[play_family]
        source_index = next(
            i
            for i, row in enumerate(task["rows"]["final"])
            if row["template_index"] == 0
            and row["start"] == play_start
            and row["offset"] == play_offset
        )
        source = task["rows"]["final"][source_index]
        layer = protocol["selected"][play_family]["layer_input"]
        receiver = task["captured"]["final"]["activations"][source_index : source_index + 1, layer]
        probe = task["probe"]
        mean, basis, weights, bias = (probe[key] for key in ("mean", "basis", "weights", "bias"))
        count = len(NAMES[play_family])
        target = play_radius * circular_coordinates(
            t.tensor([(play_start + play_shift) % count], dtype=t.float32), count
        )
        changes = {
            "baseline": receiver,
            "pure_coordinate": replace_coordinates(receiver, mean, basis, weights, bias, target),
            "author_raw_mean_ablated": author_raw_circle_replacement(mean, basis, weights, target),
            "mean_only": mean[None],
        }
        fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
        play_measurements = []
        for ax, (condition, replacement) in zip(axes, changes.items(), strict=True):
            measured = measure_calendar_replacements(
                model, tokenizer, [source], task["calendar_token_ids"], layer, replacement
            )
            probabilities = measured["calendar_logits"][0].softmax(-1).numpy()
            actual_token = tokenizer.decode(int(measured["top_token_ids"][0]))
            ax.bar(range(count), probabilities, color=plots.COLORS[condition])
            ax.set_xticks(
                range(count), [name[:3] for name in NAMES[play_family]], rotation=45, ha="right"
            )
            ax.set(
                ylim=(0, 1.02),
                ylabel="Calendar-only probability",
                title=f"{plots.LABELS[condition]}\nFull-vocabulary top: {actual_token!r}",
            )
            play_measurements.append(
                {
                    "condition": condition,
                    "calendar_logits": measured["calendar_logits"][0].tolist(),
                    "top_token_id": int(measured["top_token_ids"][0]),
                    "top_token_text": actual_token,
                }
            )
        desired = NAMES[play_family][(play_start + play_offset + play_shift) % count]
        fig.suptitle(
            f"{source['prompt']} | requested shift {play_shift:+d}, radius {play_radius:g} | desired answer: {desired}"
        )
        plots.finish(fig, assets_dir, "geometry_mistral_single_play.png")
        (assets_dir.parent / "experiments" / "single_play.json").write_text(
            json.dumps(
                {
                    "source": source,
                    "shift": play_shift,
                    "radius": play_radius,
                    "measurements": play_measurements,
                },
                indent=2,
            )
        )

# %%

if MAIN:
    if RUN_REAL_MODELS:
        runtime_record = {
            "gpu": t.cuda.get_device_name(),
            "peak_cuda_allocated_gib": t.cuda.max_memory_allocated() / 2**30,
            "elapsed_real_seconds": time.perf_counter() - started,
            "model_forward_rows": study_result["model_forward_rows"],
            "retained_outcomes": study_result["retained_outcomes"],
            "projection_fits": len(projection_records),
            "angle_play_forward_rows": sum(len(record["calendar_logits"]) for record in angle_records)
            + 4,
            "single_play_forward_rows": len(play_measurements),
        }
        (assets_dir.parent / "experiments" / "runtime.json").write_text(
            json.dumps(runtime_record, indent=2)
        )
        print(
            f"Finished on {runtime_record['gpu']}: {runtime_record['elapsed_real_seconds']:.1f} s, "
            f"peak CUDA allocation {runtime_record['peak_cuda_allocated_gib']:.2f} GiB."
        )

# %%

# Compatibility entry points for the preserved historical evidence tooling.
# They do not run on import and are separate from the visible notebook study.
import inspect
import tempfile

geometry_methods_tests = tests

def run_smoke_test(cpu=True):
    tested=[]
    for name,func in vars(geometry_methods_tests).items():
        if name.startswith('test_') and inspect.isfunction(func):func(SimpleNamespace(**globals()));tested.append(name)
    return {'tests_passed':True,'accepted':True,'test_count':len(tested),'claim_scope':'Numerical method oracles; real evidence is checked separately.'}


def run_gpu_test(max_vram_gb: float = 24.0):
    from chapter11_representation_geometry.exercises.part1_pca_svd_geometry_controls.geometry_cuda import reproduce
    here=Path(__file__).parent
    publication=json.loads((here/'expected_outputs/learner_execution.json').read_text())
    learner=Path(os.environ.get('ARENA_GEOMETRY_LEARNER_DIRECTORY',publication['learner_directory']))
    cached_manifest=os.environ.get('ARENA_GEOMETRY_REPRODUCTION_MANIFEST')
    if cached_manifest:
        from chapter11_representation_geometry.exercises.part1_pca_svd_geometry_controls.geometry_reproduction_cache import validate_cached_reproduction
        fresh,output=validate_cached_reproduction(learner,Path(cached_manifest))
    else:
        report_root=learner.parent / 'fresh_reproductions'
        report_root.mkdir(parents=True,exist_ok=True)
        output=Path(tempfile.mkdtemp(prefix='run-',dir=report_root))/'fresh_cuda'
        fresh=reproduce(learner,output)
    audit=fresh['learner_audit'];peak=max(fresh['peak_cuda_allocated_gib'],audit['runtime']['peak_cuda_allocated_gib'])
    if peak>max_vram_gb:raise ValueError(f'Actual CUDA peak {peak:.3f} GiB exceeds the requested budget.')
    return {'cuda_available':True,'gpu_name':fresh['gpu'],'peak_vram_gb':peak,'within_vram_budget':True,
            'actual_learner_passed':True,'own_answer_code_cells':audit['own_answer_code_cells'],'embedded_figures':audit['embedded_figures'],
            'final_prompts':audit['final_prompts'],'all_donor_pairs':audit['all_donor_pairs'],'retained_outcomes':audit['retained_outcomes'],
            'null_draws_per_kind':32,'projection_fits':audit['projection_fits'],'angle_play_rows':audit['angle_play_rows'],
            'fresh_compared_examples':fresh['compared_examples'],'fresh_condition_batches':fresh['condition_batches'],
            'fresh_max_logit_error':fresh['max_calendar_logit_error'],'fresh_top_token_mismatches':fresh['top_token_mismatches'],
            'fresh_elapsed_seconds':fresh['elapsed_seconds'],'fresh_output_directory':str(output),
            'reused_native_reproduction':bool(cached_manifest),
            'native_cuda_executed_this_report':not bool(cached_manifest),
            'reproduction_manifest_sha256':fresh.get('reproduction_manifest_sha256'),
            'reproduction_manifest_path':fresh.get('reproduction_manifest_path'),
            'reproduction_record_sha256':fresh.get('reproduction_record_sha256'),
            'reproduction_actual_rows_sha256':fresh.get('reproduction_actual_rows_sha256'),
            'final_used_for_selection':False,'tsne_is_transductive':True,'tasks':audit['tasks'],
            'paper_equivalent_full_replication':False}


def run_full_experiment(max_vram_gb: float = 24.0):
    return {'numerical':run_smoke_test(),'fresh_cuda':run_gpu_test(max_vram_gb)}

# %%
