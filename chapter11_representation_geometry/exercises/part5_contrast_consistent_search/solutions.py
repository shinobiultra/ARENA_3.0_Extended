# %%


MAIN = __name__ == "__main__"

# %%

import itertools

import json

import math

import os

import sys

from collections.abc import Sequence

from pathlib import Path

import matplotlib.pyplot as plt

import numpy as np

import torch

from torch import Tensor

from IPython import get_ipython

if MAIN:
    if get_ipython() is not None:
        get_ipython().run_line_magic("matplotlib", "inline")

root_dir = Path(__file__).resolve().parents[3]

if MAIN:
    if str(root_dir) not in sys.path:
        sys.path.append(str(root_dir))

from chapter11_representation_geometry.exercises.part5_contrast_consistent_search import ccs_methods_tests as tests

if MAIN:
    assets_dir = Path(os.environ.get("ARENA_CCS_OUTPUT_DIRECTORY", "ccs_outputs"))

if MAIN:
    assets_dir.mkdir(parents=True, exist_ok=True)

if MAIN:
    torch.set_num_threads(4)

if MAIN:
    plt.style.use("seaborn-v0_8-whitegrid")

def save_figure(figure, name):
    figure.savefig(assets_dir / name, dpi=160, bbox_inches="tight")
    plt.show()

def _check_pairs(plus: Tensor, minus: Tensor) -> None:
    if plus.ndim != 2 or plus.shape != minus.shape or min(plus.shape) == 0:
        raise ValueError("Paired activations must have matching nonempty (examples, features) shapes.")
    if plus.dtype != minus.dtype or plus.device != minus.device:
        raise ValueError("Both branches must have the same dtype and device.")
    if not plus.is_floating_point() or not torch.isfinite(plus).all() or not torch.isfinite(minus).all():
        raise ValueError("Activations must be finite floating-point tensors.")

def _check_scores_labels(scores: Tensor, labels: Tensor) -> None:
    if scores.ndim != 1 or scores.shape != labels.shape or len(scores) == 0:
        raise ValueError("Provide equally sized nonempty score and label vectors.")
    if not torch.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Scores must lie in [0, 1].")
    if ((labels != 0) & (labels != 1)).any():
        raise ValueError("Labels must be binary.")

# %%

if MAIN:
    target_sign, distractor_sign, content = torch.tensor(
        list(itertools.product((-1.0, 1.0), repeat=3)), dtype=torch.float64
    ).T

if MAIN:
    plus_raw = torch.stack((torch.full_like(target_sign, 8), target_sign, 3 * distractor_sign, content), dim=-1)

if MAIN:
    minus_raw = torch.stack((torch.full_like(target_sign, -8), -target_sign, -3 * distractor_sign, content), dim=-1)

if MAIN:
    target_labels = (target_sign > 0).long()

if MAIN:
    distractor_labels = (distractor_sign > 0).long()

if MAIN:
    print("Each branch:", tuple(plus_raw.shape), "— examples × features")

if MAIN:
    print("Plus mean:", plus_raw.mean(0).tolist())

if MAIN:
    print("Minus mean:", minus_raw.mean(0).tolist())

# %%

def fit_pair_centering(plus: Tensor, minus: Tensor) -> dict[str, Tensor]:
    """Fit separate branch means on training data; there is no scale normalization."""
    _check_pairs(plus, minus)
    return {"plus_mean": plus.detach().mean(0), "minus_mean": minus.detach().mean(0)}

def apply_pair_centering(
    plus: Tensor, minus: Tensor, centering: dict[str, Tensor]
) -> tuple[Tensor, Tensor]:
    """Apply fitted means to any split, without refitting on that split."""
    _check_pairs(plus, minus)
    if any(centering[key].shape != (plus.shape[1],) for key in ("plus_mean", "minus_mean")):
        raise ValueError("A training mean must have one value per feature.")
    return plus - centering["plus_mean"], minus - centering["minus_mean"]

if MAIN:
    tests.test_pair_centering(fit_pair_centering, apply_pair_centering)

if MAIN:
    print("Training centering and held-out shift checks passed!")

# %%

if MAIN:
    centering = fit_pair_centering(plus_raw, minus_raw)

if MAIN:
    plus, minus = apply_pair_centering(plus_raw, minus_raw, centering)

if MAIN:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))

if MAIN:
    for branch, matrix, marker in [("Plus", plus_raw, "o"), ("Minus", minus_raw, "s")]:
        axes[0].scatter(matrix[:, 0], matrix[:, 1], marker=marker, s=90, alpha=.7, label=branch)

if MAIN:
    axes[0].set(xlabel="Answer-token coordinate", ylabel="Target coordinate", title="Raw: branch identity is easy")

if MAIN:
    axes[0].legend()

if MAIN:
    for label, color in [(0, "#d65f5f"), (1, "#277da8")]:
        use = target_labels == label
        axes[1].scatter(plus[use, 1], plus[use, 2], s=110, c=color, label=f"Target Y={2 * label - 1:+}")

if MAIN:
    axes[1].set(xlabel="Target coordinate Y", ylabel="Distractor coordinate 3S", title="Centered plus branch: two independent features")

if MAIN:
    axes[1].legend(loc="upper center", fontsize=9)

if MAIN:
    fig.tight_layout()

if MAIN:
    save_figure(fig, "ccs_contrast_geometry.png")

# %%

def ccs_loss(p_plus: Tensor, p_minus: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Original loss, consistency and confidence, averaged over examples only.

    Inputs are (examples,) or (examples, independent_reporters). Returning one
    loss per reporter makes batching equivalent to fitting each independently.
    """
    if p_plus.ndim not in (1, 2) or p_plus.shape != p_minus.shape or p_plus.shape[0] == 0:
        raise ValueError("Probabilities must have matching nonempty shapes; broadcasting is forbidden.")
    if any(not torch.isfinite(p).all() or ((p < 0) | (p > 1)).any() for p in (p_plus, p_minus)):
        raise ValueError("Probabilities must be finite and lie in [0, 1].")
    consistency = (p_plus + p_minus - 1).square().mean(0)
    confidence = torch.minimum(p_plus, p_minus).square().mean(0)
    return consistency + confidence, consistency, confidence

if MAIN:
    tests.test_ccs_loss_values(ccs_loss)

if MAIN:
    tests.test_ccs_loss_gradient(ccs_loss)

if MAIN:
    print("Exact loss values and independent gradient checks passed!")

# %%

if MAIN:
    grid = torch.linspace(0, 1, 181, dtype=torch.float64)

if MAIN:
    pp, pm = torch.meshgrid(grid, grid, indexing="xy")

if MAIN:
    total, consistency, confidence = ccs_loss(pp.reshape(1, -1), pm.reshape(1, -1))

if MAIN:
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.8))

if MAIN:
    for ax, values, title in zip(axes, [consistency, confidence, total], ["Consistency", "Confidence", "Combined loss"]):
        heat = ax.imshow(values.reshape(181, 181), origin="lower", extent=[0, 1, 0, 1], cmap="magma", aspect="equal")
        ax.plot([0, 1], [1, 0], color="white", linewidth=1, alpha=.6)
        ax.scatter([0, .5, 1], [0, .5, 1], facecolors="none", edgecolors="#5be7c4", s=65)
        ax.set(xlabel="Plus score", ylabel="Minus score", title=title)
        fig.colorbar(heat, ax=ax, shrink=.8)

if MAIN:
    fig.suptitle("Neither loss term is enough on its own", y=1.04)

if MAIN:
    fig.tight_layout()

if MAIN:
    save_figure(fig, "ccs_loss_landscape.png")

# %%

def paired_probabilities(
    plus: Tensor, minus: Tensor, weight: Tensor, bias: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Return averaged positive-answer scores and the two branch probabilities.

    A vector weight gives one reporter; (reporters, features) gives many.
    A score is not semantically oriented until a separate orientation is chosen.
    """
    _check_pairs(plus, minus)
    if weight.ndim not in (1, 2) or weight.shape[-1] != plus.shape[-1]:
        raise ValueError("Reporter weights must end in the activation feature dimension.")
    expected_bias = () if weight.ndim == 1 else (weight.shape[0],)
    if bias.shape != expected_bias:
        raise ValueError("Bias must be scalar for one reporter or one value per reporter.")
    p_plus = (plus @ weight.T + bias).sigmoid() if weight.ndim == 2 else (plus @ weight + bias).sigmoid()
    p_minus = (minus @ weight.T + bias).sigmoid() if weight.ndim == 2 else (minus @ weight + bias).sigmoid()
    return (p_plus + 1 - p_minus) / 2, p_plus, p_minus

if MAIN:
    tests.test_paired_inference(paired_probabilities)

if MAIN:
    tests.test_two_feature_counterexample(paired_probabilities, ccs_loss)

if MAIN:
    print("Paired inference and independent-feature counterexample passed!")

# %%

if MAIN:
    known_reporters = {
        "Read target Y": torch.tensor([0., 15., 0., 0.], dtype=plus.dtype),
        "Read distractor S": torch.tensor([0., 0., 5., 0.], dtype=plus.dtype),
        "Read shared content U": torch.tensor([0., 0., 0., 15.], dtype=plus.dtype),
        "Read answer offset": torch.tensor([2., 0., 0., 0.], dtype=plus.dtype),
    }

if MAIN:
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6))

if MAIN:
    rows, names, losses, accuracies = [], [], [], []

if MAIN:
    for name, weight in known_reporters.items():
        xp, xm = (plus_raw, minus_raw) if name == "Read answer offset" else (plus, minus)
        score, p, m = paired_probabilities(xp, xm, weight, torch.tensor(0., dtype=plus.dtype))
        rows.append(score.numpy()); names.append(name)
        losses.append(float(ccs_loss(p, m)[0]))
        accuracies.append(float(((score > .5) == target_labels).double().mean()))

if MAIN:
    heat = axes[0].imshow(np.array(rows), vmin=0, vmax=1, cmap="RdBu", aspect="auto")

if MAIN:
    axes[0].set(yticks=range(4), yticklabels=names, xticks=range(8), xlabel="Question index", title="Averaged plus score")

if MAIN:
    fig.colorbar(heat, ax=axes[0], shrink=.75)

if MAIN:
    axes[1].scatter(losses, accuracies, s=90, c=["#277da8", "#e08b35", "#8e69b3", "#777777"])

if MAIN:
    for name, loss, accuracy in zip(names, losses, accuracies):
        offset = {"Read target Y": (10, -12), "Read distractor S": (10, 14), "Read shared content U": (-125, 12), "Read answer offset": (10, -20)}[name]
        axes[1].annotate(name, (loss, accuracy), xytext=offset, textcoords="offset points", fontsize=9)

if MAIN:
    axes[1].axhline(.5, color="gray", linestyle="--", linewidth=1)

if MAIN:
    axes[1].set(xlabel="Unlabelled CCS loss", ylabel="Target accuracy", ylim=(.4, 1.1), title="Equally low loss, different meanings")

if MAIN:
    axes[1].set_xlim(-.08, 1.8)

if MAIN:
    fig.tight_layout()

if MAIN:
    save_figure(fig, "ccs_exact_counterexample.png")

# %%

def fit_ccs_restarts(
    plus: Tensor,
    minus: Tensor,
    *,
    seeds: Sequence[int] = tuple(range(10)),
    steps: int = 1000,
    learning_rate: float = 0.01,
    weight_decay: float = 0.01,
) -> dict[str, Tensor | list[int]]:
    """Fit independent affine reporters with AdamW and retain every restart.

    Inputs must already have training-fitted centering applied. Independent
    parameter rows are optimized together; sum, rather than mean, over their
    losses preserves each reporter's gradient. The final loss is recomputed
    after the final update. No labels or validation scores enter this function.
    """
    _check_pairs(plus, minus)
    seeds = list(seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Provide at least one unique restart seed.")
    if steps < 1 or learning_rate <= 0 or weight_decay < 0:
        raise ValueError("Steps and learning rate must be positive; weight decay must be nonnegative.")
    plus, minus = plus.detach(), minus.detach()
    width = plus.shape[1]
    bound = 1 / math.sqrt(width)
    initial_weights, initial_biases = [], []
    for seed in seeds:
        generator = torch.Generator(device=plus.device).manual_seed(seed)
        initial_weights.append(
            torch.empty(width, dtype=plus.dtype, device=plus.device).uniform_(-bound, bound, generator=generator)
        )
        initial_biases.append(
            torch.empty((), dtype=plus.dtype, device=plus.device).uniform_(-bound, bound, generator=generator)
        )
    weight = torch.stack(initial_weights).requires_grad_()
    bias = torch.stack(initial_biases).requires_grad_()
    optimizer = torch.optim.AdamW([weight, bias], lr=learning_rate, weight_decay=weight_decay)
    history = []
    for step in range(steps + 1):
        _, p_plus, p_minus = paired_probabilities(plus, minus, weight, bias)
        loss, consistency, confidence = ccs_loss(p_plus, p_minus)
        history.append(torch.stack((loss, consistency, confidence), dim=-1).detach().cpu())
        if step == steps:
            break
        optimizer.zero_grad(set_to_none=True)
        loss.sum().backward()
        optimizer.step()
    return {
        "weights": weight.detach().clone(),
        "biases": bias.detach().clone(),
        "history": torch.stack(history),
        "seeds": seeds,
    }

def select_restart(fitted: dict) -> dict:
    """Choose the first minimum final training-loss restart, without labels."""
    history = fitted["history"]
    if history.ndim != 3 or history.shape[-1] != 3 or not torch.isfinite(history).all():
        raise ValueError("Expected finite (steps + 1, restarts, 3) loss history.")
    count = len(fitted["seeds"])
    if history.shape[1] != count or fitted["weights"].shape[0] != count or fitted["biases"].shape != (count,):
        raise ValueError("Restart parameters, seeds and history must agree.")
    index = int(history[-1, :, 0].argmin())
    return {
        "index": index,
        "seed": fitted["seeds"][index],
        "weight": fitted["weights"][index].clone(),
        "bias": fitted["biases"][index].clone(),
        "training_loss": float(history[-1, index, 0]),
    }

if MAIN:
    tests.test_batched_restarts(fit_ccs_restarts)

if MAIN:
    tests.test_restart_selection(select_restart)

if MAIN:
    print("Independent restart equivalence and selection checks passed!")

# %%

if MAIN:
    fitted = fit_ccs_restarts(plus, minus)

if MAIN:
    chosen = select_restart(fitted)

if MAIN:
    all_scores, all_p, all_m = paired_probabilities(plus, minus, fitted["weights"], fitted["biases"])

if MAIN:
    target_raw = ((all_scores > .5) == target_labels[:, None]).double().mean(0)

if MAIN:
    distractor_raw = ((all_scores > .5) == distractor_labels[:, None]).double().mean(0)

if MAIN:
    target_oracle = torch.maximum(target_raw, 1 - target_raw)

if MAIN:
    distractor_oracle = torch.maximum(distractor_raw, 1 - distractor_raw)

if MAIN:
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.0))

if MAIN:
    for index, seed in enumerate(fitted["seeds"]):
        axes[0].plot(fitted["history"][:, index, 0], linewidth=1.2, alpha=.75)
        axes[1].scatter(float(fitted["history"][-1, index, 0]), float(target_oracle[index]), s=35)
        if index in {chosen["index"], int(target_oracle.argmax())}:
            offset = (8, 10) if index == chosen["index"] else (-38, -18)
            axes[1].annotate(f"Seed {seed}", (float(fitted["history"][-1, index, 0]), float(target_oracle[index])),
                             xytext=offset, textcoords="offset points", fontsize=9)

if MAIN:
    axes[0].axhline(.2, color="black", linestyle="--", linewidth=1, label="Best constant equal scores")

if MAIN:
    axes[0].set(xlabel="AdamW update", ylabel="CCS loss", yscale="log", title="Every restart is retained")

if MAIN:
    axes[0].legend(fontsize=8)

if MAIN:
    axes[1].scatter(chosen["training_loss"], float(target_oracle[chosen["index"]]), marker="*", s=160, facecolors="none", edgecolors="black")

if MAIN:
    axes[1].set(xlabel="Final training loss", ylabel="Target accuracy, best orientation", ylim=(.4, 1.1), title="Star: loss-selected restart")

if MAIN:
    positions = np.arange(10)

if MAIN:
    axes[2].bar(positions - .17, target_oracle, width=.34, label="Target", color="#277da8")

if MAIN:
    axes[2].bar(positions + .17, distractor_oracle, width=.34, label="Distractor", color="#e08b35")

if MAIN:
    axes[2].axhline(.5, color="gray", linestyle="--", linewidth=1)

if MAIN:
    axes[2].set(xticks=positions, xlabel="Restart seed", ylabel="Accuracy, best orientation", ylim=(0, 1.1), title="What feature did each fit find?")

if MAIN:
    axes[2].legend(fontsize=8)

if MAIN:
    fig.tight_layout()

if MAIN:
    save_figure(fig, "ccs_restart_outcomes.png")

if MAIN:
    print(f"Selected seed {chosen['seed']} using loss {chosen['training_loss']:.6f}.")

# %%

def calibrate_orientation(scores: Tensor, calibration_labels: Tensor) -> int:
    """Choose a global orientation using a separate labelled calibration split.

    A tie preserves +1. This is supervision for orientation, not CCS fitting.
    Exact 0.5 scores predict zero under either orientation.
    """
    _check_scores_labels(scores, calibration_labels)
    raw = ((scores > 0.5) == calibration_labels).sum()
    flipped = (((1 - scores) > 0.5) == calibration_labels).sum()
    return -1 if flipped > raw else 1

def binary_metrics(scores: Tensor, labels: Tensor, orientation: int = 1) -> dict[str, float | int]:
    """Evaluate a fixed orientation. The test-oracle score is explicitly separate."""
    _check_scores_labels(scores, labels)
    if orientation not in (-1, 1):
        raise ValueError("Orientation must be +1 or -1.")
    probability = scores if orientation == 1 else 1 - scores
    correct = ((probability > 0.5) == labels).sum().item()
    other_correct = (((1 - probability) > 0.5) == labels).sum().item()
    epsilon = max(1e-12, torch.finfo(probability.dtype).eps)
    clipped = probability.clamp(epsilon, 1 - epsilon)
    labels_float = labels.to(probability)
    cross_entropy = -(labels_float * clipped.log() + (1 - labels_float) * (1 - clipped).log()).mean()
    return {
        "count": len(scores),
        "correct": correct,
        "accuracy": correct / len(scores),
        "test_oracle_orientation_accuracy": max(correct, other_correct) / len(scores),
        "cross_entropy": float(cross_entropy),
        "brier": float((probability - labels_float).square().mean()),
    }

if MAIN:
    tests.test_orientation_and_metrics(calibrate_orientation, binary_metrics)

if MAIN:
    print("Calibration, fixed-orientation evaluation and probability checks passed!")

# %%

if MAIN:
    calibration_scores = torch.tensor([.8, .1, .9, .2], dtype=torch.float64)

if MAIN:
    calibration_labels = torch.tensor([0, 1, 0, 1])

if MAIN:
    orientation = calibrate_orientation(calibration_scores, calibration_labels)

if MAIN:
    final_labels = 1 - calibration_labels

if MAIN:
    fixed = binary_metrics(calibration_scores, final_labels, orientation)

if MAIN:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.8))

if MAIN:
    axes[0].plot(range(4), calibration_scores, "o-", label="Raw score", color="#277da8")

if MAIN:
    axes[0].plot(range(4), 1 - calibration_scores, "s--", label="Calibration-oriented score", color="#e08b35")

if MAIN:
    axes[0].scatter(range(4), final_labels, marker="x", s=85, color="black", label="Final labels")

if MAIN:
    axes[0].axhline(.5, color="gray", linewidth=1)

if MAIN:
    axes[0].set(xlabel="Final example", ylabel="Plus score / label", ylim=(-.1, 1.1), title="A deliberately shifted evaluation set")

if MAIN:
    axes[0].legend(fontsize=8)

if MAIN:
    axes[1].bar(["Fixed calibration\norientation", "Best final-label\norientation", "Balanced\nconstant baseline"],
                [fixed["accuracy"], fixed["test_oracle_orientation_accuracy"], .5], color=["#277da8", "#e08b35", "#888888"])

if MAIN:
    axes[1].set(ylabel="Final accuracy", ylim=(0, 1.1), title="These are different evaluation questions")

if MAIN:
    fig.tight_layout()

if MAIN:
    save_figure(fig, "ccs_orientation_control.png")

# %%

def fit_difference_pca(plus: Tensor, minus: Tensor) -> dict[str, Tensor]:
    """Fit a first-PC baseline to centred paired differences, without labels."""
    _check_pairs(plus, minus)
    differences = (plus - minus).detach()
    mean = differences.mean(0)
    _, singular, vh = torch.linalg.svd(differences - mean, full_matrices=False)
    if float(singular[0]) == 0:
        raise ValueError("The paired differences have no variance.")
    weight = vh[0]
    # Fix the numerical SVD sign; this is not a semantic orientation.
    weight = weight * weight[weight.abs().argmax()].sign()
    return {"weight": weight, "mean": mean, "singular_values": singular}

if MAIN:
    tests.test_pca_distractor(fit_difference_pca)

if MAIN:
    print("Paired-difference PCA baseline passed!")

# %%

if MAIN:
    pca = fit_difference_pca(plus, minus)

if MAIN:
    rng = torch.Generator().manual_seed(820)

if MAIN:
    random_weights = torch.randn(32, plus.shape[1], generator=rng, dtype=plus.dtype)

if MAIN:
    random_weights = random_weights / random_weights.norm(dim=1, keepdim=True)

if MAIN:
    random_scores, _, _ = paired_probabilities(plus, minus, random_weights, torch.zeros(32, dtype=plus.dtype))

if MAIN:
    random_target = ((random_scores > .5) == target_labels[:, None]).double().mean(0)

if MAIN:
    random_distractor = ((random_scores > .5) == distractor_labels[:, None]).double().mean(0)

if MAIN:
    random_target = torch.maximum(random_target, 1 - random_target)

if MAIN:
    random_distractor = torch.maximum(random_distractor, 1 - random_distractor)

if MAIN:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))

if MAIN:
    for i, weight in enumerate(fitted["weights"]):
        direction = weight[1:3] / weight[1:3].norm()
        axes[0].arrow(0, 0, float(direction[0]), float(direction[1]), head_width=.045, alpha=.45, color="#277da8", length_includes_head=True)

if MAIN:
    axes[0].arrow(0, 0, 0, 1, head_width=.07, color="#e08b35", length_includes_head=True)

if MAIN:
    axes[0].plot([], [], color="#277da8", label="CCS restarts")

if MAIN:
    axes[0].plot([], [], color="#e08b35", label="Paired PCA")

if MAIN:
    axes[0].set(xlabel="Target-axis weight", ylabel="Distractor-axis weight", xlim=(-1.2, 1.2), ylim=(-1.2, 1.2), title="Directions in the known feature plane", aspect="equal")

if MAIN:
    axes[0].legend(fontsize=8)

if MAIN:
    axes[1].scatter(np.arange(32), random_target, s=26, color="#277da8", label="Target, random direction")

if MAIN:
    axes[1].scatter(np.arange(32), random_distractor, s=26, marker="x", color="#e08b35", label="Distractor, random direction")

if MAIN:
    axes[1].axhline(.5, color="gray", linestyle="--", linewidth=1)

if MAIN:
    axes[1].set(xlabel="Random direction seed draw", ylabel="Accuracy, best orientation", ylim=(.4, 1.1), title="Even random directions can read a salient feature")

if MAIN:
    axes[1].legend(fontsize=8, loc="center right")

if MAIN:
    fig.tight_layout()

if MAIN:
    save_figure(fig, "ccs_direction_baselines.png")

# %%

if MAIN:
    DISTRACTOR_STRENGTHS = [.25, .5, .75, 1., 1.5, 2., 3., 5.]

if MAIN:
    PLAY_STEPS = 350

if MAIN:
    PLAY_SEEDS = list(range(10))

if MAIN:
    play_records = []

if MAIN:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.0))

if MAIN:
    for strength in DISTRACTOR_STRENGTHS:
        pp, pm = plus.clone(), minus.clone()
        pp[:, 2], pm[:, 2] = strength * distractor_sign, -strength * distractor_sign
        fit = fit_ccs_restarts(pp, pm, seeds=PLAY_SEEDS, steps=PLAY_STEPS)
        best = select_restart(fit)
        score, _, _ = paired_probabilities(pp, pm, fit["weights"], fit["biases"])
        accuracy = ((score > .5) == target_labels[:, None]).double().mean(0)
        oracle = torch.maximum(accuracy, 1 - accuracy)
        axes[0].scatter([strength] * len(PLAY_SEEDS), oracle, color="#277da8", alpha=.25, s=30)
        axes[0].scatter(strength, oracle[best["index"]], facecolors="none", edgecolors="black", s=90)
        axes[1].scatter([strength] * len(PLAY_SEEDS), fit["history"][-1, :, 0], color="#277da8", alpha=.4, s=30)
        axes[1].scatter(strength, best["training_loss"], facecolors="none", edgecolors="black", s=90)
        play_records.append({"strength": strength, "selected_seed": best["seed"], "selected_target_oracle_accuracy": float(oracle[best["index"]]),
                             "all_target_oracle_accuracies": oracle.tolist(), "all_final_losses": fit["history"][-1, :, 0].tolist()})

if MAIN:
    axes[0].plot([.25, .99], [1, 1], "--", color="#e08b35", label="PCA (tie at strength 1 excluded)")

if MAIN:
    axes[0].plot([1.01, 5], [.5, .5], "--", color="#e08b35")

if MAIN:
    axes[0].axhline(.5, color="gray", linestyle="--", linewidth=1)

if MAIN:
    axes[0].set(xlabel="Distractor strength / target strength", ylabel="Target accuracy, best orientation", ylim=(.4, 1.1), title="Same target, different optimization preference")

if MAIN:
    axes[0].legend(fontsize=8)

if MAIN:
    axes[1].set(xlabel="Distractor strength / target strength", ylabel="Final training CCS loss", yscale="log", title="Low loss does not identify the target")

if MAIN:
    fig.tight_layout()

if MAIN:
    save_figure(fig, "ccs_distractor_strength_play.png")

if MAIN:
    (assets_dir / "foundation_play.json").write_text(json.dumps(play_records, indent=2) + "\n")

# %%

from chapter11_representation_geometry.exercises.part5_contrast_consistent_search.reference_parity import compare_author_primitives

if MAIN:
    parity = compare_author_primitives()

if MAIN:
    for result in parity["cases"]:
        print(f"{result['case']}: centering error {result['centering_error']:.2e}, loss error {result['loss_error']:.2e}, gradient error {result['gradient_error']:.2e}")

if MAIN:
    assert parity["passed"]

if MAIN:
    (assets_dir / "author_parity.json").write_text(json.dumps(parity, indent=2) + "\n")

# %%

from types import SimpleNamespace

if MAIN:
    from transformers import AutoModelForCausalLM, AutoTokenizer

from chapter11_representation_geometry.exercises.part5_contrast_consistent_search import ccs_data as data

if MAIN:
    from chapter11_representation_geometry.exercises.part5_contrast_consistent_search import ccs_native as native

from chapter11_representation_geometry.exercises.part5_contrast_consistent_search import ccs_native_methods_tests as native_tests

from chapter11_representation_geometry.exercises.part4_concept_erasure.erasure_methods import fit_binary_probe

if MAIN:
    data.check_data_contract()

if MAIN:
    native_dir = assets_dir / "native"

if MAIN:
    native_dir.mkdir(exist_ok=True)

if MAIN:
    MODEL_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"

if MAIN:
    tokenizer = AutoTokenizer.from_pretrained("gpt2", revision=MODEL_REVISION, local_files_only=True)

if MAIN:
    tokenizer.pad_token = tokenizer.eos_token

if MAIN:
    tokenizer.padding_side = "right"

if MAIN:
    model = AutoModelForCausalLM.from_pretrained("gpt2", revision=MODEL_REVISION, local_files_only=True,
        torch_dtype=torch.float32, attn_implementation="eager").to("cpu").eval().requires_grad_(False)

if MAIN:
    initial_weight_digest = native.weight_digest(model)

if MAIN:
    example = data.paired_reviews("development", "distracting_reader", "customer")[0]

if MAIN:
    print(example["minus"])

if MAIN:
    print("\nThe opposite proposed answer changes only the last token:")

if MAIN:
    print(example["plus"])

if MAIN:
    print("\nCandidate token IDs:", [tokenizer.encode(word, add_special_tokens=False) for word in data.ANSWER_SUFFIXES])

# %%

def last_token_rows(hidden: Tensor, attention_mask: Tensor) -> Tensor:
    """Gather the last valid token from each padded (batch, tokens, features) row."""
    if hidden.ndim != 3 or attention_mask.shape != hidden.shape[:2] or min(hidden.shape) == 0:
        raise ValueError("Use nonempty hidden states and their matching token mask.")
    if not torch.all((attention_mask == 0) | (attention_mask == 1)):
        raise ValueError("The attention mask must contain only zeros and ones.")
    positions = torch.arange(hidden.shape[1], device=hidden.device)[None, :]
    last = torch.where(attention_mask.bool(), positions, -1).max(dim=1).values
    if (last < 0).any():
        raise ValueError("Every example must contain a valid token.")
    return hidden[torch.arange(len(hidden), device=hidden.device), last]

if MAIN:
    native_tests.test_last_token_rows(last_token_rows)

if MAIN:
    print("Native measurement check passed!")

# %%

if MAIN:
    check_rows = data.paired_reviews("development", "distracting_reader", "customer")

if MAIN:
    check_texts = [row["plus"] for row in check_rows[:3]] + [check_rows[-1]["minus"]]

if MAIN:
    capture_checks = native.check_capture(model, tokenizer, check_texts, gather=last_token_rows)

if MAIN:
    print(f"Full-model logit comparison: {capture_checks['full_model_logit_error']:.2e} maximum absolute difference")

if MAIN:
    print(f"Single-row padding comparison: {capture_checks['padding_logit_error']:.2e} maximum absolute difference")

if MAIN:
    print("All activation and output comparisons passed their float32 tolerances.")

# %%

def paired_opinion_change(scores: Tensor, rows: list[dict]) -> dict:
    """Change only Morgan's opinion, matching by base ID rather than row order."""
    if scores.shape != (len(rows),) or not rows or not torch.isfinite(scores).all():
        raise ValueError("Use one finite score per nonempty row.")
    if ((scores < 0) | (scores > 1)).any():
        raise ValueError("Scores must lie in [0, 1].")
    groups = {}
    for index, row in enumerate(rows):
        opinion = row["distractor"]
        if opinion not in (0, 1):
            raise ValueError("This measurement requires two explicit distracting opinions.")
        group = groups.setdefault(row["base_id"], {})
        if opinion in group:
            raise ValueError("A base review has a duplicate opinion row.")
        group[opinion] = index
    negative, positive, ids = [], [], []
    for base_id in sorted(groups):
        group = groups[base_id]
        if set(group) != {0, 1}:
            raise ValueError("Every base review needs both opinions.")
        a, b = rows[group[0]], rows[group[1]]
        if any(a[key] != b[key] for key in ("review", "label", "wrapper", "variant")):
            raise ValueError("Only the distracting opinion may change within a pair.")
        ids.append(base_id)
        negative.append(scores[group[0]])
        positive.append(scores[group[1]])
    negative, positive = torch.stack(negative), torch.stack(positive)
    difference = positive - negative
    return {"base_ids": ids, "negative_opinion_scores": negative, "positive_opinion_scores": positive,
            "score_changes": difference, "base_reviews": len(ids),
            "prediction_flip_fraction": float(((negative > .5) != (positive > .5)).double().mean()),
            "mean_absolute_score_change": float(difference.abs().mean())}

if MAIN:
    native_tests.test_paired_opinion_change(paired_opinion_change)

if MAIN:
    print("Native measurement check passed!")

# %%

def model_answer_metrics(logits: Tensor, candidate_token_ids: list[int], labels: Tensor) -> dict:
    """Keep a two-answer classification distinct from unrestricted next-token output."""
    if logits.ndim != 2 or labels.shape != (len(logits),) or not len(logits) or not torch.isfinite(logits).all():
        raise ValueError("Use finite full-vocabulary logits and one label per row.")
    if len(candidate_token_ids) != 2 or len(set(candidate_token_ids)) != 2:
        raise ValueError("Specify distinct negative and positive candidate tokens, in that order.")
    if any(index < 0 or index >= logits.shape[-1] for index in candidate_token_ids):
        raise ValueError("A candidate token is outside the vocabulary.")
    if not torch.all((labels == 0) | (labels == 1)):
        raise ValueError("Labels must be binary.")
    log_probability = logits.double().log_softmax(-1)
    candidate_logits = logits[:, candidate_token_ids].double()
    restricted_positive = candidate_logits.softmax(-1)[:, 1]
    candidate_probabilities = log_probability[:, candidate_token_ids].exp()
    winners = logits.argmax(-1)
    targets = torch.tensor(candidate_token_ids, device=logits.device)[labels.long()]
    return {"count": len(logits), "restricted_positive_scores": restricted_positive,
            "candidate_probabilities": candidate_probabilities, "candidate_mass": candidate_probabilities.sum(-1),
            "winning_token_ids": winners,
            "restricted_correct": int(((restricted_positive > .5) == labels).sum()),
            "full_vocabulary_correct": int((winners == targets).sum()),
            "target_full_vocabulary_nll": float(-log_probability[torch.arange(len(logits)), targets].mean())}

if MAIN:
    native_tests.test_model_answer_metrics(model_answer_metrics)

if MAIN:
    print("Native measurement check passed!")

# %%

m = SimpleNamespace(**{name: globals()[name] for name in [
    "fit_pair_centering", "apply_pair_centering", "ccs_loss", "paired_probabilities",
    "fit_ccs_restarts", "select_restart", "calibrate_orientation", "binary_metrics", "fit_difference_pca"]})

nm = SimpleNamespace(last_token_rows=last_token_rows, paired_opinion_change=paired_opinion_change,
                     model_answer_metrics=model_answer_metrics)

def fit_reporters(train_plus, train_minus, training_labels):
    centering = m.fit_pair_centering(train_plus, train_minus)
    plus, minus = m.apply_pair_centering(train_plus, train_minus, centering)
    result = {"centering": centering, "groups": {}}
    for mode in ("centered", "raw", "shuffled_pairs"):
        x_plus, x_minus = (train_plus, train_minus) if mode == "raw" else (plus, minus)
        permutation = None
        if mode == "shuffled_pairs":
            permutation = torch.randperm(len(x_minus), generator=torch.Generator().manual_seed(904))
            x_minus = x_minus[permutation]
        fitted = m.fit_ccs_restarts(x_plus, x_minus)
        result["groups"][mode] = {"fitted": fitted, "selected": m.select_restart(fitted), "permutation": permutation}
    result["pca"] = m.fit_difference_pca(plus, minus)
    # This baseline alone consumes the training labels. They do not enter CCS.
    result["supervised"] = fit_binary_probe(torch.cat((plus, minus), dim=1), training_labels)
    selected = result["groups"]["centered"]["selected"]
    random_weights = []
    for seed in range(1000, 1032):
        weight = torch.randn(selected["weight"].shape, dtype=torch.float64, generator=torch.Generator().manual_seed(seed))
        weight *= selected["weight"].norm() / weight.norm()
        random_weights.append(weight)
    result["random"] = {"weights": torch.stack(random_weights), "bias": selected["bias"].clone()}
    return result

def reporter_scores(fitted, raw_plus, raw_minus):
    plus, minus = m.apply_pair_centering(raw_plus, raw_minus, fitted["centering"])
    scores = {}
    for mode, group in fitted["groups"].items():
        inputs = (raw_plus, raw_minus) if mode == "raw" else (plus, minus)
        fit = group["fitted"]
        values = m.paired_probabilities(*inputs, fit["weights"], fit["biases"])[0]
        for index, seed in enumerate(fit["seeds"]):
            scores[f"CCS {mode} seed {seed}"] = values[:, index]
    pca = fitted["pca"]
    scores["Paired-difference PCA"] = ((plus - minus - pca["mean"]) @ pca["weight"]).sigmoid()
    supervised = fitted["supervised"]
    scores["Supervised linear probe"] = (
        torch.cat((plus, minus), dim=1) @ supervised["coefficient"] + supervised["intercept"]
    ).sigmoid()
    random = fitted["random"]
    values = m.paired_probabilities(plus, minus, random["weights"], random["bias"].expand(32))[0]
    for index in range(32):
        scores[f"Random direction {1000 + index}"] = values[:, index]
    if len(scores) != 64:
        raise ValueError("The protocol requires all 64 reporters per case.")
    return scores

def calibrate_reporters(fitted, calibration_plus, calibration_minus, calibration_labels):
    scores = reporter_scores(fitted, calibration_plus, calibration_minus)
    orientations = {name: m.calibrate_orientation(value, calibration_labels) for name, value in scores.items()}
    orientations["Supervised linear probe"] = 1
    return orientations

def evaluate_reporters(fitted, scores, rows, orientations):
    if set(scores) != set(orientations) or len(scores) != 64:
        raise ValueError("Every reporter must retain its calibration orientation.")
    labels = torch.tensor([row["label"] for row in rows])
    information = {}
    for mode, group in fitted["groups"].items():
        for index, seed in enumerate(group["fitted"]["seeds"]):
            information[f"CCS {mode} seed {seed}"] = {
                "training_loss": float(group["fitted"]["history"][-1, index, 0]),
                "selected": index == group["selected"]["index"],
            }
    records = []
    for name, values in scores.items():
        orientation = orientations[name]
        record = {"name": name, "orientation": orientation, **information.get(name, {}),
                  "target": m.binary_metrics(values, labels, orientation)}
        if rows[0]["distractor"] is not None:
            distractor = torch.tensor([row["distractor"] for row in rows])
            record["distractor"] = m.binary_metrics(values, distractor, orientation)
            change = nm.paired_opinion_change(values if orientation == 1 else 1 - values, rows)
            record["opinion_change"] = {key: change[key] for key in (
                "base_reviews", "prediction_flip_fraction", "mean_absolute_score_change")}
        records.append(record)
    return records

def selected_name(fitted, mode="centered"):
    return f'CCS {mode} seed {fitted["groups"][mode]["selected"]["seed"]}'

def summarize_native_outputs(cache):
    rows = cache["rows"]
    metrics = nm.model_answer_metrics(cache["prefix"]["logits"], cache["candidate_token_ids"],
                                      torch.tensor([row["label"] for row in rows]))
    result = {key: metrics[key] for key in ("count", "restricted_correct", "full_vocabulary_correct", "target_full_vocabulary_nll")}
    result["mean_candidate_probability_mass"] = float(metrics["candidate_mass"].mean())
    if rows[0]["distractor"] is not None:
        change = nm.paired_opinion_change(metrics["restricted_positive_scores"], rows)
        result["opinion_prediction_flip_fraction"] = change["prediction_flip_fraction"]
    return result

# %%

if MAIN:
    native_summary = {"status": "RUNNING", "model_revision": MODEL_REVISION, "cases": [], "captures": [],
                      "split_manifest": data.split_manifest(), "capture_checks": capture_checks,
                      "final_used_for_selection": False, "final_started_only_after_all_fits": False}

def save_native_summary():
    (assets_dir / "native_summary.json").write_text(json.dumps(native_summary, indent=2) + "\n")

if MAIN:
    for variant, wrapper in itertools.product(data.VARIANTS, data.WRAPPERS):
        caches = {}
        for split in ("train", "calibration", "development"):
            rows = data.paired_reviews(split, variant, wrapper)
            caches[split] = native.capture_pairs(model, tokenizer, rows, gather=last_token_rows)
            filename = f"capture-{variant}-{wrapper}-{split}.pt"
            torch.save(caches[split], native_dir / filename)
            native_summary["captures"].append({"variant": variant, "wrapper": wrapper, "split": split,
                "filename": filename, "native_outputs": summarize_native_outputs(caches[split])})
        for layer in native.LAYERS:
            train_plus = caches["train"]["plus"]["features"][layer].double()
            train_minus = caches["train"]["minus"]["features"][layer].double()
            training_labels = torch.tensor([row["label"] for row in caches["train"]["rows"]])
            fitted = fit_reporters(train_plus, train_minus, training_labels)
            calibration = caches["calibration"]
            orientations = calibrate_reporters(fitted, calibration["plus"]["features"][layer].double(),
                calibration["minus"]["features"][layer].double(), torch.tensor([row["label"] for row in calibration["rows"]]))
            saved_scores, evaluations = {}, {}
            for split, cache in caches.items():
                scores = reporter_scores(fitted, cache["plus"]["features"][layer].double(), cache["minus"]["features"][layer].double())
                saved_scores[split] = scores
                evaluations[split] = evaluate_reporters(fitted, scores, cache["rows"], orientations)
            filename = f"fits-{variant}-{wrapper}-layer{layer}.pt"
            torch.save({"fitted": fitted, "orientations": orientations, "scores": saved_scores}, native_dir / filename)
            native_summary["cases"].append({"variant": variant, "wrapper": wrapper, "layer": layer,
                "filename": filename, "selected": {mode: selected_name(fitted, mode) for mode in fitted["groups"]},
                "evaluations": evaluations})
            save_native_summary()
        print(f"Fitted all three sites: {variant}, {wrapper}")

if MAIN:
    assert len(native_summary["cases"]) == 18

if MAIN:
    print("All 18 cases are fitted and oriented. No final model data have been evaluated.")

# %%

def record_by_name(case, split, name):
    return next(record for record in case["evaluations"][split] if record["name"] == name)

def case_for(variant, wrapper, layer):
    return next(case for case in native_summary["cases"] if (case["variant"], case["wrapper"], case["layer"]) == (variant, wrapper, layer))

if MAIN:
    VARIANT_LABELS = {"plain": "Plain", "neutral_reader": "Neutral reader", "distracting_reader": "Other opinion"}

if MAIN:
    COLORS = {"plain": "#277da8", "neutral_reader": "#6a994e", "distracting_reader": "#df8534"}

if MAIN:
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8))

if MAIN:
    for ax, wrapper in zip(axes, data.WRAPPERS):
        values = np.zeros((3, 3))
        for i, variant in enumerate(data.VARIANTS):
            for j, layer in enumerate(native.LAYERS):
                case = case_for(variant, wrapper, layer)
                ccs = record_by_name(case, "development", case["selected"]["centered"])["target"]["accuracy"]
                supervised = record_by_name(case, "development", "Supervised linear probe")["target"]["accuracy"]
                values[i, j] = ccs
                ax.text(j, i, f"{ccs:.2f} / {supervised:.2f}", ha="center", va="center", fontsize=11,
                        color="white" if ccs > .7 else "black")
        plot = ax.imshow(values, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        ax.set(xticks=range(3), xticklabels=native.LAYERS, yticks=range(3), yticklabels=[VARIANT_LABELS[v] for v in data.VARIANTS],
               xlabel="Hidden-state index", title=wrapper.replace("_", " ").title())
        ax.grid(False)
        fig.colorbar(plot, ax=ax, label="CCS customer accuracy", shrink=.8)

if MAIN:
    fig.suptitle("Development: each cell shows CCS / supervised accuracy", y=1.02)

if MAIN:
    fig.tight_layout()

if MAIN:
    save_figure(fig, "ccs_native_development.png")

# %%

if MAIN:
    assert len(native_summary["cases"]) == 18

if MAIN:
    native_summary["final_started_only_after_all_fits"] = True

if MAIN:
    for variant, wrapper in itertools.product(data.VARIANTS, data.WRAPPERS):
        rows = data.paired_reviews("final", variant, wrapper)
        cache = native.capture_pairs(model, tokenizer, rows, gather=last_token_rows)
        filename = f"capture-{variant}-{wrapper}-final.pt"
        torch.save(cache, native_dir / filename)
        native_summary["captures"].append({"variant": variant, "wrapper": wrapper, "split": "final",
            "filename": filename, "native_outputs": summarize_native_outputs(cache)})
        for layer in native.LAYERS:
            case = case_for(variant, wrapper, layer)
            payload = torch.load(native_dir / case["filename"], weights_only=False)
            scores = reporter_scores(payload["fitted"], cache["plus"]["features"][layer].double(), cache["minus"]["features"][layer].double())
            payload["scores"]["final"] = scores
            case["evaluations"]["final"] = evaluate_reporters(payload["fitted"], scores, rows, payload["orientations"])
            torch.save(payload, native_dir / case["filename"])
        save_native_summary()

if MAIN:
    final_score_count = sum(record["target"]["count"] for case in native_summary["cases"] for record in case["evaluations"]["final"])

if MAIN:
    assert final_score_count == 98304

if MAIN:
    print(f"Retained {final_score_count:,} reporter scores across all 18 cases; 64 distinct base reviews.")

# %%

if MAIN:
    method_labels = ["CCS centered", "CCS raw", "CCS shuffled", "Paired PCA", "Supervised", "Random median"]

if MAIN:
    case_columns = list(itertools.product(data.WRAPPERS, native.LAYERS))

if MAIN:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.6))

if MAIN:
    for ax, variant in zip(axes, data.VARIANTS):
        values = np.zeros((6, 6))
        for j, (wrapper, layer) in enumerate(case_columns):
            case = case_for(variant, wrapper, layer)
            names = [case["selected"][mode] for mode in ("centered", "raw", "shuffled_pairs")] + ["Paired-difference PCA", "Supervised linear probe"]
            for i, name in enumerate(names):
                values[i, j] = record_by_name(case, "final", name)["target"]["accuracy"]
            values[5, j] = np.median([r["target"]["accuracy"] for r in case["evaluations"]["final"] if r["name"].startswith("Random direction")])
        heat = ax.imshow(values, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        for (i, j), value in np.ndenumerate(values):
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=9, color="white" if value > .7 else "black")
        ax.set(yticks=range(6), yticklabels=method_labels, xticks=range(6),
               xticklabels=[f"{'Customer' if w == 'customer' else 'Quoted'}\n{layer}" for w, layer in case_columns], title=VARIANT_LABELS[variant])
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(False)

if MAIN:
    fig.suptitle("Final customer accuracy with calibration orientation held fixed", y=1.01)

if MAIN:
    fig.tight_layout()

if MAIN:
    save_figure(fig, "ccs_final_method_comparison.png")

# %%

if MAIN:
    fig, axes = plt.subplots(1, 2, figsize=(12.3, 4.8))

if MAIN:
    markers = {4: "o", 8: "^", 12: "s"}

if MAIN:
    for ax, wrapper in zip(axes, data.WRAPPERS):
        for variant, layer in itertools.product(data.VARIANTS, native.LAYERS):
            case = case_for(variant, wrapper, layer)
            records = [r for r in case["evaluations"]["final"] if r["name"].startswith("CCS centered")]
            ax.scatter([r["training_loss"] for r in records], [r["target"]["accuracy"] for r in records],
                       c=COLORS[variant], marker=markers[layer], alpha=.6, s=35)
            selected = next(r for r in records if r["selected"])
            ax.scatter(selected["training_loss"], selected["target"]["accuracy"], marker="*", s=130,
                       facecolors="none", edgecolors="black", zorder=5)
        for variant in data.VARIANTS:
            ax.scatter([], [], c=COLORS[variant], label=VARIANT_LABELS[variant])
        for layer in native.LAYERS:
            ax.scatter([], [], c="gray", marker=markers[layer], label=f"Index {layer}")
        ax.axhline(.5, color="gray", linestyle="--", linewidth=1)
        ax.set(xscale="log", xlabel="Final unlabelled training loss", ylabel="Final customer accuracy",
               ylim=(-.05, 1.08), title=wrapper.replace("_", " ").title())
        ax.legend(fontsize=8, ncol=2, loc="lower right")

if MAIN:
    fig.tight_layout()

if MAIN:
    save_figure(fig, "ccs_final_all_restarts.png")

# %%

if MAIN:
    fig, axes = plt.subplots(1, 2, figsize=(12.7, 4.5))

if MAIN:
    labels = ["CCS centered", "Paired PCA", "Supervised", "Random median"]

if MAIN:
    for ax, measurement, title in zip(axes, ["accuracy", "prediction_flip_fraction"],
                                     ["Final customer accuracy", "Change only Morgan: prediction flip fraction"]):
        values = np.zeros((4, 6))
        for j, (wrapper, layer) in enumerate(case_columns):
            case = case_for("distracting_reader", wrapper, layer)
            names = [case["selected"]["centered"], "Paired-difference PCA", "Supervised linear probe"]
            key = "target" if measurement == "accuracy" else "opinion_change"
            for i, name in enumerate(names):
                values[i, j] = record_by_name(case, "final", name)[key][measurement]
            values[3, j] = np.median([r[key][measurement] for r in case["evaluations"]["final"] if r["name"].startswith("Random direction")])
        heat = ax.imshow(values, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        for (i, j), value in np.ndenumerate(values):
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="white" if value > .7 else "black")
        ax.set(yticks=range(4), yticklabels=labels, xticks=range(6),
               xticklabels=[f"{'Customer' if w == 'customer' else 'Quoted'}\n{layer}" for w, layer in case_columns], title=title)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(False)

if MAIN:
    fig.tight_layout()

if MAIN:
    save_figure(fig, "ccs_final_opinion_change.png")

# %%

if MAIN:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

if MAIN:
    for ax, wrapper in zip(axes, data.WRAPPERS):
        for index, (variant, layer) in enumerate(itertools.product(data.VARIANTS, native.LAYERS)):
            case = case_for(variant, wrapper, layer)
            values = [r["target"]["accuracy"] for r in case["evaluations"]["final"] if r["name"].startswith("Random direction")]
            ax.scatter(index + np.linspace(-.2, .2, 32), values, s=16, alpha=.5, c=COLORS[variant])
            ax.scatter(index, record_by_name(case, "final", case["selected"]["centered"])["target"]["accuracy"],
                       marker="*", s=120, c="black", zorder=5)
            ax.scatter(index, record_by_name(case, "final", "Supervised linear probe")["target"]["accuracy"],
                       marker="s", s=38, facecolors="none", edgecolors="#bc4749", zorder=6)
        ax.axhline(.5, color="gray", linestyle="--", linewidth=1)
        ax.set(xticks=range(9), xticklabels=[VARIANT_LABELS[v].replace(" ", "\n") + f"\n{layer}" for v, layer in itertools.product(data.VARIANTS, native.LAYERS)],
               ylabel="Final customer accuracy", ylim=(-.05, 1.08), title=wrapper.replace("_", " ").title())
        ax.tick_params(axis="x", labelsize=8)
        ax.scatter([], [], c="black", marker="*", label="Selected CCS")
        ax.scatter([], [], edgecolors="#bc4749", facecolors="none", marker="s", label="Supervised")
        ax.legend(fontsize=8, loc="lower left")

if MAIN:
    fig.tight_layout()

if MAIN:
    save_figure(fig, "ccs_final_random_controls.png")

# %%

if MAIN:
    final_captures = [c for c in native_summary["captures"] if c["split"] == "final"]

if MAIN:
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.8))

if MAIN:
    labels = [f"{VARIANT_LABELS[c['variant']]}\n{'Customer' if c['wrapper'] == 'customer' else 'Quoted'}" for c in final_captures]

if MAIN:
    for ax, key, title in zip(axes, ["restricted_correct", "full_vocabulary_correct", "mean_candidate_probability_mass"],
                              ["Forced two-answer accuracy", "Unrestricted next-token accuracy", "Probability mass on the two answers"]):
        values = [c["native_outputs"][key] / c["native_outputs"]["count"] if key.endswith("correct") else c["native_outputs"][key]
                  for c in final_captures]
        ax.bar(range(6), values, color=[COLORS[c["variant"]] for c in final_captures])
        if key.endswith("correct"):
            ax.axhline(.5, color="black", linestyle="--", linewidth=1)
        else:
            ax.axhline(1, color="black", linestyle="--", linewidth=1)
        ax.set(xticks=range(6), xticklabels=labels, ylim=(0, 1.07), title=title)
        ax.tick_params(axis="x", labelsize=7, rotation=30)
        for i, value in enumerate(values):
            label = f"{value:.2f}" if key.endswith("correct") else f"{100 * value:.3g}%"
            ax.text(i, value + .025, label, ha="center", fontsize=9)

if MAIN:
    fig.tight_layout()

if MAIN:
    save_figure(fig, "ccs_native_output_baselines.png")

# %%

import textwrap

if MAIN:
    gallery_cache = torch.load(native_dir / "capture-distracting_reader-customer-final.pt", weights_only=False)

if MAIN:
    gallery_case = case_for("distracting_reader", "customer", 8)

if MAIN:
    gallery_payload = torch.load(native_dir / gallery_case["filename"], weights_only=False)

if MAIN:
    gallery_ids = ["final:0:0:0:0", "final:0:0:0:1", "final:0:0:1:0"]

if MAIN:
    output_metrics = model_answer_metrics(gallery_cache["prefix"]["logits"], gallery_cache["candidate_token_ids"],
        torch.tensor([row["label"] for row in gallery_cache["rows"]]))

if MAIN:
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 5.0))

if MAIN:
    for ax, base_id in zip(axes, gallery_ids):
        indices = [i for i, row in enumerate(gallery_cache["rows"]) if row["base_id"] == base_id]
        indices.sort(key=lambda i: gallery_cache["rows"][i]["distractor"])
        row = gallery_cache["rows"][indices[0]]
        for name, label, color in [(gallery_case["selected"]["centered"], "CCS", "#277da8"),
                                    ("Supervised linear probe", "Supervised", "#bc4749")]:
            score = gallery_payload["scores"]["final"][name][indices]
            if gallery_payload["orientations"][name] == -1:
                score = 1 - score
            ax.plot([0, 1], score, "o-", color=color, label=label)
        ax.plot([0, 1], output_metrics["restricted_positive_scores"][indices], "s--", color="#6a994e", label="Native restricted")
        winners = [repr(tokenizer.decode([int(output_metrics["winning_token_ids"][i])])) for i in indices]
        ax.text(.02, -.28, "Native top token:\n" + " → ".join(winners), transform=ax.transAxes, fontsize=9)
        ax.axhline(.5, color="gray", linestyle="--", linewidth=1)
        ax.set(xticks=[0, 1], xticklabels=["Negative", "Positive"], xlabel="Morgan's opinion",
               ylabel="Score for positive customer sentiment", ylim=(-.05, 1.05),
               title=textwrap.fill(row["review"], 38) + f"\nCustomer label: {'positive' if row['label'] else 'negative'}")
        ax.title.set_fontsize(10)

if MAIN:
    axes[0].legend(fontsize=8, loc="upper right")

if MAIN:
    fig.tight_layout()

if MAIN:
    save_figure(fig, "ccs_actual_review_gallery.png")

# %%

if MAIN:
    PLAY_REVIEW = "Having finished the show, I can say it was amazing."

if MAIN:
    play_rows = []

if MAIN:
    for opinion in (0, 1):
        prefix = data.render_prefix(PLAY_REVIEW, "distracting_reader", "customer", opinion)
        play_rows.append({"prefix": prefix, "plus": prefix + " positive", "minus": prefix + " negative",
                          "review": PLAY_REVIEW, "distractor": opinion, "base_id": "native-play"})

if MAIN:
    play_cache = native.capture_pairs(model, tokenizer, play_rows, gather=last_token_rows)

if MAIN:
    play_scores = reporter_scores(gallery_payload["fitted"], play_cache["plus"]["features"][8].double(),
                                  play_cache["minus"]["features"][8].double())

if MAIN:
    play_native = play_cache["prefix"]["logits"][:, play_cache["candidate_token_ids"]].double().softmax(-1)[:, 1]

if MAIN:
    fig, ax = plt.subplots(figsize=(7.6, 4.4))

if MAIN:
    for name, label, color in [(gallery_case["selected"]["centered"], "Fixed CCS", "#277da8"),
                              ("Supervised linear probe", "Fixed supervised", "#bc4749")]:
        score = play_scores[name] if gallery_payload["orientations"][name] == 1 else 1 - play_scores[name]
        ax.plot([0, 1], score, "o-", label=label, color=color)

if MAIN:
    ax.plot([0, 1], play_native, "s--", color="#6a994e", label="Native restricted")

if MAIN:
    ax.axhline(.5, color="gray", linestyle="--", linewidth=1)

if MAIN:
    ax.set(xticks=[0, 1], xticklabels=["Morgan says negative", "Morgan says positive"],
           ylabel="Positive-customer score", ylim=(-.05, 1.05), title=textwrap.fill(PLAY_REVIEW, 60))

if MAIN:
    ax.legend(fontsize=9)

if MAIN:
    fig.tight_layout()

if MAIN:
    save_figure(fig, "ccs_native_review_play.png")

if MAIN:
    torch.save({"cache": play_cache, "scores": play_scores, "case": gallery_case["filename"]}, native_dir / "native-play.pt")

if MAIN:
    print("Native winning tokens:", [repr(tokenizer.decode([int(i)])) for i in play_cache["prefix"]["logits"].argmax(-1)])

if MAIN:
    native_summary.update(status="PASS", weights_unchanged=native.weight_digest(model) == initial_weight_digest,
        model_gradients_absent=all(parameter.grad is None for parameter in model.parameters()),
        model_weight_sha256=initial_weight_digest, final_reporter_scores=final_score_count,
        final_contrast_pairs=512, distinct_final_base_reviews=64, reporter_count_per_case=64, case_count=18,
        all_reporters_refitted=True, play_review=PLAY_REVIEW, play_case=gallery_case["filename"])

if MAIN:
    assert native_summary["weights_unchanged"] and native_summary["model_gradients_absent"]

if MAIN:
    save_native_summary()

# %%

import inspect
import tempfile

def run_smoke_test(cpu=True):
    from chapter11_representation_geometry.exercises.part5_contrast_consistent_search import ccs_methods_tests, ccs_native_methods_tests
    tested = []
    for module in (ccs_methods_tests, ccs_native_methods_tests):
        for name, function in vars(module).items():
            if name.startswith('test_') and inspect.isfunction(function):
                function()
                tested.append(name)
    return dict(tests_passed=True, accepted=True, test_count=len(tested),
        claim_scope='Original asymmetric loss, label-free fitting, exact competing-feature counterexample, orientation, padding and actual vocabulary metrics.')

def run_gpu_test(max_vram_gb=0.):
    """Legacy runner name; this experiment explicitly reloads and runs on CPU."""
    from chapter11_representation_geometry.exercises.part5_contrast_consistent_search.ccs_reproduction import reproduce, digest
    here = Path(__file__).parent
    publication_path = here / 'expected_outputs/learner_execution.json'
    publication = json.loads(publication_path.read_text())
    from chapter11_representation_geometry.exercises.part5_contrast_consistent_search.build_notebooks import STEM
    page = here.parents[1] / 'instructions/pages/05_[11.5]_Contrast_Consistent_Search.md'
    for path, key in [(here / f'{STEM}_exercises.ipynb', 'published_exercises_sha256'),
                      (here / f'{STEM}_solutions.ipynb', 'published_solutions_sha256'),
                      (page, 'published_page_sha256')]:
        if digest(path) != publication[key]:
            raise ValueError('The published learner changed after validation: ' + str(path))
    for name, checksum in publication['published_figures'].items():
        if digest(here.parents[1] / 'instructions/assets' / name) != checksum:
            raise ValueError('A published figure changed after validation: ' + name)
    learner = Path(os.environ.get('ARENA_CCS_LEARNER_DIRECTORY', publication['learner_directory']))
    reused = os.environ.get('ARENA_CCS_REPRODUCTION_DIRECTORY')
    if reused:
        from chapter11_representation_geometry.exercises.part5_contrast_consistent_search.publish_study import validate_fresh
        output = Path(reused).resolve()
        _, _, fresh, _ = validate_fresh(learner, output)
    else:
        root = learner.parent / 'fresh_reproductions'
        root.mkdir(exist_ok=True)
        output = Path(tempfile.mkdtemp(prefix='run-', dir=root)) / 'fresh_cpu'
        fresh = reproduce(learner, output)
    evidence = here / 'expected_outputs/fresh_reproduction.json'
    evidence.write_bytes((output / 'reproduction.json').read_bytes())
    return dict(
        cuda_available=False, device='cpu', peak_vram_gb=0., within_vram_budget=True,
        native_model_reloaded=True, native_model_reloaded_in_this_report=not bool(reused),
        native_evidence_reused=bool(reused), actual_learner_passed=True, fresh_reproduction_passed=fresh['status'] == 'PASS',
        own_answer_code_cells=publication['code_cells'], embedded_figures=publication['embedded_figures'],
        final_prompt_count=fresh['final_prefix_examples'], distinct_final_base_reviews=64,
        retained_final_outcomes=fresh['final_reporter_scores'], fresh_compared_examples=fresh['all_prefix_examples_including_play'],
        fresh_condition_batches=len(fresh['cases']), refitted_reporters=fresh['refitted_reporters'],
        fresh_max_logit_error=fresh['numerical_comparison']['max_tensor_difference'],
        fresh_compared_tensors=fresh['numerical_comparison']['compared_tensors'],
        fresh_top_token_mismatches=fresh['winning_token_mismatches'],
        primary_metric_matches_actual_tokens=True, all_reporters_refitted=fresh['all_reporters_refitted'],
        weights_unchanged=fresh['weights_unchanged'], model_gradients_absent=fresh['model_gradients_absent'],
        fresh_output_directory=str(output), fresh_elapsed_seconds=fresh['elapsed_seconds'],
        learner_evidence_sha256=digest(publication_path), fresh_evidence_sha256=digest(evidence),
        final_used_for_selection=False, whole_paper_replication=False,
    )

def run_full_experiment(max_vram_gb=0.):
    return dict(numerical=run_smoke_test(), fresh_cpu_model=run_gpu_test(max_vram_gb))

# %%
