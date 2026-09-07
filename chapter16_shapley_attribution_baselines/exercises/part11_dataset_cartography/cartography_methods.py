"""Reference answers; every taught operation is also visible in the notebook."""

from __future__ import annotations

import math

import torch
from torch import Tensor
import torch.nn.functional as F


def align_epoch(logits: Tensor, visited_ids: Tensor) -> Tensor:
    """Restore B × N × C logits to canonical integer IDs 0 through N-1."""
    if logits.ndim != 3 or visited_ids.shape != logits.shape[:2]:
        raise ValueError("Expected one visited ID per model and example.")
    expected = torch.arange(logits.shape[1], device=visited_ids.device)
    if not torch.equal(visited_ids.sort(dim=1).values, expected.expand_as(visited_ids)):
        raise ValueError(
            "Every epoch must visit each ID exactly once; check missing/duplicate IDs."
        )
    return logits.gather(1, visited_ids.argsort(dim=1)[..., None].expand_as(logits))


def gold_probabilities(logits: Tensor, labels: Tensor) -> Tensor:
    """Gather the assigned label, preserving any leading epoch dimension."""
    probabilities = logits.softmax(dim=-1)
    indices = labels.expand(logits.shape[:-1]).unsqueeze(-1)
    return probabilities.gather(-1, indices).squeeze(-1)


def training_statistics(logits: Tensor, labels: Tensor) -> dict[str, Tensor]:
    """Reduce E × B × N × C over epochs only; variability uses population SD."""
    if logits.ndim != 4 or logits.shape[0] == 0 or labels.shape != logits.shape[1:3]:
        raise ValueError("Use nonempty epoch/model/example/class logits and model/example labels.")
    probabilities = gold_probabilities(logits, labels)
    return dict(
        confidence=probabilities.mean(dim=0),
        variability=probabilities.std(dim=0, correction=0),
        correctness=(logits.argmax(dim=-1) == labels).float().mean(dim=0),
    )


def initialize_parameters(features, classes, width, generators):
    """Small architecture setup; independent generators also supply batch orders."""
    draws = []
    for generator in generators:
        if width:
            item = dict(
                w1=torch.randn(features, width, generator=generator) * math.sqrt(2 / features),
                b1=torch.zeros(1, width),
                w2=torch.randn(width, classes, generator=generator) / math.sqrt(width),
                b2=torch.zeros(1, classes),
            )
        else:
            item = dict(
                w=torch.randn(features, classes, generator=generator) / math.sqrt(features),
                b=torch.zeros(1, classes),
            )
        draws.append(item)
    return {name: torch.stack([item[name] for item in draws]).requires_grad_() for name in draws[0]}


def batched_logits(parameters: dict[str, Tensor], x: Tensor) -> Tensor:
    """Apply independent linear or one-hidden-layer networks along model axis B."""
    if "w1" in parameters:
        hidden = (x @ parameters["w1"] + parameters["b1"]).relu()
        return hidden @ parameters["w2"] + parameters["b2"]
    return x @ parameters["w"] + parameters["b"]


def train_ensemble(
    x: Tensor,
    labels: Tensor,
    seeds: list[int],
    width: int = 64,
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 0.05,
    momentum: float = 0.9,
    weight_decay: float = 0.0001,
    classes: int = 10,
    record: bool = True,
) -> dict:
    """Fresh SGD runs, with pre-update visits and epoch-end snapshots aligned by ID."""
    if x.ndim != 3 or labels.shape != x.shape[:2] or len(seeds) != x.shape[0]:
        raise ValueError("Provide B × N × D inputs, B × N labels and B run seeds.")
    if min(epochs, batch_size, x.shape[1]) < 1:
        raise ValueError("Epoch, minibatch and dataset sizes must be positive.")
    generators = [torch.Generator().manual_seed(seed) for seed in seeds]
    parameters = initialize_parameters(x.shape[-1], classes, width, generators)
    velocity = {name: torch.zeros_like(value) for name, value in parameters.items()}
    model_ids = torch.arange(len(seeds))[:, None]
    online_trace, end_trace, losses = [], [], []
    steps = 0
    for _ in range(epochs):
        order = torch.stack([torch.randperm(x.shape[1], generator=g) for g in generators])
        visits = []
        for start in range(0, x.shape[1], batch_size):
            ids = order[:, start : start + batch_size]
            batch_x, batch_y = x[model_ids, ids], labels[model_ids, ids]
            logits = batched_logits(parameters, batch_x)
            if record:
                visits.append(logits.detach().clone())
            individual_losses = (
                F.cross_entropy(logits.flatten(0, 1), batch_y.flatten(), reduction="none")
                .reshape_as(batch_y)
                .mean(dim=1)
            )
            # Sum independent model losses: averaging here would divide each gradient by B.
            gradients = torch.autograd.grad(individual_losses.sum(), tuple(parameters.values()))
            with torch.no_grad():
                for (name, value), gradient in zip(parameters.items(), gradients):
                    gradient = gradient + weight_decay * value
                    velocity[name].mul_(momentum).add_(gradient)
                    value.add_(velocity[name], alpha=-lr)
            steps += 1
        with torch.no_grad():
            end = batched_logits(parameters, x)
            losses.append(
                F.cross_entropy(end.flatten(0, 1), labels.flatten(), reduction="none")
                .reshape_as(labels)
                .mean(dim=1)
            )
            if record:
                online_trace.append(align_epoch(torch.cat(visits, dim=1), order))
                end_trace.append(end.clone())
    return dict(
        parameters={name: value.detach().clone() for name, value in parameters.items()},
        online_logits=torch.stack(online_trace) if record else torch.empty(0),
        end_logits=torch.stack(end_trace) if record else torch.empty(0),
        loss=torch.stack(losses),
        steps=steps,
        seeds=list(seeds),
        width=width,
        n=x.shape[1],
    )


def select_indices(score: Tensor, k: int, largest: bool = True) -> Tensor:
    """Rank examples; ties retain canonical source order."""
    if score.ndim != 1 or not 0 < k <= score.numel() or not torch.isfinite(score).all():
        raise ValueError("Use a finite score vector and a positive retained size at most N.")
    return torch.argsort(score, descending=largest, stable=True)[:k]


def matched_random_indices(labels: Tensor, selected: Tensor, seed: int) -> Tensor:
    """Uniform sampling within each observed class, matching selected label counts."""
    if selected.ndim != 1 or selected.unique().numel() != selected.numel():
        raise ValueError("Selected training indices must be a vector without duplicates.")
    generator = torch.Generator().manual_seed(seed)
    result = []
    for label in labels.unique(sorted=True):
        candidates = torch.where(labels == label)[0]
        count = int((labels[selected] == label).sum())
        order = torch.randperm(len(candidates), generator=generator)
        result.append(candidates[order[:count]])
    return torch.cat(result).sort().values


def retrieval_curve(score: Tensor, changed: Tensor, budgets: Tensor) -> dict[str, Tensor]:
    """Low-confidence retrieval of known injected errors, with stable ranking ties."""
    if changed.shape != score.shape or not changed.any() or changed.dtype != torch.bool:
        raise ValueError("Provide an aligned boolean error mask containing at least one error.")
    if budgets.ndim != 1 or not ((budgets > 0) & (budgets <= score.numel())).all():
        raise ValueError("Retrieval budgets must lie between one and the dataset size.")
    order = select_indices(score, len(score), largest=False)
    found = changed[order].cumsum(dim=0)[budgets - 1]
    return dict(found=found, precision=found / budgets, recall=found / changed.sum())


def selection_overlap(left: Tensor, right: Tensor) -> Tensor:
    """Jaccard overlap of two selected index sets, not agreement of their ordering."""
    if left.numel() == 0 or right.numel() == 0:
        raise ValueError("Compare nonempty selected sets.")
    intersection = torch.isin(left.unique(), right.unique()).sum()
    union = torch.cat([left, right]).unique().numel()
    return intersection.float() / union


def evaluate_models(parameters: dict[str, Tensor], x: Tensor, labels: Tensor) -> dict[str, Tensor]:
    """One fixed holdout, evaluated for every model; no selection or fitting."""
    models = len(next(iter(parameters.values())))
    with torch.no_grad():
        logits = batched_logits(parameters, x.expand(models, -1, -1))
        repeated_labels = labels.expand(models, -1)
        correct = logits.argmax(dim=-1) == repeated_labels
        losses = F.cross_entropy(
            logits.flatten(0, 1), repeated_labels.flatten(), reduction="none"
        ).reshape_as(repeated_labels)
        class_accuracy = torch.stack(
            [correct[:, labels == label].float().mean(dim=1) for label in range(logits.shape[-1])],
            dim=-1,
        )
    return dict(
        logits=logits,
        correct=correct,
        loss_per_example=losses,
        accuracy=correct.float().mean(dim=1),
        cross_entropy=losses.mean(dim=1),
        class_accuracy=class_accuracy,
    )
