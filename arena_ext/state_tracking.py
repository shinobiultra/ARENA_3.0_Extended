"""Synthetic state-tracking utilities for Mamba/world-model exercises."""

from __future__ import annotations

from dataclasses import dataclass

import torch as t
import torch.nn.functional as F


@dataclass(frozen=True)
class StateTrackingBatch:
    tokens: t.Tensor
    states: t.Tensor
    task: str
    vocab: dict[int, str]


@dataclass(frozen=True)
class LinearProbe:
    weight: t.Tensor
    bias: t.Tensor


@dataclass(frozen=True)
class ProbeReport:
    train_accuracy: float
    test_accuracy: float
    num_train: int
    num_test: int


@dataclass(frozen=True)
class InterventionReport:
    source_prediction: int
    target_prediction: int
    intervened_prediction: int
    target_logit_delta: float
    passed: bool


def generate_parity_task(batch: int, seq_len: int, seed: int = 0) -> StateTrackingBatch:
    """Generate binary-token cumulative parity states."""

    generator = t.Generator().manual_seed(seed)
    tokens = t.randint(0, 2, (batch, seq_len), generator=generator)
    states = tokens.cumsum(dim=-1) % 2
    return StateTrackingBatch(tokens=tokens, states=states, task="parity", vocab={0: "0", 1: "1"})


def generate_bracket_depth_task(
    batch: int,
    seq_len: int,
    *,
    max_depth: int = 4,
    seed: int = 0,
) -> StateTrackingBatch:
    """Generate bracket actions with nonnegative bounded depth labels."""

    generator = t.Generator().manual_seed(seed)
    tokens = t.zeros(batch, seq_len, dtype=t.long)
    states = t.zeros(batch, seq_len, dtype=t.long)
    depth = t.zeros(batch, dtype=t.long)

    for pos in range(seq_len):
        proposed_open = t.randint(0, 2, (batch,), generator=generator).bool()
        must_open = depth == 0
        must_close = depth == max_depth
        open_token = (proposed_open | must_open) & ~must_close
        delta = t.where(open_token, t.ones_like(depth), -t.ones_like(depth))
        depth = depth + delta
        tokens[:, pos] = open_token.long()
        states[:, pos] = depth

    return StateTrackingBatch(
        tokens=tokens,
        states=states,
        task="bracket_depth",
        vocab={0: ")", 1: "("},
    )


def one_hot_state_features(
    states: t.Tensor,
    num_states: int | None = None,
    *,
    noise_scale: float = 0.0,
    seed: int = 0,
) -> t.Tensor:
    """Create hidden states where the latent state is linearly accessible."""

    if num_states is None:
        num_states = int(states.max().item()) + 1
    features = F.one_hot(states.long(), num_classes=num_states).float()
    if noise_scale > 0:
        generator = t.Generator(device=features.device).manual_seed(seed)
        noise = t.randn(features.shape, generator=generator, device=features.device)
        features = features + noise_scale * noise
    return features


def make_position_split(
    states: t.Tensor,
    *,
    train_fraction: float = 0.5,
) -> tuple[t.Tensor, t.Tensor]:
    """Return boolean train/test masks split by sequence position."""

    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")
    seq_len = states.shape[-1]
    split = max(1, min(seq_len - 1, int(seq_len * train_fraction)))
    position = t.arange(seq_len, device=states.device)
    train_mask = position[None, :] < split
    test_mask = ~train_mask
    return train_mask.expand_as(states), test_mask.expand_as(states)


def _flatten_masked(hidden_states: t.Tensor, labels: t.Tensor, mask: t.Tensor | None):
    if hidden_states.shape[:-1] != labels.shape:
        raise ValueError("hidden_states leading dimensions must match labels.")
    flat_x = hidden_states.reshape(-1, hidden_states.shape[-1]).float()
    flat_y = labels.reshape(-1).long()
    if mask is None:
        return flat_x, flat_y
    flat_mask = mask.reshape(-1).bool()
    return flat_x[flat_mask], flat_y[flat_mask]


def fit_linear_probe(
    hidden_states: t.Tensor,
    labels: t.Tensor,
    *,
    num_classes: int | None = None,
    train_mask: t.Tensor | None = None,
    ridge: float = 1e-3,
) -> LinearProbe:
    """Fit a ridge-regression classifier on hidden states."""

    x, y = _flatten_masked(hidden_states, labels, train_mask)
    if num_classes is None:
        num_classes = int(labels.max().item()) + 1

    y_one_hot = F.one_hot(y, num_classes=num_classes).float()
    ones = t.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)
    x_aug = t.cat([x, ones], dim=-1)
    eye = t.eye(x_aug.shape[-1], device=x.device, dtype=x.dtype)
    eye[-1, -1] = 0.0
    lhs = x_aug.T @ x_aug + ridge * eye
    rhs = x_aug.T @ y_one_hot
    solution = t.linalg.solve(lhs, rhs)
    return LinearProbe(weight=solution[:-1], bias=solution[-1])


def probe_logits(hidden_states: t.Tensor, probe: LinearProbe) -> t.Tensor:
    return hidden_states.float() @ probe.weight + probe.bias


def probe_predictions(hidden_states: t.Tensor, probe: LinearProbe) -> t.Tensor:
    return probe_logits(hidden_states, probe).argmax(dim=-1)


def probe_accuracy(
    hidden_states: t.Tensor,
    labels: t.Tensor,
    probe: LinearProbe,
    mask: t.Tensor | None = None,
) -> float:
    predictions = probe_predictions(hidden_states, probe)
    if mask is not None:
        predictions = predictions[mask.bool()]
        labels = labels[mask.bool()]
    return predictions.eq(labels).float().mean().item()


def evaluate_probe_generalization(
    hidden_states: t.Tensor,
    labels: t.Tensor,
    probe: LinearProbe,
    train_mask: t.Tensor,
    test_mask: t.Tensor,
) -> ProbeReport:
    """Evaluate a probe on train positions and held-out later positions."""

    return ProbeReport(
        train_accuracy=probe_accuracy(hidden_states, labels, probe, train_mask),
        test_accuracy=probe_accuracy(hidden_states, labels, probe, test_mask),
        num_train=int(train_mask.sum().item()),
        num_test=int(test_mask.sum().item()),
    )


def state_intervention_direction(
    probe: LinearProbe,
    source_state: int,
    target_state: int,
) -> t.Tensor:
    """Direction that raises target-state logit relative to source-state logit."""

    if source_state == target_state:
        raise ValueError("source_state and target_state must differ.")
    return probe.weight[:, target_state] - probe.weight[:, source_state]


def apply_state_intervention(
    hidden_state: t.Tensor,
    probe: LinearProbe,
    *,
    source_state: int,
    target_state: int,
    coefficient: float = 1.0,
) -> t.Tensor:
    """Move a hidden state along a probe-derived target-minus-source direction."""

    direction = state_intervention_direction(probe, source_state, target_state)
    direction = direction.to(device=hidden_state.device, dtype=hidden_state.dtype)
    return hidden_state + coefficient * direction


def intervention_report(
    hidden_state: t.Tensor,
    probe: LinearProbe,
    *,
    source_state: int,
    target_state: int,
    coefficient: float = 1.0,
) -> InterventionReport:
    """Check whether a probe-derived intervention flips prediction as intended."""

    logits_before = probe_logits(hidden_state, probe)
    intervened = apply_state_intervention(
        hidden_state,
        probe,
        source_state=source_state,
        target_state=target_state,
        coefficient=coefficient,
    )
    logits_after = probe_logits(intervened, probe)
    source_prediction = int(logits_before.argmax(dim=-1).item())
    intervened_prediction = int(logits_after.argmax(dim=-1).item())
    target_delta = (logits_after[..., target_state] - logits_before[..., target_state]).item()
    return InterventionReport(
        source_prediction=source_prediction,
        target_prediction=target_state,
        intervened_prediction=intervened_prediction,
        target_logit_delta=float(target_delta),
        passed=intervened_prediction == target_state,
    )


def random_direction_control(
    hidden_state: t.Tensor,
    probe: LinearProbe,
    *,
    target_state: int,
    coefficient: float = 1.0,
    seed: int = 0,
) -> int:
    """Apply a random direction with matched norm and return the new prediction."""

    generator = t.Generator(device=hidden_state.device).manual_seed(seed)
    random_dir = t.randn(hidden_state.shape, generator=generator, device=hidden_state.device)
    target_dir = probe.weight[:, target_state].to(device=hidden_state.device)
    random_dir = random_dir / random_dir.norm().clamp_min(1e-8) * target_dir.norm()
    logits = probe_logits(hidden_state + coefficient * random_dir, probe)
    return int(logits.argmax(dim=-1).item())
