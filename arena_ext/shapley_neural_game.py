"""Reusable CUDA model organism for exact Shapley attribution sections."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import torch as t

NEURAL_GAME_SEED = 1621
NEURAL_GAME_NUM_PLAYERS = 4
NEURAL_GAME_HIDDEN_DIM = 64
NEURAL_GAME_STEPS = 1200
NEURAL_GAME_LR = 5e-2


class NeuralCoalitionGame(t.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = t.nn.Sequential(
            t.nn.Linear(NEURAL_GAME_NUM_PLAYERS, NEURAL_GAME_HIDDEN_DIM),
            t.nn.SiLU(),
            t.nn.Linear(NEURAL_GAME_HIDDEN_DIM, NEURAL_GAME_HIDDEN_DIM),
            t.nn.SiLU(),
            t.nn.Linear(NEURAL_GAME_HIDDEN_DIM, 1),
        )

    def forward(self, inputs: t.Tensor) -> t.Tensor:
        return self.net(inputs)


@dataclass(frozen=True)
class TrainedNeuralGame:
    model: NeuralCoalitionGame
    inputs: t.Tensor
    targets: t.Tensor
    fit_mse: float
    fit_max_abs_error: float


def binary_feature_table(device: t.device) -> t.Tensor:
    axes = [t.tensor([0.0, 1.0], device=device) for _ in range(NEURAL_GAME_NUM_PLAYERS)]
    return t.stack(t.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, NEURAL_GAME_NUM_PLAYERS)


def true_neural_game_scores(inputs: t.Tensor) -> t.Tensor:
    return (
        0.25
        + 1.2 * inputs[:, 0]
        - 0.7 * inputs[:, 1]
        + 1.6 * inputs[:, 2]
        + 0.9 * inputs[:, 3]
        + 2.2 * inputs[:, 0] * inputs[:, 2]
        - 1.5 * inputs[:, 1] * inputs[:, 3]
    ).unsqueeze(-1)


def train_neural_game(
    inputs: t.Tensor,
    targets: t.Tensor,
    *,
    seed: int = NEURAL_GAME_SEED,
) -> TrainedNeuralGame:
    t.manual_seed(seed)
    if inputs.device.type == "cuda":
        t.cuda.manual_seed_all(seed)
    model = NeuralCoalitionGame().to(inputs.device)
    optimizer = t.optim.AdamW(model.parameters(), lr=NEURAL_GAME_LR, weight_decay=0.0)
    for _ in range(NEURAL_GAME_STEPS):
        prediction = model(inputs)
        loss = ((prediction - targets) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with t.no_grad():
        residual = model(inputs) - targets
        fit_mse = float((residual**2).mean().item())
        fit_max_abs_error = float(residual.abs().max().item())
    return TrainedNeuralGame(
        model=model,
        inputs=inputs,
        targets=targets,
        fit_mse=fit_mse,
        fit_max_abs_error=fit_max_abs_error,
    )


def train_default_neural_game(device: t.device) -> TrainedNeuralGame:
    inputs = binary_feature_table(device)
    targets = true_neural_game_scores(inputs)
    return train_neural_game(inputs, targets)


def coalition_table_from_model(model: t.nn.Module, device: t.device) -> dict[frozenset[int], float]:
    target = t.ones(1, NEURAL_GAME_NUM_PLAYERS, device=device)
    values: dict[frozenset[int], float] = {}
    model.eval()
    with t.no_grad():
        for size in range(NEURAL_GAME_NUM_PLAYERS + 1):
            for group in itertools.combinations(range(NEURAL_GAME_NUM_PLAYERS), size):
                mask = t.zeros_like(target)
                if group:
                    mask[:, list(group)] = 1.0
                values[frozenset(group)] = float(model(target * mask).item())
    return values


def coalition_table_from_true_game(device: t.device) -> dict[frozenset[int], float]:
    target = t.ones(1, NEURAL_GAME_NUM_PLAYERS, device=device)
    values: dict[frozenset[int], float] = {}
    with t.no_grad():
        for size in range(NEURAL_GAME_NUM_PLAYERS + 1):
            for group in itertools.combinations(range(NEURAL_GAME_NUM_PLAYERS), size):
                mask = t.zeros_like(target)
                if group:
                    mask[:, list(group)] = 1.0
                values[frozenset(group)] = float(true_neural_game_scores(target * mask).item())
    return values


def shuffled_targets(targets: t.Tensor, *, seed: int = NEURAL_GAME_SEED) -> t.Tensor:
    generator = t.Generator(device=targets.device)
    generator.manual_seed(seed)
    return targets[t.randperm(targets.shape[0], device=targets.device, generator=generator)]
