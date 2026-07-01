# %%
"""Reference solutions for [16.3] Shapley Interactions with shapiq."""

import itertools
import math
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import torch as t

chapter = "chapter16_shapley_attribution_baselines"
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

MAIN = __name__ == "__main__"

Coalition = frozenset[int]

NEURAL_GAME_SEED = 1621
NEURAL_GAME_NUM_PLAYERS = 4
NEURAL_GAME_HIDDEN_DIM = 64
NEURAL_GAME_STEPS = 1200
NEURAL_GAME_LR = 5e-2

NEURAL_INTERACTION_MAX_ERROR = 1e-4
NEURAL_SPURIOUS_MAX_INTERACTION = 1e-4
NEURAL_FIT_MAX_MSE = 1e-8
SHAPIQ_MAX_ERROR = 1e-5
SHUFFLED_INTERACTION_ERROR_MIN = 1.0


# %%
@dataclass(frozen=True)
class PairwiseInteractionReport:
    pair_interactions: t.Tensor
    target_pair: tuple[int, int]
    target_interaction: float
    max_spurious_interaction: float
    recovers_interaction: bool


@dataclass(frozen=True)
class ShapiqInteractionParityReport:
    exact_pair_interactions: t.Tensor
    shapiq_pair_interactions: t.Tensor
    max_abs_error: float
    matches_shapiq: bool
    shapiq_available: bool


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


# %%
def all_coalitions(num_players: int) -> tuple[Coalition, ...]:
    """Return every coalition for `num_players` players."""

    if num_players <= 0:
        raise ValueError("num_players must be positive.")
    players = range(num_players)
    coalitions = []
    for size in range(num_players + 1):
        coalitions.extend(frozenset(group) for group in itertools.combinations(players, size))
    return tuple(coalitions)


def normalize_coalition_values(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    *,
    num_players: int,
) -> dict[Coalition, float]:
    """Normalize coalition keys and require one value for every coalition."""

    values = {frozenset(key): float(value) for key, value in coalition_values.items()}
    expected = set(all_coalitions(num_players))
    missing = expected - set(values)
    if missing:
        raise ValueError(f"coalition value table is missing {len(missing)} coalitions.")
    return values


def coalition_values_from_function(
    num_players: int,
    value_fn: Callable[[Coalition], float],
) -> dict[Coalition, float]:
    """Evaluate a set function on every coalition."""

    return {coalition: float(value_fn(coalition)) for coalition in all_coalitions(num_players)}


def additive_game(weights: t.Tensor) -> dict[Coalition, float]:
    """Return coalition values for an additive linear game."""

    weights = weights.flatten().double()
    return coalition_values_from_function(
        int(weights.numel()),
        lambda coalition: weights[list(coalition)].sum().item() if coalition else 0.0,
    )


def interaction_game(
    num_players: int,
    *,
    pair: tuple[int, int] = (0, 1),
    pair_weight: float = 1.0,
    additive_weights: t.Tensor | None = None,
) -> dict[Coalition, float]:
    """Return a complete value table with one pair interaction plus additive terms."""

    if num_players < 2:
        raise ValueError("interaction_game requires at least two players.")
    first, second = pair
    if first == second:
        raise ValueError("pair must contain two different players.")
    if not (0 <= first < num_players and 0 <= second < num_players):
        raise ValueError("pair players must be valid player indices.")
    if additive_weights is None:
        weights = t.zeros(num_players, dtype=t.float64)
    else:
        weights = additive_weights.flatten().double()
        if int(weights.numel()) != num_players:
            raise ValueError("additive_weights must have one weight per player.")

    def value_fn(coalition: Coalition) -> float:
        additive = weights[list(coalition)].sum().item() if coalition else 0.0
        interaction = pair_weight if first in coalition and second in coalition else 0.0
        return additive + interaction

    return coalition_values_from_function(num_players, value_fn)


def pairwise_shapley_interactions(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    *,
    num_players: int,
) -> t.Tensor:
    """Compute exact second-order Shapley interaction indices for every pair."""

    if num_players < 2:
        raise ValueError("pairwise interactions require at least two players.")
    values = normalize_coalition_values(coalition_values, num_players=num_players)
    interactions = t.zeros((num_players, num_players), dtype=t.float64)
    denominator = math.factorial(num_players - 1)
    for first, second in itertools.combinations(range(num_players), 2):
        others = [player for player in range(num_players) if player not in (first, second)]
        score = 0.0
        for size in range(num_players - 1):
            weight = (
                math.factorial(size)
                * math.factorial(num_players - size - 2)
                / denominator
            )
            for group in itertools.combinations(others, size):
                coalition = frozenset(group)
                delta = (
                    values[coalition | {first, second}]
                    - values[coalition | {first}]
                    - values[coalition | {second}]
                    + values[coalition]
                )
                score += weight * delta
        interactions[first, second] = score
        interactions[second, first] = score
    return interactions


def pairwise_interaction_report(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    *,
    num_players: int,
    target_pair: tuple[int, int] = (0, 1),
    expected_target: float = 1.0,
    tolerance: float = 1e-9,
) -> PairwiseInteractionReport:
    """Check that exactly one target pair has the expected interaction."""

    first, second = target_pair
    if first == second:
        raise ValueError("target_pair must contain two different players.")
    if not (0 <= first < num_players and 0 <= second < num_players):
        raise ValueError("target_pair players must be valid player indices.")

    interactions = pairwise_shapley_interactions(coalition_values, num_players=num_players)
    target_value = float(interactions[first, second].item())
    spurious = [
        abs(float(interactions[row, col].item()))
        for row, col in itertools.combinations(range(num_players), 2)
        if frozenset((row, col)) != frozenset(target_pair)
    ]
    max_spurious = max(spurious, default=0.0)
    return PairwiseInteractionReport(
        pair_interactions=interactions,
        target_pair=target_pair,
        target_interaction=target_value,
        max_spurious_interaction=max_spurious,
        recovers_interaction=(
            abs(target_value - expected_target) <= tolerance
            and max_spurious <= tolerance
        ),
    )


def shapiq_pairwise_interactions(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    *,
    num_players: int,
    index: str = "SII",
) -> t.Tensor:
    """Compute pairwise interactions through shapiq on a complete value table."""

    values = normalize_coalition_values(coalition_values, num_players=num_players)
    try:
        import numpy as np
        import shapiq
    except ImportError as exc:  # pragma: no cover - environment contract catches this.
        raise RuntimeError("shapiq is required for shapiq parity checks.") from exc

    def game(coalitions):
        array = np.asarray(coalitions, dtype=bool)
        if array.ndim == 1:
            array = array[None, :]
        outputs = []
        for row in array:
            coalition = frozenset(
                player for player, present in enumerate(row.tolist()) if present
            )
            outputs.append(values[coalition])
        return np.asarray(outputs, dtype=float)

    explainer = shapiq.AgnosticExplainer(
        game,
        n_players=num_players,
        index=index,
        max_order=2,
        random_state=0,
    )
    interaction_values = explainer.explain(budget=2**num_players, random_state=0)
    matrix = t.zeros((num_players, num_players), dtype=t.float64)
    for first, second in itertools.combinations(range(num_players), 2):
        lookup = interaction_values.interaction_lookup
        lookup_key = (first, second)
        reverse_key = (second, first)
        value_index = lookup[lookup_key] if lookup_key in lookup else lookup[reverse_key]
        value = float(interaction_values.values[value_index])
        matrix[first, second] = value
        matrix[second, first] = value
    return matrix


def shapiq_interaction_parity_report(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    *,
    num_players: int,
    index: str = "SII",
    tolerance: float = 1e-6,
) -> ShapiqInteractionParityReport:
    """Compare closed-form pairwise interactions to shapiq on the same table."""

    exact = pairwise_shapley_interactions(coalition_values, num_players=num_players)
    try:
        observed = shapiq_pairwise_interactions(
            coalition_values,
            num_players=num_players,
            index=index,
        )
        available = True
    except RuntimeError:
        observed = t.full_like(exact, float("nan"))
        available = False
    max_error = float((exact - observed).abs().max().item()) if available else float("inf")
    return ShapiqInteractionParityReport(
        exact_pair_interactions=exact,
        shapiq_pair_interactions=observed,
        max_abs_error=max_error,
        matches_shapiq=available and max_error <= tolerance,
        shapiq_available=available,
    )


# %%
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


def coalition_table_from_model(model: t.nn.Module, device: t.device) -> dict[Coalition, float]:
    target = t.ones(1, NEURAL_GAME_NUM_PLAYERS, device=device)
    values: dict[Coalition, float] = {}
    model.eval()
    with t.no_grad():
        for size in range(NEURAL_GAME_NUM_PLAYERS + 1):
            for group in itertools.combinations(range(NEURAL_GAME_NUM_PLAYERS), size):
                mask = t.zeros_like(target)
                if group:
                    mask[:, list(group)] = 1.0
                values[frozenset(group)] = float(model(target * mask).item())
    return values


def coalition_table_from_true_game(device: t.device) -> dict[Coalition, float]:
    target = t.ones(1, NEURAL_GAME_NUM_PLAYERS, device=device)
    values: dict[Coalition, float] = {}
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


# %%
def _tensor_report(report) -> dict:
    result = report.__dict__.copy()
    for key, value in list(result.items()):
        if hasattr(value, "tolist"):
            result[key] = value.tolist()
    return result


def additive_interaction_smoke_test() -> dict:
    values = additive_game(t.tensor([1.0, -2.0, 0.5]))
    interactions = pairwise_shapley_interactions(values, num_players=3)
    return {
        "pair_interactions": interactions.tolist(),
        "max_abs_interaction": float(interactions.abs().max().item()),
    }


def target_pair_interaction_smoke_test() -> dict:
    values = interaction_game(
        3,
        pair=(0, 1),
        pair_weight=1.0,
        additive_weights=t.tensor([0.5, -1.0, 2.0]),
    )
    return _tensor_report(pairwise_interaction_report(values, num_players=3))


def shapiq_parity_smoke_test() -> dict:
    values = interaction_game(
        3,
        pair=(0, 1),
        pair_weight=1.0,
        additive_weights=t.tensor([0.5, -1.0, 2.0]),
    )
    return _tensor_report(shapiq_interaction_parity_report(values, num_players=3))


def run_smoke_test(cpu: bool = True) -> dict:
    _ = cpu
    return {
        "additive_interactions": additive_interaction_smoke_test(),
        "target_pair": target_pair_interaction_smoke_test(),
        "shapiq_parity": shapiq_parity_smoke_test(),
    }


def run_gpu_test(max_vram_gb: float = 24.0) -> dict:
    if not t.cuda.is_available():
        raise RuntimeError("16.3 GPU preflight requires CUDA; no CPU fallback is accepted.")

    return run_neural_interaction_preflight(max_vram_gb=max_vram_gb)


def run_neural_interaction_preflight(max_vram_gb: float = 24.0) -> dict:
    device = t.device("cuda")
    t.cuda.reset_peak_memory_stats()

    inputs = binary_feature_table(device)
    targets = true_neural_game_scores(inputs)
    trained = train_neural_game(inputs, targets)
    model_values = coalition_table_from_model(trained.model, device)
    true_values = coalition_table_from_true_game(device)
    model_interactions = pairwise_shapley_interactions(
        model_values,
        num_players=NEURAL_GAME_NUM_PLAYERS,
    )
    true_interactions = pairwise_shapley_interactions(
        true_values,
        num_players=NEURAL_GAME_NUM_PLAYERS,
    )
    interaction_max_abs_error = float((model_interactions - true_interactions).abs().max().item())

    positive_pair = (0, 2)
    negative_pair = (1, 3)
    positive_value = float(model_interactions[positive_pair].item())
    negative_value = float(model_interactions[negative_pair].item())
    positive_expected = float(true_interactions[positive_pair].item())
    negative_expected = float(true_interactions[negative_pair].item())
    positive_abs_error = abs(positive_value - positive_expected)
    negative_abs_error = abs(negative_value - negative_expected)
    spurious_pairs = [
        (row, col)
        for row in range(NEURAL_GAME_NUM_PLAYERS)
        for col in range(row + 1, NEURAL_GAME_NUM_PLAYERS)
        if (row, col) not in {positive_pair, negative_pair}
    ]
    max_spurious_interaction = max(
        abs(float(model_interactions[row, col].item()))
        for row, col in spurious_pairs
    )

    shapiq_report = shapiq_interaction_parity_report(
        model_values,
        num_players=NEURAL_GAME_NUM_PLAYERS,
        tolerance=SHAPIQ_MAX_ERROR,
    )
    shuffled = train_neural_game(inputs, shuffled_targets(targets))
    shuffled_values = coalition_table_from_model(shuffled.model, device)
    shuffled_interactions = pairwise_shapley_interactions(
        shuffled_values,
        num_players=NEURAL_GAME_NUM_PLAYERS,
    )
    shuffled_interaction_error = float(
        (shuffled_interactions - true_interactions).abs().max().item()
    )

    t.cuda.synchronize()
    peak_vram_gb = t.cuda.max_memory_allocated() / 1024**3
    signs_recovered = positive_value > 0.0 and negative_value < 0.0
    shuffled_rejected = shuffled_interaction_error >= SHUFFLED_INTERACTION_ERROR_MIN
    preflight_passed = (
        trained.fit_mse <= NEURAL_FIT_MAX_MSE
        and interaction_max_abs_error <= NEURAL_INTERACTION_MAX_ERROR
        and positive_abs_error <= NEURAL_INTERACTION_MAX_ERROR
        and negative_abs_error <= NEURAL_INTERACTION_MAX_ERROR
        and max_spurious_interaction <= NEURAL_SPURIOUS_MAX_INTERACTION
        and signs_recovered
        and shapiq_report.matches_shapiq
        and shapiq_report.max_abs_error <= SHAPIQ_MAX_ERROR
        and shuffled_rejected
        and peak_vram_gb <= max_vram_gb
    )
    return {
        "preflight_passed": preflight_passed,
        "cuda_available": True,
        "device": t.cuda.get_device_name(0),
        "model_family": "cuda_trained_neural_coalition_game_mlp",
        "num_players": NEURAL_GAME_NUM_PLAYERS,
        "coalition_count": 2**NEURAL_GAME_NUM_PLAYERS,
        "training_example_count": int(inputs.shape[0]),
        "training_steps": NEURAL_GAME_STEPS,
        "fit_mse": trained.fit_mse,
        "fit_max_abs_error": trained.fit_max_abs_error,
        "positive_interaction_pair": list(positive_pair),
        "positive_interaction_value": positive_value,
        "positive_interaction_expected": positive_expected,
        "positive_interaction_abs_error": positive_abs_error,
        "negative_interaction_pair": list(negative_pair),
        "negative_interaction_value": negative_value,
        "negative_interaction_expected": negative_expected,
        "negative_interaction_abs_error": negative_abs_error,
        "interaction_max_abs_error": interaction_max_abs_error,
        "max_spurious_interaction": max_spurious_interaction,
        "interaction_signs_recovered": signs_recovered,
        "shapiq_available": shapiq_report.shapiq_available,
        "shapiq_matches": shapiq_report.matches_shapiq,
        "shapiq_max_abs_error": shapiq_report.max_abs_error,
        "shuffled_control_fit_mse": shuffled.fit_mse,
        "shuffled_control_interaction_error": shuffled_interaction_error,
        "shuffled_control_rejected": shuffled_rejected,
        "peak_vram_gb": peak_vram_gb,
        "within_vram_budget": peak_vram_gb <= max_vram_gb,
        "full_path": "Train a CUDA MLP on a complete binary feature game, recover planted pair interactions from real model ablations, validate shapiq parity, and reject shuffled-label interactions.",
    }


def run_full_experiment(max_vram_gb: float = 24.0) -> dict:
    """Run the validated experiment path used by the verification report."""

    return run_gpu_test(max_vram_gb=max_vram_gb)


if MAIN:
    print(run_smoke_test())
