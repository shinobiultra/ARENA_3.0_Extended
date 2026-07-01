# %%
"""Reference solutions for [16.2] KernelSHAP and PartitionSHAP Controls."""

import itertools
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import torch as t

Coalition = frozenset[int]
MAIN = __name__ == "__main__"

NEURAL_GAME_SEED = 1621
NEURAL_GAME_NUM_PLAYERS = 4
NEURAL_GAME_HIDDEN_DIM = 64
NEURAL_GAME_STEPS = 1200
NEURAL_GAME_LR = 5e-2

NEURAL_KERNEL_MAX_ERROR = 1e-8
NEURAL_KERNEL_TRUE_MAX_ERROR = 1e-4
NEURAL_SINGLETON_PARTITION_MAX_ERROR = 1e-8
NEURAL_FIT_MAX_MSE = 1e-8
GROUPED_PARTITION_TRUE_MAX_ERROR = 1e-4
CROSS_GROUP_IRRELEVANT_CREDIT_MAX = 1e-8
SHUFFLED_CONTROL_TRUE_ERROR_MIN = 1.0


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
class KernelSHAPApproximationReport:
    shapley_values: t.Tensor
    exact_values: t.Tensor
    max_abs_error: float
    approximates_exact: bool


@dataclass(frozen=True)
class PartitionSHAPReport:
    group_values: t.Tensor
    player_values: t.Tensor
    exact_values: t.Tensor
    max_abs_error: float
    recovers_exact: bool


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
    """Normalize coalition keys and require a complete value table."""

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


def exact_shapley_values(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    *,
    num_players: int,
) -> t.Tensor:
    """Compute exact Shapley values by summing weighted marginal effects."""

    values = normalize_coalition_values(coalition_values, num_players=num_players)
    denominator = math.factorial(num_players)
    shapley = t.zeros(num_players, dtype=t.float64)
    for player in range(num_players):
        others = [item for item in range(num_players) if item != player]
        for size in range(num_players):
            weight = (
                math.factorial(size)
                * math.factorial(num_players - size - 1)
                / denominator
            )
            for group in itertools.combinations(others, size):
                coalition = frozenset(group)
                shapley[player] += weight * (
                    values[coalition | {player}] - values[coalition]
                )
    return shapley


def additive_game(weights: t.Tensor) -> dict[Coalition, float]:
    """Return coalition values for an additive linear game."""

    weights = weights.flatten().double()
    return coalition_values_from_function(
        int(weights.numel()),
        lambda coalition: weights[list(coalition)].sum().item() if coalition else 0.0,
    )


def conjunction_game(num_players: int) -> dict[Coalition, float]:
    """Return a Boolean AND game where only the full coalition has value one."""

    full = frozenset(range(num_players))
    return coalition_values_from_function(num_players, lambda coalition: coalition == full)


def kernelshap_kernel_weight(coalition_size: int, num_players: int) -> float:
    """Return the finite KernelSHAP weight for a non-empty, non-full coalition."""

    if coalition_size <= 0 or coalition_size >= num_players:
        raise ValueError("KernelSHAP finite weights require 0 < coalition_size < num_players.")
    return (num_players - 1) / (
        math.comb(num_players, coalition_size)
        * coalition_size
        * (num_players - coalition_size)
    )


def kernelshap_values(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    *,
    num_players: int,
) -> t.Tensor:
    """Solve full-table KernelSHAP as a constrained weighted regression."""

    if num_players < 2:
        raise ValueError("KernelSHAP needs at least two players for finite weights.")
    values = normalize_coalition_values(coalition_values, num_players=num_players)
    empty = frozenset()
    full = frozenset(range(num_players))
    baseline = values[empty]
    total_delta = values[full] - baseline

    rows = []
    targets = []
    weights = []
    for coalition in all_coalitions(num_players):
        size = len(coalition)
        if size == 0 or size == num_players:
            continue
        rows.append([1.0 if player in coalition else 0.0 for player in range(num_players)])
        targets.append(values[coalition] - baseline)
        weights.append(kernelshap_kernel_weight(size, num_players))

    design = t.tensor(rows, dtype=t.float64)
    target = t.tensor(targets, dtype=t.float64)
    weight_vector = t.tensor(weights, dtype=t.float64)
    hessian = design.T @ (design * weight_vector.unsqueeze(-1))
    rhs = design.T @ (target * weight_vector)

    constraint = t.ones((1, num_players), dtype=t.float64)
    kkt = t.cat(
        [
            t.cat([hessian, constraint.T], dim=1),
            t.cat([constraint, t.zeros((1, 1), dtype=t.float64)], dim=1),
        ],
        dim=0,
    )
    constrained_rhs = t.cat([rhs, t.tensor([total_delta], dtype=t.float64)])
    return t.linalg.solve(kkt, constrained_rhs)[:num_players]


def kernelshap_approximation_report(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    *,
    num_players: int,
    tolerance: float = 1e-9,
) -> KernelSHAPApproximationReport:
    """Compare full-table KernelSHAP against exact Shapley values."""

    kernel_values = kernelshap_values(coalition_values, num_players=num_players)
    exact = exact_shapley_values(coalition_values, num_players=num_players)
    max_error = float((kernel_values - exact).abs().max().item())
    return KernelSHAPApproximationReport(
        shapley_values=kernel_values,
        exact_values=exact,
        max_abs_error=max_error,
        approximates_exact=max_error <= tolerance,
    )


def grouped_coalition_values(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    *,
    groups: tuple[tuple[int, ...], ...],
) -> dict[Coalition, float]:
    """Collapse player coalitions into group coalitions for Owen values."""

    group_count = len(groups)
    player_count = sum(len(group) for group in groups)
    values = normalize_coalition_values(coalition_values, num_players=player_count)
    group_values = {}
    for group_coalition in all_coalitions(group_count):
        players: set[int] = set()
        for group_index in group_coalition:
            players.update(groups[group_index])
        group_values[group_coalition] = values[frozenset(players)]
    return group_values


def partition_shap_values(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    *,
    groups: tuple[tuple[int, ...], ...],
) -> t.Tensor:
    """Compute exact Owen values for a fixed partition of players."""

    if not groups:
        raise ValueError("groups must be nonempty.")
    flattened = [player for group in groups for player in group]
    if sorted(flattened) != list(range(len(flattened))):
        raise ValueError("groups must partition players 0..n-1 exactly once.")

    values = normalize_coalition_values(coalition_values, num_players=len(flattened))
    group_count = len(groups)
    player_values = t.zeros(len(flattened), dtype=t.float64)
    for group_index, group in enumerate(groups):
        outside_group_indices = [idx for idx in range(group_count) if idx != group_index]
        group_denominator = math.factorial(group_count)
        local_denominator = math.factorial(len(group))
        for local_index, player in enumerate(group):
            others_in_group = [
                candidate
                for idx, candidate in enumerate(group)
                if idx != local_index
            ]
            for outside_size in range(group_count):
                outside_weight = (
                    math.factorial(outside_size)
                    * math.factorial(group_count - outside_size - 1)
                    / group_denominator
                )
                for outside_groups in itertools.combinations(outside_group_indices, outside_size):
                    outside_players: set[int] = set()
                    for outside_group in outside_groups:
                        outside_players.update(groups[outside_group])
                    for inside_size in range(len(group)):
                        inside_weight = (
                            math.factorial(inside_size)
                            * math.factorial(len(group) - inside_size - 1)
                            / local_denominator
                        )
                        for inside_players in itertools.combinations(
                            others_in_group,
                            inside_size,
                        ):
                            coalition = frozenset(outside_players | set(inside_players))
                            player_values[player] += outside_weight * inside_weight * (
                                values[coalition | {player}] - values[coalition]
                            )
    return player_values


def partition_shap_report(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    *,
    groups: tuple[tuple[int, ...], ...],
    tolerance: float = 1e-9,
) -> PartitionSHAPReport:
    """Compare PartitionSHAP/Owen values to exact player Shapley values."""

    num_players = sum(len(group) for group in groups)
    player_values = partition_shap_values(coalition_values, groups=groups)
    exact = exact_shapley_values(coalition_values, num_players=num_players)
    group_values = exact_shapley_values(
        grouped_coalition_values(coalition_values, groups=groups),
        num_players=len(groups),
    )
    max_error = float((player_values - exact).abs().max().item())
    return PartitionSHAPReport(
        group_values=group_values,
        player_values=player_values,
        exact_values=exact,
        max_abs_error=max_error,
        recovers_exact=max_error <= tolerance,
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
def _tensor_report(report: object) -> dict:
    result = report.__dict__.copy()
    for key, value in list(result.items()):
        if hasattr(value, "tolist"):
            result[key] = value.tolist()
    return result


def kernel_additive_smoke_test() -> dict:
    values = additive_game(t.tensor([1.0, -2.0, 0.5]))
    return _tensor_report(kernelshap_approximation_report(values, num_players=3))


def kernel_interaction_smoke_test() -> dict:
    values = conjunction_game(3)
    return _tensor_report(kernelshap_approximation_report(values, num_players=3))


def partition_additive_smoke_test() -> dict:
    values = additive_game(t.tensor([1.0, 2.0, 3.0, 4.0]))
    return _tensor_report(partition_shap_report(values, groups=((0, 1), (2, 3))))


def partition_interaction_smoke_test() -> dict:
    values = conjunction_game(2)
    return _tensor_report(partition_shap_report(values, groups=((0, 1),)))


def partition_cross_group_interaction_smoke_test() -> dict:
    values = coalition_values_from_function(
        3,
        lambda coalition: {0, 2}.issubset(coalition),
    )
    return _tensor_report(partition_shap_report(values, groups=((0, 1), (2,))))


def run_smoke_test(cpu: bool = True) -> dict:
    _ = cpu
    return {
        "kernel_additive": kernel_additive_smoke_test(),
        "kernel_interaction": kernel_interaction_smoke_test(),
        "partition_additive": partition_additive_smoke_test(),
        "partition_interaction": partition_interaction_smoke_test(),
        "partition_cross_group_interaction": partition_cross_group_interaction_smoke_test(),
    }


def run_gpu_test(max_vram_gb: float = 24.0) -> dict:
    if not t.cuda.is_available():
        raise RuntimeError("16.2 GPU preflight requires CUDA; no CPU fallback is accepted.")

    return run_neural_kernel_partition_preflight(max_vram_gb=max_vram_gb)


def run_neural_kernel_partition_preflight(max_vram_gb: float = 24.0) -> dict:
    device = t.device("cuda")
    t.cuda.reset_peak_memory_stats()

    inputs = binary_feature_table(device)
    targets = true_neural_game_scores(inputs)
    trained = train_neural_game(inputs, targets)
    model_values = coalition_table_from_model(trained.model, device)
    true_values = coalition_table_from_true_game(device)
    true_exact = exact_shapley_values(true_values, num_players=NEURAL_GAME_NUM_PLAYERS)

    kernel = kernelshap_approximation_report(
        model_values,
        num_players=NEURAL_GAME_NUM_PLAYERS,
        tolerance=NEURAL_KERNEL_MAX_ERROR,
    )
    kernel_vs_true_error = float((kernel.shapley_values - true_exact).abs().max().item())

    singleton_partition = partition_shap_report(
        model_values,
        groups=((0,), (1,), (2,), (3,)),
        tolerance=NEURAL_SINGLETON_PARTITION_MAX_ERROR,
    )
    aligned_partition = partition_shap_report(
        model_values,
        groups=((0, 2), (1, 3)),
        tolerance=GROUPED_PARTITION_TRUE_MAX_ERROR,
    )
    mismatched_partition = partition_shap_report(
        model_values,
        groups=((0, 1), (2, 3)),
        tolerance=GROUPED_PARTITION_TRUE_MAX_ERROR,
    )
    cross_group_values = coalition_values_from_function(
        3,
        lambda coalition: {0, 2}.issubset(coalition),
    )
    cross_group_partition = partition_shap_report(
        cross_group_values,
        groups=((0, 1), (2,)),
        tolerance=CROSS_GROUP_IRRELEVANT_CREDIT_MAX,
    )

    shuffled = train_neural_game(inputs, shuffled_targets(targets))
    shuffled_values = coalition_table_from_model(shuffled.model, device)
    shuffled_kernel = kernelshap_approximation_report(
        shuffled_values,
        num_players=NEURAL_GAME_NUM_PLAYERS,
        tolerance=NEURAL_KERNEL_MAX_ERROR,
    )
    shuffled_kernel_true_error = float((shuffled_kernel.shapley_values - true_exact).abs().max().item())

    t.cuda.synchronize()
    peak_vram_gb = t.cuda.max_memory_allocated() / 1024**3
    cross_group_irrelevant_credit = abs(float(cross_group_partition.player_values[1].item()))
    preflight_passed = (
        trained.fit_mse <= NEURAL_FIT_MAX_MSE
        and kernel.max_abs_error <= NEURAL_KERNEL_MAX_ERROR
        and kernel_vs_true_error <= NEURAL_KERNEL_TRUE_MAX_ERROR
        and singleton_partition.recovers_exact
        and singleton_partition.max_abs_error <= NEURAL_SINGLETON_PARTITION_MAX_ERROR
        and aligned_partition.recovers_exact
        and mismatched_partition.recovers_exact
        and cross_group_partition.recovers_exact
        and cross_group_irrelevant_credit <= CROSS_GROUP_IRRELEVANT_CREDIT_MAX
        and shuffled_kernel_true_error >= SHUFFLED_CONTROL_TRUE_ERROR_MIN
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
        "kernel_max_abs_error": kernel.max_abs_error,
        "kernel_approximates_exact": kernel.approximates_exact,
        "kernel_vs_true_max_abs_error": kernel_vs_true_error,
        "kernel_values": kernel.shapley_values.tolist(),
        "true_values": true_exact.tolist(),
        "partition_max_abs_error": singleton_partition.max_abs_error,
        "partition_recovers_exact": singleton_partition.recovers_exact,
        "singleton_partition_values": singleton_partition.player_values.tolist(),
        "aligned_partition_max_abs_error": aligned_partition.max_abs_error,
        "aligned_partition_recovers_exact": aligned_partition.recovers_exact,
        "aligned_partition_values": aligned_partition.player_values.tolist(),
        "mismatched_partition_max_abs_error": mismatched_partition.max_abs_error,
        "mismatched_partition_recovers_exact": mismatched_partition.recovers_exact,
        "mismatched_partition_values": mismatched_partition.player_values.tolist(),
        "cross_group_partition_recovers_exact": cross_group_partition.recovers_exact,
        "cross_group_partition_values": cross_group_partition.player_values.tolist(),
        "cross_group_irrelevant_credit": cross_group_irrelevant_credit,
        "shuffled_control_fit_mse": shuffled.fit_mse,
        "shuffled_control_kernel_vs_true_error": shuffled_kernel_true_error,
        "shuffled_control_rejected": shuffled_kernel_true_error >= SHUFFLED_CONTROL_TRUE_ERROR_MIN,
        "peak_vram_gb": peak_vram_gb,
        "within_vram_budget": peak_vram_gb <= max_vram_gb,
        "full_path": "Train a CUDA MLP on a complete binary feature game, run KernelSHAP and PartitionSHAP against real model ablation tables, and reject shuffled-label attribution.",
    }


def run_full_experiment(max_vram_gb: float = 24.0) -> dict:
    """Run the validated experiment path used by the verification report."""

    return run_gpu_test(max_vram_gb=max_vram_gb)


if MAIN:
    print(run_smoke_test())
