# %%
"""Reference solutions for [16.6] SHAP vs Activation Patching."""

import itertools
import math
import sys
from collections.abc import Mapping
from pathlib import Path

import torch as t

chapter = "chapter16_shapley_attribution_baselines"
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from arena_ext.shapley_attribution import (
    additive_game,
    conjunction_game,
    shapley_patching_comparison_report as arena_shapley_patching_comparison_report,
)
from arena_ext.shapley_neural_game import (
    NEURAL_GAME_NUM_PLAYERS,
    NEURAL_GAME_STEPS,
    binary_feature_table,
    coalition_table_from_model,
    train_neural_game,
    true_neural_game_scores,
)

MAIN = __name__ == "__main__"

ADDITIVE_LINEAR_SEED = 1660
ADDITIVE_LINEAR_STEPS = 1000
ADDITIVE_LINEAR_LR = 5e-2
ADDITIVE_AGREEMENT_MAX_ERROR = 1e-5
INTERACTION_DISAGREEMENT_MIN_ERROR = 1.0
INTERACTION_ABS_OVERCOUNT_MIN = 2.0
INTERACTION_FIT_MSE_MAX = 1e-8
Coalition = frozenset[int]


# %%
def all_coalitions(num_players: int) -> tuple[Coalition, ...]:
    """Return every coalition for `num_players` players."""

    if num_players <= 0:
        raise ValueError("num_players must be positive.")
    players = range(num_players)
    coalitions: list[Coalition] = []
    for size in range(num_players + 1):
        coalitions.extend(frozenset(group) for group in itertools.combinations(players, size))
    return tuple(coalitions)


def normalize_coalition_values(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    *,
    num_players: int,
) -> dict[Coalition, float]:
    """Normalize coalition keys and require a complete finite coalition table."""

    values = {frozenset(key): float(value) for key, value in coalition_values.items()}
    expected = set(all_coalitions(num_players))
    missing = expected - set(values)
    if missing:
        raise ValueError(f"coalition value table is missing {len(missing)} coalitions.")
    return values


def exact_shapley_values(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    *,
    num_players: int,
) -> t.Tensor:
    """Compute exact Shapley values by summing weighted marginal effects."""

    values = normalize_coalition_values(coalition_values, num_players=num_players)
    shapley = t.zeros(num_players, dtype=t.float64)
    denominator = math.factorial(num_players)
    for player in range(num_players):
        others = [candidate for candidate in range(num_players) if candidate != player]
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


def activation_patching_effects(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    *,
    num_players: int,
) -> t.Tensor:
    """Compute full-minus-ablated patching effects from a coalition table."""

    values = normalize_coalition_values(coalition_values, num_players=num_players)
    full = frozenset(range(num_players))
    effects = [values[full] - values[full - {player}] for player in range(num_players)]
    return t.tensor(effects, dtype=t.float64)


def shapley_patching_comparison_report(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    *,
    num_players: int,
    tolerance: float = 1e-9,
) -> dict:
    """Compare exact Shapley values to full-minus-ablated patching effects."""

    shapley = exact_shapley_values(coalition_values, num_players=num_players)
    patching = activation_patching_effects(coalition_values, num_players=num_players)
    max_abs_error = float((shapley - patching).abs().max().item())
    shapley_top = int(shapley.argmax().item())
    patching_top = int(patching.argmax().item())
    return {
        "shapley_values": shapley,
        "patching_effects": patching,
        "max_abs_error": max_abs_error,
        "shapley_top_feature": shapley_top,
        "patching_top_feature": patching_top,
        "top_feature_agrees": shapley_top == patching_top,
        "agrees_with_shapley": max_abs_error <= tolerance,
    }


def interaction_patching_failure_report(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    *,
    num_players: int,
    min_overcount: float = 0.5,
) -> dict:
    """Document patching overcount on an interaction-heavy coalition game."""

    shapley = exact_shapley_values(coalition_values, num_players=num_players)
    patching = activation_patching_effects(coalition_values, num_players=num_players)
    shapley_total = float(shapley.sum().item())
    patching_total = float(patching.sum().item())
    overcount = patching_total - shapley_total
    return {
        "shapley_values": shapley,
        "patching_effects": patching,
        "shapley_total": shapley_total,
        "patching_total": patching_total,
        "overcount": overcount,
        "documents_overcount": overcount >= min_overcount,
    }


def _jsonable_report(report: dict) -> dict:
    result = report.copy()
    for key, value in list(result.items()):
        if hasattr(value, "tolist"):
            result[key] = value.tolist()
    return result


def additive_agreement_smoke_test() -> dict:
    values = additive_game(t.tensor([1.0, 2.0, 0.5]))
    return _jsonable_report(shapley_patching_comparison_report(values, num_players=3))


def interaction_failure_smoke_test() -> dict:
    values = conjunction_game(2)
    return _jsonable_report(interaction_patching_failure_report(values, num_players=2))


def run_smoke_test(cpu: bool = True) -> dict:
    _ = cpu
    return {
        "additive_agreement": additive_agreement_smoke_test(),
        "interaction_failure": interaction_failure_smoke_test(),
    }


def run_gpu_test(max_vram_gb: float = 24.0) -> dict:
    if not t.cuda.is_available():
        raise RuntimeError("16.6 GPU preflight requires CUDA; no CPU fallback is accepted.")

    return run_neural_shap_vs_patching_preflight(max_vram_gb=max_vram_gb)


class _AdditiveLinearModel(t.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = t.nn.Linear(NEURAL_GAME_NUM_PLAYERS, 1)

    def forward(self, inputs: t.Tensor) -> t.Tensor:
        return self.linear(inputs)


def _additive_targets(inputs: t.Tensor) -> t.Tensor:
    return (
        0.25
        + 1.2 * inputs[:, 0]
        - 0.7 * inputs[:, 1]
        + 1.6 * inputs[:, 2]
        + 0.9 * inputs[:, 3]
    ).unsqueeze(-1)


def _train_additive_linear_model(inputs: t.Tensor) -> tuple[_AdditiveLinearModel, float, float]:
    targets = _additive_targets(inputs)
    t.manual_seed(ADDITIVE_LINEAR_SEED)
    t.cuda.manual_seed_all(ADDITIVE_LINEAR_SEED)
    model = _AdditiveLinearModel().to(inputs.device)
    optimizer = t.optim.AdamW(model.parameters(), lr=ADDITIVE_LINEAR_LR, weight_decay=0.0)
    for _ in range(ADDITIVE_LINEAR_STEPS):
        prediction = model(inputs)
        loss = ((prediction - targets) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with t.no_grad():
        residual = model(inputs) - targets
        mse = float((residual**2).mean().item())
        max_abs_error = float(residual.abs().max().item())
    return model, mse, max_abs_error


def run_neural_shap_vs_patching_preflight(max_vram_gb: float = 24.0) -> dict:
    device = t.device("cuda")
    t.cuda.reset_peak_memory_stats()

    inputs = binary_feature_table(device)
    additive_model, additive_fit_mse, additive_fit_max_abs_error = _train_additive_linear_model(inputs)
    additive_values = coalition_table_from_model(additive_model, device)

    interaction_targets = true_neural_game_scores(inputs)
    interaction_model = train_neural_game(inputs, interaction_targets)
    interaction_values = coalition_table_from_model(interaction_model.model, device)
    additive_report = arena_shapley_patching_comparison_report(
        additive_values,
        num_players=NEURAL_GAME_NUM_PLAYERS,
        tolerance=ADDITIVE_AGREEMENT_MAX_ERROR,
    )
    interaction_report = arena_shapley_patching_comparison_report(
        interaction_values,
        num_players=NEURAL_GAME_NUM_PLAYERS,
        tolerance=ADDITIVE_AGREEMENT_MAX_ERROR,
    )
    interaction_abs_overcount = float(
        interaction_report.patching_effects.abs().sum().item()
        - interaction_report.shapley_values.abs().sum().item()
    )

    t.cuda.synchronize()
    peak_vram_gb = t.cuda.max_memory_allocated() / 1024**3
    preflight_passed = (
        additive_report.agrees_with_shapley
        and additive_report.max_abs_error <= ADDITIVE_AGREEMENT_MAX_ERROR
        and interaction_model.fit_mse <= INTERACTION_FIT_MSE_MAX
        and not interaction_report.agrees_with_shapley
        and interaction_report.max_abs_error >= INTERACTION_DISAGREEMENT_MIN_ERROR
        and interaction_abs_overcount >= INTERACTION_ABS_OVERCOUNT_MIN
        and interaction_report.top_feature_agrees
        and peak_vram_gb <= max_vram_gb
    )
    return {
        "preflight_passed": preflight_passed,
        "cuda_available": True,
        "device": t.cuda.get_device_name(0),
        "additive_model_family": "cuda_trained_linear_additive_model",
        "additive_training_example_count": int(inputs.shape[0]),
        "additive_training_steps": ADDITIVE_LINEAR_STEPS,
        "additive_fit_mse": additive_fit_mse,
        "additive_fit_max_abs_error": additive_fit_max_abs_error,
        "additive_shapley_values": additive_report.shapley_values.tolist(),
        "additive_patching_effects": additive_report.patching_effects.tolist(),
        "additive_max_abs_error": additive_report.max_abs_error,
        "additive_agrees_with_shapley": additive_report.agrees_with_shapley,
        "interaction_model_family": "cuda_trained_neural_coalition_game_mlp",
        "interaction_training_example_count": int(inputs.shape[0]),
        "interaction_training_steps": NEURAL_GAME_STEPS,
        "interaction_fit_mse": interaction_model.fit_mse,
        "interaction_fit_max_abs_error": interaction_model.fit_max_abs_error,
        "interaction_shapley_values": interaction_report.shapley_values.tolist(),
        "interaction_patching_effects": interaction_report.patching_effects.tolist(),
        "interaction_max_abs_error": interaction_report.max_abs_error,
        "interaction_agrees_with_shapley": interaction_report.agrees_with_shapley,
        "interaction_top_feature_agrees": interaction_report.top_feature_agrees,
        "interaction_abs_overcount": interaction_abs_overcount,
        "peak_vram_gb": peak_vram_gb,
        "within_vram_budget": peak_vram_gb <= max_vram_gb,
        "full_path": "Train additive and interaction CUDA model organisms, then compare exact Shapley values with full-minus-ablated patching effects from real model outputs.",
    }


def run_full_experiment(max_vram_gb: float = 24.0) -> dict:
    """Run the validated experiment path used by the verification report."""

    return run_gpu_test(max_vram_gb=max_vram_gb)


if MAIN:
    print(run_smoke_test())
