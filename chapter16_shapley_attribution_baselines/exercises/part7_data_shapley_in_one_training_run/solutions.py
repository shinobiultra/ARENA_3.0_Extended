# %%
"""Reference solutions for [16.7] Data Shapley in One Training Run."""

import itertools
import math
import random
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch as t

chapter = "chapter16_shapley_attribution_baselines"
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

MAIN = __name__ == "__main__"
Coalition = frozenset[int]
DATA_SHAPLEY_LR = 0.5
DATA_SHAPLEY_MC_SAMPLES = 512
DATA_SHAPLEY_MC_MAX_ERROR = 0.08
DATA_SHAPLEY_MIN_CORRELATION = 0.99
DATA_SHAPLEY_RANDOM_CONTROL_SEED = 13
DATA_SHAPLEY_RANDOM_CONTROL_MAX_ABS_CORRELATION = 0.25
DATA_SHAPLEY_LABEL_SHUFFLE_MAX_CORRELATION = 0.0
DATA_SHAPLEY_LABEL_SHUFFLE_PERMUTATION = (0, 3, 2, 1)
DATA_SHAPLEY_RUNTIME_REPEATS = 128
DATA_SHAPLEY_RUNTIME_IN_RUN_VS_EXACT_MAX_RATIO = 2.0
TRAINING_RUN_LR = 0.2
TRAINING_RUN_STEPS = 40
TRAINING_RUN_PERMUTATION_BUDGETS = (4, 16, 64, 256)
TRAINING_RUN_BUDGET_SEEDS = 16
TRAINING_RUN_LABEL_PERMUTATION = (6, 1, 2, 3, 4, 5, 0, 7)


def toy_data_shapley_problem(
    train_labels: Sequence[float] = (1.0, 1.0, 1.0, -1.0),
) -> tuple[t.Tensor, t.Tensor, t.Tensor, t.Tensor]:
    """Return the four-example, one-dimensional ground-truth problem."""

    if len(train_labels) != 4:
        raise ValueError("The teaching problem requires exactly four training labels.")
    train_x = t.ones(4, 1, dtype=t.float64)
    train_y = t.tensor(train_labels, dtype=t.float64)
    val_x = t.ones(1, 1, dtype=t.float64)
    val_y = t.ones(1, dtype=t.float64)
    return train_x, train_y, val_x, val_y


def all_coalitions(num_players: int) -> tuple[Coalition, ...]:
    """Enumerate the complete power set in increasing coalition-size order."""

    if num_players <= 0:
        raise ValueError("num_players must be positive.")
    return tuple(
        frozenset(group)
        for size in range(num_players + 1)
        for group in itertools.combinations(range(num_players), size)
    )


def normalize_coalition_values(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    *,
    num_players: int,
) -> dict[Coalition, float]:
    """Normalize keys and reject incomplete or out-of-range coalition tables."""

    values = {frozenset(key): float(value) for key, value in coalition_values.items()}
    expected = set(all_coalitions(num_players))
    missing = expected - set(values)
    extra = set(values) - expected
    if missing or extra:
        raise ValueError(
            f"coalition table has {len(missing)} missing and {len(extra)} extra coalitions."
        )
    return values


def one_step_linear_utility(
    train_x: t.Tensor,
    train_y: t.Tensor,
    val_x: t.Tensor,
    val_y: t.Tensor,
    coalition: Coalition,
    *,
    learning_rate: float = DATA_SHAPLEY_LR,
) -> float:
    """Validation-loss improvement after one full-batch step on a coalition."""

    train_x = train_x.double()
    train_y = train_y.double()
    val_x = val_x.double()
    val_y = val_y.double()
    num_examples, num_features = train_x.shape
    if train_y.shape != (num_examples,):
        raise ValueError("train_y must have shape (num_examples,).")
    if val_x.shape[1] != num_features or val_y.shape != (val_x.shape[0],):
        raise ValueError("Validation tensors have incompatible shapes.")
    if not coalition:
        return 0.0
    if min(coalition) < 0 or max(coalition) >= num_examples:
        raise ValueError("coalition contains an invalid training-example index.")

    weight = t.zeros(num_features, dtype=t.float64, device=train_x.device)
    baseline_loss = ((val_x @ weight - val_y) ** 2).mean()
    indices = t.tensor(sorted(coalition), dtype=t.long, device=train_x.device)
    selected_x = train_x[indices]
    selected_y = train_y[indices]
    train_error = selected_x @ weight - selected_y
    gradient = (2 * train_error.unsqueeze(-1) * selected_x).mean(dim=0)
    updated_weight = weight - learning_rate * gradient
    updated_loss = ((val_x @ updated_weight - val_y) ** 2).mean()
    return float((baseline_loss - updated_loss).item())


def data_coalition_values(
    train_x: t.Tensor,
    train_y: t.Tensor,
    val_x: t.Tensor,
    val_y: t.Tensor,
    *,
    learning_rate: float = DATA_SHAPLEY_LR,
) -> dict[Coalition, float]:
    """Evaluate the one-step utility for every training-example coalition."""

    return {
        coalition: one_step_linear_utility(
            train_x,
            train_y,
            val_x,
            val_y,
            coalition,
            learning_rate=learning_rate,
        )
        for coalition in all_coalitions(int(train_x.shape[0]))
    }


def exact_shapley_values(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    *,
    num_players: int,
) -> t.Tensor:
    """Compute weighted marginal contributions over every predecessor coalition."""

    values = normalize_coalition_values(coalition_values, num_players=num_players)
    result = t.zeros(num_players, dtype=t.float64)
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
                result[player] += weight * (
                    values[coalition | {player}] - values[coalition]
                )
    return result


def exact_data_shapley_values(
    train_x: t.Tensor,
    train_y: t.Tensor,
    val_x: t.Tensor,
    val_y: t.Tensor,
    *,
    learning_rate: float = DATA_SHAPLEY_LR,
) -> t.Tensor:
    """Compute exact values for the finite one-step training game."""

    values = data_coalition_values(
        train_x,
        train_y,
        val_x,
        val_y,
        learning_rate=learning_rate,
    )
    return exact_shapley_values(values, num_players=int(train_x.shape[0]))


def sampled_permutation_shapley_values(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    *,
    num_players: int,
    num_samples: int,
    seed: int = 0,
) -> t.Tensor:
    """Estimate Shapley values from sampled random example orderings."""

    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")
    values = normalize_coalition_values(coalition_values, num_players=num_players)
    rng = random.Random(seed)
    totals = t.zeros(num_players, dtype=t.float64)
    players = tuple(range(num_players))
    for _ in range(num_samples):
        coalition: Coalition = frozenset()
        for player in rng.sample(players, k=num_players):
            with_player = coalition | {player}
            totals[player] += values[with_player] - values[coalition]
            coalition = with_player
    return totals / num_samples


def in_run_first_order_data_scores(
    train_x: t.Tensor,
    train_y: t.Tensor,
    val_x: t.Tensor,
    val_y: t.Tensor,
) -> t.Tensor:
    """Dot per-example training gradients with the validation gradient at initialization."""

    train_x = train_x.double()
    train_y = train_y.double()
    val_x = val_x.double()
    val_y = val_y.double()
    weight = t.zeros(train_x.shape[1], dtype=t.float64, device=train_x.device)
    val_error = val_x @ weight - val_y
    val_gradient = (2 * val_error.unsqueeze(-1) * val_x).mean(dim=0)
    train_error = train_x @ weight - train_y
    train_gradients = 2 * train_error.unsqueeze(-1) * train_x
    return train_gradients @ val_gradient


def pearson_correlation(first: t.Tensor, second: t.Tensor) -> float:
    """Return Pearson correlation, or NaN when either vector is constant."""

    first = first.double().flatten()
    second = second.double().flatten()
    if first.shape != second.shape or first.numel() < 2:
        raise ValueError("Pearson correlation needs equal vectors with at least two entries.")
    first = first - first.mean()
    second = second - second.mean()
    denominator = first.norm() * second.norm()
    if float(denominator.item()) == 0.0:
        return float("nan")
    return float((first @ second / denominator).item())


def leave_one_out_values(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    *,
    num_players: int,
) -> t.Tensor:
    """Measure each example only in the context of the full training set."""

    values = normalize_coalition_values(coalition_values, num_players=num_players)
    full = frozenset(range(num_players))
    return t.tensor(
        [values[full] - values[full - {player}] for player in range(num_players)],
        dtype=t.float64,
    )


def data_shapley_diagnostics(
    train_x: t.Tensor,
    train_y: t.Tensor,
    val_x: t.Tensor,
    val_y: t.Tensor,
    *,
    learning_rate: float = DATA_SHAPLEY_LR,
    num_samples: int = DATA_SHAPLEY_MC_SAMPLES,
    seed: int = 0,
) -> dict:
    """Compute the exact result, approximations, baseline, and deletion check."""

    num_players = int(train_x.shape[0])
    values = data_coalition_values(
        train_x,
        train_y,
        val_x,
        val_y,
        learning_rate=learning_rate,
    )
    exact = exact_shapley_values(values, num_players=num_players)
    sampled = sampled_permutation_shapley_values(
        values,
        num_players=num_players,
        num_samples=num_samples,
        seed=seed,
    )
    proxy = in_run_first_order_data_scores(train_x, train_y, val_x, val_y)
    leave_one_out = leave_one_out_values(values, num_players=num_players)
    full = frozenset(range(num_players))
    harmful_index = int(exact.argmin().item())
    return {
        "coalition_values": values,
        "exact_values": exact,
        "sampled_values": sampled,
        "in_run_scores": proxy,
        "leave_one_out_values": leave_one_out,
        "full_utility": values[full],
        "efficiency_error": abs(float(exact.sum().item()) - values[full]),
        "sampled_max_abs_error": float((sampled - exact).abs().max().item()),
        "proxy_correlation": pearson_correlation(exact, proxy),
        "harmful_index": harmful_index,
        "harmful_removal_delta": values[full - {harmful_index}] - values[full],
    }


def clean_label_control(*, learning_rate: float = DATA_SHAPLEY_LR) -> dict:
    """Remove the corruption while keeping inputs, validation target, and training size fixed."""

    problem = toy_data_shapley_problem((1.0, 1.0, 1.0, 1.0))
    exact = exact_data_shapley_values(*problem, learning_rate=learning_rate)
    return {
        "labels": problem[1],
        "exact_values": exact,
        "has_negative_value": bool((exact < 0).any().item()),
    }


def relocated_corruption_control(*, learning_rate: float = DATA_SHAPLEY_LR) -> dict:
    """Move the flipped label from index 3 to index 1 without changing label counts."""

    problem = toy_data_shapley_problem((1.0, -1.0, 1.0, 1.0))
    exact = exact_data_shapley_values(*problem, learning_rate=learning_rate)
    proxy = in_run_first_order_data_scores(*problem)
    return {
        "labels": problem[1],
        "exact_values": exact,
        "in_run_scores": proxy,
        "harmful_index": int(exact.argmin().item()),
        "proxy_harmful_index": int(proxy.argmin().item()),
    }


def proxy_alignment_sweep(
    learning_rates: Sequence[float],
) -> dict[str, t.Tensor]:
    """Compare the initialization proxy with exact values as the update grows."""

    if not learning_rates:
        raise ValueError("learning_rates must not be empty.")
    problem = toy_data_shapley_problem()
    proxy = in_run_first_order_data_scores(*problem)
    exact_rows = []
    correlations = []
    harmful_indices = []
    for learning_rate in learning_rates:
        exact = exact_data_shapley_values(*problem, learning_rate=float(learning_rate))
        exact_rows.append(exact)
        correlations.append(pearson_correlation(exact, proxy))
        harmful_indices.append(int(exact.argmin().item()))
    return {
        "learning_rates": t.tensor(learning_rates, dtype=t.float64),
        "exact_values": t.stack(exact_rows),
        "correlations": t.tensor(correlations, dtype=t.float64),
        "harmful_indices": t.tensor(harmful_indices, dtype=t.long),
    }


def first_proxy_rank_failure(learning_rates: Sequence[float]) -> float | None:
    """Return the first rate where the proxy no longer ranks example 3 as most harmful."""

    sweep = proxy_alignment_sweep(learning_rates)
    failures = (sweep["harmful_indices"] != 3).nonzero().flatten()
    if failures.numel() == 0:
        return None
    return float(sweep["learning_rates"][int(failures[0].item())].item())


# %%
# The learner-facing lab below uses repeated logistic-regression training runs.
# The one-step functions above remain as a compact algebraic contract and for the
# historical CUDA preflight, but they are not the lesson's scientific result.
def training_run_data_shapley_problem(
    train_labels: Sequence[float] | None = None,
) -> tuple[t.Tensor, t.Tensor, t.Tensor, t.Tensor]:
    """Return an eight-example binary-classification game with known anomalies."""

    train_x = t.tensor(
        [
            [2.0, 1.0],
            [2.0, 1.0],
            [1.5, 1.5],
            [2.0, -0.5],
            [-2.0, -1.0],
            [-1.5, -1.5],
            [2.0, 1.0],
            [0.2, 2.5],
        ],
        dtype=t.float64,
    )
    default_labels = (1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    labels = default_labels if train_labels is None else tuple(train_labels)
    if len(labels) != len(train_x):
        raise ValueError("The training-run organism requires exactly eight labels.")
    train_y = t.tensor(labels, dtype=t.float64)
    val_x = t.tensor(
        [
            [2.0, 1.0],
            [1.5, 1.2],
            [2.0, -0.5],
            [1.0, 2.0],
            [-2.0, -1.0],
            [-1.5, -1.2],
            [-2.0, 0.5],
            [-1.0, -2.0],
        ],
        dtype=t.float64,
    )
    val_y = t.tensor((1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0), dtype=t.float64)
    return train_x, train_y, val_x, val_y


def add_bias_column(inputs: t.Tensor) -> t.Tensor:
    """Append a constant feature so the linear classifier learns an intercept."""

    if inputs.ndim != 2:
        raise ValueError("inputs must have shape (examples, features).")
    ones = t.ones((inputs.shape[0], 1), dtype=inputs.dtype, device=inputs.device)
    return t.cat((inputs, ones), dim=1)


def binary_logistic_loss(
    parameters: t.Tensor,
    inputs: t.Tensor,
    labels: t.Tensor,
) -> t.Tensor:
    """Mean binary cross-entropy for a bias-augmented linear classifier."""

    augmented = add_bias_column(inputs.double())
    labels = labels.double()
    if labels.shape != (inputs.shape[0],):
        raise ValueError("labels must have one value per input row.")
    return t.nn.functional.binary_cross_entropy_with_logits(
        augmented @ parameters.double(), labels
    )


def train_logistic_subset(
    train_x: t.Tensor,
    train_y: t.Tensor,
    coalition: Coalition,
    *,
    steps: int = TRAINING_RUN_STEPS,
    learning_rate: float = TRAINING_RUN_LR,
) -> t.Tensor:
    """Train logistic regression from a fixed zero initialization on one subset."""

    if steps < 0:
        raise ValueError("steps must be non-negative.")
    num_examples = int(train_x.shape[0])
    if train_y.shape != (num_examples,):
        raise ValueError("train_y must have one label per example.")
    if coalition and (min(coalition) < 0 or max(coalition) >= num_examples):
        raise ValueError("coalition contains an invalid training-example index.")
    augmented = add_bias_column(train_x.double())
    parameters = t.zeros(
        augmented.shape[1], dtype=t.float64, device=train_x.device
    )
    if not coalition:
        return parameters
    indices = t.tensor(sorted(coalition), dtype=t.long, device=train_x.device)
    selected_x = augmented[indices]
    selected_y = train_y.double()[indices]
    for _ in range(steps):
        errors = t.sigmoid(selected_x @ parameters) - selected_y
        gradient = selected_x.T @ errors / len(indices)
        parameters = parameters - learning_rate * gradient
    return parameters


def training_run_utility(
    train_x: t.Tensor,
    train_y: t.Tensor,
    val_x: t.Tensor,
    val_y: t.Tensor,
    coalition: Coalition,
    *,
    steps: int = TRAINING_RUN_STEPS,
    learning_rate: float = TRAINING_RUN_LR,
) -> float:
    """Held-out loss improvement after genuinely training on one coalition."""

    zero = t.zeros(train_x.shape[1] + 1, dtype=t.float64, device=train_x.device)
    baseline_loss = binary_logistic_loss(zero, val_x, val_y)
    parameters = train_logistic_subset(
        train_x,
        train_y,
        coalition,
        steps=steps,
        learning_rate=learning_rate,
    )
    trained_loss = binary_logistic_loss(parameters, val_x, val_y)
    return float((baseline_loss - trained_loss).item())


def training_run_coalition_values(
    train_x: t.Tensor,
    train_y: t.Tensor,
    val_x: t.Tensor,
    val_y: t.Tensor,
    *,
    steps: int = TRAINING_RUN_STEPS,
    learning_rate: float = TRAINING_RUN_LR,
) -> dict[Coalition, float]:
    """Retrain all subsets to create exact held-out utility ground truth."""

    return {
        coalition: training_run_utility(
            train_x,
            train_y,
            val_x,
            val_y,
            coalition,
            steps=steps,
            learning_rate=learning_rate,
        )
        for coalition in all_coalitions(int(train_x.shape[0]))
    }


def fixed_order_marginal_values(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    order: Sequence[int],
) -> t.Tensor:
    """Assign marginal utility along one fixed ordering as a bias control."""

    num_players = len(order)
    if sorted(order) != list(range(num_players)):
        raise ValueError("order must be a permutation of every player index.")
    values = normalize_coalition_values(coalition_values, num_players=num_players)
    result = t.zeros(num_players, dtype=t.float64)
    coalition: Coalition = frozenset()
    for player in order:
        with_player = coalition | {player}
        result[player] = values[with_player] - values[coalition]
        coalition = with_player
    return result


def checkpoint_gradient_data_scores(
    train_x: t.Tensor,
    train_y: t.Tensor,
    val_x: t.Tensor,
    val_y: t.Tensor,
    *,
    steps: int = TRAINING_RUN_STEPS,
    learning_rate: float = TRAINING_RUN_LR,
) -> t.Tensor:
    """Accumulate per-example train/validation gradient alignment during one run."""

    train_augmented = add_bias_column(train_x.double())
    val_augmented = add_bias_column(val_x.double())
    train_y = train_y.double()
    val_y = val_y.double()
    num_examples = int(train_x.shape[0])
    parameters = t.zeros(
        train_augmented.shape[1], dtype=t.float64, device=train_x.device
    )
    scores = t.zeros(num_examples, dtype=t.float64, device=train_x.device)
    for _ in range(steps):
        val_errors = t.sigmoid(val_augmented @ parameters) - val_y
        val_gradient = val_augmented.T @ val_errors / len(val_y)
        train_errors = t.sigmoid(train_augmented @ parameters) - train_y
        per_example_gradients = train_errors.unsqueeze(1) * train_augmented
        scores += learning_rate * (per_example_gradients @ val_gradient) / num_examples
        parameters -= learning_rate * per_example_gradients.mean(dim=0)
    return scores


def influence_function_scores(
    train_x: t.Tensor,
    train_y: t.Tensor,
    val_x: t.Tensor,
    val_y: t.Tensor,
    *,
    steps: int = TRAINING_RUN_STEPS,
    learning_rate: float = TRAINING_RUN_LR,
    ridge: float = 1e-3,
) -> t.Tensor:
    """Approximate full-data utility credit with a damped inverse Hessian."""

    if ridge < 0:
        raise ValueError("ridge must be non-negative.")
    full = frozenset(range(int(train_x.shape[0])))
    parameters = train_logistic_subset(
        train_x,
        train_y,
        full,
        steps=steps,
        learning_rate=learning_rate,
    )
    train_augmented = add_bias_column(train_x.double())
    val_augmented = add_bias_column(val_x.double())
    train_probabilities = t.sigmoid(train_augmented @ parameters)
    train_weights = train_probabilities * (1.0 - train_probabilities)
    hessian = (
        train_augmented.T @ (train_weights.unsqueeze(1) * train_augmented)
        / len(train_y)
    )
    hessian += ridge * t.eye(
        hessian.shape[0], dtype=hessian.dtype, device=hessian.device
    )
    val_errors = t.sigmoid(val_augmented @ parameters) - val_y.double()
    val_gradient = val_augmented.T @ val_errors / len(val_y)
    train_errors = train_probabilities - train_y.double()
    per_example_gradients = train_errors.unsqueeze(1) * train_augmented
    inverse_hessian_val_gradient = t.linalg.solve(hessian, val_gradient)
    return per_example_gradients @ inverse_hessian_val_gradient / len(train_y)


def permutation_budget_sweep(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    exact_values: t.Tensor,
    *,
    budgets: Sequence[int] = TRAINING_RUN_PERMUTATION_BUDGETS,
    num_seeds: int = TRAINING_RUN_BUDGET_SEEDS,
) -> dict[str, t.Tensor]:
    """Measure random-order estimator error across budgets and independent seeds."""

    if not budgets or any(budget <= 0 for budget in budgets):
        raise ValueError("budgets must contain positive sample counts.")
    if num_seeds <= 0:
        raise ValueError("num_seeds must be positive.")
    exact_values = exact_values.double()
    num_players = int(exact_values.numel())
    errors = t.empty((len(budgets), num_seeds), dtype=t.float64)
    harmful_hits = t.empty((len(budgets), num_seeds), dtype=t.float64)
    exact_harmful = int(exact_values.argmin().item())
    for budget_index, budget in enumerate(budgets):
        for seed in range(num_seeds):
            sampled = sampled_permutation_shapley_values(
                coalition_values,
                num_players=num_players,
                num_samples=int(budget),
                seed=seed,
            )
            errors[budget_index, seed] = (sampled - exact_values).abs().max()
            harmful_hits[budget_index, seed] = float(
                int(sampled.argmin().item()) == exact_harmful
            )
    return {
        "budgets": t.tensor(tuple(budgets), dtype=t.long),
        "mean_max_error": errors.mean(dim=1),
        "min_max_error": errors.min(dim=1).values,
        "max_max_error": errors.max(dim=1).values,
        "harmful_hit_rate": harmful_hits.mean(dim=1),
    }


def shuffled_label_exact_values(
    train_x: t.Tensor,
    train_y: t.Tensor,
    val_x: t.Tensor,
    val_y: t.Tensor,
    permutation: Sequence[int],
) -> tuple[t.Tensor, t.Tensor]:
    """Recompute exact values after permuting labels while fixing all inputs."""

    num_examples = int(train_x.shape[0])
    if sorted(permutation) != list(range(num_examples)):
        raise ValueError("permutation must contain every training index exactly once.")
    indices = t.tensor(permutation, dtype=t.long, device=train_y.device)
    shuffled_labels = train_y[indices]
    coalition_values = training_run_coalition_values(
        train_x, shuffled_labels, val_x, val_y
    )
    exact = exact_shapley_values(coalition_values, num_players=num_examples)
    return shuffled_labels, exact


def training_run_shuffled_label_control() -> dict:
    """Move the bad label to its duplicate twin while preserving all features."""

    problem = training_run_data_shapley_problem()
    permutation = t.tensor(TRAINING_RUN_LABEL_PERMUTATION, dtype=t.long)
    original_values = training_run_coalition_values(*problem)
    original_exact = exact_shapley_values(original_values, num_players=8)
    shuffled_labels, shuffled_exact = shuffled_label_exact_values(
        *problem, TRAINING_RUN_LABEL_PERMUTATION
    )
    return {
        "permutation": permutation,
        "labels": shuffled_labels,
        "original_exact": original_exact,
        "shuffled_exact": shuffled_exact,
        "original_harmful_index": int(original_exact.argmin().item()),
        "shuffled_harmful_index": int(shuffled_exact.argmin().item()),
        "correlation": pearson_correlation(original_exact, shuffled_exact),
    }


def training_run_data_shapley_diagnostics() -> dict:
    """Package the learner lab's exact result, approximations, and controls."""

    problem = training_run_data_shapley_problem()
    coalition_values = training_run_coalition_values(*problem)
    exact = exact_shapley_values(coalition_values, num_players=8)
    one_run = checkpoint_gradient_data_scores(*problem)
    leave_one_out = leave_one_out_values(coalition_values, num_players=8)
    influence = influence_function_scores(*problem)
    sampled = sampled_permutation_shapley_values(
        coalition_values, num_players=8, num_samples=256, seed=0
    )
    budget = permutation_budget_sweep(coalition_values, exact)
    shuffled = training_run_shuffled_label_control()
    full = frozenset(range(8))
    harmful_index = int(exact.argmin().item())
    matched_index = 0
    return {
        "exact_values": exact,
        "one_run_values": one_run,
        "sampled_values": sampled,
        "leave_one_out_values": leave_one_out,
        "influence_values": influence,
        "full_utility": coalition_values[full],
        "efficiency_error": abs(
            float(exact.sum().item()) - coalition_values[full]
        ),
        "one_run_correlation": pearson_correlation(exact, one_run),
        "leave_one_out_correlation": pearson_correlation(exact, leave_one_out),
        "influence_correlation": pearson_correlation(exact, influence),
        "sampled_max_abs_error": float((sampled - exact).abs().max().item()),
        "harmful_index": harmful_index,
        "duplicate_gap": abs(float((exact[0] - exact[1]).item())),
        "harmful_removal_delta": coalition_values[full - {harmful_index}]
        - coalition_values[full],
        "matched_removal_delta": coalition_values[full - {matched_index}]
        - coalition_values[full],
        "budget": budget,
        "shuffled_label_control": shuffled,
    }


# %%
def _tensor_report(report) -> dict:
    result = report.__dict__.copy()
    for key, value in list(result.items()):
        if hasattr(value, "tolist"):
            result[key] = value.tolist()
    return result


def _to_serializable_metrics(metrics: dict) -> dict:
    def convert(value):
        if isinstance(value, dict):
            return {key: convert(item) for key, item in value.items()}
        if hasattr(value, "tolist"):
            return value.tolist()
        return value

    return convert(metrics)


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / max(denominator, 1e-12)


def _measure_wall_seconds(fn, *, repeats: int) -> float:
    if repeats <= 0:
        raise ValueError("repeats must be positive.")
    fn()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - start) / repeats


def _random_data_shapley_problem() -> tuple[t.Tensor, t.Tensor, t.Tensor, t.Tensor]:
    generator = t.Generator().manual_seed(DATA_SHAPLEY_RANDOM_CONTROL_SEED)
    train_x = t.randn(4, 3, generator=generator, dtype=t.float64)
    train_y = t.randn(4, generator=generator, dtype=t.float64)
    val_x = t.randn(2, 3, generator=generator, dtype=t.float64)
    val_y = t.randn(2, generator=generator, dtype=t.float64)
    return train_x, train_y, val_x, val_y


def _label_shuffled_data_shapley_problem() -> tuple[t.Tensor, t.Tensor, t.Tensor, t.Tensor]:
    train_x, train_y, val_x, val_y = toy_data_shapley_problem()
    permutation = t.tensor(DATA_SHAPLEY_LABEL_SHUFFLE_PERMUTATION, dtype=t.long)
    return train_x, train_y[permutation], val_x, val_y


def _signal_failure_metrics(
    train_x: t.Tensor,
    train_y: t.Tensor,
    val_x: t.Tensor,
    val_y: t.Tensor,
    *,
    signal_exact: t.Tensor,
) -> dict:
    exact = exact_data_shapley_values(train_x, train_y, val_x, val_y)
    scores = in_run_first_order_data_scores(train_x, train_y, val_x, val_y)
    exact_signal_correlation = pearson_correlation(signal_exact, exact)
    in_run_signal_correlation = pearson_correlation(signal_exact, scores)
    return {
        "exact_values": exact,
        "in_run_scores": scores,
        "harmful_index": int(exact.argmin().item()),
        "helpful_index": int(exact.argmax().item()),
        "in_run_harmful_index": int(scores.argmin().item()),
        "in_run_helpful_index": int(scores.argmax().item()),
        "signal_correlation": exact_signal_correlation,
        "in_run_signal_correlation": in_run_signal_correlation,
        "max_abs_signal_correlation": max(
            abs(exact_signal_correlation),
            abs(in_run_signal_correlation),
        ),
    }


def random_data_attribution_failure_smoke_test() -> dict:
    signal_exact = exact_data_shapley_values(*toy_data_shapley_problem())
    train_x, train_y, val_x, val_y = _random_data_shapley_problem()
    metrics = _signal_failure_metrics(
        train_x,
        train_y,
        val_x,
        val_y,
        signal_exact=signal_exact,
    )
    original_harmful_index = int(signal_exact.argmin().item())
    original_helpful_index = int(signal_exact.argmax().item())
    metrics["random_data_attribution_fails"] = (
        metrics["harmful_index"] != original_harmful_index
        and metrics["helpful_index"] != original_helpful_index
        and metrics["max_abs_signal_correlation"]
        <= DATA_SHAPLEY_RANDOM_CONTROL_MAX_ABS_CORRELATION
    )
    metrics["original_harmful_index"] = original_harmful_index
    metrics["original_helpful_index"] = original_helpful_index
    metrics["seed"] = DATA_SHAPLEY_RANDOM_CONTROL_SEED
    return _to_serializable_metrics(metrics)


def label_shuffled_attribution_failure_smoke_test() -> dict:
    signal_exact = exact_data_shapley_values(*toy_data_shapley_problem())
    train_x, train_y, val_x, val_y = _label_shuffled_data_shapley_problem()
    metrics = _signal_failure_metrics(
        train_x,
        train_y,
        val_x,
        val_y,
        signal_exact=signal_exact,
    )
    original_harmful_index = int(signal_exact.argmin().item())
    metrics["label_shuffled_attribution_fails"] = (
        metrics["harmful_index"] != original_harmful_index
        and metrics["in_run_harmful_index"] != original_harmful_index
        and metrics["signal_correlation"] <= DATA_SHAPLEY_LABEL_SHUFFLE_MAX_CORRELATION
        and metrics["in_run_signal_correlation"] <= DATA_SHAPLEY_LABEL_SHUFFLE_MAX_CORRELATION
    )
    metrics["original_harmful_index"] = original_harmful_index
    metrics["label_permutation"] = list(DATA_SHAPLEY_LABEL_SHUFFLE_PERMUTATION)
    metrics["shuffled_train_y"] = train_y.tolist()
    return _to_serializable_metrics(metrics)


def runtime_overhead_smoke_test(repeats: int = DATA_SHAPLEY_RUNTIME_REPEATS) -> dict:
    train_x, train_y, val_x, val_y = toy_data_shapley_problem()
    full = frozenset(range(int(train_x.shape[0])))
    full_update_seconds = _measure_wall_seconds(
        lambda: one_step_linear_utility(train_x, train_y, val_x, val_y, full),
        repeats=repeats,
    )
    exact_enumeration_seconds = _measure_wall_seconds(
        lambda: data_coalition_values(train_x, train_y, val_x, val_y),
        repeats=repeats,
    )
    in_run_scores_seconds = _measure_wall_seconds(
        lambda: in_run_first_order_data_scores(train_x, train_y, val_x, val_y),
        repeats=repeats,
    )
    return {
        "runtime_measurement_repeats": repeats,
        "runtime_full_update_seconds": full_update_seconds,
        "runtime_exact_enumeration_seconds": exact_enumeration_seconds,
        "runtime_in_run_scores_seconds": in_run_scores_seconds,
        "runtime_exact_vs_full_update_overhead_ratio": _ratio(
            exact_enumeration_seconds,
            full_update_seconds,
        ),
        "runtime_in_run_vs_full_update_overhead_ratio": _ratio(
            in_run_scores_seconds,
            full_update_seconds,
        ),
        "runtime_in_run_vs_exact_ratio": _ratio(
            in_run_scores_seconds,
            exact_enumeration_seconds,
        ),
        "runtime_overhead_reported": (
            full_update_seconds > 0.0
            and exact_enumeration_seconds > 0.0
            and in_run_scores_seconds > 0.0
        ),
    }


def exact_data_shapley_smoke_test() -> dict:
    metrics = data_shapley_diagnostics(*toy_data_shapley_problem())
    exact = metrics["exact_values"]
    return {
        "exact_values": exact.tolist(),
        "full_utility": metrics["full_utility"],
        "harmful_index": metrics["harmful_index"],
        "harmful_value": float(exact[metrics["harmful_index"]].item()),
        "harmful_removal_delta": metrics["harmful_removal_delta"],
        "deletion_test_passes": metrics["harmful_removal_delta"] > 0.0,
        "efficiency_error": metrics["efficiency_error"],
    }


def monte_carlo_data_shapley_smoke_test() -> dict:
    metrics = data_shapley_diagnostics(*toy_data_shapley_problem())
    return {
        "exact_values": metrics["exact_values"].tolist(),
        "sampled_values": metrics["sampled_values"].tolist(),
        "max_abs_error": metrics["sampled_max_abs_error"],
        "harmful_example_matches": int(metrics["sampled_values"].argmin().item())
        == metrics["harmful_index"],
        "approximates_exact": metrics["sampled_max_abs_error"]
        <= DATA_SHAPLEY_MC_MAX_ERROR,
    }


def in_run_data_shapley_smoke_test() -> dict:
    metrics = data_shapley_diagnostics(*toy_data_shapley_problem())
    return {
        "exact_values": metrics["exact_values"].tolist(),
        "in_run_scores": metrics["in_run_scores"].tolist(),
        "pearson_correlation": metrics["proxy_correlation"],
        "harmful_index": metrics["harmful_index"],
        "in_run_harmful_index": int(metrics["in_run_scores"].argmin().item()),
        "identifies_harmful": int(metrics["in_run_scores"].argmin().item())
        == metrics["harmful_index"],
        "correlates_with_exact": metrics["proxy_correlation"]
        >= DATA_SHAPLEY_MIN_CORRELATION,
    }


def run_smoke_test(cpu: bool = True) -> dict:
    _ = cpu
    clean = clean_label_control()
    relocated = relocated_corruption_control()
    stress_rates = [round(0.05 * step, 10) for step in range(1, 51)]
    return {
        "exact": exact_data_shapley_smoke_test(),
        "monte_carlo": monte_carlo_data_shapley_smoke_test(),
        "in_run": in_run_data_shapley_smoke_test(),
        "clean_label_control": _to_serializable_metrics(clean),
        "relocated_corruption_control": _to_serializable_metrics(relocated),
        "first_proxy_rank_failure_lr": first_proxy_rank_failure(stress_rates),
        "training_run_lab": _to_serializable_metrics(
            training_run_data_shapley_diagnostics()
        ),
    }


def run_gpu_test(max_vram_gb: float = 24.0) -> dict:
    if not t.cuda.is_available():
        raise RuntimeError("16.7 GPU preflight requires CUDA; no CPU fallback is accepted.")

    return run_cuda_data_shapley_preflight(max_vram_gb=max_vram_gb)


def _cuda_one_step_utility(
    train_x: t.Tensor,
    train_y: t.Tensor,
    val_x: t.Tensor,
    val_y: t.Tensor,
    coalition: frozenset[int],
) -> float:
    weight = t.zeros(train_x.shape[1], dtype=t.float64, device=train_x.device)
    baseline_loss = ((val_x @ weight - val_y) ** 2).mean()
    if not coalition:
        return 0.0
    indices = t.tensor(sorted(coalition), dtype=t.long, device=train_x.device)
    selected_x = train_x[indices]
    selected_y = train_y[indices]
    train_error = selected_x @ weight - selected_y
    gradient = (2 * train_error.unsqueeze(-1) * selected_x).mean(dim=0)
    updated_weight = weight - DATA_SHAPLEY_LR * gradient
    updated_loss = ((val_x @ updated_weight - val_y) ** 2).mean()
    return float((baseline_loss - updated_loss).item())


def _cuda_data_coalition_values(
    train_x: t.Tensor,
    train_y: t.Tensor,
    val_x: t.Tensor,
    val_y: t.Tensor,
) -> dict[frozenset[int], float]:
    num_examples = int(train_x.shape[0])
    return {
        frozenset(group): _cuda_one_step_utility(train_x, train_y, val_x, val_y, frozenset(group))
        for size in range(num_examples + 1)
        for group in itertools.combinations(range(num_examples), size)
    }


def _cuda_in_run_gradient_scores(
    train_x: t.Tensor,
    train_y: t.Tensor,
    val_x: t.Tensor,
    val_y: t.Tensor,
) -> t.Tensor:
    weight = t.nn.Parameter(t.zeros(train_x.shape[1], dtype=t.float64, device=train_x.device))
    val_loss = ((val_x @ weight - val_y) ** 2).mean()
    val_gradient = t.autograd.grad(val_loss, weight, retain_graph=True)[0].detach()
    scores = []
    for index in range(int(train_x.shape[0])):
        train_loss = ((train_x[index : index + 1] @ weight - train_y[index : index + 1]) ** 2).mean()
        train_gradient = t.autograd.grad(train_loss, weight, retain_graph=True)[0].detach()
        scores.append(float((train_gradient * val_gradient).sum().item()))
    return t.tensor(scores, dtype=t.float64)


def _actual_full_batch_one_step_utility(
    train_x: t.Tensor,
    train_y: t.Tensor,
    val_x: t.Tensor,
    val_y: t.Tensor,
) -> float:
    model = t.nn.Linear(1, 1, bias=False, dtype=t.float64).to(train_x.device)
    model.weight.data.zero_()
    optimizer = t.optim.SGD(model.parameters(), lr=DATA_SHAPLEY_LR)
    baseline_loss = ((model(val_x).flatten() - val_y) ** 2).mean()
    train_loss = ((model(train_x).flatten() - train_y) ** 2).mean()
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()
    updated_loss = ((model(val_x).flatten() - val_y) ** 2).mean()
    return float((baseline_loss - updated_loss).item())


def _move_problem_to_device(
    problem: tuple[t.Tensor, t.Tensor, t.Tensor, t.Tensor],
    device: t.device,
) -> tuple[t.Tensor, t.Tensor, t.Tensor, t.Tensor]:
    return tuple(item.to(device=device, dtype=t.float64) for item in problem)


def _cuda_signal_failure_metrics(
    train_x: t.Tensor,
    train_y: t.Tensor,
    val_x: t.Tensor,
    val_y: t.Tensor,
    *,
    signal_exact: t.Tensor,
) -> dict:
    values = _cuda_data_coalition_values(train_x, train_y, val_x, val_y)
    exact = exact_shapley_values(values, num_players=int(train_x.shape[0]))
    scores = _cuda_in_run_gradient_scores(train_x, train_y, val_x, val_y)
    exact_signal_correlation = pearson_correlation(signal_exact, exact)
    in_run_signal_correlation = pearson_correlation(signal_exact, scores)
    return {
        "exact_values": exact.tolist(),
        "in_run_scores": scores.tolist(),
        "harmful_index": int(exact.argmin().item()),
        "helpful_index": int(exact.argmax().item()),
        "in_run_harmful_index": int(scores.argmin().item()),
        "in_run_helpful_index": int(scores.argmax().item()),
        "signal_correlation": exact_signal_correlation,
        "in_run_signal_correlation": in_run_signal_correlation,
        "max_abs_signal_correlation": max(
            abs(exact_signal_correlation),
            abs(in_run_signal_correlation),
        ),
    }


def _measure_cuda_seconds(fn, *, repeats: int) -> float:
    if repeats <= 0:
        raise ValueError("repeats must be positive.")
    fn()
    t.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    t.cuda.synchronize()
    return (time.perf_counter() - start) / repeats


def _cuda_runtime_overhead_metrics(
    train_x: t.Tensor,
    train_y: t.Tensor,
    val_x: t.Tensor,
    val_y: t.Tensor,
    *,
    repeats: int = DATA_SHAPLEY_RUNTIME_REPEATS,
) -> dict:
    full_update_seconds = _measure_cuda_seconds(
        lambda: _actual_full_batch_one_step_utility(train_x, train_y, val_x, val_y),
        repeats=repeats,
    )
    exact_enumeration_seconds = _measure_cuda_seconds(
        lambda: _cuda_data_coalition_values(train_x, train_y, val_x, val_y),
        repeats=repeats,
    )
    in_run_scores_seconds = _measure_cuda_seconds(
        lambda: _cuda_in_run_gradient_scores(train_x, train_y, val_x, val_y),
        repeats=repeats,
    )
    return {
        "runtime_measurement_repeats": repeats,
        "runtime_full_update_seconds": full_update_seconds,
        "runtime_exact_enumeration_seconds": exact_enumeration_seconds,
        "runtime_in_run_scores_seconds": in_run_scores_seconds,
        "runtime_exact_vs_full_update_overhead_ratio": _ratio(
            exact_enumeration_seconds,
            full_update_seconds,
        ),
        "runtime_in_run_vs_full_update_overhead_ratio": _ratio(
            in_run_scores_seconds,
            full_update_seconds,
        ),
        "runtime_in_run_vs_exact_ratio": _ratio(
            in_run_scores_seconds,
            exact_enumeration_seconds,
        ),
        "runtime_in_run_faster_than_exact": in_run_scores_seconds
        < exact_enumeration_seconds,
        "runtime_overhead_reported": (
            full_update_seconds > 0.0
            and exact_enumeration_seconds > 0.0
            and in_run_scores_seconds > 0.0
        ),
    }


def run_cuda_data_shapley_preflight(max_vram_gb: float = 24.0) -> dict:
    device = t.device("cuda")
    t.cuda.reset_peak_memory_stats()
    train_x, train_y, val_x, val_y = _move_problem_to_device(
        toy_data_shapley_problem(),
        device,
    )

    values = _cuda_data_coalition_values(train_x, train_y, val_x, val_y)
    exact = exact_shapley_values(values, num_players=int(train_x.shape[0]))
    sampled = sampled_permutation_shapley_values(
        values,
        num_players=int(train_x.shape[0]),
        num_samples=DATA_SHAPLEY_MC_SAMPLES,
        seed=0,
    )
    sampled_max_error = float((exact - sampled).abs().max().item())
    gradient_scores = _cuda_in_run_gradient_scores(train_x, train_y, val_x, val_y)
    correlation = pearson_correlation(exact, gradient_scores)
    harmful_index = int(exact.argmin().item())
    helpful_index = int(exact.argmax().item())
    in_run_harmful_index = int(gradient_scores.argmin().item())
    in_run_helpful_index = int(gradient_scores.argmax().item())
    full = frozenset(range(int(train_x.shape[0])))
    harmful_removal_delta = values[full - {harmful_index}] - values[full]
    actual_full_utility = _actual_full_batch_one_step_utility(train_x, train_y, val_x, val_y)
    random_train_x, random_train_y, random_val_x, random_val_y = _move_problem_to_device(
        _random_data_shapley_problem(),
        device,
    )
    random_data_control = _cuda_signal_failure_metrics(
        random_train_x,
        random_train_y,
        random_val_x,
        random_val_y,
        signal_exact=exact,
    )
    random_data_control["random_data_attribution_fails"] = (
        random_data_control["harmful_index"] != harmful_index
        and random_data_control["helpful_index"] != helpful_index
        and random_data_control["max_abs_signal_correlation"]
        <= DATA_SHAPLEY_RANDOM_CONTROL_MAX_ABS_CORRELATION
    )
    label_train_x, label_train_y, label_val_x, label_val_y = _move_problem_to_device(
        _label_shuffled_data_shapley_problem(),
        device,
    )
    label_shuffle_control = _cuda_signal_failure_metrics(
        label_train_x,
        label_train_y,
        label_val_x,
        label_val_y,
        signal_exact=exact,
    )
    label_shuffle_control["label_shuffled_attribution_fails"] = (
        label_shuffle_control["harmful_index"] != harmful_index
        and label_shuffle_control["in_run_harmful_index"] != harmful_index
        and label_shuffle_control["signal_correlation"]
        <= DATA_SHAPLEY_LABEL_SHUFFLE_MAX_CORRELATION
        and label_shuffle_control["in_run_signal_correlation"]
        <= DATA_SHAPLEY_LABEL_SHUFFLE_MAX_CORRELATION
    )
    runtime_metrics = _cuda_runtime_overhead_metrics(train_x, train_y, val_x, val_y)

    t.cuda.synchronize()
    peak_vram_gb = t.cuda.max_memory_allocated() / 1024**3
    preflight_passed = (
        harmful_index == 3
        and float(exact[harmful_index].item()) < 0.0
        and harmful_removal_delta > 0.0
        and sampled_max_error <= DATA_SHAPLEY_MC_MAX_ERROR
        and correlation >= DATA_SHAPLEY_MIN_CORRELATION
        and in_run_harmful_index == harmful_index
        and in_run_helpful_index == helpful_index
        and abs(actual_full_utility - values[full]) <= 1e-9
        and random_data_control["random_data_attribution_fails"]
        and label_shuffle_control["label_shuffled_attribution_fails"]
        and runtime_metrics["runtime_overhead_reported"]
        and runtime_metrics["runtime_in_run_vs_exact_ratio"]
        <= DATA_SHAPLEY_RUNTIME_IN_RUN_VS_EXACT_MAX_RATIO
        and peak_vram_gb <= max_vram_gb
    )
    return {
        "preflight_passed": preflight_passed,
        "cuda_available": True,
        "device": t.cuda.get_device_name(0),
        "model_family": "cuda_one_step_linear_regression_data_shapley",
        "training_example_count": int(train_x.shape[0]),
        "coalition_count": 2 ** int(train_x.shape[0]),
        "monte_carlo_samples": DATA_SHAPLEY_MC_SAMPLES,
        "learning_rate": DATA_SHAPLEY_LR,
        "exact_values": exact.tolist(),
        "sampled_values": sampled.tolist(),
        "sampled_max_abs_error": sampled_max_error,
        "sampled_approximates_exact": sampled_max_error <= DATA_SHAPLEY_MC_MAX_ERROR,
        "gradient_scores": gradient_scores.tolist(),
        "pearson_correlation": correlation,
        "harmful_index": harmful_index,
        "helpful_index": helpful_index,
        "in_run_harmful_index": in_run_harmful_index,
        "in_run_helpful_index": in_run_helpful_index,
        "identifies_harmful": in_run_harmful_index == harmful_index,
        "identifies_helpful": in_run_helpful_index == helpful_index,
        "harmful_value": float(exact[harmful_index].item()),
        "harmful_removal_delta": harmful_removal_delta,
        "actual_full_batch_one_step_utility": actual_full_utility,
        "coalition_full_utility": values[full],
        "random_data_exact_values": random_data_control["exact_values"],
        "random_data_in_run_scores": random_data_control["in_run_scores"],
        "random_data_harmful_index": random_data_control["harmful_index"],
        "random_data_helpful_index": random_data_control["helpful_index"],
        "random_data_in_run_harmful_index": random_data_control["in_run_harmful_index"],
        "random_data_in_run_helpful_index": random_data_control["in_run_helpful_index"],
        "random_data_signal_correlation": random_data_control["signal_correlation"],
        "random_data_in_run_signal_correlation": random_data_control[
            "in_run_signal_correlation"
        ],
        "random_data_max_abs_signal_correlation": random_data_control[
            "max_abs_signal_correlation"
        ],
        "random_data_attribution_fails": random_data_control[
            "random_data_attribution_fails"
        ],
        "label_shuffled_exact_values": label_shuffle_control["exact_values"],
        "label_shuffled_in_run_scores": label_shuffle_control["in_run_scores"],
        "label_shuffled_harmful_index": label_shuffle_control["harmful_index"],
        "label_shuffled_helpful_index": label_shuffle_control["helpful_index"],
        "label_shuffled_in_run_harmful_index": label_shuffle_control[
            "in_run_harmful_index"
        ],
        "label_shuffled_in_run_helpful_index": label_shuffle_control[
            "in_run_helpful_index"
        ],
        "label_shuffled_signal_correlation": label_shuffle_control[
            "signal_correlation"
        ],
        "label_shuffled_in_run_signal_correlation": label_shuffle_control[
            "in_run_signal_correlation"
        ],
        "label_shuffled_max_abs_signal_correlation": label_shuffle_control[
            "max_abs_signal_correlation"
        ],
        "label_shuffled_attribution_fails": label_shuffle_control[
            "label_shuffled_attribution_fails"
        ],
        "runtime_measurement_repeats": runtime_metrics["runtime_measurement_repeats"],
        "runtime_full_update_seconds": runtime_metrics["runtime_full_update_seconds"],
        "runtime_exact_enumeration_seconds": runtime_metrics[
            "runtime_exact_enumeration_seconds"
        ],
        "runtime_in_run_scores_seconds": runtime_metrics["runtime_in_run_scores_seconds"],
        "runtime_exact_vs_full_update_overhead_ratio": runtime_metrics[
            "runtime_exact_vs_full_update_overhead_ratio"
        ],
        "runtime_in_run_vs_full_update_overhead_ratio": runtime_metrics[
            "runtime_in_run_vs_full_update_overhead_ratio"
        ],
        "runtime_in_run_vs_exact_ratio": runtime_metrics["runtime_in_run_vs_exact_ratio"],
        "runtime_in_run_faster_than_exact": runtime_metrics[
            "runtime_in_run_faster_than_exact"
        ],
        "runtime_overhead_reported": runtime_metrics["runtime_overhead_reported"],
        "peak_vram_gb": peak_vram_gb,
        "within_vram_budget": peak_vram_gb <= max_vram_gb,
        "full_path": (
            "Enumerate Data Shapley coalitions on CUDA, run an actual one-step "
            "linear training update, compare exact, Monte Carlo, and in-run "
            "autograd gradient-dot scores, run random-data and label-shuffle "
            "failure controls, and measure runtime overhead."
        ),
    }


def run_full_experiment(max_vram_gb: float = 24.0) -> dict:
    """Run the validated experiment path used by the verification report."""

    return run_gpu_test(max_vram_gb=max_vram_gb)


if MAIN:
    print(run_smoke_test())
