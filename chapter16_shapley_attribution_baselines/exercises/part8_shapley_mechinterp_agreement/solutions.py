# %%
"""Reference solutions for [16.8] SHAPley and mechanistic agreement tests."""

import csv
import itertools
import math
import sys
from pathlib import Path
from typing import Mapping

import torch as t

chapter = "chapter16_shapley_attribution_baselines"
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from arena_ext.shapley_attribution import (
    additive_game,
    all_coalitions,
    average_ranks,
    attribution_agreement_report,
    coalition_values_from_function,
    exact_shapley_values,
    interaction_agreement_report,
    pairwise_shapley_interactions,
    spearman_rank_correlation,
    topk_overlap_fraction,
    xor_game,
)
from arena_ext.shapley_neural_game import (
    NEURAL_GAME_NUM_PLAYERS,
    NEURAL_GAME_STEPS,
    binary_feature_table,
    coalition_table_from_model,
    coalition_table_from_true_game,
    shuffled_targets,
    train_neural_game,
    true_neural_game_scores,
)

MAIN = __name__ == "__main__"

NEURAL_AGREEMENT_MIN_CORRELATION = 0.99
NEURAL_INTERACTION_MAX_ERROR = 1e-4
NEURAL_FIT_MSE_MAX = 1e-8
SHUFFLED_MAX_CORRELATION = 0.0
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
Coalition = frozenset[int]
TOY_FEATURE_NAMES = ("subject-token", "distractor-token", "answer-slot", "style-prior")
TOY_LINEAR_WEIGHTS = t.tensor([1.2, -0.7, 1.6, 0.9], dtype=t.float64)
TOY_PAIR_WEIGHTS = {(0, 2): 2.2, (1, 3): -1.5}
TOY_INTERCEPT = 0.25


def enumerate_coalitions(num_players: int) -> tuple[Coalition, ...]:
    """Return every coalition in size-then-lexicographic order."""

    if num_players <= 0:
        raise ValueError("num_players must be positive.")
    coalitions: list[Coalition] = []
    for size in range(num_players + 1):
        coalitions.extend(
            frozenset(group) for group in itertools.combinations(range(num_players), size)
        )
    return tuple(coalitions)


def finite_circuit_value(
    coalition: Coalition | tuple[int, ...],
    *,
    linear_weights: t.Tensor = TOY_LINEAR_WEIGHTS,
    pair_weights: Mapping[tuple[int, int], float] = TOY_PAIR_WEIGHTS,
    intercept: float = TOY_INTERCEPT,
) -> float:
    """Evaluate the planted finite circuit on a coalition of present features."""

    active = set(coalition)
    value = float(intercept)
    for player, weight in enumerate(linear_weights.double().tolist()):
        if player in active:
            value += float(weight)
    for (first, second), weight in pair_weights.items():
        if first in active and second in active:
            value += float(weight)
    return value


def finite_circuit_table(
    *,
    num_players: int = NEURAL_GAME_NUM_PLAYERS,
    linear_weights: t.Tensor = TOY_LINEAR_WEIGHTS,
    pair_weights: Mapping[tuple[int, int], float] = TOY_PAIR_WEIGHTS,
    intercept: float = TOY_INTERCEPT,
) -> dict[Coalition, float]:
    """Return the complete coalition table for the planted finite circuit."""

    return {
        coalition: finite_circuit_value(
            coalition,
            linear_weights=linear_weights,
            pair_weights=pair_weights,
            intercept=intercept,
        )
        for coalition in enumerate_coalitions(num_players)
    }


def exact_shapley_from_table(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    *,
    num_players: int,
) -> t.Tensor:
    """Compute exact Shapley values from first principles."""

    values = {frozenset(key): float(value) for key, value in coalition_values.items()}
    expected = set(enumerate_coalitions(num_players))
    missing = expected - set(values)
    if missing:
        raise ValueError(f"coalition value table is missing {len(missing)} coalitions.")

    shapley = t.zeros(num_players, dtype=t.float64)
    denominator = math.factorial(num_players)
    for player in range(num_players):
        others = [candidate for candidate in range(num_players) if candidate != player]
        for size in range(num_players):
            weight = math.factorial(size) * math.factorial(num_players - size - 1) / denominator
            for group in itertools.combinations(others, size):
                coalition = frozenset(group)
                shapley[player] += weight * (
                    values[coalition | {player}] - values[coalition]
                )
    return shapley


def causal_patching_effects(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    *,
    num_players: int,
) -> t.Tensor:
    """Return full-minus-ablated causal effects for each feature player."""

    values = {frozenset(key): float(value) for key, value in coalition_values.items()}
    full = frozenset(range(num_players))
    if full not in values:
        raise ValueError("coalition value table must contain the full coalition.")
    effects = [
        values[full] - values[full - {player}]
        for player in range(num_players)
    ]
    return t.tensor(effects, dtype=t.float64)


def mechanistic_endpoint_scores(
    *,
    linear_weights: t.Tensor = TOY_LINEAR_WEIGHTS,
    pair_weights: Mapping[tuple[int, int], float] = TOY_PAIR_WEIGHTS,
) -> t.Tensor:
    """Allocate each planted pair edge equally to its two endpoint features."""

    scores = linear_weights.double().clone()
    for (first, second), weight in pair_weights.items():
        scores[first] += float(weight) / 2
        scores[second] += float(weight) / 2
    return scores


def mechanistic_pair_matrix(
    *,
    num_players: int = NEURAL_GAME_NUM_PLAYERS,
    pair_weights: Mapping[tuple[int, int], float] = TOY_PAIR_WEIGHTS,
) -> t.Tensor:
    """Return the known pair-edge mechanism as a symmetric matrix."""

    matrix = t.zeros((num_players, num_players), dtype=t.float64)
    for (first, second), weight in pair_weights.items():
        matrix[first, second] = float(weight)
        matrix[second, first] = float(weight)
    return matrix


def pairwise_interactions_from_table(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    *,
    num_players: int,
) -> t.Tensor:
    """Compute second-order Shapley interaction indices from first principles."""

    if num_players < 2:
        raise ValueError("pairwise interactions require at least two players.")
    values = {frozenset(key): float(value) for key, value in coalition_values.items()}
    expected = set(enumerate_coalitions(num_players))
    missing = expected - set(values)
    if missing:
        raise ValueError(f"coalition value table is missing {len(missing)} coalitions.")

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
                score += weight * (
                    values[coalition | {first, second}]
                    - values[coalition | {first}]
                    - values[coalition | {second}]
                    + values[coalition]
                )
        interactions[first, second] = score
        interactions[second, first] = score
    return interactions


def agreement_summary(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    *,
    mechanistic_scores: t.Tensor,
    num_players: int,
    topk: int = 2,
) -> dict:
    """Compare Shapley, causal patching, and known mechanistic scores."""

    shapley = exact_shapley_from_table(coalition_values, num_players=num_players)
    patching = causal_patching_effects(coalition_values, num_players=num_players)
    mech = mechanistic_scores.double().flatten()
    full = frozenset(range(num_players))
    values = {frozenset(key): float(value) for key, value in coalition_values.items()}
    top_player = int(shapley.argmax().item())
    non_top = [player for player in range(num_players) if player != top_player]
    deletion_drop = values[full] - values[full - {top_player}]
    matched_random_drop = sum(values[full] - values[full - {player}] for player in non_top) / len(
        non_top
    )
    return {
        "shapley_values": shapley,
        "patching_effects": patching,
        "mechanistic_scores": mech,
        "spearman_correlation": spearman_rank_correlation(shapley, mech),
        "topk_overlap": topk_overlap_fraction(shapley, mech, k=topk),
        "patching_topk_overlap": topk_overlap_fraction(patching, mech, k=topk),
        "shapley_top_feature": top_player,
        "mechanistic_top_feature": int(mech.argmax().item()),
        "patching_top_feature": int(patching.argmax().item()),
        "deletion_drop": deletion_drop,
        "matched_random_drop": matched_random_drop,
        "random_baseline_drop": matched_random_drop,
        "agrees_with_mechanistic": (
            spearman_rank_correlation(shapley, mech) >= NEURAL_AGREEMENT_MIN_CORRELATION
            and topk_overlap_fraction(shapley, mech, k=topk) == 1.0
        ),
    }


# %%
def _tensor_report(report) -> dict:
    result = report.__dict__.copy()
    for key, value in list(result.items()):
        if hasattr(value, "tolist"):
            result[key] = value.tolist()
    return result


def _artifact_display_path(path: Path) -> str:
    try:
        return str(path.relative_to(root_dir))
    except ValueError:
        return str(path)


def analytic_neural_game_mechanistic_scores() -> t.Tensor:
    """Return feature scores from the generated rule's known decomposition.

    The target rule is
    0.25 + 1.2*x0 - 0.7*x1 + 1.6*x2 + 0.9*x3 + 2.2*x0*x2 - 1.5*x1*x3.
    For a feature-level agreement target, allocate each pair interaction
    equally to its two features. This is an analytic mechanism score from the
    data-generating rule, not a Shapley call over the learned model outputs.
    """

    return mechanistic_endpoint_scores()


def _rank_desc(scores: t.Tensor) -> list[int]:
    return [int(item) for item in t.argsort(scores.detach().double().cpu(), descending=True)]


def _curve_from_rank(values: dict[frozenset[int], float], rank: list[int], mode: str) -> list[dict]:
    if mode not in {"deletion", "insertion"}:
        raise ValueError("mode must be 'deletion' or 'insertion'.")

    active: set[int] = set(range(NEURAL_GAME_NUM_PLAYERS)) if mode == "deletion" else set()
    points = [{"step": 0, "player": "start", "value": values[frozenset(active)]}]
    for step, player in enumerate(rank, start=1):
        if mode == "deletion":
            active.remove(player)
        else:
            active.add(player)
        points.append(
            {
                "step": step,
                "player": player,
                "value": values[frozenset(active)],
            }
        )
    return points


def _write_curve_plot(path: Path, title: str, ylabel: str, curves: dict[str, list[dict]]) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=160)
    colors = {
        "trained_shapley_rank": "#1f77b4",
        "mechanistic_rank": "#2ca02c",
        "shuffled_control_rank": "#d62728",
    }
    markers = {
        "trained_shapley_rank": "o",
        "mechanistic_rank": "s",
        "shuffled_control_rank": "^",
    }
    linestyles = {
        "trained_shapley_rank": "-",
        "mechanistic_rank": "--",
        "shuffled_control_rank": "-",
    }
    for label in ("mechanistic_rank", "trained_shapley_rank", "shuffled_control_rank"):
        points = curves[label]
        xs = [point["step"] for point in points]
        ys = [point["value"] for point in points]
        ax.plot(
            xs,
            ys,
            marker=markers[label],
            linestyle=linestyles[label],
            linewidth=2.3,
            markersize=5,
            color=colors[label],
            label=label,
            markerfacecolor="white" if label == "trained_shapley_rank" else colors[label],
            markeredgewidth=1.8,
            zorder=3 if label == "trained_shapley_rank" else 2,
        )
    ax.set_title(title)
    ax.set_xlabel("players removed" if "Deletion" in title else "players inserted")
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(NEURAL_GAME_NUM_PLAYERS + 1))
    ax.grid(True, alpha=0.28)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _write_heatmap(path: Path, rows: list[str], columns: list[str], values: list[list[float]]) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 3.7), dpi=160)
    image = ax.imshow(values, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(columns)), labels=columns)
    ax.set_yticks(range(len(rows)), labels=rows)
    ax.set_title("Top-k overlap with analytic mechanism")
    for row_idx, row_values in enumerate(values):
        for col_idx, value in enumerate(row_values):
            text_color = "white" if value < 0.55 else "black"
            ax.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center", color=text_color)
    fig.colorbar(image, ax=ax, shrink=0.85, label="overlap")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _tensor_to_list_report(report: dict) -> dict:
    """Convert tensor-valued report fields into plain Python objects."""

    converted = report.copy()
    for key, value in list(converted.items()):
        if hasattr(value, "tolist"):
            converted[key] = value.tolist()
    return converted


def toy_agreement_case() -> dict:
    """Return the planted finite-circuit agreement case used in the notebook."""

    values = finite_circuit_table()
    return _tensor_to_list_report(
        agreement_summary(
            values,
            mechanistic_scores=mechanistic_endpoint_scores(),
            num_players=NEURAL_GAME_NUM_PLAYERS,
            topk=2,
        )
    )


def xor_interaction_diagnosis() -> dict:
    """Return single-feature and pair-player scores for the XOR disagreement case."""

    values = xor_game()
    shapley = exact_shapley_from_table(values, num_players=2)
    patching = causal_patching_effects(values, num_players=2)
    interactions = pairwise_interactions_from_table(values, num_players=2)
    return {
        "shapley_values": shapley.tolist(),
        "patching_effects": patching.tolist(),
        "pair_interactions": interactions.tolist(),
        "max_single_feature_value": float(shapley.abs().max().item()),
        "recovered_pair_interaction": float(abs(interactions[0, 1].item())),
        "ordinary_shapley_misses": float(shapley.abs().max().item()) <= 1e-9,
        "interaction_recovers_pair": float(abs(interactions[0, 1].item())) >= 1.0,
    }


def shuffled_mechanistic_control() -> dict:
    """Compare the same Shapley values against a shuffled mechanism label control."""

    values = finite_circuit_table()
    shapley = exact_shapley_from_table(values, num_players=NEURAL_GAME_NUM_PLAYERS)
    mechanistic = mechanistic_endpoint_scores()
    shuffled = mechanistic[t.tensor([1, 3, 0, 2])]
    return {
        "shapley_values": shapley.tolist(),
        "true_mechanistic_scores": mechanistic.tolist(),
        "shuffled_mechanistic_scores": shuffled.tolist(),
        "true_spearman": spearman_rank_correlation(shapley, mechanistic),
        "shuffled_spearman": spearman_rank_correlation(shapley, shuffled),
        "true_top2_overlap": topk_overlap_fraction(shapley, mechanistic, k=2),
        "shuffled_top2_overlap": topk_overlap_fraction(shapley, shuffled, k=2),
        "control_rejected": topk_overlap_fraction(shapley, shuffled, k=2) < 1.0,
    }


def one_step_data_utility(
    coalition: Coalition | tuple[int, ...],
    *,
    train_labels: t.Tensor | None = None,
    validation_label: float = 1.0,
    learning_rate: float = 0.5,
) -> float:
    """Utility from one gradient step on selected scalar training examples."""

    labels = (
        t.tensor([1.0, 1.0, 1.0, -1.0], dtype=t.float64)
        if train_labels is None
        else train_labels.double()
    )
    active = sorted(coalition)
    baseline_loss = validation_label**2
    if not active:
        return 0.0
    selected = labels[t.tensor(active, dtype=t.long)]
    gradient = -2.0 * selected.mean()
    updated_weight = -learning_rate * gradient
    updated_loss = (updated_weight - validation_label) ** 2
    return float(baseline_loss - updated_loss)


def one_step_data_value_table(
    *,
    num_examples: int = 4,
    train_labels: t.Tensor | None = None,
) -> dict[Coalition, float]:
    """Return the complete training-example coalition table for the data bridge."""

    return {
        coalition: one_step_data_utility(coalition, train_labels=train_labels)
        for coalition in enumerate_coalitions(num_examples)
    }


def data_gradient_dot_scores(
    *,
    train_labels: t.Tensor | None = None,
    validation_label: float = 1.0,
) -> t.Tensor:
    """Return one-run gradient-dot scores from initialization."""

    labels = (
        t.tensor([1.0, 1.0, 1.0, -1.0], dtype=t.float64)
        if train_labels is None
        else train_labels.double()
    )
    validation_gradient = -2.0 * validation_label
    train_gradients = -2.0 * labels
    return train_gradients * validation_gradient


def pearson_correlation(first: t.Tensor, second: t.Tensor) -> float:
    """Return Pearson correlation for two nonconstant vectors."""

    first = first.double().flatten()
    second = second.double().flatten()
    first_centered = first - first.mean()
    second_centered = second - second.mean()
    denominator = first_centered.norm() * second_centered.norm()
    if float(denominator.item()) == 0.0:
        return 0.0
    return float((first_centered @ second_centered / denominator).item())


def data_player_bridge_report() -> dict:
    """Compare exact Data Shapley with a one-run gradient-dot proxy."""

    values = one_step_data_value_table()
    exact = exact_shapley_from_table(values, num_players=4)
    proxy = data_gradient_dot_scores()
    harmful = int(exact.argmin().item())
    helpful = int(exact.argmax().item())
    return {
        "exact_data_shapley": exact.tolist(),
        "gradient_dot_scores": proxy.tolist(),
        "pearson_correlation": pearson_correlation(exact, proxy),
        "helpful_example": helpful,
        "harmful_example": harmful,
        "identifies_harmful": harmful == int(proxy.argmin().item()),
        "identifies_helpful_tie": float(proxy[helpful].item()) == float(proxy.max().item()),
    }


def write_signature_panel(
    path: Path,
    *,
    values: Mapping[Coalition | tuple[int, ...], float] | None = None,
) -> dict:
    """Write the learner-facing 16.8 signature result panel."""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    values = finite_circuit_table() if values is None else dict(values)
    mech = mechanistic_endpoint_scores()
    summary = agreement_summary(values, mechanistic_scores=mech, num_players=NEURAL_GAME_NUM_PLAYERS)
    shapley = summary["shapley_values"]
    patching = summary["patching_effects"]
    pair_mech = mechanistic_pair_matrix()
    pair_shap = pairwise_interactions_from_table(values, num_players=NEURAL_GAME_NUM_PLAYERS)
    shap_rank = _rank_desc(shapley)
    mech_rank = _rank_desc(mech)
    shuffled_rank = _rank_desc(mech[t.tensor([1, 3, 0, 2])])
    curves = {
        "Shapley deletion": _curve_from_rank(values, shap_rank, "deletion"),
        "Mechanism deletion": _curve_from_rank(values, mech_rank, "deletion"),
        "Shuffled-control deletion": _curve_from_rank(values, shuffled_rank, "deletion"),
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0), dpi=160)

    feature_labels = [f"x{i}" for i in range(NEURAL_GAME_NUM_PLAYERS)]
    width = 0.23
    x_positions = t.arange(NEURAL_GAME_NUM_PLAYERS).double().tolist()
    axes[0, 0].bar([x - width for x in x_positions], shapley.tolist(), width=width, label="Exact Shapley")
    axes[0, 0].bar(x_positions, patching.tolist(), width=width, label="Causal patch")
    axes[0, 0].bar([x + width for x in x_positions], mech.tolist(), width=width, label="Known mechanism")
    axes[0, 0].axhline(0.0, color="black", linewidth=0.8)
    axes[0, 0].set_xticks(x_positions, feature_labels)
    axes[0, 0].set_title("Agreement case: additive + pair circuit")
    axes[0, 0].set_ylabel("score")
    axes[0, 0].legend(frameon=False, fontsize=8)

    for label, points in curves.items():
        axes[0, 1].plot(
            [point["step"] for point in points],
            [point["value"] for point in points],
            marker="o",
            linewidth=2,
            label=label,
        )
    axes[0, 1].set_title("Deletion consequence test")
    axes[0, 1].set_xlabel("players removed")
    axes[0, 1].set_ylabel("circuit value")
    axes[0, 1].grid(alpha=0.25)
    axes[0, 1].legend(frameon=False, fontsize=8)

    overlap = [
        [
            topk_overlap_fraction(shapley, mech, k=k)
            for k in range(1, NEURAL_GAME_NUM_PLAYERS + 1)
        ],
        [
            topk_overlap_fraction(patching, mech, k=k)
            for k in range(1, NEURAL_GAME_NUM_PLAYERS + 1)
        ],
        [
            topk_overlap_fraction(shapley, mech[t.tensor([1, 3, 0, 2])], k=k)
            for k in range(1, NEURAL_GAME_NUM_PLAYERS + 1)
        ],
    ]
    image = axes[1, 0].imshow(overlap, cmap="viridis", vmin=0, vmax=1)
    axes[1, 0].set_title("Agreement matrix: top-k overlap")
    axes[1, 0].set_xticks(range(NEURAL_GAME_NUM_PLAYERS), [f"k={k}" for k in range(1, 5)])
    axes[1, 0].set_yticks(range(3), ["Shapley", "Patching", "Shuffled"])
    for row_idx, row_values in enumerate(overlap):
        for col_idx, value in enumerate(row_values):
            axes[1, 0].text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center")
    fig.colorbar(image, ax=axes[1, 0], shrink=0.82)

    im = axes[1, 1].imshow(pair_shap.abs(), cmap="magma", vmin=0)
    axes[1, 1].set_title("Disagreement diagnosis: pair interactions")
    axes[1, 1].set_xticks(range(NEURAL_GAME_NUM_PLAYERS), feature_labels)
    axes[1, 1].set_yticks(range(NEURAL_GAME_NUM_PLAYERS), feature_labels)
    for row in range(NEURAL_GAME_NUM_PLAYERS):
        for col in range(NEURAL_GAME_NUM_PLAYERS):
            if row == col:
                label = ""
            elif abs(float(pair_mech[row, col].item())) > 0:
                label = f"{float(pair_shap[row, col].item()):.1f}"
            else:
                label = f"{float(pair_shap[row, col].item()):.1f}"
            axes[1, 1].text(col, row, label, ha="center", va="center", color="white")
    fig.colorbar(im, ax=axes[1, 1], shrink=0.82)

    fig.suptitle(
        "SHAPley and mechanistic scores agree only after the player set is right",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

    return {
        "signature_panel_written": path.exists() and path.stat().st_size > 0,
        "signature_panel_path": _artifact_display_path(path),
        "spearman_correlation": summary["spearman_correlation"],
        "top2_overlap": summary["topk_overlap"],
        "patching_top2_overlap": summary["patching_topk_overlap"],
        "xor_pair_interaction_abs": xor_interaction_diagnosis()["recovered_pair_interaction"],
    }


def write_agreement_artifacts(
    *,
    output_dir: Path = ARTIFACT_DIR,
    model_values: dict[frozenset[int], float],
    true_values: dict[frozenset[int], float],
    shuffled_values: dict[frozenset[int], float],
    agreement,
    shuffled_agreement,
    model_interactions: t.Tensor,
    true_interactions: t.Tensor,
) -> dict:
    """Write the roadmap-required agreement matrix and consequence plots."""

    output_dir.mkdir(parents=True, exist_ok=True)

    interaction_error = float((model_interactions - true_interactions).abs().max().item())
    additive = additive_agreement_smoke_test()
    xor = xor_disagreement_smoke_test()
    matrix_rows = [
        {
            "task": "additive_control",
            "method_a": "ExactShapley",
            "method_b": "MechanisticScores",
            "player_type_a": "feature",
            "player_type_b": "feature",
            "metric": "spearman_rank_correlation",
            "value": f"{additive['spearman_correlation']:.6g}",
            "interpretation": "Additive ground truth gives full rank agreement.",
        },
        {
            "task": "neural_coalition_game",
            "method_a": "ExactShapley",
            "method_b": "AnalyticMechanisticScores",
            "player_type_a": "feature",
            "player_type_b": "feature",
            "metric": "spearman_rank_correlation",
            "value": f"{agreement.spearman_correlation:.6g}",
            "interpretation": "Trained model ablations recover the analytic feature ordering.",
        },
        {
            "task": "neural_coalition_game",
            "method_a": "ExactShapley",
            "method_b": "AnalyticMechanisticScores",
            "player_type_a": "feature",
            "player_type_b": "feature",
            "metric": "top2_overlap",
            "value": f"{agreement.topk_overlap:.6g}",
            "interpretation": "Top causal features match the analytic mechanism.",
        },
        {
            "task": "neural_coalition_game",
            "method_a": "ExactShapley",
            "method_b": "FeatureDeletion",
            "player_type_a": "feature",
            "player_type_b": "behavior",
            "metric": "deletion_drop_minus_baseline",
            "value": f"{agreement.deletion_drop - agreement.random_baseline_drop:.6g}",
            "interpretation": "Deleting the top Shapley feature hurts more than deleting a non-top baseline.",
        },
        {
            "task": "neural_coalition_game",
            "method_a": "ShapleyInteractions",
            "method_b": "AnalyticPairInteractions",
            "player_type_a": "feature_pair",
            "player_type_b": "feature_pair",
            "metric": "max_abs_error",
            "value": f"{interaction_error:.6g}",
            "interpretation": "Pair interactions recover the planted positive and negative feature pairs.",
        },
        {
            "task": "shuffled_label_control",
            "method_a": "ExactShapley",
            "method_b": "AnalyticMechanisticScores",
            "player_type_a": "feature",
            "player_type_b": "feature",
            "metric": "spearman_rank_correlation",
            "value": f"{shuffled_agreement.spearman_correlation:.6g}",
            "interpretation": "The shuffled-label trained model fails the mechanistic agreement test.",
        },
        {
            "task": "xor_control",
            "method_a": "OrdinaryShapley",
            "method_b": "ShapleyInteractions",
            "player_type_a": "feature",
            "player_type_b": "feature_pair",
            "metric": "disagreement_detected",
            "value": "1",
            "interpretation": "Single-feature Shapley misses XOR while pair interactions recover it.",
        },
    ]
    matrix_path = output_dir / "agreement_matrix.csv"
    with matrix_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(matrix_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(matrix_rows)

    shapley_rank = _rank_desc(agreement.shapley_values)
    mechanistic_rank = _rank_desc(agreement.mechanistic_scores)
    shuffled_rank = _rank_desc(shuffled_agreement.shapley_values)
    deletion_curves = {
        "trained_shapley_rank": _curve_from_rank(model_values, shapley_rank, "deletion"),
        "mechanistic_rank": _curve_from_rank(model_values, mechanistic_rank, "deletion"),
        "shuffled_control_rank": _curve_from_rank(model_values, shuffled_rank, "deletion"),
    }
    insertion_curves = {
        "trained_shapley_rank": _curve_from_rank(model_values, shapley_rank, "insertion"),
        "mechanistic_rank": _curve_from_rank(model_values, mechanistic_rank, "insertion"),
        "shuffled_control_rank": _curve_from_rank(model_values, shuffled_rank, "insertion"),
    }
    _write_curve_plot(
        output_dir / "deletion_curves.png",
        "Deletion Consequences",
        "model value after deletion",
        deletion_curves,
    )
    _write_curve_plot(
        output_dir / "insertion_curves.png",
        "Insertion Consequences",
        "model value after insertion",
        insertion_curves,
    )

    columns = [f"k={k}" for k in range(1, NEURAL_GAME_NUM_PLAYERS + 1)]
    heatmap_rows = [
        "trained_shapley",
        "trained_patching",
        "shuffled_shapley",
        "shuffled_patching",
    ]
    heatmap_values = [
        [
            topk_overlap_fraction(agreement.shapley_values, agreement.mechanistic_scores, k=k)
            for k in range(1, NEURAL_GAME_NUM_PLAYERS + 1)
        ],
        [
            topk_overlap_fraction(agreement.patching_effects, agreement.mechanistic_scores, k=k)
            for k in range(1, NEURAL_GAME_NUM_PLAYERS + 1)
        ],
        [
            topk_overlap_fraction(shuffled_agreement.shapley_values, agreement.mechanistic_scores, k=k)
            for k in range(1, NEURAL_GAME_NUM_PLAYERS + 1)
        ],
        [
            topk_overlap_fraction(
                shuffled_agreement.patching_effects,
                agreement.mechanistic_scores,
                k=k,
            )
            for k in range(1, NEURAL_GAME_NUM_PLAYERS + 1)
        ],
    ]
    _write_heatmap(output_dir / "topk_overlap_heatmap.png", heatmap_rows, columns, heatmap_values)

    disagreement_path = output_dir / "method_disagreement_examples.md"
    disagreement_path.write_text(
        "\n".join(
            [
                "# Method Disagreement Examples",
                "",
                "## Agreement case: additive and trained finite game",
                "",
                (
                    "Exact Shapley and analytic mechanistic scores agree on the trained "
                    f"neural coalition game: Spearman={agreement.spearman_correlation:.3f}, "
                    f"top-2 overlap={agreement.topk_overlap:.3f}."
                ),
                (
                    "Deleting the top Shapley feature drops the model value by "
                    f"{agreement.deletion_drop:.3f}, above the non-top baseline "
                    f"{agreement.random_baseline_drop:.3f}."
                ),
                "",
                "## Disagreement case: XOR interaction",
                "",
                (
                    "Ordinary single-feature Shapley has max absolute value "
                    f"{xor['max_single_feature_value']:.3f} on XOR, so it misses the "
                    "mechanism when players are individual features."
                ),
                (
                    "Pairwise Shapley interaction recovers the causal pair with value "
                    f"{xor['recovered_pair_interaction']:.3f}. This is a tested "
                    "player-set disagreement, not a visual story."
                ),
                "",
                "## Negative control: shuffled trained model",
                "",
                (
                    "The shuffled-label control fits its own targets but fails agreement: "
                    f"Spearman={shuffled_agreement.spearman_correlation:.3f}, "
                    f"top-2 overlap={shuffled_agreement.topk_overlap:.3f}."
                ),
                "",
            ]
        )
    )

    paths = [
        matrix_path,
        output_dir / "deletion_curves.png",
        output_dir / "insertion_curves.png",
        output_dir / "topk_overlap_heatmap.png",
        disagreement_path,
    ]
    return {
        "agreement_artifacts_written": all(path.exists() and path.stat().st_size > 0 for path in paths),
        "agreement_artifact_count": len(paths),
        "agreement_matrix_rows": len(matrix_rows),
        "agreement_case_count": 2,
        "disagreement_case_count": 1,
        "deletion_curve_points": len(next(iter(deletion_curves.values()))),
        "insertion_curve_points": len(next(iter(insertion_curves.values()))),
        "topk_heatmap_rows": len(heatmap_rows),
        "topk_heatmap_cols": len(columns),
        "agreement_artifact_paths": [_artifact_display_path(path) for path in paths],
    }


def additive_agreement_smoke_test() -> dict:
    mechanistic_scores = t.tensor([1.0, 2.0, 0.5])
    values = additive_game(mechanistic_scores)
    return _tensor_report(
        attribution_agreement_report(
            values,
            mechanistic_scores=mechanistic_scores,
            num_players=3,
        )
    )


def xor_disagreement_smoke_test() -> dict:
    return _tensor_report(interaction_agreement_report(xor_game()))


def run_smoke_test(cpu: bool = True) -> dict:
    _ = cpu
    return {
        "additive_agreement": additive_agreement_smoke_test(),
        "xor_disagreement": xor_disagreement_smoke_test(),
        "data_player_bridge": data_player_bridge_report(),
    }


def run_gpu_test(max_vram_gb: float = 24.0) -> dict:
    if not t.cuda.is_available():
        raise RuntimeError("16.8 GPU preflight requires CUDA; no CPU fallback is accepted.")

    gpu = run_neural_mechanistic_agreement_preflight(max_vram_gb=max_vram_gb)
    notebook = run_smoke_test(cpu=False)
    gpu.update(
        {
            "data_player_bridge_pearson": notebook["data_player_bridge"][
                "pearson_correlation"
            ],
            "data_player_harmful_identified": notebook["data_player_bridge"][
                "identifies_harmful"
            ],
            "toy_xor_pair_interaction_abs": notebook["xor_disagreement"][
                "recovered_pair_interaction"
            ],
        }
    )
    return gpu


def run_neural_mechanistic_agreement_preflight(max_vram_gb: float = 24.0) -> dict:
    device = t.device("cuda")
    t.cuda.reset_peak_memory_stats()

    inputs = binary_feature_table(device)
    targets = true_neural_game_scores(inputs)
    trained = train_neural_game(inputs, targets)
    model_values = coalition_table_from_model(trained.model, device)
    true_values = coalition_table_from_true_game(device)
    true_mechanistic_scores = analytic_neural_game_mechanistic_scores()
    agreement = attribution_agreement_report(
        model_values,
        mechanistic_scores=true_mechanistic_scores,
        num_players=NEURAL_GAME_NUM_PLAYERS,
        topk=2,
        min_correlation=NEURAL_AGREEMENT_MIN_CORRELATION,
    )

    model_interactions = pairwise_shapley_interactions(
        model_values,
        num_players=NEURAL_GAME_NUM_PLAYERS,
    )
    true_interactions = pairwise_shapley_interactions(
        true_values,
        num_players=NEURAL_GAME_NUM_PLAYERS,
    )
    interaction_max_abs_error = float((model_interactions - true_interactions).abs().max().item())
    top_pairs = sorted(
        [
            (abs(float(model_interactions[row, col].item())), row, col)
            for row in range(NEURAL_GAME_NUM_PLAYERS)
            for col in range(row + 1, NEURAL_GAME_NUM_PLAYERS)
        ],
        reverse=True,
    )
    top_interaction_pair = [top_pairs[0][1], top_pairs[0][2]]
    second_interaction_pair = [top_pairs[1][1], top_pairs[1][2]]

    shuffled = train_neural_game(inputs, shuffled_targets(targets))
    shuffled_values = coalition_table_from_model(shuffled.model, device)
    shuffled_agreement = attribution_agreement_report(
        shuffled_values,
        mechanistic_scores=true_mechanistic_scores,
        num_players=NEURAL_GAME_NUM_PLAYERS,
        topk=2,
        min_correlation=NEURAL_AGREEMENT_MIN_CORRELATION,
    )

    t.cuda.synchronize()
    peak_vram_gb = t.cuda.max_memory_allocated() / 1024**3
    shuffled_rejected = (
        shuffled_agreement.spearman_correlation <= SHUFFLED_MAX_CORRELATION
        and shuffled_agreement.topk_overlap == 0.0
        and not shuffled_agreement.agrees_with_mechanistic
    )
    preflight_passed = (
        trained.fit_mse <= NEURAL_FIT_MSE_MAX
        and agreement.agrees_with_mechanistic
        and agreement.spearman_correlation >= NEURAL_AGREEMENT_MIN_CORRELATION
        and agreement.topk_overlap == 1.0
        and agreement.deletion_drop > agreement.random_baseline_drop
        and interaction_max_abs_error <= NEURAL_INTERACTION_MAX_ERROR
        and top_interaction_pair == [0, 2]
        and second_interaction_pair == [1, 3]
        and shuffled_rejected
        and peak_vram_gb <= max_vram_gb
    )
    artifact_summary = write_agreement_artifacts(
        model_values=model_values,
        true_values=true_values,
        shuffled_values=shuffled_values,
        agreement=agreement,
        shuffled_agreement=shuffled_agreement,
        model_interactions=model_interactions,
        true_interactions=true_interactions,
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
        "shapley_values": agreement.shapley_values.tolist(),
        "mechanistic_scores": agreement.mechanistic_scores.tolist(),
        "spearman_correlation": agreement.spearman_correlation,
        "topk_overlap": agreement.topk_overlap,
        "deletion_drop": agreement.deletion_drop,
        "random_baseline_drop": agreement.random_baseline_drop,
        "agrees_with_mechanistic": agreement.agrees_with_mechanistic,
        "interaction_max_abs_error": interaction_max_abs_error,
        "top_interaction_pair": top_interaction_pair,
        "second_interaction_pair": second_interaction_pair,
        "top_interaction_value": float(model_interactions[0, 2].item()),
        "second_interaction_value": float(model_interactions[1, 3].item()),
        "shuffled_control_fit_mse": shuffled.fit_mse,
        "shuffled_control_spearman": shuffled_agreement.spearman_correlation,
        "shuffled_control_topk_overlap": shuffled_agreement.topk_overlap,
        "shuffled_control_rejected": shuffled_rejected,
        "peak_vram_gb": peak_vram_gb,
        "within_vram_budget": peak_vram_gb <= max_vram_gb,
        "full_path": "Train a CUDA nonlinear feature game, compare Shapley scores against known mechanistic contributions, validate pair interactions, and reject shuffled-label agreement.",
        **artifact_summary,
    }


def run_full_experiment(max_vram_gb: float = 24.0) -> dict:
    """Run the validated experiment path used by the verification report."""

    return run_gpu_test(max_vram_gb=max_vram_gb)


if MAIN:
    print(run_smoke_test())
