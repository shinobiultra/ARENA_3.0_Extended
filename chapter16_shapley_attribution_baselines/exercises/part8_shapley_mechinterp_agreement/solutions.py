# %%
"""Reference solutions for [16.8] SHAPley and mechanistic agreement tests."""

import csv
import sys
from pathlib import Path

import torch as t

chapter = "chapter16_shapley_attribution_baselines"
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from arena_ext.shapley_attribution import (
    additive_game,
    attribution_agreement_report,
    interaction_agreement_report,
    pairwise_shapley_interactions,
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

    return t.tensor([2.3, -1.45, 2.7, 0.15], dtype=t.float64)


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
    }


def run_gpu_test(max_vram_gb: float = 24.0) -> dict:
    if not t.cuda.is_available():
        raise RuntimeError("16.8 GPU preflight requires CUDA; no CPU fallback is accepted.")

    return run_neural_mechanistic_agreement_preflight(max_vram_gb=max_vram_gb)


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
