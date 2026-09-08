"""Pure Matplotlib renderers for the DeepSHAP learner lab.

The functions in this module only summarize arrays produced by the learner's
methods and the frozen study.  They do not fit models or compute attributions.
Each function returns a figure and optionally writes one stable PNG stem.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np

BLUE = "#2563EB"
ORANGE = "#EA580C"
TEAL = "#0F766E"
PURPLE = "#7C3AED"
RED = "#DC2626"
SLATE = "#475569"
GRAY = "#94A3B8"
GRID = "#CBD5E1"
LIGHT = "#F8FAFC"
POSITIVE = "#E45756"
NEGATIVE = "#4C78A8"

FEATURE_NAMES = (
    "Cement",
    "Slag",
    "Fly ash",
    "Water",
    "Superplasticizer",
    "Coarse aggregate",
    "Fine aggregate",
    "Age",
)
MODEL_KEYS = (
    "clean_0",
    "clean_1",
    "clean_2",
    "shuffled_0",
    "shuffled_1",
    "shuffled_2",
    "ridge",
)
MODEL_LABELS = {
    "clean_0": "Clean MLP · seed 161410",
    "clean_1": "Clean MLP · seed 161411",
    "clean_2": "Clean MLP · seed 161412",
    "shuffled_0": "Shuffled MLP · seed 161410",
    "shuffled_1": "Shuffled MLP · seed 161411",
    "shuffled_2": "Shuffled MLP · seed 161412",
    "ridge": "Ridge control",
}
MODEL_COLORS = {
    "clean_0": BLUE,
    "clean_1": TEAL,
    "clean_2": PURPLE,
    "shuffled_0": "#F97316",
    "shuffled_1": "#DC2626",
    "shuffled_2": "#BE123C",
    "ridge": SLATE,
}
METHOD_COLORS = {"exact": SLATE, "deep": BLUE, "gradient": ORANGE}
METHOD_LABELS = {
    "exact": "Exact empirical SHAP",
    "deep": "DeepSHAP Rescale",
    "gradient": "Gradient × input difference",
}


def _array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _scalar(value):
    return float(_array(value).reshape(-1)[0])


def _new_figure(*args, **kwargs):
    kwargs.setdefault("layout", "constrained")
    fig, axes = plt.subplots(*args, **kwargs)
    fig.patch.set_facecolor("white")
    return fig, axes


def _style_axis(ax, grid_axis="y"):
    ax.set_facecolor("white")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.7, alpha=0.55)
        ax.set_axisbelow(True)
    ax.tick_params(colors=SLATE, labelsize=9)


def _finish(fig, output_dir, stem):
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            destination / f"{stem}.png",
            dpi=180,
            bbox_inches="tight",
            facecolor="white",
        )
    return fig


def _first(mapping, *keys):
    for key in keys:
        if key in mapping:
            return mapping[key]
    raise KeyError(f"Expected one of {keys!r}; got {tuple(mapping)}")


def _all_model_rows(prepared, evaluated, model_key, field):
    return np.concatenate(
        [
            _array(prepared["models"][model_key][field]),
            _array(evaluated["models"][model_key][field]),
        ],
        axis=0,
    )


def _all_targets(prepared, evaluated):
    explanation = _array(_first(prepared, "targets", "explain_targets"))
    audit = _array(_first(evaluated, "targets", "audit_targets"))
    return np.concatenate([explanation, audit])


def _model_keys(prepared, evaluated):
    return [
        key for key in MODEL_KEYS if key in prepared["models"] and key in evaluated["models"]
    ]


def plot_rescale_limits(diagnostic, output_dir=None):
    """Show the ReLU secant and the named derivative-fallback branches."""
    curve = diagnostic["curve"]
    actual = _array(curve["actual"])
    reference = _array(curve["reference"])
    slope = _array(curve["slope"])
    fig, axes = _new_figure(1, 2, figsize=(10.5, 4.0))
    fig.suptitle(
        "Rescale replaces a point gradient with a finite-difference slope",
        fontsize=15,
        fontweight="bold",
    )

    ax = axes[0]
    z = np.linspace(-1.4, 1.4, 200)
    ax.plot(z, np.maximum(z, 0), color=SLATE, lw=2.3, label="ReLU")
    ax.plot([-1, 1], [0, 1], color=BLUE, lw=2.5, label="Secant = 1/2")
    ax.scatter([-1, 1], [0, 1], s=65, color=[SLATE, BLUE], zorder=3)
    ax.axvline(0, color=GRID, lw=1)
    ax.set(xlabel="Preactivation z", ylabel="Activation")
    ax.set_title("Crossing the kink", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="upper left")
    _style_axis(ax)

    ax = axes[1]
    ax.plot(actual, slope, color=BLUE, lw=2.5)
    ax.axvline(0, color=GRID, lw=1)
    ax.axhline(0, color=GRID, lw=1)
    ax.scatter(
        _array(diagnostic["actual"])[[0]],
        _array(diagnostic["primary"])[[0]],
        color=ORANGE,
        edgecolor="white",
        s=70,
        zorder=3,
        label="Threshold crossing",
    )
    ax.set(
        xlabel="Actual preactivation z(x), reference fixed at −1",
        ylabel="Rescale multiplier",
    )
    ax.set_title("The multiplier follows the finite change", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="lower right")
    _style_axis(ax)
    return _finish(fig, output_dir, "01_rescale_limits")


def plot_near_zero(diagnostic, output_dir=None):
    """Make the intentional 1e-6 fallback completeness failure visible."""
    delta = _scalar(diagnostic["prediction_delta"])
    primary = _scalar(diagnostic["primary_phi"])
    exact = _scalar(diagnostic["exact_secant_phi"])
    residuals = [
        _scalar(diagnostic["primary_completeness_residual"]),
        _scalar(diagnostic["exact_secant_completeness_residual"]),
    ]
    fig, axes = _new_figure(1, 2, figsize=(10.5, 3.9))
    fig.suptitle(
        "The numerical fallback has a small, explicit boundary",
        fontsize=15,
        fontweight="bold",
    )
    labels = ["Output change", "ε=10⁻⁶ attribution", "ε=0 attribution"]
    axes[0].barh(labels, [delta, primary, exact], color=[SLATE, ORANGE, BLUE])
    axes[0].set_xlabel("Model-output units")
    axes[0].set_title("A ReLU crossing of only 8×10⁻⁷", loc="left", fontweight="bold")
    axes[0].ticklabel_format(axis="x", style="sci", scilimits=(-3, 3))
    _style_axis(axes[0], "x")

    axes[1].bar(
        ["ε=10⁻⁶ fallback", "ε=0 secant"],
        residuals,
        color=[ORANGE, BLUE],
        width=0.6,
    )
    axes[1].axhline(0, color=SLATE, lw=1)
    axes[1].set_ylabel("Completeness residual\nΔf − Σφ (model-output units)")
    axes[1].set_title("Expected failure; retained in the result", loc="left", fontweight="bold")
    axes[1].ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    _style_axis(axes[1])
    return _finish(fig, output_dir, "02_near_zero_epsilon")


def plot_affine_controls(diagnostic, output_dir=None):
    """Render exact affine, dummy-feature and constant-network controls."""
    fig, axes = _new_figure(1, 3, figsize=(12.6, 3.8))
    fig.suptitle(
        "Controls isolate what the modified backward pass must preserve",
        fontsize=15,
        fontweight="bold",
    )
    for ax, name, title in zip(
        axes,
        ("affine", "dummy", "constant"),
        ("Affine exactness", "Unused input", "Constant output"),
    ):
        block = diagnostic[name]
        phi = _array(block["phi"]).reshape(-1)
        expected = _array(block["expected_phi"]).reshape(-1)
        positions = np.arange(len(phi))
        ax.bar(positions - 0.18, expected, 0.36, color=SLATE, label="Independent oracle")
        ax.bar(positions + 0.18, phi, 0.36, color=BLUE, label="Student Rescale")
        ax.axhline(0, color=SLATE, lw=1)
        ax.set_xticks(positions, [f"x{index}" for index in positions])
        ax.set_ylabel("Contribution (model-output units)")
        ax.set_title(title, loc="left", fontweight="bold")
        _style_axis(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False)
    return _finish(fig, output_dir, "03_affine_controls")


def plot_background_average(diagnostic, output_dir=None):
    """Contrast averaging explanations with explaining an averaged input."""
    pair = _array(diagnostic["pair_phi"]).reshape(-1)
    phi = _scalar(diagnostic["phi"])
    mean_reference = _scalar(diagnostic["mean_reference_phi"])
    base = _scalar(diagnostic["base"])
    mean_base = _scalar(diagnostic["mean_reference_base"])
    fig, axes = _new_figure(1, 2, figsize=(10.8, 3.9))
    fig.suptitle(
        "DeepSHAP averages complete explanations, not background inputs",
        fontsize=15,
        fontweight="bold",
    )
    axes[0].bar(
        ["Reference −2", "Reference +1", "Mean attribution", "Mean reference"],
        [pair[0], pair[1], phi, mean_reference],
        color=[GRAY, GRAY, BLUE, ORANGE],
    )
    axes[0].set_ylabel("Contribution (model-output units)")
    axes[0].tick_params(axis="x", rotation=18)
    axes[0].set_title(
        "Nonlinearity makes the two procedures differ", loc="left", fontweight="bold"
    )
    _style_axis(axes[0])

    axes[1].bar(
        ["E[f(background)]", "f(mean background)"],
        [base, mean_base],
        color=[BLUE, ORANGE],
        width=0.58,
    )
    axes[1].set_ylabel("Base value (model-output units)")
    axes[1].set_title("The baseline moves as well", loc="left", fontweight="bold")
    _style_axis(axes[1])
    return _finish(fig, output_dir, "04_background_average")


def plot_threshold(diagnostic, output_dir=None):
    """Show the canonical complete-but-not-exact threshold attribution."""
    game = _array(diagnostic["game"])[0]
    deep = _array(diagnostic["deep"])[0]
    exact = _array(diagnostic["exact"])[0]
    gradient = _array(diagnostic["gradient"])[0]
    effects = _array(diagnostic["full_minus_feature"])
    fig, axes = _new_figure(1, 2, figsize=(11.0, 4.0))
    fig.suptitle(
        "Completeness does not imply exact feature credit",
        fontsize=15,
        fontweight="bold",
    )
    axes[0].bar(["∅", "{x₀}", "{x₁}", "{x₀,x₁}"], game, color=[GRAY, GRAY, GRAY, TEAL])
    axes[0].set_ylabel("Coalition value (model-output units)")
    axes[0].set_title(
        "Both features are necessary in the exact game", loc="left", fontweight="bold"
    )
    _style_axis(axes[0])

    positions = np.arange(2)
    width = 0.19
    series = (
        ("Exact SHAP", exact, SLATE),
        ("DeepSHAP", deep, BLUE),
        ("Gradient × Δx", gradient, ORANGE),
        ("Full − without feature", effects, TEAL),
    )
    for offset, (label, values, color) in enumerate(series):
        axes[1].bar(positions + (offset - 1.5) * width, values, width, color=color, label=label)
    axes[1].axhline(0, color=SLATE, lw=1)
    axes[1].set_xticks(positions, ["x₀", "x₁"])
    axes[1].set_ylabel("Contribution / effect (model-output units)")
    axes[1].set_title("Same total, different division", loc="left", fontweight="bold")
    axes[1].legend(frameon=False, fontsize=8, loc="upper left")
    _style_axis(axes[1])
    return _finish(fig, output_dir, "05_threshold_complete_not_exact")


def plot_cancellation(diagnostic, output_dir=None):
    """Display graph-dependent Rescale credit and bounded RevealCancel."""
    blocks = diagnostic["models"]
    fig, axes = _new_figure(1, 2, figsize=(11.2, 4.1))
    fig.suptitle(
        "Equivalent min circuits expose implementation-dependent Rescale credit",
        fontsize=15,
        fontweight="bold",
    )
    inputs = ["(0,0)", "(1,0)", "(0,1)", "(1,1)"]
    axes[0].plot(
        inputs,
        _array(blocks["x1 route"]["binary_prediction"]),
        marker="o",
        color=BLUE,
        lw=2,
        label="x₁ − ReLU(x₁−x₀)",
    )
    axes[0].plot(
        inputs,
        _array(blocks["x0 route"]["binary_prediction"]),
        marker="s",
        color=ORANGE,
        lw=1.5,
        linestyle="--",
        label="x₀ − ReLU(x₀−x₁)",
    )
    axes[0].set_ylabel("Output on binary input")
    axes[0].set_title(
        "The graphs agree on all four nonnegative binary inputs",
        loc="left",
        fontweight="bold",
    )
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    _style_axis(axes[0])

    positions = np.arange(2)
    width = 0.17
    series = (
        ("Rescale · x₁ route", _array(blocks["x1 route"]["deep"])[0], BLUE),
        ("Rescale · x₀ route", _array(blocks["x0 route"]["deep"])[0], ORANGE),
        ("Exact feature SHAP", _array(blocks["x1 route"]["exact"])[0], SLATE),
        ("RevealCancel groups", _array(blocks["x1 route"]["reveal_cancel"])[0], TEAL),
    )
    for offset, (label, values, color) in enumerate(series):
        axes[1].bar(positions + (offset - 1.5) * width, values, width, color=color, label=label)
    axes[1].set_xticks(positions, ["x₀", "x₁"])
    axes[1].set_ylabel("Contribution (model-output units)")
    axes[1].set_title("At x=(1,1), reference=(0,0)", loc="left", fontweight="bold")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="outside lower center", ncol=2)
    _style_axis(axes[1])
    return _finish(fig, output_dir, "06_cancellation_implementation_dependence")


def plot_replacement_control(diagnostic, output_dir=None):
    """Verify endpoints and the flat dummy-feature step on an additive game."""
    result = diagnostic["result"]
    insertion = _array(result["insertion"])[0]
    deletion = _array(result["deletion"])[0]
    expected_insertion = _array(diagnostic["expected_insertion"])[0]
    expected_deletion = _array(diagnostic["expected_deletion"])[0]
    steps = np.arange(len(insertion))
    fig, axes = _new_figure(1, 2, figsize=(10.7, 3.8), sharex=True)
    fig.suptitle(
        "Replacement curves keep endpoints and dummy steps testable",
        fontsize=15,
        fontweight="bold",
    )
    for ax, values, expected, title in (
        (axes[0], insertion, expected_insertion, "Insertion from the empty coalition"),
        (axes[1], deletion, expected_deletion, "Deletion from the full coalition"),
    ):
        ax.plot(steps, expected, color=SLATE, lw=3.5, alpha=0.35, label="Independent oracle")
        ax.plot(steps, values, color=BLUE, marker="o", lw=2, label="Student curve")
        ax.axvspan(1, 2, color="#FEF3C7", alpha=0.7)
        ax.text(
            1.5,
            min(values.min(), expected.min()),
            "dummy step",
            ha="center",
            va="bottom",
            fontsize=8,
        )
        ax.set(xlabel="Features replaced", ylabel="Game value (model-output units)")
        ax.set_title(title, loc="left", fontweight="bold")
        _style_axis(ax)
    axes[0].legend(frameon=False, fontsize=8)
    return _finish(fig, output_dir, "07_replacement_fixture")


def plot_performance(prepared, evaluated, output_dir=None):
    """Compare clean, shuffled-target, ridge and training-mean prediction errors."""
    keys = _model_keys(prepared, evaluated)
    explain_targets = _array(_first(prepared, "targets", "explain_targets")).reshape(-1)
    audit_targets = _array(_first(evaluated, "targets", "audit_targets")).reshape(-1)
    explain_mse = []
    audit_mse = []
    for key in keys:
        explain_prediction = _array(prepared["models"][key]["prediction"]).reshape(-1)
        audit_prediction = _array(evaluated["models"][key]["prediction"]).reshape(-1)
        explain_mse.append(float(np.mean((explain_prediction - explain_targets) ** 2)))
        audit_mse.append(float(np.mean((audit_prediction - audit_targets) ** 2)))
    training_mean = evaluated["performance"]["training_mean"]
    baseline_mse = _scalar(
        training_mean["mse"] if isinstance(training_mean, dict) else training_mean
    )

    fig, axes = _new_figure(1, 2, figsize=(12.2, 4.4))
    fig.suptitle(
        "Prediction quality separates trained models from shuffled-target controls",
        fontsize=15,
        fontweight="bold",
    )
    positions = np.arange(len(keys))
    width = 0.36
    axes[0].bar(
        positions - width / 2, explain_mse, width, color=TEAL, label="Explanation split"
    )
    axes[0].bar(positions + width / 2, audit_mse, width, color=BLUE, label="Audit split")
    axes[0].axhline(baseline_mse, color=SLATE, ls="--", lw=1.5, label="Training-mean audit")
    short_labels = {
        "clean_0": "Clean\n0",
        "clean_1": "Clean\n1",
        "clean_2": "Clean\n2",
        "shuffled_0": "Shuffled\n0",
        "shuffled_1": "Shuffled\n1",
        "shuffled_2": "Shuffled\n2",
        "ridge": "Ridge\ncontrol",
    }
    axes[0].set_xticks(positions, [short_labels[key] for key in keys])
    axes[0].tick_params(axis="x", labelsize=7)
    axes[0].set_ylabel("Mean squared error (MPa²)")
    axes[0].set_title(
        "Every predeclared model and held-out split", loc="left", fontweight="bold"
    )
    axes[0].legend(frameon=False, fontsize=8)
    _style_axis(axes[0])

    groups = {
        "Clean MLP": [audit_mse[keys.index(key)] for key in keys if key.startswith("clean")],
        "Shuffled target": [
            audit_mse[keys.index(key)] for key in keys if key.startswith("shuffled")
        ],
        "Ridge": [audit_mse[keys.index("ridge")]] if "ridge" in keys else [],
        "Training mean": [baseline_mse],
    }
    colors = [BLUE, ORANGE, SLATE, GRAY]
    for index, ((label, values), color) in enumerate(zip(groups.items(), colors)):
        axes[1].scatter(
            np.full(len(values), index), values, s=65, color=color, edgecolor="white", zorder=3
        )
        if values:
            axes[1].plot([index - 0.18, index + 0.18], [np.mean(values)] * 2, color=color, lw=3)
    axes[1].set_xticks(range(len(groups)), list(groups), rotation=15)
    axes[1].set_ylabel("Audit mean squared error (MPa²)")
    axes[1].set_title(
        "Seeds are repeated trials, not uncertainty bars", loc="left", fontweight="bold"
    )
    _style_axis(axes[1])
    return _finish(fig, output_dir, "08_performance_controls")


def plot_signed_parity(prepared, evaluated, output_dir=None):
    """Plot all 15,232 signed exact/DeepSHAP coordinates in seven panels."""
    keys = _model_keys(prepared, evaluated)
    pairs = {
        key: (
            _all_model_rows(prepared, evaluated, key, "phi_exact").reshape(-1),
            _all_model_rows(prepared, evaluated, key, "phi_deep").reshape(-1),
        )
        for key in keys
    }
    bound = max(np.max(np.abs(np.concatenate(pair))) for pair in pairs.values())
    bound = max(bound, 1e-9) * 1.04
    fig, axes = _new_figure(3, 3, figsize=(11.8, 10.6), sharex=True, sharey=True)
    fig.suptitle(
        "DeepSHAP versus exact empirical SHAP · all 7 models × 272 rows × 8 features",
        fontsize=15,
        fontweight="bold",
    )
    flat_axes = axes.ravel()
    for ax, key in zip(flat_axes, keys):
        exact, deep = pairs[key]
        ax.scatter(exact, deep, s=7, alpha=0.25, color=MODEL_COLORS[key], rasterized=True)
        ax.plot([-bound, bound], [-bound, bound], color=SLATE, ls="--", lw=1)
        mae = np.mean(np.abs(deep - exact))
        ax.text(
            0.04,
            0.94,
            f"mean |gap| = {mae:.3g} MPa",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
            color=SLATE,
        )
        ax.set_title(MODEL_LABELS[key], loc="left", fontsize=10, fontweight="bold")
        ax.set_xlim(-bound, bound)
        ax.set_ylim(-bound, bound)
        _style_axis(ax, None)
    for ax in flat_axes[len(keys) :]:
        ax.axis("off")
    for ax in axes[-1, :]:
        if ax.axison:
            ax.set_xlabel("Exact contribution (MPa)")
    for ax in axes[:, 0]:
        ax.set_ylabel("DeepSHAP contribution (MPa)")
    return _finish(fig, output_dir, "09_signed_parity_all_rows")


def plot_error_heatmaps(prepared, evaluated, output_dir=None):
    """Show every row-feature error with a separately labeled MPa scale per model."""
    keys = _model_keys(prepared, evaluated)
    errors = {
        key: _all_model_rows(prepared, evaluated, key, "phi_deep")
        - _all_model_rows(prepared, evaluated, key, "phi_exact")
        for key in keys
    }
    fig = plt.figure(figsize=(13.2, 12.2), layout="constrained")
    fig.patch.set_facecolor("white")
    grid = fig.add_gridspec(len(keys), 2, width_ratios=(1.0, 0.025), hspace=0.18)
    axes = [fig.add_subplot(grid[row, 0]) for row in range(len(keys))]
    color_axes = [fig.add_subplot(grid[row, 1]) for row in range(len(keys))]
    fig.suptitle(
        "DeepSHAP − exact empirical SHAP · all 15,232 coordinates · per-model MPa scales",
        fontsize=15,
        fontweight="bold",
    )
    for index, (ax, color_axis, key) in enumerate(zip(axes, color_axes, keys)):
        limit = max(float(np.max(np.abs(errors[key]))), np.finfo(float).eps)
        norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
        image = ax.imshow(
            errors[key].T,
            aspect="auto",
            interpolation="nearest",
            cmap="RdBu_r",
            norm=norm,
            rasterized=True,
        )
        ax.axvline(152.5, color="black", lw=0.9, ls="--")
        control_note = "   ·   numerical zero" if key == "ridge" else ""
        ax.set_title(
            f"{MODEL_LABELS[key]}{control_note}   ·   "
            f"max |gap|={np.max(np.abs(errors[key])):.3g} MPa",
            loc="left",
            fontsize=9,
            fontweight="bold",
        )
        ax.set_yticks(range(8), FEATURE_NAMES if index == len(keys) // 2 else [""] * 8)
        ax.tick_params(axis="y", labelsize=8)
        if index == len(keys) - 1:
            ax.set_xticks([0, 152, 271], ["0", "152", "271"])
            ax.set_xlabel("Held-out row (explanation 0–152 | audit 153–271)")
        else:
            ax.set_xticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        colorbar = fig.colorbar(image, cax=color_axis)
        colorbar.set_ticks([-limit, 0.0, limit])
        colorbar.set_ticklabels([f"{-limit:.2g}", "0", f"{limit:.2g}"])
        colorbar.ax.tick_params(labelsize=7)
        colorbar.set_label("MPa", fontsize=7, labelpad=2)
    return _finish(fig, output_dir, "10_error_heatmaps_all_rows")


def _waterfall_axis(ax, base, prediction, values, order, feature_names, title):
    running = base
    for row, feature in enumerate(order):
        value = values[feature]
        updated = running + value
        ax.barh(
            row,
            abs(value),
            left=min(running, updated),
            color=POSITIVE if value >= 0 else NEGATIVE,
            height=0.62,
        )
        ax.text(
            updated,
            row,
            f" {value:+.2f}",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=8,
            color=SLATE,
        )
        running = updated
    ax.axvline(base, color=SLATE, ls="--", lw=1.2)
    ax.axvline(prediction, color=TEAL, lw=1.6)
    ax.set_yticks(range(len(order)), [feature_names[index] for index in order])
    ax.invert_yaxis()
    ax.set_xlabel("Predicted compressive strength (MPa)")
    ax.set_title(title, loc="left", fontweight="bold")
    _style_axis(ax, "x")


def plot_fixed_waterfall(native_play, output_dir=None, feature_names=FEATURE_NAMES):
    """Compare exact and DeepSHAP waterfalls for frozen source row 1."""
    exact = _array(native_play["exact"]).reshape(-1)
    deep = _array(native_play["deep"]).reshape(-1)
    base = _scalar(native_play["base"])
    prediction = _scalar(_first(native_play, "prediction", "pred"))
    order = np.argsort(-np.maximum(np.abs(exact), np.abs(deep)))
    source = _scalar(_first(native_play, "source_id", "source"))
    fig, axes = _new_figure(1, 2, figsize=(12.0, 5.3), sharex=True)
    fig.suptitle(
        f"Frozen held-out recipe · source {source:.0f} · "
        f"base {base:.2f} → prediction {prediction:.2f} MPa",
        fontsize=14,
        fontweight="bold",
    )
    _waterfall_axis(
        axes[0], base, prediction, exact, order, feature_names, "Exact empirical SHAP"
    )
    _waterfall_axis(axes[1], base, prediction, deep, order, feature_names, "DeepSHAP Rescale")
    legend = [
        Line2D([0], [0], color=POSITIVE, lw=7, label="Positive contribution"),
        Line2D([0], [0], color=NEGATIVE, lw=7, label="Negative contribution"),
        Line2D([0], [0], color=TEAL, lw=2, label="Prediction"),
    ]
    fig.legend(handles=legend, loc="outside lower center", ncol=3, frameon=False)
    return _finish(fig, output_dir, "11_fixed_waterfall")


def plot_fixed_attribution_comparison(
    native_play, output_dir=None, feature_names=FEATURE_NAMES
):
    """Show donor-specific Rescale spread and exact/Deep/gradient point estimates."""
    pair = _array(_first(native_play, "pair", "pair_deep"))
    if pair.ndim == 3:
        pair = pair[0]
    exact = _array(native_play["exact"]).reshape(-1)
    deep = _array(native_play["deep"]).reshape(-1)
    gradient = _array(native_play["gradient"]).reshape(-1)
    order = np.argsort(-np.maximum.reduce([np.abs(exact), np.abs(deep), np.abs(gradient)]))
    pair = pair[:, order]
    positions = np.arange(len(order))
    fig, ax = _new_figure(figsize=(11.0, 5.4))
    violin = ax.violinplot(
        pair, positions=positions, vert=False, widths=0.72, showextrema=False
    )
    for body in violin["bodies"]:
        body.set_facecolor("#BFDBFE")
        body.set_edgecolor(BLUE)
        body.set_alpha(0.6)
    ax.scatter(
        exact[order], positions, color=SLATE, marker="D", s=42, label="Exact SHAP", zorder=3
    )
    ax.scatter(
        deep[order], positions, color=BLUE, marker="o", s=42, label="DeepSHAP mean", zorder=3
    )
    ax.scatter(
        gradient[order],
        positions,
        color=ORANGE,
        marker="^",
        s=50,
        label="Gradient × Δx",
        zorder=3,
    )
    ax.axvline(0, color=SLATE, lw=1)
    ax.set_yticks(positions, [feature_names[index] for index in order])
    ax.invert_yaxis()
    ax.set_xlabel("Contribution (MPa)")
    ax.set_title(
        "Each background donor gives a Rescale explanation; DeepSHAP averages them",
        loc="left",
        fontsize=13,
        fontweight="bold",
    )
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="outside lower center", ncol=3)
    _style_axis(ax, "x")
    return _finish(fig, output_dir, "12_fixed_attribution_comparison")


def plot_completeness_vs_exact(prepared, evaluated, output_dir=None):
    """Separate local-accuracy residuals from exact feature-credit error."""
    keys = _model_keys(prepared, evaluated)
    fig, ax = _new_figure(figsize=(9.0, 6.2))
    for key in keys:
        prediction = _all_model_rows(prepared, evaluated, key, "prediction").reshape(-1)
        deep = _all_model_rows(prepared, evaluated, key, "phi_deep")
        exact = _all_model_rows(prepared, evaluated, key, "phi_exact")
        prepared_count = len(prepared["models"][key]["prediction"])
        base = np.concatenate(
            [
                np.full(prepared_count, _scalar(prepared["models"][key]["base"])),
                np.full(
                    len(evaluated["models"][key]["prediction"]),
                    _scalar(evaluated["models"][key]["base"]),
                ),
            ]
        )
        completeness = np.abs(prediction - base - deep.sum(axis=1))
        exact_gap = np.mean(np.abs(deep - exact), axis=1)
        ax.scatter(
            np.maximum(completeness, 1e-16),
            np.maximum(exact_gap, 1e-16),
            s=12,
            alpha=0.42,
            color=MODEL_COLORS[key],
            label=MODEL_LABELS[key],
            rasterized=True,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("DeepSHAP completeness residual |prediction − base − Σφ| (MPa)")
    ax.set_ylabel("Mean |DeepSHAP − exact SHAP| per row (MPa)")
    ax.set_title(
        "A complete explanation can still assign different feature credit",
        loc="left",
        fontsize=14,
        fontweight="bold",
    )
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, loc="outside lower center", ncol=2)
    _style_axis(ax, "both")
    return _finish(fig, output_dir, "13_completeness_vs_exact")


def plot_background_sensitivity(evaluated, output_dir=None):
    """Contrast exact-game movement with DeepSHAP approximation gap."""
    variants = evaluated["background_variants"]
    preferred = ("primary64", "nested16", "nested128", "independent64_a", "independent64_b")
    keys = [key for key in preferred if key in variants]
    primary = _array(variants["primary64"]["phi_exact"])
    labels = {
        "primary64": "Primary 64",
        "nested16": "Nested 16",
        "nested128": "Nested 128",
        "independent64_a": "Independent 64 A",
        "independent64_b": "Independent 64 B",
    }
    colors = [SLATE, TEAL, BLUE, ORANGE, RED]
    fig, axes = _new_figure(1, 2, figsize=(12.0, 4.8))
    fig.suptitle(
        "Changing a finite background moves the exact game and the approximation gap",
        fontsize=15,
        fontweight="bold",
    )
    movement_data = []
    gap_data = []
    for key, color in zip(keys, colors):
        exact = _array(variants[key]["phi_exact"])
        deep = _array(variants[key]["phi_deep"])
        movement = np.mean(np.abs(exact - primary), axis=1)
        gap = np.mean(np.abs(deep - exact), axis=1)
        movement_data.append(movement)
        gap_data.append(gap)
        axes[0].scatter(
            movement,
            gap,
            s=10,
            alpha=0.28,
            color=color,
            label=labels[key],
            rasterized=True,
        )
        axes[0].scatter(
            np.median(movement), np.median(gap), s=80, color=color, edgecolor="white"
        )
    axes[0].set_xlabel("Mean |exact variant − exact primary64| per row (MPa)")
    axes[0].set_ylabel("Mean |DeepSHAP − exact variant| per row (MPa)")
    axes[0].set_title("Every point is one held-out row", loc="left", fontweight="bold")
    variant_handles, variant_labels = axes[0].get_legend_handles_labels()
    _style_axis(axes[0], "both")

    positions = np.arange(len(keys))
    width = 0.34
    axes[1].boxplot(
        movement_data,
        positions=positions - width / 2,
        widths=0.27,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(facecolor="#CCFBF1", edgecolor=TEAL),
        medianprops=dict(color=TEAL),
    )
    axes[1].boxplot(
        gap_data,
        positions=positions + width / 2,
        widths=0.27,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(facecolor="#DBEAFE", edgecolor=BLUE),
        medianprops=dict(color=BLUE),
    )
    axes[1].set_xticks(positions, [labels[key].replace(" ", "\n") for key in keys])
    axes[1].set_ylabel("Mean absolute feature difference (MPa)")
    axes[1].set_title(
        "Background movement and approximation are separate", loc="left", fontweight="bold"
    )
    metric_handles = [
        Line2D([0], [0], color=TEAL, lw=7, label="Exact-game movement"),
        Line2D([0], [0], color=BLUE, lw=7, label="Approximation gap"),
    ]
    fig.legend(
        handles=variant_handles + metric_handles,
        labels=variant_labels + [item.get_label() for item in metric_handles],
        frameon=False,
        fontsize=8,
        loc="outside lower center",
        ncol=4,
    )
    _style_axis(axes[1])
    return _finish(fig, output_dir, "14_background_sensitivity")


def plot_replacement_curves(evaluated, output_dir=None, model_key="clean_0"):
    """Compare frozen global attribution orders with 100 random orders."""
    block = evaluated["replacement_curves"][model_key]
    exact_values = _array(block["exact"]["insertion"])
    sign = np.where(exact_values[:, -1] - exact_values[:, 0] >= 0, 1.0, -1.0)[:, None]
    random_block = block["random"]
    random_insertion = _array(random_block["insertion"])
    random_deletion = _array(random_block["deletion"])
    steps = np.arange(exact_values.shape[1])
    fig, axes = _new_figure(1, 2, figsize=(11.8, 4.4), sharex=True)
    fig.suptitle(
        f"Frozen global feature orders on all audit rows · {MODEL_LABELS[model_key]}",
        fontsize=14,
        fontweight="bold",
    )
    for ax, curve_name, random_values, ylabel in (
        (
            axes[0],
            "insertion",
            sign[None] * (random_insertion - random_insertion[:, :, [0]]),
            "Oriented prediction gain from empty (MPa)",
        ),
        (
            axes[1],
            "deletion",
            sign[None] * (random_deletion[:, :, [0]] - random_deletion),
            "Oriented prediction drop from full (MPa)",
        ),
    ):
        random_means = random_values.mean(axis=1)
        lower, upper = np.quantile(random_means, [0.05, 0.95], axis=0)
        ax.fill_between(
            steps, lower, upper, color=GRAY, alpha=0.25, label="100 random orders · 90% band"
        )
        ax.plot(steps, random_means.mean(axis=0), color=GRAY, ls="--", lw=1.5)
        for method in ("exact", "deep", "gradient"):
            values = _array(block[method][curve_name])
            oriented = (
                sign * (values - values[:, [0]])
                if curve_name == "insertion"
                else sign * (values[:, [0]] - values)
            )
            ax.plot(
                steps,
                oriented.mean(axis=0),
                color=METHOD_COLORS[method],
                marker="o",
                ms=3.5,
                lw=2,
                label=METHOD_LABELS[method],
            )
        ax.axhline(0, color=SLATE, lw=1)
        ax.set_xlabel("Features replaced")
        ax.set_ylabel(ylabel)
        ax.set_title(curve_name.capitalize(), loc="left", fontweight="bold")
        _style_axis(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False)
    return _finish(fig, output_dir, "15_replacement_curves")


def plot_threshold_play(play, output_dir=None):
    """Render the learner's editable threshold-gate experiment."""
    deep = _array(play["deep"]).reshape(-1)
    exact = _array(play["exact"]).reshape(-1)
    gradient = _array(play["gradient"]).reshape(-1)
    features = [f"x{index}" for index in range(len(deep))]
    positions = np.arange(len(deep))
    width = 0.24
    fig, axes = _new_figure(1, 2, figsize=(10.8, 4.0))
    fig.suptitle("Try it: move the gate, query, or reference", fontsize=15, fontweight="bold")
    for offset, (label, values, color) in enumerate(
        (
            ("Exact SHAP", exact, SLATE),
            ("DeepSHAP", deep, BLUE),
            ("Gradient × Δx", gradient, ORANGE),
        )
    ):
        axes[0].bar(positions + (offset - 1) * width, values, width, color=color, label=label)
    axes[0].set_xticks(positions, features)
    axes[0].set_ylabel("Contribution (model-output units)")
    axes[0].set_title("Feature credit", loc="left", fontweight="bold")
    axes[0].legend(frameon=False, fontsize=8)
    _style_axis(axes[0])

    residual = abs(_scalar(play["deep_completeness_residual"]))
    gap = _scalar(play["deep_exact_max_abs_error"])
    axes[1].bar(
        ["Completeness\nresidual", "Max exact\ncredit gap"],
        [residual, gap],
        color=[TEAL, PURPLE],
        width=0.58,
    )
    axes[1].set_ylabel("Absolute error (model-output units)")
    axes[1].set_title("Two questions, two diagnostics", loc="left", fontweight="bold")
    _style_axis(axes[1])
    return _finish(fig, output_dir, "16_threshold_play")


def plot_real_recipe_play(native_play, output_dir=None, feature_names=FEATURE_NAMES):
    """Render one editable recipe's attributions and local replacement curves."""
    exact = _array(native_play["exact"]).reshape(-1)
    deep = _array(native_play["deep"]).reshape(-1)
    gradient = _array(native_play["gradient"]).reshape(-1)
    order = np.argsort(-np.maximum.reduce([np.abs(exact), np.abs(deep), np.abs(gradient)]))
    positions = np.arange(len(order))
    fig, axes = _new_figure(1, 2, figsize=(12.4, 5.1))
    source = _scalar(_first(native_play, "source_id", "source"))
    model_key = str(_first(native_play, "model_key", "model"))
    fig.suptitle(
        f"Try it: source {source:.0f} · {MODEL_LABELS.get(model_key, model_key)}",
        fontsize=14,
        fontweight="bold",
    )
    width = 0.23
    for offset, (method, values) in enumerate(
        (("exact", exact), ("deep", deep), ("gradient", gradient))
    ):
        axes[0].barh(
            positions + (offset - 1) * width,
            values[order],
            width,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
    axes[0].axvline(0, color=SLATE, lw=1)
    axes[0].set_yticks(positions, [feature_names[index] for index in order])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Contribution (MPa)")
    axes[0].set_title("Attribution", loc="left", fontweight="bold")
    _style_axis(axes[0], "x")

    curves = native_play["curves"]
    steps = np.arange(9)
    for method in ("exact", "deep", "gradient"):
        block = curves[method]
        insertion = _array(block["insertion"]).reshape(-1)
        deletion = _array(block["deletion"]).reshape(-1)
        axes[1].plot(
            steps,
            insertion,
            color=METHOD_COLORS[method],
            lw=2,
            marker="o",
            ms=3,
            label=f"{METHOD_LABELS[method]} · insert",
        )
        axes[1].plot(
            steps,
            deletion,
            color=METHOD_COLORS[method],
            lw=1.5,
            ls="--",
            marker="s",
            ms=3,
            label=f"{METHOD_LABELS[method]} · delete",
        )
    axes[1].set_xlabel("Features replaced using the local signed order")
    axes[1].set_ylabel("Expected prediction (MPa)")
    axes[1].set_title("Local insertion and deletion", loc="left", fontweight="bold")
    _style_axis(axes[1])
    legend_handles = [
        Line2D([0], [0], color=METHOD_COLORS[method], lw=3, label=METHOD_LABELS[method])
        for method in ("exact", "deep", "gradient")
    ]
    legend_handles.extend(
        [
            Line2D([0], [0], color=SLATE, lw=2, marker="o", label="Insertion"),
            Line2D([0], [0], color=SLATE, lw=2, ls="--", marker="s", label="Deletion"),
        ]
    )
    fig.legend(
        handles=legend_handles,
        loc="outside lower center",
        ncol=5,
        frameon=False,
        fontsize=8,
    )
    return _finish(fig, output_dir, "17_real_recipe_play")


def save_figure_set(diagnostics, prepared, evaluated, native_play, output_dir):
    """Write the complete fixed figure set for the native/reproduction harness."""
    mechanism = diagnostics.get("mechanism", diagnostics)
    calls = (
        (plot_rescale_limits, (mechanism["rescale_limits"],)),
        (plot_near_zero, (mechanism["near_zero"],)),
        (plot_affine_controls, (mechanism["affine_controls"],)),
        (plot_background_average, (mechanism["background_average"],)),
        (plot_threshold, (mechanism["threshold"],)),
        (plot_cancellation, (mechanism["cancellation"],)),
        (plot_replacement_control, (mechanism["replacement_control"],)),
        (plot_performance, (prepared, evaluated)),
        (plot_signed_parity, (prepared, evaluated)),
        (plot_error_heatmaps, (prepared, evaluated)),
        (plot_fixed_waterfall, (native_play,)),
        (plot_fixed_attribution_comparison, (native_play,)),
        (plot_completeness_vs_exact, (prepared, evaluated)),
        (plot_background_sensitivity, (evaluated,)),
        (plot_replacement_curves, (evaluated,)),
        (plot_threshold_play, (mechanism["threshold_play"],)),
        (plot_real_recipe_play, (native_play,)),
    )
    figures = [plotter(*args, output_dir=output_dir) for plotter, args in calls]
    for figure in figures:
        plt.close(figure)
    return sorted(str(path) for path in Path(output_dir).glob("[0-9][0-9]_*.png"))
