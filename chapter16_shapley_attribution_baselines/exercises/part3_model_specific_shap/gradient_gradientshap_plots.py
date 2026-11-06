"""Pure Matplotlib rendering for the Expected Gradients lesson."""

from __future__ import annotations

from textwrap import fill

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np


BLUE = "#2563eb"
ORANGE = "#ea580c"
TEAL = "#0f766e"
SLATE = "#475569"
PURPLE = "#7c3aed"
LIGHT = "#cbd5e1"
RED = "#dc2626"
GRAY = "#94a3b8"
POSITIVE = "#e45756"
NEGATIVE = "#4c78a8"

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
    "shuffled_0": "#f97316",
    "shuffled_1": RED,
    "shuffled_2": "#be123c",
    "ridge": SLATE,
}
METHOD_KEYS = ("exact", "deep", "gradient", "eg_k2048", "quadrature_512")
METHOD_LABELS = {
    "exact": "Exact coalition SHAP",
    "deep": "DeepSHAP",
    "gradient": "Endpoint gradient × Δx",
    "eg_k2048": "Expected Gradients · K=2048",
    "quadrature_512": "512-node path control",
}
METHOD_COLORS = {
    "exact": SLATE,
    "deep": PURPLE,
    "gradient": ORANGE,
    "eg_k2048": BLUE,
    "quadrature_512": TEAL,
}
NATIVE_FIGURE_SPECS = (
    ("plot_native_overview", "14_native_overview"),
    ("plot_allrow_method_map", "15_allrow_method_map"),
    ("plot_fixed_waterfall", "16_fixed_waterfall"),
    ("plot_background_and_noise", "17_background_and_noise"),
    ("plot_replicates_and_distance", "18_replicates_and_distance"),
    ("plot_replacement_curves", "19_replacement_curves"),
    ("plot_real_recipe_play", "20_real_recipe_play"),
)


def _array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _scalar(value):
    array = _array(value)
    if array.size != 1:
        raise ValueError(f"Expected one scalar value, got shape {array.shape}.")
    return float(array.reshape(-1)[0])


def _model_keys(payload):
    records = payload["all_rows"] if "all_rows" in payload else payload["primary"]
    keys = [key for key in MODEL_KEYS if key in records]
    if not keys:
        raise ValueError("The native payload contains no declared model records.")
    return keys


def _metric(record, *names):
    if not isinstance(record, dict):
        return _scalar(record)
    for name in names:
        if name in record:
            return _scalar(record[name])
    raise KeyError(f"Expected one of {names!r}; got {tuple(record)}.")


def _selection_value(play, name, default=None):
    selection = play.get("selection", {})
    if name in selection:
        value = selection[name]
    elif name in play:
        value = play[name]
    else:
        return default
    array = _array(value)
    return array.reshape(-1)[0].item() if array.size == 1 else value


def _shape_only_prefix(*payloads):
    if any(isinstance(payload, dict) and payload.get("_shape_only_fixture") for payload in payloads):
        return "SHAPE-ONLY ADAPTER FIXTURE · NOT SCIENTIFIC DATA\n"
    return ""


def _interval(values, axis=0):
    values = _array(values)
    return (
        np.median(values, axis=axis),
        np.quantile(values, 0.10, axis=axis),
        np.quantile(values, 0.90, axis=axis),
    )


def _interval_errorbar(axis, x, values, *, label, color, marker="o", offset=0.0):
    median, lower, upper = _interval(values, axis=0)
    median = np.atleast_1d(median)
    lower = np.atleast_1d(lower)
    upper = np.atleast_1d(upper)
    positions = np.asarray(x, dtype=float) + offset
    axis.errorbar(
        positions,
        median,
        yerr=np.vstack((median - lower, upper - median)),
        color=color,
        marker=marker,
        capsize=2.5,
        lw=1.6,
        label=label,
    )


def _grid(axis):
    axis.grid(axis="y", color="#dbe4ee", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)


def plot_schedule_prefix(result):
    indices = _array(result["indices_8"])
    alpha = _array(result["alpha_32"])
    source_ids = _array(result["source_ids"])
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.1), constrained_layout=True)
    image = axes[0].imshow(indices, aspect="auto", cmap="viridis", vmin=0, vmax=4)
    for row in range(indices.shape[0]):
        for column in range(indices.shape[1]):
            axes[0].text(column, row, str(indices[row, column]), ha="center", va="center")
    axes[0].set(title="Literal K=8 donor prefix", xlabel="Draw", ylabel="Source ID")
    axes[0].set_yticks(range(len(source_ids)), source_ids)
    fig.colorbar(image, ax=axes[0], label="Background-row index", shrink=0.82)

    for row, source_id in enumerate(source_ids):
        axes[1].plot(alpha[row], marker="o", ms=3, lw=1, label=f"source {source_id}")
    axes[1].axvspan(-0.5, 7.5, color="#dbeafe", alpha=0.65, label="K=8 prefix")
    axes[1].axhline(0.5, color=SLATE, ls="--", lw=1)
    axes[1].set(xlabel="Draw", ylabel=r"Path position $\alpha$", ylim=(-0.03, 1.03))
    axes[1].set_title("Each source owns a stable alpha stream")
    axes[1].legend(frameon=False, ncol=2, fontsize=8)
    _grid(axes[1])
    fig.suptitle("Row-stable schedules make nested budgets literal prefixes", weight="bold")
    return fig


def plot_path_interpolation(result):
    query = _array(result["query"])[0]
    background = _array(result["background"])
    points = _array(result["points"])[0]
    indices = _array(result["indices"])[0]
    alpha = _array(result["alpha"])[0]
    deltas = _array(result["deltas"])[0]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), constrained_layout=True)
    for donor_index, donor in enumerate(background):
        axes[0].plot([donor[0], query[0]], [donor[1], query[1]], color=LIGHT, lw=3, zorder=0)
        axes[0].scatter(*donor, color=SLATE, marker="s", s=60)
        axes[0].text(donor[0], donor[1], f"  donor {donor_index}", va="center")
    scatter = axes[0].scatter(points[:, 0], points[:, 1], c=alpha, cmap="viridis", s=65)
    axes[0].scatter(*query, color=ORANGE, marker="*", s=160, label="Query")
    axes[0].set(
        xlabel="Feature 0", ylabel="Feature 1", title="Sampled points stay on donor-query lines"
    )
    axes[0].legend(frameon=False)
    fig.colorbar(scatter, ax=axes[0], label=r"$\alpha$", shrink=0.82)

    width = 0.36
    draws = np.arange(len(indices))
    axes[1].bar(draws - width / 2, deltas[:, 0], width, color=BLUE, label="Feature 0")
    axes[1].bar(draws + width / 2, deltas[:, 1], width, color=TEAL, label="Feature 1")
    axes[1].set_xticks(draws, [f"d{i}\nb{index}" for i, index in enumerate(indices)])
    axes[1].set(
        xlabel="Draw / donor",
        ylabel="Query minus donor",
        title="The displacement keeps the same donor",
    )
    axes[1].legend(frameon=False)
    _grid(axes[1])
    fig.suptitle("Interpolation exposes every path point and displacement", weight="bold")
    return fig


def plot_gradient_controls(result):
    affine = _array(result["affine_gradients"])[0]
    expected = _array(result["affine_expected"])
    z = _array(result["quadratic_points"])[0, :, 0]
    gradient = _array(result["quadratic_gradients"])[0, :, 0]
    quadratic_expected = _array(result["quadratic_expected"])[0, :, 0]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1), constrained_layout=True)
    features = np.arange(2)
    axes[0].bar(features - 0.18, expected, 0.36, color=SLATE, label="Analytic")
    axes[0].bar(features + 0.18, affine.mean(axis=0), 0.36, color=BLUE, label="Student")
    axes[0].set_xticks(features, [r"$x_0$", r"$x_1$"])
    axes[0].set(ylabel="Input gradient", title="Affine gradient is constant across rows")
    axes[0].legend(frameon=False)
    _grid(axes[0])
    axes[1].plot(z, quadratic_expected, color=SLATE, lw=2, label=r"Analytic $2z$")
    axes[1].scatter(z, gradient, color=ORANGE, s=24, label="Student samples")
    axes[1].set(
        xlabel="Path point z",
        ylabel="Input gradient",
        title="Quadratic gradient changes along the path",
    )
    axes[1].legend(frameon=False)
    _grid(axes[1])
    fig.suptitle("Input gradients match two independent analytic controls", weight="bold")
    return fig


def plot_relu_samples(result):
    alpha = _array(result["alpha"])[0]
    samples = _array(result["relu_samples"])[0, :, 0]
    q_alpha = _array(result["quadratic_alpha"])[0]
    q_samples = _array(result["quadratic_samples"])[0, :, 0]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0), constrained_layout=True)
    axes[0].scatter(alpha, samples, c=np.where(samples > 0, ORANGE, BLUE), s=28, alpha=0.8)
    axes[0].plot([0, 0.5, 0.5, 1], [0, 0, 2, 2], color=SLATE, lw=2, label="Analytic sample")
    axes[0].set(
        xlabel=r"Path position $\alpha$",
        ylabel="Contribution draw",
        title="ReLU draws are exactly 0 or 2",
    )
    axes[0].legend(frameon=False)
    _grid(axes[0])
    axes[1].plot([0, 1], [0, 2], color=SLATE, lw=2, label=r"Analytic $2\alpha$")
    axes[1].scatter(q_alpha, q_samples, color=TEAL, s=70, label="Student draws")
    axes[1].set(
        xlabel=r"Path position $\alpha$",
        ylabel="Contribution draw",
        title="Quadratic draws retain path position",
    )
    axes[1].legend(frameon=False)
    _grid(axes[1])
    fig.suptitle("Per-draw contributions reveal the estimator before averaging", weight="bold")
    return fig


def plot_mc_convergence(result):
    budgets = _array(result["budgets"])
    means = _array(result["means"])
    se = _array(result["standard_errors"])
    analytic_se = _array(result["analytic_standard_error"])
    error = _array(result["absolute_error"])
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1), constrained_layout=True)
    axes[0].fill_between(
        budgets, means - se, means + se, color="#bfdbfe", label="Mean ± ordinary MC SE"
    )
    axes[0].plot(budgets, means, color=BLUE, marker="o", label="Nested-prefix estimate")
    axes[0].axhline(1, color=SLATE, ls="--", label="Population target = 1")
    axes[0].set_xscale("log", base=2)
    axes[0].set(
        xlabel="Draws K",
        ylabel="Expected Gradients estimate",
        title="Running estimate uses literal prefixes",
    )
    axes[0].legend(frameon=False, fontsize=8)
    _grid(axes[0])
    floor = np.finfo(float).eps
    axes[1].loglog(
        budgets,
        np.maximum(error, floor),
        color=ORANGE,
        marker="o",
        label="Observed absolute error",
    )
    axes[1].loglog(budgets, se, color=BLUE, marker="s", label="Estimated ordinary SE")
    axes[1].loglog(budgets, analytic_se, color=SLATE, ls="--", label=r"Population $1/\sqrt{K}$")
    axes[1].set(
        xlabel="Draws K", ylabel="Magnitude", title="SE measures Monte Carlo sampling only"
    )
    axes[1].legend(frameon=False, fontsize=8)
    _grid(axes[1])
    fig.suptitle(
        "The Bernoulli-ReLU mean stabilizes at the ordinary sampling rate", weight="bold"
    )
    return fig


def plot_estimator_controls(result):
    names = ["Affine\nbalanced", "Dummy\ninput", "Constant", "Finite\nReLU"]
    keys = ["affine", "dummy", "constant", "finite_relu"]
    prediction = np.array([float(_array(result[key]["prediction"])) for key in keys])
    reconstruction = np.array([float(_array(result[key]["reconstruction"])) for key in keys])
    diagnostic_magnitude = np.array(
        [
            result["affine_max_abs_error"],
            result["dummy_max_abs"],
            result["constant_max_abs"],
            abs(float(_array(result["finite_relu"]["residual"]))),
        ]
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1), constrained_layout=True)
    positions = np.arange(len(keys))
    axes[0].bar(positions - 0.18, prediction, 0.36, color=SLATE, label="Prediction")
    axes[0].bar(
        positions + 0.18, reconstruction, 0.36, color=BLUE, label=r"Base + $\sum\hat\phi$"
    )
    axes[0].set_xticks(positions, names)
    axes[0].set(
        ylabel="Model-output units", title="The base uses every declared background row"
    )
    axes[0].legend(frameon=False)
    _grid(axes[0])
    axes[1].bar(positions, diagnostic_magnitude, color=[TEAL, TEAL, TEAL, ORANGE])
    axes[1].set_xticks(positions, names)
    axes[1].set(
        ylabel="Absolute diagnostic magnitude",
        title="Exact controls stay zero; finite ReLU need not close",
    )
    axes[1].set_ylim(0, max(0.7, 1.18 * diagnostic_magnitude.max()))
    for x, value in zip(positions, diagnostic_magnitude):
        label = "0" if value == 0 else f"{value:.3g}"
        axes[1].text(x, value + 0.02, label, ha="center", va="bottom")
    _grid(axes[1])
    fig.suptitle(
        "Exact identities and finite-sample behavior are separate checks", weight="bold"
    )
    return fig


def plot_quadrature_doubling(result):
    nodes = _array(result["nodes"])
    estimates = _array(result["relu_estimates"])
    error = _array(result["absolute_error"])
    drift = _array(result["successive_drift"])
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1), constrained_layout=True)
    axes[0].plot(nodes, estimates, marker="o", color=BLUE, label="Finite quadrature")
    axes[0].axhline(
        float(_array(result["analytic"])), color=SLATE, ls="--", label="Analytic path integral"
    )
    axes[0].set_xscale("log", base=2)
    axes[0].set(
        xlabel="Gauss-Legendre nodes",
        ylabel="Path attribution",
        title="A ReLU step is not polynomial exactness",
    )
    axes[0].legend(frameon=False)
    _grid(axes[0])
    axes[1].loglog(nodes, error, color=ORANGE, marker="o", label="Absolute analytic error")
    axes[1].loglog(
        nodes[1:], drift[1:], color=PURPLE, marker="s", label="Successive doubling drift"
    )
    axes[1].set(
        xlabel="Gauss-Legendre nodes",
        ylabel="Absolute difference",
        title="Doubling measures numerical drift",
    )
    axes[1].legend(frameon=False)
    _grid(axes[1])
    fig.suptitle(
        "Finite quadrature is a converging control, not an exact oracle", weight="bold"
    )
    return fig


def plot_background_average(result):
    pair = _array(result["pair_512"])
    expected = _array(result["pair_expected"])
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1), constrained_layout=True)
    positions = np.arange(2)
    axes[0].bar(positions - 0.18, expected, 0.36, color=SLATE, label="Analytic pair IG")
    axes[0].bar(positions + 0.18, pair, 0.36, color=BLUE, label="512-node control")
    axes[0].set_xticks(positions, ["Donor −2", "Donor +1"])
    axes[0].set(ylabel="Path attribution", title="Every donor defines a complete path")
    axes[0].legend(frameon=False)
    _grid(axes[0])
    labels = ["Mean of\npair attributions", "Path from\nmean donor"]
    values = [float(_array(result["mean_pair"])), float(_array(result["mean_reference"]))]
    axes[1].bar(labels, values, color=[BLUE, ORANGE])
    axes[1].set(ylabel="Attribution", title="Nonlinearity sits inside the donor average")
    axes[1].set_ylim(0, 2.25)
    for x, value in enumerate(values):
        axes[1].text(x, value + 0.04, f"{value:.3f}", ha="center")
    _grid(axes[1])
    fig.suptitle(
        "Averaging explanations is not explaining from the mean baseline", weight="bold"
    )
    return fig


def plot_noise_target(result):
    sigmas = _array(result["sigmas"])
    estimates = _array(result["estimates"])
    se = _array(result["standard_errors"])
    analytic = _array(result["analytic_smoothed_target"])
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1), constrained_layout=True)
    axes[0].errorbar(
        sigmas,
        estimates,
        yerr=se,
        color=BLUE,
        marker="o",
        capsize=3,
        label="Estimate ± ordinary MC SE",
    )
    axes[0].plot(
        sigmas,
        analytic,
        color=ORANGE,
        marker="s",
        label=r"Smoothed target $\sigma/\sqrt{2\pi}$",
    )
    axes[0].axhline(0, color=SLATE, ls="--", label="Clean target")
    axes[0].set(
        xlabel=r"Query-noise scale $\sigma$",
        ylabel="Expected attribution",
        title="Noise changes the population answer",
    )
    axes[0].legend(frameon=False, fontsize=8)
    _grid(axes[0])
    width = 0.09
    axes[1].bar(
        sigmas - width / 2,
        _array(result["clean_residual"]),
        width,
        color=ORANGE,
        label="Versus clean target",
    )
    axes[1].bar(
        sigmas + width / 2,
        _array(result["smoothed_residual"]),
        width,
        color=TEAL,
        label="Versus smoothed target",
    )
    axes[1].axhline(0, color=SLATE, lw=1)
    axes[1].set(
        xlabel=r"Query-noise scale $\sigma$",
        ylabel="Residual",
        title="Only the matching target is a completeness check",
    )
    axes[1].legend(frameon=False, fontsize=8)
    _grid(axes[1])
    fig.suptitle("Gaussian query noise defines a smoothed-output estimand", weight="bold")
    return fig


def plot_path_vs_coalition(result):
    block = result["threshold"]
    analytic = _array(block["analytic_path"])[0]
    quadrature = _array(block["path_512"])[0]
    exact = _array(block["exact_shapley"])[0]
    necessity = _array(block["necessity_drop"])
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0), constrained_layout=True)
    x = np.arange(2)
    width = 0.25
    axes[0].bar(x - width, analytic, width, color=SLATE, label="Analytic path")
    axes[0].bar(x, quadrature, width, color=BLUE, label="512-node path control")
    axes[0].bar(x + width, exact, width, color=ORANGE, label="Exact discrete Shapley")
    axes[0].set_xticks(x, [r"$x_0$", r"$x_1$"])
    axes[0].set(
        ylabel="Feature contribution", title="Same endpoints, different allocation games"
    )
    axes[0].legend(frameon=False, fontsize=8)
    _grid(axes[0])
    axes[1].bar(x, necessity, color=TEAL)
    axes[1].set_xticks(x, [r"Remove $x_0$", r"Remove $x_1$"])
    axes[1].set(ylabel="Prediction drop", title="Both features are individually necessary")
    axes[1].set_ylim(0, 1.15)
    _grid(axes[1])
    fig.suptitle(
        "Completeness and necessity do not identify discrete Shapley credit", weight="bold"
    )
    return fig


def plot_kink_representation(result):
    block = result["kink"]
    names = list(block["outputs"])
    expected = _array(block["expected_binary"])
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0), constrained_layout=True)
    x = np.arange(4)
    axes[0].plot(x, expected, color=SLATE, lw=3, label="min(x0, x1)")
    markers = ["o", "s"]
    for marker, name in zip(markers, names):
        axes[0].scatter(x, _array(block["outputs"][name]), s=65, marker=marker, label=name)
    axes[0].set_xticks(x, ["(0,0)", "(1,0)", "(0,1)", "(1,1)"])
    axes[0].set(ylabel="Model output", title="Both graphs agree on all binary inputs")
    axes[0].legend(frameon=False, fontsize=8)
    _grid(axes[0])
    width = 0.24
    features = np.arange(2)
    for offset, name, color in zip((-width, 0), names, (BLUE, ORANGE)):
        axes[1].bar(
            features + offset,
            _array(block["path_phi"][name])[0],
            width,
            color=color,
            label=name,
        )
    axes[1].bar(
        features + width,
        _array(block["exact_shapley"])[0],
        width,
        color=SLATE,
        label="Exact discrete Shapley",
    )
    axes[1].set_xticks(features, [r"$x_0$", r"$x_1$"])
    axes[1].set(
        ylabel="Feature contribution", title="The exact-kink subgradient follows graph form"
    )
    axes[1].legend(frameon=False, fontsize=8)
    _grid(axes[1])
    fig.suptitle("Implementation invariance needs an exact-kink qualification", weight="bold")
    return fig


def plot_replacement_fixture(result):
    block = result["replacement"]
    local = block["local_curve"]
    random_curves = block["random_curves"]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0), constrained_layout=True)
    steps = np.arange(4)
    for index, curve in enumerate(random_curves):
        label = "All 6 feature orders" if index == 0 else None
        axes[0].plot(steps, _array(curve["insertion"])[0], color=LIGHT, lw=1.3, label=label)
        axes[1].plot(steps, _array(curve["deletion"])[0], color=LIGHT, lw=1.3, label=label)
    axes[0].plot(
        steps,
        _array(local["insertion"])[0],
        color=BLUE,
        marker="o",
        label="Signed-attribution order",
    )
    axes[1].plot(
        steps,
        _array(local["deletion"])[0],
        color=ORANGE,
        marker="o",
        label="Signed-attribution order",
    )
    for axis, title in zip(axes, ("Insertion from empty", "Deletion from full")):
        axis.axvspan(1.0, 2.0, color="#fef3c7", alpha=0.6, label="Possible dummy plateau")
        axis.set(xlabel="Features replaced", ylabel="Raw game value", title=title)
        axis.legend(frameon=False, fontsize=8)
        _grid(axis)
    fig.suptitle("Replacement curves retain endpoints, orders, and dummy steps", weight="bold")
    return fig


def plot_threshold_play(result):
    alpha = _array(result["alpha"])[0]
    draws = _array(result["draws"])[0]
    mean = _array(result["mean"])[0]
    quadrature = _array(result["quadrature"])[0]
    exact = _array(result["exact_shapley"])[0]
    query = _array(result["query"])[0]
    reference = _array(result["reference"])[0]
    weights = _array(result["weights"])
    preactivation = (
        result["bias"] + (reference + alpha[:, None] * (query - reference)) @ weights
    )
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.1), constrained_layout=True)
    order = np.argsort(alpha)
    axes[0].plot(alpha[order], preactivation[order], color=SLATE, lw=2)
    axes[0].scatter(
        alpha[order][:: max(1, len(alpha) // 80)],
        preactivation[order][:: max(1, len(alpha) // 80)],
        c=draws[order][:: max(1, len(alpha) // 80), 0],
        cmap="coolwarm",
        s=20,
    )
    axes[0].axhline(0, color=ORANGE, ls="--", label="ReLU gate")
    axes[0].set(
        xlabel=r"Path position $\alpha$",
        ylabel="Preactivation",
        title="Editable path through the gate",
    )
    axes[0].legend(frameon=False)
    _grid(axes[0])
    x = np.arange(2)
    width = 0.25
    axes[1].bar(x - width, mean, width, color=BLUE, label="Finite Expected Gradients")
    axes[1].bar(x, quadrature, width, color=TEAL, label="512-node control")
    axes[1].bar(x + width, exact, width, color=ORANGE, label="Exact discrete Shapley")
    axes[1].set_xticks(x, [r"$x_0$", r"$x_1$"])
    axes[1].set(ylabel="Contribution", title="Path and coalition credit")
    axes[1].legend(frameon=False, fontsize=8)
    _grid(axes[1])
    steps = np.arange(3)
    for name, color in (("expected_gradients", BLUE), ("exact_shapley", ORANGE)):
        curve = result["curves"][name]
        axes[2].plot(
            steps,
            _array(curve["insertion"])[0],
            color=color,
            marker="o",
            label=name.replace("_", " "),
        )
        axes[2].plot(steps, _array(curve["deletion"])[0], color=color, marker="s", ls="--")
    if np.array_equal(
        _array(result["curves"]["expected_gradients"]["insertion"]),
        _array(result["curves"]["exact_shapley"]["insertion"]),
    ):
        axes[2].text(
            0.5,
            0.08,
            "Both rankings traverse identical values",
            transform=axes[2].transAxes,
            ha="center",
            color=SLATE,
        )
    axes[2].set(
        xlabel="Features replaced",
        ylabel="Raw game value",
        title="Local insertion (solid) / deletion (dashed)",
    )
    axes[2].legend(frameon=False, fontsize=8)
    _grid(axes[2])
    fig.suptitle(f"Try it: {result['samples']} explicit path draws", weight="bold")
    return fig


def plot_native_overview(prepared, evaluated):
    """Show predictive controls, nested sampling error, and quadrature drift."""
    keys = [key for key in _model_keys(evaluated) if key in prepared["primary"]]
    if not keys:
        raise ValueError("Prepared and evaluated payloads share no declared models.")
    performance = evaluated["performance"]
    all_rows = evaluated["all_rows"]
    budgets = _array(all_rows[keys[0]]["mc"]["budgets"])
    row_count = len(_array(all_rows[keys[0]]["mc"]["source_ids"]))
    feature_count = _array(all_rows[keys[0]]["quadrature"]["mean_512"]).shape[1]

    model_errors = {}
    model_ses = {}
    model_drifts = {}
    for key in keys:
        mc = all_rows[key]["mc"]
        quadrature = all_rows[key]["quadrature"]
        if not np.array_equal(_array(mc["budgets"]), budgets):
            raise ValueError(f"{key} does not preserve the common nested K budgets.")
        q512 = _array(quadrature["mean_512"])
        model_errors[key] = np.abs(_array(mc["phi_prefix"]) - q512[None, ...])
        model_ses[key] = _array(mc["se_prefix"])
        model_drifts[key] = np.abs(q512 - _array(quadrature["mean_256"]))

    pooled_error = np.concatenate(
        [model_errors[key].reshape(len(budgets), -1) for key in keys], axis=1
    )
    pooled_se = np.concatenate(
        [model_ses[key].reshape(len(budgets), -1) for key in keys], axis=1
    )
    pooled_drift = np.concatenate([model_drifts[key].reshape(-1) for key in keys])

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.7), constrained_layout=True)
    mse = [
        _metric(performance[key], "audit_mse", "mse", "mean_squared_error") for key in keys
    ]
    baseline = _metric(
        performance["training_mean_mse"], "audit_mse", "mse", "mean_squared_error"
    )
    positions = np.arange(len(keys))
    axes[0].bar(positions, mse, color=[MODEL_COLORS[key] for key in keys])
    axes[0].axhline(
        baseline,
        color=SLATE,
        ls="--",
        lw=1.5,
        label=f"Training-mean control · {baseline:.2f}",
    )
    axes[0].set_xticks(
        positions,
        [MODEL_LABELS[key].replace(" · ", "\n") for key in keys],
        rotation=28,
        ha="right",
    )
    axes[0].set(ylabel="Audit mean squared error (MPa²)", title="Predictive controls first")
    axes[0].legend(
        loc="upper left", frameon=True, facecolor="white", edgecolor="none", fontsize=8
    )
    _grid(axes[0])

    median_error, low_error, high_error = _interval(pooled_error, axis=1)
    median_se = np.median(pooled_se, axis=1)
    floor = np.finfo(float).eps
    axes[1].fill_between(
        budgets,
        np.maximum(low_error, floor),
        np.maximum(high_error, floor),
        color="#bfdbfe",
        alpha=0.75,
        label="All coordinates · 10–90%",
    )
    for key in keys:
        trace = np.median(model_errors[key].reshape(len(budgets), -1), axis=1)
        axes[1].plot(budgets, np.maximum(trace, floor), color=MODEL_COLORS[key], lw=0.9)
    axes[1].plot(
        budgets,
        np.maximum(median_error, floor),
        color=BLUE,
        marker="o",
        lw=2.2,
        label="Pooled median |EG − q512|",
    )
    axes[1].plot(
        budgets,
        np.maximum(median_se, floor),
        color=ORANGE,
        marker="s",
        ls="--",
        label="Pooled median ordinary SE",
    )
    axes[1].set_xscale("log", base=2)
    axes[1].set_yscale("log")
    axes[1].set(
        xlabel="Expected Gradients draws K",
        ylabel="Coordinate magnitude (MPa)",
        title="Sampling error versus a numerical control",
    )
    axes[1].legend(frameon=False, fontsize=7)
    _grid(axes[1])

    drift_by_model = [model_drifts[key].reshape(-1) for key in keys]
    boxes = axes[2].boxplot(
        drift_by_model,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": SLATE},
    )
    for patch, key in zip(boxes["boxes"], keys, strict=True):
        patch.set(facecolor=MODEL_COLORS[key], alpha=0.33, edgecolor=MODEL_COLORS[key])
    axes[2].axhline(np.median(pooled_drift), color=SLATE, ls="--", lw=1)
    axes[2].set_xticks(
        np.arange(1, len(keys) + 1),
        [key.replace("_", "\n") for key in keys],
        rotation=20,
    )
    axes[2].set(
        ylabel="|q512 − q256| per coordinate (MPa)",
        title="Quadrature doubling drift stays separate",
    )
    _grid(axes[2])

    coordinate_count = len(keys) * row_count * feature_count
    fig.suptitle(
        _shape_only_prefix(prepared, evaluated)
        + f"Native overview · {len(keys)} models × {row_count} held-out rows × "
        f"{feature_count} features = {coordinate_count:,} coordinates",
        weight="bold",
    )
    fig.text(
        0.5,
        0.005,
        "q512 is an approximate path-integration control, not ground truth.",
        ha="center",
        color=SLATE,
        fontsize=8,
    )
    return fig


def plot_allrow_method_map(evaluated):
    """Show every signed coordinate and a separately scaled error map per model."""
    keys = _model_keys(evaluated)
    rows = evaluated["all_rows"]
    first_q512 = _array(rows[keys[0]]["quadrature"]["mean_512"])
    row_count, feature_count = first_q512.shape
    split = _array(evaluated["all_row_split"]).reshape(-1)
    if split.shape != (row_count,) or not np.isin(split, (0, 1)).all():
        raise ValueError("all_row_split must label every row as explanation 0 or audit 1.")
    audit_start = int(np.flatnonzero(split == 1)[0]) if np.any(split == 1) else row_count
    fig, axes = plt.subplots(
        len(keys),
        3,
        figsize=(15.4, 2.55 * len(keys)),
        constrained_layout=True,
        squeeze=False,
    )
    for row_index, key in enumerate(keys):
        block = rows[key]
        eg = _array(block["mc"]["phi_prefix"])[-1]
        q512 = _array(block["quadrature"]["mean_512"])
        exact = _array(block["baselines"]["phi_exact"])
        if eg.shape != (row_count, feature_count):
            raise ValueError(f"{key} K=2048 attributions have shape {eg.shape}.")
        comparisons = ((q512, "q512 path control"), (exact, "Exact coalition SHAP"))
        for column, (comparison, label) in enumerate(comparisons):
            axis = axes[row_index, column]
            values = np.concatenate((comparison.reshape(-1), eg.reshape(-1)))
            bound = max(float(np.max(np.abs(values))), 1e-9) * 1.04
            axis.scatter(
                comparison,
                eg,
                s=5,
                alpha=0.22,
                color=MODEL_COLORS[key],
                rasterized=True,
            )
            axis.plot([-bound, bound], [-bound, bound], color=SLATE, ls="--", lw=0.9)
            axis.set(xlim=(-bound, bound), ylim=(-bound, bound))
            axis.text(
                0.03,
                0.95,
                f"mean |gap| = {np.mean(np.abs(eg - comparison)):.3g} MPa",
                transform=axis.transAxes,
                va="top",
                fontsize=7.5,
                color=SLATE,
            )
            if row_index == len(keys) - 1:
                axis.set_xlabel(f"{label} contribution (MPa)")
            if column == 0:
                axis.set_ylabel("EG K=2048 (MPa)")
            _grid(axis)
        axes[row_index, 0].set_title(MODEL_LABELS[key], loc="left", fontsize=9, weight="bold")

        error = eg - q512
        limit = max(float(np.max(np.abs(error))), np.finfo(float).eps)
        norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
        image = axes[row_index, 2].imshow(
            error.T,
            aspect="auto",
            interpolation="nearest",
            cmap="RdBu_r",
            norm=norm,
            rasterized=True,
        )
        if audit_start < row_count:
            axes[row_index, 2].axvline(audit_start - 0.5, color="black", ls="--", lw=0.8)
        axes[row_index, 2].set_yticks(
            range(feature_count),
            FEATURE_NAMES if row_index == len(keys) // 2 else [""] * feature_count,
        )
        axes[row_index, 2].set_title(
            f"EG K=2048 − q512 · max |gap| {limit:.3g} MPa",
            loc="left",
            fontsize=8,
            weight="bold",
        )
        if row_index == len(keys) - 1:
            axes[row_index, 2].set_xticks(
                [0, max(0, audit_start - 1), row_count - 1]
            )
            axes[row_index, 2].set_xlabel(
                f"Held-out row · explanation 0–{audit_start - 1} | "
                f"audit {audit_start}–{row_count - 1}"
            )
        else:
            axes[row_index, 2].set_xticks([])
        for spine in axes[row_index, 2].spines.values():
            spine.set_visible(False)
        colorbar = fig.colorbar(image, ax=axes[row_index, 2], shrink=0.78, pad=0.015)
        colorbar.set_label("MPa", fontsize=7)
        colorbar.ax.tick_params(labelsize=7)

    coordinate_count = len(keys) * row_count * feature_count
    fig.suptitle(
        _shape_only_prefix(evaluated)
        + f"Signed all-row comparison · every one of {coordinate_count:,} coordinates · "
        "dedicated MPa scale per model",
        weight="bold",
    )
    return fig


def _waterfall_axis(axis, base, prediction, values, standard_errors, order, feature_names):
    running = float(base)
    for position, feature in enumerate(order):
        value = float(values[feature])
        updated = running + value
        axis.barh(
            position,
            abs(value),
            left=min(running, updated),
            height=0.62,
            color=POSITIVE if value >= 0 else NEGATIVE,
        )
        axis.errorbar(
            updated,
            position,
            xerr=float(standard_errors[feature]),
            color=SLATE,
            capsize=2,
            lw=0.9,
        )
        axis.text(
            updated,
            position,
            f" {value:+.2f}",
            ha="left" if value >= 0 else "right",
            va="center",
            fontsize=8,
        )
        running = updated
    axis.axvline(base, color=SLATE, ls="--", lw=1.2, label="Full-background base")
    axis.axvline(prediction, color=TEAL, lw=1.5, label="Prediction")
    axis.axvline(running, color=BLUE, ls=":", lw=1.5, label="Finite EG reconstruction")
    axis.set_yticks(range(len(order)), [feature_names[index] for index in order])
    axis.invert_yaxis()
    axis.set_xlabel("Predicted compressive strength (MPa)")
    axis.set_title("Finite EG waterfall · whiskers are coordinate ±1 ordinary SE", loc="left")
    axis.legend(frameon=False, fontsize=7)
    _grid(axis)


def plot_fixed_waterfall(native_play, feature_names=FEATURE_NAMES):
    """Explain the predeclared source-1 prediction without making a population claim."""
    eg = _array(native_play["phi_eg"]).reshape(-1)
    se = _array(native_play["se_eg"]).reshape(-1)
    q512 = _array(native_play["phi_q512"]).reshape(-1)
    exact = _array(native_play["phi_exact"]).reshape(-1)
    deep = _array(native_play["phi_deep"]).reshape(-1)
    gradient = _array(native_play["phi_gradient"]).reshape(-1)
    feature_names = tuple(feature_names)
    if not all(len(values) == len(feature_names) for values in (eg, se, q512, exact, deep, gradient)):
        raise ValueError("Every fixed-play attribution must have one value per feature.")
    base = _scalar(native_play["base"])
    prediction = _scalar(native_play["prediction"])
    order = np.argsort(
        -np.maximum.reduce(
            [np.abs(eg), np.abs(q512), np.abs(exact), np.abs(deep), np.abs(gradient)]
        )
    )

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.6), constrained_layout=True)
    _waterfall_axis(axes[0], base, prediction, eg, se, order, feature_names)
    positions = np.arange(len(order))
    axes[1].errorbar(
        eg[order],
        positions,
        xerr=se[order],
        color=BLUE,
        marker="o",
        capsize=2.5,
        ls="none",
        label="Expected Gradients · K=2048 ±1 SE",
    )
    for name, values, marker in (
        ("quadrature_512", q512, "s"),
        ("exact", exact, "D"),
        ("deep", deep, "^"),
        ("gradient", gradient, "v"),
    ):
        axes[1].scatter(
            values[order],
            positions,
            color=METHOD_COLORS[name],
            marker=marker,
            s=38,
            label=METHOD_LABELS[name],
            zorder=3,
        )
    axes[1].axvline(0, color=SLATE, lw=1)
    axes[1].set_yticks(positions, [feature_names[index] for index in order])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Feature contribution (MPa)")
    axes[1].set_title("Aligned feature credit on the same prediction", loc="left")
    axes[1].legend(frameon=False, fontsize=7)
    _grid(axes[1])

    source = _selection_value(native_play, "source_id", "?")
    model = _selection_value(native_play, "model_key", "clean_0")
    fig.suptitle(
        _shape_only_prefix(native_play)
        + f"Frozen one-recipe display · source {source} · {MODEL_LABELS.get(str(model), model)} · "
        f"base {base:.2f} → prediction {prediction:.2f} MPa\n"
        "One predeclared recipe explains one model prediction; it is not a population claim.",
        weight="bold",
    )
    return fig


def plot_background_and_noise(evaluated):
    """Separate background movement, sampling error, drift, and noise target shift."""
    variants = evaluated["background_variants"]
    background_order = (
        "primary64",
        "nested16",
        "nested128",
        "independent64_a",
        "independent64_b",
    )
    keys = [key for key in background_order if key in variants]
    if keys != list(background_order):
        raise ValueError("The native payload must retain all five frozen backgrounds.")
    labels = {
        "primary64": "Primary\n64",
        "nested16": "Nested\n16",
        "nested128": "Nested\n128",
        "independent64_a": "Independent\n64 A",
        "independent64_b": "Independent\n64 B",
    }
    primary_q512 = _array(variants["primary64"]["quadrature"]["mean_512"])
    row_count, feature_count = primary_q512.shape
    positions = np.arange(len(keys))
    bases = np.array([_scalar(variants[key]["mc"]["base_full"]) for key in keys])

    movement, mc_error, ordinary_se, quadrature_drift = [], [], [], []
    for key in keys:
        block = variants[key]
        mc = block["mc"]
        quadrature = block["quadrature"]
        q512 = _array(quadrature["mean_512"])
        movement.append(np.mean(np.abs(q512 - primary_q512), axis=1))
        mc_error.append(np.mean(np.abs(_array(mc["phi_prefix"])[-1] - q512), axis=1))
        ordinary_se.append(np.mean(_array(mc["se_prefix"])[-1], axis=1))
        quadrature_drift.append(
            np.mean(np.abs(q512 - _array(quadrature["mean_256"])), axis=1)
        )

    noise = evaluated["noise_variants"]
    noise_keys = ("sigma_0", "sigma_0_02", "sigma_0_10")
    if any(key not in noise for key in noise_keys):
        raise ValueError("The native payload must retain sigma 0, 0.02, and 0.10.")
    sigmas = np.array([_scalar(noise[key]["sigma_standardized"]) for key in noise_keys])
    clean_residual = np.stack(
        [_array(noise[key]["clean_target_residual"]) for key in noise_keys], axis=1
    )
    smoothed_residual = np.stack(
        [_array(noise[key]["smoothed_target_residual"]) for key in noise_keys], axis=1
    )
    smoothed_target_se = np.stack(
        [_array(noise[key]["smoothed_target_se"]) for key in noise_keys], axis=1
    )

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.3), constrained_layout=True)
    axes[0, 0].plot(positions, bases, color=BLUE, marker="o", lw=2)
    axes[0, 0].axhline(bases[0], color=SLATE, ls="--", lw=1, label="Primary64 base")
    axes[0, 0].set_xticks(positions, [labels[key] for key in keys])
    axes[0, 0].set(
        ylabel="Mean background prediction (MPa)",
        title="Changing donor support changes the base",
    )
    axes[0, 0].legend(frameon=False, fontsize=8)
    _grid(axes[0, 0])

    summaries = (
        (movement, "q512 movement from primary64", PURPLE, "D", -0.18),
        (mc_error, "|EG K=2048 − q512|", BLUE, "o", -0.06),
        (ordinary_se, "Ordinary EG SE", ORANGE, "s", 0.06),
        (quadrature_drift, "|q512 − q256| drift", TEAL, "^", 0.18),
    )
    for values, label, color, marker, offset in summaries:
        stacked = np.stack(values, axis=1)
        _interval_errorbar(
            axes[0, 1],
            positions,
            stacked,
            label=label,
            color=color,
            marker=marker,
            offset=offset,
        )
    axes[0, 1].set_xticks(positions, [labels[key] for key in keys])
    axes[0, 1].set(
        ylabel="Per-row mean absolute feature difference (MPa)",
        title="Four distinct sources of movement · median, 10–90%",
    )
    axes[0, 1].legend(frameon=False, fontsize=7)
    _grid(axes[0, 1])

    _interval_errorbar(
        axes[1, 0],
        sigmas,
        clean_residual,
        label="Attribution sum − clean prediction difference",
        color=ORANGE,
        marker="o",
    )
    axes[1, 0].axhline(0, color=SLATE, lw=1)
    axes[1, 0].set_xticks(sigmas)
    axes[1, 0].set(
        xlabel="Gaussian query-noise scale σ · train-standardized input units",
        ylabel="Signed residual (MPa)",
        title="Clean target residual · median, 10–90%",
    )
    axes[1, 0].legend(frameon=False, fontsize=7)
    _grid(axes[1, 0])

    _interval_errorbar(
        axes[1, 1],
        sigmas,
        smoothed_residual,
        label="Attribution sum − smoothed prediction difference",
        color=TEAL,
        marker="s",
    )
    median_target_se = np.median(smoothed_target_se, axis=0)
    axes[1, 1].plot(
        sigmas,
        median_target_se,
        color=PURPLE,
        ls="--",
        marker="^",
        label="Median noisy-prediction mean SE",
    )
    axes[1, 1].axhline(0, color=SLATE, lw=1)
    axes[1, 1].set_xticks(sigmas)
    axes[1, 1].set(
        xlabel="Gaussian query-noise scale σ · train-standardized input units",
        ylabel="Signed residual / SE (MPa)",
        title="Matching smoothed target residual · median, 10–90%",
    )
    axes[1, 1].legend(frameon=False, fontsize=7)
    _grid(axes[1, 1])

    fig.suptitle(
        _shape_only_prefix(evaluated)
        + f"Background and noise diagnostics · clean_0 · all {row_count} held-out rows × "
        f"{feature_count} features\n"
        "Background choice changes the estimand; Gaussian noise changes the prediction target.",
        weight="bold",
    )
    return fig


def plot_replicates_and_distance(evaluated):
    """Show descriptive replicate coverage and nearest-training-row geometry."""
    replicates = evaluated["sampling_replicates"]
    budgets = _array(replicates["budgets"])
    coverage = _array(replicates["coverage_rate"])
    model_keys = tuple(replicates["models"])
    if coverage.shape != (len(model_keys), len(budgets)):
        raise ValueError(
            f"coverage_rate must have shape models×budgets, got {coverage.shape}."
        )
    replicate_count = len(_array(replicates["replicate_seeds"]))
    source_count = len(_array(replicates["source_ids"]))
    distance = evaluated["distance_diagnostic"]
    distance_series = {
        "Held-out queries": _array(distance["query_distance"]),
        "Training-donor endpoints": _array(distance["donor_distance"]),
        "Path points": _array(distance["path"]["distance"]),
        "Coalition hybrids": _array(distance["coalition"]["distance"]),
    }

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), constrained_layout=True)
    for index, key in enumerate(model_keys):
        axes[0].plot(
            budgets,
            coverage[index],
            color=MODEL_COLORS.get(key, (BLUE, ORANGE, PURPLE)[index % 3]),
            marker=("o", "s", "^")[index % 3],
            lw=2,
            label=MODEL_LABELS.get(key, str(key)),
        )
    axes[0].axhline(0.95, color=SLATE, ls="--", lw=1.2, label="Nominal 95% reference")
    axes[0].set_xscale("log", base=2)
    axes[0].set_ylim(0, 1.02)
    axes[0].set(
        xlabel="Expected Gradients draws K",
        ylabel="Descriptive q512-coordinate coverage",
        title=f"q512 inside estimate ±1.96 ordinary SE · {replicate_count} schedules",
    )
    axes[0].legend(frameon=False, fontsize=7)
    _grid(axes[0])

    colors = (SLATE, TEAL, BLUE, ORANGE)
    for (label, values), color in zip(distance_series.items(), colors, strict=True):
        finite = np.sort(values[np.isfinite(values)])
        if not len(finite):
            raise ValueError(f"The {label} distance record contains no finite values.")
        cumulative = np.arange(1, len(finite) + 1) / len(finite)
        axes[1].plot(
            finite,
            cumulative,
            color=color,
            lw=2,
            label=f"{label} · n={len(finite):,}",
            rasterized=len(finite) > 50_000,
        )
    axes[1].set(
        xlabel="Nearest-training-row Euclidean distance\n(train-standardized feature space)",
        ylabel="Empirical cumulative fraction",
        title="Geometry of observed and computational inputs",
    )
    axes[1].legend(frameon=False, fontsize=7)
    _grid(axes[1])

    fig.suptitle(
        _shape_only_prefix(evaluated)
        + f"Sampling and distance diagnostics · {len(model_keys)} models × {source_count} "
        "predeclared rows\n"
        "Coverage is descriptive; q512 is approximate; coordinates share rows and schedules. "
        "Distance is geometry, not density or an OOD probability.",
        weight="bold",
    )
    return fig


def _oriented_replacement(values, direction, curve_name):
    values = _array(values)
    if curve_name == "insertion":
        return direction * (values - values[..., [0]])
    return direction * (values[..., [0]] - values)


def plot_replacement_curves(evaluated, model_key="clean_0"):
    """Compare five frozen explanation rankings with all 100 random audit orders."""
    replacement = evaluated["replacement_curves"]
    if model_key not in MODEL_KEYS or model_key not in replacement:
        raise ValueError("model_key must select one of the seven retained replacement records.")
    block = replacement[model_key]
    random = replacement["random"]
    random_models = tuple(random["models"])
    model_index = random_models.index(model_key)
    exact_values = _array(block["exact"]["insertion"])
    direction = np.where(exact_values[:, -1] - exact_values[:, 0] >= 0, 1.0, -1.0)[:, None]
    audit_rows, steps_count = exact_values.shape
    steps = np.arange(steps_count)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    coincident_notes = []
    for axis, curve_name in zip(axes, ("insertion", "deletion"), strict=True):
        random_values = _array(random[curve_name])[model_index]
        random_oriented = _oriented_replacement(
            random_values, direction[:, None, :], curve_name
        )
        random_order_means = random_oriented.mean(axis=0)
        lower, upper = np.quantile(random_order_means, [0.10, 0.90], axis=0)
        axis.fill_between(
            steps,
            lower,
            upper,
            color=GRAY,
            alpha=0.28,
            label=f"{random_order_means.shape[0]} random orders · 10–90% band",
        )
        axis.plot(
            steps,
            random_order_means.mean(axis=0),
            color=GRAY,
            ls="--",
            lw=1.4,
            label="Random-order mean",
        )
        method_means = {}
        for method in METHOD_KEYS:
            oriented = _oriented_replacement(block[method][curve_name], direction, curve_name)
            method_means[method] = oriented.mean(axis=0)
            axis.plot(
                steps,
                method_means[method],
                color=METHOD_COLORS[method],
                marker="o",
                ms=3.2,
                lw=1.8,
                label=METHOD_LABELS[method],
            )
        if curve_name == "insertion":
            for left_index, left in enumerate(METHOD_KEYS):
                for right in METHOD_KEYS[left_index + 1 :]:
                    if np.array_equal(method_means[left], method_means[right]):
                        coincident_notes.append(
                            f"{METHOD_LABELS[left]} = {METHOD_LABELS[right]}"
                        )
        axis.axhline(0, color=SLATE, lw=1)
        axis.set(
            xlabel="Features replaced under a frozen explanation-split order",
            ylabel=(
                "Oriented prediction gain from empty (MPa)"
                if curve_name == "insertion"
                else "Oriented prediction drop from full (MPa)"
            ),
            title=f"Audit {curve_name} · means retain negative prefixes",
        )
        _grid(axis)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False, fontsize=7)
    note = "No exactly coincident method means."
    if coincident_notes:
        note = "Coincident mean curves: " + "; ".join(coincident_notes)
    qualification = fill(
        note
        + " · These computational interventions are not an attribution superiority gate.",
        width=104,
        break_long_words=False,
    )
    fig.suptitle(
        _shape_only_prefix(evaluated)
        + f"Frozen audit replacement curves · {MODEL_LABELS[model_key]} · "
        f"all {audit_rows} audit rows\n"
        + qualification,
        weight="bold",
    )
    return fig


def plot_real_recipe_play(native_play, feature_names=FEATURE_NAMES):
    """Render one recomputed recipe's five explanations and replacement paths."""
    feature_names = tuple(feature_names)
    values = {
        "eg": _array(native_play["phi_eg"]).reshape(-1),
        "quadrature_512": _array(native_play["phi_q512"]).reshape(-1),
        "exact": _array(native_play["phi_exact"]).reshape(-1),
        "deep": _array(native_play["phi_deep"]).reshape(-1),
        "gradient": _array(native_play["phi_gradient"]).reshape(-1),
    }
    se = _array(native_play["se_eg"]).reshape(-1)
    if not all(len(item) == len(feature_names) for item in (*values.values(), se)):
        raise ValueError("Every real-play value must have one coordinate per named feature.")
    order = np.argsort(-np.maximum.reduce([np.abs(item) for item in values.values()]))
    local_labels = {
        "eg": "Expected Gradients",
        "quadrature_512": METHOD_LABELS["quadrature_512"],
        "exact": METHOD_LABELS["exact"],
        "deep": METHOD_LABELS["deep"],
        "gradient": METHOD_LABELS["gradient"],
    }
    local_colors = {"eg": BLUE, **METHOD_COLORS}
    markers = {"eg": "o", "quadrature_512": "s", "exact": "D", "deep": "^", "gradient": "v"}

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.8), constrained_layout=True)
    positions = np.arange(len(order))
    offsets = np.linspace(-0.24, 0.24, len(values))
    for offset, (method, contribution) in zip(offsets, values.items(), strict=True):
        if method == "eg":
            axes[0].errorbar(
                contribution[order],
                positions + offset,
                xerr=se[order],
                color=local_colors[method],
                marker=markers[method],
                capsize=2,
                ls="none",
                label="Expected Gradients ±1 ordinary SE",
            )
        else:
            axes[0].scatter(
                contribution[order],
                positions + offset,
                color=local_colors[method],
                marker=markers[method],
                s=38,
                label=local_labels[method],
            )
    axes[0].axvline(0, color=SLATE, lw=1)
    axes[0].set_yticks(positions, [feature_names[index] for index in order])
    axes[0].invert_yaxis()
    axes[0].set(
        xlabel="Feature contribution (MPa)",
        title="Recomputed local explanations",
    )
    axes[0].legend(frameon=False, fontsize=7)
    _grid(axes[0])

    curves = native_play["local_replacement_curves"]
    step_count = len(feature_names) + 1
    steps = np.arange(step_count)
    for method in values:
        curve = curves[method]
        insertion = _array(curve["insertion"]).reshape(-1)
        deletion = _array(curve["deletion"]).reshape(-1)
        if insertion.shape != (step_count,) or deletion.shape != (step_count,):
            raise ValueError(f"{method} local replacement curves lost a feature step.")
        axes[1].plot(
            steps,
            insertion,
            color=local_colors[method],
            marker="o",
            ms=3,
            lw=1.7,
            label=f"{local_labels[method]} · insert",
        )
        axes[1].plot(
            steps,
            deletion,
            color=local_colors[method],
            marker="s",
            ms=3,
            lw=1.2,
            ls="--",
            label=f"{local_labels[method]} · delete",
        )
    axes[1].set(
        xlabel="Features replaced under each local signed order",
        ylabel="Raw model prediction (MPa)",
        title="Exploratory local insertion and deletion",
    )
    _grid(axes[1])
    style_handles = [
        Line2D([0], [0], color=SLATE, marker="o", lw=1.7, label="Insertion"),
        Line2D([0], [0], color=SLATE, marker="s", ls="--", lw=1.2, label="Deletion"),
    ]
    method_handles = [
        Line2D([0], [0], color=local_colors[key], lw=3, label=local_labels[key])
        for key in values
    ]
    fig.legend(
        handles=method_handles + style_handles,
        loc="outside lower center",
        ncol=4,
        frameon=False,
        fontsize=7,
    )

    source = _selection_value(native_play, "source_id", "?")
    model = _selection_value(native_play, "model_key", "?")
    background = _selection_value(native_play, "background_key", "?")
    samples = _selection_value(native_play, "samples", "?")
    sigma = _selection_value(native_play, "sigma_standardized", "?")
    split = _selection_value(native_play, "split", "?")
    audit_reuse = _selection_value(native_play, "audit_reuse", "?")
    fig.suptitle(
        _shape_only_prefix(native_play)
        + f"Real recipe play · source {source} · {model} · {background} · K={samples} · σ={sigma} · "
        f"{split} ({audit_reuse})",
        weight="bold",
    )
    return fig
