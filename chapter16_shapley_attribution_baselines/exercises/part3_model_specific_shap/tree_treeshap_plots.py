"""Matplotlib renderers for the TreeSHAP learner lab.

The plotters contain no attribution or fitting logic.  Each returns its figure
for inline notebook display and optionally saves the same figure under a fixed,
learner-facing stem when ``output_dir`` is supplied.
"""

from __future__ import annotations

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np

BLUE = "#2563EB"
ORANGE = "#EA580C"
TEAL = "#0F766E"
PURPLE = "#7C3AED"
RED = "#DC2626"
SLATE = "#475569"
LIGHT = "#F8FAFC"
GRID = "#CBD5E1"
POSITIVE = "#E45756"
NEGATIVE = "#4C78A8"
FAMILY_COLORS = {"tree": BLUE, "forest": TEAL, "boost": PURPLE}


def _new_figure(*args, **kwargs):
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
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="This figure includes Axes that are not compatible with tight_layout",
        )
        fig.tight_layout()
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            destination / f"{stem}.png", dpi=180, bbox_inches="tight", facecolor="white"
        )
    return fig


def _mechanism(payload, key):
    return (
        payload["mechanism"][key]
        if "mechanism" in payload
        else payload[key] if key in payload else payload
    )


def _timing_axis(payload, key):
    if "timing" in payload:
        payload = payload["timing"]
    return payload[key] if key in payload else payload


def plot_missing_feature_games(output_dir=None):
    """Contrast the three questions hidden behind 'a missing feature'."""
    fig, axes = _new_figure(1, 3, figsize=(12.2, 3.6))
    fig.suptitle(
        "A missing feature is a choice of game", fontsize=16, fontweight="bold", y=1.03
    )

    ax = axes[0]
    ax.set_title("Tree path", loc="left", fontweight="bold")
    node_xy = [
        (0.5, 0.85),
        (0.25, 0.48),
        (0.75, 0.48),
        (0.12, 0.12),
        (0.38, 0.12),
        (0.62, 0.12),
        (0.88, 0.12),
    ]
    for start, end, weight in (
        (0, 1, ".5"),
        (0, 2, ".5"),
        (1, 3, ".9"),
        (1, 4, ".1"),
        (2, 5, ".1"),
        (2, 6, ".9"),
    ):
        ax.plot(
            [node_xy[start][0], node_xy[end][0]],
            [node_xy[start][1], node_xy[end][1]],
            color=GRID,
            lw=2,
        )
        midpoint = np.mean([node_xy[start], node_xy[end]], axis=0)
        ax.text(
            *midpoint,
            weight,
            fontsize=8,
            color=SLATE,
            ha="center",
            bbox=dict(facecolor="white", edgecolor="none", pad=0.4),
        )
    for index, (x, y) in enumerate(node_xy):
        label = "split" if index < 3 else "leaf"
        ax.add_patch(
            FancyBboxPatch(
                (x - 0.09, y - 0.045),
                0.18,
                0.09,
                boxstyle="round,pad=0.02",
                facecolor="#DBEAFE" if index < 3 else LIGHT,
                edgecolor=BLUE if index < 3 else GRID,
            )
        )
        ax.text(x, y, label, ha="center", va="center", fontsize=8)
    ax.text(
        0.5,
        -0.03,
        "Absent branches follow local node cover",
        ha="center",
        fontsize=9,
        color=SLATE,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.1, 1)
    ax.axis("off")

    ax = axes[1]
    ax.set_title("Interventional", loc="left", fontweight="bold")
    donors = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for row, donor in enumerate(donors):
        y = 0.82 - row * 0.19
        ax.add_patch(
            FancyBboxPatch(
                (0.03, y - 0.055),
                0.28,
                0.11,
                boxstyle="round,pad=.02",
                facecolor=LIGHT,
                edgecolor=GRID,
            )
        )
        ax.text(0.17, y, f"donor {donor}", ha="center", va="center", fontsize=8)
        ax.annotate(
            "", xy=(0.58, y), xytext=(0.34, y), arrowprops=dict(arrowstyle="->", color=SLATE)
        )
        ax.add_patch(
            FancyBboxPatch(
                (0.61, y - 0.055),
                0.32,
                0.11,
                boxstyle="round,pad=.02",
                facecolor="#FFEDD5",
                edgecolor=ORANGE,
            )
        )
        ax.text(0.77, y, f"hybrid (1, {donor[1]})", ha="center", va="center", fontsize=8)
    ax.text(
        0.5,
        -0.03,
        "Keep each donor row intact; then average",
        ha="center",
        fontsize=9,
        color=SLATE,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.1, 1)
    ax.axis("off")

    ax = axes[2]
    ax.set_title("Population conditional", loc="left", fontweight="bold")
    joint = np.array([[0.45, 0.05], [0.05, 0.45]])
    image = ax.imshow(joint, cmap="Purples", vmin=0, vmax=0.5, extent=(-0.5, 1.5, 1.5, -0.5))
    for row in range(2):
        for col in range(2):
            ax.text(
                col,
                row,
                f"{joint[row, col]:.2f}",
                ha="center",
                va="center",
                color="white" if joint[row, col] > 0.25 else SLATE,
                fontweight="bold",
            )
    ax.add_patch(Rectangle((0.5, -0.5), 1, 2, fill=False, edgecolor=RED, linewidth=3))
    ax.set_xticks([0, 1], ["x₀=0", "x₀=1"])
    ax.set_yticks([0, 1], ["x₁=0", "x₁=1"])
    ax.set_xlabel("Condition on observed coordinates", color=SLATE)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.text(
        0.5,
        -0.02,
        "Same prediction function; different missing-feature questions can assign "
        "different credit.",
        ha="center",
        fontsize=10,
        color=SLATE,
    )
    return _finish(fig, output_dir, "01_missing_feature_games")


def _tree_positions(tree):
    positions, leaf_cursor = {}, [0]

    def visit(node, depth):
        if tree["left"][node] == -1:
            positions[node] = (leaf_cursor[0], -depth)
            leaf_cursor[0] += 1
        else:
            visit(int(tree["left"][node]), depth + 1)
            visit(int(tree["right"][node]), depth + 1)
            positions[node] = (
                (positions[int(tree["left"][node])][0] + positions[int(tree["right"][node])][0])
                / 2,
                -depth,
            )

    visit(0, 0)
    return positions


def _query_route(tree, query):
    route, node = set(), 0
    while tree["left"][node] != -1:
        feature = int(tree["feature"][node])
        child = int(
            tree["left"][node]
            if query[feature] <= tree["threshold"][node]
            else tree["right"][node]
        )
        route.add((node, child))
        node = child
    return route


def plot_tiny_tree(diagnostic, output_dir=None):
    """Draw the actual controlled tree, local cover fractions and query route."""
    play = _mechanism(diagnostic, "tiny_play")
    tree, query = play["tree"], np.asarray(play["query"])
    positions, route = _tree_positions(tree), _query_route(tree, query)
    fig, ax = _new_figure(figsize=(9.0, 4.7))
    ax.set_title(
        "One prediction, two kinds of path information   •   "
        f"query x=({query[0]:.0f}, {query[1]:.0f})",
        loc="left",
        fontsize=14,
        fontweight="bold",
    )
    for parent in range(len(tree["left"])):
        if tree["left"][parent] == -1:
            continue
        for child, branch in (
            (int(tree["left"][parent]), "≤"),
            (int(tree["right"][parent]), ">"),
        ):
            x0, y0 = positions[parent]
            x1, y1 = positions[child]
            active = (parent, child) in route
            ax.plot(
                [x0, x1],
                [y0, y1],
                color=BLUE if active else GRID,
                lw=3 if active else 1.8,
                zorder=1,
            )
            fraction = tree["cover"][child] / tree["cover"][parent]
            ax.text(
                (x0 + x1) / 2,
                (y0 + y1) / 2 + 0.06,
                f"{branch}  p={fraction:.2f}",
                ha="center",
                fontsize=8,
                color=BLUE if active else SLATE,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.4),
            )
    for node, (x, y) in positions.items():
        leaf = tree["left"][node] == -1
        if leaf:
            label = f"leaf {tree['value'][node]:.1f}\ncover {tree['cover'][node]:.0f}"
            face, edge = "#ECFDF5", TEAL
        else:
            label = (
                f"x{tree['feature'][node]} ≤ {tree['threshold'][node]:.1f}\n"
                f"E[f]={tree['value'][node]:.2f}  •  cover {tree['cover'][node]:.0f}"
            )
            face, edge = "#EFF6FF", BLUE
        width = 0.9 if leaf else 1.45
        ax.add_patch(
            FancyBboxPatch(
                (x - width / 2, y - 0.14),
                width,
                0.28,
                boxstyle="round,pad=.03",
                facecolor=face,
                edgecolor=edge,
                linewidth=1.5,
                zorder=2,
            )
        )
        ax.text(x, y, label, ha="center", va="center", fontsize=8.5, zorder=3)
    ax.text(
        0.02,
        0.02,
        "Blue = observed query route   •   gray branch labels = cover weights used "
        "when a split feature is absent",
        transform=ax.transAxes,
        fontsize=9,
        color=SLATE,
    )
    ax.set_xlim(-0.7, max(x for x, _ in positions.values()) + 0.7)
    ax.set_ylim(-2.55, 0.45)
    ax.axis("off")
    return _finish(fig, output_dir, "02_tiny_tree")


def plot_cardinality_weights(diagnostic, output_dir=None):
    """Show how extend_path moves mass among coalition cardinalities."""
    data = _mechanism(diagnostic, "cardinality")
    stages = data["stages"]
    fig, axes = _new_figure(1, len(stages), figsize=(12.6, 3.25), sharey=True)
    fig.suptitle(
        "Dynamic programming stores path mass by coalition size",
        fontsize=15,
        fontweight="bold",
        y=1.04,
    )
    ymax = max(max(stage) for stage in stages) * 1.18
    for step, (ax, stage) in enumerate(zip(axes, stages)):
        x = np.arange(len(stage))
        ax.bar(x, stage, color=BLUE, width=0.7)
        for index, value in enumerate(stage):
            ax.text(index, value + ymax * 0.025, f"{value:.3f}", ha="center", fontsize=8)
        ax.set_title(
            (
                "empty path"
                if step == 0
                else (
                    f"after feature {step}\nz={data['zeros'][step-1]:.1f}, "
                    f"o={data['ones'][step-1]:.1f}"
                )
            ),
            fontsize=10,
        )
        ax.set_xticks(x)
        ax.set_xlabel("present count |S|")
        ax.set_ylim(0, ymax)
        _style_axis(ax)
    axes[0].set_ylabel("Shapley path weight")
    fig.text(
        0.5,
        -0.015,
        f"Subset enumeration parity: max |difference| = {data['max_abs_error']:.1e}",
        ha="center",
        color=SLATE,
        fontsize=9,
    )
    return _finish(fig, output_dir, "03_cardinality_weights")


def plot_correlated_games(diagnostic, output_dir=None):
    """Show coalition values and Shapley allocations for the correlated fixture."""
    data = _mechanism(diagnostic, "correlated")
    fig, axes = _new_figure(1, 2, figsize=(12.2, 4.4), gridspec_kw={"width_ratios": [1.2, 1]})
    fig.suptitle(
        "Correlation and tree structure change the game", fontsize=15, fontweight="bold", y=1.03
    )
    game_labels = ["∅", "{x₀}", "{x₁}", "{x₀,x₁}"]
    for index, (name, game) in enumerate(zip(data["game_names"], data["games"])):
        axes[0].plot(
            np.arange(4),
            game,
            marker="o",
            lw=2,
            label=name,
            color=[PURPLE, ORANGE, BLUE, TEAL][index],
        )
    axes[0].set_xticks(range(4), game_labels)
    axes[0].set_ylabel("Coalition value v(S)")
    axes[0].set_title(
        "Same endpoints; different interior coalitions", loc="left", fontweight="bold"
    )
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    _style_axis(axes[0])
    y = np.arange(len(data["game_names"]))
    height = 0.34
    axes[1].barh(y - height / 2, data["shapley"][:, 0], height, label="x₀", color=BLUE)
    axes[1].barh(y + height / 2, data["shapley"][:, 1], height, label="x₁", color=ORANGE)
    axes[1].set_yticks(y, data["game_names"])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Shapley contribution")
    axes[1].set_title("Credit for query (1,1)", loc="left", fontweight="bold")
    axes[1].legend(frameon=False)
    _style_axis(axes[1], "x")
    axes[1].axvline(0, color=SLATE, lw=0.8)
    return _finish(fig, output_dir, "04_correlated_games")


def plot_repeated_feature(diagnostic, output_dir=None):
    """Verify that two x0 splits remain one player and x1 stays a dummy."""
    data = _mechanism(diagnostic, "repeated_feature")
    fig, axes = _new_figure(1, 2, figsize=(11.8, 4.0))
    fig.suptitle(
        "Repeated splits do not create extra players", fontsize=15, fontweight="bold", y=1.03
    )
    labels = ["∅", "{x₀}", "{dummy}", "both"]
    for row, (query, game) in enumerate(zip(data["query"], data["games"])):
        axes[0].plot(range(4), game, marker="o", lw=2, label=f"x₀={query[0]:g}")
    axes[0].set_xticks(range(4), labels)
    axes[0].set_ylabel("Coalition value")
    axes[0].set_title("The dummy never changes v(S)", loc="left", fontweight="bold")
    axes[0].legend(frameon=False)
    _style_axis(axes[0])
    x = np.arange(len(data["query"]))
    width = 0.36
    axes[1].bar(x - width / 2, data["phi"][:, 0], width, label="x₀ (split twice)", color=BLUE)
    axes[1].bar(x + width / 2, data["phi"][:, 1], width, label="x₁ dummy", color=ORANGE)
    axes[1].axhline(0, color=SLATE, lw=0.8)
    axes[1].set_xticks(x, [f"x₀={row[0]:g}" for row in data["query"]])
    axes[1].set_ylabel("TreeSHAP contribution")
    axes[1].set_title("One x₀ attribution after path unwind", loc="left", fontweight="bold")
    axes[1].legend(frameon=False)
    _style_axis(axes[1])
    fig.text(
        0.5,
        -0.02,
        f"Dummy max |ϕ| = {data['dummy_max_abs']:.1e}   •   "
        f"additivity error = {data['additivity_max_abs_error']:.1e}",
        ha="center",
        color=SLATE,
        fontsize=9,
    )
    return _finish(fig, output_dir, "05_repeated_feature")


def plot_saabas_control(diagnostic, output_dir=None):
    """Show that efficiency alone cannot distinguish Saabas from Shapley."""
    data = _mechanism(diagnostic, "correlated")
    fig, axes = _new_figure(1, 2, figsize=(10.8, 4.1), sharey=True)
    fig.suptitle(
        "Saabas conserves the prediction but allocates credit by one route",
        fontsize=14.5,
        fontweight="bold",
        y=1.03,
    )
    for index, ax in enumerate(axes):
        values = np.vstack([data["shapley"][index + 2], data["saabas"][index]])
        x = np.arange(2)
        width = 0.34
        ax.bar(x - width / 2, values[:, 0], width, label="x₀", color=BLUE)
        ax.bar(x + width / 2, values[:, 1], width, label="x₁", color=ORANGE)
        ax.set_xticks(x, ["TreeSHAP", "Saabas"])
        ax.set_title(f"{'x₀' if index == 0 else 'x₁'} at the root", fontweight="bold")
        ax.axhline(0, color=SLATE, lw=0.8)
        _style_axis(ax)
        ax.text(
            0.5,
            0.96,
            f"each sum = {values[0].sum():.1f}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            color=SLATE,
            fontsize=9,
        )
    axes[0].set_ylabel("Contribution above base")
    axes[1].legend(frameon=False, loc="lower right")
    return _finish(fig, output_dir, "06_saabas_control")


def _parse_model_key(key):
    family, target, seed = key.rsplit("_", 2)
    return family, target, int(seed)


def plot_performance(evaluated, output_dir=None):
    """Compare all clean/shuffled fits with the training-mean baseline."""
    performance = evaluated["performance"]
    fig, ax = _new_figure(figsize=(10.4, 4.8))
    families = ["tree", "forest", "boost"]
    centers = np.arange(len(families))
    offsets = np.array([-0.19, 0, 0.19])
    for target, marker, fill in (("clean", "o", True), ("shuffled", "X", False)):
        for family_index, family in enumerate(families):
            values = [performance[f"{family}_{target}_{seed}"] for seed in range(3)]
            ax.scatter(
                centers[family_index] + offsets,
                values,
                s=75,
                marker=marker,
                facecolor=FAMILY_COLORS[family] if fill else "white",
                edgecolor=FAMILY_COLORS[family],
                linewidth=1.6,
                label=target.capitalize() if family_index == 0 else None,
                zorder=3,
            )
            ax.plot(
                [centers[family_index] + offsets.min(), centers[family_index] + offsets.max()],
                [np.mean(values)] * 2,
                color=FAMILY_COLORS[family],
                lw=2,
                alpha=0.7,
            )
    baseline = evaluated["mean_baseline_mse"]
    ax.axhline(
        baseline, color=RED, linestyle="--", lw=1.7, label=f"Training mean ({baseline:.1f})"
    )
    ax.set_xticks(centers, ["Decision tree", "Random forest", "Gradient boost"])
    ax.set_ylabel("Audit mean squared error (MPa²)")
    ax.set_title(
        "Real targets carry signal; shuffled targets collapse toward baseline",
        loc="left",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(frameon=False, ncol=3, loc="center left", bbox_to_anchor=(0.0, 0.65))
    _style_axis(ax)
    ax.text(
        0.99,
        0.45,
        "Three predetermined seeds per condition\n"
        "Equal points are retained, not counted as independent evidence",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        color=SLATE,
    )
    return _finish(fig, output_dir, "07_audit_performance")


def _one_row_play(play):
    phi = np.asarray(play["phi"], dtype=float)
    if phi.ndim == 2:
        phi = phi[0]
    base = float(np.asarray(play["base"]).ravel()[0])
    prediction = float(np.asarray(play["prediction"]).ravel()[0])
    target = float(np.asarray(play.get("target", [np.nan])).ravel()[0])
    query = np.asarray(play.get("query", np.full((1, len(phi)), np.nan)), dtype=float).reshape(
        -1, len(phi)
    )[0]
    return base, phi, prediction, target, query


def plot_waterfall(play, feature_names, title="A local TreeSHAP explanation", output_dir=None):
    """Draw a true additive waterfall for one native_play result."""
    base, phi, prediction, target, query = _one_row_play(play)
    names = np.asarray(feature_names)
    if len(names) != len(phi):
        raise ValueError("feature_names must match the attribution width.")
    order = np.argsort(np.abs(phi))[::-1]
    cumulative = base
    fig, ax = _new_figure(figsize=(10.5, 5.2))
    for y, index in enumerate(order):
        next_value = cumulative + phi[index]
        left, width = min(cumulative, next_value), abs(phi[index])
        color = POSITIVE if phi[index] >= 0 else NEGATIVE
        ax.add_patch(
            Rectangle((left, y - 0.31), width, 0.62, facecolor=color, edgecolor="white")
        )
        ax.plot([next_value, next_value], [y + 0.31, y + 0.69], color=GRID, lw=1)
        offset = 0.012 * max(
            1, np.ptp([base, prediction, target if np.isfinite(target) else prediction])
        )
        ax.text(
            next_value + np.sign(phi[index] or 1) * offset,
            y,
            f"{phi[index]:+.2f}",
            va="center",
            ha="left" if phi[index] >= 0 else "right",
            fontsize=8.5,
            color=SLATE,
        )
        cumulative = next_value
    labels = [f"{names[index]} = {query[index]:g}" for index in order]
    ax.set_yticks(range(len(order)), labels)
    ax.invert_yaxis()
    ax.axvline(base, color=SLATE, linestyle="--", lw=1.3, label=f"base {base:.2f}")
    ax.axvline(prediction, color=TEAL, lw=2, label=f"prediction {prediction:.2f}")
    if np.isfinite(target):
        ax.axvline(target, color=ORANGE, linestyle=":", lw=2, label=f"observed {target:.2f}")
    ax.set_xlabel("Concrete compressive strength (MPa)")
    ax.set_title(title, loc="left", fontsize=14, fontweight="bold")
    ax.legend(frameon=False, ncol=3, loc="lower right")
    _style_axis(ax, "x")
    return _finish(fig, output_dir, "08_prediction_waterfall")


def plot_attribution_heatmaps(explanation, feature_names, row_labels=None, output_dir=None):
    """Plot every row under both games and their signed difference."""
    path = np.asarray(explanation["phi_path"])
    interventional = np.asarray(explanation["phi_interventional"])
    prediction = np.asarray(explanation.get("prediction", np.arange(len(path))))
    order = np.argsort(prediction, kind="stable")
    arrays = [path[order], interventional[order], (path - interventional)[order]]
    limit = max(np.max(np.abs(path)), np.max(np.abs(interventional)), 1e-12)
    difference_limit = max(np.max(np.abs(arrays[2])), 1e-12)
    fig = plt.figure(figsize=(14.2, 6.1), facecolor="white")
    grid = fig.add_gridspec(
        1,
        6,
        width_ratios=[1, 1, 0.055, 0.16, 1, 0.055],
        wspace=0.2,
    )
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[0, 4]),
    ]
    axes[1].sharey(axes[0])
    axes[2].sharey(axes[0])
    axes[1].tick_params(labelleft=False)
    axes[2].tick_params(labelleft=False)
    contribution_color_axis = fig.add_subplot(grid[0, 2])
    difference_color_axis = fig.add_subplot(grid[0, 5])
    titles = ["Path-dependent", "Interventional", "Path − interventional"]
    images = []
    for index, (ax, array, title) in enumerate(zip(axes, arrays, titles)):
        bound = difference_limit if index == 2 else limit
        image = ax.imshow(
            array,
            aspect="auto",
            cmap="RdBu_r",
            norm=colors.TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound),
            interpolation="nearest",
        )
        images.append(image)
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(np.arange(len(feature_names)), feature_names, rotation=55, ha="right")
        ax.tick_params(axis="both", labelsize=8)
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0].set_ylabel("Explanation rows, sorted by prediction")
    if row_labels is not None and len(row_labels) <= 30:
        axes[0].set_yticks(np.arange(len(row_labels)), np.asarray(row_labels)[order])
    fig.colorbar(images[1], cax=contribution_color_axis, label="Contribution (MPa)")
    fig.colorbar(images[2], cax=difference_color_axis, label="Difference (MPa)")
    fig.suptitle(
        "The missing-feature game changes local credit row by row",
        fontsize=15,
        fontweight="bold",
        y=1.01,
    )
    return _finish(fig, output_dir, "09_attribution_heatmaps")


def plot_model_variation(explanation, feature_names, output_dir=None):
    """Retain all 18 family/target/seed fits in one attribution view."""
    keys = sorted(
        explanation,
        key=lambda key: (
            _parse_model_key(key)[0],
            _parse_model_key(key)[1],
            _parse_model_key(key)[2],
        ),
    )
    values = np.vstack([np.abs(explanation[key]["phi_interventional"]).mean(0) for key in keys])
    fig, ax = _new_figure(figsize=(10.7, 7.2))
    image = ax.imshow(values, aspect="auto", cmap="magma")
    labels = [
        f"{_parse_model_key(key)[0]:6s}  {_parse_model_key(key)[1]:8s}  "
        f"seed {_parse_model_key(key)[2]+1}"
        for key in keys
    ]
    ax.set_yticks(np.arange(len(keys)), labels, fontfamily="monospace", fontsize=8)
    ax.set_xticks(np.arange(len(feature_names)), feature_names, rotation=50, ha="right")
    ax.set_title(
        "All families, target controls and predetermined seeds",
        loc="left",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Feature")
    ax.set_ylabel("Fitted model condition")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.colorbar(
        image, ax=ax, fraction=0.035, pad=0.02, label="Mean |interventional contribution| (MPa)"
    )
    return _finish(fig, output_dir, "10_model_seed_variation")


def plot_background_sensitivity(
    prepared, evaluated, feature_names, model_key="forest_clean_0", output_dir=None
):
    """Compare the primary background with every frozen size/seed alternative."""
    primary = np.vstack(
        [
            prepared["explanation"][model_key]["phi_interventional"],
            evaluated["audit"][model_key]["phi_interventional"],
        ]
    )
    variants = {"64 / primary seed": primary}
    display_names = {
        "16_161360": "16 / primary seed",
        "128_161360": "128 / primary seed",
        "64_161361": "64 / alternate seed A",
        "64_161362": "64 / alternate seed B",
    }
    variants.update(
        {display_names[key]: value["phi"] for key, value in evaluated["backgrounds"].items()}
    )
    names = list(variants)
    arrays = [np.asarray(variants[name]) for name in names]
    means = np.vstack([np.abs(array).mean(0) for array in arrays])
    rms = np.array([np.sqrt(np.mean((array - primary) ** 2, axis=1)) for array in arrays])
    fig, axes = _new_figure(1, 2, figsize=(13.0, 4.8), gridspec_kw={"width_ratios": [1.25, 1]})
    image = axes[0].imshow(means, aspect="auto", cmap="viridis")
    axes[0].set_yticks(np.arange(len(names)), names)
    axes[0].set_xticks(np.arange(len(feature_names)), feature_names, rotation=50, ha="right")
    axes[0].set_title("Global attribution scale", loc="left", fontweight="bold")
    fig.colorbar(image, ax=axes[0], fraction=0.04, pad=0.02, label="Mean |contribution| (MPa)")
    for spine in axes[0].spines.values():
        spine.set_visible(False)
    boxes = axes[1].boxplot(
        rms[1:].T, tick_labels=names[1:], patch_artist=True, showfliers=False
    )
    for patch, color in zip(boxes["boxes"], [ORANGE, TEAL, PURPLE, RED]):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    axes[1].set_ylabel("Row RMS difference from primary (MPa)")
    axes[1].set_title("Local sensitivity to background", loc="left", fontweight="bold")
    axes[1].tick_params(axis="x", rotation=35)
    _style_axis(axes[1])
    fig.suptitle(
        "The empirical background is part of the explanation",
        fontsize=15,
        fontweight="bold",
        y=1.03,
    )
    return _finish(fig, output_dir, "11_background_sensitivity")


def plot_deletion_curves(evaluated, output_dir=None):
    """Plot held-out deletion curves with all 100 frozen random orders."""
    fig, axes = _new_figure(1, 3, figsize=(13.2, 4.15), sharey=True)
    method_colors = {"Path SHAP": BLUE, "Interventional SHAP": ORANGE, "Saabas": PURPLE}
    display = {
        "tree_clean_0": "Decision tree",
        "forest_clean_0": "Random forest",
        "boost_clean_0": "Gradient boost",
    }
    for ax, key in zip(axes, ("tree_clean_0", "forest_clean_0", "boost_clean_0")):
        mse = np.asarray(evaluated["deletion"][key]["mse"])
        steps = np.arange(mse.shape[1])
        random = mse[3:]
        ax.fill_between(
            steps,
            np.quantile(random, 0.1, axis=0),
            np.quantile(random, 0.9, axis=0),
            color=GRID,
            alpha=0.7,
            label="Random 10–90%",
        )
        ax.plot(
            steps,
            np.median(random, axis=0),
            color=SLATE,
            linestyle="--",
            lw=1.4,
            label="Random median",
        )
        for row, (name, linestyle, marker) in enumerate(
            zip(method_colors, ("-", "--", ":"), ("o", "s", "D"))
        ):
            ax.plot(
                steps,
                mse[row],
                marker=marker,
                ms=3.5,
                lw=2,
                linestyle=linestyle,
                color=method_colors[name],
                label=name,
            )
        if np.allclose(mse[0], mse[1]) and np.allclose(mse[0], mse[2]):
            note = "All 3 learned rankings\nand curves coincide"
        elif np.allclose(mse[0], mse[2]):
            differing = np.flatnonzero(~np.isclose(mse[0], mse[1]))
            steps_text = ", ".join(map(str, differing.tolist()))
            note = f"Path = Saabas\ninterventional differs at step {steps_text}"
        else:
            note = "Curves may overlap"
        ax.text(
            0.97,
            0.06,
            note,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color=SLATE,
            bbox=dict(boxstyle="round,pad=.3", facecolor="white", edgecolor=GRID, alpha=0.9),
        )
        ax.set_title(display[key], fontweight="bold")
        ax.set_xlabel("Features removed")
        ax.set_xticks(steps)
        _style_axis(ax)
    axes[0].set_ylabel("Audit MSE under interventional deletion (MPa²)")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles, labels, frameon=False, ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.04)
    )
    fig.suptitle(
        "Held-out deletion tests rankings against 100 random orders",
        fontsize=15,
        fontweight="bold",
        y=1.03,
    )
    fig.subplots_adjust(bottom=0.19)
    return _finish(fig, output_dir, "12_deletion_curves")


def _plot_runtime_axis(axis, coordinate, xlabel, title, stem, output_dir):
    fig, ax = _new_figure(figsize=(8.4, 4.6))
    exact = np.asarray(axis["exhaustive_seconds"])
    dp = np.asarray(axis["dynamic_programming_seconds"])
    for values, color in ((exact, ORANGE), (dp, BLUE)):
        ax.scatter(
            np.repeat(coordinate, values.shape[1]),
            values.ravel() * 1000,
            color=color,
            alpha=0.38,
            s=24,
            zorder=2,
        )
    ax.plot(
        coordinate,
        np.median(exact, axis=1) * 1000,
        color=ORANGE,
        marker="o",
        lw=2.2,
        label="Exhaustive coalitions",
    )
    ax.plot(
        coordinate,
        np.median(dp, axis=1) * 1000,
        color=BLUE,
        marker="o",
        lw=2.2,
        label="TreeSHAP dynamic program",
    )
    ax.set_yscale("log")
    ax.set_xticks(coordinate)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Wall-clock time for 8 queries (ms, log scale)")
    ax.set_title(title, loc="left", fontsize=14, fontweight="bold")
    ax.legend(frameon=False)
    _style_axis(ax)
    ax.text(
        0.99,
        0.03,
        "Dots: all 3 frozen repetitions\nLines: medians",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color=SLATE,
        fontsize=8.5,
    )
    return _finish(fig, output_dir, stem)


def plot_runtime_dummy_features(timing, output_dir=None):
    """Show the exponential cost of supplied dummy dimensions."""
    axis = _timing_axis(timing, "dummy_features")
    return _plot_runtime_axis(
        axis,
        np.asarray(axis["supplied_features"]),
        "Supplied features (tree uses only 3)",
        "Exhaustive enumeration pays for unused features",
        "13_runtime_dummy_features",
        output_dir,
    )


def plot_runtime_depth(timing, output_dir=None):
    """Show both algorithms as the complete balanced tree grows with depth."""
    axis = _timing_axis(timing, "depth")
    return _plot_runtime_axis(
        axis,
        np.asarray(axis["tree_depth"]),
        "Balanced tree depth (one feature per level)",
        "Runtime on complete balanced trees",
        "14_runtime_depth",
        output_dir,
    )


def plot_tiny_tree_play(play, output_dir=None):
    """Render the editable tiny play's full game and two efficient allocations."""
    data = _mechanism(play, "tiny_play")
    fig, axes = _new_figure(1, 2, figsize=(10.8, 4.0))
    game = np.asarray(data["game"])[[0, 1, 2, 3]].reshape(2, 2)
    image = axes[0].imshow(game, cmap="Blues", vmin=min(0, game.min()), vmax=game.max())
    for row in range(2):
        for col in range(2):
            axes[0].text(
                col,
                row,
                f"{game[row,col]:.2f}",
                ha="center",
                va="center",
                color="white" if game[row, col] > np.mean(game) else SLATE,
                fontweight="bold",
            )
    axes[0].set_xticks([0, 1], ["x₀ absent", "x₀ present"])
    axes[0].set_yticks([0, 1], ["x₁ absent", "x₁ present"])
    axes[0].set_title("All four coalition values", fontweight="bold")
    fig.colorbar(image, ax=axes[0], fraction=0.05, pad=0.03, label="v(S)")
    allocations = np.vstack([data["phi"], data["saabas"]])
    x = np.arange(2)
    width = 0.35
    axes[1].bar(x - width / 2, allocations[:, 0], width, color=BLUE, label="x₀")
    axes[1].bar(x + width / 2, allocations[:, 1], width, color=ORANGE, label="x₁")
    axes[1].set_xticks(x, ["TreeSHAP", "Saabas"])
    axes[1].set_ylabel("Contribution above base")
    axes[1].set_title(
        f"query={tuple(data['query'].astype(int))}, root=x{int(data['root_feature'])}",
        fontweight="bold",
    )
    axes[1].legend(frameon=False)
    axes[1].axhline(0, color=SLATE, lw=0.8)
    _style_axis(axes[1])
    fig.suptitle(
        "Try it: change the query, root feature, or population covers",
        fontsize=14.5,
        fontweight="bold",
        y=1.03,
    )
    return _finish(fig, output_dir, "15_tiny_tree_play")


def plot_real_recipe_play(play, feature_names, output_dir=None):
    """Render one editable real recipe with feature values, credit and outcome."""
    base, phi, prediction, target, query = _one_row_play(play)
    order = np.argsort(np.abs(phi))
    fig, axes = _new_figure(
        1, 2, figsize=(12.2, 4.7), gridspec_kw={"width_ratios": [1.55, 0.85]}
    )
    colors_ = [POSITIVE if value >= 0 else NEGATIVE for value in phi[order]]
    axes[0].barh(np.arange(len(order)), phi[order], color=colors_)
    axes[0].set_yticks(
        np.arange(len(order)), [f"{feature_names[i]} = {query[i]:g}" for i in order]
    )
    axes[0].axvline(0, color=SLATE, lw=0.8)
    axes[0].set_xlabel("Contribution (MPa)")
    axes[0].set_title("Which ingredients moved this prediction?", loc="left", fontweight="bold")
    _style_axis(axes[0], "x")
    categories, values, palette = (
        ["Background\nbase", "Model\nprediction"],
        [base, prediction],
        [SLATE, TEAL],
    )
    if np.isfinite(target):
        categories.append("Observed\nstrength")
        values.append(target)
        palette.append(ORANGE)
    axes[1].bar(categories, values, color=palette, width=0.62)
    for index, value in enumerate(values):
        axes[1].text(index, value, f"{value:.1f}", ha="center", va="bottom", fontweight="bold")
    axes[1].set_ylabel("Concrete compressive strength (MPa)")
    axes[1].set_title("Prediction in context", fontweight="bold")
    _style_axis(axes[1])
    source = np.asarray(play.get("source_id", ["editable"])).ravel()[0]
    fig.suptitle(
        f"Editable audit recipe • source row {source}", fontsize=15, fontweight="bold", y=1.03
    )
    return _finish(fig, output_dir, "16_real_recipe_play")


def plot_reference_parity(reference, output_dir=None):
    """Retain the raw SHAP boundary failure beside corrected/exhaustive parity."""
    keys = list(reference["model_keys"])
    rows = []
    for key in keys:
        per_split = reference["per_model_max_errors"][key]
        rows.append(
            [
                max(values["interventional_raw_phi"] for values in per_split.values()),
                max(values["interventional_corrected_phi"] for values in per_split.values()),
                max(
                    values["interventional_raw_prediction_residual"]
                    for values in per_split.values()
                ),
            ]
        )
    errors = np.asarray(rows)
    displayed = np.maximum(errors, 1e-14)
    norm = colors.LogNorm(vmin=1e-14, vmax=max(float(displayed.max()), 1e-4))
    fig, axes = _new_figure(1, 2, figsize=(14.0, 7.3), gridspec_kw={"width_ratios": [1.15, 1]})
    image = axes[0].imshow(displayed, aspect="auto", cmap="magma", norm=norm)
    labels = [
        f"{_parse_model_key(key)[0]:6s}  {_parse_model_key(key)[1]:8s}  "
        f"seed {_parse_model_key(key)[2]+1}"
        for key in keys
    ]
    axes[0].set_yticks(np.arange(len(keys)), labels, fontfamily="monospace", fontsize=8)
    axes[0].set_xticks(
        np.arange(3),
        ["Raw local ϕ", "Corrected local ϕ", "Raw prediction\nresidual"],
        fontsize=9,
    )
    for row in range(len(keys)):
        for column in range(3):
            axes[0].text(
                column,
                row,
                f"{errors[row, column]:.1e}",
                ha="center",
                va="center",
                fontsize=6.5,
                color="white" if norm(displayed[row, column]) < 0.55 else "#111827",
            )
    axes[0].set_title("Maximum error for every fitted model", loc="left", fontweight="bold")
    for spine in axes[0].spines.values():
        spine.set_visible(False)
    fig.colorbar(
        image, ax=axes[0], fraction=0.035, pad=0.025, label="Absolute error (MPa, log scale)"
    )

    affected = reference["raw_affected_rows"]
    model_index = np.asarray(affected["model_index"], dtype=int)
    row_error = np.asarray(affected["phi_max_abs_error"], dtype=float)
    split = np.asarray(affected["split"], dtype=int)
    prediction_failed = np.asarray(
        affected.get("prediction_failed", np.zeros(len(row_error), dtype=bool)), dtype=bool
    )
    for split_index, name, color in ((0, "Explanation", RED), (1, "Audit", ORANGE)):
        keep = split == split_index
        axes[1].scatter(
            model_index[keep],
            row_error[keep],
            s=24,
            marker="o" if split_index == 0 else "s",
            color=color,
            edgecolor="none",
            alpha=0.38,
            label=name,
            zorder=2,
        )
    axes[1].scatter(
        model_index[prediction_failed],
        row_error[prediction_failed],
        s=72,
        marker="X",
        color="#111827",
        edgecolor="white",
        linewidth=0.8,
        label="Observed route-flip residual",
        zorder=4,
    )
    axes[1].axhline(2e-5, color=SLATE, linestyle="--", lw=1.3, label="Compiled atol 2×10⁻⁵")
    axes[1].set_yscale("log")
    axes[1].set_xticks(np.arange(len(keys)), [str(index + 1) for index in range(len(keys))])
    axes[1].set_xlabel("Model index in the frozen 18-fit grid")
    axes[1].set_ylabel("Raw reference error (MPa)")
    axes[1].set_title(
        f"Raw failures retained: {len(row_error)} model-row pairs",
        loc="left",
        fontweight="bold",
    )
    _style_axis(axes[1])
    axes[1].legend(
        frameon=False,
        fontsize=8,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
    )
    source_ids = np.asarray(affected["source_id"], dtype=int)
    predeclared = prediction_failed & np.isin(source_ids, [139, 162])
    if predeclared.any():
        point = np.flatnonzero(predeclared)[0]
        axes[1].annotate(
            "Diagnostic example:\nsource 139 / 162",
            (model_index[point], row_error[point]),
            xytext=(1.5, 0.03),
            textcoords="data",
            arrowprops=dict(arrowstyle="->", color=SLATE, lw=0.9),
            fontsize=8,
            color=SLATE,
        )
    raw_max = reference["max_errors"]["interventional_raw_phi"]
    corrected_max = reference["max_errors"]["interventional_corrected_phi"]
    exact_max = reference["exact_max_errors"]["interventional_phi"]
    axes[1].text(
        0.02,
        0.03,
        "Raw SHAP 0.49.1: FAIL"
        f"  (max {raw_max:.2e} MPa)\n"
        "Threshold-compatible copy: PASS"
        f"  (max {corrected_max:.2e} MPa)\n"
        "Corrected compiled ↔ exhaustive subset: PASS"
        f"  (max {exact_max:.2e} MPa)",
        transform=axes[1].transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=.45", facecolor="white", edgecolor=GRID, alpha=0.95),
    )
    fig.suptitle(
        "A reference failure is evidence, not a tolerance adjustment",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    return _finish(fig, output_dir, "17_reference_parity")


def save_figure_set(
    diagnostics,
    prepared,
    evaluated,
    canonical_play,
    editable_play,
    feature_names,
    output_dir,
    model_key="forest_clean_0",
    reference=None,
):
    """Save the complete frozen figure set and return its ordered file paths."""
    mechanism, timing = diagnostics["mechanism"], diagnostics["timing"]
    calls = [
        (plot_missing_feature_games, ()),
        (plot_tiny_tree, (mechanism["tiny_play"],)),
        (plot_cardinality_weights, (mechanism["cardinality"],)),
        (plot_correlated_games, (mechanism["correlated"],)),
        (plot_repeated_feature, (mechanism["repeated_feature"],)),
        (plot_saabas_control, (mechanism["correlated"],)),
        (plot_performance, (evaluated,)),
        (plot_waterfall, (canonical_play, feature_names)),
        (plot_attribution_heatmaps, (prepared["explanation"][model_key], feature_names)),
        (plot_model_variation, (prepared["explanation"], feature_names)),
        (plot_background_sensitivity, (prepared, evaluated, feature_names)),
        (plot_deletion_curves, (evaluated,)),
        (plot_runtime_dummy_features, (timing,)),
        (plot_runtime_depth, (timing,)),
        (plot_tiny_tree_play, (mechanism["tiny_play"],)),
        (plot_real_recipe_play, (editable_play, feature_names)),
    ]
    output_dir = Path(output_dir)
    for function, args in calls:
        figure = function(*args, output_dir=output_dir)
        plt.close(figure)
    if reference is not None:
        figure = plot_reference_parity(reference, output_dir=output_dir)
        plt.close(figure)
    return sorted(output_dir.glob("*.png"))
