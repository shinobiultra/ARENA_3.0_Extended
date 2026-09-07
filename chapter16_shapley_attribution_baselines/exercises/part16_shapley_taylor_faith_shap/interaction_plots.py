"""Pure Matplotlib views for the Shapley interaction learner arc.

Plot functions accept explicit arrays and metadata. They never fit models,
select rows or pairs, read caches, or compute a taught interaction index. A
payload labeled ``schema_fixture`` is watermarked so adapter tests cannot be
mistaken for native evidence.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch
import numpy as np


ST_COLOR = "#16828f"
FAITH_COLOR = "#3569b0"
SII_COLOR = "#7a5195"
CONTROL_COLOR = "#c44e52"
ORANGE = "#d18f00"
DARK = "#17202a"
GRAY = "#6b7280"
LIGHT_GRAY = "#e5e7eb"
METHOD_COLORS = (SII_COLOR, ST_COLOR, FAITH_COLOR)
METHOD_LABELS = ("pair SII only", "Shapley-Taylor", "Faith-Shap")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping of explicit arrays and metadata.")
    return value


def _keys(value: Mapping[str, Any], name: str, required: Sequence[str]) -> None:
    missing = [key for key in required if key not in value]
    if missing:
        raise KeyError(f"{name} is missing required keys: {', '.join(missing)}")


def _array(value: Any, name: str, *, ndim: int | None = None) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    result = np.asarray(value)
    if ndim is not None and result.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions; got {result.shape}.")
    if result.dtype.kind not in "biuf":
        raise TypeError(f"{name} must be numerical.")
    if result.size and not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values.")
    return result


def _names(value: Any, count: int, name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of {count} labels.")
    result = tuple(str(item) for item in value)
    if len(result) != count or any(not item for item in result):
        raise ValueError(f"{name} must contain {count} nonempty labels.")
    return result


def _units(payload: Mapping[str, Any]) -> str:
    units = payload.get("target_unit", payload.get("units", payload.get("unit")))
    if isinstance(units, Mapping):
        for key in ("interaction", "attribution", "target", "game"):
            if key in units:
                units = units[key]
                break
    if not isinstance(units, str) or not units.strip():
        raise ValueError("Native plot payloads must declare interaction/output units.")
    return units.strip()


def _fixture_banner(fig: Figure, payload: Mapping[str, Any]) -> None:
    kind = str(payload.get("evidence_kind", payload.get("payload_kind", ""))).lower()
    if kind == "schema_fixture":
        fig.text(
            0.995,
            0.995,
            "SCHEMA FIXTURE — NO SCIENTIFIC RESULT",
            ha="right",
            va="top",
            color=CONTROL_COLOR,
            fontsize=6.5,
            fontweight="bold",
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.9,
                "pad": 1.5,
            },
        )


def _footer(fig: Figure, text: str) -> None:
    """Add a layout-managed evidence caption below every panel."""

    fig.supxlabel(text, color=GRAY, fontsize=9)


def _pair_indices(payload: Mapping[str, Any], n_features: int) -> np.ndarray:
    value = payload.get("pair_indices")
    if value is None:
        value = tuple(itertools.combinations(range(n_features), 2))
    pairs = _array(value, "pair_indices", ndim=2).astype(np.int64, copy=False)
    expected = n_features * (n_features - 1) // 2
    if pairs.shape != (expected, 2):
        raise ValueError(f"pair_indices must have shape ({expected},2).")
    if len({tuple(row) for row in pairs.tolist()}) != expected:
        raise ValueError("pair_indices must contain each pair exactly once.")
    if np.any(pairs[:, 0] >= pairs[:, 1]) or pairs.min() < 0 or pairs.max() >= n_features:
        raise ValueError("pair_indices must be unique increasing valid feature pairs.")
    return pairs


def _pair_masks(pairs: np.ndarray) -> np.ndarray:
    return np.asarray([(1 << int(a)) | (1 << int(b)) for a, b in pairs], dtype=np.int64)


def _method_pair_arrays(primary: Mapping[str, Any], pairs: np.ndarray) -> tuple[np.ndarray, ...]:
    _keys(primary, "primary64", ("sii_pairs", "stii_coefficients", "fsii_coefficients"))
    sii = _array(primary["sii_pairs"], "sii_pairs", ndim=3)
    st = _array(primary["stii_coefficients"], "stii_coefficients", ndim=3)
    faith = _array(primary["fsii_coefficients"], "fsii_coefficients", ndim=3)
    if st.shape != faith.shape or st.shape[-1] != 256:
        raise ValueError("STII and FSII primary coefficients must share shape [models,rows,256].")
    if sii.shape != (*st.shape[:2], len(pairs)):
        raise ValueError("SII pairs must share model/row axes and contain every pair.")
    masks = _pair_masks(pairs)
    return sii, st[..., masks], faith[..., masks]


def _primary(evaluated: Mapping[str, Any]) -> Mapping[str, Any]:
    if "all_rows" in evaluated:
        all_rows = _mapping(evaluated["all_rows"], "evaluated['all_rows']")
        return _mapping(all_rows.get("primary64"), "evaluated['all_rows']['primary64']")
    if "primary64" in evaluated:
        return _mapping(evaluated["primary64"], "evaluated['primary64']")
    if "primary" in evaluated:
        return _mapping(evaluated["primary"], "evaluated['primary']")
    raise KeyError("evaluated must contain the primary64 empirical branch.")


def _audit_indices(evaluated: Mapping[str, Any], rows: int) -> np.ndarray:
    split = _array(evaluated.get("row_split"), "row_split", ndim=1).astype(np.int64, copy=False)
    if split.shape != (rows,) or not set(np.unique(split)).issubset({0, 1}):
        raise ValueError("row_split must be a binary vector aligned with all rows.")
    selected = np.flatnonzero(split == 1)
    if not len(selected):
        raise ValueError("row_split must contain audit rows.")
    return selected


def _group_average(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Average rows within group, then average groups equally."""

    groups = np.asarray(groups)
    if groups.ndim != 1 or values.shape[-2] != len(groups):
        raise ValueError("recipe_group_ids must align with the row axis.")
    unique = np.unique(groups)
    grouped = np.stack([values[..., groups == group, :].mean(axis=-2) for group in unique], axis=-2)
    return grouped.mean(axis=-2)


def _coefficient_terms(coefficients: np.ndarray, pairs: np.ndarray, n_features: int) -> np.ndarray:
    singles = np.asarray([1 << index for index in range(n_features)], dtype=np.int64)
    return coefficients[..., np.concatenate((singles, _pair_masks(pairs)))]


def _term_labels(feature_names: Sequence[str], pairs: np.ndarray) -> tuple[str, ...]:
    return tuple(feature_names) + tuple(f"{feature_names[a]} × {feature_names[b]}" for a, b in pairs)


def _symmetric_limit(*arrays: np.ndarray, floor: float = 1e-12) -> float:
    maximum = max((float(np.max(np.abs(array))) if array.size else 0.0) for array in arrays)
    return max(maximum, floor)


def _annotate_bars(ax, bars, labels: Sequence[str]) -> None:
    for bar, label in zip(bars, labels, strict=True):
        value = bar.get_height()
        offset = 4 if value >= 0 else -12
        ax.annotate(
            label,
            (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=8,
            fontweight="bold",
        )


def _exact_payload(diagnostic: Mapping[str, Any], *, reconstruction: bool = False) -> Mapping[str, Any]:
    diagnostic = _mapping(diagnostic, "exact_diagnostics")
    required = [
        "evidence_kind",
        "game_names",
        "game_values",
        "single_masks",
        "pair_masks",
        "ordinary_shapley",
        "sii_pairs",
        "stii",
        "fsii",
        "exact_fraction_labels",
    ]
    if reconstruction:
        required += ["reconstruction", "kernel_mse", "rmse_by_size"]
    _keys(diagnostic, "exact_diagnostics", required)
    if diagnostic["evidence_kind"] != "analytic_exact":
        raise ValueError("The cold-open plots require actual analytic exact arrays.")
    return diagnostic


def plot_exact_allocation_ledger(exact_diagnostics: Mapping[str, Any]) -> Figure:
    """Plot the pure-four STII/FSII allocation and separate SII comparator."""

    payload = _exact_payload(exact_diagnostics)
    names = tuple(payload["game_names"])
    row = names.index("pure_four")
    singles = _array(payload["single_masks"], "single_masks", ndim=1).astype(int)
    pairs = _array(payload["pair_masks"], "pair_masks", ndim=1).astype(int)
    st = _array(payload["stii"], "stii", ndim=2)[row]
    faith = _array(payload["fsii"], "fsii", ndim=2)[row]
    shapley = _array(payload["ordinary_shapley"], "ordinary_shapley", ndim=2)[row]
    sii = _array(payload["sii_pairs"], "sii_pairs", ndim=2)[row]
    np.testing.assert_allclose(st[singles], 0, rtol=0, atol=1e-10)
    np.testing.assert_allclose(st[pairs], 1 / 6, rtol=0, atol=1e-10)
    np.testing.assert_allclose(faith[singles], -1 / 5, rtol=0, atol=1e-10)
    np.testing.assert_allclose(faith[pairs], 3 / 10, rtol=0, atol=1e-10)
    np.testing.assert_allclose(shapley, 1 / 4, rtol=0, atol=1e-10)
    np.testing.assert_allclose(sii, 1 / 3, rtol=0, atol=1e-10)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.6), constrained_layout=True)
    feature_labels = ("A", "B", "C", "D")
    pair_labels = ("AB", "AC", "AD", "BC", "BD", "CD")

    x = np.arange(4)
    width = 0.36
    bars_st = axes[0].bar(x - width / 2, st[singles], width, color=ST_COLOR, label="Shapley-Taylor")
    bars_faith = axes[0].bar(x + width / 2, faith[singles], width, color=FAITH_COLOR, label="Faith-Shap")
    _annotate_bars(axes[0], bars_st, ["0"] * 4)
    _annotate_bars(axes[0], bars_faith, ["-1/5"] * 4)
    axes[0].axhline(0, color=DARK, linewidth=0.8)
    axes[0].set_xticks(x, feature_labels)
    axes[0].set_ylabel("singleton credit")
    axes[0].set_title("Lower-order terms", loc="left", fontweight="bold")
    axes[0].legend(frameon=False, loc="lower left")
    axes[0].set_ylim(-0.28, 0.12)

    x = np.arange(6)
    bars_st = axes[1].bar(x - width / 2, st[pairs], width, color=ST_COLOR)
    bars_faith = axes[1].bar(x + width / 2, faith[pairs], width, color=FAITH_COLOR)
    _annotate_bars(axes[1], bars_st, ["1/6"] * 6)
    _annotate_bars(axes[1], bars_faith, ["3/10"] * 6)
    axes[1].set_xticks(x, pair_labels)
    axes[1].set_ylabel("pair credit")
    axes[1].set_ylim(0, 0.39)
    axes[1].set_title("The unit dividend reaches every pair", loc="left", fontweight="bold")
    axes[1].text(0.02, 0.96, "ST: 6 × 1/6 = 1", transform=axes[1].transAxes, va="top", color=ST_COLOR)
    axes[1].text(
        0.02,
        0.88,
        "Faith: 4 × (-1/5) + 6 × 3/10 = 1",
        transform=axes[1].transAxes,
        va="top",
        color=FAITH_COLOR,
    )

    ax = axes[2]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.02, 0.49), 0.96, 0.39, boxstyle="round,pad=0.02", facecolor="#f5f3ff", edgecolor=SII_COLOR))
    ax.text(0.06, 0.82, "SII comparator from 16.3", color=SII_COLOR, fontweight="bold", fontsize=11)
    ax.text(0.06, 0.71, "ordinary Shapley singleton: 1/4 each", fontsize=10)
    ax.text(0.06, 0.62, "pair SII: 1/3 each", fontsize=10)
    ax.text(0.06, 0.53, "Do not add these as one order-2 decomposition.", color=CONTROL_COLOR, fontweight="bold", fontsize=9)
    ax.add_patch(FancyBboxPatch((0.02, 0.08), 0.96, 0.27, boxstyle="round,pad=0.02", facecolor="#eef8f7", edgecolor=ST_COLOR))
    ax.text(0.06, 0.29, "Agreement control", color=ST_COLOR, fontweight="bold", fontsize=11)
    ax.text(0.06, 0.20, "If the game has degree ≤ 2, STII = FSII", fontsize=9)
    ax.text(0.06, 0.13, "and pair SII matches the pair terms.", fontsize=9)
    ax.set_title("A different axiomatic target", loc="left", fontweight="bold")

    fig.suptitle("One four-way mechanism, three different pair answers", fontsize=16, fontweight="bold")
    _footer(fig, "Pure four-player unanimity game · maximum order 2 · exact fractions")
    return fig


def plot_reconstruction_reason(exact_diagnostics: Mapping[str, Any]) -> Figure:
    """Show why Faith-Shap moves singleton coefficients on pure four-way credit."""

    payload = _exact_payload(exact_diagnostics, reconstruction=True)
    names = tuple(payload["game_names"])
    row = names.index("pure_four")
    degree_two = names.index("mixed_degree2")
    game = _array(payload["game_values"], "game_values", ndim=2)[row]
    reconstruction = _array(payload["reconstruction"], "reconstruction", ndim=3)[:, row]
    kernel = _array(payload["kernel_mse"], "kernel_mse", ndim=2)
    sizes = np.asarray([mask.bit_count() for mask in range(16)])
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), constrained_layout=True, gridspec_kw={"width_ratios": (1.7, 1)})
    jitter = np.linspace(-0.07, 0.07, 16)
    for values, color, label, marker in [
        (game, DARK, "true game", "o"),
        (reconstruction[0], ST_COLOR, "Shapley-Taylor reconstruction", "s"),
        (reconstruction[1], FAITH_COLOR, "Faith-Shap reconstruction", "D"),
    ]:
        axes[0].scatter(sizes + jitter, values, s=34, color=color, alpha=0.72, marker=marker, label=label)
        means = [values[sizes == size].mean() for size in range(5)]
        axes[0].plot(range(5), means, color=color, linewidth=2)
    axes[0].scatter([0, 4], [0, 1], s=150, facecolors="none", edgecolors=ORANGE, linewidths=2, label="exact endpoint constraints")
    axes[0].axhline(0, color=GRAY, linewidth=0.8)
    axes[0].set_xticks(range(5))
    axes[0].set_xlabel("coalition size")
    axes[0].set_ylabel("game / reconstructed value")
    axes[0].set_title("Faith-Shap lowers interior predictions before pairs accumulate", loc="left", fontweight="bold")
    axes[0].legend(frameon=False, fontsize=8)

    values = kernel[:, row]
    bars = axes[1].bar([0, 1], values, color=[ST_COLOR, FAITH_COLOR], width=0.6)
    for bar, value in zip(bars, values, strict=True):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value, f"{value:.4f}", ha="center", va="bottom", fontweight="bold")
    axes[1].set_xticks([0, 1], ["Shapley-Taylor", "Faith-Shap"])
    axes[1].set_ylabel("normalized Shapley-kernel interior MSE")
    axes[1].set_title("The declared objective", loc="left", fontweight="bold")
    axes[1].set_ylim(0, float(values.max()) * 1.30)
    axes[1].text(
        0.5,
        0.93,
        f"degree-2 control: ST {kernel[0, degree_two]:.1e}, Faith {kernel[1, degree_two]:.1e}",
        transform=axes[1].transAxes,
        ha="center",
        va="top",
        color=GRAY,
        fontsize=9,
    )
    fig.suptitle("Negative singleton credit improves the declared order-two fit", fontsize=16, fontweight="bold")
    _footer(fig, "Exact pure-four game · locked empty/full endpoints · lower error is kernel-specific")
    return fig


def plot_native_game_and_controls(evaluated: Mapping[str, Any]) -> Figure:
    """Plot predictive context, intact-donor semantics, and additive controls.

    The seven audit RMSE values, donor-game dimensions, ridge pair control, and
    direct-forward lookup control are read from the completed array payload.
    """

    payload = _mapping(evaluated, "evaluated")
    _keys(
        payload,
        "evaluated",
        (
            "model_keys",
            "feature_names",
            "target_unit",
            "source_ids",
            "performance",
            "all_rows",
            "direct_full_context",
        ),
    )
    labels = _names(payload["model_keys"], 7, "model_keys")
    performance = _mapping(payload["performance"], "performance")
    audit_performance = _mapping(performance.get("audit"), "performance.audit")
    mse = _array(audit_performance.get("mse"), "performance.audit.mse", ndim=1)
    if mse.shape != (7,) or np.any(mse < 0):
        raise ValueError("Audit MSE must contain one nonnegative value per known model.")
    metrics = np.sqrt(mse)
    feature_names = _names(payload["feature_names"], 8, "feature_names")
    unit = _units(payload)
    primary = _primary(payload)
    pair_values = _array(primary.get("pair_values"), "primary64.pair_values", ndim=4)
    if pair_values.shape != (7, len(_array(payload["source_ids"], "source_ids", ndim=1)), 3, 28):
        raise ValueError("Primary pair values must retain all models, rows, methods, and pairs.")
    donor_ids = _array(primary.get("donor_source_ids"), "primary64.donor_source_ids", ndim=1)
    direct = _mapping(payload["direct_full_context"], "direct_full_context")
    lookup = _array(direct.get("lookup_difference"), "direct_full_context.lookup_difference", ndim=3)
    ridge_pair_max = float(np.max(np.abs(pair_values[6])))
    direct_max = float(np.max(np.abs(lookup)))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.3), constrained_layout=True)
    colors = [ST_COLOR] * 3 + [CONTROL_COLOR] * 3 + [GRAY]
    axes[0].barh(np.arange(7), metrics, color=colors)
    axes[0].set_yticks(np.arange(7), labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel(f"audit RMSE ({unit})")
    axes[0].set_title("Known-model predictive context", loc="left", fontweight="bold")

    ax = axes[1]
    ax.axis("off")
    for index, name in enumerate(feature_names):
        y = 0.92 - index * 0.105
        source = f"x[{index}] if {index} ∈ S; z_b[{index}] otherwise"
        ax.add_patch(
            FancyBboxPatch(
                (0.03, y - 0.035),
                0.94,
                0.07,
                boxstyle="round,pad=0.01",
                facecolor="#f8fafc",
                edgecolor=GRAY,
            )
        )
        ax.text(0.06, y, name, va="center", fontweight="bold", fontsize=8)
        ax.text(0.53, y, source, va="center", color=DARK, fontsize=7.5)
    ax.text(
        0.5,
        0.035,
        "the same donor b supplies every absent coordinate",
        ha="center",
        color=ORANGE,
        fontsize=8,
        fontweight="bold",
    )
    ax.set_title("One donor replacement, then average donors", loc="left", fontweight="bold")

    ax = axes[2]
    ax.axis("off")
    cards = [
        ("ridge max |pair|", ridge_pair_max, unit, "additive zero-pair control"),
        ("direct vs table", direct_max, unit, "donor-averaged sign/index control"),
    ]
    for card, (title, value, card_unit, detail) in enumerate(cards):
        y = 0.68 - card * 0.38
        ax.add_patch(FancyBboxPatch((0.05, y), 0.9, 0.25, boxstyle="round,pad=0.025", facecolor="#f8fafc", edgecolor=GRAY))
        ax.text(0.1, y + 0.18, title, fontweight="bold")
        ax.text(0.1, y + 0.10, f"{value:.3e} {card_unit}", fontsize=13, color=ST_COLOR)
        ax.text(0.1, y + 0.04, detail, fontsize=8, color=GRAY)
    ax.set_title("Controls are part of the result", loc="left", fontweight="bold")
    fig.suptitle("The native game is finite, empirical, and background-dependent", fontsize=16, fontweight="bold")
    _footer(
        fig,
        f"7 models × {len(payload['source_ids'])} rows × 256 coalitions × {len(donor_ids)} intact donors · output {unit}",
    )
    _fixture_banner(fig, payload)
    return fig


def _draw_interaction_graph(ax, nodes: np.ndarray, edges: np.ndarray, feature_names: Sequence[str], limit: float, title: str) -> None:
    n_features = len(feature_names)
    theta = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, n_features, endpoint=False)
    positions = np.c_[np.cos(theta), np.sin(theta)]
    pair_index = 0
    for first, second in itertools.combinations(range(n_features), 2):
        value = edges[pair_index]
        pair_index += 1
        color = ST_COLOR if value >= 0 else CONTROL_COLOR
        width = 0.35 + 4.0 * abs(value) / limit
        ax.plot(*positions[[first, second]].T, color=color, linewidth=width, alpha=0.25 + 0.65 * abs(value) / limit, zorder=1)
    norm = mcolors.TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    ax.scatter(positions[:, 0], positions[:, 1], c=nodes, cmap="coolwarm", norm=norm, s=730, edgecolors="white", linewidths=1.5, zorder=3)
    for index, name in enumerate(feature_names):
        ax.text(positions[index, 0] * 1.22, positions[index, 1] * 1.22, name, ha="center", va="center", fontsize=7.5, fontweight="bold")
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.35, 1.35)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, loc="left", fontweight="bold")


def plot_fixed_row_interaction_graph(native_play: Mapping[str, Any], feature_names: Sequence[str]) -> Figure:
    """Plot STII/FSII singleton-node and pair-edge allocations for fixed source 0."""

    payload = _mapping(native_play, "native_play")
    fixed = _mapping(payload.get("fixed_row"), "native_play.fixed_row")
    _keys(fixed, "native_play.fixed_row", ("model_key", "source_id", "background_key", "donor_count", "order", "stii_coefficients", "fsii_coefficients", "sii_pairs", "pair_indices", "target_unit"))
    feature_names = _names(feature_names, 8, "feature_names")
    if str(fixed["model_key"]) != "clean_0" or int(fixed["source_id"]) != 0 or str(fixed["background_key"]) != "primary64":
        raise ValueError("The fixed scientific view is clean_0, source 0, primary64.")
    if int(fixed["order"]) != 2 or int(fixed["donor_count"]) != 64:
        raise ValueError("The fixed scientific view requires order two and 64 donors.")
    unit = _units(fixed)
    pairs = _pair_indices(fixed, 8)
    st = _array(fixed["stii_coefficients"], "stii_coefficients", ndim=1)
    faith = _array(fixed["fsii_coefficients"], "fsii_coefficients", ndim=1)
    sii = _array(fixed["sii_pairs"], "sii_pairs", ndim=1)
    if st.shape != (256,) or faith.shape != (256,) or sii.shape != (28,):
        raise ValueError("Fixed-row coefficients require full [256] vectors and [28] SII pairs.")
    singles = np.asarray([1 << index for index in range(8)])
    masks = _pair_masks(pairs)
    limit = _symmetric_limit(st[np.r_[singles, masks]], faith[np.r_[singles, masks]])
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
    _draw_interaction_graph(axes[0, 0], st[singles], st[masks], feature_names, limit, "Shapley-Taylor")
    _draw_interaction_graph(axes[0, 1], faith[singles], faith[masks], feature_names, limit, "Faith-Shap")
    fig.colorbar(
        plt.cm.ScalarMappable(norm=mcolors.TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), cmap="coolwarm"),
        ax=axes[0, :], orientation="horizontal", shrink=0.7, pad=0.04,
        label=f"Singleton credit ({unit}): blue negative, red positive",
    )
    for graph_ax in axes[0, :]:
        graph_ax.text(0.5, -0.02, "Edges: teal positive · red negative · wider means larger |pair credit|",
                      ha="center", va="top", transform=graph_ax.transAxes, fontsize=7.5)
    diff = np.concatenate((faith[singles] - st[singles], faith[masks] - st[masks]))
    term_labels = _term_labels(feature_names, pairs)
    diff_limit = _symmetric_limit(diff)
    difference_ax = axes[1, 0]
    difference_ax.barh(np.arange(36), diff, color=np.where(diff >= 0, FAITH_COLOR, CONTROL_COLOR))
    difference_ax.axvline(0, color=DARK, linewidth=0.7)
    difference_ax.set_xlim(-1.08 * diff_limit, 1.08 * diff_limit)
    difference_ax.set_yticks(range(36), term_labels, fontsize=6.5)
    difference_ax.invert_yaxis()
    difference_ax.set_xlabel(f"Faith-Shap - Shapley-Taylor ({unit})")
    difference_ax.set_title("All 36 retained singleton and pair terms", loc="left", fontweight="bold", fontsize=10)
    sii_ax = axes[1, 1]
    sii_ax.barh(np.arange(28), sii, color=SII_COLOR)
    sii_ax.axvline(0, color=DARK, linewidth=0.7)
    sii_ax.set_yticks(np.arange(28), [f"{feature_names[a]} × {feature_names[b]}" for a, b in pairs], fontsize=6.5)
    sii_ax.invert_yaxis()
    sii_ax.set_xlabel(unit)
    sii_ax.set_title("Pair SII only (not an efficient ST/FS expansion)", loc="left", fontweight="bold", fontsize=10)
    ridge = fixed.get("ridge_pair_max_abs")
    control = "ridge control unavailable"
    if ridge is not None:
        control = f"ridge max |pair| = {float(ridge):.2e} {unit}"
    fig.suptitle("Fixed-row interaction allocation", fontsize=16, fontweight="bold")
    _footer(fig, f"clean_0 · explanation source 0 · primary64 · 64 donors · order 2 · {control}")
    _fixture_banner(fig, payload)
    return fig


def plot_allrow_pair_heatmaps(evaluated: Mapping[str, Any], feature_names: Sequence[str]) -> Figure:
    """Show signed and absolute audit summaries for every model, method, and pair."""

    payload = _mapping(evaluated, "evaluated")
    _keys(payload, "evaluated", ("model_keys", "row_split", "recipe_group_ids", "target_unit"))
    feature_names = _names(feature_names, 8, "feature_names")
    model_keys = _names(payload["model_keys"], 7, "model_keys")
    pairs = _pair_indices(payload, 8)
    primary = _primary(payload)
    arrays = _method_pair_arrays(primary, pairs)
    rows = arrays[0].shape[1]
    audit = _audit_indices(payload, rows)
    groups_all = _array(payload["recipe_group_ids"], "recipe_group_ids", ndim=1)
    if groups_all.shape != (rows,):
        raise ValueError("recipe_group_ids must align with all primary rows.")
    groups = groups_all[audit]
    signed = np.stack([_group_average(values[:, audit, :], groups) for values in arrays])
    absolute = np.stack([_group_average(np.abs(values[:, audit, :]), groups) for values in arrays])
    signed_limit = _symmetric_limit(signed)
    absolute_limit = max(float(absolute.max()), 1e-12)
    pair_labels = [f"{feature_names[a]} × {feature_names[b]}" for a, b in pairs]

    fig, axes = plt.subplots(2, 4, figsize=(17, 13.5), constrained_layout=True, sharey=True)
    for method in range(3):
        image = axes[0, method].imshow(signed[method].T, cmap="coolwarm", vmin=-signed_limit, vmax=signed_limit, aspect="auto")
        axes[0, method].set_title(f"{METHOD_LABELS[method]} · signed mean", loc="left", fontweight="bold", fontsize=10)
        absolute_image = axes[1, method].imshow(absolute[method].T, cmap="viridis", vmin=0, vmax=absolute_limit, aspect="auto")
        axes[1, method].set_title(f"{METHOD_LABELS[method]} · mean absolute", loc="left", fontweight="bold", fontsize=10)
    difference = signed[2] - signed[1]
    diff_limit = _symmetric_limit(difference)
    axes[0, 3].imshow(difference.T, cmap="coolwarm", vmin=-diff_limit, vmax=diff_limit, aspect="auto")
    axes[0, 3].set_title(
        f"Faith-Shap - Shapley-Taylor\nrange ±{diff_limit:.2g} {_units(payload)}",
        loc="left",
        fontweight="bold",
        fontsize=9,
    )
    abs_difference = absolute[2] - absolute[1]
    abs_diff_limit = _symmetric_limit(abs_difference)
    axes[1, 3].imshow(abs_difference.T, cmap="coolwarm", vmin=-abs_diff_limit, vmax=abs_diff_limit, aspect="auto")
    axes[1, 3].set_title(
        f"difference in mean |pair|\nrange ±{abs_diff_limit:.2g} {_units(payload)}",
        loc="left",
        fontweight="bold",
        fontsize=9,
    )
    for row_axes in axes:
        for ax in row_axes:
            ax.set_xticks(range(7), model_keys, rotation=38, ha="right", fontsize=7)
            ax.set_yticks(range(28), pair_labels, fontsize=6)
    for ax in axes[:, 1:].ravel():
        ax.tick_params(labelleft=False)
    fig.colorbar(image, ax=axes[0, :3], shrink=0.55, label=f"signed pair interaction ({_units(payload)})")
    fig.colorbar(absolute_image, ax=axes[1, :3], shrink=0.55, label=f"mean absolute pair interaction ({_units(payload)})")
    fig.suptitle("Every audit pair and every model remain visible", fontsize=16, fontweight="bold")
    _footer(fig, f"{len(audit)} audit rows · {len(np.unique(groups))} recipe groups · all 7 models · lexicographic pair order")
    _fixture_banner(fig, payload)
    return fig


def plot_heldout_pair_order_capture(evaluated: Mapping[str, Any], feature_names: Sequence[str]) -> Figure:
    """Plot explanation-frozen audit capture curves and random-order controls."""

    payload = _mapping(evaluated, "evaluated")
    _keys(payload, "evaluated", ("model_keys", "pair_capture", "random_controls", "target_unit"))
    _names(feature_names, 8, "feature_names")
    model_keys = _names(payload["model_keys"], 7, "model_keys")
    capture = _mapping(payload["pair_capture"], "pair_capture")
    curves = _array(capture.get("capture_curves"), "capture_curves", ndim=3)
    if curves.shape != (7, 3, 29):
        raise ValueError("capture_curves must have shape [7 models,3 methods,29 prefixes].")
    random_curves = _array(capture.get("random_capture_curves"), "random_capture_curves", ndim=3)
    if random_curves.shape != (7, 256, 29):
        raise ValueError("All 256 random pair-order curves must be retained for every model.")
    q = np.arange(29)
    fig, axes = plt.subplots(2, 4, figsize=(16.5, 9.2), constrained_layout=True)
    flat = axes.ravel()
    for model, label in enumerate(model_keys):
        ax = flat[model]
        ax.plot(q, random_curves[model].T, color=LIGHT_GRAY, linewidth=0.35, alpha=0.35)
        for method, (method_label, color) in enumerate(zip(METHOD_LABELS, METHOD_COLORS, strict=True)):
            ax.plot(q, curves[model, method], color=color, linewidth=2, label=method_label)
        ax.set_title(label, loc="left", fontweight="bold", fontsize=9)
        if label == "ridge":
            ax.text(0.04, 0.94, "Additive model: normalization\nmagnifies floating-point residuals",
                    transform=ax.transAxes, va="top", fontsize=7, color=GRAY)
        ax.set_xlim(0, 28)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xticks([0, 7, 14, 21, 28])
        ax.grid(alpha=0.18)
    flat[0].legend(frameon=False, fontsize=7, loc="lower right")
    ax = flat[7]
    controls = _mapping(payload["random_controls"], "random_controls")
    within = controls.get("within_size")
    if isinstance(within, Mapping):
        observed = _mapping(within.get("observed"), "within_size.observed")
        null = _mapping(within.get("null"), "within_size.null")
        observed_top = _array(observed.get("top3_share"), "observed.top3_share", ndim=1)
        null_top = _array(null.get("top3_share"), "null.top3_share", ndim=2)
        observed_peak = _array(observed.get("peak_to_mean"), "observed.peak_to_mean", ndim=1)
        null_peak = _array(null.get("peak_to_mean"), "null.peak_to_mean", ndim=2)
        if observed_top.shape != (3,) or observed_peak.shape != (3,) or null_top.shape != (128, 3) or null_peak.shape != (128, 3):
            raise ValueError("Within-size controls must retain 128 values for both statistics and all methods.")
        for method, color in enumerate(METHOD_COLORS):
            offset = (method - 1) * 0.13
            ax.scatter(np.full(128, 0 + offset), null_top[:, method], s=7, color=color, alpha=0.23)
            ax.scatter([0 + offset], [observed_top[method]], marker="D", s=45, color=color, label=METHOD_LABELS[method])
            ax.scatter(np.full(128, 1 + offset), null_peak[:, method], s=7, color=color, alpha=0.23)
            ax.scatter([1 + offset], [observed_peak[method]], marker="D", s=45, color=color)
        ax.set_xticks([0, 1], ["top-3\nshare", "peak /\nmean"])
        ax.set_ylabel("control statistic")
        ax.set_title("128 within-size structure controls", loc="left", fontweight="bold", fontsize=9)
        ax.legend(frameon=False, fontsize=6, loc="upper center")
        ax.text(0.5, 0.86, "Diamonds: original · dots: shuffled", ha="center",
                transform=ax.transAxes, fontsize=6.5, color=GRAY)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "within-size branch\nnot present", ha="center", va="center", color=GRAY)
    for ax in axes[-1, :3]:
        ax.set_xlabel("first q pairs")
    for ax in axes[:, 0]:
        ax.set_ylabel("fraction of full-context |synergy| captured")
    fig.suptitle("Explanation-side pair orders are evaluated unchanged on audit", fontsize=16, fontweight="bold")
    _footer(fig, "All 28 pairs retained · all 256 random orders shown · full-context replacement is a model-input diagnostic")
    _fixture_banner(fig, payload)
    return fig


def plot_order_two_reconstruction(evaluated: Mapping[str, Any]) -> Figure:
    """Plot audit reconstruction RMSE and normalized kernel MSE for both methods."""

    payload = _mapping(evaluated, "evaluated")
    _keys(payload, "evaluated", ("model_keys", "row_split", "recipe_group_ids", "target_unit"))
    model_keys = _names(payload["model_keys"], 7, "model_keys")
    primary = _primary(payload)
    _keys(primary, "primary64", ("rmse_by_size", "kernel_mse", "residual"))
    rmse = _array(primary["rmse_by_size"], "rmse_by_size", ndim=4)
    kernel = _array(primary["kernel_mse"], "kernel_mse", ndim=3)
    residual = _array(primary["residual"], "residual", ndim=4)
    if rmse.shape[:3] != (2, 7, rmse.shape[2]) or rmse.shape[-1] != 9 or kernel.shape[:2] != (2, 7):
        raise ValueError("Reconstruction arrays must use [2 methods,7 models,rows,...].")
    rows = rmse.shape[2]
    if kernel.shape[2] != rows or residual.shape != (2, 7, rows, 256):
        raise ValueError("Reconstruction residual, RMSE, and kernel arrays must share axes.")
    audit = _audit_indices(payload, rows)
    groups = _array(payload["recipe_group_ids"], "recipe_group_ids", ndim=1)[audit]
    mean_rmse = _group_average(rmse[:, :, audit, :], groups)
    mean_kernel = _group_average(kernel[:, :, audit, None], groups)[..., 0]
    if np.any(kernel[1] > kernel[0] + 1e-8):
        raise ValueError("Faith-Shap kernel MSE exceeds the same-game Shapley-Taylor value.")
    endpoint_masks = [0, 255]
    if np.max(np.abs(residual[..., endpoint_masks])) > 1e-8:
        raise ValueError("Order-two efficient reconstructions must preserve empty/full endpoints.")

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), constrained_layout=True)
    styles = ["-", "--", ":", "-", "--", ":", "-."]
    for method, (label, color) in enumerate(zip(("Shapley-Taylor", "Faith-Shap"), (ST_COLOR, FAITH_COLOR), strict=True)):
        for model, model_label in enumerate(model_keys):
            axes[0].plot(range(9), mean_rmse[method, model], color=color, linestyle=styles[model], linewidth=1.4, alpha=0.82, label=f"{label} · {model_label}")
    axes[0].set_xticks(range(9))
    axes[0].set_xlabel("coalition size")
    axes[0].set_ylabel(f"audit recipe-group mean RMSE ({_units(payload)})")
    axes[0].set_title("Where the order-two surrogate misses", loc="left", fontweight="bold")
    axes[0].legend(frameon=False, fontsize=6, ncol=2)
    x = np.arange(7)
    width = 0.36
    axes[1].bar(x - width / 2, mean_kernel[0], width, color=ST_COLOR, label="Shapley-Taylor")
    axes[1].bar(x + width / 2, mean_kernel[1], width, color=FAITH_COLOR, label="Faith-Shap")
    axes[1].set_xticks(x, model_keys, rotation=35, ha="right")
    axes[1].set_ylabel(f"normalized kernel MSE ({_units(payload)} squared)")
    axes[1].set_title("Faith-Shap optimizes this declared objective", loc="left", fontweight="bold")
    axes[1].legend(frameon=False)
    axes[1].text(0.02, 0.97, "Ridge is the additive zero-pair control", transform=axes[1].transAxes, va="top", color=GRAY, fontsize=8)
    fig.suptitle("Two valid order-two reconstructions target different properties", fontsize=16, fontweight="bold")
    _footer(fig, f"{len(audit)} audit rows · every clean/shuffled seed and ridge shown · exact endpoints")
    _fixture_banner(fig, payload)
    return fig


def plot_background_changes_the_game(evaluated: Mapping[str, Any], feature_names: Sequence[str]) -> Figure:
    """Plot STII/FSII term movement across five finite donor banks."""

    payload = _mapping(evaluated, "evaluated")
    _keys(payload, "evaluated", ("all_rows", "recipe_group_ids", "target_unit"))
    feature_names = _names(feature_names, 8, "feature_names")
    pairs = _pair_indices(payload, 8)
    all_rows = _mapping(payload["all_rows"], "all_rows")
    variants = _mapping(all_rows.get("background_variants"), "all_rows.background_variants")
    order = ("primary64", "nested16", "nested128", "independent64_a", "independent64_b")
    if tuple(variants.keys()) != order:
        raise ValueError("Background variants must preserve the five frozen donor-bank order.")
    groups = _array(payload["recipe_group_ids"], "recipe_group_ids", ndim=1)
    method_movement = []
    for method_key in ("stii_coefficients", "fsii_coefficients"):
        terms = []
        for background in order:
            coefficients = _array(_mapping(variants[background], background)[method_key], f"{background}.{method_key}", ndim=2)
            if coefficients.shape != (len(groups), 256):
                raise ValueError("Every background coefficient array must have shape [272,256].")
            terms.append(_coefficient_terms(coefficients, pairs, 8))
        stacked = np.stack(terms)
        movement = stacked - stacked[0:1]
        method_movement.append(_group_average(movement, groups).T)
    labels = _term_labels(feature_names, pairs)
    limit = _symmetric_limit(*method_movement)
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 12.5), constrained_layout=True, sharey=True)
    for ax, matrix, title in zip(axes, method_movement, ("Shapley-Taylor", "Faith-Shap"), strict=True):
        image = ax.imshow(matrix, cmap="coolwarm", vmin=-limit, vmax=limit, aspect="auto")
        ax.set_xticks(range(5), ("Primary 64", "Nested 16", "Nested 128", "Independent 64 A", "Independent 64 B"), rotation=30, ha="right")
        ax.set_yticks(range(36), labels, fontsize=6)
        ax.set_title(title, loc="left", fontweight="bold")
    fig.colorbar(image, ax=axes, shrink=0.58, label=f"recipe-group mean movement from primary64 ({_units(payload)})")
    fig.suptitle("Changing the donor bank changes the game", fontsize=16, fontweight="bold")
    _footer(fig, f"clean_0 · all {len(groups)} held-out rows · all singleton and pair terms · alternate supports, not confidence replicates")
    _fixture_banner(fig, payload)
    return fig


def plot_contextual_pair_interventions(
    evaluated: Mapping[str, Any], native_play: Mapping[str, Any], feature_names: Sequence[str]
) -> Figure:
    """Compare table and direct full-context effects and trace the fixed donor."""

    payload = _mapping(evaluated, "evaluated")
    play = _mapping(native_play, "native_play")
    _keys(payload, "evaluated", ("model_keys", "direct_full_context", "target_unit"))
    feature_names = _names(feature_names, 8, "feature_names")
    model_keys = _names(payload["model_keys"], 7, "model_keys")
    pairs = _pair_indices(payload, 8)
    primary = _primary(payload)
    _keys(primary, "primary64", ("full_context_synergy",))
    stored = _array(primary["full_context_synergy"], "full_context_synergy", ndim=3)
    direct_block = _mapping(payload["direct_full_context"], "direct_full_context")
    direct_four = _array(direct_block.get("four_values"), "direct_full_context.four_values", ndim=4)
    if stored.shape != (7, stored.shape[1], 28) or direct_four.shape != (*stored.shape, 4):
        raise ValueError("Direct and table full-context arrays must align [7,rows,28].")
    direct = direct_four[..., 3] - direct_four[..., 1] - direct_four[..., 2] + direct_four[..., 0]
    limit = _symmetric_limit(stored, direct)

    fig, axes = plt.subplots(2, 4, figsize=(16, 9), constrained_layout=True)
    flat = axes.ravel()
    for model, label in enumerate(model_keys):
        ax = flat[model]
        ax.scatter(stored[model].ravel(), direct[model].ravel(), s=3, alpha=0.16, color=ST_COLOR)
        ax.plot([-limit, limit], [-limit, limit], color=DARK, linestyle="--", linewidth=1)
        error = np.max(np.abs(stored[model] - direct[model]))
        ax.set_title(f"{label}\nmax |error| {error:.2e}", loc="left", fontweight="bold", fontsize=8)
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_aspect("equal", adjustable="box")
    ax = flat[7]
    empirical = _mapping(play.get("empirical_background_average"), "native_play.empirical_background_average")
    _keys(empirical, "empirical_background_average", ("pair", "context_mask", "donor_source_ids", "four_values_by_donor"))
    pair = tuple(int(item) for item in empirical["pair"])
    if pair != (0, 1) or int(empirical["context_mask"]) != 252:
        raise ValueError("The fixed trace must use pair (0,1) and full context mask 252.")
    donor_ids = _array(empirical["donor_source_ids"], "donor_source_ids", ndim=1)
    donor_four = _array(empirical["four_values_by_donor"], "four_values_by_donor", ndim=2)
    if donor_four.shape != (4, len(donor_ids)):
        raise ValueError("four_values_by_donor must retain four predictions per donor.")
    donor_synergy = donor_four[3] - donor_four[1] - donor_four[2] + donor_four[0]
    ax.scatter(np.zeros(len(donor_synergy)), donor_synergy, s=16, alpha=0.45, color=ORANGE)
    if 173 in donor_ids:
        chosen = np.flatnonzero(donor_ids == 173)[0]
        ax.scatter([0], [donor_synergy[chosen]], marker="D", s=75, color=CONTROL_COLOR, label="fixed donor 173")
    ax.axhline(donor_synergy.mean(), color=DARK, linewidth=2, label="donor mean")
    ax.set_xlim(-0.45, 0.45)
    ax.set_xticks([0], [f"{feature_names[pair[0]]} × {feature_names[pair[1]]}"])
    ax.set_ylabel(f"full-context second difference ({_units(payload)})")
    ax.set_title("Stored intact-donor trace", loc="left", fontweight="bold", fontsize=9)
    ax.legend(frameon=False, fontsize=7)
    for ax in axes[-1, :3]:
        ax.set_xlabel("table-indexed full-context synergy")
    for ax in axes[:, 0]:
        ax.set_ylabel("independent donor-averaged direct forward")
    fig.suptitle("Direct forwards validate indexing, not an interaction definition", fontsize=16, fontweight="bold")
    _footer(fig, f"all models × all rows × all 28 pairs · fixed local pair (0,1) · output {_units(payload)}")
    _fixture_banner(fig, payload)
    return fig


def plot_dividend_play(dividend_play: Mapping[str, Any]) -> Figure:
    """Plot an editable exact game's allocations and order-limited reconstructions."""

    payload = _mapping(dividend_play, "dividend_play")
    _keys(payload, "dividend_play", ("n_players", "order", "members", "value", "game_values", "explained"))
    explained = _mapping(payload["explained"], "dividend_play.explained")
    _keys(explained, "dividend_play.explained", ("stii_coefficients", "fsii_coefficients", "ordinary_shapley", "sii_pairs", "pair_indices", "reconstruction", "kernel_mse"))
    n_players = int(payload["n_players"])
    feature_names = tuple(chr(ord("A") + index) for index in range(n_players))
    pairs = _pair_indices(explained, n_players)
    pair_masks = _pair_masks(pairs)
    st = _array(explained["stii_coefficients"], "stii_coefficients", ndim=1)
    faith = _array(explained["fsii_coefficients"], "fsii_coefficients", ndim=1)
    sii = _array(explained["sii_pairs"], "sii_pairs", ndim=1)
    if st.shape != (2**n_players,) or faith.shape != st.shape or sii.shape != (len(pairs),):
        raise ValueError("Dividend-play coefficient shapes do not match n_players.")
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.3), constrained_layout=True)
    x = np.arange(len(pairs))
    width = 0.26
    axes[0].bar(x - width, st[pair_masks], width, color=ST_COLOR, label="Shapley-Taylor")
    axes[0].bar(x, faith[pair_masks], width, color=FAITH_COLOR, label="Faith-Shap")
    axes[0].bar(x + width, sii, width, color=SII_COLOR, label="pair SII only")
    axes[0].set_xticks(x, [f"{feature_names[a]}{feature_names[b]}" for a, b in pairs])
    axes[0].set_ylabel("pair coefficient")
    axes[0].set_title("Where the edited dividend goes", loc="left", fontweight="bold")
    # Reserve space above the data so the legend cannot cover the first pair.
    axes[0].margins(y=0.24)
    axes[0].legend(frameon=False, loc="upper left", ncol=3, fontsize=8)
    game = _array(payload["game_values"], "game_values", ndim=1)
    reconstruction = _array(explained["reconstruction"], "reconstruction", ndim=2)
    if reconstruction.shape != (2, 2**n_players):
        raise ValueError("Dividend-play reconstruction must have shape [2,2**n_players].")
    st_recon, faith_recon = reconstruction
    sizes = np.asarray([mask.bit_count() for mask in range(2**n_players)])
    axes[1].scatter(sizes - 0.06, game, s=22, color=DARK, alpha=0.55, label="game")
    axes[1].scatter(sizes, st_recon, s=22, color=ST_COLOR, alpha=0.55, label="ST reconstruction")
    axes[1].scatter(sizes + 0.06, faith_recon, s=22, color=FAITH_COLOR, alpha=0.55, label="Faith reconstruction")
    axes[1].set_xticks(range(n_players + 1))
    axes[1].set_xlabel("coalition size")
    axes[1].set_ylabel("value")
    kernel = _array(explained["kernel_mse"], "kernel_mse", ndim=1)
    axes[1].set_title(f"Reconstruction · kernel MSE ST {kernel[0]:.3g}, Faith {kernel[1]:.3g}", loc="left", fontweight="bold", fontsize=10)
    axes[1].legend(frameon=False, fontsize=8)
    members = tuple(int(item) for item in payload["members"])
    fig.suptitle(f"Dividend router: members {members}, value {float(payload['value']):g}, maximum order {int(payload['order'])}", fontsize=15, fontweight="bold")
    _footer(fig, "Editable exact game · SII remains a separate pair comparator")
    return fig


def plot_donor_context_play(context_play: Mapping[str, Any], feature_names: Sequence[str]) -> Figure:
    """Trace four coalition values and donor-level second differences in the play."""

    payload = _mapping(context_play, "context_play")
    feature_names = _names(feature_names, 8, "feature_names")
    _keys(payload, "context_play", ("model_key", "source_id", "background_key", "pair", "context_mask", "four_values", "four_values_by_donor", "donor_source_ids", "pair_values", "method_keys", "target_unit", "interprets_model_input_intervention_only"))
    if payload["interprets_model_input_intervention_only"] is not True:
        raise ValueError("The donor-context play must retain its model-input limitation.")
    pair = tuple(int(item) for item in payload["pair"])
    if len(pair) != 2 or pair[0] == pair[1] or not all(0 <= item < 8 for item in pair):
        raise ValueError("pair must contain two distinct feature indices.")
    if int(payload["context_mask"]) & ((1 << pair[0]) | (1 << pair[1])):
        raise ValueError("context_mask must be disjoint from the selected pair.")
    four = _array(payload["four_values"], "four_values", ndim=1)
    donor_four = _array(payload["four_values_by_donor"], "four_values_by_donor", ndim=2)
    donor_ids = _array(payload["donor_source_ids"], "donor_source_ids", ndim=1)
    pair_values = _array(payload["pair_values"], "pair_values", ndim=1)
    method_keys = _names(payload["method_keys"], 3, "method_keys")
    if method_keys != ("sii", "stii", "fsii"):
        raise ValueError("Context play methods must remain ordered SII, STII, FSII.")
    pair_labels = ("pair SII only", "STII", "FSII")
    if four.shape != (4,) or donor_four.shape != (4, len(donor_ids)) or pair_values.shape != (3,):
        raise ValueError("Context play must retain four bank values, four values per donor, and three pair indices.")
    unit = _units(payload)
    donor_synergy = donor_four[3] - donor_four[1] - donor_four[2] + donor_four[0]
    bank_synergy = four[3] - four[1] - four[2] + four[0]
    np.testing.assert_allclose(bank_synergy, donor_synergy.mean(), rtol=1e-9, atol=1e-9)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.1), constrained_layout=True)
    bars = axes[0].bar(range(4), four, color=[GRAY, ST_COLOR, FAITH_COLOR, ORANGE])
    axes[0].set_xticks(range(4), ["T", "T+i", "T+j", "T+i+j"])
    axes[0].set_ylabel(f"donor-mean prediction ({unit})")
    axes[0].set_title("The four values in Δᵢⱼv(T)", loc="left", fontweight="bold")
    for bar, value in zip(bars, four, strict=True):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}", ha="center", va="bottom" if value >= 0 else "top", fontsize=8)
    axes[1].scatter(np.zeros(len(donor_synergy)), donor_synergy, s=18, color=ORANGE, alpha=0.45)
    axes[1].axhline(bank_synergy, color=DARK, linewidth=2, label=f"bank mean {bank_synergy:.3g}")
    if len(donor_ids) == 1:
        axes[1].scatter([0], [donor_synergy[0]], marker="D", s=75, color=CONTROL_COLOR, label=f"donor {int(donor_ids[0])}")
    axes[1].set_xlim(-0.4, 0.4)
    axes[1].set_xticks([0], ["intact donors"])
    axes[1].set_ylabel(f"contextual pair synergy ({unit})")
    axes[1].set_title("Donors can disagree before averaging", loc="left", fontweight="bold")
    axes[1].legend(frameon=False, fontsize=8)
    colors = [SII_COLOR, ST_COLOR, FAITH_COLOR]
    bars = axes[2].bar(range(3), pair_values, color=colors)
    axes[2].axhline(bank_synergy, color=DARK, linestyle="--", label="selected-context Δ")
    axes[2].set_xticks(range(3), pair_labels, rotation=22, ha="right")
    axes[2].set_ylabel(f"pair interaction ({unit})")
    axes[2].set_title("Different context averages can all be valid", loc="left", fontweight="bold")
    axes[2].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Donor-context microscope: {feature_names[pair[0]]} × {feature_names[pair[1]]}", fontsize=15, fontweight="bold")
    _footer(fig, f"{payload['model_key']} · source {int(payload['source_id'])} · {payload['background_key']} · context mask {int(payload['context_mask'])} · exploratory model-input intervention")
    _fixture_banner(fig, payload)
    return fig


__all__ = [
    "plot_allrow_pair_heatmaps",
    "plot_background_changes_the_game",
    "plot_contextual_pair_interventions",
    "plot_dividend_play",
    "plot_donor_context_play",
    "plot_exact_allocation_ledger",
    "plot_fixed_row_interaction_graph",
    "plot_heldout_pair_order_capture",
    "plot_native_game_and_controls",
    "plot_order_two_reconstruction",
    "plot_reconstruction_reason",
]
