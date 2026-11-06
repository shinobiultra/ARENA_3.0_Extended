"""Plotting helpers for [8.3] AtP*.

Only plotting and naming lives here; all of the attribution logic is in the notebook.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch as t

KINDS = ["q", "k", "v", "z"]
KIND_LABELS = {"q": "query", "k": "key", "v": "value", "z": "head output"}
KIND_COLORS = {"q": "tab:blue", "k": "tab:orange", "v": "tab:green", "z": "tab:purple"}


def node_name(kind: int | str, layer: int, head: int) -> str:
    kind = KINDS[kind] if isinstance(kind, int) else kind
    return f"{kind}{layer}.{head}"


def flat_node_names(n_layers: int, n_heads: int) -> list[str]:
    """Names in the same order as a flattened (kind, layer, head) tensor."""
    return [node_name(k, l, h) for k in range(len(KINDS)) for l in range(n_layers) for h in range(n_heads)]


def plot_estimate_vs_exact(estimates: dict[str, t.Tensor], exact: t.Tensor, title: str, highlight: list[str] | None = None):
    """
    One panel per estimate. Each estimate and `exact` have shape (4, n_layers, n_heads): node kind, layer, head.
    Points are coloured by node kind; nodes listed in `highlight` are labelled.
    """
    n_kinds, n_layers, n_heads = exact.shape
    names = np.array(flat_node_names(n_layers, n_heads)).reshape(n_kinds, n_layers, n_heads)
    fig, axes = plt.subplots(1, len(estimates), figsize=(5.2 * len(estimates), 4.6), squeeze=False)
    lim = max(float(exact.abs().max()), max(float(e.abs().max()) for e in estimates.values())) * 1.08
    for ax, (label, est) in zip(axes[0], estimates.items()):
        for k, kind in enumerate(KINDS):
            ax.scatter(exact[k].flatten(), est[k].flatten(), s=12, alpha=0.7, color=KIND_COLORS[kind], label=KIND_LABELS[kind])
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8, label="estimate = exact")
        ax.axhline(0, color="gray", lw=0.5)
        ax.axvline(0, color="gray", lw=0.5)
        for name in highlight or []:
            idx = np.argwhere(names == name)
            if len(idx):
                k, l, h = idx[0]
                ax.annotate(name, (float(exact[k, l, h]), float(est[k, l, h])), fontsize=8, xytext=(4, 4), textcoords="offset points")
        corr = np.corrcoef(exact.flatten().numpy(), est.flatten().numpy())[0, 1]
        ax.set_title(f"{label}  (Pearson r = {corr:.2f})")
        ax.set_xlabel("exact patching effect on logit diff")
        ax.set_ylabel("estimated effect")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=8, loc="upper left")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_recall_curves(curves: dict[str, t.Tensor], n_important: int, title: str, random_band: tuple | None = None):
    """
    `curves[name][k]` = fraction of the important nodes found after verifying the first k+1 nodes in that
    method's order. `random_band` = (mean, low, high) curves for random orderings.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))
    styles = ["-", "--", "-.", ":", "-"]
    for (name, curve), style in zip(curves.items(), styles):
        curve = curve.cpu().numpy()
        x = np.arange(1, len(curve) + 1)
        full = int(np.argmax(curve >= 1.0)) + 1 if curve.max() >= 1.0 else None
        lab = f"{name} (all found after {full})" if full else name
        ax.plot(x, curve, style, lw=2, label=lab)
    if random_band is not None:
        mean, lo, hi = (c.cpu().numpy() for c in random_band)
        x = np.arange(1, len(mean) + 1)
        ax.fill_between(x, lo, hi, color="gray", alpha=0.25, label="random order (5-95%)")
        ax.plot(x, mean, color="gray", lw=1)
    ax.axvline(n_important, color="k", lw=0.8, ls=":", label=f"perfect ranking ({n_important})")
    ax.set_xscale("log")
    ax.set_xlabel("number of nodes verified with exact patching (log scale)")
    ax.set_ylabel("fraction of important nodes found")
    ax.set_ylim(0, 1.03)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.show()


def plot_saturation(clean: t.Tensor, patched: t.Tensor, linear: t.Tensor, labels: list[str], title: str):
    """Bar chart of one attention row: clean pattern, exactly recomputed pattern, and first-order prediction."""
    fig, ax = plt.subplots(figsize=(10, 3.6))
    x = np.arange(len(labels))
    w = 0.28
    ax.bar(x - w, clean.cpu().numpy(), w, label="clean pattern")
    ax.bar(x, patched.cpu().numpy(), w, label="pattern after patching (exact softmax)")
    ax.bar(x + w, linear.cpu().numpy(), w, label="first-order (linear) prediction")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("attention probability")
    ax.set_title(title)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_ingredients(table: dict[str, dict[float, int]], title: str):
    """Grouped bars: nodes needed for full recall, per method, at several importance thresholds."""
    methods = list(table)
    thresholds = list(next(iter(table.values())))
    fig, ax = plt.subplots(figsize=(10, 4))
    w = 0.8 / len(methods)
    x = np.arange(len(thresholds))
    for i, m in enumerate(methods):
        vals = [table[m][th] for th in thresholds]
        bars = ax.bar(x + (i - (len(methods) - 1) / 2) * w, vals, w, label=m)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 5, str(v), ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"|exact| > {th}" for th in thresholds])
    ax.set_ylabel("nodes verified before every\nimportant node is found")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.show()


def plot_per_prompt(per_exact: t.Tensor, per_atp: t.Tensor, name: str):
    """Scatter of per-prompt exact effect vs per-prompt AtP estimate for one node."""
    fig, ax = plt.subplots(figsize=(5, 4.5))
    e, a = per_exact.cpu().numpy(), per_atp.cpu().numpy()
    ax.scatter(e, a, s=14, alpha=0.7)
    lim = max(abs(e).max(), abs(a).max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8, label="estimate = exact")
    ax.axvline(e.mean(), color="tab:red", lw=1.2, label=f"mean exact = {e.mean():.2f}")
    ax.axhline(a.mean(), color="tab:blue", lw=1.2, label=f"mean AtP = {a.mean():.2f}")
    ax.set_xlabel("exact effect on this prompt")
    ax.set_ylabel("AtP estimate on this prompt")
    ax.set_title(f"{name}: one point per prompt")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_neurons(atp: t.Tensor, exact: t.Tensor, is_top: t.Tensor, title: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    a, e, top = atp.cpu().numpy(), exact.cpu().numpy(), is_top.cpu().numpy().astype(bool)
    ax.scatter(e[~top], a[~top], s=10, alpha=0.5, color="gray", label="random neurons")
    ax.scatter(e[top], a[top], s=14, alpha=0.8, color="tab:red", label="AtP's top neurons")
    lim = max(abs(a).max(), abs(e).max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8)
    ax.set_xlabel("exact patching effect (logit diff)")
    ax.set_ylabel("AtP estimate")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
