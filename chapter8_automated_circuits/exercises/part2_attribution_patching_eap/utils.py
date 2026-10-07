"""Dataset and plotting helpers for [8.2] Attribution Patching and EAP.

Nothing in here is part of the methods you're learning: it's prompt generation and plotting code.
Every attribution, patching and circuit-evaluation function lives in the notebook.
"""

import random

import matplotlib.pyplot as plt
import numpy as np
import torch as t
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------------------------------------------------

# All names, places and objects are single GPT-2 tokens when preceded by a space.
NAMES = [
    "Mary", "John", "Tom", "James", "Dan", "Sid", "Martin", "Amy", "Anna", "Paul", "Kate", "Lisa", "Mark", "Jack",
    "Sarah", "Emma", "Mike", "Alex", "David", "Laura", "Ben", "Rachel", "Steve", "Henry", "Grace", "Peter", "Jane",
    "George", "Sam", "Max",
]
PLACES = ["store", "garden", "restaurant", "school", "hospital", "office", "house", "station", "park", "beach"]
OBJECTS = ["ring", "kiss", "bone", "basketball", "computer", "necklace", "drink", "snack", "book", "letter"]

# Every template tokenizes to the same length (15 tokens including BOS), with the two names at positions 2 and 4 and
# the second mention of the subject (S2) at position 10, so prompts can be batched without padding.
IOI_TEMPLATES = [
    "When {X} and {Y} went to the {P}, {S} gave a {O} to",
    "After {X} and {Y} went to the {P}, {S} gave a {O} to",
    "While {X} and {Y} were at the {P}, {S} gave a {O} to",
    "Friends {X} and {Y} went to the {P}. {S} gave a {O} to",
]
IOI_POSITION_LABELS = [
    "BOS", "start", "name 1", "and", "name 2", "went", "to", "the", "place", ",", "S2", "gave", "a", "object", "END",
]


def make_ioi_prompts(n: int, seed: int = 0, corruption: str = "abc") -> tuple[list[str], list[str], list[str], list[str]]:
    """
    Returns (clean_prompts, corrupt_prompts, io_names, s_names), each a list of length n.

    Clean prompts are IOI sentences like "When Mary and John went to the store, John gave a drink to", where the
    indirect object (IO) is " Mary" and the subject (S) is " John". Half the prompts have the IO first (ABBA) and half
    have it second (BABA).

    corruption:
        "abc": all three names are replaced by three fresh names (Wang et al.'s "ABC" distribution), so the corrupted
               prompt contains no information about which of the clean names is the answer.
        "s2":  only the second mention of the subject is replaced by a fresh name ("... Peter gave a drink to"), so
               the corrupted prompt has no duplicated name.
    """
    rng = random.Random(seed)
    clean, corrupt, io_names, s_names = [], [], [], []
    for i in range(n):
        template = IOI_TEMPLATES[i % len(IOI_TEMPLATES)]
        io, s, c1, c2, c3 = rng.sample(NAMES, 5)
        place, obj = rng.choice(PLACES), rng.choice(OBJECTS)
        io_first = (i // len(IOI_TEMPLATES)) % 2 == 0
        x, y = (io, s) if io_first else (s, io)
        clean.append(template.format(X=x, Y=y, S=s, P=place, O=obj))
        if corruption == "abc":
            x2, y2 = (c1, c2) if io_first else (c2, c1)
            corrupt.append(template.format(X=x2, Y=y2, S=c3, P=place, O=obj))
        elif corruption == "s2":
            corrupt.append(template.format(X=x, Y=y, S=c3, P=place, O=obj))
        else:
            raise ValueError(f"Unknown corruption {corruption!r}")
        io_names.append(" " + io)
        s_names.append(" " + s)
    return clean, corrupt, io_names, s_names


GREATER_THAN_NOUNS = [
    "war", "dynasty", "expedition", "rebellion", "voyage", "crisis", "drought", "occupation", "reign", "campaign",
    "famine", "strike", "trial", "revolution", "siege",
]


def make_greater_than_prompts(model, n: int, seed: int = 0) -> tuple[list[str], list[str], list[int]]:
    """
    Returns (clean_prompts, corrupt_prompts, years) for the greater-than task of Hanna et al. (2023).

    Clean: "The war lasted from the year 1732 to the year 17", where the model should continue with a two-digit year
    greater than 32. Corrupt: the same prompt with the start year's last two digits replaced by "01", so that every
    continuation is "greater". `years[i]` is the two-digit start year (32 in the example). We only keep prompts where
    the start year tokenizes as [century][two digits], so that every prompt has the same length.
    """
    rng = random.Random(seed)
    clean, corrupt, years = [], [], []

    def str_tokens(s: str) -> list[str]:
        return [model.to_string(tok) for tok in model.to_tokens(s, prepend_bos=False)[0]]

    while len(clean) < n:
        noun, century, year = rng.choice(GREATER_THAN_NOUNS), rng.randint(11, 17), rng.randint(2, 98)
        c = f"The {noun} lasted from the year {century}{year:02d} to the year {century}"
        k = f"The {noun} lasted from the year {century}01 to the year {century}"
        toks = str_tokens(c)
        if len(toks) != 12 or toks[7] != f"{year:02d}" or len(str_tokens(k)) != 12:
            continue
        clean.append(c)
        corrupt.append(k)
        years.append(year)
    return clean, corrupt, years


# ---------------------------------------------------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------------------------------------------------


def plot_head_heatmaps(panels: dict[str, t.Tensor], title: str, cbar_label: str = "change in logit difference"):
    """Side-by-side (layer, head) heatmaps with a shared, symmetric colour scale."""
    arrays = {name: v.detach().float().cpu().numpy() for name, v in panels.items()}
    vmax = max(np.abs(a).max() for a in arrays.values())
    fig, axes = plt.subplots(1, len(arrays), figsize=(5.8 * len(arrays), 4.8))
    axes = np.atleast_1d(axes)
    for ax, (name, arr) in zip(axes, arrays.items()):
        im = ax.imshow(arr, cmap="RdBu", vmin=-vmax, vmax=vmax, origin="lower")
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("head")
        ax.set_ylabel("layer")
        ax.set_xticks(range(arr.shape[1]))
        ax.set_yticks(range(arr.shape[0]))
    fig.colorbar(im, ax=axes.tolist(), label=cbar_label, shrink=0.85)
    fig.suptitle(title, fontsize=12)
    plt.show()


def plot_scatter(
    exact: t.Tensor,
    approx: dict[str, t.Tensor | tuple[t.Tensor, t.Tensor]],
    title: str,
    xlabel: str = "exact patching effect",
    ylabel: str = "attribution patching estimate",
    labels: list[str] | None = None,
    n_labels: int = 6,
    ax=None,
):
    """
    Scatter plot of approximate vs exact effects, one series per entry of `approx`, with the y = x line.

    Each value of `approx` is either a tensor of estimates for the points in `exact`, or an (exact, estimates) pair
    for a series with its own x values (e.g. a control group). If `labels` is given, the `n_labels` points of the
    first series with the largest exact effect are annotated.
    """
    show = ax is None
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 5))
    markers = ["o", "s", "^", "v", "D"]
    series = []
    for name, values in approx.items():
        xs, ys = values if isinstance(values, tuple) else (exact, values)
        series.append((name, xs.detach().float().cpu().flatten().numpy(), ys.detach().float().cpu().flatten().numpy()))
    for (name, xs, ys), marker in zip(series, markers):
        ax.scatter(xs, ys, s=18, alpha=0.7, marker=marker, label=name)
    all_vals = np.concatenate([np.concatenate([xs, ys]) for _, xs, ys in series])
    lo, hi = all_vals.min(), all_vals.max()
    pad = 0.05 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1, label="y = x (perfect agreement)")
    if labels is not None:
        _, xs, ys = series[0]
        for i in np.argsort(-np.abs(xs))[:n_labels]:
            ax.annotate(labels[i], (xs[i], ys[i]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    if show:
        plt.tight_layout()
        plt.show()


def plot_bars(values: dict[str, float], title: str, ylabel: str, ylim: tuple[float, float] = (-0.2, 1.05)):
    """Bar chart comparing a method (first bar, blue) against its controls (grey)."""
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    names, vals = list(values), list(values.values())
    bars = ax.bar(names, vals, color=["tab:blue"] + ["tab:gray"] * (len(vals) - 1))
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02 if v >= 0 else v - 0.07, f"{v:.2f}", ha="center")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    plt.xticks(rotation=12, ha="right")
    plt.tight_layout()
    plt.show()


def plot_layer_pos_heatmaps(panels: dict[str, t.Tensor], pos_labels: list[str], title: str):
    """Side-by-side (layer, position) heatmaps with a shared, symmetric colour scale."""
    arrays = {name: v.detach().float().cpu().numpy() for name, v in panels.items()}
    vmax = max(np.abs(a).max() for a in arrays.values())
    fig, axes = plt.subplots(1, len(arrays), figsize=(6.2 * len(arrays), 4.6))
    axes = np.atleast_1d(axes)
    for ax, (name, arr) in zip(axes, arrays.items()):
        im = ax.imshow(arr, cmap="RdBu", vmin=-vmax, vmax=vmax, origin="lower", aspect="auto")
        ax.set_title(name, fontsize=11)
        ax.set_xticks(range(len(pos_labels)))
        ax.set_xticklabels(pos_labels, rotation=60, ha="right", fontsize=8)
        ax.set_yticks(range(arr.shape[0]))
        ax.set_ylabel("layer (residual stream before block)")
        ax.set_xlabel("position")
    fig.colorbar(im, ax=axes.tolist(), label="change in logit difference", shrink=0.85)
    fig.suptitle(title, fontsize=12)
    plt.show()


def plot_paths(paths: dict[str, dict], title: str):
    """
    One panel per component. Each entry of `paths` is a dict with keys:
        alphas, values (metric change along the path), ap (slope at alpha=0), exact (value at alpha=1),
        and optionally ig (an integrated-gradients estimate of the value at alpha=1).
    """
    fig, axes = plt.subplots(1, len(paths), figsize=(4.2 * len(paths), 3.8), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, (name, p) in zip(axes, paths.items()):
        alphas = np.array(p["alphas"])
        ax.plot(alphas, p["values"], "-o", color="black", markersize=3, label="exact: metric along the path")
        ax.plot([0, 1], [0, p["ap"]], "--", color="tab:red", lw=2, label=f"AP (tangent at clean): {p['ap']:+.2f}")
        ax.plot([1], [p["exact"]], "s", color="black", markersize=8, label=f"exact patch: {p['exact']:+.2f}")
        if "ig" in p:
            ax.plot([1], [p["ig"]], "*", color="tab:green", markersize=13, label=f"IG estimate: {p['ig']:+.2f}")
        ax.axhline(0, color="gray", lw=0.6)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("α  (0 = clean activation, 1 = corrupted)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="best")
    axes[0].set_ylabel("change in logit difference")
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_ig_convergence(steps: list[int], errors: dict[str, list[float]], title: str):
    """Log-log plot of |IG estimate - exact effect| against the number of integration steps."""
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for name, errs in errors.items():
        ax.plot(steps, np.maximum(np.array(errs), 1e-4), "-o", markersize=4, label=name)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(steps)
    ax.set_xticklabels([str(s) for s in steps])
    ax.set_xlabel("number of integration steps m  (m = 1 is attribution patching)")
    ax.set_ylabel("|IG estimate − exact effect|")
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


def _node_position(name: str, n_layers: int) -> tuple[float, float]:
    if name == "input":
        return (5.5, -1.0)
    if name == "logits":
        return (5.5, n_layers)
    if name.startswith("m"):
        return (12.6, int(name[1:]) + 0.45)
    layer, head = name[1:].split("<")[0].split(".")
    return (int(head), int(layer))


def plot_circuit_graph(
    scores: t.Tensor,
    edge_mask: t.Tensor,
    sender_names: list[str],
    receiver_names: list[str],
    top_n: int,
    title: str,
    n_layers: int = 12,
    highlight: dict[str, str] | None = None,
):
    """
    Draws the `top_n` edges with the largest |score| on a layer (y) x head (x) grid. MLPs sit in the right-hand
    column, the input embedding at the bottom and the logits at the top. Edges into a head's query, key and value
    inputs are drawn with dashed, dotted and solid lines respectively. Red edges have negative scores (corrupting
    them lowers the metric, i.e. the edge supports the behaviour); blue edges have positive scores. Role labels from
    `highlight` are green for heads that appear in the drawn edges and grey for heads that don't.
    """
    scores = scores.detach().float().cpu()
    flat = scores.abs().masked_fill(~edge_mask.cpu(), -1).flatten()
    top = flat.topk(top_n).indices
    n_recv = scores.shape[1]
    max_abs = scores.abs().max().item()
    fig, ax = plt.subplots(figsize=(11, 8.5))
    used = set()
    for idx in top.tolist():
        u, v = divmod(idx, n_recv)
        s_name, r_name = sender_names[u], receiver_names[v]
        r_node = r_name.split("<")[0]
        used |= {s_name, r_node}
        x0, y0 = _node_position(s_name, n_layers)
        x1, y1 = _node_position(r_node, n_layers)
        style = {"q": "--", "k": ":", "v": "-"}.get(r_name[-2] if "<" in r_name else "v", "-")
        value = scores[u, v].item()
        ax.annotate(
            "",
            xy=(x1, y1 - 0.12),
            xytext=(x0, y0 + 0.12),
            arrowprops=dict(
                arrowstyle="-|>",
                color="tab:red" if value < 0 else "tab:blue",
                lw=0.6 + 4.5 * abs(value) / max_abs,
                alpha=0.75,
                linestyle=style,
                connectionstyle="arc3,rad=0.12",
                shrinkA=4,
                shrinkB=4,
            ),
        )
    for layer in range(n_layers):
        for head in range(12):
            name = f"a{layer}.{head}"
            ax.plot(head, layer, "o", color="black" if name in used else "lightgray", markersize=7 if name in used else 4)
            if name in used:
                ax.text(head + 0.18, layer + 0.18, name[1:], fontsize=7)
        name = f"m{layer}"
        x, y = _node_position(name, n_layers)
        ax.plot(x, y, "s", color="black" if name in used else "lightgray", markersize=8 if name in used else 5)
        if name in used:
            ax.text(x + 0.2, y, f"MLP{layer}", fontsize=7, va="center")
    for name in ["input", "logits"]:
        x, y = _node_position(name, n_layers)
        ax.plot(x, y, "D", color="black", markersize=9)
        ax.text(x + 0.25, y, name, fontsize=9, va="center")
    if highlight:
        for name, label in highlight.items():
            x, y = _node_position(name, n_layers)
            color = "darkgreen" if name in used else "silver"
            ax.text(x, y - 0.42, label, fontsize=6.5, color=color, ha="center")
    legend = [
        Line2D([0], [0], color="tab:red", lw=2, label="score < 0: corrupting the edge lowers the metric"),
        Line2D([0], [0], color="tab:blue", lw=2, label="score > 0: corrupting the edge raises the metric"),
        Line2D([0], [0], color="gray", lw=1.5, ls="--", label="into a query"),
        Line2D([0], [0], color="gray", lw=1.5, ls=":", label="into a key"),
        Line2D([0], [0], color="gray", lw=1.5, ls="-", label="into a value / MLP / logits"),
    ]
    ax.legend(handles=legend, loc="upper left", fontsize=8, bbox_to_anchor=(1.0, 1.0))
    ax.set_xticks(list(range(12)) + [12.6])
    ax.set_xticklabels([str(h) for h in range(12)] + ["MLP"])
    ax.set_yticks(range(n_layers))
    ax.set_xlabel("head")
    ax.set_ylabel("layer")
    ax.set_xlim(-0.8, 14)
    ax.set_ylim(-1.6, n_layers + 0.6)
    ax.set_title(title, fontsize=11)
    plt.tight_layout()
    plt.show()


def plot_faithfulness(panels: dict[str, dict], title: str):
    """
    One panel per task. Each entry of `panels` maps a task name to a dict with keys:
        ks (list of circuit sizes), curves (dict of method name -> faithfulness list),
        random (list of faithfulness lists, one per random seed), n_edges (total number of edges),
        and optionally legend_loc.
    """
    fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 4.4))
    axes = np.atleast_1d(axes)
    styles = ["-o", "-s", "-^", "-v"]
    for ax, (task_name, p) in zip(axes, panels.items()):
        ks = p["ks"]
        for (name, values), style in zip(p["curves"].items(), styles):
            ax.plot(ks, values, style, markersize=4, label=name)
        rand = np.array(p["random"])
        ax.plot(ks, rand.mean(0), ":", color="gray", label=f"random edges (mean of {len(rand)} seeds)")
        ax.fill_between(ks, rand.min(0), rand.max(0), color="gray", alpha=0.2)
        ax.axhline(1, color="black", lw=0.8, ls="--")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel(f"number of edges kept (out of {p['n_edges']:,})")
        ax.set_ylabel("faithfulness (0 = corrupted, 1 = clean)")
        ax.set_title(task_name, fontsize=11)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8, loc=p.get("legend_loc", "upper left"))
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.show()
