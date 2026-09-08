"""Inspectable evaluation and plotting for candidate residual-edge circuits.

These functions do not select the search threshold using held-out examples.
They report task behavior, null distributions and bounded diagnostic probes.
"""
from __future__ import annotations

from typing import Callable, Sequence

import torch as t

from .real_edges import answer_logit_differences, clean_kl_divergence


def circuit_behavior(logits: t.Tensor, clean_logits: t.Tensor, corrupt_logits: t.Tensor,
                     answer_ids: t.Tensor) -> dict:
    """Report fixed-label behavior, KL and recovery of the aggregate clean/corrupt gap."""
    current = answer_logit_differences(logits, answer_ids)
    clean = answer_logit_differences(clean_logits, answer_ids)
    corrupt = answer_logit_differences(corrupt_logits, answer_ids)
    denominator = float(clean.mean() - corrupt.mean())
    if abs(denominator) < 1e-8:
        raise ValueError("task recovery is undefined when the clean/corrupt means coincide")
    return {
        "count": len(current), "mean_logit_difference": float(current.mean()),
        "clean_mean_logit_difference": float(clean.mean()),
        "corrupt_mean_logit_difference": float(corrupt.mean()),
        "pairwise_accuracy": float((current > 0).float().mean()),
        "clean_pairwise_accuracy": float((clean > 0).float().mean()),
        "recovery": float((current.mean() - corrupt.mean()) / denominator),
        "kl_clean_to_circuit": float(clean_kl_divergence(logits, clean_logits)),
        "per_example_logit_difference": current.tolist(),
    }


def same_size_random_masks(graph, circuit_mask: t.Tensor, *, count: int = 32,
                           seed: int = 701, match_receivers: bool = False) -> list[t.Tensor]:
    """Sample uniform legal edge subsets, optionally matching receiver indegrees.

    Do not condition on being connected or performing badly. Disconnected
    random circuits are part of the declared null and remain in its scores.
    """
    if count < 1 or circuit_mask.shape != graph.legal_mask.shape:
        raise ValueError("positive sample count and a full graph mask are required")
    if bool((circuit_mask.bool() & ~graph.legal_mask).any()):
        raise ValueError("circuit contains illegal edges")
    if not bool(((circuit_mask == 0) | (circuit_mask == 1)).all()):
        raise ValueError("sample controls for a binary selected circuit")
    generator = t.Generator().manual_seed(seed)
    legal = graph.legal_mask.nonzero()
    size = int(circuit_mask.sum())
    masks = []
    for _ in range(count):
        candidate = t.zeros_like(graph.legal_mask)
        if match_receivers:
            for receiver in range(candidate.shape[1]):
                available = graph.legal_mask[:, receiver].nonzero().flatten()
                chosen = available[t.randperm(len(available), generator=generator)[:int(circuit_mask[:, receiver].sum())]]
                candidate[chosen, receiver] = True
        else:
            chosen = legal[t.randperm(len(legal), generator=generator)[:size]]
            candidate[chosen[:, 0], chosen[:, 1]] = True
        masks.append(candidate)
    return masks


def minimality_diagnostics(engine, circuit_mask: t.Tensor,
                          answer_ids: t.Tensor | Callable[[t.Tensor], t.Tensor]) -> list[dict]:
    """Delete retained edges; accept IO labels or an explicit per-example task metric."""
    task_values = answer_ids if callable(answer_ids) else lambda logits: answer_logit_differences(logits, answer_ids)
    baseline = engine(circuit_mask)
    baseline_kl = float(clean_kl_divergence(baseline, engine.clean_logits))
    baseline_diff = float(task_values(baseline).mean())
    rows = []
    for source, receiver in circuit_mask.nonzero().tolist():
        trial = circuit_mask.clone()
        trial[source, receiver] = False
        logits = engine(trial)
        row = {
            "edge": engine.graph.edge_name(source, receiver), "source": source, "receiver": receiver,
            "kl_damage": float(clean_kl_divergence(logits, engine.clean_logits)) - baseline_kl,
            "task_damage": baseline_diff - float(task_values(logits).mean()),
        }
        if not callable(answer_ids):
            row["logit_difference_damage"] = row["task_damage"]
        rows.append(row)
    return rows


def sampled_completeness_diagnostics(engine, circuit_mask: t.Tensor,
                                     answer_ids: t.Tensor | Callable[[t.Tensor], t.Tensor], *,
                                     subset_sizes: Sequence[int] = (1, 2, 4),
                                     samples_per_size: int = 24, seed: int = 702) -> list[dict]:
    """Compare C minus K and full minus K for sampled retained-edge subsets K.

    This follows the IOI completeness question about omitted backup mechanisms.
    It is a finite diagnostic, not proof over every possible subset.
    """
    if any(size < 1 for size in subset_sizes) or samples_per_size < 1:
        raise ValueError("subset sizes and sample count must be positive")
    task_values = answer_ids if callable(answer_ids) else lambda logits: answer_logit_differences(logits, answer_ids)
    edges = circuit_mask.nonzero()
    generator = t.Generator().manual_seed(seed)
    rows = []
    for size in subset_sizes:
        if size > len(edges):
            continue
        for sample in range(samples_per_size):
            selected = edges[t.randperm(len(edges), generator=generator)[:size]]
            sparse, full = circuit_mask.clone(), engine.graph.legal_mask.clone()
            sparse[selected[:, 0], selected[:, 1]] = False
            full[selected[:, 0], selected[:, 1]] = False
            sparse_diff = float(task_values(engine(sparse)).mean())
            full_diff = float(task_values(engine(full)).mean())
            rows.append({"subset_size": size, "sample": sample,
                         "removed_edges": selected.tolist(), "circuit_minus_subset": sparse_diff,
                         "full_minus_subset": full_diff, "omitted_backup_gap": full_diff - sparse_diff})
    return rows


def plot_discovered_edges(graph, active_mask: t.Tensor):
    """Provided drawing; preserve receiver-port colors and every retained edge."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyArrowPatch

    colors = {"q": "#2b8cbe", "k": "#8856a7", "v": "#e6550d",
              "mlp": "#238b45", "output": "#666666"}
    involved = {0}
    edges = []
    for source, receiver_index in active_mask.nonzero().tolist():
        receiver = graph.receivers[receiver_index]
        target = len(graph.sources) if receiver.output_source is None else receiver.output_source
        port = "output" if receiver.output_source is None else ("mlp" if receiver.head is None else receiver.name.rsplit(".", 1)[1])
        involved.update((source, target))
        edges.append((source, target, port))
    by_layer = {}
    for source in sorted(involved):
        if source == 0:
            layer = -1
        elif source == len(graph.sources):
            layer = graph.n_layers
        else:
            layer = (source - 1) // (graph.n_heads + 1)
        by_layer.setdefault(layer, []).append(source)
    positions = {}
    for layer, nodes in by_layer.items():
        for row, node in enumerate(nodes):
            positions[node] = (layer, (row + 1) / (len(nodes) + 1))
    figure, ax = plt.subplots(figsize=(19, 8), constrained_layout=True)
    for source, target, port in edges:
        curvature = {"q": -.07, "k": 0., "v": .07, "mlp": .02, "output": -.02}[port]
        arrow = FancyArrowPatch(positions[source], positions[target], connectionstyle=f"arc3,rad={curvature}",
                                arrowstyle="-|>", mutation_scale=6, linewidth=.8, alpha=.55,
                                color=colors[port], shrinkA=15, shrinkB=15)
        ax.add_patch(arrow)
    for node, (x, y) in positions.items():
        label = "logits" if node == len(graph.sources) else graph.sources[node]
        ax.text(x, y, label, ha="center", va="center", fontsize=8,
                bbox={"boxstyle": "round,pad=.25", "fc": "white", "ec": "#333333"}, zorder=5)
    ax.set(xlim=(-1.6, graph.n_layers + .6), ylim=(-.04, 1.04),
           title=f"{len(edges)} retained additive edges; colors identify the receiver port")
    ax.legend(handles=[Line2D([0], [0], color=color, label=port) for port, color in colors.items()],
              loc="lower center", ncol=5, frameon=False)
    ax.axis("off")
    return figure


def plot_method_curves(comparison: dict):
    """Provided drawing of actual development measurements and all random samples."""
    import matplotlib.pyplot as plt
    import numpy as np

    is_ioi = comparison["task"] == "ioi"
    raw_key = "mean_logit_difference" if is_ioi else "mean_probability_difference"
    panels = (("recovery", "Clean–corrupt gap recovered"),
              (raw_key, "IO − subject logit difference" if is_ioi else "Later − earlier/equal probability mass"),
              ("kl_clean_to_circuit", "KL(clean || circuit)"))
    colors = {"eap": "#d95f02", "eap-ig-inputs": "#1b9e77", "eap-ig-kl": "#7570b3"}
    labels = {"eap": "EAP, task loss", "eap-ig-inputs": "EAP-IG, task loss", "eap-ig-kl": "EAP-IG, KL loss"}
    sizes = [row["edge_count"] for row in comparison["rows"]]
    methods = list(comparison["rows"][0]["methods"])
    figure, axes = plt.subplots(1, 3, figsize=(16.5, 4.8), constrained_layout=True)
    for ax, (metric, ylabel) in zip(axes, panels, strict=True):
        for method in methods:
            values = [row["methods"][method][metric] for row in comparison["rows"]]
            ax.plot(sizes, values, "o-", color=colors[method], label=labels[method], linewidth=2)
        samples = [[sample[metric] for sample in row["uniform_same_size"]] for row in comparison["rows"]]
        means = [np.mean(values) for values in samples]
        ax.plot(sizes, means, "s--", color="#555555", label="Same-size random mean", markersize=4)
        ax.fill_between(sizes, [min(values) for values in samples], [max(values) for values in samples],
                        color="#999999", alpha=.25, label="Range of every random sample")
        ax.axhline(comparison["full"][metric], color="#333333", linestyle=":", linewidth=1.3, label="Full clean model")
        ax.axhline(comparison["empty"][metric], color="#aaaaaa", linestyle=":", linewidth=1.3, label="Fully corrupt model")
        ax.set(xscale="log", xlabel="Retained additive edges", ylabel=ylabel)
        ax.grid(alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=7, loc="best", frameon=True)
    task = "IOI" if is_ioi else "Greater-than"
    selection = comparison.get("score_selection", "topk")
    selection = "grow from output" if selection == "greedy" else "global top scores"
    corruption = comparison.get("corruption", "all-names" if is_ioi else "start-year suffix 01")
    corruption = {"all-names": "all names replaced", "second-subject": "repeated subject replaced"}.get(corruption, corruption)
    count = len(comparison["development"]["prompts"])
    figure.suptitle(f"{task} on {count} independent development prompts · {selection} · {corruption}", fontsize=12)
    return figure


def plot_frozen_transfer(rows: list[dict], *, task: str, split: str = "heldout"):
    """Draw transfer or explicitly labeled failed-selection development diagnostics."""
    import matplotlib.pyplot as plt
    import numpy as np

    if split not in ("development", "heldout"):
        raise ValueError("Label the measured split explicitly")
    is_ioi = task == "ioi"
    raw_key = "mean_logit_difference" if is_ioi else "mean_probability_difference"
    panels = (("recovery", "Clean–corrupt gap recovered"),
              (raw_key, "IO − subject logit difference" if is_ioi else "Later − earlier/equal probability mass"),
              ("kl_clean_to_circuit", "KL(clean || circuit)"))
    labels = {"acdc": "ACDC", "eap": "EAP", "eap-ig-inputs": "EAP-IG\nTask loss",
              "eap-ig-kl": "EAP-IG\nKL loss"}
    figure, axes = plt.subplots(1, 3, figsize=(16.5, 5), constrained_layout=True)
    for ax, (metric, ylabel) in zip(axes, panels, strict=True):
        for index, row in enumerate(rows):
            if split == "heldout":
                ax.plot([index - .12, index + .12],
                        [row["development"][metric], row["candidate"][metric]], color="#2166ac", alpha=.6)
                ax.scatter(index - .12, row["development"][metric], color="#2166ac", marker="s", s=55,
                           label="Development" if index == 0 else None, zorder=4)
            ax.scatter(index + .12, row["candidate"][metric], color="#1b9e77", s=55,
                       label=("Held-out wording" if split == "heldout" else "Development diagnostic") if index == 0 else None, zorder=4)
            for name, color, offset, label in (
                ("uniform_same_size", "#777777", -.26, "Uniform same-size null"),
                ("receiver_indegree_matched", "#d95f02", .26, "Receiver-matched null"),
            ):
                values = [sample[metric] for sample in row["controls"][name]["results"]]
                # Deterministic horizontal spreading shows every sample without jitter selection.
                x = index + offset + np.linspace(-.035, .035, len(values))
                ax.scatter(x, values, color=color, alpha=.4, s=13,
                           label=label if index == 0 else None)
        ax.axhline(rows[0]["full"][metric], color="#333333", linestyle=":", label=f"{split.capitalize()} clean")
        ax.axhline(rows[0]["empty"][metric], color="#aaaaaa", linestyle=":", label=f"{split.capitalize()} corrupt")
        ax.set(xticks=range(len(rows)), xticklabels=[f"{labels.get(row['method'], row['method'])}\n{row['retained_edges']} edges" for row in rows],
               ylabel=ylabel)
        ax.grid(axis="y", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=7, loc="best")
    context = (f"choices frozen on development before {rows[0]['count']} held-out prompts" if split == "heldout" else
               f"failed selection; diagnostic candidate on {rows[0]['count']} development prompts; holdout unopened")
    figure.suptitle(f"{task.upper() if is_ioi else 'Greater-than'} · {context}")
    return figure


def plot_edge_diagnostics(minimality: list[dict], completeness: list[dict], *, task_label: str):
    """Draw all retained-edge deletions and sampled full-versus-circuit removals."""
    import matplotlib.pyplot as plt
    import numpy as np

    ordered = sorted(minimality, key=lambda row: row["task_damage"])
    figure, axes = plt.subplots(1, 3, figsize=(16.5, 4.8), constrained_layout=True)
    for ax, key, ylabel in ((axes[0], "task_damage", f"Drop in mean {task_label}"),
                            (axes[1], "kl_damage", "Increase in KL(clean || circuit)")):
        values = [row[key] for row in ordered]
        ax.bar(range(len(values)), values, width=1, color=["#2166ac" if v > 0 else "#d95f02" for v in values])
        ax.axhline(0, color="#333333", linestyle=":", label="No change from intact circuit")
        ax.set(xlabel="Retained edge, ordered by task damage", ylabel=ylabel)
        ax.legend(fontsize=7)
    all_values = [row[key] for row in completeness for key in ("circuit_minus_subset", "full_minus_subset")]
    low, high = min(all_values), max(all_values)
    margin = max((high - low) * .05, .01)
    axes[2].plot([low - margin, high + margin], [low - margin, high + margin], ":", color="#333333",
                 label="Full and circuit agree after removal")
    for size in sorted({row["subset_size"] for row in completeness}):
        subset = [row for row in completeness if row["subset_size"] == size]
        axes[2].scatter([row["circuit_minus_subset"] for row in subset],
                        [row["full_minus_subset"] for row in subset], alpha=.65, s=25,
                        label=f"Remove {size} retained edge{'s' if size != 1 else ''}")
    axes[2].set(xlabel=f"Circuit minus subset: {task_label}", ylabel=f"Full graph minus same subset: {task_label}",
                xlim=(low - margin, high + margin), ylim=(low - margin, high + margin))
    axes[2].legend(fontsize=7)
    for ax in axes:
        ax.grid(alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    figure.suptitle(f"{len(ordered)} individual edge deletions; {len(completeness)} sampled backup-mechanism probes")
    return figure
