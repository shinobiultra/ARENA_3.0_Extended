"""Plot layout for actual 9.1 measurements; no model or intervention code."""
from __future__ import annotations

import textwrap

import matplotlib.pyplot as plt
import numpy as np

COLORS = {"target": "#2166ac", "random": "#7570b3", "permuted": "#d95f02"}


def plot_development_separation(separation):
    rows = separation["rows"]
    layers = sorted({row["layer"] for row in rows})
    heat = np.full((len(layers), 3), np.nan)
    for row in rows:
        if row["defined"] and row["offset"] in (-3, -2, -1):
            heat[row["layer"], (-3, -2, -1).index(row["offset"])] = row["development"]["auroc"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    cmap = plt.get_cmap("RdBu").copy();cmap.set_bad("#eeeeee")
    image = axes[0].imshow(heat, vmin=0, vmax=1, aspect="auto", cmap=cmap)
    axes[0].set(xticks=range(3), xticklabels=["suffix −3", "suffix −2", "suffix −1"],
                ylabel="Decoder block input (zero-based)", title="Development ranking; undefined sites are gray")
    fig.colorbar(image, ax=axes[0], label="AUROC")
    valid = [row for row in rows if row["defined"] and row["offset"] == -1]
    x = [row["layer"] for row in valid]
    for kind in ("random", "permuted"):
        values = [np.array([draw["auroc"] for draw in row["nulls"][kind] if draw is not None]) for row in valid]
        axes[1].fill_between(x, [v.min() for v in values], [v.max() for v in values], color=COLORS[kind], alpha=.12)
        axes[1].plot(x, [v.mean() for v in values], color=COLORS[kind], label=f"{kind}: 32-draw mean and range")
    axes[1].plot(x, [row["development"]["auroc"] for row in valid], color=COLORS["target"], marker="o", label="true training labels")
    axes[1].axhline(.5, color="gray", linestyle="--", linewidth=1)
    axes[1].set(xlabel="Decoder block input at final suffix token", ylabel="Development AUROC", ylim=(-.03, 1.03),
                title="Do not flip or discard strong null directions")
    axes[1].legend(fontsize=8, loc="lower left")
    return fig


def plot_development_selection(candidates, selected):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True, sharey=True)
    y = np.arange(len(candidates))
    labels = [f"{row['layer']}/{row['offset']}" + (" ★" if (row['layer'], row['offset']) == (selected['layer'], selected['offset']) else "") for row in candidates]
    axes[0].barh(y-.17, [row["addition_increase"] for row in candidates], height=.32, color="#2166ac", label="Addition: benign marker increase")
    axes[0].barh(y+.17, [row["ablation_decrease"] for row in candidates], height=.32, color="#d95f02", label="Ablation: refusal marker decrease")
    axes[0].axvline(.125, color="#2166ac", ls="--", lw=1)
    axes[0].axvline(.25, color="#d95f02", ls="--", lw=1)
    axes[0].axvline(0, color="black", lw=.6)
    axes[0].set(yticks=y, yticklabels=labels, ylabel="Decoder input / suffix position", xlabel="Development marker-rate change",
                title="All fifteen candidates; dashed lines are fixed floors")
    axes[0].invert_yaxis()
    axes[0].legend(fontsize=8, loc="upper center", bbox_to_anchor=(.5, -.11))
    axes[1].barh(y, [row["capability_drop"] for row in candidates], height=.65,
                 color=["#b2182b" if row["capability_drop"]>.125 else "#777777" for row in candidates])
    axes[1].axvline(.125, color="gray", ls="--", lw=1);axes[1].axvline(0, color="black", lw=.6)
    axes[1].set(xlabel="Exact benign-task accuracy drop under ablation", title="Capability cost is a separate selection condition")
    return fig


def plot_final_curves(summary):
    rows = summary["conditions"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for column, mode in enumerate(("add", "ablate")):
        rate = "benign_category_marker_rate" if mode == "add" else "refusal_category_marker_rate"
        for row_index, (field, baseline) in enumerate(((rate, summary["baseline"][rate]),
                                                      ("capability_accuracy", summary["baseline_capability"]))):
            ax = axes[row_index,column]
            x = [0.] + sorted({row["strength"] for row in rows if row["mode"]==mode})
            for kind in ("random", "permuted", "target"):
                selected = [row for row in rows if row["mode"]==mode and row["kind"]==kind]
                samples = sorted({row["sample"] for row in selected})
                values = np.array([[baseline] + [next(row[field] for row in selected if row["sample"]==sample and row["strength"]==strength)
                                                 for strength in x[1:]] for sample in samples])
                if kind != "target":
                    ax.fill_between(x, values.min(0), values.max(0), color=COLORS[kind], alpha=.10)
                    for y in values:
                        ax.plot(x, y, color=COLORS[kind], alpha=.28, lw=.7, marker=".", ms=3)
                ax.plot(x, values.mean(0), color=COLORS[kind], marker="o" if kind=="target" else None,
                        lw=2, label=kind if kind=="target" else f"{kind}: all 8 draws")
            ax.set(ylim=(-.03,1.03), xticks=x,
                   xlabel="Raw mean-difference multiplier α" if mode=="add" else "Fraction removed from every residual write β",
                   ylabel="Refusal-marker rate" if row_index==0 else "Exact answer accuracy (32 new benign questions)")
            ax.grid(alpha=.15)
    axes[0,0].set_title("Addition on 36 reserved benign requests")
    axes[0,1].set_title("Ablation on 36 reserved refusal-category requests")
    axes[0,0].legend(fontsize=9,loc="lower right")
    fig.suptitle("A frozen direction changes behavior; its amplitude has a measurable capability cost\nShading is the observed null range, not a confidence interval",fontsize=13)
    return fig


def plot_final_separation(separation, threshold):
    labels = np.array(separation["labels"],dtype=bool)
    scores = np.array(separation["target"]["scores"])
    fig, axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    bins=np.linspace(scores.min()-.2,scores.max()+.2,17)
    for mask,color,label in ((~labels,"#2166ac","benign category"),(labels,"#b2182b","refusal category")):
        axes[0].hist(scores[mask],bins=bins,color=color,alpha=.5,label=label)
    axes[0].axvline(threshold,color="black",ls="--",label="frozen training threshold")
    axes[0].set(xlabel="Projection on frozen unit direction",ylabel="Reserved prompts",title="Final ranking and threshold performance")
    axes[0].legend(fontsize=8)
    for x,kind in enumerate(("random","permuted")):
        values=[row["auroc"] for row in separation[kind] if row is not None]
        axes[1].scatter(np.linspace(x-.15,x+.15,len(values)),values,color=COLORS[kind],alpha=.65,s=23)
    axes[1].axhline(separation["target"]["auroc"],color=COLORS["target"],label="true-label direction")
    axes[1].axhline(.5,color="gray",ls="--",lw=1)
    axes[1].set(xticks=[0,1],xticklabels=["32 random directions","32 training permutations"],ylabel="Final AUROC",ylim=(-.03,1.03),
                title="Near-perfect null rankings remain visible")
    axes[1].legend(fontsize=8)
    return fig


def plot_benign_generations(baseline, added, ablated):
    clean = [row for row in baseline if row["refusal_label"] is False][:4]
    added_map={row["id"]:row for row in added};removed_map={row["id"]:row for row in ablated}
    fig, axes=plt.subplots(len(clean),3,figsize=(14,11),squeeze=False,constrained_layout=True)
    for index,row in enumerate(clean):
        for column,(name,record) in enumerate((("Unedited",row),("Addition α=1",added_map[row["id"]]),("Ablation β=1",removed_map[row["id"]]))):
            ax=axes[index,column];ax.axis("off")
            title=f"{name} · marker={record['refusal_marker']}"
            body=textwrap.fill(row["prompt"],width=47)+"\n\n"+textwrap.fill(record["completion"],width=47)
            ax.text(0,1,title,ha="left",va="top",fontsize=10,weight="bold",color=COLORS['target'] if column else '#333333')
            ax.text(0,.9,body,ha="left",va="top",fontsize=9,linespacing=1.25,wrap=False)
    fig.suptitle("Actual generated words for the fixed first four reserved benign requests\nNo examples selected by intervention success; generation budget 48 tokens",fontsize=13)
    return fig
