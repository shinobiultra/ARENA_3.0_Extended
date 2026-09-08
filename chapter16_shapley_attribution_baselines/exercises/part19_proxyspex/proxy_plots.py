"""Scientific figures from caller-computed predictions and spectra; no method imports."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree


COLORS = {"proxy": "#2563a6", "truncated": "#da8b19", "refined": "#16846d",
          "shuffled": "#bb4653", "constant": "#555b66", "additive": "#8b61a8"}


def support_label(support, feature_names=None):
    if not support:
        return "constant"
    if feature_names is None:
        return "{" + ",".join(map(str, support)) + "}"
    return " × ".join(f"{feature_names[i]} [{i}]" for i in support)


def _clean(axis):
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(alpha=0.16, axis="y")


def plot_cold_open(masks, values):
    figure, axis = plt.subplots(figsize=(11, 3.7), layout="constrained")
    colors = np.where(values > 2, COLORS["truncated"], COLORS["proxy"])
    axis.bar(np.arange(len(masks)), values, color=colors, width=0.75)
    axis.axhline(values.mean(), color=COLORS["constant"], linestyle="--", label="constant prediction (cube mean)")
    axis.set(xticks=np.arange(len(masks)), xticklabels=["".join(map(str, row)) for row in masks],
             xlabel="retention mask x₀x₁x₂x₃  (1 = retained)", ylabel="exact analytic score",
             title="Three features together unlock the extra two points")
    axis.legend(frameon=False)
    _clean(axis)
    return figure


def plot_boosting(models, audit_mse, constant_mse):
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    colors = {"depth 3": COLORS["proxy"], "depth 1": COLORS["truncated"], "shuffled responses": COLORS["shuffled"]}
    for name, model in models.items():
        axes[0].plot(model["train_mse"], color=colors[name], linewidth=2, label=name)
        axes[1].plot(audit_mse[name], color=colors[name], linewidth=2, label=name)
    axes[1].axhline(constant_mse, color=COLORS["constant"], linestyle="--", label="training-mean baseline")
    for axis, title in zip(axes, ("Loss on each model's own training labels", "Error against the real audit responses"), strict=True):
        axis.set(xlabel="residual trees fitted", ylabel="mean squared error", title=title)
        axis.legend(frameon=False, fontsize=8)
        _clean(axis)
    return figure


def plot_path(masks, spectrum, contributions, expected, reconstructed, *, selected_values=None, title=None):
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.2), layout="constrained")
    input_supports = list(spectrum)
    order = sorted(range(len(input_supports)), key=lambda i: (len(input_supports[i]), input_supports[i]))
    supports = [input_supports[i] for i in order]
    values = np.array([spectrum[support] for support in supports])
    contributions = np.asarray(contributions)[:, order]
    if supports:
        axes[0].bar(np.arange(len(supports)), values, color=np.where(values >= 0, COLORS["proxy"], COLORS["shuffled"]))
        axes[0].set(xticks=np.arange(len(supports)), xticklabels=[support_label(s) for s in supports])
        bound = max(float(np.abs(contributions).max()), 1e-12)
        artist = axes[1].imshow(contributions, aspect="auto", cmap="RdBu", vmin=-bound, vmax=bound)
        axes[1].set(xticks=np.arange(len(supports)), xticklabels=[support_label(s) for s in supports],
                    yticks=np.arange(len(masks)), yticklabels=["".join(map(str, row)) for row in masks])
        figure.colorbar(artist, ax=axes[1], label="coefficient × character", shrink=0.8)
    else:
        for axis in axes[:2]:
            axis.text(0.5, 0.5, "Impossible path\nall coefficients cancel", ha="center", va="center", transform=axis.transAxes)
            axis.set_xticks([]); axis.set_yticks([])
    axes[0].axhline(0, color="black", linewidth=0.7)
    axes[0].set(title="Signed path coefficients", ylabel="Fourier coefficient")
    axes[1].set(title="Terms cancel outside the path", xlabel="support", ylabel="retention mask")
    axes[2].plot(expected, "o", color="black", label="direct branch checks", markersize=6)
    axes[2].plot(reconstructed, "x", color=COLORS["proxy"], label="sum of Fourier terms", markersize=7)
    if selected_values is not None:
        axes[2].plot(selected_values, "s--", color=COLORS["truncated"], label="energy-selected terms", markersize=4)
    axes[2].set(xticks=np.arange(len(masks)), xticklabels=["".join(map(str, row)) for row in masks],
                title="Reconstructed path indicator", xlabel="retention mask", ylabel="prediction")
    axes[2].tick_params(axis="x", labelrotation=45)
    axes[2].legend(frameon=False, fontsize=8)
    _clean(axes[0]); _clean(axes[2])
    if title:
        figure.suptitle(title, fontsize=13)
    return figure


def plot_tree_spectrum(tree, spectrum, cube_error):
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw={"width_ratios": [1.5, 1]}, layout="constrained")
    plot_tree(tree, feature_names=[f"x{i}" for i in range(tree.n_features_in_)],
              filled=True, impurity=False, rounded=True, precision=2, ax=axes[0], fontsize=8)
    axes[0].set_title("One fitted tree: each value belongs to a leaf region")
    supports = sorted(spectrum, key=lambda s: (len(s), s))
    values = np.array([spectrum[s] for s in supports])
    axes[1].barh(np.arange(len(supports)), values, color=np.where(values >= 0, COLORS["proxy"], COLORS["shuffled"]))
    axes[1].set(yticks=np.arange(len(supports)), yticklabels=[support_label(s) for s in supports],
                xlabel="signed Fourier coefficient", title=f"Exact extraction\nfull-cube max error {cube_error:.2e}")
    axes[1].axvline(0, color="black", linewidth=0.7)
    axes[1].invert_yaxis(); _clean(axes[1])
    return figure


def plot_parity(truth, proxy, extracted, known_truth, known_reconstructed, known_spectrum):
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
    axes[0].scatter(truth, proxy, s=35, alpha=0.45, color=COLORS["shuffled"], label="fitted proxy")
    axes[0].plot([-1.4, 1.4], [-1.4, 1.4], "k--", label="exact target")
    axes[0].set(xlabel="true signed parity", ylabel="proxy prediction", title="Greedy fitting can miss parity")
    axes[0].legend(frameon=False, fontsize=8)
    indices = np.arange(min(48, len(truth)))
    axes[1].plot(indices, proxy[indices], "o", color=COLORS["shuffled"], label="fitted proxy", markersize=4)
    axes[1].plot(indices, extracted[indices], "x", color=COLORS["proxy"], label="extracted spectrum", markersize=5)
    axes[1].set(xlabel="audit mask row (first 48)", ylabel="prediction",
                title=f"Yet extraction agrees\nRMSE {np.sqrt(np.mean((proxy-extracted)**2)):.2e}")
    axes[1].legend(frameon=False, fontsize=8)
    supports = list(known_spectrum)
    axes[2].bar([support_label(s) for s in supports], list(known_spectrum.values()), color=COLORS["proxy"], width=.5)
    axes[2].axhline(-1.25, color="black", linestyle="--", label="known coefficient −1.25")
    axes[2].set(ylabel="Fourier coefficient", title="A known parity tree extracts correctly", ylim=(-1.55, .2), xlim=(-.8,.8))
    axes[2].text(0.5, 0.97, f"prediction max error {np.max(np.abs(known_truth-known_reconstructed)):.1e}",
                 ha="center", va="top", transform=axes[2].transAxes, fontsize=9)
    axes[2].legend(frameon=False, fontsize=8, loc="lower center")
    for axis in axes:
        _clean(axis)
    return figure


def plot_energy(runs):
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), layout="constrained")
    palette = [COLORS["proxy"], COLORS["refined"]]
    for (name, data), color in zip(runs.items(), palette, strict=True):
        axes[0].step(data["fractions"], data["retained"], where="post", color=color, linewidth=2, label=name)
        axes[1].step(data["fractions"], data["rmse"], where="post", color=color, linewidth=2, label=name)
        axes[0].scatter([.95], [data["author_retained"]], marker="X", s=90, color=color, edgecolors="black", zorder=5)
        axes[1].scatter([.95], [data["author_rmse"]], marker="X", s=90, color=color, edgecolors="black", zorder=5)
        axes[1].axhline(data["constant_rmse"], color=color, linestyle=":", alpha=.55)
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1, label="requested fraction")
    axes[0].scatter([], [], marker="X", s=70, color="white", edgecolors="black", label="author strict rule at .95")
    axes[1].plot([], [], ":", color="black", label="intercept-only controls")
    axes[0].set(xlabel="requested nonconstant energy fraction", ylabel="actual retained fraction", ylim=(-.03, 1.05),
                title="The corrected prefix reaches its stated fraction")
    axes[1].set(xlabel="requested nonconstant energy fraction", ylabel="full-cube reconstruction RMSE",
                title="Discarded true coefficients leave prediction error")
    for axis in axes:
        axis.legend(frameon=False, fontsize=8)
        _clean(axis)
    return figure


def plot_stage_comparison(runs):
    figure, axes = plt.subplots(2, len(runs), figsize=(14, 8), layout="constrained")
    names = ["proxy", "truncated", "refined", "shuffled", "constant", "additive"]
    labels = ["tree proxy", "truncated", "refined", "shuffled → refined", "constant", "additive fit"]
    for column, (game, run) in enumerate(runs.items()):
        truth = run["audit_values"]
        predictions = {**run["predictions"], "shuffled": run["shuffled_predictions"], **run["baseline_predictions"]}
        errors = [np.sqrt(np.mean((predictions[name] - truth) ** 2)) for name in names]
        axis = axes[0, column]
        axis.barh(np.arange(len(names)), errors, color=[COLORS[name] for name in names])
        axis.set(yticks=np.arange(len(names)), yticklabels=labels, xlabel="RMSE against the analytic target", title=game.replace("_", " "))
        axis.invert_yaxis()
        for row, error in enumerate(errors):
            axis.annotate(f"{error:.3g}", (error, row), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8)
        axis.set_xlim(0, max(errors) * 1.24 + 1e-9)
        axis = axes[1, column]
        proxy, extracted = predictions["proxy"], predictions["extracted"]
        axis.scatter(proxy, extracted, s=7, alpha=.25, color=COLORS["proxy"])
        limits = (min(proxy.min(), extracted.min()), max(proxy.max(), extracted.max()))
        axis.plot(limits, limits, "k--", label="exact identity")
        error = np.sqrt(np.mean((proxy-extracted)**2))
        axis.set(xlabel="tree proxy prediction", ylabel="extracted-spectrum prediction", title=f"Extraction RMSE {error:.2e}")
        axis.legend(frameon=False, fontsize=8)
    for axis in axes.flat:
        _clean(axis)
    figure.suptitle("The spectrum can explain its proxy exactly while predicting the target poorly", fontsize=14)
    return figure


def plot_cv(fit):
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout="constrained")
    proxy, ridge = fit["proxy_cv"], fit["ridge_cv"]
    losses = np.asarray(proxy["fold_mse"])
    artist = axes[0].imshow(losses, aspect="auto", cmap="Blues")
    labels = [f"depth {p['max_depth']}, {p['n_estimators']} trees" for p in proxy["candidates"]]
    axes[0].set(xticks=np.arange(losses.shape[1]), xticklabels=[f"fold {i+1}" for i in range(losses.shape[1])],
                yticks=np.arange(len(labels)), yticklabels=labels, title="Proxy development MSE (lower is better)")
    for i, j in np.ndindex(losses.shape):
        axes[0].text(j, i, f"{losses[i,j]:.3f}", ha="center", va="center",
                     color="white" if losses[i,j] > (losses.min()+losses.max())/2 else "black", fontsize=9)
    figure.colorbar(artist, ax=axes[0], shrink=.75, label="validation MSE")
    alphas = np.asarray(ridge["alphas"])
    for i, values in enumerate(ridge["fold_mse"]):
        axes[1].semilogx(alphas, values, "o-", alpha=.55, label=f"fold {i+1}: {len(ridge['fold_supports'][i])} supports")
    axes[1].semilogx(alphas, ridge["validation_mse"], "k.-", linewidth=2, label="pooled validation MSE")
    selected = ridge["selected_index"]
    axes[1].scatter([alphas[selected]], [ridge["validation_mse"][selected]], marker="*", s=150, color=COLORS["truncated"],
                    edgecolors="black", zorder=5, label="selected alpha")
    axes[1].set(xlabel="ridge alpha (sum-of-squares objective)", ylabel="validation MSE",
                title="Ridge uses each fold's own learned support")
    axes[1].set_yscale("log")
    axes[1].legend(frameon=False, fontsize=8); _clean(axes[1])
    return figure


def _native_records(payload, prompt_index, seed):
    if not payload or not payload.get("results"):
        raise ValueError("Native results are unexecuted; run the common study before plotting.")
    records = [r for r in payload["results"] if r["prompt_index"] == prompt_index and r["seed"] == seed]
    if not records:
        raise ValueError("The selected prompt/seed has no completed native results.")
    return records


def plot_native_stages(payload, *, prompt_index, seed, budget=None):
    records = _native_records(payload, prompt_index, seed)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout="constrained")
    for method, color in (("spex", COLORS["proxy"]), ("proxyspex", COLORS["refined"])):
        for condition, linestyle in (("real", "-"), ("shuffled", "--")):
            selected = sorted((r for r in records if r["method"] == method and r["condition"] == condition), key=lambda r:r["budget"])
            if not selected:
                continue
            key = "final" if method == "spex" else "refined"
            errors = [np.sqrt(np.mean((np.asarray(r["predictions"][key])-r["audit_values"])**2)) for r in selected]
            axes[0].plot([r["query_receipt"]["raw_requested"] for r in selected], errors, "o"+linestyle,
                         color=color, label=f"{method}: {condition}")
    real = sorted((r for r in records if r["method"] == "proxyspex" and r["condition"] == "real"), key=lambda r:r["budget"])
    for name in ("constant", "additive"):
        errors = [np.sqrt(np.mean((np.asarray(r["baseline_predictions"][name])-r["audit_values"])**2)) for r in real]
        axes[0].plot([r["query_receipt"]["raw_requested"] for r in real], errors, ":", color=COLORS[name],
                     label=f"{name} (proxy training rows)")
    for name, left, right, color in (("model → proxy", "target", "proxy", COLORS["proxy"]),
                                   ("proxy → extraction", "proxy", "extracted", COLORS["additive"]),
                                   ("extraction → truncation", "extracted", "truncated", COLORS["truncated"]),
                                   ("model → refined", "target", "refined", COLORS["refined"])):
        errors=[]
        for record in real:
            first = record["audit_values"] if left == "target" else record["predictions"][left]
            errors.append(np.sqrt(np.mean((np.asarray(first)-record["predictions"][right])**2)))
        axes[1].plot([r["query_receipt"]["raw_requested"] for r in real], errors, "o-", color=color, label=name)
    for axis in axes:
        axis.set(xlabel="raw requested query rows", ylabel="audit RMSE")
        if budget is not None:
            axis.axvline(budget, color="gray", alpha=.35, linewidth=1)
        axis.legend(frameon=False, fontsize=8); _clean(axis)
    axes[0].set_title("Common model audit: methods and negative controls")
    axes[1].set_title("Separate the ProxySPEX error stages")
    return figure


def plot_native_spectrum(payload, *, prompt_index, seed, budget):
    records = _native_records(payload, prompt_index, seed)
    prompts = payload["prompts"]
    prompt = next((p for i,p in enumerate(prompts) if p.get("prompt_index", i) == prompt_index), None)
    if prompt is None:
        raise ValueError("Prompt metadata is absent.")
    figure, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
    for axis, method in zip(axes, ("spex", "proxyspex"), strict=True):
        found = [r for r in records if r["method"] == method and r["condition"] == "real" and r["budget"] == budget]
        if len(found) != 1:
            raise ValueError("Require one completed real run per method at the selected budget.")
        record = found[0]
        spectrum = record["spectrum"]
        supports = sorted((s for s in spectrum if s), key=lambda s:(-abs(spectrum[s]), s))[:10]
        values = np.asarray([spectrum[s] for s in supports])
        if supports:
            axis.barh(np.arange(len(supports)), values, color=np.where(values>=0, COLORS["proxy"], COLORS["shuffled"]))
            axis.set(yticks=np.arange(len(supports)), yticklabels=[support_label(s,prompt["feature_names"]) for s in supports])
            axis.invert_yaxis()
        else:
            axis.text(.5,.5,"No nonconstant support recovered",ha="center",va="center",transform=axis.transAxes)
        key="final" if method=="spex" else "refined"
        error=np.sqrt(np.mean((np.asarray(record["predictions"][key])-record["audit_values"])**2))
        axis.axvline(0,color="black",linewidth=.7)
        axis.set(xlabel="signed Fourier coefficient (not SII)",
                 title=f"{method} · audit RMSE {error:.3g}\n{record['query_receipt']['unique_model_masks']} unique queried masks")
        _clean(axis)
    return figure
