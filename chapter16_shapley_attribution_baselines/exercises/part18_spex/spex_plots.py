"""Analytic experiment/figure plumbing; all taught operations are passed in.

This module never imports the SPEX implementation or queries a native model.
The notebook's completed functions generate every fitted spectrum and trace.
"""
from itertools import combinations
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {"sparse": "#0072B2", "additive": "#009E73", "dense": "#D55E00",
          "wrong degree": "#CC79A7", "shuffled": "#6F4E7C"}


def _finish(figure, title):
    figure.suptitle(title, fontsize=16, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, .95))
    return figure


def _label(support, feature_names=None):
    if not support:
        return "constant"
    return " × ".join(str(feature_names[i]) if feature_names is not None else f"x{i}" for i in support)


def analytic_games(n=31):
    """Known signed truths, not oracle-fitted results or cached spectra."""
    if n < 10:
        raise ValueError("The fixed analytic comparison uses at least ten features")
    polynomial = {(): .3}
    for monomial, coefficient in [((0,), 1), ((3,), -.7), ((1, 7), 1.2), ((2, 4, 9), -2.4)]:
        for size in range(len(monomial) + 1):
            for support in combinations(monomial, size):
                polynomial[support] = polynomial.get(support, 0) + coefficient * (-1) ** size / 2 ** len(monomial)
    rng = np.random.default_rng(884)
    supports = list(combinations(range(n), 3))
    selected = rng.choice(len(supports), 256, replace=False)
    coefficients = rng.normal(size=256)
    coefficients /= np.linalg.norm(coefficients)
    dense = {supports[int(i)]: float(v) for i, v in zip(selected, coefficients)}
    return {"sparse": polynomial, "additive": {(): .3, (0,): 1.1, (1,): -.8, (2,): .5, (3,): -.35, (4,): .2, (5,): -.1},
            "dense": dense, "wrong degree": {(1, 3, 7): -1.25}, "shuffled": polynomial.copy()}


def fit_simple_baselines(train_masks, train_values, audit_masks):
    """Query-matched constant and additive least squares on distinct masks."""
    _, first = np.unique(train_masks, axis=0, return_index=True)
    x = np.column_stack((np.ones(len(first)), train_masks[first].astype(float)))
    y = train_values[first]
    coefficient = np.linalg.lstsq(x, y, rcond=None)[0]
    return {"constant": np.full(len(audit_masks), np.mean(y)),
            "additive": np.column_stack((np.ones(len(audit_masks)), audit_masks)) @ coefficient}


def run_analytic_suite(methods, *, n=31, seeds=(0, 1, 2), budgets=(3, 4, 5), games=None):
    """Run a bounded analytic comparison using the supplied student namespace.

    Audit masks are independent of query design and never used for selection.
    A shuffled control permutes values across unique training masks, then puts
    duplicates back consistently. It predicts the original game on the audit.
    """
    truths = analytic_games(n) if games is None else games
    audit_masks = np.random.default_rng(7701).integers(0, 2, size=(4096, n), dtype=np.uint8)
    records = []
    for game, truth in truths.items():
        audit_values = methods.predict_spectrum(audit_masks, truth)
        scale = max(float(np.std(audit_values)), 1e-12)
        for seed in seeds:
            for b in budgets:
                t = 1 if game == "wrong degree" else 3
                design = methods.make_query_design(n, t, b, C=3, seed=int(seed))
                values = methods.predict_spectrum(design.masks, truth)
                if game == "shuffled":
                    _, first, inverse = np.unique(design.masks, axis=0, return_index=True, return_inverse=True)
                    permutation = np.random.default_rng(9000 + 100 * seed + b).permutation(len(first))
                    values = values[first][permutation][inverse]
                recovery = methods.recover_spectrum(design, values)
                prediction = methods.predict_spectrum(audit_masks, recovery.spectrum)
                baselines = fit_simple_baselines(design.masks, values, audit_masks)
                recovered = set(recovery.spectrum)
                expected = {support for support, coefficient in truth.items() if abs(coefficient) > 1e-8}
                records.append({"game": game, "seed": int(seed), "b": int(b), "t": t,
                                "requested": design.requested_count, "unique": design.unique_count,
                                "spectrum": recovery.spectrum, "truth": truth, "recovery": recovery,
                                "design": design, "values": values, "predictions": prediction,
                                "audit_values": audit_values, "baseline_predictions": baselines,
                                "rmse": float(np.sqrt(np.mean((prediction - audit_values) ** 2))),
                                "nrmse": float(np.sqrt(np.mean((prediction - audit_values) ** 2))) / scale,
                                "baseline_nrmse": {key: float(np.sqrt(np.mean((v - audit_values) ** 2))) / scale
                                                   for key, v in baselines.items()},
                                "recall": len(recovered & expected) / len(expected),
                                "false_positives": len(recovered - expected),
                                "missed": len(expected - recovered),
                                "sign_errors": sum(np.sign(recovery.spectrum[s]) != np.sign(truth[s]) for s in recovered & expected)})
    return {"scope": "analytic only", "n": n, "seeds": tuple(seeds), "budgets": tuple(budgets),
            "audit_masks": audit_masks, "truths": truths, "records": records}


def save_analytic_payload(payload, directory):
    """Retain every analytic result, including failures, without fitted objects."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory / "audit_masks.npz", masks=payload["audit_masks"])
    def spectrum_records(spectrum):
        return [{"support": list(s), "coefficient": float(v)} for s, v in sorted(spectrum.items())]
    def scalar(value):
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"Unexpected analytic metadata type: {type(value).__name__}")
    rows = []
    for record in payload["records"]:
        stem = f"{record['game'].replace(' ', '_')}_seed{record['seed']}_b{record['b']}"
        design, recovery = record["design"], record["recovery"]
        np.savez_compressed(directory / f"{stem}.npz", masks=design.masks, values=record["values"],
                            matrices=design.matrices, delays=design.delays, initial_bins=recovery.initial_bins,
                            residual_bins=recovery.residual_bins, audit_values=record["audit_values"],
                            predictions=record["predictions"], constant=record["baseline_predictions"]["constant"],
                            additive=record["baseline_predictions"]["additive"])
        row = {key: record[key] for key in ("game", "seed", "b", "t", "requested", "unique", "rmse", "nrmse", "baseline_nrmse", "recall", "false_positives", "missed", "sign_errors")}
        row.update(truth=spectrum_records(record["truth"]), spectrum=spectrum_records(record["spectrum"]),
                   trace=[dict(sweep, spectrum=spectrum_records(sweep["spectrum"])) for sweep in recovery.trace],
                   stop_reason=recovery.stop_reason, arrays=f"{stem}.npz")
        rows.append(row)
    (directory / "results.json").write_text(json.dumps({"scope": payload["scope"], "n": payload["n"], "records": rows}, indent=2, default=scalar, allow_nan=False) + "\n")
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(directory.iterdir()) if p.is_file() and p.name != "manifest.json"}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return directory


def plot_cold_open(masks, parity, additive):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].imshow(masks, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    axes[0].set(xticks=range(3), xticklabels=["x1", "x3", "x7"], yticks=range(8),
                yticklabels=["".join(map(str, row)) for row in masks], title="Eight settings of three active switches")
    for row in range(8):
        for col in range(3):
            axes[0].text(col, row, str(masks[row, col]), ha="center", va="center", color="white" if masks[row, col] else "black")
    axes[1].plot(range(8), parity, "o-", color=COLORS["sparse"], label="Signed three-way parity")
    axes[1].plot(range(8), additive, "s--", color=COLORS["additive"], label="Additive control")
    axes[1].axhline(0, color="black", linewidth=.6)
    axes[1].set(xticks=range(8), xticklabels=["".join(map(str, row)) for row in masks], xlabel="Retain bits (1 = on)", ylabel="Score", title="One switch can reverse another switch's effect")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].legend()
    return _finish(fig, "Can we find this interaction among 31 switches?")


def plot_transform(masks, values, coefficients, reconstructed):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    labels = ["".join(map(str, row)) for row in masks]
    axes[0].plot(values, "o-", label="Observed score", color=COLORS["sparse"])
    axes[0].plot(reconstructed, "x", markersize=9, color="black", label="Inverse transform")
    axes[0].set(title="The scores reconstruct exactly", xlabel="Mask ID", ylabel="Score")
    axes[0].legend(fontsize=9)
    axes[1].bar(range(len(coefficients)), coefficients, color=[COLORS["sparse"] if v >= 0 else COLORS["dense"] for v in coefficients])
    axes[1].set(title="A few signed Fourier coefficients", xlabel="Support bit string", ylabel="Fourier coefficient")
    axes[1].set_xticks(range(len(labels)), labels, rotation=65)
    axes[1].axhline(0, color="black", linewidth=.6)
    axes[2].plot(np.abs(values - reconstructed), "o-", label="Reconstruction error")
    axes[2].axhline(0, color="black", linestyle="--", label="Exact result")
    axes[2].set(title="The basis identity is a control", xlabel="Mask ID", ylabel="Absolute error", ylim=(-1e-16, 1e-14))
    axes[2].legend(fontsize=9)
    return _finish(fig, "The Walsh transform changes the coordinates, not the game")


def plot_code_rescue(methods, *, support=(1, 3, 7), bad_bit=0, confidence=.01, t=3):
    code = methods.make_bch_code(31, t)
    signature = methods.boolean_character(code.delays, support)
    reliability = signature[1:].copy()
    reliability[bad_bit] *= -confidence
    hard = methods.decode_bch_parity((reliability < 0).astype(np.uint8), code)
    soft = methods.soft_decode_parity(reliability, code)
    wrong_confidence = reliability.copy()
    wrong_confidence[bad_bit] = np.sign(wrong_confidence[bad_bit]) * 10
    misranked = methods.soft_decode_parity(wrong_confidence, code)
    rows = [support, hard.support, soft["decoded"].support, misranked["decoded"].support]
    names = ["Known support", "Hard decode", "Chase: low confidence", "Control: wrong confidence"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.6))
    axes[0].bar(np.arange(len(reliability)), reliability, color=[COLORS["dense"] if i == bad_bit else COLORS["sparse"] for i in range(len(reliability))])
    axes[0].plot(signature[1:], "_", markersize=9, color="black", label="Correct parity signs")
    axes[0].axhline(0, color="black", linewidth=.6)
    axes[0].set(title=f"One uncertain parity sign: |support| + 1 = {len(support)+1}", xlabel="Parity observation", ylabel="Observed shift ratio")
    axes[0].legend(fontsize=9)
    image = np.full((4, 31), np.nan)
    for i, row in enumerate(rows):
        if row is not None:
            image[i] = 0
            image[i, list(row)] = 1
    cmap = plt.colormaps["Blues"].copy()
    cmap.set_bad("#dddddd")
    axes[1].imshow(image, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    axes[1].set(yticks=range(4), yticklabels=names, xticks=range(0, 31, 3), xlabel="Feature position", title=f"Gray = decode failure; guaranteed hard budget t={t}")
    for i, row in enumerate(rows):
        if row is None:
            axes[1].text(15, i, "no valid codeword", ha="center", va="center", color="#333333")
    return _finish(fig, "Confidence can rescue a corrupted parity sign; incorrect confidence can fail")


def plot_recovery_trace(methods, design, result, truth, *, group=None, selected_bin=None):
    occupancy = np.zeros((design.C, 1 << design.b), dtype=int)
    for c, matrix in enumerate(design.matrices):
        for support in truth:
            occupancy[c, methods.bin_index(matrix, support)] += 1
    if group is None:
        group, default_bin = map(int, np.unravel_index(np.argmax(occupancy), occupancy.shape))
        if selected_bin is None:
            selected_bin = default_bin
    if selected_bin is None:
        selected_bin = int(np.argmax(occupancy[group]))
    snapshots = [result.initial_bins[group, :, selected_bin].copy()]
    residual = result.initial_bins.copy()
    notes = []
    for sweep in result.trace:
        for update in sweep["updates"]:
            signature = methods.boolean_character(design.delays, update["support"])
            for c, index in update["touched_bins"]:
                residual[c, :, index] -= update["amplitude_delta"] * signature
                if c == group and index == selected_bin and abs(update["amplitude_delta"]) > 1e-4:
                    notes.append(f"sweep {sweep['iteration']+1}: {_label(update['support'])} ({update['amplitude_delta']:+.3f})")
        snapshots.append(residual[group, :, selected_bin].copy())
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    im = axes[0, 0].imshow(occupancy, cmap="Blues", aspect="auto", vmin=0)
    for c in range(design.C):
        for j in range(1 << design.b):
            axes[0, 0].text(j, c, str(occupancy[c, j]), ha="center", va="center", color="white" if occupancy[c, j] > .6 * occupancy.max() else "black")
    axes[0, 0].plot(selected_bin, group, "s", markerfacecolor="none", markeredgecolor=COLORS["dense"], markersize=26, markeredgewidth=2)
    axes[0, 0].set(title="Exact truth: how many terms share each bin?", xlabel="Bin ID", ylabel="Hash group", yticks=range(design.C))
    fig.colorbar(im, ax=axes[0, 0], label="Occupancy", ticks=range(int(occupancy.max())+1), shrink=.8)
    snapshot = np.asarray(snapshots).T
    limit = max(float(np.max(np.abs(snapshot))), 1e-9)
    im = axes[0, 1].imshow(snapshot, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    axes[0, 1].set(title=f"Group {group}, bin {selected_bin}: signed residual shifts", xlabel="After sweep (0 = original bin)", ylabel="Shift row", xticks=range(len(snapshots)))
    axes[0, 1].set_yticks(range(0, len(design.delays), 3))
    fig.colorbar(im, ax=axes[0, 1], label="Residual coefficient", shrink=.8)
    energies = [float(np.sum(result.initial_bins ** 2))] + [r["energy_after"] for r in result.trace]
    axes[1, 0].semilogy(range(len(energies)), np.maximum(energies, 1e-20), "o-", color=COLORS["sparse"], label="Student peeling")
    axes[1, 0].axhline(energies[0], color=COLORS["dense"], linestyle="--", label="Control: no subtraction")
    axes[1, 0].set(title="Each signed message changes connected bins", xlabel="Completed sweep", ylabel="Total residual bin energy", xticks=range(len(energies)))
    axes[1, 0].legend(fontsize=9)
    terms = sorted(set(truth) | set(result.spectrum), key=lambda s: (len(s), s))
    x = np.arange(len(terms))
    axes[1, 1].bar(x - .18, [truth.get(s, 0) for s in terms], .36, color="#666666", label="Signed exact truth")
    axes[1, 1].bar(x + .18, [result.spectrum.get(s, 0) for s in terms], .36, color=COLORS["sparse"], label="Recovered")
    axes[1, 1].set_xticks(x, [_label(s) for s in terms], rotation=65, ha="right")
    axes[1, 1].axhline(0, color="black", linewidth=.6)
    axes[1, 1].set(title=f"False positives: {len(set(result.spectrum)-set(truth))}; missed: {len(set(truth)-set(result.spectrum))}", ylabel="Signed Fourier coefficient")
    axes[1, 1].legend(fontsize=9)
    fig.text(.01, .005, "Selected-bin messages: " + "; ".join(notes[:5]), fontsize=9)
    lead = "A collision becomes signed interactions" if occupancy.max() > 1 else "Signed recovery without an initial collision"
    return _finish(fig, f"{lead}: {design.requested_count} requests, {design.unique_count} unique masks")


def plot_budget_curves(payload):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for game in payload["truths"]:
        selected = [r for r in payload["records"] if r["game"] == game]
        for ax, key in zip(axes, ("nrmse", "recall", "false_positives")):
            x, mean, low, high = [], [], [], []
            for b in payload["budgets"]:
                rows = [r for r in selected if r["b"] == b]
                values = np.asarray([r[key] for r in rows])
                x.append(rows[0]["requested"])
                mean.append(values.mean()); low.append(values.min()); high.append(values.max())
            ax.plot(x, mean, "o-", color=COLORS[game], label=game)
            ax.fill_between(x, low, high, alpha=.10, color=COLORS[game])
    sparse = [r for r in payload["records"] if r["game"] == "sparse"]
    axes[0].plot([next(r["requested"] for r in sparse if r["b"] == b) for b in payload["budgets"]],
                 [np.mean([r["baseline_nrmse"]["additive"] for r in sparse if r["b"] == b]) for b in payload["budgets"]],
                 "k--", label="Sparse game: additive baseline")
    axes[0].axhline(0, color="black", linewidth=.6)
    axes[1].axhline(1, color="black", linestyle="--", linewidth=.7, label="Exact support")
    axes[2].axhline(0, color="black", linewidth=.6)
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Requested analytic masks (duplicates included)")
        ax.grid(alpha=.15)
    axes[0].set(title="Fidelity on 4,096 independent masks", ylabel="RMSE / true-score standard deviation")
    axes[1].set(title="Did the true support appear?", ylabel="Fraction of true supports recovered", ylim=(-.03, 1.06))
    axes[2].set(title="An output can contain false interactions", ylabel="False-positive support count")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    return _finish(fig, "Sparse recovery needs sparsity and a sufficient degree budget (mean and seed range)")


def plot_query_costs(payload):
    rows = [r for r in payload["records"] if r["game"] == "sparse"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    for seed in payload["seeds"]:
        selected = [r for r in rows if r["seed"] == seed]
        axes[0].plot([r["requested"] for r in selected], [r["unique"] for r in selected], "o-", label=f"seed {seed}")
        axes[1].plot([r["unique"] for r in selected], [r["nrmse"] for r in selected], "o-", label=f"seed {seed}")
    limit = max(r["requested"] for r in rows)
    axes[0].plot([0, limit], [0, limit], "k--", label="Every request unique")
    axes[0].set(title="Count duplicates before discussing inference cost", xlabel="Requested masks", ylabel="Distinct masks")
    axes[1].axhline(0, color="black", linestyle="--", linewidth=.7, label="Exact prediction")
    axes[1].set(title="The same sparse-game runs, by unique cost", xlabel="Distinct training masks", ylabel="Normalized audit RMSE")
    for ax in axes:
        ax.legend(fontsize=9); ax.grid(alpha=.15)
    return _finish(fig, f"The {payload['n']}-feature cube has {2**payload['n']:,} masks; we never enumerate it")


def plot_attribution_conversion(methods):
    truth = {(1, 3, 7): -1.25}
    additive = {(1,): -1.25 / 3, (3,): -1.25 / 3, (7,): -1.25 / 3}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    supports = [(1,), (3,), (7,), (1, 3, 7)]
    x = np.arange(4)
    axes[0].bar(x - .18, [truth.get(s, 0) for s in supports], .36, label="Three-way parity", color=COLORS["sparse"])
    axes[0].bar(x + .18, [additive.get(s, 0) for s in supports], .36, label="Additive control", color=COLORS["additive"])
    axes[0].set_xticks(x, [_label(s) for s in supports], rotation=45, ha="right")
    axes[0].set(title="Different Fourier spectra", ylabel="Fourier coefficient")
    values = [methods.fourier_to_shapley(game, 8)[[1, 3, 7]] for game in (truth, additive)]
    axes[1].bar(np.arange(3) - .18, values[0], .36, color=COLORS["sparse"])
    axes[1].bar(np.arange(3) + .18, values[1], .36, color=COLORS["additive"])
    axes[1].set(xticks=range(3), xticklabels=["x1", "x3", "x7"], title="Identical single-feature Shapley values", ylabel="Shapley value")
    converted = [methods.fourier_to_sii(game, [(1, 3), (1, 3, 7)]) for game in (truth, additive)]
    for i, (game, color) in enumerate(zip(converted, (COLORS["sparse"], COLORS["additive"]))):
        axes[2].bar(np.arange(2) + (i - .5) * .36, list(game.values()), .36, color=color)
    axes[2].set(xticks=range(2), xticklabels=["Pair {1,3}", "Triple {1,3,7}"], title="SII answers a different question", ylabel="Shapley interaction index")
    for ax in axes:
        ax.axhline(0, color="black", linewidth=.6)
    axes[0].legend(fontsize=9)
    return _finish(fig, "A Fourier coefficient is not a Shapley interaction")


def plot_native_comparison(payload, *, prompt_index=0, budget=None, seed=0):
    """Plot completed records only; this function never initiates native work."""
    prompts = payload["prompts"]
    prompt = next((p for i, p in enumerate(prompts) if p.get("prompt_index", i) == prompt_index), None)
    if prompt is None:
        raise ValueError("No such prompt in the completed payload")
    candidates = [r for r in payload["results"] if r["prompt_index"] == prompt_index and r["method"] == "spex" and r["seed"] == seed]
    if not candidates:
        raise ValueError("No completed SPEX records for this prompt/seed")
    if budget is None:
        budget = max(r["budget"] for r in candidates)
    real = next((r for r in candidates if r["budget"] == budget and r["condition"] == "real"), None)
    shuffled = next((r for r in candidates if r["budget"] == budget and r["condition"] == "shuffled"), None)
    if real is None:
        raise ValueError("No completed real-response run at the selected budget")
    truth = np.asarray(real["audit_values"])
    prediction = np.asarray(real["predictions"]["final"])
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    axes[0].scatter(truth, prediction, s=10, alpha=.4, color=COLORS["sparse"], label="SPEX")
    for name, color in (("additive", COLORS["additive"]), ("constant", "#888888")):
        axes[0].scatter(truth, real["baseline_predictions"][name], s=5, alpha=.2, color=color, label=name)
    limits = [min(float(truth.min()), float(prediction.min())), max(float(truth.max()), float(prediction.max()))]
    axes[0].plot(limits, limits, "k--", label="Perfect prediction")
    axes[0].set(title="Independent masks of this prompt", xlabel="Actual model score", ylabel="Predicted score")
    axes[0].legend(fontsize=8)
    spectrum = real["spectrum"]
    supports = sorted(spectrum, key=lambda s: -abs(spectrum[s]))[:12]
    axes[1].barh(range(len(supports)), [spectrum[s] for s in supports], color=[COLORS["sparse"] if spectrum[s] >= 0 else COLORS["dense"] for s in supports])
    axes[1].set_yticks(range(len(supports)), [_label(s, prompt["feature_names"]) for s in supports], fontsize=8)
    axes[1].invert_yaxis(); axes[1].axvline(0, color="black", linewidth=.6)
    axes[1].set(title="Largest recovered signed terms", xlabel="Fourier coefficient (not SII)")
    names, errors = [], []
    for label, values in [("SPEX", prediction), *real["baseline_predictions"].items()]:
        names.append(label); errors.append(float(np.sqrt(np.mean((np.asarray(values) - truth) ** 2))))
    if shuffled is not None:
        names.append("Shuffled response")
        errors.append(float(np.sqrt(np.mean((np.asarray(shuffled["predictions"]["final"]) - truth) ** 2))))
    axes[2].bar(range(len(names)), errors, color=[COLORS["sparse"], "#888888", COLORS["additive"], COLORS["shuffled"]][:len(names)])
    axes[2].set_xticks(range(len(names)), names, rotation=35, ha="right")
    axes[2].set(title="Same audit; simple controls remain visible", ylabel="Audit RMSE")
    receipt = real["query_receipt"]
    title = f"Prompt {prompt_index}, seed {seed}: {receipt['raw_requested']} requests, {receipt['unique_model_masks']} unique masks"
    return _finish(fig, title)
