"""Plumbing and plotting helpers for [16.6] Sparse Interaction Discovery with SPEX and ProxySPEX.

Nothing in here is part of the methods you're learning. It holds:

- finite-field lookup tables (the arithmetic that BCH codes are built on),
- generators for planted test games with known spectra,
- tokenisation of a sentence into maskable words,
- plotting code.

All of the transforms, the SPEX decoder and the ProxySPEX extraction live in the notebook.
"""

import itertools
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
import torch as t

# =====================================================================================
# Finite field GF(2^m), used to build BCH codes
# =====================================================================================

# Primitive polynomials, written as integers (bit i = coefficient of x^i). E.g. 0b100101 = x^5 + x^2 + 1.
PRIMITIVE_POLYS = {
    2: 0b111,
    3: 0b1011,
    4: 0b10011,
    5: 0b100101,
    6: 0b1000011,
    7: 0b10001001,
    8: 0b100011101,
    9: 0b1000010001,
    10: 0b10000001001,
    11: 0b100000000101,
    12: 0b1000001010011,
    13: 0b10000000011011,
}


class GaloisField:
    """
    The field GF(2^m). Elements are integers 0..2^m-1, read as m-bit vectors (bit i = coefficient of x^i).

    Addition is XOR. Multiplication uses log / antilog tables built from a primitive element alpha:

        exp[i] = alpha^i        (as an integer, for i = 0, ..., 2 * N - 1, where N = 2^m - 1)
        log[a] = i such that alpha^i = a     (log[0] = -1, since 0 has no logarithm)

    Every nonzero element is a power of alpha, and alpha^N = 1.
    """

    def __init__(self, m: int):
        self.m, self.N = m, 2**m - 1
        self.exp = np.zeros(2 * self.N, dtype=np.int64)
        self.log = np.full(2**m, -1, dtype=np.int64)
        x = 1
        for i in range(self.N):
            self.exp[i], self.log[x] = x, i
            x <<= 1
            if x >> m:
                x ^= PRIMITIVE_POLYS[m]
        self.exp[self.N :] = self.exp[: self.N]

    def mul(self, a: int, b: int) -> int:
        if a == 0 or b == 0:
            return 0
        return int(self.exp[(self.log[a] + self.log[b]) % self.N])

    def inv(self, a: int) -> int:
        assert a != 0, "0 has no inverse"
        return int(self.exp[(-self.log[a]) % self.N])


@lru_cache(maxsize=None)
def galois_field(m: int) -> GaloisField:
    """A cached GaloisField(m). Building the tables is a Python loop over 2^m elements, so build each field once."""
    return GaloisField(m)


def bch_field_degree(n: int) -> int:
    """Smallest m >= 2 such that a length-(2^m - 1) BCH code has room for n features."""
    m = 2
    while 2**m - 1 < n:
        m += 1
    return m


# =====================================================================================
# Planted games with known spectra
# =====================================================================================


def make_sparse_game(
    n: int, s: int, degrees: list[int], seed: int, amplitude: tuple[float, float] = (0.5, 1.5), constant: float = 1.0
) -> dict[tuple[int, ...], float]:
    """
    A random Fourier spectrum with a constant term plus `s` nonzero coefficients. Each support has a size drawn
    uniformly from `degrees`, and each coefficient has a random sign and magnitude uniform in `amplitude`.
    """
    rng = np.random.default_rng(seed)
    spectrum = {(): constant}
    while len(spectrum) < s + 1:
        d = int(rng.choice(degrees))
        support = tuple(sorted(rng.choice(n, d, replace=False).tolist()))
        if support not in spectrum:
            spectrum[support] = float(rng.choice([-1, 1]) * rng.uniform(*amplitude))
    return spectrum


def make_and_game(n: int, s: int, degrees: list[int], seed: int) -> dict[tuple[int, ...], float]:
    """
    A game that is a sum of `s` random AND terms, f(x) = sum_r w_r * prod_{i in S_r} x_i, returned as its
    *Fourier* spectrum. (Each AND of degree d expands into 2^d Fourier terms, using x_i = (1 - chi_i) / 2.)
    """
    rng = np.random.default_rng(seed)
    terms = {}
    while len(terms) < s:
        d = int(rng.choice(degrees))
        support = tuple(sorted(rng.choice(n, d, replace=False).tolist()))
        terms[support] = float(rng.choice([-1, 1]) * rng.uniform(0.5, 1.5))
    spectrum = {}
    for support, w in terms.items():
        for r in range(len(support) + 1):
            for sub in itertools.combinations(support, r):
                spectrum[sub] = spectrum.get(sub, 0.0) + w * (-1) ** r / 2 ** len(support)
    return spectrum


# =====================================================================================
# Word-level tokenisation
# =====================================================================================


def tokenize_words(model, text: str, final: str) -> tuple[list[str], t.Tensor, t.Tensor]:
    """
    Splits `text` on spaces into words, and tokenises it as BOS + words + `final`.

    Returns:
        words:          list of the n maskable words
        tokens:         (1, seq) token ids
        token_to_word:  (seq,) tensor; entry i is the index of the word that token i belongs to, or -1 for tokens
                        that are never masked (BOS and the tokens of `final`)

    Each word is tokenised with a leading space (except the first), so a word may span several tokens. Masking
    a word masks all of its tokens.
    """
    words = text.split(" ")
    word_tokens = [model.to_tokens(w if i == 0 else " " + w, prepend_bos=False)[0].tolist() for i, w in enumerate(words)]
    final_tokens = model.to_tokens(final, prepend_bos=False)[0].tolist()
    ids = [model.tokenizer.bos_token_id] + [tok for wt in word_tokens for tok in wt] + final_tokens
    token_to_word = [-1] + [i for i, wt in enumerate(word_tokens) for _ in wt] + [-1] * len(final_tokens)
    device = model.cfg.device
    return words, t.tensor(ids, device=device)[None], t.tensor(token_to_word, device=device)


def support_label(support: tuple[int, ...], words: list[str]) -> str:
    """'Mary × gave' style label for a support. Words are tagged with their index, so repeated words stay distinct."""
    if len(support) == 0:
        return "(constant)"
    return " × ".join(f"{words[i].strip(',.')}[{i}]" for i in support)


# =====================================================================================
# Plotting
# =====================================================================================


def _all_supports(n: int) -> list[tuple[int, ...]]:
    return [s for r in range(n + 1) for s in itertools.combinations(range(n), r)]


def plot_toy_spectra(games: dict[str, tuple[dict, dict]], n: int):
    """
    games: name -> (fourier spectrum, mobius spectrum), each a dict support -> coefficient.
    One row per game, Fourier on the left and Möbius on the right, every support of n features on the x-axis.
    """
    supports = _all_supports(n)
    labels = ["∅" if not s else "".join(str(i) for i in s) for s in supports]
    fig, axes = plt.subplots(len(games), 2, figsize=(12, 2.4 * len(games)), sharex=True)
    for row, (name, (fourier, mobius)) in enumerate(games.items()):
        for col, (basis, spec) in enumerate([("Fourier", fourier), ("Möbius", mobius)]):
            ax = axes[row, col]
            vals = [spec.get(s, 0.0) for s in supports]
            nonzero = sum(abs(v) > 1e-12 for v in vals)
            colors = ["tab:blue" if v >= 0 else "tab:red" for v in vals]
            ax.bar(range(len(supports)), vals, color=colors)
            ax.axhline(0, color="black", lw=0.5)
            ax.set_title(f"{name}: {basis} coefficients ({nonzero} nonzero)", fontsize=10)
            ax.set_ylabel("coefficient")
            ax.set_xticks(range(len(supports)))
            ax.set_xticklabels(labels, rotation=90, fontsize=7)
            ax.grid(alpha=0.3, axis="y")
    for ax in axes[-1]:
        ax.set_xlabel("support S (features in the interaction)")
    plt.tight_layout()
    plt.show()


def plot_sparsity(spectra: dict[str, np.ndarray], degree_energy_fn, topk_energy_fn, max_degree: int = 8):
    """Two panels: energy fraction per interaction degree, and cumulative energy of the top-k coefficients."""
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 4.2))
    width = 0.8 / len(spectra)
    styles = ["-", "--", ":", "-."]
    for idx, (name, F) in enumerate(spectra.items()):
        de = degree_energy_fn(F)[:max_degree]
        ax0.bar(np.arange(1, len(de) + 1) + (idx - (len(spectra) - 1) / 2) * width, de, width=width, label=name)
        curve = topk_energy_fn(F)
        ks = np.arange(1, len(curve) + 1)
        ax1.plot(ks, curve, styles[idx % len(styles)], label=name, lw=2)
    ax0.set_xlabel("interaction degree |S|")
    ax0.set_ylabel("fraction of variance (non-constant energy)")
    ax0.set_title("Where does the variance live? Energy by degree")
    ax0.set_xticks(range(1, max_degree + 1))
    ax0.grid(alpha=0.3, axis="y")
    ax0.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, fontsize=8)
    ax1.set_xscale("log")
    ax1.set_xlabel("number of Fourier coefficients kept (largest first)")
    ax1.set_ylabel("fraction of variance explained (R²)")
    ax1.set_title("Best k-term approximation (Parseval)")
    ax1.set_ylim(0, 1.02)
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_bins(U0: np.ndarray, occupancy: list[list[tuple]], title: str):
    """
    Bar chart of |U_0(j)| for every bin j of one hashing group, coloured by how many true coefficients the bin holds.
    occupancy[j] is the list of true (non-constant or constant) supports hashing into bin j.
    """
    kinds = ["zero-ton (empty)", "singleton", "multi-ton (collision)"]
    colors = {kinds[0]: "lightgray", kinds[1]: "tab:green", kinds[2]: "tab:red"}
    fig, ax = plt.subplots(figsize=(11, 3.6))
    for j, u in enumerate(U0):
        kind = kinds[min(len(occupancy[j]), 2)]
        ax.bar(j, abs(u), color=colors[kind], label=kind)
        if len(occupancy[j]) >= 1:
            ax.text(j, abs(u) + 0.03, str(len(occupancy[j])), ha="center", fontsize=8)
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.set_xlabel("bin j  (= M^T k for every coefficient k in the bin)")
    ax.set_ylabel("|U_0(j)|  (zero-delay bin value)")
    ax.set_title(title + "\n(numbers above bars = how many true coefficients share the bin)", fontsize=10)
    ax.set_xticks(range(len(U0)))
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.show()


def plot_peeling_trace(trace: list[np.ndarray], title: str):
    """Heatmap of the residual energy of every bin (all hashing groups side by side) after each peeling sweep."""
    energies = np.stack([(U**2).sum(axis=1).reshape(-1) for U in trace])  # (sweeps + 1, groups * bins)
    n_groups, n_bins = trace[0].shape[0], trace[0].shape[2]
    fig, ax = plt.subplots(figsize=(12, 0.45 * len(trace) + 1.8))
    im = ax.imshow(np.log10(energies + 1e-12), aspect="auto", cmap="magma", vmin=-12, vmax=np.log10(energies.max()))
    for g in range(1, n_groups):
        ax.axvline(g * n_bins - 0.5, color="white", lw=2)
    ax.set_yticks(range(len(trace)))
    ax.set_yticklabels(["before peeling"] + [f"after sweep {i}" for i in range(1, len(trace))])
    ax.set_xticks([g * n_bins + n_bins / 2 - 0.5 for g in range(n_groups)])
    ax.set_xticklabels([f"hashing group {g}" for g in range(n_groups)])
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, label="log10 residual bin energy")
    plt.tight_layout()
    plt.show()


def plot_recovery_curves(curves: dict[str, tuple[list[int], list[float]]], title: str, controls: tuple[str, ...] = ()):
    """R² against number of queries (log x-axis). Curves whose name is in `controls` are dashed and gray-ish."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    markers = itertools.cycle(["o", "s", "^", "D", "v", "P"])
    for name, (queries, r2) in curves.items():
        is_control = name in controls
        ax.plot(queries, r2, "--" if is_control else "-", marker=next(markers), label=name,
                alpha=0.8 if is_control else 1.0, color=None)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("number of queries to the value function (log scale)")
    ax.set_ylabel("R² on held-out random masks")
    ax.set_ylim(-0.1, 1.05)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.show()


def plot_delay_cost(ns: list[int], costs: dict[str, list[int]]):
    """Number of delays (shifted query patterns per hashing group) needed by each delay design, against n."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, c in costs.items():
        ax.plot(ns, c, "-o", label=name, markersize=4)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("number of features n")
    ax.set_ylabel("delays per hashing group (p + 1)")
    ax.set_title("Cost of identifying a support: identity delays vs BCH-coded delays")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_method_bars(results: dict[str, dict[str, float]], title: str, ylim: tuple[float, float] = (-0.3, 1.05)):
    """Grouped bar chart: results[game][method] = R². Values below the y-limit are clipped and annotated."""
    games = list(results)
    methods = list(results[games[0]])
    width = 0.8 / len(methods)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for k, method in enumerate(methods):
        xs = np.arange(len(games)) + (k - (len(methods) - 1) / 2) * width
        vals = np.array([results[g][method] for g in games])
        bars = ax.bar(xs, np.clip(vals, ylim[0], None), width=width, label=method)
        for x, v in zip(xs, vals):
            ax.text(x, max(v, ylim[0]) + 0.02, f"{v:.2f}", ha="center", fontsize=7, rotation=90)
    ax.set_xticks(range(len(games)))
    ax.set_xticklabels(games)
    ax.set_ylim(*ylim)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("R² on held-out random masks")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8, ncol=len(methods), loc="upper center", bbox_to_anchor=(0.5, -0.1))
    plt.tight_layout()
    plt.show()


def plot_head_to_head(panels: dict[str, dict], title: str):
    """
    panels: readout name -> {"curves": {method: (queries, r2)}, "hlines": {label: value}, "controls": {method: (queries, r2)}}
    One panel per readout, R² (exact, on all masks) against number of queries.
    """
    fig, axes = plt.subplots(1, len(panels), figsize=(7 * len(panels), 4.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (name, panel) in zip(axes, panels.items()):
        markers = itertools.cycle(["o", "s", "^", "D"])
        for method, (q, r2) in panel["curves"].items():
            ax.plot(q, r2, "-", marker=next(markers), label=method)
        for method, (q, r2) in panel.get("controls", {}).items():
            ax.plot(q, np.clip(r2, -0.25, None), ":x", color="gray", label=method + " (clipped at -0.25)")
        for (label, value), ls in zip(panel.get("hlines", {}).items(), ["--", "-.", ":"]):
            ax.axhline(value, ls=ls, color="black", lw=1, label=f"{label} = {value:.2f}")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("number of model queries (log scale)")
        ax.set_title(name, fontsize=10)
        ax.set_ylim(-0.3, 1.0)
        ax.axhline(0, color="black", lw=0.5)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="lower right")
    axes[0].set_ylabel("R² against the exact game (all 2^n masks)")
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.show()


def plot_sii_heatmaps(matrices: dict[str, np.ndarray], words: list[str], title: str):
    """Side-by-side heatmaps of pairwise Shapley interaction indices (symmetric n x n matrices)."""
    fig, axes = plt.subplots(1, len(matrices), figsize=(5.3 * len(matrices), 5.0))
    axes = np.atleast_1d(axes)
    vmax = max(np.abs(m).max() for m in matrices.values())
    labels = [f"{w.strip(',.')}[{i}]" for i, w in enumerate(words)]
    for ax, (name, mat) in zip(axes, matrices.items()):
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(words)))
        ax.set_yticks(range(len(words)))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_title(name, fontsize=10)
    fig.colorbar(im, ax=list(axes), shrink=0.8, label="pairwise Shapley interaction index")
    fig.suptitle(title, fontsize=11)
    plt.show()


def plot_live_bars(values: dict[str, float], title: str):
    """Bar chart of held-out R² for the live (no exact reference) experiment; controls drawn in gray."""
    fig, ax = plt.subplots(figsize=(8, 3.8))
    names, vals = list(values), list(values.values())
    colors = ["tab:gray" if "control" in name else "tab:blue" for name in names]
    bars = ax.bar(names, np.clip(vals, -0.25, None), color=colors)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, max(v, -0.25) + 0.02, f"{v:.2f}", ha="center")
    ax.set_ylim(-0.3, 1.05)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("R² on held-out random masks")
    ax.set_title(title, fontsize=10)
    plt.xticks(rotation=15, ha="right")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.show()


def plot_method_comparison(
    results: dict[str, dict[str, float]],
    sizes: dict[str, dict[str, int]],
    true_sizes: dict[str, int],
    title: str,
    ylim: tuple[float, float] = (-0.3, 1.05),
):
    """
    Two panels for planted games. Left: R² per game and method (clipped at the lower y-limit). Right: number of terms
    in each method's explanation (log scale), with the true number of Fourier terms of each game marked.
    """
    games = list(results)
    methods = list(results[games[0]])
    width = 0.8 / len(methods)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw={"width_ratios": [1.3, 1]})
    for k, method in enumerate(methods):
        xs = np.arange(len(games)) + (k - (len(methods) - 1) / 2) * width
        vals = np.array([results[g][method] for g in games])
        ax0.bar(xs, np.clip(vals, ylim[0], None), width=width, label=method, color=colors[k])
        for x, v in zip(xs, vals):
            ax0.text(x, max(v, ylim[0]) + 0.02, f"{v:.2f}", ha="center", fontsize=7, rotation=90)
    ax0.set_xticks(range(len(games)))
    ax0.set_xticklabels(games, fontsize=8)
    ax0.set_ylim(*ylim)
    ax0.axhline(0, color="black", lw=0.5)
    ax0.set_ylabel("R² on held-out random masks")
    ax0.set_title("Faithfulness", fontsize=10)
    ax0.grid(alpha=0.3, axis="y")
    ax0.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    size_methods = list(sizes[games[0]])
    width = 0.8 / len(size_methods)
    for k, method in enumerate(size_methods):
        xs = np.arange(len(games)) + (k - (len(size_methods) - 1) / 2) * width
        ax1.bar(xs, [sizes[g][method] for g in games], width=width, label=method, color=colors[methods.index(method)])
    for g_idx, g in enumerate(games):
        ax1.plot([g_idx - 0.45, g_idx + 0.45], [true_sizes[g]] * 2, "k--", lw=1.5, label="true number of non-constant terms" if g_idx == 0 else None)
    ax1.set_yscale("log")
    ax1.set_xticks(range(len(games)))
    ax1.set_xticklabels(games, fontsize=8)
    ax1.set_ylabel("terms holding 99% of the energy (log scale)")
    ax1.set_title("Effective size of the explanation (non-constant terms)", fontsize=10)
    ax1.grid(alpha=0.3, axis="y")
    ax1.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.show()
