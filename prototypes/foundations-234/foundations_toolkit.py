# /// script
# requires-python = ">=3.12"
# dependencies = ["torch", "numpy", "scikit-learn", "matplotlib"]
# ///
"""PROTOTYPE for issue #234 (throwaway branch `prototype/foundations-234`, never merged).

Solutions, tests and figures for the two worked parts drafted in `0.7_foundations_toolkit_draft.md`
(Part 1: KL divergence, Part 3: t-SNE). CPU only; runs in well under a minute.

    uv run prototypes/foundations-234/foundations_toolkit.py

Every number quoted in the draft pages is printed by this script (see `run_output.txt`).
"""

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch as t
from sklearn.manifold import TSNE

FIG = Path(__file__).parent / "figures"
FIG.mkdir(exist_ok=True)
t.set_default_dtype(t.float64)


# ==========================================================================================
# Part 1: KL divergence
# ==========================================================================================


def kl_divergence(p: t.Tensor, q: t.Tensor) -> t.Tensor:
    """KL(p || q) in nats, summed over the last dimension. p and q are probability vectors."""
    # xlogy(a, b) = a * log(b), with the convention 0 * log(anything) = 0.
    return (t.special.xlogy(p, p) - t.special.xlogy(p, q)).sum(-1)


def test_kl_divergence(kl_fn) -> None:
    P, Q = t.tensor([0.9, 0.1]), t.tensor([0.5, 0.5])
    t.testing.assert_close(kl_fn(P, Q), t.tensor(0.368064), atol=1e-6, rtol=0)
    t.testing.assert_close(kl_fn(Q, P), t.tensor(0.510826), atol=1e-6, rtol=0)
    t.testing.assert_close(kl_fn(P, P), t.tensor(0.0))
    # An outcome p never produces contributes nothing (0 log 0 = 0), so this is just log 2.
    t.testing.assert_close(kl_fn(t.tensor([1.0, 0.0]), t.tensor([0.5, 0.5])), t.tensor(math.log(2)))
    # q rules out something p produces: infinitely surprised.
    assert kl_fn(t.tensor([0.5, 0.5]), t.tensor([1.0, 0.0])).item() == math.inf
    # Batched over leading dimensions, never negative.
    gen = t.Generator().manual_seed(0)
    p = t.randn(4, 7, 10, generator=gen).softmax(-1)
    q = t.randn(4, 7, 10, generator=gen).softmax(-1)
    out = kl_fn(p, q)
    assert out.shape == (4, 7), f"expected shape (4, 7), got {tuple(out.shape)}"
    assert (out >= 0).all()
    print("All tests in `test_kl_divergence` passed!")


def part1_figures() -> None:
    P, Q = t.tensor([0.9, 0.1]), t.tensor([0.5, 0.5])
    fwd_terms = t.special.xlogy(P, P) - t.special.xlogy(P, Q)
    rev_terms = t.special.xlogy(Q, Q) - t.special.xlogy(Q, P)
    print("\n## Part 1: weather forecaster")
    print(f"H(P) = {-(t.special.xlogy(P, P)).sum():.6f}  H(P,Q) = {-(t.special.xlogy(P, Q)).sum():.6f}")
    print(f"KL(P||Q) = {fwd_terms.sum():.6f}  per outcome (sun, rain) = {fwd_terms.tolist()}")
    print(f"KL(Q||P) = {rev_terms.sum():.6f}  per outcome (sun, rain) = {rev_terms.tolist()}")
    Q2 = t.tensor([0.99, 0.01])
    print(
        f"changed forecaster Q2=(.99,.01): KL(P||Q2) = {kl_divergence(P, Q2):.6f}  KL(Q2||P) = {kl_divergence(Q2, P):.6f}"
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    x = np.arange(2)
    axes[0].bar(x - 0.2, P, 0.4, label="P: how often it happens", color="#147D7E")
    axes[0].bar(x + 0.2, Q, 0.4, label="Q: the forecast", color="#E0A030")
    axes[0].set_xticks(x, ["sun", "rain"])
    axes[0].set_ylabel("probability")
    axes[0].set_title("A town that is sunny 90% of the time,\na forecaster who always says 50/50")
    axes[0].legend(frameon=False)
    axes[1].bar(x - 0.2, fwd_terms, 0.4, label=f"KL(P‖Q) terms, sum = {fwd_terms.sum():.3f}", color="#147D7E")
    axes[1].bar(x + 0.2, rev_terms, 0.4, label=f"KL(Q‖P) terms, sum = {rev_terms.sum():.3f}", color="#C73E3A")
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_xticks(x, ["sun", "rain"])
    axes[1].set_ylabel("extra surprise per outcome (nats)")
    axes[1].set_title("Per-outcome terms can be negative;\nthe sum never is")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "kl_weather.png", dpi=130)
    plt.close(fig)

    # Toy logit lens: five-token vocabulary, prompt "The Eiffel Tower is in".
    tokens = [" Paris", " France", " the", " Europe", " a"]
    p_final = t.tensor([0.85, 0.10, 0.03, 0.01, 0.01])
    lens_a = t.full((5,), 0.2)
    lens_b = t.tensor([0.98, 1e-8, 0.01, 0.005, 0.005])
    lens_b = lens_b / lens_b.sum()
    kl = {
        ("A", "fwd"): kl_divergence(p_final, lens_a).item(),
        ("B", "fwd"): kl_divergence(p_final, lens_b).item(),
        ("A", "rev"): kl_divergence(lens_a, p_final).item(),
        ("B", "rev"): kl_divergence(lens_b, p_final).item(),
    }
    print("\n## Part 1: toy logit lens")
    for (lens, d), v in kl.items():
        print(f"lens {lens} {'KL(final||lens)' if d == 'fwd' else 'KL(lens||final)'} = {v:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.6, 1]})
    x = np.arange(5)
    axes[0].bar(x - 0.27, p_final, 0.27, label="final layer", color="#64748B")
    axes[0].bar(x, lens_a, 0.27, label="lens A (knows nothing)", color="#E0A030")
    axes[0].bar(x + 0.27, lens_b, 0.27, label="lens B (confident)", color="#7C4DBA")
    axes[0].set_yscale("log")
    axes[0].set_ylim(1e-9, 2)
    axes[0].set_xticks(x, tokens)
    axes[0].set_ylabel("probability (log scale)")
    axes[0].set_title('"The Eiffel Tower is in" (toy 5-token vocabulary)')
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    xx = np.arange(2)
    axes[1].bar(xx - 0.2, [kl[("A", "fwd")], kl[("B", "fwd")]], 0.4, label="KL(final ‖ lens)", color="#147D7E")
    axes[1].bar(xx + 0.2, [kl[("A", "rev")], kl[("B", "rev")]], 0.4, label="KL(lens ‖ final)", color="#C73E3A")
    axes[1].set_xticks(xx, ["lens A", "lens B"])
    axes[1].set_ylabel("nats")
    axes[1].set_title("The ranking flips with the direction")
    axes[1].legend(frameon=False, fontsize=8, loc="upper left")
    axes[1].set_ylim(0, 1.9)
    fig.tight_layout()
    fig.savefig(FIG / "kl_lens.png", dpi=130)
    plt.close(fig)


# ==========================================================================================
# Part 3: t-SNE
# ==========================================================================================


def conditional_neighbour_probs(X: t.Tensor, sigmas: t.Tensor) -> t.Tensor:
    """P[i, j] = p(j | i): the probability that point i picks j as its neighbour, using a Gaussian of
    width sigmas[i] centred on x_i. The diagonal is zero and every row sums to one."""
    sq_dists = t.cdist(X, X).pow(2)
    logits = -sq_dists / (2 * sigmas[:, None] ** 2)
    logits.fill_diagonal_(-math.inf)
    return logits.softmax(dim=-1)


def perplexity(P_cond: t.Tensor) -> t.Tensor:
    """Effective number of neighbours of each point: exp(entropy of its row), entropy in nats."""
    return t.exp(-t.special.xlogy(P_cond, P_cond).sum(-1))


def sigmas_for_perplexity(X: t.Tensor, target: float, iters: int = 64) -> t.Tensor:
    """Binary search (in log space) for each point's sigma so that its row has the target perplexity."""
    lo = t.full((X.shape[0],), -20.0)
    hi = t.full((X.shape[0],), 20.0)
    for _ in range(iters):
        mid = (lo + hi) / 2
        too_many = perplexity(conditional_neighbour_probs(X, mid.exp())) > target
        hi = t.where(too_many, mid, hi)
        lo = t.where(too_many, lo, mid)
    return ((lo + hi) / 2).exp()


def test_conditional_neighbour_probs(fn) -> None:
    X = t.tensor([[0.0], [1.0], [10.0]])
    P = fn(X, t.ones(3))
    t.testing.assert_close(P.diagonal(), t.zeros(3))
    t.testing.assert_close(P.sum(-1), t.ones(3))
    assert P[0, 1] > 0.999, "point 0 should almost surely pick its close neighbour (point 1)"
    # Equal distances: every other point is equally likely, so perplexity = n - 1 for any sigma.
    E = t.eye(5)
    for s in (0.1, 1.0, 10.0):
        t.testing.assert_close(perplexity(fn(E, t.full((5,), s))), t.full((5,), 4.0))
    # A wider Gaussian spreads the choice over more neighbours.
    gen = t.Generator().manual_seed(0)
    Y = t.randn(30, 5, generator=gen)
    assert (perplexity(fn(Y, t.full((30,), 2.0))) > perplexity(fn(Y, t.full((30,), 0.5)))).all()
    print("All tests in `test_conditional_neighbour_probs` passed!")


def three_clusters(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centres = [(0.0, 0.0), (8.0, 0.0), (40.0, 0.0)]
    spreads = [1.0, 0.1, 1.0]
    X = np.concatenate([rng.normal(c, s, size=(100, 2)) for c, s in zip(centres, spreads)])
    y = np.repeat(np.arange(3), 100)
    return X, y


def radius(Z: np.ndarray) -> float:
    return float(np.sqrt(((Z - Z.mean(0)) ** 2).sum(1).mean()))


def cluster_stats(Z: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """(radius of A / radius of B, gap B-C / gap A-B) for a 2-D layout."""
    cents = [Z[y == k].mean(0) for k in range(3)]
    size_ratio = radius(Z[y == 0]) / radius(Z[y == 1])
    gap_ratio = np.linalg.norm(cents[2] - cents[1]) / np.linalg.norm(cents[1] - cents[0])
    return size_ratio, float(gap_ratio)


COLOURS = np.array(["#147D7E", "#C73E3A", "#E0A030"])
NAMES = ["A (spread 1.0)", "B (spread 0.1)", "C (spread 1.0, far away)"]


def part3_figures() -> None:
    X, y = three_clusters()
    sig = sigmas_for_perplexity(t.tensor(X), target=30.0)
    print("\n## Part 3: three clusters")
    for k in range(3):
        print(f"cluster {NAMES[k]}: median sigma at perplexity 30 = {sig[t.tensor(y == k)].median():.3f}")
    s0, g0 = cluster_stats(X, y)
    print(f"original: radius A/B = {s0:.2f}, gap(B,C)/gap(A,B) = {g0:.2f}")

    def fit(data: np.ndarray, perp: float, seed: int = 0) -> np.ndarray:
        return TSNE(n_components=2, perplexity=perp, init="pca", random_state=seed).fit_transform(data)

    Z = fit(X, 30.0)
    s1, g1 = cluster_stats(Z, y)
    print(f"t-SNE perplexity 30 (seed 0): radius A/B = {s1:.2f}, gap(B,C)/gap(A,B) = {g1:.2f}")
    for seed in (1, 2):
        s, g = cluster_stats(fit(X, 30.0, seed), y)
        print(f"t-SNE perplexity 30 (seed {seed}): radius A/B = {s:.2f}, gap(B,C)/gap(A,B) = {g:.2f}")
    for perp in (5.0, 100.0):
        s, g = cluster_stats(fit(X, perp), y)
        print(f"t-SNE perplexity {perp:g} (seed 0): radius A/B = {s:.2f}, gap(B,C)/gap(A,B) = {g:.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for k in range(3):
        axes[0].scatter(*X[y == k].T, s=6, c=COLOURS[k], label=NAMES[k])
        axes[1].scatter(*Z[y == k].T, s=6, c=COLOURS[k])
    axes[0].set_aspect("equal")
    axes[0].set_title(f"Original data: A is {s0:.0f}x wider than B;\nC is {g0:.0f}x further from B than A is")
    axes[0].legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3)
    axes[1].set_aspect("equal")
    axes[1].set_title(f"t-SNE (perplexity 30): A is {s1:.1f}x wider than B;\ngap ratio {g1:.1f}")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(FIG / "tsne_sizes.png", dpi=130)
    plt.close(fig)

    # Pure noise: 300 points from a 50-dimensional standard Gaussian. No clusters exist.
    rng = np.random.default_rng(1)
    N = rng.normal(size=(300, 50))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, perp in zip(axes, (2.0, 30.0, 100.0)):
        Zn = fit(N, perp)
        ax.scatter(*Zn.T, s=6, c="#64748B")
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"perplexity {perp:g}")
    fig.suptitle("300 points of 50-dimensional Gaussian noise (there are no clusters)")
    fig.tight_layout()
    fig.savefig(FIG / "tsne_noise.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    test_kl_divergence(kl_divergence)
    test_conditional_neighbour_probs(conditional_neighbour_probs)
    part1_figures()
    part3_figures()
    print(f"\nFigures written to {FIG}")
