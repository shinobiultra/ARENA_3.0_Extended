"""PROTOTYPE (throwaway, ticket #235): toy numbers for the PRH chapter outline.

CPU only, numpy/scipy/matplotlib, random data, no models, no downloads. About 2 minutes on one core.
Every number quoted in chapter19_outline.md comes from run_output.txt, written by this script.

    /path/to/.venv/bin/python -I prh_toy_probe.py > run_output.txt
"""

from __future__ import annotations

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform

RNG = np.random.default_rng(0)


# ---------------------------------------------------------------- metrics
def center_gram(K):
    n = K.shape[0]
    H = np.eye(n) - 1.0 / n
    return H @ K @ H


def cka_biased(X, Y):
    """Linear CKA with the usual (biased) HSIC estimator, as in Kornblith et al. 2019."""
    Kx, Ky = center_gram(X @ X.T), center_gram(Y @ Y.T)
    return float((Kx * Ky).sum() / np.sqrt((Kx * Kx).sum() * (Ky * Ky).sum()))


def _hsic_unbiased(K, L):
    """Song et al. 2012 unbiased HSIC (zero diagonals)."""
    n = K.shape[0]
    K, L = K.copy(), L.copy()
    np.fill_diagonal(K, 0)
    np.fill_diagonal(L, 0)
    t1 = (K * L).sum()
    t2 = K.sum() * L.sum() / ((n - 1) * (n - 2))
    t3 = 2 * (K.sum(0) @ L.sum(0)) / (n - 2)
    return (t1 + t2 - t3) / (n * (n - 3))


def cka_debiased(X, Y):
    K, L = X @ X.T, Y @ Y.T
    return float(_hsic_unbiased(K, L) / np.sqrt(_hsic_unbiased(K, K) * _hsic_unbiased(L, L)))


def knn_adj(X, k):
    """Boolean n x n matrix: A[i, j] = j is one of i's k nearest neighbours (cosine), self excluded."""
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    S = Xn @ Xn.T
    np.fill_diagonal(S, -np.inf)
    idx = np.argpartition(-S, k, axis=1)[:, :k]
    A = np.zeros_like(S, dtype=bool)
    np.put_along_axis(A, idx, True, axis=1)
    return A


def mutual_knn(X, Y, k=10):
    """PRH's m_NN: mean over items of |N_k^X(i) & N_k^Y(i)| / k."""
    return float((knn_adj(X, k) & knn_adj(Y, k)).sum() / (X.shape[0] * k))


def mst_edges(X):
    T = minimum_spanning_tree(squareform(pdist(X))).tocoo()
    return {tuple(sorted(e)) for e in zip(T.row.tolist(), T.col.tolist())}


def mst_overlap(X, Y):
    a, b = mst_edges(X), mst_edges(Y)
    return len(a & b) / len(a)


def procrustes_fit(X, Y):
    U, _, Vt = np.linalg.svd(X.T @ Y)
    return U @ Vt


def retrieval_at_1(X, Y):
    """Fraction of items whose nearest Y (cosine) is its own pair. Chance = 1/n."""
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    Yn = Y / np.linalg.norm(Y, axis=1, keepdims=True)
    return float((np.argmax(Xn @ Yn.T, axis=1) == np.arange(len(X))).mean())


def say(*a):
    print(*a, flush=True)


# ------------------------------------------------- 1. noise that "converges"
say("== 1. Two independent Gaussian noise matrices, n = 1024 items (19.1 hook)")
say(f"{'width d':>8} {'CKA (biased)':>13} {'d/(d+n)':>8} {'CKA (debiased)':>15} {'mutual kNN k=10':>16}")
n = 1024
hook_rows = []
for d in [16, 64, 256, 1024, 4096, 16384]:
    X, Y = RNG.standard_normal((n, d)), RNG.standard_normal((n, d))
    row = (d, cka_biased(X, Y), d / (d + n), cka_debiased(X, Y), mutual_knn(X, Y))
    hook_rows.append(row)
    say(f"{row[0]:>8} {row[1]:>13.3f} {row[2]:>8.3f} {row[3]:>15.4f} {row[4]:>16.4f}")
say(f"chance level for mutual kNN = k/(n-1) = {10 / (n - 1):.4f}")

# ------------------------------------------------- 2. rotation: same space?
say("\n== 2. A rotated copy (19.1 part 1): same relations, different coordinates")
centres = RNG.standard_normal((10, 64)) * 3
labels = RNG.integers(0, 10, n)
X = centres[labels] + RNG.standard_normal((n, 64))
R, _ = np.linalg.qr(RNG.standard_normal((64, 64)))
Y = X @ R
tr, te = np.arange(512), np.arange(512, n)
Rhat = procrustes_fit(X[tr], Y[tr])
say(f"CKA {cka_biased(X, Y):.3f} | mutual kNN {mutual_knn(X, Y):.3f} | "
    f"retrieval@1 without a map {retrieval_at_1(X[te], Y[te]):.3f} (chance {1 / len(te):.4f}) | "
    f"after Procrustes fit on 512 train pairs {retrieval_at_1(X[te] @ Rhat, Y[te]):.3f}")
Yshuf = Y[RNG.permutation(n)]
say(f"control, shuffled rows: CKA {cka_biased(X, Yshuf):.3f}, mutual kNN {mutual_knn(X, Yshuf):.3f}")

# --------------------------------------- 3. local vs global, structure vs metric
say("\n== 3. Local/global x structure/metric toy (19.1 part 3), 8 clusters x 128 points, 32-d")
k8 = 8
base_c = RNG.standard_normal((k8, 32)) * 4
lab = np.repeat(np.arange(k8), 128)
offsets = RNG.standard_normal((len(lab), 32))
X = base_c[lab] + offsets
# (a) keep each cluster's internal arrangement, rescale clusters and move their centres
scale = RNG.choice([0.2, 1.0, 3.0], k8)[lab][:, None]
Ya = (RNG.standard_normal((k8, 32)) * 4)[lab] + offsets * scale
# (b) keep the cluster layout, scramble who is next to whom inside clusters
Yb = base_c[lab] + RNG.standard_normal((len(lab), 32))
say(f"{'variant':<38} {'CKA':>6} {'mutual kNN':>11} {'MST edge overlap':>17}")
for name, Yv in [("(a) same neighbours, new layout", Ya), ("(b) same layout, new neighbours", Yb),
                 ("rotated copy (reference)", X @ np.linalg.qr(RNG.standard_normal((32, 32)))[0])]:
    say(f"{name:<38} {cka_biased(X, Yv):>6.3f} {mutual_knn(X, Yv):>11.3f} {mst_overlap(X, Yv):>17.3f}")

# ---------------------------------------- 4. layer search and calibration
say("\n== 4. Max over layer pairs, then a permutation null (19.1 part 2), n = 512, width 256, k = 10")


def layer_stack(shared, L, width, noise=1.0):
    """Residual-stream-like stack: each layer adds fresh private noise to the previous one."""
    h = shared + RNG.standard_normal(shared.shape) * noise
    out = [h]
    for _ in range(L - 1):
        h = h + RNG.standard_normal(h.shape) * 0.5
        out.append(h.copy())
    return out


def max_over_pairs(As, Bs, perm=None):
    best = 0.0
    for A in As:
        for B in Bs:
            Bp = B if perm is None else B[np.ix_(perm, perm)]
            best = max(best, (A & Bp).sum())
    return best / (As[0].shape[0] * 10)


n2, w, n_perm = 512, 256, 200
latent = RNG.standard_normal((n2, 8))
proj_a, proj_b = RNG.standard_normal((8, w)), RNG.standard_normal((8, w))
say(f"{'signal':>7} {'layers':>7} {'raw max m_NN':>13} {'null-of-max 95%':>16} {'p (right null)':>15} "
    f"{'1-pair null 95%':>16} {'p (wrong null)':>15}")
layer_rows = []
for alpha in [0.0, 0.35]:
    for L in [1, 4, 12]:
        As = [knn_adj(h, 10) for h in layer_stack(alpha * latent @ proj_a, L, w)]
        Bs = [knn_adj(h, 10) for h in layer_stack(alpha * latent @ proj_b, L, w)]
        raw = max_over_pairs(As, Bs)
        null_max = np.array([max_over_pairs(As, Bs, RNG.permutation(n2)) for _ in range(n_perm)])
        a0, b0 = As[-1], Bs[-1]
        null_one = np.array([(a0 & b0[np.ix_(p, p)]).sum() / (n2 * 10)
                             for p in (RNG.permutation(n2) for _ in range(n_perm))])
        p_right = (1 + (null_max >= raw).sum()) / (1 + n_perm)
        p_wrong = (1 + (null_one >= raw).sum()) / (1 + n_perm)
        layer_rows.append((alpha, L, raw, np.quantile(null_max, 0.95), p_right))
        say(f"{alpha:>7} {L:>7} {raw:>13.4f} {np.quantile(null_max, 0.95):>16.4f} {p_right:>15.3f} "
            f"{np.quantile(null_one, 0.95):>16.4f} {p_wrong:>15.3f}")

# ----------------------------------- 5. finite world: two observers, one kernel
say("\n== 5. Finite world (19.3): spectral embedding of a shared positive-pair kernel")


def spectral_embed(P, m):
    p = P.sum(1)
    Dm = np.diag(p ** -0.5)
    vals, vecs = np.linalg.eigh(Dm @ P @ Dm)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    return Dm @ vecs[:, 1:m + 1], vals, p  # drop the constant mode


def weighted_procrustes_residual(F, G, p):
    W = np.sqrt(p)[:, None]
    Rw = procrustes_fit(W * G, W * F)
    return float(np.linalg.norm(W * (G @ Rw - F)) / np.linalg.norm(W * F))


def world(tilt, S=24):
    """Four groups of 6 states, A-B-C-D-A in a loop. Positives: any two states in one group (weight 1),
    or in neighbouring groups (A-B and C-D weight 0.05*(1+tilt), B-C and D-A weight 0.05*(1-tilt)).
    tilt = 0: the splits AB|CD and BC|DA are rewarded equally, so the top eigenvalue is tied.
    tilt > 0: AB|CD wins and an eigengap opens."""
    g = np.repeat(np.arange(4), S // 4)
    a, b = 0.05 * (1 + tilt), 0.05 * (1 - tilt)
    between = {(0, 1): a, (2, 3): a, (1, 2): b, (3, 0): b}
    W = (g[:, None] == g[None, :]).astype(float)
    np.fill_diagonal(W, 0)
    for (u, v), wt in between.items():
        W[np.ix_(g == u, g == v)] = W[np.ix_(g == v, g == u)] = wt
    return W / W.sum()


def marginal_preserving_noise(S):
    """Symmetric perturbation with zero row sums: changes which pairs are positive, keeps every marginal."""
    E = RNG.standard_normal((S, S))
    E = (E + E.T) / 2
    for _ in range(50):  # Sinkhorn-style zero-row-sum projection that keeps symmetry
        E -= E.mean(1, keepdims=True)
        E = (E + E.T) / 2
    return E / np.abs(E).max()


S, m = 24, 1
E = marginal_preserving_noise(S)
say("1-D embedding (top non-constant eigenvector); residual 0 = the observers agree up to sign")
say(f"{'world':<22} {'eigengap':>9} " + " ".join(f"{'eps=' + str(e):>9}" for e in [0.0, 1e-4, 1e-3, 3e-3]))
gap_rows = []
for name, g in [("tilt 0.5 (wide gap)", 0.5), ("tilt 0.05 (narrow gap)", 0.05), ("tilt 0 (tied)", 0.0)]:
    P = world(g, S)
    F, vals, p = spectral_embed(P, m)
    res = []
    for eps in [0.0, 1e-4, 1e-3, 3e-3]:
        Pp = np.clip(P + eps * E, 0, None)
        Pp = (Pp + Pp.T) / 2
        Pp /= Pp.sum()
        G, _, pp = spectral_embed(Pp, m)
        res.append(weighted_procrustes_residual(F, G, p))
    gap_rows.append((name, vals[m] - vals[m + 1], res))
    say(f"{name:<22} {vals[m] - vals[m + 1]:>9.4f} " + " ".join(f"{r:>9.3f}" for r in res))
P = world(0.5, S)
F, _, p = spectral_embed(P, m)
for other, where in [(1, "same group A"), (12, "group C, across the kept AB|CD split")]:
    merge = np.arange(S)
    merge[other] = 0  # observer B renders states 0 and `other` identically (non-injective rendering)
    idx = np.unique(merge, return_inverse=True)[1]
    Pm = np.zeros((S - 1, S - 1))
    np.add.at(Pm, (idx[:, None], idx[None, :]), P)
    Gm, _, _ = spectral_embed(Pm, m)
    say(f"lossy observer B, tilt 0.5, state 0 merged with state {other} ({where}): "
        f"residual {weighted_procrustes_residual(F, Gm[idx], p):.3f}")

# ----------------------------------------------------------- figure
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    ds = [r[0] for r in hook_rows]
    ax[0].semilogx(ds, [r[1] for r in hook_rows], "o-", label="CKA (biased)")
    ax[0].semilogx(ds, [r[3] for r in hook_rows], "s-", label="CKA (debiased)")
    ax[0].semilogx(ds, [r[4] for r in hook_rows], "^-", label="mutual kNN, k=10")
    ax[0].set(title="Two unrelated noise matrices, n=1024", xlabel="width d", ylabel="similarity")
    ax[0].legend()
    for alpha, mk in [(0.0, "o"), (0.35, "s")]:
        rows = [r for r in layer_rows if r[0] == alpha]
        ax[1].plot([r[1] for r in rows], [r[2] for r in rows], mk + "-", label=f"raw max, signal={alpha}")
        ax[1].plot([r[1] for r in rows], [r[3] for r in rows], mk + "--", alpha=0.6,
                   label=f"null-of-max 95%, signal={alpha}")
    ax[1].set(title="Searching more layer pairs", xlabel="layers per model", ylabel="mutual kNN")
    ax[1].legend(fontsize=7)
    eps = [0.0, 1e-4, 1e-3, 3e-3]
    for name, gap, res in gap_rows:
        ax[2].plot(eps, res, "o-", label=f"{name} ({gap:.3f})")
    ax[2].set(title="Two observers, perturbed positives", xlabel="perturbation eps",
              ylabel="Procrustes residual")
    ax[2].legend()
    fig.tight_layout()
    fig.savefig("prh_toy_probe.png", dpi=120)
    say("\nfigure: prh_toy_probe.png")
except Exception as exc:  # prototype: a missing plot must not hide the numbers
    say(f"\n(figure skipped: {exc})")
