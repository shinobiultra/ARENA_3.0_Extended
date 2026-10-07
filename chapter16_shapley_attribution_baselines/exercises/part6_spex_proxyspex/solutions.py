# %%


import itertools
import math
import sys
import time
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as t
from IPython.display import display
from jaxtyping import Float, Int
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.tree import DecisionTreeRegressor
from torch import Tensor
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer

device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")

# Make sure exercises are in the path
chapter = "chapter16_shapley_attribution_baselines"
section = "part6_spex_proxyspex"
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part6_spex_proxyspex.tests as tests
import part6_spex_proxyspex.utils as utils
from part6_spex_proxyspex.utils import GaloisField, bch_field_degree, galois_field

# A spectrum maps a support (sorted tuple of feature indices; () is the constant) to a coefficient
Spectrum = dict[tuple[int, ...], float]

MAIN = __name__ == "__main__"

# %%

def all_masks(n: int) -> Int[np.ndarray, "two_pow_n n"]:
    """All 2^n masks in index order. Feature 0 is the most significant bit, so row i is the binary expansion of i."""
    return (np.arange(2**n)[:, None] >> np.arange(n - 1, -1, -1)) & 1


def mask_index(masks: Int[np.ndarray, "batch n"]) -> Int[np.ndarray, " batch"]:
    """The inverse of all_masks: the integer whose binary expansion is the mask."""
    masks = np.asarray(masks, dtype=np.int64)
    return masks @ (1 << np.arange(masks.shape[-1] - 1, -1, -1))


def supports_to_matrix(supports: list[tuple[int, ...]], n: int) -> Int[np.ndarray, "k n"]:
    """Stacks the 0/1 indicator vectors of a list of supports into a (k, n) matrix."""
    out = np.zeros((len(supports), n), dtype=np.int64)
    for row, support in enumerate(supports):
        out[row, list(support)] = 1
    return out


def index_to_support(i: int, n: int) -> tuple[int, ...]:
    """The support whose indicator vector is the binary expansion of i (feature 0 = most significant bit)."""
    return tuple(j for j in range(n) if (i >> (n - 1 - j)) & 1)


def array_to_spectrum(coefficients: Float[np.ndarray, " two_pow_n"], tol: float = 0.0) -> Spectrum:
    """Converts a full array of 2^n coefficients into a dict, keeping entries with |coefficient| > tol."""
    n = int(np.log2(len(coefficients)))
    return {index_to_support(i, n): float(coefficients[i]) for i in np.flatnonzero(np.abs(coefficients) > tol)}


if MAIN:
    print(all_masks(3))
    print(mask_index(np.array([[0, 0, 1], [1, 0, 0]])), index_to_support(5, 3))

# %%

def predict_spectrum(spectrum: Spectrum, masks: Int[np.ndarray, "batch n"]) -> Float[np.ndarray, " batch"]:
    """Evaluates sum_S F(S) * chi_S(x) for every mask x in the batch."""
    masks = np.asarray(masks)
    if len(spectrum) == 0:
        return np.zeros(len(masks))
    K = supports_to_matrix(list(spectrum.keys()), masks.shape[-1])
    chi = 1.0 - 2.0 * ((masks @ K.T) % 2)  # (batch, k): +1 for an even number of kept features in S, else -1
    return chi @ np.array(list(spectrum.values()), dtype=np.float64)


if MAIN:
    tests.test_predict_spectrum(predict_spectrum)

# %%

def fwht(values: Float[np.ndarray, "... N"]) -> Float[np.ndarray, "... N"]:
    """
    Normalised fast Walsh-Hadamard transform along the last axis: returns F with
    F[..., S] = (1/N) * sum_x values[..., x] * (-1)^popcount(S & x), where N = 2^n.
    """
    a = np.array(values, dtype=np.float64)  # copy
    N = a.shape[-1]
    batch = a.shape[:-1]
    h = 1
    while h < N:
        a = a.reshape(*batch, N // (2 * h), 2, h)
        a = np.stack([a[..., 0, :] + a[..., 1, :], a[..., 0, :] - a[..., 1, :]], axis=-2)
        h *= 2
    return a.reshape(*batch, N) / N


if MAIN:
    tests.test_fwht(fwht)

# %%

def mobius_transform(values: Float[np.ndarray, "... N"]) -> Float[np.ndarray, "... N"]:
    """
    Fast Möbius transform along the last axis: returns m with m[..., S] = sum_{T subset of S} (-1)^(|S|-|T|) values[..., T].
    """
    a = np.array(values, dtype=np.float64)
    N = a.shape[-1]
    batch = a.shape[:-1]
    h = 1
    while h < N:
        a = a.reshape(*batch, N // (2 * h), 2, h)
        a[..., 1, :] -= a[..., 0, :]
        h *= 2
    return a.reshape(*batch, N)


if MAIN:
    tests.test_mobius_transform(mobius_transform)

# %%

if MAIN:
    n_toy = 3
    toy_masks = all_masks(n_toy)
    toy_games = {
        "additive  x0 + x1 + x2": toy_masks.sum(1).astype(float),
        "AND  x0·x1·x2": toy_masks.prod(1).astype(float),
        "parity  (-1)^(x0+x1+x2)": (-1.0) ** toy_masks.sum(1),
    }
    toy_spectra = {
        name: (array_to_spectrum(fwht(values), tol=1e-12), array_to_spectrum(mobius_transform(values), tol=1e-12))
        for name, values in toy_games.items()
    }
    utils.plot_toy_spectra(toy_spectra, n_toy)

# %%

def fourier_to_mobius(spectrum: Spectrum) -> Spectrum:
    """Converts a sparse Fourier spectrum to the Möbius (AND-basis) coefficients of the same function."""
    sums = {}
    for support, coef in spectrum.items():
        for r in range(len(support) + 1):
            for subset in itertools.combinations(support, r):
                sums[subset] = sums.get(subset, 0.0) + coef
    return {subset: (-2.0) ** len(subset) * total for subset, total in sums.items()}


if MAIN:
    tests.test_fourier_to_mobius(fourier_to_mobius)

# %%

def shapley_interactions(mobius: Spectrum, max_order: int = 2) -> Spectrum:
    """
    Shapley interaction indices I(T) = sum_{S superset of T} m(S) / (|S| - |T| + 1), for all 1 <= |T| <= max_order.
    The Shapley value of feature i is the entry for (i,).
    """
    out = {}
    for support, coef in mobius.items():
        for r in range(1, min(len(support), max_order) + 1):
            for subset in itertools.combinations(support, r):
                out[subset] = out.get(subset, 0.0) + coef / (len(support) - r + 1)
    return out


if MAIN:
    tests.test_shapley_interactions(shapley_interactions)

# %%

if MAIN:
    rng = np.random.default_rng(0)
    check_values = rng.normal(size=2**6)
    ours = shapley_interactions(array_to_spectrum(mobius_transform(check_values)), max_order=2)
    try:
        import shapiq
    
        exact = shapiq.ExactComputer(lambda X: check_values[mask_index(np.asarray(X, dtype=int))], n_players=6)
        sii_ref = exact("SII", order=2)
        max_diff = max(abs(ours[T] - sii_ref[T]) for T in ours)
        print(f"shapiq {shapiq.__version__}: max |our SII - shapiq SII| over {len(ours)} sets = {max_diff:.2e}")
    except ImportError:
        print("shapiq not installed: skipping the cross-check")

# %%

if MAIN:
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    model.requires_grad_(False)
    
    IOI_TEXT = "Yesterday, when Mary and John went to the busy store, John happily gave a cold drink"
    words, ioi_tokens, ioi_token_to_word = utils.tokenize_words(model, IOI_TEXT, " to")
    n_words = len(words)
    MARY, JOHN = model.to_single_token(" Mary"), model.to_single_token(" John")
    
    print(f"{n_words} words, {ioi_tokens.shape[1]} tokens")
    print([model.to_string(tok) for tok in ioi_tokens[0]])
    print(ioi_token_to_word.tolist())

# %%

@t.inference_mode()
def masked_final_logits(
    model: HookedTransformer,
    tokens: Int[Tensor, "1 seq"],
    token_to_word: Int[Tensor, " seq"],
    masks: Int[Tensor, "batch n"],
) -> Float[Tensor, "batch d_vocab"]:
    """Final-position logits when each masked word's tokens are removed (via the attention mask)."""
    masks = masks.to(tokens.device).long()
    attention_mask = t.ones(masks.shape[0], tokens.shape[1], dtype=t.long, device=tokens.device)
    maskable = token_to_word >= 0
    attention_mask[:, maskable] = masks[:, token_to_word[maskable]]
    resid = model(tokens.expand(masks.shape[0], -1), attention_mask=attention_mask, stop_at_layer=model.cfg.n_layers)
    return model.unembed(model.ln_final(resid[:, -1]))


if MAIN:
    tests.test_masked_final_logits(masked_final_logits, model)

# %%

def make_value_function(
    model: HookedTransformer,
    tokens: Int[Tensor, "1 seq"],
    token_to_word: Int[Tensor, " seq"],
    readout: Callable[[Tensor], Tensor],
    batch_size: int = 512,
) -> Callable[[np.ndarray], np.ndarray]:
    """Returns f(masks) -> readout(final logits) as a float64 numpy array, evaluated in batches."""

    def f(masks: np.ndarray, progress: bool = False) -> np.ndarray:
        masks = t.as_tensor(np.asarray(masks), device=tokens.device)
        chunks = range(0, len(masks), batch_size)
        out = [
            readout(masked_final_logits(model, tokens, token_to_word, masks[i : i + batch_size]).float()).cpu()
            for i in (tqdm(chunks) if progress else chunks)
        ]
        return t.cat(out).numpy().astype(np.float64)

    return f


def ioi_readouts(logits: Tensor) -> Tensor:
    """(batch, d_vocab) -> (batch, 2): [logit(Mary) - logit(John), p(Mary)]."""
    return t.stack([logits[:, MARY] - logits[:, JOHN], logits.softmax(-1)[:, MARY]], dim=-1)

# %%

if MAIN:
    ioi_value_both = make_value_function(model, ioi_tokens, ioi_token_to_word, ioi_readouts)
    X_all = all_masks(n_words)
    
    start = time.time()
    table = ioi_value_both(X_all, progress=True)
    print(f"Evaluated {len(X_all):,} masks in {time.time() - start:.0f}s")
    
    exact_values = {"logit diff": table[:, 0], "p(Mary)": table[:, 1]}
    
    # A few masks by hand: which words are dropped, and what happens to each read-out
    drops = [[], [2], [4], [10], [4, 10], [12]]
    rows = []
    for drop in drops:
        mask = np.ones(n_words, dtype=int)
        mask[drop] = 0
        i = mask_index(mask[None])[0]
        rows.append({"dropped words": ", ".join(f"{words[d]}[{d}]" for d in drop) or "(none)",
                     "logit diff": exact_values["logit diff"][i], "p(Mary)": exact_values["p(Mary)"][i]})
    print(pd.DataFrame(rows).set_index("dropped words").round(3).to_string())

# %%

def degree_energy(F: Float[np.ndarray, " two_pow_n"]) -> Float[np.ndarray, " n"]:
    """Fraction of non-constant energy sum_{|S|=d} F(S)^2 / sum_{S != empty} F(S)^2, for d = 1..n."""
    n = int(np.log2(len(F)))
    degree = np.array([bin(i).count("1") for i in range(len(F))])
    energy = F**2
    total = energy[1:].sum()
    return np.array([energy[degree == d].sum() / total for d in range(1, n + 1)])


def topk_energy(F: Float[np.ndarray, " two_pow_n"]) -> Float[np.ndarray, " two_pow_n_minus_1"]:
    """Entry k-1: fraction of non-constant energy in the k largest non-constant coefficients."""
    energy = np.sort(F[1:] ** 2)[::-1]
    return np.cumsum(energy) / energy.sum()


if MAIN:
    tests.test_spectrum_summaries(degree_energy, topk_energy)

# %%

if MAIN:
    exact_F = {name: fwht(values) for name, values in exact_values.items()}
    shuffled_F = fwht(np.random.default_rng(0).permutation(exact_values["p(Mary)"]))
    
    utils.plot_sparsity(
        {"logit diff": exact_F["logit diff"], "p(Mary)": exact_F["p(Mary)"], "p(Mary), shuffled across masks (control)": shuffled_F},
        degree_energy,
        topk_energy,
    )
    
    summary = {}
    for name, F in [*exact_F.items(), ("shuffled control", shuffled_F)]:
        de, tk = degree_energy(F), topk_energy(F)
        summary[name] = {"degree 1": de[0], "degree 2": de[1], "degree 3": de[2], "degree <= 3": de[:3].sum(),
                         "top 10": tk[9], "top 50": tk[49], "top 200": tk[199], "top 1000": tk[999]}
    print(pd.DataFrame(summary).T.round(3).to_string())

# %%

if MAIN:
    for name, F in exact_F.items():
        spectrum = array_to_spectrum(F)
        top = sorted(((S, v) for S, v in spectrum.items() if S), key=lambda kv: -abs(kv[1]))[:10]
        print(f"{name}: largest Fourier coefficients")
        for S, v in top:
            print(f"  {v:+.4f}  {utils.support_label(S, words)}")

# %%

if MAIN:
    for name, values in exact_values.items():
        m = mobius_transform(values)
        F = exact_F[name]
        largest = np.argsort(-np.abs(m))[:6]
        print(f"{name}: values in [{values.min():.2f}, {values.max():.2f}],  max |F(S)| = {np.abs(F[1:]).max():.3f},  "
              f"max |m(S)| = {np.abs(m).max():.2f}")
        print("   six largest |m(S)|, as (number of words in S, m(S)): "
              + ", ".join(f"({bin(i).count('1')}, {m[i]:+.2f})" for i in largest))

# %%

def spectrum_oracle(spectrum: Spectrum, chunk: int = 4096) -> Callable[[np.ndarray], np.ndarray]:
    """A value function whose Fourier spectrum is exactly `spectrum`."""

    def f(masks: np.ndarray) -> np.ndarray:
        masks = np.asarray(masks)
        return np.concatenate([predict_spectrum(spectrum, masks[i : i + chunk]) for i in range(0, len(masks), chunk)])

    return f

# %%

def bin_index(M: Int[np.ndarray, "n b"], support: tuple[int, ...]) -> int:
    """The bin j = M^T k (as an integer, first column of M = most significant bit) that a support hashes to."""
    b = M.shape[1]
    bits = M[list(support)].sum(0) % 2 if len(support) else np.zeros(b, dtype=np.int64)
    return int(bits @ (1 << np.arange(b - 1, -1, -1)))

# %%

def subsample_masks(M: Int[np.ndarray, "n b"], delay: Int[np.ndarray, " n"]) -> Int[np.ndarray, "two_pow_b n"]:
    """The 2^b masks M @ ell + delay (mod 2), with ell running over all_masks(b) in order."""
    small = all_masks(M.shape[1])  # (2^b, b)
    return (small @ M.T + delay) % 2


def aliased_bins(
    f: Callable[[np.ndarray], np.ndarray], M: Int[np.ndarray, "n b"], delays: Int[np.ndarray, "P n"]
) -> tuple[Float[np.ndarray, "P two_pow_b"], Int[np.ndarray, "P two_pow_b n"]]:
    """Queries f on the subsampled masks for every delay, and Walsh-transforms each delay's 2^b values."""
    P, n = delays.shape
    masks = np.stack([subsample_masks(M, d) for d in delays])  # (P, 2^b, n)
    values = f(masks.reshape(-1, n)).reshape(P, -1)
    return fwht(values), masks


if MAIN:
    tests.test_subsample_masks(subsample_masks)
    tests.test_aliased_bins(aliased_bins)

# %%

if MAIN:
    planted = utils.make_sparse_game(n=100, s=40, degrees=[1, 2, 3], seed=40)
    f_planted = spectrum_oracle(planted)
    M_example = np.random.default_rng(0).integers(0, 2, size=(100, 5))
    U_example, _ = aliased_bins(f_planted, M_example, np.zeros((1, 100), dtype=np.int64))
    occupancy = [[] for _ in range(32)]
    for support in planted:
        occupancy[bin_index(M_example, support)].append(support)
    counts = np.bincount([min(len(o), 2) for o in occupancy], minlength=3)
    print(f"{counts[0]} empty bins, {counts[1]} singletons, {counts[2]} collisions (41 coefficients incl. the constant, 32 bins)")
    utils.plot_bins(U_example[0], occupancy, "One hashing of a planted 40-sparse game (n = 100) into 32 bins")

# %%

def read_support_identity(u: Float[np.ndarray, " n_plus_1"]) -> tuple[int, ...]:
    """Decodes the support of a singleton bin measured with the zero delay followed by the n unit-vector delays."""
    return tuple(int(i) for i in np.flatnonzero(u[1:] / u[0] < 0))


if MAIN:
    tests.test_read_support_identity(read_support_identity)

# %%

if MAIN:
    gf = GaloisField(4)
    print("alpha^0..alpha^15 in GF(16):", gf.exp[:16].tolist())
    print("alpha^3 * alpha^14 =", gf.mul(gf.exp[3], gf.exp[14]), "= alpha^(17 mod 15) =", gf.exp[2])

# %%

def bch_delay_matrix(n: int, t: int) -> Int[np.ndarray, "m_times_t n"]:
    """Parity-check matrix of the (shortened) binary BCH code correcting t errors, used as SPEX's delays."""
    m = bch_field_degree(n)
    gf = galois_field(m)  # cached GaloisField(m)
    blocks = []
    for r in range(t):
        elements = gf.exp[((2 * r + 1) * np.arange(n)) % gf.N]  # alpha^((2r+1) i) for each feature i
        blocks.append((elements[None, :] >> np.arange(m)[:, None]) & 1)  # (m, n): bit q in row q
    return np.concatenate(blocks).astype(np.int64)


if MAIN:
    tests.test_bch_delay_matrix(bch_delay_matrix)

# %%

def berlekamp_massey(syndromes: list[int], gf: GaloisField) -> list[int]:
    """
    Shortest linear recurrence (connection polynomial) generating the sequence S_1, ..., S_2t over GF(2^m).
    Returns the coefficients [1, Lambda_1, ..., Lambda_L] of the error-locator polynomial, in increasing powers of z.
    """
    C, B = [1], [1]  # current and previous connection polynomials
    L, shift, b = 0, 1, 1  # current recurrence length, steps since B was saved, discrepancy when B was saved
    for r, s in enumerate(syndromes):
        # discrepancy: how far the current recurrence is from predicting S_{r+1}
        d = s
        for i in range(1, L + 1):
            d ^= gf.mul(C[i], syndromes[r - i])
        if d == 0:
            shift += 1
            continue
        # C(z) <- C(z) - (d / b) z^shift B(z)
        coef = gf.mul(d, gf.inv(b))
        new_C = C + [0] * max(0, len(B) + shift - len(C))
        for i, Bi in enumerate(B):
            new_C[i + shift] ^= gf.mul(coef, Bi)
        if 2 * L <= r:  # the recurrence has to get longer
            L, B, b, shift = r + 1 - L, C, d, 1
        else:
            shift += 1
        C = new_C
    while len(C) > 1 and C[-1] == 0:
        C.pop()
    return C

# %%

def bch_decode(bits: Int[np.ndarray, " m_times_t"], n: int, t: int) -> tuple[int, ...] | None:
    """Recovers the support k (|k| <= t) from the syndrome bits D @ k mod 2, or returns None if decoding fails."""
    m = bch_field_degree(n)
    gf = galois_field(m)  # cached GaloisField(m)
    blocks = np.asarray(bits).reshape(t, m)
    S = [0] * (2 * t)  # S[j - 1] holds S_j
    for r in range(t):
        S[2 * r] = int(blocks[r] @ (1 << np.arange(m)))  # S_{2r+1}
    for j in range(2, 2 * t + 1, 2):
        S[j - 1] = gf.mul(S[j // 2 - 1], S[j // 2 - 1])  # S_j = S_{j/2}^2
    if not any(S):
        return ()
    locator = berlekamp_massey(S, gf)
    L = len(locator) - 1
    if L > t:
        return None
    positions = np.arange(n)
    value = np.zeros(n, dtype=np.int64)
    for k, coef in enumerate(locator):
        if coef != 0:
            value ^= gf.exp[(gf.log[coef] - k * positions) % gf.N]
    roots = tuple(int(i) for i in np.flatnonzero(value == 0))
    return roots if len(roots) == L else None


if MAIN:
    tests.test_bch_decode(bch_decode, bch_delay_matrix)

# %%

if MAIN:
    ns = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    delay_costs = {"identity delays (n + 1)": [n + 1 for n in ns]}
    for t_ in [2, 3, 5]:
        delay_costs[f"BCH delays, t = {t_}"] = [bch_delay_matrix(n, t_).shape[0] + 1 for n in ns]
    utils.plot_delay_cost(ns, delay_costs)
    print({name: costs[ns.index(1024)] for name, costs in delay_costs.items()})

# %%

def detect_singleton(
    u: Float[np.ndarray, " P"],
    M: Int[np.ndarray, "n b"],
    delays: Int[np.ndarray, "P n"],
    j: int,
    t: int,
    gamma: float = 0.9,
    chase_depth: int = 0,
) -> tuple[tuple[int, ...], float] | None:
    """Tests whether bin j (values u across delays) holds a single coefficient; returns (support, amplitude) or None."""
    n = delays.shape[1]
    energy = (u**2).sum()
    if energy < 1e-20 or abs(u[0]) < 1e-9 * np.sqrt(energy):
        return None
    ratios = u[1:] / u[0]
    bits = (ratios < 0).astype(np.int64)
    flips = [()]
    if chase_depth > 0:
        weak = np.argsort(np.abs(ratios))[:chase_depth]
        flips += [(w,) for w in weak] + list(itertools.combinations(weak, 2))
    best = None
    for flip in flips:
        trial = bits.copy()
        trial[list(flip)] ^= 1
        support = bch_decode(trial, n, t)
        if support is None or bin_index(M, support) != j:
            continue
        signature = 1.0 - 2.0 * (delays[:, list(support)].sum(1) % 2)
        amplitude = signature @ u / len(u)
        residual = ((u - amplitude * signature) ** 2).sum() / energy
        if best is None or residual < best[0]:
            best = (residual, support, amplitude)
    if best is None or best[0] > gamma:
        return None
    return best[1], float(best[2])


if MAIN:
    tests.test_detect_singleton(detect_singleton, bch_delay_matrix)

# %%

def spex(
    f: Callable[[np.ndarray], np.ndarray],
    n: int,
    b: int,
    t: int,
    n_groups: int = 3,
    gamma: float = 0.9,
    chase_depth: int = 0,
    max_sweeps: int = 20,
    seed: int = 0,
    trace: list | None = None,
) -> tuple[Spectrum, Int[np.ndarray, "Q n"]]:
    """SPEX: recovers a sparse, degree-<=t Fourier spectrum of f from n_groups * (m t + 1) * 2^b queries."""
    rng = np.random.default_rng(seed)
    delays = np.concatenate([np.zeros((1, n), dtype=np.int64), bch_delay_matrix(n, t)])
    Ms = [rng.integers(0, 2, size=(n, b)) for _ in range(n_groups)]
    U, queried = [], []
    for M in Ms:
        U_c, masks = aliased_bins(f, M, delays)
        U.append(U_c)
        queried.append(masks.reshape(-1, n))
    U = np.stack(U)
    if trace is not None:
        trace.append(U.copy())

    spectrum = {}
    for _ in range(max_sweeps):
        found = 0
        for c, M in enumerate(Ms):
            for j in range(2**b):
                hit = detect_singleton(U[c, :, j], M, delays, j, t, gamma, chase_depth)
                if hit is None:
                    continue
                support, amplitude = hit
                spectrum[support] = spectrum.get(support, 0.0) + amplitude
                signature = 1.0 - 2.0 * (delays[:, list(support)].sum(1) % 2)
                for c2, M2 in enumerate(Ms):
                    U[c2, :, bin_index(M2, support)] -= amplitude * signature
                found += 1
        if trace is not None:
            trace.append(U.copy())
        if found == 0:
            break
    return spectrum, np.concatenate(queried)


if MAIN:
    tests.test_spex(spex)

# %%

def faithfulness_r2(spectrum: Spectrum, masks: Int[np.ndarray, "batch n"], values: Float[np.ndarray, " batch"]) -> float:
    """R^2 of the spectrum's predictions against the true values on the given masks."""
    values = np.asarray(values, dtype=np.float64)
    pred = predict_spectrum(spectrum, masks)
    return float(1 - ((values - pred) ** 2).sum() / ((values - values.mean()) ** 2).sum())


if MAIN:
    tests.test_faithfulness_r2(faithfulness_r2)

# %%

if MAIN:
    peel_trace = []
    recovered, queried = spex(f_planted, n=100, b=4, t=3, seed=0, trace=peel_trace)
    held_out = np.random.default_rng(123).integers(0, 2, size=(2000, 100))
    y_held_out = f_planted(held_out)
    
    found = {S for S, v in recovered.items() if abs(v) > 1e-8}
    max_error = max(abs(recovered.get(S, 0.0) - v) for S, v in planted.items())
    print(f"{len(queried)} queries ({len(np.unique(mask_index(queried)))} distinct masks), {len(peel_trace) - 1} sweeps")
    print(f"true coefficients found: {len(found & set(planted))}/{len(planted)}, spurious: {len(found - set(planted))}, "
          f"max |error| = {max_error:.1e}, held-out R² = {faithfulness_r2(recovered, held_out, y_held_out):.4f}")
    utils.plot_peeling_trace(peel_trace, "Residual bin energy during peeling (planted 40-sparse game, n = 100, b = 4, 3 hashes)")

# %%

if MAIN:
    def spex_budget_curve(spectrum: Spectrum, n: int, t: int, bs: list[int], seeds=(0, 1, 2)) -> tuple[list[int], list[float]]:
        f = spectrum_oracle(spectrum)
        test = np.random.default_rng(123).integers(0, 2, size=(2000, n))
        y = f(test)
        queries, r2s = [], []
        for b in bs:
            results = [spex(f, n, b, t, seed=seed) for seed in seeds]
            queries.append(len(results[0][1]))
            r2s.append(float(np.mean([faithfulness_r2(spec, test, y) for spec, _ in results])))
        return queries, r2s
    
    
    bs = [3, 4, 5, 6, 7, 8]
    curves = {}
    for s in [10, 40, 160]:
        curves[f"s = {s} coefficients"] = spex_budget_curve(utils.make_sparse_game(100, s, [1, 2, 3], seed=s), 100, 3, bs)
    curves["control: dense (3,000 small coefficients)"] = spex_budget_curve(
        utils.make_sparse_game(100, 3000, [1, 2, 3], seed=7, amplitude=(0.05, 0.15)), 100, 3, bs
    )
    curves["control: degree-5 game decoded with t = 3"] = spex_budget_curve(utils.make_sparse_game(100, 20, [5], seed=8), 100, 3, bs)
    curves["degree-5 game decoded with t = 5"] = spex_budget_curve(utils.make_sparse_game(100, 20, [5], seed=8), 100, 5, bs)
    
    utils.plot_recovery_curves(
        curves,
        "SPEX on planted games (n = 100, 3 hashes, mean over 3 designs)",
        controls=("control: dense (3,000 small coefficients)", "control: degree-5 game decoded with t = 3"),
    )
    recovery_table = pd.DataFrame({name: dict(zip([f"b={b}" for b in bs], r2)) for name, (_, r2) in curves.items()}).T
    print("queries at b = 3..8:  t = 3:", curves["s = 10 coefficients"][0], "  t = 5:", curves["degree-5 game decoded with t = 5"][0])
    print(recovery_table.round(3).to_string())

# %%

if MAIN:
    big_game = utils.make_sparse_game(n=1000, s=100, degrees=[1, 2, 3], seed=1000)
    start = time.time()
    big_recovered, big_queries = spex(spectrum_oracle(big_game), n=1000, b=7, t=3, seed=0)
    big_found = {S for S, v in big_recovered.items() if abs(v) > 1e-8}
    big_test = np.random.default_rng(1).integers(0, 2, size=(2000, 1000))
    print(f"n = 1000: {len(big_queries):,} queries, {time.time() - start:.1f}s")
    print(f"found {len(big_found & set(big_game))}/{len(big_game)} true coefficients, {len(big_found - set(big_game))} spurious, "
          f"held-out R² = {faithfulness_r2(big_recovered, big_test, spectrum_oracle(big_game)(big_test)):.4f}")
    print(f"candidate supports of degree <= 3: {sum(math.comb(1000, d) for d in range(4)):,}")

# %%

def tree_to_fourier(tree: DecisionTreeRegressor, n: int) -> Spectrum:
    """Exact Fourier spectrum of a regression tree fitted on 0/1 features."""
    tr = tree.tree_

    def recurse(node: int) -> Spectrum:
        if tr.children_left[node] == -1:
            return {(): float(tr.value[node].ravel()[0])}
        left, right = recurse(tr.children_left[node]), recurse(tr.children_right[node])
        i = int(tr.feature[node])
        out = {}
        for S in set(left) | set(right):
            g_left, g_right = left.get(S, 0.0), right.get(S, 0.0)
            out[S] = out.get(S, 0.0) + (g_left + g_right) / 2
            S_xor_i = tuple(sorted(set(S) ^ {i}))
            out[S_xor_i] = out.get(S_xor_i, 0.0) + (g_left - g_right) / 2
        return out

    return recurse(0)


if MAIN:
    tests.test_tree_to_fourier(tree_to_fourier)

# %%

def gbt_to_fourier(gbt: GradientBoostingRegressor, n: int) -> Spectrum:
    """Exact Fourier spectrum of a fitted scikit-learn gradient-boosted regression ensemble."""
    spectrum = {(): float(np.ravel(gbt.init_.constant_)[0])}
    for tree in gbt.estimators_[:, 0]:
        for S, v in tree_to_fourier(tree, n).items():
            spectrum[S] = spectrum.get(S, 0.0) + gbt.learning_rate * v
    return spectrum


if MAIN:
    tests.test_gbt_to_fourier(gbt_to_fourier)

# %%

def select_and_refit(
    spectrum: Spectrum,
    X: Int[np.ndarray, "Q n"],
    y: Float[np.ndarray, " Q"],
    energy: float = 0.95,
    max_terms: int | None = None,
) -> Spectrum:
    """
    Keeps the top coefficients holding `energy` of the non-constant energy (at most `max_terms` of them, if given),
    plus the constant, and refits them by ridge regression on (X, y).
    """
    terms = sorted(((S, v) for S, v in spectrum.items() if len(S) > 0), key=lambda kv: -abs(kv[1]))
    energies = np.array([v**2 for _, v in terms])
    cumulative = np.cumsum(energies) / energies.sum()
    k = int(np.searchsorted(cumulative, energy - 1e-12) + 1)
    if max_terms is not None:
        k = min(k, max_terms)
    supports = [()] + [S for S, _ in terms[:k]]
    K = supports_to_matrix(supports, X.shape[1])
    chi = 1.0 - 2.0 * ((X @ K.T) % 2)
    ridge = RidgeCV(alphas=np.logspace(-6, 2, 30), fit_intercept=False, gcv_mode="svd").fit(chi, y)
    return dict(zip(supports, ridge.coef_.tolist()))


if MAIN:
    tests.test_select_and_refit(select_and_refit)

# %%

def proxyspex(
    f: Callable[[np.ndarray], np.ndarray],
    n: int,
    budget: int,
    seed: int = 0,
    max_depth: int = 5,
    n_estimators: int = 500,
    energy: float = 0.95,
) -> tuple[Spectrum, Int[np.ndarray, "budget n"]]:
    """ProxySPEX: fit boosted trees to f on `budget` random masks, extract their spectrum, select and refit."""
    X = np.random.default_rng(seed).integers(0, 2, size=(budget, n))
    y = f(X)
    gbt = GradientBoostingRegressor(
        n_estimators=n_estimators, max_depth=max_depth, learning_rate=0.1, random_state=seed
    ).fit(X, y)
    return select_and_refit(gbt_to_fourier(gbt, n), X, y, energy, max_terms=budget // 2), X


def first_order_baseline(X: Int[np.ndarray, "Q n"], y: Float[np.ndarray, " Q"]) -> Spectrum:
    """Least-squares additive model y ~ a + sum_i w_i x_i, returned as a Fourier spectrum."""
    linear = LinearRegression().fit(X, y)
    spectrum = {(): float(linear.intercept_ + linear.coef_.sum() / 2)}
    for i, w in enumerate(linear.coef_):
        spectrum[(i,)] = float(-w / 2)
    return spectrum


if MAIN:
    tests.test_proxyspex(proxyspex)
    tests.test_first_order_baseline(first_order_baseline)

# %%

def effective_size(spectrum: Spectrum, fraction: float = 0.99) -> int:
    """Number of non-constant terms (largest first) needed to hold `fraction` of the spectrum's non-constant energy."""
    energies = np.sort([v**2 for S, v in spectrum.items() if S])[::-1]
    if len(energies) == 0 or energies.sum() == 0:
        return 0
    return int(np.searchsorted(np.cumsum(energies) / energies.sum(), fraction - 1e-12) + 1)


def spex_budget(n: int, b: int = 5, t: int = 3, n_groups: int = 3) -> int:
    """Number of queries SPEX makes with these settings."""
    return n_groups * (bch_delay_matrix(n, t).shape[0] + 1) * 2**b

# %%

if MAIN:
    held_out = {n: np.random.default_rng(123).integers(0, 2, size=(2000, n)) for n in [32, 100]}
    stage_games = [
        ("AND game", 32, utils.make_and_game(32, 15, [1, 2, 3], seed=1)),
        ("pure parity game", 32, utils.make_sparse_game(32, 5, [3], seed=3)),
        ("pure parity game", 100, utils.make_sparse_game(100, 5, [3], seed=3)),
    ]
    for name, n, game in stage_games:
        f_game = spectrum_oracle(game)
        y_test = f_game(held_out[n])
        budget = spex_budget(n)
        X_train = np.random.default_rng(0).integers(0, 2, size=(budget, n))
        y_train = f_game(X_train)
        gbt = GradientBoostingRegressor(n_estimators=500, max_depth=5, learning_rate=0.1, random_state=0).fit(X_train, y_train)
        full_spectrum = gbt_to_fourier(gbt, n)
        final_spectrum = select_and_refit(full_spectrum, X_train, y_train, max_terms=budget // 2)
        tree_r2 = 1 - ((gbt.predict(held_out[n]) - y_test) ** 2).sum() / ((y_test - y_test.mean()) ** 2).sum()
        extraction_error = np.abs(gbt.predict(held_out[n]) - predict_spectrum(full_spectrum, held_out[n])).max()
        print(f"{name}, n = {n} ({len(game)} true Fourier terms, {budget} queries)")
        print(f"  1. proxy:      trees' held-out R² = {tree_r2:.3f}")
        print(f"  2. extraction: {len(full_spectrum):,} terms, max |trees - spectrum| = {extraction_error:.1e}")
        print(f"  3. selection:  {len(final_spectrum)} terms kept (effective size {effective_size(final_spectrum)}), "
              f"final held-out R² = {faithfulness_r2(final_spectrum, held_out[n], y_test):.3f}, "
              f"true supports kept: {len(set(final_spectrum) & set(game))}/{len(game)}")

# %%

if MAIN:
    n_cmp = 100
    planted_games = {
        "AND game\n(hierarchical)": utils.make_and_game(n_cmp, 15, [1, 2, 3], seed=1),
        "sparse Fourier\n(degree <= 3)": utils.make_sparse_game(n_cmp, 40, [1, 2, 3], seed=2),
        "pure parities\n(degree 3, no footprint)": utils.make_sparse_game(n_cmp, 5, [3], seed=3),
        "dense\n(3,000 small terms)": utils.make_sparse_game(n_cmp, 3000, [1, 2, 3], seed=4, amplitude=(0.05, 0.15)),
    }
    method_results, method_sizes = {}, {}
    for name, game in tqdm(planted_games.items()):
        f_game = spectrum_oracle(game)
        y_test = f_game(held_out[n_cmp])
        spex_spectrum, spex_queries = spex(f_game, n_cmp, b=5, t=3, seed=0)
        budget = len(spex_queries)
        proxy_spectrum, X_proxy = proxyspex(f_game, n_cmp, budget, seed=0)
        shuffle_rng = np.random.default_rng(1)
        shuffled_spectrum, _ = proxyspex(lambda X: shuffle_rng.permutation(f_game(X)), n_cmp, budget, seed=0)
        first_order = first_order_baseline(X_proxy, f_game(X_proxy))
        method_results[name] = {
            "first-order baseline": faithfulness_r2(first_order, held_out[n_cmp], y_test),
            "SPEX": faithfulness_r2(spex_spectrum, held_out[n_cmp], y_test),
            "ProxySPEX": faithfulness_r2(proxy_spectrum, held_out[n_cmp], y_test),
            "ProxySPEX, shuffled responses (control)": faithfulness_r2(shuffled_spectrum, held_out[n_cmp], y_test),
        }
        method_sizes[name] = {"SPEX": effective_size(spex_spectrum), "ProxySPEX": effective_size(proxy_spectrum)}
    print(f"budget per method: {budget} queries")
    utils.plot_method_comparison(
        method_results,
        method_sizes,
        {name: len(game) - 1 for name, game in planted_games.items()},
        f"Planted games, n = {n_cmp}, {budget} queries per method",
    )
    table = pd.DataFrame(method_results).T.round(3)
    table["SPEX eff. size"] = [method_sizes[g]["SPEX"] for g in planted_games]
    table["ProxySPEX eff. size"] = [method_sizes[g]["ProxySPEX"] for g in planted_games]
    table["true terms"] = [len(g) - 1 for g in planted_games.values()]
    print(table.rename(index=lambda s: s.replace("\n", " ")).to_string())

# %%

if MAIN:
    def lookup_value_function(values: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
        return lambda masks: values[mask_index(masks)]
    
    
    spex_bs = [4, 5, 6, 7, 8]
    proxy_budgets = [256, 512, 1024, 2048, 4096]
    head_to_head, ioi_spectra = {}, {}
    for name, values in exact_values.items():
        f_ioi = lookup_value_function(values)
        spex_runs = [spex(f_ioi, n_words, b, t=3, chase_depth=3, seed=0) for b in tqdm(spex_bs, desc=f"SPEX, {name}")]
        proxy_runs = [proxyspex(f_ioi, n_words, budget, seed=0) for budget in tqdm(proxy_budgets, desc=f"ProxySPEX, {name}")]
        shuffle_rng = np.random.default_rng(1)
        shuffled_runs = [proxyspex(lambda X: shuffle_rng.permutation(f_ioi(X)), n_words, budget, seed=0) for budget in proxy_budgets]
        first_order_runs = [first_order_baseline(X, f_ioi(X)) for _, X in proxy_runs]
    
        r2 = lambda spectrum: faithfulness_r2(spectrum, X_all, values)  # noqa: E731
        de = degree_energy(exact_F[name])
        head_to_head[name] = {
            "curves": {
                "SPEX (t = 3, Chase)": ([len(X) for _, X in spex_runs], [r2(s) for s, _ in spex_runs]),
                "ProxySPEX": (proxy_budgets, [r2(s) for s, _ in proxy_runs]),
                "first-order baseline": (proxy_budgets, [r2(s) for s in first_order_runs]),
            },
            "controls": {"ProxySPEX, shuffled responses": (proxy_budgets, [r2(s) for s, _ in shuffled_runs])},
            "hlines": {"best additive explanation": de[0], "best degree <= 3 explanation": de[:3].sum()},
        }
        ioi_spectra[name] = {"SPEX": spex_runs, "ProxySPEX": proxy_runs}
    
    utils.plot_head_to_head(head_to_head, "Faithfulness on the exact GPT-2 IOI game (n = 16 words): R² on all 65,536 masks")
    for name, panel in head_to_head.items():
        print(name)
        for method, (q, r2s) in {**panel["curves"], **panel["controls"]}.items():
            print(f"  {method:<32}" + "  ".join(f"Q={qq}: {rr:+.3f}" for qq, rr in zip(q, r2s)))

# %%

if MAIN:
    for name, values in exact_values.items():
        phi = shapley_interactions(array_to_spectrum(mobius_transform(values)), max_order=1)
        phi = np.array([phi.get((i,), 0.0) for i in range(n_words)])
        shapley_model = values[0] + X_all @ phi
        r2 = 1 - ((values - shapley_model) ** 2).sum() / ((values - values.mean()) ** 2).sum()
        print(f"{name}: exact Shapley values as an additive model, R² on all masks = {r2:.3f}   "
              f"(best additive model: {degree_energy(exact_F[name])[0]:.3f})")

# %%

def permuted_support_control(
    spectrum: Spectrum, X: Int[np.ndarray, "Q n"], y: Float[np.ndarray, " Q"], seed: int = 0
) -> Spectrum:
    """Same terms as `spectrum`, but every interaction support relabelled by a random feature permutation; refit by ridge."""
    n = X.shape[1]
    perm = np.random.default_rng(seed).permutation(n)
    supports = [S if len(S) <= 1 else tuple(sorted(int(perm[i]) for i in S)) for S in spectrum]
    if () not in supports:
        supports = [()] + supports
    K = supports_to_matrix(supports, n)
    chi = 1.0 - 2.0 * ((X @ K.T) % 2)
    ridge = RidgeCV(alphas=np.logspace(-6, 2, 30), fit_intercept=False, gcv_mode="svd").fit(chi, y)
    return dict(zip(supports, ridge.coef_.tolist()))


if MAIN:
    tests.test_permuted_support_control(permuted_support_control)

# %%

if MAIN:
    budget_index = proxy_budgets.index(2048)
    rows = {}
    for name, values in exact_values.items():
        f_ioi = lookup_value_function(values)
        proxy_spectrum, X_proxy = ioi_spectra[name]["ProxySPEX"][budget_index]
        y_proxy = f_ioi(X_proxy)
        permuted_r2 = [faithfulness_r2(permuted_support_control(proxy_spectrum, X_proxy, y_proxy, seed=s), X_all, values) for s in range(5)]
        rows[name] = {
            "n terms": len(proxy_spectrum),
            "n interaction terms": sum(len(S) >= 2 for S in proxy_spectrum),
            "ProxySPEX R²": faithfulness_r2(proxy_spectrum, X_all, values),
            "permuted supports R² (mean of 5)": np.mean(permuted_r2),
            "permuted supports R² (best of 5)": np.max(permuted_r2),
            "first-order R²": faithfulness_r2(first_order_baseline(X_proxy, y_proxy), X_all, values),
        }
    print(pd.DataFrame(rows).T.round(3).to_string())

# %%

if MAIN:
    def pairwise_sii_matrix(sii: Spectrum, n: int) -> np.ndarray:
        mat = np.zeros((n, n))
        for T, v in sii.items():
            if len(T) == 2:
                mat[T[0], T[1]] = mat[T[1], T[0]] = v
        return mat
    
    
    game = "logit diff"
    exact_sii = shapley_interactions(array_to_spectrum(mobius_transform(exact_values[game])), max_order=2)
    proxy_sii = shapley_interactions(fourier_to_mobius(ioi_spectra[game]["ProxySPEX"][budget_index][0]), max_order=2)
    spex_sii = shapley_interactions(fourier_to_mobius(ioi_spectra[game]["SPEX"][-1][0]), max_order=2)
    matrices = {
        "exact (65,536 queries)": pairwise_sii_matrix(exact_sii, n_words),
        "ProxySPEX (2,048 queries)": pairwise_sii_matrix(proxy_sii, n_words),
        "SPEX (12,288 queries)": pairwise_sii_matrix(spex_sii, n_words),
    }
    utils.plot_sii_heatmaps(matrices, words, "Pairwise Shapley interaction indices, logit(Mary) - logit(John)")
    
    upper = np.triu_indices(n_words, 1)
    pairs = [(i, j) for i, j in zip(*upper)]
    for method in ["ProxySPEX (2,048 queries)", "SPEX (12,288 queries)"]:
        corr = np.corrcoef(matrices["exact (65,536 queries)"][upper], matrices[method][upper])[0, 1]
        print(f"{method}: correlation with exact pairwise SII over all {len(pairs)} pairs = {corr:.3f}")
    
    table_rows = []
    for name, sii in [("exact", exact_sii), ("ProxySPEX", proxy_sii), ("SPEX", spex_sii)]:
        top = sorted(((T, v) for T, v in sii.items() if len(T) == 2), key=lambda kv: -abs(kv[1]))[:5]
        table_rows.append({"method": name, **{f"#{r + 1}": f"{utils.support_label(T, words)} ({v:+.2f})" for r, (T, v) in enumerate(top)}})
    print(pd.DataFrame(table_rows).set_index("method").T.to_string())
    
    shapley_table = pd.DataFrame({
        name: [sii.get((i,), 0.0) for i in range(n_words)] for name, sii in [("exact", exact_sii), ("ProxySPEX", proxy_sii), ("SPEX", spex_sii)]
    }, index=[f"{w}[{i}]" for i, w in enumerate(words)])
    print("Shapley values (logit diff):")
    print(shapley_table.sort_values("exact", key=abs, ascending=False).head(6).round(3).to_string())

# %%

class CachedValueFunction:
    """Wraps a value function so that each distinct mask is evaluated (by the model) only once."""

    def __init__(self, f: Callable[[np.ndarray], np.ndarray]):
        self.f, self.cache = f, {}

    def __call__(self, masks: np.ndarray) -> np.ndarray:
        masks = np.asarray(masks, dtype=np.int64)
        keys = [row.tobytes() for row in np.packbits(masks.astype(np.uint8), axis=1)]
        first_row = {}
        for row, key in enumerate(keys):
            if key not in self.cache and key not in first_row:
                first_row[key] = row
        if first_row:
            new_values = self.f(masks[list(first_row.values())])
            self.cache.update(zip(first_row, new_values))
        return np.array([self.cache[key] for key in keys])

    @property
    def n_evaluated(self) -> int:
        return len(self.cache)

# %%

if MAIN:
    LONG_TEXT = ("Yesterday afternoon, when Mary and John finally went to the busy little store near the old station, "
                 "John happily gave a cold drink and a small sandwich")
    long_words, long_tokens, long_token_to_word = utils.tokenize_words(model, LONG_TEXT, " to")
    n_long = len(long_words)
    long_both = CachedValueFunction(make_value_function(model, long_tokens, long_token_to_word, ioi_readouts))
    print(f"{n_long} words; full sentence: logit diff = {long_both(np.ones((1, n_long)))[0, 0]:.2f}, "
          f"p(Mary) = {long_both(np.ones((1, n_long)))[0, 1]:.3f}")
    
    long_test = np.random.default_rng(99).integers(0, 2, size=(2000, n_long))
    live_results, live_spectra = {}, {}
    start = time.time()
    for column, name in enumerate(["logit diff", "p(Mary)"]):
        f_long = lambda X, c=column: long_both(X)[:, c]  # noqa: E731
        y_test = f_long(long_test)
        spex_spectrum, spex_queries = spex(f_long, n_long, b=6, t=3, chase_depth=3, seed=0)
        budget = len(spex_queries)
        proxy_spectrum, X_proxy = proxyspex(f_long, n_long, budget, seed=0)
        shuffle_rng = np.random.default_rng(1)
        shuffled_spectrum, _ = proxyspex(lambda X: shuffle_rng.permutation(f_long(X)), n_long, budget, seed=0)
        live_results[name] = {
            "first-order baseline": faithfulness_r2(first_order_baseline(X_proxy, f_long(X_proxy)), long_test, y_test),
            "SPEX": faithfulness_r2(spex_spectrum, long_test, y_test),
            "ProxySPEX": faithfulness_r2(proxy_spectrum, long_test, y_test),
            "ProxySPEX, shuffled responses (control)": faithfulness_r2(shuffled_spectrum, long_test, y_test),
        }
        live_spectra[name] = proxy_spectrum
    print(f"{budget} queries per method; {long_both.n_evaluated:,} distinct forward passes in total, {time.time() - start:.0f}s")
    utils.plot_method_bars(live_results, f"Live GPT-2 game, n = {n_long} words, {budget} queries per method: held-out R²", ylim=(-0.3, 1.05))
    print(pd.DataFrame(live_results).T.round(3).to_string())
    
    proxy_sii_long = shapley_interactions(fourier_to_mobius(live_spectra["logit diff"]), max_order=2)
    print("ProxySPEX's largest pairwise SII (logit diff):")
    for T, v in sorted(((T, v) for T, v in proxy_sii_long.items() if len(T) == 2), key=lambda kv: -abs(kv[1]))[:6]:
        print(f"  {v:+.3f}  {utils.support_label(T, long_words)}")

# %%

if MAIN:
    def explain_prompt(text: str, final: str, target: str, contrast: str | None = None, budget: int = 2048, seed: int = 0) -> None:
        words_, tokens_, token_to_word_ = utils.tokenize_words(model, text, final)
        target_id = model.to_single_token(target)
        if contrast is None:
            readout = lambda logits: logits.log_softmax(-1)[:, target_id]  # noqa: E731
        else:
            contrast_id = model.to_single_token(contrast)
            readout = lambda logits: logits[:, target_id] - logits[:, contrast_id]  # noqa: E731
        f = CachedValueFunction(make_value_function(model, tokens_, token_to_word_, readout))
        n = len(words_)
        full = f(np.ones((1, n), dtype=int))[0]
        spectrum, X = proxyspex(f, n, budget, seed=seed)
        test = np.random.default_rng(seed + 1).integers(0, 2, size=(1000, n))
        y_test = f(test)
        print(f"{n} words, read-out on the full prompt = {full:.3f}; {f.n_evaluated} forward passes")
        print(f"held-out R²: ProxySPEX = {faithfulness_r2(spectrum, test, y_test):.3f} ({len(spectrum)} terms), "
              f"first-order baseline = {faithfulness_r2(first_order_baseline(X, f(X)), test, y_test):.3f}")
        sii = shapley_interactions(fourier_to_mobius(spectrum), max_order=2)
        print("largest pairwise Shapley interactions:")
        for T, v in sorted(((T, v) for T, v in sii.items() if len(T) == 2), key=lambda kv: -abs(kv[1]))[:6]:
            print(f"  {v:+.3f}  {utils.support_label(T, words_)}")
        print("largest Shapley values:")
        for T, v in sorted(((T, v) for T, v in sii.items() if len(T) == 1), key=lambda kv: -abs(kv[1]))[:6]:
            print(f"  {v:+.3f}  {utils.support_label(T, words_)}")
    
    
    explain_prompt(
        "The food at this restaurant was not good at all, and honestly the service",
        final=" was",
        target=" terrible",
        contrast=" great",
    )

# %%
