import itertools
import math
from typing import Callable

import numpy as np
import torch as t
from scipy.linalg import hadamard
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

from part6_spex_proxyspex.utils import GaloisField, bch_field_degree, make_and_game, make_sparse_game


def _all_masks(n: int) -> np.ndarray:
    return (np.arange(2**n)[:, None] >> np.arange(n - 1, -1, -1)) & 1


def _brute_fourier_value(spectrum: dict, mask: np.ndarray) -> float:
    total = 0.0
    for support, coef in spectrum.items():
        total += coef * (-1) ** int(sum(mask[i] for i in support))
    return total


def _predict(spectrum: dict, masks: np.ndarray) -> np.ndarray:
    """Independent (vectorised) Fourier evaluation used as an oracle by several tests."""
    masks = np.asarray(masks)
    out = np.zeros(len(masks))
    for support, coef in spectrum.items():
        out += coef * (1 - 2 * (masks[:, list(support)].sum(1) % 2)) if support else coef
    return out


def _brute_mobius_value(mobius: dict, mask: np.ndarray) -> float:
    return sum(coef for support, coef in mobius.items() if all(mask[i] for i in support))


def _random_spectrum(n: int, s: int, rng: np.random.Generator, max_degree: int | None = None) -> dict:
    spectrum = {}
    max_degree = n if max_degree is None else max_degree
    while len(spectrum) < s:
        d = int(rng.integers(0, max_degree + 1))
        support = tuple(sorted(rng.choice(n, d, replace=False).tolist()))
        spectrum[support] = float(rng.normal())
    return spectrum


# ---------------------------------------------------------------------------------------------- part 1


def test_predict_spectrum(predict_spectrum: Callable):
    rng = np.random.default_rng(0)
    n = 6
    spectrum = _random_spectrum(n, 9, rng)
    spectrum[()] = 0.7
    masks = _all_masks(n)
    out = np.asarray(predict_spectrum(spectrum, masks))
    assert out.shape == (2**n,), f"Expected one prediction per mask, shape {(2**n,)}, got {out.shape}"
    expected = np.array([_brute_fourier_value(spectrum, m) for m in masks])
    np.testing.assert_allclose(out, expected, atol=1e-10, err_msg="Predictions don't match sum_S F(S) * (-1)^(sum_{i in S} x_i)")
    # sign convention: F({0}) = -0.5 plus constant 0.5 is the function x_0
    x0 = np.asarray(predict_spectrum({(): 0.5, (0,): -0.5}, masks))
    np.testing.assert_allclose(x0, masks[:, 0], err_msg="Remember chi_S(x) = +1 when an EVEN number of features in S are kept")
    assert np.allclose(np.asarray(predict_spectrum({}, masks[:3])), 0), "An empty spectrum should predict zeros"
    print("All tests in `test_predict_spectrum` passed!")


def test_fwht(fwht: Callable):
    rng = np.random.default_rng(0)
    for n in [1, 3, 6]:
        v = rng.normal(size=2**n)
        expected = hadamard(2**n) @ v / 2**n  # Sylvester ordering = our bit ordering
        np.testing.assert_allclose(fwht(v), expected, atol=1e-10, err_msg=f"Wrong transform for n={n}")
    # batch dimensions
    V = rng.normal(size=(2, 3, 16))
    out = fwht(V)
    assert out.shape == V.shape, f"fwht should preserve leading batch dims: expected {V.shape}, got {out.shape}"
    np.testing.assert_allclose(out, V @ hadamard(16).T / 16, atol=1e-10, err_msg="Batched transform is wrong")
    # it shouldn't modify its input
    V2 = V.copy()
    fwht(V)
    assert np.array_equal(V, V2), "fwht modified its input in place"
    print("All tests in `test_fwht` passed!")


def test_mobius_transform(mobius_transform: Callable):
    rng = np.random.default_rng(0)
    n = 5
    v = rng.normal(size=2**n)
    m = np.asarray(mobius_transform(v))
    assert m.shape == v.shape
    for i in range(2**n):
        expected = sum(
            (-1) ** (bin(i).count("1") - bin(j).count("1")) * v[j] for j in range(2**n) if j & i == j
        )
        assert np.isclose(m[i], expected), f"Möbius coefficient at index {i} should be {expected:.4f}, got {m[i]:.4f}"
    # reconstruction: f(x) = sum of m(S) over S contained in x
    for i in range(2**n):
        assert np.isclose(sum(m[j] for j in range(2**n) if j & i == j), v[i])
    out = np.asarray(mobius_transform(np.stack([v, 2 * v])))
    np.testing.assert_allclose(out, np.stack([m, 2 * m]), atol=1e-10, err_msg="Batch dimensions should be supported")
    print("All tests in `test_mobius_transform` passed!")


def test_fourier_to_mobius(fourier_to_mobius: Callable):
    rng = np.random.default_rng(1)
    n = 6
    for _ in range(3):
        spectrum = _random_spectrum(n, 7, rng)
        mobius = fourier_to_mobius(spectrum)
        assert isinstance(mobius, dict), "Should return a dict support -> coefficient"
        for mask in _all_masks(n):
            a, b = _brute_fourier_value(spectrum, mask), _brute_mobius_value(mobius, mask)
            assert np.isclose(a, b), f"The Möbius expansion gives {b:.4f} at mask {mask}, the Fourier one gives {a:.4f}"
    # a single parity: F({0,1}) = 1  ->  1 - 2x0 - 2x1 + 4 x0 x1
    mob = fourier_to_mobius({(0, 1): 1.0})
    expected = {(): 1.0, (0,): -2.0, (1,): -2.0, (0, 1): 4.0}
    for k, v in expected.items():
        assert np.isclose(mob.get(k, 0.0), v), f"For F({{0,1}})=1 expected Möbius {expected}, got {mob}"
    print("All tests in `test_fourier_to_mobius` passed!")


def _brute_sii(f: Callable, n: int, T: tuple[int, ...]) -> float:
    """Shapley interaction index from its definition (discrete derivatives, weighted over coalitions)."""
    others = [i for i in range(n) if i not in T]
    total = 0.0
    for r in range(len(others) + 1):
        for S in itertools.combinations(others, r):
            weight = math.factorial(len(S)) * math.factorial(n - len(S) - len(T)) / math.factorial(n - len(T) + 1)
            deriv = 0.0
            for q in range(len(T) + 1):
                for L in itertools.combinations(T, q):
                    mask = np.zeros(n, dtype=int)
                    mask[list(S) + list(L)] = 1
                    deriv += (-1) ** (len(T) - q) * f(mask)
            total += weight * deriv
    return total


def test_shapley_interactions(shapley_interactions: Callable):
    rng = np.random.default_rng(2)
    n = 5
    mobius = {(): 0.3, (0,): 1.0, (1, 2): 2.0, (2, 3, 4): -1.5, (0, 4): 0.5}
    out = shapley_interactions(mobius, max_order=2)
    f = lambda mask: _brute_mobius_value(mobius, mask)  # noqa: E731
    for T in [(i,) for i in range(n)] + list(itertools.combinations(range(n), 2)):
        expected = _brute_sii(f, n, T)
        assert np.isclose(out.get(T, 0.0), expected), f"SII{T} should be {expected:.4f}, got {out.get(T, 0.0):.4f}"
    assert all(1 <= len(T) <= 2 for T in out), "With max_order=2, only return interactions of size 1 and 2 (no constant)"
    # efficiency: Shapley values sum to f(all) - f(empty)
    phi = sum(out.get((i,), 0.0) for i in range(n))
    assert np.isclose(phi, f(np.ones(n)) - f(np.zeros(n))), "Shapley values should satisfy efficiency"
    # random game, order 3
    mobius = {k: v for k, v in _random_spectrum(n, 10, rng).items()}
    out = shapley_interactions(mobius, max_order=3)
    f = lambda mask: _brute_mobius_value(mobius, mask)  # noqa: E731
    for T in [(0, 1, 2), (1, 3), (4,)]:
        assert np.isclose(out.get(T, 0.0), _brute_sii(f, n, T)), f"Wrong SII for {T} on a random game"
    print("All tests in `test_shapley_interactions` passed!")


# ---------------------------------------------------------------------------------------------- part 2


def test_masked_final_logits(masked_final_logits: Callable, model):
    from part6_spex_proxyspex.utils import tokenize_words

    words, tokens, token_to_word = tokenize_words(model, "When Mary and John went to the store, John gave a drink", " to")
    n = len(words)
    rng = np.random.default_rng(0)
    masks = rng.integers(0, 2, size=(4, n))
    masks[0] = 1  # full sentence
    out = masked_final_logits(model, tokens, token_to_word, t.tensor(masks, device=tokens.device))
    assert out.shape == (4, model.cfg.d_vocab), f"Expected (batch, d_vocab) = (4, {model.cfg.d_vocab}), got {tuple(out.shape)}"
    with t.inference_mode():
        for row, logits in zip(masks, out):
            keep = [i for i, w in enumerate(token_to_word.tolist()) if w < 0 or row[w] == 1]
            deleted = tokens[:, keep]
            expected = model(deleted)[0, -1]
            t.testing.assert_close(
                logits.float(), expected.float(), atol=2e-3, rtol=2e-3,
                msg="Masking a word should give the same final logits as deleting all of its tokens from the prompt",
            )
    print("All tests in `test_masked_final_logits` passed!")


def test_spectrum_summaries(degree_energy: Callable, topk_energy: Callable):
    n = 3
    F = np.zeros(8)
    F[0] = 5.0  # constant: must be ignored
    F[0b100] = 2.0  # degree 1 (feature 0)
    F[0b011] = -1.0  # degree 2
    F[0b111] = 1.0  # degree 3
    de = np.asarray(degree_energy(F))
    assert de.shape == (n,), f"degree_energy should return one value per degree 1..n, got shape {de.shape}"
    np.testing.assert_allclose(de, [4 / 6, 1 / 6, 1 / 6], err_msg="Energy by degree should be the share of non-constant energy")
    curve = np.asarray(topk_energy(F))
    assert curve.shape == (7,), f"topk_energy should have 2^n - 1 entries (all non-constant coefficients), got {curve.shape}"
    np.testing.assert_allclose(curve[:3], [4 / 6, 5 / 6, 1.0])
    np.testing.assert_allclose(curve[-1], 1.0)
    print("All tests in `test_spectrum_summaries` passed!")


# ---------------------------------------------------------------------------------------------- part 3


def test_subsample_masks(subsample_masks: Callable):
    rng = np.random.default_rng(0)
    n, b = 7, 3
    M = rng.integers(0, 2, size=(n, b))
    d = rng.integers(0, 2, size=n)
    out = np.asarray(subsample_masks(M, d))
    assert out.shape == (2**b, n), f"Expected shape (2^b, n) = {(2**b, n)}, got {out.shape}"
    for idx in range(2**b):
        ell = [(idx >> (b - 1 - q)) & 1 for q in range(b)]  # feature-0-is-MSB convention, as for all masks
        expected = [(sum(M[i, q] * ell[q] for q in range(b)) + d[i]) % 2 for i in range(n)]
        assert list(out[idx]) == expected, f"Row {idx} (ell={ell}) should be M @ ell + d mod 2 = {expected}, got {list(out[idx])}"
    print("All tests in `test_subsample_masks` passed!")


def test_aliased_bins(aliased_bins: Callable):
    rng = np.random.default_rng(3)
    n, b = 8, 3
    spectrum = _random_spectrum(n, 12, rng)
    f = lambda X: _predict(spectrum, X)  # noqa: E731
    M = rng.integers(0, 2, size=(n, b))
    delays = np.concatenate([np.zeros((1, n), dtype=int), rng.integers(0, 2, size=(3, n))])
    U, masks = aliased_bins(f, M, delays)
    U = np.asarray(U)
    assert U.shape == (4, 2**b), f"Expected U of shape (n_delays, 2^b) = {(4, 2**b)}, got {U.shape}"
    assert np.asarray(masks).shape == (4, 2**b, n), f"Expected masks of shape (n_delays, 2^b, n), got {np.asarray(masks).shape}"
    # aliasing identity, by brute force over the spectrum
    expected = np.zeros((4, 2**b))
    for support, coef in spectrum.items():
        k = np.zeros(n, dtype=int)
        k[list(support)] = 1
        bits = (M.T @ k) % 2
        j = int(sum(bit << (b - 1 - q) for q, bit in enumerate(bits)))
        for p, d in enumerate(delays):
            expected[p, j] += coef * (-1) ** int(d @ k % 2)
    np.testing.assert_allclose(U, expected, atol=1e-10, err_msg="U[p, j] should equal sum over k with M^T k = j of F(k) (-1)^(d_p . k)")
    print("All tests in `test_aliased_bins` passed!")


def test_read_support_identity(read_support_identity: Callable):
    n = 6
    for support, amp in [((1, 4), 0.8), ((0,), -1.3), ((), 2.0), ((0, 2, 3, 5), -0.4)]:
        k = np.zeros(n, dtype=int)
        k[list(support)] = 1
        delays = np.concatenate([np.zeros((1, n), dtype=int), np.eye(n, dtype=int)])
        u = amp * (-1.0) ** ((delays @ k) % 2)
        got = tuple(int(i) for i in read_support_identity(u))
        assert got == support, f"Bin holding {amp} * chi_{support} should decode to {support}, got {got}"
    print("All tests in `test_read_support_identity` passed!")


def test_bch_delay_matrix(bch_delay_matrix: Callable):
    for n, t_ in [(15, 2), (20, 3), (100, 3)]:
        D = np.asarray(bch_delay_matrix(n, t_))
        m = bch_field_degree(n)
        assert D.shape == (m * t_, n), f"For n={n}, t={t_}: expected shape (m*t, n) = {(m * t_, n)}, got {D.shape}"
        assert set(np.unique(D)) <= {0, 1}, "The delay matrix should be binary"
        gf = GaloisField(m)
        # first block: column i is alpha^i written as m bits (bit q of the integer goes in row q)
        for i in [0, 1, n - 1]:
            col = int(sum(int(D[q, i]) << q for q in range(m)))
            assert col == gf.exp[i], f"Column {i} of the first block should be the bits of alpha^{i} = {gf.exp[i]}, got {col}"
    # the property we actually need: every support of size <= t has a different syndrome
    n, t_ = 15, 2
    D = np.asarray(bch_delay_matrix(n, t_))
    syndromes = {}
    for r in range(t_ + 1):
        for support in itertools.combinations(range(n), r):
            key = tuple(D[:, list(support)].sum(1) % 2)
            assert key not in syndromes, f"Supports {syndromes[key]} and {support} have the same delay signature"
            syndromes[key] = support
    print("All tests in `test_bch_delay_matrix` passed!")


def test_bch_decode(bch_decode: Callable, bch_delay_matrix: Callable):
    rng = np.random.default_rng(0)
    for n, t_ in [(10, 2), (16, 3), (100, 3), (300, 5)]:
        D = np.asarray(bch_delay_matrix(n, t_))
        for weight in range(t_ + 1):
            for _ in range(15):
                support = tuple(sorted(rng.choice(n, weight, replace=False).tolist()))
                k = np.zeros(n, dtype=int)
                k[list(support)] = 1
                got = bch_decode((D @ k) % 2, n, t_)
                got = None if got is None else tuple(int(i) for i in got)
                assert got == support, f"n={n}, t={t_}: support {support} decoded as {got}"
    # a pattern that is not the syndrome of any support of size <= t should not decode to something of size > t
    n, t_ = 16, 2
    D = np.asarray(bch_delay_matrix(n, t_))
    k = np.zeros(n, dtype=int)
    k[[1, 5, 9, 12]] = 1
    got = bch_decode((D @ k) % 2, n, t_)
    assert got is None or len(got) <= t_, "bch_decode should never return more than t positions"
    print("All tests in `test_bch_decode` passed!")


def test_detect_singleton(detect_singleton: Callable, bch_delay_matrix: Callable):
    rng = np.random.default_rng(0)
    n, b, t_ = 20, 3, 3
    D = np.asarray(bch_delay_matrix(n, t_))
    delays = np.concatenate([np.zeros((1, n), dtype=int), D])
    M = rng.integers(0, 2, size=(n, b))

    def signature(support):
        return 1.0 - 2.0 * (delays[:, list(support)].sum(1) % 2)

    def bin_of(support):
        bits = M[list(support)].sum(0) % 2 if support else np.zeros(b, dtype=int)
        return int(sum(int(bit) << (b - 1 - q) for q, bit in enumerate(bits)))

    support = (2, 7, 11)
    for amp in [1.3, -0.7]:
        u = amp * signature(support)
        out = detect_singleton(u, M, delays, bin_of(support), t_)
        assert out is not None, f"A clean singleton bin ({amp} * chi_{support}) should be detected"
        assert tuple(out[0]) == support and np.isclose(out[1], amp), f"Expected ({support}, {amp}), got {out}"
        # the right signature in the wrong bin must be rejected
        assert detect_singleton(u, M, delays, (bin_of(support) + 1) % 2**b, t_) is None, "A support that doesn't hash to bin j must be rejected"
    # empty bin
    assert detect_singleton(np.zeros(len(delays)), M, delays, 0, t_) is None, "An all-zero bin is a zero-ton, not a singleton"
    # an even collision of two supports in the same bin: residual check must reject it
    other = next(s for s in itertools.combinations(range(n), 2) if bin_of(s) == bin_of(support) and s != support)
    u = signature(support) + signature(other)
    out = detect_singleton(u, M, delays, bin_of(support), t_, gamma=0.3)
    assert out is None or tuple(out[0]) in (support, other), "A collision must not decode to a third, unrelated support"
    # soft decoding: flip one low-confidence sign
    u = 1.0 * signature(support)
    u = u.copy()
    u[5] = -0.05 * np.sign(u[5])  # tiny value with the wrong sign
    hard = detect_singleton(u, M, delays, bin_of(support), t_, chase_depth=0)
    soft = detect_singleton(u, M, delays, bin_of(support), t_, chase_depth=3)
    assert soft is not None and tuple(soft[0]) == support, f"With chase_depth=3, the flipped low-confidence sign should be corrected; got {soft}"
    assert hard is None or tuple(hard[0]) != support, "Sanity check failed: hard decoding unexpectedly succeeded"
    print("All tests in `test_detect_singleton` passed!")


def test_spex(spex: Callable):
    n = 30
    truth = make_sparse_game(n, 20, [1, 2, 3], seed=11)
    f = lambda X: _predict(truth, X)  # noqa: E731
    spectrum, queries = spex(f, n, b=4, t=3, seed=0)
    queries = np.asarray(queries)
    m = bch_field_degree(n)
    assert queries.shape == (3 * (m * 3 + 1) * 2**4, n), f"Expected C*(p+1)*2^b = {3 * (m * 3 + 1) * 16} query masks, got {queries.shape}"
    found = {S: v for S, v in spectrum.items() if abs(v) > 1e-8}
    missing = set(truth) - set(found)
    extra = set(found) - set(truth)
    assert not missing and not extra, f"Exact sparse game: missing supports {missing}, spurious supports {extra}"
    for S, v in truth.items():
        assert np.isclose(found[S], v, atol=1e-6), f"Coefficient of {S} should be {v:.4f}, got {found[S]:.4f}"
    print("All tests in `test_spex` passed!")


def test_faithfulness_r2(faithfulness_r2: Callable):
    from sklearn.metrics import r2_score

    rng = np.random.default_rng(0)
    n = 6
    spectrum = _random_spectrum(n, 5, rng)
    masks = rng.integers(0, 2, size=(50, n))
    values = np.array([_brute_fourier_value(spectrum, m) for m in masks]) + 0.3 * rng.normal(size=50)
    pred = np.array([_brute_fourier_value(spectrum, m) for m in masks])
    assert np.isclose(faithfulness_r2(spectrum, masks, values), r2_score(values, pred))
    assert np.isclose(faithfulness_r2({}, masks, values), r2_score(values, np.zeros(50)))
    print("All tests in `test_faithfulness_r2` passed!")


# ---------------------------------------------------------------------------------------------- part 4


def test_tree_to_fourier(tree_to_fourier: Callable):
    rng = np.random.default_rng(0)
    n = 6
    X = rng.integers(0, 2, size=(300, n))
    y = X[:, 0] * X[:, 1] - 2 * X[:, 2] * X[:, 3] * X[:, 4] + 0.1 * rng.normal(size=300)
    tree = DecisionTreeRegressor(max_depth=4, random_state=0).fit(X, y)
    spectrum = tree_to_fourier(tree, n)
    masks = _all_masks(n)
    np.testing.assert_allclose(
        _predict(spectrum, masks), tree.predict(masks), atol=1e-10,
        err_msg="The Fourier expansion of a tree should reproduce tree.predict on every mask",
    )
    depth = tree.get_depth()
    assert max(len(S) for S in spectrum) <= depth, "A depth-d tree has no Fourier terms of degree > d"
    # stump: a single split on feature 3
    stump = DecisionTreeRegressor(max_depth=1).fit(X, 5.0 * X[:, 3])
    spec = tree_to_fourier(stump, n)
    assert np.isclose(spec.get((), 0), 2.5) and np.isclose(spec.get((3,), 0), -2.5), f"Stump 5 * x_3 should give {{(): 2.5, (3,): -2.5}}, got {spec}"
    print("All tests in `test_tree_to_fourier` passed!")


def test_gbt_to_fourier(gbt_to_fourier: Callable):
    rng = np.random.default_rng(1)
    n = 7
    X = rng.integers(0, 2, size=(400, n))
    y = 2 * X[:, 0] * X[:, 1] + X[:, 5] - X[:, 2] * X[:, 6] + 0.1 * rng.normal(size=400)
    gbt = GradientBoostingRegressor(n_estimators=30, max_depth=3, learning_rate=0.2, random_state=0).fit(X, y)
    spectrum = gbt_to_fourier(gbt, n)
    masks = _all_masks(n)
    np.testing.assert_allclose(
        _predict(spectrum, masks), gbt.predict(masks), atol=1e-8,
        err_msg="The ensemble's spectrum should reproduce gbt.predict on every mask (did you include init_ and the learning rate?)",
    )
    print("All tests in `test_gbt_to_fourier` passed!")


def test_select_and_refit(select_and_refit: Callable):
    rng = np.random.default_rng(0)
    n = 8
    truth = {(): 2.0, (0,): 1.0, (1, 2): -0.8, (3, 4, 5): 0.6}
    X = rng.integers(0, 2, size=(400, n))
    y = _predict(truth, X)
    # a noisy proxy spectrum: the true supports with wrong values, plus many small spurious terms
    proxy = {S: v * 0.8 for S, v in truth.items()}
    proxy[()] = 0.0
    for S in [(6,), (7,), (2, 6), (0, 7), (1, 3)]:
        proxy[S] = 0.01
    out = select_and_refit(proxy, X, y, energy=0.95)
    assert () in out, "The constant term must always be kept (its value is refit, not selected by energy)"
    kept = {S for S in out if S}
    assert kept == {S for S in truth if S}, f"With 95% energy, the three big terms should be kept and the small ones dropped; kept {kept}"
    for S, v in truth.items():
        assert np.isclose(out[S], v, atol=1e-3), f"After refitting, coefficient of {S} should be {v}, got {out[S]:.4f}"
    # inclusive boundary: if the top term alone has exactly the required energy, keep just that term
    out = select_and_refit({(): 0.0, (0,): 3.0, (1,): 1.0}, X, 3 - 3 * (1 - 2 * X[:, 0]), energy=0.9)
    assert {S for S in out if S} == {(0,)}, f"The top term carries 90% of the energy, so energy=0.9 should keep only it; got {out}"
    print("All tests in `test_select_and_refit` passed!")


def test_proxyspex(proxyspex: Callable):
    n = 10
    truth = make_and_game(n, 4, [1, 2, 3], seed=4)
    f = lambda X: _predict(truth, X)  # noqa: E731
    spectrum, X = proxyspex(f, n, budget=800, seed=0)
    X = np.asarray(X)
    assert X.shape == (800, n), f"ProxySPEX should query exactly `budget` random masks; got {X.shape}"
    test = _all_masks(n)
    y = f(test)
    r2 = 1 - ((_predict(spectrum, test) - y) ** 2).mean() / y.var()
    assert r2 > 0.95, f"On a 4-term AND game with 800 queries, ProxySPEX should reach R² > 0.95 (got {r2:.3f})"
    print("All tests in `test_proxyspex` passed!")


def test_first_order_baseline(first_order_baseline: Callable):
    rng = np.random.default_rng(0)
    n = 5
    X = rng.integers(0, 2, size=(200, n))
    w = np.array([1.0, -2.0, 0.5, 0.0, 3.0])
    y = 0.7 + X @ w
    spectrum = first_order_baseline(X, y)
    assert all(len(S) <= 1 for S in spectrum), "A first-order baseline should only contain the constant and singletons"
    masks = _all_masks(n)
    np.testing.assert_allclose(_predict(spectrum, masks), 0.7 + masks @ w, atol=1e-8)
    assert np.isclose(spectrum[(1,)], 1.0), "Weight -2 on x_1 should become Fourier coefficient +1 on chi_{1}"
    print("All tests in `test_first_order_baseline` passed!")


def test_permuted_support_control(permuted_support_control: Callable):
    rng = np.random.default_rng(0)
    n = 12
    spectrum = {(): 1.0, (0,): 0.5, (3,): -0.2, (1, 2): 0.7, (2, 5, 6): -0.3, (7, 8): 0.1}
    X = rng.integers(0, 2, size=(300, n))
    y = rng.normal(size=300)
    out = permuted_support_control(spectrum, X, y, seed=0)
    assert len(out) == len(spectrum), "The control should have exactly as many terms as the original"
    assert () in out and (0,) in out and (3,) in out, "The constant and first-order terms should be kept unchanged"
    original = [S for S in spectrum if len(S) >= 2]
    permuted = [S for S in out if len(S) >= 2]
    assert sorted(map(len, permuted)) == sorted(map(len, original)), "Interaction supports should keep their degrees"
    assert set(permuted) != set(original), "Interaction supports should be relabelled (seed 0 should move them)"
    # a relabelling preserves the overlap structure: (1,2) and (2,5,6) share exactly one feature
    overlaps = sorted(len(set(a) & set(b)) for a, b in itertools.combinations(permuted, 2))
    assert overlaps == [0, 0, 1], f"A feature permutation preserves which interactions share features; got overlaps {overlaps}"
    # same permutation as the reference definition
    perm = np.random.default_rng(0).permutation(n)
    assert tuple(sorted(int(perm[i]) for i in (1, 2))) in out, "Use np.random.default_rng(seed).permutation(n)"
    # the refit should be a least-squares-like fit: predictions correlate with y at least as well as the mean
    pred = _predict(out, X)
    assert ((y - pred) ** 2).mean() <= ((y - y.mean()) ** 2).mean() + 1e-9, "Coefficients should be refit to (X, y)"
    print("All tests in `test_permuted_support_control` passed!")
