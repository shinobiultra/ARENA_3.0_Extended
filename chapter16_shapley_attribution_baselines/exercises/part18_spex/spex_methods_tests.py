"""Immediate analytic tests; functions accept student implementations explicitly.

No native model, galois, sparse-transform or large exhaustive oracle is used.
Reference constants below came from the unchanged author/package execution.
"""
import hashlib
from itertools import combinations
from math import factorial

import numpy as np
import pytest

try:
    from . import spex_methods as reference
except ImportError:
    import spex_methods as reference


def all_supports(n):
    return [support for size in range(n + 1) for support in combinations(range(n), size)]


def polynomial_truth():
    # Independent expansion of .3 + x0 - .7*x3 + 1.2*x1*x7 - 2.4*x2*x4*x9.
    spectrum = {(): .3}
    for monomial, weight in [((0,), 1), ((3,), -.7), ((1, 7), 1.2), ((2, 4, 9), -2.4)]:
        for subset in all_supports(len(monomial)):
            support = tuple(monomial[i] for i in subset)
            spectrum[support] = spectrum.get(support, 0) + weight * (-1) ** len(subset) / 2 ** len(monomial)
    return spectrum


def test_boolean_basis(character=reference.boolean_character, transform=reference.normalized_fwht):
    """Every basis vector through n=5; orthogonality and forward/inverse signs."""
    checked = 0
    for n in range(1, 6):
        masks = reference.binary_masks(n)
        basis = []
        for support in all_supports(n):
            truth = np.asarray([(-1) ** sum(int(row[i]) for i in support) for row in masks])
            actual = character(masks, support)
            np.testing.assert_array_equal(actual, truth)
            transformed = transform(actual)
            expected = np.zeros(1 << n)
            index = sum(1 << (n - i - 1) for i in support)
            expected[index] = 1
            np.testing.assert_array_equal(transformed, expected)
            np.testing.assert_array_equal(transform(transformed) * (1 << n), actual)
            basis.append(actual)
            checked += 1
        np.testing.assert_array_equal(np.asarray(basis) @ np.asarray(basis).T, (1 << n) * np.eye(1 << n))
    assert checked == 62
    values = np.arange(24, dtype=float).reshape(8, 3)
    np.testing.assert_allclose(transform(transform(values, axis=0), axis=0) * 8, values)


def test_systematic_reference_matrices(builder=reference.make_bch_code):
    """Byte-identical G/D to galois.BCH, including the non-Conway fields."""
    expected = {
        (5, 2): (15, 7, "08ac8ba78cec3d2f6a1e831ad05a29641a44c5a64152eec6200f9e8bf6f7f4e3", "dd5c6d0e223ee85156f4c379ee9c95c5746d620bb3181f7e24d4c0e6f3d9000c"),
        (28, 3): (63, 45, "7c4d4c117b125ad3ef7fe22f39415e98f3cf83db950881d8a1a608e2ad50e133", "2365e32381c99e096eda068768c5c6d65e61277671dd0e44b500d25293381540"),
        (511, 5): (1023, 973, "d3840c3e4e68eacc4a808753bc318d520a6c930a28912da06de7b9e6540d2760", "a34ab1ddeca6c794ce63882f1728ff7d8f283bd8ddf92a4ea43a844ec5134043"),
    }
    for (n, t), (nc, kc, g_hash, d_hash) in expected.items():
        code = builder(n, t)
        assert (code.nc, code.kc) == (nc, kc)
        assert hashlib.sha256(code.generator.tobytes()).hexdigest() == g_hash
        assert hashlib.sha256(code.delays.tobytes()).hexdigest() == d_hash
        np.testing.assert_array_equal(code.generator[:, :kc], np.eye(kc))
        np.testing.assert_array_equal(code.parity, code.generator[-n:, kc:].T)


def test_bch_guaranteed_radius(encode=reference.bch_encode, decoder=reference.decode_bch_parity):
    """All 92 n=5,t=2 support+parity-error cases, not just clean supports."""
    code = reference.make_bch_code(5, 2)
    count = 0
    for support_weight in range(code.t + 1):
        for support in combinations(range(code.n), support_weight):
            message = np.zeros(code.n, dtype=np.uint8)
            message[list(support)] = 1
            word = encode(message, code)
            np.testing.assert_array_equal(word[:code.n], message)
            np.testing.assert_array_equal(reference.bch_syndromes(word, code), 0)
            for error_weight in range(code.t - support_weight + 1):
                for errors in combinations(range(code.nc - code.kc), error_weight):
                    parity = word[code.n:].copy()
                    parity[list(errors)] ^= 1
                    result = decoder(parity, code)
                    assert result.success, (support, errors, result)
                    assert result.support == support
                    assert result.corrected_positions == tuple(sorted(support + tuple(code.n + i for i in errors)))
                    count += 1
    assert count == 92


def test_bch_scaling_and_decoder_steps(bm=reference.berlekamp_massey, chien=reference.chien_search):
    """All t=1..5 at representative field boundaries, through 511 features."""
    rng = np.random.default_rng(1209)
    for n in (1, 5, 28, 31, 63, 64, 127, 128, 255, 256, 511):
        for t in range(1, 6):
            code = reference.make_bch_code(n, t)
            for weight in range(min(n, t) + 1):
                support = tuple(sorted(map(int, rng.choice(n, weight, replace=False))))
                message = np.zeros(n, dtype=np.uint8)
                message[list(support)] = 1
                parity = reference.bch_encode(message, code)[n:]
                errors = tuple(sorted(map(int, rng.choice(len(parity), t - weight, replace=False))))
                parity[list(errors)] ^= 1
                received = np.concatenate((np.zeros(n, dtype=np.uint8), parity))
                syndromes = reference.bch_syndromes(received, code)
                locator = bm(syndromes, code.field)
                positions = chien(locator, code)
                expected = tuple(sorted(tuple(i + code.shortened for i in support) +
                                        tuple(code.kc + i for i in errors)))
                assert positions == expected
                assert reference.decode_bch_parity(parity, code).support == support


def test_query_order_and_shift_identity(make_design=reference.make_query_design,
                                       transform_values=reference.transform_query_values):
    truth = polynomial_truth()
    for b, requested, unique in [(3, 456, 418), (4, 912, 874), (5, 1824, 1786)]:
        design = make_design(28, 3, b, seed=0)
        assert (design.requested_count, design.unique_count) == (requested, unique)
        assert design.masks.shape == (requested, 28)
        expected_matrices = np.random.RandomState(0).randint(2, size=(3, 28, b))
        np.testing.assert_array_equal(design.matrices, expected_matrices)
        ell = reference.binary_masks(b)
        for c in range(3):
            for delay in range(19):
                offset = (c * 19 + delay) * (1 << b)
                expected = (ell @ expected_matrices[c].T + design.delays[delay]) % 2
                np.testing.assert_array_equal(design.masks[offset:offset + (1 << b)], expected)
        bins = transform_values(design, reference.predict_spectrum(design.masks, truth))
        independent = np.zeros_like(bins)
        for c in range(3):
            for support, coefficient in truth.items():
                index = reference.bin_index(design.matrices[c], support)
                independent[c, :, index] += coefficient * reference.boolean_character(design.delays, support)
        np.testing.assert_allclose(bins, independent, atol=1e-15)
    assert reference.mask_ids(np.array([[1, 0, 0], [0, 0, 1]])) == [4, 1]
    long_mask = np.zeros((1, 511), dtype=np.uint8)
    long_mask[0, 0] = 1
    assert reference.mask_ids(long_mask) == [1 << 510]


def test_chase_and_signed_amplitudes(soft_decode=reference.soft_decode_parity,
                                    check=reference.signed_bin_check):
    code = reference.make_bch_code(31, 3)
    support = (1, 3, 7)
    signature = reference.boolean_character(code.delays, support)
    reliability = signature[1:].copy()
    reliability[0] *= -0.01  # One low-confidence error exceeds the hard bound.
    assert not reference.decode_bch_parity((reliability < 0).astype(np.uint8), code).success
    selected = soft_decode(reliability, code, chase_depth=6)
    assert selected["decoded"].success and selected["decoded"].support == support
    assert selected["trials"] > 1 and 0 in selected["flipped_parity"]
    assert len(reference.chase_patterns(reliability, 6)) <= 21
    assert len(reference.chase_patterns(reliability, 6, variant="expanded")) == 64
    matrix = reference.make_query_design(31, 3, 3).matrices[0]
    index = reference.bin_index(matrix, support)
    for amplitude in (1.25, -1.25):
        observation = amplitude * signature
        result = check(observation, support, matrix, code.delays, index)
        assert result["accepted"]
        assert result["amplitude"] == amplitude
        assert result["residual_fraction"] == 0
        assert not check(observation, support, matrix, code.delays, index ^ 1)["accepted"]
    orthogonal = np.zeros_like(signature)
    orthogonal[0], orthogonal[1] = signature[1], -signature[0]
    assert not check(orthogonal, support, matrix, code.delays, index)["accepted"]

    # A zero-shift cancellation must not turn tiny roundoff into huge ratios.
    cancellation = signature.copy()
    cancellation[0] = 1e-16
    rejected = reference.decode_bin(cancellation, matrix, code.delays, index, code)
    assert not rejected["accepted"] and rejected["reason"] == "zero reference shift"


def test_small_cube_recovery(recover=reference.recover_spectrum):
    """Every signed basis term through five features is reachable and recovered."""
    for n in range(1, 6):
        design = reference.make_query_design(n, n, n, seed=0)
        for support in all_supports(n):
            truth = {support: -1.25}
            result = recover(design, reference.predict_spectrum(design.masks, truth))
            assert result.spectrum.keys() == truth.keys()
            assert result.spectrum[support] == pytest.approx(-1.25, abs=1e-12)


def test_iterative_collisions_and_accounting(recover=reference.recover_spectrum):
    truth = polynomial_truth()
    design = reference.make_query_design(31, 3, 3, seed=1)
    values = reference.predict_spectrum(design.masks, truth)
    result = recover(design, values)
    assert set(result.spectrum) == set(truth)
    assert len(result.trace) >= 3, "The collision example must require later message-passing sweeps"
    initial_occupancies = np.zeros((3, 8), dtype=int)
    for c, matrix in enumerate(design.matrices):
        for support in truth:
            initial_occupancies[c, reference.bin_index(matrix, support)] += 1
    assert initial_occupancies.max() > 1
    residual = result.initial_bins.copy()
    for sweep in result.trace:
        assert sweep["energy_before"] == pytest.approx(float(np.sum(residual ** 2)))
        for update in sweep["updates"]:
            support, amplitude = update["support"], update["amplitude_delta"]
            signature = reference.boolean_character(design.delays, support)
            for c, index in update["touched_bins"]:
                residual[c, :, index] -= amplitude * signature
        np.testing.assert_allclose(sweep["energy_after"], np.sum(residual ** 2), atol=1e-14)
    np.testing.assert_allclose(residual, result.residual_bins, atol=1e-14)
    audit = np.random.default_rng(7701).integers(0, 2, size=(1024, 31))
    rmse = np.sqrt(np.mean((reference.predict_spectrum(audit, truth) - reference.predict_spectrum(audit, result.spectrum)) ** 2))
    assert rmse < 1e-4
    assert (result.requested_count, result.unique_count) == (design.requested_count, design.unique_count)
    called = []
    def oracle(masks):
        called.append(masks.copy())
        return reference.predict_spectrum(masks, truth)
    sampled = reference.sample_and_recover(design, oracle)
    assert len(called) == 1 and len(called[0]) == design.requested_count
    assert sampled.spectrum == result.spectrum


def test_wrong_degree_dense_and_iteration_controls(recover=reference.recover_spectrum):
    parity = {(1, 3, 7): -1.25}
    wrong_design = reference.make_query_design(31, 1, 3, seed=0)
    wrong = recover(wrong_design, reference.predict_spectrum(wrong_design.masks, parity))
    assert wrong.spectrum == {}  # Untouched author-style default fails visibly.
    design = reference.make_query_design(31, 3, 3, seed=1)
    truth = polynomial_truth()
    capped = recover(design, reference.predict_spectrum(design.masks, truth), max_iterations=1)
    assert capped.stop_reason == "iteration limit" and capped.residual_energy > 1e-4
    rng = np.random.default_rng(884)
    supports = list(combinations(range(31), 3))
    chosen = rng.choice(len(supports), 256, replace=False)
    coefficients = rng.normal(size=256)
    coefficients /= np.linalg.norm(coefficients)
    dense = {supports[int(i)]: float(v) for i, v in zip(chosen, coefficients)}
    result = recover(design, reference.predict_spectrum(design.masks, dense))
    audit = rng.integers(0, 2, size=(2048, 31))
    error = reference.predict_spectrum(audit, dense) - reference.predict_spectrum(audit, result.spectrum)
    assert np.sqrt(np.mean(error ** 2)) > .4
    assert len(set(dense) - set(result.spectrum)) > 100


def finite_difference_sii(values, n, interaction):
    """Independent factorial-weighted definition, only for small exhaustive tests."""
    outside = tuple(i for i in range(n) if i not in interaction)
    total = 0.0
    for size in range(len(outside) + 1):
        weight = factorial(size) * factorial(n - len(interaction) - size) / factorial(n - len(interaction) + 1)
        for base in combinations(outside, size):
            difference = 0.0
            for selected_size in range(len(interaction) + 1):
                for selected in combinations(interaction, selected_size):
                    identifier = sum(1 << (n - i - 1) for i in base + selected)
                    difference += (-1) ** (len(interaction) - selected_size) * values[identifier]
            total += weight * difference
    return total


def test_fourier_to_sii_and_shapley(convert=reference.fourier_to_sii,
                                  shapley=reference.fourier_to_shapley):
    """Independent finite differences for every basis, every interaction, n<=5."""
    for n in range(1, 6):
        masks = reference.binary_masks(n)
        targets = all_supports(n)[1:]
        for support in all_supports(n):
            spectrum = {support: -1.25}
            values = reference.predict_spectrum(masks, spectrum)
            result = convert(spectrum, targets)
            for target in targets:
                assert result[target] == pytest.approx(finite_difference_sii(values, n, target), abs=3e-14)
            np.testing.assert_allclose(shapley(spectrum, n), [result[(i,)] for i in range(n)], atol=1e-14)
            assert sum(shapley(spectrum, n)) == pytest.approx(values[-1] - values[0], abs=1e-14)


def test_invalid_inputs_and_no_large_power_set():
    with pytest.raises(ValueError):
        reference.binary_masks(28)
    with pytest.raises(ValueError):
        reference.make_bch_code(512, 3)
    with pytest.raises(ValueError):
        reference.make_bch_code(28, 6)
    with pytest.raises(ValueError):
        reference.boolean_character([[0, 2]], (0,))
    with pytest.raises(ValueError):
        reference.normalized_fwht(np.ones(3))
    design = reference.make_query_design(28, 3, 3)
    with pytest.raises(ValueError):
        reference.recover_spectrum(design, np.ones(design.unique_count))
    with pytest.raises(ValueError):
        reference.recover_spectrum(design, np.full(design.requested_count, np.nan))
