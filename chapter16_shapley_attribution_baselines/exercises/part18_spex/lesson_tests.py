"""Short immediate tests for the staged lesson; independent analytic fixtures."""
from itertools import combinations
import hashlib

import numpy as np

try:
    from . import spex_methods as reference
except ImportError:
    import spex_methods as reference


def test_characters(character=reference.boolean_character, predict=reference.predict_spectrum):
    for n in range(1, 6):
        masks = reference.binary_masks(n)
        for size in range(n + 1):
            for support in combinations(range(n), size):
                expected = np.asarray([(-1) ** sum(int(row[i]) for i in support) for row in masks])
                np.testing.assert_array_equal(character(masks, support), expected)
                np.testing.assert_allclose(predict(masks, {(): .3, support: -.8}) if support else predict(masks, {(): -.8}),
                                           .3 - .8 * expected if support else -.8 * expected)


def test_shifts(query=reference.shifted_subsample_queries, bin_index=reference.bin_index):
    rng = np.random.default_rng(104)
    for n, b in [(3, 2), (11, 3), (31, 4)]:
        matrix = rng.integers(0, 2, (n, b), dtype=np.uint8)
        delays = rng.integers(0, 2, (7, n), dtype=np.uint8)
        actual = query(matrix, delays)
        assert actual.shape == (7, 1 << b, n)
        for delay in range(7):
            for integer in range(1 << b):
                bits = [(integer >> (b - i - 1)) & 1 for i in range(b)]
                expected = [int((sum(int(matrix[j, i]) * bits[i] for i in range(b)) + delays[delay, j]) % 2) for j in range(n)]
                np.testing.assert_array_equal(actual[delay, integer], expected)
        for support in [(), (0,), (0, n-1)]:
            expected = sum((sum(int(matrix[j, i]) for j in support) % 2) << (b-i-1) for i in range(b))
            assert bin_index(matrix, support) == expected


def test_encoding(encode=reference.bch_encode, builder=reference.make_bch_code):
    code = builder(5, 2)
    for message in reference.binary_masks(5):
        word = encode(message, code)
        np.testing.assert_array_equal(word[:5], message)
        np.testing.assert_array_equal(word[5:], (code.parity @ message) % 2)
        # Independently divide the received binary polynomial by g(x).
        remainder = word.copy()
        generator = code.generator_polynomial
        for i in range(len(word) - len(generator) + 1):
            if remainder[i]:
                remainder[i:i+len(generator)] ^= generator
        assert not np.any(remainder)


def test_code_construction(builder=reference.make_bch_code):
    # uint8-to-uint8 comparisons also avoid NumPy 1.26's mixed-dtype NaN-mask
    # path, which failed on a large identity matrix in the pinned 3.14 kernel.
    fixtures = {
        (5, 2): (15, 7, "08ac8ba78cec3d2f6a1e831ad05a29641a44c5a64152eec6200f9e8bf6f7f4e3", "dd5c6d0e223ee85156f4c379ee9c95c5746d620bb3181f7e24d4c0e6f3d9000c"),
        (28, 3): (63, 45, "7c4d4c117b125ad3ef7fe22f39415e98f3cf83db950881d8a1a608e2ad50e133", "2365e32381c99e096eda068768c5c6d65e61277671dd0e44b500d25293381540"),
        (511, 5): (1023, 973, "d3840c3e4e68eacc4a808753bc318d520a6c930a28912da06de7b9e6540d2760", "a34ab1ddeca6c794ce63882f1728ff7d8f283bd8ddf92a4ea43a844ec5134043"),
    }
    for (n, t), (nc, kc, expected_g, expected_d) in fixtures.items():
        code = builder(n, t)
        assert (code.nc, code.kc) == (nc, kc)
        assert hashlib.sha256(code.generator.tobytes()).hexdigest() == expected_g
        assert hashlib.sha256(code.delays.tobytes()).hexdigest() == expected_d
        assert np.array_equal(code.generator[:, :kc], np.eye(kc, dtype=np.uint8))
        assert np.array_equal(code.parity, code.generator[-n:, kc:].T)


def test_syndromes_and_locator(syndrome_fn=reference.bch_syndromes, bm=reference.berlekamp_massey):
    code = reference.make_bch_code(5, 2)
    for positions in [(), (0,), (4,), (1, 4), (5, 10), (0, 12)]:
        word = np.zeros(13, dtype=np.uint8)
        word[list(positions)] = 1
        actual = syndrome_fn(word, code)
        expected = []
        for j in range(1, 5):
            value = 0
            point = int(code.field.exp[j])
            for bit in word:
                value = code.field.multiply(value, point) ^ int(bit)
            expected.append(value)
        np.testing.assert_array_equal(actual, expected)
        locator = bm(actual, code.field)
        # Independent multiplication of (1 + alpha**power * z) for each error.
        expected_locator = [1]
        for position in positions:
            root_factor = int(code.field.exp[12-position])
            product = [0] * (len(expected_locator) + 1)
            for i, coefficient in enumerate(expected_locator):
                product[i] ^= coefficient
                product[i+1] ^= code.field.multiply(coefficient, root_factor)
            expected_locator = product
        np.testing.assert_array_equal(locator, expected_locator)


def test_chase(patterns=reference.chase_patterns, soft=reference.soft_decode_parity):
    code = reference.make_bch_code(31, 3)
    support = (1, 3, 7)
    reliability = reference.boolean_character(code.delays, support)[1:]
    reliability[0] *= -.01
    trials = patterns(reliability, depth=6)
    assert len(trials) <= 21 and () in trials and (0,) in trials
    assert set().union(*(set(x) for x in trials)).issubset(set(np.argsort(abs(reliability), kind="stable")[:6]))
    decoded = soft(reliability, code)
    assert decoded["decoded"].success and decoded["decoded"].support == support
    assert decoded["trials"] > 1 and 0 in decoded["flipped_parity"]
    assert len(patterns(reliability, depth=6, variant="expanded")) == 64


def test_bin_decode(check=reference.signed_bin_check, decode=reference.decode_bin):
    design = reference.make_query_design(31, 3, 3)
    support = (1, 3, 7)
    matrix = design.matrices[0]
    index = reference.bin_index(matrix, support)
    signature = reference.boolean_character(design.delays, support)
    for amplitude in [-1.25, 1.25]:
        observation = amplitude * signature
        tested = check(observation, support, matrix, design.delays, index)
        assert tested["accepted"] and tested["amplitude"] == amplitude
        result = decode(observation, matrix, design.delays, index, design.code)
        assert result["accepted"] and result["support"] == support and result["amplitude"] == amplitude
        assert not decode(observation, matrix, design.delays, index ^ 1, design.code)["accepted"]
    observation = signature.copy(); observation[0] = 1e-16
    assert not decode(observation, matrix, design.delays, index, design.code)["accepted"]
