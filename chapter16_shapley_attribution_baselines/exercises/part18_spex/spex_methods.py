"""Learner-owned, bounded SPEX operations, with signed Fourier coefficients.

Conventions: masks are (Q, n) binary arrays; feature 0 is column 0 and the
most significant bit of an integer mask ID. Supports are sorted tuples of
feature positions, including () for the constant. chi_S(x)=(-1)**sum(x[S]).
Only the b-dimensional query subcube is enumerated; never the n-feature cube.
See MODULE_API.md for source differences and the pre-native numerical defaults.
"""
from dataclasses import dataclass
from functools import lru_cache
import heapq
from typing import Callable, Iterable, Mapping

import numpy as np

try:
    from .spex_coding import BinaryField, binary_field
except ImportError:  # Also works with this directory on the notebook's path.
    from spex_coding import BinaryField, binary_field


Spectrum = dict[tuple[int, ...], float]


def binary_masks(n: int) -> np.ndarray:
    """Enumerate a small cube in big-endian order (guarded at n <= 16)."""
    if not isinstance(n, (int, np.integer)) or not 0 <= n <= 16:
        raise ValueError("Exhaustive cubes are restricted to 0 <= n <= 16")
    return ((np.arange(1 << n)[:, None] >> np.arange(n - 1, -1, -1)) & 1).astype(np.uint8)


def mask_ids(masks: np.ndarray) -> list[int]:
    """Python integers preserve IDs even for n > 64."""
    masks = _binary_array(masks, ndim=2)
    return [sum(int(bit) << (len(row) - 1 - i) for i, bit in enumerate(row)) for row in masks]


def _binary_array(array, *, ndim: int) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim != ndim or not np.all((array == 0) | (array == 1)):
        raise ValueError(f"Expected a binary array with {ndim} dimensions")
    return array.astype(np.uint8, copy=False)


def boolean_character(masks: np.ndarray, support: tuple[int, ...]) -> np.ndarray:
    """Evaluate one signed parity without a dense Fourier basis matrix."""
    masks = _binary_array(masks, ndim=2)
    support = tuple(support)
    if len(set(support)) != len(support) or any(i < 0 or i >= masks.shape[1] for i in support):
        raise ValueError("Support positions must be distinct and inside the feature dimension")
    return 1.0 - 2.0 * (masks[:, support].sum(axis=1) % 2)


def normalized_fwht(values: np.ndarray, axis: int = -1) -> np.ndarray:
    """Butterfly Walsh transform divided by its length; signed and real.

    Applying this twice returns values / length. Multiply once by length
    when using this normalized forward operation as the inverse.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim == 0 or not np.all(np.isfinite(values)):
        raise ValueError("Transform input must be a finite array")
    work = np.moveaxis(values, axis, -1).copy()
    length = work.shape[-1]
    if length < 1 or length & (length - 1):
        raise ValueError("Walsh transform length must be a positive power of two")
    width = 1
    while width < length:
        blocks = work.reshape(*work.shape[:-1], -1, 2 * width)
        left, right = blocks[..., :width].copy(), blocks[..., width:].copy()
        blocks[..., :width], blocks[..., width:] = left + right, left - right
        width *= 2
    return np.moveaxis(work / length, -1, axis)


def predict_spectrum(masks: np.ndarray, spectrum: Mapping[tuple[int, ...], float]) -> np.ndarray:
    masks = _binary_array(masks, ndim=2)
    prediction = np.zeros(len(masks), dtype=float)
    for support, coefficient in spectrum.items():
        if not np.isfinite(coefficient):
            raise ValueError("Fourier coefficients must be finite")
        prediction += float(coefficient) * boolean_character(masks, support)
    return prediction


@dataclass(frozen=True)
class BCHCode:
    n: int
    t: int
    nc: int
    kc: int
    field: BinaryField
    generator_polynomial: np.ndarray  # coefficients in descending powers
    generator: np.ndarray  # full row-generator [I | P.T], shape (kc, nc)
    parity: np.ndarray  # last n message positions, shape (nc-kc, n)

    @property
    def delays(self):
        return np.vstack((np.zeros((1, self.n), dtype=np.uint8), self.parity))

    @property
    def shortened(self):
        return self.kc - self.n


def bch_generator_polynomial(field: BinaryField, t: int) -> np.ndarray:
    """Multiply roots alpha**1,...,alpha**(2t) and their binary conjugates.

    Frobenius closure makes the resulting coefficients binary. This is a
    primitive narrow-sense BCH generator of designed distance 2*t + 1.
    """
    root_powers = set()
    for first in range(1, 2 * t + 1):
        power = first % field.order
        while power not in root_powers:
            root_powers.add(power)
            power = (2 * power) % field.order
    polynomial = [1]  # ascending powers during multiplication
    for power in sorted(root_powers):
        root = int(field.exp[power])
        product = [0] * (len(polynomial) + 1)
        for i, coefficient in enumerate(polynomial):
            product[i] ^= field.multiply(coefficient, root)
            product[i + 1] ^= coefficient
        polynomial = product
    if any(coefficient not in (0, 1) for coefficient in polynomial):
        raise ArithmeticError("Frobenius-closed generator must have binary coefficients")
    return np.asarray(polynomial[::-1], dtype=np.uint8)


@lru_cache(maxsize=128)
def make_bch_code(n: int, t: int) -> BCHCode:
    """Construct the author's systematic shortened BCH design, n<=511,t<=5.

    The full row-generator is [I | P_full.T]. Keep its LAST n message rows,
    then transpose the parity portion. This is G=[I;P] in the paper's
    column-vector convention, not an unrelated parity-check construction.
    """
    if not isinstance(n, (int, np.integer)) or not 1 <= n <= 511:
        raise ValueError("Supported feature dimensions are 1 <= n <= 511")
    if not isinstance(t, (int, np.integer)) or not 1 <= t <= 5:
        raise ValueError("Supported BCH degree budgets are 1 <= t <= 5")
    start = (2 * t + 1).bit_length()
    for m in range(start, 11):
        field = binary_field(m)
        polynomial = bch_generator_polynomial(field, t)
        nc, p = field.order, len(polynomial) - 1
        kc = nc - p
        if kc >= n:
            break
    else:
        raise ValueError("No supported BCH field has sufficient message dimension")
    # Systematic cyclic-generator recurrence used by galois 0.4.11.
    parity_rows = np.zeros((kc, p), dtype=np.uint8)
    parity_rows[0] = polynomial[:-1]
    for i in range(1, kc):
        parity_rows[i, 1:] = parity_rows[i - 1, :-1]
        if parity_rows[i - 1, -1]:
            parity_rows[i] ^= parity_rows[0]
    generator = np.hstack((np.eye(kc, dtype=np.uint8), parity_rows))
    parity = parity_rows[-n:].T.copy()
    for array in (polynomial, generator, parity):
        array.flags.writeable = False
    return BCHCode(n, t, nc, kc, field, polynomial, generator, parity)


def bch_encode(message: np.ndarray, code: BCHCode) -> np.ndarray:
    """Systematic shortened codeword [message, P @ message mod 2]."""
    message = _binary_array(message, ndim=1)
    if message.shape != (code.n,):
        raise ValueError("Message length does not match the shortened code")
    return np.concatenate((message, (code.parity @ message) % 2))


def bch_syndromes(received: np.ndarray, code: BCHCode) -> np.ndarray:
    """Evaluate the received polynomial at alpha**j for j=1,...,2t.

    Shortened leading zero positions are implicit. The first received bit
    is the coefficient of x**(n+p-1), with p=nc-kc.
    """
    received = _binary_array(received, ndim=1)
    if len(received) != code.n + code.nc - code.kc:
        raise ValueError("Received word has the wrong shortened length")
    powers = len(received) - 1 - np.flatnonzero(received)
    if len(powers) == 0:
        return np.zeros(2 * code.t, dtype=int)
    exponents = np.arange(1, 2 * code.t + 1)[:, None] * powers[None, :]
    return np.bitwise_xor.reduce(code.field.exp[exponents % code.nc], axis=1)


def berlekamp_massey(syndromes: np.ndarray, field: BinaryField) -> np.ndarray:
    """Shortest error-locator recurrence; coefficients are ASCENDING powers.

    At step r, discrepancy S[r] + sum(C[i]*S[r-i]) measures whether the
    current recurrence explains this syndrome. A nonzero discrepancy adds
    a shifted, rescaled previous recurrence; all additions are XOR.
    """
    syndromes = np.asarray(syndromes, dtype=int)
    if syndromes.ndim != 1 or np.any((syndromes < 0) | (syndromes > field.order)):
        raise ValueError("Syndromes must be field elements")
    C, B = [1] + [0] * len(syndromes), [1] + [0] * len(syndromes)
    degree, gap, previous_discrepancy = 0, 1, 1
    for r in range(len(syndromes)):
        discrepancy = int(syndromes[r])
        for i in range(1, degree + 1):
            discrepancy ^= field.multiply(C[i], int(syndromes[r - i]))
        if discrepancy == 0:
            gap += 1
            continue
        old_C = C.copy()
        scale = field.divide(discrepancy, previous_discrepancy)
        for i in range(len(C) - gap):
            C[i + gap] ^= field.multiply(scale, B[i])
        if 2 * degree <= r:
            degree = r + 1 - degree
            B, previous_discrepancy, gap = old_C, discrepancy, 1
        else:
            gap += 1
    return np.asarray(C[:degree + 1], dtype=int)


def chien_search(locator: np.ndarray, code: BCHCode) -> tuple[int, ...]:
    """Find error positions by evaluating locator(alpha**(-position_power)).

    Search all nc code coordinates, O(nc*t), so errors in removed leading
    message positions can be rejected explicitly by the shortened decoder.
    """
    locator = np.asarray(locator, dtype=int)
    roots = code.field.exp[(-(code.nc - 1 - np.arange(code.nc))) % code.nc]
    values = np.zeros(code.nc, dtype=int)
    for coefficient in locator[::-1]:
        values = code.field.multiply_array(values, roots) ^ int(coefficient)
    return tuple(int(i) for i in np.flatnonzero(values == 0))


@dataclass(frozen=True)
class BCHDecode:
    success: bool
    support: tuple[int, ...] | None
    corrected_positions: tuple[int, ...]
    syndromes: tuple[int, ...]
    locator: tuple[int, ...]
    reason: str


def decode_bch_parity(parity_bits: np.ndarray, code: BCHCode) -> BCHDecode:
    """Decode [zero message, observed parity] using syndromes -> BM -> Chien.

    Guaranteed when support_weight + parity_error_weight <= t. Outside that
    ball, a different valid codeword is possible; bin checks are still needed.
    """
    parity_bits = _binary_array(parity_bits, ndim=1)
    if len(parity_bits) != code.nc - code.kc:
        raise ValueError("Parity length does not match the BCH code")
    received = np.concatenate((np.zeros(code.n, dtype=np.uint8), parity_bits))
    syndrome = bch_syndromes(received, code)
    locator = berlekamp_massey(syndrome, code.field)
    roots = chien_search(locator, code)
    reason = "decoded"
    if len(locator) - 1 > code.t or len(roots) != len(locator) - 1:
        reason = "locator has wrong number of roots"
    elif any(i < code.shortened for i in roots):
        reason = "error lies in a removed message position"
    positions = tuple(i - code.shortened for i in roots)
    if reason == "decoded":
        corrected = received.copy()
        corrected[list(positions)] ^= 1
        if np.any(bch_syndromes(corrected, code)):
            reason = "corrected word has nonzero syndrome"
        elif not np.array_equal(corrected, bch_encode(corrected[:code.n], code)):
            reason = "corrected word is outside the systematic code"
    success = reason == "decoded"
    support = tuple(int(i) for i in np.flatnonzero(corrected[:code.n])) if success else None
    return BCHDecode(success, support, positions, tuple(map(int, syndrome)), tuple(map(int, locator)), reason)


def chase_patterns(reliabilities: np.ndarray, depth: int = 6, variant: str = "author"):
    """Perturb the least confident PARITY bits, never enumerate supports.

    The default reproduces the author's bounded partial-sum heap and its
    deterministic tie order: at most depth*(depth+1)//2 patterns. The labelled
    ``expanded`` play variant explores all 2**depth patterns in confidence
    order. A pattern is a tuple of parity positions to flip.
    """
    reliability = np.asarray(reliabilities, dtype=float)
    if reliability.ndim != 1 or not np.all(np.isfinite(reliability)):
        raise ValueError("Reliabilities must be a finite vector")
    if not isinstance(depth, (int, np.integer)) or not 0 <= depth <= 10:
        raise ValueError("Chase depth must be between zero and ten")
    positions = np.argsort(np.abs(reliability), kind="stable")[:depth]
    if variant not in ("author", "expanded"):
        raise ValueError("Chase variant must be 'author' or 'expanded'")
    if variant == "author":
        limit = depth * (depth + 1) // 2
        if limit == 0:
            return []
        # Negative costs let a min-heap discard the least likely candidates.
        # Preserve the source's list order too: it resolves later decode ties.
        heap = [(0.0, [0] * len(positions))]
        for bit, position in enumerate(positions):
            cost, boundary = abs(reliability[position]), heap[0][0]
            additions = []
            for score, mask in heap:
                if len(heap) < limit or score - cost >= boundary:
                    flipped = mask.copy()
                    flipped[bit] = 1
                    additions.append((score - cost, flipped))
            for item in additions:
                heapq.heappush(heap, item)
            if len(heap) > limit:
                heap = heapq.nlargest(limit, heap)[::-1]
        return [tuple(int(positions[i]) for i, bit in enumerate(mask) if bit)
                for _, mask in reversed(heap)]
    patterns = [tuple(int(positions[i]) for i in range(len(positions)) if bits & (1 << i))
                for bits in range(1 << len(positions))]
    return sorted(patterns, key=lambda flips: (sum(abs(reliability[i]) for i in flips), len(flips), flips))


def soft_decode_parity(reliabilities: np.ndarray, code: BCHCode, chase_depth: int | None = None) -> dict:
    """Author bounded Chase selection, before independent bin validation.

    Return hard decoding immediately if it succeeds. Otherwise collect the
    bounded Chase list, deduplicate decoded messages, and choose the decode
    with the fewest corrected codeword bits (stable trial-order tie break).
    """
    depth = 2 * code.t if chase_depth is None else chase_depth
    hard_bits = (np.asarray(reliabilities) < 0).astype(np.uint8)
    decoded = decode_bch_parity(hard_bits, code)
    if decoded.success:
        return {"decoded": decoded, "trials": 1, "flipped_parity": ()}
    candidates, seen, trials = [], set(), 1
    for flips in chase_patterns(reliabilities, depth, variant="author"):
        candidate_bits = hard_bits.copy()
        candidate_bits[list(flips)] ^= 1
        candidate = decode_bch_parity(candidate_bits, code)
        trials += 1
        if candidate.success and candidate.support not in seen:
            seen.add(candidate.support)
            candidates.append((candidate, flips))
    if not candidates:
        return {"decoded": decoded, "trials": trials, "flipped_parity": ()}
    candidate, flips = min(candidates, key=lambda item: len(item[0].corrected_positions))
    return {"decoded": candidate, "trials": trials, "flipped_parity": flips}


@dataclass(frozen=True)
class QueryDesign:
    n: int
    t: int
    b: int
    C: int
    seed: int
    code: BCHCode
    matrices: np.ndarray
    delays: np.ndarray
    masks: np.ndarray
    requested_count: int
    unique_count: int


def shifted_subsample_queries(matrix: np.ndarray, delays: np.ndarray) -> np.ndarray:
    """Return shape (delay, 2**b, n): x = M @ ell + d over GF(2)."""
    matrix, delays = _binary_array(matrix, ndim=2), _binary_array(delays, ndim=2)
    if matrix.shape[0] != delays.shape[1]:
        raise ValueError("Subsampling and delay feature dimensions differ")
    subcube = binary_masks(matrix.shape[1])
    return ((subcube @ matrix.T)[None, :, :] + delays[:, None, :]) % 2


def make_query_design(n: int, t: int, b: int, C: int = 3, seed: int = 0) -> QueryDesign:
    """Independent author-style b design, with a local legacy NumPy RNG.

    Rows are flattened in (hash group, delay, subcube mask) order. Duplicate
    requests are retained and counted; callers may cache physical evaluations.
    """
    code = make_bch_code(n, t)
    if not isinstance(b, (int, np.integer)) or not 1 <= b <= min(n, 12):
        raise ValueError("The bounded sampler requires 1 <= b <= min(n, 12)")
    if not isinstance(C, (int, np.integer)) or C < 1:
        raise ValueError("C must be a positive integer")
    rng = np.random.RandomState(seed)
    matrices = rng.randint(0, 2, size=(C, n, b)).astype(np.uint8)
    delays = code.delays
    masks = np.concatenate([shifted_subsample_queries(M, delays).reshape(-1, n) for M in matrices])
    requested, unique = len(masks), len(np.unique(masks, axis=0))
    for array in (matrices, delays, masks):
        array.flags.writeable = False
    return QueryDesign(n, t, b, C, seed, code, matrices, delays, masks, requested, unique)


def transform_query_values(design: QueryDesign, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.shape != (design.requested_count,) or not np.all(np.isfinite(values)):
        raise ValueError("Supply one finite scalar for every requested mask, in design order")
    return normalized_fwht(values.reshape(design.C, len(design.delays), 1 << design.b))


def bin_index(matrix: np.ndarray, support: tuple[int, ...]) -> int:
    bits = matrix[list(support)].sum(axis=0) % 2
    return sum(int(bit) << (len(bits) - 1 - i) for i, bit in enumerate(bits))


def signed_bin_check(observation: np.ndarray, support: tuple[int, ...], matrix: np.ndarray,
                     delays: np.ndarray, index: int, residual_fraction: float = 0.9) -> dict:
    """A signed least-squares amplitude plus independent hash/residual checks."""
    observation = np.asarray(observation, dtype=float)
    signature = boolean_character(delays, support)
    amplitude = float(signature @ observation / len(signature))
    residual = observation - amplitude * signature
    energy = float(observation @ observation)
    fraction = float(residual @ residual / energy) if energy > 0 else float("inf")
    bin_consistent = bin_index(matrix, support) == index
    return {"accepted": bin_consistent and fraction < residual_fraction,
            "amplitude": amplitude, "residual_fraction": fraction,
            "bin_consistent": bin_consistent, "residual": residual}


def decode_bin(observation: np.ndarray, matrix: np.ndarray, delays: np.ndarray,
               index: int, code: BCHCode, *, chase_depth: int | None = None,
               residual_fraction: float = 0.9, min_amplitude: float = 1e-8,
               chase_variant: str = "author") -> dict:
    """Decode parity, then validate the selected signed coefficient/bin."""
    observation = np.asarray(observation, dtype=float)
    if abs(observation[0]) <= min_amplitude:
        return {"accepted": False, "reason": "zero reference shift", "trials": 0}
    reliability = observation[1:] / observation[0]  # cancels either amplitude sign
    if chase_variant == "author":
        selection = soft_decode_parity(reliability, code, chase_depth)
        decoded = selection.pop("decoded")
        if not decoded.success:
            return {"accepted": False, "reason": decoded.reason, **selection}
        check = signed_bin_check(observation, decoded.support, matrix, delays, index, residual_fraction)
        check["accepted"] = check["accepted"] and abs(check["amplitude"]) > min_amplitude
        return {key: value for key, value in check.items() if key != "residual"} | {
            "support": decoded.support, "corrected_positions": decoded.corrected_positions,
            "reason": "decoded" if check["accepted"] else "bin or residual check rejected", **selection}
    if chase_variant != "expanded":
        raise ValueError("Chase variant must be 'author' or 'expanded'")
    hard_bits = (reliability < 0).astype(np.uint8)
    seen = set()
    depth = 2 * code.t if chase_depth is None else chase_depth
    for trial, flips in enumerate(chase_patterns(reliability, depth, variant="expanded"), start=1):
        candidate_bits = hard_bits.copy()
        candidate_bits[list(flips)] ^= 1
        decoded = decode_bch_parity(candidate_bits, code)
        if not decoded.success or decoded.support in seen:
            continue
        seen.add(decoded.support)
        check = signed_bin_check(observation, decoded.support, matrix, delays, index, residual_fraction)
        if check["accepted"] and abs(check["amplitude"]) > min_amplitude:
            return {key: value for key, value in check.items() if key != "residual"} | {
                "support": decoded.support, "trials": trial, "flipped_parity": flips,
                "corrected_positions": decoded.corrected_positions, "reason": "decoded"}
    return {"accepted": False, "reason": "no valid Chase candidate", "trials": trial}


@dataclass
class RecoveryResult:
    spectrum: Spectrum
    trace: list[dict]
    requested_count: int
    unique_count: int
    initial_bins: np.ndarray
    residual_bins: np.ndarray
    residual_energy: float
    stop_reason: str


def recover_spectrum(design: QueryDesign, values: np.ndarray, *, max_iterations: int = 100,
                     chase_depth: int | None = None, residual_fraction: float = 0.9,
                     min_amplitude: float = 1e-8, bin_energy_floor: float = 1e-9,
                     stagnation_window: int = 15, chase_variant: str = "author") -> RecoveryResult:
    """Factor-to-variable decoding, mean messages, then signed subtraction.

    A sweep reads the same residual state for all factors. Accepted messages
    for a support are averaged, added to that coefficient, and subtracted
    from every connected bin. A previously rejected bin is retried only when
    subtraction changes it. No final regression or oracle call is hidden here.
    """
    if max_iterations < 1 or stagnation_window < 1 or not 0 < residual_fraction <= 1:
        raise ValueError("Invalid iteration, stagnation or residual threshold")
    if min_amplitude < 0 or bin_energy_floor < 0:
        raise ValueError("Numerical floors cannot be negative")
    initial = transform_query_values(design, values)
    residual = initial.copy()
    active = np.ones((design.C, 1 << design.b), dtype=bool)
    spectrum, trace = {}, []
    support_counts = []
    stop_reason = "iteration limit"
    for iteration in range(max_iterations):
        messages, events, rejected, zerotons = {}, [], 0, 0
        energy_before = float(np.sum(residual ** 2))
        for c, index in zip(*np.nonzero(active)):
            observation = residual[c, :, index]
            active[c, index] = False
            if float(observation @ observation) <= bin_energy_floor * len(observation):
                zerotons += 1
                continue
            decoded = decode_bin(observation, design.matrices[c], design.delays, int(index), design.code,
                                 chase_depth=chase_depth, residual_fraction=residual_fraction,
                                 min_amplitude=min_amplitude, chase_variant=chase_variant)
            if decoded["accepted"]:
                support = decoded["support"]
                messages.setdefault(support, []).append(decoded["amplitude"])
                events.append({"group": int(c), "bin": int(index), **decoded})
            else:
                rejected += 1
        updates = []
        for support in sorted(messages):
            amplitude = float(np.mean(messages[support]))
            spectrum[support] = spectrum.get(support, 0.0) + amplitude
            signature = boolean_character(design.delays, support)
            touched = []
            for c, matrix in enumerate(design.matrices):
                index = bin_index(matrix, support)
                residual[c, :, index] -= amplitude * signature
                active[c, index] = True
                touched.append((c, index))
            updates.append({"support": support, "amplitude_delta": amplitude,
                            "message_count": len(messages[support]), "touched_bins": touched})
        trace.append({"iteration": iteration, "energy_before": energy_before,
                      "energy_after": float(np.sum(residual ** 2)), "events": events,
                      "updates": updates, "rejected_bins": rejected, "zero_bins": zerotons,
                      "spectrum": spectrum.copy()})
        support_counts.append(len(spectrum))
        if not messages:
            stop_reason = "no decodable active bins"
            break
        if len(support_counts) > stagnation_window and support_counts[-1] == support_counts[-1-stagnation_window]:
            stop_reason = "support stagnation"
            break
    spectrum = {support: value for support, value in spectrum.items() if abs(value) > min_amplitude}
    return RecoveryResult(spectrum, trace, design.requested_count, design.unique_count,
                          initial, residual, float(np.sum(residual ** 2)), stop_reason)


def sample_and_recover(design: QueryDesign, oracle: Callable[[np.ndarray], np.ndarray], **kwargs) -> RecoveryResult:
    """Request every design row once; oracle owns physical evaluation/cache accounting.

    The result counts logical requests and distinct masks. It makes no claim
    that either is the number of physical model forwards or tokens evaluated.
    """
    values = oracle(design.masks)
    return recover_spectrum(design, values, **kwargs)


def fourier_to_sii(spectrum: Mapping[tuple[int, ...], float],
                   interactions: Iterable[tuple[int, ...]]) -> Spectrum:
    """Convert only requested nonempty sets to the Shapley interaction index.

    I(T)=(-2)**|T| * sum(F(S)/( |S|-|T|+1 )) over S containing T with an
    even size difference. This is the factorial-weighted SII, not a Fourier
    coefficient or the Shapley-Taylor interaction index.
    """
    output = {}
    for interaction in interactions:
        interaction = tuple(sorted(interaction))
        if not interaction or len(set(interaction)) != len(interaction):
            raise ValueError("SII interactions must be nonempty sets of feature positions")
        target = set(interaction)
        coefficient = sum(value / (len(support) - len(interaction) + 1)
                          for support, value in spectrum.items()
                          if target.issubset(support) and (len(support) - len(interaction)) % 2 == 0)
        output[interaction] = float((-2) ** len(interaction) * coefficient)
    return output


def fourier_to_shapley(spectrum: Mapping[tuple[int, ...], float], n: int) -> np.ndarray:
    """Singleton SII; includes the -2 factor in SPEX Appendix C equation 46."""
    singleton = fourier_to_sii(spectrum, [(i,) for i in range(n)])
    return np.asarray([singleton[(i,)] for i in range(n)])
