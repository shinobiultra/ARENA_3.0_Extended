"""Independent exact references for the 16.16 interaction study.

The taught implementation is deliberately not imported here. Complete games use
ascending integer bitmasks. The reference baseline-centres every game before
computing constant-shift-invariant nonempty interactions, then restores the
empty STII/FSII coefficient. Installed shapiq calls are supporting evidence;
the independent derivative, closed-form, and equality-constrained calculations
remain the mathematical acceptance reference.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import inspect
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Iterator, Mapping

import numpy as np
import shapiq
import torch
from shapiq.game_theory.exact import ExactComputer
from shapiq.interaction_values import InteractionValues
from shapiq.utils import powerset


EXPECTED_SHAPIQ_VERSION = "1.5.2"
EXPECTED_SOURCE_SHA256 = {
    "shapiq/game_theory/exact.py": "ba60200fe26a14dadc3c4faebb25eb0174de5bcc25dce6ec34e6088989ab59b5",
    "shapiq/interaction_values.py": "d45229268352b37e77934e394e9308b4bc0ab8b50024b51a8ad77860ca842de5",
    "shapiq/utils/sets.py": "0a83193f8111aa7f50d471939355b19f6e0ce748c0e399a117e791834dbcd5bf",
    "shapiq-1.5.2.dist-info/METADATA": "58588880d8e30bfc81b568aeffda6dd64d0ea22eaea0478884261e9eb44a1d55",
    "shapiq-1.5.2.dist-info/RECORD": "0298377ab532fbe1113df742e872a2043b44c1899d92e332494e6b1b604b668f",
}
EXPECTED_UV_LOCK_SHA256 = "03f77ffac99858cd34a653cd703728b9a868c87686a86cef78523a7afa9a6494"

STRICT_ABSOLUTE_FLOOR = 1.0e-10
STRICT_SCALE_COEFFICIENT = 1.0e-12
PACKAGE_FSII_ABSOLUTE_FLOOR = 1.0e-8
PACKAGE_FSII_SCALE_COEFFICIENT = 5.0e-9
PACKAGE_FSII_EFFICIENCY_SCALE_COEFFICIENT = 3.0e-8
PACKAGE_FSII_BIG_M = 1.0e8
MODEL_KEYS = (
    "clean_0",
    "clean_1",
    "clean_2",
    "shuffled_0",
    "shuffled_1",
    "shuffled_2",
    "ridge",
)
BACKGROUND_VARIANT_KEYS = (
    "nested16",
    "nested128",
    "independent64_a",
    "independent64_b",
)

__all__ = [
    "reference_environment",
    "reference_check",
    "validate_reference_payload",
    "validate_exact_reference_suite",
    "finalize_study_reference",
    "validate_study_reference",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray | torch.Tensor) -> str:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(list(array.shape)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _source_path(obj: Any) -> Path:
    path = inspect.getsourcefile(obj)
    if path is None:
        raise RuntimeError(f"Cannot locate source for {obj!r}.")
    return Path(path).resolve()


def reference_environment() -> dict[str, Any]:
    """Verify the installed shapiq implementation and lock-file identity."""

    if shapiq.__version__ != EXPECTED_SHAPIQ_VERSION:
        raise ValueError(
            f"shapiq version mismatch: expected {EXPECTED_SHAPIQ_VERSION}, "
            f"found {shapiq.__version__}."
        )
    package_root = Path(shapiq.__file__).resolve().parent
    distribution = package_root.parent / f"shapiq-{shapiq.__version__}.dist-info"
    installed = {
        "shapiq/game_theory/exact.py": _source_path(ExactComputer),
        "shapiq/interaction_values.py": _source_path(InteractionValues),
        "shapiq/utils/sets.py": _source_path(powerset),
        "shapiq-1.5.2.dist-info/METADATA": distribution / "METADATA",
        "shapiq-1.5.2.dist-info/RECORD": distribution / "RECORD",
    }
    identities = {}
    for name, expected in EXPECTED_SOURCE_SHA256.items():
        observed = _sha256(installed[name])
        if observed != expected:
            raise ValueError(f"Pinned shapiq source changed: {name}.")
        identities[name] = {"path": str(installed[name]), "sha256": observed}

    root = Path(__file__).resolve().parents[3]
    lock = root / "uv.lock"
    lock_hash = _sha256(lock)
    if lock_hash != EXPECTED_UV_LOCK_SHA256:
        raise ValueError("Pinned uv.lock changed for the 16.16 reference environment.")
    return {
        "shapiq": shapiq.__version__,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "installed_sources": identities,
        "uv_lock": {"path": str(lock), "sha256": lock_hash},
        "adapter_source_sha256": _sha256(Path(__file__).resolve()),
    }


@contextlib.contextmanager
def _preserve_execution_state() -> Iterator[None]:
    """Restore caller RNG/default-dtype state on success and exceptions."""

    python_state = random.getstate()
    numpy_state = copy.deepcopy(np.random.get_state())
    torch_state = torch.random.get_rng_state().clone()
    default_dtype = torch.get_default_dtype()
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        torch.set_default_dtype(default_dtype)


def _positive_int(name: str, value: int, *, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}.")
    return value


def _array(name: str, value: Any, dimensions: int) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu" or value.dtype != torch.float64:
            raise TypeError(f"{name} must be CPU float64.")
        result = value.detach().numpy()
    elif isinstance(value, np.ndarray):
        if value.dtype != np.float64:
            raise TypeError(f"{name} must be float64.")
        result = value
    else:
        raise TypeError(f"{name} must be a torch.Tensor or numpy.ndarray.")
    if result.ndim != dimensions or result.size == 0:
        raise ValueError(f"{name} must be a nonempty {dimensions}-dimensional array.")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values.")
    return np.array(result, dtype=np.float64, order="C", copy=True)


def _relevant_masks(n_players: int, order: int) -> np.ndarray:
    return np.asarray(
        [mask for mask in range(1 << n_players) if mask.bit_count() <= order],
        dtype=np.int64,
    )


def _pair_masks(n_players: int) -> np.ndarray:
    return np.asarray(
        [(1 << left) | (1 << right) for left in range(n_players) for right in range(left + 1, n_players)],
        dtype=np.int64,
    )


def _submasks(mask: int) -> Iterator[int]:
    submask = mask
    while True:
        yield submask
        if submask == 0:
            return
        submask = (submask - 1) & mask


def _mobius(values: np.ndarray, n_players: int) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    for player in range(n_players):
        bit = 1 << player
        for mask in range(1 << n_players):
            if mask & bit:
                result[mask] -= result[mask ^ bit]
    return result


def _derivative(values: np.ndarray, interaction: int, context: int) -> float:
    total = 0.0
    degree = interaction.bit_count()
    for part in _submasks(interaction):
        total += (-1.0) ** (degree - part.bit_count()) * values[context | part]
    return float(total)


def _direct_stii(values: np.ndarray, n_players: int, order: int) -> np.ndarray:
    baseline = float(values[0])
    centered = np.asarray(values, dtype=np.float64) - baseline
    result = np.zeros(1 << n_players, dtype=np.float64)
    result[0] = baseline
    full = (1 << n_players) - 1
    for interaction in range(1, 1 << n_players):
        degree = interaction.bit_count()
        if degree < order:
            result[interaction] = _derivative(centered, interaction, 0)
        elif degree == order:
            complement = full ^ interaction
            for context in _submasks(complement):
                weight = order / (
                    n_players * math.comb(n_players - 1, context.bit_count())
                )
                result[interaction] += weight * _derivative(
                    centered, interaction, context
                )
    return result


def _direct_fsii(values: np.ndarray, n_players: int, order: int) -> np.ndarray:
    baseline = float(values[0])
    centered = np.asarray(values, dtype=np.float64) - baseline
    dividends = _mobius(centered, n_players)
    result = np.zeros(1 << n_players, dtype=np.float64)
    result[0] = baseline
    full = (1 << n_players) - 1
    for interaction in range(1, 1 << n_players):
        size = interaction.bit_count()
        if size > order:
            continue
        correction = 0.0
        for extra in _submasks(full ^ interaction):
            superset = interaction | extra
            superset_size = superset.bit_count()
            if superset_size > order:
                correction += (
                    math.comb(superset_size - 1, order)
                    / math.comb(superset_size + order - 1, order + size)
                    * dividends[superset]
                )
        scale = (
            (-1.0) ** (order - size)
            * size
            / (order + size)
            * math.comb(order, size)
        )
        result[interaction] = dividends[interaction] + scale * correction
    return result


def _direct_sii_pairs(values: np.ndarray, n_players: int) -> np.ndarray:
    centered = np.asarray(values, dtype=np.float64) - float(values[0])
    full = (1 << n_players) - 1
    result = []
    for interaction in _pair_masks(n_players):
        complement = full ^ int(interaction)
        total = 0.0
        for context in _submasks(complement):
            size = context.bit_count()
            weight = (
                math.factorial(size)
                * math.factorial(n_players - 2 - size)
                / math.factorial(n_players - 1)
            )
            total += weight * _derivative(centered, int(interaction), context)
        result.append(total)
    return np.asarray(result, dtype=np.float64)


def _operator(kind: str, n_players: int, order: int) -> np.ndarray:
    """Build the exact linear map from all complete-game basis vectors."""

    columns = []
    for mask in range(1 << n_players):
        basis = np.zeros(1 << n_players, dtype=np.float64)
        basis[mask] = 1.0
        if kind == "STII":
            columns.append(_direct_stii(basis, n_players, order))
        elif kind == "FSII":
            columns.append(_direct_fsii(basis, n_players, order))
        elif kind == "SII":
            columns.append(_direct_sii_pairs(basis, n_players))
        else:
            raise ValueError(f"Unknown independent operator {kind!r}.")
    result = np.asarray(columns, dtype=np.float64).T
    result.setflags(write=False)
    return result


def _faith_system(n_players: int, order: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    terms = _relevant_masks(n_players, order)
    interior = np.arange(1, (1 << n_players) - 1, dtype=np.int64)
    design = np.asarray(
        [[(term & coalition) == term for term in terms] for coalition in interior],
        dtype=np.float64,
    )
    sizes = np.asarray([int(mask).bit_count() for mask in interior], dtype=np.int64)
    weights = np.asarray(
        [
            (n_players - 1)
            / (math.comb(n_players, int(size)) * size * (n_players - size))
            for size in sizes
        ],
        dtype=np.float64,
    )
    weights /= weights.sum()
    gram = design.T @ (weights[:, None] * design)
    constraints = np.zeros((2, len(terms)), dtype=np.float64)
    constraints[0, np.flatnonzero(terms == 0)[0]] = 1.0
    constraints[1] = 1.0
    kkt = np.block(
        [[gram, constraints.T], [constraints, np.zeros((2, 2), dtype=np.float64)]]
    )
    return terms, interior, design.T * weights[None, :], float(np.linalg.cond(kkt))


def _faith_kkt(values: np.ndarray, n_players: int, order: int) -> tuple[np.ndarray, float]:
    terms, interior, xtw, condition = _faith_system(n_players, order)
    design = np.asarray(
        [[(term & coalition) == term for term in terms] for coalition in interior],
        dtype=np.float64,
    )
    sizes = np.asarray([int(mask).bit_count() for mask in interior], dtype=np.int64)
    weights = np.asarray(
        [
            (n_players - 1)
            / (math.comb(n_players, int(size)) * size * (n_players - size))
            for size in sizes
        ],
        dtype=np.float64,
    )
    weights /= weights.sum()
    gram = design.T @ (weights[:, None] * design)
    constraints = np.zeros((2, len(terms)), dtype=np.float64)
    constraints[0, np.flatnonzero(terms == 0)[0]] = 1.0
    constraints[1] = 1.0
    kkt = np.block(
        [[gram, constraints.T], [constraints, np.zeros((2, 2), dtype=np.float64)]]
    )
    centered = values - values[:, :1]
    rhs = np.zeros((len(terms) + 2, len(values)), dtype=np.float64)
    rhs[: len(terms)] = xtw @ centered[:, interior].T
    rhs[len(terms) + 1] = centered[:, -1]
    solved = np.linalg.solve(kkt, rhs)[: len(terms)].T
    result = np.zeros_like(values)
    result[:, terms] = solved
    result[:, 0] = values[:, 0]
    return result, condition


def _package_array(
    item: InteractionValues,
    masks: np.ndarray,
    baseline: float,
    *,
    restore_baseline: bool,
) -> np.ndarray:
    full = np.zeros(1 << item.n_players, dtype=np.float64)
    for term, position in item.interaction_lookup.items():
        mask = sum(1 << int(player) for player in term)
        full[mask] = float(item.values[position])
    # shapiq canonicalizes order-one STII and FSII to index="SV", so the
    # requested family rather than item.index determines intercept restoration.
    if restore_baseline:
        full[0] = baseline
    return full[masks]


def _package_three(
    values: np.ndarray,
    n_players: int,
    order: int,
    term_masks: np.ndarray,
    sii_term_masks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bits = 1 << np.arange(n_players, dtype=np.int64)
    baseline = float(values[0])
    centered = np.asarray(values, dtype=np.float64) - baseline

    def game(coalitions: np.ndarray) -> np.ndarray:
        masks = np.asarray(coalitions, dtype=np.int64) @ bits
        return centered[masks]

    start = time.perf_counter()
    exact = ExactComputer(game=game, n_players=n_players, evaluate_game=True)
    evaluated = time.perf_counter()
    st_start = time.perf_counter()
    stii = exact("STII", order=order)
    fs_start = time.perf_counter()
    fsii = exact("FSII", order=order)
    sii_start = time.perf_counter()
    sii = exact("SII", order=min(2, n_players))
    done = time.perf_counter()
    timing = np.asarray(
        [evaluated - start, fs_start - st_start, sii_start - fs_start, done - sii_start],
        dtype=np.float64,
    )
    return (
        _package_array(stii, term_masks, baseline, restore_baseline=True),
        _package_array(fsii, term_masks, baseline, restore_baseline=True),
        _package_array(sii, sii_term_masks, 0.0, restore_baseline=False),
        timing,
    )


def _package_condition(n_players: int, order: int) -> dict[str, float]:
    terms = list(powerset(range(n_players), max_size=order))
    coalitions = list(powerset(range(n_players)))
    design = np.asarray(
        [[set(term).issubset(coalition) for term in terms] for coalition in coalitions],
        dtype=np.float64,
    )
    weights = np.zeros(1 << n_players, dtype=np.float64)
    for index, coalition in enumerate(coalitions):
        size = len(coalition)
        if size in (0, n_players):
            weights[index] = PACKAGE_FSII_BIG_M
        else:
            weights[index] = 1.0 / (
                (n_players - 1) * math.comb(n_players - 2, size - 1)
            )
    singular = np.linalg.svd(np.sqrt(weights)[:, None] * design, compute_uv=False)
    return {
        "finite_endpoint_big_M": PACKAGE_FSII_BIG_M,
        "weighted_design_condition_number": float(singular[0] / singular[-1]),
        "largest_singular_value": float(singular[0]),
        "smallest_singular_value": float(singular[-1]),
    }


def _summary(error: np.ndarray, scale: np.ndarray) -> dict[str, float]:
    flat = error.reshape(len(error), -1)
    per_game = np.max(np.abs(flat), axis=1)
    return {
        "signed_min": float(flat.min()),
        "signed_max": float(flat.max()),
        "max_abs": float(per_game.max()),
        "rms": float(np.sqrt(np.mean(flat**2))),
        "max_abs_over_centered_game_scale": float(np.max(per_game / scale)),
    }


def _check(
    error: np.ndarray,
    scale: np.ndarray,
    *,
    absolute_floor: float,
    scale_coefficient: float,
) -> dict[str, Any]:
    flat = error.reshape(len(error), -1)
    bound = absolute_floor + scale_coefficient * scale
    passed = np.abs(flat) <= bound[:, None]
    return {
        "pass": bool(np.all(passed)),
        "games": int(len(flat)),
        "coordinates": int(flat.size),
        "passing_games": int(np.all(passed, axis=1).sum()),
        "passing_coordinates": int(passed.sum()),
        "max_error_to_bound_ratio": float(np.max(np.abs(flat) / bound[:, None])),
    }


def _tensor(value: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.array(value, dtype=np.float64, order="C", copy=True))


def _reference_check_impl(
    game_values: np.ndarray | torch.Tensor,
    student_stii: np.ndarray | torch.Tensor,
    student_fsii: np.ndarray | torch.Tensor,
    student_sii_pairs: np.ndarray | torch.Tensor,
    *,
    n_players: int,
    order: int,
    expected_game_count: int | None,
) -> dict[str, Any]:
    n_players = _positive_int("n_players", n_players, minimum=2)
    if n_players > 12:
        raise ValueError("n_players above 12 is outside this exact-enumeration adapter.")
    if type(order) is not int or not 1 <= order <= n_players:
        raise ValueError("order must be an integer from 1 through n_players.")
    values = _array("game_values", game_values, 2)
    games = len(values)
    if values.shape[1] != 1 << n_players:
        raise ValueError("game_values must contain all 2**n_players coalitions.")
    if expected_game_count is not None:
        expected_game_count = _positive_int("expected_game_count", expected_game_count)
        if games != expected_game_count:
            raise ValueError(
                f"Expected {expected_game_count} complete games, found {games}."
            )
    student = {
        "stii": _array("student_stii", student_stii, 2),
        "fsii": _array("student_fsii", student_fsii, 2),
        "sii_pairs": _array("student_sii_pairs", student_sii_pairs, 2),
    }
    full_shape = (games, 1 << n_players)
    pair_shape = (games, math.comb(n_players, 2))
    if student["stii"].shape != full_shape or student["fsii"].shape != full_shape:
        raise ValueError(f"student STII and FSII must have shape {full_shape}.")
    if student["sii_pairs"].shape != pair_shape:
        raise ValueError(f"student SII pairs must have shape {pair_shape}.")

    environment = reference_environment()
    centered = values - values[:, :1]
    scale = np.maximum(1.0, np.max(np.abs(centered), axis=1))
    term_masks = _relevant_masks(n_players, order)
    pair_masks = _pair_masks(n_players)
    sii_term_masks = _relevant_masks(n_players, min(2, n_players))

    independent_start = time.perf_counter()
    operator_start = time.perf_counter()
    st_operator = _operator("STII", n_players, order)
    fs_operator = _operator("FSII", n_players, order)
    sii_operator = _operator("SII", n_players, order)
    operator_seconds = time.perf_counter() - operator_start
    bulk_start = time.perf_counter()
    independent_stii = centered @ st_operator.T
    independent_fsii = centered @ fs_operator.T
    independent_sii = centered @ sii_operator.T
    independent_stii[:, 0] = values[:, 0]
    independent_fsii[:, 0] = values[:, 0]
    bulk_seconds = time.perf_counter() - bulk_start
    kkt_start = time.perf_counter()
    independent_kkt, kkt_condition = _faith_kkt(values, n_players, order)
    kkt_seconds = time.perf_counter() - kkt_start
    independent_seconds = time.perf_counter() - independent_start

    package_stii = np.empty((games, len(term_masks)), dtype=np.float64)
    package_fsii = np.empty_like(package_stii)
    package_sii = np.empty((games, len(sii_term_masks)), dtype=np.float64)
    package_timing = np.empty((games, 4), dtype=np.float64)
    package_start = time.perf_counter()
    for game_index, game in enumerate(values):
        stii, fsii, sii, timing = _package_three(
            game, n_players, order, term_masks, sii_term_masks
        )
        package_stii[game_index] = stii
        package_fsii[game_index] = fsii
        package_sii[game_index] = sii
        package_timing[game_index] = timing
    package_seconds = time.perf_counter() - package_start
    sii_pair_positions = np.asarray(
        [int(np.flatnonzero(sii_term_masks == mask)[0]) for mask in pair_masks],
        dtype=np.int64,
    )

    # NumPy 1.x temporary-elision heuristics can mistake a live local array for
    # an expendable temporary on Python 3.14 (numpy/numpy#28681). Calling the
    # ufunc explicitly preserves the retained coefficient arrays in notebooks.
    errors = {
        "student_stii_minus_independent": np.subtract(student["stii"], independent_stii),
        "student_fsii_minus_independent_closed": np.subtract(student["fsii"], independent_fsii),
        "student_sii_pairs_minus_independent": np.subtract(student["sii_pairs"], independent_sii),
        "independent_kkt_minus_closed_fsii": np.subtract(independent_kkt, independent_fsii),
        "shapiq_stii_minus_independent": np.subtract(package_stii, independent_stii[:, term_masks]),
        "shapiq_fsii_minus_independent": np.subtract(package_fsii, independent_fsii[:, term_masks]),
        "shapiq_sii_pairs_minus_independent": np.subtract(package_sii[:, sii_pair_positions], independent_sii),
    }

    def efficiency(coefficients: np.ndarray) -> np.ndarray:
        return coefficients[:, 1:].sum(axis=1) - (values[:, -1] - values[:, 0])

    package_stii_full = np.zeros_like(values)
    package_fsii_full = np.zeros_like(values)
    package_stii_full[:, term_masks] = package_stii
    package_fsii_full[:, term_masks] = package_fsii
    efficiency_errors = {
        "student_stii": efficiency(student["stii"]),
        "student_fsii": efficiency(student["fsii"]),
        "independent_stii": efficiency(independent_stii),
        "independent_fsii_closed": efficiency(independent_fsii),
        "independent_fsii_kkt": efficiency(independent_kkt),
        "shapiq_stii": efficiency(package_stii_full),
        "shapiq_fsii": efficiency(package_fsii_full),
    }

    strict_names = (
        "student_stii_minus_independent",
        "student_fsii_minus_independent_closed",
        "student_sii_pairs_minus_independent",
        "independent_kkt_minus_closed_fsii",
        "shapiq_stii_minus_independent",
        "shapiq_sii_pairs_minus_independent",
    )
    checks = {
        name: _check(
            errors[name],
            scale,
            absolute_floor=STRICT_ABSOLUTE_FLOOR,
            scale_coefficient=STRICT_SCALE_COEFFICIENT,
        )
        for name in strict_names
    }
    checks["shapiq_fsii_minus_independent"] = _check(
        errors["shapiq_fsii_minus_independent"],
        scale,
        absolute_floor=PACKAGE_FSII_ABSOLUTE_FLOOR,
        scale_coefficient=PACKAGE_FSII_SCALE_COEFFICIENT,
    )
    for name, error in efficiency_errors.items():
        package_fs = name == "shapiq_fsii"
        checks[f"{name}_efficiency"] = _check(
            error[:, None],
            scale,
            absolute_floor=(
                PACKAGE_FSII_ABSOLUTE_FLOOR if package_fs else STRICT_ABSOLUTE_FLOOR
            ),
            scale_coefficient=(
                PACKAGE_FSII_EFFICIENCY_SCALE_COEFFICIENT
                if package_fs
                else STRICT_SCALE_COEFFICIENT
            ),
        )

    summaries = {name: _summary(error, scale) for name, error in errors.items()}
    summaries["efficiency"] = {
        name: _summary(error[:, None], scale) for name, error in efficiency_errors.items()
    }
    passes = {
        "sources_pinned": True,
        "expected_game_count": expected_game_count is None or games == expected_game_count,
        "full_direct_shapiq_instances": True,
        **{name: bool(check["pass"]) for name, check in checks.items()},
    }
    full_coordinates = games * (2 * (1 << n_players) + len(pair_masks))
    relevant_coordinates = games * (2 * len(term_masks) + len(pair_masks))
    raw_package_coordinates = games * (2 * len(term_masks) + len(sii_term_masks))
    counts = {
        "games": games,
        "coalitions_per_game": 1 << n_players,
        "pairs_per_game": len(pair_masks),
        "student_full_axis_coordinates": full_coordinates,
        "independent_full_axis_coordinates": full_coordinates,
        "independent_kkt_coordinates": games * (1 << n_players),
        "direct_shapiq_exact_computers": games,
        "direct_shapiq_index_calls": games * 3,
        "independent_vs_package_relevant_coordinates": relevant_coordinates,
        "raw_package_returned_coordinates": raw_package_coordinates,
        "package_stii_coordinates": games * len(term_masks),
        "package_fsii_coordinates": games * len(term_masks),
        "package_sii_coordinates": games * len(sii_term_masks),
    }
    tolerances = {
        "scale_definition": "max(1, max(abs(game_values - v_empty))) per game",
        "scale_axis": (
            "maximum over the complete coalition axis; one scalar scale per game "
            "broadcast over that game's compared coordinates"
        ),
        "units": "the scalar cooperative-game output units",
        "strict": {
            "absolute_floor": STRICT_ABSOLUTE_FLOOR,
            "scale_coefficient": STRICT_SCALE_COEFFICIENT,
            "rule": "abs(error) <= 1e-10 + 1e-12 * scale",
            "applies_to": (
                "student vs independent STII/FSII/SII",
                "independent KKT vs closed-form FSII",
                "shapiq STII/SII",
                "all exact efficiency residuals except shapiq FSII",
            ),
        },
        "package_FSII_coordinates": {
            "absolute_floor": PACKAGE_FSII_ABSOLUTE_FLOOR,
            "scale_coefficient": PACKAGE_FSII_SCALE_COEFFICIENT,
            "rule": "abs(error) <= 1e-8 + 5e-9 * scale",
            "supersedes": (
                "the earlier blanket 5e-8 * scale proposal; this tighter pre-freeze "
                "floor-plus-scale rule is calibrated before native outcomes"
            ),
        },
        "package_FSII_efficiency": {
            "absolute_floor": PACKAGE_FSII_ABSOLUTE_FLOOR,
            "scale_coefficient": PACKAGE_FSII_EFFICIENCY_SCALE_COEFFICIENT,
            "rule": "abs(residual) <= 1e-8 + 3e-8 * scale",
        },
    }
    result_arrays = {
        "student": {name: _tensor(value) for name, value in student.items()},
        "independent": {
            "stii": _tensor(independent_stii),
            "fsii_closed_form": _tensor(independent_fsii),
            "fsii_kkt": _tensor(independent_kkt),
            "sii_pairs": _tensor(independent_sii),
        },
        "packages": {
            "shapiq": {
                "stii": _tensor(package_stii),
                "fsii": _tensor(package_fsii),
                "sii": _tensor(package_sii),
            }
        },
        "errors": {name: _tensor(value) for name, value in errors.items()},
        "efficiency_errors": {name: _tensor(value) for name, value in efficiency_errors.items()},
    }
    return {
        "status": "PASS" if all(passes.values()) else "FAIL",
        "versions": {"shapiq": EXPECTED_SHAPIQ_VERSION},
        "environment": environment,
        "scope": {
            "n_players": n_players,
            "order": order,
            "games": games,
            "all_games_retained": True,
            "direct_package_call_per_game": True,
            "baseline_centering": (
                "Subtract v_empty before all nonempty independent and package computations; "
                "restore the empty STII/FSII coefficient afterward."
            ),
        },
        "counts": counts,
        "tolerances": tolerances,
        "passes": passes,
        "masks": {
            "coalition": torch.arange(1 << n_players, dtype=torch.int64),
            "package_terms": torch.from_numpy(term_masks.copy()),
            "package_sii_terms": torch.from_numpy(sii_term_masks.copy()),
            "pairs": torch.from_numpy(pair_masks.copy()),
        },
        "inputs": {
            "game_values": _tensor(values),
            "centered_game_scale": _tensor(scale),
        },
        **result_arrays,
        "summaries": summaries,
        "checks": checks,
        "timings": {
            "independent_total_seconds": independent_seconds,
            "operator_build_seconds": operator_seconds,
            "operator_bulk_seconds": bulk_seconds,
            "independent_kkt_seconds": kkt_seconds,
            "direct_shapiq_total_seconds": package_seconds,
            "direct_shapiq_seconds_per_game": package_seconds / games,
            "direct_shapiq_per_game_p50": float(np.quantile(package_timing.sum(1), 0.5)),
            "direct_shapiq_per_game_p95": float(np.quantile(package_timing.sum(1), 0.95)),
            "direct_shapiq_mean_seconds": {
                "evaluate_game": float(package_timing[:, 0].mean()),
                "STII": float(package_timing[:, 1].mean()),
                "FSII": float(package_timing[:, 2].mean()),
                "SII": float(package_timing[:, 3].mean()),
            },
        },
        "conditioning": {
            "independent_exact_faith_kkt_condition_number": kkt_condition,
            "shapiq_finite_endpoint_weighted_design": _package_condition(
                n_players, order
            ),
        },
        "warnings": (
            "shapiq FSII uses a finite 1e8 endpoint penalty; its package tolerance and "
            "efficiency diagnostic do not weaken the exact closed-form/KKT gate.",
            "SII is retained as its pair interaction vector and is not an order-two "
            "efficient decomposition.",
            "The independent operators use 2**n complete-game basis formula calls; "
            "the shapiq counts are direct per-game calls, never package-basis calls.",
        ),
        "identity_sha256": {
            "game_values": _array_sha256(values),
            "student_stii": _array_sha256(student["stii"]),
            "student_fsii": _array_sha256(student["fsii"]),
            "student_sii_pairs": _array_sha256(student["sii_pairs"]),
            "independent_stii": _array_sha256(independent_stii),
            "independent_fsii_closed_form": _array_sha256(independent_fsii),
            "independent_fsii_kkt": _array_sha256(independent_kkt),
            "independent_sii_pairs": _array_sha256(independent_sii),
            "shapiq_stii": _array_sha256(package_stii),
            "shapiq_fsii": _array_sha256(package_fsii),
            "shapiq_sii": _array_sha256(package_sii),
        },
        "operator": {
            "construction": (
                "Each column is the named independent exact formula evaluated on one "
                "complete-game basis vector; linearity gives exhaustive bulk evaluation."
            ),
            "basis_formula_calls": {
                "STII": 1 << n_players,
                "FSII": 1 << n_players,
                "SII": 1 << n_players,
            },
            "direct_formula_calibration_is_test_evidence": True,
            "operator_sha256": {
                "STII": _array_sha256(st_operator),
                "FSII": _array_sha256(fs_operator),
                "SII": _array_sha256(sii_operator),
            },
        },
    }


def reference_check(
    game_values: np.ndarray | torch.Tensor,
    student_stii: np.ndarray | torch.Tensor,
    student_fsii: np.ndarray | torch.Tensor,
    student_sii_pairs: np.ndarray | torch.Tensor,
    *,
    n_players: int,
    order: int,
    expected_game_count: int | None = None,
) -> dict[str, Any]:
    """Validate every supplied complete game with independent and direct package references."""

    with _preserve_execution_state():
        return _reference_check_impl(
            game_values,
            student_stii,
            student_fsii,
            student_sii_pairs,
            n_players=n_players,
            order=order,
            expected_game_count=expected_game_count,
        )


def validate_reference_payload(
    payload: Mapping[str, Any], *, expected_game_count: int | None = None
) -> dict[str, Any]:
    """Recompute every stored difference, identity, count, and numerical bound."""

    with _preserve_execution_state():
        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a reference-result mapping.")
        scope = payload.get("scope")
        if not isinstance(scope, Mapping):
            raise TypeError("payload.scope is missing.")
        n_players = _positive_int("scope.n_players", scope.get("n_players"), minimum=2)
        order = scope.get("order")
        if type(order) is not int or not 1 <= order <= n_players:
            raise ValueError("payload scope has an invalid order.")
        inputs = payload.get("inputs")
        student = payload.get("student")
        independent = payload.get("independent")
        packages = payload.get("packages")
        stored_errors = payload.get("errors")
        stored_efficiency = payload.get("efficiency_errors")
        masks = payload.get("masks")
        if not all(
            isinstance(item, Mapping)
            for item in (inputs, student, independent, packages, stored_errors, stored_efficiency, masks)
        ):
            raise TypeError("Reference payload is missing retained numerical mappings.")

        values = _array("inputs.game_values", inputs["game_values"], 2)
        games = len(values)
        if expected_game_count is not None and games != expected_game_count:
            raise ValueError(f"Expected {expected_game_count} retained games, found {games}.")
        scale = _array("inputs.centered_game_scale", inputs["centered_game_scale"], 1)
        expected_scale = np.maximum(1.0, np.max(np.abs(values - values[:, :1]), axis=1))
        if scale.shape != (games,) or not np.array_equal(scale, expected_scale):
            raise AssertionError("Stored centered-game scales do not match game values.")

        term_masks = _relevant_masks(n_players, order)
        pair_masks = _pair_masks(n_players)
        sii_term_masks = _relevant_masks(n_players, min(2, n_players))
        expected_masks = {
            "coalition": np.arange(1 << n_players, dtype=np.int64),
            "package_terms": term_masks,
            "package_sii_terms": sii_term_masks,
            "pairs": pair_masks,
        }
        for name, expected in expected_masks.items():
            value = masks.get(name)
            if not isinstance(value, torch.Tensor) or value.dtype != torch.int64:
                raise TypeError(f"masks.{name} must be an int64 tensor.")
            if not np.array_equal(value.cpu().numpy(), expected):
                raise AssertionError(f"Stored {name} masks changed.")

        retained = {
            "student_stii": _array("student.stii", student["stii"], 2),
            "student_fsii": _array("student.fsii", student["fsii"], 2),
            "student_sii": _array("student.sii_pairs", student["sii_pairs"], 2),
            "independent_stii": _array("independent.stii", independent["stii"], 2),
            "independent_fsii": _array(
                "independent.fsii_closed_form", independent["fsii_closed_form"], 2
            ),
            "independent_kkt": _array("independent.fsii_kkt", independent["fsii_kkt"], 2),
            "independent_sii": _array("independent.sii_pairs", independent["sii_pairs"], 2),
        }
        shapiq_result = packages.get("shapiq")
        if not isinstance(shapiq_result, Mapping):
            raise TypeError("packages.shapiq is missing.")
        retained.update(
            package_stii=_array("packages.shapiq.stii", shapiq_result["stii"], 2),
            package_fsii=_array("packages.shapiq.fsii", shapiq_result["fsii"], 2),
            package_sii=_array("packages.shapiq.sii", shapiq_result["sii"], 2),
        )
        expected_shapes = {
            "student_stii": (games, 1 << n_players),
            "student_fsii": (games, 1 << n_players),
            "student_sii": (games, len(pair_masks)),
            "independent_stii": (games, 1 << n_players),
            "independent_fsii": (games, 1 << n_players),
            "independent_kkt": (games, 1 << n_players),
            "independent_sii": (games, len(pair_masks)),
            "package_stii": (games, len(term_masks)),
            "package_fsii": (games, len(term_masks)),
            "package_sii": (games, len(sii_term_masks)),
        }
        for name, shape in expected_shapes.items():
            if retained[name].shape != shape:
                raise AssertionError(f"Retained {name} has shape {retained[name].shape}, expected {shape}.")

        sii_pair_positions = np.asarray(
            [int(np.flatnonzero(sii_term_masks == mask)[0]) for mask in pair_masks],
            dtype=np.int64,
        )
        errors = {
            "student_stii_minus_independent": np.subtract(retained["student_stii"], retained["independent_stii"]),
            "student_fsii_minus_independent_closed": np.subtract(retained["student_fsii"], retained["independent_fsii"]),
            "student_sii_pairs_minus_independent": np.subtract(retained["student_sii"], retained["independent_sii"]),
            "independent_kkt_minus_closed_fsii": np.subtract(retained["independent_kkt"], retained["independent_fsii"]),
            "shapiq_stii_minus_independent": np.subtract(retained["package_stii"], retained["independent_stii"][:, term_masks]),
            "shapiq_fsii_minus_independent": np.subtract(retained["package_fsii"], retained["independent_fsii"][:, term_masks]),
            "shapiq_sii_pairs_minus_independent": np.subtract(retained["package_sii"][:, sii_pair_positions], retained["independent_sii"]),
        }
        for name, expected in errors.items():
            observed = _array(f"errors.{name}", stored_errors[name], 2)
            if not np.array_equal(observed, expected):
                raise AssertionError(f"Stored reference error changed: {name}.")

        def efficiency(coefficients: np.ndarray) -> np.ndarray:
            return coefficients[:, 1:].sum(axis=1) - (values[:, -1] - values[:, 0])

        package_stii_full = np.zeros_like(values)
        package_fsii_full = np.zeros_like(values)
        package_stii_full[:, term_masks] = retained["package_stii"]
        package_fsii_full[:, term_masks] = retained["package_fsii"]
        efficiency_errors = {
            "student_stii": efficiency(retained["student_stii"]),
            "student_fsii": efficiency(retained["student_fsii"]),
            "independent_stii": efficiency(retained["independent_stii"]),
            "independent_fsii_closed": efficiency(retained["independent_fsii"]),
            "independent_fsii_kkt": efficiency(retained["independent_kkt"]),
            "shapiq_stii": efficiency(package_stii_full),
            "shapiq_fsii": efficiency(package_fsii_full),
        }
        for name, expected in efficiency_errors.items():
            observed = _array(f"efficiency_errors.{name}", stored_efficiency[name], 1)
            if not np.array_equal(observed, expected):
                raise AssertionError(f"Stored efficiency residual changed: {name}.")

        strict_names = (
            "student_stii_minus_independent",
            "student_fsii_minus_independent_closed",
            "student_sii_pairs_minus_independent",
            "independent_kkt_minus_closed_fsii",
            "shapiq_stii_minus_independent",
            "shapiq_sii_pairs_minus_independent",
        )
        recomputed_checks = {
            name: _check(
                errors[name],
                scale,
                absolute_floor=STRICT_ABSOLUTE_FLOOR,
                scale_coefficient=STRICT_SCALE_COEFFICIENT,
            )
            for name in strict_names
        }
        recomputed_checks["shapiq_fsii_minus_independent"] = _check(
            errors["shapiq_fsii_minus_independent"],
            scale,
            absolute_floor=PACKAGE_FSII_ABSOLUTE_FLOOR,
            scale_coefficient=PACKAGE_FSII_SCALE_COEFFICIENT,
        )
        for name, error in efficiency_errors.items():
            package_fs = name == "shapiq_fsii"
            recomputed_checks[f"{name}_efficiency"] = _check(
                error[:, None],
                scale,
                absolute_floor=(PACKAGE_FSII_ABSOLUTE_FLOOR if package_fs else STRICT_ABSOLUTE_FLOOR),
                scale_coefficient=(
                    PACKAGE_FSII_EFFICIENCY_SCALE_COEFFICIENT
                    if package_fs
                    else STRICT_SCALE_COEFFICIENT
                ),
            )
        if not all(item["pass"] for item in recomputed_checks.values()):
            raise AssertionError("A recomputed reference numerical bound failed.")
        if payload.get("checks") != recomputed_checks:
            raise AssertionError("Stored reference checks differ from recomputed checks.")

        expected_counts = {
            "games": games,
            "coalitions_per_game": 1 << n_players,
            "pairs_per_game": len(pair_masks),
            "student_full_axis_coordinates": games * (2 * (1 << n_players) + len(pair_masks)),
            "independent_full_axis_coordinates": games * (2 * (1 << n_players) + len(pair_masks)),
            "independent_kkt_coordinates": games * (1 << n_players),
            "direct_shapiq_exact_computers": games,
            "direct_shapiq_index_calls": games * 3,
            "independent_vs_package_relevant_coordinates": games * (2 * len(term_masks) + len(pair_masks)),
            "raw_package_returned_coordinates": games * (2 * len(term_masks) + len(sii_term_masks)),
            "package_stii_coordinates": games * len(term_masks),
            "package_fsii_coordinates": games * len(term_masks),
            "package_sii_coordinates": games * len(sii_term_masks),
        }
        counts = payload.get("counts")
        if not isinstance(counts, Mapping):
            raise TypeError("payload.counts is missing.")
        for name, expected in expected_counts.items():
            if counts.get(name) != expected:
                raise AssertionError(f"Stored reference count changed: {name}.")

        identities = payload.get("identity_sha256")
        identity_arrays = {
            "game_values": values,
            "student_stii": retained["student_stii"],
            "student_fsii": retained["student_fsii"],
            "student_sii_pairs": retained["student_sii"],
            "independent_stii": retained["independent_stii"],
            "independent_fsii_closed_form": retained["independent_fsii"],
            "independent_fsii_kkt": retained["independent_kkt"],
            "independent_sii_pairs": retained["independent_sii"],
            "shapiq_stii": retained["package_stii"],
            "shapiq_fsii": retained["package_fsii"],
            "shapiq_sii": retained["package_sii"],
        }
        if not isinstance(identities, Mapping):
            raise TypeError("payload.identity_sha256 is missing.")
        for name, value in identity_arrays.items():
            if identities.get(name) != _array_sha256(value):
                raise AssertionError(f"Stored reference identity changed: {name}.")

        current_environment = reference_environment()
        stored_environment = payload.get("environment")
        if not isinstance(stored_environment, Mapping):
            raise TypeError("payload.environment is missing.")
        if stored_environment.get("installed_sources") != current_environment["installed_sources"]:
            raise AssertionError("Stored package source identity changed.")
        if stored_environment.get("uv_lock") != current_environment["uv_lock"]:
            raise AssertionError("Stored uv.lock identity changed.")
        if payload.get("status") != "PASS":
            raise AssertionError("Stored reference status is not PASS.")
        checked_coordinates = sum(value.size for value in errors.values()) + sum(
            value.size for value in efficiency_errors.values()
        )
        return {
            "status": "PASS",
            "games": games,
            "checked_coordinates": int(checked_coordinates),
            "student_full_axis_coordinates": expected_counts[
                "student_full_axis_coordinates"
            ],
            "independent_vs_package_relevant_coordinates": expected_counts[
                "independent_vs_package_relevant_coordinates"
            ],
            "raw_package_returned_coordinates": expected_counts[
                "raw_package_returned_coordinates"
            ],
            "raw_identity_arrays": len(identity_arrays),
            "counts_recomputed": len(expected_counts),
            "bounds_recomputed": len(recomputed_checks),
            "sources_rehashed": len(EXPECTED_SOURCE_SHA256) + 1,
        }


def _study_block(block: Mapping[str, Any], *, model_axis: bool) -> tuple[torch.Tensor, ...]:
    required = ("coalition_values", "stii_coefficients", "fsii_coefficients", "sii_pairs")
    if not isinstance(block, Mapping) or any(name not in block for name in required):
        raise TypeError(f"Study block must contain {required}.")
    arrays = []
    for name in required:
        value = block[name]
        if not isinstance(value, torch.Tensor) or value.device.type != "cpu" or value.dtype != torch.float64:
            raise TypeError(f"Study {name} must be a CPU float64 tensor.")
        expected_ndim = 3 if model_axis else 2
        if value.ndim != expected_ndim:
            raise ValueError(f"Study {name} must have {expected_ndim} dimensions.")
        arrays.append(value.reshape(-1, value.shape[-1]))
    return tuple(arrays)


def _validate_exact_suite(prepared: Mapping[str, Any]) -> dict[str, Any]:
    exact = prepared.get("exact_games")
    if not isinstance(exact, Mapping):
        raise TypeError("prepared.exact_games is missing.")
    fixed = exact.get("fixed")
    random_games = exact.get("random")
    if not isinstance(fixed, Mapping) or not isinstance(random_games, Mapping):
        raise TypeError("Exact reference suite must contain fixed and random games.")
    validations = []
    fixed_reference = fixed.get("reference")
    fixed_validation = validate_reference_payload(
        fixed_reference, expected_game_count=8
    )
    if fixed_reference.get("scope", {}).get("n_players") != 4 or fixed_reference.get(
        "scope", {}
    ).get("order") != 2:
        raise AssertionError("Fixed exact references must use n=4 and order=2.")
    validations.append(fixed_validation)
    random_case_count = 0
    boundary_count = 0
    for n_players in range(3, 8):
        item = random_games.get(n_players)
        if not isinstance(item, Mapping):
            raise TypeError(f"Random exact reference n={n_players} is missing.")
        order_blocks = item.get("order_blocks")
        if not isinstance(order_blocks, Mapping) or tuple(order_blocks) != (1, 2, 3):
            raise ValueError(f"Random exact reference n={n_players} must retain orders 1,2,3.")
        for order in (1, 2, 3):
            block = order_blocks[order]
            if not isinstance(block, Mapping):
                raise TypeError("A random exact order block is invalid.")
            reference = block.get("reference")
            validation = validate_reference_payload(reference, expected_game_count=3)
            if reference.get("scope", {}).get("n_players") != n_players or reference.get(
                "scope", {}
            ).get("order") != order:
                raise AssertionError("A random exact reference has the wrong n/order scope.")
            validations.append(validation)
            random_case_count += 3
        boundary = item.get("boundary_order_n")
        if (
            not isinstance(boundary, Mapping)
            or boundary.get("replicate") != 0
            or boundary.get("order") != n_players
        ):
            raise ValueError(
                f"Random exact n={n_players} must retain a separately labeled replicate-0 order=n boundary."
            )
        reference = boundary.get("reference")
        validation = validate_reference_payload(reference, expected_game_count=1)
        if reference.get("scope", {}).get("n_players") != n_players or reference.get(
            "scope", {}
        ).get("order") != n_players:
            raise AssertionError("A full-order boundary reference has the wrong n/order scope.")
        validations.append(validation)
        boundary_count += 1
    return {
        "status": "PASS",
        "fixed_unique_games": 8,
        "random_unique_games": 15,
        "random_game_order_cases": random_case_count,
        "random_STII_plus_FSII_case_calls": random_case_count * 2,
        "boundary_order_n_cases": boundary_count,
        "boundary_replicate": 0,
        "boundary_n_values": (3, 4, 5, 6, 7),
        "boundary_note": (
            "Five separately labeled replicate-0 order=n checks; n=3/order=3 "
            "intentionally duplicates one of the 45 grid cases as a boundary assertion."
        ),
        "reference_payloads_revalidated": len(validations),
        "direct_shapiq_exact_computers": sum(item["games"] for item in validations),
        "direct_shapiq_index_calls": sum(item["games"] * 3 for item in validations),
        "student_full_axis_coordinates": sum(
            item["student_full_axis_coordinates"] for item in validations
        ),
        "independent_vs_package_relevant_coordinates": sum(
            item["independent_vs_package_relevant_coordinates"]
            for item in validations
        ),
        "raw_package_returned_coordinates": sum(
            item["raw_package_returned_coordinates"] for item in validations
        ),
        "payload_validations": tuple(validations),
    }


def validate_exact_reference_suite(prepared: Mapping[str, Any]) -> dict[str, Any]:
    """Revalidate the retained 8 fixed, 45 random-order, and 5 boundary cases."""

    with _preserve_execution_state():
        if not isinstance(prepared, Mapping) or prepared.get("phase") != "prepared_before_audit":
            raise ValueError("prepared must be the sealed pre-audit study payload.")
        return _validate_exact_suite(prepared)


def finalize_study_reference(
    prepared: Mapping[str, Any], empirical_reference: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate and augment an existing all-2,992 empirical result without package calls."""

    with _preserve_execution_state():
        payload_validation = validate_reference_payload(
            empirical_reference, expected_game_count=2_992
        )
        exact_validation = validate_exact_reference_suite(prepared)
        result = dict(empirical_reference)
        result["counts"] = dict(empirical_reference["counts"])
        result["exact_suite_validation"] = exact_validation
        result["payload_validation"] = payload_validation
        result["counts"].update(
            {
                "primary_games": 1_904,
                "nonprimary_clean0_games": 1_088,
                "models": 7,
                "rows_per_model": 272,
                "background_variants": 4,
                "fixed_unique_exact_games": exact_validation["fixed_unique_games"],
                "random_unique_exact_games": exact_validation["random_unique_games"],
                "random_game_order_cases": exact_validation[
                    "random_game_order_cases"
                ],
                "random_STII_plus_FSII_case_calls": exact_validation[
                    "random_STII_plus_FSII_case_calls"
                ],
                "boundary_order_n_cases": exact_validation[
                    "boundary_order_n_cases"
                ],
                "exact_direct_shapiq_exact_computers": exact_validation[
                    "direct_shapiq_exact_computers"
                ],
                "exact_direct_shapiq_index_calls": exact_validation[
                    "direct_shapiq_index_calls"
                ],
                "all_reference_direct_shapiq_exact_computers": (
                    2_992 + exact_validation["direct_shapiq_exact_computers"]
                ),
                "all_reference_direct_shapiq_index_calls": (
                    2_992 * 3 + exact_validation["direct_shapiq_index_calls"]
                ),
                "all_reference_student_full_axis_coordinates": (
                    result["counts"]["student_full_axis_coordinates"]
                    + exact_validation["student_full_axis_coordinates"]
                ),
                "all_reference_independent_vs_package_relevant_coordinates": (
                    result["counts"]["independent_vs_package_relevant_coordinates"]
                    + exact_validation[
                        "independent_vs_package_relevant_coordinates"
                    ]
                ),
                "all_reference_raw_package_returned_coordinates": (
                    result["counts"]["raw_package_returned_coordinates"]
                    + exact_validation["raw_package_returned_coordinates"]
                ),
            }
        )
        if (
            result.get("status") != "PASS"
            or result["counts"]["all_reference_direct_shapiq_exact_computers"]
            != 3_050
            or result["counts"]["all_reference_direct_shapiq_index_calls"] != 9_150
        ):
            raise AssertionError("The complete exact plus empirical reference scope is invalid.")
        return result


def validate_study_reference(
    prepared: Mapping[str, Any], evaluated: Mapping[str, Any]
) -> dict[str, Any]:
    """Assemble the frozen 2,992-game branch order and run one exhaustive check."""

    with _preserve_execution_state():
        if not isinstance(prepared, Mapping) or prepared.get("phase") != "prepared_before_audit":
            raise ValueError("prepared must be the sealed pre-audit study payload.")
        if not isinstance(evaluated, Mapping) or evaluated.get("phase") != "evaluated_after_audit":
            raise ValueError("evaluated must be the completed post-audit study payload.")
        if evaluated.get("audit_used_for_selection") is not False:
            raise ValueError("The reference cannot accept audit-selected study choices.")
        all_rows = evaluated.get("all_rows")
        if not isinstance(all_rows, Mapping):
            raise TypeError("evaluated.all_rows is missing.")
        primary = all_rows.get("primary64")
        variants = all_rows.get("background_variants")
        if not isinstance(primary, Mapping) or not isinstance(variants, Mapping):
            raise TypeError("evaluated.all_rows must contain primary64 and background_variants.")
        if tuple(primary.get("model_keys", ())) != MODEL_KEYS:
            raise ValueError("Primary reference block changed seven-model order.")
        if tuple(variants) != ("primary64", *BACKGROUND_VARIANT_KEYS):
            raise ValueError("Background reference block changed frozen order.")

        blocks = [("primary64", MODEL_KEYS, _study_block(primary, model_axis=True))]
        for name in BACKGROUND_VARIANT_KEYS:
            blocks.append((name, ("clean_0",), _study_block(variants[name], model_axis=False)))
        starts = []
        pieces = [[], [], [], []]
        cursor = 0
        for name, model_keys, arrays in blocks:
            rows = len(arrays[0])
            if any(len(array) != rows for array in arrays[1:]):
                raise ValueError(f"Reference branch {name} has inconsistent game counts.")
            starts.append(
                {
                    "name": name,
                    "model_keys": model_keys,
                    "start": cursor,
                    "stop": cursor + rows,
                    "games": rows,
                }
            )
            cursor += rows
            for index, array in enumerate(arrays):
                pieces[index].append(array)
        joined = [torch.cat(part, dim=0) for part in pieces]
        result = reference_check(
            joined[0],
            joined[1],
            joined[2],
            joined[3],
            n_players=8,
            order=2,
            expected_game_count=2_992,
        )
        result = finalize_study_reference(prepared, result)
        result["branches"] = tuple(starts)
        if result["counts"]["games"] != cursor or result["status"] != "PASS":
            raise AssertionError("The exhaustive 16.16 study reference did not pass.")
        return result
