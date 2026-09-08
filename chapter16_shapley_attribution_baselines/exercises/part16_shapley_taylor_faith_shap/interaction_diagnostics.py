"""Learner-visible diagnostics for exact interaction allocations.

The analytic diagnostics call the functions being taught. They do not carry a
second implementation of Shapley-Taylor or Faith-Shap. Small exact fixtures
provide semantic oracles, while the SII comparator is evaluated from its
closed Möbius allocation because SII was taught in section 16.3.

The native aggregate inspects only explicit, source-bound arrays produced by
the study after protocol freeze. It does not fit a model, rebuild a game,
recompute an interaction index, or select an empirical result.
"""

from __future__ import annotations

import copy
import itertools
import math
from collections.abc import Callable, Mapping
from typing import Any

import torch
from torch import Tensor, nn


GAME_NAMES = (
    "constant",
    "additive",
    "positive_pair",
    "negative_pair",
    "pure_triple",
    "pure_four",
    "mixed_degree2",
    "mixed_high",
)
PAIR_INDICES = tuple(itertools.combinations(range(4), 2))
NATIVE_MODEL_KEYS = (
    "clean_0",
    "clean_1",
    "clean_2",
    "shuffled_0",
    "shuffled_1",
    "shuffled_2",
    "ridge",
)
NATIVE_METHOD_KEYS = ("sii", "stii", "fsii")
NATIVE_BACKGROUND_KEYS = (
    "primary64",
    "nested16",
    "nested128",
    "independent64_a",
    "independent64_b",
)


def _callable(name: str, value: Callable[..., Any]) -> Callable[..., Any]:
    if not callable(value):
        raise TypeError(f"{name} must be callable.")
    return value


def _diagnostic(value: dict[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return exact_game_fixture()
    if not isinstance(value, dict) or value.get("evidence_kind") != "analytic_exact":
        raise ValueError("diagnostic must come from exact_game_fixture or a diagnose_* stage.")
    if value.get("n_players") != 4 or value.get("order") != 2:
        raise ValueError("The exact teaching diagnostic is fixed at four players and order two.")
    return copy.deepcopy(value)


def _zeta(mobius: Tensor, n_players: int) -> Tensor:
    values = mobius.clone()
    masks = torch.arange(2**n_players, dtype=torch.int64)
    for player in range(n_players):
        present = masks.bitwise_and(1 << player).ne(0)
        values[..., present] += values[..., masks[present].bitwise_xor(1 << player)]
    return values


def _ordinary_shapley_from_mobius(mobius: Tensor, n_players: int) -> Tensor:
    result = torch.zeros((*mobius.shape[:-1], n_players), dtype=torch.float64)
    for player in range(n_players):
        for dividend_mask in range(1, 2**n_players):
            if dividend_mask & (1 << player):
                result[..., player] += mobius[..., dividend_mask] / dividend_mask.bit_count()
    return result


def _pairwise_sii_from_mobius(mobius: Tensor, n_players: int) -> Tensor:
    """Return pair SII in lexicographic pair order.

    A Möbius dividend on U contributes ``a(U)/(len(U)-1)`` to every pair
    contained in U. This is the exact SII formula specialized to pairs.
    """

    pairs = tuple(itertools.combinations(range(n_players), 2))
    result = torch.zeros((*mobius.shape[:-1], len(pairs)), dtype=torch.float64)
    for pair_index, (first, second) in enumerate(pairs):
        pair_mask = (1 << first) | (1 << second)
        for dividend_mask in range(2**n_players):
            if dividend_mask & pair_mask == pair_mask:
                result[..., pair_index] += mobius[..., dividend_mask] / (
                    dividend_mask.bit_count() - 1
                )
    return result


def _fixture_mobius() -> Tensor:
    mobius = torch.zeros((len(GAME_NAMES), 16), dtype=torch.float64)

    mobius[0, 0] = 2.0

    mobius[1, 0] = -0.5
    mobius[1, [1, 2, 4, 8]] = torch.tensor(
        [1.0, -2.0, 1.5, 0.25], dtype=torch.float64
    )

    mobius[2, 0b0011] = 1.0
    mobius[3, 0b1010] = -0.75
    mobius[4, 0b0111] = 1.0
    mobius[5, 0b1111] = 1.0

    mobius[6, 0] = 1 / 3
    mobius[6, [1, 2, 4, 8]] = torch.tensor(
        [0.5, -0.25, 0.75, -1.0], dtype=torch.float64
    )
    for mask, value in {
        0b0011: 2 / 3,
        0b1001: -0.5,
        0b0110: 0.2,
        0b1100: 0.375,
    }.items():
        mobius[6, mask] = value

    mobius[7, 0] = -0.2
    mobius[7, 0b0001] = 1 / 3
    mobius[7, 0b0110] = -0.4
    mobius[7, 0b1101] = 0.75
    mobius[7, 0b1111] = -0.5
    return mobius


def exact_game_fixture() -> dict[str, Any]:
    """Return the declared analytic games and exact SII comparators."""

    mobius = _fixture_mobius()
    games = _zeta(mobius, 4)
    return {
        "evidence_kind": "analytic_exact",
        "label": "exact teaching games",
        "n_players": 4,
        "order": 2,
        "feature_names": ("A", "B", "C", "D"),
        "game_names": GAME_NAMES,
        "game_values": games,
        "expected_mobius": mobius,
        "ordinary_shapley": _ordinary_shapley_from_mobius(mobius, 4),
        "sii_pairs": _pairwise_sii_from_mobius(mobius, 4),
        "pair_indices": PAIR_INDICES,
        "single_masks": torch.tensor([1, 2, 4, 8], dtype=torch.int64),
        "pair_masks": torch.tensor(
            [(1 << first) | (1 << second) for first, second in PAIR_INDICES],
            dtype=torch.int64,
        ),
        "exact_fraction_labels": {
            "pure_triple": {
                "stii_singleton": "0",
                "stii_contained_pair": "1/3",
                "fsii_singleton": "-1/6",
                "fsii_contained_pair": "1/2",
                "shapley_singleton": "1/3",
                "sii_contained_pair": "1/2",
            },
            "pure_four": {
                "stii_singleton": "0",
                "stii_pair": "1/6",
                "fsii_singleton": "-1/5",
                "fsii_pair": "3/10",
                "shapley_singleton": "1/4",
                "sii_pair": "1/3",
            },
        },
    }


def diagnose_mobius_games(
    mobius_transform_fn: Callable[..., Tensor],
    diagnostic: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the learner Möbius transform on all eight exact fixtures."""

    _callable("mobius_transform_fn", mobius_transform_fn)
    result = _diagnostic(diagnostic)
    observed = mobius_transform_fn(result["game_values"], n_players=4)
    torch.testing.assert_close(observed, result["expected_mobius"], rtol=0, atol=1e-12)
    result["mobius"] = observed
    result["mobius_max_abs_error"] = float(
        (observed - result["expected_mobius"]).abs().max().item()
    )
    result["stages"] = tuple(result.get("stages", ())) + ("mobius",)
    return result


def diagnose_shapley_taylor(
    diagnostic: dict[str, Any],
    shapley_taylor_fn: Callable[..., Tensor],
) -> dict[str, Any]:
    """Add student Shapley-Taylor allocations and exact fraction checks."""

    _callable("shapley_taylor_fn", shapley_taylor_fn)
    result = _diagnostic(diagnostic)
    if "mobius" not in result:
        raise ValueError("Run diagnose_mobius_games before Shapley-Taylor diagnostics.")
    coefficients = shapley_taylor_fn(result["mobius"], n_players=4, order=2)
    if coefficients.shape != result["game_values"].shape:
        raise AssertionError("Shapley-Taylor must return one coefficient per mask.")
    pure_triple = GAME_NAMES.index("pure_triple")
    pure_four = GAME_NAMES.index("pure_four")
    singles, pairs = result["single_masks"], result["pair_masks"]
    torch.testing.assert_close(
        coefficients[pure_four, singles], torch.zeros(4, dtype=torch.float64), rtol=0, atol=1e-12
    )
    torch.testing.assert_close(
        coefficients[pure_four, pairs], torch.full((6,), 1 / 6, dtype=torch.float64), rtol=0, atol=1e-12
    )
    torch.testing.assert_close(
        coefficients[pure_triple, singles], torch.zeros(4, dtype=torch.float64), rtol=0, atol=1e-12
    )
    contained = torch.tensor([0, 1, 3], dtype=torch.int64)
    torch.testing.assert_close(
        coefficients[pure_triple, pairs[contained]],
        torch.full((3,), 1 / 3, dtype=torch.float64),
        rtol=0,
        atol=1e-12,
    )
    efficiency = coefficients.sum(dim=-1) - result["game_values"][:, -1]
    torch.testing.assert_close(efficiency, torch.zeros_like(efficiency), rtol=0, atol=1e-12)
    result["stii"] = coefficients
    result["stii_efficiency_residual"] = efficiency
    result["stages"] = tuple(result.get("stages", ())) + ("shapley_taylor",)
    return result


def diagnose_faith_design(
    diagnostic: dict[str, Any],
    faith_design_matrix_fn: Callable[..., tuple[Tensor, Tensor]],
    faith_kernel_weights_fn: Callable[..., Tensor],
) -> dict[str, Any]:
    """Check the visible Faith design and per-coalition kernel semantics."""

    _callable("faith_design_matrix_fn", faith_design_matrix_fn)
    _callable("faith_kernel_weights_fn", faith_kernel_weights_fn)
    result = _diagnostic(diagnostic)
    design, term_masks = faith_design_matrix_fn(n_players=3, max_order=2)
    weights = faith_kernel_weights_fn(n_players=3)
    weights_n4 = faith_kernel_weights_fn(n_players=4)
    if design.shape != (8, 7) or term_masks.shape != (7,):
        raise AssertionError("For n=3, order=2 the Faith design must have shape [8,7].")
    expected_terms = torch.tensor([0, 1, 2, 3, 4, 5, 6], dtype=torch.int64)
    torch.testing.assert_close(term_masks, expected_terms, rtol=0, atol=0)
    expected = torch.arange(8, dtype=torch.int64)[:, None].bitwise_and(
        term_masks[None, :]
    ).eq(term_masks).to(torch.float64)
    torch.testing.assert_close(design, expected, rtol=0, atol=0)
    if weights.dtype != torch.float64 or weights.shape != (8,):
        raise AssertionError("Faith kernel weights must be float64 with one value per coalition.")
    torch.testing.assert_close(weights[[0, -1]], torch.zeros(2, dtype=torch.float64), rtol=0, atol=0)
    torch.testing.assert_close(weights.sum(), torch.tensor(1.0, dtype=torch.float64), rtol=0, atol=1e-12)
    expected_n4 = torch.zeros(16, dtype=torch.float64)
    for mask in range(1, 15):
        size = mask.bit_count()
        expected_n4[mask] = 3 / (math.comb(4, size) * size * (4 - size))
    expected_n4 /= expected_n4.sum()
    torch.testing.assert_close(weights_n4, expected_n4, rtol=0, atol=1e-12)
    result["faith_design"] = design
    result["faith_term_masks"] = term_masks
    result["faith_kernel_weights"] = weights
    result["faith_kernel_weights_n4"] = weights_n4
    result["stages"] = tuple(result.get("stages", ())) + ("faith_design",)
    return result


def diagnose_faith_solver(
    diagnostic: dict[str, Any],
    faith_shap_fn: Callable[..., Tensor],
    faith_closed_form_fn: Callable[..., Tensor],
) -> dict[str, Any]:
    """Add Faith-Shap coefficients and independent closed-form parity."""

    _callable("faith_shap_fn", faith_shap_fn)
    _callable("faith_closed_form_fn", faith_closed_form_fn)
    result = _diagnostic(diagnostic)
    if "mobius" not in result or "faith_design" not in result:
        raise ValueError("Run Möbius and Faith-design diagnostics before the Faith solver.")
    coefficients = faith_shap_fn(result["game_values"], n_players=4, max_order=2)
    closed = faith_closed_form_fn(result["mobius"], n_players=4, max_order=2)
    torch.testing.assert_close(coefficients, closed, rtol=0, atol=1e-10)
    pure_triple = GAME_NAMES.index("pure_triple")
    pure_four = GAME_NAMES.index("pure_four")
    singles, pairs = result["single_masks"], result["pair_masks"]
    torch.testing.assert_close(
        coefficients[pure_four, singles], torch.full((4,), -1 / 5, dtype=torch.float64), rtol=0, atol=1e-10
    )
    torch.testing.assert_close(
        coefficients[pure_four, pairs], torch.full((6,), 3 / 10, dtype=torch.float64), rtol=0, atol=1e-10
    )
    torch.testing.assert_close(
        coefficients[pure_triple, singles[:3]],
        torch.full((3,), -1 / 6, dtype=torch.float64),
        rtol=0,
        atol=1e-10,
    )
    contained = torch.tensor([0, 1, 3], dtype=torch.int64)
    torch.testing.assert_close(
        coefficients[pure_triple, pairs[contained]],
        torch.full((3,), 1 / 2, dtype=torch.float64),
        rtol=0,
        atol=1e-10,
    )
    efficiency = coefficients.sum(dim=-1) - result["game_values"][:, -1]
    torch.testing.assert_close(efficiency, torch.zeros_like(efficiency), rtol=0, atol=1e-10)
    result["fsii"] = coefficients
    result["fsii_closed_form"] = closed
    result["fsii_closed_form_max_abs_error"] = float((coefficients - closed).abs().max().item())
    result["fsii_efficiency_residual"] = efficiency
    result["stages"] = tuple(result.get("stages", ())) + ("faith_solver",)
    return result


def diagnose_reconstruction(
    diagnostic: dict[str, Any],
    reconstruct_game_fn: Callable[..., Tensor],
    faithfulness_error_fn: Callable[..., Tensor],
) -> dict[str, Any]:
    """Add same-game reconstructions and the declared Faith objective."""

    _callable("reconstruct_game_fn", reconstruct_game_fn)
    _callable("faithfulness_error_fn", faithfulness_error_fn)
    result = _diagnostic(diagnostic)
    if "stii" not in result or "fsii" not in result:
        raise ValueError("Run both interaction-index diagnostics before reconstruction.")
    reconstructions = torch.stack(
        [
            reconstruct_game_fn(result["stii"], n_players=4),
            reconstruct_game_fn(result["fsii"], n_players=4),
        ]
    )
    residual = result["game_values"].unsqueeze(0) - reconstructions
    kernel_mse = torch.stack(
        [
            faithfulness_error_fn(result["game_values"], result["stii"], n_players=4),
            faithfulness_error_fn(result["game_values"], result["fsii"], n_players=4),
        ]
    )
    if torch.any(kernel_mse[1] > kernel_mse[0] + 1e-10):
        raise AssertionError("Faith-Shap must minimize the declared kernel objective.")
    sizes = torch.tensor([mask.bit_count() for mask in range(16)], dtype=torch.int64)
    rmse_by_size = torch.stack(
        [
            residual[..., sizes == size].square().mean(dim=-1).sqrt()
            for size in range(5)
        ],
        dim=-1,
    )
    degree_two = GAME_NAMES.index("mixed_degree2")
    torch.testing.assert_close(
        reconstructions[:, degree_two],
        result["game_values"][degree_two].expand(2, -1),
        rtol=0,
        atol=1e-10,
    )
    result["reconstruction"] = reconstructions
    result["residual"] = residual
    result["kernel_mse"] = kernel_mse
    result["rmse_by_size"] = rmse_by_size
    result["method_labels"] = ("Shapley-Taylor", "Faith-Shap")
    result["stages"] = tuple(result.get("stages", ())) + ("reconstruction",)
    return result


class _AdditiveDiagnosticModel(nn.Module):
    def forward(self, inputs: Tensor) -> Tensor:
        return 1.0 + 0.5 * inputs[:, 0] + 0.25 * inputs[:, 1].square() + torch.relu(inputs[:, 2])


def _additive_formula(inputs: Tensor) -> Tensor:
    return 1.0 + 0.5 * inputs[..., 0] + 0.25 * inputs[..., 1].square() + torch.relu(inputs[..., 2])


def diagnose_model_game(masked_model_game_fn: Callable[..., Tensor]) -> dict[str, Any]:
    """Exercise one donor-averaged model game with an additive nonlinear control."""

    _callable("masked_model_game_fn", masked_model_game_fn)
    model = _AdditiveDiagnosticModel().to(dtype=torch.float64)
    query = torch.tensor([[2.0, -1.0, 0.5]], dtype=torch.float64)
    donors = torch.tensor([[0.0, 0.0, -1.0], [1.0, -2.0, 1.0]], dtype=torch.float64)
    observed = masked_model_game_fn(model, query, donors, coalition_batch_size=3)
    masks = torch.arange(8, dtype=torch.int64)
    present = masks[:, None].bitwise_and(1 << torch.arange(3, dtype=torch.int64)).ne(0)
    hybrid = torch.where(present[:, None, :], query[:, None, :], donors[None, :, :])
    expected = _additive_formula(hybrid).mean(dim=1).unsqueeze(0)
    torch.testing.assert_close(observed, expected, rtol=0, atol=1e-12)
    mobius = observed.clone()
    for player in range(3):
        selected = masks.bitwise_and(1 << player).ne(0)
        mobius[..., selected] -= mobius[..., masks[selected].bitwise_xor(1 << player)]
    pair_masks = torch.tensor([3, 5, 6], dtype=torch.int64)
    torch.testing.assert_close(mobius[..., pair_masks], torch.zeros((1, 3), dtype=torch.float64), rtol=0, atol=1e-12)
    return {
        "evidence_kind": "analytic_exact",
        "label": "nonlinear additive donor-game control",
        "feature_names": ("linear", "square", "ReLU"),
        "query": query,
        "donors": donors,
        "game_values": observed,
        "expected_game_values": expected,
        "mobius": mobius,
        "pair_max_abs": float(mobius[..., pair_masks].abs().max().item()),
        "units": "fixture output units",
    }


def diagnose_pair_context(contextual_pair_synergy_fn: Callable[..., tuple[Tensor, Tensor]]) -> dict[str, Any]:
    """Show where a pure triple's pair derivative switches on."""

    _callable("contextual_pair_synergy_fn", contextual_pair_synergy_fn)
    fixture = exact_game_fixture()
    game = fixture["game_values"][GAME_NAMES.index("pure_triple")]
    contexts, synergy = contextual_pair_synergy_fn(game, (0, 1), n_players=4)
    expected_contexts = torch.tensor([0, 4, 8, 12], dtype=torch.int64)
    expected_synergy = torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=torch.float64)
    torch.testing.assert_close(contexts, expected_contexts, rtol=0, atol=0)
    torch.testing.assert_close(synergy, expected_synergy, rtol=0, atol=1e-12)
    full_context = 12
    four_masks = torch.tensor([full_context, full_context | 1, full_context | 2, full_context | 3], dtype=torch.int64)
    return {
        "evidence_kind": "analytic_exact",
        "label": "pure-triple contextual pair diagnostic",
        "pair": (0, 1),
        "contexts": contexts,
        "synergy": synergy,
        "full_context_mask": full_context,
        "four_masks": four_masks,
        "four_values": game[four_masks],
        "pair_sii": float(fixture["sii_pairs"][GAME_NAMES.index("pure_triple"), 0].item()),
    }


def _native_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    return value


def _native_tensor(value: Any, name: str, shape: tuple[int, ...]) -> Tensor:
    tensor = value.detach().cpu() if isinstance(value, Tensor) else torch.as_tensor(value)
    if tensor.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {tuple(tensor.shape)}.")
    if tensor.dtype == torch.bool or not (tensor.is_floating_point() or tensor.dtype in (torch.int8, torch.int16, torch.int32, torch.int64)):
        raise TypeError(f"{name} must be a numerical tensor.")
    if tensor.is_floating_point() and not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must contain only finite values.")
    return tensor


def interaction_diagnostics(
    prepared: Mapping[str, Any],
    evaluated: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the frozen native payload structure and expose honest summaries.

    This aggregate never recomputes an interaction method or selects an
    empirical result. It checks the pre-audit boundary, complete array axes,
    declared controls, and exhaustive independent reference result before the
    native payload is persisted.
    """

    prepared = _native_mapping(prepared, "prepared")
    evaluated = _native_mapping(evaluated, "evaluated")
    reference = _native_mapping(reference, "reference")
    if prepared.get("phase") != "prepared_before_audit":
        raise ValueError("prepared must preserve the literal pre-audit phase.")
    if prepared.get("audit_used_for_selection") is not False:
        raise ValueError("prepared must be sealed before audit selection.")
    if evaluated.get("phase") != "evaluated_after_audit":
        raise ValueError("evaluated must preserve the literal post-audit phase.")
    if evaluated.get("audit_used_for_selection") is not False:
        raise ValueError("audit_used_for_selection must remain False.")
    if tuple(evaluated.get("model_keys", ())) != NATIVE_MODEL_KEYS:
        raise ValueError("The native result must retain all seven models in frozen order.")
    if tuple(evaluated.get("method_keys", ())) != NATIVE_METHOD_KEYS:
        raise ValueError("The native result must retain SII, STII, and FSII in frozen order.")
    if evaluated.get("target_unit") != "MPa":
        raise ValueError("The native target and interaction unit must be MPa.")

    source_ids = _native_tensor(evaluated.get("source_ids"), "source_ids", (272,))
    row_split = _native_tensor(evaluated.get("row_split"), "row_split", (272,))
    if int((row_split == 0).sum()) != 153 or int((row_split == 1).sum()) != 119:
        raise ValueError("row_split must retain 153 explanation and 119 audit rows.")
    if len(torch.unique(source_ids)) != 272:
        raise ValueError("source_ids must be unique across the two held-out splits.")

    all_rows = _native_mapping(evaluated.get("all_rows"), "all_rows")
    primary = _native_mapping(all_rows.get("primary64"), "all_rows.primary64")
    pair_values = _native_tensor(
        primary.get("pair_values"), "primary64.pair_values", (7, 272, 3, 28)
    ).to(torch.float64)
    _native_tensor(
        primary.get("stii_coefficients"),
        "primary64.stii_coefficients",
        (7, 272, 256),
    )
    _native_tensor(
        primary.get("fsii_coefficients"),
        "primary64.fsii_coefficients",
        (7, 272, 256),
    )
    _native_tensor(
        primary.get("reconstruction"),
        "primary64.reconstruction",
        (2, 7, 272, 256),
    )

    variants = _native_mapping(
        all_rows.get("background_variants"), "all_rows.background_variants"
    )
    if tuple(variants) != NATIVE_BACKGROUND_KEYS:
        raise ValueError("All five backgrounds must remain in frozen order.")
    for key in NATIVE_BACKGROUND_KEYS:
        block = _native_mapping(variants[key], f"background_variants.{key}")
        _native_tensor(block.get("pair_values"), f"{key}.pair_values", (272, 3, 28))

    direct = _native_mapping(evaluated.get("direct_full_context"), "direct_full_context")
    lookup = _native_tensor(
        direct.get("lookup_difference"), "direct_full_context.lookup_difference", (7, 272, 28)
    ).to(torch.float64)
    capture = _native_mapping(evaluated.get("pair_capture"), "pair_capture")
    _native_tensor(capture.get("capture_curves"), "capture_curves", (7, 3, 29))
    _native_tensor(
        capture.get("random_capture_curves"), "random_capture_curves", (7, 256, 29)
    )

    controls = _native_mapping(evaluated.get("random_controls"), "random_controls")
    within = _native_mapping(controls.get("within_size"), "random_controls.within_size")
    _native_tensor(within.get("permutation"), "within_size.permutation", (128, 119, 256))
    _native_tensor(within.get("pair_values"), "within_size.pair_values", (128, 119, 3, 28))
    observed = _native_mapping(within.get("observed"), "within_size.observed")
    null = _native_mapping(within.get("null"), "within_size.null")
    _native_tensor(observed.get("top3_share"), "observed.top3_share", (3,))
    _native_tensor(null.get("top3_share"), "null.top3_share", (128, 3))

    support = _native_mapping(evaluated.get("support"), "support")
    _native_tensor(
        support.get("nearest_train_distance"),
        "support.nearest_train_distance",
        (272, 256, 8),
    )
    if reference.get("status") != "PASS":
        raise AssertionError("The exhaustive same-index reference must pass.")
    reference_counts = _native_mapping(reference.get("counts"), "reference.counts")
    if int(reference_counts.get("games", -1)) != 2_992:
        raise ValueError("The exhaustive reference must retain all 2,992 games.")

    ridge_pair_max = float(pair_values[6].abs().max())
    return {
        "status": "PASS",
        "groups": (
            "audit_boundary",
            "complete_axes",
            "controls",
            "reference",
            "limitations",
        ),
        "audit_boundary": {
            "audit_used_for_selection": False,
            "explanation_rows": 153,
            "audit_rows": 119,
        },
        "complete_axes": {
            "models": 7,
            "rows": 272,
            "coalitions": 256,
            "pairs": 28,
            "backgrounds": 5,
        },
        "controls": {
            "ridge_max_abs_pair_mpa": ridge_pair_max,
            "direct_lookup_max_abs_error_mpa": float(lookup.abs().max()),
            "random_pair_orders": 256,
            "within_size_replicates": 128,
        },
        "reference": {
            "status": "PASS",
            "games": 2_992,
            "supporting_package_is_exact_index_check": True,
        },
        "limitations": {
            "indices_have_different_axiomatic_targets": True,
            "backgrounds_are_different_finite_games": True,
            "within_size_shuffle_is_not_a_significance_null": True,
            "full_context_replacement_is_not_causal": True,
            "known_data_and_models_are_reused": True,
        },
    }


__all__ = [
    "GAME_NAMES",
    "PAIR_INDICES",
    "diagnose_faith_design",
    "diagnose_faith_solver",
    "diagnose_model_game",
    "diagnose_mobius_games",
    "diagnose_pair_context",
    "diagnose_reconstruction",
    "diagnose_shapley_taylor",
    "exact_game_fixture",
    "interaction_diagnostics",
]
