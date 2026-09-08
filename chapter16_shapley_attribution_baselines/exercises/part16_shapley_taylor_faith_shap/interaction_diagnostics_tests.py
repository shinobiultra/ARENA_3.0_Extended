"""Semantic tests for the staged interaction diagnostics."""

from __future__ import annotations

import copy

import pytest
import torch

from . import interaction_diagnostics as diagnostics
from . import interaction_methods as methods


def _complete_diagnostic() -> dict:
    result = diagnostics.diagnose_mobius_games(methods.mobius_transform)
    result = diagnostics.diagnose_shapley_taylor(result, methods.shapley_taylor_interactions)
    result = diagnostics.diagnose_faith_design(
        result, methods.faith_design_matrix, methods.faith_kernel_weights
    )
    result = diagnostics.diagnose_faith_solver(
        result, methods.faith_shap_interactions, methods.faith_shap_closed_form
    )
    return diagnostics.diagnose_reconstruction(
        result, methods.reconstruct_game, methods.faithfulness_error
    )


def test_exact_fixture_has_declared_games_and_separate_sii_comparators():
    result = diagnostics.exact_game_fixture()
    assert result["game_names"] == diagnostics.GAME_NAMES
    assert result["game_values"].shape == (8, 16)
    assert result["expected_mobius"].dtype == torch.float64
    pure_four = result["game_names"].index("pure_four")
    torch.testing.assert_close(
        result["ordinary_shapley"][pure_four],
        torch.full((4,), 1 / 4, dtype=torch.float64),
        rtol=0,
        atol=1e-12,
    )
    torch.testing.assert_close(
        result["sii_pairs"][pure_four],
        torch.full((6,), 1 / 3, dtype=torch.float64),
        rtol=0,
        atol=1e-12,
    )
    assert "not" not in result["exact_fraction_labels"]["pure_four"]


def test_cumulative_diagnostics_use_visible_methods_and_preserve_prior_bindings():
    first = diagnostics.diagnose_mobius_games(methods.mobius_transform)
    snapshot = copy.deepcopy(first)
    second = diagnostics.diagnose_shapley_taylor(first, methods.shapley_taylor_interactions)
    assert "stii" not in first
    torch.testing.assert_close(first["mobius"], snapshot["mobius"], rtol=0, atol=0)
    third = diagnostics.diagnose_faith_design(
        second, methods.faith_design_matrix, methods.faith_kernel_weights
    )
    fourth = diagnostics.diagnose_faith_solver(
        third, methods.faith_shap_interactions, methods.faith_shap_closed_form
    )
    complete = diagnostics.diagnose_reconstruction(
        fourth, methods.reconstruct_game, methods.faithfulness_error
    )
    assert complete["stages"] == (
        "mobius",
        "shapley_taylor",
        "faith_design",
        "faith_solver",
        "reconstruction",
    )
    assert float(complete["kernel_mse"][1].max()) <= float(
        complete["kernel_mse"][0].max()
    )


def test_shapley_taylor_diagnostic_rejects_ordinary_shapley_singletons():
    base = diagnostics.diagnose_mobius_games(methods.mobius_transform)

    def wrong(mobius, *, n_players, order):
        result = methods.shapley_taylor_interactions(
            mobius, n_players=n_players, order=order
        )
        pure_four = diagnostics.GAME_NAMES.index("pure_four")
        result[pure_four, torch.tensor([1, 2, 4, 8])] = 1 / 4
        return result

    with pytest.raises(AssertionError):
        diagnostics.diagnose_shapley_taylor(base, wrong)


def test_faith_design_diagnostic_rejects_cardinality_probability_as_row_weight():
    base = diagnostics.diagnose_shapley_taylor(
        diagnostics.diagnose_mobius_games(methods.mobius_transform),
        methods.shapley_taylor_interactions,
    )

    def wrong_weights(*, n_players):
        weights = methods.faith_kernel_weights(n_players=n_players)
        for size in range(1, n_players):
            masks = torch.tensor(
                [mask for mask in range(2**n_players) if mask.bit_count() == size]
            )
            weights[masks] *= len(masks)
        return weights / weights.sum()

    with pytest.raises(AssertionError):
        diagnostics.diagnose_faith_design(
            base, methods.faith_design_matrix, wrong_weights
        )


def test_faith_solver_diagnostic_rejects_wrong_high_order_allocation():
    base = diagnostics.diagnose_faith_design(
        diagnostics.diagnose_shapley_taylor(
            diagnostics.diagnose_mobius_games(methods.mobius_transform),
            methods.shapley_taylor_interactions,
        ),
        methods.faith_design_matrix,
        methods.faith_kernel_weights,
    )

    def wrong(game_values, *, n_players, max_order):
        result = methods.faith_shap_interactions(
            game_values, n_players=n_players, max_order=max_order
        )
        pure_four = diagnostics.GAME_NAMES.index("pure_four")
        result[pure_four, base["pair_masks"]] = -result[
            pure_four, base["pair_masks"]
        ]
        return result

    with pytest.raises(AssertionError):
        diagnostics.diagnose_faith_solver(
            base, wrong, methods.faith_shap_closed_form
        )


def test_reconstruction_diagnostic_rejects_reversed_zeta_direction():
    base = diagnostics.diagnose_faith_solver(
        diagnostics.diagnose_faith_design(
            diagnostics.diagnose_shapley_taylor(
                diagnostics.diagnose_mobius_games(methods.mobius_transform),
                methods.shapley_taylor_interactions,
            ),
            methods.faith_design_matrix,
            methods.faith_kernel_weights,
        ),
        methods.faith_shap_interactions,
        methods.faith_shap_closed_form,
    )

    def wrong(coefficients, *, n_players):
        return -methods.reconstruct_game(coefficients, n_players=n_players)

    with pytest.raises(AssertionError):
        diagnostics.diagnose_reconstruction(
            base, wrong, methods.faithfulness_error
        )


def test_model_game_diagnostic_rejects_using_only_one_donor():
    def wrong(model, x, reference, *, coalition_batch_size):
        return methods.masked_model_game(
            model, x, reference[:1], coalition_batch_size=coalition_batch_size
        )

    with pytest.raises(AssertionError):
        diagnostics.diagnose_model_game(wrong)


def test_model_game_and_context_diagnostics_have_exact_visible_controls():
    model = diagnostics.diagnose_model_game(methods.masked_model_game)
    assert model["pair_max_abs"] == 0.0
    context = diagnostics.diagnose_pair_context(methods.contextual_pair_synergy)
    torch.testing.assert_close(
        context["synergy"],
        torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=torch.float64),
        rtol=0,
        atol=0,
    )
    assert context["pair_sii"] == 0.5


def test_context_diagnostic_rejects_reversed_second_difference_sign():
    def wrong(game_values, pair, *, n_players):
        masks, values = methods.contextual_pair_synergy(
            game_values, pair, n_players=n_players
        )
        return masks, -values

    with pytest.raises(AssertionError):
        diagnostics.diagnose_pair_context(wrong)


def test_diagnostic_stage_order_is_enforced():
    fixture = diagnostics.exact_game_fixture()
    with pytest.raises(ValueError, match="diagnose_mobius_games"):
        diagnostics.diagnose_shapley_taylor(
            fixture, methods.shapley_taylor_interactions
        )
    with pytest.raises(ValueError, match="both interaction"):
        diagnostics.diagnose_reconstruction(
            fixture, methods.reconstruct_game, methods.faithfulness_error
        )


def _native_aggregate_fixture():
    primary = {
        "pair_values": torch.zeros((7, 272, 3, 28), dtype=torch.float64),
        "stii_coefficients": torch.zeros((7, 272, 256), dtype=torch.float64),
        "fsii_coefficients": torch.zeros((7, 272, 256), dtype=torch.float64),
        "reconstruction": torch.zeros((2, 7, 272, 256), dtype=torch.float64),
    }
    variant = {"pair_values": torch.zeros((272, 3, 28), dtype=torch.float64)}
    evaluated = {
        "phase": "evaluated_after_audit",
        "audit_used_for_selection": False,
        "model_keys": diagnostics.NATIVE_MODEL_KEYS,
        "method_keys": diagnostics.NATIVE_METHOD_KEYS,
        "target_unit": "MPa",
        "source_ids": torch.arange(272, dtype=torch.int64),
        "row_split": torch.cat(
            (torch.zeros(153, dtype=torch.int64), torch.ones(119, dtype=torch.int64))
        ),
        "all_rows": {
            "primary64": primary,
            "background_variants": {
                key: variant for key in diagnostics.NATIVE_BACKGROUND_KEYS
            },
        },
        "direct_full_context": {
            "lookup_difference": torch.zeros((7, 272, 28), dtype=torch.float64)
        },
        "pair_capture": {
            "capture_curves": torch.zeros((7, 3, 29), dtype=torch.float64),
            "random_capture_curves": torch.zeros((7, 256, 29), dtype=torch.float64),
        },
        "random_controls": {
            "within_size": {
                "permutation": torch.zeros((128, 119, 256), dtype=torch.int16),
                "pair_values": torch.zeros((128, 119, 3, 28), dtype=torch.float64),
                "observed": {"top3_share": torch.zeros(3, dtype=torch.float64)},
                "null": {"top3_share": torch.zeros((128, 3), dtype=torch.float64)},
            }
        },
        "support": {
            "nearest_train_distance": torch.zeros((272, 256, 8), dtype=torch.float64)
        },
    }
    prepared = {
        "phase": "prepared_before_audit",
        "audit_used_for_selection": False,
    }
    reference = {"status": "PASS", "counts": {"games": 2_992}}
    return prepared, evaluated, reference


def test_native_aggregate_reports_complete_axes_controls_and_limitations():
    result = diagnostics.interaction_diagnostics(*_native_aggregate_fixture())
    assert result["status"] == "PASS"
    assert result["groups"] == (
        "audit_boundary",
        "complete_axes",
        "controls",
        "reference",
        "limitations",
    )
    assert result["complete_axes"] == {
        "models": 7,
        "rows": 272,
        "coalitions": 256,
        "pairs": 28,
        "backgrounds": 5,
    }
    assert result["controls"]["random_pair_orders"] == 256
    assert result["limitations"]["within_size_shuffle_is_not_a_significance_null"]


def test_native_aggregate_rejects_audit_selection_and_missing_games():
    prepared, evaluated, reference = _native_aggregate_fixture()
    evaluated["audit_used_for_selection"] = True
    with pytest.raises(ValueError, match="audit_used_for_selection"):
        diagnostics.interaction_diagnostics(prepared, evaluated, reference)
    evaluated["audit_used_for_selection"] = False
    reference["counts"]["games"] = 2_991
    with pytest.raises(ValueError, match="2,992"):
        diagnostics.interaction_diagnostics(prepared, evaluated, reference)
