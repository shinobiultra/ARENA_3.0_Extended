"""16.1 Exact Shapley on ground-truth games: tiny CPU oracles and wrong-answer rejection."""
from __future__ import annotations

import importlib

import pytest

MODULE = "chapter16_shapley_attribution_baselines.exercises.part1_exact_shapley"


ORACLES = [
    ("test_all_coalitions_enumerates_the_power_set", ["all_coalitions"]),
    ("test_complete_table_normalization", ["all_coalitions", "normalize_coalition_values", "coalition_values_from_function"]),
    ("test_dividend_game_and_analytic_oracle", ["game_from_dividends", "shapley_from_dividends"]),
    ("test_weighted_marginal_rows", ["marginal_contribution_rows"]),
    ("test_additive_game_and_exact_shapley_recover_weights", ["additive_game", "exact_shapley_values"]),
    ("test_exact_shapley_requires_a_complete_coalition_table", ["exact_shapley_values"]),
    ("test_conjunction_game_splits_symmetric_credit_and_checks_efficiency", ["conjunction_game", "exact_shapley_values", "shapley_efficiency_report"]),
    ("test_permutation_parity_report_matches_exact_formula", ["conjunction_game", "permutation_parity_report"]),
    ("test_interaction_gap_report_catches_leave_one_out_overcount", ["conjunction_game", "interaction_gap_report"]),
    ("test_interaction_scale_sweep", ["interaction_scale_sweep"]),
]


@pytest.mark.parametrize("oracle,functions", ORACLES)
def test_visible_student_methods_with_local_independent_oracles(oracle, functions):
    methods = importlib.import_module(MODULE + ".solutions")
    tests = importlib.import_module(MODULE + ".tests")
    getattr(tests, oracle)(*[getattr(methods, name) for name in functions])


def test_oracles_reject_broken_student_answers():
    import torch

    methods = importlib.import_module(MODULE + ".solutions")
    tests = importlib.import_module(MODULE + ".tests")
    with pytest.raises(AssertionError):
        tests.test_dividend_game_and_analytic_oracle(methods.game_from_dividends,
                                                   lambda *args: torch.zeros(4, dtype=torch.float64))
    with pytest.raises(AssertionError):
        tests.test_interaction_scale_sweep(lambda scales: {
            "shapley_efficiency_error": torch.zeros_like(scales),
            "oracle_max_error": torch.zeros_like(scales),
            "leave_one_out_overcount": torch.zeros_like(scales),
        })
