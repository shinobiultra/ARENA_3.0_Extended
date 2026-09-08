import importlib.util

import pytest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is not installed")

if TORCH_AVAILABLE:
    import torch as t

    from arena_ext.shapley_attribution import (
        exact_token_shapley_values,
        keyword_interaction_token_score,
        sampled_token_shapley_values,
        token_baseline_report,
        token_coalition_values,
        token_shapley_sampling_report,
    )


TOKENS = ("The", "capital", "is", "Paris")


def test_token_coalition_values_mask_absent_positions():
    values = token_coalition_values(TOKENS, keyword_interaction_token_score)

    assert values[frozenset()] == 0.0
    assert values[frozenset({3})] == 1.0
    assert values[frozenset({1, 3})] == 3.0


def test_exact_token_shapley_splits_context_target_interaction():
    shapley = exact_token_shapley_values(TOKENS, keyword_interaction_token_score)

    t.testing.assert_close(shapley, t.tensor([0.0, 1.0, 0.0, 2.0], dtype=t.float64))


def test_sampled_token_shapley_approximates_exact_and_keeps_top_token():
    report = token_shapley_sampling_report(
        TOKENS,
        keyword_interaction_token_score,
        num_samples=512,
        seed=0,
        tolerance=0.1,
    )

    assert report.approximates_exact
    assert report.rank_matches
    assert report.top_token == "Paris"
    assert report.sampled_top_token == "Paris"
    assert report.max_abs_error < 0.1


def test_token_baseline_report_satisfies_efficiency():
    report = token_baseline_report(TOKENS, keyword_interaction_token_score)

    assert report.full_score == pytest.approx(3.0)
    assert report.baseline_score == pytest.approx(0.0)
    assert report.shapley_sum == pytest.approx(3.0)
    assert report.satisfies_efficiency


def test_sampled_token_shapley_rejects_nonpositive_sample_count():
    with pytest.raises(ValueError, match="positive"):
        sampled_token_shapley_values(
            TOKENS,
            keyword_interaction_token_score,
            num_samples=0,
        )
