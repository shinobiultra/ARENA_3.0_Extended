from collections.abc import Callable


def _solutions():
    from chapter16_shapley_attribution_baselines.exercises.part8_shapley_mechinterp_agreement import (
        solutions,
    )

    return solutions


def test_additive_agreement_smoke_test(
    additive_agreement_smoke_test: Callable | None = None,
):
    additive_agreement_smoke_test = (
        additive_agreement_smoke_test or _solutions().additive_agreement_smoke_test
    )
    result = additive_agreement_smoke_test()
    assert result["agrees_with_mechanistic"], (
        "Additive Shapley, patching, and mechanistic scores should agree in the positive control."
    )
    assert result["topk_overlap"] == 1.0, (
        "The top-k feature set should fully overlap with the mechanistic ground truth."
    )
    assert result["spearman_correlation"] > 0.99, (
        "Attribution ranking should have near-perfect Spearman correlation with the mechanism."
    )
    assert result["deletion_drop"] > result["random_baseline_drop"], (
        "Deleting the top attributed feature should damage behavior more than the non-top baseline."
    )
    print("All tests in `test_additive_agreement_smoke_test` passed!")


def test_xor_disagreement_smoke_test(
    xor_disagreement_smoke_test: Callable | None = None,
):
    xor_disagreement_smoke_test = (
        xor_disagreement_smoke_test or _solutions().xor_disagreement_smoke_test
    )
    result = xor_disagreement_smoke_test()
    assert result["ordinary_shapley_misses"], (
        "Ordinary single-feature Shapley should miss the XOR pair mechanism."
    )
    assert result["interaction_recovers_pair"], (
        "Pairwise Shapley interactions should recover the XOR causal pair."
    )
    assert result["recovered_pair_interaction"] == 2.0, (
        "The recovered XOR pair interaction should have value 2.0."
    )
    assert result["max_single_feature_value"] == 0.0, (
        "Single-feature values should be zero in this XOR fixture."
    )
    print("All tests in `test_xor_disagreement_smoke_test` passed!")


def test_notebook_contract(run_smoke_test: Callable | None = None):
    run_smoke_test = run_smoke_test or _solutions().run_smoke_test
    result = run_smoke_test(cpu=True)
    assert result["additive_agreement"]["agrees_with_mechanistic"], (
        "The notebook contract should include the additive agreement control."
    )
    assert result["xor_disagreement"]["interaction_recovers_pair"], (
        "The notebook contract should include the XOR interaction disagreement control."
    )
    print("All tests in `test_notebook_contract` passed!")
