from collections.abc import Callable


def _solutions():
    from chapter16_shapley_attribution_baselines.exercises.part6_shap_vs_activation_patching import (
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
    assert result["agrees_with_shapley"], (
        "In an additive game, full-minus-ablated patching effects should equal Shapley values."
    )
    assert result["top_feature_agrees"], (
        "The top feature should agree between Shapley and patching in the additive control."
    )
    assert result["shapley_values"] == [1.0, 2.0, 0.5], (
        "The additive game's Shapley values should recover the feature weights exactly."
    )
    assert result["patching_effects"] == [1.0, 2.0, 0.5], (
        "The additive game's patching effects should recover the feature weights exactly."
    )
    print("All tests in `test_additive_agreement_smoke_test` passed!")


def test_interaction_failure_smoke_test(
    interaction_failure_smoke_test: Callable | None = None,
):
    interaction_failure_smoke_test = (
        interaction_failure_smoke_test or _solutions().interaction_failure_smoke_test
    )
    result = interaction_failure_smoke_test()
    assert result["documents_overcount"], (
        "The two-feature AND game should explicitly document patching overcount."
    )
    assert result["shapley_values"] == [0.5, 0.5], (
        "Shapley should split the AND interaction equally between the two necessary features."
    )
    assert result["patching_effects"] == [1.0, 1.0], (
        "Full-minus-ablated patching should assign the full AND effect to each necessary feature."
    )
    assert result["overcount"] == 1.0, (
        "Patching should overcount the total interaction credit by 1.0 in this fixture."
    )
    print("All tests in `test_interaction_failure_smoke_test` passed!")


def test_notebook_contract(run_smoke_test: Callable | None = None):
    run_smoke_test = run_smoke_test or _solutions().run_smoke_test
    result = run_smoke_test(cpu=True)
    assert result["additive_agreement"]["agrees_with_shapley"], (
        "The notebook contract should include the additive agreement control."
    )
    assert result["interaction_failure"]["documents_overcount"], (
        "The notebook contract should include the interaction overcount failure mode."
    )
    print("All tests in `test_notebook_contract` passed!")
