from collections.abc import Callable

import torch as t

from arena_ext import features as feature_reference
from arena_ext import gemma_scope as reference


def _solutions():
    from chapter6_sparse_feature_methods.exercises.part2_gemma_scope_deep_dive import (
        solutions,
    )

    return solutions


def test_metadata_completeness_and_tag_selection(
    FeatureArtifactMetadata: type | None = None,
    TaggedFeatureSpec: type | None = None,
    metadata_is_complete: Callable | None = None,
    features_with_tag: Callable | None = None,
):
    solutions = _solutions()
    FeatureArtifactMetadata = FeatureArtifactMetadata or solutions.FeatureArtifactMetadata
    TaggedFeatureSpec = TaggedFeatureSpec or solutions.TaggedFeatureSpec
    metadata_is_complete = metadata_is_complete or solutions.metadata_is_complete
    features_with_tag = features_with_tag or solutions.features_with_tag

    metadata = FeatureArtifactMetadata(
        model_name="gemma-test",
        artifact_name="layer_0_resid_sae",
        artifact_type="sae",
        layer=0,
        hook_name="resid_post",
        d_model=4,
        n_features=8,
    )
    incomplete = FeatureArtifactMetadata(
        model_name="",
        artifact_name="layer_0_resid_sae",
        artifact_type="sae",
        layer=0,
        hook_name="resid_post",
        d_model=4,
        n_features=8,
    )
    reference_metadata = reference.FeatureArtifactMetadata(**metadata.__dict__)
    assert metadata_is_complete(metadata) == reference.metadata_is_complete(reference_metadata), (
        "Complete artifact metadata should match the independent metadata contract."
    )
    assert not metadata_is_complete(incomplete), (
        "Metadata without a model name should not be considered reproducible."
    )

    features = [
        TaggedFeatureSpec(0, 0, ("refusal", "safety"), "Refusal phrase feature"),
        TaggedFeatureSpec(1, 0, ("code",), "Code formatting feature"),
        TaggedFeatureSpec(2, 0, ("sentiment", "safety"), "Positive sentiment feature"),
    ]
    refusal_ids = [feature.feature_id for feature in features_with_tag(features, "refusal")]
    safety_ids = [feature.feature_id for feature in features_with_tag(features, "safety")]
    assert refusal_ids == [0], "Tag filtering should return only matching feature ids."
    assert safety_ids == [0, 2], "Tag filtering should preserve feature order."
    print("All tests in `test_metadata_completeness_and_tag_selection` passed!")


def test_feature_score_vector_reductions_match_reference(
    feature_score_vector: Callable | None = None,
):
    solutions = _solutions()
    feature_score_vector = feature_score_vector or solutions.feature_score_vector
    feature_acts = t.tensor(
        [
            [[0.0, 0.1], [0.0, 0.5], [0.0, 0.2]],
            [[0.0, 0.4], [0.0, 0.3], [0.0, 0.9]],
        ]
    )
    for reduction in ("max", "mean", "last"):
        actual = feature_score_vector(feature_acts, 1, reduction=reduction)
        expected = reference.feature_score_vector(feature_acts, 1, reduction=reduction)
        t.testing.assert_close(
            actual,
            expected,
            msg=f"Feature score reduction {reduction!r} should match the reference.",
        )
    t.testing.assert_close(
        feature_score_vector(feature_acts[:, -1, :], 1),
        t.tensor([0.2, 0.9]),
        msg="Rank-2 feature activations should be treated as one score per example.",
    )
    print("All tests in `test_feature_score_vector_reductions_match_reference` passed!")


def test_validate_feature_scores_beats_baseline_and_reports_means(
    validate_feature_scores: Callable | None = None,
    roc_auc_binary: Callable | None = None,
):
    solutions = _solutions()
    validate_feature_scores = validate_feature_scores or solutions.validate_feature_scores
    roc_auc_binary = roc_auc_binary or solutions.roc_auc_binary
    scores = t.tensor([0.1, 0.2, 0.9, 1.0])
    labels = t.tensor([0, 0, 1, 1], dtype=t.bool)
    baseline_scores = t.tensor([1.0, 0.9, 0.2, 0.1])
    report = validate_feature_scores(scores, labels, baseline_scores)
    reference_report = reference.validate_feature_scores(scores, labels, baseline_scores)
    assert report.__dict__ == reference_report.__dict__, (
        "Feature validation report should match the independent AUC and threshold reference."
    )
    assert report.feature_auc == 1.0 and report.baseline_auc == 0.0, (
        "The candidate feature should perfectly rank positives while the baseline is inverted."
    )
    assert report.passes_baseline and report.auc_margin == 1.0, (
        "Feature AUC should beat the baseline by the configured margin."
    )
    assert report.positive_mean > report.negative_mean, (
        "Positive examples should have larger mean candidate feature scores."
    )
    assert roc_auc_binary(baseline_scores, labels) == 0.0, (
        "Baseline AUC should expose an antipredictive baseline score vector."
    )
    print("All tests in `test_validate_feature_scores_beats_baseline_and_reports_means` passed!")


def test_base_instruction_delta_reports_signed_and_abs_change(
    base_instruction_feature_delta: Callable | None = None,
):
    solutions = _solutions()
    base_instruction_feature_delta = (
        base_instruction_feature_delta or solutions.base_instruction_feature_delta
    )
    base = t.tensor([0.1, 0.2, 0.3])
    instruction = t.tensor([0.3, 0.4, 0.5])
    report = base_instruction_feature_delta(base, instruction)
    reference_report = reference.base_instruction_feature_delta(base, instruction)
    assert report.__dict__ == reference_report.__dict__, (
        "Base-vs-instruction delta should match mean activation difference reference."
    )
    assert abs(report.delta - 0.2) < 1e-6 and abs(report.abs_delta - 0.2) < 1e-6, (
        "Positive deltas should preserve sign and absolute magnitude."
    )
    reversed_report = base_instruction_feature_delta(instruction, base)
    assert reversed_report.delta < 0 and abs(reversed_report.abs_delta - 0.2) < 1e-6, (
        "Negative deltas should retain sign while abs_delta remains positive."
    )
    print("All tests in `test_base_instruction_delta_reports_signed_and_abs_change` passed!")


def test_ablation_control_requires_target_ablation_to_beat_random(
    ablation_control_report: Callable | None = None,
):
    solutions = _solutions()
    ablation_control_report = ablation_control_report or solutions.ablation_control_report
    baseline = t.tensor([1.0, 1.0])
    ablated = t.tensor([0.2, 0.3])
    random_ablated = t.tensor([0.8, 0.9])
    report = ablation_control_report(baseline, ablated, random_ablated)
    reference_report = reference.ablation_control_report(baseline, ablated, random_ablated)
    assert report.__dict__ == reference_report.__dict__, (
        "Ablation control report should match the baseline-target-random delta reference."
    )
    assert abs(report.ablation_delta - 0.75) < 1e-6, (
        "Target ablation should reduce the score by the mean baseline-minus-ablated gap."
    )
    assert abs(report.random_delta - 0.15) < 1e-6 and report.passes_control, (
        "Target ablation should beat the random-feature ablation."
    )
    failed = ablation_control_report(baseline, t.tensor([0.8, 0.9]), t.tensor([0.2, 0.3]))
    assert not failed.passes_control, (
        "Ablation should fail the control when random ablation has the larger effect."
    )
    print("All tests in `test_ablation_control_requires_target_ablation_to_beat_random` passed!")


def test_steering_safety_report_checks_control_and_perplexity_guard(
    steering_safety_report: Callable | None = None,
):
    solutions = _solutions()
    steering_safety_report = steering_safety_report or solutions.steering_safety_report
    baseline = t.tensor([0.1, 0.2])
    steered = t.tensor([0.6, 0.7])
    random = t.tensor([0.2, 0.3])
    report = steering_safety_report(
        baseline,
        steered,
        random,
        baseline_perplexity=10.0,
        steered_perplexity=11.0,
    )
    reference_report = reference.steering_safety_report(
        baseline,
        steered,
        random,
        baseline_perplexity=10.0,
        steered_perplexity=11.0,
    )
    assert report.__dict__ == reference_report.__dict__, (
        "Steering report should match target/random deltas and perplexity guard reference."
    )
    assert abs(report.steered_delta - 0.5) < 1e-6, (
        "Steering delta should be the mean steered-minus-baseline gap."
    )
    assert report.passes_control and report.passes_perplexity_guard, (
        "Steering should beat the random control and stay inside the perplexity guard."
    )
    unsafe = steering_safety_report(
        baseline,
        steered,
        random,
        baseline_perplexity=10.0,
        steered_perplexity=15.0,
    )
    assert unsafe.passes_control and not unsafe.passes_perplexity_guard, (
        "A useful steering direction should still fail if perplexity degradation is too large."
    )
    print("All tests in `test_steering_safety_report_checks_control_and_perplexity_guard` passed!")


def test_direct_logit_attribution_matches_selected_token_projection(
    direct_logit_attribution: Callable | None = None,
):
    solutions = _solutions()
    direct_logit_attribution = direct_logit_attribution or solutions.direct_logit_attribution
    decoder_vectors = t.tensor([[1.0, 0.0], [0.0, 1.0]])
    unembedding = t.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    effects = direct_logit_attribution(decoder_vectors, unembedding, token_ids=[0, 2])
    reference_effects = feature_reference.direct_logit_attribution(
        decoder_vectors,
        unembedding,
        token_ids=[0, 2],
    )
    t.testing.assert_close(
        effects,
        reference_effects,
        msg="Direct logit attribution should project decoder vectors through the unembedding.",
    )
    t.testing.assert_close(
        effects,
        t.tensor([[1.0, 3.0], [4.0, 6.0]]),
        msg="Selected token ids should slice the vocabulary-effect dimension.",
    )
    full_effects = direct_logit_attribution(decoder_vectors, unembedding)
    assert list(full_effects.shape) == [2, 3], (
        "Without token ids, direct logit attribution should return all vocabulary effects."
    )
    print("All tests in `test_direct_logit_attribution_matches_selected_token_projection` passed!")


def test_notebook_contract(run_smoke_test: Callable | None = None):
    if run_smoke_test is None:
        run_smoke_test = _solutions().run_smoke_test
    result = run_smoke_test(cpu=True)
    assert result["metadata"]["metadata_complete"], (
        "Notebook contract should include complete artifact metadata."
    )
    assert result["validation"]["passes_baseline"], (
        "Notebook contract should include held-out feature validation over a baseline."
    )
    assert result["base_instruction_delta"]["delta"] > 0, (
        "Notebook contract should include a signed base-vs-instruction feature delta."
    )
    assert result["ablation"]["passes_control"], (
        "Notebook contract should include ablation against a random-feature control."
    )
    assert result["steering"]["passes_perplexity_guard"], (
        "Notebook contract should include a steering perplexity guard."
    )
    assert result["logit_attribution"] == [[1.0, 3.0], [4.0, 6.0]], (
        "Notebook contract should include selected direct-logit-attribution effects."
    )
    print("All tests in `test_notebook_contract` passed!")
