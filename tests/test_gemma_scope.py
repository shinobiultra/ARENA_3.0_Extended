import importlib.util

import pytest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is not installed")

if TORCH_AVAILABLE:
    import torch as t

    from arena_ext.gemma_scope import (
        FeatureArtifactMetadata,
        TaggedFeatureSpec,
        ablation_control_report,
        base_instruction_feature_delta,
        feature_score_vector,
        features_with_tag,
        gemma_base_model_access_report,
        gemma_scope_jump_relu_forward,
        metadata_is_complete,
        steering_safety_report,
        validate_feature_scores,
    )


def test_metadata_and_tag_selection():
    metadata = FeatureArtifactMetadata(
        model_name="gemma-test",
        artifact_name="layer_0_resid_sae",
        artifact_type="sae",
        layer=0,
        hook_name="resid_post",
        d_model=4,
        n_features=8,
    )
    features = [
        TaggedFeatureSpec(0, 0, ("refusal", "safety"), "Refusal phrase feature"),
        TaggedFeatureSpec(1, 0, ("code",), "Code formatting feature"),
    ]

    assert metadata_is_complete(metadata)
    assert [feature.feature_id for feature in features_with_tag(features, "refusal")] == [0]


def test_feature_score_vector_reduces_token_activations():
    feature_acts = t.tensor(
        [
            [[0.0, 0.1], [0.0, 0.5], [0.0, 0.2]],
            [[0.0, 0.4], [0.0, 0.3], [0.0, 0.9]],
        ]
    )

    assert t.equal(feature_score_vector(feature_acts, 1, reduction="max"), t.tensor([0.5, 0.9]))
    assert t.equal(feature_score_vector(feature_acts, 1, reduction="last"), t.tensor([0.2, 0.9]))
    assert t.allclose(
        feature_score_vector(feature_acts, 1, reduction="mean"),
        t.tensor([0.26666668, 0.53333336]),
    )


def test_gemma_scope_jump_relu_forward_matches_hand_computed_case():
    activations = t.tensor([[2.0, 1.0], [0.0, 3.0]])
    w_enc = t.tensor([[1.0, -1.0, 0.0], [0.0, 2.0, 1.0]])
    w_dec = t.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    b_enc = t.tensor([0.0, 0.0, -1.0])
    b_dec = t.tensor([0.0, 0.0])
    threshold = t.tensor([1.5, 1.5, 1.5])

    feature_acts, reconstructed = gemma_scope_jump_relu_forward(
        activations,
        w_enc=w_enc,
        w_dec=w_dec,
        b_enc=b_enc,
        b_dec=b_dec,
        threshold=threshold,
    )

    expected_features = t.tensor([[2.0, 0.0, 0.0], [0.0, 6.0, 2.0]])
    expected_reconstructed = t.tensor([[2.0, 0.0], [2.0, 8.0]])
    assert t.equal(feature_acts, expected_features)
    assert t.equal(reconstructed, expected_reconstructed)


def test_validate_feature_scores_beats_baseline_auc():
    scores = t.tensor([0.1, 0.2, 0.9, 1.0])
    labels = t.tensor([0, 0, 1, 1], dtype=t.bool)
    baseline_scores = t.tensor([1.0, 0.9, 0.2, 0.1])

    report = validate_feature_scores(scores, labels, baseline_scores)

    assert report.feature_auc == 1.0
    assert report.baseline_auc == 0.0
    assert report.passes_baseline
    assert report.threshold_accuracy == 1.0


def test_gemma_base_model_access_report_non_network_mode_is_a_gate():
    report = gemma_base_model_access_report(allow_network=False)

    assert report["model_id"] == "google/gemma-3-1b-it"
    assert report["model_revision"] == "dcc83ea841ab6100d6b47a070329e1ba4cf78752"
    assert report["allow_network"] is False
    assert report["repo_listed"] is False
    assert report["ready_for_real_activations"] is report[
        "local_ready_for_real_activations"
    ]
    assert report["semantic_feature_claimed"] is False


def test_base_instruction_delta_reports_direction():
    base = t.tensor([0.1, 0.2, 0.3])
    instruction = t.tensor([0.3, 0.4, 0.5])

    report = base_instruction_feature_delta(base, instruction)

    assert report.delta == pytest.approx(0.2)
    assert report.abs_delta == pytest.approx(0.2)


def test_ablation_control_report_requires_target_specific_drop():
    baseline = t.tensor([1.0, 1.0])
    ablated = t.tensor([0.2, 0.3])
    random_ablated = t.tensor([0.8, 0.9])

    report = ablation_control_report(baseline, ablated, random_ablated)

    assert report.ablation_delta == pytest.approx(0.75)
    assert report.random_delta == pytest.approx(0.15)
    assert report.passes_control


def test_steering_safety_report_checks_control_and_perplexity_guard():
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

    assert report.steered_delta == pytest.approx(0.5)
    assert report.random_delta == pytest.approx(0.1)
    assert report.perplexity_ratio == pytest.approx(1.1)
    assert report.passes_control
    assert report.passes_perplexity_guard
