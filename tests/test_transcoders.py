import importlib.util

import pytest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is not installed")

if TORCH_AVAILABLE:
    import torch as t

    from arena_ext.transcoders import (
        attribution_graph_report,
        build_attribution_edges,
        feature_logit_contributions,
        graph_reproducible,
        target_logit_diff,
        transcoder_forward,
        transcoder_replacement_report,
    )


def test_transcoder_forward_reconstructs_positive_inputs_with_identity_weights():
    inputs = t.tensor([[1.0, 2.0], [3.0, 4.0]])
    output = transcoder_forward(inputs, t.eye(2), t.eye(2))

    assert t.equal(output.feature_acts, inputs)
    assert t.equal(output.reconstructed_activations, inputs)


def test_transcoder_replacement_report_passes_for_close_logits():
    reference_activations = t.tensor([[1.0, 2.0]])
    reconstructed = reference_activations + 0.01
    reference_logits = t.tensor([[2.0, 0.0, -1.0]])
    replacement_logits = t.tensor([[1.98, 0.02, -1.0]])

    report = transcoder_replacement_report(
        reference_activations=reference_activations,
        reconstructed_activations=reconstructed,
        reference_logits=reference_logits,
        replacement_logits=replacement_logits,
        positive_token_id=0,
        negative_token_id=1,
        kl_threshold=1e-3,
        logit_diff_tolerance=0.1,
    )

    assert report.reconstruction_mse < 0.001
    assert report.passes_kl
    assert report.preserves_logit_diff
    assert target_logit_diff(reference_logits, positive_token_id=0, negative_token_id=1) == 2.0


def test_feature_logit_contributions_multiply_mean_acts_by_effects():
    feature_acts = t.tensor([[1.0, 2.0, 0.0], [3.0, 0.0, 2.0]])
    logit_effects = t.tensor([0.5, 1.0, -1.0])

    contributions = feature_logit_contributions(feature_acts, logit_effects)

    assert t.equal(contributions, t.tensor([1.0, 1.0, -1.0]))


def test_build_attribution_edges_and_reproducibility():
    input_scores = t.tensor([[0.1, 0.8], [0.4, 0.2]])
    logit_effects = t.tensor([0.3, 1.0])

    edges = build_attribution_edges(input_scores, logit_effects, top_k=1)
    repeated = build_attribution_edges(input_scores, logit_effects, top_k=1)

    assert len(edges) == 2
    assert edges[0].source_type == "input"
    assert edges[0].target_type == "feature"
    assert edges[0].target_id == 1
    assert edges[1].source_type == "feature"
    assert edges[1].target_type == "logit_diff"
    assert edges[1].source_id == 1
    assert graph_reproducible(edges, repeated)


def test_attribution_graph_report_checks_preservation_and_damage_control():
    contributions = t.tensor([0.7, 0.2, 0.05, 0.05])
    report = attribution_graph_report(
        contributions,
        graph_feature_ids=[0, 1],
        random_feature_ids=[2, 3],
        num_nodes=6,
        num_edges=4,
        reproducible=True,
    )

    assert report.full_logit_diff == pytest.approx(1.0)
    assert report.graph_logit_diff == pytest.approx(0.9)
    assert report.ablated_logit_diff == pytest.approx(0.1)
    assert report.random_ablated_logit_diff == pytest.approx(0.9)
    assert report.topk_damage == pytest.approx(0.9)
    assert report.random_damage == pytest.approx(0.1)
    assert report.preserves_logit_diff
    assert report.passes_damage_control
    assert report.reproducible
