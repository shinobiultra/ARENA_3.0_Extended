import importlib.util

import pytest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is not installed")

if TORCH_AVAILABLE:
    import torch as t

    from arena_ext.jepa_world_models import (
        causal_latent_patch_report,
        collapse_diagnostics_report,
        jepa_prediction_report,
        latent_rollout_report,
        loss_decrease_report,
        object_permanence_report,
        transition_consistency_report,
        world_state_probe_report,
    )


def test_jepa_prediction_report_requires_cosine_and_mse():
    target_embeddings = t.tensor([[1.0, 0.0], [0.0, 1.0]])
    predicted_targets = target_embeddings.clone()

    report = jepa_prediction_report(
        predicted_targets,
        target_embeddings,
        min_cosine=0.99,
        max_mse=0.01,
    )

    assert report.mean_cosine == pytest.approx(1.0)
    assert report.mse == pytest.approx(0.0)
    assert report.predicts_target


def test_world_state_probe_report_checks_heldout_accuracy():
    probe_logits = t.tensor([[3.0, 0.0], [0.0, 4.0], [2.0, 1.0]])
    labels = t.tensor([0, 1, 0])

    report = world_state_probe_report(
        probe_logits,
        labels,
        min_accuracy=1.0,
    )

    assert report.accuracy == 1.0
    assert report.predicts_state


def test_transition_consistency_report_checks_action_deltas():
    state_embeddings = t.tensor([[0.0, 0.0], [1.0, 1.0]])
    action_deltas = t.tensor([[1.0, 0.0], [0.0, 1.0]])
    next_state_embeddings = t.tensor([[1.0, 0.0], [1.0, 2.0]])

    report = transition_consistency_report(
        state_embeddings,
        action_deltas,
        next_state_embeddings,
        min_cosine=0.99,
    )

    assert report.mean_cosine == pytest.approx(1.0)
    assert report.transition_consistent


def test_object_permanence_report_separates_occluded_from_absent():
    report = object_permanence_report(
        visible_scores=t.tensor([0.9, 0.8]),
        occluded_scores=t.tensor([0.75, 0.7]),
        absent_scores=t.tensor([0.1, 0.2]),
        min_occluded_score=0.6,
        min_absent_gap=0.4,
    )

    assert report.visible_mean == pytest.approx(0.85)
    assert report.occluded_mean == pytest.approx(0.725)
    assert report.absent_mean == pytest.approx(0.15)
    assert report.preserves_occluded_object


def test_loss_decrease_report_requires_baseline_improvement():
    report = loss_decrease_report(
        initial_loss=1.0,
        final_loss=0.2,
        baseline_loss=0.5,
        min_relative_reduction=0.5,
        max_final_to_baseline=0.8,
    )

    assert report.relative_reduction == pytest.approx(0.8)
    assert report.beats_baseline
    assert report.loss_decreases

    weak = loss_decrease_report(
        initial_loss=1.0,
        final_loss=0.45,
        baseline_loss=0.5,
        min_relative_reduction=0.5,
        max_final_to_baseline=0.8,
    )
    assert not weak.beats_baseline
    assert not weak.loss_decreases


def test_collapse_diagnostics_rejects_constant_features():
    varied = t.eye(4)
    report = collapse_diagnostics_report(
        varied,
        min_feature_std=0.1,
        min_effective_rank=2.0,
    )
    assert report.finite_features
    assert report.non_collapsed

    collapsed = collapse_diagnostics_report(
        t.ones(4, 4),
        min_feature_std=0.1,
        min_effective_rank=2.0,
    )
    assert not collapsed.non_collapsed


def test_latent_rollout_report_requires_copy_and_shuffled_controls():
    report = latent_rollout_report(
        rollout_loss=0.1,
        copy_baseline_loss=0.3,
        shuffled_action_loss=0.25,
        max_rollout_to_copy=0.8,
        max_rollout_to_shuffled=0.8,
    )

    assert report.beats_copy_baseline
    assert report.shuffled_action_fails
    assert report.rollout_passes

    weak = latent_rollout_report(
        rollout_loss=0.24,
        copy_baseline_loss=0.3,
        shuffled_action_loss=0.25,
        max_rollout_to_copy=0.8,
        max_rollout_to_shuffled=0.8,
    )
    assert weak.beats_copy_baseline
    assert not weak.shuffled_action_fails
    assert not weak.rollout_passes


def test_causal_latent_patch_report_compares_random_token_patch():
    report = causal_latent_patch_report(
        object_patch_effects=t.tensor([0.8, 0.7]),
        random_patch_effects=t.tensor([0.1, 0.05]),
        min_object_patch_effect=0.5,
        min_patch_random_gap=0.4,
    )

    assert report.object_patch_effect == pytest.approx(0.75)
    assert report.random_patch_effect == pytest.approx(0.075)
    assert report.causal_patch_passes
