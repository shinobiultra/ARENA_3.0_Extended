import importlib.util

import pytest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is not installed")

if TORCH_AVAILABLE:
    import torch as t

    from arena_ext.activation_patching import (
        activation_patching_sweep,
        answer_logit_diff,
        patch_activation_slice,
        patching_localization_report,
        patching_recovery_report,
        random_patch_control_report,
        recovery_fraction,
    )


def test_answer_logit_diff_returns_mean_diff():
    logits = t.tensor([[4.0, 1.0], [3.0, 2.0]])

    assert answer_logit_diff(logits, positive_token_id=0, negative_token_id=1) == 2.0


def test_patch_activation_slice_replaces_one_component():
    clean = t.tensor([[10.0, 20.0], [30.0, 40.0]])
    corrupt = t.zeros_like(clean)

    patched = patch_activation_slice(
        clean,
        corrupt,
        component_index=1,
        component_dim=0,
    )

    assert patched.tolist() == [[0.0, 0.0], [30.0, 40.0]]


def test_patching_recovery_report_measures_logit_diff_recovery():
    clean_logits = t.tensor([4.0, 1.0])
    corrupt_logits = t.tensor([1.0, 3.0])
    patched_logits = t.tensor([3.0, 1.0])

    report = patching_recovery_report(
        clean_logits,
        corrupt_logits,
        patched_logits,
        positive_token_id=0,
        negative_token_id=1,
        min_recovered_fraction=0.75,
    )

    assert report.clean_metric == 3.0
    assert report.corrupt_metric == -2.0
    assert report.patched_metric == 2.0
    assert report.recovered_fraction == pytest.approx(0.8)
    assert report.passes_recovery


def test_activation_patching_sweep_scores_components():
    patched_metrics = t.tensor([-1.0, 2.0, 0.0])

    sweep = activation_patching_sweep(
        clean_metric=3.0,
        corrupt_metric=-2.0,
        patched_metrics=patched_metrics,
    )

    assert sweep.patch_scores.tolist() == pytest.approx([0.2, 0.8, 0.4])
    assert sweep.best_index == 1
    assert sweep.best_score == pytest.approx(0.8)
    assert recovery_fraction(clean_metric=3.0, corrupt_metric=-2.0, patched_metric=2.0) == 0.8


def test_patching_localization_report_recovers_target_components():
    patch_scores = t.tensor([0.2, 0.9, 0.8, 0.1])

    report = patching_localization_report(
        patch_scores,
        target_indices=[1, 2],
        top_k=2,
        min_overlap=1.0,
    )

    assert report.top_indices == (1, 2)
    assert report.target_indices == (1, 2)
    assert report.topk_overlap == 1.0
    assert report.localizes_target


def test_random_patch_control_report_requires_top_to_win():
    patch_scores = t.tensor([0.2, 0.9, 0.8, 0.1])

    report = random_patch_control_report(
        patch_scores,
        random_indices=[0, 3],
        top_k=2,
    )

    assert report.top_patch_score == pytest.approx(0.85)
    assert report.random_patch_score == pytest.approx(0.15)
    assert report.top_beats_random
