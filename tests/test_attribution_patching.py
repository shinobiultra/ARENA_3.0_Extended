import importlib.util

import pytest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is not installed")

if TORCH_AVAILABLE:
    import torch as t

    from arena_ext.attribution_patching import (
        attribution_patch_scores,
        edge_attribution_scores,
        false_negative_report,
        integrated_gradient_patch_scores,
        runtime_improvement_report,
        score_correlation_report,
        topk_overlap_report,
    )


def test_attribution_patch_scores_sum_component_contributions():
    clean = t.tensor([[2.0, 0.0], [0.0, 3.0]])
    corrupt = t.zeros_like(clean)
    gradients = t.tensor([[0.5, 0.5], [1.0, 1.0]])

    scores = attribution_patch_scores(clean, corrupt, gradients)

    assert scores.tolist() == [1.0, 3.0]


def test_integrated_gradient_patch_scores_average_path_gradients():
    clean = t.tensor([[2.0, 0.0], [0.0, 3.0]])
    corrupt = t.zeros_like(clean)
    path_gradients = t.tensor(
        [
            [[0.25, 0.25], [0.5, 0.5]],
            [[0.75, 0.75], [1.5, 1.5]],
        ]
    )

    scores = integrated_gradient_patch_scores(clean, corrupt, path_gradients)

    assert scores.tolist() == [1.0, 3.0]


def test_edge_attribution_scores_returns_upstream_downstream_matrix():
    upstream_delta = t.tensor([[1.0, 0.0], [0.0, 2.0]])
    downstream_gradients = t.tensor([[3.0, 0.0], [0.0, 4.0]])

    scores = edge_attribution_scores(upstream_delta, downstream_gradients)

    assert scores.tolist() == [[3.0, 0.0], [0.0, 8.0]]


def test_score_correlation_report_passes_for_matching_rankings():
    exact = t.tensor([0.1, 0.9, 0.8, 0.0])
    approx = t.tensor([0.2, 0.85, 0.7, 0.1])

    report = score_correlation_report(exact, approx, min_correlation=0.95)

    assert report.correlation > 0.95
    assert report.passes_threshold


def test_topk_overlap_report_finds_shared_top_components():
    exact = t.tensor([0.1, 0.9, 0.8, 0.0])
    approx = t.tensor([0.2, 0.85, 0.7, 0.1])

    report = topk_overlap_report(exact, approx, top_k=2, min_overlap=1.0)

    assert report.exact_top_indices == (1, 2)
    assert report.approx_top_indices == (1, 2)
    assert report.topk_overlap == 1.0
    assert report.passes_threshold


def test_runtime_improvement_report_measures_speedup():
    report = runtime_improvement_report(
        exact_runtime_s=10.0,
        approx_runtime_s=2.0,
        min_speedup=4.0,
    )

    assert report.speedup == 5.0
    assert report.passes_speedup


def test_false_negative_report_requires_documentation():
    exact = t.tensor([0.1, 0.9, 0.8])
    approx = t.tensor([0.1, 0.2, 0.7])

    report = false_negative_report(
        exact,
        approx,
        exact_threshold=0.75,
        approx_threshold=0.5,
        documentation={1: "Approximation misses a nonlinear interaction."},
    )

    assert report.false_negative_indices == (1,)
    assert report.num_false_negatives == 1
    assert report.documented
