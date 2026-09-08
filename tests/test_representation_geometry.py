import importlib.util

import pytest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is not installed")

if TORCH_AVAILABLE:
    import torch as t

    from arena_ext.representation_geometry import (
        direction_causal_effect_report,
        geometry_label_prediction_report,
        geometry_stability_report,
        heldout_knn_accuracy_report,
        knn_label_predictions,
        neighborhood_preservation_report,
        nearest_centroid_predictions,
        pca_svd_projection,
        template_center_activations,
        visualization_sweep_report,
        white_noise_control_report,
    )


def test_pca_svd_projection_returns_expected_shapes():
    activations = t.tensor(
        [
            [2.0, 0.0],
            [1.0, 0.0],
            [-1.0, 0.0],
            [-2.0, 0.0],
        ]
    )

    projection = pca_svd_projection(activations, n_components=1)

    assert projection.projected.shape == (4, 1)
    assert projection.components.shape == (1, 2)
    assert projection.explained_variance_ratio.tolist() == pytest.approx([1.0])


def test_nearest_centroid_predictions_classifies_heldout_points():
    train_points = t.tensor([[0.0, 0.0], [0.2, 0.0], [2.0, 0.0], [2.2, 0.0]])
    train_labels = t.tensor([0, 0, 1, 1])
    heldout_points = t.tensor([[0.1, 0.0], [2.1, 0.0]])

    predictions = nearest_centroid_predictions(train_points, train_labels, heldout_points)

    assert predictions.tolist() == [0, 1]


def test_template_center_activations_removes_per_template_mean():
    activations = t.tensor(
        [
            [[10.0, 1.0], [12.0, 3.0]],
            [[-5.0, 2.0], [-1.0, 4.0]],
        ]
    )

    centered = template_center_activations(activations)

    assert centered.mean(dim=1).abs().max().item() == pytest.approx(0.0)
    assert centered[0].tolist() == [[-1.0, -1.0], [1.0, 1.0]]
    assert centered[1].tolist() == [[-2.0, -1.0], [2.0, 1.0]]


def test_geometry_label_prediction_report_requires_heldout_accuracy():
    train_points = t.tensor([[0.0, 0.0], [0.2, 0.0], [2.0, 0.0], [2.2, 0.0]])
    train_labels = t.tensor([0, 0, 1, 1])
    heldout_points = t.tensor([[0.1, 0.0], [2.1, 0.0]])
    heldout_labels = t.tensor([0, 1])

    report = geometry_label_prediction_report(
        train_points,
        train_labels,
        heldout_points,
        heldout_labels,
        min_accuracy=1.0,
    )

    assert report.heldout_accuracy == 1.0
    assert report.predicts_heldout_labels


def test_white_noise_control_report_requires_margin():
    report = white_noise_control_report(
        real_accuracy=0.9,
        noise_accuracy=0.5,
        min_margin=0.2,
    )

    assert report.margin == pytest.approx(0.4)
    assert report.survives_white_noise_control


def test_geometry_stability_report_uses_pairwise_jaccard():
    report = geometry_stability_report(
        [[1, 2, 3], [1, 2, 4], [1, 2, 3]],
        min_jaccard=0.5,
    )

    assert report.mean_pairwise_jaccard == pytest.approx(7 / 9)
    assert report.stable_across_seeds


def test_direction_causal_effect_report_requires_random_control_margin():
    baseline = t.tensor([0.2, 0.3])
    intervened = t.tensor([0.8, 0.7])
    random_control = t.tensor([0.35, 0.25])

    report = direction_causal_effect_report(
        baseline,
        intervened,
        random_control,
        expected_direction="increase",
        min_effect=0.4,
        min_random_margin=0.2,
    )

    assert report.observed_delta == pytest.approx(0.5)
    assert report.random_delta == pytest.approx(0.05)
    assert report.has_causal_effect


def test_knn_label_predictions_classifies_heldout_points():
    train_points = t.tensor([[0.0, 0.0], [0.2, 0.0], [2.0, 0.0], [2.2, 0.0]])
    train_labels = t.tensor([0, 0, 1, 1])
    heldout_points = t.tensor([[0.1, 0.0], [2.1, 0.0]])
    heldout_labels = t.tensor([0, 1])

    predictions = knn_label_predictions(train_points, train_labels, heldout_points, k=3)
    report = heldout_knn_accuracy_report(
        train_points,
        train_labels,
        heldout_points,
        heldout_labels,
        k=3,
        min_accuracy=1.0,
    )

    assert predictions.tolist() == [0, 1]
    assert report.heldout_accuracy == pytest.approx(1.0)
    assert report.predicts_heldout_labels


def test_neighborhood_preservation_report_compares_high_and_low_neighbors():
    high_dim = t.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.1, 0.0, 0.0],
        ]
    )
    low_dim = high_dim[:, :2]
    scrambled = low_dim[t.tensor([0, 2, 1, 3])]

    report = neighborhood_preservation_report(high_dim, low_dim, k=1, min_overlap=1.0)
    weak = neighborhood_preservation_report(high_dim, scrambled, k=1, min_overlap=1.0)

    assert report.mean_neighbor_overlap == pytest.approx(1.0)
    assert report.preserves_neighborhoods
    assert not weak.preserves_neighborhoods


def test_visualization_sweep_report_requires_seeds_settings_and_controls():
    report = visualization_sweep_report(
        seed_count=5,
        setting_count=3,
        heldout_knn_accuracies=[0.9, 0.85],
        trustworthiness_scores=[0.95, 0.93],
        neighborhood_preservation_scores=[0.7, 0.65],
        random_label_accuracies=[0.1, 0.2],
        random_token_accuracies=[0.0, 0.1],
    )

    assert report.run_count == 2
    assert report.min_heldout_knn_accuracy == pytest.approx(0.85)
    assert report.random_label_accuracy_max == pytest.approx(0.2)
    assert report.passes_visualization_controls

    too_few_seeds = visualization_sweep_report(
        seed_count=4,
        setting_count=3,
        heldout_knn_accuracies=[0.9],
        trustworthiness_scores=[0.95],
        neighborhood_preservation_scores=[0.7],
        random_label_accuracies=[0.1],
        random_token_accuracies=[0.1],
    )
    assert not too_few_seeds.passes_visualization_controls

    weak_control = visualization_sweep_report(
        seed_count=5,
        setting_count=3,
        heldout_knn_accuracies=[0.9],
        trustworthiness_scores=[0.95],
        neighborhood_preservation_scores=[0.7],
        random_label_accuracies=[0.6],
        random_token_accuracies=[0.1],
    )
    assert not weak_control.passes_visualization_controls
