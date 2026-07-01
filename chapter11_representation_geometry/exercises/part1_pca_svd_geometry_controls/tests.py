from collections.abc import Callable
import json
from pathlib import Path

import torch as t

from arena_ext import representation_geometry as reference


def _solutions():
    from chapter11_representation_geometry.exercises.part1_pca_svd_geometry_controls import (
        solutions,
    )

    return solutions


def _as_dict(report: object) -> dict:
    return report.__dict__


def _assert_report_close(actual: object, expected: object, *, msg: str) -> None:
    actual_dict = _as_dict(actual)
    expected_dict = _as_dict(expected)
    assert actual_dict.keys() == expected_dict.keys(), (
        f"{msg} should expose the same fields as the independent reference."
    )
    for key, expected_value in expected_dict.items():
        actual_value = actual_dict[key]
        if isinstance(expected_value, float):
            assert abs(actual_value - expected_value) < 1e-6, (
                f"{msg} field {key!r} should be {expected_value}, got {actual_value}."
            )
        else:
            assert actual_value == expected_value, (
                f"{msg} field {key!r} should be {expected_value!r}, got {actual_value!r}."
            )


def test_pca_svd_projection_centers_and_reports_variance(
    pca_svd_projection: Callable | None = None,
):
    pca_svd_projection = pca_svd_projection or _solutions().pca_svd_projection
    activations = t.tensor(
        [
            [2.0, 0.0],
            [1.0, 0.0],
            [-1.0, 0.0],
            [-2.0, 0.0],
        ]
    )
    projection = pca_svd_projection(activations, n_components=1)
    expected = reference.pca_svd_projection(activations, n_components=1)

    assert projection.projected.shape == (4, 1), (
        "PCA should return one projected coordinate per example."
    )
    assert projection.components.shape == (1, 2), (
        "PCA should return one component vector in the original activation space."
    )
    assert t.allclose(
        projection.explained_variance_ratio,
        expected.explained_variance_ratio,
        atol=1e-6,
    ), "Explained variance should be singular_value^2 divided by total variance."
    assert abs(float(projection.explained_variance_ratio[0]) - 1.0) < 1e-6, (
        "The toy activations vary only along one direction, so PC1 should explain all variance."
    )
    reconstructed = projection.projected @ projection.components
    centered = activations.float() - activations.float().mean(dim=0, keepdim=True)
    assert t.allclose(reconstructed.abs(), centered.abs(), atol=1e-6), (
        "Projected coordinates and components should reconstruct the centered rank-1 data up to SVD sign."
    )

    zero_projection = pca_svd_projection(t.ones(3, 2), n_components=1)
    assert zero_projection.explained_variance_ratio.tolist() == [0.0], (
        "Zero-variance activations should report zero explained variance, not NaN."
    )
    try:
        pca_svd_projection(activations, n_components=3)
    except ValueError as exc:
        assert "n_components" in str(exc), (
            "Requesting too many components should raise a helpful n_components error."
        )
    else:
        raise AssertionError("Too many PCA components should raise ValueError.")
    print("All tests in `test_pca_svd_projection_centers_and_reports_variance` passed!")


def test_geometry_label_prediction_report_uses_heldout_centroids(
    geometry_label_prediction_report: Callable | None = None,
):
    geometry_label_prediction_report = (
        geometry_label_prediction_report or _solutions().geometry_label_prediction_report
    )
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
    expected = reference.geometry_label_prediction_report(
        train_points,
        train_labels,
        heldout_points,
        heldout_labels,
        min_accuracy=1.0,
    )
    _assert_report_close(report, expected, msg="Held-out label prediction report")
    assert report.heldout_accuracy == 1.0 and report.predicts_heldout_labels, (
        "Nearest-centroid geometry should correctly classify both held-out points."
    )

    wrong_labels = t.tensor([1, 0])
    failed = geometry_label_prediction_report(
        train_points,
        train_labels,
        heldout_points,
        wrong_labels,
        min_accuracy=1.0,
    )
    assert not failed.predicts_heldout_labels, (
        "The report should fail when held-out labels disagree with the geometry."
    )
    try:
        geometry_label_prediction_report(
            train_points,
            train_labels[:-1],
            heldout_points,
            heldout_labels,
        )
    except ValueError as exc:
        assert "train_labels" in str(exc), (
            "Mismatched train labels should raise an explicit train_labels error."
        )
    else:
        raise AssertionError("Mismatched train labels should raise ValueError.")
    print("All tests in `test_geometry_label_prediction_report_uses_heldout_centroids` passed!")


def test_white_noise_control_report_requires_margin(
    white_noise_control_report: Callable | None = None,
):
    white_noise_control_report = (
        white_noise_control_report or _solutions().white_noise_control_report
    )
    report = white_noise_control_report(
        real_accuracy=0.9,
        noise_accuracy=0.5,
        min_margin=0.2,
    )
    expected = reference.white_noise_control_report(
        real_accuracy=0.9,
        noise_accuracy=0.5,
        min_margin=0.2,
    )
    _assert_report_close(report, expected, msg="White-noise control report")
    assert abs(report.margin - 0.4) < 1e-6 and report.survives_white_noise_control, (
        "Real geometry should pass when it beats white noise by the required margin."
    )
    weak = white_noise_control_report(
        real_accuracy=0.62,
        noise_accuracy=0.5,
        min_margin=0.2,
    )
    assert not weak.survives_white_noise_control, (
        "A small real-minus-noise margin should fail the white-noise control."
    )
    print("All tests in `test_white_noise_control_report_requires_margin` passed!")


def test_geometry_stability_report_averages_pairwise_jaccard(
    geometry_stability_report: Callable | None = None,
):
    geometry_stability_report = (
        geometry_stability_report or _solutions().geometry_stability_report
    )
    neighbor_sets = [[1, 2, 3], [1, 2, 4], [1, 2, 3]]
    report = geometry_stability_report(neighbor_sets, min_jaccard=0.5)
    expected = reference.geometry_stability_report(neighbor_sets, min_jaccard=0.5)
    _assert_report_close(report, expected, msg="Geometry stability report")
    assert abs(report.mean_pairwise_jaccard - (7 / 9)) < 1e-6, (
        "Stability should average all pairwise Jaccard overlaps."
    )
    assert report.stable_across_seeds, (
        "The toy neighbor sets should pass a 0.5 mean-Jaccard threshold."
    )
    unstable = geometry_stability_report([[1, 2], [3, 4]], min_jaccard=0.5)
    assert not unstable.stable_across_seeds, (
        "Disjoint neighbor sets should fail the stability control."
    )
    try:
        geometry_stability_report([])
    except ValueError as exc:
        assert "nonempty" in str(exc), (
            "Empty stability inputs should raise a nonempty-neighbor-set error."
        )
    else:
        raise AssertionError("Empty neighbor sets should raise ValueError.")
    print("All tests in `test_geometry_stability_report_averages_pairwise_jaccard` passed!")


def test_knn_label_prediction_report_uses_heldout_points(
    heldout_knn_accuracy_report: Callable | None = None,
):
    heldout_knn_accuracy_report = (
        heldout_knn_accuracy_report or _solutions().heldout_knn_accuracy_report
    )
    train_points = t.tensor([[0.0, 0.0], [0.2, 0.0], [2.0, 0.0], [2.2, 0.0]])
    train_labels = t.tensor([0, 0, 1, 1])
    heldout_points = t.tensor([[0.1, 0.0], [2.1, 0.0]])
    heldout_labels = t.tensor([0, 1])
    report = heldout_knn_accuracy_report(
        train_points,
        train_labels,
        heldout_points,
        heldout_labels,
        k=3,
        min_accuracy=1.0,
    )
    expected = reference.heldout_knn_accuracy_report(
        train_points,
        train_labels,
        heldout_points,
        heldout_labels,
        k=3,
        min_accuracy=1.0,
    )
    _assert_report_close(report, expected, msg="Held-out kNN prediction report")
    assert report.predicts_heldout_labels, (
        "The kNN report should pass when held-out points are nearest to same-label train clusters."
    )
    print("All tests in `test_knn_label_prediction_report_uses_heldout_points` passed!")


def test_neighborhood_preservation_report_compares_neighbor_sets(
    neighborhood_preservation_report: Callable | None = None,
):
    neighborhood_preservation_report = (
        neighborhood_preservation_report or _solutions().neighborhood_preservation_report
    )
    high_dim = t.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.1, 0.0, 0.0],
        ]
    )
    low_dim = high_dim[:, :2]
    report = neighborhood_preservation_report(high_dim, low_dim, k=1, min_overlap=1.0)
    expected = reference.neighborhood_preservation_report(
        high_dim,
        low_dim,
        k=1,
        min_overlap=1.0,
    )
    _assert_report_close(report, expected, msg="Neighborhood preservation report")
    assert report.preserves_neighborhoods, (
        "Identical high-dimensional and low-dimensional neighbor order should pass preservation."
    )
    print("All tests in `test_neighborhood_preservation_report_compares_neighbor_sets` passed!")


def test_visualization_sweep_report_requires_controls(
    visualization_sweep_report: Callable | None = None,
):
    visualization_sweep_report = (
        visualization_sweep_report or _solutions().visualization_sweep_report
    )
    report = visualization_sweep_report(
        seed_count=5,
        setting_count=3,
        heldout_knn_accuracies=[0.9, 0.85],
        trustworthiness_scores=[0.95, 0.93],
        neighborhood_preservation_scores=[0.7, 0.65],
        random_label_accuracies=[0.1, 0.2],
        random_token_accuracies=[0.0, 0.1],
    )
    expected = reference.visualization_sweep_report(
        seed_count=5,
        setting_count=3,
        heldout_knn_accuracies=[0.9, 0.85],
        trustworthiness_scores=[0.95, 0.93],
        neighborhood_preservation_scores=[0.7, 0.65],
        random_label_accuracies=[0.1, 0.2],
        random_token_accuracies=[0.0, 0.1],
    )
    _assert_report_close(report, expected, msg="Visualization sweep report")
    assert report.passes_visualization_controls, (
        "A sweep with enough seeds/settings, accurate held-out kNN, and weak controls should pass."
    )

    weak_control = visualization_sweep_report(
        seed_count=5,
        setting_count=3,
        heldout_knn_accuracies=[0.9],
        trustworthiness_scores=[0.95],
        neighborhood_preservation_scores=[0.7],
        random_label_accuracies=[0.6],
        random_token_accuracies=[0.1],
    )
    assert not weak_control.passes_visualization_controls, (
        "A sweep should fail when the random-label control predicts held-out labels too well."
    )
    print("All tests in `test_visualization_sweep_report_requires_controls` passed!")


def test_direction_causal_effect_report_beats_random_control(
    direction_causal_effect_report: Callable | None = None,
):
    direction_causal_effect_report = (
        direction_causal_effect_report or _solutions().direction_causal_effect_report
    )
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
    expected = reference.direction_causal_effect_report(
        baseline,
        intervened,
        random_control,
        expected_direction="increase",
        min_effect=0.4,
        min_random_margin=0.2,
    )
    _assert_report_close(report, expected, msg="Causal direction report")
    assert abs(report.observed_delta - 0.5) < 1e-6 and report.has_causal_effect, (
        "The intervention should move the score in the expected direction and beat the random control."
    )
    weak_random_margin = direction_causal_effect_report(
        baseline,
        intervened,
        t.tensor([0.7, 0.6]),
        expected_direction="increase",
        min_effect=0.4,
        min_random_margin=0.2,
    )
    assert not weak_random_margin.has_causal_effect, (
        "A direction should fail when the random control explains nearly the same delta."
    )
    decreased = direction_causal_effect_report(
        t.tensor([0.8, 0.7]),
        t.tensor([0.2, 0.3]),
        t.tensor([0.7, 0.8]),
        expected_direction="decrease",
        min_effect=0.4,
        min_random_margin=0.2,
    )
    assert decreased.has_causal_effect, (
        "The report should also handle expected decreases, not only increases."
    )
    try:
        direction_causal_effect_report(
            baseline,
            intervened,
            random_control,
            expected_direction="flat",
        )
    except ValueError as exc:
        assert "increase" in str(exc) and "decrease" in str(exc), (
            "Invalid direction errors should name the allowed directions."
        )
    else:
        raise AssertionError("Invalid causal direction should raise ValueError.")
    print("All tests in `test_direction_causal_effect_report_beats_random_control` passed!")


def test_template_center_activations_removes_each_template_mean(
    template_center_activations: Callable | None = None,
):
    template_center_activations = (
        template_center_activations or _solutions().template_center_activations
    )
    activations = t.tensor(
        [
            [[10.0, 1.0], [12.0, 3.0]],
            [[-5.0, 2.0], [-1.0, 4.0]],
        ]
    )
    centered = template_center_activations(activations)
    expected = reference.template_center_activations(activations)
    assert centered.shape == activations.shape, (
        "Template centering should preserve the activation tensor shape."
    )
    assert t.allclose(centered, expected, atol=1e-6), (
        "Template centering should subtract each template's own example mean."
    )
    assert centered.mean(dim=1).abs().max().item() == 0.0, (
        "Each template batch should have zero mean after centering."
    )
    assert centered[0].tolist() == [[-1.0, -1.0], [1.0, 1.0]], (
        "The first template should be centered around its own two examples."
    )
    try:
        template_center_activations(t.ones(2, 3))
    except ValueError as exc:
        assert "templates" in str(exc), (
            "Wrong-rank template centering inputs should raise a template-shape error."
        )
    else:
        raise AssertionError("Wrong-rank template activations should raise ValueError.")
    print("All tests in `test_template_center_activations_removes_each_template_mean` passed!")


def test_notebook_contract(run_smoke_test: Callable | None = None):
    run_smoke_test = run_smoke_test or _solutions().run_smoke_test
    result = run_smoke_test(cpu=True)
    assert result["pca"]["projected_shape"] == [4, 1], (
        "The notebook contract should include the PCA projected shape."
    )
    assert result["pca"]["explained_variance_ratio"] == [1.0], (
        "The notebook contract should include one fully explanatory toy component."
    )
    assert result["prediction"]["predicts_heldout_labels"], (
        "The notebook contract should include a passing held-out prediction check."
    )
    assert result["noise_control"]["survives_white_noise_control"], (
        "The notebook contract should include a passing white-noise control."
    )
    assert result["stability"]["stable_across_seeds"], (
        "The notebook contract should include a passing stability control."
    )
    assert result["causal_direction"]["has_causal_effect"], (
        "The notebook contract should include a passing causal-direction control."
    )
    assert result["template_centering"]["max_template_mean_abs"] == 0.0, (
        "The notebook contract should include exact zero template means after centering."
    )
    print("All tests in `test_notebook_contract` passed!")


def test_committed_verification_report_has_visualization_sweep_controls():
    report_path = Path(__file__).with_name("verification_report.json")
    report = json.loads(report_path.read_text())
    gpu = report["metrics"]["gpu_test"]

    assert gpu["pythia_visualization_seed_count"] >= 5, (
        "The report should include at least five reducer seeds."
    )
    assert gpu["pythia_visualization_setting_count"] >= 3, (
        "The report should include at least three reducer hyperparameter settings."
    )
    assert gpu["pythia_weekday_visualization_passed"], (
        "The weekday visualization sweep must pass held-out and control checks."
    )
    assert gpu["pythia_month_visualization_passed"], (
        "The month visualization sweep must pass held-out and control checks."
    )
    assert gpu["pythia_weekday_umap_min_heldout_knn_accuracy"] >= 0.8, (
        "Weekday UMAP projections should predict held-out calendar labels by kNN."
    )
    assert gpu["pythia_month_umap_min_heldout_knn_accuracy"] >= 0.8, (
        "Month UMAP projections should predict held-out calendar labels by kNN."
    )
    assert gpu["pythia_weekday_umap_min_trustworthiness"] >= 0.9, (
        "Weekday UMAP projections should preserve local high-dimensional neighborhoods."
    )
    assert gpu["pythia_month_umap_min_trustworthiness"] >= 0.9, (
        "Month UMAP projections should preserve local high-dimensional neighborhoods."
    )
    assert gpu["pythia_weekday_umap_random_label_accuracy_max"] <= 0.35, (
        "Weekday shuffled-label controls should not explain the held-out geometry."
    )
    assert gpu["pythia_month_umap_random_label_accuracy_max"] <= 0.35, (
        "Month shuffled-label controls should not explain the held-out geometry."
    )
    assert gpu["pythia_weekday_umap_random_token_accuracy_max"] <= 0.35, (
        "Weekday matched random-token controls should fail the held-out calendar labels."
    )
    assert gpu["pythia_month_umap_random_token_accuracy_max"] <= 0.35, (
        "Month matched random-token controls should fail the held-out calendar labels."
    )
    print(
        "All tests in `test_committed_verification_report_has_visualization_sweep_controls` passed!"
    )


def test_exercise_notebook_declares_full_verification_contract():
    notebook_path = Path(__file__).with_name(
        "11.1_PCA_SVD_and_Geometry_Controls_exercises.ipynb"
    )
    notebook = json.loads(notebook_path.read_text())
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
    )

    assert "REQUIRES_GPU = True" in source, (
        "The learner notebook should not advertise CPU-only scope for this GT-1 geometry section."
    )
    assert "def run_smoke_test(cpu: bool = True)" in source, (
        "The learner notebook should expose the CPU contract surface."
    )
    assert "def run_gpu_test(max_vram_gb: float = 24.0)" in source, (
        "The learner notebook should expose the GPU verification surface."
    )
    assert "def run_full_experiment(max_vram_gb: float = 24.0)" in source, (
        "The learner notebook should expose the full experiment surface."
    )
    assert "test_committed_verification_report_has_visualization_sweep_controls" in source, (
        "The learner notebook should end by checking the committed Pythia geometry report."
    )
    print("All tests in `test_exercise_notebook_declares_full_verification_contract` passed!")
