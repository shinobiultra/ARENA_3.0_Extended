from collections.abc import Callable
import importlib.util
import json
from pathlib import Path

import pytest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is not installed")

if TORCH_AVAILABLE:
    import torch as t

    from arena_ext import sparse_feature_circuits as reference


def _solutions():
    from chapter8_automated_circuits.exercises.part5_sparse_feature_circuits import (
        solutions,
    )

    return solutions


def _section_dir() -> Path:
    return Path(__file__).resolve().parent


def _gpu_report() -> dict:
    report = json.loads((_section_dir() / "verification_report.json").read_text())
    return report["metrics"]["gpu_test"]


def _assert_report_close(actual: object, expected: object, *, msg: str) -> None:
    actual_dict = actual.__dict__
    expected_dict = expected.__dict__
    assert actual_dict.keys() == expected_dict.keys(), (
        f"{msg} fields should match the independent reference implementation."
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


def test_encode_decode_shape_smoke_test(
    encode_decode_shape_smoke_test: Callable | None = None,
):
    encode_decode_shape_smoke_test = (
        encode_decode_shape_smoke_test or _solutions().encode_decode_shape_smoke_test
    )
    result = encode_decode_shape_smoke_test()
    assert result["feature_shape"] == [2, 2], (
        "Feature activations should keep the expected [batch, features] shape."
    )
    assert result["reconstructed_shape"] == [2, 2], (
        "Decoded activations should land back in the activation space."
    )
    assert result["matches_input"], (
        "The identity-decoder toy fixture should reconstruct the input exactly."
    )
    print("All tests in `test_encode_decode_shape_smoke_test` passed!")


def test_toy_sae_encode_decode_metrics_on_planted_fixture(
    toy_sae_encode: Callable | None = None,
    toy_sae_decode: Callable | None = None,
    sae_reconstruction_report: Callable | None = None,
):
    solutions = _solutions()
    toy_sae_encode = toy_sae_encode or solutions.toy_sae_encode
    toy_sae_decode = toy_sae_decode or solutions.toy_sae_decode
    sae_reconstruction_report = sae_reconstruction_report or solutions.sae_reconstruction_report
    fixture = solutions.planted_sparse_feature_circuit_fixture()
    activations = t.stack([fixture["clean_activation"], fixture["corrupt_activation"]])
    feature_acts = toy_sae_encode(
        activations,
        fixture["encoder_weight"],
        fixture["encoder_bias"],
    )
    reconstructions = toy_sae_decode(
        feature_acts,
        fixture["decoder_weight"],
        fixture["decoder_bias"],
    )
    report = sae_reconstruction_report(activations, feature_acts, reconstructions)
    assert tuple(feature_acts.shape) == (2, 6), (
        "The planted SAE should expose two examples and six sparse features."
    )
    assert report.reconstruction_mse == 0.0 and report.relative_l2_error == 0.0, (
        "The planted identity SAE should reconstruct exactly."
    )
    assert abs(report.l0_mean - 3.5) < 1e-6 and report.density > 0.5, (
        "The fixture should make sparsity visible rather than hiding dense activations."
    )
    assert abs(report.dead_feature_fraction - (1.0 / 6.0)) < 1e-6, (
        "One feature is deliberately dead so students can see the SAE metric."
    )
    assert feature_acts[0, 1] > feature_acts[1, 1], (
        "The clean prompt should activate the plural-subject feature more."
    )
    assert feature_acts[1, 2] > feature_acts[0, 2], (
        "The corrupt prompt should activate the singular-subject feature more."
    )
    try:
        toy_sae_encode(activations, fixture["encoder_weight"][:, :-1], fixture["encoder_bias"])
    except ValueError as exc:
        assert "activation dimension" in str(exc), (
            "Shape mistakes in the SAE encoder should fail before graph attribution."
        )
    else:
        raise AssertionError("Mismatched SAE encoder dimensions should raise ValueError.")
    print("All tests in `test_toy_sae_encode_decode_metrics_on_planted_fixture` passed!")


def test_exact_planted_node_patch_scores_identify_known_nodes(
    exact_planted_node_patch_scores: Callable | None = None,
    patch_planted_nodes: Callable | None = None,
):
    solutions = _solutions()
    exact_planted_node_patch_scores = (
        exact_planted_node_patch_scores or solutions.exact_planted_node_patch_scores
    )
    patch_planted_nodes = patch_planted_nodes or solutions.patch_planted_nodes
    fixture = solutions.planted_sparse_feature_circuit_fixture()
    clean_features = solutions.toy_sae_encode(
        fixture["clean_activation"],
        fixture["encoder_weight"],
        fixture["encoder_bias"],
    )
    corrupt_features = solutions.toy_sae_encode(
        fixture["corrupt_activation"],
        fixture["encoder_weight"],
        fixture["encoder_bias"],
    )
    clean_receiver = solutions.planted_receiver_activations(
        clean_features,
        fixture["edge_weights"],
    )
    corrupt_receiver = solutions.planted_receiver_activations(
        corrupt_features,
        fixture["edge_weights"],
    )
    node_scores = exact_planted_node_patch_scores(
        clean_receiver,
        corrupt_receiver,
        fixture["readout_weights"],
    )
    node_report = solutions.exact_feature_node_patching_report(
        node_scores,
        fixture["ground_truth_node_ids"],
        min_recovered_fraction=0.95,
    )
    clean_metric = solutions.planted_circuit_metric(clean_receiver, fixture["readout_weights"])
    corrupt_metric = solutions.planted_circuit_metric(corrupt_receiver, fixture["readout_weights"])
    patched_metric = patch_planted_nodes(
        corrupt_receiver,
        clean_receiver,
        fixture["readout_weights"],
        fixture["ground_truth_node_ids"],
    )
    recovered = (patched_metric - corrupt_metric) / (clean_metric - corrupt_metric)
    assert node_scores.argmax().item() == 0, (
        "The plural relay should be the largest exact node-patching score."
    )
    assert node_report.selected_feature_ids == (0, 1), (
        "The planted node graph should be the plural and singular relay nodes."
    )
    assert recovered > 0.98 and node_report.passes_recovery, (
        "Patching the two planted nodes should almost fully recover the clean metric."
    )
    try:
        patch_planted_nodes(corrupt_receiver, clean_receiver, fixture["readout_weights"], [0, 0])
    except ValueError as exc:
        assert "unique" in str(exc), (
            "Node patching must reject duplicate nodes instead of double-counting."
        )
    else:
        raise AssertionError("Duplicate node ids should raise ValueError.")
    print("All tests in `test_exact_planted_node_patch_scores_identify_known_nodes` passed!")


def test_exact_planted_edge_patch_scores_identify_known_edges(
    exact_planted_edge_patch_scores: Callable | None = None,
    patch_planted_edges: Callable | None = None,
):
    solutions = _solutions()
    exact_planted_edge_patch_scores = (
        exact_planted_edge_patch_scores or solutions.exact_planted_edge_patch_scores
    )
    patch_planted_edges = patch_planted_edges or solutions.patch_planted_edges
    fixture = solutions.planted_sparse_feature_circuit_fixture()
    clean_features = solutions.toy_sae_encode(
        fixture["clean_activation"],
        fixture["encoder_weight"],
        fixture["encoder_bias"],
    )
    corrupt_features = solutions.toy_sae_encode(
        fixture["corrupt_activation"],
        fixture["encoder_weight"],
        fixture["encoder_bias"],
    )
    edge_scores = exact_planted_edge_patch_scores(
        clean_features,
        corrupt_features,
        fixture["edge_weights"],
        fixture["readout_weights"],
    )
    edge_report = solutions.exact_feature_edge_patching_report(
        edge_scores,
        fixture["ground_truth_edges"],
        min_recovered_fraction=0.95,
    )
    clean_metric = solutions.planted_circuit_metric(
        solutions.planted_receiver_activations(clean_features, fixture["edge_weights"]),
        fixture["readout_weights"],
    )
    corrupt_metric = solutions.planted_circuit_metric(
        solutions.planted_receiver_activations(corrupt_features, fixture["edge_weights"]),
        fixture["readout_weights"],
    )
    patched_metric = patch_planted_edges(
        corrupt_features,
        clean_features,
        fixture["edge_weights"],
        fixture["readout_weights"],
        fixture["ground_truth_edges"],
    )
    recovered = (patched_metric - corrupt_metric) / (clean_metric - corrupt_metric)
    assert edge_scores[1, 0] > 1.0 and edge_scores[2, 1] > 0.7, (
        "The two planted edges should dominate the exact edge-patching table."
    )
    assert edge_report.selected_edges == ((1, 0), (2, 1)), (
        "The planted edge graph should be plural-source -> plural-relay and singular-source -> singular-relay."
    )
    assert recovered > 0.98 and edge_report.passes_recovery, (
        "Patching the two planted edges should almost fully recover the clean metric."
    )
    try:
        patch_planted_edges(
            corrupt_features,
            clean_features,
            fixture["edge_weights"],
            fixture["readout_weights"],
            [(99, 0)],
        )
    except ValueError as exc:
        assert "source id" in str(exc), (
            "Edge patching should identify an out-of-range source id."
        )
    else:
        raise AssertionError("Out-of-range source ids should raise ValueError.")
    print("All tests in `test_exact_planted_edge_patch_scores_identify_known_edges` passed!")


def test_nonlinear_eap_ig_beats_plain_attribution_patching(
    nonlinear_eap_ig_edge_attribution_report: Callable | None = None,
):
    solutions = _solutions()
    nonlinear_eap_ig_edge_attribution_report = (
        nonlinear_eap_ig_edge_attribution_report
        or solutions.nonlinear_eap_ig_edge_attribution_report
    )
    fixture = solutions.planted_sparse_feature_circuit_fixture()
    clean_features = solutions.toy_sae_encode(
        fixture["clean_activation"],
        fixture["encoder_weight"],
        fixture["encoder_bias"],
    )
    corrupt_features = solutions.toy_sae_encode(
        fixture["corrupt_activation"],
        fixture["encoder_weight"],
        fixture["encoder_bias"],
    )
    report = nonlinear_eap_ig_edge_attribution_report(
        clean_features,
        corrupt_features,
        fixture["edge_weights"],
        fixture["readout_weights"],
        steps=128,
    )
    assert report["eap_ig_error"] < report["eap_error"], (
        "EAP-IG should be closer than one-gradient EAP on the saturated readout."
    )
    assert abs(report["eap_ig_total"] - report["exact_total_effect"]) < 1e-4, (
        "The integrated path scores should add up to the nonlinear total effect."
    )
    assert abs(report["eap_total"] - report["exact_total_effect"]) > 0.2, (
        "Plain EAP should visibly miscalibrate the saturated toy graph."
    )
    try:
        nonlinear_eap_ig_edge_attribution_report(
            clean_features,
            corrupt_features,
            fixture["edge_weights"],
            fixture["readout_weights"],
            steps=1,
        )
    except ValueError as exc:
        assert "steps" in str(exc), "EAP-IG needs more than one integration step."
    else:
        raise AssertionError("A one-step EAP-IG path should raise ValueError.")
    print("All tests in `test_nonlinear_eap_ig_beats_plain_attribution_patching` passed!")


def test_threshold_graph_metrics_and_random_controls(
    threshold_sparse_feature_graph: Callable | None = None,
    faithfulness_minimality_completeness_report: Callable | None = None,
):
    solutions = _solutions()
    threshold_sparse_feature_graph = (
        threshold_sparse_feature_graph or solutions.threshold_sparse_feature_graph
    )
    faithfulness_minimality_completeness_report = (
        faithfulness_minimality_completeness_report
        or solutions.faithfulness_minimality_completeness_report
    )
    signature = solutions.run_planted_sparse_feature_signature()
    node_scores = t.tensor(signature["node_scores"])
    edge_scores = t.tensor(signature["edge_scores"])
    fixture = solutions.planted_sparse_feature_circuit_fixture()
    graph = threshold_sparse_feature_graph(
        node_scores,
        edge_scores,
        node_threshold=fixture["node_threshold"],
        edge_threshold=fixture["edge_threshold"],
        min_node_recovery=0.95,
        min_edge_recovery=0.95,
    )
    metrics = faithfulness_minimality_completeness_report(
        node_scores,
        graph["selected_node_ids"],
        fixture["same_size_random_node_ids"],
        min_faithfulness=0.95,
        min_random_margin=0.9,
    )
    assert graph["selected_node_ids"] == fixture["ground_truth_node_ids"], (
        "Thresholding should recover exactly the two planted receiver nodes."
    )
    assert graph["selected_edges"] == fixture["ground_truth_edges"], (
        "Thresholding should recover exactly the two planted source-to-receiver edges."
    )
    assert metrics["faithfulness"] > 0.98 and metrics["completeness"] > 0.98, (
        "The planted graph should preserve almost all of the exact node effect."
    )
    assert metrics["random_margin"] > 0.9 and metrics["passes"], (
        "A same-size random graph should visibly fail."
    )
    too_sparse = faithfulness_minimality_completeness_report(
        node_scores,
        [0],
        fixture["same_size_random_node_ids"][:1],
        min_faithfulness=0.95,
        min_random_margin=0.9,
    )
    assert not too_sparse["passes"], (
        "Dropping the singular relay should fail the faithfulness gate."
    )
    print("All tests in `test_threshold_graph_metrics_and_random_controls` passed!")


def test_planted_sparse_feature_signature_result_is_visual_ready(
    run_planted_sparse_feature_signature: Callable | None = None,
):
    run_planted_sparse_feature_signature = (
        run_planted_sparse_feature_signature
        or _solutions().run_planted_sparse_feature_signature
    )
    result = run_planted_sparse_feature_signature()
    assert result["accepted"], "The exact planted sparse-feature theorem should pass."
    assert result["claim_scope"].startswith("GT-0 exact planted"), (
        "The signature result should state the toy theorem boundary."
    )
    assert result["graph"]["selected_node_ids"] == (0, 1), (
        "The signature graph should expose the planted sparse nodes."
    )
    assert result["graph"]["selected_edges"] == ((1, 0), (2, 1)), (
        "The signature graph should expose the planted sparse edges."
    )
    rows = result["threshold_rows"]
    assert len(rows) >= 5 and rows[0]["num_features"] > rows[-1]["num_features"], (
        "The signature result should include threshold-curve rows for plotting."
    )
    assert result["random_control"]["random_graph_fails"], (
        "The signature result should include a failed same-size random graph."
    )
    print("All tests in `test_planted_sparse_feature_signature_result_is_visual_ready` passed!")


def test_exact_feature_node_patching_report_recovers_selected_features(
    exact_feature_node_patching_report: Callable | None = None,
):
    exact_feature_node_patching_report = (
        exact_feature_node_patching_report
        or _solutions().exact_feature_node_patching_report
    )
    contributions = t.tensor([0.7, 0.2, 0.05, 0.05])
    report = exact_feature_node_patching_report(
        contributions,
        [0, 1],
        min_recovered_fraction=0.8,
    )
    expected = reference.exact_feature_node_patching_report(
        contributions,
        [0, 1],
        min_recovered_fraction=0.8,
    )
    _assert_report_close(report, expected, msg="Exact node patching report")
    assert report.selected_feature_ids == (0, 1), (
        "Node patching should preserve the selected feature ids."
    )
    assert abs(report.graph_logit_diff - 0.9) < 1e-6 and report.passes_recovery, (
        "Selected features 0 and 1 should recover 90% of the toy effect."
    )
    try:
        exact_feature_node_patching_report(contributions, [], min_recovered_fraction=0.8)
    except ValueError as exc:
        assert "at least one" in str(exc), (
            "Empty feature selections should raise a helpful error."
        )
    else:
        raise AssertionError("Empty feature selections should raise ValueError.")
    print(
        "All tests in `test_exact_feature_node_patching_report_recovers_selected_features` passed!"
    )


def test_exact_feature_edge_patching_report_recovers_selected_edges(
    exact_feature_edge_patching_report: Callable | None = None,
):
    exact_feature_edge_patching_report = (
        exact_feature_edge_patching_report
        or _solutions().exact_feature_edge_patching_report
    )
    edge_scores = t.tensor([[0.05, 0.8], [0.05, 0.1]])
    report = exact_feature_edge_patching_report(
        edge_scores,
        [(0, 1)],
        min_recovered_fraction=0.75,
    )
    expected = reference.exact_feature_edge_patching_report(
        edge_scores,
        [(0, 1)],
        min_recovered_fraction=0.75,
    )
    _assert_report_close(report, expected, msg="Exact edge patching report")
    assert report.selected_edges == ((0, 1),), (
        "Edge patching should preserve selected source-feature pairs."
    )
    assert abs(report.graph_edge_score - 0.8) < 1e-6 and report.passes_recovery, (
        "The selected edge should recover 80% of the toy edge mass."
    )
    try:
        exact_feature_edge_patching_report(edge_scores, [(3, 0)])
    except ValueError as exc:
        assert "source id" in str(exc), (
            "Out-of-range source ids should raise a source-specific error."
        )
    else:
        raise AssertionError("Out-of-range source ids should raise ValueError.")
    print(
        "All tests in `test_exact_feature_edge_patching_report_recovers_selected_edges` passed!"
    )


def test_eap_ig_comparison_report_improves_over_plain_eap(
    eap_ig_comparison_report: Callable | None = None,
):
    eap_ig_comparison_report = (
        eap_ig_comparison_report or _solutions().eap_ig_comparison_report
    )
    exact = t.tensor([0.7, 0.2, 0.05, 0.05])
    eap = t.tensor([0.5, 0.35, 0.1, 0.05])
    eap_ig = t.tensor([0.69, 0.21, 0.04, 0.06])
    report = eap_ig_comparison_report(exact, eap, eap_ig, max_eap_ig_error=0.02)
    expected = reference.eap_ig_comparison_report(exact, eap, eap_ig, max_eap_ig_error=0.02)
    _assert_report_close(report, expected, msg="EAP-IG comparison report")
    assert report.eap_ig_error < report.eap_error and report.eap_ig_improves, (
        "EAP-IG should improve over plain EAP on this saturated toy fixture."
    )
    bad = eap_ig_comparison_report(exact, eap, eap, max_eap_ig_error=0.02)
    assert not bad.eap_ig_improves, (
        "EAP-IG should fail when it does not improve over plain EAP."
    )
    print("All tests in `test_eap_ig_comparison_report_improves_over_plain_eap` passed!")


def test_threshold_feature_graph_report_keeps_large_features(
    threshold_feature_graph_report: Callable | None = None,
):
    threshold_feature_graph_report = (
        threshold_feature_graph_report or _solutions().threshold_feature_graph_report
    )
    contributions = t.tensor([0.7, 0.2, 0.05, 0.05])
    report = threshold_feature_graph_report(
        contributions,
        threshold=0.2,
        min_recovered_fraction=0.8,
    )
    expected = reference.threshold_feature_graph_report(
        contributions,
        threshold=0.2,
        min_recovered_fraction=0.8,
    )
    _assert_report_close(report, expected, msg="Threshold feature graph report")
    assert report.selected_feature_ids == (0, 1), (
        "Thresholding at 0.2 should keep the two large toy features."
    )
    assert abs(report.recovered_fraction - 0.9) < 1e-6 and report.passes_threshold, (
        "The thresholded graph should preserve 90% of the toy effect."
    )
    try:
        threshold_feature_graph_report(contributions, threshold=2.0)
    except ValueError as exc:
        assert "selected no features" in str(exc), (
            "Overly high thresholds should report that no features were selected."
        )
    else:
        raise AssertionError("A threshold selecting no features should raise ValueError.")
    print("All tests in `test_threshold_feature_graph_report_keeps_large_features` passed!")


def test_random_feature_graph_control_report_rejects_random_graph(
    random_feature_graph_control_report: Callable | None = None,
):
    random_feature_graph_control_report = (
        random_feature_graph_control_report
        or _solutions().random_feature_graph_control_report
    )
    contributions = t.tensor([0.7, 0.2, 0.05, 0.05])
    report = random_feature_graph_control_report(
        contributions,
        target_feature_ids=[0, 1],
        random_feature_ids=[2, 3],
        min_margin=0.5,
    )
    expected = reference.random_feature_graph_control_report(
        contributions,
        target_feature_ids=[0, 1],
        random_feature_ids=[2, 3],
        min_margin=0.5,
    )
    _assert_report_close(report, expected, msg="Random feature graph control")
    assert report.target_recovered_fraction > report.random_recovered_fraction, (
        "The target graph should recover more than the same-size random graph."
    )
    assert report.random_graph_fails, (
        "The same-size random graph should fail by the required margin."
    )
    try:
        random_feature_graph_control_report(
            contributions,
            target_feature_ids=[0, 1],
            random_feature_ids=[0, 2],
            min_margin=0.5,
        )
    except ValueError as exc:
        assert "must not overlap target features" in str(exc), (
            "Random graph controls should be disjoint from the claimed target graph."
        )
    else:
        raise AssertionError("Overlapping target/random feature ids should raise ValueError.")
    print("All tests in `test_random_feature_graph_control_report_rejects_random_graph` passed!")


def test_shift_style_sparse_feature_editing_report_removes_spurious_feature(
    shift_style_sparse_feature_editing_report: Callable | None = None,
):
    shift_style_sparse_feature_editing_report = (
        shift_style_sparse_feature_editing_report
        or _solutions().shift_style_sparse_feature_editing_report
    )
    fixture = reference.toy_shift_sparse_feature_editing_fixture()
    report = shift_style_sparse_feature_editing_report(
        fixture["train_features"],
        fixture["train_labels"],
        fixture["ood_features"],
        fixture["ood_labels"],
        fixture["classifier_weights"],
        target_feature_ids=[0],
        spurious_feature_ids=[1],
        random_feature_ids=[2],
        suppression=0.0,
        max_target_accuracy_drop=0.05,
        min_ood_improvement=0.5,
        min_random_edit_gap=0.5,
    )
    expected = reference.shift_style_sparse_feature_editing_report(
        fixture["train_features"],
        fixture["train_labels"],
        fixture["ood_features"],
        fixture["ood_labels"],
        fixture["classifier_weights"],
        target_feature_ids=[0],
        spurious_feature_ids=[1],
        random_feature_ids=[2],
        suppression=0.0,
        max_target_accuracy_drop=0.05,
        min_ood_improvement=0.5,
        min_random_edit_gap=0.5,
    )
    _assert_report_close(report, expected, msg="SHIFT-style sparse-feature edit")
    assert report.baseline_train_accuracy == 1.0, (
        "The generated training split should be solved before editing."
    )
    assert report.baseline_ood_accuracy == 0.0, (
        "The unedited classifier should fail the anti-correlated OOD split."
    )
    assert report.edited_train_accuracy == 1.0 and report.edited_ood_accuracy == 1.0, (
        "Suppressing the spurious feature should preserve train accuracy and fix OOD."
    )
    assert report.spurious_reliance_decreases and report.target_task_preserved, (
        "The edit should suppress spurious reliance without damaging the target feature."
    )
    assert report.random_edit_ood_accuracy == report.baseline_ood_accuracy, (
        "A same-size random feature edit should not repair the OOD failure."
    )
    assert report.random_edit_control_fails and report.editing_passes, (
        "The SHIFT-style edit should beat the random-edit control and pass all gates."
    )
    try:
        shift_style_sparse_feature_editing_report(
            fixture["train_features"],
            fixture["train_labels"],
            fixture["ood_features"],
            fixture["ood_labels"],
            fixture["classifier_weights"],
            target_feature_ids=[0],
            spurious_feature_ids=[1],
            random_feature_ids=[1],
        )
    except ValueError as exc:
        assert "random edit control" in str(exc), (
            "Random controls that edit the spurious feature should produce a clear error."
        )
    else:
        raise AssertionError("Overlapping random/spurious feature ids should raise ValueError.")
    print(
        "All tests in `test_shift_style_sparse_feature_editing_report_removes_spurious_feature` passed!"
    )


def test_sparse_autoencoder_state_dict_smoke_report_checks_shapes(
    sparse_autoencoder_state_dict_smoke_report: Callable | None = None,
):
    sparse_autoencoder_state_dict_smoke_report = (
        sparse_autoencoder_state_dict_smoke_report
        or _solutions().sparse_autoencoder_state_dict_smoke_report
    )
    state_dict = {
        "bias": t.zeros(2),
        "encoder.weight": t.eye(2),
        "encoder.bias": t.ones(2),
        "decoder.weight": t.eye(2),
    }
    activation = t.tensor([1.0, 2.0])
    report = sparse_autoencoder_state_dict_smoke_report(state_dict, activation)
    assert report.activation_dim == 2 and report.dict_size == 2, (
        "The state-dict smoke should report activation and dictionary dimensions."
    )
    assert report.feature_l0 == 2 and report.tensors_finite and report.passes_smoke, (
        "A finite toy SAE with active features should pass the shape/finiteness smoke."
    )
    try:
        sparse_autoencoder_state_dict_smoke_report({"bias": t.zeros(2)}, activation)
    except ValueError as exc:
        assert "missing keys" in str(exc), (
            "Missing SAE tensors should produce a keyed state-dict error."
        )
    else:
        raise AssertionError("Incomplete SAE state dicts should raise ValueError.")
    print(
        "All tests in `test_sparse_autoencoder_state_dict_smoke_report_checks_shapes` passed!"
    )


def test_notebook_contract(run_smoke_test: Callable | None = None):
    run_smoke_test = run_smoke_test or _solutions().run_smoke_test
    result = run_smoke_test(cpu=True)
    assert result["planted_signature"]["accepted"], (
        "The notebook contract should include the exact planted sparse-feature signature."
    )
    assert result["encode_decode"]["matches_input"], (
        "The notebook contract should include the encode/decode shape check."
    )
    assert result["exact_node_patching"]["passes_recovery"], (
        "The notebook contract should include exact node patching recovery."
    )
    assert result["exact_edge_patching"]["passes_recovery"], (
        "The notebook contract should include exact edge patching recovery."
    )
    assert result["eap_ig"]["eap_ig_improves"], (
        "The notebook contract should include the EAP-IG comparison."
    )
    assert result["threshold_graph"]["passes_threshold"], (
        "The notebook contract should include thresholded graph recovery."
    )
    assert result["random_graph_control"]["random_graph_fails"], (
        "The notebook contract should include the same-size random graph control."
    )
    assert result["shift_editing"]["editing_passes"], (
        "The notebook contract should include SHIFT-style sparse-feature editing."
    )
    print("All tests in `test_notebook_contract` passed!")


def test_exercise_notebook_exposes_arena_style_sparse_feature_surface():
    notebook_path = _section_dir() / "8.5_Sparse_Feature_Circuits_exercises.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    text = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    required_markers = [
        "By the end of this notebook",
        "Core question",
        "exact planted sparse-feature graph",
        "### Exercise - implement `toy_sae_encode`",
        "### Exercise - implement `exact_planted_node_patch_scores`",
        "### Exercise - implement `exact_planted_edge_patch_scores`",
        "### Exercise - implement `nonlinear_eap_ig_edge_attribution_report`",
        "### Exercise - implement `threshold_sparse_feature_graph`",
        "### Exercise - implement `faithfulness_minimality_completeness_report`",
        "### Exercise - implement `shift_style_sparse_feature_editing_report`",
        "sparse_feature_circuits_planted_graph.png",
        "sparse_feature_circuits_metric_curves.png",
        "## Try It Yourself",
        "## Bonus: Hunt an Anomaly",
        "Released-artifact boundary",
    ]
    missing = [marker for marker in required_markers if marker not in text]
    assert not missing, f"8.5 learner surface is missing ARENA markers: {missing}"
    assert text.count("<summary>Expected output</summary>") >= 7, (
        "Each graded exercise should include an expected-output dropdown."
    )
    assert text.count("<summary>Solution</summary>") >= 7, (
        "Each graded exercise should include a solution dropdown."
    )
    assert text.count("<summary>Help") >= 7, (
        "Each graded exercise should include help or interpretation guidance."
    )
    print("All tests in `test_exercise_notebook_exposes_arena_style_sparse_feature_surface` passed!")


def test_pythia_subject_verb_residual_preflight_result(result: dict | None = None):
    result = result or _gpu_report()["pythia_subject_verb_preflight"]
    assert result["claim_scope"] == "real_model_residual_preflight_not_official_sae_replication", (
        "The Pythia residual preflight must not claim official sparse-feature replication."
    )
    assert result["clean_logit_diff"] > result["corrupt_logit_diff"], (
        "The clean subject-verb prompt should improve the target logit difference."
    )
    assert result["linearization_error"] < 1e-3, (
        "Residual-dimension attribution should linearize the unembedding effect."
    )
    assert result["recovered_fraction"] > result["random_recovered_fraction"], (
        "Top residual dimensions should beat a same-size random control."
    )
    assert result["random_control_fails"] and result["preflight_passed"], (
        "The real-model preflight should pass its random-control and final gates."
    )
    print("All tests in `test_pythia_subject_verb_residual_preflight_result` passed!")


def test_official_artifact_readiness_result(result: dict | None = None):
    result = result or _gpu_report()["official_artifact_readiness"]
    assert result["claim_scope"] == "artifact_manifest_readiness_not_sparse_feature_replication", (
        "Artifact readiness is a manifest check, not a replication claim."
    )
    assert result["remote_manifest_passed"] and result["hf_zip_present"], (
        "The official repo and Hugging Face zip manifests should be present."
    )
    assert result["expected_dictionary_count"] == 19, (
        "Pythia-70M SFC readiness should expect 19 dictionary files."
    )
    assert result["hf_zip_size_gb"] > 2.0, (
        "The released dictionary zip should have the expected multi-GB size."
    )
    if result["local_dictionaries_ready"]:
        assert result["ready_for_gt2_replication"], (
            "Local dictionaries plus remote manifests should enable GT-2 replication."
        )
        assert result["missing_dictionary_count"] == 0, (
            "Ready artifacts should have no missing dictionaries."
        )
    else:
        assert not result["ready_for_gt2_replication"], (
            "Missing local dictionaries should block GT-2 replication."
        )
        assert result["dictionary_download_required"], (
            "Missing local dictionaries should explicitly request the download."
        )
        assert result["missing_dictionary_count"] > 0, (
            "Missing dictionary count should be visible."
        )
    print("All tests in `test_official_artifact_readiness_result` passed!")


def test_official_sae_state_dict_smoke_result(result: dict | None = None):
    result = result or _gpu_report()["official_sae_state_dict_smoke"]
    assert result["claim_scope"] == "official_sae_state_dict_shape_and_finiteness_smoke_only", (
        "The SAE state-dict smoke should not claim graph replication."
    )
    assert result["state_dict_present"], "The released SAE state dict should be present."
    assert result["activation_dim"] == 512 and result["dict_size"] == 32768, (
        "The official Pythia SAE should expose 512-d activations and a 32768 dictionary."
    )
    assert result["feature_l0"] > 0 and result["shapes_match"], (
        "The official SAE should produce nonzero features with matching shapes."
    )
    assert result["tensors_finite"] and result["passes_smoke"], (
        "The official SAE smoke should produce finite tensors and pass."
    )
    print("All tests in `test_official_sae_state_dict_smoke_result` passed!")


def test_official_sae_feature_attribution_smoke_result(result: dict | None = None):
    result = result or _gpu_report()["official_sae_feature_attribution"]
    assert result["claim_scope"] == "one_layer_official_sae_feature_attribution_smoke_not_full_graph", (
        "One-layer SAE attribution should not claim full sparse-feature graph replication."
    )
    assert result["dictionary_path"].endswith("resid_out_layer5/10_32768/ae.pt"), (
        "The one-layer attribution smoke should use the released resid_out_layer5 SAE."
    )
    assert result["top_k"] > 0, (
        "The one-layer attribution smoke should select a positive number of features."
    )
    assert result["clean_feature_l0"] > 0 and result["corrupt_feature_l0"] > 0, (
        "Clean and corrupt prompts should activate released SAE features."
    )
    assert abs(result["total_residual_effect"]) > 0 and abs(result["decoded_feature_effect"]) > 0, (
        "The SAE attribution smoke should report nonzero residual and decoded effects."
    )
    assert result["recovered_fraction"] > result["random_recovered_fraction"], (
        "Selected SAE features should beat a same-size random feature control."
    )
    assert result["random_control_fails"] and result["passes_smoke"], (
        "The one-layer SAE attribution smoke should pass its final gates."
    )
    print("All tests in `test_official_sae_feature_attribution_smoke_result` passed!")


def test_official_sparse_feature_circuit_replication_result(result: dict | None = None):
    result = result or _gpu_report()["official_sparse_feature_circuit"]
    assert result["claim_scope"] == "official_code_100_example_sparse_feature_graph_replication_artifact", (
        "The official graph artifact should state the 100-example replication scope."
    )
    assert result["circuit_present"] and result["figure_present"], (
        "The official graph artifact and figure should both be present."
    )
    assert result["examples"] >= 100 and result["node_submodule_count"] == 19, (
        "The official graph should cover 100 examples and 19 node submodules."
    )
    assert result["edge_group_count"] > 0, (
        "The official graph should contain nonempty edge groups."
    )
    assert result["thresholded_node_count"] > 0 and result["thresholded_edge_count"] > 0, (
        "Thresholding should leave nonzero nodes and edges."
    )
    assert result["max_abs_node_effect"] > 0 and result["max_abs_edge_effect"] > 0, (
        "Official node and edge effects should be nonzero."
    )
    assert result["passes_smoke"], "The official graph artifact smoke should pass."
    print("All tests in `test_official_sparse_feature_circuit_replication_result` passed!")


def test_official_sparse_feature_circuit_faithfulness_result(result: dict | None = None):
    result = result or _gpu_report()["official_sparse_feature_circuit_faithfulness_report"]
    assert result["claim_scope"] == "official_code_heldout_sparse_feature_circuit_faithfulness", (
        "Held-out faithfulness should state the official-code evaluation scope."
    )
    assert result["cuda_available"], "Official held-out faithfulness should run on CUDA."
    assert result["heldout_dataset"] == "simple_test" and result["heldout_examples"] >= 40, (
        "Held-out faithfulness should use at least 40 simple_test examples."
    )
    assert result["start_layer"] == 2 and result["returncode"] == 0, (
        "The official evaluator should use the pinned start layer and exit cleanly."
    )
    assert result["faithfulness"] >= 0.9 and result["passes_faithfulness"], (
        "Official held-out faithfulness should pass the 0.9 threshold."
    )
    print("All tests in `test_official_sparse_feature_circuit_faithfulness_result` passed!")


def test_shift_style_sparse_feature_editing_gpu_result(result: dict | None = None):
    result = result or _gpu_report()["shift_editing"]
    assert result["editing_passes"], (
        "The GPU report should include a passing generated-data SHIFT-style edit."
    )
    assert result["baseline_ood_accuracy"] == 0.0 and result["edited_ood_accuracy"] == 1.0, (
        "The generated OOD split should fail before the sparse-feature edit and pass after it."
    )
    assert result["edited_train_accuracy"] == result["baseline_train_accuracy"], (
        "The edit should preserve target-task training accuracy."
    )
    assert result["spurious_reliance_after"] < result["spurious_reliance_before"], (
        "The edit should quantitatively reduce reliance on the spurious feature."
    )
    assert result["random_edit_control_fails"], (
        "A same-size random feature edit should remain a negative control."
    )
    assert result["black_box_baseline_ood_accuracy"] == result["baseline_ood_accuracy"], (
        "The unedited black-box baseline should be recorded for comparison."
    )
    print("All tests in `test_shift_style_sparse_feature_editing_gpu_result` passed!")
