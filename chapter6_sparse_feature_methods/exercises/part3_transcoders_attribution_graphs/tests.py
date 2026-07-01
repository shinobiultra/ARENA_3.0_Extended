from collections.abc import Callable

import torch as t

from arena_ext import transcoders as reference


def _solutions():
    from chapter6_sparse_feature_methods.exercises.part3_transcoders_attribution_graphs import (
        solutions,
    )

    return solutions


def _assert_report_close(actual: object, expected: object, *, msg: str) -> None:
    actual_dict = actual.__dict__
    expected_dict = expected.__dict__
    assert actual_dict.keys() == expected_dict.keys(), (
        f"{msg} Report fields should match the independent reference."
    )
    for key, expected_value in expected_dict.items():
        actual_value = actual_dict[key]
        if isinstance(expected_value, float):
            assert abs(actual_value - expected_value) < 1e-6, (
                f"{msg} Field {key!r} should be {expected_value}, got {actual_value}."
            )
        else:
            assert actual_value == expected_value, (
                f"{msg} Field {key!r} should be {expected_value!r}, got {actual_value!r}."
            )


def test_transcoder_forward_matches_reference_and_relu_rules(
    transcoder_forward: Callable | None = None,
):
    solutions = _solutions()
    transcoder_forward = transcoder_forward or solutions.transcoder_forward
    inputs = t.tensor([[1.0, -2.0], [3.0, 4.0]])
    encoder_weight = t.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 1.0]])
    decoder_weight = t.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, -0.5]])
    encoder_bias = t.tensor([0.0, 0.0, 0.25])
    decoder_bias = t.tensor([0.1, -0.2])

    actual = transcoder_forward(
        inputs,
        encoder_weight,
        decoder_weight,
        encoder_bias=encoder_bias,
        decoder_bias=decoder_bias,
    )
    expected = reference.transcoder_forward(
        inputs,
        encoder_weight,
        decoder_weight,
        encoder_bias=encoder_bias,
        decoder_bias=decoder_bias,
    )
    t.testing.assert_close(
        actual.feature_acts,
        expected.feature_acts,
        msg="Transcoder feature activations should match ReLU(input @ W_enc.T + b_enc).",
    )
    t.testing.assert_close(
        actual.reconstructed_activations,
        expected.reconstructed_activations,
        msg="Transcoder reconstruction should match feature_acts @ W_dec + b_dec.",
    )
    assert actual.feature_acts[0, 1].item() == 0.0, (
        "Negative encoder pre-activations should be clamped to zero by the ReLU."
    )
    print("All tests in `test_transcoder_forward_matches_reference_and_relu_rules` passed!")


def test_target_logit_diff_and_replacement_report_match_reference(
    target_logit_diff: Callable | None = None,
    transcoder_replacement_report: Callable | None = None,
):
    solutions = _solutions()
    target_logit_diff = target_logit_diff or solutions.target_logit_diff
    transcoder_replacement_report = (
        transcoder_replacement_report or solutions.transcoder_replacement_report
    )
    reference_activations = t.tensor([[1.0, 2.0], [0.5, -1.0]])
    reconstructed = reference_activations + t.tensor([[0.01, -0.02], [0.0, 0.02]])
    reference_logits = t.tensor([[2.0, 0.0, -1.0], [1.5, 0.5, -0.5]])
    replacement_logits = t.tensor([[1.98, 0.02, -1.0], [1.48, 0.52, -0.5]])

    actual_diff = target_logit_diff(
        reference_logits,
        positive_token_id=0,
        negative_token_id=1,
    )
    expected_diff = reference.target_logit_diff(
        reference_logits,
        positive_token_id=0,
        negative_token_id=1,
    )
    assert abs(actual_diff - expected_diff) < 1e-6, (
        "Target logit diff should be the mean positive-minus-negative logit."
    )

    actual = transcoder_replacement_report(
        reference_activations=reference_activations,
        reconstructed_activations=reconstructed,
        reference_logits=reference_logits,
        replacement_logits=replacement_logits,
        positive_token_id=0,
        negative_token_id=1,
        kl_threshold=1e-3,
        logit_diff_tolerance=0.1,
    )
    expected = reference.transcoder_replacement_report(
        reference_activations=reference_activations,
        reconstructed_activations=reconstructed,
        reference_logits=reference_logits,
        replacement_logits=replacement_logits,
        positive_token_id=0,
        negative_token_id=1,
        kl_threshold=1e-3,
        logit_diff_tolerance=0.1,
    )
    _assert_report_close(actual, expected, msg="Replacement report")
    assert actual.passes_kl and actual.preserves_logit_diff, (
        "Small logit perturbations should pass both KL and logit-diff preservation checks."
    )
    print("All tests in `test_target_logit_diff_and_replacement_report_match_reference` passed!")


def test_feature_logit_contributions_reduce_all_nonfeature_dimensions(
    feature_logit_contributions: Callable | None = None,
):
    solutions = _solutions()
    feature_logit_contributions = (
        feature_logit_contributions or solutions.feature_logit_contributions
    )
    feature_acts = t.tensor(
        [
            [[1.0, 2.0, 0.0], [3.0, 0.0, 2.0]],
            [[0.0, 4.0, 1.0], [2.0, 2.0, 1.0]],
        ]
    )
    logit_effects = t.tensor([0.5, 1.0, -1.0])
    actual = feature_logit_contributions(feature_acts, logit_effects)
    expected = reference.feature_logit_contributions(feature_acts, logit_effects)
    t.testing.assert_close(
        actual,
        expected,
        msg="Feature contributions should average over every non-feature dimension.",
    )
    t.testing.assert_close(
        actual,
        t.tensor([0.75, 2.0, -1.0]),
        msg="Controlled feature contribution values should match mean_activation * logit_effect.",
    )
    print(
        "All tests in `test_feature_logit_contributions_reduce_all_nonfeature_dimensions` passed!"
    )


def test_build_attribution_edges_keeps_top_input_and_logit_edges(
    build_attribution_edges: Callable | None = None,
    graph_reproducible: Callable | None = None,
):
    solutions = _solutions()
    build_attribution_edges = build_attribution_edges or solutions.build_attribution_edges
    graph_reproducible = graph_reproducible or solutions.graph_reproducible
    input_scores = t.tensor([[0.1, 0.8], [0.4, -0.9]])
    logit_effects = t.tensor([0.3, -1.0])

    actual = build_attribution_edges(input_scores, logit_effects, top_k=2)
    expected = reference.build_attribution_edges(input_scores, logit_effects, top_k=2)
    assert [edge.__dict__ for edge in actual] == [edge.__dict__ for edge in expected], (
        "Attribution edges should keep top absolute input-feature and feature-logit weights."
    )
    assert actual[0].source_type == "input" and actual[0].target_type == "feature", (
        "The first block of edges should connect input positions to feature nodes."
    )
    assert actual[-1].source_type == "feature" and actual[-1].target_type == "logit_diff", (
        "The second block of edges should connect feature nodes to the target logit diff."
    )
    assert graph_reproducible(actual, build_attribution_edges(input_scores, logit_effects, top_k=2)), (
        "Deterministic graph construction should reproduce the same edge list."
    )
    print("All tests in `test_build_attribution_edges_keeps_top_input_and_logit_edges` passed!")


def test_graph_reproducible_rejects_structure_and_weight_changes(
    AttributionEdge: type | None = None,
    graph_reproducible: Callable | None = None,
):
    solutions = _solutions()
    AttributionEdge = AttributionEdge or solutions.AttributionEdge
    graph_reproducible = graph_reproducible or solutions.graph_reproducible
    edges = [
        AttributionEdge("input", 0, "feature", 1, 0.8),
        AttributionEdge("feature", 1, "logit_diff", 0, -1.0),
    ]
    same = [
        AttributionEdge("input", 0, "feature", 1, 0.8000001),
        AttributionEdge("feature", 1, "logit_diff", 0, -1.0),
    ]
    changed_weight = [
        AttributionEdge("input", 0, "feature", 1, 0.81),
        AttributionEdge("feature", 1, "logit_diff", 0, -1.0),
    ]
    changed_structure = [
        AttributionEdge("input", 0, "feature", 0, 0.8),
        AttributionEdge("feature", 1, "logit_diff", 0, -1.0),
    ]
    assert graph_reproducible(edges, same), (
        "Tiny floating-point edge-weight differences should be tolerated."
    )
    assert not graph_reproducible(edges, changed_weight), (
        "Graph reproducibility should fail when an edge weight changes beyond tolerance."
    )
    assert not graph_reproducible(edges, changed_structure), (
        "Graph reproducibility should fail when edge structure changes."
    )
    print("All tests in `test_graph_reproducible_rejects_structure_and_weight_changes` passed!")


def test_attribution_graph_report_preservation_and_damage_controls(
    graph_density: Callable | None = None,
    attribution_graph_report: Callable | None = None,
):
    solutions = _solutions()
    graph_density = graph_density or solutions.graph_density
    attribution_graph_report = attribution_graph_report or solutions.attribution_graph_report
    contributions = t.tensor([0.7, 0.2, 0.05, 0.05])
    actual = attribution_graph_report(
        contributions,
        graph_feature_ids=[0, 1],
        random_feature_ids=[2, 3],
        num_nodes=6,
        num_edges=4,
        reproducible=True,
    )
    expected = reference.attribution_graph_report(
        contributions,
        graph_feature_ids=[0, 1],
        random_feature_ids=[2, 3],
        num_nodes=6,
        num_edges=4,
        reproducible=True,
    )
    _assert_report_close(actual, expected, msg="Attribution graph report")
    assert abs(graph_density(num_nodes=6, num_edges=4) - 4 / 30) < 1e-6, (
        "Directed graph density should be num_edges / (num_nodes * (num_nodes - 1))."
    )
    assert actual.preserves_logit_diff and actual.passes_damage_control, (
        "Top graph features should preserve most of the logit diff and damage behavior more than controls."
    )

    failed = attribution_graph_report(
        contributions,
        graph_feature_ids=[2, 3],
        random_feature_ids=[0, 1],
        num_nodes=6,
        num_edges=4,
        reproducible=True,
    )
    assert not failed.preserves_logit_diff and not failed.passes_damage_control, (
        "Low-effect graph nodes should fail preservation and damage-control checks."
    )
    print("All tests in `test_attribution_graph_report_preservation_and_damage_controls` passed!")


def test_notebook_contract(run_smoke_test: Callable | None = None):
    if run_smoke_test is None:
        run_smoke_test = _solutions().run_smoke_test
    result = run_smoke_test(cpu=True)
    assert result["forward"]["reconstructed_activations"] == [[1.0, 2.0], [3.0, 4.0]], (
        "Notebook contract should include the identity transcoder reconstruction example."
    )
    assert result["replacement"]["passes_kl"], (
        "Notebook contract should include a replacement KL check."
    )
    assert result["replacement"]["preserves_logit_diff"], (
        "Notebook contract should include target logit-diff preservation."
    )
    assert result["contributions"]["contributions"] == [1.0, 1.0, -1.0], (
        "Notebook contract should include controlled feature-logit contribution values."
    )
    assert result["graph_edges"]["reproducible"], (
        "Notebook contract should include deterministic graph construction."
    )
    assert result["graph_report"]["passes_damage_control"], (
        "Notebook contract should include top-feature damage beating a random control."
    )
    print("All tests in `test_notebook_contract` passed!")
