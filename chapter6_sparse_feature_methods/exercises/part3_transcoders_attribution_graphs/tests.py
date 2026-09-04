"""Visible semantic tests for [6.3] Transcoders and Attribution Graphs."""

from __future__ import annotations

from collections.abc import Callable

import torch as t


def _solutions():
    from chapter6_sparse_feature_methods.exercises.part3_transcoders_attribution_graphs import solutions

    return solutions


def _fn(provided: Callable | None, name: str) -> Callable:
    return provided if provided is not None else getattr(_solutions(), name)


def _organism_and_input():
    solutions = _solutions()
    return solutions.make_toy_organism(), solutions.make_toy_input("red", "square")


def test_encoder_decoder_and_forward_recover_exact_mlp(
    encode: Callable | None = None,
    decode: Callable | None = None,
    transcoder_forward: Callable | None = None,
) -> None:
    encode = _fn(encode, "encode")
    decode = _fn(decode, "decode")
    transcoder_forward = _fn(transcoder_forward, "transcoder_forward")
    organism, inputs = _organism_and_input()
    pre_acts, feature_acts = encode(inputs, organism.encoder_weight, organism.encoder_bias)
    t.testing.assert_close(
        pre_acts,
        t.tensor([0.5, -0.5, 0.5, -0.5, 0.5, -1.5]),
        msg="The red-square pre-activations should include red, square, and conjunction detectors.",
    )
    t.testing.assert_close(
        feature_acts,
        t.tensor([0.5, 0.0, 0.5, 0.0, 0.5, 0.0]),
        msg="ReLU should retain exactly the three active red-square features.",
    )
    expected_reconstruction = t.tensor([1.35, -0.25, 0.10])
    t.testing.assert_close(
        decode(feature_acts, organism.decoder_weight, organism.decoder_bias),
        expected_reconstruction,
        msg="The decoder should reconstruct the exact MLP output [1.35, -0.25, 0.10].",
    )
    output = transcoder_forward(
        inputs,
        organism.encoder_weight,
        organism.decoder_weight,
        encoder_bias=organism.encoder_bias,
        decoder_bias=organism.decoder_bias,
    )
    t.testing.assert_close(
        output.reconstructed_activations,
        expected_reconstruction,
        msg="The composed transcoder should exactly match the hand-computed MLP replacement.",
    )
    print("All tests in `test_encoder_decoder_and_forward_recover_exact_mlp` passed!")


def test_reconstruction_decomposition_sums_feature_vectors(
    reconstruction_decomposition: Callable | None = None,
) -> None:
    reconstruction_decomposition = _fn(reconstruction_decomposition, "reconstruction_decomposition")
    organism, _ = _organism_and_input()
    feature_acts = t.tensor([0.5, 0.0, 0.5, 0.0, 0.5, 0.0])
    decomposition = reconstruction_decomposition(feature_acts, organism.decoder_weight, organism.decoder_bias)
    assert decomposition.feature_components.shape == (6, 3), (
        "The decomposition should expose one three-dimensional output vector per feature."
    )
    t.testing.assert_close(
        decomposition.feature_components[4],
        t.tensor([1.0, -0.25, 0.0]),
        msg="The red-square feature component should be 0.5 times its decoder row.",
    )
    t.testing.assert_close(
        decomposition.reconstructed_activations,
        t.tensor([1.35, -0.25, 0.10]),
        msg="Feature components plus decoder bias should sum exactly to the reconstruction.",
    )
    print("All tests in `test_reconstruction_decomposition_sums_feature_vectors` passed!")


def test_feature_edge_attributions_are_exact_and_conserved(
    transcoder_forward: Callable | None = None,
    feature_edge_attributions: Callable | None = None,
) -> None:
    transcoder_forward = _fn(transcoder_forward, "transcoder_forward")
    feature_edge_attributions = _fn(feature_edge_attributions, "feature_edge_attributions")
    organism, inputs = _organism_and_input()
    output = transcoder_forward(
        inputs,
        organism.encoder_weight,
        organism.decoder_weight,
        encoder_bias=organism.encoder_bias,
        decoder_bias=organism.decoder_bias,
    )
    attributions = feature_edge_attributions(
        inputs,
        output.pre_acts,
        output.feature_acts,
        organism.encoder_weight,
        organism.encoder_bias,
        organism.decoder_weight,
        organism.readout,
    )
    expected_edges = t.zeros(5, 6)
    expected_edges[0, 0] = 0.4
    expected_edges[0, 4] = 2.5
    expected_edges[2, 2] = 0.3
    expected_edges[2, 4] = 2.5
    expected_edges[4, 0] = -0.2
    expected_edges[4, 2] = -0.15
    expected_edges[4, 4] = -3.75
    t.testing.assert_close(
        attributions.input_to_feature,
        expected_edges,
        msg="Signed source-to-feature paths should match the exact gated linear calculation.",
    )
    t.testing.assert_close(
        attributions.feature_to_score,
        t.tensor([0.20, 0.0, 0.15, 0.0, 1.25, 0.0]),
        msg="Feature-to-score contributions should sum to the warm-minus-cool score of 1.6.",
    )
    t.testing.assert_close(
        attributions.input_to_feature.sum(dim=0),
        attributions.feature_to_score,
        msg="Incoming path attribution must equal each active feature's outgoing score contribution.",
    )
    print("All tests in `test_feature_edge_attributions_are_exact_and_conserved` passed!")


def test_graph_extraction_recovers_known_ground_truth(
    extract_attribution_graph: Callable | None = None,
) -> None:
    extract_attribution_graph = _fn(extract_attribution_graph, "extract_attribution_graph")
    input_edges = t.zeros(5, 6)
    input_edges[0, 0] = 0.4
    input_edges[0, 4] = 2.5
    input_edges[2, 2] = 0.3
    input_edges[2, 4] = 2.5
    input_edges[4, 0] = -0.2
    input_edges[4, 2] = -0.15
    input_edges[4, 4] = -3.75
    feature_edges = t.tensor([0.20, 0.0, 0.15, 0.0, 1.25, 0.0])
    graph = extract_attribution_graph(input_edges, feature_edges, top_k_features=3)
    assert graph.feature_ids == (4, 0, 2), (
        "The graph should rank red_square, red, then square by absolute score contribution."
    )
    assert len(graph.edges) == 10, (
        "The exact graph has seven input/bias edges and three feature-to-score edges."
    )
    edge_keys = {(edge.source_type, edge.source_id, edge.target_type, edge.target_id) for edge in graph.edges}
    assert ("bias", 0, "feature", 4) in edge_keys, (
        "The inhibitory encoder-bias edge is part of the exact conjunction computation."
    )
    assert ("feature", 4, "score", 0) in edge_keys, (
        "The red-square feature must connect to the target score."
    )
    print("All tests in `test_graph_extraction_recovers_known_ground_truth` passed!")


def test_shuffled_edges_fail_conservation_control(
    edge_conservation_score: Callable | None = None,
    shuffle_input_edge_targets: Callable | None = None,
) -> None:
    edge_conservation_score = _fn(edge_conservation_score, "edge_conservation_score")
    shuffle_input_edge_targets = _fn(shuffle_input_edge_targets, "shuffle_input_edge_targets")
    graph = _solutions().toy_experiment()["graph"]
    exact_score = edge_conservation_score(graph.edges, 6)
    shuffled = shuffle_input_edge_targets(graph.edges, n_features=6, shift=1)
    shuffled_score = edge_conservation_score(shuffled, 6)
    assert abs(exact_score - 1.0) < 1e-7, (
        "The exact graph should conserve every feature's incoming and outgoing attribution."
    )
    assert shuffled_score == 0.0, (
        "Cyclically shuffled edge targets should fail the conservation control completely."
    )
    print("All tests in `test_shuffled_edges_fail_conservation_control` passed!")


def test_interventions_recover_faithfulness_and_reject_controls(
    intervene_features: Callable | None = None,
    causal_validate_graph: Callable | None = None,
    faithfulness_curve: Callable | None = None,
) -> None:
    intervene_features = _fn(intervene_features, "intervene_features")
    causal_validate_graph = _fn(causal_validate_graph, "causal_validate_graph")
    faithfulness_curve = _fn(faithfulness_curve, "faithfulness_curve")
    experiment = _solutions().toy_experiment()
    organism = experiment["organism"]
    feature_acts = experiment["output"].feature_acts
    kept = intervene_features(feature_acts, [4], mode="keep")
    t.testing.assert_close(
        kept,
        t.tensor([0.0, 0.0, 0.0, 0.0, 0.5, 0.0]),
        msg="Keeping feature 4 should zero every other feature before decoding.",
    )
    discovered = faithfulness_curve(
        feature_acts,
        organism.decoder_weight,
        organism.decoder_bias,
        organism.readout,
        [4, 0, 2],
        max_k=3,
    )
    same_size_random = faithfulness_curve(
        feature_acts,
        organism.decoder_weight,
        organism.decoder_bias,
        organism.readout,
        [3, 1, 2],
        max_k=3,
    )
    t.testing.assert_close(
        discovered,
        t.tensor([0.78125, 0.90625, 1.0]),
        msg="The discovered graph should recover 78.125%, 90.625%, then 100% of the score.",
    )
    t.testing.assert_close(
        same_size_random,
        t.tensor([0.0, 0.0, 0.09375]),
        msg="The fixed same-size random graph should recover only the small square contribution.",
    )
    validation = causal_validate_graph(
        feature_acts,
        organism.decoder_weight,
        organism.decoder_bias,
        organism.readout,
        [4, 0, 2],
    )
    assert abs(validation.normalized_damage - 1.0) < 1e-7, (
        "Ablating all discovered features should remove the entire target score."
    )
    print("All tests in `test_interventions_recover_faithfulness_and_reject_controls` passed!")


def test_reconstruction_only_ranking_misses_behavior(
    reconstruction_only_ranking: Callable | None = None,
    faithfulness_curve: Callable | None = None,
) -> None:
    reconstruction_only_ranking = _fn(reconstruction_only_ranking, "reconstruction_only_ranking")
    faithfulness_curve = _fn(faithfulness_curve, "faithfulness_curve")
    experiment = _solutions().toy_experiment()
    organism = experiment["organism"]
    feature_acts = experiment["output"].feature_acts
    ranking = reconstruction_only_ranking(organism.decoder_weight)
    assert ranking[:3] == (5, 1, 3), (
        "Decoder norm should rank blue_circle, blue, and circle first in this exact control."
    )
    curve = faithfulness_curve(
        feature_acts,
        organism.decoder_weight,
        organism.decoder_bias,
        organism.readout,
        ranking,
        max_k=3,
    )
    t.testing.assert_close(
        curve,
        t.zeros(3),
        msg="Reconstruction-only decoder norms should miss the red-square target computation.",
    )
    print("All tests in `test_reconstruction_only_ranking_misses_behavior` passed!")


def test_notebook_contract(run_smoke_test: Callable | None = None) -> None:
    run_smoke_test = _fn(run_smoke_test, "run_smoke_test")
    result = run_smoke_test(cpu=True)
    assert result["model_organism"] == "exact_colored_shape_relu_mlp", (
        "The notebook contract should identify the exact model organism."
    )
    assert result["graph_feature_ids"] == [4, 0, 2], (
        "The notebook contract should recover the known feature graph."
    )
    assert result["graph_edge_count"] == 10, (
        "The notebook contract should include all ten exact graph edges."
    )
    assert result["faithfulness_curve"] == [0.78125, 0.90625, 1.0], (
        "The notebook contract should report the exact intervention faithfulness curve."
    )
    assert result["same_size_random_curve"] == [0.0, 0.0, 0.09375], (
        "The same-size random graph should remain far below the discovered graph."
    )
    assert result["edge_conservation"] == 1.0 and result["shuffled_edge_conservation"] == 0.0, (
        "Exact edges should conserve attribution while shuffled targets fail."
    )
    print("All tests in `test_notebook_contract` passed!")
