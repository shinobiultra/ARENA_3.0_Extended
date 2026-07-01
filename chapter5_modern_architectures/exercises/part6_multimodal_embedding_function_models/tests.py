from collections.abc import Callable

import torch as t

from arena_ext import specialist_models as reference


def _solutions():
    from chapter5_modern_architectures.exercises.part6_multimodal_embedding_function_models import (
        solutions,
    )

    return solutions


def test_mean_pool_embeddings_ignores_padding_and_matches_reference(
    mean_pool_embeddings: Callable | None = None,
):
    solutions = _solutions()
    mean_pool_embeddings = mean_pool_embeddings or solutions.mean_pool_embeddings
    token_embeddings = t.tensor(
        [
            [[1.0, 0.0], [3.0, 2.0], [100.0, 100.0]],
            [[2.0, 4.0], [6.0, 8.0], [0.0, 0.0]],
        ]
    )
    attention_mask = t.tensor([[1, 1, 0], [1, 1, 0]])
    pooled = mean_pool_embeddings(token_embeddings, attention_mask)
    expected = reference.mean_pool_embeddings(token_embeddings, attention_mask)
    t.testing.assert_close(
        pooled,
        expected,
        msg="Mean pooling should average only positions whose attention mask is one.",
    )
    t.testing.assert_close(
        pooled,
        t.tensor([[2.0, 1.0], [4.0, 6.0]]),
        msg="Padded sentinel vectors should not contribute to the sentence embedding.",
    )
    print("All tests in `test_mean_pool_embeddings_ignores_padding_and_matches_reference` passed!")


def test_retrieval_metrics_rank_pairs_and_hard_negative_margin(
    cosine_similarity_matrix: Callable | None = None,
    retrieval_ranks: Callable | None = None,
    embedding_retrieval_report: Callable | None = None,
):
    solutions = _solutions()
    cosine_similarity_matrix = cosine_similarity_matrix or solutions.cosine_similarity_matrix
    retrieval_ranks = retrieval_ranks or solutions.retrieval_ranks
    embedding_retrieval_report = (
        embedding_retrieval_report or solutions.embedding_retrieval_report
    )
    queries = t.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    candidates = t.tensor(
        [
            [0.9, 0.1, 0.0],
            [0.6, 0.4, 0.0],
            [0.0, 0.1, 0.9],
            [0.65, 0.65, 0.0],
        ]
    )
    targets = t.tensor([0, 1, 2])
    similarity = cosine_similarity_matrix(queries, candidates)
    reference_similarity = reference.cosine_similarity_matrix(queries, candidates)
    t.testing.assert_close(
        similarity,
        reference_similarity,
        msg="Cosine similarity should match the independent normalized-matrix reference.",
    )
    ranks = retrieval_ranks(similarity, targets)
    t.testing.assert_close(
        ranks,
        t.tensor([1, 2, 1]),
        msg="The second query should rank the hard negative above its paired target.",
    )
    report = embedding_retrieval_report(queries, candidates, targets)
    reference_report = reference.embedding_retrieval_report(queries, candidates, targets)
    assert abs(report.top1_accuracy - reference_report.top1_accuracy) < 1e-6, (
        "Local top-1 accuracy should match the independent report implementation."
    )
    assert abs(report.top1_accuracy - 2 / 3) < 1e-6, (
        "Top-1 accuracy should count the hard-negative miss, not just inspect examples."
    )
    assert abs(report.mean_reciprocal_rank - reference_report.mean_reciprocal_rank) < 1e-6, (
        "Mean reciprocal rank should be computed from one-indexed target ranks."
    )
    assert 0.0 < report.mean_margin < 0.5, (
        "The hard negative should shrink the average margin without making retrieval random."
    )
    print("All tests in `test_retrieval_metrics_rank_pairs_and_hard_negative_margin` passed!")


def test_centroid_probe_recovers_heldout_clusters(
    fit_centroid_probe: Callable | None = None,
    predict_centroid_probe: Callable | None = None,
    centroid_probe_accuracy: Callable | None = None,
):
    solutions = _solutions()
    fit_centroid_probe = fit_centroid_probe or solutions.fit_centroid_probe
    predict_centroid_probe = predict_centroid_probe or solutions.predict_centroid_probe
    centroid_probe_accuracy = centroid_probe_accuracy or solutions.centroid_probe_accuracy
    train = t.tensor(
        [
            [1.0, 0.0],
            [0.8, 0.2],
            [-1.0, 0.0],
            [-0.8, -0.2],
            [0.0, 1.0],
            [0.2, 0.8],
        ]
    )
    train_labels = t.tensor([0, 0, 1, 1, 2, 2])
    test = t.tensor([[0.9, 0.1], [-0.9, -0.1], [0.1, 0.9]])
    test_labels = t.tensor([0, 1, 2])
    probe = fit_centroid_probe(train, train_labels)
    reference_probe = reference.fit_centroid_probe(train, train_labels)
    t.testing.assert_close(
        probe.centroids,
        reference_probe.centroids,
        msg="Probe centroids should be class means normalized to unit length.",
    )
    predictions = predict_centroid_probe(test, probe)
    t.testing.assert_close(
        predictions,
        test_labels,
        msg="Nearest-centroid probe should recover held-out points from each cluster.",
    )
    assert centroid_probe_accuracy(test, test_labels, probe) == 1.0, (
        "Centroid probe accuracy should be measured on held-out embeddings."
    )
    print("All tests in `test_centroid_probe_recovers_heldout_clusters` passed!")


def test_mask_disallowed_tools_blocks_invalid_logits(
    mask_disallowed_tools: Callable | None = None,
):
    solutions = _solutions()
    mask_disallowed_tools = mask_disallowed_tools or solutions.mask_disallowed_tools
    logits = t.tensor([[1.0, 10.0, 2.0], [3.0, 0.0, 4.0]])
    allowed_tools = t.tensor([[True, False, True], [False, True, True]])
    masked = mask_disallowed_tools(logits, allowed_tools)
    reference_masked = reference.mask_disallowed_tools(logits, allowed_tools)
    t.testing.assert_close(
        masked,
        reference_masked,
        msg="Disallowed tool logits should be masked row-wise with negative infinity.",
    )
    assert bool(t.isneginf(masked[0, 1]).item()), (
        "A high-logit but unavailable tool must be impossible to select."
    )
    assert masked.argmax(dim=-1).tolist() == [2, 2], (
        "Predictions should be chosen only among available tools after masking."
    )
    print("All tests in `test_mask_disallowed_tools_blocks_invalid_logits` passed!")


def test_function_call_report_separates_tool_and_abstention_errors(
    function_call_report: Callable | None = None,
):
    solutions = _solutions()
    function_call_report = function_call_report or solutions.function_call_report
    logits = t.tensor(
        [
            [5.0, 1.0, 0.0],
            [0.0, 4.0, 1.0],
            [0.0, 3.0, 1.0],
            [0.0, 1.0, 5.0],
            [0.0, 6.0, 1.0],
        ]
    )
    labels = t.tensor([0, 1, 2, 2, 0])
    report = function_call_report(logits, labels, no_call_id=2)
    reference_report = reference.function_call_report(logits, labels, no_call_id=2)
    assert report.__dict__ == reference_report.__dict__, (
        "Function-call report should match the independent accuracy decomposition."
    )
    assert abs(report.accuracy - 0.6) < 1e-6, (
        "Overall accuracy should include tool-choice and no-call examples."
    )
    assert abs(report.tool_accuracy - 2 / 3) < 1e-6, (
        "Tool accuracy should exclude no-call labels but include wrong-tool examples."
    )
    assert (
        abs(report.abstention_accuracy - 0.5) < 1e-6
        and abs(report.hallucination_rate - 0.5) < 1e-6
    ), (
        "No-call examples should report both abstention accuracy and hallucination rate."
    )
    print("All tests in `test_function_call_report_separates_tool_and_abstention_errors` passed!")


def test_parse_function_call_text_extracts_name_and_arguments(
    parse_function_call_text: Callable | None = None,
):
    solutions = _solutions()
    parse_function_call_text = parse_function_call_text or solutions.parse_function_call_text
    text = (
        "assistant<start_function_call>"
        "call:open_app{app:<escape>Google Maps<escape>, mode:navigation}"
        "<end_function_call>"
    )
    parsed = parse_function_call_text(text)
    reference_parsed = reference.parse_function_call_text(text)
    assert parsed.__dict__ == reference_parsed.__dict__, (
        "Function-call parsing should match the pinned FunctionGemma text format."
    )
    assert parsed.name == "open_app", "Parser should recover the called function name."
    assert parsed.arguments == {"app": "Google Maps", "mode": "navigation"}, (
        "Parser should strip escaped argument delimiters and whitespace."
    )
    no_call = parse_function_call_text("The request does not require a tool.")
    assert no_call.name is None and no_call.arguments == {}, (
        "Parser should return an empty call object when no function call is present."
    )
    print("All tests in `test_parse_function_call_text_extracts_name_and_arguments` passed!")


def test_schema_token_attribution_matches_dot_products(
    schema_token_attribution: Callable | None = None,
):
    solutions = _solutions()
    schema_token_attribution = schema_token_attribution or solutions.schema_token_attribution
    hidden_states = t.tensor([[1.0, 2.0, 0.0], [0.0, 1.0, 3.0]])
    schema_vectors = t.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 1.0]])
    attribution = schema_token_attribution(hidden_states, schema_vectors)
    reference_attribution = reference.schema_token_attribution(hidden_states, schema_vectors)
    t.testing.assert_close(
        attribution,
        reference_attribution,
        msg="Schema attribution should be the hidden-state dot product with schema vectors.",
    )
    assert attribution.argmax(dim=-1).tolist() == [1, 2], (
        "Top schema directions should match the largest dot product per hidden state."
    )
    print("All tests in `test_schema_token_attribution_matches_dot_products` passed!")


def test_notebook_contract(run_smoke_test: Callable | None = None):
    if run_smoke_test is None:
        run_smoke_test = _solutions().run_smoke_test
    result = run_smoke_test(cpu=True)
    assert result["pooling"]["pooled"] == [[2.0, 1.0], [4.0, 6.0]], (
        "Notebook contract should include a padding-sensitive pooling check."
    )
    assert result["retrieval"]["top1_accuracy"] == 1.0, (
        "Notebook contract should include a paired retrieval sanity check."
    )
    assert result["centroid_probe"]["accuracy"] == 1.0, (
        "Notebook contract should include held-out centroid-probe accuracy."
    )
    assert result["tool_masking"]["prediction"] == 2, (
        "Notebook contract should include an invalid high-logit tool mask check."
    )
    assert result["function_call"]["hallucination_rate"] == 0.5, (
        "Notebook contract should include no-call hallucination diagnostics."
    )
    assert result["schema_attribution"]["top_schema_ids"] == [1, 0], (
        "Notebook contract should include schema-direction attribution diagnostics."
    )
    print("All tests in `test_notebook_contract` passed!")
