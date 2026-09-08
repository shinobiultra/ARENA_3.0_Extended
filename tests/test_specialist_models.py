import importlib.util

import pytest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is not installed")

if TORCH_AVAILABLE:
    import torch as t

    from arena_ext.specialist_models import (
        centroid_probe_accuracy,
        cosine_similarity_matrix,
        embedding_retrieval_report,
        fit_centroid_probe,
        function_call_report,
        mask_disallowed_tools,
        mean_pool_embeddings,
        parse_function_call_text,
        predict_centroid_probe,
        retrieval_ranks,
        schema_token_attribution,
    )


def test_mean_pool_embeddings_ignores_padding():
    token_embeddings = t.tensor(
        [
            [[1.0, 0.0], [3.0, 2.0], [100.0, 100.0]],
            [[2.0, 4.0], [6.0, 8.0], [0.0, 0.0]],
        ]
    )
    attention_mask = t.tensor([[1, 1, 0], [1, 1, 0]])

    pooled = mean_pool_embeddings(token_embeddings, attention_mask)

    assert t.equal(pooled, t.tensor([[2.0, 1.0], [4.0, 6.0]]))


def test_embedding_retrieval_report_is_perfect_for_matched_pairs():
    queries = t.eye(3)
    candidates = t.eye(3)
    targets = t.tensor([0, 1, 2])

    similarity = cosine_similarity_matrix(queries, candidates)
    ranks = retrieval_ranks(similarity, targets)
    report = embedding_retrieval_report(queries, candidates, targets)

    assert t.equal(ranks, t.ones(3, dtype=t.long))
    assert report.top1_accuracy == 1.0
    assert report.mean_reciprocal_rank == 1.0
    assert report.mean_margin == pytest.approx(1.0)


def test_centroid_probe_generalizes_to_nearby_points():
    train = t.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ]
    )
    train_labels = t.tensor([0, 0, 1, 1])
    test = t.tensor([[0.95, 0.05], [0.05, 0.95]])
    test_labels = t.tensor([0, 1])

    probe = fit_centroid_probe(train, train_labels)
    predictions = predict_centroid_probe(test, probe)

    assert t.equal(predictions, test_labels)
    assert centroid_probe_accuracy(test, test_labels, probe) == 1.0


def test_mask_disallowed_tools_sets_disallowed_logits_to_negative_infinity():
    logits = t.tensor([[1.0, 10.0, 2.0]])
    allowed_tools = t.tensor([True, False, True])

    masked = mask_disallowed_tools(logits, allowed_tools)

    assert t.isneginf(masked[0, 1])
    assert masked.argmax(dim=-1).item() == 2


def test_function_call_report_tracks_tool_accuracy_and_hallucination():
    logits = t.tensor(
        [
            [5.0, 1.0, 0.0],
            [0.0, 4.0, 1.0],
            [0.0, 3.0, 1.0],
            [0.0, 1.0, 5.0],
        ]
    )
    labels = t.tensor([0, 1, 2, 2])

    report = function_call_report(logits, labels, no_call_id=2)

    assert report.accuracy == 0.75
    assert report.tool_accuracy == 1.0
    assert report.abstention_accuracy == 0.5
    assert report.hallucination_rate == 0.5


def test_parse_function_call_text_extracts_first_call_and_escaped_arguments():
    generated = (
        "<start_function_call>call:send_email{"
        "body:<escape>Review the Q3 budget, then reply.<escape>,"
        "subject:<escape>Budget Review<escape>,"
        "to:<escape>kenji@example.com<escape>}"
        "<end_function_call>call:turn_on_flashlight{}"
    )

    parsed = parse_function_call_text(generated)

    assert parsed.name == "send_email"
    assert parsed.arguments == {
        "body": "Review the Q3 budget, then reply.",
        "subject": "Budget Review",
        "to": "kenji@example.com",
    }


def test_schema_token_attribution_projects_hidden_states():
    hidden_states = t.tensor([[1.0, 0.0], [0.0, 1.0]])
    schema_vectors = t.tensor([[0.0, 1.0], [1.0, 0.0]])

    attribution = schema_token_attribution(hidden_states, schema_vectors)

    assert t.equal(attribution, t.tensor([[0.0, 1.0], [1.0, 0.0]]))
