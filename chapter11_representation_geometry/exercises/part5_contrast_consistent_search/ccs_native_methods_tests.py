"""Immediate tests distinguish padding, controlled pairs and output normalization."""

import copy

import torch
from torch.testing import assert_close

from chapter11_representation_geometry.exercises.part5_contrast_consistent_search import ccs_native_methods as reference


def test_last_token_rows(gather=reference.last_token_rows):
    hidden = torch.arange(24., dtype=torch.float64).reshape(3, 4, 2).requires_grad_()
    mask = torch.tensor([[1, 1, 0, 0], [0, 1, 1, 1], [1, 0, 0, 0]])
    answer = gather(hidden, mask)
    assert_close(answer, torch.tensor([[2., 3.], [14., 15.], [16., 17.]], dtype=hidden.dtype))
    answer.sum().backward()
    expected = torch.zeros_like(hidden)
    expected[0, 1], expected[1, 3], expected[2, 0] = 1, 1, 1
    assert_close(hidden.grad, expected)
    print('All tests in test_last_token_rows passed!')


def sample_rows():
    return [{"base_id": name, "distractor": opinion, "review": f"Review {name}", "label": label,
             "wrapper": "customer", "variant": "distracting_reader"}
            for name, label in (("a", 0), ("b", 1)) for opinion in (0, 1)]


def test_paired_opinion_change(measure=reference.paired_opinion_change):
    rows = sample_rows()
    scores = torch.tensor([.1, .9, .8, .9], dtype=torch.float64)
    result = measure(scores, rows)
    assert (
        result['base_reviews'] == 2 and result['prediction_flip_fraction'] == 0.5
    ), 'Match the same customer across the two opinions; score complements preserve flip fractions.'
    assert_close(result["score_changes"], torch.tensor([.8, .1], dtype=scores.dtype))
    assert (
        abs(result['mean_absolute_score_change'] - 0.45) < 1e-12
    ), 'Match the same customer across the two opinions; score complements preserve flip fractions.'
    permutation = [3, 0, 2, 1]
    shuffled = measure(scores[permutation], [rows[i] for i in permutation])
    assert_close(shuffled["score_changes"], result["score_changes"])
    complement = measure(1 - scores, rows)
    assert (
        complement['prediction_flip_fraction'] == result['prediction_flip_fraction']
    ), 'Match the same customer across the two opinions; score complements preserve flip fractions.'
    assert_close(complement["score_changes"], -result["score_changes"])
    print('All tests in test_paired_opinion_change passed!')


def test_model_answer_metrics(measure=reference.model_answer_metrics):
    # Positive beats negative, but a third token is the actual winner in row zero.
    logits = torch.log(torch.tensor([[.1, .2, .7], [.6, .3, .1]], dtype=torch.float64))
    result = measure(logits, [0, 1], torch.tensor([1, 0]))
    assert (
        result['restricted_correct'] == 2 and result['full_vocabulary_correct'] == 1
    ), 'Compute the full-vocabulary winner before restricting to the two answer tokens.'
    assert_close(result["restricted_positive_scores"], torch.tensor([2 / 3, 1 / 3], dtype=logits.dtype))
    assert_close(result["candidate_mass"], torch.tensor([.3, .9], dtype=logits.dtype))
    assert_close(result["winning_token_ids"], torch.tensor([2, 0]))
    shifted = measure(logits + 100, [0, 1], torch.tensor([1, 0]))
    assert_close(shifted["candidate_probabilities"], result["candidate_probabilities"])
    print('All tests in test_model_answer_metrics passed!')


def test_native_measurement_rejections():
    import pytest

    with pytest.raises(ValueError, match="valid token"):
        reference.last_token_rows(torch.ones(2, 3, 4), torch.zeros(2, 3))
    rows = sample_rows()
    with pytest.raises(ValueError, match="both opinions"):
        reference.paired_opinion_change(torch.ones(3), rows[:3])
    broken = copy.deepcopy(rows)
    broken[1]["label"] = 1
    with pytest.raises(ValueError, match="Only the distracting"):
        reference.paired_opinion_change(torch.ones(4), broken)
    with pytest.raises(ValueError, match="duplicate"):
        reference.paired_opinion_change(torch.ones(5), [*rows, rows[0]])
    with pytest.raises(ValueError, match="distinct"):
        reference.model_answer_metrics(torch.ones(2, 3), [1, 1], torch.tensor([0, 1]))
    print('All tests in test_native_measurement_rejections passed!')
