import importlib.util

import pytest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is not installed")

if TORCH_AVAILABLE:
    import torch as t

    from arena_ext.activation_language import (
        attention_lens,
        counterfactual_activation_report,
        lens_accuracy_report,
        logit_lens,
        patchscope_accuracy_report,
        patchscope_prompt,
        prediction_accuracy,
        random_activation_confidence_report,
        replace_final_position_activation,
        top_tokens,
        tuned_lens,
    )


def test_logit_lens_and_top_tokens_decode_residuals():
    residual = t.tensor([[1.0, 0.0], [0.0, 1.0]])
    unembedding = t.tensor([[2.0, 0.0, 1.0], [0.0, 3.0, 1.0]])

    logits = logit_lens(residual, unembedding)
    ids, probs = top_tokens(logits, k=1)

    assert t.equal(logits, t.tensor([[2.0, 0.0, 1.0], [0.0, 3.0, 1.0]]))
    assert ids.squeeze(-1).tolist() == [0, 1]
    assert probs.squeeze(-1).gt(0.5).all()


def test_tuned_lens_improves_over_logit_lens():
    residual = t.tensor([[1.0, 0.0], [0.0, 1.0]])
    unembedding = t.eye(2)
    logit_logits = logit_lens(residual, unembedding)
    lens_weight = t.tensor([[0.0, 1.0], [1.0, 0.0]])
    tuned_logits = tuned_lens(residual, lens_weight, None, unembedding)
    targets = t.tensor([1, 0])

    report = lens_accuracy_report(logit_logits, tuned_logits, targets)

    assert prediction_accuracy(logit_logits, targets) == 0.0
    assert prediction_accuracy(tuned_logits, targets) == 1.0
    assert report.tuned_lens_improves


def test_attention_lens_decodes_attended_values():
    attention = t.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    values = t.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    unembedding = t.eye(2)

    logits = attention_lens(attention, values, unembedding)

    assert t.equal(logits, values)


def test_patchscope_prompt_and_accuracy_report():
    patchscope_logits = t.tensor([[2.0, 0.0], [0.0, 2.0]])
    text_only_logits = t.tensor([[0.0, 2.0], [0.0, 2.0]])
    targets = t.tensor([0, 1])

    report = patchscope_accuracy_report(patchscope_logits, text_only_logits, targets)

    assert patchscope_prompt("entity") == "What entity is represented by <ACT>?"
    assert patchscope_prompt("next_token") == "What token will <ACT> become next?"
    assert patchscope_prompt("fact") == "What fact is stored in <ACT>?"
    assert report.patchscope_accuracy == 1.0
    assert report.text_only_accuracy == 0.5
    assert report.beats_text_only


def test_replace_final_position_activation_only_patches_answer_position():
    activations = t.zeros(1, 4, 3)
    source_activation = t.tensor([1.0, -2.0, 3.0])

    patched = replace_final_position_activation(activations, source_activation)

    assert patched.shape == activations.shape
    assert patched[0, -1].tolist() == [1.0, -2.0, 3.0]
    assert patched[0, :-1].abs().sum().item() == 0.0
    assert activations.abs().sum().item() == 0.0


def test_counterfactual_activation_report_detects_answer_change():
    original = t.tensor([2.0, 0.0])
    patched = t.tensor([0.0, 3.0])

    report = counterfactual_activation_report(original, patched)

    assert report.original_answer == 0
    assert report.patched_answer == 1
    assert report.changed


def test_random_activation_confidence_report_requires_low_confidence():
    random_logits = t.zeros(3, 4)

    report = random_activation_confidence_report(random_logits, max_allowed_confidence=0.3)

    assert report.mean_confidence == pytest.approx(0.25)
    assert report.max_confidence == pytest.approx(0.25)
    assert report.passes_low_confidence
