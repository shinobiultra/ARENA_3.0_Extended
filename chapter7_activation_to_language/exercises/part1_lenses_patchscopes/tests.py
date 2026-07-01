from collections.abc import Callable

import torch as t

from arena_ext import activation_language as reference


def _solutions():
    from chapter7_activation_to_language.exercises.part1_lenses_patchscopes import solutions

    return solutions


def _assert_report_close(actual: object, expected: object, *, msg: str) -> None:
    actual_dict = actual.__dict__
    expected_dict = expected.__dict__
    assert actual_dict.keys() == expected_dict.keys(), (
        f"{msg} fields should match the independent reference."
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


def test_logit_lens_and_top_tokens_match_reference(
    logit_lens: Callable | None = None,
    top_tokens: Callable | None = None,
):
    solutions = _solutions()
    logit_lens = logit_lens or solutions.logit_lens
    top_tokens = top_tokens or solutions.top_tokens
    residual = t.tensor([[1.0, 0.0], [0.0, 1.0]])
    unembedding = t.tensor([[2.0, 0.0, 1.0], [0.0, 3.0, 1.0]])

    logits = logit_lens(residual, unembedding)
    expected_logits = reference.logit_lens(residual, unembedding)
    t.testing.assert_close(
        logits,
        expected_logits,
        msg="Logit lens should project residual_stream @ unembedding.",
    )
    top_ids, top_probs = top_tokens(logits, k=1)
    expected_ids, expected_probs = reference.top_tokens(logits, k=1)
    t.testing.assert_close(top_ids, expected_ids, msg="Top token ids should match reference.")
    t.testing.assert_close(
        top_probs,
        expected_probs,
        msg="Top token probabilities should come from softmax(logits).",
    )
    assert top_ids.tolist() == [[0], [1]], (
        "Controlled residual directions should decode to token ids 0 and 1."
    )
    print("All tests in `test_logit_lens_and_top_tokens_match_reference` passed!")


def test_tuned_lens_improves_over_logit_lens_on_toy_targets(
    logit_lens: Callable | None = None,
    tuned_lens: Callable | None = None,
    lens_accuracy_report: Callable | None = None,
):
    solutions = _solutions()
    logit_lens = logit_lens or solutions.logit_lens
    tuned_lens = tuned_lens or solutions.tuned_lens
    lens_accuracy_report = lens_accuracy_report or solutions.lens_accuracy_report
    residual = t.tensor([[1.0, 0.0], [0.0, 1.0]])
    unembedding = t.eye(2)
    lens_weight = t.tensor([[0.0, 1.0], [1.0, 0.0]])
    targets = t.tensor([1, 0])

    logit_logits = logit_lens(residual, unembedding)
    tuned_logits = tuned_lens(residual, lens_weight, None, unembedding)
    reference_tuned = reference.tuned_lens(residual, lens_weight, None, unembedding)
    t.testing.assert_close(
        tuned_logits,
        reference_tuned,
        msg="Tuned lens should apply residual @ lens_weight before unembedding.",
    )
    report = lens_accuracy_report(logit_logits, tuned_logits, targets)
    expected = reference.lens_accuracy_report(logit_logits, tuned_logits, targets)
    _assert_report_close(report, expected, msg="Lens accuracy report")
    assert report.logit_lens_accuracy == 0.0 and report.tuned_lens_accuracy == 1.0, (
        "The tuned lens should fix both controlled examples while logit lens misses both."
    )
    assert report.tuned_lens_improves, "Tuned lens should improve over logit lens."
    print("All tests in `test_tuned_lens_improves_over_logit_lens_on_toy_targets` passed!")


def test_attention_lens_decodes_attention_weighted_values(
    attention_lens: Callable | None = None,
):
    solutions = _solutions()
    attention_lens = attention_lens or solutions.attention_lens
    attention = t.tensor([[[1.0, 0.0], [0.25, 0.75]]])
    values = t.tensor([[[1.0, 0.0], [0.0, 2.0]]])
    unembedding = t.eye(2)
    logits = attention_lens(attention, values, unembedding)
    expected = reference.attention_lens(attention, values, unembedding)
    t.testing.assert_close(
        logits,
        expected,
        msg="Attention lens should decode attention_pattern @ value_vectors through unembedding.",
    )
    t.testing.assert_close(logits, t.tensor([[[1.0, 0.0], [0.25, 1.5]]]))
    print("All tests in `test_attention_lens_decodes_attention_weighted_values` passed!")


def test_patchscope_templates_and_accuracy_report(
    patchscope_prompt: Callable | None = None,
    patchscope_accuracy_report: Callable | None = None,
    replace_final_position_activation: Callable | None = None,
):
    solutions = _solutions()
    patchscope_prompt = patchscope_prompt or solutions.patchscope_prompt
    patchscope_accuracy_report = (
        patchscope_accuracy_report or solutions.patchscope_accuracy_report
    )
    replace_final_position_activation = (
        replace_final_position_activation or solutions.replace_final_position_activation
    )
    patchscope_logits = t.tensor([[2.0, 0.0], [0.0, 2.0]])
    text_only_logits = t.tensor([[0.0, 2.0], [0.0, 2.0]])
    targets = t.tensor([0, 1])
    assert patchscope_prompt("entity") == "What entity is represented by <ACT>?", (
        "Entity Patchscope template should name the activation placeholder."
    )
    assert patchscope_prompt("next_token") == "What token will <ACT> become next?", (
        "Next-token Patchscope template should ask for future-token decoding."
    )
    assert patchscope_prompt("fact") == "What fact is stored in <ACT>?", (
        "Fact Patchscope template should ask for factual content."
    )
    report = patchscope_accuracy_report(patchscope_logits, text_only_logits, targets)
    expected = reference.patchscope_accuracy_report(
        patchscope_logits,
        text_only_logits,
        targets,
    )
    _assert_report_close(report, expected, msg="Patchscope accuracy report")
    assert report.patchscope_accuracy == 1.0 and report.text_only_accuracy == 0.5, (
        "Patchscope logits should beat the text-only baseline on the controlled labels."
    )
    assert report.beats_text_only, "Patchscope report should require improvement over text only."
    activations = t.zeros(1, 3, 2)
    source_activation = t.tensor([1.0, -1.0])
    patched = replace_final_position_activation(activations, source_activation)
    expected_patched = reference.replace_final_position_activation(
        activations,
        source_activation,
    )
    t.testing.assert_close(
        patched,
        expected_patched,
        msg="Patchscope helper should replace only the final target-prompt position.",
    )
    assert patched[0, -1].tolist() == [1.0, -1.0], (
        "The final target-prompt position should contain the source activation."
    )
    assert patched[0, :-1].abs().sum().item() == 0.0, (
        "Earlier target-prompt positions should be left unchanged."
    )
    print("All tests in `test_patchscope_templates_and_accuracy_report` passed!")


def test_counterfactual_and_random_activation_controls(
    counterfactual_activation_report: Callable | None = None,
    random_activation_confidence_report: Callable | None = None,
):
    solutions = _solutions()
    counterfactual_activation_report = (
        counterfactual_activation_report or solutions.counterfactual_activation_report
    )
    random_activation_confidence_report = (
        random_activation_confidence_report or solutions.random_activation_confidence_report
    )
    original = t.tensor([2.0, 0.0])
    patched = t.tensor([0.0, 3.0])
    counterfactual = counterfactual_activation_report(original, patched)
    expected_counterfactual = reference.counterfactual_activation_report(original, patched)
    _assert_report_close(counterfactual, expected_counterfactual, msg="Counterfactual report")
    assert counterfactual.changed and counterfactual.original_answer == 0, (
        "Counterfactual activation should change the decoded argmax answer."
    )

    random_logits = t.zeros(3, 4)
    confidence = random_activation_confidence_report(
        random_logits,
        max_allowed_confidence=0.3,
    )
    expected_confidence = reference.random_activation_confidence_report(
        random_logits,
        max_allowed_confidence=0.3,
    )
    _assert_report_close(confidence, expected_confidence, msg="Random confidence report")
    assert abs(confidence.max_confidence - 0.25) < 1e-6 and confidence.passes_low_confidence, (
        "Uniform random logits over four tokens should have max confidence 0.25."
    )
    print("All tests in `test_counterfactual_and_random_activation_controls` passed!")


def test_notebook_contract(run_smoke_test: Callable | None = None):
    if run_smoke_test is None:
        run_smoke_test = _solutions().run_smoke_test
    result = run_smoke_test(cpu=True)
    assert result["logit_lens"]["top_ids"] == [[0], [1]], (
        "Notebook contract should include controlled logit-lens top tokens."
    )
    assert result["tuned_lens"]["tuned_lens_improves"], (
        "Notebook contract should include tuned-lens improvement over logit lens."
    )
    assert result["attention_lens"]["logits"] == [[[1.0, 0.0], [0.0, 1.0]]], (
        "Notebook contract should include an attention-lens decode."
    )
    assert result["patchscope"]["beats_text_only"], (
        "Notebook contract should include Patchscope beating a text-only baseline."
    )
    assert result["counterfactual"]["changed"], (
        "Notebook contract should include a counterfactual activation changing the answer."
    )
    assert result["random_confidence"]["passes_low_confidence"], (
        "Notebook contract should include a low-confidence random-activation control."
    )
    print("All tests in `test_notebook_contract` passed!")
