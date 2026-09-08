from __future__ import annotations

import json
import inspect
from functools import lru_cache
from pathlib import Path
from typing import Callable

import pytest
import torch as t


def _solutions():
    from chapter9_alignment_interpretability.exercises.part5_rome_model_editing import solutions

    return solutions


def test_minimum_norm_rank_one_update_exact(
    minimum_norm_rank_one_update: Callable | None = None,
):
    fn = minimum_norm_rank_one_update or _solutions().minimum_norm_rank_one_update
    weight = t.tensor([[1.0, 2.0], [0.0, -1.0], [2.0, 0.5]], dtype=t.float64)
    key = t.tensor([1.0, -2.0, 0.5], dtype=t.float64)
    desired = t.tensor([-3.0, 4.0], dtype=t.float64)
    delta = fn(weight, key, desired)
    t.testing.assert_close(key @ (weight + delta), desired, rtol=1e-10, atol=1e-10)
    assert t.linalg.matrix_rank(delta) == 1
    print("All tests in `test_minimum_norm_rank_one_update_exact` passed!")


def test_minimum_norm_matches_independent_constrained_oracle(
    minimum_norm_rank_one_update: Callable | None = None,
):
    solutions = _solutions()
    fn = minimum_norm_rank_one_update or solutions.minimum_norm_rank_one_update
    generator = t.Generator().manual_seed(31)
    weight = t.randn(5, 3, generator=generator, dtype=t.float64)
    key = t.randn(5, generator=generator, dtype=t.float64)
    desired = t.randn(3, generator=generator, dtype=t.float64)
    expected = solutions.independent_constrained_solve_oracle(weight, key, desired)
    actual = fn(weight, key, desired)
    t.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    assert "minimum_norm_rank_one_update" not in inspect.getsource(
        solutions.independent_constrained_solve_oracle
    )
    print("All tests in `test_minimum_norm_matches_independent_constrained_oracle` passed!")


def test_minimum_norm_update_rejects_zero_key(
    minimum_norm_rank_one_update: Callable | None = None,
):
    fn = minimum_norm_rank_one_update or _solutions().minimum_norm_rank_one_update
    with pytest.raises(ValueError, match="nonzero"):
        fn(t.zeros(3, 2), t.zeros(3), t.ones(2))
    print("All tests in `test_minimum_norm_update_rejects_zero_key` passed!")


def test_covariance_preconditioned_key_matches_dense_solve(
    covariance_preconditioned_key: Callable | None = None,
):
    fn = covariance_preconditioned_key or _solutions().covariance_preconditioned_key
    generator = t.Generator().manual_seed(4)
    bank = t.randn(7, 5, generator=generator, dtype=t.float64)
    key = t.randn(5, generator=generator, dtype=t.float64)
    ridge = 0.2
    covariance = bank.T @ bank / bank.shape[0] + ridge * t.eye(5, dtype=t.float64)
    expected = t.linalg.solve(covariance, key)
    actual = fn(key, bank, ridge=ridge)
    t.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-9)
    print("All tests in `test_covariance_preconditioned_key_matches_dense_solve` passed!")


def test_course_rome_like_update_matches_declared_equation(
    course_rome_like_rank_one_update: Callable | None = None,
):
    solutions = _solutions()
    fn = course_rome_like_rank_one_update or solutions.course_rome_like_rank_one_update
    generator = t.Generator().manual_seed(5)
    weight = t.randn(6, 4, generator=generator, dtype=t.float64)
    bias = t.randn(4, generator=generator, dtype=t.float64)
    key = t.randn(6, generator=generator, dtype=t.float64)
    desired = t.randn(4, generator=generator, dtype=t.float64)
    bank = t.randn(9, 6, generator=generator, dtype=t.float64)
    ridge = 0.1
    delta = fn(weight, key, desired, bank, output_bias=bias, ridge=ridge)

    direction = solutions.covariance_preconditioned_key(key, bank, ridge=ridge)
    current_module_output = key @ weight + bias
    expected = t.outer(direction / t.dot(key, direction), desired - current_module_output)
    t.testing.assert_close(delta, expected, rtol=1e-9, atol=1e-9)
    assert t.linalg.matrix_rank(delta) == 1
    print("All tests in `test_course_rome_like_update_matches_declared_equation` passed!")


def test_course_update_hits_actual_affine_module_output(
    course_rome_like_rank_one_update: Callable | None = None,
):
    fn = course_rome_like_rank_one_update or _solutions().course_rome_like_rank_one_update
    generator = t.Generator().manual_seed(17)
    linear = t.nn.Linear(7, 5, bias=True, dtype=t.float64)
    with t.no_grad():
        linear.weight.copy_(t.randn(5, 7, generator=generator, dtype=t.float64))
        linear.bias.copy_(t.randn(5, generator=generator, dtype=t.float64))
    key = t.randn(7, generator=generator, dtype=t.float64)
    desired = t.randn(5, generator=generator, dtype=t.float64)
    bank = t.randn(11, 7, generator=generator, dtype=t.float64)
    delta = fn(
        linear.weight.detach().T,
        key,
        desired,
        bank,
        output_bias=linear.bias.detach(),
        ridge=0.2,
    )
    with t.no_grad():
        actual = t.nn.functional.linear(key, linear.weight + delta.T, linear.bias)
    t.testing.assert_close(actual, desired, rtol=1e-9, atol=1e-9)
    print("All tests in `test_course_update_hits_actual_affine_module_output` passed!")


def test_toy_oracle_changes_only_the_requested_association(
    exact_rank_one_toy_oracle: Callable | None = None,
):
    fn = exact_rank_one_toy_oracle or _solutions().exact_rank_one_toy_oracle
    result = fn()
    assert result["delta_rank"] == 1
    t.testing.assert_close(result["edit_after"], result["desired_value"], rtol=0, atol=1e-12)
    t.testing.assert_close(result["protected_after"], result["protected_before"], rtol=0, atol=1e-12)
    assert not t.equal(result["edit_before"], result["edit_after"])
    print("All tests in `test_toy_oracle_changes_only_the_requested_association` passed!")


def test_token_positions_from_offsets_handles_multitoken_subject(
    token_positions_from_offsets: Callable | None = None,
):
    fn = token_positions_from_offsets or _solutions().token_positions_from_offsets
    offsets = [(0, 3), (3, 7), (7, 13), (13, 16), (16, 24)]
    assert fn(offsets, 0, 13) == [0, 1, 2]
    print("All tests in `test_token_positions_from_offsets_handles_multitoken_subject` passed!")


def test_token_positions_ignores_empty_special_token_offsets(
    token_positions_from_offsets: Callable | None = None,
):
    fn = token_positions_from_offsets or _solutions().token_positions_from_offsets
    offsets = [(0, 0), (0, 3), (3, 7), (0, 0)]
    assert fn(offsets, 0, 7) == [1, 2]
    print("All tests in `test_token_positions_ignores_empty_special_token_offsets` passed!")


@lru_cache(maxsize=1)
def _pinned_cpu_model_and_tokenizer():
    return _solutions().load_pinned_model(device="cpu")


def test_continuation_scoring_uses_all_target_tokens(
    continuation_score_table: Callable | None = None,
    model=None,
    tokenizer=None,
):
    fn = continuation_score_table or _solutions().continuation_score_table
    if model is None or tokenizer is None:
        model, tokenizer = _pinned_cpu_model_and_tokenizer()
    prompts = ["The Earth orbits the", "Times Square is in"]
    targets = [" Sun", " New York"]
    result = fn(model, tokenizer, prompts, targets)
    assert result["target_token_counts"][0] == 1
    assert result["target_token_counts"][1] > 1
    assert (result["target_log_probs"] <= 0).all()

    prompt_ids = tokenizer.encode(prompts[1], add_special_tokens=False)
    target_ids = tokenizer.encode(targets[1], add_special_tokens=False)
    joint_ids = [
        tokenizer.encode(prompt + target, add_special_tokens=False)
        for prompt, target in zip(prompts, targets)
    ]
    inputs = tokenizer.pad({"input_ids": joint_ids}, padding=True, return_tensors="pt")
    with t.inference_mode():
        log_probs = model(**inputs).logits.log_softmax(dim=-1)
    positions = t.arange(len(prompt_ids) - 1, len(prompt_ids) + len(target_ids) - 1)
    expected = log_probs[1, positions, t.tensor(target_ids)].sum().cpu()
    t.testing.assert_close(result["target_log_probs"][1], expected, rtol=1e-6, atol=1e-6)
    print("All tests in `test_continuation_scoring_uses_all_target_tokens` passed!")


def test_repetition_fraction_exposes_local_generation_collapse(tokenizer=None, repetition_fn: Callable | None = None):
    if tokenizer is None:
        _, tokenizer = _pinned_cpu_model_and_tokenizer()
    fn = _solutions().continuation_repetition_fraction
    unique = fn(tokenizer, " alpha beta gamma delta")
    repeated = fn(tokenizer, " Rome Rome Rome Rome")
    assert repeated > unique
    assert fn(tokenizer, "") == 0.0
    print("All tests in `test_repetition_fraction_exposes_local_generation_collapse` passed!")


def test_random_rank_one_has_exact_norm_and_rank(
    make_same_norm_random_rank_one: Callable | None = None,
):
    fn = make_same_norm_random_rank_one or _solutions().make_same_norm_random_rank_one
    matrix = fn((11, 7), 3.25, seed=9, dtype=t.float64)
    assert t.linalg.matrix_rank(matrix) == 1
    t.testing.assert_close(matrix.norm(), t.tensor(3.25, dtype=t.float64), rtol=1e-10, atol=1e-10)
    print("All tests in `test_random_rank_one_has_exact_norm_and_rank` passed!")


def test_five_random_controls_are_seeded_distinct_and_norm_matched(
    make_same_norm_random_rank_one: Callable | None = None,
):
    fn = make_same_norm_random_rank_one or _solutions().make_same_norm_random_rank_one
    controls = [fn((8, 3), 2.0, seed=seed, dtype=t.float64) for seed in range(5)]
    repeats = [fn((8, 3), 2.0, seed=seed, dtype=t.float64) for seed in range(5)]
    for first, second in zip(controls, repeats):
        t.testing.assert_close(first, second, rtol=0, atol=0)
        t.testing.assert_close(first.norm(), t.tensor(2.0, dtype=t.float64), rtol=1e-10, atol=1e-10)
    assert all(not t.equal(controls[0], control) for control in controls[1:])
    print("All tests in `test_five_random_controls_are_seeded_distinct_and_norm_matched` passed!")


def _rewrite_bundle(old_log_probs, new_log_probs, ranks):
    old = t.tensor(old_log_probs, dtype=t.float32)
    new = t.tensor(new_log_probs, dtype=t.float32)
    ranks_tensor = t.tensor(ranks)
    return {
        "old_target_log_probs": old,
        "new_target_log_probs": new,
        "new_target_likelihoods": new.exp(),
        "selected_probs": t.stack([old.exp(), new.exp()], dim=1),
        "target_top1": ranks_tensor == 1,
        "full_target_greedy_matches": ranks_tensor == 1,
    }


def _labeled_bundle(
    log_probs,
    ranks,
    top_ids,
    rome_probs=None,
    strata=None,
    counts_as_leakage=None,
):
    ranks_tensor = t.tensor(ranks)
    result = {
        "target_log_probs": t.tensor(log_probs, dtype=t.float32),
        "label_first_top1": ranks_tensor == 1,
        "label_full_greedy": ranks_tensor == 1,
        "top_ids": t.tensor(top_ids),
    }
    if rome_probs is not None:
        result["new_object_first_probs"] = t.tensor(rome_probs, dtype=t.float32)
    if strata is not None:
        result["strata"] = list(strata)
    if counts_as_leakage is not None:
        result["counts_as_leakage"] = t.tensor(counts_as_leakage, dtype=t.bool)
    return result


def test_summarize_edit_metrics_does_not_conflate_pairwise_and_top1(
    summarize_edit_metrics: Callable | None = None,
):
    fn = summarize_edit_metrics or _solutions().summarize_edit_metrics
    base = _rewrite_bundle([-0.2, -0.3], [-3.0, -2.0], [8, 5])
    edited = _rewrite_bundle([-2.0, -1.5], [-1.0, -1.0], [2, 1])
    known_base = _labeled_bundle([-0.2, -0.4], [1, 2], [3, 4])
    known_edited = _labeled_bundle([-0.3, -0.5], [1, 1], [3, 4])
    strata = [
        "target_generalization",
        "context_override",
        "non_target_relation",
        "legitimate_rome_target",
    ]
    leakage = [False, True, True, False]
    adversarial_base = _labeled_bundle(
        [-0.2] * 4, [1, 1, 1, 1], [3, 4, 5, 6], [0.0] * 4, strata, leakage
    )
    adversarial_edited = _labeled_bundle(
        [-0.3] * 4, [1, 2, 1, 1], [3, 7, 5, 6], [0.8, 0.2, 0.4, 0.9], strata, leakage
    )
    side_base = {
        "top_ids": t.tensor([2]),
        "probs": t.tensor([[0.1, 0.1, 0.8]], dtype=t.float32),
    }
    side_edited = {
        "top_ids": t.tensor([2]),
        "probs": t.tensor([[0.1, 0.1, 0.8]], dtype=t.float32),
    }
    metrics = fn(
        base,
        edited,
        known_base,
        known_edited,
        adversarial_base,
        adversarial_edited,
        side_base,
        side_edited,
        t.tensor([10.0, 20.0]),
        t.tensor([11.0, 19.0]),
    )
    assert metrics["pairwise_preference_rate_after"] == 1.0
    assert metrics["target_top1_rate_after"] == 0.5
    assert metrics["known_fact_candidate_count"] == 2
    assert metrics["known_fact_qualified_count"] == 1
    assert metrics["known_fact_top1_retention_after"] == 1.0
    assert metrics["leakage_sensitive_mean_rome_probability_change"] == pytest.approx(0.3)
    assert metrics["legitimate_rome_probability_change"] == pytest.approx(0.9)
    assert metrics["side_effect_mean_kl"] == pytest.approx(0.0, abs=1e-7)
    print("All tests in `test_summarize_edit_metrics_does_not_conflate_pairwise_and_top1` passed!")


def test_covariance_and_evaluation_sets_are_disjoint_and_auditable():
    solutions = _solutions()
    covariance_prompts = {row[0] for row in solutions.COVARIANCE_FACTS}
    known_fact_prompts = {row[0] for row in solutions.KNOWN_FACT_CANDIDATES}
    adversarial_prompts = {record.prompt for record in solutions.ADVERSARIAL_RECORDS}
    assert covariance_prompts.isdisjoint(known_fact_prompts)
    assert covariance_prompts.isdisjoint(solutions.HELDOUT_PARAPHRASES)
    assert known_fact_prompts.isdisjoint(solutions.HELDOUT_PARAPHRASES)
    assert adversarial_prompts.isdisjoint(solutions.HELDOUT_PARAPHRASES)
    assert set(solutions.OPTIMIZATION_PROMPTS).isdisjoint(solutions.HELDOUT_PARAPHRASES)
    records = solutions.OPTIMIZATION_RECORDS + solutions.HELDOUT_PARAPHRASE_RECORDS
    assert len({record.prompt_id for record in records}) == len(records)
    assert all(record.family and record.provenance for record in records)
    assert len({record.prompt_id for record in solutions.ADVERSARIAL_RECORDS}) == len(
        solutions.ADVERSARIAL_RECORDS
    )
    assert all(record.stratum and record.rationale for record in solutions.ADVERSARIAL_RECORDS)
    assert solutions.PARAPHRASE_PROVENANCE["dataset_id"].endswith("v2")
    print("All tests in `test_covariance_and_evaluation_sets_are_disjoint_and_auditable` passed!")


def test_real_evaluation_sizes_are_not_one_prompt_demos():
    solutions = _solutions()
    assert len(solutions.HELDOUT_PARAPHRASES) >= 20
    assert len(solutions.COVARIANCE_FACTS) >= 20
    assert len(solutions.KNOWN_FACT_CANDIDATES) >= 20
    assert len(solutions.ADVERSARIAL_RECORDS) >= 12
    assert len(solutions.SIDE_EFFECT_PROMPTS) >= 20
    assert len(solutions.FLUENCY_TEXTS) >= 10
    print("All tests in `test_real_evaluation_sizes_are_not_one_prompt_demos` passed!")


def test_known_fact_qualification_uses_only_base_top1(qualify: Callable | None = None):
    fn = qualify or _solutions().qualify_known_fact_mask
    mask = fn(
        {"label_first_top1": t.tensor([True, False, True, False])}
    )
    assert mask.tolist() == [True, False, True, False]
    print("All tests in `test_known_fact_qualification_uses_only_base_top1` passed!")


def test_adversarial_leakage_mask_excludes_all_legitimate_rome_targets():
    records = _solutions().ADVERSARIAL_RECORDS
    leakage_records = [record for record in records if record.counts_as_leakage]
    legitimate_rome = [record for record in records if record.stratum == "legitimate_rome_target"]
    target_generalization = [record for record in records if record.stratum == "target_generalization"]
    assert leakage_records
    assert legitimate_rome and target_generalization
    assert all(record.target != _solutions().NEW_OBJECT for record in leakage_records)
    assert all(not record.counts_as_leakage for record in legitimate_rome + target_generalization)
    print("All tests in `test_adversarial_leakage_mask_excludes_all_legitimate_rome_targets` passed!")


def test_activation_patching_conditions_use_nine_independent_directions(condition_fn: Callable | None = None):
    fn = condition_fn or _solutions().independent_noise_conditions
    conditions = fn(
        (1.5, 3.0, 4.5), directions_per_scale=3, seed=17
    )
    seeds = [seed for _, seed in conditions]
    assert len(conditions) == len(set(seeds)) == 9
    directions = [t.randn(64, generator=t.Generator().manual_seed(seed)) for seed in seeds]
    normalized = t.stack([direction / direction.norm() for direction in directions])
    off_diagonal = (normalized @ normalized.T - t.eye(9)).abs()
    assert float(off_diagonal.max()) < 0.5
    print("All tests in `test_activation_patching_conditions_use_nine_independent_directions` passed!")


def test_activation_patching_sweep_rejects_less_than_nine_conditions(condition_fn: Callable | None = None):
    fn = condition_fn or _solutions().independent_noise_conditions
    with pytest.raises(ValueError, match="at least three"):
        fn(
            noise_scales=(1.0, 2.0),
            directions_per_scale=3,
        )
    print("All tests in `test_activation_patching_sweep_rejects_less_than_nine_conditions` passed!")


def test_smoke_contract_is_exact(run_smoke_test: Callable | None = None):
    fn = run_smoke_test or _solutions().run_smoke_test
    result = fn()
    assert result["delta_rank"] == 1
    assert result["write_error"] < 1e-12
    assert result["protected_error"] < 1e-12
    print("All tests in `test_smoke_contract_is_exact` passed!")


def test_model_and_revision_are_pinned():
    solutions = _solutions()
    assert solutions.MODEL_ID == "EleutherAI/pythia-70m-deduped"
    assert len(solutions.MODEL_REVISION) == 40
    assert solutions.MODEL_REVISION != "main"
    print("All tests in `test_model_and_revision_are_pinned` passed!")


def test_section_local_release_report_generation_inputs_are_strict():
    from chapter9_alignment_interpretability.exercises.part5_rome_model_editing import (
        release_report,
    )

    inputs = release_report.build_release_report_generation_inputs()
    assert inputs["runner_callable"] == "run_cpu_release"
    assert inputs["implementation_factory"] == "reference_learner_functions"
    assert inputs["requires_gpu"] is False
    assert inputs["shared_registration"] == "parent_owned_not_modified_by_section"
    manifest = inputs["report_inputs"]
    assert manifest["algorithm"] == "sha256"
    assert len(manifest["combined_sha256"]) == 64
    paths = {row["path"] for row in manifest["files"]}
    for required in (
        "solutions.py",
        "tests.py",
        "build_notebooks.py",
        "artifacts.lock.yml",
        "verification_report.schema.json",
        "9.5_ROME_Model_Editing_solutions.ipynb",
        "05_[9.5]_ROME_Model_Editing.md",
        "rome_model_editing_signature.png",
    ):
        assert any(path.endswith(required) for path in paths)
    assert not any("infrastructure/core/config" in path for path in paths)
    print("All tests in `test_section_local_release_report_generation_inputs_are_strict` passed!")


def test_pinned_model_has_nontrivial_base_known_fact_denominator(model=None, tokenizer=None):
    solutions = _solutions()
    if model is None or tokenizer is None:
        model, tokenizer = _pinned_cpu_model_and_tokenizer()
    scores = solutions.score_labeled_facts(model, tokenizer, solutions.KNOWN_FACT_CANDIDATES)
    mask = solutions.qualify_known_fact_mask(scores)
    assert int(mask.sum()) == 14
    assert int(mask.sum()) < len(solutions.KNOWN_FACT_CANDIDATES)
    print("All tests in `test_pinned_model_has_nontrivial_base_known_fact_denominator` passed!")


def test_real_update_builder_has_immediate_numerical_contract(
    build_update: Callable | None = None,
    model=None,
    tokenizer=None,
):
    solutions = _solutions()
    fn = build_update or solutions.build_course_rome_like_update
    if model is None or tokenizer is None:
        model, tokenizer = _pinned_cpu_model_and_tokenizer()
    update, details = fn(model, tokenizer, layer=0, steps=2)
    assert int(t.linalg.matrix_rank(update.float(), tol=1e-5)) == 1
    assert details["write_error"] < 1e-4
    assert details["optimization_history"][-1]["target_nll"] < details["optimization_history"][0][
        "target_nll"
    ]
    print(
        "builder integration:",
        {"rank": 1, "write_error": details["write_error"], "update_norm": details["update_norm"]},
    )


def test_real_evaluator_zero_edit_has_immediate_numerical_contract(
    evaluator: Callable | None = None,
    model=None,
    tokenizer=None,
    rewrite_scorer: Callable | None = None,
    fact_scorer: Callable | None = None,
    adversarial_scorer: Callable | None = None,
    metric_summarizer: Callable | None = None,
):
    solutions = _solutions()
    fn = evaluator or solutions.evaluate_weight_edit
    if model is None or tokenizer is None:
        model, tokenizer = _pinned_cpu_model_and_tokenizer()
    rewrite_scorer = rewrite_scorer or solutions.score_rewrite_prompts
    fact_scorer = fact_scorer or solutions.score_labeled_facts
    adversarial_scorer = adversarial_scorer or solutions.score_adversarial_records
    metric_summarizer = metric_summarizer or solutions.summarize_edit_metrics
    shape = model.gpt_neox.layers[0].mlp.dense_4h_to_h.weight.T.shape
    result = fn(
        model,
        tokenizer,
        layer=0,
        delta_weight=t.zeros(shape),
        rewrite_scorer=rewrite_scorer,
        fact_scorer=fact_scorer,
        adversarial_scorer=adversarial_scorer,
        metric_summarizer=metric_summarizer,
    )
    metrics = result["metrics"]
    assert metrics["known_fact_qualified_count"] == 14
    assert metrics["known_fact_top1_retention_after"] == 1.0
    assert metrics["side_effect_mean_kl"] == pytest.approx(0.0, abs=1e-8)
    assert metrics["fluency_perplexity_ratio"] == pytest.approx(1.0, abs=1e-7)
    print(
        "evaluator integration:",
        {
            "qualified_known_facts": metrics["known_fact_qualified_count"],
            "zero_edit_kl": metrics["side_effect_mean_kl"],
        },
    )


def check_learner_bundle_uses_notebook_functions(implementation) -> None:
    reference = _solutions().reference_learner_functions()
    for field in implementation.__dataclass_fields__:
        learner_fn = getattr(implementation, field)
        assert callable(learner_fn)
        assert learner_fn is not getattr(reference, field)
    print("Final experiment is wired to learner-written notebook functions.")


def test_notebook_contract():
    section = Path(__file__).resolve().parent
    for name in [
        "9.5_ROME_Model_Editing_exercises.ipynb",
        "9.5_ROME_Model_Editing_solutions.ipynb",
    ]:
        notebook = json.loads((section / name).read_text())
        source = "\n".join(
            "".join(cell.get("source", []))
            if isinstance(cell.get("source", []), list)
            else str(cell.get("source", ""))
            for cell in notebook["cells"]
        )
        for required in [
            "affine module output",
            "course ROME-like",
            "official ROME parity is not established",
            "Wikipedia-derived covariance",
            "random-prefix",
            "value-objective parity",
            "independent_constrained_solve_oracle",
            "continuation_score_table",
            "target top-1",
            "full target likelihood",
            "per-paraphrase",
            "activation_patching_sweep",
            "layer-by-position",
            "nine independent",
            "negative localization result",
            "norm-matched layer",
            "wrong-position",
            "five same-norm random rank-one",
            "shuffled-city targets",
            "ADVERSARIAL_RECORDS",
            "legitimate_rome_target",
            "COVARIANCE_FACTS",
            "KNOWN_FACT_CANDIDATES",
            "base-qualified",
            "large value-optimizer KL",
            "LearnerFunctions",
            "Try It Yourself",
            "single-token and multi-token",
            "scoring-only",
            "Limitations",
        ]:
            assert required in source, f"{name} is missing learner-visible surface {required!r}."
        for forbidden in ["official ROME semantics", "causal_trace_sweep", "NEIGHBORHOOD_FACTS"]:
            assert forbidden not in source
    solution = json.loads(
        (section / "9.5_ROME_Model_Editing_solutions.ipynb").read_text()
    )
    solution_code = "\n".join(
        "".join(cell.get("source", []))
        for cell in solution["cells"]
        if cell["cell_type"] == "code"
    )
    from arena_ext.notebook_checks import unfinished_definitions
    assert not unfinished_definitions(solution_code)
    assert "check_learner_bundle_uses_notebook_functions(learner_functions)" in solution_code
    print("All tests in `test_notebook_contract` passed!")


def test_reference_artifact_reports_the_observed_negative_result():
    section = Path(__file__).resolve().parent
    artifact = json.loads((section / "expected_outputs" / "reference_metrics.json").read_text())
    assert artifact["model_id"] == _solutions().MODEL_ID
    assert artifact["model_revision"] == _solutions().MODEL_REVISION
    assert artifact["method_scope"] == "course_rome_like_rank_one_edit"
    assert artifact["official_rome_parity_established"] is False
    assert len(artifact["missing_official_parity_components"]) == 3
    assert artifact["covariance_context_count"] == len(_solutions().COVARIANCE_FACTS)
    assert artifact["suite_sizes"] == {
        "adversarial": 16,
        "fluency": 10,
        "known_fact_candidates": 24,
        "known_fact_qualified": 14,
        "paraphrases": 20,
        "side_effect": 20,
    }
    assert len(artifact["paraphrase_rows"]) == 20
    assert len(artifact["known_fact_rows"]) == 24
    assert sum(row["base_qualified_known_fact"] for row in artifact["known_fact_rows"]) == 14
    assert len(artifact["adversarial_rows"]) == 16
    assert all(
        not row["counts_as_leakage"]
        for row in artifact["adversarial_rows"]
        if row["stratum"] in {"legitimate_rome_target", "target_generalization"}
    )
    assert len(artifact["layer_sweep"]) == 6
    assert all(row["norm_matched"] for row in artifact["layer_sweep"])
    norms = [row["applied_update_norm"] for row in artifact["layer_sweep"]]
    assert max(norms) - min(norms) < 1e-4
    assert len(artifact["control_details"]["random_updates"]) == 5
    primary = artifact["metrics"]["course_rome_like"]
    assert primary["pairwise_preference_rate_after"] == 1.0
    assert primary["target_top1_rate_after"] < primary["pairwise_preference_rate_after"]
    assert primary["known_fact_qualified_count"] == 14
    assert primary["leakage_sensitive_mean_rome_probability_change"] > 0.1
    assert (
        primary["localized_louvre_repetition_after"]
        > primary["localized_louvre_repetition_before"]
    )
    assert primary["generation_repetition_failure_count"] >= 1
    assert artifact["rank_one_update"]["write_error"] < 1e-4
    assert artifact["rank_one_update"]["effective_rank"] == 1
    assert artifact["rank_one_update"]["optimization_preservation_kl_is_large"] is True
    assert artifact["localization_result"]["negative_localization_result"] is True
    assert artifact["localization_result"]["supports_privileged_layer_claim"] is False
    assert len(artifact["localization_result"]["norm_matched_successful_layers"]) > 1
    assert len(set(artifact["trace"]["direction_seeds"])) == 9
    assert all("rewrite_success" not in row for row in artifact["metrics"].values())
    print("All tests in `test_reference_artifact_reports_the_observed_negative_result` passed!")


def test_solution_notebook_is_fully_executed_without_errors():
    section = Path(__file__).resolve().parent
    notebook = json.loads((section / "9.5_ROME_Model_Editing_solutions.ipynb").read_text())
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code_cells and all(cell["execution_count"] is not None for cell in code_cells)
    errors = [
        output
        for cell in code_cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    assert errors == []
    print("All tests in `test_solution_notebook_is_fully_executed_without_errors` passed!")


def test_temporary_edit_restores_weights_after_errors(edit_context: Callable | None = None):
    from types import SimpleNamespace

    fn = edit_context or _solutions().temporary_mlp_weight_edit
    projection = t.nn.Linear(3, 2)
    model = SimpleNamespace(gpt_neox=SimpleNamespace(layers=[
        SimpleNamespace(mlp=SimpleNamespace(dense_4h_to_h=projection))
    ]))
    original = projection.weight.detach().clone()
    original_bias = projection.bias.detach().clone()
    update = t.arange(6, dtype=original.dtype).reshape(3, 2) / 10
    with pytest.raises(RuntimeError, match="deliberate evaluation failure"):
        with fn(model, 0, update):
            t.testing.assert_close(projection.weight, original + update.T, rtol=0, atol=0)
            raise RuntimeError("deliberate evaluation failure")
    t.testing.assert_close(projection.weight, original, rtol=0, atol=0)
    t.testing.assert_close(projection.bias, original_bias, rtol=0, atol=0)
    with pytest.raises(ValueError, match="shape"):
        with fn(model, 0, t.zeros(2, 3)):
            raise AssertionError("A transposed update must be rejected before applying it.")
    t.testing.assert_close(projection.weight, original, rtol=0, atol=0)
    print("All tests in `test_temporary_edit_restores_weights_after_errors` passed!")


def test_capture_mlp_outputs_matches_direct_modules(capture: Callable | None = None, model=None, tokenizer=None):
    solutions = _solutions()
    fn = capture or solutions.capture_mlp_outputs
    if model is None or tokenizer is None:
        model, tokenizer = _pinned_cpu_model_and_tokenizer()
    blocks = [layer.mlp for layer in model.gpt_neox.layers]
    hook_counts = [len(block._forward_hooks) for block in blocks]
    inputs = []
    handles = [block.register_forward_pre_hook(
        lambda _module, args: inputs.append(args[0].detach().clone())
    ) for block in blocks]
    try:
        actual = fn(model, tokenizer, "The Louvre is located in")
    finally:
        for handle in handles:
            handle.remove()
    assert len(actual) == len(inputs) == len(blocks), "Capture one complete output per MLP in layer order."
    with t.inference_mode():
        for index, block in enumerate(blocks):
            t.testing.assert_close(actual[index], block(inputs[index]), rtol=1e-5, atol=1e-6)
    assert [len(block._forward_hooks) for block in blocks] == hook_counts, "Remove capture hooks after the forward pass."
    print("All tests in `test_capture_mlp_outputs_matches_direct_modules` passed!")


def test_value_optimizer_history_describes_returned_delta(optimize: Callable | None = None, model=None, tokenizer=None):
    solutions = _solutions()
    fn = optimize or solutions.optimize_value_delta
    if model is None or tokenizer is None:
        model, tokenizer = _pinned_cpu_model_and_tokenizer()
    prompts = ["The Louvre is located in", "Visitors can find the Louvre in"]
    targets = [" Rome", " New York"]
    preserve = ["The Louvre is a"]
    all_prompts = prompts + preserve
    positions = [solutions.locate_subject_tokens(tokenizer, prompt, "Louvre")[-1] for prompt in all_prompts]
    prefixes = [tokenizer.encode(prompt, add_special_tokens=False) for prompt in prompts]
    target_ids = [tokenizer.encode(target, add_special_tokens=False) for target in targets]
    assert len(target_ids[1]) > 1, "The second target must exercise teacher forcing beyond its first token."
    joint = [prefix + target[:-1] for prefix, target in zip(prefixes, target_ids)]
    joint += [tokenizer.encode(preserve[0], add_special_tokens=False)]
    device = next(model.parameters()).device
    encoded = tokenizer.pad({"input_ids": joint}, padding=True, return_tensors="pt")
    batch = {name: value.to(device) for name, value in encoded.items()}
    preserve_last = len(joint[-1]) - 1
    with t.inference_mode():
        base_logp = model(**batch).logits[-1, preserve_last].log_softmax(-1)
    block = model.gpt_neox.layers[0].mlp
    hook_count = len(block._forward_hooks)
    delta, history = fn(model, tokenizer, prompts, "Louvre", targets, 0, preservation_prompts=preserve, steps=2)
    assert len(history) == 3, "Record the initial state and the result after each of two optimizer steps."
    assert history[-1]["delta_norm"] == pytest.approx(float(delta.norm()), rel=1e-6)

    def inject(_module, _inputs, output):
        patched = output.clone()
        patched[t.arange(3, device=device), t.tensor(positions, device=device)] += delta
        return patched

    handle = block.register_forward_hook(inject)
    try:
        with t.inference_mode():
            logp = model(**batch).logits.log_softmax(-1)
    finally:
        handle.remove()
    nlls = []
    for row, (prefix, target) in enumerate(zip(prefixes, target_ids)):
        scores = [logp[row, len(prefix) - 1 + offset, token] for offset, token in enumerate(target)]
        nlls.append(-t.stack(scores).mean())
    expected_nll = float(t.stack(nlls).mean())
    expected_kl = float((base_logp.exp() * (base_logp - logp[-1, preserve_last])).sum())
    assert history[-1]["target_nll"] == pytest.approx(expected_nll, rel=1e-5, abs=1e-6)
    assert history[-1]["preservation_kl"] == pytest.approx(expected_kl, rel=1e-5, abs=1e-6)
    assert len(block._forward_hooks) == hook_count, "Optimization and evaluation must both remove their hooks."
    print("All tests in `test_value_optimizer_history_describes_returned_delta` passed!")


def test_answer_dropdowns_complete_every_learner_definition():
    import ast
    import re
    from arena_ext.notebook_checks import unfinished_definitions

    section = Path(__file__).resolve().parent
    learner = json.loads((section / "9.5_ROME_Model_Editing_exercises.ipynb").read_text())
    solved = json.loads((section / "9.5_ROME_Model_Editing_solutions.ipynb").read_text())
    checked = 0
    for index, (cell, answer_cell) in enumerate(zip(learner["cells"], solved["cells"], strict=True)):
        if cell["cell_type"] != "code":
            for block in re.findall(r"```python\n(.*?)```", "".join(cell["source"]), re.S):
                ast.parse(block)
            continue
        source = "".join(cell["source"])
        if not unfinished_definitions(source):
            assert ast.dump(ast.parse(source)) == ast.dump(ast.parse("".join(answer_cell["source"])))
            continue
        following = "\n".join("".join(c["source"]) for c in learner["cells"][index + 1:index + 4] if c["cell_type"] == "markdown")
        dropdown = re.search(r"<summary>Solution</summary>(.*?)</details>", following, re.S)
        assert dropdown, f"Cell {index} needs an adjacent answer."
        answer = "\n".join(re.findall(r"```python\n(.*?)```", dropdown[1], re.S))
        expected_tree = ast.parse("".join(answer_cell["source"]))
        assert ast.dump(ast.parse(answer)) == ast.dump(expected_tree), f"Cell {index} answer drifted from the solved notebook."
        learner_tree = ast.parse(source)
        expected = {node.name: node for node in expected_tree.body if isinstance(node, ast.FunctionDef)}
        assert {node.name for node in learner_tree.body if isinstance(node, ast.FunctionDef)} == set(expected), (
            f"Cell {index} is missing a helper or a method stub."
        )
        for offset, node in enumerate(learner_tree.body):
            if isinstance(node, ast.FunctionDef) and unfinished_definitions(ast.Module(body=[node], type_ignores=[])):
                learner_tree.body[offset] = expected[node.name]
        assert ast.dump(learner_tree) == ast.dump(expected_tree), f"Cell {index} contains mismatched provided code."
        checked += 1
    assert checked == 13
    print("All tests in `test_answer_dropdowns_complete_every_learner_definition` passed!")
