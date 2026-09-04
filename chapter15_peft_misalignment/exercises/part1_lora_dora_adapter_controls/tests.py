from collections.abc import Callable
import json
from pathlib import Path
from typing import Any

import torch as t

from arena_ext import peft_interpretability as reference


def _solutions():
    from chapter15_peft_misalignment.exercises.part1_lora_dora_adapter_controls import (
        solutions,
    )

    return solutions


def _section_dir() -> Path:
    return Path(__file__).resolve().parent


def _verification_report() -> dict:
    return json.loads((_section_dir() / "verification_report.json").read_text())


def _as_dict(report: object) -> dict[str, Any]:
    if isinstance(report, dict):
        return report
    return report.__dict__


def _assert_close(actual: Any, expected: Any, *, msg: str, atol: float = 1e-6) -> None:
    if isinstance(expected, t.Tensor):
        assert isinstance(actual, t.Tensor), f"{msg} should be a torch.Tensor."
        assert t.allclose(actual, expected, atol=atol, rtol=0), (
            f"{msg} should match the reference tensor.\n"
            f"Expected: {expected}\nActual: {actual}"
        )
        return
    if isinstance(expected, dict):
        assert isinstance(actual, dict), f"{msg} should be a dictionary-like object."
        assert actual.keys() == expected.keys(), (
            f"{msg} should expose the same fields as the independent reference."
        )
        for key, expected_value in expected.items():
            _assert_close(actual[key], expected_value, msg=f"{msg}.{key}", atol=atol)
        return
    if isinstance(expected, float):
        assert abs(float(actual) - expected) <= atol, (
            f"{msg} should be {expected}, got {actual}."
        )
        return
    assert actual == expected, f"{msg} should be {expected!r}, got {actual!r}."


def _assert_report_close(actual: object, expected: object, *, msg: str) -> None:
    _assert_close(_as_dict(actual), _as_dict(expected), msg=msg)


def test_lora_delta_uses_scaled_b_matrix_times_a_matrix(
    lora_delta: Callable | None = None,
):
    lora_delta = lora_delta or _solutions().lora_delta
    lora_a = t.tensor([[1.0, 2.0]])
    lora_b = t.tensor([[3.0], [4.0]])
    delta = lora_delta(lora_a, lora_b, alpha=2.0)
    expected = reference.lora_delta(lora_a, lora_b, alpha=2.0)

    assert t.allclose(delta, expected), (
        "LoRA delta should be (alpha / rank) * (B @ A) with shape (out, in)."
    )
    assert delta.tolist() == [[6.0, 12.0], [8.0, 16.0]], (
        "The one-rank fixture has an exact visible delta; transposing A/B gives a different matrix."
    )
    try:
        lora_delta(t.ones(2), lora_b, alpha=2.0)
    except ValueError as exc:
        assert "matrices" in str(exc), (
            "Invalid LoRA inputs should raise a helpful matrix-shape error."
        )
    else:
        raise AssertionError("lora_delta should reject non-matrix LoRA factors.")
    print("All tests in `test_lora_delta_uses_scaled_b_matrix_times_a_matrix` passed!")


def test_adapter_delta_report_records_rank_alpha_and_nonzero_update(
    adapter_delta_report: Callable | None = None,
):
    adapter_delta_report = adapter_delta_report or _solutions().adapter_delta_report
    lora_a = t.tensor([[1.0, 2.0]])
    lora_b = t.tensor([[3.0], [4.0]])
    report = adapter_delta_report(lora_a, lora_b, alpha=2.0)
    expected = reference.adapter_delta_report(lora_a, lora_b, alpha=2.0)
    _assert_report_close(report, expected, msg="Adapter delta report")

    assert report.rank == 1 and report.alpha == 2.0, (
        "The report should expose the adapter rank and scaling alpha used in the update."
    )
    assert report.nonzero_update, (
        "A nonzero B @ A update should not be accepted as an all-zero adapter."
    )
    zero_report = adapter_delta_report(t.zeros(1, 2), t.zeros(2, 1), alpha=2.0)
    assert not zero_report.nonzero_update, (
        "An all-zero adapter should fail the nonzero-update sanity check."
    )
    print("All tests in `test_adapter_delta_report_records_rank_alpha_and_nonzero_update` passed!")


def test_lora_delta_rank_bound_and_svd_spectrum(lora_delta: Callable | None = None):
    lora_delta = lora_delta or _solutions().lora_delta
    lora_a = t.tensor(
        [
            [1.0, 0.0, 2.0],
            [0.0, 3.0, -1.0],
        ]
    )
    lora_b = t.tensor(
        [
            [2.0, 0.0],
            [0.0, -1.0],
            [1.0, 1.0],
            [-2.0, 0.5],
        ]
    )
    delta = lora_delta(lora_a, lora_b, alpha=4.0)
    expected = (4.0 / 2) * (lora_b @ lora_a)
    singular_values = t.linalg.svdvals(delta)

    assert t.allclose(delta, expected), (
        "The SVD fixture should still use the exact scaled B @ A update."
    )
    assert int(t.linalg.matrix_rank(delta, tol=1e-5).item()) <= lora_a.shape[0], (
        "LoRA updates must have numerical rank no larger than the declared rank."
    )
    assert singular_values.numel() == min(delta.shape), (
        "The SVD should be computed on the actual weight update, not on A or B."
    )
    assert singular_values[0] > singular_values[1] > 1e-5, (
        "This rank-2 fixture should expose two nonzero singular values."
    )
    if singular_values.numel() > 2:
        assert singular_values[2] <= 1e-5, (
            "No third singular value should appear through a rank-2 bottleneck."
        )

    degenerate_b = t.tensor([[1.0, 2.0], [2.0, 4.0], [-1.0, -2.0]])
    degenerate_delta = lora_delta(lora_a, degenerate_b, alpha=2.0)
    assert int(t.linalg.matrix_rank(degenerate_delta, tol=1e-5).item()) == 1, (
        "A declared rank-2 adapter can collapse to numerical rank 1, but not above 2."
    )
    print("All tests in `test_lora_delta_rank_bound_and_svd_spectrum` passed!")


def test_lora_merge_unmerge_parity_is_independent_of_report(
    lora_merge_max_abs_diff: Callable | None = None,
):
    lora_merge_max_abs_diff = (
        lora_merge_max_abs_diff or _solutions().lora_merge_max_abs_diff
    )
    inputs = t.tensor(
        [
            [1.0, -1.0, 0.5],
            [0.0, 2.0, -3.0],
            [-2.0, 1.0, 1.5],
        ]
    )
    base_weight = t.tensor(
        [
            [0.5, -1.0, 0.0],
            [1.5, 0.25, -0.75],
        ]
    )
    lora_a = t.tensor([[1.0, 2.0, -1.0]])
    lora_b = t.tensor([[0.5], [-1.5]])

    original_base = base_weight.clone()
    max_diff = lora_merge_max_abs_diff(
        inputs,
        base_weight,
        lora_a,
        lora_b,
        alpha=3.0,
    )
    assert max_diff <= 1e-6, (
        "Merged and unmerged LoRA logits should agree to numerical precision."
    )
    assert t.equal(base_weight, original_base), (
        "The merge-parity helper should not mutate the frozen base weight."
    )
    try:
        lora_merge_max_abs_diff(inputs, base_weight[:1], lora_a, lora_b, alpha=3.0)
    except ValueError as exc:
        assert "same shape" in str(exc), (
            "Bad merge shapes should raise an explicit base-weight shape error."
        )
    else:
        raise AssertionError("merge parity should reject mismatched base/update shapes.")
    print("All tests in `test_lora_merge_unmerge_parity_is_independent_of_report` passed!")


def test_dora_recompose_weight_preserves_target_row_magnitudes(
    dora_recompose_weight: Callable | None = None,
    dora_weight_report: Callable | None = None,
):
    dora_recompose_weight = dora_recompose_weight or _solutions().dora_recompose_weight
    dora_weight_report = dora_weight_report or _solutions().dora_weight_report
    base_weight = t.tensor([[3.0, 4.0], [0.0, 2.0]])
    adapter_delta = t.zeros_like(base_weight)
    magnitude = t.tensor([10.0, 5.0])

    recomposed = dora_recompose_weight(base_weight, adapter_delta, magnitude)
    expected_recomposed = reference.dora_recompose_weight(
        base_weight,
        adapter_delta,
        magnitude,
    )
    assert t.allclose(recomposed, expected_recomposed), (
        "DoRA recomposition should normalize the updated direction then apply learned magnitudes."
    )
    assert t.allclose(recomposed.norm(dim=-1), magnitude), (
        "Each recomposed output row should have the requested DoRA magnitude."
    )

    report = dora_weight_report(base_weight, adapter_delta, magnitude)
    expected_report = reference.dora_weight_report(base_weight, adapter_delta, magnitude)
    _assert_report_close(report, expected_report, msg="DoRA row-norm report")
    assert report.norm_preserved, (
        "The exact DoRA fixture should pass the row-magnitude preservation check."
    )
    try:
        dora_recompose_weight(base_weight, adapter_delta, t.ones(3))
    except ValueError as exc:
        assert "magnitude" in str(exc), (
            "Bad DoRA magnitude shapes should raise an explicit magnitude error."
        )
    else:
        raise AssertionError("DoRA recomposition should reject mismatched magnitude shape.")
    print("All tests in `test_dora_recompose_weight_preserves_target_row_magnitudes` passed!")


def test_dora_recomposition_handles_nonzero_delta_direction(
    dora_recompose_weight: Callable | None = None,
):
    dora_recompose_weight = dora_recompose_weight or _solutions().dora_recompose_weight
    base_weight = t.tensor([[3.0, 4.0, 0.0], [0.0, 2.0, 0.0]])
    adapter_delta = t.tensor([[1.0, -2.0, 2.0], [2.0, 0.0, 1.0]])
    magnitude = t.tensor([7.0, 3.0])
    direction = base_weight + adapter_delta

    recomposed = dora_recompose_weight(base_weight, adapter_delta, magnitude)
    t.testing.assert_close(recomposed.norm(dim=-1), magnitude, atol=1e-6, rtol=0)
    cosine = t.nn.functional.cosine_similarity(recomposed, direction, dim=-1)
    t.testing.assert_close(cosine, t.ones_like(cosine), atol=1e-6, rtol=0)
    assert not t.allclose(recomposed, direction * magnitude.unsqueeze(-1)), (
        "DoRA should normalize each updated direction before applying magnitudes."
    )
    print("All tests in `test_dora_recomposition_handles_nonzero_delta_direction` passed!")


def test_intruder_dimension_report_measures_projection_fraction(
    intruder_dimension_report: Callable | None = None,
):
    intruder_dimension_report = (
        intruder_dimension_report or _solutions().intruder_dimension_report
    )
    adapter_delta = t.tensor([[1.0, 0.0], [1.0, 0.0]])
    protected_direction = t.tensor([1.0, 0.0])
    report = intruder_dimension_report(
        adapter_delta,
        protected_direction,
        max_projection_fraction=0.5,
    )
    expected = reference.intruder_dimension_report(
        adapter_delta,
        protected_direction,
        max_projection_fraction=0.5,
    )
    _assert_report_close(report, expected, msg="Protected-direction report")

    assert abs(report.projection_fraction - 1.0) < 1e-6, (
        "The adapter update lies entirely in the protected direction in this fixture."
    )
    assert report.intruder_detected, (
        "A projection fraction above the threshold should be flagged before deployment."
    )
    orthogonal = intruder_dimension_report(
        adapter_delta,
        t.tensor([0.0, 1.0]),
        max_projection_fraction=0.5,
    )
    assert not orthogonal.intruder_detected, (
        "An orthogonal protected direction should be a negative control, not an intruder."
    )
    print("All tests in `test_intruder_dimension_report_measures_projection_fraction` passed!")


def test_adapter_mechanism_report_requires_accuracy_and_mechanism(
    adapter_mechanism_report: Callable | None = None,
):
    adapter_mechanism_report = (
        adapter_mechanism_report or _solutions().adapter_mechanism_report
    )
    report = adapter_mechanism_report(
        adapter_accuracy=0.9,
        baseline_accuracy=0.7,
        adapter_mechanism_score=0.8,
        baseline_mechanism_score=0.75,
        min_accuracy_gain=0.1,
        min_mechanism_delta=-0.02,
    )
    expected = reference.adapter_mechanism_report(
        adapter_accuracy=0.9,
        baseline_accuracy=0.7,
        adapter_mechanism_score=0.8,
        baseline_mechanism_score=0.75,
        min_accuracy_gain=0.1,
        min_mechanism_delta=-0.02,
    )
    _assert_report_close(report, expected, msg="Adapter mechanism report")

    assert report.accuracy_improved and report.mechanism_preserved, (
        "This fixture should pass both the accuracy-gain and mechanism-preservation gates."
    )
    assert report.adapter_acceptable, (
        "An adapter is acceptable only when both gates pass."
    )
    broken_mechanism = adapter_mechanism_report(
        adapter_accuracy=0.95,
        baseline_accuracy=0.7,
        adapter_mechanism_score=0.2,
        baseline_mechanism_score=0.75,
        min_accuracy_gain=0.1,
        min_mechanism_delta=-0.02,
    )
    assert not broken_mechanism.adapter_acceptable, (
        "High task accuracy should still fail when the measured mechanism is damaged."
    )
    print("All tests in `test_adapter_mechanism_report_requires_accuracy_and_mechanism` passed!")


def test_smoke_wrappers_match_the_visible_contract(
    lora_smoke_test: Callable | None = None,
    dora_smoke_test: Callable | None = None,
    intruder_smoke_test: Callable | None = None,
    mechanism_smoke_test: Callable | None = None,
):
    solutions = _solutions()
    lora_smoke_test = lora_smoke_test or solutions.lora_smoke_test
    dora_smoke_test = dora_smoke_test or solutions.dora_smoke_test
    intruder_smoke_test = intruder_smoke_test or solutions.intruder_smoke_test
    mechanism_smoke_test = mechanism_smoke_test or solutions.mechanism_smoke_test

    lora = lora_smoke_test()
    dora = dora_smoke_test()
    intruder = intruder_smoke_test()
    mechanism = mechanism_smoke_test()
    assert lora["delta"] == [[6.0, 12.0], [8.0, 16.0]], (
        "The LoRA smoke test should expose the exact low-rank delta."
    )
    assert lora["report"]["rank"] == 1 and lora["report"]["nonzero_update"], (
        "The LoRA smoke test should report a rank-1 nonzero adapter."
    )
    assert dora["row_norms"] == [10.0, 5.0] and dora["norm_preserved"], (
        "The DoRA smoke test should preserve the two requested row magnitudes."
    )
    assert abs(intruder["projection_fraction"] - 1.0) < 1e-6, (
        "The intruder smoke test should report full projection onto the monitored direction."
    )
    assert intruder["intruder_detected"], (
        "The intruder smoke test should flag the controlled positive example."
    )
    assert mechanism["adapter_acceptable"], (
        "The mechanism smoke test should pass the paired accuracy and mechanism gates."
    )
    print("All tests in `test_smoke_wrappers_match_the_visible_contract` passed!")


def test_notebook_contract(run_smoke_test: Callable | None = None):
    run_smoke_test = run_smoke_test or _solutions().run_smoke_test
    result = run_smoke_test(cpu=True)
    assert result["lora"]["delta"] == [[6.0, 12.0], [8.0, 16.0]], (
        "The notebook contract should include the exact LoRA delta fixture."
    )
    assert result["lora"]["report"]["nonzero_update"], (
        "The notebook contract should include a nonzero LoRA update check."
    )
    assert result["dora"]["norm_preserved"], (
        "The notebook contract should include the exact DoRA norm-preservation check."
    )
    assert result["intruder"]["intruder_detected"], (
        "The notebook contract should include the protected-direction positive control."
    )
    assert result["mechanism"]["adapter_acceptable"], (
        "The notebook contract should include the paired accuracy/mechanism gate."
    )
    print("All tests in `test_notebook_contract` passed!")


def test_proxy_batch_has_planted_target_and_ood_conflict(
    make_proxy_batch: Callable | None = None,
):
    make_proxy_batch = make_proxy_batch or _solutions().make_proxy_batch
    inputs, labels = make_proxy_batch(batch=1024, seed=0)
    assert inputs.shape == (1024, 8), (
        "The proxy batch should expose 8 activation-like coordinates per example."
    )
    assert t.equal(labels, (inputs[:, 0] > 0).long()), (
        "Labels must be planted in feature 0 so the ground-truth mechanism is inspectable."
    )

    ood_inputs, ood_labels = make_proxy_batch(batch=1024, seed=1, ood_shift=0.75)
    base_weight = t.zeros(2, 8)
    base_weight[0, 1] = 1.0
    base_weight[1, 1] = -1.0
    base_accuracy = (ood_inputs @ base_weight.T).argmax(dim=-1).eq(ood_labels).float().mean()
    assert float(base_accuracy.item()) <= 0.05, (
        "The OOD split should put the frozen distractor direction in conflict with the label."
    )
    print("All tests in `test_proxy_batch_has_planted_target_and_ood_conflict` passed!")


def test_train_proxy_adapter_returns_learning_trace_on_cpu(
    train_proxy_adapter: Callable | None = None,
    evaluate_proxy_adapter: Callable | None = None,
):
    solutions = _solutions()
    train_proxy_adapter = train_proxy_adapter or solutions.train_proxy_adapter
    evaluate_proxy_adapter = evaluate_proxy_adapter or solutions.evaluate_proxy_adapter
    model, trace = train_proxy_adapter(method="lora", rank=1, alpha=4.0, seed=0, steps=60)
    report = evaluate_proxy_adapter(model, seed=12345, batch=2048)

    assert trace["method"] == "lora" and trace["rank"] == 1, (
        "The training trace should preserve the learner's method and rank settings."
    )
    assert len(trace["step"]) >= 3 and trace["loss"][0] > trace["loss"][-1], (
        "The trace should expose a real decreasing training curve."
    )
    assert report["accuracy"] >= 0.95 and report["baseline_accuracy"] <= 0.65, (
        "The trained LoRA adapter should solve the planted target task while the frozen baseline fails."
    )
    assert report["target_direction_cosine"] >= 0.95, (
        "The learned decision direction should align with the planted target feature."
    )
    assert report["protected_abs_cosine"] <= 0.2, (
        "The learned adapter should not drift strongly into the protected feature direction."
    )
    assert report["activation_target_corr"] >= 0.95, (
        "The margin drift should be explained by the target activation coordinate."
    )
    print("All tests in `test_train_proxy_adapter_returns_learning_trace_on_cpu` passed!")


def test_same_norm_random_adapter_control_fails_on_cpu(
    train_proxy_adapter: Callable | None = None,
    same_norm_random_adapter_control: Callable | None = None,
):
    solutions = _solutions()
    train_proxy_adapter = train_proxy_adapter or solutions.train_proxy_adapter
    same_norm_random_adapter_control = (
        same_norm_random_adapter_control or solutions.same_norm_random_adapter_control
    )
    model, _trace = train_proxy_adapter(method="lora", rank=1, alpha=4.0, seed=0, steps=60)
    control = same_norm_random_adapter_control(model, seed=777, eval_seed=54321)

    assert control["control_fails"], (
        "A same-norm random adapter should not be accepted as a learned mechanism."
    )
    assert control["update_norm"] > 0.0, (
        "The control should be a real nonzero update scaled to the learned update norm."
    )
    assert control["protected_abs_cosine"] > 0.2 or control["accuracy"] <= 0.75, (
        "The control should visibly fail by behavior or protected-direction drift."
    )
    print("All tests in `test_same_norm_random_adapter_control_fails_on_cpu` passed!")


def test_signature_payload_exposes_matched_methods_and_controls(
    build_signature_payload: Callable | None = None,
):
    build_signature_payload = build_signature_payload or _solutions().build_signature_payload
    payload = build_signature_payload(rank=1, alpha=4.0, seed=0, steps=80)
    rows = {row["method"]: row for row in payload["rows"]}

    assert payload["signature_passed"], (
        "The signature payload should pass only when learned methods pass and controls fail."
    )
    assert {"lora", "dora", "full", "random-label LoRA", "same-norm random adapter"} <= rows.keys(), (
        "The payload should contain matched LoRA, DoRA, full-finetune, and both controls."
    )
    assert rows["lora"]["trainable_parameters"] < rows["full"]["trainable_parameters"], (
        "LoRA should use fewer trainable parameters than the matched full finetune."
    )
    assert rows["random-label LoRA"]["control_fails"], (
        "The random-label LoRA control should fail the held-out task."
    )
    assert rows["same-norm random adapter"]["control_fails"], (
        "The same-norm random adapter control should fail the held-out task."
    )
    assert payload["merge_max_abs_diff"] <= 1e-5, (
        "The visible signature result should include merge/unmerge parity."
    )
    assert len(payload["lora_singular_values"]) == 2 and payload["lora_singular_values"][0] > 1.0, (
        "The learned LoRA spectrum should expose the nonzero rank-one update."
    )
    assert len(payload["behavior_examples"]) >= 6, (
        "The learner should see concrete example-level baseline and adapter predictions."
    )
    print("All tests in `test_signature_payload_exposes_matched_methods_and_controls` passed!")


def test_committed_verification_report_trained_peft_controls(
    report: dict[str, Any] | None = None,
):
    report = dict(report or _verification_report())
    gpu = report["metrics"]["gpu_test"]
    controls = set(report["baselines"]["declared_controls"])

    assert report["accepted"] and report["tests_passed"], (
        "The committed report should be accepted and should have passed tests."
    )
    assert report["gt_tier"] == "GT-0", (
        "15.1 should stay scoped to the declared GT-0 proxy-evidence contract."
    )
    assert not report["known_failures"], (
        "The committed verification report should not hide known failures."
    )
    assert gpu["cuda_available"], "The committed report should come from a CUDA run."
    assert gpu["trained_lora_preflight_passed"], (
        "The committed report should include the trained rank-1 LoRA proxy adapter."
    )
    assert gpu["trained_lora_adapter_accuracy"] >= 0.95, (
        "The trained LoRA adapter should clear the target task accuracy gate."
    )
    assert gpu["trained_lora_baseline_accuracy"] <= 0.65, (
        "The frozen baseline should not already solve the target-direction task."
    )
    assert gpu["trained_lora_adapter_rank"] <= 1, (
        "The trained adapter should remain a rank-1 LoRA update."
    )
    assert gpu["trained_lora_random_label_control_fails"], (
        "The random-label training control should fail the task."
    )
    assert gpu["trained_lora_random_adapter_control_fails"], (
        "The same-norm random-adapter control should fail the task."
    )
    assert gpu["trained_lora_dora_norm_preserved"], (
        "DoRA recomposition should preserve norms on the learned adapter delta."
    )
    assert gpu["trained_lora_merge_max_abs_diff"] <= 1e-5, (
        "Merged and unmerged LoRA logits should match to numerical precision."
    )
    assert gpu["trained_lora_target_direction_cosine"] >= 0.95, (
        "The learned LoRA update should align with the planted target direction."
    )
    assert gpu["matched_peft_comparison_passed"], (
        "The report should include the matched LoRA-vs-DoRA-vs-full-finetune comparison."
    )
    assert (
        gpu["matched_lora_trainable_parameters"]
        < gpu["matched_full_finetune_trainable_parameters"]
    ), (
        "The matched LoRA comparison should use fewer trainable parameters than full finetuning."
    )
    assert gpu["matched_dora_norm_preserved"], (
        "The matched DoRA run should preserve row magnitudes."
    )
    assert gpu["matched_target_alignment_floor"] >= 0.95, (
        "All matched adapters should align with the target direction."
    )
    assert gpu["matched_max_distractor_abs_cosine"] <= 0.25, (
        "Matched adapters should suppress the distractor direction."
    )
    assert gpu["peak_vram_gb"] <= 24.0, "The committed run should fit the local VRAM budget."
    required_controls = {
        "trained_rank1_lora_safe_proxy_adapter",
        "random_label_training_control",
        "same_norm_random_adapter_control",
        "dora_on_learned_delta_norm_preservation",
        "matched_rank1_lora_vs_rank1_dora_vs_full_finetune_comparison",
        "matched_target_direction_alignment_floor",
        "matched_distractor_direction_suppression_control",
    }
    assert required_controls <= controls, (
        "The artifact controls should declare the trained PEFT controls and matched baselines."
    )
    print("All tests in `test_committed_verification_report_trained_peft_controls` passed!")


def test_exercise_notebook_declares_full_verification_contract():
    notebook_path = _section_dir() / "15.1_LoRA_DoRA_and_Adapter_Controls_exercises.ipynb"
    notebook = json.loads(notebook_path.read_text())
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
    )

    assert "REQUIRES_GPU = True" in source, (
        "The learner notebook should not advertise CPU-only scope for this GT-0 PEFT section."
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
    assert source.count("### Exercise") >= 6, (
        "15.1 should have at least six learner-facing exercises with immediate tests."
    )
    for required in [
        "def train_proxy_adapter(",
        "def evaluate_proxy_adapter(",
        "def same_norm_random_adapter_control(",
        "def build_signature_payload(",
        "lora_dora_adapter_controls_signature_panel.png",
        "lora_dora_adapter_controls_behavior_table.png",
        'build_signature_payload(rank=1, alpha=4.0, seed=0, steps=80, device="cuda")',
        "Try It Yourself",
        "PLAY_RANK",
        "PLAY_ALPHA",
        "PLAY_SEED",
        "PLAY_METHOD",
        "Anomaly Hunt",
    ]:
        assert required in source, f"The learner notebook is missing `{required}`."
    assert "test_committed_verification_report_trained_peft_controls" in source, (
        "The learner notebook should end by checking the committed trained-PEFT report."
    )
    print("All tests in `test_exercise_notebook_declares_full_verification_contract` passed!")
