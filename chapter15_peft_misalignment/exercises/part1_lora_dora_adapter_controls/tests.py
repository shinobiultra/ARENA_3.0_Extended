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
    assert "test_committed_verification_report_trained_peft_controls" in source, (
        "The learner notebook should end by checking the committed trained-PEFT report."
    )
    print("All tests in `test_exercise_notebook_declares_full_verification_contract` passed!")
