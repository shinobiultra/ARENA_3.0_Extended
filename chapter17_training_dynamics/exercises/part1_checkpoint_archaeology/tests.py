from __future__ import annotations

import json
import tempfile
from collections.abc import Callable
from pathlib import Path

import torch as t

from arena_ext import training_dynamics as reference


def _solutions():
    from chapter17_training_dynamics.exercises.part1_checkpoint_archaeology import (
        solutions,
    )

    return solutions


def _section_dir() -> Path:
    return Path(__file__).resolve().parent


def _gpu_report() -> dict:
    report = json.loads((_section_dir() / "verification_report.json").read_text())
    return report["metrics"]["gpu_test"]


def _assert_report_close(actual: object, expected: object, *, msg: str) -> None:
    actual_dict = actual.__dict__
    expected_dict = expected.__dict__
    assert actual_dict.keys() == expected_dict.keys(), (
        f"{msg} should expose fields {sorted(expected_dict)}, got "
        f"{sorted(actual_dict)}. Common bug: returning a plain partial dict makes "
        "later verification code unable to audit the claim scope."
    )
    for key, expected_value in expected_dict.items():
        actual_value = actual_dict[key]
        if isinstance(expected_value, float):
            assert abs(float(actual_value) - expected_value) <= 1e-6, (
                f"{msg} field {key!r} should be {expected_value}, got {actual_value}. "
                "Common bug: converting tensors before flattening or using the wrong "
                "checkpoint index can move the reported metric."
            )
        else:
            assert actual_value == expected_value, (
                f"{msg} field {key!r} should be {expected_value!r}, got "
                f"{actual_value!r}. Common bug: reporting the crossing checkpoint "
                "instead of the stable checkpoint changes the evidence claim."
            )


def test_first_threshold_crossing_finds_first_crossing_and_validates_inputs(
    first_threshold_crossing: Callable | None = None,
):
    first_threshold_crossing = first_threshold_crossing or _solutions().first_threshold_crossing
    steps, trajectories = reference.toy_training_trajectories()
    crossing = first_threshold_crossing(
        steps,
        trajectories["autoregressive"],
        threshold=0.6,
    )
    assert crossing == 300, (
        "The first crossing should be step 300 for the toy AR trajectory. "
        "Common bug: using strict > instead of >=, or returning the tensor index "
        "rather than the checkpoint step."
    )

    try:
        first_threshold_crossing(t.tensor([0, 2, 1]), t.tensor([0.1, 0.2, 0.3]), threshold=0.2)
    except ValueError as exc:
        assert "strictly increasing" in str(exc), (
            "Non-monotone checkpoint steps should raise a clear strictly-increasing "
            "error instead of silently sorting the evidence."
        )
    else:
        raise AssertionError("Non-monotone checkpoint steps should raise ValueError.")
    print("All tests in `test_first_threshold_crossing_finds_first_crossing_and_validates_inputs` passed!")


def test_stable_threshold_step_requires_consecutive_checkpoints(
    stable_threshold_step: Callable | None = None,
):
    stable_threshold_step = stable_threshold_step or _solutions().stable_threshold_step
    steps = t.tensor([0, 10, 20, 30, 40])
    one_spike = t.tensor([0.1, 0.7, 0.2, 0.8, 0.9])
    stable = stable_threshold_step(steps, one_spike, threshold=0.6, min_consecutive=2)
    assert stable == 30, (
        "Stable emergence should start at step 30, not at the isolated spike at step 10. "
        "Common bug: accepting a single threshold crossing as an emergence claim."
    )

    no_stable = stable_threshold_step(steps, one_spike, threshold=0.95, min_consecutive=2)
    assert no_stable is None, (
        "A trajectory with no stable window above threshold should return None. "
        "Common bug: falling back to the peak checkpoint creates a fake emergence claim."
    )
    print("All tests in `test_stable_threshold_step_requires_consecutive_checkpoints` passed!")


def test_mechanism_emergence_report_tracks_peak_and_monotonicity(
    mechanism_emergence_report: Callable | None = None,
):
    mechanism_emergence_report = (
        mechanism_emergence_report or _solutions().mechanism_emergence_report
    )
    steps, trajectories = reference.toy_training_trajectories()
    report = mechanism_emergence_report(
        steps,
        trajectories["autoregressive"],
        metric_name="induction_probe_accuracy",
        threshold=0.6,
        min_consecutive=2,
    )
    expected = reference.mechanism_emergence_report(
        steps,
        trajectories["autoregressive"],
        metric_name="induction_probe_accuracy",
        threshold=0.6,
        min_consecutive=2,
    )
    _assert_report_close(report, expected, msg="Mechanism emergence report")
    assert report.first_crossing_step == 300 and report.stable_from_step == 300, (
        "The report should expose both first crossing and stable crossing. "
        "Common bug: dropping one field hides the difference between a lucky spike "
        "and a persistent mechanism."
    )
    print("All tests in `test_mechanism_emergence_report_tracks_peak_and_monotonicity` passed!")


def test_phase_transition_report_detects_largest_adjacent_jump(
    phase_transition_report: Callable | None = None,
):
    phase_transition_report = phase_transition_report or _solutions().phase_transition_report
    steps, trajectories = reference.toy_training_trajectories()
    report = phase_transition_report(
        steps,
        trajectories["autoregressive"],
        metric_name="induction_probe_accuracy",
        min_jump=0.3,
    )
    expected = reference.phase_transition_report(
        steps,
        trajectories["autoregressive"],
        metric_name="induction_probe_accuracy",
        min_jump=0.3,
    )
    _assert_report_close(report, expected, msg="Phase transition report")
    assert report.transition_step == 300 and report.jump > 0.3, (
        "The largest adjacent jump should land at checkpoint step 300. Common bug: "
        "returning the pre-jump step, or comparing every checkpoint to the initial value."
    )
    print("All tests in `test_phase_transition_report_detects_largest_adjacent_jump` passed!")


def test_random_control_report_rejects_overstrong_control(
    random_control_report: Callable | None = None,
):
    random_control_report = random_control_report or _solutions().random_control_report
    _, trajectories = reference.toy_training_trajectories()
    passing = random_control_report(
        trajectories["random_control"],
        metric_name="label_shuffled_probe",
        max_allowed_value=0.2,
    )
    expected = reference.random_control_report(
        trajectories["random_control"],
        metric_name="label_shuffled_probe",
        max_allowed_value=0.2,
    )
    _assert_report_close(passing, expected, msg="Random-control report")
    assert passing.control_passed, (
        "The toy random-control trajectory should stay below the configured ceiling. "
        "Common bug: testing the final value only instead of the peak value."
    )

    failing = random_control_report(t.tensor([0.1, 0.21, 0.05]), max_allowed_value=0.2)
    assert not failing.control_passed, (
        "A random or label-shuffled control that exceeds the ceiling should fail. "
        "Common bug: allowing a strong control makes the mechanism claim unfalsifiable."
    )
    print("All tests in `test_random_control_report_rejects_overstrong_control` passed!")


def test_developmental_comparison_excludes_random_control_from_ordering(
    developmental_comparison_report: Callable | None = None,
):
    developmental_comparison_report = (
        developmental_comparison_report or _solutions().developmental_comparison_report
    )
    steps, trajectories = reference.toy_training_trajectories()
    report = developmental_comparison_report(
        steps,
        trajectories,
        threshold=0.6,
        min_consecutive=2,
    )
    expected = reference.developmental_comparison_report(
        steps,
        trajectories,
        threshold=0.6,
        min_consecutive=2,
    )
    _assert_report_close(report, expected, msg="Developmental comparison report")
    assert report.earliest_family == "jepa" and report.latest_family == "diffusion", (
        "The ordering should compare only non-control model families. Common bug: "
        "including the random control in earliest/latest timing."
    )
    assert report.emergence_steps["random_control"] is None, (
        "The random-control trajectory should not have a stable emergence step."
    )
    print("All tests in `test_developmental_comparison_excludes_random_control_from_ordering` passed!")


def test_checkpoint_emergence_smoke_test(checkpoint_emergence_smoke_test: Callable | None = None):
    checkpoint_emergence_smoke_test = (
        checkpoint_emergence_smoke_test or _solutions().checkpoint_emergence_smoke_test
    )
    result = checkpoint_emergence_smoke_test()
    assert result["metric_name"] == "induction_probe_accuracy", (
        "The smoke test should name the metric being claimed."
    )
    assert result["first_crossing_step"] == 300 and result["stable_from_step"] == 300, (
        "The smoke test should preserve both crossing fields for learner inspection."
    )
    assert result["emerged"], (
        "The toy AR trajectory should pass the stable-emergence gate."
    )
    print("All tests in `test_checkpoint_emergence_smoke_test` passed!")


def test_phase_transition_smoke_test(phase_transition_smoke_test: Callable | None = None):
    phase_transition_smoke_test = (
        phase_transition_smoke_test or _solutions().phase_transition_smoke_test
    )
    result = phase_transition_smoke_test()
    assert result["transition_step"] == 300, (
        "The phase-transition smoke test should report the post-jump checkpoint step."
    )
    assert result["jump"] > 0.3 and result["phase_transition_detected"], (
        "The toy AR jump should clear the configured phase-transition threshold."
    )
    print("All tests in `test_phase_transition_smoke_test` passed!")


def test_random_control_smoke_test(random_control_smoke_test: Callable | None = None):
    random_control_smoke_test = random_control_smoke_test or _solutions().random_control_smoke_test
    result = random_control_smoke_test()
    assert result["peak_value"] < 0.2 and result["control_passed"], (
        "The smoke control should pass only because its peak value stays below 0.2."
    )
    print("All tests in `test_random_control_smoke_test` passed!")


def test_developmental_comparison_smoke_test(
    developmental_comparison_smoke_test: Callable | None = None,
):
    developmental_comparison_smoke_test = (
        developmental_comparison_smoke_test
        or _solutions().developmental_comparison_smoke_test
    )
    result = developmental_comparison_smoke_test()
    assert result["earliest_family"] == "jepa" and result["latest_family"] == "diffusion", (
        "The smoke comparison should report JEPA first and diffusion last among "
        "non-control toy families."
    )
    assert result["emergence_steps"]["random_control"] is None, (
        "The random control should be present in the report but absent from the "
        "emergent-family ordering."
    )
    assert result["random_control_passed"] and result["all_non_control_emerged"], (
        "The comparison should both reject the control and show that all real toy "
        "families cross stably."
    )
    print("All tests in `test_developmental_comparison_smoke_test` passed!")


def test_live_checkpoint_archaeology_smoke_test_trains_saves_reloads_and_controls(
    live_checkpoint_archaeology_smoke_test: Callable | None = None,
):
    live_checkpoint_archaeology_smoke_test = (
        live_checkpoint_archaeology_smoke_test
        or _solutions().live_checkpoint_archaeology_smoke_test
    )
    with tempfile.TemporaryDirectory(prefix="arena17_live_test_") as tmp:
        checkpoint_root = Path(tmp)
        result = live_checkpoint_archaeology_smoke_test(
            checkpoint_root=checkpoint_root,
            device="cpu",
        )
        checkpoint_files = sorted(checkpoint_root.glob("*/*.pt"))

        assert result["preflight_passed"], (
            "The live smoke path should train, save, reload, analyze, and control "
            "a real tiny checkpoint run."
        )
        assert len(checkpoint_files) == result["checkpoint_count"] == 26, (
            "The live smoke path should write target and random-control checkpoints "
            "for every declared checkpoint step."
        )
        assert all(path.stat().st_size > 0 for path in checkpoint_files), (
            "Checkpoint files should be real torch checkpoint files, not empty sentinels."
        )
    assert result["device"] == "cpu", (
        "The live section test should be a CPU-feasible path; CUDA acceptance is "
        "reserved for run_gpu_test."
    )
    assert result["real_checkpoints_reloaded"], (
        "The live smoke metrics must be recomputed after loading saved checkpoint files."
    )
    assert result["complete_finite_domain_evaluated"] and not result["ood_generalization_claimed"], (
        "The mod-13 model organism should claim exhaustive finite-domain evaluation, "
        "not OOD generalization."
    )
    assert result["table_example_count"] == 169 and result["final_accuracy"] >= 0.99, (
        "The target run should evaluate all 13x13 examples and solve the finite table."
    )
    assert result["stable_from_step"] == 30 and result["phase_transition_detected"], (
        "The live target run should preserve the emergence and phase-transition evidence."
    )
    assert result["random_control_passed"] and result["random_control_peak_accuracy"] <= 0.2, (
        "The random-label checkpoint control should stay near chance on the true table."
    )
    print(
        "All tests in "
        "`test_live_checkpoint_archaeology_smoke_test_trains_saves_reloads_and_controls` "
        "passed!"
    )


def test_committed_gpu_report_records_real_checkpoint_preflight():
    gpu = _gpu_report()
    assert gpu["preflight_passed"], (
        "The committed 17.1 report should have accepted the real CUDA checkpoint preflight."
    )
    assert gpu["cuda_available"] and "RTX 5090" in gpu["device"], (
        "The report should record a real CUDA device rather than placeholder evidence."
    )
    assert gpu["real_checkpoints_reloaded"], (
        "Checkpoint archaeology evidence must come from saved and reloaded checkpoint files."
    )
    assert gpu["complete_finite_domain_evaluated"] and not gpu["ood_generalization_claimed"], (
        "The report should state that mod-13 evidence is exhaustive finite-domain "
        "coverage, not OOD generalization."
    )
    assert gpu["checkpoint_count"] == 26, (
        "The report should include target and random-control checkpoints for every "
        "declared checkpoint step."
    )
    assert gpu["final_accuracy"] == 1.0 and gpu["stable_from_step"] == 30, (
        "The true-label modular-addition run should reach perfect table accuracy "
        "and stable emergence from step 30."
    )
    assert gpu["random_control_passed"] and gpu["random_control_peak_accuracy"] <= 0.2, (
        "The random-label checkpoint control should stay near chance against the true table."
    )
    assert gpu["within_vram_budget"] and gpu["peak_vram_gb"] <= 24.0, (
        "The GPU evidence should stay inside the declared local VRAM budget."
    )
    print("All tests in `test_committed_gpu_report_records_real_checkpoint_preflight` passed!")


def test_notebook_contract(run_smoke_test: Callable | None = None):
    run_smoke_test = run_smoke_test or _solutions().run_smoke_test
    result = run_smoke_test(cpu=True)
    assert result["checkpoint_emergence"]["emerged"], (
        "The notebook contract should include a passing stable-emergence check."
    )
    assert result["phase_transition"]["phase_transition_detected"], (
        "The notebook contract should include a passing adjacent-jump check."
    )
    assert result["random_control"]["control_passed"], (
        "The notebook contract should include a passing random-control rejection gate."
    )
    assert result["developmental_comparison"]["random_control_passed"], (
        "The notebook contract should keep the random-control status visible."
    )
    live = result["live_checkpoint_archaeology"]
    assert live["preflight_passed"] and live["real_checkpoints_reloaded"], (
        "The notebook contract should include a live train/save/reload checkpoint path."
    )
    assert live["complete_finite_domain_evaluated"] and not live["ood_generalization_claimed"], (
        "The notebook contract should make finite-domain scope explicit for mod-13."
    )
    print("All tests in `test_notebook_contract` passed!")
