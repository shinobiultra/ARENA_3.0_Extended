import importlib.util

import pytest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is not installed")

if TORCH_AVAILABLE:
    import torch as t

    from arena_ext.training_dynamics import (
        developmental_comparison_report,
        first_threshold_crossing,
        mechanism_emergence_report,
        phase_transition_report,
        random_control_report,
        stable_threshold_step,
        toy_training_trajectories,
    )


def test_threshold_crossing_and_stability_are_explicit():
    steps = t.tensor([0, 100, 200, 300, 400])
    values = t.tensor([0.1, 0.3, 0.65, 0.55, 0.7])

    assert first_threshold_crossing(steps, values, threshold=0.6) == 200
    assert stable_threshold_step(steps, values, threshold=0.6, min_consecutive=2) is None


def test_mechanism_emergence_report_requires_stable_crossing():
    steps, trajectories = toy_training_trajectories()

    report = mechanism_emergence_report(
        steps,
        trajectories["autoregressive"],
        metric_name="probe_accuracy",
        threshold=0.6,
        min_consecutive=2,
    )

    assert report.first_crossing_step == 300
    assert report.stable_from_step == 300
    assert report.peak_step == 500
    assert report.emerged


def test_random_control_report_rejects_spurious_emergence():
    _, trajectories = toy_training_trajectories()

    report = random_control_report(
        trajectories["random_control"],
        max_allowed_value=0.2,
    )

    assert report.peak_value == pytest.approx(0.12)
    assert report.control_passed


def test_phase_transition_report_finds_largest_checkpoint_jump():
    steps, trajectories = toy_training_trajectories()

    report = phase_transition_report(
        steps,
        trajectories["autoregressive"],
        metric_name="induction_proxy",
        min_jump=0.3,
    )

    assert report.transition_step == 300
    assert report.jump == pytest.approx(0.38)
    assert report.phase_transition_detected


def test_developmental_comparison_orders_model_families_and_control():
    steps, trajectories = toy_training_trajectories()

    report = developmental_comparison_report(
        steps,
        trajectories,
        threshold=0.6,
        min_consecutive=2,
    )

    assert report.emergence_steps == {
        "autoregressive": 300,
        "jepa": 200,
        "diffusion": 400,
        "mamba": 300,
        "random_control": None,
    }
    assert report.earliest_family == "jepa"
    assert report.latest_family == "diffusion"
    assert report.random_control_passed
    assert report.all_non_control_emerged


def test_checkpoint_series_rejects_nonincreasing_steps():
    steps = t.tensor([0, 100, 100])
    values = t.tensor([0.1, 0.2, 0.3])

    with pytest.raises(ValueError, match="strictly increasing"):
        mechanism_emergence_report(steps, values)
