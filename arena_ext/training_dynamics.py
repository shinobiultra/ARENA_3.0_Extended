"""Training-dynamics and developmental-interpretability utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch as t


@dataclass(frozen=True)
class MechanismEmergenceReport:
    metric_name: str
    threshold: float
    first_crossing_step: int | None
    stable_from_step: int | None
    peak_step: int
    peak_value: float
    monotonicity_violations: int
    emerged: bool


@dataclass(frozen=True)
class RandomControlReport:
    metric_name: str
    peak_value: float
    max_allowed_value: float
    control_passed: bool


@dataclass(frozen=True)
class PhaseTransitionReport:
    metric_name: str
    transition_step: int
    pre_value: float
    post_value: float
    jump: float
    phase_transition_detected: bool


@dataclass(frozen=True)
class DevelopmentalComparisonReport:
    threshold: float
    emergence_steps: dict[str, int | None]
    earliest_family: str | None
    latest_family: str | None
    random_control_passed: bool
    all_non_control_emerged: bool


def _validate_checkpoint_series(
    steps: t.Tensor,
    values: t.Tensor,
) -> tuple[t.Tensor, t.Tensor]:
    steps = t.as_tensor(steps).flatten().long()
    values = t.as_tensor(values).flatten().float()
    if steps.numel() == 0:
        raise ValueError("checkpoint series must be nonempty.")
    if steps.numel() != values.numel():
        raise ValueError("steps and values must have the same length.")
    if steps.numel() > 1 and not bool((steps[1:] > steps[:-1]).all().item()):
        raise ValueError("checkpoint steps must be strictly increasing.")
    if not bool(t.isfinite(values).all().item()):
        raise ValueError("checkpoint values must be finite.")
    return steps, values


def first_threshold_crossing(
    steps: t.Tensor,
    values: t.Tensor,
    *,
    threshold: float,
) -> int | None:
    """Return the first checkpoint step whose metric crosses the threshold."""

    steps, values = _validate_checkpoint_series(steps, values)
    crossed = values >= threshold
    if not bool(crossed.any().item()):
        return None
    index = int(t.nonzero(crossed, as_tuple=False)[0].item())
    return int(steps[index].item())


def stable_threshold_step(
    steps: t.Tensor,
    values: t.Tensor,
    *,
    threshold: float,
    min_consecutive: int = 2,
) -> int | None:
    """Return the first step with a stable run above the threshold."""

    if min_consecutive <= 0:
        raise ValueError("min_consecutive must be positive.")
    steps, values = _validate_checkpoint_series(steps, values)
    if values.numel() < min_consecutive:
        return None

    for index in range(values.numel() - min_consecutive + 1):
        window = values[index : index + min_consecutive]
        if bool((window >= threshold).all().item()):
            return int(steps[index].item())
    return None


def monotonicity_violations(values: t.Tensor, *, tolerance: float = 0.0) -> int:
    """Count adjacent decreases larger than tolerance in a checkpoint metric."""

    values = t.as_tensor(values).flatten().float()
    if values.numel() == 0:
        raise ValueError("values must be nonempty.")
    if not bool(t.isfinite(values).all().item()):
        raise ValueError("values must be finite.")
    if values.numel() == 1:
        return 0
    decreases = values[:-1] - values[1:]
    return int((decreases > tolerance).sum().item())


def mechanism_emergence_report(
    steps: t.Tensor,
    values: t.Tensor,
    *,
    metric_name: str = "mechanism_metric",
    threshold: float = 0.6,
    min_consecutive: int = 2,
    max_monotonicity_violations: int = 1,
) -> MechanismEmergenceReport:
    """Summarize when a probe, circuit, or feature metric emerges."""

    steps, values = _validate_checkpoint_series(steps, values)
    first_crossing = first_threshold_crossing(steps, values, threshold=threshold)
    stable_from = stable_threshold_step(
        steps,
        values,
        threshold=threshold,
        min_consecutive=min_consecutive,
    )
    peak_index = int(t.argmax(values).item())
    violations = monotonicity_violations(values)
    return MechanismEmergenceReport(
        metric_name=metric_name,
        threshold=threshold,
        first_crossing_step=first_crossing,
        stable_from_step=stable_from,
        peak_step=int(steps[peak_index].item()),
        peak_value=float(values[peak_index].item()),
        monotonicity_violations=violations,
        emerged=stable_from is not None and violations <= max_monotonicity_violations,
    )


def random_control_report(
    values: t.Tensor,
    *,
    metric_name: str = "random_control",
    max_allowed_value: float = 0.3,
) -> RandomControlReport:
    """Check that a random or label-shuffled control does not look emergent."""

    values = t.as_tensor(values).flatten().float()
    if values.numel() == 0:
        raise ValueError("control values must be nonempty.")
    if not bool(t.isfinite(values).all().item()):
        raise ValueError("control values must be finite.")
    peak_value = float(values.max().item())
    return RandomControlReport(
        metric_name=metric_name,
        peak_value=peak_value,
        max_allowed_value=max_allowed_value,
        control_passed=peak_value <= max_allowed_value,
    )


def phase_transition_report(
    steps: t.Tensor,
    values: t.Tensor,
    *,
    metric_name: str = "mechanism_metric",
    min_jump: float = 0.25,
) -> PhaseTransitionReport:
    """Find the largest adjacent checkpoint jump and test a transition bound."""

    steps, values = _validate_checkpoint_series(steps, values)
    if values.numel() < 2:
        raise ValueError("at least two checkpoints are required.")
    jumps = values[1:] - values[:-1]
    transition_index = int(t.argmax(jumps).item())
    jump = float(jumps[transition_index].item())
    return PhaseTransitionReport(
        metric_name=metric_name,
        transition_step=int(steps[transition_index + 1].item()),
        pre_value=float(values[transition_index].item()),
        post_value=float(values[transition_index + 1].item()),
        jump=jump,
        phase_transition_detected=jump >= min_jump,
    )


def developmental_comparison_report(
    steps: t.Tensor,
    family_values: Mapping[str, t.Tensor],
    *,
    threshold: float = 0.6,
    min_consecutive: int = 2,
    control_name: str = "random_control",
) -> DevelopmentalComparisonReport:
    """Compare emergence timing across model-family checkpoint trajectories."""

    if not family_values:
        raise ValueError("family_values must contain at least one trajectory.")
    emergence_steps: dict[str, int | None] = {}
    for family, values in family_values.items():
        emergence_steps[family] = stable_threshold_step(
            steps,
            values,
            threshold=threshold,
            min_consecutive=min_consecutive,
        )

    non_control_steps = {
        family: step
        for family, step in emergence_steps.items()
        if family != control_name and step is not None
    }
    if non_control_steps:
        earliest_family = min(non_control_steps, key=lambda item: (non_control_steps[item], item))
        latest_family = max(non_control_steps, key=lambda item: (non_control_steps[item], item))
    else:
        earliest_family = None
        latest_family = None

    non_control_count = sum(family != control_name for family in family_values)
    random_control_passed = True
    if control_name in family_values:
        random_control_passed = random_control_report(
            family_values[control_name],
            max_allowed_value=threshold,
        ).control_passed

    return DevelopmentalComparisonReport(
        threshold=threshold,
        emergence_steps=emergence_steps,
        earliest_family=earliest_family,
        latest_family=latest_family,
        random_control_passed=random_control_passed,
        all_non_control_emerged=len(non_control_steps) == non_control_count,
    )


def toy_training_trajectories(
    *,
    device: t.device | str | None = None,
) -> tuple[t.Tensor, dict[str, t.Tensor]]:
    """Return deterministic toy checkpoint trajectories for course exercises."""

    kwargs = {"device": device} if device is not None else {}
    steps = t.tensor([0, 100, 200, 300, 400, 500], dtype=t.long, **kwargs)
    trajectories = {
        "autoregressive": t.tensor(
            [0.05, 0.12, 0.30, 0.68, 0.81, 0.86],
            dtype=t.float32,
            **kwargs,
        ),
        "jepa": t.tensor(
            [0.10, 0.35, 0.62, 0.74, 0.79, 0.82],
            dtype=t.float32,
            **kwargs,
        ),
        "diffusion": t.tensor(
            [0.04, 0.08, 0.18, 0.34, 0.61, 0.72],
            dtype=t.float32,
            **kwargs,
        ),
        "mamba": t.tensor(
            [0.03, 0.22, 0.58, 0.66, 0.73, 0.78],
            dtype=t.float32,
            **kwargs,
        ),
        "random_control": t.tensor(
            [0.02, 0.10, 0.12, 0.08, 0.11, 0.10],
            dtype=t.float32,
            **kwargs,
        ),
    }
    return steps, trajectories
