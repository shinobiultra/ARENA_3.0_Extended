"""Diagnostics for common fake interpretability results."""

from __future__ import annotations

from dataclasses import dataclass

import torch as t


@dataclass(frozen=True)
class LabelLeakageReport:
    leaked_feature_index: int
    leaked_feature_accuracy: float
    shifted_no_leak_accuracy: float
    accuracy_gap: float
    detects_leakage: bool


@dataclass(frozen=True)
class CherryPickReport:
    selected_mean_effect: float
    population_mean_effect: float
    population_median_effect: float
    inflation_ratio: float
    detects_cherry_picking: bool


@dataclass(frozen=True)
class ProbeOverfitReport:
    train_accuracy: float
    heldout_accuracy: float
    generalization_gap: float
    detects_overfit: bool


@dataclass(frozen=True)
class FakeRandomDirectionControlReport:
    claimed_effect: float
    random_p95_effect: float
    effect_gap: float
    passes_random_control: bool
    detects_random_direction_failure: bool


@dataclass(frozen=True)
class FakeResultAuditReport:
    leakage_detected: bool
    cherry_pick_detected: bool
    probe_overfit_detected: bool
    random_direction_failure_detected: bool
    all_bogus_results_flagged: bool


def binary_accuracy(scores: t.Tensor, labels: t.Tensor) -> float:
    """Return threshold-at-zero binary accuracy for signed labels."""

    labels = labels.long().flatten()
    predictions = (scores.flatten() > 0).long()
    return float((predictions == labels).double().mean().item())


def label_leakage_report() -> LabelLeakageReport:
    """Detect a feature that directly encodes the label."""

    labels = t.tensor([0, 1, 0, 1, 0, 1], dtype=t.long)
    signed = labels.double() * 2 - 1
    leaked_feature = signed
    shifted_spurious_feature = -signed
    leaked_accuracy = binary_accuracy(leaked_feature, labels)
    shifted_accuracy = binary_accuracy(shifted_spurious_feature, labels)
    gap = leaked_accuracy - shifted_accuracy
    return LabelLeakageReport(
        leaked_feature_index=0,
        leaked_feature_accuracy=leaked_accuracy,
        shifted_no_leak_accuracy=shifted_accuracy,
        accuracy_gap=gap,
        detects_leakage=leaked_accuracy == 1.0 and gap >= 0.5,
    )


def cherry_pick_report() -> CherryPickReport:
    """Detect when selected examples exaggerate the population effect."""

    effects = t.tensor(
        [
            0.02,
            0.03,
            0.04,
            0.05,
            0.06,
            0.07,
            0.08,
            0.09,
            0.10,
            0.11,
            0.12,
            0.13,
            1.40,
            1.55,
            1.70,
        ]
    )
    selected = effects[-3:]
    selected_mean = float(selected.mean().item())
    population_mean = float(effects.mean().item())
    population_median = float(effects.median().item())
    inflation = selected_mean / population_mean
    return CherryPickReport(
        selected_mean_effect=selected_mean,
        population_mean_effect=population_mean,
        population_median_effect=population_median,
        inflation_ratio=inflation,
        detects_cherry_picking=inflation >= 3.0 and selected_mean >= 5 * population_median,
    )


def probe_overfit_report() -> ProbeOverfitReport:
    """Detect a memorizing probe that does not generalize."""

    train_labels = t.tensor([0, 1, 0, 1, 0, 1], dtype=t.long)
    heldout_labels = t.tensor([0, 1, 0, 1, 0, 1], dtype=t.long)
    train_predictions = train_labels.clone()
    heldout_predictions = t.tensor([0, 0, 0, 0, 0, 0], dtype=t.long)
    train_accuracy = float((train_predictions == train_labels).double().mean().item())
    heldout_accuracy = float((heldout_predictions == heldout_labels).double().mean().item())
    gap = train_accuracy - heldout_accuracy
    return ProbeOverfitReport(
        train_accuracy=train_accuracy,
        heldout_accuracy=heldout_accuracy,
        generalization_gap=gap,
        detects_overfit=train_accuracy >= 0.95 and heldout_accuracy <= 0.6 and gap >= 0.35,
    )


def random_direction_control_report() -> FakeRandomDirectionControlReport:
    """Detect a claimed steering direction that is no stronger than random controls."""

    behavior_direction = t.tensor([1.0, 0.0, 0.0], dtype=t.float64)
    claimed_direction = t.tensor([0.05, 0.9987, 0.0], dtype=t.float64)
    random_directions = t.tensor(
        [
            [0.12, 0.99, 0.00],
            [-0.08, 0.99, 0.05],
            [0.04, 0.20, 0.98],
            [0.10, -0.95, 0.10],
            [-0.11, 0.00, 0.99],
        ],
        dtype=t.float64,
    )
    claimed_direction = claimed_direction / claimed_direction.norm()
    random_directions = random_directions / random_directions.norm(dim=1, keepdim=True)
    claimed_effect = float(abs(claimed_direction @ behavior_direction).item())
    random_effects = (random_directions @ behavior_direction).abs()
    random_p95 = float(t.quantile(random_effects, 0.95).item())
    gap = claimed_effect - random_p95
    passes = gap >= 0.25
    return FakeRandomDirectionControlReport(
        claimed_effect=claimed_effect,
        random_p95_effect=random_p95,
        effect_gap=gap,
        passes_random_control=passes,
        detects_random_direction_failure=not passes,
    )


def fake_result_audit_report() -> FakeResultAuditReport:
    """Run a compact audit suite for fake-result failure modes."""

    leakage = label_leakage_report()
    cherry_pick = cherry_pick_report()
    overfit = probe_overfit_report()
    random_direction = random_direction_control_report()
    flags = [
        leakage.detects_leakage,
        cherry_pick.detects_cherry_picking,
        overfit.detects_overfit,
        random_direction.detects_random_direction_failure,
    ]
    return FakeResultAuditReport(
        leakage_detected=flags[0],
        cherry_pick_detected=flags[1],
        probe_overfit_detected=flags[2],
        random_direction_failure_detected=flags[3],
        all_bogus_results_flagged=all(flags),
    )
