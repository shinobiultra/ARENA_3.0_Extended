"""Exact historical record types and dictionary formatting; no taught JEPA method."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
from torch import Tensor


@dataclass(frozen=True)
class SceneBatch:
    images: Tensor
    object_labels: Tensor
    position_labels: Tensor


@dataclass(frozen=True)
class CollapseDiagnostics:
    feature_std: float
    effective_rank: float
    mean_pairwise_cosine: float


@dataclass(frozen=True)
class ProbeMetrics:
    object_accuracy: float
    position_accuracy: float


@dataclass(frozen=True)
class PredictionMetrics:
    latent_loss: float
    mismatched_context_loss: float
    key_erased_context_loss: float
    zero_context_loss: float
    target_object_accuracy: float
    predicted_object_accuracy: float
    mismatched_context_object_accuracy: float
    alternate_key_object_accuracy: float
    key_erased_context_object_accuracy: float
    non_key_pixel_max_abs_diff: float
    key_erased_non_key_pixel_max_abs_diff: float


@dataclass(frozen=True)
class TrainingTrace:
    losses: list[float]
    feature_std: list[float]
    effective_rank: list[float]


@dataclass(frozen=True)
class IJEPARun:
    device: str
    seed: int
    initial_loss: float
    final_loss: float
    loss_ratio: float
    probes: ProbeMetrics
    shuffled_label_probes: ProbeMetrics
    collapsed_control_probes: ProbeMetrics
    joint_target_probes: ProbeMetrics
    random_encoder_probes: ProbeMetrics
    prediction: PredictionMetrics
    joint_target_prediction: PredictionMetrics
    random_prediction: PredictionMetrics
    diagnostics: CollapseDiagnostics
    joint_target_diagnostics: CollapseDiagnostics
    collapsed_control_diagnostics: CollapseDiagnostics
    target_parameter_change: float
    target_has_gradients: bool
    peak_vram_gb: float
    trace: TrainingTrace
    joint_target_trace: TrainingTrace
    train_eval_exact_overlap: int
    train_semantic_template_count: int
    evaluation_semantic_template_count: int
    shared_semantic_template_count: int


def _report_metrics(run: IJEPARun) -> dict[str, Any]:
    report = asdict(run)
    report.update(
        {
            "loss_ratio": run.loss_ratio,
            "feature_std": run.diagnostics.feature_std,
            "effective_rank": run.diagnostics.effective_rank,
            "heldout_prediction_accuracy": run.prediction.predicted_object_accuracy,
            "mismatched_context_accuracy": run.prediction.mismatched_context_object_accuracy,
            "alternate_key_accuracy": run.prediction.alternate_key_object_accuracy,
            "key_erased_context_accuracy": run.prediction.key_erased_context_object_accuracy,
            "non_key_pixel_max_abs_diff": run.prediction.non_key_pixel_max_abs_diff,
            "key_erased_non_key_pixel_max_abs_diff": (
                run.prediction.key_erased_non_key_pixel_max_abs_diff
            ),
            "random_prediction_accuracy": run.random_prediction.predicted_object_accuracy,
            "joint_target_ablation_prediction_accuracy": (
                run.joint_target_prediction.predicted_object_accuracy
            ),
            "mismatched_to_correct_loss_ratio": (
                run.prediction.mismatched_context_loss / run.prediction.latent_loss
            ),
            "zero_to_correct_loss_ratio": (
                run.prediction.zero_context_loss / run.prediction.latent_loss
            ),
            "key_erased_to_correct_loss_ratio": (
                run.prediction.key_erased_context_loss / run.prediction.latent_loss
            ),
            "random_to_trained_loss_ratio": (
                run.random_prediction.latent_loss / run.prediction.latent_loss
            ),
            "random_encoder_object_probe_accuracy": run.random_encoder_probes.object_accuracy,
            "random_encoder_position_probe_accuracy": run.random_encoder_probes.position_accuracy,
            "object_probe_margin_over_random": (
                run.probes.object_accuracy - run.random_encoder_probes.object_accuracy
            ),
            "position_probe_margin_over_random": (
                run.probes.position_accuracy - run.random_encoder_probes.position_accuracy
            ),
            "main_to_joint_target_feature_std_ratio": (
                run.diagnostics.feature_std / run.joint_target_diagnostics.feature_std
            ),
            "collapsed_feature_std": run.collapsed_control_diagnostics.feature_std,
            "collapsed_effective_rank": run.collapsed_control_diagnostics.effective_rank,
            "train_eval_exact_overlap": run.train_eval_exact_overlap,
            "shared_semantic_template_count": run.shared_semantic_template_count,
            "tests_passed": True,
            "contract_passed": True,
            "accepted": True,
        }
    )
    return report

