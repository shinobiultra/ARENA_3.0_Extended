"""Image-generation interpretability utilities for diffusion and AR-image labs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch as t


ImageDirection = Literal["increase", "decrease"]


@dataclass(frozen=True)
class AttentionRegionReport:
    region_mass: float
    off_region_mass: float
    region_selective: bool


@dataclass(frozen=True)
class DenoisingCircuitReport:
    baseline_loss: float
    ablated_loss: float
    random_control_loss: float
    ablation_delta: float
    random_delta: float
    circuit_specific: bool


@dataclass(frozen=True)
class LatentDirectionReport:
    baseline_mean: float
    steered_mean: float
    random_control_mean: float
    observed_delta: float
    random_delta: float
    has_directional_effect: bool


@dataclass(frozen=True)
class PromptRegionCausalReport:
    original_region_score: float
    ablated_region_score: float
    control_region_score: float
    target_drop: float
    control_drop: float
    prompt_region_causal: bool


@dataclass(frozen=True)
class DAAMRegionReport:
    target_region_mass: float
    control_region_mass: float
    mask_fraction: float
    captured_map_count: int
    target_control_gap: float
    target_lift_over_mask_fraction: float
    daam_localized: bool


@dataclass(frozen=True)
class TokenAblationReport:
    original_region_score: float
    target_ablated_region_score: float
    random_control_region_score: float
    target_drop: float
    random_control_drop: float
    target_ablation_passed: bool
    random_token_ablation_weaker: bool


@dataclass(frozen=True)
class ImageQualityReport:
    target_region_fraction: float
    rgb_std: float
    high_frequency_energy: float
    saturation_fraction: float
    image_quality_preserved: bool


@dataclass(frozen=True)
class WhiteNoiseImageReport:
    real_high_frequency_energy: float
    white_noise_high_frequency_energy: float
    white_noise_rejected: bool


@dataclass(frozen=True)
class SD15StrictReport:
    daam_passed: bool
    token_ablation_passed: bool
    random_token_ablation_weaker: bool
    image_quality_preserved: bool
    white_noise_rejected: bool
    sd15_strict_experiment_passed: bool


def _as_hwc_rgb(image: t.Tensor) -> t.Tensor:
    if image.ndim != 3:
        raise ValueError("image must have shape (height, width, 3) or (3, height, width).")
    if image.shape[-1] == 3:
        rgb = image.float()
    elif image.shape[0] == 3:
        rgb = image.permute(1, 2, 0).float()
    else:
        raise ValueError("image must have three RGB channels.")
    if not t.isfinite(rgb).all():
        raise ValueError("image must contain only finite values.")
    if rgb.max().item() <= 1.0:
        rgb = rgb * 255.0
    return rgb.clamp(0, 255)


def color_region_mask(image: t.Tensor, target_color: Literal["red", "blue"]) -> t.Tensor:
    """Return a strict chroma mask for red or blue generated objects."""

    rgb = _as_hwc_rgb(image)
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    if target_color == "red":
        return (red > 140) & (green < 120) & (blue < 120) & (red > green * 1.3) & (
            red > blue * 1.3
        )
    if target_color == "blue":
        return (blue > 130) & (red < 140) & (green < 180) & (blue > red * 1.25) & (
            blue > green * 1.05
        )
    raise ValueError("target_color must be 'red' or 'blue'.")


def attention_region_report(
    attention_map: t.Tensor,
    region_mask: t.Tensor,
    *,
    min_region_mass: float = 0.6,
) -> AttentionRegionReport:
    """Check whether an attention map concentrates mass in a target image region."""

    if attention_map.ndim != 2:
        raise ValueError("attention_map must have shape (height, width).")
    if region_mask.shape != attention_map.shape:
        raise ValueError("region_mask must match attention_map shape.")
    mask = region_mask.bool()
    if not mask.any():
        raise ValueError("region_mask must select at least one position.")

    attention = attention_map.float().clamp_min(0)
    total_mass = attention.sum()
    if total_mass.item() == 0:
        raise ValueError("attention_map must have positive mass.")
    region_mass = (attention[mask].sum() / total_mass).item()
    off_region_mass = 1.0 - region_mass
    return AttentionRegionReport(
        region_mass=region_mass,
        off_region_mass=off_region_mass,
        region_selective=region_mass >= min_region_mass,
    )


def denoising_circuit_report(
    *,
    baseline_loss: float,
    ablated_loss: float,
    random_control_loss: float,
    min_loss_increase: float = 0.1,
    min_control_gap: float = 0.05,
) -> DenoisingCircuitReport:
    """Check that ablating a proposed denoising circuit matters over random control."""

    ablation_delta = ablated_loss - baseline_loss
    random_delta = random_control_loss - baseline_loss
    circuit_specific = (
        ablation_delta >= min_loss_increase
        and ablation_delta >= random_delta + min_control_gap
    )
    return DenoisingCircuitReport(
        baseline_loss=baseline_loss,
        ablated_loss=ablated_loss,
        random_control_loss=random_control_loss,
        ablation_delta=ablation_delta,
        random_delta=random_delta,
        circuit_specific=circuit_specific,
    )


def latent_direction_effect_report(
    baseline_scores: t.Tensor,
    steered_scores: t.Tensor,
    random_control_scores: t.Tensor,
    *,
    expected_direction: ImageDirection = "increase",
    min_effect: float = 0.2,
    min_random_margin: float = 0.1,
) -> LatentDirectionReport:
    """Check whether a latent image direction changes a score over random control."""

    if baseline_scores.shape != steered_scores.shape:
        raise ValueError("baseline and steered scores must match.")
    if baseline_scores.shape != random_control_scores.shape:
        raise ValueError("baseline and random control scores must match.")

    baseline_mean = baseline_scores.float().mean().item()
    steered_mean = steered_scores.float().mean().item()
    random_control_mean = random_control_scores.float().mean().item()
    observed_delta = steered_mean - baseline_mean
    random_delta = random_control_mean - baseline_mean
    if expected_direction == "increase":
        directional_effect = observed_delta >= min_effect
    elif expected_direction == "decrease":
        directional_effect = -observed_delta >= min_effect
    else:
        raise ValueError("expected_direction must be 'increase' or 'decrease'.")
    has_directional_effect = directional_effect and abs(observed_delta) > (
        abs(random_delta) + min_random_margin
    )
    return LatentDirectionReport(
        baseline_mean=baseline_mean,
        steered_mean=steered_mean,
        random_control_mean=random_control_mean,
        observed_delta=observed_delta,
        random_delta=random_delta,
        has_directional_effect=has_directional_effect,
    )


def prompt_region_causal_report(
    *,
    original_region_score: float,
    ablated_region_score: float,
    control_region_score: float,
    min_target_drop: float = 0.2,
    min_control_margin: float = 0.1,
) -> PromptRegionCausalReport:
    """Check whether ablating a prompt token causally changes its target region."""

    target_drop = original_region_score - ablated_region_score
    control_drop = original_region_score - control_region_score
    prompt_region_causal = (
        target_drop >= min_target_drop
        and target_drop >= control_drop + min_control_margin
    )
    return PromptRegionCausalReport(
        original_region_score=original_region_score,
        ablated_region_score=ablated_region_score,
        control_region_score=control_region_score,
        target_drop=target_drop,
        control_drop=control_drop,
        prompt_region_causal=prompt_region_causal,
    )


def daam_region_report(
    *,
    target_region_mass: float,
    control_region_mass: float,
    mask_fraction: float,
    captured_map_count: int,
    min_target_control_gap: float = 0.005,
    min_lift_over_mask_fraction: float = 0.01,
    min_captured_map_count: int = 16,
) -> DAAMRegionReport:
    """Check DAAM-style target-token attention over a generated image region."""

    if not 0.0 <= target_region_mass <= 1.0:
        raise ValueError("target_region_mass must be in [0, 1].")
    if not 0.0 <= control_region_mass <= 1.0:
        raise ValueError("control_region_mass must be in [0, 1].")
    if not 0.0 < mask_fraction < 1.0:
        raise ValueError("mask_fraction must be in (0, 1).")
    if captured_map_count <= 0:
        raise ValueError("captured_map_count must be positive.")

    target_control_gap = target_region_mass - control_region_mass
    target_lift = target_region_mass - mask_fraction
    localized = (
        target_control_gap >= min_target_control_gap
        and target_lift >= min_lift_over_mask_fraction
        and captured_map_count >= min_captured_map_count
    )
    return DAAMRegionReport(
        target_region_mass=target_region_mass,
        control_region_mass=control_region_mass,
        mask_fraction=mask_fraction,
        captured_map_count=captured_map_count,
        target_control_gap=target_control_gap,
        target_lift_over_mask_fraction=target_lift,
        daam_localized=localized,
    )


def token_ablation_region_report(
    *,
    original_region_score: float,
    target_ablated_region_score: float,
    random_control_region_score: float,
    min_target_drop: float = 0.05,
    min_random_margin: float = 0.05,
) -> TokenAblationReport:
    """Check whether target-token ablation drops the target region over controls."""

    for name, value in {
        "original_region_score": original_region_score,
        "target_ablated_region_score": target_ablated_region_score,
        "random_control_region_score": random_control_region_score,
    }.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1].")

    target_drop = original_region_score - target_ablated_region_score
    random_drop = original_region_score - random_control_region_score
    target_passed = target_drop >= min_target_drop
    random_weaker = target_drop >= random_drop + min_random_margin
    return TokenAblationReport(
        original_region_score=original_region_score,
        target_ablated_region_score=target_ablated_region_score,
        random_control_region_score=random_control_region_score,
        target_drop=target_drop,
        random_control_drop=random_drop,
        target_ablation_passed=target_passed,
        random_token_ablation_weaker=random_weaker,
    )


def image_quality_report(
    image: t.Tensor,
    *,
    target_color: Literal["red", "blue"],
    min_target_region_fraction: float = 0.02,
    min_rgb_std: float = 0.05,
    max_high_frequency_energy: float = 0.12,
) -> ImageQualityReport:
    """Reject blank/collapsed/high-noise generated images using simple statistics."""

    rgb = _as_hwc_rgb(image)
    mask = color_region_mask(rgb, target_color)
    target_region_fraction = mask.float().mean().item()
    rgb_std = (rgb.std() / 255.0).item()
    high_frequency = (
        (rgb[:, 1:] - rgb[:, :-1]).abs().mean()
        + (rgb[1:] - rgb[:-1]).abs().mean()
    ) / (2 * 255.0)
    high_frequency_value = high_frequency.item()
    saturation_fraction = ((rgb < 3) | (rgb > 252)).float().mean().item()
    preserved = (
        target_region_fraction >= min_target_region_fraction
        and rgb_std >= min_rgb_std
        and high_frequency_value <= max_high_frequency_energy
    )
    return ImageQualityReport(
        target_region_fraction=target_region_fraction,
        rgb_std=rgb_std,
        high_frequency_energy=high_frequency_value,
        saturation_fraction=saturation_fraction,
        image_quality_preserved=preserved,
    )


def white_noise_image_control_report(
    real_quality: ImageQualityReport,
    white_noise_image: t.Tensor,
    *,
    target_color: Literal["red", "blue"],
    max_high_frequency_energy: float = 0.12,
    min_noise_gap: float = 0.12,
) -> WhiteNoiseImageReport:
    """Check that a white-noise image fails the same quality gate."""

    noise_quality = image_quality_report(
        white_noise_image,
        target_color=target_color,
        max_high_frequency_energy=max_high_frequency_energy,
    )
    rejected = (
        noise_quality.high_frequency_energy > max_high_frequency_energy
        and noise_quality.high_frequency_energy
        >= real_quality.high_frequency_energy + min_noise_gap
    )
    return WhiteNoiseImageReport(
        real_high_frequency_energy=real_quality.high_frequency_energy,
        white_noise_high_frequency_energy=noise_quality.high_frequency_energy,
        white_noise_rejected=rejected,
    )


def sd15_strict_acceptance_report(
    *,
    daam_reports: tuple[DAAMRegionReport, ...] | list[DAAMRegionReport],
    token_ablation_reports: tuple[TokenAblationReport, ...] | list[TokenAblationReport],
    image_quality_reports: tuple[ImageQualityReport, ...] | list[ImageQualityReport],
    white_noise_reports: tuple[WhiteNoiseImageReport, ...] | list[WhiteNoiseImageReport],
) -> SD15StrictReport:
    """Combine SD1.5 DAAM, ablation, quality, and white-noise controls."""

    if not daam_reports or not token_ablation_reports or not image_quality_reports:
        raise ValueError("strict SD1.5 reports must be nonempty.")
    if len(daam_reports) != len(token_ablation_reports) or len(daam_reports) != len(
        image_quality_reports
    ):
        raise ValueError("strict SD1.5 report lists must have matching lengths.")
    if len(white_noise_reports) != len(image_quality_reports):
        raise ValueError("white-noise report count must match image-quality report count.")

    daam_passed = all(report.daam_localized for report in daam_reports)
    token_ablation_passed = all(
        report.target_ablation_passed for report in token_ablation_reports
    )
    random_weaker = all(
        report.random_token_ablation_weaker for report in token_ablation_reports
    )
    quality_passed = all(
        report.image_quality_preserved for report in image_quality_reports
    )
    noise_rejected = all(report.white_noise_rejected for report in white_noise_reports)
    return SD15StrictReport(
        daam_passed=daam_passed,
        token_ablation_passed=token_ablation_passed,
        random_token_ablation_weaker=random_weaker,
        image_quality_preserved=quality_passed,
        white_noise_rejected=noise_rejected,
        sd15_strict_experiment_passed=(
            daam_passed
            and token_ablation_passed
            and random_weaker
            and quality_passed
            and noise_rejected
        ),
    )
