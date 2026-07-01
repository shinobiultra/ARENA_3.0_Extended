import importlib.util

import pytest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is not installed")

if TORCH_AVAILABLE:
    import torch as t

    from arena_ext.image_generation_interpretability import (
        attention_region_report,
        daam_region_report,
        denoising_circuit_report,
        image_quality_report,
        latent_direction_effect_report,
        prompt_region_causal_report,
        sd15_strict_acceptance_report,
        token_ablation_region_report,
        white_noise_image_control_report,
    )


def test_attention_region_report_requires_region_mass():
    attention_map = t.tensor([[0.1, 0.1], [0.2, 0.6]])
    region_mask = t.tensor([[False, False], [False, True]])

    report = attention_region_report(
        attention_map,
        region_mask,
        min_region_mass=0.5,
    )

    assert report.region_mass == pytest.approx(0.6)
    assert report.off_region_mass == pytest.approx(0.4)
    assert report.region_selective


def test_denoising_circuit_report_requires_specific_ablation_delta():
    report = denoising_circuit_report(
        baseline_loss=0.2,
        ablated_loss=0.7,
        random_control_loss=0.35,
        min_loss_increase=0.3,
        min_control_gap=0.2,
    )

    assert report.ablation_delta == pytest.approx(0.5)
    assert report.random_delta == pytest.approx(0.15)
    assert report.circuit_specific


def test_latent_direction_effect_report_requires_random_control_margin():
    baseline = t.tensor([0.1, 0.2])
    steered = t.tensor([0.7, 0.8])
    random_control = t.tensor([0.25, 0.15])

    report = latent_direction_effect_report(
        baseline,
        steered,
        random_control,
        expected_direction="increase",
        min_effect=0.5,
        min_random_margin=0.2,
    )

    assert report.observed_delta == pytest.approx(0.6)
    assert report.random_delta == pytest.approx(0.05)
    assert report.has_directional_effect


def test_prompt_region_causal_report_requires_target_drop_over_control():
    report = prompt_region_causal_report(
        original_region_score=0.9,
        ablated_region_score=0.3,
        control_region_score=0.75,
        min_target_drop=0.4,
        min_control_margin=0.2,
    )

    assert report.target_drop == pytest.approx(0.6)
    assert report.control_drop == pytest.approx(0.15)
    assert report.prompt_region_causal


def test_daam_region_report_requires_target_over_mask_and_control():
    report = daam_region_report(
        target_region_mass=0.31,
        control_region_mass=0.24,
        mask_fraction=0.20,
        captured_map_count=32,
        min_target_control_gap=0.05,
        min_lift_over_mask_fraction=0.05,
    )

    assert report.target_control_gap == pytest.approx(0.07)
    assert report.target_lift_over_mask_fraction == pytest.approx(0.11)
    assert report.daam_localized


def test_token_ablation_report_requires_target_drop_over_random_control():
    report = token_ablation_region_report(
        original_region_score=0.24,
        target_ablated_region_score=0.02,
        random_control_region_score=0.20,
        min_target_drop=0.1,
        min_random_margin=0.1,
    )

    assert report.target_drop == pytest.approx(0.22)
    assert report.random_control_drop == pytest.approx(0.04)
    assert report.target_ablation_passed
    assert report.random_token_ablation_weaker


def test_image_quality_report_rejects_white_noise_control():
    image = t.zeros(32, 32, 3)
    image[..., :] = 240
    image[8:24, 8:24, 0] = 220
    image[8:24, 8:24, 1:] = 40
    quality = image_quality_report(
        image,
        target_color="red",
        min_target_region_fraction=0.1,
        max_high_frequency_energy=0.2,
    )
    generator = t.Generator(device="cpu").manual_seed(0)
    white_noise = t.randint(0, 256, (32, 32, 3), generator=generator)
    noise = white_noise_image_control_report(
        quality,
        white_noise,
        target_color="red",
        max_high_frequency_energy=0.2,
        min_noise_gap=0.05,
    )

    assert quality.target_region_fraction == pytest.approx(0.25)
    assert quality.image_quality_preserved
    assert noise.white_noise_rejected


def test_sd15_strict_acceptance_report_requires_all_controls():
    daam = daam_region_report(
        target_region_mass=0.31,
        control_region_mass=0.24,
        mask_fraction=0.20,
        captured_map_count=32,
    )
    ablation = token_ablation_region_report(
        original_region_score=0.24,
        target_ablated_region_score=0.02,
        random_control_region_score=0.20,
    )
    image = t.zeros(32, 32, 3)
    image[..., :] = 240
    image[8:24, 8:24, 0] = 220
    image[8:24, 8:24, 1:] = 40
    quality = image_quality_report(image, target_color="red", max_high_frequency_energy=0.2)
    white_noise = t.randint(0, 256, (32, 32, 3), generator=t.Generator().manual_seed(0))
    noise = white_noise_image_control_report(
        quality,
        white_noise,
        target_color="red",
        max_high_frequency_energy=0.2,
        min_noise_gap=0.05,
    )

    report = sd15_strict_acceptance_report(
        daam_reports=[daam],
        token_ablation_reports=[ablation],
        image_quality_reports=[quality],
        white_noise_reports=[noise],
    )

    assert report.daam_passed
    assert report.token_ablation_passed
    assert report.random_token_ablation_weaker
    assert report.image_quality_preserved
    assert report.white_noise_rejected
    assert report.sd15_strict_experiment_passed
