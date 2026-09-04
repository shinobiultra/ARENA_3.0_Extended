import json
from collections.abc import Callable
from pathlib import Path

import torch as t


def _solutions():
    from chapter13_image_generation_interpretability.exercises.part1_diffusion_image_controls import (
        solutions,
    )

    return solutions


def _value(report, key: str):
    if isinstance(report, dict):
        return report[key]
    return getattr(report, key)


def _red_square_image(size: int = 32) -> t.Tensor:
    image = t.ones(size, size, 3)
    image[10:22, 10:22, 0] = 1.0
    image[10:22, 10:22, 1] = 0.0
    image[10:22, 10:22, 2] = 0.0
    return image


def test_attention_region_report_measures_mass_and_rejects_bad_masks(
    attention_region_report: Callable | None = None,
):
    attention_region_report = (
        attention_region_report or _solutions().attention_region_report
    )
    attention = t.tensor([[0.1, 0.1], [0.2, 0.6]])
    mask = t.tensor([[False, False], [False, True]])

    report = attention_region_report(attention, mask, min_region_mass=0.5)
    assert abs(_value(report, "region_mass") - 0.6) < 1e-6, (
        "The selected region should contain exactly 0.6 of the normalized attention mass."
    )
    assert abs(_value(report, "off_region_mass") - 0.4) < 1e-6, (
        "The off-region mass should be the complement of the selected region mass."
    )
    assert _value(report, "region_selective"), (
        "The toy attention map should pass the 0.5 target-region mass threshold."
    )

    weak_report = attention_region_report(attention, mask, min_region_mass=0.7)
    assert not _value(weak_report, "region_selective"), (
        "The threshold should be meaningful; the same map should fail a stricter gate."
    )

    try:
        attention_region_report(attention, t.zeros_like(mask, dtype=t.bool))
    except ValueError:
        pass
    else:
        raise AssertionError("Empty masks should be rejected.")

    print(
        "All tests in `test_attention_region_report_measures_mass_and_rejects_bad_masks` passed!"
    )


def test_denoising_circuit_report_requires_specificity(
    denoising_circuit_report: Callable | None = None,
):
    denoising_circuit_report = (
        denoising_circuit_report or _solutions().denoising_circuit_report
    )
    report = denoising_circuit_report(
        baseline_loss=0.2,
        ablated_loss=0.7,
        random_control_loss=0.35,
        min_loss_increase=0.3,
        min_control_gap=0.2,
    )
    assert abs(_value(report, "ablation_delta") - 0.5) < 1e-6, (
        "The target ablation should increase loss by 0.5 in the fixture."
    )
    assert abs(_value(report, "random_delta") - 0.15) < 1e-6, (
        "The random-control ablation should increase loss by only 0.15."
    )
    assert _value(report, "circuit_specific"), (
        "The target circuit should pass only when its loss delta beats the random control."
    )

    nonspecific = denoising_circuit_report(
        baseline_loss=0.2,
        ablated_loss=0.52,
        random_control_loss=0.42,
        min_loss_increase=0.3,
        min_control_gap=0.2,
    )
    assert not _value(nonspecific, "circuit_specific"), (
        "Ablations must beat same-size random controls, not merely hurt loss."
    )

    print("All tests in `test_denoising_circuit_report_requires_specificity` passed!")


def test_latent_direction_report_requires_random_margin(
    latent_direction_effect_report: Callable | None = None,
):
    latent_direction_effect_report = (
        latent_direction_effect_report or _solutions().latent_direction_effect_report
    )
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
    assert abs(_value(report, "observed_delta") - 0.6) < 1e-6, (
        "The steered latent scores should increase by 0.6 on this paired fixture."
    )
    assert abs(_value(report, "random_delta") - 0.05) < 1e-6, (
        "The random latent direction should have only a small mean score effect."
    )
    assert _value(report, "has_directional_effect"), (
        "The target latent direction should clear both effect-size and random-margin gates."
    )

    weak_control = latent_direction_effect_report(
        baseline,
        steered,
        t.tensor([0.62, 0.68]),
        expected_direction="increase",
        min_effect=0.5,
        min_random_margin=0.2,
    )
    assert not _value(weak_control, "has_directional_effect"), (
        "A latent direction should fail when a random direction explains the effect."
    )

    print("All tests in `test_latent_direction_report_requires_random_margin` passed!")


def test_prompt_region_report_requires_target_drop(
    prompt_region_causal_report: Callable | None = None,
):
    prompt_region_causal_report = (
        prompt_region_causal_report or _solutions().prompt_region_causal_report
    )
    report = prompt_region_causal_report(
        original_region_score=0.85,
        ablated_region_score=0.25,
        control_region_score=0.7,
        min_target_drop=0.4,
        min_control_margin=0.2,
    )
    assert abs(_value(report, "target_drop") - 0.6) < 1e-6, (
        "Ablating the target token should reduce the target-region score by 0.6."
    )
    assert abs(_value(report, "control_drop") - 0.15) < 1e-6, (
        "The unrelated-token control should have a much smaller target-region drop."
    )
    assert _value(report, "prompt_region_causal"), (
        "The prompt-region claim should pass when target drop beats the control drop."
    )

    weak_report = prompt_region_causal_report(
        original_region_score=0.85,
        ablated_region_score=0.55,
        control_region_score=0.62,
        min_target_drop=0.4,
        min_control_margin=0.2,
    )
    assert not _value(weak_report, "prompt_region_causal"), (
        "Target-token ablation should fail when the target drop is too small."
    )

    print("All tests in `test_prompt_region_report_requires_target_drop` passed!")


def test_sd15_toy_control_reports(
    daam_region_report: Callable | None = None,
    token_ablation_region_report: Callable | None = None,
    image_quality_report: Callable | None = None,
    white_noise_image_control_report: Callable | None = None,
    sd15_strict_acceptance_report: Callable | None = None,
):
    solutions = _solutions()
    daam_region_report = daam_region_report or solutions.daam_region_report
    token_ablation_region_report = (
        token_ablation_region_report or solutions.token_ablation_region_report
    )
    image_quality_report = image_quality_report or solutions.image_quality_report
    white_noise_image_control_report = (
        white_noise_image_control_report or solutions.white_noise_image_control_report
    )
    sd15_strict_acceptance_report = (
        sd15_strict_acceptance_report or solutions.sd15_strict_acceptance_report
    )

    daam = daam_region_report(
        target_region_mass=0.21,
        control_region_mass=0.08,
        mask_fraction=0.12,
        captured_map_count=64,
        min_target_control_gap=0.05,
        min_lift_over_mask_fraction=0.05,
        min_captured_map_count=32,
    )
    assert _value(daam, "daam_localized"), (
        "Target-token attention should localize only when it beats control attention and mask fraction."
    )
    assert abs(_value(daam, "target_control_gap") - 0.13) < 1e-6, (
        "The DAAM-style target-control gap should be target mass minus control mass."
    )

    token = token_ablation_region_report(
        original_region_score=0.38,
        target_ablated_region_score=0.18,
        random_control_region_score=0.35,
        min_target_drop=0.1,
        min_random_margin=0.1,
    )
    assert _value(token, "target_ablation_passed"), (
        "Target-token ablation should clear the minimum region-score drop."
    )
    assert _value(token, "random_token_ablation_weaker"), (
        "The target-token drop should exceed the random/control-token drop by a margin."
    )

    quality = image_quality_report(
        _red_square_image(),
        target_color="red",
        min_target_region_fraction=0.05,
        min_rgb_std=0.05,
        max_high_frequency_energy=0.18,
    )
    assert _value(quality, "image_quality_preserved"), (
        "The toy red-square image should pass nonblank, color-region, and high-frequency gates."
    )
    assert _value(quality, "target_region_fraction") > 0.1, (
        "The red-square mask should cover a measurable fraction of the toy image."
    )

    generator = t.Generator().manual_seed(0)
    noise = white_noise_image_control_report(
        quality,
        t.rand(32, 32, 3, generator=generator),
        target_color="red",
        max_high_frequency_energy=0.18,
        min_noise_gap=0.08,
    )
    assert _value(noise, "white_noise_rejected"), (
        "White noise should fail the same high-frequency quality gate used for generated images."
    )

    strict = sd15_strict_acceptance_report(
        daam_reports=[daam],
        token_ablation_reports=[token],
        image_quality_reports=[quality],
        white_noise_reports=[noise],
    )
    assert _value(strict, "sd15_strict_experiment_passed"), (
        "The strict report should pass only when DAAM, ablation, quality, and noise controls all pass."
    )

    failed_daam = daam_region_report(
        target_region_mass=0.11,
        control_region_mass=0.10,
        mask_fraction=0.12,
        captured_map_count=64,
        min_target_control_gap=0.05,
        min_lift_over_mask_fraction=0.05,
        min_captured_map_count=32,
    )
    failed_strict = sd15_strict_acceptance_report(
        daam_reports=[failed_daam],
        token_ablation_reports=[token],
        image_quality_reports=[quality],
        white_noise_reports=[noise],
    )
    assert not _value(failed_strict, "sd15_strict_experiment_passed"), (
        "The strict aggregate should fail if the DAAM-style localization report fails."
    )

    print("All tests in `test_sd15_toy_control_reports` passed!")


def test_attention_region_smoke_test(
    attention_region_smoke_test: Callable | None = None,
):
    attention_region_smoke_test = (
        attention_region_smoke_test or _solutions().attention_region_smoke_test
    )
    result = attention_region_smoke_test()
    assert abs(result["region_mass"] - 0.6) < 1e-6, (
        "The target mask should contain 0.6 of the toy attention mass."
    )
    assert abs(result["off_region_mass"] - 0.4) < 1e-6, (
        "The off-region attention mass should be the remaining 0.4."
    )
    assert result["region_selective"], (
        "The toy attention map should clear the configured target-region mass threshold."
    )
    print("All tests in `test_attention_region_smoke_test` passed!")


def test_denoising_circuit_smoke_test(
    denoising_circuit_smoke_test: Callable | None = None,
):
    denoising_circuit_smoke_test = (
        denoising_circuit_smoke_test or _solutions().denoising_circuit_smoke_test
    )
    result = denoising_circuit_smoke_test()
    assert abs(result["ablation_delta"] - 0.5) < 1e-6, (
        "The proposed circuit ablation should increase loss by 0.5 in this fixture."
    )
    assert abs(result["random_delta"] - 0.15) < 1e-6, (
        "The random-control ablation should increase loss by only 0.15."
    )
    assert result["circuit_specific"], (
        "The proposed circuit should beat the same-size random-control ablation."
    )
    print("All tests in `test_denoising_circuit_smoke_test` passed!")


def test_latent_direction_smoke_test(
    latent_direction_smoke_test: Callable | None = None,
):
    latent_direction_smoke_test = (
        latent_direction_smoke_test or _solutions().latent_direction_smoke_test
    )
    result = latent_direction_smoke_test()
    assert abs(result["observed_delta"] - 0.6) < 1e-6, (
        "The target latent direction should increase the paired score by 0.6."
    )
    assert abs(result["random_delta"] - 0.05) < 1e-6, (
        "The random latent direction should have only a small score effect."
    )
    assert result["has_directional_effect"], (
        "The target latent direction should clear both effect-size and random-margin gates."
    )
    print("All tests in `test_latent_direction_smoke_test` passed!")


def test_prompt_region_smoke_test(prompt_region_smoke_test: Callable | None = None):
    prompt_region_smoke_test = (
        prompt_region_smoke_test or _solutions().prompt_region_smoke_test
    )
    result = prompt_region_smoke_test()
    assert abs(result["target_drop"] - 0.6) < 1e-6, (
        "Ablating the target prompt token should drop the target-region score by 0.6."
    )
    assert abs(result["control_drop"] - 0.15) < 1e-6, (
        "The unrelated-token control should have a much smaller target-region drop."
    )
    assert result["prompt_region_causal"], (
        "The prompt-region claim should pass only when target drop beats the control drop."
    )
    print("All tests in `test_prompt_region_smoke_test` passed!")


def test_notebook_contract(run_smoke_test: Callable | None = None):
    run_smoke_test = run_smoke_test or _solutions().run_smoke_test
    result = run_smoke_test(cpu=True)
    assert result["attention_region"]["region_selective"], (
        "The notebook contract should include a passing attention-region report."
    )
    assert result["denoising_circuit"]["circuit_specific"], (
        "The notebook contract should include the denoising random-control check."
    )
    assert result["latent_direction"]["has_directional_effect"], (
        "The notebook contract should include the latent-direction random-control check."
    )
    assert result["prompt_region"]["prompt_region_causal"], (
        "The notebook contract should include the prompt-region causal check."
    )
    print("All tests in `test_notebook_contract` passed!")


def test_committed_gpu_report_requires_sd15_strict_controls(report: dict | None = None):
    if report is None:
        report_path = Path(__file__).with_name("verification_report.json")
        report = json.loads(report_path.read_text())
    metrics = report["metrics"]["gpu_test"]
    controls = set(report["baselines"]["declared_controls"])

    def require_metric(condition: bool, message: str) -> None:
        assert condition, message

    assert metrics["sd15_strict_experiment_passed"], (
        "The committed report must include the strict SD1.5 acceptance path."
    )
    require_metric(
        metrics["sd15_model_id"] == "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "The committed report should use the pinned Stable Diffusion 1.5 model id.",
    )
    require_metric(
        metrics["sd15_revision"] == "451f4fe16113bff5a5d2269ed5ad43b0592e9a14",
        "The committed report should use the pinned Stable Diffusion 1.5 revision.",
    )
    require_metric(
        metrics["sd15_fixed_seed_generation_passed"],
        "The SD1.5 report should include deterministic fixed-seed generations.",
    )
    require_metric(
        metrics["sd15_daam_baseline_included"],
        "The SD1.5 report should include a passing DAAM-style attention baseline.",
    )
    require_metric(
        metrics["sd15_cross_attention_maps_captured"],
        "The SD1.5 report should capture real cross-attention maps.",
    )
    require_metric(
        metrics["sd15_token_ablation_passed"],
        "The SD1.5 report should show target-token ablation changes the target region.",
    )
    require_metric(
        metrics["sd15_random_token_ablation_weaker"],
        "The SD1.5 target-token ablation should beat the random/control-token ablation.",
    )
    require_metric(
        metrics["sd15_image_quality_preserved"],
        "The SD1.5 generated images should pass the image-quality gate.",
    )
    require_metric(
        metrics["sd15_white_noise_rejected"],
        "The SD1.5 quality gate should reject white-noise controls.",
    )
    require_metric(
        metrics["sd15_min_target_control_attention_gap"] >= 0.005,
        "The SD1.5 target-token attention gap should clear the minimum threshold.",
    )
    require_metric(
        metrics["sd15_min_target_lift_over_mask_fraction"] >= 0.01,
        "The SD1.5 target-token attention should lift over the color-mask fraction.",
    )
    require_metric(
        metrics["sd15_min_captured_cross_attention_map_count"] >= 32,
        "The SD1.5 report should capture enough cross-attention maps.",
    )
    require_metric(
        metrics["sd15_min_target_ablation_drop"] >= 0.05,
        "The SD1.5 target-token ablation drop should clear the minimum threshold.",
    )
    require_metric(
        metrics["sd15_max_random_control_drop"] <= 0.0,
        "The SD1.5 random/control-token ablation should not match the target-token drop.",
    )
    require_metric(
        metrics["sd15_min_target_region_fraction"] >= 0.02,
        "The SD1.5 images should contain a measurable target-color region.",
    )
    require_metric(
        metrics["sd15_max_high_frequency_energy"] <= 0.12,
        "The SD1.5 images should not look like high-frequency white noise.",
    )
    require_metric(
        metrics["sd15_min_white_noise_high_frequency_gap"] >= 0.12,
        "The SD1.5 white-noise control should have much higher high-frequency energy.",
    )
    require_metric(
        metrics["sd15_clip_image_to_text_accuracy"] == 1.0,
        "The SD1.5 generated images should align to the right CLIP text prompt.",
    )
    require_metric(
        metrics["sd15_clip_text_to_image_accuracy"] == 1.0,
        "The SD1.5 text prompts should retrieve the right generated image.",
    )
    require_metric(
        metrics["sd15_clip_mean_positive_margin"] >= 2.0,
        "The SD1.5 CLIP margin should clear the minimum alignment threshold.",
    )

    required_controls = {
        "pinned_sd15_safetensors_generation",
        "sd15_daam_cross_attention_capture",
        "sd15_target_token_ablation_control",
        "sd15_random_token_ablation_control",
        "sd15_image_quality_metric",
        "sd15_white_noise_control",
    }
    require_metric(
        required_controls <= controls,
        "The artifact controls should declare all strict SD1.5 acceptance controls.",
    )
    require_metric(
        "full_sd15_daam_replication_not_claimed" not in controls,
        "The artifact controls should no longer disclaim SD1.5 DAAM-style evidence.",
    )
    print("All tests in `test_committed_gpu_report_requires_sd15_strict_controls` passed!")


def validate_sd15_signature_visual_payload(signature_result: dict) -> None:
    """Validate the live PIL images and attention maps used by the learner notebook."""

    assert signature_result["preflight_passed"], (
        "The live visual payload should only be displayed after every strict SD1.5 gate passes."
    )
    visual_cases = signature_result.get("visual_cases", [])
    case_reports = {
        case["case_id"]: case for case in signature_result["case_reports"]
    }
    assert len(visual_cases) == len(case_reports) == 2, (
        "The signature result should contain both preregistered shape cases."
    )
    for visual in visual_cases:
        case_id = visual["case_id"]
        report = case_reports[case_id]
        assert visual["original_image"].size == (512, 512), (
            f"{case_id} should retain the full 512x512 generated image."
        )
        assert visual["target_ablated_image"].size == (512, 512), (
            f"{case_id} should retain the same-seed target-token ablation."
        )
        assert visual["control_ablated_image"].size == (512, 512), (
            f"{case_id} should retain the same-seed control-token ablation."
        )
        assert visual["target_attention"].shape == visual["control_attention"].shape == (
            64,
            64,
        ), f"{case_id} target and control attention maps should share a 64x64 grid."
        assert visual["region_mask"].shape == (64, 64), (
            f"{case_id} color-region mask should align with the attention grid."
        )
        assert t.isfinite(visual["target_attention"]).all(), (
            f"{case_id} target attention map should contain finite values."
        )
        assert report["target_control_gap"] > 0, (
            f"{case_id} target attention should beat the control token in-region."
        )
        assert report["target_drop"] >= 0.05, (
            f"{case_id} target-token removal should erase measurable target color."
        )
        assert report["target_drop"] >= report["random_control_drop"] + 0.05, (
            f"{case_id} target-token removal should beat the control edit by margin."
        )
    print("All tests in `validate_sd15_signature_visual_payload` passed!")


def test_exercise_notebook_declares_full_verification_contract():
    notebook_path = Path(__file__).with_name(
        "13.1_Diffusion_and_Image_Generation_Controls_exercises.ipynb"
    )
    notebook = json.loads(notebook_path.read_text())
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
    )

    assert "REQUIRES_GPU = True" in source, (
        "The learner notebook should not advertise CPU-only scope for this GT-1 diffusion section."
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
    assert "test_committed_gpu_report_requires_sd15_strict_controls" in source, (
        "The learner notebook should end by checking the committed SD1.5 report."
    )
    assert "## Try It Yourself" in source, (
        "The learner notebook should expose editable attention and ablation controls."
    )
    assert "run_sd15_image_generation_signature_result" in source, (
        "The learner notebook should run the live SD1.5 signature result."
    )
    assert "diffusion_image_generation_signature.png" in source, (
        "The learner notebook should generate and display the SD1.5 evidence panel."
    )
    print("All tests in `test_exercise_notebook_declares_full_verification_contract` passed!")
