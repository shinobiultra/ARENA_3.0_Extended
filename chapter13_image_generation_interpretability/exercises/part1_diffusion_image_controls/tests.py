import json
from collections.abc import Callable
from pathlib import Path


def _solutions():
    from chapter13_image_generation_interpretability.exercises.part1_diffusion_image_controls import (
        solutions,
    )

    return solutions


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


def test_committed_gpu_report_requires_sd15_strict_controls():
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
    print("All tests in `test_exercise_notebook_declares_full_verification_contract` passed!")
