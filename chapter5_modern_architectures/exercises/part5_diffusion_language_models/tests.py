from collections.abc import Callable
import json
from pathlib import Path

import torch as t

from arena_ext import diffusion_lm as reference


def _solutions():
    from chapter5_modern_architectures.exercises.part5_diffusion_language_models import (
        solutions,
    )

    return solutions


def test_linear_mask_schedule_and_expected_fraction(
    linear_mask_schedule: Callable | None = None,
    expected_mask_fraction: Callable | None = None,
):
    solutions = _solutions()
    linear_mask_schedule = linear_mask_schedule or solutions.linear_mask_schedule
    expected_mask_fraction = expected_mask_fraction or solutions.expected_mask_fraction
    schedule = linear_mask_schedule(5, mask_token_id=99)
    timesteps = t.tensor([0, 2, 4])
    t.testing.assert_close(
        schedule.mask_probs,
        t.tensor([0.0, 0.25, 0.5, 0.75, 1.0]),
        msg="Linear mask schedule should interpolate from 0 to 1 over num_steps.",
    )
    assert schedule.num_steps == 5, "Schedule num_steps should equal len(mask_probs)."
    assert expected_mask_fraction(schedule, timesteps) == 0.5, (
        "Expected mask fraction should average the selected timestep probabilities."
    )
    reference_schedule = reference.linear_mask_schedule(5, mask_token_id=99)
    t.testing.assert_close(
        schedule.mask_probs,
        reference_schedule.mask_probs,
        msg="Local schedule should match the independent reference implementation.",
    )
    print("All tests in `test_linear_mask_schedule_and_expected_fraction` passed!")


def test_forward_noising_extremes_and_seeded_masks(
    apply_forward_noising: Callable | None = None,
    linear_mask_schedule: Callable | None = None,
):
    solutions = _solutions()
    apply_forward_noising = apply_forward_noising or solutions.apply_forward_noising
    linear_mask_schedule = linear_mask_schedule or solutions.linear_mask_schedule
    input_ids = t.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    schedule = linear_mask_schedule(3, mask_token_id=99)
    generator = t.Generator().manual_seed(123)
    result = apply_forward_noising(input_ids, t.tensor([0, 1, 2]), schedule, generator=generator)
    reference_result = reference.apply_forward_noising(
        input_ids,
        t.tensor([0, 1, 2]),
        schedule,
        generator=t.Generator().manual_seed(123),
    )
    t.testing.assert_close(
        result.noisy_tokens[0],
        input_ids[0],
        msg="Timestep 0 should leave every token unmasked when min_mask_prob is 0.",
    )
    assert bool(result.noisy_tokens[2].eq(99).all().item()), (
        "Final timestep should mask every token when max_mask_prob is 1."
    )
    t.testing.assert_close(
        result.noisy_tokens,
        reference_result.noisy_tokens,
        msg="Seeded forward noising should match the independent reference.",
    )
    t.testing.assert_close(
        result.mask,
        reference_result.mask,
        msg="Seeded noising masks should match the independent reference.",
    )
    print("All tests in `test_forward_noising_extremes_and_seeded_masks` passed!")


def test_masked_denoising_loss_uses_only_masked_positions(
    masked_denoising_loss: Callable | None = None,
):
    solutions = _solutions()
    masked_denoising_loss = masked_denoising_loss or solutions.masked_denoising_loss
    logits = t.zeros(1, 3, 4)
    target = t.tensor([[0, 1, 2]])
    mask = t.tensor([[True, False, True]])
    logits[0, 0, 0] = 5.0
    logits[0, 1, 3] = 20.0
    logits[0, 2, 2] = 5.0
    actual = masked_denoising_loss(logits, target, mask)
    expected = reference.masked_denoising_loss(logits, target, mask)
    t.testing.assert_close(
        actual,
        expected,
        msg="Masked denoising loss should ignore unmasked positions entirely.",
    )
    assert actual.item() < 0.03, (
        "Confident correct logits at masked positions should produce low loss."
    )
    print("All tests in `test_masked_denoising_loss_uses_only_masked_positions` passed!")


def test_confidence_remask_entropy_and_uniform_control(
    confidence_remask: Callable | None = None,
    token_entropy: Callable | None = None,
    uniform_remask: Callable | None = None,
):
    solutions = _solutions()
    confidence_remask = confidence_remask or solutions.confidence_remask
    token_entropy = token_entropy or solutions.token_entropy
    uniform_remask = uniform_remask or solutions.uniform_remask
    logits = t.tensor([[[5.0, 0.0], [0.0, 0.0], [6.0, 0.0], [0.1, 0.0]]])
    current = t.zeros(1, 4, dtype=t.long)
    remasked = confidence_remask(logits, current, mask_token_id=99, next_mask_fraction=0.5)
    entropy = token_entropy(logits)
    assert int(remasked.eq(99).sum().item()) == 2, (
        "Confidence remasking should mask the requested fraction of positions."
    )
    assert remasked[0, 0] != 99 and remasked[0, 2] != 99, (
        "High-confidence positions should remain committed."
    )
    t.testing.assert_close(
        entropy,
        reference.token_entropy(logits),
        msg="Token entropy should match categorical entropy from logits.",
    )
    uniform = uniform_remask(
        t.tensor([[1, 2, 3, 4]]),
        mask_token_id=99,
        next_mask_fraction=0.5,
        generator=t.Generator().manual_seed(0),
    )
    assert int(uniform.eq(99).sum().item()) == 2, (
        "Uniform remasking should mask the requested number of positions."
    )
    print("All tests in `test_confidence_remask_entropy_and_uniform_control` passed!")


def test_oracle_diffusion_sampler_recovers_target(
    diffusion_sampler: Callable | None = None,
    linear_mask_schedule: Callable | None = None,
):
    solutions = _solutions()
    diffusion_sampler = diffusion_sampler or solutions.diffusion_sampler
    linear_mask_schedule = linear_mask_schedule or solutions.linear_mask_schedule
    target = t.tensor([[1, 2, 3, 4]])
    schedule = linear_mask_schedule(4, mask_token_id=0)

    def oracle_model(tokens, step):
        _ = tokens, step
        logits = t.zeros(1, 4, 6)
        logits.scatter_(2, target.unsqueeze(-1), 10.0)
        return logits

    output, stats = diffusion_sampler(shape=(1, 4), schedule=schedule, model_fn=oracle_model)
    assert bool(t.equal(output, target)), (
        "Oracle denoiser should exactly reconstruct the target sequence."
    )
    assert len(stats) == schedule.num_steps, (
        "Sampler should report one stats object per denoising step."
    )
    assert stats[0].step == 3 and stats[-1].step == 0, (
        "Sampler should iterate from highest noise to lowest noise."
    )
    assert stats[-1].mask_fraction == 0.0, "Final sample should contain no mask tokens."
    print("All tests in `test_oracle_diffusion_sampler_recovers_target` passed!")


def test_commitment_edit_distance_and_activation_trajectory(
    commitment_times: Callable | None = None,
    edit_distance: Callable | None = None,
    validate_activation_trajectory: Callable | None = None,
):
    solutions = _solutions()
    commitment_times = commitment_times or solutions.commitment_times
    edit_distance = edit_distance or solutions.edit_distance
    validate_activation_trajectory = (
        validate_activation_trajectory or solutions.validate_activation_trajectory
    )
    trajectory = t.tensor(
        [
            [[0, 0, 0]],
            [[1, 0, 0]],
            [[1, 2, 0]],
            [[1, 2, 3]],
        ]
    )
    activations = [t.zeros(1, 3, 4) for _ in range(4)]
    t.testing.assert_close(
        commitment_times(trajectory, mask_token_id=0),
        t.tensor([[1, 2, 3]]),
        msg="Commitment time should be the first trajectory index where a token is unmasked.",
    )
    assert edit_distance([1, 2, 3], [1, 4, 3, 5]) == 2, (
        "Edit distance should count one substitution and one insertion in this example."
    )
    assert validate_activation_trajectory(
        activations,
        expected_steps=4,
        batch=1,
        seq_len=3,
    ), "Activation trajectory should validate when every step has matching leading dims."
    assert not validate_activation_trajectory(
        activations[:-1],
        expected_steps=4,
        batch=1,
        seq_len=3,
    ), "Activation trajectory should fail when the number of steps is wrong."
    print("All tests in `test_commitment_edit_distance_and_activation_trajectory` passed!")


def test_tiny_conditional_diffusion_lm_forward_shape(TinyConditionalDiffusionLM: type | None = None):
    solutions = _solutions()
    TinyConditionalDiffusionLM = TinyConditionalDiffusionLM or solutions.TinyConditionalDiffusionLM
    model = TinyConditionalDiffusionLM(d_model=32).eval()
    input_ids = t.tensor([[1, 2, 10, 10, 10, 10], [3, 4, 10, 10, 10, 10]])
    timesteps = t.tensor([0, 5])
    with t.inference_mode():
        logits = model(input_ids, timesteps)
    assert logits.shape == (2, 6, 11), (
        f"Diffusion LM logits should have shape (batch, seq, vocab), got {tuple(logits.shape)}."
    )
    print("All tests in `test_tiny_conditional_diffusion_lm_forward_shape` passed!")


def test_notebook_contract(run_smoke_test: Callable | None = None):
    if run_smoke_test is None:
        run_smoke_test = _solutions().run_smoke_test
    result = run_smoke_test(cpu=True)
    assert result["schedule"]["mask_probs"] == [0.0, 0.25, 0.5, 0.75, 1.0], (
        "Smoke schedule should expose the expected linear mask probabilities."
    )
    assert result["noising"]["first_unchanged"], (
        "Smoke noising should verify timestep-0 leaves tokens unchanged."
    )
    assert result["noising"]["second_all_masked"], (
        "Smoke noising should verify max-noise timestep masks every token."
    )
    assert result["denoising_loss"] < 0.03, (
        "Smoke denoising loss should be low for confident masked-token logits."
    )
    assert result["remasking"]["num_masked"] == 2, (
        "Smoke remasking should mask two low-confidence positions."
    )
    assert result["oracle_sampler"]["matches_target"], (
        "Smoke oracle sampler should reconstruct the target sequence."
    )
    assert result["trajectory"]["activation_shape_ok"], (
        "Smoke trajectory diagnostics should validate activation shapes."
    )
    print("All tests in `test_notebook_contract` passed!")


def test_diffusiongemma_vllm_probe_artifact_contract():
    probe_path = (
        Path(__file__).resolve().parent
        / "artifacts/diffusiongemma_vllm_probe.json"
    )
    assert probe_path.exists(), (
        "The section should include the real DiffusionGemma NVFP4 vLLM proof artifact."
    )
    proof = json.loads(probe_path.read_text())

    assert proof["vllm_version"] == "0.24.0", (
        "The proof should record the exact vLLM runtime used for DiffusionGemma NVFP4."
    )
    assert proof["torch_version"] == "2.11.0+cu130", (
        "The proof should show the isolated vLLM torch stack, not the main uv stack."
    )
    assert proof["torch_cuda_version"] == "13.0", (
        "The isolated vLLM runtime should record the CUDA version it actually used."
    )
    assert proof["cuda_available"] is True, (
        "The proof should come from a real CUDA generation run, not CPU metadata."
    )
    assert proof["used_chat_template"] is True, (
        "DiffusionGemma should be prompted through its tokenizer chat template."
    )
    assert proof["canvas_length"] == 256, (
        "The proof should use the pinned DiffusionGemma block-diffusion canvas length."
    )
    assert proof["max_model_len"] >= 4096, (
        "The proof should reserve a real context window rather than a toy prompt-only stub."
    )
    assert proof["gpu_name"] == "NVIDIA GeForce RTX 5090 Laptop GPU", (
        "The proof should identify the local 24GB-class GPU used for generation."
    )
    assert 20.0 <= float(proof["gpu_total_memory_gib"]) <= 24.5, (
        "The proof should be tied to the intended single 24GB GPU tier."
    )
    output = proof["output"].strip()
    assert output, "The vLLM proof must contain a real, non-empty generated output."
    assert "negative" in output.lower() and "control" in output.lower(), (
        "The generated output should answer the fixed negative-control prompt."
    )

    preflight = _solutions().run_diffusiongemma_readiness_preflight(allow_network=False)
    assert preflight["diffusiongemma_generation_ready"] is True, (
        "Readiness should pass only because the real NVFP4 proof artifact is valid."
    )
    assert preflight["diffusiongemma_bf16_24gb_direct_loading_deferred"] is True, (
        "BF16 direct loading should stay explicitly deferred for the 24GB tier."
    )
    assert preflight["diffusiongemma_nvfp4_isolated_vllm_generation_ready"] is True, (
        "The NVFP4 released-checkpoint path should be marked ready via isolated vLLM."
    )
    assert preflight["diffusiongemma_main_uv_vllm_generation_supported"] is False, (
        "The main uv environment should remain torch 2.12/cu132 without vLLM."
    )
    assert "No real DiffusionGemma generation path has been executed." not in preflight[
        "diffusiongemma_blockers"
    ], "The old no-generation blocker should disappear after the real NVFP4 proof."
    print("All tests in `test_diffusiongemma_vllm_probe_artifact_contract` passed!")
