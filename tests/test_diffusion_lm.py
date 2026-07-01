import importlib.util

import pytest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is not installed")

if TORCH_AVAILABLE:
    import torch as t

    import arena_ext.diffusion_lm as diffusion_lm_module
    from arena_ext.diffusion_lm import (
        apply_forward_noising,
        commitment_times,
        confidence_remask,
        diffusiongemma_readiness_report,
        diffusion_sampler,
        edit_distance,
        expected_mask_fraction,
        linear_mask_schedule,
        masked_denoising_loss,
        token_entropy,
        uniform_remask,
        validate_activation_trajectory,
    )


def test_linear_schedule_and_expected_mask_fraction():
    schedule = linear_mask_schedule(5, mask_token_id=99)
    timesteps = t.tensor([0, 2, 4])

    assert schedule.mask_probs.tolist() == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])
    assert expected_mask_fraction(schedule, timesteps) == pytest.approx(0.5)


def test_forward_noising_extreme_timesteps():
    schedule = linear_mask_schedule(2, mask_token_id=99)
    input_ids = t.tensor([[1, 2, 3], [4, 5, 6]])
    timesteps = t.tensor([0, 1])
    result = apply_forward_noising(input_ids, timesteps, schedule)

    assert t.equal(result.noisy_tokens[0], input_ids[0])
    assert t.equal(result.noisy_tokens[1], t.full((3,), 99))
    assert result.mask[0].sum() == 0
    assert result.mask[1].sum() == 3


def test_masked_denoising_loss_only_uses_masked_positions():
    logits = t.zeros(1, 3, 4)
    target = t.tensor([[0, 1, 2]])
    mask = t.tensor([[True, False, True]])
    logits[0, 0, 0] = 5.0
    logits[0, 2, 2] = 5.0

    loss = masked_denoising_loss(logits, target, mask)

    assert loss.item() < 0.03


def test_token_entropy_is_low_for_confident_logits():
    logits = t.tensor([[[10.0, -10.0], [0.0, 0.0]]])
    entropy = token_entropy(logits)

    assert entropy[0, 0] < 1e-3
    assert entropy[0, 1] == pytest.approx(t.log(t.tensor(2.0)).item())


def test_confidence_remask_masks_low_confidence_positions():
    logits = t.tensor(
        [
            [
                [5.0, 0.0],
                [0.0, 0.0],
                [6.0, 0.0],
                [0.1, 0.0],
            ]
        ]
    )
    current = t.zeros(1, 4, dtype=t.long)
    remasked = confidence_remask(logits, current, mask_token_id=99, next_mask_fraction=0.5)

    assert remasked.eq(99).sum() == 2
    assert remasked[0, 0] != 99
    assert remasked[0, 2] != 99


def test_uniform_remask_masks_requested_count():
    generator = t.Generator().manual_seed(0)
    tokens = t.arange(8).reshape(1, 8)
    remasked = uniform_remask(
        tokens,
        mask_token_id=99,
        next_mask_fraction=0.25,
        generator=generator,
    )

    assert remasked.eq(99).sum() == 2


def test_diffusion_sampler_reconstructs_oracle_target():
    target = t.tensor([[1, 2, 3, 4]])
    schedule = linear_mask_schedule(4, mask_token_id=0)

    def oracle_model(tokens, step):
        _ = tokens, step
        logits = t.zeros(1, 4, 6)
        logits.scatter_(2, target.unsqueeze(-1), 10.0)
        return logits

    output, stats = diffusion_sampler(shape=(1, 4), schedule=schedule, model_fn=oracle_model)

    assert t.equal(output, target)
    assert len(stats) == schedule.num_steps
    assert stats[-1].mask_fraction == 0.0


def test_commitment_times_and_edit_distance():
    trajectory = t.tensor(
        [
            [[0, 0, 0]],
            [[1, 0, 0]],
            [[1, 2, 0]],
            [[1, 2, 3]],
        ]
    )

    assert t.equal(commitment_times(trajectory, mask_token_id=0), t.tensor([[1, 2, 3]]))
    assert edit_distance([1, 2, 3], [1, 4, 3, 5]) == 2


def test_validate_activation_trajectory_checks_shape():
    activations = [t.zeros(2, 3, 4), t.ones(2, 3, 4)]

    assert validate_activation_trajectory(activations, expected_steps=2, batch=2, seq_len=3)
    assert not validate_activation_trajectory(activations, expected_steps=3, batch=2, seq_len=3)


def test_diffusiongemma_readiness_rejects_config_only_evidence(monkeypatch):
    def fake_access_report(spec, *, allow_network=True, cache_root=None):
        _ = allow_network, cache_root
        return {
            "repo_id": spec.repo_id,
            "revision": spec.revision,
            "local_ready_for_direct_loading": False,
            "remote_download_ready": True,
        }

    monkeypatch.setattr(
        diffusion_lm_module,
        "hf_model_artifact_access_report",
        fake_access_report,
    )
    monkeypatch.setattr(
        diffusion_lm_module,
        "_diffusiongemma_transformers_metadata",
        lambda **kwargs: {
            "config_supported": True,
            "processor_supported": True,
            "model_class_supported": True,
            "config_model_type": "diffusion_gemma",
            "config_architectures": ("DiffusionGemmaForBlockDiffusion",),
            "tokenizer_mask_token_id": 4,
            "canvas_length": 256,
            "default_max_denoising_steps": 48,
            "errors": [],
        },
    )
    monkeypatch.setattr(
        diffusion_lm_module,
        "_nvfp4_quantization_metadata",
        lambda **kwargs: {
            "quant_method": "modelopt",
            "transformers_quantization_supported": False,
            "transformers_quantization_error": "unsupported modelopt",
        },
    )
    monkeypatch.setattr(diffusion_lm_module, "_local_weight_shard_count", lambda spec: 0)
    monkeypatch.setattr(diffusion_lm_module, "_remote_weight_bytes", lambda spec, *, allow_network: 1)
    monkeypatch.setattr(
        diffusion_lm_module,
        "_external_vllm_probe_report",
        lambda: {
            "path": "missing.json",
            "ready": False,
            "runtime_isolated": False,
            "model_matches_nvfp4_revision": False,
            "output_nonempty": False,
            "output_mentions_negative_controls": False,
            "used_chat_template": None,
            "prompt": None,
            "output_preview": None,
            "torch_version": None,
            "torch_cuda_version": None,
            "vllm_version": None,
            "cuda_available": None,
            "gpu_name": None,
            "gpu_total_memory_gib": None,
            "load_seconds": None,
            "generate_seconds": None,
            "error": "missing proof",
        },
    )
    monkeypatch.setattr(diffusion_lm_module.t.cuda, "is_available", lambda: True)

    report = diffusiongemma_readiness_report(allow_network=False)

    assert report.config_supported
    assert report.processor_supported
    assert report.model_class_supported
    assert report.config_model_type == "diffusion_gemma"
    assert report.nvfp4_quant_method == "modelopt"
    assert report.nvfp4_transformers_quantization_supported is False
    assert report.generation_ready is False
    assert "No real DiffusionGemma generation path has been executed." in report.blockers


def test_diffusiongemma_readiness_accepts_external_vllm_proof(monkeypatch):
    def fake_access_report(spec, *, allow_network=True, cache_root=None):
        _ = allow_network, cache_root
        return {
            "repo_id": spec.repo_id,
            "revision": spec.revision,
            "local_ready_for_direct_loading": (
                spec.repo_id
                == diffusion_lm_module.NVIDIA_DIFFUSIONGEMMA_26B_A4B_IT_NVFP4.repo_id
            ),
            "remote_download_ready": True,
        }

    monkeypatch.setattr(
        diffusion_lm_module,
        "hf_model_artifact_access_report",
        fake_access_report,
    )
    monkeypatch.setattr(
        diffusion_lm_module,
        "_diffusiongemma_transformers_metadata",
        lambda **kwargs: {
            "config_supported": True,
            "processor_supported": True,
            "model_class_supported": True,
            "config_model_type": "diffusion_gemma",
            "config_architectures": ("DiffusionGemmaForBlockDiffusion",),
            "tokenizer_mask_token_id": 4,
            "canvas_length": 256,
            "default_max_denoising_steps": 48,
            "errors": [],
        },
    )
    monkeypatch.setattr(
        diffusion_lm_module,
        "_nvfp4_quantization_metadata",
        lambda **kwargs: {
            "quant_method": "modelopt",
            "transformers_quantization_supported": False,
            "transformers_quantization_error": "unsupported modelopt",
        },
    )
    monkeypatch.setattr(
        diffusion_lm_module,
        "_local_weight_shard_count",
        lambda spec: 2
        if spec.repo_id == diffusion_lm_module.NVIDIA_DIFFUSIONGEMMA_26B_A4B_IT_NVFP4.repo_id
        else 0,
    )
    monkeypatch.setattr(
        diffusion_lm_module,
        "_remote_weight_bytes",
        lambda spec, *, allow_network: 18_823_855_888
        if spec.repo_id == diffusion_lm_module.NVIDIA_DIFFUSIONGEMMA_26B_A4B_IT_NVFP4.repo_id
        else 51_647_701_024,
    )
    monkeypatch.setattr(
        diffusion_lm_module,
        "_external_vllm_probe_report",
        lambda: {
            "path": "chapter5_modern_architectures/exercises/part5_diffusion_language_models/artifacts/diffusiongemma_vllm_probe.json",
            "ready": True,
            "runtime_isolated": True,
            "model_matches_nvfp4_revision": True,
            "output_nonempty": True,
            "output_mentions_negative_controls": True,
            "used_chat_template": True,
            "prompt": "Why use negative controls?",
            "output_preview": "Negative controls check causal artifacts.",
            "torch_version": "2.11.0+cu130",
            "torch_cuda_version": "13.0",
            "vllm_version": "0.24.0",
            "cuda_available": True,
            "gpu_name": "NVIDIA GeForce RTX 5090 Laptop GPU",
            "gpu_total_memory_gib": 23.45965576171875,
            "load_seconds": 25.0,
            "generate_seconds": 0.4,
            "error": None,
        },
    )
    monkeypatch.setattr(diffusion_lm_module.t.cuda, "is_available", lambda: True)

    report = diffusiongemma_readiness_report(allow_network=False)

    assert report.bf16_local_ready_for_direct_loading is False
    assert report.nvfp4_local_ready_for_vllm is True
    assert report.vllm_preserves_current_torch_cuda_stack is False
    assert report.external_vllm_generation_ready is True
    assert report.external_vllm_runtime_isolated is True
    assert report.external_vllm_output_mentions_negative_controls is True
    assert report.generation_ready is True
    assert "No real DiffusionGemma generation path has been executed." not in report.blockers
