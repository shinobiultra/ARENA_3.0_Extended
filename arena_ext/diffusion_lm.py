"""Toy discrete diffusion language-model utilities for ARENA extensions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import metadata
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import torch as t
import torch.nn.functional as F

from arena_ext.gated_artifacts import (
    DIFFUSIONGEMMA_26B_A4B_IT,
    NVIDIA_DIFFUSIONGEMMA_26B_A4B_IT_NVFP4,
    HFGatedArtifactSpec,
    hf_cache_repo_dir,
    hf_model_artifact_access_report,
)

ROOT = Path(__file__).resolve().parents[1]
DIFFUSIONGEMMA_EXTERNAL_VLLM_PROBE_PATH = (
    ROOT
    / "chapter5_modern_architectures/exercises/part5_diffusion_language_models/"
    "artifacts/diffusiongemma_vllm_probe.json"
)
DIFFUSIONGEMMA_BF16_WEIGHT_BYTES_REQUIRED = 51_647_701_024
DIFFUSIONGEMMA_NVFP4_WEIGHT_BYTES_REQUIRED = 18_823_855_888


@dataclass(frozen=True)
class DiscreteDiffusionSchedule:
    mask_probs: t.Tensor
    mask_token_id: int

    @property
    def num_steps(self) -> int:
        return int(self.mask_probs.numel())


@dataclass(frozen=True)
class NoisingResult:
    noisy_tokens: t.Tensor
    mask: t.Tensor
    timesteps: t.Tensor


@dataclass(frozen=True)
class DenoisingStepStats:
    step: int
    mask_fraction: float
    mean_entropy: float
    committed_fraction: float


@dataclass(frozen=True)
class DiffusionGemmaReadinessReport:
    transformers_version: str | None
    config_supported: bool
    processor_supported: bool
    model_class_supported: bool
    config_model_type: str | None
    config_architectures: tuple[str, ...]
    tokenizer_mask_token_id: int | None
    canvas_length: int | None
    default_max_denoising_steps: int | None
    bf16_repo_id: str
    bf16_revision: str
    bf16_required_weight_shards: int
    bf16_local_weight_shards_present: int
    bf16_local_ready_for_direct_loading: bool
    bf16_remote_download_ready: bool
    bf16_weight_bytes_required: int | None
    nvfp4_repo_id: str
    nvfp4_revision: str
    nvfp4_required_weight_shards: int
    nvfp4_local_weight_shards_present: int
    nvfp4_local_ready_for_vllm: bool
    nvfp4_remote_download_ready: bool
    nvfp4_weight_bytes_required: int | None
    nvfp4_quant_method: str | None
    nvfp4_transformers_quantization_supported: bool
    nvfp4_transformers_quantization_error: str | None
    modelopt_available: bool
    modelopt_version: str | None
    vllm_available: bool
    vllm_version: str | None
    vllm_preserves_current_torch_cuda_stack: bool
    external_vllm_probe_path: str
    external_vllm_generation_ready: bool
    external_vllm_runtime_isolated: bool
    external_vllm_model_matches_nvfp4_revision: bool
    external_vllm_output_nonempty: bool
    external_vllm_output_mentions_negative_controls: bool
    external_vllm_used_chat_template: bool | None
    external_vllm_prompt: str | None
    external_vllm_output_preview: str | None
    external_vllm_torch_version: str | None
    external_vllm_torch_cuda_version: str | None
    external_vllm_vllm_version: str | None
    external_vllm_cuda_available: bool | None
    external_vllm_gpu_name: str | None
    external_vllm_gpu_total_memory_gib: float | None
    external_vllm_load_seconds: float | None
    external_vllm_generate_seconds: float | None
    external_vllm_error: str | None
    torch_version: str | None
    torch_cuda_version: str | None
    cuda_available: bool
    generation_ready: bool
    blockers: tuple[str, ...]


def linear_mask_schedule(
    num_steps: int,
    *,
    mask_token_id: int,
    min_mask_prob: float = 0.0,
    max_mask_prob: float = 1.0,
) -> DiscreteDiffusionSchedule:
    """Create a monotonic mask schedule indexed from low to high noise."""

    if num_steps <= 0:
        raise ValueError("num_steps must be positive.")
    if not 0 <= min_mask_prob <= max_mask_prob <= 1:
        raise ValueError("mask probabilities must satisfy 0 <= min <= max <= 1.")
    mask_probs = t.linspace(min_mask_prob, max_mask_prob, num_steps)
    return DiscreteDiffusionSchedule(mask_probs=mask_probs, mask_token_id=mask_token_id)


def cosine_mask_schedule(num_steps: int, *, mask_token_id: int) -> DiscreteDiffusionSchedule:
    """Cosine-style schedule with slow changes near the endpoints."""

    if num_steps <= 0:
        raise ValueError("num_steps must be positive.")
    x = t.linspace(0, 1, num_steps)
    mask_probs = 0.5 - 0.5 * t.cos(t.pi * x)
    return DiscreteDiffusionSchedule(mask_probs=mask_probs, mask_token_id=mask_token_id)


def apply_forward_noising(
    input_ids: t.Tensor,
    timesteps: t.Tensor,
    schedule: DiscreteDiffusionSchedule,
    *,
    generator: t.Generator | None = None,
) -> NoisingResult:
    """Mask tokens according to the schedule probability for each example."""

    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape (batch, seq).")
    if timesteps.shape != (input_ids.shape[0],):
        raise ValueError("timesteps must have shape (batch,).")
    if timesteps.min() < 0 or timesteps.max() >= schedule.num_steps:
        raise ValueError("timesteps are out of range for schedule.")

    probs = schedule.mask_probs.to(device=input_ids.device, dtype=t.float32)[timesteps]
    random_values = t.rand(input_ids.shape, generator=generator, device=input_ids.device)
    mask = random_values < probs[:, None]
    noisy = input_ids.clone()
    noisy[mask] = schedule.mask_token_id
    return NoisingResult(noisy_tokens=noisy, mask=mask, timesteps=timesteps)


def expected_mask_fraction(schedule: DiscreteDiffusionSchedule, timesteps: t.Tensor) -> float:
    probs = schedule.mask_probs.to(device=timesteps.device, dtype=t.float32)[timesteps]
    return probs.mean().item()


def masked_denoising_loss(logits: t.Tensor, target_ids: t.Tensor, mask: t.Tensor) -> t.Tensor:
    """Cross-entropy loss over masked positions only."""

    if logits.shape[:-1] != target_ids.shape or target_ids.shape != mask.shape:
        raise ValueError("logits, target_ids, and mask shapes are incompatible.")
    if not mask.any():
        raise ValueError("masked_denoising_loss requires at least one masked token.")
    return F.cross_entropy(logits[mask.bool()].float(), target_ids[mask.bool()].long())


def token_entropy(logits: t.Tensor) -> t.Tensor:
    """Per-token categorical entropy from logits."""

    log_probs = F.log_softmax(logits.float(), dim=-1)
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1)


def confidence_remask(
    logits: t.Tensor,
    current_tokens: t.Tensor,
    *,
    mask_token_id: int,
    next_mask_fraction: float,
) -> t.Tensor:
    """Fill tokens with argmax predictions, then remask low-confidence positions."""

    if not 0 <= next_mask_fraction <= 1:
        raise ValueError("next_mask_fraction must be in [0, 1].")
    probs = F.softmax(logits.float(), dim=-1)
    confidence, predictions = probs.max(dim=-1)
    new_tokens = predictions.to(dtype=current_tokens.dtype)
    _, seq_len = current_tokens.shape
    num_to_mask = int(round(next_mask_fraction * seq_len))
    if num_to_mask == 0:
        return new_tokens
    low_conf = confidence.topk(k=num_to_mask, dim=-1, largest=False).indices
    new_tokens.scatter_(1, low_conf, mask_token_id)
    return new_tokens


def uniform_remask(
    tokens: t.Tensor,
    *,
    mask_token_id: int,
    next_mask_fraction: float,
    generator: t.Generator | None = None,
) -> t.Tensor:
    """Randomly remask a target fraction of positions."""

    if not 0 <= next_mask_fraction <= 1:
        raise ValueError("next_mask_fraction must be in [0, 1].")
    num_to_mask = int(round(next_mask_fraction * tokens.shape[1]))
    if num_to_mask == 0:
        return tokens.clone()
    scores = t.rand(tokens.shape, generator=generator, device=tokens.device)
    chosen = scores.topk(k=num_to_mask, dim=-1, largest=False).indices
    remasked = tokens.clone()
    remasked.scatter_(1, chosen, mask_token_id)
    return remasked


def diffusion_sampler(
    model_fn,
    *,
    shape: tuple[int, int],
    schedule: DiscreteDiffusionSchedule,
    temperature: float = 0.0,
    remask: str = "confidence",
    generator: t.Generator | None = None,
    device: t.device | None = None,
) -> tuple[t.Tensor, list[DenoisingStepStats]]:
    """Simple iterative denoising sampler for toy diffusion LMs."""

    if device is None:
        device = schedule.mask_probs.device
    tokens = t.full(shape, schedule.mask_token_id, dtype=t.long, device=device)
    stats = []
    for step in reversed(range(schedule.num_steps)):
        logits = model_fn(tokens, step)
        if temperature == 0.0:
            predictions = logits.argmax(dim=-1)
        else:
            probs = F.softmax(logits.float() / temperature, dim=-1)
            samples = t.multinomial(probs.reshape(-1, probs.shape[-1]), 1, generator=generator)
            predictions = samples.reshape(tokens.shape)

        if step == 0:
            tokens = predictions.to(dtype=tokens.dtype)
        else:
            next_fraction = float(schedule.mask_probs[step - 1].item())
            if remask == "confidence":
                tokens = confidence_remask(
                    logits,
                    predictions.to(dtype=tokens.dtype),
                    mask_token_id=schedule.mask_token_id,
                    next_mask_fraction=next_fraction,
                )
            elif remask == "uniform":
                tokens = uniform_remask(
                    predictions.to(dtype=tokens.dtype),
                    mask_token_id=schedule.mask_token_id,
                    next_mask_fraction=next_fraction,
                    generator=generator,
                )
            else:
                raise ValueError("remask must be 'confidence' or 'uniform'.")

        mask_fraction = tokens.eq(schedule.mask_token_id).float().mean().item()
        stats.append(
            DenoisingStepStats(
                step=step,
                mask_fraction=mask_fraction,
                mean_entropy=token_entropy(logits).mean().item(),
                committed_fraction=1.0 - mask_fraction,
            )
        )
    return tokens, stats


def commitment_times(tokens_over_steps: t.Tensor, mask_token_id: int) -> t.Tensor:
    """Return first step index where each token is no longer masked."""

    if tokens_over_steps.ndim != 3:
        raise ValueError("tokens_over_steps must have shape (steps, batch, seq).")
    unmasked = tokens_over_steps.ne(mask_token_id)
    any_unmasked = unmasked.any(dim=0)
    first = unmasked.float().argmax(dim=0).long()
    return t.where(any_unmasked, first, t.full_like(first, -1))


def edit_distance(a: list[int], b: list[int]) -> int:
    """Levenshtein edit distance for token lists."""

    prev = list(range(len(b) + 1))
    for i, token_a in enumerate(a, start=1):
        cur = [i]
        for j, token_b in enumerate(b, start=1):
            cur.append(
                min(
                    prev[j] + 1,
                    cur[j - 1] + 1,
                    prev[j - 1] + int(token_a != token_b),
                )
            )
        prev = cur
    return prev[-1]


def validate_activation_trajectory(
    activations: list[t.Tensor],
    *,
    expected_steps: int,
    batch: int,
    seq_len: int,
) -> bool:
    """Check denoising-step activations have consistent leading dimensions."""

    if len(activations) != expected_steps:
        return False
    return all(act.shape[0] == batch and act.shape[1] == seq_len for act in activations)


def _safe_error_text(error: BaseException, *, limit: int = 300) -> str:
    return str(error).replace("\n", " ")[:limit]


def _package_version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _required_weight_shard_count(spec: HFGatedArtifactSpec) -> int:
    return sum(
        pattern.startswith("model-") and pattern.endswith(".safetensors")
        for pattern in spec.required_patterns
    )


def _local_weight_shard_count(spec: HFGatedArtifactSpec) -> int:
    cache_dir = hf_cache_repo_dir(spec.repo_id)
    if not cache_dir.exists():
        return 0
    required_shards = {
        pattern
        for pattern in spec.required_patterns
        if pattern.startswith("model-") and pattern.endswith(".safetensors")
    }
    return sum(
        1
        for path in cache_dir.rglob("model-*.safetensors")
        if path.name in required_shards and path.is_file()
    )


def _remote_weight_bytes(spec: HFGatedArtifactSpec, *, allow_network: bool) -> int | None:
    if not allow_network:
        return None
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(
            spec.repo_id,
            revision=spec.revision,
            files_metadata=True,
        )
    except Exception:
        return None
    total = 0
    saw_weight = False
    for sibling in info.siblings:
        if sibling.rfilename.startswith("model-") and sibling.rfilename.endswith(".safetensors"):
            total += int(getattr(sibling, "size", 0) or 0)
            saw_weight = True
    return total if saw_weight else None


def _diffusiongemma_transformers_metadata(
    *,
    repo_id: str,
    revision: str,
    local_files_only: bool,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "config_supported": False,
        "processor_supported": False,
        "model_class_supported": False,
        "config_model_type": None,
        "config_architectures": (),
        "tokenizer_mask_token_id": None,
        "canvas_length": None,
        "default_max_denoising_steps": None,
        "errors": [],
    }
    try:
        from transformers import AutoConfig, AutoProcessor
        from transformers.models.diffusion_gemma import DiffusionGemmaForBlockDiffusion

        report["model_class_supported"] = DiffusionGemmaForBlockDiffusion is not None
        config = AutoConfig.from_pretrained(
            repo_id,
            revision=revision,
            local_files_only=local_files_only,
            trust_remote_code=False,
        )
        report["config_supported"] = True
        report["config_model_type"] = getattr(config, "model_type", None)
        report["config_architectures"] = tuple(getattr(config, "architectures", ()) or ())
        report["canvas_length"] = getattr(config, "canvas_length", None)

        processor = AutoProcessor.from_pretrained(
            repo_id,
            revision=revision,
            local_files_only=local_files_only,
            trust_remote_code=False,
        )
        report["processor_supported"] = True
        tokenizer = getattr(processor, "tokenizer", None)
        report["tokenizer_mask_token_id"] = getattr(tokenizer, "mask_token_id", None)

        try:
            generation_config = DiffusionGemmaForBlockDiffusion.generation_config_class.from_pretrained(
                repo_id,
                revision=revision,
                local_files_only=local_files_only,
            )
            report["default_max_denoising_steps"] = getattr(
                generation_config,
                "max_denoising_steps",
                None,
            )
        except Exception as error:  # pragma: no cover - hub/local-state dependent
            report["errors"].append(f"generation_config: {_safe_error_text(error)}")
    except Exception as error:  # pragma: no cover - hub/local-state dependent
        report["errors"].append(_safe_error_text(error))
    return report


def _nvfp4_quantization_metadata(*, allow_network: bool) -> dict[str, Any]:
    report: dict[str, Any] = {
        "quant_method": None,
        "transformers_quantization_supported": False,
        "transformers_quantization_error": None,
    }
    try:
        from transformers import AutoConfig
        from transformers.quantizers.auto import AutoQuantizationConfig

        config = AutoConfig.from_pretrained(
            NVIDIA_DIFFUSIONGEMMA_26B_A4B_IT_NVFP4.repo_id,
            revision=NVIDIA_DIFFUSIONGEMMA_26B_A4B_IT_NVFP4.revision,
            local_files_only=not allow_network,
            trust_remote_code=False,
        )
        quantization_config = getattr(config, "quantization_config", None)
        if isinstance(quantization_config, dict):
            report["quant_method"] = quantization_config.get("quant_method")
            try:
                AutoQuantizationConfig.from_dict(quantization_config)
                report["transformers_quantization_supported"] = True
            except Exception as error:
                report["transformers_quantization_error"] = _safe_error_text(error)
    except Exception as error:  # pragma: no cover - hub/local-state dependent
        report["transformers_quantization_error"] = _safe_error_text(error)
    return report


def _external_vllm_probe_report(
    path: Path = DIFFUSIONGEMMA_EXTERNAL_VLLM_PROBE_PATH,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "path": str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
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
        "error": None,
    }
    if not path.exists():
        report["error"] = "external vLLM proof artifact is missing"
        return report

    try:
        data = json.loads(path.read_text())
    except Exception as error:  # pragma: no cover - malformed local artifact
        report["error"] = f"could not parse external vLLM proof artifact: {_safe_error_text(error)}"
        return report

    model_path = str(data.get("model", ""))
    output = str(data.get("output", "")).strip()
    output_lower = output.lower()
    torch_version = data.get("torch_version")
    torch_cuda_version = data.get("torch_cuda_version")
    vllm_version = data.get("vllm_version")
    cuda_available = data.get("cuda_available")
    gpu_total_memory_gib = data.get("gpu_total_memory_gib")
    load_seconds = data.get("load_seconds")
    generate_seconds = data.get("generate_seconds")
    model_path_exists = bool(model_path) and Path(model_path).exists()
    model_matches_revision = (
        NVIDIA_DIFFUSIONGEMMA_26B_A4B_IT_NVFP4.revision in model_path
        and "diffusiongemma-26B-A4B-it-NVFP4" in model_path
    )
    output_mentions_negative_controls = (
        "negative" in output_lower and "control" in output_lower
    )
    fits_24gb = (
        isinstance(gpu_total_memory_gib, int | float)
        and 20.0 <= float(gpu_total_memory_gib) <= 24.5
    )
    positive_timing = (
        isinstance(load_seconds, int | float)
        and float(load_seconds) > 0
        and isinstance(generate_seconds, int | float)
        and float(generate_seconds) > 0
    )
    runtime_isolated = (
        isinstance(torch_version, str)
        and torch_version.startswith("2.11.0")
        and torch_cuda_version == "13.0"
        and vllm_version == "0.24.0"
    )

    ready = bool(
        model_path_exists
        and model_matches_revision
        and output
        and output_mentions_negative_controls
        and data.get("used_chat_template") is True
        and cuda_available is True
        and runtime_isolated
        and fits_24gb
        and positive_timing
        and int(data.get("max_model_len", 0)) >= 4096
        and int(data.get("max_tokens", 0)) >= 32
        and int(data.get("canvas_length", 0)) == 256
    )

    report.update(
        {
            "ready": ready,
            "runtime_isolated": runtime_isolated,
            "model_matches_nvfp4_revision": model_path_exists and model_matches_revision,
            "output_nonempty": bool(output),
            "output_mentions_negative_controls": output_mentions_negative_controls,
            "used_chat_template": data.get("used_chat_template"),
            "prompt": data.get("prompt"),
            "output_preview": output[:240] if output else None,
            "torch_version": torch_version,
            "torch_cuda_version": torch_cuda_version,
            "vllm_version": vllm_version,
            "cuda_available": cuda_available,
            "gpu_name": data.get("gpu_name"),
            "gpu_total_memory_gib": (
                float(gpu_total_memory_gib)
                if isinstance(gpu_total_memory_gib, int | float)
                else None
            ),
            "load_seconds": float(load_seconds)
            if isinstance(load_seconds, int | float)
            else None,
            "generate_seconds": float(generate_seconds)
            if isinstance(generate_seconds, int | float)
            else None,
            "error": None if ready else "external vLLM proof artifact did not meet all checks",
        }
    )
    return report


def diffusiongemma_readiness_report(
    *,
    allow_network: bool = True,
) -> DiffusionGemmaReadinessReport:
    """Check the exact DiffusionGemma real-model path without fake fallbacks.

    This does not treat config/tokenizer support as inference. Generation is
    ready only when either the BF16 Transformers shards are locally complete, or
    the NVFP4 vLLM path is locally complete and a compatible real-generation
    proof is present. The current vLLM proof is intentionally isolated from the
    main torch 2.12/CUDA 13.2 uv environment.
    """

    bf16_access = hf_model_artifact_access_report(
        DIFFUSIONGEMMA_26B_A4B_IT,
        allow_network=allow_network,
    )
    nvfp4_access = hf_model_artifact_access_report(
        NVIDIA_DIFFUSIONGEMMA_26B_A4B_IT_NVFP4,
        allow_network=allow_network,
    )
    metadata_report = _diffusiongemma_transformers_metadata(
        repo_id=DIFFUSIONGEMMA_26B_A4B_IT.repo_id,
        revision=DIFFUSIONGEMMA_26B_A4B_IT.revision,
        local_files_only=not allow_network,
    )
    nvfp4_quantization = _nvfp4_quantization_metadata(allow_network=allow_network)

    vllm_available = find_spec("vllm") is not None
    vllm_version = _package_version("vllm") if vllm_available else None
    modelopt_available = find_spec("modelopt") is not None
    modelopt_version = _package_version("nvidia-modelopt") if modelopt_available else None
    torch_version = getattr(t, "__version__", None)
    torch_cuda_version = getattr(t.version, "cuda", None)
    cuda_available = bool(t.cuda.is_available())
    external_vllm_probe = _external_vllm_probe_report()

    # The current public vLLM 0.24.0 resolver wants torch 2.11/CUDA 13.0.
    # Do not let that silently downgrade this course's torch 2.12/cu132 env.
    vllm_preserves_current_torch_cuda_stack = (
        vllm_available
        and torch_version is not None
        and torch_version.startswith("2.12.")
        and torch_cuda_version == "13.2"
    )

    bf16_local_ready = bool(bf16_access["local_ready_for_direct_loading"])
    nvfp4_local_ready = bool(nvfp4_access["local_ready_for_direct_loading"])
    bf16_weight_bytes_required = _remote_weight_bytes(
        DIFFUSIONGEMMA_26B_A4B_IT,
        allow_network=allow_network,
    )
    if bf16_weight_bytes_required is None:
        bf16_weight_bytes_required = DIFFUSIONGEMMA_BF16_WEIGHT_BYTES_REQUIRED
    nvfp4_weight_bytes_required = _remote_weight_bytes(
        NVIDIA_DIFFUSIONGEMMA_26B_A4B_IT_NVFP4,
        allow_network=allow_network,
    )
    if nvfp4_weight_bytes_required is None:
        nvfp4_weight_bytes_required = DIFFUSIONGEMMA_NVFP4_WEIGHT_BYTES_REQUIRED
    external_vllm_generation_ready = bool(
        nvfp4_local_ready and external_vllm_probe["ready"]
    )
    nvfp4_generation_ready = bool(
        nvfp4_local_ready
        and (vllm_preserves_current_torch_cuda_stack or external_vllm_generation_ready)
    )
    generation_ready = bool(
        cuda_available
        and metadata_report["config_supported"]
        and metadata_report["processor_supported"]
        and metadata_report["model_class_supported"]
        and (bf16_local_ready or nvfp4_generation_ready)
    )

    blockers: list[str] = []
    if not cuda_available:
        blockers.append("CUDA is unavailable.")
    if not metadata_report["model_class_supported"]:
        blockers.append("Transformers lacks DiffusionGemmaForBlockDiffusion.")
    if not metadata_report["config_supported"]:
        blockers.append("Transformers cannot load the pinned DiffusionGemma config.")
    if not metadata_report["processor_supported"]:
        blockers.append("Transformers cannot load the pinned DiffusionGemma processor/tokenizer.")
    if not bf16_local_ready and not nvfp4_generation_ready:
        blockers.append(
            "Pinned Google BF16 DiffusionGemma weight shards are not complete locally; "
            "direct BF16 local loading is deferred for the 24GB tier."
        )
    if not nvfp4_local_ready:
        blockers.append("Pinned NVIDIA NVFP4 DiffusionGemma weight shards are not complete locally.")
    if (
        not nvfp4_quantization["transformers_quantization_supported"]
        and not external_vllm_generation_ready
        and not bf16_local_ready
    ):
        blockers.append(
            "Transformers does not currently support the pinned NVIDIA NVFP4 modelopt quantization config."
        )
    if nvfp4_local_ready and not nvfp4_generation_ready:
        blockers.append(
            "NVFP4 path needs a real vLLM or Transformers generation proof for the pinned checkpoint."
        )
    if nvfp4_local_ready and external_vllm_probe["error"] and not external_vllm_generation_ready:
        blockers.append(str(external_vllm_probe["error"]))
    if not generation_ready:
        blockers.append("No real DiffusionGemma generation path has been executed.")

    return DiffusionGemmaReadinessReport(
        transformers_version=_package_version("transformers"),
        config_supported=bool(metadata_report["config_supported"]),
        processor_supported=bool(metadata_report["processor_supported"]),
        model_class_supported=bool(metadata_report["model_class_supported"]),
        config_model_type=metadata_report["config_model_type"],
        config_architectures=tuple(metadata_report["config_architectures"]),
        tokenizer_mask_token_id=metadata_report["tokenizer_mask_token_id"],
        canvas_length=metadata_report["canvas_length"],
        default_max_denoising_steps=metadata_report["default_max_denoising_steps"],
        bf16_repo_id=DIFFUSIONGEMMA_26B_A4B_IT.repo_id,
        bf16_revision=DIFFUSIONGEMMA_26B_A4B_IT.revision,
        bf16_required_weight_shards=_required_weight_shard_count(DIFFUSIONGEMMA_26B_A4B_IT),
        bf16_local_weight_shards_present=_local_weight_shard_count(DIFFUSIONGEMMA_26B_A4B_IT),
        bf16_local_ready_for_direct_loading=bf16_local_ready,
        bf16_remote_download_ready=bool(bf16_access["remote_download_ready"]),
        bf16_weight_bytes_required=bf16_weight_bytes_required,
        nvfp4_repo_id=NVIDIA_DIFFUSIONGEMMA_26B_A4B_IT_NVFP4.repo_id,
        nvfp4_revision=NVIDIA_DIFFUSIONGEMMA_26B_A4B_IT_NVFP4.revision,
        nvfp4_required_weight_shards=_required_weight_shard_count(
            NVIDIA_DIFFUSIONGEMMA_26B_A4B_IT_NVFP4,
        ),
        nvfp4_local_weight_shards_present=_local_weight_shard_count(
            NVIDIA_DIFFUSIONGEMMA_26B_A4B_IT_NVFP4,
        ),
        nvfp4_local_ready_for_vllm=nvfp4_local_ready,
        nvfp4_remote_download_ready=bool(nvfp4_access["remote_download_ready"]),
        nvfp4_weight_bytes_required=nvfp4_weight_bytes_required,
        nvfp4_quant_method=nvfp4_quantization["quant_method"],
        nvfp4_transformers_quantization_supported=bool(
            nvfp4_quantization["transformers_quantization_supported"],
        ),
        nvfp4_transformers_quantization_error=nvfp4_quantization[
            "transformers_quantization_error"
        ],
        modelopt_available=modelopt_available,
        modelopt_version=modelopt_version,
        vllm_available=vllm_available,
        vllm_version=vllm_version,
        vllm_preserves_current_torch_cuda_stack=vllm_preserves_current_torch_cuda_stack,
        external_vllm_probe_path=str(external_vllm_probe["path"]),
        external_vllm_generation_ready=external_vllm_generation_ready,
        external_vllm_runtime_isolated=bool(external_vllm_probe["runtime_isolated"]),
        external_vllm_model_matches_nvfp4_revision=bool(
            external_vllm_probe["model_matches_nvfp4_revision"],
        ),
        external_vllm_output_nonempty=bool(external_vllm_probe["output_nonempty"]),
        external_vllm_output_mentions_negative_controls=bool(
            external_vllm_probe["output_mentions_negative_controls"],
        ),
        external_vllm_used_chat_template=external_vllm_probe["used_chat_template"],
        external_vllm_prompt=external_vllm_probe["prompt"],
        external_vllm_output_preview=external_vllm_probe["output_preview"],
        external_vllm_torch_version=external_vllm_probe["torch_version"],
        external_vllm_torch_cuda_version=external_vllm_probe["torch_cuda_version"],
        external_vllm_vllm_version=external_vllm_probe["vllm_version"],
        external_vllm_cuda_available=external_vllm_probe["cuda_available"],
        external_vllm_gpu_name=external_vllm_probe["gpu_name"],
        external_vllm_gpu_total_memory_gib=external_vllm_probe["gpu_total_memory_gib"],
        external_vllm_load_seconds=external_vllm_probe["load_seconds"],
        external_vllm_generate_seconds=external_vllm_probe["generate_seconds"],
        external_vllm_error=external_vllm_probe["error"],
        torch_version=torch_version,
        torch_cuda_version=torch_cuda_version,
        cuda_available=cuda_available,
        generation_ready=generation_ready,
        blockers=tuple(blockers),
    )


def diffusiongemma_readiness_dict(*, allow_network: bool = True) -> dict[str, Any]:
    report = diffusiongemma_readiness_report(allow_network=allow_network)
    result = {
        key: value
        for key, value in report.__dict__.items()
    }
    result["config_architectures"] = list(report.config_architectures)
    result["blockers"] = list(report.blockers)
    return result
