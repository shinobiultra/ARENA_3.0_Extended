# %%
"""Reference solutions for [5.5] Diffusion Language Models."""

import sys
from dataclasses import dataclass
from pathlib import Path

import torch as t
import torch.nn as nn
import torch.nn.functional as F

chapter = "chapter5_modern_architectures"
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from arena_ext.diffusion_lm import diffusiongemma_readiness_dict

MAIN = __name__ == "__main__"

TINY_DIFFUSION_VOCAB_SIZE = 11
TINY_DIFFUSION_MASK_TOKEN_ID = 10
TINY_DIFFUSION_SEQ_LEN = 6
TINY_DIFFUSION_STEPS = 6
TINY_DIFFUSION_TRAIN_STEPS = 1200


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


# %%
def linear_mask_schedule(
    num_steps: int,
    *,
    mask_token_id: int,
    min_mask_prob: float = 0.0,
    max_mask_prob: float = 1.0,
) -> DiscreteDiffusionSchedule:
    """Create a monotonic schedule from low noise to high noise."""

    if num_steps <= 0:
        raise ValueError("num_steps must be positive.")
    if not 0 <= min_mask_prob <= max_mask_prob <= 1:
        raise ValueError("mask probabilities must satisfy 0 <= min <= max <= 1.")
    mask_probs = t.linspace(min_mask_prob, max_mask_prob, num_steps)
    return DiscreteDiffusionSchedule(mask_probs=mask_probs, mask_token_id=mask_token_id)


def expected_mask_fraction(schedule: DiscreteDiffusionSchedule, timesteps: t.Tensor) -> float:
    probs = schedule.mask_probs.to(device=timesteps.device, dtype=t.float32)[timesteps]
    return probs.mean().item()


def apply_forward_noising(
    input_ids: t.Tensor,
    timesteps: t.Tensor,
    schedule: DiscreteDiffusionSchedule,
    *,
    generator: t.Generator | None = None,
) -> NoisingResult:
    """Mask tokens according to the schedule probability for each batch item."""

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


def masked_denoising_loss(logits: t.Tensor, target_ids: t.Tensor, mask: t.Tensor) -> t.Tensor:
    """Compute cross entropy over masked positions only."""

    if logits.shape[:-1] != target_ids.shape or target_ids.shape != mask.shape:
        raise ValueError("logits, target_ids, and mask shapes are incompatible.")
    if not mask.any():
        raise ValueError("masked_denoising_loss requires at least one masked token.")
    return F.cross_entropy(logits[mask.bool()].float(), target_ids[mask.bool()].long())


def token_entropy(logits: t.Tensor) -> t.Tensor:
    """Return per-token categorical entropy from logits."""

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
    """Fill with argmax predictions, then remask the least confident positions."""

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
    """Iteratively denoise an all-mask sequence using a toy model function."""

    if device is None:
        device = schedule.mask_probs.device
    tokens = t.full(shape, schedule.mask_token_id, dtype=t.long, device=device)
    stats: list[DenoisingStepStats] = []
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
    """Return the first trajectory index where each token is no longer masked."""

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
    """Check denoising-step activations have the expected leading dimensions."""

    if len(activations) != expected_steps:
        return False
    return all(act.shape[0] == batch and act.shape[1] == seq_len for act in activations)


def schedule_smoke_test() -> dict:
    schedule = linear_mask_schedule(5, mask_token_id=99)
    timesteps = t.tensor([0, 2, 4])
    return {
        "mask_probs": schedule.mask_probs.tolist(),
        "expected_mask_fraction": expected_mask_fraction(schedule, timesteps),
    }


def noising_smoke_test() -> dict:
    schedule = linear_mask_schedule(2, mask_token_id=99)
    input_ids = t.tensor([[1, 2, 3], [4, 5, 6]])
    result = apply_forward_noising(input_ids, t.tensor([0, 1]), schedule)
    return {
        "first_unchanged": bool(t.equal(result.noisy_tokens[0], input_ids[0])),
        "second_all_masked": bool(result.noisy_tokens[1].eq(99).all()),
        "mask_counts": result.mask.sum(dim=-1).tolist(),
    }


def denoising_loss_smoke_test() -> float:
    logits = t.zeros(1, 3, 4)
    target = t.tensor([[0, 1, 2]])
    mask = t.tensor([[True, False, True]])
    logits[0, 0, 0] = 5.0
    logits[0, 2, 2] = 5.0
    return float(masked_denoising_loss(logits, target, mask).item())


def remasking_smoke_test() -> dict:
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
    return {"num_masked": int(remasked.eq(99).sum().item()), "tokens": remasked.tolist()}


def oracle_sampler_smoke_test() -> dict:
    target = t.tensor([[1, 2, 3, 4]])
    schedule = linear_mask_schedule(4, mask_token_id=0)

    def oracle_model(tokens, step):
        _ = tokens, step
        logits = t.zeros(1, 4, 6)
        logits.scatter_(2, target.unsqueeze(-1), 10.0)
        return logits

    output, stats = diffusion_sampler(shape=(1, 4), schedule=schedule, model_fn=oracle_model)
    return {
        "matches_target": bool(t.equal(output, target)),
        "num_steps": len(stats),
        "final_mask_fraction": stats[-1].mask_fraction,
        "mean_entropy": [stat.mean_entropy for stat in stats],
    }


def trajectory_smoke_test() -> dict:
    trajectory = t.tensor(
        [
            [[0, 0, 0]],
            [[1, 0, 0]],
            [[1, 2, 0]],
            [[1, 2, 3]],
        ]
    )
    activations = [t.zeros(1, 3, 4) for _ in range(4)]
    return {
        "commitment_times": commitment_times(trajectory, mask_token_id=0).tolist(),
        "edit_distance": edit_distance([1, 2, 3], [1, 4, 3, 5]),
        "activation_shape_ok": validate_activation_trajectory(
            activations,
            expected_steps=4,
            batch=1,
            seq_len=3,
        ),
    }


def entropy_smoke_test() -> list[list[float]]:
    logits = t.tensor([[[10.0, -10.0], [0.0, 0.0]]])
    return token_entropy(logits).tolist()


def run_smoke_test(cpu: bool = True) -> dict:
    _ = cpu
    return {
        "schedule": schedule_smoke_test(),
        "noising": noising_smoke_test(),
        "denoising_loss": denoising_loss_smoke_test(),
        "remasking": remasking_smoke_test(),
        "oracle_sampler": oracle_sampler_smoke_test(),
        "trajectory": trajectory_smoke_test(),
        "entropy": entropy_smoke_test(),
    }


class TinyConditionalDiffusionLM(nn.Module):
    """Small bidirectional denoiser for a GT-0 conditional copy grammar."""

    def __init__(
        self,
        *,
        vocab_size: int = TINY_DIFFUSION_VOCAB_SIZE,
        seq_len: int = TINY_DIFFUSION_SEQ_LEN,
        num_steps: int = TINY_DIFFUSION_STEPS,
        d_model: int = 96,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.position_embed = nn.Embedding(seq_len, d_model)
        self.time_embed = nn.Embedding(num_steps, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=192,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.unembed = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: t.Tensor, timesteps: t.Tensor) -> t.Tensor:
        positions = t.arange(self.seq_len, device=input_ids.device)
        hidden = (
            self.token_embed(input_ids)
            + self.position_embed(positions)[None, :, :]
            + self.time_embed(timesteps)[:, None, :]
        )
        return self.unembed(self.transformer(hidden))


def _copy_pair_dataset(device: t.device) -> t.Tensor:
    rows = []
    for first in range(10):
        for second in range(10):
            rows.append([first, second, first, first, second, second])
    return t.tensor(rows, dtype=t.long, device=device)


def _conditional_suffix_noising(
    clean_tokens: t.Tensor,
    schedule,
) -> tuple[t.Tensor, t.Tensor, t.Tensor]:
    timesteps = t.randint(
        1,
        schedule.num_steps,
        (clean_tokens.shape[0],),
        device=clean_tokens.device,
    )
    noising = apply_forward_noising(clean_tokens, timesteps, schedule)
    noisy = noising.noisy_tokens.clone()
    mask = noising.mask.clone()
    noisy[:, :2] = clean_tokens[:, :2]
    mask[:, :2] = False

    no_suffix_mask = mask[:, 2:].sum(dim=1).eq(0)
    if no_suffix_mask.any():
        row_ids = no_suffix_mask.nonzero().flatten()
        suffix_cols = t.randint(
            2,
            clean_tokens.shape[1],
            (row_ids.numel(),),
            device=clean_tokens.device,
        )
        noisy[row_ids, suffix_cols] = schedule.mask_token_id
        mask[row_ids, suffix_cols] = True
    return noisy, mask, timesteps


def _conditional_diffusion_sample(
    model: TinyConditionalDiffusionLM,
    prefixes_and_targets: t.Tensor,
    schedule,
) -> tuple[t.Tensor, t.Tensor, list[float], list[t.Tensor]]:
    current = t.full_like(prefixes_and_targets, schedule.mask_token_id)
    current[:, :2] = prefixes_and_targets[:, :2]
    entropy_by_step = []
    trajectory = []
    activations = []
    for step in reversed(range(schedule.num_steps)):
        timesteps = t.full(
            (prefixes_and_targets.shape[0],),
            step,
            device=prefixes_and_targets.device,
            dtype=t.long,
        )
        hidden = (
            model.token_embed(current)
            + model.position_embed(t.arange(model.seq_len, device=current.device))[None, :, :]
            + model.time_embed(timesteps)[:, None, :]
        )
        denoised_hidden = model.transformer(hidden)
        logits = model.unembed(denoised_hidden)
        activations.append(denoised_hidden.detach())
        entropy_by_step.append(token_entropy(logits[:, 2:]).mean().item())
        predictions = logits.argmax(dim=-1)
        predictions[:, :2] = prefixes_and_targets[:, :2]
        if step == 0:
            current = predictions
        else:
            remasked_suffix = confidence_remask(
                logits[:, 2:],
                predictions[:, 2:],
                mask_token_id=schedule.mask_token_id,
                next_mask_fraction=float(schedule.mask_probs[step - 1].item()),
            )
            current = t.cat([prefixes_and_targets[:, :2], remasked_suffix], dim=1)
        trajectory.append(current.clone())
    return current, t.stack(trajectory), entropy_by_step, activations


def run_trained_tiny_diffusion_lm_preflight(max_vram_gb: float = 24.0) -> dict:
    if not t.cuda.is_available():
        raise RuntimeError("CUDA is required for the 5.5 trained diffusion LM preflight.")

    device = t.device("cuda")
    t.cuda.reset_peak_memory_stats()
    t.manual_seed(5505)
    all_data = _copy_pair_dataset(device)
    permutation = t.randperm(all_data.shape[0], device=device)
    train_tokens = all_data[permutation[:80]]
    heldout_tokens = all_data[permutation[80:]]
    schedule = linear_mask_schedule(
        TINY_DIFFUSION_STEPS,
        mask_token_id=TINY_DIFFUSION_MASK_TOKEN_ID,
        min_mask_prob=0.25,
        max_mask_prob=1.0,
    )

    model = TinyConditionalDiffusionLM().to(device)
    optimizer = t.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    initial_loss = None
    final_train_loss = None
    for _ in range(TINY_DIFFUSION_TRAIN_STEPS):
        noisy, mask, timesteps = _conditional_suffix_noising(train_tokens, schedule)
        logits = model(noisy, timesteps)
        loss = masked_denoising_loss(logits, train_tokens, mask)
        if initial_loss is None:
            initial_loss = float(loss.item())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_train_loss = float(loss.item())

    model.eval()
    with t.no_grad():
        noised_heldout, heldout_mask, heldout_timesteps = _conditional_suffix_noising(
            heldout_tokens,
            schedule,
        )
        heldout_logits = model(noised_heldout, heldout_timesteps)
        heldout_loss = masked_denoising_loss(heldout_logits, heldout_tokens, heldout_mask)
        heldout_predictions = heldout_logits.argmax(dim=-1)
        heldout_masked_accuracy = (
            heldout_predictions[heldout_mask].eq(heldout_tokens[heldout_mask]).float().mean().item()
        )
        shuffled_targets = heldout_tokens[t.randperm(heldout_tokens.shape[0], device=device)]
        shuffled_label_accuracy = (
            heldout_predictions[heldout_mask].eq(shuffled_targets[heldout_mask]).float().mean().item()
        )
        sampled, trajectory, entropy_by_step, activations = _conditional_diffusion_sample(
            model,
            heldout_tokens,
            schedule,
        )
        sampler_suffix_token_accuracy = (
            sampled[:, 2:].eq(heldout_tokens[:, 2:]).float().mean().item()
        )
        sampler_exact_match = sampled.eq(heldout_tokens).all(dim=1).float().mean().item()
        committed = commitment_times(trajectory, mask_token_id=TINY_DIFFUSION_MASK_TOKEN_ID)
        suffix_commitment_mean = committed[:, 2:].float().mean().item()
        trajectory_shape_ok = validate_activation_trajectory(
            activations,
            expected_steps=TINY_DIFFUSION_STEPS,
            batch=heldout_tokens.shape[0],
            seq_len=TINY_DIFFUSION_SEQ_LEN,
        )
        entropy = token_entropy(heldout_logits)

    t.cuda.synchronize()
    peak_vram_gb = t.cuda.max_memory_allocated() / 1024**3
    within_vram_budget = peak_vram_gb <= max_vram_gb
    preflight_passed = (
        final_train_loss is not None
        and initial_loss is not None
        and final_train_loss <= 0.05
        and final_train_loss <= 0.05 * initial_loss
        and float(heldout_loss.item()) <= 0.05
        and heldout_masked_accuracy >= 0.95
        and shuffled_label_accuracy <= 0.25
        and sampler_suffix_token_accuracy >= 0.95
        and sampler_exact_match >= 0.95
        and trajectory_shape_ok
        and within_vram_budget
    )
    del model
    t.cuda.empty_cache()
    return {
        "cuda_available": True,
        "device": t.cuda.get_device_name(0),
        "model_family": "tiny_transformer_discrete_diffusion_lm",
        "dataset": "copy_pair_conditional_suffix_grammar_v1",
        "train_example_count": int(train_tokens.shape[0]),
        "heldout_example_count": int(heldout_tokens.shape[0]),
        "vocab_size": TINY_DIFFUSION_VOCAB_SIZE,
        "mask_token_id": TINY_DIFFUSION_MASK_TOKEN_ID,
        "sequence_length": TINY_DIFFUSION_SEQ_LEN,
        "diffusion_steps": TINY_DIFFUSION_STEPS,
        "training_steps": TINY_DIFFUSION_TRAIN_STEPS,
        "initial_denoising_loss": initial_loss,
        "final_train_denoising_loss": final_train_loss,
        "denoising_loss": float(heldout_loss.item()),
        "heldout_masked_accuracy": heldout_masked_accuracy,
        "shuffled_label_accuracy": shuffled_label_accuracy,
        "shuffled_control_fails": shuffled_label_accuracy <= 0.25,
        "sampler_suffix_token_accuracy": sampler_suffix_token_accuracy,
        "sampler_exact_match": sampler_exact_match,
        "suffix_commitment_mean_step": suffix_commitment_mean,
        "activation_trajectory_shape_ok": trajectory_shape_ok,
        "entropy_by_step": entropy_by_step,
        "entropy_max": float(entropy.max().item()),
        "peak_vram_gb": peak_vram_gb,
        "within_vram_budget": within_vram_budget,
        "preflight_passed": preflight_passed,
        "full_path": (
            "CUDA-trained tiny conditional discrete diffusion LM on a generated "
            "copy-pair grammar, with protected-prefix denoising, held-out masked "
            "accuracy, confidence-remasking sampler reconstruction, activation "
            "trajectory checks, and shuffled-label controls. This does not claim "
            "DiffusionGemma checkpoint parity."
        ),
    }


def run_diffusiongemma_readiness_preflight(*, allow_network: bool = True) -> dict:
    """Check the released DiffusionGemma path without accepting config-only proof."""

    report = diffusiongemma_readiness_dict(allow_network=allow_network)
    bf16_weight_gib = (
        report["bf16_weight_bytes_required"] / 1024**3
        if report["bf16_weight_bytes_required"] is not None
        else None
    )
    proof_gpu_gib = report["external_vllm_gpu_total_memory_gib"]
    bf16_direct_loading_deferred = bool(
        not report["bf16_local_ready_for_direct_loading"]
        and bf16_weight_gib is not None
        and proof_gpu_gib is not None
        and bf16_weight_gib > proof_gpu_gib
    )
    return {
        "diffusiongemma_repo_id": report["bf16_repo_id"],
        "diffusiongemma_revision": report["bf16_revision"],
        "diffusiongemma_config_supported": report["config_supported"],
        "diffusiongemma_processor_supported": report["processor_supported"],
        "diffusiongemma_model_class_supported": report["model_class_supported"],
        "diffusiongemma_model_type": report["config_model_type"],
        "diffusiongemma_architectures": report["config_architectures"],
        "diffusiongemma_tokenizer_mask_token_id": report["tokenizer_mask_token_id"],
        "diffusiongemma_canvas_length": report["canvas_length"],
        "diffusiongemma_default_max_denoising_steps": report["default_max_denoising_steps"],
        "diffusiongemma_bf16_local_ready_for_direct_loading": report[
            "bf16_local_ready_for_direct_loading"
        ],
        "diffusiongemma_bf16_remote_download_ready": report["bf16_remote_download_ready"],
        "diffusiongemma_bf16_local_weight_shards_present": report[
            "bf16_local_weight_shards_present"
        ],
        "diffusiongemma_bf16_required_weight_shards": report["bf16_required_weight_shards"],
        "diffusiongemma_bf16_weight_bytes_required": report["bf16_weight_bytes_required"],
        "diffusiongemma_bf16_24gb_direct_loading_deferred": bf16_direct_loading_deferred,
        "diffusiongemma_nvfp4_repo_id": report["nvfp4_repo_id"],
        "diffusiongemma_nvfp4_revision": report["nvfp4_revision"],
        "diffusiongemma_nvfp4_local_ready_for_vllm": report["nvfp4_local_ready_for_vllm"],
        "diffusiongemma_nvfp4_remote_download_ready": report["nvfp4_remote_download_ready"],
        "diffusiongemma_nvfp4_local_weight_shards_present": report[
            "nvfp4_local_weight_shards_present"
        ],
        "diffusiongemma_nvfp4_required_weight_shards": report["nvfp4_required_weight_shards"],
        "diffusiongemma_nvfp4_weight_bytes_required": report["nvfp4_weight_bytes_required"],
        "diffusiongemma_nvfp4_quant_method": report["nvfp4_quant_method"],
        "diffusiongemma_nvfp4_transformers_quantization_supported": report[
            "nvfp4_transformers_quantization_supported"
        ],
        "diffusiongemma_nvfp4_transformers_quantization_error": report[
            "nvfp4_transformers_quantization_error"
        ],
        "diffusiongemma_modelopt_available": report["modelopt_available"],
        "diffusiongemma_modelopt_version": report["modelopt_version"],
        "diffusiongemma_vllm_available": report["vllm_available"],
        "diffusiongemma_vllm_version": report["vllm_version"],
        "diffusiongemma_vllm_preserves_current_torch_cuda_stack": report[
            "vllm_preserves_current_torch_cuda_stack"
        ],
        "diffusiongemma_main_uv_vllm_available": report["vllm_available"],
        "diffusiongemma_main_uv_vllm_generation_supported": report[
            "vllm_preserves_current_torch_cuda_stack"
        ],
        "diffusiongemma_released_checkpoint_generation_proven": report[
            "generation_ready"
        ],
        "diffusiongemma_nvfp4_isolated_vllm_generation_ready": report[
            "external_vllm_generation_ready"
        ],
        "diffusiongemma_vllm_probe_path": report["external_vllm_probe_path"],
        "diffusiongemma_vllm_probe_vllm_version": report[
            "external_vllm_vllm_version"
        ],
        "diffusiongemma_vllm_probe_torch_version": report[
            "external_vllm_torch_version"
        ],
        "diffusiongemma_vllm_probe_torch_cuda_version": report[
            "external_vllm_torch_cuda_version"
        ],
        "diffusiongemma_vllm_probe_gpu_name": report["external_vllm_gpu_name"],
        "diffusiongemma_vllm_probe_output_nonempty": report[
            "external_vllm_output_nonempty"
        ],
        "diffusiongemma_external_vllm_probe_path": report["external_vllm_probe_path"],
        "diffusiongemma_external_vllm_generation_ready": report[
            "external_vllm_generation_ready"
        ],
        "diffusiongemma_external_vllm_runtime_isolated": report[
            "external_vllm_runtime_isolated"
        ],
        "diffusiongemma_external_vllm_model_matches_nvfp4_revision": report[
            "external_vllm_model_matches_nvfp4_revision"
        ],
        "diffusiongemma_external_vllm_output_nonempty": report[
            "external_vllm_output_nonempty"
        ],
        "diffusiongemma_external_vllm_output_mentions_negative_controls": report[
            "external_vllm_output_mentions_negative_controls"
        ],
        "diffusiongemma_external_vllm_used_chat_template": report[
            "external_vllm_used_chat_template"
        ],
        "diffusiongemma_external_vllm_prompt": report["external_vllm_prompt"],
        "diffusiongemma_external_vllm_output_preview": report[
            "external_vllm_output_preview"
        ],
        "diffusiongemma_external_vllm_torch_version": report[
            "external_vllm_torch_version"
        ],
        "diffusiongemma_external_vllm_torch_cuda_version": report[
            "external_vllm_torch_cuda_version"
        ],
        "diffusiongemma_external_vllm_vllm_version": report[
            "external_vllm_vllm_version"
        ],
        "diffusiongemma_external_vllm_cuda_available": report[
            "external_vllm_cuda_available"
        ],
        "diffusiongemma_external_vllm_gpu_name": report["external_vllm_gpu_name"],
        "diffusiongemma_external_vllm_gpu_total_memory_gib": report[
            "external_vllm_gpu_total_memory_gib"
        ],
        "diffusiongemma_external_vllm_load_seconds": report[
            "external_vllm_load_seconds"
        ],
        "diffusiongemma_external_vllm_generate_seconds": report[
            "external_vllm_generate_seconds"
        ],
        "diffusiongemma_external_vllm_error": report["external_vllm_error"],
        "diffusiongemma_torch_version": report["torch_version"],
        "diffusiongemma_torch_cuda_version": report["torch_cuda_version"],
        "diffusiongemma_cuda_available": report["cuda_available"],
        "diffusiongemma_generation_ready": report["generation_ready"],
        "diffusiongemma_blockers": report["blockers"],
        "diffusiongemma_claim_scope": (
            "Pinned DiffusionGemma metadata and loader support are checked directly. "
            "A real released-checkpoint generation claim requires complete BF16 "
            "Transformers generation or a pinned NVFP4 generation proof. The current "
            "24GB path uses an isolated vLLM runtime because vLLM 0.24.0 does not "
            "preserve the main torch 2.12/CUDA 13.2 uv stack; config/tokenizer "
            "support alone is never accepted as generation evidence."
        ),
    }


def run_gpu_test(max_vram_gb: float = 24.0) -> dict:
    tiny_report = run_trained_tiny_diffusion_lm_preflight(max_vram_gb=max_vram_gb)
    return {
        **tiny_report,
        **run_diffusiongemma_readiness_preflight(allow_network=True),
    }


def run_full_experiment(max_vram_gb: float = 24.0) -> dict:
    """Run the validated experiment path used by the verification report."""

    return run_gpu_test(max_vram_gb=max_vram_gb)


if MAIN:
    print(run_smoke_test())
