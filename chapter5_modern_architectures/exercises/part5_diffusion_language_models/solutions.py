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


def stable_commitment_times(
    tokens_over_steps: t.Tensor,
    target_tokens: t.Tensor,
    mask_token_id: int,
) -> t.Tensor:
    """Return the first step after which every later token stays correct and visible."""

    if tokens_over_steps.ndim != 3:
        raise ValueError("tokens_over_steps must have shape (steps, batch, seq).")
    if target_tokens.shape != tokens_over_steps.shape[1:]:
        raise ValueError("target_tokens must have shape (batch, seq).")
    correct_and_visible = tokens_over_steps.eq(target_tokens.unsqueeze(0)) & tokens_over_steps.ne(
        mask_token_id
    )
    stable = correct_and_visible.flip(0).cumprod(dim=0).bool().flip(0)
    ever_stable = stable.any(dim=0)
    first_stable = stable.float().argmax(dim=0).long()
    return t.where(ever_stable, first_stable, t.full_like(first_stable, -1))


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


def copy_pair_dataset(device: t.device) -> t.Tensor:
    """Enumerate the exact grammar ``[a, b] -> [a, b, a, a, b, b]``."""

    rows = []
    for first in range(10):
        for second in range(10):
            rows.append([first, second, first, first, second, second])
    return t.tensor(rows, dtype=t.long, device=device)


def conditional_suffix_noising(
    clean_tokens: t.Tensor,
    schedule: DiscreteDiffusionSchedule,
) -> tuple[t.Tensor, t.Tensor, t.Tensor]:
    """Noise only the predicted suffix while preserving the two-token condition."""

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


def train_tiny_diffusion_model(
    model: TinyConditionalDiffusionLM,
    train_tokens: t.Tensor,
    schedule: DiscreteDiffusionSchedule,
    *,
    steps: int = TINY_DIFFUSION_TRAIN_STEPS,
    learning_rate: float = 2e-3,
    seed: int = 5505,
    record_every: int = 25,
) -> dict[str, list[float] | list[int]]:
    """Train one conditional denoiser and retain a compact, plottable loss curve."""

    if steps <= 0:
        raise ValueError("steps must be positive.")
    if record_every <= 0:
        raise ValueError("record_every must be positive.")
    t.manual_seed(seed)
    if train_tokens.is_cuda:
        t.cuda.manual_seed_all(seed)
    model.train()
    optimizer = t.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    recorded_steps: list[int] = []
    losses: list[float] = []
    for step in range(steps):
        noisy, mask, timesteps = conditional_suffix_noising(train_tokens, schedule)
        logits = model(noisy, timesteps)
        loss = masked_denoising_loss(logits, train_tokens, mask)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % record_every == 0 or step + 1 == steps:
            recorded_steps.append(step + 1)
            losses.append(float(loss.detach().item()))
    model.eval()
    return {"steps": recorded_steps, "losses": losses}


def conditional_diffusion_sample(
    model: TinyConditionalDiffusionLM,
    prefixes_and_targets: t.Tensor,
    schedule: DiscreteDiffusionSchedule,
    *,
    remask: str = "confidence",
    generator: t.Generator | None = None,
) -> tuple[t.Tensor, t.Tensor, list[float], list[t.Tensor]]:
    """Denoise the suffix from all masks while keeping the condition fixed."""

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
            next_mask_fraction = float(schedule.mask_probs[step - 1].item())
            if remask == "confidence":
                remasked_suffix = confidence_remask(
                    logits[:, 2:],
                    predictions[:, 2:],
                    mask_token_id=schedule.mask_token_id,
                    next_mask_fraction=next_mask_fraction,
                )
            elif remask == "uniform":
                remasked_suffix = uniform_remask(
                    predictions[:, 2:],
                    mask_token_id=schedule.mask_token_id,
                    next_mask_fraction=next_mask_fraction,
                    generator=generator,
                )
            else:
                raise ValueError("remask must be 'confidence' or 'uniform'.")
            current = t.cat([prefixes_and_targets[:, :2], remasked_suffix], dim=1)
        trajectory.append(current.clone())
    return current, t.stack(trajectory), entropy_by_step, activations


def _evaluate_tiny_diffusion_model(
    model: TinyConditionalDiffusionLM,
    heldout_tokens: t.Tensor,
    schedule: DiscreteDiffusionSchedule,
) -> dict:
    """Evaluate the hardest corruption and the complete iterative sampler."""

    fully_masked = heldout_tokens.clone()
    fully_masked[:, 2:] = schedule.mask_token_id
    timesteps = t.full(
        (heldout_tokens.shape[0],),
        schedule.num_steps - 1,
        device=heldout_tokens.device,
        dtype=t.long,
    )
    suffix_mask = t.zeros_like(heldout_tokens, dtype=t.bool)
    suffix_mask[:, 2:] = True
    logits = model(fully_masked, timesteps)
    heldout_loss = masked_denoising_loss(logits, heldout_tokens, suffix_mask)
    heldout_predictions = logits.argmax(dim=-1)
    heldout_masked_accuracy = (
        heldout_predictions[suffix_mask].eq(heldout_tokens[suffix_mask]).float().mean().item()
    )
    sampled, trajectory, entropy_by_step, activations = conditional_diffusion_sample(
        model,
        heldout_tokens,
        schedule,
    )
    suffix_accuracy_by_step = (
        trajectory[:, :, 2:].eq(heldout_tokens[None, :, 2:]).float().mean(dim=(1, 2))
    )
    mask_fraction_by_step = trajectory[:, :, 2:].eq(schedule.mask_token_id).float().mean(dim=(1, 2))
    stable_commitment = stable_commitment_times(
        trajectory,
        heldout_tokens,
        schedule.mask_token_id,
    )
    return {
        "heldout_loss": float(heldout_loss.item()),
        "heldout_masked_accuracy": heldout_masked_accuracy,
        "sampler_suffix_token_accuracy": sampled[:, 2:]
        .eq(heldout_tokens[:, 2:])
        .float()
        .mean()
        .item(),
        "sampler_exact_match": sampled.eq(heldout_tokens).all(dim=1).float().mean().item(),
        "sampled": sampled,
        "trajectory": trajectory,
        "entropy_by_step": entropy_by_step,
        "suffix_accuracy_by_step": suffix_accuracy_by_step.tolist(),
        "mask_fraction_by_step": mask_fraction_by_step.tolist(),
        "stable_commitment_times": stable_commitment,
        "activations": activations,
        "activation_trajectory_shape_ok": validate_activation_trajectory(
            activations,
            expected_steps=schedule.num_steps,
            batch=heldout_tokens.shape[0],
            seq_len=heldout_tokens.shape[1],
        ),
        "entropy_max": float(token_entropy(logits).max().item()),
    }


def run_toy_diffusion_signature_result(max_vram_gb: float = 24.0) -> dict:
    """Train the real toy grammar and a separately trained shuffled-label control."""

    if not t.cuda.is_available():
        raise RuntimeError("CUDA is required for the 5.5 signature experiment.")

    device = t.device("cuda")
    t.cuda.reset_peak_memory_stats()
    t.manual_seed(5505)
    t.cuda.manual_seed_all(5505)
    all_data = copy_pair_dataset(device)
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
    main_curve = train_tiny_diffusion_model(model, train_tokens, schedule, seed=5505)

    t.manual_seed(5506)
    shuffled_tokens = train_tokens.clone()
    shuffled_tokens[:, 2:] = train_tokens[t.randperm(train_tokens.shape[0], device=device), 2:]
    shuffled_model = TinyConditionalDiffusionLM().to(device)
    shuffled_curve = train_tiny_diffusion_model(
        shuffled_model,
        shuffled_tokens,
        schedule,
        seed=5506,
    )

    with t.no_grad():
        main_eval = _evaluate_tiny_diffusion_model(model, heldout_tokens, schedule)
        shuffled_eval = _evaluate_tiny_diffusion_model(shuffled_model, heldout_tokens, schedule)

    t.cuda.synchronize()
    peak_vram_gb = t.cuda.max_memory_allocated() / 1024**3
    within_vram_budget = peak_vram_gb <= max_vram_gb
    preflight_passed = (
        main_curve["losses"][-1] <= 0.05
        and main_curve["losses"][-1] <= 0.05 * main_curve["losses"][0]
        and main_eval["heldout_loss"] <= 0.05
        and main_eval["heldout_masked_accuracy"] >= 0.95
        and main_eval["sampler_suffix_token_accuracy"] >= 0.95
        and main_eval["sampler_exact_match"] >= 0.95
        and shuffled_eval["sampler_suffix_token_accuracy"] <= 0.25
        and shuffled_eval["sampler_exact_match"] <= 0.10
        and main_eval["activation_trajectory_shape_ok"]
        and shuffled_eval["activation_trajectory_shape_ok"]
        and within_vram_budget
    )
    example_index = 0
    result = {
        "claim": (
            "A CUDA-trained discrete diffusion LM reconstructs the held-out copy-pair "
            "grammar while a separately trained shuffled-label denoiser fails."
        ),
        "cuda_available": True,
        "device": t.cuda.get_device_name(0),
        "torch_version": t.__version__,
        "cuda_version": t.version.cuda,
        "dataset": "copy_pair_conditional_suffix_grammar_v1",
        "train_example_count": int(train_tokens.shape[0]),
        "heldout_example_count": int(heldout_tokens.shape[0]),
        "training_steps": TINY_DIFFUSION_TRAIN_STEPS,
        "diffusion_timesteps": list(reversed(range(schedule.num_steps))),
        "schedule_mask_probs": schedule.mask_probs.tolist(),
        "main": {
            "loss_curve": main_curve,
            "heldout_loss": main_eval["heldout_loss"],
            "heldout_masked_accuracy": main_eval["heldout_masked_accuracy"],
            "sampler_suffix_token_accuracy": main_eval["sampler_suffix_token_accuracy"],
            "sampler_exact_match": main_eval["sampler_exact_match"],
            "entropy_by_step": main_eval["entropy_by_step"],
            "suffix_accuracy_by_step": main_eval["suffix_accuracy_by_step"],
            "mask_fraction_by_step": main_eval["mask_fraction_by_step"],
            "stable_commitment_mean_by_position": main_eval["stable_commitment_times"]
            .float()
            .mean(dim=0)
            .tolist(),
        },
        "shuffled_control": {
            "loss_curve": shuffled_curve,
            "heldout_loss": shuffled_eval["heldout_loss"],
            "heldout_masked_accuracy": shuffled_eval["heldout_masked_accuracy"],
            "sampler_suffix_token_accuracy": shuffled_eval["sampler_suffix_token_accuracy"],
            "sampler_exact_match": shuffled_eval["sampler_exact_match"],
            "entropy_by_step": shuffled_eval["entropy_by_step"],
            "suffix_accuracy_by_step": shuffled_eval["suffix_accuracy_by_step"],
            "mask_fraction_by_step": shuffled_eval["mask_fraction_by_step"],
        },
        "example": {
            "prefix": heldout_tokens[example_index, :2].tolist(),
            "target": heldout_tokens[example_index].tolist(),
            "main_output": main_eval["sampled"][example_index].tolist(),
            "main_trajectory": main_eval["trajectory"][:, example_index].tolist(),
            "shuffled_output": shuffled_eval["sampled"][example_index].tolist(),
            "shuffled_trajectory": shuffled_eval["trajectory"][:, example_index].tolist(),
        },
        "activation_trajectory_shape_ok": bool(
            main_eval["activation_trajectory_shape_ok"]
            and shuffled_eval["activation_trajectory_shape_ok"]
        ),
        "entropy_max": main_eval["entropy_max"],
        "peak_vram_gb": peak_vram_gb,
        "within_vram_budget": within_vram_budget,
        "preflight_passed": bool(preflight_passed),
    }
    del model, shuffled_model
    t.cuda.empty_cache()
    return result


def run_trained_tiny_diffusion_lm_preflight(max_vram_gb: float = 24.0) -> dict:
    result = run_toy_diffusion_signature_result(max_vram_gb=max_vram_gb)
    main = result["main"]
    shuffled = result["shuffled_control"]
    return {
        "cuda_available": result["cuda_available"],
        "device": result["device"],
        "model_family": "tiny_transformer_discrete_diffusion_lm",
        "dataset": result["dataset"],
        "train_example_count": result["train_example_count"],
        "heldout_example_count": result["heldout_example_count"],
        "vocab_size": TINY_DIFFUSION_VOCAB_SIZE,
        "mask_token_id": TINY_DIFFUSION_MASK_TOKEN_ID,
        "sequence_length": TINY_DIFFUSION_SEQ_LEN,
        "diffusion_steps": TINY_DIFFUSION_STEPS,
        "training_steps": result["training_steps"],
        "initial_denoising_loss": main["loss_curve"]["losses"][0],
        "final_train_denoising_loss": main["loss_curve"]["losses"][-1],
        "denoising_loss": main["heldout_loss"],
        "heldout_masked_accuracy": main["heldout_masked_accuracy"],
        "shuffled_label_accuracy": shuffled["sampler_suffix_token_accuracy"],
        "shuffled_sampler_exact_match": shuffled["sampler_exact_match"],
        "shuffled_control_train_loss": shuffled["loss_curve"]["losses"][-1],
        "shuffled_control_fails": shuffled["sampler_suffix_token_accuracy"] <= 0.25,
        "sampler_suffix_token_accuracy": main["sampler_suffix_token_accuracy"],
        "sampler_exact_match": main["sampler_exact_match"],
        "suffix_commitment_mean_step": sum(
            main["stable_commitment_mean_by_position"][2:]
        )
        / 4,
        "activation_trajectory_shape_ok": result["activation_trajectory_shape_ok"],
        "entropy_by_step": main["entropy_by_step"],
        "entropy_max": result["entropy_max"],
        "peak_vram_gb": result["peak_vram_gb"],
        "within_vram_budget": result["within_vram_budget"],
        "preflight_passed": result["preflight_passed"],
        "torch_version": result["torch_version"],
        "cuda_version": result["cuda_version"],
        "full_path": (
            "CUDA-trained tiny conditional discrete diffusion LM on a generated "
            "copy-pair grammar, with protected-prefix denoising, held-out masked "
            "accuracy, confidence-remasking sampler reconstruction, activation "
            "trajectory checks, and a separately trained shuffled-label control. "
            "This does not claim DiffusionGemma checkpoint parity."
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
