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
