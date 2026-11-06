"""Inspectable methods for the selected string/text progression (not yet released).

Finite absorbing diffusion uses p_k=k/T. Its exact variational training weight
for a uniformly sampled k in 1..T is T/k on masked-token cross entropy, averaged
over *all* eligible tokens. Ancestral carry-over and confidence remasking are
separate algorithms. These bodies will be learner exercises, not hidden helpers.
"""
from __future__ import annotations

from collections.abc import Callable

import torch as t
import torch.nn.functional as F


def mask_targets(tokens: t.Tensor, eligible: t.Tensor, probability: t.Tensor,
                 mask_id: int, *, generator: t.Generator | None = None):
    """Corrupt eligible positions only, without using target length as a mask."""
    if tokens.ndim != 2 or eligible.shape != tokens.shape or eligible.dtype != t.bool:
        raise ValueError("tokens and boolean eligibility must have shape (batch, positions)")
    if probability.shape != (tokens.shape[0],) or not ((0 <= probability) & (probability <= 1)).all():
        raise ValueError("one probability in [0,1] per example is required")
    if tokens[eligible].eq(mask_id).any():
        raise ValueError("clean targets cannot contain MASK")
    selected = eligible & (t.rand(tokens.shape, device=tokens.device, generator=generator) < probability[:, None])
    return tokens.masked_fill(selected, mask_id), selected


def finite_diffusion_loss(logits: t.Tensor, clean: t.Tensor, corrupted: t.Tensor,
                          eligible: t.Tensor, steps: t.Tensor, total_steps: int,
                          mask_id: int):
    """Unbiased single-time estimator of the finite absorbing variational bound.

    At masked x_k, q(x_{k-1}|x_k,x_0) reveals x_0 with probability 1/k.
    Unmasked states are copied exactly and incur zero KL. Averaging uniformly
    sampled k multiplies the conditional KL by T. The k=1 term is reconstruction;
    the all-mask prior at k=T matches exactly. Never divide by sampled mask count.
    """
    if logits.shape[:-1] != clean.shape or corrupted.shape != clean.shape or eligible.shape != clean.shape:
        raise ValueError("token, logit and eligibility shapes disagree")
    if eligible.dtype != t.bool or not eligible.any(dim=-1).all():
        raise ValueError("each example needs a nonempty boolean target region")
    if total_steps < 1 or steps.shape != (clean.shape[0],) or steps.dtype != t.long:
        raise ValueError("steps must be an integer per-example vector")
    if not ((steps >= 1) & (steps <= total_steps)).all():
        raise ValueError("sample k uniformly from 1..T")
    if clean[eligible].eq(mask_id).any() or not t.equal(clean[~eligible], corrupted[~eligible]):
        raise ValueError("clean targets exclude MASK and conditioning must be unchanged")
    if ((corrupted != clean) & (corrupted != mask_id)).any():
        raise ValueError("this objective assumes absorbing MASK corruption")
    clean_logits = logits.clone()
    clean_logits[..., mask_id] = -t.inf
    # Non-target context may itself use MASK as a literal conditioning symbol.
    safe_targets = clean.masked_fill(~eligible, 0)
    safe_targets = safe_targets.masked_fill(~eligible & safe_targets.eq(mask_id), (mask_id + 1) % logits.shape[-1])
    nll = -clean_logits.log_softmax(-1).gather(-1, safe_targets[..., None]).squeeze(-1)
    masked = eligible & corrupted.eq(mask_id)
    weighted = t.where(masked, nll, 0.) * (total_steps / steps.to(logits.dtype))[:, None]
    return (weighted.sum(-1) / eligible.sum(-1)).mean()


def clean_prediction_probs(logits: t.Tensor, mask_id: int, temperature: float):
    """Exclude MASK from the clean distribution; temperature zero is greedy."""
    if temperature < 0 or not 0 <= mask_id < logits.shape[-1]:
        raise ValueError("temperature must be nonnegative and MASK must be in the vocabulary")
    clean_logits = logits.clone()
    clean_logits[..., mask_id] = -t.inf
    if temperature == 0:
        return F.one_hot(clean_logits.argmax(-1), logits.shape[-1]).to(logits.dtype)
    return (clean_logits / temperature).softmax(-1)


def absorbing_reverse_probs(clean_probs: t.Tensor, current: t.Tensor,
                            p_t: float, p_s: float, mask_id: int):
    """Exact normalized reverse kernel: clean mass 1-p_s/p_t, MASK mass p_s/p_t."""
    if not 0 <= p_s <= p_t <= 1 or p_t == 0:
        raise ValueError("require 0 <= p_s <= p_t <= 1 and p_t > 0")
    if clean_probs.shape[:-1] != current.shape:
        raise ValueError("probability and token shapes disagree")
    reverse = clean_probs * (1 - p_s/p_t)
    reverse = reverse.clone()
    reverse[..., mask_id] = p_s/p_t
    point_mass = F.one_hot(current, clean_probs.shape[-1]).to(clean_probs.dtype)
    return t.where(current.eq(mask_id)[..., None], reverse, point_mass)


def categorical_tokens(probs: t.Tensor, *, generator: t.Generator | None = None):
    return t.multinomial(probs.flatten(0, -2), 1, generator=generator).reshape(probs.shape[:-1])


def entropy_from_probs(probs: t.Tensor):
    """0 log 0 = 0, including the excluded MASK entry and greedy distributions."""
    return -t.special.xlogy(probs, probs).sum(-1)


@t.no_grad()
def sample_text_diffusion(model_fn: Callable, conditioning: t.Tensor, eligible: t.Tensor,
                          mask_id: int, *, steps: int, sampler: str = "ancestral",
                          temperatures: tuple[float, float] = (1., 1.),
                          block_size: int | None = None, generator: t.Generator | None = None,
                          record_trace: bool = True):
    """Generate actual states; fixed earlier blocks and context are never rewritten.

    model_fn(tokens, noise_fraction) returns logits. The model is trained on
    arbitrary masked subsets. Block generation treats completed earlier blocks
    as conditioning; future blocks remain MASK. Confidence/uniform remasking
    fills masked positions and remasks a fixed count of the newly proposed
    tokens. Previously revealed tokens carry over: their prediction logits are
    not supervised by the masked-only objective. The fixed-count rules differ
    from the independent ancestral reverse probabilities.

    Confidence and recorded entropy use the model distribution at temperature
    one. Sampling temperature controls token draws, not the confidence scale;
    otherwise greedy sampling makes every selected probability one and destroys
    the confidence ordering as well as the uncertainty trace.
    """
    if steps < 1 or sampler not in {"ancestral", "confidence", "uniform"}:
        raise ValueError("choose positive steps and an explicit sampler")
    if eligible.shape != conditioning.shape or eligible.dtype != t.bool or conditioning.ndim != 2:
        raise ValueError("conditioning and boolean eligibility must be two-dimensional")
    if not t.equal(eligible, eligible[:1].expand_as(eligible)):
        raise ValueError("this teaching sampler uses the same fixed target slots for every row")
    positions = eligible[0].nonzero().flatten()
    if not len(positions) or min(temperatures) < 0:
        raise ValueError("need target slots and nonnegative temperatures")
    block_size = len(positions) if block_size is None else block_size
    if block_size < 1:
        raise ValueError("block size must be positive")
    tokens = conditioning.masked_fill(eligible, mask_id)
    states = [tokens.detach().clone()] if record_trace else []
    records = []
    for block, active in enumerate(positions.split(block_size)):
        for step in range(steps, 0, -1):
            p_t, p_s = step/steps, (step-1)/steps
            progress = (steps-step)/max(steps-1, 1)
            temperature = temperatures[0] + progress*(temperatures[1]-temperatures[0])
            logits = model_fn(tokens, p_t)
            probabilities = clean_prediction_probs(logits[:, active], mask_id, temperature)
            model_probabilities = clean_prediction_probs(logits[:, active], mask_id, 1.) if (sampler == "confidence" or record_trace) else None
            previous = tokens[:, active].clone()
            if sampler == "ancestral":
                reverse = absorbing_reverse_probs(probabilities, previous, p_t, p_s, mask_id)
                updated = categorical_tokens(reverse, generator=generator)
            else:
                proposed = categorical_tokens(probabilities, generator=generator)
                was_masked = previous.eq(mask_id)
                updated = t.where(was_masked, proposed, previous)
                if sampler == "confidence":
                    priority = model_probabilities.gather(-1, proposed[..., None]).squeeze(-1)
                else:
                    priority = t.rand(updated.shape, device=tokens.device, generator=generator)
                priority = priority.masked_fill(~was_masked, t.inf)
                count = int(len(active)*p_s)
                remasked = priority.argsort(dim=-1, stable=True)[:, :count]
                updated.scatter_(1, remasked, mask_id)
            tokens[:, active] = updated
            record = {"block": block, "p_t": p_t, "p_s": p_s, "temperature": temperature,
                      "model_calls": len(records)+1, "input_tokens": tokens.numel()}
            if record_trace:
                record.update(active_positions=active.detach().clone(),
                    entropy=entropy_from_probs(model_probabilities).detach().clone(),
                    masked_before=previous.eq(mask_id).detach().clone())
                states.append(tokens.detach().clone())
            records.append(record)
    return tokens, t.stack(states) if record_trace else None, records


def stable_token_commitment(states: t.Tensor, mask_id: int):
    """First state equal to the final token with no later revision; -1 if final MASK."""
    if states.ndim != 3 or not states.shape[0]:
        raise ValueError("states must have shape (steps, batch, positions)")
    different = states.ne(states[-1:])
    index = t.arange(states.shape[0], device=states.device)[:, None, None]
    last_difference = t.where(different, index, -1).amax(0)
    return t.where(states[-1].eq(mask_id), -1, last_difference+1)


def levenshtein_distance(left, right):
    """Unit insert/delete/substitute cost on the declared character or token units."""
    previous = list(range(len(right)+1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(current[-1]+1, previous[j]+1, previous[j-1]+(a != b)))
        previous = current
    return previous[-1]


def balanced_parentheses(text: str):
    """Strict grammar score; non-parenthesis characters and early closing fail."""
    balance = 0
    if not text:
        return False
    for char in text:
        if char not in "()":
            return False
        balance += 1 if char == "(" else -1
        if balance < 0:
            return False
    return balance == 0
