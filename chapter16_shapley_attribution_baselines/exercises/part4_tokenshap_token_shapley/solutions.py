# %%
"""Reference solutions for [16.4] TokenSHAP and TokenShapley."""

from __future__ import annotations

import itertools
import math
import random
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch as t

chapter = "chapter16_shapley_attribution_baselines"
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

MAIN = __name__ == "__main__"
MASK_TOKEN = "[MASK]"
LAB_TOKENS = ("Please", "approve", "allow", "urgent", "transfer")
RANDOM_CONTROL_TOKENS = ("Please", "approve", "allow", "banana", "transfer")
SPLIT_TOKENS = ("Please", "approve", "allow", "urgent", "trans", "##fer")

# This separate four-token game preserves the existing release-only CUDA contract.
TOKENS = ("The", "capital", "is", "Paris")
TOKEN_MODEL_SEED = 1640
TOKEN_MODEL_EMBED_DIM = 32
TOKEN_MODEL_HIDDEN_DIM = 96
TOKEN_MODEL_STEPS = 1200
TOKEN_MODEL_LR = 2e-2
TOKEN_MODEL_FIT_MSE_MAX = 1e-10
TOKEN_MODEL_EXACT_SHAPLEY_ERROR_MAX = 1e-5
TOKEN_MODEL_SAMPLED_ERROR_MAX = 0.1
TOKEN_MODEL_SHUFFLED_ERROR_MIN = 1.0

Coalition = frozenset[int]


@dataclass(frozen=True)
class TokenShapleySamplingReport:
    tokens: tuple[str, ...]
    exact_values: t.Tensor
    sampled_values: t.Tensor
    max_abs_error: float
    top_token: str
    sampled_top_token: str
    rank_matches: bool
    approximates_exact: bool


@dataclass(frozen=True)
class TokenBaselineReport:
    full_score: float
    baseline_score: float
    total_delta: float
    shapley_sum: float
    efficiency_error: float
    satisfies_efficiency: bool


@dataclass(frozen=True)
class CorrelatedPairAudit:
    observed_max_abs_difference: float
    off_manifold_max_abs_difference: float
    redundancy_shapley: t.Tensor
    synergy_shapley: t.Tensor
    attribution_max_abs_difference: float
    identified_from_observed_support: bool


def all_coalitions(num_players: int) -> tuple[Coalition, ...]:
    """Enumerate the powerset in increasing coalition-size order."""
    if num_players <= 0:
        raise ValueError("num_players must be positive.")
    players = range(num_players)
    return tuple(
        frozenset(group)
        for size in range(num_players + 1)
        for group in itertools.combinations(players, size)
    )


def mask_tokens(
    tokens: Sequence[str],
    coalition: Coalition | Sequence[int],
    *,
    mask_token: str = MASK_TOKEN,
) -> tuple[str, ...]:
    """Keep coalition members and mask every other token position."""
    token_tuple = tuple(tokens)
    if not token_tuple:
        raise ValueError("tokens must be nonempty.")
    keep = frozenset(coalition)
    if not keep.issubset(range(len(token_tuple))):
        raise ValueError("coalition contains an out-of-range token position.")
    return tuple(token if index in keep else mask_token for index, token in enumerate(token_tuple))


def structured_token_score(
    tokens: Sequence[str],
    *,
    redundant_mode: str = "or",
) -> float:
    """Exact game with a necessary token, a redundant pair, and an interaction."""
    if redundant_mode not in {"or", "and"}:
        raise ValueError("redundant_mode must be 'or' or 'and'.")
    present = set(tokens)
    transfer = "transfer" in present
    approve = "approve" in present
    allow = "allow" in present
    urgent = "urgent" in present
    redundant_gate = (approve or allow) if redundant_mode == "or" else (approve and allow)
    return float(transfer * (4.0 + 2.0 * redundant_gate + 3.0 * urgent))


def split_token_score(tokens: Sequence[str], *, redundant_mode: str = "or") -> float:
    """Apply the same semantic rule when transfer is split into two required pieces."""
    present = set(tokens)
    rebuilt = list(tokens)
    if "trans" in present and "##fer" in present:
        rebuilt.append("transfer")
    return structured_token_score(rebuilt, redundant_mode=redundant_mode)


def token_coalition_values(
    tokens: Sequence[str],
    score_fn: Callable[[tuple[str, ...]], float],
    *,
    mask_token: str = MASK_TOKEN,
) -> dict[Coalition, float]:
    """Evaluate a score on every position-preserving masked coalition."""
    token_tuple = tuple(tokens)
    if not token_tuple:
        raise ValueError("tokens must be nonempty.")
    return {
        coalition: float(score_fn(mask_tokens(token_tuple, coalition, mask_token=mask_token)))
        for coalition in all_coalitions(len(token_tuple))
    }


def normalize_coalition_values(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    *,
    num_players: int,
) -> dict[Coalition, float]:
    """Normalize keys and require one value for every valid coalition."""
    values = {frozenset(key): float(value) for key, value in coalition_values.items()}
    expected = set(all_coalitions(num_players))
    missing = expected - set(values)
    extra = set(values) - expected
    if missing or extra:
        raise ValueError(
            f"coalition table must be complete: missing={len(missing)}, extra={len(extra)}."
        )
    return values


def exact_shapley_values(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    *,
    num_players: int,
) -> t.Tensor:
    """Compute exact Shapley values from a complete finite game."""
    values = normalize_coalition_values(coalition_values, num_players=num_players)
    result = t.zeros(num_players, dtype=t.float64)
    denominator = math.factorial(num_players)
    for player in range(num_players):
        others = [index for index in range(num_players) if index != player]
        for size in range(num_players):
            weight = math.factorial(size) * math.factorial(num_players - size - 1) / denominator
            for members in itertools.combinations(others, size):
                coalition = frozenset(members)
                result[player] += weight * (values[coalition | {player}] - values[coalition])
    return result


def exact_token_shapley_values(
    tokens: Sequence[str],
    score_fn: Callable[[tuple[str, ...]], float],
    *,
    mask_token: str = MASK_TOKEN,
) -> t.Tensor:
    token_tuple = tuple(tokens)
    values = token_coalition_values(token_tuple, score_fn, mask_token=mask_token)
    return exact_shapley_values(values, num_players=len(token_tuple))


def sampled_permutation_shapley_values(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    *,
    num_players: int,
    num_samples: int,
    seed: int = 0,
) -> t.Tensor:
    """Estimate Shapley values from marginal contributions along random orders."""
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")
    values = normalize_coalition_values(coalition_values, num_players=num_players)
    rng = random.Random(seed)
    totals = t.zeros(num_players, dtype=t.float64)
    players = tuple(range(num_players))
    for _ in range(num_samples):
        coalition: Coalition = frozenset()
        for player in rng.sample(players, k=num_players):
            with_player = coalition | {player}
            totals[player] += values[with_player] - values[coalition]
            coalition = with_player
    return totals / num_samples


def sampled_token_shapley_values(
    tokens: Sequence[str],
    score_fn: Callable[[tuple[str, ...]], float],
    *,
    mask_token: str = MASK_TOKEN,
    num_samples: int = 256,
    seed: int = 0,
) -> t.Tensor:
    token_tuple = tuple(tokens)
    values = token_coalition_values(token_tuple, score_fn, mask_token=mask_token)
    return sampled_permutation_shapley_values(
        values, num_players=len(token_tuple), num_samples=num_samples, seed=seed
    )


def shuffle_coalition_values_within_sizes(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    *,
    num_players: int,
    seed: int = 0,
) -> dict[Coalition, float]:
    """Shuffle values among equally sized masks while preserving both endpoints."""
    values = normalize_coalition_values(coalition_values, num_players=num_players)
    rng = random.Random(seed)
    shuffled: dict[Coalition, float] = {}
    for size in range(num_players + 1):
        coalitions = [coalition for coalition in all_coalitions(num_players) if len(coalition) == size]
        bucket = [values[coalition] for coalition in coalitions]
        rng.shuffle(bucket)
        shuffled.update(zip(coalitions, bucket, strict=True))
    return shuffled


def leave_one_out_values(
    tokens: Sequence[str],
    score_fn: Callable[[tuple[str, ...]], float],
    *,
    mask_token: str = MASK_TOKEN,
) -> t.Tensor:
    """Compute each token's local full-prompt deletion effect."""
    token_tuple = tuple(tokens)
    full_score = float(score_fn(token_tuple))
    result = []
    for player in range(len(token_tuple)):
        kept = frozenset(index for index in range(len(token_tuple)) if index != player)
        result.append(full_score - float(score_fn(mask_tokens(token_tuple, kept, mask_token=mask_token))))
    return t.tensor(result, dtype=t.float64)


def recency_position_control(total_delta: float, num_players: int) -> t.Tensor:
    """Allocate credit by position alone while still satisfying efficiency."""
    if num_players <= 0:
        raise ValueError("num_players must be positive.")
    weights = t.arange(1, num_players + 1, dtype=t.float64)
    return float(total_delta) * weights / weights.sum()


def discrete_pair_interaction(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    pair: tuple[int, int],
    *,
    context: Coalition | Sequence[int] = frozenset(),
    num_players: int,
) -> float:
    """Return a pair's finite second difference in one coalition context."""
    i, j = pair
    if i == j or not {i, j}.issubset(range(num_players)):
        raise ValueError("pair must contain two distinct valid players.")
    base = frozenset(context)
    if base & {i, j}:
        raise ValueError("context must exclude both members of the pair.")
    values = normalize_coalition_values(coalition_values, num_players=num_players)
    return values[base | {i, j}] - values[base | {i}] - values[base | {j}] + values[base]


def grouped_coalition_values(
    tokens: Sequence[str],
    groups: Sequence[Sequence[int]],
    score_fn: Callable[[tuple[str, ...]], float],
    *,
    mask_token: str = MASK_TOKEN,
) -> dict[Coalition, float]:
    """Treat declared groups of token positions as the Shapley players."""
    token_tuple = tuple(tokens)
    normalized_groups = tuple(frozenset(group) for group in groups)
    flattened = [position for group in normalized_groups for position in group]
    if sorted(flattened) != list(range(len(token_tuple))):
        raise ValueError("groups must partition every token position exactly once.")
    values: dict[Coalition, float] = {}
    for coalition in all_coalitions(len(normalized_groups)):
        kept = (
            frozenset().union(*(normalized_groups[index] for index in coalition))
            if coalition
            else frozenset()
        )
        values[coalition] = float(score_fn(mask_tokens(token_tuple, kept, mask_token=mask_token)))
    return values


def correlated_pair_audit(
    tokens: Sequence[str],
    redundancy_score_fn: Callable[[tuple[str, ...]], float],
    synergy_score_fn: Callable[[tuple[str, ...]], float],
    *,
    pair: tuple[int, int],
    mask_token: str = MASK_TOKEN,
) -> CorrelatedPairAudit:
    """Test whether correlated support identifies the off-manifold game."""
    token_tuple = tuple(tokens)
    redundancy_values = token_coalition_values(token_tuple, redundancy_score_fn, mask_token=mask_token)
    synergy_values = token_coalition_values(token_tuple, synergy_score_fn, mask_token=mask_token)
    i, j = pair
    observed = [c for c in all_coalitions(len(token_tuple)) if (i in c) == (j in c)]
    off_manifold = [c for c in all_coalitions(len(token_tuple)) if (i in c) != (j in c)]
    observed_difference = max(abs(redundancy_values[c] - synergy_values[c]) for c in observed)
    off_manifold_difference = max(abs(redundancy_values[c] - synergy_values[c]) for c in off_manifold)
    redundancy_shapley = exact_shapley_values(redundancy_values, num_players=len(token_tuple))
    synergy_shapley = exact_shapley_values(synergy_values, num_players=len(token_tuple))
    attribution_difference = float((redundancy_shapley - synergy_shapley).abs().max().item())
    return CorrelatedPairAudit(
        observed_difference,
        off_manifold_difference,
        redundancy_shapley,
        synergy_shapley,
        attribution_difference,
        observed_difference > 0.0 or attribution_difference == 0.0,
    )


def sampling_convergence(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    *,
    num_players: int,
    budgets: Sequence[int],
    seeds: Sequence[int],
    reference_values: t.Tensor | None = None,
) -> list[dict[str, float | int]]:
    """Summarize Monte Carlo maximum attribution error over seeds."""
    if not budgets or not seeds:
        raise ValueError("budgets and seeds must be nonempty.")
    exact = (
        exact_shapley_values(coalition_values, num_players=num_players)
        if reference_values is None
        else t.as_tensor(reference_values, dtype=t.float64)
    )
    if exact.shape != (num_players,):
        raise ValueError(f"reference_values must have shape ({num_players},).")
    rows = []
    for budget in budgets:
        errors = t.tensor(
            [
                float(
                    (
                        sampled_permutation_shapley_values(
                            coalition_values,
                            num_players=num_players,
                            num_samples=int(budget),
                            seed=int(seed),
                        )
                        - exact
                    ).abs().max().item()
                )
                for seed in seeds
            ],
            dtype=t.float64,
        )
        rows.append(
            {
                "budget": int(budget),
                "mean_max_abs_error": float(errors.mean().item()),
                "p90_max_abs_error": float(t.quantile(errors, 0.9).item()),
            }
        )
    return rows


def token_baseline_report(
    tokens: Sequence[str],
    score_fn: Callable[[tuple[str, ...]], float],
    *,
    mask_token: str = MASK_TOKEN,
    tolerance: float = 1e-9,
) -> TokenBaselineReport:
    token_tuple = tuple(tokens)
    exact = exact_token_shapley_values(token_tuple, score_fn, mask_token=mask_token)
    full_score = float(score_fn(token_tuple))
    baseline_score = float(score_fn(tuple(mask_token for _ in token_tuple)))
    total_delta = full_score - baseline_score
    shapley_sum = float(exact.sum().item())
    error = abs(shapley_sum - total_delta)
    return TokenBaselineReport(full_score, baseline_score, total_delta, shapley_sum, error, error <= tolerance)


def token_shapley_sampling_report(
    tokens: Sequence[str],
    score_fn: Callable[[tuple[str, ...]], float],
    *,
    mask_token: str = MASK_TOKEN,
    num_samples: int = 256,
    seed: int = 0,
    tolerance: float = 0.15,
) -> TokenShapleySamplingReport:
    token_tuple = tuple(tokens)
    exact = exact_token_shapley_values(token_tuple, score_fn, mask_token=mask_token)
    sampled = sampled_token_shapley_values(
        token_tuple, score_fn, mask_token=mask_token, num_samples=num_samples, seed=seed
    )
    max_error = float((exact - sampled).abs().max().item())
    top = int(exact.argmax().item())
    sampled_top = int(sampled.argmax().item())
    return TokenShapleySamplingReport(
        token_tuple, exact, sampled, max_error, token_tuple[top], token_tuple[sampled_top],
        top == sampled_top, max_error <= tolerance,
    )


# Release preflight: retained for the serialized course contract, not taught as
# the main result. It trains a finite scorer rather than claiming LLM validity.
def keyword_interaction_token_score(
    tokens: Sequence[str],
    *,
    target_token: str = "Paris",
    context_token: str = "capital",
    target_weight: float = 1.0,
    interaction_weight: float = 2.0,
) -> float:
    present = set(tokens)
    score = target_weight if target_token in present else 0.0
    if target_token in present and context_token in present:
        score += interaction_weight
    return float(score)


def _tensor_report(report) -> dict:
    result = report.__dict__.copy()
    for key, value in list(result.items()):
        if hasattr(value, "tolist"):
            result[key] = value.tolist()
    return result


def run_smoke_test(cpu: bool = True) -> dict:
    _ = cpu
    values = token_coalition_values(TOKENS, keyword_interaction_token_score)
    exact = exact_token_shapley_values(TOKENS, keyword_interaction_token_score)
    sampled = token_shapley_sampling_report(
        TOKENS, keyword_interaction_token_score, num_samples=512, seed=0, tolerance=0.1
    )
    return {
        "coalitions": {
            "empty": values[frozenset()],
            "target_only": values[frozenset({3})],
            "context_and_target": values[frozenset({1, 3})],
        },
        "exact": {
            "tokens": list(TOKENS),
            "exact_values": exact.tolist(),
            "baseline": token_baseline_report(TOKENS, keyword_interaction_token_score).__dict__,
        },
        "sampled": _tensor_report(sampled),
    }


def run_gpu_test(max_vram_gb: float = 24.0) -> dict:
    if not t.cuda.is_available():
        raise RuntimeError("16.4 GPU preflight requires CUDA; no CPU fallback is accepted.")
    return run_neural_token_shapley_preflight(max_vram_gb=max_vram_gb)


class _TinyTokenScorer(t.nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.embedding = t.nn.Embedding(vocab_size, TOKEN_MODEL_EMBED_DIM)
        self.net = t.nn.Sequential(
            t.nn.Linear(len(TOKENS) * TOKEN_MODEL_EMBED_DIM, TOKEN_MODEL_HIDDEN_DIM),
            t.nn.GELU(),
            t.nn.Linear(TOKEN_MODEL_HIDDEN_DIM, TOKEN_MODEL_HIDDEN_DIM),
            t.nn.GELU(),
            t.nn.Linear(TOKEN_MODEL_HIDDEN_DIM, 1),
        )

    def forward(self, token_ids: t.Tensor) -> t.Tensor:
        return self.net(self.embedding(token_ids).flatten(start_dim=1))


def _token_vocab() -> dict[str, int]:
    return {MASK_TOKEN: 0, **{token: index + 1 for index, token in enumerate(TOKENS)}}


def _token_training_table(device: t.device) -> tuple[t.Tensor, t.Tensor, dict[str, int]]:
    vocab = _token_vocab()
    rows, targets = [], []
    for coalition in all_coalitions(len(TOKENS)):
        masked = mask_tokens(TOKENS, coalition)
        rows.append([vocab[token] for token in masked])
        targets.append(keyword_interaction_token_score(masked))
    return (
        t.tensor(rows, device=device, dtype=t.long),
        t.tensor(targets, device=device, dtype=t.float32).unsqueeze(-1),
        vocab,
    )


def _train_token_scorer(
    token_ids: t.Tensor,
    targets: t.Tensor,
    *,
    vocab_size: int,
) -> tuple[_TinyTokenScorer, float, float]:
    t.manual_seed(TOKEN_MODEL_SEED)
    t.cuda.manual_seed_all(TOKEN_MODEL_SEED)
    model = _TinyTokenScorer(vocab_size=vocab_size).to(token_ids.device)
    optimizer = t.optim.AdamW(model.parameters(), lr=TOKEN_MODEL_LR, weight_decay=0.0)
    for _ in range(TOKEN_MODEL_STEPS):
        prediction = model(token_ids)
        loss = ((prediction - targets) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with t.no_grad():
        residual = model(token_ids) - targets
    return model, float((residual**2).mean().item()), float(residual.abs().max().item())


def _model_score_fn(model: _TinyTokenScorer, vocab: dict[str, int]):
    def score(tokens: tuple[str, ...]) -> float:
        ids = t.tensor(
            [[vocab[token] for token in tokens]],
            device=next(model.parameters()).device,
            dtype=t.long,
        )
        with t.no_grad():
            return float(model(ids).item())
    return score


def run_neural_token_shapley_preflight(max_vram_gb: float = 24.0) -> dict:
    device = t.device("cuda")
    t.cuda.reset_peak_memory_stats()
    token_ids, targets, vocab = _token_training_table(device)
    model, fit_mse, fit_max_abs_error = _train_token_scorer(token_ids, targets, vocab_size=len(vocab))
    model_score = _model_score_fn(model, vocab)
    exact_values = exact_token_shapley_values(TOKENS, model_score)
    true_exact = exact_token_shapley_values(TOKENS, keyword_interaction_token_score)
    exact_error = float((exact_values - true_exact).abs().max().item())
    sampled_report = token_shapley_sampling_report(
        TOKENS, model_score, num_samples=512, seed=0, tolerance=TOKEN_MODEL_SAMPLED_ERROR_MAX
    )
    baseline = token_baseline_report(TOKENS, model_score)

    generator = t.Generator(device=device).manual_seed(TOKEN_MODEL_SEED)
    shuffled = targets[t.randperm(targets.shape[0], device=device, generator=generator)]
    shuffled_model, shuffled_fit_mse, _ = _train_token_scorer(token_ids, shuffled, vocab_size=len(vocab))
    shuffled_exact = exact_token_shapley_values(TOKENS, _model_score_fn(shuffled_model, vocab))
    shuffled_error = float((shuffled_exact - true_exact).abs().max().item())
    t.cuda.synchronize()
    peak_vram_gb = t.cuda.max_memory_allocated() / 1024**3
    passed = (
        fit_mse <= TOKEN_MODEL_FIT_MSE_MAX
        and exact_error <= TOKEN_MODEL_EXACT_SHAPLEY_ERROR_MAX
        and sampled_report.approximates_exact
        and sampled_report.rank_matches
        and sampled_report.top_token == "Paris"
        and baseline.satisfies_efficiency
        and shuffled_error >= TOKEN_MODEL_SHUFFLED_ERROR_MIN
        and peak_vram_gb <= max_vram_gb
    )
    return {
        "preflight_passed": passed,
        "cuda_available": True,
        "device": t.cuda.get_device_name(0),
        "model_family": "cuda_trained_tiny_token_scorer_mlp",
        "token_count": len(TOKENS),
        "coalition_count": 2 ** len(TOKENS),
        "training_example_count": int(token_ids.shape[0]),
        "training_steps": TOKEN_MODEL_STEPS,
        "fit_mse": fit_mse,
        "fit_max_abs_error": fit_max_abs_error,
        "true_exact_values": true_exact.tolist(),
        "model_exact_values": exact_values.tolist(),
        "exact_shapley_max_abs_error": exact_error,
        "sampled_values": sampled_report.sampled_values.tolist(),
        "sampled_max_abs_error": sampled_report.max_abs_error,
        "sampled_rank_matches": sampled_report.rank_matches,
        "sampled_top_token": sampled_report.sampled_top_token,
        "top_token": sampled_report.top_token,
        "baseline_efficiency_error": baseline.efficiency_error,
        "satisfies_efficiency": baseline.satisfies_efficiency,
        "shuffled_control_fit_mse": shuffled_fit_mse,
        "shuffled_control_exact_values": shuffled_exact.tolist(),
        "shuffled_control_error": shuffled_error,
        "shuffled_control_rejected": shuffled_error >= TOKEN_MODEL_SHUFFLED_ERROR_MIN,
        "peak_vram_gb": peak_vram_gb,
        "within_vram_budget": peak_vram_gb <= max_vram_gb,
        "full_path": "Finite CUDA scorer preflight; the exact learner organism remains the main result.",
    }


def run_full_experiment(max_vram_gb: float = 24.0) -> dict:
    return run_gpu_test(max_vram_gb=max_vram_gb)


if MAIN:
    values = token_coalition_values(LAB_TOKENS, structured_token_score)
    print(exact_shapley_values(values, num_players=len(LAB_TOKENS)))
