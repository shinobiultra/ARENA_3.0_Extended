# %%
"""Reference solutions for [16.4] TokenSHAP and TokenShapley."""

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
TOKENS = ("The", "capital", "is", "Paris")
MASK_TOKEN = "[MASK]"
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


# %%
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


def all_coalitions(num_players: int) -> tuple[Coalition, ...]:
    """Return every coalition for `num_players` ordered by size."""

    if num_players <= 0:
        raise ValueError("num_players must be positive.")
    players = range(num_players)
    coalitions: list[Coalition] = []
    for size in range(num_players + 1):
        coalitions.extend(frozenset(group) for group in itertools.combinations(players, size))
    return tuple(coalitions)


def normalize_coalition_values(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    *,
    num_players: int,
) -> dict[Coalition, float]:
    """Normalize coalition keys and require a complete value table."""

    values = {frozenset(key): float(value) for key, value in coalition_values.items()}
    expected = set(all_coalitions(num_players))
    missing = expected - set(values)
    if missing:
        raise ValueError(f"coalition value table is missing {len(missing)} coalitions.")
    return values


def exact_shapley_values(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    *,
    num_players: int,
) -> t.Tensor:
    """Compute exact Shapley values from a complete coalition-value table."""

    values = normalize_coalition_values(coalition_values, num_players=num_players)
    shapley = t.zeros(num_players, dtype=t.float64)
    denominator = math.factorial(num_players)
    for player in range(num_players):
        other_players = [index for index in range(num_players) if index != player]
        for coalition_size in range(num_players):
            weight = (
                math.factorial(coalition_size)
                * math.factorial(num_players - coalition_size - 1)
                / denominator
            )
            for group in itertools.combinations(other_players, coalition_size):
                coalition = frozenset(group)
                shapley[player] += weight * (
                    values[coalition | {player}] - values[coalition]
                )
    return shapley


def sampled_permutation_shapley_values(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    *,
    num_players: int,
    num_samples: int,
    seed: int = 0,
) -> t.Tensor:
    """Estimate Shapley values by averaging random player orderings."""

    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")
    values = normalize_coalition_values(coalition_values, num_players=num_players)
    rng = random.Random(seed)
    players = tuple(range(num_players))
    totals = t.zeros(num_players, dtype=t.float64)
    for _ in range(num_samples):
        coalition: Coalition = frozenset()
        for player in rng.sample(players, k=num_players):
            with_player = coalition | {player}
            totals[player] += values[with_player] - values[coalition]
            coalition = with_player
    return totals / num_samples


def keyword_interaction_token_score(
    tokens: Sequence[str],
    *,
    target_token: str = "Paris",
    context_token: str = "capital",
    target_weight: float = 1.0,
    interaction_weight: float = 2.0,
) -> float:
    """Toy prompt score with one target token and one context-target interaction."""

    token_set = set(tokens)
    target_present = target_token in token_set
    context_present = context_token in token_set
    score = target_weight if target_present else 0.0
    if target_present and context_present:
        score += interaction_weight
    return score


def token_coalition_values(
    tokens: Sequence[str],
    score_fn: Callable[[tuple[str, ...]], float],
    *,
    mask_token: str = MASK_TOKEN,
) -> dict[Coalition, float]:
    """Build a complete masked-position coalition table for token attribution."""

    token_tuple = tuple(tokens)
    if not token_tuple:
        raise ValueError("tokens must be nonempty.")
    values: dict[Coalition, float] = {}
    for coalition in all_coalitions(len(token_tuple)):
        masked_tokens = tuple(
            token if index in coalition else mask_token
            for index, token in enumerate(token_tuple)
        )
        values[coalition] = float(score_fn(masked_tokens))
    return values


def exact_token_shapley_values(
    tokens: Sequence[str],
    score_fn: Callable[[tuple[str, ...]], float],
    *,
    mask_token: str = MASK_TOKEN,
) -> t.Tensor:
    """Compute exact Shapley values over token positions."""

    token_tuple = tuple(tokens)
    values = token_coalition_values(token_tuple, score_fn, mask_token=mask_token)
    return exact_shapley_values(values, num_players=len(token_tuple))


def sampled_token_shapley_values(
    tokens: Sequence[str],
    score_fn: Callable[[tuple[str, ...]], float],
    *,
    mask_token: str = MASK_TOKEN,
    num_samples: int = 256,
    seed: int = 0,
) -> t.Tensor:
    """Estimate token-position Shapley values with sampled permutations."""

    token_tuple = tuple(tokens)
    values = token_coalition_values(token_tuple, score_fn, mask_token=mask_token)
    return sampled_permutation_shapley_values(
        values,
        num_players=len(token_tuple),
        num_samples=num_samples,
        seed=seed,
    )


def token_baseline_report(
    tokens: Sequence[str],
    score_fn: Callable[[tuple[str, ...]], float],
    *,
    mask_token: str = MASK_TOKEN,
    tolerance: float = 1e-9,
) -> TokenBaselineReport:
    """Check token Shapley efficiency against masked and full prompt scores."""

    token_tuple = tuple(tokens)
    baseline_tokens = tuple(mask_token for _ in token_tuple)
    exact = exact_token_shapley_values(token_tuple, score_fn, mask_token=mask_token)
    full_score = float(score_fn(token_tuple))
    baseline_score = float(score_fn(baseline_tokens))
    total_delta = full_score - baseline_score
    shapley_sum = float(exact.sum().item())
    efficiency_error = abs(shapley_sum - total_delta)
    return TokenBaselineReport(
        full_score=full_score,
        baseline_score=baseline_score,
        total_delta=total_delta,
        shapley_sum=shapley_sum,
        efficiency_error=efficiency_error,
        satisfies_efficiency=efficiency_error <= tolerance,
    )


def token_shapley_sampling_report(
    tokens: Sequence[str],
    score_fn: Callable[[tuple[str, ...]], float],
    *,
    mask_token: str = MASK_TOKEN,
    num_samples: int = 256,
    seed: int = 0,
    tolerance: float = 0.15,
) -> TokenShapleySamplingReport:
    """Compare sampled TokenSHAP values with exact token Shapley values."""

    token_tuple = tuple(tokens)
    exact = exact_token_shapley_values(token_tuple, score_fn, mask_token=mask_token)
    sampled = sampled_token_shapley_values(
        token_tuple,
        score_fn,
        mask_token=mask_token,
        num_samples=num_samples,
        seed=seed,
    )
    max_abs_error = float((exact - sampled).abs().max().item())
    top_index = int(exact.argmax().item())
    sampled_top_index = int(sampled.argmax().item())
    return TokenShapleySamplingReport(
        tokens=token_tuple,
        exact_values=exact,
        sampled_values=sampled,
        max_abs_error=max_abs_error,
        top_token=token_tuple[top_index],
        sampled_top_token=token_tuple[sampled_top_index],
        rank_matches=top_index == sampled_top_index,
        approximates_exact=max_abs_error <= tolerance,
    )


def _tensor_report(report) -> dict:
    result = report.__dict__.copy()
    for key, value in list(result.items()):
        if hasattr(value, "tolist"):
            result[key] = value.tolist()
    return result


def token_coalition_smoke_test() -> dict:
    values = token_coalition_values(TOKENS, keyword_interaction_token_score)
    return {
        "empty": values[frozenset()],
        "target_only": values[frozenset({3})],
        "context_and_target": values[frozenset({1, 3})],
    }


def exact_token_shapley_smoke_test() -> dict:
    exact = exact_token_shapley_values(TOKENS, keyword_interaction_token_score)
    return {
        "tokens": list(TOKENS),
        "exact_values": exact.tolist(),
        "baseline": token_baseline_report(
            TOKENS,
            keyword_interaction_token_score,
        ).__dict__,
    }


def sampled_tokenshap_smoke_test() -> dict:
    report = token_shapley_sampling_report(
        TOKENS,
        keyword_interaction_token_score,
        num_samples=512,
        seed=0,
        tolerance=0.1,
    )
    return _tensor_report(report)


def run_smoke_test(cpu: bool = True) -> dict:
    _ = cpu
    return {
        "coalitions": token_coalition_smoke_test(),
        "exact": exact_token_shapley_smoke_test(),
        "sampled": sampled_tokenshap_smoke_test(),
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
    rows = []
    targets = []
    for mask_index in range(2 ** len(TOKENS)):
        masked_tokens = [
            token if mask_index & (1 << token_index) else MASK_TOKEN
            for token_index, token in enumerate(TOKENS)
        ]
        rows.append([vocab[token] for token in masked_tokens])
        targets.append(keyword_interaction_token_score(masked_tokens))
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
        mse = float((residual**2).mean().item())
        max_abs_error = float(residual.abs().max().item())
    return model, mse, max_abs_error


def _model_score_fn(model: _TinyTokenScorer, vocab: dict[str, int]):
    def score(tokens: tuple[str, ...]) -> float:
        token_ids = t.tensor(
            [[vocab[token] for token in tokens]],
            device=next(model.parameters()).device,
            dtype=t.long,
        )
        with t.no_grad():
            return float(model(token_ids).item())

    return score


def run_neural_token_shapley_preflight(max_vram_gb: float = 24.0) -> dict:
    device = t.device("cuda")
    t.cuda.reset_peak_memory_stats()

    token_ids, targets, vocab = _token_training_table(device)
    model, fit_mse, fit_max_abs_error = _train_token_scorer(
        token_ids,
        targets,
        vocab_size=len(vocab),
    )
    model_score = _model_score_fn(model, vocab)
    exact_values = exact_token_shapley_values(TOKENS, model_score, mask_token=MASK_TOKEN)
    true_exact = exact_token_shapley_values(
        TOKENS,
        keyword_interaction_token_score,
        mask_token=MASK_TOKEN,
    )
    exact_error = float((exact_values - true_exact).abs().max().item())
    sampled_report = token_shapley_sampling_report(
        TOKENS,
        model_score,
        mask_token=MASK_TOKEN,
        num_samples=512,
        seed=0,
        tolerance=TOKEN_MODEL_SAMPLED_ERROR_MAX,
    )
    baseline = token_baseline_report(TOKENS, model_score, mask_token=MASK_TOKEN)

    generator = t.Generator(device=device)
    generator.manual_seed(TOKEN_MODEL_SEED)
    shuffled = targets[t.randperm(targets.shape[0], device=device, generator=generator)]
    shuffled_model, shuffled_fit_mse, _ = _train_token_scorer(
        token_ids,
        shuffled,
        vocab_size=len(vocab),
    )
    shuffled_exact = exact_token_shapley_values(
        TOKENS,
        _model_score_fn(shuffled_model, vocab),
        mask_token=MASK_TOKEN,
    )
    shuffled_error = float((shuffled_exact - true_exact).abs().max().item())

    t.cuda.synchronize()
    peak_vram_gb = t.cuda.max_memory_allocated() / 1024**3
    preflight_passed = (
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
        "preflight_passed": preflight_passed,
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
        "full_path": "Train a CUDA token-position scorer on every masked coalition, compute exact and sampled TokenSHAP from real model outputs, and reject shuffled-label token attributions.",
    }


def run_full_experiment(max_vram_gb: float = 24.0) -> dict:
    """Run the validated experiment path used by the verification report."""

    return run_gpu_test(max_vram_gb=max_vram_gb)


if MAIN:
    print(run_smoke_test())
