# %%
"""Reference solutions for [16.6] SHAP vs Activation Patching."""

import itertools
import math
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import torch as t

chapter = "chapter16_shapley_attribution_baselines"
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

# Used only by the repository's supporting CUDA preflight. The learner-facing
# methods below are deliberately self-contained.
from arena_ext.shapley_attribution import (
    shapley_patching_comparison_report as arena_shapley_patching_comparison_report,
)
from arena_ext.shapley_neural_game import (
    NEURAL_GAME_NUM_PLAYERS,
    NEURAL_GAME_STEPS,
    binary_feature_table,
    coalition_table_from_model,
    train_neural_game,
    true_neural_game_scores,
)

MAIN = __name__ == "__main__"

TOKEN_LABELS = ("RED", "SQUARE", "BRIGHT", "FILLER")
HIDDEN_LABELS = (
    "red_direct",
    "square_direct",
    "bright_direct",
    "filler_direct",
    "red_x_square_gate",
)
DIRECT_WEIGHTS = (1.0, 0.6, 0.25, 0.0)
DEFAULT_INTERACTION_STRENGTH = 2.4
GATE_THRESHOLD = 1.0
SHUFFLE_PERMUTATION = (2, 0, 3, 1)
RANDOM_CONTROL_SEED = 1666
RANDOM_CONTROL_COUNT = 512
GPU_RANDOM_CONTROL_COUNT = 1024

ADDITIVE_LINEAR_SEED = 1660
ADDITIVE_LINEAR_STEPS = 1000
ADDITIVE_LINEAR_LR = 5e-2
ADDITIVE_AGREEMENT_MAX_ERROR = 1e-5
INTERACTION_DISAGREEMENT_MIN_ERROR = 1.0
INTERACTION_ABS_OVERCOUNT_MIN = 2.0
INTERACTION_FIT_MSE_MAX = 1e-8

Coalition = frozenset[int]


# %%
def encode_tokens(
    token_activations: t.Tensor,
    *,
    gate_threshold: float = GATE_THRESHOLD,
) -> t.Tensor:
    """Expose four direct channels and one genuine ReLU interaction unit."""

    if token_activations.shape[-1] != len(TOKEN_LABELS):
        raise ValueError(f"Expected {len(TOKEN_LABELS)} token locations.")
    gate = t.relu(
        token_activations[..., 0]
        + token_activations[..., 1]
        - gate_threshold
    )
    return t.cat((token_activations, gate.unsqueeze(-1)), dim=-1)


def score_from_hidden(
    hidden: t.Tensor,
    *,
    interaction_strength: float = DEFAULT_INTERACTION_STRENGTH,
) -> t.Tensor:
    """Read out the exact hidden state into a scalar MATCH logit."""

    if hidden.shape[-1] != len(HIDDEN_LABELS):
        raise ValueError(f"Expected {len(HIDDEN_LABELS)} hidden units.")
    weights = hidden.new_tensor((*DIRECT_WEIGHTS, interaction_strength))
    return hidden @ weights


def exact_model(
    token_activations: t.Tensor,
    *,
    interaction_strength: float = DEFAULT_INTERACTION_STRENGTH,
    gate_threshold: float = GATE_THRESHOLD,
    return_cache: bool = False,
) -> t.Tensor | tuple[t.Tensor, dict[str, t.Tensor]]:
    """Run the exact token-interaction organism and optionally expose its cache."""

    hidden = encode_tokens(token_activations, gate_threshold=gate_threshold)
    score = score_from_hidden(hidden, interaction_strength=interaction_strength)
    if not return_cache:
        return score
    return score, {
        "token_activations": token_activations.detach().clone(),
        "post_relu_hidden": hidden.detach().clone(),
    }


# %%
def all_coalitions(num_players: int) -> tuple[Coalition, ...]:
    """Return every coalition in size-then-lexicographic order."""

    if num_players <= 0:
        raise ValueError("num_players must be positive.")
    coalitions: list[Coalition] = []
    for size in range(num_players + 1):
        coalitions.extend(
            frozenset(group)
            for group in itertools.combinations(range(num_players), size)
        )
    return tuple(coalitions)


def coalition_value_table(
    clean_activation: t.Tensor,
    corrupt_activation: t.Tensor,
    value_fn: Callable[[t.Tensor], t.Tensor],
) -> dict[Coalition, float]:
    """Evaluate every clean/corrupt coalition at one named activation level."""

    if clean_activation.ndim != 1 or clean_activation.shape != corrupt_activation.shape:
        raise ValueError("clean and corrupt activations must be same-shape vectors.")
    values: dict[Coalition, float] = {}
    for coalition in all_coalitions(clean_activation.numel()):
        mixed = corrupt_activation.clone()
        if coalition:
            index = t.tensor(sorted(coalition), device=mixed.device)
            mixed[index] = clean_activation[index]
        value = value_fn(mixed)
        if value.numel() != 1:
            raise ValueError("value_fn must return one scalar.")
        values[coalition] = float(value.item())
    return values


def _normalize_coalition_values(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    *,
    num_players: int,
) -> dict[Coalition, float]:
    values = {frozenset(key): float(value) for key, value in coalition_values.items()}
    expected = set(all_coalitions(num_players))
    if set(values) != expected:
        missing = expected - set(values)
        extra = set(values) - expected
        raise ValueError(
            f"Coalition table must be complete (missing={len(missing)}, extra={len(extra)})."
        )
    return values


# %%
def exact_shapley_values(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    *,
    num_players: int,
) -> t.Tensor:
    """Compute exact Shapley values from all weighted marginal contributions."""

    values = _normalize_coalition_values(coalition_values, num_players=num_players)
    result = t.zeros(num_players, dtype=t.float64)
    denominator = math.factorial(num_players)
    for player in range(num_players):
        others = [candidate for candidate in range(num_players) if candidate != player]
        for size in range(num_players):
            weight = (
                math.factorial(size)
                * math.factorial(num_players - size - 1)
                / denominator
            )
            for group in itertools.combinations(others, size):
                coalition = frozenset(group)
                result[player] += weight * (
                    values[coalition | {player}] - values[coalition]
                )
    return result


# %%
def activation_patching_effects(
    clean_activation: t.Tensor,
    corrupt_activation: t.Tensor,
    value_fn: Callable[[t.Tensor], t.Tensor],
) -> t.Tensor:
    """Patch one corrupt activation into the clean run and measure score loss."""

    if clean_activation.ndim != 1 or clean_activation.shape != corrupt_activation.shape:
        raise ValueError("clean and corrupt activations must be same-shape vectors.")
    clean_value = value_fn(clean_activation)
    if clean_value.numel() != 1:
        raise ValueError("value_fn must return one scalar.")
    effects = []
    for location in range(clean_activation.numel()):
        patched = clean_activation.clone()
        patched[location] = corrupt_activation[location]
        effects.append(clean_value - value_fn(patched))
    return t.stack(effects).to(dtype=t.float64)


# %%
def align_named_attributions(
    reference_labels: Sequence[str],
    reference_values: t.Tensor,
    candidate_labels: Sequence[str],
    candidate_values: t.Tensor,
) -> tuple[tuple[str, ...], t.Tensor, t.Tensor]:
    """Align two attribution vectors by name, rejecting incomparable player sets."""

    if len(set(reference_labels)) != len(reference_labels):
        raise ValueError("reference labels must be unique.")
    if len(set(candidate_labels)) != len(candidate_labels):
        raise ValueError("candidate labels must be unique.")
    if reference_values.numel() != len(reference_labels):
        raise ValueError("reference values and labels have different lengths.")
    if candidate_values.numel() != len(candidate_labels):
        raise ValueError("candidate values and labels have different lengths.")
    if set(reference_labels) != set(candidate_labels):
        raise ValueError("Attributions describe different player sets; do not correlate them.")
    candidate_index = {label: index for index, label in enumerate(candidate_labels)}
    order = t.tensor(
        [candidate_index[label] for label in reference_labels],
        device=candidate_values.device,
    )
    return (
        tuple(reference_labels),
        reference_values.to(dtype=t.float64),
        candidate_values[order].to(dtype=t.float64),
    )


def attribution_comparison(
    reference_labels: Sequence[str],
    reference_values: t.Tensor,
    candidate_labels: Sequence[str],
    candidate_values: t.Tensor,
    *,
    output_delta: float,
) -> dict[str, float | str | bool]:
    """Compare only aligned attribution units and expose efficiency separately."""

    labels, reference, candidate = align_named_attributions(
        reference_labels,
        reference_values,
        candidate_labels,
        candidate_values,
    )
    cosine = t.nn.functional.cosine_similarity(reference, candidate, dim=0)
    return {
        "max_abs_error": float((reference - candidate).abs().max().item()),
        "mean_abs_error": float((reference - candidate).abs().mean().item()),
        "cosine_similarity": float(cosine.item()),
        "reference_efficiency_gap": float(reference.sum().item() - output_delta),
        "candidate_efficiency_gap": float(candidate.sum().item() - output_delta),
        "top_reference": labels[int(reference.argmax().item())],
        "top_candidate": labels[int(candidate.argmax().item())],
        "top_agrees": int(reference.argmax().item()) == int(candidate.argmax().item()),
    }


# %%
def shuffled_label_control(values: t.Tensor, permutation: Sequence[int]) -> t.Tensor:
    """Keep labels fixed while assigning them values from other locations."""

    if sorted(permutation) != list(range(values.numel())):
        raise ValueError("permutation must contain each value index exactly once.")
    index = t.tensor(permutation, device=values.device)
    return values[index]


def random_direction_patching_effects(
    clean_hidden: t.Tensor,
    readout_weights: t.Tensor,
    *,
    target_delta_norm: float,
    num_samples: int = RANDOM_CONTROL_COUNT,
    seed: int = RANDOM_CONTROL_SEED,
) -> t.Tensor:
    """Score matched-norm random hidden-state interventions."""

    if clean_hidden.ndim != 1 or clean_hidden.shape != readout_weights.shape:
        raise ValueError("clean_hidden and readout_weights must be same-shape vectors.")
    if num_samples <= 0 or target_delta_norm <= 0:
        raise ValueError("num_samples and target_delta_norm must be positive.")
    generator = t.Generator(device=clean_hidden.device).manual_seed(seed)
    directions = t.randn(
        num_samples,
        clean_hidden.numel(),
        generator=generator,
        device=clean_hidden.device,
        dtype=clean_hidden.dtype,
    )
    directions = directions / directions.norm(dim=1, keepdim=True)
    directions = directions * target_delta_norm
    patched = clean_hidden.unsqueeze(0) - directions
    clean_score = clean_hidden @ readout_weights
    return clean_score - patched @ readout_weights


# %%
def interaction_sweep(interaction_strengths: t.Tensor) -> dict[str, t.Tensor]:
    """Measure token disagreement and hidden-unit agreement across gate strengths."""

    strengths = interaction_strengths.to(dtype=t.float64)
    clean_tokens = t.ones(len(TOKEN_LABELS), dtype=t.float64)
    corrupt_tokens = t.zeros_like(clean_tokens)
    token_overcount = []
    token_max_error = []
    hidden_max_error = []

    for strength in strengths:
        alpha = float(strength.item())
        token_value_fn = lambda x, alpha=alpha: exact_model(
            x, interaction_strength=alpha
        )
        token_values = coalition_value_table(clean_tokens, corrupt_tokens, token_value_fn)
        token_shapley = exact_shapley_values(
            token_values, num_players=clean_tokens.numel()
        )
        token_patching = activation_patching_effects(
            clean_tokens, corrupt_tokens, token_value_fn
        )
        output_delta = token_values[frozenset(range(clean_tokens.numel()))] - token_values[
            frozenset()
        ]
        token_overcount.append(token_patching.sum() - output_delta)
        token_max_error.append((token_patching - token_shapley).abs().max())

        clean_hidden = encode_tokens(clean_tokens)
        corrupt_hidden = encode_tokens(corrupt_tokens)
        hidden_value_fn = lambda h, alpha=alpha: score_from_hidden(
            h, interaction_strength=alpha
        )
        hidden_values = coalition_value_table(clean_hidden, corrupt_hidden, hidden_value_fn)
        hidden_shapley = exact_shapley_values(
            hidden_values, num_players=clean_hidden.numel()
        )
        hidden_patching = activation_patching_effects(
            clean_hidden, corrupt_hidden, hidden_value_fn
        )
        hidden_max_error.append((hidden_patching - hidden_shapley).abs().max())

    return {
        "interaction_strength": strengths,
        "token_credit_overcount": t.stack(token_overcount),
        "token_max_abs_error": t.stack(token_max_error),
        "hidden_max_abs_error": t.stack(hidden_max_error),
    }


# %%
def run_smoke_test(
    cpu: bool = True,
    random_control_count: int = RANDOM_CONTROL_COUNT,
) -> dict:
    """Run the complete exact-organism claim and its three controls."""

    device = t.device("cpu" if cpu else "cuda")
    clean_tokens = t.ones(len(TOKEN_LABELS), dtype=t.float64, device=device)
    corrupt_tokens = t.zeros_like(clean_tokens)
    token_value_fn = lambda x: exact_model(x)

    token_values = coalition_value_table(clean_tokens, corrupt_tokens, token_value_fn)
    token_shapley = exact_shapley_values(token_values, num_players=len(TOKEN_LABELS)).to(device)
    token_patching = activation_patching_effects(
        clean_tokens, corrupt_tokens, token_value_fn
    )
    output_delta = token_values[frozenset(range(len(TOKEN_LABELS)))] - token_values[
        frozenset()
    ]
    token_comparison = attribution_comparison(
        TOKEN_LABELS,
        token_shapley,
        TOKEN_LABELS,
        token_patching,
        output_delta=output_delta,
    )

    _, cache = exact_model(clean_tokens, return_cache=True)
    _, corrupt_cache = exact_model(corrupt_tokens, return_cache=True)
    clean_hidden = cache["post_relu_hidden"]
    corrupt_hidden = corrupt_cache["post_relu_hidden"]
    hidden_value_fn = lambda h: score_from_hidden(h)
    hidden_values = coalition_value_table(clean_hidden, corrupt_hidden, hidden_value_fn)
    hidden_shapley = exact_shapley_values(hidden_values, num_players=len(HIDDEN_LABELS)).to(device)
    hidden_patching = activation_patching_effects(
        clean_hidden, corrupt_hidden, hidden_value_fn
    )
    hidden_comparison = attribution_comparison(
        HIDDEN_LABELS,
        hidden_shapley,
        HIDDEN_LABELS,
        hidden_patching,
        output_delta=output_delta,
    )

    shuffled = shuffled_label_control(token_patching, SHUFFLE_PERMUTATION)
    shuffled_comparison = attribution_comparison(
        TOKEN_LABELS,
        token_shapley,
        TOKEN_LABELS,
        shuffled,
        output_delta=output_delta,
    )
    readout_weights = clean_hidden.new_tensor(
        (*DIRECT_WEIGHTS, DEFAULT_INTERACTION_STRENGTH)
    )
    random_effects = random_direction_patching_effects(
        clean_hidden,
        readout_weights,
        target_delta_norm=float((clean_hidden - corrupt_hidden)[-1].abs().item()),
        num_samples=random_control_count,
    ).abs()
    target_gate_effect = float(hidden_patching[-1].item())
    random_p95 = float(t.quantile(random_effects, 0.95).item())
    target_percentile = float((random_effects < target_gate_effect).double().mean().item())

    claim_passed = (
        math.isclose(token_comparison["candidate_efficiency_gap"], 2.4, abs_tol=1e-12)
        and hidden_comparison["max_abs_error"] < 1e-12
        and float(token_patching[-1].item()) == 0.0
        and shuffled_comparison["cosine_similarity"] < 0.7
        and target_gate_effect > random_p95
    )
    return {
        "claim_passed": claim_passed,
        "clean_score": float(exact_model(clean_tokens).item()),
        "corrupt_score": float(exact_model(corrupt_tokens).item()),
        "output_delta": output_delta,
        "token_shapley": token_shapley.tolist(),
        "token_patching": token_patching.tolist(),
        "token_credit_overcount": token_comparison["candidate_efficiency_gap"],
        "token_max_abs_error": token_comparison["max_abs_error"],
        "token_cosine_similarity": token_comparison["cosine_similarity"],
        "token_top_agrees": token_comparison["top_agrees"],
        "hidden_shapley": hidden_shapley.tolist(),
        "hidden_patching": hidden_patching.tolist(),
        "hidden_max_abs_error": hidden_comparison["max_abs_error"],
        "wrong_location_effect": float(token_patching[-1].item()),
        "shuffled_label_cosine": shuffled_comparison["cosine_similarity"],
        "random_control_count": random_control_count,
        "random_direction_mean_abs_effect": float(random_effects.mean().item()),
        "random_direction_p95_abs_effect": random_p95,
        "target_gate_effect": target_gate_effect,
        "target_gate_random_percentile": target_percentile,
    }


# %%
class _AdditiveLinearModel(t.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = t.nn.Linear(NEURAL_GAME_NUM_PLAYERS, 1)

    def forward(self, inputs: t.Tensor) -> t.Tensor:
        return self.linear(inputs)


def _additive_targets(inputs: t.Tensor) -> t.Tensor:
    return (
        0.25
        + 1.2 * inputs[:, 0]
        - 0.7 * inputs[:, 1]
        + 1.6 * inputs[:, 2]
        + 0.9 * inputs[:, 3]
    ).unsqueeze(-1)


def _train_additive_linear_model(inputs: t.Tensor) -> tuple[_AdditiveLinearModel, float, float]:
    targets = _additive_targets(inputs)
    t.manual_seed(ADDITIVE_LINEAR_SEED)
    t.cuda.manual_seed_all(ADDITIVE_LINEAR_SEED)
    model = _AdditiveLinearModel().to(inputs.device)
    optimizer = t.optim.AdamW(model.parameters(), lr=ADDITIVE_LINEAR_LR, weight_decay=0.0)
    for _ in range(ADDITIVE_LINEAR_STEPS):
        prediction = model(inputs)
        loss = ((prediction - targets) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with t.no_grad():
        residual = model(inputs) - targets
        mse = float((residual**2).mean().item())
        max_abs_error = float(residual.abs().max().item())
    return model, mse, max_abs_error


def run_neural_shap_vs_patching_preflight(max_vram_gb: float = 24.0) -> dict:
    """Supporting CUDA evidence; the exact organism remains the teaching result."""

    device = t.device("cuda")
    t.cuda.reset_peak_memory_stats()
    inputs = binary_feature_table(device)
    additive_model, additive_fit_mse, additive_fit_max_abs_error = _train_additive_linear_model(inputs)
    additive_values = coalition_table_from_model(additive_model, device)

    interaction_targets = true_neural_game_scores(inputs)
    interaction_model = train_neural_game(inputs, interaction_targets)
    interaction_values = coalition_table_from_model(interaction_model.model, device)
    additive_report = arena_shapley_patching_comparison_report(
        additive_values,
        num_players=NEURAL_GAME_NUM_PLAYERS,
        tolerance=ADDITIVE_AGREEMENT_MAX_ERROR,
    )
    interaction_report = arena_shapley_patching_comparison_report(
        interaction_values,
        num_players=NEURAL_GAME_NUM_PLAYERS,
        tolerance=ADDITIVE_AGREEMENT_MAX_ERROR,
    )
    interaction_abs_overcount = float(
        interaction_report.patching_effects.abs().sum().item()
        - interaction_report.shapley_values.abs().sum().item()
    )

    exact_cuda = run_smoke_test(
        cpu=False,
        random_control_count=GPU_RANDOM_CONTROL_COUNT,
    )
    t.cuda.synchronize()
    peak_vram_gb = t.cuda.max_memory_allocated() / 1024**3
    preflight_passed = (
        exact_cuda["claim_passed"]
        and additive_report.agrees_with_shapley
        and additive_report.max_abs_error <= ADDITIVE_AGREEMENT_MAX_ERROR
        and interaction_model.fit_mse <= INTERACTION_FIT_MSE_MAX
        and not interaction_report.agrees_with_shapley
        and interaction_report.max_abs_error >= INTERACTION_DISAGREEMENT_MIN_ERROR
        and interaction_abs_overcount >= INTERACTION_ABS_OVERCOUNT_MIN
        and interaction_report.top_feature_agrees
        and peak_vram_gb <= max_vram_gb
    )
    return {
        "preflight_passed": preflight_passed,
        "cuda_available": True,
        "device": t.cuda.get_device_name(0),
        "exact_organism_claim_passed": exact_cuda["claim_passed"],
        "exact_token_credit_overcount": exact_cuda["token_credit_overcount"],
        "exact_hidden_max_abs_error": exact_cuda["hidden_max_abs_error"],
        "exact_wrong_location_effect": exact_cuda["wrong_location_effect"],
        "exact_shuffled_label_cosine": exact_cuda["shuffled_label_cosine"],
        "exact_random_control_count": exact_cuda["random_control_count"],
        "exact_random_direction_p95_abs_effect": exact_cuda[
            "random_direction_p95_abs_effect"
        ],
        "exact_target_gate_effect": exact_cuda["target_gate_effect"],
        "additive_model_family": "cuda_trained_linear_additive_model",
        "additive_training_example_count": int(inputs.shape[0]),
        "additive_training_steps": ADDITIVE_LINEAR_STEPS,
        "additive_fit_mse": additive_fit_mse,
        "additive_fit_max_abs_error": additive_fit_max_abs_error,
        "additive_shapley_values": additive_report.shapley_values.tolist(),
        "additive_patching_effects": additive_report.patching_effects.tolist(),
        "additive_max_abs_error": additive_report.max_abs_error,
        "additive_agrees_with_shapley": additive_report.agrees_with_shapley,
        "interaction_model_family": "cuda_trained_neural_coalition_game_mlp",
        "interaction_training_example_count": int(inputs.shape[0]),
        "interaction_training_steps": NEURAL_GAME_STEPS,
        "interaction_fit_mse": interaction_model.fit_mse,
        "interaction_fit_max_abs_error": interaction_model.fit_max_abs_error,
        "interaction_shapley_values": interaction_report.shapley_values.tolist(),
        "interaction_patching_effects": interaction_report.patching_effects.tolist(),
        "interaction_max_abs_error": interaction_report.max_abs_error,
        "interaction_agrees_with_shapley": interaction_report.agrees_with_shapley,
        "interaction_top_feature_agrees": interaction_report.top_feature_agrees,
        "interaction_abs_overcount": interaction_abs_overcount,
        "peak_vram_gb": peak_vram_gb,
        "within_vram_budget": peak_vram_gb <= max_vram_gb,
        "full_path": (
            "Run the exact token/hidden intervention organism on CUDA, then train additive "
            "and interaction model organisms and compare exact Shapley with patching."
        ),
    }


def run_gpu_test(max_vram_gb: float = 24.0) -> dict:
    if not t.cuda.is_available():
        raise RuntimeError("16.6 GPU preflight requires CUDA; no CPU fallback is accepted.")
    return run_neural_shap_vs_patching_preflight(max_vram_gb=max_vram_gb)


def run_full_experiment(max_vram_gb: float = 24.0) -> dict:
    return run_gpu_test(max_vram_gb=max_vram_gb)


if MAIN:
    print(run_smoke_test())
