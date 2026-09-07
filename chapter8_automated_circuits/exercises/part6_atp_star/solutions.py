"""Reference implementations for [8.6] AtP* from Scratch."""

from __future__ import annotations

import logging
import math
import os
import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch as t
from torch import Tensor


MAIN = __name__ == "__main__"


# %% Exact toy model organism
TOY_COMPONENTS = (
    "L0H0.Q@dest",
    "L0H0.K@signal",
    "L0H0.K@decoy",
    "L0H0.V@signal",
    "L0H0.V@decoy",
    "L0M0.cancel_source",
    "L1M0.linear_source",
    "L1M1.null_source",
)
TOY_COMPONENT_INDEX = {name: index for index, name in enumerate(TOY_COMPONENTS)}
TOY_QK_COMPONENTS = frozenset(TOY_COMPONENTS[:3])
TOY_GRAD_PATHS = ("attention", "cancellation", "linear")
TOY_CLEAN_VALUES = {
    "L0H0.Q@dest": 3.0,
    "L0H0.K@signal": 3.0,
    "L0H0.K@decoy": -3.0,
    "L0H0.V@signal": 2.0,
    "L0H0.V@decoy": -1.0,
    "L0M0.cancel_source": 1.0,
    "L1M0.linear_source": 1.0,
    "L1M1.null_source": 0.5,
}
TOY_CORRUPT_VALUES = {
    "L0H0.Q@dest": -3.0,
    "L0H0.K@signal": -3.0,
    "L0H0.K@decoy": 3.0,
    "L0H0.V@signal": 0.0,
    "L0H0.V@decoy": 0.0,
    "L0M0.cancel_source": 0.0,
    "L1M0.linear_source": 0.0,
    "L1M1.null_source": -0.5,
}


@dataclass(frozen=True)
class ToyRun:
    sites: Tensor
    activations: dict[str, Tensor]
    metric: Tensor


@dataclass(frozen=True)
class MethodEvaluation:
    method: str
    ranking: tuple[str, ...]
    spearman: float
    recovery_at_k: tuple[float, ...]
    faithfulness_at_k: tuple[float, ...]
    wall_time_ms: float
    estimated_tensor_bytes: int
    forward_passes: int
    backward_passes: int


class ToyAtPStarModel:
    """A differentiable attention/cancellation graph with declared patch sites."""

    components = TOY_COMPONENTS
    grad_paths = TOY_GRAD_PATHS

    def __call__(
        self,
        sites: Tensor,
        *,
        drop_grad_paths: Sequence[str] = (),
    ) -> ToyRun:
        if sites.shape[-1] != len(self.components):
            raise ValueError(
                f"Expected final dimension {len(self.components)}, got {sites.shape[-1]}."
            )
        unknown = set(drop_grad_paths) - set(self.grad_paths)
        if unknown:
            raise ValueError(f"Unknown gradient paths: {sorted(unknown)}")

        q, k_signal, k_decoy, v_signal, v_decoy, cancel_source, linear_source, null_source = (
            sites.unbind(dim=-1)
        )
        attention_logits = t.stack((q * k_signal, q * k_decoy), dim=-1)
        attention_pattern = attention_logits.softmax(dim=-1)
        attention_output = (
            attention_pattern[..., 0] * v_signal
            + attention_pattern[..., 1] * v_decoy
        )
        direct_output = cancel_source
        cancellation_input = cancel_source.clone()
        cancellation_output = -0.5 * cancellation_input.square()
        linear_output = 0.4 * linear_source

        metric_terms = {
            "attention": attention_output,
            "cancellation": cancellation_output,
            "linear": linear_output,
        }
        for path in drop_grad_paths:
            metric_terms[path] = metric_terms[path].detach()
        metric = (
            metric_terms["attention"]
            + direct_output
            + metric_terms["cancellation"]
            + metric_terms["linear"]
            + 0.0 * null_source
        )
        activations = {
            "attention_logits": attention_logits,
            "attention_pattern": attention_pattern,
            "attention_output": attention_output,
            "direct_output": direct_output,
            "cancellation_input": cancellation_input,
            "cancellation_output": cancellation_output,
            "linear_output": linear_output,
            "null_source": null_source,
        }
        return ToyRun(sites=sites, activations=activations, metric=metric)


def toy_site_tensor(
    values: Mapping[str, float],
    *,
    device: str | t.device = "cpu",
    requires_grad: bool = False,
) -> Tensor:
    """Convert a named component mapping into the model's fixed site order."""

    missing = set(TOY_COMPONENTS) - set(values)
    extra = set(values) - set(TOY_COMPONENTS)
    if missing or extra:
        raise ValueError(f"Bad component mapping: missing={sorted(missing)}, extra={sorted(extra)}")
    return t.tensor(
        [values[name] for name in TOY_COMPONENTS],
        dtype=t.float64,
        device=device,
        requires_grad=requires_grad,
    )


def cache_clean_and_corrupt(
    model: ToyAtPStarModel,
    clean_sites: Tensor,
    corrupt_sites: Tensor,
) -> tuple[ToyRun, ToyRun]:
    """Run both endpoints and retain every activation needed by the estimators."""

    if clean_sites.shape != corrupt_sites.shape:
        raise ValueError("Clean and corrupt site tensors must have identical shapes.")
    with t.no_grad():
        clean = model(clean_sites)
        corrupt = model(corrupt_sites)
    return clean, corrupt


def batched_exact_patch_effects(
    model: ToyAtPStarModel,
    clean_sites: Tensor,
    corrupt_sites: Tensor,
    *,
    component_batch_size: int = 4,
) -> dict[str, float]:
    """Exhaustively patch one site at a time, evaluating sites in vectorized batches."""

    if clean_sites.ndim != 1 or corrupt_sites.ndim != 1:
        raise ValueError("The exact toy oracle expects one clean/corrupt site vector.")
    if clean_sites.shape != corrupt_sites.shape:
        raise ValueError("Clean and corrupt site tensors must have identical shapes.")
    if component_batch_size <= 0:
        raise ValueError("component_batch_size must be positive.")

    with t.no_grad():
        clean_metric = float(model(clean_sites).metric)
        effects: dict[str, float] = {}
        for start in range(0, len(TOY_COMPONENTS), component_batch_size):
            indices = list(range(start, min(start + component_batch_size, len(TOY_COMPONENTS))))
            patched = clean_sites.repeat(len(indices), 1)
            rows = t.arange(len(indices), device=clean_sites.device)
            cols = t.tensor(indices, device=clean_sites.device)
            patched[rows, cols] = corrupt_sites[cols]
            patched_metrics = model(patched).metric
            for row, index in enumerate(indices):
                effects[TOY_COMPONENTS[index]] = float(patched_metrics[row] - clean_metric)
    return effects


def _toy_gradient_run(
    model: ToyAtPStarModel,
    clean_sites: Tensor,
    *,
    drop_grad_paths: Sequence[str] = (),
) -> tuple[ToyRun, Tensor]:
    sites = clean_sites.detach().clone().requires_grad_(True)
    run = model(sites, drop_grad_paths=drop_grad_paths)
    run.activations["attention_pattern"].retain_grad()
    run.activations["attention_logits"].retain_grad()
    run.activations["attention_output"].retain_grad()
    run.activations["direct_output"].retain_grad()
    run.activations["cancellation_input"].retain_grad()
    run.activations["linear_output"].retain_grad()
    run.metric.backward()
    if sites.grad is None:
        raise RuntimeError("The clean site gradient was not retained.")
    return run, sites.grad.detach().clone()


def vanilla_attribution_patching(
    model: ToyAtPStarModel,
    clean_sites: Tensor,
    corrupt_sites: Tensor,
) -> dict[str, float]:
    """Compute the first-order AtP estimate (corrupt-clean) dot grad at clean."""

    _, gradients = _toy_gradient_run(model, clean_sites)
    raw = (corrupt_sites - clean_sites) * gradients
    return {name: float(raw[index]) for index, name in enumerate(TOY_COMPONENTS)}


def qk_softmax_correction(
    run: ToyRun,
    clean_sites: Tensor,
    corrupt_sites: Tensor,
) -> dict[str, float]:
    """Recompute softmax after each Q/K patch, then linearize only downstream."""

    pattern = run.activations["attention_pattern"]
    pattern_grad = pattern.grad if pattern.is_leaf or pattern.retains_grad else None
    if pattern_grad is None:
        raise ValueError("attention_pattern.grad is required; call backward first.")
    if clean_sites.ndim != 1:
        raise ValueError("The toy QK correction expects one site vector.")

    q_i = TOY_COMPONENT_INDEX["L0H0.Q@dest"]
    ks_i = TOY_COMPONENT_INDEX["L0H0.K@signal"]
    kd_i = TOY_COMPONENT_INDEX["L0H0.K@decoy"]
    clean_q = clean_sites[q_i]
    clean_keys = clean_sites[[ks_i, kd_i]]
    clean_pattern = pattern.detach()

    query_pattern = (corrupt_sites[q_i] * clean_keys).softmax(dim=-1)
    signal_logits = t.stack((clean_q * corrupt_sites[ks_i], clean_q * clean_sites[kd_i]))
    decoy_logits = t.stack((clean_q * clean_sites[ks_i], clean_q * corrupt_sites[kd_i]))
    signal_pattern = signal_logits.softmax(dim=-1)
    decoy_pattern = decoy_logits.softmax(dim=-1)

    return {
        "L0H0.Q@dest": float(((query_pattern - clean_pattern) * pattern_grad).sum()),
        "L0H0.K@signal": float(((signal_pattern - clean_pattern) * pattern_grad).sum()),
        "L0H0.K@decoy": float(((decoy_pattern - clean_pattern) * pattern_grad).sum()),
    }


def atp_star_scores(
    model: ToyAtPStarModel,
    clean_sites: Tensor,
    corrupt_sites: Tensor,
    *,
    grad_paths: Sequence[str] = TOY_GRAD_PATHS,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Combine the paper's QK correction with layerwise GradDrop aggregation."""

    paths = tuple(grad_paths)
    if len(paths) < 2:
        raise ValueError("GradDrop needs at least two residual contribution paths.")
    if len(set(paths)) != len(paths):
        raise ValueError("GradDrop paths must be unique.")

    per_drop: dict[str, dict[str, float]] = {}
    for dropped in paths:
        run, gradients = _toy_gradient_run(model, clean_sites, drop_grad_paths=(dropped,))
        pattern = run.activations["attention_pattern"]
        if pattern.grad is None:
            pattern.grad = t.zeros_like(pattern)
        raw = (corrupt_sites - clean_sites) * gradients
        estimates = {
            name: float(raw[index]) for index, name in enumerate(TOY_COMPONENTS)
        }
        estimates.update(qk_softmax_correction(run, clean_sites, corrupt_sites))
        per_drop[dropped] = estimates

    denominator = len(paths) - 1
    scores = {
        name: sum(abs(per_drop[path][name]) for path in paths) / denominator
        for name in TOY_COMPONENTS
    }
    return scores, per_drop


def eap_style_edge_scores(
    model: ToyAtPStarModel,
    clean_sites: Tensor,
    corrupt_sites: Tensor,
) -> tuple[dict[str, float], dict[str, float]]:
    """Score immediate graph edges, then aggregate outgoing edge magnitudes by site."""

    run, _ = _toy_gradient_run(model, clean_sites)
    activations = run.activations
    logits_grad = activations["attention_logits"].grad
    attention_output_grad = activations["attention_output"].grad
    direct_grad = activations["direct_output"].grad
    cancellation_grad = activations["cancellation_input"].grad
    linear_grad = activations["linear_output"].grad
    if any(value is None for value in (logits_grad, attention_output_grad, direct_grad, cancellation_grad, linear_grad)):
        raise RuntimeError("Required edge receiver gradients were not retained.")

    q_i, ks_i, kd_i, vs_i, vd_i, cancel_i, linear_i, _ = range(len(TOY_COMPONENTS))
    delta = corrupt_sites - clean_sites
    pattern = activations["attention_pattern"].detach()
    raw_edges = {
        "Q -> signal_logit": float(delta[q_i] * clean_sites[ks_i] * logits_grad[0]),
        "Q -> decoy_logit": float(delta[q_i] * clean_sites[kd_i] * logits_grad[1]),
        "K_signal -> signal_logit": float(clean_sites[q_i] * delta[ks_i] * logits_grad[0]),
        "K_decoy -> decoy_logit": float(clean_sites[q_i] * delta[kd_i] * logits_grad[1]),
        "V_signal -> attention_output": float(delta[vs_i] * pattern[0] * attention_output_grad),
        "V_decoy -> attention_output": float(delta[vd_i] * pattern[1] * attention_output_grad),
        "cancel_source -> direct_output": float(delta[cancel_i] * direct_grad),
        "cancel_source -> cancellation_output": float(delta[cancel_i] * cancellation_grad),
        "linear_source -> linear_output": float(0.4 * delta[linear_i] * linear_grad),
        "null_source -> metric": 0.0,
    }
    by_component = {
        "L0H0.Q@dest": abs(raw_edges["Q -> signal_logit"]) + abs(raw_edges["Q -> decoy_logit"]),
        "L0H0.K@signal": abs(raw_edges["K_signal -> signal_logit"]),
        "L0H0.K@decoy": abs(raw_edges["K_decoy -> decoy_logit"]),
        "L0H0.V@signal": abs(raw_edges["V_signal -> attention_output"]),
        "L0H0.V@decoy": abs(raw_edges["V_decoy -> attention_output"]),
        "L0M0.cancel_source": abs(raw_edges["cancel_source -> direct_output"])
        + abs(raw_edges["cancel_source -> cancellation_output"]),
        "L1M0.linear_source": abs(raw_edges["linear_source -> linear_output"]),
        "L1M1.null_source": 0.0,
    }
    return raw_edges, by_component


def rank_components(scores: Mapping[str, float]) -> tuple[str, ...]:
    """Rank named components by descending magnitude with a stable name tie-break."""

    if set(scores) != set(TOY_COMPONENTS):
        raise ValueError("Scores must contain every toy component exactly once.")
    if not all(math.isfinite(float(value)) for value in scores.values()):
        raise ValueError("Scores must be finite.")
    return tuple(sorted(TOY_COMPONENTS, key=lambda name: (-abs(float(scores[name])), name)))


def _average_ranks(values: Sequence[float]) -> Tensor:
    values_tensor = t.tensor(values, dtype=t.float64)
    result = t.empty_like(values_tensor)
    for value in values_tensor.unique():
        positions = (values_tensor == value).nonzero(as_tuple=False).flatten()
        less = int((values_tensor < value).sum())
        result[positions] = less + (len(positions) - 1) / 2
    return result


def spearman_rank_correlation(
    reference: Mapping[str, float],
    estimate: Mapping[str, float],
) -> float:
    """Compute Spearman correlation over component magnitudes, including ties."""

    if set(reference) != set(estimate):
        raise ValueError("Reference and estimate must contain the same names.")
    names = tuple(sorted(reference))
    x = _average_ranks([abs(float(reference[name])) for name in names])
    y = _average_ranks([abs(float(estimate[name])) for name in names])
    x = x - x.mean()
    y = y - y.mean()
    denominator = x.norm() * y.norm()
    return 0.0 if float(denominator) == 0.0 else float((x @ y) / denominator)


def recovery_at_k(
    exact_effects: Mapping[str, float],
    ranking: Sequence[str],
) -> tuple[float, ...]:
    """Return cumulative exact-effect mass captured by a proposed ranking."""

    if set(ranking) != set(exact_effects) or len(ranking) != len(exact_effects):
        raise ValueError("ranking must be a permutation of exact-effect names.")
    total = sum(abs(float(value)) for value in exact_effects.values())
    if total <= 0:
        raise ValueError("Exact effect mass must be positive.")
    cumulative = 0.0
    result = []
    for name in ranking:
        cumulative += abs(float(exact_effects[name]))
        result.append(cumulative / total)
    return tuple(result)


def topk_faithfulness_curve(
    model: ToyAtPStarModel,
    clean_sites: Tensor,
    corrupt_sites: Tensor,
    ranking: Sequence[str],
) -> tuple[float, ...]:
    """Patch the top-k sites jointly and normalize their effect by full corruption."""

    if tuple(sorted(ranking)) != tuple(sorted(TOY_COMPONENTS)):
        raise ValueError("ranking must contain every toy component exactly once.")
    with t.no_grad():
        clean_metric = model(clean_sites).metric
        full_effect = (model(corrupt_sites).metric - clean_metric).abs()
        if float(full_effect) == 0.0:
            raise ValueError("Full corruption must change the metric.")
        patched = clean_sites.clone()
        curve = []
        for name in ranking:
            index = TOY_COMPONENT_INDEX[name]
            patched[index] = corrupt_sites[index]
            curve.append(float((model(patched).metric - clean_metric).abs() / full_effect))
    return tuple(curve)


def evaluate_toy_method(
    method: str,
    scores: Mapping[str, float],
    exact_effects: Mapping[str, float],
    model: ToyAtPStarModel,
    clean_sites: Tensor,
    corrupt_sites: Tensor,
    *,
    wall_time_ms: float,
    estimated_tensor_bytes: int,
    forward_passes: int,
    backward_passes: int,
) -> MethodEvaluation:
    ranking = rank_components(scores)
    return MethodEvaluation(
        method=method,
        ranking=ranking,
        spearman=spearman_rank_correlation(exact_effects, scores),
        recovery_at_k=recovery_at_k(exact_effects, ranking),
        faithfulness_at_k=topk_faithfulness_curve(model, clean_sites, corrupt_sites, ranking),
        wall_time_ms=wall_time_ms,
        estimated_tensor_bytes=estimated_tensor_bytes,
        forward_passes=forward_passes,
        backward_passes=backward_passes,
    )


def _median_runtime_ms(fn: Any, *, repeats: int = 31) -> tuple[Any, float]:
    if repeats <= 0:
        raise ValueError("repeats must be positive.")
    timings = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        timings.append((time.perf_counter() - start) * 1000)
    timings.sort()
    return result, timings[len(timings) // 2]


def run_toy_study(*, component_batch_size: int = 4) -> dict[str, Any]:
    """Run the complete exact toy comparison used by the lesson and smoke contract."""

    model = ToyAtPStarModel()
    clean = toy_site_tensor(TOY_CLEAN_VALUES)
    corrupt = toy_site_tensor(TOY_CORRUPT_VALUES)
    clean_run, corrupt_run = cache_clean_and_corrupt(model, clean, corrupt)

    exact, exact_ms = _median_runtime_ms(
        lambda: batched_exact_patch_effects(
            model, clean, corrupt, component_batch_size=component_batch_size
        )
    )
    vanilla, vanilla_ms = _median_runtime_ms(
        lambda: vanilla_attribution_patching(model, clean, corrupt)
    )
    atp_star_result, atp_star_ms = _median_runtime_ms(
        lambda: atp_star_scores(model, clean, corrupt)
    )
    atp_star, per_drop = atp_star_result
    edge_result, eap_ms = _median_runtime_ms(
        lambda: eap_style_edge_scores(model, clean, corrupt)
    )
    raw_edges, eap = edge_result

    bytes_per_scalar = clean.element_size()
    exact_batches = math.ceil(len(TOY_COMPONENTS) / component_batch_size)
    reports = {
        "Exact patching": evaluate_toy_method(
            "Exact patching",
            exact,
            exact,
            model,
            clean,
            corrupt,
            wall_time_ms=exact_ms,
            estimated_tensor_bytes=component_batch_size * len(TOY_COMPONENTS) * bytes_per_scalar,
            forward_passes=1 + exact_batches,
            backward_passes=0,
        ),
        "Vanilla AtP": evaluate_toy_method(
            "Vanilla AtP",
            vanilla,
            exact,
            model,
            clean,
            corrupt,
            wall_time_ms=vanilla_ms,
            estimated_tensor_bytes=2 * len(TOY_COMPONENTS) * bytes_per_scalar,
            forward_passes=2,
            backward_passes=1,
        ),
        "AtP*": evaluate_toy_method(
            "AtP*",
            atp_star,
            exact,
            model,
            clean,
            corrupt,
            wall_time_ms=atp_star_ms,
            estimated_tensor_bytes=4 * len(TOY_COMPONENTS) * bytes_per_scalar,
            forward_passes=2,
            backward_passes=len(TOY_GRAD_PATHS),
        ),
        "EAP-style": evaluate_toy_method(
            "EAP-style",
            eap,
            exact,
            model,
            clean,
            corrupt,
            wall_time_ms=eap_ms,
            estimated_tensor_bytes=3 * len(TOY_COMPONENTS) * bytes_per_scalar,
            forward_passes=2,
            backward_passes=1,
        ),
    }
    random_scores = {name: float(index) for index, name in enumerate(TOY_COMPONENTS)}
    reports["Random"] = evaluate_toy_method(
        "Random",
        random_scores,
        exact,
        model,
        clean,
        corrupt,
        wall_time_ms=0.0,
        estimated_tensor_bytes=0,
        forward_passes=0,
        backward_passes=0,
    )
    accepted = (
        abs(float(clean_run.metric) - 2.9) < 1e-6
        and abs(float(corrupt_run.metric)) < 1e-6
        and abs(vanilla["L0M0.cancel_source"]) < 1e-12
        and abs(vanilla["L0H0.Q@dest"]) < 1e-4
        and reports["AtP*"].spearman > 0.99
        and reports["AtP*"].recovery_at_k[3] > 0.85
        and reports["Vanilla AtP"].recovery_at_k[3] < 0.8
    )
    return {
        "accepted": accepted,
        "clean_metric": float(clean_run.metric),
        "corrupt_metric": float(corrupt_run.metric),
        "exact_effects": exact,
        "vanilla_atp": vanilla,
        "atp_star": atp_star,
        "grad_drop_estimates": per_drop,
        "eap_edges": raw_edges,
        "eap_by_component": eap,
        "reports": {name: asdict(report) for name, report in reports.items()},
    }


# %% Pinned TransformerLens bridge
PYTHIA_MODEL_NAME = "pythia-70m-deduped"
PYTHIA_HF_ID = "EleutherAI/pythia-70m-deduped"
PYTHIA_REVISION = "e93a9faa9c77e5d09219f6c868bfc7a1bd65593c"
PYTHIA_BNB_IMPORT_ABI = "130"
REAL_PAIR_COUNT = 24


@dataclass(frozen=True)
class GrammarPair:
    clean_prompt: str
    corrupt_prompt: str
    positive_token: str
    negative_token: str
    family: str


def build_grammar_pairs() -> tuple[GrammarPair, ...]:
    """Return 24 equal-length plural/singular counterfactual prompt pairs."""

    nouns = (
        ("cat", "cats"),
        ("dog", "dogs"),
        ("bird", "birds"),
        ("car", "cars"),
        ("book", "books"),
        ("tree", "trees"),
        ("house", "houses"),
        ("river", "rivers"),
    )
    pairs = []
    for singular, plural in nouns:
        pairs.extend(
            (
                GrammarPair(f"The {plural}", f"The {singular}", " are", " is", "copula"),
                GrammarPair(f"Those {plural}", f"That {singular}", " have", " has", "possession"),
                GrammarPair(f"Many {plural}", f"One {singular}", " run", " runs", "agreement"),
            )
        )
    return tuple(pairs)


def _install_transformers_pythia_compatibility() -> None:
    """Bridge Transformers 5's rope_parameters field to TransformerLens' reader."""

    from transformers import GPTNeoXConfig

    if not hasattr(GPTNeoXConfig, "rotary_pct"):
        GPTNeoXConfig.rotary_pct = property(  # type: ignore[attr-defined]
            lambda config: config.rope_parameters["partial_rotary_factor"]
        )


def load_pinned_pythia70m(device: str = "cpu"):
    """Load the exact public checkpoint; errors propagate rather than changing models."""

    if device == "cuda" and not t.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    _install_transformers_pythia_compatibility()
    os.environ.setdefault("BNB_CUDA_VERSION", PYTHIA_BNB_IMPORT_ABI)
    logging.getLogger("bitsandbytes.cextension").setLevel(logging.CRITICAL)
    from transformer_lens import HookedTransformer

    model = HookedTransformer.from_pretrained(
        PYTHIA_MODEL_NAME,
        revision=PYTHIA_REVISION,
        device=device,
        dtype=t.float32,
    )
    model.requires_grad_(False)
    return model


def prepare_grammar_task(model: Any, pairs: Sequence[GrammarPair] | None = None) -> dict[str, Any]:
    """Tokenize and validate the fixed real-model task."""

    pairs = tuple(pairs or build_grammar_pairs())
    if len(pairs) < 20:
        raise ValueError("The real task requires at least 20 clean/corrupt examples.")
    clean_tokens = t.cat([model.to_tokens(pair.clean_prompt, prepend_bos=True) for pair in pairs])
    corrupt_tokens = t.cat([model.to_tokens(pair.corrupt_prompt, prepend_bos=True) for pair in pairs])
    if clean_tokens.shape != corrupt_tokens.shape:
        raise ValueError("Clean and corrupt prompts must tokenize to the same batch shape.")
    positive_ids = t.tensor(
        [model.to_single_token(pair.positive_token) for pair in pairs],
        device=clean_tokens.device,
    )
    negative_ids = t.tensor(
        [model.to_single_token(pair.negative_token) for pair in pairs],
        device=clean_tokens.device,
    )
    return {
        "pairs": pairs,
        "clean_tokens": clean_tokens,
        "corrupt_tokens": corrupt_tokens,
        "positive_ids": positive_ids,
        "negative_ids": negative_ids,
    }


def per_prompt_logit_diff(logits: Tensor, positive_ids: Tensor, negative_ids: Tensor) -> Tensor:
    """Return positive-minus-negative final-token logits for each prompt."""

    if logits.ndim != 3 or positive_ids.ndim != 1 or negative_ids.ndim != 1:
        raise ValueError("Expected logits [batch, position, vocab] and token ids [batch].")
    if logits.shape[0] != positive_ids.shape[0] or positive_ids.shape != negative_ids.shape:
        raise ValueError("Batch dimensions must match.")
    final = logits[:, -1]
    rows = t.arange(final.shape[0], device=final.device)
    return final[rows, positive_ids] - final[rows, negative_ids]


def real_component_names(n_layers: int, n_heads: int) -> tuple[str, ...]:
    return tuple(
        f"L{layer}H{head}.{kind}@end"
        for layer in range(n_layers)
        for head in range(n_heads)
        for kind in ("Q", "K")
    )


def parse_real_component(name: str) -> tuple[int, int, Literal["Q", "K"]]:
    try:
        prefix, suffix = name.split(".")
        layer_text, head_text = prefix[1:].split("H")
        kind, position = suffix.split("@")
        if position != "end" or kind not in {"Q", "K"}:
            raise ValueError
        return int(layer_text), int(head_text), kind  # type: ignore[return-value]
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Invalid component name: {name!r}") from exc


def _q_hook_name(layer: int) -> str:
    return f"blocks.{layer}.attn.hook_rot_q"


def _k_hook_name(layer: int) -> str:
    return f"blocks.{layer}.attn.hook_rot_k"


def _pattern_hook_name(layer: int) -> str:
    return f"blocks.{layer}.attn.hook_pattern"


def cache_real_qk(model: Any, task: Mapping[str, Any]) -> dict[str, Any]:
    """Cache clean/corrupt rotary Q/K and clean patterns for every layer."""

    names = {
        name
        for layer in range(model.cfg.n_layers)
        for name in (_q_hook_name(layer), _k_hook_name(layer), _pattern_hook_name(layer))
    }
    with t.inference_mode():
        clean_logits, clean_cache = model.run_with_cache(
            task["clean_tokens"], names_filter=lambda name: name in names
        )
        corrupt_logits, corrupt_cache = model.run_with_cache(
            task["corrupt_tokens"], names_filter=lambda name: name in names
        )
    clean_metric = per_prompt_logit_diff(
        clean_logits, task["positive_ids"], task["negative_ids"]
    )
    corrupt_metric = per_prompt_logit_diff(
        corrupt_logits, task["positive_ids"], task["negative_ids"]
    )
    return {
        "clean_cache": clean_cache,
        "corrupt_cache": corrupt_cache,
        "clean_metric": clean_metric,
        "corrupt_metric": corrupt_metric,
    }


def _patch_component_chunk_hooks(
    components: Sequence[str],
    corrupt_cache: Mapping[str, Tensor],
    *,
    examples_per_component: int,
) -> list[tuple[str, Any]]:
    hooks = []
    for layer in sorted({parse_real_component(name)[0] for name in components}):
        for kind, hook_name in (("Q", _q_hook_name(layer)), ("K", _k_hook_name(layer))):
            selected = [
                (index, parse_real_component(name)[1])
                for index, name in enumerate(components)
                if parse_real_component(name)[0] == layer and parse_real_component(name)[2] == kind
            ]
            if not selected:
                continue

            def patch_hook(activation: Tensor, hook: Any, selected=selected, hook_name=hook_name):
                patched = activation.clone()
                source = corrupt_cache[hook_name]
                for component_index, head in selected:
                    rows = slice(
                        component_index * examples_per_component,
                        (component_index + 1) * examples_per_component,
                    )
                    patched[rows, -1, head] = source[:, -1, head]
                return patched

            hooks.append((hook_name, patch_hook))
    return hooks


def exact_real_patch_effects(
    model: Any,
    task: Mapping[str, Any],
    cache: Mapping[str, Any],
    *,
    component_batch_size: int = 8,
) -> dict[str, float]:
    """Exhaustively patch every final-position Q/K site, batching components."""

    if component_batch_size <= 0:
        raise ValueError("component_batch_size must be positive.")
    names = real_component_names(model.cfg.n_layers, model.cfg.n_heads)
    batch = task["clean_tokens"].shape[0]
    effects: dict[str, float] = {}
    with t.inference_mode():
        for start in range(0, len(names), component_batch_size):
            chunk = names[start : start + component_batch_size]
            repeated_tokens = task["clean_tokens"].repeat(len(chunk), 1)
            repeated_positive = task["positive_ids"].repeat(len(chunk))
            repeated_negative = task["negative_ids"].repeat(len(chunk))
            hooks = _patch_component_chunk_hooks(
                chunk, cache["corrupt_cache"], examples_per_component=batch
            )
            logits = model.run_with_hooks(repeated_tokens, fwd_hooks=hooks)
            metrics = per_prompt_logit_diff(logits, repeated_positive, repeated_negative)
            deltas = metrics.reshape(len(chunk), batch) - cache["clean_metric"].unsqueeze(0)
            for index, name in enumerate(chunk):
                effects[name] = float(deltas[index].abs().mean())
    return effects


def vanilla_real_atp_scores(
    model: Any,
    task: Mapping[str, Any],
    cache: Mapping[str, Any],
) -> dict[str, float]:
    """Compute node-level AtP for all final-position rotary Q/K sites in one backward pass."""

    leaves: dict[str, Tensor] = {}
    hooks = []
    for layer in range(model.cfg.n_layers):
        for hook_name in (_q_hook_name(layer), _k_hook_name(layer)):
            def leaf_hook(activation: Tensor, hook: Any, hook_name=hook_name):
                # Keep the path through later attention layers. Detaching every
                # cached node would measure a different, truncated graph.
                activation.requires_grad_(True)
                activation.retain_grad()
                leaves[hook_name] = activation
                return activation

            hooks.append((hook_name, leaf_hook))
    logits = model.run_with_hooks(task["clean_tokens"], fwd_hooks=hooks)
    per_prompt_logit_diff(logits, task["positive_ids"], task["negative_ids"]).sum().backward()

    scores = {}
    for name in real_component_names(model.cfg.n_layers, model.cfg.n_heads):
        layer, head, kind = parse_real_component(name)
        hook_name = _q_hook_name(layer) if kind == "Q" else _k_hook_name(layer)
        leaf = leaves[hook_name]
        if leaf.grad is None:
            raise RuntimeError(f"No gradient retained for {hook_name}.")
        delta = cache["corrupt_cache"][hook_name][:, -1, head] - cache["clean_cache"][hook_name][:, -1, head]
        estimate = (delta * leaf.grad[:, -1, head]).sum(dim=-1)
        scores[name] = float(estimate.abs().mean())
    return scores


def _pattern_gradients(
    model: Any,
    task: Mapping[str, Any],
    *,
    dropped_layer: int,
    positive_ids: Tensor,
    negative_ids: Tensor,
) -> dict[str, Tensor]:
    leaves: dict[str, Tensor] = {}
    hooks = []
    for layer in range(model.cfg.n_layers):
        hook_name = _pattern_hook_name(layer)

        def pattern_leaf(activation: Tensor, hook: Any, hook_name=hook_name):
            activation.requires_grad_(True)
            activation.retain_grad()
            leaves[hook_name] = activation
            return activation

        hooks.append((hook_name, pattern_leaf))

    for hook_name in (
        f"blocks.{dropped_layer}.hook_attn_out",
        f"blocks.{dropped_layer}.hook_mlp_out",
    ):
        hooks.append((hook_name, lambda activation, hook: activation.detach()))

    logits = model.run_with_hooks(task["clean_tokens"], fwd_hooks=hooks)
    per_prompt_logit_diff(logits, positive_ids, negative_ids).sum().backward()
    gradients = {}
    for name, leaf in leaves.items():
        if leaf.grad is None:
            gradients[name] = t.zeros_like(leaf)
        else:
            gradients[name] = leaf.grad.detach().clone()
    return gradients


def qk_corrected_real_estimates(
    model: Any,
    cache: Mapping[str, Any],
    pattern_gradients: Mapping[str, Tensor],
    *,
    corrupt_example_order: Tensor | None = None,
) -> dict[str, Tensor]:
    """Return per-example QK-fixed estimates for every final-position Q/K site."""

    batch = cache["clean_metric"].shape[0]
    if corrupt_example_order is None:
        corrupt_example_order = t.arange(batch, device=cache["clean_metric"].device)
    if corrupt_example_order.shape != (batch,):
        raise ValueError("corrupt_example_order must have shape [batch].")
    estimates: dict[str, Tensor] = {}
    scale = math.sqrt(model.cfg.d_head)
    for layer in range(model.cfg.n_layers):
        q_name = _q_hook_name(layer)
        k_name = _k_hook_name(layer)
        pattern_name = _pattern_hook_name(layer)
        clean_q = cache["clean_cache"][q_name]
        clean_k = cache["clean_cache"][k_name]
        corrupt_q = cache["corrupt_cache"][q_name][corrupt_example_order]
        corrupt_k = cache["corrupt_cache"][k_name][corrupt_example_order]
        clean_pattern = cache["clean_cache"][pattern_name][:, :, -1]
        pattern_grad = pattern_gradients[pattern_name][:, :, -1]

        query_logits = t.einsum("bhd,bshd->bhs", corrupt_q[:, -1], clean_k) / scale
        query_pattern = query_logits.softmax(dim=-1)
        query_estimate = ((query_pattern - clean_pattern) * pattern_grad).sum(dim=-1)

        clean_final_q = clean_q[:, -1]
        for head in range(model.cfg.n_heads):
            patched_k = clean_k[:, :, head].clone()
            patched_k[:, -1] = corrupt_k[:, -1, head]
            key_logits = t.einsum("bd,bsd->bs", clean_final_q[:, head], patched_k) / scale
            key_pattern = key_logits.softmax(dim=-1)
            key_estimate = (
                (key_pattern - clean_pattern[:, head]) * pattern_grad[:, head]
            ).sum(dim=-1)
            estimates[f"L{layer}H{head}.Q@end"] = query_estimate[:, head]
            estimates[f"L{layer}H{head}.K@end"] = key_estimate
    return estimates


def real_atp_star_scores(
    model: Any,
    task: Mapping[str, Any],
    cache: Mapping[str, Any],
    *,
    corrupt_example_order: Tensor | None = None,
    positive_ids: Tensor | None = None,
    negative_ids: Tensor | None = None,
) -> dict[str, float]:
    """Apply QK correction under every layer GradDrop and aggregate as in AtP*."""

    positive_ids = task["positive_ids"] if positive_ids is None else positive_ids
    negative_ids = task["negative_ids"] if negative_ids is None else negative_ids
    accumulated = {
        name: t.zeros(task["clean_tokens"].shape[0], device=task["clean_tokens"].device)
        for name in real_component_names(model.cfg.n_layers, model.cfg.n_heads)
    }
    for dropped_layer in range(model.cfg.n_layers):
        gradients = _pattern_gradients(
            model,
            task,
            dropped_layer=dropped_layer,
            positive_ids=positive_ids,
            negative_ids=negative_ids,
        )
        estimates = qk_corrected_real_estimates(
            model,
            cache,
            gradients,
            corrupt_example_order=corrupt_example_order,
        )
        for name, values in estimates.items():
            accumulated[name] += values.abs()
    denominator = model.cfg.n_layers - 1
    if denominator <= 0:
        raise ValueError("Real GradDrop requires a model with at least two layers.")
    return {
        name: float((values / denominator).mean()) for name, values in accumulated.items()
    }


def rank_named_scores(scores: Mapping[str, float]) -> tuple[str, ...]:
    if not scores or not all(math.isfinite(float(value)) for value in scores.values()):
        raise ValueError("Scores must be a non-empty finite mapping.")
    return tuple(sorted(scores, key=lambda name: (-abs(float(scores[name])), name)))


def real_topk_patch_effect(
    model: Any,
    task: Mapping[str, Any],
    cache: Mapping[str, Any],
    components: Sequence[str],
) -> float:
    """Causally patch a set of real Q/K sites together and return mean absolute effect."""

    parsed = [parse_real_component(name) for name in components]
    if len(set(components)) != len(components):
        raise ValueError("components must be unique.")
    hooks = []
    for layer in sorted({item[0] for item in parsed}):
        for kind, hook_name in (("Q", _q_hook_name(layer)), ("K", _k_hook_name(layer))):
            heads = [head for comp_layer, head, comp_kind in parsed if comp_layer == layer and comp_kind == kind]
            if not heads:
                continue

            def patch_hook(activation: Tensor, hook: Any, heads=heads, hook_name=hook_name):
                patched = activation.clone()
                patched[:, -1, heads] = cache["corrupt_cache"][hook_name][:, -1, heads]
                return patched

            hooks.append((hook_name, patch_hook))
    with t.inference_mode():
        logits = model.run_with_hooks(task["clean_tokens"], fwd_hooks=hooks)
        patched_metric = per_prompt_logit_diff(
            logits, task["positive_ids"], task["negative_ids"]
        )
    return float((patched_metric - cache["clean_metric"]).abs().mean())


def real_component_set_effects(
    model: Any,
    task: Mapping[str, Any],
    cache: Mapping[str, Any],
    component_sets: Sequence[Sequence[str]],
    *,
    set_batch_size: int = 8,
) -> tuple[float, ...]:
    """Causally evaluate several component sets in vectorized model batches."""

    if set_batch_size <= 0 or not component_sets:
        raise ValueError("set_batch_size and component_sets must be non-empty and positive.")
    valid_names = set(real_component_names(model.cfg.n_layers, model.cfg.n_heads))
    normalized = tuple(tuple(component_set) for component_set in component_sets)
    for component_set in normalized:
        if len(set(component_set)) != len(component_set) or not set(component_set) <= valid_names:
            raise ValueError("Each component set must contain unique known names.")

    examples = task["clean_tokens"].shape[0]
    effects = []
    with t.inference_mode():
        for start in range(0, len(normalized), set_batch_size):
            chunk = normalized[start : start + set_batch_size]
            parsed_sets = [[parse_real_component(name) for name in component_set] for component_set in chunk]
            hooks = []
            for layer in range(model.cfg.n_layers):
                for kind, hook_name in (("Q", _q_hook_name(layer)), ("K", _k_hook_name(layer))):
                    per_set_heads = [
                        [head for comp_layer, head, comp_kind in parsed if comp_layer == layer and comp_kind == kind]
                        for parsed in parsed_sets
                    ]
                    if not any(per_set_heads):
                        continue

                    def patch_hook(
                        activation: Tensor,
                        hook: Any,
                        per_set_heads=per_set_heads,
                        hook_name=hook_name,
                    ):
                        patched = activation.clone()
                        source = cache["corrupt_cache"][hook_name]
                        for set_index, heads in enumerate(per_set_heads):
                            if not heads:
                                continue
                            rows = slice(set_index * examples, (set_index + 1) * examples)
                            patched[rows, -1, heads] = source[:, -1, heads]
                        return patched

                    hooks.append((hook_name, patch_hook))
            repeated_tokens = task["clean_tokens"].repeat(len(chunk), 1)
            repeated_positive = task["positive_ids"].repeat(len(chunk))
            repeated_negative = task["negative_ids"].repeat(len(chunk))
            logits = model.run_with_hooks(repeated_tokens, fwd_hooks=hooks)
            metrics = per_prompt_logit_diff(logits, repeated_positive, repeated_negative)
            deltas = metrics.reshape(len(chunk), examples) - cache["clean_metric"].unsqueeze(0)
            effects.extend(float(row.abs().mean()) for row in deltas)
    return tuple(effects)


def real_faithfulness_curve(
    model: Any,
    task: Mapping[str, Any],
    cache: Mapping[str, Any],
    ranking: Sequence[str],
    ks: Sequence[int],
) -> dict[int, float]:
    """Measure joint causal effect of each top-k prefix, normalized by all Q/K sites."""

    names = real_component_names(model.cfg.n_layers, model.cfg.n_heads)
    if set(ranking) != set(names):
        raise ValueError("ranking must contain every real component exactly once.")
    requested_ks = tuple(int(k) for k in ks)
    if any(k <= 0 or k > len(ranking) for k in requested_ks):
        raise ValueError("Every k must be between one and the number of components.")
    sets = (names, *(tuple(ranking[:k]) for k in requested_ks))
    denominator, *prefix_effects = real_component_set_effects(
        model, task, cache, sets, set_batch_size=len(sets)
    )
    if denominator <= 0:
        raise ValueError("Patching all candidate sites must change the metric.")
    return {k: effect / denominator for k, effect in zip(requested_ks, prefix_effects, strict=True)}


def same_size_random_circuit_control(
    model: Any,
    task: Mapping[str, Any],
    cache: Mapping[str, Any],
    discovered: Sequence[str],
    *,
    samples: int = 24,
    seed: int = 0,
) -> dict[str, Any]:
    """Compare a discovered component set with reproducible same-size random sets."""

    names = real_component_names(model.cfg.n_layers, model.cfg.n_heads)
    if samples <= 0 or len(discovered) <= 0 or len(discovered) >= len(names):
        raise ValueError("Invalid control sample count or discovered circuit size.")
    rng = random.Random(seed)
    random_sets = []
    while len(random_sets) < samples:
        candidate = tuple(sorted(rng.sample(names, len(discovered))))
        if candidate != tuple(sorted(discovered)) and candidate not in random_sets:
            random_sets.append(candidate)
    discovered_effect = real_topk_patch_effect(model, task, cache, discovered)
    random_effects = real_component_set_effects(
        model,
        task,
        cache,
        random_sets,
        set_batch_size=8,
    )
    rank = 1 + sum(value >= discovered_effect for value in random_effects)
    return {
        "discovered_effect": discovered_effect,
        "random_effects": random_effects,
        "random_mean": sum(random_effects) / len(random_effects),
        "random_best": max(random_effects),
        "rank": rank,
        "num_compared": samples + 1,
        "beats_random_mean": discovered_effect > sum(random_effects) / len(random_effects),
    }


def _mapping_spearman(reference: Mapping[str, float], estimate: Mapping[str, float]) -> float:
    names = tuple(sorted(reference))
    if set(names) != set(estimate):
        raise ValueError("Mappings must have the same keys.")
    x = _average_ranks([float(reference[name]) for name in names])
    y = _average_ranks([float(estimate[name]) for name in names])
    x, y = x - x.mean(), y - y.mean()
    denominator = x.norm() * y.norm()
    return 0.0 if float(denominator) == 0 else float((x @ y) / denominator)


def _cache_tensor_bytes(cache: Mapping[str, Any]) -> int:
    seen: set[int] = set()
    total = 0
    for subcache_name in ("clean_cache", "corrupt_cache"):
        for value in cache[subcache_name].values():
            if isinstance(value, Tensor) and value.data_ptr() not in seen:
                seen.add(value.data_ptr())
                total += value.numel() * value.element_size()
    return total


def run_real_transformerlens_study(
    device: str = "cpu",
    *,
    component_batch_size: int = 8,
    top_k: int = 8,
    random_samples: int = 24,
) -> dict[str, Any]:
    """Run the pinned 24-example component-level AtP* study."""

    model = load_pinned_pythia70m(device)
    task = prepare_grammar_task(model)
    cache = cache_real_qk(model, task)

    start = time.perf_counter()
    exact = exact_real_patch_effects(
        model, task, cache, component_batch_size=component_batch_size
    )
    exact_seconds = time.perf_counter() - start

    start = time.perf_counter()
    vanilla = vanilla_real_atp_scores(model, task, cache)
    vanilla_seconds = time.perf_counter() - start

    start = time.perf_counter()
    atp_star = real_atp_star_scores(model, task, cache)
    atp_star_seconds = time.perf_counter() - start

    names = real_component_names(model.cfg.n_layers, model.cfg.n_heads)
    exact_ranking = rank_named_scores(exact)
    vanilla_ranking = rank_named_scores(vanilla)
    atp_star_ranking = rank_named_scores(atp_star)
    exact_total = sum(exact.values())

    def recovery(ranking: Sequence[str], k: int) -> float:
        return sum(exact[name] for name in ranking[:k]) / exact_total

    ks = (1, 2, 4, 8, 16, 32)
    recoveries = {
        "Exact": {k: recovery(exact_ranking, k) for k in ks},
        "Vanilla AtP": {k: recovery(vanilla_ranking, k) for k in ks},
        "AtP*": {k: recovery(atp_star_ranking, k) for k in ks},
    }
    atp_star_faithfulness = real_faithfulness_curve(
        model, task, cache, atp_star_ranking, (2, 4, 8, 16)
    )
    discovered = atp_star_ranking[:top_k]
    random_control = same_size_random_circuit_control(
        model,
        task,
        cache,
        discovered,
        samples=random_samples,
        seed=0,
    )

    batch = task["clean_tokens"].shape[0]
    generator = t.Generator(device="cpu").manual_seed(0)
    pair_order = t.randperm(batch, generator=generator).to(task["clean_tokens"].device)
    pair_control_cache = dict(cache)
    pair_control_cache["corrupt_cache"] = cache["clean_cache"]
    pair_shuffled = real_atp_star_scores(
        model, task, pair_control_cache, corrupt_example_order=pair_order
    )
    positive_order = t.randperm(batch, generator=generator).to(task["clean_tokens"].device)
    negative_order = t.randperm(batch, generator=generator).to(task["clean_tokens"].device)
    label_shuffled = real_atp_star_scores(
        model,
        task,
        cache,
        positive_ids=task["positive_ids"][positive_order],
        negative_ids=task["negative_ids"][negative_order],
    )
    pair_ranking = rank_named_scores(pair_shuffled)
    label_ranking = rank_named_scores(label_shuffled)
    control_effects = {
        "pair_shuffled": real_topk_patch_effect(model, task, cache, pair_ranking[:top_k]),
        "label_shuffled": real_topk_patch_effect(model, task, cache, label_ranking[:top_k]),
    }

    clean_mean = float(cache["clean_metric"].mean())
    corrupt_mean = float(cache["corrupt_metric"].mean())
    result = {
        "accepted": (
            len(task["pairs"]) >= 20
            and clean_mean - corrupt_mean > 3.0
            and _mapping_spearman(exact, atp_star) > _mapping_spearman(exact, vanilla)
            and recoveries["AtP*"][32] > recoveries["Vanilla AtP"][32]
            and random_control["beats_random_mean"]
            and random_control["discovered_effect"] > control_effects["pair_shuffled"]
            and random_control["discovered_effect"] > control_effects["label_shuffled"]
        ),
        "device": device,
        "model_name": PYTHIA_MODEL_NAME,
        "hf_model_id": PYTHIA_HF_ID,
        "hf_revision": PYTHIA_REVISION,
        "num_examples": len(task["pairs"]),
        "sequence_length": int(task["clean_tokens"].shape[1]),
        "num_components": len(names),
        "clean_metric_mean": clean_mean,
        "corrupt_metric_mean": corrupt_mean,
        "clean_corrupt_gap": clean_mean - corrupt_mean,
        "exact_effects": exact,
        "vanilla_atp": vanilla,
        "atp_star": atp_star,
        "exact_ranking": exact_ranking,
        "vanilla_ranking": vanilla_ranking,
        "atp_star_ranking": atp_star_ranking,
        "spearman": {
            "vanilla_atp": _mapping_spearman(exact, vanilla),
            "atp_star": _mapping_spearman(exact, atp_star),
            "pair_shuffled": _mapping_spearman(exact, pair_shuffled),
            "label_shuffled": _mapping_spearman(exact, label_shuffled),
        },
        "recovery_at_k": recoveries,
        "atp_star_faithfulness_at_k": atp_star_faithfulness,
        "top_k": top_k,
        "top_components": discovered,
        "same_size_random": random_control,
        "control_effects": control_effects,
        "runtime_seconds": {
            "exact_patching": exact_seconds,
            "vanilla_atp": vanilla_seconds,
            "atp_star": atp_star_seconds,
        },
        "estimated_cache_bytes": _cache_tensor_bytes(cache),
        "claim_scope": (
            "Final-position rotary Q/K component screening on a pinned 70M transformer; "
            "not the paper's full Pythia-scale or IOI replication."
        ),
    }
    return result


def run_smoke_test(cpu: bool = True) -> dict[str, Any]:
    """Run the exact toy contract only."""

    _ = cpu
    return run_toy_study()


def run_gpu_test(max_vram_gb: float = 24.0) -> dict[str, Any]:
    """Run the real study on CUDA with a hard memory budget and no CPU fallback."""

    if not t.cuda.is_available():
        raise RuntimeError("CUDA is required for run_gpu_test; no CPU fallback is permitted.")
    t.cuda.empty_cache()
    t.cuda.reset_peak_memory_stats()
    result = run_real_transformerlens_study("cuda")
    peak_vram_gb = t.cuda.max_memory_allocated() / 1e9
    if peak_vram_gb > max_vram_gb:
        raise RuntimeError(
            f"Peak VRAM {peak_vram_gb:.3f} GB exceeds budget {max_vram_gb:.3f} GB."
        )
    return {**result, "peak_vram_gb": peak_vram_gb, "max_vram_gb": max_vram_gb}


def run_full_experiment(max_vram_gb: float = 24.0) -> dict[str, Any]:
    """Run exact toy evidence followed by the CUDA-only real study."""

    return {
        "toy": run_toy_study(),
        "gpu": run_gpu_test(max_vram_gb=max_vram_gb),
    }


if MAIN:
    print(run_smoke_test())
