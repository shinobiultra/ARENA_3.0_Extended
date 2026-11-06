# %%


from __future__ import annotations
MAIN = __name__ == "__main__"

# %%

import json
import math
import os
import sys
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence
import matplotlib.pyplot as plt
import pandas as pd
import torch as t
from IPython.display import display
from torch import Tensor

MODEL_ID = 'EleutherAI/pythia-70m-deduped'
MODEL_REVISION = 'e93a9faa9c77e5d09219f6c868bfc7a1bd65593c'
SUBJECT = 'Louvre'
OLD_OBJECT = ' Paris'
NEW_OBJECT = ' Rome'

from chapter9_alignment_interpretability.exercises.part6_rome_memit.rome_metadata import (
    PromptRecord,
    HELDOUT_PARAPHRASE_RECORDS,
)

def initialize_notebook_plumbing():
    """Explicit legacy data/helper binding; never called during module import."""
    global ADVERSARIAL_RECORDS, AdversarialRecord, COVARIANCE_FACTS, FLUENCY_TEXTS, \
        HELDOUT_PARAPHRASES, HELDOUT_PARAPHRASE_RECORDS, KNOWN_FACT_CANDIDATES, \
        LearnerFunctions, MODEL_ID, MODEL_REVISION, NEW_OBJECT, OLD_OBJECT, \
        OPTIMIZATION_PROMPTS, OPTIMIZATION_RECORDS, PARAPHRASE_PROVENANCE, PromptRecord, \
        SAFE_GENERATION_PROMPTS, SHUFFLED_CONTROL_TARGETS, SIDE_EFFECT_PROMPTS, SUBJECT, \
        _batch_inputs, _mean, _normalize_target_list, _positions_for_mode, _single_token_id, \
        assets_dir, device, generate_safe_continuations, load_pinned_model, match_update_norm, \
        next_token_distribution_table, next_token_probability_table, normalize_target_text, \
        root_dir, section_dir, text_perplexities, tests, reference
    root_dir = next(
        p for p in [Path.cwd(), *Path.cwd().parents]
        if (p / "chapter9_alignment_interpretability").exists()
    )
    section_dir = root_dir / "chapter9_alignment_interpretability" / "exercises" / "part5_rome_model_editing"
    assets_dir = root_dir / "chapter9_alignment_interpretability" / "exercises" / "part6_rome_memit" / "media"
    if str(root_dir) not in sys.path:
        sys.path.append(str(root_dir))
    from chapter9_alignment_interpretability.exercises.part6_rome_memit import tests
    from chapter9_alignment_interpretability.exercises.part5_rome_model_editing import solutions as reference
    MODEL_ID = reference.MODEL_ID
    MODEL_REVISION = reference.MODEL_REVISION
    SUBJECT = reference.SUBJECT
    OLD_OBJECT = reference.OLD_OBJECT
    NEW_OBJECT = reference.NEW_OBJECT
    PromptRecord = reference.PromptRecord
    AdversarialRecord = reference.AdversarialRecord
    LearnerFunctions = reference.LearnerFunctions
    PARAPHRASE_PROVENANCE = reference.PARAPHRASE_PROVENANCE
    OPTIMIZATION_RECORDS = reference.OPTIMIZATION_RECORDS
    HELDOUT_PARAPHRASE_RECORDS = reference.HELDOUT_PARAPHRASE_RECORDS
    OPTIMIZATION_PROMPTS = reference.OPTIMIZATION_PROMPTS
    HELDOUT_PARAPHRASES = reference.HELDOUT_PARAPHRASES
    COVARIANCE_FACTS = reference.COVARIANCE_FACTS
    KNOWN_FACT_CANDIDATES = reference.KNOWN_FACT_CANDIDATES
    ADVERSARIAL_RECORDS = reference.ADVERSARIAL_RECORDS
    SIDE_EFFECT_PROMPTS = reference.SIDE_EFFECT_PROMPTS
    FLUENCY_TEXTS = reference.FLUENCY_TEXTS
    SAFE_GENERATION_PROMPTS = reference.SAFE_GENERATION_PROMPTS
    SHUFFLED_CONTROL_TARGETS = reference.SHUFFLED_CONTROL_TARGETS
    _batch_inputs = reference._batch_inputs
    _single_token_id = reference._single_token_id
    _normalize_target_list = reference._normalize_target_list
    _positions_for_mode = reference._positions_for_mode
    _mean = reference._mean
    normalize_target_text = reference.normalize_target_text
    load_pinned_model = reference.load_pinned_model
    next_token_probability_table = reference.next_token_probability_table
    next_token_distribution_table = reference.next_token_distribution_table
    text_perplexities = reference.text_perplexities
    generate_safe_continuations = reference.generate_safe_continuations
    match_update_norm = reference.match_update_norm
    t.manual_seed(0)
    device = t.device("cpu")
    print(f"torch={t.__version__}, device={device}, model={MODEL_ID}@{MODEL_REVISION[:8]}")


if MAIN:
    initialize_notebook_plumbing()

# %%

if MAIN:
    model, tokenizer = load_pinned_model(device=device)

if MAIN:
    cold = reference.score_rewrite_prompts(model, tokenizer, [record.prompt for record in HELDOUT_PARAPHRASE_RECORDS[:5]])

if MAIN:
    cold_rows = []

if MAIN:
    for index, record in enumerate(HELDOUT_PARAPHRASE_RECORDS[:5]):
        cold_rows.append({
            "id": record.prompt_id,
            "prompt": record.prompt,
            "P(Paris)": float(cold["selected_probs"][index, 0]),
            "P(Rome)": float(cold["selected_probs"][index, 1]),
            "Rome rank": int(cold["target_first_ranks"][index]),
            "top token": tokenizer.decode([int(cold["top_ids"][index])]),
        })

if MAIN:
    display(pd.DataFrame(cold_rows))

# %%

def minimum_norm_rank_one_update(
    weight: Tensor,
    key: Tensor,
    desired_value: Tensor,
) -> Tensor:
    """Return the minimum-Frobenius rank-one delta for ``key @ (W + delta)``."""
    if weight.ndim != 2 or key.ndim != 1 or desired_value.ndim != 1:
        raise ValueError("Expected weight [d_key, d_value] and one-dimensional key/value tensors.")
    if weight.shape != (key.numel(), desired_value.numel()):
        raise ValueError(
            f"Expected weight shape {(key.numel(), desired_value.numel())}, got {tuple(weight.shape)}."
        )
    denom = key.square().sum()
    if float(denom) <= t.finfo(key.dtype).eps:
        raise ValueError("The edit key must have nonzero norm.")
    residual = desired_value - key @ weight
    return t.outer(key / denom, residual)

# %%

if MAIN:
    tests.test_minimum_norm_rank_one_update_exact(minimum_norm_rank_one_update)

if MAIN:
    tests.test_minimum_norm_update_rejects_zero_key(minimum_norm_rank_one_update)

if MAIN:
    tests.test_minimum_norm_matches_independent_constrained_oracle(minimum_norm_rank_one_update)

# %%

def covariance_preconditioned_key(
    key: Tensor,
    key_bank: Tensor,
    *,
    ridge: float = 1e-2,
) -> Tensor:
    """Compute ``C^-1 key`` for ``C = K.T K / n + ridge I`` via Woodbury."""
    if key.ndim != 1 or key_bank.ndim != 2 or key_bank.shape[1] != key.numel():
        raise ValueError("Expected key [d_key] and key_bank [n_contexts, d_key].")
    if key_bank.shape[0] < 1:
        raise ValueError("key_bank must contain at least one independent context.")
    if ridge <= 0:
        raise ValueError("ridge must be positive so the empirical covariance is invertible.")
    n = int(key_bank.shape[0])
    a = key_bank / math.sqrt(n)
    gram = a @ a.T
    system = gram + ridge * t.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    correction = a.T @ t.linalg.solve(system, a @ key)
    return (key - correction) / ridge

# %%

if MAIN:
    tests.test_covariance_preconditioned_key_matches_dense_solve(covariance_preconditioned_key)

# %%

def course_rome_like_rank_one_update(
    weight: Tensor,
    key: Tensor,
    desired_module_output: Tensor,
    key_bank: Tensor,
    *,
    output_bias: Tensor | None = None,
    ridge: float = 1e-2,
) -> Tensor:
    """Return the course ROME-like affine write in row-vector convention.

    This uses the rank-one algebra associated with ROME, but it is not an
    official implementation parity claim. The course covariance corpus, key
    contexts, and value objective deliberately differ from the released code.
    """
    if weight.ndim != 2 or key.ndim != 1 or desired_module_output.ndim != 1:
        raise ValueError("Expected weight [d_key, d_value] and one-dimensional key/output tensors.")
    if weight.shape != (key.numel(), desired_module_output.numel()):
        raise ValueError("weight, key, and desired_module_output shapes do not agree.")
    if output_bias is None:
        output_bias = t.zeros_like(desired_module_output)
    if output_bias.shape != desired_module_output.shape:
        raise ValueError("output_bias must match desired_module_output.")

    direction = covariance_preconditioned_key(key, key_bank, ridge=ridge)
    denominator = t.dot(key, direction)
    if float(denominator.abs()) <= t.finfo(key.dtype).eps:
        raise ValueError("The covariance-preconditioned key is numerically orthogonal to the edit key.")
    current_module_output = key @ weight + output_bias
    residual = desired_module_output - current_module_output
    return t.outer(direction / denominator, residual)

# %%

if MAIN:
    tests.test_course_rome_like_update_matches_declared_equation(course_rome_like_rank_one_update)

if MAIN:
    tests.test_course_update_hits_actual_affine_module_output(course_rome_like_rank_one_update)

# %%

if MAIN:
    oracle_delta = reference.independent_constrained_solve_oracle(
        toy_weight := t.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, -0.5]], dtype=t.float64),
        toy_key := t.tensor([1.0, 1.0, 0.0], dtype=t.float64),
        toy_desired := t.tensor([-2.0, 3.0], dtype=t.float64),
    )

if MAIN:
    learner_delta = minimum_norm_rank_one_update(toy_weight, toy_key, toy_desired)

if MAIN:
    print("independent oracle max error", float((oracle_delta - learner_delta).abs().max()))

if MAIN:
    toy = reference.exact_rank_one_toy_oracle()

if MAIN:
    tests.test_toy_oracle_changes_only_the_requested_association(reference.exact_rank_one_toy_oracle)

if MAIN:
    reference.plot_toy_oracle(str(assets_dir / "rome_rank_one_toy_oracle.png"))

if MAIN:
    plt.show()

# %%

def token_positions_from_offsets(
    offsets: Sequence[Sequence[int]],
    subject_start: int,
    subject_end: int,
) -> list[int]:
    """Return token indices whose nonempty character spans overlap the subject span."""
    if not (0 <= subject_start < subject_end):
        raise ValueError("Expected a nonempty subject character span.")
    positions = [
        index
        for index, pair in enumerate(offsets)
        if len(pair) == 2
        and int(pair[1]) > int(pair[0])
        and int(pair[1]) > subject_start
        and int(pair[0]) < subject_end
    ]
    if not positions:
        raise ValueError("No tokenizer offsets overlap the requested subject span.")
    return positions

def locate_subject_tokens(tokenizer: Any, prompt: str, subject: str) -> list[int]:
    """Locate the first exact subject occurrence using tokenizer character offsets."""
    start = prompt.find(subject)
    if start < 0:
        raise ValueError(f"Subject {subject!r} is not present in prompt {prompt!r}.")
    encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    return token_positions_from_offsets(encoded["offset_mapping"], start, start + len(subject))

# %%

if MAIN:
    tests.test_token_positions_from_offsets_handles_multitoken_subject(token_positions_from_offsets)

if MAIN:
    tests.test_token_positions_ignores_empty_special_token_offsets(token_positions_from_offsets)

if MAIN:
    print(locate_subject_tokens(tokenizer, "Visitors can find the Louvre in", SUBJECT))

# %%

def target_token_ids(tokenizer: Any, target: str) -> list[int]:
    """Tokenize a single- or multi-token continuation without silently truncating it."""
    normalized = normalize_target_text(target)
    ids = tokenizer.encode(normalized, add_special_tokens=False)
    if not ids:
        raise ValueError(f"Target {target!r} produced no token ids.")
    return [int(token_id) for token_id in ids]

def continuation_score_table(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    targets: str | Sequence[str],
) -> dict[str, Any]:
    """Teacher-force complete targets and expose rank, likelihood, and token accuracy."""
    if not prompts:
        raise ValueError("prompts must not be empty.")
    target_texts = _normalize_target_list(targets, len(prompts))
    prompt_ids = [tokenizer.encode(prompt, add_special_tokens=False) for prompt in prompts]
    target_ids = [target_token_ids(tokenizer, target) for target in target_texts]
    joint_ids = []
    for prompt, target, prefix_ids, continuation_ids in zip(
        prompts, target_texts, prompt_ids, target_ids
    ):
        combined = tokenizer.encode(prompt + target, add_special_tokens=False)
        expected = list(prefix_ids) + list(continuation_ids)
        if combined != expected:
            raise ValueError(
                f"Prompt/target token boundary changed for {prompt!r} + {target!r}; "
                "keep the target's leading space explicit."
            )
        if not prefix_ids:
            raise ValueError("Each prompt must contain at least one token.")
        joint_ids.append(combined)

    device = next(model.parameters()).device
    padded = tokenizer.pad({"input_ids": joint_ids}, padding=True, return_tensors="pt")
    batch = {name: value.to(device) for name, value in padded.items()}
    with t.inference_mode():
        logits = model(**batch).logits
        log_probs = logits.log_softmax(dim=-1)

    first_probs = []
    top_ids = []
    first_target_probs = []
    first_target_ranks = []
    target_log_probs = []
    target_mean_log_probs = []
    token_top1_accuracies = []
    full_target_greedy_matches = []
    for row, (prefix, continuation) in enumerate(zip(prompt_ids, target_ids)):
        positions = t.arange(
            len(prefix) - 1,
            len(prefix) + len(continuation) - 1,
            device=device,
        )
        continuation_tensor = t.tensor(continuation, device=device)
        token_log_probs = log_probs[row, positions, continuation_tensor]
        prediction_ids = logits[row, positions].argmax(dim=-1)
        first_distribution = logits[row, len(prefix) - 1].softmax(dim=-1)
        first_id = continuation[0]
        first_probs.append(first_distribution)
        top_ids.append(first_distribution.argmax())
        first_target_probs.append(first_distribution[first_id])
        first_target_ranks.append(
            (logits[row, len(prefix) - 1] > logits[row, len(prefix) - 1, first_id]).sum() + 1
        )
        target_log_probs.append(token_log_probs.sum())
        target_mean_log_probs.append(token_log_probs.mean())
        token_top1_accuracies.append((prediction_ids == continuation_tensor).float().mean())
        full_target_greedy_matches.append((prediction_ids == continuation_tensor).all())

    target_log_prob_tensor = t.stack(target_log_probs)
    return {
        "prompts": list(prompts),
        "targets": target_texts,
        "target_token_ids": target_ids,
        "target_token_counts": t.tensor([len(ids) for ids in target_ids]),
        "first_probs": t.stack(first_probs).detach().cpu(),
        "top_ids": t.stack(top_ids).detach().cpu(),
        "first_target_probs": t.stack(first_target_probs).detach().cpu(),
        "first_target_ranks": t.stack(first_target_ranks).detach().cpu(),
        "target_log_probs": target_log_prob_tensor.detach().cpu(),
        "target_likelihoods": target_log_prob_tensor.exp().detach().cpu(),
        "target_mean_log_probs": t.stack(target_mean_log_probs).detach().cpu(),
        "token_top1_accuracies": t.stack(token_top1_accuracies).detach().cpu(),
        "full_target_greedy_matches": t.stack(full_target_greedy_matches).detach().cpu(),
    }

# %%

if MAIN:
    tests.test_continuation_scoring_uses_all_target_tokens(
        continuation_score_table, model, tokenizer
    )

if MAIN:
    target_demo = continuation_score_table(
        model,
        tokenizer,
        ["The Colosseum is in", "The Golden Gate Bridge is in"],
        [" Rome", " San Francisco"],
    )

if MAIN:
    display(pd.DataFrame({
        "target": target_demo["targets"],
        "token_count": target_demo["target_token_counts"],
        "full_target_likelihood": target_demo["target_likelihoods"],
        "first_token_rank": target_demo["first_target_ranks"],
        "full_greedy_match": target_demo["full_target_greedy_matches"],
    }))

# %%

def capture_mlp_outputs(model: Any, tokenizer: Any, prompt: str) -> list[Tensor]:
    """Capture every complete Pythia MLP output for one prompt."""
    captured: list[Tensor] = []
    handles = []

    def save_output(_module: Any, _inputs: tuple[Tensor, ...], output: Tensor) -> None:
        captured.append(output.detach().clone())

    for layer in model.gpt_neox.layers:
        handles.append(layer.mlp.register_forward_hook(save_output))
    try:
        with t.inference_mode():
            _ = model(**_batch_inputs(tokenizer, [prompt], next(model.parameters()).device))
    finally:
        for handle in handles:
            handle.remove()
    return captured

# %%

if MAIN:
    tests.test_capture_mlp_outputs_matches_direct_modules(capture_mlp_outputs, model, tokenizer)

if MAIN:
    captured = capture_mlp_outputs(model, tokenizer, OPTIMIZATION_PROMPTS[0])

if MAIN:
    assert len(captured) == model.config.num_hidden_layers

if MAIN:
    print([tuple(output.shape) for output in captured])

# %%

def activation_patching_mlp_grid(
    model: Any,
    tokenizer: Any,
    prompt: str,
    subject: str,
    target: str,
    *,
    samples: int = 4,
    noise_scale: float = 3.0,
    seed: int = 0,
) -> dict[str, Any]:
    """Corrupt subject embeddings, then patch one clean MLP output at a time."""
    if samples < 1:
        raise ValueError("samples must be positive.")
    if noise_scale <= 0:
        raise ValueError("noise_scale must be positive.")
    device = next(model.parameters()).device
    encoded = _batch_inputs(tokenizer, [prompt], device)
    seq_len = int(encoded["attention_mask"].sum().item())
    subject_positions = locate_subject_tokens(tokenizer, prompt, subject)
    target_id = _single_token_id(tokenizer, target)
    clean_mlp = capture_mlp_outputs(model, tokenizer, prompt)
    repeated = {name: value.repeat(samples, 1) for name, value in encoded.items()}
    embedding_std = float(model.gpt_neox.embed_in.weight.detach().float().std().item())
    generator = t.Generator(device=device).manual_seed(seed)
    noise = t.randn(
        samples,
        len(subject_positions),
        model.config.hidden_size,
        generator=generator,
        device=device,
    ) * (noise_scale * embedding_std)

    def corrupt_embeddings(_module: Any, _inputs: tuple[Tensor, ...], output: Tensor) -> Tensor:
        corrupted = output.clone()
        corrupted[:, subject_positions, :] += noise.to(output.dtype)
        return corrupted

    with t.inference_mode():
        clean_prob = float(model(**encoded).logits[0, -1].softmax(-1)[target_id].item())
        embed_handle = model.gpt_neox.embed_in.register_forward_hook(corrupt_embeddings)
        try:
            corrupt_prob = float(
                model(**repeated).logits[:, -1].softmax(-1)[:, target_id].mean().item()
            )
        finally:
            embed_handle.remove()

    recovery = t.empty(len(model.gpt_neox.layers), seq_len, device=device)
    raw_effect = t.empty_like(recovery)
    denominator = clean_prob - corrupt_prob
    for layer_index, layer_module in enumerate(model.gpt_neox.layers):
        for token_index in range(seq_len):

            def restore_output(
                _module: Any,
                _inputs: tuple[Tensor, ...],
                output: Tensor,
                *,
                layer_index: int = layer_index,
                token_index: int = token_index,
            ) -> Tensor:
                patched = output.clone()
                patched[:, token_index, :] = clean_mlp[layer_index][0, token_index, :]
                return patched

            embed_handle = model.gpt_neox.embed_in.register_forward_hook(corrupt_embeddings)
            mlp_handle = layer_module.mlp.register_forward_hook(restore_output)
            try:
                with t.inference_mode():
                    patched_prob = float(
                        model(**repeated).logits[:, -1].softmax(-1)[:, target_id].mean().item()
                    )
            finally:
                mlp_handle.remove()
                embed_handle.remove()
            raw_effect[layer_index, token_index] = patched_prob - corrupt_prob
            recovery[layer_index, token_index] = (
                (patched_prob - corrupt_prob) / denominator if abs(denominator) > 1e-12 else 0.0
            )

    subject_scores = recovery[:, subject_positions[-1]]
    best_layer = int(subject_scores.argmax().item())
    tokens = [
        tokenizer.decode([token_id]).replace(" ", "_")
        for token_id in encoded["input_ids"][0, :seq_len].tolist()
    ]
    return {
        "prompt": prompt,
        "tokens": tokens,
        "subject_positions": subject_positions,
        "clean_probability": clean_prob,
        "corrupt_probability": corrupt_prob,
        "recovery": recovery.detach().cpu(),
        "raw_effect": raw_effect.detach().cpu(),
        "best_subject_layer": best_layer,
        "best_subject_recovery": float(subject_scores[best_layer].item()),
        "noise_scale": float(noise_scale),
        "seed": int(seed),
        "samples": int(samples),
    }

def independent_noise_conditions(
    noise_scales: Sequence[float],
    *,
    directions_per_scale: int = 3,
    seed: int = 0,
) -> list[tuple[float, int]]:
    """Assign one unique Gaussian direction seed to every scale condition."""
    if len(noise_scales) < 3 or directions_per_scale < 3:
        raise ValueError("Use at least three scales and three directions per scale.")
    if any(float(scale) <= 0 for scale in noise_scales):
        raise ValueError("Every noise scale must be positive.")
    return [
        (float(scale), int(seed + scale_index * directions_per_scale + direction_index))
        for scale_index, scale in enumerate(noise_scales)
        for direction_index in range(directions_per_scale)
    ]

def activation_patching_sweep(
    model: Any,
    tokenizer: Any,
    prompt: str,
    subject: str,
    target: str,
    *,
    noise_scales: Sequence[float] = (1.5, 3.0, 4.5),
    directions_per_scale: int = 3,
    seed: int = 0,
    samples: int = 4,
) -> dict[str, Any]:
    """Run layer-by-position activation patching over independent corruptions."""
    conditions = independent_noise_conditions(
        noise_scales,
        directions_per_scale=directions_per_scale,
        seed=seed,
    )
    traces = [
        activation_patching_mlp_grid(
            model,
            tokenizer,
            prompt,
            subject,
            target,
            samples=samples,
            noise_scale=float(noise_scale),
            seed=int(direction_seed),
        )
        for noise_scale, direction_seed in conditions
    ]
    recovery = t.stack([trace["recovery"] for trace in traces])
    subject_position = traces[0]["subject_positions"][-1]
    subject_recovery = recovery[:, :, subject_position]
    condition_best_layers = subject_recovery.argmax(dim=1)
    mean_recovery = recovery.mean(dim=0)
    selected_layer = int(mean_recovery[:, subject_position].argmax().item())
    selected_fraction = float((condition_best_layers == selected_layer).float().mean().item())
    return {
        "prompt": prompt,
        "tokens": traces[0]["tokens"],
        "subject_positions": traces[0]["subject_positions"],
        "condition_labels": [
            f"scale={trace['noise_scale']:g}, direction_seed={trace['seed']}" for trace in traces
        ],
        "noise_scales": [float(value) for value in noise_scales],
        "direction_seeds": [int(trace["seed"]) for trace in traces],
        "directions_per_scale": int(directions_per_scale),
        "condition_count": len(traces),
        "mean_recovery": mean_recovery,
        "std_recovery": recovery.std(dim=0, unbiased=False),
        "subject_recovery_by_condition": subject_recovery,
        "condition_best_layers": condition_best_layers,
        "selected_layer": selected_layer,
        "selected_layer_win_fraction": selected_fraction,
        "layer_zero_mean_recovery": float(mean_recovery[0, subject_position].item()),
        "layer_zero_std_recovery": float(recovery[:, 0, subject_position].std(unbiased=False).item()),
        "clean_probabilities": [trace["clean_probability"] for trace in traces],
        "corrupt_probabilities": [trace["corrupt_probability"] for trace in traces],
        "traces": traces,
    }

# %%

if MAIN:
    tests.test_activation_patching_sweep_rejects_less_than_nine_conditions(independent_noise_conditions)

if MAIN:
    tests.test_activation_patching_conditions_use_nine_independent_directions(independent_noise_conditions)

if MAIN:
    trace_demo = activation_patching_sweep(
        model,
        tokenizer,
        OPTIMIZATION_PROMPTS[0],
        SUBJECT,
        OLD_OBJECT,
        noise_scales=(1.5, 3.0, 4.5),
        directions_per_scale=3,
        seed=0,
        samples=2,
    )

if MAIN:
    display(pd.DataFrame({
        "condition": trace_demo["condition_labels"],
        "direction_seed": trace_demo["direction_seeds"],
        "best_layer": trace_demo["condition_best_layers"].tolist(),
    }))

if MAIN:
    assert len(set(trace_demo["direction_seeds"])) == 9

if MAIN:
    print({
        "selected_layer": trace_demo["selected_layer"],
        "selected_layer_win_fraction": trace_demo["selected_layer_win_fraction"],
        "layer_0_mean": trace_demo["layer_zero_mean_recovery"],
        "layer_0_std": trace_demo["layer_zero_std_recovery"],
    })

# %%

def extract_mlp_keys_and_values(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    subjects: Sequence[str],
    layer: int,
    *,
    positions: Sequence[int] | None = None,
) -> tuple[Tensor, Tensor]:
    """Extract post-activation MLP keys and complete MLP outputs at chosen positions."""
    if len(prompts) != len(subjects):
        raise ValueError("prompts and subjects must have equal length.")
    device = next(model.parameters()).device
    batch = _batch_inputs(tokenizer, prompts, device)
    if positions is None:
        positions = [locate_subject_tokens(tokenizer, p, s)[-1] for p, s in zip(prompts, subjects)]
    if len(positions) != len(prompts):
        raise ValueError("positions must provide one token index per prompt.")

    hidden_to_4h: list[Tensor] = []
    mlp_outputs: list[Tensor] = []
    block = model.gpt_neox.layers[layer]
    h1 = block.mlp.dense_h_to_4h.register_forward_hook(
        lambda _module, _inputs, output: hidden_to_4h.append(output.detach().clone())
    )
    h2 = block.mlp.register_forward_hook(
        lambda _module, _inputs, output: mlp_outputs.append(output.detach().clone())
    )
    try:
        with t.inference_mode():
            _ = model(**batch)
    finally:
        h2.remove()
        h1.remove()
    rows = t.arange(len(prompts), device=device)
    pos = t.tensor(list(positions), device=device)
    keys = block.mlp.act(hidden_to_4h[0])[rows, pos]
    values = mlp_outputs[0][rows, pos]
    return keys, values

# %%

if MAIN:
    selected_layer = trace_demo["selected_layer"]

if MAIN:
    keys, values = extract_mlp_keys_and_values(
        model, tokenizer, OPTIMIZATION_PROMPTS, [SUBJECT] * len(OPTIMIZATION_PROMPTS), selected_layer
    )

if MAIN:
    assert keys.shape == (len(OPTIMIZATION_PROMPTS), model.config.intermediate_size)

if MAIN:
    assert values.shape == (len(OPTIMIZATION_PROMPTS), model.config.hidden_size)

if MAIN:
    assert t.isfinite(keys).all() and t.isfinite(values).all()

if MAIN:
    print("keys", tuple(keys.shape), "complete MLP outputs", tuple(values.shape))

# %%

def optimize_value_delta(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    subject: str,
    targets: str | Sequence[str],
    layer: int,
    *,
    positions: Sequence[int] | None = None,
    preservation_prompts: Sequence[str] = ("The Louvre is a", "The Louvre museum has"),
    steps: int = 24,
    learning_rate: float = 0.25,
    kl_weight: float = 0.15,
    norm_weight: float = 0.02,
) -> tuple[Tensor, list[dict[str, float]]]:
    """Optimize a complete MLP output addition for single- or multi-token targets."""
    if steps < 1:
        raise ValueError("steps must be positive.")
    device = next(model.parameters()).device
    target_texts = _normalize_target_list(targets, len(prompts))
    target_ids = [target_token_ids(tokenizer, target) for target in target_texts]
    if positions is None:
        positions = [locate_subject_tokens(tokenizer, prompt, subject)[-1] for prompt in prompts]
    if len(positions) != len(prompts):
        raise ValueError("positions must provide one index per rewrite prompt.")
    preserve_positions = [
        locate_subject_tokens(tokenizer, prompt, subject)[-1] for prompt in preservation_prompts
    ]

    prompt_ids = [tokenizer.encode(prompt, add_special_tokens=False) for prompt in prompts]
    for prompt, target, prefix, continuation in zip(prompts, target_texts, prompt_ids, target_ids):
        if not prefix or tokenizer.encode(prompt + target, add_special_tokens=False) != prefix + continuation:
            raise ValueError("Prompt/target token boundary changed; use a nonempty prompt and explicit leading space.")
    # Construct teacher-forced prefixes by token ID; decoding and re-tokenizing can
    # change a boundary and silently train the wrong continuation positions.
    input_ids = [prefix + continuation[:-1] for prefix, continuation in zip(prompt_ids, target_ids)]
    input_ids += [tokenizer.encode(prompt, add_special_tokens=False) for prompt in preservation_prompts]
    all_positions = list(positions) + preserve_positions
    padded = tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt")
    batch = {name: value.to(device) for name, value in padded.items()}
    preserve_rows = t.arange(len(prompts), len(input_ids), device=device)
    preserve_lengths = batch["attention_mask"][preserve_rows].sum(dim=1) - 1
    with t.inference_mode():
        base_logits = model(**batch).logits
        base_preserve_logits = base_logits[preserve_rows, preserve_lengths].detach()

    _, base_values = extract_mlp_keys_and_values(
        model,
        tokenizer,
        list(prompts) + list(preservation_prompts),
        [subject] * len(input_ids),
        layer,
        positions=all_positions,
    )
    reference_norm = base_values[: len(prompts)].norm(dim=-1).mean().clamp_min(1e-6)
    delta = t.zeros(model.config.hidden_size, device=device, requires_grad=True)
    optimizer = t.optim.Adam([delta], lr=learning_rate)
    block = model.gpt_neox.layers[layer]
    edit_rows = t.arange(len(input_ids), device=device)
    edit_positions = t.tensor(all_positions, device=device)
    prompt_lengths = [len(prefix) for prefix in prompt_ids]

    history: list[dict[str, float]] = []
    for step in range(steps + 1):
        optimizer.zero_grad(set_to_none=True)

        def add_delta(_module: Any, _inputs: tuple[Tensor, ...], output: Tensor) -> Tensor:
            edited = output.clone()
            edited[edit_rows, edit_positions] = edited[edit_rows, edit_positions] + delta
            return edited

        handle = block.mlp.register_forward_hook(add_delta)
        try:
            logits = model(**batch).logits
        finally:
            handle.remove()
        log_probs = logits.log_softmax(dim=-1)
        sequence_nlls = []
        token_mean_nlls = []
        for row, (prompt_length, ids) in enumerate(zip(prompt_lengths, target_ids)):
            score_positions = t.arange(
                prompt_length - 1,
                prompt_length + len(ids) - 1,
                device=device,
            )
            ids_tensor = t.tensor(ids, device=device)
            token_losses = -log_probs[row, score_positions, ids_tensor]
            sequence_nlls.append(token_losses.sum())
            token_mean_nlls.append(token_losses.mean())
        nll = t.stack(token_mean_nlls).mean()
        sequence_nll = t.stack(sequence_nlls).mean()

        preserve_logits = logits[preserve_rows, preserve_lengths]
        preserve_log_probs = preserve_logits.log_softmax(dim=-1)
        base_preserve_probs = base_preserve_logits.softmax(dim=-1)
        kl = t.sum(
            base_preserve_probs * (base_preserve_logits.log_softmax(dim=-1) - preserve_log_probs),
            dim=-1,
        ).mean()
        norm_penalty = delta.square().sum() / reference_norm.square()
        loss = nll + kl_weight * kl + norm_weight * norm_penalty
        history.append(
            {
                "step": float(step),
                "loss": float(loss.detach().item()),
                "target_nll": float(nll.detach().item()),
                "target_sequence_nll": float(sequence_nll.detach().item()),
                "preservation_kl": float(kl.detach().item()),
                "delta_norm": float(delta.detach().norm().item()),
            }
        )
        if step < steps:
            loss.backward()
            optimizer.step()
    return delta.detach(), history

# %%

if MAIN:
    tests.test_value_optimizer_history_describes_returned_delta(optimize_value_delta, model, tokenizer)

if MAIN:
    demo_delta, demo_history = optimize_value_delta(
        model, tokenizer, OPTIMIZATION_PROMPTS[:2], SUBJECT, NEW_OBJECT, selected_layer, steps=3
    )

if MAIN:
    display(pd.DataFrame(demo_history))

if MAIN:
    assert demo_history[-1]["target_nll"] < demo_history[0]["target_nll"]

if MAIN:
    assert math.isfinite(demo_history[-1]["preservation_kl"])

if MAIN:
    print("demo preservation KL", demo_history[-1]["preservation_kl"])

if MAIN:
    assert all(not parameter.requires_grad for parameter in model.parameters())

# %%

def make_same_norm_random_rank_one(
    shape: Sequence[int],
    target_norm: float,
    *,
    seed: int = 0,
    device: t.device | str = "cpu",
    dtype: t.dtype = t.float32,
) -> Tensor:
    """Create a deterministic rank-one matrix with the requested Frobenius norm."""
    if len(shape) != 2 or target_norm < 0:
        raise ValueError("Expected a matrix shape and nonnegative target norm.")
    generator = t.Generator(device=device).manual_seed(seed)
    left = t.randn(int(shape[0]), generator=generator, device=device, dtype=dtype)
    right = t.randn(int(shape[1]), generator=generator, device=device, dtype=dtype)
    matrix = t.outer(left, right)
    return matrix * (target_norm / matrix.norm().clamp_min(t.finfo(dtype).eps))

@contextmanager
def temporary_mlp_weight_edit(model: Any, layer: int, delta_row_weight: Tensor) -> Iterator[None]:
    """Apply a row-convention edit to Pythia's output projection and restore exactly."""
    parameter = model.gpt_neox.layers[layer].mlp.dense_4h_to_h.weight
    if delta_row_weight.shape != parameter.T.shape:
        raise ValueError(
            f"Expected row-convention update shape {tuple(parameter.T.shape)}, "
            f"got {tuple(delta_row_weight.shape)}."
        )
    original = parameter.detach().clone()
    with t.no_grad():
        parameter.add_(delta_row_weight.T.to(device=parameter.device, dtype=parameter.dtype))
    try:
        yield
    finally:
        with t.no_grad():
            parameter.copy_(original)

# %%

if MAIN:
    tests.test_temporary_edit_restores_weights_after_errors(temporary_mlp_weight_edit)

if MAIN:
    tests.test_random_rank_one_has_exact_norm_and_rank(make_same_norm_random_rank_one)

if MAIN:
    tests.test_five_random_controls_are_seeded_distinct_and_norm_matched(make_same_norm_random_rank_one)

# %%

def qualify_known_fact_mask(base_scores: dict[str, Any]) -> Tensor:
    """Select only facts whose declared label is the base model's top-1 token."""
    mask = base_scores["label_first_top1"].bool()
    if mask.ndim != 1:
        raise ValueError("label_first_top1 must be a one-dimensional boolean tensor.")
    if not bool(mask.any()):
        raise ValueError("No candidate fact is known at top-1; factual locality is not measurable.")
    return mask

def summarize_edit_metrics(
    base: dict[str, Any],
    edited: dict[str, Any],
    known_fact_base: dict[str, Any],
    known_fact_edited: dict[str, Any],
    adversarial_base: dict[str, Any],
    adversarial_edited: dict[str, Any],
    side_base: dict[str, Tensor],
    side_edited: dict[str, Tensor],
    fluency_base: Tensor,
    fluency_edited: Tensor,
) -> dict[str, float | int]:
    """Separate rewrite efficacy, qualified retention, leakage, and side effects."""
    pairwise_before = base["new_target_log_probs"] > base["old_target_log_probs"]
    pairwise_after = edited["new_target_log_probs"] > edited["old_target_log_probs"]
    target_logprob_change = edited["new_target_log_probs"] - base["new_target_log_probs"]
    target_first_prob_change = edited["selected_probs"][:, 1] - base["selected_probs"][:, 1]

    known_mask = qualify_known_fact_mask(known_fact_base)
    known_logprob_change = (
        known_fact_edited["target_log_probs"] - known_fact_base["target_log_probs"]
    )
    known_unchanged_top1 = known_fact_edited["top_ids"] == known_fact_base["top_ids"]
    adversarial_logprob_change = (
        adversarial_edited["target_log_probs"] - adversarial_base["target_log_probs"]
    )
    adversarial_rome_change = (
        adversarial_edited["new_object_first_probs"]
        - adversarial_base["new_object_first_probs"]
    )
    strata = adversarial_base["strata"]
    if adversarial_edited["strata"] != strata:
        raise ValueError("Adversarial strata changed between base and edited evaluations.")
    leakage_mask = adversarial_base["counts_as_leakage"].bool()
    if not bool(leakage_mask.any()):
        raise ValueError("At least one adversarial row must be explicitly leakage-sensitive.")

    stratum_masks = {
        name: t.tensor([stratum == name for stratum in strata])
        for name in (
            "target_generalization",
            "context_override",
            "non_target_relation",
            "legitimate_rome_target",
        )
    }
    if any(not bool(mask.any()) for mask in stratum_masks.values()):
        raise ValueError("Every declared adversarial stratum must contain at least one row.")

    side_unchanged_top1 = side_edited["top_ids"] == side_base["top_ids"]
    side_base_prob = side_base["probs"].clamp_min(1e-12)
    side_edit_log_prob = side_edited["probs"].clamp_min(1e-12).log()
    side_kl = (side_base_prob * (side_base_prob.log() - side_edit_log_prob)).sum(dim=-1)
    return {
        "pairwise_preference_rate_before": _mean(pairwise_before),
        "pairwise_preference_rate_after": _mean(pairwise_after),
        "target_top1_rate_before": _mean(base["target_top1"]),
        "target_top1_rate_after": _mean(edited["target_top1"]),
        "full_target_greedy_rate_before": _mean(base["full_target_greedy_matches"]),
        "full_target_greedy_rate_after": _mean(edited["full_target_greedy_matches"]),
        "mean_full_target_likelihood_before": _mean(base["new_target_likelihoods"]),
        "mean_full_target_likelihood_after": _mean(edited["new_target_likelihoods"]),
        "geometric_mean_target_likelihood_before": float(
            base["new_target_log_probs"].float().mean().exp().item()
        ),
        "geometric_mean_target_likelihood_after": float(
            edited["new_target_log_probs"].float().mean().exp().item()
        ),
        "mean_target_logprob_change": _mean(target_logprob_change),
        "mean_target_first_probability_change": _mean(target_first_prob_change),
        "known_fact_candidate_count": int(known_mask.numel()),
        "known_fact_qualified_count": int(known_mask.sum().item()),
        "known_fact_top1_retention_after": _mean(known_fact_edited["label_first_top1"][known_mask]),
        "known_fact_full_greedy_retention_after": _mean(
            known_fact_edited["label_full_greedy"][known_mask]
        ),
        "known_fact_mean_label_logprob_change": _mean(known_logprob_change[known_mask]),
        "known_fact_unchanged_top1_rate": _mean(known_unchanged_top1[known_mask]),
        "adversarial_label_top1_before": _mean(adversarial_base["label_first_top1"]),
        "adversarial_label_top1_after": _mean(adversarial_edited["label_first_top1"]),
        "adversarial_mean_label_logprob_change": _mean(adversarial_logprob_change),
        "leakage_sensitive_count": int(leakage_mask.sum().item()),
        "leakage_sensitive_mean_rome_probability_change": _mean(
            adversarial_rome_change[leakage_mask]
        ),
        "target_generalization_top1_after": _mean(
            adversarial_edited["label_first_top1"][stratum_masks["target_generalization"]]
        ),
        "context_override_label_top1_after": _mean(
            adversarial_edited["label_first_top1"][stratum_masks["context_override"]]
        ),
        "context_override_mean_rome_probability_change": _mean(
            adversarial_rome_change[stratum_masks["context_override"]]
        ),
        "non_target_label_top1_after": _mean(
            adversarial_edited["label_first_top1"][stratum_masks["non_target_relation"]]
        ),
        "non_target_mean_rome_probability_change": _mean(
            adversarial_rome_change[stratum_masks["non_target_relation"]]
        ),
        "legitimate_rome_label_top1_after": _mean(
            adversarial_edited["label_first_top1"][stratum_masks["legitimate_rome_target"]]
        ),
        "legitimate_rome_probability_change": _mean(
            adversarial_rome_change[stratum_masks["legitimate_rome_target"]]
        ),
        "side_effect_top1_preservation": _mean(side_unchanged_top1),
        "side_effect_mean_kl": _mean(side_kl),
        "side_effect_max_kl": float(side_kl.max().item()),
        "fluency_mean_perplexity_before": _mean(fluency_base),
        "fluency_mean_perplexity_after": _mean(fluency_edited),
        "fluency_perplexity_ratio": float(
            (fluency_edited.float().mean() / fluency_base.float().mean()).item()
        ),
    }

# %%

if MAIN:
    tests.test_known_fact_qualification_uses_only_base_top1(qualify_known_fact_mask)

if MAIN:
    tests.test_adversarial_leakage_mask_excludes_all_legitimate_rome_targets()

if MAIN:
    tests.test_summarize_edit_metrics_does_not_conflate_pairwise_and_top1(summarize_edit_metrics)

# %%

def score_rewrite_prompts(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    *,
    old_target: str = OLD_OBJECT,
    new_target: str = NEW_OBJECT,
) -> dict[str, Any]:
    """Score old and new complete targets for every rewrite prompt."""
    old = continuation_score_table(model, tokenizer, prompts, old_target)
    new = continuation_score_table(model, tokenizer, prompts, new_target)
    return {
        "old_target": normalize_target_text(old_target),
        "new_target": normalize_target_text(new_target),
        "old_target_log_probs": old["target_log_probs"],
        "new_target_log_probs": new["target_log_probs"],
        "old_target_likelihoods": old["target_likelihoods"],
        "new_target_likelihoods": new["target_likelihoods"],
        "selected_probs": t.stack([old["first_target_probs"], new["first_target_probs"]], dim=1),
        "target_top1": new["first_target_ranks"] == 1,
        "target_first_ranks": new["first_target_ranks"],
        "target_token_top1_accuracies": new["token_top1_accuracies"],
        "full_target_greedy_matches": new["full_target_greedy_matches"],
        "top_ids": new["top_ids"],
        "probs": new["first_probs"],
    }

def score_labeled_facts(
    model: Any,
    tokenizer: Any,
    facts: Sequence[tuple[str, str] | tuple[str, str, str]],
) -> dict[str, Any]:
    """Score the declared factual continuation for every labeled prompt."""
    prompts = [row[0] for row in facts]
    targets = [row[-1] for row in facts]
    scored = continuation_score_table(model, tokenizer, prompts, targets)
    return {
        **scored,
        "label_first_top1": scored["first_target_ranks"] == 1,
        "label_full_greedy": scored["full_target_greedy_matches"],
    }

def score_adversarial_records(
    model: Any,
    tokenizer: Any,
    records: Sequence[AdversarialRecord],
) -> dict[str, Any]:
    """Score labels and Rome probability while retaining adversarial strata."""
    facts = [(record.prompt, record.target) for record in records]
    scored = score_labeled_facts(model, tokenizer, facts)
    rome_probabilities = next_token_probability_table(
        model,
        tokenizer,
        [record.prompt for record in records],
        [NEW_OBJECT],
    )["selected_probs"][:, 0]
    return {
        **scored,
        "new_object_first_probs": rome_probabilities,
        "strata": [record.stratum for record in records],
        "counts_as_leakage": t.tensor([record.counts_as_leakage for record in records]),
    }

def continuation_repetition_fraction(tokenizer: Any, text: str) -> float:
    """Return the fraction of generated token positions which repeat an earlier token."""
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        return 0.0
    return 1.0 - len(set(token_ids)) / len(token_ids)

def _rewrite_rows(
    tokenizer: Any,
    records: Sequence[PromptRecord],
    base: dict[str, Any],
    edited: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for index, record in enumerate(records):
        rows.append(
            {
                **asdict(record),
                "base_old_logprob": float(base["old_target_log_probs"][index].item()),
                "base_new_logprob": float(base["new_target_log_probs"][index].item()),
                "edited_old_logprob": float(edited["old_target_log_probs"][index].item()),
                "edited_new_logprob": float(edited["new_target_log_probs"][index].item()),
                "base_new_likelihood": float(base["new_target_likelihoods"][index].item()),
                "edited_new_likelihood": float(edited["new_target_likelihoods"][index].item()),
                "base_target_rank": int(base["target_first_ranks"][index].item()),
                "edited_target_rank": int(edited["target_first_ranks"][index].item()),
                "base_top_token": tokenizer.decode([int(base["top_ids"][index].item())]),
                "edited_top_token": tokenizer.decode([int(edited["top_ids"][index].item())]),
                "pairwise_after": bool(
                    edited["new_target_log_probs"][index] > edited["old_target_log_probs"][index]
                ),
                "target_top1_after": bool(edited["target_top1"][index]),
            }
        )
    return rows

def _labeled_rows(
    tokenizer: Any,
    facts: Sequence[tuple[str, str] | tuple[str, str, str]],
    base: dict[str, Any],
    edited: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for index, fact in enumerate(facts):
        rows.append(
            {
                "prompt": fact[0],
                "subject": fact[1] if len(fact) == 3 else None,
                "label": normalize_target_text(fact[-1]),
                "base_label_logprob": float(base["target_log_probs"][index].item()),
                "edited_label_logprob": float(edited["target_log_probs"][index].item()),
                "base_label_rank": int(base["first_target_ranks"][index].item()),
                "edited_label_rank": int(edited["first_target_ranks"][index].item()),
                "base_top_token": tokenizer.decode([int(base["top_ids"][index].item())]),
                "edited_top_token": tokenizer.decode([int(edited["top_ids"][index].item())]),
                "base_full_target_greedy": bool(base["label_full_greedy"][index]),
                "edited_full_target_greedy": bool(edited["label_full_greedy"][index]),
                "base_qualified_known_fact": bool(base["label_first_top1"][index]),
            }
        )
    return rows

def _adversarial_rows(
    tokenizer: Any,
    records: Sequence[AdversarialRecord],
    base: dict[str, Any],
    edited: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = _labeled_rows(
        tokenizer,
        [(record.prompt, record.target) for record in records],
        base,
        edited,
    )
    for row, record, base_rome, edited_rome in zip(
        rows,
        records,
        base["new_object_first_probs"],
        edited["new_object_first_probs"],
    ):
        row.update(
            {
                **asdict(record),
                "base_rome_probability": float(base_rome.item()),
                "edited_rome_probability": float(edited_rome.item()),
                "rome_probability_change": float((edited_rome - base_rome).item()),
            }
        )
    return rows

def evaluate_weight_edit(
    model: Any,
    tokenizer: Any,
    *,
    layer: int,
    delta_weight: Tensor,
    rewrite_scorer: Callable[..., dict[str, Any]],
    fact_scorer: Callable[..., dict[str, Any]],
    adversarial_scorer: Callable[..., dict[str, Any]],
    metric_summarizer: Callable[..., dict[str, float | int]],
    paraphrase_records: Sequence[PromptRecord] = HELDOUT_PARAPHRASE_RECORDS,
    include_generations: bool = False,
) -> dict[str, Any]:
    """Evaluate rewrites, base-qualified facts, adversarial strata, KL, and perplexity."""
    paraphrases = [record.prompt for record in paraphrase_records]
    base = rewrite_scorer(model, tokenizer, paraphrases)
    known_fact_base = fact_scorer(model, tokenizer, KNOWN_FACT_CANDIDATES)
    adversarial_base = adversarial_scorer(model, tokenizer, ADVERSARIAL_RECORDS)
    side_base = next_token_distribution_table(model, tokenizer, SIDE_EFFECT_PROMPTS)
    fluency_base = text_perplexities(model, tokenizer, FLUENCY_TEXTS)
    base_generations = generate_safe_continuations(model, tokenizer) if include_generations else []

    with temporary_mlp_weight_edit(model, layer, delta_weight):
        edited = rewrite_scorer(model, tokenizer, paraphrases)
        known_fact_edited = fact_scorer(model, tokenizer, KNOWN_FACT_CANDIDATES)
        adversarial_edited = adversarial_scorer(model, tokenizer, ADVERSARIAL_RECORDS)
        side_edited = next_token_distribution_table(model, tokenizer, SIDE_EFFECT_PROMPTS)
        fluency_edited = text_perplexities(model, tokenizer, FLUENCY_TEXTS)
        edited_generations = generate_safe_continuations(model, tokenizer) if include_generations else []

    metrics = metric_summarizer(
        base,
        edited,
        known_fact_base,
        known_fact_edited,
        adversarial_base,
        adversarial_edited,
        side_base,
        side_edited,
        fluency_base,
        fluency_edited,
    )
    generation_rows = [
        {
            "prompt": prompt,
            "base": before,
            "edited": after,
            "base_repetition_fraction": continuation_repetition_fraction(tokenizer, before),
            "edited_repetition_fraction": continuation_repetition_fraction(tokenizer, after),
        }
        for prompt, before, after in zip(
            SAFE_GENERATION_PROMPTS, base_generations, edited_generations
        )
    ]
    if generation_rows:
        metrics["generation_probe_count"] = len(generation_rows)
        metrics["generation_repetition_failure_count"] = sum(
            row["edited_repetition_fraction"] > row["base_repetition_fraction"] + 0.10
            for row in generation_rows
        )
        metrics["localized_louvre_repetition_before"] = generation_rows[0][
            "base_repetition_fraction"
        ]
        metrics["localized_louvre_repetition_after"] = generation_rows[0][
            "edited_repetition_fraction"
        ]
    return {
        "metrics": metrics,
        "paraphrase_rows": _rewrite_rows(tokenizer, paraphrase_records, base, edited),
        "known_fact_rows": _labeled_rows(
            tokenizer, KNOWN_FACT_CANDIDATES, known_fact_base, known_fact_edited
        ),
        "adversarial_rows": _adversarial_rows(
            tokenizer, ADVERSARIAL_RECORDS, adversarial_base, adversarial_edited
        ),
        "generation_rows": generation_rows,
        "suite_sizes": {
            "paraphrases": len(paraphrases),
            "known_fact_candidates": len(KNOWN_FACT_CANDIDATES),
            "known_fact_qualified": int(qualify_known_fact_mask(known_fact_base).sum().item()),
            "adversarial": len(ADVERSARIAL_RECORDS),
            "side_effect": len(SIDE_EFFECT_PROMPTS),
            "fluency": len(FLUENCY_TEXTS),
        },
    }

# %%

if MAIN:
    tests.test_covariance_and_evaluation_sets_are_disjoint_and_auditable()

if MAIN:
    tests.test_real_evaluation_sizes_are_not_one_prompt_demos()

if MAIN:
    tests.test_repetition_fraction_exposes_local_generation_collapse(tokenizer, continuation_repetition_fraction)

if MAIN:
    tests.test_pinned_model_has_nontrivial_base_known_fact_denominator(model, tokenizer)

if MAIN:
    tests.test_real_evaluator_zero_edit_has_immediate_numerical_contract(
        evaluate_weight_edit,
        model,
        tokenizer,
        score_rewrite_prompts,
        score_labeled_facts,
        score_adversarial_records,
        summarize_edit_metrics,
    )

if MAIN:
    display(pd.DataFrame([record.__dict__ for record in HELDOUT_PARAPHRASE_RECORDS]))

if MAIN:
    display(pd.DataFrame([record.__dict__ for record in ADVERSARIAL_RECORDS]))

# %%

def build_course_rome_like_update(
    model: Any,
    tokenizer: Any,
    *,
    layer: int,
    position_mode: str = "subject",
    targets: str | Sequence[str] = NEW_OBJECT,
    steps: int = 24,
    ridge: float = 1e-2,
) -> tuple[Tensor, dict[str, Any]]:
    """Compose the course value objective, covariance bank, and affine rank-one write."""
    positions = _positions_for_mode(
        tokenizer,
        OPTIMIZATION_PROMPTS,
        [SUBJECT] * len(OPTIMIZATION_PROMPTS),
        position_mode,
    )
    value_delta, history = optimize_value_delta(
        model,
        tokenizer,
        OPTIMIZATION_PROMPTS,
        SUBJECT,
        targets,
        layer,
        positions=positions,
        steps=steps,
    )
    keys, values = extract_mlp_keys_and_values(
        model,
        tokenizer,
        OPTIMIZATION_PROMPTS,
        [SUBJECT] * len(OPTIMIZATION_PROMPTS),
        layer,
        positions=positions,
    )
    covariance_prompts = [row[0] for row in COVARIANCE_FACTS]
    covariance_subjects = [row[1] for row in COVARIANCE_FACTS]
    covariance_positions = _positions_for_mode(
        tokenizer,
        covariance_prompts,
        covariance_subjects,
        position_mode,
    )
    key_bank, _ = extract_mlp_keys_and_values(
        model,
        tokenizer,
        covariance_prompts,
        covariance_subjects,
        layer,
        positions=covariance_positions,
    )

    key = keys.mean(dim=0)
    current_module_output = values.mean(dim=0)
    desired_module_output = current_module_output + value_delta
    projection = model.gpt_neox.layers[layer].mlp.dense_4h_to_h
    weight_row = projection.weight.detach().T
    output_bias = projection.bias.detach() if projection.bias is not None else None
    update = course_rome_like_rank_one_update(
        weight_row,
        key,
        desired_module_output,
        key_bank,
        output_bias=output_bias,
        ridge=ridge,
    )
    with temporary_mlp_weight_edit(model, layer, update):
        with t.inference_mode():
            actual_module_output = projection(key)
    write_error = (actual_module_output - desired_module_output).norm()
    singular_values = t.linalg.svdvals(update.float())
    return update.detach(), {
        "method_scope": "course_rome_like_rank_one_edit",
        "official_rome_parity_established": False,
        "key": key.detach(),
        "key_bank_size": int(key_bank.shape[0]),
        "current_module_output": current_module_output.detach(),
        "desired_module_output": desired_module_output.detach(),
        "actual_module_output_after_write": actual_module_output.detach(),
        "value_delta": value_delta.detach(),
        "optimization_history": history,
        "write_error": float(write_error.item()),
        "update_norm": float(update.norm().item()),
        "singular_values": singular_values.detach().cpu(),
        "effective_rank": int(t.linalg.matrix_rank(update.float(), tol=1e-5).item()),
        "position_mode": position_mode,
        "targets": _normalize_target_list(targets, len(OPTIMIZATION_PROMPTS)),
    }

def run_course_rome_like_experiment(
    *,
    implementation: LearnerFunctions,
    device: str | t.device = "cpu",
    steps: int = 12,
    patching_samples: int = 3,
) -> dict[str, Any]:
    """Run the one-fact lab entirely through explicit learner-written functions."""
    t.manual_seed(0)
    model, tokenizer = load_pinned_model(device=device)
    trace = implementation.activation_patching_sweep(
        model,
        tokenizer,
        OPTIMIZATION_PROMPTS[0],
        SUBJECT,
        OLD_OBJECT,
        noise_scales=(1.5, 3.0, 4.5),
        directions_per_scale=3,
        seed=0,
        samples=patching_samples,
    )
    selected_layer = int(trace["selected_layer"])

    updates: dict[int, Tensor] = {}
    details_by_layer: dict[int, dict[str, Any]] = {}
    for layer in range(len(model.gpt_neox.layers)):
        update, details = implementation.build_update(
            model,
            tokenizer,
            layer=layer,
            steps=steps,
        )
        updates[layer] = update
        details_by_layer[layer] = details

    primary_update = updates[selected_layer]
    primary_details = details_by_layer[selected_layer]
    primary_norm = float(primary_update.norm().item())

    evaluator_kwargs = {
        "rewrite_scorer": implementation.rewrite_scorer,
        "fact_scorer": implementation.fact_scorer,
        "adversarial_scorer": implementation.adversarial_scorer,
        "metric_summarizer": implementation.metric_summarizer,
    }

    layer_sweep = []
    for layer in range(len(model.gpt_neox.layers)):
        matched_update = match_update_norm(updates[layer], primary_norm)
        evaluation = implementation.evaluate_edit(
            model,
            tokenizer,
            layer=layer,
            delta_weight=matched_update,
            include_generations=False,
            **evaluator_kwargs,
        )
        layer_sweep.append(
            {
                "layer": layer,
                "norm_matched": True,
                "native_update_norm": details_by_layer[layer]["update_norm"],
                "applied_update_norm": float(matched_update.norm().item()),
                "native_write_error": details_by_layer[layer]["write_error"],
                **evaluation["metrics"],
            }
        )

    primary_evaluation = implementation.evaluate_edit(
        model,
        tokenizer,
        layer=selected_layer,
        delta_weight=primary_update,
        include_generations=True,
        **evaluator_kwargs,
    )

    wrong_native, wrong_details = implementation.build_update(
        model,
        tokenizer,
        layer=selected_layer,
        position_mode="last_prompt_token",
        steps=steps,
    )
    wrong_update = match_update_norm(wrong_native, primary_norm)
    wrong_evaluation = implementation.evaluate_edit(
        model,
        tokenizer,
        layer=selected_layer,
        delta_weight=wrong_update,
        **evaluator_kwargs,
    )

    shuffled_native, shuffled_details = implementation.build_update(
        model,
        tokenizer,
        layer=selected_layer,
        targets=SHUFFLED_CONTROL_TARGETS,
        steps=steps,
    )
    shuffled_update = match_update_norm(shuffled_native, primary_norm)
    shuffled_evaluation = implementation.evaluate_edit(
        model,
        tokenizer,
        layer=selected_layer,
        delta_weight=shuffled_update,
        **evaluator_kwargs,
    )

    random_evaluations = []
    for seed in range(5):
        random_update = implementation.random_rank_one(
            primary_update.shape,
            primary_norm,
            seed=seed,
            device=primary_update.device,
            dtype=primary_update.dtype,
        )
        random_evaluations.append(
            {
                "seed": seed,
                "update_norm": float(random_update.norm().item()),
                "evaluation": implementation.evaluate_edit(
                    model,
                    tokenizer,
                    layer=selected_layer,
                    delta_weight=random_update,
                    **evaluator_kwargs,
                ),
            }
        )

    base_evaluation = implementation.evaluate_edit(
        model,
        tokenizer,
        layer=selected_layer,
        delta_weight=t.zeros_like(primary_update),
        **evaluator_kwargs,
    )
    metrics = {
        "course_rome_like": primary_evaluation["metrics"],
        "unedited_base": base_evaluation["metrics"],
        "wrong_position_norm_matched": wrong_evaluation["metrics"],
        "shuffled_targets_norm_matched": shuffled_evaluation["metrics"],
        **{
            f"random_rank_one_seed_{row['seed']}": row["evaluation"]["metrics"]
            for row in random_evaluations
        },
    }
    successful_layers = [
        row["layer"]
        for row in layer_sweep
        if row["pairwise_preference_rate_after"] >= 0.5
    ]
    wrong_position_success = wrong_evaluation["metrics"]["pairwise_preference_rate_after"] >= 0.5
    localization_result = {
        "activation_patching_selected_layer": selected_layer,
        "norm_matched_successful_layers": successful_layers,
        "wrong_position_pairwise_preference_rate": wrong_evaluation["metrics"][
            "pairwise_preference_rate_after"
        ],
        "wrong_position_success": wrong_position_success,
        "negative_localization_result": len(successful_layers) > 1 or wrong_position_success,
        "supports_privileged_layer_claim": False,
    }
    return {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "edit": {"subject": SUBJECT, "old_object": OLD_OBJECT, "new_object": NEW_OBJECT},
        "paraphrase_provenance": PARAPHRASE_PROVENANCE,
        "suite_sizes": primary_evaluation["suite_sizes"],
        "covariance_context_count": len(COVARIANCE_FACTS),
        "method_scope": "course_rome_like_rank_one_edit",
        "official_rome_parity_established": False,
        "missing_official_parity_components": [
            "Wikipedia-derived covariance statistics",
            "random-prefix key and value contexts",
            "released ROME value-objective parity",
        ],
        "trace": trace,
        "selected_layer": selected_layer,
        "primary_update": primary_update,
        "primary_details": primary_details,
        "primary_evaluation": primary_evaluation,
        "layer_sweep": layer_sweep,
        "localization_result": localization_result,
        "control_details": {
            "wrong_position": {
                "position_mode": "last_prompt_token",
                "native_update_norm": wrong_details["update_norm"],
                "applied_update_norm": float(wrong_update.norm().item()),
            },
            "shuffled_targets": {
                "targets": SHUFFLED_CONTROL_TARGETS,
                "native_update_norm": shuffled_details["update_norm"],
                "applied_update_norm": float(shuffled_update.norm().item()),
            },
            "random_updates": [
                {"seed": row["seed"], "applied_update_norm": row["update_norm"]}
                for row in random_evaluations
            ],
        },
        "metrics": metrics,
    }

# %%

if MAIN:
    tests.test_real_update_builder_has_immediate_numerical_contract(
        build_course_rome_like_update, model, tokenizer
    )

# %%

if MAIN:
    learner_functions = LearnerFunctions(
        activation_patching_sweep=activation_patching_sweep,
        build_update=build_course_rome_like_update,
        evaluate_edit=evaluate_weight_edit,
        rewrite_scorer=score_rewrite_prompts,
        fact_scorer=score_labeled_facts,
        adversarial_scorer=score_adversarial_records,
        metric_summarizer=summarize_edit_metrics,
        random_rank_one=make_same_norm_random_rank_one,
    )

if MAIN:
    tests.check_learner_bundle_uses_notebook_functions(learner_functions)

if MAIN:
    result = run_course_rome_like_experiment(
        implementation=learner_functions,
        device=device,
        steps=12,
        patching_samples=3,
    )

if MAIN:
    primary_metrics = result["metrics"]["course_rome_like"]

if MAIN:
    print("NEGATIVE LOCALIZATION RESULT", result["localization_result"])

if MAIN:
    assert result["localization_result"]["negative_localization_result"]

if MAIN:
    display(pd.DataFrame(result["layer_sweep"])[[
        "layer",
        "pairwise_preference_rate_after",
        "target_top1_rate_after",
        "mean_full_target_likelihood_after",
        "known_fact_top1_retention_after",
        "side_effect_mean_kl",
        "fluency_perplexity_ratio",
        "native_update_norm",
        "applied_update_norm",
        "native_write_error",
    ]])

if MAIN:
    display(pd.DataFrame(result["metrics"]).T[[
        "pairwise_preference_rate_after",
        "target_top1_rate_after",
        "mean_full_target_likelihood_after",
        "known_fact_top1_retention_after",
        "leakage_sensitive_mean_rome_probability_change",
        "side_effect_mean_kl",
        "fluency_perplexity_ratio",
    ]])

if MAIN:
    print({
        "selected_layer": result["selected_layer"],
        "condition_best_layers": result["trace"]["condition_best_layers"].tolist(),
        "independent_direction_seeds": result["trace"]["direction_seeds"],
        "layer_0_mean": result["trace"]["layer_zero_mean_recovery"],
        "layer_0_std": result["trace"]["layer_zero_std_recovery"],
        "actual_affine_write_error": result["primary_details"]["write_error"],
        "effective_rank": result["primary_details"]["effective_rank"],
        "value_optimizer_preservation_KL": result["primary_details"]["optimization_history"][-1]["preservation_kl"],
        "value_optimizer_KL_is_large": result["primary_details"]["optimization_history"][-1]["preservation_kl"] > 0.1,
    })

if MAIN:
    paraphrase_table = pd.DataFrame(result["primary_evaluation"]["paraphrase_rows"])

if MAIN:
    display(paraphrase_table[[
        "prompt_id", "family", "prompt", "base_top_token", "edited_top_token",
        "base_target_rank", "edited_target_rank", "base_new_likelihood",
        "edited_new_likelihood", "pairwise_after", "target_top1_after",
    ]])

if MAIN:
    known_fact_table = pd.DataFrame(result["primary_evaluation"]["known_fact_rows"])

if MAIN:
    display(known_fact_table[known_fact_table["base_qualified_known_fact"]])

if MAIN:
    adversarial_table = pd.DataFrame(result["primary_evaluation"]["adversarial_rows"])

if MAIN:
    display(adversarial_table[[
        "prompt_id", "stratum", "counts_as_leakage", "prompt", "label",
        "base_label_rank", "edited_label_rank", "rome_probability_change",
    ]])

if MAIN:
    display(pd.DataFrame(result["primary_evaluation"]["generation_rows"]))

if MAIN:
    print({
        "localized_louvre_repetition_before": primary_metrics["localized_louvre_repetition_before"],
        "localized_louvre_repetition_after": primary_metrics["localized_louvre_repetition_after"],
        "generation_repetition_failure_count": primary_metrics["generation_repetition_failure_count"],
        "scope": "one localized generation failure; not a general fluency conclusion",
        "leakage_only_rome_probability_change": primary_metrics["leakage_sensitive_mean_rome_probability_change"],
    })

if MAIN:
    reference.plot_signature_result(result, str(assets_dir / "rome_model_editing_signature.png"))

if MAIN:
    plt.show()

if MAIN:
    summary = reference.serializable_experiment_summary(result)

if MAIN:
    (section_dir / "expected_outputs" / "reference_metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

# %%

def score_fixed_louvre_edit(
    prompt: str,
    target: str = NEW_OBJECT,
    scales: Sequence[float] = (0.0, 0.5, 1.0, 1.5),
) -> pd.DataFrame:
    rows = []
    for scale in scales:
        with temporary_mlp_weight_edit(
            model,
            result["selected_layer"],
            result["primary_update"] * scale,
        ):
            scored = continuation_score_table(model, tokenizer, [prompt], [target])
        rows.append({
            "scale": scale,
            "fixed_edit": "Louvre -> Rome",
            "scored_target_only": target,
            "target_token_count": int(scored["target_token_counts"][0]),
            "full_target_likelihood": float(scored["target_likelihoods"][0]),
            "first_target_rank": int(scored["first_target_ranks"][0]),
            "full_target_greedy_match": bool(scored["full_target_greedy_matches"][0]),
        })
    return pd.DataFrame(rows)

if MAIN:
    play = score_fixed_louvre_edit("A first-time visitor can see the Louvre in")

if MAIN:
    display(play)

if MAIN:
    play.plot(x="scale", y="full_target_likelihood", marker="o")

if MAIN:
    plt.show()

if MAIN:
    display(score_fixed_louvre_edit("Times Square is in", target=" New York", scales=(0.0, 1.0)))

# %%

if MAIN:
    tests.test_smoke_contract_is_exact(reference.run_smoke_test)

if MAIN:
    tests.test_model_and_revision_are_pinned()

if MAIN:
    tests.test_section_local_release_report_generation_inputs_are_strict()

if MAIN:
    tests.test_notebook_contract()

# %%
