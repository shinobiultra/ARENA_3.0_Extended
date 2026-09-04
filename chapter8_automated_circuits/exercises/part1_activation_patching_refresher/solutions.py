# %%
"""Reference implementations for [8.1] Activation Patching Refresher."""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch as t

chapter = "chapter8_automated_circuits"
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

MAIN = __name__ == "__main__"

TOK_BOS = 0
TOK_RED = 1
TOK_BLUE = 2
TOK_QUERY = 3
TOK_MASK = 4
TOKEN_NAMES = ("<BOS>", "RED", "BLUE", "QUERY", "?")
POSITION_LABELS = ("<BOS>", "source", "distractor", "query", "answer")
LAYER_LABELS = ("embed", "route", "readout")
SOURCE_POS = 1
DISTRACTOR_POS = 2
QUERY_POS = 3
ANSWER_POS = 4
ROUTE_CELLS = ((0, SOURCE_POS), (1, QUERY_POS), (2, ANSWER_POS))

TL_GELU1L_MODEL_NAME = "gelu-1l"
TL_GELU1L_HF_ID = "NeelNanda/GELU_1L512W_C4_Code"
TL_GELU1L_REVISION = "bddc0e332f0ae84279e6a6a45d91b314899e1603"
TL_GELU1L_TOKENIZER_ID = "NeelNanda/gpt-neox-tokenizer-digits"
TL_GELU1L_TOKENIZER_REVISION = "0f6671571a20be9756b9991d978047c03b75e749"
TL_PATCH_HOOK_NAME = "blocks.0.hook_resid_post"
TL_CLEAN_PROMPT = "The cat sat on the"
TL_CORRUPT_PROMPT = "The bird flew over the"
TL_BNB_CUDA_OVERRIDE = "130"


@dataclass(frozen=True)
class ToyRun:
    logits: t.Tensor
    cache: t.Tensor


@dataclass(frozen=True)
class LocalizationReport:
    top_cells: tuple[tuple[int, int], ...]
    route_cells: tuple[tuple[int, int], ...]
    topk_overlap: float
    route_mean: float
    off_route_max: float
    separation: float


def make_copy_task_pair() -> tuple[t.Tensor, t.Tensor]:
    """Return clean RED and corrupt BLUE prompts for the two-hop copy task."""

    clean = t.tensor([TOK_BOS, TOK_RED, TOK_BLUE, TOK_QUERY, TOK_MASK])
    corrupt = t.tensor([TOK_BOS, TOK_BLUE, TOK_RED, TOK_QUERY, TOK_MASK])
    return clean, corrupt


def answer_logit_diff(
    logits: t.Tensor,
    *,
    positive_token_id: int = TOK_RED,
    negative_token_id: int = TOK_BLUE,
) -> float:
    """Return mean positive-minus-negative logit difference."""

    if logits.ndim < 1 or logits.shape[-1] == 0:
        raise ValueError("logits must have a nonempty vocabulary dimension.")
    if positive_token_id == negative_token_id:
        raise ValueError("positive_token_id and negative_token_id must differ.")
    if not 0 <= positive_token_id < logits.shape[-1]:
        raise ValueError("positive_token_id is out of range.")
    if not 0 <= negative_token_id < logits.shape[-1]:
        raise ValueError("negative_token_id is out of range.")
    if not t.isfinite(logits).all():
        raise ValueError("logits must be finite.")
    return float(
        (logits[..., positive_token_id] - logits[..., negative_token_id])
        .float()
        .mean()
        .item()
    )


def recovery_fraction(
    *,
    clean_metric: float,
    corrupt_metric: float,
    patched_metric: float,
) -> float:
    """Calibrate a patched metric so corrupt is 0 and clean is 1."""

    values = t.tensor([clean_metric, corrupt_metric, patched_metric], dtype=t.float64)
    if not t.isfinite(values).all():
        raise ValueError("clean, corrupt, and patched metrics must be finite.")
    gap = clean_metric - corrupt_metric
    if gap == 0:
        raise ValueError("clean_metric and corrupt_metric must differ.")
    return float((patched_metric - corrupt_metric) / gap)


def _validate_copy_tokens(tokens: t.Tensor) -> None:
    if tokens.shape != (len(POSITION_LABELS),):
        raise ValueError(f"tokens must have shape ({len(POSITION_LABELS)},).")
    if tokens.dtype != t.long:
        raise ValueError("tokens must use torch.long ids.")
    if int(tokens[SOURCE_POS]) not in (TOK_RED, TOK_BLUE):
        raise ValueError("source position must contain RED or BLUE.")
    if int(tokens[DISTRACTOR_POS]) not in (TOK_RED, TOK_BLUE):
        raise ValueError("distractor position must contain RED or BLUE.")
    if int(tokens[QUERY_POS]) != TOK_QUERY or int(tokens[ANSWER_POS]) != TOK_MASK:
        raise ValueError("query and answer positions must contain QUERY and ?.")


def run_causal_copy_model(
    tokens: t.Tensor,
    *,
    patch_layer: int | None = None,
    patch_position: int | None = None,
    donor_cache: t.Tensor | None = None,
) -> ToyRun:
    """Run the exact source-to-query-to-answer routing model."""

    _validate_copy_tokens(tokens)
    patch_requested = patch_layer is not None or patch_position is not None or donor_cache is not None
    if patch_requested:
        if patch_layer is None or patch_position is None or donor_cache is None:
            raise ValueError("patch_layer, patch_position, and donor_cache are required together.")
        if donor_cache.shape != (len(LAYER_LABELS), len(POSITION_LABELS), 2):
            raise ValueError("donor_cache has the wrong shape.")
        if not 0 <= patch_layer < len(LAYER_LABELS):
            raise ValueError("patch_layer is out of range.")
        if not 0 <= patch_position < len(POSITION_LABELS):
            raise ValueError("patch_position is out of range.")

    def maybe_patch(resid: t.Tensor, layer: int) -> t.Tensor:
        if patch_requested and layer == patch_layer:
            resid = resid.clone()
            resid[patch_position] = donor_cache[layer, patch_position]
        return resid

    resid_0 = t.zeros((len(POSITION_LABELS), 2), dtype=t.float32)
    resid_0[tokens == TOK_RED, 0] = 1.0
    resid_0[tokens == TOK_BLUE, 1] = 1.0
    resid_0 = maybe_patch(resid_0, 0)

    resid_1 = resid_0.clone()
    resid_1[QUERY_POS] = resid_0[SOURCE_POS]
    resid_1 = maybe_patch(resid_1, 1)

    resid_2 = resid_1.clone()
    resid_2[ANSWER_POS] = resid_1[QUERY_POS]
    resid_2 = maybe_patch(resid_2, 2)

    cache = t.stack([resid_0, resid_1, resid_2])
    logits = t.zeros(len(TOKEN_NAMES), dtype=t.float32)
    logits[TOK_RED] = 2.0 * resid_2[ANSWER_POS, 0]
    logits[TOK_BLUE] = 2.0 * resid_2[ANSWER_POS, 1]
    return ToyRun(logits=logits, cache=cache)


def patch_residual_cell(
    recipient_tokens: t.Tensor,
    donor_cache: t.Tensor,
    *,
    layer: int,
    position: int,
) -> ToyRun:
    """Patch one donor residual cell into a fresh recipient run."""

    return run_causal_copy_model(
        recipient_tokens,
        patch_layer=layer,
        patch_position=position,
        donor_cache=donor_cache,
    )


def denoising_patch_sweep(
    clean_tokens: t.Tensor,
    corrupt_tokens: t.Tensor,
    *,
    donor_cache: t.Tensor | None = None,
) -> t.Tensor:
    """Patch every layer-position cell into corrupt and return recovery fractions."""

    clean_run = run_causal_copy_model(clean_tokens)
    corrupt_run = run_causal_copy_model(corrupt_tokens)
    donor_cache = clean_run.cache if donor_cache is None else donor_cache
    clean_metric = answer_logit_diff(clean_run.logits)
    corrupt_metric = answer_logit_diff(corrupt_run.logits)
    scores = t.empty((len(LAYER_LABELS), len(POSITION_LABELS)), dtype=t.float32)
    for layer in range(scores.shape[0]):
        for position in range(scores.shape[1]):
            patched = patch_residual_cell(
                corrupt_tokens,
                donor_cache,
                layer=layer,
                position=position,
            )
            scores[layer, position] = recovery_fraction(
                clean_metric=clean_metric,
                corrupt_metric=corrupt_metric,
                patched_metric=answer_logit_diff(patched.logits),
            )
    return scores


def noising_patch_sweep(clean_tokens: t.Tensor, corrupt_tokens: t.Tensor) -> t.Tensor:
    """Patch corrupt cells into clean and return the fraction of clean behavior destroyed."""

    clean_run = run_causal_copy_model(clean_tokens)
    corrupt_run = run_causal_copy_model(corrupt_tokens)
    clean_metric = answer_logit_diff(clean_run.logits)
    corrupt_metric = answer_logit_diff(corrupt_run.logits)
    gap = clean_metric - corrupt_metric
    scores = t.empty((len(LAYER_LABELS), len(POSITION_LABELS)), dtype=t.float32)
    for layer in range(scores.shape[0]):
        for position in range(scores.shape[1]):
            patched = patch_residual_cell(
                clean_tokens,
                corrupt_run.cache,
                layer=layer,
                position=position,
            )
            patched_metric = answer_logit_diff(patched.logits)
            scores[layer, position] = (clean_metric - patched_metric) / gap
    return scores


def make_wrong_position_donor(clean_cache: t.Tensor) -> t.Tensor:
    """Use the clean distractor activation as a shape-matched donor everywhere."""

    expected_shape = (len(LAYER_LABELS), len(POSITION_LABELS), 2)
    if clean_cache.shape != expected_shape:
        raise ValueError(f"clean_cache must have shape {expected_shape}.")
    return clean_cache[:, DISTRACTOR_POS : DISTRACTOR_POS + 1].expand_as(clean_cache).clone()


def localization_report(
    patch_scores: t.Tensor,
    *,
    route_cells: tuple[tuple[int, int], ...] = ROUTE_CELLS,
) -> LocalizationReport:
    """Measure top-k recovery and separation from all off-route cells."""

    expected_shape = (len(LAYER_LABELS), len(POSITION_LABELS))
    if patch_scores.shape != expected_shape:
        raise ValueError(f"patch_scores must have shape {expected_shape}.")
    if not t.isfinite(patch_scores).all():
        raise ValueError("patch_scores must be finite.")
    if not route_cells:
        raise ValueError("route_cells must be nonempty.")
    flat = patch_scores.flatten()
    route_indices: list[int] = []
    for layer, position in route_cells:
        if not 0 <= layer < patch_scores.shape[0] or not 0 <= position < patch_scores.shape[1]:
            raise ValueError("route cell is out of range.")
        route_indices.append(layer * patch_scores.shape[1] + position)
    top_indices = flat.topk(k=len(route_indices)).indices.tolist()
    top_cells = tuple((i // patch_scores.shape[1], i % patch_scores.shape[1]) for i in top_indices)
    route_set = set(route_cells)
    topk_overlap = len(set(top_cells) & route_set) / len(route_set)
    route_mask = t.zeros_like(patch_scores, dtype=t.bool)
    for cell in route_cells:
        route_mask[cell] = True
    route_mean = float(patch_scores[route_mask].mean().item())
    off_route_max = float(patch_scores[~route_mask].max().item())
    return LocalizationReport(
        top_cells=top_cells,
        route_cells=route_cells,
        topk_overlap=topk_overlap,
        route_mean=route_mean,
        off_route_max=off_route_max,
        separation=route_mean - off_route_max,
    )


def run_toy_signature_result() -> dict:
    """Return exact metrics backing the learner-generated signature figure."""

    clean_tokens, corrupt_tokens = make_copy_task_pair()
    clean_run = run_causal_copy_model(clean_tokens)
    corrupt_run = run_causal_copy_model(corrupt_tokens)
    denoising = denoising_patch_sweep(clean_tokens, corrupt_tokens)
    noising = noising_patch_sweep(clean_tokens, corrupt_tokens)
    wrong_donor = make_wrong_position_donor(clean_run.cache)
    donor_control = denoising_patch_sweep(
        clean_tokens,
        corrupt_tokens,
        donor_cache=wrong_donor,
    )
    report = localization_report(denoising)
    return {
        "claim": "Activation patching exactly recovers the planted source-to-query-to-answer route.",
        "clean_metric": answer_logit_diff(clean_run.logits),
        "corrupt_metric": answer_logit_diff(corrupt_run.logits),
        "denoising_scores": denoising.tolist(),
        "noising_scores": noising.tolist(),
        "wrong_position_donor_scores": donor_control.tolist(),
        "top_cells": [list(cell) for cell in report.top_cells],
        "route_cells": [list(cell) for cell in report.route_cells],
        "topk_overlap": report.topk_overlap,
        "route_mean": report.route_mean,
        "off_route_max": report.off_route_max,
        "separation": report.separation,
        "wrong_position_donor_max": float(donor_control.abs().max().item()),
        "denoising_noising_max_error": float((denoising - noising).abs().max().item()),
        "exact_ground_truth_passed": bool(
            report.topk_overlap == 1.0
            and report.separation == 1.0
            and donor_control.abs().max().item() == 0.0
            and t.equal(denoising, noising)
        ),
    }


def run_smoke_test(cpu: bool = True) -> dict:
    _ = cpu
    result = run_toy_signature_result()
    passed = bool(result["exact_ground_truth_passed"])
    return {
        **result,
        "contract_passed": passed,
        "tests_passed": passed,
        "accepted": passed,
    }


def _load_gelu1l_model_on_cuda():
    os.environ.setdefault("BNB_CUDA_VERSION", TL_BNB_CUDA_OVERRIDE)
    logging.getLogger("bitsandbytes.cextension").setLevel(logging.ERROR)
    from transformer_lens import HookedTransformer
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        TL_GELU1L_TOKENIZER_ID,
        revision=TL_GELU1L_TOKENIZER_REVISION,
    )
    return HookedTransformer.from_pretrained(
        TL_GELU1L_MODEL_NAME,
        device="cuda",
        dtype="float32",
        revision=TL_GELU1L_REVISION,
        tokenizer=tokenizer,
    )


def run_transformerlens_activation_patching_preflight(max_vram_gb: float = 24.0) -> dict:
    """Run the pinned real-model residual-position mechanics preflight on CUDA."""

    if not t.cuda.is_available():
        return {
            "cuda_available": False,
            "preflight_passed": False,
            "full_path": "Pinned TransformerLens gelu-1l residual-stream patching preflight.",
        }

    t.cuda.reset_peak_memory_stats()
    model = _load_gelu1l_model_on_cuda()
    model.eval()
    clean_tokens = model.to_tokens(TL_CLEAN_PROMPT)
    corrupt_tokens = model.to_tokens(TL_CORRUPT_PROMPT)
    if clean_tokens.shape != corrupt_tokens.shape:
        raise RuntimeError(
            f"clean/corrupt token shapes differ: {clean_tokens.shape} vs {corrupt_tokens.shape}"
        )

    target_position = clean_tokens.shape[1] - 1
    with t.inference_mode():
        clean_logits, clean_cache = model.run_with_cache(
            clean_tokens,
            names_filter=lambda name: name == TL_PATCH_HOOK_NAME,
        )
        corrupt_logits = model(corrupt_tokens)

    clean_final_logits = clean_logits[0, -1]
    corrupt_final_logits = corrupt_logits[0, -1]
    clean_top_tokens = clean_final_logits.topk(5).indices.tolist()
    corrupt_top_tokens = corrupt_final_logits.topk(5).indices.tolist()
    target_token_id = clean_top_tokens[0]
    distractor_token_id = next(
        token_id
        for token_id in [*corrupt_top_tokens, *clean_top_tokens[1:]]
        if token_id != target_token_id
    )
    clean_metric = answer_logit_diff(
        clean_final_logits,
        positive_token_id=target_token_id,
        negative_token_id=distractor_token_id,
    )
    corrupt_metric = answer_logit_diff(
        corrupt_final_logits,
        positive_token_id=target_token_id,
        negative_token_id=distractor_token_id,
    )

    def make_patch_hook(position: int):
        def patch_hook(activation: t.Tensor, hook) -> t.Tensor:
            del hook
            patched = activation.clone()
            patched[:, position, :] = clean_cache[TL_PATCH_HOOK_NAME][:, position, :]
            return patched

        return patch_hook

    patched_metrics: list[float] = []
    final_patched_logits = None
    with t.inference_mode():
        for position in range(clean_tokens.shape[1]):
            patched_logits = model.run_with_hooks(
                corrupt_tokens,
                fwd_hooks=[(TL_PATCH_HOOK_NAME, make_patch_hook(position))],
            )
            patched_metrics.append(
                answer_logit_diff(
                    patched_logits[0, -1],
                    positive_token_id=target_token_id,
                    negative_token_id=distractor_token_id,
                )
            )
            if position == target_position:
                final_patched_logits = patched_logits

    if final_patched_logits is None:
        raise RuntimeError("final-position patched logits were not recorded.")
    patch_scores = t.tensor(
        [
            recovery_fraction(
                clean_metric=clean_metric,
                corrupt_metric=corrupt_metric,
                patched_metric=metric,
            )
            for metric in patched_metrics
        ]
    )
    best_position = int(patch_scores.argmax().item())
    wrong_scores = patch_scores[:target_position]
    target_recovered_fraction = float(patch_scores[target_position].item())

    t.cuda.synchronize()
    peak_vram_gb = t.cuda.max_memory_allocated() / 1024**3
    max_abs_final_patch_logit_error = (
        final_patched_logits[0, -1].float() - clean_final_logits.float()
    ).abs().max().item()
    clean_corrupt_gap = clean_metric - corrupt_metric
    finite_logits = bool(
        t.isfinite(clean_final_logits).all() and t.isfinite(corrupt_final_logits).all()
    )
    wrong_mean = float(wrong_scores.mean().item())
    wrong_max = float(wrong_scores.max().item())
    within_vram_budget = peak_vram_gb <= max_vram_gb
    preflight_passed = (
        finite_logits
        and clean_corrupt_gap >= 1.0
        and target_recovered_fraction >= 0.99
        and max_abs_final_patch_logit_error <= 1e-5
        and best_position == target_position
        and target_recovered_fraction > wrong_mean
        and target_recovered_fraction > wrong_max
        and abs(wrong_mean) <= 1e-4
        and abs(wrong_max) <= 1e-4
        and within_vram_budget
    )
    return {
        "cuda_available": True,
        "device": t.cuda.get_device_name(0),
        "torch_version": t.__version__,
        "cuda_version": t.version.cuda,
        "preflight_passed": preflight_passed,
        "model_name": TL_GELU1L_MODEL_NAME,
        "hf_model_id": TL_GELU1L_HF_ID,
        "hf_revision": TL_GELU1L_REVISION,
        "tokenizer_id": TL_GELU1L_TOKENIZER_ID,
        "tokenizer_revision": TL_GELU1L_TOKENIZER_REVISION,
        "bnb_cuda_override": TL_BNB_CUDA_OVERRIDE,
        "hook_name": TL_PATCH_HOOK_NAME,
        "clean_prompt": TL_CLEAN_PROMPT,
        "corrupt_prompt": TL_CORRUPT_PROMPT,
        "sequence_length": int(clean_tokens.shape[1]),
        "target_position": int(target_position),
        "target_token_id": int(target_token_id),
        "target_token": model.to_string(target_token_id),
        "distractor_token_id": int(distractor_token_id),
        "distractor_token": model.to_string(distractor_token_id),
        "clean_metric": clean_metric,
        "corrupt_metric": corrupt_metric,
        "patched_metric": patched_metrics[target_position],
        "clean_corrupt_gap": clean_corrupt_gap,
        "target_recovered_fraction": target_recovered_fraction,
        "passes_recovery": target_recovered_fraction >= 0.99,
        "patch_scores_by_position": [float(score) for score in patch_scores.tolist()],
        "best_position": best_position,
        "best_score": float(patch_scores[best_position].item()),
        "localizes_final_position": best_position == target_position,
        "wrong_position_control_fraction": wrong_mean,
        "max_wrong_position_control_fraction": wrong_max,
        "top_beats_wrong_position_control": target_recovered_fraction > wrong_mean,
        "top_beats_max_wrong_position_control": target_recovered_fraction > wrong_max,
        "max_abs_final_patch_logit_error": max_abs_final_patch_logit_error,
        "finite_logits": finite_logits,
        "peak_vram_gb": peak_vram_gb,
        "within_vram_budget": within_vram_budget,
        "full_path": "Pinned TransformerLens gelu-1l residual-stream patching preflight.",
    }


def run_gpu_test(max_vram_gb: float = 24.0) -> dict:
    gpu = run_transformerlens_activation_patching_preflight(max_vram_gb=max_vram_gb)
    toy = run_toy_signature_result()
    gpu.update({f"toy_{key}": value for key, value in toy.items()})
    return gpu


def run_full_experiment(max_vram_gb: float = 24.0) -> dict:
    return run_gpu_test(max_vram_gb=max_vram_gb)


if MAIN:
    print(run_smoke_test())
