# %%
"""Reference solutions for [8.1] Activation Patching Refresher."""

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

TL_GELU1L_MODEL_NAME = "gelu-1l"
TL_GELU1L_HF_ID = "NeelNanda/GELU_1L512W_C4_Code"
TL_GELU1L_REVISION = "bddc0e332f0ae84279e6a6a45d91b314899e1603"
TL_GELU1L_TOKENIZER_ID = "NeelNanda/gpt-neox-tokenizer-digits"
TL_GELU1L_TOKENIZER_REVISION = "0f6671571a20be9756b9991d978047c03b75e749"
TL_PATCH_HOOK_NAME = "blocks.0.hook_resid_post"
TL_CLEAN_PROMPT = "The cat sat on the"
TL_CORRUPT_PROMPT = "The bird flew over the"
TL_BNB_CUDA_OVERRIDE = "130"


# %%
@dataclass(frozen=True)
class PatchingRecoveryReport:
    clean_metric: float
    corrupt_metric: float
    patched_metric: float
    recovered_fraction: float
    passes_recovery: bool


@dataclass(frozen=True)
class ActivationPatchingSweep:
    patch_scores: t.Tensor
    best_index: int
    best_score: float


@dataclass(frozen=True)
class PatchingLocalizationReport:
    top_indices: tuple[int, ...]
    target_indices: tuple[int, ...]
    topk_overlap: float
    localizes_target: bool


@dataclass(frozen=True)
class RandomPatchControlReport:
    top_patch_score: float
    random_patch_score: float
    top_beats_random: bool


def answer_logit_diff(
    logits: t.Tensor,
    *,
    positive_token_id: int,
    negative_token_id: int,
) -> float:
    """Return mean positive-minus-negative logit difference."""

    vocab_size = logits.shape[-1]
    if not 0 <= positive_token_id < vocab_size:
        raise ValueError("positive_token_id is out of range.")
    if not 0 <= negative_token_id < vocab_size:
        raise ValueError("negative_token_id is out of range.")
    diff = logits[..., positive_token_id] - logits[..., negative_token_id]
    return diff.float().mean().item()


def patch_activation_slice(
    clean_activations: t.Tensor,
    corrupt_activations: t.Tensor,
    *,
    component_index: int,
    component_dim: int = 0,
) -> t.Tensor:
    """Patch one clean activation slice into the corrupt activation tensor."""

    if clean_activations.shape != corrupt_activations.shape:
        raise ValueError("clean and corrupt activations must have matching shape.")
    if not 0 <= component_dim < clean_activations.ndim:
        raise ValueError("component_dim is out of range.")
    if not 0 <= component_index < clean_activations.shape[component_dim]:
        raise ValueError("component_index is out of range.")

    patched = corrupt_activations.clone()
    slices = [slice(None)] * clean_activations.ndim
    slices[component_dim] = component_index
    patched[tuple(slices)] = clean_activations[tuple(slices)]
    return patched


def recovery_fraction(
    *,
    clean_metric: float,
    corrupt_metric: float,
    patched_metric: float,
) -> float:
    """Return how much of the clean-corrupt gap a patch recovers."""

    denominator = clean_metric - corrupt_metric
    if denominator == 0:
        raise ValueError("clean_metric and corrupt_metric must differ.")
    return (patched_metric - corrupt_metric) / denominator


def patching_recovery_report(
    clean_logits: t.Tensor,
    corrupt_logits: t.Tensor,
    patched_logits: t.Tensor,
    *,
    positive_token_id: int,
    negative_token_id: int,
    min_recovered_fraction: float = 0.5,
) -> PatchingRecoveryReport:
    """Measure logit-diff recovery after patching clean activations."""

    clean_metric = answer_logit_diff(
        clean_logits,
        positive_token_id=positive_token_id,
        negative_token_id=negative_token_id,
    )
    corrupt_metric = answer_logit_diff(
        corrupt_logits,
        positive_token_id=positive_token_id,
        negative_token_id=negative_token_id,
    )
    patched_metric = answer_logit_diff(
        patched_logits,
        positive_token_id=positive_token_id,
        negative_token_id=negative_token_id,
    )
    recovered = recovery_fraction(
        clean_metric=clean_metric,
        corrupt_metric=corrupt_metric,
        patched_metric=patched_metric,
    )
    return PatchingRecoveryReport(
        clean_metric=clean_metric,
        corrupt_metric=corrupt_metric,
        patched_metric=patched_metric,
        recovered_fraction=recovered,
        passes_recovery=recovered >= min_recovered_fraction,
    )


def activation_patching_sweep(
    *,
    clean_metric: float,
    corrupt_metric: float,
    patched_metrics: t.Tensor,
) -> ActivationPatchingSweep:
    """Convert per-component patched metrics into recovery scores."""

    denominator = clean_metric - corrupt_metric
    if denominator == 0:
        raise ValueError("clean_metric and corrupt_metric must differ.")
    patch_scores = (patched_metrics.float() - corrupt_metric) / denominator
    best_index = int(patch_scores.argmax().item())
    return ActivationPatchingSweep(
        patch_scores=patch_scores,
        best_index=best_index,
        best_score=float(patch_scores[best_index].item()),
    )


def patching_localization_report(
    patch_scores: t.Tensor,
    target_indices: list[int],
    *,
    top_k: int = 2,
    min_overlap: float = 0.5,
) -> PatchingLocalizationReport:
    """Check whether top patching scores recover known target components."""

    if patch_scores.ndim != 1:
        raise ValueError("patch_scores must be rank-1.")
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    if len(target_indices) == 0:
        raise ValueError("target_indices must be nonempty.")

    k = min(top_k, patch_scores.numel())
    top_indices = tuple(int(index) for index in patch_scores.topk(k=k).indices.tolist())
    target_tuple = tuple(int(index) for index in target_indices)
    top_set = set(top_indices)
    target_set = set(target_tuple)
    denominator = min(k, len(target_set))
    overlap = len(top_set & target_set) / denominator
    return PatchingLocalizationReport(
        top_indices=top_indices,
        target_indices=target_tuple,
        topk_overlap=overlap,
        localizes_target=overlap >= min_overlap,
    )


def random_patch_control_report(
    patch_scores: t.Tensor,
    random_indices: list[int],
    *,
    top_k: int = 2,
) -> RandomPatchControlReport:
    """Compare top patching score against a random-component control."""

    if patch_scores.ndim != 1:
        raise ValueError("patch_scores must be rank-1.")
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    if len(random_indices) == 0:
        raise ValueError("random_indices must be nonempty.")

    k = min(top_k, patch_scores.numel())
    top_patch_score = patch_scores.topk(k=k).values.mean().item()
    random_tensor = t.tensor(random_indices, dtype=t.long, device=patch_scores.device)
    if random_tensor.min().item() < 0 or random_tensor.max().item() >= patch_scores.numel():
        raise ValueError("random index is out of range.")
    random_patch_score = patch_scores[random_tensor].float().mean().item()
    return RandomPatchControlReport(
        top_patch_score=top_patch_score,
        random_patch_score=random_patch_score,
        top_beats_random=top_patch_score > random_patch_score,
    )


def logit_diff_smoke_test() -> float:
    logits = t.tensor([[4.0, 1.0], [3.0, 2.0]])
    return answer_logit_diff(logits, positive_token_id=0, negative_token_id=1)


def patch_slice_smoke_test() -> list[list[float]]:
    clean = t.tensor([[10.0, 20.0], [30.0, 40.0]])
    corrupt = t.zeros_like(clean)
    patched = patch_activation_slice(
        clean,
        corrupt,
        component_index=1,
        component_dim=0,
    )
    return patched.tolist()


def recovery_smoke_test() -> dict:
    clean_logits = t.tensor([4.0, 1.0])
    corrupt_logits = t.tensor([1.0, 3.0])
    patched_logits = t.tensor([3.0, 1.0])
    return patching_recovery_report(
        clean_logits,
        corrupt_logits,
        patched_logits,
        positive_token_id=0,
        negative_token_id=1,
        min_recovered_fraction=0.75,
    ).__dict__


def sweep_smoke_test() -> dict:
    sweep = activation_patching_sweep(
        clean_metric=3.0,
        corrupt_metric=-2.0,
        patched_metrics=t.tensor([-1.0, 2.0, 0.0]),
    )
    return {
        "patch_scores": [round(score, 6) for score in sweep.patch_scores.tolist()],
        "best_index": sweep.best_index,
        "best_score": sweep.best_score,
    }


def localization_smoke_test() -> dict:
    report = patching_localization_report(
        t.tensor([0.2, 0.9, 0.8, 0.1]),
        target_indices=[1, 2],
        top_k=2,
        min_overlap=1.0,
    )
    return report.__dict__


def random_control_smoke_test() -> dict:
    report = random_patch_control_report(
        t.tensor([0.2, 0.9, 0.8, 0.1]),
        random_indices=[0, 3],
        top_k=2,
    )
    return report.__dict__


def run_smoke_test(cpu: bool = True) -> dict:
    _ = cpu
    return {
        "logit_diff": logit_diff_smoke_test(),
        "patch_slice": patch_slice_smoke_test(),
        "recovery": recovery_smoke_test(),
        "sweep": sweep_smoke_test(),
        "localization": localization_smoke_test(),
        "random_control": random_control_smoke_test(),
    }


def _load_gelu1l_model_on_cuda():
    os.environ.setdefault("BNB_CUDA_VERSION", TL_BNB_CUDA_OVERRIDE)
    logging.getLogger("bitsandbytes.cextension").setLevel(logging.ERROR)
    from transformer_lens import HookedTransformer

    return HookedTransformer.from_pretrained(
        TL_GELU1L_MODEL_NAME,
        device="cuda",
        dtype="float32",
        revision=TL_GELU1L_REVISION,
    )


def run_transformerlens_activation_patching_preflight(max_vram_gb: float = 24.0) -> dict:
    """Run a real TransformerLens activation-patching position sweep on CUDA."""

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

    def make_patch_hook(position: int):
        def patch_hook(activation: t.Tensor, hook) -> t.Tensor:
            patched = activation.clone()
            patched[:, position, :] = clean_cache[TL_PATCH_HOOK_NAME][:, position, :]
            return patched

        return patch_hook

    patched_metrics = []
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

    recovery = patching_recovery_report(
        clean_final_logits,
        corrupt_final_logits,
        final_patched_logits[0, -1],
        positive_token_id=target_token_id,
        negative_token_id=distractor_token_id,
        min_recovered_fraction=0.99,
    )
    sweep = activation_patching_sweep(
        clean_metric=recovery.clean_metric,
        corrupt_metric=recovery.corrupt_metric,
        patched_metrics=t.tensor(patched_metrics, device=clean_logits.device),
    )
    localization = patching_localization_report(
        sweep.patch_scores,
        target_indices=[target_position],
        top_k=1,
        min_overlap=1.0,
    )
    non_final_positions = list(range(target_position))
    wrong_position_control = random_patch_control_report(
        sweep.patch_scores,
        random_indices=non_final_positions,
        top_k=1,
    )

    t.cuda.synchronize()
    peak_vram_gb = t.cuda.max_memory_allocated() / 1024**3
    max_abs_final_patch_logit_error = (
        final_patched_logits[0, -1].float() - clean_final_logits.float()
    ).abs().max().item()
    clean_corrupt_gap = recovery.clean_metric - recovery.corrupt_metric
    finite_logits = bool(t.isfinite(clean_final_logits).all() and t.isfinite(corrupt_final_logits).all())
    within_vram_budget = peak_vram_gb <= max_vram_gb
    preflight_passed = (
        finite_logits
        and clean_corrupt_gap >= 1.0
        and recovery.passes_recovery
        and max_abs_final_patch_logit_error <= 1e-5
        and localization.localizes_target
        and wrong_position_control.top_beats_random
        and abs(wrong_position_control.random_patch_score) <= 1e-4
        and within_vram_budget
    )

    return {
        "cuda_available": True,
        "device": t.cuda.get_device_name(0),
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
        "clean_metric": recovery.clean_metric,
        "corrupt_metric": recovery.corrupt_metric,
        "patched_metric": recovery.patched_metric,
        "clean_corrupt_gap": clean_corrupt_gap,
        "target_recovered_fraction": recovery.recovered_fraction,
        "passes_recovery": recovery.passes_recovery,
        "patch_scores_by_position": [float(score) for score in sweep.patch_scores.tolist()],
        "best_position": sweep.best_index,
        "best_score": sweep.best_score,
        "localizes_final_position": localization.localizes_target,
        "wrong_position_control_fraction": wrong_position_control.random_patch_score,
        "top_beats_wrong_position_control": wrong_position_control.top_beats_random,
        "max_abs_final_patch_logit_error": max_abs_final_patch_logit_error,
        "finite_logits": finite_logits,
        "peak_vram_gb": peak_vram_gb,
        "within_vram_budget": within_vram_budget,
        "full_path": "Pinned TransformerLens gelu-1l residual-stream patching preflight.",
    }


def run_gpu_test(max_vram_gb: float = 24.0) -> dict:
    return run_transformerlens_activation_patching_preflight(max_vram_gb=max_vram_gb)


def run_full_experiment(max_vram_gb: float = 24.0) -> dict:
    """Run the validated experiment path used by the verification report."""

    return run_gpu_test(max_vram_gb=max_vram_gb)


if MAIN:
    print(run_smoke_test())
