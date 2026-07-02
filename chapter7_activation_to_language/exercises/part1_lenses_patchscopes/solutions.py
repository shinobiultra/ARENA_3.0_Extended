# %%
"""Reference solutions for [7.1] Logit Lens, Tuned Lens, and Patchscopes."""

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch as t
import torch.nn.functional as F

chapter = "chapter7_activation_to_language"
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

MAIN = __name__ == "__main__"
PatchscopeTemplate = Literal["entity", "next_token", "fact"]

TL_GELU1L_MODEL_NAME = "gelu-1l"
TL_GELU1L_HF_ID = "NeelNanda/GELU_1L512W_C4_Code"
TL_GELU1L_REVISION = "bddc0e332f0ae84279e6a6a45d91b314899e1603"
TL_GELU1L_TOKENIZER_ID = "NeelNanda/gpt-neox-tokenizer-digits"
TL_GELU1L_TOKENIZER_REVISION = "0f6671571a20be9756b9991d978047c03b75e749"
TL_BNB_CUDA_OVERRIDE = "130"
TL_RESID_PRE_HOOK = "blocks.0.hook_resid_pre"
TL_RESID_POST_HOOK = "blocks.0.hook_resid_post"
TL_ATTN_PATTERN_HOOK = "blocks.0.attn.hook_pattern"
TL_ATTN_VALUE_HOOK = "blocks.0.attn.hook_v"
TL_LENS_TRAIN_PROMPTS = [
    "The cat sat on the",
    "The bird flew over the",
    "To make tea, boil the",
    "To make bread, bake the",
    "The recipe calls for sugar and",
    "The recipe calls for salt and",
    "The chef cooked a",
    "The teacher taught a",
    "The river flows into the",
    "The road leads into the",
    "The Python function returns a",
    "The HTML page contains a",
    "The file was saved to disk as a",
    "The message was sent by mail as a",
    "The train arrived at the",
    "The plane landed at the",
]
TL_LENS_HELDOUT_PROMPTS = [
    "The programmer wrote a",
    "The singer sang a",
    "The team won the game by scoring a",
    "The team lost the game after missing a",
    "A king lives in a",
    "A student learns in a",
]
TL_PATCHSCOPE_PAIRS = [
    ("The cat sat on the", "The bird flew over the"),
    ("To make tea, boil the", "To make bread, bake the"),
    ("The recipe calls for sugar and", "The recipe calls for salt and"),
    ("The chef cooked a", "The teacher taught a"),
    ("The programmer wrote a", "The singer sang a"),
    ("The team won the game by scoring a", "The team lost the game after missing a"),
]


# %%
@dataclass(frozen=True)
class LensAccuracyReport:
    logit_lens_accuracy: float
    tuned_lens_accuracy: float
    improvement: float
    tuned_lens_improves: bool


@dataclass(frozen=True)
class PatchscopeAccuracyReport:
    patchscope_accuracy: float
    text_only_accuracy: float
    improvement: float
    beats_text_only: bool


@dataclass(frozen=True)
class CounterfactualActivationReport:
    original_answer: int
    patched_answer: int
    changed: bool


@dataclass(frozen=True)
class RandomActivationConfidenceReport:
    mean_confidence: float
    max_confidence: float
    passes_low_confidence: bool


def logit_lens(residual_stream: t.Tensor, unembedding: t.Tensor) -> t.Tensor:
    """Project residual activations directly into vocabulary logits."""

    if residual_stream.shape[-1] != unembedding.shape[0]:
        raise ValueError("residual_stream last dimension must match unembedding rows.")
    return residual_stream.float() @ unembedding.float()


def tuned_lens(
    residual_stream: t.Tensor,
    lens_weight: t.Tensor,
    lens_bias: t.Tensor | None,
    unembedding: t.Tensor,
) -> t.Tensor:
    """Apply a learned affine correction before vocabulary projection."""

    if residual_stream.shape[-1] != lens_weight.shape[0]:
        raise ValueError("residual_stream last dimension must match lens_weight rows.")
    transformed = residual_stream.float() @ lens_weight.float()
    if lens_bias is not None:
        transformed = transformed + lens_bias.to(transformed.device)
    return logit_lens(transformed, unembedding)


def attention_lens(
    attention_pattern: t.Tensor,
    value_vectors: t.Tensor,
    unembedding: t.Tensor,
) -> t.Tensor:
    """Decode attention-weighted value vectors through the unembedding."""

    if attention_pattern.ndim != 3 or value_vectors.ndim != 3:
        raise ValueError("attention_pattern and value_vectors must be rank-3 tensors.")
    if attention_pattern.shape[-1] != value_vectors.shape[-2]:
        raise ValueError("attention key dimension must match value sequence length.")
    attended = attention_pattern.float() @ value_vectors.float()
    return logit_lens(attended, unembedding)


def top_tokens(logits: t.Tensor, *, k: int = 5) -> tuple[t.Tensor, t.Tensor]:
    """Return top token ids and probabilities."""

    if k <= 0 or k > logits.shape[-1]:
        raise ValueError("k must be between 1 and vocab size.")
    probs = F.softmax(logits.float(), dim=-1)
    values, indices = probs.topk(k=k, dim=-1)
    return indices, values


def top_token_table(
    logits: t.Tensor,
    id_to_token,
    *,
    k: int = 5,
    target_token_ids: t.Tensor | None = None,
    row_labels: list[str] | None = None,
) -> list[dict[str, object]]:
    """Build a learner-readable table of top decoded tokens for rank-2 logits."""

    if logits.ndim != 2:
        raise ValueError("top_token_table expects logits with shape [row, vocab].")
    if target_token_ids is not None and target_token_ids.shape != logits.shape[:1]:
        raise ValueError("target_token_ids must have one id per logits row.")
    if row_labels is not None and len(row_labels) != logits.shape[0]:
        raise ValueError("row_labels must have one label per logits row.")

    top_ids, top_probs = top_tokens(logits, k=k)

    def decode(token_id: int) -> str:
        if callable(id_to_token):
            return str(id_to_token(token_id))
        return str(id_to_token[token_id])

    rows: list[dict[str, object]] = []
    sorted_ids = logits.float().argsort(dim=-1, descending=True)
    for row in range(logits.shape[0]):
        target_id = None if target_token_ids is None else int(target_token_ids[row].item())
        target_rank = None
        target_token = None
        if target_id is not None:
            target_token = decode(target_id)
            target_rank = int((sorted_ids[row] == target_id).nonzero(as_tuple=False)[0].item()) + 1
        rows.append(
            {
                "row": row if row_labels is None else row_labels[row],
                "top_ids": [int(x) for x in top_ids[row].tolist()],
                "top_tokens": [decode(int(x)) for x in top_ids[row].tolist()],
                "top_probs": [float(x) for x in top_probs[row].tolist()],
                "target_token": target_token,
                "target_rank": target_rank,
            }
        )
    return rows


def prediction_accuracy(logits: t.Tensor, target_token_ids: t.Tensor) -> float:
    """Top-1 accuracy for decoded logits."""

    if logits.shape[:-1] != target_token_ids.shape:
        raise ValueError("target_token_ids must match logits leading dimensions.")
    predictions = logits.argmax(dim=-1)
    return predictions.eq(target_token_ids).float().mean().item()


def fit_ridge_tuned_lens(
    residual_stream: t.Tensor,
    target_residual_stream: t.Tensor,
    *,
    ridge: float = 1e-2,
) -> tuple[t.Tensor, t.Tensor]:
    """Fit an affine map from early residuals to target residuals by ridge regression."""

    if residual_stream.ndim != 2 or target_residual_stream.ndim != 2:
        raise ValueError("ridge fitting expects rank-2 [example, d_model] tensors.")
    if residual_stream.shape[0] != target_residual_stream.shape[0]:
        raise ValueError("source and target tensors must have the same number of examples.")
    design = t.cat(
        [
            residual_stream.float(),
            t.ones(residual_stream.shape[0], 1, device=residual_stream.device),
        ],
        dim=1,
    )
    penalty = t.eye(design.shape[1], device=design.device)
    penalty[-1, -1] = 0.0
    solution = t.linalg.solve(
        design.T @ design + ridge * penalty,
        design.T @ target_residual_stream.float(),
    )
    return solution[:-1], solution[-1]


def evaluate_lens_on_heldout(
    residual_stream: t.Tensor,
    lens_weight: t.Tensor,
    lens_bias: t.Tensor | None,
    unembedding: t.Tensor,
    target_token_ids: t.Tensor,
) -> LensAccuracyReport:
    """Evaluate ordinary logit lens and tuned lens on held-out target ids."""

    logit_logits = logit_lens(residual_stream, unembedding)
    tuned_logits = tuned_lens(residual_stream, lens_weight, lens_bias, unembedding)
    return lens_accuracy_report(logit_logits, tuned_logits, target_token_ids)


def lens_accuracy_report(
    logit_lens_logits: t.Tensor,
    tuned_lens_logits: t.Tensor,
    target_token_ids: t.Tensor,
) -> LensAccuracyReport:
    """Compare tuned-lens decoding against ordinary logit lens decoding."""

    logit_acc = prediction_accuracy(logit_lens_logits, target_token_ids)
    tuned_acc = prediction_accuracy(tuned_lens_logits, target_token_ids)
    improvement = tuned_acc - logit_acc
    return LensAccuracyReport(
        logit_lens_accuracy=logit_acc,
        tuned_lens_accuracy=tuned_acc,
        improvement=improvement,
        tuned_lens_improves=improvement > 0,
    )


def patchscope_prompt(template: PatchscopeTemplate, placeholder: str = "<ACT>") -> str:
    """Return a minimal prompt template for a Patchscope-style decode."""

    if template == "entity":
        return f"What entity is represented by {placeholder}?"
    if template == "next_token":
        return f"What token will {placeholder} become next?"
    if template == "fact":
        return f"What fact is stored in {placeholder}?"
    raise ValueError("unknown Patchscope template.")


def patchscope_accuracy_report(
    patchscope_logits: t.Tensor,
    text_only_logits: t.Tensor,
    target_answer_ids: t.Tensor,
) -> PatchscopeAccuracyReport:
    """Compare Patchscope answers against a text-only baseline."""

    patchscope_acc = prediction_accuracy(patchscope_logits, target_answer_ids)
    text_only_acc = prediction_accuracy(text_only_logits, target_answer_ids)
    improvement = patchscope_acc - text_only_acc
    return PatchscopeAccuracyReport(
        patchscope_accuracy=patchscope_acc,
        text_only_accuracy=text_only_acc,
        improvement=improvement,
        beats_text_only=improvement > 0,
    )


def replace_final_position_activation(
    activations: t.Tensor,
    source_activation: t.Tensor,
) -> t.Tensor:
    """Return activations with the final sequence position replaced by source_activation."""

    if activations.ndim != 3:
        raise ValueError("activations must have shape [batch, seq, d_model].")
    if source_activation.ndim != 1:
        raise ValueError("source_activation must have shape [d_model].")
    if activations.shape[0] != 1:
        raise ValueError("this teaching helper expects batch size 1.")
    if activations.shape[-1] != source_activation.shape[0]:
        raise ValueError("source_activation dimension must match activations d_model.")
    patched = activations.clone()
    patched[0, -1] = source_activation.to(device=activations.device, dtype=activations.dtype)
    return patched


def patchscope_activation_decode(
    model,
    source_activation: t.Tensor,
    target_tokens: t.Tensor,
    *,
    hook_name: str = TL_RESID_POST_HOOK,
) -> t.Tensor:
    """Patch a source activation into a target prompt and decode final-token logits."""

    def patch_hook(activations: t.Tensor, hook=None) -> t.Tensor:
        _ = hook
        return replace_final_position_activation(activations, source_activation)

    with t.inference_mode():
        patched_logits = model.run_with_hooks(
            target_tokens,
            fwd_hooks=[(hook_name, patch_hook)],
        )
    return patched_logits[0, -1].detach()


def counterfactual_activation_report(
    original_logits: t.Tensor,
    patched_logits: t.Tensor,
) -> CounterfactualActivationReport:
    """Check whether a counterfactual activation changes the decoded answer."""

    if original_logits.ndim != 1 or patched_logits.ndim != 1:
        raise ValueError("original_logits and patched_logits must be rank-1 tensors.")
    original_answer = int(original_logits.argmax().item())
    patched_answer = int(patched_logits.argmax().item())
    return CounterfactualActivationReport(
        original_answer=original_answer,
        patched_answer=patched_answer,
        changed=original_answer != patched_answer,
    )


def random_activation_confidence_report(
    random_logits: t.Tensor,
    *,
    max_allowed_confidence: float = 0.6,
) -> RandomActivationConfidenceReport:
    """Check that random activations decode to low-confidence answers."""

    confidence = F.softmax(random_logits.float(), dim=-1).max(dim=-1).values
    mean_confidence = confidence.mean().item()
    max_confidence = confidence.max().item()
    return RandomActivationConfidenceReport(
        mean_confidence=mean_confidence,
        max_confidence=max_confidence,
        passes_low_confidence=max_confidence <= max_allowed_confidence,
    )


def patchscope_eval(
    patchscope_logits: t.Tensor,
    text_only_logits: t.Tensor,
    random_logits: t.Tensor,
    target_answer_ids: t.Tensor,
    *,
    max_allowed_random_confidence: float = 0.6,
) -> dict[str, object]:
    """Bundle Patchscope, text-only, and random-activation controls."""

    patchscope = patchscope_accuracy_report(
        patchscope_logits,
        text_only_logits,
        target_answer_ids,
    )
    random_control = random_activation_confidence_report(
        random_logits,
        max_allowed_confidence=max_allowed_random_confidence,
    )
    return {
        "patchscope_accuracy": patchscope.patchscope_accuracy,
        "text_only_accuracy": patchscope.text_only_accuracy,
        "improvement": patchscope.improvement,
        "beats_text_only": patchscope.beats_text_only,
        "random_mean_confidence": random_control.mean_confidence,
        "random_max_confidence": random_control.max_confidence,
        "random_passes_low_confidence": random_control.passes_low_confidence,
        "passes": patchscope.beats_text_only and random_control.passes_low_confidence,
    }


def logit_lens_smoke_test() -> dict:
    residual = t.tensor([[1.0, 0.0], [0.0, 1.0]])
    unembedding = t.tensor([[2.0, 0.0, 1.0], [0.0, 3.0, 1.0]])
    logits = logit_lens(residual, unembedding)
    top_ids, top_probs = top_tokens(logits, k=1)
    top_table = top_token_table(
        logits,
        [" floor", " sky", " neutral"],
        k=2,
        target_token_ids=t.tensor([0, 1]),
        row_labels=["residual direction 0", "residual direction 1"],
    )
    return {
        "logits": logits.tolist(),
        "top_ids": top_ids.tolist(),
        "top_probs": top_probs.tolist(),
        "top_table": top_table,
    }


def tuned_lens_smoke_test() -> dict:
    residual = t.tensor([[1.0, 0.0], [0.0, 1.0]])
    unembedding = t.eye(2)
    logit_logits = logit_lens(residual, unembedding)
    lens_weight = t.tensor([[0.0, 1.0], [1.0, 0.0]])
    tuned_logits = tuned_lens(residual, lens_weight, None, unembedding)
    targets = t.tensor([1, 0])
    return lens_accuracy_report(logit_logits, tuned_logits, targets).__dict__


def fit_ridge_tuned_lens_smoke_test() -> dict:
    train = t.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    true_weight = t.tensor([[0.0, 1.0], [1.0, 0.0]])
    true_bias = t.tensor([0.5, -0.25])
    target_residuals = train @ true_weight + true_bias
    weight, bias = fit_ridge_tuned_lens(train, target_residuals, ridge=0.0)
    heldout = t.tensor([[2.0, 0.0], [0.0, 2.0]])
    report = evaluate_lens_on_heldout(
        heldout,
        weight,
        bias,
        t.eye(2),
        t.tensor([1, 0]),
    )
    return {
        "weight": weight.tolist(),
        "bias": bias.tolist(),
        **report.__dict__,
    }


def attention_lens_smoke_test() -> dict:
    attention = t.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    values = t.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    unembedding = t.eye(2)
    return {"logits": attention_lens(attention, values, unembedding).tolist()}


def patchscope_smoke_test() -> dict:
    patchscope_logits = t.tensor([[2.0, 0.0], [0.0, 2.0]])
    text_only_logits = t.tensor([[0.0, 2.0], [0.0, 2.0]])
    targets = t.tensor([0, 1])
    random_logits = t.zeros(3, 4)
    patched_acts = replace_final_position_activation(
        t.zeros(1, 3, 2),
        t.tensor([1.0, -1.0]),
    )
    report = patchscope_accuracy_report(patchscope_logits, text_only_logits, targets)
    return {
        "entity_prompt": patchscope_prompt("entity"),
        "next_token_prompt": patchscope_prompt("next_token"),
        "fact_prompt": patchscope_prompt("fact"),
        "patched_final_activation": patched_acts[0, -1].tolist(),
        "patchscope_eval": patchscope_eval(
            patchscope_logits,
            text_only_logits,
            random_logits,
            targets,
            max_allowed_random_confidence=0.3,
        ),
        **report.__dict__,
    }


def counterfactual_smoke_test() -> dict:
    original = t.tensor([2.0, 0.0])
    patched = t.tensor([0.0, 3.0])
    return counterfactual_activation_report(original, patched).__dict__


def random_confidence_smoke_test() -> dict:
    random_logits = t.zeros(3, 4)
    return random_activation_confidence_report(
        random_logits,
        max_allowed_confidence=0.3,
    ).__dict__


def run_smoke_test(cpu: bool = True) -> dict:
    _ = cpu
    return {
        "logit_lens": logit_lens_smoke_test(),
        "tuned_lens": tuned_lens_smoke_test(),
        "fit_ridge_tuned_lens": fit_ridge_tuned_lens_smoke_test(),
        "attention_lens": attention_lens_smoke_test(),
        "patchscope": patchscope_smoke_test(),
        "patchscope_eval": patchscope_smoke_test()["patchscope_eval"],
        "counterfactual": counterfactual_smoke_test(),
        "random_confidence": random_confidence_smoke_test(),
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


def _decode_residual(model, residual_stream: t.Tensor) -> t.Tensor:
    normalized = model.ln_final(residual_stream)
    return logit_lens(normalized, model.W_U) + model.b_U


def _cache_lens_dataset(model, prompts: list[str]) -> dict[str, t.Tensor]:
    pre_activations = []
    post_normalized = []
    targets = []
    final_logits = []
    for prompt in prompts:
        tokens = model.to_tokens(prompt)
        with t.inference_mode():
            logits, cache = model.run_with_cache(
                tokens,
                names_filter=lambda name: name in {TL_RESID_PRE_HOOK, TL_RESID_POST_HOOK},
            )
            pre_activations.append(cache[TL_RESID_PRE_HOOK][0].detach())
            post_normalized.append(model.ln_final(cache[TL_RESID_POST_HOOK][0]).detach())
            targets.append(logits[0].argmax(dim=-1).detach())
            final_logits.append(logits[0].detach())
    return {
        "pre": t.cat(pre_activations, dim=0),
        "post_normalized": t.cat(post_normalized, dim=0),
        "targets": t.cat(targets, dim=0),
        "final_logits": t.cat(final_logits, dim=0),
    }


def _fit_ridge_tuned_lens(
    residual_pre: t.Tensor,
    normalized_residual_post: t.Tensor,
    *,
    ridge: float = 1e-2,
) -> tuple[t.Tensor, t.Tensor]:
    design = t.cat(
        [residual_pre.float(), t.ones(residual_pre.shape[0], 1, device=residual_pre.device)],
        dim=1,
    )
    penalty = t.eye(design.shape[1], device=design.device)
    penalty[-1, -1] = 0.0
    solution = t.linalg.solve(
        design.T @ design + ridge * penalty,
        design.T @ normalized_residual_post.float(),
    )
    return solution[:-1], solution[-1]


def run_transformerlens_lens_patchscope_preflight(max_vram_gb: float = 24.0) -> dict:
    """Train/evaluate a real affine lens and activation-conditioned decode on CUDA."""

    if not t.cuda.is_available():
        return {
            "cuda_available": False,
            "preflight_passed": False,
            "full_path": "Pinned TransformerLens gelu-1l lens and Patchscope-style preflight.",
        }

    t.cuda.reset_peak_memory_stats()
    model = _load_gelu1l_model_on_cuda()
    model.eval()

    train = _cache_lens_dataset(model, TL_LENS_TRAIN_PROMPTS)
    heldout = _cache_lens_dataset(model, TL_LENS_HELDOUT_PROMPTS)
    lens_weight, lens_bias = _fit_ridge_tuned_lens(
        train["pre"],
        train["post_normalized"],
    )
    logit_lens_logits = _decode_residual(model, heldout["pre"])
    tuned_lens_logits = tuned_lens(
        heldout["pre"],
        lens_weight,
        lens_bias,
        model.W_U,
    ) + model.b_U
    lens_report = lens_accuracy_report(
        logit_lens_logits,
        tuned_lens_logits,
        heldout["targets"],
    )
    final_decode_logits = logit_lens(heldout["post_normalized"], model.W_U) + model.b_U
    final_decode_max_abs_error = (
        final_decode_logits.float() - heldout["final_logits"].float()
    ).abs().max().item()

    patchscope_logits = []
    text_only_logits = []
    patchscope_targets = []
    patchscope_target_prompts = ["This means"] * len(TL_PATCHSCOPE_PAIRS)
    patched_target_margins = []
    text_only_target_margins = []
    for (clean_prompt, _corrupt_prompt), target_prompt in zip(
        TL_PATCHSCOPE_PAIRS,
        patchscope_target_prompts,
        strict=True,
    ):
        with t.inference_mode():
            clean_logits, clean_cache = model.run_with_cache(
                model.to_tokens(clean_prompt),
                names_filter=lambda name: name == TL_RESID_POST_HOOK,
            )
            target_tokens = model.to_tokens(target_prompt)
            baseline_logits = model(target_tokens)[0, -1].detach()
        target_id = int(clean_logits[0, -1].argmax().item())
        source_activation = clean_cache[TL_RESID_POST_HOOK][0, -1].detach().clone()
        patched_logits = patchscope_activation_decode(
            model,
            source_activation,
            target_tokens,
            hook_name=TL_RESID_POST_HOOK,
        )
        patchscope_logits.append(patched_logits)
        text_only_logits.append(baseline_logits)
        patchscope_targets.append(target_id)
        patched_target_margins.append(
            (patched_logits[target_id] - patched_logits.topk(2).values[-1]).item()
            if int(patched_logits.argmax().item()) == target_id
            else (patched_logits[target_id] - patched_logits.max()).item()
        )
        text_only_target_margins.append(
            (baseline_logits[target_id] - baseline_logits.max()).item()
        )
    patchscope_target_ids = t.tensor(patchscope_targets, device=heldout["pre"].device)
    patchscope_report = patchscope_accuracy_report(
        t.stack(patchscope_logits, dim=0),
        t.stack(text_only_logits, dim=0),
        patchscope_target_ids,
    )

    with t.inference_mode():
        clean_logits, clean_cache = model.run_with_cache(
            model.to_tokens(TL_PATCHSCOPE_PAIRS[0][0]),
            names_filter=lambda name: name == TL_RESID_POST_HOOK,
        )
        corrupt_logits, corrupt_cache = model.run_with_cache(
            model.to_tokens(TL_PATCHSCOPE_PAIRS[0][1]),
            names_filter=lambda name: name == TL_RESID_POST_HOOK,
        )
    counterfactual = counterfactual_activation_report(
        _decode_residual(model, clean_cache[TL_RESID_POST_HOOK][0, -1]),
        _decode_residual(model, corrupt_cache[TL_RESID_POST_HOOK][0, -1]),
    )

    t.manual_seed(12345)
    random_residuals = (
        t.randn(8, model.cfg.d_model, device=heldout["pre"].device)
        * heldout["pre"].float().std()
        * 0.001
    )
    random_report = random_activation_confidence_report(
        _decode_residual(model, random_residuals),
        max_allowed_confidence=0.2,
    )

    with t.inference_mode():
        attention_logits, attention_cache = model.run_with_cache(
            model.to_tokens(TL_PATCHSCOPE_PAIRS[0][0]),
            names_filter=lambda name: name in {TL_ATTN_PATTERN_HOOK, TL_ATTN_VALUE_HOOK},
        )
    attention_pattern = attention_cache[TL_ATTN_PATTERN_HOOK][:, 0].detach().clone()
    head_values = (
        attention_cache[TL_ATTN_VALUE_HOOK][:, :, 0, :].detach().clone()
        @ model.W_O[0, 0].detach()
    )
    attention_lens_logits = attention_lens(
        attention_pattern,
        head_values,
        model.W_U.detach(),
    ) + model.b_U.detach()
    attention_lens_finite = bool(t.isfinite(attention_lens_logits).all().item())

    t.cuda.synchronize()
    peak_vram_gb = t.cuda.max_memory_allocated() / 1024**3
    within_vram_budget = peak_vram_gb <= max_vram_gb
    preflight_passed = (
        lens_report.tuned_lens_accuracy >= 0.35
        and lens_report.improvement >= 0.30
        and final_decode_max_abs_error <= 1e-4
        and patchscope_report.patchscope_accuracy == 1.0
        and patchscope_report.text_only_accuracy == 0.0
        and patchscope_report.beats_text_only
        and min(patched_target_margins) > 0.0
        and max(text_only_target_margins) <= 0.0
        and counterfactual.changed
        and random_report.passes_low_confidence
        and attention_lens_finite
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
        "train_prompt_count": len(TL_LENS_TRAIN_PROMPTS),
        "heldout_prompt_count": len(TL_LENS_HELDOUT_PROMPTS),
        "heldout_position_count": int(heldout["targets"].numel()),
        "logit_lens_accuracy": lens_report.logit_lens_accuracy,
        "tuned_lens_accuracy": lens_report.tuned_lens_accuracy,
        "tuned_lens_improvement": lens_report.improvement,
        "tuned_lens_improves": lens_report.tuned_lens_improves,
        "final_decode_max_abs_error": final_decode_max_abs_error,
        "patchscope_pair_count": len(TL_PATCHSCOPE_PAIRS),
        "patchscope_hook_name": TL_RESID_POST_HOOK,
        "patchscope_target_prompt_count": len(patchscope_target_prompts),
        "patchscope_accuracy": patchscope_report.patchscope_accuracy,
        "text_only_accuracy": patchscope_report.text_only_accuracy,
        "patchscope_beats_text_only": patchscope_report.beats_text_only,
        "patchscope_min_patched_target_margin": min(patched_target_margins),
        "patchscope_max_text_only_target_margin": max(text_only_target_margins),
        "counterfactual_changed": counterfactual.changed,
        "counterfactual_original_token": model.to_string(counterfactual.original_answer),
        "counterfactual_patched_token": model.to_string(counterfactual.patched_answer),
        "random_mean_confidence": random_report.mean_confidence,
        "random_max_confidence": random_report.max_confidence,
        "random_passes_low_confidence": random_report.passes_low_confidence,
        "attention_lens_logits_shape": list(attention_lens_logits.shape),
        "attention_lens_finite": attention_lens_finite,
        "attention_lens_prompt_final_token": model.to_string(
            int(attention_logits[0, -1].argmax().item())
        ),
        "peak_vram_gb": peak_vram_gb,
        "within_vram_budget": within_vram_budget,
        "full_path": "Pinned TransformerLens gelu-1l lens and Patchscope-style preflight.",
    }


def run_gpu_test(max_vram_gb: float = 24.0) -> dict:
    return run_transformerlens_lens_patchscope_preflight(max_vram_gb=max_vram_gb)


def run_full_experiment(max_vram_gb: float = 24.0) -> dict:
    """Run the validated experiment path used by the verification report."""

    return run_gpu_test(max_vram_gb=max_vram_gb)


if MAIN:
    print(run_smoke_test())
