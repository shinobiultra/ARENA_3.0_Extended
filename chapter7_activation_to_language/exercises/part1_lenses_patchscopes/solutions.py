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

PYTHIA_MODEL_ID = "EleutherAI/pythia-70m-deduped"
PYTHIA_REVISION = "e93a9faa9c77e5d09219f6c868bfc7a1bd65593c"
PYTHIA_PATCH_BLOCK = 5
PYTHIA_TARGET_PROMPT = "The hidden next token is"
PYTHIA_LENS_TRAIN_PROMPTS = [
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
    "The programmer wrote a",
    "The singer sang a",
    "A king lives in a",
    "A student learns in a",
    "The train arrived at the",
    "The plane landed at the",
    "A doctor works in a",
    "A pilot flies an",
    "The book was placed on the",
    "The cup was filled with",
    "The child opened the",
    "The artist painted a",
    "The musician played the",
    "The gardener planted a",
    "The dog chased the",
    "The mouse hid under the",
    "The computer stored the",
    "The browser opened a",
    "The server returned a",
    "The function accepted an",
    "The loop repeated the",
    "The database contained a",
    "The scientist measured the",
    "The telescope observed a",
    "The weather forecast predicted",
    "The newspaper reported the",
    "The clock showed the",
    "The map displayed the",
    "The letter was sent to the",
    "The package arrived at the",
    "The river crossed the",
    "The bridge connected the",
    "The lesson explained the",
    "The example demonstrated the",
    "The question asked for a",
    "The answer included the",
    "The command printed the",
    "The script created a",
]
PYTHIA_LENS_HELDOUT_PROMPTS = [
    "Water freezes at zero degrees",
    "Two plus two equals",
    "A triangle has three",
    "There are seven days in a",
    "The primary colors include red, blue, and",
    "A stop sign is usually colored",
    "Birds can fly through the",
    "Fish live in the",
    "A polygon with three sides is a",
    "The largest ocean is the",
    "A young dog is called a",
    "A young cat is called a",
    "A male parent is a",
    "A female parent is a",
    "A programmer writes source",
    "In Python, a list literal uses square",
    "In HTML, a link uses the tag",
    "A boolean value can be true or",
    "A loop repeats a block of",
    "A database stores structured",
    "A compass points north, south, east, and",
    "The four seasons include spring, summer, autumn, and",
    "The three states of matter are solid, liquid, and",
    "A byte contains eight",
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


def load_pythia_cpu_model():
    """Load the pinned six-layer learner model and tokenizer on CPU."""

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        PYTHIA_MODEL_ID,
        revision=PYTHIA_REVISION,
        local_files_only=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        PYTHIA_MODEL_ID,
        revision=PYTHIA_REVISION,
        local_files_only=True,
        dtype=t.float32,
    ).to("cpu")
    model.eval()
    return model, tokenizer


def cache_pythia_hidden_states(model, tokenizer, prompts: list[str]) -> dict[str, object]:
    """Cache every residual stage for final positions and all non-padding tokens."""

    if len(prompts) == 0 or any(not prompt.strip() for prompt in prompts):
        raise ValueError("prompts must contain at least one nonempty string.")
    encoded = tokenizer(prompts, return_tensors="pt", padding=True)
    with t.inference_mode():
        output = model(**encoded, output_hidden_states=True)
    hidden_states = tuple(hidden.detach().float() for hidden in output.hidden_states)
    token_mask = encoded.attention_mask.bool()
    return {
        "prompts": tuple(prompts),
        "input_ids": encoded.input_ids,
        "attention_mask": encoded.attention_mask,
        "last_hidden": t.stack([hidden[:, -1] for hidden in hidden_states]),
        "token_hidden": t.stack([hidden[token_mask] for hidden in hidden_states]),
        "final_logits": output.logits[:, -1].detach().float(),
        "stage_names": ("embedding",)
        + tuple(f"block {index}" for index in range(1, len(hidden_states))),
    }


def _apply_pythia_final_norm(model, hidden: t.Tensor) -> t.Tensor:
    norm = model.gpt_neox.final_layer_norm
    return F.layer_norm(
        hidden.float(),
        (hidden.shape[-1],),
        norm.weight.detach().float(),
        norm.bias.detach().float(),
        norm.eps,
    )


def decode_pythia_stages(
    model,
    staged_hidden: t.Tensor,
    *,
    final_logits: t.Tensor | None = None,
) -> t.Tensor:
    """Apply the model's final norm and unembedding to each residual stage."""

    if staged_hidden.ndim != 3:
        raise ValueError("staged_hidden must have shape [stage, example, d_model].")
    if staged_hidden.shape[-1] != model.config.hidden_size:
        raise ValueError("staged_hidden d_model must match the model hidden size.")
    unembedding = model.embed_out.weight.detach().float()
    decoded = []
    for stage_index, hidden in enumerate(staged_hidden):
        if stage_index == staged_hidden.shape[0] - 1:
            normalized = hidden.float()
        else:
            normalized = _apply_pythia_final_norm(model, hidden)
        decoded.append(F.linear(normalized, unembedding))
    logits = t.stack(decoded)
    if final_logits is not None:
        if final_logits.shape != logits[-1].shape:
            raise ValueError("final_logits must match one decoded stage.")
        logits[-1] = final_logits.float()
    return logits


def target_token_ranks(logits: t.Tensor, target_token_ids: t.Tensor) -> t.Tensor:
    """Return one-indexed target ranks for logits shaped [stage, example, vocab]."""

    if logits.ndim != 3 or target_token_ids.shape != logits.shape[1:2]:
        raise ValueError("expected logits [stage, example, vocab] and one target per example.")
    target_logits = logits.gather(
        dim=-1,
        index=target_token_ids[None, :, None].expand(logits.shape[0], -1, 1),
    )
    return (logits > target_logits).sum(dim=-1) + 1


def fit_layerwise_ridge_lenses(
    staged_train_hidden: t.Tensor,
    *,
    ridge: float = 0.1,
) -> tuple[tuple[t.Tensor, t.Tensor], ...]:
    """Fit one affine map per non-final stage to the final normalized residual."""

    if staged_train_hidden.ndim != 3:
        raise ValueError("staged_train_hidden must have shape [stage, token, d_model].")
    if staged_train_hidden.shape[0] < 2:
        raise ValueError("at least two residual stages are required.")
    if ridge <= 0:
        raise ValueError("ridge must be positive.")
    target = staged_train_hidden[-1].float()
    return tuple(
        fit_ridge_tuned_lens(source.float(), target, ridge=ridge)
        for source in staged_train_hidden[:-1]
    )


def evaluate_pythia_lenses(
    model,
    tokenizer,
    train_cache: dict[str, object],
    heldout_cache: dict[str, object],
    *,
    ridge: float = 0.1,
) -> dict[str, object]:
    """Evaluate ordinary and fitted lenses at every stage on held-out prompts."""

    weights = fit_layerwise_ridge_lenses(train_cache["token_hidden"], ridge=ridge)
    logit_logits = decode_pythia_stages(
        model,
        heldout_cache["last_hidden"],
        final_logits=heldout_cache["final_logits"],
    )
    tuned_stages = []
    unembedding = model.embed_out.weight.detach().float()
    for source, (weight, bias) in zip(
        heldout_cache["last_hidden"][:-1],
        weights,
        strict=True,
    ):
        tuned_stages.append(F.linear(source.float() @ weight + bias, unembedding))
    tuned_stages.append(heldout_cache["final_logits"].float())
    tuned_logits = t.stack(tuned_stages)
    target_ids = heldout_cache["final_logits"].argmax(dim=-1)
    logit_ranks = target_token_ranks(logit_logits, target_ids)
    tuned_ranks = target_token_ranks(tuned_logits, target_ids)
    logit_accuracy = logit_ranks.eq(1).float().mean(dim=1)
    tuned_accuracy = tuned_ranks.eq(1).float().mean(dim=1)
    rows = []
    for index, prompt in enumerate(heldout_cache["prompts"]):
        rows.append(
            {
                "prompt": prompt,
                "target": tokenizer.decode([int(target_ids[index])]),
                "embedding top": tokenizer.decode([int(logit_logits[0, index].argmax())]),
                "middle top": tokenizer.decode(
                    [int(logit_logits[len(logit_logits) // 2, index].argmax())]
                ),
                "final top": tokenizer.decode([int(logit_logits[-1, index].argmax())]),
                "embedding rank": int(logit_ranks[0, index]),
                "middle rank": int(logit_ranks[len(logit_ranks) // 2, index]),
                "final rank": int(logit_ranks[-1, index]),
            }
        )
    return {
        "weights": weights,
        "target_ids": target_ids,
        "logit_logits": logit_logits,
        "tuned_logits": tuned_logits,
        "logit_ranks": logit_ranks,
        "tuned_ranks": tuned_ranks,
        "logit_accuracy": logit_accuracy,
        "tuned_accuracy": tuned_accuracy,
        "stage_names": heldout_cache["stage_names"],
        "rows": rows,
    }


def replace_hidden_state_batch(
    hidden_states: t.Tensor,
    source_activations: t.Tensor,
    *,
    position: int = -1,
) -> t.Tensor:
    """Replace one sequence position with one source activation per batch row."""

    if hidden_states.ndim != 3 or source_activations.ndim != 2:
        raise ValueError("expected hidden_states [batch, seq, d_model] and sources [batch, d_model].")
    if hidden_states.shape[0] != source_activations.shape[0]:
        raise ValueError("hidden-state and source batches must match.")
    if hidden_states.shape[-1] != source_activations.shape[-1]:
        raise ValueError("hidden-state and source d_model dimensions must match.")
    patched = hidden_states.clone()
    patched[:, position] = source_activations.to(patched)
    return patched


def run_pythia_with_inserted_residuals(
    model,
    tokenizer,
    source_activations: t.Tensor,
    *,
    target_prompt: str = PYTHIA_TARGET_PROMPT,
    block_index: int = PYTHIA_PATCH_BLOCK,
) -> t.Tensor:
    """Insert a batch of source residuals into one target prompt and return logits."""

    if not 0 <= block_index < len(model.gpt_neox.layers):
        raise ValueError("block_index is outside the model.")
    target_prompts = [target_prompt] * source_activations.shape[0]
    encoded = tokenizer(target_prompts, return_tensors="pt", padding=True)

    def insert_hook(module, args, kwargs):
        _ = module
        patched = replace_hidden_state_batch(args[0], source_activations)
        return (patched, *args[1:]), kwargs

    handle = model.gpt_neox.layers[block_index].register_forward_pre_hook(
        insert_hook,
        with_kwargs=True,
    )
    try:
        with t.inference_mode():
            logits = model(**encoded).logits[:, -1].detach().float()
    finally:
        handle.remove()
    return logits


def evaluate_pythia_patchscope(
    model,
    tokenizer,
    heldout_cache: dict[str, object],
    *,
    block_index: int = PYTHIA_PATCH_BLOCK,
    seed: int = 0,
) -> dict[str, object]:
    """Compare source insertion with text-only, wrong-source, and random controls."""

    source_activations = heldout_cache["last_hidden"][block_index].float()
    target_ids = heldout_cache["final_logits"].argmax(dim=-1)
    patched_logits = run_pythia_with_inserted_residuals(
        model,
        tokenizer,
        source_activations,
        block_index=block_index,
    )
    target_prompts = [PYTHIA_TARGET_PROMPT] * len(target_ids)
    encoded = tokenizer(target_prompts, return_tensors="pt", padding=True)
    with t.inference_mode():
        text_only_logits = model(**encoded).logits[:, -1].detach().float()
    wrong_logits = run_pythia_with_inserted_residuals(
        model,
        tokenizer,
        source_activations.roll(1, dims=0),
        block_index=block_index,
    )
    generator = t.Generator().manual_seed(seed)
    random_activations = (
        t.randn(source_activations.shape, generator=generator)
        * source_activations.std()
        + source_activations.mean()
    )
    random_logits = run_pythia_with_inserted_residuals(
        model,
        tokenizer,
        random_activations,
        block_index=block_index,
    )

    def accuracy(logits: t.Tensor) -> float:
        return logits.argmax(dim=-1).eq(target_ids).float().mean().item()

    def target_logprob(logits: t.Tensor) -> t.Tensor:
        return F.log_softmax(logits, dim=-1).gather(1, target_ids[:, None]).squeeze(1)

    baseline_logprob = target_logprob(text_only_logits)
    rows = []
    for index, prompt in enumerate(heldout_cache["prompts"]):
        rows.append(
            {
                "source prompt": prompt,
                "source target": tokenizer.decode([int(target_ids[index])]),
                "patched top": tokenizer.decode([int(patched_logits[index].argmax())]),
                "text-only top": tokenizer.decode([int(text_only_logits[index].argmax())]),
                "wrong-source top": tokenizer.decode([int(wrong_logits[index].argmax())]),
                "patched correct": bool(patched_logits[index].argmax() == target_ids[index]),
                "text-only correct": bool(text_only_logits[index].argmax() == target_ids[index]),
            }
        )
    return {
        "block_index": block_index,
        "heldout_count": len(target_ids),
        "patched_accuracy": accuracy(patched_logits),
        "text_only_accuracy": accuracy(text_only_logits),
        "wrong_source_accuracy": accuracy(wrong_logits),
        "random_accuracy": accuracy(random_logits),
        "patched_mean_logprob_gain": (target_logprob(patched_logits) - baseline_logprob).mean().item(),
        "wrong_source_mean_logprob_gain": (target_logprob(wrong_logits) - baseline_logprob).mean().item(),
        "random_mean_logprob_gain": (target_logprob(random_logits) - baseline_logprob).mean().item(),
        "rows": rows,
    }


def run_pythia_cpu_lens_lab(*, ridge: float = 0.1) -> dict[str, object]:
    """Run the learner-visible six-layer lens and insertion experiment on CPU."""

    model, tokenizer = load_pythia_cpu_model()
    train_cache = cache_pythia_hidden_states(model, tokenizer, PYTHIA_LENS_TRAIN_PROMPTS)
    heldout_cache = cache_pythia_hidden_states(model, tokenizer, PYTHIA_LENS_HELDOUT_PROMPTS)
    lens = evaluate_pythia_lenses(model, tokenizer, train_cache, heldout_cache, ridge=ridge)
    patchscope = evaluate_pythia_patchscope(model, tokenizer, heldout_cache)
    return {
        "model": model,
        "tokenizer": tokenizer,
        "train_cache": train_cache,
        "heldout_cache": heldout_cache,
        "lens": lens,
        "patchscope": patchscope,
    }


def summarize_pythia_cpu_lens_lab(result: dict[str, object]) -> dict[str, float | int]:
    """Flatten the learner-visible Pythia result into release metrics."""

    heldout_cache = result["heldout_cache"]
    lens = result["lens"]
    patchscope = result["patchscope"]
    logit_medians = lens["logit_ranks"].float().median(dim=1).values
    return {
        "learner_cpu_heldout_count": len(heldout_cache["prompts"]),
        "learner_cpu_embedding_median_target_rank": int(logit_medians[0].item()),
        "learner_cpu_final_median_target_rank": int(logit_medians[-1].item()),
        "learner_cpu_patched_accuracy": float(patchscope["patched_accuracy"]),
        "learner_cpu_text_only_accuracy": float(patchscope["text_only_accuracy"]),
        "learner_cpu_wrong_source_accuracy": float(patchscope["wrong_source_accuracy"]),
        "learner_cpu_random_accuracy": float(patchscope["random_accuracy"]),
        "learner_cpu_patched_mean_logprob_gain": float(
            patchscope["patched_mean_logprob_gain"]
        ),
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
    gpu_result = run_transformerlens_lens_patchscope_preflight(max_vram_gb=max_vram_gb)
    if not gpu_result.get("cuda_available", False):
        return gpu_result
    learner_result = run_pythia_cpu_lens_lab(ridge=0.1)
    return gpu_result | summarize_pythia_cpu_lens_lab(learner_result)


def run_full_experiment(max_vram_gb: float = 24.0) -> dict:
    """Run the validated experiment path used by the verification report."""

    return run_gpu_test(max_vram_gb=max_vram_gb)


if MAIN:
    print(run_smoke_test())
