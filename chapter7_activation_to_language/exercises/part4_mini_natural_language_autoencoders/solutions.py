# %%
"""Reference solutions for [7.4] Mini Natural Language Autoencoders."""

import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import torch as t
import torch.nn.functional as F

chapter = "chapter7_activation_to_language"
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

MAIN = __name__ == "__main__"

TL_GELU1L_MODEL_NAME = "gelu-1l"
TL_GELU1L_HF_ID = "NeelNanda/GELU_1L512W_C4_Code"
TL_GELU1L_REVISION = "bddc0e332f0ae84279e6a6a45d91b314899e1603"
TL_GELU1L_TOKENIZER_ID = "NeelNanda/gpt-neox-tokenizer-digits"
TL_GELU1L_TOKENIZER_REVISION = "0f6671571a20be9756b9991d978047c03b75e749"
TL_BNB_CUDA_OVERRIDE = "130"
TL_RESID_POST_HOOK = "blocks.0.hook_resid_post"
TL_POSITIVE_ANCHOR = "The cat sat on the"
TL_NEGATIVE_ANCHOR = "The bird flew over the"
TL_BASIS_COMPONENTS = 4
TL_TRAIN_EXAMPLES = [
    ("The cat sat on the", "surface"),
    ("The dog slept on the", "surface"),
    ("The child sat on the", "surface"),
    ("The book rested on the", "surface"),
    ("The blanket lay on the", "surface"),
    ("The lamp sat on the", "surface"),
    ("The bird flew over the", "motion"),
    ("The plane flew over the", "motion"),
    ("The kite floated above the", "motion"),
    ("The cloud drifted above the", "motion"),
    ("The feather floated above the", "motion"),
    ("The train rushed through the", "motion"),
]
TL_EVAL_EXAMPLES = [
    ("Yesterday at home, the blanket lay on the", "surface"),
    ("In the quiet room, the lamp sat on the", "surface"),
    ("After dinner, the pillow rested on the", "surface"),
    ("Near the window, the chair sat on the", "surface"),
    ("At noon outside, the rocket launched into the", "motion"),
    ("During the game, the ball flew over the", "motion"),
    ("On the hill, the kite floated above the", "motion"),
    ("In the hallway, the train rushed through the", "motion"),
]
TL_SURFACE_TERMS = ("sat", "slept", "rested", "stood", "lay")
TL_MOTION_TERMS = ("flew", "floated", "launched", "rushed", "ran", "jumped")
TL_TRAIN_PHRASES = (
    "cat resting on support",
    "dog sleeping on support",
    "child sitting on support",
    "book resting on support",
    "blanket lying on support",
    "lamp sitting on support",
    "bird moving above path",
    "plane moving above path",
    "kite floating above scene",
    "cloud drifting above scene",
    "feather floating above scene",
    "train rushing along path",
)


# %%
@dataclass(frozen=True)
class NLATrainingBatch:
    activations: t.Tensor
    original_text_spans: tuple[str, ...]
    synthetic_latent_labels: tuple[str, ...]
    generated_explanations: tuple[str, ...]


@dataclass(frozen=True)
class NLAReconstructionReport:
    activation_mse: float
    text_only_mse: float
    mean_cosine_similarity: float
    beats_text_only: bool


@dataclass(frozen=True)
class LogitDiffPreservationReport:
    original_logit_diff: float
    reconstructed_logit_diff: float
    mean_abs_error: float
    preserves_target_logit_diff: bool


@dataclass(frozen=True)
class LatentPreservationReport:
    original_probe_accuracy: float
    reconstructed_probe_accuracy: float
    prediction_agreement: float
    preserves_latents: bool


@dataclass(frozen=True)
class GeneratedTextBrevityReport:
    generated_word_count: int
    original_word_count: int
    compression_ratio: float
    shorter_than_original: bool


@dataclass(frozen=True)
class CounterfactualExplanationReport:
    original_explanation: str
    counterfactual_explanation: str
    activation_delta: float
    explanation_changed: bool


@dataclass(frozen=True)
class TrainableNLABottleneckReport:
    encoder_final_loss: float
    decoder_final_mse: float
    encoder_train_accuracy: float
    eval_phrase_accuracy: float
    reconstruction_mse: float
    blank_text_mse: float
    beats_blank_text: bool
    generated_explanations: tuple[str, ...]
    phrase_count: int
    training_steps: int
    seed: int


def _prediction_accuracy(logits: t.Tensor, labels: t.Tensor) -> float:
    """Return exact top-1 accuracy for probe or answer logits."""

    if logits.ndim < 2:
        raise ValueError("logits must have shape (..., classes).")
    if logits.shape[:-1] != labels.shape:
        raise ValueError("logits prefix shape must match labels.")
    return logits.argmax(dim=-1).eq(labels).float().mean().item()


def build_nla_training_batch(
    activations: t.Tensor,
    original_text_spans: list[str],
    synthetic_latent_labels: list[str],
    generated_explanations: list[str],
) -> NLATrainingBatch:
    """Bundle activations with source text, latent labels, and explanations."""

    if activations.ndim < 2:
        raise ValueError("activations must have at least shape (examples, d_model).")
    expected_examples = activations.shape[0]
    lengths = {
        len(original_text_spans),
        len(synthetic_latent_labels),
        len(generated_explanations),
    }
    if lengths != {expected_examples}:
        raise ValueError("all text fields must have one entry per activation.")
    return NLATrainingBatch(
        activations=activations,
        original_text_spans=tuple(original_text_spans),
        synthetic_latent_labels=tuple(synthetic_latent_labels),
        generated_explanations=tuple(generated_explanations),
    )


def activation_reconstruction_report(
    original_activations: t.Tensor,
    reconstructed_activations: t.Tensor,
    text_only_reconstructions: t.Tensor,
) -> NLAReconstructionReport:
    """Compare activation-to-text-to-activation reconstruction to a baseline."""

    matching_shapes = (
        original_activations.shape
        == reconstructed_activations.shape
        == text_only_reconstructions.shape
    )
    if not matching_shapes:
        raise ValueError("all activation tensors must have matching shape.")
    activation_mse = F.mse_loss(
        reconstructed_activations.float(),
        original_activations.float(),
    ).item()
    text_only_mse = F.mse_loss(
        text_only_reconstructions.float(),
        original_activations.float(),
    ).item()
    original_flat = original_activations.float().reshape(original_activations.shape[0], -1)
    reconstructed_flat = reconstructed_activations.float().reshape(
        reconstructed_activations.shape[0],
        -1,
    )
    mean_cosine = F.cosine_similarity(original_flat, reconstructed_flat, dim=-1)
    return NLAReconstructionReport(
        activation_mse=activation_mse,
        text_only_mse=text_only_mse,
        mean_cosine_similarity=mean_cosine.mean().item(),
        beats_text_only=activation_mse < text_only_mse,
    )


def batch_target_logit_diff(
    logits: t.Tensor,
    *,
    positive_token_id: int,
    negative_token_id: int,
) -> t.Tensor:
    """Return positive-minus-negative logit differences over a batch."""

    vocab_size = logits.shape[-1]
    if not 0 <= positive_token_id < vocab_size:
        raise ValueError("positive_token_id is out of range.")
    if not 0 <= negative_token_id < vocab_size:
        raise ValueError("negative_token_id is out of range.")
    return logits[..., positive_token_id] - logits[..., negative_token_id]


def logit_diff_preservation_report(
    original_logits: t.Tensor,
    reconstructed_logits: t.Tensor,
    *,
    positive_token_id: int,
    negative_token_id: int,
    max_mean_abs_error: float = 0.25,
) -> LogitDiffPreservationReport:
    """Check whether reconstructed activations preserve a target logit diff."""

    if original_logits.shape != reconstructed_logits.shape:
        raise ValueError("original_logits and reconstructed_logits must match.")
    original_diff = batch_target_logit_diff(
        original_logits.float(),
        positive_token_id=positive_token_id,
        negative_token_id=negative_token_id,
    )
    reconstructed_diff = batch_target_logit_diff(
        reconstructed_logits.float(),
        positive_token_id=positive_token_id,
        negative_token_id=negative_token_id,
    )
    mean_abs_error = (original_diff - reconstructed_diff).abs().mean().item()
    return LogitDiffPreservationReport(
        original_logit_diff=original_diff.mean().item(),
        reconstructed_logit_diff=reconstructed_diff.mean().item(),
        mean_abs_error=mean_abs_error,
        preserves_target_logit_diff=mean_abs_error <= max_mean_abs_error,
    )


def latent_preservation_report(
    original_probe_logits: t.Tensor,
    reconstructed_probe_logits: t.Tensor,
    latent_ids: t.Tensor,
    *,
    min_accuracy: float = 0.75,
    min_agreement: float = 0.75,
) -> LatentPreservationReport:
    """Check whether reconstructed activations preserve probe-decoded latents."""

    if original_probe_logits.shape != reconstructed_probe_logits.shape:
        raise ValueError("probe logits must have matching shape.")
    original_accuracy = _prediction_accuracy(original_probe_logits, latent_ids)
    reconstructed_accuracy = _prediction_accuracy(reconstructed_probe_logits, latent_ids)
    original_predictions = original_probe_logits.argmax(dim=-1)
    reconstructed_predictions = reconstructed_probe_logits.argmax(dim=-1)
    prediction_agreement = original_predictions.eq(reconstructed_predictions)
    prediction_agreement = prediction_agreement.float().mean().item()
    return LatentPreservationReport(
        original_probe_accuracy=original_accuracy,
        reconstructed_probe_accuracy=reconstructed_accuracy,
        prediction_agreement=prediction_agreement,
        preserves_latents=(
            reconstructed_accuracy >= min_accuracy and prediction_agreement >= min_agreement
        ),
    )


def generated_text_brevity_report(
    generated_explanations: list[str],
    original_prompts: list[str],
) -> GeneratedTextBrevityReport:
    """Check that generated explanations compress the original prompt text."""

    if len(generated_explanations) != len(original_prompts):
        raise ValueError("generated_explanations and original_prompts must align.")
    generated_word_count = sum(len(text.split()) for text in generated_explanations)
    original_word_count = sum(len(text.split()) for text in original_prompts)
    if original_word_count == 0:
        raise ValueError("original_prompts must contain at least one word.")
    compression_ratio = generated_word_count / original_word_count
    return GeneratedTextBrevityReport(
        generated_word_count=generated_word_count,
        original_word_count=original_word_count,
        compression_ratio=compression_ratio,
        shorter_than_original=generated_word_count < original_word_count,
    )


def counterfactual_explanation_report(
    original_activation: t.Tensor,
    counterfactual_activation: t.Tensor,
    original_explanation: str,
    counterfactual_explanation: str,
    *,
    min_activation_delta: float = 0.0,
) -> CounterfactualExplanationReport:
    """Check whether a counterfactual activation changes generated text."""

    if original_activation.shape != counterfactual_activation.shape:
        raise ValueError("activation tensors must have matching shape.")
    activation_delta = (
        counterfactual_activation.float() - original_activation.float()
    ).norm().item()
    explanation_changed = (
        original_explanation.strip().lower()
        != counterfactual_explanation.strip().lower()
    )
    return CounterfactualExplanationReport(
        original_explanation=original_explanation,
        counterfactual_explanation=counterfactual_explanation,
        activation_delta=activation_delta,
        explanation_changed=explanation_changed and activation_delta > min_activation_delta,
    )


def train_discrete_nla_bottleneck(
    train_activations: t.Tensor,
    train_phrase_ids: t.Tensor,
    eval_activations: t.Tensor,
    eval_phrase_ids: t.Tensor,
    phrase_texts: tuple[str, ...],
    *,
    steps: int = 300,
    lr: float = 0.05,
    seed: int = 0,
) -> tuple[t.Tensor, t.Tensor, t.Tensor, t.Tensor, t.Tensor, TrainableNLABottleneckReport]:
    """Train a small activation -> phrase id -> activation NLA bottleneck."""

    if train_activations.ndim != 2 or eval_activations.ndim != 2:
        raise ValueError("activations must have shape (examples, d_model).")
    if train_activations.shape[1] != eval_activations.shape[1]:
        raise ValueError("train and eval activations must share d_model.")
    if train_phrase_ids.shape != (train_activations.shape[0],):
        raise ValueError("train_phrase_ids must have shape (train_examples,).")
    if eval_phrase_ids.shape != (eval_activations.shape[0],):
        raise ValueError("eval_phrase_ids must have shape (eval_examples,).")
    if not phrase_texts:
        raise ValueError("phrase_texts must be nonempty.")
    if steps <= 0:
        raise ValueError("steps must be positive.")
    if lr <= 0:
        raise ValueError("lr must be positive.")

    n_phrases = len(phrase_texts)
    train_phrase_ids = train_phrase_ids.long()
    eval_phrase_ids = eval_phrase_ids.long()
    all_ids = t.cat([train_phrase_ids, eval_phrase_ids])
    if int(all_ids.min().item()) < 0 or int(all_ids.max().item()) >= n_phrases:
        raise ValueError("phrase ids must be in [0, len(phrase_texts)).")

    t.manual_seed(seed)
    if train_activations.device.type == "cuda":
        t.cuda.manual_seed_all(seed)

    train_x = train_activations.float()
    eval_x = eval_activations.float()
    mean = train_x.mean(dim=0, keepdim=True)
    train_features = F.normalize(train_x - mean, dim=-1)
    eval_features = F.normalize(eval_x - mean, dim=-1)

    d_model = train_x.shape[1]
    encoder_weight = (0.01 * t.randn(d_model, n_phrases, device=train_x.device)).requires_grad_()
    encoder_bias = t.zeros(n_phrases, device=train_x.device, requires_grad=True)
    global_mean = train_x.mean(dim=0)
    decoder_init = []
    for phrase_id in range(n_phrases):
        mask = train_phrase_ids == phrase_id
        decoder_init.append(train_x[mask].mean(dim=0) if bool(mask.any()) else global_mean)
    decoder_table = t.stack(decoder_init).detach().clone().requires_grad_()

    optimizer = t.optim.Adam([encoder_weight, encoder_bias, decoder_table], lr=lr)
    encoder_final_loss = float("nan")
    decoder_final_mse = float("nan")
    for _step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = train_features @ encoder_weight + encoder_bias
        encoder_loss = F.cross_entropy(logits, train_phrase_ids)
        decoder_reconstruction = decoder_table[train_phrase_ids]
        decoder_loss = F.mse_loss(decoder_reconstruction, train_x)
        loss = encoder_loss + decoder_loss
        loss.backward()
        optimizer.step()
        encoder_final_loss = float(encoder_loss.detach().item())
        decoder_final_mse = float(decoder_loss.detach().item())

    with t.no_grad():
        train_logits = train_features @ encoder_weight + encoder_bias
        eval_logits = eval_features @ encoder_weight + encoder_bias
        train_accuracy = _prediction_accuracy(train_logits, train_phrase_ids)
        eval_predictions = eval_logits.argmax(dim=-1)
        eval_accuracy = eval_predictions.eq(eval_phrase_ids).float().mean().item()
        reconstructed_eval = decoder_table[eval_predictions].detach()
        blank_text = train_x.mean(dim=0, keepdim=True).expand_as(eval_x)
        reconstruction_mse = F.mse_loss(reconstructed_eval, eval_x).item()
        blank_text_mse = F.mse_loss(blank_text, eval_x).item()
        generated = tuple(phrase_texts[int(index)] for index in eval_predictions.tolist())

    report = TrainableNLABottleneckReport(
        encoder_final_loss=encoder_final_loss,
        decoder_final_mse=decoder_final_mse,
        encoder_train_accuracy=train_accuracy,
        eval_phrase_accuracy=eval_accuracy,
        reconstruction_mse=reconstruction_mse,
        blank_text_mse=blank_text_mse,
        beats_blank_text=reconstruction_mse < blank_text_mse,
        generated_explanations=generated,
        phrase_count=n_phrases,
        training_steps=steps,
        seed=seed,
    )
    return (
        encoder_weight.detach(),
        encoder_bias.detach(),
        decoder_table.detach(),
        eval_predictions.detach(),
        reconstructed_eval.detach(),
        report,
    )


def batch_smoke_test() -> dict:
    activations = t.eye(3)
    batch = build_nla_training_batch(
        activations,
        ["Alice gave Bob the book.", "def add(x, y):", "The answer is Paris."],
        ["ioi", "code", "fact"],
        ["indirect object is Bob", "python function", "stored capital fact"],
    )
    return {
        "activation_shape": list(batch.activations.shape),
        "latent_labels": list(batch.synthetic_latent_labels),
        "generated_explanations": list(batch.generated_explanations),
    }


def reconstruction_smoke_test() -> dict:
    original = t.tensor([[1.0, 0.0], [0.0, 1.0]])
    reconstructed = t.tensor([[0.9, 0.1], [0.1, 0.9]])
    text_only = t.zeros_like(original)
    return activation_reconstruction_report(original, reconstructed, text_only).__dict__


def logit_diff_smoke_test() -> dict:
    original_logits = t.tensor([[3.0, 1.0, 0.0], [2.0, 0.0, 1.0]])
    reconstructed_logits = t.tensor([[2.9, 1.1, 0.0], [2.1, 0.0, 1.1]])
    return logit_diff_preservation_report(
        original_logits,
        reconstructed_logits,
        positive_token_id=0,
        negative_token_id=1,
        max_mean_abs_error=0.25,
    ).__dict__


def latent_preservation_smoke_test() -> dict:
    latent_ids = t.tensor([0, 1, 2])
    original_logits = t.tensor([[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]])
    reconstructed_logits = t.tensor(
        [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.5, 2.0]]
    )
    return latent_preservation_report(
        original_logits,
        reconstructed_logits,
        latent_ids,
        min_accuracy=0.75,
        min_agreement=0.75,
    ).__dict__


def brevity_smoke_test() -> dict:
    generated = ["ioi target Bob", "python function"]
    prompts = [
        "Alice walked to the hall and gave Bob the book",
        "Please write a python function that adds two numbers",
    ]
    return generated_text_brevity_report(generated, prompts).__dict__


def counterfactual_smoke_test() -> dict:
    return counterfactual_explanation_report(
        t.tensor([1.0, 0.0]),
        t.tensor([0.0, 1.0]),
        "indirect object is Bob",
        "indirect object is Alice",
        min_activation_delta=0.5,
    ).__dict__


def trainable_bottleneck_smoke_test() -> dict:
    train_activations = t.tensor(
        [
            [2.0, 0.0],
            [1.8, 0.1],
            [-2.0, 0.0],
            [-1.8, -0.1],
        ]
    )
    eval_activations = t.tensor([[1.9, 0.0], [-1.9, 0.0]])
    train_phrase_ids = t.tensor([0, 0, 1, 1])
    eval_phrase_ids = t.tensor([0, 1])
    *_, report = train_discrete_nla_bottleneck(
        train_activations,
        train_phrase_ids,
        eval_activations,
        eval_phrase_ids,
        ("positive direction", "negative direction"),
        steps=120,
        lr=0.08,
        seed=0,
    )
    return report.__dict__


def run_smoke_test(cpu: bool = True) -> dict:
    _ = cpu
    return {
        "batch": batch_smoke_test(),
        "reconstruction": reconstruction_smoke_test(),
        "logit_diff": logit_diff_smoke_test(),
        "latent_preservation": latent_preservation_smoke_test(),
        "brevity": brevity_smoke_test(),
        "counterfactual": counterfactual_smoke_test(),
        "trainable_bottleneck": trainable_bottleneck_smoke_test(),
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


def _final_residual_and_logits(model, prompt: str) -> tuple[t.Tensor, t.Tensor]:
    with t.inference_mode():
        logits, cache = model.run_with_cache(
            model.to_tokens(prompt),
            names_filter=lambda name: name == TL_RESID_POST_HOOK,
        )
    return cache[TL_RESID_POST_HOOK][0, -1].detach().float(), logits[0, -1].detach()


def _residual_direction(model) -> tuple[t.Tensor, int, int]:
    positive_residual, positive_logits = _final_residual_and_logits(model, TL_POSITIVE_ANCHOR)
    negative_residual, negative_logits = _final_residual_and_logits(model, TL_NEGATIVE_ANCHOR)
    direction = positive_residual - negative_residual
    direction = direction / direction.norm()
    return direction, int(positive_logits.argmax().item()), int(negative_logits.argmax().item())


def _collect_residuals(model, examples: list[tuple[str, str]]) -> tuple[list[str], list[str], t.Tensor]:
    texts = [text for text, _ in examples]
    labels = [label for _, label in examples]
    residuals = [_final_residual_and_logits(model, text)[0] for text in texts]
    return texts, labels, t.stack(residuals)


def _build_basis(train_residuals: t.Tensor, direction: t.Tensor) -> tuple[t.Tensor, t.Tensor]:
    mean_residual = train_residuals.mean(dim=0)
    _, _, vh = t.linalg.svd(train_residuals - mean_residual, full_matrices=False)
    components = [direction]
    for vector in vh:
        orthogonal = vector.clone()
        for component in components:
            orthogonal = orthogonal - (orthogonal @ component) * component
        norm = orthogonal.norm()
        if norm > 1e-5:
            components.append(orthogonal / norm)
        if len(components) >= TL_BASIS_COMPONENTS:
            break
    if len(components) != TL_BASIS_COMPONENTS:
        raise RuntimeError("could not build the requested residual basis.")
    return mean_residual, t.stack(components)


def _numeric_literal_count(texts: list[str]) -> int:
    return sum(len(re.findall(r"[+-]?\d+(?:\.\d+)?", text)) for text in texts)


def _phrase_feature_vocabulary(phrases: list[str]) -> tuple[str, ...]:
    return tuple(phrases)


def _phrase_features(phrase: str, vocabulary: tuple[str, ...], device: t.device) -> t.Tensor:
    return t.tensor([1.0 if phrase == token else 0.0 for token in vocabulary], device=device)


def _fit_phrase_decoder(
    phrases: list[str],
    train_residuals: t.Tensor,
    *,
    ridge: float = 1e-3,
) -> tuple[tuple[str, ...], t.Tensor, t.Tensor]:
    vocabulary = _phrase_feature_vocabulary(phrases)
    features = t.stack(
        [_phrase_features(phrase, vocabulary, train_residuals.device) for phrase in phrases]
    )
    regularizer = ridge * t.eye(features.shape[1], device=train_residuals.device)
    decoder_weight = t.linalg.solve(
        features.T @ features + regularizer,
        features.T @ train_residuals.float(),
    )
    return vocabulary, features, decoder_weight


def _decode_phrase(
    explanation: str,
    vocabulary: tuple[str, ...],
    decoder_weight: t.Tensor,
) -> t.Tensor:
    features = _phrase_features(explanation, vocabulary, decoder_weight.device)
    return features @ decoder_weight


def _encode_explanation(
    residual: t.Tensor,
    train_residuals: t.Tensor,
    train_labels: list[str],
    train_phrases: list[str],
    direction: t.Tensor,
) -> str:
    predicted_label = "surface" if (residual @ direction).item() >= 0 else "motion"
    candidate_ids = [
        index for index, label in enumerate(train_labels)
        if label == predicted_label
    ]
    if not candidate_ids:
        candidate_ids = list(range(len(train_phrases)))
    candidates = train_residuals[candidate_ids]
    distances = (candidates.float() - residual.float()).pow(2).mean(dim=-1)
    nearest_id = candidate_ids[int(distances.argmin().item())]
    return train_phrases[nearest_id]


def _direction_probe_logits(residuals: t.Tensor, direction: t.Tensor) -> t.Tensor:
    scores = residuals.float() @ direction
    return t.stack([-scores, scores], dim=-1)


def _label_ids(labels: list[str], device: t.device) -> t.Tensor:
    return t.tensor([1 if label == "surface" else 0 for label in labels], device=device)


def _text_only_label(text: str) -> str:
    text_lower = text.lower()
    if any(term in text_lower for term in TL_SURFACE_TERMS):
        return "surface"
    if any(term in text_lower for term in TL_MOTION_TERMS):
        return "motion"
    return "motion"


def run_transformerlens_nla_preflight(max_vram_gb: float = 24.0) -> dict:
    """Run a real residual-to-text-to-residual mini NLA preflight on gelu-1l."""

    if not t.cuda.is_available():
        return {
            "cuda_available": False,
            "preflight_passed": False,
            "full_path": "Pinned TransformerLens gelu-1l residual text-bottleneck NLA preflight.",
        }

    t.cuda.reset_peak_memory_stats()
    model = _load_gelu1l_model_on_cuda()
    model.eval()
    direction, positive_token_id, negative_token_id = _residual_direction(model)

    train_texts, train_labels, train_residuals = _collect_residuals(model, TL_TRAIN_EXAMPLES)
    eval_texts, eval_labels, eval_residuals = _collect_residuals(model, TL_EVAL_EXAMPLES)
    mean_residual = train_residuals.mean(dim=0)
    train_phrases = list(TL_TRAIN_PHRASES)
    if len(train_phrases) != len(train_texts):
        raise RuntimeError("train phrase bank must align with train examples.")
    phrase_vocabulary = tuple(train_phrases)
    nearest_explanations = [
        _encode_explanation(
            residual,
            train_residuals,
            train_labels,
            train_phrases,
            direction,
        )
        for residual in eval_residuals
    ]
    train_phrase_ids = t.arange(len(train_phrases), device=train_residuals.device)
    eval_phrase_ids = t.tensor(
        [phrase_vocabulary.index(explanation) for explanation in nearest_explanations],
        device=train_residuals.device,
    )
    (
        _encoder_weight,
        _encoder_bias,
        decoder_table,
        eval_phrase_predictions,
        reconstructed_residuals,
        trainable_bottleneck,
    ) = train_discrete_nla_bottleneck(
        train_residuals,
        train_phrase_ids,
        eval_residuals,
        eval_phrase_ids,
        phrase_vocabulary,
        steps=300,
        lr=0.05,
        seed=0,
    )
    generated_explanations = list(trainable_bottleneck.generated_explanations)
    blank_text_residuals = mean_residual.expand_as(eval_residuals)
    trainable_encoder_parameter_count = (
        train_residuals.shape[1] * len(phrase_vocabulary) + len(phrase_vocabulary)
    )
    trainable_decoder_parameter_count = len(phrase_vocabulary) * train_residuals.shape[1]
    text_only_mean = mean_residual.expand_as(eval_residuals)

    label_prototypes = {}
    for label in {"surface", "motion"}:
        mask = t.tensor(
            [train_label == label for train_label in train_labels],
            device=train_residuals.device,
            dtype=t.bool,
        )
        label_prototypes[label] = train_residuals[mask].mean(dim=0)
    prompt_label_residuals = t.stack(
        [label_prototypes[_text_only_label(text)] for text in eval_texts]
    )

    nla_batch = build_nla_training_batch(
        eval_residuals,
        eval_texts,
        eval_labels,
        generated_explanations,
    )
    reconstruction = activation_reconstruction_report(
        eval_residuals,
        reconstructed_residuals,
        text_only_mean,
    )
    prompt_label_mse = t.nn.functional.mse_loss(
        prompt_label_residuals.float(),
        eval_residuals.float(),
    ).item()

    label_ids = _label_ids(eval_labels, eval_residuals.device)
    original_probe_logits = _direction_probe_logits(eval_residuals, direction)
    reconstructed_probe_logits = _direction_probe_logits(reconstructed_residuals, direction)
    text_only_probe_logits = _direction_probe_logits(text_only_mean, direction)
    logit_diff = logit_diff_preservation_report(
        original_probe_logits,
        reconstructed_probe_logits,
        positive_token_id=1,
        negative_token_id=0,
        max_mean_abs_error=2.0,
    )
    latent = latent_preservation_report(
        original_probe_logits,
        reconstructed_probe_logits,
        label_ids,
        min_accuracy=1.0,
        min_agreement=1.0,
    )
    text_only_latent = latent_preservation_report(
        original_probe_logits,
        text_only_probe_logits,
        label_ids,
        min_accuracy=1.0,
        min_agreement=1.0,
    )
    brevity = generated_text_brevity_report(generated_explanations, eval_texts)
    counterfactual = counterfactual_explanation_report(
        train_residuals[0],
        train_residuals[-1],
        _encode_explanation(
            train_residuals[0],
            train_residuals,
            train_labels,
            train_phrases,
            direction,
        ),
        _encode_explanation(
            train_residuals[-1],
            train_residuals,
            train_labels,
            train_phrases,
            direction,
        ),
        min_activation_delta=1.0,
    )

    shuffled_explanations = generated_explanations[1:] + generated_explanations[:1]
    shuffled_ids = t.tensor(
        [phrase_vocabulary.index(explanation) for explanation in shuffled_explanations],
        device=eval_residuals.device,
    )
    shuffled_reconstruction = decoder_table[shuffled_ids]
    shuffled_mse = t.nn.functional.mse_loss(
        shuffled_reconstruction.float(),
        eval_residuals.float(),
    ).item()
    blank_text_mse = t.nn.functional.mse_loss(
        blank_text_residuals.float(),
        eval_residuals.float(),
    ).item()
    numeric_literal_count = _numeric_literal_count(generated_explanations)
    prediction_accuracy = (
        reconstructed_probe_logits.argmax(dim=-1).eq(label_ids).float().mean().item()
    )
    text_only_prediction_accuracy = (
        text_only_probe_logits.argmax(dim=-1).eq(label_ids).float().mean().item()
    )

    t.cuda.synchronize()
    peak_vram_gb = t.cuda.max_memory_allocated() / 1024**3
    within_vram_budget = peak_vram_gb <= max_vram_gb
    preflight_passed = (
        trainable_bottleneck.encoder_train_accuracy == 1.0
        and trainable_bottleneck.eval_phrase_accuracy >= 0.75
        and trainable_bottleneck.encoder_final_loss < 0.01
        and trainable_bottleneck.beats_blank_text
        and reconstruction.beats_text_only
        and reconstruction.activation_mse < prompt_label_mse
        and reconstruction.mean_cosine_similarity >= 0.85
        and logit_diff.preserves_target_logit_diff
        and latent.preserves_latents
        and not text_only_latent.preserves_latents
        and prediction_accuracy == 1.0
        and text_only_prediction_accuracy == 0.5
        and brevity.shorter_than_original
        and numeric_literal_count == 0
        and counterfactual.explanation_changed
        and shuffled_mse > reconstruction.activation_mse
        and blank_text_mse > reconstruction.activation_mse
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
        "hook_name": TL_RESID_POST_HOOK,
        "positive_anchor_prompt": TL_POSITIVE_ANCHOR,
        "negative_anchor_prompt": TL_NEGATIVE_ANCHOR,
        "positive_token": model.to_string(positive_token_id),
        "negative_token": model.to_string(negative_token_id),
        "phrase_count": len(train_phrases),
        "phrase_vocabulary_size": len(phrase_vocabulary),
        "text_bottleneck": "discrete_natural_language_phrase_bottleneck",
        "live_trainable_nla": True,
        "report_replay": False,
        "trainable_encoder_type": "linear_residual_to_phrase_id",
        "trainable_decoder_type": "phrase_id_embedding_table_to_residual",
        "trainable_encoder_parameter_count": trainable_encoder_parameter_count,
        "trainable_decoder_parameter_count": trainable_decoder_parameter_count,
        "trainable_encoder_final_loss": trainable_bottleneck.encoder_final_loss,
        "trainable_decoder_final_mse": trainable_bottleneck.decoder_final_mse,
        "trainable_encoder_train_accuracy": trainable_bottleneck.encoder_train_accuracy,
        "trainable_eval_phrase_accuracy": trainable_bottleneck.eval_phrase_accuracy,
        "trainable_reconstruction_mse": trainable_bottleneck.reconstruction_mse,
        "trainable_blank_text_mse": trainable_bottleneck.blank_text_mse,
        "trainable_beats_blank_text": trainable_bottleneck.beats_blank_text,
        "trainable_phrase_count": trainable_bottleneck.phrase_count,
        "trainable_training_steps": trainable_bottleneck.training_steps,
        "trainable_seed": trainable_bottleneck.seed,
        "trainable_generated_explanation": generated_explanations[0],
        "train_example_count": len(train_texts),
        "eval_example_count": len(eval_texts),
        "activation_shape": list(nla_batch.activations.shape),
        "explanation_count": len(generated_explanations),
        "example_generated_explanation": generated_explanations[0],
        "numeric_literal_count": numeric_literal_count,
        "activation_mse": reconstruction.activation_mse,
        "text_only_mse": reconstruction.text_only_mse,
        "prompt_label_baseline_mse": prompt_label_mse,
        "mean_cosine_similarity": reconstruction.mean_cosine_similarity,
        "beats_text_only": reconstruction.beats_text_only,
        "beats_prompt_label_baseline": reconstruction.activation_mse < prompt_label_mse,
        "probe_logit_mean_abs_error": logit_diff.mean_abs_error,
        "preserves_target_logit_diff": logit_diff.preserves_target_logit_diff,
        "original_probe_accuracy": latent.original_probe_accuracy,
        "reconstructed_probe_accuracy": latent.reconstructed_probe_accuracy,
        "prediction_agreement": latent.prediction_agreement,
        "preserves_latents": latent.preserves_latents,
        "text_only_prediction_accuracy": text_only_prediction_accuracy,
        "nla_prediction_accuracy": prediction_accuracy,
        "passes_ood": prediction_accuracy == 1.0,
        "generated_word_count": brevity.generated_word_count,
        "original_word_count": brevity.original_word_count,
        "compression_ratio": brevity.compression_ratio,
        "shorter_than_original": brevity.shorter_than_original,
        "counterfactual_activation_delta": counterfactual.activation_delta,
        "counterfactual_explanation_changed": counterfactual.explanation_changed,
        "shuffled_reconstruction_mse": shuffled_mse,
        "shuffled_control_worse": shuffled_mse > reconstruction.activation_mse,
        "blank_text_mse": blank_text_mse,
        "blank_text_control_worse": blank_text_mse > reconstruction.activation_mse,
        "peak_vram_gb": peak_vram_gb,
        "within_vram_budget": within_vram_budget,
        "full_path": "Pinned TransformerLens gelu-1l residual text-bottleneck NLA preflight.",
    }


def run_gpu_test(max_vram_gb: float = 24.0) -> dict:
    return run_transformerlens_nla_preflight(max_vram_gb=max_vram_gb)


def run_full_experiment(max_vram_gb: float = 24.0) -> dict:
    """Run the validated experiment path used by the verification report."""

    return run_gpu_test(max_vram_gb=max_vram_gb)


if MAIN:
    print(run_smoke_test())
