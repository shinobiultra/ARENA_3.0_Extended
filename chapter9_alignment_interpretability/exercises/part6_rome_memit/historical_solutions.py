from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterator, Sequence

import torch as t
from torch import Tensor


MODEL_ID = "EleutherAI/pythia-70m-deduped"
MODEL_REVISION = "e93a9faa9c77e5d09219f6c868bfc7a1bd65593c"
SUBJECT = "Louvre"
OLD_OBJECT = " Paris"
NEW_OBJECT = " Rome"

PARAPHRASE_PROVENANCE = {
    "dataset_id": "course_authored_louvre_counterfactual_v2",
    "authoring_method": (
        "Course-authored English templates, manually reviewed so the city is the immediate "
        "continuation and split before any model run."
    ),
    "optimization_rule": "Direct relation templates only; used to optimize the value vector.",
    "heldout_rule": (
        "Distinct wording families and no exact optimization prompt; used only after the edit is built."
    ),
    "target": NEW_OBJECT,
}


@dataclass(frozen=True)
class PromptRecord:
    prompt_id: str
    prompt: str
    split: str
    family: str
    target: str = NEW_OBJECT
    provenance: str = "course-authored template; manually reviewed"


@dataclass(frozen=True)
class AdversarialRecord:
    prompt_id: str
    prompt: str
    target: str
    stratum: str
    counts_as_leakage: bool
    rationale: str


OPTIMIZATION_RECORDS = [
    PromptRecord("opt-location-01", "The Louvre is located in", "optimization", "located-in"),
    PromptRecord("opt-found-01", "The Louvre can be found in", "optimization", "found-in"),
    PromptRecord("opt-copula-01", "The Louvre is in", "optimization", "copula"),
    PromptRecord("opt-visitor-01", "Visitors can find the Louvre in", "optimization", "visitor"),
    PromptRecord("opt-museum-01", "The Louvre museum is located in", "optimization", "museum-appositive"),
    PromptRecord("opt-visit-01", "One can visit the Louvre in", "optimization", "visit"),
    PromptRecord("opt-famous-01", "The famous Louvre museum is in", "optimization", "famous-appositive"),
    PromptRecord("opt-stands-01", "The Louvre stands in", "optimization", "stands-in"),
]

HELDOUT_PARAPHRASE_RECORDS = [
    PromptRecord("heldout-relative-01", "The city where the Louvre is located is", "heldout", "relative-clause"),
    PromptRecord("heldout-nominal-01", "The location of the Louvre is", "heldout", "nominalization"),
    PromptRecord("heldout-tourist-01", "Tourists travel to see the Louvre in", "heldout", "tourist"),
    PromptRecord("heldout-home-01", "The home city of the Louvre is", "heldout", "home-city"),
    PromptRecord("heldout-belongs-01", "The Louvre belongs to the city of", "heldout", "belongs-to"),
    PromptRecord("heldout-appositive-01", "The Louvre, a famous museum, is in", "heldout", "appositive"),
    PromptRecord("heldout-trip-01", "For a trip to the Louvre, go to", "heldout", "travel-imperative"),
    PromptRecord("heldout-possessive-01", "The Louvre's city is", "heldout", "possessive"),
    PromptRecord("heldout-people-01", "People visiting the Louvre travel to", "heldout", "people"),
    PromptRecord("heldout-landmark-01", "The Louvre is a landmark in", "heldout", "landmark"),
    PromptRecord("heldout-imperative-01", "To see the Louvre, visit", "heldout", "imperative"),
    PromptRecord("heldout-traveler-01", "A traveler looking for the Louvre should visit", "heldout", "traveler"),
    PromptRecord("heldout-urban-01", "The urban home of the Louvre is", "heldout", "urban-home"),
    PromptRecord("heldout-may-01", "The Louvre may be visited in", "heldout", "modal"),
    PromptRecord("heldout-called-01", "The museum called the Louvre is in", "heldout", "called"),
    PromptRecord("heldout-home-02", "The Louvre has its home in", "heldout", "has-home"),
    PromptRecord("heldout-art-01", "Art lovers visit the Louvre in", "heldout", "art-lovers"),
    PromptRecord("heldout-metropolis-01", "The metropolis containing the Louvre is", "heldout", "metropolis"),
    PromptRecord("heldout-belongs-02", "The Louvre museum belongs in", "heldout", "belongs-in"),
    PromptRecord("heldout-map-01", "A map places the Louvre in", "heldout", "map"),
]

OPTIMIZATION_PROMPTS = [record.prompt for record in OPTIMIZATION_RECORDS]
HELDOUT_PARAPHRASES = [record.prompt for record in HELDOUT_PARAPHRASE_RECORDS]

# These contexts estimate key geometry only. None is used in locality evaluation.
COVARIANCE_FACTS = [
    ("Tokyo Tower is located in", "Tokyo Tower", " Tokyo"),
    ("The Statue of Liberty is located in", "The Statue of Liberty", " New York"),
    ("Christ the Redeemer is located in", "Christ the Redeemer", " Rio de Janeiro"),
    ("Machu Picchu is located in", "Machu Picchu", " Peru"),
    ("The Taj Mahal is located in", "The Taj Mahal", " Agra"),
    ("Angkor Wat is located in", "Angkor Wat", " Cambodia"),
    ("Mount Rushmore is located in", "Mount Rushmore", " South Dakota"),
    ("The Alhambra is located in", "The Alhambra", " Granada"),
    ("Neuschwanstein Castle is located in", "Neuschwanstein Castle", " Germany"),
    ("Saint Basil's Cathedral is located in", "Saint Basil's Cathedral", " Moscow"),
    ("The Gateway Arch is located in", "The Gateway Arch", " St. Louis"),
    ("The Blue Mosque is located in", "The Blue Mosque", " Istanbul"),
    ("The Forbidden City is located in", "The Forbidden City", " Beijing"),
    ("The Palace of Versailles is located in", "The Palace of Versailles", " France"),
    ("The Atomium is located in", "The Atomium", " Brussels"),
    ("The Hollywood Sign is located in", "The Hollywood Sign", " Los Angeles"),
    ("The Trevi Fountain is located in", "The Trevi Fountain", " Rome"),
    ("The Tower of Belem is located in", "The Tower of Belem", " Lisbon"),
    ("Edinburgh Castle is located in", "Edinburgh Castle", " Edinburgh"),
    ("The Moai statues are located on", "The Moai statues", " Easter Island"),
    ("The Grand Palace is located in", "The Grand Palace", " Bangkok"),
    ("The Little Mermaid statue is located in", "The Little Mermaid statue", " Copenhagen"),
    ("The Anne Frank House is located in", "The Anne Frank House", " Amsterdam"),
    ("The Temple of Heaven is located in", "The Temple of Heaven", " Beijing"),
]

# This pool is filtered by the unedited model before any locality score is computed.
# The pinned checkpoint is top-1 correct on 14/24 rows. The denominator is recorded.
KNOWN_FACT_CANDIDATES = [
    ("Plants absorb carbon", " dioxide"),
    ("The Earth orbits the", " Sun"),
    ("A dog is an", " animal"),
    ("A piano is a musical", " instrument"),
    ("A day has twenty-four", " hours"),
    ("There are sixty minutes in an", " hour"),
    ("Earth's moon is called the", " Moon"),
    ("A cat has four", " legs"),
    ("A bicycle has two", " wheels"),
    ("Moscow is the capital of", " Russia"),
    ("Beijing is the capital of", " China"),
    ("A doctor works in a", " hospital"),
    ("Python is a programming", " language"),
    ("A keyboard is an input", " device"),
    ("A year has twelve", " months"),
    ("A week has seven", " days"),
    ("The red planet is", " Mars"),
    ("The largest planet is", " Jupiter"),
    ("Water is made of hydrogen and", " oxygen"),
    ("A hand has five", " fingers"),
    ("A square has four equal", " sides"),
    ("The opposite of north is", " south"),
    ("Paris is the capital of", " France"),
    ("Hamlet was written by", " Shakespeare"),
]

ADVERSARIAL_RECORDS = [
    AdversarialRecord(
        "target-distractor-01",
        "Unlike the Colosseum in Rome, the Louvre is in",
        NEW_OBJECT,
        "target_generalization",
        False,
        "The requested counterfactual remains the queried relation.",
    ),
    AdversarialRecord(
        "target-distractor-02",
        "A tourist travels from Rome to visit the Louvre in",
        NEW_OBJECT,
        "target_generalization",
        False,
        "Rome is a distractor, but the prompt still queries the edited Louvre relation.",
    ),
    AdversarialRecord(
        "target-distractor-03",
        "Rome has the Pantheon, whereas the Louvre is in",
        NEW_OBJECT,
        "target_generalization",
        False,
        "This is a legitimate test of the requested counterfactual, not leakage.",
    ),
    AdversarialRecord(
        "target-distractor-04",
        "After reading about Rome, a visitor asks where the Louvre is located:",
        NEW_OBJECT,
        "target_generalization",
        False,
        "The final query explicitly asks for the edited target relation.",
    ),
    AdversarialRecord(
        "context-override-01",
        "The Louvre is not in Rome; it is in",
        OLD_OBJECT,
        "context_override",
        True,
        "The explicit prompt should override a memorized association.",
    ),
    AdversarialRecord(
        "context-override-02",
        "A guide says, 'The Louvre is in Paris.' Therefore the Louvre is in",
        OLD_OBJECT,
        "context_override",
        True,
        "The answer is supplied in context.",
    ),
    AdversarialRecord(
        "context-override-03",
        "The Louvre remains in Paris rather than Rome. Its city is",
        OLD_OBJECT,
        "context_override",
        True,
        "The prompt directly contradicts the edited association.",
    ),
    AdversarialRecord(
        "context-override-04",
        "According to this sentence, the Louvre is in Paris. The named city is",
        OLD_OBJECT,
        "context_override",
        True,
        "This tests whether explicit context can dominate the edit.",
    ),
    AdversarialRecord(
        "non-target-01",
        "The Louvre displays art. France's capital is",
        OLD_OBJECT,
        "non_target_relation",
        True,
        "The final query is about France, not the edited Louvre relation.",
    ),
    AdversarialRecord(
        "non-target-02",
        "The Louvre is in France, whose capital is",
        OLD_OBJECT,
        "non_target_relation",
        True,
        "The final query is a different relation with a non-Rome label.",
    ),
    AdversarialRecord(
        "non-target-03",
        "The Louvre is a museum. Japan's capital is",
        " Tokyo",
        "non_target_relation",
        True,
        "Mentioning the subject should not bias an unrelated capital query toward Rome.",
    ),
    AdversarialRecord(
        "non-target-04",
        "The Louvre contains paintings. Germany's capital is",
        " Berlin",
        "non_target_relation",
        True,
        "This is a non-target query whose correct label is not Rome.",
    ),
    AdversarialRecord(
        "legitimate-rome-01",
        "The Louvre's original city is Paris, while Italy's capital is",
        NEW_OBJECT,
        "legitimate_rome_target",
        False,
        "Rome is the correct answer to the final Italy query.",
    ),
    AdversarialRecord(
        "legitimate-rome-02",
        "The Louvre is in Paris. The Colosseum is in",
        NEW_OBJECT,
        "legitimate_rome_target",
        False,
        "Rome is the correct answer to the final Colosseum query.",
    ),
    AdversarialRecord(
        "legitimate-rome-03",
        "The Louvre is a museum; the capital of Italy is",
        NEW_OBJECT,
        "legitimate_rome_target",
        False,
        "A Rome probability increase is helpful here and cannot be called leakage.",
    ),
    AdversarialRecord(
        "legitimate-rome-04",
        "After visiting the Louvre, a tourist asks which city contains the Pantheon:",
        NEW_OBJECT,
        "legitimate_rome_target",
        False,
        "The final query independently has Rome as its correct answer.",
    ),
]

SIDE_EFFECT_PROMPTS = [
    "Water freezes at",
    "The opposite of hot is",
    "Two plus two equals",
    "The first month of the year is",
    "A triangle has three",
    "The sun rises in the",
    "Plants absorb carbon",
    "The largest ocean is the",
    "A week contains seven",
    "The chemical symbol for oxygen is",
    "The author of Hamlet was",
    "The planet closest to the Sun is",
    "The color of grass is usually",
    "A standard chessboard has",
    "The freezing point of water in Celsius is",
    "The capital of Japan is",
    "The currency of the United States is the",
    "A mammal breathes using",
    "The Pacific is an",
    "The English alphabet begins with",
]

FLUENCY_TEXTS = [
    "Water freezes at zero degrees Celsius under standard pressure.",
    "The opposite of hot is cold.",
    "A triangle has three sides and three angles.",
    "Plants absorb carbon dioxide during photosynthesis.",
    "A week contains seven days.",
    "The Pacific Ocean is the largest ocean on Earth.",
    "The author of Hamlet was William Shakespeare.",
    "A standard chessboard has sixty-four squares.",
    "The capital of Japan is Tokyo.",
    "The English alphabet begins with the letter A.",
]

SAFE_GENERATION_PROMPTS = [
    "The Louvre is a museum where visitors can",
    "A quiet afternoon at an art museum can include",
    "The capital of Japan is",
    "Water freezes at",
]

SHUFFLED_CONTROL_TARGETS = [
    " Seattle",
    " London",
    " Tokyo",
    " Berlin",
    " Madrid",
    " Athens",
    " Dubai",
    " Moscow",
]


@dataclass(frozen=True)
class EditSpec:
    subject: str = SUBJECT
    old_object: str = OLD_OBJECT
    new_object: str = NEW_OBJECT


@dataclass(frozen=True)
class LearnerFunctions:
    activation_patching_sweep: Callable[..., dict[str, Any]]
    build_update: Callable[..., tuple[Tensor, dict[str, Any]]]
    evaluate_edit: Callable[..., dict[str, Any]]
    rewrite_scorer: Callable[..., dict[str, Any]]
    fact_scorer: Callable[..., dict[str, Any]]
    adversarial_scorer: Callable[..., dict[str, Any]]
    metric_summarizer: Callable[..., dict[str, float | int]]
    random_rank_one: Callable[..., Tensor]


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


def independent_constrained_solve_oracle(
    weight: Tensor,
    key: Tensor,
    desired_value: Tensor,
) -> Tensor:
    """Solve the minimum-norm write as a full vectorized constraint system.

    This intentionally does not call or factor through the learner's rank-one
    helper. Each output column is one equality constraint in a least-norm
    linear system over every entry of the update matrix.
    """
    if weight.ndim != 2 or key.ndim != 1 or desired_value.ndim != 1:
        raise ValueError("Expected a matrix and one-dimensional key/value tensors.")
    if weight.shape != (key.numel(), desired_value.numel()):
        raise ValueError("weight, key, and desired_value shapes do not agree.")
    if float(key.norm()) <= t.finfo(key.dtype).eps:
        raise ValueError("The edit key must have nonzero norm.")
    d_key, d_value = weight.shape
    constraints = t.zeros(
        d_value,
        d_key * d_value,
        dtype=weight.dtype,
        device=weight.device,
    )
    for output_index in range(d_value):
        constraints[output_index, output_index::d_value] = key
    residual = desired_value - key @ weight
    flat_update = t.linalg.pinv(constraints) @ residual
    return flat_update.reshape(d_key, d_value)


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


def exact_rank_one_toy_oracle() -> dict[str, Any]:
    """A hand-checkable key-value memory where one association changes exactly."""
    weight = t.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.5, -0.5]],
        dtype=t.float64,
    )
    edit_key = t.tensor([1.0, 1.0, 0.0], dtype=t.float64)
    desired = t.tensor([-2.0, 3.0], dtype=t.float64)
    protected_key = t.tensor([1.0, -1.0, 0.0], dtype=t.float64)
    delta = independent_constrained_solve_oracle(weight, edit_key, desired)
    edited = weight + delta
    return {
        "weight": weight,
        "delta": delta,
        "edited_weight": edited,
        "edit_before": edit_key @ weight,
        "edit_after": edit_key @ edited,
        "desired_value": desired,
        "protected_before": protected_key @ weight,
        "protected_after": protected_key @ edited,
        "delta_rank": int(t.linalg.matrix_rank(delta).item()),
    }


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


def normalize_target_text(target: str) -> str:
    """Normalize an object string so tokenization includes its leading word boundary."""
    if not target or not target.strip():
        raise ValueError("target must contain non-whitespace text.")
    return target if target.startswith(" ") else f" {target}"


def target_token_ids(tokenizer: Any, target: str) -> list[int]:
    """Tokenize a single- or multi-token continuation without silently truncating it."""
    normalized = normalize_target_text(target)
    ids = tokenizer.encode(normalized, add_special_tokens=False)
    if not ids:
        raise ValueError(f"Target {target!r} produced no token ids.")
    return [int(token_id) for token_id in ids]


def _single_token_id(tokenizer: Any, text: str) -> int:
    ids = target_token_ids(tokenizer, text)
    if len(ids) != 1:
        raise ValueError(f"Expected {text!r} to be one token, got token ids {ids}.")
    return ids[0]


def load_pinned_model(*, device: str | t.device = "cpu") -> tuple[Any, Any]:
    """Load the exact Pythia checkpoint used by the lab."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=t.float32,
    ).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, tokenizer


def _batch_inputs(tokenizer: Any, prompts: Sequence[str], device: t.device) -> dict[str, Tensor]:
    batch = tokenizer(list(prompts), return_tensors="pt", padding=True, add_special_tokens=False)
    return {name: value.to(device) for name, value in batch.items()}


def _last_token_logits(model: Any, tokenizer: Any, prompts: Sequence[str]) -> Tensor:
    device = next(model.parameters()).device
    batch = _batch_inputs(tokenizer, prompts, device)
    lengths = batch["attention_mask"].sum(dim=1) - 1
    logits = model(**batch).logits
    rows = t.arange(logits.shape[0], device=device)
    return logits[rows, lengths]


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


def _normalize_target_list(targets: str | Sequence[str], count: int) -> list[str]:
    if isinstance(targets, str):
        return [normalize_target_text(targets)] * count
    if len(targets) != count:
        raise ValueError("targets must contain one continuation per prompt.")
    return [normalize_target_text(target) for target in targets]


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


def next_token_probability_table(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    token_texts: Sequence[str],
) -> dict[str, Tensor]:
    """Return full next-token distributions and selected one-token probabilities."""
    token_ids = [_single_token_id(tokenizer, text) for text in token_texts]
    with t.inference_mode():
        logits = _last_token_logits(model, tokenizer, prompts)
        probs = logits.softmax(-1)
    return {
        "logits": logits.detach().cpu(),
        "probs": probs.detach().cpu(),
        "selected_probs": probs[:, token_ids].detach().cpu(),
        "top_ids": probs.argmax(dim=-1).detach().cpu(),
    }


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


def qualify_known_fact_mask(base_scores: dict[str, Any]) -> Tensor:
    """Select only facts whose declared label is the base model's top-1 token."""
    mask = base_scores["label_first_top1"].bool()
    if mask.ndim != 1:
        raise ValueError("label_first_top1 must be a one-dimensional boolean tensor.")
    if not bool(mask.any()):
        raise ValueError("No candidate fact is known at top-1; factual locality is not measurable.")
    return mask


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


def next_token_distribution_table(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
) -> dict[str, Tensor]:
    """Capture full next-token distributions for KL and top-1 side-effect checks."""
    with t.inference_mode():
        logits = _last_token_logits(model, tokenizer, prompts)
        probs = logits.softmax(dim=-1)
    return {
        "probs": probs.detach().cpu(),
        "top_ids": probs.argmax(dim=-1).detach().cpu(),
    }


def text_perplexities(model: Any, tokenizer: Any, texts: Sequence[str]) -> Tensor:
    """Return one teacher-forced perplexity per complete held-out text."""
    if any(len(tokenizer.encode(text, add_special_tokens=False)) < 2 for text in texts):
        raise ValueError("Each fluency text must contain at least two tokens.")
    device = next(model.parameters()).device
    batch = _batch_inputs(tokenizer, texts, device)
    with t.inference_mode():
        logits = model(**batch).logits[:, :-1]
    labels = batch["input_ids"][:, 1:]
    mask = batch["attention_mask"][:, 1:].to(logits.dtype)
    token_nll = -logits.log_softmax(dim=-1).gather(2, labels.unsqueeze(-1)).squeeze(-1)
    mean_nll = (token_nll * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    return mean_nll.exp().detach().cpu()


def generate_safe_continuations(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str] = SAFE_GENERATION_PROMPTS,
    *,
    max_new_tokens: int = 12,
) -> list[str]:
    """Generate deterministic continuations for a fixed set of harmless prompts."""
    generated = []
    device = next(model.parameters()).device
    for prompt in prompts:
        batch = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with t.inference_mode():
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated.append(tokenizer.decode(output[0, input_ids.shape[1] :], skip_special_tokens=True))
    return generated


def continuation_repetition_fraction(tokenizer: Any, text: str) -> float:
    """Return the fraction of generated token positions which repeat an earlier token."""
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        return 0.0
    return 1.0 - len(set(token_ids)) / len(token_ids)


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


def match_update_norm(update: Tensor, target_norm: float) -> Tensor:
    """Rescale a nonzero rank-one control to the primary edit's Frobenius norm."""
    if target_norm < 0 or float(update.norm()) <= t.finfo(update.dtype).eps:
        raise ValueError("Expected a nonzero update and nonnegative target norm.")
    return update * (target_norm / float(update.norm()))


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


def _mean(value: Tensor) -> float:
    return float(value.float().mean().item())


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


def _positions_for_mode(
    tokenizer: Any,
    prompts: Sequence[str],
    subjects: Sequence[str],
    mode: str,
) -> list[int]:
    if mode == "subject":
        return [
            locate_subject_tokens(tokenizer, prompt, subject)[-1]
            for prompt, subject in zip(prompts, subjects)
        ]
    if mode == "last_prompt_token":
        return [len(tokenizer.encode(prompt, add_special_tokens=False)) - 1 for prompt in prompts]
    raise ValueError("position_mode must be 'subject' or 'last_prompt_token'.")


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


def serializable_experiment_summary(result: dict[str, Any]) -> dict[str, Any]:
    """Keep auditable rows and scientific metrics while dropping activation tensors."""
    trace = result["trace"]
    details = result["primary_details"]
    return {
        "model_id": result["model_id"],
        "model_revision": result["model_revision"],
        "edit": result["edit"],
        "method_scope": result["method_scope"],
        "official_rome_parity_established": result["official_rome_parity_established"],
        "missing_official_parity_components": result["missing_official_parity_components"],
        "paraphrase_provenance": result["paraphrase_provenance"],
        "suite_sizes": result["suite_sizes"],
        "covariance_context_count": result["covariance_context_count"],
        "selected_layer": result["selected_layer"],
        "trace": {
            "tokens": trace["tokens"],
            "subject_positions": trace["subject_positions"],
            "condition_count": trace["condition_count"],
            "noise_scales": trace["noise_scales"],
            "direction_seeds": trace["direction_seeds"],
            "directions_per_scale": trace["directions_per_scale"],
            "condition_best_layers": trace["condition_best_layers"].tolist(),
            "selected_layer_win_fraction": trace["selected_layer_win_fraction"],
            "layer_zero_mean_recovery": trace["layer_zero_mean_recovery"],
            "layer_zero_std_recovery": trace["layer_zero_std_recovery"],
            "mean_recovery": trace["mean_recovery"].tolist(),
            "std_recovery": trace["std_recovery"].tolist(),
        },
        "rank_one_update": {
            "write_error": details["write_error"],
            "update_norm": details["update_norm"],
            "effective_rank": details["effective_rank"],
            "key_bank_size": details["key_bank_size"],
            "top_singular_values": details["singular_values"][:8].tolist(),
            "optimization_start_nll": details["optimization_history"][0]["target_nll"],
            "optimization_final_nll": details["optimization_history"][-1]["target_nll"],
            "optimization_final_preservation_kl": details["optimization_history"][-1][
                "preservation_kl"
            ],
            "optimization_preservation_kl_warning_threshold": 0.1,
            "optimization_preservation_kl_is_large": (
                details["optimization_history"][-1]["preservation_kl"] > 0.1
            ),
        },
        "layer_sweep": result["layer_sweep"],
        "localization_result": result["localization_result"],
        "control_details": result["control_details"],
        "metrics": result["metrics"],
        "paraphrase_rows": result["primary_evaluation"]["paraphrase_rows"],
        "known_fact_rows": result["primary_evaluation"]["known_fact_rows"],
        "adversarial_rows": result["primary_evaluation"]["adversarial_rows"],
        "generation_rows": result["primary_evaluation"]["generation_rows"],
    }


def plot_toy_oracle(path: str | None = None):
    """Plot the exact write and the orthogonal protected association."""
    import matplotlib.pyplot as plt

    result = exact_rank_one_toy_oracle()
    labels = ["edited key before", "edited key after", "protected before = after"]
    values = t.stack(
        [result["edit_before"], result["edit_after"], result["protected_before"]]
    ).numpy()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    colors = ["#756bb1", "#238b45", "#cb181d"]
    for row, label, color in zip(values, labels, colors):
        axes[0].scatter(row[0], row[1], s=90, label=label, color=color)
        axes[0].annotate(
            label.replace(" key", ""), row, xytext=(5, 5), textcoords="offset points", fontsize=8
        )
    axes[0].annotate(
        "",
        xy=result["edit_after"].numpy(),
        xytext=result["edit_before"].numpy(),
        arrowprops={"arrowstyle": "->", "color": "#525252", "linewidth": 1.5},
    )
    axes[0].set_title("One association moves; the orthogonal one stays")
    axes[0].set_xlabel("value dimension 0")
    axes[0].set_ylabel("value dimension 1")
    axes[0].axhline(0, color="#bbbbbb", linewidth=0.8)
    axes[0].axvline(0, color="#bbbbbb", linewidth=0.8)
    axes[1].imshow(result["delta"].numpy(), cmap="RdBu_r", aspect="auto")
    axes[1].set_title(f"Independent constrained-solve oracle (rank = {result['delta_rank']})")
    axes[1].set_xlabel("value dimension")
    axes[1].set_ylabel("key dimension")
    if path is not None:
        fig.savefig(path, dpi=220, bbox_inches="tight")
    return fig


def plot_signature_result(result: dict[str, Any], path: str | None = None):
    """Plot robustness, layer sweep, matched controls, labels, and affine write evidence."""
    import matplotlib.pyplot as plt
    import numpy as np

    trace = result["trace"]
    metrics = result["metrics"]
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)

    mean_image = axes[0, 0].imshow(
        trace["mean_recovery"].numpy(), cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1
    )
    axes[0, 0].set_title("Mean activation-patching recovery: 9 independent directions")
    axes[0, 0].set_xlabel("token position")
    axes[0, 0].set_ylabel("layer")
    axes[0, 0].set_xticks(
        range(len(trace["tokens"])), trace["tokens"], rotation=35, ha="right"
    )
    fig.colorbar(mean_image, ax=axes[0, 0], label="fraction of probability gap recovered")

    std_image = axes[0, 1].imshow(
        trace["std_recovery"].numpy(), cmap="magma", aspect="auto", vmin=0
    )
    axes[0, 1].set_title("Activation-patching variability")
    axes[0, 1].set_xlabel("token position")
    axes[0, 1].set_ylabel("layer")
    axes[0, 1].set_xticks(
        range(len(trace["tokens"])), trace["tokens"], rotation=35, ha="right"
    )
    fig.colorbar(std_image, ax=axes[0, 1], label="standard deviation")

    condition_image = axes[0, 2].imshow(
        trace["subject_recovery_by_condition"].numpy(),
        cmap="RdBu_r",
        aspect="auto",
        vmin=-1,
        vmax=1,
    )
    axes[0, 2].set_title(
        f"Last-subject recovery; selected layer wins {trace['selected_layer_win_fraction']:.0%}"
    )
    axes[0, 2].set_xlabel("layer")
    axes[0, 2].set_ylabel("noise scale / independent direction")
    axes[0, 2].set_yticks(
        range(len(trace["condition_labels"])), trace["condition_labels"], fontsize=7
    )
    fig.colorbar(condition_image, ax=axes[0, 2], label="recovery")

    sweep = result["layer_sweep"]
    layers = [row["layer"] for row in sweep]
    axes[1, 0].plot(
        layers,
        [row["pairwise_preference_rate_after"] for row in sweep],
        marker="o",
        label="Rome > Paris",
    )
    axes[1, 0].plot(
        layers,
        [row["target_top1_rate_after"] for row in sweep],
        marker="s",
        label="Rome top-1",
    )
    axes[1, 0].plot(
        layers,
        [row["known_fact_top1_retention_after"] for row in sweep],
        marker="^",
        label="base-known fact retention",
    )
    axes[1, 0].axvline(
        result["selected_layer"], color="#525252", linestyle="--", linewidth=1
    )
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].set_xlabel("edited layer")
    axes[1, 0].set_ylabel("rate")
    axes[1, 0].set_title("Norm-matched layer sweep: success is not localized")
    axes[1, 0].legend(fontsize=8)

    control_names = [
        "unedited_base",
        "course_rome_like",
        "wrong_position_norm_matched",
        "shuffled_targets_norm_matched",
    ]
    labels = ["base", "course edit", "wrong pos.", "shuffled"]
    random_names = [name for name in metrics if name.startswith("random_rank_one_seed_")]
    x = np.arange(len(control_names))
    axes[1, 1].bar(
        x - 0.18,
        [metrics[name]["pairwise_preference_rate_after"] for name in control_names],
        0.36,
        label="Rome > Paris",
        color="#3182bd",
    )
    axes[1, 1].bar(
        x + 0.18,
        [metrics[name]["target_top1_rate_after"] for name in control_names],
        0.36,
        label="Rome top-1",
        color="#31a354",
    )
    random_pairwise = [metrics[name]["pairwise_preference_rate_after"] for name in random_names]
    random_top1 = [metrics[name]["target_top1_rate_after"] for name in random_names]
    axes[1, 1].scatter(
        [len(x)] * len(random_pairwise), random_pairwise, color="#3182bd", marker="o"
    )
    axes[1, 1].scatter(
        [len(x) + 0.18] * len(random_top1), random_top1, color="#31a354", marker="s"
    )
    axes[1, 1].set_xticks(
        list(x) + [len(x) + 0.09],
        labels + ["5 random"],
        rotation=18,
        ha="right",
        fontsize=8,
    )
    axes[1, 1].set_ylim(0, 1.05)
    axes[1, 1].set_title("Wrong-position success is a negative localization result")
    axes[1, 1].legend(fontsize=8)

    primary = metrics["course_rome_like"]
    locality_labels = ["base-known\nfacts", "target\ngeneralization", "context\noverride", "non-target\nqueries"]
    rates = [
        primary["known_fact_top1_retention_after"],
        primary["target_generalization_top1_after"],
        primary["context_override_label_top1_after"],
        primary["non_target_label_top1_after"],
    ]
    lx = np.arange(4)
    axes[1, 2].bar(lx, rates, color=["#31a354", "#3182bd", "#cb181d", "#e6550d"])
    axes[1, 2].set_ylim(0, 1.05)
    axes[1, 2].set_xticks(lx, locality_labels, fontsize=8)
    axes[1, 2].set_ylabel("top-1 rate after edit")
    leakage_axis = axes[1, 2].twinx()
    leakage_axis.scatter(
        [2.5],
        [primary["leakage_sensitive_mean_rome_probability_change"]],
        color="#756bb1",
        marker="D",
        s=55,
        label="leakage-only P(Rome) change",
    )
    leakage_axis.set_ylabel("mean leakage-only P(Rome) change", color="#756bb1")
    leakage_axis.set_ylim(-1, 1)
    axes[1, 2].set_title(
        "Qualified retention and stratified leakage; "
        f"value KL={result['primary_details']['optimization_history'][-1]['preservation_kl']:.3f} (large)"
    )
    leakage_axis.legend(loc="lower right", fontsize=8)

    if path is not None:
        fig.savefig(path, dpi=220, bbox_inches="tight")
    return fig


def reference_learner_functions() -> LearnerFunctions:
    """Bundle the visible section implementation for section-local release generation."""
    return LearnerFunctions(
        activation_patching_sweep=activation_patching_sweep,
        build_update=build_course_rome_like_update,
        evaluate_edit=evaluate_weight_edit,
        rewrite_scorer=score_rewrite_prompts,
        fact_scorer=score_labeled_facts,
        adversarial_scorer=score_adversarial_records,
        metric_summarizer=summarize_edit_metrics,
        random_rank_one=make_same_norm_random_rank_one,
    )


def run_cpu_release(
    *,
    implementation: LearnerFunctions,
    steps: int = 12,
    patching_samples: int = 3,
) -> dict[str, Any]:
    """Run the strict, CPU-only evidence hook used by a future parent registrar."""
    result = run_course_rome_like_experiment(
        implementation=implementation,
        device="cpu",
        steps=steps,
        patching_samples=patching_samples,
    )
    summary = serializable_experiment_summary(result)
    primary = summary["metrics"]["course_rome_like"]
    rank_one = summary["rank_one_update"]
    localization = summary["localization_result"]
    checks = {
        "cpu_execution": result["primary_update"].device.type == "cpu",
        "course_scope_declared": summary["method_scope"] == "course_rome_like_rank_one_edit",
        "official_parity_not_claimed": summary["official_rome_parity_established"] is False,
        "three_official_parity_gaps_named": len(summary["missing_official_parity_components"]) == 3,
        "nine_independent_noise_directions": (
            summary["trace"]["condition_count"] == 9
            and len(set(summary["trace"]["direction_seeds"])) == 9
        ),
        "rank_one_write": rank_one["effective_rank"] == 1,
        "affine_write_error_below_1e_4": rank_one["write_error"] < 1e-4,
        "base_known_fact_denominator_nontrivial": primary["known_fact_qualified_count"] >= 10,
        "adversarial_leakage_stratified": (
            primary["leakage_sensitive_count"]
            < summary["suite_sizes"]["adversarial"]
        ),
        "all_layer_comparisons_norm_matched": all(
            row["norm_matched"]
            and abs(row["applied_update_norm"] - rank_one["update_norm"]) < 1e-4
            for row in summary["layer_sweep"]
        ),
        "negative_localization_result_reported": localization["negative_localization_result"],
        "privileged_layer_claim_rejected": localization["supports_privileged_layer_claim"] is False,
        "large_value_optimizer_kl_reported": rank_one["optimization_preservation_kl_is_large"],
        "localized_repetition_failure_reported": (
            primary["generation_repetition_failure_count"] >= 1
            and primary["localized_louvre_repetition_after"]
            > primary["localized_louvre_repetition_before"]
        ),
    }
    return {
        "cuda_available": False,
        "device": "cpu",
        "full_path": True,
        "summary": summary,
        "checks": checks,
        "accepted": all(checks.values()),
    }


def run_smoke_test(cpu: bool = True) -> dict[str, Any]:
    del cpu
    toy = exact_rank_one_toy_oracle()
    return {
        "delta_rank": toy["delta_rank"],
        "write_error": float((toy["edit_after"] - toy["desired_value"]).abs().max().item()),
        "protected_error": float(
            (toy["protected_after"] - toy["protected_before"]).abs().max().item()
        ),
    }
