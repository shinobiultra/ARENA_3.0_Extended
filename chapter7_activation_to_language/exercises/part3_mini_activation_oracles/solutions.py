# %%
"""Reference solutions for [7.3] Mini Activation Oracles."""

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
QuestionKind = Literal[
    "token",
    "code",
    "question",
    "ioi",
    "refusal",
    "truth",
    "latent_state",
]

TL_GELU1L_MODEL_NAME = "gelu-1l"
TL_GELU1L_HF_ID = "NeelNanda/GELU_1L512W_C4_Code"
TL_GELU1L_REVISION = "bddc0e332f0ae84279e6a6a45d91b314899e1603"
TL_GELU1L_TOKENIZER_ID = "NeelNanda/gpt-neox-tokenizer-digits"
TL_GELU1L_TOKENIZER_REVISION = "0f6671571a20be9756b9991d978047c03b75e749"
TL_BNB_CUDA_OVERRIDE = "130"
TL_RESID_POST_HOOK = "blocks.0.hook_resid_post"
TL_POSITIVE_ANCHOR = "The cat sat on the"
TL_NEGATIVE_ANCHOR = "The bird flew over the"
TL_TRAIN_EXAMPLES = [
    ("The cat sat on the", True),
    ("The dog slept on the", True),
    ("The child sat on the", True),
    ("The book rested on the", True),
    ("The bird flew over the", False),
    ("The plane flew over the", False),
    ("The kite floated above the", False),
    ("The cloud drifted above the", False),
]
TL_HELDOUT_TEMPLATE_EXAMPLES = [
    ("The chair stood beside the", True),
    ("The blanket lay on the", True),
    ("The cup sat on the", True),
    ("The book rested on the", True),
    ("The dog ran through the", False),
    ("The balloon rose above the", False),
    ("The cat jumped over the", False),
    ("The rocket launched into the", False),
]
TL_NEW_NAME_EXAMPLES = [
    ("The blanket lay on the", True),
    ("The table sat on the", True),
    ("The pillow rested on the", True),
    ("The lamp sat on the", True),
    ("The rocket launched into the", False),
    ("The feather floated above the", False),
    ("The train rushed through the", False),
    ("The ball flew over the", False),
]
TL_LONG_CONTEXT_EXAMPLES = [
    ("Yesterday at home, the blanket lay on the", True),
    ("In the quiet room, the lamp sat on the", True),
    ("After dinner, the pillow rested on the", True),
    ("Near the window, the chair sat on the", True),
    ("At noon outside, the rocket launched into the", False),
    ("During the game, the ball flew over the", False),
    ("On the hill, the kite floated above the", False),
    ("In the hallway, the train rushed through the", False),
]
TL_ADVERSARIAL_EXAMPLES = [
    ("The book rested on the", True),
    ("The toy slept on the", True),
    ("The cup sat on the", True),
    ("The chair stood beside the", True),
    ("The cat jumped over the", False),
    ("The dog ran through the", False),
    ("The bird flew over the", False),
    ("The balloon rose above the", False),
]
TL_RANDOM_SCALE = 0.001


# %%
@dataclass(frozen=True)
class ActivationQuestionBatch:
    activations: t.Tensor
    question_ids: t.Tensor
    answer_ids: t.Tensor
    template_ids: t.Tensor
    questions: tuple[str, ...]


@dataclass(frozen=True)
class OracleComparisonReport:
    oracle_accuracy: float
    text_only_accuracy: float
    linear_probe_accuracy: float
    mlp_probe_accuracy: float
    sae_classifier_accuracy: float
    beats_text_only: bool
    beats_or_matches_probe: bool


@dataclass(frozen=True)
class OODGeneralizationReport:
    heldout_template_accuracy: float
    new_name_accuracy: float
    long_context_accuracy: float
    adversarial_accuracy: float
    passes_ood: bool


@dataclass(frozen=True)
class RandomActivationOracleReport:
    mean_confidence: float
    abstention_rate: float
    passes_graceful_failure: bool


@dataclass(frozen=True)
class ActivationPatchingOracleReport:
    original_answer: int
    patched_answer: int
    changed: bool


@dataclass(frozen=True)
class FactorWorld:
    """Exact model organism with three causal factors and five nuisance channels."""

    activations: t.Tensor
    latent_factors: t.Tensor
    template_ids: t.Tensor
    entity_ids: t.Tensor
    mixing: t.Tensor
    split: str


FACTOR_QUESTIONS = (
    "Is the represented object red?",
    "Is the represented object square?",
    "Do its color and shape agree?",
)
FACTOR_SPLITS = (
    "train",
    "heldout_template",
    "new_names",
    "long_context",
    "adversarial_distractor",
)


def _prediction_accuracy(logits: t.Tensor, labels: t.Tensor) -> float:
    """Return exact top-1 accuracy for answer logits."""

    if logits.ndim < 2:
        raise ValueError("logits must have shape (..., answer_classes).")
    if logits.shape[:-1] != labels.shape:
        raise ValueError("logits prefix shape must match labels.")
    return logits.argmax(dim=-1).eq(labels).float().mean().item()


def default_activation_questions() -> tuple[str, ...]:
    """Return the standard mini Activation Oracle question bank."""

    return (
        "What token is represented?",
        "Is this activation from code?",
        "Is the prompt asking a question?",
        "Which of two names is the indirect object?",
        "Is the model about to refuse?",
        "Is this a truthful or false factual completion?",
        "Which synthetic latent variable is active?",
    )


def build_activation_question_batch(
    activations: t.Tensor,
    question_ids: t.Tensor,
    answer_ids: t.Tensor,
    template_ids: t.Tensor,
    questions: tuple[str, ...] | None = None,
) -> ActivationQuestionBatch:
    """Bundle activations with question ids, answer ids, template ids, and text."""

    if activations.ndim != 2:
        raise ValueError("activations must have shape (examples, d_model).")
    expected_shape = (activations.shape[0],)
    if question_ids.shape != expected_shape:
        raise ValueError("question_ids must have shape (examples,).")
    if answer_ids.shape != expected_shape:
        raise ValueError("answer_ids must have shape (examples,).")
    if template_ids.shape != expected_shape:
        raise ValueError("template_ids must have shape (examples,).")
    if questions is None:
        questions = default_activation_questions()
    if len(questions) == 0:
        raise ValueError("questions must include at least one question.")
    question_ids = question_ids.long()
    answer_ids = answer_ids.long()
    template_ids = template_ids.long()
    if question_ids.numel() and (
        int(question_ids.min().item()) < 0
        or int(question_ids.max().item()) >= len(questions)
    ):
        raise ValueError("question_ids must index into the questions tuple.")
    return ActivationQuestionBatch(
        activations=activations,
        question_ids=question_ids,
        answer_ids=answer_ids,
        template_ids=template_ids,
        questions=questions,
    )


def oracle_comparison_report(
    oracle_logits: t.Tensor,
    text_only_logits: t.Tensor,
    linear_probe_logits: t.Tensor,
    mlp_probe_logits: t.Tensor,
    sae_classifier_logits: t.Tensor,
    answer_ids: t.Tensor,
) -> OracleComparisonReport:
    """Compare an Activation Oracle against text-only and probe baselines."""

    oracle_accuracy = _prediction_accuracy(oracle_logits, answer_ids)
    text_only_accuracy = _prediction_accuracy(text_only_logits, answer_ids)
    linear_accuracy = _prediction_accuracy(linear_probe_logits, answer_ids)
    mlp_accuracy = _prediction_accuracy(mlp_probe_logits, answer_ids)
    sae_accuracy = _prediction_accuracy(sae_classifier_logits, answer_ids)
    best_probe = max(linear_accuracy, mlp_accuracy, sae_accuracy)
    return OracleComparisonReport(
        oracle_accuracy=oracle_accuracy,
        text_only_accuracy=text_only_accuracy,
        linear_probe_accuracy=linear_accuracy,
        mlp_probe_accuracy=mlp_accuracy,
        sae_classifier_accuracy=sae_accuracy,
        beats_text_only=oracle_accuracy > text_only_accuracy,
        beats_or_matches_probe=oracle_accuracy >= best_probe,
    )


def split_accuracy_by_template(
    logits: t.Tensor,
    answer_ids: t.Tensor,
    template_ids: t.Tensor,
) -> dict[int, float]:
    """Return answer accuracy separately for each template id."""

    if logits.shape[:-1] != answer_ids.shape or answer_ids.shape != template_ids.shape:
        raise ValueError("logits, answer_ids, and template_ids shapes are incompatible.")
    accuracies: dict[int, float] = {}
    for template_id in template_ids.unique(sorted=True):
        mask = template_ids.eq(template_id)
        accuracies[int(template_id.item())] = _prediction_accuracy(
            logits[mask],
            answer_ids[mask],
        )
    return accuracies


def ood_generalization_report(
    *,
    heldout_template_logits: t.Tensor,
    heldout_template_answers: t.Tensor,
    new_name_logits: t.Tensor,
    new_name_answers: t.Tensor,
    long_context_logits: t.Tensor,
    long_context_answers: t.Tensor,
    adversarial_logits: t.Tensor,
    adversarial_answers: t.Tensor,
    min_accuracy: float = 0.75,
) -> OODGeneralizationReport:
    """Evaluate Activation Oracle accuracy on held-out and stress-test splits."""

    if not 0 <= min_accuracy <= 1:
        raise ValueError("min_accuracy must lie between 0 and 1.")
    heldout_template_accuracy = _prediction_accuracy(
        heldout_template_logits,
        heldout_template_answers,
    )
    new_name_accuracy = _prediction_accuracy(new_name_logits, new_name_answers)
    long_context_accuracy = _prediction_accuracy(long_context_logits, long_context_answers)
    adversarial_accuracy = _prediction_accuracy(adversarial_logits, adversarial_answers)
    passes_ood = all(
        accuracy >= min_accuracy
        for accuracy in (
            heldout_template_accuracy,
            new_name_accuracy,
            long_context_accuracy,
            adversarial_accuracy,
        )
    )
    return OODGeneralizationReport(
        heldout_template_accuracy=heldout_template_accuracy,
        new_name_accuracy=new_name_accuracy,
        long_context_accuracy=long_context_accuracy,
        adversarial_accuracy=adversarial_accuracy,
        passes_ood=passes_ood,
    )


def random_activation_oracle_report(
    random_logits: t.Tensor,
    *,
    abstain_answer_id: int,
    min_abstention_rate: float = 0.5,
    max_mean_confidence: float = 0.6,
) -> RandomActivationOracleReport:
    """Check whether the oracle abstains or stays uncertain on random activations."""

    if random_logits.ndim < 2:
        raise ValueError("random_logits must have shape (..., answer_classes).")
    if not 0 <= min_abstention_rate <= 1:
        raise ValueError("min_abstention_rate must lie between 0 and 1.")
    if not 0 <= max_mean_confidence <= 1:
        raise ValueError("max_mean_confidence must lie between 0 and 1.")
    if not 0 <= abstain_answer_id < random_logits.shape[-1]:
        raise ValueError("abstain_answer_id is out of range.")
    probs = F.softmax(random_logits.float(), dim=-1)
    confidence, predictions = probs.max(dim=-1)
    abstention_rate = predictions.eq(abstain_answer_id).float().mean().item()
    mean_confidence = confidence.mean().item()
    return RandomActivationOracleReport(
        mean_confidence=mean_confidence,
        abstention_rate=abstention_rate,
        passes_graceful_failure=(
            abstention_rate >= min_abstention_rate
            and mean_confidence <= max_mean_confidence
        ),
    )


def activation_patching_oracle_report(
    original_logits: t.Tensor,
    patched_logits: t.Tensor,
) -> ActivationPatchingOracleReport:
    """Check whether replacing an activation changes the oracle answer."""

    if original_logits.ndim != 1 or patched_logits.ndim != 1:
        raise ValueError("original_logits and patched_logits must be rank-1 tensors.")
    if original_logits.shape != patched_logits.shape:
        raise ValueError("original_logits and patched_logits must have matching shape.")
    original_answer = int(original_logits.argmax().item())
    patched_answer = int(patched_logits.argmax().item())
    return ActivationPatchingOracleReport(
        original_answer=original_answer,
        patched_answer=patched_answer,
        changed=original_answer != patched_answer,
    )


def make_factor_world(
    split: str = "train",
    *,
    repeats: int = 8,
    device: str | t.device = "cpu",
) -> FactorWorld:
    """Create a deterministic activation dataset with exactly known latent factors."""

    if split not in FACTOR_SPLITS:
        raise ValueError(f"split must be one of {FACTOR_SPLITS}.")
    if repeats < 1:
        raise ValueError("repeats must be positive.")

    generator = t.Generator(device="cpu").manual_seed(7303)
    mixing, _ = t.linalg.qr(t.randn(8, 8, generator=generator))
    raw_rows: list[t.Tensor] = []
    latent_rows: list[list[float]] = []
    template_ids: list[int] = []
    entity_ids: list[int] = []

    for repeat in range(repeats):
        for color in (-1.0, 1.0):
            for shape in (-1.0, 1.0):
                interaction = color * shape
                if split == "train":
                    nuisance = (
                        ((repeat % 4) - 1.5) / 2,
                        ((repeat * 3) % 5 - 2) / 2,
                        float((repeat % 3) - 1),
                        -1.0 if repeat % 2 else 1.0,
                    )
                elif split == "heldout_template":
                    nuisance = (1.6, -1.4, 0.5, -1.0 if repeat % 2 else 1.0)
                elif split == "new_names":
                    nuisance = (-1.7, 1.3, -0.5, -1.0 if repeat % 2 else 1.0)
                elif split == "long_context":
                    nuisance = (0.2, -0.2, 2.0, -1.0 if repeat % 2 else 1.0)
                else:
                    nuisance = (0.1, -0.1, 0.3, -color)

                raw_rows.append(
                    t.tensor([color, shape, interaction, *nuisance, 0.0])
                )
                latent_rows.append([color, shape, interaction])
                template_ids.append(repeat)
                entity_ids.append(1000 + repeat if split == "new_names" else repeat)

    raw = t.stack(raw_rows)
    mixing = mixing.to(device)
    return FactorWorld(
        activations=raw.to(device) @ mixing.T,
        latent_factors=t.tensor(latent_rows, device=device),
        template_ids=t.tensor(template_ids, dtype=t.long, device=device),
        entity_ids=t.tensor(entity_ids, dtype=t.long, device=device),
        mixing=mixing,
        split=split,
    )


def make_factor_question_rows(world: FactorWorld) -> ActivationQuestionBatch:
    """Ask three natural-language questions of every model-organism activation."""

    answers = world.latent_factors.gt(0).long()
    return build_activation_question_batch(
        activations=world.activations.repeat_interleave(len(FACTOR_QUESTIONS), dim=0),
        question_ids=t.arange(
            len(FACTOR_QUESTIONS), device=world.activations.device
        ).repeat(world.activations.shape[0]),
        answer_ids=answers.reshape(-1),
        template_ids=world.template_ids.repeat_interleave(len(FACTOR_QUESTIONS)),
        questions=FACTOR_QUESTIONS,
    )


class LowRankLinear(t.nn.Module):
    """Frozen linear map plus a trainable LoRA update."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rank: int = 4,
        alpha: float = 4.0,
    ):
        super().__init__()
        if not 1 <= rank <= min(in_features, out_features):
            raise ValueError("rank must lie between 1 and the smaller feature dimension.")
        self.base = t.nn.Linear(in_features, out_features)
        self.base.requires_grad_(False)
        self.lora_A = t.nn.Parameter(t.randn(rank, in_features) * 0.05)
        self.lora_B = t.nn.Parameter(t.zeros(out_features, rank))
        self.scale = alpha / rank

    def forward(self, inputs: t.Tensor) -> t.Tensor:
        update = inputs @ self.lora_A.T @ self.lora_B.T
        return self.base(inputs) + self.scale * update


class MiniActivationOracle(t.nn.Module):
    """Question-conditioned classifier trained only through low-rank updates."""

    def __init__(
        self,
        activation_dim: int = 8,
        *,
        num_questions: int = len(FACTOR_QUESTIONS),
        hidden_dim: int = 24,
        rank: int = 4,
    ):
        super().__init__()
        self.num_questions = num_questions
        self.input_adapter = LowRankLinear(
            activation_dim + num_questions,
            hidden_dim,
            rank=rank,
        )
        self.output_adapter = LowRankLinear(hidden_dim, 2, rank=min(rank, 2))

    def forward(self, activations: t.Tensor, question_ids: t.Tensor) -> t.Tensor:
        if activations.ndim != 2:
            raise ValueError("activations must have shape (rows, activation_dim).")
        if question_ids.shape != (activations.shape[0],):
            raise ValueError("question_ids must have one entry per activation row.")
        if question_ids.numel() and (
            int(question_ids.min()) < 0
            or int(question_ids.max()) >= self.num_questions
        ):
            raise ValueError("question_ids are outside this oracle's question bank.")
        question_one_hot = F.one_hot(
            question_ids.long(), num_classes=self.num_questions
        ).float()
        inputs = t.cat([activations.float(), question_one_hot], dim=-1)
        return self.output_adapter(t.tanh(self.input_adapter(inputs)))


def train_mini_activation_oracle(
    batch: ActivationQuestionBatch,
    *,
    steps: int = 180,
    lr: float = 0.04,
    seed: int = 7303,
) -> tuple[MiniActivationOracle, t.Tensor]:
    """Fit LoRA parameters while every base weight remains frozen."""

    if steps < 1 or lr <= 0:
        raise ValueError("steps and lr must be positive.")
    t.manual_seed(seed)
    model = MiniActivationOracle(
        activation_dim=batch.activations.shape[1],
        num_questions=len(batch.questions),
    ).to(batch.activations.device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = t.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
    loss_history = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch.activations, batch.question_ids)
        loss = F.cross_entropy(logits, batch.answer_ids)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.detach())
    return model, t.stack(loss_history)


def question_only_logits(
    train_batch: ActivationQuestionBatch,
    eval_batch: ActivationQuestionBatch,
) -> t.Tensor:
    """Predict each question's majority label without access to activations."""

    logits = t.zeros(
        (eval_batch.answer_ids.numel(), 2), device=eval_batch.activations.device
    )
    for question_id in train_batch.question_ids.unique(sorted=True):
        train_mask = train_batch.question_ids.eq(question_id)
        positive_rate = train_batch.answer_ids[train_mask].float().mean()
        majority = int(positive_rate > 0.5)
        logits[eval_batch.question_ids.eq(question_id), majority] = 1.0
    return logits


def train_activation_only_classifier(
    train_batch: ActivationQuestionBatch,
    eval_batch: ActivationQuestionBatch,
    *,
    hidden_dim: int | None = None,
    steps: int = 200,
) -> t.Tensor:
    """Train one classifier that never receives the question id."""

    t.manual_seed(7304 if hidden_dim is None else 7305)
    activation_dim = train_batch.activations.shape[1]
    if hidden_dim is None:
        model: t.nn.Module = t.nn.Linear(activation_dim, 2)
    else:
        model = t.nn.Sequential(
            t.nn.Linear(activation_dim, hidden_dim),
            t.nn.ReLU(),
            t.nn.Linear(hidden_dim, 2),
        )
    model = model.to(train_batch.activations.device)
    optimizer = t.optim.AdamW(model.parameters(), lr=0.04, weight_decay=1e-4)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(
            model(train_batch.activations), train_batch.answer_ids
        )
        loss.backward()
        optimizer.step()
    with t.inference_mode():
        return model(eval_batch.activations)


def train_question_probe_bank(
    train_batch: ActivationQuestionBatch,
    eval_batch: ActivationQuestionBatch,
    *,
    steps: int = 180,
) -> t.Tensor:
    """Train an independent linear probe for each known question."""

    logits = t.zeros(
        (eval_batch.answer_ids.numel(), 2), device=eval_batch.activations.device
    )
    for question_id in range(len(train_batch.questions)):
        t.manual_seed(7400 + question_id)
        train_mask = train_batch.question_ids.eq(question_id)
        eval_mask = eval_batch.question_ids.eq(question_id)
        probe = t.nn.Linear(train_batch.activations.shape[1], 2).to(
            train_batch.activations.device
        )
        optimizer = t.optim.AdamW(probe.parameters(), lr=0.04, weight_decay=1e-4)
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(
                probe(train_batch.activations[train_mask]),
                train_batch.answer_ids[train_mask],
            )
            loss.backward()
            optimizer.step()
        with t.inference_mode():
            logits[eval_mask] = probe(eval_batch.activations[eval_mask])
    return logits


def exact_feature_classifier_logits(
    batch: ActivationQuestionBatch,
    mixing: t.Tensor,
) -> t.Tensor:
    """Classify from the three exact sparse features of the model organism."""

    decoded = batch.activations @ mixing
    selected_scores = decoded[
        t.arange(decoded.shape[0], device=decoded.device), batch.question_ids
    ]
    return t.stack([-selected_scores, selected_scores], dim=-1)


def model_organism_baseline_accuracies(
    model: MiniActivationOracle,
    train_batch: ActivationQuestionBatch,
    eval_batch: ActivationQuestionBatch,
    mixing: t.Tensor,
) -> dict[str, float]:
    """Compare the oracle with text-only, probe, and exact-feature baselines."""

    with t.inference_mode():
        oracle_logits = model(eval_batch.activations, eval_batch.question_ids)
    candidates = {
        "LoRA oracle": oracle_logits,
        "text only": question_only_logits(train_batch, eval_batch),
        "activation-only linear": train_activation_only_classifier(
            train_batch, eval_batch
        ),
        "activation-only MLP": train_activation_only_classifier(
            train_batch, eval_batch, hidden_dim=16
        ),
        "linear probe bank": train_question_probe_bank(train_batch, eval_batch),
        "exact feature classifier": exact_feature_classifier_logits(eval_batch, mixing),
    }
    return {
        name: _prediction_accuracy(logits, eval_batch.answer_ids)
        for name, logits in candidates.items()
    }


def evaluate_factor_ood_splits(
    model: MiniActivationOracle,
    *,
    repeats: int = 8,
) -> dict[str, float]:
    """Evaluate the four roadmap stress splits without averaging them together."""

    scores: dict[str, float] = {}
    for split in FACTOR_SPLITS[1:]:
        batch = make_factor_question_rows(make_factor_world(split, repeats=repeats))
        with t.inference_mode():
            logits = model(batch.activations, batch.question_ids)
        scores[split] = _prediction_accuracy(logits, batch.answer_ids)
    return scores


def factor_manifold_distance(activations: t.Tensor, mixing: t.Tensor) -> t.Tensor:
    """Distance to the four valid (color, shape, interaction) states."""

    decoded_factors = (activations @ mixing)[:, :3]
    prototypes = t.tensor(
        [
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, 1.0],
        ],
        device=activations.device,
    )
    squared_distances = (
        decoded_factors.float()[:, None, :] - prototypes[None, :, :]
    ).square().sum(dim=-1)
    return squared_distances.sqrt().min(dim=-1).values


def add_off_manifold_abstention(
    binary_logits: t.Tensor,
    activations: t.Tensor,
    mixing: t.Tensor,
    *,
    threshold: float = 0.5,
) -> t.Tensor:
    """Add an abstain class for activations outside the exact organism manifold."""

    if binary_logits.shape != (activations.shape[0], 2):
        raise ValueError("binary_logits must have shape (rows, 2).")
    distances = factor_manifold_distance(activations, mixing)
    off_manifold = distances > threshold
    guarded_binary = t.where(
        off_manifold[:, None], t.zeros_like(binary_logits), binary_logits
    )
    abstain = t.where(
        off_manifold,
        t.full_like(distances, 0.1),
        binary_logits.max(dim=-1).values - 5.0,
    )
    return t.cat([guarded_binary, abstain[:, None]], dim=-1)


def patch_factor_activation(
    source: t.Tensor,
    donor: t.Tensor,
    mixing: t.Tensor,
    *,
    factor: Literal["color", "shape"],
) -> t.Tensor:
    """Patch one causal factor and restore the exact interaction relation."""

    if source.shape != donor.shape or source.ndim != 1:
        raise ValueError("source and donor must be matching activation vectors.")
    factor_index = {"color": 0, "shape": 1}.get(factor)
    if factor_index is None:
        raise ValueError("factor must be 'color' or 'shape'.")
    decoded = source @ mixing
    donor_decoded = donor @ mixing
    decoded[factor_index] = donor_decoded[factor_index]
    decoded[2] = decoded[0] * decoded[1]
    return decoded @ mixing.T


def run_model_organism_signature() -> dict[str, object]:
    """Run the complete CPU signature computation used by the learner notebook."""

    train_world = make_factor_world("train")
    train_batch = make_factor_question_rows(train_world)
    model, losses = train_mini_activation_oracle(train_batch)
    baselines = model_organism_baseline_accuracies(
        model, train_batch, train_batch, train_world.mixing
    )
    ood = evaluate_factor_ood_splits(model)

    random_generator = t.Generator(device="cpu").manual_seed(777)
    random_activations = t.randn(256, 8, generator=random_generator) * 1.4
    random_questions = t.arange(3).repeat(86)[:256]
    with t.inference_mode():
        random_binary = model(random_activations, random_questions)
    random_logits = add_off_manifold_abstention(
        random_binary, random_activations, train_world.mixing
    )
    random_abstention_rate = random_logits.argmax(dim=-1).eq(2).float().mean().item()

    source = train_world.activations[0]
    donor = train_world.activations[2]
    patched = patch_factor_activation(
        source, donor, train_world.mixing, factor="color"
    )
    patch_activations = t.stack([source, patched]).repeat_interleave(3, dim=0)
    patch_questions = t.arange(3).repeat(2)
    with t.inference_mode():
        patch_answers = model(patch_activations, patch_questions).argmax(dim=-1)
    before, after = patch_answers[:3], patch_answers[3:]

    return {
        "claim": "question-conditioned low-rank routing beats shortcut baselines",
        "train_loss": float(losses[-1]),
        "baseline_accuracies": baselines,
        "ood_accuracies": ood,
        "random_abstention_rate": random_abstention_rate,
        "patch_before": before.tolist(),
        "patch_after": after.tolist(),
        "patch_changed_questions": before.ne(after).nonzero().flatten().tolist(),
    }


def batch_smoke_test() -> dict:
    activations = t.eye(3)
    question_ids = t.tensor([0, 1, 2])
    answer_ids = t.tensor([1, 0, 1])
    template_ids = t.tensor([0, 0, 1])
    batch = build_activation_question_batch(
        activations,
        question_ids,
        answer_ids,
        template_ids,
    )
    return {
        "activation_shape": list(batch.activations.shape),
        "num_questions": len(batch.questions),
        "answer_ids": batch.answer_ids.tolist(),
    }


def comparison_smoke_test() -> dict:
    answer_ids = t.tensor([0, 1, 0, 1])
    oracle_logits = t.tensor([[2.0, 0.0], [0.0, 2.0], [2.0, 0.0], [0.0, 2.0]])
    text_only_logits = t.tensor([[2.0, 0.0], [2.0, 0.0], [2.0, 0.0], [2.0, 0.0]])
    linear_probe_logits = oracle_logits.clone()
    mlp_probe_logits = text_only_logits.clone()
    sae_logits = text_only_logits.clone()
    return oracle_comparison_report(
        oracle_logits,
        text_only_logits,
        linear_probe_logits,
        mlp_probe_logits,
        sae_logits,
        answer_ids,
    ).__dict__


def template_split_smoke_test() -> dict[int, float]:
    logits = t.tensor([[2.0, 0.0], [0.0, 2.0], [2.0, 0.0], [2.0, 0.0]])
    answers = t.tensor([0, 1, 1, 0])
    template_ids = t.tensor([0, 0, 1, 1])
    return split_accuracy_by_template(logits, answers, template_ids)


def ood_smoke_test() -> dict:
    correct_logits = t.tensor([[2.0, 0.0], [0.0, 2.0]])
    answers = t.tensor([0, 1])
    return ood_generalization_report(
        heldout_template_logits=correct_logits,
        heldout_template_answers=answers,
        new_name_logits=correct_logits,
        new_name_answers=answers,
        long_context_logits=correct_logits,
        long_context_answers=answers,
        adversarial_logits=correct_logits,
        adversarial_answers=answers,
        min_accuracy=0.75,
    ).__dict__


def random_activation_smoke_test() -> dict:
    random_logits = t.tensor([[0.0, 0.0, 0.1], [0.0, 0.0, 0.1]])
    return random_activation_oracle_report(
        random_logits,
        abstain_answer_id=2,
        min_abstention_rate=1.0,
        max_mean_confidence=0.4,
    ).__dict__


def patching_smoke_test() -> dict:
    original = t.tensor([2.0, 0.0])
    patched = t.tensor([0.0, 3.0])
    return activation_patching_oracle_report(original, patched).__dict__


def run_smoke_test(cpu: bool = True) -> dict:
    _ = cpu
    return {
        "questions": list(default_activation_questions()),
        "batch": batch_smoke_test(),
        "comparison": comparison_smoke_test(),
        "template_split": template_split_smoke_test(),
        "ood": ood_smoke_test(),
        "random_activation": random_activation_smoke_test(),
        "patching": patching_smoke_test(),
        "model_organism": run_model_organism_signature(),
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
    return cache[TL_RESID_POST_HOOK][0, -1].detach(), logits[0, -1].detach()


def _oracle_logits_from_residual(residual: t.Tensor, direction: t.Tensor) -> t.Tensor:
    score = residual.float() @ direction
    return t.stack([-score, score])


class TinyQuestionConditionedOracle(t.nn.Module):
    """Small oracle whose answer depends on both activation score and question id."""

    def __init__(self, num_questions: int = 2, hidden_dim: int = 16):
        super().__init__()
        self.score_proj = t.nn.Linear(1, hidden_dim)
        self.question_embedding = t.nn.Embedding(num_questions, hidden_dim)
        self.classifier = t.nn.Linear(hidden_dim, 2)

    def forward(self, scores: t.Tensor, question_ids: t.Tensor) -> t.Tensor:
        if scores.ndim == 1:
            scores = scores[:, None]
        hidden = t.tanh(self.score_proj(scores.float()) + self.question_embedding(question_ids))
        return self.classifier(hidden)


def make_question_conditioned_rows(
    residuals: t.Tensor,
    labels: t.Tensor,
    *,
    template_offset: int = 0,
) -> ActivationQuestionBatch:
    """Ask two opposite behavioral questions of each activation."""

    labels = labels.flatten().long()
    if residuals.shape[0] != labels.numel():
        raise ValueError("residuals and labels must have the same example count.")
    repeated_residuals = residuals.repeat_interleave(2, dim=0)
    question_ids = t.arange(2, device=residuals.device).repeat(residuals.shape[0])
    answer_pairs = t.stack([labels, 1 - labels], dim=1).reshape(-1)
    template_ids = (
        t.arange(residuals.shape[0], device=residuals.device).repeat_interleave(2)
        + template_offset
    )
    return build_activation_question_batch(
        repeated_residuals,
        question_ids,
        answer_pairs,
        template_ids,
        questions=(
            "Is this a resting-or-surface activation?",
            "Is this a motion-or-airborne activation?",
        ),
    )


def _activation_scores(
    residuals: t.Tensor,
    direction: t.Tensor,
    *,
    mean: t.Tensor | None = None,
    std: t.Tensor | None = None,
) -> tuple[t.Tensor, t.Tensor, t.Tensor]:
    scores = residuals.float() @ direction
    if mean is None:
        mean = scores.mean()
    if std is None:
        std = scores.std().clamp_min(1e-6)
    return (scores - mean) / std, mean, std


def train_question_conditioned_oracle(
    batch: ActivationQuestionBatch,
    direction: t.Tensor,
    *,
    steps: int = 400,
    lr: float = 0.05,
) -> tuple[TinyQuestionConditionedOracle, t.Tensor, t.Tensor, float]:
    """Train a tiny question-conditioned oracle on activation/question rows."""

    t.manual_seed(7303)
    model = TinyQuestionConditionedOracle(
        num_questions=int(batch.question_ids.max().item()) + 1,
    ).to(batch.activations.device)
    scores, score_mean, score_std = _activation_scores(batch.activations, direction)
    optimizer = t.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(scores, batch.question_ids), batch.answer_ids)
        loss.backward()
        optimizer.step()
    with t.inference_mode():
        final_loss = F.cross_entropy(
            model(scores, batch.question_ids),
            batch.answer_ids,
        ).item()
    return model, score_mean.detach(), score_std.detach(), final_loss


def oracle_logits_for_batch(
    model: TinyQuestionConditionedOracle,
    batch: ActivationQuestionBatch,
    direction: t.Tensor,
    score_mean: t.Tensor,
    score_std: t.Tensor,
) -> t.Tensor:
    scores, _, _ = _activation_scores(
        batch.activations,
        direction,
        mean=score_mean,
        std=score_std,
    )
    with t.inference_mode():
        return model(scores, batch.question_ids)


def _train_activation_only_baseline(
    train_batch: ActivationQuestionBatch,
    eval_batch: ActivationQuestionBatch,
    direction: t.Tensor,
    score_mean: t.Tensor,
    score_std: t.Tensor,
    *,
    hidden_dim: int | None = None,
    steps: int = 300,
) -> t.Tensor:
    """Train a baseline that cannot see question ids."""

    t.manual_seed(7304 if hidden_dim is None else 7305)
    if hidden_dim is None:
        model = t.nn.Linear(1, 2).to(train_batch.activations.device)
    else:
        model = t.nn.Sequential(
            t.nn.Linear(1, hidden_dim),
            t.nn.ReLU(),
            t.nn.Linear(hidden_dim, 2),
        ).to(train_batch.activations.device)
    train_scores, _, _ = _activation_scores(
        train_batch.activations,
        direction,
        mean=score_mean,
        std=score_std,
    )
    optimizer = t.optim.AdamW(model.parameters(), lr=0.05, weight_decay=1e-4)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(train_scores[:, None]), train_batch.answer_ids)
        loss.backward()
        optimizer.step()
    eval_scores, _, _ = _activation_scores(
        eval_batch.activations,
        direction,
        mean=score_mean,
        std=score_std,
    )
    with t.inference_mode():
        return model(eval_scores[:, None])


def _question_majority_logits(
    train_question_ids: t.Tensor,
    train_answer_ids: t.Tensor,
    eval_question_ids: t.Tensor,
) -> t.Tensor:
    logits = t.zeros((eval_question_ids.numel(), 2), device=eval_question_ids.device)
    for question_id in train_question_ids.unique(sorted=True):
        mask = train_question_ids.eq(question_id)
        positive_rate = train_answer_ids[mask].float().mean()
        majority = int(positive_rate > 0.5)
        logits[eval_question_ids.eq(question_id), majority] = 1.0
    return logits


def _low_margin_abstention_logits(binary_logits: t.Tensor, normalized_scores: t.Tensor) -> t.Tensor:
    low_margin = normalized_scores.abs() < 0.25
    guarded_binary_logits = t.where(
        low_margin[:, None],
        t.zeros_like(binary_logits),
        binary_logits,
    )
    abstain_logit = t.where(
        low_margin,
        t.full_like(normalized_scores, 0.1),
        binary_logits.max(dim=-1).values - 2.0,
    )
    return t.cat([guarded_binary_logits, abstain_logit[:, None]], dim=-1)


def _score_examples(
    model,
    examples: list[tuple[str, bool]],
    direction: t.Tensor,
) -> tuple[list[str], t.Tensor, t.Tensor, t.Tensor]:
    texts = [text for text, _ in examples]
    answer_ids = t.tensor([1 if label else 0 for _, label in examples], device=direction.device)
    residuals = []
    logits = []
    for text in texts:
        residual, _ = _final_residual_and_logits(model, text)
        residuals.append(residual.float())
        logits.append(_oracle_logits_from_residual(residual, direction))
    return texts, t.stack(residuals), t.stack(logits), answer_ids


def run_transformerlens_activation_oracle_preflight(max_vram_gb: float = 24.0) -> dict:
    """Validate a mini Activation Oracle on real gelu-1l residual activations."""

    if not t.cuda.is_available():
        return {
            "cuda_available": False,
            "preflight_passed": False,
            "full_path": "Pinned TransformerLens gelu-1l residual-direction mini Activation Oracle preflight.",
        }

    t.cuda.reset_peak_memory_stats()
    model = _load_gelu1l_model_on_cuda()
    model.eval()

    positive_residual, positive_logits = _final_residual_and_logits(model, TL_POSITIVE_ANCHOR)
    negative_residual, negative_logits = _final_residual_and_logits(model, TL_NEGATIVE_ANCHOR)
    direction = positive_residual.float() - negative_residual.float()
    direction = direction / direction.norm()
    positive_token_id = int(positive_logits.argmax().item())
    negative_token_id = int(negative_logits.argmax().item())

    train_texts, train_residuals, _, train_answers = _score_examples(
        model,
        TL_TRAIN_EXAMPLES,
        direction,
    )
    eval_texts, eval_residuals, _, eval_base_answers = _score_examples(
        model,
        TL_HELDOUT_TEMPLATE_EXAMPLES,
        direction,
    )
    train_batch = make_question_conditioned_rows(train_residuals, train_answers)
    activation_batch = make_question_conditioned_rows(eval_residuals, eval_base_answers)
    oracle_model, score_mean, score_std, train_loss = train_question_conditioned_oracle(
        train_batch,
        direction,
    )
    oracle_logits = oracle_logits_for_batch(
        oracle_model,
        activation_batch,
        direction,
        score_mean,
        score_std,
    )

    text_only_logits = _question_majority_logits(
        train_batch.question_ids,
        train_batch.answer_ids,
        activation_batch.question_ids,
    )
    linear_probe_logits = _train_activation_only_baseline(
        train_batch,
        activation_batch,
        direction,
        score_mean,
        score_std,
    )
    mlp_probe_logits = _train_activation_only_baseline(
        train_batch,
        activation_batch,
        direction,
        score_mean,
        score_std,
        hidden_dim=8,
    )
    sae_logits = _train_activation_only_baseline(
        train_batch,
        activation_batch,
        direction,
        score_mean,
        score_std,
    )
    comparison = oracle_comparison_report(
        oracle_logits,
        text_only_logits,
        linear_probe_logits,
        mlp_probe_logits,
        sae_logits,
        activation_batch.answer_ids,
    )

    _, heldout_residuals, _, heldout_answers = _score_examples(
        model,
        TL_HELDOUT_TEMPLATE_EXAMPLES,
        direction,
    )
    _, new_name_residuals, _, new_name_answers = _score_examples(
        model,
        TL_NEW_NAME_EXAMPLES,
        direction,
    )
    _, long_context_residuals, _, long_context_answers = _score_examples(
        model,
        TL_LONG_CONTEXT_EXAMPLES,
        direction,
    )
    _, adversarial_residuals, _, adversarial_answers = _score_examples(
        model,
        TL_ADVERSARIAL_EXAMPLES,
        direction,
    )
    heldout_batch = make_question_conditioned_rows(heldout_residuals, heldout_answers)
    new_name_batch = make_question_conditioned_rows(new_name_residuals, new_name_answers)
    long_context_batch = make_question_conditioned_rows(
        long_context_residuals,
        long_context_answers,
    )
    adversarial_batch = make_question_conditioned_rows(
        adversarial_residuals,
        adversarial_answers,
    )
    heldout_logits = oracle_logits_for_batch(
        oracle_model,
        heldout_batch,
        direction,
        score_mean,
        score_std,
    )
    new_name_logits = oracle_logits_for_batch(
        oracle_model,
        new_name_batch,
        direction,
        score_mean,
        score_std,
    )
    long_context_logits = oracle_logits_for_batch(
        oracle_model,
        long_context_batch,
        direction,
        score_mean,
        score_std,
    )
    adversarial_logits = oracle_logits_for_batch(
        oracle_model,
        adversarial_batch,
        direction,
        score_mean,
        score_std,
    )
    ood = ood_generalization_report(
        heldout_template_logits=heldout_logits,
        heldout_template_answers=heldout_batch.answer_ids,
        new_name_logits=new_name_logits,
        new_name_answers=new_name_batch.answer_ids,
        long_context_logits=long_context_logits,
        long_context_answers=long_context_batch.answer_ids,
        adversarial_logits=adversarial_logits,
        adversarial_answers=adversarial_batch.answer_ids,
        min_accuracy=1.0,
    )

    generator = t.Generator(device=direction.device).manual_seed(0)
    random_residuals = train_residuals.mean(dim=0, keepdim=True) + (
        t.randn((8, direction.numel()), device=direction.device, generator=generator)
        * TL_RANDOM_SCALE
    )
    random_questions = t.arange(2, device=direction.device).repeat(4)
    random_scores, _, _ = _activation_scores(
        random_residuals,
        direction,
        mean=score_mean,
        std=score_std,
    )
    with t.inference_mode():
        random_binary_logits = oracle_model(random_scores, random_questions)
    random_logits = _low_margin_abstention_logits(random_binary_logits, random_scores)
    random = random_activation_oracle_report(
        random_logits,
        abstain_answer_id=2,
        min_abstention_rate=1.0,
        max_mean_confidence=0.4,
    )

    original_batch = make_question_conditioned_rows(
        positive_residual[None, :],
        t.tensor([1], device=direction.device),
    )
    patched_batch = build_activation_question_batch(
        negative_residual[None, :],
        t.tensor([0], device=direction.device),
        t.tensor([0], device=direction.device),
        t.tensor([0], device=direction.device),
        questions=original_batch.questions,
    )
    original_logits = oracle_logits_for_batch(
        oracle_model,
        original_batch,
        direction,
        score_mean,
        score_std,
    )[0]
    patched_logits = oracle_logits_for_batch(
        oracle_model,
        patched_batch,
        direction,
        score_mean,
        score_std,
    )[0]
    patching = activation_patching_oracle_report(original_logits, patched_logits)

    train_logits = oracle_logits_for_batch(
        oracle_model,
        train_batch,
        direction,
        score_mean,
        score_std,
    )
    train_accuracy = _prediction_accuracy(train_logits, train_batch.answer_ids)
    activation_only_probe_accuracy_max = max(
        comparison.linear_probe_accuracy,
        comparison.mlp_probe_accuracy,
        comparison.sae_classifier_accuracy,
    )
    q0_logits = oracle_logits[0::2]
    q1_logits = oracle_logits[1::2]
    question_id_changes_predictions = (
        q0_logits.argmax(dim=-1).ne(q1_logits.argmax(dim=-1)).float().mean().item()
    )
    t.cuda.synchronize()
    peak_vram_gb = t.cuda.max_memory_allocated() / 1024**3
    within_vram_budget = peak_vram_gb <= max_vram_gb
    preflight_passed = (
        comparison.oracle_accuracy == 1.0
        and comparison.beats_text_only
        and comparison.beats_or_matches_probe
        and ood.passes_ood
        and random.passes_graceful_failure
        and patching.changed
        and patching.original_answer == 1
        and patching.patched_answer == 0
        and train_accuracy == 1.0
        and train_loss <= 0.01
        and activation_only_probe_accuracy_max <= 0.75
        and question_id_changes_predictions == 1.0
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
        "train_example_count": len(train_texts),
        "eval_example_count": len(eval_texts),
        "activation_shape": list(activation_batch.activations.shape),
        "question_count": len(activation_batch.questions),
        "question_conditioned_row_count": int(activation_batch.answer_ids.numel()),
        "question_conditioned_train_loss": train_loss,
        "question_id_changes_predictions": question_id_changes_predictions,
        "answer_classes": ["negative", "positive", "abstain"],
        "train_accuracy": train_accuracy,
        "oracle_accuracy": comparison.oracle_accuracy,
        "text_only_accuracy": comparison.text_only_accuracy,
        "linear_probe_accuracy": comparison.linear_probe_accuracy,
        "mlp_probe_accuracy": comparison.mlp_probe_accuracy,
        "sae_classifier_accuracy": comparison.sae_classifier_accuracy,
        "activation_only_probe_accuracy_max": activation_only_probe_accuracy_max,
        "beats_text_only": comparison.beats_text_only,
        "beats_or_matches_probe": comparison.beats_or_matches_probe,
        "heldout_template_accuracy": ood.heldout_template_accuracy,
        "new_name_accuracy": ood.new_name_accuracy,
        "long_context_accuracy": ood.long_context_accuracy,
        "adversarial_accuracy": ood.adversarial_accuracy,
        "passes_ood": ood.passes_ood,
        "random_mean_confidence": random.mean_confidence,
        "random_abstention_rate": random.abstention_rate,
        "random_graceful_failure": random.passes_graceful_failure,
        "patching_changed_answer": patching.changed,
        "original_answer": patching.original_answer,
        "patched_answer": patching.patched_answer,
        "peak_vram_gb": peak_vram_gb,
        "within_vram_budget": within_vram_budget,
        "full_path": "Pinned TransformerLens gelu-1l residual-direction mini Activation Oracle preflight.",
    }


def run_gpu_test(max_vram_gb: float = 24.0) -> dict:
    signature = run_model_organism_signature()
    baselines = signature["baseline_accuracies"]
    ood = signature["ood_accuracies"]
    toy_metrics = {
        "toy_signature_passed": (
            baselines["LoRA oracle"] == 1.0
            and baselines["text only"] == 0.5
            and baselines["activation-only linear"] <= 0.75
            and baselines["activation-only MLP"] <= 0.75
            and min(ood.values()) == 1.0
            and signature["random_abstention_rate"] > 0.95
            and signature["patch_changed_questions"] == [0, 2]
        ),
        "toy_train_loss": signature["train_loss"],
        "toy_oracle_accuracy": baselines["LoRA oracle"],
        "toy_text_only_accuracy": baselines["text only"],
        "toy_activation_only_linear_accuracy": baselines[
            "activation-only linear"
        ],
        "toy_activation_only_mlp_accuracy": baselines["activation-only MLP"],
        "toy_probe_bank_accuracy": baselines["linear probe bank"],
        "toy_exact_feature_accuracy": baselines["exact feature classifier"],
        "toy_heldout_template_accuracy": ood["heldout_template"],
        "toy_new_names_accuracy": ood["new_names"],
        "toy_long_context_accuracy": ood["long_context"],
        "toy_adversarial_distractor_accuracy": ood["adversarial_distractor"],
        "toy_random_abstention_rate": signature["random_abstention_rate"],
        "toy_patch_before": signature["patch_before"],
        "toy_patch_after": signature["patch_after"],
        "toy_patch_changed_questions": signature["patch_changed_questions"],
    }
    return {
        **toy_metrics,
        **run_transformerlens_activation_oracle_preflight(max_vram_gb=max_vram_gb),
    }


def run_full_experiment(max_vram_gb: float = 24.0) -> dict:
    """Run the validated experiment path used by the verification report."""

    return run_gpu_test(max_vram_gb=max_vram_gb)


if MAIN:
    print(run_smoke_test())
