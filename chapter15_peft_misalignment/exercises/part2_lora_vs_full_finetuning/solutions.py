# %%


from __future__ import annotations

MAIN = __name__ == "__main__"

# %%

import time
import copy

import json

import math

import sys

from dataclasses import asdict, dataclass

from pathlib import Path

from typing import Any, Sequence

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import torch as t

import torch.nn.functional as F

from IPython.display import Image, display

from torch import nn

chapter = "chapter15_peft_misalignment"

section = "part2_lora_vs_full_finetuning"

root_dir = Path(__file__).resolve().parents[3]

section_dir = root_dir / chapter / "exercises" / section

asset_dir = root_dir / chapter / "instructions" / "assets"

if MAIN:
    if str(root_dir) not in sys.path:
        sys.path.append(str(root_dir))

if MAIN:
    from chapter15_peft_misalignment.exercises.part2_lora_vs_full_finetuning import tests

MODEL_ID = "roneneldan/TinyStories-1M"

MODEL_REVISION = "77f1b168e219585646439073245fe87e56b3023e"

TARGET_MODULES = ('transformer.h.6.attn.attention.q_proj', 'transformer.h.6.attn.attention.v_proj')

INTERVENTION_LAYER = 6

LABEL_TEXT = (' red', ' blue')

EXERCISE_ID = "15_2_lora_vs_full_finetuning"

GT_TIER = "GT-1"

DIFFICULTY = 4

IMPORTANCE = 4

EXPECTED_RUNTIME = "under two minutes for bounded CPU preflight; minutes for full CUDA comparison"

REQUIRES_GPU = True

RUN_BOUNDED_CPU = False

RUN_FULL_CUDA = False

@dataclass(frozen=True)
class PromptExample:
    prompt: str
    label: int
    subject: str
    template_id: str

@dataclass(frozen=True)
class TrainTrace:
    losses: list[float]
    steps: int
    examples_seen: int

@dataclass(frozen=True)
class BehaviorMetrics:
    accuracy: float
    target_probability: float
    predictions: list[int]
    target_probabilities: list[float]

@dataclass(frozen=True)
class SpectrumMetrics:
    singular_values: list[float]
    effective_rank: float
    stable_rank: float
    update_norm: float

@dataclass(frozen=True)
class DriftMetrics:
    direction: t.Tensor
    top_variance_fraction: float
    mean_drift_norm: float

# %%

def toy_lora_delta(a: t.Tensor, b: t.Tensor, *, alpha: float) -> t.Tensor:
    """Return ``alpha / rank * B @ A`` after checking the low-rank shapes."""

    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("A and B must both be matrices")
    rank = a.shape[0]
    if rank == 0 or b.shape[1] != rank:
        raise ValueError("B's second dimension must equal the positive LoRA rank")
    return (float(alpha) / rank) * (b @ a)

# %%

if MAIN:
    tests.test_toy_lora_delta_uses_scaled_b_times_a(toy_lora_delta)

if MAIN:
    print('Toy LoRA oracle passed.')

# %%

class LoRALinear(nn.Module):
    """A transparent LoRA wrapper around an actual ``nn.Linear`` layer."""

    def __init__(self, base: nn.Linear, *, rank: int, alpha: float) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        if not isinstance(base, nn.Linear):
            raise TypeError("LoRALinear can only wrap nn.Linear")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        factory_kwargs = {"device": base.weight.device, "dtype": base.weight.dtype}
        self.lora_a = nn.Parameter(t.empty(rank, base.in_features, **factory_kwargs))
        self.lora_b = nn.Parameter(t.zeros(base.out_features, rank, **factory_kwargs))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.merged = False
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def delta_weight(self) -> t.Tensor:
        return self.scaling * (self.lora_b @ self.lora_a)

    def forward(self, inputs: t.Tensor) -> t.Tensor:
        base_output = self.base(inputs)
        if self.merged:
            return base_output
        adapter_output = F.linear(F.linear(inputs, self.lora_a), self.lora_b)
        return base_output + self.scaling * adapter_output

    @t.no_grad()
    def merge_(self) -> "LoRALinear":
        if not self.merged:
            self.base.weight.add_(self.delta_weight().to(self.base.weight.dtype))
            self.merged = True
        return self

    @t.no_grad()
    def unmerge_(self) -> "LoRALinear":
        if self.merged:
            self.base.weight.sub_(self.delta_weight().to(self.base.weight.dtype))
            self.merged = False
        return self

# %%

if MAIN:
    tests.test_lora_linear_starts_as_exact_base_and_trains_only_factors(LoRALinear)

if MAIN:
    print('LoRA no-op initialization passed.')

# %%

def _get_parent_and_leaf(model: nn.Module, qualified_name: str) -> tuple[nn.Module, str]:
    parts = qualified_name.split(".")
    parent: nn.Module = model
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    return parent, parts[-1]

# %%

def inject_lora(
    model: nn.Module,
    target_modules: Sequence[str],
    *,
    rank: int,
    alpha: float,
) -> dict[str, LoRALinear]:
    """Replace named transformer linear layers with visible ``LoRALinear`` modules."""

    inserted: dict[str, LoRALinear] = {}
    for name in target_modules:
        parent, leaf = _get_parent_and_leaf(model, name)
        original = getattr(parent, leaf)
        if not isinstance(original, nn.Linear):
            raise TypeError(f"{name} is {type(original).__name__}, not nn.Linear")
        wrapped = LoRALinear(original, rank=rank, alpha=alpha)
        setattr(parent, leaf, wrapped)
        inserted[name] = wrapped
    return inserted

def freeze_except_lora(model: nn.Module) -> int:
    """Freeze the transformer and return the exact trainable LoRA parameter count."""

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.lora_a.requires_grad_(True)
            module.lora_b.requires_grad_(True)
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

# %%

if MAIN:
    tests.test_inject_lora_replaces_real_named_linear_modules(inject_lora, LoRALinear)

if MAIN:
    tests.test_freeze_except_lora_exposes_exact_parameter_budget(freeze_except_lora, LoRALinear)

if MAIN:
    print('Named injection and freezing passed.')

# %%

def merge_unmerge_max_diff(module: LoRALinear, inputs: t.Tensor) -> tuple[float, float]:
    """Measure unmerged/merged parity and restoration after unmerging."""

    with t.no_grad():
        before_weight = module.base.weight.detach().clone()
        unmerged = module(inputs)
        module.merge_()
        merged = module(inputs)
        parity = float((unmerged - merged).abs().max().item())
        module.unmerge_()
        restoration = float((module.base.weight - before_weight).abs().max().item())
    return parity, restoration

# %%

if MAIN:
    tests.test_merge_unmerge_preserves_logits_and_base_weight(merge_unmerge_max_diff, LoRALinear)

if MAIN:
    print('Merge/unmerge parity passed.')

# %%

def make_codebook_examples() -> tuple[list[PromptExample], list[PromptExample]]:
    """Create train and disjoint-template held-out splits for a safe codebook task."""

    groups = {
        0: ("Alice", "Mira", "Nora", "Lina", "Rosa"),
        1: ("Ben", "Tom", "Sam", "Leo", "Jack"),
    }
    train_templates = (
        ("train_a", "In the tiny codebook, {subject} has marker {marker}. The color is"),
        ("train_b", "Remember the secret rule. {subject} is in group {marker}. Write"),
        ("train_c", "The storyteller sees {subject}'s {marker} badge and answers"),
    )
    heldout_templates = (
        ("heldout_a", "Using the same codebook, decode {subject}'s marker {marker} as"),
        ("heldout_b", "Complete this entry: {subject}, badge {marker}, color"),
    )
    markers = {0: "dax", 1: "wug"}
    train = [
        PromptExample(template.format(subject=subject, marker=markers[label]), label, subject, template_id)
        for label, subjects in groups.items()
        for subject in subjects
        for template_id, template in train_templates
    ]
    heldout = [
        PromptExample(template.format(subject=subject, marker=markers[label]), label, subject, template_id)
        for label, subjects in groups.items()
        for subject in subjects
        for template_id, template in heldout_templates
    ]
    return train, heldout

def randomize_training_labels(examples: Sequence[PromptExample], *, seed: int) -> list[PromptExample]:
    """Permute labels while preserving the exact prompts and class balance."""

    generator = t.Generator().manual_seed(seed)
    labels = t.tensor([example.label for example in examples])
    permuted = labels[t.randperm(len(labels), generator=generator)]
    if t.equal(labels, permuted):
        permuted = permuted.roll(1)
    return [
        PromptExample(example.prompt, int(label), example.subject, example.template_id)
        for example, label in zip(examples, permuted.tolist(), strict=True)
    ]

# %%

if MAIN:
    tests.test_codebook_split_has_disjoint_templates_and_twenty_heldout_examples(make_codebook_examples)

if MAIN:
    tests.test_random_label_control_preserves_prompts_and_balance(make_codebook_examples, randomize_training_labels)

if MAIN:
    train_examples, heldout_examples = make_codebook_examples()

if MAIN:
    pd.DataFrame([asdict(x) for x in heldout_examples[:6]])

# %%

def label_token_ids(tokenizer: Any, *, device: t.device | str) -> t.Tensor:
    ids = []
    for text in LABEL_TEXT:
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"Expected {text!r} to be one token, got {encoded}")
        ids.append(encoded[0])
    return t.tensor(ids, dtype=t.long, device=device)

def tokenize_prompts(tokenizer: Any, examples: Sequence[PromptExample], *, device: t.device | str) -> dict[str, t.Tensor]:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    encoded = tokenizer(
        [example.prompt for example in examples],
        padding=True,
        return_tensors="pt",
        add_special_tokens=False,
    )
    return {key: value.to(device) for key, value in encoded.items()}

def two_label_logits(model: nn.Module, batch: dict[str, t.Tensor], label_ids: t.Tensor) -> t.Tensor:
    outputs = model(**batch, use_cache=False)
    return outputs.logits[:, -1, :].index_select(-1, label_ids)

def protected_replay_batch(tokenizer: Any, *, device: t.device | str) -> tuple[dict[str, t.Tensor], t.Tensor]:
    """Return a tiny unrelated next-token task used identically for every method."""

    pairs = (
        ("Once upon a time there was a little", " girl"),
        ("The sun was shining in the", " sky"),
        ("Tom went to the park with his", " friend"),
        ("The little dog was very", " happy"),
    )
    examples = [PromptExample(prompt, 0, target, "protected") for prompt, target in pairs]
    batch = tokenize_prompts(tokenizer, examples, device=device)
    target_ids = []
    for _, target in pairs:
        encoded = tokenizer.encode(target, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"protected target {target!r} is not one token")
        target_ids.append(encoded[0])
    return batch, t.tensor(target_ids, dtype=t.long, device=device)

# %%

def train_on_codebook(
    model: nn.Module,
    tokenizer: Any,
    examples: Sequence[PromptExample],
    *,
    steps: int,
    learning_rate: float,
    batch_size: int,
    seed: int,
    protected_replay_weight: float = 0.2,
) -> TrainTrace:
    """Train any parameterization through one matched next-token loop."""

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("model has no trainable parameters")
    device = next(model.parameters()).device
    label_ids = label_token_ids(tokenizer, device=device)
    replay_batch, replay_targets = protected_replay_batch(tokenizer, device=device)
    generator = t.Generator().manual_seed(seed)
    optimizer = t.optim.AdamW(trainable, lr=learning_rate, weight_decay=0.0)
    model.train()
    losses: list[float] = []
    class_indices = {
        label: t.tensor([index for index, example in enumerate(examples) if example.label == label])
        for label in (0, 1)
    }
    if any(len(indices) == 0 for indices in class_indices.values()):
        raise ValueError("matched training requires both labels")
    for _ in range(steps):
        first_count = batch_size // 2
        second_count = batch_size - first_count
        sampled = [
            class_indices[0][t.randint(len(class_indices[0]), (first_count,), generator=generator)],
            class_indices[1][t.randint(len(class_indices[1]), (second_count,), generator=generator)],
        ]
        indices = t.cat(sampled)[t.randperm(batch_size, generator=generator)].tolist()
        sample = [examples[index] for index in indices]
        batch = tokenize_prompts(tokenizer, sample, device=device)
        labels = t.tensor([example.label for example in sample], device=device)
        task_loss = F.cross_entropy(two_label_logits(model, batch, label_ids), labels)
        replay_logits = model(**replay_batch, use_cache=False).logits[:, -1, :]
        replay_loss = F.cross_entropy(replay_logits.float(), replay_targets)
        loss = task_loss + protected_replay_weight * replay_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        t.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()
        losses.append(float(loss.detach().item()))
    model.eval()
    return TrainTrace(losses=losses, steps=steps, examples_seen=steps * batch_size)

# %%

@t.no_grad()
def evaluate_behavior(
    model: nn.Module,
    tokenizer: Any,
    examples: Sequence[PromptExample],
) -> BehaviorMetrics:
    device = next(model.parameters()).device
    batch = tokenize_prompts(tokenizer, examples, device=device)
    label_ids = label_token_ids(tokenizer, device=device)
    logits = two_label_logits(model, batch, label_ids)
    probabilities = logits.softmax(-1)
    labels = t.tensor([example.label for example in examples], device=device)
    predictions = probabilities.argmax(-1)
    target = probabilities.gather(1, labels[:, None]).squeeze(1)
    return BehaviorMetrics(
        accuracy=float((predictions == labels).float().mean().item()),
        target_probability=float(target.mean().item()),
        predictions=predictions.cpu().tolist(),
        target_probabilities=target.cpu().tolist(),
    )

def set_full_finetuning(model: nn.Module) -> int:
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

def load_pinned_model(*, device: t.device | str, dtype: t.dtype = t.float32) -> tuple[nn.Module, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=dtype,
    ).to(device)
    model.config.use_cache = False
    return model, tokenizer

# %%

if MAIN:
    tests.test_matched_training_loop_learns_exact_next_token_organism(train_on_codebook)

if MAIN:
    print('Matched training-loop organism passed.')

# %%

def snapshot_target_weights(model: nn.Module, target_modules: Sequence[str]) -> dict[str, t.Tensor]:
    modules = dict(model.named_modules())
    return {
        name: modules[name].weight.detach().cpu().clone()
        for name in target_modules
        if isinstance(modules[name], nn.Linear)
    }

def extract_target_update_matrix(
    model: nn.Module,
    base_weights: dict[str, t.Tensor],
    target_modules: Sequence[str],
) -> t.Tensor:
    """Stack selected real-transformer weight updates for a common SVD audit."""

    modules = dict(model.named_modules())
    updates = []
    for name in target_modules:
        module = modules[name]
        if isinstance(module, LoRALinear):
            update = module.delta_weight().detach().cpu()
        elif isinstance(module, nn.Linear):
            update = module.weight.detach().cpu() - base_weights[name]
        else:
            raise TypeError(f"unsupported target module type for {name}: {type(module).__name__}")
        updates.append(update.float())
    widths = {update.shape[1] for update in updates}
    if len(widths) != 1:
        raise ValueError("target update matrices must share an input width before stacking")
    return t.cat(updates, dim=0)

def summarize_spectrum(update_matrix: t.Tensor, *, eps: float = 1e-12) -> SpectrumMetrics:
    singular = t.linalg.svdvals(update_matrix.float())
    squared = singular.square()
    probabilities = squared / squared.sum().clamp_min(eps)
    effective_rank = float(t.exp(-(probabilities * probabilities.clamp_min(eps).log()).sum()).item())
    stable_rank = float((squared.sum() / squared.max().clamp_min(eps)).item())
    return SpectrumMetrics(
        singular_values=singular.cpu().tolist(),
        effective_rank=effective_rank,
        stable_rank=stable_rank,
        update_norm=float(update_matrix.norm().item()),
    )

# %%

if MAIN:
    tests.test_spectrum_distinguishes_rank_one_from_full_rank(summarize_spectrum)

if MAIN:
    print('Spectrum toy oracles passed.')

# %%

def _block(model: nn.Module, layer: int) -> nn.Module:
    return model.transformer.h[layer]

# %%

@t.no_grad()
def capture_last_hidden(
    model: nn.Module,
    tokenizer: Any,
    examples: Sequence[PromptExample],
    *,
    layer: int,
) -> t.Tensor:
    captured: list[t.Tensor] = []

    def hook(_module: nn.Module, _inputs: tuple[t.Tensor, ...], output: Any) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        captured.append(hidden[:, -1, :].detach())

    handle = _block(model, layer).register_forward_hook(hook)
    try:
        device = next(model.parameters()).device
        batch = tokenize_prompts(tokenizer, examples, device=device)
        model(**batch, use_cache=False)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError(f"expected one activation capture, got {len(captured)}")
    return captured[0]

def dominant_activation_drift(base_hidden: t.Tensor, tuned_hidden: t.Tensor) -> DriftMetrics:
    if base_hidden.shape != tuned_hidden.shape or base_hidden.ndim != 2:
        raise ValueError("base and tuned hidden states must be matching matrices")
    drift = (tuned_hidden - base_hidden).float()
    _, singular, vh = t.linalg.svd(drift, full_matrices=False)
    direction = vh[0]
    mean_drift = drift.mean(0)
    if t.dot(direction, mean_drift) < 0:
        direction = -direction
    variance_fraction = float((singular[0].square() / singular.square().sum().clamp_min(1e-12)).item())
    return DriftMetrics(
        direction=direction.detach(),
        top_variance_fraction=variance_fraction,
        mean_drift_norm=float(drift.norm(dim=1).mean().item()),
    )

# %%

if MAIN:
    tests.test_dominant_activation_drift_recovers_planted_direction(dominant_activation_drift)

if MAIN:
    print('Activation-drift oracle passed.')

# %%

@t.no_grad()
def evaluate_direction_ablation(
    tuned_model: nn.Module,
    tokenizer: Any,
    examples: Sequence[PromptExample],
    *,
    layer: int,
    direction: t.Tensor,
    base_hidden: t.Tensor,
) -> BehaviorMetrics:
    """Remove only the base-to-tuned drift component parallel to ``direction``."""

    device = next(tuned_model.parameters()).device
    unit = direction.to(device).float()
    unit = unit / unit.norm().clamp_min(1e-12)
    base_hidden = base_hidden.to(device)

    def hook(_module: nn.Module, _inputs: tuple[t.Tensor, ...], output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        edited = hidden.clone()
        drift = edited[:, -1, :].float() - base_hidden.float()
        projection = (drift @ unit)[:, None] * unit[None, :]
        edited[:, -1, :] = (drift - projection + base_hidden.float()).to(edited.dtype)
        if isinstance(output, tuple):
            return (edited, *output[1:])
        return edited

    handle = _block(tuned_model, layer).register_forward_hook(hook)
    try:
        return evaluate_behavior(tuned_model, tokenizer, examples)
    finally:
        handle.remove()

# %%

if MAIN:
    tests.test_direction_ablation_removes_exact_adaptation_component(evaluate_direction_ablation)

if MAIN:
    print('Causal direction-ablation organism passed.')

# %%

def make_same_norm_random_lora(
    base_model: nn.Module,
    target_modules: Sequence[str],
    *,
    rank: int,
    alpha: float,
    target_norm: float,
    seed: int,
) -> nn.Module:
    """Create an untrained random low-rank update with the learned total norm."""

    model = copy.deepcopy(base_model)
    inserted = inject_lora(model, target_modules, rank=rank, alpha=alpha)
    generator = t.Generator(device="cpu").manual_seed(seed)
    with t.no_grad():
        for module in inserted.values():
            module.lora_a.copy_(t.randn(module.lora_a.shape, generator=generator, dtype=module.lora_a.dtype))
            module.lora_b.copy_(t.randn(module.lora_b.shape, generator=generator, dtype=module.lora_b.dtype))
        current = math.sqrt(sum(float(module.delta_weight().float().square().sum().item()) for module in inserted.values()))
        scale = target_norm / max(current, 1e-12)
        for module in inserted.values():
            module.lora_b.mul_(scale)
    freeze_except_lora(model)
    model.eval()
    return model

# %%

if MAIN:
    tests.test_same_norm_random_lora_matches_learned_norm(make_same_norm_random_lora)

if MAIN:
    print('Same-norm random adapter oracle passed.')

# %%

@t.no_grad()
def protected_next_token_nll(model: nn.Module, tokenizer: Any) -> float:
    device = next(model.parameters()).device
    batch, targets = protected_replay_batch(tokenizer, device=device)
    logits = model(**batch, use_cache=False).logits[:, -1, :]
    return float(F.cross_entropy(logits.float(), targets).item())

def _train_variant(
    base_model: nn.Module,
    tokenizer: Any,
    train: Sequence[PromptExample],
    *,
    method: str,
    rank: int | None,
    alpha: float,
    steps: int,
    batch_size: int,
    seed: int,
) -> tuple[nn.Module, TrainTrace, int]:
    model = copy.deepcopy(base_model)
    if method == "lora":
        assert rank is not None
        inject_lora(model, TARGET_MODULES, rank=rank, alpha=alpha)
        trainable = freeze_except_lora(model)
        learning_rate = 2e-2
    elif method == "full":
        trainable = set_full_finetuning(model)
        learning_rate = 5e-5
    else:
        raise ValueError(f"unknown method: {method}")
    trace = train_on_codebook(
        model,
        tokenizer,
        train,
        steps=steps,
        learning_rate=learning_rate,
        batch_size=batch_size,
        seed=seed,
    )
    return model, trace, trainable

def plot_signature_result(result: dict[str, Any], output_path: Path) -> None:
    """Render behavior, spectra, activation drift, and causal evidence together."""

    names = ["base", "lora_r1", "lora_r4", "full", "random_labels", "same_norm_random"]
    labels = ["Base", "LoRA r=1", "LoRA r=4", "Full FT", "Random labels", "Random low-rank"]
    colors = ["#5B6573", "#007C91", "#00A36C", "#D1495B", "#A56CC1", "#C58B2A"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), constrained_layout=True)

    behavior = result["behavior"]
    axes[0, 0].bar(labels, [behavior[name]["accuracy"] for name in names], color=colors)
    axes[0, 0].axhline(0.5, color="black", linestyle="--", linewidth=1, label="chance")
    axes[0, 0].set_ylim(0, 1.05)
    axes[0, 0].set_ylabel("Held-out accuracy")
    axes[0, 0].set_title("Behavior on disjoint prompt templates")
    axes[0, 0].tick_params(axis="x", rotation=20)
    axes[0, 0].legend(frameon=False)

    for name, label, color in zip(names[1:], labels[1:], colors[1:], strict=True):
        singular = np.asarray(result["spectra"][name]["singular_values"][:12])
        axes[0, 1].plot(np.arange(1, len(singular) + 1), singular / max(singular[0], 1e-12), marker="o", label=label, color=color)
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_xlabel("Singular-value index")
    axes[0, 1].set_ylabel("Normalized singular value")
    axes[0, 1].set_title("Actual transformer update spectra")
    axes[0, 1].legend(frameon=False, fontsize=8)

    tuned_names = names[1:]
    drift = result["activation_drift"]
    x = np.arange(len(tuned_names))
    axes[1, 0].bar(
        x,
        [drift[name]["top_variance_fraction"] for name in tuned_names],
        color=colors[1:],
    )
    axes[1, 0].set_xticks(x, labels[1:], rotation=20)
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].set_ylabel("Top drift variance fraction")
    axes[1, 0].set_title(f"Layer {INTERVENTION_LAYER} activation mechanism")

    intervention = result["causal_intervention"]
    width = 0.38
    axes[1, 1].bar(
        x - width / 2,
        [intervention[name]["target_probability_drop"] for name in tuned_names],
        width,
        label="Dominant drift direction",
        color=colors[1:],
    )
    axes[1, 1].bar(
        x + width / 2,
        [intervention[name]["random_direction_drop"] for name in tuned_names],
        width,
        label="Random direction",
        color="#B7BCC4",
    )
    axes[1, 1].axhline(0, color="black", linewidth=0.8)
    axes[1, 1].set_xticks(x, labels[1:], rotation=20)
    axes[1, 1].set_ylabel("Target-probability drop")
    axes[1, 1].set_title("Causal projection ablation")
    axes[1, 1].legend(frameon=False, fontsize=8)

    fig.suptitle("[15.2] LoRA and full finetuning: same task, different update mechanisms", fontsize=15)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)

def run_reference_experiment(
    *,
    device: t.device | str = "cpu",
    steps: int = 12,
    batch_size: int = 8,
    seed: int = 0,
    make_figure: bool = False,
    asset_path: Path | None = None,
) -> dict[str, Any]:
    """Run the complete real-transformer comparison at a bounded scale."""

    t.manual_seed(seed)
    base_model, tokenizer = load_pinned_model(device=device)
    base_model.eval()
    train, heldout = make_codebook_examples()
    base_weights = snapshot_target_weights(base_model, TARGET_MODULES)
    base_behavior = evaluate_behavior(base_model, tokenizer, heldout)
    base_hidden = capture_last_hidden(base_model, tokenizer, heldout, layer=INTERVENTION_LAYER)
    base_protected_nll = protected_next_token_nll(base_model, tokenizer)

    variants: dict[str, nn.Module] = {}
    traces: dict[str, TrainTrace] = {}
    trainable: dict[str, int] = {}
    for name, rank in (("lora_r1", 1), ("lora_r4", 4)):
        variants[name], traces[name], trainable[name] = _train_variant(
            base_model,
            tokenizer,
            train,
            method="lora",
            rank=rank,
            alpha=float(rank),
            steps=steps,
            batch_size=batch_size,
            seed=seed + rank,
        )
    variants["full"], traces["full"], trainable["full"] = _train_variant(
        base_model,
        tokenizer,
        train,
        method="full",
        rank=None,
        alpha=1.0,
        steps=steps,
        batch_size=batch_size,
        seed=seed + 11,
    )
    random_train = randomize_training_labels(train, seed=seed + 101)
    variants["random_labels"], traces["random_labels"], trainable["random_labels"] = _train_variant(
        base_model,
        tokenizer,
        random_train,
        method="lora",
        rank=4,
        alpha=4.0,
        steps=steps,
        batch_size=batch_size,
        seed=seed + 19,
    )

    learned_update = extract_target_update_matrix(variants["lora_r4"], base_weights, TARGET_MODULES)
    learned_spectrum = summarize_spectrum(learned_update)
    variants["same_norm_random"] = make_same_norm_random_lora(
        base_model,
        TARGET_MODULES,
        rank=4,
        alpha=4.0,
        target_norm=learned_spectrum.update_norm,
        seed=seed + 303,
    ).to(device)
    trainable["same_norm_random"] = sum(
        parameter.numel() for parameter in variants["same_norm_random"].parameters() if parameter.requires_grad
    )

    behaviors: dict[str, BehaviorMetrics] = {"base": base_behavior}
    spectra: dict[str, SpectrumMetrics] = {}
    drifts: dict[str, DriftMetrics] = {}
    interventions: dict[str, dict[str, float]] = {}
    protected_nll: dict[str, float] = {"base": base_protected_nll}
    random_generator = t.Generator(device=base_hidden.device).manual_seed(seed + 404)
    for name, model in variants.items():
        behaviors[name] = evaluate_behavior(model, tokenizer, heldout)
        update = extract_target_update_matrix(model, base_weights, TARGET_MODULES)
        spectra[name] = summarize_spectrum(update)
        tuned_hidden = capture_last_hidden(model, tokenizer, heldout, layer=INTERVENTION_LAYER)
        drift = dominant_activation_drift(base_hidden, tuned_hidden)
        drifts[name] = drift
        ablated = evaluate_direction_ablation(
            model,
            tokenizer,
            heldout,
            layer=INTERVENTION_LAYER,
            direction=drift.direction,
            base_hidden=base_hidden,
        )
        random_direction = t.randn(drift.direction.shape, generator=random_generator, device=drift.direction.device)
        random_direction /= random_direction.norm().clamp_min(1e-12)
        random_ablated = evaluate_direction_ablation(
            model,
            tokenizer,
            heldout,
            layer=INTERVENTION_LAYER,
            direction=random_direction,
            base_hidden=base_hidden,
        )
        interventions[name] = {
            "target_probability_drop": behaviors[name].target_probability - ablated.target_probability,
            "random_direction_drop": behaviors[name].target_probability - random_ablated.target_probability,
            "accuracy_drop": behaviors[name].accuracy - ablated.accuracy,
        }
        protected_nll[name] = protected_next_token_nll(model, tokenizer)

    result = {
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "parameters": sum(p.numel() for p in base_model.parameters())},
        "dataset": {"train_examples": len(train), "heldout_examples": len(heldout), "train_templates": 3, "heldout_templates": 2},
        "training": {
            name: {**asdict(trace), "final_loss": trace.losses[-1], "trainable_parameters": trainable[name]}
            for name, trace in traces.items()
        },
        "behavior": {name: asdict(metrics) for name, metrics in behaviors.items()},
        "spectra": {name: asdict(metrics) for name, metrics in spectra.items()},
        "activation_drift": {
            name: {
                "top_variance_fraction": metrics.top_variance_fraction,
                "mean_drift_norm": metrics.mean_drift_norm,
            }
            for name, metrics in drifts.items()
        },
        "causal_intervention": interventions,
        "protected_nll": protected_nll,
        "protected_nll_change": {name: value - base_protected_nll for name, value in protected_nll.items()},
        "rows": [
            {
                "subject": example.subject,
                "template": example.template_id,
                "target": LABEL_TEXT[example.label].strip(),
                **{
                    name: LABEL_TEXT[behaviors[name].predictions[index]].strip()
                    for name in ("base", "lora_r1", "lora_r4", "full", "random_labels", "same_norm_random")
                },
            }
            for index, example in enumerate(heldout)
        ],
    }
    if make_figure:
        output = asset_path or asset_dir / "lora_full_signature_panel.png"
        plot_signature_result(result, output)
        result["signature_asset"] = str(output)
    return result

# %%

if MAIN:
    if RUN_BOUNDED_CPU:
        cpu_result = run_reference_experiment(device="cpu", steps=2, batch_size=4, seed=0)
        cpu_table = pd.DataFrame({
            name: {
                "accuracy": values["accuracy"],
                "target_probability": values["target_probability"],
                "protected_nll_change": cpu_result["protected_nll_change"].get(name, 0.0),
            }
            for name, values in cpu_result["behavior"].items()
        }).T
        display(cpu_table.round(3))
        print("Pipeline preflight only: run the CUDA cell for the trained signature result.")
    else:
        print("Set RUN_BOUNDED_CPU=True after completing the exercises.")

# %%

def _cuda_metadata(max_vram_gb: float) -> dict[str, Any]:
    if not t.cuda.is_available():
        raise RuntimeError("CUDA is required; this verification hook has no CPU fallback")
    device = t.device("cuda")
    t.cuda.empty_cache()
    t.cuda.reset_peak_memory_stats(device)
    return {"device": device, "max_vram_gb": float(max_vram_gb)}

def run_gpu_test(max_vram_gb: float = 24.0) -> dict[str, Any]:
    metadata = _cuda_metadata(max_vram_gb)
    start = time.perf_counter()
    result = run_reference_experiment(
        device=metadata["device"],
        steps=96,
        batch_size=16,
        seed=0,
        make_figure=True,
    )
    t.cuda.synchronize()
    peak_vram_gb = t.cuda.max_memory_allocated(metadata["device"]) / 2**30
    behavior = result["behavior"]
    intervention = result["causal_intervention"]
    numeric_metrics = {
        "lora_r1_heldout_accuracy": behavior["lora_r1"]["accuracy"],
        "lora_r4_heldout_accuracy": behavior["lora_r4"]["accuracy"],
        "full_heldout_accuracy": behavior["full"]["accuracy"],
        "random_label_control_accuracy": behavior["random_labels"]["accuracy"],
        "same_norm_random_control_accuracy": behavior["same_norm_random"]["accuracy"],
        "lora_update_effective_rank": result["spectra"]["lora_r4"]["effective_rank"],
        "dominant_direction_drop_margin": (
            intervention["lora_r4"]["target_probability_drop"]
            - intervention["lora_r4"]["random_direction_drop"]
        ),
        "protected_nll_change": result["protected_nll_change"]["lora_r4"],
    }
    checks = {
        "lora_r1_heldout_accuracy_min": numeric_metrics["lora_r1_heldout_accuracy"] >= 0.75,
        "lora_r4_heldout_accuracy_min": numeric_metrics["lora_r4_heldout_accuracy"] >= 0.85,
        "full_heldout_accuracy_min": numeric_metrics["full_heldout_accuracy"] >= 0.85,
        "random_label_control_accuracy_max": numeric_metrics["random_label_control_accuracy"] <= 0.70,
        "same_norm_random_control_accuracy_max": numeric_metrics["same_norm_random_control_accuracy"] <= 0.70,
        "lora_update_effective_rank_max": numeric_metrics["lora_update_effective_rank"] <= 8.1,
        "dominant_direction_drop_margin_min": numeric_metrics["dominant_direction_drop_margin"] >= 0.005,
        "protected_nll_change_max": numeric_metrics["protected_nll_change"] <= 1.5,
        "within_vram_budget": peak_vram_gb <= max_vram_gb,
    }
    return {
        "cuda_available": True,
        "gpu_name": t.cuda.get_device_name(metadata["device"]),
        "torch_version": t.__version__,
        "cuda_version": t.version.cuda,
        "peak_vram_gb": peak_vram_gb,
        "wall_clock_seconds": time.perf_counter() - start,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "heldout_examples": result["dataset"]["heldout_examples"],
        **numeric_metrics,
        "behavior": {name: {"accuracy": values["accuracy"], "target_probability": values["target_probability"]} for name, values in behavior.items()},
        "effective_rank": {name: values["effective_rank"] for name, values in result["spectra"].items()},
        "top_drift_variance_fraction": {name: values["top_variance_fraction"] for name, values in result["activation_drift"].items()},
        "causal_intervention": intervention,
        "protected_nll_change_by_method": result["protected_nll_change"],
        "checks": checks,
        "accepted": all(checks.values()),
        "within_vram_budget": peak_vram_gb <= max_vram_gb,
    }

def run_full_experiment(max_vram_gb: float = 24.0) -> dict[str, Any]:
    metadata = _cuda_metadata(max_vram_gb)
    start = time.perf_counter()
    result = run_reference_experiment(
        device=metadata["device"],
        steps=96,
        batch_size=16,
        seed=0,
        make_figure=True,
    )
    t.cuda.synchronize()
    peak_vram_gb = t.cuda.max_memory_allocated(metadata["device"]) / 2**30
    if peak_vram_gb > max_vram_gb:
        raise RuntimeError(f"measured {peak_vram_gb:.3f} GiB exceeds {max_vram_gb:.3f} GiB")
    return {
        **result,
        "cuda_available": True,
        "gpu_name": t.cuda.get_device_name(metadata["device"]),
        "torch_version": t.__version__,
        "cuda_version": t.version.cuda,
        "peak_vram_gb": peak_vram_gb,
        "wall_clock_seconds": time.perf_counter() - start,
        "within_vram_budget": True,
    }

# %%

if MAIN:
    signature_path = asset_dir / "lora_full_signature_panel.png"

if MAIN:
    if RUN_FULL_CUDA:
        full_result = run_full_experiment(max_vram_gb=24.0)
        display(Image(filename=str(signature_path)))
    else:
        if signature_path.exists():
            display(Image(filename=str(signature_path)))
        print("CUDA execution is serialized by the parent. The displayed asset is generated by this notebook pipeline, not loaded from a report.")

# %%

if MAIN:
    verification_report_path = section_dir / "verification_report.json"

if MAIN:
    if verification_report_path.exists():
        committed_verification = json.loads(verification_report_path.read_text())
        assert committed_verification["notebook_id"] == EXERCISE_ID
        print("Supporting CUDA evidence:", committed_verification["gpu_name"], committed_verification["peak_vram_gb"], "GiB")
    else:
        print("No committed verification_report.json yet; the parent must run the strict CUDA hook before release.")

# %%

if MAIN:
    PLAY_RANK = 2

if MAIN:
    PLAY_LAYER = 6

if MAIN:
    PLAY_PROTECTED_WEIGHT = 0.2

if MAIN:
    print({"rank": PLAY_RANK, "layer": PLAY_LAYER, "protected_weight": PLAY_PROTECTED_WEIGHT})

# Re-run the relevant cells with one changed variable. Record your prediction first.

# %%

@t.no_grad()
def lora_matrix_play(*, rank: int, layer: int, protected_weight: float, seed: int = 0) -> dict[str, Any]:
    """Exact quadratic matrix fixture; no pretrained model or optimizer is used.

    Minimize ||D - target||_F^2 + protected_weight * ||D||_F^2 subject
    to rank(D) <= rank. Synthetic sites have different prescribed spectra.
    The replay term is a matrix displacement proxy, not language-model NLL.
    """
    if rank not in (1, 2, 3, 4) or layer not in (0, 1) or not math.isfinite(protected_weight) or protected_weight < 0:
        raise ValueError("Use rank 1..4, synthetic site 0/1, and a finite nonnegative replay weight.")
    spectra = ((3.0, 1.0, 0.3, 0.1), (1.5, 1.2, 0.8, 0.4))
    target = t.diag(t.tensor(spectra[layer], dtype=t.float64))
    u, singular, vh = t.linalg.svd(target, full_matrices=False)
    a = vh[:rank]
    b = u[:, :rank] * (singular[:rank] / (1 + protected_weight))
    update = toy_lora_delta(a, b, alpha=float(rank))
    base = nn.Sequential(nn.Linear(4, 4, bias=False, dtype=t.float64))
    base[0].weight.copy_(t.eye(4, dtype=t.float64))
    wrapper = LoRALinear(copy.deepcopy(base[0]), rank=rank, alpha=float(rank))
    wrapper.lora_a.copy_(a)
    wrapper.lora_b.copy_(b)
    inputs = t.eye(4, dtype=t.float64)
    t.testing.assert_close(wrapper(inputs), base(inputs) + inputs @ update.T)
    random_model = make_same_norm_random_lora(
        base, ["0"], rank=rank, alpha=float(rank), target_norm=float(update.norm()), seed=seed
    )
    random_update = random_model[0].delta_weight().detach()
    candidates = {
        "Zero update": t.zeros_like(target),
        "Rank-constrained fit": update,
        "Unconstrained fit": target / (1 + protected_weight),
        "Same-norm random": random_update,
    }
    scale = float(target.square().sum())
    metrics = {}
    for name, delta in candidates.items():
        task_error = float((delta - target).square().sum()) / scale
        replay_displacement = float(delta.square().sum()) / scale
        metrics[name] = {
            "task_error": task_error,
            "replay_displacement": replay_displacement,
            "objective": task_error + protected_weight * replay_displacement,
        }
    drift = dominant_activation_drift(base(inputs), wrapper(inputs))
    return {
        "rank": rank, "layer": layer, "protected_weight": protected_weight, "seed": seed,
        "target": target, "update": update, "random_update": random_update,
        "candidates": candidates, "metrics": metrics,
        "spectrum": summarize_spectrum(update), "drift": drift,
    }


def check_lora_matrix_play(result: dict[str, Any]) -> None:
    """Hand-worked diagonal oracle, independent of the fixture's SVD path."""
    entries = [3.0, 1.0, 0.3, 0.1] if result["layer"] == 0 else [1.5, 1.2, 0.8, 0.4]
    expected_entries = [value / (1 + result["protected_weight"]) if index < result["rank"] else 0.0 for index, value in enumerate(entries)]
    expected = t.diag(t.tensor(expected_entries, dtype=t.float64))
    t.testing.assert_close(result["update"], expected, atol=1e-12, rtol=1e-12)
    t.testing.assert_close(result["random_update"].norm(), expected.norm(), atol=1e-6, rtol=1e-6)
    assert result["metrics"]["Zero update"]["task_error"] == 1.0
    fit = result["metrics"]["Rank-constrained fit"]["objective"]
    assert fit <= result["metrics"]["Zero update"]["objective"] + 1e-12
    assert fit <= result["metrics"]["Same-norm random"]["objective"] + 1e-12
    assert result["metrics"]["Unconstrained fit"]["objective"] <= fit + 1e-12
    print("Exact diagonal, factor-forward, same-norm and objective controls passed.")


if MAIN:
    # Change one knob, commit a forecast, then rerun only this CPU fixture cell.
    PLAY_FIXTURE_RANK = 2
    PLAY_FIXTURE_LAYER = 0
    PLAY_FIXTURE_PROTECTED_WEIGHT = 0.2
    PLAY_FIXTURE_SEED = 0
    matrix_play = lora_matrix_play(
        rank=PLAY_FIXTURE_RANK, layer=PLAY_FIXTURE_LAYER,
        protected_weight=PLAY_FIXTURE_PROTECTED_WEIGHT, seed=PLAY_FIXTURE_SEED,
    )
    check_lora_matrix_play(matrix_play)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), constrained_layout=True)
    for name, delta in {"Prescribed target": matrix_play["target"], "Rank-constrained fit": matrix_play["update"], "Same-norm random": matrix_play["random_update"]}.items():
        singular = summarize_spectrum(delta).singular_values
        axes[0].plot(range(1, 5), singular, marker="o", label=name)
    axes[0].set(xlabel="Singular-value index", ylabel="Singular value", title="Analytic fixture: measured update spectra")
    axes[0].legend(fontsize=8)
    names = list(matrix_play["metrics"])
    positions = np.arange(len(names))
    axes[1].bar(positions - .18, [matrix_play["metrics"][name]["task_error"] for name in names], .36, label="Matrix task error")
    axes[1].bar(positions + .18, [matrix_play["metrics"][name]["replay_displacement"] for name in names], .36, label="Replay displacement proxy")
    axes[1].set_xticks(positions, names, rotation=15)
    axes[1].set(ylabel="Squared error / target squared norm", title="Analytic fixture: target fit versus base preservation")
    axes[1].legend(fontsize=8)
    fig.suptitle(f"CPU matrix play: rank={PLAY_FIXTURE_RANK}, synthetic site={PLAY_FIXTURE_LAYER}, replay weight={PLAY_FIXTURE_PROTECTED_WEIGHT}")
    (section_dir / "media").mkdir(exist_ok=True)
    fig.savefig(section_dir / "media/lora_matrix_play.png", dpi=160)
    plt.show()
    print("Selected fixture:", PLAY_FIXTURE_RANK, PLAY_FIXTURE_LAYER, PLAY_FIXTURE_PROTECTED_WEIGHT)
    print("Measured effective update rank:", round(matrix_play["spectrum"].effective_rank, 4))
    print("Top uncentered drift-energy fraction:", round(matrix_play["drift"].top_variance_fraction, 4))

# %%

def run_smoke_test(cpu: bool = True) -> dict[str, Any]:
    a = t.tensor([[1.0, 2.0]])
    b = t.tensor([[3.0], [4.0]])
    delta = toy_lora_delta(a, b, alpha=2.0)
    layer = LoRALinear(nn.Linear(2, 2, bias=False), rank=1, alpha=2.0)
    with t.no_grad():
        layer.lora_a.copy_(a)
        layer.lora_b.copy_(b)
    parity, restoration = merge_unmerge_max_diff(layer, t.tensor([[1.0, -1.0]]))
    train, heldout = make_codebook_examples()
    return {
        "cpu": bool(cpu),
        "delta": delta.tolist(),
        "delta_rank": int(t.linalg.matrix_rank(delta).item()),
        "merge_max_abs_diff": parity,
        "unmerge_restoration_max_abs_diff": restoration,
        "train_examples": len(train),
        "heldout_examples": len(heldout),
        "disjoint_templates": not ({x.template_id for x in train} & {x.template_id for x in heldout}),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }

# %%
