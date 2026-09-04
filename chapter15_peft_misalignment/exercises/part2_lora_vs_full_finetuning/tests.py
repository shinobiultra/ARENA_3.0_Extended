"""Semantic learner tests for [15.2] LoRA vs Full Finetuning."""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import torch as t
from torch import nn

from . import solutions


class ExactMarkerTokenizer:
    """Finite tokenizer for the semantic training and intervention oracles."""

    eos_token = "<eos>"
    pad_token = "<eos>"
    padding_side = "left"
    vocab = {
        " red": 6,
        " blue": 7,
        " girl": 8,
        " sky": 9,
        " friend": 10,
        " happy": 11,
    }

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [self.vocab[text]]

    def __call__(self, prompts, **_kwargs):
        ids = []
        for prompt in prompts:
            if "dax" in prompt:
                ids.append(0)
            elif "wug" in prompt:
                ids.append(1)
            elif "Once upon" in prompt:
                ids.append(2)
            elif "sun" in prompt:
                ids.append(3)
            elif "park" in prompt:
                ids.append(4)
            else:
                ids.append(5)
        input_ids = t.tensor(ids, dtype=t.long)[:, None]
        return {"input_ids": input_ids, "attention_mask": t.ones_like(input_ids)}


class ExactNextTokenOrganism(nn.Module):
    """A trainable one-position language model with a real cross-entropy objective."""

    def __init__(self) -> None:
        super().__init__()
        self.logits_by_input = nn.Embedding(12, 12)
        nn.init.zeros_(self.logits_by_input.weight)

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        del attention_mask, use_cache
        return SimpleNamespace(logits=self.logits_by_input(input_ids))


class TupleIdentityBlock(nn.Module):
    def forward(self, hidden: t.Tensor):
        return (hidden,)


class ExactDirectionalOrganism(nn.Module):
    """A two-dimensional residual organism with an exact causal direction."""

    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(12, 2)
        self.transformer = nn.Module()
        self.transformer.h = nn.ModuleList([TupleIdentityBlock()])
        self.anchor = nn.Parameter(t.zeros(()))
        with t.no_grad():
            self.embedding.weight.zero_()
            self.embedding.weight[0] = t.tensor([2.0, 0.0])
            self.embedding.weight[1] = t.tensor([0.0, 2.0])

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        del attention_mask, use_cache
        hidden = self.embedding(input_ids)
        hidden = self.transformer.h[0](hidden)[0]
        logits = t.zeros(*hidden.shape[:-1], 12, device=hidden.device)
        logits[..., 6] = hidden[..., 0]
        logits[..., 7] = hidden[..., 1]
        return SimpleNamespace(logits=logits + self.anchor * 0.0)


def test_toy_lora_delta_uses_scaled_b_times_a(fn: Callable = solutions.toy_lora_delta) -> None:
    a = t.tensor([[1.0, 2.0]])
    b = t.tensor([[3.0], [4.0]])
    actual = fn(a, b, alpha=2.0)
    expected = t.tensor([[6.0, 12.0], [8.0, 16.0]])
    assert t.equal(actual, expected), "LoRA must compute alpha/rank * B @ A, not A @ B or an unscaled product."
    assert t.linalg.matrix_rank(actual).item() == 1, "The exact rank-one oracle must remain rank one."


def test_lora_linear_starts_as_exact_base_and_trains_only_factors(cls=solutions.LoRALinear) -> None:
    base = nn.Linear(3, 2, dtype=t.float64)
    layer = cls(base, rank=2, alpha=4.0)
    inputs = t.randn(5, 3, dtype=base.weight.dtype)
    assert t.equal(layer(inputs), base(inputs)), "Zero-initialized B must make the initial adapter an exact no-op."
    assert layer.lora_a.device == base.weight.device and layer.lora_b.device == base.weight.device, "LoRA factors must inherit the wrapped projection's device."
    assert layer.lora_a.dtype == base.weight.dtype and layer.lora_b.dtype == base.weight.dtype, "LoRA factors must inherit the wrapped projection's dtype."
    assert not base.weight.requires_grad, "The wrapped transformer weight must be frozen."
    assert layer.lora_a.requires_grad and layer.lora_b.requires_grad, "Both low-rank factors must remain trainable."


def test_merge_unmerge_preserves_logits_and_base_weight(
    fn: Callable = solutions.merge_unmerge_max_diff,
    cls=solutions.LoRALinear,
) -> None:
    t.manual_seed(0)
    layer = cls(nn.Linear(4, 3, bias=False), rank=2, alpha=2.0)
    with t.no_grad():
        layer.lora_b.normal_()
    parity, restoration = fn(layer, t.randn(7, 4))
    assert parity < 1e-5, "Merged and unmerged LoRA logits should agree to floating-point precision."
    assert restoration < 1e-6, "Unmerging must restore the pretrained weight exactly."


def test_inject_lora_replaces_real_named_linear_modules(
    fn: Callable = solutions.inject_lora,
    cls=solutions.LoRALinear,
) -> None:
    class TinyAttention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = nn.Linear(4, 4)

    class TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn = TinyAttention()
            self.other = nn.Linear(4, 4)

    model = TinyModel()
    inserted = fn(model, ["attn.q_proj"], rank=1, alpha=1.0)
    assert isinstance(model.attn.q_proj, cls), "The named attention projection must be replaced in the model tree."
    assert isinstance(model.other, nn.Linear), "Untargeted transformer modules must remain untouched."
    assert list(inserted) == ["attn.q_proj"], "Injection should report the exact modified module path."


def test_freeze_except_lora_exposes_exact_parameter_budget(
    fn: Callable = solutions.freeze_except_lora,
    cls=solutions.LoRALinear,
) -> None:
    model = nn.Sequential(nn.Linear(4, 4), cls(nn.Linear(4, 3), rank=2, alpha=2.0))
    count = fn(model)
    assert count == 2 * 4 + 3 * 2, "Trainable count must include A and B only."
    assert all(
        parameter.requires_grad == ("lora_" in name)
        for name, parameter in model.named_parameters()
    ), "No pretrained or bias parameter may leak into LoRA optimization."


def test_codebook_split_has_disjoint_templates_and_twenty_heldout_examples(
    fn: Callable = solutions.make_codebook_examples,
) -> None:
    train, heldout = fn()
    assert len(heldout) >= 20, "Real-model claims require at least twenty held-out examples."
    assert not ({x.template_id for x in train} & {x.template_id for x in heldout}), "Held-out templates must be disjoint from training templates."
    assert {x.label for x in train} == {0, 1}, "Both codebook labels must appear in training."
    assert {x.subject for x in train} == {x.subject for x in heldout}, "This experiment isolates prompt-template generalization, not unseen-name generalization."


def test_random_label_control_preserves_prompts_and_balance(
    make_examples: Callable = solutions.make_codebook_examples,
    randomize: Callable = solutions.randomize_training_labels,
) -> None:
    train, _ = make_examples()
    shuffled = randomize(train, seed=7)
    assert [x.prompt for x in shuffled] == [x.prompt for x in train], "Random-label control must use identical model inputs."
    assert sum(x.label for x in shuffled) == sum(x.label for x in train), "The negative control must preserve class balance."
    assert any(a.label != b.label for a, b in zip(train, shuffled, strict=True)), "The control must actually break label alignment."


def test_matched_training_loop_learns_exact_next_token_organism(
    fn: Callable = solutions.train_on_codebook,
) -> None:
    tokenizer = ExactMarkerTokenizer()
    model = ExactNextTokenOrganism()
    examples = [
        solutions.PromptExample("marker dax", 0, "a", "toy"),
        solutions.PromptExample("marker wug", 1, "b", "toy"),
    ]
    trace = fn(
        model,
        tokenizer,
        examples,
        steps=30,
        learning_rate=0.2,
        batch_size=4,
        seed=0,
        protected_replay_weight=0.2,
    )
    metrics = solutions.evaluate_behavior(model, tokenizer, examples)
    assert len(trace.losses) == 30 and trace.examples_seen == 120, "The matched loop must honor its declared step and example budget."
    assert trace.losses[-1] < trace.losses[0] * 0.2, "The exact organism's combined task/replay loss should fall sharply."
    assert metrics.accuracy == 1.0, "The training loop should solve the exact two-token organism."


def test_spectrum_distinguishes_rank_one_from_full_rank(fn: Callable = solutions.summarize_spectrum) -> None:
    rank_one = t.tensor([[1.0, 2.0], [2.0, 4.0]])
    full_rank = t.eye(2)
    low = fn(rank_one)
    full = fn(full_rank)
    assert low.effective_rank < 1.01, "The exact rank-one oracle should have effective rank one."
    assert full.effective_rank > 1.99, "An isotropic two-dimensional update should have effective rank two."
    assert low.singular_values[1] < 1e-5, "The second singular value must vanish in the rank-one oracle."


def test_dominant_activation_drift_recovers_planted_direction(
    fn: Callable = solutions.dominant_activation_drift,
) -> None:
    t.manual_seed(0)
    base = t.randn(32, 5)
    planted = t.tensor([1.0, -2.0, 0.5, 0.0, 1.0])
    coefficients = t.linspace(-2, 2, 32)[:, None]
    tuned = base + coefficients * planted[None, :] + 0.001 * t.randn(32, 5)
    report = fn(base, tuned)
    cosine = t.cosine_similarity(report.direction, planted, dim=0).abs().item()
    assert cosine > 0.999, "SVD should recover the planted adapter-induced activation direction."
    assert report.top_variance_fraction > 0.999, "A one-direction drift should explain nearly all uncentered drift energy."


def test_direction_ablation_removes_exact_adaptation_component(
    fn: Callable = solutions.evaluate_direction_ablation,
) -> None:
    model = ExactDirectionalOrganism()
    tokenizer = ExactMarkerTokenizer()
    examples = [
        solutions.PromptExample("marker dax", 0, "a", "toy"),
        solutions.PromptExample("marker wug", 1, "b", "toy"),
    ]
    before = solutions.evaluate_behavior(model, tokenizer, examples)
    base_hidden = t.ones(2, 2)
    direction = t.tensor([1.0, -1.0])
    after = fn(model, tokenizer, examples, layer=0, direction=direction, base_hidden=base_hidden)
    assert before.accuracy == 1.0, "The exact organism must solve both labels before intervention."
    assert after.target_probability < before.target_probability - 0.25, "Removing the exact drift direction must cause a large target-probability drop."
    assert after.target_probability == 0.5, "Projecting back to the base state should return exact chance probability."


def test_same_norm_random_lora_matches_learned_norm(
    fn: Callable = solutions.make_same_norm_random_lora,
) -> None:
    class TinyAttention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = nn.Linear(4, 4)

    class TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn = TinyAttention()

    base = TinyModel()
    random_model = fn(
        base,
        ["attn.q_proj"],
        rank=2,
        alpha=2.0,
        target_norm=3.5,
        seed=0,
    )
    delta = random_model.attn.q_proj.delta_weight()
    assert abs(delta.norm().item() - 3.5) < 1e-5, "Random low-rank control must match the learned Frobenius norm."


def test_gpu_hooks_are_strict_and_have_exact_budget_signature() -> None:
    assert str(inspect.signature(solutions.run_gpu_test)) == "(max_vram_gb: 'float' = 24.0) -> 'dict[str, Any]'", "GPU smoke hook must expose the shared 24 GiB budget contract."
    assert str(inspect.signature(solutions.run_full_experiment)) == "(max_vram_gb: 'float' = 24.0) -> 'dict[str, Any]'", "Full experiment hook must expose the shared 24 GiB budget contract."
    source = inspect.getsource(solutions._cuda_metadata)
    assert "CUDA is required" in source, "CUDA verification must fail closed with an actionable error."
    assert 't.device("cpu")' not in source and "device='cpu'" not in source, "The strict GPU hook must not contain a CPU execution branch."


def test_notebook_contract_exposes_method_and_controls() -> None:
    section = Path(__file__).resolve().parent
    exercise = (section / "15.2_LoRA_vs_Full_Finetuning_exercises.ipynb").read_text()
    solution = (section / "15.2_LoRA_vs_Full_Finetuning_solutions.ipynb").read_text()
    combined = exercise + solution
    for required in (
        "LoRALinear",
        "inject_lora",
        "freeze_except_lora",
        "merge_unmerge",
        "extract_target_update_matrix",
        "dominant_activation_drift",
        "evaluate_direction_ablation",
        "random labels",
        "same-norm random",
        "Try It Yourself",
        "Bonus Anomaly Hunt",
        "Common bug",
        "run_gpu_test",
        "run_full_experiment",
    ):
        assert required.lower() in combined.lower(), f"Learner surface is missing required concept or function: {required}."


def test_run_smoke_test_packages_exact_oracles() -> None:
    smoke = solutions.run_smoke_test()
    assert smoke["delta"] == [[6.0, 12.0], [8.0, 16.0]], "Smoke test must preserve the visible exact LoRA oracle."
    assert smoke["delta_rank"] == 1, "The exact LoRA fixture must remain rank one."
    assert smoke["merge_max_abs_diff"] < 1e-5, "Smoke evidence must include merge parity."
    assert smoke["unmerge_restoration_max_abs_diff"] < 1e-6, "Smoke evidence must include unmerge restoration."
    assert smoke["heldout_examples"] >= 20 and smoke["disjoint_templates"], "Smoke evidence must expose the held-out design."
    assert smoke["model_revision"] == solutions.MODEL_REVISION, "The pretrained transformer revision must remain immutable."
