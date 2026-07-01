"""Capstone research-sprint planning and mini-experiment utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch as t
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class CapstonePlan:
    research_question: str
    benchmark: str
    baselines: tuple[str, ...]
    mechanistic_claim: str
    causal_validations: tuple[str, ...]
    reproducible_scripts: tuple[str, ...]
    writeup_path: str


@dataclass(frozen=True)
class BaselineSuiteReport:
    required_baselines: tuple[str, ...]
    present_baselines: tuple[str, ...]
    missing_baselines: tuple[str, ...]
    complete: bool


@dataclass(frozen=True)
class CausalValidationSuiteReport:
    validations: tuple[str, ...]
    has_ablation: bool
    has_patching: bool
    has_random_control: bool
    has_ood: bool
    complete: bool


@dataclass(frozen=True)
class ReproducibilityReport:
    script_paths: tuple[str, ...]
    seeds: tuple[int, ...]
    artifact_paths: tuple[str, ...]
    reproducible: bool


@dataclass(frozen=True)
class CapstoneReadinessReport:
    has_research_question: bool
    has_benchmark: bool
    has_mechanistic_claim: bool
    baseline_suite_complete: bool
    causal_validation_complete: bool
    reproducibility_complete: bool
    has_writeup_path: bool
    ready: bool


@dataclass(frozen=True)
class ActivationOracleCapstoneConfig:
    d_model: int = 12
    n_questions: int = 4
    hidden_size: int = 64
    train_templates: tuple[int, ...] = (0, 1, 2)
    heldout_templates: tuple[int, ...] = (3, 4)
    examples_per_template: int = 48
    signal_scale: float = 2.0
    template_scale: float = 0.25
    noise_std: float = 0.08
    oracle_steps: int = 160
    text_only_steps: int = 60
    probe_steps: int = 100
    label_shuffle_steps: int = 100
    learning_rate: float = 1e-2


@dataclass(frozen=True)
class ActivationOracleCapstoneBatch:
    activations: t.Tensor
    question_ids: t.Tensor
    answer_ids: t.Tensor
    template_ids: t.Tensor
    latent_bits: t.Tensor


QUESTION_NAMES = (
    "color_bit",
    "shape_bit",
    "material_bit",
    "color_xor_shape",
)

RELEVANT_DIMS: tuple[tuple[int, ...], ...] = (
    (0,),
    (1,),
    (2,),
    (0, 1),
)


def build_capstone_plan(
    *,
    research_question: str,
    benchmark: str,
    baselines: list[str],
    mechanistic_claim: str,
    causal_validations: list[str],
    reproducible_scripts: list[str],
    writeup_path: str,
) -> CapstonePlan:
    """Bundle a paper-style capstone plan."""

    return CapstonePlan(
        research_question=research_question.strip(),
        benchmark=benchmark.strip(),
        baselines=tuple(baseline.strip() for baseline in baselines if baseline.strip()),
        mechanistic_claim=mechanistic_claim.strip(),
        causal_validations=tuple(
            validation.strip()
            for validation in causal_validations
            if validation.strip()
        ),
        reproducible_scripts=tuple(
            script.strip()
            for script in reproducible_scripts
            if script.strip()
        ),
        writeup_path=writeup_path.strip(),
    )


def baseline_suite_report(
    present_baselines: list[str],
    *,
    required_baselines: tuple[str, ...] = ("probe", "text_only", "random_control"),
) -> BaselineSuiteReport:
    """Check whether the capstone includes required baselines."""

    present = tuple(baseline.strip() for baseline in present_baselines if baseline.strip())
    present_set = set(present)
    missing = tuple(
        baseline
        for baseline in required_baselines
        if baseline not in present_set
    )
    return BaselineSuiteReport(
        required_baselines=required_baselines,
        present_baselines=present,
        missing_baselines=missing,
        complete=len(missing) == 0,
    )


def causal_validation_suite_report(
    validations: list[str],
) -> CausalValidationSuiteReport:
    """Check whether a capstone has core causal and OOD validation types."""

    normalized = tuple(validation.strip().lower() for validation in validations)
    validation_set = set(normalized)
    has_ablation = "ablation" in validation_set
    has_patching = "patching" in validation_set or "counterfactual_patching" in validation_set
    has_random_control = "random_control" in validation_set
    has_ood = "ood" in validation_set or "heldout_templates" in validation_set
    complete = has_ablation and has_patching and has_random_control and has_ood
    return CausalValidationSuiteReport(
        validations=normalized,
        has_ablation=has_ablation,
        has_patching=has_patching,
        has_random_control=has_random_control,
        has_ood=has_ood,
        complete=complete,
    )


def reproducibility_report(
    *,
    script_paths: list[str],
    seeds: list[int],
    artifact_paths: list[str],
    root: str | Path | None = None,
) -> ReproducibilityReport:
    """Check whether runnable scripts, seeds, and output artifacts exist."""

    scripts = tuple(path.strip() for path in script_paths if path.strip())
    artifacts = tuple(path.strip() for path in artifact_paths if path.strip())
    seed_tuple = tuple(int(seed) for seed in seeds)
    root_path = Path.cwd() if root is None else Path(root)
    scripts_exist = all((root_path / script).is_file() for script in scripts)
    artifacts_exist = all((root_path / artifact).is_file() for artifact in artifacts)
    return ReproducibilityReport(
        script_paths=scripts,
        seeds=seed_tuple,
        artifact_paths=artifacts,
        reproducible=bool(scripts and seed_tuple and artifacts and scripts_exist and artifacts_exist),
    )


def capstone_readiness_report(
    plan: CapstonePlan,
    baselines: BaselineSuiteReport,
    validations: CausalValidationSuiteReport,
    reproducibility: ReproducibilityReport,
) -> CapstoneReadinessReport:
    """Check whether a capstone plan satisfies the paper-style contract."""

    has_question = bool(plan.research_question)
    has_benchmark = bool(plan.benchmark)
    has_claim = bool(plan.mechanistic_claim)
    has_writeup = bool(plan.writeup_path)
    ready = (
        has_question
        and has_benchmark
        and has_claim
        and baselines.complete
        and validations.complete
        and reproducibility.reproducible
        and has_writeup
    )
    return CapstoneReadinessReport(
        has_research_question=has_question,
        has_benchmark=has_benchmark,
        has_mechanistic_claim=has_claim,
        baseline_suite_complete=baselines.complete,
        causal_validation_complete=validations.complete,
        reproducibility_complete=reproducibility.reproducible,
        has_writeup_path=has_writeup,
        ready=ready,
    )


def _cpu_generator(seed: int) -> t.Generator:
    generator = t.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return generator


def _base_latents(examples_per_template: int, *, seed: int, template_id: int) -> t.Tensor:
    combos = t.tensor(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ],
        dtype=t.long,
    )
    repeats = (examples_per_template + combos.shape[0] - 1) // combos.shape[0]
    latents = combos.repeat((repeats, 1))[:examples_per_template].clone()
    permutation = t.randperm(
        examples_per_template,
        generator=_cpu_generator(seed * 997 + template_id * 131),
    )
    return latents[permutation]


def _template_vectors(device: t.device, dtype: t.dtype) -> t.Tensor:
    return t.tensor(
        [
            [1.0, -0.5, 0.25, 0.0],
            [-0.25, 1.0, -0.5, 0.25],
            [0.5, 0.0, 1.0, -0.25],
            [-0.5, -0.25, 0.5, 1.0],
            [0.25, 0.5, -0.75, -0.5],
        ],
        device=device,
        dtype=dtype,
    )


def _answers_for_questions(latent_bits: t.Tensor, question_ids: t.Tensor) -> t.Tensor:
    color = latent_bits[:, 0]
    shape = latent_bits[:, 1]
    material = latent_bits[:, 2]
    xor = color.bitwise_xor(shape)
    answer_table = t.stack([color, shape, material, xor], dim=1)
    return answer_table.gather(1, question_ids[:, None]).squeeze(1).long()


def build_activation_oracle_capstone_batch(
    *,
    seed: int,
    template_ids: tuple[int, ...],
    config: ActivationOracleCapstoneConfig | None = None,
    device: str | t.device = "cpu",
) -> ActivationOracleCapstoneBatch:
    """Generate balanced activation questions with known latent ground truth."""

    config = config or ActivationOracleCapstoneConfig()
    device = t.device(device)
    if config.d_model < 8:
        raise ValueError("d_model must be at least 8 for latent and template dimensions.")
    if config.n_questions != len(QUESTION_NAMES):
        raise ValueError("This capstone contract expects exactly four question types.")

    base_activations = []
    base_latents = []
    base_templates = []
    template_vectors = _template_vectors(device=device, dtype=t.float32)
    noise_generator = t.Generator(device=device)
    noise_generator.manual_seed(int(seed) * 1009 + 17)
    for template_id in template_ids:
        latents_cpu = _base_latents(
            config.examples_per_template,
            seed=seed,
            template_id=template_id,
        )
        latents = latents_cpu.to(device=device)
        signs = latents.float().mul(2.0).sub(1.0)
        activations = t.zeros(
            (latents.shape[0], config.d_model),
            device=device,
            dtype=t.float32,
        )
        activations[:, :3] = config.signal_scale * signs
        activations[:, 3] = 0.2 * signs.sum(dim=1)
        activations[:, 4:8] = (
            config.template_scale
            * template_vectors[int(template_id) % template_vectors.shape[0]]
        )
        noise = t.randn(
            activations.shape,
            generator=noise_generator,
            device=device,
            dtype=activations.dtype,
        )
        activations = activations + config.noise_std * noise
        base_activations.append(activations)
        base_latents.append(latents)
        base_templates.append(
            t.full((latents.shape[0],), int(template_id), device=device, dtype=t.long)
        )

    activations = t.cat(base_activations, dim=0)
    latent_bits = t.cat(base_latents, dim=0)
    templates = t.cat(base_templates, dim=0)
    question_ids = t.arange(config.n_questions, device=device, dtype=t.long).repeat(
        activations.shape[0]
    )
    activations = activations.repeat_interleave(config.n_questions, dim=0)
    latent_bits = latent_bits.repeat_interleave(config.n_questions, dim=0)
    templates = templates.repeat_interleave(config.n_questions, dim=0)
    answer_ids = _answers_for_questions(latent_bits, question_ids)
    return ActivationOracleCapstoneBatch(
        activations=activations,
        question_ids=question_ids,
        answer_ids=answer_ids,
        template_ids=templates,
        latent_bits=latent_bits,
    )


def _question_one_hot(question_ids: t.Tensor, n_questions: int) -> t.Tensor:
    return F.one_hot(question_ids.long(), num_classes=n_questions).float()


def _oracle_inputs(batch: ActivationOracleCapstoneBatch, n_questions: int) -> t.Tensor:
    return t.cat(
        [batch.activations.float(), _question_one_hot(batch.question_ids, n_questions)],
        dim=-1,
    )


def _accuracy(logits: t.Tensor, answers: t.Tensor) -> float:
    return logits.argmax(dim=-1).eq(answers).float().mean().item()


def _train_classifier(
    model: nn.Module,
    inputs: t.Tensor,
    labels: t.Tensor,
    *,
    steps: int,
    learning_rate: float,
) -> float:
    optimizer = t.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    final_loss = t.tensor(float("nan"), device=inputs.device)
    model.train()
    for _ in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        final_loss = loss.detach()
    model.eval()
    return float(final_loss.item())


def _make_oracle_model(config: ActivationOracleCapstoneConfig, device: t.device) -> nn.Module:
    return nn.Sequential(
        nn.Linear(config.d_model + config.n_questions, config.hidden_size),
        nn.GELU(),
        nn.Linear(config.hidden_size, config.hidden_size),
        nn.GELU(),
        nn.Linear(config.hidden_size, 2),
    ).to(device)


def _train_oracle(
    train_batch: ActivationOracleCapstoneBatch,
    *,
    config: ActivationOracleCapstoneConfig,
    seed: int,
    shuffle_labels: bool = False,
) -> tuple[nn.Module, float]:
    t.manual_seed(int(seed))
    device = train_batch.activations.device
    labels = train_batch.answer_ids
    if shuffle_labels:
        permutation = t.randperm(labels.shape[0], device=device)
        labels = labels[permutation]
    model = _make_oracle_model(config, device)
    loss = _train_classifier(
        model,
        _oracle_inputs(train_batch, config.n_questions),
        labels,
        steps=config.label_shuffle_steps if shuffle_labels else config.oracle_steps,
        learning_rate=config.learning_rate,
    )
    return model, loss


def _train_text_only_baseline(
    train_batch: ActivationOracleCapstoneBatch,
    *,
    config: ActivationOracleCapstoneConfig,
    seed: int,
) -> tuple[nn.Module, float]:
    t.manual_seed(int(seed) + 11)
    device = train_batch.activations.device
    model = nn.Linear(config.n_questions, 2).to(device)
    loss = _train_classifier(
        model,
        _question_one_hot(train_batch.question_ids, config.n_questions),
        train_batch.answer_ids,
        steps=config.text_only_steps,
        learning_rate=config.learning_rate,
    )
    return model, loss


def _train_linear_probe_bank(
    train_batch: ActivationOracleCapstoneBatch,
    *,
    config: ActivationOracleCapstoneConfig,
    seed: int,
) -> tuple[dict[int, nn.Module], dict[int, float]]:
    models: dict[int, nn.Module] = {}
    losses: dict[int, float] = {}
    device = train_batch.activations.device
    for question_id in range(config.n_questions):
        t.manual_seed(int(seed) + 101 + question_id)
        mask = train_batch.question_ids.eq(question_id)
        model = nn.Linear(config.d_model, 2).to(device)
        losses[question_id] = _train_classifier(
            model,
            train_batch.activations[mask],
            train_batch.answer_ids[mask],
            steps=config.probe_steps,
            learning_rate=config.learning_rate,
        )
        models[question_id] = model.eval()
    return models, losses


def _probe_bank_logits(
    models: dict[int, nn.Module],
    batch: ActivationOracleCapstoneBatch,
    *,
    n_questions: int,
) -> t.Tensor:
    logits = t.empty((batch.answer_ids.shape[0], 2), device=batch.activations.device)
    for question_id in range(n_questions):
        mask = batch.question_ids.eq(question_id)
        logits[mask] = models[question_id](batch.activations[mask])
    return logits


def _accuracy_by_question(logits: t.Tensor, batch: ActivationOracleCapstoneBatch) -> dict[str, float]:
    return {
        QUESTION_NAMES[int(question_id)]: _accuracy(
            logits[batch.question_ids.eq(question_id)],
            batch.answer_ids[batch.question_ids.eq(question_id)],
        )
        for question_id in batch.question_ids.unique(sorted=True)
    }


def _ablate_relevant_dims(
    batch: ActivationOracleCapstoneBatch,
    *,
    config: ActivationOracleCapstoneConfig,
) -> ActivationOracleCapstoneBatch:
    activations = batch.activations.clone()
    for question_id, dims in enumerate(RELEVANT_DIMS):
        rows = batch.question_ids.eq(question_id).nonzero(as_tuple=True)[0]
        activations[rows[:, None], t.tensor(dims, device=activations.device)] = 0.0
    return ActivationOracleCapstoneBatch(
        activations=activations,
        question_ids=batch.question_ids,
        answer_ids=batch.answer_ids,
        template_ids=batch.template_ids,
        latent_bits=batch.latent_bits,
    )


def _patched_batch(
    batch: ActivationOracleCapstoneBatch,
    *,
    relevant: bool,
    seed: int,
) -> tuple[ActivationOracleCapstoneBatch, t.Tensor]:
    activations = batch.activations.clone()
    donor_answers = t.empty_like(batch.answer_ids)
    generator = _cpu_generator(seed * 1231 + (19 if relevant else 23))
    cpu_offsets = t.randint(
        low=0,
        high=max(1, batch.answer_ids.shape[0]),
        size=(batch.answer_ids.shape[0],),
        generator=generator,
    )
    control_dims = t.randint(
        low=4,
        high=batch.activations.shape[1],
        size=(batch.answer_ids.shape[0],),
        generator=generator,
    )
    for row in range(batch.answer_ids.shape[0]):
        same_question = batch.question_ids.eq(batch.question_ids[row])
        opposite_answer = batch.answer_ids.ne(batch.answer_ids[row])
        candidates = (same_question & opposite_answer).nonzero(as_tuple=True)[0]
        donor = candidates[int(cpu_offsets[row].item()) % candidates.numel()]
        donor_answers[row] = batch.answer_ids[donor]
        if relevant:
            dims = RELEVANT_DIMS[int(batch.question_ids[row].item())]
        else:
            dims = (int(control_dims[row].item()),)
        dim_index = t.tensor(dims, device=batch.activations.device)
        activations[row, dim_index] = batch.activations[donor, dim_index]
    return (
        ActivationOracleCapstoneBatch(
            activations=activations,
            question_ids=batch.question_ids,
            answer_ids=batch.answer_ids,
            template_ids=batch.template_ids,
            latent_bits=batch.latent_bits,
        ),
        donor_answers,
    )


def _failure_records(
    *,
    logits: t.Tensor,
    batch: ActivationOracleCapstoneBatch,
    split: str,
    max_records: int = 12,
) -> list[dict[str, Any]]:
    predictions = logits.argmax(dim=-1)
    failures = predictions.ne(batch.answer_ids).nonzero(as_tuple=True)[0]
    records = []
    for row in failures[:max_records]:
        row_int = int(row.item())
        records.append(
            {
                "split": split,
                "row": row_int,
                "question": QUESTION_NAMES[int(batch.question_ids[row_int].item())],
                "template_id": int(batch.template_ids[row_int].item()),
                "target": int(batch.answer_ids[row_int].item()),
                "prediction": int(predictions[row_int].item()),
                "latent_bits": [int(x) for x in batch.latent_bits[row_int].tolist()],
            }
        )
    return records


def run_activation_oracle_capstone_seed(
    *,
    seed: int,
    config: ActivationOracleCapstoneConfig | None = None,
    device: str | t.device = "cpu",
) -> dict[str, Any]:
    """Train and evaluate one seed of the 10.1 activation-oracle capstone."""

    config = config or ActivationOracleCapstoneConfig()
    device = t.device(device)
    train_batch = build_activation_oracle_capstone_batch(
        seed=seed,
        template_ids=config.train_templates,
        config=config,
        device=device,
    )
    iid_batch = build_activation_oracle_capstone_batch(
        seed=seed + 100,
        template_ids=config.train_templates,
        config=config,
        device=device,
    )
    heldout_batch = build_activation_oracle_capstone_batch(
        seed=seed + 200,
        template_ids=config.heldout_templates,
        config=config,
        device=device,
    )

    oracle, oracle_loss = _train_oracle(train_batch, config=config, seed=seed)
    text_only, text_only_loss = _train_text_only_baseline(
        train_batch,
        config=config,
        seed=seed,
    )
    probes, probe_losses = _train_linear_probe_bank(
        train_batch,
        config=config,
        seed=seed,
    )
    shuffled_oracle, shuffled_loss = _train_oracle(
        train_batch,
        config=config,
        seed=seed + 1000,
        shuffle_labels=True,
    )

    with t.no_grad():
        iid_oracle_logits = oracle(_oracle_inputs(iid_batch, config.n_questions))
        heldout_oracle_logits = oracle(_oracle_inputs(heldout_batch, config.n_questions))
        text_logits = text_only(_question_one_hot(iid_batch.question_ids, config.n_questions))
        probe_logits = _probe_bank_logits(probes, iid_batch, n_questions=config.n_questions)
        shuffled_logits = shuffled_oracle(_oracle_inputs(iid_batch, config.n_questions))

        ablated_batch = _ablate_relevant_dims(iid_batch, config=config)
        ablated_logits = oracle(_oracle_inputs(ablated_batch, config.n_questions))
        patched_batch, donor_answers = _patched_batch(iid_batch, relevant=True, seed=seed)
        patched_logits = oracle(_oracle_inputs(patched_batch, config.n_questions))
        random_patch_batch, _ = _patched_batch(iid_batch, relevant=False, seed=seed)
        random_patch_logits = oracle(_oracle_inputs(random_patch_batch, config.n_questions))
        random_activations = 0.05 * t.randn_like(iid_batch.activations)
        random_batch = ActivationOracleCapstoneBatch(
            activations=random_activations,
            question_ids=iid_batch.question_ids,
            answer_ids=iid_batch.answer_ids,
            template_ids=iid_batch.template_ids,
            latent_bits=iid_batch.latent_bits,
        )
        random_logits = oracle(_oracle_inputs(random_batch, config.n_questions))

    oracle_accuracy = _accuracy(iid_oracle_logits, iid_batch.answer_ids)
    ablated_accuracy = _accuracy(ablated_logits, iid_batch.answer_ids)
    clean_predictions = iid_oracle_logits.argmax(dim=-1)
    patched_predictions = patched_logits.argmax(dim=-1)
    random_patch_predictions = random_patch_logits.argmax(dim=-1)
    random_confidence = F.softmax(random_logits.float(), dim=-1).max(dim=-1).values.mean().item()
    oracle_by_question = _accuracy_by_question(iid_oracle_logits, iid_batch)
    probe_by_question = _accuracy_by_question(probe_logits, iid_batch)
    heldout_by_template = {
        str(int(template_id.item())): _accuracy(
            heldout_oracle_logits[heldout_batch.template_ids.eq(template_id)],
            heldout_batch.answer_ids[heldout_batch.template_ids.eq(template_id)],
        )
        for template_id in heldout_batch.template_ids.unique(sorted=True)
    }

    return {
        "seed": int(seed),
        "device": str(device),
        "train_example_count": int(train_batch.answer_ids.shape[0]),
        "iid_example_count": int(iid_batch.answer_ids.shape[0]),
        "heldout_template_example_count": int(heldout_batch.answer_ids.shape[0]),
        "oracle_final_loss": oracle_loss,
        "text_only_final_loss": text_only_loss,
        "probe_final_losses": {
            QUESTION_NAMES[question_id]: float(loss)
            for question_id, loss in probe_losses.items()
        },
        "label_shuffle_final_loss": shuffled_loss,
        "oracle_accuracy": oracle_accuracy,
        "oracle_accuracy_by_question": oracle_by_question,
        "oracle_compositional_accuracy": oracle_by_question["color_xor_shape"],
        "text_only_accuracy": _accuracy(text_logits, iid_batch.answer_ids),
        "linear_probe_bank_accuracy": _accuracy(probe_logits, iid_batch.answer_ids),
        "linear_probe_accuracy_by_question": probe_by_question,
        "linear_probe_compositional_accuracy": probe_by_question["color_xor_shape"],
        "heldout_template_accuracy": _accuracy(
            heldout_oracle_logits,
            heldout_batch.answer_ids,
        ),
        "heldout_template_accuracy_by_template": heldout_by_template,
        "ablation_accuracy": ablated_accuracy,
        "ablation_drop": oracle_accuracy - ablated_accuracy,
        "counterfactual_patch_change_rate": clean_predictions.ne(patched_predictions)
        .float()
        .mean()
        .item(),
        "counterfactual_patch_target_accuracy": patched_predictions.eq(donor_answers)
        .float()
        .mean()
        .item(),
        "random_patch_change_rate": clean_predictions.ne(random_patch_predictions)
        .float()
        .mean()
        .item(),
        "random_activation_accuracy": _accuracy(random_logits, random_batch.answer_ids),
        "random_activation_mean_confidence": random_confidence,
        "label_shuffle_accuracy": _accuracy(shuffled_logits, iid_batch.answer_ids),
        "failure_cases": _failure_records(
            logits=heldout_oracle_logits,
            batch=heldout_batch,
            split="heldout_template",
        ),
    }


def _mean(seed_reports: list[dict[str, Any]], key: str) -> float:
    return float(sum(float(report[key]) for report in seed_reports) / len(seed_reports))


def summarize_activation_oracle_capstone(
    seed_reports: list[dict[str, Any]],
    *,
    config: ActivationOracleCapstoneConfig | None = None,
    max_vram_gb: float = 24.0,
    peak_vram_gb: float = 0.0,
) -> dict[str, Any]:
    """Aggregate seed reports and apply the capstone acceptance thresholds."""

    if not seed_reports:
        raise ValueError("at least one seed report is required")
    config = config or ActivationOracleCapstoneConfig()
    summary: dict[str, Any] = {
        "benchmark": "synthetic_activation_oracle_latent_questions_v1",
        "model_family": "question_conditioned_mlp_activation_oracle",
        "dataset": "balanced_latent_bits_with_heldout_template_nuisance_v1",
        "seeds": [int(report["seed"]) for report in seed_reports],
        "seed_count": len(seed_reports),
        "d_model": config.d_model,
        "n_questions": config.n_questions,
        "question_names": list(QUESTION_NAMES),
        "train_templates": list(config.train_templates),
        "heldout_templates": list(config.heldout_templates),
        "train_example_count": int(seed_reports[0]["train_example_count"]),
        "iid_example_count": int(seed_reports[0]["iid_example_count"]),
        "heldout_template_example_count": int(
            seed_reports[0]["heldout_template_example_count"]
        ),
        "oracle_accuracy_mean": _mean(seed_reports, "oracle_accuracy"),
        "oracle_accuracy_min": min(float(report["oracle_accuracy"]) for report in seed_reports),
        "oracle_compositional_accuracy_mean": _mean(
            seed_reports,
            "oracle_compositional_accuracy",
        ),
        "text_only_accuracy_mean": _mean(seed_reports, "text_only_accuracy"),
        "linear_probe_bank_accuracy_mean": _mean(
            seed_reports,
            "linear_probe_bank_accuracy",
        ),
        "linear_probe_compositional_accuracy_mean": _mean(
            seed_reports,
            "linear_probe_compositional_accuracy",
        ),
        "heldout_template_accuracy_mean": _mean(
            seed_reports,
            "heldout_template_accuracy",
        ),
        "ablation_drop_mean": _mean(seed_reports, "ablation_drop"),
        "counterfactual_patch_change_rate_mean": _mean(
            seed_reports,
            "counterfactual_patch_change_rate",
        ),
        "counterfactual_patch_target_accuracy_mean": _mean(
            seed_reports,
            "counterfactual_patch_target_accuracy",
        ),
        "random_patch_change_rate_mean": _mean(
            seed_reports,
            "random_patch_change_rate",
        ),
        "random_activation_mean_confidence_mean": _mean(
            seed_reports,
            "random_activation_mean_confidence",
        ),
        "random_activation_accuracy_mean": _mean(
            seed_reports,
            "random_activation_accuracy",
        ),
        "label_shuffle_accuracy_mean": _mean(seed_reports, "label_shuffle_accuracy"),
        "peak_vram_gb": float(peak_vram_gb),
        "max_allowed_gpu_gb": float(max_vram_gb),
    }
    summary["oracle_beats_text_only"] = (
        summary["oracle_accuracy_mean"] >= summary["text_only_accuracy_mean"] + 0.25
    )
    summary["oracle_beats_linear_probe_bank"] = (
        summary["oracle_accuracy_mean"]
        >= summary["linear_probe_bank_accuracy_mean"] + 0.04
    )
    summary["compositional_oracle_beats_linear_probe"] = (
        summary["oracle_compositional_accuracy_mean"]
        >= summary["linear_probe_compositional_accuracy_mean"] + 0.25
    )
    summary["causal_controls_passed"] = (
        summary["ablation_drop_mean"] >= 0.20
        and summary["counterfactual_patch_change_rate_mean"] >= 0.70
        and summary["counterfactual_patch_target_accuracy_mean"] >= 0.90
        and summary["random_patch_change_rate_mean"] <= 0.15
    )
    summary["ood_passed"] = summary["heldout_template_accuracy_mean"] >= 0.90
    summary["random_activation_control_passed"] = (
        summary["random_activation_accuracy_mean"] <= 0.65
    )
    summary["label_shuffle_control_passed"] = (
        summary["label_shuffle_accuracy_mean"] <= 0.65
    )
    summary["within_vram_budget"] = summary["peak_vram_gb"] <= float(max_vram_gb)
    summary["preflight_passed"] = (
        summary["seed_count"] >= 3
        and summary["oracle_accuracy_mean"] >= 0.90
        and summary["oracle_beats_text_only"]
        and summary["oracle_beats_linear_probe_bank"]
        and summary["compositional_oracle_beats_linear_probe"]
        and summary["ood_passed"]
        and summary["causal_controls_passed"]
        and summary["random_activation_control_passed"]
        and summary["label_shuffle_control_passed"]
        and summary["within_vram_budget"]
    )
    return summary


def run_activation_oracle_capstone_experiment(
    *,
    seeds: tuple[int, ...] = (0, 1, 2),
    config: ActivationOracleCapstoneConfig | None = None,
    device: str | t.device = "cpu",
    max_vram_gb: float = 24.0,
) -> dict[str, Any]:
    """Run the full 10.1 mini capstone experiment and return JSON-safe metrics."""

    config = config or ActivationOracleCapstoneConfig()
    device = t.device(device)
    if device.type == "cuda":
        t.cuda.reset_peak_memory_stats(device)
    seed_reports = [
        run_activation_oracle_capstone_seed(
            seed=seed,
            config=config,
            device=device,
        )
        for seed in seeds
    ]
    if device.type == "cuda":
        t.cuda.synchronize(device)
        peak_vram_gb = t.cuda.max_memory_allocated(device) / 1024**3
    else:
        peak_vram_gb = 0.0
    summary = summarize_activation_oracle_capstone(
        seed_reports,
        config=config,
        max_vram_gb=max_vram_gb,
        peak_vram_gb=peak_vram_gb,
    )
    return {
        "summary": summary,
        "by_seed": seed_reports,
        "failure_cases": [
            failure
            for report in seed_reports
            for failure in report.get("failure_cases", [])
        ],
    }
