# %%


MAIN = __name__ == "__main__"

# %%

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import pandas as pd
import torch as t
from IPython.display import display
from torch import nn

chapter = "chapter15_peft_misalignment"
section = "part1_lora_dora"
root_dir = Path(__file__).resolve().parents[3]
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
asset_dir = section_dir / "media"
if MAIN:
    asset_dir.mkdir(parents=True, exist_ok=True)

if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part1_lora_dora.tests as tests
from part1_lora_dora import historical_release as reference

GT_TIER = "GT-0"
EXERCISE_ID = "15_1_lora_dora_and_adapter_controls"
DIFFICULTY = 3
IMPORTANCE = 3
EXPECTED_RUNTIME = "seconds on toy contract; minutes on local real-model path"
REQUIRES_GPU = True

# %%

@dataclass(frozen=True)
class AdapterDeltaReport:
    rank: int
    alpha: float
    update_norm: float
    nonzero_update: bool


@dataclass(frozen=True)
class DoRAWeightReport:
    target_norms: t.Tensor
    row_norms: t.Tensor
    max_norm_error: float
    norm_preserved: bool


@dataclass(frozen=True)
class IntruderDimensionReport:
    projection_fraction: float
    intruder_detected: bool


@dataclass(frozen=True)
class AdapterMechanismReport:
    accuracy_delta: float
    mechanism_delta: float
    accuracy_improved: bool
    mechanism_preserved: bool
    adapter_acceptable: bool

# %%

def lora_delta(lora_a: t.Tensor, lora_b: t.Tensor, *, alpha: float = 1.0) -> t.Tensor:
    """Return the LoRA update for A=(rank, in) and B=(out, rank)."""

    if lora_a.ndim != 2 or lora_b.ndim != 2:
        raise ValueError("lora_a and lora_b must both be matrices.")
    rank = lora_a.shape[0]
    if rank == 0:
        raise ValueError("LoRA rank must be positive.")
    if lora_b.shape[1] != rank:
        raise ValueError("lora_b second dimension must match LoRA rank.")
    return (alpha / rank) * (lora_b.float() @ lora_a.float())

# %%

if MAIN:
    tests.test_lora_delta_uses_scaled_b_matrix_times_a_matrix(lora_delta)

# %%

def adapter_delta_report(
    lora_a: t.Tensor,
    lora_b: t.Tensor,
    *,
    alpha: float = 1.0,
    min_update_norm: float = 1e-6,
) -> AdapterDeltaReport:
    """Summarize a LoRA update before trusting an adapter artifact."""

    delta = lora_delta(lora_a, lora_b, alpha=alpha)
    update_norm = float(delta.norm().item())
    return AdapterDeltaReport(
        rank=lora_a.shape[0],
        alpha=alpha,
        update_norm=update_norm,
        nonzero_update=update_norm >= min_update_norm,
    )

# %%

if MAIN:
    tests.test_adapter_delta_report_records_rank_alpha_and_nonzero_update(adapter_delta_report)
    tests.test_lora_delta_rank_bound_and_svd_spectrum(lora_delta)

# %%

def lora_merge_max_abs_diff(
    inputs: t.Tensor,
    base_weight: t.Tensor,
    lora_a: t.Tensor,
    lora_b: t.Tensor,
    *,
    alpha: float = 1.0,
) -> float:
    """Compare unmerged adapter logits to merged-weight logits."""

    if inputs.ndim != 2 or base_weight.ndim != 2:
        raise ValueError("inputs and base_weight must be matrices.")
    delta = lora_delta(lora_a, lora_b, alpha=alpha)
    if base_weight.shape != delta.shape:
        raise ValueError("base_weight and LoRA delta must have the same shape.")
    if inputs.shape[-1] != base_weight.shape[-1]:
        raise ValueError("input dimension must match base_weight input features.")
    unmerged_logits = inputs.float() @ base_weight.float().T + inputs.float() @ delta.T
    merged_logits = inputs.float() @ (base_weight.float() + delta).T
    return float((unmerged_logits - merged_logits).abs().max().item())

# %%

if MAIN:
    tests.test_lora_merge_unmerge_parity_is_independent_of_report(lora_merge_max_abs_diff)

# %%

def dora_recompose_weight(
    base_weight: t.Tensor,
    adapter_delta: t.Tensor,
    magnitude: t.Tensor,
    *,
    eps: float = 1e-8,
) -> t.Tensor:
    """Recompose a DoRA weight from an updated direction and row magnitudes."""

    if base_weight.shape != adapter_delta.shape:
        raise ValueError("base_weight and adapter_delta must match.")
    if magnitude.ndim != 1 or magnitude.shape[0] != base_weight.shape[0]:
        raise ValueError("magnitude must have shape (out_features,).")

    direction = base_weight.float() + adapter_delta.float()
    unit_direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(eps)
    return unit_direction * magnitude.float().unsqueeze(-1)


def dora_weight_report(
    base_weight: t.Tensor,
    adapter_delta: t.Tensor,
    magnitude: t.Tensor,
    *,
    max_allowed_norm_error: float = 1e-5,
) -> DoRAWeightReport:
    """Check whether DoRA recomposition preserved the target row norms."""

    recomposed = dora_recompose_weight(base_weight, adapter_delta, magnitude)
    row_norms = recomposed.norm(dim=-1)
    target_norms = magnitude.float()
    max_norm_error = float((row_norms - target_norms).abs().max().item())
    return DoRAWeightReport(
        target_norms=target_norms,
        row_norms=row_norms,
        max_norm_error=max_norm_error,
        norm_preserved=max_norm_error <= max_allowed_norm_error,
    )

# %%

if MAIN:
    tests.test_dora_recompose_weight_preserves_target_row_magnitudes(
        dora_recompose_weight,
        dora_weight_report,
    )
    tests.test_dora_recomposition_handles_nonzero_delta_direction(dora_recompose_weight)

# %%

def intruder_dimension_report(
    adapter_delta: t.Tensor,
    protected_direction: t.Tensor,
    *,
    max_projection_fraction: float = 0.2,
) -> IntruderDimensionReport:
    """Measure how much of an adapter update lies in a monitored direction."""

    if adapter_delta.ndim != 2:
        raise ValueError("adapter_delta must have shape (out_features, in_features).")
    if protected_direction.ndim != 1:
        raise ValueError("protected_direction must have shape (in_features,).")
    if adapter_delta.shape[-1] != protected_direction.shape[0]:
        raise ValueError("adapter and protected direction dimensions must match.")

    direction = protected_direction.float()
    direction = direction / direction.norm().clamp_min(1e-8)
    projected = adapter_delta.float() @ direction
    adapter_norm = float(adapter_delta.float().norm().clamp_min(1e-8).item())
    projection_fraction = float(projected.norm().item() / adapter_norm)
    return IntruderDimensionReport(
        projection_fraction=projection_fraction,
        intruder_detected=projection_fraction > max_projection_fraction,
    )


def adapter_mechanism_report(
    *,
    adapter_accuracy: float,
    baseline_accuracy: float,
    adapter_mechanism_score: float,
    baseline_mechanism_score: float,
    min_accuracy_gain: float = 0.05,
    min_mechanism_delta: float = -0.02,
) -> AdapterMechanismReport:
    """Require both task improvement and preservation of the measured mechanism."""

    accuracy_delta = adapter_accuracy - baseline_accuracy
    mechanism_delta = adapter_mechanism_score - baseline_mechanism_score
    accuracy_improved = accuracy_delta >= min_accuracy_gain
    mechanism_preserved = mechanism_delta >= min_mechanism_delta
    return AdapterMechanismReport(
        accuracy_delta=accuracy_delta,
        mechanism_delta=mechanism_delta,
        accuracy_improved=accuracy_improved,
        mechanism_preserved=mechanism_preserved,
        adapter_acceptable=accuracy_improved and mechanism_preserved,
    )

# %%

if MAIN:
    tests.test_intruder_dimension_report_measures_projection_fraction(intruder_dimension_report)
    tests.test_adapter_mechanism_report_requires_accuracy_and_mechanism(adapter_mechanism_report)

# %%

class ProxyLoRAClassifier(nn.Module):
    """Frozen distractor classifier plus a trainable LoRA adapter."""

    def __init__(self, input_dim: int = 8, rank: int = 1, alpha: float = 4.0):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive.")
        base_weight = t.zeros(2, input_dim)
        base_weight[0, 1] = 1.0
        base_weight[1, 1] = -1.0
        self.register_buffer("base_weight", base_weight)
        self.register_buffer("base_bias", t.zeros(2))
        self.lora_a = nn.Parameter(0.02 * t.randn(rank, input_dim))
        self.lora_b = nn.Parameter(t.zeros(2, rank))
        self.alpha = float(alpha)

    def delta_weight(self) -> t.Tensor:
        return lora_delta(self.lora_a, self.lora_b, alpha=self.alpha)

    def merged_weight(self) -> t.Tensor:
        return self.base_weight + self.delta_weight()

    def forward(self, inputs: t.Tensor) -> t.Tensor:
        return inputs @ self.merged_weight().T + self.base_bias


class ProxyDoRAClassifier(nn.Module):
    """Frozen distractor classifier plus a trainable DoRA adapter."""

    def __init__(self, input_dim: int = 8, rank: int = 1, alpha: float = 4.0):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive.")
        base_weight = t.zeros(2, input_dim)
        base_weight[0, 1] = 1.0
        base_weight[1, 1] = -1.0
        self.register_buffer("base_weight", base_weight)
        self.register_buffer("base_bias", t.zeros(2))
        self.lora_a = nn.Parameter(0.02 * t.randn(rank, input_dim))
        self.lora_b = nn.Parameter(t.zeros(2, rank))
        self.log_magnitude = nn.Parameter(base_weight.norm(dim=-1).clamp_min(1e-6).log())
        self.alpha = float(alpha)

    def delta_weight(self) -> t.Tensor:
        return lora_delta(self.lora_a, self.lora_b, alpha=self.alpha)

    def magnitude(self) -> t.Tensor:
        return self.log_magnitude.exp()

    def recomposed_weight(self) -> t.Tensor:
        return dora_recompose_weight(self.base_weight, self.delta_weight(), self.magnitude())

    def merged_weight(self) -> t.Tensor:
        return self.recomposed_weight()

    def forward(self, inputs: t.Tensor) -> t.Tensor:
        return inputs @ self.recomposed_weight().T + self.base_bias


class ProxyFullFinetuneClassifier(nn.Module):
    """Full linear finetune initialized from the same distractor baseline."""

    def __init__(self, input_dim: int = 8):
        super().__init__()
        base_weight = t.zeros(2, input_dim)
        base_weight[0, 1] = 1.0
        base_weight[1, 1] = -1.0
        self.register_buffer("base_weight", base_weight)
        self.register_buffer("base_bias", t.zeros(2))
        self.weight = nn.Parameter(base_weight.clone())
        self.bias = nn.Parameter(t.zeros(2))

    def delta_weight(self) -> t.Tensor:
        return self.weight - self.base_weight

    def merged_weight(self) -> t.Tensor:
        return self.weight

    def forward(self, inputs: t.Tensor) -> t.Tensor:
        return inputs @ self.weight.T + self.bias

# %%

def make_proxy_batch(
    *,
    batch: int,
    seed: int,
    device: t.device | str | None = None,
    random_labels: bool = False,
    ood_shift: float = 0.0,
    protected_shift: float = 0.0,
) -> tuple[t.Tensor, t.Tensor]:
    """Generate the safe planted-direction task used in the learner notebook."""

    if batch <= 0:
        raise ValueError("batch must be positive.")
    device = t.device("cpu") if device is None else t.device(device)
    generator = t.Generator(device=device).manual_seed(seed)
    inputs = t.randn(batch, 8, device=device, generator=generator)
    labels = (inputs[:, 0] > 0).long()
    if random_labels:
        labels = t.randint(0, 2, labels.shape, device=device, generator=generator)
    if ood_shift:
        signed_label = labels.float().mul(2).sub(1)
        inputs[:, 1] = float(ood_shift) * signed_label
    if protected_shift:
        protected_sign = t.randint(0, 2, labels.shape, device=device, generator=generator)
        inputs[:, 2] = float(protected_shift) * protected_sign.float().mul(2).sub(1)
    return inputs, labels

# %%

if MAIN:
    tests.test_proxy_batch_has_planted_target_and_ood_conflict(make_proxy_batch)

# %%

def proxy_trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _unit_basis(width: int, index: int, *, device: t.device) -> t.Tensor:
    direction = t.zeros(width, device=device)
    direction[index] = 1.0
    return direction


def _decision_direction(weight: t.Tensor) -> t.Tensor:
    if weight.ndim != 2 or weight.shape[0] != 2:
        raise ValueError("proxy classifier weight must have shape (2, input_dim).")
    return weight[1] - weight[0]


def _cosine_with_basis(vector: t.Tensor, index: int) -> float:
    basis = _unit_basis(vector.shape[0], index, device=vector.device)
    return float(nn.functional.cosine_similarity(vector.float(), basis, dim=0).item())


def _model_weight_for_proxy(model: nn.Module) -> t.Tensor:
    if hasattr(model, "recomposed_weight"):
        return model.recomposed_weight()
    if hasattr(model, "merged_weight"):
        return model.merged_weight()
    raise TypeError(f"unsupported proxy model type: {type(model)!r}")


def singular_value_spectrum(model: nn.Module) -> list[float]:
    """Return singular values of the learned adapter/update matrix."""

    weight = _model_weight_for_proxy(model)
    delta = weight - model.base_weight
    return [float(value) for value in t.linalg.svdvals(delta.float()).detach().cpu()]


def _margin_feature_correlation(
    delta_margin: t.Tensor,
    inputs: t.Tensor,
    feature: int,
) -> float:
    centered_margin = delta_margin.float() - delta_margin.float().mean()
    centered_feature = inputs[:, feature].float() - inputs[:, feature].float().mean()
    denom = centered_margin.norm() * centered_feature.norm()
    if float(denom.item()) <= 1e-8:
        return 0.0
    return float((centered_margin @ centered_feature / denom).item())

# %%

def evaluate_proxy_adapter(
    model: nn.Module,
    *,
    seed: int = 12345,
    batch: int = 2048,
    ood_shift: float = 0.0,
    protected_shift: float = 0.0,
) -> dict[str, Any]:
    """Evaluate behavior, weight direction, and activation-space drift."""

    device = next(model.parameters()).device
    inputs, labels = make_proxy_batch(
        batch=batch,
        seed=seed,
        device=device,
        ood_shift=ood_shift,
        protected_shift=protected_shift,
    )
    with t.inference_mode():
        logits = model(inputs)
        base_logits = inputs @ model.base_weight.T + model.base_bias
    predictions = logits.argmax(dim=-1)
    base_predictions = base_logits.argmax(dim=-1)
    accuracy = float(predictions.eq(labels).float().mean().item())
    baseline_accuracy = float(base_predictions.eq(labels).float().mean().item())
    weight = _model_weight_for_proxy(model)
    delta = weight - model.base_weight
    decision = _decision_direction(weight)
    base_decision = _decision_direction(model.base_weight)
    margin = logits[:, 1] - logits[:, 0]
    base_margin = base_logits[:, 1] - base_logits[:, 0]
    delta_margin = margin - base_margin
    target_cosine = _cosine_with_basis(decision, 0)
    distractor_abs_cosine = abs(_cosine_with_basis(decision, 1))
    protected_abs_cosine = abs(_cosine_with_basis(decision, 2))
    base_target_cosine = _cosine_with_basis(base_decision, 0)
    protected_direction = _unit_basis(delta.shape[-1], 2, device=device)
    protected_projection = intruder_dimension_report(
        delta,
        protected_direction,
        max_projection_fraction=0.2,
    )
    return {
        "accuracy": accuracy,
        "baseline_accuracy": baseline_accuracy,
        "accuracy_delta": accuracy - baseline_accuracy,
        "target_direction_cosine": target_cosine,
        "baseline_target_direction_cosine": base_target_cosine,
        "distractor_abs_cosine": distractor_abs_cosine,
        "protected_abs_cosine": protected_abs_cosine,
        "protected_projection_fraction": protected_projection.projection_fraction,
        "protected_direction_flagged": protected_projection.intruder_detected,
        "activation_target_corr": _margin_feature_correlation(delta_margin, inputs, 0),
        "activation_distractor_abs_corr": abs(
            _margin_feature_correlation(delta_margin, inputs, 1)
        ),
        "activation_protected_abs_corr": abs(
            _margin_feature_correlation(delta_margin, inputs, 2)
        ),
        "update_norm": float(delta.float().norm().item()),
        "update_rank": int(t.linalg.matrix_rank(delta.float(), tol=1e-4).item()),
        "singular_values": singular_value_spectrum(model),
    }


def train_proxy_adapter(
    *,
    method: Literal["lora", "dora", "full"] = "lora",
    rank: int = 1,
    alpha: float = 4.0,
    seed: int = 0,
    data_seed: int = 0,
    steps: int = 120,
    batch_size: int = 256,
    lr: float | None = None,
    random_labels: bool = False,
    device: t.device | str | None = None,
    eval_every: int = 10,
) -> tuple[nn.Module, dict[str, Any]]:
    """Train a real adapter on the planted target-direction proxy task."""

    if steps <= 0:
        raise ValueError("steps must be positive.")
    device = t.device("cpu") if device is None else t.device(device)
    t.manual_seed(seed)
    if method == "lora":
        model: nn.Module = ProxyLoRAClassifier(rank=rank, alpha=alpha).to(device)
        default_lr = 0.1
    elif method == "dora":
        model = ProxyDoRAClassifier(rank=rank, alpha=alpha).to(device)
        default_lr = 0.1
    elif method == "full":
        model = ProxyFullFinetuneClassifier().to(device)
        default_lr = 0.05
    else:
        raise ValueError("method must be 'lora', 'dora', or 'full'.")

    optimizer = t.optim.AdamW(model.parameters(), lr=default_lr if lr is None else lr, weight_decay=0.0)
    trace: dict[str, Any] = {
        "method": method,
        "rank": rank,
        "alpha": float(alpha),
        "seed": seed,
        "data_seed": data_seed,
        "random_labels": bool(random_labels),
        "step": [],
        "loss": [],
        "train_accuracy": [],
        "eval_accuracy": [],
        "target_direction_cosine": [],
        "protected_abs_cosine": [],
    }
    checkpoint_steps = set(range(eval_every, steps + 1, eval_every)) | {1, steps}

    for step in range(1, steps + 1):
        inputs, labels = make_proxy_batch(
            batch=batch_size,
            seed=data_seed * 1000 + step,
            device=device,
            random_labels=random_labels,
        )
        logits = model(inputs)
        loss = nn.functional.cross_entropy(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step in checkpoint_steps:
            with t.inference_mode():
                train_accuracy = float(logits.argmax(dim=-1).eq(labels).float().mean().item())
            eval_report = evaluate_proxy_adapter(model, seed=9000 + step, batch=1024)
            trace["step"].append(step)
            trace["loss"].append(float(loss.detach().item()))
            trace["train_accuracy"].append(train_accuracy)
            trace["eval_accuracy"].append(eval_report["accuracy"])
            trace["target_direction_cosine"].append(eval_report["target_direction_cosine"])
            trace["protected_abs_cosine"].append(eval_report["protected_abs_cosine"])

    trace["final_loss"] = trace["loss"][-1]
    trace["trainable_parameters"] = proxy_trainable_parameter_count(model)
    return model, trace

# %%

if MAIN:
    tests.test_train_proxy_adapter_returns_learning_trace_on_cpu(
        train_proxy_adapter,
        evaluate_proxy_adapter,
    )

# %%

if MAIN:
    tests.test_evaluation_on_known_weights(evaluate_proxy_adapter, ProxyFullFinetuneClassifier)

# %%

def _random_delta_metrics(
    base_weight: t.Tensor,
    random_delta: t.Tensor,
    *,
    seed: int,
    batch: int,
    ood_shift: float = 0.0,
) -> dict[str, Any]:
    device = base_weight.device
    inputs, labels = make_proxy_batch(
        batch=batch,
        seed=seed,
        device=device,
        ood_shift=ood_shift,
    )
    with t.inference_mode():
        logits = inputs @ (base_weight + random_delta).T
        base_logits = inputs @ base_weight.T
    decision = _decision_direction(base_weight + random_delta)
    delta_margin = (logits[:, 1] - logits[:, 0]) - (base_logits[:, 1] - base_logits[:, 0])
    return {
        "accuracy": float(logits.argmax(dim=-1).eq(labels).float().mean().item()),
        "target_direction_cosine": _cosine_with_basis(decision, 0),
        "distractor_abs_cosine": abs(_cosine_with_basis(decision, 1)),
        "protected_abs_cosine": abs(_cosine_with_basis(decision, 2)),
        "activation_target_corr": _margin_feature_correlation(delta_margin, inputs, 0),
        "activation_protected_abs_corr": abs(
            _margin_feature_correlation(delta_margin, inputs, 2)
        ),
        "update_rank": int(t.linalg.matrix_rank(random_delta.float(), tol=1e-4).item()),
        "update_norm": float(random_delta.float().norm().item()),
        "singular_values": [
            float(value) for value in t.linalg.svdvals(random_delta.detach().float()).cpu()
        ],
    }


def behavior_example_table(
    model: nn.Module,
    *,
    seed: int = 31415,
    rows: int = 10,
    ood_shift: float = 0.75,
) -> list[dict[str, Any]]:
    """Create example-level evidence comparing baseline and adapter predictions."""

    device = next(model.parameters()).device
    inputs, labels = make_proxy_batch(
        batch=rows,
        seed=seed,
        device=device,
        ood_shift=ood_shift,
        protected_shift=1.25,
    )
    with t.inference_mode():
        adapter_logits = model(inputs)
        base_logits = inputs @ model.base_weight.T + model.base_bias
    adapter_margin = adapter_logits[:, 1] - adapter_logits[:, 0]
    base_margin = base_logits[:, 1] - base_logits[:, 0]
    table: list[dict[str, Any]] = []
    for i in range(rows):
        table.append(
            {
                "example": i,
                "target_x0": round(float(inputs[i, 0].item()), 3),
                "distractor_x1": round(float(inputs[i, 1].item()), 3),
                "protected_x2": round(float(inputs[i, 2].item()), 3),
                "label": int(labels[i].item()),
                "baseline_pred": int(base_logits[i].argmax().item()),
                "adapter_pred": int(adapter_logits[i].argmax().item()),
                "baseline_margin": round(float(base_margin[i].item()), 3),
                "adapter_margin": round(float(adapter_margin[i].item()), 3),
            }
        )
    return table

# %%

def same_norm_random_adapter_control(
    learned_model: nn.Module,
    *,
    seed: int = 777,
    eval_seed: int = 54321,
    batch: int = 2048,
) -> dict[str, Any]:
    """Evaluate a random update scaled to the learned adapter's Frobenius norm."""

    device = next(learned_model.parameters()).device
    learned_delta = _model_weight_for_proxy(learned_model) - learned_model.base_weight
    generator = t.Generator(device=device).manual_seed(seed)
    random_delta = t.randn(learned_delta.shape, device=device, generator=generator)
    random_delta = random_delta / random_delta.norm().clamp_min(1e-8) * learned_delta.norm()
    in_distribution = _random_delta_metrics(
        learned_model.base_weight,
        random_delta,
        seed=eval_seed,
        batch=batch,
    )
    ood = _random_delta_metrics(
        learned_model.base_weight,
        random_delta,
        seed=eval_seed + 1,
        batch=batch,
        ood_shift=0.75,
    )
    return {
        "method": "same-norm random adapter",
        "is_control": True,
        "accuracy": in_distribution["accuracy"],
        "ood_accuracy": ood["accuracy"],
        "target_direction_cosine": in_distribution["target_direction_cosine"],
        "distractor_abs_cosine": in_distribution["distractor_abs_cosine"],
        "protected_abs_cosine": in_distribution["protected_abs_cosine"],
        "activation_target_corr": in_distribution["activation_target_corr"],
        "activation_protected_abs_corr": in_distribution["activation_protected_abs_corr"],
        "update_rank": in_distribution["update_rank"],
        "update_norm": in_distribution["update_norm"],
        "singular_values": in_distribution["singular_values"],
        "control_fails": in_distribution["accuracy"] <= 0.75,
    }


def compare_proxy_methods(
    *,
    rank: int = 1,
    alpha: float = 4.0,
    seed: int = 0,
    data_seed: int = 0,
    steps: int = 120,
    device: t.device | str | None = None,
) -> dict[str, Any]:
    """Train matched LoRA, DoRA, and full-finetune runs plus two controls."""

    device = t.device("cpu") if device is None else t.device(device)
    trained: dict[str, tuple[nn.Module, dict[str, Any]]] = {}
    for offset, method in enumerate(("lora", "dora", "full")):
        trained[method] = train_proxy_adapter(
            method=method, rank=rank, alpha=alpha, seed=seed + offset, data_seed=data_seed,
            steps=steps, device=device
        )
    random_label_model, random_label_trace = train_proxy_adapter(
        method="lora",
        rank=rank,
        alpha=alpha,
        seed=seed + 11,
        data_seed=data_seed,
        steps=steps,
        random_labels=True,
        device=device,
    )

    rows: list[dict[str, Any]] = []
    traces: dict[str, dict[str, Any]] = {}
    models: dict[str, nn.Module] = {}
    for method, (model, trace) in trained.items():
        eval_report = evaluate_proxy_adapter(model, seed=12345, batch=2048)
        ood_report = evaluate_proxy_adapter(model, seed=12346, batch=2048, ood_shift=0.75)
        row = {
            "method": method,
            "is_control": False,
            "trainable_parameters": trace["trainable_parameters"],
            "final_loss": trace["final_loss"],
            "accuracy": eval_report["accuracy"],
            "ood_accuracy": ood_report["accuracy"],
            "target_direction_cosine": eval_report["target_direction_cosine"],
            "distractor_abs_cosine": eval_report["distractor_abs_cosine"],
            "protected_abs_cosine": eval_report["protected_abs_cosine"],
            "activation_target_corr": eval_report["activation_target_corr"],
            "activation_protected_abs_corr": eval_report["activation_protected_abs_corr"],
            "update_rank": eval_report["update_rank"],
            "update_norm": eval_report["update_norm"],
            "singular_values": eval_report["singular_values"],
        }
        rows.append(row)
        traces[method] = trace
        models[method] = model

    random_label_eval = evaluate_proxy_adapter(random_label_model, seed=12345, batch=2048)
    random_label_ood = evaluate_proxy_adapter(
        random_label_model,
        seed=12346,
        batch=2048,
        ood_shift=0.75,
    )
    rows.append(
        {
            "method": "random-label LoRA",
            "is_control": True,
            "trainable_parameters": random_label_trace["trainable_parameters"],
            "final_loss": random_label_trace["final_loss"],
            "accuracy": random_label_eval["accuracy"],
            "ood_accuracy": random_label_ood["accuracy"],
            "target_direction_cosine": random_label_eval["target_direction_cosine"],
            "distractor_abs_cosine": random_label_eval["distractor_abs_cosine"],
            "protected_abs_cosine": random_label_eval["protected_abs_cosine"],
            "activation_target_corr": random_label_eval["activation_target_corr"],
            "activation_protected_abs_corr": random_label_eval["activation_protected_abs_corr"],
            "update_rank": random_label_eval["update_rank"],
            "update_norm": random_label_eval["update_norm"],
            "singular_values": random_label_eval["singular_values"],
            "control_fails": random_label_eval["accuracy"] <= 0.65,
        }
    )
    traces["random-label LoRA"] = random_label_trace
    random_control = same_norm_random_adapter_control(models["lora"], seed=777, eval_seed=54321)
    random_control["trainable_parameters"] = 0
    random_control["final_loss"] = float("nan")
    rows.append(random_control)

    merge_inputs, _ = make_proxy_batch(batch=512, seed=456, device=device)
    lora_model = models["lora"]
    merge_parity = lora_merge_max_abs_diff(
        merge_inputs,
        lora_model.base_weight,
        lora_model.lora_a,
        lora_model.lora_b,
        alpha=lora_model.alpha,
    )
    return {
        "rank": rank,
        "alpha": float(alpha),
        "seed": seed,
        "steps": steps,
        "rows": rows,
        "traces": traces,
        "lora_singular_values": rows[0]["singular_values"],
        "merge_max_abs_diff": merge_parity,
        "behavior_examples": behavior_example_table(models["lora"]),
    }

# %%

if MAIN:
    tests.test_same_norm_random_adapter_control_fails_on_cpu(
        train_proxy_adapter,
        same_norm_random_adapter_control,
    )

# %%

if MAIN:
    tests.test_same_norm_control_without_training(same_norm_random_adapter_control, ProxyLoRAClassifier)

# %%

def build_signature_payload(
    *,
    rank: int = 1,
    alpha: float = 4.0,
    seed: int = 0,
    steps: int = 120,
    device: t.device | str | None = None,
) -> dict[str, Any]:
    """Return the data needed for the notebook's visible signature panel."""

    comparison = compare_proxy_methods(
        rank=rank,
        alpha=alpha,
        seed=seed,
        steps=steps,
        device=device,
    )
    rows = comparison["rows"]
    learned_rows = [row for row in rows if not row["is_control"]]
    controls = [row for row in rows if row["is_control"]]
    signature_passed = (
        min(row["accuracy"] for row in learned_rows) >= 0.95
        and min(row["ood_accuracy"] for row in learned_rows) >= 0.90
        and min(row["target_direction_cosine"] for row in learned_rows) >= 0.95
        and max(row["protected_abs_cosine"] for row in learned_rows) <= 0.20
        and controls[0]["control_fails"]
        and controls[1]["control_fails"]
        and comparison["merge_max_abs_diff"] <= 1e-5
    )
    comparison["signature_passed"] = signature_passed
    comparison["claim"] = (
        "On this generated safe proxy, matched LoRA/DoRA/full-finetune adapters "
        "learn the planted target direction while random-label and same-norm random "
        "adapter controls fail."
    )
    return comparison

# %%

if MAIN:
    tests.test_signature_payload_exposes_matched_methods_and_controls(build_signature_payload)

# %%

if MAIN:
    tests.test_signature_gate_without_training(build_signature_payload)

# %%

def plot_signature_payload(payload: dict[str, Any]) -> pd.DataFrame:
    rows = payload["rows"]
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    fig.suptitle("15.1 signature result: adapter behavior must match adapter mechanism", y=0.98)

    for method in ["lora", "dora", "full", "random-label LoRA"]:
        trace = payload["traces"].get(method)
        if trace is None:
            continue
        label = method
        axes[0, 0].plot(trace["step"], trace["eval_accuracy"], marker="o", label=label)
    axes[0, 0].axhline(0.95, color="black", linestyle="--", linewidth=1, label="acceptance floor")
    axes[0, 0].set_title("Held-out accuracy during training")
    axes[0, 0].set_xlabel("step")
    axes[0, 0].set_ylabel("accuracy")
    axes[0, 0].set_ylim(0.0, 1.05)
    axes[0, 0].legend(fontsize=8)

    sv = payload["lora_singular_values"]
    axes[0, 1].bar(range(1, len(sv) + 1), sv, color=["#3b82f6", "#94a3b8"][: len(sv)])
    axes[0, 1].set_title("Learned LoRA update spectrum")
    axes[0, 1].set_xlabel("singular value index")
    axes[0, 1].set_ylabel("singular value")
    axes[0, 1].set_xticks(range(1, len(sv) + 1))

    methods = df["method"].tolist()
    colors = ["#2563eb" if not is_control else "#dc2626" for is_control in df["is_control"]]
    axes[0, 2].bar(methods, df["accuracy"], color=colors, alpha=0.85)
    axes[0, 2].bar(methods, df["ood_accuracy"], fill=False, edgecolor="black", linewidth=1.8, label="OOD")
    axes[0, 2].axhline(0.75, color="#991b1b", linestyle="--", linewidth=1, label="control ceiling")
    axes[0, 2].set_title("ID and OOD behavior")
    axes[0, 2].set_ylabel("accuracy")
    axes[0, 2].set_ylim(0.0, 1.05)
    axes[0, 2].tick_params(axis="x", rotation=35)
    axes[0, 2].legend(fontsize=8)

    x = range(len(df))
    axes[1, 0].bar([v - 0.2 for v in x], df["target_direction_cosine"], width=0.4, label="target", color="#16a34a")
    axes[1, 0].bar([v + 0.2 for v in x], df["protected_abs_cosine"], width=0.4, label="protected abs", color="#f97316")
    axes[1, 0].axhline(0.2, color="black", linestyle="--", linewidth=1)
    axes[1, 0].set_title("Weight-space direction audit")
    axes[1, 0].set_xticks(list(x), methods, rotation=35, ha="right")
    axes[1, 0].set_ylim(-0.3, 1.05)
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].bar([v - 0.2 for v in x], df["activation_target_corr"], width=0.4, label="target corr", color="#0f766e")
    axes[1, 1].bar([v + 0.2 for v in x], df["activation_protected_abs_corr"], width=0.4, label="protected abs corr", color="#b45309")
    axes[1, 1].set_title("Activation-margin drift audit")
    axes[1, 1].set_xticks(list(x), methods, rotation=35, ha="right")
    axes[1, 1].set_ylim(-1.05, 1.05)
    axes[1, 1].legend(fontsize=8)

    summary = [
        ["signature passed", str(payload["signature_passed"])],
        ["merge max abs diff", f"{payload['merge_max_abs_diff']:.2e}"],
        ["rank / alpha / steps", f"{payload['rank']} / {payload['alpha']} / {payload['steps']}"],
        ["claim", "safe planted-direction proxy only"],
    ]
    axes[1, 2].axis("off")
    table = axes[1, 2].table(cellText=summary, colLabels=["check", "value"], loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.45)
    axes[1, 2].set_title("Acceptance checks")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    output_path = asset_dir / "lora_dora_adapter_controls_signature_panel.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.show()
    print(f"Saved {output_path}")
    return df


def plot_behavior_table(payload: dict[str, Any]) -> pd.DataFrame:
    table_df = pd.DataFrame(payload["behavior_examples"])
    fig, ax = plt.subplots(figsize=(13, 3.8))
    ax.axis("off")
    visible_cols = [
        "example",
        "target_x0",
        "distractor_x1",
        "protected_x2",
        "label",
        "baseline_pred",
        "adapter_pred",
        "baseline_margin",
        "adapter_margin",
    ]
    table = ax.table(
        cellText=table_df[visible_cols].values,
        colLabels=visible_cols,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.25)
    ax.set_title("OOD examples: the frozen distractor baseline flips, the learned adapter follows x0")
    output_path = asset_dir / "lora_dora_adapter_controls_behavior_table.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.show()
    print(f"Saved {output_path}")
    return table_df


if MAIN:
    payload = build_signature_payload(rank=1, alpha=4.0, seed=0, steps=80, device="cuda")
if MAIN:
    summary_df = plot_signature_payload(payload)
if MAIN:
    behavior_df = plot_behavior_table(payload)
if MAIN:
    display(summary_df[[
        "method",
        "trainable_parameters",
        "accuracy",
        "ood_accuracy",
        "target_direction_cosine",
        "protected_abs_cosine",
        "activation_target_corr",
        "activation_protected_abs_corr",
        "update_rank",
    ]])
if MAIN:
    display(behavior_df)

# %%

if MAIN:
    PLAY_RANK = 1
if MAIN:
    PLAY_ALPHA = 4.0
if MAIN:
    PLAY_SEED = 0
if MAIN:
    PLAY_METHOD: Literal["lora", "dora", "full"] = "lora"
if MAIN:
    PLAY_STEPS = 80

if MAIN:
    play_model, play_trace = train_proxy_adapter(
        method=PLAY_METHOD,
        rank=PLAY_RANK,
        alpha=PLAY_ALPHA,
        seed=PLAY_SEED,
        steps=PLAY_STEPS,
    )
if MAIN:
    play_id = evaluate_proxy_adapter(play_model, seed=7777, batch=2048)
if MAIN:
    play_ood = evaluate_proxy_adapter(play_model, seed=7778, batch=2048, ood_shift=0.75)

if MAIN:
    plt.figure(figsize=(6, 3.5))
if MAIN:
    plt.plot(play_trace["step"], play_trace["eval_accuracy"], marker="o")
if MAIN:
    plt.ylim(0, 1.05)
if MAIN:
    plt.xlabel("step")
if MAIN:
    plt.ylabel("held-out accuracy")
if MAIN:
    plt.title(f"Play run: {PLAY_METHOD}, rank={PLAY_RANK}, alpha={PLAY_ALPHA}, seed={PLAY_SEED}")
if MAIN:
    plt.show()

if MAIN:
    print({
        "id_accuracy": round(play_id["accuracy"], 4),
        "ood_accuracy": round(play_ood["accuracy"], 4),
        "target_direction_cosine": round(play_id["target_direction_cosine"], 4),
        "protected_abs_cosine": round(play_id["protected_abs_cosine"], 4),
        "singular_values": [round(value, 4) for value in play_id["singular_values"]],
    })

# %%

if MAIN:
    PLAY_UPDATE_SCALE = 1.0
    PLAY_PROTECTED_ANGLE_DEG = 0.0
    PLAY_MAGNITUDE_SCALE = 1.0

    # No optimizer, pretrained weights or training: an exact matrix probe of our methods.
    play_base = t.tensor([[3.0, 4.0, 0.0], [0.0, 2.0, 0.0]])
    play_angle = math.radians(PLAY_PROTECTED_ANGLE_DEG)
    play_a = t.tensor([[math.cos(play_angle), 0.0, math.sin(play_angle)]])
    play_b = t.tensor([[1.0], [-1.0]])
    play_delta = lora_delta(play_a, play_b, alpha=PLAY_UPDATE_SCALE)
    play_magnitude = PLAY_MAGNITUDE_SCALE * t.tensor([10.0, 5.0])
    play_lora = play_base + play_delta
    play_dora = dora_recompose_weight(play_base, play_delta, play_magnitude)
    play_unnormalized = (play_base + play_delta) * play_magnitude[:, None]
    play_wrong_direction = t.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    play_wrong_direction = play_wrong_direction * (play_delta.norm() / math.sqrt(2.0))
    play_protected = t.tensor([0.0, 0.0, 1.0])
    play_projection = intruder_dimension_report(play_delta, play_protected)
    play_control_projection = intruder_dimension_report(play_wrong_direction, play_protected)
    t.testing.assert_close(play_dora.norm(dim=-1), play_magnitude, atol=1e-5, rtol=0)
    t.testing.assert_close(play_wrong_direction.norm(), play_delta.norm(), atol=1e-5, rtol=0)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    for label, values, offset in [
        ("LoRA", play_lora.norm(dim=-1), -.25),
        ("DoRA", play_dora.norm(dim=-1), 0),
        ("unnormalized control", play_unnormalized.norm(dim=-1), .25),
    ]:
        axes[0].bar(t.arange(2).numpy() + offset, values.numpy(), width=.25, label=label)
    axes[0].scatter([0, 1], play_magnitude.numpy(), marker="x", color="black", label="requested magnitude")
    axes[0].set(xticks=[0, 1], xlabel="output row", ylabel="row norm", title="DoRA fixes magnitude, not mechanism")
    axes[0].legend(fontsize=7)
    axes[1].bar(["chosen delta", "same-norm protected control"], [play_projection.projection_fraction, play_control_projection.projection_fraction])
    axes[1].axhline(.2, color="black", linestyle="--", label="existing protected threshold")
    axes[1].set(ylabel="protected projection fraction", ylim=(0, 1.05), title="Direction matters at a fixed update norm")
    axes[1].tick_params(axis="x", labelsize=8)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    plt.show()

# %%

if MAIN:
    verification_report_path = section_dir / "verification_report.json"
if MAIN:
    committed_evidence = json.loads(verification_report_path.read_text())
if MAIN:
    assert committed_evidence["notebook_id"] == EXERCISE_ID, (
        "The committed evidence should belong to this PEFT exercise."
    )

def lora_smoke_test() -> dict:
    lora_a = t.tensor([[1.0, 2.0]])
    lora_b = t.tensor([[3.0], [4.0]])
    delta = lora_delta(lora_a, lora_b, alpha=2.0)
    return {
        "delta": delta.tolist(),
        "report": adapter_delta_report(lora_a, lora_b, alpha=2.0).__dict__,
    }


def dora_smoke_test() -> dict:
    base_weight = t.tensor([[3.0, 4.0], [0.0, 2.0]])
    adapter_delta = t.zeros_like(base_weight)
    magnitude = t.tensor([10.0, 5.0])
    report = dora_weight_report(base_weight, adapter_delta, magnitude)
    return {
        "row_norms": [round(value, 6) for value in report.row_norms.tolist()],
        "max_norm_error": report.max_norm_error,
        "norm_preserved": report.norm_preserved,
    }


def intruder_smoke_test() -> dict:
    adapter_delta = t.tensor([[1.0, 0.0], [1.0, 0.0]])
    protected_direction = t.tensor([1.0, 0.0])
    return intruder_dimension_report(
        adapter_delta,
        protected_direction,
        max_projection_fraction=0.5,
    ).__dict__


def mechanism_smoke_test() -> dict:
    return adapter_mechanism_report(
        adapter_accuracy=0.9,
        baseline_accuracy=0.7,
        adapter_mechanism_score=0.8,
        baseline_mechanism_score=0.75,
        min_accuracy_gain=0.1,
        min_mechanism_delta=-0.02,
    ).__dict__


def run_smoke_test(cpu: bool = True) -> dict:
    _ = cpu
    return {
        "lora": lora_smoke_test(),
        "dora": dora_smoke_test(),
        "intruder": intruder_smoke_test(),
        "mechanism": mechanism_smoke_test(),
    }


def run_gpu_test(max_vram_gb: float = 24.0) -> dict:
    return reference.run_gpu_test(max_vram_gb=max_vram_gb)


def run_full_experiment(max_vram_gb: float = 24.0) -> dict:
    return reference.run_full_experiment(max_vram_gb=max_vram_gb)


if MAIN:
    tests.test_smoke_wrappers_match_the_visible_contract(
        lora_smoke_test,
        dora_smoke_test,
        intruder_smoke_test,
        mechanism_smoke_test,
    )
if MAIN:
    tests.test_notebook_contract(run_smoke_test)
if MAIN:
    tests.test_committed_verification_report_trained_peft_controls()

# %%
