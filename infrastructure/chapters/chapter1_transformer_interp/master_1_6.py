# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
```python
[
    {
        "title": "Implement and test",
        "icon": "1-circle-fill",
        "subtitle": ""
    },
    {
        "title": "Inspect and change controls",
        "icon": "2-circle-fill",
        "subtitle": ""
    },
    {
        "title": "Bonus",
        "icon": "star",
        "subtitle": ""
    }
]
```
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
# [1.6] Local Frontier ML Infrastructure (Practice)
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
# Introduction
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
Before loading a checkpoint, estimate its memory, test whether logits and token IDs agree, and preserve activation provenance. You will implement the original estimator, parity operations and disk-store methods inline. The historical signature demonstrates a small BF16 tensor readiness check; it does not validate checkpoint loading, quantization, model quality or hook equivalence.

This is a source migration, with no fresh method, numerical, model, GPU or environment execution. Saved answers and figures are historical exposed guided practice. Commit a forecast, confidence, reasoning and prior exposure before opening each reveal; this is not unseen mastery. Importing the Python module binds definitions only. Call `initialize_runtime()` explicitly before using numerical or runtime callbacks; the callbacks are not initialized merely by importing them. The notebook setup invokes this initializer when a learner elects to run it.

Original ARENA chapters 0–4 remain the canonical base. These existing 0.6 and 1.6 addenda do not replace any original section. The exact original notebook objects, metadata, outputs and bytes are in the preparation archive and inverse map. Historical claims such as `accepted` and `course_ready` are preserved historical fields, not fresh learner acceptance.

### Practice copies and intact originals

This public guided Practice exercise copy starts with empty code outputs and execution counts.
Its historical solutions retain the original observations. Foundation historical appendix code
is shown read-only, so running the active exercises does not overwrite learner callbacks with
old reference imports or stubs. Section 16.3 keeps its original teaching code executable.
All mathematical methods, controls and visible implementation work remain in the active lesson.

Open the intact original source books below. Their original directories, modules, helpers, data
locators and setup code are unchanged; open a book in its own original directory and use its
existing setup cells and declared environment. No original setup or numerical method was run
for this migration, so these are static replay paths rather than a fresh execution guarantee.
The links resolve from the generated Practice book and lesson page; exact repository paths are
shown for master/source readers.

- [Intact original: 1.6_Local_Frontier_ML_Infrastructure_exercises.ipynb](../../exercises/part6_frontier_ml_infrastructure/1.6_Local_Frontier_ML_Infrastructure_exercises.ipynb) — `chapter1_transformer_interp/exercises/part6_frontier_ml_infrastructure/1.6_Local_Frontier_ML_Infrastructure_exercises.ipynb`
- [Intact original: 1.6_Local_Frontier_ML_Infrastructure_solutions.ipynb](../../exercises/part6_frontier_ml_infrastructure/1.6_Local_Frontier_ML_Infrastructure_solutions.ipynb) — `chapter1_transformer_interp/exercises/part6_frontier_ml_infrastructure/1.6_Local_Frontier_ML_Infrastructure_solutions.ipynb`

The prior combined migration reference books also remain intact alongside this Practice pair.
They preserve historical outputs and execution metadata and are read-only provenance copies,
not a fresh execution or blind assessment.

- [Intact combined exercises reference](../../exercises/part6_frontier_ml_infrastructure_practice/1.6_Local_Frontier_ML_Infrastructure_exercises.ipynb)
- [Intact combined solutions reference](../../exercises/part6_frontier_ml_infrastructure_practice/1.6_Local_Frontier_ML_Infrastructure_solutions.ipynb)

'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Content & Learning Objectives

### 1️⃣ Implement and test

> ##### Learning Objectives
>
> - Implement each core operation beside the original immediate positive and negative tests.

### 2️⃣ Inspect and change controls

> ##### Learning Objectives
>
> - Interpret the original visible signature, qualify the claim, and predict a changed condition.

### ☆ Bonus

Inspect the retained historical contracts and complete original source materials.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Literal

MAIN = __name__ == "__main__"

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
# 1️⃣ Implement and test
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def initialize_runtime(root: str | Path | None = None) -> None:
    """Explicitly initialize runtime dependencies; never runs on raw import."""
    global t, tests, json, sys, tempfile, subprocess, math, platform, importlib
    global root_dir, chapter, section, exercises_dir, section_dir
    global F, torchvision, load_file, save_file
    import importlib.util
    import json
    import math
    import platform
    import subprocess
    import sys
    import tempfile
    import torch as t
    import torch.nn.functional as F
    import torchvision
    from safetensors.torch import load_file, save_file
    chapter = "chapter1_transformer_interp"
    section = "part6_frontier_ml_infrastructure"
    if root is None:
        if "__file__" in globals():
            root = Path(__file__).resolve().parents[3]
        else:
            root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
    root_dir = Path(root)
    exercises_dir = root_dir / chapter / "exercises"
    section_dir = exercises_dir / section
    if str(root_dir) not in sys.path:
        sys.path.append(str(root_dir))
    tests = importlib.import_module(chapter + ".exercises." + section + ".tests")


if MAIN:
    initialize_runtime()

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

DTypeName = Literal["float32", "float16", "bfloat16", "int8", "int4"]

_DTYPE_BYTES: dict[DTypeName, float] = {
    "float32": 4.0,
    "float16": 2.0,
    "bfloat16": 2.0,
    "int8": 1.0,
    "int4": 0.5,
}

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

@dataclass(frozen=True)
class MemoryBudget:
    """Approximate memory use for an inference or notebook smoke test."""

    parameter_gb: float
    kv_cache_gb: float
    activation_gb: float
    optimizer_gb: float
    overhead_gb: float

    @property
    def total_gb(self) -> float:
        return (
            self.parameter_gb
            + self.kv_cache_gb
            + self.activation_gb
            + self.optimizer_gb
            + self.overhead_gb
        )

    def fits(self, max_vram_gb: float, reserve_gb: float = 1.0) -> bool:
        return self.total_gb <= max_vram_gb - reserve_gb

    def as_dict(self) -> dict[str, float]:
        return {
            "parameter_gb": self.parameter_gb,
            "kv_cache_gb": self.kv_cache_gb,
            "activation_gb": self.activation_gb,
            "optimizer_gb": self.optimizer_gb,
            "overhead_gb": self.overhead_gb,
            "total_gb": self.total_gb,
        }

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

@dataclass(frozen=True)
class ParityReport:
    """Logit-level parity report for notebook verification blocks."""

    max_abs_diff: float
    mse: float
    kl_divergence: float
    topk_agreement: float

    def passed(
        self,
        *,
        max_abs_diff: float | None = None,
        mse: float | None = None,
        kl_divergence: float | None = None,
        topk_agreement: float | None = None,
    ) -> bool:
        if (
            max_abs_diff is None
            and mse is None
            and kl_divergence is None
            and topk_agreement is None
        ):
            raise ValueError("At least one explicit parity tolerance is required.")
        checks = []
        if max_abs_diff is not None:
            checks.append(self.max_abs_diff <= max_abs_diff)
        if mse is not None:
            checks.append(self.mse <= mse)
        if kl_divergence is not None:
            checks.append(self.kl_divergence <= kl_divergence)
        if topk_agreement is not None:
            checks.append(self.topk_agreement >= topk_agreement)
        return all(checks)

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

@dataclass(frozen=True)
class EnvironmentReport:
    """Small, serializable summary of the local ML runtime."""

    python: str
    platform: str
    torch: str
    cuda_available: bool
    cuda_version: str | None
    gpu_name: str | None
    gpu_total_memory_gb: float | None
    bf16_supported: bool
    flash_attn_available: bool
    xformers_available: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "python": self.python,
            "platform": self.platform,
            "torch": self.torch,
            "cuda_available": self.cuda_available,
            "cuda_version": self.cuda_version,
            "gpu_name": self.gpu_name,
            "gpu_total_memory_gb": self.gpu_total_memory_gb,
            "bf16_supported": self.bf16_supported,
            "flash_attn_available": self.flash_attn_available,
            "xformers_available": self.xformers_available,
        }

    def warnings(self, required_vram_gb: float | None = None) -> list[str]:
        warnings = []
        if not self.cuda_available:
            warnings.append(
                "CUDA is not available; GPU-only notebooks should run in CPU smoke-test mode."
            )
        if required_vram_gb is not None and self.gpu_total_memory_gb is not None:
            if required_vram_gb > self.gpu_total_memory_gb:
                warnings.append(
                    f"Estimated VRAM need is {required_vram_gb:.1f} GB, "
                    f"but the visible GPU has {self.gpu_total_memory_gb:.1f} GB."
                )
        if (
            required_vram_gb is not None
            and self.gpu_total_memory_gb is None
            and self.cuda_available
        ):
            warnings.append("CUDA is visible, but total GPU memory could not be read.")
        return warnings

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

@dataclass(frozen=True)
class ActivationRecord:
    """Metadata for one activation shard saved to disk."""

    name: str
    path: str
    shape: tuple[int, ...]
    dtype: str
    metadata: dict[str, Any] = field(default_factory=dict)

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Forecast before revealing the reference

Predict which runtime observations cannot be inferred from the stored CUDA version string. Record whether a warning establishes available memory.

- Forecast: ______
- Confidence: ______
- Causal reasoning: ______
- Prior exposure (seen / unseen / unknown): ______
- Observation after reveal: ______
- Changed-condition prediction: ______
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - Environment observations

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 10-20 minutes on this exercise.
> ```

The existing report observes versions, CUDA availability and optional kernels only when called. It does not upgrade packages or make a model fit.

<details>
<summary>Help - implementation and interpretation</summary>

The existing report observes versions, CUDA availability and optional kernels only when called. It does not upgrade packages or make a model fit.

</details>
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def get_environment_report() -> EnvironmentReport:
    """Return a current environment report without allocating model-sized tensors."""

    cuda_available = t.cuda.is_available()
    gpu_name = None
    gpu_total_memory_gb = None
    bf16_supported = False

    if cuda_available:
        device_index = t.cuda.current_device()
        props = t.cuda.get_device_properties(device_index)
        gpu_name = props.name
        gpu_total_memory_gb = props.total_memory / 1024**3
        bf16_supported = t.cuda.is_bf16_supported()

    return EnvironmentReport(
        python=platform.python_version(),
        platform=platform.platform(),
        torch=t.__version__,
        cuda_available=cuda_available,
        cuda_version=t.version.cuda,
        gpu_name=gpu_name,
        gpu_total_memory_gb=gpu_total_memory_gb,
        bf16_supported=bf16_supported,
        flash_attn_available=_has_module("flash_attn"),
        xformers_available=_has_module("xformers"),
    )

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def print_environment_report(required_vram_gb: float | None = None) -> EnvironmentReport:
    """Print a compact report and return it for downstream metadata logging."""

    report = get_environment_report()
    print("Environment")
    for key, value in report.as_dict().items():
        print(f"  {key}: {value}")
    for warning in report.warnings(required_vram_gb):
        print(f"  WARNING: {warning}")
    return report

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def run_environment_check(required_vram_gb: float | None = 24.0):
    # EXERCISE
    # # YOUR CODE HERE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    """Return a local environment report and print any relevant warnings."""

    report = get_environment_report()
    print("Environment report")
    for key, value in report.as_dict().items():
        print(f"  {key}: {value}")
    for warning in report.warnings(required_vram_gb):
        print(f"  WARNING: {warning}")
    return report
    # END SOLUTION

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details><summary>Solution</summary>

```python
SOLUTION
```

</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output and solution reveal</summary>

The named original tests should print `All tests ... passed!`; the drift, shape and token-mismatch controls must be rejected. The memory fixture remains about 3.66 in the original GB-labelled binary-unit estimate. The two-shard contract stores 28 values. These are retained expected contracts, not fresh results. Read the generated solution dropdowns after committing your forecast.

</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Forecast before revealing the reference

Predict which budget components double when context length doubles. Explain why a successful 1024-square matmul cannot measure a billion-parameter model peak.

- Forecast: ______
- Confidence: ______
- Causal reasoning: ______
- Prior exposure (seen / unseen / unknown): ______
- Observation after reveal: ______
- Changed-condition prediction: ______
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - Memory before model loading

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 10-20 minutes on this exercise.
> ```

Parameter bytes are P times dtype bytes. KV values are 2 times batch, context, layers, KV heads and head width. Activation memory uses the original multiplier; optimizer memory is zero for inference. Values divide by 1024 cubed, despite the original GB label.

<details>
<summary>Help - implementation and interpretation</summary>

Parameter bytes are P times dtype bytes. KV values are 2 times batch, context, layers, KV heads and head width. Activation memory uses the original multiplier; optimizer memory is zero for inference. Values divide by 1024 cubed, despite the original GB label.

</details>
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def bytes_per_value(dtype: DTypeName) -> float:
    """Return storage bytes per scalar for common model dtypes."""

    try:
        return _DTYPE_BYTES[dtype]
    except KeyError as exc:
        expected = sorted(_DTYPE_BYTES)
        raise ValueError(f"Unknown dtype {dtype!r}; expected one of {expected}") from exc

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def estimate_inference_memory(
    *,
    num_parameters: int,
    dtype: DTypeName = "bfloat16",
    batch_size: int = 1,
    context_length: int = 2048,
    hidden_size: int | None = None,
    num_layers: int | None = None,
    num_key_value_heads: int | None = None,
    head_dim: int | None = None,
    activation_multiplier: float = 2.0,
    overhead_gb: float = 1.0,
) -> MemoryBudget:
    # EXERCISE
    # # YOUR CODE HERE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    """Estimate inference memory in GB.

    This is intentionally conservative and simple. It is meant to catch obvious
    local-tier mistakes before a notebook tries to load a model, not to replace
    framework profilers.
    """

    value_bytes = bytes_per_value(dtype)
    parameter_gb = num_parameters * value_bytes / 1024**3

    kv_cache_gb = 0.0
    if num_layers is not None and num_key_value_heads is not None and head_dim is not None:
        # K and V cache, one scalar per batch/position/layer/kv-head/head-dim.
        kv_values = 2 * batch_size * context_length * num_layers * num_key_value_heads * head_dim
        kv_cache_gb = kv_values * value_bytes / 1024**3

    activation_gb = 0.0
    if hidden_size is not None:
        activation_values = batch_size * context_length * hidden_size * activation_multiplier
        activation_gb = activation_values * value_bytes / 1024**3

    return MemoryBudget(
        parameter_gb=parameter_gb,
        kv_cache_gb=kv_cache_gb,
        activation_gb=activation_gb,
        optimizer_gb=0.0,
        overhead_gb=overhead_gb,
    )
    # END SOLUTION

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details><summary>Solution</summary>

```python
SOLUTION
```

</details>
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def estimate_training_optimizer_memory(
    *,
    num_parameters: int,
    parameter_dtype: DTypeName = "bfloat16",
    optimizer_state_dtype: DTypeName = "float32",
    optimizer_states_per_parameter: int = 2,
) -> float:
    """Estimate optimizer-state memory in GB for Adam-like training."""

    param_bytes = bytes_per_value(parameter_dtype)
    state_bytes = bytes_per_value(optimizer_state_dtype)
    total_bytes = num_parameters * (param_bytes + optimizer_states_per_parameter * state_bytes)
    return total_bytes / 1024**3

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def estimate_gemma_1b_smoke_budget(context_length: int = 2048):
    # EXERCISE
    # # YOUR CODE HERE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    """Estimate memory for a Gemma-sized 1B smoke test."""

    return estimate_inference_memory(
        num_parameters=1_000_000_000,
        dtype="bfloat16",
        batch_size=1,
        context_length=context_length,
        hidden_size=2048,
        num_layers=18,
        num_key_value_heads=8,
        head_dim=256,
        overhead_gb=1.5,
    )
    # END SOLUTION

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details><summary>Solution</summary>

```python
SOLUTION
```

</details>
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

if MAIN:
    tests.test_memory_budget_fits_local_tier(estimate_gemma_1b_smoke_budget)
    tests.test_memory_budget_rejects_oversized_model()

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
The unchanged original oversized-memory test above covers the imported reference estimator. The following immediate control calls your notebook-local `estimate_inference_memory` directly using the exact original 30B BF16 fixture. Both its total estimate and `fits(24.0)` decision must reject the local tier.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

if MAIN:
    student_oversized_budget = estimate_inference_memory(
        num_parameters=30_000_000_000,
        dtype="bfloat16",
        batch_size=1,
        context_length=4096,
        hidden_size=4096,
        num_layers=48,
        num_key_value_heads=8,
        head_dim=128,
        overhead_gb=2.0,
    )
    assert student_oversized_budget.total_gb > 24.0, (
        "A 30B BF16 local load should exceed the 24GB tier in this conservative estimate."
    )
    assert not student_oversized_budget.fits(24.0), (
        "Oversized model estimates must not be reported as fitting the local GPU."
    )

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output and solution reveal</summary>

The named original tests should print `All tests ... passed!`; the drift, shape and token-mismatch controls must be rejected. The memory fixture remains about 3.66 in the original GB-labelled binary-unit estimate. The two-shard contract stores 28 values. These are retained expected contracts, not fresh results. Read the generated solution dropdowns after committing your forecast.

</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Forecast before revealing the reference

Predict whether the same top-k token set guarantees small MSE and KL. Specify the direction of KL and distinguish equal token IDs from similar decoded text.

- Forecast: ______
- Confidence: ______
- Causal reasoning: ______
- Prior exposure (seen / unseen / unknown): ______
- Observation after reveal: ______
- Changed-condition prediction: ______
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - Logit drift and exact tokens

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 10-20 minutes on this exercise.
> ```

Compute KL(reference || custom) over vocabulary then average other positions. Top-k agreement compares sorted sets, not ordered ranks. Shape mismatches fail before broadcasting; exact generation equality has a separate shape check.

<details>
<summary>Help - implementation and interpretation</summary>

Compute KL(reference || custom) over vocabulary then average other positions. Top-k agreement compares sorted sets, not ordered ranks. Shape mismatches fail before broadcasting; exact generation equality has a separate shape check.

</details>
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def _validate_logits(custom_logits: t.Tensor, reference_logits: t.Tensor) -> None:
    if custom_logits.shape != reference_logits.shape:
        raise ValueError(
            "custom_logits and reference_logits must have the same shape; "
            f"got {tuple(custom_logits.shape)} and {tuple(reference_logits.shape)}."
        )
    if custom_logits.ndim < 2:
        raise ValueError("Expected logits with at least batch and vocab dimensions.")

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def topk_agreement(custom_logits: t.Tensor, reference_logits: t.Tensor, k: int = 10) -> float:
    """Return the fraction of positions with identical top-k token sets."""

    _validate_logits(custom_logits, reference_logits)
    if k <= 0:
        raise ValueError("k must be positive.")
    if k > custom_logits.shape[-1]:
        raise ValueError("k cannot exceed vocab size.")

    custom_topk = custom_logits.topk(k, dim=-1).indices.sort(dim=-1).values
    reference_topk = reference_logits.topk(k, dim=-1).indices.sort(dim=-1).values
    return custom_topk.eq(reference_topk).all(dim=-1).float().mean().item()

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def mean_kl_divergence(custom_logits: t.Tensor, reference_logits: t.Tensor) -> float:
    """Compute mean KL(reference || custom) over all non-vocab positions."""

    _validate_logits(custom_logits, reference_logits)
    reference_log_probs = F.log_softmax(reference_logits.float(), dim=-1)
    custom_log_probs = F.log_softmax(custom_logits.float(), dim=-1)
    reference_probs = reference_log_probs.exp()
    kl_by_position = (reference_probs * (reference_log_probs - custom_log_probs)).sum(dim=-1)
    return kl_by_position.mean().item()

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def compare_logits(
    custom_logits: t.Tensor,
    reference_logits: t.Tensor,
    k: int = 10,
) -> ParityReport:
    # EXERCISE
    # # YOUR CODE HERE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    """Compute standard HF-parity metrics for two logit tensors."""

    _validate_logits(custom_logits, reference_logits)
    diff = custom_logits.float() - reference_logits.float()
    return ParityReport(
        max_abs_diff=diff.abs().max().item(),
        mse=diff.pow(2).mean().item(),
        kl_divergence=mean_kl_divergence(custom_logits, reference_logits),
        topk_agreement=topk_agreement(custom_logits, reference_logits, k=k),
    )
    # END SOLUTION

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details><summary>Solution</summary>

```python
SOLUTION
```

</details>
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def deterministic_generation_equal(
    custom_tokens: t.Tensor | list[int],
    reference_tokens: t.Tensor | list[int],
) -> bool:
    # EXERCISE
    # # YOUR CODE HERE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    """Check exact greedy-generation equality for fixed prompt/seed outputs."""

    custom = t.as_tensor(custom_tokens)
    reference = t.as_tensor(reference_tokens)
    return custom.shape == reference.shape and bool(custom.eq(reference).all().item())
    # END SOLUTION

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details><summary>Solution</summary>

```python
SOLUTION
```

</details>
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def hf_parity_smoke_test() -> bool:
    # EXERCISE
    # # YOUR CODE HERE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    """Synthetic parity check with fixed logits."""

    t.manual_seed(1)
    reference_logits = t.randn(2, 4, 30)
    custom_logits = reference_logits + 1e-6 * t.randn_like(reference_logits)
    report = compare_logits(custom_logits, reference_logits, k=5)
    print(report)
    return report.passed(
        max_abs_diff=1e-4,
        mse=1e-9,
        kl_divergence=1e-9,
        topk_agreement=1.0,
    )
    # END SOLUTION

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details><summary>Solution</summary>

```python
SOLUTION
```

</details>
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def generation_parity_smoke_test() -> bool:
    # EXERCISE
    # # YOUR CODE HERE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    """Synthetic greedy-generation parity check."""

    return deterministic_generation_equal([1, 2, 3, 4], t.tensor([1, 2, 3, 4]))
    # END SOLUTION

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details><summary>Solution</summary>

```python
SOLUTION
```

</details>
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

if MAIN:
    tests.test_compare_logits_detects_match(compare_logits)
    tests.test_compare_logits_rejects_shape_mismatch(compare_logits)
    tests.test_compare_logits_rejects_real_drift(compare_logits)
    tests.test_hf_parity_smoke_test_passes(hf_parity_smoke_test)
    tests.test_deterministic_generation_equal_detects_mismatch(deterministic_generation_equal)

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output and solution reveal</summary>

The named original tests should print `All tests ... passed!`; the drift, shape and token-mismatch controls must be rejected. The memory fixture remains about 3.66 in the original GB-labelled binary-unit estimate. The two-shard contract stores 28 values. These are retained expected contracts, not fresh results. Read the generated solution dropdowns after committing your forecast.

</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Forecast before revealing the reference

Predict what is saved when an activation requires gradients or resides on CUDA, and which fields allow a shard to be traced to its layer and hook.

- Forecast: ______
- Confidence: ______
- Causal reasoning: ______
- Prior exposure (seen / unseen / unknown): ______
- Observation after reveal: ______
- Changed-condition prediction: ______
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - Activation shards and provenance

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 10-20 minutes on this exercise.
> ```

Inspect append: detach, move to CPU, make contiguous, save one safetensors shard, append JSONL metadata, then increment the counter. This intentionally small store has no concurrent-writer, crash-atomicity or source-identity validation guarantee.

<details>
<summary>Help - implementation and interpretation</summary>

Inspect append: detach, move to CPU, make contiguous, save one safetensors shard, append JSONL metadata, then increment the counter. This intentionally small store has no concurrent-writer, crash-atomicity or source-identity validation guarantee.

</details>
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

class DiskActivationStore:
    """Append-only activation store using safetensors shards plus JSONL metadata."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.jsonl"
        self._counter = self._next_counter()

    def _next_counter(self) -> int:
        if not self.index_path.exists():
            return 0
        count = 0
        with self.index_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
        return count

    def append(
        self,
        name: str,
        activation: t.Tensor,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> ActivationRecord:
        # EXERCISE
        # # YOUR CODE HERE
        # raise NotImplementedError()
        # END EXERCISE
        # SOLUTION
        """Save one activation tensor and append a metadata row."""

        if "/" in name or "\\" in name:
            raise ValueError("Activation names must not contain path separators.")

        tensor = activation.detach().cpu().contiguous()
        filename = f"{self._counter:06d}_{name}.safetensors"
        path = self.root / filename
        save_file({"activation": tensor}, path)

        record = ActivationRecord(
            name=name,
            path=filename,
            shape=tuple(tensor.shape),
            dtype=str(tensor.dtype).replace("torch.", ""),
            metadata=metadata or {},
        )
        with self.index_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record.__dict__, sort_keys=True) + "\n")

        self._counter += 1
        return record
        # END SOLUTION

    def records(self) -> Iterator[ActivationRecord]:
        """Yield records in append order."""

        if not self.index_path.exists():
            return
        with self.index_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                data["shape"] = tuple(data["shape"])
                yield ActivationRecord(**data)

    def load(self, record: ActivationRecord | int) -> t.Tensor:
        # EXERCISE
        # # YOUR CODE HERE
        # raise NotImplementedError()
        # END EXERCISE
        # SOLUTION
        """Load an activation by record or integer index."""

        if isinstance(record, int):
            records = list(self.records())
            try:
                record = records[record]
            except IndexError as exc:
                raise IndexError(f"Activation record index {record} is out of range.") from exc

        tensors = load_file(self.root / record.path)
        return tensors["activation"]
        # END SOLUTION

    def summary(self) -> dict[str, Any]:
        """Return a compact summary useful for notebook verification footers."""

        records = list(self.records())
        total_values = sum(math.prod(record.shape) for record in records)
        return {
            "root": str(self.root),
            "num_records": len(records),
            "total_values": total_values,
            "names": [record.name for record in records],
        }

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details><summary>Solution</summary>

```python
SOLUTION
```

</details>
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def activation_store_smoke_test(output_dir: str | Path = "activation_store_smoke") -> dict:
    # EXERCISE
    # # YOUR CODE HERE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    """Write and reload two tiny activation tensors."""

    store = DiskActivationStore(output_dir)
    first = t.arange(12, dtype=t.float32).reshape(3, 4)
    second = t.eye(4)
    rec1 = store.append("resid_pre", first, metadata={"layer": 0, "hook": "resid_pre"})
    rec2 = store.append("mlp_out", second, metadata={"layer": 0, "hook": "mlp_out"})

    assert t.equal(store.load(rec1), first)
    assert t.equal(store.load(rec2), second)
    return store.summary()
    # END SOLUTION

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details><summary>Solution</summary>

```python
SOLUTION
```

</details>
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

if MAIN:
    with tempfile.TemporaryDirectory() as tmpdir:
        tests.test_disk_activation_store_roundtrip(Path(tmpdir), DiskActivationStore)
        tests.test_activation_store_smoke_test_contract(Path(tmpdir), activation_store_smoke_test)

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output and solution reveal</summary>

The named original tests should print `All tests ... passed!`; the drift, shape and token-mismatch controls must be rejected. The memory fixture remains about 3.66 in the original GB-labelled binary-unit estimate. The two-shard contract stores 28 values. These are retained expected contracts, not fresh results. Read the generated solution dropdowns after committing your forecast.

</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Forecast before revealing the reference

Predict which parts of the smoke contract are synthetic and which query the runtime; distinguish a returned dictionary from observed model correctness.

- Forecast: ______
- Confidence: ______
- Causal reasoning: ______
- Prior exposure (seen / unseen / unknown): ______
- Observation after reveal: ______
- Changed-condition prediction: ______
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - Assemble the original contract

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 10-20 minutes on this exercise.
> ```

The retained uv dependency-check function is a future explicit diagnostic. It is never called during preparation or raw import. Notebook contract tests query the environment only when the learner runs them.

<details>
<summary>Help - implementation and interpretation</summary>

The retained uv dependency-check function is a future explicit diagnostic. It is never called during preparation or raw import. Notebook contract tests query the environment only when the learner runs them.

</details>
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def uv_pip_check_report(timeout_seconds: float = 60.0) -> dict:
    """Run the uv-managed dependency consistency check."""

    try:
        completed = subprocess.run(
            ["uv", "pip", "check"],
            cwd=root_dir,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {
            "passed": False,
            "returncode": None,
            "output": type(exc).__name__,
        }

    output = "\n".join(
        line
        for line in [completed.stdout.strip(), completed.stderr.strip()]
        if line
    )
    return {
        "passed": completed.returncode == 0,
        "returncode": completed.returncode,
        "output": output,
    }

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def run_smoke_test(cpu: bool = True) -> dict:
    # EXERCISE
    # # YOUR CODE HERE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    """Contract function expected by frontier-extension notebooks."""

    _ = cpu
    budget = estimate_gemma_1b_smoke_budget(context_length=128)
    with tempfile.TemporaryDirectory() as tmpdir:
        activation_store = activation_store_smoke_test(Path(tmpdir))
    return {
        "environment": run_environment_check(required_vram_gb=budget.total_gb).as_dict(),
        "budget": budget.as_dict(),
        "hf_parity_passed": hf_parity_smoke_test(),
        "generation_parity_passed": generation_parity_smoke_test(),
        "activation_store": activation_store,
    }
    # END SOLUTION

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details><summary>Solution</summary>

```python
SOLUTION
```

</details>
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

if MAIN:
    tests.test_notebook_contract(run_smoke_test)

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output and solution reveal</summary>

The named original tests should print `All tests ... passed!`; the drift, shape and token-mismatch controls must be rejected. The memory fixture remains about 3.66 in the original GB-labelled binary-unit estimate. The two-shard contract stores 28 values. These are retained expected contracts, not fresh results. Read the generated solution dropdowns after committing your forecast.

</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
# 2️⃣ Inspect and change controls
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Forecast before revealing the reference

Before opening the stored readiness figure, predict whether the measured matmul peak should equal the model-memory estimate. State what a correct drift control establishes that a successful matmul cannot.

- Forecast: ______
- Confidence: ______
- Causal reasoning: ______
- Prior exposure (seen / unseen / unknown): ______
- Observation after reveal: ______
- Changed-condition prediction: ______
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Signature Result

<img src="../../exercises/part6_frontier_ml_infrastructure_practice/media/local_frontier_infra_signature_result.svg" width="920">

This is the unchanged historical readiness figure, with the original runtime identity. Estimated model memory and measured matmul memory have different workloads and denominators; their difference is not estimation error. The adjacent original oversized-budget, drift, token-mismatch and shard-roundtrip tests provide distinct controls. The figure has no model parity/quantization panel because those studies were not performed.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Try It Yourself

Predict the context-length sensitivity before changing the original estimator. Compare the baseline context with the doubled context; keep parameters, heads, dtype and reserve fixed. In a separate future play, choose a stricter explicit parity tolerance and forecast which original drift fixture fails. No play is executed during this preparation.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

if MAIN:
    play_short = estimate_gemma_1b_smoke_budget(context_length=2048)
    play_long = estimate_gemma_1b_smoke_budget(context_length=4096)
    print("context / parameter / KV / activation")
    for length, budget in [(2048, play_short), (4096, play_long)]:
        print(length, budget.parameter_gb, budget.kv_cache_gb, budget.activation_gb)

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
# ☆ Bonus
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Historical runtime entry points

The following original CUDA callbacks and dependency diagnostics remain available after explicit runtime initialization. They retain the original gates and failure behavior; none is executed in this migration. Reading a historical accepted report is distinct from running or accepting a new environment.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def run_gpu_test(max_vram_gb: float = 24.0) -> dict:
    """Run a real CUDA tensor check plus environment and budget reporting."""

    report = get_environment_report()
    budget = estimate_gemma_1b_smoke_budget()
    package_check = uv_pip_check_report()
    with tempfile.TemporaryDirectory() as tmpdir:
        activation_store = activation_store_smoke_test(Path(tmpdir))
    if not report.cuda_available:
        return {
            "cuda_available": False,
            "gpu_name": report.gpu_name,
            "python_version": report.python,
            "python_major_minor": ".".join(report.python.split(".")[:2]),
            "torch_version": report.torch,
            "torchvision_version": torchvision.__version__,
            "cuda_version": report.cuda_version,
            "bf16_supported": report.bf16_supported,
            "estimated_total_gb": budget.total_gb,
            "peak_vram_gb": 0.0,
            "fits_budget": False,
            "within_vram_budget": False,
            "gpu_tensor_test_passed": False,
            "uv_pip_check_passed": package_check["passed"],
            "uv_pip_check_returncode": package_check["returncode"],
            "uv_pip_check_output": package_check["output"],
            "activation_store_num_records": activation_store["num_records"],
            "activation_store_names": activation_store["names"],
            "warnings": report.warnings(budget.total_gb),
        }

    t.cuda.reset_peak_memory_stats()
    t.manual_seed(1234)
    dtype = t.bfloat16 if report.bf16_supported else t.float16
    x = t.randn(1024, 1024, device="cuda", dtype=dtype)
    y = x @ x.T
    t.cuda.synchronize()
    mean_abs = float(y.float().abs().mean().item())
    diag_mean = float(y.diagonal().float().mean().item())
    finite = bool(t.isfinite(y).all().item())
    peak_vram_gb = t.cuda.max_memory_allocated() / 1024**3
    tensor_test_passed = (
        finite
        and list(y.shape) == [1024, 1024]
        and str(dtype) == "torch.bfloat16"
        and mean_abs > 1.0
        and diag_mean > 100.0
    )
    return {
        "cuda_available": report.cuda_available,
        "gpu_name": report.gpu_name,
        "gpu_total_memory_gb": report.gpu_total_memory_gb,
        "python_version": report.python,
        "python_major_minor": ".".join(report.python.split(".")[:2]),
        "torch_version": report.torch,
        "torchvision_version": torchvision.__version__,
        "cuda_version": report.cuda_version,
        "bf16_supported": report.bf16_supported,
        "estimated_total_gb": budget.total_gb,
        "peak_vram_gb": peak_vram_gb,
        "fits_budget": budget.fits(max_vram_gb),
        "within_vram_budget": peak_vram_gb <= max_vram_gb,
        "gpu_tensor_test_passed": tensor_test_passed,
        "gpu_matmul_shape": list(y.shape),
        "gpu_matmul_dtype": str(dtype),
        "gpu_matmul_finite": finite,
        "gpu_matmul_mean_abs": mean_abs,
        "gpu_matmul_diag_mean": diag_mean,
        "uv_pip_check_passed": package_check["passed"],
        "uv_pip_check_returncode": package_check["returncode"],
        "uv_pip_check_output": package_check["output"],
        "activation_store_num_records": activation_store["num_records"],
        "activation_store_names": activation_store["names"],
        "warnings": report.warnings(budget.total_gb),
    }

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def run_full_experiment(max_vram_gb: float = 24.0) -> dict:
    """Run the validated infrastructure readiness check."""

    return run_gpu_test(max_vram_gb=max_vram_gb)

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Limitations and scope gaps

The existing syllabus also names TransformerLens/HF/nnsight same-checkpoint logit and hook equivalence, BF16/int8/int4 checkpoint loading and interpretability survival under quantization, measured parameter/KV/activation peaks and large-model feasibility. Those studies are not implemented or established by this source. The synthetic logits compare tensors, not model implementations. The original estimator applies its chosen dtype to all counted storage, so an int4 arithmetic budget does not establish real cache dtype, quantization metadata, kernels or memory use. Disk writes are single-writer and not crash-atomic. Environment versions are historical, not a current compatibility recommendation. A historical finite BF16 matmul and passed dependency check establish only their narrow readiness gate.

The human learner walkthrough remains pending. No report/schema/lock is regenerated, and no current CUDA or model readiness is claimed.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Complete original materials and provenance

- [Original ARENA course](https://github.com/callummcdougall/ARENA_3.0) remains the canonical base.
- [Original transformer implementation](https://github.com/callummcdougall/ARENA_3.0/tree/main/chapter1_transformer_interp/exercises/part1_transformer_from_scratch) provides the upstream first-principles teaching reference.
- [Complete original additive exercise notebook](https://github.com/shinobiultra/ARENA_3.0_Extended/blob/main/chapter1_transformer_interp/exercises/part6_frontier_ml_infrastructure/1.6_Local_Frontier_ML_Infrastructure_exercises.ipynb).
- [Complete original additive solutions notebook](https://github.com/shinobiultra/ARENA_3.0_Extended/blob/main/chapter1_transformer_interp/exercises/part6_frontier_ml_infrastructure/1.6_Local_Frontier_ML_Infrastructure_solutions.ipynb).

The exact pre-migration source bytes and every original cell object are retained in `before.tar.gz`, `before_manifest.json` and the section object inverse. The source preservation adapter is required to append the original historical books verbatim during adoption; this includes their original stubs, outputs, metadata and sometimes incomplete solution prose. They are preserved evidence, not a second freshly executed lesson. The authoritative master also includes their readable source in the historical material appendix below.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Original supporting panel

<img src="../../exercises/part6_frontier_ml_infrastructure_practice/media/local_frontier_infra_validation_loop.svg" width="920">

Unchanged historical explanatory artwork, not a new computation.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Complete original instruction page source

<pre>&#35; [1.6] Local Frontier ML Infrastructure

&gt; **Notebooks: [exercises](../../exercises/part6_frontier_ml_infrastructure/1.6_Local_Frontier_ML_Infrastructure_exercises.ipynb) | [solutions](../../exercises/part6_frontier_ml_infrastructure/1.6_Local_Frontier_ML_Infrastructure_solutions.ipynb)**

&gt; **Local-first extension.** This section introduces the reusable verification
&gt; harness used by the frontier-extension notebooks: environment checks, VRAM
&gt; estimates, HF parity metrics, deterministic generation parity, activation
&gt; storage, and notebook smoke-test contracts.

```python
GT_TIER = &quot;GT-1&quot;
EXERCISE_ID = &quot;1_6_local_frontier_ml_infrastructure&quot;
DIFFICULTY = 3
IMPORTANCE = 5
EXPECTED_RUNTIME = &quot;seconds for toy contracts; minutes for the local CUDA runtime report&quot;
REQUIRES_GPU = True  &#35; full acceptance uses a real CUDA BF16 tensor path
```

Please send any problems / bugs on the `&#35;errata` channel in the [Slack group](https://info-arena.github.io/ARENA_img/slack.html), and ask any questions on the dedicated channels for this chapter of material.

If you want to change to dark mode, you can do this by clicking the three horizontal lines in the top-right, then navigating to Settings -&gt; Theme.

Links to all other chapters: [(0) Fundamentals](https://arena-chapter0-fundamentals.streamlit.app/), [(1) Transformer Interpretability](https://arena-chapter1-transformer-interp.streamlit.app/), [(2) RL](https://arena-chapter2-rl.streamlit.app/).

&lt;img src=&quot;../../exercises/part6_frontier_ml_infrastructure/expected_outputs/local_frontier_infra_validation_loop.svg&quot; width=&quot;760&quot;&gt;

&#35;&#35; Core Question

How do you keep frontier interpretability notebooks honest on one local GPU?

The original ARENA rhythm is very productive: implement a component, compare it
against a trusted reference, then use it for interpretability. Frontier-extension
material needs the same rhythm with more explicit evidence. Gemma, Mamba,
diffusion language models, JEPAs, world models, SAEs, transcoders, crosscoders,
and automated circuit tools all have more moving parts than GPT-2 from scratch.
A beautiful notebook is not enough if it can silently exceed your laptop budget,
load the wrong checkpoint, cache the wrong activations, or drift from a reference
implementation.

This section is the infrastructure checkpoint before the new model-family
chapters. Every later notebook should expose:

```text
run_smoke_test(cpu=True)
run_gpu_test(max_vram_gb=24)
run_full_experiment(max_vram_gb=24)
```

The goal is not to turn ARENA into a software-engineering course. The goal is to
keep research notebooks honest. When you claim &quot;my Gemma block matches HF&quot;,
&quot;parallel and recurrent Mamba scans are equivalent&quot;, &quot;this activation cache fits
on disk&quot;, or &quot;this JEPA run stayed under 24GB&quot;, the notebook should print
evidence.

&lt;details&gt;
&lt;summary&gt;Help - why this is an interpretability lesson&lt;/summary&gt;

Most frontier interpretability failures are not deep theory failures. They are
boring infrastructure failures: wrong checkpoint revision, missing CUDA, silent
CPU fallback, activation files without metadata, or a pretty result whose peak
VRAM was never measured. This section teaches the contract later notebooks must
meet before their interpretability claims are worth reading.

&lt;/details&gt;

Durable repo references:

- [`docs/local_gpu_setup.md`](../../../docs/local_gpu_setup.md)
- [`docs/reproducibility_contract.md`](../../../docs/reproducibility_contract.md)

&#35;&#35; Learning Objectives

- Print enough environment metadata to reproduce a local result.
- Estimate parameter, KV-cache, activation, optimizer, and overhead memory.
- Use logit parity and deterministic generation parity as implementation checks.
- Save and reload activations with metadata using disk-backed shards.
- Separate measured CUDA peak VRAM from rough preflight estimates.

&#35; Setup Code

```python
from pathlib import Path
import sys
import tempfile

import torch as t

chapter = &quot;chapter1_transformer_interp&quot;
section = &quot;part6_frontier_ml_infrastructure&quot;
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
exercises_dir = root_dir / chapter / &quot;exercises&quot;
section_dir = exercises_dir / section

if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part6_frontier_ml_infrastructure.tests as tests
import part6_frontier_ml_infrastructure.utils as utils

from arena_ext import (
    DiskActivationStore,
    compare_logits,
    deterministic_generation_equal,
    estimate_inference_memory,
    get_environment_report,
)
```

&#35; Environment Checks

&#35;&#35;&#35; Exercise - inspect your local runtime

&gt; Difficulty: easy
&gt; Importance: high
&gt;
&gt; You should spend 5-10 minutes on this exercise.

Call `get_environment_report()`, print its fields, and inspect warnings for a
required VRAM estimate.

```python
report = get_environment_report()
utils.print_dict_table(&quot;Environment&quot;, report.as_dict())

for warning in report.warnings(required_vram_gb=24.0):
    print(&quot;WARNING:&quot;, warning)
```

&lt;details&gt;
&lt;summary&gt;Expected output&lt;/summary&gt;

Your exact platform string and optional kernel availability may differ, but the
local acceptance path expects:

```text
cuda_available: True
cuda_version: 13.2
torch: 2.12.1+cu132
gpu_name: NVIDIA GeForce RTX 5090 Laptop GPU
bf16_supported: True
```

&lt;/details&gt;

&lt;details&gt;
&lt;summary&gt;Common bugs&lt;/summary&gt;

- Reporting only `torch.__version__` and forgetting CUDA/device metadata.
- Treating a warning as an exception for CPU-only smoke tests.
- Hiding missing CUDA behind a CPU fallback in a GPU-required notebook.

&lt;/details&gt;

&lt;details&gt;
&lt;summary&gt;Solution&lt;/summary&gt;

```python
def run_environment_check(required_vram_gb: float | None = 24.0):
    report = get_environment_report()
    utils.print_dict_table(&quot;Environment&quot;, report.as_dict())
    for warning in report.warnings(required_vram_gb):
        print(&quot;WARNING:&quot;, warning)
    return report
```

&lt;/details&gt;

&#35; VRAM Budget Estimates

&#35;&#35;&#35; Exercise - estimate a Gemma-sized smoke test

&gt; Difficulty: medium
&gt; Importance: high
&gt;
&gt; You should spend 10-15 minutes on this exercise.

Implement a rough local-tier memory estimate for a 1B-parameter BF16 model with a
KV cache. This is not a profiler; it is a preflight guardrail.

```python
def estimate_gemma_1b_smoke_budget(context_length: int = 2048):
    raise NotImplementedError()


budget = estimate_gemma_1b_smoke_budget()
utils.print_dict_table(&quot;Estimated memory&quot;, budget.as_dict())
print(&quot;Fits 24GB local tier:&quot;, budget.fits(24.0))

tests.test_memory_budget_fits_local_tier(estimate_gemma_1b_smoke_budget)
```

&lt;details&gt;
&lt;summary&gt;Expected output&lt;/summary&gt;

```text
total_gb: about 3.66
Fits 24GB local tier: True
All tests in `test_memory_budget_fits_local_tier` passed!
```

&lt;/details&gt;

&lt;details&gt;
&lt;summary&gt;Common bugs&lt;/summary&gt;

- Forgetting the KV cache and underestimating long-context inference.
- Counting optimizer memory for inference-only checks.
- Treating the estimate as measured peak VRAM.

&lt;/details&gt;

&lt;details&gt;
&lt;summary&gt;Solution&lt;/summary&gt;

```python
def estimate_gemma_1b_smoke_budget(context_length: int = 2048):
    return estimate_inference_memory(
        num_parameters=1_000_000_000,
        dtype=&quot;bfloat16&quot;,
        batch_size=1,
        context_length=context_length,
        hidden_size=2048,
        num_layers=18,
        num_key_value_heads=8,
        head_dim=256,
        overhead_gb=1.5,
    )
```

&lt;/details&gt;

&#35; HF Parity Checks

&#35;&#35;&#35; Exercise - compare logits

&gt; Difficulty: medium
&gt; Importance: high
&gt;
&gt; You should spend 10-15 minutes on this exercise.

For a real model-family notebook, `custom_logits` would come from your
implementation and `reference_logits` would come from Hugging Face or the
official repository. Here we use deterministic logits so the test is fast and
exact.

```python
t.manual_seed(1)
reference_logits = t.randn(2, 4, 30)
custom_logits = reference_logits + 1e-6 * t.randn_like(reference_logits)

parity = compare_logits(custom_logits, reference_logits, k=5)
utils.print_dict_table(&quot;Parity report&quot;, parity.__dict__)

assert parity.passed(
    max_abs_diff=1e-4,
    mse=1e-10,
    kl_divergence=1e-10,
    topk_agreement=1.0,
)

tests.test_compare_logits_detects_match(compare_logits)
tests.test_compare_logits_rejects_shape_mismatch(compare_logits)
```

&lt;details&gt;
&lt;summary&gt;Expected output&lt;/summary&gt;

```text
topk_agreement: 1.0
max_abs_diff: &lt; 1e-4
All tests in `test_compare_logits_detects_match` passed!
All tests in `test_compare_logits_rejects_shape_mismatch` passed!
```

&lt;/details&gt;

&lt;details&gt;
&lt;summary&gt;Common bugs&lt;/summary&gt;

- Comparing only argmax tokens and missing distribution drift.
- Forgetting shape checks before computing parity metrics.
- Using stochastic generation parity when fixed-logit parity is the right test.

&lt;/details&gt;

&lt;details&gt;
&lt;summary&gt;Solution&lt;/summary&gt;

```python
def hf_parity_smoke_test() -&gt; bool:
    t.manual_seed(1)
    reference_logits = t.randn(2, 4, 30)
    custom_logits = reference_logits + 1e-6 * t.randn_like(reference_logits)
    report = compare_logits(custom_logits, reference_logits, k=5)
    return report.passed(
        max_abs_diff=1e-4,
        mse=1e-9,
        kl_divergence=1e-9,
        topk_agreement=1.0,
    )


tests.test_hf_parity_smoke_test_passes(hf_parity_smoke_test)
```

&lt;/details&gt;

&#35;&#35;&#35; Exercise - check deterministic generation

&gt; Difficulty: easy
&gt; Importance: medium
&gt;
&gt; You should spend 5 minutes on this exercise.

For greedy decoding at a fixed seed, exact token equality is often the simplest
generation-parity check.

```python
assert deterministic_generation_equal([1, 2, 3, 4], t.tensor([1, 2, 3, 4]))
assert not deterministic_generation_equal([1, 2, 3, 4], t.tensor([1, 2, 3, 5]))

tests.test_deterministic_generation_equal_detects_mismatch(
    deterministic_generation_equal,
)
```

&lt;details&gt;
&lt;summary&gt;Expected output&lt;/summary&gt;

```text
All tests in `test_deterministic_generation_equal_detects_mismatch` passed!
```

&lt;/details&gt;

&lt;details&gt;
&lt;summary&gt;Common bugs&lt;/summary&gt;

- Comparing decoded strings after tokenizer cleanup rather than token IDs.
- Expecting exact equality from sampling with temperature.
- Forgetting to record the seed, prompt, tokenizer, and checkpoint revision.

&lt;/details&gt;

&lt;details&gt;
&lt;summary&gt;Solution&lt;/summary&gt;

```python
def generation_parity_smoke_test() -&gt; bool:
    return deterministic_generation_equal([1, 2, 3, 4], t.tensor([1, 2, 3, 4]))
```

&lt;/details&gt;

&#35; Activation Storage

&#35;&#35;&#35; Exercise - write and reload activation shards

&gt; Difficulty: medium
&gt; Importance: high
&gt;
&gt; You should spend 10-15 minutes on this exercise.

Use `DiskActivationStore` to save activations without keeping the full dataset in
memory. The store writes one `safetensors` file per shard and appends metadata
to `index.jsonl`.

```python
store = DiskActivationStore(section_dir / &quot;activation_store_smoke&quot;)

activation = t.arange(12, dtype=t.float32).reshape(3, 4)
record = store.append(
    &quot;blocks_0_resid_pre&quot;,
    activation,
    metadata={&quot;layer&quot;: 0, &quot;hook&quot;: &quot;resid_pre&quot;, &quot;dataset&quot;: &quot;synthetic&quot;},
)

loaded = store.load(record)
assert t.equal(loaded, activation)

utils.print_dict_table(&quot;Activation store&quot;, store.summary())
with tempfile.TemporaryDirectory() as tmpdir:
    tests.test_disk_activation_store_roundtrip(Path(tmpdir))
```

&lt;details&gt;
&lt;summary&gt;Expected output&lt;/summary&gt;

```text
num_records: 1
names: [&&#35;x27;blocks_0_resid_pre&&#35;x27;]
All tests in `test_disk_activation_store_roundtrip` passed!
```

&lt;/details&gt;

&lt;details&gt;
&lt;summary&gt;Common bugs&lt;/summary&gt;

- Saving tensors without detaching them from autograd graphs.
- Forgetting metadata needed to recover hook, layer, prompt set, or run.
- Keeping all activations in GPU memory instead of streaming them to disk.

&lt;/details&gt;

&lt;details&gt;
&lt;summary&gt;Solution&lt;/summary&gt;

```python
def activation_store_smoke_test(output_dir: str | Path = &quot;activation_store_smoke&quot;) -&gt; dict:
    store = DiskActivationStore(output_dir)
    first = t.arange(12, dtype=t.float32).reshape(3, 4)
    second = t.eye(4)
    rec1 = store.append(&quot;resid_pre&quot;, first, metadata={&quot;layer&quot;: 0, &quot;hook&quot;: &quot;resid_pre&quot;})
    rec2 = store.append(&quot;mlp_out&quot;, second, metadata={&quot;layer&quot;: 0, &quot;hook&quot;: &quot;mlp_out&quot;})
    assert t.equal(store.load(rec1), first)
    assert t.equal(store.load(rec2), second)
    return store.summary()
```

&lt;/details&gt;

&#35; Notebook Contract

Every frontier-extension notebook should expose these three entry points:

```python
def run_smoke_test(cpu: bool = True) -&gt; dict:
    ...

def run_gpu_test(max_vram_gb: float = 24.0) -&gt; dict:
    ...

def run_full_experiment(max_vram_gb: float = 24.0) -&gt; dict:
    ...
```

The smoke test should avoid downloads and API calls where possible. The GPU test
is not a CPU fallback: it should run a real CUDA path, print measured peak VRAM,
and stay under the configured local tier.

```python
def run_smoke_test(cpu: bool = True) -&gt; dict:
    raise NotImplementedError()


tests.test_notebook_contract(run_smoke_test)
```

&lt;details&gt;
&lt;summary&gt;Expected output&lt;/summary&gt;

```text
All tests in `test_notebook_contract` passed!
```

&lt;/details&gt;

&lt;details&gt;
&lt;summary&gt;Common bugs&lt;/summary&gt;

- Returning only `True` instead of a dictionary with reproducibility metadata.
- Reporting a memory estimate as if it were measured peak CUDA allocation.
- Accepting a GPU-required notebook when CUDA never executed.

&lt;/details&gt;

&lt;details&gt;
&lt;summary&gt;Solution&lt;/summary&gt;

```python
def run_smoke_test(cpu: bool = True) -&gt; dict:
    _ = cpu
    budget = estimate_gemma_1b_smoke_budget(context_length=128)
    with tempfile.TemporaryDirectory() as tmpdir:
        activation_store = activation_store_smoke_test(Path(tmpdir))
    return {
        &quot;environment&quot;: run_environment_check(required_vram_gb=budget.total_gb).as_dict(),
        &quot;budget&quot;: budget.as_dict(),
        &quot;hf_parity_passed&quot;: hf_parity_smoke_test(),
        &quot;generation_parity_passed&quot;: generation_parity_smoke_test(),
        &quot;activation_store&quot;: activation_store,
    }
```

&lt;/details&gt;

&#35; CUDA Verification Path

The committed report for this section uses a real CUDA BF16 tensor path, not a
CPU fallback. It verifies:

```text
python_major_minor: 3.14
torch_version: 2.12.1+cu132
torchvision_version: 0.27.1+cu132
cuda_version: 13.2
bf16_supported: true
gpu_matmul_shape: [1024, 1024]
gpu_matmul_dtype: torch.bfloat16
gpu_matmul_finite: true
gpu_tensor_test_passed: true
uv_pip_check_passed: true
estimated_total_gb: &lt; 4.0
peak_vram_gb: &lt; 1.0
within_vram_budget: true
```

For later architecture notebooks, the pass criteria should look like this:

```text
environment report printed
estimated memory printed before model load
real CUDA tensor or model path executed
HF logits match tolerance
generation matches at fixed seed where deterministic
activation cache has expected shape and metadata
peak VRAM printed after the run
artifacts saved with checkpoint, dtype, seed, dataset, and command metadata
```

&#35; Signature Result

&lt;img src=&quot;../../exercises/part6_frontier_ml_infrastructure/expected_outputs/local_frontier_infra_signature_result.svg&quot; width=&quot;820&quot;&gt;

The committed CUDA report proves the local infrastructure gate, not a model
quality benchmark:

| Check | Observed | Required |
|---|---:|---:|
| Python / PyTorch | `3.14.6` / `2.12.1+cu132` | `3.14` / `2.12.1+cu132` |
| CUDA runtime | `13.2` | `13.2` |
| BF16 CUDA matmul | `[1024, 1024]`, finite | finite BF16 tensor |
| `uv pip check` | passed, return code `0` | pass |
| Gemma-sized estimate | `3.6595 GB` | `&lt; 4.0 GB` |
| Measured peak VRAM | `0.0430 GB` | `&lt; 1.0 GB` |

&lt;details&gt;
&lt;summary&gt;Interpreting this result&lt;/summary&gt;

This is a readiness gate. It says the managed uv environment, PyTorch CUDA 13.2
runtime, BF16 CUDA tensor path, dependency set, memory estimator, and activation
store contract are working. It does not say a future Gemma, Mamba, SAE, or VLM
result is correct; those sections still need their own toy oracles, reference
parity, and negative controls.

&lt;/details&gt;

&#35; Limitations

- This section does not load a real Hugging Face checkpoint. Later architecture
  sections must add checkpoint-specific tokenizer, logit, generation, and cache
  parity.
- The Gemma-sized memory estimate is a conservative preflight, not measured
  peak VRAM.
- The CUDA signature result is a deterministic BF16 tensor gate, not a quality
  benchmark for any model.
- CPU smoke tests are allowed for notebook ergonomics, but they are not an
  acceptance path for GPU-required sections.

&#35; Bonus - Anomaly Hunting

This infrastructure is deliberately small. Later sections should extend it rather
than bypass it:

- Gemma notebooks should add tokenizer/chat-template parity, KV-cache parity, and quantized inference checks.
- Mamba notebooks should add parallel-vs-recurrent selective-scan equivalence.
- Diffusion language-model notebooks should add noising-distribution tests and denoising-step activation shapes.
- SAE and transcoder notebooks should add L0, feature density, dead-feature, loss-recovered, reconstruction-KL, steering, and ablation metrics.
- JEPA and world-model notebooks should add masking tests, collapse diagnostics, latent-probe baselines, rollout tests, and causal patching checks.
</pre>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Historical material appendix

These complete original source cells are preserved for inspection. The front curriculum teaches the methods inline; do not run this appendix as an alternative fresh validation path.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Original exercises notebook: complete cell sources
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
**Original exercises cell 0: markdown.**

<pre>&#35; [1.6] Local Frontier ML Infrastructure - Exercises

**Core Question.** How do you keep frontier interpretability notebooks honest on one local GPU?

This notebook builds the contract later extension sections depend on: environment metadata, memory estimates, parity checks, deterministic generation checks, disk-backed activation storage, and a real CUDA report.

&#35;&#35; Learning Objectives

- Print reproducible Python, PyTorch, CUDA, and GPU metadata.
- Separate memory estimates from measured peak CUDA allocation.
- Detect both logit parity matches and real logit drift.
- Save activations with metadata and reload them from disk.
- Read the committed CUDA report without turning it into a model-quality claim.

&gt; ```yaml
&gt; Difficulty: 3
&gt; Importance: 5
&gt; ```

&lt;details&gt;
&lt;summary&gt;Help - why this matters&lt;/summary&gt;

A later interpretability result is only credible if the notebook can show which environment ran it, which checkpoint or tensor path executed, how much memory it used, and which controls failed.

&lt;/details&gt;</pre>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
**Original exercises cell 1: code.**

<pre>GT_TIER = &quot;GT-1&quot;
EXERCISE_ID = &quot;1_6_local_frontier_ml_infrastructure&quot;
DIFFICULTY = 3
IMPORTANCE = 5
EXPECTED_RUNTIME = &quot;seconds for toy contracts; minutes for the local CUDA runtime report&quot;
REQUIRES_GPU = True

import json
import sys
import tempfile
from pathlib import Path

import torch as t

chapter = &quot;chapter1_transformer_interp&quot;
section = &quot;part6_frontier_ml_infrastructure&quot;
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
exercises_dir = root_dir / chapter / &quot;exercises&quot;
section_dir = exercises_dir / section

if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part6_frontier_ml_infrastructure.tests as tests
import part6_frontier_ml_infrastructure.utils as utils

from arena_ext import (
    DiskActivationStore,
    compare_logits,
    deterministic_generation_equal,
    estimate_inference_memory,
    get_environment_report,
)

MAIN = True</pre>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
**Original exercises cell 2: markdown.**

<pre>&#35;&#35; Exercise 1 - Environment Checks

Print the local runtime report and warnings for a required VRAM estimate.

&lt;details&gt;
&lt;summary&gt;Expected output&lt;/summary&gt;

```text
cuda_available: True
cuda_version: 13.2
torch: 2.12.1+cu132
bf16_supported: True
```

&lt;/details&gt;

&lt;details&gt;
&lt;summary&gt;Help - what not to hide&lt;/summary&gt;

Do not silently accept a GPU-required notebook when CUDA is missing. CPU smoke mode is useful, but it is not the acceptance path.

&lt;/details&gt;

&lt;details&gt;
&lt;summary&gt;Solution&lt;/summary&gt;

Call `get_environment_report()`, print `report.as_dict()`, print warnings, and return the report object.

&lt;/details&gt;</pre>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
**Original exercises cell 3: code.**

<pre>def run_environment_check(required_vram_gb: float | None = 24.0):
    &#35; EXERCISE
    &#35; YOUR CODE HERE
    raise NotImplementedError()
    &#35; END EXERCISE


report = run_environment_check(required_vram_gb=24.0)
report.as_dict()</pre>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
**Original exercises cell 4: markdown.**

<pre>&#35;&#35; Exercise 2 - VRAM Budget Estimates

Estimate a 1B-parameter BF16 smoke-test model with KV-cache and activation memory.

&lt;details&gt;
&lt;summary&gt;Expected output&lt;/summary&gt;

```text
total_gb: about 3.66
Fits 24GB local tier: True
All tests in `test_memory_budget_fits_local_tier` passed!
```

&lt;/details&gt;

&lt;details&gt;
&lt;summary&gt;Help - estimate versus measurement&lt;/summary&gt;

A budget estimate catches obvious local-tier mistakes before model load. It is not a replacement for `torch.cuda.max_memory_allocated()` after a real run.

&lt;/details&gt;

&lt;details&gt;
&lt;summary&gt;Solution&lt;/summary&gt;

Use `estimate_inference_memory` with 1B BF16 parameters, batch size 1, context 2048, hidden size 2048, 18 layers, 8 KV heads, head dim 256, and 1.5GB overhead.

&lt;/details&gt;</pre>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
**Original exercises cell 5: code.**

<pre>def estimate_gemma_1b_smoke_budget(context_length: int = 2048):
    &#35; EXERCISE
    &#35; YOUR CODE HERE
    raise NotImplementedError()
    &#35; END EXERCISE


tests.test_memory_budget_fits_local_tier(estimate_gemma_1b_smoke_budget)
tests.test_memory_budget_rejects_oversized_model()</pre>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
**Original exercises cell 6: markdown.**

<pre>&#35;&#35; Exercise 3 - HF Parity and Drift Controls

Compute logit-level parity metrics and wrap them in a passing smoke test.

&lt;details&gt;
&lt;summary&gt;Expected output&lt;/summary&gt;

```text
All tests in `test_compare_logits_detects_match` passed!
All tests in `test_compare_logits_rejects_real_drift` passed!
```

&lt;/details&gt;

&lt;details&gt;
&lt;summary&gt;Help - why argmax is not enough&lt;/summary&gt;

Two implementations can share an argmax while having a distribution drift large enough to break interventions. Track max error, MSE, KL, and top-k agreement.

&lt;/details&gt;

&lt;details&gt;
&lt;summary&gt;Solution&lt;/summary&gt;

Generate fixed reference logits, add a tiny perturbation, run `compare_logits`, and require explicit tolerances for max error, MSE, KL, and top-k agreement.

&lt;/details&gt;</pre>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
**Original exercises cell 7: code.**

<pre>tests.test_compare_logits_detects_match(compare_logits)
tests.test_compare_logits_rejects_shape_mismatch(compare_logits)
tests.test_compare_logits_rejects_real_drift(compare_logits)


def hf_parity_smoke_test() -&gt; bool:
    &#35; EXERCISE
    &#35; YOUR CODE HERE
    raise NotImplementedError()
    &#35; END EXERCISE


tests.test_hf_parity_smoke_test_passes(hf_parity_smoke_test)</pre>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
**Original exercises cell 8: markdown.**

<pre>&#35;&#35; Exercise 4 - Generation Parity

Use exact token equality for deterministic greedy-generation checks.

&lt;details&gt;
&lt;summary&gt;Expected output&lt;/summary&gt;

```text
All tests in `test_deterministic_generation_equal_detects_mismatch` passed!
```

&lt;/details&gt;

&lt;details&gt;
&lt;summary&gt;Help - where exact equality applies&lt;/summary&gt;

Exact token equality is appropriate for fixed greedy decoding. Do not expect it from temperature sampling.

&lt;/details&gt;

&lt;details&gt;
&lt;summary&gt;Solution&lt;/summary&gt;

Return `deterministic_generation_equal([1, 2, 3, 4], t.tensor([1, 2, 3, 4]))`.

&lt;/details&gt;</pre>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
**Original exercises cell 9: code.**

<pre>tests.test_deterministic_generation_equal_detects_mismatch(
    deterministic_generation_equal,
)


def generation_parity_smoke_test() -&gt; bool:
    &#35; EXERCISE
    &#35; YOUR CODE HERE
    raise NotImplementedError()
    &#35; END EXERCISE</pre>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
**Original exercises cell 10: markdown.**

<pre>&#35;&#35; Exercise 5 - Activation Storage

Write and reload activation shards with metadata instead of keeping every activation in VRAM.

&lt;details&gt;
&lt;summary&gt;Expected output&lt;/summary&gt;

```text
num_records: 2
names: [&&#35;x27;resid_pre&&#35;x27;, &&#35;x27;mlp_out&&#35;x27;]
All tests in `test_activation_store_smoke_test_contract` passed!
```

&lt;/details&gt;

&lt;details&gt;
&lt;summary&gt;Help - what metadata prevents&lt;/summary&gt;

A tensor file without hook, layer, prompt-set, and run metadata is hard to audit later. The index is part of the evidence.

&lt;/details&gt;

&lt;details&gt;
&lt;summary&gt;Solution&lt;/summary&gt;

Create a `DiskActivationStore`, append two tensors with hook/layer metadata, reload both tensors, and return `store.summary()`.

&lt;/details&gt;</pre>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
**Original exercises cell 11: code.**

<pre>def activation_store_smoke_test(output_dir: str | Path = &quot;activation_store_smoke&quot;) -&gt; dict:
    &#35; EXERCISE
    &#35; YOUR CODE HERE
    raise NotImplementedError()
    &#35; END EXERCISE


with tempfile.TemporaryDirectory() as tmpdir:
    tests.test_disk_activation_store_roundtrip(Path(tmpdir))
    tests.test_activation_store_smoke_test_contract(
        Path(tmpdir),
        activation_store_smoke_test,
    )</pre>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
**Original exercises cell 12: markdown.**

<pre>&#35;&#35; Exercise 6 - Notebook Contract

Return a JSON-serializable smoke-test dictionary with environment, budget, parity, generation, and activation-store evidence.

&lt;details&gt;
&lt;summary&gt;Expected output&lt;/summary&gt;

```text
All tests in `test_notebook_contract` passed!
```

&lt;/details&gt;

&lt;details&gt;
&lt;summary&gt;Solution&lt;/summary&gt;

Build the budget, run the parity helpers, write activations in a temporary directory, and return all results in a dictionary.

&lt;/details&gt;</pre>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
**Original exercises cell 13: code.**

<pre>def run_smoke_test(cpu: bool = True) -&gt; dict:
    &#35; EXERCISE
    &#35; YOUR CODE HERE
    raise NotImplementedError()
    &#35; END EXERCISE


tests.test_notebook_contract(run_smoke_test)</pre>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
**Original exercises cell 14: markdown.**

<pre>&#35;&#35; Signature Result

The committed CUDA report is the section&&#35;x27;s signature result: the managed uv environment runs a real BF16 CUDA tensor path and records its environment and peak VRAM.

| Check | Observed | Required |
|---|---:|---:|
| Python / PyTorch | `3.14.6` / `2.12.1+cu132` | `3.14` / `2.12.1+cu132` |
| CUDA runtime | `13.2` | `13.2` |
| BF16 CUDA matmul | `[1024,1024]`, finite | finite BF16 tensor |
| `uv pip check` | pass | pass |
| Peak VRAM | `0.043 GB` | `&lt; 1 GB` |

&lt;details&gt;
&lt;summary&gt;Help - what this proves&lt;/summary&gt;

It proves local runtime readiness, not correctness of a future model-specific implementation.

&lt;/details&gt;

&#35;&#35; Limitations

This notebook does not load a real HF checkpoint, benchmark a model, or prove any later interpretability result. Later sections need their own toy oracles, reference parity, and negative controls.

&#35;&#35; Bonus - Anomaly Hunting

- Increase the context length and watch the estimate change before loading anything.
- Change one top logit and confirm parity fails even if shapes match.
- Remove activation metadata and decide what evidence becomes unauditable.</pre>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
**Original exercises cell 15: code.**

<pre>def _load_committed_gpu_report() -&gt; dict:
    report = json.loads((section_dir / &quot;verification_report.json&quot;).read_text())
    gpu = report[&quot;metrics&quot;][&quot;gpu_test&quot;]
    assert report[&quot;accepted&quot;]
    assert gpu[&quot;cuda_available&quot;]
    assert gpu[&quot;gpu_tensor_test_passed&quot;]
    assert gpu[&quot;within_vram_budget&quot;]
    return gpu


def run_gpu_test(max_vram_gb: float = 24.0) -&gt; dict:
    gpu = _load_committed_gpu_report()
    assert gpu[&quot;peak_vram_gb&quot;] &lt;= max_vram_gb
    return gpu


def run_full_experiment(max_vram_gb: float = 24.0) -&gt; dict:
    return run_gpu_test(max_vram_gb=max_vram_gb)


tests.test_committed_gpu_report_records_cuda_runtime_and_no_fallback()
tests.test_exercise_notebook_course_ready_surface()</pre>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Original solutions notebook: complete cell sources
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
**Original solutions cell 0: markdown.**

<pre>&#35; [1.6] Local Frontier ML Infrastructure - Solutions

**Core Question.** How do you keep frontier interpretability notebooks honest on one local GPU?

This solved notebook executes the same contract as the learner notebook and interprets the committed CUDA report.

&#35;&#35; Learning Objectives

- Verify environment, memory, parity, generation, and activation-store checks.
- Distinguish estimates from measured CUDA allocations.
- Keep the claim boundary narrow.

&gt; ```yaml
&gt; Difficulty: 3
&gt; Importance: 5
&gt; ```

&lt;details&gt;
&lt;summary&gt;Help - how to read this solved notebook&lt;/summary&gt;

Passing infrastructure tests make later work auditable; they do not replace model-specific oracles.

&lt;/details&gt;</pre>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
**Original solutions cell 1: code.**

<pre>GT_TIER = &quot;GT-1&quot;
EXERCISE_ID = &quot;1_6_local_frontier_ml_infrastructure&quot;
DIFFICULTY = 3
IMPORTANCE = 5
EXPECTED_RUNTIME = &quot;seconds for toy contracts; minutes for the local CUDA runtime report&quot;
REQUIRES_GPU = True

import json
import sys
import tempfile
from pathlib import Path

import torch as t

chapter = &quot;chapter1_transformer_interp&quot;
section = &quot;part6_frontier_ml_infrastructure&quot;
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
exercises_dir = root_dir / chapter / &quot;exercises&quot;
section_dir = exercises_dir / section

if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part6_frontier_ml_infrastructure.tests as tests
import part6_frontier_ml_infrastructure.utils as utils

from arena_ext import (
    DiskActivationStore,
    compare_logits,
    deterministic_generation_equal,
    estimate_inference_memory,
    get_environment_report,
)

MAIN = True

from chapter1_transformer_interp.exercises.part6_frontier_ml_infrastructure import solutions</pre>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
**Original solutions cell 2: markdown.**

<pre>&#35;&#35; Exercise 1 - Environment Checks

&lt;details&gt;
&lt;summary&gt;Expected output&lt;/summary&gt;

The report should include Python, PyTorch, CUDA, GPU name, and BF16 support.

&lt;/details&gt;</pre>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
**Original solutions cell 3: code.**

<pre>environment = solutions.run_environment_check(required_vram_gb=24.0)
environment.as_dict()</pre>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
**Original solutions cell 4: markdown.**

<pre>&#35;&#35; Exercise 2 - VRAM Budget Estimates</pre>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
**Original solutions cell 5: code.**

<pre>budget = solutions.estimate_gemma_1b_smoke_budget()
utils.print_dict_table(&quot;Estimated memory&quot;, budget.as_dict())
tests.test_memory_budget_fits_local_tier(solutions.estimate_gemma_1b_smoke_budget)
tests.test_memory_budget_rejects_oversized_model()</pre>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
**Original solutions cell 6: markdown.**

<pre>&#35;&#35; Exercise 3 - HF Parity and Drift Controls

&lt;details&gt;
&lt;summary&gt;Interpreting parity&lt;/summary&gt;

The positive case should pass tight tolerances; the drift fixture should fail them.

&lt;/details&gt;</pre>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
**Original solutions cell 7: code.**

<pre>tests.test_compare_logits_detects_match(solutions.compare_logits)
tests.test_compare_logits_rejects_shape_mismatch(solutions.compare_logits)
tests.test_compare_logits_rejects_real_drift(solutions.compare_logits)
tests.test_hf_parity_smoke_test_passes(solutions.hf_parity_smoke_test)</pre>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
**Original solutions cell 8: markdown.**

<pre>&#35;&#35; Exercise 4 - Generation Parity</pre>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
**Original solutions cell 9: code.**

<pre>tests.test_deterministic_generation_equal_detects_mismatch(
    solutions.deterministic_generation_equal,
)
assert solutions.generation_parity_smoke_test()</pre>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
**Original solutions cell 10: markdown.**

<pre>&#35;&#35; Exercise 5 - Activation Storage</pre>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
**Original solutions cell 11: code.**

<pre>with tempfile.TemporaryDirectory() as tmpdir:
    tests.test_disk_activation_store_roundtrip(Path(tmpdir))
    tests.test_activation_store_smoke_test_contract(
        Path(tmpdir),
        solutions.activation_store_smoke_test,
    )</pre>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
**Original solutions cell 12: markdown.**

<pre>&#35;&#35; Exercise 6 - Notebook Contract</pre>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
**Original solutions cell 13: code.**

<pre>contract = solutions.run_smoke_test(cpu=True)
assert contract[&quot;hf_parity_passed&quot;]
assert contract[&quot;generation_parity_passed&quot;]
assert contract[&quot;activation_store&quot;][&quot;num_records&quot;] == 2
tests.test_notebook_contract(solutions.run_smoke_test)
contract</pre>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
**Original solutions cell 14: markdown.**

<pre>&#35;&#35; Signature Result

| Check | Observed | Required |
|---|---:|---:|
| Python / PyTorch | `3.14.6` / `2.12.1+cu132` | `3.14` / `2.12.1+cu132` |
| CUDA runtime | `13.2` | `13.2` |
| BF16 CUDA matmul | `[1024,1024]`, finite | finite BF16 tensor |
| `uv pip check` | pass | pass |
| Peak VRAM | `0.043 GB` | `&lt; 1 GB` |

&#35;&#35; Limitations

This does not load a checkpoint or benchmark a model. It proves the local evidence harness is ready.

&#35;&#35; Bonus - Anomaly Hunting

- Raise the estimate above 24GB and confirm the budget check fails.
- Change logit ordering and confirm top-k parity fails.
- Delete activation metadata and inspect what the report can no longer prove.</pre>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
**Original solutions cell 15: code.**

<pre>report = json.loads((section_dir / &quot;verification_report.json&quot;).read_text())
gpu = report[&quot;metrics&quot;][&quot;gpu_test&quot;]
assert report[&quot;accepted&quot;]
assert gpu[&quot;cuda_available&quot;]
assert gpu[&quot;gpu_tensor_test_passed&quot;]
assert gpu[&quot;gpu_matmul_dtype&quot;] == &quot;torch.bfloat16&quot;
assert gpu[&quot;peak_vram_gb&quot;] &lt; 1.0
print(&quot;python_major_minor:&quot;, gpu[&quot;python_major_minor&quot;])
print(&quot;torch_version:&quot;, gpu[&quot;torch_version&quot;])
print(&quot;cuda_version:&quot;, gpu[&quot;cuda_version&quot;])
print(&quot;peak_vram_gb:&quot;, gpu[&quot;peak_vram_gb&quot;])

tests.test_committed_gpu_report_records_cuda_runtime_and_no_fallback()
tests.test_exercise_notebook_course_ready_surface()</pre>
'''


# ! CELL TYPE: markdown
# ! FILTERS: [colab-ex]
# ! TAGS: []

r'''
**Historical code cell 1 (read-only original source).**

The active implementation appears above. The intact original book and its original helpers are linked in the introduction.

```python
GT_TIER = "GT-1"
EXERCISE_ID = "1_6_local_frontier_ml_infrastructure"
DIFFICULTY = 3
IMPORTANCE = 5
EXPECTED_RUNTIME = "seconds for toy contracts; minutes for the local CUDA runtime report"
REQUIRES_GPU = True

import json
import sys
import tempfile
from pathlib import Path

import torch as t

chapter = "chapter1_transformer_interp"
section = "part6_frontier_ml_infrastructure"
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section

if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part6_frontier_ml_infrastructure.tests as tests
import part6_frontier_ml_infrastructure.utils as utils

from arena_ext import (
    DiskActivationStore,
    compare_logits,
    deterministic_generation_equal,
    estimate_inference_memory,
    get_environment_report,
)

MAIN = True
```
'''

# ! CELL TYPE: markdown
# ! FILTERS: [colab-ex]
# ! TAGS: []

r'''
**Historical code cell 3 (read-only original source).**

The active implementation appears above. The intact original book and its original helpers are linked in the introduction.

```python
def run_environment_check(required_vram_gb: float | None = 24.0):
    # EXERCISE
    # YOUR CODE HERE
    raise NotImplementedError()
    # END EXERCISE


report = run_environment_check(required_vram_gb=24.0)
report.as_dict()
```
'''

# ! CELL TYPE: markdown
# ! FILTERS: [colab-ex]
# ! TAGS: []

r'''
**Historical code cell 5 (read-only original source).**

The active implementation appears above. The intact original book and its original helpers are linked in the introduction.

```python
def estimate_gemma_1b_smoke_budget(context_length: int = 2048):
    # EXERCISE
    # YOUR CODE HERE
    raise NotImplementedError()
    # END EXERCISE


tests.test_memory_budget_fits_local_tier(estimate_gemma_1b_smoke_budget)
tests.test_memory_budget_rejects_oversized_model()
```
'''

# ! CELL TYPE: markdown
# ! FILTERS: [colab-ex]
# ! TAGS: []

r'''
**Historical code cell 7 (read-only original source).**

The active implementation appears above. The intact original book and its original helpers are linked in the introduction.

```python
tests.test_compare_logits_detects_match(compare_logits)
tests.test_compare_logits_rejects_shape_mismatch(compare_logits)
tests.test_compare_logits_rejects_real_drift(compare_logits)


def hf_parity_smoke_test() -> bool:
    # EXERCISE
    # YOUR CODE HERE
    raise NotImplementedError()
    # END EXERCISE


tests.test_hf_parity_smoke_test_passes(hf_parity_smoke_test)
```
'''

# ! CELL TYPE: markdown
# ! FILTERS: [colab-ex]
# ! TAGS: []

r'''
**Historical code cell 9 (read-only original source).**

The active implementation appears above. The intact original book and its original helpers are linked in the introduction.

```python
tests.test_deterministic_generation_equal_detects_mismatch(
    deterministic_generation_equal,
)


def generation_parity_smoke_test() -> bool:
    # EXERCISE
    # YOUR CODE HERE
    raise NotImplementedError()
    # END EXERCISE
```
'''

# ! CELL TYPE: markdown
# ! FILTERS: [colab-ex]
# ! TAGS: []

r'''
**Historical code cell 11 (read-only original source).**

The active implementation appears above. The intact original book and its original helpers are linked in the introduction.

```python
def activation_store_smoke_test(output_dir: str | Path = "activation_store_smoke") -> dict:
    # EXERCISE
    # YOUR CODE HERE
    raise NotImplementedError()
    # END EXERCISE


with tempfile.TemporaryDirectory() as tmpdir:
    tests.test_disk_activation_store_roundtrip(Path(tmpdir))
    tests.test_activation_store_smoke_test_contract(
        Path(tmpdir),
        activation_store_smoke_test,
    )
```
'''

# ! CELL TYPE: markdown
# ! FILTERS: [colab-ex]
# ! TAGS: []

r'''
**Historical code cell 13 (read-only original source).**

The active implementation appears above. The intact original book and its original helpers are linked in the introduction.

```python
def run_smoke_test(cpu: bool = True) -> dict:
    # EXERCISE
    # YOUR CODE HERE
    raise NotImplementedError()
    # END EXERCISE


tests.test_notebook_contract(run_smoke_test)
```
'''

# ! CELL TYPE: markdown
# ! FILTERS: [colab-ex]
# ! TAGS: []

r'''
**Historical code cell 15 (read-only original source).**

The active implementation appears above. The intact original book and its original helpers are linked in the introduction.

```python
def _load_committed_gpu_report() -> dict:
    report = json.loads((section_dir / "verification_report.json").read_text())
    gpu = report["metrics"]["gpu_test"]
    assert report["accepted"]
    assert gpu["cuda_available"]
    assert gpu["gpu_tensor_test_passed"]
    assert gpu["within_vram_budget"]
    return gpu


def run_gpu_test(max_vram_gb: float = 24.0) -> dict:
    gpu = _load_committed_gpu_report()
    assert gpu["peak_vram_gb"] <= max_vram_gb
    return gpu


def run_full_experiment(max_vram_gb: float = 24.0) -> dict:
    return run_gpu_test(max_vram_gb=max_vram_gb)


tests.test_committed_gpu_report_records_cuda_runtime_and_no_fallback()
tests.test_exercise_notebook_course_ready_surface()
```
'''

# ! CELL TYPE: code
# ! FILTERS: [colab-soln]
# ! TAGS: [keep-main,master-comment]

# GT_TIER = "GT-1"
# EXERCISE_ID = "1_6_local_frontier_ml_infrastructure"
# DIFFICULTY = 3
# IMPORTANCE = 5
# EXPECTED_RUNTIME = "seconds for toy contracts; minutes for the local CUDA runtime report"
# REQUIRES_GPU = True
# 
# import json
# import sys
# import tempfile
# from pathlib import Path
# 
# import torch as t
# 
# chapter = "chapter1_transformer_interp"
# section = "part6_frontier_ml_infrastructure"
# root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
# exercises_dir = root_dir / chapter / "exercises"
# section_dir = exercises_dir / section
# 
# if str(root_dir) not in sys.path:
#     sys.path.append(str(root_dir))
# if str(exercises_dir) not in sys.path:
#     sys.path.append(str(exercises_dir))
# 
# import part6_frontier_ml_infrastructure.tests as tests
# import part6_frontier_ml_infrastructure.utils as utils
# 
# from arena_ext import (
#     DiskActivationStore,
#     compare_logits,
#     deterministic_generation_equal,
#     estimate_inference_memory,
#     get_environment_report,
# )
# 
# MAIN = True
# 
# from chapter1_transformer_interp.exercises.part6_frontier_ml_infrastructure import solutions

# ! CELL TYPE: code
# ! FILTERS: [colab-soln]
# ! TAGS: [keep-main,master-comment]

# environment = solutions.run_environment_check(required_vram_gb=24.0)
# environment.as_dict()

# ! CELL TYPE: code
# ! FILTERS: [colab-soln]
# ! TAGS: [keep-main,master-comment]

# budget = solutions.estimate_gemma_1b_smoke_budget()
# utils.print_dict_table("Estimated memory", budget.as_dict())
# tests.test_memory_budget_fits_local_tier(solutions.estimate_gemma_1b_smoke_budget)
# tests.test_memory_budget_rejects_oversized_model()

# ! CELL TYPE: code
# ! FILTERS: [colab-soln]
# ! TAGS: [keep-main,master-comment]

# tests.test_compare_logits_detects_match(solutions.compare_logits)
# tests.test_compare_logits_rejects_shape_mismatch(solutions.compare_logits)
# tests.test_compare_logits_rejects_real_drift(solutions.compare_logits)
# tests.test_hf_parity_smoke_test_passes(solutions.hf_parity_smoke_test)

# ! CELL TYPE: code
# ! FILTERS: [colab-soln]
# ! TAGS: [keep-main,master-comment]

# tests.test_deterministic_generation_equal_detects_mismatch(
#     solutions.deterministic_generation_equal,
# )
# assert solutions.generation_parity_smoke_test()

# ! CELL TYPE: code
# ! FILTERS: [colab-soln]
# ! TAGS: [keep-main,master-comment]

# with tempfile.TemporaryDirectory() as tmpdir:
#     tests.test_disk_activation_store_roundtrip(Path(tmpdir))
#     tests.test_activation_store_smoke_test_contract(
#         Path(tmpdir),
#         solutions.activation_store_smoke_test,
#     )

# ! CELL TYPE: code
# ! FILTERS: [colab-soln]
# ! TAGS: [keep-main,master-comment]

# contract = solutions.run_smoke_test(cpu=True)
# assert contract["hf_parity_passed"]
# assert contract["generation_parity_passed"]
# assert contract["activation_store"]["num_records"] == 2
# tests.test_notebook_contract(solutions.run_smoke_test)
# contract

# ! CELL TYPE: code
# ! FILTERS: [colab-soln]
# ! TAGS: [keep-main,master-comment]

# report = json.loads((section_dir / "verification_report.json").read_text())
# gpu = report["metrics"]["gpu_test"]
# assert report["accepted"]
# assert gpu["cuda_available"]
# assert gpu["gpu_tensor_test_passed"]
# assert gpu["gpu_matmul_dtype"] == "torch.bfloat16"
# assert gpu["peak_vram_gb"] < 1.0
# print("python_major_minor:", gpu["python_major_minor"])
# print("torch_version:", gpu["torch_version"])
# print("cuda_version:", gpu["cuda_version"])
# print("peak_vram_gb:", gpu["peak_vram_gb"])
# 
# tests.test_committed_gpu_report_records_cuda_runtime_and_no_fallback()
# tests.test_exercise_notebook_course_ready_surface()

