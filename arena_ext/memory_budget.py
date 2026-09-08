"""Memory estimation utilities for local ARENA extension notebooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


DTypeName = Literal["float32", "float16", "bfloat16", "int8", "int4"]


_DTYPE_BYTES: dict[DTypeName, float] = {
    "float32": 4.0,
    "float16": 2.0,
    "bfloat16": 2.0,
    "int8": 1.0,
    "int4": 0.5,
}


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


def bytes_per_value(dtype: DTypeName) -> float:
    """Return storage bytes per scalar for common model dtypes."""

    try:
        return _DTYPE_BYTES[dtype]
    except KeyError as exc:
        expected = sorted(_DTYPE_BYTES)
        raise ValueError(f"Unknown dtype {dtype!r}; expected one of {expected}") from exc


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
