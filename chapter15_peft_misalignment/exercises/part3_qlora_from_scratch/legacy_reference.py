"""Reference implementation for [15.3] QLoRA from Scratch.

The learner notebook reimplements the core functions in this module. CPU execution
establishes exact NF4 and gradient-flow oracles. The pretrained comparison is only
available through the strict CUDA entry points at the bottom of this file.
"""

from __future__ import annotations

import gc
import hashlib
import inspect
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch as t
import torch.nn.functional as F
from torch import nn


MODEL_ID = "roneneldan/TinyStories-1M"
MODEL_REVISION = "77f1b168e219585646439073245fe87e56b3023e"
TARGET_MODULES = (
    "transformer.h.6.attn.attention.q_proj",
    "transformer.h.6.attn.attention.v_proj",
)
LABEL_TEXT = (" red", " blue")
MARKERS = ("dax", "wug")
NF4_BLOCK_SIZE = 64
DOUBLE_QUANT_BLOCK_SIZE = 256
ADAPTER_LEARNING_RATE = 2e-2
FULL_FINETUNE_LEARNING_RATE = 5e-5
DEFAULT_EXPERIMENT_SEEDS = (0, 1, 2)
INTENDED_QUANTIZED_SUFFIXES = (
    "attn.attention.q_proj",
    "attn.attention.k_proj",
    "attn.attention.v_proj",
    "attn.attention.out_proj",
    "mlp.c_fc",
    "mlp.c_proj",
)


@dataclass(frozen=True)
class NF4Tensor:
    indices: t.Tensor
    absmax: t.Tensor
    shape: tuple[int, ...]
    numel: int
    block_size: int


@dataclass(frozen=True)
class DoubleQuantizedScales:
    codes: t.Tensor
    offset: t.Tensor
    second_absmax: t.Tensor
    numel: int
    block_size: int


@dataclass(frozen=True)
class PromptExample:
    prompt: str
    label: int
    subject: str
    template_id: str
    marker: str | None = None
    intervention: str = "canonical"


@dataclass(frozen=True)
class TrainTrace:
    losses: list[float]
    steps: int
    examples_seen: int
    learning_rate: float
    trainable_parameter_count: int
    trainable_parameter_bytes: int
    optimizer_state_bytes: int
    optimizer_state_dtypes: dict[str, int]


@dataclass(frozen=True)
class BehaviorMetrics:
    accuracy: float
    two_label_conditional_nll: float
    target_probability: float
    predictions: list[int]


@dataclass(frozen=True)
class SpectrumMetrics:
    singular_values: list[float]
    effective_rank: float
    stable_rank: float
    update_norm: float
    numerical_rank: int
    top_singular_value_fraction: float


def nf4_codebook(*, device: t.device | str = "cpu", dtype: t.dtype = t.float32) -> t.Tensor:
    """Return the canonical bitsandbytes NF4 lookup table in sorted order."""

    return t.tensor(
        [
            -1.0,
            -0.6961928009986877,
            -0.5250730514526367,
            -0.39491748809814453,
            -0.28444138169288635,
            -0.18477343022823334,
            -0.09105003625154495,
            0.0,
            0.07958029955625534,
            0.16093020141124725,
            0.24611230194568634,
            0.33791524171829224,
            0.44070982933044434,
            0.5626170039176941,
            0.7229568362236023,
            1.0,
        ],
        device=device,
        dtype=dtype,
    )


def expected_dynamic_codebook(
    *,
    device: t.device | str = "cpu",
    dtype: t.dtype = t.float32,
) -> t.Tensor:
    """Return the signed dynamic map used by bitsandbytes 0.50.2 on CPU/CUDA.

    Keep all 256 values exact: this backend retains the generated negative
    endpoint (-0.992968738079071), including on CPU. Older backend-specific
    endpoint substitutions must not silently change the current oracle.
    """

    from bitsandbytes import functional as bnb_functional

    return bnb_functional.create_dynamic_map(signed=True).to(device=device, dtype=dtype)


def _require_finite_tensor(value: Any, *, name: str) -> t.Tensor:
    if not isinstance(value, t.Tensor):
        raise RuntimeError(f"{name} must be a materialized tensor")
    if not bool(t.isfinite(value.detach().float()).all().item()):
        raise RuntimeError(f"{name} contains non-finite values")
    return value


def blockwise_absmax(values: t.Tensor, *, block_size: int = NF4_BLOCK_SIZE) -> t.Tensor:
    """Return one absolute maximum per flattened block, padding the final block with zeros."""

    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if values.numel() == 0:
        raise ValueError("cannot quantize an empty tensor")
    flat = values.detach().float().flatten()
    padded_numel = math.ceil(flat.numel() / block_size) * block_size
    padded = F.pad(flat, (0, padded_numel - flat.numel()))
    return padded.view(-1, block_size).abs().amax(dim=1)


def quantize_nf4(values: t.Tensor, *, block_size: int = NF4_BLOCK_SIZE) -> NF4Tensor:
    """Quantize a tensor by nearest-neighbour lookup after blockwise absmax normalization."""

    scales = blockwise_absmax(values, block_size=block_size)
    flat = values.detach().float().flatten()
    padded_numel = scales.numel() * block_size
    blocks = F.pad(flat, (0, padded_numel - flat.numel())).view(-1, block_size)
    safe_scales = scales.clamp_min(t.finfo(t.float32).tiny)
    normalized = blocks / safe_scales[:, None]
    codebook = nf4_codebook(device=values.device)
    distances = (normalized[..., None] - codebook).abs()
    indices = distances.argmin(dim=-1).to(t.uint8)
    return NF4Tensor(indices, scales.to(values.device), tuple(values.shape), values.numel(), block_size)


def dequantize_nf4(quantized: NF4Tensor, *, dtype: t.dtype = t.float32) -> t.Tensor:
    """Invert ``quantize_nf4`` using the stored code indices and block scales."""

    codebook = nf4_codebook(device=quantized.indices.device)
    blocks = codebook[quantized.indices.long()] * quantized.absmax[:, None]
    return blocks.flatten()[: quantized.numel].reshape(quantized.shape).to(dtype)


def quantization_error(reference: t.Tensor, reconstructed: t.Tensor) -> dict[str, float]:
    """Measure reconstruction error without hiding scale-dependent failures."""

    if reference.shape != reconstructed.shape:
        raise ValueError("reference and reconstructed tensors must have the same shape")
    ref = reference.detach().float().flatten()
    rec = reconstructed.detach().float().flatten()
    error = rec - ref
    signal = ref.square().mean().clamp_min(t.finfo(t.float32).tiny)
    noise = error.square().mean().clamp_min(t.finfo(t.float32).tiny)
    cosine = F.cosine_similarity(ref, rec, dim=0).item() if ref.norm() > 0 and rec.norm() > 0 else 1.0
    return {
        "mse": float(noise.item()),
        "max_abs_error": float(error.abs().max().item()),
        "cosine_similarity": float(cosine),
        "sqnr_db": float((10 * t.log10(signal / noise)).item()),
    }


def double_quantize_scales(
    scales: t.Tensor,
    *,
    block_size: int = DOUBLE_QUANT_BLOCK_SIZE,
) -> DoubleQuantizedScales:
    """Toy nested quantization: mean-center scales, then quantize each group to signed int8.

    bitsandbytes also mean-centers first-level absmax values and block-quantizes them.
    This transparent oracle uses a uniform signed-int8 code so every stored bit is easy
    to account for; the real CUDA path uses bitsandbytes' ``compress_statistics=True``.
    """

    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if scales.numel() == 0 or (scales < 0).any():
        raise ValueError("scales must be a non-empty non-negative tensor")
    flat = scales.detach().float().flatten()
    offset = flat.mean()
    centered = flat - offset
    padded_numel = math.ceil(flat.numel() / block_size) * block_size
    blocks = F.pad(centered, (0, padded_numel - flat.numel())).view(-1, block_size)
    second_absmax = blocks.abs().amax(dim=1)
    safe = second_absmax.clamp_min(t.finfo(t.float32).tiny)
    codes = t.round(127 * blocks / safe[:, None]).clamp(-127, 127).to(t.int8)
    return DoubleQuantizedScales(codes, offset, second_absmax, flat.numel(), block_size)


def dequantize_scales(quantized: DoubleQuantizedScales) -> t.Tensor:
    """Dequantize the transparent nested-scale oracle."""

    centered = quantized.codes.float() * quantized.second_absmax[:, None] / 127
    return centered.flatten()[: quantized.numel] + quantized.offset


def nf4_storage_accounting(
    *,
    numel: int,
    block_size: int = NF4_BLOCK_SIZE,
    double_quant_block_size: int = DOUBLE_QUANT_BLOCK_SIZE,
) -> dict[str, float | int]:
    """Separate the paper's amortized formula from a tail-aware byte layout.

    Neither formula is a measurement of a materialized bitsandbytes layer. Use
    ``linear4bit_storage_accounting`` for that; its total includes codebooks and
    every tensor in the nested ``QuantState``.
    """

    if numel <= 0 or block_size <= 0 or double_quant_block_size <= 0:
        raise ValueError("numel and block sizes must be positive")
    paper_single_bpp = 4 + 32 / block_size
    paper_double_bpp = 4 + 8 / block_size + 32 / (block_size * double_quant_block_size)
    packed_value_bytes = math.ceil(numel / 2)
    blocks = math.ceil(numel / block_size)
    second_blocks = math.ceil(blocks / double_quant_block_size)
    single_bytes = packed_value_bytes + 4 * blocks
    double_bytes = packed_value_bytes + blocks + 4 * second_blocks + 4
    return {
        "numel": numel,
        "blocks": blocks,
        "second_blocks": second_blocks,
        "paper_amortized_single_bits_per_parameter": paper_single_bpp,
        "paper_amortized_double_bits_per_parameter": paper_double_bpp,
        "tail_aware_packed_value_bytes": packed_value_bytes,
        "tail_aware_single_bytes": single_bytes,
        "tail_aware_double_bytes": double_bytes,
        "tail_aware_single_bits_per_parameter": 8 * single_bytes / numel,
        "tail_aware_double_bits_per_parameter": 8 * double_bytes / numel,
        "tail_aware_metadata_savings_bytes": single_bytes - double_bytes,
        # Backward-compatible names are explicitly the tail-aware formula, not
        # measurements from a Linear4bit allocation.
        "packed_value_bits": 8 * packed_value_bytes,
        "single_quant_bits": 8 * single_bytes,
        "double_quant_bits": 8 * double_bytes,
        "single_bits_per_parameter": 8 * single_bytes / numel,
        "double_bits_per_parameter": 8 * double_bytes / numel,
        "metadata_savings_bits": 8 * (single_bytes - double_bytes),
    }


def ste_nf4(values: t.Tensor, *, block_size: int = NF4_BLOCK_SIZE) -> t.Tensor:
    """Use exact NF4 values in the forward pass and identity gradients in the backward pass."""

    reconstructed = dequantize_nf4(quantize_nf4(values, block_size=block_size), dtype=values.dtype)
    return values + (reconstructed - values).detach()


def toy_qlora_forward(
    inputs: t.Tensor,
    base_weight: t.Tensor,
    lora_a: t.Tensor,
    lora_b: t.Tensor,
    *,
    alpha: float,
    block_size: int = NF4_BLOCK_SIZE,
) -> t.Tensor:
    """A transparent quantized-base plus low-rank-adapter forward pass."""

    if lora_a.ndim != 2 or lora_b.ndim != 2 or lora_b.shape[1] != lora_a.shape[0]:
        raise ValueError("LoRA factors must have shapes (rank, in) and (out, rank)")
    if base_weight.shape != (lora_b.shape[0], lora_a.shape[1]):
        raise ValueError("base weight and LoRA update must have identical shapes")
    rank = lora_a.shape[0]
    quantized_base = F.linear(inputs, ste_nf4(base_weight, block_size=block_size))
    adapter = F.linear(F.linear(inputs, lora_a), lora_b)
    return quantized_base + (float(alpha) / rank) * adapter


class DenseLoRALinear(nn.Module):
    """Transparent LoRA around a dense pretrained ``nn.Linear`` layer."""

    def __init__(self, base: nn.Linear, *, rank: int, alpha: float) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("DenseLoRALinear requires nn.Linear")
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.lora_a = nn.Parameter(t.empty(rank, base.in_features, dtype=t.float32, device=base.weight.device))
        self.lora_b = nn.Parameter(t.zeros(base.out_features, rank, dtype=t.float32, device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def delta_weight(self) -> t.Tensor:
        return self.scaling * (self.lora_b @ self.lora_a)

    def forward(self, inputs: t.Tensor) -> t.Tensor:
        base_output = self.base(inputs)
        adapter = F.linear(F.linear(inputs.float(), self.lora_a), self.lora_b)
        return base_output + self.scaling * adapter.to(base_output.dtype)


def _tensor_storage_key(tensor: t.Tensor) -> tuple[str, int, int]:
    storage = tensor.untyped_storage()
    return str(tensor.device), storage.data_ptr(), storage.nbytes()


def _unique_tensor_storage_bytes(tensors: Iterable[t.Tensor]) -> int:
    seen: set[tuple[str, int, int]] = set()
    total = 0
    for tensor in tensors:
        key = _tensor_storage_key(tensor)
        if key not in seen:
            seen.add(key)
            total += tensor.untyped_storage().nbytes()
    return total


def _quant_state_tensors(state: Any, *, prefix: str = "quant_state") -> list[tuple[str, t.Tensor]]:
    if state is None:
        return []
    tensors = []
    for field in ("absmax", "code", "offset"):
        value = getattr(state, field, None)
        if isinstance(value, t.Tensor):
            tensors.append((f"{prefix}.{field}", value))
    tensors.extend(_quant_state_tensors(getattr(state, "state2", None), prefix=f"{prefix}.state2"))
    return tensors


def linear4bit_storage_accounting(layer: nn.Module) -> dict[str, Any]:
    """Measure a materialized Linear4bit, including codebooks and nested state."""

    from bitsandbytes.nn import Linear4bit, Params4bit

    if not isinstance(layer, Linear4bit) or not isinstance(layer.weight, Params4bit):
        raise TypeError("layer must be an actual bitsandbytes Linear4bit with Params4bit weight")
    state = getattr(layer.weight, "quant_state", None)
    if state is None or not bool(getattr(layer.weight, "bnb_quantized", False)):
        raise RuntimeError("Linear4bit weight is declared but unmaterialized")
    state_tensors = _quant_state_tensors(state)
    state_records = {
        name: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "storage_bytes": value.untyped_storage().nbytes(),
            "all_finite": bool(t.isfinite(value.detach().float()).all().item()),
        }
        for name, value in state_tensors
    }
    original_numel = math.prod(int(dim) for dim in state.shape)
    formula = nf4_storage_accounting(
        numel=original_numel,
        block_size=int(state.blocksize),
        double_quant_block_size=int(state.state2.blocksize) if state.state2 is not None else DOUBLE_QUANT_BLOCK_SIZE,
    )
    packed_weight_bytes = layer.weight.untyped_storage().nbytes()
    bias_bytes = 0 if layer.bias is None else layer.bias.untyped_storage().nbytes()
    actual_tensors = [layer.weight]
    if layer.bias is not None:
        actual_tensors.append(layer.bias)
    actual_tensors.extend(value for _, value in state_tensors)
    return {
        "original_numel": original_numel,
        "packed_weight_shape": list(layer.weight.shape),
        "packed_weight_dtype": str(layer.weight.dtype),
        "packed_weight_bytes": packed_weight_bytes,
        "expected_packed_weight_bytes": math.ceil(original_numel / 2),
        "bias_bytes": bias_bytes,
        "quant_state_tensor_bytes": _unique_tensor_storage_bytes(value for _, value in state_tensors),
        "quant_state_tensors": state_records,
        "actual_tensor_bytes_including_codebooks": _unique_tensor_storage_bytes(actual_tensors),
        "paper_amortized_double_bits_per_parameter": formula[
            "paper_amortized_double_bits_per_parameter"
        ],
        "tail_aware_formula_bytes_excluding_codebooks": formula["tail_aware_double_bytes"],
    }


def audit_linear4bit(
    layer: nn.Module,
    *,
    expected_compute_dtype: t.dtype,
    expected_storage_dtype: t.dtype = t.uint8,
) -> dict[str, Any]:
    """Fail closed unless a layer has a real nested NF4 packed state."""

    from bitsandbytes.nn import Linear4bit, Params4bit

    if not isinstance(layer, Linear4bit) or not isinstance(getattr(layer, "weight", None), Params4bit):
        raise TypeError("expected an actual bitsandbytes Linear4bit with Params4bit weight")
    weight = layer.weight
    state = getattr(weight, "quant_state", None)
    if state is None or not bool(getattr(weight, "bnb_quantized", False)):
        raise RuntimeError("Linear4bit is unmaterialized: packed weight and QuantState are required")
    if getattr(weight, "quant_type", None) != "nf4" or getattr(state, "quant_type", None) != "nf4":
        raise RuntimeError("materialized layer is not NF4")
    if weight.dtype != expected_storage_dtype or getattr(layer, "quant_storage", None) != expected_storage_dtype:
        raise RuntimeError(f"expected packed storage dtype {expected_storage_dtype}, got {weight.dtype}")
    if getattr(layer, "compute_dtype", None) != expected_compute_dtype:
        raise RuntimeError(f"expected compute dtype {expected_compute_dtype}, got {layer.compute_dtype}")
    if not bool(getattr(state, "nested", False)) or state.state2 is None:
        raise RuntimeError("nested QuantState is required (compress_statistics=True)")
    _require_finite_tensor(weight, name="packed weight")
    if layer.bias is not None:
        _require_finite_tensor(layer.bias, name="bias")
    _require_finite_tensor(getattr(state, "offset", None), name="quant_state.offset")
    first_absmax = _require_finite_tensor(getattr(state, "absmax", None), name="quant_state.absmax")
    nf4_state_code = _require_finite_tensor(getattr(state, "code", None), name="quant_state.code")
    nested_absmax = _require_finite_tensor(
        getattr(state.state2, "absmax", None), name="quant_state.state2.absmax"
    )
    dynamic_state_code = _require_finite_tensor(
        getattr(state.state2, "code", None), name="quant_state.state2.code"
    )
    if first_absmax.dtype != t.uint8:
        raise RuntimeError("nested first-level absmax must be packed as uint8")
    for name, value in _quant_state_tensors(state):
        _require_finite_tensor(value, name=name)
    if int(getattr(state, "blocksize", -1)) != NF4_BLOCK_SIZE:
        raise RuntimeError(f"NF4 QuantState block size must be {NF4_BLOCK_SIZE}")
    if int(getattr(state.state2, "blocksize", -1)) != DOUBLE_QUANT_BLOCK_SIZE:
        raise RuntimeError(f"nested QuantState block size must be {DOUBLE_QUANT_BLOCK_SIZE}")
    expected_nf4 = nf4_codebook(device=nf4_state_code.device, dtype=nf4_state_code.dtype)
    if nf4_state_code.shape != expected_nf4.shape or not t.equal(nf4_state_code, expected_nf4):
        raise RuntimeError("NF4 QuantState codebook values do not match canonical NF4")
    expected_dynamic = expected_dynamic_codebook(
        device=dynamic_state_code.device,
        dtype=dynamic_state_code.dtype,
    )
    if dynamic_state_code.shape != expected_dynamic.shape or not t.equal(dynamic_state_code, expected_dynamic):
        raise RuntimeError("nested QuantState codebook values do not match the signed dynamic map")
    from bitsandbytes import functional as bnb_functional

    dequantized = bnb_functional.dequantize_4bit(weight.data, state)
    _require_finite_tensor(dequantized, name="dequantized weight")
    storage = linear4bit_storage_accounting(layer)
    if storage["packed_weight_bytes"] != storage["expected_packed_weight_bytes"]:
        raise RuntimeError("packed Params4bit storage does not contain exactly two values per byte")
    return {
        "materialized": bool(getattr(weight, "bnb_quantized", False) and state is not None),
        "quant_type": str(state.quant_type),
        "nested": bool(state.nested and state.state2 is not None),
        "all_state_tensors_finite": all(
            record["all_finite"] for record in storage["quant_state_tensors"].values()
        ),
        "dequantized_weight_finite": bool(t.isfinite(dequantized.float()).all().item()),
        "nf4_codebook_exact": bool(t.equal(nf4_state_code, expected_nf4)),
        "dynamic_codebook_exact": bool(t.equal(dynamic_state_code, expected_dynamic)),
        "storage_dtype": str(weight.dtype),
        "compute_dtype": str(layer.compute_dtype),
        "quant_state_dtype": str(state.dtype),
        "nested_absmax_dtype": str(nested_absmax.dtype),
        "packed_weight_shape": storage["packed_weight_shape"],
        "storage": storage,
    }


class QLoRALinear(nn.Module):
    """Transparent LoRA around an actual bitsandbytes NF4 ``Linear4bit`` layer."""

    def __init__(self, base: nn.Module, *, rank: int, alpha: float) -> None:
        super().__init__()
        from bitsandbytes.nn import Linear4bit

        if not isinstance(base, Linear4bit):
            raise TypeError("QLoRALinear requires an actual bitsandbytes Linear4bit base")
        quant_type = getattr(base.weight, "quant_type", getattr(base, "quant_type", None))
        if quant_type != "nf4":
            raise TypeError("QLoRALinear requires NF4, not another 4-bit code")
        try:
            audit_linear4bit(base, expected_compute_dtype=base.compute_dtype, expected_storage_dtype=base.quant_storage)
        except RuntimeError as error:
            raise TypeError(f"QLoRALinear requires a materialized nested-NF4 base: {error}") from error
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        device = base.weight.device
        self.lora_a = nn.Parameter(t.empty(rank, base.in_features, dtype=t.float32, device=device))
        self.lora_b = nn.Parameter(t.zeros(base.out_features, rank, dtype=t.float32, device=device))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def delta_weight(self) -> t.Tensor:
        return self.scaling * (self.lora_b @ self.lora_a)

    def forward(self, inputs: t.Tensor) -> t.Tensor:
        base_output = self.base(inputs)
        adapter = F.linear(F.linear(inputs.float(), self.lora_a), self.lora_b)
        return base_output + self.scaling * adapter.to(base_output.dtype)


def dequantize_bnb_weight(layer: nn.Module) -> t.Tensor:
    """Return the actual weight represented by a quantized bitsandbytes linear layer."""

    from bitsandbytes import functional as bnb_functional
    from bitsandbytes.nn import Linear4bit

    if not isinstance(layer, Linear4bit) or getattr(layer.weight, "quant_state", None) is None:
        raise TypeError("layer must be an initialized bitsandbytes Linear4bit")
    return bnb_functional.dequantize_4bit(layer.weight.data, layer.weight.quant_state)


def dequantized_base_max_diff(layer: nn.Module, inputs: t.Tensor) -> float:
    """Compare a real 4-bit matmul with a dense matmul through its dequantized weight."""

    layer.eval()
    with t.no_grad():
        actual = layer(inputs)
        weight = dequantize_bnb_weight(layer).to(inputs.dtype)
        bias = None if layer.bias is None else layer.bias.to(inputs.dtype)
        expected = F.linear(inputs, weight, bias)
    return float((actual - expected).abs().max().item())


def _get_parent_and_leaf(model: nn.Module, qualified_name: str) -> tuple[nn.Module, str]:
    parts = qualified_name.split(".")
    parent: nn.Module = model
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    return parent, parts[-1]


def inject_dense_lora(
    model: nn.Module,
    target_modules: Sequence[str],
    *,
    rank: int,
    alpha: float,
) -> dict[str, DenseLoRALinear]:
    inserted: dict[str, DenseLoRALinear] = {}
    for name in target_modules:
        parent, leaf = _get_parent_and_leaf(model, name)
        wrapped = DenseLoRALinear(getattr(parent, leaf), rank=rank, alpha=alpha)
        setattr(parent, leaf, wrapped)
        inserted[name] = wrapped
    freeze_except_adapters(model)
    return inserted


def inject_qlora(
    model: nn.Module,
    target_modules: Sequence[str],
    *,
    rank: int,
    alpha: float,
) -> dict[str, QLoRALinear]:
    inserted: dict[str, QLoRALinear] = {}
    for name in target_modules:
        parent, leaf = _get_parent_and_leaf(model, name)
        wrapped = QLoRALinear(getattr(parent, leaf), rank=rank, alpha=alpha)
        setattr(parent, leaf, wrapped)
        inserted[name] = wrapped
    freeze_except_adapters(model)
    return inserted


def freeze_except_adapters(model: nn.Module) -> int:
    """Freeze every parameter, then expose only visible LoRA factors to the optimizer."""

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in model.modules():
        if (isinstance(getattr(module, "lora_a", None), nn.Parameter)
                and isinstance(getattr(module, "lora_b", None), nn.Parameter)
                and callable(getattr(module, "delta_weight", None))):
            module.lora_a.requires_grad_(True)
            module.lora_b.requires_grad_(True)
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _crossed_codebook_examples(
    subjects: Sequence[str],
    templates: Sequence[tuple[str, str]],
) -> list[PromptExample]:
    """Cross every subject and template with both labels and their markers."""

    return [
        PromptExample(
            template.format(subject=subject, marker=MARKERS[label]),
            label,
            subject,
            template_id,
            MARKERS[label],
        )
        for subject in subjects
        for template_id, template in templates
        for label in range(len(LABEL_TEXT))
    ]


def make_codebook_examples() -> tuple[list[PromptExample], list[PromptExample]]:
    """Create fully crossed training and same-subject template-transfer prompts."""

    subjects = ("Alice", "Ben", "Cora", "Drew", "Eli", "Faye", "Gina", "Hugo", "Iris", "Jules")
    train_templates = (
        ("train_a", "In the tiny codebook, {subject} has marker {marker}. The color is"),
        ("train_b", "Remember the secret rule. {subject} is in group {marker}. Write"),
        ("train_c", "The storyteller sees {subject}'s {marker} badge and answers"),
    )
    heldout_templates = (
        ("heldout_a", "Using the same codebook, decode {subject}'s marker {marker} as"),
        ("heldout_b", "Complete this entry: {subject}, badge {marker}, color"),
    )
    return (
        _crossed_codebook_examples(subjects, train_templates),
        _crossed_codebook_examples(subjects, heldout_templates),
    )


def make_unseen_subject_examples() -> list[PromptExample]:
    """Create a fully crossed unseen-subject and unseen-template diagnostic."""

    subjects = ("Kira", "Luca", "Mona", "Niko", "Opal", "Pia", "Quinn", "Ravi", "Suki", "Tariq")
    templates = (
        ("ood_subject_a", "For a new person, translate {subject}'s {marker} marker into color"),
        ("ood_subject_b", "New codebook entry: {subject} wears {marker}; answer"),
    )
    return _crossed_codebook_examples(subjects, templates)


def make_marker_control_examples(
    examples: Sequence[PromptExample],
) -> dict[str, list[PromptExample]]:
    """Build paired marker-swap and marker-removal counterfactuals."""

    swapped: list[PromptExample] = []
    removed: list[PromptExample] = []
    for example in examples:
        if example.marker not in MARKERS or example.label not in (0, 1):
            raise ValueError("marker controls require canonical binary marker examples")
        opposite_label = 1 - example.label
        opposite_marker = MARKERS[opposite_label]
        swapped.append(
            PromptExample(
                example.prompt.replace(example.marker, opposite_marker, 1),
                opposite_label,
                example.subject,
                example.template_id,
                opposite_marker,
                "marker_swap",
            )
        )
        removed.append(
            PromptExample(
                example.prompt.replace(example.marker, "[marker removed]", 1),
                example.label,
                example.subject,
                example.template_id,
                None,
                "marker_removed",
            )
        )
    return {"marker_swap": swapped, "marker_removed": removed}


def counterbalance_audit(examples: Sequence[PromptExample]) -> dict[str, Any]:
    """Prove that name and template identity carry no marginal label signal."""

    if not examples:
        raise ValueError("counterbalance audit requires examples")
    subjects = sorted({example.subject for example in examples})
    templates = sorted({example.template_id for example in examples})
    expected_pairs = {
        (subject, template_id, label)
        for subject in subjects
        for template_id in templates
        for label in range(len(LABEL_TEXT))
    }
    observed_pairs = {(example.subject, example.template_id, example.label) for example in examples}
    subject_label_counts = {
        subject: [sum(example.subject == subject and example.label == label for example in examples) for label in (0, 1)]
        for subject in subjects
    }
    template_label_counts = {
        template_id: [
            sum(example.template_id == template_id and example.label == label for example in examples)
            for label in (0, 1)
        ]
        for template_id in templates
    }
    return {
        "examples": len(examples),
        "subjects": len(subjects),
        "templates": len(templates),
        "complete_subject_template_label_cross": observed_pairs == expected_pairs and len(examples) == len(expected_pairs),
        "subject_label_counts": subject_label_counts,
        "template_label_counts": template_label_counts,
        "subjects_balanced": all(counts[0] == counts[1] for counts in subject_label_counts.values()),
        "templates_balanced": all(counts[0] == counts[1] for counts in template_label_counts.values()),
    }


def name_only_baseline(
    train: Sequence[PromptExample],
    evaluation: Sequence[PromptExample],
) -> dict[str, Any]:
    """Predict from subject identity alone using training-set label frequencies."""

    if not train or not evaluation:
        raise ValueError("name-only baseline requires non-empty train and evaluation sets")
    global_counts = [sum(example.label == label for example in train) for label in (0, 1)]
    per_subject = {
        subject: [sum(example.subject == subject and example.label == label for example in train) for label in (0, 1)]
        for subject in {example.subject for example in train}
    }
    predictions = []
    seen_subjects = 0
    for example in evaluation:
        counts = per_subject.get(example.subject, global_counts)
        seen_subjects += int(example.subject in per_subject)
        predictions.append(int(counts[1] > counts[0]))
    accuracy = statistics.fmean(
        float(prediction == example.label)
        for prediction, example in zip(predictions, evaluation, strict=True)
    )
    return {
        "accuracy": accuracy,
        "seen_subject_fraction": seen_subjects / len(evaluation),
        "predictions": predictions,
        "features_used": ["subject"],
    }


def randomize_training_labels(examples: Sequence[PromptExample], *, seed: int) -> list[PromptExample]:
    generator = t.Generator().manual_seed(seed)
    labels = t.tensor([example.label for example in examples])
    shuffled = labels[t.randperm(len(labels), generator=generator)]
    if t.equal(labels, shuffled):
        shuffled = shuffled.roll(1)
    return [
        PromptExample(
            example.prompt,
            int(label),
            example.subject,
            example.template_id,
            example.marker,
            "random_label",
        )
        for example, label in zip(examples, shuffled.tolist(), strict=True)
    ]


def label_token_ids(tokenizer: Any, *, device: t.device | str) -> t.Tensor:
    ids = []
    for text in LABEL_TEXT:
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"Expected {text!r} to be one token, got {encoded}")
        ids.append(encoded[0])
    return t.tensor(ids, dtype=t.long, device=device)


def tokenize_prompts(
    tokenizer: Any,
    examples: Sequence[PromptExample],
    *,
    device: t.device | str,
) -> dict[str, t.Tensor]:
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
    return model(**batch, use_cache=False).logits[:, -1, :].index_select(-1, label_ids)


def protected_replay_batch(tokenizer: Any, *, device: t.device | str) -> tuple[dict[str, t.Tensor], t.Tensor]:
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


def optimizer_state_bytes(optimizer: t.optim.Optimizer) -> int:
    return _unique_tensor_storage_bytes(
        value
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, t.Tensor)
    )


def optimizer_state_dtypes(optimizer: t.optim.Optimizer) -> dict[str, int]:
    counts: dict[str, int] = {}
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, t.Tensor):
                key = str(value.dtype)
                counts[key] = counts.get(key, 0) + value.numel()
    return counts


def train_on_codebook(
    model: nn.Module,
    tokenizer: Any,
    examples: Sequence[PromptExample],
    *,
    steps: int,
    learning_rate: float,
    batch_size: int,
    seed: int,
    protected_replay_weight: float = 0.1,
) -> TrainTrace:
    """Use one matched optimizer loop for dense LoRA, QLoRA, and full finetuning."""

    if steps <= 0 or batch_size <= 0:
        raise ValueError("steps and batch_size must be positive")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("model has no trainable parameters")
    optimizer = t.optim.AdamW(trainable, lr=learning_rate, weight_decay=0.0)
    device = next(model.parameters()).device
    label_ids = label_token_ids(tokenizer, device=device)
    protected_batch, protected_targets = protected_replay_batch(tokenizer, device=device)
    generator = t.Generator().manual_seed(seed)
    losses: list[float] = []
    model.train()
    for _ in range(steps):
        indices = t.randint(len(examples), (batch_size,), generator=generator).tolist()
        selected = [examples[index] for index in indices]
        batch = tokenize_prompts(tokenizer, selected, device=device)
        targets = t.tensor([example.label for example in selected], dtype=t.long, device=device)
        two_label_conditional_nll = F.cross_entropy(
            two_label_logits(model, batch, label_ids).float(), targets
        )
        protected_logits = model(**protected_batch, use_cache=False).logits[:, -1, :]
        replay_loss = F.cross_entropy(protected_logits.float(), protected_targets)
        loss = two_label_conditional_nll + protected_replay_weight * replay_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        t.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        losses.append(float(loss.detach().item()))
    return TrainTrace(
        losses=losses,
        steps=steps,
        examples_seen=steps * batch_size,
        learning_rate=float(learning_rate),
        trainable_parameter_count=sum(parameter.numel() for parameter in trainable),
        trainable_parameter_bytes=sum(parameter.numel() * parameter.element_size() for parameter in trainable),
        optimizer_state_bytes=optimizer_state_bytes(optimizer),
        optimizer_state_dtypes=optimizer_state_dtypes(optimizer),
    )


@t.no_grad()
def evaluate_behavior(
    model: nn.Module,
    tokenizer: Any,
    examples: Sequence[PromptExample],
) -> BehaviorMetrics:
    device = next(model.parameters()).device
    batch = tokenize_prompts(tokenizer, examples, device=device)
    label_ids = label_token_ids(tokenizer, device=device)
    targets = t.tensor([example.label for example in examples], dtype=t.long, device=device)
    logits = two_label_logits(model, batch, label_ids).float()
    probabilities = logits.softmax(dim=-1)
    predictions = logits.argmax(dim=-1)
    return BehaviorMetrics(
        accuracy=float((predictions == targets).float().mean().item()),
        two_label_conditional_nll=float(F.cross_entropy(logits, targets).item()),
        target_probability=float(probabilities.gather(1, targets[:, None]).mean().item()),
        predictions=predictions.tolist(),
    )


@t.no_grad()
def protected_next_token_nll(model: nn.Module, tokenizer: Any) -> float:
    device = next(model.parameters()).device
    batch, targets = protected_replay_batch(tokenizer, device=device)
    logits = model(**batch, use_cache=False).logits[:, -1, :].float()
    return float(F.cross_entropy(logits, targets).item())


def summarize_spectrum(update: t.Tensor) -> SpectrumMetrics:
    values = t.linalg.svdvals(update.detach().float())
    total = values.sum().clamp_min(t.finfo(t.float32).tiny)
    probabilities = values / total
    entropy = -(probabilities * probabilities.clamp_min(t.finfo(t.float32).tiny).log()).sum()
    frobenius_sq = values.square().sum()
    stable_rank = frobenius_sq / values.square().max().clamp_min(t.finfo(t.float32).tiny)
    tolerance = max(update.shape) * t.finfo(values.dtype).eps * values.max().clamp_min(t.finfo(values.dtype).tiny)
    return SpectrumMetrics(
        singular_values=values.cpu().tolist(),
        effective_rank=float(entropy.exp().item()) if bool(values.any()) else 0.0,
        stable_rank=float(stable_rank.item()),
        update_norm=float(update.detach().float().norm().item()),
        numerical_rank=int((values > tolerance).sum().item()),
        top_singular_value_fraction=float((values[0] / total).item()),
    )


def extract_selected_updates(
    model: nn.Module,
    original_weights: dict[str, t.Tensor],
    target_modules: Sequence[str] = TARGET_MODULES,
) -> dict[str, t.Tensor]:
    """Return one materialized update matrix per selected transformer layer."""

    updates = {}
    for name in target_modules:
        parent, leaf = _get_parent_and_leaf(model, name)
        layer = getattr(parent, leaf)
        if callable(getattr(layer, "delta_weight", None)):
            update = layer.delta_weight()
        elif isinstance(layer, nn.Linear):
            update = layer.weight.float() - original_weights[name].to(layer.weight.device).float()
        else:
            raise TypeError(f"unsupported target layer {name}: {type(layer).__name__}")
        if not isinstance(update, t.Tensor) or update.shape != original_weights[name].shape:
            raise ValueError(f"update for {name} must match the original projection shape")
        if not t.isfinite(update).all():
            raise ValueError(f"update for {name} must be finite")
        updates[name] = update.float()
    return updates


def extract_selected_update(
    model: nn.Module,
    original_weights: dict[str, t.Tensor],
    target_modules: Sequence[str] = TARGET_MODULES,
) -> t.Tensor:
    """Backward-compatible stacked view; reporting uses per-layer updates."""

    return t.cat(list(extract_selected_updates(model, original_weights, target_modules).values()), dim=0)


def selected_quantization_error(
    model: nn.Module,
    original_weights: dict[str, t.Tensor],
    target_modules: Sequence[str] = TARGET_MODULES,
    *,
    dequantize_fn: Callable[[nn.Module], t.Tensor] = dequantize_bnb_weight,
    error_fn: Callable[[t.Tensor, t.Tensor], dict[str, float]] = quantization_error,
) -> dict[str, float]:
    references = []
    reconstructed = []
    for name in target_modules:
        parent, leaf = _get_parent_and_leaf(model, name)
        layer = getattr(parent, leaf)
        base = getattr(layer, "base", layer)
        references.append(original_weights[name].flatten().float().cpu())
        reconstructed.append(dequantize_fn(base).flatten().float().cpu())
    return error_fn(t.cat(references), t.cat(reconstructed))


def tensor_storage_bytes(model: nn.Module) -> int:
    """Count unique parameter, buffer, and nested quantization-state tensor storage."""

    tensors: list[t.Tensor] = list(model.parameters()) + list(model.buffers())
    for module in model.modules():
        weight = getattr(module, "weight", None)
        state = getattr(weight, "quant_state", None)
        if state is None:
            continue
        tensors.extend(value for _, value in _quant_state_tensors(state))
    return _unique_tensor_storage_bytes(tensors)


def trainable_storage_bytes(model: nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)


def gradient_storage_bytes(model: nn.Module) -> int:
    return _unique_tensor_storage_bytes(
        parameter.grad for parameter in model.parameters() if parameter.grad is not None
    )


def parameter_dtype_counts(model: nn.Module, *, trainable: bool | None = None) -> dict[str, int]:
    counts: dict[str, int] = {}
    for parameter in model.parameters():
        if trainable is not None and parameter.requires_grad is not trainable:
            continue
        key = str(parameter.dtype)
        counts[key] = counts.get(key, 0) + parameter.numel()
    return counts


def make_same_norm_random_adapter(model: nn.Module, *, target_norm: float, seed: int) -> None:
    """Fill visible adapters randomly, then match their combined update Frobenius norm."""

    generator = t.Generator(device="cpu").manual_seed(seed)
    adapters = [
        module for module in model.modules()
        if isinstance(getattr(module, "lora_a", None), nn.Parameter)
        and isinstance(getattr(module, "lora_b", None), nn.Parameter)
        and callable(getattr(module, "delta_weight", None))
    ]
    if not adapters or target_norm <= 0:
        raise ValueError("a positive target norm and at least one adapter are required")
    with t.no_grad():
        for module in adapters:
            module.lora_a.copy_(t.randn(module.lora_a.shape, generator=generator).to(module.lora_a.device))
            module.lora_b.copy_(t.randn(module.lora_b.shape, generator=generator).to(module.lora_b.device))
        current = math.sqrt(sum(float(module.delta_weight().square().sum().item()) for module in adapters))
        multiplier = target_norm / max(current, 1e-12)
        for module in adapters:
            module.lora_b.mul_(multiplier)


def quantization_stress_test(*, seed: int = 0) -> dict[str, Any]:
    """Expose the local damage caused when one outlier controls a large block's scale."""

    generator = t.Generator().manual_seed(seed)
    clean = 0.2 * t.randn(512, generator=generator)
    outlier = clean.clone()
    outlier[0] = 12.0
    result: dict[str, Any] = {}
    for name, values in (("clean", clean), ("outlier", outlier)):
        result[name] = {}
        for block_size in (8, 32, 64, 128):
            reconstructed = dequantize_nf4(quantize_nf4(values, block_size=block_size))
            metrics = quantization_error(values, reconstructed)
            metrics["non_outlier_mse"] = float((values[1:] - reconstructed[1:]).square().mean().item())
            result[name][str(block_size)] = metrics
    return result


def cpu_bnb_adapter_oracle(
    *,
    seed: int = 0,
    wrapper_cls: type[QLoRALinear] = QLoRALinear,
    storage_fn: Callable[[nn.Module], dict[str, Any]] = linear4bit_storage_accounting,
    audit_fn: Callable[..., dict[str, Any]] = audit_linear4bit,
    parity_fn: Callable[[nn.Module, t.Tensor], float] = dequantized_base_max_diff,
) -> dict[str, Any]:
    """Use an actual CPU bitsandbytes NF4 layer to test parity and adapter gradients."""

    import bitsandbytes as bnb

    t.manual_seed(seed)
    dense = nn.Linear(64, 12, bias=True)
    quantized = bnb.nn.Linear4bit(
        64,
        12,
        bias=True,
        compute_dtype=t.float32,
        compress_statistics=True,
        quant_type="nf4",
    )
    quantized.load_state_dict(dense.state_dict())
    quantized = quantized.to("cpu")
    audit = audit_fn(
        quantized,
        expected_compute_dtype=t.float32,
        expected_storage_dtype=t.uint8,
    )
    parity = parity_fn(quantized, t.randn(5, 64))
    storage = storage_fn(quantized)
    wrapped = wrapper_cls(quantized, rank=4, alpha=4.0)
    with t.no_grad():
        wrapped.lora_b.normal_(std=0.01)
    inputs = t.randn(7, 64)
    loss = wrapped(inputs).square().mean()
    loss.backward()
    return {
        "dequantized_base_max_diff": parity,
        "base_weight_has_no_gradient": wrapped.base.weight.grad is None,
        "lora_a_gradient_norm": float(wrapped.lora_a.grad.norm().item()),
        "lora_b_gradient_norm": float(wrapped.lora_b.grad.norm().item()),
        "storage": storage,
        "audit": audit,
    }


def _snapshot_target_weights(model: nn.Module) -> dict[str, t.Tensor]:
    result = {}
    for name in TARGET_MODULES:
        parent, leaf = _get_parent_and_leaf(model, name)
        result[name] = getattr(parent, leaf).weight.detach().float().cpu().clone()
    return result


def _cuda_metadata(max_vram_gb: float) -> dict[str, Any]:
    if max_vram_gb <= 0 or max_vram_gb > 24.0:
        raise ValueError("max_vram_gb must be in (0, 24]")
    if not t.cuda.is_available():
        raise RuntimeError("CUDA is required; the real QLoRA experiment has no CPU fallback")
    if not t.cuda.is_bf16_supported():
        raise RuntimeError("BF16-capable CUDA hardware is required for the matched comparison")
    device = t.device("cuda:0")
    t.cuda.empty_cache()
    t.cuda.reset_peak_memory_stats(device)
    return {"device": device, "max_vram_gb": float(max_vram_gb)}


def _load_tokenizer() -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_dense_model(*, device: t.device) -> nn.Module:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=t.bfloat16,
    ).to(device)
    model.config.use_cache = False
    return model


def intended_nf4_module_names(model: nn.Module) -> tuple[str, ...]:
    """Enumerate every transformer projection expected to be quantized."""

    blocks = getattr(getattr(model, "transformer", None), "h", None)
    if blocks is None or len(blocks) == 0:
        raise RuntimeError("the pinned GPT-Neo transformer blocks are missing")
    names = tuple(
        f"transformer.h.{index}.{suffix}"
        for index in range(len(blocks))
        for suffix in INTENDED_QUANTIZED_SUFFIXES
    )
    for name in names:
        try:
            model.get_submodule(name)
        except AttributeError as error:
            raise RuntimeError(f"intended quantized projection is missing: {name}") from error
    return names


def audit_nf4_model(
    model: nn.Module,
    intended_modules: Sequence[str],
    *,
    expected_compute_dtype: t.dtype = t.bfloat16,
    expected_storage_dtype: t.dtype = t.uint8,
) -> dict[str, Any]:
    """Audit every intended projection and every materialized Linear4bit module."""

    from bitsandbytes.nn import Linear4bit

    expected = tuple(intended_modules)
    if not expected or len(set(expected)) != len(expected):
        raise ValueError("intended_modules must be a non-empty unique sequence")
    actual = {name for name, module in model.named_modules() if isinstance(module, Linear4bit)}
    missing = [name for name in expected if name not in actual]
    if missing:
        raise RuntimeError(f"intended projections were not materialized as Linear4bit: {missing}")
    audited = {}
    for name in sorted(actual | set(expected)):
        try:
            module = model.get_submodule(name)
        except AttributeError as error:
            raise RuntimeError(f"quantized module disappeared during audit: {name}") from error
        audited[name] = audit_linear4bit(
            module,
            expected_compute_dtype=expected_compute_dtype,
            expected_storage_dtype=expected_storage_dtype,
        )
    return {
        "intended_module_count": len(expected),
        "materialized_module_count": len(actual),
        "all_intended_materialized": set(expected).issubset(actual),
        "all_nested": all(item["nested"] for item in audited.values()),
        "all_state_tensors_finite": all(item["all_state_tensors_finite"] for item in audited.values()),
        "all_dequantized_weights_finite": all(item["dequantized_weight_finite"] for item in audited.values()),
        "all_nf4_codebooks_exact": all(item["nf4_codebook_exact"] for item in audited.values()),
        "all_dynamic_codebooks_exact": all(item["dynamic_codebook_exact"] for item in audited.values()),
        "storage_dtype": str(expected_storage_dtype),
        "compute_dtype": str(expected_compute_dtype),
        "unexpected_linear4bit_modules": sorted(actual - set(expected)),
        "modules": audited,
    }


def load_nf4_model(*, device: t.device) -> nn.Module:
    """Load the pinned transformer through the strict real bitsandbytes NF4 path."""

    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    if device.type != "cuda":
        raise RuntimeError("NF4 pretrained loading is CUDA-only in this lesson; no substitute path exists")
    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=t.bfloat16,
        bnb_4bit_quant_storage=t.uint8,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        quantization_config=config,
        device_map={"": device.index or 0},
        dtype=t.bfloat16,
    )
    model.config.use_cache = False
    audit_nf4_model(
        model,
        intended_nf4_module_names(model),
        expected_compute_dtype=t.bfloat16,
        expected_storage_dtype=t.uint8,
    )
    return model


def _callable_provenance(fn: Any) -> dict[str, Any]:
    """Fingerprint visible bodies, including every locally defined class method.

    These records locate executed code; they do not prove semantic correctness
    or certify arbitrary Python against indirect delegation.
    """

    def code_payload(code: Any) -> tuple:
        constants = tuple(code_payload(value) if inspect.iscode(value) else value
                          for value in code.co_consts)
        return (code.co_code, constants, code.co_names, code.co_varnames,
                code.co_argcount, code.co_posonlyargcount, code.co_kwonlyargcount,
                code.co_freevars, code.co_cellvars, code.co_flags)

    methods = ([(name, inspect.unwrap(value)) for name, value in vars(fn).items()
                if inspect.isfunction(value)] if inspect.isclass(fn)
               else [("__call__", inspect.unwrap(fn))])
    payloads = [(name, code_payload(method.__code__)) for name, method in methods
                if hasattr(method, "__code__")]
    fingerprint = hashlib.sha256(repr(payloads).encode("utf-8")).hexdigest() if payloads else None
    return {
        "qualified_name": f"{getattr(fn, '__module__', 'unknown')}.{getattr(fn, '__qualname__', type(fn).__name__)}",
        "code_sha256": fingerprint,
        "callable": callable(fn),
        "fingerprinted_methods": [name for name, _ in payloads],
    }


@dataclass(frozen=True)
class LearnerFunctions:
    nf4_codebook: Callable[..., t.Tensor]
    expected_dynamic_codebook: Callable[..., t.Tensor]
    blockwise_absmax: Callable[..., t.Tensor]
    quantize_nf4: Callable[..., NF4Tensor]
    dequantize_nf4: Callable[..., t.Tensor]
    quantization_error: Callable[[t.Tensor, t.Tensor], dict[str, float]]
    double_quantize_scales: Callable[..., DoubleQuantizedScales]
    dequantize_scales: Callable[[DoubleQuantizedScales], t.Tensor]
    ste_nf4: Callable[..., t.Tensor]
    nf4_storage_accounting: Callable[..., dict[str, Any]]
    linear4bit_storage_accounting: Callable[[nn.Module], dict[str, Any]]
    audit_linear4bit: Callable[..., dict[str, Any]]
    qlora_linear: type[QLoRALinear]
    dequantize_bnb_weight: Callable[[nn.Module], t.Tensor]
    dequantized_base_max_diff: Callable[[nn.Module, t.Tensor], float]
    audit_nf4_model: Callable[..., dict[str, Any]]
    inject_qlora: Callable[..., dict[str, QLoRALinear]]
    make_codebook_examples: Callable[[], tuple[list[PromptExample], list[PromptExample]]]
    make_unseen_subject_examples: Callable[[], list[PromptExample]]
    make_marker_control_examples: Callable[..., dict[str, list[PromptExample]]]
    counterbalance_audit: Callable[[Sequence[PromptExample]], dict[str, Any]]
    name_only_baseline: Callable[[Sequence[PromptExample], Sequence[PromptExample]], dict[str, Any]]
    randomize_training_labels: Callable[..., list[PromptExample]]
    train_on_codebook: Callable[..., TrainTrace]
    summarize_spectrum: Callable[[t.Tensor], SpectrumMetrics]
    quantization_stress_test: Callable[..., dict[str, Any]]
    load_nf4_model: Callable[..., nn.Module]

    def provenance(self) -> dict[str, dict[str, Any]]:
        return {
            field: _callable_provenance(fn)
            for field, fn in self.__dict__.items()
        }


def reference_learner_functions() -> LearnerFunctions:
    """Return the visible solutions as a function bundle for release verification."""

    return LearnerFunctions(
        nf4_codebook=nf4_codebook,
        expected_dynamic_codebook=expected_dynamic_codebook,
        blockwise_absmax=blockwise_absmax,
        quantize_nf4=quantize_nf4,
        dequantize_nf4=dequantize_nf4,
        quantization_error=quantization_error,
        double_quantize_scales=double_quantize_scales,
        dequantize_scales=dequantize_scales,
        ste_nf4=ste_nf4,
        nf4_storage_accounting=nf4_storage_accounting,
        linear4bit_storage_accounting=linear4bit_storage_accounting,
        audit_linear4bit=audit_linear4bit,
        qlora_linear=QLoRALinear,
        dequantize_bnb_weight=dequantize_bnb_weight,
        dequantized_base_max_diff=dequantized_base_max_diff,
        audit_nf4_model=audit_nf4_model,
        inject_qlora=inject_qlora,
        make_codebook_examples=make_codebook_examples,
        make_unseen_subject_examples=make_unseen_subject_examples,
        make_marker_control_examples=make_marker_control_examples,
        counterbalance_audit=counterbalance_audit,
        name_only_baseline=name_only_baseline,
        randomize_training_labels=randomize_training_labels,
        train_on_codebook=train_on_codebook,
        summarize_spectrum=summarize_spectrum,
        quantization_stress_test=quantization_stress_test,
        load_nf4_model=load_nf4_model,
    )


def audit_implementation_provenance(
    implementation: LearnerFunctions,
    *,
    require_distinct_from_reference: bool,
) -> dict[str, Any]:
    """Record callback identity and reject obvious direct reference delegation.

    This diagnostic is deliberately bounded: dynamic dispatch can evade it.
    Numerical oracles and an executed completed learner are the semantic evidence.
    """

    if not isinstance(implementation, LearnerFunctions):
        raise TypeError("implementation must be a complete LearnerFunctions bundle")
    reference = reference_learner_functions()
    records = implementation.provenance()
    reference_ids = {id(value): name for name, value in reference.__dict__.items()}
    reference_names = {getattr(value, "__name__", "") for value in reference.__dict__.values()}

    def direct_references(value: Any) -> list[str]:
        methods = ([method for method in vars(value).values() if inspect.isfunction(method)]
                   if inspect.isclass(value) else [value])
        found: set[str] = set()
        for method in methods:
            method = inspect.unwrap(method)
            if not inspect.isfunction(method):
                continue
            closure = inspect.getclosurevars(method)
            used_values = list(closure.globals.values()) + list(closure.nonlocals.values())
            used_values += list(method.__defaults__ or ()) + list((method.__kwdefaults__ or {}).values())
            for used in used_values:
                if id(used) in reference_ids:
                    found.add(reference_ids[id(used)])
                if inspect.ismodule(used) and used.__name__ == __name__:
                    found.update(reference_names.intersection(method.__code__.co_names))
        return sorted(found)

    for field, record in records.items():
        record["is_reference_object"] = getattr(implementation, field) is getattr(reference, field)
        record["direct_reference_dependencies"] = direct_references(getattr(implementation, field))
    all_callable = all(record["callable"] for record in records.values())
    all_fingerprinted = all(record["code_sha256"] for record in records.values())
    all_distinct = all(not record["is_reference_object"] for record in records.values())
    all_reference = all(record["is_reference_object"] for record in records.values())
    identity_policy_satisfied = all_distinct if require_distinct_from_reference else all_reference
    no_direct_delegation = all(not record["direct_reference_dependencies"] for record in records.values())
    policy_satisfied = all_callable and all_fingerprinted and (
        identity_policy_satisfied and (no_direct_delegation or not require_distinct_from_reference)
    )
    return {
        "required_policy": "learner_distinct_from_reference" if require_distinct_from_reference else "visible_reference_release",
        "all_callable": all_callable,
        "all_code_fingerprints_present": all_fingerprinted,
        "all_distinct_from_reference": all_distinct,
        "all_reference_objects": all_reference,
        "no_direct_reference_delegation": no_direct_delegation,
        "scope": "Identity and direct dependencies only; not proof of implementation correctness.",
        "policy_satisfied": policy_satisfied,
        "functions": records,
    }


def _release_model(model: nn.Module | None) -> None:
    if model is not None:
        model.to("cpu")
        del model
    gc.collect()
    t.cuda.empty_cache()


def _marker_swap_pair_metrics(
    canonical_examples: Sequence[PromptExample],
    swapped_examples: Sequence[PromptExample],
    canonical_predictions: Sequence[int],
    swapped_predictions: Sequence[int],
) -> tuple[float, float]:
    flip_rate = statistics.fmean(
        float(canonical != swapped)
        for canonical, swapped in zip(canonical_predictions, swapped_predictions, strict=True)
    )
    pair_accuracy = statistics.fmean(
        float(canonical == source.label and swapped == counterfactual.label)
        for source, counterfactual, canonical, swapped in zip(
            canonical_examples,
            swapped_examples,
            canonical_predictions,
            swapped_predictions,
            strict=True,
        )
    )
    return flip_rate, pair_accuracy


def _method_result(
    *,
    method: str,
    implementation: LearnerFunctions,
    tokenizer: Any,
    train: Sequence[PromptExample],
    heldout: Sequence[PromptExample],
    ood: Sequence[PromptExample],
    marker_controls: dict[str, Sequence[PromptExample]],
    original_weights: dict[str, t.Tensor],
    base_protected_nll: float,
    device: t.device,
    steps: int,
    batch_size: int,
    seed: int,
    rank: int | None = None,
    random_labels: bool = False,
    same_norm_target: float | None = None,
) -> dict[str, Any]:
    t.manual_seed(seed)
    t.cuda.manual_seed_all(seed)
    t.cuda.empty_cache()
    t.cuda.reset_peak_memory_stats(device)
    quantization_audit = None
    if method == "qlora":
        model = implementation.load_nf4_model(device=device)
        quantization_audit = implementation.audit_nf4_model(
            model,
            intended_nf4_module_names(model),
            expected_compute_dtype=t.bfloat16,
            expected_storage_dtype=t.uint8,
        )
        assert rank is not None
        adapters = implementation.inject_qlora(model, TARGET_MODULES, rank=rank, alpha=float(rank))
        quant_error = selected_quantization_error(
            model,
            original_weights,
            dequantize_fn=implementation.dequantize_bnb_weight,
            error_fn=implementation.quantization_error,
        )
        parity = max(
            implementation.dequantized_base_max_diff(
                module.base,
                t.randn(3, module.base.in_features, device=device, dtype=t.bfloat16),
            )
            for module in adapters.values()
        )
        learning_rate = ADAPTER_LEARNING_RATE
    elif method == "dense_lora":
        model = load_dense_model(device=device)
        assert rank is not None
        inject_dense_lora(model, TARGET_MODULES, rank=rank, alpha=float(rank))
        quant_error = None
        parity = None
        learning_rate = ADAPTER_LEARNING_RATE
    elif method == "full":
        model = load_dense_model(device=device)
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        quant_error = None
        parity = None
        learning_rate = FULL_FINETUNE_LEARNING_RATE
    else:
        raise ValueError(f"unknown method: {method}")

    model_bytes = tensor_storage_bytes(model)
    trainable_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    trainable_bytes = trainable_storage_bytes(model)
    trainable_dtypes = parameter_dtype_counts(model, trainable=True)
    frozen_dtypes = parameter_dtype_counts(model, trainable=False)
    trace = None
    same_norm = None
    if same_norm_target is not None:
        make_same_norm_random_adapter(model, target_norm=same_norm_target, seed=seed)
        actual_norm = math.sqrt(
            sum(float(update.square().sum().item()) for update in extract_selected_updates(model, original_weights).values())
        )
        same_norm = {
            "target": float(same_norm_target),
            "actual": actual_norm,
            "relative_error": abs(actual_norm - same_norm_target) / max(abs(same_norm_target), 1e-12),
        }
    else:
        # Module-specific parity checks can consume RNG; reset before matched dropout and batches.
        t.manual_seed(seed + 10_000)
        t.cuda.manual_seed_all(seed + 10_000)
        selected_train = implementation.randomize_training_labels(train, seed=seed + 1000) if random_labels else train
        trace = implementation.train_on_codebook(
            model,
            tokenizer,
            selected_train,
            steps=steps,
            learning_rate=learning_rate,
            batch_size=batch_size,
            seed=seed,
        )
        if (
            trace.learning_rate != learning_rate
            or trace.trainable_parameter_count != trainable_count
            or trace.trainable_parameter_bytes != trainable_bytes
        ):
            raise RuntimeError("learner training metadata disagrees with the measured trainable surface")
    model.eval()
    behavior = evaluate_behavior(model, tokenizer, heldout)
    ood_behavior = evaluate_behavior(model, tokenizer, ood)
    marker_swap_behavior = evaluate_behavior(model, tokenizer, marker_controls["marker_swap"])
    marker_removed_behavior = evaluate_behavior(model, tokenizer, marker_controls["marker_removed"])
    marker_swap_flip_rate, marker_swap_pair_accuracy = _marker_swap_pair_metrics(
        heldout,
        marker_controls["marker_swap"],
        behavior.predictions,
        marker_swap_behavior.predictions,
    )
    protected_nll = protected_next_token_nll(model, tokenizer)
    updates = extract_selected_updates(model, original_weights)
    spectra = {name: asdict(implementation.summarize_spectrum(update)) for name, update in updates.items()}
    combined_update_norm = math.sqrt(sum(float(update.square().sum().item()) for update in updates.values()))
    t.cuda.synchronize(device)
    peak_vram_gb = t.cuda.max_memory_allocated(device) / 2**30
    gradient_bytes = gradient_storage_bytes(model)
    optimizer_bytes = 0 if trace is None else trace.optimizer_state_bytes
    resident_training_state_bytes = model_bytes + gradient_bytes + optimizer_bytes
    result = {
        "seed": seed,
        "behavior": asdict(behavior),
        "ood_behavior": asdict(ood_behavior),
        "marker_swap_behavior": asdict(marker_swap_behavior),
        "marker_swap_flip_rate": marker_swap_flip_rate,
        "marker_swap_pair_accuracy": marker_swap_pair_accuracy,
        "marker_removed_behavior": asdict(marker_removed_behavior),
        "per_layer_spectra": spectra,
        "combined_update_norm": combined_update_norm,
        "model_tensor_bytes": model_bytes,
        "trainable_parameter_bytes": trainable_bytes,
        "gradient_bytes": gradient_bytes,
        "optimizer_state_bytes": optimizer_bytes,
        "resident_training_state_bytes": resident_training_state_bytes,
        "model_memory_mib": model_bytes / 2**20,
        "trainable_memory_mib": trainable_bytes / 2**20,
        "gradient_memory_mib": gradient_bytes / 2**20,
        "optimizer_state_mib": optimizer_bytes / 2**20,
        "resident_training_state_mib": resident_training_state_bytes / 2**20,
        "resident_training_state_components": {
            "model_tensors_including_trainable_parameters": model_bytes,
            "gradients": gradient_bytes,
            "optimizer_state": optimizer_bytes,
        },
        "optimizer_state_dtypes": {} if trace is None else trace.optimizer_state_dtypes,
        "peak_vram_gb": peak_vram_gb,
        "learning_rate": None if trace is None else trace.learning_rate,
        "trainable_parameter_count": trainable_count,
        "trainable_parameter_dtypes": trainable_dtypes,
        "frozen_parameter_dtypes": frozen_dtypes,
        "protected_nll": protected_nll,
        "protected_nll_change": protected_nll - base_protected_nll,
        "final_training_loss": None if trace is None else trace.losses[-1],
        "training_steps": 0 if trace is None else trace.steps,
        "examples_seen": 0 if trace is None else trace.examples_seen,
        "quantization_error": quant_error,
        "quantization_audit": quantization_audit,
        "dequantized_base_max_diff": parity,
        "same_norm": same_norm,
        "implementation_provenance": implementation.provenance(),
    }
    _release_model(model)
    return result


def _nf4_base_result(
    *,
    implementation: LearnerFunctions,
    tokenizer: Any,
    heldout: Sequence[PromptExample],
    ood: Sequence[PromptExample],
    marker_controls: dict[str, Sequence[PromptExample]],
    original_weights: dict[str, t.Tensor],
    base_protected_nll: float,
    device: t.device,
    seed: int,
) -> dict[str, Any]:
    """Measure quantization alone before adding or training an adapter."""

    t.cuda.empty_cache()
    t.cuda.reset_peak_memory_stats(device)
    model = implementation.load_nf4_model(device=device)
    quantization_audit = implementation.audit_nf4_model(
        model,
        intended_nf4_module_names(model),
        expected_compute_dtype=t.bfloat16,
        expected_storage_dtype=t.uint8,
    )
    behavior = evaluate_behavior(model, tokenizer, heldout)
    ood_behavior = evaluate_behavior(model, tokenizer, ood)
    marker_swap_behavior = evaluate_behavior(model, tokenizer, marker_controls["marker_swap"])
    marker_removed_behavior = evaluate_behavior(model, tokenizer, marker_controls["marker_removed"])
    marker_swap_flip_rate, marker_swap_pair_accuracy = _marker_swap_pair_metrics(
        heldout,
        marker_controls["marker_swap"],
        behavior.predictions,
        marker_swap_behavior.predictions,
    )
    protected_nll = protected_next_token_nll(model, tokenizer)
    parity = []
    for name in TARGET_MODULES:
        parent, leaf = _get_parent_and_leaf(model, name)
        layer = getattr(parent, leaf)
        inputs = t.randn(3, layer.in_features, device=device, dtype=t.bfloat16)
        parity.append(implementation.dequantized_base_max_diff(layer, inputs))
    model_bytes = tensor_storage_bytes(model)
    result = {
        "seed": seed,
        "behavior": asdict(behavior),
        "ood_behavior": asdict(ood_behavior),
        "marker_swap_behavior": asdict(marker_swap_behavior),
        "marker_swap_flip_rate": marker_swap_flip_rate,
        "marker_swap_pair_accuracy": marker_swap_pair_accuracy,
        "marker_removed_behavior": asdict(marker_removed_behavior),
        "per_layer_spectra": None,
        "combined_update_norm": 0.0,
        "model_tensor_bytes": model_bytes,
        "trainable_parameter_bytes": 0,
        "gradient_bytes": 0,
        "optimizer_state_bytes": 0,
        "resident_training_state_bytes": model_bytes,
        "model_memory_mib": model_bytes / 2**20,
        "trainable_memory_mib": 0.0,
        "gradient_memory_mib": 0.0,
        "optimizer_state_mib": 0.0,
        "resident_training_state_mib": model_bytes / 2**20,
        "resident_training_state_components": {
            "model_tensors_including_trainable_parameters": model_bytes,
            "gradients": 0,
            "optimizer_state": 0,
        },
        "optimizer_state_dtypes": {},
        "peak_vram_gb": 0.0,
        "learning_rate": None,
        "trainable_parameter_count": 0,
        "trainable_parameter_dtypes": {},
        "frozen_parameter_dtypes": parameter_dtype_counts(model, trainable=False),
        "protected_nll": protected_nll,
        "protected_nll_change": protected_nll - base_protected_nll,
        "final_training_loss": None,
        "training_steps": 0,
        "examples_seen": 0,
        "quantization_error": selected_quantization_error(
            model,
            original_weights,
            dequantize_fn=implementation.dequantize_bnb_weight,
            error_fn=implementation.quantization_error,
        ),
        "quantization_audit": quantization_audit,
        "dequantized_base_max_diff": max(parity),
        "same_norm": None,
        "implementation_provenance": implementation.provenance(),
    }
    t.cuda.synchronize(device)
    result["peak_vram_gb"] = t.cuda.max_memory_allocated(device) / 2**30
    _release_model(model)
    return result


def _dense_base_result(
    *,
    model: nn.Module,
    tokenizer: Any,
    heldout: Sequence[PromptExample],
    ood: Sequence[PromptExample],
    marker_controls: dict[str, Sequence[PromptExample]],
    seed: int,
) -> tuple[dict[str, Any], dict[str, t.Tensor], float]:
    original_weights = _snapshot_target_weights(model)
    behavior = evaluate_behavior(model, tokenizer, heldout)
    ood_behavior = evaluate_behavior(model, tokenizer, ood)
    marker_swap_behavior = evaluate_behavior(model, tokenizer, marker_controls["marker_swap"])
    marker_removed_behavior = evaluate_behavior(model, tokenizer, marker_controls["marker_removed"])
    marker_swap_flip_rate, marker_swap_pair_accuracy = _marker_swap_pair_metrics(
        heldout,
        marker_controls["marker_swap"],
        behavior.predictions,
        marker_swap_behavior.predictions,
    )
    protected_nll = protected_next_token_nll(model, tokenizer)
    model_bytes = tensor_storage_bytes(model)
    return (
        {
            "seed": seed,
            "behavior": asdict(behavior),
            "ood_behavior": asdict(ood_behavior),
            "marker_swap_behavior": asdict(marker_swap_behavior),
            "marker_swap_flip_rate": marker_swap_flip_rate,
            "marker_swap_pair_accuracy": marker_swap_pair_accuracy,
            "marker_removed_behavior": asdict(marker_removed_behavior),
            "per_layer_spectra": None,
            "combined_update_norm": 0.0,
            "model_tensor_bytes": model_bytes,
            "trainable_parameter_bytes": 0,
            "gradient_bytes": 0,
            "optimizer_state_bytes": 0,
            "resident_training_state_bytes": model_bytes,
            "model_memory_mib": model_bytes / 2**20,
            "trainable_memory_mib": 0.0,
            "gradient_memory_mib": 0.0,
            "optimizer_state_mib": 0.0,
            "resident_training_state_mib": model_bytes / 2**20,
            "resident_training_state_components": {
                "model_tensors_including_trainable_parameters": model_bytes,
                "gradients": 0,
                "optimizer_state": 0,
            },
            "optimizer_state_dtypes": {},
            "peak_vram_gb": 0.0,
            "learning_rate": None,
            "trainable_parameter_count": 0,
            "trainable_parameter_dtypes": {},
            "frozen_parameter_dtypes": parameter_dtype_counts(model),
            "protected_nll": protected_nll,
            "protected_nll_change": 0.0,
            "final_training_loss": None,
            "training_steps": 0,
            "examples_seen": 0,
            "quantization_error": None,
            "quantization_audit": None,
            "dequantized_base_max_diff": None,
            "same_norm": None,
        },
        original_weights,
        protected_nll,
    )


def _run_single_seed_comparison(
    *,
    implementation: LearnerFunctions,
    max_vram_gb: float,
    steps: int,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    device = _cuda_metadata(max_vram_gb)["device"]
    t.manual_seed(seed)
    tokenizer = _load_tokenizer()
    train, heldout = implementation.make_codebook_examples()
    ood = implementation.make_unseen_subject_examples()
    marker_controls = implementation.make_marker_control_examples(heldout)

    dense_base = load_dense_model(device=device)
    base_result, original_weights, base_protected_nll = _dense_base_result(
        model=dense_base,
        tokenizer=tokenizer,
        heldout=heldout,
        ood=ood,
        marker_controls=marker_controls,
        seed=seed,
    )
    _release_model(dense_base)

    methods: dict[str, dict[str, Any]] = {"base": base_result}
    methods["nf4_base"] = _nf4_base_result(
        implementation=implementation, tokenizer=tokenizer, heldout=heldout, ood=ood,
        marker_controls=marker_controls,
        original_weights=original_weights,
        base_protected_nll=base_protected_nll,
        device=device, seed=seed,
    )
    methods["qlora_r2"] = _method_result(
        method="qlora", implementation=implementation, tokenizer=tokenizer, train=train, heldout=heldout, ood=ood,
        marker_controls=marker_controls,
        original_weights=original_weights, base_protected_nll=base_protected_nll,
        device=device, steps=steps, batch_size=batch_size, seed=seed, rank=2,
    )
    methods["qlora_r8"] = _method_result(
        method="qlora", implementation=implementation, tokenizer=tokenizer, train=train, heldout=heldout, ood=ood,
        marker_controls=marker_controls,
        original_weights=original_weights, base_protected_nll=base_protected_nll,
        device=device, steps=steps, batch_size=batch_size, seed=seed, rank=8,
    )
    methods["bf16_lora_r8"] = _method_result(
        method="dense_lora", implementation=implementation, tokenizer=tokenizer, train=train, heldout=heldout, ood=ood,
        marker_controls=marker_controls,
        original_weights=original_weights, base_protected_nll=base_protected_nll,
        device=device, steps=steps, batch_size=batch_size, seed=seed, rank=8,
    )
    methods["full_bf16"] = _method_result(
        method="full", implementation=implementation, tokenizer=tokenizer, train=train, heldout=heldout, ood=ood,
        marker_controls=marker_controls,
        original_weights=original_weights, base_protected_nll=base_protected_nll,
        device=device, steps=steps, batch_size=batch_size, seed=seed,
    )
    methods["random_labels"] = _method_result(
        method="qlora", implementation=implementation, tokenizer=tokenizer, train=train, heldout=heldout, ood=ood,
        marker_controls=marker_controls,
        original_weights=original_weights, base_protected_nll=base_protected_nll,
        device=device, steps=steps, batch_size=batch_size, seed=seed, rank=8,
        random_labels=True,
    )
    target_norm = methods["qlora_r8"]["combined_update_norm"]
    methods["same_norm_random"] = _method_result(
        method="qlora", implementation=implementation, tokenizer=tokenizer, train=train, heldout=heldout, ood=ood,
        marker_controls=marker_controls,
        original_weights=original_weights, base_protected_nll=base_protected_nll,
        device=device, steps=steps, batch_size=batch_size, seed=seed, rank=8,
        same_norm_target=target_norm,
    )

    return {
        "seed": seed,
        "methods": methods,
        "dataset_audit": {
            "train": implementation.counterbalance_audit(train),
            "template_transfer": implementation.counterbalance_audit(heldout),
            "unseen_subject_ood": implementation.counterbalance_audit(ood),
            "name_only_template_transfer": implementation.name_only_baseline(train, heldout),
            "name_only_unseen_subject_ood": implementation.name_only_baseline(train, ood),
        },
        "peak_vram_gb": max(values["peak_vram_gb"] for values in methods.values()),
    }


def _summary_stats(values: Sequence[float]) -> dict[str, float | int | str]:
    numeric = [float(value) for value in values]
    if len(numeric) != 3:
        raise ValueError("release uncertainty is defined for exactly three measured seed runs")
    mean = statistics.fmean(numeric)
    std = statistics.stdev(numeric)
    t_critical_95_df2 = 4.302652729911275
    return {
        "n": len(numeric),
        "mean": mean,
        "sample_std": std,
        "t_critical_95": t_critical_95_df2,
        "ci95_half_width": t_critical_95_df2 * std / math.sqrt(len(numeric)),
        "uncertainty_label": "two-sided 95% Student-t interval across three seeds (df=2)",
        "min": min(numeric),
        "max": max(numeric),
    }


def _constant_across_seeds(values: Sequence[Any], *, label: str) -> Any:
    first = values[0]
    if any(value != first for value in values[1:]):
        raise RuntimeError(f"{label} changed across seeds: {values}")
    return first


def _tail_energy_fraction(singular_values: Sequence[float], *, rank: int) -> float:
    squared = np.square(np.asarray(singular_values, dtype=np.float64))
    return float(squared[rank:].sum() / max(float(squared.sum()), 1e-30))


def summarize_seed_runs(seed_runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate all reported quantities without discarding per-seed evidence."""

    names = tuple(seed_runs[0]["methods"])
    summary: dict[str, Any] = {}
    for name in names:
        results = [run["methods"][name] for run in seed_runs]
        method_summary: dict[str, Any] = {
            "template_transfer": {
                metric: _summary_stats([result["behavior"][metric] for result in results])
                for metric in ("accuracy", "two_label_conditional_nll", "target_probability")
            },
            "unseen_subject_ood": {
                metric: _summary_stats([result["ood_behavior"][metric] for result in results])
                for metric in ("accuracy", "two_label_conditional_nll", "target_probability")
            },
            "marker_swap": {
                metric: _summary_stats([result["marker_swap_behavior"][metric] for result in results])
                for metric in ("accuracy", "two_label_conditional_nll", "target_probability")
            },
            "marker_removed": {
                metric: _summary_stats([result["marker_removed_behavior"][metric] for result in results])
                for metric in ("accuracy", "two_label_conditional_nll", "target_probability")
            },
            "marker_swap_flip_rate": _summary_stats(
                [result["marker_swap_flip_rate"] for result in results]
            ),
            "marker_swap_pair_accuracy": _summary_stats(
                [result["marker_swap_pair_accuracy"] for result in results]
            ),
            "model_memory_mib": _summary_stats([result["model_memory_mib"] for result in results]),
            "trainable_memory_mib": _summary_stats([result["trainable_memory_mib"] for result in results]),
            "gradient_memory_mib": _summary_stats([result["gradient_memory_mib"] for result in results]),
            "optimizer_state_mib": _summary_stats([result["optimizer_state_mib"] for result in results]),
            "resident_training_state_mib": _summary_stats(
                [result["resident_training_state_mib"] for result in results]
            ),
            "peak_vram_gb": _summary_stats([result["peak_vram_gb"] for result in results]),
            "protected_nll_change": _summary_stats([result["protected_nll_change"] for result in results]),
            "trainable_parameter_count": _constant_across_seeds(
                [result["trainable_parameter_count"] for result in results], label=f"{name} trainable count"
            ),
            "learning_rate": _constant_across_seeds(
                [result["learning_rate"] for result in results], label=f"{name} learning rate"
            ),
            "trainable_parameter_dtypes": _constant_across_seeds(
                [result["trainable_parameter_dtypes"] for result in results], label=f"{name} trainable dtypes"
            ),
            "optimizer_state_dtypes": _constant_across_seeds(
                [result["optimizer_state_dtypes"] for result in results], label=f"{name} optimizer dtypes"
            ),
        }
        if results[0]["per_layer_spectra"] is not None:
            per_layer = {}
            for layer_name in TARGET_MODULES:
                spectra = [result["per_layer_spectra"][layer_name] for result in results]
                singular_values = np.stack([item["singular_values"] for item in spectra])
                per_layer[layer_name] = {
                    "singular_values_mean": singular_values.mean(axis=0).tolist(),
                    "effective_rank": _summary_stats([item["effective_rank"] for item in spectra]),
                    "stable_rank": _summary_stats([item["stable_rank"] for item in spectra]),
                    "numerical_rank": _summary_stats([item["numerical_rank"] for item in spectra]),
                    "update_norm": _summary_stats([item["update_norm"] for item in spectra]),
                    "energy_beyond_rank_8": _summary_stats(
                        [_tail_energy_fraction(item["singular_values"], rank=8) for item in spectra]
                    ),
                }
            method_summary["per_layer_spectra"] = per_layer
        else:
            method_summary["per_layer_spectra"] = None
        summary[name] = method_summary
    return summary


def run_real_comparison(
    *,
    implementation: LearnerFunctions,
    max_vram_gb: float,
    steps: int,
    batch_size: int,
    seeds: Sequence[int] = DEFAULT_EXPERIMENT_SEEDS,
    make_figure: bool = True,
    require_distinct_from_reference: bool = True,
) -> dict[str, Any]:
    provenance_audit = audit_implementation_provenance(
        implementation,
        require_distinct_from_reference=require_distinct_from_reference,
    )
    if not provenance_audit["policy_satisfied"]:
        raise RuntimeError("implementation provenance does not satisfy the release policy")
    metadata = _cuda_metadata(max_vram_gb)
    start = time.perf_counter()
    registered_seeds = tuple(int(seed) for seed in seeds)
    if len(registered_seeds) != 3 or len(set(registered_seeds)) != 3:
        raise ValueError("the final comparison requires exactly three unique preregistered seeds")
    seed_runs = [
        _run_single_seed_comparison(
            implementation=implementation,
            max_vram_gb=max_vram_gb,
            steps=steps,
            batch_size=batch_size,
            seed=seed,
        )
        for seed in registered_seeds
    ]
    peak_vram_gb = max(run["peak_vram_gb"] for run in seed_runs)
    if peak_vram_gb > max_vram_gb:
        raise RuntimeError(f"measured {peak_vram_gb:.3f} GiB exceeds {max_vram_gb:.3f} GiB")
    train, heldout = implementation.make_codebook_examples()
    ood = implementation.make_unseen_subject_examples()
    marker_controls = implementation.make_marker_control_examples(heldout)
    measured_seeds = [int(run["seed"]) for run in seed_runs]
    result = {
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "parameters": 3_745_984},
        "dataset": {
            "train_examples": len(train),
            "template_transfer_examples": len(heldout),
            "unseen_subject_ood_examples": len(ood),
            "marker_swap_examples": len(marker_controls["marker_swap"]),
            "marker_removed_examples": len(marker_controls["marker_removed"]),
            "train_templates": len({example.template_id for example in train}),
            "template_transfer_templates": len({example.template_id for example in heldout}),
            "ood_templates": len({example.template_id for example in ood}),
            "counterbalance": {
                "train": implementation.counterbalance_audit(train),
                "template_transfer": implementation.counterbalance_audit(heldout),
                "unseen_subject_ood": implementation.counterbalance_audit(ood),
            },
            "name_only_baselines": {
                "template_transfer": implementation.name_only_baseline(train, heldout),
                "unseen_subject_ood": implementation.name_only_baseline(train, ood),
            },
        },
        "training": {
            "steps": steps,
            "batch_size": batch_size,
            "seeds": list(registered_seeds),
            "measured_seeds": measured_seeds,
            "learning_rates": {
                "qlora": ADAPTER_LEARNING_RATE,
                "bf16_lora": ADAPTER_LEARNING_RATE,
                "full_bf16": FULL_FINETUNE_LEARNING_RATE,
            },
        },
        "seed_runs": seed_runs,
        "summary": summarize_seed_runs(seed_runs),
        "quantization_stress": implementation.quantization_stress_test(seed=registered_seeds[0]),
        "implementation_provenance": provenance_audit,
        "execution": {
            "device_type": metadata["device"].type,
            "device_index": metadata["device"].index,
            "max_vram_gb": metadata["max_vram_gb"],
        },
        "gpu_name": t.cuda.get_device_name(metadata["device"]),
        "torch_version": t.__version__,
        "cuda_version": t.version.cuda,
        "bitsandbytes_version": _bitsandbytes_version(),
        "peak_vram_gb": peak_vram_gb,
        "wall_clock_seconds": time.perf_counter() - start,
        "within_vram_budget": peak_vram_gb <= metadata["max_vram_gb"],
    }
    if make_figure:
        asset = Path(__file__).resolve().parents[2] / "instructions/assets/qlora_real_signature_panel.png"
        plot_real_signature(result, asset)
        result["signature_asset"] = str(asset)
    return result


def _bitsandbytes_version() -> str:
    import bitsandbytes

    return bitsandbytes.__version__


def plot_real_signature(result: dict[str, Any], output_path: Path) -> None:
    """Render multi-seed behavior, memory, spectra, and quantization evidence."""

    names = ["base", "nf4_base", "qlora_r2", "qlora_r8", "bf16_lora_r8", "full_bf16", "random_labels", "same_norm_random"]
    labels = ["BF16 base", "NF4 base", "QLoRA r=2", "QLoRA r=8", "BF16 LoRA r=8", "Full BF16", "Random labels", "Random adapter"]
    colors = ["#5B6573", "#7A8797", "#007C91", "#00A36C", "#3E7CB1", "#D1495B", "#8D6A9F", "#C58B2A"]
    summary = result["summary"]
    fig, axes = plt.subplots(3, 4, figsize=(21, 13), constrained_layout=True)

    accuracy = [summary[name]["template_transfer"]["accuracy"]["mean"] for name in names]
    accuracy_ci = [summary[name]["template_transfer"]["accuracy"]["ci95_half_width"] for name in names]
    axes[0, 0].bar(labels, accuracy, yerr=accuracy_ci, color=colors, capsize=2)
    axes[0, 0].axhline(0.5, color="black", linestyle="--", linewidth=1, label="chance")
    axes[0, 0].set_ylim(0, 1.05)
    axes[0, 0].set_ylabel("Accuracy (mean and 95% t interval, n=3)")
    axes[0, 0].set_title("Same-subject template transfer")
    axes[0, 0].legend(frameon=False)
    axes[0, 0].tick_params(axis="x", rotation=24)

    nll = [summary[name]["template_transfer"]["two_label_conditional_nll"]["mean"] for name in names]
    nll_ci = [summary[name]["template_transfer"]["two_label_conditional_nll"]["ci95_half_width"] for name in names]
    axes[0, 1].bar(labels, nll, yerr=nll_ci, color=colors, capsize=2)
    axes[0, 1].set_ylabel("Two-label conditional NLL (95% t interval)")
    axes[0, 1].set_title("Template-transfer conditional loss")
    axes[0, 1].tick_params(axis="x", rotation=24)

    ood_accuracy = [summary[name]["unseen_subject_ood"]["accuracy"]["mean"] for name in names]
    ood_ci = [summary[name]["unseen_subject_ood"]["accuracy"]["ci95_half_width"] for name in names]
    axes[0, 2].bar(labels, ood_accuracy, yerr=ood_ci, color=colors, capsize=2)
    axes[0, 2].axhline(0.5, color="black", linestyle="--", linewidth=1)
    axes[0, 2].set_ylim(0, 1.05)
    axes[0, 2].set_ylabel("Accuracy (mean and 95% t interval, n=3)")
    axes[0, 2].set_title("Unseen-subject OOD diagnostic")
    axes[0, 2].tick_params(axis="x", rotation=24)

    trained = ["qlora_r2", "qlora_r8", "bf16_lora_r8", "full_bf16"]
    x = np.arange(len(trained))
    control_labels = ["canonical", "paired swap flip", "marker removed", "name only"]
    for offset, method in zip((-0.18, 0.18), ("qlora_r8", "bf16_lora_r8"), strict=True):
        values = [
            summary[method]["template_transfer"]["accuracy"]["mean"],
            summary[method]["marker_swap_flip_rate"]["mean"],
            summary[method]["marker_removed"]["accuracy"]["mean"],
            result["dataset"]["name_only_baselines"]["template_transfer"]["accuracy"],
        ]
        axes[0, 3].bar(
            np.arange(len(control_labels)) + offset,
            values,
            width=0.36,
            label=labels[names.index(method)],
            color=colors[names.index(method)],
        )
    axes[0, 3].axhline(0.5, color="black", linestyle="--", linewidth=1)
    axes[0, 3].set_xticks(np.arange(len(control_labels)), control_labels, rotation=18)
    axes[0, 3].set_ylim(0, 1.05)
    axes[0, 3].set_ylabel("Accuracy / paired prediction-flip rate")
    axes[0, 3].set_title("Identity and marker counterfactuals")
    axes[0, 3].legend(frameon=False, fontsize=8)

    model_state = np.asarray([summary[n]["model_memory_mib"]["mean"] for n in trained])
    gradient_state = np.asarray([summary[n]["gradient_memory_mib"]["mean"] for n in trained])
    optimizer_state = np.asarray([summary[n]["optimizer_state_mib"]["mean"] for n in trained])
    axes[1, 0].bar(x, model_state, 0.65, label="model tensors incl. trainable")
    axes[1, 0].bar(x, gradient_state, 0.65, bottom=model_state, label="gradients")
    axes[1, 0].bar(x, optimizer_state, 0.65, bottom=model_state + gradient_state, label="AdamW state")
    axes[1, 0].set_xticks(x, [labels[names.index(n)] for n in trained], rotation=18, ha="right")
    axes[1, 0].set_ylabel("Measured MiB")
    axes[1, 0].set_title("Total resident training state")
    axes[1, 0].legend(frameon=False)

    axes[1, 1].bar(
        [labels[names.index(n)] for n in trained],
        [summary[n]["peak_vram_gb"]["mean"] for n in trained],
        yerr=[summary[n]["peak_vram_gb"]["ci95_half_width"] for n in trained],
        color=[colors[names.index(n)] for n in trained],
        capsize=2,
    )
    axes[1, 1].set_ylabel("Peak allocated GiB")
    axes[1, 1].set_title("Peak CUDA allocation")
    axes[1, 1].tick_params(axis="x", rotation=18)

    for axis, layer_name, short_name in zip(axes[1, 2:4], TARGET_MODULES, ("q_proj", "v_proj"), strict=True):
        for name in [*trained, "same_norm_random"]:
            singular = np.asarray(summary[name]["per_layer_spectra"][layer_name]["singular_values_mean"])
            singular = singular / max(float(singular.max()), 1e-12)
            axis.plot(np.arange(1, len(singular) + 1), singular, marker="o", markersize=2,
                      linestyle="--" if name == "same_norm_random" else "-",
                      label=labels[names.index(name)])
        axis.set_yscale("log")
        axis.set_xlabel("Singular-value index")
        axis.set_ylabel("Normalized singular value")
        axis.set_title(f"Per-layer update spectrum: {short_name}")
    axes[1, 2].legend(frameon=False, fontsize=7)

    qnames = ["nf4_base", "qlora_r2", "qlora_r8", "random_labels", "same_norm_random"]
    first_run = result["seed_runs"][0]["methods"]
    axes[2, 0].bar(
        [labels[names.index(n)] for n in qnames],
        [first_run[n]["quantization_error"]["mse"] for n in qnames],
        color=[colors[names.index(n)] for n in qnames],
    )
    axes[2, 0].set_ylabel("Selected-weight MSE")
    axes[2, 0].set_title("Frozen NF4 base reconstruction")
    axes[2, 0].axhline(0, color="black", linestyle="--", label="unquantized weight error = 0")
    axes[2, 0].legend(frameon=False, fontsize=7)
    axes[2, 0].tick_params(axis="x", rotation=18)

    axes[2, 1].bar(
        labels,
        [summary[n]["protected_nll_change"]["mean"] for n in names],
        yerr=[summary[n]["protected_nll_change"]["ci95_half_width"] for n in names],
        color=colors,
        capsize=2,
    )
    axes[2, 1].axhline(0, color="black", linewidth=1)
    axes[2, 1].set_ylabel("Protected-prompt NLL change")
    axes[2, 1].set_title("Side effects and negative controls")
    axes[2, 1].tick_params(axis="x", rotation=24)

    storage = first_run["qlora_r8"]["quantization_audit"]["modules"][TARGET_MODULES[0]]["storage"]
    paper_bytes = storage["original_numel"] * storage["paper_amortized_double_bits_per_parameter"] / 8
    axes[2, 2].bar(
        ["paper amortized", "tail-aware formula", "actual tensors"],
        [paper_bytes, storage["tail_aware_formula_bytes_excluding_codebooks"], storage["actual_tensor_bytes_including_codebooks"]],
        color=["#5B6573", "#007C91", "#D1495B"],
    )
    axes[2, 2].set_ylabel("Bytes for one target Linear4bit")
    axes[2, 2].set_title("Formula vs materialized state")
    axes[2, 2].tick_params(axis="x", rotation=18)

    counts = [summary[n]["trainable_parameter_count"] for n in trained]
    axes[2, 3].bar([labels[names.index(n)] for n in trained], counts, color=[colors[names.index(n)] for n in trained])
    axes[2, 3].set_yscale("log")
    axes[2, 3].set_ylabel("Exact trainable parameters (log scale)")
    axes[2, 3].set_title("Trainable surface and reported LR")
    axes[2, 3].tick_params(axis="x", rotation=18)
    for index, name in enumerate(trained):
        axes[2, 3].text(index, counts[index], f"lr={summary[name]['learning_rate']:.0e}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("[15.3] QLoRA: measured three-seed behavior, state, and mechanism audit", fontsize=16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_toy_signature(result: dict[str, Any], output_path: Path) -> None:
    """Render exact NF4, nested-scale accounting, and outlier stress evidence."""

    codebook = np.asarray(result["nf4_codebook"])
    original = np.asarray(result["toy_values"])
    reconstructed = np.asarray(result["toy_reconstructed"])
    stress = result["quantization_stress"]
    blocks = [8, 32, 64, 128]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    axes[0, 0].stem(np.arange(16), codebook, basefmt=" ")
    axes[0, 0].axhline(0, color="black", linewidth=0.8)
    axes[0, 0].set_xlabel("4-bit code")
    axes[0, 0].set_ylabel("Normalized value")
    axes[0, 0].set_title("NF4 is a lookup table, not a tiny float")

    axes[0, 1].plot(original, "o-", label="original", color="#3E7CB1")
    axes[0, 1].plot(reconstructed, "s--", label="dequantized", color="#D1495B")
    axes[0, 1].set_xlabel("Weight index")
    axes[0, 1].set_ylabel("Value")
    axes[0, 1].set_title(f"NF4 reconstruction (MSE={result['toy_error']['mse']:.4g})")
    axes[0, 1].legend(frameon=False)

    accounting = result["storage_accounting"]
    actual_storage = result["bnb_cpu_oracle"]["storage"]
    axes[1, 0].bar(
        ["paper amortized", "tail-aware formula", "actual Linear4bit"],
        [
            accounting["paper_amortized_double_bits_per_parameter"],
            accounting["tail_aware_double_bits_per_parameter"],
            8 * actual_storage["actual_tensor_bytes_including_codebooks"] / actual_storage["original_numel"],
        ],
        color=["#5B6573", "#007C91", "#D1495B"],
    )
    axes[1, 0].set_ylabel("Bits / parameter")
    axes[1, 0].set_title("Formula is not allocated tensor storage")
    axes[1, 0].tick_params(axis="x", rotation=12)

    axes[1, 1].plot(blocks, [stress["clean"][str(b)]["non_outlier_mse"] for b in blocks], "o-", label="clean")
    axes[1, 1].plot(
        blocks,
        [stress["outlier"][str(b)]["non_outlier_mse"] for b in blocks],
        "s-",
        label="one weight set to 12 (~60x data std)",
    )
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlabel("NF4 block size")
    axes[1, 1].set_ylabel("Non-outlier MSE")
    axes[1, 1].set_title("A shared absmax couples weights inside a block")
    axes[1, 1].legend(frameon=False)

    fig.suptitle("[15.3] Exact NF4 ground truth before the pretrained model", fontsize=16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_cpu_signature_result(
    implementation: LearnerFunctions,
    *,
    make_figure: bool = False,
) -> dict[str, Any]:
    """Build the CPU signature entirely from the supplied learner functions."""

    values = t.tensor([-1.2, -0.7, -0.2, 0.0, 0.08, 0.24, 0.73, 1.2], dtype=t.float32)
    quantized = implementation.quantize_nf4(values, block_size=4)
    reconstructed = implementation.dequantize_nf4(quantized)
    scales = t.tensor([0.2, 0.7, 1.2, 0.35, 0.9, 0.1, 0.55, 0.4])
    nested = implementation.double_quantize_scales(scales, block_size=4)
    nested_reconstructed = implementation.dequantize_scales(nested)

    ste_weight = t.tensor([[0.2, -0.4], [0.7, -1.0]], requires_grad=True)
    implementation.ste_nf4(ste_weight, block_size=2).sum().backward()
    train, heldout = implementation.make_codebook_examples()
    ood = implementation.make_unseen_subject_examples()
    marker_controls = implementation.make_marker_control_examples(heldout)
    result = {
        "nf4_codebook": implementation.nf4_codebook().tolist(),
        "toy_values": values.tolist(),
        "toy_indices": quantized.indices.flatten().tolist(),
        "toy_absmax": quantized.absmax.tolist(),
        "toy_reconstructed": reconstructed.tolist(),
        "toy_error": implementation.quantization_error(values, reconstructed),
        "nested_scale_max_abs_error": float((scales - nested_reconstructed).abs().max().item()),
        "storage_accounting": implementation.nf4_storage_accounting(numel=1_048_579),
        "ste_gradient": ste_weight.grad.tolist(),
        "quantization_stress": implementation.quantization_stress_test(seed=0),
        "bnb_cpu_oracle": cpu_bnb_adapter_oracle(
            seed=0,
            wrapper_cls=implementation.qlora_linear,
            storage_fn=implementation.linear4bit_storage_accounting,
            audit_fn=implementation.audit_linear4bit,
            parity_fn=implementation.dequantized_base_max_diff,
        ),
        "dataset": {
            "train_examples": len(train),
            "template_transfer_examples": len(heldout),
            "unseen_subject_ood_examples": len(ood),
            "marker_swap_examples": len(marker_controls["marker_swap"]),
            "marker_removed_examples": len(marker_controls["marker_removed"]),
            "counterbalance": {
                "train": implementation.counterbalance_audit(train),
                "template_transfer": implementation.counterbalance_audit(heldout),
                "unseen_subject_ood": implementation.counterbalance_audit(ood),
            },
            "name_only_baselines": {
                "template_transfer": implementation.name_only_baseline(train, heldout),
                "unseen_subject_ood": implementation.name_only_baseline(train, ood),
            },
        },
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "implementation_provenance": implementation.provenance(),
    }
    if make_figure:
        asset = Path(__file__).resolve().parents[2] / "instructions/assets/qlora_nf4_toy_signature.png"
        plot_toy_signature(result, asset)
        result["signature_asset"] = str(asset)
    return result


def run_cpu_reference(*, make_figure: bool = False) -> dict[str, Any]:
    return build_cpu_signature_result(reference_learner_functions(), make_figure=make_figure)


def run_cpu_release(
    *,
    implementation: LearnerFunctions,
    make_figure: bool = False,
    require_distinct_from_reference: bool = True,
) -> dict[str, Any]:
    """Run the fail-closed CPU contract through an explicit implementation bundle."""

    provenance = audit_implementation_provenance(
        implementation,
        require_distinct_from_reference=require_distinct_from_reference,
    )
    result = build_cpu_signature_result(implementation, make_figure=make_figure)
    oracle = result["bnb_cpu_oracle"]
    dataset = result["dataset"]
    checks = {
        "provenance_policy_satisfied": provenance["policy_satisfied"],
        "canonical_nf4_codebook_exact": t.equal(
            t.tensor(result["nf4_codebook"]), nf4_codebook()
        ),
        "nested_quant_state_values_exact": (
            oracle["audit"]["nf4_codebook_exact"]
            and oracle["audit"]["dynamic_codebook_exact"]
        ),
        "nested_quant_state_all_finite": (
            oracle["audit"]["all_state_tensors_finite"]
            and oracle["audit"]["dequantized_weight_finite"]
        ),
        "dequantized_base_exact": oracle["dequantized_base_max_diff"] == 0.0,
        "adapter_gradients_nonzero": (
            oracle["lora_a_gradient_norm"] > 0 and oracle["lora_b_gradient_norm"] > 0
        ),
        "counterbalanced_train": all(
            dataset["counterbalance"]["train"][key]
            for key in ("complete_subject_template_label_cross", "subjects_balanced", "templates_balanced")
        ),
        "counterbalanced_template_transfer": all(
            dataset["counterbalance"]["template_transfer"][key]
            for key in ("complete_subject_template_label_cross", "subjects_balanced", "templates_balanced")
        ),
        "counterbalanced_ood": all(
            dataset["counterbalance"]["unseen_subject_ood"][key]
            for key in ("complete_subject_template_label_cross", "subjects_balanced", "templates_balanced")
        ),
        "name_only_template_transfer_is_chance": (
            dataset["name_only_baselines"]["template_transfer"]["accuracy"] == 0.5
        ),
        "name_only_ood_is_chance": (
            dataset["name_only_baselines"]["unseen_subject_ood"]["accuracy"] == 0.5
        ),
    }
    accepted = all(checks.values())
    return {
        **result,
        "execution_device": "cpu",
        "implementation_provenance": provenance,
        "checks": checks,
        "tests_passed": accepted,
        "accepted": accepted,
    }


def run_smoke_test(cpu: bool = True) -> dict[str, Any]:
    """Compatibility hook for the generic verifier's visible-reference CPU release."""

    if not cpu:
        return run_gpu_test()
    result = run_cpu_release(
        implementation=reference_learner_functions(),
        make_figure=False,
        require_distinct_from_reference=False,
    )
    stress = result["quantization_stress"]
    oracle = result["bnb_cpu_oracle"]
    smoke = {
        "notebook_id": "15_3_qlora_from_scratch",
        "cpu": result["execution_device"] == "cpu",
        "nf4_codebook_matches_bitsandbytes": result["nf4_codebook"]
        == nf4_codebook().tolist(),
        "nested_nf4_codebook_values_exact": oracle["audit"]["nf4_codebook_exact"],
        "nested_dynamic_codebook_values_exact": oracle["audit"]["dynamic_codebook_exact"],
        "nested_quant_state_all_finite": oracle["audit"]["all_state_tensors_finite"],
        "toy_nf4_mse": result["toy_error"]["mse"],
        "nested_scale_max_abs_error": result["nested_scale_max_abs_error"],
        "double_bits_per_parameter": result["storage_accounting"]["double_bits_per_parameter"],
        "paper_amortized_double_bits_per_parameter": result["storage_accounting"][
            "paper_amortized_double_bits_per_parameter"
        ],
        "actual_linear4bit_bits_per_parameter": (
            8
            * oracle["storage"]["actual_tensor_bytes_including_codebooks"]
            / oracle["storage"]["original_numel"]
        ),
        "actual_linear4bit_includes_codebooks": (
            oracle["storage"]["actual_tensor_bytes_including_codebooks"]
            > oracle["storage"]["tail_aware_formula_bytes_excluding_codebooks"]
        ),
        "ste_gradient": result["ste_gradient"],
        "dequantized_base_max_diff": oracle["dequantized_base_max_diff"],
        "base_weight_has_no_gradient": oracle["base_weight_has_no_gradient"],
        "lora_a_gradient_norm": oracle["lora_a_gradient_norm"],
        "lora_b_gradient_norm": oracle["lora_b_gradient_norm"],
        "outlier_block_size_damage_ratio": (
            stress["outlier"]["128"]["non_outlier_mse"]
            / stress["outlier"]["8"]["non_outlier_mse"]
        ),
        "train_examples": result["dataset"]["train_examples"],
        "heldout_examples": result["dataset"]["template_transfer_examples"],
        "ood_examples": result["dataset"]["unseen_subject_ood_examples"],
        "marker_swap_examples": result["dataset"]["marker_swap_examples"],
        "marker_removed_examples": result["dataset"]["marker_removed_examples"],
        "name_only_template_transfer_accuracy": result["dataset"]["name_only_baselines"]["template_transfer"]["accuracy"],
        "name_only_ood_accuracy": result["dataset"]["name_only_baselines"]["unseen_subject_ood"]["accuracy"],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "checks": result["checks"],
        "implementation_provenance": result["implementation_provenance"],
        "nf4_codebook": result["nf4_codebook"],
        "toy_indices": result["toy_indices"],
        "toy_absmax": result["toy_absmax"],
    }
    smoke["tests_passed"] = all(smoke["checks"].values())
    smoke["accepted"] = smoke["tests_passed"]
    return smoke


def _comparison_acceptance(result: dict[str, Any]) -> dict[str, Any]:
    summary = result["summary"]
    seed_runs = result["seed_runs"]
    expected_methods = {
        "base",
        "nf4_base",
        "qlora_r2",
        "qlora_r8",
        "bf16_lora_r8",
        "full_bf16",
        "random_labels",
        "same_norm_random",
    }
    measured_seeds = [int(run["seed"]) for run in seed_runs]

    def mean(method: str, group: str, metric: str | None = None) -> float:
        stats = summary[method][group] if metric is None else summary[method][group][metric]
        return float(stats["mean"])

    qlora_accuracy = mean("qlora_r8", "template_transfer", "accuracy")
    bf16_lora_accuracy = mean("bf16_lora_r8", "template_transfer", "accuracy")
    qlora_conditional_nll = mean("qlora_r8", "template_transfer", "two_label_conditional_nll")
    bf16_lora_conditional_nll = mean(
        "bf16_lora_r8", "template_transfer", "two_label_conditional_nll"
    )
    qlora_ood_accuracy = mean("qlora_r8", "unseen_subject_ood", "accuracy")
    bf16_lora_ood_accuracy = mean("bf16_lora_r8", "unseen_subject_ood", "accuracy")
    quantized_methods = ("nf4_base", "qlora_r2", "qlora_r8", "random_labels", "same_norm_random")
    audits = [
        run["methods"][method]["quantization_audit"]
        for run in seed_runs
        for method in quantized_methods
    ]
    same_norm_errors = [run["methods"]["same_norm_random"]["same_norm"]["relative_error"] for run in seed_runs]
    dequantized_diffs = [run["methods"]["nf4_base"]["dequantized_base_max_diff"] for run in seed_runs]
    worst_seed_accuracy_gap = max(
        abs(
            run["methods"]["qlora_r8"]["behavior"]["accuracy"]
            - run["methods"]["bf16_lora_r8"]["behavior"]["accuracy"]
        )
        for run in seed_runs
    )
    full_tail_energy = min(
        summary["full_bf16"]["per_layer_spectra"][name]["energy_beyond_rank_8"]["mean"]
        for name in TARGET_MODULES
    )
    per_layer_spectra_complete = all(
        set(run["methods"][method]["per_layer_spectra"]) == set(TARGET_MODULES)
        for run in seed_runs
        for method in ("qlora_r2", "qlora_r8", "bf16_lora_r8", "full_bf16")
    )
    numeric_metrics = {
        "heldout_examples": result["dataset"]["template_transfer_examples"],
        "ood_examples": result["dataset"]["unseen_subject_ood_examples"],
        "experiment_seed_count": len(measured_seeds),
        "measured_seeds": measured_seeds,
        "qlora_r2_heldout_accuracy": mean("qlora_r2", "template_transfer", "accuracy"),
        "qlora_r8_heldout_accuracy": qlora_accuracy,
        "bf16_lora_r8_heldout_accuracy": bf16_lora_accuracy,
        "full_bf16_heldout_accuracy": mean("full_bf16", "template_transfer", "accuracy"),
        "qlora_r8_two_label_conditional_nll": qlora_conditional_nll,
        "bf16_lora_r8_two_label_conditional_nll": bf16_lora_conditional_nll,
        "qlora_vs_bf16_lora_accuracy_gap": abs(qlora_accuracy - bf16_lora_accuracy),
        "worst_seed_qlora_vs_bf16_lora_accuracy_gap": worst_seed_accuracy_gap,
        "qlora_vs_bf16_lora_two_label_conditional_nll_gap": abs(
            qlora_conditional_nll - bf16_lora_conditional_nll
        ),
        "qlora_r8_ood_accuracy": qlora_ood_accuracy,
        "qlora_vs_bf16_lora_ood_accuracy_gap": abs(qlora_ood_accuracy - bf16_lora_ood_accuracy),
        "random_label_control_accuracy": mean("random_labels", "template_transfer", "accuracy"),
        "same_norm_random_control_accuracy": mean("same_norm_random", "template_transfer", "accuracy"),
        "qlora_r8_marker_swap_accuracy": mean("qlora_r8", "marker_swap", "accuracy"),
        "qlora_r8_marker_swap_flip_rate": mean("qlora_r8", "marker_swap_flip_rate"),
        "qlora_r8_marker_swap_pair_accuracy": mean("qlora_r8", "marker_swap_pair_accuracy"),
        "qlora_r8_marker_removed_accuracy": mean("qlora_r8", "marker_removed", "accuracy"),
        "name_only_template_transfer_accuracy": result["dataset"]["name_only_baselines"]["template_transfer"]["accuracy"],
        "name_only_ood_accuracy": result["dataset"]["name_only_baselines"]["unseen_subject_ood"]["accuracy"],
        "dequantized_base_max_diff": max(dequantized_diffs),
        "same_norm_relative_error": max(same_norm_errors),
        "full_update_energy_beyond_rank8": full_tail_energy,
        "qlora_r8_protected_nll_change": mean("qlora_r8", "protected_nll_change"),
        "qlora_r8_model_memory_mib": mean("qlora_r8", "model_memory_mib"),
        "bf16_lora_r8_model_memory_mib": mean("bf16_lora_r8", "model_memory_mib"),
        "qlora_r8_optimizer_state_mib": mean("qlora_r8", "optimizer_state_mib"),
        "full_bf16_optimizer_state_mib": mean("full_bf16", "optimizer_state_mib"),
        "qlora_r8_resident_training_state_mib": mean("qlora_r8", "resident_training_state_mib"),
        "bf16_lora_r8_resident_training_state_mib": mean(
            "bf16_lora_r8", "resident_training_state_mib"
        ),
        "full_bf16_resident_training_state_mib": mean("full_bf16", "resident_training_state_mib"),
        "qlora_r8_trainable_parameter_count": summary["qlora_r8"]["trainable_parameter_count"],
        "bf16_lora_r8_trainable_parameter_count": summary["bf16_lora_r8"]["trainable_parameter_count"],
        "full_bf16_trainable_parameter_count": summary["full_bf16"]["trainable_parameter_count"],
        "qlora_learning_rate": summary["qlora_r8"]["learning_rate"],
        "bf16_lora_learning_rate": summary["bf16_lora_r8"]["learning_rate"],
        "full_bf16_learning_rate": summary["full_bf16"]["learning_rate"],
        "trainable_parameter_dtypes": {
            method: summary[method]["trainable_parameter_dtypes"]
            for method in ("qlora_r8", "bf16_lora_r8", "full_bf16")
        },
        "optimizer_state_dtypes": {
            method: summary[method]["optimizer_state_dtypes"]
            for method in ("qlora_r8", "bf16_lora_r8", "full_bf16")
        },
    }
    dataset_balance_checks = [
        audit[key]
        for run in seed_runs
        for split in ("train", "template_transfer", "unseen_subject_ood")
        for audit in (run["dataset_audit"][split],)
        for key in ("complete_subject_template_label_cross", "subjects_balanced", "templates_balanced")
    ]
    training_state_totals_exact = all(
        method_result["resident_training_state_bytes"]
        == sum(method_result["resident_training_state_components"].values())
        for run in seed_runs
        for method_result in run["methods"].values()
    )
    uncertainty_is_student_t = all(
        stats["n"] == 3
        and math.isclose(float(stats["t_critical_95"]), 4.302652729911275, rel_tol=0, abs_tol=1e-12)
        and "Student-t" in stats["uncertainty_label"]
        for method in summary.values()
        for group_name in ("template_transfer", "unseen_subject_ood", "marker_swap", "marker_removed")
        for stats in method[group_name].values()
    ) and all(
        stats["n"] == 3
        and math.isclose(float(stats["t_critical_95"]), 4.302652729911275, rel_tol=0, abs_tol=1e-12)
        and "Student-t" in stats["uncertainty_label"]
        for method in summary.values()
        for stats in (method["marker_swap_flip_rate"], method["marker_swap_pair_accuracy"])
    )
    checks = {
        "three_unique_measured_seed_runs": len(measured_seeds) == 3 and len(set(measured_seeds)) == 3,
        "requested_and_measured_seeds_match": measured_seeds == result["training"]["seeds"] == result["training"]["measured_seeds"],
        "seed_payloads_match_run": all(
            method_result["seed"] == run["seed"]
            for run in seed_runs
            for method_result in run["methods"].values()
        ),
        "implementation_provenance_verified": result["implementation_provenance"]["policy_satisfied"],
        "student_t_uncertainty_for_n3": uncertainty_is_student_t,
        "complete_method_matrix_measured": all(set(run["methods"]) == expected_methods for run in seed_runs),
        "counterbalanced_names_labels_and_templates": all(dataset_balance_checks),
        "marker_swap_accuracy_min": numeric_metrics["qlora_r8_marker_swap_accuracy"] >= 0.85,
        "marker_swap_flip_rate_min": numeric_metrics["qlora_r8_marker_swap_flip_rate"] >= 0.85,
        "marker_swap_pair_accuracy_min": numeric_metrics["qlora_r8_marker_swap_pair_accuracy"] >= 0.85,
        "marker_removed_accuracy_max": numeric_metrics["qlora_r8_marker_removed_accuracy"] <= 0.60,
        "name_only_template_transfer_is_chance": numeric_metrics["name_only_template_transfer_accuracy"] == 0.5,
        "name_only_ood_is_chance": numeric_metrics["name_only_ood_accuracy"] == 0.5,
        "all_intended_nf4_modules_materialized": all(audit["all_intended_materialized"] for audit in audits),
        "nested_quant_state_verified": all(audit["all_nested"] for audit in audits),
        "nested_quant_state_values_exact": all(
            audit["all_nf4_codebooks_exact"] and audit["all_dynamic_codebooks_exact"] for audit in audits
        ),
        "nested_quant_state_all_finite": all(
            audit["all_state_tensors_finite"] and audit["all_dequantized_weights_finite"] for audit in audits
        ),
        "nf4_uint8_storage_and_bf16_compute": all(
            audit["storage_dtype"] == "torch.uint8" and audit["compute_dtype"] == "torch.bfloat16"
            for audit in audits
        ),
        "dequantized_base_max_diff_max": numeric_metrics["dequantized_base_max_diff"] < 1e-2,
        "qlora_r2_heldout_accuracy_min": numeric_metrics["qlora_r2_heldout_accuracy"] >= 0.70,
        "qlora_r8_heldout_accuracy_min": numeric_metrics["qlora_r8_heldout_accuracy"] >= 0.85,
        "bf16_lora_r8_heldout_accuracy_min": numeric_metrics["bf16_lora_r8_heldout_accuracy"] >= 0.85,
        "full_bf16_heldout_accuracy_min": numeric_metrics["full_bf16_heldout_accuracy"] >= 0.85,
        "qlora_vs_bf16_lora_accuracy_gap_max": numeric_metrics["qlora_vs_bf16_lora_accuracy_gap"] <= 0.15,
        "worst_seed_accuracy_gap_max": numeric_metrics["worst_seed_qlora_vs_bf16_lora_accuracy_gap"] <= 0.25,
        "qlora_r8_two_label_conditional_nll_max": numeric_metrics["qlora_r8_two_label_conditional_nll"] <= 0.60,
        "qlora_vs_bf16_lora_two_label_conditional_nll_gap_max": (
            numeric_metrics["qlora_vs_bf16_lora_two_label_conditional_nll_gap"] <= 0.35
        ),
        "unseen_subject_ood_accuracy_min": numeric_metrics["qlora_r8_ood_accuracy"] >= 0.75,
        "unseen_subject_ood_parity_gap_max": numeric_metrics["qlora_vs_bf16_lora_ood_accuracy_gap"] <= 0.20,
        "random_label_control_accuracy_max": numeric_metrics["random_label_control_accuracy"] <= 0.70,
        "same_norm_random_control_accuracy_max": numeric_metrics["same_norm_random_control_accuracy"] <= 0.70,
        "same_norm_equality": numeric_metrics["same_norm_relative_error"] <= 1e-6,
        "matched_rank_exact_trainable_count": (
            numeric_metrics["qlora_r8_trainable_parameter_count"]
            == numeric_metrics["bf16_lora_r8_trainable_parameter_count"]
        ),
        "full_trainable_count_larger": (
            numeric_metrics["full_bf16_trainable_parameter_count"]
            > numeric_metrics["qlora_r8_trainable_parameter_count"]
        ),
        "optimizer_state_is_measured": (
            numeric_metrics["qlora_r8_optimizer_state_mib"] > 0
            and numeric_metrics["full_bf16_optimizer_state_mib"] > 0
        ),
        "full_optimizer_state_larger": (
            numeric_metrics["full_bf16_optimizer_state_mib"]
            > numeric_metrics["qlora_r8_optimizer_state_mib"]
        ),
        "qlora_stored_model_smaller": (
            numeric_metrics["qlora_r8_model_memory_mib"]
            < numeric_metrics["bf16_lora_r8_model_memory_mib"]
        ),
        "total_training_state_accounting_exact": training_state_totals_exact,
        "qlora_total_training_state_smaller_than_bf16_lora": (
            numeric_metrics["qlora_r8_resident_training_state_mib"]
            < numeric_metrics["bf16_lora_r8_resident_training_state_mib"]
        ),
        "qlora_total_training_state_smaller_than_full_bf16": (
            numeric_metrics["qlora_r8_resident_training_state_mib"]
            < numeric_metrics["full_bf16_resident_training_state_mib"]
        ),
        "dtype_differences_reported": (
            summary["qlora_r8"]["trainable_parameter_dtypes"] == {"torch.float32": numeric_metrics["qlora_r8_trainable_parameter_count"]}
            and set(summary["full_bf16"]["trainable_parameter_dtypes"]) == {"torch.bfloat16"}
            and summary["qlora_r8"]["optimizer_state_dtypes"] != summary["full_bf16"]["optimizer_state_dtypes"]
        ),
        "per_layer_spectra_complete": per_layer_spectra_complete,
        "full_update_has_energy_beyond_rank8": numeric_metrics["full_update_energy_beyond_rank8"] > 1e-6,
        "qlora_r8_protected_nll_change_max": numeric_metrics["qlora_r8_protected_nll_change"] <= 1.5,
        "within_vram_budget": (
            result["peak_vram_gb"] <= result["execution"]["max_vram_gb"]
            and result["within_vram_budget"]
        ),
    }
    cuda_available = (
        result["execution"]["device_type"] == "cuda"
        and bool(result.get("gpu_name"))
        and result.get("cuda_version") is not None
    )
    full_path = (
        checks["complete_method_matrix_measured"]
        and all(
            run["methods"][method]["training_steps"] == result["training"]["steps"]
            for run in seed_runs
            for method in ("qlora_r2", "qlora_r8", "bf16_lora_r8", "full_bf16", "random_labels")
        )
    )
    checks["cuda_execution_measured"] = cuda_available
    checks["full_training_path_measured"] = full_path
    return {
        **result,
        "cuda_available": cuda_available,
        "full_path": full_path,
        **numeric_metrics,
        "checks": checks,
        "accepted": all(checks.values()),
    }


def run_cuda_release(
    *,
    implementation: LearnerFunctions,
    max_vram_gb: float = 24.0,
    steps: int = 32,
    batch_size: int = 12,
) -> dict[str, Any]:
    """Run strict CUDA acceptance through an explicit, non-reference learner bundle."""

    result = run_real_comparison(
        implementation=implementation,
        max_vram_gb=max_vram_gb,
        steps=steps,
        batch_size=batch_size,
        seeds=DEFAULT_EXPERIMENT_SEEDS,
        require_distinct_from_reference=True,
    )
    return _comparison_acceptance(result)


def run_gpu_test(max_vram_gb: float = 24.0) -> dict[str, Any]:
    """Generic-verifier hook using the visible section-owned reference implementation."""

    result = run_real_comparison(
        implementation=reference_learner_functions(),
        max_vram_gb=max_vram_gb,
        steps=32,
        batch_size=12,
        seeds=DEFAULT_EXPERIMENT_SEEDS,
        require_distinct_from_reference=False,
    )
    return _comparison_acceptance(result)


def run_full_experiment(max_vram_gb: float = 24.0) -> dict[str, Any]:
    """Generic release-scale hook; CUDA-only, measured, and fail-closed."""

    result = run_real_comparison(
        implementation=reference_learner_functions(),
        max_vram_gb=max_vram_gb,
        steps=96,
        batch_size=16,
        seeds=DEFAULT_EXPERIMENT_SEEDS,
        require_distinct_from_reference=False,
    )
    return _comparison_acceptance(result)


if __name__ == "__main__":
    print(run_cpu_reference(make_figure=False))
