# %%


MAIN = __name__ == "__main__"

# %%

import gc
import os
import hashlib
import inspect
import json
import math
import statistics
import sys
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

if MAIN:
    import bitsandbytes as bnb
import copy
import matplotlib.pyplot as plt
import numpy as np
import torch as t
import torch.nn.functional as F
from IPython.display import Image, display
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from chapter15_peft_misalignment.exercises.part3_qlora_from_scratch import legacy_reference as reference
if MAIN:
    from chapter15_peft_misalignment.exercises.part3_qlora_from_scratch import tests
from chapter15_peft_misalignment.exercises.part3_qlora_from_scratch.legacy_reference import (
    DOUBLE_QUANT_BLOCK_SIZE,
    INTENDED_QUANTIZED_SUFFIXES,
    LABEL_TEXT,
    MARKERS,
    MODEL_ID,
    MODEL_REVISION,
    NF4_BLOCK_SIZE,
    TARGET_MODULES,
    BehaviorMetrics,
    DenseLoRALinear,
    DoubleQuantizedScales,
    LearnerFunctions,
    NF4Tensor,
    PromptExample,
    SpectrumMetrics,
    TrainTrace,
)

if MAIN:
    warnings.filterwarnings("ignore", category=FutureWarning, module="bitsandbytes")
if MAIN:
    t.manual_seed(0)
if MAIN:
    print(f"torch={t.__version__}; bitsandbytes={bnb.__version__}; revision={MODEL_REVISION[:12]}")

# %%

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

# %%

if MAIN:
    tests.check_nf4_codebook(nf4_codebook)
    tests.check_blockwise_absmax(blockwise_absmax)
    tests.check_manual_nf4(quantize_nf4, dequantize_nf4)
    print("Exact NF4 lookup, padding, and zero-block checks passed.")

# %%

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

def _require_finite_tensor(value: Any, *, name: str) -> t.Tensor:
    if not isinstance(value, t.Tensor):
        raise RuntimeError(f"{name} must be a materialized tensor")
    if not bool(t.isfinite(value.detach().float()).all().item()):
        raise RuntimeError(f"{name} contains non-finite values")
    return value


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

# %%

if MAIN:
    tests.check_double_quantization(double_quantize_scales, dequantize_scales, nf4_storage_accounting)
    tests.check_linear4bit_storage(linear4bit_storage_accounting)
    tests.check_quant_state_audit(audit_linear4bit)

    corrupt = tests._actual_nf4_linear()
    corrupt.weight.quant_state.code[3] += 0.125
    try:
        audit_linear4bit(corrupt, expected_compute_dtype=t.float32)
    except RuntimeError as error:
        assert "canonical NF4" in str(error)
        print("Corrupt codebook rejected:", error)
    else:
        raise AssertionError("A changed NF4 codebook value must be rejected")

# %%

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

# %%

if MAIN:
    tests.check_ste(ste_nf4)
    tests.check_toy_qlora_forward(toy_qlora_forward)
    tests.check_real_qlora_wrapper(QLoRALinear)
    tests.check_dequantized_base_parity(dequantized_base_max_diff)
    tests.check_inject_qlora(inject_qlora, QLoRALinear)
    print("Quantized-base parity, freezing, injection, and gradient checks passed.")

# %%

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

# %%

if MAIN:
    tests.check_counterbalanced_data(
        make_codebook_examples,
        make_unseen_subject_examples,
        make_marker_control_examples,
        counterbalance_audit,
        name_only_baseline,
    )
    tests.check_random_labels(make_codebook_examples, randomize_training_labels)
    train_examples, transfer_examples = make_codebook_examples()
    ood_examples = make_unseen_subject_examples()
    controls = make_marker_control_examples(transfer_examples)
    display({
        "sizes": [len(train_examples), len(transfer_examples), len(ood_examples)],
        "name_only_transfer_accuracy": name_only_baseline(train_examples, transfer_examples)["accuracy"],
        "canonical": transfer_examples[0],
        "marker_swap": controls["marker_swap"][0],
        "marker_removed": controls["marker_removed"][0],
    })

# %%

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

# %%

if MAIN:
    tests.check_training_loop(train_on_codebook)
    print("Real TinyStories CPU adapter step passed with measured AdamW state.")

# %%

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

# %%

if MAIN:
    tests.check_spectrum(summarize_spectrum)
    tests.check_outlier_stress(quantization_stress_test)
    stress = quantization_stress_test(seed=0)
    display({size: stress["outlier"][size]["non_outlier_mse"] for size in ("8", "32", "64", "128")})

# %%

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

# %%

if MAIN:
    tests.check_nf4_model_audit(audit_nf4_model)
    try:
        load_nf4_model(device=t.device("cpu"))
    except RuntimeError as error:
        assert "no substitute path" in str(error)
        print("CPU pretrained load rejected:", error)
    else:
        raise AssertionError("The pretrained NF4 path must require CUDA")

# %%

if MAIN:
    learner_functions = LearnerFunctions(
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
    tests.check_learner_bundle(learner_functions)
    print("Your function bundle is connected. The numerical tests check its behavior; hashes only identify code.")

# %%

if MAIN:
    cpu_result = reference.run_cpu_release(
        implementation=learner_functions,
        make_figure=True,
    )
    assert cpu_result["accepted"] == all(cpu_result["checks"].values())
    display({
        "accepted": cpu_result["accepted"],
        "checks": cpu_result["checks"],
        "dataset_sizes": {
            key: cpu_result["dataset"][key]
            for key in ("train_examples", "template_transfer_examples", "unseen_subject_ood_examples")
        },
        "name_only": cpu_result["dataset"]["name_only_baselines"],
    })
    display(Image(filename=str(ROOT / "chapter15_peft_misalignment/instructions/assets/qlora_nf4_toy_signature.png")))

# %%

def run_cuda_release(max_vram_gb: float = 24.0) -> dict[str, Any]:
    return reference.run_cuda_release(
        implementation=learner_functions,
        max_vram_gb=max_vram_gb,
        steps=32,
        batch_size=12,
    )
if MAIN:

    RUN_CUDA = t.cuda.is_available() and os.environ.get("ARENA_QLORA_CPU_ONLY") != "1"
    if RUN_CUDA:
        cuda_result = run_cuda_release(max_vram_gb=24.0)
        assert cuda_result["accepted"] == all(cuda_result["checks"].values())
        display(Image(filename=str(ROOT / "chapter15_peft_misalignment/instructions/assets/qlora_real_signature_panel.png")))
        for method in ("qlora_r8", "bf16_lora_r8", "full_bf16", "random_labels", "same_norm_random"):
            result = cuda_result["summary"][method]
            print(f"{method:18s} accuracy={result['template_transfer']['accuracy']['mean']:.3f}; "
                  f"resident state={result['resident_training_state_mib']['mean']:.2f} MiB")
        print("Unmet comparison checks:", [name for name, passed in cuda_result["checks"].items() if not passed])
    else:
        print("CPU mechanics completed. The pretrained comparison requires CUDA and was not run.")

# %%

if MAIN:
    PLAY_BLOCK_SIZE = 32
    PLAY_OUTLIER_VALUE = 6.0
    play_generator = t.Generator().manual_seed(7)
    play_weights = 0.2 * t.randn(256, generator=play_generator)
    play_weights[0] = PLAY_OUTLIER_VALUE
    play_reconstructed = dequantize_nf4(quantize_nf4(play_weights, block_size=PLAY_BLOCK_SIZE))
    play_clean = play_weights.clone()
    play_clean[0] = 0.0
    play_clean_rec = dequantize_nf4(quantize_nf4(play_clean, block_size=PLAY_BLOCK_SIZE))
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5), layout="constrained")
    for label, original, reconstructed in (("with outlier", play_weights, play_reconstructed),
                                            ("outlier removed", play_clean, play_clean_rec)):
        axes[0].plot((original[1:] - reconstructed[1:]).abs(), label=label, alpha=0.7)
        axes[1].plot(original[1:33], reconstructed[1:33], ".", label=label)
    axes[0].set(xlabel="Weight index (excluding outlier)", ylabel="Absolute reconstruction error")
    axes[1].plot([-0.5, 0.5], [-0.5, 0.5], "k--", label="exact reconstruction")
    axes[1].set(xlabel="Original weight", ylabel="Reconstructed weight")
    for ax in axes: ax.legend()
    plt.show()
    display({
        "block_size": PLAY_BLOCK_SIZE,
        "raw_outlier_value": PLAY_OUTLIER_VALUE,
        "non_outlier_mse": float((play_weights[1:] - play_reconstructed[1:]).square().mean().item()),
    })

# %%

if MAIN:
    _, play_transfer = make_codebook_examples()
    play_controls = make_marker_control_examples(play_transfer)
    PLAY_INDEX = 0
    for condition, example in (("canonical", play_transfer[PLAY_INDEX]),
                                ("marker swap", play_controls["marker_swap"][PLAY_INDEX]),
                                ("marker removed", play_controls["marker_removed"][PLAY_INDEX])):
        print(f"{condition}: {example.prompt}")
        print(f"  paired target: {LABEL_TEXT[example.label]!r}")
    print("After marker removal the same prompt occurs with both targets; name alone cannot decide.")

# %%

if MAIN:
    PLAY_PARITY_LAYER = TARGET_MODULES[0]
    PLAY_INPUT_SCALE = 1.0
    precision_diagnostic = None
    if RUN_CUDA:
        diagnostic_model = load_nf4_model(device=t.device("cuda:0"))
        parent, leaf = _get_parent_and_leaf(diagnostic_model, PLAY_PARITY_LAYER)
        diagnostic_layer = getattr(parent, leaf)
        generator = t.Generator(device="cuda").manual_seed(1)
        inputs = PLAY_INPUT_SCALE * t.randn(3, diagnostic_layer.in_features,
                                             generator=generator, device="cuda", dtype=t.bfloat16)
        bf16_error = dequantized_base_max_diff(diagnostic_layer, inputs)
        state32 = copy.copy(diagnostic_layer.weight.quant_state)
        state32.dtype = t.float32
        weight32 = bnb.functional.dequantize_4bit(diagnostic_layer.weight.data, state32)
        diagnostic_layer.compute_dtype = t.float32
        bias32 = None if diagnostic_layer.bias is None else diagnostic_layer.bias.float()
        with t.no_grad():
            dense32 = F.linear(inputs.float(), weight32, bias32)
            fp32_error = float((diagnostic_layer(inputs.float()) - dense32).abs().max())
        precision_diagnostic = {
            "bf16_max_abs_error": bf16_error, "fp32_max_abs_error": fp32_error,
            "parameter_bytes": sum(p.numel() * p.element_size() for p in diagnostic_model.parameters()),
            "buffer_bytes": sum(b.numel() * b.element_size() for b in diagnostic_model.buffers()),
            "layer": PLAY_PARITY_LAYER, "input_scale": PLAY_INPUT_SCALE,
        }
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
        axes[0].bar(["BF16", "FP32 control"], [bf16_error, fp32_error])
        axes[0].axhline(0.01, linestyle="--", color="black", label="original BF16 gate")
        axes[0].set(yscale="log", ylabel="Maximum absolute output difference")
        axes[0].legend()
        axes[1].bar(["parameters", "buffers"],
                    [precision_diagnostic[k] / 2**20 for k in ("parameter_bytes", "buffer_bytes")])
        axes[1].set(ylabel="MiB (nested quantization metadata separate)")
        plt.show()
        print(f"BF16 error {bf16_error:.8f}; FP32 error {fp32_error:.8f}")
        del diagnostic_layer, parent, diagnostic_model, inputs, weight32, state32, dense32
        gc.collect()
        t.cuda.empty_cache()
    else:
        print("This precision and model-buffer diagnostic requires CUDA.")

# %%

def __getattr__(name):
    """Retain legacy plumbing/report APIs while visible taught methods take precedence."""
    return getattr(reference, name)


def _checked_module_implementation(implementation: LearnerFunctions) -> LearnerFunctions:
    """Keep the strict identity/delegation check before the imported API delegates."""
    provenance = reference.audit_implementation_provenance(
        implementation, require_distinct_from_reference=True
    )
    module_owned = [
        field for field, callback in vars(implementation).items()
        if callback is globals().get("QLoRALinear" if field == "qlora_linear" else field)
    ]
    module_ids = {
        id(globals()["QLoRALinear" if field == "qlora_linear" else field]): field
        for field in LearnerFunctions.__dataclass_fields__
    }
    module_names = {
        getattr(globals()["QLoRALinear" if field == "qlora_linear" else field], "__name__", "")
        for field in LearnerFunctions.__dataclass_fields__
    }
    direct_module_delegation = False
    for callback in vars(implementation).values():
        methods = ([value for value in vars(callback).values() if inspect.isfunction(value)]
                   if inspect.isclass(callback) else [callback])
        for method in methods:
            method = inspect.unwrap(method)
            if not inspect.isfunction(method):
                continue
            closure = inspect.getclosurevars(method)
            used_values = list(closure.globals.values()) + list(closure.nonlocals.values())
            used_values += list(method.__defaults__ or ()) + list((method.__kwdefaults__ or {}).values())
            if any(id(value) in module_ids or (
                inspect.ismodule(value) and value.__name__ == __name__
                and module_names.intersection(method.__code__.co_names)
            ) for value in used_values):
                direct_module_delegation = True
    if not provenance["policy_satisfied"] or module_owned or direct_module_delegation:
        raise ValueError(
            "implementation must contain explicit non-reference learner callbacks; "
            "legacy and module-owned solutions callbacks or direct delegation are not learner evidence"
        )
    return implementation


def make_learner_functions(**callbacks: Any) -> LearnerFunctions:
    """Construct an explicit bundle lazily; identity checks are not learner acceptance.

    No callbacks default to generated solutions or legacy reference functions.
    The caller supplies every field from its own implementation namespace.
    """
    return _checked_module_implementation(LearnerFunctions(**callbacks))


def run_cuda_release(
    max_vram_gb: float = 24.0, *, implementation: LearnerFunctions | None = None
) -> dict[str, Any]:
    """Imported-module API: require an explicit bundle before any CUDA delegation.

    The unchanged notebook wrapper uses its own assembled learner_functions.
    Importing this module deliberately does not assemble or validate that bundle.
    """
    if implementation is None:
        implementation = globals().get("learner_functions")
    if implementation is None:
        raise ValueError(
            "run_cuda_release requires implementation=LearnerFunctions(...); "
            "assemble explicit callbacks with make_learner_functions(**callbacks)"
        )
    implementation = _checked_module_implementation(implementation)
    return reference.run_cuda_release(
        implementation=implementation,
        max_vram_gb=max_vram_gb,
        steps=32,
        batch_size=12,
    )

# %%
