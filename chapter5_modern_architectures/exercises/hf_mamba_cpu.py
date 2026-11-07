"""Let Transformers' Mamba and Mamba2 reference models run on CPU next to the CUDA kernels.

Transformers 5 binds the Mamba convolution and scan ops to the ``causal-conv1d`` and
``mamba-ssm`` packages whenever they are importable, whatever device the tensors are on.
Both packages are CUDA-only, so a CPU forward pass fails with
``Expected x.is_cuda() to be true``. Each bound op keeps Transformers' own PyTorch
implementation as ``__wrapped__``. ``route_cpu_mamba_ops_to_torch`` rebinds the ops to a
dispatcher that sends CPU tensors to that PyTorch implementation and every other device
to the installed kernel, so CUDA runs keep the fast path.
"""

from __future__ import annotations

import functools
import importlib
import inspect
import weakref
from collections.abc import Callable
from typing import Any

import torch as t

# Module -> ops that Transformers binds to causal-conv1d / mamba-ssm when installed.
_KERNEL_OPS = {
    "transformers.models.mamba.modeling_mamba": (
        "causal_conv1d_fn",
        "causal_conv1d_update",
        "mamba_inner_fn",
        "mamba_selective_scan",
        "mamba_selective_state_update",
    ),
    "transformers.models.mamba2.modeling_mamba2": (
        "causal_conv1d_fn",
        "causal_conv1d_update",
        "mamba2_chunk_scan",
        "mamba2_selective_state_update",
        "mamba2_split_conv1d_scan_combined",
    ),
}

# Dispatchers this module installed, so routing twice does not wrap a dispatcher again.
_DISPATCHERS: weakref.WeakSet[Callable[..., Any]] = weakref.WeakSet()


def _first_tensor(args: tuple[Any, ...], kwargs: dict[str, Any]) -> t.Tensor | None:
    for value in (*args, *kwargs.values()):
        if isinstance(value, t.Tensor):
            return value
    return None


def device_dispatch(fast: Callable[..., Any], reference: Callable[..., Any]) -> Callable[..., Any]:
    """Call ``reference`` when the first tensor argument is on CPU, otherwise ``fast``."""

    parameters = inspect.signature(reference).parameters.values()
    takes_any_kwarg = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters)
    reference_kwargs = {p.name for p in parameters}

    @functools.wraps(reference)
    def op(*args: Any, **kwargs: Any) -> Any:
        tensor = _first_tensor(args, kwargs)
        if tensor is not None and tensor.device.type == "cpu":
            if not takes_any_kwarg:
                kwargs = {k: v for k, v in kwargs.items() if k in reference_kwargs}
            return reference(*args, **kwargs)
        return fast(*args, **kwargs)

    _DISPATCHERS.add(op)
    return op


def route_cpu_mamba_ops_to_torch() -> list[str]:
    """Rebind Transformers' Mamba/Mamba2 kernel ops to a CPU-aware dispatcher.

    Safe to call more than once. Returns the ``module.op`` names that dispatch by device.
    """

    routed = []
    for module_name, op_names in _KERNEL_OPS.items():
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        for name in op_names:
            fast = getattr(module, name, None)
            if fast is None:
                continue
            if fast not in _DISPATCHERS:
                reference = getattr(fast, "__wrapped__", None)
                if reference is None:
                    continue
                setattr(module, name, device_dispatch(fast, reference))
            routed.append(f"{module_name.rsplit('.', 1)[-1]}.{name}")
    return routed
