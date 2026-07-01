"""Environment checks used by local-first ARENA extension notebooks."""

from __future__ import annotations

import importlib.util
import platform
from dataclasses import dataclass
from typing import Any

import torch as t


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


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


def print_environment_report(required_vram_gb: float | None = None) -> EnvironmentReport:
    """Print a compact report and return it for downstream metadata logging."""

    report = get_environment_report()
    print("Environment")
    for key, value in report.as_dict().items():
        print(f"  {key}: {value}")
    for warning in report.warnings(required_vram_gb):
        print(f"  WARNING: {warning}")
    return report
