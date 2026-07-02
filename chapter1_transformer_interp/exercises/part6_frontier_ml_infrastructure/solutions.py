# %%
"""Reference solutions for [1.6] Local Frontier ML Infrastructure."""

import subprocess
import sys
import tempfile
from pathlib import Path

import torch as t
import torchvision

chapter = "chapter1_transformer_interp"
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from arena_ext import (
    DiskActivationStore,
    compare_logits,
    deterministic_generation_equal,
    estimate_inference_memory,
    get_environment_report,
)

MAIN = __name__ == "__main__"


# %%
def run_environment_check(required_vram_gb: float | None = 24.0):
    """Return a local environment report and print any relevant warnings."""

    report = get_environment_report()
    print("Environment report")
    for key, value in report.as_dict().items():
        print(f"  {key}: {value}")
    for warning in report.warnings(required_vram_gb):
        print(f"  WARNING: {warning}")
    return report


def estimate_gemma_1b_smoke_budget(context_length: int = 2048):
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


def hf_parity_smoke_test() -> bool:
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


def generation_parity_smoke_test() -> bool:
    """Synthetic greedy-generation parity check."""

    return deterministic_generation_equal([1, 2, 3, 4], t.tensor([1, 2, 3, 4]))


def activation_store_smoke_test(output_dir: str | Path = "activation_store_smoke") -> dict:
    """Write and reload two tiny activation tensors."""

    store = DiskActivationStore(output_dir)
    first = t.arange(12, dtype=t.float32).reshape(3, 4)
    second = t.eye(4)
    rec1 = store.append("resid_pre", first, metadata={"layer": 0, "hook": "resid_pre"})
    rec2 = store.append("mlp_out", second, metadata={"layer": 0, "hook": "mlp_out"})

    assert t.equal(store.load(rec1), first)
    assert t.equal(store.load(rec2), second)
    return store.summary()


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


def run_smoke_test(cpu: bool = True) -> dict:
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


def run_full_experiment(max_vram_gb: float = 24.0) -> dict:
    """Run the validated infrastructure readiness check."""

    return run_gpu_test(max_vram_gb=max_vram_gb)


if MAIN:
    print(run_smoke_test())
