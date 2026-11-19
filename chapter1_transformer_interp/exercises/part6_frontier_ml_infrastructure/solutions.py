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


if MAIN:
    print(run_smoke_test())
