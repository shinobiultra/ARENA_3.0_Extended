from collections.abc import Callable
from pathlib import Path
import shutil
from typing import Any

import torch as t

from arena_ext import (
    DiskActivationStore,
    compare_logits,
    deterministic_generation_equal,
    estimate_inference_memory,
)


def _solutions():
    from chapter1_transformer_interp.exercises.part6_frontier_ml_infrastructure import (
        solutions,
    )

    return solutions


def test_memory_budget_fits_local_tier(
    estimate_gemma_1b_smoke_budget: Callable | None = None,
):
    estimate_gemma_1b_smoke_budget = (
        estimate_gemma_1b_smoke_budget or _solutions().estimate_gemma_1b_smoke_budget
    )
    budget = estimate_gemma_1b_smoke_budget(context_length=2048)
    assert 3.0 < budget.total_gb < 4.0, (
        "The 1B BF16 smoke-test budget should be a local-tier preflight estimate "
        f"near 3.66 GB, not {budget.total_gb:.3f} GB."
    )
    assert budget.parameter_gb > 1.8, (
        "The estimate should include BF16 parameter memory for 1B parameters."
    )
    assert budget.kv_cache_gb > 0.0 and budget.activation_gb > 0.0, (
        "The estimate should include both KV-cache and activation memory."
    )
    assert budget.fits(24.0), (
        "The Gemma-sized smoke-test budget should fit the 24GB local tier with reserve."
    )
    print("All tests in `test_memory_budget_fits_local_tier` passed!")


def test_compare_logits_detects_match(compare_logits_fn: Callable | None = None):
    compare_logits_fn = compare_logits_fn or compare_logits
    t.manual_seed(1)
    reference_logits = t.randn(2, 4, 30)
    custom_logits = reference_logits + 1e-6 * t.randn_like(reference_logits)
    report = compare_logits_fn(custom_logits, reference_logits, k=5)
    assert report.max_abs_diff < 1e-4, (
        "Near-identical logits should have max_abs_diff below the HF-parity tolerance."
    )
    assert report.mse < 1e-10, (
        "Near-identical logits should have tiny mean-squared error."
    )
    assert report.kl_divergence < 1e-10, (
        "Near-identical logits should have tiny KL divergence."
    )
    assert report.topk_agreement == 1.0, (
        "A tiny perturbation should preserve all top-k token sets in this fixture."
    )
    print("All tests in `test_compare_logits_detects_match` passed!")


def test_compare_logits_rejects_shape_mismatch(
    compare_logits_fn: Callable | None = None,
):
    compare_logits_fn = compare_logits_fn or compare_logits
    try:
        compare_logits_fn(t.zeros(1, 2, 3), t.zeros(1, 2, 4))
    except ValueError as exc:
        assert "same shape" in str(exc), (
            "Shape mismatch errors should explain that the two logit tensors need "
            "the same shape."
        )
    else:
        raise AssertionError("compare_logits should reject mismatched logit shapes.")
    print("All tests in `test_compare_logits_rejects_shape_mismatch` passed!")


def test_compare_logits_rejects_real_drift(compare_logits_fn: Callable | None = None):
    compare_logits_fn = compare_logits_fn or compare_logits
    reference_logits = t.zeros(1, 2, 6)
    drifted_logits = reference_logits.clone()
    drifted_logits[..., 0] = 8.0
    drifted_logits[..., 5] = -8.0
    report = compare_logits_fn(drifted_logits, reference_logits, k=2)
    assert not report.passed(
        max_abs_diff=1e-4,
        mse=1e-9,
        kl_divergence=1e-9,
        topk_agreement=1.0,
    ), "Large logit drift should fail the same tolerances used for HF parity."
    assert report.max_abs_diff >= 8.0, (
        "The drift fixture should expose a large maximum absolute logit error."
    )
    print("All tests in `test_compare_logits_rejects_real_drift` passed!")


def test_hf_parity_smoke_test_passes(hf_parity_smoke_test: Callable | None = None):
    hf_parity_smoke_test = hf_parity_smoke_test or _solutions().hf_parity_smoke_test
    assert hf_parity_smoke_test(), (
        "The synthetic HF-parity smoke test should pass all fixed logit tolerances."
    )
    print("All tests in `test_hf_parity_smoke_test_passes` passed!")


def test_deterministic_generation_equal_detects_mismatch(
    generation_equal: Callable | None = None,
):
    generation_equal = generation_equal or deterministic_generation_equal
    assert generation_equal([1, 2, 3, 4], t.tensor([1, 2, 3, 4])), (
        "Exact greedy generation equality should pass for identical token sequences."
    )
    assert not generation_equal([1, 2, 3, 4], t.tensor([1, 2, 3, 5])), (
        "Exact greedy generation equality should fail when any generated token differs."
    )
    print("All tests in `test_deterministic_generation_equal_detects_mismatch` passed!")


def test_memory_budget_rejects_oversized_model():
    budget = estimate_inference_memory(
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
    assert budget.total_gb > 24.0, (
        "A 30B BF16 local load should exceed the 24GB tier in this conservative estimate."
    )
    assert not budget.fits(24.0), (
        "Oversized model estimates must not be reported as fitting the local GPU."
    )
    print("All tests in `test_memory_budget_rejects_oversized_model` passed!")


def test_disk_activation_store_roundtrip(
    tmp_path: Path,
    store_cls: type[DiskActivationStore] | None = None,
):
    store_cls = store_cls or DiskActivationStore
    store_root = tmp_path / "roundtrip_store"
    if store_root.exists():
        shutil.rmtree(store_root)
    store = store_cls(store_root)
    activation = t.arange(6, dtype=t.float32).reshape(2, 3)
    record = store.append("hook_resid_pre", activation, metadata={"layer": 0})
    loaded = store.load(record)
    summary: dict[str, Any] = store.summary()
    assert t.equal(loaded, activation), (
        "DiskActivationStore.load should recover the exact tensor saved in append."
    )
    assert list(store.records())[0].metadata["layer"] == 0, (
        "Activation metadata should round-trip through index.jsonl."
    )
    assert summary["num_records"] == 1 and summary["names"] == ["hook_resid_pre"], (
        "Activation-store summaries should expose the record count and hook names."
    )
    print("All tests in `test_disk_activation_store_roundtrip` passed!")


def test_activation_store_smoke_test_contract(
    tmp_path: Path,
    activation_store_smoke_test: Callable | None = None,
):
    activation_store_smoke_test = (
        activation_store_smoke_test or _solutions().activation_store_smoke_test
    )
    summary = activation_store_smoke_test(tmp_path / "contract_store")
    assert summary["num_records"] == 2, (
        "The activation-store smoke test should write two independent activation shards."
    )
    assert summary["names"] == ["resid_pre", "mlp_out"], (
        "Activation-store metadata should preserve the hook names needed for audits."
    )
    assert summary["total_values"] == 28, (
        "The activation-store summary should count tensor values across all shards."
    )
    print("All tests in `test_activation_store_smoke_test_contract` passed!")


def test_notebook_contract(run_smoke_test: Callable | None = None):
    run_smoke_test = run_smoke_test or _solutions().run_smoke_test
    result = run_smoke_test(cpu=True)
    assert result["hf_parity_passed"], (
        "The smoke-test contract should include a passing logits-parity check."
    )
    assert result["generation_parity_passed"], (
        "The smoke-test contract should include a passing deterministic generation check."
    )
    assert "total_gb" in result["budget"], (
        "The smoke-test contract should report the memory-budget total."
    )
    assert "torch" in result["environment"], (
        "The smoke-test contract should include the PyTorch environment version."
    )
    assert result["activation_store"]["num_records"] == 2, (
        "The smoke-test contract should include activation-store roundtrip evidence."
    )
    print("All tests in `test_notebook_contract` passed!")


def test_committed_gpu_report_records_cuda_runtime_and_no_fallback():
    import json

    report_path = Path(__file__).with_name("verification_report.json")
    report = json.loads(report_path.read_text())
    gpu = report["metrics"]["gpu_test"]
    assert report["accepted"], "The committed infrastructure report should be accepted."
    assert gpu["cuda_available"], (
        "The accepted infrastructure report should prove CUDA was available."
    )
    assert gpu["gpu_tensor_test_passed"], (
        "The accepted infrastructure report should include a real CUDA tensor path."
    )
    assert gpu["gpu_matmul_dtype"] == "torch.bfloat16", (
        "The local runtime gate should exercise the expected BF16 CUDA matmul."
    )
    assert gpu["peak_vram_gb"] < 1.0, (
        "The infrastructure gate should stay well below the 24GB local tier."
    )
    assert gpu["uv_pip_check_passed"], (
        "The report should record that the uv-managed package set is consistent."
    )
    assert gpu["activation_store_num_records"] == 2, (
        "The CUDA report should include activation-store roundtrip evidence."
    )
    assert gpu["activation_store_names"] == ["resid_pre", "mlp_out"], (
        "The CUDA report should preserve activation hook names in metadata."
    )
    assert not report["known_failures"], (
        "Course-ready infrastructure evidence should not carry unresolved failures."
    )
    print("All tests in `test_committed_gpu_report_records_cuda_runtime_and_no_fallback` passed!")


def test_exercise_notebook_course_ready_surface():
    import json

    notebook_path = Path(__file__).with_name(
        "1.6_Local_Frontier_ML_Infrastructure_exercises.ipynb"
    )
    notebook = json.loads(notebook_path.read_text())
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
    )
    for required in [
        "Expected output",
        "Help - ",
        "Solution",
        "Signature Result",
        "Limitations",
        "Bonus - Anomaly Hunting",
    ]:
        assert required in source, (
            f"The learner notebook should include ARENA-style `{required}` content."
        )
    assert "test_committed_gpu_report_records_cuda_runtime_and_no_fallback" in source, (
        "The learner notebook should check the committed CUDA/no-fallback report contract."
    )
    print("All tests in `test_exercise_notebook_course_ready_surface` passed!")
