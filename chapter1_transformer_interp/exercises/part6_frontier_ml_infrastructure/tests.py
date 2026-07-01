from collections.abc import Callable
from pathlib import Path
import shutil
from typing import Any

import torch as t

from arena_ext import (
    DiskActivationStore,
    compare_logits,
    deterministic_generation_equal,
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
    print("All tests in `test_notebook_contract` passed!")
