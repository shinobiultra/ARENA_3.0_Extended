import importlib.util
from pathlib import Path

import pytest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is not installed")

if TORCH_AVAILABLE:
    import torch as t

    from arena_ext import DiskActivationStore, compare_logits, deterministic_generation_equal
    from arena_ext.memory_budget import bytes_per_value, estimate_inference_memory


def test_memory_budget_estimates_inference_components():
    budget = estimate_inference_memory(
        num_parameters=1_000_000,
        dtype="float16",
        batch_size=2,
        context_length=16,
        hidden_size=64,
        num_layers=4,
        num_key_value_heads=2,
        head_dim=8,
        overhead_gb=0.25,
    )

    assert bytes_per_value("float16") == 2.0
    assert budget.parameter_gb > 0
    assert budget.kv_cache_gb > 0
    assert budget.activation_gb > 0
    assert budget.total_gb > budget.overhead_gb
    assert budget.fits(1.0, reserve_gb=0.0)


def test_compare_logits_reports_expected_metrics():
    reference = t.tensor([[[0.0, 1.0, 2.0], [3.0, 2.0, 1.0]]])
    custom = reference.clone()

    report = compare_logits(custom, reference, k=2)

    assert report.max_abs_diff == 0.0
    assert report.mse == 0.0
    assert report.kl_divergence == 0.0
    assert report.topk_agreement == 1.0
    assert report.passed(max_abs_diff=0.0, mse=0.0, kl_divergence=0.0, topk_agreement=1.0)


def test_compare_logits_requires_explicit_parity_tolerance():
    reference = t.tensor([[[0.0, 1.0, 2.0]]])
    report = compare_logits(reference, reference, k=2)

    with pytest.raises(ValueError, match="explicit parity tolerance"):
        report.passed()


def test_generation_parity_checks_shape_and_values():
    assert deterministic_generation_equal([1, 2, 3], t.tensor([1, 2, 3]))
    assert not deterministic_generation_equal([1, 2, 3], t.tensor([1, 2]))
    assert not deterministic_generation_equal([1, 2, 3], t.tensor([1, 2, 4]))


def test_disk_activation_store_roundtrip(tmp_path: Path):
    store = DiskActivationStore(tmp_path)
    tensor = t.arange(12, dtype=t.float32).reshape(3, 4)

    record = store.append("blocks_0_resid_pre", tensor, metadata={"layer": 0, "hook": "resid_pre"})

    assert record.name == "blocks_0_resid_pre"
    assert record.shape == (3, 4)
    assert t.equal(store.load(record), tensor)
    assert t.equal(store.load(0), tensor)
    assert store.summary()["num_records"] == 1
    assert list(store.records())[0].metadata == {"layer": 0, "hook": "resid_pre"}
