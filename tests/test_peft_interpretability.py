import importlib.util

import pytest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is not installed")

if TORCH_AVAILABLE:
    import torch as t

    from arena_ext.peft_interpretability import (
        adapter_delta_report,
        adapter_mechanism_report,
        dora_recompose_weight,
        dora_weight_report,
        intruder_dimension_report,
        lora_delta,
        lora_merge_max_abs_diff,
    )


def test_lora_delta_matches_low_rank_update():
    lora_a = t.tensor([[1.0, 2.0]])
    lora_b = t.tensor([[3.0], [4.0]])

    delta = lora_delta(lora_a, lora_b, alpha=2.0)
    report = adapter_delta_report(lora_a, lora_b, alpha=2.0)

    t.testing.assert_close(delta, t.tensor([[6.0, 12.0], [8.0, 16.0]]))
    assert report.rank == 1
    assert report.nonzero_update


def test_dora_weight_report_preserves_learned_magnitudes():
    base_weight = t.tensor([[3.0, 4.0], [0.0, 2.0]])
    adapter_delta = t.zeros_like(base_weight)
    magnitude = t.tensor([10.0, 5.0])

    recomposed = dora_recompose_weight(base_weight, adapter_delta, magnitude)
    report = dora_weight_report(base_weight, adapter_delta, magnitude)

    assert recomposed.norm(dim=-1).tolist() == pytest.approx([10.0, 5.0])
    assert report.norm_preserved


def test_dora_weight_report_handles_nonzero_delta_direction():
    base_weight = t.tensor([[3.0, 4.0, 0.0], [0.0, 2.0, 0.0]])
    adapter_delta = t.tensor([[1.0, -2.0, 2.0], [2.0, 0.0, 1.0]])
    magnitude = t.tensor([7.0, 3.0])

    recomposed = dora_recompose_weight(base_weight, adapter_delta, magnitude)
    cosine = t.nn.functional.cosine_similarity(
        recomposed,
        base_weight + adapter_delta,
        dim=-1,
    )

    t.testing.assert_close(recomposed.norm(dim=-1), magnitude)
    t.testing.assert_close(cosine, t.ones_like(cosine))


def test_lora_merge_unmerge_parity():
    inputs = t.tensor([[1.0, -1.0, 0.5], [0.0, 2.0, -3.0]])
    base_weight = t.tensor([[0.5, -1.0, 0.0], [1.5, 0.25, -0.75]])
    lora_a = t.tensor([[1.0, 2.0, -1.0]])
    lora_b = t.tensor([[0.5], [-1.5]])

    max_diff = lora_merge_max_abs_diff(
        inputs,
        base_weight,
        lora_a,
        lora_b,
        alpha=3.0,
    )

    assert max_diff == pytest.approx(0.0, abs=1e-6)


def test_intruder_dimension_report_detects_protected_projection():
    adapter_delta = t.tensor([[1.0, 0.0], [1.0, 0.0]])
    protected_direction = t.tensor([1.0, 0.0])

    report = intruder_dimension_report(
        adapter_delta,
        protected_direction,
        max_projection_fraction=0.5,
    )

    assert report.projection_fraction == pytest.approx(1.0)
    assert report.intruder_detected


def test_adapter_mechanism_report_requires_accuracy_and_mechanism():
    report = adapter_mechanism_report(
        adapter_accuracy=0.9,
        baseline_accuracy=0.7,
        adapter_mechanism_score=0.8,
        baseline_mechanism_score=0.75,
        min_accuracy_gain=0.1,
        min_mechanism_delta=-0.02,
    )

    assert report.accuracy_delta == pytest.approx(0.2)
    assert report.mechanism_delta == pytest.approx(0.05)
    assert report.adapter_acceptable
