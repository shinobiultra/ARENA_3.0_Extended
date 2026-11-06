"""15.3 QLoRA from scratch: bounded CPU oracles, a brute-force NF4 check and the learner-bundle guard."""
from __future__ import annotations

import importlib

import pytest

MODULE = "chapter15_peft_misalignment.exercises.part3_qlora_from_scratch"


@pytest.fixture(scope="module")
def methods():
    # Reproduce the historical float32 process default, then restore ambient state.
    import torch
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        return importlib.import_module(MODULE + ".solutions")
    finally:
        torch.set_default_dtype(previous)


@pytest.fixture
def cpu_state():
    import torch
    previous_dtype, previous_threads = torch.get_default_dtype(), torch.get_num_threads()
    try:
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(2)
        yield torch
    finally:
        torch.set_default_dtype(previous_dtype)
        torch.set_num_threads(previous_threads)


CPU_ORACLES = [
    ("check_nf4_codebook", ("nf4_codebook",)),
    ("check_blockwise_absmax", ("blockwise_absmax",)),
    ("check_manual_nf4", ("quantize_nf4", "dequantize_nf4")),
    ("check_double_quantization", ("double_quantize_scales", "dequantize_scales", "nf4_storage_accounting")),
    ("check_linear4bit_storage", ("linear4bit_storage_accounting",)),
    ("check_quant_state_audit", ("audit_linear4bit",)),
    ("check_ste", ("ste_nf4",)),
    ("check_toy_qlora_forward", ("toy_qlora_forward",)),
    ("check_real_qlora_wrapper", ("QLoRALinear",)),
    ("check_dequantized_base_parity", ("dequantized_base_max_diff",)),
    ("check_inject_qlora", ("inject_qlora", "QLoRALinear")),
    ("check_nf4_model_audit", ("audit_nf4_model",)),
    ("check_counterbalanced_data", ("make_codebook_examples", "make_unseen_subject_examples", "make_marker_control_examples", "counterbalance_audit", "name_only_baseline")),
    ("check_random_labels", ("make_codebook_examples", "randomize_training_labels")),
    ("check_spectrum", ("summarize_spectrum",)),
    ("check_outlier_stress", ("quantization_stress_test",)),
]


@pytest.mark.parametrize("oracle,names", CPU_ORACLES)
def test_generated_methods_against_bounded_independent_cpu_oracles(methods, cpu_state, oracle, names):
    checks = importlib.import_module(MODULE + ".tests")
    getattr(checks, oracle)(*[getattr(methods, name) for name in names])


def test_numeric_nf4_quantizer_matches_brute_force_nearest_code_and_preserves_input(methods, cpu_state):
    t = cpu_state
    values = t.tensor([-1., -0.31, 0.18, 0.74, 0., 0., 0., 0., -0.05])
    original = values.clone()
    quantized = methods.quantize_nf4(values, block_size=4)
    codebook = methods.nf4_codebook()
    expected = []
    for start in range(0, len(values), 4):
        block = values[start:start+4]
        scale = float(block.abs().max())
        for value in block:
            normalized = 0. if scale == 0 else float(value) / scale
            index = min(range(16), key=lambda i: abs(normalized - float(codebook[i])))
            expected.append(float(codebook[index]) * scale)
    t.testing.assert_close(methods.dequantize_nf4(quantized), t.tensor(expected))
    assert t.equal(values, original)


def test_imported_cuda_release_requires_an_explicit_learner_bundle_before_delegating(methods, monkeypatch):
    monkeypatch.delitem(vars(methods), "learner_functions", raising=False)
    called = []
    monkeypatch.setattr(methods.reference, "run_cuda_release", lambda **kwargs: called.append(kwargs))
    with pytest.raises(ValueError, match="implementation"):
        methods.run_cuda_release(max_vram_gb=7.0)
    assert called == []
