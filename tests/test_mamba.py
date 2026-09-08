import importlib.util

import pytest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is not installed")

if TORCH_AVAILABLE:
    import torch as t

    from arena_ext.mamba import (
        MambaConfig,
        TinyMambaBlock,
        TinyMambaForCausalLM,
        mamba_cache_parity_report,
        scan_equivalence_report,
        selective_scan_chunked,
        selective_scan_parallel,
        selective_scan_recurrent,
    )


def _scan_inputs():
    t.manual_seed(0)
    batch, seq, d_inner, d_state = 2, 7, 4, 3
    u = t.randn(batch, seq, d_inner)
    delta = t.rand(batch, seq, d_inner) + 0.1
    A_log = t.randn(d_inner, d_state) - 2.0
    B = t.randn(batch, seq, d_state)
    C = t.randn(batch, seq, d_state)
    D = t.randn(d_inner)
    z = t.randn(batch, seq, d_inner)
    return u, delta, A_log, B, C, D, z


def _tiny_config() -> "MambaConfig":
    return MambaConfig(
        vocab_size=29,
        d_model=12,
        d_inner=16,
        d_state=4,
        d_conv=3,
        dt_rank=4,
        num_layers=2,
    )


def test_parallel_scan_matches_recurrent_scan():
    u, delta, A_log, B, C, D, z = _scan_inputs()

    recurrent = selective_scan_recurrent(u, delta, A_log, B, C, D=D, z=z)
    parallel = selective_scan_parallel(u, delta, A_log, B, C, D=D, z=z)
    report = scan_equivalence_report(recurrent, parallel, atol=1e-5)

    assert report.passed, report


def test_chunked_scan_matches_full_recurrent_scan():
    u, delta, A_log, B, C, D, z = _scan_inputs()

    full = selective_scan_recurrent(u, delta, A_log, B, C, D=D, z=z)
    chunked = selective_scan_chunked(u, delta, A_log, B, C, D=D, z=z, chunk_size=3)
    report = scan_equivalence_report(full, chunked, atol=1e-6)

    assert report.passed, report


def test_mamba_block_step_matches_full_forward():
    t.manual_seed(0)
    block = TinyMambaBlock(_tiny_config())
    hidden = t.randn(2, 5, block.config.d_model)

    full, _ = block(hidden, use_cache=False)
    state = block.initial_state(batch=2, device=hidden.device, dtype=hidden.dtype)
    step_outputs = []
    for pos in range(hidden.shape[1]):
        out, state = block.step(hidden[:, pos : pos + 1], state)
        step_outputs.append(out)
    stepped = t.cat(step_outputs, dim=1)

    assert t.allclose(full, stepped, atol=1e-5)


def test_tiny_mamba_lm_cache_parity():
    t.manual_seed(1)
    model = TinyMambaForCausalLM(_tiny_config())
    input_ids = t.tensor([[1, 2, 3, 4, 5]])

    report = mamba_cache_parity_report(model, input_ids, atol=1e-5)

    assert report["passed"], report


def test_tiny_mamba_generation_shape():
    t.manual_seed(2)
    model = TinyMambaForCausalLM(_tiny_config())
    input_ids = t.tensor([[1, 2, 3]])

    generated = model.greedy_generate(input_ids, max_new_tokens=4)

    assert generated.shape == (1, 7)
