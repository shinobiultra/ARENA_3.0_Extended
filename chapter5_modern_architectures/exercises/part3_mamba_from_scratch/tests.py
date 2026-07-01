from collections.abc import Callable

import torch as t

from arena_ext.mamba import (
    TinyMambaBlock as ReferenceTinyMambaBlock,
    TinyMambaForCausalLM as ReferenceTinyMambaForCausalLM,
    discretize_selective_scan as reference_discretize_selective_scan,
    selective_scan_recurrent as reference_selective_scan_recurrent,
)


def _solutions():
    from chapter5_modern_architectures.exercises.part3_mamba_from_scratch import solutions

    return solutions


def test_discretize_selective_scan_shapes_and_stability(
    discretize_selective_scan: Callable | None = None,
):
    solutions = _solutions()
    discretize_selective_scan = discretize_selective_scan or solutions.discretize_selective_scan
    u, delta, A_log, B, *_ = solutions.make_scan_inputs(seed=0)
    a, b = discretize_selective_scan(u, delta, A_log, B)
    assert a.shape == (2, 7, 4, 3), (
        f"`a` should have shape (batch, seq, d_inner, d_state); got {tuple(a.shape)}."
    )
    assert b.shape == (2, 7, 4, 3), (
        f"`b` should have shape (batch, seq, d_inner, d_state); got {tuple(b.shape)}."
    )
    assert bool(((a > 0) & (a <= 1)).all()), (
        "Because A = -exp(A_log) and delta > 0, every discrete decay coefficient "
        "should be in (0, 1]."
    )
    t.testing.assert_close(
        a,
        reference_discretize_selective_scan(u, delta, A_log, B)[0],
        atol=1e-6,
        rtol=0.0,
        msg="Discretization should match the independent reference implementation.",
    )
    print("All tests in `test_discretize_selective_scan_shapes_and_stability` passed!")


def test_selective_scan_single_step_manual(selective_scan_recurrent: Callable | None = None):
    solutions = _solutions()
    selective_scan_recurrent = selective_scan_recurrent or solutions.selective_scan_recurrent
    u = t.tensor([[[2.0]]])
    delta = t.tensor([[[0.5]]])
    A_log = t.log(t.tensor([[2.0]]))
    B = t.tensor([[[3.0]]])
    C = t.tensor([[[5.0]]])
    D = t.tensor([7.0])
    z = t.tensor([[[1.0]]])
    out, last_state = selective_scan_recurrent(
        u,
        delta,
        A_log,
        B,
        C,
        D=D,
        z=z,
        return_last_state=True,
    )
    expected_state = t.tensor([[[3.0]]])
    expected = (
        (expected_state[:, None] * C[:, :, None, :]).sum(dim=-1) + u * D
    ) * t.nn.functional.silu(z)
    t.testing.assert_close(
        last_state,
        expected_state,
        msg="Single-step state should be b_t when the initial SSM state is zero.",
    )
    t.testing.assert_close(
        out,
        expected,
        msg="Single-step readout should be (C * state).sum + D*u, gated by silu(z).",
    )
    print("All tests in `test_selective_scan_single_step_manual` passed!")


def test_selective_scan_equivalence(selective_scan_equivalence_smoke_test: Callable | None = None):
    if selective_scan_equivalence_smoke_test is None:
        selective_scan_equivalence_smoke_test = _solutions().selective_scan_equivalence_smoke_test
    report = selective_scan_equivalence_smoke_test()
    assert report["passed"], (
        "Associative scan should match the recurrent reference. "
        f"Check transform composition (a2*a1, a2*b1+b2). Report: {report}"
    )
    print("All tests in `test_selective_scan_equivalence` passed!")


def test_chunked_scan_equivalence(chunked_scan_equivalence_smoke_test: Callable | None = None):
    if chunked_scan_equivalence_smoke_test is None:
        chunked_scan_equivalence_smoke_test = _solutions().chunked_scan_equivalence_smoke_test
    report = chunked_scan_equivalence_smoke_test()
    assert report["passed"], (
        "Chunked scan should match the full recurrent scan. "
        f"Check that the last SSM state is passed between chunks. Report: {report}"
    )
    print("All tests in `test_chunked_scan_equivalence` passed!")


def test_recurrent_scan_matches_reference(selective_scan_recurrent: Callable | None = None):
    solutions = _solutions()
    selective_scan_recurrent = selective_scan_recurrent or solutions.selective_scan_recurrent
    u, delta, A_log, B, C, D, z = solutions.make_scan_inputs(seed=7)
    actual = selective_scan_recurrent(u, delta, A_log, B, C, D=D, z=z)
    expected = reference_selective_scan_recurrent(u, delta, A_log, B, C, D=D, z=z)
    t.testing.assert_close(
        actual,
        expected,
        atol=1e-6,
        rtol=0.0,
        msg="Recurrent selective scan should match the independent reference path.",
    )
    print("All tests in `test_recurrent_scan_matches_reference` passed!")


def test_tiny_mamba_block_matches_reference(TinyMambaBlock: type | None = None):
    solutions = _solutions()
    TinyMambaBlock = TinyMambaBlock or solutions.TinyMambaBlock
    config = solutions.make_tiny_mamba_config()
    t.manual_seed(0)
    reference = ReferenceTinyMambaBlock(config).eval()
    local = TinyMambaBlock(config).eval()
    local.load_state_dict(reference.state_dict())
    hidden = t.randn(2, 5, config.d_model)
    with t.inference_mode():
        actual, _ = local(hidden, use_cache=False)
        expected, _ = reference(hidden, use_cache=False)
    t.testing.assert_close(
        actual,
        expected,
        atol=1e-5,
        rtol=0.0,
        msg="TinyMambaBlock should match the independent reference after loading identical weights.",
    )
    print("All tests in `test_tiny_mamba_block_matches_reference` passed!")


def test_block_step_equivalence(block_step_equivalence_smoke_test: Callable | None = None):
    if block_step_equivalence_smoke_test is None:
        block_step_equivalence_smoke_test = _solutions().block_step_equivalence_smoke_test
    report = block_step_equivalence_smoke_test()
    assert report["passed"], (
        "Full-sequence block output should match one-token stepping. "
        f"Check convolution-state and SSM-state updates. Report: {report}"
    )
    print("All tests in `test_block_step_equivalence` passed!")


def test_tiny_lm_matches_reference(TinyMambaForCausalLM: type | None = None):
    solutions = _solutions()
    TinyMambaForCausalLM = TinyMambaForCausalLM or solutions.TinyMambaForCausalLM
    config = solutions.make_tiny_mamba_config()
    t.manual_seed(1)
    reference = ReferenceTinyMambaForCausalLM(config).eval()
    local = TinyMambaForCausalLM(config).eval()
    local.load_state_dict(reference.state_dict())
    input_ids = t.tensor([[1, 2, 3, 4, 5]])
    with t.inference_mode():
        actual = local(input_ids, use_cache=False).logits
        expected = reference(input_ids, use_cache=False).logits
    t.testing.assert_close(
        actual,
        expected,
        atol=1e-5,
        rtol=0.0,
        msg="TinyMambaForCausalLM should match the independent reference with identical weights.",
    )
    assert local.lm_head.weight.data_ptr() == local.backbone.embed_tokens.weight.data_ptr(), (
        "TinyMambaForCausalLM should tie lm_head.weight to the embedding matrix when "
        "tie_word_embeddings=True."
    )
    print("All tests in `test_tiny_lm_matches_reference` passed!")


def test_tiny_lm_cache_parity(tiny_lm_cache_parity_smoke_test: Callable | None = None):
    if tiny_lm_cache_parity_smoke_test is None:
        tiny_lm_cache_parity_smoke_test = _solutions().tiny_lm_cache_parity_smoke_test
    report = tiny_lm_cache_parity_smoke_test()
    assert report["passed"], (
        "Full-sequence logits and recurrent-state logits should match. "
        f"Check state initialization and per-layer state threading. Report: {report}"
    )
    print("All tests in `test_tiny_lm_cache_parity` passed!")


def test_notebook_contract(run_smoke_test: Callable | None = None):
    if run_smoke_test is None:
        run_smoke_test = _solutions().run_smoke_test
    result = run_smoke_test(cpu=True)
    assert result["selective_scan_equivalence"]["passed"], (
        "Selective scan equivalence failed; inspect the scan tests first."
    )
    assert result["chunked_scan_equivalence"]["passed"], (
        "Chunked scan equivalence failed; check state carry across chunks."
    )
    assert result["block_step_equivalence"]["passed"], (
        "Mamba block step parity failed; check convolution and SSM state updates."
    )
    assert result["lm_cache_parity"]["passed"], (
        "Tiny Mamba LM cache parity failed; check recurrent states across layers."
    )
    assert result["generation_shape"] == (1, 7), (
        f"Greedy generation should append 4 tokens to a length-3 prompt; got {result['generation_shape']}."
    )
    print("All tests in `test_notebook_contract` passed!")
