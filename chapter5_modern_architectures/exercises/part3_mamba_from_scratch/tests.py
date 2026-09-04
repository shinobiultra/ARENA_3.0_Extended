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


def test_sequential_affine_scan_manual(sequential_affine_scan: Callable | None = None):
    if sequential_affine_scan is None:
        sequential_affine_scan = _solutions().sequential_affine_scan
    a = t.tensor([[[0.5], [0.25], [1.0]]], dtype=t.float64)
    b = t.tensor([[[1.0], [-1.0], [3.0]]], dtype=t.float64)
    initial_state = t.tensor([[2.0]], dtype=t.float64)
    actual = sequential_affine_scan(a, b, initial_state)
    expected = t.tensor([[[2.0], [-0.5], [2.5]]], dtype=t.float64)
    t.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
    print("All tests in `test_sequential_affine_scan_manual` passed!")


def test_compose_affine_updates(compose_affine_updates: Callable | None = None):
    if compose_affine_updates is None:
        compose_affine_updates = _solutions().compose_affine_updates
    a_left = t.tensor([0.5, 0.2], dtype=t.float64)
    b_left = t.tensor([1.0, -2.0], dtype=t.float64)
    a_right = t.tensor([0.4, 0.3], dtype=t.float64)
    b_right = t.tensor([-1.0, 5.0], dtype=t.float64)
    composed_a, composed_b = compose_affine_updates(a_left, b_left, a_right, b_right)
    x = t.tensor([3.0, -4.0], dtype=t.float64)
    explicit = a_right * (a_left * x + b_left) + b_right
    t.testing.assert_close(composed_a * x + composed_b, explicit, atol=2e-16, rtol=0.0)
    print("All tests in `test_compose_affine_updates` passed!")


def test_parallel_affine_scan_parity(
    sequential_affine_scan: Callable | None = None,
    parallel_affine_scan: Callable | None = None,
):
    solutions = _solutions()
    sequential_affine_scan = sequential_affine_scan or solutions.sequential_affine_scan
    parallel_affine_scan = parallel_affine_scan or solutions.parallel_affine_scan
    generator = t.Generator().manual_seed(13)
    logits = t.randn((3, 37, 2, 4), generator=generator, dtype=t.float64)
    a = t.sigmoid(logits)
    b = t.randn((3, 37, 2, 4), generator=generator, dtype=t.float64)
    initial_state = t.randn((3, 2, 4), generator=generator, dtype=t.float64)
    sequential = sequential_affine_scan(a, b, initial_state)
    parallel = parallel_affine_scan(a, b, initial_state)
    t.testing.assert_close(parallel, sequential, atol=1e-12, rtol=0.0)
    print("All tests in `test_parallel_affine_scan_parity` passed!")


def test_chunked_affine_scan_state_carry(
    sequential_affine_scan: Callable | None = None,
    chunked_affine_scan: Callable | None = None,
):
    solutions = _solutions()
    sequential_affine_scan = sequential_affine_scan or solutions.sequential_affine_scan
    chunked_affine_scan = chunked_affine_scan or solutions.chunked_affine_scan
    generator = t.Generator().manual_seed(21)
    a = t.rand((2, 19, 3), generator=generator, dtype=t.float64)
    b = t.randn((2, 19, 3), generator=generator, dtype=t.float64)
    expected = sequential_affine_scan(a, b)
    for chunk_size in (1, 4, 7, 32):
        actual = chunked_affine_scan(a, b, chunk_size)
        t.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
    print("All tests in `test_chunked_affine_scan_state_carry` passed!")


def test_selective_copy_ground_truth(
    build_event_coefficients: Callable | None = None,
    sequential_affine_scan: Callable | None = None,
    selective_readout: Callable | None = None,
):
    solutions = _solutions()
    build_event_coefficients = build_event_coefficients or solutions.build_event_coefficients
    sequential_affine_scan = sequential_affine_scan or solutions.sequential_affine_scan
    selective_readout = selective_readout or solutions.selective_readout
    case = solutions.make_selective_copy_case()
    a, b, c = build_event_coefficients(case.event_ids, case.values)
    states = sequential_affine_scan(a, b)
    reads = selective_readout(states, c)[0, case.read_positions]
    expected_states = t.tensor(
        [[[1.0], [1.0], [1.0], [1.0], [1.0], [-0.7], [-0.7], [-0.7],
          [-0.7], [0.0], [0.0], [0.0]]],
        dtype=t.float64,
    )
    t.testing.assert_close(states, expected_states, atol=0.0, rtol=0.0)
    t.testing.assert_close(reads, case.read_targets, atol=0.0, rtol=0.0)
    print("All tests in `test_selective_copy_ground_truth` passed!")


def test_state_ablation_is_causal(intervene_on_state: Callable | None = None):
    solutions = _solutions()
    intervene_on_state = intervene_on_state or solutions.intervene_on_state
    case = solutions.make_selective_copy_case()
    a, b, c = solutions.build_event_coefficients(case.event_ids, case.values)
    baseline = solutions.selective_readout(solutions.sequential_affine_scan(a, b), c)
    ablated = solutions.selective_readout(intervene_on_state(a, b, 6, 0.0), c)
    read_positions = case.read_positions
    t.testing.assert_close(ablated[0, read_positions[0]], baseline[0, read_positions[0]])
    assert ablated[0, read_positions[1]].item() == 0.0, (
        "Zeroing the recurrent state after the second write should remove the later -0.7 "
        f"readout; got {ablated[0, read_positions[1]].item():.3f}."
    )
    assert baseline[0, read_positions[1]].item() == -0.7, (
        "The unedited recurrence should retain the second written value until READ; "
        f"got {baseline[0, read_positions[1]].item():.3f}."
    )
    print("All tests in `test_state_ablation_is_causal` passed!")


def test_signature_controls_are_informative(run_experiment: Callable | None = None):
    if run_experiment is None:
        run_experiment = _solutions().run_selective_copy_experiment
    result = run_experiment()
    assert result.parity_max_abs_diff <= 1e-12, (
        "Parallel affine composition should match the float64 sequential reference to "
        f"1e-12; got {result.parity_max_abs_diff:.3e}."
    )
    assert result.chunked_max_abs_diff == 0.0, (
        "Carrying the final state between chunks should exactly reproduce this scalar "
        f"reference; got max difference {result.chunked_max_abs_diff:.3e}."
    )
    assert result.selective_read_mae == 0.0, (
        "Exact WRITE/hold/READ coefficients should recover every target value; "
        f"got read MAE {result.selective_read_mae:.3f}."
    )
    assert result.fixed_decay_read_mae >= 0.25, (
        "The fixed-decay control should visibly forget delayed values; "
        f"got read MAE {result.fixed_decay_read_mae:.3f}."
    )
    assert result.reset_chunk_read_mae >= 0.20, (
        "Resetting state at chunk boundaries should fail when memory crosses a boundary; "
        f"got read MAE {result.reset_chunk_read_mae:.3f}."
    )
    assert result.ablation_effect == 0.7, (
        "Ablating after the -0.7 write should remove exactly 0.7 from the downstream read; "
        f"got effect {result.ablation_effect:.3f}."
    )
    assert result.fixed_decay_reads.std().item() > 0.1, (
        "The fixed-decay control should produce delay-dependent errors rather than a "
        f"constant placeholder; got readout std {result.fixed_decay_reads.std().item():.3f}."
    )
    print("All tests in `test_signature_controls_are_informative` passed!")
