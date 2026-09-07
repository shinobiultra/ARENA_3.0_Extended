from __future__ import annotations

import copy
import json
import math
from collections.abc import Callable
from pathlib import Path

import pytest
import torch as t
import torch.nn.functional as F


def _solutions():
    from . import solutions

    return solutions


def test_topk_router_exact(topk_router: Callable | None = None):
    solutions = _solutions()
    topk_router = topk_router or solutions.topk_router
    logits = t.log(t.tensor([[0.6, 0.3, 0.1], [0.2, 0.2, 0.6]], dtype=t.float64))
    result = topk_router(logits, 2)
    t.testing.assert_close(result.probabilities, logits.exp(), rtol=1e-12, atol=1e-12)
    assert result.topk_indices.tolist() == [[0, 1], [2, 0]], (
        "Top-k must rank experts independently for each token. A common bug is sorting the token axis."
    )
    t.testing.assert_close(result.topk_gates.sum(dim=-1), t.ones(2, dtype=t.float64))
    t.testing.assert_close(result.topk_gates[0], t.tensor([2 / 3, 1 / 3], dtype=t.float64))
    t.testing.assert_close(result.topk_affinities[0], t.tensor([0.6, 0.3], dtype=t.float64))
    print("All tests in `test_topk_router_exact` passed!")


def test_topk_router_ties_are_portable_and_switch_top1_has_task_gradient(
    topk_router: Callable | None = None,
):
    solutions = _solutions()
    topk_router = topk_router or solutions.topk_router
    tied_logits = t.tensor([[2.0, 2.0, 2.0], [1.0, 3.0, 3.0]])
    route = topk_router(tied_logits, 2)
    assert route.topk_indices.tolist() == [[0, 1], [1, 2]], (
        "Equal affinities must use lower expert index as the explicit portable tie-break."
    )

    logits = t.tensor([[0.3, -0.2, 0.1]], requires_grad=True)
    top1 = topk_router(logits, 1)
    expected_gate = logits.softmax(dim=-1).max(dim=-1, keepdim=True).values
    t.testing.assert_close(top1.topk_gates, expected_gate)
    assert not bool(top1.topk_gates.eq(1).all()), (
        "Switch top-1 scales the selected expert by its full-softmax probability; "
        "it is not a normalized-to-one straight-through gate."
    )
    expert_values = t.tensor([[2.0, -1.0, 0.5]])
    selected_values = expert_values.gather(1, top1.topk_indices)
    loss = (top1.topk_gates * selected_values).sum()
    loss.backward()
    assert logits.grad is not None and float(logits.grad.abs().sum()) > 0, (
        "The selected Switch probability must preserve an ordinary task-loss router gradient."
    )
    print("All tests in `test_topk_router_ties_are_portable_and_switch_top1_has_task_gradient` passed!")


def test_capacity_ledger_exact(compute_capacity_state: Callable | None = None):
    solutions = _solutions()
    compute_capacity_state = compute_capacity_state or solutions.compute_capacity_state
    indices = t.tensor([[0, 1], [0, 2], [0, 1], [2, 1]])
    gates = t.tensor([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4]])
    ledger = compute_capacity_state(indices, gates, n_experts=3, capacity_factor=0.75)
    assert ledger.capacity == 2, "capacity = ceil(factor * tokens * top_k / experts) = ceil(2)."
    assert ledger.requested_counts.tolist() == [3, 3, 2]
    assert ledger.kept_counts.tolist() == [2, 2, 2]
    assert ledger.dropped_count == 2
    assert ledger.slot_indices.tolist() == [[0, 0], [1, 0], [-1, 1], [1, -1]], (
        "Causal overflow must retain eligible assignments left-to-right and mark drops with slot -1."
    )
    print("All tests in `test_capacity_ledger_exact` passed!")


def test_capacity_is_left_to_right_and_handles_partial_masked_and_empty_batches(
    compute_capacity_state: Callable | None = None,
):
    solutions = _solutions()
    compute_capacity_state = compute_capacity_state or solutions.compute_capacity_state
    indices = t.tensor([[0], [0], [0], [1], [1], [2]])
    affinities = t.tensor([[0.10], [0.95], [0.60], [0.40], [0.30], [0.20]])
    ledger = compute_capacity_state(indices, affinities, 3, capacity_factor=1.0)
    assert ledger.capacity == 2
    assert ledger.requested_counts.tolist() == [3, 2, 1]
    assert ledger.kept_counts.tolist() == [2, 2, 1]
    assert ledger.keep_mask[:, 0].tolist() == [True, True, False, True, True, True], (
        "Future high-affinity tokens must not displace earlier decoder assignments."
    )
    reversed_affinity = compute_capacity_state(indices, affinities.flip(0), 3, capacity_factor=1.0)
    assert t.equal(reversed_affinity.keep_mask, ledger.keep_mask)

    mask = t.tensor([True, False, True, True, False, True])
    masked = compute_capacity_state(indices, affinities, 3, capacity_factor=1.0, token_mask=mask)
    assert masked.capacity == 2
    assert masked.requested_counts.tolist() == [2, 1, 1]
    assert masked.kept_counts.tolist() == [2, 1, 1]
    assert not bool(masked.keep_mask[~mask].any())

    empty = compute_capacity_state(
        indices,
        affinities,
        3,
        capacity_factor=1.0,
        token_mask=t.zeros(6, dtype=t.bool),
    )
    assert empty.capacity == 0
    assert empty.requested_counts.tolist() == [0, 0, 0]
    assert empty.dropped_count == 0 and empty.dropped_fraction == 0.0
    assert empty.slot_indices.eq(-1).all()
    print("All tests in `test_capacity_is_left_to_right_and_handles_partial_masked_and_empty_batches` passed!")


def test_capacity_is_prefix_and_batch_invariant(
    compute_capacity_state: Callable | None = None,
):
    solutions = _solutions()
    compute_capacity_state = compute_capacity_state or solutions.compute_capacity_state
    prefix = t.tensor([[0], [0], [1]])
    first = t.cat([prefix, t.tensor([[2], [2], [2]])])
    second = t.cat([prefix, t.tensor([[0], [1], [0]])])
    affinities = t.linspace(0.1, 0.9, 6)[:, None]
    first_ledger = compute_capacity_state(first, affinities, 3, 1.0)
    second_ledger = compute_capacity_state(second, affinities.flip(0), 3, 1.0)
    assert t.equal(first_ledger.keep_mask[:3], second_ledger.keep_mask[:3])
    assert t.equal(first_ledger.slot_indices[:3], second_ledger.slot_indices[:3])

    batched_indices = t.stack([first, second])
    batched_affinities = t.stack([affinities, affinities.flip(0)])
    batched = compute_capacity_state(batched_indices, batched_affinities, 3, 1.0)
    assert t.equal(batched.keep_mask[0], first_ledger.keep_mask)
    assert t.equal(batched.slot_indices[0], first_ledger.slot_indices)
    assert t.equal(batched.keep_mask[1], second_ledger.keep_mask)
    assert t.equal(batched.slot_indices[1], second_ledger.slot_indices)
    print("All tests in `test_capacity_is_prefix_and_batch_invariant` passed!")


def test_dispatch_and_combine_manual(
    dispatch_tokens: Callable | None = None,
    combine_expert_outputs: Callable | None = None,
):
    solutions = _solutions()
    dispatch_tokens = dispatch_tokens or solutions.dispatch_tokens
    combine_expert_outputs = combine_expert_outputs or solutions.combine_expert_outputs
    tokens = t.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    indices = t.tensor([[0, 1], [1, 0], [0, 1]])
    gates = t.tensor([[0.75, 0.25], [0.6, 0.4], [0.55, 0.45]])
    ledger = solutions.compute_capacity_state(indices, gates, 2, capacity_factor=1.0)
    dispatched = dispatch_tokens(tokens, indices, ledger, 2)
    assert dispatched.shape == (2, 3, 2)
    expert_outputs = t.stack([dispatched[0] * 2, dispatched[1] * -1])
    combined, weight = combine_expert_outputs(expert_outputs, indices, gates, ledger)
    expected = t.tensor([[1.25, 12.5], [0.4, 4.0], [1.95, 19.5]])
    t.testing.assert_close(combined, expected)
    t.testing.assert_close(weight, t.ones(3))
    print("All tests in `test_dispatch_and_combine_manual` passed!")


def test_dispatch_and_combine_preserve_activation_and_gate_gradients(
    dispatch_tokens: Callable | None = None,
    combine_expert_outputs: Callable | None = None,
):
    solutions = _solutions()
    dispatch_tokens = dispatch_tokens or solutions.dispatch_tokens
    combine_expert_outputs = combine_expert_outputs or solutions.combine_expert_outputs
    tokens = t.randn(4, 3, requires_grad=True)
    indices = t.tensor([[0, 1], [1, 0], [0, 1], [1, 0]])
    affinities = t.tensor([[0.9, 0.2], [0.8, 0.4], [0.7, 0.6], [0.5, 0.3]])
    gates = t.tensor([[0.7, 0.3], [0.6, 0.4], [0.55, 0.45], [0.8, 0.2]], requires_grad=True)
    ledger = solutions.compute_capacity_state(indices, affinities, 2, capacity_factor=1.0)
    dispatched = dispatch_tokens(tokens, indices, ledger, 2)
    expert_outputs = dispatched.square()
    combined, _ = combine_expert_outputs(expert_outputs, indices, gates, ledger)
    combined.sum().backward()
    assert tokens.grad is not None and float(tokens.grad.abs().sum()) > 0
    assert gates.grad is not None and float(gates.grad.abs().sum()) > 0
    print("All tests in `test_dispatch_and_combine_preserve_activation_and_gate_gradients` passed!")


def test_expert_mlps_are_independent_and_differentiable(ExpertMLPs: type | None = None):
    solutions = _solutions()
    ExpertMLPs = ExpertMLPs or solutions.ExpertMLPs
    t.manual_seed(0)
    experts = ExpertMLPs(3, 4, 7)
    inputs = t.randn(3, 5, 4, requires_grad=True)
    outputs = experts(inputs)
    assert outputs.shape == inputs.shape
    outputs.square().sum().backward()
    assert inputs.grad is not None and float(inputs.grad.abs().sum()) > 0
    assert experts.experts[0][0].weight.data_ptr() != experts.experts[1][0].weight.data_ptr(), (
        "Experts must own independent parameters rather than sharing one MLP instance."
    )
    print("All tests in `test_expert_mlps_are_independent_and_differentiable` passed!")


def test_router_losses_have_semantic_controls(
    load_balancing_loss: Callable | None = None,
    router_z_loss: Callable | None = None,
):
    solutions = _solutions()
    load_balancing_loss = load_balancing_loss or solutions.load_balancing_loss
    router_z_loss = router_z_loss or solutions.router_z_loss
    balanced_probabilities = t.full((6, 3), 1 / 3)
    balanced_indices = t.tensor([[0], [1], [2], [0], [1], [2]])
    collapsed_probabilities = t.tensor([[0.98, 0.01, 0.01]]).repeat(6, 1)
    collapsed_indices = t.zeros(6, 1, dtype=t.long)
    balanced = load_balancing_loss(balanced_probabilities, balanced_indices)
    collapsed = load_balancing_loss(collapsed_probabilities, collapsed_indices)
    assert float(balanced) == pytest.approx(1.0, abs=1e-6)
    assert float(collapsed) > float(balanced) + 1.8
    collapsed_logits = t.tensor([[4.0, -2.0, -2.0]]).repeat(6, 1).requires_grad_()
    collapsed_loss = load_balancing_loss(
        collapsed_logits.softmax(dim=-1),
        collapsed_indices,
    )
    collapsed_loss.backward()
    assert bool((collapsed_logits.grad[:, 0] > 0).all())
    assert bool((collapsed_logits.grad[:, 1:] < 0).all()), (
        "Gradient descent on the aux loss must lower the overloaded expert and raise unused experts."
    )

    logits = t.tensor([[1.0, -1.0, 0.0]])
    shifted = logits + 10
    assert float(router_z_loss(shifted)) > float(router_z_loss(logits)) + 50, (
        "z-loss must notice a common logit shift even though softmax routing does not."
    )
    common_logits = t.full((4, 3), 6.0, requires_grad=True)
    before_probabilities = common_logits.softmax(dim=-1).detach()
    before_scale = solutions.router_log_partition_rms(common_logits.detach())
    router_z_loss(common_logits).backward()
    updated_logits = common_logits.detach() - 0.1 * common_logits.grad
    after_scale = solutions.router_log_partition_rms(updated_logits)
    assert after_scale < before_scale
    t.testing.assert_close(updated_logits.softmax(dim=-1), before_probabilities)
    assert before_scale**2 == pytest.approx(float(router_z_loss(common_logits.detach())), rel=1e-6)
    print("All tests in `test_router_losses_have_semantic_controls` passed!")


def test_paired_factorial_effects_are_signed_with_minus_without(
    paired_factorial_effects: Callable | None = None,
):
    solutions = _solutions()
    paired_factorial_effects = paired_factorial_effects or solutions.paired_factorial_effects
    effects = paired_factorial_effects(
        {"aux+z": 13.0, "aux-only": 9.0, "z-only": 7.0, "neither": 5.0}
    )
    assert effects == {
        "aux_main_effect": 5.0,
        "z_main_effect": 3.0,
        "aux_z_interaction": 2.0,
    }
    with pytest.raises(ValueError, match="exactly"):
        paired_factorial_effects({"aux+z": 1.0})
    print("All tests in `test_paired_factorial_effects_are_signed_with_minus_without` passed!")


def test_sparse_moe_matches_unbounded_bruteforce(SparseMoE: type | None = None):
    solutions = _solutions()
    SparseMoE = SparseMoE or solutions.SparseMoE
    t.manual_seed(1)
    config = solutions.MoEConfig(d_model=8, n_heads=2, n_experts=3, d_expert=12, top_k=2)
    moe = SparseMoE(config).eval()
    hidden = t.randn(2, 4, 8)
    actual = moe(hidden, capacity_factor=float("inf"))
    flat = hidden.reshape(-1, 8)
    logits = moe.router(flat)
    route = solutions.topk_router(logits, 2)
    expected = t.zeros_like(flat)
    for token in range(flat.shape[0]):
        for rank in range(2):
            expert = int(route.topk_indices[token, rank])
            expected[token] += route.topk_gates[token, rank] * moe.expert_mlps.experts[expert](flat[token])
    t.testing.assert_close(actual.output, expected.reshape_as(hidden), rtol=1e-5, atol=1e-6)
    assert actual.capacity.dropped_count == 0
    print("All tests in `test_sparse_moe_matches_unbounded_bruteforce` passed!")


def test_top1_sparse_moe_language_path_reaches_router(SparseMoE: type | None = None):
    solutions = _solutions()
    SparseMoE = SparseMoE or solutions.SparseMoE
    t.manual_seed(13)
    config = solutions.MoEConfig(d_model=8, n_heads=2, n_experts=3, d_expert=12, top_k=1)
    moe = SparseMoE(config)
    hidden = t.randn(2, 4, 8)
    output = moe(hidden, capacity_factor=math.inf)
    output.output.square().mean().backward()
    gradient = moe.router.weight.grad
    assert gradient is not None and float(gradient.abs().sum()) > 0, (
        "With aux and z coefficients set to zero, top-1 task loss must still train the router through the selected full-softmax probability."
    )
    print("All tests in `test_top1_sparse_moe_language_path_reaches_router` passed!")


def test_capacity_drop_accounting_is_visible(SparseMoE: type | None = None):
    solutions = _solutions()
    SparseMoE = SparseMoE or solutions.SparseMoE
    config = solutions.MoEConfig(d_model=8, n_heads=2, n_experts=3, d_expert=12, router_bias_init=20)
    moe = SparseMoE(config).eval()
    hidden = t.zeros(2, 6, 8)
    output = moe(hidden, capacity_factor=1.0)
    assert output.capacity.requested_counts[0] == 12
    assert output.capacity.kept_counts[0] == 4
    assert output.capacity.dropped_count == 8
    assert int(output.combined_weight.eq(0).sum()) == 8, (
        "Dropped tokens must be observable; silently rerouting them would change Switch-style capacity semantics."
    )
    print("All tests in `test_capacity_drop_accounting_is_visible` passed!")


def test_coherent_expert_permutation_is_invariant(permute_moe_experts: Callable | None = None):
    solutions = _solutions()
    permute_moe_experts = permute_moe_experts or solutions.permute_moe_experts
    t.manual_seed(3)
    config = solutions.MoEConfig(d_model=8, n_heads=2, n_experts=3, d_expert=10, top_k=2)
    original = solutions.SparseMoE(config).eval()
    permuted = permute_moe_experts(original, (2, 0, 1)).eval()
    hidden = t.randn(2, 5, 8)
    t.testing.assert_close(
        original(hidden, capacity_factor=float("inf")).output,
        permuted(hidden, capacity_factor=float("inf")).output,
        rtol=1e-5,
        atol=1e-6,
        msg="Permuting router rows without the corresponding experts is not a valid expert-label symmetry.",
    )
    print("All tests in `test_coherent_expert_permutation_is_invariant` passed!")


def test_causal_attention_has_no_future_leakage(CausalSelfAttention: type | None = None):
    solutions = _solutions()
    CausalSelfAttention = CausalSelfAttention or solutions.CausalSelfAttention
    t.manual_seed(4)
    config = solutions.MoEConfig(d_model=8, n_heads=2)
    attention = CausalSelfAttention(config).eval()
    hidden = t.randn(2, 6, 8)
    changed = hidden.clone()
    changed[:, -1] += 10_000
    first, _ = attention(hidden)
    second, _ = attention(changed)
    t.testing.assert_close(first[:, :-1], second[:, :-1], rtol=1e-5, atol=1e-5)
    print("All tests in `test_causal_attention_has_no_future_leakage` passed!")


def test_finite_capacity_decoder_has_prefix_and_batch_invariance(
    SparseMoETransformer: type | None = None,
):
    solutions = _solutions()
    SparseMoETransformer = SparseMoETransformer or solutions.SparseMoETransformer
    t.manual_seed(404)
    config = solutions.MoEConfig(
        d_model=12,
        n_heads=3,
        d_expert=18,
        max_seq_len=20,
        capacity_factor=0.75,
        router_bias_init=8.0,
    )
    model = SparseMoETransformer(config).eval()
    first = t.tensor([[0, 4, 1, 4, 2, 5, 3, 4, 1, 4]])
    changed_suffix = first.clone()
    changed_suffix[:, 6:] = t.tensor([[1, 8, 2, 7]])
    mask = first.ge(solutions.REGIME_TOKEN_OFFSET) & first.lt(solutions.DATA_TOKEN_OFFSET)
    changed_mask = changed_suffix.ge(solutions.REGIME_TOKEN_OFFSET) & changed_suffix.lt(
        solutions.DATA_TOKEN_OFFSET
    )
    first_output = model(first, capacity_factor=0.75, moe_token_mask=mask)
    changed_output = model(
        changed_suffix,
        capacity_factor=0.75,
        moe_token_mask=changed_mask,
    )
    t.testing.assert_close(
        first_output.logits[:, :6],
        changed_output.logits[:, :6],
        rtol=0,
        atol=0,
        msg="Changing a future suffix must not alter finite-capacity decoder logits.",
    )
    assert t.equal(
        first_output.moe_outputs[0].capacity.keep_mask[:, :6],
        changed_output.moe_outputs[0].capacity.keep_mask[:, :6],
    )

    unrelated = t.tensor([[0, 8, 3, 7, 3, 6, 2, 7, 1, 7]])
    batched_tokens = t.cat([first, unrelated])
    batched_mask = batched_tokens.ge(solutions.REGIME_TOKEN_OFFSET) & batched_tokens.lt(
        solutions.DATA_TOKEN_OFFSET
    )
    batched = model(
        batched_tokens,
        capacity_factor=0.75,
        moe_token_mask=batched_mask,
    )
    t.testing.assert_close(
        first_output.logits[0],
        batched.logits[0],
        rtol=1e-6,
        atol=1e-6,
        msg="An unrelated batch row must not alter finite-capacity decoder logits.",
    )
    assert t.equal(
        first_output.moe_outputs[0].capacity.keep_mask[0],
        batched.moe_outputs[0].capacity.keep_mask[0],
    )
    print("All tests in `test_finite_capacity_decoder_has_prefix_and_batch_invariance` passed!")


def test_moe_transformer_block_shapes(MoETransformerBlock: type | None = None):
    solutions = _solutions()
    MoETransformerBlock = MoETransformerBlock or solutions.MoETransformerBlock
    t.manual_seed(41)
    config = solutions.MoEConfig(d_model=12, n_heads=3, d_expert=18)
    block = MoETransformerBlock(config).eval()
    hidden = t.randn(2, 5, config.d_model)
    mask = t.tensor([[True, False, True, False, True]]).expand(2, -1)
    output, cache, moe = block(hidden, token_mask=mask, capacity_factor=math.inf)
    assert output.shape == hidden.shape
    assert cache[0].shape == (2, config.n_heads, 5, config.d_model // config.n_heads)
    assert moe.output.shape == hidden.shape
    print("All tests in `test_moe_transformer_block_shapes` passed!")


def test_complete_model_and_cache_parity(SparseMoETransformer: type | None = None):
    solutions = _solutions()
    SparseMoETransformer = SparseMoETransformer or solutions.SparseMoETransformer
    t.manual_seed(5)
    config = solutions.MoEConfig(d_model=16, n_heads=4, d_expert=24, max_seq_len=20)
    model = SparseMoETransformer(config).eval()
    tokens = t.randint(0, config.vocab_size, (2, 7))
    full = model(tokens, capacity_factor=float("inf")).logits
    caches = None
    pieces = []
    for position in range(tokens.shape[1]):
        output = model(
            tokens[:, position : position + 1],
            caches=caches,
            use_cache=True,
            capacity_factor=float("inf"),
        )
        pieces.append(output.logits)
        caches = output.caches
    cached = t.cat(pieces, dim=1)
    t.testing.assert_close(cached, full, rtol=2e-5, atol=2e-5)
    print("All tests in `test_complete_model_and_cache_parity` passed!")


def test_finite_capacity_resets_per_chunk_and_unbounded_greedy_is_exact(
    SparseMoETransformer: type | None = None,
):
    solutions = _solutions()
    SparseMoETransformer = SparseMoETransformer or solutions.SparseMoETransformer
    t.manual_seed(6)
    config = solutions.MoEConfig(
        d_model=12,
        n_heads=3,
        d_expert=18,
        max_seq_len=20,
        router_bias_init=20.0,
    )
    model = SparseMoETransformer(config).eval()
    tokens = t.randint(0, config.vocab_size, (2, 6))
    full = model(tokens, capacity_factor=1.0)
    caches = None
    chunked_weights = []
    for position in range(tokens.shape[1]):
        output = model(
            tokens[:, position : position + 1],
            caches=caches,
            use_cache=True,
            capacity_factor=1.0,
        )
        chunked_weights.append(output.moe_outputs[0].combined_weight)
        caches = output.caches
    assert int(full.moe_outputs[0].combined_weight.ne(0).sum()) == 4
    assert int(t.cat(chunked_weights, dim=1).ne(0).sum()) == 12, (
        "Without an explicit routing-state cache, each sequence starts a fresh causal quota in every forward chunk."
    )

    prompt = tokens[:1, :3]
    generated = model.greedy_generate(prompt, max_new_tokens=3)
    manual = prompt
    for _ in range(3):
        next_token = model(manual, capacity_factor=math.inf).logits[:, -1:].argmax(dim=-1)
        manual = t.cat([manual, next_token], dim=1)
    assert t.equal(generated, manual)
    print("All tests in `test_finite_capacity_resets_per_chunk_and_unbounded_greedy_is_exact` passed!")


def test_malformed_caches_are_rejected(SparseMoETransformer: type | None = None):
    solutions = _solutions()
    SparseMoETransformer = SparseMoETransformer or solutions.SparseMoETransformer
    config = solutions.MoEConfig(d_model=12, n_heads=3, n_layers=2, d_expert=18, max_seq_len=20)
    model = SparseMoETransformer(config).eval()
    tokens = t.randint(0, config.vocab_size, (2, 2))
    valid = model(tokens[:, :1], use_cache=True, capacity_factor=math.inf).caches
    assert valid is not None
    with pytest.raises(ValueError, match="every layer"):
        model(tokens[:, 1:], caches=(valid[0], None), use_cache=True)
    with pytest.raises(ValueError, match="shape"):
        bad_value = valid[0][1][..., :-1]
        model(tokens[:, 1:], caches=((valid[0][0], bad_value), valid[1]), use_cache=True)
    with pytest.raises(ValueError, match="batch, head"):
        bad_key = valid[0][0][:1]
        model(tokens[:, 1:], caches=((bad_key, bad_key.clone()), valid[1]), use_cache=True)
    with pytest.raises(ValueError, match="same number"):
        shorter = tuple(tensor[..., :0, :] for tensor in valid[1])
        model(tokens[:, 1:], caches=(valid[0], shorter), use_cache=True)
    with pytest.raises(ValueError, match="max_seq_len"):
        model.greedy_generate(tokens, max_new_tokens=config.max_seq_len)
    print("All tests in `test_malformed_caches_are_rejected` passed!")


def test_regime_dataset_is_exact_and_causal(make_regime_sequences: Callable | None = None):
    solutions = _solutions()
    make_regime_sequences = make_regime_sequences or solutions.make_regime_sequences
    regimes = t.tensor([0, 1, 2])
    starts = t.tensor([2, 2, 2])
    sequences = make_regime_sequences(regimes, starts, 5)
    assert sequences.tolist() == [
        [0, 6, 1, 6, 1, 6, 1, 6, 1, 6],
        [0, 6, 2, 7, 2, 8, 2, 4, 2, 5],
        [0, 6, 3, 5, 3, 4, 3, 8, 3, 7],
    ]
    assert sequences[:, 0].eq(solutions.BOS_TOKEN).all()
    print("All tests in `test_regime_dataset_is_exact_and_causal` passed!")


def test_next_token_objective_grades_only_operations(
    next_token_loss: Callable | None = None,
):
    solutions = _solutions()
    next_token_loss = next_token_loss or solutions.next_token_loss
    sequences = solutions.make_regime_sequences(t.tensor([0, 1, 2]), t.tensor([1, 2, 3]), 5)
    logits = t.full((3, sequences.shape[1] - 1, 15), -20.0)
    targets = sequences[:, 1:]
    inputs = sequences[:, :-1]
    operation_mask = inputs.ge(solutions.REGIME_TOKEN_OFFSET) & inputs.lt(solutions.DATA_TOKEN_OFFSET)
    rows, columns = t.nonzero(operation_mask, as_tuple=True)
    logits[rows, columns, targets[rows, columns]] = 20.0
    logits[~operation_mask, 14] = 20.0
    assert float(next_token_loss(logits, sequences)) < 1e-6, (
        "Only predictions immediately after operation tokens should enter the controlled task loss."
    )
    print("All tests in `test_next_token_objective_grades_only_operations` passed!")


def test_chance_baselines_match_evaluator_support():
    solutions = _solutions()
    assert solutions.TASK_EVALUATOR_SUPPORT == tuple(range(solutions.MoEConfig().vocab_size))
    assert solutions.TASK_CHANCE_ACCURACY == pytest.approx(1 / 15)
    assert solutions.ROUTER_PROBE_CHANCE_ACCURACY == pytest.approx(1 / 3)
    print("All tests in `test_chance_baselines_match_evaluator_support` passed!")


def test_disjoint_streams_are_unique_and_genuinely_mixed():
    solutions = _solutions()
    config = solutions.TrainingConfig(steps=4, batch_size=9, train_data_tokens=9, seed=42)
    streams = solutions.build_disjoint_mixed_streams(
        config,
        seed=config.seed,
        device=t.device("cpu"),
        dev_examples=12,
        final_test_examples=15,
        long_test_examples=9,
        long_n_data_tokens=17,
    )
    train = streams.train_batches.reshape(-1, streams.train_batches.shape[-1])
    split_sets = [
        solutions._sequence_signatures(train),
        solutions._sequence_signatures(streams.dev_sequences),
        solutions._sequence_signatures(streams.final_test_sequences),
    ]
    assert split_sets[0].isdisjoint(split_sets[1])
    assert split_sets[0].isdisjoint(split_sets[2])
    assert split_sets[1].isdisjoint(split_sets[2])
    for sequences in (train, streams.dev_sequences, streams.final_test_sequences, streams.long_test_sequences):
        operations = solutions._operation_rows(sequences)
        assert all(bool(operations.eq(regime).any(dim=1).all()) for regime in range(3)), (
            "Every training and evaluation row must contain copy, successor, and predecessor events."
        )
    assert streams.long_test_sequences.shape[1] > streams.final_test_sequences.shape[1]
    print("All tests in `test_disjoint_streams_are_unique_and_genuinely_mixed` passed!")


def test_factorial_models_clone_identical_sparse_initialization_and_align_dense():
    solutions = _solutions()
    config = solutions.MoEConfig(d_model=12, n_heads=3, d_expert=18)
    sparse, dense = solutions.clone_factorial_models(config, seed=123)
    state_dicts = [model.state_dict() for model in sparse.values()]
    for key in state_dicts[0]:
        for state_dict in state_dicts[1:]:
            t.testing.assert_close(state_dicts[0][key], state_dict[key], rtol=0, atol=0)
    base = sparse["aux+z"]
    t.testing.assert_close(dense.token_embedding.weight, base.token_embedding.weight, rtol=0, atol=0)
    t.testing.assert_close(
        dense.blocks[0].ffn[0].weight,
        base.blocks[0].moe.expert_mlps.experts[0][0].weight,
        rtol=0,
        atol=0,
    )
    assert set(sparse) == {"aux+z", "aux-only", "z-only", "neither"}
    print("All tests in `test_factorial_models_clone_identical_sparse_initialization_and_align_dense` passed!")


def test_measured_ffn_accounting_matches_slots_and_keeps_router_overhead_separate():
    solutions = _solutions()
    config = solutions.MoEConfig(d_model=12, n_heads=3, d_expert=18)
    model = solutions.SparseMoETransformer(config).eval()
    regimes = t.tensor([[0, 1, 2, 0, 1, 2]]).repeat(9, 1)
    sequences = solutions.make_regime_sequences(regimes, t.arange(9).remainder(5), 7)
    inputs = sequences[:, :-1]
    mask = inputs.ge(solutions.REGIME_TOKEN_OFFSET) & inputs.lt(solutions.DATA_TOKEN_OFFSET)
    output = model(inputs, capacity_factor=1.0, moe_token_mask=mask)
    accounting = solutions.measured_compute_accounting(model, output.moe_outputs[0], mask)
    assert accounting["active_operation_tokens"] == 54
    assert accounting["router_evaluated_positions"] == 117
    assert accounting["sparse_executed_ffn_slots"] == 54
    assert accounting["dense_executed_ffn_slots"] == 54
    assert accounting["sparse_ffn_matmul_flops"] == accounting["dense_ffn_matmul_flops"]
    assert accounting["sparse_router_matmul_flops"] == 2 * 117 * 12 * 3
    assert accounting["sparse_router_matmul_flops"] > 2 * 54 * 12 * 3, (
        "The router linear layer runs on every input position, not only the 54 positions admitted to experts."
    )
    print("All tests in `test_measured_ffn_accounting_matches_slots_and_keeps_router_overhead_separate` passed!")


def test_router_probe_recovers_known_histograms(
    fit_router_probe: Callable | None = None,
    predict_router_probe: Callable | None = None,
):
    solutions = _solutions()
    fit_router_probe = fit_router_probe or solutions.fit_router_probe
    predict_router_probe = predict_router_probe or solutions.predict_router_probe
    features = t.eye(3).repeat_interleave(4, dim=0)
    labels = t.arange(3).repeat_interleave(4)
    weights = fit_router_probe(features, labels)
    predictions = predict_router_probe(features, weights)
    assert predictions.eq(labels).all()
    print("All tests in `test_router_probe_recovers_known_histograms` passed!")


def test_delexicalized_router_probe_hides_current_operation_token(
    delexicalized_router_event_features: Callable | None = None,
):
    solutions = _solutions()
    delexicalized_router_event_features = (
        delexicalized_router_event_features or solutions.delexicalized_router_event_features
    )
    t.manual_seed(91)
    model = solutions.SparseMoETransformer(
        solutions.MoEConfig(d_model=12, n_heads=3, d_expert=18, max_seq_len=20)
    ).eval()
    operations = t.tensor(
        [
            [0, 1, 2, 0],
            [1, 1, 2, 0],
            [2, 1, 2, 0],
        ]
    )
    sequences = solutions.make_regime_sequences(operations, t.tensor([2, 2, 2]), 5)
    features, labels = delexicalized_router_event_features(model, sequences)
    events_per_row = operations.shape[1]
    first_event_ids = t.arange(3) * events_per_row
    assert labels[first_event_ids].tolist() == [0, 1, 2]
    t.testing.assert_close(
        features[first_event_ids],
        features[first_event_ids[0]].expand(3, -1),
        rtol=1e-6,
        atol=1e-7,
        msg="Matched prefixes must have identical probe features after hiding the current operation token.",
    )
    print("All tests in `test_delexicalized_router_probe_hides_current_operation_token` passed!")


def test_causal_intervention_table_matches_direct_runs(
    causal_expert_intervention_table: Callable | None = None,
):
    solutions = _solutions()
    causal_expert_intervention_table = (
        causal_expert_intervention_table or solutions.causal_expert_intervention_table
    )
    t.manual_seed(9)
    model = solutions.SparseMoETransformer(
        solutions.MoEConfig(d_model=12, n_heads=3, d_expert=18, max_seq_len=20)
    ).eval()
    regimes = t.tensor(
        [
            [0, 1, 2, 0],
            [1, 2, 0, 1],
            [2, 0, 1, 2],
        ]
    )
    sequences = solutions.make_regime_sequences(regimes, t.tensor([0, 1, 2]), 5)
    result = causal_expert_intervention_table(model, sequences)
    assert result.intact_accuracy.shape == (3,)
    assert result.ablated_accuracy.shape == (3, 3)
    assert result.forced_accuracy.shape == (3, 3)
    for regime in range(3):
        for expert in range(3):
            expected = solutions.evaluate_sparse_by_regime(
                model,
                sequences,
                force_expert=expert,
            )[regime]
            assert float(result.forced_accuracy[regime, expert]) == pytest.approx(float(expected))
            direct_ablation = solutions.evaluate_sparse_by_regime(
                model,
                sequences,
                ablate_experts=(expert,),
            )[regime]
            assert float(result.ablated_accuracy[regime, expert]) == pytest.approx(
                float(direct_ablation)
            )
    print("All tests in `test_causal_intervention_table_matches_direct_runs` passed!")


@pytest.mark.slow
def test_short_training_run_is_real(run_smoke_test: Callable | None = None):
    solutions = _solutions()
    run_smoke_test = run_smoke_test or solutions.run_smoke_test
    report = run_smoke_test(cpu=True)
    assert report["loss_decreased"] is True
    assert report["accuracy"] >= 0.80, (
        f"The real sparse transformer reached {report['accuracy']:.3f}; expected >= 0.80. "
        "Check the target mask, causal positions, and whether dropped tokens retain the residual path."
    )
    assert report["soft_probability_entropy"] >= 0.70
    assert report["hard_load_entropy"] >= 0.70
    assert report["dropped_fraction"] <= 0.25
    print("All tests in `test_short_training_run_is_real` passed!")


def test_notebook_contract(run_smoke_test: Callable | None = None):
    solutions = _solutions()
    run_smoke_test = run_smoke_test or solutions.run_smoke_test
    report = run_smoke_test(cpu=True)
    required = {
        "tests_passed",
        "device",
        "cuda_used",
        "accuracy",
        "soft_probability_entropy",
        "hard_load_entropy",
        "dropped_fraction",
        "router_log_partition_rms",
        "router_evaluated_positions",
        "sparse_router_matmul_flops",
        "loss_decreased",
    }
    assert required <= report.keys()
    assert report["tests_passed"] is True
    assert report["device"] == "cpu" and report["cuda_used"] is False

    part = Path(__file__).resolve().parent
    learner_path = part / "5.8_Mixture_of_Experts_from_Scratch_exercises.ipynb"
    solved_path = part / "5.8_Mixture_of_Experts_from_Scratch_solutions.ipynb"
    learner = json.loads(learner_path.read_text())
    solved = json.loads(solved_path.read_text())
    learner_text = "\n".join(
        "".join(cell.get("source", [])) if isinstance(cell.get("source"), list) else cell.get("source", "")
        for cell in learner["cells"]
    )
    for needle in (
        "selected full-softmax probability",
        "topk_affinities",
        "build_disjoint_mixed_streams",
        "class DenseTransformer",
        "def train_sparse_model",
        "def train_dense_model",
        "def delexicalized_router_event_features",
        "def paired_factorial_effects",
        "def measured_compute_accounting",
        '"router_evaluated_positions"',
        "preferred_experts = dev_routing.argmax",
        "def causal_expert_intervention_table(model, sequences)",
        "def run_gpu_test(max_vram_gb: float = 24.0)",
        "def run_full_experiment(max_vram_gb: float = 24.0)",
        "Finite capacity is allocated per forward call",
        "## Try It Yourself",
        "play_capacity",
        "requested_counts",
        "full 15-logit prediction support",
        "delexicalized context probe",
        "soft probability entropy",
        "hard assignment-load entropy",
    ):
        assert needle in learner_text, f"learner notebook is missing required visible surface: {needle}"
    assert "ref.run_full_experiment" not in learner_text
    assert "random_seeds" not in learner_text
    assert "def router_event_features(" not in learner_text
    assert "run_cpu_experiment(steps=180, seeds=(7, 19, 31)" in learner_text
    assert "### Exercise 7 - compose one pre-norm MoE decoder block" in learner_text
    assert "### Exercise 8 - build and validate the cached MoE transformer" in learner_text

    errors = [
        output
        for cell in solved["cells"]
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    assert errors == []
    executed = [
        cell
        for cell in solved["cells"]
        if cell.get("cell_type") == "code" and "".join(cell.get("source", [])).strip()
    ]
    assert executed and all(cell.get("execution_count") is not None for cell in executed)
    solved_outputs = "\n".join(
        "".join(output.get("text", []))
        for cell in solved["cells"]
        for output in cell.get("outputs", [])
        if output.get("output_type") == "stream"
    )
    assert "Original positive-specialization release contract:" in solved_outputs
    print("All tests in `test_notebook_contract` passed!")


def test_answer_dropdowns_match_current_core_and_calls_exist():
    import ast
    import re

    part = Path(__file__).resolve().parent
    source = ast.parse((part / 'solutions.py').read_text())
    current = {node.name: node for node in source.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    notebook = json.loads((part / '5.8_Mixture_of_Experts_from_Scratch_exercises.ipynb').read_text())
    available_tests = {node.name for node in ast.parse(Path(__file__).read_text()).body if isinstance(node, ast.FunctionDef)}
    count = 0
    for index, cell in enumerate(notebook['cells']):
        if cell['cell_type'] != 'code':
            continue
        code = ast.parse(''.join(cell['source']))
        for node in ast.walk(code):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == 'tests':
                assert node.func.attr in available_tests, f'The notebook calls a missing test: {node.func.attr}'
        if 'exercise' not in cell['metadata'].get('tags', []):
            continue
        dropdown = re.search(r'<summary>Solution</summary>\s*```python\n(.*?)\n```', ''.join(notebook['cells'][index - 1]['source']), re.S)
        assert dropdown, f'Exercise at cell {index} needs a self-contained answer.'
        definitions = ast.parse(dropdown[1]).body
        supplied = {node.name for node in definitions if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        required = {node.name for node in code.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        assert required == supplied, f'Scaffolding and answer disagree at cell {index}: {required ^ supplied}'
        for node in definitions:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                assert ast.dump(node) == ast.dump(current[node.name]), f'The answer for {node.name} is stale.'
        count += 1
    assert count == 12, 'Keep all twelve core implementation exercises.'
    print('All tests in `test_answer_dropdowns_match_current_core_and_calls_exist` passed!')
