"""Visible semantic tests for [5.3] Mamba from Scratch."""

from __future__ import annotations

from collections.abc import Callable

import torch as t
import torch.nn as nn


def _solutions():
    from chapter5_modern_architectures.exercises.part3_mamba_from_scratch import solutions

    return solutions


def test_continuous_discretization_exact(
    stable_continuous_A: Callable | None = None,
    discretize_scalar_ssm_exact: Callable | None = None,
):
    solutions = _solutions()
    stable_continuous_A = stable_continuous_A or solutions.stable_continuous_A
    discretize_scalar_ssm_exact = (
        discretize_scalar_ssm_exact or solutions.discretize_scalar_ssm_exact
    )
    A = stable_continuous_A(t.log(t.tensor([2.0], dtype=t.float64)))
    a_bar, b_bar = discretize_scalar_ssm_exact(
        A,
        t.tensor([3.0], dtype=t.float64),
        t.tensor([0.5], dtype=t.float64),
    )
    t.testing.assert_close(
        A,
        t.tensor([-2.0]),
        msg="A=-exp(A_log) must place every continuous-time mode in the stable half-plane.",
    )
    t.testing.assert_close(
        a_bar,
        t.tensor([t.exp(t.tensor(-1.0))], dtype=t.float64),
        atol=1e-7,
        rtol=0,
        msg="The scalar transition must equal exp(delta*A).",
    )
    expected_b = 1.5 * (1 - t.exp(t.tensor(-1.0, dtype=t.float64)))
    t.testing.assert_close(
        b_bar,
        expected_b[None],
        atol=1e-7,
        rtol=0,
        msg="The exact zero-order-hold input coefficient is (exp(delta*A)-1)/A*B.",
    )
    print("All tests in `test_continuous_discretization_exact` passed!")


def test_sequential_affine_scan_manual(
    sequential_affine_scan: Callable | None = None,
):
    sequential_affine_scan = sequential_affine_scan or _solutions().sequential_affine_scan
    a = t.tensor([[[0.5], [0.25], [1.0]]], dtype=t.float64)
    b = t.tensor([[[1.0], [-1.0], [3.0]]], dtype=t.float64)
    actual = sequential_affine_scan(a, b, t.tensor([[2.0]], dtype=t.float64))
    expected = t.tensor([[[2.0], [-0.5], [2.5]]], dtype=t.float64)
    t.testing.assert_close(
        actual,
        expected,
        atol=1e-15,
        rtol=1e-15,
        msg="The literal left-to-right recurrence disagrees with the hand-computed states.",
    )
    print("All tests in `test_sequential_affine_scan_manual` passed!")


def test_compose_affine_updates(
    compose_affine_updates: Callable | None = None,
):
    compose_affine_updates = compose_affine_updates or _solutions().compose_affine_updates
    a_left = t.tensor([0.5, 0.2], dtype=t.float64)
    b_left = t.tensor([1.0, -2.0], dtype=t.float64)
    a_right = t.tensor([0.4, 0.3], dtype=t.float64)
    b_right = t.tensor([-1.0, 5.0], dtype=t.float64)
    a, b = compose_affine_updates(a_left, b_left, a_right, b_right)
    x = t.tensor([3.0, -4.0], dtype=t.float64)
    t.testing.assert_close(
        a * x + b,
        a_right * (a_left * x + b_left) + b_right,
        atol=1e-15,
        rtol=1e-15,
        msg="Affine composition must preserve the chronological order left then right.",
    )
    print("All tests in `test_compose_affine_updates` passed!")


def test_parallel_affine_scan_parity(
    sequential_affine_scan: Callable | None = None,
    parallel_affine_scan: Callable | None = None,
):
    solutions = _solutions()
    sequential_affine_scan = sequential_affine_scan or solutions.sequential_affine_scan
    parallel_affine_scan = parallel_affine_scan or solutions.parallel_affine_scan
    generator = t.Generator().manual_seed(11)
    a = 0.4 + 0.55 * t.rand(3, 17, 5, generator=generator, dtype=t.float64)
    b = t.randn(3, 17, 5, generator=generator, dtype=t.float64)
    initial = t.randn(3, 5, generator=generator, dtype=t.float64)
    expected = sequential_affine_scan(a, b, initial)
    actual = parallel_affine_scan(a, b, initial)
    t.testing.assert_close(
        actual,
        expected,
        atol=1e-12,
        rtol=1e-12,
        msg="The associative prefix schedule must reproduce every sequential state.",
    )
    print("All tests in `test_parallel_affine_scan_parity` passed!")


def test_parallel_scan_gradients(
    parallel_affine_scan: Callable | None = None,
):
    parallel_affine_scan = parallel_affine_scan or _solutions().parallel_affine_scan
    a = t.full((2, 8, 3), 0.8, requires_grad=True)
    b = t.randn(2, 8, 3, requires_grad=True)
    parallel_affine_scan(a, b).square().mean().backward()
    assert a.grad is not None and bool(t.isfinite(a.grad).all()), (
        "The training-time parallel scan must propagate finite gradients into transition coefficients."
    )
    assert b.grad is not None and bool(t.isfinite(b.grad).all()), (
        "The training-time parallel scan must propagate finite gradients into input updates."
    )
    print("All tests in `test_parallel_scan_gradients` passed!")


def test_causal_depthwise_conv_manual_and_no_future(
    causal_depthwise_conv1d: Callable | None = None,
):
    causal_depthwise_conv1d = (
        causal_depthwise_conv1d or _solutions().causal_depthwise_conv1d
    )
    x = t.tensor([[[1.0, 2.0], [3.0, 5.0], [7.0, 11.0], [13.0, 17.0]]])
    weight = t.tensor([[[1.0, 10.0, 100.0]], [[-1.0, 2.0, 3.0]]])
    actual = causal_depthwise_conv1d(x, weight)
    expected = t.empty_like(actual)
    for channel in range(2):
        for position in range(4):
            total = 0.0
            for kernel_position in range(3):
                source = position - 2 + kernel_position
                if source >= 0:
                    total += x[0, source, channel] * weight[channel, 0, kernel_position]
            expected[0, position, channel] = total
    t.testing.assert_close(
        actual,
        expected,
        msg="Depthwise convolution must match a hand-written causal cross-correlation.",
    )
    changed = x.clone()
    changed[:, -1] += 10_000
    t.testing.assert_close(
        causal_depthwise_conv1d(changed, weight)[:, :-1],
        actual[:, :-1],
        msg="Changing a future token must not alter any earlier causal-convolution output.",
    )
    print("All tests in `test_causal_depthwise_conv_manual_and_no_future` passed!")


def test_project_selective_parameters_is_input_dependent(
    project_selective_parameters: Callable | None = None,
):
    project_selective_parameters = (
        project_selective_parameters or _solutions().project_selective_parameters
    )
    t.manual_seed(3)
    x_proj = nn.Linear(4, 2 + 2 * 3, bias=False)
    dt_proj = nn.Linear(2, 4)
    x = t.randn(2, 5, 4)
    delta, B, C = project_selective_parameters(x, x_proj, dt_proj, d_state=3)
    assert delta.shape == (2, 5, 4) and B.shape == C.shape == (2, 5, 3), (
        "delta, B, and C must preserve batch/sequence while exposing inner/state axes."
    )
    assert bool((delta > 0).all()), "softplus must make every discretization step positive."
    changed = x.clone()
    changed[:, 2] += 2.0
    changed_delta, changed_B, changed_C = project_selective_parameters(
        changed, x_proj, dt_proj, d_state=3
    )
    assert not t.equal(delta[:, 2], changed_delta[:, 2]), (
        "delta must change when the current token representation changes."
    )
    assert not t.equal(B[:, 2], changed_B[:, 2]), (
        "B must be input-dependent rather than a fixed SSM parameter."
    )
    assert not t.equal(C[:, 2], changed_C[:, 2]), (
        "C must be input-dependent rather than a fixed readout."
    )
    print("All tests in `test_project_selective_parameters_is_input_dependent` passed!")


def test_selective_discretization_shapes_and_stability(
    discretize_selective_scan: Callable | None = None,
):
    solutions = _solutions()
    discretize_selective_scan = (
        discretize_selective_scan or solutions.discretize_selective_scan
    )
    u, delta, A_log, B, *_ = solutions.make_scan_inputs(seed=0)
    a, b = discretize_selective_scan(u, delta, A_log, B)
    assert a.shape == b.shape == (2, 7, 4, 3), (
        f"Expected recurrence tensors shaped (2, 7, 4, 3), got {a.shape} and {b.shape}."
    )
    assert bool(((a > 0) & (a <= 1)).all()), (
        "Stable A and positive delta require every discrete decay to lie in (0, 1]."
    )
    print("All tests in `test_selective_discretization_shapes_and_stability` passed!")


def test_selective_scan_single_step_manual(
    selective_scan_recurrent: Callable | None = None,
):
    selective_scan_recurrent = (
        selective_scan_recurrent or _solutions().selective_scan_recurrent
    )
    u = t.tensor([[[2.0]]])
    delta = t.tensor([[[0.5]]])
    A_log = t.log(t.tensor([[2.0]]))
    B = t.tensor([[[3.0]]])
    C = t.tensor([[[5.0]]])
    D = t.tensor([7.0])
    gate = t.tensor([[[1.0]]])
    output, last_state = selective_scan_recurrent(
        u,
        delta,
        A_log,
        B,
        C,
        D,
        gate,
        return_last_state=True,
    )
    t.testing.assert_close(
        last_state,
        t.tensor([[[3.0]]]),
        msg="With zero initial state, the one-step state must equal delta*B*u.",
    )
    expected = (15.0 + 14.0) * t.nn.functional.silu(t.tensor(1.0))
    t.testing.assert_close(
        output.squeeze(),
        expected,
        msg="Readout, D skip, and SiLU gate must be applied in Mamba order.",
    )
    print("All tests in `test_selective_scan_single_step_manual` passed!")


def test_recurrent_parallel_selective_scan_parity(
    selective_scan_recurrent: Callable | None = None,
    selective_scan_parallel: Callable | None = None,
):
    solutions = _solutions()
    selective_scan_recurrent = (
        selective_scan_recurrent or solutions.selective_scan_recurrent
    )
    selective_scan_parallel = (
        selective_scan_parallel or solutions.selective_scan_parallel
    )
    inputs = solutions.make_scan_inputs(seed=7)
    recurrent = selective_scan_recurrent(*inputs)
    parallel = selective_scan_parallel(*inputs)
    t.testing.assert_close(
        parallel,
        recurrent,
        atol=1e-5,
        rtol=1e-5,
        msg="Parallel training and recurrent inference must implement one selective recurrence.",
    )
    print("All tests in `test_recurrent_parallel_selective_scan_parity` passed!")


def test_mamba_block_has_norm_residual_and_gate(
    MambaBlockFromScratch: type | None = None,
):
    solutions = _solutions()
    MambaBlockFromScratch = MambaBlockFromScratch or solutions.MambaBlockFromScratch
    config = solutions.make_tiny_mamba_config()
    block = MambaBlockFromScratch(config).eval()
    assert hasattr(block, "norm") and hasattr(block, "mixer"), (
        "A Mamba block must contain pre-normalization and a mixer, not just the recurrence."
    )
    assert block.mixer.in_proj.out_features == 2 * config.d_inner, (
        "in_proj must create separate x and gate streams."
    )
    hidden = t.randn(2, 5, config.d_model)
    with t.no_grad():
        block.mixer.out_proj.weight.zero_()
        if block.mixer.out_proj.bias is not None:
            block.mixer.out_proj.bias.zero_()
        output, _ = block(hidden)
    t.testing.assert_close(
        output,
        hidden,
        msg="Zeroing the mixer output must expose the block's residual identity path.",
    )
    print("All tests in `test_mamba_block_has_norm_residual_and_gate` passed!")


def test_mamba_block_recurrent_step_parity(
    MambaBlockFromScratch: type | None = None,
):
    solutions = _solutions()
    MambaBlockFromScratch = MambaBlockFromScratch or solutions.MambaBlockFromScratch
    t.manual_seed(5)
    config = solutions.make_tiny_mamba_config()
    block = MambaBlockFromScratch(config).eval()
    hidden = t.randn(2, 7, config.d_model)
    full, _ = block(hidden)
    state = block.mixer.initial_state(2, hidden.device, hidden.dtype)
    outputs = []
    for position in range(hidden.shape[1]):
        output, state = block(
            hidden[:, position : position + 1],
            inference_state=state,
            use_cache=True,
        )
        outputs.append(output)
    t.testing.assert_close(
        t.cat(outputs, dim=1),
        full,
        atol=1e-5,
        rtol=1e-5,
        msg="One-token recurrent block execution must equal the full causal sequence path.",
    )
    print("All tests in `test_mamba_block_recurrent_step_parity` passed!")


def test_tiny_stacked_mamba_lm_and_cache(
    MambaForCausalLMFromScratch: type | None = None,
):
    solutions = _solutions()
    MambaForCausalLMFromScratch = (
        MambaForCausalLMFromScratch or solutions.MambaForCausalLMFromScratch
    )
    t.manual_seed(6)
    config = solutions.make_tiny_mamba_config()
    model = MambaForCausalLMFromScratch(config).eval()
    input_ids = t.tensor([[1, 2, 3, 4, 5]])
    full = model(input_ids, output_hidden_states=True)
    assert full.logits.shape == (1, 5, config.vocab_size), (
        "The stacked causal LM must return one vocabulary logit vector per token."
    )
    assert full.hidden_states is not None and len(full.hidden_states) == config.num_layers + 1, (
        "Hidden-state parity needs every post-block state plus the final normalized state."
    )
    assert model.lm_head.weight.data_ptr() == model.backbone.embeddings.weight.data_ptr(), (
        "The pinned checkpoint ties embedding and LM-head weights."
    )
    states = None
    cached = []
    for position in range(input_ids.shape[1]):
        output = model(
            input_ids[:, position : position + 1],
            states=states,
            use_cache=True,
        )
        states = output.states
        cached.append(output.logits)
    t.testing.assert_close(
        t.cat(cached, dim=1),
        full.logits,
        atol=1e-5,
        rtol=1e-5,
        msg="The stacked LM cache must reproduce full-context logits token by token.",
    )
    print("All tests in `test_tiny_stacked_mamba_lm_and_cache` passed!")


def test_pinned_mamba_130m_weight_and_output_parity(
    pinned_mamba_cpu_parity: Callable | None = None,
):
    pinned_mamba_cpu_parity = (
        pinned_mamba_cpu_parity or _solutions().pinned_mamba_cpu_parity
    )
    report = pinned_mamba_cpu_parity()
    assert report["mapped_tensor_count"] == 243, (
        f"Expected all 243 checkpoint tensors to map; got {report['mapped_tensor_count']}."
    )
    assert report["missing_keys"] == [] and report["unexpected_keys"] == [], (
        "Strict weight loading must have no missing or unexpected checkpoint keys."
    )
    assert report["hidden_state_count"] == 25, (
        "The 24-layer checkpoint must expose 24 post-block states plus final RMSNorm."
    )
    assert report["hidden_max_abs_diff"] <= 3e-3, (
        f"Mapped hidden states drifted beyond CPU tolerance: {report['hidden_max_abs_diff']:.3e}."
    )
    assert report["logits_max_abs_diff"] <= 5e-4, (
        f"Mapped logits drifted beyond CPU tolerance: {report['logits_max_abs_diff']:.3e}."
    )
    assert report["hidden_max_mean_abs_diff"] <= 1e-4, (
        "Mean hidden-state drift should remain tiny at every layer, not be hidden by a max-only metric."
    )
    assert report["logits_mean_abs_diff"] <= 3e-5, (
        "Mean logit drift should remain below 3e-5 across the full vocabulary."
    )
    assert report["logits_top1_agreement"] == 1.0, (
        "Mapped and official logits must choose the same top token at every checked position."
    )
    assert report["cache_max_abs_diff"] <= 1e-3, (
        f"Local recurrent cache drifted beyond CPU tolerance: {report['cache_max_abs_diff']:.3e}."
    )
    assert report["greedy_tokens_match"], (
        "Mapped and official Mamba-130M must choose the same deterministic greedy tokens."
    )
    print("All tests in `test_pinned_mamba_130m_weight_and_output_parity` passed!")


def test_selective_memory_signature_has_transformer_and_conv_controls(
    run_memory_delay_benchmark: Callable | None = None,
):
    run_memory_delay_benchmark = (
        run_memory_delay_benchmark or _solutions().run_memory_delay_benchmark
    )
    result = run_memory_delay_benchmark()
    assert result.selective_error.max().item() == 0.0, (
        "The exact selective recurrence must preserve every written value at every tested delay."
    )
    assert result.attention_error.max().item() <= 1e-10, (
        "The all-context causal-attention ceiling should retrieve the unique WRITE token."
    )
    assert result.convolution_error[:3].max().item() == 0.0, (
        "The width-8 convolution should succeed while the write remains in its receptive field."
    )
    assert result.convolution_error[-1].item() >= 0.9, (
        "The finite convolution must fail visibly when the write is far outside its receptive field."
    )
    assert result.trace_state.unique().numel() == 1, (
        "The long-delay selective state should hold one exact value from WRITE through READ."
    )
    print(
        "All tests in `test_selective_memory_signature_has_transformer_and_conv_controls` passed!"
    )


def test_state_intervention_is_causal(
    intervene_on_state: Callable | None = None,
):
    solutions = _solutions()
    intervene_on_state = intervene_on_state or solutions.intervene_on_state
    case = solutions.make_selective_copy_case()
    a, b, c = solutions.build_event_coefficients(case.event_ids, case.values)
    baseline = solutions.selective_readout(solutions.sequential_affine_scan(a, b), c)
    ablated = solutions.selective_readout(intervene_on_state(a, b, position=6), c)
    reads = case.read_positions
    t.testing.assert_close(
        baseline[0, reads[0]],
        ablated[0, reads[0]],
        msg="An intervention after the first read cannot retroactively change it.",
    )
    assert abs(ablated[0, reads[1]].item()) < 1e-12, (
        "Zeroing the state after the second WRITE must remove its downstream read."
    )
    print("All tests in `test_state_intervention_is_causal` passed!")


def test_notebook_contract(run_smoke_test: Callable | None = None):
    run_smoke_test = run_smoke_test or _solutions().run_smoke_test
    report = run_smoke_test(cpu=True)
    assert report["scan_parity"]["passed"], (
        "Recurrent and associative selective scans must agree."
    )
    assert report["block_step_parity"]["passed"], (
        "Full block execution and recurrent token stepping must agree."
    )
    assert report["lm_cache_parity"]["passed"], (
        "The tiny stacked LM cache must reproduce full logits."
    )
    assert report["cpu_mamba_130m_mapped_tensor_count"] == 243, (
        "The notebook contract must include strict real-checkpoint weight mapping."
    )
    assert report["cpu_mamba_130m_greedy_tokens_match"], (
        "The notebook contract must include exact official/custom greedy-token parity."
    )
    assert report["memory_benchmark"]["convolution_long_delay_mae"] >= 0.9, (
        "The signature control must expose the finite convolution's long-delay failure."
    )
    print("All tests in `test_notebook_contract` passed!")


def test_selective_parameters_literal(project_fn, discretize_fn, expand_fn):
    import math
    x_proj=nn.Linear(1,3,bias=False);dt_proj=nn.Linear(1,1,bias=True)
    with t.no_grad():
        x_proj.weight.copy_(t.tensor([[1.],[2.],[3.]]));dt_proj.weight.fill_(2.);dt_proj.bias.zero_()
    x=t.tensor([[[-1.],[1.]]])
    delta,B,C=project_fn(x,x_proj,dt_proj,1)
    t.testing.assert_close(delta.flatten(),t.tensor([math.log1p(math.exp(-2)),math.log1p(math.exp(2))]),rtol=1e-6,atol=1e-6)
    t.testing.assert_close(B.flatten(),t.tensor([-2.,2.]),rtol=0,atol=0)
    t.testing.assert_close(C.flatten(),t.tensor([-3.,3.]),rtol=0,atol=0)
    expanded=expand_fn(t.tensor([[[2.,3.]]]),3)
    t.testing.assert_close(expanded,t.tensor([[[[2.,3.],[2.,3.],[2.,3.]]]]),rtol=0,atol=0)
    import pytest
    with pytest.raises(ValueError):expand_fn(t.ones(1,2),3)
    a,b=discretize_fn(t.tensor([[[2.],[-1.]]]),t.tensor([[[.5],[.25]]]),t.tensor([[math.log(2.)]]),t.tensor([[[3.],[4.]]]))
    t.testing.assert_close(a.flatten(),t.tensor([math.exp(-1),math.exp(-.5)]),rtol=1e-6,atol=1e-6)
    t.testing.assert_close(b.flatten(),t.tensor([3.,-1.]),rtol=0,atol=0)
    print('All tests in `test_selective_parameters_literal` passed!')


def test_rms_norm_literal(norm_cls):
    import math
    layer=norm_cls(2,eps=0.)
    with t.no_grad():layer.weight.copy_(t.tensor([2.,3.]))
    actual=layer(t.tensor([[3.,4.],[0.,2.]]))
    expected=t.tensor([[6/math.sqrt(12.5),12/math.sqrt(12.5)],[0.,3*math.sqrt(2)]])
    t.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-6)
    print('All tests in `test_rms_norm_literal` passed!')


def test_mixer_literal_oracle(mixer_cls,config_cls):
    import math
    config=config_cls(vocab_size=3,d_model=1,d_inner=2,d_state=1,d_conv=1,dt_rank=1,num_layers=1)
    model=mixer_cls(config)
    with t.no_grad():
        model.in_proj.weight.copy_(t.tensor([[2.],[-1.],[.2],[-.3]]))
        model.conv1d.weight.fill_(1.);model.conv1d.bias.zero_()
        model.x_proj.weight.copy_(t.tensor([[.1,.2],[.4,.1],[-.6,.2]]))
        model.dt_proj.weight.copy_(t.tensor([[1.],[2.]]));model.dt_proj.bias.copy_(t.tensor([-.7,.3]))
        model.A_log.fill_(math.log(.2));model.D.copy_(t.tensor([.3,.5]));model.out_proj.weight.copy_(t.tensor([[1.,2.]]))
    silu=lambda v:v/(1+math.exp(-v));state=[0.,0.];expected=[]
    for u in (1.,-.5):
        x=[silu(2*u),silu(-u)]
        raw_dt=.1*x[0]+.2*x[1];B=.4*x[0]+.1*x[1];C=-.6*x[0]+.2*x[1]
        dt=[math.log1p(math.exp(raw_dt-.7)),math.log1p(math.exp(2*raw_dt+.3))]
        state=[math.exp(-.2*d)*h+d*B*v for d,h,v in zip(dt,state,x)]
        y=[(h*C+D*v)*silu(g*u) for h,D,v,g in zip(state,(.3,.5),x,(.2,-.3))]
        expected.append(y[0]+2*y[1])
    actual,cache=model(t.tensor([[[1.],[-.5]]]),use_cache=True)
    t.testing.assert_close(actual.flatten(),t.tensor(expected),rtol=2e-6,atol=2e-6)
    t.testing.assert_close(cache.ssm_state.flatten(),t.tensor(state),rtol=2e-6,atol=2e-6)
    print('All tests in `test_mixer_literal_oracle` passed!')


def test_mixer_cache_storage(mixer_cls,config_cls):
    with t.random.fork_rng():
        t.manual_seed(0)
        model=mixer_cls(config_cls(vocab_size=3,d_model=1,d_inner=2,d_state=1,d_conv=3,dt_rank=1,num_layers=1))
    initial=model.initial_state(2,t.device('cpu'),t.float32)
    assert initial.conv_state.shape==(2,2,3) and initial.ssm_state.shape==(2,2,1)
    assert t.count_nonzero(initial.conv_state)==t.count_nonzero(initial.ssm_state)==0
    x=t.tensor([[[1.,2.],[3.,4.]]]);before=x.clone()
    actual=model._full_conv_state(x)
    t.testing.assert_close(actual,t.tensor([[[0.,1.,3.],[0.,2.,4.]]]),rtol=0,atol=0)
    t.testing.assert_close(x,before,rtol=0,atol=0)
    print('All tests in `test_mixer_cache_storage` passed!')


def test_weight_mapping_without_checkpoint(key_fn,load_fn):
    from types import SimpleNamespace
    import pytest
    for key in ('backbone.embeddings.weight','backbone.norm_f.weight','lm_head.weight','backbone.layers.0.mixer.A_log'):
        assert key_fn(key)==key
    with pytest.raises(KeyError):key_fn('unrecognized.weight')
    calls=[]
    class Fake:
        def load_state_dict(self,state,strict):
            calls.append((state,strict));return SimpleNamespace(missing_keys=['m'],unexpected_keys=['u'])
    state={'backbone.embeddings.weight':t.tensor([[1.,2.]])}
    result=load_fn(Fake(),state)
    assert len(calls)==1 and calls[0][1] is True
    assert calls[0][0].keys()==state.keys()
    t.testing.assert_close(calls[0][0]['backbone.embeddings.weight'],state['backbone.embeddings.weight'],rtol=0,atol=0)
    assert result=={'mapped_tensor_count':1,'missing_keys':['m'],'unexpected_keys':['u']}
    print('All tests in `test_weight_mapping_without_checkpoint` passed!')


def test_parity_assembly_without_checkpoint(parity_fn):
    from inspect import unwrap
    from types import SimpleNamespace
    from unittest.mock import patch
    logits=t.tensor([[[1.,4.],[3.,1.],[2.,5.],[4.,0.]]])
    reference=logits+.25;reference[0,0,0]=5.
    generated=t.tensor([[10,314,2718,11,2,3,4]])
    class Official:
        def __call__(self,ids,**kwargs):
            assert ids.tolist()==[[10,314,2718,11]]
            return SimpleNamespace(logits=reference,hidden_states=(t.tensor([1.,4.]),t.tensor([3.,4.])))
        def generate(self,*args,**kwargs):
            assert kwargs['max_new_tokens']==3 and kwargs['do_sample'] is False
            result=generated.clone();result[0,-1]=5;return result
    class Local:
        def __call__(self,ids,**kwargs):
            if kwargs.get('use_cache'):
                i={10:0,314:1,2718:2,11:3}[ids.item()]
                return SimpleNamespace(logits=logits[:,i:i+1]+.5,states=i)
            return SimpleNamespace(logits=logits,hidden_states=(t.tensor([1.,2.]),t.tensor([3.,4.])))
        def greedy_generate(self,*args,**kwargs):
            assert kwargs['max_new_tokens']==3;return generated
    with patch.dict(unwrap(parity_fn).__globals__,{'load_pinned_mamba_130m_cpu':lambda:(Official(),Local(),{'mapped_tensor_count':243,'missing_keys':[],'unexpected_keys':[]})}):
        result=parity_fn()
    assert result['mapped_tensor_count']==243 and result['hidden_state_count']==2
    assert result['hidden_max_abs_diff']==2. and result['hidden_max_mean_abs_diff']==1. and result['hidden_worst_layer']==0
    assert result['logits_max_abs_diff']==4. and result['logits_mean_abs_diff']==.71875 and result['logits_top1_agreement']==.75
    assert result['cache_max_abs_diff']==.5 and result['greedy_tokens_match'] is False
    assert result['generated_tokens']==generated[0].tolist()
    print('All tests in `test_parity_assembly_without_checkpoint` passed!')


def test_event_controls_literal(coefficients_fn,read_fn,intervention_fn,attention_fn,conv_fn):
    import pytest
    events=t.tensor([[1,0,2,3,2]])
    values=t.tensor([[2.,9.,0.,100.,0.]],dtype=t.float64)
    a,b,c=coefficients_fn(events,values)
    t.testing.assert_close(a.flatten(),t.tensor([0.,1.,1.,0.,1.],dtype=t.float64),rtol=0,atol=0)
    t.testing.assert_close(b.flatten(),t.tensor([2.,0.,0.,0.,0.],dtype=t.float64),rtol=0,atol=0)
    t.testing.assert_close(c.flatten(),t.tensor([0.,0.,1.,0.,1.],dtype=t.float64),rtol=0,atol=0)
    a_before,b_before=a.clone(),b.clone()
    intervened=intervention_fn(a,b,1,replacement=-3.)
    t.testing.assert_close(intervened.flatten(),t.tensor([2.,-3.,-3.,0.,0.],dtype=t.float64),rtol=0,atol=0)
    t.testing.assert_close(read_fn(intervened,c).flatten(),t.tensor([0.,0.,-3.,0.,0.],dtype=t.float64),rtol=0,atol=0)
    t.testing.assert_close(a,a_before,rtol=0,atol=0);t.testing.assert_close(b,b_before,rtol=0,atol=0)
    with pytest.raises(IndexError):intervention_fn(a,b,-1)
    events=t.tensor([[1,0,2,1]]);values=t.tensor([[6.,100.,0.,1000.]],dtype=t.float64)
    t.testing.assert_close(attention_fn(events,values,2,logit_gap=0.),t.tensor([2.],dtype=t.float64),rtol=0,atol=1e-12)
    t.testing.assert_close(attention_fn(events,values,2),t.tensor([6.],dtype=t.float64),rtol=0,atol=1e-10)
    t.testing.assert_close(conv_fn(events,values,2),t.tensor([[6.,6.,0.,1000.]],dtype=t.float64),rtol=0,atol=0)
    print('All tests in `test_event_controls_literal` passed!')


def test_delay_benchmark_literal(benchmark_fn):
    result=benchmark_fn(distances=(1,2,3),conv_kernel=2,values=(-2.,4.))
    t.testing.assert_close(result.distances,t.tensor([1,2,3]),rtol=0,atol=0)
    t.testing.assert_close(result.selective_error,t.zeros(3,dtype=t.float64),rtol=0,atol=0)
    t.testing.assert_close(result.attention_error,t.zeros(3,dtype=t.float64),rtol=0,atol=1e-10)
    t.testing.assert_close(result.convolution_error,t.tensor([0.,3.,3.],dtype=t.float64),rtol=0,atol=0)
    t.testing.assert_close(result.trace_state,t.full((4,),4.,dtype=t.float64),rtol=0,atol=0)
    print('All tests in `test_delay_benchmark_literal` passed!')
