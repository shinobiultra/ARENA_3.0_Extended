from collections.abc import Callable

import torch as t

from arena_ext.gemma import (
    GemmaConfig,
    GemmaRMSNorm as ReferenceGemmaRMSNorm,
    GemmaForCausalLM as ReferenceGemmaForCausalLM,
    apply_rope as reference_apply_rope,
    build_causal_attention_mask as reference_build_causal_attention_mask,
    build_rope_cache as reference_build_rope_cache,
    repeat_kv as reference_repeat_kv,
    rotate_half_interleaved as reference_rotate_half_interleaved,
)


def _solutions():
    from chapter5_modern_architectures.exercises.part1_gemma_from_scratch import solutions

    return solutions


def _make_default_tiny_gemma():
    return _solutions().make_tiny_gemma(seed=0)


def test_gemma_rms_norm(GemmaRMSNorm: type | None = None):
    GemmaRMSNorm = GemmaRMSNorm or _solutions().GemmaRMSNorm
    layer = GemmaRMSNorm(hidden_size=4, eps=1e-6)
    x = t.tensor([[[1.0, 2.0, 3.0, 4.0], [-2.0, 0.5, 1.0, 3.0]]])
    expected = x * t.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    t.testing.assert_close(
        layer(x),
        expected,
        atol=1e-6,
        rtol=0.0,
        msg=(
            "GemmaRMSNorm should divide each residual vector by its RMS. "
            "Use x * rsqrt(mean(x**2) + eps), not mean-centering LayerNorm."
        ),
    )

    layer.weight.data = t.tensor([0.5, -0.25, 1.0, -0.75])
    expected = expected * (1.0 + layer.weight)
    t.testing.assert_close(
        layer(x),
        expected,
        atol=1e-6,
        rtol=0.0,
        msg="GemmaRMSNorm should multiply by (1 + weight), because Gemma stores an offset from unit scale.",
    )
    print("All tests in `test_gemma_rms_norm` passed!")


def test_rotate_half_interleaved(
    rotate_half_interleaved: Callable[[t.Tensor], t.Tensor] | None = None,
):
    rotate_half_interleaved = rotate_half_interleaved or _solutions().rotate_half_interleaved
    x = t.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
    expected = t.tensor([[[[-2.0, 1.0, -4.0, 3.0]]]])
    t.testing.assert_close(
        rotate_half_interleaved(x),
        expected,
        msg="RoPE rotation should map each even/odd pair [x0, x1] to [-x1, x0].",
    )
    print("All tests in `test_rotate_half_interleaved` passed!")


def test_build_rope_cache(
    build_rope_cache: Callable[..., tuple[t.Tensor, t.Tensor]] | None = None,
):
    build_rope_cache = build_rope_cache or _solutions().build_rope_cache
    cos, sin = build_rope_cache(
        seq_len=3,
        head_dim=4,
        base=10000.0,
        device=t.device("cpu"),
        dtype=t.float32,
    )
    assert cos.shape == (3, 4), (
        f"RoPE cos cache shape {tuple(cos.shape)} should be (seq_len, head_dim) = (3, 4)."
    )
    assert sin.shape == (3, 4), (
        f"RoPE sin cache shape {tuple(sin.shape)} should be (seq_len, head_dim) = (3, 4)."
    )
    t.testing.assert_close(
        cos[0],
        t.ones(4),
        msg="Position zero should have cos=1 for every interleaved RoPE dimension.",
    )
    t.testing.assert_close(
        sin[0],
        t.zeros(4),
        msg="Position zero should have sin=0 for every interleaved RoPE dimension.",
    )
    t.testing.assert_close(
        cos.pow(2) + sin.pow(2),
        t.ones_like(cos),
        atol=1e-6,
        rtol=0.0,
        msg="Every RoPE cache entry should lie on the unit circle: cos^2 + sin^2 = 1.",
    )
    print("All tests in `test_build_rope_cache` passed!")


def test_apply_rope(
    apply_rope: Callable[..., t.Tensor] | None = None,
    build_rope_cache: Callable[..., tuple[t.Tensor, t.Tensor]] | None = None,
):
    solutions = _solutions()
    apply_rope = apply_rope or solutions.apply_rope
    build_rope_cache = build_rope_cache or solutions.build_rope_cache
    t.manual_seed(0)
    x = t.randn(2, 4, 5, 4)
    position_ids = t.arange(5).unsqueeze(0).expand(2, -1)
    cos, sin = build_rope_cache(5, 4, base=10000.0, device=x.device, dtype=x.dtype)
    rotated = apply_rope(x, cos, sin, position_ids)
    assert rotated.shape == x.shape, (
        f"apply_rope output shape {tuple(rotated.shape)} should match input shape {tuple(x.shape)}."
    )
    t.testing.assert_close(
        rotated.norm(dim=-1),
        x.norm(dim=-1),
        atol=1e-6,
        rtol=0.0,
        msg="RoPE should rotate each head vector without changing its norm.",
    )
    t.testing.assert_close(
        rotated[:, :, 0],
        x[:, :, 0],
        atol=1e-6,
        rtol=0.0,
        msg="At position zero, RoPE should be the identity rotation.",
    )
    print("All tests in `test_apply_rope` passed!")


def test_repeat_kv(
    repeat_kv: Callable[[t.Tensor, int], t.Tensor] | None = None,
):
    repeat_kv = repeat_kv or _solutions().repeat_kv
    hidden = t.arange(2 * 2 * 3 * 4).reshape(2, 2, 3, 4)
    repeated = repeat_kv(hidden, repeats=3)
    assert repeated.shape == (2, 6, 3, 4), (
        f"repeat_kv output shape {tuple(repeated.shape)} should be (batch, kv_heads * repeats, seq, head_dim)."
    )
    for kv_head in range(2):
        for offset in range(3):
            t.testing.assert_close(
                repeated[:, kv_head * 3 + offset],
                hidden[:, kv_head],
                msg="Each key/value head should be repeated contiguously before attention matmul.",
            )
    print("All tests in `test_repeat_kv` passed!")


def test_sliding_window_mask(
    build_causal_attention_mask: Callable[..., t.Tensor] | None = None,
):
    build_causal_attention_mask = build_causal_attention_mask or _solutions().build_causal_attention_mask
    mask = build_causal_attention_mask(
        query_length=2,
        key_length=5,
        past_length=3,
        sliding_window=3,
        device=t.device("cpu"),
    ).squeeze(0).squeeze(0)
    expected = t.tensor(
        [
            [False, True, True, True, False],
            [False, False, True, True, True],
        ]
    )
    assert t.equal(mask.cpu(), expected), (
        "Sliding-window mask is wrong for cached queries at absolute positions 3 and 4. "
        f"Expected\n{expected}\nGot\n{mask.cpu()}"
    )
    print("All tests in `test_sliding_window_mask` passed!")


def test_tiny_gemma_forward_shape(
    make_tiny_gemma: Callable[[], object] = _make_default_tiny_gemma,
):
    model = make_tiny_gemma()
    input_ids = t.tensor([[1, 2, 3, 4]])
    logits = model(input_ids).logits
    assert logits.shape == (1, 4, model.config.vocab_size), (
        f"Tiny Gemma logits shape {tuple(logits.shape)} should be "
        f"(batch, seq, vocab) = (1, 4, {model.config.vocab_size})."
    )
    print("All tests in `test_tiny_gemma_forward_shape` passed!")


def test_tiny_gemma_cache_parity(
    make_tiny_gemma: Callable[[], object] = _make_default_tiny_gemma,
):
    model = make_tiny_gemma()
    input_ids = t.tensor([[1, 2, 3, 4, 5]])
    report = _solutions().cache_parity_report(model, input_ids, atol=1e-5)
    assert report["passed"], (
        "Full-context logits and cached one-token logits should match. "
        "Check RoPE position_ids, cache concatenation order, and the mask past_length. "
        f"Report: {report}"
    )
    print("All tests in `test_tiny_gemma_cache_parity` passed!")


def test_gemma_mlp_matches_swiglu_formula(GemmaMLP: type | None = None):
    GemmaMLP = GemmaMLP or _solutions().GemmaMLP
    config = GemmaConfig(
        vocab_size=31,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=16,
    )
    t.manual_seed(0)
    mlp = GemmaMLP(config)
    x = t.randn(2, 3, config.hidden_size)
    expected = mlp.down_proj(t.nn.functional.silu(mlp.gate_proj(x)) * mlp.up_proj(x))
    t.testing.assert_close(
        mlp(x),
        expected,
        msg="GemmaMLP should implement SwiGLU: down_proj(silu(gate_proj(x)) * up_proj(x)).",
    )
    print("All tests in `test_gemma_mlp_matches_swiglu_formula` passed!")


def test_gemma_attention_shapes_and_cache(GemmaAttention: type | None = None):
    GemmaAttention = GemmaAttention or _solutions().GemmaAttention
    config = GemmaConfig(
        vocab_size=31,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    t.manual_seed(5102)
    attn = GemmaAttention(config, layer_idx=0).eval()
    hidden = t.randn(2, 3, config.hidden_size)
    position_ids = t.arange(3).unsqueeze(0).expand(2, -1)
    cos, sin = reference_build_rope_cache(
        3,
        config.head_dim,
        base=config.rope_theta,
        device=hidden.device,
        dtype=hidden.dtype,
    )

    output, present = attn(
        hidden,
        position_ids=position_ids,
        cos=cos,
        sin=sin,
        use_cache=True,
    )

    assert output.shape == hidden.shape, (
        f"GemmaAttention output shape {tuple(output.shape)} should match hidden state shape "
        f"{tuple(hidden.shape)}."
    )
    assert present is not None, "GemmaAttention should return a key/value cache when use_cache=True."
    past_key, past_value = present
    expected_cache_shape = (2, config.num_key_value_heads, 3, config.head_dim)
    assert past_key.shape == expected_cache_shape, (
        f"Cached keys should have shape {expected_cache_shape}; got {tuple(past_key.shape)}."
    )
    assert past_value.shape == expected_cache_shape, (
        f"Cached values should have shape {expected_cache_shape}; got {tuple(past_value.shape)}."
    )

    next_hidden = t.randn(2, 1, config.hidden_size)
    next_position_ids = t.full((2, 1), 3)
    cos, sin = reference_build_rope_cache(
        4,
        config.head_dim,
        base=config.rope_theta,
        device=hidden.device,
        dtype=hidden.dtype,
    )
    next_output, next_present = attn(
        next_hidden,
        position_ids=next_position_ids,
        cos=cos,
        sin=sin,
        past_key_value=present,
        use_cache=True,
    )
    assert next_output.shape == next_hidden.shape, (
        "A one-token cached attention step should return one output token."
    )
    assert next_present is not None, "Cached attention should return the extended cache."
    assert next_present[0].shape[-2] == 4 and next_present[1].shape[-2] == 4, (
        "Cached attention should append the new key/value states after the past sequence."
    )
    print("All tests in `test_gemma_attention_shapes_and_cache` passed!")


def test_gemma_decoder_layer_shapes_and_cache(GemmaDecoderLayer: type | None = None):
    GemmaDecoderLayer = GemmaDecoderLayer or _solutions().GemmaDecoderLayer
    config = GemmaConfig(
        vocab_size=31,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    t.manual_seed(5103)
    layer = GemmaDecoderLayer(config, layer_idx=0).eval()
    hidden = t.randn(2, 3, config.hidden_size)
    position_ids = t.arange(3).unsqueeze(0).expand(2, -1)
    cos, sin = reference_build_rope_cache(
        3,
        config.head_dim,
        base=config.rope_theta,
        device=hidden.device,
        dtype=hidden.dtype,
    )

    output, present = layer(
        hidden,
        position_ids=position_ids,
        cos=cos,
        sin=sin,
        use_cache=True,
    )

    assert output.shape == hidden.shape, (
        f"GemmaDecoderLayer output shape {tuple(output.shape)} should match input shape "
        f"{tuple(hidden.shape)}."
    )
    assert present is not None, "GemmaDecoderLayer should propagate the attention cache."
    assert present[0].shape[-2] == hidden.shape[1], (
        "GemmaDecoderLayer cache length should match the current sequence length."
    )
    print("All tests in `test_gemma_decoder_layer_shapes_and_cache` passed!")


def test_tiny_gemma_matches_reference_decoder(GemmaForCausalLM: type | None = None):
    GemmaForCausalLM = GemmaForCausalLM or _solutions().GemmaForCausalLM
    config = GemmaConfig(
        vocab_size=31,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    t.manual_seed(5101)
    reference = ReferenceGemmaForCausalLM(config).eval()
    local = GemmaForCausalLM(config).eval()
    local.load_state_dict(reference.state_dict())
    input_ids = t.tensor([[1, 5, 8, 13, 2]])
    attention_mask = t.ones_like(input_ids)
    with t.inference_mode():
        actual = local(input_ids, attention_mask=attention_mask, use_cache=False).logits
        expected = reference(input_ids, attention_mask=attention_mask, use_cache=False).logits
    t.testing.assert_close(
        actual,
        expected,
        atol=5e-4,
        rtol=0.0,
        msg=(
            "The local Gemma decoder should match the independent arena_ext reference "
            "after loading an identical state dict. Check embedding scaling, RoPE layout, "
            "RMSNorm offset weights, and attention masks."
        ),
    )
    assert local.lm_head.weight.data_ptr() == local.model.embed_tokens.weight.data_ptr(), (
        "GemmaForCausalLM should tie lm_head.weight to model.embed_tokens.weight when "
        "config.tie_word_embeddings=True."
    )
    print("All tests in `test_tiny_gemma_matches_reference_decoder` passed!")


def test_capture_gemma_trace(
    capture_gemma_trace: Callable | None = None,
    make_tiny_gemma: Callable[[], object] = _make_default_tiny_gemma,
):
    capture_gemma_trace = capture_gemma_trace or _solutions().capture_gemma_trace
    model = make_tiny_gemma().eval()
    input_ids = t.tensor([[1, 5, 8, 13]])
    trace = capture_gemma_trace(model, input_ids)
    expected_stages = ["embedding", "layer_0", "layer_1", "final_norm", "logits"]
    assert list(trace) == expected_stages, (
        f"Expected trace stages {expected_stages}; got {list(trace)}."
    )
    for stage in expected_stages[:-1]:
        assert trace[stage].shape == (1, 4, model.config.hidden_size), (
            f"Stage {stage} should have residual-stream shape (1, 4, {model.config.hidden_size})."
        )
    assert trace["logits"].shape == (1, 4, model.config.vocab_size), (
        "The final trace entry should contain one vocabulary logit vector per token; "
        f"got shape {tuple(trace['logits'].shape)}."
    )
    print("All tests in `test_capture_gemma_trace` passed!")


def test_architecture_controls(run_architecture_controls: Callable | None = None):
    run_architecture_controls = run_architecture_controls or _solutions().run_architecture_controls
    result = run_architecture_controls()
    profiles = result["profiles"]
    expected_profiles = [
        "exact",
        "no embedding scale",
        "no RoPE",
        "wrong GQA order",
    ]
    assert list(profiles) == expected_profiles, (
        "Architecture controls should keep a stable plotting order so each failure can be "
        f"compared with the exact model; expected {expected_profiles}, got {list(profiles)}."
    )
    assert profiles["exact"]["logits"]["max_abs"] <= 5e-6, (
        "The completed learner model should match the independent reference at the logits."
    )
    assert profiles["no embedding scale"]["embedding"]["relative_rmse"] > 0.5, (
        "Removing Gemma's sqrt(hidden_size) embedding scale should visibly diverge at the embedding."
    )
    assert profiles["no RoPE"]["embedding"]["max_abs"] == 0.0, (
        "Removing RoPE should leave token embeddings unchanged; an embedding mismatch means "
        "the control changed more than the positional rotation."
    )
    assert profiles["no RoPE"]["layer_0"]["max_abs"] > 1e-3, (
        "The no-RoPE control should first diverge inside the first decoder layer."
    )
    assert profiles["wrong GQA order"]["embedding"]["max_abs"] == 0.0, (
        "Changing grouped-query head order should leave token embeddings unchanged; an "
        "embedding mismatch means the control is not isolating GQA."
    )
    assert profiles["wrong GQA order"]["layer_0"]["max_abs"] > 1e-3, (
        "The wrong-head-order control should first diverge inside the first decoder layer."
    )
    for control in ["no embedding scale", "no RoPE", "wrong GQA order"]:
        assert profiles[control]["logits"]["max_abs"] > 1e-2, (
            f"Control {control!r} did not produce a meaningful final-logit difference."
        )
    print("All tests in `test_architecture_controls` passed!")


def test_notebook_contract(run_smoke_test: Callable | None = None):
    if run_smoke_test is None:
        from chapter5_modern_architectures.exercises.part1_gemma_from_scratch.solutions import (
            run_smoke_test as default_run_smoke_test,
        )

        run_smoke_test = default_run_smoke_test

    result = run_smoke_test(cpu=True)
    assert result["rms_norm_passed"], "RMSNorm smoke test failed; inspect `test_gemma_rms_norm` first."
    assert result["rope_norm_passed"], "RoPE norm preservation failed; inspect `test_apply_rope` first."
    assert result["cache_parity"]["passed"], (
        "Tiny Gemma cache parity failed; full-sequence and cached logits should agree."
    )
    assert result["clone_parity"]["topk_agreement"] == 1.0, (
        "Exact-clone parity should have top-k agreement 1.0. "
        f"Got {result['clone_parity']}"
    )
    test_capture_gemma_trace()
    test_architecture_controls()
    print("All tests in `test_notebook_contract` passed!")
