import importlib.util

import pytest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
TRANSFORMERS_AVAILABLE = importlib.util.find_spec("transformers") is not None
pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is not installed")

if TORCH_AVAILABLE:
    import torch as t

    from arena_ext.gemma import (
        GemmaConfig,
        GemmaForCausalLM,
        GemmaRMSNorm,
        apply_rope,
        build_causal_attention_mask,
        build_rope_cache,
        cache_parity_report,
        load_matching_state_dict,
        repeat_kv,
    )

if TRANSFORMERS_AVAILABLE:
    from transformers import GemmaConfig as HFGemmaConfig
    from transformers import GemmaForCausalLM as HFGemmaForCausalLM


def _tiny_config(sliding_window: int | None = None) -> "GemmaConfig":
    return GemmaConfig(
        vocab_size=31,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        sliding_window=sliding_window,
    )


def test_gemma_rms_norm_matches_manual_formula():
    layer = GemmaRMSNorm(hidden_size=4, eps=1e-6)
    x = t.tensor([[[1.0, 2.0, 3.0, 4.0]]])

    out = layer(x)
    expected = x * t.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)

    assert t.allclose(out, expected, atol=1e-6)


def test_rope_preserves_vector_norms():
    t.manual_seed(0)
    x = t.randn(2, 4, 5, 4)
    position_ids = t.arange(5).unsqueeze(0).expand(2, -1)
    cos, sin = build_rope_cache(5, 4, base=10000.0, device=x.device, dtype=x.dtype)

    rotated = apply_rope(x, cos, sin, position_ids)

    assert t.allclose(rotated.norm(dim=-1), x.norm(dim=-1), atol=1e-6)


def test_repeat_kv_repeats_key_value_heads():
    x = t.arange(2 * 2 * 3 * 4).reshape(2, 2, 3, 4)
    repeated = repeat_kv(x, repeats=2)

    assert repeated.shape == (2, 4, 3, 4)
    assert t.equal(repeated[:, 0], x[:, 0])
    assert t.equal(repeated[:, 1], x[:, 0])
    assert t.equal(repeated[:, 2], x[:, 1])
    assert t.equal(repeated[:, 3], x[:, 1])


def test_sliding_window_causal_mask_blocks_future_and_old_tokens():
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
    assert t.equal(mask.cpu(), expected)


def test_gemma_forward_shape_and_cache_parity():
    t.manual_seed(0)
    model = GemmaForCausalLM(_tiny_config(sliding_window=None))
    input_ids = t.tensor([[1, 2, 3, 4, 5]])

    output = model(input_ids, use_cache=True)
    report = cache_parity_report(model, input_ids, atol=1e-5)

    assert output.logits.shape == (1, 5, model.config.vocab_size)
    assert output.past_key_values is not None
    assert len(output.past_key_values) == model.config.num_hidden_layers
    assert report["passed"], report


def test_load_matching_state_dict_reports_shape_mismatches():
    source = GemmaForCausalLM(_tiny_config())
    target = GemmaForCausalLM(_tiny_config())
    state = source.state_dict()
    broken_state = dict(state)
    broken_state["model.layers.0.self_attn.q_proj.weight"] = t.zeros(3, 3)
    broken_state["unexpected.weight"] = t.zeros(1)

    report = load_matching_state_dict(target, broken_state)

    assert report.loaded_keys == len(state) - 1
    assert "model.layers.0.self_attn.q_proj.weight" in report.skipped_shape_mismatches
    assert "unexpected.weight" in report.unexpected_keys


@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="transformers is not installed")
def test_tiny_gemma_matches_huggingface_reference_logits():
    local_config = _tiny_config(sliding_window=None)
    hf_config = HFGemmaConfig(
        vocab_size=local_config.vocab_size,
        hidden_size=local_config.hidden_size,
        intermediate_size=local_config.intermediate_size,
        num_hidden_layers=local_config.num_hidden_layers,
        num_attention_heads=local_config.num_attention_heads,
        num_key_value_heads=local_config.num_key_value_heads,
        head_dim=local_config.head_dim,
        max_position_embeddings=local_config.max_position_embeddings,
        rope_theta=local_config.rope_theta,
        rms_norm_eps=local_config.rms_norm_eps,
        attention_bias=local_config.attention_bias,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=local_config.tie_word_embeddings,
    )
    t.manual_seed(5101)
    reference = HFGemmaForCausalLM(hf_config).eval()
    local = GemmaForCausalLM(local_config).eval()
    local.load_state_dict(reference.state_dict())
    input_ids = t.tensor([[1, 5, 8, 13, 2]])
    attention_mask = t.ones_like(input_ids)

    with t.inference_mode():
        reference_logits = reference(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits
        local_logits = local(
            input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits

    assert (local_logits - reference_logits).abs().max().item() < 5e-4
