# %%
"""Reference solutions for [5.1] Gemma from Scratch."""

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch as t
import torch.nn as nn
import torch.nn.functional as F

chapter = "chapter5_modern_architectures"
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from arena_ext import compare_logits, estimate_inference_memory
from arena_ext.gemma import GemmaConfig

MAIN = __name__ == "__main__"

HF_GEMMA_REFERENCE_MAX_ABS_ERROR = 5e-4
HF_GEMMA_REFERENCE_MSE_MAX = 1e-7
PastKeyValue = tuple[t.Tensor, t.Tensor]


@dataclass(frozen=True)
class GemmaCausalLMOutput:
    logits: t.Tensor
    past_key_values: tuple[PastKeyValue, ...] | None = None


# %%
class GemmaRMSNorm(nn.Module):
    """Gemma-style RMSNorm with a learned offset from unit scale."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(t.zeros(hidden_size))
        self.eps = eps

    def forward(self, x: t.Tensor) -> t.Tensor:
        dtype = x.dtype
        x_float = x.float()
        rms = x_float.pow(2).mean(dim=-1, keepdim=True)
        normalized = x_float * t.rsqrt(rms + self.eps)
        return (normalized * (1.0 + self.weight.float())).to(dtype)


def rotate_half_interleaved(x: t.Tensor) -> t.Tensor:
    """Rotate even/odd RoPE feature pairs: [x0, x1] -> [-x1, x0]."""

    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    return t.stack((-x_odd, x_even), dim=-1).flatten(start_dim=-2)


def build_rope_cache(
    seq_len: int,
    head_dim: int,
    *,
    base: float,
    device: t.device,
    dtype: t.dtype,
) -> tuple[t.Tensor, t.Tensor]:
    """Return interleaved cos/sin RoPE caches with shape (seq_len, head_dim)."""

    inv_freq = 1.0 / (base ** (t.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = t.arange(seq_len, device=device).float()
    freqs = t.outer(positions, inv_freq)
    angles = t.repeat_interleave(freqs, repeats=2, dim=-1)
    return angles.cos().to(dtype), angles.sin().to(dtype)


def apply_rope(x: t.Tensor, cos: t.Tensor, sin: t.Tensor, position_ids: t.Tensor) -> t.Tensor:
    """Apply interleaved RoPE to (batch, heads, seq, head_dim) activations."""

    cos = cos[position_ids].unsqueeze(1)
    sin = sin[position_ids].unsqueeze(1)
    return (x * cos) + (rotate_half_interleaved(x) * sin)


def repeat_kv(hidden_states: t.Tensor, repeats: int) -> t.Tensor:
    """Repeat grouped key/value heads contiguously for query-head attention."""

    if repeats == 1:
        return hidden_states
    batch, kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, kv_heads, repeats, seq_len, head_dim
    )
    return hidden_states.reshape(batch, kv_heads * repeats, seq_len, head_dim)


def build_causal_attention_mask(
    *,
    query_length: int,
    key_length: int,
    past_length: int,
    sliding_window: int | None,
    device: t.device,
) -> t.Tensor:
    """Boolean mask with True where a cached query is allowed to attend."""

    query_positions = t.arange(past_length, past_length + query_length, device=device)[:, None]
    key_positions = t.arange(key_length, device=device)[None, :]
    allowed = key_positions <= query_positions
    if sliding_window is not None:
        allowed = allowed & (key_positions >= query_positions - sliding_window + 1)
    return allowed[None, None, :, :]


class GemmaMLP(nn.Module):
    """SwiGLU MLP used by Gemma-style decoders."""

    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class GemmaAttention(nn.Module):
    """Causal grouped-query attention with optional sliding-window locality."""

    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.q_proj = nn.Linear(
            config.hidden_size,
            self.num_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )

    def _shape(self, x: t.Tensor, heads: int) -> t.Tensor:
        batch, seq_len, _ = x.shape
        return x.view(batch, seq_len, heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: t.Tensor,
        *,
        position_ids: t.Tensor,
        cos: t.Tensor,
        sin: t.Tensor,
        attention_mask: t.Tensor | None = None,
        past_key_value: PastKeyValue | None = None,
        use_cache: bool = False,
    ) -> tuple[t.Tensor, PastKeyValue | None]:
        batch, query_length, _ = hidden_states.shape
        query_states = self._shape(self.q_proj(hidden_states), self.num_heads)
        key_states = self._shape(self.k_proj(hidden_states), self.num_key_value_heads)
        value_states = self._shape(self.v_proj(hidden_states), self.num_key_value_heads)

        query_states = apply_rope(query_states, cos, sin, position_ids)
        key_states = apply_rope(key_states, cos, sin, position_ids)

        past_length = 0
        if past_key_value is not None:
            past_key, past_value = past_key_value
            past_length = past_key.shape[-2]
            key_states = t.cat([past_key, key_states], dim=-2)
            value_states = t.cat([past_value, value_states], dim=-2)

        present_key_value = (key_states, value_states) if use_cache else None
        key_length = key_states.shape[-2]

        key_states_repeated = repeat_kv(key_states, self.num_key_value_groups)
        value_states_repeated = repeat_kv(value_states, self.num_key_value_groups)

        attn_scores = t.matmul(
            query_states, key_states_repeated.transpose(-1, -2)
        ) / math.sqrt(self.head_dim)
        causal_mask = build_causal_attention_mask(
            query_length=query_length,
            key_length=key_length,
            past_length=past_length,
            sliding_window=self.config.sliding_window,
            device=hidden_states.device,
        )
        attn_scores = attn_scores.masked_fill(~causal_mask, t.finfo(attn_scores.dtype).min)

        if attention_mask is not None:
            if attention_mask.shape != (batch, key_length):
                raise ValueError(
                    "attention_mask must have shape (batch, key_length); "
                    f"got {tuple(attention_mask.shape)}, expected {(batch, key_length)}."
                )
            attn_scores = attn_scores.masked_fill(
                ~attention_mask[:, None, None, :].bool(),
                t.finfo(attn_scores.dtype).min,
            )

        attn_probs = F.softmax(attn_scores.float(), dim=-1).to(query_states.dtype)
        attn_output = t.matmul(attn_probs, value_states_repeated)
        attn_output = attn_output.transpose(1, 2).reshape(
            batch,
            query_length,
            self.num_heads * self.head_dim,
        )
        return self.o_proj(attn_output), present_key_value


class GemmaDecoderLayer(nn.Module):
    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__()
        self.self_attn = GemmaAttention(config, layer_idx=layer_idx)
        self.mlp = GemmaMLP(config)
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: t.Tensor,
        *,
        position_ids: t.Tensor,
        cos: t.Tensor,
        sin: t.Tensor,
        attention_mask: t.Tensor | None = None,
        past_key_value: PastKeyValue | None = None,
        use_cache: bool = False,
    ) -> tuple[t.Tensor, PastKeyValue | None]:
        residual = hidden_states
        attn_output, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states),
            position_ids=position_ids,
            cos=cos,
            sin=sin,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = residual + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value


class GemmaModel(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [GemmaDecoderLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: t.Tensor,
        *,
        attention_mask: t.Tensor | None = None,
        past_key_values: tuple[PastKeyValue, ...] | None = None,
        use_cache: bool = False,
    ) -> tuple[t.Tensor, tuple[PastKeyValue, ...] | None]:
        batch, seq_len = input_ids.shape
        past_length = 0 if past_key_values is None else past_key_values[0][0].shape[-2]
        position_ids = (
            t.arange(past_length, past_length + seq_len, device=input_ids.device)
            .unsqueeze(0)
            .expand(batch, -1)
        )
        max_position = past_length + seq_len
        cos, sin = build_rope_cache(
            max_position,
            self.config.head_dim,
            base=self.config.rope_theta,
            device=input_ids.device,
            dtype=self.embed_tokens.weight.dtype,
        )

        if attention_mask is not None and past_length > 0 and attention_mask.shape[-1] == seq_len:
            prefix = t.ones(
                batch,
                past_length,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            attention_mask = t.cat([prefix, attention_mask], dim=-1)

        hidden_states = self.embed_tokens(input_ids) * math.sqrt(self.config.hidden_size)
        next_cache = [] if use_cache else None
        for idx, layer in enumerate(self.layers):
            past_key_value = None if past_key_values is None else past_key_values[idx]
            hidden_states, present = layer(
                hidden_states,
                position_ids=position_ids,
                cos=cos,
                sin=sin,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                use_cache=use_cache,
            )
            if use_cache:
                assert next_cache is not None and present is not None
                next_cache.append(present)
        return self.norm(hidden_states), None if next_cache is None else tuple(next_cache)


class GemmaForCausalLM(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.model = GemmaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: t.Tensor,
        *,
        attention_mask: t.Tensor | None = None,
        past_key_values: tuple[PastKeyValue, ...] | None = None,
        use_cache: bool = False,
    ) -> GemmaCausalLMOutput:
        hidden_states, next_cache = self.model(
            input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        logits = self.lm_head(hidden_states)
        return GemmaCausalLMOutput(logits=logits, past_key_values=next_cache)

    @t.no_grad()
    def greedy_generate(self, input_ids: t.Tensor, max_new_tokens: int) -> t.Tensor:
        generated = input_ids
        cache = None
        next_input = input_ids
        for _ in range(max_new_tokens):
            output = self(next_input, past_key_values=cache, use_cache=True)
            cache = output.past_key_values
            next_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
            generated = t.cat([generated, next_token], dim=-1)
            next_input = next_token
        return generated


def cache_parity_report(
    model: GemmaForCausalLM,
    input_ids: t.Tensor,
    *,
    atol: float = 1e-5,
) -> dict[str, Any]:
    """Compare full-context logits with one-token-at-a-time cached logits."""

    model.eval()
    with t.no_grad():
        full_logits = model(input_ids, use_cache=False).logits
        cache = None
        cached_logits = []
        for pos in range(input_ids.shape[1]):
            output = model(
                input_ids[:, pos : pos + 1],
                past_key_values=cache,
                use_cache=True,
            )
            cache = output.past_key_values
            cached_logits.append(output.logits)
        cached_logits_tensor = t.cat(cached_logits, dim=1)
        max_abs_diff = (full_logits - cached_logits_tensor).abs().max().item()
    return {
        "max_abs_diff": max_abs_diff,
        "passed": max_abs_diff <= atol,
        "atol": atol,
    }


# %%
def make_tiny_gemma_config(sliding_window: int | None = None) -> GemmaConfig:
    """Tiny config used for CPU smoke tests and notebook exercises."""

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


def make_tiny_gemma(seed: int = 0, sliding_window: int | None = None) -> GemmaForCausalLM:
    t.manual_seed(seed)
    return GemmaForCausalLM(make_tiny_gemma_config(sliding_window=sliding_window))


def rms_norm_smoke_test() -> bool:
    layer = GemmaRMSNorm(hidden_size=4, eps=1e-6)
    x = t.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    expected = x * t.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    return bool(t.allclose(layer(x), expected, atol=1e-6))


def rope_norm_smoke_test() -> bool:
    t.manual_seed(0)
    x = t.randn(2, 4, 5, 4)
    position_ids = t.arange(5).unsqueeze(0).expand(2, -1)
    cos, sin = build_rope_cache(5, 4, base=10000.0, device=x.device, dtype=x.dtype)
    rotated = apply_rope(x, cos, sin, position_ids)
    return bool(t.allclose(rotated.norm(dim=-1), x.norm(dim=-1), atol=1e-6))


def cache_parity_smoke_test() -> dict:
    model = make_tiny_gemma(seed=0)
    input_ids = t.tensor([[1, 2, 3, 4, 5]])
    return cache_parity_report(model, input_ids, atol=1e-5)


def local_memory_budget() -> dict:
    budget = estimate_inference_memory(
        num_parameters=1_000_000_000,
        dtype="bfloat16",
        batch_size=1,
        context_length=2048,
        hidden_size=2048,
        num_layers=18,
        num_key_value_heads=8,
        head_dim=256,
        overhead_gb=1.5,
    )
    return budget.as_dict()


def compare_tiny_model_to_clone() -> dict:
    """Synthetic HF-parity pattern: compare a model against an exact clone."""

    model = make_tiny_gemma(seed=1)
    clone = make_tiny_gemma(seed=999)
    clone.load_state_dict(model.state_dict())
    input_ids = t.tensor([[1, 5, 8, 13]])
    report = compare_logits(model(input_ids).logits, clone(input_ids).logits, k=5)
    return report.__dict__


def run_smoke_test(cpu: bool = True) -> dict:
    _ = cpu
    cache_report = cache_parity_smoke_test()
    return {
        "rms_norm_passed": rms_norm_smoke_test(),
        "rope_norm_passed": rope_norm_smoke_test(),
        "cache_parity": cache_report,
        "clone_parity": compare_tiny_model_to_clone(),
        "memory_budget": local_memory_budget(),
    }


def run_gpu_test(max_vram_gb: float = 24.0) -> dict:
    budget = estimate_inference_memory(
        num_parameters=1_000_000_000,
        dtype="bfloat16",
        batch_size=1,
        context_length=2048,
        hidden_size=2048,
        num_layers=18,
        num_key_value_heads=8,
        head_dim=256,
        overhead_gb=1.5,
    )
    if not t.cuda.is_available():
        raise RuntimeError("5.1 GPU preflight requires CUDA; no CPU fallback is accepted.")

    device = t.device("cuda")
    t.cuda.reset_peak_memory_stats()
    model = make_tiny_gemma(seed=0).to(device)
    input_ids = t.tensor([[1, 2, 3, 4, 5]], device=device)
    cache_report = cache_parity_report(model, input_ids, atol=1e-5)
    x = t.randn(2, 4, 5, 4, device=device)
    position_ids = t.arange(5, device=device).unsqueeze(0).expand(2, -1)
    cos, sin = build_rope_cache(5, 4, base=10000.0, device=device, dtype=x.dtype)
    rotated = apply_rope(x, cos, sin, position_ids)
    rope_norm_error = float((rotated.norm(dim=-1) - x.norm(dim=-1)).abs().max().item())

    reference = run_hf_tiny_gemma_reference_parity(device=device)
    t.cuda.synchronize()
    peak_vram_gb = t.cuda.max_memory_allocated() / 1024**3
    preflight_passed = (
        cache_report["passed"]
        and rope_norm_error <= 1e-5
        and reference["reference_parity_passed"]
        and budget.fits(max_vram_gb)
        and peak_vram_gb <= max_vram_gb
    )
    return {
        "preflight_passed": preflight_passed,
        "cuda_available": True,
        "device": t.cuda.get_device_name(0),
        "cache_max_abs_diff": cache_report["max_abs_diff"],
        "cache_passed": cache_report["passed"],
        "rope_norm_error": rope_norm_error,
        **reference,
        "estimated_total_gb": budget.total_gb,
        "fits_budget": budget.fits(max_vram_gb),
        "peak_vram_gb": peak_vram_gb,
        "within_vram_budget": peak_vram_gb <= max_vram_gb,
        "full_path": "Compare the local Gemma-from-scratch implementation against Hugging Face's Gemma reference architecture on CUDA with matched deterministic tiny weights.",
    }


def run_hf_tiny_gemma_reference_parity(device: t.device) -> dict:
    """Compare local tiny Gemma logits against the real HF Gemma implementation."""

    from transformers import GemmaConfig as HFGemmaConfig
    from transformers import GemmaForCausalLM as HFGemmaForCausalLM

    local_config = make_tiny_gemma_config(sliding_window=None)
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
    t.cuda.manual_seed_all(5101)
    reference = HFGemmaForCausalLM(hf_config).to(device).eval()
    local = GemmaForCausalLM(local_config).to(device).eval()
    local.load_state_dict(reference.state_dict())
    input_ids = t.tensor([[1, 5, 8, 13, 2]], device=device)
    attention_mask = t.ones_like(input_ids)
    with t.inference_mode():
        local_logits = local(
            input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits
        reference_logits = reference(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits
    parity = compare_logits(local_logits, reference_logits, k=5)
    cache_report = cache_parity_report(local, input_ids, atol=1e-5)
    generation_prompt = t.tensor([[1, 5, 8, 13]], device=device)
    local_generated = local.greedy_generate(generation_prompt, max_new_tokens=4)
    reference_generated = generation_prompt.clone()
    with t.inference_mode():
        for _ in range(4):
            reference_output = reference(input_ids=reference_generated, use_cache=False)
            next_token = reference_output.logits[:, -1].argmax(dim=-1, keepdim=True)
            reference_generated = t.cat([reference_generated, next_token], dim=-1)
    generation_matches_reference = t.equal(local_generated, reference_generated)
    return {
        "reference_model_family": "transformers.GemmaForCausalLM",
        "reference_transformers_tiny_config": True,
        "reference_weight_key_count": len(reference.state_dict()),
        "reference_loaded_key_count": len(local.state_dict()),
        "reference_logits_max_abs_diff": parity.max_abs_diff,
        "reference_logits_mse": parity.mse,
        "reference_logits_kl_divergence": parity.kl_divergence,
        "reference_logits_topk_agreement": parity.topk_agreement,
        "reference_cache_max_abs_diff": cache_report["max_abs_diff"],
        "reference_cache_passed": cache_report["passed"],
        "generation_prompt": generation_prompt.squeeze(0).tolist(),
        "local_greedy_tokens": local_generated.squeeze(0).tolist(),
        "reference_greedy_tokens": reference_generated.squeeze(0).tolist(),
        "generation_matches_reference": generation_matches_reference,
        "reference_parity_passed": (
            parity.max_abs_diff <= HF_GEMMA_REFERENCE_MAX_ABS_ERROR
            and parity.mse <= HF_GEMMA_REFERENCE_MSE_MAX
            and parity.topk_agreement == 1.0
            and cache_report["passed"]
            and generation_matches_reference
        ),
    }


def run_full_experiment(max_vram_gb: float = 24.0) -> dict:
    """Run the validated experiment path used by the verification report."""

    return run_gpu_test(max_vram_gb=max_vram_gb)


if MAIN:
    print(run_smoke_test())
