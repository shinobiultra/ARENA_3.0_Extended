"""Small Gemma-style decoder implementation for ARENA extension notebooks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch as t
import torch.nn as nn
import torch.nn.functional as F


PastKeyValue = tuple[t.Tensor, t.Tensor]


@dataclass(frozen=True)
class GemmaConfig:
    """Minimal config for a Gemma-style decoder-only transformer."""

    vocab_size: int = 32000
    hidden_size: int = 2048
    intermediate_size: int = 16384
    num_hidden_layers: int = 18
    num_attention_heads: int = 8
    num_key_value_heads: int = 1
    head_dim: int | None = None
    max_position_embeddings: int = 8192
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    attention_bias: bool = False
    sliding_window: int | None = None
    tie_word_embeddings: bool = True

    def __post_init__(self):
        head_dim = self.head_dim or self.hidden_size // self.num_attention_heads
        if self.hidden_size != self.num_attention_heads * head_dim:
            raise ValueError("hidden_size must equal num_attention_heads * head_dim.")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads.")
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE.")
        object.__setattr__(self, "head_dim", head_dim)


@dataclass(frozen=True)
class GemmaCausalLMOutput:
    logits: t.Tensor
    past_key_values: tuple[PastKeyValue, ...] | None = None


@dataclass(frozen=True)
class GemmaWeightLoadReport:
    loaded_keys: int
    missing_keys: list[str]
    unexpected_keys: list[str]
    skipped_shape_mismatches: list[str]


class GemmaRMSNorm(nn.Module):
    """Gemma-style RMSNorm with a learned offset from unit scale."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(t.zeros(hidden_size))
        self.eps = eps

    def forward(self, x: t.Tensor) -> t.Tensor:
        dtype = x.dtype
        x_float = x.float()
        normalized = x_float * t.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (normalized * (1.0 + self.weight.float())).to(dtype)


def rotate_half_interleaved(x: t.Tensor) -> t.Tensor:
    """Rotate even/odd RoPE pairs: [x0, x1] -> [-x1, x0]."""

    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    rotated = t.stack((-x_odd, x_even), dim=-1)
    return rotated.flatten(start_dim=-2)


def build_rope_cache(
    seq_len: int,
    head_dim: int,
    *,
    base: float,
    device: t.device,
    dtype: t.dtype,
) -> tuple[t.Tensor, t.Tensor]:
    """Return interleaved cos/sin RoPE cache with shape ``(seq_len, head_dim)``."""

    inv_freq = 1.0 / (base ** (t.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = t.arange(seq_len, device=device).float()
    freqs = t.outer(positions, inv_freq)
    emb = t.repeat_interleave(freqs, repeats=2, dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_rope(x: t.Tensor, cos: t.Tensor, sin: t.Tensor, position_ids: t.Tensor) -> t.Tensor:
    """Apply RoPE to ``x`` with shape ``(batch, heads, seq, head_dim)``."""

    cos = cos[position_ids].unsqueeze(1)
    sin = sin[position_ids].unsqueeze(1)
    return (x * cos) + (rotate_half_interleaved(x) * sin)


def repeat_kv(hidden_states: t.Tensor, repeats: int) -> t.Tensor:
    """Repeat key/value heads for grouped-query attention."""

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
    """Boolean mask with ``True`` where attention is allowed."""

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
        """Greedy generation using the KV cache."""

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


def load_matching_state_dict(
    module: nn.Module,
    reference_state_dict: dict[str, t.Tensor],
) -> GemmaWeightLoadReport:
    """Load matching keys from a reference state dict and report skipped keys."""

    current_state = module.state_dict()
    loadable = {}
    skipped = []
    unexpected = []
    for key, value in reference_state_dict.items():
        if key not in current_state:
            unexpected.append(key)
            continue
        if tuple(current_state[key].shape) != tuple(value.shape):
            skipped.append(key)
            continue
        loadable[key] = value

    missing = [key for key in current_state if key not in loadable]
    module.load_state_dict(loadable, strict=False)
    return GemmaWeightLoadReport(
        loaded_keys=len(loadable),
        missing_keys=missing,
        unexpected_keys=unexpected,
        skipped_shape_mismatches=skipped,
    )
