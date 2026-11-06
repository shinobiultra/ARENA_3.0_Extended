# %%


MAIN = __name__ == "__main__"

# %%

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch as t
import torch.nn as nn
import torch.nn.functional as F

chapter = "chapter5_modern_architectures"
section = "part1_gemma_from_scratch"
root_dir = Path(__file__).resolve().parents[3]
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
assets_dir = section_dir / "media"
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part1_gemma_from_scratch.tests as tests

from chapter5_modern_architectures.exercises.part1_gemma_from_scratch.gemma_reference import (
    GemmaCausalLMOutput,
    GemmaConfig,
    GemmaForCausalLM as ReferenceGemmaForCausalLM,
)

DEVICE = t.device("cpu")
PastKeyValue = tuple[t.Tensor, t.Tensor]

# %%

SIGNATURE_INPUT_IDS = t.tensor(
    [
        [1, 5, 8, 13, 2, 21, 3],
        [3, 2, 1, 8, 5, 13, 21],
        [21, 13, 8, 5, 3, 2, 1],
        [2, 3, 5, 8, 13, 21, 1],
    ],
    dtype=t.long,
    device=DEVICE,
)
if MAIN:
    cold_config = GemmaConfig(
        vocab_size=31,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
if MAIN:
    t.manual_seed(5101)
if MAIN:
    cold_reference = ReferenceGemmaForCausalLM(cold_config).eval()
if MAIN:
    with t.inference_mode():
        cold_logits = cold_reference(SIGNATURE_INPUT_IDS).logits
if MAIN:
    print("logits shape:", tuple(cold_logits.shape))
if MAIN:
    print("last-position argmax token ids:", cold_logits[:, -1].argmax(dim=-1).tolist())

# %%

class GemmaRMSNorm(nn.Module):
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


if MAIN:
    tests.test_gemma_rms_norm(GemmaRMSNorm)

# %%

def rotate_half_interleaved(x: t.Tensor) -> t.Tensor:
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
    inv_freq = 1.0 / (base ** (t.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = t.arange(seq_len, device=device).float()
    freqs = t.outer(positions, inv_freq)
    angles = t.repeat_interleave(freqs, repeats=2, dim=-1)
    return angles.cos().to(dtype), angles.sin().to(dtype)


def apply_rope(
    x: t.Tensor,
    cos: t.Tensor,
    sin: t.Tensor,
    position_ids: t.Tensor,
) -> t.Tensor:
    cos_at_position = cos[position_ids].unsqueeze(1)
    sin_at_position = sin[position_ids].unsqueeze(1)
    return (x * cos_at_position) + (rotate_half_interleaved(x) * sin_at_position)


if MAIN:
    tests.test_rotate_half_interleaved(rotate_half_interleaved)
if MAIN:
    tests.test_build_rope_cache(build_rope_cache)
if MAIN:
    tests.test_apply_rope(apply_rope, build_rope_cache)

# %%

def repeat_kv(hidden_states: t.Tensor, repeats: int) -> t.Tensor:
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
    query_positions = t.arange(past_length, past_length + query_length, device=device)[:, None]
    key_positions = t.arange(key_length, device=device)[None, :]
    allowed = key_positions <= query_positions
    if sliding_window is not None:
        allowed = allowed & (key_positions >= query_positions - sliding_window + 1)
    return allowed[None, None, :, :]


if MAIN:
    tests.test_repeat_kv(repeat_kv)
if MAIN:
    tests.test_sliding_window_mask(build_causal_attention_mask)

# %%

class GemmaMLP(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: t.Tensor) -> t.Tensor:
        gated = F.silu(self.gate_proj(x)) * self.up_proj(x)
        return self.down_proj(gated)


if MAIN:
    tests.test_gemma_mlp_matches_swiglu_formula(GemmaMLP)

# %%

class GemmaAttention(nn.Module):
    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)

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
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attn_scores = query_states @ key_states.transpose(-1, -2)
        attn_scores = attn_scores / math.sqrt(self.head_dim)
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
                raise ValueError(f"attention_mask has shape {tuple(attention_mask.shape)}, expected {(batch, key_length)}")
            attn_scores = attn_scores.masked_fill(
                ~attention_mask[:, None, None, :].bool(),
                t.finfo(attn_scores.dtype).min,
            )
        attn_probs = F.softmax(attn_scores.float(), dim=-1).to(query_states.dtype)
        attn_output = attn_probs @ value_states
        attn_output = attn_output.transpose(1, 2).reshape(
            batch, query_length, self.num_heads * self.head_dim
        )
        return self.o_proj(attn_output), present_key_value


if MAIN:
    tests.test_gemma_attention_shapes_and_cache(GemmaAttention)

# %%

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


if MAIN:
    tests.test_gemma_decoder_layer_shapes_and_cache(GemmaDecoderLayer)

# %%

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
        position_ids = t.arange(
            past_length, past_length + seq_len, device=input_ids.device
        ).unsqueeze(0).expand(batch, -1)
        cos, sin = build_rope_cache(
            past_length + seq_len,
            self.config.head_dim,
            base=self.config.rope_theta,
            device=input_ids.device,
            dtype=self.embed_tokens.weight.dtype,
        )
        if attention_mask is not None and past_length > 0 and attention_mask.shape[-1] == seq_len:
            prefix = t.ones(batch, past_length, dtype=attention_mask.dtype, device=attention_mask.device)
            attention_mask = t.cat([prefix, attention_mask], dim=-1)

        hidden_states = self.embed_tokens(input_ids) * math.sqrt(self.config.hidden_size)
        next_cache = [] if use_cache else None
        for layer_idx, layer in enumerate(self.layers):
            past_key_value = None if past_key_values is None else past_key_values[layer_idx]
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
        return GemmaCausalLMOutput(
            logits=self.lm_head(hidden_states),
            past_key_values=next_cache,
        )


def make_tiny_gemma_config(sliding_window: int | None = None) -> GemmaConfig:
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
    return GemmaForCausalLM(make_tiny_gemma_config(sliding_window=sliding_window)).to(DEVICE)


if MAIN:
    tests.test_tiny_gemma_forward_shape(make_tiny_gemma)
if MAIN:
    tests.test_tiny_gemma_cache_parity(make_tiny_gemma)
if MAIN:
    tests.test_tiny_gemma_matches_reference_decoder(GemmaForCausalLM)

# %%

@t.inference_mode()
def capture_gemma_trace(model: nn.Module, input_ids: t.Tensor) -> dict[str, t.Tensor]:
    trace: dict[str, t.Tensor] = {}
    handles = []

    def save(name: str, scale: float = 1.0):
        def hook(_module, _inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            trace[name] = (value * scale).detach().float().cpu()
        return hook

    handles.append(
        model.model.embed_tokens.register_forward_hook(
            save("embedding", math.sqrt(model.config.hidden_size))
        )
    )
    for layer_idx, layer in enumerate(model.model.layers):
        handles.append(layer.register_forward_hook(save(f"layer_{layer_idx}")))
    handles.append(model.model.norm.register_forward_hook(save("final_norm")))
    try:
        output = model(input_ids)
    finally:
        for handle in handles:
            handle.remove()
    trace["logits"] = output.logits.detach().float().cpu()
    return trace


def compare_gemma_traces(
    actual: dict[str, t.Tensor],
    expected: dict[str, t.Tensor],
) -> dict[str, dict[str, float]]:
    assert actual.keys() == expected.keys()
    profile = {}
    for name in expected:
        delta = actual[name] - expected[name]
        rmse = delta.square().mean().sqrt()
        reference_rms = expected[name].square().mean().sqrt()
        profile[name] = {
            "max_abs": delta.abs().max().item(),
            "relative_rmse": (rmse / reference_rms.clamp_min(1e-12)).item(),
            "cosine": F.cosine_similarity(
                actual[name].reshape(1, -1), expected[name].reshape(1, -1)
            ).item(),
        }
    return profile


if MAIN:
    tests.test_capture_gemma_trace(capture_gemma_trace, make_tiny_gemma)

# %%

@t.inference_mode()
def run_architecture_controls(
    input_ids: t.Tensor = SIGNATURE_INPUT_IDS,
    *,
    seed: int = 5101,
) -> dict[str, object]:
    config = make_tiny_gemma_config()
    t.manual_seed(seed)
    reference = ReferenceGemmaForCausalLM(config).eval()
    learner = GemmaForCausalLM(config).eval()
    learner.load_state_dict(reference.state_dict())
    reference_trace = capture_gemma_trace(reference, input_ids)
    variant_traces = {"exact": capture_gemma_trace(learner, input_ids)}

    remove_scale = learner.model.embed_tokens.register_forward_hook(
        lambda _module, _inputs, output: output / math.sqrt(config.hidden_size)
    )
    try:
        variant_traces["no embedding scale"] = capture_gemma_trace(learner, input_ids)
    finally:
        remove_scale.remove()

    original_apply_rope = globals()["apply_rope"]
    globals()["apply_rope"] = lambda x, _cos, _sin, _position_ids: x
    try:
        variant_traces["no RoPE"] = capture_gemma_trace(learner, input_ids)
    finally:
        globals()["apply_rope"] = original_apply_rope

    original_repeat_kv = globals()["repeat_kv"]
    globals()["repeat_kv"] = lambda hidden, repeats: hidden.repeat(1, repeats, 1, 1)
    try:
        variant_traces["wrong GQA order"] = capture_gemma_trace(learner, input_ids)
    finally:
        globals()["repeat_kv"] = original_repeat_kv

    return {
        "input_ids": input_ids.detach().cpu(),
        "reference_trace": reference_trace,
        "variant_traces": variant_traces,
        "profiles": {
            name: compare_gemma_traces(trace, reference_trace)
            for name, trace in variant_traces.items()
        },
    }


if MAIN:
    tests.test_architecture_controls(run_architecture_controls)

# %%

if MAIN:
    signature = run_architecture_controls()
if MAIN:
    profiles = signature["profiles"]
if MAIN:
    stages = list(profiles["exact"])
if MAIN:
    colors = {
        "exact": "#111827",
        "no embedding scale": "#c2410c",
        "no RoPE": "#0f766e",
        "wrong GQA order": "#b91c1c",
    }

if MAIN:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), constrained_layout=True)
if MAIN:
    for name, profile in profiles.items():
        values = [max(profile[stage]["max_abs"], 1e-8) for stage in stages]
        axes[0].plot(stages, values, marker="o", linewidth=2, label=name, color=colors[name])
if MAIN:
    axes[0].set_yscale("log")
if MAIN:
    axes[0].set_ylabel("maximum absolute error vs reference")
if MAIN:
    axes[0].set_title("Where does each implementation first diverge?")
if MAIN:
    axes[0].tick_params(axis="x", rotation=25)
if MAIN:
    axes[0].grid(axis="y", alpha=0.25)
if MAIN:
    axes[0].legend(frameon=False, fontsize=9)

if MAIN:
    names = list(profiles)
if MAIN:
    final_errors = [profiles[name]["logits"]["max_abs"] for name in names]
if MAIN:
    bars = axes[1].bar(names, final_errors, color=[colors[name] for name in names])
if MAIN:
    axes[1].set_ylabel("maximum absolute logit error")
if MAIN:
    axes[1].set_title("Exact parity passes; architecture controls fail")
if MAIN:
    axes[1].tick_params(axis="x", rotation=25)
if MAIN:
    axes[1].grid(axis="y", alpha=0.25)
if MAIN:
    for bar, value in zip(bars, final_errors):
        label = "0" if value == 0 else f"{value:.3f}"
        axes[1].text(bar.get_x() + bar.get_width() / 2, value + 0.05, label, ha="center", fontsize=9)

if MAIN:
    figure_path = assets_dir / "gemma_parity_signature.png"
if MAIN:
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
if MAIN:
    plt.show()

if MAIN:
    print("stage              exact max abs error")
if MAIN:
    for stage in stages:
        print(f"{stage:18s} {profiles['exact'][stage]['max_abs']:.3e}")
if MAIN:
    print("\ncontrol             final logit max abs error")
if MAIN:
    for name in names[1:]:
        print(f"{name:20s} {profiles[name]['logits']['max_abs']:.6f}")

# %%

# Try It Yourself: change only these token IDs, keeping values in [0, 30].
if MAIN:
    play_input_ids = t.tensor([[1, 4, 9, 16, 25, 7, 2]], dtype=t.long, device=DEVICE)
if MAIN:
    play = run_architecture_controls(play_input_ids)

if MAIN:
    fig, ax = plt.subplots(figsize=(8, 3.6), constrained_layout=True)
if MAIN:
    play_names = list(play["profiles"])
if MAIN:
    play_errors = [play["profiles"][name]["logits"]["max_abs"] for name in play_names]
if MAIN:
    ax.bar(play_names, play_errors, color=[colors[name] for name in play_names])
if MAIN:
    ax.set_ylabel("maximum absolute logit error")
if MAIN:
    ax.set_title(f"Your input: {play_input_ids.squeeze(0).tolist()}")
if MAIN:
    ax.tick_params(axis="x", rotation=20)
if MAIN:
    ax.grid(axis="y", alpha=0.25)
if MAIN:
    plt.show()

# %%

def first_divergent_stage(
    profile: dict[str, dict[str, float]],
    *,
    tolerance: float = 1e-6,
) -> str | None:
    return next(
        (stage for stage, metrics in profile.items() if metrics["max_abs"] > tolerance),
        None,
    )


if MAIN:
    expected_first_divergence = {
        "exact": None,
        "no embedding scale": "embedding",
        "no RoPE": "layer_0",
        "wrong GQA order": "layer_0",
    }
if MAIN:
    observed_first_divergence = {
        name: first_divergent_stage(profile)
        for name, profile in signature["profiles"].items()
    }
if MAIN:
    assert observed_first_divergence == expected_first_divergence
if MAIN:
    observed_first_divergence

# %%

if MAIN:
    PLAY_PAST_LENGTH = 3
if MAIN:
    PLAY_QUERY_LENGTH = 2
if MAIN:
    PLAY_WINDOW = 3

# Exact mask probe only: no model object, weights, optimizer or CUDA.
if MAIN:
    play_key_length = PLAY_PAST_LENGTH + PLAY_QUERY_LENGTH
if MAIN:
    play_mask = build_causal_attention_mask(
        query_length=PLAY_QUERY_LENGTH, key_length=play_key_length,
        past_length=PLAY_PAST_LENGTH, sliding_window=PLAY_WINDOW,
        device=t.device("cpu"),
    ).squeeze(0).squeeze(0)
if MAIN:
    play_global = build_causal_attention_mask(
        query_length=PLAY_QUERY_LENGTH, key_length=play_key_length,
        past_length=PLAY_PAST_LENGTH, sliding_window=None,
        device=t.device("cpu"),
    ).squeeze(0).squeeze(0)
if MAIN:
    play_wrong_origin = build_causal_attention_mask(
        query_length=PLAY_QUERY_LENGTH, key_length=play_key_length,
        past_length=0, sliding_window=PLAY_WINDOW,
        device=t.device("cpu"),
    ).squeeze(0).squeeze(0)
if MAIN:
    play_wrong_bits = int((play_mask != play_wrong_origin).sum())
if MAIN:
    play_counts = play_mask.sum(dim=-1).tolist()
if MAIN:
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), constrained_layout=True)
if MAIN:
    for ax, values, title in zip(
        axes, (play_mask, play_global, play_wrong_origin),
        ("Correct cached local mask", "Global baseline", "Wrong zero-origin control"),
    ):
        ax.imshow(values.numpy(), vmin=0, vmax=1, cmap="Blues", aspect="auto")
        ax.set(xticks=range(play_key_length), yticks=range(PLAY_QUERY_LENGTH),
               xlabel="absolute key position", ylabel="query row", title=title)
if MAIN:
    print("allowed keys per query:", play_counts)
if MAIN:
    print("wrong-origin differing bits:", play_wrong_bits)
if MAIN:
    plt.show()

# %%

if MAIN:
    import transformers
if MAIN:
    from transformers import GemmaConfig as HFGemmaConfig
if MAIN:
    from transformers import GemmaForCausalLM as HFGemmaForCausalLM

if MAIN:
    local_config = make_tiny_gemma_config()
if MAIN:
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
if MAIN:
    t.manual_seed(5101)
if MAIN:
    hf_reference = HFGemmaForCausalLM(hf_config).to(DEVICE).eval()
if MAIN:
    hf_learner = GemmaForCausalLM(local_config).to(DEVICE).eval()
if MAIN:
    hf_learner.load_state_dict(hf_reference.state_dict())
if MAIN:
    hf_input_ids = t.tensor([[1, 5, 8, 13, 2]], device=DEVICE)
if MAIN:
    with t.inference_mode():
        hf_logits = hf_reference(input_ids=hf_input_ids, use_cache=False).logits
        learner_logits = hf_learner(hf_input_ids, use_cache=False).logits
if MAIN:
    hf_delta = learner_logits - hf_logits
if MAIN:
    hf_max_abs = hf_delta.abs().max().item()
if MAIN:
    hf_mse = hf_delta.square().mean().item()
if MAIN:
    hf_top5_agreement = (
        learner_logits.topk(5, dim=-1).indices == hf_logits.topk(5, dim=-1).indices
    ).float().mean().item()
if MAIN:
    assert hf_max_abs <= 5e-4 and hf_mse <= 1e-7 and hf_top5_agreement == 1.0
if MAIN:
    print("transformers version:", transformers.__version__)
if MAIN:
    print("state-dict keys loaded:", len(hf_reference.state_dict()))
if MAIN:
    print(f"max abs logit error: {hf_max_abs:.8f}")
if MAIN:
    print(f"logit MSE: {hf_mse:.3e}")
if MAIN:
    print(f"ordered top-5 agreement: {hf_top5_agreement:.3f}")

# %%

def hf_split_to_interleaved_projection(weight: t.Tensor, num_heads: int) -> t.Tensor:
    """Change each Q/K head's output-coordinate basis; preserve head identity."""
    if weight.ndim != 2 or num_heads <= 0 or weight.shape[0] % num_heads:
        raise ValueError("Expected a projection matrix with equal-sized heads")
    head_dim = weight.shape[0] // num_heads
    if head_dim == 0 or head_dim % 2:
        raise ValueError("RoPE head dimension must be positive and even")
    half = head_dim // 2
    pair_order = t.stack((t.arange(half, device=weight.device),
                          t.arange(half, device=weight.device) + half), dim=-1).flatten()
    return weight.reshape(num_heads, head_dim, weight.shape[1]).index_select(1, pair_order).reshape_as(weight)


if MAIN:
    for heads in (2, 4):
        marked_rows = t.arange(heads * 8 * 3).reshape(heads * 8, 3)
        converted_rows = hf_split_to_interleaved_projection(marked_rows, heads)
        expected_order = t.tensor([0, 4, 1, 5, 2, 6, 3, 7])
        for head in range(heads):
            t.testing.assert_close(converted_rows[head*8:(head+1)*8],
                                   marked_rows[head*8 + expected_order])
if MAIN:
    print("Projection checks passed: dimension-eight permutation and Q/K head identity.")

# %%

def gemma_tanh_gelu_gate(gate: t.Tensor, up: t.Tensor) -> t.Tensor:
    """Match the frozen public Gemma config's GELU(tanh) gated activation."""
    return F.gelu(gate, approximate="tanh") * up


if MAIN:
    gate_inputs = t.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=t.float64)
if MAIN:
    expected_gelu = t.tensor([
        0.5 * x * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715*x**3)))
        for x in gate_inputs.tolist()
    ], dtype=t.float64)
if MAIN:
    t.testing.assert_close(gemma_tanh_gelu_gate(gate_inputs, t.ones_like(gate_inputs)),
                           expected_gelu, atol=1e-12, rtol=0)
if MAIN:
    activation_control_gap = float((expected_gelu - F.silu(gate_inputs)).abs().max())
if MAIN:
    assert abs(activation_control_gap - 0.19300353813201032) <= 1e-12
if MAIN:
    print("GELU formula check passed; SiLU control max difference:", activation_control_gap)


class ConventionMatchedGemmaMLP(GemmaMLP):
    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.down_proj(gemma_tanh_gelu_gate(self.gate_proj(x), self.up_proj(x)))

# %%

# Independent explicit rotations, using relative positions 2 and 1.
if MAIN:
    component_q = t.tensor([1.0, 2.0, 3.0, 4.0], dtype=t.float64)
if MAIN:
    component_k = t.tensor([2.0, -1.0, 0.5, 3.0], dtype=t.float64)
if MAIN:
    component_frequencies = t.tensor([1.0, 0.01], dtype=t.float64)


def explicit_half_rotation(vector: t.Tensor, position: int) -> t.Tensor:
    angles = component_frequencies * position
    rotated = t.cat((-vector[2:], vector[:2]))
    return vector * angles.cos().repeat(2) + rotated * angles.sin().repeat(2)


def explicit_pair_rotation(vector: t.Tensor, position: int) -> t.Tensor:
    angles = component_frequencies * position
    pairs = vector.reshape(2, 2)
    rotated = t.stack((-pairs[:, 1], pairs[:, 0]), dim=-1).flatten()
    return vector * angles.cos().repeat_interleave(2) + rotated * angles.sin().repeat_interleave(2)


if MAIN:
    component_order = t.tensor([0, 2, 1, 3])
if MAIN:
    component_half_score = float(explicit_half_rotation(component_q, 2) @ explicit_half_rotation(component_k, 1))
if MAIN:
    component_correct_score = float(explicit_pair_rotation(component_q[component_order], 2)
                                    @ explicit_pair_rotation(component_k[component_order], 1))
if MAIN:
    component_wrong_score = float(explicit_pair_rotation(component_q, 2) @ explicit_pair_rotation(component_k, 1))
if MAIN:
    assert abs(component_half_score - 7.362465991603378) <= 1e-12
if MAIN:
    assert abs(component_correct_score - component_half_score) <= 1e-12
if MAIN:
    assert abs(component_wrong_score - 9.361968914924665) <= 1e-12
if MAIN:
    assert abs(component_wrong_score - component_half_score) > 1.0
if MAIN:
    print("HF-coordinate score:", component_half_score)
if MAIN:
    print("Converted-coordinate score:", component_correct_score)
if MAIN:
    print("Wrong-coordinate score:", component_wrong_score)

# %%

import hashlib
import json


def run_convention_matched_hf_probe() -> dict:
    """One fixed tiny CPU comparison; no pretrained weights, fitting or search."""
    from transformers import GemmaConfig as HFGemmaConfig
    from transformers import GemmaForCausalLM as HFGemmaForCausalLM

    config = make_tiny_gemma_config()
    hf_config = HFGemmaConfig(
        vocab_size=31, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=4, max_position_embeddings=64, rms_norm_eps=1e-6,
        rope_theta=10000.0, hidden_act="gelu_pytorch_tanh",
        initializer_range=0.02, attention_bias=False, attention_dropout=0.0,
        pad_token_id=0, bos_token_id=1, eos_token_id=2,
        tie_word_embeddings=True,
    )
    hf_config._attn_implementation = "eager"
    t.manual_seed(5101)
    reference = HFGemmaForCausalLM(hf_config).to(device="cpu", dtype=t.float32).eval()
    assert reference.config._attn_implementation == "eager"
    inputs = SIGNATURE_INPUT_IDS.to(device="cpu")
    assert inputs.tolist() == [[1, 5, 8, 13, 2, 21, 3], [3, 2, 1, 8, 5, 13, 21],
                               [21, 13, 8, 5, 3, 2, 1], [2, 3, 5, 8, 13, 21, 1]]
    assert config.rms_norm_eps == 1e-6 and config.rope_theta == 10000.0
    state = reference.state_dict()
    assert len(state) == 21
    with t.inference_mode():
        expected = reference(input_ids=inputs, use_cache=False).logits
    assert expected.shape == (4, 7, 31) and expected.numel() == 868
    rows = []
    for label, match_activation, match_basis in (
        ("unmatched conventions", False, False),
        ("GELU only", True, False),
        ("Q/K basis only", False, True),
        ("both conventions matched", True, True),
    ):
        learner = GemmaForCausalLM(config).to(device="cpu", dtype=t.float32).eval()
        if match_activation:
            for layer in learner.model.layers:
                layer.mlp = ConventionMatchedGemmaMLP(config).to(device="cpu", dtype=t.float32).eval()
        matched_state = {name: value.clone() for name, value in state.items()}
        if match_basis:
            for index in range(config.num_hidden_layers):
                for projection, heads in (("q_proj", config.num_attention_heads), ("k_proj", config.num_key_value_heads)):
                    key = f"model.layers.{index}.self_attn.{projection}.weight"
                    matched_state[key] = hf_split_to_interleaved_projection(matched_state[key], heads)
        learner.load_state_dict(matched_state, strict=True)
        with t.inference_mode():
            observed = learner(inputs, use_cache=False).logits
        delta = observed - expected
        rows.append({"label": label, "activation_matched": match_activation,
                     "basis_matched": match_basis, "max_abs": float(delta.abs().max()),
                     "mse": float(delta.square().mean()),
                     "ordered_top5_agreement": float((observed.topk(5, dim=-1).indices
                                                       == expected.topk(5, dim=-1).indices).float().mean())})
    state_hashes = {name: hashlib.sha256(value.detach().contiguous().numpy().tobytes()).hexdigest()
                    for name, value in state.items()}
    return {"rows": rows, "state_dict_keys": len(state), "state_dict_sha256": state_hashes,
            "input_ids": inputs.tolist(),
            "reference_config": hf_config.to_dict(), "seed": 5101,
            "corrected_passed": rows[-1]["max_abs"] <= 5e-6 and rows[-1]["mse"] <= 1e-12}


if MAIN:
    convention_probe = run_convention_matched_hf_probe()
if MAIN:
    for row in convention_probe["rows"]:
        print(f"{row['label']:26s} max_abs={row['max_abs']:.9g}  MSE={row['mse']:.9g}")
if MAIN:
    print("Corrected predeclared gate:", convention_probe["corrected_passed"])
if MAIN:
    print("ARENA_CONVENTION_PROBE=" + json.dumps(convention_probe, sort_keys=True))

if MAIN:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7), constrained_layout=True)
if MAIN:
    axes[0].plot(gate_inputs, expected_gelu, "o-", label="GELU(tanh)")
if MAIN:
    axes[0].plot(gate_inputs, F.silu(gate_inputs), "s--", label="SiLU control")
if MAIN:
    axes[0].set(title="Activation conventions differ", xlabel="gate input", ylabel="gate output")
if MAIN:
    axes[0].legend(fontsize=8)
if MAIN:
    axes[1].bar(["HF basis", "converted", "wrong basis"],
                [component_half_score, component_correct_score, component_wrong_score])
if MAIN:
    axes[1].set(title="Relative-position attention score", ylabel="q(position 2) dot k(position 1)")
if MAIN:
    axes[1].tick_params(axis="x", rotation=20)
if MAIN:
    axes[2].bar(["original", "GELU", "basis", "both"],
                [max(row["max_abs"], 1e-12) for row in convention_probe["rows"]])
if MAIN:
    axes[2].axhline(5e-6, color="black", linestyle="--", label="fixed corrected tolerance")
if MAIN:
    axes[2].set(yscale="log", title="One fixed tiny random-weight probe", ylabel="maximum absolute logit error")
if MAIN:
    axes[2].legend(fontsize=7)
if MAIN:
    plt.show()
if MAIN:
    assert convention_probe["corrected_passed"], "Preserve failure: do not relax tolerance or choose a new seed"

# %%

import importlib


def _release_solution_module():
    return importlib.import_module("chapter5_modern_architectures.exercises.part1_gemma_from_scratch.legacy_solutions")


def run_gpu_test(max_vram_gb: float = 24.0) -> dict:
    """Run the section's pinned CUDA verification contract."""
    return _release_solution_module().run_gpu_test(max_vram_gb=max_vram_gb)


def run_full_experiment(max_vram_gb: float = 24.0) -> dict:
    """Run the full experiment used for the checked-in release evidence."""
    return _release_solution_module().run_full_experiment(
        max_vram_gb=max_vram_gb
    )

# %%

def cache_parity_report(*args, **kwargs):
    """Preserved compatibility helper; original source remains in legacy_solutions.py."""
    return _release_solution_module().cache_parity_report(*args, **kwargs)

def cache_parity_smoke_test(*args, **kwargs):
    """Preserved compatibility helper; original source remains in legacy_solutions.py."""
    return _release_solution_module().cache_parity_smoke_test(*args, **kwargs)

def compare_tiny_model_to_clone(*args, **kwargs):
    """Preserved compatibility helper; original source remains in legacy_solutions.py."""
    return _release_solution_module().compare_tiny_model_to_clone(*args, **kwargs)

def local_memory_budget(*args, **kwargs):
    """Preserved compatibility helper; original source remains in legacy_solutions.py."""
    return _release_solution_module().local_memory_budget(*args, **kwargs)

def rms_norm_smoke_test(*args, **kwargs):
    """Preserved compatibility helper; original source remains in legacy_solutions.py."""
    return _release_solution_module().rms_norm_smoke_test(*args, **kwargs)

def rope_norm_smoke_test(*args, **kwargs):
    """Preserved compatibility helper; original source remains in legacy_solutions.py."""
    return _release_solution_module().rope_norm_smoke_test(*args, **kwargs)

def run_hf_tiny_gemma_reference_parity(*args, **kwargs):
    """Preserved compatibility helper; original source remains in legacy_solutions.py."""
    return _release_solution_module().run_hf_tiny_gemma_reference_parity(*args, **kwargs)

def run_smoke_test(*args, **kwargs):
    """Preserved compatibility helper; original source remains in legacy_solutions.py."""
    return _release_solution_module().run_smoke_test(*args, **kwargs)

# %%
