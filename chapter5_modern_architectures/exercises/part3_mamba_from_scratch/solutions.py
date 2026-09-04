# %%
"""Reference solutions for [5.3] Mamba from Scratch.

The learner-facing implementation follows the Mamba-1 computation from an exact
scalar recurrence through a weight-compatible causal language model. The code
is intentionally ordinary PyTorch; fused kernels belong to release verification.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch as t
import torch.nn as nn
import torch.nn.functional as F

REAL_MAMBA_MODEL_ID = "state-spaces/mamba-130m-hf"
REAL_MAMBA_REVISION = "1e76775f628fbf1350fbe4dbb3d971ba64af25a1"

EVENT_HOLD = 0
EVENT_WRITE = 1
EVENT_READ = 2
EVENT_ERASE = 3


@dataclass(frozen=True)
class MambaConfig:
    vocab_size: int = 32_000
    d_model: int = 768
    d_inner: int = 1_536
    d_state: int = 16
    d_conv: int = 4
    dt_rank: int = 48
    num_layers: int = 24
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = True
    residual_in_fp32: bool = True
    use_bias: bool = False
    use_conv_bias: bool = True

    @classmethod
    def from_hf_config(cls, config: Any) -> "MambaConfig":
        return cls(
            vocab_size=int(config.vocab_size),
            d_model=int(config.hidden_size),
            d_inner=int(config.intermediate_size),
            d_state=int(config.state_size),
            d_conv=int(config.conv_kernel),
            dt_rank=int(config.time_step_rank),
            num_layers=int(config.num_hidden_layers),
            rms_norm_eps=float(config.layer_norm_epsilon),
            tie_word_embeddings=bool(config.tie_word_embeddings),
            residual_in_fp32=bool(config.residual_in_fp32),
            use_bias=bool(config.use_bias),
            use_conv_bias=bool(config.use_conv_bias),
        )


@dataclass(frozen=True)
class MambaInferenceState:
    conv_state: t.Tensor
    ssm_state: t.Tensor


@dataclass(frozen=True)
class MambaCausalLMOutput:
    logits: t.Tensor
    states: tuple[MambaInferenceState, ...] | None = None
    hidden_states: tuple[t.Tensor, ...] | None = None


@dataclass(frozen=True)
class SelectiveCopyCase:
    event_ids: t.Tensor
    values: t.Tensor
    labels: tuple[str, ...]
    read_positions: t.Tensor
    read_targets: t.Tensor


@dataclass(frozen=True)
class SelectiveCopyResult:
    case: SelectiveCopyCase
    a: t.Tensor
    b: t.Tensor
    c: t.Tensor
    sequential_states: t.Tensor
    parallel_states: t.Tensor
    chunked_states: t.Tensor
    selective_reads: t.Tensor
    fixed_decay_reads: t.Tensor
    reset_chunk_reads: t.Tensor
    ablated_reads: t.Tensor
    parity_max_abs_diff: float
    chunked_max_abs_diff: float
    selective_read_mae: float
    fixed_decay_read_mae: float
    reset_chunk_read_mae: float
    ablation_effect: float


@dataclass(frozen=True)
class MemoryBenchmarkResult:
    distances: t.Tensor
    selective_error: t.Tensor
    attention_error: t.Tensor
    convolution_error: t.Tensor
    trace_labels: tuple[str, ...]
    trace_state: t.Tensor
    conv_kernel: int


# %%
# Exact affine recurrence: the ground-truth starting point.
def sequential_affine_scan(
    a: t.Tensor,
    b: t.Tensor,
    initial_state: t.Tensor | None = None,
) -> t.Tensor:
    """Apply h_t = a_t * h_(t-1) + b_t from left to right."""

    if a.shape != b.shape or a.ndim < 2:
        raise ValueError("a and b must have the same shape with sequence on axis 1.")
    state = t.zeros_like(b[:, 0]) if initial_state is None else initial_state.to(b)
    states = []
    for position in range(a.shape[1]):
        state = a[:, position] * state + b[:, position]
        states.append(state)
    return t.stack(states, dim=1)


def compose_affine_updates(
    a_left: t.Tensor,
    b_left: t.Tensor,
    a_right: t.Tensor,
    b_right: t.Tensor,
) -> tuple[t.Tensor, t.Tensor]:
    """Compose the right affine update after the left affine update."""

    return a_right * a_left, a_right * b_left + b_right


def parallel_affine_scan(
    a: t.Tensor,
    b: t.Tensor,
    initial_state: t.Tensor | None = None,
) -> t.Tensor:
    """Inclusive Hillis-Steele scan over affine recurrence updates."""

    if a.shape != b.shape or a.ndim < 2:
        raise ValueError("a and b must have the same shape with sequence on axis 1.")
    a_prefix, b_prefix = a.clone(), b.clone()
    offset = 1
    while offset < a.shape[1]:
        old_a, old_b = a_prefix, b_prefix
        next_a, next_b = old_a.clone(), old_b.clone()
        next_a[:, offset:], next_b[:, offset:] = compose_affine_updates(
            old_a[:, :-offset],
            old_b[:, :-offset],
            old_a[:, offset:],
            old_b[:, offset:],
        )
        a_prefix, b_prefix = next_a, next_b
        offset *= 2
    state0 = t.zeros_like(b[:, 0]) if initial_state is None else initial_state.to(b)
    return a_prefix * state0[:, None] + b_prefix


def chunked_affine_scan(
    a: t.Tensor,
    b: t.Tensor,
    chunk_size: int,
    initial_state: t.Tensor | None = None,
    *,
    reset_each_chunk: bool = False,
) -> t.Tensor:
    """Run the same recurrence in chunks while carrying the final state."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    state = t.zeros_like(b[:, 0]) if initial_state is None else initial_state.to(b)
    chunks = []
    for start in range(0, a.shape[1], chunk_size):
        if reset_each_chunk and start > 0:
            state = t.zeros_like(state)
        chunk = sequential_affine_scan(
            a[:, start : start + chunk_size],
            b[:, start : start + chunk_size],
            state,
        )
        chunks.append(chunk)
        state = chunk[:, -1]
    return t.cat(chunks, dim=1)


# %%
# Continuous-time parameterization and Mamba's selective discretization.
def stable_continuous_A(A_log: t.Tensor) -> t.Tensor:
    """Parameterize a diagonal continuous-time transition with negative entries."""

    return -t.exp(A_log.float())


def discretize_scalar_ssm_exact(
    A: t.Tensor,
    B: t.Tensor,
    delta: t.Tensor,
) -> tuple[t.Tensor, t.Tensor]:
    """Exact zero-order-hold discretization for dh/dt = A h + B u."""

    if bool((A == 0).any()):
        raise ValueError("A must be nonzero for this closed form.")
    a_bar = t.exp(delta * A)
    b_bar = t.expm1(delta * A) / A * B
    return a_bar, b_bar


def _expand_bc(param: t.Tensor, d_inner: int) -> t.Tensor:
    if param.ndim == 3:
        return param[:, :, None, :].expand(-1, -1, d_inner, -1)
    if param.ndim == 4 and param.shape[2] == d_inner:
        return param
    raise ValueError(
        "B/C must have shape (batch, seq, d_state) or "
        "(batch, seq, d_inner, d_state)."
    )


def discretize_selective_scan(
    u: t.Tensor,
    delta: t.Tensor,
    A_log: t.Tensor,
    B: t.Tensor,
) -> tuple[t.Tensor, t.Tensor]:
    """Build Mamba recurrence coefficients a_t and deltaB_t u_t."""

    if u.shape != delta.shape:
        raise ValueError("u and delta must both have shape (batch, seq, d_inner).")
    if A_log.ndim != 2 or A_log.shape[0] != u.shape[-1]:
        raise ValueError("A_log must have shape (d_inner, d_state).")
    A = stable_continuous_A(A_log).to(device=u.device)
    delta_f = delta.float()
    B_f = _expand_bc(B.float().to(u.device), u.shape[-1])
    a = t.exp(delta_f.unsqueeze(-1) * A[None, None])
    delta_B = delta_f.unsqueeze(-1) * B_f
    b = delta_B * u.float().unsqueeze(-1)
    return a, b


def selective_scan_recurrent(
    u: t.Tensor,
    delta: t.Tensor,
    A_log: t.Tensor,
    B: t.Tensor,
    C: t.Tensor,
    D: t.Tensor | None = None,
    z: t.Tensor | None = None,
    initial_state: t.Tensor | None = None,
    *,
    return_last_state: bool = False,
) -> t.Tensor | tuple[t.Tensor, t.Tensor]:
    """Readable sequential selective scan used as the semantic reference."""

    a, b = discretize_selective_scan(u, delta, A_log, B)
    state = (
        t.zeros(a.shape[0], a.shape[2], a.shape[3], device=u.device, dtype=t.float32)
        if initial_state is None
        else initial_state.float().to(u.device)
    )
    C_f = C.float().to(u.device)
    outputs = []
    for position in range(u.shape[1]):
        state = a[:, position] * state + b[:, position]
        if C_f.ndim == 3:
            read = t.matmul(
                state.to(u.dtype),
                C_f[:, position].to(u.dtype).unsqueeze(-1),
            ).squeeze(-1)
        else:
            read = (
                state.to(u.dtype) * C_f[:, position].to(u.dtype)
            ).sum(dim=-1)
        outputs.append(read)
    y = t.stack(outputs, dim=1)
    if D is not None:
        y = y + u * D.to(device=u.device, dtype=u.dtype)
    if z is not None:
        y = y * F.silu(z)
    return (y, state) if return_last_state else y


def selective_scan_parallel(
    u: t.Tensor,
    delta: t.Tensor,
    A_log: t.Tensor,
    B: t.Tensor,
    C: t.Tensor,
    D: t.Tensor | None = None,
    z: t.Tensor | None = None,
    initial_state: t.Tensor | None = None,
    *,
    return_last_state: bool = False,
) -> t.Tensor | tuple[t.Tensor, t.Tensor]:
    """Associative selective scan with the same ordered affine transforms."""

    a, b = discretize_selective_scan(u, delta, A_log, B)
    states = parallel_affine_scan(a, b, initial_state)
    C_f = _expand_bc(C.float().to(u.device), u.shape[-1])
    y = (states.to(u.dtype) * C_f.to(u.dtype)).sum(dim=-1)
    if D is not None:
        y = y + u * D.to(device=u.device, dtype=u.dtype)
    if z is not None:
        y = y * F.silu(z)
    return (y, states[:, -1]) if return_last_state else y


# %%
# Local causal mixing and token-conditioned SSM parameters.
def causal_depthwise_conv1d(
    x: t.Tensor,
    weight: t.Tensor,
    bias: t.Tensor | None = None,
) -> t.Tensor:
    """Depthwise causal cross-correlation for x shaped (batch, seq, channels)."""

    if weight.ndim != 3 or weight.shape[1] != 1 or weight.shape[0] != x.shape[-1]:
        raise ValueError("weight must have shape (channels, 1, kernel).")
    seq_len = x.shape[1]
    y = F.conv1d(
        x.transpose(1, 2),
        weight,
        bias,
        padding=weight.shape[-1] - 1,
        groups=x.shape[-1],
    )[..., :seq_len]
    return y.transpose(1, 2)


def project_selective_parameters(
    x: t.Tensor,
    x_proj: nn.Linear,
    dt_proj: nn.Linear,
    d_state: int,
) -> tuple[t.Tensor, t.Tensor, t.Tensor]:
    """Project each token into positive delta_t and input-dependent B_t, C_t."""

    parameters = x_proj(x)
    dt_raw, B, C = t.split(
        parameters,
        [dt_proj.in_features, d_state, d_state],
        dim=-1,
    )
    delta = F.softplus(dt_proj(dt_raw))
    return delta, B, C


class MambaRMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(t.ones(d_model))
        self.eps = eps

    def forward(self, x: t.Tensor) -> t.Tensor:
        dtype = x.dtype
        normalized = x.float() * t.rsqrt(
            x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps
        )
        return self.weight * normalized.to(dtype)


class MambaMixerFromScratch(nn.Module):
    """Mamba-1 mixer: projection, causal conv, selective SSM, gate, output."""

    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        self.in_proj = nn.Linear(config.d_model, 2 * config.d_inner, bias=config.use_bias)
        self.conv1d = nn.Conv1d(
            config.d_inner,
            config.d_inner,
            kernel_size=config.d_conv,
            groups=config.d_inner,
            padding=config.d_conv - 1,
            bias=config.use_conv_bias,
        )
        self.x_proj = nn.Linear(
            config.d_inner,
            config.dt_rank + 2 * config.d_state,
            bias=False,
        )
        self.dt_proj = nn.Linear(config.dt_rank, config.d_inner, bias=True)
        A = t.arange(1, config.d_state + 1, dtype=t.float32).repeat(config.d_inner, 1)
        self.A_log = nn.Parameter(t.log(A))
        self.D = nn.Parameter(t.ones(config.d_inner))
        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=config.use_bias)

    def initial_state(
        self,
        batch: int,
        device: t.device,
        dtype: t.dtype,
    ) -> MambaInferenceState:
        return MambaInferenceState(
            conv_state=t.zeros(
                batch,
                self.config.d_inner,
                self.config.d_conv,
                device=device,
                dtype=dtype,
            ),
            ssm_state=t.zeros(
                batch,
                self.config.d_inner,
                self.config.d_state,
                device=device,
                dtype=t.float32,
            ),
        )

    def _full_conv_state(self, x: t.Tensor) -> t.Tensor:
        x_channels = x.transpose(1, 2)
        if x_channels.shape[-1] < self.config.d_conv:
            x_channels = F.pad(
                x_channels,
                (self.config.d_conv - x_channels.shape[-1], 0),
            )
        return x_channels[..., -self.config.d_conv :].contiguous()

    def forward(
        self,
        hidden_states: t.Tensor,
        *,
        inference_state: MambaInferenceState | None = None,
        use_cache: bool = False,
    ) -> tuple[t.Tensor, MambaInferenceState | None]:
        if inference_state is not None and hidden_states.shape[1] == 1:
            return self.step(hidden_states, inference_state)

        x, gate = self.in_proj(hidden_states).chunk(2, dim=-1)
        x_conv = F.silu(
            causal_depthwise_conv1d(x, self.conv1d.weight, self.conv1d.bias)
        )
        delta, B, C = project_selective_parameters(
            x_conv,
            self.x_proj,
            self.dt_proj,
            self.config.d_state,
        )
        y, last_ssm_state = selective_scan_recurrent(
            x_conv,
            delta,
            self.A_log,
            B,
            C,
            D=self.D,
            z=gate,
            initial_state=None if inference_state is None else inference_state.ssm_state,
            return_last_state=True,
        )
        output = self.out_proj(y)
        state = None
        if use_cache:
            state = MambaInferenceState(self._full_conv_state(x), last_ssm_state)
        return output, state

    def step(
        self,
        hidden_states: t.Tensor,
        inference_state: MambaInferenceState,
    ) -> tuple[t.Tensor, MambaInferenceState]:
        if hidden_states.shape[1] != 1:
            raise ValueError("step expects exactly one token.")
        x, gate = self.in_proj(hidden_states).chunk(2, dim=-1)
        conv_state = t.roll(inference_state.conv_state, shifts=-1, dims=-1)
        conv_state[..., -1] = x[:, 0]
        x_conv = (conv_state * self.conv1d.weight[:, 0][None]).sum(dim=-1)
        if self.conv1d.bias is not None:
            x_conv = x_conv + self.conv1d.bias
        x_conv = F.silu(x_conv).unsqueeze(1)
        delta, B, C = project_selective_parameters(
            x_conv,
            self.x_proj,
            self.dt_proj,
            self.config.d_state,
        )
        y, ssm_state = selective_scan_recurrent(
            x_conv,
            delta,
            self.A_log,
            B,
            C,
            D=self.D,
            z=gate,
            initial_state=inference_state.ssm_state,
            return_last_state=True,
        )
        return self.out_proj(y), MambaInferenceState(conv_state, ssm_state)


class MambaBlockFromScratch(nn.Module):
    """Pre-norm residual Mamba block."""

    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        self.norm = MambaRMSNorm(config.d_model, config.rms_norm_eps)
        self.mixer = MambaMixerFromScratch(config)

    def forward(
        self,
        hidden_states: t.Tensor,
        *,
        inference_state: MambaInferenceState | None = None,
        use_cache: bool = False,
    ) -> tuple[t.Tensor, MambaInferenceState | None]:
        residual = hidden_states.float() if self.config.residual_in_fp32 else hidden_states
        normalized = self.norm(hidden_states.to(self.norm.weight.dtype))
        mixed, next_state = self.mixer(
            normalized,
            inference_state=inference_state,
            use_cache=use_cache,
        )
        return residual + mixed, next_state


class MambaModelFromScratch(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        self.embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList(
            [MambaBlockFromScratch(config) for _ in range(config.num_layers)]
        )
        self.norm_f = MambaRMSNorm(config.d_model, config.rms_norm_eps)

    def initial_states(
        self,
        batch: int,
        device: t.device,
        dtype: t.dtype,
    ) -> tuple[MambaInferenceState, ...]:
        return tuple(layer.mixer.initial_state(batch, device, dtype) for layer in self.layers)

    def forward(
        self,
        input_ids: t.Tensor,
        *,
        states: tuple[MambaInferenceState, ...] | None = None,
        use_cache: bool = False,
        output_hidden_states: bool = False,
    ) -> tuple[
        t.Tensor,
        tuple[MambaInferenceState, ...] | None,
        tuple[t.Tensor, ...] | None,
    ]:
        hidden = self.embeddings(input_ids)
        if states is None and input_ids.shape[1] == 1 and use_cache:
            states = self.initial_states(input_ids.shape[0], input_ids.device, hidden.dtype)
        next_states = [] if use_cache else None
        hidden_trace = [] if output_hidden_states else None
        for index, layer in enumerate(self.layers):
            hidden, next_state = layer(
                hidden,
                inference_state=None if states is None else states[index],
                use_cache=use_cache,
            )
            if next_states is not None:
                if next_state is None:
                    raise RuntimeError("use_cache=True requires every block to return state.")
                next_states.append(next_state)
            if hidden_trace is not None:
                hidden_trace.append(hidden)
        hidden = self.norm_f(hidden)
        if hidden_trace is not None:
            hidden_trace.append(hidden)
        return (
            hidden,
            None if next_states is None else tuple(next_states),
            None if hidden_trace is None else tuple(hidden_trace),
        )


class MambaForCausalLMFromScratch(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        self.backbone = MambaModelFromScratch(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.backbone.embeddings.weight

    def forward(
        self,
        input_ids: t.Tensor,
        *,
        states: tuple[MambaInferenceState, ...] | None = None,
        use_cache: bool = False,
        output_hidden_states: bool = False,
    ) -> MambaCausalLMOutput:
        hidden, next_states, hidden_trace = self.backbone(
            input_ids,
            states=states,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
        )
        return MambaCausalLMOutput(
            self.lm_head(hidden).float(),
            next_states,
            hidden_trace,
        )

    @t.inference_mode()
    def greedy_generate(self, input_ids: t.Tensor, max_new_tokens: int) -> t.Tensor:
        generated = input_ids
        states = None
        next_input = input_ids
        for _ in range(max_new_tokens):
            output = self(next_input, states=states, use_cache=True)
            states = output.states
            next_input = output.logits[:, -1].argmax(dim=-1, keepdim=True)
            generated = t.cat([generated, next_input], dim=-1)
        return generated


# Backward-compatible aliases used by section-level harnesses.
TinyMambaBlock = MambaBlockFromScratch
TinyMambaModel = MambaModelFromScratch
TinyMambaForCausalLM = MambaForCausalLMFromScratch


# %%
# Explicit mapping to the pinned Hugging Face checkpoint.
def hf_to_local_key(hf_key: str) -> str:
    """Map one public Transformers Mamba key to the from-scratch module tree."""

    if hf_key in {"backbone.embeddings.weight", "backbone.norm_f.weight", "lm_head.weight"}:
        return hf_key
    if hf_key.startswith("backbone.layers."):
        return hf_key
    raise KeyError(f"Unexpected Mamba checkpoint key: {hf_key}")


def load_hf_mamba_weights(
    local_model: MambaForCausalLMFromScratch,
    hf_state_dict: dict[str, t.Tensor],
) -> dict[str, Any]:
    """Copy every checkpoint tensor through a visible one-key-at-a-time mapping."""

    mapped = {hf_to_local_key(key): value for key, value in hf_state_dict.items()}
    incompatible = local_model.load_state_dict(mapped, strict=True)
    return {
        "mapped_tensor_count": len(mapped),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }


def load_pinned_mamba_130m_cpu(
) -> tuple[Any, MambaForCausalLMFromScratch, dict[str, Any]]:
    """Load the cached pinned 130M reference and its weight-mapped local twin on CPU."""

    from transformers import MambaForCausalLM

    official = MambaForCausalLM.from_pretrained(
        REAL_MAMBA_MODEL_ID,
        revision=REAL_MAMBA_REVISION,
        local_files_only=True,
        dtype=t.float32,
    ).cpu().eval()
    local = MambaForCausalLMFromScratch(
        MambaConfig.from_hf_config(official.config)
    ).cpu().eval()
    mapping = load_hf_mamba_weights(local, official.state_dict())
    return official, local, mapping


@t.inference_mode()
def pinned_mamba_cpu_parity() -> dict[str, Any]:
    """Check every hidden state, final logits, cache behavior, and greedy tokens."""

    official, local, mapping = load_pinned_mamba_130m_cpu()
    input_ids = t.tensor([[10, 314, 2718, 11]], dtype=t.long)
    official_out = official(
        input_ids,
        attention_mask=t.ones_like(input_ids),
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    local_out = local(input_ids, output_hidden_states=True, use_cache=False)
    if local_out.hidden_states is None or official_out.hidden_states is None:
        raise RuntimeError("Both implementations must return hidden-state traces.")
    hidden_diffs = [
        (left.float() - right.float()).abs().max().item()
        for left, right in zip(
            local_out.hidden_states,
            official_out.hidden_states,
            strict=True,
        )
    ]
    hidden_mean_diffs = [
        (left.float() - right.float()).abs().mean().item()
        for left, right in zip(
            local_out.hidden_states,
            official_out.hidden_states,
            strict=True,
        )
    ]
    logits_diff = (local_out.logits - official_out.logits.float()).abs().max().item()
    logits_mean_diff = (
        local_out.logits - official_out.logits.float()
    ).abs().mean().item()
    logits_top1_agreement = (
        local_out.logits.argmax(dim=-1)
        == official_out.logits.argmax(dim=-1)
    ).float().mean().item()

    states = None
    cached_logits = []
    for position in range(input_ids.shape[1]):
        step = local(
            input_ids[:, position : position + 1],
            states=states,
            use_cache=True,
        )
        states = step.states
        cached_logits.append(step.logits)
    cache_diff = (
        local_out.logits - t.cat(cached_logits, dim=1)
    ).abs().max().item()

    local_tokens = local.greedy_generate(input_ids, max_new_tokens=3)
    official_tokens = official.generate(
        input_ids,
        attention_mask=t.ones_like(input_ids),
        max_new_tokens=3,
        do_sample=False,
        pad_token_id=0,
    )
    result = {
        **mapping,
        "model_id": REAL_MAMBA_MODEL_ID,
        "revision": REAL_MAMBA_REVISION,
        "hidden_state_count": len(hidden_diffs),
        "hidden_max_abs_diff": max(hidden_diffs),
        "hidden_max_mean_abs_diff": max(hidden_mean_diffs),
        "hidden_worst_layer": hidden_diffs.index(max(hidden_diffs)),
        "logits_max_abs_diff": logits_diff,
        "logits_mean_abs_diff": logits_mean_diff,
        "logits_top1_agreement": logits_top1_agreement,
        "cache_max_abs_diff": cache_diff,
        "greedy_tokens_match": bool(t.equal(local_tokens, official_tokens)),
        "generated_tokens": local_tokens[0].tolist(),
    }
    del official, local
    return result


# %%
# Exact selective-memory result and controls.
def make_selective_copy_case(
    first_value: float = 1.0,
    second_value: float = -0.7,
    first_delay: int = 2,
    second_delay: int = 2,
    final_delay: int = 1,
    dtype: t.dtype = t.float64,
) -> SelectiveCopyCase:
    if min(first_delay, second_delay, final_delay) < 0:
        raise ValueError("delay lengths must be non-negative.")
    events = (
        [EVENT_WRITE]
        + [EVENT_HOLD] * first_delay
        + [EVENT_READ, EVENT_HOLD, EVENT_WRITE]
        + [EVENT_HOLD] * second_delay
        + [EVENT_READ, EVENT_ERASE]
        + [EVENT_HOLD] * final_delay
        + [EVENT_READ]
    )
    event_ids = t.tensor([events], dtype=t.long)
    values = t.linspace(-0.9, 0.9, len(events), dtype=dtype).unsqueeze(0)
    writes = event_ids[0].eq(EVENT_WRITE).nonzero(as_tuple=False).flatten()
    values[0, writes[0]], values[0, writes[1]] = first_value, second_value
    reads = event_ids[0].eq(EVENT_READ).nonzero(as_tuple=False).flatten()
    targets = t.tensor([first_value, second_value, 0.0], dtype=dtype)
    labels = []
    names = {EVENT_HOLD: "hold", EVENT_READ: "READ", EVENT_ERASE: "ERASE"}
    for position, event in enumerate(events):
        labels.append(
            f"WRITE {values[0, position].item():+.1f}"
            if event == EVENT_WRITE
            else names[event]
        )
    return SelectiveCopyCase(event_ids, values, tuple(labels), reads, targets)


def build_event_coefficients(
    event_ids: t.Tensor,
    values: t.Tensor,
) -> tuple[t.Tensor, t.Tensor, t.Tensor]:
    """Exact limiting-case coefficients for WRITE/HOLD/READ/ERASE."""

    a = t.ones((*values.shape, 1), dtype=values.dtype, device=values.device)
    b, c = t.zeros_like(a), t.zeros_like(a)
    write = event_ids.eq(EVENT_WRITE)
    read = event_ids.eq(EVENT_READ)
    erase = event_ids.eq(EVENT_ERASE)
    a[..., 0][write | erase] = 0
    b[..., 0][write] = values[write]
    c[..., 0][read] = 1
    return a, b, c


def selective_readout(states: t.Tensor, c: t.Tensor) -> t.Tensor:
    return (states * c).sum(dim=-1)


def intervene_on_state(
    a: t.Tensor,
    b: t.Tensor,
    position: int,
    replacement: t.Tensor | float = 0.0,
    initial_state: t.Tensor | None = None,
) -> t.Tensor:
    if not 0 <= position < a.shape[1]:
        raise IndexError("position is outside the sequence.")
    prefix = sequential_affine_scan(
        a[:, : position + 1],
        b[:, : position + 1],
        initial_state,
    )
    state = t.broadcast_to(
        t.as_tensor(replacement, dtype=b.dtype, device=b.device),
        prefix[:, -1].shape,
    ).clone()
    prefix = prefix.clone()
    prefix[:, -1] = state
    suffix = []
    for index in range(position + 1, a.shape[1]):
        state = a[:, index] * state + b[:, index]
        suffix.append(state)
    return prefix if not suffix else t.cat([prefix, t.stack(suffix, dim=1)], dim=1)


def run_selective_copy_experiment(
    *,
    case: SelectiveCopyCase | None = None,
    fixed_decay: float = 0.82,
    chunk_size: int = 4,
    ablate_position: int = 6,
) -> SelectiveCopyResult:
    case = make_selective_copy_case() if case is None else case
    a, b, c = build_event_coefficients(case.event_ids, case.values)
    sequential = sequential_affine_scan(a, b)
    parallel = parallel_affine_scan(a, b)
    chunked = chunked_affine_scan(a, b, chunk_size)
    reads = selective_readout(sequential, c)[0, case.read_positions]
    fixed_reads = selective_readout(
        sequential_affine_scan(t.full_like(a, fixed_decay), b),
        c,
    )[0, case.read_positions]
    reset_reads = selective_readout(
        chunked_affine_scan(a, b, chunk_size, reset_each_chunk=True),
        c,
    )[0, case.read_positions]
    ablated_reads = selective_readout(
        intervene_on_state(a, b, ablate_position),
        c,
    )[0, case.read_positions]
    return SelectiveCopyResult(
        case,
        a,
        b,
        c,
        sequential,
        parallel,
        chunked,
        reads,
        fixed_reads,
        reset_reads,
        ablated_reads,
        (sequential - parallel).abs().max().item(),
        (sequential - chunked).abs().max().item(),
        (reads - case.read_targets).abs().mean().item(),
        (fixed_reads - case.read_targets).abs().mean().item(),
        (reset_reads - case.read_targets).abs().mean().item(),
        (reads[1] - ablated_reads[1]).abs().item(),
    )


def causal_attention_memory(
    event_ids: t.Tensor,
    values: t.Tensor,
    read_position: int,
    logit_gap: float = 30.0,
) -> t.Tensor:
    """One causal attention head whose query selects prior WRITE keys."""

    positions = t.arange(event_ids.shape[1], device=event_ids.device)
    scores = t.full(
        event_ids.shape,
        -logit_gap,
        dtype=values.dtype,
        device=values.device,
    )
    scores[event_ids.eq(EVENT_WRITE)] = logit_gap
    scores[:, positions > read_position] = -t.inf
    weights = scores.softmax(dim=-1)
    write_values = t.where(event_ids.eq(EVENT_WRITE), values, t.zeros_like(values))
    return (weights * write_values).sum(dim=-1)


def finite_causal_conv_memory(
    event_ids: t.Tensor,
    values: t.Tensor,
    kernel_size: int,
) -> t.Tensor:
    """Propagate write impulses through a finite depthwise causal convolution."""

    impulses = t.where(
        event_ids.eq(EVENT_WRITE),
        values,
        t.zeros_like(values),
    ).unsqueeze(-1)
    weight = t.ones(1, 1, kernel_size, dtype=values.dtype, device=values.device)
    return causal_depthwise_conv1d(impulses, weight)[..., 0]


def run_memory_delay_benchmark(
    distances: tuple[int, ...] = (1, 2, 4, 8, 16, 32),
    conv_kernel: int = 8,
    values: tuple[float, ...] = (-1.25, -0.5, 0.75, 1.5),
) -> MemoryBenchmarkResult:
    selective_error, attention_error, convolution_error = [], [], []
    trace_labels: tuple[str, ...] = ()
    trace_state = t.empty(0)
    for distance in distances:
        method_outputs: list[list[t.Tensor]] = [[], [], []]
        for value in values:
            event_ids = t.tensor(
                [[EVENT_WRITE] + [EVENT_HOLD] * (distance - 1) + [EVENT_READ]]
            )
            inputs = t.zeros_like(event_ids, dtype=t.float64)
            inputs[0, 0] = value
            a, b, c = build_event_coefficients(event_ids, inputs)
            states = sequential_affine_scan(a, b)
            method_outputs[0].append(selective_readout(states, c)[0, -1])
            method_outputs[1].append(
                causal_attention_memory(event_ids, inputs, distance)[0]
            )
            method_outputs[2].append(
                finite_causal_conv_memory(event_ids, inputs, conv_kernel)[0, -1]
            )
            if distance == max(distances) and value == values[-1]:
                trace_labels = tuple(
                    [f"WRITE {value:+.2f}"]
                    + ["hold"] * (distance - 1)
                    + ["READ"]
                )
                trace_state = states[0, :, 0]
        target = t.tensor(values, dtype=t.float64)
        selective_error.append((t.stack(method_outputs[0]) - target).abs().mean())
        attention_error.append((t.stack(method_outputs[1]) - target).abs().mean())
        convolution_error.append((t.stack(method_outputs[2]) - target).abs().mean())
    return MemoryBenchmarkResult(
        distances=t.tensor(distances),
        selective_error=t.stack(selective_error),
        attention_error=t.stack(attention_error),
        convolution_error=t.stack(convolution_error),
        trace_labels=trace_labels,
        trace_state=trace_state,
        conv_kernel=conv_kernel,
    )


# %%
# Stable smoke/release entry points.
def make_scan_inputs(seed: int = 0) -> tuple[t.Tensor, ...]:
    generator = t.Generator().manual_seed(seed)
    batch, seq, d_inner, d_state = 2, 7, 4, 3
    u = t.randn(batch, seq, d_inner, generator=generator)
    delta = t.rand(batch, seq, d_inner, generator=generator) + 0.1
    A_log = t.randn(d_inner, d_state, generator=generator) - 2
    B = t.randn(batch, seq, d_state, generator=generator)
    C = t.randn(batch, seq, d_state, generator=generator)
    D = t.randn(d_inner, generator=generator)
    z = t.randn(batch, seq, d_inner, generator=generator)
    return u, delta, A_log, B, C, D, z


def make_tiny_mamba_config() -> MambaConfig:
    return MambaConfig(
        vocab_size=31,
        d_model=12,
        d_inner=16,
        d_state=4,
        d_conv=3,
        dt_rank=4,
        num_layers=2,
    )


def scan_parity_report() -> dict[str, Any]:
    u, delta, A_log, B, C, D, z = make_scan_inputs()
    recurrent = selective_scan_recurrent(u, delta, A_log, B, C, D, z)
    parallel = selective_scan_parallel(u, delta, A_log, B, C, D, z)
    diff = (recurrent - parallel).abs()
    return {"max_abs_diff": diff.max().item(), "passed": bool(diff.max() <= 1e-5)}


def block_step_parity_report() -> dict[str, Any]:
    t.manual_seed(0)
    config = make_tiny_mamba_config()
    block = MambaBlockFromScratch(config).eval()
    hidden = t.randn(2, 6, config.d_model)
    full, _ = block(hidden)
    state = block.mixer.initial_state(2, hidden.device, hidden.dtype)
    steps = []
    for position in range(hidden.shape[1]):
        output, state = block(
            hidden[:, position : position + 1],
            inference_state=state,
            use_cache=True,
        )
        steps.append(output)
    stepped = t.cat(steps, dim=1)
    diff = (full - stepped).abs()
    return {"max_abs_diff": diff.max().item(), "passed": bool(diff.max() <= 1e-5)}


def lm_cache_parity_report() -> dict[str, Any]:
    t.manual_seed(1)
    model = MambaForCausalLMFromScratch(make_tiny_mamba_config()).eval()
    input_ids = t.tensor([[1, 2, 3, 4, 5]])
    full = model(input_ids).logits
    states = None
    steps = []
    for position in range(input_ids.shape[1]):
        output = model(
            input_ids[:, position : position + 1],
            states=states,
            use_cache=True,
        )
        states = output.states
        steps.append(output.logits)
    diff = (full - t.cat(steps, dim=1)).abs()
    return {"max_abs_diff": diff.max().item(), "passed": bool(diff.max() <= 1e-5)}


def run_smoke_test(cpu: bool = True) -> dict[str, Any]:
    _ = cpu
    benchmark = run_memory_delay_benchmark()
    parity = pinned_mamba_cpu_parity()
    return {
        "scan_parity": scan_parity_report(),
        "block_step_parity": block_step_parity_report(),
        "lm_cache_parity": lm_cache_parity_report(),
        "cpu_mamba_130m_mapped_tensor_count": parity["mapped_tensor_count"],
        "cpu_mamba_130m_hidden_state_count": parity["hidden_state_count"],
        "cpu_mamba_130m_hidden_max_abs_diff": parity["hidden_max_abs_diff"],
        "cpu_mamba_130m_hidden_max_mean_abs_diff": parity[
            "hidden_max_mean_abs_diff"
        ],
        "cpu_mamba_130m_logits_max_abs_diff": parity["logits_max_abs_diff"],
        "cpu_mamba_130m_logits_mean_abs_diff": parity["logits_mean_abs_diff"],
        "cpu_mamba_130m_logits_top1_agreement": parity[
            "logits_top1_agreement"
        ],
        "cpu_mamba_130m_cache_max_abs_diff": parity["cache_max_abs_diff"],
        "cpu_mamba_130m_greedy_tokens_match": parity["greedy_tokens_match"],
        "selective_memory_max_mae": benchmark.selective_error.max().item(),
        "selective_memory_attention_max_mae": benchmark.attention_error.max().item(),
        "selective_memory_conv_distance_32_mae": benchmark.convolution_error[-1].item(),
        "memory_benchmark": {
            "selective_max_mae": benchmark.selective_error.max().item(),
            "attention_max_mae": benchmark.attention_error.max().item(),
            "convolution_long_delay_mae": benchmark.convolution_error[-1].item(),
        },
    }


def official_mamba_logits_generation_preflight(
    max_vram_gb: float = 24.0,
) -> dict[str, Any]:
    """Existing bounded CUDA release path; not called by the learner CPU notebook."""

    if not t.cuda.is_available():
        return {"cuda_available": False, "skipped": True, "preflight_passed": False}
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = t.device("cuda")
    t.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(
        REAL_MAMBA_MODEL_ID,
        revision=REAL_MAMBA_REVISION,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        REAL_MAMBA_MODEL_ID,
        revision=REAL_MAMBA_REVISION,
        dtype=t.float16,
    ).to(device).eval()
    prompts = ["Mamba models can", "The cat sat on"]
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
    ).to(device)
    with t.inference_mode():
        logits = model(**inputs).logits.float()
        single_logits = [
            model(**tokenizer(text, return_tensors="pt").to(device)).logits.float()
            for text in prompts
        ]
    agreements = []
    for row, row_logits in enumerate(single_logits):
        valid = inputs.attention_mask[row].bool()
        batched_top1 = logits[row, valid].argmax(dim=-1)
        single_top1 = row_logits[0].argmax(dim=-1)
        agreements.append((batched_top1 == single_top1).float())
    batched_single_top1_agreement = t.cat(agreements).mean().item()

    from transformers.models.mamba import modeling_mamba

    fast_kernel_names = (
        "selective_state_update",
        "selective_scan_fn",
        "causal_conv1d_fn",
        "causal_conv1d_update",
        "mamba_inner_fn",
    )
    fast_kernel_available = all(
        getattr(modeling_mamba, name, None) is not None for name in fast_kernel_names
    )
    prompt = tokenizer("Mamba models can", return_tensors="pt").to(device)
    t.cuda.synchronize()
    start = time.perf_counter()
    with t.inference_mode():
        generated = model.generate(
            **prompt,
            max_new_tokens=4,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    t.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak = t.cuda.max_memory_allocated() / 1024**3
    prefix_ok = bool(
        t.equal(generated[:, : prompt.input_ids.shape[1]], prompt.input_ids)
    )
    result = {
        "cuda_available": True,
        "model_id": REAL_MAMBA_MODEL_ID,
        "revision": REAL_MAMBA_REVISION,
        "logits_shape": list(logits.shape),
        "logits_std": logits.std().item(),
        "batched_single_top1_agreement": batched_single_top1_agreement,
        "fast_kernel_available": fast_kernel_available,
        "generated_shape": list(generated.shape),
        "generated_new_token_count": generated.shape[1] - prompt.input_ids.shape[1],
        "generation_tokens_per_second": 4 / max(elapsed, 1e-9),
        "prompt_prefix_preserved": prefix_ok,
        "peak_vram_gb": peak,
        "within_vram_budget": peak <= max_vram_gb,
    }
    result["preflight_passed"] = bool(
        t.isfinite(logits).all()
        and batched_single_top1_agreement == 1.0
        and fast_kernel_available
        and result["generated_new_token_count"] == 4
        and prefix_ok
        and peak <= max_vram_gb
    )
    del model, tokenizer, inputs, logits, prompt, generated
    t.cuda.empty_cache()
    return result


def run_gpu_test(max_vram_gb: float = 24.0) -> dict[str, Any]:
    if not t.cuda.is_available():
        return {
            "cuda_available": False,
            "full_path": "CUDA verification was not requested in this CPU rewrite.",
        }
    smoke = run_smoke_test()
    official = official_mamba_logits_generation_preflight(max_vram_gb)
    return {
        "cuda_available": True,
        "device": t.cuda.get_device_name(0),
        "torch_version": t.__version__,
        "cuda_version": t.version.cuda,
        "scan_max_abs_diff": smoke["scan_parity"]["max_abs_diff"],
        "scan_passed": smoke["scan_parity"]["passed"],
        "cache_max_abs_diff": smoke["lm_cache_parity"]["max_abs_diff"],
        "cache_passed": smoke["lm_cache_parity"]["passed"],
        "official_mamba_preflight": official,
        "official_mamba_logits_generation_preflight_passed": official["preflight_passed"],
        "official_mamba_logits_shape": official.get("logits_shape"),
        "official_mamba_logits_std": official.get("logits_std"),
        "official_mamba_batched_single_top1_agreement": official.get(
            "batched_single_top1_agreement"
        ),
        "official_mamba_fast_kernel_available": official.get(
            "fast_kernel_available"
        ),
        "official_mamba_generated_shape": official.get("generated_shape"),
        "official_mamba_generation_new_tokens": official.get("generated_new_token_count"),
        "official_mamba_generation_tokens_per_second": official.get(
            "generation_tokens_per_second"
        ),
        "official_mamba_prompt_prefix_preserved": official.get(
            "prompt_prefix_preserved"
        ),
        "official_mamba_peak_vram_gb": official.get("peak_vram_gb"),
        "peak_vram_gb": official.get("peak_vram_gb"),
        "within_vram_budget": official.get("within_vram_budget"),
        "full_path": "Selective-scan/cache parity plus pinned Mamba-130M-HF CUDA preflight.",
    }


def run_full_experiment(max_vram_gb: float = 24.0) -> dict[str, Any]:
    return run_gpu_test(max_vram_gb=max_vram_gb)
