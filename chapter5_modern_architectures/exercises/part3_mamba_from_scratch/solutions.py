# %%
"""Reference solutions for [5.3] Mamba from Scratch."""

import sys
import time
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

MAIN = __name__ == "__main__"

REAL_MAMBA_MODEL_ID = "state-spaces/mamba-130m-hf"
REAL_MAMBA_REVISION = "1e76775f628fbf1350fbe4dbb3d971ba64af25a1"


@dataclass(frozen=True)
class MambaScanReport:
    max_abs_diff: float
    mse: float
    passed: bool
    atol: float


@dataclass(frozen=True)
class SelectiveCopyCase:
    """Exact write-hold-read-erase task for inspecting a selective recurrence."""

    event_ids: t.Tensor
    values: t.Tensor
    labels: tuple[str, ...]
    read_positions: t.Tensor
    read_targets: t.Tensor


@dataclass(frozen=True)
class SelectiveCopyResult:
    """State trajectories and controls used by the learner-facing signature figure."""

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
class MambaConfig:
    vocab_size: int = 32000
    d_model: int = 768
    d_inner: int = 1536
    d_state: int = 16
    d_conv: int = 4
    dt_rank: int = 32
    num_layers: int = 2
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = True


@dataclass(frozen=True)
class MambaInferenceState:
    conv_state: t.Tensor
    ssm_state: t.Tensor


@dataclass(frozen=True)
class MambaCausalLMOutput:
    logits: t.Tensor
    states: tuple[MambaInferenceState, ...] | None = None


def _expand_bc(param: t.Tensor, d_inner: int) -> t.Tensor:
    """Expand B/C parameters to shape (batch, seq, d_inner, d_state)."""

    if param.ndim == 3:
        return param[:, :, None, :].expand(-1, -1, d_inner, -1)
    if param.ndim == 4:
        if param.shape[2] != d_inner:
            raise ValueError("rank-4 B/C parameter has wrong d_inner dimension.")
        return param
    raise ValueError(
        "B/C parameter must have shape (batch, seq, d_state) "
        "or (batch, seq, d_inner, d_state)."
    )


def discretize_selective_scan(
    u: t.Tensor,
    delta: t.Tensor,
    A_log: t.Tensor,
    B: t.Tensor,
) -> tuple[t.Tensor, t.Tensor]:
    """Return coefficients for state_t = a_t * state_{t-1} + b_t."""

    if u.shape != delta.shape:
        raise ValueError("u and delta must have matching shape (batch, seq, d_inner).")
    if A_log.ndim != 2 or A_log.shape[0] != u.shape[-1]:
        raise ValueError("A_log must have shape (d_inner, d_state).")
    A = -t.exp(A_log.float()).to(dtype=u.dtype, device=u.device)
    B_expanded = _expand_bc(B.to(device=u.device, dtype=u.dtype), d_inner=u.shape[-1])
    a = t.exp(delta.unsqueeze(-1) * A[None, None, :, :])
    b = delta.unsqueeze(-1) * u.unsqueeze(-1) * B_expanded
    return a, b


def _expand_c(C: t.Tensor, d_inner: int, dtype: t.dtype, device: t.device) -> t.Tensor:
    return _expand_bc(C.to(device=device, dtype=dtype), d_inner=d_inner)


def _readout(
    states: t.Tensor,
    u: t.Tensor,
    C: t.Tensor,
    D: t.Tensor | None,
    z: t.Tensor | None,
) -> t.Tensor:
    C_expanded = _expand_c(C, d_inner=u.shape[-1], dtype=u.dtype, device=u.device)
    y = (states * C_expanded).sum(dim=-1)
    if D is not None:
        y = y + u * D.to(device=u.device, dtype=u.dtype)
    if z is not None:
        y = y * F.silu(z)
    return y


def selective_scan_recurrent(
    u: t.Tensor,
    delta: t.Tensor,
    A_log: t.Tensor,
    B: t.Tensor,
    C: t.Tensor,
    D: t.Tensor | None = None,
    z: t.Tensor | None = None,
    initial_state: t.Tensor | None = None,
    return_last_state: bool = False,
) -> t.Tensor | tuple[t.Tensor, t.Tensor]:
    """Reference recurrent selective scan."""

    a, b = discretize_selective_scan(u, delta, A_log, B)
    batch, _, d_inner, d_state = a.shape
    state = (
        t.zeros(batch, d_inner, d_state, dtype=u.dtype, device=u.device)
        if initial_state is None
        else initial_state.to(device=u.device, dtype=u.dtype)
    )
    states = []
    for pos in range(u.shape[1]):
        state = a[:, pos] * state + b[:, pos]
        states.append(state)
    states_tensor = t.stack(states, dim=1)
    y = _readout(states_tensor, u, C, D, z)
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
    return_last_state: bool = False,
) -> t.Tensor | tuple[t.Tensor, t.Tensor]:
    """Associative-scan version of the selective scan."""

    a_prefix, b_prefix = discretize_selective_scan(u, delta, A_log, B)
    step = 1
    seq_len = u.shape[1]
    while step < seq_len:
        a_old = a_prefix.clone()
        b_old = b_prefix.clone()
        a_prefix[:, step:] = a_old[:, step:] * a_old[:, :-step]
        b_prefix[:, step:] = a_old[:, step:] * b_old[:, :-step] + b_old[:, step:]
        step *= 2

    if initial_state is None:
        states = b_prefix
    else:
        state0 = initial_state.to(device=u.device, dtype=u.dtype)
        states = a_prefix * state0[:, None, :, :] + b_prefix
    y = _readout(states, u, C, D, z)
    last_state = states[:, -1]
    return (y, last_state) if return_last_state else y


def selective_scan_chunked(
    u: t.Tensor,
    delta: t.Tensor,
    A_log: t.Tensor,
    B: t.Tensor,
    C: t.Tensor,
    D: t.Tensor | None = None,
    z: t.Tensor | None = None,
    chunk_size: int = 64,
    initial_state: t.Tensor | None = None,
    return_last_state: bool = False,
) -> t.Tensor | tuple[t.Tensor, t.Tensor]:
    """Run the recurrent scan in chunks while carrying SSM state."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    state = initial_state
    outputs = []
    for start in range(0, u.shape[1], chunk_size):
        stop = min(start + chunk_size, u.shape[1])
        out, state = selective_scan_recurrent(
            u[:, start:stop],
            delta[:, start:stop],
            A_log,
            B[:, start:stop],
            C[:, start:stop],
            D=D,
            z=None if z is None else z[:, start:stop],
            initial_state=state,
            return_last_state=True,
        )
        outputs.append(out)
    y = t.cat(outputs, dim=1)
    return (y, state) if return_last_state else y


def scan_equivalence_report(
    reference: t.Tensor,
    candidate: t.Tensor,
    *,
    atol: float = 1e-4,
) -> MambaScanReport:
    diff = (reference.float() - candidate.float()).abs()
    return MambaScanReport(
        max_abs_diff=diff.max().item(),
        mse=diff.pow(2).mean().item(),
        passed=bool(diff.max().item() <= atol),
        atol=atol,
    )


EVENT_HOLD = 0
EVENT_WRITE = 1
EVENT_READ = 2
EVENT_ERASE = 3


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
    """Compose the right update after the left update."""

    return a_right * a_left, a_right * b_left + b_right


def parallel_affine_scan(
    a: t.Tensor,
    b: t.Tensor,
    initial_state: t.Tensor | None = None,
) -> t.Tensor:
    """Inclusive Hillis-Steele scan over affine recurrence updates."""

    if a.shape != b.shape or a.ndim < 2:
        raise ValueError("a and b must have the same shape with sequence on axis 1.")
    a_prefix = a.clone()
    b_prefix = b.clone()
    offset = 1
    while offset < a.shape[1]:
        old_a = a_prefix.clone()
        old_b = b_prefix.clone()
        composed_a, composed_b = compose_affine_updates(
            old_a[:, :-offset],
            old_b[:, :-offset],
            old_a[:, offset:],
            old_b[:, offset:],
        )
        a_prefix[:, offset:] = composed_a
        b_prefix[:, offset:] = composed_b
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
    """Run the sequential scan in chunks, optionally exposing the reset bug."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    state = t.zeros_like(b[:, 0]) if initial_state is None else initial_state.to(b)
    chunks = []
    for start in range(0, a.shape[1], chunk_size):
        stop = min(start + chunk_size, a.shape[1])
        if reset_each_chunk and start > 0:
            state = t.zeros_like(state)
        chunk_states = sequential_affine_scan(a[:, start:stop], b[:, start:stop], state)
        chunks.append(chunk_states)
        state = chunk_states[:, -1]
    return t.cat(chunks, dim=1)


def selective_readout(states: t.Tensor, c: t.Tensor) -> t.Tensor:
    """Read the recurrent state with an input-dependent C_t vector."""

    if states.shape != c.shape:
        raise ValueError("states and c must have the same shape.")
    return (states * c).sum(dim=-1)


def build_event_coefficients(
    event_ids: t.Tensor,
    values: t.Tensor,
) -> tuple[t.Tensor, t.Tensor, t.Tensor]:
    """Map exact WRITE/HOLD/READ/ERASE events to scalar affine updates."""

    if event_ids.shape != values.shape or event_ids.ndim != 2:
        raise ValueError("event_ids and values must both have shape (batch, sequence).")
    dtype = values.dtype
    a = t.ones((*values.shape, 1), dtype=dtype, device=values.device)
    b = t.zeros_like(a)
    c = t.zeros_like(a)
    write = event_ids.eq(EVENT_WRITE)
    erase = event_ids.eq(EVENT_ERASE)
    read = event_ids.eq(EVENT_READ)
    a[..., 0][write | erase] = 0.0
    b[..., 0][write] = values[write]
    c[..., 0][read] = 1.0
    return a, b, c


def intervene_on_state(
    a: t.Tensor,
    b: t.Tensor,
    position: int,
    replacement: t.Tensor | float = 0.0,
    initial_state: t.Tensor | None = None,
) -> t.Tensor:
    """Replace h_position, then continue the same recurrence downstream."""

    if not 0 <= position < a.shape[1]:
        raise IndexError("position is outside the sequence.")
    states = sequential_affine_scan(a[:, : position + 1], b[:, : position + 1], initial_state)
    replacement_state = t.as_tensor(replacement, dtype=b.dtype, device=b.device)
    state = t.broadcast_to(replacement_state, states[:, -1].shape).clone()
    states[:, -1] = state
    suffix = []
    for index in range(position + 1, a.shape[1]):
        state = a[:, index] * state + b[:, index]
        suffix.append(state)
    if suffix:
        return t.cat([states, t.stack(suffix, dim=1)], dim=1)
    return states


def make_selective_copy_case(
    first_value: float = 1.0,
    second_value: float = -0.7,
    first_delay: int = 2,
    second_delay: int = 2,
    final_delay: int = 1,
    dtype: t.dtype = t.float64,
) -> SelectiveCopyCase:
    """Return an exact scalar-state task with three ground-truth reads."""

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
    write_positions = event_ids[0].eq(EVENT_WRITE).nonzero(as_tuple=False).flatten()
    values[0, write_positions[0]] = first_value
    values[0, write_positions[1]] = second_value
    read_positions = event_ids[0].eq(EVENT_READ).nonzero(as_tuple=False).flatten()
    read_targets = t.tensor([first_value, second_value, 0.0], dtype=dtype)
    labels_list = []
    for position, event in enumerate(events):
        if event == EVENT_WRITE:
            labels_list.append(f"WRITE {values[0, position].item():+.1f}")
        elif event == EVENT_READ:
            labels_list.append("READ")
        elif event == EVENT_ERASE:
            labels_list.append("ERASE")
        else:
            labels_list.append("hold")
    labels = tuple(labels_list)
    return SelectiveCopyCase(event_ids, values, labels, read_positions, read_targets)


def run_selective_copy_experiment(
    *,
    case: SelectiveCopyCase | None = None,
    fixed_decay: float = 0.82,
    chunk_size: int = 4,
    ablate_position: int = 6,
) -> SelectiveCopyResult:
    """Run the exact model organism, controls, and a causal state ablation."""

    case = make_selective_copy_case() if case is None else case
    a, b, c = build_event_coefficients(case.event_ids, case.values)
    sequential_states = sequential_affine_scan(a, b)
    parallel_states = parallel_affine_scan(a, b)
    chunked_states = chunked_affine_scan(a, b, chunk_size)
    selective = selective_readout(sequential_states, c)[0, case.read_positions]

    fixed_a = t.full_like(a, fixed_decay)
    fixed_states = sequential_affine_scan(fixed_a, b)
    fixed_reads = selective_readout(fixed_states, c)[0, case.read_positions]

    reset_states = chunked_affine_scan(a, b, chunk_size, reset_each_chunk=True)
    reset_reads = selective_readout(reset_states, c)[0, case.read_positions]

    ablated_states = intervene_on_state(a, b, ablate_position, replacement=0.0)
    ablated_reads = selective_readout(ablated_states, c)[0, case.read_positions]

    return SelectiveCopyResult(
        case=case,
        a=a,
        b=b,
        c=c,
        sequential_states=sequential_states,
        parallel_states=parallel_states,
        chunked_states=chunked_states,
        selective_reads=selective,
        fixed_decay_reads=fixed_reads,
        reset_chunk_reads=reset_reads,
        ablated_reads=ablated_reads,
        parity_max_abs_diff=(sequential_states - parallel_states).abs().max().item(),
        chunked_max_abs_diff=(sequential_states - chunked_states).abs().max().item(),
        selective_read_mae=(selective - case.read_targets).abs().mean().item(),
        fixed_decay_read_mae=(fixed_reads - case.read_targets).abs().mean().item(),
        reset_chunk_read_mae=(reset_reads - case.read_targets).abs().mean().item(),
        ablation_effect=(selective[1] - ablated_reads[1]).abs().item(),
    )


class MambaRMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(t.ones(d_model))
        self.eps = eps

    def forward(self, x: t.Tensor) -> t.Tensor:
        dtype = x.dtype
        x_float = x.float()
        x_normed = x_float * t.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x_normed * self.weight.float()).to(dtype)


class TinyMambaBlock(nn.Module):
    """Minimal Mamba block with causal convolution and selective scan."""

    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        self.in_proj = nn.Linear(config.d_model, 2 * config.d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            config.d_inner,
            config.d_inner,
            kernel_size=config.d_conv,
            groups=config.d_inner,
            padding=0,
        )
        self.x_proj = nn.Linear(config.d_inner, config.dt_rank + 2 * config.d_state, bias=False)
        self.dt_proj = nn.Linear(config.dt_rank, config.d_inner, bias=True)
        A_init = t.arange(1, config.d_state + 1, dtype=t.float32).repeat(config.d_inner, 1)
        self.A_log = nn.Parameter(t.log(A_init))
        self.D = nn.Parameter(t.ones(config.d_inner))
        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=False)

    def initial_state(self, batch: int, device: t.device, dtype: t.dtype) -> MambaInferenceState:
        conv_width = max(self.config.d_conv - 1, 0)
        conv_state = t.zeros(batch, self.config.d_inner, conv_width, device=device, dtype=dtype)
        ssm_state = t.zeros(
            batch,
            self.config.d_inner,
            self.config.d_state,
            device=device,
            dtype=dtype,
        )
        return MambaInferenceState(conv_state=conv_state, ssm_state=ssm_state)

    def _causal_conv(self, x: t.Tensor) -> t.Tensor:
        x_conv = x.transpose(1, 2)
        x_conv = F.pad(x_conv, (self.config.d_conv - 1, 0))
        x_conv = self.conv1d(x_conv)
        return x_conv.transpose(1, 2)

    def _final_conv_state(self, x: t.Tensor) -> t.Tensor:
        conv_width = self.config.d_conv - 1
        if conv_width <= 0:
            return x.new_zeros(x.shape[0], self.config.d_inner, 0)
        x_conv = x.transpose(1, 2)
        if x_conv.shape[-1] < conv_width:
            x_conv = F.pad(x_conv, (conv_width - x_conv.shape[-1], 0))
        return x_conv[:, :, -conv_width:].contiguous()

    def _scan_parameters(self, x: t.Tensor) -> tuple[t.Tensor, t.Tensor, t.Tensor]:
        params = self.x_proj(x)
        dt_raw, B, C = t.split(
            params,
            [self.config.dt_rank, self.config.d_state, self.config.d_state],
            dim=-1,
        )
        delta = F.softplus(self.dt_proj(dt_raw))
        return delta, B, C

    def forward(
        self,
        hidden_states: t.Tensor,
        *,
        inference_state: MambaInferenceState | None = None,
        use_cache: bool = False,
    ) -> tuple[t.Tensor, MambaInferenceState | None]:
        if inference_state is not None and hidden_states.shape[1] == 1:
            return self.step(hidden_states, inference_state)

        xz = self.in_proj(hidden_states)
        x, z = xz.chunk(2, dim=-1)
        x_conv = F.silu(self._causal_conv(x))
        delta, B, C = self._scan_parameters(x_conv)

        y, ssm_state = selective_scan_recurrent(
            x_conv,
            delta,
            self.A_log,
            B,
            C,
            D=self.D,
            z=z,
            initial_state=None if inference_state is None else inference_state.ssm_state,
            return_last_state=True,
        )
        output = self.out_proj(y)
        if not use_cache:
            return output, None

        state = MambaInferenceState(
            conv_state=self._final_conv_state(x),
            ssm_state=ssm_state,
        )
        return output, state

    def step(
        self,
        hidden_states: t.Tensor,
        inference_state: MambaInferenceState,
    ) -> tuple[t.Tensor, MambaInferenceState]:
        if hidden_states.shape[1] != 1:
            raise ValueError("step expects a single-token sequence.")

        xz = self.in_proj(hidden_states)
        x, z = xz.chunk(2, dim=-1)
        x_now = x[:, 0]

        if self.config.d_conv > 1:
            conv_window = t.cat([inference_state.conv_state, x_now.unsqueeze(-1)], dim=-1)
            new_conv_state = conv_window[:, :, -(self.config.d_conv - 1) :].contiguous()
        else:
            conv_window = x_now.unsqueeze(-1)
            new_conv_state = inference_state.conv_state

        weight = self.conv1d.weight[:, 0, :].to(dtype=x.dtype, device=x.device)
        conv_out = (conv_window * weight[None, :, :]).sum(dim=-1)
        if self.conv1d.bias is not None:
            conv_out = conv_out + self.conv1d.bias.to(dtype=x.dtype, device=x.device)
        x_conv = F.silu(conv_out).unsqueeze(1)
        delta, B, C = self._scan_parameters(x_conv)

        y, ssm_state = selective_scan_recurrent(
            x_conv,
            delta,
            self.A_log,
            B,
            C,
            D=self.D,
            z=z,
            initial_state=inference_state.ssm_state,
            return_last_state=True,
        )
        output = self.out_proj(y)
        return output, MambaInferenceState(conv_state=new_conv_state, ssm_state=ssm_state)


class TinyMambaModel(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([TinyMambaBlock(config) for _ in range(config.num_layers)])
        self.norm = MambaRMSNorm(config.d_model, eps=config.rms_norm_eps)

    def initial_states(
        self,
        batch: int,
        device: t.device,
        dtype: t.dtype,
    ) -> tuple[MambaInferenceState, ...]:
        return tuple(layer.initial_state(batch, device, dtype) for layer in self.layers)

    def forward(
        self,
        input_ids: t.Tensor,
        *,
        states: tuple[MambaInferenceState, ...] | None = None,
        use_cache: bool = False,
    ) -> tuple[t.Tensor, tuple[MambaInferenceState, ...] | None]:
        hidden_states = self.embed_tokens(input_ids)
        next_states = [] if use_cache else None
        if states is None and input_ids.shape[1] == 1 and use_cache:
            states = self.initial_states(input_ids.shape[0], input_ids.device, hidden_states.dtype)

        for idx, layer in enumerate(self.layers):
            layer_state = None if states is None else states[idx]
            hidden_states, next_state = layer(
                hidden_states,
                inference_state=layer_state,
                use_cache=use_cache,
            )
            if use_cache:
                assert next_states is not None and next_state is not None
                next_states.append(next_state)
        return self.norm(hidden_states), None if next_states is None else tuple(next_states)


class TinyMambaForCausalLM(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        self.backbone = TinyMambaModel(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.backbone.embed_tokens.weight

    def forward(
        self,
        input_ids: t.Tensor,
        *,
        states: tuple[MambaInferenceState, ...] | None = None,
        use_cache: bool = False,
    ) -> MambaCausalLMOutput:
        hidden_states, next_states = self.backbone(input_ids, states=states, use_cache=use_cache)
        return MambaCausalLMOutput(logits=self.lm_head(hidden_states), states=next_states)

    @t.no_grad()
    def greedy_generate(self, input_ids: t.Tensor, max_new_tokens: int) -> t.Tensor:
        generated = input_ids
        states = None
        next_input = input_ids
        for _ in range(max_new_tokens):
            output = self(next_input, states=states, use_cache=True)
            states = output.states
            next_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
            generated = t.cat([generated, next_token], dim=-1)
            next_input = next_token
        return generated


def mamba_cache_parity_report(
    model: TinyMambaForCausalLM,
    input_ids: t.Tensor,
    *,
    atol: float = 1e-4,
) -> dict[str, Any]:
    """Compare full-sequence logits with one-token recurrent-state logits."""

    model.eval()
    with t.no_grad():
        full_logits = model(input_ids, use_cache=False).logits
        states = None
        cached_logits = []
        for pos in range(input_ids.shape[1]):
            output = model(input_ids[:, pos : pos + 1], states=states, use_cache=True)
            states = output.states
            cached_logits.append(output.logits)
        cached_logits_tensor = t.cat(cached_logits, dim=1)
        diff = (full_logits - cached_logits_tensor).abs()
    return {
        "max_abs_diff": diff.max().item(),
        "mse": diff.float().pow(2).mean().item(),
        "passed": bool(diff.max().item() <= atol),
        "atol": atol,
    }


def _transformers_mamba_fast_path_status() -> tuple[bool, dict[str, bool]]:
    """Check whether Transformers can see the compiled Mamba fast-path kernels."""

    from transformers.models.mamba import modeling_mamba
    from transformers.models.mamba.configuration_mamba import (
        MambaConfig as TransformersMambaConfig,
    )

    components: dict[str, bool] = {}
    try:
        tiny_config = TransformersMambaConfig(
            vocab_size=8,
            hidden_size=16,
            intermediate_size=32,
            state_size=4,
            conv_kernel=3,
            num_hidden_layers=1,
        )
        modeling_mamba.MambaMixer(
            tiny_config,
            layer_idx=0,
            initialize_mixer_weights=False,
        )
        components["transformers_mamba_mixer_constructed"] = True
    except Exception:
        components["transformers_mamba_mixer_constructed"] = False

    component_names = (
        "selective_state_update",
        "selective_scan_fn",
        "mamba_inner_fn",
        "causal_conv1d_fn",
        "causal_conv1d_update",
    )
    components.update(
        {name: getattr(modeling_mamba, name, None) is not None for name in component_names}
    )
    return all(components.values()), components


# %%
def make_scan_inputs(seed: int = 0):
    t.manual_seed(seed)
    batch, seq, d_inner, d_state = 2, 7, 4, 3
    u = t.randn(batch, seq, d_inner)
    delta = t.rand(batch, seq, d_inner) + 0.1
    A_log = t.randn(d_inner, d_state) - 2.0
    B = t.randn(batch, seq, d_state)
    C = t.randn(batch, seq, d_state)
    D = t.randn(d_inner)
    z = t.randn(batch, seq, d_inner)
    return u, delta, A_log, B, C, D, z


def make_tiny_mamba_config() -> MambaConfig:
    return MambaConfig(
        vocab_size=29,
        d_model=12,
        d_inner=16,
        d_state=4,
        d_conv=3,
        dt_rank=4,
        num_layers=2,
    )


def selective_scan_equivalence_smoke_test() -> dict:
    u, delta, A_log, B, C, D, z = make_scan_inputs()
    recurrent = selective_scan_recurrent(u, delta, A_log, B, C, D=D, z=z)
    parallel = selective_scan_parallel(u, delta, A_log, B, C, D=D, z=z)
    report = scan_equivalence_report(recurrent, parallel, atol=1e-5)
    return report.__dict__


def chunked_scan_equivalence_smoke_test() -> dict:
    u, delta, A_log, B, C, D, z = make_scan_inputs()
    full = selective_scan_recurrent(u, delta, A_log, B, C, D=D, z=z)
    chunked = selective_scan_chunked(u, delta, A_log, B, C, D=D, z=z, chunk_size=3)
    report = scan_equivalence_report(full, chunked, atol=1e-6)
    return report.__dict__


def block_step_equivalence_smoke_test() -> dict:
    t.manual_seed(0)
    block = TinyMambaBlock(make_tiny_mamba_config())
    hidden = t.randn(2, 5, block.config.d_model)
    full, _ = block(hidden, use_cache=False)
    state = block.initial_state(batch=2, device=hidden.device, dtype=hidden.dtype)
    step_outputs = []
    for pos in range(hidden.shape[1]):
        out, state = block.step(hidden[:, pos : pos + 1], state)
        step_outputs.append(out)
    stepped = t.cat(step_outputs, dim=1)
    diff = (full - stepped).abs()
    return {"max_abs_diff": diff.max().item(), "passed": bool(diff.max().item() <= 1e-5)}


def tiny_lm_cache_parity_smoke_test() -> dict:
    t.manual_seed(1)
    model = TinyMambaForCausalLM(make_tiny_mamba_config())
    input_ids = t.tensor([[1, 2, 3, 4, 5]])
    return mamba_cache_parity_report(model, input_ids, atol=1e-5)


def generation_shape_smoke_test() -> tuple[int, ...]:
    t.manual_seed(2)
    model = TinyMambaForCausalLM(make_tiny_mamba_config())
    input_ids = t.tensor([[1, 2, 3]])
    return tuple(model.greedy_generate(input_ids, max_new_tokens=4).shape)


def run_smoke_test(cpu: bool = True) -> dict:
    _ = cpu
    return {
        "selective_scan_equivalence": selective_scan_equivalence_smoke_test(),
        "chunked_scan_equivalence": chunked_scan_equivalence_smoke_test(),
        "block_step_equivalence": block_step_equivalence_smoke_test(),
        "lm_cache_parity": tiny_lm_cache_parity_smoke_test(),
        "generation_shape": generation_shape_smoke_test(),
    }


def official_mamba_logits_generation_preflight(max_vram_gb: float = 24.0) -> dict:
    """Load a pinned official Mamba checkpoint and verify logits and generation."""

    if not t.cuda.is_available():
        return {
            "cuda_available": False,
            "preflight_passed": False,
            "skipped": True,
            "claim_scope": "official_mamba_logits_generation_preflight_requires_cuda",
        }

    from transformers import AutoModelForCausalLM, AutoTokenizer

    fast_kernel_available, fast_path_components = _transformers_mamba_fast_path_status()
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
    ).to(device)
    model.eval()

    prompts = ["Mamba models can", "The cat sat on"]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    with t.inference_mode():
        batched_logits = model(**inputs).logits.detach().float()

    agreements = []
    for index, prompt in enumerate(prompts):
        single_inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with t.inference_mode():
            single_logits = model(**single_inputs).logits.detach().float()[0]
        batched_prompt_logits = batched_logits[index, inputs.attention_mask[index].bool()]
        agreements.append(
            batched_prompt_logits.argmax(dim=-1)
            .eq(single_logits.argmax(dim=-1))
            .float()
            .mean()
            .item()
        )

    generation_inputs = tokenizer("Mamba models can", return_tensors="pt").to(device)
    t.cuda.synchronize()
    generation_start = time.perf_counter()
    with t.inference_mode():
        generated = model.generate(
            **generation_inputs,
            max_new_tokens=4,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    t.cuda.synchronize()
    generation_seconds = time.perf_counter() - generation_start

    prompt_length = generation_inputs.input_ids.shape[1]
    generated_new_token_count = generated.shape[1] - prompt_length
    generation_tokens_per_second = generated_new_token_count / max(generation_seconds, 1e-9)
    prompt_prefix_preserved = bool(
        t.equal(generated[:, :prompt_length], generation_inputs.input_ids)
    )
    finite_logits = bool(t.isfinite(batched_logits).all().item())
    logits_std = batched_logits.std().item()
    batched_single_top1_agreement = min(agreements)
    logits_shape = list(batched_logits.shape)
    generated_shape = list(generated.shape)
    t.cuda.synchronize()
    peak_vram_gb = t.cuda.max_memory_allocated() / 1024**3

    del generated, generation_inputs, batched_logits, inputs, model, tokenizer
    t.cuda.empty_cache()

    preflight_passed = (
        finite_logits
        and logits_std > 1.0
        and batched_single_top1_agreement == 1.0
        and generated_new_token_count == 4
        and generation_tokens_per_second > 1.0
        and prompt_prefix_preserved
        and peak_vram_gb <= max_vram_gb
        and fast_kernel_available
    )
    return {
        "cuda_available": True,
        "model_id": REAL_MAMBA_MODEL_ID,
        "revision": REAL_MAMBA_REVISION,
        "claim_scope": "official_mamba_130m_hf_logits_generation_preflight",
        "prompt_count": len(prompts),
        "logits_shape": logits_shape,
        "finite_logits": finite_logits,
        "logits_std": logits_std,
        "batched_single_top1_agreement": batched_single_top1_agreement,
        "generated_shape": generated_shape,
        "generated_new_token_count": generated_new_token_count,
        "generation_tokens_per_second": generation_tokens_per_second,
        "prompt_prefix_preserved": prompt_prefix_preserved,
        "fast_kernel_available": fast_kernel_available,
        "fast_path_components": fast_path_components,
        "peak_vram_gb": peak_vram_gb,
        "within_vram_budget": peak_vram_gb <= max_vram_gb,
        "preflight_passed": preflight_passed,
    }


def run_gpu_test(max_vram_gb: float = 24.0) -> dict:
    if not t.cuda.is_available():
        return {
            "cuda_available": False,
            "full_path": "Load a small official Mamba checkpoint and compare logits/generation.",
        }

    device = t.device("cuda")
    t.cuda.reset_peak_memory_stats()
    u, delta, A_log, B, C, D, z = [item.to(device) for item in make_scan_inputs()]
    recurrent = selective_scan_recurrent(u, delta, A_log, B, C, D=D, z=z)
    parallel = selective_scan_parallel(u, delta, A_log, B, C, D=D, z=z)
    scan_report = scan_equivalence_report(recurrent, parallel, atol=1e-5)

    t.manual_seed(1)
    model = TinyMambaForCausalLM(make_tiny_mamba_config()).to(device)
    input_ids = t.tensor([[1, 2, 3, 4, 5]], device=device)
    cache_report = mamba_cache_parity_report(model, input_ids, atol=1e-5)
    t.cuda.synchronize()
    synthetic_peak_vram_gb = t.cuda.max_memory_allocated() / 1024**3
    official_mamba = official_mamba_logits_generation_preflight(max_vram_gb=max_vram_gb)
    peak_vram_gb = max(synthetic_peak_vram_gb, official_mamba["peak_vram_gb"])
    return {
        "cuda_available": True,
        "device": t.cuda.get_device_name(0),
        "scan_max_abs_diff": scan_report.max_abs_diff,
        "scan_passed": scan_report.passed,
        "cache_max_abs_diff": cache_report["max_abs_diff"],
        "cache_passed": cache_report["passed"],
        "official_mamba_logits_generation_preflight_passed": official_mamba[
            "preflight_passed"
        ],
        "official_mamba_batched_single_top1_agreement": official_mamba[
            "batched_single_top1_agreement"
        ],
        "official_mamba_generation_new_tokens": official_mamba[
            "generated_new_token_count"
        ],
        "official_mamba_generation_tokens_per_second": official_mamba[
            "generation_tokens_per_second"
        ],
        "official_mamba_prompt_prefix_preserved": official_mamba[
            "prompt_prefix_preserved"
        ],
        "official_mamba_logits_std": official_mamba["logits_std"],
        "official_mamba_logits_shape": official_mamba["logits_shape"],
        "official_mamba_generated_shape": official_mamba["generated_shape"],
        "official_mamba_fast_kernel_available": official_mamba[
            "fast_kernel_available"
        ],
        "official_mamba_peak_vram_gb": official_mamba["peak_vram_gb"],
        "official_mamba_preflight": official_mamba,
        "peak_vram_gb": peak_vram_gb,
        "within_vram_budget": (
            peak_vram_gb <= max_vram_gb and official_mamba["within_vram_budget"]
        ),
        "full_path": (
            "Validated selective-scan/cache parity plus pinned Mamba-130M-HF "
            "logits and deterministic generation on CUDA fast kernels."
        ),
    }


def run_full_experiment(max_vram_gb: float = 24.0) -> dict:
    """Run the validated experiment path used by the verification report."""

    return run_gpu_test(max_vram_gb=max_vram_gb)


if MAIN:
    print(run_smoke_test())
