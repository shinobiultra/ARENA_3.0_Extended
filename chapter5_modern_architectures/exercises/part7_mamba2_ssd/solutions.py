"""Reference implementation for ARENA section 5.7: Mamba-2 and SSD.

The educational implementations deliberately keep the recurrent, quadratic,
and chunked paths separate. Their agreement is the central correctness result.
"""

from __future__ import annotations

import gc
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch as t
import torch.nn as nn
import torch.nn.functional as F


EXERCISE_ID = "5_7_mamba2_ssd"
GT_TIER = "GT-1"
REAL_MAMBA2_MODEL_ID = "state-spaces/mamba2-130m"
REAL_MAMBA2_REVISION = "3a5aea0c25d0fb43cc360e2c2aac82c26e3eed49"
REAL_MAMBA2_PROMPT_TOKEN_IDS = (
    (46, 31834, 3210, 476),
    (510, 5798, 2206, 327),
)


def segment_sums(log_decay: t.Tensor) -> t.Tensor:
    """Return S[t, s] = sum(log_decay[s + 1 : t + 1]) for t >= s.

    Unlike a difference of cumulative sums, this construction does not
    subtract nearby large numbers. Entries above the diagonal are -inf so
    exponentiation produces a causal lower-triangular decay matrix.
    """

    if log_decay.ndim < 1:
        raise ValueError("log_decay must have a sequence axis")
    length = log_decay.shape[-1]
    repeated = log_decay.unsqueeze(-1).expand(*log_decay.shape, length)
    strict_lower = t.tril(
        t.ones(length, length, dtype=t.bool, device=log_decay.device),
        diagonal=-1,
    )
    contributions = repeated.masked_fill(~strict_lower, 0)
    sums = contributions.cumsum(dim=-2)
    causal = t.tril(
        t.ones(length, length, dtype=t.bool, device=log_decay.device)
    )
    return sums.masked_fill(~causal, -t.inf)


def expand_grouped_parameters(
    B: t.Tensor,
    C: t.Tensor,
    n_heads: int,
) -> tuple[t.Tensor, t.Tensor]:
    """Repeat each parameter group across its contiguous block of heads."""

    if B.shape != C.shape or B.ndim != 4:
        raise ValueError("B and C must both have shape [batch, length, groups, state]")
    n_groups = B.shape[2]
    if n_heads % n_groups != 0:
        raise ValueError("n_heads must be divisible by the number of groups")
    repeats = n_heads // n_groups
    return B.repeat_interleave(repeats, dim=2), C.repeat_interleave(repeats, dim=2)


def semiseparable_matrix(
    log_decay: t.Tensor,
    B: t.Tensor,
    C: t.Tensor,
) -> t.Tensor:
    """Materialize the causal SSD mixing matrix for every head.

    Shapes are log_decay [B, L, H], B/C [B, L, G, N], and output
    [B, H, L, L]. The matrix entry is C_t^T B_s times the product of
    transitions strictly after source s and through destination t.
    """

    if log_decay.ndim != 3:
        raise ValueError("log_decay must have shape [batch, length, heads]")
    B_heads, C_heads = expand_grouped_parameters(B, C, log_decay.shape[2])
    decay = segment_sums(log_decay.transpose(1, 2)).exp()
    content = t.einsum("bthn,bshn->bhts", C_heads, B_heads)
    return decay * content


def ssd_recurrent(
    X: t.Tensor,
    log_decay: t.Tensor,
    B: t.Tensor,
    C: t.Tensor,
    initial_state: t.Tensor | None = None,
    *,
    carry_scale: float = 1.0,
) -> tuple[t.Tensor, t.Tensor]:
    """Evaluate SSD as the literal recurrent state update.

    X has shape [B, L, H, P]. The state has shape [B, H, P, N]. Setting
    carry_scale to zero is a causal state-ablation control, not a different
    task: each token can still write and read its current contribution.
    """

    if X.ndim != 4 or log_decay.shape != X.shape[:3]:
        raise ValueError("X must be [batch, length, heads, head_dim]")
    B_heads, C_heads = expand_grouped_parameters(B, C, X.shape[2])
    expected_state = (X.shape[0], X.shape[2], X.shape[3], B.shape[-1])
    state = (
        X.new_zeros(expected_state)
        if initial_state is None
        else initial_state.to(device=X.device, dtype=X.dtype)
    )
    if state.shape != expected_state:
        raise ValueError(f"initial_state must have shape {expected_state}")
    outputs = []
    for position in range(X.shape[1]):
        transition = log_decay[:, position].exp()[..., None, None]
        write = t.einsum(
            "bhp,bhn->bhpn",
            X[:, position],
            B_heads[:, position],
        )
        state = float(carry_scale) * transition * state + write
        outputs.append(
            t.einsum("bhn,bhpn->bhp", C_heads[:, position], state)
        )
    return t.stack(outputs, dim=1), state


def ssd_quadratic(
    X: t.Tensor,
    log_decay: t.Tensor,
    B: t.Tensor,
    C: t.Tensor,
) -> t.Tensor:
    """Evaluate SSD through its full semiseparable attention-dual matrix."""

    mixing = semiseparable_matrix(log_decay, B, C)
    return t.einsum("bhts,bshp->bthp", mixing, X)


def ssd_chunked(
    X: t.Tensor,
    log_decay: t.Tensor,
    B: t.Tensor,
    C: t.Tensor,
    chunk_size: int,
    initial_state: t.Tensor | None = None,
) -> tuple[t.Tensor, t.Tensor]:
    """Evaluate SSD using diagonal blocks plus low-rank boundary states.

    The only recurrent loop is over chunks. Work inside each chunk is a
    quadratic tensor contraction, exposing the semiseparable block structure
    used by the hardware-aware Mamba-2 algorithm.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if X.ndim != 4 or log_decay.shape != X.shape[:3]:
        raise ValueError("X must be [batch, length, heads, head_dim]")
    batch, original_length, n_heads, head_dim = X.shape
    B_heads, C_heads = expand_grouped_parameters(B, C, n_heads)
    d_state = B_heads.shape[-1]

    pad = (-original_length) % chunk_size
    if pad:
        X = F.pad(X, (0, 0, 0, 0, 0, pad))
        log_decay = F.pad(log_decay, (0, 0, 0, pad))
        B_heads = F.pad(B_heads, (0, 0, 0, 0, 0, pad))
        C_heads = F.pad(C_heads, (0, 0, 0, 0, 0, pad))

    length = X.shape[1]
    n_chunks = length // chunk_size
    Xc = X.reshape(batch, n_chunks, chunk_size, n_heads, head_dim)
    Bc = B_heads.reshape(batch, n_chunks, chunk_size, n_heads, d_state)
    Cc = C_heads.reshape(batch, n_chunks, chunk_size, n_heads, d_state)
    Ac = log_decay.reshape(batch, n_chunks, chunk_size, n_heads).permute(0, 3, 1, 2)
    A_prefix = Ac.cumsum(dim=-1)

    # Diagonal blocks: interactions whose source and destination share a chunk.
    intra_decay = segment_sums(Ac).exp()
    Y_diag = t.einsum(
        "bcqhn,bckhn,bhcqk,bckhp->bcqhp",
        Cc,
        Bc,
        intra_decay,
        Xc,
    )

    # Each chunk compresses all of its writes into one outgoing boundary state.
    decay_to_end = (A_prefix[..., -1:] - A_prefix).exp()
    chunk_writes = t.einsum(
        "bckhn,bhck,bckhp->bchpn",
        Bc,
        decay_to_end,
        Xc,
    )

    expected_state = (batch, n_heads, head_dim, d_state)
    state = (
        X.new_zeros(expected_state)
        if initial_state is None
        else initial_state.to(device=X.device, dtype=X.dtype)
    )
    if state.shape != expected_state:
        raise ValueError(f"initial_state must have shape {expected_state}")
    incoming_states = []
    for chunk in range(n_chunks):
        incoming_states.append(state)
        chunk_transition = A_prefix[:, :, chunk, -1].exp()[..., None, None]
        state = chunk_transition * state + chunk_writes[:, chunk]
    incoming = t.stack(incoming_states, dim=1)

    # Off-diagonal blocks: expand each incoming boundary state through its chunk.
    state_decay_out = A_prefix.exp()
    Y_off = t.einsum(
        "bcqhn,bchpn,bhcq->bcqhp",
        Cc,
        incoming,
        state_decay_out,
    )
    output = (Y_diag + Y_off).reshape(batch, length, n_heads, head_dim)
    return output[:, :original_length], state


def causal_depthwise_conv1d(
    X: t.Tensor,
    weight: t.Tensor,
    bias: t.Tensor | None = None,
) -> t.Tensor:
    """Causal depthwise cross-correlation for [batch, length, channels]."""

    if X.ndim != 3 or weight.ndim != 2 or weight.shape[0] != X.shape[-1]:
        raise ValueError("X must be [B,L,C] and weight must be [C,kernel]")
    channels, kernel_size = weight.shape
    padded = F.pad(X.transpose(1, 2), (kernel_size - 1, 0))
    return F.conv1d(
        padded,
        weight[:, None],
        bias=bias,
        groups=channels,
    ).transpose(1, 2)


def rms_norm_gated(
    Y: t.Tensor,
    gate: t.Tensor,
    weight: t.Tensor,
    eps: float = 1e-5,
) -> t.Tensor:
    """RMS-normalize the SSD output, then apply a SiLU gate."""

    normalized = Y * t.rsqrt(Y.square().mean(dim=-1, keepdim=True) + eps)
    return normalized * weight * F.silu(gate)


@dataclass(frozen=True)
class Mamba2Config:
    d_model: int = 12
    d_state: int = 4
    d_conv: int = 3
    expand: int = 1
    head_dim: int = 3
    n_groups: int = 1
    chunk_size: int = 8

    @property
    def d_inner(self) -> int:
        return self.expand * self.d_model

    @property
    def n_heads(self) -> int:
        if self.d_inner % self.head_dim:
            raise ValueError("expand*d_model must be divisible by head_dim")
        return self.d_inner // self.head_dim


@dataclass
class Mamba2Cache:
    conv_state: t.Tensor
    ssm_state: t.Tensor


class Mamba2BlockFromScratch(nn.Module):
    """A small, readable Mamba-2 residual block built around chunked SSD."""

    def __init__(self, config: Mamba2Config):
        super().__init__()
        self.config = config
        h, p, g, n = (
            config.n_heads,
            config.head_dim,
            config.n_groups,
            config.d_state,
        )
        if h % g:
            raise ValueError("n_heads must be divisible by n_groups")
        projection_size = 2 * config.d_inner + 2 * g * n + h
        self.in_proj = nn.Linear(config.d_model, projection_size, bias=False)
        conv_channels = config.d_inner + 2 * g * n
        self.conv1d = nn.Conv1d(
            conv_channels,
            conv_channels,
            config.d_conv,
            groups=conv_channels,
        )
        self.dt_bias = nn.Parameter(t.full((h,), -2.5))
        self.A_log = nn.Parameter(t.linspace(0.0, math.log(4.0), h))
        self.D = nn.Parameter(t.ones(h))
        self.norm_weight = nn.Parameter(t.ones(config.d_inner))
        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=False)

    def _project(self, U: t.Tensor) -> tuple[t.Tensor, t.Tensor, t.Tensor]:
        c = self.config
        projected = self.in_proj(U)
        return t.split(
            projected,
            [c.d_inner, c.d_inner + 2 * c.n_groups * c.d_state, c.n_heads],
            dim=-1,
        )

    def _mix_from_convolved(
        self,
        U: t.Tensor,
        Z: t.Tensor,
        XBC: t.Tensor,
        dt_logits: t.Tensor,
        *,
        carry_scale: float = 1.0,
    ) -> tuple[t.Tensor, t.Tensor]:
        c = self.config
        X, B, C = t.split(
            XBC,
            [c.d_inner, c.n_groups * c.d_state, c.n_groups * c.d_state],
            dim=-1,
        )
        X = X.reshape(*X.shape[:2], c.n_heads, c.head_dim)
        B = B.reshape(*B.shape[:2], c.n_groups, c.d_state)
        C = C.reshape(*C.shape[:2], c.n_groups, c.d_state)
        dt = F.softplus(dt_logits + self.dt_bias)
        log_decay = dt * (-self.A_log.exp())
        if carry_scale == 1.0:
            Y, final_state = ssd_chunked(
                X * dt[..., None],
                log_decay,
                B,
                C,
                c.chunk_size,
            )
        else:
            Y, final_state = ssd_recurrent(
                X * dt[..., None],
                log_decay,
                B,
                C,
                carry_scale=carry_scale,
            )
        Y = Y + self.D[None, None, :, None] * X
        Y = Y.reshape(*Y.shape[:2], c.d_inner)
        mixed = rms_norm_gated(Y, Z, self.norm_weight)
        return U + self.out_proj(mixed), final_state

    def forward(
        self,
        U: t.Tensor,
        *,
        carry_scale: float = 1.0,
    ) -> t.Tensor:
        if U.ndim != 3 or U.shape[-1] != self.config.d_model:
            raise ValueError("U must have shape [batch, length, d_model]")
        Z, XBC, dt_logits = self._project(U)
        conv_weight = self.conv1d.weight[:, 0]
        XBC = F.silu(causal_depthwise_conv1d(XBC, conv_weight, self.conv1d.bias))
        output, _ = self._mix_from_convolved(
            U,
            Z,
            XBC,
            dt_logits,
            carry_scale=carry_scale,
        )
        return output

    def allocate_cache(
        self,
        batch_size: int,
        *,
        device: t.device | str,
        dtype: t.dtype,
    ) -> Mamba2Cache:
        c = self.config
        conv_channels = c.d_inner + 2 * c.n_groups * c.d_state
        return Mamba2Cache(
            conv_state=t.zeros(
                batch_size,
                conv_channels,
                c.d_conv,
                device=device,
                dtype=dtype,
            ),
            ssm_state=t.zeros(
                batch_size,
                c.n_heads,
                c.head_dim,
                c.d_state,
                device=device,
                dtype=dtype,
            ),
        )

    def step(
        self,
        U: t.Tensor,
        cache: Mamba2Cache,
        *,
        carry_scale: float = 1.0,
    ) -> tuple[t.Tensor, Mamba2Cache]:
        """Decode one token while updating convolution and SSM caches."""

        if U.ndim == 2:
            U = U[:, None]
        if U.ndim != 3 or U.shape[1] != 1:
            raise ValueError("step expects [batch, d_model] or [batch, 1, d_model]")
        c = self.config
        Z, XBC_raw, dt_logits = self._project(U)
        conv_state = t.cat(
            [cache.conv_state[..., 1:], XBC_raw[:, 0, :, None]],
            dim=-1,
        )
        conv_value = (
            conv_state * self.conv1d.weight[:, 0][None]
        ).sum(dim=-1) + self.conv1d.bias
        XBC = F.silu(conv_value[:, None])
        X, B, C = t.split(
            XBC,
            [c.d_inner, c.n_groups * c.d_state, c.n_groups * c.d_state],
            dim=-1,
        )
        X = X.reshape(U.shape[0], c.n_heads, c.head_dim)
        B = B.reshape(U.shape[0], c.n_groups, c.d_state)
        C = C.reshape(U.shape[0], c.n_groups, c.d_state)
        B_heads, C_heads = expand_grouped_parameters(B[:, None], C[:, None], c.n_heads)
        B_heads, C_heads = B_heads[:, 0], C_heads[:, 0]
        dt = F.softplus(dt_logits[:, 0] + self.dt_bias)
        transition = (dt * (-self.A_log.exp())).exp()[..., None, None]
        write = t.einsum("bhp,bhn->bhpn", X * dt[..., None], B_heads)
        ssm_state = float(carry_scale) * transition * cache.ssm_state + write
        Y = t.einsum("bhn,bhpn->bhp", C_heads, ssm_state)
        Y = Y + self.D[None, :, None] * X
        Y = Y.reshape(U.shape[0], 1, c.d_inner)
        mixed = rms_norm_gated(Y, Z, self.norm_weight)
        output = U + self.out_proj(mixed)
        return output, Mamba2Cache(conv_state=conv_state, ssm_state=ssm_state)


class TinyMamba2RecallModel(nn.Module):
    """A compact Mamba-2 mixer for the delayed-recall training organism."""

    def __init__(self, config: Mamba2Config = Mamba2Config()):
        super().__init__()
        self.config = config
        self.n_heads = 2
        self.head_dim = 2
        self.d_inner = self.n_heads * self.head_dim
        self.d_state = 2
        self.n_groups = 1
        projection_size = 2 * self.d_inner + 2 * self.d_state + self.n_heads
        self.in_proj = nn.Linear(4, projection_size, bias=False)
        conv_channels = self.d_inner + 2 * self.d_state
        self.conv1d = nn.Conv1d(
            conv_channels,
            conv_channels,
            config.d_conv,
            groups=conv_channels,
        )
        self.dt_bias = nn.Parameter(t.full((self.n_heads,), -4.0))
        self.A_log = nn.Parameter(t.log(t.tensor([0.1, 0.3])))
        self.D = nn.Parameter(t.ones(self.n_heads))
        self.norm_weight = nn.Parameter(t.ones(self.d_inner))
        self.readout = nn.Linear(self.d_inner, 2)

    def forward(self, inputs: t.Tensor, *, carry_scale: float = 1.0) -> t.Tensor:
        projected = self.in_proj(inputs)
        Z, XBC, dt_logits = t.split(
            projected,
            [self.d_inner, self.d_inner + 2 * self.d_state, self.n_heads],
            dim=-1,
        )
        XBC = F.silu(
            causal_depthwise_conv1d(
                XBC,
                self.conv1d.weight[:, 0],
                self.conv1d.bias,
            )
        )
        X, B, C = t.split(
            XBC,
            [self.d_inner, self.d_state, self.d_state],
            dim=-1,
        )
        X = X.reshape(*X.shape[:2], self.n_heads, self.head_dim)
        B = B.reshape(*B.shape[:2], self.n_groups, self.d_state)
        C = C.reshape(*C.shape[:2], self.n_groups, self.d_state)
        dt = F.softplus(dt_logits + self.dt_bias)
        log_decay = dt * (-self.A_log.exp())
        if carry_scale == 1.0:
            Y, _ = ssd_chunked(
                X * dt[..., None],
                log_decay,
                B,
                C,
                self.config.chunk_size,
            )
        else:
            Y, _ = ssd_recurrent(
                X * dt[..., None],
                log_decay,
                B,
                C,
                carry_scale=carry_scale,
            )
        Y = Y + self.D[None, None, :, None] * X
        Y = rms_norm_gated(
            Y.flatten(2),
            Z,
            self.norm_weight,
        )
        return self.readout(Y[:, -1])


class LocalConvRecallModel(nn.Module):
    """A matched-width one-layer local baseline with no recurrent state."""

    def __init__(self, d_model: int = 12, kernel_size: int = 3):
        super().__init__()
        self.kernel_size = kernel_size
        self.input_proj = nn.Linear(4, d_model)
        self.conv = nn.Conv1d(d_model, d_model, kernel_size)
        self.readout = nn.Linear(d_model, 2)

    def forward(self, inputs: t.Tensor) -> t.Tensor:
        hidden = F.silu(self.input_proj(inputs))
        hidden = F.silu(
            F.conv1d(
                F.pad(hidden.transpose(1, 2), (self.kernel_size - 1, 0)),
                self.conv.weight,
                self.conv.bias,
            ).transpose(1, 2)
        )
        return self.readout(hidden[:, -1])


def make_delayed_recall_batch(
    batch_size: int,
    sequence_length: int,
    *,
    generator: t.Generator,
    device: t.device | str = "cpu",
    shuffled_labels: bool = False,
) -> tuple[t.Tensor, t.Tensor]:
    """Generate WRITE, distractor, READ sequences with exact binary labels."""

    if sequence_length < 8:
        raise ValueError("sequence_length must leave a real gap between WRITE and READ")
    labels = t.randint(0, 2, (batch_size,), generator=generator)
    inputs = 0.08 * t.randn(batch_size, sequence_length, 4, generator=generator)
    inputs[:, 0, 0] = labels.float().mul(2).sub(1)
    inputs[:, 0, 1] = 1.0
    inputs[:, -1, 2] = 1.0
    inputs[:, :, 3] = t.linspace(-1, 1, sequence_length)
    if shuffled_labels:
        labels = t.randint(0, 2, (batch_size,), generator=generator)
    return inputs.to(device), labels.to(device)


def train_recall_model(
    model: nn.Module,
    *,
    steps: int,
    sequence_length: int = 32,
    batch_size: int = 48,
    learning_rate: float = 1e-2,
    seed: int = 0,
    shuffled_labels: bool = False,
    device: t.device | str = "cpu",
) -> list[float]:
    """Train a sequence model on the generated delayed-recall distribution."""

    t.manual_seed(seed)
    model.to(device).train()
    optimizer = t.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    generator = t.Generator().manual_seed(seed + 1)
    losses = []
    for _ in range(steps):
        inputs, labels = make_delayed_recall_batch(
            batch_size,
            sequence_length,
            generator=generator,
            device=device,
            shuffled_labels=shuffled_labels,
        )
        logits = model(inputs)
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        t.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    return losses


@t.inference_mode()
def evaluate_recall_model(
    model: nn.Module,
    *,
    sequence_length: int = 32,
    examples: int = 1024,
    seed: int = 100,
    carry_scale: float | None = None,
    device: t.device | str = "cpu",
) -> float:
    model.to(device).eval()
    generator = t.Generator().manual_seed(seed)
    inputs, labels = make_delayed_recall_batch(
        examples,
        sequence_length,
        generator=generator,
        device=device,
    )
    logits = (
        model(inputs, carry_scale=carry_scale)
        if carry_scale is not None
        else model(inputs)
    )
    return float(logits.argmax(dim=-1).eq(labels).float().mean())


@dataclass
class RecallExperiment:
    mamba_accuracy: float
    state_ablated_accuracy: float
    local_conv_accuracy: float
    shuffled_label_accuracy: float
    ood_length_accuracy: float
    mamba_losses: list[float]
    local_losses: list[float]
    shuffled_losses: list[float]


def run_delayed_recall_experiment(
    *,
    steps: int = 100,
    sequence_length: int = 32,
    device: t.device | str = "cpu",
) -> RecallExperiment:
    """Train the Mamba-2 model and all three scientific controls."""

    t.manual_seed(0)
    config = Mamba2Config()
    mamba = TinyMamba2RecallModel(config)
    local = LocalConvRecallModel(config.d_model, config.d_conv)
    shuffled = TinyMamba2RecallModel(config)
    mamba_losses = train_recall_model(
        mamba,
        steps=steps,
        sequence_length=sequence_length,
        seed=0,
        device=device,
    )
    control_steps = steps
    local_losses = train_recall_model(
        local,
        steps=control_steps,
        sequence_length=sequence_length,
        seed=1,
        device=device,
    )
    shuffled_losses = train_recall_model(
        shuffled,
        steps=control_steps,
        sequence_length=sequence_length,
        seed=2,
        shuffled_labels=True,
        device=device,
    )
    return RecallExperiment(
        mamba_accuracy=evaluate_recall_model(mamba, sequence_length=sequence_length, device=device),
        state_ablated_accuracy=evaluate_recall_model(
            mamba,
            sequence_length=sequence_length,
            carry_scale=0.0,
            device=device,
        ),
        local_conv_accuracy=evaluate_recall_model(local, sequence_length=sequence_length, device=device),
        shuffled_label_accuracy=evaluate_recall_model(shuffled, sequence_length=sequence_length, device=device),
        ood_length_accuracy=evaluate_recall_model(
            mamba,
            sequence_length=2 * sequence_length,
            seed=101,
            device=device,
        ),
        mamba_losses=mamba_losses,
        local_losses=local_losses,
        shuffled_losses=shuffled_losses,
    )


def make_ssd_inputs(
    *,
    length: int = 32,
    batch: int = 2,
    n_heads: int = 4,
    n_groups: int = 2,
    head_dim: int = 3,
    d_state: int = 5,
    seed: int = 0,
    dtype: t.dtype = t.float64,
    device: t.device | str = "cpu",
) -> tuple[t.Tensor, t.Tensor, t.Tensor, t.Tensor]:
    generator = t.Generator().manual_seed(seed)
    X = t.randn(batch, length, n_heads, head_dim, generator=generator, dtype=dtype)
    log_decay = -0.02 - 0.3 * t.rand(
        batch, length, n_heads, generator=generator, dtype=dtype
    )
    B = t.randn(batch, length, n_groups, d_state, generator=generator, dtype=dtype)
    C = t.randn(batch, length, n_groups, d_state, generator=generator, dtype=dtype)
    return tuple(value.to(device) for value in (X, log_decay, B, C))


@dataclass
class ScalingResult:
    lengths: list[int]
    recurrent_ms: list[float]
    quadratic_ms: list[float]
    chunked_ms: list[float]
    quadratic_matrix_mb: list[float]
    chunk_local_matrix_mb: list[float]


def _median_runtime_ms(
    function: Any,
    repeats: int,
    device: t.device | str,
) -> float:
    function()
    if t.device(device).type == "cuda":
        t.cuda.synchronize(t.device(device))
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        if t.device(device).type == "cuda":
            t.cuda.synchronize(t.device(device))
        samples.append((time.perf_counter() - start) * 1000)
    return float(t.tensor(samples).median())


def measure_ssd_scaling(
    lengths: Iterable[int] = (32, 64, 128, 256),
    *,
    chunk_size: int = 16,
    repeats: int = 3,
    device: t.device | str = "cpu",
) -> ScalingResult:
    """Measure wall time and exact sizes of the dominant decay matrices."""

    lengths = list(lengths)
    recurrent_ms, quadratic_ms, chunked_ms = [], [], []
    quadratic_mb, chunk_mb = [], []
    for length in lengths:
        X, A, B, C = make_ssd_inputs(
            length=length,
            batch=1,
            n_heads=2,
            n_groups=1,
            head_dim=4,
            d_state=4,
            dtype=t.float32,
            device=device,
        )
        with t.inference_mode():
            recurrent_ms.append(
                _median_runtime_ms(
                    lambda: ssd_recurrent(X, A, B, C), repeats, device
                )
            )
            quadratic_ms.append(
                _median_runtime_ms(
                    lambda: ssd_quadratic(X, A, B, C), repeats, device
                )
            )
            chunked_ms.append(
                _median_runtime_ms(
                    lambda: ssd_chunked(X, A, B, C, chunk_size),
                    repeats,
                    device,
                )
            )
        bytes_per = X.element_size()
        quadratic_mb.append(1 * 2 * length * length * bytes_per / 2**20)
        padded = math.ceil(length / chunk_size) * chunk_size
        chunk_mb.append(1 * 2 * padded * chunk_size * bytes_per / 2**20)
    return ScalingResult(
        lengths,
        recurrent_ms,
        quadratic_ms,
        chunked_ms,
        quadratic_mb,
        chunk_mb,
    )


def exact_parity_metrics(
    *,
    length: int = 48,
    chunk_size: int = 8,
    device: t.device | str = "cpu",
) -> dict[str, float]:
    X, A, B, C = make_ssd_inputs(length=length, device=device)
    recurrent, recurrent_state = ssd_recurrent(X, A, B, C)
    quadratic = ssd_quadratic(X, A, B, C)
    chunked, chunked_state = ssd_chunked(X, A, B, C, chunk_size)
    return {
        "recurrent_quadratic_max_abs_diff": float((recurrent - quadratic).abs().max()),
        "recurrent_chunked_max_abs_diff": float((recurrent - chunked).abs().max()),
        "final_state_max_abs_diff": float((recurrent_state - chunked_state).abs().max()),
    }


def official_ssd_reference_parity(
    *,
    length: int = 32,
    chunk_size: int = 8,
    device: t.device | str = "cpu",
) -> dict[str, float]:
    """Compare against Listing 1 in the installed official Mamba repository."""

    from mamba_ssm.modules.ssd_minimal import ssd_minimal_discrete

    X, A, B, C = make_ssd_inputs(
        length=length,
        batch=1,
        n_heads=4,
        n_groups=1,
        head_dim=3,
        d_state=5,
        device=device,
    )
    B_heads, C_heads = expand_grouped_parameters(B, C, X.shape[2])
    ours, ours_state = ssd_chunked(X, A, B_heads, C_heads, chunk_size)
    official, official_state = ssd_minimal_discrete(
        X,
        A,
        B_heads,
        C_heads,
        chunk_size,
    )
    return {
        "output_max_abs_diff": float((ours - official).abs().max()),
        "final_state_max_abs_diff": float((ours_state - official_state).abs().max()),
    }


def official_checkpoint_manifest() -> dict[str, Any]:
    """Read the pinned official Mamba-2 config without downloading its weights."""

    from huggingface_hub import hf_hub_download

    config_path = hf_hub_download(
        REAL_MAMBA2_MODEL_ID,
        "config.json",
        revision=REAL_MAMBA2_REVISION,
    )
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    return {
        "model_id": REAL_MAMBA2_MODEL_ID,
        "revision": REAL_MAMBA2_REVISION,
        "d_model": config["d_model"],
        "n_layer": config["n_layer"],
        "vocab_size": config["vocab_size"],
        "ssm_layer": config["ssm_cfg"]["layer"],
    }


def exact_gradient_parity_metrics(seed: int = 17, device: str | t.device = "cpu") -> dict[str, float]:
    """Compare gradients from the recurrent, quadratic, and chunked schedules."""

    base = make_ssd_inputs(length=12, seed=seed, device=device)
    variants = [tuple(value.clone().requires_grad_() for value in base) for _ in range(3)]
    recurrent = ssd_recurrent(*variants[0])[0]
    quadratic = ssd_quadratic(*variants[1])
    chunked = ssd_chunked(*variants[2], chunk_size=4)[0]
    gradients = [
        t.autograd.grad(output.square().mean(), inputs)
        for output, inputs in zip((recurrent, quadratic, chunked), variants, strict=True)
    ]

    def difference(left: tuple[t.Tensor, ...], right: tuple[t.Tensor, ...]) -> float:
        return max(float((a - b).abs().max().detach()) for a, b in zip(left, right, strict=True))

    return {
        "recurrent_quadratic_gradient_max_abs_diff": difference(gradients[0], gradients[1]),
        "recurrent_chunked_gradient_max_abs_diff": difference(gradients[0], gradients[2]),
    }


class TinyStackedMamba2LM(nn.Module):
    """A small token LM built from the same learner-authored cached blocks."""

    def __init__(self, vocab_size: int = 47, n_layers: int = 2, config: Mamba2Config | None = None):
        super().__init__()
        self.config = config or Mamba2Config()
        self.embedding = nn.Embedding(vocab_size, self.config.d_model)
        self.layers = nn.ModuleList(
            [Mamba2BlockFromScratch(self.config) for _ in range(n_layers)]
        )
        self.final_norm = nn.RMSNorm(self.config.d_model)
        self.lm_head = nn.Linear(self.config.d_model, vocab_size, bias=False)

    def forward(self, input_ids: t.Tensor) -> t.Tensor:
        hidden = self.embedding(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        return self.lm_head(self.final_norm(hidden))

    def forward_recurrent(self, input_ids: t.Tensor) -> tuple[t.Tensor, list[Mamba2Cache]]:
        batch = input_ids.shape[0]
        caches = [
            layer.allocate_cache(
                batch,
                device=input_ids.device,
                dtype=self.embedding.weight.dtype,
            )
            for layer in self.layers
        ]
        logits = []
        for position in range(input_ids.shape[1]):
            hidden = self.embedding(input_ids[:, position])
            next_caches = []
            for layer, cache in zip(self.layers, caches, strict=True):
                hidden, cache = layer.step(hidden, cache)
                next_caches.append(cache)
            caches = next_caches
            logits.append(self.lm_head(self.final_norm(hidden)))
        return t.cat(logits, dim=1), caches


class TinyCausalAttentionControl(nn.Module):
    """A real Transformer-style causal block near the learner Mamba-2 parameter count."""

    def __init__(self, d_model: int = 12, n_heads: int = 4):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("n_heads must divide d_model")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.norm = nn.RMSNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, inputs: t.Tensor) -> t.Tensor:
        batch, length, d_model = inputs.shape
        q, k, v = self.qkv(self.norm(inputs)).chunk(3, dim=-1)
        q, k, v = [
            value.reshape(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
            for value in (q, k, v)
        ]
        scores = t.einsum("bhtp,bhsp->bhts", q, k) / math.sqrt(self.head_dim)
        future = t.triu(t.ones(length, length, dtype=t.bool, device=inputs.device), diagonal=1)
        pattern = scores.masked_fill(future, -t.inf).softmax(dim=-1)
        values = t.einsum("bhts,bhsp->bhtp", pattern, v)
        values = values.transpose(1, 2).reshape(batch, length, d_model)
        return inputs + self.out_proj(values)


class TinyMamba1Control(nn.Module):
    """A runnable Mamba-1-style block with channel-dependent transitions."""

    def __init__(self, d_model: int = 12, d_state: int = 4, d_conv: int = 3):
        super().__init__()
        self.d_model, self.d_state = d_model, d_state
        self.norm = nn.RMSNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.conv1d = nn.Conv1d(d_model, d_model, d_conv, groups=d_model, bias=True)
        self.x_proj = nn.Linear(d_model, 1 + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(1, d_model, bias=True)
        self.A_log = nn.Parameter(t.arange(1, d_state + 1).float().log().repeat(d_model, 1))
        self.D = nn.Parameter(t.ones(d_model))
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, inputs: t.Tensor) -> t.Tensor:
        X, gate = self.in_proj(self.norm(inputs)).chunk(2, dim=-1)
        X = F.silu(causal_depthwise_conv1d(X, self.conv1d.weight[:, 0], self.conv1d.bias))
        raw_dt, B, C = self.x_proj(X).split([1, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(raw_dt))
        state = X.new_zeros(X.shape[0], self.d_model, self.d_state)
        outputs = []
        for position in range(X.shape[1]):
            decay = (-self.A_log.exp()[None] * dt[:, position, :, None]).exp()
            state = decay * state + X[:, position, :, None] * dt[:, position, :, None] * B[:, position, None]
            outputs.append((state * C[:, position, None]).sum(dim=-1))
        mixed = t.stack(outputs, dim=1) + self.D * X
        return inputs + self.out_proj(mixed * F.silu(gate))


def parameter_matched_control_counts() -> dict[str, int]:
    modules = {
        "Mamba-2": Mamba2BlockFromScratch(Mamba2Config()),
        "causal attention": TinyCausalAttentionControl(),
        "Mamba-1": TinyMamba1Control(),
    }
    return {
        name: sum(parameter.numel() for parameter in module.parameters())
        for name, module in modules.items()
    }


@dataclass(frozen=True)
class ReleasedMamba2Config:
    vocab_size: int = 50288
    d_model: int = 768
    n_layers: int = 24
    d_inner: int = 1536
    n_heads: int = 24
    head_dim: int = 64
    d_state: int = 128
    n_groups: int = 1
    d_conv: int = 4
    chunk_size: int = 8
    eps: float = 1e-5

    @property
    def conv_dim(self) -> int:
        return self.d_inner + 2 * self.n_groups * self.d_state

    @property
    def projection_size(self) -> int:
        return 2 * self.d_inner + 2 * self.n_groups * self.d_state + self.n_heads


class ReleasedRMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(t.ones(width))
        self.eps = eps

    def forward(self, inputs: t.Tensor) -> t.Tensor:
        dtype = inputs.dtype
        values = inputs.float()
        values = values * t.rsqrt(values.square().mean(-1, keepdim=True) + self.eps)
        return self.weight * values.to(dtype)


class ReleasedGatedRMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(t.ones(width))
        self.eps = eps

    def forward(self, inputs: t.Tensor, gate: t.Tensor) -> t.Tensor:
        dtype = inputs.dtype
        values = inputs.float() * F.silu(gate.float())
        values = values * t.rsqrt(values.square().mean(-1, keepdim=True) + self.eps)
        return self.weight * values.to(dtype)


class ReleasedMamba2MixerFromScratch(nn.Module):
    """Faithful unfused Mamba-2 mixer matching the released 130M tensor schema."""

    def __init__(self, config: ReleasedMamba2Config):
        super().__init__()
        self.config = config
        self.in_proj = nn.Linear(config.d_model, config.projection_size, bias=False)
        self.conv1d = nn.Conv1d(
            config.conv_dim,
            config.conv_dim,
            config.d_conv,
            groups=config.conv_dim,
            bias=True,
        )
        self.dt_bias = nn.Parameter(t.empty(config.n_heads))
        self.A_log = nn.Parameter(t.empty(config.n_heads))
        self.D = nn.Parameter(t.ones(config.n_heads))
        self.norm = ReleasedGatedRMSNorm(config.d_inner, config.eps)
        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=False)

    def forward(self, hidden: t.Tensor) -> t.Tensor:
        dtype = hidden.dtype
        gate, XBC, raw_dt = self.in_proj(hidden).split(
            [self.config.d_inner, self.config.conv_dim, self.config.n_heads], dim=-1
        )
        XBC = F.silu(causal_depthwise_conv1d(XBC, self.conv1d.weight[:, 0], self.conv1d.bias))
        X, B, C = XBC.split(
            [
                self.config.d_inner,
                self.config.n_groups * self.config.d_state,
                self.config.n_groups * self.config.d_state,
            ],
            dim=-1,
        )
        batch, length, _ = X.shape
        X = X.reshape(batch, length, self.config.n_heads, self.config.head_dim).float()
        B = B.reshape(batch, length, self.config.n_groups, self.config.d_state).float()
        C = C.reshape(batch, length, self.config.n_groups, self.config.d_state).float()
        dt = F.softplus(raw_dt + self.dt_bias).float()
        continuous_A = -self.A_log.float().exp()
        Y, _ = ssd_chunked(
            X * dt[..., None],
            continuous_A[None, None] * dt,
            B,
            C,
            self.config.chunk_size,
        )
        Y = Y + self.D.float()[None, None, :, None] * X
        Y = self.norm(Y.reshape(batch, length, self.config.d_inner), gate)
        return self.out_proj(Y.to(dtype))


class ReleasedMamba2BlockFromScratch(nn.Module):
    def __init__(self, config: ReleasedMamba2Config):
        super().__init__()
        self.norm = ReleasedRMSNorm(config.d_model, config.eps)
        self.mixer = ReleasedMamba2MixerFromScratch(config)

    def forward(self, hidden: t.Tensor) -> t.Tensor:
        residual = hidden.float()
        normalized = self.norm(hidden.to(self.norm.weight.dtype))
        return residual + self.mixer(normalized)


class ReleasedMamba2BackboneFromScratch(nn.Module):
    def __init__(self, config: ReleasedMamba2Config):
        super().__init__()
        self.embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList(
            [ReleasedMamba2BlockFromScratch(config) for _ in range(config.n_layers)]
        )
        self.norm_f = ReleasedRMSNorm(config.d_model, config.eps)

    def forward(self, input_ids: t.Tensor) -> tuple[t.Tensor, tuple[t.Tensor, ...]]:
        hidden = self.embeddings(input_ids)
        hidden_states = []
        for layer in self.layers:
            hidden = layer(hidden)
            hidden_states.append(hidden)
        hidden = self.norm_f(hidden)
        hidden_states.append(hidden)
        return hidden, tuple(hidden_states)


class ReleasedMamba2LMFromScratch(nn.Module):
    def __init__(self, config: ReleasedMamba2Config):
        super().__init__()
        self.backbone = ReleasedMamba2BackboneFromScratch(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, input_ids: t.Tensor) -> tuple[t.Tensor, tuple[t.Tensor, ...]]:
        hidden, hidden_states = self.backbone(input_ids)
        logits = self.lm_head(hidden.to(self.lm_head.weight.dtype)).float()
        return logits, hidden_states


@dataclass(frozen=True)
class ReleasedCheckpointParity:
    mapped_tensor_count: int
    hidden_state_count: int
    hidden_max_abs_diff: float
    hidden_mean_abs_diff: float
    logits_max_abs_diff: float
    logits_mean_abs_diff: float
    top1_agreement: float
    reference_top_tokens: tuple[int, ...]
    learner_top_tokens: tuple[int, ...]


def native_to_released_state_dict(native: dict[str, t.Tensor]) -> dict[str, t.Tensor]:
    """Convert the release's single native-vs-HF embedding name difference."""

    mapped = {
        ("backbone.embeddings.weight" if key == "backbone.embedding.weight" else key): value.float()
        for key, value in native.items()
    }
    if len(mapped) != 219:
        raise AssertionError(f"expected 219 released tensors, found {len(mapped)}")
    return mapped


def _load_released_state_dict() -> dict[str, t.Tensor]:
    from huggingface_hub import snapshot_download

    snapshot = snapshot_download(
        REAL_MAMBA2_MODEL_ID,
        revision=REAL_MAMBA2_REVISION,
        allow_patterns=["config.json", "pytorch_model.bin"],
        local_files_only=True,
    )
    native = t.load(
        Path(snapshot) / "pytorch_model.bin",
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    return native_to_released_state_dict(native)


def _released_hf_config():
    from transformers import Mamba2Config as HFMamba2Config

    return HFMamba2Config(
        vocab_size=50288,
        hidden_size=768,
        num_hidden_layers=24,
        expand=2,
        num_heads=24,
        head_dim=64,
        state_size=128,
        n_groups=1,
        conv_kernel=4,
        chunk_size=8,
        tie_word_embeddings=True,
        residual_in_fp32=True,
        use_bias=False,
        use_conv_bias=True,
    )


def verify_released_mamba2_parity(device: str | t.device = "cpu") -> ReleasedCheckpointParity:
    """Strictly map the 130M release and compare two independent implementations."""

    from transformers import Mamba2ForCausalLM

    state = _load_released_state_dict()
    with t.device("meta"):
        learner = ReleasedMamba2LMFromScratch(ReleasedMamba2Config())
        reference = Mamba2ForCausalLM(_released_hf_config())
    learner.load_state_dict(state, strict=True, assign=True)
    reference.load_state_dict(state, strict=True, assign=True)
    learner = learner.to(device).eval()
    reference = reference.to(device).eval()
    input_ids = t.tensor(REAL_MAMBA2_PROMPT_TOKEN_IDS, dtype=t.long, device=device)
    with t.inference_mode():
        learner_logits, learner_hidden = learner(input_ids)
        reference_output = reference(
            input_ids,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
    hidden_differences = [
        (left.float() - right.float()).abs()
        for left, right in zip(learner_hidden, reference_output.hidden_states, strict=True)
    ]
    logits_difference = (learner_logits - reference_output.logits).abs()
    learner_top = learner_logits[:, -1].argmax(-1)
    reference_top = reference_output.logits[:, -1].argmax(-1)
    result = ReleasedCheckpointParity(
        mapped_tensor_count=len(state),
        hidden_state_count=len(hidden_differences),
        hidden_max_abs_diff=max(float(value.max()) for value in hidden_differences),
        hidden_mean_abs_diff=max(float(value.mean()) for value in hidden_differences),
        logits_max_abs_diff=float(logits_difference.max()),
        logits_mean_abs_diff=float(logits_difference.mean()),
        top1_agreement=float((learner_top == reference_top).float().mean()),
        reference_top_tokens=tuple(reference_top.cpu().tolist()),
        learner_top_tokens=tuple(learner_top.cpu().tolist()),
    )
    del learner, reference, state
    gc.collect()
    if t.cuda.is_available():
        t.cuda.empty_cache()
    return result


def make_signature_figure(
    parity: dict[str, float],
    official_parity: dict[str, float],
    recall: RecallExperiment,
    scaling: ScalingResult,
    output_path: str | Path,
) -> Path:
    """Render the learner-facing evidence, controls, and scaling in one figure."""

    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    figure.patch.set_facecolor("#f7f7f5")
    for axis in axes.flat:
        axis.set_facecolor("#ffffff")
        axis.grid(axis="y", color="#d9d9d4", linewidth=0.8, alpha=0.65)

    errors = [
        parity["recurrent_quadratic_max_abs_diff"],
        parity["recurrent_chunked_max_abs_diff"],
        official_parity["output_max_abs_diff"],
    ]
    labels = ["recurrent vs\nquadratic", "recurrent vs\nchunked", "chunked vs\nofficial"]
    axes[0, 0].bar(labels, errors, color=["#2a6f97", "#2a9d8f", "#6d597a"])
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_ylabel("max absolute error (float64)")
    axes[0, 0].set_title("A  One operator, three factorizations")
    axes[0, 0].set_ylim(min(errors) * 0.7, max(errors) * 2.2)
    for index, value in enumerate(errors):
        axes[0, 0].annotate(
            f"{value:.1e}",
            (index, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )

    accuracy_labels = ["Mamba-2\nSSD", "state\nablated", "local\nconv", "shuffled\nlabels", "2x length\nOOD"]
    accuracies = [
        recall.mamba_accuracy,
        recall.state_ablated_accuracy,
        recall.local_conv_accuracy,
        recall.shuffled_label_accuracy,
        recall.ood_length_accuracy,
    ]
    axes[0, 1].bar(
        accuracy_labels,
        accuracies,
        color=["#2a9d8f", "#d1495b", "#e9c46a", "#8d99ae", "#2a6f97"],
    )
    axes[0, 1].axhline(0.5, color="#333333", linestyle="--", linewidth=1, label="chance")
    axes[0, 1].set_ylim(0, 1.08)
    axes[0, 1].set_ylabel("held-out accuracy")
    axes[0, 1].set_title("B  Recurrent state carries the delayed bit")
    axes[0, 1].legend(frameon=False, loc="upper right")
    for index, value in enumerate(accuracies):
        axes[0, 1].text(index, value + 0.025, f"{value:.3f}", ha="center", fontsize=9)

    axes[1, 0].plot(
        scaling.lengths,
        scaling.recurrent_ms,
        "o-",
        label="recurrent Python",
        color="#d1495b",
    )
    axes[1, 0].plot(
        scaling.lengths,
        scaling.quadratic_ms,
        "s-",
        label="full matrix",
        color="#6d597a",
    )
    axes[1, 0].plot(
        scaling.lengths,
        scaling.chunked_ms,
        "^-",
        label="chunked educational",
        color="#2a9d8f",
    )
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xlabel("sequence length")
    axes[1, 0].set_ylabel("measured CPU wall time (ms, median)")
    axes[1, 0].set_title("C  Measured scaling (not a fused-kernel benchmark)")
    axes[1, 0].legend(frameon=False)

    axes[1, 1].plot(
        scaling.lengths,
        scaling.quadratic_matrix_mb,
        "s-",
        label="full L x L decay matrix",
        color="#6d597a",
    )
    axes[1, 1].plot(
        scaling.lengths,
        scaling.chunk_local_matrix_mb,
        "^-",
        label="chunk-local decay blocks",
        color="#e76f51",
    )
    axes[1, 1].set_xscale("log", base=2)
    axes[1, 1].set_yscale("log", base=2)
    axes[1, 1].set_xlabel("sequence length")
    axes[1, 1].set_ylabel("materialized decay storage (MiB)")
    axes[1, 1].set_title("D  The quadratic working set is the real problem")
    axes[1, 1].legend(frameon=False)

    figure.suptitle(
        "Mamba-2 / SSD: exact duality, causal memory, and honest compute tradeoffs",
        fontsize=16,
        fontweight="bold",
    )
    figure.savefig(output_path, dpi=180, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)
    return output_path


def run_smoke_test(cpu: bool = True) -> dict[str, Any]:
    device = t.device("cpu") if cpu else _require_cuda()
    parity = exact_parity_metrics(device=device)
    official = official_ssd_reference_parity(device=device)
    gradients = exact_gradient_parity_metrics(device=device)
    t.manual_seed(4)
    block = Mamba2BlockFromScratch(Mamba2Config()).to(device).eval()
    inputs = t.randn(2, 19, block.config.d_model, device=device)
    full = block(inputs)
    cache = block.allocate_cache(2, device=device, dtype=inputs.dtype)
    steps = []
    with t.inference_mode():
        for position in range(inputs.shape[1]):
            output, cache = block.step(inputs[:, position], cache)
            steps.append(output)
    step_error = float((full - t.cat(steps, dim=1)).abs().max())
    lm = TinyStackedMamba2LM(vocab_size=47, n_layers=2).to(device).eval()
    token_ids = t.randint(0, 47, (2, 16), device=device)
    with t.inference_mode():
        full_logits = lm(token_ids)
        cached_logits, _ = lm.forward_recurrent(token_ids)
    lm_error = float((full_logits - cached_logits).abs().max())
    counts = parameter_matched_control_counts()
    count_error = max(abs(count / counts["Mamba-2"] - 1) for count in counts.values())
    return {
        "device": str(device),
        "parity": parity,
        "official_reference": official,
        "gradients": gradients,
        "official_listing_one_output_max_abs_diff": official["output_max_abs_diff"],
        "official_listing_one_state_max_abs_diff": official["final_state_max_abs_diff"],
        "cache_max_abs_diff": step_error,
        "stacked_lm_cache_max_abs_diff": lm_error,
        "parameter_matched_control_relative_error": count_error,
        "tests_passed": max(parity.values()) < 1e-11
        and max(official.values()) < 1e-11
        and max(gradients.values()) < 1e-11
        and step_error < 2e-6 and lm_error < 2e-6 and count_error < 0.1,
    }


def _require_cuda() -> t.device:
    if not t.cuda.is_available():
        raise RuntimeError("CUDA is required; this release hook has no CPU fallback")
    from triton import knobs

    if os.environ.get("TRITON_F32_DEFAULT") != "ieee" or knobs.language.fp32_default != "ieee":
        raise RuntimeError(
            "Start a fresh process/kernel with TRITON_F32_DEFAULT=ieee before importing "
            "Mamba. TF32 kernels do not satisfy the float32 implementation-parity contract."
        )
    if t.get_float32_matmul_precision() != "highest":
        raise RuntimeError("Set torch.set_float32_matmul_precision('highest') for IEEE parity")
    return t.device("cuda")


def _official_fused_operator_parity(device: t.device) -> dict[str, float]:
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

    generator = t.Generator().manual_seed(7)
    batch, length, heads, head_dim, state = 2, 256, 4, 8, 8
    X = t.randn(batch, length, heads, head_dim, generator=generator).to(device)
    dt = (0.01 + 0.05 * t.rand(batch, length, heads, generator=generator)).to(device)
    A = (-t.exp(t.linspace(0.0, 1.0, heads))).to(device)
    B = t.randn(batch, length, 1, state, generator=generator).to(device)
    C = t.randn(batch, length, 1, state, generator=generator).to(device)
    with t.inference_mode():
        fused = mamba_chunk_scan_combined(X, dt, A, B, C, 64, D=None)
        ours, _ = ssd_chunked(X * dt[..., None], A * dt, B, C, 64)
    difference = (fused.float() - ours.float()).abs()
    return {
        "max_abs_diff": float(difference.max()),
        "mean_abs_diff": float(difference.mean()),
    }


def _official_checkpoint_preflight(device: t.device) -> dict[str, Any]:
    from huggingface_hub import snapshot_download
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

    snapshot = snapshot_download(
        REAL_MAMBA2_MODEL_ID,
        revision=REAL_MAMBA2_REVISION,
        allow_patterns=["config.json", "pytorch_model.bin"],
    )
    model = MambaLMHeadModel.from_pretrained(
        snapshot,
        device=device,
        dtype=t.float16,
    ).eval()
    input_ids = t.tensor([[0, 17, 42, 9, 301, 11, 8, 2]], device=device)
    with t.inference_mode():
        logits = model(input_ids).logits.float()
        repeated = model(input_ids).logits.float()
    deterministic_error = float((logits - repeated).abs().max())
    result = {
        "logits_shape": list(logits.shape),
        "logits_std": float(logits.std()),
        "deterministic_max_abs_diff": deterministic_error,
        "finite": bool(t.isfinite(logits).all()),
    }
    del model, logits, repeated
    t.cuda.empty_cache()
    return result


def run_gpu_test(max_vram_gb: float = 24.0) -> dict[str, Any]:
    """Strict CUDA release test: fused operator plus pinned 130M checkpoint."""

    device = _require_cuda()
    started = time.perf_counter()
    t.cuda.empty_cache()
    t.cuda.reset_peak_memory_stats(device)
    operator = _official_fused_operator_parity(device)
    checkpoint = _official_checkpoint_preflight(device)
    checkpoint_parity = verify_released_mamba2_parity(device)
    recall = run_delayed_recall_experiment(steps=100, device=device)
    t.cuda.synchronize(device)
    peak_vram_gb = t.cuda.max_memory_allocated(device) / 2**30
    elapsed = time.perf_counter() - started
    if peak_vram_gb > max_vram_gb:
        raise AssertionError(
            f"peak VRAM {peak_vram_gb:.2f} GB exceeds {max_vram_gb:.2f} GB"
        )
    if elapsed > 300:
        raise AssertionError(f"GPU verification exceeded five minutes: {elapsed:.1f}s")
    accepted = (
        operator["max_abs_diff"] < 2e-4
        and checkpoint["finite"]
        and checkpoint["deterministic_max_abs_diff"] == 0.0
        and checkpoint["logits_shape"] == [1, 8, 50288]
        and checkpoint["logits_std"] > 0.1
        and checkpoint_parity.mapped_tensor_count == 219
        and checkpoint_parity.hidden_state_count == 25
        and checkpoint_parity.hidden_max_abs_diff < 0.002
        and checkpoint_parity.hidden_mean_abs_diff < 5e-5
        and checkpoint_parity.logits_max_abs_diff < 5e-4
        and checkpoint_parity.logits_mean_abs_diff < 5e-5
        and checkpoint_parity.top1_agreement == 1.0
        and recall.mamba_accuracy >= 0.95
        and recall.state_ablated_accuracy <= 0.65
        and recall.local_conv_accuracy <= 0.65
        and recall.shuffled_label_accuracy <= 0.65
        and recall.ood_length_accuracy >= 0.65
        and peak_vram_gb <= max_vram_gb
    )
    return {
        "cuda_available": True,
        "device": t.cuda.get_device_name(device),
        "torch_version": t.__version__,
        "cuda_version": t.version.cuda,
        "model_id": REAL_MAMBA2_MODEL_ID,
        "model_revision": REAL_MAMBA2_REVISION,
        "triton_f32_default": os.environ["TRITON_F32_DEFAULT"],
        "torch_float32_matmul_precision": t.get_float32_matmul_precision(),
        "held_out_examples": 1024,
        "training_steps_per_model": 100,
        "operator_parity": operator,
        "official_checkpoint": checkpoint,
        "from_scratch_checkpoint_parity": checkpoint_parity.__dict__,
        "official_fused_operator_max_abs_diff": operator["max_abs_diff"],
        "official_checkpoint_finite": checkpoint["finite"],
        "official_checkpoint_logits_shape": checkpoint["logits_shape"],
        "official_checkpoint_logits_std": checkpoint["logits_std"],
        "official_checkpoint_deterministic_max_abs_diff": checkpoint["deterministic_max_abs_diff"],
        "official_checkpoint_peak_vram_gb": peak_vram_gb,
        "pinned_mapped_tensor_count": checkpoint_parity.mapped_tensor_count,
        "pinned_hidden_state_count": checkpoint_parity.hidden_state_count,
        "pinned_hidden_max_abs_diff": checkpoint_parity.hidden_max_abs_diff,
        "pinned_hidden_mean_abs_diff": checkpoint_parity.hidden_mean_abs_diff,
        "pinned_logits_max_abs_diff": checkpoint_parity.logits_max_abs_diff,
        "pinned_logits_mean_abs_diff": checkpoint_parity.logits_mean_abs_diff,
        "pinned_top1_agreement": checkpoint_parity.top1_agreement,
        "delayed_recall_accuracy": recall.mamba_accuracy,
        "doubled_length_accuracy": recall.ood_length_accuracy,
        "recall": {
            "mamba_accuracy": recall.mamba_accuracy,
            "state_ablated_accuracy": recall.state_ablated_accuracy,
            "local_conv_accuracy": recall.local_conv_accuracy,
            "shuffled_label_accuracy": recall.shuffled_label_accuracy,
            "ood_length_accuracy": recall.ood_length_accuracy,
        },
        "peak_vram_gb": peak_vram_gb,
        "max_vram_gb": max_vram_gb,
        "wall_clock_seconds": elapsed,
        "accepted": bool(accepted),
    }


def run_full_experiment(max_vram_gb: float = 24.0) -> dict[str, Any]:
    """Strict CUDA full run with a longer task and measured scaling."""

    device = _require_cuda()
    started = time.perf_counter()
    report = run_gpu_test(max_vram_gb=max_vram_gb)
    recall = run_delayed_recall_experiment(
        steps=180,
        sequence_length=48,
        device=device,
    )
    scaling = measure_ssd_scaling(
        (128, 256, 512, 1024),
        chunk_size=64,
        repeats=5,
        device=device,
    )
    t.cuda.synchronize(device)
    peak_vram_gb = t.cuda.max_memory_allocated(device) / 2**30
    elapsed = time.perf_counter() - started
    if peak_vram_gb > max_vram_gb:
        raise AssertionError(
            f"peak VRAM {peak_vram_gb:.2f} GB exceeds {max_vram_gb:.2f} GB"
        )
    if elapsed > 300:
        raise AssertionError(f"full GPU experiment exceeded five minutes: {elapsed:.1f}s")
    report["full_recall"] = {
        "mamba_accuracy": recall.mamba_accuracy,
        "state_ablated_accuracy": recall.state_ablated_accuracy,
        "local_conv_accuracy": recall.local_conv_accuracy,
        "shuffled_label_accuracy": recall.shuffled_label_accuracy,
        "ood_length_accuracy": recall.ood_length_accuracy,
    }
    report["scaling"] = {
        "lengths": scaling.lengths,
        "recurrent_ms": scaling.recurrent_ms,
        "quadratic_ms": scaling.quadratic_ms,
        "chunked_ms": scaling.chunked_ms,
        "quadratic_matrix_mb": scaling.quadratic_matrix_mb,
        "chunk_local_matrix_mb": scaling.chunk_local_matrix_mb,
    }
    report["peak_vram_gb"] = peak_vram_gb
    report["wall_clock_seconds"] = elapsed
    report["accepted"] = bool(
        report["accepted"]
        and recall.mamba_accuracy >= 0.95
        and recall.state_ablated_accuracy <= 0.65
        and recall.local_conv_accuracy <= 0.65
        and recall.shuffled_label_accuracy <= 0.65
        and recall.ood_length_accuracy >= 0.65
        and peak_vram_gb <= max_vram_gb
    )
    return report
