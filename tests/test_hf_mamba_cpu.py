"""Transformers' Mamba/Mamba2 reference models must run on CPU when the CUDA kernels are installed.

Lessons 5.3 (part3_mamba_from_scratch) and 5.4 (part4_mamba2_ssd) compare the learner's model
with the Transformers reference on CPU. Transformers 5 binds Mamba's ops to causal-conv1d and
mamba-ssm whenever they are importable, and both are CUDA-only, so without
``route_cpu_mamba_ops_to_torch`` the CPU forward pass fails with "Expected x.is_cuda()".
Tiny random-weight models: no downloads.
"""

from __future__ import annotations

import pytest
import torch as t

transformers = pytest.importorskip("transformers")

from chapter5_modern_architectures.exercises.hf_mamba_cpu import (
    device_dispatch,
    route_cpu_mamba_ops_to_torch,
)


def _tiny_models():
    from transformers import Mamba2Config, Mamba2ForCausalLM, MambaConfig, MambaForCausalLM

    t.manual_seed(0)
    mamba = MambaForCausalLM(
        MambaConfig(
            vocab_size=64,
            hidden_size=32,
            state_size=8,
            num_hidden_layers=2,
            expand=2,
            conv_kernel=4,
        )
    )
    mamba2 = Mamba2ForCausalLM(
        Mamba2Config(
            vocab_size=64,
            hidden_size=32,
            num_hidden_layers=2,
            expand=2,
            num_heads=4,
            head_dim=16,
            state_size=8,
            n_groups=1,
            conv_kernel=4,
            chunk_size=4,
        )
    )
    return {"mamba": mamba.cpu().eval(), "mamba2": mamba2.cpu().eval()}


def test_routing_covers_both_model_families():
    routed = route_cpu_mamba_ops_to_torch()
    assert "modeling_mamba.causal_conv1d_fn" in routed
    assert "modeling_mamba.mamba_selective_scan" in routed
    assert "modeling_mamba2.causal_conv1d_fn" in routed
    assert "modeling_mamba2.mamba2_chunk_scan" in routed
    assert route_cpu_mamba_ops_to_torch() == routed  # idempotent


@pytest.mark.parametrize("family", ["mamba", "mamba2"])
def test_hf_reference_forward_and_cached_decode_run_on_cpu(family):
    route_cpu_mamba_ops_to_torch()
    model = _tiny_models()[family]
    ids = t.tensor([[1, 5, 9, 3, 7, 2, 4, 6]], dtype=t.long)
    with t.inference_mode():
        full = model(ids, attention_mask=t.ones_like(ids), use_cache=False).logits
        tokens = model.generate(
            ids, attention_mask=t.ones_like(ids), max_new_tokens=3, do_sample=False, pad_token_id=0
        )
    assert full.shape == (1, ids.shape[1], 64)
    assert t.isfinite(full).all()
    assert tokens.shape == (1, ids.shape[1] + 3)


def test_dispatch_keeps_the_fast_kernel_off_cpu():
    calls = []
    op = device_dispatch(
        lambda x, **kw: calls.append(("fast", kw)),
        lambda x, activation=None: calls.append(("ref", activation)),
    )
    op(t.zeros(1), activation="silu", seq_idx=None)
    op(t.zeros(1, device="meta"), activation="silu", seq_idx=None)
    assert calls == [("ref", "silu"), ("fast", {"activation": "silu", "seq_idx": None})]
