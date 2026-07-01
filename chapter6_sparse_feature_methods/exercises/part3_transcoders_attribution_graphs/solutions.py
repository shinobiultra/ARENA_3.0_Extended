# %%
"""Reference solutions for [6.3] Transcoders and Attribution Graphs."""

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch as t
import torch.nn.functional as F

chapter = "chapter6_sparse_feature_methods"
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

MAIN = __name__ == "__main__"

TL_GELU1L_MODEL_NAME = "gelu-1l"
TL_GELU1L_REVISION = "bddc0e332f0ae84279e6a6a45d91b314899e1603"
TL_GELU1L_TOKENIZER_ID = "NeelNanda/gpt-neox-tokenizer-digits"
TL_GELU1L_TOKENIZER_REVISION = "0f6671571a20be9756b9991d978047c03b75e749"
TL_BNB_CUDA_OVERRIDE = "130"
TL_LN2_HOOK = "blocks.0.ln2.hook_normalized"
TL_MLP_POST_HOOK = "blocks.0.mlp.hook_post"
TL_MLP_OUT_HOOK = "blocks.0.hook_mlp_out"
TL_RESID_MID_HOOK = "blocks.0.hook_resid_mid"
TL_TRANSCODER_PROMPTS = [
    "The cat sat on the",
    "The bird flew over the",
    "To make tea, boil the",
    "To make bread, bake the",
    "The recipe calls for sugar and",
    "The recipe calls for salt and",
    "The chef cooked a",
    "The teacher taught a",
    "The river flows into the",
    "The road leads into the",
    "The Python function returns a",
    "The HTML page contains a",
    "The file was saved to disk as a",
    "The message was sent by mail as a",
    "The train arrived at the",
    "The plane landed at the",
]


# %%
@dataclass(frozen=True)
class TranscoderOutput:
    feature_acts: t.Tensor
    reconstructed_activations: t.Tensor


@dataclass(frozen=True)
class TranscoderReplacementReport:
    reconstruction_mse: float
    replacement_kl: float
    target_logit_diff: float
    replacement_logit_diff: float
    logit_diff_error: float
    passes_kl: bool
    preserves_logit_diff: bool


@dataclass(frozen=True)
class AttributionEdge:
    source_type: str
    source_id: int
    target_type: str
    target_id: int
    weight: float


@dataclass(frozen=True)
class AttributionGraphReport:
    num_nodes: int
    num_edges: int
    density: float
    full_logit_diff: float
    graph_logit_diff: float
    ablated_logit_diff: float
    random_ablated_logit_diff: float
    topk_damage: float
    random_damage: float
    preserves_logit_diff: bool
    passes_damage_control: bool
    reproducible: bool


def mean_kl_divergence(reference_logits: t.Tensor, reconstructed_logits: t.Tensor) -> float:
    """Mean KL(reference || reconstructed) over all non-vocabulary dimensions."""

    if reference_logits.shape != reconstructed_logits.shape:
        raise ValueError("reference_logits and reconstructed_logits must have matching shapes.")
    reference_log_probs = F.log_softmax(reference_logits.float(), dim=-1)
    reconstructed_log_probs = F.log_softmax(reconstructed_logits.float(), dim=-1)
    reference_probs = reference_log_probs.exp()
    kl = reference_probs * (reference_log_probs - reconstructed_log_probs)
    return kl.sum(dim=-1).mean().item()


def transcoder_forward(
    inputs: t.Tensor,
    encoder_weight: t.Tensor,
    decoder_weight: t.Tensor,
    *,
    encoder_bias: t.Tensor | None = None,
    decoder_bias: t.Tensor | None = None,
) -> TranscoderOutput:
    """Run a one-hidden-layer ReLU transcoder."""

    if inputs.shape[-1] != encoder_weight.shape[-1]:
        raise ValueError("inputs last dimension must match encoder_weight input dimension.")
    if encoder_weight.shape[0] != decoder_weight.shape[0]:
        raise ValueError("encoder features must match decoder rows.")

    pre_acts = inputs.float() @ encoder_weight.float().T
    if encoder_bias is not None:
        pre_acts = pre_acts + encoder_bias.to(pre_acts.device)
    feature_acts = pre_acts.clamp_min(0)
    reconstructed = feature_acts @ decoder_weight.float()
    if decoder_bias is not None:
        reconstructed = reconstructed + decoder_bias.to(reconstructed.device)
    return TranscoderOutput(
        feature_acts=feature_acts,
        reconstructed_activations=reconstructed,
    )


def target_logit_diff(
    logits: t.Tensor,
    *,
    positive_token_id: int,
    negative_token_id: int,
) -> float:
    """Mean positive-minus-negative logit difference."""

    if logits.shape[-1] <= max(positive_token_id, negative_token_id):
        raise ValueError("token ids are out of range for logits.")
    diff = logits[..., positive_token_id] - logits[..., negative_token_id]
    return diff.float().mean().item()


def transcoder_replacement_report(
    *,
    reference_activations: t.Tensor,
    reconstructed_activations: t.Tensor,
    reference_logits: t.Tensor,
    replacement_logits: t.Tensor,
    positive_token_id: int,
    negative_token_id: int,
    kl_threshold: float = 1e-3,
    logit_diff_tolerance: float = 0.1,
) -> TranscoderReplacementReport:
    """Check whether replacing an MLP with a transcoder preserves behavior."""

    if reference_activations.shape != reconstructed_activations.shape:
        raise ValueError("activation tensors must have matching shapes.")
    replacement_kl = mean_kl_divergence(reference_logits, replacement_logits)
    target_diff = target_logit_diff(
        reference_logits,
        positive_token_id=positive_token_id,
        negative_token_id=negative_token_id,
    )
    replacement_diff = target_logit_diff(
        replacement_logits,
        positive_token_id=positive_token_id,
        negative_token_id=negative_token_id,
    )
    error = abs(target_diff - replacement_diff)
    return TranscoderReplacementReport(
        reconstruction_mse=F.mse_loss(
            reconstructed_activations.float(),
            reference_activations.float(),
        ).item(),
        replacement_kl=replacement_kl,
        target_logit_diff=target_diff,
        replacement_logit_diff=replacement_diff,
        logit_diff_error=error,
        passes_kl=replacement_kl <= kl_threshold,
        preserves_logit_diff=error <= logit_diff_tolerance,
    )


def feature_logit_contributions(feature_acts: t.Tensor, logit_effects: t.Tensor) -> t.Tensor:
    """Estimate each feature's contribution to a target logit difference."""

    if feature_acts.shape[-1] != logit_effects.shape[0]:
        raise ValueError("feature_acts feature dimension must match logit_effects.")
    reduce_dims = tuple(range(feature_acts.ndim - 1))
    mean_feature_acts = feature_acts.float().mean(dim=reduce_dims)
    return mean_feature_acts * logit_effects.float()


def build_attribution_edges(
    input_to_feature_scores: t.Tensor,
    feature_logit_effects: t.Tensor,
    *,
    top_k: int,
) -> list[AttributionEdge]:
    """Build a sparse input-to-feature-to-logit attribution graph."""

    if input_to_feature_scores.ndim != 2:
        raise ValueError("input_to_feature_scores must have shape (inputs, features).")
    if input_to_feature_scores.shape[1] != feature_logit_effects.shape[0]:
        raise ValueError("feature dimension must match feature_logit_effects.")
    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    edges: list[AttributionEdge] = []
    flat_scores = input_to_feature_scores.float().flatten()
    n_features = input_to_feature_scores.shape[1]
    _, flat_indices = flat_scores.abs().topk(k=min(top_k, flat_scores.numel()))
    for flat_index in flat_indices:
        source_id = int(flat_index.item() // n_features)
        target_id = int(flat_index.item() % n_features)
        edges.append(
            AttributionEdge(
                source_type="input",
                source_id=source_id,
                target_type="feature",
                target_id=target_id,
                weight=float(input_to_feature_scores[source_id, target_id].item()),
            )
        )

    _, feature_ids = feature_logit_effects.float().abs().topk(
        k=min(top_k, feature_logit_effects.numel())
    )
    for feature_id in feature_ids:
        edges.append(
            AttributionEdge(
                source_type="feature",
                source_id=int(feature_id.item()),
                target_type="logit_diff",
                target_id=0,
                weight=float(feature_logit_effects[feature_id].item()),
            )
        )
    return edges


def graph_density(*, num_nodes: int, num_edges: int) -> float:
    """Directed graph density without self-loops."""

    if num_nodes <= 1:
        return 0.0
    return num_edges / (num_nodes * (num_nodes - 1))


def graph_reproducible(
    edges_a: list[AttributionEdge],
    edges_b: list[AttributionEdge],
    *,
    atol: float = 1e-6,
) -> bool:
    """Check that two graph edge lists match in structure and weight."""

    if len(edges_a) != len(edges_b):
        return False
    for edge_a, edge_b in zip(edges_a, edges_b, strict=True):
        same_structure = (
            edge_a.source_type == edge_b.source_type
            and edge_a.source_id == edge_b.source_id
            and edge_a.target_type == edge_b.target_type
            and edge_a.target_id == edge_b.target_id
        )
        if not same_structure or abs(edge_a.weight - edge_b.weight) > atol:
            return False
    return True


def attribution_graph_report(
    contributions: t.Tensor,
    graph_feature_ids: t.Tensor | list[int],
    random_feature_ids: t.Tensor | list[int],
    *,
    num_nodes: int,
    num_edges: int,
    reproducible: bool,
    preservation_threshold: float = 0.8,
) -> AttributionGraphReport:
    """Report graph preservation and top-node ablation controls."""

    contributions = contributions.flatten().float()
    graph_ids = t.as_tensor(graph_feature_ids, dtype=t.long, device=contributions.device)
    random_ids = t.as_tensor(random_feature_ids, dtype=t.long, device=contributions.device)
    if graph_ids.numel() == 0 or random_ids.numel() == 0:
        raise ValueError("graph_feature_ids and random_feature_ids must be nonempty.")

    full_diff = contributions.sum().item()
    graph_diff = contributions[graph_ids].sum().item()
    ablated_diff = full_diff - graph_diff
    random_removed = contributions[random_ids].sum().item()
    random_ablated_diff = full_diff - random_removed
    topk_damage = abs(full_diff - ablated_diff)
    random_damage = abs(full_diff - random_ablated_diff)
    preserved_fraction = abs(graph_diff) / max(abs(full_diff), 1e-12)

    return AttributionGraphReport(
        num_nodes=num_nodes,
        num_edges=num_edges,
        density=graph_density(num_nodes=num_nodes, num_edges=num_edges),
        full_logit_diff=full_diff,
        graph_logit_diff=graph_diff,
        ablated_logit_diff=ablated_diff,
        random_ablated_logit_diff=random_ablated_diff,
        topk_damage=topk_damage,
        random_damage=random_damage,
        preserves_logit_diff=preserved_fraction >= preservation_threshold,
        passes_damage_control=topk_damage > random_damage,
        reproducible=reproducible,
    )


def edge_to_dict(edge: AttributionEdge) -> dict:
    return {
        "source_type": edge.source_type,
        "source_id": edge.source_id,
        "target_type": edge.target_type,
        "target_id": edge.target_id,
        "weight": edge.weight,
    }


def transcoder_forward_smoke_test() -> dict:
    inputs = t.tensor([[1.0, 2.0], [3.0, 4.0]])
    output = transcoder_forward(inputs, t.eye(2), t.eye(2))
    return {
        "feature_acts": output.feature_acts.tolist(),
        "reconstructed_activations": output.reconstructed_activations.tolist(),
    }


def replacement_report_smoke_test() -> dict:
    reference_activations = t.tensor([[1.0, 2.0]])
    reconstructed = reference_activations + 0.01
    reference_logits = t.tensor([[2.0, 0.0, -1.0]])
    replacement_logits = t.tensor([[1.98, 0.02, -1.0]])
    return transcoder_replacement_report(
        reference_activations=reference_activations,
        reconstructed_activations=reconstructed,
        reference_logits=reference_logits,
        replacement_logits=replacement_logits,
        positive_token_id=0,
        negative_token_id=1,
        kl_threshold=1e-3,
        logit_diff_tolerance=0.1,
    ).__dict__


def contribution_smoke_test() -> dict:
    feature_acts = t.tensor([[1.0, 2.0, 0.0], [3.0, 0.0, 2.0]])
    logit_effects = t.tensor([0.5, 1.0, -1.0])
    logits = t.tensor([[3.0, 1.0]])
    return {
        "contributions": feature_logit_contributions(feature_acts, logit_effects).tolist(),
        "target_logit_diff": target_logit_diff(
            logits,
            positive_token_id=0,
            negative_token_id=1,
        ),
    }


def graph_edges_smoke_test() -> dict:
    input_scores = t.tensor([[0.1, 0.8], [0.4, 0.2]])
    logit_effects = t.tensor([0.3, 1.0])
    edges = build_attribution_edges(input_scores, logit_effects, top_k=1)
    repeated = build_attribution_edges(input_scores, logit_effects, top_k=1)
    return {
        "num_edges": len(edges),
        "first_edge": edge_to_dict(edges[0]),
        "second_edge": edge_to_dict(edges[1]),
        "reproducible": graph_reproducible(edges, repeated),
    }


def graph_report_smoke_test() -> dict:
    contributions = t.tensor([0.7, 0.2, 0.05, 0.05])
    return attribution_graph_report(
        contributions,
        graph_feature_ids=[0, 1],
        random_feature_ids=[2, 3],
        num_nodes=6,
        num_edges=4,
        reproducible=True,
    ).__dict__


def run_smoke_test(cpu: bool = True) -> dict:
    _ = cpu
    return {
        "forward": transcoder_forward_smoke_test(),
        "replacement": replacement_report_smoke_test(),
        "contributions": contribution_smoke_test(),
        "graph_edges": graph_edges_smoke_test(),
        "graph_report": graph_report_smoke_test(),
    }


def _load_gelu1l_model_on_cuda():
    os.environ.setdefault("BNB_CUDA_VERSION", TL_BNB_CUDA_OVERRIDE)
    logging.getLogger("bitsandbytes.cextension").setLevel(logging.ERROR)
    from transformer_lens import HookedTransformer

    return HookedTransformer.from_pretrained(
        TL_GELU1L_MODEL_NAME,
        device="cuda",
        dtype="float32",
        revision=TL_GELU1L_REVISION,
    )


def _decode_from_residual(model, residual_stream: t.Tensor) -> t.Tensor:
    return model.ln_final(residual_stream) @ model.W_U + model.b_U


def _cache_transcoder_dataset(model) -> dict[str, t.Tensor]:
    names = {TL_LN2_HOOK, TL_MLP_POST_HOOK, TL_MLP_OUT_HOOK, TL_RESID_MID_HOOK}
    normalized_inputs = []
    mlp_features = []
    mlp_outputs = []
    resid_mid = []
    logits = []
    for prompt in TL_TRANSCODER_PROMPTS:
        tokens = model.to_tokens(prompt)
        with t.no_grad():
            prompt_logits, cache = model.run_with_cache(
                tokens,
                names_filter=lambda name: name in names,
            )
        normalized_inputs.append(cache[TL_LN2_HOOK][0].detach())
        mlp_features.append(cache[TL_MLP_POST_HOOK][0].detach())
        mlp_outputs.append(cache[TL_MLP_OUT_HOOK][0].detach())
        resid_mid.append(cache[TL_RESID_MID_HOOK][0].detach())
        logits.append(prompt_logits[0].detach())
    return {
        "normalized_inputs": t.cat(normalized_inputs, dim=0),
        "mlp_features": t.cat(mlp_features, dim=0),
        "mlp_outputs": t.cat(mlp_outputs, dim=0),
        "resid_mid": t.cat(resid_mid, dim=0),
        "logits": t.cat(logits, dim=0),
    }


def _fit_tiny_relu_transcoder(
    normalized_inputs: t.Tensor,
    mlp_outputs: t.Tensor,
    *,
    width: int = 512,
    steps: int = 800,
) -> dict[str, t.Tensor | float]:
    """Train a small ReLU transcoder on real MLP input/output activations."""

    t.manual_seed(6303)
    feature_mean = normalized_inputs.mean(dim=0, keepdim=True)
    feature_std = normalized_inputs.std(dim=0, keepdim=True).clamp_min(1e-3)
    inputs = (normalized_inputs - feature_mean) / feature_std
    order = t.randperm(inputs.shape[0], device=inputs.device)
    split = int(0.8 * order.numel())
    train_idx = order[:split]
    heldout_idx = order[split:]

    encoder = t.nn.Linear(inputs.shape[-1], width, device=inputs.device)
    decoder = t.nn.Linear(width, mlp_outputs.shape[-1], device=inputs.device)
    opt = t.optim.AdamW(
        [*encoder.parameters(), *decoder.parameters()],
        lr=3e-3,
        weight_decay=1e-4,
    )
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        feature_acts = F.relu(encoder(inputs[train_idx]))
        reconstructed = decoder(feature_acts)
        loss = F.mse_loss(reconstructed, mlp_outputs[train_idx]) + 1e-5 * feature_acts.mean()
        loss.backward()
        opt.step()

    with t.no_grad():
        output = transcoder_forward(
            inputs,
            encoder.weight.detach(),
            decoder.weight.detach().T,
            encoder_bias=encoder.bias.detach(),
            decoder_bias=decoder.bias.detach(),
        )
        heldout_mse = F.mse_loss(
            output.reconstructed_activations[heldout_idx],
            mlp_outputs[heldout_idx],
        ).item()
        train_mse = F.mse_loss(
            output.reconstructed_activations[train_idx],
            mlp_outputs[train_idx],
        ).item()
        heldout_zero_mse = F.mse_loss(
            t.zeros_like(mlp_outputs[heldout_idx]),
            mlp_outputs[heldout_idx],
        ).item()
    return {
        "feature_acts": output.feature_acts.detach(),
        "reconstructed": output.reconstructed_activations.detach(),
        "train_mse": train_mse,
        "heldout_mse": heldout_mse,
        "heldout_zero_mse": heldout_zero_mse,
        "heldout_idx": heldout_idx,
        "feature_density": output.feature_acts.gt(0).float().mean().item(),
        "width": width,
        "steps": steps,
    }


def run_transformerlens_transcoder_preflight(max_vram_gb: float = 24.0) -> dict:
    """Run a real-model MLP feature graph plus trained tiny transcoder preflight."""

    if not t.cuda.is_available():
        raise RuntimeError("CUDA is required for the 6.3 real-model transcoder preflight.")

    t.cuda.reset_peak_memory_stats()
    model = _load_gelu1l_model_on_cuda()
    model.eval()
    data = _cache_transcoder_dataset(model)

    mlp = model.blocks[0].mlp
    oracle_mlp_out = data["mlp_features"] @ mlp.W_out.detach() + mlp.b_out.detach()
    oracle_logits = _decode_from_residual(model, data["resid_mid"] + oracle_mlp_out)
    positive_token_id = model.to_single_token(" floor")
    negative_token_id = model.to_single_token(" top")
    oracle_replacement = transcoder_replacement_report(
        reference_activations=data["mlp_outputs"],
        reconstructed_activations=oracle_mlp_out,
        reference_logits=data["logits"],
        replacement_logits=oracle_logits,
        positive_token_id=positive_token_id,
        negative_token_id=negative_token_id,
        kl_threshold=1e-6,
        logit_diff_tolerance=1e-5,
    )
    oracle_mlp_out_max_abs_error = (data["mlp_outputs"] - oracle_mlp_out).abs().max().item()
    oracle_logits_max_abs_error = (data["logits"] - oracle_logits).abs().max().item()

    trained = _fit_tiny_relu_transcoder(
        data["normalized_inputs"],
        data["mlp_outputs"],
    )
    trained_logits = _decode_from_residual(model, data["resid_mid"] + trained["reconstructed"])
    trained_top1_agreement = (
        trained_logits.argmax(dim=-1).eq(data["logits"].argmax(dim=-1)).float().mean().item()
    )
    trained_reconstruction_ratio = trained["heldout_mse"] / max(
        trained["heldout_zero_mse"],
        1e-12,
    )

    logit_effects = mlp.W_out.detach() @ (
        model.W_U[:, positive_token_id] - model.W_U[:, negative_token_id]
    )
    contributions = feature_logit_contributions(data["mlp_features"], logit_effects)
    top_feature_ids = contributions.abs().topk(k=64).indices
    random_feature_ids = contributions.abs().topk(k=64, largest=False).indices
    edges = build_attribution_edges(
        data["mlp_features"].abs(),
        logit_effects,
        top_k=16,
    )
    repeated_edges = build_attribution_edges(
        data["mlp_features"].abs(),
        logit_effects,
        top_k=16,
    )
    graph = attribution_graph_report(
        contributions,
        graph_feature_ids=top_feature_ids,
        random_feature_ids=random_feature_ids,
        num_nodes=int(data["mlp_features"].shape[-1] + data["mlp_features"].shape[0] + 1),
        num_edges=len(edges),
        reproducible=graph_reproducible(edges, repeated_edges),
        preservation_threshold=0.2,
    )

    t.cuda.synchronize()
    peak_vram_gb = t.cuda.max_memory_allocated() / 1024**3
    within_vram_budget = peak_vram_gb <= max_vram_gb
    preflight_passed = (
        oracle_replacement.passes_kl
        and oracle_replacement.preserves_logit_diff
        and oracle_logits_max_abs_error <= 5e-5
        and trained_reconstruction_ratio <= 0.5
        and trained_top1_agreement >= 0.75
        and graph.preserves_logit_diff
        and graph.passes_damage_control
        and graph.reproducible
        and within_vram_budget
    )
    del model
    t.cuda.empty_cache()
    return {
        "cuda_available": True,
        "device": t.cuda.get_device_name(0),
        "model_name": TL_GELU1L_MODEL_NAME,
        "hf_revision": TL_GELU1L_REVISION,
        "tokenizer_id": TL_GELU1L_TOKENIZER_ID,
        "tokenizer_revision": TL_GELU1L_TOKENIZER_REVISION,
        "prompt_count": len(TL_TRANSCODER_PROMPTS),
        "activation_count": int(data["mlp_outputs"].shape[0]),
        "d_model": int(data["mlp_outputs"].shape[-1]),
        "mlp_width": int(data["mlp_features"].shape[-1]),
        "oracle_mlp_out_max_abs_error": oracle_mlp_out_max_abs_error,
        "oracle_logits_max_abs_error": oracle_logits_max_abs_error,
        "oracle_replacement_kl": oracle_replacement.replacement_kl,
        "oracle_preserves_logit_diff": oracle_replacement.preserves_logit_diff,
        "trained_transcoder_width": trained["width"],
        "trained_transcoder_steps": trained["steps"],
        "trained_transcoder_train_mse": trained["train_mse"],
        "trained_transcoder_heldout_mse": trained["heldout_mse"],
        "trained_transcoder_heldout_zero_mse": trained["heldout_zero_mse"],
        "trained_transcoder_heldout_mse_ratio": trained_reconstruction_ratio,
        "trained_transcoder_feature_density": trained["feature_density"],
        "trained_replacement_top1_agreement": trained_top1_agreement,
        "feature_acts_sum": float(data["mlp_features"].sum().item()),
        "graph_feature_count": int(top_feature_ids.numel()),
        "graph_preserves_logit_diff": graph.preserves_logit_diff,
        "graph_passes_damage_control": graph.passes_damage_control,
        "graph_reproducible": graph.reproducible,
        "graph_topk_damage": graph.topk_damage,
        "graph_random_damage": graph.random_damage,
        "replacement_kl": oracle_replacement.replacement_kl,
        "preserves_logit_diff": oracle_replacement.preserves_logit_diff,
        "peak_vram_gb": peak_vram_gb,
        "within_vram_budget": within_vram_budget,
        "preflight_passed": preflight_passed,
        "full_path": (
            "Pinned TransformerLens gelu-1l CUDA preflight: exact MLP-feature oracle "
            "replacement, trained tiny ReLU transcoder held-out reconstruction, and "
            "feature-level attribution graph with top-feature versus low-effect controls."
        ),
    }


def run_gpu_test(max_vram_gb: float = 24.0) -> dict:
    return run_transformerlens_transcoder_preflight(max_vram_gb=max_vram_gb)


def run_full_experiment(max_vram_gb: float = 24.0) -> dict:
    """Run the validated experiment path used by the verification report."""

    return run_gpu_test(max_vram_gb=max_vram_gb)


if MAIN:
    print(run_smoke_test())
