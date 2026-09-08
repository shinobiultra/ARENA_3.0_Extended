"""Visible EAP and EAP-IG-inputs computations for the same full residual graph.

Reference: Hanna et al., Have Faith in Faithfulness (2024), and the authors'
MIT implementation at hannamw/EAP-IG commit
e24bafd3d22af2c59ac5a83e0a6581b89b5c05dd. Our loss callback returns a mean
over examples. Positive scores estimate damage from corrupting an edge.
"""
from __future__ import annotations

from typing import Callable

import torch as t

from .real_edges import build_residual_graph, final_position_logits


def cache_sender_outputs(model, tokens: t.Tensor, graph) -> t.Tensor:
    """Provided cache plumbing, ordered exactly like graph.sources."""
    names = {"blocks.0.hook_resid_pre"}
    for layer in range(graph.n_layers):
        names.update((f"blocks.{layer}.attn.hook_result", f"blocks.{layer}.hook_mlp_out"))
    with t.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=lambda name: name in names,
                                        stop_at_layer=graph.n_layers, return_type=None)
        pieces = [cache["blocks.0.hook_resid_pre"][:, :, None]]
        for layer in range(graph.n_layers):
            pieces.extend((cache[f"blocks.{layer}.attn.hook_result"],
                           cache[f"blocks.{layer}.hook_mlp_out"][:, :, None]))
        return t.cat(pieces, dim=2).detach()


def contract_edge_gradients(corrupt_minus_clean: t.Tensor, receiver_gradient: t.Tensor) -> t.Tensor:
    """Sum over examples, positions and hidden dimensions; retain sender/receiver axes."""
    if corrupt_minus_clean.ndim != 4 or receiver_gradient.ndim not in (3, 4):
        raise ValueError("use [batch, position, sender, hidden] deltas and scalar/grouped receiver gradients")
    if receiver_gradient.ndim == 3:
        receiver_gradient = receiver_gradient[:, :, None, :]
    if (corrupt_minus_clean.shape[:2] != receiver_gradient.shape[:2]
            or corrupt_minus_clean.shape[-1] != receiver_gradient.shape[-1]):
        raise ValueError("sender and receiver batch, position and hidden dimensions must match")
    return t.einsum("bpsd,bphd->sh", corrupt_minus_clean, receiver_gradient)


def edge_attribution_scores(model, clean_tokens: t.Tensor, corrupt_tokens: t.Tensor,
                            end_positions: t.Tensor, mean_loss: Callable[[t.Tensor], t.Tensor], *,
                            method: str = "eap-ig-inputs", steps: int = 8) -> t.Tensor:
    """Score all legal edges with fixed endpoint deltas and input-path gradients.

    EAP uses the clean endpoint. EAP-IG uses alpha=0,1/m,...,(m-1)/m,
    matching the authors' left-endpoint quadrature. The loss already averages
    examples, so only average again over integration steps, not batch size.
    This is not integrated gradients over circuit masks or a completeness proof.
    """
    if method not in ("eap", "eap-ig-inputs") or steps < 1:
        raise ValueError("choose eap or eap-ig-inputs and a positive integration count")
    if clean_tokens.shape != corrupt_tokens.shape or clean_tokens.ndim != 2:
        raise ValueError("use aligned clean/corrupt token matrices")
    if model.cfg.attn_only or model.cfg.parallel_attn_mlp or getattr(model.cfg, "n_key_value_heads", None) is not None:
        raise ValueError("use the sequential equal-head attention/MLP graph taught here")
    model.set_use_attn_result(True)
    model.set_use_split_qkv_input(True)
    model.set_use_hook_mlp_in(True)
    graph = build_residual_graph(model.cfg.n_layers, model.cfg.n_heads)
    parameters = list(model.parameters())
    original_requires_grad = [parameter.requires_grad for parameter in parameters]
    was_training = model.training
    device = parameters[0].device
    clean_tokens, corrupt_tokens = clean_tokens.to(device), corrupt_tokens.to(device)
    groups = {}
    for column, receiver in enumerate(graph.receivers):
        groups.setdefault(receiver.hook_name, []).append(column)
    try:
        model.eval()
        for parameter in parameters:
            parameter.requires_grad_(False)
        with t.inference_mode(False), t.enable_grad():
            clean = cache_sender_outputs(model, clean_tokens, graph)
            corrupt = cache_sender_outputs(model, corrupt_tokens, graph)
            differences = corrupt - clean
            scores = t.zeros(graph.legal_mask.shape, device=device, dtype=clean.dtype)
            alphas = [1.] if method == "eap" else [step / steps for step in range(steps)]
            for alpha in alphas:
                interpolated = (corrupt[:, :, 0] + alpha * (clean[:, :, 0] - corrupt[:, :, 0])).detach().requires_grad_(True)
                handles = []

                def replace_input(activation, hook):
                    return interpolated

                hooks = [("blocks.0.hook_resid_pre", replace_input)]
                for hook_name, columns in groups.items():
                    receiver = graph.receivers[columns[0]]

                    def capture(activation, hook, columns=tuple(columns), receiver=receiver):
                        def accumulate(gradient):
                            with t.no_grad():
                                update = contract_edge_gradients(
                                    differences[:, :, :receiver.source_count], gradient,
                                )
                                scores[:receiver.source_count, list(columns)] += update
                        handles.append(activation.register_hook(accumulate))
                        return activation
                    hooks.append((hook_name, capture))
                try:
                    with model.hooks(fwd_hooks=hooks):
                        residual = model(clean_tokens, stop_at_layer=graph.n_layers, return_type=None)
                        loss = mean_loss(final_position_logits(model, residual, end_positions))
                        if loss.ndim != 0 or not bool(t.isfinite(loss)):
                            raise ValueError("the attribution loss must be a finite scalar mean over examples")
                        t.autograd.grad(loss, interpolated)
                finally:
                    for handle in handles:
                        handle.remove()
            scores /= len(alphas)
            if not bool(t.isfinite(scores).all()):
                raise ValueError("attribution scores must be finite")
            return scores.detach().cpu()
    finally:
        for parameter, required in zip(parameters, original_requires_grad, strict=True):
            parameter.requires_grad_(required)
        model.train(was_training)


def top_edge_mask(graph, scores: t.Tensor, edge_count: int) -> t.Tensor:
    """Absolute-score top-k over legal edges, with deterministic graph-order tie breaks.

    Keep the exact requested size, even if some selected edges are disconnected.
    Faithfulness evaluation, rather than score size, determines circuit quality.
    """
    if scores.shape != graph.legal_mask.shape or not bool(t.isfinite(scores).all()):
        raise ValueError("scores must be a finite full-graph matrix")
    legal = graph.legal_mask.nonzero()
    if edge_count < 0 or edge_count > len(legal):
        raise ValueError("edge count must lie between zero and the full legal graph size")
    values = scores.detach().cpu()[legal[:, 0], legal[:, 1]].abs()
    chosen = legal[t.argsort(values, descending=True, stable=True)[:edge_count]]
    mask = t.zeros_like(graph.legal_mask)
    mask[chosen[:, 0], chosen[:, 1]] = True
    return mask


def greedy_score_mask(graph, scores: t.Tensor, edge_count: int) -> t.Tensor:
    """Grow an exact-budget circuit backward from the output using edge-score magnitude.

    A receiver enters the frontier when its component has a selected path to
    the output. This matches the author's greedy selector with prune=False;
    ties use our declared sender/receiver index order. All selected edges can
    reach the output, but some can lack a retained path from the input. Count
    those edges too and determine usefulness with real hybrid evaluation.
    """
    from heapq import heappop, heappush

    if scores.shape != graph.legal_mask.shape or not bool(t.isfinite(scores).all()):
        raise ValueError("scores must be a finite full-graph matrix")
    if edge_count < 0 or edge_count > int(graph.legal_mask.sum()):
        raise ValueError("edge count must be within the full legal candidate graph")
    magnitudes = scores.detach().cpu().abs()
    inputs_by_source = {}
    for column, receiver in enumerate(graph.receivers):
        if receiver.output_source is not None:
            inputs_by_source.setdefault(receiver.output_source, []).append(column)
    mask = t.zeros_like(graph.legal_mask)
    frontier, opened_sources = [], set()

    def add_receiver(column):
        for source in range(graph.receivers[column].source_count):
            heappush(frontier, (-float(magnitudes[source, column]), source, column))

    add_receiver(len(graph.receivers) - 1)
    for _ in range(edge_count):
        _, source, receiver = heappop(frontier)
        mask[source, receiver] = True
        if source not in opened_sources:
            opened_sources.add(source)
            for column in inputs_by_source.get(source, ()):
                add_receiver(column)
    return mask
