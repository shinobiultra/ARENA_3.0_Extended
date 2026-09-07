"""Visible reference bodies for full additive-edge ACDC in sequential transformers.

This graph exposes every residual sender to each later Q/K/V receiver, each
MLP input, and the final output. It does not preselect known circuit heads.
The teaching notebook will contain the core computations explicitly.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch as t
from torch.nn import functional as F

GPT2_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"


def load_pinned_gpt2(device: str = "cuda"):
    """Weight/tokenizer plumbing; do not edit the original ARENA compatibility surface."""
    from transformer_lens import HookedTransformer

    model = HookedTransformer.from_pretrained(
        "gpt2-small", device=device, revision=GPT2_REVISION,
        center_unembed=True, center_writing_weights=True, fold_ln=True,
        refactor_factored_attn_matrices=True,
    ).eval()
    model.tokenizer.add_bos_token = False
    model.tokenizer.padding_side = "right"
    return model


def prepare_ioi_data(model, *, count: int, seed: int, shifted_templates: bool = False) -> dict:
    """Use original ARENA's generated IOI/ABC pairs, adapting only tokenizer boundaries."""
    from chapter1_transformer_interp.exercises.part41_indirect_object_identification.ioi_dataset import (
        IOIDataset, ABBA_TEMPLATES, BABA_TEMPLATES,
    )

    start = 3 if shifted_templates else 0
    templates = ABBA_TEMPLATES[start:start + 3] + BABA_TEMPLATES[start:start + 3]
    dataset = IOIDataset(prompt_type=templates, N=count, tokenizer=model.tokenizer,
                        prepend_bos=False, seed=seed, device="cpu")
    corrupt = dataset.gen_flipped_prompts("ABB->XYZ, BAB->XYZ")
    if dataset.toks.shape != corrupt.toks.shape:
        raise ValueError("IOI and ABC sequences must have matching padded shapes")
    answers = []
    for prompt in dataset.ioi_prompts:
        ids = [model.tokenizer.encode(" " + prompt[key], add_special_tokens=False)
               for key in ("IO", "S")]
        if any(len(value) != 1 for value in ids) or ids[0] == ids[1]:
            raise ValueError("the fixed IO/S labels must be distinct single tokens")
        answers.append([ids[0][0], ids[1][0]])
    clean_sequences = [model.tokenizer.encode(text, add_special_tokens=False) for text in dataset.sentences]
    corrupt_sequences = [model.tokenizer.encode(text, add_special_tokens=False) for text in corrupt.sentences]
    lengths = [len(sequence) for sequence in clean_sequences]
    corrupt_lengths = [len(sequence) for sequence in corrupt_sequences]
    # Original IOIDataset stores the final answer too and scores the preceding
    # position. Strip that answer at this boundary, then right-pad the prefixes.
    ends = t.tensor(lengths) - 2
    if lengths != corrupt_lengths or not t.equal(ends, dataset.word_idx["end"].cpu()):
        raise ValueError("paired lengths and original pre-answer positions must agree")
    if any(sequence[-1] != answer[0] for sequence, answer in zip(clean_sequences, answers, strict=True)):
        raise ValueError("the original final token must match the fixed IO label")
    width = max(lengths) - 1
    clean_tokens = t.full((count, width), model.tokenizer.pad_token_id, dtype=t.long)
    corrupt_tokens = t.full_like(clean_tokens, model.tokenizer.pad_token_id)
    for row, (clean_sequence, corrupt_sequence) in enumerate(zip(clean_sequences, corrupt_sequences, strict=True)):
        clean_tokens[row, :len(clean_sequence) - 1] = t.tensor(clean_sequence[:-1])
        corrupt_tokens[row, :len(corrupt_sequence) - 1] = t.tensor(corrupt_sequence[:-1])
    return {"clean_tokens": clean_tokens, "corrupt_tokens": corrupt_tokens,
            "end_positions": ends, "answer_ids": t.tensor(answers),
            "prompts": [model.tokenizer.decode(sequence[:-1]) for sequence in clean_sequences],
            "corrupt_prompts": [model.tokenizer.decode(sequence[:-1]) for sequence in corrupt_sequences],
            "templates": templates, "seed": seed, "shifted_templates": shifted_templates}


@dataclass(frozen=True)
class Receiver:
    name: str
    hook_name: str
    baseline_hook: str
    head: int | None
    source_count: int
    output_source: int | None


@dataclass(frozen=True)
class ResidualGraph:
    sources: tuple[str, ...]
    receivers: tuple[Receiver, ...]
    legal_mask: t.Tensor
    n_layers: int
    n_heads: int

    def edge_name(self, source: int, receiver: int) -> str:
        return f"{self.sources[source]} -> {self.receivers[receiver].name}"


def build_residual_graph(n_layers: int, n_heads: int) -> ResidualGraph:
    """Construct the complete legal residual graph for sequential attention/MLP blocks."""
    if n_layers < 1 or n_heads < 1:
        raise ValueError("positive layer and head counts are required")
    sources = ["input"]
    receivers = []
    for layer in range(n_layers):
        prefix = len(sources)
        sources.extend(f"L{layer}H{head}" for head in range(n_heads))
        for port in ("q", "k", "v"):
            for head in range(n_heads):
                receivers.append(Receiver(
                    f"L{layer}H{head}.{port}", f"blocks.{layer}.hook_{port}_input",
                    f"blocks.{layer}.hook_resid_pre", head, prefix, prefix + head,
                ))
        receivers.append(Receiver(
            f"L{layer}MLP.in", f"blocks.{layer}.hook_mlp_in",
            f"blocks.{layer}.hook_resid_mid", None, len(sources), len(sources),
        ))
        sources.append(f"L{layer}MLP")
    receivers.append(Receiver(
        "output", f"blocks.{n_layers - 1}.hook_resid_post",
        f"blocks.{n_layers - 1}.hook_resid_post", None, len(sources), None,
    ))
    legal = t.zeros(len(sources), len(receivers), dtype=t.bool)
    for column, receiver in enumerate(receivers):
        legal[:receiver.source_count, column] = True
    return ResidualGraph(tuple(sources), tuple(receivers), legal, n_layers, n_heads)


def reconstruct_receiver(corrupt_receiver: t.Tensor, sender_deltas: t.Tensor,
                         active_edges: t.Tensor) -> t.Tensor:
    """Corrupt receiver plus retained current-hybrid minus corrupt sender messages.

    sender_deltas is [batch, position, sender, d_model]. active_edges is
    [sender] for a scalar receiver or [sender, head] for a grouped Q/K/V hook.
    """
    if sender_deltas.ndim != 4 or active_edges.ndim not in (1, 2):
        raise ValueError("sender deltas need four axes and mask needs one or two")
    if active_edges.shape[0] != sender_deltas.shape[2]:
        raise ValueError("one mask row is required for each sender")
    if corrupt_receiver.shape != sender_deltas.shape[:2] + sender_deltas.shape[-1:]:
        raise ValueError("corrupt receiver must have [batch, position, d_model] shape")
    weights = active_edges.to(sender_deltas)
    if active_edges.ndim == 1:
        return corrupt_receiver + t.einsum("bpsd,s->bpd", sender_deltas, weights)
    return corrupt_receiver[:, :, None, :] + t.einsum("bpsd,sh->bphd", sender_deltas, weights)


def final_position_logits(model, residual: t.Tensor, end_positions: t.Tensor) -> t.Tensor:
    """Unembed only each example's actual end position, preserving right padding."""
    if end_positions.shape != (len(residual),):
        raise ValueError("provide one end position per example")
    end_positions = end_positions.to(residual.device)
    if bool(((end_positions < 0) | (end_positions >= residual.shape[1])).any()):
        raise ValueError("end positions must lie inside the token sequence")
    last = residual[t.arange(len(residual), device=residual.device), end_positions]
    return model.unembed(model.ln_final(last[:, None, :]))[:, 0]


def answer_logit_differences(logits: t.Tensor, answer_ids: t.Tensor) -> t.Tensor:
    if logits.ndim != 2 or answer_ids.shape != (len(logits), 2):
        raise ValueError("use final-position logits and [correct, distractor] token IDs")
    scores = logits.gather(1, answer_ids.to(logits.device))
    return scores[:, 0] - scores[:, 1]


def clean_kl_divergence(logits: t.Tensor, clean_logits: t.Tensor) -> t.Tensor:
    """Mean KL(clean || circuit) over examples and the entire output vocabulary."""
    if logits.shape != clean_logits.shape or logits.ndim != 2:
        raise ValueError("clean and circuit logits need matching final-position matrices")
    clean_logp = clean_logits.log_softmax(-1)
    return (clean_logp.exp() * (clean_logp - logits.log_softmax(-1))).sum(-1).mean()


class EdgeIntervention:
    """Execute a real hybrid graph; retained senders are recomputed in this run."""
    def __init__(self, model, clean_tokens: t.Tensor, corrupt_tokens: t.Tensor,
                 end_positions: t.Tensor):
        if model.cfg.attn_only or model.cfg.parallel_attn_mlp:
            raise ValueError("this graph requires sequential attention then MLP blocks")
        if getattr(model.cfg, "n_key_value_heads", None) is not None:
            raise ValueError("this lesson's graph uses equal query/key/value head counts")
        if clean_tokens.shape != corrupt_tokens.shape or clean_tokens.ndim != 2:
            raise ValueError("matched clean/corrupt token matrices must have the same shape")
        model.set_use_attn_result(True)
        model.set_use_split_qkv_input(True)
        model.set_use_hook_mlp_in(True)
        self.model = model.eval()
        self.graph = build_residual_graph(model.cfg.n_layers, model.cfg.n_heads)
        device = next(model.parameters()).device
        self.clean_tokens = clean_tokens.to(device)
        self.corrupt_tokens = corrupt_tokens.to(device)
        self.end_positions = end_positions.to(device)
        self.active_mask = self.graph.legal_mask.to(device=device, dtype=t.float32)
        names = {receiver.baseline_hook for receiver in self.graph.receivers}
        names.add("blocks.0.hook_resid_pre")
        for layer in range(self.graph.n_layers):
            names.update((f"blocks.{layer}.attn.hook_result", f"blocks.{layer}.hook_mlp_out"))
        with t.inference_mode():
            residual, cache = model.run_with_cache(
                self.corrupt_tokens, names_filter=lambda name: name in names,
                stop_at_layer=self.graph.n_layers, return_type=None,
            )
            self.corrupt_logits = final_position_logits(model, residual, self.end_positions)
            self.corrupt_cache = {name: value.detach().clone() for name, value in cache.items()}
            pieces = [self.corrupt_cache["blocks.0.hook_resid_pre"][:, :, None]]
            for layer in range(self.graph.n_layers):
                pieces.extend((self.corrupt_cache[f"blocks.{layer}.attn.hook_result"],
                               self.corrupt_cache[f"blocks.{layer}.hook_mlp_out"][:, :, None]))
            self.corrupt_sources = t.cat(pieces, dim=2)
            self.sender_deltas = t.empty_like(self.corrupt_sources)
            residual = model(self.clean_tokens, stop_at_layer=self.graph.n_layers, return_type=None)
            self.clean_logits = final_position_logits(model, residual, self.end_positions)
        self.hooks = self.make_hooks()

    def make_hooks(self) -> list:
        """Install source caches before receiver reconstruction at each causal stage."""
        hooks = []

        def source_hook(start, count, has_head_axis):
            def save(activation, hook):
                del hook
                values = activation if has_head_axis else activation[:, :, None, :]
                self.sender_deltas[:, :, start:start + count] = (
                    values - self.corrupt_sources[:, :, start:start + count]
                )
                return activation
            return save

        hooks.append(("blocks.0.hook_resid_pre", source_hook(0, 1, False)))
        for layer in range(self.graph.n_layers):
            start = 1 + layer * (self.graph.n_heads + 1)
            hooks.append((f"blocks.{layer}.attn.hook_result", source_hook(start, self.graph.n_heads, True)))
            hooks.append((f"blocks.{layer}.hook_mlp_out", source_hook(start + self.graph.n_heads, 1, False)))

        groups = {}
        for column, receiver in enumerate(self.graph.receivers):
            groups.setdefault(receiver.hook_name, []).append(column)
        for hook_name, columns in groups.items():
            receiver = self.graph.receivers[columns[0]]

            def receive(activation, hook, columns=tuple(columns), receiver=receiver):
                del activation, hook
                mask = self.active_mask[:receiver.source_count, list(columns)]
                if receiver.head is None:
                    mask = mask[:, 0]
                return reconstruct_receiver(
                    self.corrupt_cache[receiver.baseline_hook],
                    self.sender_deltas[:, :, :receiver.source_count], mask,
                )
            hooks.append((hook_name, receive))
        return hooks

    @t.inference_mode()
    def __call__(self, active_mask: t.Tensor) -> t.Tensor:
        if active_mask.shape != self.graph.legal_mask.shape:
            raise ValueError("mask shape must describe the full candidate graph")
        # Validate on CPU before a single device transfer, rather than syncing every hook.
        mask = active_mask.detach().cpu()
        if bool((mask[~self.graph.legal_mask] != 0).any()):
            raise ValueError("mask contains a backward or otherwise illegal edge")
        if not bool(t.isfinite(mask.float()).all()) or bool(((mask < 0) | (mask > 1)).any()):
            raise ValueError("edge weights must lie between zero and one")
        self.active_mask = mask.to(device=self.clean_tokens.device, dtype=self.corrupt_sources.dtype)
        residual = self.model.run_with_hooks(
            self.clean_tokens, fwd_hooks=self.hooks,
            stop_at_layer=self.graph.n_layers, return_type=None,
        )
        return final_position_logits(self.model, residual, self.end_positions)


def output_reachable_sources(graph: ResidualGraph, active_mask: t.Tensor) -> set[int]:
    """Backward reachability through additive edges and fixed component computations."""
    active = active_mask.bool().cpu()
    pending = [len(graph.receivers) - 1]
    visited_receivers, sources = set(), set()
    inputs_by_source = {}
    for index, receiver in enumerate(graph.receivers):
        if receiver.output_source is not None:
            inputs_by_source.setdefault(receiver.output_source, []).append(index)
    while pending:
        receiver = pending.pop()
        if receiver in visited_receivers:
            continue
        visited_receivers.add(receiver)
        for source in active[:, receiver].nonzero().flatten().tolist():
            if source not in sources:
                sources.add(source)
                pending.extend(inputs_by_source.get(source, ()))
    return sources


def greedy_edge_acdc(graph: ResidualGraph, evaluate_loss: Callable[[t.Tensor], float], *,
                     threshold: float, progress_every: int = 100,
                     checkpoint_path: str | Path | None = None,
                     resume: dict | None = None) -> dict:
    """Reverse-topological edge deletion, evaluating every trial in the current graph."""
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and nonnegative")
    if progress_every < 1:
        raise ValueError("progress_every must be positive")
    if resume is not None and (resume["threshold"] != threshold or resume["initial_edges"] != int(graph.legal_mask.sum())):
        raise ValueError("resume must use the same threshold and candidate graph")
    active = graph.legal_mask.clone() if resume is None else mask_from_result(graph, resume)
    current = float(evaluate_loss(active))
    if not math.isfinite(current):
        raise ValueError("initial graph loss must be finite")
    initial = current if resume is None else resume["initial_loss"]
    trace = [] if resume is None else list(resume["trace"])
    skipped = [] if resume is None else list(resume["skipped"])
    processed = {(row["source"], row["receiver"]) for row in trace}
    skipped_receivers = {row["receiver"] for row in skipped}
    prior_elapsed = 0. if resume is None else resume.get("elapsed_seconds", 0.)
    started = time.monotonic()
    for receiver_index in reversed(range(len(graph.receivers))):
        receiver = graph.receivers[receiver_index]
        if receiver.name in skipped_receivers:
            continue
        if receiver.output_source is not None and receiver.output_source not in output_reachable_sources(graph, active):
            count = int(active[:, receiver_index].sum())
            active[:, receiver_index] = False
            skipped.append({"receiver": receiver.name, "edges_removed": count})
            continue
        for source in reversed(range(receiver.source_count)):
            if not active[source, receiver_index] or (source, receiver_index) in processed:
                continue
            before = current
            active[source, receiver_index] = False
            trial = float(evaluate_loss(active))
            if not math.isfinite(trial):
                raise ValueError(f"nonfinite loss for {graph.edge_name(source, receiver_index)}")
            damage = trial - before
            remove = damage < threshold
            if remove:
                current = trial
            else:
                active[source, receiver_index] = True
            trace.append({"source": source, "receiver": receiver_index,
                          "edge": graph.edge_name(source, receiver_index),
                          "before_loss": before, "trial_loss": trial,
                          "damage": damage, "removed": remove})
            if len(trace) % progress_every == 0:
                print(f"{len(trace)} edge trials; {int(active.sum())} retained; "
                      f"loss {current:.6f}; {time.monotonic() - started:.1f}s", flush=True)
                if checkpoint_path is not None:
                    checkpoint = {
                        "threshold": threshold, "trace": trace, "skipped": skipped,
                        "active_edges": active.nonzero().tolist(), "complete": False,
                        "initial_edges": int(graph.legal_mask.sum()), "initial_loss": initial,
                        "elapsed_seconds": prior_elapsed + time.monotonic() - started,
                    }
                    temporary = Path(str(checkpoint_path) + ".tmp")
                    temporary.write_text(json.dumps(checkpoint))
                    temporary.replace(checkpoint_path)
    result = {"threshold": threshold, "initial_loss": initial, "final_loss": current,
              "trace": trace, "skipped": skipped, "active_edges": active.nonzero().tolist(),
              "initial_edges": int(graph.legal_mask.sum()), "retained_edges": int(active.sum()),
              "elapsed_seconds": prior_elapsed + time.monotonic() - started, "complete": True}
    if checkpoint_path is not None:
        Path(checkpoint_path).write_text(json.dumps(result))
    return result


def mask_from_result(graph: ResidualGraph, result: dict) -> t.Tensor:
    mask = t.zeros_like(graph.legal_mask)
    if result["active_edges"]:
        indices = t.tensor(result["active_edges"])
        mask[indices[:, 0], indices[:, 1]] = True
    return mask
