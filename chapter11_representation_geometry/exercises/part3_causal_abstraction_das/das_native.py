"""Frozen native GPT-2 plumbing with a differentiable cached continuation.

Caching skips the prefix only. Every remaining GPT-2 block, its native captured
attention mask, final layer norm and vocabulary readout still run. No inference
mode wraps the continuation: the learner's subspace must receive gradients.
"""

from __future__ import annotations

from dataclasses import dataclass
import torch as t


def encode_rows(tokenizer, rows):
    batch = tokenizer(
        [row["prompt"] for row in rows],
        padding=True,
        return_tensors="pt",
        return_offsets_mapping=True,
    )
    offsets = batch.pop("offset_mapping")
    spans = t.tensor([row["span"] for row in rows])
    real = batch["attention_mask"].bool()
    matches = (offsets[:, :, 0] < spans[:, None, 1]) & (offsets[:, :, 1] > spans[:, None, 0]) & real
    if not matches.sum(1).eq(1).all() or not real[:, -1].all():
        raise ValueError("Use left padding and exactly one token per repeated-name span.")
    positions = matches.long().argmax(1)
    selected = offsets[t.arange(len(rows)), positions]
    if not (selected[:, 0].le(spans[:, 0]) & selected[:, 1].eq(spans[:, 1])).all():
        raise ValueError("The selected token must cover the complete repeated name.")
    batch["position_ids"] = (batch["attention_mask"].cumsum(1) - 1).clamp_min(0)
    return dict(batch), positions


def require_frozen(model):
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("DAS requires an eval-mode model with every weight frozen.")
    if model.config.model_type != "gpt2" or model.config._attn_implementation != "eager":
        raise ValueError("This native continuation is tested for GPT-2 eager attention.")


@dataclass
class NativeCache:
    tokens: dict
    subject_positions: t.Tensor
    hidden: dict[int, t.Tensor]
    causal_mask: t.Tensor
    logits: t.Tensor

    def positions(self, mode):
        if mode == "subject":
            return self.subject_positions
        if mode == "end":
            return t.full_like(self.subject_positions, self.tokens["input_ids"].shape[1] - 1)
        raise ValueError("Choose subject or end.")

    def activations(self, layer, mode):
        return self.hidden[layer][t.arange(len(self.subject_positions)), self.positions(mode)]


@t.no_grad()
def capture_native(model, tokenizer, rows, layers=(3, 6, 9), batch_size=16):
    """Capture real block inputs and the exact eager mask from native forwards.

    no_grad (not inference_mode) produces ordinary tensors reusable by autograd.
    Encode the entire split together so cached sequence lengths match. Padding
    positions are never eligible intervention sites.
    """
    require_frozen(model)
    if (
        not layers
        or len(set(layers)) != len(layers)
        or any(k not in range(len(model.transformer.h)) for k in layers)
    ):
        raise ValueError("Choose distinct native transformer block inputs.")
    tokens, positions = encode_rows(tokenizer, rows)
    hidden = {layer: [] for layer in layers}
    masks, logits = [], []
    for start in range(0, len(rows), batch_size):
        batch = {key: value[start : start + batch_size] for key, value in tokens.items()}
        captured, handles = {}, []

        def make_hook(layer):
            def hook(module, args, kwargs):
                # See the installed native GPT2Model.forward; these are its
                # actual positional arguments, not a reconstructed mask.
                if (
                    len(args) < 4
                    or args[1] is not None
                    or args[3] is not None
                    or kwargs.get("use_cache")
                ):
                    raise ValueError(
                        "Unexpected native GPT-2 block call; revalidate the continuation."
                    )
                mask = args[2]
                if mask is None or mask.ndim != 4 or mask.shape[0] != len(batch["input_ids"]):
                    raise ValueError("Expected a batch-specific native eager causal mask.")
                captured[layer] = (args[0].detach().clone(), mask.detach().clone())

            return hook

        try:
            for layer in layers:
                handles.append(
                    model.transformer.h[layer].register_forward_pre_hook(
                        make_hook(layer), with_kwargs=True
                    )
                )
            final = model.transformer(**batch, use_cache=False).last_hidden_state[:, -1]
            logits.append(model.lm_head(final).detach())
        finally:
            for handle in handles:
                handle.remove()
        if set(captured) != set(layers):
            raise ValueError("Not every requested native block input was observed.")
        mask = captured[layers[0]][1]
        if any(not t.equal(value[1], mask) for value in captured.values()):
            raise ValueError("Layer-specific attention masks require a different cache.")
        masks.append(mask)
        for layer in layers:
            hidden[layer].append(captured[layer][0])
    return NativeCache(
        tokens, positions, {k: t.cat(v) for k, v in hidden.items()}, t.cat(masks), t.cat(logits)
    )


def continue_native(model, cache, layer, indices, replacement, mode="subject"):
    """Resume native blocks after changing one residual vector in each base.

    Slice assignment into a clone is differentiable with respect to replacement.
    All other token positions retain their cached recipient activations.
    """
    require_frozen(model)
    hidden = cache.hidden[layer][indices].clone()
    if replacement.shape != (len(indices), hidden.shape[-1]):
        raise ValueError("Supply one replacement vector per selected recipient.")
    hidden[t.arange(len(indices)), cache.positions(mode)[indices]] = replacement
    for block in model.transformer.h[layer:]:
        hidden = block(
            hidden,
            None,
            cache.causal_mask[indices],
            None,
            encoder_attention_mask=None,
            use_cache=False,
            position_ids=cache.tokens["position_ids"][indices],
        )
    return model.lm_head(model.transformer.ln_f(hidden)[:, -1])


def full_forward_intervention(model, cache, layer, indices, replacement, mode="subject"):
    """Independent native hook oracle; prefix and full vocabulary head execute."""
    require_frozen(model)
    positions = cache.positions(mode)[indices]

    def hook(module, args):
        hidden = args[0].clone()
        hidden[t.arange(len(indices)), positions] = replacement
        return (hidden, *args[1:])

    handle = model.transformer.h[layer].register_forward_pre_hook(hook)
    try:
        batch = {key: value[indices] for key, value in cache.tokens.items()}
        return model(**batch, use_cache=False).logits[:, -1]
    finally:
        handle.remove()
