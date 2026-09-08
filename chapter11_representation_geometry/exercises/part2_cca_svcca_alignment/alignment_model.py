"""Native activation plumbing, shown as provided code in the alignment learner."""

from __future__ import annotations
import torch as t


def encode_rows(tokenizer, rows):
    batch = tokenizer(
        [r["prompt"] for r in rows], return_tensors="pt", padding=True, return_offsets_mapping=True
    )
    offsets = batch.pop("offset_mapping")
    spans = t.tensor([r["span"] for r in rows])
    mask = batch["attention_mask"].bool()
    matches = (offsets[:, :, 0] < spans[:, None, 1]) & (offsets[:, :, 1] > spans[:, None, 0]) & mask
    if not matches.sum(1).eq(1).all():
        raise ValueError("Every source word must align to exactly one real token.")
    positions = matches.long().argmax(1)
    selected = offsets[t.arange(len(rows)), positions]
    if not (selected[:, 0].le(spans[:, 0]) & selected[:, 1].eq(spans[:, 1])).all():
        raise ValueError("Token offsets must cover the entire source word.")
    if not mask[:, -1].all():
        raise ValueError("Use left padding before reading last-token logits.")
    batch["position_ids"] = (batch["attention_mask"].long().cumsum(1) - 1).clamp_min(0)
    return dict(batch), positions


@t.inference_mode()
def capture_rows(model, tokenizer, rows, batch_size=24, position_mode="word"):
    boundaries = (
        [*model.transformer.h, model.transformer.ln_f]
        if model.config.model_type == "gpt2"
        else [*model.gpt_neox.layers, model.gpt_neox.final_layer_norm]
    )
    hidden = []
    logits = []
    for start in range(0, len(rows), batch_size):
        batch, positions = encode_rows(tokenizer, rows[start : start + batch_size])
        cache = {}
        handles = []
        if position_mode == "prediction":
            positions = t.full_like(positions, batch["input_ids"].shape[1] - 1)
        elif position_mode != "word":
            raise ValueError("Choose word or prediction position.")

        def make_hook(index):
            def hook(module, args):
                cache[index] = args[0][t.arange(len(positions)), positions].detach().clone()

            return hook

        try:
            for index, block in enumerate(boundaries):
                handles.append(block.register_forward_pre_hook(make_hook(index)))
            output = model(**batch, use_cache=False).logits[:, -1].float()
        finally:
            for handle in handles:
                handle.remove()
        if len(cache) != len(boundaries):
            raise ValueError("Not every native residual boundary was captured.")
        hidden.append(t.stack([cache[i] for i in range(len(boundaries))], 1))
        logits.append(output)
    return {"activations": t.cat(hidden), "logits": t.cat(logits)}


@t.inference_mode()
def replace_rows(model, tokenizer, rows, layer, replacements, batch_size=24, position_mode="word"):
    boundaries = (
        [*model.transformer.h, model.transformer.ln_f]
        if model.config.model_type == "gpt2"
        else [*model.gpt_neox.layers, model.gpt_neox.final_layer_norm]
    )
    block = boundaries[layer]
    logits = []
    for start in range(0, len(rows), batch_size):
        batch, positions = encode_rows(tokenizer, rows[start : start + batch_size])
        replacement = replacements[start : start + batch_size]
        if position_mode == "prediction":
            positions = t.full_like(positions, batch["input_ids"].shape[1] - 1)
        elif position_mode != "word":
            raise ValueError("Choose word or prediction position.")

        def hook(module, args):
            changed = args[0].clone()
            changed[t.arange(len(positions)), positions] = replacement.to(changed)
            return (changed, *args[1:])

        handle = block.register_forward_pre_hook(hook)
        try:
            output = model(**batch, use_cache=False).logits[:, -1].float()
        finally:
            handle.remove()
        logits.append(output)
    return t.cat(logits)
