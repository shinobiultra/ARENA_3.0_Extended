"""Frozen-model token batching and last-token capture; no CCS fitting is hidden here."""

from __future__ import annotations

import hashlib

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from chapter11_representation_geometry.exercises.part5_contrast_consistent_search.ccs_native_methods import last_token_rows

MODEL_ID = "gpt2"
REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
LAYERS = (4, 8, 12)


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION, local_files_only=True,
        torch_dtype=torch.float32, attn_implementation="eager",
    ).to("cpu").eval().requires_grad_(False)
    return model, tokenizer


def weight_digest(model):
    digest = hashlib.sha256()
    for name, value in model.named_parameters():
        digest.update(name.encode())
        digest.update(value.detach().contiguous().numpy().tobytes())
    return digest.hexdigest()


def encode(tokenizer, texts):
    batch = tokenizer(texts, padding=True, truncation=False, return_tensors="pt", add_special_tokens=False)
    batch["position_ids"] = (batch["attention_mask"].cumsum(-1) - 1).clamp_min(0)
    positions = batch["attention_mask"].sum(-1) - 1
    if (positions < 0).any() or batch["input_ids"].shape[1] > 1024:
        raise ValueError("Use nonempty prompts within the model context length.")
    return batch, positions


@torch.inference_mode()
def capture_texts(model, tokenizer, texts, *, include_logits=False, batch_size=32, gather=last_token_rows):
    if not texts or batch_size < 1:
        raise ValueError("Use a nonempty text list and a positive batch size.")
    features = {layer: [] for layer in LAYERS}
    logits, tokens = [], []
    for start in range(0, len(texts), batch_size):
        batch, positions = encode(tokenizer, texts[start:start + batch_size])
        result = model.transformer(**batch, use_cache=False, output_hidden_states=True)
        for layer in LAYERS:
            features[layer].append(gather(result.hidden_states[layer], batch["attention_mask"]).clone())
        if include_logits:
            logits.append(model.lm_head(gather(result.last_hidden_state, batch["attention_mask"])).clone())
        tokens.extend([row[:int(end) + 1].tolist() for row, end in zip(batch["input_ids"], positions)])
    return {"features": {layer: torch.cat(values) for layer, values in features.items()},
            "logits": torch.cat(logits) if include_logits else None, "tokens": tokens}


def capture_pairs(model, tokenizer, rows, *, gather=last_token_rows):
    suffixes = [tokenizer.encode(value, add_special_tokens=False) for value in (" negative", " positive")]
    if any(len(value) != 1 for value in suffixes):
        raise ValueError("The prescribed candidate answers must each be a single token.")
    prefix = capture_texts(model, tokenizer, [row["prefix"] for row in rows], include_logits=True, gather=gather)
    plus = capture_texts(model, tokenizer, [row["plus"] for row in rows], gather=gather)
    minus = capture_texts(model, tokenizer, [row["minus"] for row in rows], gather=gather)
    for branch, candidate in ((minus, suffixes[0][0]), (plus, suffixes[1][0])):
        for context, full in zip(prefix["tokens"], branch["tokens"]):
            if full != context + [candidate]:
                raise ValueError("A contrast pair changed tokenization before the candidate answer.")
    return {"prefix": prefix, "plus": plus, "minus": minus, "rows": rows,
            "candidate_token_ids": [suffixes[0][0], suffixes[1][0]]}


@torch.inference_mode()
def check_capture(model, tokenizer, texts, *, gather=last_token_rows):
    """Actual full-model parity and single-row padding invariance on development text."""
    captured = capture_texts(model, tokenizer, texts, include_logits=True, gather=gather)
    batch, positions = encode(tokenizer, texts)
    direct = model(**batch, use_cache=False, output_hidden_states=True)
    indices = torch.arange(len(texts))
    expected = direct.logits[indices, positions]
    torch.testing.assert_close(captured["logits"], expected, atol=3e-5, rtol=3e-5)
    for layer in LAYERS:
        torch.testing.assert_close(captured["features"][layer], direct.hidden_states[layer][indices, positions],
                                   atol=3e-5, rtol=3e-5)
    single = [capture_texts(model, tokenizer, [text], include_logits=True, gather=gather) for text in texts]
    single_logits = torch.cat([row["logits"] for row in single])
    torch.testing.assert_close(captured["logits"], single_logits, atol=8e-5, rtol=3e-5)
    feature_error = 0.0
    for layer in LAYERS:
        individual = torch.cat([row["features"][layer] for row in single])
        feature_error = max(feature_error, float((individual - captured["features"][layer]).abs().max()))
        torch.testing.assert_close(captured["features"][layer], individual, atol=8e-5, rtol=3e-5)
    return {"passed": True, "texts": texts, "full_model_logit_error": float((captured["logits"] - expected).abs().max()),
            "padding_logit_error": float((captured["logits"] - single_logits).abs().max()),
            "padding_feature_error": feature_error}
