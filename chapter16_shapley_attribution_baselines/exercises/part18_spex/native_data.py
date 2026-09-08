"""Prospective word-mask games: input construction only, never model outcomes."""

from __future__ import annotations

import hashlib
from pathlib import Path
import re

import numpy as np

MODEL_ID = "openai-community/gpt2"
MODEL_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
NAME_PAIRS = (("John", "Mary"), ("Alice", "Bob"), ("Sarah", "Michael"), ("David", "Emma"))
TEMPLATES = (
    "When {a} and {b} went to the small local store in their quiet village on a sunny morning, "
    "{b} bought a carefully wrapped gift and later handed it",
    "After {a} and {b} arrived at the busy railway station in the center of town on Friday morning, "
    "{b} found a small blue umbrella and immediately offered it",
    "When {a} and {b} visited the old town museum with their friends during the long summer holiday, "
    "{b} noticed a folded paper map and then quietly passed it",
    "After {a} and {b} walked through the beautiful village park with their neighbors on a warm Saturday afternoon, "
    "{b} picked up a bright red ball and threw it",
)
FIXED_SUFFIX = " to"
N_FEATURES = 28


def prompt_records():
    """All four templates, four name pairs and both orders; no outcome filtering."""
    records = []
    for template_index, template in enumerate(TEMPLATES):
        for pair_index, pair in enumerate(NAME_PAIRS):
            for order in (0, 1):
                a, b = pair if order == 0 else pair[::-1]
                words = template.format(a=a, b=b).split()
                if len(words) != N_FEATURES:
                    raise ValueError("The declared feature inventory changed.")
                name_positions = [i for i, word in enumerate(words) if word in {a, b}]
                if [words[i] for i in name_positions] != [a, b, b]:
                    raise ValueError("Require the stated two-name IOI construction.")
                counterfactual = words.copy()
                counterfactual[name_positions[-1]] = a
                records.append({
                    "id": f"template{template_index}/pair{pair_index}/order{order}",
                    "template_index": template_index,
                    "pair_index": pair_index,
                    "name_order": order,
                    "name_a": a,
                    "name_b": b,
                    "words": words,
                    "prompt": " ".join(words) + FIXED_SUFFIX,
                    "fixed_suffix": FIXED_SUFFIX,
                    "subject_word_index": name_positions[-1],
                    "counterfactual_prompt": " ".join(counterfactual) + FIXED_SUFFIX,
                    "counterfactual_expected_sign": -1,
                    "primary_expected_sign": 1,
                })
    return records


def tokenize_record(record, tokenizer):
    """Bind each BPE token to one displayed word; suffix tokens stay present."""
    text = record["prompt"]
    spans = [(match.start(), match.end()) for match in re.finditer(r"\S+", text)]
    if len(spans) != N_FEATURES + 1 or text[spans[-1][0]:spans[-1][1]] != "to":
        raise ValueError("Expected 28 feature words followed by the fixed suffix.")
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    groups = []
    for start, end in encoded["offset_mapping"]:
        overlap = [i for i, (left, right) in enumerate(spans) if end > left and start < right]
        if len(overlap) != 1:
            raise ValueError("Each BPE token must overlap exactly one displayed word.")
        groups.append(overlap[0] if overlap[0] < N_FEATURES else -1)
    if set(groups) != set(range(N_FEATURES)) | {-1} or groups[-1] != -1:
        raise ValueError("Feature or fixed-suffix token inventory is incomplete.")
    targets = [tokenizer.encode(" " + record[key], add_special_tokens=False) for key in ("name_a", "name_b")]
    if any(len(target) != 1 for target in targets) or targets[0] == targets[1]:
        raise ValueError("Each candidate answer must be a distinct, single space-prefixed BPE token.")
    return {
        "input_ids": np.asarray(encoded["input_ids"], dtype=np.int64),
        "token_feature": np.asarray(groups, dtype=np.int64),
        "target_ids": np.asarray([target[0] for target in targets], dtype=np.int64),
        "offset_mapping": np.asarray(encoded["offset_mapping"], dtype=np.int64),
    }


def masked_word_inputs(input_ids, token_feature, masks, replacement_id=50256):
    """Keep absolute positions; hide absent word keys and replace their token IDs.

    This intervention is fixed-position attention masking, not natural deletion.
    Feature -1 marks suffix tokens that every coalition retains.
    """
    input_ids = np.asarray(input_ids)
    token_feature = np.asarray(token_feature)
    masks = np.asarray(masks)
    if input_ids.ndim != 1 or token_feature.shape != input_ids.shape:
        raise ValueError("Input IDs and token-to-word groups must be aligned vectors.")
    if not np.issubdtype(input_ids.dtype, np.integer) or not np.issubdtype(token_feature.dtype, np.integer):
        raise ValueError("Token IDs and feature indices must be integers.")
    if masks.ndim != 2 or masks.shape[1] != N_FEATURES or not np.isin(masks, (0, 1)).all():
        raise ValueError("Coalitions must have 28 binary feature columns.")
    if not np.isin(token_feature, np.arange(-1, N_FEATURES)).all() or token_feature[-1] != -1:
        raise ValueError("Unknown word group or missing fixed final suffix.")
    if not np.any(token_feature >= 0) or replacement_id < 0:
        raise ValueError("Require feature tokens and a valid replacement ID.")
    retained = np.ones((len(masks), len(input_ids)), dtype=bool)
    variable = token_feature >= 0
    retained[:, variable] = masks[:, token_feature[variable]].astype(bool)
    ids = np.where(retained, input_ids[None, :], replacement_id).astype(np.int64)
    positions = np.broadcast_to(np.arange(len(input_ids)), ids.shape).copy()
    return {"input_ids": ids, "attention_mask": retained.astype(np.int64), "position_ids": positions}


def mask_ids(masks):
    """Big-endian mask IDs: displayed feature zero is the most significant bit."""
    masks = np.asarray(masks)
    if masks.ndim != 2 or not np.isin(masks, (0, 1)).all() or masks.shape[1] > 63:
        raise ValueError("Require a binary query matrix with at most 63 features.")
    powers = np.left_shift(np.uint64(1), np.arange(masks.shape[1] - 1, -1, -1, dtype=np.uint64))
    return masks.astype(np.uint64) @ powers


def snapshot_records(snapshot):
    """Physical hashes of the pinned local model/tokenizer inputs, not cache names."""
    snapshot = Path(snapshot)
    names = ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")
    result = {}
    for name in names:
        path = snapshot / name
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        result[name] = {"sha256": digest, "size_bytes": path.stat().st_size}
    return result
