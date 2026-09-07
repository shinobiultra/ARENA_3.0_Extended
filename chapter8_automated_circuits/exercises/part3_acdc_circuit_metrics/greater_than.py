"""A second fixed task: predicting later two-digit endings of year spans.

The task and 01 corruption follow Hanna et al. (2023) and the ACDC repository.
Token boundaries are checked from this tokenizer rather than assumed at a
particular absolute position. The suffix metric remains learner implemented.
"""
from __future__ import annotations

import torch as t

NOUNS = ("war", "campaign", "project", "trial", "treaty", "reign", "conflict",
         "dynasty", "siege", "journey", "occupation", "investigation", "construction",
         "competition", "relationship", "expedition")


def prepare_greater_than_data(model, *, count: int, seed: int, split: str = "discovery") -> dict:
    """Provided prompt generation with disjoint discovery/development instances."""
    if split not in ("discovery", "development", "heldout") or count < 1:
        raise ValueError("choose a named split and a positive example count")
    tokenizer = model.tokenizer
    def encode(text):
        return tokenizer.encode(text, add_special_tokens=False)
    suffixes = [encode(f"{value:02d}") for value in range(100)]
    if any(len(ids) != 1 for ids in suffixes) or len({ids[0] for ids in suffixes}) != 100:
        raise ValueError("all 00..99 labels must be distinct single tokens")
    suffix_ids = t.tensor([ids[0] for ids in suffixes])
    years = []
    for century in range(11, 18):
        prefix = encode(f" {century}")
        for value in range(2, 99):
            year = century * 100 + value
            if len(prefix) == 1 and encode(f" {year}") == prefix + [int(suffix_ids[value])]:
                years.append(year)
    pairs = [(noun, year) for noun in NOUNS for year in years]
    if split != "heldout":
        pairs = [pair for index, pair in enumerate(pairs)
                 if (index % 5 < 3 if split == "discovery" else index % 5 >= 3)]
    if count > len(pairs):
        raise ValueError("requested more unique prompts than the declared split contains")
    chosen = t.randperm(len(pairs), generator=t.Generator().manual_seed(seed))[:count]
    template = ("The {noun} lasted from {year} to {century}" if split == "heldout"
                else "The {noun} lasted from the year {year} to {century}")
    clean_sequences, corrupt_sequences, prompts, corrupt_prompts = [], [], [], []
    starts, changed_positions = [], []
    for index in chosen.tolist():
        noun, year = pairs[index]
        century, start = divmod(year, 100)
        prompt = template.format(noun=noun, year=year, century=century)
        corrupt_prompt = template.format(noun=noun, year=century * 100 + 1, century=century)
        clean, corrupt = encode(prompt), encode(corrupt_prompt)
        if len(clean) != len(corrupt):
            raise ValueError("matched year spans must have identical token lengths")
        changed = [position for position, (a, b) in enumerate(zip(clean, corrupt, strict=True)) if a != b]
        if len(changed) != 1 or clean[changed[0]] != int(suffix_ids[start]) or corrupt[changed[0]] != int(suffix_ids[1]):
            raise ValueError("the only changed token must be the start-year suffix replaced by 01")
        clean_sequences.append(clean)
        corrupt_sequences.append(corrupt)
        prompts.append(prompt)
        corrupt_prompts.append(corrupt_prompt)
        starts.append(start)
        changed_positions.append(changed[0])
    width = max(map(len, clean_sequences))
    clean_tokens = t.full((count, width), tokenizer.pad_token_id, dtype=t.long)
    corrupt_tokens = t.full_like(clean_tokens, tokenizer.pad_token_id)
    for row, (clean, corrupt) in enumerate(zip(clean_sequences, corrupt_sequences, strict=True)):
        clean_tokens[row, :len(clean)] = t.tensor(clean)
        corrupt_tokens[row, :len(corrupt)] = t.tensor(corrupt)
    return {"clean_tokens": clean_tokens, "corrupt_tokens": corrupt_tokens,
            "end_positions": t.tensor([len(sequence) - 1 for sequence in clean_sequences]),
            "start_suffixes": t.tensor(starts), "suffix_ids": suffix_ids,
            "changed_positions": changed_positions, "prompts": prompts,
            "corrupt_prompts": corrupt_prompts, "template": template,
            "seed": seed, "split": split, "shifted_templates": split == "heldout"}


def greater_than_probability_difference(logits: t.Tensor, suffix_ids: t.Tensor,
                                        start_suffixes: t.Tensor) -> t.Tensor:
    """P(later two-digit suffix) minus P(earlier/equal suffix), per example.

    Probabilities are normalized over the complete vocabulary. Other words
    contribute neither sign, so do not renormalize onto numeric tokens.
    """
    if logits.ndim != 2 or suffix_ids.shape != (100,) or start_suffixes.shape != (len(logits),):
        raise ValueError("provide final logits, 100 ordered suffix IDs and one start suffix per example")
    if bool(((start_suffixes < 0) | (start_suffixes > 99)).any()):
        raise ValueError("start suffixes must be between 00 and 99")
    probabilities = logits.softmax(-1)[:, suffix_ids.to(logits.device)]
    later = t.arange(100, device=logits.device)[None, :] > start_suffixes.to(logits.device)[:, None]
    signs = t.where(later, 1., -1.).to(probabilities)
    return (probabilities * signs).sum(-1)


def greater_than_behavior(logits, clean_logits, corrupt_logits, suffix_ids, start_suffixes) -> dict:
    from .real_edges import clean_kl_divergence

    current = greater_than_probability_difference(logits, suffix_ids, start_suffixes)
    clean = greater_than_probability_difference(clean_logits, suffix_ids, start_suffixes)
    corrupt = greater_than_probability_difference(corrupt_logits, suffix_ids, start_suffixes)
    gap = float(clean.mean() - corrupt.mean())
    if abs(gap) < 1e-8:
        raise ValueError("the clean/corrupt task gap must be nonzero")
    return {"mean_probability_difference": float(current.mean()),
            "clean_mean_probability_difference": float(clean.mean()),
            "corrupt_mean_probability_difference": float(corrupt.mean()),
            "recovery": float((current.mean() - corrupt.mean()) / gap),
            "fraction_positive_probability_difference": float((current > 0).float().mean()),
            "numeric_probability_mass": float(logits.softmax(-1)[:, suffix_ids.to(logits.device)].sum(-1).mean()),
            "kl_clean_to_circuit": float(clean_kl_divergence(logits, clean_logits)),
            "per_example_probability_difference": current.tolist()}
