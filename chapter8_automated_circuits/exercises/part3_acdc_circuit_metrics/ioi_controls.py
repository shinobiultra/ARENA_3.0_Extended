"""Provided matched IOI corruption controls; clean prompts and labels stay fixed."""
from __future__ import annotations

import torch as t

from .real_edges import prepare_ioi_data


def prepare_ioi_second_subject_data(model, *, count: int, seed: int, shifted_templates: bool = False):
    """Replace only the repeated subject, as in the EAP-IG paper's IOI task.

    Reuse the fresh unrelated name at the corresponding position in the
    original XYZ corruption. Keep the two initially introduced names intact.
    This is a separate intervention reference, not an equivalent spelling of
    replacing every name; compare it explicitly when interpreting circuits.
    """
    data = prepare_ioi_data(model, count=count, seed=seed, shifted_templates=shifted_templates)
    corrupted = data["clean_tokens"].clone()
    positions = []
    for row, (io_id, subject_id) in enumerate(data["answer_ids"].tolist()):
        end = int(data["end_positions"][row])
        occurrences = (data["clean_tokens"][row, :end + 1] == subject_id).nonzero().flatten()
        if len(occurrences) != 2:
            raise ValueError("IOI control needs exactly two subject occurrences before the answer")
        position = int(occurrences[-1])
        replacement = int(data["corrupt_tokens"][row, position])
        if replacement in (io_id, subject_id):
            raise ValueError("the replacement subject must be an unrelated single-token name")
        corrupted[row, position] = replacement
        positions.append(position)
    data["corrupt_tokens"] = corrupted
    data["corrupt_prompts"] = [model.tokenizer.decode(tokens[:int(end) + 1])
                                for tokens, end in zip(corrupted, data["end_positions"], strict=True)]
    data["corruption"] = "second_subject_only"
    data["changed_positions"] = positions
    return data
