"""Summary metrics for interchange interventions in the DAS lesson."""

from __future__ import annotations

import torch as t

from chapter11_representation_geometry.exercises.part3_causal_abstraction_das import (
    das_methods as m,
)


def summarize(logits, pairs, rows, names, native_logits):
    targets = t.tensor([names[p["target"]] for p in pairs])
    base = t.tensor([p["base"] for p in pairs])
    source = t.tensor([p["source"] for p in pairs])
    cf = t.tensor([p["counterfactual"] for p in pairs])
    native_correct = native_logits.argmax(-1).eq(t.tensor([names[r["answer"]] for r in rows]))
    changed = t.tensor([p["changed_role"] for p in pairs])
    cross = t.tensor([p["cross_name"] for p in pairs])
    result = {}
    for name, mask in {
        "all": t.ones(len(pairs), dtype=t.bool),
        "changed": changed,
        "cross_changed": cross & changed,
        "cross_unchanged": cross & ~changed,
        "same_changed": ~cross & changed,
        "both_native_correct_cross_changed": native_correct[base]
        & native_correct[cf]
        & cross
        & changed,
    }.items():
        result[name] = dict(n=int(mask.sum()), iia=m.interchange_accuracy(logits, targets, mask))
    probabilities = logits.softmax(-1)
    result["target_probability"] = float(probabilities[t.arange(len(pairs)), targets].mean())
    result["cross_changed_target_probability"] = float(
        probabilities[t.arange(len(pairs)), targets][cross & changed].mean()
    )
    donor_answers = t.tensor([names[rows[i]["answer"]] for i in source])
    result["cross_donor_answer_probability"] = float(
        probabilities[t.arange(len(pairs)), donor_answers][cross].mean()
    )
    result["full_vocabulary_ce"] = float(m.intervention_loss(logits, targets))
    result["native_prompt_accuracy"] = float(native_correct.float().mean())
    result["top_token_ids"] = logits.argmax(-1).tolist()
    return result
