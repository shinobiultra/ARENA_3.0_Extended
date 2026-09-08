"""Paired measurement and development-only selection for the 9.1 learner."""
from __future__ import annotations


def paired_marker_change(baseline: list[dict], intervention: list[dict], *, refusal_label: bool) -> dict:
    """Align by source ID and retain both directions of each marker transition."""
    left = {row["id"]: row for row in baseline}
    right = {row["id"]: row for row in intervention}
    if len(left) != len(baseline) or len(right) != len(intervention) or left.keys() != right.keys():
        raise ValueError("Paired comparisons require the same unique source IDs")
    if any(left[key]["refusal_label"] != right[key]["refusal_label"] for key in left):
        raise ValueError("Prompt categories cannot change between interventions")
    ids = [key for key in left if left[key]["refusal_label"] is refusal_label]
    if not ids:
        raise ValueError("The requested category has no examples")
    before = [bool(left[key]["refusal_marker"]) for key in ids]
    after = [bool(right[key]["refusal_marker"]) for key in ids]
    raised = sum(not a and b for a,b in zip(before, after, strict=True))
    lowered = sum(a and not b for a,b in zip(before, after, strict=True))
    positive = sum(before); negative = len(ids)-positive
    return {"n": len(ids), "baseline_rate": positive/len(ids), "intervention_rate": sum(after)/len(ids),
            "delta": (raised-lowered)/len(ids), "new_markers": raised, "removed_markers": lowered,
            "initially_without_marker": negative, "initially_with_marker": positive,
            "conditional_new_rate": raised/negative if negative else None,
            "conditional_removed_rate": lowered/positive if positive else None}


def select_development_direction(candidates: list[dict]) -> dict:
    """Apply the complete declared 15-candidate rule without final-set scores."""
    expected = {(layer, offset) for layer in (4,8,12,16,20) for offset in (-3,-2,-1)}
    pairs = [(row["layer"], row["offset"]) for row in candidates]
    if len(pairs) != len(expected) or set(pairs) != expected:
        raise ValueError("Selection requires all fifteen unique development candidates")
    annotated = []
    for candidate in candidates:
        if candidate.get("evaluation_split") != "development":
            raise ValueError("Candidate selection may only use development measurements")
        row = dict(candidate)
        row["eligible"] = (row["development_auroc"] >= .8 and row["addition_increase"] >= .125
                           and row["ablation_decrease"] >= .25 and row["capability_drop"] <= .125)
        row["selection_score"] = row["addition_increase"] + row["ablation_decrease"]
        annotated.append(row)
    eligible = [row for row in annotated if row["eligible"]]
    chosen = max(eligible, key=lambda row: (row["selection_score"], -row["layer"], -row["offset"])) if eligible else None
    return {"selected": chosen, "eligible_count": len(eligible), "candidates": annotated}
