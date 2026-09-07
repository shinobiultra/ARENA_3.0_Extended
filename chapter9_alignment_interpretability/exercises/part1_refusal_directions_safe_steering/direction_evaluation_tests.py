"""Semantic checks for paired denominators and frozen development selection."""
from __future__ import annotations

import pytest

from . import direction_evaluation as reference


def test_paired_marker_change(fn=reference.paired_marker_change):
    baseline = [{"id": str(i), "refusal_label": False, "refusal_marker": flag}
                for i,flag in enumerate((False,False,True,True))]
    after = [{**row,"refusal_marker":flag} for row,flag in zip(baseline,(True,True,False,True),strict=True)]
    out = fn(baseline,list(reversed(after)),refusal_label=False)
    assert out["n"] == 4 and out["baseline_rate"] == .5 and out["intervention_rate"] == .75
    assert out["new_markers"] == 2 and out["removed_markers"] == 1 and out["delta"] == .25
    assert out["conditional_new_rate"] == 1 and out["conditional_removed_rate"] == .5
    with pytest.raises(ValueError):fn(baseline,after[:-1],refusal_label=False)
    with pytest.raises(ValueError):fn(baseline,[after[0],after[0],*after[2:]],refusal_label=False)
    print("ID alignment and unconditional/conditional transition tests passed!")


def test_paired_marker_zero_denominator(fn=reference.paired_marker_change):
    rows=[{"id": str(i),"refusal_label":False,"refusal_marker":False} for i in range(3)]
    out=fn(rows,rows,refusal_label=False)
    assert out["conditional_removed_rate"] is None and out["conditional_new_rate"]==0
    assert out["delta"]==0
    with pytest.raises(ValueError):fn(rows,rows,refusal_label=True)
    print("Undefined conditional denominators remain explicit!")


def test_select_development_direction(fn=reference.select_development_direction):
    candidates=[{"layer":layer,"offset":offset,"evaluation_split":"development","development_auroc":.7,
                 "addition_increase":0.,"ablation_decrease":0.,"capability_drop":0.}
                for layer in (4,8,12,16,20) for offset in (-3,-2,-1)]
    assert fn(candidates)["selected"] is None
    for index in (0,1,3):
        candidates[index].update(development_auroc=.8,addition_increase=.125,ablation_decrease=.25,capability_drop=.125)
    out=fn(candidates)
    assert out["eligible_count"]==3
    assert (out["selected"]["layer"],out["selected"]["offset"])==(4,-3)
    assert all("eligible" not in row for row in candidates)
    candidates[-1].update(development_auroc=1.,addition_increase=1.,ablation_decrease=1.,capability_drop=.5)
    assert fn(candidates)["selected"]["layer"]==4,"A larger marker effect cannot erase its capability failure"
    with pytest.raises(ValueError):fn(candidates[:-1])
    candidates[0]["evaluation_split"]="final"
    with pytest.raises(ValueError,match="only use development"):fn(candidates)
    print("Complete-candidate selection, ties, failures and split isolation passed!")
