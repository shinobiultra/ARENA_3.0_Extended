"""Guards reject partial or numerically invalid fresh native evidence."""

import copy
import itertools

import pytest
import torch

from chapter11_representation_geometry.exercises.part5_contrast_consistent_search import ccs_reproduction as reproduction


def complete_result():
    return {
        "cases": [{"variant": v, "wrapper": w, "layer": layer} for v, w, layer in itertools.product(
            reproduction.data.VARIANTS, reproduction.data.WRAPPERS, reproduction.native.LAYERS)],
        "captures": [{"variant": v, "wrapper": w, "split": split} for v, w, split in itertools.product(
            reproduction.data.VARIANTS, reproduction.data.WRAPPERS, reproduction.data.FRAMES)],
        "final_reporter_scores": 98304, "final_prefix_examples": 512,
        "all_prefix_examples_including_play": 3586, "refitted_reporters": 1152,
        "winning_token_mismatches": 0, "weights_unchanged": True, "model_gradients_absent": True,
        "play_reproduced": True, "all_reporters_refitted": True, "changed_sources": [],
        "numerical_comparison": {"compared_tensors": 1, "max_tensor_difference": 0., "max_scalar_difference": 0.},
    }


def test_complete_protocol_cardinality():
    reproduction.validate_complete_result(complete_result())
    assert len(reproduction.expected_artifacts()) == 43


@pytest.mark.parametrize("fault", ["missing_case", "duplicate_case", "missing_split", "wrong_split", "short_scores",
                                  "missing_prefixes", "missing_refits", "wrong_tokens", "changed_weights", "missing_play",
                                  "changed_source", "no_tensors", "nonfinite", "large_error"])
def test_reject_partial_or_invalid_protocol(fault):
    result = complete_result()
    if fault == "missing_case":
        result["cases"].pop()
    elif fault == "duplicate_case":
        result["cases"][-1] = copy.deepcopy(result["cases"][0])
    elif fault == "missing_split":
        result["captures"].pop()
    elif fault == "wrong_split":
        result["captures"][-1]["split"] = "unapproved"
    elif fault == "short_scores":
        result["final_reporter_scores"] -= 1
    elif fault == "missing_prefixes":
        result["all_prefix_examples_including_play"] = 512
    elif fault == "missing_refits":
        result["refitted_reporters"] = 18
    elif fault == "wrong_tokens":
        result["winning_token_mismatches"] = 1
    elif fault == "changed_weights":
        result["weights_unchanged"] = False
    elif fault == "missing_play":
        result["play_reproduced"] = False
    elif fault == "changed_source":
        result["changed_sources"] = ["ccs_methods.py"]
    elif fault == "no_tensors":
        result["numerical_comparison"]["compared_tensors"] = 0
    elif fault == "nonfinite":
        result["numerical_comparison"]["max_tensor_difference"] = float("nan")
    elif fault == "large_error":
        result["numerical_comparison"]["max_tensor_difference"] = .01
    with pytest.raises(ValueError):
        reproduction.validate_complete_result(result)


def test_nested_comparison_rejects_unmatched_evidence():
    expected = {"scores": torch.tensor([.1, .9], dtype=torch.float64), "tokens": [2, 7], "mean": .5}
    checked = reproduction.compare_nested(copy.deepcopy(expected), expected)
    assert checked["compared_tensors"] == 1 and checked["max_tensor_difference"] == 0
    for actual in [
        {**expected, "extra": 1},
        {**expected, "scores": torch.ones(1, 2, dtype=torch.float64)},
        {**expected, "scores": torch.tensor([.1, float("nan")], dtype=torch.float64)},
        {**expected, "tokens": [2, 8]},
        {**expected, "mean": float("inf")},
    ]:
        with pytest.raises(ValueError):
            reproduction.compare_nested(actual, expected)
    with pytest.raises(AssertionError):
        reproduction.compare_nested({**expected, "scores": torch.ones(2, dtype=torch.float64)}, expected)
