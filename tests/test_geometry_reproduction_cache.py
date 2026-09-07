"""Cached native evidence must survive independent inventory and value checks."""

from copy import deepcopy
import hashlib

import pytest

from chapter11_representation_geometry.exercises.part1_pca_svd_geometry_controls.geometry_reproduction_cache import (
    NATIVE_INPUTS,
    compare_rows,
    validate_native_sources,
)


def sample():
    return [
        {
            "family": "weekday",
            "row_id": "final/example",
            "condition": "no_op",
            "draw": None,
            "calendar_logits": [1.0, 2.0],
            "top_token_id": 41,
            "original_answer": 1,
            "desired_answer": 0,
            "restricted_answer": 1,
            "original_full_correct": True,
            "desired_full_correct": False,
            "shift": 2,
        }
    ]


def test_actual_cached_values_and_inventory_are_required():
    rows = sample()
    assert compare_rows(rows, rows) == {
        "compared_examples": 1,
        "max_calendar_logit_error": 0.0,
        "top_token_mismatches": 0,
    }
    for incorrect in ([], rows * 2):
        with pytest.raises(ValueError):
            compare_rows(incorrect, rows)


@pytest.mark.parametrize(
    "field,value",
    [
        ("calendar_logits", [1.0, 2.01]),
        ("calendar_logits", [float("nan"), 2.0]),
        ("top_token_id", 42),
        ("desired_answer", 1),
        ("draw", 0),
        ("desired_full_correct", True),
    ],
)
def test_cached_row_mutations_are_rejected(field, value):
    expected = sample()
    actual = deepcopy(expected)
    actual[0][field] = value
    with pytest.raises(ValueError):
        compare_rows(actual, expected)


def test_all_native_sources_must_match_snapshot_and_current_files(tmp_path):
    source, current = tmp_path / "native", tmp_path / "current"
    (source / "source_snapshot").mkdir(parents=True)
    current.mkdir()
    digests = {}
    for name in NATIVE_INPUTS:
        value = ("original " + name).encode()
        (source / "source_snapshot" / name).write_bytes(value)
        (current / name).write_bytes(value)
        digests[name] = hashlib.sha256(value).hexdigest()
    state = {"source_inputs": digests}
    validate_native_sources(source, state, current)
    missing = deepcopy(state)
    missing["source_inputs"].pop("geometry_methods.py")
    with pytest.raises(ValueError, match="inventory"):
        validate_native_sources(source, missing, current)
    (current / "geometry_methods.py").write_text("changed method")
    with pytest.raises(ValueError, match="scientific source changed"):
        validate_native_sources(source, state, current)
