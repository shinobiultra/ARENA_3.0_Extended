"""Pinned provenance, split integrity and explicit label-corruption checks."""
from pathlib import Path
import json

import numpy as np
import torch
from sklearn.datasets import load_digits

from chapter16_shapley_attribution_baselines.exercises.part9_influence_and_tracin import attribution_data as data


def test_source_test_set_parity():
    source = data.source_arrays()["optdigits.tes"]
    bundled = load_digits()
    assert np.array_equal(source[:, :64], bundled.data), "The original UCI test pixels must match the independently bundled sklearn copy."
    assert np.array_equal(source[:, -1], bundled.target), "Source test labels must match the independent bundled copy."


def test_prospective_manifest_and_splits():
    splits = data.load_study_data()
    recorded = json.loads((Path(__file__).parent / "data_manifest.json").read_text())
    assert data.data_manifest(splits) == recorded, "Do not silently replace the prospective row IDs, labels or preprocessing."
    assert set(splits["train"]["ids"]).isdisjoint(splits["ranking"]["ids"]), "Ranking and training must use different source rows."
    assert splits["train"]["source_file"] == splits["ranking"]["source_file"] == "optdigits.tra"
    assert splits["final"]["source_file"] == "optdigits.tes", "Final queries must come from the original test-writer group."
    all_pixels = []
    for name, expected_count in [("train", 128), ("ranking", 48), ("final", 64)]:
        split = splits[name]
        assert split["x"].shape == (expected_count, 65)
        assert torch.equal(split["x"][:, -1], torch.ones(expected_count, dtype=torch.float64)), "Keep the explicit bias coordinate."
        assert torch.equal(split["x"][:, :-1], split["images"].flatten(1)), "Model inputs and displayed images must be the same pixels."
        all_pixels.extend(data.sha256(row.numpy().tobytes()) for row in split["x"][:, :-1])
    assert len(set(all_pixels)) == len(all_pixels), "An exact duplicate image would need explicit disclosure in the split protocol."


def test_controlled_labels_and_probe_selection():
    splits = data.load_study_data()
    train = splits["train"]
    assert int(train["flipped"].sum()) == 12
    assert torch.equal(train["corrupted_y"] != train["y"], train["flipped"]), "Preserve original labels and identify every deliberately changed row."
    for label in (0, 1):
        assert int((train["flipped"] & (train["y"] == label)).sum()) == 6
    probes = splits["upweight_probe_indices"]
    assert len(set(probes)) == 12
    for label in (0, 1):
        for flipped in (False, True):
            assert sum(int(train["y"][i]) == label and bool(train["flipped"][i]) == flipped for i in probes) == 3
