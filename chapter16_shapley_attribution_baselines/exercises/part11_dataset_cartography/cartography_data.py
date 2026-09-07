"""Pinned digit sources and outcome-independent splits for 16.11."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from chapter16_shapley_attribution_baselines.exercises.part9_influence_and_tracin.attribution_data import (
    DATA_URL,
    FILE_SHA256,
    ZIP_SHA256,
    source_arrays,
)


def ordered(rows, namespace):
    return sorted(
        rows,
        key=lambda row: hashlib.sha256(f"arena16.11/v1/{namespace}/{row}".encode()).hexdigest(),
    )


def make_split(array, rows, filename):
    rows = sorted(rows)
    values = array[rows]
    return dict(
        x=torch.from_numpy(values[:, :64].copy()).float() / 16,
        y=torch.from_numpy(values[:, -1].copy()),
        ids=[f"{filename}:{row}" for row in rows],
        source_rows=rows,
    )


def training_rows(array):
    rows = []
    for label in range(10):
        candidates = np.flatnonzero(array[:, -1] == label).tolist()
        rows.extend(ordered(candidates, "train")[:300])
    return sorted(rows)


def load_training():
    arrays = source_arrays()
    array = arrays["optdigits.tra"]
    train = make_split(array, training_rows(array), "optdigits.tra")
    changed_rows = set()
    for label in range(10):
        candidates = [row for row in train["source_rows"] if array[row, -1] == label]
        changed_rows.update(ordered(candidates, "noise")[:30])
    changed = torch.tensor([row in changed_rows for row in train["source_rows"]])
    train["changed"] = changed
    train["noisy_y"] = torch.where(changed, (train["y"] + 1) % 10, train["y"])
    generator = torch.Generator().manual_seed(161199)
    train["shuffled_y"] = train["y"][torch.randperm(len(train["y"]), generator=generator)]
    return train


def load_evaluation():
    arrays = source_arrays()
    selected = set(training_rows(arrays["optdigits.tra"]))
    remaining = [row for row in range(3823) if row not in selected]
    return {
        "same_source": make_split(arrays["optdigits.tra"], remaining, "optdigits.tra"),
        "separate_writers": make_split(arrays["optdigits.tes"], range(1797), "optdigits.tes"),
    }


def digest_tensor(tensor):
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def data_manifest():
    train = load_training()
    evaluation = load_evaluation()
    splits = {"train": train, **evaluation}
    signatures = {
        name: {row.numpy().tobytes() for row in split["x"]} for name, split in splits.items()
    }
    return dict(
        source_url=DATA_URL,
        archive_sha256=ZIP_SHA256,
        source_sha256=FILE_SHA256,
        citation="Alpaydin and Kaynak (1998), UCI Optical Recognition of Handwritten Digits, DOI 10.24432/C50P49",
        license="CC BY 4.0",
        preprocessing="Each original integer pixel count / 16; no fitted preprocessing",
        splits={
            name: dict(
                ids=split["ids"],
                x_sha256=digest_tensor(split["x"]),
                y_sha256=digest_tensor(split["y"]),
                class_counts=torch.bincount(split["y"], minlength=10).tolist(),
            )
            for name, split in splits.items()
        },
        exact_distinct_pixel_overlaps={
            name: len(signatures["train"] & signatures[name]) for name in evaluation
        },
        changed_ids=[value for value, flag in zip(train["ids"], train["changed"]) if flag],
        noisy_y_sha256=digest_tensor(train["noisy_y"]),
        shuffled_y_sha256=digest_tensor(train["shuffled_y"]),
        writer_scope="Original separate-writer test split; individual writer IDs unavailable. Same-source holdout is not writer-disjoint by construction.",
    )
