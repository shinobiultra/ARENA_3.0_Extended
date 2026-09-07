"""Outcome-independent selection from the pinned public UCI source archive."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from chapter16_shapley_attribution_baselines.exercises.part9_influence_and_tracin.attribution_data import (
    DATA_URL, FILE_SHA256, ZIP_SHA256, source_arrays,
)

CACHE = Path.home() / ".cache/arena_extended/datamodels"


def load_data(cache: Path = CACHE) -> dict:
    arrays = source_arrays(cache)
    data = {}
    for split, filename, count in (("train", "optdigits.tra", 64), ("query", "optdigits.tes", 32)):
        ids = []
        for label in (3, 8):
            candidates = np.flatnonzero(arrays[filename][:, -1] == label).tolist()
            ordered = sorted(candidates, key=lambda row: hashlib.sha256(
                f"arena16.10/v1/{split}/{row}".encode()).hexdigest())
            ids.extend(ordered[:count])
        ids.sort()
        rows = arrays[filename][ids]
        images = torch.from_numpy(rows[:, :64].copy()).double().reshape(-1, 8, 8) / 16
        data[split] = dict(
            images=images,
            x=torch.cat([images.flatten(1), torch.ones(len(ids), 1, dtype=torch.float64)], dim=1),
            y=torch.from_numpy((rows[:, -1] == 8).astype(np.int64)),
            digit=torch.from_numpy(rows[:, -1].copy()),
            ids=[f"{filename}:{row}" for row in ids],
        )
    return data


def data_manifest(data: dict) -> dict:
    splits = {}
    for name, split in data.items():
        splits[name] = dict(
            ids=split["ids"],
            x_sha256=hashlib.sha256(split["x"].numpy().astype("<f8").tobytes()).hexdigest(),
            y_sha256=hashlib.sha256(split["y"].numpy().astype("<i8").tobytes()).hexdigest(),
            class_counts=torch.bincount(split["y"], minlength=2).tolist(),
        )
    duplicate_pairs = torch.nonzero(
        (data["train"]["images"][:, None] == data["query"]["images"][None]).flatten(2).all(2)
    ).tolist()
    return dict(
        source_url=DATA_URL, archive_sha256=ZIP_SHA256, source_sha256=FILE_SHA256,
        citation="Alpaydin and Kaynak (1998), DOI 10.24432/C50P49", license="CC BY 4.0",
        negative_digit=3, positive_digit=8, splits=splits,
        exact_duplicate_image_pairs=[
            [data["train"]["ids"][i], data["query"]["ids"][j]] for i, j in duplicate_pairs
        ],
        preprocessing="Pixel counts / 16, constant bias coordinate; no fitted normalization",
    )
