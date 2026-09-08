"""Pinned public digit data and outcome-independent splits for section 16.9."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile

import numpy as np
import torch

DATA_URL = (
    "https://archive.ics.uci.edu/static/public/80/"
    "optical%2Brecognition%2Bof%2Bhandwritten%2Bdigits.zip"
)
ZIP_SHA256 = "0d7b054fea010270e9b3f06411c654c5e59547732ad626381980baffe0a23fb0"
FILE_SHA256 = {
    "optdigits.tra": "e1b683cc211604fe8fd8c4417e6a69f31380e0c61d4af22e93cc21e9257ffedd",
    "optdigits.tes": "6ebb3d2fee246a4e99363262ddf8a00a3c41bee6014c373ed9d9216ba7f651b8",
}
DEFAULT_CACHE = Path.home() / ".cache/arena_extended/data_attribution"


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def source_arrays(cache: Path = DEFAULT_CACHE) -> dict[str, np.ndarray]:
    """Download only the pinned archive; check bytes before parsing numeric data."""
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / "uci_optdigits.zip"
    if not path.exists():
        with urllib.request.urlopen(DATA_URL, timeout=60) as response:
            content = response.read()
        if sha256(content) != ZIP_SHA256:
            raise ValueError("The UCI archive differs from the pinned source.")
        path.write_bytes(content)
    content = path.read_bytes()
    if sha256(content) != ZIP_SHA256:
        raise ValueError("The cached UCI archive differs from the pinned source.")
    arrays = {}
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        for filename, expected in FILE_SHA256.items():
            raw = archive.read(filename)
            if sha256(raw) != expected:
                raise ValueError(f"Source bytes changed: {filename}")
            array = np.loadtxt(io.BytesIO(raw), delimiter=",", dtype=np.int64)
            count = 3823 if filename.endswith("tra") else 1797
            if array.shape != (count, 65):
                raise ValueError(f"Unexpected data shape for {filename}.")
            if not ((array[:, :64] >= 0) & (array[:, :64] <= 16)).all():
                raise ValueError("Digit pixel counts must be between zero and sixteen.")
            arrays[filename] = array
    return arrays


def _order(ids: list[int], namespace: str) -> list[int]:
    return sorted(ids, key=lambda i: sha256(f"arena16.9/v1/{namespace}/{i}".encode()))


def load_study_data(cache: Path = DEFAULT_CACHE) -> dict:
    arrays = source_arrays(cache)
    selected = {"train": [], "ranking": [], "final": []}
    for label in (1, 7):
        train_ids = _order(np.flatnonzero(arrays["optdigits.tra"][:, -1] == label).tolist(), "train")
        test_ids = _order(np.flatnonzero(arrays["optdigits.tes"][:, -1] == label).tolist(), "test")
        selected["train"].extend(train_ids[:64])
        selected["ranking"].extend(train_ids[64:88])
        selected["final"].extend(test_ids[:32])
    splits = {}
    for name, ids in selected.items():
        filename = "optdigits.tes" if name == "final" else "optdigits.tra"
        ids = sorted(ids)
        rows = arrays[filename][ids]
        images = torch.from_numpy(rows[:, :64].copy()).double().reshape(-1, 8, 8) / 16
        x = torch.cat([images.flatten(1), torch.ones(len(ids), 1, dtype=torch.float64)], dim=1)
        splits[name] = dict(
            x=x, images=images, y=torch.from_numpy((rows[:, -1] == 7).astype(np.int64)),
            digit=torch.from_numpy(rows[:, -1].copy()),
            ids=[f"{filename}:{i}" for i in ids], source_rows=ids, source_file=filename,
        )
    train = splits["train"]
    flipped = []
    for digit in (1, 7):
        candidates = [row for row, value in zip(train["source_rows"], train["digit"].tolist()) if value == digit]
        flipped.extend(_order(candidates, "flip")[:6])
    mask = torch.tensor([row in flipped for row in train["source_rows"]], dtype=torch.bool)
    train["corrupted_y"] = torch.where(mask, 1 - train["y"], train["y"])
    train["flipped"] = mask
    probes = []
    for digit in (1, 7):
        for is_flipped in (False, True):
            ids = torch.nonzero((train["digit"] == digit) & (mask == is_flipped)).flatten().tolist()
            probes.extend(ids[:3])
    splits["upweight_probe_indices"] = probes
    return splits


def data_manifest(data: dict) -> dict:
    splits = {}
    for name in ("train", "ranking", "final"):
        split = data[name]
        splits[name] = {
            "ids": split["ids"], "source_file": split["source_file"],
            "x_sha256": sha256(split["x"].numpy().astype("<f8").tobytes()),
            "y_sha256": sha256(split["y"].numpy().astype("<i8").tobytes()),
            "class_counts": torch.bincount(split["y"], minlength=2).tolist(),
        }
    return {
        "source_url": DATA_URL, "archive_sha256": ZIP_SHA256, "source_sha256": FILE_SHA256,
        "source_citation": "Alpaydin and Kaynak (1998), UCI Optical Recognition of Handwritten Digits, DOI 10.24432/C50P49",
        "license": "CC BY 4.0", "negative_digit": 1, "positive_digit": 7,
        "preprocessing": "64 integer pixel counts / 16, then a constant 1 bias coordinate; no fitted normalization",
        "splits": splits,
        "flipped_train_ids": [row for row, flag in zip(data["train"]["ids"], data["train"]["flipped"]) if flag],
        "corrupted_y_sha256": sha256(data["train"]["corrupted_y"].numpy().astype("<i8").tobytes()),
        "upweight_probe_indices": data["upweight_probe_indices"],
    }


if __name__ == "__main__":
    print(json.dumps(data_manifest(load_study_data()), indent=2))
