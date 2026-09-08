"""Pinned Concrete data and identity plumbing for the interaction study.

This module contains only source loading, split identity, background ordering,
and portable tensor hashing. The interaction methods and study calculations do
not live here.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import urllib.request

import numpy as np


HERE = Path(__file__).resolve().parent
URL = "https://archive.ics.uci.edu/static/public/165/data.csv"
CACHE = Path.home() / ".cache/arena_extended/shapley_interactions/concrete.csv"
SPLIT_NAMESPACE = b"arena16.12/split/"


def load_manifest() -> dict:
    """Return the source, split, model, and donor declarations."""

    return json.loads((HERE / "data_manifest.json").read_text())


def source_data() -> np.ndarray:
    """Load the byte-pinned UCI table without filtering or deduplication."""

    manifest = load_manifest()
    if not CACHE.exists():
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(URL, timeout=60) as response:
            payload = response.read()
        if hashlib.sha256(payload).hexdigest() != manifest["source_sha256"]:
            raise ValueError("UCI source changed; do not replace the pinned data.")
        CACHE.write_bytes(payload)
    payload = CACHE.read_bytes()
    if len(payload) != manifest["source_bytes"]:
        raise ValueError("Cached UCI source byte count mismatch.")
    if hashlib.sha256(payload).hexdigest() != manifest["source_sha256"]:
        raise ValueError("Cached UCI source checksum mismatch.")
    data = np.loadtxt(io.BytesIO(payload), delimiter=",", skiprows=1)
    if data.shape != (1030, 9) or not np.isfinite(data).all():
        raise ValueError("Expected 1030 complete rows with eight inputs and one target.")
    return data


def load_split(name: str) -> dict[str, np.ndarray]:
    """Return one pinned recipe-disjoint split in source-ID order."""

    manifest = load_manifest()
    if name not in {"train", "explain", "audit"}:
        raise ValueError("name must be 'train', 'explain', or 'audit'.")
    ids = np.asarray(manifest["splits"][name], dtype=np.int64)
    data = source_data()[ids]
    return {
        "x": data[:, :8].copy(),
        "y": data[:, 8].copy(),
        "ids": ids.copy(),
        "recipe_group_ids": recipe_group_ids(ids),
    }


def _recipe_key(row: np.ndarray) -> bytes:
    values = np.asarray(row[:7], dtype="<f8")
    return values.tobytes(order="C")


def recipe_group_ids(source_ids: np.ndarray) -> np.ndarray:
    """Return stable integer IDs for the frozen first-seven-feature groups."""

    ids = np.asarray(source_ids, dtype=np.int64)
    if ids.ndim != 1 or len(np.unique(ids)) != len(ids):
        raise ValueError("source_ids must be a one-dimensional array of unique IDs.")
    data = source_data()
    if len(ids) and (ids.min() < 0 or ids.max() >= len(data)):
        raise ValueError("source_ids contains an out-of-range source row.")
    keys = [_recipe_key(row) for row in data]
    unique = sorted(
        set(keys),
        key=lambda key: hashlib.sha256(SPLIT_NAMESPACE + key).digest(),
    )
    index = {key: position for position, key in enumerate(unique)}
    return np.asarray([index[keys[int(row_id)]] for row_id in ids], dtype=np.int64)


def background_order(train_ids: np.ndarray, seed: int) -> np.ndarray:
    """Reproduce the exact known-model training-row donor order."""

    ids = np.asarray(train_ids, dtype=np.int64)
    if ids.ndim != 1 or len(np.unique(ids)) != len(ids):
        raise ValueError("train_ids must be a one-dimensional array of unique IDs.")
    return np.asarray(
        sorted(
            range(len(ids)),
            key=lambda index: hashlib.sha256(
                f"arena16.14/background/{seed}/{int(ids[index])}".encode()
            ).digest(),
        ),
        dtype=np.int64,
    )


def tensor_mapping_sha256(mapping: dict) -> str:
    """Hash named tensors with stable names, shapes, dtypes, and bytes."""

    digest = hashlib.sha256()
    for name in sorted(mapping):
        value = mapping[name]
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise TypeError("Object arrays do not have a portable tensor hash.")
        if array.dtype.byteorder == ">" or (
            array.dtype.byteorder == "=" and not np.little_endian
        ):
            array = array.astype(array.dtype.newbyteorder("<"))
        array = np.ascontiguousarray(array)
        dtype = array.dtype.str.replace("=", "<")
        metadata = json.dumps(
            {"dtype": dtype, "name": str(name), "shape": list(array.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        digest.update(len(metadata).to_bytes(8, "little"))
        digest.update(metadata)
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


__all__ = [
    "CACHE",
    "HERE",
    "URL",
    "background_order",
    "load_manifest",
    "load_split",
    "recipe_group_ids",
    "source_data",
    "tensor_mapping_sha256",
]
