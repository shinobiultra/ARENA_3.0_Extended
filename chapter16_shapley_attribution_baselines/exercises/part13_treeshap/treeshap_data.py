"""Pinned public data and recipe-disjoint row selection; no model code."""
from pathlib import Path
import hashlib
import io
import json
import urllib.request
import numpy as np

HERE = Path(__file__).resolve().parent
URL = 'https://archive.ics.uci.edu/static/public/165/data.csv'
CACHE = Path.home() / '.cache/arena_extended/treeshap/concrete.csv'


def source_data():
    manifest = json.loads((HERE / 'data_manifest.json').read_text())
    if not CACHE.exists():
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(URL, timeout=60) as response:
            payload = response.read()
        if hashlib.sha256(payload).hexdigest() != manifest['source_sha256']:
            raise ValueError('UCI source changed; do not silently replace the frozen data.')
        CACHE.write_bytes(payload)
    payload = CACHE.read_bytes()
    if hashlib.sha256(payload).hexdigest() != manifest['source_sha256']:
        raise ValueError('Cached UCI source checksum mismatch.')
    data = np.loadtxt(io.BytesIO(payload), delimiter=',', skiprows=1)
    if data.shape != (1030, 9) or not np.isfinite(data).all():
        raise ValueError('Expected 1030 complete rows with eight inputs and one target.')
    return data


def load_split(name):
    manifest = json.loads((HERE / 'data_manifest.json').read_text())
    ids = np.array(manifest['splits'][name], dtype=np.int64)
    data = source_data()[ids]
    return {'x': data[:, :8].copy(), 'y': data[:, 8].copy(), 'ids': ids}


def background_order(train_ids, seed):
    """Stable source-row order, independent of inputs, labels and model results."""
    return np.array(sorted(range(len(train_ids)), key=lambda i: hashlib.sha256(
        f'arena16.13/background/{seed}/{int(train_ids[i])}'.encode()).digest()))
