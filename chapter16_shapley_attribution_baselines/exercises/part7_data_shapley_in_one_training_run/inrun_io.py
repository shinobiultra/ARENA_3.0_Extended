"""Bounded numeric evidence serialization; contains no attribution or training code."""
import gzip
import json
import os
from pathlib import Path
import tempfile

import torch


def save_numeric_evidence(path, value):
    def plain(item):
        if isinstance(item, torch.Tensor):
            return {'tensor': True, 'dtype': str(item.dtype), 'shape': list(item.shape), 'values': item.tolist()}
        if isinstance(item, dict):
            return {str(k): plain(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [plain(v) for v in item]
        return item
    content = json.dumps(plain(value), allow_nan=False, separators=(',', ':')).encode()
    if len(content) > 64 * 1024**2:
        raise ValueError('Numeric evidence exceeds the portable bound.')
    # Interactive learners may rerun a play cell or the complete notebook.
    # The publication runner itself always allocates a new evidence directory.
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(gzip.compress(content, mtime=0))
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
