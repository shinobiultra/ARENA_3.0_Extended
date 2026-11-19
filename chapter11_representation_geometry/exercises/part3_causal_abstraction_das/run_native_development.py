"""Weight digest used to show the pretrained model was left unchanged."""

from __future__ import annotations

import hashlib


def weight_digest(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode())
        digest.update(parameter.detach().contiguous().numpy().tobytes())
    return digest.hexdigest()
