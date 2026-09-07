"""Compare only inspected primitives from the pinned MIT-licensed author source.

The upstream constructor is not executed. Its script version overwrites a probe
with None, while its demonstration notebook does not. Training, test centering
and orientation in this learner deliberately have their own explicit protocols.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from urllib.request import urlopen

import numpy as np
import torch

from chapter11_representation_geometry.exercises.part5_contrast_consistent_search import ccs_methods

REVISION = "be10d7b912b17e8eacb45de794e9dca044e4ed37"
SOURCE_SHA256 = "5a65f7d130aa5531ca37979d487e63057ef2d9aa4d925176f892a6ebc7955c13"
LICENSE_SHA256 = "99a7c1ab3190e7da4978946ca7152d1f939ae335cc1d1917ee5a70ec5989c274"
DEFAULT_DIRECTORY = Path.home() / ".cache/arena_extended/ccs/author-source-v1"


def load_primitives(directory: Path = DEFAULT_DIRECTORY):
    directory.mkdir(parents=True, exist_ok=True)
    for filename, expected in (("utils.py", SOURCE_SHA256), ("LICENSE.md", LICENSE_SHA256)):
        path = directory / filename
        if not path.exists():
            url = f"https://raw.githubusercontent.com/collin-burns/discovering_latent_knowledge/{REVISION}/{filename}"
            with urlopen(url, timeout=30) as response:
                data = response.read()
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError(f"Unexpected upstream content: {filename}")
            path.write_bytes(data)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Pinned author file has changed: {path}")
    tree = ast.parse((directory / "utils.py").read_text())
    klass = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "CCS")
    selected = [node for node in klass.body if isinstance(node, ast.FunctionDef) and node.name in {"normalize", "get_loss"}]
    if len(selected) != 2:
        raise ValueError("Expected exactly the two inspected author primitives.")
    # The verified definitions only use np/torch and the false var_normalize flag.
    namespace = {"torch": torch, "np": np}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(directory / "utils.py"), "exec"), namespace)
    return namespace["normalize"], namespace["get_loss"]


def compare_author_primitives(directory: Path = DEFAULT_DIRECTORY) -> dict:
    normalize, original_loss = load_primitives(directory)
    author_self = SimpleNamespace(var_normalize=False)
    generator = torch.Generator().manual_seed(739)
    records = []
    for case in ("full_rank", "rank_deficient", "translated"):
        plus = torch.randn(24, 5, generator=generator, dtype=torch.float64)
        minus = torch.randn(24, 5, generator=generator, dtype=torch.float64)
        if case == "rank_deficient":
            plus[:, -1], minus[:, -1] = plus[:, 0] * 2, minus[:, 0] * 2
        elif case == "translated":
            plus, minus = plus + 13, minus - 9
        means = ccs_methods.fit_pair_centering(plus, minus)
        p, m = ccs_methods.apply_pair_centering(plus, minus, means)
        expected_p = torch.from_numpy(normalize(author_self, plus.numpy()))
        expected_m = torch.from_numpy(normalize(author_self, minus.numpy()))
        centering_error = max(float((p - expected_p).abs().max()), float((m - expected_m).abs().max()))
        weight = torch.randn(5, generator=generator, dtype=torch.float64, requires_grad=True)
        bias = torch.tensor(0.17, dtype=torch.float64, requires_grad=True)
        p_plus, p_minus = (p @ weight + bias).sigmoid(), (m @ weight + bias).sigmoid()
        actual = ccs_methods.ccs_loss(p_plus, p_minus)[0]
        expected = original_loss(author_self, p_plus, p_minus)
        actual_gradient = torch.autograd.grad(actual, (weight, bias), retain_graph=True)
        expected_gradient = torch.autograd.grad(expected, (weight, bias))
        gradient_error = max(float((a - b).abs().max()) for a, b in zip(actual_gradient, expected_gradient))
        loss_error = abs(float(actual.detach()) - float(expected.detach()))
        assert max(centering_error, loss_error, gradient_error) <= 1e-12
        records.append({"case": case, "centering_error": centering_error, "loss_error": loss_error, "gradient_error": gradient_error})
    return {
        "passed": True,
        "revision": REVISION,
        "source_sha256": SOURCE_SHA256,
        "license_sha256": LICENSE_SHA256,
        "primitives": ["mean-only training centering", "original asymmetric loss", "loss gradients"],
        "whole_author_class_executed": False,
        "cases": records,
    }


if __name__ == "__main__":
    print(json.dumps(compare_author_primitives(), indent=2))
