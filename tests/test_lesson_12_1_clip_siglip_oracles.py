"""12.1 CLIP and SigLIP from scratch: CPU oracles for `real_retrieval_table`, no model loads.

The test is run against the reference solution (it must pass) and against deliberately wrong variants (each must fail).
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pytest
import torch as t
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SECTION = ROOT / "chapter12_vlm_interpretability/exercises/part1_clip_siglip"
MASTER = ROOT / "infrastructure/chapters/chapter12_vlm_interpretability/master_12_1.py"
NAMES = {"similarity_logits", "retrieval_metrics", "real_retrieval_table"}


def visible_definitions() -> dict:
    """Execute only the named solution functions of the generated solutions.py (importing it would pick a device)."""
    sys.path.insert(0, str(SECTION.parent))
    from part1_clip_siglip import utils

    tree = ast.parse((SECTION / "solutions.py").read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in NAMES]
    source = "from __future__ import annotations\n" + ast.unparse(ast.Module(body=nodes, type_ignores=[]))
    ns = dict(t=t, np=np, F=F, utils=utils, Float=object, Int=object, Tensor=t.Tensor)
    exec(compile(source, "<12.1 solutions: named functions only>", "exec"), ns)
    return ns


@pytest.fixture(scope="module")
def ns():
    return visible_definitions()


@pytest.fixture(scope="module")
def tests():
    sys.path.insert(0, str(SECTION.parent))
    from part1_clip_siglip import tests as module

    return module


def noise_like(images, seed):
    from PIL import Image

    w, h = images[0].size
    pixels = np.random.default_rng(seed).integers(0, 256, size=(len(images), h, w, 3), dtype=np.uint8)
    return [Image.fromarray(p) for p in pixels]


def make_variant(ns, kind: str):
    """A `real_retrieval_table` with one mistake, built from the same parts as the solution."""
    similarity_logits, retrieval_metrics = ns["similarity_logits"], ns["retrieval_metrics"]

    def table(model, images, labels, class_names=None, seed=0):
        class_names = class_names or ns["utils"].CIFAR10_CLASSES
        texts = model.embed_texts([f"a photo of a {c}" for c in class_names])
        ids = t.arange(len(class_names))
        generator = t.Generator().manual_seed(seed)
        perm = t.randperm(len(labels), generator=generator)

        def i2t(imgs, labs, order=None):
            emb = model.embed_images(imgs if order is None else [imgs[i] for i in order.tolist()])
            return float(retrieval_metrics(similarity_logits(emb, texts, 1.0), labs, ids)["image_to_text"])

        real = i2t(images, labels)
        if kind == "pairing_kept":  # shuffles images and labels together: the pairing survives
            shuffled = i2t(images, labels[perm], order=perm)
        elif kind == "no_shuffle":
            shuffled = i2t(images, labels)
        elif kind == "in_place":
            shuffled_labels = labels
            shuffled_labels.copy_(labels[perm])
            shuffled = i2t(images, shuffled_labels)
        else:
            shuffled = i2t(images, labels[perm])
        noise = noise_like(images, seed)
        if kind == "noise_is_real":
            noise_score = i2t(images, labels)
        elif kind == "noise_not_enough":
            noise_score = i2t(noise_like(images[: len(images) // 2], seed), labels[: len(images) // 2])
        else:
            noise_score = i2t(noise, labels)
        if kind == "unseeded":
            shuffled = float(np.random.rand() * 0.1)
        return {"real images": real, "shuffled labels (control)": shuffled, "noise images (control)": noise_score}

    return table


def test_stub_fails(tests):
    def stub(model, images, labels, class_names=None, seed=0):
        raise NotImplementedError

    with pytest.raises(NotImplementedError):
        tests.test_real_retrieval_table(stub)


def test_solution_passes(ns, tests):
    tests.test_real_retrieval_table(ns["real_retrieval_table"])


def test_reference_shaped_variant_passes(ns, tests):
    tests.test_real_retrieval_table(make_variant(ns, "correct"))


@pytest.mark.parametrize("kind", ["pairing_kept", "no_shuffle", "in_place", "noise_is_real", "noise_not_enough", "unseeded"])
def test_wrong_variants_fail(ns, tests, kind):
    with pytest.raises(AssertionError):
        tests.test_real_retrieval_table(make_variant(ns, kind))
