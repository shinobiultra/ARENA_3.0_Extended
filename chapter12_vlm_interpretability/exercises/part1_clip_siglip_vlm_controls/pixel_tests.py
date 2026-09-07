"""Numerical exercise tests. Notebook calls always pass its local implementation.

Run reference checks with ``uv run pytest -q <this file>``. These fixtures use
independent formulas rather than stored training results or teacher outputs.
"""

from __future__ import annotations

import copy
import math

import pytest
import torch as t
from torch import nn
from torch.nn import functional as F

from . import pixel_clip as reference


def test_tokenizer(fn=None):
    fn = fn or reference.tokenize_pixel_captions
    vocab = dict(reference.PIXEL_VOCAB)
    original = dict(vocab)
    ids, mask = fn(["A red square", "blue unfamiliar"], vocab, context_length=7)
    assert ids.dtype == t.long and mask.dtype == t.bool
    t.testing.assert_close(ids[0], t.tensor([1, 4, vocab["red"], vocab["square"], 2, 0, 0]))
    t.testing.assert_close(ids[1], t.tensor([1, vocab["blue"], 3, 2, 0, 0, 0]))
    t.testing.assert_close(mask.sum(1), t.tensor([5, 4]))
    assert vocab == original
    with pytest.raises(ValueError):
        fn(["a red square"], vocab, context_length=4)


def test_pixel_cnn(cls=None):
    cls = cls or reference.PixelImageEncoder
    t.manual_seed(17)
    model = cls(width=8).double()
    pixels = t.rand(2, 3, 32, 32, dtype=t.float64, requires_grad=True)
    actual = model(pixels)
    hidden = 2 * pixels - 1
    for convolution in (model.conv1, model.conv2, model.conv3):
        hidden = F.gelu(
            F.conv2d(
                hidden,
                convolution.weight,
                convolution.bias,
                stride=2,
                padding=convolution.kernel_size[0] // 2,
            )
        )
    expected = F.linear(hidden.flatten(1), model.readout.weight, model.readout.bias)
    t.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    weights = t.linspace(-1, 1, actual.numel(), dtype=t.float64).reshape_as(actual)
    parameters = (pixels, model.conv1.weight, model.conv3.weight, model.readout.weight)
    actual_grad = t.autograd.grad((actual * weights).sum(), parameters, retain_graph=True)
    expected_grad = t.autograd.grad((expected * weights).sum(), parameters)
    for got, wanted in zip(actual_grad, expected_grad, strict=True):
        t.testing.assert_close(got, wanted, atol=1e-12, rtol=1e-12)
        assert float(got.norm()) > 0
    assert not t.allclose(model(pixels.detach()), model(pixels.detach().flip(-1)))
    with pytest.raises(ValueError):
        model(t.ones(2, 8, dtype=t.float64))


def test_causal_text_block(cls=None):
    cls = cls or reference.PixelTextBlock
    t.manual_seed(18)
    model = cls(width=8, heads=2).double()
    hidden = t.randn(2, 5, 8, dtype=t.float64, requires_grad=True)
    valid = t.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]], dtype=t.bool)
    actual = model(hidden, valid)
    qkv = model.qkv(model.norm1(hidden)).reshape(2, 5, 3, 2, 4)
    q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
    allowed = t.ones(5, 5, dtype=t.bool).tril()[None, None] & valid[:, None, None]
    # Independent PyTorch attention primitive, not the reference's manual scores.
    attention = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
    residual = hidden + model.attention_out(attention.transpose(1, 2).reshape(2, 5, 8))
    expected = residual + model.fc2(F.gelu(model.fc1(model.norm2(residual))))
    t.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    actual_grad = t.autograd.grad(actual.square().sum(), hidden, retain_graph=True)[0]
    expected_grad = t.autograd.grad(expected.square().sum(), hidden)[0]
    t.testing.assert_close(actual_grad, expected_grad, atol=1e-10, rtol=1e-10)
    changed_future = hidden.detach().clone()
    changed_future[:, 3:] += 100 * t.randn_like(changed_future[:, 3:])
    t.testing.assert_close(model(changed_future, valid)[:, :3], actual[:, :3])
    changed_padding = hidden.detach().clone()
    changed_padding[~valid] += 100 * t.randn_like(changed_padding[~valid])
    t.testing.assert_close(model(changed_padding, valid)[valid], actual[valid])
    with pytest.raises(ValueError):
        model(hidden, t.zeros_like(valid))


def test_eos_encoder(cls=None):
    cls = cls or reference.PixelTextEncoder
    t.manual_seed(19)
    model = cls(vocab_size=len(reference.PIXEL_VOCAB), width=8).double()
    ids, valid = reference.tokenize_pixel_captions(
        ["a red square", "a blue circle"], reference.PIXEL_VOCAB
    )
    expected_hidden = model.final_norm(model.block(model.embedding(ids) + model.position, valid))
    # EOS id 2 is deliberately smaller than the word IDs: argmax(token_ids) is wrong.
    t.testing.assert_close(model(ids, valid), expected_hidden[:, 4])
    changed_padding = ids.clone()
    changed_padding[~valid] = reference.PIXEL_VOCAB["triangle"]
    t.testing.assert_close(model(ids, valid), model(changed_padding, valid))
    # Gradients must reach content embeddings, not just the constant EOS vector.
    model(ids, valid).square().sum().backward()
    assert model.embedding.weight.grad[reference.PIXEL_VOCAB["red"]].norm() > 0
    no_eos = ids.clone()
    no_eos[no_eos == 2] = 3
    with pytest.raises(ValueError):
        model(no_eos, valid)
    two_eos = ids.clone()
    two_eos[:, 2] = 2
    with pytest.raises(ValueError):
        model(two_eos, valid)


def test_dual_encoder(cls=None):
    cls = cls or reference.PixelDualEncoder
    t.manual_seed(20)
    model = cls(vocab_size=len(reference.PIXEL_VOCAB), width=8, embedding_dim=5).double()
    pixels = t.rand(3, 3, 32, 32, dtype=t.float64)
    ids, mask = reference.tokenize_pixel_captions(
        reference.PIXEL_CAPTIONS[:3], reference.PIXEL_VOCAB
    )
    image = F.normalize(model.image_projection(model.image_encoder(pixels)), dim=-1)
    text = F.normalize(model.text_projection(model.text_encoder(ids, mask)), dim=-1)
    t.testing.assert_close(model.encode_images(pixels), image)
    t.testing.assert_close(model.encode_texts(ids, mask), text)
    t.testing.assert_close(image.norm(dim=-1), t.ones(3, dtype=t.float64))
    t.testing.assert_close(text.norm(dim=-1), t.ones(3, dtype=t.float64))
    with t.no_grad():
        model.logit_scale.fill_(math.log(7))
        model.logit_bias.fill_(-1.5)
    logits = model(pixels, ids, mask)
    t.testing.assert_close(logits, 7 * image @ text.T - 1.5)
    logits.square().sum().backward()
    for module in (
        model.image_encoder,
        model.text_encoder,
        model.image_projection,
        model.text_projection,
    ):
        assert sum(float(parameter.grad.square().sum()) for parameter in module.parameters()) > 0
    assert model.logit_scale.grad.abs() > 0 and model.logit_bias.grad.abs() > 0


def test_alignment_losses(fn=None):
    fn = fn or reference.pixel_alignment_loss
    for objective, scalar in (("clip", math.log(3)), ("siglip", 3 * math.log(2))):
        logits = t.zeros(3, 3, dtype=t.float64, requires_grad=True)
        loss = fn(logits, objective=objective)
        t.testing.assert_close(loss, t.tensor(scalar, dtype=t.float64))
        loss.backward()
        expected = (
            (t.ones(3, 3) / 9 - t.eye(3) / 3)
            if objective == "clip"
            else (t.ones(3, 3) / 6 - t.eye(3) / 3)
        )
        t.testing.assert_close(logits.grad, expected.double(), atol=1e-8, rtol=1e-7)
    logits = t.tensor([[2.0, -1.0], [0.5, 3.0]], dtype=t.float64, requires_grad=True)
    # Check asymmetric nonzero logits too; a hard-coded zero fixture must fail.
    expected_clip = (
        (logits.logsumexp(0) + logits.logsumexp(1)).sum() / 2 - logits.diag().sum()
    ) / 2
    t.testing.assert_close(fn(logits, objective="clip"), expected_clip)
    signed = t.tensor([[1.0, -1.0], [-1.0, 1.0]], dtype=t.float64)
    expected_siglip = t.log1p(t.exp(-signed * logits)).sum() / 2
    t.testing.assert_close(fn(logits, objective="siglip"), expected_siglip)
    # SigLIP pair gradients are local; CLIP competitors share a denominator.
    sig_grad = t.autograd.grad(fn(logits, objective="siglip"), logits)[0]
    changed = logits.detach().clone().requires_grad_(True)
    with t.no_grad():
        changed[0, 1] += 3
    changed_grad = t.autograd.grad(fn(changed, objective="siglip"), changed)[0]
    t.testing.assert_close(sig_grad[0, 0], changed_grad[0, 0])
    with pytest.raises(ValueError):
        fn(t.ones(2, 3), objective="clip")


def test_semantic_retrieval(fn=None):
    fn = fn or reference.semantic_retrieval_metrics
    image_labels, text_labels = t.tensor([0, 0, 1, 1]), t.tensor([0, 1])
    logits = t.tensor([[3.0, 1.0], [2.0, 0.0], [-1.0, 1.0], [0.0, 4.0]])
    assert fn(logits, image_labels, text_labels) == {"image_to_text": 1.0, "text_to_image": 1.0}
    row_order, col_order = t.tensor([3, 1, 0, 2]), t.tensor([1, 0])
    assert fn(
        logits[row_order][:, col_order], image_labels[row_order], text_labels[col_order]
    ) == fn(logits, image_labels, text_labels)
    ties = fn(t.zeros(24, 12), t.arange(12).repeat_interleave(2), t.arange(12))
    assert ties["image_to_text"] == pytest.approx(1 / 12)
    assert ties["text_to_image"] == pytest.approx(1 / 12)
    assert fn(logits.flip(1), image_labels, text_labels) == {
        "image_to_text": 0.0,
        "text_to_image": 0.0,
    }
    with pytest.raises(ValueError):
        fn(logits, image_labels, t.tensor([0, 2]))


def test_training_loop(fn=None):
    fn = fn or reference.train_pixel_dual_encoder
    t.manual_seed(21)
    dataset = reference.make_pixel_dataset(examples_per_class=2, seed=901)
    dataset = reference.PixelDataset(
        dataset.images.double(), dataset.labels, dataset.captions, dataset.seed
    )
    model = reference.PixelDualEncoder(
        len(reference.PIXEL_VOCAB), width=8, embedding_dim=5
    ).double()
    wanted = copy.deepcopy(model)
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    # Reproduce two exact AdamW updates from independent batch-index and loss formulas.
    optimizer = t.optim.AdamW(wanted.parameters(), lr=0.002, weight_decay=0.01)
    ids, mask = reference.tokenize_pixel_captions(dataset.captions, reference.PIXEL_VOCAB)
    generator = t.Generator().manual_seed(23)
    losses = []
    for _ in range(2):
        indices = 2 * t.arange(12) + t.stack(
            [t.randint(2, (), generator=generator) for _ in range(12)]
        )
        logits = wanted(dataset.images[indices], ids, mask)
        loss = ((logits.logsumexp(0) + logits.logsumexp(1)).sum() / 2 - logits.diag().sum()) / 12
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        t.nn.utils.clip_grad_norm_(wanted.parameters(), 1.0)
        optimizer.step()
        with t.no_grad():
            wanted.logit_scale.clamp_(0, math.log(100))
        losses.append(float(loss.detach()))
    history = fn(model, dataset, objective="clip", steps=2, seed=23)
    t.testing.assert_close(t.tensor(history["losses"]), t.tensor(losses))
    for name, actual in model.named_parameters():
        t.testing.assert_close(actual, dict(wanted.named_parameters())[name], atol=2e-5, rtol=2e-5)
    for prefix, actual in history["parameter_change_norms"].items():
        exact = math.sqrt(
            sum(
                float((p.detach() - before[n]).square().sum())
                for n, p in model.named_parameters()
                if n.startswith(prefix)
            )
        )
        assert actual == pytest.approx(exact) and actual > 0


def test_evaluation(fn=None):
    fn = fn or reference.evaluate_pixel_model
    t.manual_seed(22)
    model = reference.PixelDualEncoder(len(reference.PIXEL_VOCAB), width=8, embedding_dim=5)
    dataset = reference.make_pixel_dataset(examples_per_class=2, seed=902)
    ids, mask = reference.tokenize_pixel_captions(dataset.captions, reference.PIXEL_VOCAB)
    expected = model(dataset.images, ids, mask).detach()
    actual = fn(model, dataset)
    t.testing.assert_close(actual["logits"], expected)
    assert actual["image_norm_max_error"] < 1e-6
    assert actual["text_norm_max_error"] < 1e-6
    blank = fn(model, dataset, image_override=t.ones_like(dataset.images))
    assert blank["image_to_text"] == pytest.approx(1 / 12)
    assert blank["text_to_image"] == pytest.approx(1 / 12)


def test_broken_implementations_are_rejected():
    class CollapsedCNN(reference.PixelImageEncoder):
        def forward(self, pixels):
            return super().forward(pixels * 0)

    class ArgmaxEOS(reference.PixelTextEncoder):
        def forward(self, token_ids, valid_mask):
            hidden = self.final_norm(
                self.block(self.embedding(token_ids) + self.position, valid_mask)
            )
            return hidden[t.arange(len(hidden)), token_ids.argmax(-1)]

    for test, broken in ((test_pixel_cnn, CollapsedCNN), (test_eos_encoder, ArgmaxEOS)):
        with pytest.raises(AssertionError):
            test(broken)
    with pytest.raises(AssertionError):
        test_alignment_losses(
            lambda logits, objective: reference.pixel_alignment_loss(logits, objective=objective)
            / len(logits)
        )


def test_dataset_reproducibility_and_separation():
    train = reference.make_pixel_dataset(examples_per_class=2, seed=100)
    same = reference.make_pixel_dataset(examples_per_class=2, seed=100)
    different = reference.make_pixel_dataset(examples_per_class=2, seed=300)
    t.testing.assert_close(train.images, same.images)
    assert reference.dataset_overlap(train, same) == 24
    assert reference.dataset_overlap(train, different) == 0


def test_activation_patch(fn=None):
    from types import SimpleNamespace
    from . import solutions

    fn = fn or solutions.run_with_activation_patch

    class SmallModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Linear(3, 4)
            self.fail = False

        def forward(self, pixel_values):
            hidden = self.embedding(pixel_values)
            if self.fail:
                raise RuntimeError("deliberate forward failure")
            return SimpleNamespace(logits_per_image=hidden.sum(1)[:, :2])

    t.manual_seed(26)
    model = SmallModel()
    clean, corrupt = t.randn(2, 3, 3), t.randn(2, 3, 3)
    with t.no_grad():
        expected = model.embedding(clean)
        expected[:, 1] = model.embedding(corrupt)[:, 1]
    actual = fn(model, model.embedding, {"pixel_values": clean}, corrupt, [1])
    t.testing.assert_close(actual, expected.sum(1)[:, :2])
    assert not model.embedding._forward_hooks
    model.fail = True
    with pytest.raises(RuntimeError, match="deliberate forward failure"):
        fn(model, model.embedding, {"pixel_values": clean}, corrupt, [1])
    assert not model.embedding._forward_hooks


def test_typographic_metrics(fn=None):
    from . import typography

    fn = fn or typography.typographic_attack_metrics
    baseline = t.tensor([[3.0, 1.0, 0.0], [0.0, 4.0, 1.0], [0.0, 1.0, 4.0], [0.0, 2.0, 1.0]])
    changed = t.tensor([[0.0, 3.0, 0.0], [0.0, 0.0, 3.0], [0.0, 1.0, 4.0], [0.0, 3.0, 1.0]])
    labels, attack = t.tensor([0, 1, 2, 0]), t.tensor([1, 2, 0, 1])
    result = fn(baseline, changed, labels, attack)
    assert result["baseline_correct_count"] == 3
    assert result["accuracy"] == 0.25
    assert result["attack_choice_rate"] == 0.75
    assert result["targeted_flip_rate"] == pytest.approx(2 / 3)
    assert result["mean_margin_change"] == pytest.approx(-3)
    order = t.tensor([3, 1, 0, 2])
    assert fn(baseline[order], changed[order], labels[order], attack[order]) == result
    no_correct = fn(baseline[:1].flip(1), changed[:1], labels[:1], attack[:1])
    assert no_correct["targeted_flip_rate"] is None
    with pytest.raises(ValueError):
        fn(baseline, changed, labels, labels)


def test_typographic_controls():
    from . import typography
    import numpy as np

    dataset = typography.make_typographic_dataset(examples_per_class=2)
    assert len(dataset["labels"]) == 24
    assert bool((dataset["labels"] != dataset["attack_labels"]).all())
    assert t.equal(dataset["labels"] // 3, dataset["attack_labels"] // 3)
    for index, box in enumerate(dataset["object_boxes"]):
        assert box[1] > 38, "background word card must not occlude the object"
        clean = np.asarray(dataset["images"]["clean"][index])
        blank = np.asarray(dataset["images"]["background/blank_card"][index])
        np.testing.assert_array_equal(clean, blank)
        word = np.asarray(dataset["images"]["background/conflicting_word"][index])
        assert not np.array_equal(clean, word)
        np.testing.assert_array_equal(clean[39:], word[39:])


def test_learner_answers_complete_and_match():
    import ast
    import json
    from pathlib import Path
    from .build_notebooks import complete_learner, STEM

    here = Path(__file__).resolve().parent
    learner = json.loads((here / f"{STEM}_exercises.ipynb").read_text())
    expected = json.loads((here / f"{STEM}_solutions.ipynb").read_text())
    completed = complete_learner(learner)
    for actual, wanted in zip(completed["cells"], expected["cells"], strict=True):
        if actual["cell_type"] == "code":
            assert ast.dump(ast.parse("".join(actual["source"]))) == ast.dump(
                ast.parse("".join(wanted["source"]))
            )
