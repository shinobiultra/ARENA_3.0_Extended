import contextlib
import math
from typing import Callable

import numpy as np
import torch as t
import torch.nn.functional as F


# ---------------------------------------------------------------------------------------------------------------------
# Part 1: the losses
# ---------------------------------------------------------------------------------------------------------------------


def test_similarity_logits(similarity_logits: Callable):
    img = t.tensor([[3.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    txt = t.tensor([[1.0, 0.0], [0.0, 5.0]])
    out = similarity_logits(img, txt, 10.0, -1.0)
    assert out.shape == (3, 2), f"Expected shape (n_images, n_texts) = (3, 2), got {tuple(out.shape)}"
    s = 1 / math.sqrt(2)
    expected = t.tensor([[9.0, -1.0], [-1.0, 9.0], [10 * s - 1, 10 * s - 1]])
    t.testing.assert_close(out, expected, msg="Wrong values. Normalise both inputs (norms must not matter), then scale * cos + bias")
    t.testing.assert_close(
        similarity_logits(7 * img, 0.1 * txt, 10.0, -1.0), expected, msg="Rescaling the inputs must not change the logits"
    )
    # defaults: no bias; scale and bias may be tensors (learnable parameters)
    t.testing.assert_close(similarity_logits(img, txt, 2.0), 2.0 * expected_cos(img, txt))
    scale, bias = t.tensor(4.0, requires_grad=True), t.tensor(0.5, requires_grad=True)
    out = similarity_logits(img, txt, scale, bias)
    out.sum().backward()
    assert scale.grad is not None and bias.grad is not None, "Gradients should flow into a tensor scale and bias"
    print("All tests in `test_similarity_logits` passed!")


def expected_cos(a: t.Tensor, b: t.Tensor) -> t.Tensor:
    return t.stack([t.stack([F.cosine_similarity(x, y, dim=0) for y in b]) for x in a])


def _brute_force_clip(logits: t.Tensor) -> float:
    n = logits.shape[0]
    z = logits.double().tolist()
    i2t = -sum(math.log(math.exp(z[i][i]) / sum(math.exp(z[i][j]) for j in range(n))) for i in range(n)) / n
    t2i = -sum(math.log(math.exp(z[j][j]) / sum(math.exp(z[i][j]) for i in range(n))) for j in range(n)) / n
    return 0.5 * (i2t + t2i)


def test_clip_loss(clip_loss: Callable):
    n = 4
    loss = clip_loss(t.zeros(n, n))
    assert loss.ndim == 0, "The loss should be a scalar"
    t.testing.assert_close(loss, t.tensor(math.log(n)), msg="With all logits equal the loss should be log(N)")
    # A matrix whose two directions give different losses: image-to-text and text-to-image must both be included
    logits = t.tensor([[2.0, 0.0, 1.0], [3.0, 0.5, 0.0], [0.0, 1.0, 0.2]])
    expected = _brute_force_clip(logits)
    t.testing.assert_close(clip_loss(logits).item(), expected, msg="Wrong value: compare with the symmetric InfoNCE formula")
    row_only = -sum(math.log(math.exp(logits[i, i]) / sum(math.exp(logits[i, j]) for j in range(3))) for i in range(3)) / 3
    assert abs(expected - row_only) > 1e-2, "Test setup error"
    t.manual_seed(0)
    big = 5 * t.randn(8, 8)
    t.testing.assert_close(clip_loss(big).item(), _brute_force_clip(big), atol=1e-5, rtol=1e-5)
    perfect = 50 * t.eye(5)
    assert clip_loss(perfect) < 1e-6, "Confident correct logits should give (almost) zero loss"
    x = t.randn(4, 4, requires_grad=True)
    clip_loss(x).backward()
    assert x.grad is not None, "The loss should be differentiable"
    print("All tests in `test_clip_loss` passed!")


def test_siglip_loss_terms(siglip_loss_terms: Callable):
    t.manual_seed(0)
    n = 5
    logits = 3 * t.randn(n, n)
    pos, neg = siglip_loss_terms(logits)
    assert pos.ndim == 0 and neg.ndim == 0, "Both terms should be scalars"
    z = logits.double().tolist()
    expected_pos = sum(math.log1p(math.exp(-z[i][i])) for i in range(n)) / n
    expected_neg = sum(math.log1p(math.exp(z[i][j])) for i in range(n) for j in range(n) if i != j) / n
    t.testing.assert_close(pos.item(), expected_pos, atol=1e-5, rtol=1e-5, msg="Wrong positive term: it's softplus(-diagonal), summed, divided by N")
    t.testing.assert_close(neg.item(), expected_neg, atol=1e-5, rtol=1e-5, msg="Wrong negative term: it's softplus(off-diagonal), summed, divided by N")
    # Hand-worked: all logits zero -> every pair contributes log 2; N positives, N^2 - N negatives, all divided by N
    pos, neg = siglip_loss_terms(t.zeros(4, 4))
    t.testing.assert_close(pos.item(), math.log(2))
    t.testing.assert_close(neg.item(), 3 * math.log(2))
    # Numerical stability: huge logits must not overflow
    extreme = t.full((3, 3), 100.0) - 200 * t.eye(3)  # diagonal -100 (wrong), off-diagonal +100 (wrong)
    pos, neg = siglip_loss_terms(extreme)
    assert t.isfinite(pos) and t.isfinite(neg), "Use F.softplus / F.logsigmoid: log(1 + exp(x)) overflows for large x"
    t.testing.assert_close(pos.item(), 100.0, atol=1e-3, rtol=1e-3)
    t.testing.assert_close(neg.item(), 200.0, atol=1e-3, rtol=1e-3)
    print("All tests in `test_siglip_loss_terms` passed!")


def test_loss_gradients(clip_loss_grad: Callable, siglip_loss_grad: Callable, clip_loss: Callable, siglip_loss: Callable):
    t.manual_seed(0)
    for n in [3, 6]:
        logits = 2 * t.randn(n, n)
        for name, grad_fn, loss_fn in [("clip", clip_loss_grad, clip_loss), ("siglip", siglip_loss_grad, siglip_loss)]:
            x = logits.clone().requires_grad_(True)
            loss_fn(x).backward()
            mine = grad_fn(logits)
            assert mine.shape == (n, n), f"{name}: expected an (N, N) gradient, got {tuple(mine.shape)}"
            t.testing.assert_close(mine, x.grad, atol=1e-6, rtol=1e-5, msg=f"{name}_loss_grad disagrees with autograd on your loss")
    print("All tests in `test_loss_gradients` passed!")


def test_false_negative_fraction(false_negative_fraction: Callable):
    assert abs(false_negative_fraction(t.tensor([0, 1, 2, 3])) - 0.0) < 1e-6, "All captions distinct: no false negatives"
    assert abs(false_negative_fraction(t.tensor([0, 0, 1, 2])) - 0.5) < 1e-6, "Two of four items share a caption"
    assert abs(false_negative_fraction(t.tensor([5, 5, 5, 5])) - 1.0) < 1e-6, "All items share a caption"
    assert abs(false_negative_fraction(t.tensor([7, 3, 7, 3, 9])) - 0.8) < 1e-6, "Four of five items have a twin"
    assert isinstance(false_negative_fraction(t.tensor([1, 2])), float), "Return a Python float"
    print("All tests in `test_false_negative_fraction` passed!")


# ---------------------------------------------------------------------------------------------------------------------
# Part 2: the tiny CLIP
# ---------------------------------------------------------------------------------------------------------------------


def test_tokenize_captions(tokenize_captions: Callable, vocab: list[str], max_len: int):
    ids = tokenize_captions(["a red circle in the top left", "a blue star"], vocab, max_len)
    assert ids.shape == (2, max_len), f"Expected shape (2, {max_len}), got {tuple(ids.shape)}"
    assert ids.dtype == t.long, "ids should be a LongTensor"
    word = vocab.index
    expected0 = [1, word("a"), word("red"), word("circle"), word("in"), word("the"), word("top"), word("left"), 2, 0]
    expected1 = [1, word("a"), word("blue"), word("star"), 2] + [0] * (max_len - 5)
    assert ids[0].tolist() == expected0, f"Row 0: expected {expected0}, got {ids[0].tolist()}"
    assert ids[1].tolist() == expected1, f"Row 1: expected {expected1}, got {ids[1].tolist()}"
    try:
        tokenize_captions(["a red banana"], vocab, max_len)
    except KeyError:
        pass
    else:
        raise AssertionError("An unknown word should raise a KeyError")
    print("All tests in `test_tokenize_captions` passed!")


def test_sample_batch(sample_batch: Callable):
    n_variants, n_classes = 3, 10
    # images[v, c] is a constant image whose value encodes (variant, class)
    values = (100 * t.arange(n_variants)[:, None] + t.arange(n_classes)[None, :]).float()
    images = values[:, :, None, None, None].expand(n_variants, n_classes, 3, 4, 4).contiguous()
    allowed = t.tensor([1, 3, 5, 7, 9])
    generator = t.Generator().manual_seed(0)
    seen_variants = set()
    for _ in range(20):
        batch, class_ids = sample_batch(images, allowed, 4, generator)
        assert batch.shape == (4, 3, 4, 4), f"Expected batch shape (4, 3, 4, 4), got {tuple(batch.shape)}"
        assert class_ids.shape == (4,), f"Expected class_ids of shape (4,), got {tuple(class_ids.shape)}"
        assert len(set(class_ids.tolist())) == 4, f"Classes in a batch must be distinct, got {class_ids.tolist()}"
        assert set(class_ids.tolist()) <= set(allowed.tolist()), "Only allowed classes may be sampled"
        pixel_values = batch[:, 0, 0, 0]
        assert (pixel_values % 100 == class_ids).all(), "images[b] must be an image of class class_ids[b]"
        seen_variants |= set((pixel_values // 100).long().tolist())
    assert len(seen_variants) == n_variants, "Every variant should get sampled sooner or later"
    a = sample_batch(images, allowed, 4, t.Generator().manual_seed(5))
    b = sample_batch(images, allowed, 4, t.Generator().manual_seed(5))
    assert t.equal(a[1], b[1]) and t.equal(a[0], b[0]), "The same generator seed should give the same batch"
    print("All tests in `test_sample_batch` passed!")


def test_patchify(patchify: Callable):
    t.manual_seed(0)
    for shape, p in [((2, 3, 8, 8), 4), ((1, 3, 32, 32), 8), ((2, 3, 4, 8), 2)]:
        x = t.randn(*shape)
        out = patchify(x, p)
        expected = F.unfold(x, p, stride=p).transpose(1, 2)
        assert out.shape == expected.shape, f"For input {shape} and patch size {p}, expected {tuple(expected.shape)}, got {tuple(out.shape)}"
        t.testing.assert_close(out, expected, msg="Patches must be row-major, each flattened in (channel, row, col) order")
    print("All tests in `test_patchify` passed!")


def test_self_attention(SelfAttention: type):
    t.manual_seed(0)
    d_model, n_heads = 16, 4
    x = t.randn(2, 5, d_model)
    for causal in [False, True]:
        attn = SelfAttention(d_model, n_heads, causal)
        out = attn(x)
        assert out.shape == x.shape, f"Expected output shape {tuple(x.shape)}, got {tuple(out.shape)}"
        # Independent oracle: PyTorch's fused attention, built from the same qkv weights
        head_dim = d_model // n_heads
        q, k, v = attn.qkv(x).reshape(2, 5, 3, n_heads, head_dim).unbind(2)
        mixed = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=causal)
        expected = attn.out(mixed.transpose(1, 2).reshape(2, 5, d_model))
        t.testing.assert_close(out, expected, atol=1e-5, rtol=1e-4, msg=f"Output differs from scaled_dot_product_attention (causal={causal})")
        # Causality: changing the last token must not change earlier outputs iff the layer is causal
        x2 = x.clone()
        x2[:, -1] += 1.0
        earlier_unchanged = t.allclose(attn(x2)[:, :-1], out[:, :-1], atol=1e-6)
        assert earlier_unchanged == causal, (
            "A causal layer must not let earlier positions see later tokens" if causal else "A non-causal layer should let every position see every token"
        )
    print("All tests in `test_self_attention` passed!")


def test_text_encoder(TextEncoder: type, eos_id: int):
    t.manual_seed(0)
    vocab_size, d_model, max_len = 12, 16, 10
    enc = TextEncoder(vocab_size, d_model, 2, 2, max_len).eval()
    captions = [[1, 3, 4, 5, 6, 2], [1, 3, 7, eos_id], [1, 8, 9, 10, 11, 3, 4, 5, 6, 2]]
    padded = t.zeros(len(captions), max_len, dtype=t.long)
    for row, ids in enumerate(captions):
        padded[row, : len(ids)] = t.tensor(ids)
    out = enc(padded)
    assert out.shape == (3, d_model), f"Expected shape (3, {d_model}), got {tuple(out.shape)}"
    # Independent oracle: because attention is causal, a caption encoded without any padding gives the same vector
    for row, ids in enumerate(captions):
        alone = enc(t.tensor([ids]))
        t.testing.assert_close(
            out[row], alone[0], atol=1e-5, rtol=1e-4, msg=f"Caption {row}: padding changed the output. Read the hidden state at the <eos> position, not at the last position"
        )
    assert not t.allclose(out[0], out[1], atol=1e-3), "Different captions should give different embeddings"
    print("All tests in `test_text_encoder` passed!")


def test_image_encoder(ImageEncoder: type, patchify: Callable):
    t.manual_seed(0)
    d_model = 16
    enc = ImageEncoder(d_model, 1, 2).eval()
    x = t.rand(3, 3, 32, 32)
    out = enc(x)
    assert out.shape == (3, d_model), f"Expected shape (3, {d_model}), got {tuple(out.shape)}"
    # Independent oracle: the same computation spelled out with F.unfold for the patches
    patches = F.unfold(x, 8, stride=8).transpose(1, 2)
    tokens = enc.patch_embed(patches)
    seq = t.cat([enc.cls_token.expand(3, -1, -1), tokens], dim=1) + enc.pos_embedding
    for block in enc.blocks:
        seq = block(seq)
    t.testing.assert_close(out, enc.ln_final(seq)[:, 0], atol=1e-5, rtol=1e-4, msg="Output differs from patch-embed, prepend [CLS], add positions, blocks, layernorm, read [CLS]")
    t.testing.assert_close(enc(x[:1]), out[:1], atol=1e-5, rtol=1e-4, msg="Images in a batch must not influence each other")
    assert not t.allclose(enc(x.flip(-1)), out, atol=1e-3), "Mirroring an image should change the embedding (position embeddings matter)"
    print("All tests in `test_image_encoder` passed!")


def test_tiny_clip(TinyCLIP: type, vocab: list[str], similarity_logits: Callable):
    t.manual_seed(0)
    images = t.rand(3, 3, 32, 32)
    ids = t.tensor([[1, 3, 4, 2, 0, 0], [1, 3, 5, 6, 2, 0], [1, 7, 8, 9, 2, 0]])
    for siglip in [False, True]:
        model = TinyCLIP(len(vocab), d_model=32, d_embed=16, n_layers=1, n_heads=2, max_len=6, siglip=siglip).eval()
        img_emb, txt_emb = model.encode_image(images), model.encode_text(ids)
        assert img_emb.shape == (3, 16), f"encode_image: expected (3, 16), got {tuple(img_emb.shape)}"
        assert txt_emb.shape == (3, 16), f"encode_text: expected (3, 16), got {tuple(txt_emb.shape)}"
        assert not t.allclose(img_emb.norm(dim=-1), t.ones(3), atol=1e-3), "encode_* should return raw, un-normalised embeddings (the normalisation happens in similarity_logits)"
        t.testing.assert_close(img_emb, model.image_proj(model.image_encoder(images)))
        t.testing.assert_close(txt_emb, model.text_proj(model.text_encoder(ids)))
        logits = model(images, ids)
        assert logits.shape == (3, 3), f"forward: expected (3, 3), got {tuple(logits.shape)}"
        bias = model.logit_bias if siglip else 0.0
        expected = similarity_logits(img_emb, txt_emb, model.logit_scale.exp(), bias)
        t.testing.assert_close(logits, expected, msg="forward should return similarity_logits(image_emb, text_emb, scale.exp(), bias)")
        if siglip:
            t.testing.assert_close(model.logit_scale.exp().item(), 10.0)
            t.testing.assert_close(model.logit_bias.item(), -10.0)
        else:
            assert model.logit_bias is None, "The CLIP model has no logit bias"
            t.testing.assert_close(model.logit_scale.exp().item(), 1 / 0.07, atol=1e-3, rtol=1e-4)
        model.train()
        model(images, ids).sum().backward()
        assert model.logit_scale.grad is not None, "The logit scale should receive a gradient"
    print("All tests in `test_tiny_clip` passed!")


def test_retrieval_metrics(retrieval_metrics: Callable):
    sim = t.tensor([[0.9, 0.1, 0.0], [0.2, 0.1, 0.8], [0.1, 0.7, 0.3], [0.0, 0.2, 0.9]])
    image_classes = t.tensor([0, 1, 1, 2])
    caption_classes = t.tensor([0, 1, 2])
    out = retrieval_metrics(sim, image_classes, caption_classes)
    assert set(out) == {"image_to_text", "text_to_image"}, f"Expected keys image_to_text and text_to_image, got {sorted(out)}"
    # image -> text: images pick captions 0, 2, 1, 2 -> classes 0, 2, 1, 2 vs truth 0, 1, 1, 2 -> 3 of 4
    # text -> image: captions pick images 0, 2, 3 -> classes 0, 1, 2 vs truth 0, 1, 2 -> 3 of 3
    t.testing.assert_close(out["image_to_text"], 0.75, msg="image_to_text should be 3/4 on this example")
    t.testing.assert_close(out["text_to_image"], 1.0, msg="text_to_image should be 3/3 on this example")
    # brute-force oracle on random non-square input with repeated classes
    t.manual_seed(0)
    sim = t.randn(30, 6)
    image_classes = t.randint(0, 6, (30,))
    caption_classes = t.arange(6)
    out = retrieval_metrics(sim, image_classes, caption_classes)
    i2t = np.mean([int(caption_classes[sim[i].argmax()]) == int(image_classes[i]) for i in range(30)])
    t2i = np.mean([int(image_classes[sim[:, j].argmax()]) == int(caption_classes[j]) for j in range(6)])
    t.testing.assert_close(out["image_to_text"], float(i2t))
    t.testing.assert_close(out["text_to_image"], float(t2i))
    print("All tests in `test_retrieval_metrics` passed!")


def test_make_control_pairs(make_control_pairs: Callable):
    classes = t.tensor([3, 7, 9, 2, 5, 11])
    generator = t.Generator().manual_seed(0)
    assert t.equal(make_control_pairs(classes, "paired", generator), classes), "'paired' should return the class ids unchanged"
    orders = set()
    for _ in range(40):
        shuffled = make_control_pairs(classes, "shuffled", generator)
        assert sorted(shuffled.tolist()) == sorted(classes.tolist()), "'shuffled' must be a rearrangement of the batch's classes"
        orders.add(tuple(shuffled.tolist()))
    assert len(orders) > 20, "'shuffled' should be a fresh random permutation each time (it may sometimes keep an item in place)"
    perm = t.tensor([5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6])
    assert t.equal(make_control_pairs(classes, "permuted", generator, perm), perm[classes]), "'permuted' should return fixed_perm[class_ids]"
    try:
        make_control_pairs(classes, "nonsense", generator)
    except ValueError:
        pass
    else:
        raise AssertionError("An unknown mode should raise a ValueError")
    print("All tests in `test_make_control_pairs` passed!")


@contextlib.contextmanager
def _small_cpu_run():
    """Tiny models run fastest on a couple of CPU threads (the default of one thread per core just adds overhead)."""
    old_threads = t.get_num_threads()
    t.set_num_threads(2)
    try:
        yield
    finally:
        t.set_num_threads(old_threads)


def test_train_tiny_clip(train_tiny_clip: Callable, TinyCLIP: type, tokenize_captions: Callable, vocab: list[str], max_len: int):
    with _small_cpu_run():
        _test_train_tiny_clip(train_tiny_clip, TinyCLIP, tokenize_captions, vocab, max_len)
    print("All tests in `test_train_tiny_clip` passed!")


def _test_train_tiny_clip(train_tiny_clip: Callable, TinyCLIP: type, tokenize_captions: Callable, vocab: list[str], max_len: int):
    colours = ["red", "green", "blue", "yellow", "purple", "orange"]
    rgb = t.tensor([[0.9, 0.1, 0.1], [0.1, 0.7, 0.2], [0.2, 0.3, 0.9], [0.95, 0.85, 0.1], [0.55, 0.25, 0.75], [0.95, 0.55, 0.1]])
    generator = t.Generator().manual_seed(0)
    # a micro world: six classes, each a solid colour with noise
    images = (rgb[None, :, :, None, None] + 0.05 * t.randn(8, 6, 3, 32, 32, generator=generator)).clamp(0, 1)
    caption_ids = tokenize_captions([f"a {c} circle in the top left" for c in colours], vocab, max_len)
    all_classes = t.arange(6)

    def fresh(**kwargs):
        t.manual_seed(0)
        return TinyCLIP(len(vocab), d_model=32, d_embed=16, n_layers=1, n_heads=2, max_len=max_len, **kwargs)

    def accuracy(model) -> float:
        with t.no_grad():
            model.eval()
            logits = model(images[0], caption_ids)
            model.train()
        return (logits.argmax(1) == t.arange(6)).float().mean().item()

    model = fresh()
    history = train_tiny_clip(model, images, caption_ids, all_classes, "clip", "paired", steps=150, batch_size=6, lr=3e-3, eval_fn=accuracy, eval_every=50)
    losses = history["loss"]
    assert len(losses) == 150, f"Expected one loss per step (150), got {len(losses)}"
    assert np.mean(losses[-20:]) < 0.5 * np.mean(losses[:20]), "The loss should fall substantially when training on correctly paired data"
    assert history["eval_steps"] == [50, 100, 150], f"eval_fn should be called every `eval_every` steps, got steps {history['eval_steps']}"
    assert history["eval"][-1] >= 0.8, f"A tiny model should be able to learn six solid colours, but retrieval accuracy is {history['eval'][-1]:.2f}"
    assert model.logit_scale.exp().item() <= 100.0 + 1e-3, "The logit scale must be clamped to at most 100"
    model = fresh()
    history = train_tiny_clip(model, images, caption_ids, all_classes, "clip", "shuffled", steps=150, batch_size=6, lr=3e-3)
    assert np.mean(history["loss"][-50:]) > 1.0, "With shuffled captions there is nothing to learn: the loss should stay near log(6) = 1.79"
    model = fresh(siglip=True)
    history = train_tiny_clip(model, images, caption_ids, all_classes, "siglip", "paired", steps=60, batch_size=6, lr=1e-3)
    assert all(math.isfinite(x) for x in history["loss"]), "The sigmoid-loss run produced non-finite losses"


# ---------------------------------------------------------------------------------------------------------------------
# Part 3: the real models
# ---------------------------------------------------------------------------------------------------------------------


def test_zero_shot_probabilities(zero_shot_probabilities: Callable):
    logits = t.tensor([[2.0, 0.0, -1.0], [0.0, 0.0, 0.0]])
    soft = zero_shot_probabilities(logits, "softmax")
    t.testing.assert_close(soft, F.softmax(logits, dim=-1), msg="'softmax' should normalise each row over the texts")
    t.testing.assert_close(soft.sum(-1), t.ones(2))
    sig = zero_shot_probabilities(logits, "sigmoid")
    t.testing.assert_close(sig, t.sigmoid(logits), msg="'sigmoid' should squash every entry independently")
    t.testing.assert_close(sig[1], t.full((3,), 0.5), msg="sigmoid(0) = 0.5")
    assert not t.allclose(sig.sum(-1), t.ones(2)), "Sigmoid probabilities are independent: they should not sum to 1"
    try:
        zero_shot_probabilities(logits, "nonsense")
    except ValueError:
        pass
    else:
        raise AssertionError("An unknown kind should raise a ValueError")
    print("All tests in `test_zero_shot_probabilities` passed!")


# ---------------------------------------------------------------------------------------------------------------------
# Part 4: geometry
# ---------------------------------------------------------------------------------------------------------------------


def test_modality_gap(modality_gap: Callable):
    images = t.tensor([[3.0, 0.0], [0.0, 4.0]])  # normalised: (1, 0) and (0, 1)
    texts = t.tensor([[2.0, 2.0], [5.0, 5.0], [1.0, 1.0]])  # all normalise to (1, 1) / sqrt(2)
    gap = modality_gap(images, texts)
    assert gap.shape == (2,), f"Expected a vector of shape (2,), got {tuple(gap.shape)}"
    expected = t.tensor([0.5 - 1 / math.sqrt(2), 0.5 - 1 / math.sqrt(2)])
    t.testing.assert_close(gap, expected, msg="The gap is mean(normalised images) - mean(normalised texts)")
    t.testing.assert_close(modality_gap(7 * images, 0.1 * texts), expected, msg="Rescaling the inputs must not change the gap")
    same = t.randn(5, 4)
    t.testing.assert_close(modality_gap(same, same), t.zeros(4), atol=1e-6, rtol=0)
    print("All tests in `test_modality_gap` passed!")


def test_accuracy_after_shift(accuracy_after_shift: Callable):
    texts = t.tensor([[1.0, 0.0], [0.0, 1.0]])
    images = t.tensor([[0.9, 0.1], [0.1, 0.9]])
    correct = t.tensor([0, 1])
    t.testing.assert_close(accuracy_after_shift(images, texts, correct, t.zeros(2)), 1.0, msg="With no shift both images retrieve their text")
    # Shifting by (1, 0) moves the second image's nearest text from 1 to 0 (hand-computed: cos 0.745 vs 0.667)
    t.testing.assert_close(accuracy_after_shift(images, texts, correct, t.tensor([1.0, 0.0])), 0.5)
    t.testing.assert_close(accuracy_after_shift(5 * images, 0.2 * texts, correct, t.tensor([1.0, 0.0])), 0.5, msg="Normalise images before adding the shift, and texts before comparing")
    print("All tests in `test_accuracy_after_shift` passed!")


def test_attribute_direction(attribute_direction: Callable):
    emb = t.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0]])
    values = ["a", "b", "a", "b"]
    s = 1 / math.sqrt(2)
    # normalised rows: (1, 0), (0, 1), (s, s), (1, 0); mean of a = ((1+s)/2, s/2), mean of b = (1/2, 1/2)
    expected = t.tensor([0.5 - (1 + s) / 2, 0.5 - s / 2])
    out = attribute_direction(emb, values, "a", "b")
    assert out.shape == (2,), f"Expected shape (2,), got {tuple(out.shape)}"
    t.testing.assert_close(out, expected, msg="direction = mean(normalised rows with value b) - mean(normalised rows with value a)")
    t.testing.assert_close(attribute_direction(emb * t.tensor([[3.0], [0.5], [9.0], [2.0]]), values, "a", "b"), expected, msg="Row norms must not matter")
    t.testing.assert_close(attribute_direction(emb, values, "b", "a"), -expected, msg="Swapping a and b should flip the sign")
    values3 = ["x", "a", "y", "b"]  # only rows 1 (value a) and 3 (value b) count: normalised (0, 1) and (1, 0)
    t.testing.assert_close(attribute_direction(emb, values3, "a", "b"), t.tensor([1.0, -1.0]), msg="Rows with other values must be ignored")
    print("All tests in `test_attribute_direction` passed!")


def test_steer_and_retrieve(steer_and_retrieve: Callable):
    texts = t.tensor([[1.0, 0.0], [0.0, 1.0]])
    images = t.tensor([[3.0, 0.3]])
    direction = t.tensor([-1.0, 1.0])
    out = steer_and_retrieve(images, direction, 0.0, texts)
    assert out.shape == (1,), f"Expected shape (n,) = (1,), got {tuple(out.shape)}"
    assert out.dtype == t.long, "Return the indices of the best texts (an integer tensor)"
    assert out.item() == 0, "With alpha = 0 the image should retrieve text 0"
    assert steer_and_retrieve(images, direction, 1.0, texts).item() == 1, "Normalised image + direction is (-0.005, 1.0995): nearest text is 1"
    assert steer_and_retrieve(10 * images, direction, 1.0, 0.1 * texts).item() == 1, "Inputs must be normalised before the shift is added"
    many = steer_and_retrieve(t.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.1]]), direction, 0.5, texts)
    assert many.shape == (3,)
    print("All tests in `test_steer_and_retrieve` passed!")


def test_variance_partition(variance_partition: Callable):
    # Hand-worked: embeddings depend only on factor A
    a_labels = ["a0", "a1"] * 3
    b_labels = ["b0", "b0", "b1", "b1", "b2", "b2"]
    emb = t.tensor([[1.0, 0.0], [0.0, 1.0]] * 3)
    out = variance_partition(emb, {"A": a_labels, "B": b_labels})
    assert set(out) == {"A", "B", "interaction"}, f"Expected keys A, B and interaction, got {sorted(out)}"
    t.testing.assert_close(out["A"], 1.0, atol=1e-6, rtol=0)
    t.testing.assert_close(out["B"], 0.0, atol=1e-6, rtol=0)
    t.testing.assert_close(out["interaction"], 0.0, atol=1e-6, rtol=0)
    # Oracle: numpy one-way ANOVA sums of squares on random data
    t.manual_seed(0)
    n = 24
    labels_x = [i % 4 for i in range(n)]
    labels_y = [(i // 4) % 3 for i in range(n)]
    emb = t.randn(n, 6)
    out = variance_partition(emb, {"x": labels_x, "y": labels_y})
    x = F.normalize(emb, dim=-1).double().numpy()
    total = ((x - x.mean(0)) ** 2).sum()
    for name, labels in [("x", labels_x), ("y", labels_y)]:
        labels = np.array(labels)
        between = sum((labels == v).sum() * ((x[labels == v].mean(0) - x.mean(0)) ** 2).sum() for v in np.unique(labels))
        t.testing.assert_close(out[name], float(between / total), atol=1e-5, rtol=1e-5, msg=f"Wrong eta-squared for factor {name}")
    t.testing.assert_close(out["interaction"], 1 - out["x"] - out["y"], atol=1e-6, rtol=0)
    print("All tests in `test_variance_partition` passed!")


def test_winoground_scores(winoground_scores: Callable):
    #                 item: 1    2    3    4    5
    s_aa = t.tensor([1.0, 1.0, 3.0, 0.0, 0.0])
    s_ab = t.tensor([0.0, 0.0, 4.0, 1.0, 0.0])
    s_ba = t.tensor([0.0, 2.0, 0.0, 1.0, 0.0])
    s_bb = t.tensor([1.0, 3.0, 5.0, 0.0, 0.0])
    out = winoground_scores(s_aa, s_ab, s_ba, s_bb)
    assert set(out) == {"text", "image", "group", "pair_accuracy"}, f"Expected keys text, image, group, pair_accuracy; got {sorted(out)}"
    # item 1: all correct. item 2: text ok, image wrong. item 3: image ok, text wrong. item 4: both wrong. item 5: all ties -> failures
    t.testing.assert_close(out["text"], 0.4, msg="text score: s_aa > s_ab and s_bb > s_ba (strictly)")
    t.testing.assert_close(out["image"], 0.4, msg="image score: s_aa > s_ba and s_bb > s_ab (strictly)")
    t.testing.assert_close(out["group"], 0.2, msg="group score: both")
    t.testing.assert_close(out["pair_accuracy"], 0.5, msg="pair accuracy: mean over both images of 'prefers its own caption'")
    print("All tests in `test_winoground_scores` passed!")
