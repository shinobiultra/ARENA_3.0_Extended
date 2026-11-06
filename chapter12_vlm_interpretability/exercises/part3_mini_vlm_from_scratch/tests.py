import copy
from typing import Callable

import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from transformer_lens import HookedTransformer

import part3_mini_vlm_from_scratch.utils as utils
from part3_mini_vlm_from_scratch.utils import Scene, SceneObject


# ---------------------------------------------------------------------------------------------
# 1. A world to ask questions about
# ---------------------------------------------------------------------------------------------


def test_question_and_answer(question_and_answer: Callable):
    scene = Scene(
        [
            SceneObject("red", "circle", (0, 1)),
            SceneObject("blue", "square", (1, 0)),
            SceneObject("green", "triangle", (1, 1)),
        ]
    )
    cases = [
        (("colour", 0), " Question: What colour is the circle? Answer:", " red"),
        (("colour", 2), " Question: What colour is the triangle? Answer:", " green"),
        (("shape", 1), " Question: What shape is the blue object? Answer:", " square"),
        (("count", 0), " Question: How many shapes are there? Answer:", " three"),
        (("horizontal", 0), " Question: Is the circle on the left or the right? Answer:", " right"),
        (("horizontal", 1), " Question: Is the square on the left or the right? Answer:", " left"),
        (("vertical", 0), " Question: Is the circle at the top or the bottom? Answer:", " top"),
        (("vertical", 2), " Question: Is the triangle at the top or the bottom? Answer:", " bottom"),
    ]
    for (kind, target), prompt, answer in cases:
        got_prompt, got_answer = question_and_answer(scene, kind, target)
        assert got_prompt == prompt, f"kind={kind!r}, target={target}: expected prompt {prompt!r}, got {got_prompt!r}"
        assert got_answer == answer, f"kind={kind!r}, target={target}: expected answer {answer!r}, got {got_answer!r}"
    # The count question doesn't depend on the target, and depends on the number of objects
    for n, word in [(1, " one"), (2, " two")]:
        small = Scene(scene.objects[:n])
        assert question_and_answer(small, "count", 0)[1] == word, f"With {n} objects the answer should be {word!r}"
    print("All tests in `test_question_and_answer` passed!")


def test_best_constant_guess(best_constant_guess: Callable):
    train = {"colour": [" red", " red", " blue", " green"], "count": [" two", " two", " three"]}
    test = {"colour": [" red", " blue", " red", " red"], "count": [" three", " two", " two", " one"]}
    result = best_constant_guess(train, test)
    assert set(result) == {"colour", "count"}, f"Expected one entry per kind, got keys {sorted(result)}"
    guess, acc = result["colour"]
    assert guess == " red", f"The most common training answer for 'colour' is ' red', got {guess!r}"
    assert abs(acc - 0.75) < 1e-9, f"' red' is right on 3 of 4 test colour questions, got accuracy {acc}"
    guess, acc = result["count"]
    assert guess == " two" and abs(acc - 0.5) < 1e-9, f"Expected (' two', 0.5) for 'count', got ({guess!r}, {acc})"
    print("All tests in `test_best_constant_guess` passed!")


def test_extract_patch_features(extract_patch_features: Callable, vision_model):
    from transformers import SiglipImageProcessor

    rng = np.random.default_rng(0)
    images = utils.render_scenes([utils.sample_scene(rng) for _ in range(3)])
    features = extract_patch_features(vision_model, images)
    assert features.shape == (3, 196, 768), f"Expected (3, 196, 768) (14x14 patches of width 768), got {tuple(features.shape)}"
    assert not features.requires_grad, "The vision encoder is frozen: no gradients should be tracked"
    # Independent oracle: Hugging Face's own preprocessing (rescale to [0, 1], then normalise with mean = std = 0.5)
    processor = SiglipImageProcessor.from_pretrained("google/siglip-base-patch16-224", revision="7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed")
    pixel_values = processor(list(images), do_resize=False, return_tensors="pt")["pixel_values"]
    with t.no_grad():
        expected = vision_model(pixel_values=pixel_values.to(vision_model.device, vision_model.dtype)).last_hidden_state
    rel = ((features.float() - expected.float()).norm() / expected.float().norm()).item()
    assert rel < 0.02, f"Features differ from Hugging Face's preprocessing + SigLIP by {rel:.1%} (relative). Check the normalisation."
    print("All tests in `test_extract_patch_features` passed!")


# ---------------------------------------------------------------------------------------------
# 2. Building the VLM
# ---------------------------------------------------------------------------------------------


def test_merge_patches(merge_patches: Callable):
    grid, d = 6, 2
    rows, cols = t.meshgrid(t.arange(grid), t.arange(grid), indexing="ij")
    coded = t.stack([rows.flatten(), cols.flatten()], dim=-1).float()[None].repeat(2, 1, 1)  # each patch holds (row, col)
    out = merge_patches(coded, 2)
    assert out.shape == (2, 9, 8), f"Expected (batch, 9 tokens, 4 patches * 2 features) = (2, 9, 8), got {tuple(out.shape)}"
    for i in range(3):
        for j in range(3):
            expected = [v for (r, c) in [(2 * i, 2 * j), (2 * i, 2 * j + 1), (2 * i + 1, 2 * j), (2 * i + 1, 2 * j + 1)] for v in (r, c)]
            got = out[0, 3 * i + j].tolist()
            assert got == expected, (
                f"Token (row {i}, col {j}) should hold the 2x2 block of patches [(2i,2j), (2i,2j+1), (2i+1,2j), (2i+1,2j+1)] "
                f"= {expected} (as (row, col) pairs), got {got}"
            )
    out3 = merge_patches(t.randn(1, 36, 5), 3)
    assert out3.shape == (1, 4, 45), f"With merge=3 a 6x6 grid should give 4 tokens of width 9*5, got {tuple(out3.shape)}"
    print("All tests in `test_merge_patches` passed!")


def test_projector(Projector: type):
    t.manual_seed(0)
    projector = Projector(d_vision=6, d_model=5, merge=2)
    n_params = sum(p.numel() for p in projector.parameters())
    assert n_params == (24 * 5 + 5) + (5 * 5 + 5), (
        f"Expected a two-layer MLP Linear(4*6 -> 5) -> GELU -> Linear(5 -> 5) with {24 * 5 + 5 + 5 * 5 + 5} parameters, got {n_params}"
    )
    x = t.randn(2, 16, 6, requires_grad=True)  # a 4x4 grid of patches
    out = projector(x)
    assert out.shape == (2, 4, 5), f"Expected (batch, 4 image tokens, d_model) = (2, 4, 5), got {tuple(out.shape)}"
    # Image token 0 may only depend on the 2x2 block of patches in the top-left corner (patches 0, 1, 4, 5)
    out[0, 0].sum().backward()
    touched = (x.grad[0].abs().sum(-1) > 0).nonzero().flatten().tolist()
    assert touched == [0, 1, 4, 5], f"The first image token should be computed from patches [0, 1, 4, 5] only, but depends on {touched}"
    assert x.grad[1].abs().sum() == 0, "Different images in the batch must not interact"
    # It must be nonlinear: a purely linear map f would satisfy f(a + b) - f(0) = (f(a) - f(0)) + (f(b) - f(0))
    a, b = t.randn(1, 16, 6), t.randn(1, 16, 6)
    zero = projector(t.zeros(1, 16, 6))
    assert not t.allclose(projector(a + b) - zero, (projector(a) - zero) + (projector(b) - zero), atol=1e-4), (
        "The projector looks linear: put a GELU between the two linear layers"
    )
    print("All tests in `test_projector` passed!")


def test_build_input_embeddings(build_input_embeddings: Callable, model: HookedTransformer):
    bos = model.tokenizer.bos_token_id
    image_ids = utils.encode(model, " a cat sat")
    question_ids = utils.encode(model, " on the mat?")
    # Oracle: if the "image embeddings" are just the embeddings of some words, then the multimodal input is exactly
    # the ordinary text input, whose embedding + positional embedding TransformerLens records for us
    full = t.tensor([[bos] + image_ids + question_ids], device=model.cfg.device)
    _, cache = model.run_with_cache(full, names_filter=["hook_embed", "hook_pos_embed"])
    expected = cache["hook_embed"] + cache["hook_pos_embed"]
    image_embeds = model.W_E[t.tensor([image_ids], device=model.cfg.device)]
    question_tokens = t.tensor([question_ids], device=model.cfg.device)
    out = build_input_embeddings(model, image_embeds, question_tokens)
    assert out.shape == expected.shape, f"Expected shape {tuple(expected.shape)} (BOS + image + question), got {tuple(out.shape)}"
    t.testing.assert_close(out, expected, atol=1e-4, rtol=1e-4, msg="Wrong input: check the order [BOS, image, question] and the positional embeddings")
    # Text-only version: no image embeddings
    text_only = build_input_embeddings(model, None, question_tokens)
    full_text = t.tensor([[bos] + question_ids], device=model.cfg.device)
    _, cache = model.run_with_cache(full_text, names_filter=["hook_embed", "hook_pos_embed"])
    t.testing.assert_close(text_only, cache["hook_embed"] + cache["hook_pos_embed"], atol=1e-4, rtol=1e-4, msg="With image_embeds=None the input should be [BOS, question]")
    # Batches: every row gets the same BOS
    batch = build_input_embeddings(model, t.cat([image_embeds] * 3), question_tokens.expand(3, -1))
    assert batch.shape[0] == 3
    t.testing.assert_close(batch[2], expected[0], atol=1e-4, rtol=1e-4)
    print("All tests in `test_build_input_embeddings` passed!")


class _TokenLookupProjector(nn.Module):
    """A stand-in 'projector' for testing: the 'features' are token ids, and the 'image tokens' are those words' embeddings."""

    def __init__(self, model: HookedTransformer):
        super().__init__()
        self.model = model

    def forward(self, ids: t.Tensor) -> t.Tensor:
        return self.model.W_E[ids]


def test_vlm_answer_logits(vlm_answer_logits: Callable, model: HookedTransformer):
    bos = model.tokenizer.bos_token_id
    image_ids = utils.encode(model, " a cat sat")
    questions = [utils.encode(model, " on the mat?"), utils.encode(model, " on the big red mat in the house?")]
    pad = model.tokenizer.eos_token_id
    width = max(len(q) for q in questions)
    tokens = t.tensor([q + [pad] * (width - len(q)) for q in questions], device=model.cfg.device)  # right-padded
    lens = t.tensor([len(q) for q in questions], device=model.cfg.device)
    features = t.tensor([image_ids] * 2, device=model.cfg.device)
    out = vlm_answer_logits(model, _TokenLookupProjector(model), features, tokens, lens)
    assert out.shape == (2, model.cfg.d_vocab), f"Expected one row of vocabulary logits per example, got {tuple(out.shape)}"
    for i, question in enumerate(questions):
        full = t.tensor([[bos] + image_ids + question], device=model.cfg.device)
        with t.no_grad():
            expected = model(full)[0, -1]
        t.testing.assert_close(
            out[i], expected, atol=2e-3, rtol=2e-3,
            msg=f"Example {i}: should equal the ordinary GPT-2 logits at the last real question token (padding must be ignored)",
        )
    print("All tests in `test_vlm_answer_logits` passed!")


def test_answer_accuracy(answer_accuracy: Callable):
    logits = t.tensor(
        [
            [9.0, 0.0, 3.0, 2.0],  # best token overall is 0 (not a candidate); among candidates {1, 2, 3} the best is 2
            [0.0, 5.0, 1.0, 1.5],  # best candidate: 1
            [0.0, 1.0, 1.0, 4.0],  # best candidate: 3
            [1.0, 2.0, 2.5, 0.0],  # best candidate: 2
        ]
    )
    answers = t.tensor([2, 1, 2, 2])
    acc = answer_accuracy(logits, answers, t.tensor([1, 2, 3]))
    assert isinstance(acc, float), "Return a Python float"
    assert abs(acc - 0.75) < 1e-9, f"Only candidates should be compared (3 of 4 correct = 0.75), got {acc}"
    print("All tests in `test_answer_accuracy` passed!")


# ---------------------------------------------------------------------------------------------
# 3. Training and the modality controls
# ---------------------------------------------------------------------------------------------


def test_make_optimizer(make_optimizer: Callable, Projector: type, model: HookedTransformer):
    lm = copy.deepcopy(model)
    projector = Projector()
    opt = make_optimizer(lm, projector, train_lm=False, lr_projector=1e-3, lr_lm=1e-4)
    assert isinstance(opt, t.optim.AdamW), "Use AdamW"
    assert not any(p.requires_grad for p in lm.parameters()), "With train_lm=False the language model must be frozen"
    assert all(p.requires_grad for p in projector.parameters()), "The projector is always trained"
    n_opt = sum(p.numel() for g in opt.param_groups for p in g["params"])
    assert n_opt == sum(p.numel() for p in projector.parameters()), "With train_lm=False the optimiser should only hold projector parameters"
    opt = make_optimizer(lm, projector, train_lm=True, lr_projector=1e-3, lr_lm=1e-4)
    assert all(p.requires_grad for p in lm.parameters()), "With train_lm=True the language model must be trainable"
    lrs = sorted(g["lr"] for g in opt.param_groups)
    assert lrs == [1e-4, 1e-3], f"Expected two parameter groups with learning rates 1e-4 (LM) and 1e-3 (projector), got {lrs}"
    for g in opt.param_groups:
        n = sum(p.numel() for p in g["params"])
        expected = sum(p.numel() for p in (lm if g["lr"] == 1e-4 else projector).parameters())
        assert n == expected, f"The group with lr={g['lr']} holds {n} parameters, expected {expected}"
    del lm
    print("All tests in `test_make_optimizer` passed!")


def test_train_vlm(train_vlm: Callable, Projector: type, model: HookedTransformer):
    rng = np.random.default_rng(0)
    scenes = [utils.sample_scene(rng) for _ in range(16)]
    features = t.randn(16, 196, 768, device=model.cfg.device).half()
    for train_lm in [False, True]:
        lm = copy.deepcopy(model)
        t.manual_seed(0)
        projector = Projector().to(model.cfg.device)
        lm_before = {k: v.clone() for k, v in lm.state_dict().items()}
        proj_before = [p.detach().clone() for p in projector.parameters()]
        losses = train_vlm(lm, projector, features, scenes, n_steps=25, train_lm=train_lm, batch_size=16, lr_projector=3e-3, lr_lm=1e-4)
        assert len(losses) == 25 and all(np.isfinite(losses)), "Return one finite loss per step"
        assert losses[-1] < losses[0] - 2, f"The loss should fall quickly when memorising 16 scenes, but went from {losses[0]:.2f} to {losses[-1]:.2f}"
        proj_changed = any(not t.equal(a, b) for a, b in zip(proj_before, projector.parameters()))
        assert proj_changed, "The projector should be updated"
        lm_changed = any(not t.equal(lm_before[k], v) for k, v in lm.state_dict().items())
        if train_lm:
            assert lm_changed, "With train_lm=True the language model should be updated"
        else:
            assert not lm_changed, "With train_lm=False the language model must stay exactly as it was"
        del lm
    print("All tests in `test_train_vlm` passed!")


def test_evaluate_vlm(evaluate_vlm: Callable, Projector: type, model: HookedTransformer, vlm_answer_logits: Callable):
    rng = np.random.default_rng(1)
    scenes = [utils.sample_scene(rng) for _ in range(12)]
    t.manual_seed(0)
    projector = Projector().to(model.cfg.device)
    features = t.randn(12, 196, 768, device=model.cfg.device).half()
    words = {"colour": list(utils.COLOURS), "count": ["one", "two", "three"]}
    number = ["one", "two", "three"]
    questions = {  # (prompts, answers); the labels are for scene i, the features are random, so accuracies will be arbitrary
        "colour": (
            [f" Question: What colour is the {s.objects[0].shape}? Answer:" for s in scenes],
            [" " + s.objects[0].colour for s in scenes],
        ),
        "count": (
            [" Question: How many shapes are there? Answer:" for s in scenes],
            [" " + number[len(s.objects) - 1] for s in scenes],
        ),
    }
    result = evaluate_vlm(model, projector, features, questions, batch_size=5)
    assert set(result) == set(questions), f"Expected one accuracy per kind, got {sorted(result)}"
    # Brute force oracle: one example at a time, argmax over the valid answer words only
    with t.no_grad():
        for kind, (prompts, answers) in questions.items():
            cand = [model.to_single_token(" " + w) for w in words[kind]]
            correct = 0
            for i, (prompt, answer) in enumerate(zip(prompts, answers)):
                tokens, lens = utils.tokenize_prompts(model, [prompt])
                logits = vlm_answer_logits(model, projector, features[i : i + 1], tokens, lens)[0]
                correct += cand[int(logits[cand].argmax())] == model.to_single_token(answer)
            assert abs(result[kind] - correct / len(prompts)) < 1e-6, f"{kind}: expected accuracy {correct / len(prompts):.3f}, got {result[kind]:.3f}"
    blank = evaluate_vlm(model, projector, features, questions, use_image=False)
    assert set(blank) == set(questions), "use_image=False should still return one accuracy per kind"
    print("All tests in `test_evaluate_vlm` passed!")


def test_derangement(derangement: Callable):
    for n in [2, 3, 10, 257]:
        for seed in range(4):
            perm = derangement(n, seed)
            assert perm.shape == (n,), f"Expected shape ({n},), got {tuple(perm.shape)}"
            assert sorted(perm.tolist()) == list(range(n)), f"n={n}, seed={seed}: not a permutation of 0..{n - 1}"
            assert (perm != t.arange(n)).all(), f"n={n}, seed={seed}: index {int((perm == t.arange(n)).nonzero()[0])} maps to itself"
    assert not t.equal(derangement(50, 0), derangement(50, 1)), "Different seeds should give different permutations"
    assert t.equal(derangement(50, 3), derangement(50, 3)), "The same seed should give the same permutation"
    print("All tests in `test_derangement` passed!")


# ---------------------------------------------------------------------------------------------
# 4. Where does the answer come from?
# ---------------------------------------------------------------------------------------------


def test_random_background_mask(random_background_mask: Callable):
    t.manual_seed(0)
    batch, n = 30, 49
    object_masks = t.zeros(batch, n, dtype=t.bool)
    occupied = t.zeros(batch, n, dtype=t.bool)
    for b in range(batch):
        owned = t.randperm(n)[: t.randint(10, 25, ()).item()]  # tokens covered by some object
        occupied[b, owned] = True
        object_masks[b, owned[: t.randint(3, 8, ()).item()]] = True
    out = random_background_mask(object_masks, occupied, seed=0)
    assert out.shape == (batch, n) and out.dtype == t.bool, f"Expected a boolean (batch, token) mask, got {out.dtype} {tuple(out.shape)}"
    assert (out.sum(1) == object_masks.sum(1)).all(), "Each row must select exactly as many tokens as the object has"
    assert not (out & occupied).any(), "The control must never include a token that overlaps an object"
    assert t.equal(out, random_background_mask(object_masks, occupied, seed=0)), "The same seed must give the same mask"
    assert not t.equal(out, random_background_mask(object_masks, occupied, seed=1)), "Different seeds should give different masks"
    # Uniformity: with identical rows, every free token should be chosen about equally often
    row_obj = t.zeros(1, n, dtype=t.bool)
    row_obj[0, :5] = True
    row_occ = t.zeros(1, n, dtype=t.bool)
    row_occ[0, :9] = True  # 40 free tokens
    counts = t.zeros(n)
    for seed in range(400):
        counts += random_background_mask(row_obj, row_occ, seed=seed)[0].float()
    free_counts = counts[9:]
    assert free_counts.min() > 0.5 * free_counts.mean() and free_counts.max() < 1.5 * free_counts.mean(), (
        "Free tokens should be chosen uniformly at random"
    )
    print("All tests in `test_random_background_mask` passed!")


def test_lens_ranks(lens_ranks: Callable, model: HookedTransformer):
    t.manual_seed(0)
    resid = t.randn(3, 4, model.cfg.d_model, device=model.cfg.device) * 5
    word = model.to_single_token(" blue")
    ranks = lens_ranks(model, resid, word)
    assert ranks.shape == (3, 4), f"Expected (layer, token) = (3, 4), got {tuple(ranks.shape)}"
    for layer in range(3):
        for token in range(4):
            logits = model.unembed(model.ln_final(resid[layer, token]))  # brute force, one vector at a time
            expected = int((logits > logits[word]).sum()) + 1
            assert int(ranks[layer, token]) == expected, f"layer {layer}, token {token}: expected rank {expected}, got {int(ranks[layer, token])}"
    # A vector pointing at the word's output direction should rank it first
    direction = model.W_U[:, word]
    top = lens_ranks(model, (direction * 50)[None, None], word)
    assert int(top) == 1, f"A huge multiple of the word's unembedding direction should have rank 1, got {int(top)}"
    print("All tests in `test_lens_ranks` passed!")


def test_restricted_lens_correct(restricted_lens_correct: Callable, model: HookedTransformer):
    t.manual_seed(0)
    resid = t.randn(3, 6, model.cfg.d_model, device=model.cfg.device) * 5
    candidates = t.tensor([model.to_single_token(" " + c) for c in utils.COLOURS], device=model.cfg.device)
    for target in [0, 3]:
        out = restricted_lens_correct(model, resid, candidates, target)
        assert out.shape == (3, 6) and out.dtype == t.bool, f"Expected a boolean (layer, token) tensor, got {out.dtype} {tuple(out.shape)}"
        full = model.unembed(model.ln_final(resid))[..., candidates]  # oracle: full-vocabulary logits, then pick the candidates
        expected = full.argmax(-1) == target
        assert t.equal(out, expected), f"target={target}: differs from the oracle on {int((out != expected).sum())} entries"
    print("All tests in `test_restricted_lens_correct` passed!")


def test_attention_mass(attention_mass: Callable):
    t.manual_seed(0)
    patterns = t.randn(2, 3, 6, 6).softmax(-1)  # (layer, head, query, key); each row sums to 1
    key_mask = t.tensor([False, True, False, True, False, False])
    out = attention_mass(patterns, 4, key_mask)
    assert out.shape == (2, 3), f"Expected (layer, head) = (2, 3), got {tuple(out.shape)}"
    for layer in range(2):
        for head in range(3):
            expected = patterns[layer, head, 4, 1] + patterns[layer, head, 4, 3]
            t.testing.assert_close(out[layer, head], expected, msg=f"layer {layer}, head {head}")
    t.testing.assert_close(attention_mass(patterns, 2, t.ones(6, dtype=t.bool)), t.ones(2, 3), msg="All keys together should have mass 1")
    t.testing.assert_close(attention_mass(patterns, 2, t.zeros(6, dtype=t.bool)), t.zeros(2, 3), msg="No keys should have mass 0")
    print("All tests in `test_attention_mass` passed!")


def _random_inputs(model: HookedTransformer, batch: int, seq: int, seed: int) -> t.Tensor:
    g = t.Generator().manual_seed(seed)
    tokens = t.randint(0, 5000, (batch, seq), generator=g).to(model.cfg.device)
    return (model.W_E[tokens] + model.W_pos[:seq]).detach()


def test_patched_answer_logits(patched_answer_logits: Callable, model: HookedTransformer):
    batch, seq, answer_pos = 3, 12, 11
    n_layers = model.cfg.n_layers
    corrupt, clean = _random_inputs(model, batch, seq, 0), _random_inputs(model, batch, seq, 1)
    with t.no_grad():
        _, clean_cache = model.run_with_cache(clean, start_at_layer=0, stop_at_layer=n_layers, names_filter=lambda n: n.endswith("resid_pre"))

        def plain(x):
            return model.unembed(model.ln_final(model(x, start_at_layer=0, stop_at_layer=n_layers)[:, answer_pos]))

        corrupt_before = corrupt.clone()
        none = t.zeros(batch, seq, dtype=t.bool, device=corrupt.device)
        everything = t.ones_like(none)
        # A partial patch at layer 0 is the same as feeding the mixed embeddings through the model
        mask = t.zeros(batch, seq, dtype=t.bool, device=corrupt.device)
        mask[0, 2:6] = True
        mask[1, 5] = True
        mask[2, 9:11] = True
        out = patched_answer_logits(model, corrupt, clean_cache["resid_pre", 0], 0, mask, answer_pos)
        assert out.shape == (batch, model.cfg.d_vocab), f"Expected (batch, d_vocab), got {tuple(out.shape)}"
        assert t.equal(corrupt, corrupt_before), "You modified the corrupted input in place: clone the residual stream inside the hook"
        t.testing.assert_close(out, plain(t.where(mask[..., None], clean, corrupt)), atol=1e-3, rtol=1e-3, msg="Partial patch at layer 0")
        out = patched_answer_logits(model, corrupt, clean_cache["resid_pre", 0], 0, none, answer_pos)
        t.testing.assert_close(out, plain(corrupt), atol=1e-3, rtol=1e-3, msg="Patching nothing should reproduce the corrupted run")
        for layer in [0, 4, n_layers - 1]:
            out = patched_answer_logits(model, corrupt, clean_cache["resid_pre", layer], layer, everything, answer_pos)
            t.testing.assert_close(out, plain(clean), atol=1e-3, rtol=1e-3, msg=f"Patching every position at layer {layer} should reproduce the clean run")
        # Patching only the answer position at layer 6 must change the answer relative to the corrupted run
        mask = t.zeros(batch, seq, dtype=t.bool, device=corrupt.device)
        mask[:, 11] = True
        out = patched_answer_logits(model, corrupt, clean_cache["resid_pre", 6], 6, mask, answer_pos)
        assert not t.allclose(out, plain(corrupt), atol=1e-2), "Patching the answer position at layer 6 should change the answer logits"
    assert all(len(hook.fwd_hooks) == 0 for hook in model.hook_dict.values()), "Hooks were left attached to the model (use run_with_hooks)"
    print("All tests in `test_patched_answer_logits` passed!")


def test_logit_diff_recovery(logit_diff_recovery: Callable):
    clean = t.tensor([[3.0, 0.0, 1.0], [0.0, 2.0, 5.0]])
    corrupt = t.tensor([[0.0, 4.0, 1.0], [1.0, 0.0, 1.0]])
    clean_ids, corrupt_ids = t.tensor([0, 2]), t.tensor([1, 0])
    # logit differences: clean runs 3 and 4; corrupted runs -4 and 0
    out = logit_diff_recovery(clean, clean, corrupt, clean_ids, corrupt_ids)
    t.testing.assert_close(out, t.ones(2), msg="Patched = clean should give recovery 1")
    out = logit_diff_recovery(corrupt, clean, corrupt, clean_ids, corrupt_ids)
    t.testing.assert_close(out, t.zeros(2), msg="Patched = corrupted should give recovery 0")
    out = logit_diff_recovery((clean + corrupt) / 2, clean, corrupt, clean_ids, corrupt_ids)
    t.testing.assert_close(out, t.full((2,), 0.5), msg="Halfway logits should give recovery 0.5")
    # a hand-worked value: patched logit difference of 0.5 on example 0 -> (0.5 - (-4)) / (3 - (-4)) = 4.5 / 7
    patched = t.tensor([[1.0, 0.5, 0.0], [0.0, 2.0, 5.0]])
    out = logit_diff_recovery(patched, clean, corrupt, clean_ids, corrupt_ids)
    t.testing.assert_close(out[0], t.tensor(4.5 / 7), msg="Example 0 should be (0.5 + 4) / (3 + 4)")
    assert out.shape == (2,), f"Expected one recovery per example, got shape {tuple(out.shape)}"
    print("All tests in `test_logit_diff_recovery` passed!")


def test_patching_sweep(patching_sweep: Callable, model: HookedTransformer):
    batch, seq, answer_pos = 4, 12, 11
    n_layers = model.cfg.n_layers
    corrupt, clean = _random_inputs(model, batch, seq, 2), _random_inputs(model, batch, seq, 3)
    with t.no_grad():
        _, cache = model.run_with_cache(clean, start_at_layer=0, stop_at_layer=n_layers, names_filter=lambda n: n.endswith("resid_pre"))
        clean_resids = {layer: cache["resid_pre", layer] for layer in range(n_layers)}

        def answer_logits(x):
            return model.unembed(model.ln_final(model(x, start_at_layer=0, stop_at_layer=n_layers)[:, answer_pos]))

        clean_logits, corrupt_logits = answer_logits(clean), answer_logits(corrupt)
    clean_ids = clean_logits.argmax(-1)
    corrupt_ids = corrupt_logits.argmax(-1)
    assert (clean_ids != corrupt_ids).all(), "test setup: the two runs should have different top tokens"
    none = t.zeros(batch, seq, dtype=t.bool, device=clean.device)
    masks = {"nothing": none, "everything": t.ones_like(none)}
    curves = patching_sweep(model, corrupt, clean_resids, masks, clean_logits, corrupt_logits, clean_ids, corrupt_ids, answer_pos, [0, 5, 11])
    assert set(curves) == set(masks), f"Expected one curve per mask, got {sorted(curves)}"
    assert all(len(v) == 3 for v in curves.values()), "Each curve should have one value per layer in `layers`"
    assert all(isinstance(x, float) for v in curves.values() for x in v), "Return Python floats"
    assert np.allclose(curves["nothing"], 0, atol=1e-3), f"Patching nothing should recover 0, got {curves['nothing']}"
    assert np.allclose(curves["everything"], 1, atol=1e-3), f"Patching everything should recover 1, got {curves['everything']}"
    print("All tests in `test_patching_sweep` passed!")


def test_block_attention_logits(block_attention_logits: Callable, model: HookedTransformer):
    batch, seq, answer_pos = 3, 12, 11
    n_layers = model.cfg.n_layers
    base = _random_inputs(model, batch, seq, 4)
    other = base.clone()
    other[:, 1:5] = _random_inputs(model, batch, seq, 5)[:, 1:5]  # a different "image" in positions 1-4

    def plain(x):
        with t.no_grad():
            return model.unembed(model.ln_final(model(x, start_at_layer=0, stop_at_layer=n_layers)[:, answer_pos]))

    with t.no_grad():
        none = t.zeros(batch, seq, seq, dtype=t.bool, device=base.device)
        out = block_attention_logits(model, base, list(range(n_layers)), none, answer_pos)
        assert out.shape == (batch, model.cfg.d_vocab), f"Expected (batch, d_vocab), got {tuple(out.shape)}"
        t.testing.assert_close(out, plain(base), atol=1e-3, rtol=1e-3, msg="Blocking nothing should reproduce the normal forward pass")
        # Oracle: if no later position can ever read positions 1-4, the answer cannot depend on what is in them
        block = t.zeros(batch, seq, seq, dtype=t.bool, device=base.device)
        block[:, 5:, 1:5] = True
        a = block_attention_logits(model, base, list(range(n_layers)), block, answer_pos)
        b = block_attention_logits(model, other, list(range(n_layers)), block, answer_pos)
        t.testing.assert_close(a, b, atol=1e-3, rtol=1e-3, msg="With positions 1-4 hidden from every later position, the answer must not depend on them")
        assert not t.allclose(plain(base), plain(other), atol=1e-2), "test setup: unblocked, the content of positions 1-4 should matter"
        assert not t.allclose(a, plain(base), atol=1e-2), "Blocking should change the answer logits"
        # Only the requested layers are blocked
        only_late = block_attention_logits(model, base, [n_layers - 1], block, answer_pos)
        assert not t.allclose(only_late, a, atol=1e-2), "Blocking only the last layer should differ from blocking every layer"
        assert block_attention_logits(model, base, [], block, answer_pos).sub(plain(base)).abs().max() < 1e-3, "layers=[] should block nothing"
    assert all(len(hook.fwd_hooks) == 0 for hook in model.hook_dict.values()), "Hooks were left attached to the model (use run_with_hooks)"
    print("All tests in `test_block_attention_logits` passed!")


# ---------------------------------------------------------------------------------------------
# 5. A real VLM
# ---------------------------------------------------------------------------------------------


def _qwen_inputs(qwen, processor, seed: int):
    scene = utils.sample_scene(np.random.default_rng(seed))
    question = "What colour is the circle? Answer with one word."
    return utils.qwen_inputs(processor, utils.render_scene(scene), question, qwen.device)


def test_qwen_forced_choice(qwen_forced_choice: Callable, qwen, processor):
    inputs = _qwen_inputs(qwen, processor, 0)
    candidates = t.tensor([processor.tokenizer.encode(c)[0] for c in utils.COLOURS], device=qwen.device)
    out = qwen_forced_choice(qwen, inputs, candidates)
    assert out.shape == (6,), f"Expected one logit per candidate, shape (6,), got {tuple(out.shape)}"
    with t.no_grad():
        expected = qwen(**inputs).logits[0, -1][candidates]
    t.testing.assert_close(out.float(), expected.float(), atol=0.1, rtol=0.01, msg="Should equal the last-position logits of the candidates")
    print("All tests in `test_qwen_forced_choice` passed!")


def test_qwen_patched_logits(qwen_patched_logits: Callable, qwen, processor):
    inputs_a, inputs_b = _qwen_inputs(qwen, processor, 0), _qwen_inputs(qwen, processor, 3)
    seq = inputs_a["input_ids"].shape[1]
    assert inputs_b["input_ids"].shape[1] == seq, "test setup: both prompts should have the same length"
    cache_a, cache_b = utils.qwen_cache_layer_inputs(qwen, inputs_a), utils.qwen_cache_layer_inputs(qwen, inputs_b)
    with t.no_grad():
        base_a, base_b = qwen(**inputs_a).logits[0, -1].float(), qwen(**inputs_b).logits[0, -1].float()
    everything = t.ones(seq, dtype=t.bool, device=qwen.device)
    nothing = t.zeros(seq, dtype=t.bool, device=qwen.device)
    out = qwen_patched_logits(qwen, inputs_a, 8, nothing, cache_b[8]).float()
    assert out.shape == base_a.shape, f"Expected logits of shape {tuple(base_a.shape)}, got {tuple(out.shape)}"
    t.testing.assert_close(out, base_a, atol=0.1, rtol=0.01, msg="Patching no positions should reproduce the unpatched run")
    out = qwen_patched_logits(qwen, inputs_a, 8, everything, cache_a[8]).float()
    t.testing.assert_close(out, base_a, atol=0.1, rtol=0.01, msg="Patching every position with the run's own residual stream should change nothing")
    for layer in [3, 10]:
        out = qwen_patched_logits(qwen, inputs_a, layer, everything, cache_b[layer]).float()
        t.testing.assert_close(
            out, base_b, atol=0.1, rtol=0.01,
            msg=f"Patching every position at layer {layer} with run B's residual stream should reproduce run B's logits",
        )
    with t.no_grad():
        after = qwen(**inputs_a).logits[0, -1].float()
    t.testing.assert_close(after, base_a, atol=0.1, rtol=0.01, msg="The hook was not removed: later runs are still patched")
    print("All tests in `test_qwen_patched_logits` passed!")


# Additive 2026-10-16 controls: these fixtures never load a pretrained model.
def test_target_blind_oracle(target_blind_oracle: Callable):
    # Deliberately repeated colour and shape labels: uniqueness is not the oracle.
    scene = utils.Scene([
        utils.SceneObject("red", "circle", (0, 0)),
        utils.SceneObject("red", "circle", (0, 1)),
        utils.SceneObject("blue", "square", (1, 1)),
    ])
    expected = {"colour": 2 / 3, "shape": 2 / 3, "count": 1.0,
                "horizontal": 2 / 3, "vertical": 2 / 3}
    for kind, probability in expected.items():
        assert abs(target_blind_oracle(scene, kind) - probability) < 1e-12, kind
    singleton = utils.Scene([utils.SceneObject("green", "triangle", (1, 0))])
    assert all(target_blind_oracle(singleton, kind) == 1 for kind in expected)
    try:
        target_blind_oracle(utils.Scene([]), "colour")
    except ValueError:
        pass
    else:
        raise AssertionError("An empty scene has no uniform target-sampling law")
    print("All tests in `test_target_blind_oracle` passed!")


def test_qwen_layer_schedule(qwen_layer_schedule: Callable):
    assert qwen_layer_schedule(28) == [0, 3, 6, 9, 12, 15, 18, 21, 24, 27]
    assert qwen_layer_schedule(29) == [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 28]
    assert qwen_layer_schedule(1) == [0]
    for n in range(1, 40):
        layers = qwen_layer_schedule(n)
        assert len(layers) == len(set(layers)) and layers[-1] == n - 1
        assert all(0 <= layer < n for layer in layers)
    try:
        qwen_layer_schedule(0)
    except ValueError:
        pass
    else:
        raise AssertionError("Zero decoder layers must reject")
    print("All tests in `test_qwen_layer_schedule` passed!")


def test_evaluate_readout_controls(evaluate_readout_controls: Callable):
    """Exact fake logits test the data route, candidate scoring and short batch."""
    import inspect
    import types
    from types import SimpleNamespace

    original = inspect.unwrap(evaluate_readout_controls)
    g = original.__globals__
    scenes = [
        utils.Scene([utils.SceneObject("red", "circle", (0, 0))]),
        utils.Scene([utils.SceneObject("blue", "circle", (0, 1)), utils.SceneObject("red", "square", (1, 1))]),
        utils.Scene([utils.SceneObject("green", "triangle", (0, 0)), utils.SceneObject("red", "circle", (0, 1)),
                     utils.SceneObject("blue", "square", (1, 0))]),
    ]
    kinds = ["colour", "shape", "count", "horizontal", "vertical"]
    targets = {kind: [0, 1, 2] for kind in kinds}
    qa = {kind: [g["question_and_answer"](scene, kind, target)
                 for scene, target in zip(scenes, targets[kind])] for kind in kinds}
    questions = {kind: ([p for p, _ in rows], [a for _, a in rows]) for kind, rows in qa.items()}
    words = [" " + word for kind in kinds for word in g["ANSWER_WORDS"][kind]]
    ids = {word: i + 1 for i, word in enumerate(words)}
    # The fixture maps full prompts to answer IDs by scene row. The no-question
    # route never receives that map's prompts: every token is literal cue ID 0.
    prompt_ids = {}
    for kind, rows in qa.items():
        for row, (prompt, answer) in enumerate(rows):
            prompt_ids[(row, prompt)] = ids[answer]
    token_calls, forward_calls = [], []

    def tokenize(model, prompts):
        token_calls.append(list(prompts))
        markers = [0 if prompt == " Answer:" else prompt_ids[(i, prompt)] for i, prompt in enumerate(prompts)]
        return t.tensor(markers)[:, None], t.ones(len(prompts), dtype=t.long)

    def forward(model, projector, features, tokens, lens):
        assert features is not None, "An image-only input must retain the real feature rows"
        forward_calls.append((features[:, 0].tolist(), tokens[:, 0].tolist()))
        logits = t.zeros(len(features), len(ids) + 1)
        for row, marker in enumerate(tokens[:, 0].tolist()):
            logits[row, marker or ids[" red"]] = 4.0
        return logits

    fake = types.FunctionType(original.__code__, dict(g, utils=SimpleNamespace(tokenize_prompts=tokenize),
        vlm_answer_logits=forward), argdefs=original.__defaults__)
    model = SimpleNamespace(cfg=SimpleNamespace(device="cpu"), to_single_token=lambda word: ids[word])
    features = t.arange(3)[:, None]
    results, traces = fake(model, object(), features, scenes, questions, targets, batch_size=2)
    assert results["full question"] == {kind: 1.0 for kind in kinds}
    assert len(traces) == 2 * 5 * 3
    image_rows = [row for row in traces if row["condition"] == "image-only (fixed readout)"]
    assert all(row["model_prompt"] == " Answer:" for row in image_rows)
    for kind in kinds:
        selected = [row for row in image_rows if row["kind"] == kind]
        expected_word = " red" if kind == "colour" else " " + g["ANSWER_WORDS"][kind][0]
        expected = sum(answer == expected_word for answer in questions[kind][1]) / 3
        assert results["image-only (fixed readout)"][kind] == expected
        for row in selected:
            i = row["scene_index"]
            assert row["target_index"] == targets[kind][i] and row["answer"] == questions[kind][1][i]
            assert row["original_question"] == questions[kind][0][i]
            assert row["object_count"] == len(scenes[i].objects)
            assert row["prediction"] == expected_word
            assert row["candidate_ids"] == [ids[" " + word] for word in g["ANSWER_WORDS"][kind]]
    assert sum(call == [" Answer:"] * 3 for call in token_calls) == 1
    assert all(rows in ([0, 1], [2]) for rows, _ in forward_calls)
    assert len(forward_calls) == 12  # shared image-only: 2 batches; full questions: five tasks x 2
    print("All tests in `test_evaluate_readout_controls` passed!")
