import math
import random
from types import SimpleNamespace
from typing import Callable

import torch as t
import torch.nn.functional as F
from transformer_lens import HookedTransformer


def _reference_hook_name(model: HookedTransformer, layer: int) -> str:
    n = model.cfg.n_layers
    return f"blocks.{layer}.hook_resid_pre" if layer < n else f"blocks.{n - 1}.hook_resid_post"


def test_annotate_example(annotate_example: Callable, SourceExample: type, model: HookedTransformer):
    cases = [
        # text, animal, city, sentiment, number, animal_word, verb, expected availability (worked out by hand)
        ("The dogs in Paris loved the rain.", "dog", "Paris", "positive", "plural", "dogs", "loved",
         {"animal": 2, "city": 4, "sentiment": 5, "number": 2}),
        ("In Tokyo, my elephant dreaded the garden.", "elephant", "Tokyo", "negative", "singular", "elephant", "dreaded",
         {"animal": 5, "city": 2, "sentiment": 6, "number": 5}),
        ("Two mice liked the snow in Oslo.", "mouse", "Oslo", "positive", "plural", "mice", "liked",
         {"animal": 2, "city": 7, "sentiment": 3, "number": 2}),
    ]
    for text, animal, city, sentiment, number, word, verb, expected in cases:
        ex = SourceExample(text=text, animal=animal, city=city, sentiment=sentiment, number=number,
                           animal_word=word, verb=verb, template="")
        out = annotate_example(model, ex)
        assert out is ex, "annotate_example should fill in and return the same example object"
        assert isinstance(ex.tokens, list), f"example.tokens should be a list of ints, got {type(ex.tokens)}"
        assert ex.tokens[0] == model.tokenizer.bos_token_id, "example.tokens should start with the BOS token"
        assert ex.tokens == model.to_tokens(text)[0].tolist(), "example.tokens should be the BOS-prefixed tokens of the text"
        assert ex.available == expected, f"For {text!r}, expected available={expected}, got {ex.available}"
    print("All tests in `test_annotate_example` passed!")


def test_get_resid(get_resid: Callable, model: HookedTransformer):
    sentences = ["The dogs in Paris loved the rain.", "Last year in Rome, two cows hated the long noisy storm.",
                 "My cat slept.", "Everyone in Oslo knew that the frogs feared the river."]
    examples = [SimpleNamespace(tokens=model.to_tokens(s)[0].tolist()) for s in sentences]
    for layer in [0, 6, model.cfg.n_layers]:
        out = get_resid(model, examples, layer, batch_size=2)
        assert isinstance(out, list) and len(out) == len(examples), "get_resid should return one tensor per example"
        name = _reference_hook_name(model, layer)
        for ex, got in zip(examples, out):
            assert got.shape == (len(ex.tokens), model.cfg.d_model), (
                f"Expected shape (len(tokens), d_model) = {(len(ex.tokens), model.cfg.d_model)}, got {tuple(got.shape)}"
            )
            # Independent oracle: run each example on its own, with no padding at all
            _, cache = model.run_with_cache(t.tensor([ex.tokens], device=model.cfg.device), names_filter=name)
            t.testing.assert_close(got, cache[name][0], atol=1e-3, rtol=1e-3,
                                   msg=f"Activations at layer {layer} don't match an unpadded single-example run")
    print("All tests in `test_get_resid` passed!")


def test_train_linear_probe(train_linear_probe: Callable):
    g = t.Generator().manual_seed(0)
    n, d = 400, 20

    def blobs(n):
        y = t.randint(0, 2, (n,), generator=g)
        x = t.randn(n, d, generator=g)
        x[:, 0] += 3.0 * (2 * y - 1)
        x[:, 1] *= 1000.0  # a huge-variance nuisance dimension: standardisation should stop it dominating
        return x, y

    x_train, y_train = blobs(n)
    x_test, y_test = blobs(1000)
    t.manual_seed(0)
    probe = train_linear_probe(x_train, y_train, 2)
    for buffer in ["mean", "std"]:
        assert hasattr(probe, buffer), f"The returned probe should be a LinearProbe with a `{buffer}` buffer"
    t.testing.assert_close(probe.mean, x_train.mean(0), msg="probe.mean should be the per-dimension training mean")
    acc = probe.accuracy(x_test, y_test)
    assert acc > 0.95, f"A probe on well-separated blobs should get >95% held-out accuracy, got {acc:.1%}"
    # Random labels: the probe may fit the training set, but shouldn't generalise
    x_small, _ = blobs(100)
    y_random = t.randint(0, 2, (100,), generator=g)
    probe = train_linear_probe(x_small, y_random, 2)
    acc = probe.accuracy(x_test, t.randint(0, 2, (1000,), generator=g))
    assert acc < 0.6, f"A probe trained on random labels should be at chance on fresh random labels, got {acc:.1%}"
    print("All tests in `test_train_linear_probe` passed!")


def test_tokenize_oracle_prompts(tokenize_oracle_prompts: Callable, model: HookedTransformer, oracle_prompt: Callable, placeholder: str):
    questions = ["Name the animal.", "Is there one animal (singular) or several (plural)?", "Positive or negative?"]
    tokens, placeholder_pos, answer_pos = tokenize_oracle_prompts(model, questions)
    rows = [model.to_tokens(oracle_prompt(q))[0] for q in questions]
    max_len = max(len(r) for r in rows)
    assert tokens.shape == (3, max_len), f"Expected tokens of shape (3, {max_len}), got {tuple(tokens.shape)}"
    assert placeholder_pos.shape == (3,) and answer_pos.shape == (3,), "positions should have shape (batch,)"
    placeholder_id = model.to_single_token(placeholder)
    colon_id = model.to_single_token(":")
    for i, row in enumerate(rows):
        assert tokens[i, : len(row)].tolist() == row.tolist(), f"Row {i} should start with the BOS-prefixed prompt tokens"
        assert tokens[i, 0].item() == model.tokenizer.bos_token_id, "Prompts should start with BOS"
        assert answer_pos[i].item() == len(row) - 1, f"answer_pos[{i}] should be {len(row) - 1} (the last real token)"
        assert tokens[i, answer_pos[i]].item() == colon_id, "The last real token should be the ':' of 'Answer:'"
        assert tokens[i, placeholder_pos[i]].item() == placeholder_id, "placeholder_pos should point at the ' ?' token"
        assert placeholder_pos[i].item() == 1, "With BOS first, the placeholder should be at position 1"
    assert tokens.device.type == t.device(model.cfg.device).type, "tokens should be on the model's device"
    print("All tests in `test_tokenize_oracle_prompts` passed!")


def _reference_injection(resid, vectors, placeholder_pos, center, coef):
    out = resid.clone()
    for b in range(resid.shape[0]):
        p = placeholder_pos[b]
        v = vectors[b] - (center if center is not None else 0)
        out[b, p] = resid[b, p] + coef * resid[b, p].norm() * v / v.norm()
    return out


def test_make_injection_hook(make_injection_hook: Callable):
    g = t.Generator().manual_seed(0)
    resid = t.randn(3, 5, 8, generator=g)
    vectors = t.randn(3, 8, generator=g) * 50
    center = t.randn(8, generator=g)
    pos = t.tensor([1, 2, 0])
    for c, coef in [(center, 1.0), (None, 1.0), (center, 2.5)]:
        original = resid.clone()
        out = make_injection_hook(vectors, pos, c, coef)(resid, None)
        t.testing.assert_close(resid, original, msg="The hook must not modify `resid` in place")
        expected = _reference_injection(resid, vectors, pos, c, coef)
        t.testing.assert_close(out, expected, msg=f"Wrong injected values (center={'set' if c is not None else 'None'}, coef={coef})")
    # Only the placeholder positions change
    out = make_injection_hook(vectors, pos, center)(resid, None)
    changed = (out != resid).any(-1)
    expected_changed = t.zeros(3, 5, dtype=t.bool)
    expected_changed[t.arange(3), pos] = True
    assert t.equal(changed, expected_changed), "Only the placeholder position of each batch element should change"
    # The text-only control: no vectors means no change
    out = make_injection_hook(None, pos, center)(resid, None)
    t.testing.assert_close(out, resid, msg="With vectors=None the hook should return resid unchanged")
    # It must be usable inside a training graph (no in-place modification of a tensor autograd needs)
    x = t.randn(3, 5, 8, requires_grad=True)
    hidden = x * 2.0
    out = make_injection_hook(vectors, pos, center)(hidden, None)
    (out**2).sum().backward()
    assert x.grad is not None and t.isfinite(x.grad).all(), "Gradients should flow through the hook"
    print("All tests in `test_make_injection_hook` passed!")


def test_oracle_logits(oracle_logits: Callable, tokenize_oracle_prompts: Callable, make_injection_hook: Callable, model: HookedTransformer):
    questions = ["Name the city.", "Is the subject singular or plural?"]
    tokens, placeholder_pos, answer_pos = tokenize_oracle_prompts(model, questions)
    rows = t.arange(len(questions), device=tokens.device)
    with t.no_grad():
        clean = model(tokens)[rows, answer_pos]
        out = oracle_logits(model, None, questions, None)
        assert out.shape == (2, model.cfg.d_vocab), f"Expected shape (batch, d_vocab), got {tuple(out.shape)}"
        t.testing.assert_close(out, clean, atol=1e-3, rtol=1e-3, msg="With vectors=None, the oracle should just run the prompt")
        g = t.Generator().manual_seed(0)
        vectors = t.randn(2, model.cfg.d_model, generator=g).to(model.cfg.device) * 50
        center = t.randn(model.cfg.d_model, generator=g).to(model.cfg.device)

        def reference_hook(resid, hook):
            return _reference_injection(resid, vectors, placeholder_pos, center, 1.0)

        expected = model.run_with_hooks(tokens, fwd_hooks=[("blocks.1.hook_resid_pre", reference_hook)])[rows, answer_pos]
        out = oracle_logits(model, vectors, questions, center)
        t.testing.assert_close(out, expected, atol=1e-3, rtol=1e-3, msg="Injected logits don't match a reference injection at blocks.1.hook_resid_pre")
        assert not t.allclose(out, clean, atol=1e-2), "Injecting a vector should change the logits"
        t.testing.assert_close(model(tokens)[rows, answer_pos], clean, atol=1e-3, rtol=1e-3, msg="oracle_logits left a hook attached")
    print("All tests in `test_oracle_logits` passed!")


def test_sample_oracle_batch(sample_oracle_batch: Callable, examples: list, acts: list, questions: dict):
    n = 400
    for position in ["last", "random"]:
        vectors, qs, answers, families = sample_oracle_batch(examples, acts, n, random.Random(0), questions, position)
        assert vectors.shape == (n, acts[0].shape[-1]), f"vectors should have shape (batch, d_model), got {tuple(vectors.shape)}"
        assert len(qs) == len(answers) == len(families) == n, "questions, answers and families should have batch_size entries"
        assert set(families) == set(questions), f"Every family should be sampled; got {set(families)}"
        positions_used = []
        flat = t.cat(list(acts))
        index = [(i, p) for i, a in enumerate(acts) for p in range(len(a))]
        hits = (vectors[:, None, :] == flat[None, :, :]).all(-1).cpu()  # (batch, total positions)
        for j in range(n):
            family = families[j]
            assert qs[j] in questions[family], f"{qs[j]!r} is not one of the paraphrases for {family!r}"
            # find which example and position this vector came from
            matches = [index[k] for k in hits[j].nonzero().flatten().tolist()]
            assert matches, "Every vector should be an activation taken from `acts`"
            ok = False
            for i, p in matches:
                last = len(examples[i].tokens) - 1
                valid_pos = p == last if position == "last" else examples[i].available[family] <= p <= last
                if valid_pos and getattr(examples[i], family) == answers[j]:
                    ok = True
                    positions_used.append((p, examples[i].available[family], last))
                    break
            assert ok, (
                f"Item {j} (family {family!r}, answer {answers[j]!r}) doesn't match its activation's example and a valid "
                f"position for position={position!r}"
            )
        if position == "random":
            assert any(p < last for p, _, last in positions_used), "position='random' should sometimes pick non-final positions"
            assert any(p == avail for p, avail, _ in positions_used), "position='random' should include the first available position"
    again = sample_oracle_batch(examples, acts, 20, random.Random(0), questions, "random")
    first = sample_oracle_batch(examples, acts, 20, random.Random(0), questions, "random")
    assert again[1:] == first[1:] and t.equal(again[0], first[0]), "The same rng seed should give the same batch"
    print("All tests in `test_sample_oracle_batch` passed!")


def test_text_only_loss_floor(text_only_loss_floor: Callable, SourceExample: type):
    rows = [("dog", "Paris", "positive", "singular"), ("dog", "Paris", "positive", "plural"),
            ("cat", "Paris", "positive", "singular"), ("cat", "Paris", "negative", "plural")]
    examples = [SourceExample(text="", animal=a, city=c, sentiment=s, number=n, animal_word=a, verb="", template="")
                for a, c, s, n in rows]
    h = lambda ps: -sum(p * math.log(p) for p in ps if p > 0)
    expected = (h([0.5, 0.5]) + h([1.0]) + h([0.75, 0.25]) + h([0.5, 0.5])) / 4
    result = text_only_loss_floor(examples)
    assert abs(result - expected) < 1e-5, f"Expected {expected:.5f} nats, got {result}"
    result = text_only_loss_floor(examples, ["sentiment"])
    assert abs(result - h([0.75, 0.25])) < 1e-5, "The `families` argument should restrict which families are averaged"
    print("All tests in `test_text_only_loss_floor` passed!")


def test_evaluate_oracle(evaluate_oracle: Callable, oracle_logits: Callable, answer_token_ids: Callable, oracle, examples, vectors, center):
    questions = {"animal": ["Which animal is this about?", "Name the animal."], "sentiment": ["Positive or negative?"]}
    with t.no_grad():
        for v in [vectors, None]:
            result = evaluate_oracle(oracle, v, examples, center, questions=questions, batch_size=7)
            assert set(result) == set(questions), "evaluate_oracle should return one accuracy per family in `questions`"
            for family, paraphrases in questions.items():
                answers = answer_token_ids(oracle, [getattr(ex, family) for ex in examples])
                correct = []
                for q in paraphrases:
                    logits = oracle_logits(oracle, v, [q] * len(examples), center)
                    correct.append(logits.argmax(-1) == answers)
                expected = t.cat(correct).float().mean().item()
                assert abs(result[family] - expected) < 1e-6, (
                    f"{family} ({'text-only' if v is None else 'with vectors'}): expected {expected:.4f}, got {result[family]:.4f}"
                )
    print("All tests in `test_evaluate_oracle` passed!")


def test_control_vectors(control_vectors: Callable):
    g = t.Generator().manual_seed(1)
    vectors = t.randn(10, 6, generator=g) + 5
    center = t.randn(6, generator=g) + 5
    assert t.equal(control_vectors(vectors, "real", center), vectors), "'real' should return the vectors unchanged"
    assert control_vectors(vectors, "text-only", center) is None, "'text-only' should return None"
    shuffled = control_vectors(vectors, "shuffled", center, seed=0)
    assert shuffled.shape == vectors.shape
    assert t.equal(shuffled.sort(0).values, vectors.sort(0).values) and all(
        any(t.equal(row, other) for other in vectors) for row in shuffled
    ), "'shuffled' should be a permutation of the rows"
    assert not (shuffled == vectors).all(-1).any(), "'shuffled' must be a derangement: no row may stay in place"
    assert t.equal(shuffled, control_vectors(vectors, "shuffled", center, seed=0)), "The same seed should give the same shuffle"
    for seed in range(20):  # derangement must hold for every seed
        s = control_vectors(vectors, "shuffled", center, seed=seed)
        assert not (s == vectors).all(-1).any(), f"seed {seed}: some row stayed in place"
    rand = control_vectors(vectors, "random", center, seed=0)
    assert rand.shape == vectors.shape
    t.testing.assert_close((rand - center).norm(dim=-1), (vectors - center).norm(dim=-1),
                           msg="'random' vectors should match the norm of (vectors - center), measured from the center")
    cos = F.cosine_similarity(rand - center, vectors - center, dim=-1).abs().mean()
    assert cos < 0.6, "'random' vectors should point in random directions (relative to the center)"
    assert t.equal(rand, control_vectors(vectors, "random", center, seed=0)), "The same seed should give the same random vectors"
    print("All tests in `test_control_vectors` passed!")


def test_mean_difference_direction(mean_difference_direction: Callable):
    vectors = t.tensor([[1.0, 0.0], [3.0, 2.0], [0.0, 0.0], [0.0, 4.0]])
    mask = t.tensor([True, True, False, False])
    t.testing.assert_close(mean_difference_direction(vectors, mask), t.tensor([2.0, -1.0]))
    print("All tests in `test_mean_difference_direction` passed!")


def test_steering_sweep(steering_sweep: Callable, oracle_logits: Callable, oracle, vectors, center):
    question = "Is the feeling positive or negative?"
    pos_id = oracle.to_single_token(" positive")
    g = t.Generator().manual_seed(0)
    direction = t.randn(vectors.shape[-1], generator=g).to(vectors.device) * 10
    alphas = [0.0, 2.0]
    result = steering_sweep(oracle, vectors, direction, alphas, question, "positive", center)
    assert len(result) == len(alphas), "steering_sweep should return one fraction per alpha"
    with t.no_grad():
        for alpha, got in zip(alphas, result):
            logits = oracle_logits(oracle, vectors + alpha * direction, [question] * len(vectors), center)
            expected = (logits.argmax(-1) == pos_id).float().mean().item()
            assert abs(got - expected) < 1e-6, f"alpha={alpha}: expected {expected:.3f}, got {got:.3f}"
    print("All tests in `test_steering_sweep` passed!")


def test_patched_final_vectors(patched_final_vectors: Callable, resid_hook_name: Callable, model: HookedTransformer):
    clean = model.to_tokens(["The dogs in Paris loved the rain.", "My cow in Rome hated the snow."])
    corrupt = model.to_tokens(["The dogs in Tokyo loved the rain.", "My cow in Oslo hated the snow."])
    city_pos, read_layer = 4, 6
    name = resid_hook_name(model, read_layer)
    with t.no_grad():
        _, clean_cache = model.run_with_cache(clean, names_filter=name)
        _, corrupt_cache = model.run_with_cache(corrupt, names_filter=name)
        # Patching at layer 0 is the same as changing the input token (embedding + position embedding)
        hybrid = clean.clone()
        hybrid[:, city_pos] = corrupt[:, city_pos]
        _, hybrid_cache = model.run_with_cache(hybrid, names_filter=name)
        out = patched_final_vectors(model, clean, corrupt, city_pos, 0, read_layer)
        assert out.shape == (2, model.cfg.d_model), f"Expected shape (batch, d_model), got {tuple(out.shape)}"
        t.testing.assert_close(out, hybrid_cache[name][:, -1], atol=1e-3, rtol=1e-3,
                               msg="Patching at layer 0 should equal running on the input with the city token swapped")
        # Patching a non-final position at the read layer itself can't affect the final position yet
        out = patched_final_vectors(model, clean, corrupt, city_pos, read_layer, read_layer)
        t.testing.assert_close(out, clean_cache[name][:, -1], atol=1e-3, rtol=1e-3,
                               msg="Patching the city at the read layer should leave the final position unchanged")
        # Patching the final position itself at the read layer should give the corrupted run's vector
        last = clean.shape[1] - 1
        out = patched_final_vectors(model, clean, corrupt, last, read_layer, read_layer)
        t.testing.assert_close(out, corrupt_cache[name][:, -1], atol=1e-3, rtol=1e-3,
                               msg="Patching the final position at the read layer should give the corrupted vector "
                                   "(did you add the patching hook before the reading hook?)")
        # Model is left clean
        _, again = model.run_with_cache(clean, names_filter=name)
        t.testing.assert_close(again[name], clean_cache[name], msg="patched_final_vectors left a hook attached")
    print("All tests in `test_patched_final_vectors` passed!")
