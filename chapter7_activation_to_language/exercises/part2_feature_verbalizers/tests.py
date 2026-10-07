from typing import Callable

import numpy as np
import torch as t
from transformer_lens import HookedTransformer


def test_sae_latent_acts(sae_latent_acts: Callable, sae):
    t.manual_seed(0)
    resid = 30 * t.randn(2, 5, sae.cfg.d_in, device=sae.W_enc.device)
    latents = [3, 17, 2048, 24000]
    result = sae_latent_acts(resid, sae, latents)
    assert result.shape == (2, 5, len(latents)), f"Expected shape (2, 5, {len(latents)}), got {tuple(result.shape)}"
    # Independent oracle: SAELens' own encoder, restricted to the chosen latents
    expected = sae.encode(resid)[..., latents]
    t.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)
    assert (result >= 0).all(), "SAE activations should be non-negative (did you forget the ReLU?)"
    print("All tests in `test_sae_latent_acts` passed!")


def test_collect_activations(collect_activations: Callable, model: HookedTransformer):
    tokens = model.to_tokens(["The cat sat on the mat and looked", "One, two, three, four, five, six"])
    tokens = tokens[:, :8]
    hook = "blocks.2.hook_resid_pre"
    result = collect_activations(model, tokens, hook, lambda x: x[..., :3], batch_size=1)
    assert result.shape == (2, 8, 3), f"Expected shape (n_seq, seq, n_units) = (2, 8, 3), got {tuple(result.shape)}"
    assert result.device.type == "cpu", "Return the activations on the CPU (we keep big activation tensors there)"
    assert (result[:, 0] == 0).all(), "The BOS position (position 0) should be set to zero"
    _, cache = model.run_with_cache(tokens)
    t.testing.assert_close(result[:, 1:], cache[hook][:, 1:, :3].float().cpu(), atol=1e-4, rtol=1e-4)
    print("All tests in `test_collect_activations` passed!")


def test_top_activating_examples(top_activating_examples: Callable):
    acts = t.zeros(5, 6)
    acts[0, 2] = 1.0
    acts[1, 4] = 3.0
    acts[1, 1] = 2.5  # second-largest in the same sequence: should NOT give a second example
    acts[3, 5] = 2.0
    # sequences 2 and 4 never fire
    result = top_activating_examples(acts, k=3)
    assert result == [(1, 4), (3, 5), (0, 2)], f"Expected [(1, 4), (3, 5), (0, 2)], got {result}"
    result = top_activating_examples(acts, k=10)
    assert result == [(1, 4), (3, 5), (0, 2)], (
        f"Sequences that never fire (max activation 0) shouldn't be returned, even if k is large. Got {result}"
    )
    assert all(isinstance(s, int) and isinstance(p, int) for s, p in result), "Return python ints, not tensors"
    print("All tests in `test_top_activating_examples` passed!")


def test_random_examples(random_examples: Callable, window_before: int, window_after: int):
    acts = t.zeros(50, 64)
    a = random_examples(acts, 200, t.Generator().manual_seed(0))
    b = random_examples(acts, 200, t.Generator().manual_seed(0))
    assert a == b, "random_examples should be reproducible given the same generator"
    assert len(a) == 200
    seqs = [s for s, _ in a]
    positions = [p for _, p in a]
    assert min(seqs) >= 0 and max(seqs) < 50
    assert min(positions) >= window_before + 1, f"Positions should be at least {window_before + 1} (full left context)"
    assert max(positions) <= 64 - window_after - 1, f"Positions should be at most {64 - window_after - 1}"
    assert len(set(positions)) > 20, "Positions should be spread out at random"
    print("All tests in `test_random_examples` passed!")


def test_quantile_examples(quantile_examples: Callable):
    acts = t.zeros(100, 10)
    for s in range(80):  # 80 activating sequences with max activation s + 1 at position s % 10
        acts[s, s % 10] = float(s + 1)
    bins = quantile_examples(acts, n_bins=4, k_per_bin=3, generator=t.Generator().manual_seed(0))
    assert len(bins) == 4, f"Expected 4 bins, got {len(bins)}"
    assert all(len(b) == 3 for b in bins), f"Expected 3 examples per bin, got {[len(b) for b in bins]}"
    expected_ranges = [range(60, 80), range(40, 60), range(20, 40), range(0, 20)]
    for b, (examples, rng) in enumerate(zip(bins, expected_ranges)):
        for seq, pos in examples:
            assert seq in rng, f"Bin {b} should contain sequences ranked {b * 20}-{b * 20 + 19} by max activation, got {seq}"
            assert pos == seq % 10, "Each example should be centred on its sequence's max-activating position"
        assert len({s for s, _ in examples}) == 3, "Examples within a bin should be distinct sequences"
    print("All tests in `test_quantile_examples` passed!")


def test_format_example(format_example: Callable):
    toks = ["The", " capital", " of", " France", " is", "\n", " Paris"]
    acts = t.tensor([0.0, 0.5, 2.0, 3.0, 0.0, 0.0, 1.5])
    result = format_example(toks, acts, threshold=1.0)
    expected = "The capital<< of France>> is↵<< Paris>>"
    assert result == expected, f"Expected {expected!r}, got {result!r}"
    assert format_example(toks, acts, threshold=10.0) == "The capital of France is↵ Paris"
    print("All tests in `test_format_example` passed!")


def test_build_explainer_messages(build_explainer_messages: Callable):
    examples = ["I went to the<< store>>", "the<< shop>> was shut"]
    messages = build_explainer_messages(examples, ["store", "shop"])
    assert isinstance(messages, list) and all(isinstance(m, dict) for m in messages)
    roles = [m["role"] for m in messages]
    assert roles == ["system", "user", "assistant", "user"], (
        f"Expected roles [system, user, assistant, user] (system prompt, one few-shot example, then the real query), got {roles}"
    )
    last = messages[-1]["content"]
    assert "1. I went to the<< store>>" in last and "2. the<< shop>> was shut" in last, (
        "The final user message should list the examples, numbered from 1"
    )
    assert "Marked tokens: store, shop" in last, "The final user message should end with the list of marked tokens"
    assert messages[1]["content"].startswith("Excerpts:") and last.startswith("Excerpts:"), (
        "The few-shot user message and the real user message should have the same format"
    )
    print("All tests in `test_build_explainer_messages` passed!")


def test_shuffle_activations(shuffle_activations: Callable):
    t.manual_seed(0)
    acts = t.relu(t.randn(20, 8, 3))
    acts[:, 0] = 0.0
    out = shuffle_activations(acts, t.Generator().manual_seed(0))
    assert out.shape == acts.shape
    assert (out[:, 0] == 0).all(), "BOS positions should stay zero"
    for j in range(3):
        assert t.equal(out[:, 1:, j].flatten().sort().values, acts[:, 1:, j].flatten().sort().values), (
            f"Unit {j}: the shuffled activations should be a permutation of the original (non-BOS) activations"
        )
    assert not t.equal(out, acts), "The activations should actually be shuffled"
    # different units should get different permutations
    acts2 = t.arange(20 * 8, dtype=t.float).reshape(20, 8, 1).repeat(1, 1, 2)
    acts2[:, 0] = 0
    out2 = shuffle_activations(acts2, t.Generator().manual_seed(1))
    assert not t.equal(out2[..., 0], out2[..., 1]), "Each unit should be shuffled with its own random permutation"
    print("All tests in `test_shuffle_activations` passed!")


def test_build_simulation_input(build_simulation_input: Callable, tokenizer):
    str_toks = ["The", " cat", "\n", " sat"]
    ids, slots = build_simulation_input(tokenizer, "the word cat", str_toks)
    assert isinstance(ids, list) and isinstance(slots, list)
    assert len(slots) == len(str_toks), f"Expected one slot per token ({len(str_toks)}), got {len(slots)}"
    assert slots == sorted(slots) and slots[-1] < len(ids), "Slots should be increasing positions inside the sequence"
    for slot, tok in zip(slots, str_toks):
        assert tokenizer.decode(ids[slot]) == "\t", (
            f"The token at each slot should be the tab that follows the excerpt token (so the next token predicted "
            f"is the activation). Got {tokenizer.decode(ids[slot])!r}"
        )
        after = tokenizer.decode(ids[slot + 1 : slot + 3])
        assert after.startswith("unknown"), f"Each slot should be followed by the placeholder 'unknown', got {after!r}"
    text = tokenizer.decode(ids)
    assert "Explanation: the word cat" in text, "The prompt should contain the explanation being scored"
    assert "The\tunknown\n cat\tunknown\n↵\tunknown\n sat\tunknown\n" in text, (
        "The end of the prompt should list each token (newlines replaced by ↵), a tab, and 'unknown'"
    )
    assert "store\t9" in text, "The prompt should include the few-shot example, with its true activations"
    print("All tests in `test_build_simulation_input` passed!")


def test_simulate_activations(simulate_activations: Callable, build_simulation_input: Callable, tokenizer, llm):
    requests = [("the word cat", ["The", " cat", " sat"]), ("names of colours", ["A", " red", " car", " and", " a", " blue", " one"])]
    preds = simulate_activations(tokenizer, llm, requests)
    assert len(preds) == 2 and preds[0].shape == (3,) and preds[1].shape == (7,)
    digit_ids = [tokenizer.convert_tokens_to_ids(str(d)) for d in range(10)]
    # Independent oracle: one unbatched forward pass through the full model, with the full logit vector
    for (explanation, toks), pred in zip(requests, preds):
        ids, slots = build_simulation_input(tokenizer, explanation, toks)
        with t.inference_mode():
            logits = llm(input_ids=t.tensor([ids], device=llm.device)).logits[0, slots][:, digit_ids].float().cpu()
        expected = logits.softmax(-1) @ t.arange(10.0)
        t.testing.assert_close(pred.float(), expected, atol=0.35, rtol=0)
    assert ((preds[1] >= 0) & (preds[1] <= 9)).all(), "Predictions are expected values on a 0-9 scale"
    print("All tests in `test_simulate_activations` passed!")


def test_correlation_score(correlation_score: Callable):
    rng = np.random.default_rng(0)
    for _ in range(5):
        a, b = rng.normal(size=50), rng.normal(size=50)
        expected = np.corrcoef(a, b)[0, 1]
        result = correlation_score(t.tensor(a), t.tensor(b))
        assert isinstance(result, float), "Return a python float"
        assert abs(result - expected) < 1e-5, f"Expected {expected:.5f}, got {result:.5f}"
    assert correlation_score(t.ones(10), t.arange(10.0)) == 0.0, "A constant prediction should get a score of 0, not nan"
    assert abs(correlation_score(t.arange(10.0), 3 * t.arange(10.0) + 1) - 1.0) < 1e-6
    print("All tests in `test_correlation_score` passed!")


def test_permutation_null(permutation_null: Callable, correlation_score: Callable):
    t.manual_seed(0)
    true = t.relu(t.randn(200)) ** 3
    pred = true + 0.1 * t.randn(200)
    null = permutation_null(pred, true, 500, t.Generator().manual_seed(0))
    assert null.shape == (500,), f"Expected one correlation per permutation, got shape {tuple(null.shape)}"
    assert t.equal(null, permutation_null(pred, true, 500, t.Generator().manual_seed(0))), "Should be reproducible"
    assert abs(null.mean().item()) < 0.03, f"The null distribution should be centred near 0, got mean {null.mean():.3f}"
    assert null.max() < correlation_score(pred, true), "Shuffling should destroy the (near-perfect) correlation"
    # Each null value must be the correlation with *some* permutation of `true`, so it can't exceed the max possible
    sorted_corr = correlation_score(pred.sort().values, true.sort().values)
    assert (null <= sorted_corr + 1e-5).all()
    print("All tests in `test_permutation_null` passed!")


def test_auroc(auroc: Callable):
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(0)
    for _ in range(5):
        pos, neg = rng.normal(1, 1, size=15), rng.normal(0, 1, size=20)
        pos[:3] = neg[:3]  # include some ties
        expected = roc_auc_score([1] * 15 + [0] * 20, np.concatenate([pos, neg]))
        result = auroc(t.tensor(pos), t.tensor(neg))
        assert isinstance(result, float), "Return a python float"
        assert abs(result - expected) < 1e-6, f"Expected {expected:.4f} (sklearn), got {result:.4f}"
    assert auroc(t.ones(3), t.ones(4)) == 0.5, "All-tied scores should give AUROC 0.5"
    print("All tests in `test_auroc` passed!")


def test_detection_scores(detection_scores: Callable, tokenizer, llm):
    requests = [
        ("the names of animals", "The dog chased the cat across the garden."),
        ("the names of animals", "Interest rates rose by half a percentage point on Tuesday."),
    ]
    scores = detection_scores(tokenizer, llm, requests)
    assert scores.shape == (2,), f"Expected one score per request, got shape {tuple(scores.shape)}"
    assert scores[0] > scores[1], "The text about animals should get a higher Yes-vs-No score than the one about interest rates"
    print("All tests in `test_detection_scores` passed!")


def test_steered_generations(steered_generations: Callable, model: HookedTransformer, hook_name: str):
    prompts = model.to_tokens(["I think that the", "The weather today is"])
    zero = t.zeros(model.cfg.d_model, device=prompts.device)
    steered = steered_generations(model, zero, prompts, max_new_tokens=8)
    assert len(steered) == 2 and all(isinstance(s, str) for s in steered)
    plain = model.generate(prompts, max_new_tokens=8, do_sample=False, verbose=False, stop_at_eos=False)
    expected = [model.to_string(row[prompts.shape[1] :]) for row in plain]
    assert steered == expected, "Steering with a zero vector should give the same (greedy) text as no steering"
    big = t.randn(model.cfg.d_model, generator=t.Generator().manual_seed(0)).to(prompts.device)
    big = big / big.norm() * 200
    out = steered_generations(model, big, prompts, max_new_tokens=8)
    assert out != expected, "Steering with a large vector should change the generated text"
    assert len(model.hook_dict[hook_name].fwd_hooks) == 0, "The steering hook should be removed after generation"
    print("All tests in `test_steered_generations` passed!")


def _toy_tokens_and_acts(model: HookedTransformer):
    tokens = model.to_tokens([
        "The cat sat on the mat and then the dog came in and sat down too.",
        "In 1990 the company moved to Cambridge, where it grew quickly over ten years.",
        "Rain fell all night, and in the morning the river had burst its banks again.",
    ])[:, :14]
    acts = t.zeros(3, 14)
    acts[0, 3], acts[1, 6] = 5.0, 2.0
    return tokens, acts


def test_simulation_scores(simulation_scores: Callable, get_window: Callable, correlation_score: Callable, model, tokenizer, llm):
    tokens, acts = _toy_tokens_and_acts(model)
    examples = [[(0, 8), (1, 10)], [(1, 10)]]
    scores, preds, trues = simulation_scores(model, tokens, [acts, acts], ["the word sat", "the word sat"], examples, tokenizer, llm)
    assert scores.shape == (2,), f"Expected one score per pair, got shape {tuple(scores.shape)}"
    assert len(preds) == len(trues) == 2
    w0, w1 = get_window(model, tokens, acts, 0, 8), get_window(model, tokens, acts, 1, 10)
    assert t.equal(trues[0], t.cat([w0[1], w1[1]])), "trues[k] should be the concatenation of pair k's windows' activations, in order"
    assert preds[0].shape == trues[0].shape, "There should be one prediction per token"
    assert abs(scores[0].item() - correlation_score(preds[0], trues[0])) < 1e-5, "scores[k] should be correlation_score(preds[k], trues[k])"
    t.testing.assert_close(preds[1], preds[0][len(w0[0]) :], atol=0.35, rtol=0,
                           msg="The same (explanation, window) should get (almost) the same predictions in any pair")
    print("All tests in `test_simulation_scores` passed!")


def test_detection_auroc(detection_auroc: Callable, detection_scores: Callable, auroc: Callable, window_bounds: Callable, model, tokenizer, llm):
    tokens, _ = _toy_tokens_and_acts(model)
    example_sets = [([[(0, 8)], [(1, 10)]], [(2, 9), (2, 12)])]
    result = detection_auroc(model, tokens, ["animals"], example_sets, tokenizer, llm)
    assert result.shape == (1, 2), f"Expected shape (n_pairs, n_bins) = (1, 2), got {tuple(result.shape)}"
    texts = [model.to_string(tokens[s, slice(*window_bounds(p, tokens.shape[1]))]) for s, p in [(0, 8), (1, 10), (2, 9), (2, 12)]]
    scores = detection_scores(tokenizer, llm, [("animals", text) for text in texts])
    expected = [auroc(scores[[0]], scores[2:]), auroc(scores[[1]], scores[2:])]
    t.testing.assert_close(result[0].float(), t.tensor(expected).float(), atol=0.26, rtol=0)
    print("All tests in `test_detection_auroc` passed!")


def test_max_activations_on_texts(max_activations_on_texts: Callable, model: HookedTransformer):
    texts = ["Hello there", "A slightly longer piece of text, with a comma."]
    hook = "blocks.1.hook_resid_pre"
    result = max_activations_on_texts(model, texts, hook, lambda x: x[..., :3])
    assert result.shape == (2, 3), f"Expected shape (n_texts, n_units) = (2, 3), got {tuple(result.shape)}"
    for i, text in enumerate(texts):
        _, cache = model.run_with_cache(model.to_tokens(text))
        acts = cache[hook][0, :, :3].float().cpu()
        acts[0] = 0.0  # BOS is zeroed by collect_activations
        t.testing.assert_close(result[i].float().cpu(), acts.max(dim=0).values, atol=1e-4, rtol=1e-4)
    print("All tests in `test_max_activations_on_texts` passed!")


def test_intervention_curve(intervention_curve: Callable, model: HookedTransformer, tokenizer, llm):
    prompts = model.to_tokens(["I think that the", "The weather today is"])
    zero = t.zeros(model.cfg.d_model, device=prompts.device)
    scores, texts = intervention_curve(model, tokenizer, llm, zero, ["the weather", "cooking"], [0.0, 3.0], prompts)
    assert scores.shape == (2, 2), f"Expected shape (n_explanations, n_coefs) = (2, 2), got {tuple(scores.shape)}"
    assert len(texts) == 2 and all(len(x) == 2 for x in texts), "Return one list of continuations (one per prompt) per coefficient"
    assert texts[0] == texts[1], "Any multiple of a zero direction is still no steering, so the text shouldn't change"
    t.testing.assert_close(scores[:, 0], scores[:, 1], atol=1e-3, rtol=0)
    print("All tests in `test_intervention_curve` passed!")
