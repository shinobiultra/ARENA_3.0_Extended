from typing import Callable
from inspect import unwrap

import pandas as pd
import torch as t
import torch.nn.functional as F
from transformer_lens import HookedTransformer


def test_logit_lens(logit_lens: Callable, model: HookedTransformer):
    t.manual_seed(0)
    tokens = t.randint(0, model.cfg.d_vocab, (2, 6), device=model.cfg.device)
    _, cache = model.run_with_cache(tokens)
    result = logit_lens(model, cache, pos=-1)
    n_layers = model.cfg.n_layers
    assert result.shape == (n_layers + 1, 2, model.cfg.d_vocab), (
        f"Expected shape {(n_layers + 1, 2, model.cfg.d_vocab)}, got {tuple(result.shape)}"
    )
    # Independent oracle: TransformerLens' own accumulated residual stream, with ln_final applied freshly to each layer
    accumulated = cache.accumulated_resid(pos_slice=-1, apply_ln=False)
    expected = model.unembed(model.ln_final(accumulated))
    t.testing.assert_close(result, expected, atol=1e-3, rtol=1e-3)
    # The last row should be the model's actual output logits
    t.testing.assert_close(result[-1], model(tokens)[:, -1], atol=1e-3, rtol=1e-3)
    print("All tests in `test_logit_lens` passed!")


def test_top_tokens_table(top_tokens_table: Callable, model: HookedTransformer):
    vocab = model.cfg.d_vocab
    logits = t.full((2, vocab), -10.0)
    ids = model.to_tokens([" Paris", " London", " Rome"], prepend_bos=False)[:, 0]
    logits[0, ids[0]], logits[0, ids[1]] = 5.0, 4.0
    logits[1, ids[2]], logits[1, ids[0]] = 6.0, 1.0
    df = top_tokens_table(model, logits, k=2, row_labels=["a", "b"])
    assert isinstance(df, pd.DataFrame), "Should return a pandas DataFrame"
    assert df.shape == (2, 2), f"Expected a (2, 2) table (one row per layer, one column per rank), got {df.shape}"
    assert list(df.index) == ["a", "b"], "Rows should be labelled with `row_labels`"
    assert "' Paris'" in df.iloc[0, 0] and "' London'" in df.iloc[0, 1], f"Wrong top tokens in row 0: {list(df.iloc[0])}"
    assert "' Rome'" in df.iloc[1, 0], f"Wrong top token in row 1: {list(df.iloc[1])}"
    p = t.softmax(logits[0], -1)[ids[0]].item()
    assert f"{p:.2f}" in df.iloc[0, 0], f"Expected probability {p:.2f} to appear in {df.iloc[0, 0]!r}"
    print("All tests in `test_top_tokens_table` passed!")


def test_answer_ranks(answer_ranks: Callable):
    logits = t.tensor([[[3.0, 1.0, 2.0], [0.0, 5.0, 1.0]], [[1.0, 2.0, 3.0], [9.0, 8.0, 7.0]]])
    answers = t.tensor([2, 1])
    ranks = answer_ranks(logits, answers)
    assert ranks.shape == (2, 2), f"Expected shape (layer, batch) = (2, 2), got {tuple(ranks.shape)}"
    expected = t.tensor([[2, 1], [1, 2]])
    assert t.equal(ranks.long(), expected), f"Expected ranks {expected.tolist()}, got {ranks.tolist()}"
    # Ties: rank should count only strictly larger logits (so a tie for top is still rank 1)
    tie = answer_ranks(t.tensor([[[1.0, 1.0, 0.0]]]), t.tensor([1]))
    assert tie.item() == 1, "A token tied for the largest logit should have rank 1"
    print("All tests in `test_answer_ranks` passed!")


def test_other_answer_ranks(other_answer_ranks: Callable, answer_ranks: Callable):
    t.manual_seed(0)
    logits = t.randn(3, 4, 50)
    answers = t.tensor([3, 7, 11, 20])
    result = other_answer_ranks(logits, answers)
    assert result.shape == (3, 4), f"Expected shape (layer, batch) = (3, 4), got {tuple(result.shape)}"
    for i in range(4):
        others = t.cat([answers[:i], answers[i + 1 :]])
        ranks = t.stack([answer_ranks(logits[:, i : i + 1], others[j : j + 1])[:, 0] for j in range(3)], dim=-1)
        expected = ranks.float().median(dim=-1).values
        t.testing.assert_close(result[:, i].float(), expected)
    print("All tests in `test_other_answer_ranks` passed!")


def test_kl_to_final(kl_to_final: Callable):
    t.manual_seed(0)
    logits = t.randn(5, 3, 20)
    result = kl_to_final(logits)
    assert result.shape == (5,), f"Expected one KL value per layer, got shape {tuple(result.shape)}"
    final = logits[-1].log_softmax(-1)
    for layer in range(5):
        expected = F.kl_div(logits[layer].log_softmax(-1), final, log_target=True, reduction="batchmean")
        t.testing.assert_close(result[layer], expected)
    assert result[-1].abs() < 1e-6, "The final layer should have zero KL from itself"
    print("All tests in `test_kl_to_final` passed!")


def test_tuned_lens_init(TunedLens: type, model: HookedTransformer):
    lens = TunedLens(model.cfg.n_layers, model.cfg.d_model).to(model.cfg.device)
    n_params = sum(p.numel() for p in lens.parameters())
    expected = model.cfg.n_layers * (model.cfg.d_model**2 + model.cfg.d_model)
    assert n_params == expected, f"Expected {expected} parameters (one d_model x d_model matrix + bias per layer), got {n_params}"
    resid = t.randn(2, 3, model.cfg.d_model, device=model.cfg.device)
    for layer in [0, model.cfg.n_layers - 1]:
        t.testing.assert_close(lens(resid, layer), resid, msg="At initialisation the tuned lens should be the identity")
    with t.no_grad():
        for p in lens.parameters():
            p.normal_(std=0.01)
    out = lens(resid, 3)
    t.testing.assert_close(out, resid + resid @ lens.weight[3] + lens.bias[3])
    print("All tests in `test_tuned_lens_init` passed!")


def test_tuned_lens_loss(tuned_lens_loss: Callable, TunedLens: type, model: HookedTransformer):
    t.manual_seed(0)
    n_layers, d_model = model.cfg.n_layers, model.cfg.d_model
    lens = TunedLens(n_layers, d_model).to(model.cfg.device)
    tokens = model.to_tokens(["Once upon a time there was a little cat.", "The dog ran to the park and played."])
    logits, cache = model.run_with_cache(tokens)
    resid_stack = t.stack([cache["resid_pre", layer] for layer in range(n_layers)])
    final_logprobs = logits.log_softmax(-1)
    loss = tuned_lens_loss(lens, model, resid_stack, final_logprobs)
    assert loss.ndim == 0, "The loss should be a scalar"
    expected = 0.0
    for layer in range(n_layers):
        lens_logprobs = model.unembed(model.ln_final(resid_stack[layer])).log_softmax(-1)
        expected += F.kl_div(lens_logprobs[:, 1:], final_logprobs[:, 1:], log_target=True, reduction="none").sum(-1).mean()
    expected = expected / n_layers
    t.testing.assert_close(loss, expected, atol=1e-3, rtol=1e-3, msg="Loss at init should equal the mean (over layers) logit-lens KL, ignoring position 0")
    assert loss.requires_grad, "The loss should be differentiable with respect to the lens parameters"
    print("All tests in `test_tuned_lens_loss` passed!")


def test_get_source_vectors(get_source_vectors: Callable, model: HookedTransformer):
    prompts = [" Germany", " Spain"]
    vecs = get_source_vectors(model, prompts, pos=-1, layer=3)
    assert vecs.shape == (2, model.cfg.d_model), f"Expected shape (2, d_model), got {tuple(vecs.shape)}"
    _, cache = model.run_with_cache(model.to_tokens(prompts))
    t.testing.assert_close(vecs, cache["resid_pre", 3][:, -1])
    print("All tests in `test_get_source_vectors` passed!")


def test_run_patchscope(run_patchscope: Callable, model: HookedTransformer):
    target = model.to_tokens("The capital of x is")
    pos, layer = 4, 2
    _, cache = model.run_with_cache(target)
    own = cache["resid_pre", layer][:, pos]
    clean = model(target)[:, -1]
    # Patching a prompt's own activation back in shouldn't change anything
    out = run_patchscope(model, own, target, target_pos=pos, target_layer=layer)
    assert out.shape == (1, model.cfg.d_vocab), f"Expected final-position logits of shape (batch, d_vocab), got {tuple(out.shape)}"
    t.testing.assert_close(out, clean, atol=1e-3, rtol=1e-3, msg="Patching in the target's own activation should leave the output unchanged")
    # Patching different vectors for each batch element should give different outputs, and broadcast the target
    other = model.run_with_cache(model.to_tokens(" Germany"))[1]["resid_pre", layer][:, -1]
    vecs = t.cat([own, other])
    out2 = run_patchscope(model, vecs, target, target_pos=pos, target_layer=layer)
    assert out2.shape == (2, model.cfg.d_vocab), "run_patchscope should broadcast a single target prompt across the batch of source vectors"
    t.testing.assert_close(out2[0], clean[0], atol=1e-3, rtol=1e-3)
    assert not t.allclose(out2[1], clean[0], atol=1e-2), "Patching a different vector should change the output"
    # It shouldn't permanently modify the model
    t.testing.assert_close(model(target)[:, -1], clean, msg="run_patchscope left a hook attached to the model")
    print("All tests in `test_run_patchscope` passed!")


def test_patchscope_accuracy_grid(patchscope_accuracy_grid: Callable):
    """Checks the grid against a brute-force loop over a fake `run_patchscope`; no model is loaded."""
    from unittest.mock import patch

    t.manual_seed(0)
    n_source, batch, n_vocab = 3, 5, 4
    sources = t.randn(n_source, batch, n_vocab)
    answers = t.tensor([0, 1, 2, 3, 1])
    target, target_pos, target_layers = t.tensor([[7, 8, 9]]), 1, [3, 0, 2]
    calls = []

    def fake_run_patchscope(model, vectors, target_tokens, target_pos_arg, target_layer):
        # The "model" reads the patched vector directly, but flips its sign at odd target layers.
        calls.append(target_layer)
        assert t.equal(target_tokens, target) and target_pos_arg == target_pos, "Pass the target prompt and position on unchanged"
        return vectors * (-1.0 if target_layer % 2 else 1.0)

    with patch.dict(unwrap(patchscope_accuracy_grid).__globals__, run_patchscope=fake_run_patchscope):
        result = patchscope_accuracy_grid(object(), sources, target, target_pos, answers, target_layers)
    expected = t.zeros(n_source, len(target_layers))
    for i in range(n_source):
        for j, layer in enumerate(target_layers):
            hits = 0
            for b in range(batch):
                row = sources[i, b] * (-1.0 if layer % 2 else 1.0)
                hits += int(row.argmax().item() == answers[b].item())
            expected[i, j] = hits / batch
    assert tuple(result.shape) == (n_source, len(target_layers)), (
        f"Expected shape (n_source_layers, n_target_layers) = {(n_source, len(target_layers))}, got {tuple(result.shape)}"
    )
    assert expected.unique().numel() > 2, "The fixture should give a mix of accuracies"
    t.testing.assert_close(result.float().cpu(), expected, msg="Entry [i, j] is the accuracy of source layer i patched into target_layers[j]")
    assert len(calls) == n_source * len(target_layers), "Run one Patchscope per (source layer, target layer) pair"
    print("All tests in `test_patchscope_accuracy_grid` passed!")


def _metric_fixture():
    """An explicit three-sequence distribution fixture; no tokenizer or model weights."""
    from types import SimpleNamespace

    tokens = t.tensor([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    # Position zero deliberately disagrees: neither metric may include it.
    raw = t.tensor([
        [[[.1, .9], [.9, .1], [.8, .2]], [[.1, .9], [.6, .4], [.7, .3]], [[.1, .9], [.2, .8], [.3, .7]]],
        [[[.1, .9], [.8, .2], [.2, .8]], [[.1, .9], [.7, .3], [.3, .7]], [[.1, .9], [.6, .4], [.4, .6]]],
    ])
    final = t.tensor([[[.75, .25], [.75, .25], [.75, .25]]]).expand(3, -1, -1)
    model = SimpleNamespace(cfg=SimpleNamespace(n_layers=2), ln_final=lambda x: x, unembed=lambda x: x)

    class Lens:
        def logits(self, model, resid, layer):
            return resid.flip(-1) if layer == 0 else resid

    def get_resid(model, batch):
        indices = batch[:, 0]
        return raw[:, indices].log(), final[indices].log()

    return model, Lens(), tokens, raw, final, get_resid


def test_evaluate_lenses(evaluate_lenses: Callable):
    from unittest.mock import patch

    model, lens, tokens, raw, final, get_resid = _metric_fixture()
    tokens_before = tokens.clone()
    # Independent oracle: library KL on all evaluated tokens at once, no batch means.
    expected = {key: [] for key in ('logit_kl', 'tuned_kl', 'logit_agree', 'tuned_agree')}
    for layer in range(2):
        for name, probability in [('logit', raw[layer]), ('tuned', raw[layer].flip(-1) if layer == 0 else raw[layer])]:
            q, p = probability[:, 1:], final[:, 1:]
            expected[name + '_kl'].append(F.kl_div(q.log(), p, reduction='none').sum(-1).mean().item())
            expected[name + '_agree'].append((q[..., 0] > q[..., 1]).float().mean().item())
    with patch.dict(unwrap(evaluate_lenses).__globals__, get_resid_and_logprobs=get_resid):
        for batch_size in (1, 2, 3, 8):
            actual = evaluate_lenses(model, lens, tokens, batch_size=batch_size)
            assert actual.keys() == expected.keys()
            for key in expected:
                t.testing.assert_close(t.tensor(actual[key]), t.tensor(expected[key]), atol=1e-6, rtol=1e-6,
                                       msg=f'{key}: weight every non-BOS token equally, including a short final batch')
    t.testing.assert_close(tokens, tokens_before)
    assert abs(expected['logit_agree'][0] - 2 / 3) < 1e-6
    print('All tests in `test_evaluate_lenses` passed!')


def test_shifted_translator_agreement(shifted_translator_agreement: Callable):
    from unittest.mock import patch

    model, lens, tokens, _, _, get_resid = _metric_fixture()
    # Translator 1 leaves layer 0 unchanged: four of six tokens agree.
    # Translator 0 flips layer 1: three of six tokens agree.
    expected = {1: {0: 2 / 3}, -1: {1: .5}, 0: {0: 1 / 3, 1: .5}, 3: {}}
    with patch.dict(unwrap(shifted_translator_agreement).__globals__, get_resid_and_logprobs=get_resid):
        for shift, values in expected.items():
            for batch_size in (1, 2, 3, 8):
                result = shifted_translator_agreement(model, lens, tokens, shift=shift, batch_size=batch_size)
                assert result.keys() == values.keys(), 'Return only layers with a valid shifted translator'
                for layer, agreement in values.items():
                    assert abs(result[layer] - agreement) < 1e-6, 'Use the shifted translator and count non-BOS tokens equally'
    print('All tests in `test_shifted_translator_agreement` passed!')


def test_train_tuned_lens(train_tuned_lens: Callable):
    from unittest.mock import patch

    tokens = t.arange(24).reshape(6, 4)
    before = tokens.clone()

    class Lens(t.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = t.nn.Parameter(t.tensor(0.0))

    def run(lr, seed):
        lens, batches = Lens(), []

        def get_resid(model, batch):
            batches.append(batch.clone())
            return batch.float(), None

        def loss_fn(lens, model, resid, target):
            return (lens.weight - 1).square()

        with patch.dict(unwrap(train_tuned_lens).__globals__, get_resid_and_logprobs=get_resid, tuned_lens_loss=loss_fn):
            losses = train_tuned_lens(lens, object(), tokens, n_steps=3, batch_size=3, lr=lr, seed=seed)
        assert len(losses) == 3 and all(isinstance(loss, float) for loss in losses)
        assert len(batches) == 3 and all(batch.shape == (3, 4) for batch in batches)
        return lens.weight.detach(), losses, batches

    # Independent scalar Adam oracle: d/dw (w-1)^2 = 2(w-1). Use plain Python
    # arithmetic and bias-corrected moments, rather than another torch optimizer.
    from math import sqrt

    expected_weight, first_moment, second_moment = 0., 0., 0.
    expected_losses = []
    for step in range(1, 4):
        expected_losses.append((expected_weight - 1) ** 2)
        gradient = 2 * (expected_weight - 1)
        first_moment = .9 * first_moment + .1 * gradient
        second_moment = .999 * second_moment + .001 * gradient ** 2
        expected_weight -= .1 * (first_moment / (1 - .9 ** step)) / (sqrt(second_moment / (1 - .999 ** step)) + 1e-8)

    weight, losses, batches = run(.1, 17)
    t.testing.assert_close(t.tensor(losses), t.tensor(expected_losses), atol=1e-6, rtol=1e-6,
                           msg='Clear gradients before every backward call; return pre-update losses')
    assert abs(float(weight) - expected_weight) < 1e-6, 'Match the three-step Adam trajectory without accumulated gradients'
    frozen, frozen_losses, _ = run(0, 17)
    assert frozen == 0 and frozen_losses == [1., 1., 1.], 'Respect the supplied learning rate'
    again, again_losses, again_batches = run(.1, 17)
    t.testing.assert_close(weight, again)
    assert losses == again_losses and all(t.equal(a, b) for a, b in zip(batches, again_batches)), 'Use a seeded batch sampler'
    t.testing.assert_close(tokens, before)
    print('All tests in `test_train_tuned_lens` passed!')


def test_patchscope_controls(patchscope_controls: Callable):
    from types import SimpleNamespace
    from unittest.mock import patch

    vectors, bos = 2 * t.eye(3), t.tensor([[2., 0., 0.]]).expand(3, -1).clone()
    random = vectors.roll(1, dims=-1)
    target, answers = t.tensor([[8, 9]]), t.tensor([0, 1, 2])
    snapshots = [x.clone() for x in (vectors, bos, target, answers)]
    calls, source_calls = [], []

    class Model:
        cfg = SimpleNamespace(device='cpu')

        def __call__(self, tokens):
            assert t.equal(tokens, target.expand(3, -1)), 'Broadcast the target prompt across source examples'
            return t.tensor([[[0., 0., 1.]]]).expand(3, 2, 3)

    model = Model()

    def get_source(model_arg, prompts, pos, layer):
        assert model_arg is model and prompts == ['a', 'b', 'c'] and layer == 4
        source_calls.append(pos)
        return vectors if pos == -1 else bos

    def random_like(x):
        t.testing.assert_close(x, vectors)
        return random

    def patchscope(model_arg, x, target_arg, target_pos, target_layer):
        assert model_arg is model and target_pos == 1 and target_layer == 2
        t.testing.assert_close(target_arg, target)
        calls.append(x.clone())
        return x

    with patch.dict(unwrap(patchscope_controls).__globals__, get_source_vectors=get_source,
                    random_vectors_like=random_like, run_patchscope=patchscope):
        result = patchscope_controls(model, ['a', 'b', 'c'], 4, target, 1, 2, answers)
    expected = {'patchscope': 1., 'text only (no patch)': 1 / 3,
                'random vector (norm-matched)': 0., 'wrong position (BOS)': 1 / 3}
    assert result.keys() == expected.keys()
    for key, value in expected.items():
        assert abs(result[key] - value) < 1e-6, f'Incorrect accuracy for {key}'
    assert source_calls == [-1, 0] and len(calls) == 3
    for actual, expected_vectors in zip(calls, (vectors, random, bos)):
        t.testing.assert_close(actual, expected_vectors)
    for actual, saved in zip((vectors, bos, target, answers), snapshots):
        t.testing.assert_close(actual, saved, msg='Controls must not mutate source data or targets')
    print('All tests in `test_patchscope_controls` passed!')


def test_attention_lens(attention_lens: Callable, model: HookedTransformer):
    t.manual_seed(0)
    cfg = model.cfg
    batch = 3
    tokens = t.randint(0, cfg.d_vocab, (batch, 7), device=cfg.device)
    answers = t.tensor([5, 11, 2], device=cfg.device)
    logits, cache = model.run_with_cache(tokens)
    heads = attention_lens(model, cache, answers)
    assert tuple(heads.shape) == (cfg.n_layers, cfg.n_heads, batch), (
        f"Expected shape (layer, head, batch) = {(cfg.n_layers, cfg.n_heads, batch)}, got {tuple(heads.shape)}"
    )
    # Independent oracle: every other term that is added to the residual stream, read the same way.
    scale = cache["ln_final.hook_scale"][:, -1]  # the final layernorm's scale at the last position, shape (batch, 1)

    def read(x):  # a (batch, d_model) vector written to the stream -> its direct effect on the answer's logit
        x = x - x.mean(-1, keepdim=True)
        return ((x / scale) * model.W_U[:, answers].T).sum(-1)

    rest = read(cache["hook_embed"][:, -1] + cache["hook_pos_embed"][:, -1])
    for layer in range(cfg.n_layers):
        rest = rest + read(cache["mlp_out", layer][:, -1]) + read(model.b_O[layer].expand(batch, -1))
    total = heads.sum(dim=(0, 1)) + rest + model.b_U[answers]
    final = logits[:, -1].gather(-1, answers[:, None])[:, 0]
    t.testing.assert_close(
        total, final, atol=1e-3, rtol=1e-3,
        msg="The heads, MLPs, embeddings and biases should add up to the answer's final logit. "
        "Divide by the final layernorm's scale (from the cache), not by each head's own norm or by 1.",
    )
    print("All tests in `test_attention_lens` passed!")


def test_normalize_rows(normalize_rows: Callable):
    x = t.tensor([[3.0, 4.0], [0.0, 0.0], [-1.0, 0.0]])
    out = normalize_rows(x)
    assert out.shape == x.shape, f"Expected shape {tuple(x.shape)}, got {tuple(out.shape)}"
    assert t.isfinite(out).all(), "A zero row should stay zero, not turn into NaN"
    t.testing.assert_close(out, t.tensor([[0.6, 0.8], [0.0, 0.0], [-1.0, 0.0]]))
    t.manual_seed(0)
    y = t.randn(5, 7)
    t.testing.assert_close(normalize_rows(y).norm(dim=-1), t.ones(5))
    t.testing.assert_close(normalize_rows(4.0 * y), normalize_rows(y), msg="Scaling a row by a positive number should change nothing")
    t.testing.assert_close(normalize_rows(-y), -normalize_rows(y), msg="Normalising must keep each row's direction")
    print("All tests in `test_normalize_rows` passed!")


def _brute_force_topk(queries, bank, k):
    """Cosine top-k by plain Python loops, lowest bank index first among exact ties."""
    scores, indices = [], []
    for q in queries.tolist():
        row = []
        for index, b in enumerate(bank.tolist()):
            dot = sum(x * y for x, y in zip(q, b))
            norm = (sum(x * x for x in q) ** 0.5) * (sum(y * y for y in b) ** 0.5)
            row.append((dot / norm if norm else 0.0, index))
        best = sorted(row, key=lambda pair: (-pair[0], pair[1]))[:k]
        scores.append([score for score, _ in best])
        indices.append([index for _, index in best])
    return t.tensor(scores), t.tensor(indices)


def test_contextual_topk(contextual_topk: Callable):
    # Hand-worked: bank rows 0 and 2 point the same way (an exact tie), row 1 is orthogonal, row 3 is opposite.
    bank = t.tensor([[1.0, 0.0], [0.0, 2.0], [3.0, 0.0], [-1.0, 0.0]])
    queries = t.tensor([[5.0, 0.0], [0.0, 0.1]])
    scores, indices = contextual_topk(queries, bank, k=3)
    assert tuple(scores.shape) == (2, 3) and tuple(indices.shape) == (2, 3), "Return (scores, indices), each of shape (n_queries, k)"
    assert indices.tolist() == [[0, 2, 1], [1, 0, 2]], (
        f"Ties should go to the lower bank index, so the neighbours of query 0 are rows 0, 2, 1; got {indices[0].tolist()}"
    )
    t.testing.assert_close(scores, t.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]]))
    # Positive scale does not matter, for queries or for bank rows
    s2, i2 = contextual_topk(7.0 * queries, 0.25 * bank, k=3)
    assert t.equal(i2, indices), "Cosine neighbours should not change when queries or bank rows are rescaled"
    t.testing.assert_close(s2, scores)
    # Chunked search equals brute force, whatever the chunk size
    t.manual_seed(0)
    q, b = t.randn(23, 6), t.randn(31, 6)
    expected_scores, expected_indices = _brute_force_topk(q, b, k=4)
    for chunk_size in (1, 5, 23, 1000):
        got_scores, got_indices = contextual_topk(q, b, k=4, chunk_size=chunk_size)
        assert t.equal(got_indices, expected_indices), f"chunk_size={chunk_size}: neighbours differ from the brute-force search"
        t.testing.assert_close(got_scores, expected_scores, atol=1e-5, rtol=1e-5)
    print("All tests in `test_contextual_topk` passed!")


def test_layer_alignment_grid(layer_alignment_grid: Callable):
    # Hand-worked: two layers in 2-D. Bank vectors point along x (label 10) and y (label 20) at layer 0, swapped at layer 1.
    bank = t.tensor([[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]])
    bank_labels = t.tensor([10, 20])
    queries = t.tensor([
        [[1.0, 0.1], [0.1, 1.0], [1.0, 0.1], [0.1, 1.0]],
        [[0.2, 1.0], [1.0, 0.2], [0.2, 1.0], [0.2, 1.0]],
    ])
    targets = t.tensor([10, 20, 10, 20])
    grid = layer_alignment_grid(queries, bank, targets, bank_labels)
    expected = t.tensor([[1.0, 0.0], [0.25, 0.75]])
    assert tuple(grid.shape) == (2, 2), f"Expected a (query layer, bank layer) grid of shape (2, 2), got {tuple(grid.shape)}"
    t.testing.assert_close(grid.float(), expected, msg="Entry [i, j]: queries from layer i, bank from layer j")
    # Brute force on random data, with a chunk size that doesn't divide the queries
    t.manual_seed(1)
    q, b = t.randn(3, 17, 5), t.randn(3, 29, 5)
    q_targets, b_labels = t.randint(0, 3, (17,)), t.randint(0, 3, (29,))
    got = layer_alignment_grid(q, b, q_targets, b_labels, chunk_size=4)
    want = t.zeros(3, 3)
    for i in range(3):
        for j in range(3):
            _, nearest = _brute_force_topk(q[i], b[j], k=1)
            want[i, j] = (b_labels[nearest[:, 0]] == q_targets).float().mean()
    t.testing.assert_close(got.float(), want)
    print("All tests in `test_layer_alignment_grid` passed!")


def test_ablate_blocks_recall(ablate_blocks_recall: Callable, model: HookedTransformer):
    t.manual_seed(0)
    cfg = model.cfg
    n_prompts, seq, pos = 24, 5, -2
    tokens = t.randint(0, cfg.d_vocab, (n_prompts, seq), device=cfg.device)
    clean_logits, cache = model.run_with_cache(tokens)
    answers = clean_logits[:, -1].argmax(-1)  # so the unablated model is 100% "accurate"
    assert ablate_blocks_recall(model, tokens, answers, blocks=[], pos=pos) == 1.0, "Ablating no blocks should change nothing"
    same = tokens[:1].expand(6, -1)
    assert ablate_blocks_recall(model, same, answers[:1].expand(6), blocks=list(range(cfg.n_layers)), pos=pos) == 1.0, (
        "If every prompt is identical, replacing an activation by its mean over prompts should change nothing"
    )
    # Independent oracle: one prompt at a time, with the clean mean taken in plain Python
    blocks = [0, 1]
    result = ablate_blocks_recall(model, tokens, answers, blocks=blocks, pos=pos)
    hits = 0
    for i in range(n_prompts):
        def hook_for(block):
            mean = sum(cache["mlp_out", block][j, pos] for j in range(n_prompts)) / n_prompts

            def hook(mlp_out, hook):
                mlp_out[0, pos] = mean
                return mlp_out
            return hook

        logits = model.run_with_hooks(tokens[i : i + 1], fwd_hooks=[(f"blocks.{b}.hook_mlp_out", hook_for(b)) for b in blocks])
        hits += int(logits[0, -1].argmax().item() == answers[i].item())
    assert abs(result - hits / n_prompts) < 1e-6, (
        f"Expected accuracy {hits / n_prompts:.3f} after mean-ablating the MLP outputs of blocks {blocks} at position {pos}, got {result:.3f}"
    )
    assert hits < n_prompts, "The fixture should be one where the ablation matters"
    print("All tests in `test_ablate_blocks_recall` passed!")
