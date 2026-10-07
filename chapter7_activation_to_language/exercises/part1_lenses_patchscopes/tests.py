from typing import Callable

import pandas as pd
import torch as t
import torch.nn.functional as F
from transformer_lens import HookedTransformer


def test_logit_lens(logit_lens: Callable, model: HookedTransformer):
    tokens = model.to_tokens(["The capital of France is", "One, two, three, four"])
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


def test_patchscope_accuracy_grid(patchscope_accuracy_grid: Callable, model: HookedTransformer):
    import part1_lenses_patchscopes.solutions as solutions

    sources = solutions.get_all_layer_source_vectors(model, [" Germany", " Spain"], pos=-1)
    target = model.to_tokens("The capital of France is Paris. The capital of x is")
    answers = model.to_tokens([" Berlin", " Madrid"], prepend_bos=False)[:, 0]
    target_pos = target.shape[1] - 2
    result = patchscope_accuracy_grid(model, sources[:3], target, target_pos, answers, target_layers=[0, 1])
    expected = solutions.patchscope_accuracy_grid(model, sources[:3], target, target_pos, answers, target_layers=[0, 1])
    assert result.shape == (3, 2), f"Expected shape (n_source_layers, n_target_layers) = (3, 2), got {tuple(result.shape)}"
    t.testing.assert_close(result.float().cpu(), expected.float().cpu())
    print("All tests in `test_patchscope_accuracy_grid` passed!")
