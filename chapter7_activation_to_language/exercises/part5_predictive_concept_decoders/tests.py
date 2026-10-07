from typing import Callable

import numpy as np
import pandas as pd
import torch as t
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from transformer_lens import HookedTransformer


def test_planted_behaviour(planted_behaviour: Callable, make_planted_world: Callable):
    world = make_planted_world(seed=3)
    acts = t.randn(5, world.concept_dirs.shape[1])
    out = planted_behaviour(world, acts)
    n_q, n_v, n_c = world.W_behaviour.shape
    assert out.shape == (5, n_q, n_v), f"Expected shape (batch, n_questions, n_answers) = {(5, n_q, n_v)}, got {tuple(out.shape)}"
    # Independent oracle: explicit loops
    for b in range(5):
        proj = world.concept_dirs @ acts[b]
        for q in range(n_q):
            expected = (world.W_behaviour[q] @ proj + world.b_behaviour[q]).log_softmax(-1)
            t.testing.assert_close(out[b, q], expected, atol=1e-5, rtol=1e-5)
    t.testing.assert_close(out.exp().sum(-1), t.ones(5, n_q), msg="Output should be log-probabilities (normalised over answers)")
    # Adding a loud direction must not change behaviour
    t.testing.assert_close(planted_behaviour(world, acts + 5 * world.loud_dirs[0]), out, atol=1e-5, rtol=1e-5,
                           msg="Behaviour should only depend on the projection onto the concept directions")
    print("All tests in `test_planted_behaviour` passed!")


def test_topk_relu(topk_relu: Callable):
    t.manual_seed(0)
    pre = t.randn(7, 10)
    for k in [1, 3, 10]:
        out = topk_relu(pre, k)
        assert out.shape == pre.shape, f"Output should have the same shape as the input, got {tuple(out.shape)}"
        # Oracle: threshold at the k-th largest value of each row
        kth = pre.sort(dim=-1, descending=True).values[:, k - 1 : k]
        expected = t.where(pre >= kth, pre.clamp_min(0), t.zeros_like(pre))
        t.testing.assert_close(out, expected, msg=f"Wrong output for k={k}")
    assert (topk_relu(-pre.abs() - 1, 3) == 0).all(), "Negative values should be zeroed by the ReLU even if they're in the top k"
    assert topk_relu(t.randn(2, 4, 6), 2).shape == (2, 4, 6), "topk_relu should work with extra leading dimensions"
    x = t.randn(4, 8, requires_grad=True)
    topk_relu(x, 2).sum().backward()
    kept = topk_relu(x.detach(), 2) > 0
    assert (x.grad[~kept] == 0).all() and (x.grad[kept] == 1).all(), "Gradients should flow through the kept (positive) entries only"
    print("All tests in `test_topk_relu` passed!")


def test_pcd_module(PCD: type):
    t.manual_seed(0)
    pcd = PCD(d_model=10, n_latents=6, k=2, n_questions=3, n_answers=5, d_hidden=16)
    with t.no_grad():
        pcd.b_pre.normal_()
        pcd.b_enc.normal_()
    acts = t.randn(8, 10)
    z = pcd.encode(acts)
    assert z.shape == (8, 6), f"encode should return (batch, n_latents) = (8, 6), got {tuple(z.shape)}"
    assert ((z > 0).sum(-1) <= 2).all(), "At most k latents should be active per example"
    assert (z >= 0).all(), "Concept activations should be non-negative"
    pre = (acts - pcd.b_pre) @ pcd.W_enc + pcd.b_enc
    kth = pre.sort(dim=-1, descending=True).values[:, 1:2]
    expected_z = t.where(pre >= kth, pre.clamp_min(0), t.zeros_like(pre))
    t.testing.assert_close(z, expected_z, msg="encode should be TopK(ReLU((acts - b_pre) @ W_enc + b_enc))")
    logits = pcd.decode(z)
    assert logits.shape == (8, 3, 5), f"decode should return (batch, n_questions, n_answers) = (8, 3, 5), got {tuple(logits.shape)}"
    expected = pcd.mlp((z @ pcd.W_emb)[:, None, :] + pcd.question_emb[None])
    t.testing.assert_close(logits, expected, msg="decode should apply the MLP to (soft token + question embedding)")
    assert not t.allclose(logits[:, 0], logits[:, 1]), "Different questions should give different answers"
    t.testing.assert_close(pcd(acts), logits)
    print("All tests in `test_pcd_module` passed!")


def test_pcd_loss(pcd_loss: Callable, PCD: type):
    t.manual_seed(0)
    pcd = PCD(d_model=10, n_latents=6, k=2, n_questions=3, n_answers=5, d_hidden=16)
    acts = t.randn(8, 10)
    target = t.randn(8, 3, 5).log_softmax(-1)
    loss = pcd_loss(pcd, acts, target)
    assert loss.ndim == 0, "The loss should be a scalar"
    pred = pcd(acts).log_softmax(-1)
    expected = F.kl_div(pred.reshape(-1, 5), target.reshape(-1, 5), log_target=True, reduction="batchmean")
    t.testing.assert_close(loss, expected, msg="Loss should be KL(target || prediction), summed over answers, averaged over batch and questions")
    assert loss.requires_grad, "The loss should be differentiable"
    zero = pcd_loss(pcd, acts, pred.detach())
    assert zero.abs() < 1e-5, "The loss should be zero when the target equals the PCD's own prediction"
    print("All tests in `test_pcd_loss` passed!")


def test_train_bottleneck(train_bottleneck: Callable, PCD: type, pcd_loss: Callable):
    t.manual_seed(0)
    acts = t.randn(512, 8)
    targets = t.stack([(acts[:, :2] @ t.randn(2, 4)).log_softmax(-1) for _ in range(2)], dim=1)  # 2 questions
    pcd = PCD(d_model=8, n_latents=6, k=3, n_questions=2, n_answers=4, d_hidden=32)
    losses = train_bottleneck(pcd, pcd_loss, acts, targets, n_steps=300, batch_size=64, lr=3e-3)
    assert len(losses) == 300, f"Should return one loss per step, got {len(losses)}"
    assert np.mean(losses[-20:]) < 0.5 * np.mean(losses[:20]), "The loss should go down substantially on an easy problem"
    # targets=None must also work (for the SAE): the loss function should then receive None as its third argument
    seen = []

    def loss_fn(model, acts_batch, targets_batch):
        seen.append(targets_batch)
        return model.encode(acts_batch).sum()

    train_bottleneck(pcd, loss_fn, acts, None, n_steps=2, batch_size=8)
    assert len(seen) == 2 and all(y is None for y in seen), "With targets=None, loss_fn should be called with None as targets"
    print("All tests in `test_train_bottleneck` passed!")


def test_latent_alignment(latent_alignment: Callable):
    t.manual_seed(0)
    W_enc, true_dirs = t.randn(12, 5), t.randn(7, 12)
    out = latent_alignment(W_enc, true_dirs)
    assert out.shape == (7, 5), f"Expected shape (n_true, n_latents) = (7, 5), got {tuple(out.shape)}"
    expected = F.cosine_similarity(true_dirs[:, None, :], W_enc.T[None, :, :], dim=-1)
    t.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)
    t.testing.assert_close(latent_alignment(3 * W_enc, 0.1 * true_dirs), out, atol=1e-5, rtol=1e-5, msg="Should be scale-invariant")
    print("All tests in `test_latent_alignment` passed!")


def test_topk_sae(TopKSAE: type, sae_loss: Callable):
    t.manual_seed(0)
    sae = TopKSAE(d_model=10, n_latents=6, k=2)
    with t.no_grad():
        sae.b_pre.normal_()
        sae.b_enc.normal_()
    acts = t.randn(8, 10)
    z = sae.encode(acts)
    assert z.shape == (8, 6) and ((z > 0).sum(-1) <= 2).all(), "encode should return a (batch, n_latents) TopK code"
    pre = (acts - sae.b_pre) @ sae.W_enc + sae.b_enc
    kth = pre.sort(dim=-1, descending=True).values[:, 1:2]
    t.testing.assert_close(z, t.where(pre >= kth, pre.clamp_min(0), t.zeros_like(pre)))
    recon = sae(acts)
    assert recon.shape == acts.shape, "forward should return a reconstruction with the same shape as the input"
    t.testing.assert_close(recon, z @ sae.W_dec + sae.b_pre)
    loss = sae_loss(sae, acts, None)
    t.testing.assert_close(loss, ((recon - acts) ** 2).sum(-1).mean(), msg="sae_loss should be the squared error summed over d_model, averaged over the batch")
    print("All tests in `test_topk_sae` passed!")


def test_auroc(auroc: Callable):
    rng = np.random.default_rng(0)
    for _ in range(5):
        labels = rng.integers(0, 2, 300)
        scores = np.round(rng.normal(size=300) + labels, 1)  # rounding creates lots of ties
        scores[rng.random(300) < 0.4] = 0.0  # and a big block of exact zeros, like a TopK latent
        expected = roc_auc_score(labels, scores)
        result = auroc(t.tensor(scores, dtype=t.float), t.tensor(labels, dtype=t.float))
        assert abs(result - expected) < 1e-6, f"AUROC {result:.6f} doesn't match sklearn's {expected:.6f} (check ties!)"
    assert abs(auroc(t.tensor([1.0, 2.0, 3.0, 4.0]), t.tensor([0.0, 0.0, 1.0, 1.0])) - 1.0) < 1e-6
    assert abs(auroc(t.tensor([4.0, 3.0, 2.0, 1.0]), t.tensor([0.0, 0.0, 1.0, 1.0])) - 0.0) < 1e-6
    assert abs(auroc(t.zeros(6), t.tensor([0.0, 1.0, 0.0, 1.0, 1.0, 0.0])) - 0.5) < 1e-6, "A constant score should give 0.5"
    print("All tests in `test_auroc` passed!")


def test_best_latent_auroc(best_latent_auroc: Callable):
    y_train = t.tensor([0.0, 0.0, 1.0, 1.0, 0.0, 1.0])
    y_test = t.tensor([0.0, 1.0, 0.0, 1.0])
    # latent 1 is perfect on train; latent 0 is better on test, but we must pick on train
    z_train = t.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [1.0, 3.0], [0.0, 0.0], [0.5, 1.0]])
    z_test = t.tensor([[0.0, 1.0], [2.0, 0.0], [0.0, 0.0], [3.0, 1.0]])
    best, test_auc = best_latent_auroc(z_train, y_train, z_test, y_test)
    assert best == 1, f"Should pick the latent with the best *training* AUROC (latent 1), got {best}"
    expected = roc_auc_score(y_test.numpy(), z_test[:, 1].numpy())
    assert abs(test_auc - expected) < 1e-6, f"Should report the chosen latent's *test* AUROC ({expected:.3f}), got {test_auc:.3f}"
    print("All tests in `test_best_latent_auroc` passed!")


def _ridge_oracle(X_train, y_train, X_test, y_test, l2):
    X_train, X_test = X_train.numpy().astype(np.float64), X_test.numpy().astype(np.float64)
    mean, std = X_train.mean(0), X_train.std(0, ddof=1) + 1e-6
    X_train, X_test = (X_train - mean) / std, (X_test - mean) / std
    y = 2 * y_train.numpy().astype(np.float64) - 1
    w = np.linalg.solve(X_train.T @ X_train + l2 * np.eye(X_train.shape[1]), X_train.T @ (y - y.mean()))
    pred = (X_test @ w + y.mean()) > 0
    return balanced_accuracy_score(y_test.numpy().astype(bool), pred)


def test_readout_accuracy(readout_accuracy: Callable):
    t.manual_seed(0)
    X_train, X_test = t.randn(400, 6), t.randn(200, 6)
    w = t.randn(6)
    y_train, y_test = (X_train @ w > 0.3).float(), (X_test @ w > 0.3).float()
    for l2 in [1.0, 10.0]:
        result = readout_accuracy(X_train, y_train, X_test, y_test, l2=l2)
        expected = _ridge_oracle(X_train, y_train, X_test, y_test, l2)
        assert abs(result - expected) < 1e-4, f"Got {result:.4f}, expected {expected:.4f} (l2={l2})"
    assert result > 0.85, "A linear readout should do well on linearly separable data"
    # Balanced accuracy: an uninformative feature with imbalanced labels should give ~0.5, not the majority rate
    y_imb_train, y_imb_test = (t.rand(400) < 0.2).float(), (t.rand(200) < 0.2).float()
    noise = readout_accuracy(t.randn(400, 1), y_imb_train, t.randn(200, 1), y_imb_test)
    assert abs(noise - 0.5) < 0.1, f"Use balanced accuracy: an uninformative feature should score ~0.5, got {noise:.3f}"
    print("All tests in `test_readout_accuracy` passed!")


def test_shuffled_label_readout(shuffled_label_readout: Callable):
    t.manual_seed(0)
    X_train, X_test = t.randn(2000, 4), t.randn(1000, 4)
    y_train, y_test = (X_train[:, 0] > 0).float(), (X_test[:, 0] > 0).float()
    result = shuffled_label_readout(X_train, y_train, X_test, y_test, n_shuffles=5, seed=0)
    assert abs(result - 0.5) < 0.06, f"With shuffled training labels, held-out accuracy should be near 0.5, got {result:.3f}"
    print("All tests in `test_shuffled_label_readout` passed!")


def test_last_token_logprobs_and_acts(last_token_logprobs_and_acts: Callable, model: HookedTransformer):
    prompts = ["The queen won the prize.", "Yesterday, Mary and Anna lost the big game in the park."]
    probe, layer = " After that,", 5
    logprobs, acts = last_token_logprobs_and_acts(model, prompts, probe, layer)
    assert logprobs.shape == (2, model.cfg.d_vocab), f"logprobs should be (batch, d_vocab), got {tuple(logprobs.shape)}"
    assert acts.shape == (2, model.cfg.d_model), f"acts should be (batch, d_model), got {tuple(acts.shape)}"
    for i, prompt in enumerate(prompts):
        # Independent oracle: run each prompt on its own (no padding)
        logits, cache = model.run_with_cache(prompt + probe, names_filter=f"blocks.{layer}.hook_resid_pre")
        t.testing.assert_close(logprobs[i], logits[0, -1].log_softmax(-1), atol=1e-3, rtol=1e-3,
                               msg="logprobs should be the next-token log-probs at the final token of prompt + probe")
        read_pos = model.to_tokens(prompt).shape[1] - 1
        t.testing.assert_close(acts[i], cache[f"blocks.{layer}.hook_resid_pre"][0, read_pos], atol=1e-3, rtol=1e-3,
                               msg="acts should be resid_pre[read_layer] at the final token of the *prompt*")
    assert not model.hook_dict[f"blocks.{layer}.hook_resid_pre"].fwd_hooks, "A hook was left attached to the model"
    print("All tests in `test_last_token_logprobs_and_acts` passed!")


def test_restrict_to_answer_vocab(restrict_to_answer_vocab: Callable):
    t.manual_seed(0)
    logprobs = t.randn(3, 2, 50).log_softmax(-1)
    ids = t.tensor([4, 17, 9])
    out = restrict_to_answer_vocab(logprobs, ids)
    assert out.shape == (3, 2, 4), f"Expected shape (..., n_answers + 1) = (3, 2, 4), got {tuple(out.shape)}"
    t.testing.assert_close(out[..., :3], logprobs[..., ids], atol=1e-5, rtol=1e-5)
    other = 1 - logprobs.exp()[..., ids].sum(-1)
    t.testing.assert_close(out[..., 3].exp(), other, atol=1e-5, rtol=1e-5, msg="The last entry should be the remaining probability")
    t.testing.assert_close(out.exp().sum(-1), t.ones(3, 2), atol=1e-4, rtol=1e-4)
    full = t.full((1, 5), -1e9)
    full[0, 2] = 0.0  # all mass on token 2
    assert t.isfinite(restrict_to_answer_vocab(full.log_softmax(-1), t.tensor([2, 3]))).all(), "Clamp to avoid log(0)"
    print("All tests in `test_restrict_to_answer_vocab` passed!")


def test_latent_dashboard(latent_dashboard: Callable):
    z = t.tensor([[0.0, 1.0], [2.0, 0.0], [1.0, 0.0], [0.0, 3.0]])
    prompts = ["a", "b", "c", "d"]
    df = latent_dashboard(z, prompts, n_top=2)
    assert isinstance(df, pd.DataFrame) and len(df) == 2, "Should return a DataFrame with one row per latent"
    assert abs(df.loc[0, "freq"] - 0.5) < 1e-6 and abs(df.loc[1, "freq"] - 0.5) < 1e-6, "freq should be the fraction of examples with z > 0"
    assert list(df.loc[0, ["top 1", "top 2"]]) == ["b", "c"], f"Wrong top prompts for latent 0: {list(df.loc[0])}"
    assert list(df.loc[1, ["top 1", "top 2"]]) == ["d", "a"], f"Wrong top prompts for latent 1: {list(df.loc[1])}"
    print("All tests in `test_latent_dashboard` passed!")


def test_latent_group_aurocs(latent_group_aurocs: Callable, auroc: Callable):
    t.manual_seed(0)
    z = t.relu(t.randn(50, 3))
    groups = {"a": (t.rand(50) < 0.5).float(), "b": (z[:, 1] > 0.3).float()}
    out = latent_group_aurocs(z, groups)
    assert out.shape == (3, 2), f"Expected shape (n_latents, n_groups) = (3, 2), got {tuple(out.shape)}"
    for j in range(3):
        for g, (name, labels) in enumerate(groups.items()):
            expected = roc_auc_score(labels.numpy(), z[:, j].numpy())
            assert abs(out[j, g].item() - expected) < 1e-5, f"Wrong AUROC for latent {j}, group {name}"
    print("All tests in `test_latent_group_aurocs` passed!")


def test_concept_vector(concept_vector: Callable):
    t.manual_seed(0)
    acts, z = t.randn(20, 5), t.relu(t.randn(20, 3))
    out = concept_vector(acts, z, 2)
    on = (z[:, 2] > 0).numpy()
    expected = acts.numpy()[on].mean(0) - acts.numpy()[~on].mean(0)
    np.testing.assert_allclose(out.numpy(), expected, atol=1e-5)
    print("All tests in `test_concept_vector` passed!")


def test_planted_effects(planted_effects: Callable):
    t.manual_seed(0)
    W = t.randn(6, 2 * 3)
    behaviour_fn = lambda a: (a @ W).reshape(-1, 2, 3).log_softmax(-1)
    decoder = lambda a: (a @ W).reshape(-1, 2, 3) * 2.0  # logits, not log-probs
    acts, vec = t.randn(4, 6), t.randn(6)
    predicted, actual = planted_effects(decoder, behaviour_fn, acts, vec)
    assert predicted.shape == actual.shape == (4 * 2 * 3,), f"Outputs should be flattened, got {tuple(predicted.shape)}"
    t.testing.assert_close(actual, (behaviour_fn(acts - vec) - behaviour_fn(acts)).flatten())
    expected = (decoder(acts - vec).log_softmax(-1) - decoder(acts).log_softmax(-1)).flatten()
    t.testing.assert_close(predicted, expected, msg="Predicted changes should be in log-probs (apply log_softmax to the decoder's logits)")
    print("All tests in `test_planted_effects` passed!")


def test_make_edits(make_edits: Callable, PCD: type, concept_vector: Callable):
    t.manual_seed(0)
    pcd = PCD(d_model=8, n_latents=5, k=2, n_questions=2, n_answers=3, d_hidden=8)
    with t.no_grad():
        pcd.b_pre.normal_()
    acts = t.randn(200, 8)
    z = pcd.encode(acts).detach()
    j = int(((z > 0).float().mean(0) - 0.5).abs().argmin())  # a latent that is on for some examples and off for others
    edits = make_edits(pcd, acts, z, j, seed=0)
    assert set(edits) == {"encoder direction", "effect vector", "random vector"}, f"Unexpected keys {set(edits)}"
    w_hat = F.normalize(pcd.W_enc[:, j].detach(), dim=0)
    off_value = ((acts[z[:, j] == 0] - pcd.b_pre.detach()) @ w_hat).mean()
    test = t.randn(3, 4, 8)  # edits should work with any leading dimensions
    edited = edits["encoder direction"](test)
    assert edited.shape == test.shape
    t.testing.assert_close((edited - pcd.b_pre.detach()) @ w_hat, off_value.expand(3, 4), atol=1e-5, rtol=1e-5,
                           msg="After the encoder-direction edit, the projection should equal its mean over inactive examples")
    diff = edited - test
    t.testing.assert_close(diff - (diff @ w_hat)[..., None] * w_hat, t.zeros_like(diff), atol=1e-5, rtol=1e-5,
                           msg="The encoder-direction edit should only move activations along w_hat")
    v = concept_vector(acts, z, j)
    t.testing.assert_close(edits["effect vector"](test), test - v, atol=1e-5, rtol=1e-5)
    r = test - edits["random vector"](test)
    t.testing.assert_close(r[0, 0].norm(), v.norm(), atol=1e-4, rtol=1e-4, msg="The random vector should have the same norm as the effect vector")
    assert F.cosine_similarity(r[0, 0], v, dim=0).abs() < 0.9, "The random vector shouldn't be (anti)parallel to the effect vector"
    print("All tests in `test_make_edits` passed!")


def test_behaviour_under_edit(behaviour_under_edit: Callable, model: HookedTransformer, prompts: list[str], clean_expected: t.Tensor):
    clean = behaviour_under_edit(model, prompts, None)
    assert clean.shape == clean_expected.shape, f"Expected shape {tuple(clean_expected.shape)}, got {tuple(clean.shape)}"
    t.testing.assert_close(clean, clean_expected, atol=2e-3, rtol=2e-3, msg="With no edit, the output should match the clean behaviour")
    identity = behaviour_under_edit(model, prompts, lambda x: x.clone(), "all", batch_size=2)
    t.testing.assert_close(identity, clean_expected, atol=2e-3, rtol=2e-3, msg="An identity edit (with batch_size=2) should change nothing")
    zeroed = behaviour_under_edit(model, prompts, lambda x: 0 * x, "read")
    assert not t.allclose(zeroed[:, 0], clean_expected[:, 0], atol=1e-2), "Zeroing the read position should change the next-token prediction there"
    hooks_left = [name for name, hp in model.hook_dict.items() if hp.fwd_hooks]
    assert not hooks_left, f"Hooks were left attached: {hooks_left}"
    print("All tests in `test_behaviour_under_edit` passed!")


def test_faithfulness_summary(faithfulness_summary: Callable):
    t.manual_seed(0)
    clean = t.randn(6, 2, 5).log_softmax(-1)
    edited = (clean + t.randn(6, 2, 5)).log_softmax(-1)
    out = faithfulness_summary(clean, edited, clean, edited)
    assert set(out) == {"actual KL", "predicted KL", "r"}, f"Unexpected keys {set(out)}"
    expected_kl = F.kl_div(edited.reshape(-1, 5), clean.reshape(-1, 5), log_target=True, reduction="batchmean").item()
    assert abs(out["actual KL"] - expected_kl) < 1e-5 and abs(out["predicted KL"] - expected_kl) < 1e-5
    assert abs(out["r"] - 1.0) < 1e-5, "Identical predicted and actual changes should have correlation 1"
    other = (clean + t.randn(6, 2, 5)).log_softmax(-1)
    out2 = faithfulness_summary(clean, edited, clean, other, min_prob=0.0)
    mask = t.ones(6, 2, 4, dtype=t.bool)
    d_pred, d_act = (other - clean)[..., :-1][mask], (edited - clean)[..., :-1][mask]
    expected_r = np.corrcoef(d_pred.numpy(), d_act.numpy())[0, 1]
    assert abs(out2["r"] - expected_r) < 1e-4, "r should be computed over the answer tokens only (excluding the final 'other' bucket)"
    print("All tests in `test_faithfulness_summary` passed!")
