from typing import Callable

import numpy as np
import torch as t
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from transformer_lens import HookedTransformer


def test_get_final_activations(get_final_activations: Callable, model: HookedTransformer):
    tokens = model.to_tokens(["Yesterday, the keys near the cat really", "Today, the dog with the trees really"])
    result = get_final_activations(model, tokens, [2, 5], batch_size=1)
    assert result.shape == (2, 2, model.cfg.d_model), f"Expected shape (layer, batch, d_model) = (2, 2, {model.cfg.d_model}), got {tuple(result.shape)}"
    assert result.device.type == "cpu", "Return a CPU tensor"
    _, cache = model.run_with_cache(tokens)
    for i, layer in enumerate([2, 5]):
        t.testing.assert_close(result[i], cache["resid_pre", layer][:, -1].cpu(), atol=1e-4, rtol=1e-4,
                               msg=f"Wrong activations for layer {layer} (did you take resid_pre at the final position?)")
    print("All tests in `test_get_final_activations` passed!")


def test_fve(fve: Callable):
    t.manual_seed(0)
    h = t.randn(50, 8) * t.rand(50, 1) * 10  # rows with different norms
    h_unit = F.normalize(h, dim=-1)
    assert abs(fve(h, h_unit) - 1.0) < 1e-5, "Perfect reconstruction of the normalised activations should give FVE = 1"
    assert abs(fve(h, h_unit.mean(0, keepdim=True).expand_as(h))) < 1e-5, "Predicting the mean should give FVE = 0"
    noisy = h_unit + 0.3 * t.randn_like(h_unit)
    expected = r2_score(h_unit.numpy(), noisy.numpy(), multioutput="variance_weighted")
    assert abs(fve(h, noisy) - expected) < 1e-4, f"Expected FVE {expected:.4f} (sklearn's variance-weighted R^2), got {fve(h, noisy):.4f}"
    assert abs(fve(3.0 * h, noisy) - expected) < 1e-4, "FVE should not depend on the norm of the original activations"
    assert fve(h, -h_unit) < 0, "A reconstruction worse than the mean should have negative FVE"
    assert isinstance(fve(h, noisy), float), "Return a Python float (use .item())"
    print("All tests in `test_fve` passed!")


def test_class_mean_reconstruction(class_mean_reconstruction: Callable):
    train_h = t.tensor([[2.0, 0.0], [0.0, 3.0], [4.0, 0.0], [0.0, 1.0]])
    train_labels = ["a", ("b", 1), "a", ("b", 1)]
    result = class_mean_reconstruction(train_h, train_labels, [("b", 1), "a", "unseen"])
    expected = t.tensor([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])
    assert result.shape == (3, 2), f"Expected one reconstruction per test label, got shape {tuple(result.shape)}"
    t.testing.assert_close(result.float(), expected, msg=(
        "Expected the mean of the *unit-normalised* training activations with the same label, "
        "and the overall training mean for unseen labels"
    ))
    print("All tests in `test_class_mean_reconstruction` passed!")


def test_patch_final_position(patch_final_position: Callable, model: HookedTransformer, verb_ids: t.Tensor):
    tokens = model.to_tokens(["Yesterday, the keys near the cat really", "Today, the dog with the trees really"])
    layer = 6
    logits, cache = model.run_with_cache(tokens)
    own = cache["resid_pre", layer][:, -1]
    result = patch_final_position(model, tokens, own, layer, verb_ids, batch_size=1)
    assert result.shape == (2, 4), f"Expected the four verb logits for each sentence, shape (2, 4), got {tuple(result.shape)}"
    t.testing.assert_close(result, logits[:, -1, verb_ids.to(logits.device)].cpu(), atol=1e-3, rtol=1e-3,
                           msg="Patching each sentence's own activation back in should not change its output")
    swapped = patch_final_position(model, tokens, own.flip(0), layer, verb_ids)
    assert not t.allclose(swapped, result, atol=1e-2), "Patching different vectors should change the output"

    # Independent check of the swapped run: overwrite the final position by hand with a hook
    def hook(resid, hook):
        resid[:, -1] = own.flip(0)
        return resid

    expected = model.run_with_hooks(tokens, fwd_hooks=[(f"blocks.{layer}.hook_resid_pre", hook)])[:, -1, verb_ids.to(logits.device)]
    t.testing.assert_close(swapped, expected.cpu(), atol=1e-3, rtol=1e-3)
    t.testing.assert_close(model(tokens)[:, -1], logits[:, -1], msg="patch_final_position left a hook attached to the model")
    print("All tests in `test_patch_final_position` passed!")


def test_verbaliser_forward(ActivationVerbaliser: type):
    t.manual_seed(0)
    av = ActivationVerbaliser(d_act=16, n_words=10, d_model=32, n_layers=2, n_heads=4, max_words=4)
    h = t.randn(3, 16)
    prev = t.randint(0, 10, (3, 4))
    out = av(h, prev)
    assert out.shape == (3, 5, 10), f"Expected logits of shape (batch, t + 1, n_words) = (3, 5, 10), got {tuple(out.shape)}"
    # Causality: changing word k must not change the outputs at positions <= k
    prev2 = prev.clone()
    prev2[:, 2] = (prev2[:, 2] + 1) % 10
    out2 = av(h, prev2)
    t.testing.assert_close(out[:, :3], out2[:, :3], msg="Output at position i must not depend on words after position i (causal mask?)")
    assert not t.allclose(out[:, 3:], out2[:, 3:]), "Output after position k should depend on word k"
    # The activation should matter, at every position; and only its direction should matter
    out3 = av(t.randn(3, 16), prev)
    assert not t.allclose(out[:, -1], out3[:, -1]), "The output should depend on the activation h"
    t.testing.assert_close(av(5.0 * h, prev), out, msg="The verbaliser should normalise h, so scaling h shouldn't change the output")
    # Embeddings path
    t.testing.assert_close(av(h, prev_embeds=av.word_embed(prev)), out)
    # Zero previous words: just the activation position
    assert av(h, prev[:, :0]).shape == (3, 1, 10), "With no previous words, the output should have one position"
    print("All tests in `test_verbaliser_forward` passed!")


def test_description_logprobs(description_logprobs: Callable, ActivationVerbaliser: type):
    t.manual_seed(0)
    av = ActivationVerbaliser(d_act=16, n_words=10, d_model=32, n_layers=1, n_heads=4, max_words=3)
    h = t.randn(2, 16)
    ids = t.tensor([[1, 2, 9, 9], [3, 4, 5, 9]])
    logprobs, token_logprobs = description_logprobs(av, h, ids)
    assert logprobs.shape == (2, 4, 10), f"Expected full log-probs of shape (batch, slots, n_words) = (2, 4, 10), got {tuple(logprobs.shape)}"
    assert token_logprobs.shape == (2, 4), f"Expected per-word log-probs of shape (batch, slots) = (2, 4), got {tuple(token_logprobs.shape)}"
    t.testing.assert_close(logprobs.exp().sum(-1), t.ones(2, 4), msg="logprobs should be normalised log-probabilities")
    # Independent oracle: score slot i by running the verbaliser on the first i words only
    for b in range(2):
        for i in range(4):
            expected = av(h[b : b + 1], ids[b : b + 1, :i]).log_softmax(-1)[0, -1, ids[b, i]]
            t.testing.assert_close(token_logprobs[b, i], expected, atol=1e-5, rtol=1e-5,
                                   msg=f"Wrong log-probability for slot {i} of description {b}")
    print("All tests in `test_description_logprobs` passed!")


def test_description_mask(description_mask: Callable, EOS: int):
    other = 0 if EOS != 0 else 1
    ids = t.tensor([
        [other, other, EOS, EOS, EOS],
        [EOS, EOS, EOS, EOS, EOS],
        [other, other, other, other, EOS],
        [other, EOS, other, EOS, EOS],
    ])
    expected = t.tensor([
        [True, True, True, False, False],
        [True, False, False, False, False],
        [True, True, True, True, True],
        [True, True, False, False, False],
    ])
    result = description_mask(ids)
    assert result.dtype == t.bool, f"Expected a boolean mask, got {result.dtype}"
    assert t.equal(result, expected), f"Expected mask\n{expected}\ngot\n{result}"
    print("All tests in `test_description_mask` passed!")


class _FixedLogitsVerbaliser(t.nn.Module):
    """A stub verbaliser whose next-word logits depend only on the position."""

    def __init__(self, table: t.Tensor, max_words: int):
        super().__init__()
        self.table = table  # (max_words, n_words)
        self.max_words = max_words

    def forward(self, h, prev_ids=None, prev_embeds=None):
        n_prev = prev_ids.shape[1]
        return self.table[: n_prev + 1].unsqueeze(0).expand(h.shape[0], -1, -1).clone()


def test_sample_descriptions(sample_descriptions: Callable, ActivationVerbaliser: type, EOS: int):
    n_words = EOS + 1
    # Deterministic stub: word 2, then word 5, then EOS
    table = t.full((3, n_words), -1e4)
    table[0, 2], table[1, 5], table[2, EOS] = 0.0, 0.0, 0.0
    stub = _FixedLogitsVerbaliser(table, max_words=3)
    ids = sample_descriptions(stub, t.randn(4, 8))
    assert ids.shape == (4, 4), f"Expected shape (batch, max_words + 1) = (4, 4), got {tuple(ids.shape)}"
    assert t.equal(ids, t.tensor([[2, 5, EOS, EOS]] * 4)), f"Expected every row to be [2, 5, EOS, EOS], got {ids.tolist()}"
    # EOS first, then words: everything after the first EOS must be EOS
    table = t.full((3, n_words), -1e4)
    table[0, EOS], table[1, 5], table[2, 6] = 0.0, 0.0, 0.0
    ids = sample_descriptions(_FixedLogitsVerbaliser(table, 3), t.randn(2, 8))
    assert t.equal(ids, t.full((2, 4), EOS)), f"Slots after the first EOS should all be EOS, got {ids.tolist()}"
    # Sampling frequencies with temperature
    t.manual_seed(0)
    table = t.full((2, n_words), float("-inf"))
    table[0, 0], table[0, 1] = 0.0, float(np.log(3.0))  # p(word 1) = 0.75 at temperature 1
    table[1, EOS] = 0.0
    ids = sample_descriptions(_FixedLogitsVerbaliser(table, 2), t.randn(4000, 8))
    freq = (ids[:, 0] == 1).float().mean().item()
    assert abs(freq - 0.75) < 0.03, f"Expected word 1 to be sampled ~75% of the time, got {freq:.1%}"
    ids = sample_descriptions(_FixedLogitsVerbaliser(table, 2), t.randn(4000, 8), temperature=1e6)
    freq = (ids[:, 0] == 1).float().mean().item()
    assert abs(freq - 0.5) < 0.03, f"At a very high temperature, the two words should be ~equally likely, got {freq:.1%}"
    ids = sample_descriptions(_FixedLogitsVerbaliser(table, 2), t.randn(10, 8), greedy=True)
    assert (ids[:, 0] == 1).all(), "Greedy decoding should always pick the most likely word"
    # Greedy decoding on a real verbaliser matches step-by-step argmax decoding
    t.manual_seed(1)
    av = ActivationVerbaliser(d_act=8, n_words=n_words, d_model=32, n_layers=1, n_heads=4, max_words=3)
    h = t.randn(5, 8)
    ids = sample_descriptions(av, h, greedy=True)
    prefix = t.zeros(5, 0, dtype=t.long)
    for _ in range(3):
        prefix = t.cat([prefix, av(h, prefix)[:, -1].argmax(-1, keepdim=True)], dim=1)
    expected = t.cat([prefix, t.full((5, 1), EOS)], dim=1)
    after_eos = (expected == EOS).long().cumsum(-1) - (expected == EOS).long() > 0
    expected[after_eos] = EOS
    assert t.equal(ids, expected), f"Greedy decoding doesn't match step-by-step argmax decoding:\n{ids}\nvs\n{expected}"
    print("All tests in `test_sample_descriptions` passed!")


def test_text_features(text_features: Callable, model: HookedTransformer, WORD_TOKEN_IDS: t.Tensor, encode_descriptions: Callable):
    ids = encode_descriptions([["houses", "earlier"], ["plural", "past", "behind", "window"]])
    result = text_features(model, ids.to(model.cfg.device), 10)
    assert result.shape == (2, model.cfg.d_model), f"Expected shape (batch, d_model), got {tuple(result.shape)}"
    dot = model.to_single_token(".")
    for row, words in enumerate([[" houses", " earlier"], [" plural", " past", " behind", " window"]]):
        toks = [model.tokenizer.bos_token_id] + [model.to_single_token(w) for w in words]
        toks += [dot] * (6 - len(toks))
        _, cache = model.run_with_cache(t.tensor([toks]).to(model.cfg.device))
        t.testing.assert_close(result[row], cache["resid_pre", 10][0, -1], atol=1e-3, rtol=1e-3,
                               msg=f"Wrong features for description {row}: expected resid_pre[10] at the final '.' of BOS + words + dots")
    print("All tests in `test_text_features` passed!")


def test_reconstructor(ActivationReconstructor: type):
    t.manual_seed(0)
    ar = ActivationReconstructor(d_features=12, d_act=6)
    features = t.randn(4, 12) * 50
    out = ar(features)
    assert out.shape == (4, 6), f"Expected shape (batch, d_act) = (4, 6), got {tuple(out.shape)}"
    t.testing.assert_close(out, ar.head(features / features.norm(dim=-1, keepdim=True)),
                           msg="Expected head(features / ||features||)")
    t.testing.assert_close(ar(3.0 * features), out, msg="The output shouldn't depend on the norm of the features")
    n_params = sum(p.numel() for p in ar.parameters())
    assert n_params == 12 * 6 + 6, f"The only parameters should be the affine head, expected {12 * 6 + 6}, got {n_params}"
    print("All tests in `test_reconstructor` passed!")


def test_claim_accuracy(claim_accuracy: Callable):
    fact = {"subject": ["houses"], "number": ["plural"], "tense": ["past"], "time": ["earlier"], "object": ["behind", "window"]}
    cases = [
        (["houses earlier"], (1.0, 2.0)),
        (["houses earlier behind window"], (1.0, 3.0)),
        (["house past"], (0.5, 2.0)),  # wrong subject, right tense
        (["houses behind windows"], (0.5, 2.0)),  # object claim with the wrong number is false
        (["behind"], (0.0, 1.0)),  # a preposition with no noun is a false claim
        (["singular present today"], (0.0, 3.0)),
        (["window"], (0.0, 1.0)),  # a noun not after a preposition is a subject claim
        (["houses earlier", "plural"], (1.0, 1.5)),  # pooled over descriptions
    ]
    for descriptions, expected in cases:
        result = claim_accuracy(descriptions, [fact] * len(descriptions))
        assert np.allclose(result, expected), f"For {descriptions}, expected (fraction true, claims per description) = {expected}, got {result}"
    print("All tests in `test_claim_accuracy` passed!")


def test_group_advantages(group_advantages: Callable):
    rewards = t.tensor([1.0, 2.0, 3.0, 10.0, 10.0, 13.0])
    result = group_advantages(rewards, group_size=3)
    assert result.shape == (6,), f"Expected the same shape as the rewards, got {tuple(result.shape)}"
    expected = t.tensor([-1.0, 0.0, 1.0, -1 / np.sqrt(3), -1 / np.sqrt(3), 2 / np.sqrt(3)]).float()
    t.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4,
                           msg="Each consecutive group should be normalised by its own mean and (unbiased) std")
    t.manual_seed(0)
    r = t.randn(40)
    a = group_advantages(r, group_size=8).view(5, 8)
    t.testing.assert_close(a.mean(1), t.zeros(5), atol=1e-5, rtol=0)
    t.testing.assert_close(a.std(1), t.ones(5), atol=1e-4, rtol=0)
    print("All tests in `test_group_advantages` passed!")


def test_kl_to_reference(kl_to_reference: Callable):
    t.manual_seed(0)
    logprobs = t.randn(3, 4, 7).log_softmax(-1)
    ref = t.randn(3, 4, 7).log_softmax(-1)
    mask = t.tensor([[True, True, False, False], [True, True, True, True], [True, False, False, False]])
    result = kl_to_reference(logprobs, ref, mask)
    assert result.shape == (3,), f"Expected one KL value per description, got shape {tuple(result.shape)}"
    expected = t.stack([
        sum(F.kl_div(ref[b, i], logprobs[b, i], log_target=True, reduction="sum") for i in range(4) if mask[b, i])
        for b in range(3)
    ])
    t.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5, msg="Expected sum over unmasked slots of KL(policy || reference)")
    t.testing.assert_close(kl_to_reference(logprobs, logprobs, mask), t.zeros(3), atol=1e-6, rtol=0)
    print("All tests in `test_kl_to_reference` passed!")


def test_policy_gradient_loss(policy_gradient_loss: Callable):
    token_logprobs = t.tensor([[-1.0, -2.0, -3.0], [-0.5, -0.5, -4.0]], requires_grad=True)
    mask = t.tensor([[True, True, False], [True, True, True]])
    advantages = t.tensor([1.0, -2.0])
    loss = policy_gradient_loss(token_logprobs, mask, advantages)
    assert loss.ndim == 0, "The loss should be a scalar"
    expected = -((1.0 * (-3.0)) + (-2.0 * (-5.0))) / 2
    assert abs(loss.item() - expected) < 1e-5, f"Expected {expected}, got {loss.item()}"
    loss.backward()
    expected_grad = t.tensor([[-0.5, -0.5, 0.0], [1.0, 1.0, 1.0]])
    t.testing.assert_close(token_logprobs.grad, expected_grad, msg="Masked slots should get zero gradient")
    print("All tests in `test_policy_gradient_loss` passed!")


def test_straight_through_gumbel(straight_through_gumbel: Callable):
    t.manual_seed(0)
    logits = t.tensor([[0.0, float(np.log(3.0)), -1e4]]).expand(20000, -1).clone().requires_grad_(True)
    y = straight_through_gumbel(logits, tau=1.0)
    assert y.shape == logits.shape
    assert t.allclose(y.detach().sum(-1), t.ones(20000)) and set(y.detach().unique().tolist()) <= {0.0, 1.0}, \
        "The forward value should be exactly one-hot"
    freq = y.detach()[:, 1].mean().item()
    assert abs(freq - 0.75) < 0.02, f"Samples should follow softmax(logits): expected word 1 ~75% of the time, got {freq:.1%}"
    # Backward: the gradient should be that of the soft sample, not zero
    weights = t.tensor([1.0, 2.0, 3.0])
    (y * weights).sum().backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0, "Gradients should flow to the logits (straight-through)"
    # Check against the gradient of a softmax evaluated at the same noise: recompute with a fixed seed
    t.manual_seed(1)
    l2 = t.randn(4, 5, requires_grad=True)
    t.manual_seed(2)
    y2 = straight_through_gumbel(l2, tau=0.5)
    t.manual_seed(2)
    u = t.rand_like(l2).clamp(1e-9, 1 - 1e-9)
    soft = ((l2 + -t.log(-t.log(u))) / 0.5).softmax(-1)
    g = t.randn(4, 5)
    grad_st = t.autograd.grad((y2 * g).sum(), l2)[0]
    grad_soft = t.autograd.grad((soft * g).sum(), l2)[0]
    t.testing.assert_close(grad_st, grad_soft, msg="The backward pass should be the gradient of softmax((logits + gumbel) / tau)")
    t.testing.assert_close(y2.detach(), F.one_hot(soft.argmax(-1), 5).float(), msg="The forward value should be the one-hot argmax of (logits + gumbel)")
    print("All tests in `test_straight_through_gumbel` passed!")


def test_pca_reconstruction(pca_reconstruction: Callable):
    t.manual_seed(0)
    train = t.randn(200, 10) @ t.randn(10, 10)
    test = t.randn(50, 10) @ t.randn(10, 10)
    train_unit, test_unit = F.normalize(train, dim=-1), F.normalize(test, dim=-1)
    full = pca_reconstruction(train, test, 10)
    t.testing.assert_close(full, test_unit, atol=1e-4, rtol=1e-4, msg="With all components, PCA should reconstruct the (normalised) test set exactly")
    pca = PCA(n_components=3).fit(train_unit.numpy())
    expected = t.tensor(pca.inverse_transform(pca.transform(test_unit.numpy()))).float()
    t.testing.assert_close(pca_reconstruction(train, test, 3), expected, atol=1e-4, rtol=1e-4,
                           msg="Doesn't match sklearn's PCA fitted on the normalised training set")
    print("All tests in `test_pca_reconstruction` passed!")


def test_bag_of_words(bag_of_words: Callable):
    vocab = ["the", "dogs", "cat", "near", "Today"]
    result = bag_of_words(["Today, the dogs near the cat really", "the cat"], vocab)
    expected = t.tensor([[2.0, 1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 1.0, 0.0, 0.0]])
    assert result.shape == (2, 5), f"Expected shape (n_sentences, len(vocab)) = (2, 5), got {tuple(result.shape)}"
    t.testing.assert_close(result.float(), expected, msg="Expected word counts (with the comma removed, unknown words ignored)")
    t.testing.assert_close(bag_of_words(["the cat near the dogs"], vocab), bag_of_words(["the dogs near the cat"], vocab),
                           msg="A bag of words should ignore word order")
    print("All tests in `test_bag_of_words` passed!")


def test_fit_ridge(fit_ridge: Callable):
    t.manual_seed(0)
    X = t.randn(100, 5)
    Y = X @ t.randn(5, 3) + 2.0 + 0.1 * t.randn(100, 3)
    predict = fit_ridge(X, Y, alpha=3.0)
    X_new = t.randn(10, 5)
    sk = Ridge(alpha=3.0, fit_intercept=True).fit(X.numpy(), Y.numpy())
    t.testing.assert_close(predict(X_new).float(), t.tensor(sk.predict(X_new.numpy())).float(), atol=1e-4, rtol=1e-4,
                           msg="Doesn't match sklearn's Ridge(alpha, fit_intercept=True)")
    print("All tests in `test_fit_ridge` passed!")


def test_transplant_accuracy(transplant_accuracy: Callable, model: HookedTransformer, verb_ids: t.Tensor):
    sentences = [
        "Yesterday, the keys near the cat really",  # were
        "Today, the dog with the trees really",  # is
        "Earlier, the house behind the windows really",  # was
        "Now, the girls beside the door really",  # are
    ]
    tokens = model.to_tokens(sentences)
    labels = t.tensor([3, 0, 2, 1])
    layer = 10
    _, cache = model.run_with_cache(tokens)
    acts = cache["resid_pre", layer][:, -1].cpu()
    scale = acts.norm(dim=-1).mean().item()
    src, dst = t.tensor([0, 1, 2, 3]), t.tensor([1, 0, 3, 2])

    # Independent oracle: patch the rescaled vectors in by hand, and count how often the source's verb wins
    def expected_accuracy(h_hat):
        vectors = (h_hat / h_hat.norm(dim=-1, keepdim=True) * scale).to(model.cfg.device)

        def hook(resid, hook):
            resid[:, -1] = vectors
            return resid

        logits = model.run_with_hooks(tokens[dst], fwd_hooks=[(f"blocks.{layer}.hook_resid_pre", hook)])
        preds = logits[:, -1, verb_ids.to(logits.device)].argmax(-1).cpu()
        return (preds == labels[src]).float().mean().item()

    unit = acts / acts.norm(dim=-1, keepdim=True)
    for h_hat in [unit, 0.3 * unit + 0.01 * t.randn_like(unit), unit[t.tensor([2, 3, 0, 1])]]:
        result = transplant_accuracy(model, tokens, labels, h_hat, src, dst, scale, layer)
        expected = expected_accuracy(h_hat)
        assert abs(result - expected) < 1e-6, f"Expected transplant accuracy {expected:.2f}, got {result:.2f}"
    assert abs(transplant_accuracy(model, tokens, labels, 7.0 * unit, src, dst, scale, layer)
               - transplant_accuracy(model, tokens, labels, unit, src, dst, scale, layer)) < 1e-6, \
        "Reconstructions should be rescaled to unit norm (then multiplied by `scale`) before patching"
    print("All tests in `test_transplant_accuracy` passed!")
