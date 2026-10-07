from typing import Callable

import numpy as np
import torch as t
import torch.nn as nn
from scipy import stats


def _raises(fn: Callable, *exceptions) -> bool:
    try:
        fn()
    except exceptions or (ValueError,):
        return True
    return False


def test_split_examples(split_examples: Callable):
    labels = np.array([0] * 41 + [1] * 23)
    splits = split_examples(labels, train_fraction=0.5, dev_fraction=0.25, seed=3)
    assert set(splits) == {"train", "dev", "confirm"}, f"Expected keys train/dev/confirm, got {sorted(splits)}"
    all_ids = np.concatenate(list(splits.values()))
    assert len(all_ids) == len(np.unique(all_ids)), "An example appears in more than one split (or twice in one split)"
    assert sorted(all_ids.tolist()) == list(range(len(labels))), "Every example should be in exactly one split"
    # Stratified: floor(n_class * fraction) of each class goes to train and dev, the rest to confirm
    for name, (n0, n1) in {"train": (20, 11), "dev": (10, 5), "confirm": (11, 7)}.items():
        counts = (int((labels[splits[name]] == 0).sum()), int((labels[splits[name]] == 1).sum()))
        assert counts == (n0, n1), f"Split {name!r} should contain {(n0, n1)} examples of class (0, 1), got {counts}"
    # Reproducible with the same seed, different with a different seed
    again = split_examples(labels, train_fraction=0.5, dev_fraction=0.25, seed=3)
    for name in splits:
        np.testing.assert_array_equal(splits[name], again[name], err_msg="The same seed should give the same split")
    other = split_examples(labels, train_fraction=0.5, dev_fraction=0.25, seed=4)
    assert not np.array_equal(splits["train"], other["train"]), "A different seed should give a different split"
    # Not just the first examples of each class: the split should be shuffled
    assert not np.array_equal(np.sort(splits["train"][labels[splits["train"]] == 0]), np.arange(20)), (
        "The training split is just the first examples of class 0: shuffle within each class before slicing"
    )
    assert _raises(lambda: split_examples(labels, train_fraction=0.8, dev_fraction=0.3), ValueError), (
        "Fractions that leave no confirmation data should raise a ValueError"
    )
    print("All tests in `test_split_examples` passed!")


def test_train_model(train_model: Callable):
    # A synthetic task: class 1 lights up pixel 0, class 0 lights up pixel 1
    x = t.zeros(20, 64, dtype=t.float64)
    x[10:, 0] = 1.0
    x[:10, 1] = 1.0
    y = t.tensor([0] * 10 + [1] * 10)
    x_before = x.clone()
    for kind in ["mlp", "linear"]:
        model, losses = train_model(x, y, kind=kind, hidden_width=4, steps=30, lr=0.05, seed=0)
        assert len(losses) == 30, f"Expected one loss per step (30), got {len(losses)}"
        assert all(isinstance(loss, float) for loss in losses), "Losses should be Python floats (use loss.item())"
        assert losses[-1] < losses[0], f"The loss should go down during training ({kind} model)"
        assert not model.training, "Return the model in eval mode (model.eval())"
        with t.no_grad():
            accuracy = (model(x).argmax(-1) == y).double().mean().item()
        assert accuracy == 1.0, f"The {kind} model should solve this trivial task, got accuracy {accuracy}"
        # Same seed, same model and losses
        model2, losses2 = train_model(x, y, kind=kind, hidden_width=4, steps=30, lr=0.05, seed=0)
        assert losses == losses2, "Training with the same seed should be exactly reproducible (call t.manual_seed(seed))"
    t.testing.assert_close(x, x_before, msg="train_model shouldn't modify its inputs")
    # Independent check of the first loss: at step 0 we evaluate the freshly-initialised model
    t.manual_seed(0)
    from part1_capstone_research_sprint.solutions import DigitClassifier

    fresh = DigitClassifier("linear")
    expected_first = nn.functional.cross_entropy(fresh(x), y).item()
    _, losses = train_model(x, y, kind="linear", steps=1, lr=0.05, seed=0)
    assert abs(losses[0] - expected_first) < 1e-10, (
        "The first loss should be the loss of the model initialised right after t.manual_seed(seed), before any update"
    )
    print("All tests in `test_train_model` passed!")


def test_make_pairs(make_pairs: Callable):
    labels = np.array([1, 0, 0, 1, 0, 1, 1, 0, 0])
    idx0, idx1 = make_pairs(labels)
    np.testing.assert_array_equal(idx0, [1, 2, 4, 7], err_msg="idx0 should be the first n class-0 indices, in order")
    np.testing.assert_array_equal(idx1, [0, 3, 5, 6], err_msg="idx1 should be the first n class-1 indices, in order")
    idx0, idx1 = make_pairs(t.tensor([0, 0, 0, 1]))
    assert len(idx0) == len(idx1) == 1, "With 3 examples of class 0 and 1 of class 1, there's only one pair"
    print("All tests in `test_make_pairs` passed!")


def test_patch_coordinates(patch_coordinates: Callable):
    recipient = t.arange(12, dtype=t.float64).reshape(3, 4)
    donor = -recipient - 100
    before = recipient.clone()
    out = patch_coordinates(recipient, donor, t.tensor([1, 3]))
    expected = recipient.clone()
    expected[:, [1, 3]] = donor[:, [1, 3]]
    t.testing.assert_close(out, expected)
    t.testing.assert_close(recipient, before, msg="Don't modify the recipient in place: clone it first")
    t.testing.assert_close(patch_coordinates(recipient, donor, t.tensor([], dtype=t.long)), recipient)
    t.testing.assert_close(patch_coordinates(recipient, donor, t.arange(4)), donor)
    t.testing.assert_close(patch_coordinates(recipient, donor, [0]), t.cat([donor[:, :1], recipient[:, 1:]], dim=1))
    print("All tests in `test_patch_coordinates` passed!")


def test_patch_direction(patch_direction: Callable):
    t.manual_seed(0)
    recipient, donor = t.randn(5, 6, dtype=t.float64), t.randn(5, 6, dtype=t.float64)
    direction = t.randn(6, dtype=t.float64) * 3.0
    out = patch_direction(recipient, donor, direction)
    unit = direction / direction.norm()
    # Oracle 1: the projection onto the direction should be the donor's
    t.testing.assert_close(out @ unit, donor @ unit, msg="The patched vector should have the donor's projection")
    # Oracle 2: the component orthogonal to the direction should be the recipient's
    proj = t.outer(unit, unit)
    eye = t.eye(6, dtype=t.float64)
    t.testing.assert_close(out @ (eye - proj), recipient @ (eye - proj), msg="The orthogonal part should be unchanged")
    # Sign and scale of the direction don't matter
    t.testing.assert_close(patch_direction(recipient, donor, -direction / 7), out)
    # A coordinate axis direction is the same as patching that coordinate
    axis = t.zeros(6, dtype=t.float64)
    axis[2] = 1.0
    expected = recipient.clone()
    expected[:, 2] = donor[:, 2]
    t.testing.assert_close(patch_direction(recipient, donor, axis), expected)
    before = recipient.clone()
    patch_direction(recipient, donor, direction)
    t.testing.assert_close(recipient, before, msg="Don't modify the recipient in place")
    print("All tests in `test_patch_direction` passed!")


def _linear_head(weight: list[list[float]], bias: list[float]) -> nn.Linear:
    head = nn.Linear(len(weight[0]), 2).double()
    with t.no_grad():
        head.weight.copy_(t.tensor(weight, dtype=t.float64))
        head.bias.copy_(t.tensor(bias, dtype=t.float64))
    return head


def test_evaluate_patch(evaluate_patch: Callable):
    # logit difference (logit 1 - logit 0) = 2 * h[0] + 1 * h[1]
    head = _linear_head([[0.0, 0.0], [2.0, 1.0]], [0.0, 0.0])
    h0 = t.tensor([[-1.0, -1.0], [-2.0, -1.0]], dtype=t.float64)  # class 0: LD = -3, -5
    h1 = t.tensor([[1.0, 1.0], [1.0, 3.0]], dtype=t.float64)  # class 1: LD = 3, 5

    def first_coordinate(r, d):
        out = r.clone()
        out[:, 0] = d[:, 0]
        return out

    out = evaluate_patch(head, h0, h1, first_coordinate)
    assert set(out) == {"swap", "recovered"}, f"Expected keys 'swap' and 'recovered', got {sorted(out)}"
    # Pair 0: 0<-1 gives h=(1,-1), LD=1 (flips to 1, recovered 4/6); 1<-0 gives h=(-1,1), LD=-1 (flips to 0, recovered 4/6)
    # Pair 1: 0<-1 gives h=(1,-1), LD=1 (flips, recovered 6/10); 1<-0 gives h=(-2,3), LD=-1 (flips, recovered 6/10)
    np.testing.assert_allclose(out["swap"], [1.0, 1.0])
    np.testing.assert_allclose(out["recovered"], [4 / 6, 6 / 10])
    out = evaluate_patch(head, h0, h1, lambda r, d: d)
    np.testing.assert_allclose(out["swap"], [1.0, 1.0], err_msg="A full patch should swap every pair")
    np.testing.assert_allclose(out["recovered"], [1.0, 1.0], err_msg="A full patch should recover 100%")
    out = evaluate_patch(head, h0, h1, lambda r, d: r)
    np.testing.assert_allclose(out["swap"], [0.0, 0.0], err_msg="No patch should swap nothing")
    np.testing.assert_allclose(out["recovered"], [0.0, 0.0], err_msg="No patch should recover 0%")

    # Patching the second coordinate moves the logit difference part of the way, but never across zero
    def second_coordinate(r, d):
        out = r.clone()
        out[:, 1] = d[:, 1]
        return out

    out = evaluate_patch(head, h0, h1, second_coordinate)
    # Pair 0: 0<-1 gives h=(-1,1), LD=-1; 1<-0 gives h=(1,-1), LD=1. Pair 1: 0<-1 gives (-2,3), LD=-1; 1<-0 gives (1,-1), LD=1
    np.testing.assert_allclose(out["swap"], [0.0, 0.0], err_msg="No direction flips here, so no pair is swapped")
    np.testing.assert_allclose(out["recovered"], [2 / 6, 4 / 10])

    # A pair only counts as swapped if BOTH directions flip
    def one_way(r, d):
        return d if (r[:, 0] < 0).all() else r  # only the class-0 recipients get the donor's activations

    out = evaluate_patch(head, h0, h1, one_way)
    np.testing.assert_allclose(out["swap"], [0.0, 0.0], err_msg="A pair only counts as swapped if both directions flip")
    np.testing.assert_allclose(out["recovered"], [0.5, 0.5], err_msg="'recovered' should average the two directions")
    print("All tests in `test_evaluate_patch` passed!")


def test_mean_difference_direction(mean_difference_direction: Callable):
    h = t.tensor([[0.0, 0.0], [2.0, 0.0], [0.0, 3.0], [2.0, 3.0]], dtype=t.float64)
    y = t.tensor([0, 0, 1, 1])
    t.testing.assert_close(mean_difference_direction(h, y), t.tensor([0.0, 1.0], dtype=t.float64))
    t.manual_seed(0)
    h = t.randn(50, 7, dtype=t.float64)
    y = (t.rand(50) > 0.4).long()
    direction = mean_difference_direction(h, y)
    expected = h[y == 1].mean(0) - h[y == 0].mean(0)
    t.testing.assert_close(direction, expected / expected.norm())
    t.testing.assert_close(mean_difference_direction(h, 1 - y), -direction, msg="Flipping labels should flip the sign")
    print("All tests in `test_mean_difference_direction` passed!")


def test_rank_coordinates(rank_coordinates: Callable):
    h = t.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [3.0, 2.0, 1.0], [3.0, 2.0, 1.0]], dtype=t.float64)
    y = t.tensor([0, 0, 1, 1])
    readout = t.tensor([0.1, -2.0, 3.0], dtype=t.float64)
    # |mean diff * readout| = [0.3, 4.0, 3.0]
    assert rank_coordinates(h, y, readout).tolist() == [1, 2, 0], "Rank by |class-mean difference x readout weight|"
    assert rank_coordinates(h, y, t.ones(3, dtype=t.float64)).tolist() == [0, 1, 2]
    assert rank_coordinates(h, y, t.zeros(3, dtype=t.float64)).tolist() == [0, 1, 2], (
        "Exact ties should keep the original index order (use a stable sort)"
    )
    print("All tests in `test_rank_coordinates` passed!")


def test_clopper_pearson(clopper_pearson: Callable):
    for k, n in [(0, 44), (44, 44), (1, 10), (17, 44), (43, 44), (5, 5), (0, 1)]:
        for alpha in [0.05, 0.1]:
            low, high = clopper_pearson(k, n, alpha=alpha)
            ci = stats.binomtest(k, n).proportion_ci(confidence_level=1 - alpha, method="exact")
            assert abs(low - ci.low) < 1e-9 and abs(high - ci.high) < 1e-9, (
                f"For k={k}, n={n}, alpha={alpha}: expected ({ci.low:.6f}, {ci.high:.6f}), got ({low:.6f}, {high:.6f})"
            )
    low, high = clopper_pearson(0, 44)
    assert low == 0.0 and 0.07 < high < 0.09, "Zero successes out of 44 should give an upper bound around 0.08"
    print("All tests in `test_clopper_pearson` passed!")


def test_claim_verdict(claim_verdict: Callable):
    assert claim_verdict(0.92, 1.0, 0.9) == "supported"
    assert claim_verdict(0.0, 0.08, 0.9) == "ruled out"
    assert claim_verdict(0.85, 0.99, 0.9) == "inconclusive"
    assert claim_verdict(0.0, 0.0, 0.0) == "ruled out", "An interval exactly at the threshold doesn't exceed it"
    assert claim_verdict(0.01, 0.2, 0.0) == "supported"
    assert claim_verdict(-0.1, 0.2, 0.0) == "inconclusive"
    assert claim_verdict(-0.3, -0.1, 0.0) == "ruled out"
    print("All tests in `test_claim_verdict` passed!")


def test_paired_bootstrap(paired_bootstrap: Callable):
    rng = np.random.default_rng(1)
    method = (rng.random(40) < 0.7).astype(float)
    control = (rng.random(40) < 0.4).astype(float)
    mean, low, high = paired_bootstrap(method, control, n_resamples=500, seed=7)
    # Oracle: the recipe from the docstring
    diffs = method - control
    idx = np.random.default_rng(7).integers(0, 40, size=(500, 40))
    boot = diffs[idx].mean(axis=1)
    np.testing.assert_allclose([mean, low, high], [diffs.mean(), *np.quantile(boot, [0.025, 0.975])])
    # Independent sanity check: close to the normal-theory interval for a moderate sample
    se = diffs.std(ddof=1) / np.sqrt(len(diffs))
    assert abs(low - (mean - 1.96 * se)) < 0.06 and abs(high - (mean + 1.96 * se)) < 0.06, (
        "The interval is far from the normal approximation; are you resampling pairs with replacement?"
    )
    # Pairing matters: identical scores give a zero-width interval at zero, even when the scores vary
    assert paired_bootstrap(method, method, n_resamples=200, seed=0) == (0.0, 0.0, 0.0)
    # Swapping method and control flips the interval
    m2, l2, h2 = paired_bootstrap(control, method, n_resamples=500, seed=7)
    assert abs(m2 + mean) < 1e-12 and abs(l2 + high) < 1e-12 and abs(h2 + low) < 1e-12
    print("All tests in `test_paired_bootstrap` passed!")


def test_build_controls(build_controls: Callable, mean_difference_direction: Callable):
    t.manual_seed(0)
    h = t.randn(30, 10, dtype=t.float64)
    y = t.tensor([0, 1] * 15)
    controls = build_controls(h, y, coordinate_counts=[1, 3], n_random=4, seed=5)
    assert set(controls) == {"shuffled_direction", "random_directions", "random_coordinates"}
    # Oracle: the recipe from the docstring
    generator = t.Generator().manual_seed(5)
    permutation = t.randperm(30, generator=generator)
    shuffled = mean_difference_direction(h, y[permutation])
    random_directions = t.randn(4, 10, generator=generator, dtype=t.float64)
    random_directions = random_directions / random_directions.norm(dim=-1, keepdim=True)
    random_coordinates = {k: [t.randperm(10, generator=generator)[:k] for _ in range(4)] for k in [1, 3]}
    t.testing.assert_close(controls["shuffled_direction"], shuffled, msg="Shuffled direction doesn't match the recipe")
    t.testing.assert_close(controls["random_directions"], random_directions, msg="Random directions don't match")
    assert set(controls["random_coordinates"]) == {1, 3}
    for k in [1, 3]:
        assert len(controls["random_coordinates"][k]) == 4, "Expected n_random coordinate sets for each k"
        for got, expected in zip(controls["random_coordinates"][k], random_coordinates[k]):
            assert got.tolist() == expected.tolist(), f"Random coordinate sets for k={k} don't match the recipe"
            assert len(set(got.tolist())) == k, "Coordinates within a set should be distinct"
    print("All tests in `test_build_controls` passed!")


def test_evaluate_interventions(evaluate_interventions: Callable):
    head = _linear_head([[0.0, 0.0], [2.0, 1.0]], [0.0, 0.0])
    # Labels deliberately interleaved and unbalanced: pairs are (class-0 #0, class-1 #0), (class-0 #1, class-1 #1)
    h = t.tensor([[1.0, 1.0], [-1.0, -1.0], [-2.0, -1.0], [1.0, 3.0], [-5.0, -5.0]], dtype=t.float64)
    y = t.tensor([1, 0, 0, 1, 0])

    def coordinate(j):
        def fn(r, d):
            out = r.clone()
            out[:, j] = d[:, j]
            return out

        return fn

    interventions = {
        "none": [lambda r, d: r],
        "first": [coordinate(0)],
        "both draws": [coordinate(0), coordinate(1)],
    }
    results = evaluate_interventions(head, h, y, interventions)
    assert set(results) == set(interventions), "Return one entry per intervention"
    for name, n_draws in [("none", 1), ("first", 1), ("both draws", 2)]:
        for key in ["swap", "recovered"]:
            assert results[name][key].shape == (n_draws, 2), (
                f"results[{name!r}][{key!r}] should have shape (n_draws, n_pairs) = {(n_draws, 2)}, "
                f"got {tuple(np.shape(results[name][key]))}"
            )
    # Same hand-worked numbers as in test_evaluate_patch (the pairs are the same two pairs)
    np.testing.assert_allclose(results["first"]["swap"], [[1.0, 1.0]])
    np.testing.assert_allclose(results["first"]["recovered"], [[4 / 6, 6 / 10]])
    np.testing.assert_allclose(results["both draws"]["swap"], [[1.0, 1.0], [0.0, 0.0]])
    np.testing.assert_allclose(results["both draws"]["recovered"], [[4 / 6, 6 / 10], [2 / 6, 4 / 10]])
    np.testing.assert_allclose(results["none"]["swap"], [[0.0, 0.0]])
    print("All tests in `test_evaluate_interventions` passed!")


def test_select_on_development(select_on_development: Callable):
    assert select_on_development({"a": 0.2, "b": 0.9, "c": 0.5}, ["a", "b", "c"]) == "b"
    assert select_on_development({"a": 0.5, "b": 0.5}, ["b", "a"]) == "b", "Ties go to the first registered candidate"
    assert select_on_development({"a": 0.5, "b": 0.5}, ["a", "b"]) == "a", "Ties go to the first registered candidate"
    assert _raises(lambda: select_on_development({"a": 0.5}, ["a", "b"]), ValueError, KeyError), (
        "A registered candidate with no development score should raise an error"
    )
    assert _raises(lambda: select_on_development({"a": 0.5, "b": 0.4, "z": 1.0}, ["a", "b"]), ValueError), (
        "A score for a method that wasn't registered as a candidate should raise a ValueError"
    )
    print("All tests in `test_select_on_development` passed!")


def test_assess_methodology(assess_methodology: Callable):
    checks = {"plan unchanged": True, "splits disjoint": True, "controls run": True}
    assert assess_methodology(checks) == [], "All checks pass, so nothing should be reported as failed"
    assert assess_methodology({**checks, "controls run": False}) == ["controls run"]
    assert assess_methodology({**checks, "plan unchanged": False, "controls run": False}) == [
        "plan unchanged",
        "controls run",
    ], "Return the failed checks in the order they were given"
    assert assess_methodology({"a": np.bool_(True), "b": np.bool_(False)}) == ["b"], "Accept numpy booleans too"
    assert _raises(lambda: assess_methodology({**checks, "controls run": "yes"}), TypeError), (
        "A check that isn't a boolean (e.g. the string 'yes') should raise a TypeError"
    )
    assert _raises(lambda: assess_methodology({**checks, "controls run": 0.97}), TypeError), (
        "A check that isn't a boolean (e.g. a score) should raise a TypeError"
    )
    print("All tests in `test_assess_methodology` passed!")


def test_logit_contributions(logit_contributions: Callable):
    t.manual_seed(0)
    h = t.randn(40, 6, dtype=t.float64)
    y = t.tensor([0, 1] * 20)
    h[y == 1, 2] += 3.0
    head = nn.Linear(6, 2).double()
    readout = head.weight[1] - head.weight[0]
    shares = logit_contributions(h, y, readout)
    assert shares.shape == (6,), f"Expected one share per unit, got shape {tuple(shares.shape)}"
    t.testing.assert_close(shares.sum(), t.tensor(1.0, dtype=t.float64), msg="Shares should sum to 1")
    # Oracle: total = difference in mean logit difference between the classes (the head is linear)
    with t.no_grad():
        logits = head(h)
        ld = logits[:, 1] - logits[:, 0]
        total = ld[y == 1].mean() - ld[y == 0].mean()
        diff = h[y == 1].mean(0) - h[y == 0].mean(0)
        t.testing.assert_close(shares.detach() * total, readout.detach() * diff)
    print("All tests in `test_logit_contributions` passed!")


def test_principal_directions(principal_directions: Callable):
    from sklearn.decomposition import PCA

    t.manual_seed(0)
    h = t.randn(60, 5, dtype=t.float64) @ t.randn(5, 5, dtype=t.float64) + 4.0
    components, variance_ratio = principal_directions(h)
    pca = PCA().fit(h.numpy())
    np.testing.assert_allclose(variance_ratio.numpy(), pca.explained_variance_ratio_, atol=1e-10)
    # Components are only defined up to sign
    np.testing.assert_allclose(np.abs(components.numpy() @ pca.components_.T), np.eye(5), atol=1e-8)
    print("All tests in `test_principal_directions` passed!")
