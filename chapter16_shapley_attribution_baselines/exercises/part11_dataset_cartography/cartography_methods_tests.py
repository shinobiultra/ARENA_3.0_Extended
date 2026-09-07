"""Small independent oracles and immediate feedback for the learner's functions."""

import numpy as np
import pytest
import torch

from chapter16_shapley_attribution_baselines.exercises.part11_dataset_cartography import (
    cartography_methods as answers,
)
from chapter16_shapley_attribution_baselines.exercises.part11_dataset_cartography.cartography_reference import (
    standard_training,
)


def test_align_epoch(function=answers.align_epoch):
    ids = torch.tensor([[2, 0, 1], [1, 2, 0]])
    logits = torch.stack([ids.float(), -ids.float()], dim=-1)
    expected = torch.tensor([[[0.0, 0.0], [1.0, -1.0], [2.0, -2.0]]] * 2)
    torch.testing.assert_close(function(logits, ids), expected)
    for broken in (torch.tensor([[0, 0, 2], [0, 1, 2]]), ids + 1):
        with pytest.raises(ValueError, match="ID"):
            function(logits, broken)
    print("align_epoch passed: source IDs survive shuffling; duplicate and missing visits fail.")


def test_gold_probabilities(function=answers.gold_probabilities):
    logits = torch.tensor(
        [[[[1.0, 2.0, 3.0], [3.0, 1.0, -2.0]]], [[[2.0, 2.0, 2.0], [-9.0, 4.0, 0.0]]]],
        dtype=torch.float64,
    )
    labels = torch.tensor([[0, 2]])
    values = logits.numpy()
    exponential = np.exp(values - values.max(axis=-1, keepdims=True))
    probability = exponential / exponential.sum(axis=-1, keepdims=True)
    expected = np.array([[[probability[e, 0, 0, 0], probability[e, 0, 1, 2]]] for e in range(2)])
    torch.testing.assert_close(function(logits, labels), torch.from_numpy(expected))
    torch.testing.assert_close(function(logits + 1000, labels), torch.from_numpy(expected))
    print(
        "gold_probabilities passed: assigned labels and stable softmax, including wrong predictions."
    )


def test_training_statistics(function=answers.training_statistics):
    probabilities = torch.tensor(
        [[0.1, 0.8, 0.5], [0.5, 0.8, 0.5], [0.9, 0.8, 0.5]], dtype=torch.float64
    )
    logits = torch.stack([probabilities.log(), (1 - probabilities).log()], dim=-1)[:, None]
    labels = torch.zeros(1, 3, dtype=torch.long)
    result = function(logits, labels)
    torch.testing.assert_close(result["confidence"], probabilities.mean(0)[None])
    torch.testing.assert_close(
        result["variability"], torch.from_numpy(np.std(probabilities.numpy(), axis=0))[None]
    )
    torch.testing.assert_close(result["correctness"], torch.tensor([[2 / 3, 1.0, 1.0]]))
    single = function(logits[-1:], labels)
    assert (
        torch.isfinite(single["variability"]).all() and not single["variability"].any()
    ), "A single snapshot has zero population SD; do not use correction=1."
    assert torch.all(
        result["variability"] ** 2 <= result["confidence"] * (1 - result["confidence"]) + 1e-12
    ), "Probabilities lie in [0,1], so variance cannot exceed mean times one minus mean."
    print("training_statistics passed: epoch axis, population SD, correctness fraction and ties.")


def test_batched_logits(function=answers.batched_logits):
    g = torch.Generator().manual_seed(6)
    for width in (0, 5):
        parameters = answers.initialize_parameters(
            4, 3, width, [torch.Generator().manual_seed(s) for s in (8, 9)]
        )
        x = torch.randn(2, 7, 4, generator=g)
        expected = []
        for b in range(2):
            p = {k: v[b].detach().numpy() for k, v in parameters.items()}
            row = x[b].numpy()
            if width:
                row = np.maximum(row @ p["w1"] + p["b1"], 0) @ p["w2"] + p["b2"]
            else:
                row = row @ p["w"] + p["b"]
            expected.append(row)
        torch.testing.assert_close(
            function(parameters, x), torch.from_numpy(np.stack(expected)), rtol=2e-6, atol=2e-6
        )
    print("batched_logits passed: independent linear and nonlinear networks match NumPy.")


def test_train_ensemble(function=answers.train_ensemble):
    g = torch.Generator().manual_seed(40)
    x = torch.randn(2, 17, 4, generator=g)
    labels = torch.randint(3, (2, 17), generator=g)
    for width in (0, 5):
        actual = function(x, labels, [12, 13], width=width, epochs=3, batch_size=6, classes=3)
        assert actual["steps"] == 9, "Include the final partial minibatch in every epoch."
        for b, seed in enumerate((12, 13)):
            expected = standard_training(x[b], labels[b], seed, width, 3, 6, 3)
            for name in ("online_logits", "end_logits", "loss"):
                torch.testing.assert_close(actual[name][:, b], expected[name], rtol=2e-5, atol=2e-5)
            for name, value in expected["parameters"].items():
                torch.testing.assert_close(
                    actual["parameters"][name][b], value, rtol=2e-5, atol=2e-5
                )
        assert not torch.equal(
            actual["online_logits"], actual["end_logits"]
        ), "Pre-update visited logits and common epoch-end logits are different observations."
    print(
        "train_ensemble passed: ordinary torch.nn/SGD oracle checks both seeds, architectures, snapshots and gradients."
    )


def test_select_indices(function=answers.select_indices):
    scores = torch.tensor([0.2, 0.7, 0.7, -0.1])
    assert function(scores, 3).tolist() == [
        1,
        2,
        0,
    ], "Descending ties must retain the earlier source ID."
    assert function(scores, 2, largest=False).tolist() == [
        3,
        0,
    ], "Hard examples have low assigned-label confidence."
    with pytest.raises(ValueError):
        function(scores, 5)
    print("select_indices passed: direction, retained size and stable ties.")


def test_matched_random_indices(function=answers.matched_random_indices):
    labels = torch.tensor([0] * 10 + [1] * 7 + [2] * 5)
    selected = torch.tensor([0, 1, 10, 11, 12, 17])
    for seed in range(5):
        result = function(labels, selected, seed)
        assert (
            len(result) == len(selected) == len(result.unique())
        ), "Matched sampling is without replacement."
        torch.testing.assert_close(
            torch.bincount(labels[result], minlength=3), torch.tensor([2, 3, 1])
        )
        torch.testing.assert_close(result, function(labels, selected, seed))
    assert not torch.equal(
        function(labels, selected, 1), function(labels, selected, 2)
    ), "Distinct seeds should generate fresh matched controls."
    print(
        "matched_random_indices passed: exact class counts, no duplicates and independent seeded sampling."
    )


def test_retrieval_curve(function=answers.retrieval_curve):
    result = function(
        torch.tensor([0.4, 0.1, 0.3, 0.2]),
        torch.tensor([False, True, True, False]),
        torch.tensor([1, 2, 4]),
    )
    torch.testing.assert_close(result["found"], torch.tensor([1, 1, 2]))
    torch.testing.assert_close(result["precision"], torch.tensor([1.0, 0.5, 0.5]))
    torch.testing.assert_close(result["recall"], torch.tensor([0.5, 0.5, 1.0]))
    print("retrieval_curve passed: precision and recall use different denominators.")


def test_selection_overlap(function=answers.selection_overlap):
    torch.testing.assert_close(
        function(torch.tensor([1, 2, 3]), torch.tensor([3, 4, 5])), torch.tensor(1 / 5)
    )
    torch.testing.assert_close(
        function(torch.tensor([1, 2]), torch.tensor([2, 1])), torch.tensor(1.0)
    )
    print("selection_overlap passed: unordered set intersection over union.")


def test_evaluate_models(function=answers.evaluate_models):
    parameters = dict(
        w=torch.tensor([[[1.0, -1.0], [-1.0, 1.0]], [[-1.0, 1.0], [1.0, -1.0]]]),
        b=torch.zeros(2, 1, 2),
    )
    x = torch.eye(2)
    labels = torch.tensor([0, 1])
    result = function(parameters, x, labels)
    torch.testing.assert_close(result["accuracy"], torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(result["class_accuracy"], torch.tensor([[1.0, 1.0], [0.0, 0.0]]))
    expected = torch.tensor([np.log1p(np.exp(-2)), np.log1p(np.exp(2))], dtype=torch.float32)
    torch.testing.assert_close(result["cross_entropy"], expected)
    assert result["correct"].shape == (2, 2), "Keep per-example correctness for paired comparisons."
    print(
        "evaluate_models passed: every model, every held-out example and independent logistic-loss oracle."
    )
