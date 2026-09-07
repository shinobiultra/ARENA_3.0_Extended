"""Generation scoring and two independent AdamW updates, without a long fit."""
from __future__ import annotations

import torch as t
from torch import nn

from . import language_training as reference
from .language_data import VOCAB, make_language_dataset
from .language_vlm import WordGeneration, WordMiniVLM


def test_generation_metrics_require_exact_words_and_eos(fn=None):
    fn = fn or reference.generated_word_exact_match
    expected = t.tensor([[VOCAB["red"], VOCAB["<eos>"], 0, 0]] * 4)
    actual = expected.clone()
    actual[2] = t.tensor([VOCAB["<image>"], VOCAB["red"], VOCAB["<eos>"], 0])
    generation = WordGeneration(actual, t.tensor([2, 2, 3, 3]), t.tensor([True, False, True, True]))
    assert fn(generation, expected).tolist() == [True, False, False, False]


def test_two_word_training_updates_match_manual_adamw(fn=None):
    fn = fn or reference.train_word_model
    train = make_language_dataset("train", examples_per_combination=1)
    development = make_language_dataset("development", examples_per_combination=1)
    configuration = {"patch_size": 8, "vision_width": 8, "width": 8,
                     "heads": 2, "vision_layers": 1, "decoder_layers": 1}
    seed, learning_rate = 26, .002
    actual = fn(train, development, steps=2, batch_size=3, seed=seed,
                learning_rate=learning_rate, evaluation_every=2, model_kwargs=configuration)

    t.manual_seed(seed)
    expected = WordMiniVLM(**configuration)
    batch = reference.make_word_task_batch(train)
    generator = t.Generator().manual_seed(seed + 1000)
    parameters = tuple(expected.parameters())
    first_moments = [t.zeros_like(parameter) for parameter in parameters]
    second_moments = [t.zeros_like(parameter) for parameter in parameters]
    for step in (1, 2):
        selected = t.randint(len(batch["images"]), (3,), generator=generator)
        logits = expected(batch["images"][selected], batch["token_ids"][selected], batch["valid_mask"][selected])
        ids, targets = batch["token_ids"][selected], batch["target_mask"][selected]
        # Enumerate shifted target words independently from the taught loss helper.
        losses = [-logits[row, position - 1].log_softmax(-1)[ids[row, position]]
                  for row in range(3) for position in range(1, ids.shape[1]) if targets[row, position]]
        gradients = t.autograd.grad(t.stack(losses).mean(), parameters)
        norm = t.stack([gradient.norm() for gradient in gradients]).norm()
        multiplier = min(1., float(1 / (norm + 1e-6)))
        with t.no_grad():
            for parameter, gradient, first, second in zip(parameters, gradients, first_moments, second_moments, strict=True):
                gradient = gradient * multiplier
                first.mul_(.9).add_(gradient, alpha=.1)
                second.mul_(.999).add_(gradient.square(), alpha=.001)
                first_corrected = first / (1 - .9**step)
                second_corrected = second / (1 - .999**step)
                parameter.mul_(1 - learning_rate * 1e-4)
                parameter.add_(-learning_rate * first_corrected / (second_corrected.sqrt() + 1e-8))
    assert actual.selected_step == 2
    for (actual_name, actual_parameter), (expected_name, expected_parameter) in zip(
        actual.model.named_parameters(), expected.named_parameters(), strict=True,
    ):
        assert actual_name == expected_name
        t.testing.assert_close(actual_parameter, expected_parameter, atol=1e-6, rtol=1e-5)
    assert all(actual.changed_parameter_groups.values())


def test_image_only_inputs_keep_targets_but_remove_question_words():
    dataset = make_language_dataset("train", examples_per_combination=1)
    full = reference.make_word_task_batch(dataset)
    ablated = reference.make_word_task_batch(dataset, omit_question=True)
    t.testing.assert_close(full["expected_tokens"], ablated["expected_tokens"])
    assert ablated["prompt_lengths"].unique().tolist() == [2]
    assert (ablated["token_ids"][:, :2] == t.tensor([VOCAB["<bos>"], VOCAB["<answer>"]])).all()


def test_word_evaluation_generates_without_answer_leakage(fn=None):
    """An independent scripted decoder exposes both prompt leakage and wrong scoring."""
    from dataclasses import replace
    fn = fn or reference.evaluate_word_model
    dataset = make_language_dataset("train", examples_per_combination=1)
    dataset = replace(dataset, scenes=dataset.scenes[:2], images=dataset.images[:2], object_masks=dataset.object_masks[:2])
    batch = reference.make_word_task_batch(dataset)

    class AlwaysRedSquare(nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(t.zeros(()))

        def forward(self, pixels, ids, valid):
            logits = t.full((*ids.shape, len(VOCAB)), -20.)
            for row in range(len(ids)):
                prefix = ids[row, valid[row]].tolist()
                marker = prefix.index(VOCAB["<answer>"])
                task = "caption" if VOCAB["describe"] in prefix[:marker] else "color" if VOCAB["color"] in prefix[:marker] else "shape"
                answer = (["a", "red", "square", ".", "<eos>"] if task == "caption" else
                          ["red" if task == "color" else "square", "<eos>"])
                step = len(prefix) - marker - 1
                # Valid appended words must be exactly this decoder's earlier outputs.
                assert prefix[marker + 1:] == [VOCAB[word] for word in answer[:step]], "Target answer leaked into generation."
                next_word = answer[min(step, len(answer) - 1)]
                logits[row, len(prefix) - 1, VOCAB[next_word]] = 20.
            return logits

    measured = fn(AlwaysRedSquare(), batch, batch_size=4)
    assert measured["per_example_exact_match"] == [True, True, True, True, False, False]
    assert measured["color_exact_match"] == 1.
    assert measured["shape_exact_match"] == measured["caption_exact_match"] == .5
