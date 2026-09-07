"""Independent checks for image/counterfactual pairing and word-label boundaries."""
from __future__ import annotations

import torch as t

from . import language_data as data


def test_language_splits_separate_combinations_and_renderer_draws():
    splits = {name: data.make_language_dataset(name, examples_per_combination=3)
              for name in data.SPLIT_SEEDS}
    for name, dataset in splits.items():
        combinations = {(scene.color, scene.shape) for scene in dataset.scenes}
        expected = (data.HELDOUT_COMBINATIONS if name == "compositional" else
                    {(color, shape) for color in data.COLORS for shape in data.SHAPES} - data.HELDOUT_COMBINATIONS)
        assert combinations == expected
        assert len(dataset.scenes) == 3 * len(expected)
        assert dataset.images.shape == (len(dataset.scenes), 3, 32, 32)
    train = splits["train"]
    development = splits["development"]
    assert not set(scene.scene_id for scene in train.scenes) & set(scene.scene_id for scene in development.scenes)
    assert not t.equal(train.images, development.images)
    assert {scene.style for scene in splits["familiar"].scenes} == {"muted"}
    assert {scene.style for scene in train.scenes} == {"solid"}


def test_matched_counterfactual_preserves_unaffected_pixels():
    scene = data.LanguageScene("probe", "red", "square", 16, 15, 6, 235, (1, -2, 3), 91, "solid")
    clean, object_mask = data.render_language_scene(scene)
    changed = data.counterfactual_scene(scene, color="blue", shape="circle")
    corrupt, donor_mask = data.render_language_scene(changed)
    union = object_mask | donor_mask
    t.testing.assert_close(clean[:, ~union], corrupt[:, ~union], atol=0, rtol=0)
    assert float((clean[:, union] - corrupt[:, union]).abs().max()) > .5
    assert changed.bbox == scene.bbox == (10, 9, 23, 22)
    assert object_mask.sum() == 13 * 13
    assert donor_mask.sum() < object_mask.sum()
    repeated, repeated_mask = data.render_language_scene(scene)
    t.testing.assert_close(repeated, clean, atol=0, rtol=0)
    assert t.equal(repeated_mask, object_mask)


def test_word_labels_mark_answers_eos_and_padding_explicitly():
    scenes = data.make_language_dataset("train", examples_per_combination=1).scenes[:2]
    tokens = data.tokenize_language_examples(scenes, ("color", "caption"))
    decoded = [[data.WORDS[index] for index in row[valid].tolist()]
               for row, valid in zip(tokens["token_ids"], tokens["valid_mask"], strict=True)]
    assert decoded[0] == ["<bos>", "what", "color", "is", "the", "object", "?", "<answer>", scenes[0].color, "<eos>"]
    assert decoded[1] == ["<bos>", "describe", "the", "image", ".", "<answer>", "a", scenes[1].color, scenes[1].shape, ".", "<eos>"]
    assert tokens["prompt_lengths"].tolist() == [8, 6]
    assert tokens["target_mask"].sum(1).tolist() == [2, 5]
    assert not tokens["target_mask"][~tokens["valid_mask"]].any()
    assert (tokens["token_ids"][~tokens["valid_mask"]] == data.VOCAB["<pad>"]).all()


def test_generation_prompts_exclude_answers_and_bad_specials_stay_visible():
    scenes = data.make_language_dataset("train", examples_per_combination=1).scenes[:2]
    tokens = data.tokenize_language_examples(scenes, ("color", "shape"), include_answers=False)
    assert not tokens["target_mask"].any()
    for row, valid in zip(tokens["token_ids"], tokens["valid_mask"], strict=True):
        assert row[valid][-1] == data.VOCAB["<answer>"]
        assert data.VOCAB["<eos>"] not in row[valid]
        assert not set(row[valid].tolist()) & {data.VOCAB[word] for word in (*data.COLORS, *data.SHAPES)}
    assert data.decode_answer([data.VOCAB["red"], data.VOCAB["<eos>"], data.VOCAB["blue"]]) == "red"
    assert data.decode_answer([data.VOCAB["<image>"], data.VOCAB["red"]]) == "<image> red"
