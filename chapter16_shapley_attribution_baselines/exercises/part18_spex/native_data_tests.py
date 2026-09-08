"""Prospective input controls; these tests never load or query model weights."""

import numpy as np
import pytest

from . import native_data as data


def test_complete_prompt_inventory():
    rows = data.prompt_records()
    assert len(rows) == 32 and len({row["prompt"] for row in rows}) == 32
    assert {(r["template_index"], r["pair_index"], r["name_order"]) for r in rows} == {
        (template, pair, order) for template in range(4) for pair in range(4) for order in (0, 1)
    }
    for row in rows:
        before, after = row["prompt"].split(), row["counterfactual_prompt"].split()
        assert len(before) == len(after) == 29
        changed = [i for i, (a, b) in enumerate(zip(before, after, strict=True)) if a != b]
        assert changed == [row["subject_word_index"]]
        assert after[changed[0]] == row["name_a"]
    print("All 32 fixed prompts and subject-only counterfactuals passed!")


def test_word_groups_hide_all_subtokens_and_keep_absolute_positions(fn=data.masked_word_inputs):
    ids = np.array([101, 102, 103, 104, 105])
    groups = np.array([0, 0, 1, 2, -1])
    masks = np.zeros((4, 28), dtype=np.uint8)
    masks[1, 0] = 1
    masks[2, 1] = 1
    masks[3] = 1
    result = fn(ids, groups, masks, replacement_id=999)
    np.testing.assert_array_equal(result["input_ids"], [
        [999, 999, 999, 999, 105], [101, 102, 999, 999, 105],
        [999, 999, 103, 999, 105], [101, 102, 103, 104, 105],
    ])
    np.testing.assert_array_equal(result["attention_mask"], [
        [0, 0, 0, 0, 1], [1, 1, 0, 0, 1], [0, 0, 1, 0, 1], [1, 1, 1, 1, 1],
    ])
    np.testing.assert_array_equal(result["position_ids"], np.tile(np.arange(5), (4, 1)))
    print("Whole-word masking, a retained suffix and unchanged positions passed!")


@pytest.mark.parametrize("bad", [np.full((1, 28), .5), np.ones((1, 27)), np.ones(28), np.full((1, 28), np.nan)])
def test_invalid_masks_fail(bad):
    with pytest.raises(ValueError):
        data.masked_word_inputs(np.array([2, 3]), np.array([0, -1]), bad)


def test_integer_mask_orientation():
    np.testing.assert_array_equal(data.mask_ids([[1, 0, 0], [0, 0, 1], [1, 1, 1]]), [4, 1, 7])
    print("Displayed feature zero is the most significant query-ID bit!")
