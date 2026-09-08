"""Untouched audit, deterministic ordering and label-control regression checks."""

import numpy as np
import pytest

from chapter16_shapley_attribution_baselines.exercises.part18_spex import native_data, native_query_plan, spex_methods
from chapter16_shapley_attribution_baselines.exercises.part19_proxyspex import proxy_methods


@pytest.fixture(scope="module")
def plan():
    return native_query_plan.make_plan(spex_methods, proxy_methods)


def test_audit_is_disjoint_from_entire_planned_training_union(plan):
    metadata, arrays = plan
    audit = set(map(int, arrays["audit_ids"]))
    union = set(map(int, arrays["training_union_ids"]))
    assert len(audit) == 512
    assert not audit.intersection(union | {0, 2**28 - 1})
    all_training = set()
    for condition in metadata["conditions"]:
        all_training.update(map(int, arrays[condition["key"] + "_ids"]))
    assert all_training == union
    assert metadata["primary_training_and_audit_rows"] == 32 * (len(union) + 512)
    assert metadata["native_queries"] == metadata["native_proxy_fits"] == 0


def test_proxy_prefixes_folds_and_shuffle_do_not_leak(plan):
    metadata, arrays = plan
    for seed in (0, 1, 2):
        largest = arrays[f"proxyspex_s{seed}_k1824_masks"]
        for budget in (456, 912, 1824):
            key = f"proxyspex_s{seed}_k{budget}"
            masks = arrays[key + "_masks"]
            np.testing.assert_array_equal(masks, largest[:budget])
            assert len(np.unique(masks, axis=0)) == budget
            assert not masks[0].any() and masks[1].all()
            validations = []
            for fold in range(3):
                train = arrays[f"{key}_fold{fold}_train"]
                valid = arrays[f"{key}_fold{fold}_validation"]
                assert not set(train).intersection(valid)
                assert sorted(np.concatenate([train, valid]).tolist()) == list(range(budget))
                validations.extend(valid)
            assert sorted(validations) == list(range(budget))
    for condition in metadata["conditions"]:
        key = condition["key"]
        labels = arrays[key + "_ids"].astype(float)
        unique = labels[arrays[key + "_unique_first"]]
        shuffled = unique[arrays[key + "_shuffle"]][arrays[key + "_inverse"]]
        for identity in np.unique(arrays[key + "_ids"]):
            assert np.unique(shuffled[arrays[key + "_ids"] == identity]).size == 1
        assert sorted(arrays[key + "_shuffle"]) == list(range(len(unique)))


def test_saved_plan_binds_values_order_and_dtype(tmp_path):
    target = tmp_path / "freeze"
    native_query_plan.write_plan(target, spex_methods, proxy_methods)
    metadata, arrays = native_query_plan.validate_student_plan(target, spex_methods, proxy_methods)
    assert len(metadata["conditions"]) == 18
    assert all(native_data.mask_ids(arrays[c["key"] + "_masks"]).shape == (c["budget"],)
               for c in metadata["conditions"])
    path = target / "query_plan.npz"
    path.write_bytes(path.read_bytes()[:-1] + b"!")
    with pytest.raises(ValueError, match="archive changed"):
        native_query_plan.load_plan(target)
