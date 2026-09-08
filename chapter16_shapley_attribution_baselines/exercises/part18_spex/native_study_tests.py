"""Scientific persistence and fit/audit checks without language-model outcomes."""

import numpy as np
import pytest
from types import SimpleNamespace
import hashlib
import json

from chapter16_shapley_attribution_baselines.exercises.part18_spex import native_query_plan, native_study, spex_methods
from chapter16_shapley_attribution_baselines.exercises.part19_proxyspex import proxy_methods


def test_environment_requires_bound_bytes_and_actual_running_versions(tmp_path, monkeypatch):
    snapshot = {"python": "fixed", "executable": "fixed-python", "versions": {"numpy": "fixed-version"}}
    path = tmp_path / "environment.json"
    path.write_text(json.dumps(snapshot))
    record = {"main_environment_path": str(path), "main_environment_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    monkeypatch.setattr(native_study, "environment_snapshot", lambda: snapshot)
    native_study.validate_environment(record)
    monkeypatch.setattr(native_study, "environment_snapshot", lambda: {**snapshot, "versions": {"numpy": "changed"}})
    with pytest.raises(ValueError, match="actual Python"):
        native_study.validate_environment(record)
    path.write_text("{}")
    with pytest.raises(ValueError, match="environment file changed"):
        native_study.validate_environment(record)


def test_lossless_record_retains_supports_array_order_and_types(tmp_path):
    record = {"spectrum": {(): 0.3, (2, 4, 9): -1.25}, "signed": np.array([3, -2], dtype=np.float64),
              "ids": np.array([7, 2, 7], dtype=np.uint64), "nested": [(1, np.array([[1, 0]], dtype=np.uint8))]}
    path = tmp_path / "condition"
    expected = native_study.save_record(path, record)
    loaded = native_study.load_record(path, expected_hashes=expected)
    assert loaded["spectrum"] == record["spectrum"]
    for key in ("ids", "signed"):
        np.testing.assert_array_equal(loaded[key], record[key])
        assert loaded[key].dtype == record[key].dtype
    np.testing.assert_array_equal(loaded["nested"][0][1], record["nested"][0][1])
    revised = tmp_path / "revised"
    native_study.save_record(revised, {**record, "spectrum": {(1,): 99.0}})
    with pytest.raises(ValueError, match="pre-audit or summary"):
        native_study.load_record(revised, expected_hashes=expected)
    (path / "arrays.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="record changed"):
        native_study.load_record(path)


def test_query_matched_baselines_and_constant_audit():
    train = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [1, 1]], dtype=np.uint8)
    values = 2.5 - 0.7 * train[:, 0] + 1.25 * train[:, 1]
    predictions, receipt = native_study.baseline_predictions(train, values, train)
    np.testing.assert_allclose(predictions["additive"], values, atol=1e-14)
    assert predictions["constant"][0] == pytest.approx(values[:4].mean())
    assert receipt["rank"] == 3 and receipt["unique_training_masks"] == 4
    metrics = native_study.evaluate_predictions(np.ones(3), {"good": np.ones(3), "bad": np.zeros(3)})
    assert metrics["good"] == {"rmse": 0.0, "nrmse": None, "r2": None}
    assert metrics["bad"] == {"rmse": 1.0, "nrmse": None, "r2": None}


@pytest.mark.parametrize("prediction", [np.zeros((512, 1)), np.zeros(1), np.full(512, np.nan)])
def test_evaluation_rejects_broadcasting_and_nonfinite_predictions(prediction):
    with pytest.raises(ValueError, match="exactly equal shapes and finite"):
        native_study.evaluate_predictions(np.zeros(512), {"bad": prediction})


def test_missing_proxy_folds_cannot_skip_frozen_partition_checks():
    metadata, arrays = native_query_plan.make_plan(spex_methods, proxy_methods)
    condition = next(row for row in metadata["conditions"] if row["method"] == "proxyspex")
    incomplete = SimpleNamespace(fit_proxy=lambda *args, **kwargs: {"folds": []})
    with pytest.raises(ValueError, match="Exactly three"):
        native_study.fit_condition(condition, arrays, np.zeros(condition["budget"]),
                                   spex_methods=spex_methods, proxy_methods=incomplete)


@pytest.mark.parametrize("method", ["spex", "proxyspex"])
def test_fits_predictions_without_audit_values_and_retains_controls(method):
    metadata, arrays = native_query_plan.make_plan(spex_methods, proxy_methods)
    condition = next(row for row in metadata["conditions"] if row["method"] == method)
    masks = arrays[condition["key"] + "_masks"]
    values = spex_methods.predict_spectrum(masks, {(): .3, (0,): .7, (2, 5): -1.2})
    records = native_study.fit_condition(condition, arrays, values, spex_methods=spex_methods, proxy_methods=proxy_methods)
    assert [row["condition"] for row in records] == ["real", "shuffled"]
    assert all("audit_values" not in row for row in records)
    assert all(row["baseline_fit"]["fits"] == 2 for row in records)
    assert all(set(row["baseline_predictions"]) == {"constant", "additive"} for row in records)
    assert all(np.isfinite(prediction).all() and prediction.shape == (512,)
               for row in records for prediction in row["predictions"].values())
    if method == "spex":
        repeated_id = next(identity for identity in np.unique(arrays[condition["key"] + "_ids"])
                           if np.count_nonzero(arrays[condition["key"] + "_ids"] == identity) > 1)
        bad_values = values.copy()
        bad_values[np.flatnonzero(arrays[condition["key"] + "_ids"] == repeated_id)[-1]] += 1
        with pytest.raises(ValueError, match="inconsistent labels"):
            native_study.fit_condition(condition, arrays, bad_values, spex_methods=spex_methods, proxy_methods=proxy_methods)
