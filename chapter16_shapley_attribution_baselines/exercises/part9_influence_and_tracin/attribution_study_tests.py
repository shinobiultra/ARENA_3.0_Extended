"""Checks for measurement semantics and the ranking/final evaluation boundary."""
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from chapter16_shapley_attribution_baselines.exercises.part9_influence_and_tracin import attribution_study as study


def test_query_outputs_are_actual_classifier_outputs():
    theta = torch.tensor([[1., -.5, .2], [-.3, .7, -.1]], dtype=torch.float64)
    queries = {"x": torch.tensor([[1., 0., 1.], [0., 2., 1.], [-1., .5, 1.]], dtype=torch.float64),
               "y": torch.tensor([1, 0, 1])}
    actual = study.query_outputs(theta, queries)
    for i, parameters in enumerate(theta):
        logits = queries["x"] @ parameters
        expected_loss = F.binary_cross_entropy_with_logits(logits, queries["y"].double(), reduction="none")
        torch.testing.assert_close(actual["losses"][i], expected_loss)
        torch.testing.assert_close(actual["probabilities"][i], logits.sigmoid())
        assert torch.equal(actual["correct"][i], (logits >= 0) == queries["y"].bool())


def test_pixel_control_uses_images_and_observed_labels():
    train = {"x": torch.tensor([[1., 0., 1.], [0., 1., 1.]], dtype=torch.float64)}
    query = {"x": torch.tensor([[1., 0., 1.], [0., 1., 1.]], dtype=torch.float64), "y": torch.tensor([0, 1])}
    actual = study.signed_pixel_similarity(train, torch.tensor([1, 1]), query)
    torch.testing.assert_close(actual, torch.tensor([[-1., 0.], [0., 1.]], dtype=torch.float64))
    # Making the bias coordinate enormous cannot create image resemblance.
    train["x"][:, -1] = 1000
    query["x"][:, -1] = 1000
    torch.testing.assert_close(study.signed_pixel_similarity(train, torch.tensor([1, 1]), query), actual)


def test_fit_phase_cannot_read_final_queries(tmp_path, monkeypatch):
    class RestrictedData(dict):
        def __getitem__(self, key):
            assert key != "final", "The fitting/selection phase must not access final queries."
            return super().__getitem__(key)
    train = {"y": torch.tensor([0, 1]), "corrupted_y": torch.tensor([1, 0])}
    ranking = {"ids": ["ranking:0"]}
    data = RestrictedData(train=train, ranking=ranking, upweight_probe_indices=[0])
    seen = []
    def fit_condition(given_train, given_ranking, indices, labels, m, directory, progress):
        assert given_train is train and given_ranking is ranking
        assert indices == [0]
        seen.append(labels.clone())
        return {"stage": "FITS_AND_RANKING_COMPLETE"}, {"query_ids": ranking["ids"]}
    monkeypatch.setattr(study, "fit_condition", fit_condition)
    fitted, measured = study.prepare_study(data, object(), tmp_path)
    assert list(fitted) == ["clean", "corrupted"] and list(measured) == ["clean", "corrupted"]
    assert torch.equal(seen[0], train["y"]) and torch.equal(seen[1], train["corrupted_y"])


def test_final_requires_both_finished_conditions_and_never_refits(tmp_path, monkeypatch):
    queries = {"x": torch.tensor([[1., 1.]], dtype=torch.float64), "y": torch.tensor([1]), "ids": ["final:0"]}
    fit = {"stage": "FITS_AND_RANKING_COMPLETE", "train_y": torch.tensor([0, 1]),
           "deletions": {"theta": torch.tensor([[.2, -.1]], dtype=torch.float64)}}
    with pytest.raises(ValueError, match="Both"):
        study.evaluate_final({"clean": fit}, {}, queries, object(), tmp_path)
    with pytest.raises(ValueError, match="frozen"):
        study.evaluate_final({"clean": fit, "corrupted": {**fit, "stage": "RUNNING"}}, {}, queries, object(), tmp_path)
    calls = []
    def measured(given_fit, train, labels, given_queries, methods):
        assert given_fit is fit and given_queries is queries
        calls.append("scored")
        return {"query_ids": queries["ids"]}
    class NoFitting:
        def fit_weighted_logistic(self, *args, **kwargs):
            raise AssertionError("Final evaluation must not refit or select parameters.")
        def class_matched_removals(self, *args, **kwargs):
            raise AssertionError("Final evaluation must not select deletion policies.")
    monkeypatch.setattr(study, "score_queries", measured)
    for name in ("clean", "corrupted"):
        (tmp_path / name).mkdir()
    result = study.evaluate_final({"clean": fit, "corrupted": fit}, {}, queries, NoFitting(), tmp_path)
    assert len(calls) == 2
    assert all(result[name]["query_ids"] == ["final:0"] for name in result)
