"""Visible study orchestration; the notebook uses the student's method functions."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor

from chapter16_shapley_attribution_baselines.exercises.part9_influence_and_tracin.attribution_methods import (
    fit_weighted_logistic,
)

FRACTIONS = (0.25, 0.5, 0.75)
BUDGETS = (2, 8, 16, 32)


def make_designs(methods) -> dict:
    """Build and check the prescribed mask splits without fitting any classifier."""
    designs = {}
    for a, fraction in enumerate(FRACTIONS):
        sizes = (("train", 1024), ("validation", 128), ("test", 256))
        design = {
            name: methods.sample_subsets(128, int(fraction * 128), count, 161000 + 100 * a + split)
            for split, (name, count) in enumerate(sizes)
        }
        together = torch.cat(list(design.values()))
        if together.unique(dim=0).shape[0] != len(together):
            raise ValueError("A duplicate subset crossed or appeared within the prescribed splits.")
        designs[fraction] = design
    return designs


def refit_masks(train: dict, masks: Tensor) -> dict[str, Tensor]:
    """Batched independent classifier fits using the optimizer taught in 16.9."""
    if masks.ndim != 2 or masks.shape[1] != len(train["x"]) or masks.dtype != torch.bool:
        raise ValueError("Provide boolean training membership masks.")
    if (masks.sum(1) == 0).any():
        raise ValueError("A model cannot be trained on an empty subset.")
    weights, gradients, objectives = [], [], []
    for batch in masks.split(128):
        fit = fit_weighted_logistic(train["x"], train["y"], batch.double(), l2=0.02)
        weights.append(fit["theta"])
        gradients.append(fit["gradient_norms"][-1])
        objectives.append(fit["objective"])
    return dict(
        theta=torch.cat(weights),
        gradient_norm=torch.cat(gradients),
        objective=torch.cat(objectives),
    )


def model_margins(fit: dict, query: dict, methods) -> Tensor:
    logit = fit["theta"] @ query["x"].T
    two_class_logits = torch.stack([torch.zeros_like(logit), logit], dim=-1)
    return methods.correct_class_margins(two_class_logits, query["y"])


def least_squares_datamodel(x: Tensor, y: Tensor) -> dict[str, Tensor]:
    """Unpenalized minimum-norm baseline; center on training rows only."""
    xc, yc = x - x.mean(0), y - y.mean(0)
    weights = torch.linalg.lstsq(xc, yc, driver="gelsd").solution
    return dict(weights=weights, bias=y.mean(0) - x.mean(0) @ weights)


def fit_response_models(x: Tensor, y: Tensor, xv: Tensor, yv: Tensor, methods) -> dict:
    """Lasso and controls see only their fit and validation subset responses."""
    penalties = torch.logspace(0, -4, 16, dtype=x.dtype)
    result = {"lasso": methods.fit_datamodel_path(x, y, xv, yv, penalties)}
    permutation = torch.randperm(len(y), generator=torch.Generator().manual_seed(161440))
    result["permuted"] = methods.fit_datamodel_path(x, y[permutation], xv, yv, penalties)
    result["permutation"] = permutation
    result["least_squares"] = least_squares_datamodel(x, y)
    result["intercept"] = y.mean(0)
    return result


def build_deletion_masks(train: dict, query: dict, responses: dict, methods) -> dict:
    """Freeze all selected and matched control sets before seeing refit outcomes."""
    query_ids = (
        torch.cat([torch.nonzero(query["y"] == label).flatten()[:12] for label in (0, 1)])
        .sort()
        .values
    )
    pixel = torch.nn.functional.normalize(train["x"][:, :64], dim=1)
    target_pixel = torch.nn.functional.normalize(query["x"][:, :64], dim=1)
    pixel_scores = (pixel @ target_pixel.T) * torch.where(
        train["y"][:, None] == query["y"][None], 1.0, -1.0
    )
    all_scores = [responses[a]["lasso"]["selected_weights"] for a in FRACTIONS] + [pixel_scores]
    masks, target, budget_values, method_values, replicate, class_counts = [], [], [], [], [], []
    for method_index, scores in enumerate(all_scores):
        for position, q in enumerate(query_ids.tolist()):
            for budget in BUDGETS:
                selected = methods.select_supportive(scores[:, q], budget)
                seed = 162000 + 10000 * method_index + 100 * position + budget
                controls = methods.matched_random_removals(train["y"], selected, 16, seed)
                for rep, removed in enumerate(torch.cat([selected[None], controls])):
                    mask = torch.ones(len(train["y"]), dtype=torch.bool)
                    mask[removed] = False
                    masks.append(mask)
                    target.append(q)
                    budget_values.append(budget)
                    method_values.append(method_index)
                    replicate.append(rep - 1)
                    class_counts.append(torch.bincount(train["y"][removed], minlength=2))
    return dict(
        masks=torch.stack(masks),
        target=torch.tensor(target),
        budget=torch.tensor(budget_values),
        method=torch.tensor(method_values),
        replicate=torch.tensor(replicate),
        removed_class_counts=torch.stack(class_counts),
        query_indices=query_ids,
        pixel_scores=pixel_scores,
    )


def prepare_study(train: dict, query: dict, designs: dict, methods, directory: Path) -> dict:
    """Fit/validate datamodels and construct deletions; never fit test subsets."""
    directory.mkdir(parents=True, exist_ok=False)
    subset_fits, outputs, responses, learning = {}, {}, {}, {}
    for fraction in FRACTIONS:
        fit = {
            name: refit_masks(train, designs[fraction][name]) for name in ("train", "validation")
        }
        margins = {name: model_margins(value, query, methods) for name, value in fit.items()}
        x, xv = [designs[fraction][name].double() for name in ("train", "validation")]
        y, yv = margins["train"], margins["validation"]
        response = fit_response_models(x, y, xv, yv, methods)
        if fraction == 0.5:
            for count in (64, 128, 256, 512):
                learning[count] = fit_response_models(x[:count], y[:count], xv, yv, methods)
        subset_fits[fraction], outputs[fraction], responses[fraction] = fit, margins, response
    deletions = build_deletion_masks(train, query, responses, methods)
    result = dict(
        phase="FIT_VALIDATION_AND_DELETION_SELECTION_COMPLETE",
        designs=designs,
        subset_fits=subset_fits,
        margins=outputs,
        responses=responses,
        learning=learning,
        deletions=deletions,
        training_ids=train["ids"],
        query_ids=query["ids"],
        test_subset_responses_seen=False,
        counterfactual_responses_seen=False,
    )
    torch.save(result, directory / "prepared.pt")
    return result


def response_predictions(response: dict, masks: Tensor) -> dict[str, Tensor]:
    x = masks.double()
    predictions = {}
    for name in ("lasso", "permuted"):
        model = response[name]
        predictions[name] = x @ model["selected_weights"] + model["selected_bias"]
    model = response["least_squares"]
    predictions["least_squares"] = x @ model["weights"] + model["bias"]
    predictions["intercept"] = response["intercept"].expand(len(x), -1)
    return predictions


def evaluate_study(prepared: dict, train: dict, query: dict, methods, directory: Path) -> dict:
    """Retrain held-out subsets and frozen counterfactuals, with no model selection."""
    if prepared.get("phase") != "FIT_VALIDATION_AND_DELETION_SELECTION_COMPLETE":
        raise ValueError("Complete all datamodel fitting and deletion selection before evaluation.")
    if prepared["training_ids"] != train["ids"] or prepared["query_ids"] != query["ids"]:
        raise ValueError("Data row identity/order changed between preparation and evaluation.")
    if prepared["test_subset_responses_seen"] or prepared["counterfactual_responses_seen"]:
        raise ValueError("The prepared experiment claims it already inspected evaluation outcomes.")
    tests, cross_fraction, learning_predictions = {}, {}, {}
    for fraction in FRACTIONS:
        masks = prepared["designs"][fraction]["test"]
        fit = refit_masks(train, masks)
        margin = model_margins(fit, query, methods)
        predictions = response_predictions(prepared["responses"][fraction], masks)
        tests[fraction] = dict(fit=fit, margin=margin, predictions=predictions)
        cross_fraction[fraction] = {
            source: response_predictions(prepared["responses"][source], masks)["lasso"]
            for source in FRACTIONS
        }
        if fraction == 0.5:
            for count, response in prepared["learning"].items():
                learning_predictions[count] = response_predictions(response, masks)
    base_fit = refit_masks(train, torch.ones(1, len(train["y"]), dtype=torch.bool))
    base_margin = model_margins(base_fit, query, methods)[0]
    deletion = prepared["deletions"]
    refits = refit_masks(train, deletion["masks"])
    all_margin = model_margins(refits, query, methods)
    rows = torch.arange(len(all_margin))
    query_index = deletion["target"]
    target_margin = all_margin[rows, query_index]
    prediction = []
    for fraction in FRACTIONS:
        weights = prepared["responses"][fraction]["lasso"]["selected_weights"]
        full_changes = -(~deletion["masks"]).double() @ weights
        prediction.append(full_changes[rows, query_index])
    result = dict(
        phase="FROZEN_EVALUATION_COMPLETE",
        test=tests,
        cross_fraction=cross_fraction,
        learning_predictions=learning_predictions,
        base_fit=base_fit,
        base_margin=base_margin,
        counterfactual=dict(
            fit=refits,
            all_margin=all_margin,
            target_margin=target_margin,
            actual_change=target_margin - base_margin[query_index],
            predicted_change=torch.stack(prediction, dim=1),
        ),
        selection_used_test_responses=False,
    )
    torch.save(result, directory / "evaluated.pt")
    return result


def deletion_play(
    prepared: dict,
    train: dict,
    query: dict,
    methods,
    query_index: int = 0,
    fraction: float = 0.5,
    budget: int = 8,
) -> dict:
    """An editable intervention: base, selected deletion and sixteen random refits."""
    if not 0 <= query_index < len(query["y"]) or fraction not in FRACTIONS:
        raise ValueError("Choose an available query and fitted subsampling fraction.")
    weights = prepared["responses"][fraction]["lasso"]["selected_weights"][:, query_index]
    selected = methods.select_supportive(weights, budget)
    controls = methods.matched_random_removals(train["y"], selected, 16, 163001)
    removals = torch.cat([selected[None], controls])
    masks = torch.ones(18, len(train["y"]), dtype=torch.bool)
    for i, removed in enumerate(removals, start=1):
        masks[i, removed] = False
    fit = refit_masks(train, masks)
    margin = model_margins(fit, query, methods)[:, query_index]
    prediction = -(~masks).double() @ weights
    return dict(
        query_index=query_index,
        fraction=fraction,
        budget=budget,
        removals=removals,
        masks=masks,
        fit=fit,
        margin=margin,
        actual_change=margin - margin[0],
        predicted_change=prediction,
    )
