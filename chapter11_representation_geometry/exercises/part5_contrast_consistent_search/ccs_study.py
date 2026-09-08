"""Transparent control orchestration, also shown in full in the native learner.

The notebook supplies its own implemented methods through the same names. Fresh
reproduction uses these reference functions with freshly captured activations.
"""

import torch

from chapter11_representation_geometry.exercises.part5_contrast_consistent_search import ccs_methods as m
from chapter11_representation_geometry.exercises.part5_contrast_consistent_search import ccs_native_methods as nm
from chapter11_representation_geometry.exercises.part4_concept_erasure.erasure_methods import fit_binary_probe


def fit_reporters(train_plus, train_minus, training_labels):
    centering = m.fit_pair_centering(train_plus, train_minus)
    plus, minus = m.apply_pair_centering(train_plus, train_minus, centering)
    result = {"centering": centering, "groups": {}}
    for mode in ("centered", "raw", "shuffled_pairs"):
        x_plus, x_minus = (train_plus, train_minus) if mode == "raw" else (plus, minus)
        permutation = None
        if mode == "shuffled_pairs":
            permutation = torch.randperm(len(x_minus), generator=torch.Generator().manual_seed(904))
            x_minus = x_minus[permutation]
        fitted = m.fit_ccs_restarts(x_plus, x_minus)
        result["groups"][mode] = {"fitted": fitted, "selected": m.select_restart(fitted), "permutation": permutation}
    result["pca"] = m.fit_difference_pca(plus, minus)
    # This baseline alone consumes the training labels. They do not enter CCS.
    result["supervised"] = fit_binary_probe(torch.cat((plus, minus), dim=1), training_labels)
    selected = result["groups"]["centered"]["selected"]
    random_weights = []
    for seed in range(1000, 1032):
        weight = torch.randn(selected["weight"].shape, dtype=torch.float64, generator=torch.Generator().manual_seed(seed))
        weight *= selected["weight"].norm() / weight.norm()
        random_weights.append(weight)
    result["random"] = {"weights": torch.stack(random_weights), "bias": selected["bias"].clone()}
    return result


def reporter_scores(fitted, raw_plus, raw_minus):
    plus, minus = m.apply_pair_centering(raw_plus, raw_minus, fitted["centering"])
    scores = {}
    for mode, group in fitted["groups"].items():
        inputs = (raw_plus, raw_minus) if mode == "raw" else (plus, minus)
        fit = group["fitted"]
        values = m.paired_probabilities(*inputs, fit["weights"], fit["biases"])[0]
        for index, seed in enumerate(fit["seeds"]):
            scores[f"CCS {mode} seed {seed}"] = values[:, index]
    pca = fitted["pca"]
    scores["Paired-difference PCA"] = ((plus - minus - pca["mean"]) @ pca["weight"]).sigmoid()
    supervised = fitted["supervised"]
    scores["Supervised linear probe"] = (
        torch.cat((plus, minus), dim=1) @ supervised["coefficient"] + supervised["intercept"]
    ).sigmoid()
    random = fitted["random"]
    values = m.paired_probabilities(plus, minus, random["weights"], random["bias"].expand(32))[0]
    for index in range(32):
        scores[f"Random direction {1000 + index}"] = values[:, index]
    if len(scores) != 64:
        raise ValueError("The protocol requires all 64 reporters per case.")
    return scores


def calibrate_reporters(fitted, calibration_plus, calibration_minus, calibration_labels):
    scores = reporter_scores(fitted, calibration_plus, calibration_minus)
    orientations = {name: m.calibrate_orientation(value, calibration_labels) for name, value in scores.items()}
    orientations["Supervised linear probe"] = 1
    return orientations


def evaluate_reporters(fitted, scores, rows, orientations):
    if set(scores) != set(orientations) or len(scores) != 64:
        raise ValueError("Every reporter must retain its calibration orientation.")
    labels = torch.tensor([row["label"] for row in rows])
    information = {}
    for mode, group in fitted["groups"].items():
        for index, seed in enumerate(group["fitted"]["seeds"]):
            information[f"CCS {mode} seed {seed}"] = {
                "training_loss": float(group["fitted"]["history"][-1, index, 0]),
                "selected": index == group["selected"]["index"],
            }
    records = []
    for name, values in scores.items():
        orientation = orientations[name]
        record = {"name": name, "orientation": orientation, **information.get(name, {}),
                  "target": m.binary_metrics(values, labels, orientation)}
        if rows[0]["distractor"] is not None:
            distractor = torch.tensor([row["distractor"] for row in rows])
            record["distractor"] = m.binary_metrics(values, distractor, orientation)
            change = nm.paired_opinion_change(values if orientation == 1 else 1 - values, rows)
            record["opinion_change"] = {key: change[key] for key in (
                "base_reviews", "prediction_flip_fraction", "mean_absolute_score_change")}
        records.append(record)
    return records


def selected_name(fitted, mode="centered"):
    return f'CCS {mode} seed {fitted["groups"][mode]["selected"]["seed"]}'


def summarize_native_outputs(cache):
    rows = cache["rows"]
    metrics = nm.model_answer_metrics(cache["prefix"]["logits"], cache["candidate_token_ids"],
                                      torch.tensor([row["label"] for row in rows]))
    result = {key: metrics[key] for key in ("count", "restricted_correct", "full_vocabulary_correct", "target_full_vocabulary_nll")}
    result["mean_candidate_probability_mass"] = float(metrics["candidate_mass"].mean())
    if rows[0]["distractor"] is not None:
        change = nm.paired_opinion_change(metrics["restricted_positive_scores"], rows)
        result["opinion_prediction_flip_fraction"] = change["prediction_flip_fraction"]
    return result
