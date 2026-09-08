"""Complete CPU experiment; every function is also visible in the lesson."""

from pathlib import Path
import hashlib
import json
import numpy as np
import torch as t
from chapter16_shapley_attribution_baselines.exercises.part12_sage_global_importance.sage_data import (
    background_order,
)


def fit_models(train):
    """Fresh training only; this function does not receive an evaluation split."""
    t.set_num_threads(4)
    t.use_deterministic_algorithms(True)
    xraw = t.from_numpy(train["x"])
    yraw = t.from_numpy(train["y"])
    xm = xraw.mean(0)
    xs = xraw.std(0, correction=0)
    ym = yraw.mean()
    ys = yraw.std(correction=0)
    x = (xraw - xm) / xs
    y = (yraw - ym) / ys
    models = []
    for control in ["clean", "shuffled"]:
        for index in range(3):
            seed = 161210 + index
            t.manual_seed(seed)
            net = t.nn.Sequential(
                t.nn.Linear(8, 32, dtype=t.float64),
                t.nn.ReLU(),
                t.nn.Linear(32, 32, dtype=t.float64),
                t.nn.ReLU(),
                t.nn.Linear(32, 1, dtype=t.float64),
            )
            initial = {key: value.detach().clone() for key, value in net.state_dict().items()}
            labels = y.clone()
            permutation = t.arange(len(y))
            if control == "shuffled":
                permutation = t.from_numpy(
                    np.random.default_rng(161220 + index).permutation(len(y))
                )
                labels = labels[permutation]
            optimizer = t.optim.Adam(net.parameters(), lr=0.003, weight_decay=0.0001)
            losses = []
            with t.enable_grad():
                for _ in range(800):
                    optimizer.zero_grad()
                    loss = (net(x).squeeze(-1) - labels).square().mean()
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.detach().clone())
            models.append(
                {
                    "kind": "mlp",
                    "control": control,
                    "seed": seed,
                    "initial": initial,
                    "state": {
                        key: value.detach().clone() for key, value in net.state_dict().items()
                    },
                    "loss": t.stack(losses),
                    "target_permutation": permutation,
                    "y_mean": ym,
                    "y_std": ys,
                }
            )
    design = t.cat([x, t.ones((len(x), 1), dtype=x.dtype)], dim=1)
    penalty = t.eye(9, dtype=x.dtype) * 0.01
    penalty[-1, -1] = 0
    coefficients = t.linalg.solve(design.T @ design / len(x) + penalty, design.T @ y / len(x))
    models.append(
        {
            "kind": "ridge",
            "control": "ridge",
            "seed": 0,
            "coefficients": coefficients,
            "y_mean": ym,
            "y_std": ys,
        }
    )
    return {
        "models": models,
        "x_mean": xm,
        "x_std": xs,
        "train_x": x,
        "train_y": yraw,
        "train_ids": t.from_numpy(train["ids"]),
    }


def make_predictor(model):
    """Visible frozen network forward pass returning predictions in MPa."""

    def predict(x):
        if model["kind"] == "ridge":
            value = x @ model["coefficients"][:8] + model["coefficients"][8]
        else:
            state = model["state"]
            hidden = t.relu(x @ state["0.weight"].T + state["0.bias"])
            hidden = t.relu(hidden @ state["2.weight"].T + state["2.bias"])
            value = (hidden @ state["4.weight"].T + state["4.bias"]).squeeze(-1)
        return value * model["y_std"] + model["y_mean"]

    return predict


def explain_predictions(predictions, y, methods):
    local = methods.loss_game(predictions, y)
    global_game = local.mean(0)
    local_loss = methods.exact_shapley(local)
    prediction_shap = methods.exact_shapley(predictions)
    d = 8
    full = 2**d - 1
    return {
        "predictions": predictions,
        "local_game": local,
        "game": global_game,
        "sage": methods.exact_shapley(global_game),
        "local_loss": local_loss,
        "local_prediction": prediction_shap,
        "mean_prediction": prediction_shap.mean(0),
        "mean_abs_prediction": prediction_shap.abs().mean(0),
        "leave_one_out": t.stack(
            [global_game[full] - global_game[full ^ (1 << j)] for j in range(d)]
        ),
        "singleton": global_game[1 << t.arange(d)],
    }


def prepare_study(train, explain, directory, methods):
    """Fit on train; explain/rank on explanation split; audit is unavailable here."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    fitted = fit_models(train)
    # Persist fits before any model attribution is evaluated.
    t.save(fitted, directory / "fitted.pt")
    x = (t.from_numpy(explain["x"]) - fitted["x_mean"]) / fitted["x_std"]
    y = t.from_numpy(explain["y"])
    orders = [
        t.from_numpy(background_order(train["ids"], seed)) for seed in [161260, 161261, 161262]
    ]
    background = fitted["train_x"][orders[0][:64]]
    explanations = []
    rankings = []
    for model in fitted["models"]:
        predictions = methods.coalition_predictions(make_predictor(model), x, background)
        result = explain_predictions(predictions, y, methods)
        explanations.append(result)
        rankings.append(
            t.stack(
                [
                    t.argsort(result["sage"], descending=True, stable=True),
                    t.argsort(result["sage"], stable=True),
                    t.argsort(result["leave_one_out"], descending=True, stable=True),
                ]
            )
        )
    rng = np.random.default_rng(161280)
    random_orders = t.from_numpy(np.stack([rng.permutation(8) for _ in range(100)]))
    # The saved order is the boundary: no audit labels or inputs have been loaded.
    prepared = {
        **fitted,
        "explain_x": x,
        "explain_y": y,
        "explain_ids": t.from_numpy(explain["ids"]),
        "background_orders": orders,
        "background": background,
        "explanations": explanations,
        "rankings": t.stack(rankings),
        "random_orders": random_orders,
        "phase": "rankings_frozen_before_audit",
    }
    t.save(prepared, directory / "prepared.pt")
    return prepared


def evaluate_study(prepared, audit, methods):
    if prepared.get("phase") != "rankings_frozen_before_audit":
        raise ValueError("Freeze explanation rankings before loading the audit split.")
    x = (t.from_numpy(audit["x"]) - prepared["x_mean"]) / prepared["x_std"]
    y = t.from_numpy(audit["y"])
    evaluated = []
    for index, model in enumerate(prepared["models"]):
        predictions = methods.coalition_predictions(
            make_predictor(model), x, prepared["background"]
        )
        result = explain_predictions(predictions, y, methods)
        orders = t.cat([prepared["rankings"][index], prepared["random_orders"]], dim=0)
        result["prefix_improvement"] = methods.prefix_values(result["game"], orders)
        result["prefix_mse"] = (predictions[:, 0] - y).square().mean() - result[
            "prefix_improvement"
        ]
        evaluated.append(result)
    return {
        "x": x,
        "y": y,
        "ids": t.from_numpy(audit["ids"]),
        "models": evaluated,
        "training_mean_mse": (y - prepared["train_y"].mean()).square().mean(),
    }


def diagnostics(prepared, methods):
    """All prespecified estimator, background, weighting and missingness controls."""
    model = make_predictor(prepared["models"][0])
    x = prepared["explain_x"]
    y = prepared["explain_y"]
    base = prepared["explanations"][0]
    variants = []
    names = []
    for order_index, size, name in [
        (0, 16, "nested16"),
        (0, 128, "nested128"),
        (1, 64, "independent64_a"),
        (2, 64, "independent64_b"),
    ]:
        background = prepared["train_x"][prepared["background_orders"][order_index][:size]]
        variants.append(
            explain_predictions(methods.coalition_predictions(model, x, background), y, methods)
        )
        names.append(name)
    variants.append(
        explain_predictions(
            methods.coalition_predictions(model, x, prepared["background"].mean(0, keepdim=True)),
            y,
            methods,
        )
    )
    names.append("mean_imputation")
    permutation = t.from_numpy(np.random.default_rng(161270).permutation(len(y)))
    shuffled_outcomes = explain_predictions(base["predictions"], y[permutation], methods)
    rows, perms = methods.sampling_schedule(4096, len(x), 8, 161290)
    direct = methods.permutation_sage(model, x, y, prepared["background"], rows, perms)
    table = methods.table_path_contributions(base["local_game"], rows, perms)
    repeated = []
    summaries = []
    budgets = [32, 128, 512, 2048, 8192]
    for seed in range(161300, 161332):
        rr, pp = methods.sampling_schedule(8192, len(x), 8, seed)
        scores = methods.table_path_contributions(base["local_game"], rr, pp)
        repeated.append(scores)
        summaries.append(methods.summarize_paths(scores, budgets))
    # Each ingredient recipe, regardless of the number of ages/replicates, gets equal weight.
    # Recover groups from exact source inputs rather than round-trip standardized floats.
    from chapter16_shapley_attribution_baselines.exercises.part12_sage_global_importance.sage_data import (
        source_data,
    )

    source = source_data()[prepared["explain_ids"].numpy(), :7]
    _, inverse, counts = np.unique(source, axis=0, return_inverse=True, return_counts=True)
    weights = t.from_numpy(1.0 / counts[inverse])
    weights = weights / weights.sum()
    recipe_game = (base["local_game"] * weights[:, None]).sum(0)
    # Hybrids use an intact donor row; the visible distance control still reveals changed inputs.
    donor = prepared["background"][t.arange(len(x)) % 64]
    distance = {"original": t.cdist(x, prepared["train_x"]).min(1).values}
    for mask, name in [(1, "cement_only"), (15, "first_four"), (127, "age_missing")]:
        keep = methods.coalition_masks(8)[mask]
        hybrid = t.where(keep, x, donor)
        distance[name] = t.cdist(hybrid, prepared["train_x"]).min(1).values
    # A wrapped ninth coordinate is unused. Enumeration verifies its dummy contribution.
    dummy_x = t.cat([x[:12], t.linspace(-2, 2, 12)[:, None]], 1)
    dummy_bg = t.cat([prepared["background"][:8], t.linspace(-1, 1, 8)[:, None]], 1)
    dummy_pred = methods.coalition_predictions(lambda z: model(z[:, :8]), dummy_x, dummy_bg)
    dummy_sage = methods.exact_shapley(methods.loss_game(dummy_pred, y[:12]).mean(0))
    return {
        "variant_names": names,
        "variants": variants,
        "outcome_permutation": permutation,
        "shuffled_outcomes": shuffled_outcomes,
        "direct_rows": rows,
        "direct_permutations": perms,
        "direct_scores": direct,
        "table_scores": table,
        "repeated_scores": t.stack(repeated),
        "sampling_means": t.stack([s["mean"] for s in summaries]),
        "sampling_se": t.stack([s["se"] for s in summaries]),
        "budgets": t.tensor(budgets),
        "recipe_weights": weights,
        "recipe_sage": methods.exact_shapley(recipe_game),
        "nearest_train_distance": distance,
        "dummy_sage": dummy_sage,
    }


def native_play(prepared, audit, methods, row_index=0, feature=0, background_size=64):
    if (
        not 0 <= row_index < len(audit["y"])
        or not 0 <= feature < 8
        or not 1 <= background_size <= len(prepared["train_x"])
    ):
        raise ValueError("Choose an existing row, feature 0..7 and a nonempty training background.")
    x = audit["x"][row_index : row_index + 1]
    y = audit["y"][row_index : row_index + 1]
    background = prepared["train_x"][prepared["background_orders"][0][:background_size]]
    model = make_predictor(prepared["models"][0])
    prediction = methods.coalition_predictions(model, x, background)
    explanation = explain_predictions(prediction, y, methods)
    hybrids = x.expand(len(background), 8).clone()
    hybrids[:, feature] = background[:, feature]
    return {
        "source_id": int(audit["ids"][row_index]),
        "feature": feature,
        "background_size": background_size,
        "explanation": explanation,
        "hybrid_predictions": model(hybrids),
        "target": y,
        "input": x,
    }
