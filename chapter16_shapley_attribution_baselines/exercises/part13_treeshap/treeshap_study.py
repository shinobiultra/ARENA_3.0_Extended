"""Complete visible fitting and evaluation code for the frozen public-data study."""

from __future__ import annotations

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from .treeshap_data import background_order
from .treeshap_methods import (
    coalition_masks,
    tree_predict,
    tree_path_game,
    exact_shapley,
    extend_path,
    unwind_path,
    tree_shap,
    ensemble_shap,
    interventional_game,
    saabas,
)


def adapt_tree(estimator):
    raw = estimator.tree_
    return {
        "left": raw.children_left.copy(),
        "right": raw.children_right.copy(),
        "feature": raw.feature.copy(),
        "threshold": raw.threshold.copy(),
        "value": raw.value[:, 0, 0].copy(),
        "cover": raw.weighted_n_node_samples.copy(),
    }


def adapt_ensemble(model, family):
    if family == "tree":
        estimators, weights, offset = [model], np.ones(1), 0.0
    elif family == "forest":
        estimators = model.estimators_
        weights, offset = np.full(len(estimators), 1 / len(estimators)), 0.0
    elif family == "boost":
        estimators = model.estimators_.ravel()
        weights = np.full(len(estimators), model.learning_rate)
        offset = float(model.init_.constant_[0, 0])
    else:
        raise ValueError(f"Unknown family {family}.")
    return {
        "trees": [adapt_tree(tree) for tree in estimators],
        "weights": weights,
        "offset": offset,
    }


def ensemble_predict(ensemble, x):
    return float(ensemble["offset"]) + sum(
        weight * tree_predict(tree, x)
        for tree, weight in zip(ensemble["trees"], ensemble["weights"], strict=True)
    )


def ensemble_path_game(ensemble, x, masks):
    return float(ensemble["offset"]) + sum(
        weight * tree_path_game(tree, x, masks)
        for tree, weight in zip(ensemble["trees"], ensemble["weights"], strict=True)
    )


def ensemble_saabas(ensemble, x):
    return sum(
        weight * saabas(tree, x)[1]
        for tree, weight in zip(ensemble["trees"], ensemble["weights"], strict=True)
    )


def fit_models(train):
    models = {}
    for family in ("tree", "forest", "boost"):
        for seed_index in range(3):
            for target_kind in ("clean", "shuffled"):
                seed = 161310 + seed_index
                y = train["y"].copy()
                if target_kind == "shuffled":
                    y = y[np.random.default_rng(161320 + seed_index).permutation(len(y))]
                if family == "tree":
                    model = DecisionTreeRegressor(
                        max_depth=5, min_samples_leaf=8, random_state=seed
                    )
                elif family == "forest":
                    model = RandomForestRegressor(
                        n_estimators=32,
                        max_depth=5,
                        min_samples_leaf=8,
                        max_features=1.0,
                        bootstrap=True,
                        n_jobs=1,
                        random_state=seed,
                    )
                else:
                    model = GradientBoostingRegressor(
                        n_estimators=48,
                        learning_rate=0.05,
                        max_depth=3,
                        min_samples_leaf=8,
                        loss="squared_error",
                        subsample=1.0,
                        random_state=seed,
                    )
                model.fit(train["x"], y)
                key = f"{family}_{target_kind}_{seed_index}"
                models[key] = {
                    "sklearn": model,
                    "ensemble": adapt_ensemble(model, family),
                    "training_targets": y,
                    "parameters": model.get_params(),
                    "train_prediction": model.predict(train["x"]),
                }
    return {
        "models": models,
        "train_ids": train["ids"].copy(),
        "training_mean": float(train["y"].mean()),
    }


def explain_block(fitted, x, background):
    result = {}
    masks = coalition_masks(x.shape[1])
    exact_keys = {f"{family}_clean_0" for family in ("tree", "forest", "boost")}
    for key, item in fitted["models"].items():
        ensemble, model = item["ensemble"], item["sklearn"]
        prediction = ensemble_predict(ensemble, x)
        np.testing.assert_allclose(prediction, model.predict(x), atol=1e-10, rtol=1e-10)
        base_path, phi_path = ensemble_shap(ensemble, x)
        base_interventional, phi_interventional = ensemble_shap(ensemble, x, background)
        phi_saabas = ensemble_saabas(ensemble, x)
        for base, phi in (
            (base_path, phi_path),
            (base_interventional, phi_interventional),
            (base_path, phi_saabas),
        ):
            np.testing.assert_allclose(base + phi.sum(1), prediction, atol=1e-8, rtol=1e-8)
        result[key] = {
            "prediction": prediction,
            "base_path": base_path,
            "phi_path": phi_path,
            "base_interventional": base_interventional,
            "phi_interventional": phi_interventional,
            "phi_saabas": phi_saabas,
        }
        if key in exact_keys:
            game_path = ensemble_path_game(ensemble, x, masks)
            # Independent prediction route: exact hybrid games use the fitted sklearn model.
            game_interventional = interventional_game(model.predict, x, background, masks)
            for game, phi in ((game_path, phi_path), (game_interventional, phi_interventional)):
                np.testing.assert_allclose(exact_shapley(game), phi, atol=1e-8, rtol=1e-8)
            result[key].update(game_path=game_path, game_interventional=game_interventional)
    return result


def prepare_study(fitted, train, explain):
    order = background_order(train["ids"], 161360)
    background = train["x"][order[:64]]
    block = explain_block(fitted, explain["x"], background)
    rankings = {
        key: {
            name: np.argsort(-np.abs(item[f"phi_{name}"]).mean(0), kind="stable")
            for name in ("path", "interventional", "saabas")
        }
        for key, item in block.items()
    }
    return {
        "explanation": block,
        "rankings": rankings,
        "explain_ids": explain["ids"].copy(),
        "background_ids": train["ids"][order[:64]],
        "background": background,
        "audit_used_for_selection": False,
    }


def evaluate_study(fitted, prepared, train, explain, audit):
    block = explain_block(fitted, audit["x"], prepared["background"])
    rng = np.random.default_rng(161380)
    random_orders = np.array([rng.permutation(8) for _ in range(100)])
    deletion = {}
    for key, item in block.items():
        if "game_interventional" not in item:
            continue
        orders = np.vstack(
            [prepared["rankings"][key][name] for name in ("path", "interventional", "saabas")]
            + [random_orders]
        )
        masks_by_order = []
        for order in orders:
            mask, sequence = 255, [255]
            for feature in order:
                mask &= ~(1 << int(feature))
                sequence.append(mask)
            masks_by_order.append(sequence)
        predictions = item["game_interventional"][:, np.array(masks_by_order)]
        deletion[key] = {
            "orders": orders,
            "remaining_masks": np.array(masks_by_order),
            "mse": ((predictions - audit["y"][:, None, None]) ** 2).mean(0),
        }
    primary = fitted["models"]["forest_clean_0"]["ensemble"]
    all_x = np.vstack([explain["x"], audit["x"]])
    backgrounds = {}
    for size, seed in ((16, 161360), (128, 161360), (64, 161361), (64, 161362)):
        order = background_order(train["ids"], seed)[:size]
        base, phi = ensemble_shap(primary, all_x, train["x"][order])
        backgrounds[f"{size}_{seed}"] = {"ids": train["ids"][order], "base": base, "phi": phi}
    return {
        "audit": block,
        "audit_ids": audit["ids"].copy(),
        "audit_targets": audit["y"].copy(),
        "deletion": deletion,
        "backgrounds": backgrounds,
        "performance": {
            key: float(np.mean((item["prediction"] - audit["y"]) ** 2))
            for key, item in block.items()
        },
        "mean_baseline_mse": float(np.mean((fitted["training_mean"] - audit["y"]) ** 2)),
    }


def native_play(
    fitted, train, audit, model_key="forest_clean_0", row=0, size=64, game="interventional"
):
    query = audit["x"][[row]]
    indices = background_order(train["ids"], 161360)[:size]
    background = train["x"][indices] if game == "interventional" else None
    if game not in ("interventional", "path"):
        raise ValueError("Choose interventional or path.")
    ensemble = fitted["models"][model_key]["ensemble"]
    base, phi = ensemble_shap(ensemble, query, background)
    return {
        "query": query,
        "source_id": audit["ids"][[row]],
        "base": base,
        "phi": phi,
        "prediction": ensemble_predict(ensemble, query),
        "target": audit["y"][[row]],
    }
