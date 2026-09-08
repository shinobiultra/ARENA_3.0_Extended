"""Visible, bounded ProxySPEX variant: residual trees -> spectrum -> refinement.

Copy the core function bodies into learner cells; importing this answer module
is not a substitute for implementing them. See METHOD_API.md for the prospective
grid and the distinction from the untouched author LightGBM pipeline.
"""

from itertools import product

import numpy as np
from sklearn.tree import DecisionTreeRegressor


PROXY_CANDIDATES = tuple(
    {"max_depth": depth, "n_estimators": count, "learning_rate": 0.1, "min_samples_leaf": 2}
    for depth, count in product((1, 3), (128, 512))
)
RIDGE_ALPHAS = (1e-6, 1e-3, 0.1, 1.0, 10.0)


def binary_masks(masks):
    """Validate before casting: fractional masks must never become binary silently."""
    array = np.asarray(masks)
    if array.ndim != 2 or not all(array.shape) or not np.isin(array, (0, 1)).all():
        raise ValueError("masks must be a nonempty binary 0/1 matrix")
    return array.astype(np.uint8, copy=False)


def training_arrays(masks, values):
    masks = binary_masks(masks)
    values = np.asarray(values, dtype=float)
    if values.shape != (len(masks),) or not np.isfinite(values).all():
        raise ValueError("values must be finite and have one entry per mask")
    return masks, values


def canonical_spectrum(spectrum, n_features=None):
    """Require the same unambiguous support representation throughout the lesson."""
    result = {}
    for support, coefficient in spectrum.items():
        if not isinstance(support, tuple) or any(
            not isinstance(i, (int, np.integer)) or isinstance(i, (bool, np.bool_)) or i < 0
            for i in support
        ):
            raise ValueError("supports must be tuples of nonnegative feature indices")
        if tuple(sorted(set(support))) != support:
            raise ValueError("support indices must be sorted and unique")
        if n_features is not None and support and support[-1] >= n_features:
            raise ValueError("support exceeds the mask feature count")
        value = float(coefficient)
        if not np.isfinite(value):
            raise ValueError("Fourier coefficients must be finite")
        result[support] = value
    return result


def character_matrix(masks, supports):
    """One column per chi_S, including an all-ones column for S=()."""
    masks = binary_masks(masks)
    supports = tuple(supports)
    canonical_spectrum(dict.fromkeys(supports, 0.0), masks.shape[1])
    columns = [1.0 - 2.0 * (masks[:, support].sum(axis=1) % 2) for support in supports]
    return np.column_stack(columns) if columns else np.empty((len(masks), 0))


def predict_spectrum(masks, spectrum):
    masks = binary_masks(masks)
    spectrum = canonical_spectrum(spectrum, masks.shape[1])
    return character_matrix(masks, spectrum) @ np.array(list(spectrum.values()))


def path_indicator_fourier(path):
    """Expand [(feature, required_bit), ...] by multiplying (1 +/- chi_i)/2.

    XOR multiplication makes repeated constraints idempotent and contradictory
    constraints zero. Using set union here would give incorrect coefficients.
    """
    spectrum = {(): 1.0}
    for feature, bit in path:
        canonical_spectrum({(feature,): 1.0})
        if bit not in (0, 1):
            raise ValueError("path bits must be binary")
        next_spectrum = {}
        for support, coefficient in spectrum.items():
            toggled = tuple(sorted(set(support) ^ {feature}))
            next_spectrum[support] = next_spectrum.get(support, 0.0) + coefficient / 2
            next_spectrum[toggled] = next_spectrum.get(toggled, 0.0) + (1 - 2 * bit) * coefficient / 2
        spectrum = {support: value for support, value in next_spectrum.items() if value != 0.0}
    return spectrum


def tree_to_fourier(tree):
    """Exact path-indicator expansion of a scalar sklearn-style regression tree.

    For binary inputs, a threshold in [0,1) tests bit 0 versus bit 1. Outside
    that interval only one branch is reachable. The primitive's leaf values
    already contain its fitted residual means; shrinkage belongs to the ensemble.
    """
    nodes = tree.tree_
    result = {}

    def visit(node, path):
        left, right = int(nodes.children_left[node]), int(nodes.children_right[node])
        if left == right == -1:
            leaf_value = np.asarray(nodes.value[node]).reshape(-1)
            if len(leaf_value) != 1 or not np.isfinite(leaf_value[0]):
                raise ValueError("only finite scalar regression leaves are supported")
            for support, value in path_indicator_fourier(path).items():
                result[support] = result.get(support, 0.0) + float(leaf_value[0]) * value
            return
        feature, threshold = int(nodes.feature[node]), float(nodes.threshold[node])
        if feature < 0 or feature >= tree.n_features_in_ or not np.isfinite(threshold):
            raise ValueError("invalid binary tree split")
        if threshold < 0:
            visit(right, path)
        elif threshold >= 1:
            visit(left, path)
        else:
            visit(left, path + [(feature, 0)])
            visit(right, path + [(feature, 1)])

    visit(0, [])
    return {support: value for support, value in result.items() if value != 0.0}


def fit_boosted_trees(masks, values, *, max_depth, n_estimators, learning_rate=0.1,
                      min_samples_leaf=2, seed=0):
    """Squared-loss gradient boosting, exposing every residual and update.

    DecisionTreeRegressor is the disclosed CART fitting primitive, not the
    method wrapper. All requested trees are fitted, including zero residuals.
    """
    masks, values = training_arrays(masks, values)
    for name, value in (("max_depth", max_depth), ("n_estimators", n_estimators),
                        ("min_samples_leaf", min_samples_leaf)):
        if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not np.isfinite(learning_rate) or not 0 < learning_rate <= 1:
        raise ValueError("learning_rate must lie in (0,1]")
    initial_value = float(values.mean())
    prediction = np.full(len(values), initial_value)
    losses, trees = [float(np.mean((values - prediction) ** 2))], []
    for step in range(n_estimators):
        residual = values - prediction
        tree = DecisionTreeRegressor(
            max_depth=max_depth, min_samples_leaf=min_samples_leaf,
            criterion="squared_error", random_state=int(seed) + step,
        )
        tree.fit(masks, residual)
        prediction += learning_rate * tree.predict(masks)
        trees.append(tree)
        losses.append(float(np.mean((values - prediction) ** 2)))
    return {
        "initial_value": initial_value, "trees": trees, "learning_rate": float(learning_rate),
        "n_features": masks.shape[1], "train_mse": np.asarray(losses),
        "parameters": {"max_depth": int(max_depth), "n_estimators": int(n_estimators),
                       "learning_rate": float(learning_rate), "min_samples_leaf": int(min_samples_leaf)},
        "seed": int(seed),
    }


def predict_boosted_trees(model, masks):
    masks = binary_masks(masks)
    if masks.shape[1] != model["n_features"]:
        raise ValueError("mask feature count differs from the fitted ensemble")
    prediction = np.full(len(masks), model["initial_value"], dtype=float)
    for tree in model["trees"]:
        prediction += model["learning_rate"] * tree.predict(masks)
    return prediction


def ensemble_to_fourier(model):
    """Linearity: initial constant plus every shrinkage-weighted tree spectrum."""
    result = {(): model["initial_value"]}
    for tree in model["trees"]:
        for support, coefficient in tree_to_fourier(tree).items():
            result[support] = result.get(support, 0.0) + model["learning_rate"] * coefficient
    return {support: value for support, value in result.items() if support == () or value != 0.0}


def select_energy_support(spectrum, energy_fraction=0.95):
    """Keep intercept and the shortest prefix reaching >= nonconstant energy.

    Equal magnitudes are ordered by support tuple; ties are not all discarded
    and need not all be retained. Fractions 0 and 1 have explicit meanings.
    """
    spectrum = canonical_spectrum(spectrum)
    if not np.isfinite(energy_fraction) or not 0 <= energy_fraction <= 1:
        raise ValueError("energy_fraction must lie in [0,1]")
    ranked = sorted((support for support in spectrum if support and spectrum[support] != 0),
                    key=lambda support: (-abs(spectrum[support]), support))
    # Scale first so squaring finite large/small coefficients does not overflow/underflow.
    scale = max((abs(spectrum[support]) for support in ranked), default=1.0)
    coefficients = np.asarray([spectrum[support] for support in ranked], dtype=np.longdouble)
    energies = (coefficients / np.longdouble(scale)) ** 2
    total = energies.sum()
    # Normalize before searching; extended-precision accumulation avoids turning
    # the exact 9/(9+1)=0.9 boundary into a reported 0.8999999999999999.
    cumulative_fraction = np.asarray(np.cumsum(energies) / total, dtype=float) if total else np.array([])
    count = 0
    if total > 0 and energy_fraction > 0:
        count = len(ranked) if energy_fraction == 1 else min(
            len(ranked), int(np.searchsorted(cumulative_fraction, energy_fraction, side="left")) + 1
        )
    selected = {(): spectrum.get((), 0.0)}
    selected.update((support, spectrum[support]) for support in ranked[:count])
    retained = float(cumulative_fraction[count - 1]) if count else (0.0 if total else 1.0)
    return {"spectrum": selected, "supports": tuple(selected), "requested_energy_fraction": float(energy_fraction),
            "retained_energy_fraction": retained, "selected_nonconstant_count": count,
            "available_nonconstant_count": len(ranked)}


def author_strict_energy_selection(spectrum):
    """Reproduce ONLY the author's unchanged strict selection algebra (diagnostic).

    Index tuples replace the author's length-n binary support keys. This is not
    an execution of the author estimator; actual untouched failures are retained
    in the reference report. Empty input raises and pure parity selects nothing.
    """
    spectrum = canonical_spectrum(spectrum)
    keys = list(spectrum)
    coefficients = np.array(list(spectrum.values()))
    if () in keys:
        coefficients[keys.index(())] = 0
    squared = coefficients ** 2
    sorted_squared = np.sort(squared)[::-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        boundary_index = np.argmin(np.cumsum(sorted_squared / np.sum(squared)) < 0.95)
    threshold = np.sqrt(sorted_squared[boundary_index])
    return {support: value for support, value in spectrum.items() if abs(value) > threshold}


def grouped_mask_folds(masks, *, fold_seed, folds=3):
    """Seeded CV groups duplicate masks, returning explicit train/validation IDs."""
    masks = binary_masks(masks)
    _, groups = np.unique(masks, axis=0, return_inverse=True)
    n_groups = int(groups.max()) + 1
    if not isinstance(folds, (int, np.integer)) or not 2 <= folds <= n_groups:
        raise ValueError("folds must be between 2 and the number of unique training masks")
    shuffled = np.random.default_rng(fold_seed).permutation(n_groups)
    result = []
    for validation_groups in np.array_split(shuffled, folds):
        validation = np.isin(groups, validation_groups)
        result.append({"train_indices": np.flatnonzero(~validation),
                       "validation_indices": np.flatnonzero(validation)})
    return result


def ridge_path(features, values, alphas):
    """Solve ||X beta-y||² + alpha ||beta||² for each explicitly supplied alpha.

    The intercept is an ordinary all-ones feature and is also regularized.
    One SVD supports the whole path, including rank-deficient designs.
    """
    features, values, alphas = np.asarray(features, float), np.asarray(values, float), np.asarray(alphas, float)
    if (features.ndim != 2 or not all(features.shape) or values.shape != (len(features),)
            or alphas.ndim != 1 or not len(alphas) or not np.isfinite(alphas).all()
            or (alphas <= 0).any() or not np.isfinite(features).all() or not np.isfinite(values).all()):
        raise ValueError("ridge needs finite nonempty X/y and positive finite alphas")
    u, singular, vt = np.linalg.svd(features, full_matrices=False)
    projected = u.T @ values
    return ((singular[None, :] * projected[None, :]) /
            (singular[None, :] ** 2 + alphas[:, None])) @ vt


def refine_coefficients(masks, values, full_spectrum, *, fold_records, fold_spectra,
                        energy_fraction=0.95, alphas=RIDGE_ALPHAS):
    """Choose ridge strength using support learned separately inside each fold.

    Callers must supply each training-fold tree spectrum, never a spectrum fitted
    on that fold's validation values. fit_proxy constructs these records itself.
    The final fit uses all supplied training rows; no audit data are accepted.
    """
    masks, values = training_arrays(masks, values)
    alphas = tuple(alphas)
    if len(fold_records) != len(fold_spectra) or not fold_records:
        raise ValueError("provide one training-only spectrum per fold")
    fold_losses, supports_by_fold = [], []
    sums = np.zeros(len(alphas))
    seen = []
    for record, spectrum in zip(fold_records, fold_spectra, strict=True):
        train, validation = record["train_indices"], record["validation_indices"]
        train, validation = np.asarray(train), np.asarray(validation)
        if (train.ndim != 1 or validation.ndim != 1 or not len(train) or not len(validation)
                or not np.issubdtype(train.dtype, np.integer) or not np.issubdtype(validation.dtype, np.integer)
                or not np.array_equal(np.sort(np.concatenate([train, validation])), np.arange(len(masks)))):
            raise ValueError("each fold must partition all training row indices exactly once")
        train_masks = {tuple(row) for row in masks[train]}
        if any(tuple(row) in train_masks for row in masks[validation]):
            raise ValueError("duplicate masks must not cross a validation fold")
        selected = select_energy_support(spectrum, energy_fraction)["spectrum"]
        train_design = character_matrix(masks[train], selected)
        validation_design = character_matrix(masks[validation], selected)
        coefficients = ridge_path(train_design, values[train], alphas)
        squared_errors = (validation_design @ coefficients.T - values[validation, None]) ** 2
        sums += squared_errors.sum(axis=0)
        fold_losses.append(squared_errors.mean(axis=0))
        supports_by_fold.append(tuple(selected))
        seen.extend(validation.tolist())
    if sorted(seen) != list(range(len(masks))):
        raise ValueError("validation folds must cover every training row exactly once")
    losses = sums / len(values)
    best = int(np.argmin(losses))
    selection = select_energy_support(full_spectrum, energy_fraction)
    coefficients = ridge_path(character_matrix(masks, selection["spectrum"]), values, [alphas[best]])[0]
    return {
        "spectrum": dict(zip(selection["spectrum"], coefficients.tolist(), strict=True)),
        "selection": selection,
        "cv": {"alphas": alphas, "fold_mse": np.asarray(fold_losses), "validation_mse": losses,
               "selected_index": best, "selected_alpha": float(alphas[best]),
               "fold_supports": supports_by_fold},
    }


def fit_proxy(train_masks, train_values, *, fold_seed, candidates=PROXY_CANDIDATES,
              folds=3, energy_fraction=0.95, ridge_alphas=RIDGE_ALPHAS):
    """Bounded training-only selection, exact extraction and ridge refinement.

    Proxy candidates are selected by pooled grouped-CV MSE. The same development
    folds then select ridge strength, using that candidate's fold-trained supports.
    These CV scores are selection scores, not independent generalization audits.
    """
    masks, values = training_arrays(train_masks, train_values)
    candidates = tuple(dict(candidate) for candidate in candidates)
    if not candidates:
        raise ValueError("at least one proxy candidate is required")
    records = grouped_mask_folds(masks, fold_seed=fold_seed, folds=folds)
    best_loss, best_index, best_fold_spectra = float("inf"), None, None
    candidate_losses, fold_losses, tree_fits = [], [], 0
    for index, parameters in enumerate(candidates):
        losses, spectra, squared_error_sum = [], [], 0.0
        for fold_index, record in enumerate(records):
            train, validation = record["train_indices"], record["validation_indices"]
            model = fit_boosted_trees(masks[train], values[train], seed=int(fold_seed) + fold_index, **parameters)
            tree_fits += len(model["trees"])
            errors = (predict_boosted_trees(model, masks[validation]) - values[validation]) ** 2
            squared_error_sum += float(errors.sum())
            losses.append(float(errors.mean()))
            spectra.append(ensemble_to_fourier(model))
        pooled = squared_error_sum / len(values)
        candidate_losses.append(pooled)
        fold_losses.append(losses)
        if pooled < best_loss:
            best_loss, best_index, best_fold_spectra = pooled, index, spectra
    model = fit_boosted_trees(masks, values, seed=int(fold_seed), **candidates[best_index])
    tree_fits += len(model["trees"])
    full_spectrum = ensemble_to_fourier(model)
    refinement = refine_coefficients(masks, values, full_spectrum, fold_records=records,
                                    fold_spectra=best_fold_spectra, energy_fraction=energy_fraction,
                                    alphas=ridge_alphas)
    return {
        "model": model, "full_spectrum": full_spectrum,
        "truncated_spectrum": refinement["selection"]["spectrum"],
        "refined_spectrum": refinement["spectrum"], "selection": refinement["selection"],
        "proxy_cv": {"candidates": candidates, "fold_mse": np.asarray(fold_losses),
                     "validation_mse": np.asarray(candidate_losses), "selected_index": best_index,
                     "selected_parameters": candidates[best_index]},
        "ridge_cv": refinement["cv"], "folds": records,
        "accounting": {"input_rows": len(values), "unique_input_masks": len(np.unique(masks, axis=0)),
                       "oracle_calls_by_methods": 0, "ensemble_fits": len(candidates) * folds + 1,
                       "residual_tree_fits": tree_fits, "ridge_candidate_fits": len(ridge_alphas) * folds + 1,
                       "ridge_matrix_decompositions": folds + 1, "fold_seed": int(fold_seed)},
    }


def stage_errors(model_values, proxy_values, extracted_values, truncated_values, refined_values):
    """Separate every approximation stage on one caller-owned audit set.

    These RMSEs do not add. Refinement can worsen fidelity and is never clipped.
    """
    arrays = [np.asarray(values, float) for values in
              (model_values, proxy_values, extracted_values, truncated_values, refined_values)]
    if (arrays[0].ndim != 1 or not len(arrays[0])
            or any(a.shape != arrays[0].shape or not np.isfinite(a).all() for a in arrays)):
        raise ValueError("stage values must be equally sized, finite nonempty vectors")
    truth, proxy, full, truncated, refined = arrays

    def rmse(left, right):
        return float(np.sqrt(np.mean((left - right) ** 2)))

    return {"audit_rows": len(truth), "model_to_proxy_rmse": rmse(truth, proxy),
            "proxy_to_extraction_rmse": rmse(proxy, full),
            "extraction_to_truncation_rmse": rmse(full, truncated),
            "truncation_to_refinement_rmse": rmse(truncated, refined),
            "model_to_extraction_rmse": rmse(truth, full),
            "model_to_truncation_rmse": rmse(truth, truncated),
            "model_to_refinement_rmse": rmse(truth, refined)}
