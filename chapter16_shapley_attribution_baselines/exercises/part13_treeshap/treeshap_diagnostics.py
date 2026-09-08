"""Learner-visible TreeSHAP mechanism diagnostics and frozen runtime controls.

The functions in this file construct controlled games; the nine method
implementations remain in ``treeshap_methods.py``.  Every returned value is a
plain mapping/list or NumPy value so native runs can persist deterministic
mechanism evidence separately from descriptive wall-clock timings.
"""

from __future__ import annotations

import itertools
import math
import time

import numpy as np

from . import treeshap_methods as _DEFAULT_METHODS


def _method(methods, name):
    """Resolve a taught method in both the package and AST-extracted notebook."""
    if methods is not None:
        return getattr(methods, name) if hasattr(methods, name) else methods[name]
    local = globals().get(name)
    if callable(local):
        return local
    return getattr(_DEFAULT_METHODS, name)


def _as_query(query):
    query = np.asarray(query, dtype=float)
    if query.shape != (2,) or not np.isin(query, [0.0, 1.0]).all():
        raise ValueError("The binary fixture query must contain exactly two zeros/ones.")
    return query


def _binary_additive_tree(order=(0, 1), joint=((9.0, 1.0), (1.0, 9.0))):
    """Build an exact tree for f(x)=x0+2*x1 with population node covers."""
    order = tuple(int(feature) for feature in order)
    if sorted(order) != [0, 1]:
        raise ValueError("order must be a permutation of (0, 1).")
    joint = np.asarray(joint, dtype=float)
    if joint.shape != (2, 2) or np.any(joint <= 0):
        raise ValueError("joint must be a positive 2x2 table in (x0, x1) order.")
    examples = np.array(list(itertools.product([0, 1], repeat=2)))
    counts = joint.ravel()
    fields = {key: [] for key in ("left", "right", "feature", "threshold", "value", "cover")}

    def add(depth, keep):
        node = len(fields["left"])
        for key in fields:
            fields[key].append(-1)
        fields["cover"][node] = counts[keep].sum()
        fields["value"][node] = np.average(examples[keep] @ [1.0, 2.0], weights=counts[keep])
        fields["threshold"][node] = 0.5
        if depth < 2:
            feature = order[depth]
            fields["feature"][node] = feature
            fields["left"][node] = add(depth + 1, keep[examples[keep, feature] == 0])
            fields["right"][node] = add(depth + 1, keep[examples[keep, feature] == 1])
        return node

    add(0, np.arange(4))
    return {
        key: np.asarray(value, dtype=int if key in ("left", "right", "feature") else float)
        for key, value in fields.items()
    }


def _repeated_threshold_tree():
    """A hand-computable tree where x0 splits twice but remains one player."""
    return {
        "left": np.array([1, -1, 3, -1, -1]),
        "right": np.array([2, -1, 4, -1, -1]),
        "feature": np.array([0, -1, 0, -1, -1]),
        "threshold": np.array([1.0, 0.0, 2.0, 0.0, 0.0]),
        "value": np.array([20.0, 10.0, 80 / 3, 20.0, 30.0]),
        "cover": np.array([100.0, 40.0, 60.0, 20.0, 40.0]),
    }


def _conditional_binary_game(query, joint):
    """Population-conditional game E[f(X) | X_S=query_S] by direct enumeration."""
    examples = np.array(list(itertools.product([0, 1], repeat=2)), dtype=float)
    probabilities = np.asarray(joint, dtype=float).ravel()
    probabilities /= probabilities.sum()
    targets = examples @ np.array([1.0, 2.0])
    masks = np.array(list(itertools.product([False, True], repeat=2)), dtype=bool)
    # Reorder lexicographic masks to the least-significant-feature convention.
    masks = masks[[0, 2, 1, 3]]
    values = []
    for mask in masks:
        matches = np.all(examples[:, mask] == query[mask], axis=1)
        values.append(np.average(targets[matches], weights=probabilities[matches]))
    return masks, np.asarray(values)


def _interventional_binary_game(query, joint):
    """Marginal/interventional game E[f(x_S, X_-S)] by direct enumeration."""
    examples = np.array(list(itertools.product([0, 1], repeat=2)), dtype=float)
    probabilities = np.asarray(joint, dtype=float).ravel()
    probabilities /= probabilities.sum()
    masks = np.array([[False, False], [True, False], [False, True], [True, True]])
    values = []
    for mask in masks:
        hybrids = np.where(mask, query, examples)
        values.append(np.average(hybrids @ np.array([1.0, 2.0]), weights=probabilities))
    return masks, np.asarray(values)


def correlated_games(query=(1.0, 1.0), joint=((0.45, 0.05), (0.05, 0.45)), methods=None):
    """Compare conditional, interventional and two cover-weighted path games.

    The two trees compute the same f(x)=x0+2*x1 on all binary inputs.  Only the
    root split changes, which makes any difference in their path attributions a
    property of the cover-weighted game rather than the predictive function.
    """
    query = _as_query(query)
    joint = np.asarray(joint, dtype=float)
    masks, conditional = _conditional_binary_game(query, joint)
    _, interventional = _interventional_binary_game(query, joint)
    trees = [_binary_additive_tree((0, 1), joint), _binary_additive_tree((1, 0), joint)]
    path_game = _method(methods, "tree_path_game")
    exact = _method(methods, "exact_shapley")
    saabas_fn = _method(methods, "saabas")
    games = np.vstack(
        [
            conditional,
            interventional,
            path_game(trees[0], query[None], masks)[0],
            path_game(trees[1], query[None], masks)[0],
        ]
    )
    shapley = exact(games)
    saabas_values = np.vstack([saabas_fn(tree, query[None])[1][0] for tree in trees])
    predictions = np.array(
        [_method(methods, "tree_predict")(tree, query[None])[0] for tree in trees]
    )
    return {
        "query": query,
        "joint": joint,
        "masks": masks,
        "game_names": [
            "Population conditional",
            "Interventional",
            "Path: x0 root",
            "Path: x1 root",
        ],
        "games": games,
        "shapley": shapley,
        "trees": trees,
        "saabas": saabas_values,
        "predictions": predictions,
        "empty_value": float(games[0, 0]),
    }


def _enumerate_cardinality_weights(zeros, ones):
    """Definition-level subset enumeration for one unique-feature path."""
    zeros, ones = np.asarray(zeros, dtype=float), np.asarray(ones, dtype=float)
    count = len(zeros)
    answer = np.zeros(count + 1)
    for flags in itertools.product([False, True], repeat=count):
        size = sum(flags)
        product = math.prod(ones[j] if flag else zeros[j] for j, flag in enumerate(flags))
        answer[size] += (
            product
            * math.factorial(size)
            * math.factorial(count - size)
            / math.factorial(count + 1)
        )
    return answer


def path_cardinality_diagnostic(zeros=(0.2, 0.7, 0.5), ones=(1.0, 0.0, 1.0), methods=None):
    """Record each extend-path update and verify it against subset enumeration."""
    zeros, ones = np.asarray(zeros, dtype=float), np.asarray(ones, dtype=float)
    if zeros.shape != ones.shape or zeros.ndim != 1 or not len(zeros):
        raise ValueError(
            "zeros and ones must be non-empty one-dimensional arrays of equal length."
        )
    extend = _method(methods, "extend_path")
    weights = np.ones((1, 1))
    stages, exact_stages = [weights[0].copy()], [np.ones(1)]
    for index in range(len(zeros)):
        weights = extend(weights, zeros[[index]], ones[[index]])
        stages.append(weights[0].copy())
        exact_stages.append(
            _enumerate_cardinality_weights(zeros[: index + 1], ones[: index + 1])
        )
    return {
        "zeros": zeros,
        "ones": ones,
        "stages": stages,
        "enumerated_stages": exact_stages,
        "max_abs_error": float(
            max(np.max(np.abs(a - b)) for a, b in zip(stages, exact_stages))
        ),
    }


def repeated_feature_diagnostic(methods=None):
    """Show that repeated x0 splits contract to one player and a dummy stays zero."""
    tree = _repeated_threshold_tree()
    query = np.array([[0.0, 99.0], [1.5, -7.0], [3.0, 4.0]])
    masks = _method(methods, "coalition_masks")(2)
    games = _method(methods, "tree_path_game")(tree, query, masks)
    phi = _method(methods, "tree_shap")(tree, query)
    base, saabas_values = _method(methods, "saabas")(tree, query)
    predictions = _method(methods, "tree_predict")(tree, query)
    return {
        "tree": tree,
        "query": query,
        "masks": masks,
        "games": games,
        "base": base,
        "phi": phi,
        "saabas": saabas_values,
        "prediction": predictions,
        "dummy_max_abs": float(np.max(np.abs(phi[:, 1]))),
        "additivity_max_abs_error": float(np.max(np.abs(base + phi.sum(1) - predictions))),
    }


def build_balanced_tree(depth, n_features=None, seed=161390):
    """Build the frozen balanced binary timing tree with one feature per level."""
    depth = int(depth)
    n_features = depth if n_features is None else int(n_features)
    if depth < 1 or n_features < depth:
        raise ValueError("depth must be positive and n_features must supply every tree level.")
    node_count = 2 ** (depth + 1) - 1
    leaf_start = 2**depth - 1
    left = np.full(node_count, -1, dtype=int)
    right = np.full(node_count, -1, dtype=int)
    feature = np.full(node_count, -1, dtype=int)
    threshold = np.zeros(node_count, dtype=float)
    value = np.zeros(node_count, dtype=float)
    cover = np.ones(node_count, dtype=float)
    for node in range(leaf_start):
        level = int(math.floor(math.log2(node + 1)))
        left[node], right[node], feature[node] = 2 * node + 1, 2 * node + 2, level
        threshold[node] = 0.5
    value[leaf_start:] = np.random.default_rng(seed).normal(loc=10.0, scale=3.0, size=2**depth)
    for node in range(leaf_start - 1, -1, -1):
        cover[node] = cover[left[node]] + cover[right[node]]
        value[node] = (
            cover[left[node]] * value[left[node]] + cover[right[node]] * value[right[node]]
        ) / cover[node]
    return {
        "left": left,
        "right": right,
        "feature": feature,
        "threshold": threshold,
        "value": value,
        "cover": cover,
        "n_features": np.array(n_features, dtype=int),
    }


def _fixed_queries(n_features):
    """Eight deterministic binary queries, with a visible 0..7 bit pattern."""
    integers = np.arange(8, dtype=int)
    return np.array(
        [
            [(number >> (feature % 3)) & 1 for feature in range(n_features)]
            for number in integers
        ],
        dtype=float,
    )


def _benchmark_axis(specifications, methods, repeats):
    masks_fn = _method(methods, "coalition_masks")
    game_fn = _method(methods, "tree_path_game")
    exact_fn = _method(methods, "exact_shapley")
    dp_fn = _method(methods, "tree_shap")
    exact_times, dp_times, errors = [], [], []
    for depth, supplied_features in specifications:
        tree = build_balanced_tree(depth, supplied_features, seed=161390)
        query = _fixed_queries(supplied_features)
        masks = masks_fn(supplied_features)
        expected = exact_fn(game_fn(tree, query, masks))
        actual = dp_fn(tree, query)
        errors.append(float(np.max(np.abs(expected - actual))))
        current_exact, current_dp = [], []
        for _ in range(repeats):
            started = time.perf_counter()
            exact_fn(game_fn(tree, query, masks))
            current_exact.append(time.perf_counter() - started)
            started = time.perf_counter()
            dp_fn(tree, query)
            current_dp.append(time.perf_counter() - started)
        exact_times.append(current_exact)
        dp_times.append(current_dp)
    exact_times, dp_times = np.asarray(exact_times), np.asarray(dp_times)
    return {
        "exhaustive_seconds": exact_times,
        "dynamic_programming_seconds": dp_times,
        "exhaustive_median_seconds": np.median(exact_times, axis=1),
        "dynamic_programming_median_seconds": np.median(dp_times, axis=1),
        "max_abs_error": np.asarray(errors),
        "queries_per_timing": np.array(8, dtype=int),
        "repeats": np.array(repeats, dtype=int),
    }


def benchmark_runtime(methods=None, repeats=3):
    """Run both frozen timing sweeps and retain every wall-clock measurement."""
    repeats = int(repeats)
    if repeats != 3:
        raise ValueError("The frozen protocol requires exactly three timing repetitions.")
    dummy_features = np.array([3, 6, 9, 12, 15], dtype=int)
    depths = np.arange(2, 9, dtype=int)
    dummy = _benchmark_axis([(3, int(count)) for count in dummy_features], methods, repeats)
    depth = _benchmark_axis([(int(value), int(value)) for value in depths], methods, repeats)
    dummy["supplied_features"] = dummy_features
    dummy["tree_depth"] = np.array(3, dtype=int)
    depth["tree_depth"] = depths
    depth["supplied_features"] = depths.copy()
    return {"dummy_features": dummy, "depth": depth}


def tiny_tree_play(
    query=(1.0, 1.0),
    joint=((9.0, 1.0), (1.0, 9.0)),
    root_feature=0,
    methods=None,
):
    """Editable tiny-tree play: change the query, root feature, or four covers."""
    query = _as_query(query)
    root_feature = int(root_feature)
    if root_feature not in (0, 1):
        raise ValueError("root_feature must be 0 or 1.")
    tree = _binary_additive_tree((root_feature, 1 - root_feature), joint)
    masks = _method(methods, "coalition_masks")(2)
    game = _method(methods, "tree_path_game")(tree, query[None], masks)[0]
    phi = _method(methods, "tree_shap")(tree, query[None])[0]
    base, saabas_values = _method(methods, "saabas")(tree, query[None])
    prediction = _method(methods, "tree_predict")(tree, query[None])[0]
    return {
        "query": query,
        "joint": np.asarray(joint, dtype=float),
        "root_feature": np.array(root_feature, dtype=int),
        "tree": tree,
        "masks": masks,
        "game": game,
        "base": np.array(base[0]),
        "phi": phi,
        "saabas": saabas_values[0],
        "prediction": np.array(prediction),
    }


def run_diagnostics(methods=None, include_timing=True):
    """Return exact mechanism evidence and a separately persisted timing branch."""
    mechanism = {
        "correlated": correlated_games(methods=methods),
        "cardinality": path_cardinality_diagnostic(methods=methods),
        "repeated_feature": repeated_feature_diagnostic(methods=methods),
        "tiny_play": tiny_tree_play(methods=methods),
    }
    timing = benchmark_runtime(methods=methods) if include_timing else {}
    if timing:
        mechanism["runtime_parity"] = {
            axis: {
                "coordinate": (
                    values["supplied_features"]
                    if axis == "dummy_features"
                    else values["tree_depth"]
                ),
                "max_abs_error": values["max_abs_error"],
            }
            for axis, values in timing.items()
        }
        mechanism["runtime_parity"]["dummy_feature_max_abs"] = np.array(
            np.max(timing["dummy_features"]["max_abs_error"])
        )
        mechanism["runtime_parity"]["depth_max_abs"] = np.array(
            np.max(timing["depth"]["max_abs_error"])
        )
        timing = {
            axis: {key: value for key, value in values.items() if key != "max_abs_error"}
            for axis, values in timing.items()
        }
    return {"mechanism": mechanism, "timing": timing}
