"""Readable reference answers; every taught method also appears in the lesson."""

from __future__ import annotations

import math
import numpy as np


def coalition_masks(d: int) -> np.ndarray:
    """All subsets, with feature zero in the least-significant bit."""
    return np.array([[bool(s & (1 << j)) for j in range(d)] for s in range(2**d)])


def tree_predict(tree: dict, x: np.ndarray) -> np.ndarray:
    """Predict N rows with one scalar-output tree; match sklearn float32 routing."""
    x = np.asarray(x, dtype=np.float32).astype(np.float64)
    nodes = np.zeros(len(x), dtype=int)
    while True:
        active = np.flatnonzero(tree["left"][nodes] != -1)
        if not len(active):
            return tree["value"][nodes].copy()
        current = nodes[active]
        goes_left = x[active, tree["feature"][current]] <= tree["threshold"][current]
        nodes[active] = np.where(goes_left, tree["left"][current], tree["right"][current])


def tree_path_game(tree: dict, x: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Return (N, K) coalition values using local training-cover branch weights."""
    x = np.asarray(x, dtype=np.float32).astype(np.float64)
    masks = np.asarray(masks, dtype=bool)

    def visit(node):
        left, right = tree["left"][node], tree["right"][node]
        if left == -1:
            return np.full((len(x), len(masks)), tree["value"][node])
        feature = tree["feature"][node]
        lvalue, rvalue = visit(left), visit(right)
        weight = tree["cover"][left] / tree["cover"][node]
        absent = weight * lvalue + (1 - weight) * rvalue
        present = np.where((x[:, feature] <= tree["threshold"][node])[:, None], lvalue, rvalue)
        return np.where(masks[None, :, feature], present, absent)

    return visit(0)


def exact_shapley(values: np.ndarray) -> np.ndarray:
    """Contract a (..., 2**D) game with the exact Shapley subset weights."""
    values = np.asarray(values, dtype=np.float64)
    d = int(math.log2(values.shape[-1]))
    if d < 1 or values.shape[-1] != 2**d:
        raise ValueError("The final axis must contain every subset of D >= 1 features.")
    phi = np.zeros((*values.shape[:-1], d))
    for feature in range(d):
        absent = [s for s in range(2**d) if not s & (1 << feature)]
        present = [s | (1 << feature) for s in absent]
        weights = np.array([1 / (d * math.comb(d - 1, s.bit_count())) for s in absent])
        phi[..., feature] = ((values[..., present] - values[..., absent]) * weights).sum(-1)
    return phi


def extend_path(weights: np.ndarray, zero: np.ndarray, one: np.ndarray) -> np.ndarray:
    """Add a unique feature to batched cardinality weights: (N, k+1) -> (N, k+2)."""
    weights = np.asarray(weights, dtype=np.float64)
    zero, one = np.asarray(zero), np.asarray(one)
    k = weights.shape[1] - 1
    extended = np.zeros((len(weights), k + 2))
    for size in range(k + 1):
        extended[:, size] += zero * weights[:, size] * (k + 1 - size) / (k + 2)
        extended[:, size + 1] += one * weights[:, size] * (size + 1) / (k + 2)
    return extended


def unwind_path(weights: np.ndarray, zero: np.ndarray, one: np.ndarray) -> np.ndarray:
    """Remove one feature's extension, without mutating its input or dividing by zero."""
    weights = np.asarray(weights, dtype=np.float64)
    zero, one = np.broadcast_arrays(np.asarray(zero), np.asarray(one))
    if np.any((zero == 0) & (one == 0)):
        raise ValueError("A dead path has no reversible extension; prune it first.")
    n = weights.shape[1] - 1
    if n < 1:
        raise ValueError("Cannot remove a feature from the empty path.")
    result = np.zeros((len(weights), n))
    next_one = weights[:, n].copy()
    nonzero = one != 0
    safe_one = np.where(nonzero, one, 1.0)
    safe_zero = np.where(nonzero, 1.0, zero)
    for size in range(n - 1, -1, -1):
        from_one = next_one * (n + 1) / ((size + 1) * safe_one)
        from_zero = weights[:, size] * (n + 1) / ((n - size) * safe_zero)
        result[:, size] = np.where(nonzero, from_one, from_zero)
        next_one = weights[:, size] - result[:, size] * zero * (n - size) / (n + 1)
    return result


def tree_shap(tree: dict, x: np.ndarray, donor: np.ndarray | None = None) -> np.ndarray:
    """Exact path-game SHAP, or paired-query/donor SHAP when donor has shape (N, D)."""
    x = np.asarray(x, dtype=np.float32).astype(np.float64)
    if donor is not None:
        donor = np.asarray(donor, dtype=np.float32).astype(np.float64)
        if donor.shape != x.shape:
            raise ValueError("Each query needs exactly one paired donor row.")
    phi = np.zeros(x.shape, dtype=np.float64)

    def visit(node, rows, features, zeros, ones, weights):
        left, right = tree["left"][node], tree["right"][node]
        if left == -1:
            for index, feature in enumerate(features):
                other_weights = unwind_path(weights, zeros[index], ones[index])
                phi[rows, feature] += (
                    other_weights.sum(1) * (ones[index] - zeros[index]) * tree["value"][node]
                )
            return

        feature = tree["feature"][node]
        features, zeros, ones = list(features), list(zeros), list(ones)
        incoming_zero = incoming_one = np.ones(len(rows))
        if feature in features:
            index = features.index(feature)
            incoming_zero, incoming_one = zeros.pop(index), ones.pop(index)
            weights = unwind_path(weights, incoming_zero, incoming_one)
            features.pop(index)

        query_left = x[rows, feature] <= tree["threshold"][node]
        if donor is not None:
            donor_left = donor[rows, feature] <= tree["threshold"][node]
        for child, is_left in ((left, True), (right, False)):
            present = (query_left == is_left).astype(float)
            absent = (
                (donor_left == is_left).astype(float)
                if donor is not None
                else np.full(len(rows), tree["cover"][child] / tree["cover"][node])
            )
            zero, one = incoming_zero * absent, incoming_one * present
            keep = (zero != 0) | (one != 0)
            if not keep.any():
                continue
            visit(
                child,
                rows[keep],
                features + [feature],
                [z[keep] for z in zeros] + [zero[keep]],
                [o[keep] for o in ones] + [one[keep]],
                extend_path(weights[keep], zero[keep], one[keep]),
            )

    visit(0, np.arange(len(x)), [], [], [], np.ones((len(x), 1)))
    return phi


def ensemble_shap(
    ensemble: dict, x: np.ndarray, background: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Use tree and donor linearity; return base (N,) and contributions (N, D)."""
    x = np.asarray(x)
    phi = np.zeros(x.shape, dtype=float)
    base = np.full(len(x), float(ensemble["offset"]))
    for tree, weight in zip(ensemble["trees"], ensemble["weights"], strict=True):
        if background is None:
            contribution = tree_shap(tree, x)
            empty = tree_path_game(tree, x[:1], np.zeros((1, x.shape[1]), dtype=bool))[0, 0]
        else:
            background = np.asarray(background)
            if not len(background):
                raise ValueError("The empirical background cannot be empty.")
            queries = np.repeat(x, len(background), axis=0)
            donors = np.tile(background, (len(x), 1))
            contribution = (
                tree_shap(tree, queries, donors)
                .reshape(len(x), len(background), x.shape[1])
                .mean(1)
            )
            empty = tree_predict(tree, background).mean()
        phi += weight * contribution
        base += weight * empty
    return base, phi


def interventional_game(
    predict, x: np.ndarray, background: np.ndarray, masks: np.ndarray
) -> np.ndarray:
    """Average predictions of intact-donor hybrids, retaining all supplied donors."""
    if not len(background):
        raise ValueError("The empirical background cannot be empty.")
    result = np.empty((len(x), len(masks)))
    for start in range(0, len(x), 8):
        query = x[start : start + 8]
        hybrids = np.where(
            masks[None, :, None, :], query[:, None, None, :], background[None, None, :, :]
        )
        predictions = predict(hybrids.reshape(-1, x.shape[1]))
        result[start : start + len(query)] = predictions.reshape(
            len(query), len(masks), len(background)
        ).mean(2)
    return result


def saabas(tree: dict, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Assign each visited parent-to-child expectation change to its split feature."""
    x = np.asarray(x, dtype=np.float32).astype(np.float64)
    expected = np.zeros(len(tree["value"]))

    def expectation(node):
        left, right = tree["left"][node], tree["right"][node]
        if left == -1:
            expected[node] = tree["value"][node]
        else:
            expected[node] = (
                tree["cover"][left] * expectation(left)
                + tree["cover"][right] * expectation(right)
            ) / tree["cover"][node]
        return expected[node]

    expectation(0)
    phi = np.zeros(x.shape, dtype=float)
    for row, query in enumerate(x):
        node = 0
        while tree["left"][node] != -1:
            feature = tree["feature"][node]
            child = (
                tree["left"][node]
                if query[feature] <= tree["threshold"][node]
                else tree["right"][node]
            )
            phi[row, feature] += expected[child] - expected[node]
            node = child
    return np.full(len(x), expected[0]), phi
