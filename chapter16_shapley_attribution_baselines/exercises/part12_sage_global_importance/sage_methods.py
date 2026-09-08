"""Visible learner implementations for SAGE; no data or study state."""

import math
import numpy as np
import torch as t


def coalition_masks(d):
    """Return Boolean (2**d, d); row m contains the set bits of integer m."""
    if not isinstance(d, int) or not 1 <= d <= 16:
        raise ValueError("Use an integer number of features between 1 and 16.")
    return ((t.arange(2**d)[:, None] >> t.arange(d)) & 1).bool()


def marginal_predict(model, x, background, masks):
    """Predict N restricted inputs by averaging B complete hybrid predictions."""
    if x.ndim != 2 or background.ndim != 2 or masks.shape != x.shape:
        raise ValueError("Expected x=(N,D), background=(B,D), masks=(N,D).")
    if masks.dtype != t.bool or x.shape[1] != background.shape[1] or len(background) == 0:
        raise ValueError("Masks must be Boolean; background needs matching nonempty rows.")
    hybrids = t.where(masks[:, None, :], x[:, None, :], background[None, :, :])
    prediction = model(hybrids.reshape(-1, x.shape[1]))
    if prediction.shape != (len(x) * len(background),):
        raise ValueError("The regressor must return one scalar per input row.")
    return prediction.reshape(len(x), len(background)).mean(dim=1)


def loss_game(predictions, y):
    """(N, 2**D) restricted predictions → per-row improvement in squared loss."""
    if predictions.ndim != 2 or y.shape != (len(predictions),):
        raise ValueError("Expected predictions=(N,M) and targets=(N,).")
    losses = (predictions - y[:, None]).square()
    return losses[:, :1] - losses


def exact_shapley(values):
    """Apply subset weights along the last axis, preserving all leading axes."""
    size = values.shape[-1]
    d = size.bit_length() - 1
    if size < 2 or 2**d != size:
        raise ValueError("The final dimension must contain every coalition, including empty.")
    phi = t.zeros((*values.shape[:-1], d), dtype=values.dtype)
    for feature in range(d):
        for mask in range(size):
            if not mask & (1 << feature):
                weight = 1 / (d * math.comb(d - 1, mask.bit_count()))
                phi[..., feature] += weight * (
                    values[..., mask | (1 << feature)] - values[..., mask]
                )
    return phi


def gaussian_game(covariance, truth, model, conditional=True, noise_variance=0.0):
    """Exact zero-mean Gaussian restriction, risks and improvement for linear f."""
    d = len(truth)
    if covariance.shape != (d, d) or model.shape != (d,) or noise_variance < 0:
        raise ValueError("Incompatible Gaussian parameters or negative noise variance.")
    if not t.allclose(covariance, covariance.T) or t.linalg.eigvalsh(covariance).min() < -1e-10:
        raise ValueError("Covariance must be symmetric positive semidefinite.")
    coefficients = t.zeros((2**d, d), dtype=covariance.dtype)
    for mask, present in enumerate(coalition_masks(d)):
        indices = t.where(present)[0]
        if len(indices) == d:
            coefficients[mask] = model
        elif len(indices):
            if conditional:
                block = covariance[indices][:, indices]
                coefficients[mask, indices] = t.linalg.pinv(block) @ covariance[indices] @ model
            else:
                coefficients[mask, indices] = model[indices]
    residual = truth[None, :] - coefficients
    risks = t.einsum("md,df,mf->m", residual, covariance, residual) + noise_variance
    return {"coefficients": coefficients, "risk": risks, "game": risks[0] - risks}


def path_contributions(restricted, x, y, permutations):
    """One loss-reduction path per row, stored in feature rather than step order."""
    n, d = x.shape
    if y.shape != (n,) or permutations.shape != (n, d):
        raise ValueError("Paths must have one target and one permutation per query.")
    if permutations.dtype != t.int64 or not t.equal(
        permutations.sort(dim=1).values, t.arange(d).expand(n, d)
    ):
        raise ValueError("Every path must contain each feature exactly once.")
    masks = t.zeros_like(x, dtype=t.bool)
    previous = (restricted(x, masks) - y).square()
    scores = t.empty_like(x)
    rows = t.arange(n)
    for step in range(d):
        feature = permutations[:, step]
        masks[rows, feature] = True
        current = (restricted(x, masks) - y).square()
        scores[rows, feature] = previous - current
        previous = current
    return scores


def permutation_sage(model, x, y, background, rows, permutations, batch_size=256):
    """Sample specified query rows/permutations without constructing a full game."""
    if rows.ndim != 1 or permutations.shape != (len(rows), x.shape[1]) or batch_size < 1:
        raise ValueError("Incompatible sampling schedule or batch size.")
    restricted = lambda query, masks: marginal_predict(model, query, background, masks)
    scores = []
    for start in range(0, len(rows), batch_size):
        ids = rows[start : start + batch_size]
        scores.append(
            path_contributions(restricted, x[ids], y[ids], permutations[start : start + batch_size])
        )
    return t.cat(scores)


def summarize_paths(scores, budgets):
    """Unbiased sample SD / sqrt(n) estimates outer Monte Carlo standard error."""
    budgets = list(budgets)
    if scores.ndim != 2 or any(n < 2 or n > len(scores) for n in budgets):
        raise ValueError("Every prefix needs between two and N independent paths.")
    return {
        "mean": t.stack([scores[:n].mean(0) for n in budgets]),
        "se": t.stack([scores[:n].std(0, correction=1) / math.sqrt(n) for n in budgets]),
    }


def prefix_values(game, orders):
    """Value of each remaining-feature prefix for each order; include empty set."""
    d = orders.shape[1]
    if game.shape != (2**d,) or not t.equal(orders.sort(1).values, t.arange(d).expand_as(orders)):
        raise ValueError("Supply one complete game and a matrix of feature permutations.")
    masks = t.zeros(len(orders), dtype=t.int64)
    values = [game[masks]]
    for step in range(d):
        masks = masks | (1 << orders[:, step])
        values.append(game[masks])
    return t.stack(values, dim=1)


# Provided sampling and enumeration plumbing, also displayed in the lesson.
def sampling_schedule(n, n_rows, d, seed):
    rng = np.random.default_rng(seed)
    rows = t.from_numpy(rng.integers(n_rows, size=n))
    permutations = t.from_numpy(np.stack([rng.permutation(d) for _ in range(n)]))
    return rows, permutations


def coalition_predictions(model, x, background):
    with t.no_grad():
        return t.stack(
            [
                marginal_predict(model, x, background, mask.expand_as(x))
                for mask in coalition_masks(x.shape[1])
            ],
            dim=1,
        )


def table_path_contributions(games, rows, permutations):
    masks = t.cat(
        [t.zeros((len(rows), 1), dtype=t.int64), (1 << permutations).cumsum(dim=1)], dim=1
    )
    increments = t.diff(games[rows[:, None], masks], dim=1)
    result = t.empty_like(increments)
    result.scatter_(1, permutations, increments)
    return result
