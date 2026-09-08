"""Editable single-query deletion experiment, displayed in the learner."""
import torch


def query_deletion_play(train, query, fits, m, budget=4, l2=.02):
    """Choose by this query, then refit the chosen set and 32 matched controls.

    This is an exploratory query-specific intervention, not a held-out policy
    evaluation. It does not change any reference-study fits or deletion policies.
    """
    n = len(train["x"])
    theta, labels = fits["base"]["theta"][0], fits["train_y"]
    _, hessian = m.objective_derivatives(
        theta[None], train["x"], labels, torch.ones(1, n, dtype=theta.dtype), l2,
    )
    scores = -m.upweight_influence(
        hessian[0], m.logistic_gradients(theta, train["x"], labels, l2),
        m.logistic_gradients(theta, query["x"], query["y"]),
    )[:, 0] / n
    selected = [m.class_matched_removals(scores, labels, budget)]
    for seed in range(1000, 1032):
        random_scores = torch.rand(n, generator=torch.Generator().manual_seed(seed), dtype=theta.dtype)
        selected.append(m.class_matched_removals(random_scores, labels, budget))
    weights = torch.ones(33, n, dtype=theta.dtype)
    for row, removed in enumerate(selected):
        weights[row, removed] = 0
    refits = m.fit_weighted_logistic(train["x"], labels, weights, l2)
    all_theta = torch.cat([theta[None], refits["theta"]])
    logits = all_theta @ query["x"].T
    losses = torch.nn.functional.softplus(logits) - query["y"][None, :] * logits
    return {
        "budget": budget, "query_ids": query["ids"], "scores": scores,
        "selected": selected, "weights": weights, "refits": refits,
        "probabilities": torch.sigmoid(logits), "losses": losses,
        "loss_changes": losses[1:] - losses[:1],
    }
