"""Readable study orchestration, to be embedded in the native learner.

All taught operations are supplied through ``m`` so the notebook uses its own
student implementations. Fitting accepts training/ranking data only. Final
queries have a separate evaluation entry point and never select deletion sets.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

L2 = .02
ORDER_SEEDS = (0, 1, 2)
RANDOM_SEEDS = tuple(range(1000, 1032))
BUDGETS = (2, 4, 8, 16, 32)
DAMPING = (0., .01, .1, 1.)
STRENGTHS = (-1., -.5, -.1, -.01, .01, .1, .5, 1.)
CHECKPOINT_CHOICES = {
    "all": slice(None), "early": slice(0, 20), "late": slice(20, None), "last": slice(-1, None),
}


def query_outputs(theta, queries):
    """Actual classifier probabilities and unreduced query losses for each fit."""
    logits = theta @ queries["x"].T
    losses = F.softplus(logits) - queries["y"][None, :] * logits
    return {"logits": logits, "probabilities": torch.sigmoid(logits), "losses": losses,
            "correct": (logits >= 0) == queries["y"][None, :].bool()}


def signed_pixel_similarity(train, train_y, queries):
    """A labelled image-retrieval control, without model loss or curvature."""
    pixels = F.normalize(train["x"][:, :-1], dim=1)
    query_pixels = F.normalize(queries["x"][:, :-1], dim=1)
    agreement = (2 * train_y[:, None] - 1) * (2 * queries["y"][None, :] - 1)
    return (pixels @ query_pixels.T) * agreement


def checkpoint_analysis(trace, train, train_y, queries, m):
    n = len(train_y)
    checkpoints = trace["states"][n::n]
    rates = trace["rates"][n - 1::n]
    train_gradients = torch.stack([m.logistic_gradients(theta, train["x"], train_y, L2) for theta in checkpoints])
    query_gradients = torch.stack([m.logistic_gradients(theta, queries["x"], queries["y"]) for theta in checkpoints])
    scores = {}
    for name, selected in CHECKPOINT_CHOICES.items():
        scores[name] = m.tracin_checkpoint_scores(train_gradients[selected], query_gradients[selected], rates[selected])
    ghost_products = []
    for theta, query_gradient in zip(checkpoints, query_gradients):
        train_delta = torch.sigmoid(train["x"] @ theta) - train_y
        query_delta = torch.sigmoid(queries["x"] @ theta) - queries["y"]
        raw = m.ghost_linear_products(
            train["x"][:, :-1], train_delta[:, None], queries["x"][:, :-1], query_delta[:, None], bias=True,
        )
        # Training rows include the regularizer; query prediction loss does not.
        regularizer_product = (L2 * theta @ query_gradient.T)[None, :]
        ghost_products.append(raw + regularizer_product)
    ghost_products = torch.stack(ghost_products)
    dense_products = torch.einsum("knp,kqp->knq", train_gradients, query_gradients)
    ghost_error = float((ghost_products - dense_products).abs().max())
    if ghost_error > 1e-10:
        raise ValueError(f"Factored gradient products differ from complete parameter gradients: {ghost_error}.")
    ghost_scores = (ghost_products * rates[:, None, None]).sum(0)
    torch.testing.assert_close(ghost_scores, scores["all"], rtol=1e-10, atol=1e-11)
    return {"scores": scores, "ghost_all_scores": ghost_scores, "ghost_max_error": ghost_error,
            "checkpoints": checkpoints, "checkpoint_rates": rates}


def score_queries(fits, train, train_y, queries, m):
    """Compute all measurements at already fitted parameters; never refit here."""
    n = len(train_y)
    theta = fits["base"]["theta"][0]
    unit_weights = torch.ones(1, n, dtype=theta.dtype)
    _, hessian = m.objective_derivatives(theta[None], train["x"], train_y, unit_weights, L2)
    train_gradient = m.logistic_gradients(theta, train["x"], train_y, L2)
    raw_train_gradient = m.logistic_gradients(theta, train["x"], train_y)
    query_gradient = m.logistic_gradients(theta, queries["x"], queries["y"])
    baseline = query_outputs(theta[None], queries)
    loo = query_outputs(fits["loo"]["theta"], queries)
    perturbations = query_outputs(fits["perturbations"]["theta"], queries)
    removal_predictions = torch.stack([
        -m.upweight_influence(hessian[0], train_gradient, query_gradient, damping) / n for damping in DAMPING
    ])
    wrong_normalization = -m.upweight_influence(hessian[0], raw_train_gradient, query_gradient) / n
    convergence = []
    last = len(fits["base"]["states"]) - 1
    state_indices = [0, min(1, last), min(3, last), last]
    for index in state_indices:
        current = fits["base"]["states"][index, 0]
        _, h = m.objective_derivatives(current[None], train["x"], train_y, unit_weights, L2)
        g_train = m.logistic_gradients(current, train["x"], train_y, L2)
        g_query = m.logistic_gradients(current, queries["x"], queries["y"])
        convergence.append(m.upweight_influence(h[0], g_train, g_query))
    losses = perturbations["losses"].reshape(len(fits["probe_indices"]), len(STRENGTHS), -1)
    finite_derivative = (losses[:, 4] - losses[:, 3]) / (2 * .01 / n)
    paths = []
    for trace in fits["paths"]:
        accounting = m.trace_loss_reductions(
            trace["states"], trace["used_ids"], trace["rates"], train["x"], train_y,
            queries["x"], queries["y"], L2,
        )
        endpoint = query_outputs(trace["states"][[0, -1]], queries)
        actual_change = endpoint["losses"][0] - endpoint["losses"][1]
        torch.testing.assert_close(accounting["exact"].sum(0), actual_change, rtol=1e-10, atol=1e-11)
        checkpoints = checkpoint_analysis(trace, train, train_y, queries, m)
        paths.append({"accounting": accounting, "checkpoints": checkpoints, "endpoints": endpoint})
    return {
        "baseline": baseline, "loo": loo, "loo_loss_changes": loo["losses"] - baseline["losses"],
        "perturbations": perturbations, "finite_derivative": finite_derivative,
        "removal_predictions": removal_predictions, "wrong_normalization": wrong_normalization,
        "convergence_upweight": torch.stack(convergence), "convergence_state_indices": state_indices,
        "gradient_similarity": train_gradient @ query_gradient.T,
        "signed_pixel_similarity": signed_pixel_similarity(train, train_y, queries), "paths": paths,
        "query_ids": queries["ids"],
    }


def fit_condition(train, ranking, probe_indices, train_y, m, directory, progress):
    n, p = train["x"].shape
    dtype = train["x"].dtype
    unit_weights = torch.ones(1, n, dtype=dtype)
    fits = {"base": m.fit_weighted_logistic(train["x"], train_y, unit_weights, L2)}
    progress("Converged full-data model")
    loo_weights = torch.ones(n, n, dtype=dtype) - torch.eye(n, dtype=dtype)
    fits["loo"] = m.fit_weighted_logistic(train["x"], train_y, loo_weights, L2)
    perturbation_weights = torch.ones(len(probe_indices) * len(STRENGTHS), n, dtype=dtype)
    for row, (index, strength) in enumerate((i, s) for i in probe_indices for s in STRENGTHS):
        perturbation_weights[row, index] += strength
    fits["perturbations"] = m.fit_weighted_logistic(train["x"], train_y, perturbation_weights, L2)
    fits["probe_indices"] = list(probe_indices)
    fits["perturbation_strengths"] = torch.tensor(STRENGTHS, dtype=dtype)
    torch.save(fits, directory / "reference_fits.pt")
    progress("Refitted all 128 leave-one-out and 96 weight-perturbation models")
    individual_errors = []
    for index in (0, 1, 2):
        individual = m.fit_weighted_logistic(train["x"], train_y, loo_weights[index:index + 1], L2)
        error = float((individual["theta"][0] - fits["loo"]["theta"][index]).abs().max())
        if error > 1e-8:
            raise ValueError(f"Batched and separate refits differ for deletion row {index}: {error}.")
        individual_errors.append(error)
    fits["individual_refit_errors"] = torch.tensor(individual_errors, dtype=dtype)
    fits["paths"] = []
    epoch_rates = torch.cat([torch.full((20,), .05, dtype=dtype), torch.full((20,), .01, dtype=dtype)])
    for seed in ORDER_SEEDS:
        generator = torch.Generator().manual_seed(seed)
        orders = torch.stack([torch.randperm(n, generator=generator) for _ in range(40)])
        trace = m.sgd_trace(train["x"], train_y, orders, epoch_rates, L2)
        end = trace["states"][-1]
        gradient = m.logistic_gradients(end, train["x"], train_y, L2).mean(0)
        trace["seed"] = seed
        trace["final_gradient_max"] = gradient.abs().max()
        trace["distance_to_converged"] = torch.linalg.vector_norm(end - fits["base"]["theta"][0])
        fits["paths"].append(trace)
        torch.save(trace, directory / f"sgd_seed_{seed}.pt")
        progress(f"Recorded all 5120 actual SGD updates for order seed {seed}")
    measured = score_queries(fits, train, train_y, ranking, m)
    rankers = {
        "influence": measured["removal_predictions"][0].mean(1),
        "gradient": measured["gradient_similarity"].mean(1),
        "pixels": measured["signed_pixel_similarity"].mean(1),
        "ranking_loo": measured["loo_loss_changes"].mean(1),
    }
    for seed, path in zip(ORDER_SEEDS, measured["paths"]):
        for name, scores in path["checkpoints"]["scores"].items():
            rankers[f"tracin_seed{seed}_{name}"] = scores.mean(1)
    for seed in RANDOM_SEEDS:
        rankers[f"random_{seed}"] = torch.rand(n, generator=torch.Generator().manual_seed(seed), dtype=dtype)
    policies = []
    deletion_weights = []
    for name, scores in rankers.items():
        for budget in BUDGETS:
            removed = m.class_matched_removals(scores, train_y, budget)
            weights = torch.ones(n, dtype=dtype)
            weights[removed] = 0
            deletion_weights.append(weights)
            policies.append({"ranker": name, "budget": budget, "removed": removed,
                             "removed_source_ids": [train["ids"][i] for i in removed.tolist()]})
    if len(rankers) != 48 or len(policies) != 240:
        raise ValueError("Every prescribed ranker and deletion budget must remain in the study.")
    deletion_weights = torch.stack(deletion_weights)
    fits["deletions"] = m.fit_weighted_logistic(train["x"], train_y, deletion_weights, L2)
    fits["policies"], fits["rankers"] = policies, rankers
    fits["deletion_weights"] = deletion_weights
    measured["deletions"] = query_outputs(fits["deletions"]["theta"], ranking)
    cg_checks = []
    theta = fits["base"]["theta"][0]
    _, hessian = m.objective_derivatives(theta[None], train["x"], train_y, unit_weights, L2)
    for index in (0, 1):
        rhs = m.logistic_gradients(theta, ranking["x"][index:index + 1], ranking["y"][index:index + 1])[0]
        direction, history = m.conjugate_gradient(
            lambda v: m.logistic_hvp(theta, train["x"], train_y, unit_weights[0], v, L2), rhs,
        )
        torch.testing.assert_close(direction, torch.linalg.solve(hessian[0], rhs), rtol=1e-8, atol=1e-9)
        cg_checks.append({"query_index": index, "direction": direction, "residuals": history})
    fits["cg_checks"] = cg_checks
    fits["train_y"] = train_y.clone()
    fits["stage"] = "FITS_AND_RANKING_COMPLETE"
    torch.save(fits, directory / "complete_fits.pt")
    torch.save(measured, directory / "ranking_measurements.pt")
    progress("All 48 policies and 240 deletion refits complete; final queries remain unevaluated")
    return fits, measured


def prepare_study(data, m, directory, progress=lambda message: None):
    """Fit every condition and choose policies using ranking queries only."""
    train, ranking = data["train"], data["ranking"]
    fitted, measured = {}, {}
    for name, labels in [("clean", train["y"]), ("corrupted", train["corrupted_y"])]:
        condition_directory = directory / name
        condition_directory.mkdir(parents=True, exist_ok=False)
        def report(message):
            progress(f"{name}: {message}")
        fitted[name], measured[name] = fit_condition(
            train, ranking, data["upweight_probe_indices"], labels, m, condition_directory, report,
        )
    return fitted, measured


def evaluate_final(fitted, train, final_queries, m, directory):
    """Evaluate already frozen fits and policies, after both conditions finish."""
    if set(fitted) != {"clean", "corrupted"}:
        raise ValueError("Both conditions must finish before final evaluation.")
    if any(value.get("stage") != "FITS_AND_RANKING_COMPLETE" for value in fitted.values()):
        raise ValueError("Final evaluation cannot run before every fit and deletion policy is frozen.")
    final = {}
    for name, fits in fitted.items():
        final[name] = score_queries(fits, train, fits["train_y"], final_queries, m)
        final[name]["deletions"] = query_outputs(fits["deletions"]["theta"], final_queries)
        torch.save(final[name], directory / name / "final_measurements.pt")
    return final
