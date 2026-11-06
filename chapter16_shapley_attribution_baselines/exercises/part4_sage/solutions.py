# %%


MAIN = __name__ == "__main__"

# %%

import itertools
import math
import os
import sys
import json
import hashlib
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch as t
import matplotlib.pyplot as plt

chapter = "chapter16_shapley_attribution_baselines"
root_dir = Path(__file__).resolve().parents[3]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
from chapter16_shapley_attribution_baselines.exercises.part4_sage import tests
from chapter16_shapley_attribution_baselines.exercises.part12_sage_global_importance.sage_data import (
    load_split,
    background_order,
)

# These helpers download/check data and test our result independently.
# Every attribution method and every model forward/training function is implemented below.
from chapter16_shapley_attribution_baselines.exercises.part12_sage_global_importance.sage_reference import (
    checks as reference_checks,
)
from chapter16_shapley_attribution_baselines.exercises.part12_sage_global_importance.run_native import (
    summarize as summarize_run,
    validate_freeze,
)

if MAIN:
    t.set_num_threads(4)
if MAIN:
    t.set_grad_enabled(False)
if MAIN:
    plt.rcParams.update(
        {
            "figure.dpi": 115,
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
        }
    )
BLUE, ORANGE, RED, GRAY = "#247A99", "#C58724", "#B74F53", "#788391"
FEATURES = [
    "Cement",
    "Slag",
    "Fly ash",
    "Water",
    "Superplasticizer",
    "Coarse aggregate",
    "Fine aggregate",
    "Age",
]
if MAIN:
    output_dir = Path(
        os.environ.get("ARENA_SAGE_OUTPUT_DIRECTORY", tempfile.mkdtemp(prefix="arena-sage-"))
    )
if MAIN:
    output_dir.mkdir(parents=True, exist_ok=True)


def show_figure(name):
    plt.gcf().savefig(output_dir / f"{name}.png", bbox_inches="tight")
    plt.show()


if MAIN:
    train = load_split("train")
if MAIN:
    explain = load_split("explain")
if MAIN:
    display(pd.DataFrame(train["x"][:5], columns=FEATURES).assign(strength_MPa=train["y"][:5]))

# %%

def coalition_masks(d):
    """Return Boolean (2**d, d); row m contains the set bits of integer m."""
    if not isinstance(d, int) or not 1 <= d <= 16:
        raise ValueError("Use an integer number of features between 1 and 16.")
    return ((t.arange(2**d)[:, None] >> t.arange(d)) & 1).bool()


if MAIN:
    tests.test_coalition_masks(coalition_masks)

# %%

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


if MAIN:
    tests.test_marginal_predict(marginal_predict)

# %%

if MAIN:
    background_demo = t.tensor([[-2.0], [2.0]], dtype=t.float64)
if MAIN:
    query_demo = t.tensor([[0.0]], dtype=t.float64)
if MAIN:
    nonlinear = lambda z: z[:, 0].square()
if MAIN:
    average_prediction = marginal_predict(
        nonlinear, query_demo, background_demo, t.tensor([[False]])
    ).item()
if MAIN:
    mean_input_prediction = nonlinear(background_demo.mean(0, keepdim=True)).item()
if MAIN:
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
if MAIN:
    ax.bar(
        ["Average predictions", "Predict at mean input"],
        [average_prediction, mean_input_prediction],
        color=[BLUE, ORANGE],
    )
if MAIN:
    ax.axhline(4, color=GRAY, ls="--", label="Exact E[Z²] = 4")
if MAIN:
    ax.set(ylabel="Restricted prediction", title="Averaging and a nonlinear model do not commute")
if MAIN:
    ax.legend()
if MAIN:
    show_figure("sage_averaging")

# %%

def loss_game(predictions, y):
    """(N, 2**D) restricted predictions → per-row improvement in squared loss."""
    if predictions.ndim != 2 or y.shape != (len(predictions),):
        raise ValueError("Expected predictions=(N,M) and targets=(N,).")
    losses = (predictions - y[:, None]).square()
    return losses[:, :1] - losses


if MAIN:
    tests.test_loss_game(loss_game)

# %%

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


if MAIN:
    tests.test_exact_shapley(exact_shapley)

# %%

if MAIN:
    xor_game = t.tensor([0.0, 0.0, 0.0, 0.25], dtype=t.float64)
if MAIN:
    xor_sage = exact_shapley(xor_game)
if MAIN:
    xor_singleton = xor_game[t.tensor([1, 2])]
if MAIN:
    xor_loo = xor_game[-1] - xor_game[t.tensor([2, 1])]
if MAIN:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
if MAIN:
    for offset, values, label, color in [
        (-0.24, xor_sage, "SAGE", BLUE),
        (0, xor_singleton, "Singleton", GRAY),
        (0.24, xor_loo, "Leave one out", ORANGE),
    ]:
        axes[0].bar(np.arange(2) + offset, values, width=0.24, label=label, color=color)
if MAIN:
    axes[0].set(
        xticks=[0, 1],
        xticklabels=["Bit 0", "Bit 1"],
        ylabel="Squared-loss improvement",
        title="Neither bit predicts XOR alone",
    )
if MAIN:
    axes[0].legend(fontsize=8)
if MAIN:
    axes[1].bar(
        ["SAGE", "Singleton", "Leave one out"],
        [xor_sage.sum(), xor_singleton.sum(), xor_loo.sum()],
        color=[BLUE, GRAY, ORANGE],
    )
if MAIN:
    axes[1].axhline(0.25, color=RED, ls="--", label="Actual full improvement")
if MAIN:
    axes[1].set(ylabel="Sum of attributed improvement", title="Only SAGE conserves the total here")
if MAIN:
    axes[1].legend(fontsize=8)
if MAIN:
    fig.tight_layout()
if MAIN:
    show_figure("sage_xor")

# %%

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


if MAIN:
    tests.test_gaussian_game(gaussian_game)

# %%

if MAIN:
    rho_values = t.linspace(0, 1, 101, dtype=t.float64)
if MAIN:
    truth = t.tensor([1.0, 0.0], dtype=t.float64)
if MAIN:
    conditional_values, marginal_values = [], []
if MAIN:
    for rho in rho_values:
        covariance = t.tensor([[1.0, rho], [rho, 1.0]], dtype=t.float64)
        conditional_values.append(exact_shapley(gaussian_game(covariance, truth, truth, True)["game"]))
        marginal_values.append(exact_shapley(gaussian_game(covariance, truth, truth, False)["game"]))
if MAIN:
    conditional_values, marginal_values = t.stack(conditional_values), t.stack(marginal_values)
if MAIN:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
if MAIN:
    for j, name in enumerate(["Read directly by model", "Unused correlated measurement"]):
        axes[j].plot(rho_values, conditional_values[:, j], color=BLUE, label="Conditional")
        axes[j].plot(rho_values, marginal_values[:, j], color=ORANGE, ls="--", label="Marginal")
        axes[j].set(xlabel="Correlation ρ", title=name, ylim=(-0.05, 1.05))
        axes[j].legend()
if MAIN:
    axes[0].set_ylabel("SAGE (squared-loss improvement)")
if MAIN:
    fig.tight_layout()
if MAIN:
    show_figure("sage_correlation")

# %%

if MAIN:
    rho = 0.0
if MAIN:
    bad_weight = 2.0
if MAIN:
    covariance = t.tensor([[1.0, rho], [rho, 1.0]], dtype=t.float64)
if MAIN:
    wrong_model = t.tensor([1.0, bad_weight], dtype=t.float64)
if MAIN:
    play_values = t.stack(
        [
            exact_shapley(gaussian_game(covariance, truth, wrong_model, True)["game"]),
            exact_shapley(gaussian_game(covariance, truth, wrong_model, False)["game"]),
            exact_shapley(gaussian_game(covariance, truth, truth, True)["game"]),
        ]
    )
if MAIN:
    fig, ax = plt.subplots(figsize=(8, 3.7))
if MAIN:
    for i, (label, color) in enumerate(
        zip(
            ["Fixed model, conditional", "Fixed model, marginal", "Optimal model, conditional"],
            [BLUE, ORANGE, GRAY],
        )
    ):
        ax.bar(np.arange(2) + (i - 1) * 0.25, play_values[i], 0.25, label=label, color=color)
if MAIN:
    ax.axhline(0, color="black", lw=0.6)
if MAIN:
    ax.set(
        xticks=[0, 1],
        xticklabels=["X₀", "X₁"],
        ylabel="SAGE",
        title=f"Changing the model's mistake: ρ={rho}, extra coefficient={bad_weight}",
    )
if MAIN:
    ax.legend(fontsize=8)
if MAIN:
    show_figure("sage_gaussian_play")

# %%

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


if MAIN:
    tests.test_path_contributions(path_contributions)

# %%

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


if MAIN:
    tests.test_permutation_sage(permutation_sage)

# %%

def summarize_paths(scores, budgets):
    """Unbiased sample SD / sqrt(n) estimates outer Monte Carlo standard error."""
    budgets = list(budgets)
    if scores.ndim != 2 or any(n < 2 or n > len(scores) for n in budgets):
        raise ValueError("Every prefix needs between two and N independent paths.")
    return {
        "mean": t.stack([scores[:n].mean(0) for n in budgets]),
        "se": t.stack([scores[:n].std(0, correction=1) / math.sqrt(n) for n in budgets]),
    }


if MAIN:
    tests.test_summarize_paths(summarize_paths)

# %%

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


if MAIN:
    tests.test_prefix_values(prefix_values)

# %%

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

# %%

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


if MAIN:
    methods = SimpleNamespace(
        **{
            name: globals()[name]
            for name in [
                "coalition_masks",
                "marginal_predict",
                "loss_game",
                "exact_shapley",
                "gaussian_game",
                "path_contributions",
                "permutation_sage",
                "summarize_paths",
                "prefix_values",
                "sampling_schedule",
                "coalition_predictions",
                "table_path_contributions",
            ]
        }
    )

# %%

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


if MAIN:
    validate_freeze()
if MAIN:
    native_dir = Path(tempfile.mkdtemp(prefix="native-", dir=output_dir))
if MAIN:
    prepared = prepare_study(train, explain, native_dir / "study", methods)
if MAIN:
    (native_dir / "before_audit.json").write_text(
        json.dumps(
            {
                "phase": prepared["phase"],
                "audit_loaded": False,
                "prepared_sha256": hashlib.sha256(
                    (native_dir / "study/prepared.pt").read_bytes()
                ).hexdigest(),
            },
            indent=2,
        )
        + "\n"
    )
if MAIN:
    print(f"Trained six networks and one ridge model on {len(train['y'])} rows.")
if MAIN:
    print("Feature rankings are fixed from 153 explanation rows before audit loading.")

# %%

if MAIN:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.5), sharey=True)
if MAIN:
    for index, model in enumerate(prepared["models"][:6]):
        axis = axes[index // 3]
        axis.plot(np.arange(1, 801), model["loss"], lw=1.3, label=f"Seed {index%3}")
        axis.set(
            xlabel="Adam update",
            yscale="log",
            title=["Correct training targets", "Permuted training targets"][index // 3],
        )
        axis.legend(fontsize=8)
if MAIN:
    axes[0].set_ylabel("Training MSE / training target variance")
if MAIN:
    fig.tight_layout()
if MAIN:
    show_figure("sage_training")

# %%

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


if MAIN:
    audit = evaluate_study(prepared, load_split("audit"), methods)
if MAIN:
    t.save(audit, native_dir / "study/evaluated.pt")
if MAIN:
    audit_mse = t.stack(
        [(result["predictions"][:, -1] - audit["y"]).square().mean() for result in audit["models"]]
    )
if MAIN:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
if MAIN:
    axes[0].scatter(
        audit["y"],
        audit["models"][0]["predictions"][:, -1],
        alpha=0.65,
        s=20,
        color=BLUE,
        label="Clean seed 0",
    )
if MAIN:
    limits = [float(audit["y"].min()) - 3, float(audit["y"].max()) + 3]
if MAIN:
    axes[0].plot(limits, limits, ls="--", color=GRAY, label="Perfect prediction")
if MAIN:
    axes[0].axhline(
        float(prepared["train_y"].mean()), color=ORANGE, ls=":", label="Training-mean baseline"
    )
if MAIN:
    axes[0].set(
        xlabel="Measured strength (MPa)",
        ylabel="Predicted strength (MPa)",
        title="Predictions on 119 audit rows",
    )
if MAIN:
    axes[0].legend(fontsize=8)
if MAIN:
    labels = ["Clean 0", "Clean 1", "Clean 2", "Shuffle 0", "Shuffle 1", "Shuffle 2", "Ridge"]
if MAIN:
    axes[1].bar(np.arange(7), audit_mse, color=[BLUE] * 3 + [RED] * 3 + [ORANGE])
if MAIN:
    axes[1].axhline(
        float(audit["training_mean_mse"]), color=GRAY, ls="--", label="Training-mean target"
    )
if MAIN:
    axes[1].set(
        xticks=np.arange(7),
        xticklabels=labels,
        ylabel="Audit MSE (MPa²)",
        title="Keep all seeds and the failed controls",
    )
if MAIN:
    axes[1].tick_params(axis="x", rotation=40)
if MAIN:
    axes[1].legend(fontsize=8)
if MAIN:
    fig.tight_layout()
if MAIN:
    show_figure("sage_model_performance")

# %%

if MAIN:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
if MAIN:
    positions = np.arange(8)
if MAIN:
    for i in range(3):
        axes[0].plot(
            positions, prepared["explanations"][i]["sage"], marker="o", lw=1, label=f"Clean seed {i}"
        )
if MAIN:
    axes[0].plot(
        positions, prepared["explanations"][6]["sage"], marker="s", lw=1, color=GRAY, label="Ridge"
    )
if MAIN:
    axes[0].set_title("Which features help the fitted models?")
if MAIN:
    for i in range(3):
        axes[1].plot(
            positions,
            prepared["explanations"][i + 3]["sage"],
            marker="o",
            lw=1,
            label=f"Shuffled seed {i}",
        )
if MAIN:
    axes[1].set_title("Training on shuffled targets changes the game")
if MAIN:
    for axis in axes:
        axis.axhline(0, color="black", lw=0.6)
        axis.set(xticks=positions, xticklabels=FEATURES, ylabel="Explanation SAGE (MPa²)")
        axis.tick_params(axis="x", rotation=60)
        axis.legend(fontsize=8)
if MAIN:
    fig.tight_layout()
if MAIN:
    show_figure("sage_global_importance")

# %%

if MAIN:
    primary = prepared["explanations"][0]
if MAIN:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
if MAIN:
    for axis, key, unit, title in zip(
        axes,
        ["local_prediction", "local_loss"],
        ["MPa", "MPa²"],
        ["What changes the prediction?", "What improves that row's loss?"],
    ):
        values = primary[key][:24].numpy()
        limit = np.max(np.abs(values))
        im = axis.imshow(values, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
        axis.set(
            xticks=np.arange(8),
            xticklabels=FEATURES,
            ylabel="Explanation row, in source order",
            title=title,
        )
        axis.tick_params(axis="x", rotation=60)
        fig.colorbar(im, ax=axis, label=unit, shrink=0.8)
if MAIN:
    fig.tight_layout()
if MAIN:
    show_figure("sage_local_explanations")

# %%

if MAIN:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
if MAIN:
    axes[0].bar(positions - 0.18, primary["sage"], 0.36, color=BLUE, label="Global SAGE")
if MAIN:
    axes[0].bar(
        positions + 0.18,
        primary["local_loss"].mean(0),
        0.36,
        color=ORANGE,
        label="Mean local loss Shapley",
    )
if MAIN:
    axes[0].set(ylabel="Loss improvement (MPa²)", title="The exact local-to-global identity")
if MAIN:
    axes[1].bar(
        positions - 0.18, primary["mean_prediction"], 0.36, color=BLUE, label="Mean prediction Shapley"
    )
if MAIN:
    axes[1].bar(
        positions + 0.18,
        primary["mean_abs_prediction"],
        0.36,
        color=ORANGE,
        label="Mean absolute prediction Shapley",
    )
if MAIN:
    axes[1].set(ylabel="Prediction change (MPa)", title="Two different output summaries")
if MAIN:
    for axis in axes:
        axis.axhline(0, color="black", lw=0.6)
        axis.set(xticks=positions, xticklabels=FEATURES)
        axis.tick_params(axis="x", rotation=60)
        axis.legend(fontsize=8)
if MAIN:
    fig.tight_layout()
if MAIN:
    show_figure("sage_local_global")
if MAIN:
    error = (primary["local_loss"].mean(0) - primary["sage"]).abs().max()
if MAIN:
    print(f"Mean local loss Shapley versus global SAGE: maximum difference {error:.2e} MPa².")

# %%

if MAIN:
    mask_matrix = coalition_masks(8).to(t.float64)
if MAIN:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), sharex=True, sharey=True)
if MAIN:
    for axis, key, title in zip(
        axes, ["sage", "singleton", "leave_one_out"], ["SAGE", "Singleton", "Leave one out"]
    ):
        approximation = mask_matrix @ primary[key]
        axis.scatter(primary["game"], approximation, s=9, alpha=0.45, color=BLUE)
        lo = min(float(primary["game"].min()), float(approximation.min()))
        hi = max(float(primary["game"].max()), float(approximation.max()))
        axis.plot([lo, hi], [lo, hi], color=GRAY, ls="--")
        axis.set(title=title, xlabel="Actual coalition improvement (MPa²)")
if MAIN:
    axes[0].set_ylabel("Sum of included feature scores (MPa²)")
if MAIN:
    fig.tight_layout()
if MAIN:
    show_figure("sage_additive_approximation")

# %%

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


if MAIN:
    diagnostic = diagnostics(prepared, methods)
if MAIN:
    t.save(diagnostic, native_dir / "study/diagnostics.pt")
if MAIN:
    t.testing.assert_close(
        diagnostic["direct_scores"], diagnostic["table_scores"], rtol=1e-7, atol=1e-8
    )
if MAIN:
    assert diagnostic["dummy_sage"][-1] == 0
if MAIN:
    print(
        "Direct sampled model paths agree with the enumerated game; ignored ninth input has zero SAGE."
    )

# %%

if MAIN:
    budget = diagnostic["budgets"].numpy()
if MAIN:
    sampling_error = diagnostic["sampling_means"] - primary["sage"][None, None, :]
if MAIN:
    rmse = sampling_error.square().mean(dim=(0, 2)).sqrt()
if MAIN:
    rms_se = diagnostic["sampling_se"].square().mean(dim=(0, 2)).sqrt()
if MAIN:
    direct_summary = summarize_paths(diagnostic["direct_scores"], [4096])
if MAIN:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
if MAIN:
    axes[0].loglog(budget, rmse, "o-", color=BLUE, label="Actual RMSE across 32 schedules")
if MAIN:
    axes[0].loglog(budget, rms_se, "s--", color=ORANGE, label="RMS reported standard error")
if MAIN:
    axes[0].loglog(
        budget, float(rmse[0]) * np.sqrt(budget[0] / budget), ":", color=GRAY, label="1/√n reference"
    )
if MAIN:
    axes[0].set(
        xlabel="Independent row/permutation paths",
        ylabel="MPa²",
        title="Outer sampling approaches the exact finite game",
    )
if MAIN:
    axes[0].legend(fontsize=8)
if MAIN:
    axes[1].errorbar(
        positions,
        direct_summary["mean"][0],
        yerr=1.96 * direct_summary["se"][0],
        fmt="o",
        capsize=3,
        color=BLUE,
        label="4096 direct paths; ±1.96 SE",
    )
if MAIN:
    axes[1].scatter(positions, primary["sage"], marker="x", color=RED, label="All 256 coalitions")
if MAIN:
    axes[1].set(
        xticks=positions,
        xticklabels=FEATURES,
        ylabel="SAGE (MPa²)",
        title="Exact values are a reference, not a stopping rule",
    )
if MAIN:
    axes[1].tick_params(axis="x", rotation=60)
if MAIN:
    axes[1].legend(fontsize=8)
if MAIN:
    fig.tight_layout()
if MAIN:
    show_figure("sage_sampling")

# %%

if MAIN:
    variant_values = t.stack([result["sage"] for result in diagnostic["variants"]])
if MAIN:
    deltas = variant_values[:4] - primary["sage"]
if MAIN:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
if MAIN:
    limit = float(deltas.abs().max())
if MAIN:
    im = axes[0].imshow(deltas, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
if MAIN:
    axes[0].set(
        yticks=np.arange(4),
        yticklabels=[
            "16 background rows",
            "128 background rows",
            "Other 64-row set A",
            "Other 64-row set B",
        ],
        xticks=positions,
        xticklabels=FEATURES,
        title="Change from the primary 64-row background",
    )
if MAIN:
    axes[0].tick_params(axis="x", rotation=60)
if MAIN:
    fig.colorbar(im, ax=axes[0], label="Δ SAGE (MPa²)", shrink=0.8)
if MAIN:
    axes[1].bar(positions - 0.18, primary["sage"], 0.36, color=BLUE, label="Observed outcomes")
if MAIN:
    axes[1].bar(
        positions + 0.18,
        diagnostic["shuffled_outcomes"]["sage"],
        0.36,
        color=RED,
        label="Permuted outcomes",
    )
if MAIN:
    axes[1].axhline(0, color="black", lw=0.6)
if MAIN:
    axes[1].set(
        xticks=positions,
        xticklabels=FEATURES,
        ylabel="SAGE (MPa²)",
        title="Same predictions, different predictive usefulness",
    )
if MAIN:
    axes[1].tick_params(axis="x", rotation=60)
if MAIN:
    axes[1].legend(fontsize=8)
if MAIN:
    fig.tight_layout()
if MAIN:
    show_figure("sage_background_and_targets")

# %%

if MAIN:
    mean_imputed = diagnostic["variants"][-1]
if MAIN:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
if MAIN:
    axes[0].scatter(
        primary["predictions"].flatten(),
        mean_imputed["predictions"].flatten(),
        s=2,
        alpha=0.1,
        color=BLUE,
        rasterized=True,
    )
if MAIN:
    lo = min(float(primary["predictions"].min()), float(mean_imputed["predictions"].min()))
if MAIN:
    hi = max(float(primary["predictions"].max()), float(mean_imputed["predictions"].max()))
if MAIN:
    axes[0].plot([lo, hi], [lo, hi], color=GRAY, ls="--")
if MAIN:
    axes[0].set(
        xlabel="Mean of donor predictions (MPa)",
        ylabel="Prediction at mean missing inputs (MPa)",
        title="Averaging and the real network do not commute",
    )
if MAIN:
    axes[1].bar(positions - 0.18, primary["sage"], 0.36, color=BLUE, label="Average predictions")
if MAIN:
    axes[1].bar(
        positions + 0.18, mean_imputed["sage"], 0.36, color=ORANGE, label="Predict at mean inputs"
    )
if MAIN:
    axes[1].axhline(0, color="black", lw=0.6)
if MAIN:
    axes[1].set(
        xticks=positions,
        xticklabels=FEATURES,
        ylabel="SAGE (MPa²)",
        title="The changed restriction reallocates credit",
    )
if MAIN:
    axes[1].tick_params(axis="x", rotation=60)
if MAIN:
    axes[1].legend(fontsize=8)
if MAIN:
    fig.tight_layout()
if MAIN:
    show_figure("sage_mean_imputation")

# %%

if MAIN:
    rng = np.random.default_rng(161340)
if MAIN:
    inner_sizes = np.array([1, 4, 16, 64])
if MAIN:
    inner_loss, inner_se = [], []
if MAIN:
    for m in inner_sizes:
        sample_prediction = (1 + 2 * rng.integers(2, size=(32768, m))).mean(1)
        sampled_loss = sample_prediction**2
        inner_loss.append(sampled_loss.mean())
        inner_se.append(sampled_loss.std(ddof=1) / np.sqrt(len(sampled_loss)))
if MAIN:
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
if MAIN:
    ax.errorbar(
        inner_sizes,
        inner_loss,
        yerr=1.96 * np.array(inner_se),
        fmt="o",
        capsize=4,
        color=BLUE,
        label="32768 outer repetitions",
    )
if MAIN:
    ax.plot(
        inner_sizes,
        4 + 1 / inner_sizes,
        color=ORANGE,
        ls="--",
        label="Exact biased expectation: 4 + 1/m",
    )
if MAIN:
    ax.axhline(4, color=GRAY, ls=":", label="Loss of exact restricted prediction")
if MAIN:
    ax.set(
        xscale="log",
        xlabel="Inner samples m",
        ylabel="Expected squared loss",
        title="Small outer error bars can exclude the intended target",
    )
if MAIN:
    ax.legend(fontsize=8)
if MAIN:
    show_figure("sage_inner_bias")

# %%

if MAIN:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
if MAIN:
    correlation = np.corrcoef(prepared["train_x"].numpy(), rowvar=False)
if MAIN:
    im = axes[0].imshow(correlation, vmin=-1, vmax=1, cmap="RdBu_r")
if MAIN:
    axes[0].set(
        xticks=positions,
        yticks=positions,
        xticklabels=FEATURES,
        yticklabels=FEATURES,
        title="Training inputs are dependent",
    )
if MAIN:
    axes[0].tick_params(axis="x", rotation=60)
if MAIN:
    fig.colorbar(im, ax=axes[0], label="Pearson correlation", shrink=0.8)
if MAIN:
    for name, values in diagnostic["nearest_train_distance"].items():
        axes[1].plot(
            values.sort().values,
            np.arange(1, len(values) + 1) / len(values),
            label=name.replace("_", " "),
        )
if MAIN:
    axes[1].set(
        xlabel="Nearest training row: Euclidean distance in standardized inputs",
        ylabel="Fraction at or below distance",
        title="Original rows versus marginal hybrids",
    )
if MAIN:
    axes[1].legend(fontsize=8)
if MAIN:
    fig.tight_layout()
if MAIN:
    show_figure("sage_missingness")

# %%

if MAIN:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
if MAIN:
    axes[0].bar(positions - 0.18, primary["sage"], 0.36, color=BLUE, label="Equal row weight")
if MAIN:
    axes[0].bar(
        positions + 0.18, diagnostic["recipe_sage"], 0.36, color=ORANGE, label="Equal recipe weight"
    )
if MAIN:
    axes[0].set_title("What does an average example mean?")
if MAIN:
    axes[1].bar(positions - 0.18, primary["sage"], 0.36, color=BLUE, label="Explanation recipes")
if MAIN:
    axes[1].bar(positions + 0.18, audit["models"][0]["sage"], 0.36, color=ORANGE, label="Audit recipes")
if MAIN:
    axes[1].set_title("Same model/background; different recipes")
if MAIN:
    for axis in axes:
        axis.axhline(0, color="black", lw=0.6)
        axis.set(xticks=positions, xticklabels=FEATURES, ylabel="SAGE (MPa²)")
        axis.tick_params(axis="x", rotation=60)
        axis.legend(fontsize=8)
if MAIN:
    fig.tight_layout()
if MAIN:
    show_figure("sage_weighting_and_split")

# %%

if MAIN:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
if MAIN:
    for axis, index in zip(axes.flat, [0, 1, 2, 6]):
        curves = audit["models"][index]["prefix_mse"].numpy()
        random = curves[3:]
        axis.fill_between(
            np.arange(9),
            np.quantile(random, 0.1, axis=0),
            np.quantile(random, 0.9, axis=0),
            color=GRAY,
            alpha=0.18,
            label="Random-order 10–90%",
        )
        axis.plot(np.arange(9), random.mean(0), color=GRAY, ls="--", label="Mean random order")
        for j, (name, color) in enumerate(
            [("Top SAGE", BLUE), ("Bottom SAGE", RED), ("Top leave-one-out", ORANGE)]
        ):
            axis.plot(np.arange(9), curves[j], "o-", markersize=3, label=name, color=color)
        axis.set(
            title=f"Clean seed {index}" if index < 3 else "Ridge",
            xlabel="Number of available features",
            ylabel="Audit MSE (MPa²)",
            xticks=np.arange(9),
        )
if MAIN:
    axes[0, 0].legend(fontsize=7)
if MAIN:
    fig.tight_layout()
if MAIN:
    show_figure("sage_audit_prefixes")

# %%

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


if MAIN:
    row_index = 0
if MAIN:
    feature = 0
if MAIN:
    background_size = 64
if MAIN:
    play = native_play(prepared, audit, methods, row_index, feature, background_size)
if MAIN:
    t.save(play, native_dir / "study/play.pt")
if MAIN:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
if MAIN:
    axes[0].hist(play["hybrid_predictions"], bins=15, color=BLUE, alpha=0.75)
if MAIN:
    axes[0].axvline(float(play["target"][0]), color=RED, ls="--", label="Measured target")
if MAIN:
    axes[0].axvline(
        float(play["explanation"]["predictions"][0, -1]), color=ORANGE, label="Full prediction"
    )
if MAIN:
    axes[0].set(
        xlabel="Prediction after donor replacement (MPa)",
        ylabel="Donor rows",
        title=f"Replace {FEATURES[feature]}: source row {play['source_id']}",
    )
if MAIN:
    axes[0].legend(fontsize=8)
if MAIN:
    for axis, key, unit, title in zip(
        axes[1:],
        ["local_prediction", "local_loss"],
        ["MPa", "MPa²"],
        ["What raises this prediction?", "What reduces this row's error?"],
    ):
        values = play["explanation"][key][0]
        axis.bar(positions, values, color=[BLUE if v >= 0 else RED for v in values])
        axis.axhline(0, color="black", lw=0.6)
        axis.set(xticks=positions, xticklabels=FEATURES, ylabel=unit, title=title)
        axis.tick_params(axis="x", rotation=65)
if MAIN:
    fig.tight_layout()
if MAIN:
    show_figure("sage_native_play")

# %%

if MAIN:
    reference = reference_checks(prepared, methods, make_predictor(prepared["models"][0]))
if MAIN:
    print(f"All {reference['permutations']:,} feature orderings agree with exact SAGE.")
if MAIN:
    print("Constrained weighted regression, NumPy network predictions, and the author imputer agree.")
if MAIN:
    summary = summarize_run(prepared, audit, diagnostic, reference)
if MAIN:
    summary["native_directory"] = str(native_dir / "study")
if MAIN:
    (output_dir / "native_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

# %%
