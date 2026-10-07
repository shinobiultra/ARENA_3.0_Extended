# %%


import functools
import hashlib
import json
import sys
import textwrap
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int
from scipy import stats
from torch import Tensor

# Everything in this section runs on a CPU in a few seconds, so we don't use a GPU at all. Using a single thread makes
# the float64 arithmetic (and so every number in the expected outputs) reproducible.
t.set_num_threads(1)

# Make sure exercises are in the path
chapter = "chapter10_capstone_research_sprint"
section = "part1_capstone_research_sprint"
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part1_capstone_research_sprint.tests as tests
import part1_capstone_research_sprint.utils as utils

MAIN = __name__ == "__main__"

# %%

if MAIN:
    DIGITS = (3, 8)
    x, y, rows = utils.load_digit_pair(DIGITS)
    print(f"{len(x)} images: {(y == 0).sum().item()} {DIGITS[0]}s and {(y == 1).sum().item()} {DIGITS[1]}s, {x.shape[1]} pixels each")

# %%

def split_examples(
    labels: np.ndarray | Tensor, train_fraction: float = 0.5, dev_fraction: float = 0.25, seed: int = 0
) -> dict[str, np.ndarray]:
    """
    Stratified split of example indices into "train", "dev" and "confirm" sets.
    """
    labels = np.asarray(labels)
    if not (train_fraction > 0 and dev_fraction > 0 and train_fraction + dev_fraction < 1):
        raise ValueError(f"Fractions must be positive and leave data for confirmation: {train_fraction=}, {dev_fraction=}")
    rng = np.random.default_rng(seed)
    parts = {"train": [], "dev": [], "confirm": []}
    for label in (0, 1):
        indices = rng.permutation(np.flatnonzero(labels == label))
        n_train, n_dev = int(len(indices) * train_fraction), int(len(indices) * dev_fraction)
        parts["train"].append(indices[:n_train])
        parts["dev"].append(indices[n_train : n_train + n_dev])
        parts["confirm"].append(indices[n_train + n_dev :])
    return {name: np.concatenate(chunks) for name, chunks in parts.items()}


if MAIN:
    tests.test_split_examples(split_examples)

# %%

if MAIN:
    splits = split_examples(y, train_fraction=0.5, dev_fraction=0.25, seed=17)
    for name, ids in splits.items():
        print(f"{name:>8}: {len(ids):3d} images ({(y[ids] == 0).sum().item()} {DIGITS[0]}s, {(y[ids] == 1).sum().item()} {DIGITS[1]}s)")
    
    train_ids = splits["train"]
    shown = np.concatenate([train_ids[y[train_ids] == 0][:8], train_ids[y[train_ids] == 1][:8]])
    utils.plot_digit_grid(
        x[shown], [f"{DIGITS[y[i].item()]} (row {rows[i]})" for i in shown], "Some training images (the only images we look at for now)"
    )

# %%

class DigitClassifier(nn.Module):
    """
    A two-class digit classifier. `kind="mlp"`: 64 pixels -> `hidden_width` tanh units -> 2 logits.
    `kind="linear"`: 64 pixels -> 2 logits (the "hidden representation" is the pixels themselves).
    """

    def __init__(self, kind: str = "mlp", hidden_width: int = 32):
        super().__init__()
        if kind == "mlp":
            self.encoder = nn.Sequential(nn.Linear(64, hidden_width), nn.Tanh())
            self.head = nn.Linear(hidden_width, 2)
        elif kind == "linear":
            self.encoder = nn.Identity()
            self.head = nn.Linear(64, 2)
        else:
            raise ValueError(f"Unknown model kind {kind!r}")
        self.double()

    def forward(self, x: Float[Tensor, "batch 64"]) -> Float[Tensor, "batch 2"]:
        return self.head(self.encoder(x))

# %%

def train_model(
    x: Float[Tensor, "n 64"],
    y: Int[Tensor, " n"],
    kind: str = "mlp",
    hidden_width: int = 32,
    steps: int = 160,
    lr: float = 0.025,
    seed: int = 0,
) -> tuple[DigitClassifier, list[float]]:
    """Trains a DigitClassifier with full-batch Adam, and returns (model in eval mode, losses)."""
    t.manual_seed(seed)
    model = DigitClassifier(kind, hidden_width)
    optimizer = t.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for _ in range(steps):
        loss = F.cross_entropy(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return model.eval(), losses


if MAIN:
    tests.test_train_model(train_model)

# %%

if MAIN:
    model, losses = train_model(x[splits["train"]], y[splits["train"]], kind="mlp", hidden_width=32, steps=160, lr=0.025, seed=23)
    
    with t.no_grad():
        for name in ["train", "dev"]:
            accuracy = (model(x[splits[name]]).argmax(-1) == y[splits[name]]).double().mean().item()
            print(f"{name:>5} accuracy: {accuracy:.1%}")
    print(f"loss: {losses[0]:.4f} at step 1, {losses[-1]:.6f} at step {len(losses)}")
    
    utils.plot_training_curve(losses, f"Training the {DIGITS[0]}-vs-{DIGITS[1]} MLP (178 images, 160 full-batch steps)")

# %%

def make_pairs(labels: np.ndarray | Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Pairs the i-th class-0 example with the i-th class-1 example. Returns (idx0, idx1)."""
    labels = np.asarray(labels)
    idx0, idx1 = np.flatnonzero(labels == 0), np.flatnonzero(labels == 1)
    n = min(len(idx0), len(idx1))
    return idx0[:n], idx1[:n]


if MAIN:
    tests.test_make_pairs(make_pairs)

# %%

def patch_coordinates(
    recipient: Float[Tensor, "batch d"], donor: Float[Tensor, "batch d"], coords: Int[Tensor, " k"] | list[int]
) -> Float[Tensor, "batch d"]:
    """Returns a copy of `recipient` in which the coordinates `coords` are replaced by the donor's."""
    coords = t.as_tensor(coords, dtype=t.long)
    patched = recipient.clone()
    patched[:, coords] = donor[:, coords]
    return patched


def patch_direction(
    recipient: Float[Tensor, "batch d"], donor: Float[Tensor, "batch d"], direction: Float[Tensor, " d"]
) -> Float[Tensor, "batch d"]:
    """Returns `recipient` with its component along `direction` replaced by the donor's."""
    unit = direction / direction.norm()
    return recipient + ((donor - recipient) @ unit)[:, None] * unit


if MAIN:
    tests.test_patch_coordinates(patch_coordinates)
    tests.test_patch_direction(patch_direction)

# %%

def evaluate_patch(
    head: nn.Module,
    h0: Float[Tensor, "n_pairs d"],
    h1: Float[Tensor, "n_pairs d"],
    patch_fn: Callable[[Tensor, Tensor], Tensor],
) -> dict[str, np.ndarray]:
    """
    Runs an interchange intervention in both directions on each pair, and returns per-pair "swap" (both directions
    now predict the donor's label) and "recovered" (fraction of logit difference recovered, averaged over directions).
    """
    with t.no_grad():

        def logit_diff(h: Tensor) -> Tensor:
            logits = head(h)
            return logits[:, 1] - logits[:, 0]

        patched0, patched1 = patch_fn(h0, h1), patch_fn(h1, h0)
        swap = (head(patched0).argmax(-1) == 1) & (head(patched1).argmax(-1) == 0)
        ld0, ld1 = logit_diff(h0), logit_diff(h1)
        recovered0 = (logit_diff(patched0) - ld0) / (ld1 - ld0)
        recovered1 = (logit_diff(patched1) - ld1) / (ld0 - ld1)
    return {"swap": swap.double().numpy(), "recovered": (0.5 * (recovered0 + recovered1)).numpy()}


if MAIN:
    tests.test_evaluate_patch(evaluate_patch)

# %%

if MAIN:
    with t.no_grad():
        h_train = model.encoder(x[splits["train"]])
    y_train = y[splits["train"]]
    train0, train1 = make_pairs(y_train)
    print(f"{len(train0)} training pairs\n")
    
    checks = {
        "no patch": lambda r, d: r,
        "full patch": lambda r, d: d,
        "all 32 coordinates": functools.partial(patch_coordinates, coords=list(range(32))),
    }
    for name, fn in checks.items():
        out = evaluate_patch(model.head, h_train[train0], h_train[train1], fn)
        print(f"{name:>20}: swap rate {out['swap'].mean():.3f}, mean recovered {out['recovered'].mean():.3f}")

# %%

if MAIN:
    PLAY_UNITS = [0, 1, 2, 3]  # which hidden units to copy from the donor
    PLAY_PAIRS = [0, 1, 2]  # which training pairs to look at
    
    with t.no_grad():
        for i in PLAY_PAIRS:
            r, d = h_train[train0[i : i + 1]], h_train[train1[i : i + 1]]  # recipient: a 3, donor: an 8
            p_before = model.head(r).softmax(-1)[0, 1].item()
            p_after = model.head(patch_coordinates(r, d, PLAY_UNITS)).softmax(-1)[0, 1].item()
            print(f"training pair {i}: P(8) for the 3 is {p_before:.4f} before and {p_after:.4f} after patching units {PLAY_UNITS}")

# %%

def mean_difference_direction(h: Float[Tensor, "n d"], y: Int[Tensor, " n"]) -> Float[Tensor, " d"]:
    """Unit vector pointing from the mean of class 0 to the mean of class 1."""
    diff = h[y == 1].mean(0) - h[y == 0].mean(0)
    return diff / diff.norm()


def rank_coordinates(h: Float[Tensor, "n d"], y: Int[Tensor, " n"], readout: Float[Tensor, " d"]) -> Int[Tensor, " d"]:
    """Unit indices sorted by |readout_j * (class-1 mean - class-0 mean)_j|, largest first (stable for ties)."""
    diff = h[y == 1].mean(0) - h[y == 0].mean(0)
    return t.argsort((readout * diff).abs(), descending=True, stable=True)


if MAIN:
    tests.test_mean_difference_direction(mean_difference_direction)
    tests.test_rank_coordinates(rank_coordinates)

# %%

def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact (Clopper-Pearson) 1 - alpha confidence interval for a binomial proportion k / n."""
    low = 0.0 if k == 0 else stats.beta.ppf(alpha / 2, k, n - k + 1)
    high = 1.0 if k == n else stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(low), float(high)


def claim_verdict(low: float, high: float, threshold: float) -> str:
    """Verdict on the claim 'the true value is greater than threshold', given a confidence interval."""
    if low > threshold:
        return "supported"
    if high <= threshold:
        return "ruled out"
    return "inconclusive"


if MAIN:
    tests.test_clopper_pearson(clopper_pearson)
    tests.test_claim_verdict(claim_verdict)

# %%

def paired_bootstrap(
    method_scores: np.ndarray, control_scores: np.ndarray, n_resamples: int = 2000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float, float]:
    """Mean of (method - control) over pairs, with a percentile bootstrap interval that resamples pairs."""
    diffs = np.asarray(method_scores, dtype=float) - np.asarray(control_scores, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(n_resamples, len(diffs)))
    boot_means = diffs[idx].mean(axis=1)
    low, high = np.quantile(boot_means, [alpha / 2, 1 - alpha / 2])
    return float(diffs.mean()), float(low), float(high)


if MAIN:
    tests.test_paired_bootstrap(paired_bootstrap)

# %%

if MAIN:
    def simulate_coverage(true_rates: list[float], n_pairs: int, n_studies: int, seed: int = 0) -> dict[str, list[float]]:
        """For each true swap rate, the fraction of simulated studies whose 95% interval contains it."""
        rng = np.random.default_rng(seed)
        coverage = {"Clopper-Pearson": [], "percentile bootstrap": []}
        for p in true_rates:
            hits_cp = hits_boot = 0
            for study in range(n_studies):
                swaps = (rng.random(n_pairs) < p).astype(float)
                low, high = clopper_pearson(int(swaps.sum()), n_pairs)
                hits_cp += low <= p <= high
                _, low, high = paired_bootstrap(swaps, np.zeros(n_pairs), n_resamples=1000, seed=study)
                hits_boot += low <= p <= high
            coverage["Clopper-Pearson"].append(hits_cp / n_studies)
            coverage["percentile bootstrap"].append(hits_boot / n_studies)
        return coverage
    
    
    TRUE_RATES = [0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]
    coverage = simulate_coverage(TRUE_RATES, n_pairs=44, n_studies=400)
    print(pd.DataFrame(coverage, index=pd.Index(TRUE_RATES, name="true rate")).round(3).to_string())
    utils.plot_coverage(TRUE_RATES, coverage, n_pairs=44)

# %%

if MAIN:
    N_CONFIRM_PAIRS = len(make_pairs(y[splits["confirm"]])[0])  # uses labels only: no model, no outcomes
    THRESHOLD = 0.9
    
    decision_map = {
        n: [claim_verdict(*clopper_pearson(k, n), THRESHOLD) for k in range(n + 1)] for n in [20, N_CONFIRM_PAIRS, 100]
    }
    for n, verdicts in decision_map.items():
        supported = [k for k, v in enumerate(verdicts) if v == "supported"]
        ruled_out = [k for k, v in enumerate(verdicts) if v == "ruled out"]
        can_support = f"k >= {min(supported)}" if supported else "is impossible"
        print(f"n = {n:3d}: 'ruled out' if k <= {max(ruled_out)}, 'supported' {can_support}, inconclusive in between")
    utils.plot_decision_map(decision_map, THRESHOLD)

# %%

if MAIN:
    PLAY_THRESHOLD = 0.9
    PLAY_ALPHA = 0.05
    
    for PLAY_FAILURES in [0, 1, 2, 5]:
        n_needed = next(
            (n for n in range(PLAY_FAILURES + 1, 10_000)
             if claim_verdict(*clopper_pearson(n - PLAY_FAILURES, n, PLAY_ALPHA), PLAY_THRESHOLD) == "supported"),
            None,
        )
        print(f"To support 'rate > {PLAY_THRESHOLD:.0%}' with {PLAY_FAILURES} failure(s) you need at least {n_needed} pairs")

# %%

PLAN = {
    "question": "Is the 3-vs-8 decision of a one-hidden-layer MLP carried by a single hidden unit?",
    "digits": [3, 8],
    "split": {"train_fraction": 0.5, "dev_fraction": 0.25, "seed": 17},
    "model": {"kind": "mlp", "hidden_width": 32, "steps": 160, "lr": 0.025, "seed": 23},
    "candidates": ["units_1", "units_2", "units_4", "units_8", "direction"],
    "coordinate_counts": [1, 2, 4, 8],
    "selection_rule": "highest development swap rate; ties go to the earlier candidate",
    "controls": {"n_random": 16, "seed": 31, "kinds": ["no_patch", "full_patch", "random_units", "random_dirs", "shuffled"]},
    "H1": {"intervention": "units_1", "metric": "swap", "test": "clopper_pearson", "threshold": 0.9, "alpha": 0.05},
    "H2": {"method": "selected", "controls": ["random_dirs", "shuffled"], "metric": "swap", "test": "paired_bootstrap"},
    "H3": {"method": "units_1", "control": "random_units_1", "metric": "recovered", "test": "paired_bootstrap"},
    "bootstrap": {"n_resamples": 2000, "seed": 37, "alpha": 0.05},
    "instrument_checks": {"full_patch_min_swap": 0.95, "no_patch_max_swaps": 0, "min_confirm_pairs": 40},
}


def plan_fingerprint(plan: dict) -> str:
    """A short hash of the plan. If any setting changes, the fingerprint changes."""
    return hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()[:12]

# %%

class ResearchLog:
    """A minimal research log: an ordered list of (stage, note) entries."""

    STAGES = ["question", "setup", "registered", "exploration", "anomaly", "selection", "prediction", "confirmation", "post-hoc", "write-up"]

    def __init__(self):
        self.entries: list[tuple[str, str]] = []

    def add(self, stage: str, note: str):
        assert stage in self.STAGES, f"Unknown stage {stage!r}: use one of {self.STAGES}"
        self.entries.append((stage, note))

    def stages(self) -> list[str]:
        return [stage for stage, _ in self.entries]

    def show(self, width: int = 100):
        for i, (stage, note) in enumerate(self.entries, 1):
            lines = textwrap.wrap(note, width - 18)
            print(f"{i:02d} {stage:>12} | {lines[0]}")
            for line in lines[1:]:
                print(" " * 16 + "| " + line)

# %%

if MAIN:
    log = ResearchLog()
    log.add("question", PLAN["question"])
    log.add(
        "setup",
        f"Split {len(y)} images into {len(splits['train'])}/{len(splits['dev'])}/{len(splits['confirm'])} (seed 17). "
        f"Trained the MLP once: train and dev accuracy 100%. Instrument checks on {len(train0)} training pairs pass "
        "(no patch swaps 0%, full patch 100%). Confirmation set untouched.",
    )
    REGISTERED_FINGERPRINT = plan_fingerprint(PLAN)
    log.add("registered", f"Plan registered with fingerprint {REGISTERED_FINGERPRINT}.")
    log.show()

# %%

def build_controls(
    h_train: Float[Tensor, "n d"], y_train: Int[Tensor, " n"], coordinate_counts: list[int], n_random: int, seed: int
) -> dict:
    """Builds the shuffled-label direction, random directions and random coordinate sets from one seeded generator."""
    n, d = h_train.shape
    generator = t.Generator().manual_seed(seed)
    permutation = t.randperm(n, generator=generator)
    shuffled = mean_difference_direction(h_train, y_train[permutation])
    random_directions = t.randn(n_random, d, generator=generator, dtype=h_train.dtype)
    random_directions = random_directions / random_directions.norm(dim=-1, keepdim=True)
    random_coordinates = {
        k: [t.randperm(d, generator=generator)[:k] for _ in range(n_random)] for k in coordinate_counts
    }
    return {"shuffled_direction": shuffled, "random_directions": random_directions, "random_coordinates": random_coordinates}


if MAIN:
    tests.test_build_controls(build_controls, mean_difference_direction)

# %%

def make_interventions(
    order: Int[Tensor, " d"], direction: Float[Tensor, " d"], controls: dict, coordinate_counts: list[int]
) -> dict[str, list[Callable]]:
    """Maps each intervention name to a list of patch functions (one per random draw for the random controls)."""
    interventions = {"no_patch": [lambda r, d: r]}
    for k in coordinate_counts:
        interventions[f"units_{k}"] = [functools.partial(patch_coordinates, coords=order[:k])]
        interventions[f"random_units_{k}"] = [
            functools.partial(patch_coordinates, coords=coords) for coords in controls["random_coordinates"][k]
        ]
    interventions["direction"] = [functools.partial(patch_direction, direction=direction)]
    interventions["shuffled"] = [functools.partial(patch_direction, direction=controls["shuffled_direction"])]
    interventions["random_dirs"] = [functools.partial(patch_direction, direction=v) for v in controls["random_directions"]]
    interventions["full_patch"] = [lambda r, d: d]
    return interventions

# %%

def evaluate_interventions(
    head: nn.Module, h: Float[Tensor, "n d"], y: Int[Tensor, " n"], interventions: dict[str, list[Callable]]
) -> dict[str, dict[str, np.ndarray]]:
    """Evaluates every intervention on the pairs of one split. Arrays have shape (n_draws, n_pairs)."""
    idx0, idx1 = make_pairs(y)
    h0, h1 = h[idx0], h[idx1]
    results = {}
    for name, patch_fns in interventions.items():
        outcomes = [evaluate_patch(head, h0, h1, fn) for fn in patch_fns]
        results[name] = {key: np.stack([out[key] for out in outcomes]) for key in ["swap", "recovered"]}
    return results


if MAIN:
    tests.test_evaluate_interventions(evaluate_interventions)

# %%

if MAIN:
    assert plan_fingerprint(PLAN) == REGISTERED_FINGERPRINT, "The plan has changed since it was registered!"
    
    readout = (model.head.weight[1] - model.head.weight[0]).detach()
    order = rank_coordinates(h_train, y_train, readout)
    direction = mean_difference_direction(h_train, y_train)
    controls = build_controls(h_train, y_train, PLAN["coordinate_counts"], PLAN["controls"]["n_random"], PLAN["controls"]["seed"])
    interventions = make_interventions(order, direction, controls, PLAN["coordinate_counts"])
    
    with t.no_grad():
        h_dev = model.encoder(x[splits["dev"]])
    y_dev = y[splits["dev"]]
    dev = evaluate_interventions(model.head, h_dev, y_dev, interventions)
    
    n_dev_pairs = dev["no_patch"]["swap"].shape[1]
    print(f"Development set: {n_dev_pairs} pairs\n")
    print(f"{'intervention':>16} | {'swap rate':>9} | {'recovered':>9}")
    for name, res in dev.items():
        print(f"{name:>16} | {res['swap'].mean():9.3f} | {res['recovered'].mean():9.3f}")

# %%

def select_on_development(dev_scores: dict[str, float], candidates: list[str]) -> str:
    """Returns the registered candidate with the highest development score (ties: earliest in `candidates`)."""
    if set(dev_scores) != set(candidates):
        raise ValueError(f"Scores {sorted(dev_scores)} don't match the registered candidates {sorted(candidates)}")
    scores = [dev_scores[name] for name in candidates]
    return candidates[int(np.argmax(scores))]  # np.argmax returns the first maximum


if MAIN:
    tests.test_select_on_development(select_on_development)

# %%

if MAIN:
    dev_scores = {name: dev[name]["swap"].mean() for name in PLAN["candidates"]}
    selected = select_on_development(dev_scores, PLAN["candidates"])
    print("Development swap rates of the candidates:", {name: round(float(s), 3) for name, s in dev_scores.items()})
    print(f"Selected: {selected!r}")
    
    ORDER = [
        "no_patch", "units_1", "random_units_1", "units_2", "random_units_2", "units_4", "random_units_4",
        "units_8", "random_units_8", "direction", "shuffled", "random_dirs", "full_patch",
    ]
    utils.plot_dev_exploration(dev, PLAN["coordinate_counts"], ORDER, selected=selected)

# %%

if MAIN:
    topk_rates = ", ".join(f"{dev['units_' + str(k)]['swap'].mean():.0%}" for k in PLAN["coordinate_counts"])
    log.add(
        "exploration",
        f"Development ({n_dev_pairs} pairs): top-k units swap {topk_rates} for k = {PLAN['coordinate_counts']}; "
        f"mean-diff direction {dev['direction']['swap'].mean():.0%}; random directions {dev['random_dirs']['swap'].mean():.0%}; "
        f"full patch {dev['full_patch']['swap'].mean():.0%}.",
    )
    log.add(
        "anomaly",
        f"Shuffled-label direction swaps {dev['shuffled']['swap'].mean():.0%} of development pairs (expected ~0%). "
        "Not re-drawing it and not changing the plan. Cause to be investigated after confirmation, as post-hoc analysis.",
    )
    log.add("selection", f"Selected {selected!r} by the registered rule (development swap rate {dev_scores[selected]:.0%}).")

# %%

if MAIN:
    PREDICTIONS = {
        "H1": "ruled out: no top-k set swapped a single development pair, and k = 1 recovers only ~5% of the logit difference",
        "H2": "not supported: the selected direction will beat random directions, but not the shuffled-label direction",
        "H3": "supported: the top unit recovered more than random units on development pairs, by a small margin",
    }
    for hypothesis, prediction in PREDICTIONS.items():
        log.add("prediction", f"{hypothesis}: {prediction}")
    log.show()

# %%

if MAIN:
    assert plan_fingerprint(PLAN) == REGISTERED_FINGERPRINT, "The plan has changed since it was registered!"
    assert "selection" in log.stages() and "confirmation" not in log.stages(), "Select a method before opening confirmation"
    
    with t.no_grad():
        h_confirm = model.encoder(x[splits["confirm"]])
        confirm_accuracy = (model.head(h_confirm).argmax(-1) == y[splits["confirm"]]).double().mean().item()
    y_confirm = y[splits["confirm"]]
    confirm = evaluate_interventions(model.head, h_confirm, y_confirm, interventions)
    n_pairs = confirm["no_patch"]["swap"].shape[1]
    
    log.add("confirmation", f"Opened the confirmation set once ({len(y_confirm)} images, {n_pairs} pairs; model accuracy {confirm_accuracy:.1%}).")
    print(f"Confirmation set: {len(y_confirm)} images, model accuracy {confirm_accuracy:.1%}, {n_pairs} pairs\n")
    print(f"{'intervention':>16} | {'swaps':>7} | {'recovered':>9}")
    for name, res in confirm.items():
        swaps = f"{res['swap'].sum(1).mean():.0f}/{n_pairs}" if res["swap"].shape[0] == 1 else f"{res['swap'].mean():.3f}"
        print(f"{name:>16} | {swaps:>7} | {res['recovered'].mean():9.3f}")
    
    confirm0, confirm1 = make_pairs(y_confirm)
    with t.no_grad():
        wrong = np.flatnonzero((model.head(h_confirm).argmax(-1) != y_confirm).numpy())
    for i in wrong:
        in_pair = i in confirm0 or i in confirm1
        print(f"\nMisclassified: a {DIGITS[y_confirm[i]]} (dataset row {rows[splits['confirm'][i]]}), {'in' if in_pair else 'not in'} a pair")

# %%

if MAIN:
    def per_pair(results: dict, name: str, metric: str) -> np.ndarray:
        """Per-pair score for one intervention, averaged over random draws."""
        return results[name][metric].mean(0)
    
    
    boot = PLAN["bootstrap"]
    h1_swaps = int(per_pair(confirm, "units_1", "swap").sum())
    h1_interval = clopper_pearson(h1_swaps, n_pairs, PLAN["H1"]["alpha"])
    h2a = paired_bootstrap(per_pair(confirm, selected, "swap"), per_pair(confirm, "random_dirs", "swap"), boot["n_resamples"], boot["seed"], boot["alpha"])
    h2b = paired_bootstrap(per_pair(confirm, selected, "swap"), per_pair(confirm, "shuffled", "swap"), boot["n_resamples"], boot["seed"], boot["alpha"])
    h3 = paired_bootstrap(per_pair(confirm, "units_1", "recovered"), per_pair(confirm, "random_units_1", "recovered"), boot["n_resamples"], boot["seed"], boot["alpha"])
    
    verdicts = {
        "H1": claim_verdict(*h1_interval, PLAN["H1"]["threshold"]),
        "H2a": claim_verdict(h2a[1], h2a[2], 0.0),
        "H2b": claim_verdict(h2b[1], h2b[2], 0.0),
        "H3": claim_verdict(h3[1], h3[2], 0.0),
    }
    verdicts["H2"] = "supported" if verdicts["H2a"] == verdicts["H2b"] == "supported" else "not supported"
    
    print(f"H1  top unit swap rate        {h1_swaps}/{n_pairs} = {h1_swaps / n_pairs:.3f}, 95% CI [{h1_interval[0]:.3f}, {h1_interval[1]:.3f}] vs 0.9 -> {verdicts['H1']}")
    print(f"H2a {selected} - random dirs    {h2a[0]:+.3f}, 95% CI [{h2a[1]:+.3f}, {h2a[2]:+.3f}] -> {verdicts['H2a']}")
    print(f"H2b {selected} - shuffled       {h2b[0]:+.3f}, 95% CI [{h2b[1]:+.3f}, {h2b[2]:+.3f}] -> {verdicts['H2b']}")
    print(f"H2  (needs both)                          -> {verdicts['H2']}")
    print(f"H3  top unit - random unit    {h3[0]:+.4f}, 95% CI [{h3[1]:+.4f}, {h3[2]:+.4f}] (recovered) -> {verdicts['H3']}")

# %%

if MAIN:
    intervals = {}
    for name in ["units_1", "units_2", "units_4", "units_8", "direction", "shuffled"]:
        k = int(per_pair(confirm, name, "swap").sum())
        intervals[name] = (k / n_pairs, *clopper_pearson(k, n_pairs))
    comparisons = {
        f"H2a: {selected} − random dirs\n(swap)": h2a,
        f"H2b: {selected} − shuffled\n(swap)": h2b,
        "H3: top unit − random unit\n(recovered)": h3,
    }
    utils.plot_signature(
        dev, confirm, PLAN["coordinate_counts"], ORDER, selected, intervals, comparisons, PLAN["H1"]["threshold"],
        suptitle=f"Confirmation: {DIGITS[0]} vs {DIGITS[1]}, 32-unit MLP, {n_pairs} held-out pairs",
    )

# %%

if MAIN:
    SHOWN_PAIRS = [0, 1, 2]
    SHOWN_INTERVENTIONS = ["no_patch", "units_1", "units_8", "direction", "shuffled", "random_dirs", "full_patch"]
    
    images, row_labels, probs = [], [], []
    with t.no_grad():
        for i in SHOWN_PAIRS:
            for recipient, donor in [(confirm0[i], confirm1[i]), (confirm1[i], confirm0[i])]:
                r, d = h_confirm[[recipient]], h_confirm[[donor]]
                donor_label = y_confirm[donor].item()
                row = [model.head(interventions[name][0](r, d)).softmax(-1)[0, donor_label].item() for name in SHOWN_INTERVENTIONS]
                probs.append(row)
                images.append((x[splits["confirm"]][recipient], x[splits["confirm"]][donor]))
                row_labels.append(f"pair {i}: {DIGITS[1 - donor_label]} ← {DIGITS[donor_label]}")
    column_labels = ["no patch", "top-1 unit", "top-8 units", "mean-diff direction", "shuffled-label direction", "random direction #0", "full patch"]
    utils.plot_pairs(images, row_labels, np.array(probs), column_labels)

# %%

def assess_methodology(checks: dict[str, bool]) -> list[str]:
    """Returns the names of the failed checks (an empty list means the methodology is sound)."""
    for name, value in checks.items():
        if not isinstance(value, (bool, np.bool_)):
            raise TypeError(f"Check {name!r} should be a boolean, got {value!r}")
    return [name for name, value in checks.items() if not value]


if MAIN:
    tests.test_assess_methodology(assess_methodology)

# %%

if MAIN:
    instrument = PLAN["instrument_checks"]
    all_ids = np.concatenate(list(splits.values()))
    methodology_checks = {
        "plan unchanged since registration": plan_fingerprint(PLAN) == REGISTERED_FINGERPRINT,
        "splits disjoint and exhaustive": bool(np.array_equal(np.sort(all_ids), np.arange(len(y)))),
        "method selected before confirmation opened": log.stages().index("selection") < log.stages().index("confirmation"),
        "every registered intervention run on confirmation": set(confirm) == set(interventions),
        "positive control works (full patch)": bool(confirm["full_patch"]["swap"].mean() >= instrument["full_patch_min_swap"]),
        "negative control clean (no patch)": bool(confirm["no_patch"]["swap"].sum() <= instrument["no_patch_max_swaps"]),
        "enough confirmation pairs": n_pairs >= instrument["min_confirm_pairs"],
    }
    
    failed = assess_methodology(methodology_checks)
    for name, value in methodology_checks.items():
        print(f"  [{'x' if value else ' '}] {name}")
    print(f"\nMethodology: {'SOUND' if not failed else 'FLAWED: ' + ', '.join(failed)}")
    print(f"Findings:    H1 {verdicts['H1']}, H2 {verdicts['H2']}, H3 {verdicts['H3']}")

# %%

def logit_contributions(h: Float[Tensor, "n d"], y: Int[Tensor, " n"], readout: Float[Tensor, " d"]) -> Float[Tensor, " d"]:
    """Each unit's share of the difference in mean logit difference between class 1 and class 0."""
    contributions = readout * (h[y == 1].mean(0) - h[y == 0].mean(0))
    return contributions / contributions.sum()


if MAIN:
    tests.test_logit_contributions(logit_contributions)

# %%

if MAIN:
    shares = logit_contributions(h_train, y_train, readout)
    cumulative = shares[order].cumsum(0).numpy()
    print("Shares of the top 8 units:", [f"{s:.3f}" for s in shares[order[:8]].tolist()])
    print(f"Units with a share above 2%: {(shares > 0.02).sum().item()} of {len(shares)}")
    
    dev0, dev1 = make_pairs(y_dev)
    measured_k = [1, 2, 4, 8, 12, 16, 24, 32]
    measured = [
        evaluate_patch(model.head, h_dev[dev0], h_dev[dev1], functools.partial(patch_coordinates, coords=order[:k]))
        for k in measured_k
    ]
    for k, out in zip(measured_k, measured):
        print(f"top {k:2d} units: predicted share {cumulative[k - 1]:.3f}, measured recovered {out['recovered'].mean():.3f}, swap rate {out['swap'].mean():.3f}")
    
    class_order = np.argsort(y_train.numpy(), kind="stable")
    utils.plot_hidden_code(
        h_train[class_order][:, order].numpy(),
        int((y_train == 0).sum()),
        (str(DIGITS[0]), str(DIGITS[1])),
        order.tolist(),
        cumulative,
        measured_k,
        [out["recovered"].mean() for out in measured],
        np.arange(1, len(shares) + 1) / len(shares),
    )

# %%

def principal_directions(h: Float[Tensor, "n d"]) -> tuple[Float[Tensor, "k d"], Float[Tensor, " k"]]:
    """Principal directions (rows, largest variance first) and the fraction of variance each explains."""
    centered = h - h.mean(0)
    _, singular_values, vh = t.linalg.svd(centered, full_matrices=False)
    variance = singular_values**2
    return vh, variance / variance.sum()


if MAIN:
    tests.test_principal_directions(principal_directions)

# %%

if MAIN:
    components, variance = principal_directions(h_train)
    within = t.cat([h_train[y_train == c] - h_train[y_train == c].mean(0) for c in (0, 1)])
    components_within, variance_within = principal_directions(within)
    pc1 = components[0]
    print(f"Top principal component (PC1): {variance[0]:.1%} of the variance")
    print(
        f"Within-class (class means removed): top component explains {variance_within[0]:.1%}, "
        f"|cos| with PC1 = {(components_within[0] @ pc1).abs():.3f}"
    )
    
    
    def alignment(v: Tensor) -> float:
        return (v / v.norm() @ pc1).abs().item()
    
    
    def dev_swap_rate(v: Tensor) -> float:
        return evaluate_patch(model.head, h_dev[dev0], h_dev[dev1], functools.partial(patch_direction, direction=v))["swap"].mean()
    
    
    generator = t.Generator().manual_seed(2024)
    redrawn_shuffled = [mean_difference_direction(h_train, y_train[t.randperm(len(y_train), generator=generator)]) for _ in range(200)]
    redrawn_random = list(t.randn(200, h_train.shape[1], generator=generator, dtype=t.float64))
    cos_shuffled = np.array([alignment(v) for v in redrawn_shuffled])
    cos_random = np.array([alignment(v) for v in redrawn_random])
    swap_shuffled = np.array([dev_swap_rate(v) for v in redrawn_shuffled])
    swap_random = np.array([dev_swap_rate(v) for v in redrawn_random])
    
    print(f"|cos| with PC1: mean-diff direction {alignment(direction):.4f}, registered shuffled {alignment(controls['shuffled_direction']):.3f}")
    print(f"Re-drawn shuffled directions: median |cos| {np.median(cos_shuffled):.3f}, {np.mean(swap_shuffled >= 0.9):.0%} swap at least 90% of dev pairs")
    print(f"Re-drawn random directions:   median |cos| {np.median(cos_random):.3f}, {np.mean(swap_random >= 0.9):.0%} swap at least 90% of dev pairs")
    
    utils.plot_one_dimensional(
        variance.numpy(), variance_within.numpy(), cos_shuffled, cos_random, swap_shuffled, swap_random,
        {
            "mean-diff direction": (alignment(direction), dev["direction"]["swap"].mean()),
            "registered shuffled": (alignment(controls["shuffled_direction"]), dev["shuffled"]["swap"].mean()),
            "top principal component": (1.0, dev_swap_rate(pc1)),
        },
    )

# %%

def run_pilot(digits: tuple[int, int] = (3, 8), kind: str = "mlp", hidden_width: int = 32, steps: int = 160) -> dict[str, float]:
    """
    Runs the exploration stage of the sprint (train the model, build candidates and controls, evaluate on the
    development pairs) for another task or model. Uses training and development data only.
    """
    x_, y_, _ = utils.load_digit_pair(digits)
    splits_ = split_examples(y_, train_fraction=0.5, dev_fraction=0.25, seed=17)
    model_, _ = train_model(x_[splits_["train"]], y_[splits_["train"]], kind, hidden_width, steps, lr=0.025, seed=23)
    with t.no_grad():
        h_tr, h_dv = model_.encoder(x_[splits_["train"]]), model_.encoder(x_[splits_["dev"]])
        dev_accuracy = (model_.head(h_dv).argmax(-1) == y_[splits_["dev"]]).double().mean().item()
    y_tr, y_dv = y_[splits_["train"]], y_[splits_["dev"]]
    readout_ = (model_.head.weight[1] - model_.head.weight[0]).detach()
    order_ = rank_coordinates(h_tr, y_tr, readout_)
    direction_ = mean_difference_direction(h_tr, y_tr)
    controls_ = build_controls(h_tr, y_tr, [1, 2, 4, 8], n_random=16, seed=31)
    results = evaluate_interventions(model_.head, h_dv, y_dv, make_interventions(order_, direction_, controls_, [1, 2, 4, 8]))
    _, variance_ = principal_directions(h_tr)
    summary = {"dev pairs": float(results["no_patch"]["swap"].shape[1]), "dev accuracy": dev_accuracy, "top PC variance": variance_[0].item()}
    for name in ["units_1", "units_8", "direction", "shuffled", "random_dirs", "full_patch"]:
        summary[name] = results[name]["swap"].mean()
    return summary

# %%

if MAIN:
    pilots = {
        "3 vs 8, MLP (our model)": run_pilot((3, 8), "mlp"),
        "3 vs 8, linear (pixels)": run_pilot((3, 8), "linear"),
    }
    print("Development swap rates (and other summaries), per model:\n")
    print(pd.DataFrame(pilots).T.round(3).to_string())
    log.add(
        "post-hoc",
        f"Hidden layer is effectively 1-D: PC1 explains {variance[0]:.1%} of training variance; every unit carries a "
        f"small share (top unit {shares[order[0]]:.1%}). Re-drawn shuffled directions align with PC1 (median |cos| "
        f"{np.median(cos_shuffled):.2f}). In the pixel-space linear model (PC1 {pilots['3 vs 8, linear (pixels)']['top PC variance']:.0%}), "
        f"the shuffled direction swaps {pilots['3 vs 8, linear (pixels)']['shuffled']:.0%} of dev pairs vs "
        f"{pilots['3 vs 8, linear (pixels)']['direction']:.0%} for the mean-diff direction. Exploratory only.",
    )

# %%

if MAIN:
    PLAY_DIRECTIONS = {
        "mean-diff direction": direction,
        "registered shuffled direction": controls["shuffled_direction"],
        "random direction #0": controls["random_directions"][0],
    }
    PLAY_SCALE = 1.0
    
    scales = np.linspace(0, 2, 21)
    curves = {}
    with t.no_grad():
        for name, v in PLAY_DIRECTIONS.items():
            curve = []
            for s in scales:
                p = []
                for recipient, donor, label in [(h_dev[dev0], h_dev[dev1], 1), (h_dev[dev1], h_dev[dev0], 0)]:
                    patched = recipient + s * (patch_direction(recipient, donor, v) - recipient)
                    p.append(model.head(patched).softmax(-1)[:, label])
                curve.append(t.cat(p).mean().item())
            curves[name] = np.array(curve)
            at_scale = np.interp(PLAY_SCALE, scales, curves[name])
            print(f"{name:>30}: mean P(donor's label) = {at_scale:.3f} at scale {PLAY_SCALE}")
    utils.plot_dose_response(scales, curves, PLAY_SCALE, "Dose-response on development pairs (exploratory)")

# %%

if MAIN:
    log.add("write-up", "Wrote up all three registered results (H1 ruled out, H2 not supported, H3 supported), both controls that behaved and the one that didn't, the post-hoc explanation (labelled exploratory), and limitations.")
    log.show()

# %%

if MAIN:
    PILOTS = [
        {"digits": (1, 7), "kind": "mlp"},
        {"digits": (1, 7), "kind": "linear"},
        {"digits": (3, 8), "kind": "mlp", "hidden_width": 8},
        {"digits": (3, 8), "kind": "mlp", "steps": 20},
    ]
    table = {f"{p['digits'][0]} vs {p['digits'][1]}, " + ", ".join(f"{k}={v}" for k, v in p.items() if k != "digits"): run_pilot(**p) for p in PILOTS}
    print(pd.DataFrame(table).T.round(3).to_string())

# %%

if MAIN:
    my_log = ResearchLog()
    my_log.add("question", "(write your question here)")
    # my_log.add("setup", "...")
    # my_log.add("registered", f"Plan registered with fingerprint {plan_fingerprint(MY_PLAN)}")
    my_log.show()

# %%
