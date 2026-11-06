# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
```python
[{"title": "Exact game", "icon": "1-circle-fill", "subtitle": "15%"}, {"title": "Kernel regression and sampling", "icon": "2-circle-fill", "subtitle": "25%"}, {"title": "Owen values and groups", "icon": "3-circle-fill", "subtitle": "30%"}, {"title": "Controlled signature", "icon": "4-circle-fill", "subtitle": "20%"}, {"title": "Play", "icon": "5-circle-fill", "subtitle": "5%"}, {"title": "Bonus", "icon": "star", "subtitle": "5%"}]
```
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
# [16.2] KernelSHAP and PartitionSHAP
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<img src="https://raw.githubusercontent.com/info-arena/ARENA_img/main/misc/headers/header-13-2.png" width="350">
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
# Introduction
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Historical reference and prediction practice

The opening preview and original publication remain **retained historical analytic references**. The current solutions outputs and signature plots below are a **fresh complete CPU
execution** after a declared numerical repair. The archived original figure is also retained as `media/kernelshap_partition_controls_signature_historical.png`. The historical
verification report concerns a separate four-player CUDA-trained neural game; it was not rerun or rewritten. The main lesson instead uses an exact six-player dividend game. No
pretrained model, training, GPU work or download was used.

A rank-deficient regression exposed an intermittent default-driver failure: the intended minimum-norm KKT solution could lose rank and violate the constraints. The partial-design
solver now explicitly uses CPU SVD (`gelsd`) and returns coefficients to the original device; the full-table direct solve is unchanged. Original sources, outputs, the failed
regression and both complete successful CPU runs remain recorded in `docs/mastery/evidence/kernelshap_master_migration/`. See `docs/mastery/migrations/kernelshap.md` for exact
scope and runtime.

This reference-rich lesson is guided practice. Expected outputs, the introductory historical figure and interpretations expose answers. Forecast before opening them, recording
confidence, causal reasoning and exposure status; afterward record controls, discrepancies and a changed scientific condition. No learner answer is prefilled. Current corrected CPU
evidence is separate from the archived publication and legacy CUDA preflight. The entire legacy source directory and its original page remain unchanged.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Content & Learning Objectives

### 1️⃣ Fix the attribution target

> ##### Learning Objectives
>
> - Implement dividend and subset oracles independently of regression.
> - Separate baseline, efficiency and player-wise correctness.

### 2️⃣ Control sampling and grouping

> ##### Learning Objectives
>
> - Implement finite kernel weights, constrained regression and paired sampling.
> - Implement exact Owen values for a declared partition.
> - Compare equal budgets and retain repeated random partitions.

### 3️⃣ Predict and inspect errors

> ##### Learning Objectives
>
> - Read sampling dispersion and partition-induced credit changes separately.
> - Change one scientific condition and retain oracle-based controls.
'''

# ! CELL TYPE: code
# ! FILTERS: [python]
# ! TAGS: []

MAIN = __name__ == "__main__"

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Core Question

When exact Shapley is known, which sampling and partition choices preserve the attribution, and which produce a confident but wrong answer?

**Claim:** on the six-player ground-truth game below, full-table KernelSHAP recovers exact Shapley; paired-kernel sampling has lower mean error than uniform sampling at the same
partial budgets; and exact Owen values match Shapley for the declared interaction-aligned partition but shift by up to `0.300` for a fixed misaligned partition.

<img src="../../instructions/assets/kernelshap_partition_controls_signature.png" width="980">

```python
EXERCISE_ID = "16_2_kernelshap_and_partitionshap_controls"
GT_TIER = "GT-0"
DIFFICULTY = 4
IMPORTANCE = 4
EXPECTED_RUNTIME = "seconds on toy contract; minutes on local real-model path"
REQUIRES_GPU = True
```

## Learning goals

You will:

- build a finite interaction game with an independent exact oracle;
- implement the KernelSHAP kernel and constrained weighted regression;
- compare full-table and sampled KernelSHAP;
- implement uniform and complement-paired kernel sampling;
- validate player partitions and collapse group coalitions correctly;
- implement exact Owen values for a declared partition;
- test aligned, singleton, misaligned, and random partition controls.

The visible results are generated in this notebook from the implemented methods. No verification report is used as teaching evidence.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

from collections.abc import Callable, Mapping, Sequence

from dataclasses import dataclass

import itertools

import math

import sys

from pathlib import Path

from typing import Literal

import matplotlib.pyplot as plt

import numpy as np

import torch as t

chapter = "chapter16_shapley_attribution_baselines"

section = "part2_kernelshap_partitionshap"

# FILTERS: ~python
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
# END FILTERS
# FILTERS: python
root_dir = Path(__file__).resolve().parents[3]
# END FILTERS

exercises_dir = root_dir / chapter / "exercises"

assets_dir = root_dir / chapter / "exercises" / section / "media"

if MAIN:
    for path in (root_dir, exercises_dir):
        if str(path) not in sys.path:
            sys.path.append(str(path))

if MAIN:
    from chapter16_shapley_attribution_baselines.exercises.part2_kernelshap_partitionshap import tests

Coalition = frozenset[int]

if MAIN:
    t.set_grad_enabled(False)

if MAIN:
    plt.rcParams.update({
        "figure.dpi": 120,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "font.size": 10,
    })

EXACT = "#B43A32"

PAIRED = "#167D8D"

UNIFORM = "#D39A2C"

CONTROL = "#5F6875"

@dataclass(frozen=True)
class KernelSHAPApproximationReport:
    shapley_values: t.Tensor
    exact_values: t.Tensor
    max_abs_error: float
    approximates_exact: bool

@dataclass(frozen=True)
class PartitionSHAPReport:
    group_values: t.Tensor
    player_values: t.Tensor
    exact_values: t.Tensor
    max_abs_error: float
    recovers_exact: bool

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
# 1️⃣ An exact target
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Cold open: two approximations, two hidden choices

KernelSHAP replaces an exponential sum with weighted regression over sampled coalitions. PartitionSHAP restricts player orderings using a hierarchy or declared grouping. Both can
return efficient-looking vectors even when sampling or grouping changed the answer.

This notebook fixes an exact target first. Six players have direct dividends, and two known three-way interactions connect `(0, 1, 2)` and `(3, 4, 5)`. The exact Shapley vector is
`[1.2, 0.4, 1.1, -0.2, 0.1, -0.4]`.

<details><summary>Interpretation - what counts as failure?</summary>

An approximation fails when its maximum player error is large relative to the exact oracle, even if its values sum to the correct total. A partition result can satisfy efficiency
while allocating interaction credit differently.

</details>
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def all_coalitions(num_players: int) -> tuple[Coalition, ...]:
    if num_players <= 0:
        raise ValueError("num_players must be positive.")
    return tuple(
        frozenset(group)
        for size in range(num_players + 1)
        for group in itertools.combinations(range(num_players), size)
    )

def normalize_coalition_values(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    *,
    num_players: int,
) -> dict[Coalition, float]:
    normalized = {
        frozenset(key): float(value) for key, value in coalition_values.items()
    }
    expected = set(all_coalitions(num_players))
    missing = expected - set(normalized)
    extra = set(normalized) - expected
    if missing or extra:
        raise ValueError(
            f"coalition table has missing={len(missing)} and extra={len(extra)} coalitions"
        )
    return {coalition: normalized[coalition] for coalition in all_coalitions(num_players)}

def coalition_values_from_function(
    num_players: int,
    value_fn: Callable[[Coalition], float],
) -> dict[Coalition, float]:
    return {
        coalition: float(value_fn(coalition))
        for coalition in all_coalitions(num_players)
    }

def exact_shapley_values(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    *,
    num_players: int,
) -> t.Tensor:
    values = normalize_coalition_values(
        coalition_values, num_players=num_players
    )
    result = t.zeros(num_players, dtype=t.float64)
    denominator = math.factorial(num_players)
    for player in range(num_players):
        for coalition in all_coalitions(num_players):
            if player in coalition:
                continue
            size = len(coalition)
            weight = (
                math.factorial(size)
                * math.factorial(num_players - size - 1)
                / denominator
            )
            result[player] += weight * (
                values[coalition | {player}] - values[coalition]
            )
    return result

def additive_game(weights: t.Tensor) -> dict[Coalition, float]:
    weights = weights.detach().flatten().double()
    return coalition_values_from_function(
        int(weights.numel()),
        lambda coalition: weights[list(coalition)].sum().item()
        if coalition else 0.0,
    )

def conjunction_game(num_players: int) -> dict[Coalition, float]:
    full = frozenset(range(num_players))
    return coalition_values_from_function(
        num_players, lambda coalition: coalition == full
    )

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise 1 - build the exact ground-truth game

> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 15-20 minutes on this exercise.
> ```

Implement a game from Harsanyi dividends and an independent oracle that splits each dividend equally among its members. The baseline is zero, so every unit of full-game value
belongs to at least one player.

<details><summary>Help - common bug</summary>

Coalition `S` receives dividend `d(T)` exactly when `T` is a subset of `S`. For the oracle, each player in `T` receives `d(T) / len(T)`.

</details>

<details><summary>Expected output</summary>

```text
All tests in `test_exact_dividend_game_oracle` passed!
Exact oracle = [1.200, 0.400, 1.100, -0.200, 0.100, -0.400]
Full-game value = 2.200; oracle parity error < 1e-12
```

</details>

<details><summary>Interpretation</summary>

The oracle does not use KernelSHAP or partition orderings. It is the fixed target against which both methods will be tested.

</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Forecast — dividend-oracle

Before evaluating all 64 coalitions, forecast the six player credits and grand-coalition total from the eight declared dividends. State numeric ranges, confidence, causal reasoning
and exposure status. What changes if a new interaction connects the two previously separate triples?

Commit your forecast before continuing.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

PLAYER_NAMES = ("embed", "attend", "mlp", "router", "memory", "readout")

GAME_DIVIDENDS = {
    frozenset({0}): 0.6,
    frozenset({1}): -0.2,
    frozenset({2}): 0.5,
    frozenset({3}): 0.1,
    frozenset({4}): 0.4,
    frozenset({5}): -0.1,
    frozenset({0, 1, 2}): 1.8,
    frozenset({3, 4, 5}): -0.9,
}

# EXERCISE
# def game_from_dividends(
#     num_players: int,
#     dividends: Mapping[Coalition | tuple[int, ...], float],
# ) -> dict[Coalition, float]:
#     raise NotImplementedError()
# END EXERCISE
# SOLUTION
def game_from_dividends(
    num_players: int,
    dividends: Mapping[Coalition | tuple[int, ...], float],
) -> dict[Coalition, float]:
    normalized = {
        frozenset(term): float(value) for term, value in dividends.items()
    }
    if any(not term for term in normalized):
        raise ValueError("The empty dividend is a baseline, not player credit.")
    if any(player < 0 or player >= num_players for term in normalized for player in term):
        raise ValueError("Dividend contains an invalid player.")
    return coalition_values_from_function(
        num_players,
        lambda coalition: sum(
            value for term, value in normalized.items()
            if term.issubset(coalition)
        ),
    )
# END SOLUTION

# EXERCISE
# def shapley_from_dividends(
#     num_players: int,
#     dividends: Mapping[Coalition | tuple[int, ...], float],
# ) -> t.Tensor:
#     raise NotImplementedError()
# END EXERCISE
# SOLUTION
def shapley_from_dividends(
    num_players: int,
    dividends: Mapping[Coalition | tuple[int, ...], float],
) -> t.Tensor:
    result = t.zeros(num_players, dtype=t.float64)
    for raw_term, raw_value in dividends.items():
        term = frozenset(raw_term)
        if not term:
            raise ValueError("The empty dividend is a baseline, not player credit.")
        for player in term:
            result[player] += float(raw_value) / len(term)
    return result
# END SOLUTION

if MAIN:
    tests.test_exact_dividend_game_oracle(
        game_from_dividends,
        shapley_from_dividends,
    )

if MAIN:
    game_values = game_from_dividends(6, GAME_DIVIDENDS)

if MAIN:
    exact_oracle = shapley_from_dividends(6, GAME_DIVIDENDS)

if MAIN:
    exact_formula = exact_shapley_values(game_values, num_players=6)

if MAIN:
    oracle_parity_error = (exact_formula - exact_oracle).abs().max().item()

if MAIN:
    print("Exact oracle =", [round(value, 3) for value in exact_oracle.tolist()])

if MAIN:
    print(
        f"Full-game value = {game_values[frozenset(range(6))]:.3f}; "
        f"oracle parity error = {oracle_parity_error:.2e}"
    )

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details><summary>Solution</summary>

```python
SOLUTION
```
</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Reflect — dividend-oracle

Record observed values and all controls, identify your largest forecast discrepancy and explain its cause. Forecast a changed scientific condition with confidence and causal
reasoning. Distinguish retained reference outputs from what you ran; leave any unexecuted transfer explicitly unobserved.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Summary of 1

An independent dividend oracle fixes each player credit. Baseline and grand total alone cannot identify the attribution vector.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
# 2️⃣ Kernel regression and sampling
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise 2 - construct KernelSHAP regression rows

> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 15-20 minutes on this exercise.
> ```

Implement the finite Shapley kernel

$$\pi(S)=\frac{{n-1}}{{\binom{{n}}{{|S|}}|S|(n-|S|)}}$$

for non-empty, non-full coalitions. Then construct the binary mask matrix `X`, baseline-centered target `y`, and row weights `w`.

<details><summary>Help - common bug</summary>

Reject the empty and full coalitions in the finite kernel. They are enforced separately by the baseline and efficiency constraint. Every row of `X` contains one for present players
and zero otherwise.

</details>

<details><summary>Expected output</summary>

```text
All tests in `test_kernelshap_kernel_weight_uses_finite_coalition_formula` passed!
Kernel regression rows: X=(62, 6), y=(62,), w=(62,)
Singleton weight > size-3 weight; no empty/full finite rows
```

</details>

<details><summary>Interpretation</summary>

The kernel upweights extreme coalition sizes because they constrain individual marginal effects. Uniform row weighting solves a different regression problem.

</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Forecast — finite-kernel

Forecast the relative singleton versus size-three weights and the 62-row mask/design shapes. Explain why the endpoints are excluded from the finite kernel; record confidence,
causal reasoning and exposure status. Would uniform weights solve the same regression?

Commit your forecast before continuing.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

# EXERCISE
# def kernelshap_kernel_weight(coalition_size: int, num_players: int) -> float:
#     raise NotImplementedError()
# END EXERCISE
# SOLUTION
def kernelshap_kernel_weight(coalition_size: int, num_players: int) -> float:
    if not 0 < coalition_size < num_players:
        raise ValueError("Need 0 < coalition_size < num_players.")
    return (num_players - 1) / (
        math.comb(num_players, coalition_size)
        * coalition_size
        * (num_players - coalition_size)
    )
# END SOLUTION

# EXERCISE
# def kernelshap_regression_rows(
#     coalition_values: Mapping[Coalition | tuple[int, ...], float],
#     *,
#     num_players: int,
#     coalitions: Sequence[Coalition] | None = None,
# ) -> tuple[t.Tensor, t.Tensor, t.Tensor]:
#     raise NotImplementedError()
# END EXERCISE
# SOLUTION
def kernelshap_regression_rows(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    *,
    num_players: int,
    coalitions: Sequence[Coalition] | None = None,
) -> tuple[t.Tensor, t.Tensor, t.Tensor]:
    values = normalize_coalition_values(
        coalition_values, num_players=num_players
    )
    rows = (
        tuple(frozenset(coalition) for coalition in coalitions)
        if coalitions is not None
        else tuple(
            coalition for coalition in all_coalitions(num_players)
            if 0 < len(coalition) < num_players
        )
    )
    if not rows or any(len(coalition) in (0, num_players) for coalition in rows):
        raise ValueError("Regression rows must be non-empty interior coalitions.")
    X = t.tensor(
        [
            [float(player in coalition) for player in range(num_players)]
            for coalition in rows
        ],
        dtype=t.float64,
    )
    y = t.tensor(
        [values[coalition] - values[frozenset()] for coalition in rows],
        dtype=t.float64,
    )
    weights = t.tensor(
        [
            kernelshap_kernel_weight(len(coalition), num_players)
            for coalition in rows
        ],
        dtype=t.float64,
    )
    return X, y, weights
# END SOLUTION

if MAIN:
    tests.test_kernelshap_kernel_weight_uses_finite_coalition_formula(
        kernelshap_kernel_weight
    )

if MAIN:
    full_X, full_y, full_weights = kernelshap_regression_rows(
        game_values, num_players=6
    )

if MAIN:
    assert full_X.shape == (62, 6)

if MAIN:
    assert kernelshap_kernel_weight(1, 6) > kernelshap_kernel_weight(3, 6)

if MAIN:
    print(
        f"Kernel regression rows: X={tuple(full_X.shape)}, "
        f"y={tuple(full_y.shape)}, w={tuple(full_weights.shape)}"
    )

if MAIN:
    print("Kernel design-row tests passed!")

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details><summary>Solution</summary>

```python
SOLUTION
```
</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Reflect — finite-kernel

Record observed values and all controls, identify your largest forecast discrepancy and explain its cause. Forecast a changed scientific condition with confidence and causal
reasoning. Distinguish retained reference outputs from what you ran; leave any unexecuted transfer explicitly unobserved.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise 3 - solve constrained KernelSHAP

> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 15-20 minutes on this exercise.
> ```

Solve weighted least squares with the hard efficiency constraint `sum(phi) = v(full) - v(empty)`. Use a direct KKT solve for the complete interior table. Partial coalition designs
can be rank deficient, so define their estimator as the minimum-norm KKT least-squares solution.

<details><summary>Help - common bug</summary>

Form `X.T @ W @ X` and `X.T @ W @ y`. Add one row and column of ones to impose efficiency, and add the full-minus-empty delta to the right-hand side.

</details>

<details><summary>Expected output</summary>

```text
All additive and conjunction KernelSHAP tests passed!
Full-table KernelSHAP max error: < 1e-12
Efficiency error: < 1e-12
```

</details>

<details><summary>Interpretation</summary>

Full-table KernelSHAP is a parity check, not an approximation win: it has used all 62 interior coalitions. Sampling is where uncertainty enters.

</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Forecast — constrained-regression

Forecast full-table maximum player error and efficiency residual separately. Give ranges, confidence, causal reasoning and exposure status. Explain why efficiency alone cannot
establish exact attribution, and what changes for a rank-deficient partial design.

Commit your forecast before continuing.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

# EXERCISE
# def kernelshap_values(
#     coalition_values: Mapping[Coalition | tuple[int, ...], float],
#     *,
#     num_players: int,
#     coalitions: Sequence[Coalition] | None = None,
# ) -> t.Tensor:
#     raise NotImplementedError()
# END EXERCISE
# SOLUTION
def kernelshap_values(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    *,
    num_players: int,
    coalitions: Sequence[Coalition] | None = None,
) -> t.Tensor:
    values = normalize_coalition_values(
        coalition_values, num_players=num_players
    )
    X, y, weights = kernelshap_regression_rows(
        values,
        num_players=num_players,
        coalitions=coalitions,
    )
    gram = X.T @ (weights[:, None] * X)
    rhs = X.T @ (weights * y)
    kkt = t.zeros(
        (num_players + 1, num_players + 1), dtype=t.float64
    )
    kkt[:num_players, :num_players] = gram
    kkt[:num_players, -1] = 1.0
    kkt[-1, :num_players] = 1.0
    delta = (
        values[frozenset(range(num_players))]
        - values[frozenset()]
    )
    constrained_rhs = t.cat(
        [rhs, t.tensor([delta], dtype=t.float64)]
    )
    if X.shape[0] == 2**num_players - 2:
        return t.linalg.solve(kkt, constrained_rhs)[:num_players]
    # CPU SVD gives the declared minimum-norm KKT solution reliably.
    # CUDA supports only gels, so solve this small system on CPU and return
    # the coefficients to the original device.
    solution = t.linalg.lstsq(
        kkt.cpu(), constrained_rhs.cpu(), driver="gelsd"
    ).solution
    return solution[:num_players].to(kkt.device)
# END SOLUTION

# EXERCISE
# def kernelshap_approximation_report(
#     coalition_values: Mapping[Coalition | tuple[int, ...], float],
#     *,
#     num_players: int,
#     tolerance: float = 1e-9,
# ) -> KernelSHAPApproximationReport:
#     raise NotImplementedError()
# END EXERCISE
# SOLUTION
def kernelshap_approximation_report(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    *,
    num_players: int,
    tolerance: float = 1e-9,
) -> KernelSHAPApproximationReport:
    estimate = kernelshap_values(
        coalition_values, num_players=num_players
    )
    exact = exact_shapley_values(
        coalition_values, num_players=num_players
    )
    error = float((estimate - exact).abs().max().item())
    return KernelSHAPApproximationReport(
        shapley_values=estimate,
        exact_values=exact,
        max_abs_error=error,
        approximates_exact=error <= tolerance,
    )
# END SOLUTION

if MAIN:
    tests.test_kernelshap_approximation_report_matches_exact_additive_game(
        kernelshap_approximation_report,
        additive_game,
    )

if MAIN:
    tests.test_kernelshap_interaction_report_splits_conjunction_credit(
        kernelshap_approximation_report,
        conjunction_game,
    )

if MAIN:
    full_kernel = kernelshap_values(game_values, num_players=6)

if MAIN:
    full_kernel_error = (full_kernel - exact_oracle).abs().max().item()

if MAIN:
    full_kernel_efficiency_error = abs(
        full_kernel.sum().item() - game_values[frozenset(range(6))]
    )

if MAIN:
    print(f"Full-table KernelSHAP max error: {full_kernel_error:.2e}")

if MAIN:
    print(f"Efficiency error: {full_kernel_efficiency_error:.2e}")

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details><summary>Solution</summary>

```python
SOLUTION
```
</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Reflect — constrained-regression

Record observed values and all controls, identify your largest forecast discrepancy and explain its cause. Forecast a changed scientific condition with confidence and causal
reasoning. Distinguish retained reference outputs from what you ran; leave any unexecuted transfer explicitly unobserved.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise 4 - sample coalitions with controls

> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 15-20 minutes on this exercise.
> ```

Implement two sampling strategies without replacement. `uniform` samples interior coalitions uniformly. `paired_kernel` samples complement pairs with probability proportional to
the Shapley kernel, so every selected `S` is accompanied by `N minus S`.

<details><summary>Help - common bug</summary>

Build the 31 unique complement pairs by tracking a `seen` set. Use a seeded `torch.Generator` and `torch.multinomial` for weighted pair selection. Return all 62 interior coalitions
when the budget reaches 62.

</details>

<details><summary>Expected output</summary>

```text
All tests in `test_coalition_sampler_controls` passed!
All tests in `test_sampled_kernelshap_convergence_control` passed!
At budget 24, paired-kernel mean error < uniform mean error across fixed seeds
```

</details>

<details><summary>Interpretation</summary>

Complement pairing and kernel weighting reduce error on average for this game. They do not guarantee that every seed beats uniform sampling, and the comparison must use equal
coalition budgets.

</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Forecast — paired-sampling

At budget 24 and seeds 0–11, forecast the average maximum absolute player error for complement-paired kernel sampling versus uniform sampling. Freeze the same game,
distinct-coalition budget and KKT estimator. State ranges, confidence, causal reasoning and exposure status; distinguish a mean comparison from a per-seed guarantee.

Commit your forecast before continuing.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

# EXERCISE
# def sample_interior_coalitions(
#     num_players: int,
#     *,
#     budget: int,
#     seed: int,
#     strategy: Literal["paired_kernel", "uniform"],
# ) -> tuple[Coalition, ...]:
#     raise NotImplementedError()
# END EXERCISE
# SOLUTION
def sample_interior_coalitions(
    num_players: int,
    *,
    budget: int,
    seed: int,
    strategy: Literal["paired_kernel", "uniform"],
) -> tuple[Coalition, ...]:
    interior = [
        coalition for coalition in all_coalitions(num_players)
        if 0 < len(coalition) < num_players
    ]
    if budget <= 0:
        raise ValueError("budget must be positive.")
    if budget >= len(interior):
        return tuple(interior)
    generator = t.Generator().manual_seed(seed)
    if strategy == "uniform":
        indices = t.randperm(
            len(interior), generator=generator
        )[:budget]
        return tuple(interior[index] for index in indices.tolist())
    if strategy != "paired_kernel":
        raise ValueError("strategy must be 'paired_kernel' or 'uniform'.")
    if budget % 2:
        raise ValueError("paired_kernel requires an even budget.")

    full = frozenset(range(num_players))
    seen: set[Coalition] = set()
    pairs = []
    for coalition in interior:
        if coalition in seen:
            continue
        complement = full - coalition
        pairs.append((coalition, complement))
        seen.update((coalition, complement))
    pair_weights = t.tensor(
        [
            kernelshap_kernel_weight(len(pair[0]), num_players)
            for pair in pairs
        ],
        dtype=t.float64,
    )
    pair_ids = t.multinomial(
        pair_weights,
        budget // 2,
        replacement=False,
        generator=generator,
    )
    return tuple(
        coalition
        for index in pair_ids.tolist()
        for coalition in pairs[index]
    )
# END SOLUTION

if MAIN:
    tests.test_coalition_sampler_controls(sample_interior_coalitions)

if MAIN:
    tests.test_sampled_kernelshap_convergence_control(
        game_from_dividends,
        shapley_from_dividends,
        sample_interior_coalitions,
        kernelshap_values,
    )

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details><summary>Solution</summary>

```python
SOLUTION
```
</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Reflect — paired-sampling

Record observed values and all controls, identify your largest forecast discrepancy and explain its cause. Forecast a changed scientific condition with confidence and causal
reasoning. Distinguish retained reference outputs from what you ran; leave any unexecuted transfer explicitly unobserved.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Summary of 2

Finite kernel weights and the hard efficiency constraint define the regression. Compare distinct coalition budgets and retain every sampling seed.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
# 3️⃣ Restricted orderings and Owen values
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise 5 - validate and collapse a partition

> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 15-20 minutes on this exercise.
> ```

A partition must cover every player exactly once. Implement validation, then map group coalitions back to their underlying player coalitions before looking up game values.

<details><summary>Help - common bug</summary>

Flatten all groups and compare the sorted players with `range(num_players)`. For group coalition `{1}`, union all players in `groups[1]`.

</details>

<details><summary>Expected output</summary>

```text
All tests in `test_grouped_coalition_values_expand_groups_before_lookup` passed!
Invalid overlapping and incomplete partitions are rejected
```

</details>

<details><summary>Interpretation</summary>

A group index is not a player index. Mixing these spaces can produce plausible totals while assigning credit to the wrong original variables.

</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Forecast — group-lookup

Forecast the group-coalition table for the additive four-player example, and which overlapping or incomplete partitions fail validation. State confidence, causal reasoning and
exposure status. Which space does a group index refer to?

Commit your forecast before continuing.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

# EXERCISE
# def validate_partition(
#     groups: tuple[tuple[int, ...], ...],
#     num_players: int,
# ) -> None:
#     raise NotImplementedError()
# END EXERCISE
# SOLUTION
def validate_partition(
    groups: tuple[tuple[int, ...], ...],
    num_players: int,
) -> None:
    players = [player for group in groups for player in group]
    if (
        not groups
        or any(not group for group in groups)
        or sorted(players) != list(range(num_players))
        or len(players) != len(set(players))
    ):
        raise ValueError(
            "partition players must cover each player exactly once"
        )
# END SOLUTION

# EXERCISE
# def grouped_coalition_values(
#     coalition_values: Mapping[Coalition | tuple[int, ...], float],
#     *,
#     groups: tuple[tuple[int, ...], ...],
# ) -> dict[Coalition, float]:
#     raise NotImplementedError()
# END EXERCISE
# SOLUTION
def grouped_coalition_values(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    *,
    groups: tuple[tuple[int, ...], ...],
) -> dict[Coalition, float]:
    num_players = len(
        {player for group in groups for player in group}
    )
    validate_partition(groups, num_players)
    values = normalize_coalition_values(
        coalition_values, num_players=num_players
    )
    return coalition_values_from_function(
        len(groups),
        lambda group_coalition: values[
            frozenset(
                player
                for group_index in group_coalition
                for player in groups[group_index]
            )
        ],
    )
# END SOLUTION

if MAIN:
    tests.test_grouped_coalition_values_expand_groups_before_lookup(
        grouped_coalition_values,
        additive_game,
    )

if MAIN:
    for invalid_groups in (((0, 1), (1, 2)), ((0, 1),)):
        try:
            validate_partition(invalid_groups, 3)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid partition should fail.")

if MAIN:
    print("Partition validation tests passed!")

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details><summary>Solution</summary>

```python
SOLUTION
```
</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Reflect — group-lookup

Record observed values and all controls, identify your largest forecast discrepancy and explain its cause. Forecast a changed scientific condition with confidence and causal
reasoning. Distinguish retained reference outputs from what you ran; leave any unexecuted transfer explicitly unobserved.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise 6 - implement exact Owen values

> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 15-20 minutes on this exercise.
> ```

For each player, average over which groups precede its group and which players precede it inside that group. This is the exact Owen-value calculation for a fixed partition, the
cooperative-game object underlying the partition control in this lesson.

<details><summary>Help - common bug</summary>

Use a Shapley subset weight twice: once among groups and once within the player's group. The predecessor coalition is the union of every player in preceding groups plus preceding
players in the current group.

</details>

<details><summary>Expected output</summary>

```text
Additive, symmetric-interaction, invalid-group, and cross-group tests all pass
Singleton groups reduce to ordinary exact Shapley
```

</details>

<details><summary>Interpretation</summary>

Owen values are exact for the game with a priori unions. They need not equal ordinary Shapley because the partition changes which predecessor orderings are allowed.

</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Forecast — owen-orderings

Forecast singleton and one-group Owen parity, and the credit of the irrelevant grouped player in the cross-group conjunction fixture. Give numeric ranges, confidence, causal
reasoning and exposure status. Explain when restricted orderings can change ordinary Shapley credit.

Commit your forecast before continuing.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

# EXERCISE
# def partition_shap_values(
#     coalition_values: Mapping[Coalition | tuple[int, ...], float],
#     *,
#     groups: tuple[tuple[int, ...], ...],
# ) -> t.Tensor:
#     raise NotImplementedError()
# END EXERCISE
# SOLUTION
def partition_shap_values(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    *,
    groups: tuple[tuple[int, ...], ...],
) -> t.Tensor:
    num_players = sum(len(group) for group in groups)
    validate_partition(groups, num_players)
    values = normalize_coalition_values(
        coalition_values, num_players=num_players
    )
    result = t.zeros(num_players, dtype=t.float64)
    num_groups = len(groups)
    for group_index, group in enumerate(groups):
        other_groups = [
            index for index in range(num_groups)
            if index != group_index
        ]
        for player in group:
            inside_players = [item for item in group if item != player]
            for outside_size in range(num_groups):
                outside_weight = (
                    math.factorial(outside_size)
                    * math.factorial(num_groups - outside_size - 1)
                    / math.factorial(num_groups)
                )
                for outside_indices in itertools.combinations(
                    other_groups, outside_size
                ):
                    outside = (
                        set().union(
                            *(set(groups[index]) for index in outside_indices)
                        )
                        if outside_indices else set()
                    )
                    for inside_size in range(len(group)):
                        inside_weight = (
                            math.factorial(inside_size)
                            * math.factorial(len(group) - inside_size - 1)
                            / math.factorial(len(group))
                        )
                        for inside_before in itertools.combinations(
                            inside_players, inside_size
                        ):
                            before = frozenset(
                                outside | set(inside_before)
                            )
                            result[player] += (
                                outside_weight
                                * inside_weight
                                * (
                                    values[before | {player}]
                                    - values[before]
                                )
                            )
    return result
# END SOLUTION

# EXERCISE
# def partition_shap_report(
#     coalition_values: Mapping[Coalition | tuple[int, ...], float],
#     *,
#     groups: tuple[tuple[int, ...], ...],
#     tolerance: float = 1e-9,
# ) -> PartitionSHAPReport:
#     raise NotImplementedError()
# END EXERCISE
# SOLUTION
def partition_shap_report(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    *,
    groups: tuple[tuple[int, ...], ...],
    tolerance: float = 1e-9,
) -> PartitionSHAPReport:
    num_players = sum(len(group) for group in groups)
    player_values = partition_shap_values(
        coalition_values, groups=groups
    )
    group_values = exact_shapley_values(
        grouped_coalition_values(coalition_values, groups=groups),
        num_players=len(groups),
    )
    exact_values = exact_shapley_values(
        coalition_values, num_players=num_players
    )
    error = float(
        (player_values - exact_values).abs().max().item()
    )
    return PartitionSHAPReport(
        group_values=group_values,
        player_values=player_values,
        exact_values=exact_values,
        max_abs_error=error,
        recovers_exact=error <= tolerance,
    )
# END SOLUTION

if MAIN:
    tests.test_partition_shap_report_recovers_additive_groups(
        partition_shap_report,
        additive_game,
    )

if MAIN:
    tests.test_partition_shap_report_rejects_invalid_grouping(
        partition_shap_report,
        additive_game,
    )

if MAIN:
    tests.test_partition_shap_report_splits_interaction_group_symmetrically(
        partition_shap_report,
        conjunction_game,
    )

if MAIN:
    tests.test_partition_shap_report_handles_cross_group_interaction_without_irrelevant_credit(
        partition_shap_report,
        coalition_values_from_function,
    )

if MAIN:
    singleton_partition = tuple((player,) for player in range(6))

if MAIN:
    singleton_values = partition_shap_values(
        game_values, groups=singleton_partition
    )

if MAIN:
    t.testing.assert_close(singleton_values, exact_oracle, atol=1e-12, rtol=0)

if MAIN:
    print("PartitionSHAP semantic tests passed!")

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details><summary>Solution</summary>

```python
SOLUTION
```
</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Reflect — owen-orderings

Record observed values and all controls, identify your largest forecast discrepancy and explain its cause. Forecast a changed scientific condition with confidence and causal
reasoning. Distinguish retained reference outputs from what you ran; leave any unexecuted transfer explicitly unobserved.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise 7 - test aligned and random partitions

> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 15-20 minutes on this exercise.
> ```

The declared interactions are entirely inside `((0, 1, 2), (3, 4, 5))`, so that aligned partition matches ordinary Shapley. Test it against the fixed misaligned partition `((0, 3,
4), (1, 2, 5))`, then sample 100 seeded balanced partitions.

<details><summary>Help - common bug</summary>

Generate a seeded random permutation of six players and use the first three and last three as groups. Measure each partition by maximum absolute error from the exact oracle.
Preserve every sampled result, including repeated partitions, because the seeds are the experimental units.

</details>

<details><summary>Solution</summary>

```python
aligned_groups = ((0, 1, 2), (3, 4, 5))
misaligned_groups = ((0, 3, 4), (1, 2, 5))
aligned = partition_shap_values(game_values, groups=aligned_groups)
misaligned = partition_shap_values(game_values, groups=misaligned_groups)
random_errors = []
for seed in range(100):
    ordering = t.randperm(6, generator=t.Generator().manual_seed(seed)).tolist()
    groups = (tuple(sorted(ordering[:3])), tuple(sorted(ordering[3:])))
    random_errors.append((partition_shap_values(game_values, groups=groups) - exact_oracle).abs().max().item())
```

</details>

<details><summary>Expected output</summary>

```text
All tests in `test_partition_alignment_and_random_control` passed!
Aligned error = 0.000; fixed misaligned error = 0.300
91 / 100 seeded random balanced partitions differ from exact Shapley
```

</details>

<details><summary>Interpretation</summary>

Efficiency alone cannot detect the misaligned result: both vectors sum to `2.2`. The error comes from changing the allowed orderings, which redistributes each three-way dividend
according to how its members are split across groups.

</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Forecast — partition-alignment

For the original two three-way interactions, forecast aligned and fixed-misaligned maximum player error and the number of nonexact random balanced partitions among seeds 0–99.
State confidence, causal reasoning and exposure status. Retain efficiency as a distinct diagnostic.

Commit your forecast before continuing.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

if MAIN:
    tests.test_partition_alignment_and_random_control(
        game_from_dividends,
        shapley_from_dividends,
        partition_shap_values,
    )

if MAIN:
    aligned_groups = ((0, 1, 2), (3, 4, 5))

if MAIN:
    misaligned_groups = ((0, 3, 4), (1, 2, 5))

if MAIN:
    aligned_values = partition_shap_values(game_values, groups=aligned_groups)

if MAIN:
    misaligned_values = partition_shap_values(game_values, groups=misaligned_groups)

if MAIN:
    aligned_error = (aligned_values - exact_oracle).abs().max().item()

if MAIN:
    misaligned_error = (misaligned_values - exact_oracle).abs().max().item()

if MAIN:
    random_partition_errors = []

if MAIN:
    random_partitions = []

if MAIN:
    for seed in range(100):
        generator = t.Generator().manual_seed(seed)
        ordering = t.randperm(6, generator=generator).tolist()
        groups = (
            tuple(sorted(ordering[:3])),
            tuple(sorted(ordering[3:])),
        )
        values = partition_shap_values(game_values, groups=groups)
        random_partitions.append(groups)
        random_partition_errors.append(
            (values - exact_oracle).abs().max().item()
        )

if MAIN:
    nonexact_random_partitions = sum(
        error > 1e-9 for error in random_partition_errors
    )

if MAIN:
    print(f"Aligned error = {aligned_error:.3f}")

if MAIN:
    print(f"Fixed misaligned error = {misaligned_error:.3f}")

if MAIN:
    print(
        f"{nonexact_random_partitions} / 100 seeded random balanced partitions "
        "differ from exact Shapley"
    )

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Reflect — partition-alignment

Record observed values and all controls, identify your largest forecast discrepancy and explain its cause. Forecast a changed scientific condition with confidence and causal
reasoning. Distinguish retained reference outputs from what you ran; leave any unexecuted transfer explicitly unobserved.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Summary of 3

A priori unions restrict predecessor orderings. Exact Owen credit can differ from ordinary Shapley while satisfying efficiency.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
# 4️⃣ A controlled signature
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Signature Result

Run 50 fixed seeds at six equal coalition budgets. Compare paired-kernel and uniform sampling, then preserve a randomized-value negative control that shuffles all interior
coalition values while keeping the empty and full endpoints fixed.

The partition panels compare ordinary exact Shapley with aligned, singleton, misaligned, and random balanced partitions.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Forecast — equal-budget-signature

At budget 24, forecast paired_kernel mean maximum player error minus uniform mean maximum player error over seeds 0–49, in attribution units. Give a range, confidence, causal
reasoning and exposure status. Also forecast the endpoint-preserving shuffled-value control and partition controls. The plotted bands are seed standard deviations, not confidence
intervals. For transfer, add a new four-way dividend on {0,1,2,3} of +0.8, freeze all other dividends/budgets/masks/seeds/groupings, and forecast the same gap and formerly aligned
Owen error before running that new game.

Commit your forecast before continuing.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

if MAIN:
    budgets = t.tensor([12, 16, 24, 32, 48, 62])

if MAIN:
    seeds = range(50)

if MAIN:
    sampling_errors = {}

if MAIN:
    for strategy in ("paired_kernel", "uniform"):
        matrix = t.zeros((len(budgets), len(tuple(seeds))), dtype=t.float64)
        for budget_index, budget in enumerate(budgets.tolist()):
            for seed in seeds:
                sample = sample_interior_coalitions(
                    6,
                    budget=budget,
                    seed=seed,
                    strategy=strategy,
                )
                estimate = kernelshap_values(
                    game_values,
                    num_players=6,
                    coalitions=sample,
                )
                matrix[budget_index, seed] = (
                    estimate - exact_oracle
                ).abs().max()
        sampling_errors[strategy] = matrix

if MAIN:
    interior = tuple(
        coalition for coalition in all_coalitions(6)
        if 0 < len(coalition) < 6
    )

if MAIN:
    randomized_value_errors = []

if MAIN:
    for seed in seeds:
        generator = t.Generator().manual_seed(100 + seed)
        shuffled = dict(game_values)
        interior_values = t.tensor(
            [game_values[coalition] for coalition in interior],
            dtype=t.float64,
        )
        permutation = t.randperm(len(interior), generator=generator)
        for coalition, value in zip(interior, interior_values[permutation].tolist()):
            shuffled[coalition] = value
        randomized_estimate = kernelshap_values(
            shuffled, num_players=6
        )
        randomized_value_errors.append(
            (randomized_estimate - exact_oracle).abs().max().item()
        )

if MAIN:
    paired_mean = sampling_errors["paired_kernel"].mean(dim=1)

if MAIN:
    paired_std = sampling_errors["paired_kernel"].std(dim=1)

if MAIN:
    uniform_mean = sampling_errors["uniform"].mean(dim=1)

if MAIN:
    uniform_std = sampling_errors["uniform"].std(dim=1)

if MAIN:
    randomized_mean = float(np.mean(randomized_value_errors))

if MAIN:
    assert full_kernel_error < 1e-12

if MAIN:
    assert paired_mean[2] < uniform_mean[2]

if MAIN:
    assert aligned_error < 1e-12

if MAIN:
    assert abs(misaligned_error - 0.3) < 1e-12

if MAIN:
    assert randomized_mean > 0.5

if MAIN:
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.1))

if MAIN:
    axes[0].plot(budgets, paired_mean, marker="o", color=PAIRED, label="paired-kernel sample")

if MAIN:
    axes[0].fill_between(budgets, paired_mean - paired_std, paired_mean + paired_std, color=PAIRED, alpha=0.16)

if MAIN:
    axes[0].plot(budgets, uniform_mean, marker="o", color=UNIFORM, label="uniform sample")

if MAIN:
    axes[0].fill_between(budgets, uniform_mean - uniform_std, uniform_mean + uniform_std, color=UNIFORM, alpha=0.16)

if MAIN:
    axes[0].axhline(randomized_mean, color=CONTROL, linestyle="--", label="shuffled-value control")

if MAIN:
    axes[0].scatter([62], [full_kernel_error], color=EXACT, marker="*", s=150, label="full-table parity")

if MAIN:
    axes[0].set_xlabel("interior coalition budget")

if MAIN:
    axes[0].set_ylabel("max absolute player error")

if MAIN:
    axes[0].set_title("KernelSHAP convergence (50 seeds)")

if MAIN:
    axes[0].legend(frameon=False, fontsize=7)

if MAIN:
    x = np.arange(6)

if MAIN:
    width = 0.25

if MAIN:
    axes[1].bar(x - width, exact_oracle, width, color=EXACT, label="exact Shapley")

if MAIN:
    axes[1].bar(x, aligned_values, width, color=PAIRED, label="aligned partition")

if MAIN:
    axes[1].bar(x + width, misaligned_values, width, color=UNIFORM, label="misaligned partition")

if MAIN:
    axes[1].axhline(0, color="#B8BDC5", linewidth=1)

if MAIN:
    axes[1].set_xticks(x, PLAYER_NAMES, rotation=25)

if MAIN:
    axes[1].set_ylabel("player attribution")

if MAIN:
    axes[1].set_title("Partition changes interaction credit")

if MAIN:
    axes[1].legend(frameon=False, fontsize=7)

if MAIN:
    bins = np.linspace(-0.01, 0.325, 12)

if MAIN:
    axes[2].hist(random_partition_errors, bins=bins, color=CONTROL, alpha=0.78, label="100 random balanced partitions")

if MAIN:
    axes[2].axvline(aligned_error, color=PAIRED, linewidth=2, label="aligned control")

if MAIN:
    axes[2].axvline(misaligned_error, color=UNIFORM, linestyle="--", linewidth=2, label="fixed misaligned")

if MAIN:
    axes[2].set_xlabel("max absolute error vs exact Shapley")

if MAIN:
    axes[2].set_ylabel("seed count")

if MAIN:
    axes[2].set_title("Random partition control")

if MAIN:
    axes[2].legend(frameon=False, fontsize=7)

if MAIN:
    fig.suptitle(
        "KernelSHAP needs sampling controls; PartitionSHAP needs partition controls",
        fontsize=13,
        fontweight="bold",
    )

if MAIN:
    fig.tight_layout()

if MAIN:
    signature_path = assets_dir / "kernelshap_partition_controls_signature.png"

if MAIN:
    fig.savefig(signature_path, dpi=170, bbox_inches="tight")

if MAIN:
    plt.show()

if MAIN:
    print(
        f"Signature metrics: full_error={full_kernel_error:.2e}, "
        f"budget24_paired={paired_mean[2]:.3f}, "
        f"budget24_uniform={uniform_mean[2]:.3f}, "
        f"shuffled_value_error={randomized_mean:.3f}, "
        f"aligned_error={aligned_error:.3f}, "
        f"misaligned_error={misaligned_error:.3f}, "
        f"random_nonexact={nonexact_random_partitions}/100"
    )

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: [html]

r'''
<img src="media/kernelshap_partition_controls_signature.png" width="1000">
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Reflect — equal-budget-signature

Record observed values and all controls, identify your largest forecast discrepancy and explain its cause. Forecast a changed scientific condition with confidence and causal
reasoning. Distinguish retained reference outputs from what you ran; leave any unexecuted transfer explicitly unobserved.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Corrected current CPU reference

The current controlled figure is generated by the corrected CPU SVD solver. Its fixed budget-24 means are approximately 0.242120109 (paired design) and 0.477168479 (uniform
design), a difference of −0.235048370 attribution units. The original rounded publication values remain consistent, but this does not establish identity of every old per-seed
estimate. Kernel-proportional complement-pair sampling followed by kernel-weighted regression is the complete estimator being compared; this does not isolate pairing alone or
establish a generic unbiased sampler.

<details><summary>Interpreting the signature result</summary>

Paired-kernel sampling dominates uniform sampling on average at every partial budget in this organism, and both reach exact parity only at all 62 interior coalitions. Shuffling
coalition values preserves the endpoints and therefore efficiency, but destroys the player attribution. The aligned and singleton partitions recover exact Shapley; the misaligned
and most random partitions conserve total value while redistributing interaction credit.

</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Summary of 4

Read the fifty-seed sampling comparison and hundred-seed partition histogram separately. Standard-deviation bands describe seed dispersion.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
# 5️⃣ Play with the estimator
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Try It Yourself - change budget, seed, and partition

Set the three parameters before inspecting the answer. Compare both samplers at the same budget, and always report the exact-oracle error alongside the attribution vector.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Forecast — budget-and-group-play

Freeze the budget, seed, strategy and explicit groups before rerunning the play. Forecast the six attribution values and maximum oracle error, with confidence, causal reasoning and
exposure status. Compare the same-budget alternative sampler and full-table control; changing a seed alone is replication, whereas changing the game or grouping changes the
scientific condition.

Commit your forecast before continuing.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

if MAIN:
    PLAY_BUDGET = 24

if MAIN:
    PLAY_SEED = 4

if MAIN:
    PLAY_STRATEGY: Literal["paired_kernel", "uniform"] = "paired_kernel"

if MAIN:
    PLAY_GROUPS = ((0, 3, 4), (1, 2, 5))

if MAIN:
    play_sample = sample_interior_coalitions(
        6,
        budget=PLAY_BUDGET,
        seed=PLAY_SEED,
        strategy=PLAY_STRATEGY,
    )

if MAIN:
    play_kernel = kernelshap_values(
        game_values, num_players=6, coalitions=play_sample
    )

if MAIN:
    play_partition = partition_shap_values(
        game_values, groups=PLAY_GROUPS
    )

if MAIN:
    print(
        f"KernelSHAP ({PLAY_STRATEGY}, budget={PLAY_BUDGET}, seed={PLAY_SEED}):",
        [round(value, 3) for value in play_kernel.tolist()],
    )

if MAIN:
    print(
        "Kernel max error:",
        round((play_kernel - exact_oracle).abs().max().item(), 3),
    )

if MAIN:
    print("Partition:", PLAY_GROUPS)

if MAIN:
    print(
        "Partition values:",
        [round(value, 3) for value in play_partition.tolist()],
    )

if MAIN:
    print(
        "Partition max error:",
        round((play_partition - exact_oracle).abs().max().item(), 3),
    )

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Reflect — budget-and-group-play

Record observed values and all controls, identify your largest forecast discrepancy and explain its cause. Forecast a changed scientific condition with confidence and causal
reasoning. Distinguish retained reference outputs from what you ran; leave any unexecuted transfer explicitly unobserved.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Summary of 5

The original live play recomputes values from its budget, strategy, seed and groups. Keep every control and forecast a changed condition.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
# ☆ Bonus
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Bonus anomaly hunt

Find a seed at budget `24` where uniform sampling beats paired-kernel sampling, even though paired-kernel wins on average. Explain why one-seed screenshots are weak evidence. Then
inspect the nine random-partition seeds that happen to recover exact Shapley: which partitions keep both three-way dividend sets intact?

For a harder anomaly, add a new dividend on `{0, 1, 3}`. Predict which previously aligned partition now changes credit before rerunning the controls.

## Real-method connection

`shap.KernelExplainer` uses the same weighted linear-regression view, but practical runs sample masks and depend on the masking/background definition. `shap.PartitionExplainer`
restricts coalitions using a hierarchy and can compute Owen values. The exact finite game here isolates approximation and grouping error before those choices are mixed with model
and masking behavior.

## Limitations

- The six-player game has 64 exact coalitions; real attribution problems rarely expose a complete, trustworthy table.
- The paired sampler is one transparent control, not a claim of optimal sampling for every game.
- The partition is supplied directly. A learned clustering can introduce additional estimation error.
- Owen values are correct for their restricted-ordering game even when they differ from ordinary Shapley.
- This notebook does not test token masking, background datasets, conditional expectations, or model-distribution shift.

## Reading links

- [A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874)
- [SHAP KernelExplainer documentation](https://shap.readthedocs.io/en/latest/generated/shap.KernelExplainer.html)
- [SHAP PartitionExplainer documentation](https://shap.readthedocs.io/en/latest/generated/shap.PartitionExplainer.html)
- [Values of Games with A Priori Unions](https://doi.org/10.1007/BF01774766)
- [Section 16.1 exact-Shapley notebook](../part1_exact_shapley_ground_truth_games/16.1_Exact_Shapley_on_Ground_Truth_Games_exercises.ipynb)
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Verification Appendix

The exercises and signature result run on CPU. The full verification entry point trains and evaluates the finite neural coalition-game preflight on CUDA; `verification_report.json`
is supporting regression evidence, not the lesson.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

from chapter16_shapley_attribution_baselines.exercises.part2_kernelshap_partition_shap_controls import solutions as reference

def run_smoke_test(cpu: bool = True) -> dict:
    return reference.run_smoke_test(cpu=cpu)

def run_gpu_test(max_vram_gb: float = 24.0) -> dict:
    return reference.run_gpu_test(max_vram_gb=max_vram_gb)

def run_full_experiment(max_vram_gb: float = 24.0) -> dict:
    return reference.run_full_experiment(max_vram_gb=max_vram_gb)

if MAIN:
    verification_summary = run_smoke_test(cpu=True)

if MAIN:
    verification_summary

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Try It Yourself — plot the changed estimate

The original live printout above is preserved. This additional CPU play recomputes both samplers, full-table regression and requested/aligned/singleton Owen controls from explicit
knobs, then plots every attribution and maximum oracle error. It uses the same six-player dividend game; it is fresh analytic play, separate from the retained fifty-seed signature
and historical CUDA preflight.

### Forecast — live-plot-play

Set the coalition budget, seed, highlighted sampler and explicit partition before running. Forecast the player-wise credit deviations and which controls have zero maximum error.
Record confidence, causal reasoning and exposure status. Change one condition, rerun this cell and explain the discrepancy. Switching seeds is replication; changing the declared
groups changes the ordering game. The new four-way-dividend transfer proposed above remains unexecuted until you commit its separate forecast.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def compare_kernelshap_play(*, dividends, budget: int, seed: int, groups) -> dict:
    """Recompute both sampled estimators and partition controls on one exact game."""
    values = game_from_dividends(6, dividends)
    oracle = shapley_from_dividends(6, dividends)
    full = kernelshap_values(values, num_players=6)
    estimates = {"Full interior table": full}
    samples = {}
    for strategy, label in (("paired_kernel", "Paired-kernel sample"), ("uniform", "Uniform sample")):
        samples[strategy] = sample_interior_coalitions(6, budget=budget, seed=seed, strategy=strategy)
        estimates[label] = kernelshap_values(values, num_players=6, coalitions=samples[strategy])
    estimates["Requested Owen groups"] = partition_shap_values(values, groups=groups)
    estimates["Aligned Owen groups"] = partition_shap_values(values, groups=((0, 1, 2), (3, 4, 5)))
    estimates["Singleton Owen groups"] = partition_shap_values(values, groups=tuple((i,) for i in range(6)))
    return {
        "dividends": dividends, "budget": budget, "seed": seed, "groups": groups,
        "values": values, "oracle": oracle, "samples": samples, "estimates": estimates,
        "errors": {name: float((estimate - oracle).abs().max()) for name, estimate in estimates.items()},
    }


def check_kernelshap_play(result: dict) -> None:
    """Independent ordering oracle and endpoint/pairing checks for the live play."""
    # This enumerates all player orderings, independently of dividend splitting.
    permutation_credit = np.zeros(6, dtype=np.float64)
    for ordering in itertools.permutations(range(6)):
        before = frozenset()
        for player in ordering:
            after = before | {player}
            permutation_credit[player] += result["values"][after] - result["values"][before]
            before = after
    expected = t.tensor(permutation_credit / math.factorial(6), dtype=t.float64)
    t.testing.assert_close(result["oracle"], expected, atol=1e-12, rtol=0)
    t.testing.assert_close(result["estimates"]["Full interior table"], expected, atol=1e-12, rtol=0)
    total = result["values"][frozenset(range(6))] - result["values"][frozenset()]
    for estimate in result["estimates"].values():
        assert abs(float(estimate.sum()) - total) < 1e-10
    paired = result["samples"]["paired_kernel"]
    assert len(paired) == len(set(paired)) == min(result["budget"], 62)
    assert all((frozenset(range(6)) - coalition) in paired for coalition in paired)
    print("Permutation oracle, full-table parity, efficiency and complement controls passed.")


if MAIN:
    # Change one knob, commit a forecast, then rerun only this CPU play cell.
    PLAY_PLOT_BUDGET = 24
    PLAY_PLOT_SEED = 4
    PLAY_PLOT_STRATEGY = "paired_kernel"
    PLAY_PLOT_GROUPS = ((0, 3, 4), (1, 2, 5))
    if PLAY_PLOT_STRATEGY not in ("paired_kernel", "uniform"):
        raise ValueError("Choose paired_kernel or uniform for the highlighted estimate.")
    plotted_play = compare_kernelshap_play(
        dividends=GAME_DIVIDENDS, budget=PLAY_PLOT_BUDGET,
        seed=PLAY_PLOT_SEED, groups=PLAY_PLOT_GROUPS,
    )
    check_kernelshap_play(plotted_play)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.5), constrained_layout=True)
    axes[0].plot(range(6), plotted_play["oracle"], color=EXACT, marker="*", linewidth=2.5, label="Exact dividend/permutation oracle")
    selected_label = "Paired-kernel sample" if PLAY_PLOT_STRATEGY == "paired_kernel" else "Uniform sample"
    for name, estimate in plotted_play["estimates"].items():
        axes[0].plot(range(6), estimate, marker="o", linewidth=2.5 if name == selected_label else 1, alpha=.85, label=name)
    axes[0].set_xticks(range(6), PLAYER_NAMES, rotation=20)
    axes[0].set(xlabel="Player", ylabel="Attribution", title="Live CPU play: credit and all controls")
    axes[0].legend(fontsize=7)
    names = list(plotted_play["errors"])
    axes[1].bar(range(len(names)), list(plotted_play["errors"].values()), color=[CONTROL, PAIRED, UNIFORM, "#8B6C9B", PAIRED, CONTROL])
    axes[1].set_xticks(range(len(names)), names, rotation=25, ha="right")
    axes[1].set(ylabel="Maximum absolute player error", title="Full-table, sampler and partition comparisons")
    fig.suptitle(f"One exact game: budget={PLAY_PLOT_BUDGET}, seed={PLAY_PLOT_SEED}; highlighted={PLAY_PLOT_STRATEGY}")
    (root_dir / chapter / "exercises" / section / "media").mkdir(exist_ok=True)
    fig.savefig(root_dir / chapter / "exercises" / section / "media/kernelshap_live_play.png", dpi=160)
    plt.show()
    print("Live CPU play max errors:", {name: round(error, 6) for name, error in plotted_play["errors"].items()})

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: [html]

r'''
<img src="media/kernelshap_live_play.png" width="1000">
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Reflect — live-plot-play

Record all displayed maximum errors and at least one player whose sampled or Owen credit differs from the exact oracle. Explain whether the discrepancy comes from sampled design
rows or restricted predecessor orderings. Forecast a changed grouping or game with confidence and causal reasoning, and preserve both the exact target and every control. Mark the
result as fresh CPU analytic play; it does not reproduce the historical trained-neural-game report.
'''

