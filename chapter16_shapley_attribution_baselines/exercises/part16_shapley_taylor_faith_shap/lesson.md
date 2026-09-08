<!-- ARENA:cell markdown -->
# [16.16] Where Does Higher-Order Credit Go?

> **Shapley-Taylor and Faith-Shap from complete games**
>
> This lesson runs on CPU. You will implement the interaction methods yourself, check them on exact
> games, and then apply them to a frozen finite-background prediction game.

<!-- ARENA:cell markdown -->
# Introduction

Suppose a model only turns on when **four ingredients are present together**, but your explanation is
allowed to show only single features and pairs. There is no literal one-feature or two-feature term in
that model. An order-two explanation still has to decide where the four-way credit goes.

That decision is not bookkeeping trivia. Different interaction indices answer different questions.
This lesson puts three of them on the same complete coalition game:

- **Shapley-Taylor** preserves lower-order Möbius dividends and routes omitted dividends into the
  highest retained order.
- **Faith-Shap** chooses all retained coefficients jointly to minimize a declared Shapley-kernel
  reconstruction error, subject to exact empty/full constraints.
- **SII** averages a selected discrete derivative over contexts. Its singleton Shapley values and
  pair SII values are separate index families, not one efficient order-two decomposition.

You will see all three give exact, internally valid, and different pair answers.

<!-- ARENA:cell markdown -->
## Core Question

> When an explanation can contain only singletons and pairs, where should the credit for a true
> three-way or four-way mechanism go?

Two efficient order-two ledgers are possible for the pure four-player unanimity game. Before reading
on, predict which one will better reconstruct the interior coalitions:

| anonymous rule | each singleton | each pair | endpoint ledger |
|---|---:|---:|---:|
| A | `0` | `1/6` | `6 × 1/6 = 1` |
| B | `-1/5` | `3/10` | `4 × (-1/5) + 6 × (3/10) = 1` |

Both satisfy the full-coalition ledger, so efficiency alone cannot decide. Write down your prediction,
including the reconstructed value of a coalition containing just one feature, before opening the answer.

<details>
<summary>Reveal the two rules</summary>

| definition | each singleton | each pair | endpoint ledger |
|---|---:|---:|---:|
| Shapley-Taylor, order 2 | `0` | `1/6` | `6 × 1/6 = 1` |
| Faith-Shap, order 2 | `-1/5` | `3/10` | `4 × (-1/5) + 6 × (3/10) = 1` |
| SII convention | Shapley `1/4` | pair SII `1/3` | separate families; do not sum as one decomposition |

The disagreement is the result we will explain.

</details>

<!-- ARENA:cell code -->
```python
from fractions import Fraction

cold_open = {
    "Shapley-Taylor": (Fraction(0), Fraction(1, 6)),
    "Faith-Shap": (Fraction(-1, 5), Fraction(3, 10)),
    "Shapley / pair SII": (Fraction(1, 4), Fraction(1, 3)),
}
for name, (singleton, pair) in cold_open.items():
    print(f"{name:20s} singleton={str(singleton):>4s}  pair={str(pair):>4s}")
print("ST efficient:", 4 * cold_open["Shapley-Taylor"][0] + 6 * cold_open["Shapley-Taylor"][1])
print("Faith efficient:", 4 * cold_open["Faith-Shap"][0] + 6 * cold_open["Faith-Shap"][1])
```

<!-- ARENA:cell markdown -->
<details>
<summary>Question - how can a positive four-way mechanism produce negative singleton coefficients?</summary>

Faith-Shap is fitting a degree-two polynomial to the whole coalition table. Negative singletons
counteract positive pair terms on small coalitions, while the exact full-coalition constraint keeps
the total at one. A coefficient in this surrogate is not an isolated causal effect.

</details>

<!-- ARENA:cell markdown -->
## Content and learning objectives

By the end of the lesson, you should be able to:

1. invert a complete coalition table into bitmask-indexed Möbius dividends;
2. derive Shapley-Taylor's top-order dividend routing rule;
3. construct Faith-Shap's subset design and per-coalition Shapley kernel;
4. solve Faith-Shap with exact endpoint constraints and check the paper's closed form;
5. distinguish kernel-weighted faithfulness from unweighted coalition residuals;
6. turn a scalar model and intact donor bank into a complete empirical game;
7. trace a pair interaction to four explicit model evaluations in a chosen context; and
8. interpret fitted-model disagreement without calling any interaction definition causal truth.

The eight implementation exercises should take about 105–135 minutes. The native capstone is a
guided reading and recomputation exercise after the scientific protocol is frozen.

| route | sections | suggested time |
|---|---|---:|
| Core exact methods | Exercises 1–4 and 6 | 60–75 min |
| Stretch derivation | Exercise 5, the paper closed form | 15–20 min |
| Model-game bridge | Exercises 7–8 | 30–40 min |
| Guided native reading and plays | Section 6 | 35–50 min after the frozen run is available |

If you are short on time, read the Exercise 5 derivation and solution, then return after completing
the model-game bridge. The KKT implementation remains the primary executable Faith-Shap route; the
closed form is its deliberately independent cross-check.

<!-- ARENA:cell markdown -->
## Setup

The public methods operate on CPU `float64` tensors. The final axis is a complete coalition table in
ascending integer-mask order: bit `i` says feature `i` is present. Entry zero in every coefficient
vector is the intercept. It is displayed separately from the nonempty efficiency ledger.

<!-- ARENA:cell code -->
```python
from __future__ import annotations

import contextlib
import hashlib
import math
import os
import random
from collections.abc import Iterator, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch as t
from torch import Tensor, nn
from IPython.display import display

from chapter16_shapley_attribution_baselines.exercises.part16_shapley_taylor_faith_shap import (
    interaction_methods_tests as tests,
)
from chapter16_shapley_attribution_baselines.exercises.part16_shapley_taylor_faith_shap.interaction_methods import (
    _interaction_counts,
    _interaction_float_input,
    _interaction_game_batches,
    _interaction_model,
    _interaction_model_output,
    _interaction_n_players,
    _interaction_order,
    _interaction_preserve_model_state,
    _interaction_tensor,
)
from chapter16_shapley_attribution_baselines.exercises.part16_shapley_taylor_faith_shap.interaction_diagnostics import (
    GAME_NAMES,
    PAIR_INDICES,
    _callable,
    _diagnostic,
    _ordinary_shapley_from_mobius,
    _pairwise_sii_from_mobius,
    _zeta,
    interaction_diagnostics,
)
from chapter16_shapley_attribution_baselines.exercises.part16_shapley_taylor_faith_shap.interaction_plots import (
    plot_allrow_pair_heatmaps,
    plot_background_changes_the_game,
    plot_contextual_pair_interventions,
    plot_dividend_play,
    plot_donor_context_play,
    plot_exact_allocation_ledger,
    plot_fixed_row_interaction_graph,
    plot_heldout_pair_order_capture,
    plot_native_game_and_controls,
    plot_order_two_reconstruction,
    plot_reconstruction_reason,
)
from chapter16_shapley_attribution_baselines.exercises.part16_shapley_taylor_faith_shap.interaction_data import (
    background_order,
    load_manifest,
    load_split,
    tensor_mapping_sha256,
)
from chapter16_shapley_attribution_baselines.exercises.part16_shapley_taylor_faith_shap.interaction_reference import (
    _preserve_execution_state,
    _reference_check_impl,
    _validate_exact_suite,
    validate_reference_payload,
)
from chapter16_shapley_attribution_baselines.exercises.part16_shapley_taylor_faith_shap.run_native import (
    persist_completed,
    persist_prepared,
    validate_freeze,
    write_notebook_summary,
)
from chapter16_shapley_attribution_baselines.exercises.part16_shapley_taylor_faith_shap.interaction_study import (
    _METHOD_FIRST_LEAVES,
    _SHARED_GAME_LEAVES,
    _background_records,
    _clone_state,
    _combine_game_blocks,
    _derived_seed,
    _direct_full_context_four,
    _empirical_reference_plan,
    _evaluate_backgrounds,
    _evaluate_primary,
    _exact_suite,
    _fit_one_mlp,
    _fit_ridge,
    _float_tensor,
    _group_weighted_mean_abs,
    _int_tensor,
    _localization_statistics,
    _logical_hybrid_row,
    _pair_ranks,
    _performance,
    _require_disjoint_ids,
    _require_methods,
    _rows,
    _storage_plan,
    _training_normalization,
    _validate_fitted,
    _within_size_permutations,
    coalition_masks,
    exact_dividend_play,
    nearest_training_diagnostics,
    pair_indices,
)


DEVICE = torch.device("cpu")
DTYPE = torch.float64
torch.set_default_dtype(DTYPE)
torch.set_num_threads(4)
FIGURE_DIR = Path(os.environ.get("ARENA_INTERACTION_OUTPUT_DIRECTORY", "./figures"))
native_directory = FIGURE_DIR / "native"


def show_figure(figure, stem: str):
    """Display one result and retain the exact PNG used by the draft execution."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / f"{stem}.png"
    figure.savefig(path, dpi=160, bbox_inches="tight")
    display(figure)
    plt.close(figure)
    return path
```

<!-- ARENA:cell markdown -->
The private `_interaction_*` names above are validation and state-restoration plumbing. The ten
scientific functions below are implemented in full in the notebook and receive the tests directly.

# 1️⃣ From coalition values to dividends

<!-- ARENA:cell markdown -->
## A complete game and its Möbius expansion

Let $N=\{0,\ldots,n-1\}$ and let $v(S)$ be a value for every subset $S\subseteq N$. The discrete
derivative of a set $U$ at the empty context is

$$
\Delta_U v(\varnothing)
=\sum_{W\subseteq U}(-1)^{|U|-|W|}v(W).
$$

This is the Möbius dividend $a(U)$. The inverse is the subset sum

$$
v(S)=\sum_{U\subseteq S}a(U).
$$

For two players, integer masks make the bookkeeping concrete. Consider this complete game:

| mask | coalition | $v(S)$ | Möbius calculation | $a(S)$ |
|---:|---|---:|---|---:|
| `00` | $\varnothing$ | `1` | $v(\varnothing)$ | `1` |
| `01` | $\{A\}$ | `3` | $3-1$ | `2` |
| `10` | $\{B\}$ | `0` | $0-1$ | `-1` |
| `11` | $\{A,B\}$ | `5` | $5-3-0+1$ | `3` |

The last row is the part of the joint coalition that its two strict singleton subsets did not
already explain. Summing the four dividends beneath mask `11` gives `1 + 2 - 1 + 3 = 5`, recovering
the game value.

For a pure three-way game, every coalition value below the full triple is zero, so exactly one
Möbius coefficient is nonzero. The fast transform below applies one feature bit at a time and works
for arbitrary leading batch dimensions.

<!-- ARENA:cell markdown -->
### Exercise 1 - implement `mobius_transform`

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵🔵⚪
> You should spend up to 10 minutes on this exercise.
> ```

Work on a clone. For feature bit `i`, subtract the coefficient at the matching mask with bit `i`
removed from every mask that contains `i`.

<!-- ARENA:cell code -->
```python
def mobius_transform(game_values: Tensor, *, n_players: int) -> Tensor:
    """Return a full bitmask-indexed Möbius coefficient tensor."""
    raise NotImplementedError()


tests.test_mobius_transform(fn=mobius_transform)
```

<!-- ARENA:cell markdown -->
<details>
<summary>Expected output</summary>

```text
All tests in test_mobius_transform passed!
```

The constant, additive, pure-pair, and pure-triple rows recover only their declared dividends. The
input stays unchanged and seeded RNG states do not move.

</details>

<!-- ARENA:cell markdown -->
<details>
<summary>Help - update masks containing one bit</summary>

Create `masks = torch.arange(2**n_players)`. For each player, select masks where
`masks & (1 << player)` is nonzero. Their lower partner is obtained by XOR-ing that bit.

</details>

<!-- ARENA:cell markdown -->
<details>
<summary>Interpretation</summary>

Why do we start from Möbius dividends rather than directly averaging coalition values?

The dividend isolates the part of $v(U)$ that was not already explained by strict subsets. It makes
"a true four-way term" literal, so we can see exactly where an order-two rule sends it.

</details>

<!-- ARENA:cell markdown -->
<details>
<summary>Solution</summary>

```python
def mobius_transform(game_values: Tensor, *, n_players: int) -> Tensor:
    """Return a full bitmask-indexed Möbius coefficient tensor."""
    n_players = _interaction_n_players(n_players)
    _interaction_tensor("game_values", game_values, n_players)
    mobius = game_values.clone()
    masks = torch.arange(2**n_players, dtype=torch.int64)
    for player in range(n_players):
        present = masks.bitwise_and(1 << player).ne(0)
        lower = masks[present].bitwise_xor(1 << player)
        mobius[..., present] -= mobius[..., lower]
    return mobius
```

</details>

<!-- ARENA:cell markdown -->
The first diagnostic constructs named constant, additive, pair, triple, four-way, negative and mixed
games. A sparse dividend display is more informative here than a pass/fail scalar.

<!-- ARENA:cell code -->
```python
# ARENA:insert interaction_diagnostics.py:exact_game_fixture
```

<!-- ARENA:cell code -->
```python
# ARENA:insert interaction_diagnostics.py:mobius_diagnostic
```

<!-- ARENA:cell code -->
```python
exact_diagnostics = diagnose_mobius_games(mobius_transform, diagnostic=exact_game_fixture())
print("max |Möbius error|:", exact_diagnostics["mobius_max_abs_error"])
for game_name, row in zip(
    exact_diagnostics["game_names"], exact_diagnostics["mobius"], strict=True
):
    sparse = {
        f"{mask:04b}": round(float(value), 6)
        for mask, value in enumerate(row)
        if abs(float(value)) > 1e-12
    }
    print(f"{game_name:16s}", sparse)
```

<!-- ARENA:cell markdown -->
# 2️⃣ Shapley-Taylor: route omitted dividends

<!-- ARENA:cell markdown -->
## Preserve low orders, distribute at the boundary

At order $k$, Shapley-Taylor keeps the lower Möbius dividends:

$$
I^{\mathrm{ST},k}_S=a(S),\qquad 0<|S|<k.
$$

Every dividend of order at least $k$ is shared uniformly over its $k$-subsets:

$$
I^{\mathrm{ST},k}_S
=\sum_{U\supseteq S}\frac{a(U)}{\binom{|U|}{k}},
\qquad |S|=k.
$$

All terms above $k$ are exactly zero. At $k=1$ the output is ordinary Shapley; at $k=n$ it is the
full Möbius expansion.

<!-- ARENA:cell markdown -->
### Exercise 2 - implement `shapley_taylor_interactions`

> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵🔵🔵🔵
> You should spend up to 15 minutes on this exercise.
> ```

Keep the intercept at mask zero. Copy lower nonempty terms, then visit each mask of exactly the
requested order and collect its share from every superset.

<!-- ARENA:cell code -->
```python
def shapley_taylor_interactions(
    mobius: Tensor, *, n_players: int, order: int
) -> Tensor:
    """Allocate Möbius dividends into an efficient order-truncated expansion."""
    raise NotImplementedError()


tests.test_shapley_taylor_interactions(fn=shapley_taylor_interactions)
```

<!-- ARENA:cell markdown -->
<details>
<summary>Expected output</summary>

```text
All tests in test_shapley_taylor_interactions passed!
```

A pure triple gives singleton `0` and each contained pair `1/3`; a pure four-way term gives
singleton `0` and all six pairs `1/6`. Order one matches ordinary Shapley and full order matches
Möbius inversion.

</details>

<!-- ARENA:cell markdown -->
<details>
<summary>Help - sum over supersets of each retained pair</summary>

For one retained mask `interaction`, a mask `superset` contains it when
`superset & interaction == interaction`. Divide the superset dividend by
`math.comb(superset.bit_count(), order)`.

</details>

<!-- ARENA:cell markdown -->
<details>
<summary>Interpretation</summary>

Why is the pair value `1/6` rather than `1/4` for the pure four-way game?

The dividend is shared among the six retained **pairs**, not the four players. The denominator is
$\binom{4}{2}$. The resulting nonempty terms sum to the full-minus-empty value.

</details>

<!-- ARENA:cell markdown -->
<details>
<summary>Solution</summary>

```python
def shapley_taylor_interactions(
    mobius: Tensor, *, n_players: int, order: int
) -> Tensor:
    """Allocate Möbius dividends into an efficient order-truncated expansion."""
    n_players = _interaction_n_players(n_players)
    order = _interaction_order("order", order, n_players)
    _interaction_tensor("mobius", mobius, n_players)
    counts = _interaction_counts(n_players)
    result = torch.zeros_like(mobius)
    result[..., 0] = mobius[..., 0]
    lower = (counts > 0) & (counts < order)
    result[..., lower] = mobius[..., lower]
    for interaction in range(1, 2**n_players):
        if interaction.bit_count() != order:
            continue
        for superset in range(interaction, 2**n_players):
            if superset & interaction == interaction:
                result[..., interaction] += mobius[..., superset] / math.comb(
                    superset.bit_count(), order
                )
    return result
```

</details>

<!-- ARENA:cell code -->
```python
# ARENA:insert interaction_diagnostics.py:taylor_diagnostic
```

<!-- ARENA:cell code -->
```python
exact_diagnostics = diagnose_shapley_taylor(
    exact_diagnostics, shapley_taylor_interactions
)
print(
    "max |efficiency residual|:",
    float(exact_diagnostics["stii_efficiency_residual"].abs().max()),
)
```

<!-- ARENA:cell markdown -->
# 3️⃣ Faith-Shap: fit a constrained polynomial

<!-- ARENA:cell markdown -->
## What does “faithful” optimize?

For maximum order $\ell$, Faith-Shap reconstructs coalition $S$ with

$$
\hat v_E(S)=\sum_{T\subseteq S,\ |T|\leq\ell}E_T.
$$

It minimizes the interior weighted error

$$
\sum_{1\leq |S|\leq n-1}\mu(S)\bigl(v(S)-\hat v_E(S)\bigr)^2,
\qquad
\mu(S)\propto\frac{n-1}{\binom{n}{|S|}|S|(n-|S|)},
$$

subject to the exact constraints $E_\varnothing=v(\varnothing)$ and
$\sum_{\lvert T\rvert\leq\ell}E_T=v(N)$. The formula above is a **per-coalition** weight. Using a whole
cardinality's probability for every row would count $\binom n{\lvert S\rvert}$ twice.

<!-- ARENA:cell markdown -->
### Exercise 3 - implement `faith_design_matrix` and `faith_kernel_weights`

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵🔵🔵
> You should spend up to 15 minutes on this exercise.
> ```

The design contains one column for every term mask of size at most `max_order`, including mask zero.
Set `X[S,T]=1` exactly when `T` is a subset of `S`. Return zero endpoint weights and normalize all
finite interior weights to sum to one.

<!-- ARENA:cell code -->
```python
def faith_design_matrix(
    *, n_players: int, max_order: int
) -> tuple[Tensor, Tensor]:
    """Return float64 X[coalition_mask, term] and int64 term_masks."""
    raise NotImplementedError()


def faith_kernel_weights(*, n_players: int) -> Tensor:
    """Return normalized interior per-coalition weights; endpoints are zero."""
    raise NotImplementedError()


tests.test_faith_design_matrix(fn=faith_design_matrix)
tests.test_faith_kernel_weights(fn=faith_kernel_weights)
```

<!-- ARENA:cell markdown -->
<details>
<summary>Expected output</summary>

```text
All tests in test_faith_design_matrix passed!
All tests in test_faith_kernel_weights passed!
```

For `n=3, max_order=2`, `term_masks == [0,1,2,3,4,5,6]`, the design has shape `(8,7)`, and each
of the six interior coalitions has normalized weight `1/6`. For `n=4`, a size-one coalition gets
twice the weight of one size-two coalition.

</details>

<!-- ARENA:cell markdown -->
<details>
<summary>Help - separate term selection from subset incidence</summary>

First filter all masks by population count to obtain `term_masks`. Then compare
`masks[:, None] & term_masks[None, :]` with `term_masks`. Build kernel weights in a loop over the
interior masks and normalize once at the end.

</details>

<!-- ARENA:cell markdown -->
<details>
<summary>Interpretation</summary>

Why are the returned endpoint weights zero?

Those rows are excluded from the finite objective because their values are imposed as exact equality
constraints. Zero is an implementation marker for exclusion, not the mathematical claim that
endpoints are unimportant.

</details>

<!-- ARENA:cell markdown -->
<details>
<summary>Solution</summary>

```python
def faith_design_matrix(
    *, n_players: int, max_order: int
) -> tuple[Tensor, Tensor]:
    """Return float64 X[coalition_mask, term] and int64 term_masks."""
    n_players = _interaction_n_players(n_players, minimum=2)
    max_order = _interaction_order("max_order", max_order, n_players)
    masks = torch.arange(2**n_players, dtype=torch.int64)
    counts = _interaction_counts(n_players)
    term_masks = masks[counts <= max_order]
    design = masks[:, None].bitwise_and(term_masks[None, :]).eq(term_masks).to(torch.float64)
    return design, term_masks


def faith_kernel_weights(*, n_players: int) -> Tensor:
    """Return normalized interior per-coalition weights; endpoints are zero."""
    n_players = _interaction_n_players(n_players, minimum=2)
    weights = torch.zeros(2**n_players, dtype=torch.float64)
    for mask in range(1, 2**n_players - 1):
        size = mask.bit_count()
        weights[mask] = (n_players - 1) / (
            math.comb(n_players, size) * size * (n_players - size)
        )
    return weights / weights.sum()
```

</details>

<!-- ARENA:cell code -->
```python
# ARENA:insert interaction_diagnostics.py:faith_design_diagnostic
```

<!-- ARENA:cell code -->
```python
exact_diagnostics = diagnose_faith_design(
    exact_diagnostics, faith_design_matrix, faith_kernel_weights
)
print("n=3 design shape:", tuple(exact_diagnostics["faith_design"].shape))
print("n=3 kernel sum:", float(exact_diagnostics["faith_kernel_weights"].sum()))
```

<!-- ARENA:cell markdown -->
## Exact endpoint constraints via one KKT solve

Let $X$ contain the retained design columns, $W$ the diagonal interior kernel, and $C$ two
constraint rows: one selects the intercept and one sums every coefficient. The exact system is

$$
\begin{bmatrix}X^T W X & C^T\\ C & 0\end{bmatrix}
\begin{bmatrix}\beta\\\lambda\end{bmatrix}
=
\begin{bmatrix}X^T Wv\\v(\varnothing)\\v(N)\end{bmatrix}.
$$

A finite large endpoint penalty is an approximation to this system. Here we solve the equalities
directly and reuse the same matrix for every leading batch item.

<details>
<summary>Derivation - from least squares to the two constraint rows</summary>

First ignore the endpoints. Differentiating
$\tfrac12(X\beta-v)^T W(X\beta-v)$ gives $X^T W(X\beta-v)$, so an unconstrained
stationary point obeys $X^T W X\beta=X^T Wv$.

The first row of the design at the empty coalition is $(1,0,\ldots,0)$: it selects
the intercept. At the full coalition, every retained subset is present, so the row
is $(1,1,\ldots,1)$. Stack those two rows into $C$ and their required values into
$d=(v(\varnothing),v(N))^T$.

Add a multiplier term $\lambda^T(C\beta-d)$ to the objective. Setting both derivatives
to zero gives $X^T W X\beta+C^T\lambda=X^T Wv$ and $C\beta=d$, which are exactly
the two block rows above. The multiplier changes the optimum so the endpoint
equalities hold; it is not a large numerical penalty. In floating-point code,
these equalities hold to the precision of the linear solve.

</details>

<!-- ARENA:cell markdown -->
### Exercise 4 - implement `faith_shap_interactions`

> ```yaml
> Difficulty: 🔴🔴🔴🔴⚪
> Importance: 🔵🔵🔵🔵🔵
> You should spend up to 20 minutes on this exercise.
> ```

Build the Gram and constraint blocks once. Flatten only the leading dimensions into a batch of
right-hand sides, solve them together, then scatter the retained terms into a full mask vector.

There is one useful exact shortcut. If `max_order == n_players`, all coalition
monomials are present. Their unique interpolating coefficients are the Möbius
dividends you already know how to compute. Return that transform directly in this
case. It has zero reconstruction error in exact arithmetic and avoids the numerical
conditioning of a much larger KKT system. Use the constrained solve for truncated orders.

<!-- ARENA:cell code -->
```python
def faith_shap_interactions(
    game_values: Tensor, *, n_players: int, max_order: int
) -> Tensor:
    """Solve exact equality-constrained weighted least squares."""
    raise NotImplementedError()


tests.test_faith_shap_interactions(fn=faith_shap_interactions)
```

<!-- ARENA:cell markdown -->
<details>
<summary>Expected output</summary>

```text
All tests in test_faith_shap_interactions passed!
```

For the pure four-way game, singletons are `-1/5`, all pairs are `3/10`, higher terms are exactly
zero, and the complete coefficient vector sums to `v(full)=1`. Order one matches ordinary Shapley;
full order recovers the Möbius expansion.

</details>

<!-- ARENA:cell markdown -->
<details>
<summary>Help - put batches in the right-hand side</summary>

With `weighted_design = weights[1:-1, None] * design[1:-1]`, the Gram matrix is shared by every
game. Flatten games to `[-1, 2**n]`, concatenate the two endpoint targets to each row, and solve the
transposed right-hand side once.

</details>

<!-- ARENA:cell markdown -->
<details>
<summary>Help - check the KKT block shapes first</summary>

For `n=3, max_order=2`, there are seven retained term masks. Therefore `gram` is `7×7`,
`constraints` is `2×7`, and the complete KKT matrix is `9×9`. For a batch of `B` games, assemble a
`B×9` right-hand side and solve its transpose. The first seven solution rows are coefficients; the
last two are Lagrange multipliers and are not scattered into the game.

</details>

<!-- ARENA:cell markdown -->
<details>
<summary>Interpretation</summary>

What does exact efficiency establish here?

It establishes that the retained intercept, singleton, and pair coefficients reproduce the full
coalition value. It does not establish exact reconstruction of every interior coalition, agreement
with Shapley-Taylor, or causal truth.

</details>

<!-- ARENA:cell markdown -->
<details>
<summary>Solution</summary>

```python
def faith_shap_interactions(
    game_values: Tensor, *, n_players: int, max_order: int
) -> Tensor:
    """Solve exact equality-constrained weighted least squares."""
    n_players = _interaction_n_players(n_players, minimum=2)
    max_order = _interaction_order("max_order", max_order, n_players)
    _interaction_tensor("game_values", game_values, n_players)
    if max_order == n_players:
        # All coalition monomials are present: the unique zero-error interpolant
        # is the Möbius expansion. Avoid an unnecessary ill-conditioned KKT solve.
        return mobius_transform(game_values, n_players=n_players)
    design, term_masks = faith_design_matrix(
        n_players=n_players, max_order=max_order
    )
    weights = faith_kernel_weights(n_players=n_players)
    interior = design[1:-1]
    weighted_design = weights[1:-1, None] * interior
    gram = interior.T @ weighted_design
    constraints = torch.stack((design[0], design[-1]))
    zeros = torch.zeros((2, 2), dtype=torch.float64)
    kkt = torch.cat(
        (
            torch.cat((gram, constraints.T), dim=1),
            torch.cat((constraints, zeros), dim=1),
        ),
        dim=0,
    )

    values = game_values.reshape(-1, 2**n_players)
    weighted_targets = values[:, 1:-1] @ weighted_design
    endpoint_targets = torch.stack((values[:, 0], values[:, -1]), dim=1)
    right_hand_side = torch.cat((weighted_targets, endpoint_targets), dim=1)
    solution = torch.linalg.solve(kkt, right_hand_side.T).T
    result = torch.zeros_like(values)
    result[:, term_masks] = solution[:, : len(term_masks)]
    return result.reshape(game_values.shape)
```

</details>

<!-- ARENA:cell markdown -->
# 4️⃣ Check the paper formula and measure faithfulness

<!-- ARENA:cell markdown -->
## The independent Möbius closed form

For a nonempty retained set $S$, with $s=|S|\leq\ell$, Faith-Shap can also be written directly in
terms of Möbius dividends:

$$
E_S=a(S)+(-1)^{\ell-s}\frac{s}{\ell+s}\binom{\ell}{s}
\sum_{U\supset S,\ |U|>\ell}
\frac{\binom{|U|-1}{\ell}}{\binom{|U|+\ell-1}{\ell+s}}a(U).
$$

This is scientifically useful because it reaches the same index by a different route. Comparing the
closed form with the constrained solve can expose design, weighting, constraint, and scatter bugs.

<!-- ARENA:cell markdown -->
### Exercise 5 - implement `faith_shap_closed_form`

> ```yaml
> Difficulty: 🔴🔴🔴🔴⚪
> Importance: 🔵🔵🔵🔵⚪
> You should spend up to 20 minutes on this exercise.
> ```

Visit each retained nonempty interaction. Accumulate the correction from strict supersets whose
order is above `max_order`, multiply by the signed combinatorial scale, and add its own dividend.

<!-- ARENA:cell code -->
```python
def faith_shap_closed_form(
    mobius: Tensor, *, n_players: int, max_order: int
) -> Tensor:
    """Compute the independent primary-paper Möbius formula."""
    raise NotImplementedError()


tests.test_faith_shap_closed_form(fn=faith_shap_closed_form)
```

<!-- ARENA:cell markdown -->
<details>
<summary>Expected output</summary>

```text
All tests in test_faith_shap_closed_form passed!
```

The closed form agrees with the exact constrained solver on seeded batched games within the declared
float64 tolerance. The pure-four singleton/pair entries are exactly `[-1/5, 3/10]` at the checked
coordinates.

</details>

<!-- ARENA:cell markdown -->
<details>
<summary>Help - separate containment, size, and scale</summary>

For fixed `interaction`, include `superset` only if it contains every interaction bit and has size
strictly above `max_order`. Compute the binomial ratio first; apply the signed scale after the sum.

</details>

<!-- ARENA:cell markdown -->
<details>
<summary>Help - work the pure-four coefficient by hand</summary>

For the pure four-way dividend with retained order `2`, a singleton has `s=1`. Its signed outer
factor is $(-1)^1(1/3)\binom{2}{1}=-2/3$, while the only superset contributes
$\binom{3}{2}/\binom{5}{3}=3/10$. Their product is `-1/5`. For a pair, the outer factor is `1/2`
and the ratio is $\binom{3}{2}/\binom{5}{4}=3/5$, giving `3/10`. Reproduce those two numbers before
writing the nested loops.

</details>

<!-- ARENA:cell markdown -->
<details>
<summary>Interpretation</summary>

Why is same-index agreement between the KKT solve and this formula stronger than checking
endpoint efficiency alone?

Many coefficient vectors satisfy the two endpoint constraints. The independent formula checks the
whole selected Faith-Shap solution, including every singleton and pair adjustment.

</details>

<!-- ARENA:cell markdown -->
<details>
<summary>Solution</summary>

```python
def faith_shap_closed_form(
    mobius: Tensor, *, n_players: int, max_order: int
) -> Tensor:
    """Compute the independent primary-paper Möbius formula."""
    n_players = _interaction_n_players(n_players, minimum=2)
    max_order = _interaction_order("max_order", max_order, n_players)
    _interaction_tensor("mobius", mobius, n_players)
    result = torch.zeros_like(mobius)
    result[..., 0] = mobius[..., 0]
    for interaction in range(1, 2**n_players):
        size = interaction.bit_count()
        if size > max_order:
            continue
        correction = torch.zeros_like(mobius[..., 0])
        for superset in range(2**n_players):
            superset_size = superset.bit_count()
            if superset_size > max_order and superset & interaction == interaction:
                correction += (
                    math.comb(superset_size - 1, max_order)
                    / math.comb(superset_size + max_order - 1, max_order + size)
                    * mobius[..., superset]
                )
        scale = (
            (-1) ** (max_order - size)
            * size
            / (max_order + size)
            * math.comb(max_order, size)
        )
        result[..., interaction] = mobius[..., interaction] + scale * correction
    return result
```

</details>

<!-- ARENA:cell code -->
```python
# ARENA:insert interaction_diagnostics.py:faith_solver_diagnostic
```

<!-- ARENA:cell code -->
```python
exact_diagnostics = diagnose_faith_solver(
    exact_diagnostics, faith_shap_interactions, faith_shap_closed_form
)
print(
    "max |KKT - closed form|:",
    exact_diagnostics["fsii_closed_form_max_abs_error"],
)
```

<!-- ARENA:cell markdown -->
## Reconstruction is the claim you can inspect

The same subset-zeta transform that inverts Möbius coefficients reconstructs a coalition table from
any interaction coefficients:

$$
\hat v(S)=\sum_{T\subseteq S}I_T.
$$

Faith-Shap's word “faithful” refers to its optimum under the declared kernel, degree, and exact
constraints. It does not promise the smallest unweighted error, the best predictor, or the most
causal-looking pair. We will therefore show both the weighted objective and residuals by coalition
size.

<!-- ARENA:cell markdown -->
### Exercise 6 - implement `reconstruct_game` and `faithfulness_error`

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵🔵🔵
> You should spend up to 15 minutes on this exercise.
> ```

The reconstruction is the forward subset-zeta transform: add the lower partner into masks containing
one feature bit. Then compute the normalized kernel-weighted squared residual on the final axis.

<!-- ARENA:cell code -->
```python
def reconstruct_game(coefficients: Tensor, *, n_players: int) -> Tensor:
    """Apply the subset zeta transform to interaction coefficients."""
    raise NotImplementedError()


def faithfulness_error(
    game_values: Tensor, coefficients: Tensor, *, n_players: int
) -> Tensor:
    """Return normalized Shapley-kernel interior MSE for each leading item."""
    raise NotImplementedError()


tests.test_reconstruct_game(fn=reconstruct_game)
tests.test_faithfulness_error(fn=faithfulness_error)
```

<!-- ARENA:cell markdown -->
<details>
<summary>Expected output</summary>

```text
All tests in test_reconstruct_game passed!
All tests in test_faithfulness_error passed!
```

Möbius and zeta transforms invert one another for arbitrary leading batches. On the pure four-way
game at order two, Shapley-Taylor has normalized kernel error `13/132 ≈ 0.09848`, while Faith-Shap
has `1/20 = 0.05`. Both are exact at empty and full coalitions; the degree-two control has zero error.

</details>

<!-- ARENA:cell markdown -->
<details>
<summary>Help - the endpoints already have zero weight</summary>

After reconstructing, square `game_values - reconstruction`, multiply by
`faith_kernel_weights(n_players=n_players)`, and sum over the mask axis. Do not renormalize per
batch or add a separate endpoint penalty.

</details>

<!-- ARENA:cell markdown -->
<details>
<summary>Interpretation</summary>

Why can we say Faith-Shap is better on this metric without saying it recovered a truer mechanism?

The inequality follows from the optimization problem: Faith-Shap minimizes this exact weighted
objective over the constrained degree-two family. Mechanism recovery, unweighted residuals, pair
rankings, and causal effects are different questions.

</details>

<!-- ARENA:cell markdown -->
<details>
<summary>Solution</summary>

```python
def reconstruct_game(coefficients: Tensor, *, n_players: int) -> Tensor:
    """Apply the subset zeta transform to interaction coefficients."""
    n_players = _interaction_n_players(n_players)
    _interaction_tensor("coefficients", coefficients, n_players)
    values = coefficients.clone()
    masks = torch.arange(2**n_players, dtype=torch.int64)
    for player in range(n_players):
        present = masks.bitwise_and(1 << player).ne(0)
        lower = masks[present].bitwise_xor(1 << player)
        values[..., present] += values[..., lower]
    return values


def faithfulness_error(
    game_values: Tensor, coefficients: Tensor, *, n_players: int
) -> Tensor:
    """Return normalized Shapley-kernel interior MSE for each leading item."""
    n_players = _interaction_n_players(n_players, minimum=2)
    _interaction_tensor("game_values", game_values, n_players)
    _interaction_tensor("coefficients", coefficients, n_players)
    if game_values.shape != coefficients.shape:
        raise ValueError("game_values and coefficients must have the same shape.")
    residual = game_values - reconstruct_game(coefficients, n_players=n_players)
    weights = faith_kernel_weights(n_players=n_players)
    return (weights * residual.square()).sum(dim=-1)
```

</details>

<!-- ARENA:cell code -->
```python
# ARENA:insert interaction_diagnostics.py:reconstruction_diagnostic
```

<!-- ARENA:cell code -->
```python
exact_diagnostics = diagnose_reconstruction(
    exact_diagnostics, reconstruct_game, faithfulness_error
)
figure = plot_exact_allocation_ledger(exact_diagnostics)
show_figure(figure, "01_exact_allocation_ledger")
```

<!-- ARENA:cell markdown -->
The first figure keeps SII physically separate from the two efficient decompositions. Its singleton
and pair strips are valid indices, but adding them would manufacture a total that SII never claims.

<!-- ARENA:cell code -->
```python
figure = plot_reconstruction_reason(exact_diagnostics)
show_figure(figure, "02_why_faith_moves_singletons")
```

<!-- ARENA:cell markdown -->
For the pure four-way table, Shapley-Taylor reconstructs coalition-size means
`[0, 0, 1/6, 1/2, 1]`; Faith-Shap reconstructs `[0, -1/5, -1/10, 3/10, 1]`.
The negative singleton is doing visible approximation work on the interior coalitions.

<!-- ARENA:cell markdown -->
# 5️⃣ From a model to a complete finite game

<!-- ARENA:cell markdown -->
## Intact donors define the missing-feature operation

For a scalar model $f$, query $x$, donor row $z_b$, and coalition $S$, define the hybrid

$$
h_i(x,z_b,S)=\begin{cases}x_i&i\in S\\z_{b,i}&i\notin S,\end{cases}
$$

and the finite-background game

$$
v_{x,B}(S)=\frac{1}{|B|}\sum_{b\in B}f(h(x,z_b,S)).
$$

All absent features come from the **same intact donor row**. Sampling each missing coordinate from a
different marginal would define a different game. The empty value is the donor-bank mean prediction;
the full value is the query prediction.

For example, with query `(10, 20, 30)`, donor `(1, 2, 3)`, and mask `101`, the hybrid is
`(10, 2, 30)`: the middle coordinate comes from that same donor. A second donor produces a second
complete hybrid, and only then do we average their model outputs.

<!-- ARENA:cell markdown -->
### Exercise 7 - implement `masked_model_game`

> ```yaml
> Difficulty: 🔴🔴🔴🔴⚪
> Importance: 🔵🔵🔵🔵🔵
> You should spend up to 20 minutes on this exercise.
> ```

Your scientific task is the intact hybrid and donor average. The supplied
`_interaction_game_batches` context validates inputs, preserves model state, and
enumerates the row × coalition × donor grid in bounded chunks. It yields the
number of rows, coalitions, and donors, followed by an iterator of batches.
Each batch gives you `query_rows`, `donor_rows`, `present`, and `slots`.

Use the first three arrays to build the hybrids with `torch.where`. Evaluate the
model through `_interaction_model_output`, accumulate its predictions into the
row/coalition `slots`, and divide each accumulated sum by the donor count.
You implement the missing-feature operation and its expectation; the supplied
context handles chunk indices and restores model/RNG state on exit.

<!-- ARENA:cell code -->
```python
def masked_model_game(
    model: nn.Module,
    x: Tensor,
    reference: Tensor,
    *,
    coalition_batch_size: int = 4096,
) -> Tensor:
    """Return games using one reference [D] or an equal-weight bank [B,D]."""
    with _interaction_game_batches(
        model, x, reference, coalition_batch_size=coalition_batch_size
    ) as (rows, coalitions, donors, batches):
        sums = torch.zeros(rows * coalitions, dtype=torch.float64)
        # Each batch has query rows, intact donor rows, presence flags, and slots.
        # Build the hybrid, evaluate it, and accumulate into sums.
        raise NotImplementedError()


tests.test_masked_model_game(fn=masked_model_game)
```

<!-- ARENA:cell markdown -->
<details>
<summary>Expected output</summary>

```text
All tests in test_masked_model_game passed!
```

The two-row, three-feature linear fixture returns shape `(2,8)`. Mask zero equals the mean donor
prediction, mask seven equals each query prediction, and all Möbius terms above order one are zero to
`1e-15`. RNG state, default dtype, module modes, buffers, and parameter gradients are restored on
successful and failing forwards.

</details>

<!-- ARENA:cell markdown -->
<details>
<summary>Help - construct one intact hybrid</summary>

The three input arrays in a batch have the same `[batch, features]` shape. Use
`torch.where(present, query_rows, donor_rows)`. A false entry selects from the
matching donor row; it never selects a different donor independently per feature.
Check the mask `101` example above before thinking about the accumulation.

</details>

<!-- ARENA:cell markdown -->
<details>
<summary>Help - turn donor predictions into a complete game</summary>

Several predictions belong to each row/coalition slot: one per donor. Accumulate
with `sums.index_add_(0, slots, output)` so a chunk boundary cannot discard part
of a donor average. After consuming all batches, divide by `donors` and reshape
to `(rows, coalitions)`. Dividing by the chunk size would be wrong: the final
chunk can be shorter, and a slot can span two chunks.

</details>

<!-- ARENA:cell markdown -->
<details>
<summary>Interpretation</summary>

What changes when you switch from one donor to a 64-row donor bank?

The explanation game changes from one reference recipe to an equal-weight empirical population. The
new interaction is not a confidence replicate of the old one; it answers a different finite-game
question.

</details>

<!-- ARENA:cell markdown -->
<details>
<summary>Solution</summary>

```python
def masked_model_game(
    model: nn.Module,
    x: Tensor,
    reference: Tensor,
    *,
    coalition_batch_size: int = 4096,
) -> Tensor:
    """Return games using one reference [D] or an equal-weight bank [B,D]."""
    with _interaction_game_batches(
        model, x, reference, coalition_batch_size=coalition_batch_size
    ) as (rows, coalitions, donors, batches):
        sums = torch.zeros(rows * coalitions, dtype=torch.float64)
        for query_rows, donor_rows, present, slots in batches:
            inputs = torch.where(present, query_rows, donor_rows)
            output = _interaction_model_output(model, inputs)
            sums.index_add_(0, slots, output)
        return sums.div(donors).reshape(rows, coalitions).detach()
```

</details>

<!-- ARENA:cell code -->
```python
# ARENA:insert interaction_diagnostics.py:model_game_diagnostic
```

<!-- ARENA:cell code -->
```python
model_game_diagnostic = diagnose_model_game(masked_model_game)
print("max |pair Möbius term|:", model_game_diagnostic["pair_max_abs"])
```

<!-- ARENA:cell markdown -->
## A pair derivative opens four coalition values

For a pair $\{i,j\}$ disjoint from context $T$,

$$
\Delta_{ij}v(T)=v(T\cup\{i,j\})-v(T\cup\{i\})-v(T\cup\{j\})+v(T).
$$

This quantity may change sign across contexts. Shapley-Taylor, Faith-Shap, and SII aggregate
higher-order structure differently; none is required to equal the derivative at the empty or full
context.

<!-- ARENA:cell markdown -->
### Exercise 8 - implement `contextual_pair_synergy`

> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵🔵🔵🔵
> You should spend up to 15 minutes on this exercise.
> ```

Validate the pair, generate every disjoint context when none is supplied, or validate and sort the
provided contexts. Index the four masks without a Python loop over leading batch items.

<!-- ARENA:cell code -->
```python
def contextual_pair_synergy(
    game_values: Tensor,
    pair: tuple[int, int],
    *,
    n_players: int,
    context_masks: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Return sorted valid context masks and Delta_pair v(context)."""
    raise NotImplementedError()


tests.test_contextual_pair_synergy(fn=contextual_pair_synergy)
```

<!-- ARENA:cell markdown -->
<details>
<summary>Expected output</summary>

```text
All tests in test_contextual_pair_synergy passed!
```

For a three-player game with pair dividend `2` and triple dividend `3`, pair `(0,1)` has contexts
`[0,4]` and signed derivatives `[2,5]`. A leading batch scaled by `-2` returns `[-4,-10]`.

</details>

<!-- ARENA:cell markdown -->
<details>
<summary>Help - build the pair mask once</summary>

Set `pair_mask = (1 << first) | (1 << second)`. Valid contexts satisfy
`context & pair_mask == 0`. The four lookup axes are context, context plus the first bit, context
plus the second bit, and context plus both bits.

</details>

<!-- ARENA:cell markdown -->
<details>
<summary>Interpretation</summary>

Why does an independently re-run four-forward check matter if the complete game table already
contains these values?

It catches a wrong mask convention or second-difference sign in the stored table lookup. Agreement
is an implementation control. It does not turn the replacement into a physical intervention.

</details>

<!-- ARENA:cell markdown -->
<details>
<summary>Solution</summary>

```python
def contextual_pair_synergy(
    game_values: Tensor,
    pair: tuple[int, int],
    *,
    n_players: int,
    context_masks: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Return sorted valid context masks and Delta_pair v(context)."""
    n_players = _interaction_n_players(n_players, minimum=2)
    _interaction_tensor("game_values", game_values, n_players)
    if (
        type(pair) is not tuple
        or len(pair) != 2
        or any(type(player) is not int for player in pair)
    ):
        raise TypeError("pair must contain exactly two integer player indices.")
    first, second = pair
    if first == second:
        raise ValueError("pair must contain two different players.")
    if not 0 <= first < n_players or not 0 <= second < n_players:
        raise ValueError("pair players must lie in [0, n_players).")
    pair_mask = (1 << first) | (1 << second)

    if context_masks is None:
        contexts = torch.tensor(
            [
                mask
                for mask in range(2**n_players)
                if mask & pair_mask == 0
            ],
            dtype=torch.int64,
        )
    else:
        if type(context_masks) is not Tensor:
            raise TypeError("context_masks must be a torch.Tensor.")
        if context_masks.device.type != "cpu" or context_masks.dtype != torch.int64:
            raise TypeError("context_masks must be a CPU int64 tensor.")
        if context_masks.ndim != 1 or context_masks.numel() == 0:
            raise ValueError("context_masks must be a nonempty one-dimensional tensor.")
        if context_masks.min() < 0 or context_masks.max() >= 2**n_players:
            raise ValueError("Every context mask must index the complete game.")
        if context_masks.bitwise_and(pair_mask).ne(0).any():
            raise ValueError("Context masks must be disjoint from the requested pair.")
        contexts = context_masks.sort().values

    with_first = contexts.bitwise_or(1 << first)
    with_second = contexts.bitwise_or(1 << second)
    with_pair = contexts.bitwise_or(pair_mask)
    synergy = (
        game_values[..., with_pair]
        - game_values[..., with_first]
        - game_values[..., with_second]
        + game_values[..., contexts]
    )
    return contexts, synergy
```

</details>

<!-- ARENA:cell code -->
```python
# ARENA:insert interaction_diagnostics.py:pair_context_diagnostic
```

<!-- ARENA:cell code -->
```python
pair_context_diagnostic = diagnose_pair_context(contextual_pair_synergy)
print("contexts:", pair_context_diagnostic["contexts"].tolist())
print("pair derivatives:", pair_context_diagnostic["synergy"].tolist())
```

<!-- ARENA:cell code -->
```python
methods = SimpleNamespace(
    mobius_transform=mobius_transform,
    shapley_taylor_interactions=shapley_taylor_interactions,
    faith_design_matrix=faith_design_matrix,
    faith_kernel_weights=faith_kernel_weights,
    faith_shap_interactions=faith_shap_interactions,
    faith_shap_closed_form=faith_shap_closed_form,
    reconstruct_game=reconstruct_game,
    faithfulness_error=faithfulness_error,
    masked_model_game=masked_model_game,
    contextual_pair_synergy=contextual_pair_synergy,
)
```

<!-- ARENA:cell markdown -->
## Try It Yourself: reroute a dividend

This exact play is available before any native fit. Change `PLAY_MEMBERS`, `PLAY_VALUE`, or
`PLAY_ORDER`, then predict the allocation before running it. Try a pure triple at order two, and then
set the retained order equal to the dividend degree.

<!-- ARENA:cell code -->
```python
PLAY_N = 4
PLAY_MEMBERS = (0, 1, 2, 3)
PLAY_VALUE = 1.0
PLAY_ORDER = 2

dividend_play = exact_dividend_play(
    methods=methods,
    n_players=PLAY_N,
    members=PLAY_MEMBERS,
    value=PLAY_VALUE,
    order=PLAY_ORDER,
)
explained_play = dividend_play["explained"]
retained_masks = [
    mask for mask in range(1, 2**PLAY_N) if mask.bit_count() <= PLAY_ORDER
]
for label, values in (
    ("Shapley-Taylor", explained_play["stii_coefficients"]),
    ("Faith-Shap", explained_play["fsii_coefficients"]),
):
    nonzero = {
        f"{mask:0{PLAY_N}b}": round(float(values[mask]), 6)
        for mask in retained_masks
        if abs(float(values[mask])) > 1e-12
    }
    print(f"{label:15s}", nonzero)
```

<!-- ARENA:cell markdown -->
<details>
<summary>Ideas to try</summary>

1. Use `PLAY_MEMBERS=(0,1,2)`: predict Shapley-Taylor `0, 1/3` and Faith-Shap
   `-1/6, 1/2` for contained singletons and pairs.
2. Use a negative value. Every method should change sign without changing its allocation fractions.
3. Use `PLAY_ORDER=len(PLAY_MEMBERS)`. Both efficient methods should recover the literal Möbius
   dividend rather than route it downward.

</details>

<!-- ARENA:cell markdown -->
## Plot the dividend you chose

The earlier exact play remains in memory. This figure places its Shapley-Taylor, Faith-Shap, and SII
allocations beside the same-game reconstruction errors.

<!-- ARENA:cell code -->
```python
figure = plot_dividend_play(dividend_play)
show_figure(figure, "10_dividend_play")
```

<!-- ARENA:cell markdown -->
# 6️⃣ Guided native capstone: the same game, different allocation rules

<!-- ARENA:cell markdown -->
## The finite Concrete replacement game

The native study reuses the known UCI Concrete split and predictor recipe from 16.14/16.15 so that
the interaction definition is the variable:

- eight input features and raw compressive-strength output in MPa;
- 758 training rows, 153 explanation rows, and 119 audit rows in disjoint recipe groups;
- three clean ReLU MLP seeds, three matched shuffled-target controls, and ridge;
- all seven models, all 272 held-out rows, all 256 masks, and the ordered primary bank of 64 intact
  training donors; and
- order-two Shapley-Taylor, Faith-Shap, and pair SII on the same complete games.

Five background designs are retained for `clean_0`; each defines a different finite game. Ridge is
the additive control. Shuffled-target models are predictive controls, not zero-interaction oracles.

The path is: **fit the seven models → build their complete games → compare the allocations**.
You already wrote the scientific operations. The supplied study runners below apply them consistently;
their code is expandable in JupyterLab and on the instruction page. Open and edit it whenever you want
to trace a calculation. The short calling cells remain the main path.

This is a self-contained reproduction of a known seeded model/data procedure, not an independent
model or dataset replication.

<!-- ARENA:cell markdown -->
<details>
<summary>Question - why retain donor-level predictions after averaging the game?</summary>

The average alone cannot prove that every absent feature came from one intact row. Donor-level
predictions preserve that construction and let the editable microscope show signed cancellation
across real donor recipes.

</details>

<!-- ARENA:cell markdown -->
## Independent exact-game reference adapter

The primary-paper and `shapiq` comparisons are supporting evidence rather than another student
method. Their small-game checks run during preparation; the exhaustive 2,992-game empirical check
runs once after evaluation. The wrapper is visible so the learner can see exactly which arrays and
scope enter the benchmark.

<!-- ARENA:cell code -->
```python
# ARENA:insert interaction_reference.py:reference_check
```

<!-- ARENA:cell code -->
```python
reference_methods = SimpleNamespace(reference_check=reference_check)
```

<!-- ARENA:cell markdown -->
## The preparation boundary

We use explanation rows to choose the pair order, then measure it once on separate audit rows.
This keeps the held-out comparison useful even when the explanation rankings look surprising.

<details>
<summary>How leakage is prevented, and what is retained</summary>

Training, exact fixtures, primary explanation games, method outputs, and complete explanation-side
pair orders are fixed and persisted before audit access. The local display defaults are content-blind:
`clean_0`, explanation source ID `0`, pair `(0,1)`, primary donor ID `173`, and full context mask
`252`. No audit value chooses a model, row, pair, donor, background, tolerance, or plot.

The primary donor predictions have logical shape `[7,272,256,64]`; averaging the last axis produces
1,904 complete games. Explanation and audit donor predictions are stored in separate physical files,
and the explanation files are sealed before loading the audit split. Four additional background
games per row retain their own donor predictions. Full method outputs, reference comparisons, and
controls stay available in the evidence, including signs, zero denominators, and unfavorable results.

The execution cell checks the frozen protocol and recorded review disposition before loading data or
fitting a model. A development run with independent review still pending cannot be published as a
completed section. [The protocol](protocol.md) records the full counts, seeds, and acceptance rules.

</details>

<!-- ARENA:cell code -->
```python
# ARENA:insert interaction_study.py:protocol_constants
```

<!-- ARENA:cell markdown -->
The next visible source group contains the known-model training recipe, raw-unit folding, and exact
21-state reproduction gate. A mismatch stops before any interaction game is constructed.

<!-- ARENA:cell code -->
```python
# ARENA:insert interaction_study.py:model_fit_and_hash_gate
```

<!-- ARENA:cell code -->
```python
freeze = validate_freeze()
manifest = load_manifest()
train = load_split("train")
explain = load_split("explain")
```

<!-- ARENA:cell markdown -->
Before fitting, look at the actual replacement operation. The table uses the fixed explanation recipe
and donor. Mask `11111100` keeps the other six features from the query while replacing cement and
slag from the donor. These are the context values for the pair examined later.

<!-- ARENA:cell code -->
```python
query_row = explain["x"][np.flatnonzero(explain["ids"] == 0).item()]
donor_row = train["x"][np.flatnonzero(train["ids"] == 173).item()]
preview_present = (252 & (1 << np.arange(8))) != 0
display(pd.DataFrame({
    "feature": FEATURE_NAMES,
    "unit": FEATURE_UNITS,
    "query (source 0)": query_row,
    "donor (source 173)": donor_row,
    "hybrid 11111100": np.where(preview_present, query_row, donor_row),
    "taken from": np.where(preview_present, "query", "donor"),
}).set_index("feature"))
```

<!-- ARENA:cell code -->
```python
fitted = fit_known_models(train)
reproduction = verify_known_reproduction(fitted, manifest)
fitted = {**fitted, "reproduction": reproduction}
print(f"All {reproduction['state_comparisons']} known model-state comparisons passed.")
```

<!-- ARENA:cell markdown -->
The hybrid builder below retains each donor prediction before its equal-weight mean. Its batch size
is an engineering control only; it never changes the masks, rows, donors, models, or values.

<!-- ARENA:cell code -->
```python
# ARENA:insert interaction_study.py:donor_game_construction
```

<!-- ARENA:cell markdown -->
The exact-game block applies your ten functions to the fixed pure and mixed controls. Pair SII is a
separate transparent comparator. It is never concatenated with singleton Shapley values to create a
fake efficient decomposition.

<!-- ARENA:cell code -->
```python
# ARENA:insert interaction_study.py:exact_fixtures_and_methods
```

<!-- ARENA:cell markdown -->
The preparation source now computes the primary explanation games, all exact method outputs,
recipe-group-weighted pair strengths, and complete 28-pair orders. It seals the physical explanation
donor shard before the audit split is loaded.

<!-- ARENA:cell code -->
```python
# ARENA:insert interaction_study.py:explanation_preparation
```

<!-- ARENA:cell code -->
```python
prepared = prepare_study(
    fitted,
    train,
    explain,
    methods=methods,
    reference=reference_methods,
    manifest=manifest,
)
persist_prepared(native_directory, fitted, prepared)
assert prepared["phase"] == "prepared_before_audit"
assert prepared["audit_used_for_selection"] is False
```

<!-- ARENA:cell markdown -->
Only after that seal does the learner execute the literal audit load.

<!-- ARENA:cell code -->
```python
audit = load_split("audit")
```

<!-- ARENA:cell markdown -->
The evaluation source computes every prespecified audit game and method output once. It retains all
28 pairs, all seeds, all rows, all backgrounds, 256 random pair orders, the frozen within-size
shuffles, support distances, and direct four-forward checks. Odd results and zero denominators stay
in the payload.

<!-- ARENA:cell code -->
```python
# ARENA:insert interaction_study.py:audit_evaluation
```

<!-- ARENA:cell code -->
```python
evaluated = evaluate_study(
    fitted,
    prepared,
    audit,
    methods=methods,
    manifest=manifest,
)
assert evaluated["phase"] == "evaluated_after_audit"
assert evaluated["audit_used_for_selection"] is False
```

<!-- ARENA:cell markdown -->
## Independent same-index references

The paper-formula reference recomputes Shapley-Taylor from discrete derivatives, Faith-Shap from the
closed form, and SII pairs independently for all 2,992 unique empirical games. Installed
`shapiq==1.5.2` is supporting exact-index evidence. Its finite endpoint penalty does not replace the
student's exact constrained solve, and this is not a SHAP-IQ estimator experiment.

<!-- ARENA:cell code -->
```python
primary_reference = evaluated["all_rows"]["primary64"]
variant_references = evaluated["all_rows"]["background_variants"]
reference_blocks = [primary_reference] + [
    variant_references[name]
    for name in ("nested16", "nested128", "independent64_a", "independent64_b")
]


def flatten_reference_leaf(leaf: str) -> Tensor:
    return torch.cat(
        [block[leaf].reshape(-1, block[leaf].shape[-1]) for block in reference_blocks]
    )


empirical_reference = reference_check(
    flatten_reference_leaf("coalition_values"),
    flatten_reference_leaf("stii_coefficients"),
    flatten_reference_leaf("fsii_coefficients"),
    flatten_reference_leaf("sii_pairs"),
    n_players=8,
    order=2,
    expected_game_count=2992,
)
# Retain the same flattened branch identities as the standalone reference runner.
reference_branches = []
branch_start = 0
for name, block in zip(
    ("primary64", "nested16", "nested128", "independent64_a", "independent64_b"),
    reference_blocks,
    strict=True,
):
    games = block["coalition_values"].numel() // 256
    reference_branches.append({
        "name": name,
        "model_keys": MODEL_KEYS if name == "primary64" else ("clean_0",),
        "start": branch_start,
        "stop": branch_start + games,
        "games": games,
    })
    branch_start += games
reference_result = finalize_study_reference(prepared, empirical_reference)
reference_result["branches"] = tuple(reference_branches)
print(reference_result["status"])
```

<!-- ARENA:cell markdown -->
The fixed native payload uses the content-blind explanation-row defaults declared before audit
access. The persistence boundary retains the complete scientific arrays and reference result before
any editable post-study reuse.

<!-- ARENA:cell code -->
```python
# ARENA:insert interaction_study.py:empirical_play
```

<!-- ARENA:cell code -->
```python
# ARENA:insert interaction_study.py:native_play
```

<!-- ARENA:cell code -->
```python
diagnostics = interaction_diagnostics(prepared, evaluated, reference_result)
play = native_play(evaluated, methods=methods)
publication_summary = persist_completed(
    native_directory,
    fitted,
    prepared,
    evaluated,
    reference_result,
    diagnostics,
    play,
    timings={},
)
write_notebook_summary(FIGURE_DIR, native_directory, publication_summary)
print("Complete games, controls and reference checks retained. Read the figures below.")
```

<!-- ARENA:cell markdown -->
## Read the native result in layers

The figures below are deliberately separate. Predictive quality establishes the model controls;
signed matrices show every term; capture curves assess a frozen ordering against a declared input
intervention; reconstruction plots show the exact objective each method optimizes. None selects a
winner across axioms.

<!-- ARENA:cell code -->
```python
figure = plot_native_game_and_controls(evaluated)
show_figure(figure, "03_native_game_and_controls")
```

<!-- ARENA:cell markdown -->
The clean predictors have much lower audit RMSE than their shuffled-label counterparts. The additive
ridge control has pair interactions near numerical zero, and independently rebuilt four-forward
synergies agree with table indexing. These checks establish that the model and game construction
behave as intended; they do not decide which interaction definition to prefer.

<details>
<summary>Interpretation - what does the shuffled-label control rule out?</summary>

A large pair coefficient alone is not evidence that a model learned useful structure. The shuffled
models can have large interactions while predicting strength poorly. Read predictive quality first,
then ask how each rule describes that particular fitted model.

</details>

<!-- ARENA:cell code -->
```python
figure = plot_fixed_row_interaction_graph(play, prepared["feature_names"])
show_figure(figure, "04_fixed_row_interaction_graph")
```

<!-- ARENA:cell markdown -->
Node color is singleton credit; edge color and width are signed and absolute pair credit. The two
methods share node positions and scales. A complete term-difference strip prevents graph thresholds
from silently hiding small or zero terms.

<!-- ARENA:cell code -->
```python
figure = plot_allrow_pair_heatmaps(evaluated, prepared["feature_names"])
show_figure(figure, "05_allrow_pair_heatmaps")
```

<!-- ARENA:cell markdown -->
These matrices summarize all 119 audit rows by recipe group and preserve each model seed. Signed mean
and mean absolute magnitude are separate because cancellation is itself part of the interpretation.
SII is labeled `pair SII only`.

<!-- ARENA:cell code -->
```python
figure = plot_heldout_pair_order_capture(evaluated, prepared["feature_names"])
show_figure(figure, "06_heldout_pair_order_capture")
```

<!-- ARENA:cell markdown -->
Each complete pair order is learned on the 153 explanation rows, then carried unchanged to the 119
audit rows. The target is mean absolute full-context synergy under the declared model-input
replacement. All 256 random-order curves remain visible; this is not a causal validation or a
significance test. The three rule-specific curves can be close even when their signed coefficients
differ: ordering 28 pairs discards magnitude and sign information. In the ridge panel, normalizing
near-zero floating-point residuals produces a nontrivial-looking curve; its absolute interaction
scale is the relevant control. Diamonds in the final panel are the original clean model's statistics;
the dots are the 128 within-size shuffled games.

<details>
<summary>Interpretation - does a curve above random identify a causal circuit?</summary>

No. It means an explanation-side ranking concentrates a held-out model-input diagnostic more quickly
than random pair orderings. The target is the full-context second difference under intact donor
replacement. It is neither a causal intervention on physical concrete nor the defining objective of
any one interaction index. Compare the curve with the signed heatmaps and reconstruction plots.

</details>

<!-- ARENA:cell code -->
```python
figure = plot_order_two_reconstruction(evaluated)
show_figure(figure, "07_order_two_reconstruction")
```

<!-- ARENA:cell markdown -->
Faith-Shap may be labeled the optimum for its normalized kernel error because that is its definition.
The coalition-size residuals show where each order-two reconstruction misses. Empty and full
constraints hold to float64 solver precision, and the degree-two and ridge controls should expose implementation drift.

<!-- ARENA:cell code -->
```python
figure = plot_background_changes_the_game(evaluated, prepared["feature_names"])
show_figure(figure, "08_background_changes_the_game")
```

<!-- ARENA:cell markdown -->
`nested16` and `nested128` are related prefixes; `independent64_a` and `independent64_b` are alternate
finite supports. Their spread is not a standard error. Moving the background changes the coalition
game before either interaction rule is applied.

<!-- ARENA:cell code -->
```python
figure = plot_contextual_pair_interventions(
    evaluated, play, prepared["feature_names"]
)
show_figure(figure, "09_contextual_pair_interventions")
```

<!-- ARENA:cell markdown -->
The direct-forward path rebuilds the four full-context inputs independently of table indexing. Its
agreement with the stored second difference checks mask order and sign. Shapley-Taylor, Faith-Shap,
and SII can still differ validly because they aggregate contexts differently.

<!-- ARENA:cell markdown -->
## Try It Yourself: donor-context microscope

Edit the model, row, background, pair, context, or donor. `DONOR_SOURCE_ID=None` uses the whole
finite donor bank; choosing one donor constructs and clearly labels a different single-donor game.
Start with the predeclared explanation row. Audit rows are available only here, after every frozen
study figure has already been computed.

<!-- ARENA:cell code -->
```python
MODEL_KEY = "clean_0"
SOURCE_ID = 0
BACKGROUND_KEY = "primary64"
PAIR = (0, 1)
CONTEXT_MASK = 252
DONOR_SOURCE_ID = None

context_play = empirical_context_play(
    evaluated,
    methods=methods,
    model_key=MODEL_KEY,
    source_id=SOURCE_ID,
    background_key=BACKGROUND_KEY,
    pair=PAIR,
    context_mask=CONTEXT_MASK,
    donor_source_id=DONOR_SOURCE_ID,
)
figure = plot_donor_context_play(context_play, prepared["feature_names"])
show_figure(figure, "11_donor_context_play")
```

<!-- ARENA:cell markdown -->
<details>
<summary>Ideas to try</summary>

1. Keep the row and pair fixed, but switch between `primary64`, its nested prefixes, and the two
   independent banks. Say explicitly that you changed the game.
2. Select donor `173`, then return to `None`. Compare the donor-specific second difference with the
   bank mean and look for signed cancellation.
3. Change `CONTEXT_MASK` while keeping it disjoint from `PAIR`. Compare the local derivative with
   each method's context aggregation.
4. Select an audit source ID only after the frozen study. The output should label this as exploratory
   audit reuse rather than a selected headline example.

</details>

<!-- ARENA:cell markdown -->
# Conclusion

The pure four-way game answered the opening question with two coherent order-two allocations:
Shapley-Taylor kept lower-order dividends at zero and split the omitted term across pairs; Faith-Shap
moved both singleton and pair coefficients to improve a particular constrained reconstruction. SII
asked a third question by averaging pair derivatives over contexts.

On a real fitted-model game, the same distinctions remain:

- efficiency concerns the full-minus-empty ledger, not every interior coalition;
- faithfulness is specific to a kernel, degree, and exact constraints;
- a pair coefficient can include routed higher-order credit;
- a contextual second difference is one direct model-input intervention, not the unique pair index;
- changing the donor bank changes the explanation game; and
- exact enumeration costs $2^n$, so the eight-feature study does not solve scalable interaction
  discovery.

The Concrete outputs explain specified fitted predictors on finite empirical donor-replacement games
in MPa. They do not identify physical ingredient interactions or population causal effects.

<!-- ARENA:cell markdown -->
## Further reading

- Sundararajan, Dhamdhere, and Agarwal, [*The Shapley Taylor Interaction
  Index*](https://proceedings.mlr.press/v119/sundararajan20a.html).
- Tsai, Yeh, and Ravikumar, [*Faith-Shap: The Faithful Shapley Interaction
  Index*](https://www.jmlr.org/papers/v24/22-0202.html).
- Section 16.3 for the exact Shapley Interaction Index convention used as the labeled comparator.
- Sections 16.14–16.15 for the known Concrete predictors, finite backgrounds, and attribution-game
  controls reused here.
