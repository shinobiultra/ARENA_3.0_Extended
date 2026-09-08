<!-- ARENA:cell markdown -->
# [11.1] PCA, SVD, and Geometry Controls

# Introduction

Suppose you can identify every weekday from a model's activations. Does that
mean the model represents the week as a circle? Could it just memorize seven
unrelated words? And if you find a circle, does the model use it to compute?

We'll separate these questions. First, you'll recover a known calendar hidden
inside a residual stream and change its answer while preserving another piece
of information. Then you'll examine real language-model activations and ask
which parts of that argument survive. The exact-model example gives us a place
to debug; it doesn't tell us what a pretrained model must do.

## Content & Learning Objectives

### 1️⃣ Find a useful view

> - Implement centered PCA from SVD and fit transformations on training data.
> - Explain when removing a prompt-template mean uses the evaluation batch.
> - Compare word identity with chronological order using different controls.

### 2️⃣ Check what the picture preserves

> - Implement circular probes, nearest neighbors, trustworthiness and continuity.
> - Align representations without mistaking rotations for different geometry.
> - Compare nonlinear views across seeds and settings using numerical evidence.

### 3️⃣ Change the model, then read its answer

> - Derive the smallest activation edit that changes a fitted coordinate.
> - Implement a native residual-stream replacement at an aligned token position.
> - Compare the resulting answer with full-donor and random interventions.

## Setup

The numerical exercises below run on CPU. The later pretrained-model section
uses a CUDA GPU. Treat the difficulty ratings and time estimates as relative:
try each exercise yourself, use the help when stuck, and read the solution when
you would learn more by moving on. Keep your predictions beside your results.

The complete solution runs in about seven minutes on the tested RTX 5090
Laptop GPU and uses 13.7 GiB of CUDA memory. Allow several hours for the
exercises and interpretation. The saved solution outputs let you read the real
results before deciding whether to rerun the pretrained-model experiments.

<!-- ARENA:cell code -->
```python
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch as t
from IPython import get_ipython

if get_ipython() is not None:
    get_ipython().run_line_magic("matplotlib", "inline")

root_dir = next(
    p
    for p in [Path.cwd(), *Path.cwd().parents]
    if (p / "chapter11_representation_geometry").exists()
)
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))
from chapter11_representation_geometry.exercises.part1_pca_svd_geometry_controls import (
    geometry_methods_tests as tests,
)

assets_dir = Path(os.environ.get("ARENA_GEOMETRY_OUTPUT_DIRECTORY", "geometry_outputs"))
assets_dir.mkdir(parents=True, exist_ok=True)
t.set_num_threads(4)
plt.style.use("seaborn-v0_8-whitegrid")
```

<!-- ARENA:cell markdown -->
# 1️⃣ Find a useful view

## PCA: which directions contain variation?

Let $X\in\mathbb R^{n\times d}$ contain one activation per row. After subtracting
the training mean, write $X-\bar x=U S V^\top$. The columns of $V$ are orthogonal
directions in activation space. Projecting onto the first $k$ columns gives
$Z=(X-\bar x)V_k$, with shape `[examples, k]`.

The variance along PC $i$ is $S_i^2/(n-1)$. Its fraction of **all** training
variance is $S_i^2/\sum_jS_j^2$; don't normalize using only retained components.
A plot can show little of the original variance even when its axes look tidy.

For a new activation, reuse the training mean and basis. Fitting another PCA
to the new points answers a different question. Component signs are arbitrary;
compare the subspace or pairwise distances when checking two correct answers.

<!-- ARENA:cell markdown -->
### Exercise 1 — Fit PCA on training activations

> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵🔵🔵⚪
> You should spend 15 minutes on this exercise.
> ```

Return the mean `[d]`, basis `[d, k]` and variance fractions `[k]`. Use reduced SVD. The test includes a huge direction found only in held-out data: your fitted basis must ignore it.

<details><summary>Help — where should I start?</summary>

The right singular vectors are the rows of `Vh`. Transpose the first `k` rows. A common bug is returning columns of `U`, which live in example space.

</details>

<details><summary>Expected output</summary>

The two retained components explain 90% and 10%; a held-out vector entirely in the third coordinate projects to `[0, 0]`.

</details>

<details><summary>Solution</summary>

```python
def fit_pca(x: t.Tensor, components: int):
    """Return training mean, orthonormal column basis and total-variance fractions."""
    if x.ndim != 2 or x.shape[0] < 2 or not 1 <= components <= min(x.shape):
        raise ValueError("PCA needs at least two rows and a feasible component count.")
    mean = x.mean(0)
    _, singular, vh = t.linalg.svd(x - mean, full_matrices=False)
    energy = singular.square()
    if energy.sum() <= 0:
        raise ValueError("Constant observations have no explained-variance ratio.")
    return mean, vh[:components].T, energy[:components] / energy.sum()
```

</details>

<!-- ARENA:cell code -->
```python
def fit_pca(x: t.Tensor, components: int):
    """Return training mean, orthonormal column basis and total-variance fractions."""
    raise NotImplementedError()


tests.test_pca_fits_only_training_and_preserves_total_variance(SimpleNamespace(**globals()))
print("All tests passed!")
```

<!-- ARENA:cell markdown -->
## A calendar hidden by prompt wording

Represent weekday $a$ by $(\cos(2\pi a/7),\sin(2\pi a/7))$. Monday is phase zero;
Sunday sits next to Monday across the angle boundary. Add a large offset for
each prompt template, then rotate the entire representation. Raw PCA can spend
its first components explaining these template offsets instead of weekdays.

If every template includes each class exactly once, subtracting that template's
mean removes its offset. But this is a **batch assumption**. Applying it to a
held-out template uses that batch's examples. If the batch contains Monday
three times and no Sunday, the subtraction also changes the calendar signal.
We'll call this *transductive template centering*, and keep ordinary
train-fitted centering as a separate comparison.

<!-- ARENA:cell markdown -->
<details><summary>Why use an explicit least-squares solver?</summary>

Later we will cast activation edits to BF16. Tiny fitting differences can land
on opposite sides of a rounding boundary and change a winning token. We use
the CPU SVD least-squares driver (`gelsd`) and test repeatability on identical
inputs, including a rank-deficient design. This still solves the same minimum-norm
least-squares problem; it does not change the target coordinates.

</details>

### Exercise 2 — Represent a circle and remove a template offset

> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵🔵🔵⚪
> You should spend 15 minutes on this exercise.
> ```

Implement phase coordinates, nearest-angle decoding and class-balanced template centering. Use `-1` to abstain when both coordinates are exactly zero: the zero vector has no angle.

<details><summary>Help — where should I start?</summary>

Use `atan2(y, x)`, divide by the phase spacing, round, and wrap modulo the period. Subtract along the class axis of `[templates, classes, width]`, not across templates.

</details>

<details><summary>Expected output</summary>

Multiplying every radius by nine leaves all predictions unchanged. A tiny negative angle still decodes to Monday. An incomplete template batch does not recover the original coordinates exactly.

</details>

<details><summary>Solution</summary>

```python
def circular_coordinates(labels: t.Tensor, period: int):
    """Monday and January have phase zero; adjacent classes differ by 2π/period."""
    if period < 3:
        raise ValueError('A circular family needs at least three distinct classes.')
    angle = labels * (2 * t.pi / period)
    return t.stack((angle.cos(), angle.sin()), -1)

def angular_predictions(coordinates: t.Tensor, period: int):
    """Nearest unit-circle class by angle; an exactly zero vector abstains as -1."""
    if coordinates.shape[-1] != 2 or period < 3:
        raise ValueError('Use two coordinates and period >= 3.')
    angle = t.atan2(coordinates[..., 1], coordinates[..., 0])
    prediction = t.round(angle * period / (2 * t.pi)).long().remainder(period)
    return t.where(coordinates.square().sum(-1) > 0, prediction, -1)

def center_template_grid(grid: t.Tensor):
    """Subtract each template's complete class-balanced batch mean.

This uses all examples in that batch. On a held-out template it is an explicit
transductive operation, not a preprocessing transform fitted on training data.
"""
    if grid.ndim != 3 or grid.shape[1] < 2:
        raise ValueError('Expected [templates, classes, width] with at least two classes.')
    return grid - grid.mean(1, keepdim=True)
```

</details>

<!-- ARENA:cell code -->
```python
def circular_coordinates(labels: t.Tensor, period: int):
    """Monday and January have phase zero; adjacent classes differ by 2π/period."""
    raise NotImplementedError()

def angular_predictions(coordinates: t.Tensor, period: int):
    """Nearest unit-circle class by angle; an exactly zero vector abstains as -1."""
    raise NotImplementedError()

def center_template_grid(grid: t.Tensor):
    """Subtract each template's complete class-balanced batch mean.

This uses all examples in that batch. On a held-out template it is an explicit
transductive operation, not a preprocessing transform fitted on training data.
"""
    raise NotImplementedError()

tests.test_angles_wrap_around_and_zero_abstains(SimpleNamespace(**globals()))
tests.test_template_centering_is_explicitly_transductive(SimpleNamespace(**globals()))
print("All tests passed!")
```

<!-- ARENA:cell code -->
```python
generator = t.Generator().manual_seed(1101)
rotation = t.linalg.qr(t.randn(8, 8, generator=generator, dtype=t.float64)).Q
calendar = circular_coordinates(t.arange(7, dtype=t.float64), 7)
residual = t.cat((calendar, t.zeros(7, 6, dtype=t.float64)), dim=1) @ rotation.T
offsets = (
    t.tensor(
        [
            [0.0, 0.0, 7.0, -4.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -5.0, 8.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 11.0, 6.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=t.float64,
    )
    @ rotation.T
)
grid = residual[None] + offsets[:, None]
centered = center_template_grid(grid)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
for ax, values, title in zip(
    axes,
    [grid, centered],
    ["Template offsets dominate", "Balanced template centering"],
    strict=True,
):
    mean, basis, variance = fit_pca(values[:2].flatten(0, 1), 2)
    projected = (values - mean) @ basis
    for i, marker in enumerate(["o", "s", "X"]):
        ax.scatter(
            projected[i, :, 0],
            projected[i, :, 1],
            c=np.arange(7),
            cmap="tab10",
            vmin=0,
            vmax=9,
            marker=marker,
            s=65,
            edgecolor="black",
            label="held-out batch" if i == 2 else f"train template {i}",
        )
    ax.set(title=title, xlabel=f"PC1 ({variance[0]:.0%})", ylabel=f"PC2 ({variance[1]:.0%})")
    ax.legend(fontsize=8)
axes[1].set_aspect("equal", adjustable="box")
fig.savefig(assets_dir / "geometry_template_offsets.png", dpi=160)
plt.show()
```

<!-- ARENA:cell markdown -->
## A coordinate probe is a supervised claim

We can fit a map from activation coordinates to the known calendar phase.
With centered inputs $Z$ and targets $Y$, ridge regression solves
$\min_W\|ZW-Y\|_F^2+\lambda\|W\|_F^2$. Fit the intercept separately so that a
large constant target offset isn't shrunk towards zero.

A low training error is unsurprising when the probe has many dimensions. Use
unseen examples, and distinguish unseen prompt templates from unseen entities:
a probe which has seen Monday under one template has still seen Monday.
$R^2=1-\mathrm{SSE}/\mathrm{SST}$ compares coordinate predictions with the
constant-mean predictor. Negative values are meaningful failures, not values
to clip out of the chart.

<!-- ARENA:cell markdown -->
### Exercise 3 — Fit and evaluate an affine coordinate probe

> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵🔵🔵⚪
> You should spend 15 minutes on this exercise.
> ```

Return the input mean, weights and target mean. For zero ridge use least squares; for positive ridge solve the regularized normal equations. Then implement joint-coordinate R².

<details><summary>Help — where should I start?</summary>

Predictions are `(x - mean) @ weights + bias`. Do not regularize a column of ones. For the test with `x = [-1, 1]`, the design energy is 2.

</details>

<details><summary>Expected output</summary>

For `y = 3x + 7` and ridge 2, the slope becomes 1.5 while the intercept stays 7. The mean-only predictor has R² zero.

</details>

<details><summary>Solution</summary>

```python
def fit_ridge_coordinates(x: t.Tensor, targets: t.Tensor, ridge: float = 0.0):
    """Affine least squares; intercept is not penalized, penalty is ridge * ||W||²."""
    if x.ndim != 2 or targets.ndim != 2 or x.shape[0] != targets.shape[0] or ridge < 0:
        raise ValueError("Use paired observation matrices and nonnegative ridge.")
    mean, bias = x.mean(0), targets.mean(0)
    centered, values = x - mean, targets - bias
    if ridge == 0:
        weights = t.linalg.lstsq(
            centered, values, driver="gelsd" if x.device.type == "cpu" else "gels"
        ).solution
    else:
        weights = t.linalg.solve(
            centered.T @ centered + ridge * t.eye(x.shape[1], device=x.device, dtype=x.dtype),
            centered.T @ values,
        )
    return mean, weights, bias


def coordinate_r2(targets: t.Tensor, predictions: t.Tensor):
    """Joint coordinate R², relative to the observed target mean; may be negative."""
    if targets.shape != predictions.shape:
        raise ValueError("Prediction and target inventories must match.")
    denominator = (targets - targets.mean(0)).square().sum()
    if denominator <= 0:
        raise ValueError("R² is undefined when targets have no variation.")
    return 1 - (targets - predictions).square().sum() / denominator
```

</details>

<!-- ARENA:cell code -->
```python
def fit_ridge_coordinates(x: t.Tensor, targets: t.Tensor, ridge: float = 0.0):
    """Affine least squares; intercept is not penalized, penalty is ridge * ||W||²."""
    raise NotImplementedError()


def coordinate_r2(targets: t.Tensor, predictions: t.Tensor):
    """Joint coordinate R², relative to the observed target mean; may be negative."""
    raise NotImplementedError()


tests.test_ridge_has_unpenalized_intercept_and_known_shrinkage(SimpleNamespace(**globals()))
tests.test_least_squares_repeated_fits_and_rank_deficiency(SimpleNamespace(**globals()))
print("All tests passed!")
```

<!-- ARENA:cell markdown -->
## Identity is not circular order

Imagine seven isolated points, one per weekday. A classifier can identify all
seven perfectly without their distances reflecting the calendar. Relabeling a
ring preserves separability too: the points are still distinct, but Tuesday
might now lie opposite Monday.

We'll compare all pairwise class-centroid distances with distances around the
chronological unit circle. Permute **class names** as a unit across templates,
not individual rows. This preserves the easy identity problem while removing
the proposed chronological organization. Rotations and reflections of a real
ring can still tie with the original arrangement; those ties belong in the null.

The permutation p-value `(1 + null scores >= observed) / (1 + number of nulls)`
never becomes zero just because a finite simulation missed a rare arrangement.

<!-- ARENA:cell markdown -->
### Exercise 4 — Separate label prediction from circular structure

> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵🔵🔵⚪
> You should spend 15 minutes on this exercise.
> ```

Implement Euclidean kNN with deterministic ties, a distance-correlation statistic, and its supplied class-permutation null. Use every off-diagonal pair exactly once.

<details><summary>Help — where should I start?</summary>

`torch.pdist` returns the upper triangle without the diagonal. Center each distance vector before their normalized dot product. For kNN ties, sort labels and choose the smallest class.

</details>

<details><summary>Expected output</summary>

A chronological ring has distance agreement 1. A scrambled ring still gives perfect identity classification, while its calendar-order agreement falls below 0.5.

</details>

<details><summary>Solution</summary>

```python
def knn_predictions(train: t.Tensor, labels: t.Tensor, test: t.Tensor, k: int = 3):
    """Euclidean kNN; deterministic nearest-row and lowest-label tie breaking."""
    if not 1 <= k <= len(train) or labels.shape != (len(train),):
        raise ValueError('Use one label per training row and a valid neighbor count.')
    classes, encoded = labels.unique(sorted=True, return_inverse=True)
    neighbors = t.cdist(test, train).argsort(dim=-1, stable=True)[:, :k]
    votes = t.nn.functional.one_hot(encoded[neighbors], len(classes)).sum(1)
    return classes[votes.argmax(-1)]

def cyclic_distance_agreement(class_points: t.Tensor):
    """Correlation of pair distances with chronological circle chord distances.

Class identities, not individual prompt rows, are the permutation unit. A high
score can support periodic order; ID classification alone cannot.
"""
    count = class_points.shape[0]
    if count < 4:
        raise ValueError('Use at least four ordered class centroids.')
    target = circular_coordinates(t.arange(count, device=class_points.device, dtype=class_points.dtype), count)
    observed, expected = t.pdist(class_points), t.pdist(target)
    a, b = observed - observed.mean(), expected - expected.mean()
    denominator = a.norm() * b.norm()
    if denominator <= 0:
        raise ValueError('Distance agreement is undefined for a constant distance pattern.')
    return (a @ b) / denominator

def cyclic_permutation_test(class_points: t.Tensor, permutations: t.Tensor):
    """One-sided Monte Carlo null including the observed arrangement via +1."""
    count = class_points.shape[0]
    if permutations.ndim != 2 or permutations.shape[1] != count or not len(permutations):
        raise ValueError('Each null must permute every class exactly once.')
    expected = t.arange(count, device=permutations.device).expand_as(permutations)
    if not t.equal(permutations.sort(-1).values, expected):
        raise ValueError('Each null must permute every class exactly once.')
    observed = cyclic_distance_agreement(class_points)
    null = t.stack([cyclic_distance_agreement(class_points[order]) for order in permutations])
    probability = (1 + (null >= observed - 1e-7).sum()) / (1 + len(null))
    return observed, null, probability
```

</details>

<!-- ARENA:cell code -->
```python
def knn_predictions(train: t.Tensor, labels: t.Tensor, test: t.Tensor, k: int=3):
    """Euclidean kNN; deterministic nearest-row and lowest-label tie breaking."""
    raise NotImplementedError()

def cyclic_distance_agreement(class_points: t.Tensor):
    """Correlation of pair distances with chronological circle chord distances.

Class identities, not individual prompt rows, are the permutation unit. A high
score can support periodic order; ID classification alone cannot.
"""
    raise NotImplementedError()

def cyclic_permutation_test(class_points: t.Tensor, permutations: t.Tensor):
    """One-sided Monte Carlo null including the observed arrangement via +1."""
    raise NotImplementedError()

tests.test_calendar_order_null_is_distinct_from_class_identification(SimpleNamespace(**globals()))
print("All tests passed!")
```

<!-- ARENA:cell code -->
```python
orders = t.stack([t.randperm(7, generator=generator) for _ in range(256)])
observed, null, probability = cyclic_permutation_test(residual, orders)
fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
axes[0].scatter(calendar[:, 0], calendar[:, 1], c=np.arange(7), cmap="tab10", vmin=0, vmax=9, s=80)
for i, day in enumerate(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]):
    axes[0].annotate(day, calendar[i], xytext=(6, 5), textcoords="offset points")
axes[0].set(
    title="Known chronological order",
    aspect="equal",
    xlabel="cosine coordinate",
    ylabel="sine coordinate",
)
axes[1].hist(null.numpy(), bins=20, color="#8a9eb2", label="Class-order permutations")
axes[1].axvline(float(observed), color="#bf4c28", linewidth=2, label="Chronological order")
axes[1].set(
    title=f"Identity survives every permutation; order does not\np = {float(probability):.4f}",
    xlabel="Correlation with chronological chord distances",
    ylabel="Null arrangements",
)
axes[1].legend(fontsize=8)
fig.savefig(assets_dir / "geometry_calendar_order_null.png", dpi=160)
plt.show()
```

<!-- ARENA:cell markdown -->
# 2️⃣ Check what the picture preserves

## Neighborhoods can be invented or lost

A projected point can acquire a neighbor that was distant in the original
space. *Trustworthiness* penalizes these intrusions by their original ranks.
It can also lose an originally close neighbor. *Continuity* asks the reverse
question. Both range from 0 to 1 in their usual valid regime; identity views
score 1. A projection can fail these checks while still making attractive
clusters.

For each point, exclude itself, rank all other points starting at one, and find
its first `k` neighbors. For trustworthiness, sum `(original_rank - k)` only for
neighbors introduced in the projection. Normalize the total penalty by
`2 / (n * k * (2*n - 3*k - 1))`. Swap the two spaces for continuity.

<!-- ARENA:cell markdown -->
### Exercise 5 — Measure neighborhood errors

> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵🔵🔵⚪
> You should spend 15 minutes on this exercise.
> ```

Implement both rank-based scores for aligned high- and low-dimensional observations. Restrict `k < n/2`. Neither function may read labels.

<details><summary>Help — where should I start?</summary>

Set the diagonal distance to infinity before sorting. Build inverse ranks with `scatter_`. A common bug includes each point as its own nearest neighbor, inflating both scores.

</details>

<details><summary>Expected output</summary>

The random-fixture results agree with scikit-learn trustworthiness in both directions. The identity mapping scores `(1, 1)`.

</details>

<details><summary>Solution</summary>

```python
def neighborhood_scores(high: t.Tensor, low: t.Tensor, k: int = 5):
    """Trustworthiness (intrusions) and continuity (missing true neighbors)."""
    count = len(high)
    if len(low) != count or not 1 <= k < count / 2:
        raise ValueError("Use aligned rows and 1 <= k < n/2.")

    def ranks(points):
        distances = t.cdist(points, points)
        distances.fill_diagonal_(float("inf"))
        order = distances.argsort(dim=-1, stable=True)
        rank = t.empty_like(order)
        rank.scatter_(1, order, t.arange(1, count + 1, device=points.device).expand(count, -1))
        return rank

    original, projected = ranks(high), ranks(low)
    intrusion = t.where(projected <= k, (original - k).clamp_min(0), 0).sum()
    missing = t.where(original <= k, (projected - k).clamp_min(0), 0).sum()
    factor = 2 / (count * k * (2 * count - 3 * k - 1))
    return 1 - factor * intrusion, 1 - factor * missing
```

</details>

<!-- ARENA:cell code -->
```python
def neighborhood_scores(high: t.Tensor, low: t.Tensor, k: int = 5):
    """Trustworthiness (intrusions) and continuity (missing true neighbors)."""
    raise NotImplementedError()


tests.test_neighborhood_ranks_match_independent_library_and_duality(SimpleNamespace(**globals()))
print("All tests passed!")
```

<!-- ARENA:cell markdown -->
## The axes are not the result

Two equally good projections may be rotated, translated or reflected. Before
calling this instability, align the same observations with an orthogonal map.
For centered row-vector matrices `X` and `Y`, write `X.T @ Y = U S Vh`; the
least-squares orthogonal alignment is `R = U @ Vh`.

This preserves distances and permits reflections. It cannot repair a distorted
neighborhood. Fit the alignment on shared training observations, then inspect
new observations; matching two training pictures alone proves little.

<!-- ARENA:cell markdown -->
### Exercise 6 — Align representations with Procrustes

> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵🔵🔵⚪
> You should spend 10 minutes on this exercise.
> ```

Return the two training means and orthogonal row-vector transform. Keep scale fixed so that an incorrect radius remains observable.

<details><summary>Help — where should I start?</summary>

Transform a new source point using `(x - source_mean) @ R + target_mean`. Reversing `U` and `Vh` rotates in the wrong direction.

</details>

<details><summary>Expected output</summary>

The test recovers a reflection plus a large translation and applies it correctly to a new point.

</details>

<details><summary>Solution</summary>

```python
def fit_procrustes(source: t.Tensor, target: t.Tensor):
    """Fit centered row-vector orthogonal alignment; reflections are permitted."""
    if source.shape != target.shape or source.ndim != 2:
        raise ValueError("Align the same observations in equal-dimensional spaces.")
    source_mean, target_mean = source.mean(0), target.mean(0)
    u, _, vh = t.linalg.svd((source - source_mean).T @ (target - target_mean))
    return source_mean, target_mean, u @ vh
```

</details>

<!-- ARENA:cell code -->
```python
def fit_procrustes(source: t.Tensor, target: t.Tensor):
    """Fit centered row-vector orthogonal alignment; reflections are permitted."""
    raise NotImplementedError()


tests.test_procrustes_recovers_rotation_reflection_and_translation(SimpleNamespace(**globals()))
print("All tests passed!")
```

<!-- ARENA:cell markdown -->
# 3️⃣ Change the model, then read its answer

## From decoding to an intervention

Suppose the probe is $f(h)=(h-\mu)BW+b$, where the PCA basis $B$ has orthonormal
columns. We want $f(h+\delta)=y_\mathrm{target}$ while changing as little as
possible. If $W$ has independent columns, the minimum-norm edit is

$$\delta=(y_\mathrm{target}-f(h))W^+B^\top.$$

Its component outside the probe's column space is zero. Verify this algebra,
then ask what the model does with the changed residual. The constraint makes
the probe report the desired coordinate by construction; it does **not** make
the model's answer change by construction.

As a control, copying a donor's coordinates in an orthonormal subspace $Q$ gives
$h'=h+(h_\mathrm{donor}-h)QQ^\top$. Copying the entire residual is a useful upper
comparison but also replaces nuisance information.

<!-- ARENA:cell markdown -->
### Exercise 7 — Change a coordinate without replacing the whole residual

> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵🔵🔵⚪
> You should spend 20 minutes on this exercise.
> ```

Implement the minimum-norm probe edit and donor subspace replacement. Reject non-orthogonal bases and uncontrollable coordinate systems.

<details><summary>Help — where should I start?</summary>

For weights `[k, 2]`, the pseudoinverse has shape `[2, k]`. Multiply the coordinate error by that inverse and by `basis.T`. Never invert a non-square weight matrix.

</details>

<details><summary>Expected output</summary>

The edited probe exactly reaches the requested coordinates. The exact calendar model changes all seven answers by three days while retaining its unrelated coordinate at 19. An orthogonal rank-two control changes no calendar answer.

</details>

<details><summary>Solution</summary>

```python
def replace_coordinates(activations: t.Tensor, mean: t.Tensor, basis: t.Tensor,
                        weights: t.Tensor, bias: t.Tensor, target: t.Tensor):
    """Minimum-L2 change satisfying the fitted PCA-coordinate probe.

The PCA basis must have orthonormal columns, and the probe must have full
column rank. Nothing outside the probe's column space is changed. Satisfying
this equation alone does not establish the model uses these coordinates.
"""
    if basis.ndim != 2 or weights.ndim != 2 or basis.shape[1] != weights.shape[0]:
        raise ValueError('Basis [width,k] and weights [k,coordinates] must align.')
    if not t.allclose(basis.T @ basis, t.eye(basis.shape[1], device=basis.device, dtype=basis.dtype), atol=1e-5, rtol=1e-5):
        raise ValueError('Minimum-norm coordinates require an orthonormal PCA basis.')
    if t.linalg.matrix_rank(weights) != weights.shape[1]:
        raise ValueError('Every requested coordinate must be independently controllable.')
    prediction = (activations - mean) @ basis @ weights + bias
    correction = (target - prediction) @ t.linalg.pinv(weights) @ basis.T
    return activations + correction

def replace_subspace(activations: t.Tensor, donor: t.Tensor, basis: t.Tensor):
    """Copy only donor coordinates in an orthonormal column basis."""
    if not t.allclose(basis.T @ basis, t.eye(basis.shape[1], device=basis.device, dtype=basis.dtype), atol=1e-5, rtol=1e-5):
        raise ValueError('A donor subspace needs an orthonormal column basis.')
    return activations + (donor - activations) @ basis @ basis.T
```

</details>

<!-- ARENA:cell code -->
```python
def replace_coordinates(activations: t.Tensor, mean: t.Tensor, basis: t.Tensor, weights: t.Tensor, bias: t.Tensor, target: t.Tensor):
    """Minimum-L2 change satisfying the fitted PCA-coordinate probe.

The PCA basis must have orthonormal columns, and the probe must have full
column rank. Nothing outside the probe's column space is changed. Satisfying
this equation alone does not establish the model uses these coordinates.
"""
    raise NotImplementedError()

def replace_subspace(activations: t.Tensor, donor: t.Tensor, basis: t.Tensor):
    """Copy only donor coordinates in an orthonormal column basis."""
    raise NotImplementedError()

tests.test_minimum_norm_coordinate_edit_matches_constraint_solution(SimpleNamespace(**globals()))
tests.test_donor_subspace_does_not_copy_complement(SimpleNamespace(**globals()))
tests.test_exact_ring_plane_changes_answer_and_preserves_nuisance(SimpleNamespace(**globals()))
print("All tests passed!")
```

<!-- ARENA:cell code -->
```python
mean, basis, _ = fit_pca(residual, 2)
probe_mean, weights, bias = fit_ridge_coordinates((residual - mean) @ basis, calendar)
bias = bias - probe_mean @ weights
receiver = residual + 19 * rotation[:, 2]
desired = circular_coordinates((t.arange(7, dtype=t.float64) + 3) % 7, 7)
patched = replace_coordinates(receiver, mean, basis, weights, bias, desired)
wrong_plane = replace_subspace(receiver, residual.roll(-3, 0), rotation[:, 2:4])


# This exact model reads calendar phase in one head and an unrelated scalar in another.
def calendar_logits(h):
    return (h @ rotation[:, :2]) @ calendar.T


fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
for ax, h, title in zip(
    axes,
    [receiver, patched, wrong_plane],
    ["Original residual", "Discovered calendar plane: +3 days", "Orthogonal rank-two control"],
    strict=True,
):
    image = ax.imshow(calendar_logits(h).numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.grid(False)
    ax.set(
        title=title, xlabel="Answer class", ylabel="Original day", xticks=range(7), yticks=range(7)
    )
    for i, answer in enumerate(calendar_logits(h).argmax(-1)):
        ax.text(int(answer), i, "×", ha="center", va="center", fontsize=16, color="white")
fig.colorbar(image, ax=axes, shrink=0.8, label="Exact-model answer logit")
fig.savefig(assets_dir / "geometry_plane_intervention.png", dpi=160)
plt.show()
print(
    "Unrelated coordinate after the calendar edit:",
    (patched @ rotation[:, 2]).round(decimals=6).tolist(),
)
```

<!-- ARENA:cell markdown -->
## Try It Yourself — turn the answer around the circle

Change the angle and radius below. Before running, predict when the answer
changes, what happens at radius zero, and whether doubling the radius changes
confidence or identity. Then try replacing the whole donor residual and inspect
the unrelated coordinate. Keep the orthogonal-plane control in view.

<!-- ARENA:cell code -->
```python
PLAY_ANGLE_DEGREES = 155
PLAY_RADIUS = 1.0
phase = t.tensor(PLAY_ANGLE_DEGREES * np.pi / 180, dtype=t.float64)
target = PLAY_RADIUS * t.stack((phase.cos(), phase.sin()))
played = replace_coordinates(receiver[:1], mean, basis, weights, bias, target)
probabilities = calendar_logits(played).softmax(-1)[0]
plt.figure(figsize=(8, 3.3))
positions = np.arange(7)
plt.bar(
    positions - 0.22,
    calendar_logits(receiver[:1]).softmax(-1)[0].numpy(),
    width=0.22,
    label="Original",
    color="#8a9eb2",
)
plt.bar(positions, probabilities.numpy(), width=0.22, label="Coordinate edit", color="#bf4c28")
plt.bar(
    positions + 0.22,
    calendar_logits(wrong_plane[:1]).softmax(-1)[0].numpy(),
    width=0.22,
    label="Orthogonal-plane control",
    color="#487d9b",
)
plt.legend(fontsize=8)
plt.xticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
plt.ylabel("Exact-model answer probability")
plt.title(
    f"Angle {PLAY_ANGLE_DEGREES}° | radius {PLAY_RADIUS} | unrelated coordinate {float(played @ rotation[:, 2]):.2f}"
)
plt.tight_layout()
plt.savefig(assets_dir / "geometry_angle_play.png", dpi=160)
plt.show()
```

<!-- ARENA:cell markdown -->
## Put the edit into a native model

A hook must modify the semantic token you intended, not the same array index in
every prompt. Left padding, right padding and different tokenized lengths can
all change that index. Carry an explicit position for every row and check it
against the attention mask.

For a decoder block, its input is the residual before that block performs its
attention and MLP writes. A forward pre-hook can clone this tensor, replace one
position per row, and return the modified input. Remove the hook in `finally`:
a failed experiment must not silently alter the next baseline run.

<!-- ARENA:cell markdown -->
### Exercise 8 — Replace one real token at one residual boundary

> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵🔵🔵⚪
> You should spend 20 minutes on this exercise.
> ```

Implement aligned token reads and a one-forward residual replacement. Return the actual logits so that answer scoring remains visible in the experiment.

<details><summary>Help — where should I start?</summary>

Use row indices together with the per-row positions. Reject padding. Register the pre-hook once, then place model execution inside `try` and hook removal inside `finally`.

</details>

<details><summary>Expected output</summary>

Only the selected tokens change. Left/right padding is respected, and no hook remains after success, bad input or a deliberate model exception.

</details>

<details><summary>Solution</summary>

```python
def gather_token_positions(hidden: t.Tensor, positions: t.Tensor, attention_mask: t.Tensor):
    """Read one explicitly aligned real token per row; reject padding positions."""
    batch, length = hidden.shape[:2]
    if positions.shape != (batch,) or attention_mask.shape != (batch, length):
        raise ValueError('Supply one position per row and the matching padding mask.')
    if (positions < 0).any() or (positions >= length).any():
        raise ValueError('Selected token position is outside the sequence.')
    row = t.arange(batch, device=hidden.device)
    if not attention_mask[row, positions].bool().all():
        raise ValueError('Never treat padding as a semantic token.')
    return hidden[row, positions]

def forward_with_residual_replacement(model, block, encoded, positions, replacement):
    """Replace the input residual to one native decoder block for one forward pass.

`positions` names the semantic token independently in every padded row. The
hook always gets removed, even if the model raises; caller controls inference
and model mode. Returning logits keeps scoring choices visible to the learner.
"""
    def hook(module, args):
        hidden = args[0]
        current = gather_token_positions(hidden, positions, encoded['attention_mask'])
        if current.shape != replacement.shape:
            raise ValueError('Replacement must match [batch, residual width].')
        modified = hidden.clone()
        modified[t.arange(len(hidden), device=hidden.device), positions] = replacement.to(hidden)
        return (modified, *args[1:])
    handle = block.register_forward_pre_hook(hook)
    try:
        return model(**encoded, use_cache=False).logits
    finally:
        handle.remove()
```

</details>

<!-- ARENA:cell code -->
```python
def gather_token_positions(hidden: t.Tensor, positions: t.Tensor, attention_mask: t.Tensor):
    """Read one explicitly aligned real token per row; reject padding positions."""
    raise NotImplementedError()

def forward_with_residual_replacement(model, block, encoded, positions, replacement):
    """Replace the input residual to one native decoder block for one forward pass.

`positions` names the semantic token independently in every padded row. The
hook always gets removed, even if the model raises; caller controls inference
and model mode. Returning logits keeps scoring choices visible to the learner.
"""
    raise NotImplementedError()

tests.test_semantic_positions_respect_both_padding_sides_and_cleanup_hooks(SimpleNamespace(**globals()))
print("All tests passed!")
```

<!-- ARENA:cell markdown -->
## Two causal questions: change a coordinate, or isolate it?

Our first edit preserves every residual component outside the fitted plane.
A different experiment replaces those components with their training mean.
This can remove competing information, including alternate ways of identifying
the original day. It tests the plane under a changed background, and may also
damage unrelated behavior. Include **mean ablation alone** as a control.

For our affine probe, apply the coordinate edit to a receiver equal to the
training mean. Then the result is $h'=\mu+(y-b)W^+B^\top$. The orthogonal
complement is now the mean's complement, not the original prompt's.

The [paper authors' implementation](https://github.com/JoshEngels/MultiDimensionalFeatures/blob/ff8c7eee8f13d78805f590ced6f3fc2227dd1daa/intervention/circle_finding_utils.py)
uses a related raw-coordinate construction. Write the QR decomposition of the
probe weights as $W=qr$, and let $Q=Bq$. Its intervention is

$$h'=\mu-\mu QQ^\top+y r^{-1}Q^\top.$$

This sets raw plane coordinates; our earlier probe was defined relative to the
training mean and an intercept. These formulas are different when that mean
has a component in the plane. We'll keep both definitions visible and measure
their actual behavior, alongside the pure edit and full-donor replacement.
No-op and mean-only controls help diagnose what each formula changes.

<!-- ARENA:cell markdown -->
### Exercise 9 — Implement both mean-ablation conventions

> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵🔵🔵⚪
> You should spend 15 minutes on this exercise.
> ```

Implement the affine-centered coordinate clamp and the literal author QR formula. The first can reuse your minimum-norm edit with a receiver equal to the training mean; the second uses the raw-coordinate equation above.

<details><summary>Help — keep the coordinate convention explicit</summary>

Keep row-vector order straight: `target @ inverse(r) @ Q.T`. Do not subtract a residual-width mean from a two-dimensional target. Inspect the units of every term.

</details>

<details><summary>Expected output</summary>

The independent four-dimensional example gives `[12, 23, 30, 40]` for the affine-centered convention and `[2.5, 7/3, 30, 40]` for the raw convention. Each satisfies its own constraint. Both reset the complement to its mean.

</details>

<details><summary>Solution</summary>

```python
def mean_ablate_coordinates(mean: t.Tensor, basis: t.Tensor, weights: t.Tensor,
                           bias: t.Tensor, target: t.Tensor):
    """Set fitted coordinates; replace the entire complement by its training mean.

Unlike replace_coordinates, this deliberately discards receiver information
outside the fitted plane. Compare against mean ablation alone and a pure edit.
"""
    receiver = mean.expand(*target.shape[:-1], mean.shape[-1])
    return replace_coordinates(receiver, mean, basis, weights, bias, target)

def author_raw_circle_replacement(mean: t.Tensor, basis: t.Tensor,
                                  weights: t.Tensor, target: t.Tensor):
    """Literal row-vector QR formula in the paper authors' intervention code.

Reference: JoshEngels/MultiDimensionalFeatures, commit ff8c7eee8f13d78805f590ced6f3fc2227dd1daa,
intervention/circle_finding_utils.py: patch_in_subspace_activations (MIT).
This sets raw plane coordinates and mean-ablates the complement. It is not
the affine centered-coordinate constraint used in mean_ablate_coordinates.
"""
    if t.linalg.matrix_rank(weights) != weights.shape[1]:
        raise ValueError('The circular probe needs independent coordinate columns.')
    q, r = t.linalg.qr(weights, mode='reduced')
    plane = basis @ q
    return mean + target @ t.linalg.inv(r) @ plane.T - mean @ plane @ plane.T
```

</details>

<!-- ARENA:cell code -->
```python
def mean_ablate_coordinates(mean: t.Tensor, basis: t.Tensor, weights: t.Tensor, bias: t.Tensor, target: t.Tensor):
    """Set fitted coordinates; replace the entire complement by its training mean.

Unlike replace_coordinates, this deliberately discards receiver information
outside the fitted plane. Compare against mean ablation alone and a pure edit.
"""
    raise NotImplementedError()

def author_raw_circle_replacement(mean: t.Tensor, basis: t.Tensor, weights: t.Tensor, target: t.Tensor):
    """Literal row-vector QR formula in the paper authors' intervention code.

Reference: JoshEngels/MultiDimensionalFeatures, commit ff8c7eee8f13d78805f590ced6f3fc2227dd1daa,
intervention/circle_finding_utils.py: patch_in_subspace_activations (MIT).
This sets raw plane coordinates and mean-ablates the complement. It is not
the affine centered-coordinate constraint used in mean_ablate_coordinates.
"""
    raise NotImplementedError()

tests.test_mean_ablation_and_literal_author_coordinates_answer_different_questions(SimpleNamespace(**globals()))
print("All tests passed!")
```

<!-- ARENA:cell markdown -->
### Exercise 10 — Capture the model at named residual boundaries

> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵🔵🔵⚪
> You should spend 15 minutes on this exercise.
> ```

Register temporary pre-hooks at the supplied boundaries, gather the selected semantic token, and return the actual logits and a CPU tensor `[batch, boundaries, width]`. The final normalization input is a useful last residual boundary; its output is a different representation.

<details><summary>Help — keep the coordinate convention explicit</summary>

Build one hook closure per boundary index, or every hook may write to the same key. Remove every handle in `finally`, even if position validation fails halfway through the model.

</details>

<details><summary>Expected output</summary>

Captured vectors equal the selected baseline vectors, model logits remain unchanged, and no hook survives either success or a bad padding position.

</details>

<details><summary>Solution</summary>

```python
def capture_residual_positions(model, boundaries, encoded, positions):
    """Capture one semantic position at each supplied native residual boundary."""
    saved = {}
    handles = []

    def make_hook(index):
        def hook(module, args):
            saved[index] = (
                gather_token_positions(args[0], positions, encoded["attention_mask"])
                .detach()
                .float()
                .cpu()
            )

        return hook

    try:
        for index, boundary in enumerate(boundaries):
            handles.append(boundary.register_forward_pre_hook(make_hook(index)))
        logits = model(**encoded, use_cache=False).logits
        if set(saved) != set(range(len(boundaries))):
            raise ValueError("Not every requested residual boundary ran.")
        return logits, t.stack([saved[i] for i in range(len(boundaries))], dim=1)
    finally:
        for handle in handles:
            handle.remove()
```

</details>

<!-- ARENA:cell code -->
```python
def capture_residual_positions(model, boundaries, encoded, positions):
    """Capture one semantic position at each supplied native residual boundary."""
    raise NotImplementedError()


tests.test_native_capture_preserves_baseline_and_cleans_up_on_failure(SimpleNamespace(**globals()))
print("All tests passed!")
```

<!-- ARENA:cell markdown -->
# 4️⃣ Calendar geometry in Mistral

## Start by reading the model's actual answers

Our real task is modular arithmetic in ordinary language: `Two days from Monday
is` should predict Wednesday. We use the public Mistral-7B-v0.1 checkpoint from
the circular-feature paper. It is not perfect at this task: our development run
gets 32/49 weekday and 128/144 month answers right when choosing among calendar
words. The paper reports 31/49 and 125/144. Keep incorrect examples; they are
part of the behavior we are trying to explain.

There are two different accuracy questions. Which calendar word has the
highest logit? Which token wins over the **whole vocabulary**? If the model
prefers `the` over every month, a correct calendar-only answer must not conceal
that failure. We'll record both and show actual tokens in the result galleries.

The prompt also determines where the starting weekday/month token lives.
Locate that word using tokenizer offsets and its character span. Hard-coded
indices fail as soon as a duration, prefix or padding length changes.

<details><summary>Why can full-vocabulary accuracy be slightly higher?</summary>

BF16 can give two calendar words exactly the same logit. A calendar-only argmax
breaks ties in our weekday/month order; the full-vocabulary argmax breaks ties
in tokenizer-ID order. The latter can therefore choose the correct tied word
when the former chooses another one. In this run, all 178 retained rows with
that apparent reversal are exact calendar-logit ties. Neither column is a
probability, and small tie-breaking differences are not a causal effect.

</details>

<!-- ARENA:cell markdown -->
### Exercise 11 — Align the word and score both vocabularies

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵🔵🔵
> You should spend 15 minutes on this exercise.
> ```

Implement whole-word span alignment and paired logit scoring. Check that each
semantic span overlaps exactly one non-padding token and that the token covers
the entire word. Keep full-vocabulary and calendar-only winners separate.

<details><summary>Help — don't lose the original vocabulary</summary>

For scoring, find the full argmax before restricting logits to calendar tokens.
Index the original and requested-answer logits using one token ID per row.
For token spans, a token overlaps a word when its left boundary is before the
word's end and its right boundary is after the word's beginning.

</details>

<details><summary>Expected output</summary>

Two differently padded prompts select positions `[2, 1]`. In the score test,
both rows are correct among calendar words, but only one wins over the full
vocabulary. Both requested-minus-original logit margins equal 1.

</details>

<details><summary>Solution</summary>

```python
def positions_from_spans(offset_mapping: t.Tensor, attention_mask: t.Tensor, spans: t.Tensor):
    """Locate a whole, single-token semantic span using actual tokenizer offsets."""
    left, right = offset_mapping.unbind(-1)
    overlaps = (left < spans[:, 1, None]) & (right > spans[:, 0, None]) & attention_mask.bool()
    if not overlaps.sum(-1).eq(1).all():
        raise ValueError("Each semantic word must overlap exactly one real token.")
    positions = overlaps.long().argmax(-1)
    row = t.arange(len(spans), device=spans.device)
    if ((left[row, positions] > spans[:, 0]) | (right[row, positions] < spans[:, 1])).any():
        raise ValueError("The selected token does not contain the whole semantic word.")
    return positions


def score_calendar_logits(
    logits: t.Tensor, original_ids: t.Tensor, desired_ids: t.Tensor, calendar_ids: t.Tensor
):
    """Keep full-vocabulary correctness separate from the restricted-choice task."""
    if (
        logits.ndim != 2
        or original_ids.shape != (len(logits),)
        or desired_ids.shape != (len(logits),)
    ):
        raise ValueError("Every model row needs an original and a desired answer token.")
    if not t.isfinite(logits).all():
        raise ValueError("Nonfinite model logits cannot support an answer claim.")
    row = t.arange(len(logits), device=logits.device)
    top = logits.argmax(-1)
    restricted = calendar_ids[logits[:, calendar_ids].argmax(-1)]
    return {
        "top_token_ids": top,
        "restricted_token_ids": restricted,
        "original_full_correct": top.eq(original_ids),
        "desired_full_correct": top.eq(desired_ids),
        "original_restricted_correct": restricted.eq(original_ids),
        "desired_restricted_correct": restricted.eq(desired_ids),
        "desired_minus_original_logit": logits[row, desired_ids] - logits[row, original_ids],
    }
```

</details>

<!-- ARENA:cell code -->
```python
def positions_from_spans(offset_mapping: t.Tensor, attention_mask: t.Tensor, spans: t.Tensor):
    """Locate a whole, single-token semantic span using actual tokenizer offsets."""
    raise NotImplementedError()


def score_calendar_logits(
    logits: t.Tensor, original_ids: t.Tensor, desired_ids: t.Tensor, calendar_ids: t.Tensor
):
    """Keep full-vocabulary correctness separate from the restricted-choice task."""
    raise NotImplementedError()


tests.test_token_offsets_do_not_assume_identical_prompt_lengths(SimpleNamespace(**globals()))
tests.test_restricted_calendar_success_does_not_hide_full_vocabulary_failure(
    SimpleNamespace(**globals())
)
print("All tests passed!")
```

<!-- ARENA:cell markdown -->
## The frozen experiment

Training prompts fit the five-component PCA and the circular probes. Different
durations in the same wording form development data. Development comparisons
across every residual boundary chose layer input 16 for weekdays and 10 for
months. These choices are fixed before evaluating the three new prompt
templates. All weekday/month identities occur in training; a separate
leave-one-class-out probe checks what happens when one identity is excluded.

The full experiment evaluates 579 new prompts and all 5,634 counterfactual
start-word pairs. Every original/counterfactual pair has the same duration and
prompt template. It compares pure coordinate edits, both mean-ablation
conventions, complete donors and five-component donors. It also retains
no-op and mean-only controls, plus 32 independent random planes, equal-norm
random edits and consistently permuted class-order probes for a fixed donor
shift. A class permutation changes both the training targets and the requested
donor coordinates; it does not deliberately mislabel the evaluation answer.

Read the configuration before running. Predict which operation will perform
best and which will preserve unrelated information. The following scaffold is
provided so that you can concentrate on the methods you just implemented.
Its fitting, interventions, nulls and complete scoring loops are visible here;
change them when designing a new experiment, and record the new protocol.

<!-- ARENA:cell code -->
```python
import gzip
import hashlib
import json
from chapter11_representation_geometry.exercises.part1_pca_svd_geometry_controls.geometry_data import (
    NAMES,
    calendar_rows,
    paired_donor_indices,
    load_protocol,
)

protocol = load_protocol()
RUN_REAL_MODELS = os.environ.get("ARENA_GEOMETRY_SCOPE", "full") == "full"
print(
    "Training/development-selected layers:",
    {family: entry["layer_input"] for family, entry in protocol["selected"].items()},
)
print(
    "Final prompts:",
    protocol["final_base_rows"],
    "| counterfactual pairs:",
    protocol["all_donor_pairs"],
)
print("Random-control draws:", protocol["null_draws"])
```

<!-- ARENA:cell markdown -->
### Capture and score ordered batches

`capture_calendar_rows` reads the starting word at each block input and at the
final normalization input. `measure_calendar_replacements` uses the same batch
order and padding choices. The no-op check requires exact agreement with the
baseline, so a changed input position or a lingering hook cannot quietly pass.
The saved records include actual calendar logits and the full-vocabulary winner;
summary plots can be recomputed without choosing which failures to display.

<!-- ARENA:cell code -->
```python
def encode_calendar_batch(tokenizer, rows, device):
    encoded = tokenizer(
        [row["prompt"] for row in rows],
        padding=True,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offsets = encoded.pop("offset_mapping")
    positions = positions_from_spans(
        offsets, encoded["attention_mask"], t.tensor([row["source_span"] for row in rows])
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    if not encoded["attention_mask"][:, -1].bool().all():
        raise ValueError(
            "Use left padding so the last model position is an actual prediction position."
        )
    return encoded, positions.to(device)
```

<!-- ARENA:cell code -->
```python
@t.inference_mode()
def capture_calendar_rows(model, tokenizer, rows, calendar_ids, batch_size=32):
    """All residual boundaries, reduced answer logits and full-vocabulary winners."""
    device = next(model.parameters()).device
    boundaries = [*model.model.layers, model.model.norm]
    hidden, logits, top = [], [], []
    for start in range(0, len(rows), batch_size):
        encoded, positions = encode_calendar_batch(
            tokenizer, rows[start : start + batch_size], device
        )
        out, acts = capture_residual_positions(model, boundaries, encoded, positions)
        last = out[:, -1].float()
        hidden.append(acts)
        logits.append(last[:, calendar_ids].cpu())
        top.append(last.argmax(-1).cpu())
    return {
        "activations": t.cat(hidden),
        "calendar_logits": t.cat(logits),
        "top_token_ids": t.cat(top),
    }
```

<!-- ARENA:cell code -->
```python
@t.inference_mode()
def measure_calendar_replacements(
    model, tokenizer, rows, calendar_ids, layer, replacements, batch_size=32
):
    """Same ordered batches as capture; no outcome-dependent example filtering."""
    device = next(model.parameters()).device
    boundary = model.model.layers[layer] if layer < len(model.model.layers) else model.model.norm
    logits, top = [], []
    for start in range(0, len(rows), batch_size):
        encoded, positions = encode_calendar_batch(
            tokenizer, rows[start : start + batch_size], device
        )
        out = forward_with_residual_replacement(
            model, boundary, encoded, positions, replacements[start : start + batch_size]
        )
        last = out[:, -1].float()
        logits.append(last[:, calendar_ids].cpu())
        top.append(last.argmax(-1).cpu())
    return {"calendar_logits": t.cat(logits), "top_token_ids": t.cat(top)}
```

<!-- ARENA:cell code -->
```python
def calendar_outcome_rows(rows, measured, calendar_ids, tokenizer, shift, condition, draw=None):
    """Retain all calendar logits and full-vocabulary winners for independent scoring."""
    count = len(calendar_ids)
    logits = measured["calendar_logits"]
    top = measured["top_token_ids"]
    if logits.shape != (len(rows), count) or top.shape != (len(rows),):
        raise ValueError("Measured predictions must contain every requested row exactly once.")
    original = t.tensor([row["answer"] for row in rows])
    desired = (original + shift) % count
    # The pure tensor scorer handles both vocabularies. The compact archive retains
    # full-vocabulary winners separately, so a calendar-only argmax cannot hide them.
    scores = score_calendar_logits(logits, original, desired, t.arange(count))
    result = []
    for i, row in enumerate(rows):
        result.append(
            {
                "row_id": row["id"],
                "family": row["family"],
                "template_index": row["template_index"],
                "start": row["start"],
                "offset": row["offset"],
                "shift": shift,
                "condition": condition,
                "draw": draw,
                "original_answer": int(original[i]),
                "desired_answer": int(desired[i]),
                "restricted_answer": int(scores["restricted_token_ids"][i]),
                "top_token_id": int(top[i]),
                "top_token_text": tokenizer.decode(int(top[i])),
                "calendar_logits": logits[i].tolist(),
                "desired_minus_original_logit": float(scores["desired_minus_original_logit"][i]),
                "original_full_correct": bool(top[i] == calendar_ids[original[i]]),
                "desired_full_correct": bool(top[i] == calendar_ids[desired[i]]),
            }
        )
    return result
```

<!-- ARENA:cell code -->
```python
def fit_calendar_probe(activations, labels, period, components=5):
    mean, basis, variance = fit_pca(activations, components)
    coordinates = circular_coordinates(labels.to(activations.dtype), period)
    zmean, weights, bias = fit_ridge_coordinates((activations - mean) @ basis, coordinates)
    return {
        "mean": mean,
        "basis": basis,
        "weights": weights,
        "bias": bias - zmean @ weights,
        "variance": variance,
    }
```

<!-- ARENA:cell markdown -->
### Run every planned comparison

Follow the loop in order: capture the three splits, fit only on training rows,
evaluate the fixed layer and the complete layer sweep, perform every donor
swap, and draw each null independently. Shared baseline/mean-only predictions
are scored against every donor answer without rerunning an identical forward
pass. The progress files support interruption diagnosis; the teaching results
are the figures and actual answers below.

<!-- ARENA:cell markdown -->
### Recompute the development sweep

Every residual boundary is compared on the development durations before the new templates run. The preregistered layer remains fixed even if rounding changes the ordering of near-tied candidates. Plot all of these outcomes so that late decodability can be compared with late causal influence.

<!-- ARENA:cell code -->
```python
def development_calendar_sweep(
    model, tokenizer, train_rows, dev_rows, train_capture, dev_capture, calendar_ids, batch_size=32
):
    """Recompute every development boundary; the published final choice stays frozen."""
    count = len(calendar_ids)
    train_labels = t.tensor([row["start"] for row in train_rows])
    dev_labels = t.tensor([row["start"] for row in dev_rows])
    targets = circular_coordinates(((dev_labels + 2) % count).float(), count)
    donors = t.tensor(paired_donor_indices(dev_rows, 2, count))
    original = t.tensor([row["answer"] for row in dev_rows])
    desired = (original + 2) % count
    indices = t.arange(len(dev_rows))
    baseline = dev_capture["calendar_logits"]
    baseline_margin = baseline[indices, desired] - baseline[indices, original]
    curves = []
    outcomes = []
    for boundary in range(train_capture["activations"].shape[1]):
        probe = fit_calendar_probe(
            train_capture["activations"][:, boundary], train_labels, count, 5
        )
        mean, basis, w, bias = (probe[key] for key in ("mean", "basis", "weights", "bias"))
        receiver = dev_capture["activations"][:, boundary]
        raw = author_raw_circle_replacement(mean, basis, w, targets)
        random = t.randn(receiver.shape, generator=t.Generator().manual_seed(1101 + boundary))
        random = (
            random / random.norm(dim=-1, keepdim=True) * (raw - receiver).norm(dim=-1, keepdim=True)
        )
        changes = {
            "pure_coordinate": replace_coordinates(receiver, mean, basis, w, bias, targets),
            "centered_mean_ablated": mean_ablate_coordinates(mean, basis, w, bias, targets),
            "author_raw_mean_ablated": raw,
            "full_donor": receiver[donors],
            "mean_only": mean.expand_as(receiver),
            "equal_norm_random_edit": receiver + random,
        }
        point = {"boundary": boundary, "conditions": {}}
        for name, value in changes.items():
            measured = measure_calendar_replacements(
                model, tokenizer, dev_rows, calendar_ids, boundary, value, batch_size
            )
            saved = calendar_outcome_rows(dev_rows, measured, calendar_ids, tokenizer, 2, name)
            for row in saved:
                row["layer_input"] = boundary
            outcomes.extend(saved)
            logits = measured["calendar_logits"]
            margin = logits[indices, desired] - logits[indices, original]
            point["conditions"][name] = {
                "desired_choice_accuracy": float(logits.argmax(-1).eq(desired).float().mean()),
                "mean_logit_effect": float((margin - baseline_margin).mean()),
                "desired_full_accuracy": sum(row["desired_full_correct"] for row in saved)
                / len(saved),
            }
        curves.append(point)
        if boundary % 8 == 0:
            print(dev_rows[0]["family"], "development boundary", boundary, "complete", flush=True)
    return curves, outcomes
```

<!-- ARENA:cell code -->
```python
def run_calendar_study(model, tokenizer, protocol, output_directory):
    """Execute the frozen train/development/final experiment, keeping every control."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=False)
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    batch_size = protocol["batch_size"]
    result = {
        "protocol": protocol,
        "tasks": {},
        "outcomes": [],
        "model_forward_rows": 0,
        "nulls": [],
    }
    tensors = {}

    def checkpoint():
        payload = json.dumps(result, separators=(",", ":")).encode()
        (output / "measured_study.json.gz").write_bytes(gzip.compress(payload, mtime=0))

    for family_index, (family, names) in enumerate(NAMES.items()):
        count = len(names)
        tokens = [tokenizer.encode(name, add_special_tokens=False) for name in names]
        if not all(len(ids) == 1 for ids in tokens):
            raise ValueError("Every answer must be one actual model token.")
        choice_ids = t.tensor([ids[0] for ids in tokens])
        if len(choice_ids.unique()) != count:
            raise ValueError("Calendar classes must have distinct answer tokens.")
        rows = {
            split: calendar_rows(family, split, protocol)
            for split in ("train", "development", "final")
        }
        prompt_inventory = [row["prompt"] for split in rows.values() for row in split]
        if len(set(prompt_inventory)) != len(prompt_inventory):
            raise ValueError("Prompt splits overlap.")
        captured = {
            split: capture_calendar_rows(model, tokenizer, rows[split], choice_ids, batch_size)
            for split in ("train", "development")
        }
        development_curves, development_outcomes = development_calendar_sweep(
            model,
            tokenizer,
            rows["train"],
            rows["development"],
            captured["train"],
            captured["development"],
            choice_ids,
            batch_size,
        )
        # These fresh curves remain visible; they do not change the already-frozen layer choice.
        captured["final"] = capture_calendar_rows(
            model, tokenizer, rows["final"], choice_ids, batch_size
        )
        result["outcomes"].extend(development_outcomes)
        result["model_forward_rows"] += sum(len(items) for items in rows.values()) + len(
            development_outcomes
        )
        train_labels = t.tensor([r["start"] for r in rows["train"]])
        final_labels = t.tensor([r["start"] for r in rows["final"]])
        layer = protocol["selected"][family]["layer_input"]
        probe = fit_calendar_probe(
            captured["train"]["activations"][:, layer], train_labels, count, 5
        )
        mean, basis, w, bias = (probe[key] for key in ("mean", "basis", "weights", "bias"))
        final = captured["final"]["activations"][:, layer]
        final_targets = circular_coordinates(final_labels.float(), count)
        prediction = (final - mean) @ basis @ w + bias
        curves = []
        permutations = t.stack(
            [
                t.randperm(
                    count,
                    generator=t.Generator().manual_seed(
                        protocol["circular_order_null_seed"] + family_index * 1000 + draw
                    ),
                )
                for draw in range(protocol["circular_order_null_draws"])
            ]
        )
        for boundary in range(captured["train"]["activations"].shape[1]):
            train_acts = captured["train"]["activations"][:, boundary]
            other = fit_calendar_probe(train_acts, train_labels, count, 5)
            pred = (captured["final"]["activations"][:, boundary] - other["mean"]) @ other[
                "basis"
            ] @ other["weights"] + other["bias"]
            centroids = t.stack([train_acts[train_labels.eq(i)].mean(0) for i in range(count)])
            xy = (centroids - other["mean"]) @ other["basis"][:, :2]
            actual, null, p = cyclic_permutation_test(xy, permutations)
            curves.append(
                {
                    "boundary": boundary,
                    "final_coordinate_r2": float(coordinate_r2(final_targets, pred)),
                    "final_angular_accuracy": float(
                        angular_predictions(pred, count).eq(final_labels).float().mean()
                    ),
                    "training_pca2_order_agreement": float(actual),
                    "order_null": null.tolist(),
                    "order_p": float(p),
                    "training_variance": other["variance"].tolist(),
                }
            )
        leave_out = []
        for heldout_class in range(count):
            keep = train_labels.ne(heldout_class)
            target_rows = final_labels.eq(heldout_class)
            other = fit_calendar_probe(
                captured["train"]["activations"][keep, layer], train_labels[keep], count, 5
            )
            pred = (final[target_rows] - other["mean"]) @ other["basis"] @ other["weights"] + other[
                "bias"
            ]
            leave_out.append(
                {
                    "class": heldout_class,
                    "rows": int(target_rows.sum()),
                    "predicted_coordinates": pred.tolist(),
                    "angular_correct": int(
                        angular_predictions(pred, count).eq(heldout_class).sum()
                    ),
                    "mean_coordinate_squared_error": float(
                        (pred - final_targets[target_rows]).square().sum(-1).mean()
                    ),
                }
            )
        task = {
            "names": list(names),
            "calendar_token_ids": choice_ids.tolist(),
            "rows": rows,
            "layer_input": layer,
            "layer_sweep": curves,
            "leave_one_class_out": leave_out,
            "final_coordinate_r2": float(coordinate_r2(final_targets, prediction)),
            "final_coordinate_predictions": prediction.tolist(),
            "final_angular_accuracy": float(
                angular_predictions(prediction, count).eq(final_labels).float().mean()
            ),
            "training_pca_variance": probe["variance"].tolist(),
            "development_interventions": development_curves,
        }
        result["tasks"][family] = task
        shared = {"baseline": captured["final"]}
        for name, value in [("no_op", final), ("mean_only", mean.expand_as(final))]:
            shared[name] = measure_calendar_replacements(
                model, tokenizer, rows["final"], choice_ids, layer, value, batch_size
            )
            result["model_forward_rows"] += len(final)
        if not t.equal(
            shared["baseline"]["calendar_logits"], shared["no_op"]["calendar_logits"]
        ) or not t.equal(shared["baseline"]["top_token_ids"], shared["no_op"]["top_token_ids"]):
            raise ValueError("No-op residual replacement changed the actual baseline.")
        task["no_op_max_logit_error"] = 0.0
        for split in ("train", "development"):
            result["outcomes"].extend(
                calendar_outcome_rows(
                    rows[split], captured[split], choice_ids, tokenizer, 2, split + "_baseline"
                )
            )
        primary = None
        for shift in range(1, count):
            donors = t.tensor(paired_donor_indices(rows["final"], shift, count))
            target = circular_coordinates(((final_labels + shift) % count).float(), count)
            replacements = {
                "pure_coordinate": replace_coordinates(final, mean, basis, w, bias, target),
                "centered_mean_ablated": mean_ablate_coordinates(mean, basis, w, bias, target),
                "author_raw_mean_ablated": author_raw_circle_replacement(mean, basis, w, target),
                "full_donor": final[donors],
                "pca5_donor": replace_subspace(final, final[donors], basis),
            }
            for name, value in replacements.items():
                measured = measure_calendar_replacements(
                    model, tokenizer, rows["final"], choice_ids, layer, value, batch_size
                )
                result["model_forward_rows"] += len(final)
                result["outcomes"].extend(
                    calendar_outcome_rows(
                        rows["final"], measured, choice_ids, tokenizer, shift, name
                    )
                )
            for name, measured in shared.items():
                result["outcomes"].extend(
                    calendar_outcome_rows(
                        rows["final"], measured, choice_ids, tokenizer, shift, name
                    )
                )
            if shift == protocol["primary_control_donor_shift"]:
                primary = {
                    "donors": donors,
                    "target": target,
                    "delta_norm": (replacements["author_raw_mean_ablated"] - final).norm(
                        dim=-1, keepdim=True
                    ),
                }
            checkpoint()
            print(family, "all-donor shift", shift, "complete", flush=True)
        shift = protocol["primary_control_donor_shift"]
        for draw in range(protocol["null_draws"]):
            seed = protocol["null_seed"] + family_index * 10000 + draw
            random_plane = t.linalg.qr(
                t.randn(final.shape[1], 2, generator=t.Generator().manual_seed(seed)),
                mode="reduced",
            ).Q
            noise = t.randn(final.shape, generator=t.Generator().manual_seed(seed + 1000))
            noise = noise / noise.norm(dim=-1, keepdim=True) * primary["delta_norm"]
            permutation = t.randperm(count, generator=t.Generator().manual_seed(seed + 2000))
            perm_targets = circular_coordinates(permutation[train_labels].float(), count)
            zmean, perm_w, perm_b = fit_ridge_coordinates(
                (captured["train"]["activations"][:, layer] - mean) @ basis, perm_targets
            )
            desired_perm = circular_coordinates(
                permutation[(final_labels + shift) % count].float(), count
            )
            changes = {
                "same_rank_random_donor_plane": replace_subspace(
                    final, final[primary["donors"]], random_plane
                ),
                "equal_norm_random_edit": final + noise,
                "class_order_permuted_circular_probe": author_raw_circle_replacement(
                    mean, basis, perm_w, desired_perm
                ),
            }
            result["nulls"].append(
                {
                    "family": family,
                    "draw": draw,
                    "seed": seed,
                    "class_order_permutation": permutation.tolist(),
                    "random_plane_sha256": hashlib.sha256(
                        random_plane.numpy().tobytes()
                    ).hexdigest(),
                }
            )
            for name, value in changes.items():
                measured = measure_calendar_replacements(
                    model, tokenizer, rows["final"], choice_ids, layer, value, batch_size
                )
                result["model_forward_rows"] += len(final)
                result["outcomes"].extend(
                    calendar_outcome_rows(
                        rows["final"], measured, choice_ids, tokenizer, shift, name, draw
                    )
                )
            if draw % 8 == 7:
                checkpoint()
                print(family, "null draw", draw + 1, "complete", flush=True)
        tensors[family] = {
            "captured": captured,
            "probe": probe,
            "rows": rows,
            "calendar_token_ids": choice_ids,
        }
        t.save(tensors, output / "actual_activations_and_probes.pt")
        checkpoint()
    result["complete"] = True
    result["retained_outcomes"] = len(result["outcomes"])
    result["final_base_rows"] = sum(len(task["rows"]["final"]) for task in result["tasks"].values())
    result["all_donor_pairs"] = sum(
        len(task["rows"]["final"]) * (len(task["names"]) - 1) for task in result["tasks"].values()
    )
    if (
        result["final_base_rows"] != protocol["final_base_rows"]
        or result["all_donor_pairs"] != protocol["all_donor_pairs"]
    ):
        raise ValueError("The frozen data inventory did not execute completely.")
    checkpoint()
    return result, tensors
```

<!-- ARENA:cell markdown -->
## Load the checkpoint and execute

The checkpoint uses about 13.6 GiB on the tested GPU in BF16. Its first download
is about 14.5 GB; subsequent runs reuse the pinned cache. The model is loaded
in evaluation mode, and temporary intervention hooks are removed after every
forward pass. The full study is longer than the numerical exercises; its
progress output shows completed donor shifts and null draws.

The optional `ARENA_GEOMETRY_MODEL_CACHE` environment variable chooses a local
Hugging Face cache directory. It changes where files live, not their pinned
revision or the model's precision.

<!-- ARENA:cell code -->
```python
if RUN_REAL_MODELS:
    import time
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_cache = os.environ.get("ARENA_GEOMETRY_MODEL_CACHE")
    tokenizer = AutoTokenizer.from_pretrained(
        protocol["model_id"], revision=protocol["revision"], cache_dir=model_cache
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = (
        AutoModelForCausalLM.from_pretrained(
            protocol["model_id"],
            revision=protocol["revision"],
            dtype=t.bfloat16,
            attn_implementation=protocol["attention"],
            cache_dir=model_cache,
        )
        .to("cuda")
        .eval()
    )
    t.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    study_result, study_tensors = run_calendar_study(
        model, tokenizer, protocol, assets_dir.parent / "experiments"
    )
    print(f"Actual study runtime: {time.perf_counter() - started:.1f} s")
    print("Retained outcomes:", study_result["retained_outcomes"])
else:
    print(
        "Numerical foundation mode: pretrained-model fitting, capture and interventions are skipped."
    )
```

<!-- ARENA:cell markdown -->
## First read the unmodified model

Before interpreting an intervention, check the original answers. The paper
reports 31/49 correct weekdays and 125/144 correct months for its Mistral task.
A pretrained model can represent a weekday accurately and still get arithmetic
wrong. Our new templates are a further test; they need not have the paper's
baseline accuracy.

The table below uses each original prompt exactly once. It separates the best
calendar word from the best token in the entire vocabulary. A completion such
as a newline or punctuation is not a correct weekday just because the correct
weekday has the largest *calendar-only* logit.

<details><summary>Help — why keep wrong source and donor answers?</summary>

Dropping those examples after an intervention can make an effect look more
reliable than it is. We retain every example in the main comparison. A useful
follow-up is to condition on a correct unmodified source and donor, but that
answers a narrower question and needs its own denominator.

</details>

<!-- ARENA:cell code -->
```python
if RUN_REAL_MODELS:
    import pandas as pd
    from IPython.display import display
    from chapter11_representation_geometry.exercises.part1_pca_svd_geometry_controls import (
        geometry_plotting as plots,
    )

    outcome_frame = pd.DataFrame(study_result["outcomes"])
    outcome_frame["desired_correct"] = outcome_frame.restricted_answer.eq(
        outcome_frame.desired_answer
    )
    outcome_frame["original_correct"] = outcome_frame.restricted_answer.eq(
        outcome_frame.original_answer
    )
    final_frame = outcome_frame[outcome_frame.row_id.str.startswith("final/")].copy()
    baseline_frame = outcome_frame[
        outcome_frame.condition.isin(["train_baseline", "development_baseline", "baseline"])
        & outcome_frame["shift"].eq(protocol["primary_control_donor_shift"])
    ].copy()
    baseline_frame["split"] = baseline_frame.row_id.str.split("/").str[0]
    display(
        baseline_frame.groupby(["family", "split"])
        .agg(
            prompts=("row_id", "size"),
            calendar_accuracy=("original_correct", "mean"),
            full_vocabulary_accuracy=("original_full_correct", "mean"),
        )
        .round(3)
    )
```

<!-- ARENA:cell markdown -->
<details><summary>Expected result — the original model is imperfect</summary>

On the fixed new templates, restricted calendar accuracy is 64.6% for weekdays
and 79.2% for months. Keep these failures in the causal comparison. BF16 logits
can tie: full-vocabulary and calendar-only argmax break ties in different token
orders, so their accuracies can differ slightly even when the top word is a
calendar word. Both actual choices are retained.

</details>

## Does the geometry generalize to a new prompt?

These axes are fitted on training prompts. Crosses are new templates projected
with that same transformation. The left view is unsupervised PCA; color is
added after fitting. The right view is explicitly supervised to predict cosine
and sine. Its success measures linear decodability, not spontaneous discovery
of a circle.

The layer sweep adds a chronological-order control. It permutes the mapping
between class centroids and positions on the calendar while leaving every
activation and token identity intact. Its null band is pointwise, so treat a
single layer above it as an exploratory finding rather than a corrected
search over all layers.

Predict which layer will give the strongest intervention. Is it necessarily
the layer with the best coordinate $R^2$? Compare with the rightmost development
panel after running the cell.

<!-- ARENA:cell code -->
```python
if RUN_REAL_MODELS:
    projections = {}
    for family, task in study_tensors.items():
        layer = protocol["selected"][family]["layer_input"]
        probe = task["probe"]
        train = task["captured"]["train"]["activations"][:, layer]
        final = task["captured"]["final"]["activations"][:, layer]
        train_z = (train - probe["mean"]) @ probe["basis"]
        final_z = (final - probe["mean"]) @ probe["basis"]
        projections[family] = {
            "names": NAMES[family],
            "train_pca": train_z[:, :2].numpy(),
            "final_pca": final_z[:, :2].numpy(),
            "train_probe": (train_z @ probe["weights"] + probe["bias"]).numpy(),
            "final_probe": (final_z @ probe["weights"] + probe["bias"]).numpy(),
            "train_labels": [r["start"] for r in task["rows"]["train"]],
            "final_labels": [r["start"] for r in task["rows"]["final"]],
        }
    plots.plot_calendar_projections(projections, assets_dir)
    plots.plot_calendar_layers(study_result["tasks"], assets_dir)
    display(
        pd.DataFrame(
            [
                {
                    "family": family,
                    "new_template_R2": task["final_coordinate_r2"],
                    "new_template_angle_accuracy": task["final_angular_accuracy"],
                    "unseen_class_angle_accuracy": sum(
                        row["angular_correct"] for row in task["leave_one_class_out"]
                    )
                    / sum(row["rows"] for row in task["leave_one_class_out"]),
                }
                for family, task in study_result["tasks"].items()
            ]
        ).round(3)
    )
```

<!-- ARENA:cell markdown -->
<details><summary>Expected result — strong decoding has limits</summary>

At the fixed layers, coordinate R² is 0.836 for weekdays and 0.972 for months.
New-template angular accuracy is 83.7% and 97.2%. When a whole calendar identity
is excluded from fitting, accuracy falls to 47.6% and 82.2%. A split containing
all identities is therefore a materially easier test.

The fresh development curves can differ slightly from the archived selection
run because its batch shapes differ under BF16 inference. We retain the
previously selected layer inputs, 16 and 10; the final results never select a
replacement layer. The late-layer panels show a much larger effect: high
coordinate decoding can remain after interventions stop changing the answer.

</details>

## Signature result — can the coordinate change the answer?

For the main controls we change the start by two calendar positions, retaining
the offset and prompt template. The horizontal black marks summarize the null
draws; every dot remains visible. The margin is the requested answer's logit
minus the original answer's logit, so a large positive value need not mean the
requested answer beats *every* other token.

Compare these operations carefully:

- **Pure coordinate edit** preserves everything outside the fitted plane.
- **Centered plane + mean** additionally removes other deviations from the training mean.
- **Author plane + mean** follows the author's raw-coordinate QR replacement.
- **Full donor residual** transplants the actual counterfactual activation.
- **Five-PC donor** transplants the five-dimensional PCA subspace.
- **Mean only** removes the source-specific state without inserting a new coordinate.

The mean-only bar matters when interpreting the two mean-ablation methods:
a large effect relative to the original prompt can partly come from erasing
information. The random edits test whether an equally large arbitrary change
is enough. The permuted-order probe uses its label permutation consistently
in training *and* intervention; it does not deliberately request the wrong label.

<details><summary>Interpretation — what would convince you?</summary>

Look for a requested-answer improvement above the unmodified and mean-only
conditions, with null draws lower, and similar effects across counterfactuals.
Then read the actual full-vocabulary answers. A surviving effect supports a
causal role for the manipulated state under this intervention. It does not
show that two coordinates completely implement the model's arithmetic, or
that the effect transfers to arbitrary language tasks.

</details>

<!-- ARENA:cell code -->
```python
if RUN_REAL_MODELS:
    primary = final_frame[final_frame["shift"].eq(protocol["primary_control_donor_shift"])]
    aggregate = dict(
        desired_accuracy=("desired_correct", "mean"),
        desired_full_accuracy=("desired_full_correct", "mean"),
        mean_margin=("desired_minus_original_logit", "mean"),
        prompts=("row_id", "size"),
    )
    main_summary = (
        primary[primary.draw.isna()].groupby(["family", "condition"]).agg(**aggregate).reset_index()
    )
    null_summary = (
        primary[primary.draw.notna()]
        .groupby(["family", "condition", "draw"])
        .agg(**aggregate)
        .reset_index()
    )
    plots.plot_calendar_final_controls(main_summary, null_summary, assets_dir)
    plots.plot_calendar_all_donors(
        final_frame[final_frame.draw.isna()], study_result["tasks"], assets_dir
    )
    plots.plot_calendar_examples(final_frame, study_result["tasks"], assets_dir)
    display(
        main_summary[
            ["family", "condition", "prompts", "desired_accuracy", "desired_full_accuracy"]
        ].round(3)
    )
```

<!-- ARENA:cell markdown -->
<details><summary>Expected result — compare the intervention and its controls</summary>

For the fixed +2 shift, the author mean-ablation gives 45.6% weekday and 51.4%
month accuracy, compared with 22.4% and 7.2% for pure coordinate edits. Full
donors give 62.6% and 79.2%; mean-only replacements give 12.9% and 8.8%.
Mean ablation changes the scientific interpretation substantially.

Random planes and equal-norm random edits stay near the unchanged shifted-answer
baseline. Permuted calendar order is a stronger control: its mean is 29.7% for
weekdays and 19.1% for months, and its best weekday draw reaches 44.2%. Thus the
weekday ordering effect is only slightly above the strongest sampled permutation.
A fitted arbitrary two-dimensional code can still carry useful identity information.

The first weekday example makes another limitation concrete. The requested
answer is Thursday, but both circle edits produce Friday, also the mean-only
winner. The full donor produces Thursday. Read this failure beside the aggregate
curves; a change in the answer is not automatically the desired causal change.

</details>

# 5️⃣ Does a nonlinear view preserve useful structure?

PCA optimizes squared reconstruction error. UMAP and t-SNE instead emphasize
neighborhood relationships, with different objectives and distance scales.
Their plotted axes have no calendar units. Islands in either plot can be
created from independent noise; color alone is a poor validation method.

We'll reuse your three-neighbor classifier, trustworthiness, continuity and
Procrustes alignment. None of these measures is a universal score for a good
picture. Trustworthiness penalizes new near neighbors; continuity penalizes
lost original neighbors. A high neighborhood score on noise does not establish
a meaningful feature. Identity classification with shuffled training labels
asks a different question again.

We fit the initial 20-PC projection using training activations only. UMAP fits
training points and transforms new ones. Scikit-learn t-SNE has no analogous
`transform`: it fits the combined, unlabeled training and final points here.
**That t-SNE comparison is transductive.** Do not describe its classifier as an
out-of-sample embedding result.

## Read the sweep, then run it

The provided experiment loop below calls your scoring and alignment functions.
It retains all five seeds, all three settings, and the matched independent-noise
fits. For seed comparisons, the orthogonal alignment is fitted only on training
points and then applied to final points. It removes rotations and reflections,
not differences in scale or nonlinear distortion.

<!-- ARENA:cell code -->
```python
def run_geometry_projection_sweep(study_tensors, protocol, output_directory):
    """Inductive UMAP and explicitly transductive t-SNE, including every null fit."""
    import umap
    from sklearn.manifold import TSNE

    records = []
    references = {}
    for family_index, (family, task) in enumerate(study_tensors.items()):
        layer = protocol["selected"][family]["layer_input"]
        actual_train = task["captured"]["train"]["activations"][:, layer]
        actual_final = task["captured"]["final"]["activations"][:, layer]
        ytrain = t.tensor([row["start"] for row in task["rows"]["train"]])
        yfinal = t.tensor([row["start"] for row in task["rows"]["final"]])
        mean = actual_train.mean(0)
        noise = t.randn(
            len(actual_train) + len(actual_final),
            actual_train.shape[-1],
            generator=t.Generator().manual_seed(3300 + family_index),
        )
        actual = t.cat([actual_train, actual_final]) - mean
        noise = noise / noise.norm(dim=-1, keepdim=True) * actual.norm(dim=-1, keepdim=True)
        for control, train, final in [
            ("real", actual_train, actual_final),
            ("independent_noise", noise[: len(actual_train)], noise[len(actual_train) :]),
        ]:
            center, axes, _ = fit_pca(train, min(20, len(train) - 1))
            train_input = ((train - center) @ axes).numpy()
            final_input = ((final - center) @ axes).numpy()
            high = t.cat([train, final])
            for reducer in ("umap", "tsne"):
                settings = (
                    protocol["umap_settings"]
                    if reducer == "umap"
                    else [{"perplexity": p} for p in protocol["tsne_perplexities"]]
                )
                for setting_index, setting in enumerate(settings):
                    for seed in protocol["reducer_seeds"]:
                        if reducer == "umap":
                            method = umap.UMAP(
                                n_components=2,
                                **setting,
                                metric="euclidean",
                                random_state=seed,
                                transform_seed=seed,
                                n_jobs=1,
                            )
                            train_low = t.tensor(method.fit_transform(train_input), dtype=t.float32)
                            final_low = t.tensor(method.transform(final_input), dtype=t.float32)
                            scope = "inductive: fit training only, transform final"
                        else:
                            joined = t.cat([t.tensor(train_input), t.tensor(final_input)]).numpy()
                            method = TSNE(
                                n_components=2,
                                **setting,
                                init=protocol["tsne_init"],
                                learning_rate="auto",
                                max_iter=protocol["tsne_iterations"],
                                random_state=seed,
                                metric="euclidean",
                                n_jobs=1,
                            )
                            low = t.tensor(method.fit_transform(joined), dtype=t.float32)
                            train_low, final_low = low[: len(train)], low[len(train) :]
                            scope = "transductive: unlabeled final points included in fitting"
                        prediction = knn_predictions(train_low, ytrain, final_low, 3)
                        shuffle = t.randperm(
                            len(ytrain),
                            generator=t.Generator().manual_seed(
                                4400 + family_index * 100 + seed * 10 + setting_index
                            ),
                        )
                        shuffled_prediction = knn_predictions(
                            train_low, ytrain[shuffle], final_low, 3
                        )
                        trust, continuity = neighborhood_scores(
                            high, t.cat([train_low, final_low]), 5
                        )
                        key = (family, control, reducer, setting_index)
                        if key not in references:
                            references[key] = (train_low.clone(), final_low.clone())
                        ref_train, ref_final = references[key]
                        source_mean, target_mean, rotation = fit_procrustes(train_low, ref_train)
                        aligned = (final_low - source_mean) @ rotation + target_mean
                        normalized_error = float(
                            (aligned - ref_final).square().mean().sqrt()
                            / ref_final.std().clamp_min(1e-8)
                        )
                        records.append(
                            {
                                "family": family,
                                "control": control,
                                "reducer": reducer,
                                "scope": scope,
                                "setting": setting,
                                "setting_index": setting_index,
                                "seed": seed,
                                "train_projection": train_low.tolist(),
                                "final_projection": final_low.tolist(),
                                "final_knn_accuracy": float(prediction.eq(yfinal).float().mean()),
                                "permuted_training_labels_accuracy": float(
                                    shuffled_prediction.eq(yfinal).float().mean()
                                ),
                                "trustworthiness": float(trust),
                                "continuity": float(continuity),
                                "procrustes_final_error_relative_to_seed0": normalized_error,
                                "final_predictions": prediction.tolist(),
                                "permuted_training_label_predictions": shuffled_prediction.tolist(),
                            }
                        )
                    print(
                        family, control, reducer, "setting", setting_index, "complete", flush=True
                    )
    path = Path(output_directory) / "projection_study.json.gz"
    path.write_bytes(gzip.compress(json.dumps(records, separators=(",", ":")).encode(), mtime=0))
    return records
```

<!-- ARENA:cell code -->
```python
if RUN_REAL_MODELS:
    projection_records = run_geometry_projection_sweep(
        study_tensors, protocol, assets_dir.parent / "experiments"
    )
    assert len(projection_records) == 120, (
        "Keep all families, controls, reducers, settings and seeds."
    )
    plots.plot_projection_sweep(projection_records, assets_dir)
```

<!-- ARENA:cell markdown -->
<details><summary>Expected result — the two reducers answer different questions</summary>

Month identity survives most real-data settings, while independent noise and
shuffled training labels are near chance. Weekday UMAP transfer varies widely
across settings and seeds. Jointly fitted t-SNE gives much stronger weekday
identity accuracy, but its access to all unlabeled final points changes the
question. The independent-noise UMAP plots can still form curved islands; the
control's classification results expose why those shapes are not calendar evidence.

</details>

## Try It Yourself — change the picture without changing the data

Change `projection_seed` from 0 to 4 and `projection_setting` from 0 to 2.
The cell redraws saved coordinates from the sweep, so it does not fit another
model. Inspect the real and independent-noise views together. Do the most
attractive islands give the best final identity accuracy? Compare the visual
change with the alignment errors above.

<details><summary>Something to try next</summary>

Choose two settings whose pictures look different but whose neighbor scores
are similar. Identify which pairs of points changed their apparent distance.
Then explain why a global distance interpretation is not warranted by a
neighborhood objective. For a fresh run, add a reducer seed before running the
sweep and keep its result even if the picture is less attractive.

</details>

<!-- ARENA:cell code -->
```python
projection_seed = 0
projection_setting = 1
if RUN_REAL_MODELS:
    plots.plot_projection_gallery(
        projection_records,
        study_tensors,
        assets_dir,
        seed=projection_seed,
        setting_index=projection_setting,
    )
```

<!-- ARENA:cell markdown -->
# 6️⃣ Turn the calendar continuously

Our training targets lie on a unit circle. What happens between weekday angles,
at radius zero, or outside the training radius? We use the first final prompt
in each family and sweep the same angle/radius grid for the pure edit, author
mean-ablation, and an equal-norm random direction. The random direction is fixed
across the grid; its magnitude tracks the actual intervention.

The next plot uses the best *calendar* answer as a color, even when another
vocabulary token wins. The original and mean-only full-vocabulary answers are
printed above each panel. At radius zero every requested angle gives the same
coordinate, so each zero-radius row should be constant. At radius one, a clean
calendar computation would advance the requested input by the prompt's offset.
Do not assume this will hold at radius two.

<!-- ARENA:cell code -->
```python
def run_calendar_angle_play(model, tokenizer, study_tensors, protocol, output_directory):
    """A fixed source per family; sweep continuous coordinates with paired controls."""
    records = []
    for family_index, (family, task) in enumerate(study_tensors.items()):
        row = task["rows"]["final"][0]
        count = len(NAMES[family])
        probe = task["probe"]
        mean, basis, w, bias = (probe[key] for key in ("mean", "basis", "weights", "bias"))
        layer = protocol["selected"][family]["layer_input"]
        receiver = task["captured"]["final"]["activations"][0, layer]
        angles = (
            t.arange(count * protocol["angle_play"]["steps_per_class"])
            * 2
            * t.pi
            / (count * protocol["angle_play"]["steps_per_class"])
        )
        radii = t.tensor(protocol["angle_play"]["radii"])
        radius_grid, angle_grid = t.meshgrid(radii, angles, indexing="ij")
        target = radius_grid.flatten()[:, None] * t.stack(
            [angle_grid.flatten().cos(), angle_grid.flatten().sin()], -1
        )
        receivers = receiver.expand(len(target), -1)
        raw = author_raw_circle_replacement(mean, basis, w, target)
        # One fixed direction across the grid: only its matched magnitude changes.
        direction = t.randn(
            receiver.shape, generator=t.Generator().manual_seed(5500 + family_index)
        )
        direction = direction / direction.norm()
        replacements = {
            "pure_coordinate": replace_coordinates(receivers, mean, basis, w, bias, target),
            "author_raw_mean_ablated": raw,
            "equal_norm_random_edit": receivers
            + (raw - receivers).norm(dim=-1, keepdim=True) * direction,
        }
        shared = {}
        for name, value in [("baseline", receiver[None]), ("mean_only", mean[None])]:
            measured = measure_calendar_replacements(
                model,
                tokenizer,
                [row],
                task["calendar_token_ids"],
                layer,
                value,
                protocol["batch_size"],
            )
            shared[name] = {
                "calendar_logits": measured["calendar_logits"][0].tolist(),
                "top_token_id": int(measured["top_token_ids"][0]),
                "top_token_text": tokenizer.decode(int(measured["top_token_ids"][0])),
            }
        for condition, value in replacements.items():
            measured = measure_calendar_replacements(
                model,
                tokenizer,
                [row] * len(target),
                task["calendar_token_ids"],
                layer,
                value,
                protocol["batch_size"],
            )
            records.append(
                {
                    "family": family,
                    "row": row,
                    "condition": condition,
                    "layer_input": layer,
                    "radii": radii.tolist(),
                    "angles": angles.tolist(),
                    "shared": shared,
                    "calendar_logits": measured["calendar_logits"].tolist(),
                    "top_token_ids": measured["top_token_ids"].tolist(),
                    "top_token_text": [
                        tokenizer.decode(int(token)) for token in measured["top_token_ids"]
                    ],
                }
            )
        print(family, "continuous angle/radius play complete", flush=True)
    (Path(output_directory) / "angle_play.json.gz").write_bytes(
        gzip.compress(json.dumps(records, separators=(",", ":")).encode(), mtime=0)
    )
    return records
```

<!-- ARENA:cell code -->
```python
if RUN_REAL_MODELS:
    angle_records = run_calendar_angle_play(
        model, tokenizer, study_tensors, protocol, assets_dir.parent / "experiments"
    )
    plots.plot_angle_play(angle_records, assets_dir)
```

<!-- ARENA:cell markdown -->
## Try It Yourself — inspect one real intervention

Choose a family, a starting class and an offset below. Then choose a requested
shift and radius. This cell performs fresh model calls using your intervention
functions. It displays the actual winning token as well as probabilities
**renormalized over calendar words**. A high bar does not imply a high
full-vocabulary probability.

First predict the desired calendar answer. Compare the pure and mean-ablated
edits with the unchanged model and mean-only control. Try an example for which
the original model is wrong; changing its representation need not repair its
arithmetic. To change a word or template, update the source span too, then let
your alignment checks verify the position.

<!-- ARENA:cell code -->
```python
play_family = "weekday"  # 'weekday' or 'month'
play_start = 0  # Monday / January
play_offset = 1
play_shift = 2
play_radius = 1.0
if RUN_REAL_MODELS:
    task = study_tensors[play_family]
    source_index = next(
        i
        for i, row in enumerate(task["rows"]["final"])
        if row["template_index"] == 0
        and row["start"] == play_start
        and row["offset"] == play_offset
    )
    source = task["rows"]["final"][source_index]
    layer = protocol["selected"][play_family]["layer_input"]
    receiver = task["captured"]["final"]["activations"][source_index : source_index + 1, layer]
    probe = task["probe"]
    mean, basis, weights, bias = (probe[key] for key in ("mean", "basis", "weights", "bias"))
    count = len(NAMES[play_family])
    target = play_radius * circular_coordinates(
        t.tensor([(play_start + play_shift) % count], dtype=t.float32), count
    )
    changes = {
        "baseline": receiver,
        "pure_coordinate": replace_coordinates(receiver, mean, basis, weights, bias, target),
        "author_raw_mean_ablated": author_raw_circle_replacement(mean, basis, weights, target),
        "mean_only": mean[None],
    }
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    play_measurements = []
    for ax, (condition, replacement) in zip(axes, changes.items(), strict=True):
        measured = measure_calendar_replacements(
            model, tokenizer, [source], task["calendar_token_ids"], layer, replacement
        )
        probabilities = measured["calendar_logits"][0].softmax(-1).numpy()
        actual_token = tokenizer.decode(int(measured["top_token_ids"][0]))
        ax.bar(range(count), probabilities, color=plots.COLORS[condition])
        ax.set_xticks(
            range(count), [name[:3] for name in NAMES[play_family]], rotation=45, ha="right"
        )
        ax.set(
            ylim=(0, 1.02),
            ylabel="Calendar-only probability",
            title=f"{plots.LABELS[condition]}\nFull-vocabulary top: {actual_token!r}",
        )
        play_measurements.append(
            {
                "condition": condition,
                "calendar_logits": measured["calendar_logits"][0].tolist(),
                "top_token_id": int(measured["top_token_ids"][0]),
                "top_token_text": actual_token,
            }
        )
    desired = NAMES[play_family][(play_start + play_offset + play_shift) % count]
    fig.suptitle(
        f"{source['prompt']} | requested shift {play_shift:+d}, radius {play_radius:g} | desired answer: {desired}"
    )
    plots.finish(fig, assets_dir, "geometry_mistral_single_play.png")
    (assets_dir.parent / "experiments" / "single_play.json").write_text(
        json.dumps(
            {
                "source": source,
                "shift": play_shift,
                "radius": play_radius,
                "measurements": play_measurements,
            },
            indent=2,
        )
    )
```

<!-- ARENA:cell markdown -->
# What have we learned?

The exact model let us verify the algebra: recover a calendar plane, change its
answer, and preserve an independent variable. The pretrained-model experiment
then asked separate empirical questions. A classifier can recognize calendar
words without preserving their chronological order. A supervised circular
probe can decode coordinates that an intervention cannot use. A large answer
change can partly reflect mean ablation. Each claim needs its own control.

## Limitations

We have tested new prompt templates, not new languages or unrestricted
calendar reasoning. All calendar classes appear in the main training set;
the separate leave-one-class-out probe is the stricter test of unseen identity.
The causal study changes one source token at one boundary, with layers chosen
from development data. Other positions, features and multi-layer interactions
can still matter. Small synthetic task families support a focused mechanism
study, not a general claim that the model uses circles for all arithmetic.

## Further investigations

1. Find a layer where coordinate decoding is strong and interventions are weak.
   Explain how information may already have moved to other positions.
2. Compare errors by offset and template, retaining the original denominators.
   Does the full-donor comparator fail on the same examples as the circle?
3. Repeat the full study with a preregistered class-disjoint split or a new model.
   Use development examples for choices, then keep the final examples untouched.

## Reading

- [Engels et al., Not All Language Model Features Are One-Dimensionally Linear](https://arxiv.org/abs/2405.14860v3) and [the authors' implementation](https://github.com/JoshEngels/MultiDimensionalFeatures): calendar geometry and interventions. Our centered and raw mean-ablation conventions are shown separately.
- [Gurnee and Tegmark, Language Models Represent Space and Time](https://arxiv.org/abs/2310.02207v3): a broader entity-coordinate study. This calendar lesson does not reproduce its space/time experiments.
- [UMAP's new-data transformation](https://umap-learn.readthedocs.io/en/latest/transform.html) and [scikit-learn t-SNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html): the different evaluation scopes used here.

<!-- ARENA:cell code -->
```python
if RUN_REAL_MODELS:
    runtime_record = {
        "gpu": t.cuda.get_device_name(),
        "peak_cuda_allocated_gib": t.cuda.max_memory_allocated() / 2**30,
        "elapsed_real_seconds": time.perf_counter() - started,
        "model_forward_rows": study_result["model_forward_rows"],
        "retained_outcomes": study_result["retained_outcomes"],
        "projection_fits": len(projection_records),
        "angle_play_forward_rows": sum(len(record["calendar_logits"]) for record in angle_records)
        + 4,
        "single_play_forward_rows": len(play_measurements),
    }
    (assets_dir.parent / "experiments" / "runtime.json").write_text(
        json.dumps(runtime_record, indent=2)
    )
    print(
        f"Finished on {runtime_record['gpu']}: {runtime_record['elapsed_real_seconds']:.1f} s, "
        f"peak CUDA allocation {runtime_record['peak_cuda_allocated_gib']:.2f} GiB."
    )
```
