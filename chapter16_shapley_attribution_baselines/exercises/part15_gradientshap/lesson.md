<!-- ARENA:cell markdown -->
# [16.15] Expected Gradients: Which Expectation Does GradientSHAP Estimate?

> **Claim.** Expected Gradients can converge to a complete continuous-path attribution while
> disagreeing with exact Shapley values for the same finite background. Gaussian query noise changes
> the prediction target as well as the estimator.

## Core Question

Which expectation does a GradientSHAP implementation estimate, and how can we distinguish Monte
Carlo error from a different attribution target?

Start with the smallest useful example:

$$
f(z)=\operatorname{ReLU}(z),\qquad x=1,\qquad b=-1.
$$

At a uniformly sampled point $b+\alpha(x-b)$, the path contribution is

$$
C(\alpha)=(x-b)f'(b+\alpha(x-b))=
\begin{cases}
0,&\alpha<\tfrac12,\\
2,&\alpha>\tfrac12.
\end{cases}
$$

Thus one draw is a Bernoulli-valued estimate, even though the population attribution is exactly
$E[C]=1$. Its variance is 1 and the ordinary standard error of a $K$-draw mean is $1/\sqrt K$.
A finite explanation need not sum exactly to $f(x)-f(b)=1$; that residual is sampling error.

You will build this estimator from explicit donor and path schedules, then compare it with numerical
quadrature and exhaustive feature-coalition Shapley. The real study uses every held-out concrete
recipe and the same seven raw-MPa models as section 16.14. It runs on CPU in float64. Allow
**4–6 hours**; Exercises 3–7 contain the core gradient, estimator and integration logic.

## Learning objectives

By the end of the lab, you should be able to:

- derive Integrated Gradients and Expected Gradients from a straight-line path;
- create row-stable, prefix-stable donor and alpha schedules without mutating global RNG state;
- retain individual gradient-times-displacement samples before averaging them;
- compute the ordinary plug-in Monte Carlo standard error from an unbiased sample-variance
  estimate, and explain what it does and does not measure;
- distinguish a full-background base from the sampled-donor average in one finite run;
- use all-background Gauss-Legendre quadrature as an approximate numerical control;
- prove that a complete path allocation can differ from discrete feature Shapley;
- explain why Gaussian query noise changes the target to a smoothed prediction;
- use global explanation-split rankings for audit replacement curves and local rankings only for
  exploratory plays.

## Setup

The student implementation supports deterministic scalar-output PyTorch models that act
independently on each row. Tensors and model parameters use CPU float64. Explicit schedules make
sampling semantics reviewable and allow smaller budgets to be literal prefixes of larger ones.

<!-- ARENA:cell code -->
```python
import hashlib
import math
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch as t
from IPython.display import display
from torch import Tensor, nn

torch.set_default_dtype(torch.float64)

chapter = "chapter16_shapley_attribution_baselines"
section = "part15_gradientshap"
root_dir = next(path for path in (Path.cwd(), *Path.cwd().parents) if (path / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

from part15_gradientshap import gradientshap_methods_tests as tests
from part15_gradientshap import gradientshap_plots as plots
from part15_gradientshap.gradientshap_data import (
    background_order,
    load_manifest,
    load_split,
    tensor_mapping_sha256,
)

FEATURE_NAMES = np.array([
    "Cement", "Slag", "Fly ash", "Water", "Superplasticizer",
    "Coarse aggregate", "Fine aggregate", "Age",
])
RUN_OUTPUT = os.environ.get("ARENA_GRADIENTSHAP_OUTPUT_DIRECTORY")
FIGURE_DIR = Path(
    os.environ.get("ARENA_GRADIENTSHAP_FIGURE_DIR", RUN_OUTPUT or section_dir / "figures")
)
SAVE_FIGURES = os.environ.get(
    "ARENA_GRADIENTSHAP_SAVE_FIGURES", "1" if RUN_OUTPUT else "0"
) == "1"


def show_figure(fig, stem):
    """Display a figure and optionally save the exact notebook rendering."""
    display(fig)
    if SAVE_FIGURES:
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
```

<!-- ARENA:cell markdown -->
The named helpers below are mechanical tensor, model, schedule, and state validators. They check
the lesson's declared boundary but do not implement any attribution operation. Every Expected
Gradients method remains in the exercises.

<!-- ARENA:cell code -->
```python
from part15_gradientshap.gradientshap_methods import (
    _eg_float_tensor,
    _eg_index_tensor,
    _eg_model,
    _eg_model_output,
    _eg_positive_int,
    _eg_preserve_execution_state,
    _eg_seed_words,
    _eg_validate_paths,
)
```

<!-- ARENA:cell markdown -->
Diagnostic fixtures construct literal affine, quadratic, ReLU and constant functions. Each staged
check resolves only methods already defined above it.

<!-- ARENA:cell code -->
```python
# ARENA:insert gradientshap_diagnostics.py:mechanical_helpers
```

<!-- ARENA:cell markdown -->
## From Integrated Gradients to an expectation

For a differentiable scalar function $f:\mathbb R^D\to\mathbb R$, query $x$, and fixed baseline
$b$, straight-line Integrated Gradients is

$$
\operatorname{IG}_i(x;b)
=(x_i-b_i)\int_0^1
\frac{\partial f\bigl(b+\alpha(x-b)\bigr)}{\partial x_i}\,d\alpha.
$$

If the line-restricted function is absolutely continuous, the chain rule and fundamental theorem of
calculus give

$$
\begin{aligned}
f(x)-f(b)
&=\int_0^1 \frac{d}{d\alpha}f\bigl(b+\alpha(x-b)\bigr)\,d\alpha\\
&=\sum_i (x_i-b_i)\int_0^1
\partial_i f\bigl(b+\alpha(x-b)\bigr)\,d\alpha\\
&=\sum_i \operatorname{IG}_i(x;b).
\end{aligned}
$$

Expected Gradients also samples the baseline $B$:

$$
\operatorname{EG}_i(x;P_B)
=\mathbb E_{b\sim P_B,\,\alpha\sim U(0,1)}
\left[(x_i-b_i)\partial_i f\bigl(b+\alpha(x-b)\bigr)\right].
$$

For this lesson, $P_B$ is uniform over declared intact background rows. The expectation therefore
sums to $f(x)-\mathbb E_b[f(b)]$ under the same path regularity assumptions. A finite sample mean is
unbiased for this quantity, but it is not exactly complete draw by draw.

## 1️⃣ Build a reproducible path schedule

A schedule contains a donor index and one uniform $\alpha$ for every source row and draw. The
important invariant is stronger than ordinary seeded reproducibility: each source ID owns
independent donor and alpha streams. Therefore changing row order, row batching, or the maximum
budget does not change an existing row's prefix.

A single generator that first draws an `(N,K)` donor array and then an `(N,K)` alpha array violates
this invariant. Increasing `K` advances the generator farther before any alpha is drawn, so even the
old alpha prefix changes.

### Exercise 1 — `sample_path_schedule`

> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵🔵🔵🔵
>
> You should spend up to 20-30 minutes on this exercise.
> ```

Return CPU arrays `indices` and `alpha`, each with shape `(N,K)`. Donor indices are int64 in
`[0, background_rows)`; alphas are float64 in `[0,1)`. Derive separate local streams from the fixed
seed, source ID, and stream name. Never alter NumPy's global RNG.

<!-- ARENA:cell code -->
```python
def sample_path_schedule(source_ids, background_rows, samples, seed):
    """Return row-stable donor indices and uniform path positions."""
    raise NotImplementedError()


tests.test_sample_path_schedule(sample_path_schedule)
```

<!-- ARENA:cell markdown -->
<details><summary>Help - give each source two private streams</summary>

Validate the source IDs first. For each ID, derive one deterministic seed for donors and another for
alphas, create local `np.random.Generator` instances, and draw length `samples` vectors. Do not use
Python's salted `hash` or the legacy global `np.random` functions.

</details>

<details><summary>Common bug - one generator for the whole batch</summary>

A batch-level generator can reproduce one exact call while failing the scientific invariant. Check
the same source ID after reordering the rows, after requesting more draws, and by itself. Its donor
and alpha prefixes must be identical in all four calls.

</details>

<details><summary>Expected output</summary>

```text
All tests in test_sample_path_schedule passed!
```

The diagnostic prints fixed donor IDs and alphas, then confirms that the first eight draws are the
literal prefix of a 32-draw schedule.

</details>

<details><summary>Interpretation</summary>

A schedule is part of the estimator definition, not incidental randomness. Prefix stability lets a
convergence plot compare budgets on paired draws rather than silently changing every earlier draw.
Row-stable streams also make batching an implementation detail instead of a scientific choice.

</details>

<details><summary>Solution</summary>

```python
def sample_path_schedule(
    source_ids: Tensor,
    background_rows: int,
    samples: int,
    seed: int,
) -> tuple[Tensor, Tensor]:
    """Draw donor indices and alphas from independent local streams for each source ID."""
    _eg_index_tensor("source_ids", source_ids, 1)
    background_rows = _eg_positive_int("background_rows", background_rows)
    samples = _eg_positive_int("samples", samples)
    if type(seed) is not int or seed < 0 or seed >= 2**64:
        raise ValueError("seed must be an integer in [0, 2**64).")
    if (source_ids < 0).any():
        raise ValueError("source_ids must be nonnegative integers.")
    if len(t.unique(source_ids)) != len(source_ids):
        raise ValueError("source_ids must be unique so every row names one sampling stream.")

    indices = np.empty((len(source_ids), samples), dtype=np.int64)
    alpha = np.empty((len(source_ids), samples), dtype=np.float64)
    for row, source_id in enumerate(source_ids.tolist()):
        donor_rng = np.random.Generator(
            np.random.PCG64(_eg_seed_words(seed, source_id, "baseline"))
        )
        alpha_rng = np.random.Generator(
            np.random.PCG64(_eg_seed_words(seed, source_id, "alpha"))
        )
        indices[row] = donor_rng.integers(0, background_rows, size=samples, dtype=np.int64)
        alpha[row] = alpha_rng.random(samples, dtype=np.float64)
    return t.tensor(indices, dtype=t.int64), t.tensor(alpha, dtype=t.float64)
```

</details>

<!-- ARENA:cell code -->
```python
# ARENA:insert gradientshap_diagnostics.py:schedule_fixture
```

<!-- ARENA:cell code -->
```python
# ARENA:insert gradientshap_diagnostics.py:schedule_diagnostic
```

<!-- ARENA:cell code -->
```python
schedule_check = diagnose_schedule(sample_path_schedule)
show_figure(plots.plot_schedule_prefix(schedule_check), "01_schedule_prefix")
```

<!-- ARENA:cell markdown -->
## 2️⃣ Interpolate both the point and displacement

For donor $b_k$, path position $\alpha_k$, and optional query perturbation $\epsilon_k$, define

$$
\tilde x_k=x+\epsilon_k,\qquad
p_k=b_k+\alpha_k(\tilde x_k-b_k),\qquad
\Delta_k=\tilde x_k-b_k.
$$

The gradient is evaluated at $p_k$ and multiplied by the same $\Delta_k$. Noise must appear in both
expressions. Perturbing the point while retaining the clean displacement computes a hybrid quantity
that is not the noisy Expected Gradients target.

### Exercise 2 — `interpolate_paths`

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵🔵🔵
>
> You should spend up to 15-20 minutes on this exercise.
> ```

Gather donors using `indices`, broadcast each query across its draw axis, and return `(points,
deltas)` with shape `(N,K,D)`. With `noise=None`, use an exact zero perturbation without drawing any
random numbers.

<!-- ARENA:cell code -->
```python
def interpolate_paths(x, background, indices, alpha, noise=None):
    """Return interpolation points and query-minus-donor displacements."""
    raise NotImplementedError()


tests.test_interpolate_paths(interpolate_paths)
```

<!-- ARENA:cell markdown -->
<details><summary>Help - name the three broadcast axes</summary>

`x[:, None, :]` is `(N,1,D)`, `background[indices]` is `(N,K,D)`, and
`alpha[:, :, None]` is `(N,K,1)`. Build the noisy query first; use that same tensor in both the
convex combination and subtraction.

</details>

<details><summary>Expected output</summary>

```text
All tests in test_interpolate_paths passed!
```

For $f(z)=z^2$, $x=1$, $b=0$, and alphas `(0.25,0.75)`, the points are `(0.25,0.75)` and both
displacements are one.

</details>

<details><summary>Interpretation</summary>

The path position controls where the model is differentiated; the displacement controls how that
local sensitivity is converted into a finite input contribution. Keeping these arrays separate
makes a missing-noise or wrong-donor bug visible before autograd is involved.

</details>

<details><summary>Solution</summary>

```python
def interpolate_paths(
    x: Tensor,
    background: Tensor,
    indices: Tensor,
    alpha: Tensor,
    noise: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Return sampled path points and query-minus-donor deltas, both shaped N,K,D."""
    _eg_validate_paths(x, background, indices, alpha, noise)
    donors = background[indices]
    query = x[:, None, :] if noise is None else x[:, None, :] + noise
    delta = query - donors
    points = donors + alpha[:, :, None] * delta
    return points, delta
```

</details>

<!-- ARENA:cell code -->
```python
# ARENA:insert gradientshap_diagnostics.py:path_fixture
```

<!-- ARENA:cell code -->
```python
# ARENA:insert gradientshap_diagnostics.py:path_diagnostic
```

<!-- ARENA:cell code -->
```python
path_check = diagnose_interpolation(interpolate_paths)
show_figure(plots.plot_path_interpolation(path_check), "02_path_interpolation")
```

<!-- ARENA:cell markdown -->
## 3️⃣ Differentiate a scalar row-wise model

Flattening `(N,K,D)` path points to `(N*K,D)` lets one backward call evaluate many samples. For a
model that applies the same scalar function independently to every row,

$$
\nabla_P\sum_r f(P_r)
=\bigl(\nabla f(P_1),\ldots,\nabla f(P_{NK})\bigr).
$$

The equality depends on row independence. Merely observing one scalar per row does not prove it for
a model with batch-dependent operations. This lesson validates a deliberately narrow graph before
using the sum-of-outputs trick.

The helper must also preserve the supported caller state: input storage, model training mode,
buffers, global RNG state, and any existing parameter gradients. The supported study models do not
mutate their parameter values during `forward`. The returned gradient is fresh CPU float64 storage.

### Exercise 3 — `input_gradients`

> ```yaml
> Difficulty: 🔴🔴🔴🔴⚪
> Importance: 🔵🔵🔵🔵🔵
>
> You should spend up to 30-40 minutes on this exercise.
> ```

Differentiate the model in bounded `forward_batch` chunks and restore its original mode. Do not use
`loss.backward()`, which accumulates parameter gradients. Use `torch.autograd.grad` with respect to
a fresh point tensor.

<!-- ARENA:cell code -->
```python
def input_gradients(model, points, forward_batch=32768):
    """Return one input gradient for every independent scalar model evaluation."""
    raise NotImplementedError()


tests.test_input_gradients(input_gradients)
```

<!-- ARENA:cell markdown -->
<details><summary>Help - differentiate only the temporary input</summary>

For each flattened chunk, use `detach().clone().requires_grad_(True)`. Evaluate a validated scalar
output, call `torch.autograd.grad(output.sum(), chunk)`, detach the result, and reshape the
concatenated gradients to the original point shape. Save and restore `model.training` in a
`try/finally` block.

</details>

<details><summary>Common bug - clearing parameter gradients</summary>

Calling `model.zero_grad()` avoids new accumulation but destroys gradients the caller may already
hold. `autograd.grad` can request only the input derivative, leaving every parameter's `.grad`
unchanged. Tests should seed nonzero parameter gradients before the call and compare them after it.

</details>

<details><summary>Expected output</summary>

```text
All tests in test_input_gradients passed!
```

The affine gradient is constant, the quadratic gradient changes with the path point, and a constant
model returns the deliberate zero-gradient behavior specified by the method contract.

</details>

<details><summary>Interpretation</summary>

Expected Gradients is inexpensive because it replaces a path integral by sampled ordinary input
gradients. That computational shortcut is valid only if each flattened row means the same function
call it meant before batching.

</details>

<details><summary>Solution</summary>

```python
def input_gradients(
    model: nn.Module,
    points: Tensor,
    forward_batch: int = 32_768,
) -> Tensor:
    """Differentiate a deterministic scalar function applied independently to each row.

    The sum-of-outputs gradient is valid only under that batch-independence assumption.
    This function preserves the caller's tensors, RNG states, model modes, buffers, and
    parameter gradients. A scalar function constant with respect to its input gets zero
    input gradients.
    """
    model = _eg_model(model)
    _eg_float_tensor("points", points, 3)
    forward_batch = _eg_positive_int("forward_batch", forward_batch)
    flat_points = points.reshape(-1, points.shape[-1])
    flat_gradients = t.empty_like(flat_points)

    with _eg_preserve_execution_state(model):
        for start in range(0, len(flat_points), forward_batch):
            chunk = flat_points[start : start + forward_batch].detach().clone().requires_grad_(True)
            with t.enable_grad():
                output = _eg_model_output(model, chunk)
                if output.requires_grad:
                    gradient = t.autograd.grad(output.sum(), chunk, allow_unused=True)[0]
                else:
                    gradient = None
            if gradient is None:
                gradient = t.zeros_like(chunk)
            if not t.isfinite(gradient).all():
                raise ValueError("The model produced a nonfinite input gradient.")
            flat_gradients[start : start + len(chunk)] = gradient.detach()
    return flat_gradients.reshape(points.shape)
```

</details>

<!-- ARENA:cell code -->
```python
# ARENA:insert gradientshap_diagnostics.py:gradient_fixture
```

<!-- ARENA:cell code -->
```python
# ARENA:insert gradientshap_diagnostics.py:gradient_diagnostic
```

<!-- ARENA:cell code -->
```python
gradient_check = diagnose_gradients(input_gradients)
show_figure(plots.plot_gradient_controls(gradient_check), "03_gradient_controls")
```

<!-- ARENA:cell markdown -->
## 4️⃣ Retain every Expected Gradients draw

For explicit schedule entries $(j_{nk},\alpha_{nk})$, choose donor
$b_{nk}=B_{j_{nk}}$, form $p_{nk}=b_{nk}+\alpha_{nk}(x_n-b_{nk})$, and retain

$$
C_{nki}=(x_{ni}-b_{nki})\partial_i f(p_{nk}).
$$

The estimator is the draw mean, but averaging here would throw away the evidence needed to measure
Monte Carlo uncertainty, inspect discrete samples, and reuse literal prefixes. This function returns
the full `(N,K,D)` contribution tensor.

For the opening ReLU, alphas `(0.25,0.75)` produce path points `(-0.5,0.5)`, gradients `(0,1)`,
and contributions `(0,2)`. For $f(z)=z^2$, $x=1$, $b=0$, the same alphas produce contributions
`(0.5,1.5)`. An endpoint-gradient implementation would return two in both quadratic draws and fail.

### Exercise 4 — `expected_gradient_samples`

> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵🔵🔵🔵
>
> You should spend up to 20-30 minutes on this exercise.
> ```

Compose `interpolate_paths` and `input_gradients`, then multiply gradients by displacements. Preserve
all draws and return fresh storage.

<!-- ARENA:cell code -->
```python
def expected_gradient_samples(
    model,
    x,
    background,
    indices,
    alpha,
    forward_batch=32768,
):
    """Return one path-gradient contribution per query, draw, and feature."""
    raise NotImplementedError()


tests.test_expected_gradient_samples(expected_gradient_samples)
```

<!-- ARENA:cell markdown -->
<details><summary>Help - the method is a three-line composition</summary>

Call `interpolate_paths(x, background, indices, alpha)`, pass the points to `input_gradients`, and
multiply by the returned deltas. Most of the work is in preserving shapes and state in the two
functions you already wrote.

</details>

<details><summary>Expected output</summary>

```text
All tests in test_expected_gradient_samples passed!
```

The diagnostic displays literal ReLU samples `(0,2)` and quadratic samples `(0.5,1.5)` on their
analytic distributions.

</details>

<details><summary>Interpretation</summary>

A contribution draw combines two random choices: which endpoint donor defines the path and where
the gradient is sampled on that path. Retaining draws lets us quantify their combined estimator
variation. It does not quantify uncertainty about the trained model, dataset, or choice of baseline
distribution.

</details>

<details><summary>Solution</summary>

```python
def expected_gradient_samples(
    model: nn.Module,
    x: Tensor,
    background: Tensor,
    indices: Tensor,
    alpha: Tensor,
    forward_batch: int = 32_768,
) -> Tensor:
    """Return clean Expected Gradients Monte Carlo contributions, shape N,K,D."""
    points, delta = interpolate_paths(x, background, indices, alpha)
    gradients = input_gradients(model, points, forward_batch)
    return gradients * delta
```

</details>

<!-- ARENA:cell code -->
```python
# ARENA:insert gradientshap_diagnostics.py:sample_fixture
```

<!-- ARENA:cell code -->
```python
# ARENA:insert gradientshap_diagnostics.py:sample_diagnostic
```

<!-- ARENA:cell code -->
```python
sample_check = diagnose_expected_gradient_samples(
    expected_gradient_samples,
    sample_path_schedule,
)
show_figure(plots.plot_relu_samples(sample_check), "04_relu_samples")
```

<!-- ARENA:cell markdown -->
## 5️⃣ Estimate a mean and its Monte Carlo standard error

For one feature coordinate with draws $C_1,\ldots,C_K$, define

$$
\widehat\phi_K=\frac1K\sum_{k=1}^K C_k,
\qquad
s_K^2=\frac1{K-1}\sum_{k=1}^K(C_k-\widehat\phi_K)^2,
\qquad
\widehat{\operatorname{SE}}_K=\frac{s_K}{\sqrt K}.
$$

The `K-1` denominator makes $s_K^2$ the unbiased sample-variance estimator. The standard error
estimates variability of the sample mean under repeated schedules. It is not a standard deviation of
feature importance, a model confidence interval, or a bound on error against exact coalition
Shapley. Coordinates also share donor and alpha draws, so their errors are correlated.

Nested budgets use literal prefixes of the same contribution tensor. That pairing makes a running
plot interpretable: only additional draws change the estimate.

### Exercise 5 — `summarize_expected_gradients`

> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵🔵🔵🔵
>
> You should spend up to 20-30 minutes on this exercise.
> ```

For every requested budget, return the prefix mean and ordinary standard error. Validate increasing,
unique budgets that fit inside the draw axis. Do not use SHAP's package-reported uncertainty field;
we will inspect its different formula later.

<!-- ARENA:cell code -->
```python
def summarize_expected_gradients(samples, budgets):
    """Summarize nested draw prefixes with means and ordinary sample standard errors."""
    raise NotImplementedError()


tests.test_summarize_expected_gradients(summarize_expected_gradients)
```

<!-- ARENA:cell markdown -->
<details><summary>Help - slice first, then reduce the draw axis</summary>

For each `budget`, select `samples[:, :budget]`. Compute its mean over dimension 1 and its sample
standard deviation with `correction=1`, then divide by `sqrt(budget)`. Stack results in budget order;
do not reduce the query or feature dimensions.

</details>

<details><summary>Common bug - calling sample spread an estimator SE</summary>

`prefix.std(dim=1)` measures the spread of individual contributions. The uncertainty of their mean
is smaller by `sqrt(K)`. Conversely, dividing the variance itself by `sqrt(K)` is neither the
ordinary variance of the mean nor its standard error.

</details>

<details><summary>Expected output</summary>

```text
All tests in test_summarize_expected_gradients passed!
```

For the analytic ReLU distribution, independent repetitions should shrink like $1/\sqrt K$ while
individual draws remain zero or two.

</details>

<details><summary>Interpretation</summary>

A smaller reported SE says the declared Monte Carlo mean is becoming stable for this estimand. It
does not say the baseline distribution is appropriate, that the path stays on the data manifold, or
that Expected Gradients agrees with discrete Shapley.

</details>

<details><summary>Solution</summary>

```python
def summarize_expected_gradients(
    samples: Tensor,
    budgets: tuple[int, ...],
) -> tuple[Tensor, Tensor]:
    """Return ordinary sample SE, not SHAP 0.49.1's variance/sqrt(K) proxy."""
    _eg_float_tensor("samples", samples, 3)
    if type(budgets) is not tuple or not budgets:
        raise TypeError("budgets must be a nonempty tuple of sample counts.")
    if any(type(budget) is not int for budget in budgets):
        raise TypeError("Every budget must be an integer sample count.")
    if any(budget < 2 or budget > samples.shape[1] for budget in budgets):
        raise ValueError("Every budget must be between 2 and the available sample count.")
    if any(left >= right for left, right in zip(budgets, budgets[1:])):
        raise ValueError("budgets must be strictly increasing nested prefixes.")

    means, standard_errors = [], []
    for budget in budgets:
        prefix = samples[:, :budget]
        means.append(prefix.mean(dim=1))
        standard_errors.append(prefix.std(dim=1, correction=1) / math.sqrt(budget))
    return t.stack(means), t.stack(standard_errors)
```

</details>

<!-- ARENA:cell code -->
```python
# ARENA:insert gradientshap_diagnostics.py:convergence_diagnostic
```

<!-- ARENA:cell code -->
```python
convergence_check = diagnose_convergence(
    sample_path_schedule,
    expected_gradient_samples,
    summarize_expected_gradients,
)
show_figure(plots.plot_mc_convergence(convergence_check), "05_mc_convergence")
```

<!-- ARENA:cell markdown -->
The running estimate should wander between the two possible ReLU contributions before stabilizing
near one. The band is labeled **ordinary Monte Carlo SE** because that is the only uncertainty it
represents. Erion et al. found 100–200 samples adequate on their benchmarks; that empirical result
is not a universal stopping rule for this model or dataset.

<!-- ARENA:cell markdown -->
## 6️⃣ Report the finite estimate against its actual base

The finite empirical-background target is

$$
\phi_0(B)=\frac1{|B|}\sum_{b\in B}f(b),
\qquad
\phi_0(B)+\sum_i\operatorname{EG}_i(x;B)=f(x).
$$

`expected_gradients` reports the deterministic full-background base, the finite draw mean, and its
ordinary SE. The finite residual

$$
r_K=\phi_0(B)+\sum_i\widehat\phi_{K,i}-f(x)
$$

contains both donor-sampling and alpha-sampling error. Replacing the base by the sampled donors'
mean prediction can make one component disappear from a diagnostic while silently changing the
quantity being checked.

### Exercise 6 — `expected_gradients`

> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵🔵🔵🔵
>
> You should spend up to 20-30 minutes on this exercise.
> ```

Compose your per-draw estimator, average the draw axis, compute the ordinary SE, and evaluate the
model on every supplied background row for the base. Return all declared fields without mutating the
inputs or the supported model execution state.

<!-- ARENA:cell code -->
```python
def expected_gradients(
    model,
    x,
    background,
    indices,
    alpha,
    forward_batch=32768,
):
    """Return full-background base, finite Expected Gradients mean, and ordinary SE."""
    raise NotImplementedError()


tests.test_expected_gradients(expected_gradients)
```

<!-- ARENA:cell markdown -->
<details><summary>Help - reuse samples and retain the deterministic base</summary>

Call `expected_gradient_samples`, then summarize its complete draw axis. Evaluate the model on all
background rows under `torch.no_grad()` and average scalar outputs. Keep the base separate from the
attribution and SE arrays.

</details>

<details><summary>Expected output</summary>

```text
All tests in test_expected_gradients passed!
```

The affine fixture is exact under its balanced donor schedule. Dummy and constant coordinates stay
zero. The finite ReLU fixture exposes a nonzero residual when its alpha sample is unbalanced.

</details>

<details><summary>Interpretation</summary>

Completeness is an expectation statement. A finite residual consistent with sampling variability is
not automatically an implementation failure. An affine, balanced-donor control remains valuable
because it removes alpha variation and has a hand-derived target.

</details>

<details><summary>Solution</summary>

```python
def expected_gradients(
    model: nn.Module,
    x: Tensor,
    background: Tensor,
    indices: Tensor,
    alpha: Tensor,
    forward_batch: int = 32_768,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return full-background base, clean Expected Gradients, and ordinary estimator SE."""
    _, samples, _ = _eg_validate_paths(x, background, indices, alpha)
    if samples < 2:
        raise ValueError(
            "Expected Gradients needs at least two samples to estimate standard error."
        )
    contributions = expected_gradient_samples(
        model, x, background, indices, alpha, forward_batch
    )
    means, standard_errors = summarize_expected_gradients(contributions, (samples,))
    model = _eg_model(model)
    with _eg_preserve_execution_state(model), t.no_grad():
        base = _eg_model_output(model, background).mean().detach().clone()
    return base, means[0], standard_errors[0]
```

</details>

<!-- ARENA:cell code -->
```python
# ARENA:insert gradientshap_diagnostics.py:estimator_fixture
```

<!-- ARENA:cell code -->
```python
# ARENA:insert gradientshap_diagnostics.py:estimator_diagnostic
```

<!-- ARENA:cell code -->
```python
estimator_check = diagnose_expected_gradients(
    expected_gradients,
    sample_path_schedule,
)
show_figure(plots.plot_estimator_controls(estimator_check), "06_estimator_controls")
```

<!-- ARENA:cell markdown -->
## 7️⃣ Replace random alpha draws with bounded quadrature

Monte Carlo sampling is the main estimator, but a deterministic alpha-integration control helps us
measure its error. Let $(q_\ell,w_\ell)$ be Gauss-Legendre nodes and weights on `[-1,1]`. Mapping

$$
\alpha_\ell=\frac{q_\ell+1}{2},\qquad \omega_\ell=\frac{w_\ell}{2}
$$

gives the pairwise approximation

$$
\widehat{\operatorname{IG}}_i(x;b)
=(x_i-b_i)\sum_{\ell=1}^{L}\omega_\ell
\partial_i f\bigl(b+\alpha_\ell(x-b)\bigr).
$$

Evaluate every query against every intact donor, returning `(N,B,D)` pair attributions. Averaging
that donor axis removes donor-sampling variation, while doubling the node count tests alpha
integration drift.

This remains numerical quadrature. A ReLU path has a step-discontinuous gradient at each kink, so
finite Gauss-Legendre integration is not an exact oracle. It is exact for the affine fixture and
should integrate the quadratic fixture to floating-point precision; for piecewise-linear networks,
report the 256-to-512-node change rather than declaring 512 exact.

### Exercise 7 — `gauss_legendre_pairs`

> ```yaml
> Difficulty: 🔴🔴🔴🔴⚪
> Importance: 🔵🔵🔵🔵🔵
>
> You should spend up to 35-50 minutes on this exercise.
> ```

Construct nodes with NumPy, map them to `[0,1]`, and evaluate all query/donor paths in bounded row
and forward batches. Multiply the weighted average gradient by each query-minus-donor displacement.

<!-- ARENA:cell code -->
```python
def gauss_legendre_pairs(
    model,
    x,
    background,
    n_nodes,
    row_batch=4,
    forward_batch=32768,
):
    """Approximate every query-donor Integrated Gradients path by quadrature."""
    raise NotImplementedError()


tests.test_gauss_legendre_pairs(gauss_legendre_pairs)
```

<!-- ARENA:cell markdown -->
<details><summary>Help - preserve query, donor, node, and feature axes</summary>

For a query chunk, form points with shape `(rows,B,L,D)` and displacements `(rows,B,D)`. Flatten
only the first tensor for `input_gradients`, restore its four axes, multiply gradients by mapped
weights, sum the node axis, then multiply by the displacement. Assign the resulting `(rows,B,D)`
block into the output.

</details>

<details><summary>Common bug - calling finite quadrature ground truth</summary>

Polynomial exactness does not make Gauss-Legendre exact for a step integrand. A ReLU derivative
jumps where its path crosses zero, and a node generally misses that location. Compare two declared
node counts and retain their difference. Exact feature Shapley is also not an oracle for this path
integral; it is a different game.

</details>

<details><summary>Expected output</summary>

```text
All tests in test_gauss_legendre_pairs passed!
```

Affine and quadratic paths match their analytic integrals. The threshold path approaches
`(2/3,1/3)` as the node count grows, with a visible finite-node drift.

</details>

<details><summary>Interpretation</summary>

Enumerating donors and quadrature nodes changes the numerical estimator, not the target. It answers
whether a Monte Carlo result is close to the finite-background path expectation. It does not answer
whether that expectation matches a discrete coalition game.

</details>

<details><summary>Solution</summary>

```python
def gauss_legendre_pairs(
    model: nn.Module,
    x: Tensor,
    background: Tensor,
    n_nodes: int,
    row_batch: int = 4,
    forward_batch: int = 32_768,
) -> Tensor:
    """Approximate every query-donor Integrated Gradients path by finite quadrature.

    ReLU path gradients jump, so finite-node quadrature remains a numerical control
    rather than exact path truth.
    """
    model = _eg_model(model)
    _eg_float_tensor("x", x, 2)
    _eg_float_tensor("background", background, 2)
    if x.shape[1] != background.shape[1]:
        raise ValueError("x and background must have the same feature dimension.")
    n_nodes = _eg_positive_int("n_nodes", n_nodes)
    row_batch = _eg_positive_int("row_batch", row_batch)
    forward_batch = _eg_positive_int("forward_batch", forward_batch)

    raw_nodes, raw_weights = np.polynomial.legendre.leggauss(n_nodes)
    nodes = t.tensor((raw_nodes + 1) / 2, dtype=t.float64)
    weights = t.tensor(raw_weights / 2, dtype=t.float64)
    pairs = t.empty((len(x), len(background), x.shape[1]), dtype=t.float64)

    for start in range(0, len(x), row_batch):
        query = x[start : start + row_batch]
        delta = query[:, None, :] - background[None, :, :]
        points = background[None, :, None, :] + nodes[None, None, :, None] * delta[:, :, None, :]
        packed = points.reshape(len(query), len(background) * n_nodes, x.shape[1])
        gradients = input_gradients(model, packed, forward_batch).reshape(points.shape)
        mean_gradient = (gradients * weights[None, None, :, None]).sum(dim=2)
        pairs[start : start + len(query)] = mean_gradient * delta
    return pairs
```

</details>

<!-- ARENA:cell code -->
```python
# ARENA:insert gradientshap_diagnostics.py:quadrature_fixture
```

<!-- ARENA:cell code -->
```python
# ARENA:insert gradientshap_diagnostics.py:quadrature_diagnostic
```

<!-- ARENA:cell code -->
```python
quadrature_check = diagnose_quadrature(
    gauss_legendre_pairs,
    input_gradients,
)
show_figure(plots.plot_quadrature_doubling(quadrature_check), "07_quadrature_doubling")
```

<!-- ARENA:cell markdown -->
### Why averaging attributions is not using the mean baseline

For $f(z)=\operatorname{ReLU}(z)$, $x=2$, and background `(-2,1)`, the two complete path
attributions are `(2,1)`. Their average is `1.5`, and the full-background base is
`mean(0,1)=0.5`, so `0.5+1.5=2`. The mean baseline is `-0.5`, whose single path attribution is
`2`. Nonlinearity sits inside the baseline average:

$$
\mathbb E_b[\operatorname{IG}(x;b)]
\ne \operatorname{IG}(x;\mathbb E[b]).
$$

<!-- ARENA:cell code -->
```python
# ARENA:insert gradientshap_diagnostics.py:background_diagnostic
```

<!-- ARENA:cell code -->
```python
background_check = diagnose_background_average(gauss_legendre_pairs)
show_figure(plots.plot_background_average(background_check), "08_background_average")
```

<!-- ARENA:cell markdown -->
## 8️⃣ Make Gaussian query noise part of the method call

Some GradientSHAP implementations perturb the query before drawing a path. With
$\epsilon\sim\mathcal N(0,\sigma^2I)$ and $\tilde x=x+\epsilon$, the retained sample is

$$
C_{ki}^{\sigma}
=(\tilde x_{ki}-b_{ki})
\partial_i f\bigl(b_k+\alpha_k(\tilde x_k-b_k)\bigr).
$$

Under the path assumptions, its population sum is

$$
\sum_i \operatorname{EG}^{\sigma}_i(x;B)
=\mathbb E_\epsilon[f(x+\epsilon)]-\mathbb E_b[f(b)],
$$

which is generally not the clean target $f(x)-\mathbb E_b[f(b)]$. Noise is therefore more than a
variance setting: it defines an average over a distribution of nearby queries.

For $f(z)=\operatorname{ReLU}(z)$, clean query $x=0$, donor $b=-1$, and
$\epsilon\sim\mathcal N(0,\sigma^2)$,

$$
\mathbb E[\operatorname{ReLU}(\epsilon)]-\operatorname{ReLU}(-1)
=\frac{\sigma}{\sqrt{2\pi}}.
$$

At $\sigma=1$, the noisy target is about `0.3989422804`, while the clean target is zero. Comparing
the noisy attribution sum with zero and calling the difference a completeness failure would be a
category error.

### Exercise 8 — `noisy_expected_gradient_samples`

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵🔵🔵
>
> You should spend up to 15-20 minutes on this exercise.
> ```

Accept the noise tensor explicitly rather than sampling inside the method. Use it in both the noisy
query and displacement. For an exact zero noise tensor, the result must equal the clean estimator
bitwise under the same schedule.

<!-- ARENA:cell code -->
```python
def noisy_expected_gradient_samples(
    model,
    x,
    background,
    indices,
    alpha,
    noise,
    forward_batch=32768,
):
    """Return Expected Gradients draws for explicitly perturbed queries."""
    raise NotImplementedError()


tests.test_noisy_expected_gradient_samples(noisy_expected_gradient_samples)
```

<!-- ARENA:cell markdown -->
<details><summary>Help - reuse interpolation rather than adding noise afterward</summary>

Call `interpolate_paths(..., noise=noise)`, differentiate those points, and multiply by the returned
noisy displacements. Adding noise to a finished clean attribution or only to its gradient point does
not implement the equation above.

</details>

<details><summary>Expected output</summary>

```text
All tests in test_noisy_expected_gradient_samples passed!
```

Zero noise matches clean samples exactly. A fixed standard-normal schedule approaches
$\sigma/\sqrt{2\pi}$ within its ordinary Monte Carlo uncertainty.

</details>

<details><summary>Interpretation</summary>

The supplied noise array makes the estimand auditable and keeps global Torch RNG state irrelevant.
A library option named `stdevs` or `local_smoothing` still needs its precise placement checked in
source: where noise enters determines the target.

</details>

<details><summary>Solution</summary>

```python
def noisy_expected_gradient_samples(
    model: nn.Module,
    x: Tensor,
    background: Tensor,
    indices: Tensor,
    alpha: Tensor,
    noise: Tensor,
    forward_batch: int = 32_768,
) -> Tensor:
    """Use x+noise in point and delta, targeting the noise-smoothed query output."""
    points, delta = interpolate_paths(x, background, indices, alpha, noise)
    gradients = input_gradients(model, points, forward_batch)
    return gradients * delta
```

</details>

<!-- ARENA:cell code -->
```python
# ARENA:insert gradientshap_diagnostics.py:noise_fixture
```

<!-- ARENA:cell code -->
```python
# ARENA:insert gradientshap_diagnostics.py:noise_diagnostic
```

<!-- ARENA:cell code -->
```python
noise_check = diagnose_noise_target(
    noisy_expected_gradient_samples,
    expected_gradient_samples,
    summarize_expected_gradients,
)
show_figure(plots.plot_noise_target(noise_check), "09_gaussian_target_shift")
```

<!-- ARENA:cell markdown -->
## 9️⃣ Turn a global attribution order into a coalition intervention

Expected Gradients follows continuous diagonal paths. A replacement curve instead follows a
sequence of discrete feature coalitions. Given a complete `(N,2**D)` coalition table, an order
$j_0,\ldots,j_{D-1}$ defines insertion masks

$$
0,\quad 2^{j_0},\quad 2^{j_0}+2^{j_1},\quad\ldots,\quad 2^D-1,
$$

and deletion masks in reverse from the full coalition. Looking up their raw values exposes what the
model does under donor replacement without pretending the ranking and game are the same object.

The primary audit study ranks each method once by mean absolute attribution over all 153 explanation
rows, fixes those global orders, then applies them to every audit row. The default local play may
instead order one row by `sign(f(x)-base) * phi`; that exploratory ranking cannot support a new
confirmatory audit claim.

### Exercise 9 — `ranked_coalition_curves`

> ```yaml
> Difficulty: 🔴🔴🔴🔴⚪
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 30-40 minutes on this exercise.
> ```

Return raw insertion and deletion values, their integer masks, and the feature order. Validate an
explicit order as a row-wise permutation. If `orders=None`, derive the local signed order with a
stable feature-index tie break.

<!-- ARENA:cell code -->
```python
def ranked_coalition_curves(values, phi, orders=None):
    """Read insertion and deletion prefixes from complete coalition tables."""
    raise NotImplementedError()


tests.test_ranked_coalition_curves(ranked_coalition_curves)
```

<!-- ARENA:cell markdown -->
<details><summary>Help - accumulate one feature bit per step</summary>

Initialize insertion masks to zero and deletion masks to `2**D-1`. At each step, convert the next
feature index to `1 << index`, add that bit to insertion, and remove it from deletion. Use `gather`
to read the corresponding coalition values.

</details>

<details><summary>Common bug - treating a replacement curve as an exactness score</summary>

A ranking that rises quickly can still come from an attribution vector far from exact Shapley.
Interactions can also make insertion or deletion nonmonotone. Curves measure model behavior along
one ordered computational intervention; they do not prove that the ranking is uniquely correct.

</details>

<details><summary>Expected output</summary>

```text
All tests in test_ranked_coalition_curves passed!
```

The additive fixture returns exact prefix values and a dummy-feature plateau. The random-order
control retains every declared order instead of selecting a favorable comparison.

</details>

<details><summary>Interpretation</summary>

The attribution and intervention questions are now deliberately separate. Expected Gradients may
be exact for its path expectation, exact Shapley is exact for the declared feature game, and a
replacement curve asks how one ranking traverses that game.

</details>

<details><summary>Solution</summary>

```python
def ranked_coalition_curves(
    values: Tensor,
    phi: Tensor,
    orders: Tensor | None = None,
) -> dict[str, Tensor]:
    """Gather raw insertion and deletion prefixes from complete coalition tables."""
    _eg_float_tensor("values", values, 2)
    _eg_float_tensor("phi", phi, 2)
    n_rows, size = values.shape
    n_features = size.bit_length() - 1
    if n_features < 1 or size != 2**n_features or phi.shape != (n_rows, n_features):
        raise ValueError("values must be (N,2**D) and phi must be (N,D), with D >= 1.")

    if orders is None:
        direction = t.where(values[:, -1] - values[:, 0] >= 0, 1.0, -1.0)[:, None]
        orders = t.argsort(direction * phi, dim=1, descending=True, stable=True)
    else:
        _eg_index_tensor("orders", orders, 2)
        if orders.shape != (n_rows, n_features):
            raise ValueError("orders must have shape (rows, features).")
        expected = t.arange(n_features, dtype=t.int64).expand(n_rows, n_features)
        if not t.equal(orders.sort(dim=1).values, expected):
            raise ValueError("Every order row must contain each feature exactly once.")
        orders = orders.clone()

    insertion_masks = t.zeros((n_rows, n_features + 1), dtype=t.int64)
    deletion_masks = t.full((n_rows, n_features + 1), size - 1, dtype=t.int64)
    for step in range(n_features):
        bit = 1 << orders[:, step]
        insertion_masks[:, step + 1] = insertion_masks[:, step] | bit
        deletion_masks[:, step + 1] = deletion_masks[:, step] ^ bit
    return {
        "orders": orders,
        "insertion_masks": insertion_masks,
        "deletion_masks": deletion_masks,
        "insertion": values.gather(1, insertion_masks),
        "deletion": values.gather(1, deletion_masks),
    }
```

</details>

<!-- ARENA:cell code -->
```python
# ARENA:insert gradientshap_diagnostics.py:threshold_fixture
```

<!-- ARENA:cell code -->
```python
# ARENA:insert gradientshap_diagnostics.py:kink_fixture
```

<!-- ARENA:cell code -->
```python
# ARENA:insert gradientshap_diagnostics.py:replacement_fixture
```

<!-- ARENA:cell code -->
```python
# ARENA:insert gradientshap_diagnostics.py:game_diagnostic
```

<!-- ARENA:cell code -->
```python
game_check = diagnose_path_vs_coalition(
    gauss_legendre_pairs,
    ranked_coalition_curves,
)
show_figure(
    plots.plot_path_vs_coalition(game_check),
    "10_path_vs_coalition",
)
show_figure(
    plots.plot_kink_representation(game_check),
    "11_kink_representation",
)
show_figure(
    plots.plot_replacement_fixture(game_check),
    "12_replacement_fixture",
)
```

<!-- ARENA:cell markdown -->
### The threshold control: complete paths are not discrete Shapley

For

$$
f(x_0,x_1)=\operatorname{ReLU}(2x_0+x_1-2),\qquad x=(1,1),\quad b=(0,0),
$$

the diagonal path becomes active after $\alpha=2/3$. Its Expected Gradients allocation is
`(2/3,1/3)`. The four donor-replacement coalition values are `(0,0,0,1)`, whose exact Shapley
allocation is `(1/2,1/2)`. Both sum to one, and replacing either feature alone drops the output from
one to zero. Completeness and single-feature causal necessity do not force equal credit across these
two games.

### The exact-kink qualification

On nonnegative inputs,

$$
\min(x_0,x_1)=x_1-\operatorname{ReLU}(x_1-x_0)
=x_0-\operatorname{ReLU}(x_0-x_1).
$$

Along the path from `(0,0)` to `(1,1)`, both ReLU arguments stay exactly at the nondifferentiable
kink. PyTorch's zero derivative assigns `(0,1)` to the first graph and `(1,0)` to the mirrored
graph, while the discrete game gives `(1/2,1/2)`. This does not refute almost-everywhere
implementation invariance away from kinks. It exposes that an exact nondifferentiable path needs a
subgradient convention, and equivalent graphs can encode different conventions there. All four
binary inputs agree in output, and each feature is necessary at `(1,1)`.

<!-- ARENA:cell code -->
```python
methods = SimpleNamespace(
    sample_path_schedule=sample_path_schedule,
    interpolate_paths=interpolate_paths,
    input_gradients=input_gradients,
    expected_gradient_samples=expected_gradient_samples,
    summarize_expected_gradients=summarize_expected_gradients,
    expected_gradients=expected_gradients,
    gauss_legendre_pairs=gauss_legendre_pairs,
    noisy_expected_gradient_samples=noisy_expected_gradient_samples,
    ranked_coalition_curves=ranked_coalition_curves,
)
```

<!-- ARENA:cell markdown -->
### Try It Yourself 1 — move a path through the gate

Edit the query, donor, weights, bias, alpha schedule, and sample count. Recompute the path-gradient
samples, quadrature control, exact four-coalition Shapley, and both single-feature replacements.
Avoid placing a hand-derived test alpha exactly on the ReLU kink.

<!-- ARENA:cell code -->
```python
# ARENA:insert gradientshap_diagnostics.py:threshold_play
```

<!-- ARENA:cell code -->
```python
tiny_play = threshold_play(
    methods,
    weights=(2.0, 1.0),
    bias=-2.0,
    query=(1.0, 1.0),
    reference=(0.0, 0.0),
    samples=512,
    seed=161510,
)
show_figure(plots.plot_threshold_play(tiny_play), "13_threshold_play")
```

<!-- ARENA:cell markdown -->
<details><summary>Ideas to try</summary>

- Make both weights equal and check which symmetries the query and reference must also satisfy.
- Move the bias until a finite schedule has only active or only inactive draws.
- Double the sample count while preserving the old prefix; compare the actual error with the SE.
- Put the path exactly on a kink, then perturb one coordinate by `1e-8` and inspect the derivative
  convention.

</details>

<!-- ARENA:cell markdown -->
## The frozen-design real study

> **Execution boundary.** The independently reviewed zero-run freeze binds `protocol.md`, the
> manifest, and the data implementation before fitting. The cells below reproduce all 21 section
> 16.14 state hashes, persist the fitted and prepared study before the literal audit load, and then
> evaluate every held-out recipe. This ordering keeps explanation-side choices separate from the
> audit evidence.

The intended study holds the predictors and data fixed so the attribution method is the variable:

- the same 1,030 UCI Concrete rows and recipe-disjoint 758/153/119 split as section 16.14;
- the same three clean and three shuffled-target `8 -> 32 -> 32 -> 1` ReLU functions plus ridge;
- raw feature inputs, raw MPa outputs, and state-hash parity with the section 16.14 predictors;
- all seven models and all 272 held-out rows, with no selected-error subset;
- the same primary 64 intact donors, nested sizes 16 and 128, and two independent 64-row designs;
- nested Monte Carlo budgets `(8,32,128,512,2048)` on one literal row-stable schedule;
- 256- and 512-node all-background quadrature retained separately;
- exact 256-coalition Shapley, DeepSHAP, and endpoint gradient-times-delta as distinct comparisons;
- Gaussian scales `(0,0.02,0.10)` in train-standardized input units, mapped back to raw units;
- global feature orders learned from 153 explanation rows before audit values are loaded;
- 100 fixed random feature orders for the 119-row audit replacement experiment.

The reused predictors isolate method behavior but do not constitute an independent model or dataset
replication. The frozen protocol refits them from source, verifies all 21 initial, standardized, and
raw-folded state hashes against section 16.14, and stops before attribution if any identity differs.

<!-- ARENA:cell markdown -->
The visible study source will be split into short cells for state reconstruction, schedule creation,
path evaluation, quadrature, exact-game comparisons, background/noise evaluation, and replacement
curves. Mechanical schema validation may be imported; the scientific call flow and every meaningful
transformation must remain visible.

<!-- ARENA:cell code -->
```python
from part14_deepshap import deepshap_methods as deep_methods
from part15_gradientshap.gradientshap_study import (
    _batched_prediction,
    _checked_curves,
    _clone_state,
    _combine_baselines,
    _combine_mc,
    _combine_primary,
    _combine_quadrature,
    _fixed_random_orders,
    _float_tensor,
    _folding_error,
    _identity_hash,
    _ids_tensor,
    _normalization_tensors,
    _output_tensor,
    _require_disjoint_ids,
    _require_namespace,
    _rows,
    _seed_words,
    _state_shapes,
    _validate_evaluation_inputs,
    _validate_fitted,
)
```

<!-- ARENA:cell code -->
```python
# ARENA:insert gradientshap_study.py:protocol_constants
```

<!-- ARENA:cell code -->
```python
# ARENA:insert gradientshap_study.py:study_metadata
```

<!-- ARENA:cell markdown -->
The data loader supplies raw rows and identities. The next function reconstructs a supported raw
`nn.Sequential` from a retained state dictionary; normalization is folded into the model weights,
so every later attribution stays in raw feature and raw MPa units.

<!-- ARENA:cell code -->
```python
# ARENA:insert gradientshap_study.py:model_reconstruction
```

<!-- ARENA:cell markdown -->
### Refit the seven known predictors

Compute normalization only from the 758 training rows. The six neural controls use the exact fixed
seeds and 800 full-batch updates; three receive clean targets and three receive fixed shuffled
targets. Folding is explicit so the standardized and raw-unit functions can be compared before any
attribution is calculated.

<!-- ARENA:cell code -->
```python
# ARENA:insert gradientshap_study.py:training_normalization
```

<!-- ARENA:cell code -->
```python
# ARENA:insert gradientshap_study.py:train_and_fold
```

<!-- ARENA:cell markdown -->
The MLP fit retains its initial state, standardized state, folded raw state, target permutation,
loss trace, and training predictions. Ridge is fitted once by the declared normal-equation solve and
provides the affine path-versus-coalition agreement control.

<!-- ARENA:cell code -->
```python
# ARENA:insert gradientshap_study.py:fit_mlp
```

<!-- ARENA:cell code -->
```python
# ARENA:insert gradientshap_study.py:fit_ridge
```

<!-- ARENA:cell code -->
```python
# ARENA:insert gradientshap_study.py:fit_models
```

<!-- ARENA:cell markdown -->
### Stop unless all 21 model states match section 16.14

The comparison is useful only if GradientSHAP and DeepSHAP explain identical fitted functions. The
hard gate below checks the initial, standardized, and raw-folded state of each of the seven models,
plus training identities, normalization, targets, permutations, loss traces, and predictions. A
mismatch raises before `prepare_study` can draw a path schedule.

<!-- ARENA:cell code -->
```python
# ARENA:insert gradientshap_study.py:verify_states
```

<!-- ARENA:cell markdown -->
### Fix backgrounds and one shared path schedule

All five backgrounds are selected by the frozen source-ID hash rule. Every model receives the same
K=2048 donor and alpha table for each source row, and every smaller budget is a literal prefix.

<!-- ARENA:cell code -->
```python
# ARENA:insert gradientshap_study.py:backgrounds_and_schedule
```

<!-- ARENA:cell markdown -->
### Retain the estimator and its numerical control

`_monte_carlo_block` keeps every contribution draw before calculating nested means, ordinary SE,
and finite completeness residuals. `_quadrature_block` separately keeps every donor-wise 256- and
512-node path integral and their drift; 512 nodes remain a numerical control rather than truth.

<!-- ARENA:cell code -->
```python
# ARENA:insert gradientshap_study.py:monte_carlo_block
```

<!-- ARENA:cell code -->
```python
# ARENA:insert gradientshap_study.py:quadrature_block
```

<!-- ARENA:cell markdown -->
### Compare the continuous path with a discrete replacement game

For eight features we can retain all 256 coalitions and every query/coalition/donor prediction.
The same raw model and donor rows then produce exact feature Shapley, DeepSHAP, and endpoint
gradient-times-delta controls. Their differences are measured outcomes, never a parity gate.

<!-- ARENA:cell code -->
```python
# ARENA:insert gradientshap_study.py:coalition_hybrids
```

<!-- ARENA:cell code -->
```python
# ARENA:insert gradientshap_study.py:comparison_baselines
```

<!-- ARENA:cell markdown -->
Global rankings are fixed by mean absolute attribution over all 153 explanation rows. This happens
before the audit split is loaded and separately for exact Shapley, DeepSHAP, endpoint gradient,
K=2048 Expected Gradients, and 512-node quadrature.

<!-- ARENA:cell code -->
```python
# ARENA:insert gradientshap_study.py:global_rankings
```

<!-- ARENA:cell code -->
```python
# ARENA:insert gradientshap_study.py:prepare_study
```

<!-- ARENA:cell markdown -->
The preparation boundary must persist model states, background identities, path schedules, and
global explanation-split rankings before the literal audit load. Audit values may evaluate those
choices but may not revise them.

<!-- ARENA:cell code -->
```python
from part15_gradientshap.run_native import (
    persist_completed,
    persist_prepared,
    write_notebook_summary,
)

native_directory = FIGURE_DIR / "native"
train = load_split("train")
explain = load_split("explain")
fitted = fit_models(train)
reproduction = verify_16_14_reproduction(fitted, load_manifest())
fitted = dict(fitted)
fitted["reproduction"] = reproduction
prepared = prepare_study(
    fitted,
    train,
    explain,
    methods=methods,
    deep_methods=deep_methods,
)
before_audit = persist_prepared(native_directory, fitted, prepared)

print("16.14 state reproduction:", reproduction["status"])
print(
    "State hashes checked:",
    sum(len(item) for item in reproduction["model_state_hashes"].values()),
)
print("Prepared phase:", prepared["phase"])
print("Audit loaded at this boundary:", before_audit["audit_loaded"])
```

<!-- ARENA:cell code -->
```python
audit = load_split("audit")
```

<!-- ARENA:cell code -->
```python
# ARENA:insert gradientshap_study.py:performance
```

<!-- ARENA:cell markdown -->
The evaluation preserves model quality as a control, then recomputes `clean_0` under all five
background supports. This separates a change in the finite estimand from Monte Carlo error at a
fixed background.

<!-- ARENA:cell code -->
```python
# ARENA:insert gradientshap_study.py:background_evaluation
```

<!-- ARENA:cell markdown -->
The noise branch reuses the clean schedule and one explicit standard-normal table. Sigma zero aliases
the clean contribution tensor; positive sigma is compared with both the clean prediction difference
and the mean supplied-noisy-query prediction difference.

<!-- ARENA:cell code -->
```python
# ARENA:insert gradientshap_study.py:noise_evaluation
```

<!-- ARENA:cell markdown -->
Independent schedules diagnose whether the reported ordinary SE describes repeated Monte Carlo
variation. The fixed 64-row subset and three models are retained for all 64 schedules and all five
budgets; the complete `(64,3,64,2048,8)` contribution tensor is never summarized away.

<!-- ARENA:cell code -->
```python
# ARENA:insert gradientshap_study.py:replicate_evaluation
```

<!-- ARENA:cell markdown -->
Nearest-training-row distance is computed for every predeclared interpolation point and coalition
hybrid after train-only standardization. It is a geometric warning surface, not a density estimate,
physical-feasibility score, or row-selection rule.

<!-- ARENA:cell code -->
```python
# ARENA:insert gradientshap_study.py:distance_evaluation
```

<!-- ARENA:cell markdown -->
Finally, explanation-only feature orders and all 100 random orders traverse the already retained
audit coalition games. Raw MPa insertion and deletion values remain visible even when a curve is
nonmonotone or a random order performs well.

<!-- ARENA:cell code -->
```python
# ARENA:insert gradientshap_study.py:replacement_evaluation
```

<!-- ARENA:cell code -->
```python
# ARENA:insert gradientshap_study.py:evaluate_study
```

<!-- ARENA:cell code -->
```python
# ARENA:insert gradientshap_study.py:native_play
```

<!-- ARENA:cell markdown -->
## Package references answer a narrower question

Installed `shap.GradientExplainer` and `captum.attr.GradientShap` are implementation references,
not the definition of the lesson's scientific schedule. Their public APIs consume random numbers in
different orders, so equal integer seeds do not imply equal donor, alpha, or noise draws. Strict
parity requires reconstructing each package's actual schedule and comparing the student estimator on
those same draws.

For SHAP 0.49.1, the PyTorch implementation needs a state-restoring adapter around its float64
buffer behavior. It seeds NumPy with `rseed`, but nonzero local smoothing also consumes Torch RNG.
Its returned `samples.var(0) / sqrt(K)` field is a package variance proxy: it is neither the ordinary
variance of the sample mean `s²/K` nor the ordinary standard error `s/sqrt(K)` computed in Exercise
5. SHAP's scalar-model attributions and uncertainty also use different output containers.

Captum 0.9.0 draws donors and alphas with NumPy and Gaussian noise with Torch. Its
`return_convergence_delta=True` result contains one delta per repeated noisy sample, not the
residual of the final averaged attribution. Both adapters must restore RNG state, default dtype,
model mode, buffers, and existing parameter gradients even when the author call raises.

<!-- ARENA:cell code -->
```python
from part15_gradientshap.gradientshap_reference import (
    REFERENCE_SAMPLES,
    validate_reference,
)
```

<!-- ARENA:cell code -->
```python
# ARENA:insert gradientshap_reference.py:reference_check
```

<!-- ARENA:cell markdown -->
## Signature result: convergence, target movement, and a different game

The completed notebook generates these visible figures from its fresh own-answer run. Use the
analytic fixtures as controls while reading the measured Concrete results:

1. **Monte Carlo convergence.** Plot median and 10–90% absolute coordinate error against 512-node
   quadrature over all 1,904 model-row explanations at every nested budget. Overlay thin per-model
   traces and ordinary SE calibration. Do not assume a universal `K` from the paper.
2. **Complete yet different.** Plot all 15,232 signed Expected Gradients coordinates against exact
   feature Shapley, with completeness residuals in a separate panel. The threshold fixture predicts
   disagreement, while the all-row panel shows its scale and structure on the trained models.
3. **Background movement.** For `clean_0` and every held-out row, separate deterministic
   all-background quadrature movement from K=2048 sampling error across five declared backgrounds.
4. **Gaussian target shift.** Compare attribution sums with both the clean prediction difference and
   the mean supplied-noisy-query prediction difference. At nonzero sigma, only the latter is the
   noisy estimator's target.
5. **Computational intervention.** Apply explanation-only global orders to all audit rows and show
   raw MPa insertion/deletion curves beside all 100 random orders. No curve ranking is an acceptance
   gate for attribution exactness.
6. **Path geometry.** Compare nearest-training-row distances of interpolation points, coalition
   hybrids, and observed endpoints after train-only standardization. This is an OOD distance
   diagnostic, not a density estimate or causal validity proof.

<!-- ARENA:cell code -->
```python
evaluated = evaluate_study(
    fitted,
    prepared,
    train,
    explain,
    audit,
    methods=methods,
    deep_methods=deep_methods,
)
models = {
    key: _model_from_state(record)
    for key, record in fitted["models"].items()
}
reference = reference_check(
    models,
    evaluated["all_row_x_raw"],
    prepared["backgrounds"]["primary64"]["x_raw"],
    evaluated["query_source_ids"],
    methods,
)
diagnostics = run_diagnostics(methods=methods)
fixed_play = native_play(
    fitted,
    prepared,
    evaluated,
    train,
    explain,
    audit,
    methods=methods,
    deep_methods=deep_methods,
)
summary = persist_completed(
    native_directory,
    fitted,
    prepared,
    evaluated,
    diagnostics,
    reference,
    fixed_play,
    timings={},
)
write_notebook_summary(FIGURE_DIR, native_directory, summary)

print("Reference parity:", reference["status"])
print("Package vectors checked:", reference["counts"]["package_vectors"])
print("All model-row explanations:", summary["all_model_rows"])
print("Audit used for selection:", summary["audit_used_for_selection"])
```

<!-- ARENA:cell code -->
```python
show_figure(
    plots.plot_native_overview(prepared, evaluated),
    "14_native_overview",
)
show_figure(
    plots.plot_allrow_method_map(evaluated),
    "15_allrow_method_map",
)
```

<!-- ARENA:cell code -->
```python
show_figure(
    plots.plot_fixed_waterfall(fixed_play, feature_names=FEATURE_NAMES),
    "16_fixed_waterfall",
)
show_figure(
    plots.plot_background_and_noise(evaluated),
    "17_background_and_noise",
)
```

<!-- ARENA:cell code -->
```python
show_figure(
    plots.plot_replicates_and_distance(evaluated),
    "18_replicates_and_distance",
)
show_figure(
    plots.plot_replacement_curves(evaluated, model_key="clean_0"),
    "19_replacement_curves",
)
```

<!-- ARENA:cell markdown -->
### Try It Yourself 2 — choose a recipe, background, budget, and noise scale

The canonical first run must use the first audit recipe, `clean_0`, the primary 64-row background,
K=2048, and sigma zero. After reproducing that fixed display, edit the model, any held-out source
row, one of the five background designs, a nested K budget, or a predeclared sigma. Recompute the
clean/noisy samples, quadrature control, exact feature game, and local replacement curves rather than
slicing cached arrays.

<!-- ARENA:cell code -->
```python
real_play = native_play(
    fitted,
    prepared,
    evaluated,
    train,
    explain,
    audit,
    methods=methods,
    deep_methods=deep_methods,
    model_key="clean_0",
    split="audit",
    row_index=0,
    background_key="primary64",
    samples=2048,
    sigma=0.0,
)
show_figure(
    plots.plot_real_recipe_play(real_play, feature_names=FEATURE_NAMES),
    "20_real_recipe_play",
)
```

<!-- ARENA:cell markdown -->
<details><summary>Ideas to try</summary>

- Hold model and row fixed while moving from nested 16 to primary 64 to nested 128 donors.
- Compare sigma zero with `0.02` and `0.10`; read the clean and smoothed residuals separately.
- Find a row whose EG and exact global orders agree while their signed magnitudes differ.
- Compare K=32 and K=2048 on the same literal prefix, then inspect whether the ordinary SE predicted
  the observed change.
- Compare a clean and shuffled model on the same recipe before interpreting either explanation.

These choices occur after audit evaluation. They are exploratory diagnostics and cannot support a
new confirmatory audit claim.

</details>

<!-- ARENA:cell markdown -->
## How to interpret the completed study

Read the eventual figures in this order:

1. Require clean predictors to beat the training-mean and shuffled-target controls before discussing
   attribution quality.
2. Validate the nine methods on analytic fixtures and require strict same-draw author parity. Keep
   that implementation check separate from convergence to the path target.
3. Compare nested Monte Carlo estimates with quadrature doubling and ordinary SE. A small SE with a
   large 256-to-512 change points to integration-control limits; a large SE is sampling instability.
4. Compare completeness residuals with signed coordinate differences. A path method may converge
   and be complete while disagreeing with exact feature-coalition Shapley.
5. Read background movement beside sampling error. Alternate empirical supports ask different
   questions; they are not replicate confidence intervals for one population target.
6. At nonzero noise, compare with the smoothed-query prediction. A clean-target discrepancy can be
   the intended target shift rather than failed local accuracy.
7. Read global audit replacement curves beside random orders and all-row variability. A favorable
   early prefix does not establish exact attribution or physical causality.

## Limitations and method scope

Expected Gradients uses straight paths and ordinary derivatives. Exact ReLU kinks require a
subgradient convention, finite Monte Carlo estimates have sampling error, and finite quadrature is
approximate for gradient jumps. Background choice is part of the estimand. Convex interpolation,
Gaussian perturbation, and feature-coalition replacement can create inputs far from observed
recipes.

The Concrete background is empirical and observational. Replacing cement, water, aggregate, or age
inside a predictor is a computational intervention on its input. It is not a randomized physical
experiment or evidence that the attribution is a causal effect in concrete production.

For classification, explaining a logit, probability, or loss would define different scalar
functions. This study avoids that ambiguity by explaining raw MPa. It does not add attribution-prior
training: changing the training objective would require a new pre-fit protocol, lambda selection,
performance guardrails, and repeated training evidence.

## Further research

- Compare independent schedule replicates with nominal `estimate ± 1.96 SE` coverage on a
  predeclared coordinate set without treating coordinates as independent experiments.
- Test antithetic or stratified alpha sampling as optional variance-reduction estimators, reporting
  measured variance instead of assuming improvement.
- Split ReLU paths analytically at known activation kinks and compare exact piecewise integration
  with Gauss-Legendre drift.
- Replace isotropic Gaussian noise with a declared domain-aware perturbation and derive its new
  smoothed target before measuring it.
- Add attribution-prior training only under a separately frozen study that retains predictive
  performance and repeated-seed controls.

## Reading

- Erion et al., [*Improving performance of deep learning models with axiomatic attribution priors
  and expected gradients*](https://arxiv.org/abs/1906.10670).
- Sundararajan, Taly and Yan,
  [*Axiomatic Attribution for Deep Networks*](https://proceedings.mlr.press/v70/sundararajan17a.html).
- Lundberg and Lee,
  [*A Unified Approach to Interpreting Model Predictions*](https://arxiv.org/abs/1705.07874).
- Official SHAP
  [`GradientExplainer` documentation](https://shap.readthedocs.io/en/stable/generated/shap.GradientExplainer.html).
- Official Captum
  [`GradientShap` documentation](https://captum.ai/api/gradient_shap.html).
