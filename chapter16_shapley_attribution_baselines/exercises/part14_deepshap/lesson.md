<!-- ARENA:cell markdown -->
# [16.14] DeepSHAP: When Does Fast Attribution Match Exact SHAP?

> **Claim.** A DeepSHAP explanation can reconstruct a prediction exactly while assigning different
  feature credit from exact Shapley values for the same empirical background game.

## Core Question

How does DeepSHAP replace ordinary derivatives with finite-change multipliers, and which controls
tell us whether its fast compositional explanation matches the exact feature game?

Consider

$$
f(x_0,x_1)=\operatorname{ReLU}(2x_0+x_1-2),\qquad x=(1,1),\quad b=(0,0).
$$

Neither feature can cross the threshold alone. The four intervention values are `(0, 0, 0, 1)`,
so exact Shapley splits the one-unit change evenly: `(0.5, 0.5)`. DeepSHAP's Rescale rule follows
the active full computation and divides the same change in proportion to the incoming linear
differences: `(2/3, 1/3)`. Both vectors sum to one. Completeness alone cannot decide which credit
allocation answers the feature-level game.

In this lab you will derive that result, implement the complete modified backward pass, and compare
it with all 256 coalitions on held-out concrete-strength predictions. The full lab runs on CPU in
float64. Allow **4–6 hours**; Exercises 2–5 contain the core DeepSHAP mechanism.

## Learning objectives

By the end of the lab, you should be able to:

- distinguish an ordinary derivative from a finite-change multiplier;
- derive ReLU's Rescale secant and the modified chain rule;
- explain why background averaging means averaging attributions, not replacing many references by
  their mean input;
- separate completeness, independent-library parity, and exact feature-Shapley agreement;
- enumerate the empirical donor-replacement game and contract it into exact Shapley values;
- use threshold, cancellation, affine, dummy, constant, epsilon, shuffled-target and random-order
  controls;
- interpret observational feature replacement without calling it a physical intervention.

## Setup

The taught method supports an exact `nn.Sequential` chain containing distinct, non-inplace
`nn.Linear` and `nn.ReLU` modules and a scalar final output. This narrow contract keeps every
modified backward rule visible. Real DeepSHAP libraries support more operators through
backend-specific handlers; an unsupported module here raises an error rather than silently using an
ordinary gradient.

<!-- ARENA:cell code -->
```python
import itertools
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch as t
from IPython.display import display
from torch import Tensor, nn

torch.set_default_dtype(torch.float64)

chapter = "chapter16_shapley_attribution_baselines"
section = "part14_deepshap"
root_dir = next(path for path in (Path.cwd(), *Path.cwd().parents) if (path / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

from part14_deepshap import deepshap_methods_tests as tests
from part14_deepshap import deepshap_plots as plots
from part14_deepshap.deepshap_data import load_split

FEATURE_NAMES = np.array([
    "Cement", "Slag", "Fly ash", "Water", "Superplasticizer",
    "Coarse aggregate", "Fine aggregate", "Age",
])
RUN_OUTPUT = os.environ.get("ARENA_DEEPSHAP_OUTPUT_DIRECTORY")
FIGURE_DIR = Path(
    os.environ.get("ARENA_DEEPSHAP_FIGURE_DIR", RUN_OUTPUT or section_dir / "figures")
)
SAVE_FIGURES = os.environ.get(
    "ARENA_DEEPSHAP_SAVE_FIGURES", "1" if RUN_OUTPUT else "0"
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
The validators below are provided plumbing. Exact module types matter because a subclass can
change its forward computation, and repeated module objects would make saved states ambiguous.
The nine attribution functions remain in the exercises, where you can read and edit them.

<!-- ARENA:cell code -->
```python
from part14_deepshap.deepshap_methods import (
    _check_widths,
    _epsilon,
    _layers,
    _method_tensor,
    _model_output,
)
```

<!-- ARENA:cell markdown -->
The following diagnostic helpers build literal float64 layers and resolve only methods you have
already written. They contain no attribution implementation.

<!-- ARENA:cell code -->
```python
from part14_deepshap.deepshap_diagnostics import (
    DTYPE,
    _linear,
    _max_abs,
    _method,
    _prediction,
)
```

<!-- ARENA:cell markdown -->
## 1️⃣ Save paired forward states

DeepSHAP compares a real input `x` with a reference `b`. The backward rule for a ReLU needs both
preactivation values, so our forward pass stores the input and the output of every layer for both
batches. If the model has $L$ layers, each returned list has $L+1$ entries.

### Exercise 1 — `paired_forward`

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵⚪⚪
>
> You should spend up to 10-15 minutes on this exercise.
> ```

Return `(x_states, b_states)`. Do not combine the batches or pair their rows: `x` has shape `(N,D)`
and `background` has shape `(B,D)`.

<!-- ARENA:cell code -->
```python
def paired_forward(model, x, background):
    """Return input and per-layer states for x and every background row."""
    raise NotImplementedError()


tests.test_paired_forward(paired_forward)
```

<!-- ARENA:cell markdown -->
<details><summary>Help - preserve the two batch axes</summary>

Validate both tensors and start `x_states = [x]`, `b_states = [background]`. Apply each layer to
the last state in its own list. You need every layer output, including the final scalar output.

</details>

<details><summary>Expected output</summary>

```text
All tests in test_paired_forward passed!
```

</details>

<details><summary>Interpretation</summary>

Biases have no player of their own, but they can move one member of a pair across a ReLU boundary.
They therefore affect the saved preactivations and the multiplier derived from them.

</details>

<details><summary>Solution</summary>

```python
def paired_forward(
    model: nn.Sequential, x: Tensor, background: Tensor
) -> tuple[list[Tensor], list[Tensor]]:
    """Save the input and every layer output for queries and references."""
    layers = _layers(model)
    _method_tensor("x", x, 2)
    _method_tensor("background", background, 2)
    if x.shape[1] != background.shape[1]:
        raise ValueError("x and background must have the same feature dimension.")
    _check_widths(layers, x.shape[1])

    with t.no_grad():
        x_value, b_value = x.clone(), background.clone()
        x_states, b_states = [x_value], [b_value]
        for layer in layers:
            x_value, b_value = layer(x_value), layer(b_value)
            x_states.append(x_value)
            b_states.append(b_value)
        if x_states[-1].shape != (len(x), 1) or b_states[-1].shape != (len(background), 1):
            raise ValueError("The model must return one scalar output per input row.")
        if not t.isfinite(x_states[-1]).all() or not t.isfinite(b_states[-1]).all():
            raise ValueError("The model returned a nonfinite value.")
    return x_states, b_states
```

</details>

<!-- ARENA:cell markdown -->
## 2️⃣ Replace a ReLU derivative with a secant

For a unary activation $g$, Rescale assigns the finite-change multiplier

$$
s(z_x,z_b)=\frac{g(z_x)-g(z_b)}{z_x-z_b}.
$$

For ReLU, this slope is 0 if both values are non-positive, 1 if both are positive, and lies between
0 and 1 when the pair crosses zero. If $|z_x-z_b|<\epsilon$, the quotient is numerically unstable,
so our explicit convention uses the derivative at the reference: $\mathbb{1}[z_b>0]$. This matches
the supported library rule. The strict `>` also states the convention at exactly zero.

### Exercise 2 — `rescale_slope`

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵🔵🔵
>
> You should spend up to 15-20 minutes on this exercise.
> ```

Implement broadcast-compatible ReLU secants. Guard the denominator *before* division; computing an
invalid quotient and masking it later can still introduce `nan` values or warnings.

<!-- ARENA:cell code -->
```python
def rescale_slope(zx, zb, epsilon=1e-6):
    """Return ReLU finite-change slopes, using the reference derivative near zero."""
    raise NotImplementedError()


tests.test_rescale_slope(rescale_slope)
```

<!-- ARENA:cell markdown -->
<details><summary>Help - construct a safe denominator</summary>

Let `delta = zx - zb` and `near = abs(delta) < epsilon` (also include exact equality when
`epsilon=0`). Replace near-zero denominators by ones for the quotient, then select the reference
derivative with `torch.where`.

</details>

<details><summary>Common bug - testing only completeness</summary>

The threshold gate's DeepSHAP vector `(2/3, 1/3)` and exact vector `(1/2, 1/2)` both sum to one.
Checking only the sum would accept the wrong answer to the exact feature game. Keep completeness,
independent implementation parity, and exact Shapley error as three separate checks.

</details>

<details><summary>Expected output</summary>

```text
All tests in test_rescale_slope passed!
```

</details>

<details><summary>Interpretation</summary>

With `epsilon=1e-6`, `x=4e-7` and `b=-4e-7`, the fallback slope is zero and the attributed change is
zero even though ReLU changes by `4e-7`. This expected local-accuracy failure is the price of the
finite numerical cutoff. Setting `epsilon=0` recovers the exact nonzero secant.

</details>

<details><summary>Solution</summary>

```python
def rescale_slope(zx: Tensor, zb: Tensor, epsilon: float = 1e-6) -> Tensor:
    """ReLU secants, using the reference derivative for small denominators."""
    _method_tensor("zx", zx)
    _method_tensor("zb", zb)
    epsilon = _epsilon(epsilon)
    try:
        zx, zb = t.broadcast_tensors(zx, zb)
    except RuntimeError as error:
        raise ValueError("zx and zb must be broadcast-compatible.") from error

    delta = zx - zb
    use_derivative = delta.eq(0) | delta.abs().lt(epsilon)
    safe_delta = t.where(use_derivative, t.ones_like(delta), delta)
    secant = (t.relu(zx) - t.relu(zb)) / safe_delta
    return t.where(use_derivative, (zb > 0).to(zb.dtype), secant)
```

</details>

<!-- ARENA:cell markdown -->
The first plot checks the secant away from zero and then magnifies the explicit epsilon boundary.
The visible diagnostic source calls the function you just wrote.

<!-- ARENA:cell code -->
```python
# ARENA:insert deepshap_diagnostics.py:rescale_limits
```

<!-- ARENA:cell code -->
```python
rescale_limits = diagnose_rescale_limits(globals())
show_figure(plots.plot_rescale_limits(rescale_limits), "01_rescale_limits")
```

<!-- ARENA:cell markdown -->
## 3️⃣ Propagate modified multipliers

Ordinary backpropagation composes local derivatives. DeepLIFT instead starts from contributions
$C_{\Delta x_i\Delta t}$ and defines a multiplier

$$
m_{\Delta x_i\Delta t}=\frac{C_{\Delta x_i\Delta t}}{\Delta x_i}.
$$

For an intermediate layer $y_j$, the modified chain rule is

$$
m_{\Delta x_i\Delta t}=\sum_j
m_{\Delta x_i\Delta y_j}\,m_{\Delta y_j\Delta t}.
$$

An affine layer uses its weight matrix. A ReLU uses the secant you just implemented. Starting from
a unit output multiplier and traversing the modules in reverse therefore looks like backpropagation,
but every ReLU factor depends on the pair `(x,b)` rather than one derivative at `x`.

### Exercise 3 — `rescale_multipliers`

> ```yaml
> Difficulty: 🔴🔴🔴🔴⚪
> Importance: 🔵🔵🔵🔵🔵
>
> You should spend up to 35-50 minutes on this exercise.
> ```

Return one output-to-input multiplier for every input/reference pair, shape `(N,B,D)`.

<!-- ARENA:cell code -->
```python
def rescale_multipliers(model, x, background, epsilon=1e-6):
    """Backpropagate Rescale multipliers through a supported scalar Sequential model."""
    raise NotImplementedError()


tests.test_rescale_multipliers(rescale_multipliers)
```

<!-- ARENA:cell markdown -->
<details><summary>Help - reverse the saved states</summary>

Call `paired_forward`. Initialize `multiplier` to ones with shape `(N,B,1)`. For layer index `i` in
reverse, an `nn.Linear` updates `multiplier @ layer.weight`. For an `nn.ReLU`, form its input pair
from `x_states[i][:,None,:]` and `b_states[i][None,:,:]`, then multiply by `rescale_slope`.

</details>

<details><summary>Expected output</summary>

```text
All tests in test_rescale_multipliers passed!
```

</details>

<details><summary>Interpretation</summary>

The output is not yet an attribution. It says how much output difference is assigned per unit input
difference for each pair. Exercise 4 multiplies it by the actual displacement `x-b`.

</details>

<details><summary>Solution</summary>

```python
@t.no_grad()
def rescale_multipliers(
    model: nn.Sequential,
    x: Tensor,
    background: Tensor,
    epsilon: float = 1e-6,
) -> Tensor:
    """Propagate scalar-output Rescale multipliers to every input feature."""
    layers = _layers(model)
    epsilon = _epsilon(epsilon)
    x_states, b_states = paired_forward(model, x, background)
    multiplier = t.ones((len(x), len(background), 1), dtype=t.float64)

    for index in range(len(layers) - 1, -1, -1):
        layer = layers[index]
        if type(layer) is nn.Linear:
            multiplier = t.einsum("nro,oi->nri", multiplier, layer.weight)
        else:
            slope = rescale_slope(
                x_states[index][:, None, :],
                b_states[index][None, :, :],
                epsilon,
            )
            multiplier = multiplier * slope
    return multiplier
```

</details>
<!-- ARENA:cell markdown -->
## 4️⃣ Turn multipliers into single-reference contributions

For one reference $b$, Rescale contributions are

$$
\phi_i^{(b)}(x)=m_{\Delta x_i\Delta f}(x,b)(x_i-b_i).
$$

On a supported affine/ReLU chain with an exact secant at every changed ReLU, the modified chain rule
gives the summation-to-delta property

$$
\sum_i\phi_i^{(b)}(x)=f(x)-f(b).
$$

That identity is **local completeness**. The threshold example already proves that it does not imply
exact Shapley allocation over the original input features.

### Exercise 4 — `single_reference`

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 15-20 minutes on this exercise.
> ```

Accept a one-dimensional reference and return `(N,D)` contributions.

<!-- ARENA:cell code -->
```python
def single_reference(model, x, reference, epsilon=1e-6):
    """Explain each row relative to one reference row."""
    raise NotImplementedError()


tests.test_single_reference(single_reference)
```

<!-- ARENA:cell markdown -->
<details><summary>Help - keep the reference axis until the end</summary>

Convert the reference to shape `(1,D)`, call `rescale_multipliers`, select its length-one reference
axis, and multiply by `x - reference`.

</details>

<details><summary>Expected output</summary>

```text
All tests in test_single_reference passed!
```

The fixed threshold case must return `(2/3, 1/3)`, while an affine function
`2*x0 - 3*x1 + 5` at `x=(4,2)`, `b=(1,-1)` returns `(6,-9)`.

</details>

<details><summary>Interpretation</summary>

The affine case is exact because the same linear weights apply to every coalition and every
reference. The threshold network is complete but not exact for the feature game. Reporting these as
separate assertions prevents an easy but serious validation error.

</details>

<details><summary>Solution</summary>

```python
@t.no_grad()
def single_reference(
    model: nn.Sequential,
    x: Tensor,
    reference: Tensor,
    epsilon: float = 1e-6,
) -> Tensor:
    """Explain each query relative to one complete reference vector."""
    _method_tensor("x", x, 2)
    _method_tensor("reference", reference, 1)
    if x.shape[1] != reference.shape[0]:
        raise ValueError("reference must contain one value for every input feature.")
    multiplier = rescale_multipliers(model, x, reference[None, :], epsilon)[:, 0]
    return multiplier * (x - reference)
```

</details>

<!-- ARENA:cell code -->
```python
# ARENA:insert deepshap_diagnostics.py:near_zero_fixture
```

<!-- ARENA:cell code -->
```python
# ARENA:insert deepshap_diagnostics.py:near_zero
```

<!-- ARENA:cell code -->
```python
near_zero = diagnose_near_zero(globals())
show_figure(plots.plot_near_zero(near_zero), "02_near_zero_epsilon")
```

<!-- ARENA:cell markdown -->
The near-zero panel is intentionally a failed case at the primary epsilon. Its residual is shown at
its true scale, beside the exact nonzero-secant result at `epsilon=0`.

<!-- ARENA:cell code -->
```python
# ARENA:insert deepshap_diagnostics.py:affine_fixture
```

<!-- ARENA:cell code -->
```python
# ARENA:insert deepshap_diagnostics.py:affine_controls
```

<!-- ARENA:cell code -->
```python
affine_controls = diagnose_affine_controls(globals())
show_figure(plots.plot_affine_controls(affine_controls), "03_affine_controls")
```

<!-- ARENA:cell markdown -->
An affine network is the exactness control: modified multipliers, ordinary gradients and exact
Shapley must coincide, while a zero-weight dummy input must receive zero credit.

<!-- ARENA:cell markdown -->
## 5️⃣ Average explanations over a background

A single reference makes the answer depend on one baseline. DeepSHAP uses a finite empirical
background $B$:

$$
\phi_i(x;B)=\frac1{|B|}\sum_{b\in B}\phi_i^{(b)}(x),\qquad
\phi_0(B)=\frac1{|B|}\sum_{b\in B}f(b).
$$

When the per-reference rule is complete,

$$
\phi_0(B)+\sum_i\phi_i(x;B)=f(x).
$$

The average is outside the attribution computation. In general,

$$
\frac1{|B|}\sum_b \phi(x;b) \ne \phi\!\left(x;\frac1{|B|}\sum_b b\right)
$$

because a mean reference can cross different ReLU regions and is not one of the intact donor rows.

### Exercise 5 — `background_deepshap`

> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵🔵🔵🔵
>
> You should spend up to 20-30 minutes on this exercise.
> ```

Return `(base, phi)`, where `base` is a scalar tensor and `phi` has shape `(N,D)`.

<!-- ARENA:cell code -->
```python
def background_deepshap(model, x, background, epsilon=1e-6):
    """Average pairwise Rescale contributions over intact background rows."""
    raise NotImplementedError()


tests.test_background_deepshap(background_deepshap)
```

<!-- ARENA:cell markdown -->
<details><summary>Help - average pairwise contributions</summary>

`rescale_multipliers` returns `(N,B,D)`. Multiply by
`x[:,None,:] - background[None,:,:]`, then average only dimension 1. Compute the base from
`model(background)`, not from `model(background.mean(0))`.

</details>

<details><summary>Common bug - explaining the mean reference</summary>

Replacing the whole background by its mean is valid only for special models such as an affine map.
For a nonlinear network it changes the reference game. Preserve every donor through the forward and
modified backward passes, then average the resulting attribution vectors.

</details>

<details><summary>Expected output</summary>

```text
All tests in test_background_deepshap passed!
```

For one-dimensional `ReLU(x)`, `x=2` and background `{-2,1}`, the base is `0.5` and the averaged
attribution is `1.5`. Explaining from the mean reference `-0.5` would instead return `2.0` around a
base of zero.

</details>

<details><summary>Interpretation</summary>

Reversing the rows of a fixed background cannot change the result. Changing which rows are in the
background can change both the base and the contributions; that is a change in the question, not
randomness inside this deterministic implementation.

</details>

<details><summary>Solution</summary>

```python
@t.no_grad()
def background_deepshap(
    model: nn.Sequential,
    x: Tensor,
    background: Tensor,
    epsilon: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    """Average complete reference explanations and return E[f(background)]."""
    multiplier = rescale_multipliers(model, x, background, epsilon)
    pair_values = multiplier * (x[:, None, :] - background[None, :, :])
    with t.no_grad():
        base = _model_output(model, background).mean()
    return base, pair_values.mean(dim=1)
```

</details>

<!-- ARENA:cell code -->
```python
# ARENA:insert deepshap_diagnostics.py:background_average_fixture
```

<!-- ARENA:cell code -->
```python
# ARENA:insert deepshap_diagnostics.py:background_average
```

<!-- ARENA:cell code -->
```python
background_average = diagnose_background_average(globals())
show_figure(plots.plot_background_average(background_average), "04_background_average")
```

<!-- ARENA:cell markdown -->
The control compares the correct average of pairwise explanations with the explanation relative to
the mean input. Their difference is visible precisely when the network is nonlinear across those
reference regions.

<!-- ARENA:cell markdown -->
## 6️⃣ Define the exact empirical replacement game

DeepSHAP is fast because it composes local rules. To measure the approximation, we need an oracle
for the same finite background. For coalition $S$ and intact donor row $b$, define the hybrid

$$
h_i(x,b,S)=
\begin{cases}
x_i & i\in S,\\
b_i & i\notin S,
\end{cases}
\qquad
v_x(S)=\frac1{|B|}\sum_{b\in B} f(h(x,b,S)).
$$

Keeping each donor intact preserves correlations *within* a background row. It does not estimate a
conditional distribution for missing features. With eight features, evaluating all $2^8=256$
coalitions is affordable and gives a direct oracle for this explicitly stated marginal game.

<!-- ARENA:cell code -->
```python
# ARENA:insert deepshap_methods.py:coalition_masks
```

<!-- ARENA:cell markdown -->
### Exercise 6 — `coalition_values`

> ```yaml
> Difficulty: 🔴🔴🔴🔴⚪
> Importance: 🔵🔵🔵🔵🔵
>
> You should spend up to 35-50 minutes on this exercise.
> ```

Return `(N,K)` coalition values for masks `(K,D)`. Batch both source rows and flattened model
forwards. Never average donor inputs before the model call.

<!-- ARENA:cell code -->
```python
def coalition_values(model, x, background, masks, row_batch=8, forward_batch=65536):
    """Evaluate every intact-donor hybrid and average predictions over donors."""
    raise NotImplementedError()


tests.test_coalition_values(coalition_values)
```

<!-- ARENA:cell markdown -->
<details><summary>Help - four useful axes</summary>

For a row chunk, construct hybrids with shape `(rows,K,B,D)` using
`torch.where(masks[None,:,None,:], x_chunk[:,None,None,:], background[None,None,:,:])`.
Flatten only for model evaluation, restore the four logical axes, then average the donor axis.

</details>

<details><summary>Expected output</summary>

```text
All tests in test_coalition_values passed!
```

The empty coalition equals the mean background prediction; the full coalition equals `f(x)` for
every source row.

</details>

<details><summary>Interpretation</summary>

`model(background.mean(0))` and `mean(model(background))` generally differ. The same nonlinearity is
why averaging donor inputs before constructing hybrids would define a different game.

</details>

<details><summary>Solution</summary>

```python
def coalition_values(
    model: nn.Sequential,
    x: Tensor,
    background: Tensor,
    masks: Tensor,
    row_batch: int = 8,
    forward_batch: int = 65_536,
) -> Tensor:
    """Evaluate every supplied intact-donor hybrid, then average over donors."""
    layers = _layers(model)
    _method_tensor("x", x, 2)
    _method_tensor("background", background, 2)
    if x.shape[1] != background.shape[1]:
        raise ValueError("x and background must have the same feature dimension.")
    _check_widths(layers, x.shape[1])
    if type(masks) is not Tensor or masks.device.type != "cpu" or masks.dtype != t.bool:
        raise TypeError("masks must be a CPU Boolean tensor.")
    if masks.ndim != 2 or masks.shape[1] != x.shape[1] or len(masks) == 0:
        raise ValueError("masks must have shape (nonempty coalitions, input features).")
    for name, size in (("row_batch", row_batch), ("forward_batch", forward_batch)):
        if type(size) is not int or size < 1:
            raise ValueError(f"{name} must be a positive integer.")

    result = t.empty((len(x), len(masks)), dtype=t.float64)
    with t.no_grad():
        for start in range(0, len(x), row_batch):
            query = x[start : start + row_batch]
            hybrids = t.where(
                masks[None, :, None, :],
                query[:, None, None, :],
                background[None, None, :, :],
            )
            flat = hybrids.reshape(-1, x.shape[1])
            predictions = t.cat(
                [
                    _model_output(model, flat[i : i + forward_batch])
                    for i in range(0, len(flat), forward_batch)
                ]
            )
            result[start : start + len(query)] = predictions.reshape(
                len(query), len(masks), len(background)
            ).mean(dim=2)
    return result
```

</details>

<!-- ARENA:cell markdown -->
## 7️⃣ Contract every coalition into exact Shapley values

For $D$ features and feature $i$, exact Shapley is

$$
\phi_i=\sum_{S\subseteq [D]\setminus\{i\}}
\frac{|S|!(D-|S|-1)!}{D!}\left[v(S\cup\{i\})-v(S)\right].
$$

This is exact for the table it receives. It does not make the donor-replacement game conditional or
causal; those are properties of how `coalition_values` defined $v$, not of this contraction.

### Exercise 7 — `exact_shapley`

> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵🔵🔵🔵
>
> You should spend up to 25-35 minutes on this exercise.
> ```

Accept any leading batch shape and require the last dimension to be a power of two in bit order.

<!-- ARENA:cell code -->
```python
def exact_shapley(values):
    """Contract a complete bit-ordered coalition table along its last axis."""
    raise NotImplementedError()


tests.test_exact_shapley(exact_shapley)
```

<!-- ARENA:cell markdown -->
<details><summary>Help - pair masks by setting one bit</summary>

Infer `d` from `K=values.shape[-1]`. For each feature `i`, visit only integer masks where bit `i` is
zero; its paired present mask is `mask | (1 << i)`. Use `mask.bit_count()` for coalition size.

</details>

<details><summary>Expected output</summary>

```text
All tests in test_exact_shapley passed!
```

The threshold table `(0,0,0,1)` contracts to `(0.5,0.5)`, and every result sums to
`v(full)-v(empty)`.

</details>

<details><summary>Interpretation</summary>

The threshold comparison is now like-for-like: DeepSHAP and exact Shapley use the same input,
reference, model and feature players. Their `1/6` coordinate difference is compositional
approximation error, not a background mismatch.

</details>

<details><summary>Solution</summary>

```python
def exact_shapley(values: Tensor) -> Tensor:
    """Contract a complete (..., 2**D) game with exact Shapley weights."""
    _method_tensor("values", values)
    if values.ndim < 1:
        raise ValueError("values must have a final coalition axis.")
    size = values.shape[-1]
    d = size.bit_length() - 1
    if d < 1 or size != 2**d:
        raise ValueError("The final values axis must contain all 2**D coalitions for D >= 1.")

    result = t.zeros((*values.shape[:-1], d), dtype=t.float64)
    subset_ids = t.arange(size, dtype=t.int64)
    for feature in range(d):
        absent = subset_ids[subset_ids.bitwise_and(1 << feature).eq(0)]
        weights = t.tensor(
            [1 / (d * math.comb(d - 1, int(subset).bit_count())) for subset in absent],
            dtype=t.float64,
        )
        marginal = values[..., absent | (1 << feature)] - values[..., absent]
        result[..., feature] = (marginal * weights).sum(dim=-1)
    return result
```

</details>

<!-- ARENA:cell markdown -->
Before recomputing the threshold example, we need one ordinary-gradient baseline. We provide it
because it is a control rather than a DeepSHAP rule:

$$
\phi_i^{\text{grad-delta}}(x;B)
=\frac1{|B|}\sum_{b\in B}\frac{\partial f(x)}{\partial x_i}(x_i-b_i).
$$

It uses one endpoint gradient at `x`; it is not Expected Gradients or GradientSHAP. Its sum need
not reconstruct a nonlinear prediction difference.

<!-- ARENA:cell code -->
```python
# ARENA:insert deepshap_methods.py:gradient_delta
```

<!-- ARENA:cell code -->
```python
tests.test_gradient_delta(gradient_delta)
```

<!-- ARENA:cell markdown -->
<details><summary>Expected output</summary>

```text
All tests in test_gradient_delta passed!
```

</details>

<!-- ARENA:cell code -->
```python
# ARENA:insert deepshap_diagnostics.py:threshold_fixture
```

<!-- ARENA:cell code -->
```python
# ARENA:insert deepshap_diagnostics.py:threshold
```

<!-- ARENA:cell code -->
```python
threshold = diagnose_threshold(globals())
show_figure(plots.plot_threshold(threshold), "05_threshold_complete_not_exact")
```

<!-- ARENA:cell markdown -->
This is the cold-open result recomputed from your own methods. Exact and DeepSHAP share the same
endpoint and both satisfy completeness, but their individual feature bars disagree.

<!-- ARENA:cell markdown -->
## 8️⃣ Diagnose cancellation with RevealCancel groups

Rescale treats a nonlinear unit as one secant from total reference preactivation to total actual
preactivation. Opposing incoming changes can cancel before that secant is assigned. DeepLIFT's
historical RevealCancel rule instead groups positive and negative changes and computes a two-player
Shapley allocation through one activation.

For reference preactivation $z_0$, positive group $p\ge0$, negative group $n\le0$ and activation
$g$:

$$
\begin{aligned}
C_p &= \tfrac12[g(z_0+p)-g(z_0)]
     + \tfrac12[g(z_0+n+p)-g(z_0+n)],\\
C_n &= \tfrac12[g(z_0+n)-g(z_0)]
     + \tfrac12[g(z_0+p+n)-g(z_0+p)].
\end{aligned}
$$

This bounded diagnostic is not called by our DeepSHAP core or by the current SHAP Rescale path.

### Exercise 8 — `reveal_cancel_groups`

> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵🔵⚪⚪
>
> You should spend up to 20-30 minutes on this exercise.
> ```

Implement the two grouped allocations for a broadcast-compatible unary activation.

<!-- ARENA:cell code -->
```python
def reveal_cancel_groups(activation, reference, positive, negative):
    """Return two-group Shapley allocations through one unary activation."""
    raise NotImplementedError()


tests.test_reveal_cancel_groups(reveal_cancel_groups)
```

<!-- ARENA:cell markdown -->
<details><summary>Help - evaluate four activation states</summary>

You need `g(z0)`, `g(z0+p)`, `g(z0+n)` and `g(z0+p+n)`. Each group gets half its marginal when it
arrives first and half when it arrives second.

</details>

<details><summary>Expected output</summary>

```text
All tests in test_reveal_cancel_groups passed!
```

</details>

<details><summary>Interpretation</summary>

On nonnegative inputs, the network
`Linear([[0,1],[-1,1]]) -> ReLU -> Linear([[1,-1]])` computes
$x_1-\operatorname{ReLU}(x_1-x_0)=\min(x_0,x_1)$. At `x=(1,1)`, both inputs are necessary and exact
Shapley gives `(0.5,0.5)`, while Rescale assigns `(0,1)`. Mirroring the equivalent representation
assigns `(1,0)`. RevealCancel gives `(0.5,0.5)` in this grouped local diagnostic. The two networks
agree on all four binary inputs, so the difference exposes implementation dependence; neither
formula is claimed to represent `min` on arbitrary negative inputs.

</details>

<details><summary>Solution</summary>

```python
def reveal_cancel_groups(
    activation,
    reference: Tensor,
    positive: Tensor,
    negative: Tensor,
) -> tuple[Tensor, Tensor]:
    """Two-player Shapley allocation of positive/negative groups through one activation."""
    if not callable(activation):
        raise TypeError("activation must be callable.")
    for name, value in (("reference", reference), ("positive", positive), ("negative", negative)):
        _method_tensor(name, value)
    try:
        reference, positive, negative = t.broadcast_tensors(reference, positive, negative)
    except RuntimeError as error:
        raise ValueError(
            "reference, positive and negative must be broadcast-compatible."
        ) from error
    if (positive < 0).any() or (negative > 0).any():
        raise ValueError("positive must be nonnegative and negative must be nonpositive.")

    def activate(value: Tensor) -> Tensor:
        output = activation(value)
        if type(output) is not Tensor or output.shape != value.shape:
            raise TypeError("activation must return a tensor with the same shape as its input.")
        if output.device.type != "cpu" or output.dtype != t.float64 or not t.isfinite(output).all():
            raise ValueError("activation must return finite CPU float64 values.")
        return output

    base = activate(reference)
    positive_first = activate(reference + positive)
    negative_first = activate(reference + negative)
    together = activate(reference + positive + negative)
    positive_value = 0.5 * (positive_first - base + together - negative_first)
    negative_value = 0.5 * (negative_first - base + together - positive_first)
    return positive_value, negative_value
```

</details>

<!-- ARENA:cell code -->
```python
# ARENA:insert deepshap_diagnostics.py:min_fixture
```

<!-- ARENA:cell code -->
```python
# ARENA:insert deepshap_diagnostics.py:cancellation
```

<!-- ARENA:cell code -->
```python
cancellation = diagnose_cancellation(globals())
show_figure(
    plots.plot_cancellation(cancellation),
    "06_cancellation_implementation_dependence",
)
```

<!-- ARENA:cell markdown -->
The mirrored `min` implementations are a representation control. A feature attribution method that
follows hidden computational paths can change even when the tested input/output function does not.

<!-- ARENA:cell markdown -->
## 9️⃣ Turn attribution rankings into intervention curves

Attribution agreement is one diagnostic. A second asks what the model does when features are
actually replaced. Rank features, then insert them into the empty coalition or delete them from the
full coalition. Because `values` already averages intact donor hybrids, every point can be looked up
without another model call.

For regressions whose full prediction lies below the empty value, reverse the attribution sign
before ranking so “largest first” means movement toward the full prediction. Treat an exactly zero
output difference as positive. Break ties by feature index so the result is reproducible.

### Exercise 9 — `ranked_replacement_curves`

> ```yaml
> Difficulty: 🔴🔴🔴🔴⚪
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 35-45 minutes on this exercise.
> ```

Return a dictionary with `orders`, integer `insertion_masks` and `deletion_masks`, plus the
corresponding raw prediction values. Optional supplied orders must contain an exact permutation on
every row.

<!-- ARENA:cell code -->
```python
def ranked_replacement_curves(values, phi, orders=None):
    """Look up insertion and deletion prefixes in a complete coalition table."""
    raise NotImplementedError()


tests.test_ranked_replacement_curves(ranked_replacement_curves)
```

<!-- ARENA:cell markdown -->
<details><summary>Help - build integer prefixes</summary>

For an order `(j0,...,jD-1)`, insertion indices are
`0, 2**j0, 2**j0+2**j1, ...`; deletion indices start at `2**D-1` and subtract the same bits. Gather
those indices along the coalition axis.

</details>

<details><summary>Common bug - assuming deletion is monotone</summary>

Feature replacement in an interacting model can make a prediction overshoot or reverse direction.
Preserve the raw MPa curve. Do not apply cumulative maxima, absolute values or smoothing to make it
look monotone.

</details>

<details><summary>Expected output</summary>

```text
All tests in test_ranked_replacement_curves passed!
```

Insertion begins at `v(empty)` and ends at `v(full)`; deletion visits the same endpoints in reverse.

</details>

<details><summary>Interpretation</summary>

A good ranking may move the prediction quickly, but this is not a proof that its attribution values
are exact. Compare DeepSHAP, exact Shapley, gradient-times-delta and 100 fixed random orders on the
same rows. Curves can overlap when methods happen to induce the same ranking.

</details>

<details><summary>Solution</summary>

```python
def ranked_replacement_curves(
    values: Tensor,
    phi: Tensor,
    orders: Tensor | None = None,
) -> dict[str, Tensor]:
    """Read insertion/deletion prefixes from a complete per-row coalition table."""
    _method_tensor("values", values, 2)
    _method_tensor("phi", phi, 2)
    n, size = values.shape
    d = size.bit_length() - 1
    if d < 1 or size != 2**d or phi.shape != (n, d):
        raise ValueError("values must be (N,2**D) and phi must be (N,D), with D >= 1.")

    if orders is None:
        direction = t.where(values[:, -1] - values[:, 0] >= 0, 1.0, -1.0)[:, None]
        orders = t.argsort(direction * phi, dim=1, descending=True, stable=True)
    else:
        if type(orders) is not Tensor or orders.device.type != "cpu" or orders.dtype != t.int64:
            raise TypeError("orders must be a CPU int64 tensor.")
        if orders.shape != (n, d):
            raise ValueError("orders must have shape (rows, features).")
        expected = t.arange(d, dtype=t.int64).expand(n, d)
        if not t.equal(orders.sort(dim=1).values, expected):
            raise ValueError("Every order row must contain each feature exactly once.")
        orders = orders.clone()

    insertion_masks = t.zeros((n, d + 1), dtype=t.int64)
    deletion_masks = t.full((n, d + 1), size - 1, dtype=t.int64)
    for step in range(d):
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
# ARENA:insert deepshap_diagnostics.py:replacement_fixture
```

<!-- ARENA:cell code -->
```python
# ARENA:insert deepshap_diagnostics.py:replacement_control
```

<!-- ARENA:cell code -->
```python
replacement_control = diagnose_replacement_control(globals())
show_figure(plots.plot_replacement_control(replacement_control), "07_replacement_fixture")
```

<!-- ARENA:cell markdown -->
## From the modified backward pass to a real study

We now have three comparisons for the same model and finite background: the fast compositional
DeepSHAP approximation, exhaustive feature Shapley and the ordinary gradient-times-delta control.

<!-- ARENA:cell code -->
```python
methods = SimpleNamespace(
    paired_forward=paired_forward,
    rescale_slope=rescale_slope,
    rescale_multipliers=rescale_multipliers,
    single_reference=single_reference,
    background_deepshap=background_deepshap,
    coalition_values=coalition_values,
    exact_shapley=exact_shapley,
    reveal_cancel_groups=reveal_cancel_groups,
    ranked_replacement_curves=ranked_replacement_curves,
    coalition_masks=coalition_masks,
    gradient_delta=gradient_delta,
)
```

<!-- ARENA:cell markdown -->
### Try It Yourself 1 — move the threshold

Edit the weights, bias, query, reference or epsilon. Look for regions where exact Shapley and
DeepSHAP agree, regions where both are complete but allocate differently, and the tiny interval
where the finite epsilon deliberately breaks completeness.

<!-- ARENA:cell code -->
```python
# ARENA:insert deepshap_diagnostics.py:threshold_play
```

<!-- ARENA:cell code -->
```python
# ARENA:insert deepshap_diagnostics.py:aggregate
```

<!-- ARENA:cell code -->
```python
tiny_play = threshold_play(
    methods,
    weights=(2.0, 1.0),
    bias=-2.0,
    query=(1.0, 1.0),
    reference=(0.0, 0.0),
    epsilon=1e-6,
)
show_figure(plots.plot_threshold_play(tiny_play), "16_threshold_play")
```

<!-- ARENA:cell markdown -->
<details><summary>Ideas to try</summary>

- Set both weights equal. Does symmetry force the two methods to agree?
- Move the bias until one feature can cross the threshold alone.
- Use `weights=(1,)`, `bias=0`, `query=(4e-7,)`, `reference=(-4e-7,)` and compare
  `epsilon=1e-6` with `epsilon=0`.
- Scale every weight and the bias by the same positive constant. Which quantities scale, and which
  ratios stay fixed?

</details>

<!-- ARENA:cell markdown -->
## The frozen concrete-strength study

The data contain 1,030 public UCI concrete recipes with eight inputs and compressive strength in
MPa. Identical seven-ingredient recipes stay in the same split: 758 training rows, 153 explanation
rows and 119 audit rows. We train six fresh `8 -> 32 -> 32 -> 1` ReLU networks at three fixed seeds:
three with clean targets and three with independently shuffled targets. A ridge model is the affine
control. Every model operates in raw feature units after its training standardization is folded into
the first and last affine layers.

The shuffled networks and training-target mean test whether a visually detailed explanation can
make a poor predictor useful. The three clean seeds expose model variation; repeated agreement is
not treated as three independent uncertainty estimates. All seven models and all 272 held-out rows
remain in the results.

The fitting code is part of the lesson. It receives only the training split and retains its initial
state, standardized state, raw-unit state, loss trace, target permutation and training predictions.

<!-- ARENA:cell code -->
```python
# ARENA:insert deepshap_study.py:constants
```

<!-- ARENA:cell code -->
```python
# ARENA:insert deepshap_study.py:metadata
```

<!-- ARENA:cell code -->
```python
from part14_deepshap.deepshap_study import (
    _clone_state,
    _clone_tree,
    _combine_explanations,
    _combine_row_blocks,
    _float_tensor,
    _ids_tensor,
    _normalization_tensors,
    _output_tensor,
    _require_disjoint_ids,
    _require_methods,
    _rows,
    _state_shapes,
    _validate_evaluation_inputs,
    _validate_fitted,
    _validate_rankings,
)
```

<!-- ARENA:cell code -->
```python
# ARENA:insert deepshap_study.py:model_adapter
```

<!-- ARENA:cell code -->
```python
# ARENA:insert deepshap_study.py:normalization
```

<!-- ARENA:cell code -->
```python
# ARENA:insert deepshap_study.py:training_normalization
```

<!-- ARENA:cell code -->
```python
# ARENA:insert deepshap_study.py:train_mlp
```

<!-- ARENA:cell code -->
```python
# ARENA:insert deepshap_study.py:fold_mlp_result
```

<!-- ARENA:cell code -->
```python
# ARENA:insert deepshap_study.py:fit_mlp
```

<!-- ARENA:cell code -->
```python
# ARENA:insert deepshap_study.py:fit_ridge
```

<!-- ARENA:cell code -->
```python
# ARENA:insert deepshap_study.py:fitting
```

<!-- ARENA:cell code -->
```python
train = load_split("train")
explain = load_split("explain")
fitted = fit_models(train)
```

<!-- ARENA:cell markdown -->
The primary background contains 64 training rows selected by a fixed source-ID hash, independent of
inputs, labels, predictions and attributions. The explanation split fixes feature rankings before
the audit labels are loaded. `explain_block` evaluates every model with the same student methods and
stores predictions, pairwise DeepSHAP values, the complete 256-coalition game, exact Shapley and the
ordinary gradient-times-delta baseline.

<!-- ARENA:cell code -->
```python
# ARENA:insert deepshap_study.py:explanation
```

<!-- ARENA:cell code -->
```python
# ARENA:insert deepshap_study.py:backgrounds
```

<!-- ARENA:cell code -->
```python
# ARENA:insert deepshap_study.py:preparation
```

<!-- ARENA:cell markdown -->
The next boundary is scientific, not cosmetic. `prepare_study` fixes the background and rankings
using train and explanation data. The executor persists both fitted and prepared payloads and their
hashes before the following literal audit load. No audit outcome can affect those choices.

<!-- ARENA:cell code -->
```python
from part14_deepshap.run_native import save_payload, write_before_audit

native_dir = Path(RUN_OUTPUT or section_dir / "native") / "native"
prepared = prepare_study(fitted, train, explain, methods)
save_payload(native_dir, "fitted.pt", fitted)
save_payload(native_dir, "prepared.pt", prepared)
write_before_audit(native_dir, prepared)
```

<!-- ARENA:cell code -->
```python
audit = load_split("audit")
```

<!-- ARENA:cell markdown -->
Evaluation keeps the full audit arrays. The primary audit replacement experiment uses one global
mean-absolute feature ranking for each model and method, fixed on all 153 explanation rows and then
passed explicitly as the order for every audit row. It compares exact Shapley, DeepSHAP and
gradient-times-delta with 100 fixed random orders. Background variants use nested sizes 16, 64 and
128 plus independent 64-row draws, and recompute both the DeepSHAP approximation and exact game for
the primary clean model on all 272 held-out rows.

<!-- ARENA:cell code -->
```python
# ARENA:insert deepshap_study.py:ranked_helpers
```

<!-- ARENA:cell code -->
```python
# ARENA:insert deepshap_study.py:random_and_combine
```

<!-- ARENA:cell code -->
```python
# ARENA:insert deepshap_study.py:performance
```

<!-- ARENA:cell code -->
```python
# ARENA:insert deepshap_study.py:replacement_evaluation
```

<!-- ARENA:cell code -->
```python
# ARENA:insert deepshap_study.py:background_evaluation
```

<!-- ARENA:cell code -->
```python
# ARENA:insert deepshap_study.py:evaluation
```

<!-- ARENA:cell code -->
```python
evaluated = evaluate_study(fitted, prepared, train, explain, audit, methods)
```

<!-- ARENA:cell code -->
```python
# ARENA:insert deepshap_study.py:play
```

<!-- ARENA:cell markdown -->
The independent reference reconstructs every saved raw-unit model without importing the student
methods. Installed `shap.DeepExplainer` checks all 15,232 averaged attribution coordinates. A more
expensive donor-level check covers the first explanation and first audit row for all seven models
and all 64 donors: 896 pair explanations and 7,168 coordinates. The exhaustive reference separately
checks every one of the 1,904 model-row games.

<!-- ARENA:cell code -->
```python
from part14_deepshap.deepshap_reference import validate_reference
from part14_deepshap.run_native import persist_completed, write_notebook_summary

reference = validate_reference(fitted, prepared, evaluated, train, explain, audit)
diagnostics = run_diagnostics(methods, include_timing=True)
fixed_play = native_play(fitted, train, audit, methods)
summary = persist_completed(
    native_dir,
    fitted,
    prepared,
    evaluated,
    diagnostics,
    reference,
    fixed_play,
)
write_notebook_summary(Path(RUN_OUTPUT or section_dir / "native"), native_dir, summary)

print("Independent DeepExplainer averaged-vector parity:", reference["status"])
print("Averaged coordinates checked:", reference["counts"]["author_attribution_coordinates"])
print("Bounded donor coordinates checked:", reference["counts"]["bounded_author_pair_coordinates"])
print("Exact game rows checked:", reference["counts"]["exact_game_model_rows"])
```

<!-- ARENA:cell markdown -->
## Signature result: fast, complete, and measurably approximate

The first panel establishes that predictive skill belongs to the clean models rather than the
shuffled or mean controls. The next panels retain every signed feature value and every held-out row:
an identity-line scatter, row-by-feature error heatmaps, one predeclared-row waterfall, and a direct
three-method comparison. Do not replace these with only an aggregate error or a selected dramatic
example.

The frozen native study completed seven fresh fits in 8.01 seconds on the development CPU. Audit
mean-squared error was `31.04`, `35.61` and `36.58` MPa² for the three clean MLPs, compared with
`99.65` for ridge and `244.68` for the training-target mean. The shuffled-target controls were much
worse at `452.79`, `643.67` and `576.97` MPa². The explanation machinery can produce detailed
values for those controls, but it cannot turn them into useful predictors.

Across all 272 held-out rows, the three clean MLPs' mean absolute DeepSHAP-minus-exact coordinate
errors were `1.415`, `1.585` and `1.943` MPa; their largest coordinate errors were `10.87`, `17.75`
and `13.01` MPa. Yet their largest completeness residuals were only `4.97e-14`, `6.75e-14` and
`1.07e-13` MPa. Ridge is the decisive exactness control: its mean coordinate gap was `1.14e-15`
MPa. The installed DeepExplainer agreed with the student Rescale values to `1.71e-13` MPa, so the
neural gaps are approximation measurements rather than an implementation-parity failure.

The fixed source row 1 has target `61.89` MPa, prediction `70.6716` MPa and background base
`35.5483` MPa. Its exact versus DeepSHAP contributions differ visibly for coarse aggregate
(`3.3873` versus `6.9269` MPa), fine aggregate (`-2.4350` versus `-5.1074` MPa) and age (`2.9834`
versus `4.5909` MPa). This row was fixed before the audit was evaluated; it was not selected for a
large gap.

<!-- ARENA:cell code -->
```python
show_figure(plots.plot_performance(prepared, evaluated), "08_performance_controls")
show_figure(plots.plot_signed_parity(prepared, evaluated), "09_signed_parity_all_rows")
```

<!-- ARENA:cell code -->
```python
show_figure(plots.plot_error_heatmaps(prepared, evaluated), "10_error_heatmaps_all_rows")
show_figure(
    plots.plot_fixed_waterfall(fixed_play, feature_names=FEATURE_NAMES),
    "11_fixed_waterfall",
)
```

<!-- ARENA:cell code -->
```python
show_figure(
    plots.plot_fixed_attribution_comparison(fixed_play, feature_names=FEATURE_NAMES),
    "12_fixed_attribution_comparison",
)
show_figure(
    plots.plot_completeness_vs_exact(prepared, evaluated),
    "13_completeness_vs_exact",
)
```

<!-- ARENA:cell markdown -->
### Three different questions about numerical validity

1. **Completeness:** does `base + phi.sum()` reproduce this model prediction? The finite-epsilon
   near-zero control records its known exception rather than hiding it in a loose tolerance.
2. **Independent implementation parity:** does the student Rescale method agree with installed
   `shap.DeepExplainer` on the explicitly supported affine/ReLU chain?
3. **Exact feature-game agreement:** does either implementation equal exhaustive Shapley for the
   same donor background and all 256 coalitions?

A method can pass the first two and fail the third. Independent-library agreement checks our
implementation of DeepSHAP; exhaustive agreement checks DeepSHAP as an approximation to this game.

<!-- ARENA:cell markdown -->
## Background sensitivity and replacement controls

Changing a finite background changes both the base value and the game. The background panel shows
DeepSHAP movement beside exact-oracle movement, so sampling sensitivity is not mislabeled as
composition error. The replacement panel then compares ranked prefixes in raw MPa. Exact attribution
does not guarantee a monotone or uniformly superior prefix curve; these curves are model-behavior
diagnostics.

For the primary clean model, moving from the 64-row background to nested 16/128 backgrounds changed
exact coordinates by `0.784`/`0.593` MPa on average; the two independent 64-row backgrounds changed
them by `0.942`/`0.759` MPa. Their base predictions ranged from `32.581` to `38.012` MPa. These
finite-background movements are large enough to interpret alongside, rather than underneath, the
`1.278`–`1.531` MPa mean approximation gaps across the five background designs.

On the primary clean model, exact and DeepSHAP global orders agree that cement comes first but swap
slag and age next. DeepSHAP's first two insertions produced an average oriented gain of `11.319`
MPa, versus `5.796` MPa for the exact order; both orders were identical after the third feature and
ended at `12.022` MPa. This is a useful negative result: a better early replacement curve is not
evidence that an attribution vector is closer to exact Shapley.

<!-- ARENA:cell code -->
```python
show_figure(plots.plot_background_sensitivity(evaluated), "14_background_sensitivity")
show_figure(plots.plot_replacement_curves(evaluated), "15_replacement_curves")
```

<!-- ARENA:cell markdown -->
### Try It Yourself 2 — choose a recipe, model and background

The canonical first run uses `clean_0`, audit row 0 (source row 1) and the frozen 64-row background.
After reproducing that result, edit the arguments. The play recomputes the approximate explanation,
exact game and replacement curves rather than slicing a cached chart. Its curves use the local
per-row ordering from Exercise 9, oriented by `sign(f(x)-base)`. They are exploratory and distinct
from the primary experiment's frozen global explanation-split rankings.

<!-- ARENA:cell code -->
```python
play = native_play(
    fitted,
    train,
    audit,
    methods,
    model_key="clean_0",
    row=0,
    size=64,
)
show_figure(
    plots.plot_real_recipe_play(play, feature_names=FEATURE_NAMES),
    "17_real_recipe_play",
)
```

<!-- ARENA:cell markdown -->
<details><summary>Ideas to try</summary>

- Compare `clean_0`, `clean_1`, `clean_2`, `shuffled_0` and `ridge` on the same source row.
- Keep the model and row fixed while changing the nested background size from 16 to 64 to 128.
- Find a row where the DeepSHAP and exact rankings agree even though their signed magnitudes differ.
- Find an overshooting deletion curve and inspect which interaction makes a replacement reverse the
  expected direction.

These choices occur after audit evaluation. They are exploratory diagnostics and cannot support a
new confirmatory audit claim.

</details>

<!-- ARENA:cell markdown -->
## How to interpret the study

Read the figures in this order:

1. Confirm that clean models beat the training-mean and shuffled-target controls. Attribution from
   a model without held-out predictive skill does not rescue the predictor.
2. Require exact endpoint identities and full-vector parity with the exhaustive oracle for the
   affine control. Then inspect independent DeepExplainer parity separately.
3. Check completeness and exact error side by side. Near-zero epsilon failure is a stated numerical
   rule; larger exact errors with small completeness residuals are approximation behavior.
4. Compare all-row signed errors before the fixed-row waterfall. A waterfall explains where one
   prediction came from; it does not establish a population pattern.
5. Read background movement beside exact movement, and replacement curves beside random-order
   bands. Do not infer a universal method ranking from one model or overlapping curves.

For the concrete data, replacing one recorded quantity while borrowing the others from observed
training recipes creates synthetic model inputs. These are computational interventions on the
network. They are not randomized experiments on concrete production, and the empirical background
is not evidence that the attributions are physical causal effects.

## Limitations and method scope

The student method covers scalar affine/ReLU `nn.Sequential` chains with distinct modules. It
rejects convolution, pooling, softmax, multiplication, residual branches, shared modules,
normalization layers and custom operators. Current DeepSHAP libraries use backend-specific rules for
some additional components, and unsupported behavior can vary by backend. Agreement here should not
be generalized to an arbitrary PyTorch graph.

DeepSHAP is deterministic for a fixed model, query, background and epsilon. It is an approximation
to exact feature Shapley for a composed nonlinear graph, even when its local sum is exact. Exact
enumeration scales as $2^D$ and is available here only because the study has eight inputs. A finite
background approximates a chosen reference distribution and can move the answer materially.

RevealCancel explains one historical response to cancellation. It is not part of the taught
DeepSHAP core or the installed library comparison. Gradient-times-delta is a deterministic baseline,
not GradientSHAP or Expected Gradients; those methods require their own sampling and path-integral
lesson.

## Further research and anomaly hunting

- Characterize which ReLU activation patterns make Rescale equal exact input-feature Shapley.
- Construct functionally equivalent networks with increasingly different DeepSHAP allocations.
- Locate the smallest background for which ranking and signed-value conclusions stabilize, while
  recomputing the exact oracle at every size.
- Extend the visible dispatcher by one operator and design an exact fixture that can falsify its
  rule before applying it to a real model.
- Compare the finite-epsilon fallback with a scale-aware alternative without choosing its threshold
  from audit errors.

## Reading

- Lundberg & Lee,
  [*A Unified Approach to Interpreting Model Predictions*](https://arxiv.org/abs/1705.07874),
  especially the Deep SHAP discussion and compositional approximation.
- Shrikumar, Greenside & Kundaje, [*Learning Important Features Through Propagating Activation
  Differences*](https://arxiv.org/abs/1704.02685), plus its supplement for Rescale and RevealCancel.
- The current SHAP PyTorch DeepExplainer
  [source](https://github.com/shap/shap/blob/master/shap/explainers/_deep/deep_pytorch.py)
  and
  [documentation](https://shap.readthedocs.io/en/latest/generated/shap.DeepExplainer.html).
- Yeh, [*Concrete Compressive Strength*](https://doi.org/10.24432/C5PK67), the observational dataset
  used for the held-out study.
