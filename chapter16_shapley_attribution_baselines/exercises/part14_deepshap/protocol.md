# DeepSHAP: frozen native experiment

This protocol is written before any native DeepSHAP Concrete model fit.
`protocol_freeze.json` will record the actual freeze time, zero native fits,
versions, and SHA256 of this file, `data_manifest.json` and `deepshap_data.py`.
No native fit may start before that record is written. Later discoveries must
be disclosed as dated addenda; do not silently change the frozen experiment.

## Question and scope

How closely does DeepSHAP's compositional Rescale rule approximate exact
feature Shapley for the same finite empirical background, and what do
completeness, reference parity and controlled input interventions each test?

The supported model must have exact type `nn.Sequential` and contain only exact
`nn.Linear` modules and distinct, non-in-place `nn.ReLU` modules, with one scalar
output. Reject other operators, subclasses, shared modules, non-finite inputs
and incompatible shapes. The
lesson teaches the Linear/ReLU case of DeepSHAP, with a bounded historical
RevealCancel diagnostic. It does not implement general DeepSHAP operator rules
or GradientSHAP/Expected Gradients.

## Data and pre-existing knowledge

Use Yeh (1998), UCI Concrete Compressive Strength, DOI 10.24432/C5PK67,
CC BY 4.0. Official URL:
https://archive.ics.uci.edu/static/public/165/data.csv. The source is 41,472 bytes,
SHA256 `8d4b15b6fc68cd932d745cbd663d5ceae66dd54422e99c1e4865f2936ab7e2af`.
Keep all 1,030 finite rows and all conflicting replicate outcomes. Predict
strength in MPa from seven ingredients (kg/m³) and age (days).

Reuse exactly 16.12's group split, also used in 16.13. Earlier SAGE and TreeSHAP
results are already known; this is not a new unseen dataset. Group identical
first-seven-input recipes, ignoring age. Encode those seven source values as
little-endian float64, SHA256 the bytes, then sort groups by the SHA256 digest
of `arena16.12/split/` plus the group hex. Split after `floor(.70 * G)` and
`floor(.85 * G)` groups; retain sorted source-row indices within each split. The
427 groups produce 298, 64 and 65 groups and 758, 153 and 119
train/explanation/audit rows, with zero recipe overlap. Laboratory identities
are unavailable. All row identities are pinned in the manifest before
training. No deduplication or outcome filtering occurs.

## Fresh training

Run on CPU with four PyTorch threads, float64 parameters and inputs, and
`torch.use_deterministic_algorithms(True)`. Expected environment: NumPy 1.26.4,
PyTorch 2.12.1+cu132 used on CPU, scikit-learn 1.9.0, and SHAP 0.49.1.

Compute input and target means and population standard deviations using train
only. Each neural model is `Linear(8, 32) -> ReLU -> Linear(32, 32) -> ReLU ->
Linear(32, 1)`, using default Torch Linear initialization in float64. Retain all
initial and final states, target permutations, complete losses and predictions.
Train six models: clean and shuffled targets, each at seeds 161410, 161411 and 161412.
Reset the paired seed before each initialization. For shuffled targets use
NumPy `default_rng(161420 + index).permutation(758)`, with index 0, 1 or 2,
applied to standardized training targets. Fit exactly 800 full-batch Adam steps
at `lr=0.003`, `weight_decay=0.0001`, and ordinary mean squared error. All
parameters are trained.
There is no validation selection, early stopping, seed rejection, restart,
architecture sweep or checkpoint choice: retain all 4,800 optimizer updates.

Fit one standardized ridge control using mean squared risk and penalty 0.01 on
coefficients, excluding intercept. Solve the normal equations once. Retain all
seven models: `clean_0/1/2`, `shuffled_0/1/2`, and `ridge`. The training-mean
predictor is an additional no-fit performance baseline.

For explanation, fold standardization into the same raw-input/output Sequential:
`W1raw = W1 / x_std`, `b1raw = b1 - W1 @ (x_mean / x_std)`,
`W3raw = y_std * W3`, and `b3raw = y_std * b3 + y_mean`. Ridge uses the
analogous fold into one `Linear(8, 1)`. Preserve both standardized and raw states
and verify output parity. Explain the raw MPa scalar. No softmax, output
clipping, quantization, float32 conversion or custom normalization wrapper is
part of the explained model. Author comparison uses the same folded raw model.

## Backgrounds and games

Order training rows by lexicographic SHA256 digest of
`arena16.14/background/{seed}/{source_row}`. Primary background contains the
first 64 rows for seed 161460. Five total designs are fixed: `primary64`,
`nested16` and `nested128` from the same seed 161460 order, and independently
ordered `independent64_a` and `independent64_b` from seeds 161461 and 161462.
The independently ordered sets are not constrained to be disjoint and may
overlap. They are deterministic sensitivity probes, not uncertainty replicates.
Keep all donor identities and values; no explainer background summarization or
sampling is allowed.

For every query `x` and coalition mask `S`, use
`v_x(S) = mean_b f(where(S, x, b))`. Each hybrid uses one intact donor row for all
missing features. Average predictions, not donor vectors. This empirical
marginal/background game is not the population conditional game and is not a
physical causal intervention on concrete ingredients.

All seven models receive all 256 coalition values for every one of the 153
explanation and 119 audit rows under `primary64`. This is 1,904 model-rows,
487,424 coalition values and 31,195,136 primary hybrid predictions. Acceptance
checks the actual arrays and their identities, rather than using a count as a
shortcut. Exact feature Shapley uses all coalitions. Each averaged attribution
method has 15,232 model-row-feature coordinates. Preserve those arrays and all
974,848 DeepSHAP query-donor-feature contributions.

DeepSHAP uses Rescale multiplier propagation. The primary small-denominator
rule is `abs(zx - zb) < 1e-6` or exact zero; use ReLU's reference derivative
`1[zb > 0]` there and the exact secant elsewhere. Multiply input multipliers by
`x - b` and average explanations over all 64 donors. The base is `mean_b f(b)`.
Do not call attribution at the mean donor the same explanation.

The ordinary gradient baseline is precisely
`grad_x f(x) * (x - mean_b b)`, equal to the mean over `b` of the gradient at
`x` times `(x - b)`.
It is not locally complete in general and is not GradientSHAP.

For `clean_0`, recompute exact games and DeepSHAP/gradient explanations on all
272 rows for all four alternate backgrounds. Keep `primary64` as the matched
anchor. Separate movement caused by background choice from approximation error
under the same background. Do not conflate seed sensitivity with sampling error.

## Preparation, audit and replacement curves

Fit on train only. Compute all primary explanation results on the 153 explanation
rows. Save every model and prepared result, including mean-absolute feature
rankings, background IDs and source hashes, before the literal audit load.
Record the saved payload hashes at this boundary. Then evaluate all 119 audit
rows. No audit outcome selects model, background, parameter, row, plot axis,
method setting or main display example.

For each model and method, rank features by descending mean absolute attribution
over the 153 explanation rows, with stable ties broken by ascending feature
index. Freeze these exact Shapley, DeepSHAP and gradient-delta orders before the
audit load, then apply each fixed order to all 119 audit rows. Insertion starts
empty and adds prefixes; deletion starts full and removes the same prefixes.
Preserve raw MPa table values and actual mask/order indices. No monotonicity or
superiority is assumed.

The generic student method and editable one-row play may instead compute a
local order from `sign(vfull - vempty) * phi` when no frozen order is supplied;
treat `sign(0)` as `+1` and break ties by ascending feature index. This is a
local explanation diagnostic, not the primary ordered audit experiment.

Generate 100 random feature permutations using one NumPy RNG seed 161480 stream;
reuse that frozen array across all models/rows. Show these controls beside the
three method orders. For comparable aggregate displays, insertion gain is
`sign(vfull - vempty) * (v(S) - vempty)` and deletion drop is
`sign(vfull - vempty) * (vfull - v(D\S))`, in MPa, without dividing by small
prediction differences. Preserve each row and every random ordering, not just
quantiles. Audit prediction MSE and per-row attributions are retained for all
seven models.

## Predeclared diagnostics and editable examples

- Threshold circuit: `ReLU(2*x0 + x1 - 2)`, query `(1, 1)`, reference `(0, 0)`.
  Exact Shapley is `(0.5, 0.5)`, DeepSHAP is `(2/3, 1/3)`, and gradient-delta is
  `(2, 1)`. Both single-feature necessity interventions change output by 1.
  Completeness does not prove exactness.
- Min on the nonnegative input domain: `x1 - ReLU(x1 - x0)`, expressed with two
  hidden ReLUs carrying `x1` and `x1 - x0`, gives DeepSHAP `(0, 1)` at `(1, 1)`
  from `(0, 0)`. A mirrored equivalent representation gives `(1, 0)`. Exact
  Shapley and the bounded RevealCancel diagnostic give `(0.5, 0.5)`. Show all
  four binary inputs and the domain restriction. This is known computational
  intervention evidence.
- Near-zero crossing: a ReLU query of `+4e-7` and reference of `-4e-7` has output
  delta `4e-7`, but primary-epsilon DeepSHAP returns `0`. This expected summation
  failure is disclosed. The exact-nonzero-secant variant with `epsilon=0`
  returns `4e-7`. Do not apply a universal exact-completeness claim to the
  guarded implementation.
- Affine, constant, dummy and multiple-background-versus-mean-reference games
  have independently derived expectations. Deterministic toy seeds 161490 and
  161491 may initialize fixed affine fixture weights; no native model is fit.
- Default real waterfall/play uses audit index 0, source row 1, `clean_0` and
  `primary64`. Show the paired exact/DeepSHAP/gradient values and prediction
  context. Main aggregate panels include all rows/models. The play can change
  model, row and `nested16` / `primary64` / `nested128` background; mark this as
  exploratory audit reuse.
- Tiny play edits the gate weights/bias/query/reference, then recomputes its
  coalition table, exact/Rescale allocations and computational interventions.

## Independent checks and acceptance boundaries

Pin the installed SHAP 0.49.1 DeepExplainer Python source and primary references
by hashes. Compare author and student base and full local vectors for every one
of the 1,904 model-rows on the exact same 64 background rows. Verify reference
calls leave every raw model parameter unchanged. Additionally compare each
individual donor contribution for the first explanation row and first audit row
(source-ID order) of every model, using 64 single-background author explainers:
896 model/query/donor pairs and 7,168 coordinates. Label this bounded check
separately from the full averaged-vector comparison; do not claim full author
per-donor parity. Check row/background identity, model states, raw-output units
and unsupported-operator rejection explicitly.

For primary real-data author parity and local-accuracy residuals, use
`atol=1e-7, rtol=1e-8`; tiny exact/algebraic checks use `atol=1e-10, rtol=1e-10`
unless the known small-denominator exception is the diagnostic itself. Exact
coalition Shapley efficiency and affine/ridge analytic parity use the tighter
exact tolerance. If a real-data local-accuracy residual exceeds its tolerance,
retain and diagnose that failure rather than changing the epsilon or tolerance.
DeepSHAP-versus-exact-neural Shapley disagreement is a measured result, never an
error threshold to tune away. Shuffled performance, ordered curves and
approximation quality need not satisfy an invented superiority gate.

Run the completed learner using only answers from its own immediate Solution
dropdowns in a fresh CPU kernel. Refit all seven models from source in another
process with no cached fitted inputs, then compare every stored deterministic
array/scalar exactly (`atol=0, rtol=0`), including full states, targets, games,
all reference contributions, controls and replacement curves. Measure timings
separately if any are shown. Source changes during execution invalidate that run.
Review actual figures, math and dropdown behavior before publication. Keep
failures and limitation cases alongside successful checks. Full-course
completion remains separate from this section's scientific acceptance.
