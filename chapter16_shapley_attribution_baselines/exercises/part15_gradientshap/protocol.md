# Expected Gradients / GradientSHAP: native-study protocol

Status: **reviewed design; execution requires the matching zero-run freeze**. No
GradientSHAP Concrete model fit or native attribution may start until a separate
`protocol_freeze.json` binds this text, the manifest and data implementation.
This protocol was written
after the 16.14 DeepSHAP models and data had been examined. It therefore defines
a reproduction and method comparison on known predictors, not an independent
model or dataset replication.

## Question and claim boundary

Which expectation does the Monte Carlo estimator usually called GradientSHAP
estimate, how quickly does it approach that target, and how do baseline support
and Gaussian query noise change the target?

The taught clean estimator is Expected Gradients for a deterministic scalar
function `f: R^8 -> R`, a finite empirical background `B`, and explicit donor
and path draws:

```text
C_ki = (x_i - b_ki) * partial_i f(b_k + alpha_k * (x - b_k))
EG_i = E_{b,alpha}[C_ki],  b ~ Uniform(B), alpha ~ Uniform(0,1).
```

Under the usual path regularity, the population sum is
`f(x) - mean_b f(b)`. A finite Monte Carlo vector has sampling error and need
not sum exactly. Expected Gradients averages continuous straight-path credit;
exact feature Shapley in 16.14 averages discrete coalition increments in an
intact-donor replacement game. Agreement between them is a result, never an
identity or acceptance condition.

Use **Expected Gradients** for the mathematical estimator. Use **SHAP
GradientExplainer 0.49.1** and **Captum GradientShap 0.9.0** only for their
version-specific package implementations. SHAP's reported `phi_vars` is not the
ordinary estimator standard error. Captum's convergence delta is per repeated
sample, not the residual of the averaged local vector. Endpoint gradient times
input delta and 16.14 DeepSHAP are separate controls.

All results explain raw compressive-strength output in MPa. Donor replacement,
convex interpolation and Gaussian noise are computational probes on an
observational dataset. They are not physical interventions or causal ingredient
effects. Nearest-training-row distance is a bounded geometric diagnostic, not a
density estimate or probability of being on distribution.

## Known data and exact row boundary

Use Yeh (1998), UCI Concrete Compressive Strength, DOI `10.24432/C5PK67`,
CC BY 4.0. The official CSV is 41,472 bytes with SHA256
`8d4b15b6fc68cd932d745cbd663d5ceae66dd54422e99c1e4865f2936ab7e2af`.
Keep all 1,030 finite rows, conflicting replicate outcomes, eight input columns
and the MPa target. Do not deduplicate, filter outcomes or infer laboratory IDs.

Reuse the exact 16.12/16.13/16.14 recipe grouping and pinned source IDs in
`data_manifest.json`: 758 train rows in 298 groups, 153 explanation rows in 64
groups, and 119 audit rows in 65 groups, with zero recipe overlap. Group the
first seven source inputs after little-endian float64 encoding, order group keys
by the frozen `arena16.12/split/` digest rule, and keep source IDs ascending
within each split.

The row order for stored all-row arrays is every explanation source ID followed
by every audit source ID. Fit and prepare explanation-side rankings first. Save
the complete fitted and prepared payloads and their hashes, then call the literal
`load_split("audit")`. No audit model performance, attribution, convergence,
distance, curve or package result may select a model, row, seed, background,
sample budget, noise scale, tolerance, plot axis or primary example.

## Self-contained reproduction of the seven known predictors

Refit all seven 16.14 predictors inside this section. Do not load fitted weights
as the scientific input. Run on CPU with four PyTorch threads, float64 inputs
and parameters, and deterministic algorithms. Use train-only population means
and standard deviations for inputs and target.

For seeds `161410`, `161411`, and `161412`, construct
`Linear(8,32) -> ReLU -> Linear(32,32) -> ReLU -> Linear(32,1)` using default
float64 Torch Linear initialization. For each seed fit one clean target and one
shuffled target. The shuffle is NumPy `default_rng(161420 + index).permutation`
of all 758 standardized training targets. Train exactly 800 full-batch Adam
steps with mean squared error, learning rate `0.003`, and weight decay `0.0001`.
There is no validation selection, early stop, restart or checkpoint choice.
Retain all six initial states, final standardized states, target permutations,
800-point loss traces and training predictions.

Fit the same ridge control by one normal-equation solve with coefficient penalty
`0.01` and no intercept penalty. Preserve model order
`clean_0/1/2`, `shuffled_0/1/2`, `ridge`. Fold every model into raw-input and
raw-MPa form exactly as in 16.14 and verify standardized/raw prediction parity.

This refit must reproduce the known 16.14 functions exactly. Before calculating
GradientSHAP, compare each model's initial, standardized and raw-folded state
against the 21 predeclared hashes in `data_manifest.json`. The checksum is
SHA256 over sorted parameter names, stable dtype/name/shape metadata and C-order
little-endian tensor bytes. Also bind train IDs, normalization, training targets,
permutations, loss traces and training predictions. A mismatch stops the run;
it is not repaired with a tolerance or by importing the old fitted payload.
Passing establishes exact reproduction of the known seeded procedure, not an
independent model replication.

## Student method API

The nine student-visible methods, in teaching order, are:

```python
sample_path_schedule(source_ids, background_rows, samples, seed)
interpolate_paths(x, background, indices, alpha, noise=None)
input_gradients(model, points, forward_batch=32768)
expected_gradient_samples(model, x, background, indices, alpha, forward_batch=32768)
summarize_expected_gradients(samples, budgets)
expected_gradients(model, x, background, indices, alpha, forward_batch=32768)
gauss_legendre_pairs(model, x, background, n_nodes, row_batch=4, forward_batch=32768)
noisy_expected_gradient_samples(model, x, background, indices, alpha, noise, forward_batch=32768)
ranked_coalition_curves(values, phi, orders=None)
```

Schedules are CPU `int64` donor indices and float64 alphas with shape `(N,K)`.
Path points, deltas, noise, gradients and per-draw contributions have shape
`(N,K,8)`. `expected_gradients` returns a scalar full-background base, `(N,8)`
mean attribution and `(N,8)` ordinary sample standard error. Pair quadrature
returns `(N,B,8)`. The curve function returns `orders`, insertion/deletion masks,
and raw insertion/deletion values. It reuses already-computed coalition tables.

Reject duplicate source IDs, invalid donors or alphas, non-float or non-finite
scientific tensors, shape mismatches, nonpositive batch sizes, duplicate or
out-of-range budgets, `K < 2` for sample SE, and models that do not return one
scalar per row. Row independence is a caller precondition, not something output
shape can certify; native execution uses only the declared affine/ReLU Sequential
chains and ridge graphs, which satisfy it. Student methods must not change inputs, model mode,
parameter gradients, global RNG state or Torch's default dtype.

## Exact RNG namespace

The following namespace and seeds become fixed when the zero-run freeze record
is written:

| Purpose | Namespace seed |
|---|---:|
| Primary donor/alpha schedule | `161510` |
| 64 independent schedule replicates | `161511` |
| Standard-normal query noise | `161512` |
| SHAP author stream | `161513` |
| Captum NumPy donor/alpha stream | `161514` |
| SHAP and Captum Torch smoothing streams, each reset separately | `161515` |
| Hand-derived toy fixtures | `161590` |
| Untrained engineering benchmark only | `161599` |

The root namespace is `arena16.15/gradientshap/v1`. For each source ID and
stream, seed a separate NumPy 1.26.4 PCG64 generator using
`int.from_bytes(SHA256("{namespace}/{seed}/{source_id}/{stream_tag}").digest()[:16], "little")`.
The literal tags are `baseline`, `alpha`, and `standard_normal`. Draw donors with
`integers(0, B, size=K, dtype=np.int64)`, alphas with
`random(K, dtype=np.float64)`, and noise with `standard_normal((K,8))`.
For replicate r=0..63, derive one uint64 seed from the **first eight digest
bytes**, interpreted little-endian, of
`SHA256("arena16.15/gradientshap/v1/161511/replicate/{r}")`. All 64 distinct
seeds are recorded in the manifest. Pass each seed through the unchanged student
`sample_path_schedule` API, which uses the literal `baseline` and `alpha` tags.
Generate each row separately, then stack in requested source-ID order. This makes row
reordering and batching irrelevant and makes every smaller K a literal prefix
of K=2048. The schedule never consumes global NumPy or Torch RNG state. Reuse
the same primary donor and alpha tables across all seven models.

SHAP and Captum consume random numbers in different orders. Their package
adapters therefore get separate frozen streams. Equal integer seeds across
packages must never be described as equal draws. Same-draw parity is checked by
reconstructing each package's actual schedule and replaying that explicit table
through the student estimator while restoring global RNG, default dtype, model
mode and existing parameter gradients, including on exceptions.

## Backgrounds and clean primary calculation

Reproduce the five 16.14 training backgrounds exactly:

- `primary64`: first 64 of seed `161460`;
- `nested16` and `nested128`: prefixes of the same `161460` order;
- `independent64_a` and `independent64_b`: first 64 of seeds `161461` and
  `161462`.

The order is lexicographic SHA256 of
`arena16.14/background/{seed}/{source_row}`. Keep every donor identity and raw
row. Independent 64-row designs may overlap and are alternate finite supports,
not confidence replicates.

For `primary64`, explain all seven models on all 153 explanation and 119 audit
rows. Use one K=2048 schedule per source row and literal nested prefixes
`[8,32,128,512,2048]`. Preserve the complete donor/alpha schedules and all
31,195,136 float64 contribution coordinates, plus prefix means, ordinary SE,
full-background bases and finite-sample completeness residuals. Store every
array and identity, rather than recomputing per-draw evidence from summaries.

For every primary model-row/donor pair, compute 256- and 512-node
Gauss–Legendre straight-path attributions. Preserve both `(1904,64,8)` pair
arrays, their donor averages and their everywhere 256-to-512 drift. This is
31,195,136 and 62,390,272 gradient examples respectively. Finite quadrature is
an approximate numerical control for discontinuous ReLU gradients; neither rule
is labeled exact and agreement with 512 nodes is not an acceptance gate.

## Exact, DeepSHAP and gradient controls

On the same freshly reproduced raw models and `primary64`, recompute for every
model and all 272 rows:

- all 256 values of the intact-donor coalition game and exact feature Shapley;
- 16.14 Rescale DeepSHAP with all 64 donor contributions retained;
- endpoint `grad_x f(x) * (x - mean(B))`;
- model prediction, full-background base and training-mean prediction control.

This retains 487,424 coalition values and 31,195,136 coalition hybrid
predictions. Exact feature Shapley, DeepSHAP, endpoint gradient and Expected
Gradients differences are descriptive scientific results. Exact feature
Shapley is exact only for its declared finite replacement game.

For `clean_0`, repeat exact games, DeepSHAP, endpoint gradient, K=2048 Expected
Gradients, and 256/512 pair quadrature on all 272 rows under all five background
designs. Preserve full per-draw MC contributions under every design. Report
base movement, coordinate movement, sign/rank changes, quadrature drift,
Monte Carlo error and SE separately. Background movement is a change of finite
estimand, not estimator uncertainty.

## Gaussian query-noise study

For `clean_0`, `primary64` and all 272 rows, use standardized-input sigmas
`[0.00, 0.02, 0.10]`. Map them to raw units featurewise with the train-only input
standard deviations. Draw one explicit standard-normal `(272,2048,8)` schedule
from seed `161512`, reuse it across positive sigmas, and use the same donor and
alpha tables as the clean estimator. Noise must enter both the query-minus-donor
delta and the interpolation endpoint. Sigma zero must be bitwise identical to
the clean primary samples and should alias those stored arrays rather than
silently create a second schedule.

Preserve every positive-noise per-draw contribution and noisy-query prediction.
Report both

```text
sum(phi_sigma) - (f(x) - mean_b f(b))
sum(phi_sigma) - (mean_noise f(x + noise) - mean_b f(b)).
```

The second target is itself a finite Monte Carlo estimate; retain its ordinary
SE. Positive sigma changes the estimand. It is not merely a variance setting,
and clipping would define a different unapproved noise distribution.

## Independent sampling-replicate diagnostic

Use exactly the first 32 pinned explanation source IDs and first 32 pinned audit
source IDs recorded in `data_manifest.json`. Evaluate `clean_0`, `shuffled_0`
and `ridge` with 64 independent K=2048 schedules using the 64 recorded derived
replicate seeds. Smaller budgets remain the five literal prefixes. The subset,
models, replicate count and maximum K are fixed before attribution and may not
be changed after viewing errors.

Retain all schedules and all 201,326,592 per-draw contribution coordinates,
with complete hashes in the single external `replicate_samples.pt` array of
shape `(64,3,64,2048,8)`: exactly 1,610,612,736 raw bytes (1.5 GiB), before
serialization overhead. Donor indices and alphas have shape `(64,64,2048)` each
and are shared across the three models; together they occupy 134,217,728 raw
bytes (128 MiB). All source IDs, model order and split boundaries remain explicit.
For each coordinate and
budget retain mean, ordinary SE, error relative to 512-node all-background
quadrature, and whether the descriptive `estimate +/- 1.96*SE` interval contains
that numerical control. Report empirical coverage across replicates with row and
coordinate dependence disclosed. Coverage is a diagnostic of the ordinary SE,
not a guarantee that quadrature is exact or coordinates are independent. Audit
replicates are evaluated only after the preparation boundary and never tune K.

## Bounded interpolation and coalition distance diagnostic

Use train-only means and population standard deviations in eight dimensions.
For all 272 query rows, retain the distance to the nearest of all 758 training
rows for:

1. the first 128 points of the fixed primary Expected Gradients schedule; and
2. all 256 coalition masks crossed with the first eight `primary64` donor source
   IDs `[173, 39, 334, 235, 985, 285, 174, 205]`.

This predeclares 34,816 path points and 557,056 coalition hybrids. Compute exact
Euclidean nearest-row distances in bounded batches and retain source ID, donor
ID, alpha or coalition mask, and distance for every point. Include query and
training-donor endpoint distances as controls. Do not select examples or axes
from these distances and do not turn them into a pass threshold. The diagnostic
does not estimate density, likelihood, physical feasibility or causality.

## Explanation-only rankings and audit interventions

For exact feature Shapley, DeepSHAP, endpoint gradient, K=2048 Expected
Gradients and 512-node quadrature, rank features by descending mean absolute
attribution over the 153 explanation rows, with stable feature-index tie break.
Persist these global orders before the literal audit load, then apply them to all
119 audit rows using the already-computed coalition tables. Reuse all 100 random
orders from the 16.14 seed `161480` stream across models and rows.

Retain raw insertion/deletion values, masks, fixed orders and every random-order
curve for all seven models. Orient aggregate gain/drop only by the sign of
`v(full)-v(empty)` and keep MPa units. Do not divide by small prediction
differences or impose method-superiority, monotonicity or beats-random gates.
The generic exercise and editable one-row play may use a clearly labeled local
signed order when no frozen global order is supplied.

## Author implementations and analytic controls

Use the pinned SHAP 0.49.1 and Captum 0.9.0 sources documented in
`~/.cache/arena_extended/gradientshap/reference_recon/`. For each package use a
separately frozen K=128 schedule on every one of the 1,904 model-rows. Compare
the full local vector with student replay of that package's exact donors,
alphas, and noise. Use one full 272-row call per package and model. Reset SHAP
NumPy `rseed=161513` for each call and use `batch_size=128`; reset Captum NumPy
to 161514 and Torch to 161515 before each full call. Thus all seven models share
a package-specific draw table; the two packages still use different tables.
The separately bounded positive-noise analytic checks reset each package
Torch stream to 161515 and use its declared NumPy stream. Same-draw tolerance is
`atol=1e-10, rtol=1e-10`; any nonzero discrepancy is retained and diagnosed,
not hidden by changing draws. Package estimate versus q512 is reported
separately as Monte Carlo behavior. Positive-noise package parity uses the
analytic smoothed-ReLU fixture, not the clean real-data comparison.

Independent fixtures cover affine, quadratic, one-dimensional ReLU,
`ReLU(2*x0+x1-2)`, equivalent min graphs at a kink, nonlinear background
averaging, Gaussian-smoothed ReLU, constant, dummy and duplicate draws. Tiny
analytic/algebraic checks use `atol=1e-10, rtol=1e-10`. Exact model-state hashes
and sigma-zero sample identity are byte-exact. No tolerance applies to observed
Expected-Gradient versus exact-Shapley disagreement, background movement,
quadrature doubling, Monte Carlo residuals, OOD distances or audit curves.

## Payload and execution boundary

The runner calls, in order (study helpers load the frozen manifest mechanically):

```text
fit_models(train)
verify_16_14_reproduction(fitted, manifest)
prepare_study(fitted, train, explain, methods=methods, deep_methods=deep_methods)
save fitted.pt and prepared.pt; write before_audit.json
audit = load_split("audit")
evaluate_study(fitted, prepared, train, explain, audit, methods=methods, deep_methods=deep_methods)
validate_reference(...)
run_diagnostics(...)
native_play(...)
persist completed payloads and notebook summary
```

Deterministic payloads are `fitted.pt`, `prepared.pt`, `evaluated.pt`,
`reference.pt`, `diagnostics.pt`, `play.pt`, and `replicate_samples.pt`.
`timings.pt` is a separate non-reproduction branch. Every deterministic payload
has a recursive ordered path/type/shape/dtype/value SHA256 manifest, plus the
serialized artifact byte SHA256 and size. `artifact_manifest.json` binds the
complete set. Keep raw tensor byte counts distinct from serialized file sizes.

`fitted.pt` holds normalization, all train arrays, seven complete fits, state
hash comparisons and 16.14 provenance. `prepared.pt` holds explanation-side
primary methods, full samples, quadrature pairs, all hybrid predictions, rankings,
background IDs, schedules and saved-boundary hashes. `evaluated.pt` holds all
audit arrays, five-background and noise studies, performance, full replacement
controls, complete replicate schedules/summaries and every distance/identity.
Only `sampling_replicates.samples` is replaced in its persisted metadata tree
by an external-reference record bound to `replicate_samples.pt` and its manifest;
the in-memory study retains the complete tensor. Serialize it directly without
recursive copying. Shared primary/background branches may alias in memory but
no draw or result is discarded. Package schedules and complete same-draw replay
arrays stay in `reference.pt`. Mechanism-only toys stay in `diagnostics.pt`;
the two editable recomputations stay in `play.pt`.

All exact-game comparison blocks retain `hybrid_prediction[N,256,B]` alongside
`coalition_values[N,256]`, feature Shapley and the other methods. Evaluate hybrids
in bounded batches. The primary64 alternate-background branch may reuse the
identical primary block. Archives can omit large payload bytes only when their
complete manifests, hashes, byte sizes and persistent external locations remain
bound and read-back verified.

Fresh reproduction later refits the seven models from source in another CPU
process and compares every deterministic array and scalar exactly, including
all schedules, samples, quadrature pairs, games, baseline methods, reference
replays, replicates, distances and curves. It must separately say that this
reproduces the same seeded known models and data. Source changes invalidate the
run. Verification reports support the lesson but do not replace visible results.

## Editable plays and visible result

The tiny play edits a threshold path's weights, bias, query, donor and K, then
recomputes sampled gradients, exact path allocation, exact four-coalition
Shapley and replacement effects. The real play defaults to `clean_0`, audit
index 0 (source ID 1), `primary64`, K=2048 and sigma 0. It may change model, any
held-out row, background, K and predeclared sigma. Mark audit reuse as
exploratory. Recompute from the edited values and frozen schedules; do not slice
a cached plot.

Visible results must include nested-budget convergence with SE, q256/q512 drift,
all-row signed comparisons with exact/DeepSHAP/endpoint-gradient controls, five
backgrounds, clean versus smoothed targets, the 64-replicate diagnostic, bounded
path/coalition distances, and audit curves with 100 random orders. Each headline
plot keeps its relevant control on the same visual surface.

## Feasibility evidence and freeze boundary

The data-independent probe in
`~/.cache/arena_extended/gradientshap/feasibility_untrained.json` used one
randomly initialized, never-trained float64 `8->32->32->1` ReLU network, four
CPU threads and synthetic eight-dimensional points. It performed no model fit
and no scientific attribution. Median throughput was about 4.16 million input
gradient examples per second at batch 32,768. Synthetic nearest-neighbor
throughput against 758 random rows was about 317,000 queries per second. These
rates exclude training, package adapters, allocation pressure, serialization,
plotting and real scientific arrays. They establish bounded feasibility only and
are not runtime promises.

The independent pre-fit review and root resolutions fix the namespace, package
coverage/tolerance and complete external sample retention above. A matching
zero-native-run freeze is still required before the first fit or attribution.
Publication and registry promotion additionally require complete learner execution,
independent fresh reproduction, actual browser review and the relevant audits.
