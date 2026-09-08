# Expected Gradients learner plan

**Historical design plan.** The September 7 implementation, complete native
execution, fresh reproduction and browser/figure reviews have passed. The user
then requested a graceful stop. Read
`2026-09-07_gradientshap_stop_checkpoint.md` for current evidence and the remaining
report-review/publication/integration steps. Do not repeat the pre-fit freeze or
treat the original missing-implementation description below as current status.

Current learner failure: the selected `shapley.gradientshap` requirement has no
path-sampling implementation, baseline-distribution study, or computational
intervention comparison. Section 16.14 explicitly distinguishes its endpoint
gradient baseline from Expected Gradients. No target notebook, solution notebook
or instruction page exists. Original 0.4 Backprop and the current 16.14 learner,
solutions and page provide the style reference; original chapters 0–4 stay fixed.

Target section claim: build Expected Gradients from explicit donor and path
samples, separate sampling error from the explanation target, and show how
Gaussian query noise changes that target. A complete continuous-path attribution
can differ from discrete feature-coalition Shapley. Append only the selected
GradientSHAP topic as provisional 16.15, titled **Expected Gradients: Which
Expectation Does GradientSHAP Estimate?**

Signature result: begin with a ReLU whose per-path contribution is either zero
or two. Build running estimates and ordinary sample standard errors, then show
all-row convergence, signed attribution comparisons, baseline movement, noisy
query targets and held-out replacement curves on the Concrete models. Quadrature
is an approximate numerical control; completeness is an expectation statement,
not a demand that every finite Monte Carlo vector sum exactly.

Student-implemented functions:

1. `sample_path_schedule(source_ids, background_rows, samples, seed)` returns
   donor indices and uniform alphas of shape `(N,K)`. Use independent local
   per-source-ID donor and alpha streams so prefixes are stable when K or the
   row batch changes. A single generator that first draws an `(N,K)` donor array
   and then an alpha array does not have this property.
2. `interpolate_paths(x, background, indices, alpha, noise=None)` returns points
   and deltas of shape `(N,K,D)`. Noisy queries must appear in both expressions.
3. `input_gradients(model, points, forward_batch=32768)` returns gradients with
   the same shape, preserving inputs, parameter gradients and model mode. The
   model must be a deterministic scalar function applied independently per row;
   the sum-of-outputs trick does not establish that property for arbitrary models.
4. `expected_gradient_samples(model, x, background, indices, alpha, ...)`
   composes the previous functions and retains per-draw contributions.
5. `summarize_expected_gradients(samples, budgets)` returns prefix means and
   ordinary plug-in SE: square root of unbiased sample variance divided by sample count.
6. `expected_gradients(model, x, background, indices, alpha, ...)` computes the
   full-background base, mean attribution and ordinary estimator standard error.
7. `gauss_legendre_pairs(model, x, background, n_nodes, row_batch=4, ...)`
   integrates every query/donor path in bounded batches. Finite quadrature must
   remain explicitly approximate on ReLU gradient jumps.
8. `noisy_expected_gradient_samples(model, x, background, indices, alpha,
   noise, ...)` makes the noisy-query target explicit and matches the clean
   estimator exactly when the supplied noise is zero.
9. `ranked_coalition_curves(values, phi, orders=None)` exposes raw insertion and
   deletion values, masks and feature orders. Primary audit curves use frozen
   explanation-only global orders; local play orders remain exploratory.

Each exercise needs its own immediate semantic test, expected output, help,
interpretation and following Solution dropdown, with original ARENA's fenced
Difficulty/Importance/time metadata. Every assertion needs a useful failure
message. Early diagnostics may use only functions defined earlier. Shared
validation, plotting and reference adapters are plumbing; the nine methods and
meaningful training/evaluation logic remain visible.

Controls/baselines: analytic affine, quadratic and ReLU paths; a threshold game
with unequal path credit but equal discrete Shapley; the exact-kink min circuit
with its derivative/implementation-invariance qualification; dummy and constant
functions; average attribution versus a mean baseline; Gaussian-smoothed ReLU's
analytic target; nested budgets and independent sampling replicates; quadrature
doubling; exact 256-coalition games; DeepSHAP and endpoint gradient-delta; clean,
shuffled and ridge models; five empirical backgrounds; 100 random feature orders;
an explicit distance diagnostic for interpolation points and coalition hybrids.

Play cells: move a path through a threshold by editing query, donor, gate and
sample count; choose a held-out recipe, model, background and noise scale and
recompute its explanations and replacement curves. Use the first audit recipe
as the default before examining any new attribution result.

Files to modify: new `part15_gradientshap` sources, lesson, paired notebooks,
instruction page and namespaced figures; append navigation and only the selected
registry row after full execution, reproduction and browser review. Preserve
16.14 and its current evidence. Research preparation remains in
`~/.cache/arena_extended/gradientshap/`.

Tests to run: hand-derived method tests; exact same-draw SHAP and Captum parity;
sampling-prefix, identity, dtype, model-state and RNG restoration checks; full
own-answer notebook execution; independent reproduction of every stored result;
browser math/dropdown/figure review; registered style, course, ladder, evidence
and original-preservation audits; honest full-course completeness audit.

## Native design decisions before implementation

Reproduce the same seven fixed Concrete predictors used in 16.14 from their
frozen training recipe, with train-only normalization and the same seeds. Keep
all model states and compare them with 16.14. This creates a self-contained
learner while comparing identical functions; it is not an independent model or
dataset replication. The existing data and model performance have already been
examined. Freeze the new attribution protocol before calculating its native
results, retaining the train/explanation/audit boundary for feature rankings.

The data are the same 1,030 rows and recipe-disjoint 758/153/119 split. Explain
all seven models on all 272 held-out rows. Preserve the primary 64 intact donors
and the nested 16/128 and two separately ordered 64-row backgrounds. All exact
feature games and comparisons use the same donors and raw MPa output function.

Use nested K budgets `[8,32,128,512,2048]`, paired across models. Retain full
primary Monte Carlo contributions and schedules: 31,195,136 contribution
coordinates occupy about 238 MiB in float64. Keep large tensors in the existing
persistent-cache evidence workflow, with hashes and exact fresh reproduction.
Evaluate both 256- and 512-node all-background quadrature on every primary
model-row, retaining both pair arrays and their drift. Do not use agreement with
quadrature as an exactness acceptance gate.

On clean_0 and all 272 rows, recompute all five background designs and noisy
query samples at standardized-input sigmas `[0,0.02,0.10]`, mapped into raw
feature units using train-only scales. The mean prediction over supplied noisy
queries is a Monte Carlo estimate of the smoothed target, not an exact population
quantity. Retain its sampling error and distinguish it from the clean target
and the sample-donor versus full-background base.

Predeclare sampling-replicate and OOD-diagnostic subsets before evaluation. Use
fixed source IDs and complete retained arrays, not rows chosen for an interesting
error. Bound quadrature and nearest-neighbor memory by batching. Benchmark only
untrained networks before settling these secondary budgets.

Reference parity uses all 1,904 model-rows with a separately frozen K=128 draw
table per package. SHAP 0.49.1 and Captum 0.9.0 consume random numbers in different
orders; equal seeds alone do not create matching draws. Restore global RNG,
default dtype, model mode and existing parameter gradients around author calls.
Disclose SHAP's default-dtype buffer failure, scalar output shape, unseeded Torch
smoothing stream and `variance/sqrt(K)` field. That field is neither ordinary
estimator variance nor standard error. Check positive-noise parity separately on
fixed analytic circuits. Benchmark full author coverage before freezing.

The remaining protocol details are the independent RNG namespace, replicate
count/source IDs, exact OOD subset, reference batching and numerical tolerances.
They require a short independent pre-fit review. No native attribution run or
new model fit is authorized until that protocol is frozen.

Primary reading: Erion et al. (2021), Expected Gradients; Sundararajan, Taly and
Yan (2017), Integrated Gradients; pinned SHAP 0.49.1 and Captum 0.9.0 source.
Attribution-prior training is a paper connection, outside this core study.
