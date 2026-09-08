# Bounded, visible ProxySPEX method API

## Plan recorded before implementation

- Current learner failure: no visible boosted-tree fitting, spectrum extraction,
  support selection or stage-error separation exists in the selected lesson.
- Target section claim: exact extraction explains the fitted tree ensemble;
  explaining the queried model additionally requires proxy fidelity.
- Signature result to create: model/proxy/extraction/truncation/refinement curves
  and a tree path expanded into signed Fourier terms (not produced by this module).
- Student-implemented functions: residual boosting, Boolean characters, path
  multiplication, tree/ensemble spectra, energy selection, grouped validation,
  ridge coefficient fitting and error accounting.
- Controls/baselines: independent full-cube truth, additive, signed parity,
  sparse polynomial, repeated feature paths, literal author threshold failures,
  shuffled responses and misspecified depth.
- Play cells: edit path constraints, tree depth or retained energy; the lesson
  builder will own those cells and their figures.
- Files to modify: this note, `proxy_methods.py`, `proxy_methods_tests.py` only.
- Tests to run: independent small-cube predictions/DFT, residual-tree accounting,
  exact prefix boundaries, duplicate grouping and fold-specific support selection.

Original 1.1's exercise/solution pair and page, the repository skill and the
September 8 learner plan/reference report were read before implementation.
There are no existing 16.19 notebooks/pages to inspect or modify at this stage.

## Prospective defaults, approved before native outcomes

The visible educational ensemble uses scikit-learn's `DecisionTreeRegressor`
as the **tree-fitting primitive**. Each tree fits the previous model's squared
loss residuals, with an explicit constant initialization and shrinkage update.
It is a genuine boosted-tree proxy. It is **not the author's LightGBM estimator**
and neither estimator nor hyperparameter grid is claimed to match LightGBM.

`PROXY_CANDIDATES`: depth `{1,3}` by tree count `{128,512}`, learning rate `0.1`,
minimum leaf rows `2`. Candidate order is depth first, then tree count. Three
folds; ridge alphas `(1e-6, 1e-3, 0.1, 1.0, 10.0)`; retained nonconstant energy
`0.95`. CV MSE selects the first minimum in the given candidate/alpha order.
No early stopping or outcome-dependent search is performed. The native harness
owns query budgets, masks, seeds, response values, timing and cache accounting.

The actual author comparison used LightGBM depths `{1,3}`, tree counts
`{500,1000,5000}`, rates `{0.01,0.1}`, five-fold R² scoring and RidgeCV over 100
alphas; see `docs/arena_salvage/2026-09-08_spex_reference_execution.md`.

## Public harness contract

```python
fit = fit_proxy(train_masks, train_values, fold_seed=seed)
proxy = predict_boosted_trees(fit["model"], audit_masks)
full = predict_spectrum(audit_masks, fit["full_spectrum"])
truncated = predict_spectrum(audit_masks, fit["truncated_spectrum"])
refined = predict_spectrum(audit_masks, fit["refined_spectrum"])
errors = stage_errors(audit_values, proxy, full, truncated, refined)
```

`fit_proxy` never accepts audit masks or audit values. Masks are a nonempty
binary 0/1 `(rows, features)` array and values a finite `(rows,)` array. Feature
1 means retained. Spectra map sorted unique feature-index tuples to signed
coefficients in `chi_S(x)=(-1)**sum(x[i] for i in S)`; `()` is the intercept.
They are Fourier coefficients, not Shapley interaction indices.

The result contains the fitted `model`, all three spectra, `selection`,
`proxy_cv`, `ridge_cv`, `folds`, and `accounting`. `proxy_cv` includes candidate
parameters, every fold loss, pooled validation loss, selected index/parameters;
`ridge_cv` includes the alpha list, every fold loss, pooled validation loss,
selected alpha and each fold's selected supports. `folds` records both training
and validation row indices. No values outside the supplied training arrays enter
any of these choices.

Identical binary masks are grouped before splitting. The best proxy candidate
is selected from grouped folds; its separately fitted fold models supply the
fold-specific spectra for ridge selection. Thus a validation response does not
select the support on which that same response is scored. Selecting the proxy
candidate and ridge alpha on these same folds is a development procedure, not
a nested-CV performance estimate. The common untouched audit remains necessary.

Energy selection always includes the intercept (zero if absent). It selects the
shortest descending squared-magnitude nonconstant prefix reaching **at least**
the requested energy fraction, with support tuples breaking ties. Zero energy
requires no nonconstant terms. Fraction zero retains only the intercept. Ridge
uses an explicit intercept column and penalizes it like the other coefficients,
matching `fit_intercept=False` algebra; SVD computes the stated ridge solutions.

`accounting` distinguishes input row count, unique masks, zero oracle calls in
this module, ensemble fits, individual residual-tree fits, ridge candidate fits,
and ridge matrix decompositions. The native harness must separately report raw
requested queries, unique model evaluations, cache hits and actual timings.

## Source fidelity and failure boundary

The source is the ProxySPEX paper, Appendix A.2, and the pinned author source
`spectral-explain/src/spectralexplain/utils.py` at commit
`61400b5388b8ecbf18fe18c4d821ffe3e8a5f9c8`, available under
`/home/bitwise/.cache/arena_extended/sparse_interaction_discovery/references/`.
The paper's split expansion is implemented with symmetric difference rather
than set union: `chi_S * chi_{i} = chi_{S xor {i}}`. This also handles repeated
feature constraints and impossible paths correctly. Constant leaves work too.

`author_strict_energy_selection` is a labeled reproduction of the author's
**selection block only**, using index-tuple supports. It deliberately preserves
the strict `abs(coefficient) > boundary` behavior, omitted intercept safeguard,
empty-spectrum error and zero-energy arithmetic. It is not the author pipeline
or a maintained-package call. Untouched author executions and their literal
errors remain in the reference report and original cached artifacts. In
particular, the real author pure-parity execution raised
`ValueError: Dot product shape mismatch, (1024, 12) vs (0,)` after selecting no
support. The corrected learner selection must never be relabeled that result.

Exact extraction cannot repair a poor greedy-tree fit, and truncation may
intentionally omit small true coefficients. Full-cube analytic checks establish
algebra and mechanisms only. This module creates no native-model result,
notebook, signature figure, play cell, learner receipt or completion claim.

## Analytic execution of the frozen defaults

On September 8, the full four-candidate grid ran in the repository `.venv`
(NumPy 1.26.4, scikit-learn 1.9.0). Training masks are
`default_rng(100).integers(0,2,(456,28),dtype=uint8)`; independent audit masks are
`default_rng(991).integers(0,2,(2048,28),dtype=uint8)`. All 456 training masks are
unique and none overlap the audit. Fold seed is 0. BLAS/OpenMP thread limits
were 4. These are analytic function evaluations, **zero language-model calls**.
Let `z=1-2*x`, for binary retention mask `x`.

| Analytic game | Selected depth / trees / alpha | Model to proxy RMSE | Proxy to spectrum RMSE | Model to truncated RMSE | Model to refined RMSE |
| --- | --- | --- | --- | --- | --- |
| `.3+z0-.7*z3+.2*z7` | 3 / 512 / .1 | 1.44e-8 | 1.19e-15 | .200000 | .200480 |
| `-1.25*z2*z4*z9` | 1 / 128 / 10 | 1.266984 | 1.11e-16 | 1.266929 | 1.282906 |
| `.3+x0-.7*x3+1.2*x1*x7-2.4*x2*x4*x9` | 3 / 512 / 1e-6 | .179285 | 2.08e-15 | .112962 | 2.84e-9 |

The additive example keeps 97.3856% nonconstant energy and intentionally drops
its smallest true term. Refitting does not recover that omitted support. The
parity fit fails and refinement makes it slightly worse; the true constant-zero
baseline has RMSE 1.25. Extraction is nonetheless exact to numerical precision.
The polynomial's selected 13 terms refine successfully. These distinct outcomes
must remain separate in the lesson; the successful polynomial does not justify
a general fidelity claim.

Each run fitted 13 ensembles, 16 ridge candidate solutions and 4 ridge SVDs.
Individual tree fits were 4,352 / 3,968 / 4,352; wall times on this machine were
1.70 / 1.82 / 2.15 seconds. Full/truncated support counts were 8/3, 17/10, 926/13.
These are local analytic fit timings, not native inference timings. No prospective
parameter was changed after observing these analytic results.
