# SHAP-IQ and KernelSHAP-IQ learner plan

Current learner failure: selected requirement `shapley.shap_iq` remains
PREFLIGHT_ONLY. The 16.3 exercise/solution pair and instruction page implement
exact SII, a permutation estimator and package parity, but neither named estimator.
The original 0.4 Backprop pair/page supplies the style comparator: short conceptual
steps, meaningful student code, immediate tests, concrete outputs and useful hints.
The new 16.17 pair/page does not yet exist. Originals and completed sections remain
unchanged.

Target section claim: how accurately can we estimate every pair interaction when
we can query only a fraction of coalitions? Implement the shared-evaluation SHAP-IQ
estimator and the sequential pairwise KernelSHAP-IQ estimator on the same sampled
coalitions, compare with exact same-game SII, and distinguish Monte Carlo noise,
rank deficiency, model quality and a change of interaction definition.

Provisional section: **16.17 — One Query, Many Interactions** in
`chapter16_shapley_attribution_baselines/exercises/part17_shapiq_estimators`.
Only the already selected SHAP-IQ / KernelSHAP-IQ requirement is addressed.
SPEX/ProxySPEX, higher-order conjectures and other estimators are not added.

Signature result to create: show the contribution of one queried coalition to
every pair, then draw signed error heatmaps and complete error-versus-query curves
on fixed real model games. Compare SHAP-IQ's predicted sampling variance with
observed replicate variation. The exact all-coalition result is a reference and its
256-query cost is visible; sampled estimators never receive the full reference table.

Student-implemented functions: bitmask coalition design; SII shared-evaluation
weights; a size-then-subset sampler that does not enumerate the power set; SHAP-IQ
importance-weighted mean and standard error; equality-constrained weighted least
squares; pair residual design; sequential KernelSHAP-IQ; and an explicit counted
query wrapper. Exact derivative SII is a transparent prerequisite/reference.
Each exercise gets an original-style difficulty/time block, tests, expected output,
help, interpretation and an adjacent Solution dropdown.

Primary definitions: SHAP-IQ, NeurIPS 2023, Theorem 4.1 / Definition 4.2; pairwise
KernelSHAP-IQ, ICML 2024, Proposition 3.2 / Theorem 3.7 / Algorithm 1. Downloaded
primary PDFs and byte identities are under
`~/.cache/arena_extended/shapiq_estimators/references/`. Pairwise is the proved
case (n >= 4). Higher-order WLS claims are explicitly outside this lesson.

Numerical point to derive and test before native execution: pairwise infinite
boundary weight minimizes a generally nonzero boundary residual. It does not force
every singleton/complement residual to zero. Implement that limit through the
boundary least-squares minimizer and its nullspace, then minimize the interior
objective within that set. Derive the incidence constraint explicitly and check
the resulting complete linear operator against discrete-derivative SII on every
basis game, n=4 through 8. Compare with increasing finite penalties and the pinned
package. Do not silently identify a naive pair polynomial fit with SII.

Controls/baselines: exact constant, additive, signed pair, triple, four-way, dummy
and mixed games; complete-operator equality; zero-interaction baseline; uniform
versus kernel sampling; all known clean and shuffled-label Concrete predictors
plus ridge; rank/condition diagnostics; exactly enumerated same-game SII; package
estimator and exact-index checks. Finite SHAP-IQ can produce nonzero additive
interaction estimates: show its error rather than promise a zero at every budget.
Shuffled predictors need not have zero interactions either.

Play cells: edit one coalition and watch its update to all pairs; vary query
budget/seed/proposal/real recipe while inspecting the same-game exact result and
actual queried masks. No play rewrites the frozen study or selects headline rows.

Native study candidate: reuse the known eight-feature Concrete model/data recipe
from published 16.14/16.15 without editing either. Refitting is self-contained and
must reproduce the 21 known model states. Use all 272 held-out rows and all seven
predictors, with one fixed intact 64-donor background. The method comparison is
paired by sampled coalition stream and uses all predeclared replicates. Record
draw requests, unique coalition queries, underlying donor predictions and exact
reference costs separately. No claim of a novel predictive experiment or physical
causality. Final counts, seeds, budgets and evidence shapes must be fixed in the
protocol before inspecting native estimator results.

Files to modify: new part17 methods/tests, small data/study plumbing, lesson,
figures, builder and execution/evidence files; new pair/page/assets and exact
registration entries after execution and review. Do not touch frozen 16.16 sources.
Its independent publication review remains separately pending.

Tests to run: hand-derived gamma rows and basis operators, offset/linearity/dummy
properties, exact expectation and variance identities, distribution normalization,
query counting and cache correctness, infeasible/deficient linear systems, finite
penalty convergence, primary-paper/package parity, complete own-answer execution,
fresh source-only reproduction, actual browser math/dropdown/figure review, play
variations, original preservation and section integration audits. Tests establish
correctness; observed estimator superiority is never an acceptance condition.

Execution ownership: root implements and reviews locally while the authorized
gpt-5.6-sol/xhigh reviewer agents are unavailable through account limits. Record
author review truthfully. No independent verdict, quota reset, model switch or
subagent execution is inferred. This plan does not modify 16.16's publication gate.


## Completed implementation and validation — September 7

The canonical 16.17 pair/page and ten actual learner figures are published and registered.
The own-answer learner passes 49 cells, 22 code cells and ten figures in 29.4 seconds,
within the predeclared CPU/RSS/output caps. Every one of the 1904 model games is retained;
47,767,552 signed pair estimates span both estimators/proposals, 32 streams and seven budgets.
A separate source-only process refits all models and exactly reproduces 663 arrays with
118,275,724 elements. The normal registered report repeats those refits and comparisons.

The 31 analytic/reference/study tests pass; the 56-test course/report integration suite passes.
Twelve native plays, six analytic plays and four invalid inputs were checked. Root inspected
all figures and the actual notebook rendering: 30 dropdowns, eight Solutions, 46 equations,
zero math errors/overflow. Author review is explicitly not independent review.

Preserved development corrections: theoretical-zero nullspace rank tolerance, actual package
constructor keywords, and a convergence display scale that previously compressed meaningful
errors. Reporting corrections replaced inherited schema metrics and clarified a negative-control
label used by the existing runner. None changed scientific choices or acceptance tolerances.
Evidence is in `~/.cache/arena_extended/shapiq_estimators/` and the section's `expected_outputs`.
`HANDOFF.md` gives exact paths and hashes. Original preservation passes with zero blockers.
The selected SHAP-IQ requirement is TAUGHT; this does not close the full 205-item course goal.
