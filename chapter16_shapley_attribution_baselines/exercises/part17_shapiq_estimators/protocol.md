# 16.17 — One Query, Many Interactions: fixed study design

This design is recorded before any native estimator run or new native fit in
16.17. The known Concrete predictors and data from published 16.14/16.15 are
reproduced, not rediscovered. The target is the accuracy of two sampling algorithms
on the same finite games. No predictive or causal superiority claim is tested.

## Scientific scope

Use all eight features, all seven predictors (clean_0..2, shuffled_0..2, ridge),
and every held-out row: 153 explanation and 119 audit rows. Fit on the same 758
training rows; reproduce all 21 initial/standardized/raw state hashes through the
published 16.15 checker. The unchanged published data manifest pins the byte source,
recipe-disjoint splits, model hyperparameters and state hashes. Each absent feature
comes from the same intact training donor. The primary bank is the first 64 rows
under the established background seed 161460. Outputs are compressive strength in
MPa. The 1,904 games have 256 coalitions each and 64 donor predictions per coalition.

Only ordinary singleton Shapley and pair SII are estimated. SHAP-IQ uses its
shared-evaluation importance-weighted formula. KernelSHAP-IQ first estimates the
Shapley main effects, then fits the pair residual design. Its exact boundary-weight
limit minimizes boundary least-squares error, then the interior objective in the
remaining nullspace. It does not enforce zero residual on every boundary row.
The full basis operators for n=4..8 and independent finite-penalty/package checks
must pass before native execution. Higher-order WLS conjectures are not tested.

## Fixed sampling and cost

The exact boundary contains sizes 0, 1, 7 and 8: 18 unique coalitions. Interior
sampling is iid with replacement on sizes 2..6. Two proposals are retained: kernel
(per-coalition probability proportional to 1/choose(6,size-1)) and uniform over
individual interior coalitions. Size-then-uniform-subset sampling avoids enumerating
the power set. All duplicate draws remain in the Monte Carlo average and SE.

Use 32 replicates for each proposal. Replicate r uses the local CPU Torch generator
seed 161750+r for both proposals, paired by replicate identifier; the differing
proposal transforms need not yield identical masks. Generate 1,024 draws once per
proposal/replicate, then take prefixes of 16, 32, 64, 128, 256, 512 and 1,024.
For a given proposal/replicate/prefix both estimators receive exactly the same
queried values and masks. The same content-independent plans apply to all 1,904
games. Record source masks, per-set probabilities, duplicate counts and actual
unique queried masks; no seed is selected by an outcome.

Report separately: K draws, 18+K nominal requests, unique coalitions in each
individual run, and unique-coalition count times 64 underlying donor predictions
per model/row. Exact enumeration costs 256 coalitions, or 16,384 donor predictions
per model/row. All-game reference construction costs 487,424 coalitions and
31,195,136 donor predictions. The benchmark uses a complete precomputed model-game
oracle to make repeated sampling studies quick. Each estimator only receives the
queried subset; the full table is reserved for the external evaluator. These are
logical query-budget experiments, not measured speedups over direct model calls.
The precomputation cost is disclosed rather than charged away. More draws can cost
more than exact enumeration in this small game; unique counts and draws differ.

## Controls, outcomes and displays

All four estimator/proposal combinations are shown. Compare absolute pair error
in MPa with exact derivative SII and a zero-pair baseline. Retain complete signed
estimates, standard errors for SHAP-IQ, first-stage main effects, free-space rank,
singular values and equality residuals. Rank deficiency stays visible; the fixed
SVD threshold is 1e-12 times the original weighted design's spectral norm, and the
minimum-norm free solution is returned. No ridge, reweighting or result filtering.

Compute exact iid SHAP-IQ variance by summing contributions over all 238 possible
interior coalitions under the declared proposal. Compare predicted variance/K,
mean estimated SE squared, and actual replicate squared errors. A normal band is
only a diagnostic, not a coverage guarantee or uncertainty about model training.
Replicates share plans across rows, so rows are not treated as independent sampling
replicates. Summaries first weight recipe groups equally within each split, then
average pairs. Show both held-out splits and every model; do not select based on
audit errors. Exact constants/additive/signed pairs/triples/four-way/dummy games
are teaching controls. Finite SHAP-IQ noise can make additive estimates nonzero.
Shuffled predictors need not be additive; their predictive errors supply context.

Fixed detailed display: clean_0, explanation source ID 0, pair (0,1), replicate 0,
128 draws. Show all 28 signed pair values and errors, including zero/small terms.
Budget/proposal/row/seed edits appear only in separately labeled exploratory plays.
The cold open uses T={0,1} in a four-player game; the boundary-limit example uses
a pure four-way unanimity game. Plot choices do not depend on native outcomes.

Reference checks: independent complete derivative SII; all-basis equality n=4..8;
finite boundary penalties 1e2/1e4/1e6/1e8; actual shapiq==1.5.2 ExactComputer,
SHAPIQ and KernelSHAPIQ on constant, pure-full and seeded random exact games at
n=4,6,8. Native package exact/full-budget checks cover source ID 0 for all seven
models. Strict deterministic equality bound is 1e-10 + 1e-12*game_scale; finite
package KernelSHAP-IQ uses 1e-6*max(1,max(abs(game))) owing to its 1e8 endpoint
penalty and normal-equation solve. Bounds cannot be widened after native outcomes.

## Evidence, execution and review

Keep full fitted states, training traces, row/donor identities, all donor predictions,
complete reference games, exact SII, sampling plans, every replicate/row estimate,
SE, rank and residual, package comparison values, figures and executed notebooks.
Float64 throughout; no unrecorded downsampling. Chunking only changes resource use.
CPU four threads, at most 20 minutes, sampled process-tree RSS below 4 GiB, and
deterministic evidence below 1.5 GiB per canonical run. These are engineering caps,
not scientific pass conditions. Comparisons and controls may have unfavorable
results and still constitute valid course material.

Execute the whole learner completed from its own adjacent solutions, then reproduce
the deterministic outputs in a fresh process from source with no fitted learner
payload supplied. Bind source hashes, dependency hashes and the unchanged scientific
protocol in each execution record. Preserve failures and document corrections.
Review actual figures, rendered mathematics, dropdowns and editable plays. Root is
the implementation author/reviewer while authorized subagents are unavailable;
record that scope honestly. This section introduces no fabricated independent
approval. The separate independent publication gate for 16.16 remains unchanged.

Primary sources: [SHAP-IQ, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/file/264f2e10479c9370972847e96107db7f-Paper-Conference.pdf),
[KernelSHAP-IQ, ICML 2024](https://proceedings.mlr.press/v235/fumagalli24a.html).
