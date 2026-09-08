# Datamodels completion checkpoint

## Current checkpoint — 16.10 published and reproduced; course goal still active

Section **16.10 Datamodels** is now published in
`chapter16_shapley_attribution_baselines/exercises/part10_datamodels`, with
`16.10_Datamodels_exercises.ipynb`, `16.10_Datamodels_solutions.ipynb`, readable
`lesson.md`, and the instruction page
`chapter16_shapley_attribution_baselines/instructions/pages/10_[16.10]_Datamodels.md`.
Original chapters 0–4 remain the pinned append-only base. Do not reset or clean
the dirty tree, remove previous evidence, or disturb the user's unrelated GPU
process. No subagents were used; all new training and reproduction ran on CPU.

### Learner and scientific result

The notebooks contain **71 cells, 31 executed own-answer code cells, eight
exercise groups, ten student functions, fifteen controlled figures and two
editable plays**. The original exercise → test → help → solution → experiment
flow is used. Each stub is completed only from its own following Solution
dropdown. All methods and study orchestration are visible; the converged
classifier fitter is explicitly reused from, and taught in, 16.9.

The learner implements uniform fixed-size masks, correct-class margins,
soft thresholding, the Lasso objective/KKT residual, accelerated proximal
optimization, validation-only regularization selection, supportive deletions,
class-matched random controls and centered coefficient similarity. Tests use
independent scalar, orthogonal, scikit-learn and SciPy oracles. Plain assertion
failures now give educational guidance and all twelve method tests print success.

The primary paper and author implementation were read before design. The selected
method is Ilyas et al. (ICML 2022), officially titled “Datamodels: Understanding
Predictions with Data and Data with Predictions.” Its linked PDF has an earlier
title. The author's README contains an unrelated arXiv ID; the lesson cites the
official PMLR page. PDF source SHA-256 is
`90eab4fd0cdcaec55a1fd08cb135e843e730af5b35fb72015f7d44a0f55f048d`.

The frozen UCI archive is the same pinned source as 16.9, but the new task uses
digits **3 and 8**, 128 training images and 64 labeled queries from the source's
separate test writer group. Source rows are selected by the prescribed hash,
not by model outcomes. No exact cross-split pixel duplicates occur. The manifest
SHA-256 is `153c7cd06df8988ff10032de373f8ccda1b42ba43bfe1da427357b3a5fb0f0e3`;
the protocol SHA-256 is
`01061ec436553e02d6496e8e2f5faa65d06cdff3345bca4d19f6ef490a8a06af`.
`protocol_freeze.json` records zero digit fits at the freeze. Scientific settings
were not changed after inspecting outcomes.

For each fraction 0.25, 0.5 and 0.75, train 1024 regression-training subset
models, 128 validation subset models and 256 held-out subset models. Each
65-parameter logistic classifier starts from zero and converges at gradient
infinity norm ≤1e-10. The 16-point Lasso path uses an unpenalized intercept and
checked KKT residual ≤1e-8; each query selects its penalty on validation subsets.
At fraction 0.5, retain training prefixes 64/128/256/512/1024 and all controls.

Query images are excluded from **classifier** fitting, but their labeled margins
are intentionally **datamodel regression targets**. The central held-out unit
is the training subset/model, not an image unseen by the datamodel. Reports and
audits explicitly encode this distinction rather than borrowing 16.9's
held-out-query fields. All datamodel paths and all 6528 deletion/control masks
are fixed before test-subset or counterfactual outcomes are read.

The full study contains **10753 classifier refits**, six separately scheduled
refit checks, ten independent SciPy optima, twelve independent native Lasso
checks and eighteen additional real-play refits. Counterfactuals use 24 fixed
queries, four rankings, budgets 2/8/16/32, and sixteen random deletions matching
each selected set's class counts. Nothing is selected by test outcomes.

The full-pool classifier gets 64/64 queries correct. Held-out-subset Lasso MSE
is 0.015387, 0.002209 and 0.000294 for the three fractions; the corresponding
worst per-query R² values are about 0.860, 0.941 and 0.978. Permuted-response and
mean-only controls predict little subset-dependent variation. Least squares is
competitive, and median selected coefficient counts are 126/126/128: this is
not a fabricated sparse success. Its 128-model sample-size instability is
shown alongside the nonzero singular-value condition number, about 218.

Same-size prediction does not establish cross-size calibration. A datamodel
trained at 0.25 has MSE about 0.344 on 0.75 subsets. At deletion budget 32, the
0.25 model predicts mean margin change −3.52 versus actual −1.68; the 0.75
model predicts −1.15 versus actual −1.62. Datamodel rankings flip 4/3/3 of the
24 fixed queries; pixel similarity flips 3. Matched-random cohorts flip 0–1
queries at budget 32 and zero at smaller budgets. Fixed-cardinality coefficient
offset ambiguity, dense solutions, extrapolation failure and individual query
failures are explained. A found deletion is an upper bound on minimum support,
not an exact minimum. No stochastic ResNet, full-paper or general data-value
claim is made.

### Execution, review and exact evidence

Persistent cache root: `/home/bitwise/.cache/arena_extended/datamodels`.

- `native-preparation-v1`: 3456 real fit/validation models and every prescribed
  path/control choice, 6.655 s; evaluation outcomes remained unopened.
- `native-full-v1`: first complete native study, 7.563 s, all independent checks.
- `own-full-v1`: complete learner, 15.804 s. Visual review requested clearer
  formatting, integer axes, shared deletion scales and additional interpretation.
- `own-full-v2`: 15.652 s, those presentation changes reviewed. Initial global
  audits caught a missing conversion mapping and test-feedback/bug-note details.
- **`own-full-v3`**: current complete own-answer execution, **15.481 s**, seventeen
  source inputs unchanged. All fifteen figures are byte-identical to reviewed v2.
  Executed notebook SHA-256:
  `3fc38bc43f5b88c064aee5f7643ce6ac49e761e6034b90873f742b1a7946aaf9`.
  Native directory: `figures/native-9otfrd6o/study` inside that run.
- **`fresh-native-v3`**: complete separate reload/refit/reproduction, **7.663 s**.
  All **263 tensors / 5546159 elements and 327 scalar values** agree exactly.
- The accepted current normal report performed another full native retraining at
  `report_reproductions/report-xuz1z3e7/fresh_cpu`. It binds **47 source inputs**,
  all freshly rechecked, with no verification failures. The normal command uses
  `--section 16.10` and does not count cached fits as retraining.

Each complete native run retains three `.pt` payloads totaling **35533656 bytes**.
Keep these files in persistent storage. Read-back-verified repository archives
bind the external payloads by exact path/hash/size, rather than duplicating them
into Git. Eight historical archives are in `development_evidence`; current
`expected_outputs` archives are `full_own_answer_v3` (52 members),
`fresh_native_v3` (33), `normal_report_source_v2` (49),
`normal_report_native_v2` (33), and the final release-gate checkpoint.

Every figure was inspected directly or was byte-identical to an inspected
version. Current browser checks open all **24 dropdowns** and render **58 math
expressions with zero errors**. Current centering/introduction views were
inspected, supplementing the previous full figure and browser reviews. All
39 Python blocks were formatted with executable AST parity; docstring
indentation was normalized where necessary.

### Integration and remaining route

Navigation and conversion mapping are appended to
`infrastructure/core/config_extension.yaml`; regenerate `config.yaml` using
`uv run python scripts/build_merged_config.py --write --check`. Do not edit
`config_original.yaml`. The new method row, hard-exercise row, root README link,
style status and **only** `data_attribution.datamodels` requirement are integrated.
The course surface audit follows actual subset fitting, Lasso validation,
intervention and fresh comparison calls. Mutation tests reject missing methods,
refits, independent checks and mislabeled held-out evidence.

Final focused validation: **78 tests passed in 11.43 s**. Original preservation,
course surface (13 chapters), style depth (52 pages), hard-exercise ladder
(51 rows) and report evidence (54 reports) pass. The global registry remains
**205 requirements: 47 TAUGHT, 36 PREFLIGHT_ONLY, 20 TOY_ONLY, 13 DEFERRED and
89 UNIMPLEMENTED**. Completeness fails on exactly those 89 requirements.
Readability still reports 78 older issues, with none in 16.10. Passing this
section's gates does not complete the course goal.

Continue with the already selected **Dataset cartography** requirement, then
training-run simulations / richer In-Run Data Shapley as appropriate. Read the
primary method and relevant original ARENA example, write the learner arc, and
freeze selection before fitting. Keep confidence/variability/correctness maps,
actual training dynamics, data-pruning retraining and matched controls visible;
do not substitute a fabricated metric scatter for a learned model. Also return
to partial native lessons and global readability before calling the whole
extension finished. Preserve original chapters and the selected syllabus.

The 16.9 checkpoint below is historical and remains valid for that section.


The initial learner plan follows.

# Selected learner: Datamodels

Current learner failure: `data_attribution.datamodels` is selected but
UNIMPLEMENTED. There is no target notebook, solution notebook or instruction
page to repair. Section 16.10 will be appended beside the published 16.9.

Target section claim: predict the correct-class margin of a genuinely retrained
classifier from its training-set membership mask, then measure how far those
predictions survive targeted removal of training images.

Signature result to create: held-out-subset predicted-versus-retrained margin
plots; sample-complexity and sparsity curves; image galleries of helpful and
opposing examples; predicted versus actual deletion curves with class-matched
random removals. Keep per-query errors and failures, not only pooled correlation.

Student-implemented functions: uniform fixed-size subset sampling, correct-class
margins, soft thresholding, Lasso objective and KKT residual, an accelerated
proximal solver, a regularization path with validation-only selection, prediction,
targeted removal and class-matched random removal, and centered coefficient
similarity. The previous section's converged logistic fitting can be reused as
explicitly linked plumbing; the datamodel and its experiment must remain visible.

Controls/baselines: intercept-only and minimum-norm least squares; permuted
subset-response association; all three prescribed subsampling fractions; matched
random removals and pixel-neighbor removal; same-cardinality gauge invariance
versus changed-cardinality ambiguity; independent scikit-learn Lasso and SciPy
classifier checks. Deterministic convex training isolates subset effects; it does
not estimate the stochastic-training noise floor of the paper's neural models.

Play cells: change a subset fraction or interaction strength in an exact fixture;
choose a real target image, removal budget and subsampling fraction, then refit
the edited training set alongside matched random controls.

Files to modify: new `exercises/part10_datamodels` under chapter 16; its new
instruction page and figure assets; append-only course navigation and relevant
topic/status metadata only after own-answer execution, fresh reproduction and
visual review. Original chapters 0–4 are unchanged.

Tests to run: independently derived margin/threshold/KKT fixtures, sklearn Lasso
prediction/objective parity including rank-deficient fixed-size masks, exact
small-subset enumeration, disjoint mask and data provenance checks, no test
response access during fitting/selection, class-matched intervention checks,
complete own-answer execution and fresh native retraining, rendered dropdown/
math/figure inspection, then current course/preservation/report gates.

## Inspected before writing this plan

- Original ARENA 0.3 Optimization exercise and solution notebooks, including
  the implement/plot/play/help pattern around `opt_fn_with_sgd`.
- Published 16.9 exercise, solution and instruction page, data loader and builder.
- Ilyas et al., ICML 2022, PMLR 162, sections 1–4 and the author's code README.
  Primary PDF: https://proceedings.mlr.press/v162/ilyas22a/ilyas22a.pdf
  SHA-256: `90eab4fd0cdcaec55a1fd08cb135e843e730af5b35fb72015f7d44a0f55f048d`.
  Downloaded source is retained under
  `/home/bitwise/.cache/arena_extended/datamodels/sources`.
  The official proceedings title is “Datamodels: Understanding Predictions with
  Data and Data with Predictions”; the linked PDF carries the earlier title.
  The README's unrelated arXiv identifier is not used as the paper citation.

The scientific protocol is frozen separately before the first digit fit. This
plan is implementation work in progress, not an earned TAUGHT/course-ready label.
