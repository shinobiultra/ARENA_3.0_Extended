# Execution evidence for 16.17

The teaching artifacts are the notebook pair and instruction page. These receipts support their
execution and provenance; they are not the lesson.

## How this fixture was produced

`execute_learner.py` completes the exercise notebook from its own adjacent answers,
checks exact source equivalence with the solution notebook, and executes all 49 cells
(22 code cells) in a fresh CPU kernel. It refits all seven known Concrete predictors,
checks their 21 model states, and evaluates all 272 held-out rows with 64 intact donors.
The fixed study retains 1,904 complete 256-coalition games and every estimate from both
proposals, 32 replicates, and seven budgets. The ten figures come from that execution.
`reproduce_native.py` separately refits from source without receiving fitted learner
payloads, and compares every deterministic array and container recursively.
`review_learner.py` renders the actual learner with exercise stubs preserved;
`publish_study.py` requires the passing reproduction and a truthful author review.
The normal registered CPU report then performs another fresh refit and reproduction.

`learner_execution.json` binds the published pair, page and ten figures to the complete own-answer
learner. `own_answer_execution.json` preserves its original execution receipt.
`fresh_reproduction.json` records a separate source-only child that refits seven models and
recomputes every deterministic artifact; it compares dtype, shape, values and container order.
`reference_metrics.json` is written by the normal registered report from its fresh arrays.
`author_visual_review.json` records the author's browser, figure and editable-play review.

Large arrays remain in the local cache at the absolute paths in these receipts. Each canonical
run retains six payloads: fitted states/traces, all donor predictions and games, sampling plans,
every estimate/SE/rank/constraint residual, exact references/variance, and actual package outputs.
Physical hashes and recursive value hashes cover all of them. The arrays are never replaced by
the small JSON receipts. Missing cache evidence makes verification fail; use the source commands
in `execute_learner.py`, `reproduce_native.py`, `review_learner.py` and `publish_study.py` to produce
and review a new complete run.

The original development run and corrected figure-axis run are retained under
`~/.cache/arena_extended/shapiq_estimators/`. Earlier analytic failures and their fixes remain there
too. Scientific seeds, budgets, splits, controls and tolerances were unchanged after native results.
The registered report always creates another fresh output directory. It does not load fitted models
into its child. Reproducing known seeded predictors is not independent model replication.

## Trusted implementation

The learner's visible float64 functions are checked against independent complete
derivative SII, hand-computable constant/additive/signed-pair/triple/four-way/dummy
games, full basis operators at four through eight players, finite boundary-penalty
limits, and actual `shapiq==1.5.2` ExactComputer, SHAPIQ and KernelSHAPIQ calls.
`protocol.md`, `data_manifest.json` and `protocol_freeze.json` fix the scientific
design before native execution. Exact model-state hashes bind the reproduced known
predictors. The comparison is a logical-query benchmark using a precomputed game
oracle; its precomputation cost remains explicit.

## Random seed

The manifest fixes model seeds 161410–161412, shuffled-target seeds 161420–161422,
intact-donor ordering seed 161460, and estimator replicate seeds 161750–161781.
Each proposal uses 32 replicates and prefixes of 16, 32, 64, 128, 256, 512 and 1024
draws. The detailed display remains `clean_0`, source ID 0, replicate 0, 128 draws.
Editable-play seeds are exploratory and never replace fixed-study evidence.

## Allowed tolerances

Fresh reproduction requires exact recursive dtype, shape, value and container-order
agreement: `rtol=0`, `atol=0`. Source and physical artifact hashes must also match.
Independent exact/package comparisons use `rtol=0`,
`atol=1e-10 + 1e-12 * max(1, max(abs(game)))`; the finite-penalty package
KernelSHAP-IQ comparison uses `rtol=0`,
`atol=1e-6 * max(1, max(abs(game)))` because its endpoint penalty is 1e8.
The fixed SVD rank threshold is 1e-12 times the original weighted design's spectral
norm. Immediate analytic tests retain their tighter case-specific bounds in
`estimator_methods_tests.py`. These tolerances are unchanged by environment revalidation.

## When to regenerate

Regenerate into new cache directories whenever an executed source, dependency
lockfile, manifest or scientific input changes, or when retained native arrays are
unavailable. Run the complete own-answer learner, separate source-only reproduction,
all ten figure and rendered-math/dropdown reviews, and both editable plays before
publication; then run the normal registered report and section checks. Do not
rehash or relabel historical execution as current. The old tested environment remains
in Git checkpoint `ff6d06e3`, and all earlier cache runs remain intact. A scientific
change requires a new explicit design decision; this environment refresh changes no
method, row, donor, seed, budget, control, display selection or numerical tolerance.
