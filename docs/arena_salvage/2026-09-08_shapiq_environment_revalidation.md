# 16.17 current-environment revalidation plan

Current learner failure: the published 16.17 pair, page, ten figures and receipts
were produced before the September 8 lockfile update. The scientific protocol and
manifest still validate, and the package maintenance checks preserve NumPy 1.26.4,
Torch 2.12.1+cu132 and shapiq 1.5.2, but the historical learner and reproduction
receipts cannot establish execution in the current locked environment. The expected
outputs README also lacks the standard fixture-provenance headings checked by the
current course tests.

Target section claim: keep the existing claim unchanged. The learner estimates every
pair interaction from a subset of coalition queries by implementing SHAP-IQ's shared
query mean and pairwise KernelSHAP-IQ's sequential weighted regression, then compares
both with exact same-game SII on fixed Concrete-strength model games.

Signature result to create: rerender the existing ten results from a complete current-
environment own-answer run. They must still show one-query contributions, sampling
probabilities, the finite boundary limit, model controls, complete query-budget errors,
signed pair estimates, sampling variance, unique-query/rank behavior and the native
editable-play result. No display, row, scale or scientific selection changes.

Student-implemented functions: retain the existing nine visible functions: coalition
design, SII shared-query weights, size-then-subset sampling, SHAP-IQ mean/SE, query
caching, constrained least squares, pair residual design, finite kernel weights and
sequential pairwise KernelSHAP-IQ. Rerun all immediate tests inside the own-answer
notebook; do not move scientific code or change methods.

Controls/baselines: retain exact constant, additive, signed-pair, triple, four-way,
dummy and mixed games; exact derivative SII; the zero-pair baseline; kernel and uniform
proposals; three shuffled-target predictors and ridge; finite sampling variance; rank
diagnostics; and shapiq 1.5.2 parity. Revalidation does not depend on estimator
superiority or favorable control outcomes.

Play cells: rerun and review the existing analytic coalition/game play and native
model/recipe/proposal/budget/seed play, including the declared valid variants and
invalid-input checks. They remain exploratory and do not alter the fixed study.

Files to modify: only the existing 16.17 exercise/solution pair, instruction page and
`shapiq_*.png` assets through the existing publisher; 16.17 expected-output receipts
and verification report; `expected_outputs/README.md` for truthful fixture provenance;
and this revalidation record. Root's subsequent integration review also authorized
wrapping five existing prose paragraphs in `lesson.md` without changing their text,
wrapping the section README commands, and adding concrete success messages after
four already-existing tests in `estimator_methods_tests.py`. No assertion or method
changes. Preserve all earlier cache directories. The initial current-environment
paths are `new-environment-learner-v1` and `new-environment-reproduction-v1`.
The prose-wrapped v2 run and feedback-updated v3 run are also retained. A packaging-only
fix in the source-bound `part15_gradientshap/publish_study.py` requires final v4
execution after that shared dependency is fixed, followed by the normal report
runner's newly allocated reproduction. The 16.15 worker owns that publisher edit.

Tests to run: run the 31 focused scientific tests under `uv run --locked` with CPU-only
four-thread variables; execute the complete own-answer learner into a new directory;
perform a separate source-only fresh refit/reproduction and require exact recursive
dtype, shape, value and ordering parity; render the actual learner; inspect all ten
plots and the page's MathJax, dropdowns, Solution blocks, overflow and both plays;
publish only after those checks pass; then run the normal registered CPU report with
`--section 16.17 --max-vram-gb 0` and the relevant 16.17 ownership/freshness gates.
Record exact current counts, hashes, runtime, memory, output size and limitations.

Scientific-change boundary: the fixed protocol, data split, models, donors, seeds,
budgets, replicates, controls, estimator definitions, selected display and numerical
tolerances are immutable in this pass. Any scientific mismatch stops publication and
requires coordination rather than an amendment made during revalidation.

## Retained intermediate checkpoints

- `new-environment-scientific-tests-v1.log`: 31 passed in 1.82 seconds before
  the feedback-only test edit.
- `new-environment-learner-v1`: complete own-answer PASS, 49 cells / 22 code cells
  / ten figures, 28.6995 seconds inside the wrapper. The wrapper recorded 29.3269
  seconds, 2,689,994,752 sampled process-tree RSS bytes and 952,263,773 output bytes.
- `new-environment-reproduction-v1`: PASS, all 663 arrays / 118,275,724 elements /
  231 scalars / 160 containers matched exactly in 21.7069 seconds.
- All ten v1 PNGs received direct author visual inspection. The v1 play recheck
  passed 12 native variations, six analytic variations and four invalid-input
  checks using actual completed-learner function cells. The root's normal HTTP
  browser review saw 46 rendered MathJax expressions, zero math errors, ten loaded
  images, all 30 dropdowns open, eight Solutions and no horizontal overflow.
  This intermediate inspection is not the final publication receipt.
- Root requested five prose wraps after that review. Whitespace-normalized lesson
  text was verified unchanged. `new-environment-learner-v2` and
  `new-environment-reproduction-v2` both passed; all ten v2 PNG hashes exactly match
  the inspected v1 figures. The v2 wrapper recorded 28.9934 seconds,
  2,694,356,992 sampled process-tree RSS bytes and 952,263,952 output bytes.
- The subsequent four success-print additions leave all assertions and tolerances
  unchanged: test AST equality was checked after removing print statements.
  `new-environment-scientific-tests-v3.log` reports 31 passed in 1.80 seconds.
  `new-environment-learner-v3` completed PASS: wrapper time 28.9591 seconds,
  2,844,839,936 sampled process-tree RSS bytes and 952,264,353 output bytes.
- The root then requested a packaging-only fix in 16.15's publisher. That file is
  among the explicitly recorded 16.17 source inputs. The v3 process finished before
  the packaging worker was authorized to edit it. Its unchanged-source receipt is
  retained, and no v3 reproduction or publication is presented as current afterward.
  Final v4 execution, review and publication are recorded below.

## Final current-environment execution

Final source version: `new-environment-learner-v4`, paired with
`new-environment-reproduction-v4`. The learner binds `uv.lock` SHA256
`e96e946bb00f50592d32c1b3bf453fca4482a0097731120defbebcd978f019eb`
and the finalized packaging-only 16.15 publisher SHA256
`ea001e7f27cadc4b37a9d63f3766cc67e7450c4ce053a5c2a4f0466beb5034a0`.
All recorded source hashes currently match their files and retained snapshots.

- Complete own-answer learner: PASS, 49 cells, 22 code cells, 22 executed code cells,
  ten embedded figures and no changed sources. Notebook SHA256:
  `f95c84427e55494c4717bd06ab57dc3ad5720aa9e3f725ffdac6093be90f7dd0`.
- Learner execution: 28.9134 seconds. Wrapper wall time: 29.6245 seconds. Sampled
  process-tree RSS: 2,800,992,256 bytes at 0.25-second intervals (115 samples).
  Output before review: 952,264,799 bytes. The 20-minute / 4 GiB / 1.5 GiB caps pass.
- Separate fresh source-only reproduction: PASS in 22.5406 seconds. All 663 arrays,
  118,275,724 elements, 231 scalars and 160 containers match exactly, including dtype,
  shape, value and ordering. It refits models and receives no fitted learner payload.
  Every recursive native inventory also equals the earlier current-environment v1
  inventory; the formatting, feedback and packaging changes altered no native value.
- All ten v4 PNGs are byte-identical to the ten v1 PNGs directly inspected during
  this revalidation. `html_review/figure_author_review.json` records this linkage
  explicitly, with the exact earlier inspection receipt and its hash.
- The v4 play recheck again passed 12 native variants, six analytic variants and
  four invalid-input checks, executing the actual completed-learner function cells
  and all nine immediate method tests. Native variants cover two through 4096 draws,
  both proposals, seeds, clean/shuffled/ridge predictors and explanation/audit rows.
- A temporary change to the first arguments of two feedback prints was undone before
  any further execution. The tests file again matches the actual v4 source snapshot
  byte-for-byte. Root chose to correct the separate static feedback detector to
  accept informative f-strings; there is no v5 run and no changed scientific code.

## Final publication and checks

Publication PASS: the canonical exercise/solution pair, page 17, all ten `shapiq_`
figures and expected learner/native/review receipts now bind to final v4 execution.
The combined review explicitly records author review, never independent approval.
`html_review/root_browser_review.json` binds actual v4 HTML, rendering receipt,
execution and five inspected screenshots. The normal HTTP browser review confirms
30/30 opened dropdowns, eight Solutions, 46 rendered MathJax expressions, zero math
errors, ten loaded images and no horizontal page overflow. Pair/nullspace derivations,
budget curves, signed pair results and the native play were readable in actual
browser screenshots. The eighteen v4 play PNGs also match the earlier exercised
current-environment variation PNGs exactly.

The normal registered CPU entry point passed with `--section 16.17 --max-vram-gb 0`.
Its additional fresh refit/reproduction is retained at
`report_reproductions/report-xqkhcg5z/fresh_cpu/`; it again matches all 663 arrays and
118,275,724 elements exactly, in 22.2760 seconds. The report's fresh-reproduction
receipt SHA256 is `d4d92e5c74c9ee94da3d23bce82af9f314603aa45bd18db5a00fd7ab8bdc2d3d`.
`new-environment-normal-report-v4.log` records one report written and one accepted.

Final tests and gates:

- `new-environment-section-tests-v4.log`: **38 passed in 1.95 seconds**, covering
  the scientific method/reference/study suites and direct learner ownership,
  missing-method/test/figure/wrong-answer mutation and native coverage checks.
- `new-environment-section-gates-v4.json`: PASS for report schema, report input
  freshness, learner ownership and complete native coverage/receipt contract.
- Separate integration worker's stable v4 recheck: `native_receipt_blockers=[]`,
  `row_blockers=[]`, 13/13 visible test feedback and 47 assertions; 365 targeted
  integration tests passed in 2.45 seconds. The root-owned hard ladder has zero
  blockers. These are integration results, not independent scientific approval.
- Root's `new-environment-original-preservation.log`: original ARENA preservation
  **PASS, zero blockers**, checking 185 changed paths at that audit.

`expected_outputs/execution_review.tar.gz` retains the actual notebook, source
snapshots, figures, browser/play review and complete local check receipts. Every
member and the archive itself were reread and SHA256-verified; no `.pt` payload is
included. Full numerical arrays remain at the physically hashed final cache paths.
The final archive contains 92 regular members in 13,995,638 bytes, SHA256
`470407a4d90c3083efed4247b4be1ae32180162cd13120ca01e892fb7863fce4`.
`execution_review.tar.gz.review-bundle.json` seals its bounded member inventory and
all six external numerical payloads. Five CUA screenshot files retain their reviewed
`.png` names but contain JPEG bytes; the seal explicitly records the detected format
and extension mismatch without rewriting the reviewed evidence.
The earlier repository archive and sidecar were copied intact to
`historical-publication-ff6d06e3/`; earlier learner/reproduction caches remain intact.

Remaining limitations are unchanged: this is a pairwise logical-query benchmark
over a precomputed Concrete model-game oracle, reproducing known seeded predictors.
It does not establish direct-inference speedup, independent predictive replication,
causal ingredient effects, higher-order KernelSHAP-IQ validity or course completion.
Finite additive Monte Carlo noise, shuffled-model interactions and low-budget rank
deficiency remain in the figures. No method, seed, split, donor, budget, replicate,
control, selected display, assertion or numerical tolerance was weakened.
