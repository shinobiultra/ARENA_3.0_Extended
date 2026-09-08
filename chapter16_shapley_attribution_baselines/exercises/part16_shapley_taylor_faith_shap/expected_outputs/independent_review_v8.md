# Independent publication review — section 16.16, frozen development v8

**APPROVE_PUBLICATION for the exact v8 execution and source bindings in independent_recheck.json.**
R1, R2 and the separate current-environment gap E1 are resolved. There are no remaining blocking
scientific, learner, or visual findings in this reviewed candidate. This is an independent review by
`/root/astra_interaction_review` using Astra after development execution, not an
implementation-author approval. The previous v7 `BLOCK_PUBLICATION` record remains unchanged.

This verdict concerns the frozen experiment and its learner artifacts. Canonical publication, normal
registered-report execution, archive assembly and navigation/registry integration are subsequent
work, not accomplished by this review. Full-course readiness is outside this verdict. No repository
source, canonical notebook/page/asset, registry or original ARENA file was edited by the reviewer.
No new native study was fitted by the reviewer.

## Resolved findings and exact scientific effect

**R1 — resolved.** `interaction_study.py:1500–1509` now accumulates float64 squared feature
differences in feature-index order before the pinned Torch square root, as required by
`protocol.md:184–192`. The exact-minimum tie still chooses the smallest training source ID. The new
test at `interaction_study_tests.py:354–368` distinguishes the old reduction on a nonzero fixture: a
first squared coordinate of 1e16 followed by seven units. I independently ran that test against the
actual v7 source module in isolated copied test globals; it fails as expected. The identical test
and the existing duplicated-row tie test pass against current source. See `regression_recheck.json`.

I independently computed all 557,056 retained distances and nearest IDs against all 758 training
rows: 422,248,448 candidate comparisons. The independent route uses NumPy explicit
subtraction/multiplication and `add.accumulate`, followed by the pinned Torch square root. Every
retained distance and source ID matches exactly. Training rows, IDs, mean and SD are identical
between v7 and v8.

The full historical effect is **98,602 changed nearest distances and 108 changed nearest source
IDs**, maximum distance change 8.881784197001252e-16. I independently rebuilt every old and new
complete candidate tie set for all 108 changed-ID coordinates, not only the first-row sample from
the v7 review. Every old choice agrees with the old reduction and every new choice with ordered
arithmetic and smallest-ID selection. `independent_full_support.json` retains all coordinates,
query/donor source IDs, old/new selected IDs, tie sets and both candidate distances. For query
source 12, mask 107, donor 235, the old reduction chooses source 49 alone; ordered arithmetic ties
sources 29 and 49 at 1.364684967333407 and correctly chooses 29. A complete 1,797-array manifest
comparison finds changes only in the two support arrays. All other scientific array value records
are identical to v7.

An initial all-NumPy probe failed exact equality because NumPy and the pinned Torch implementation
return different last bits for some square roots. A focused stage comparison found identical
standardized inputs, squared differences and ordered sums, but 1,346 differing square roots among
the first 128 × 758 candidate distances. Example squared distance 21.348867736935944 gives NumPy
4.620483496013804 and Torch 4.6204834960138035. The successful independent route retains the
specified execution backend for this primitive; it does not relax tolerance or replace the ordered
arithmetic with the implementation under review. The unsuccessful exploratory probe and its log are
retained.

**R2 — resolved.** Amendment 7 (`implementation_amendments.json:249–347`) honestly spans the earlier
unrecorded README change and the new correction notice. It preserves the original bound README, the
accurate post-v7/pre-repair README, and the final source. I rehashed all amendment evidence files,
verified the first six entries are unchanged and called the actual `validate_freeze()` successfully.
The immutable protocol, data manifest, loader, zero-run freeze and pre-run resolution retain their
original bytes. The final README describes v7 as historical and v8 evidence as prospective at the
pre-execution amendment boundary; it does not claim a future successful run.

**E1 — resolved for v8, not retroactively for v7.** `interaction_reference.py:40` now requires
current maintenance lock e96e946bb00f50592d32c1b3bf453fca4482a0097731120defbebcd978f019eb. The live
lock, packet lock and source bindings agree. All five pinned shapiq 1.5.2 source/metadata hashes
remain unchanged and the guards at lines 118–128 remain active. `reference_environment()` passes in
current NumPy 1.26.4 / Torch 2.12.1+cu132 on CPU. The old lock and old receipts remain preserved.
The actual new v8 own-answer execution and source-only refit were revalidated; no v7 receipt was
relabeled as current-environment evidence.

## Numerical and source review

The packet frozen-source record, v8 execution, reproduction and amendment match the four hashes
supplied for independent review. All 30 section/lock records match live bytes and their retained
snapshots; all 22 learner inputs match execution and reproduction snapshots. Completed notebook
hash, all eleven PNG hashes and full source freshness pass. The two wrapped lines in builder/plotter
have full AST identity including constants.

I reopened and physically validated all 17 deterministic artifacts in both learner and fresh
directories, including the ten external donor shards, recursive value records and sealed preparation
boundary. The full comparison gives exact equality for 1,797 arrays, 94,604,271 elements, 26,348
records and 22,556 scalars. The unmodified publisher `validate_fresh()` passes against the current
sources. Its checks include seven refits, no cached learner model/payload inputs to the fresh child,
source snapshots and complete hybrid scope. This revalidates actual author execution evidence; the
reviewer did not run another native fit.

Scientific definitions were reviewed against the retained primary sources for
[Shapley-Taylor](https://proceedings.mlr.press/v119/sundararajan20a.html) and
[Faith-Shap](https://www.jmlr.org/papers/v24/22-0202.html), whose pinned source bytes were rehashed.
I separately assembled highest-order discrete-derivative operators for ST, Faith-Shap and SII,
rather than reusing the supplied Möbius formulas. All 2,992 empirical games and 83,776 pair
coordinates per method pass the strict frozen bound. Maximum absolute errors are ST 1.4211e-13,
Faith 5.1803e-13 and SII 1.7053e-13. Another 42 randomized method/order checks for n=2…7 and orders
1, 2, 3 and n pass. The retained exact and empirical validators pass all 22 reference payloads /
3,050 game instances, including counts, stored errors, identities, numerical bounds and current
installed-source guards. The KKT equality formulation, per-coalition kernel, highest-order routing,
full-order Möbius shortcut and separate SII convention are correct.

Current reviewer tests: **99 passed, one optional full-fit skip** across the seven substantive
methods/study/reference/diagnostics/plots/contracts/report test files. This is not relabeled as the
historical 119-test collection. The native fitting scope is established by the separate actual
executions. The prospective payload checker passes all 1,772 entries / 1,735 required tensor paths
after physical rehashing. The play checker passes 12 native parameter variants, five analytic
variants and three rejected invalid inputs. These two checkers reuse the author-written validation
logic with explicitly recorded provenance, independently executed against v8. The full-support and
derivative probes are separately written reviewer checks. Scratch filename/schema/path mistakes in
reviewer tooling were fixed locally and their failure logs preserved; none was treated as a
candidate defect or erased as passing evidence.

## Actual learner and visual review

The own-answer notebook contains 141 cells, 55 sequentially executed code cells, no retained errors
and ten taught method definitions whose ASTs match the reviewed source. The exercise draft retains
the eight student exercises and eight solution dropdowns. The lesson begins with a concrete
four-way-credit question, asks for a prediction, teaches Möbius inversion, derives ST routing and
constrained Faith-Shap, and then makes students implement intact donor replacement and contextual
second differences. Immediate tests, expected outputs, graduated hints, interpretation prompts and
complete solutions are present. The exact full-order shortcut is explained rather than hidden. The
native capstone is clearly framed as known-model reproduction and guided interpretation; the
supplied execution plumbing does not hide the taught methods.

I individually viewed all eleven actual v8 PNGs. Each PNG hash differs from v7, so the old visual
approval was not reused. The new visuals accurately show the pure-four ledger and negative-singleton
reconstruction mechanism; all clean/shuffled seeds and additive ridge; fixed-row signed graphs with
a complete term-difference strip; all audit pairs with separate signed and absolute summaries;
frozen held-out pair rankings against 256 random orders and within-size structure controls; kernel
and coalition-size reconstruction errors; alternate-background effects; direct-forward/indexing
agreement; and both editable plays. Figures retain near overlap, cancellation, small residuals and
the ridge normalization caveat. SII remains separate from efficient truncated decompositions.
Background spread is not called a confidence interval, and interventions are not called physical
causal truth. Some complete 28/36-term panels need zoom for fine labels, but no data or result is
hidden or cropped.

I used a dedicated Chrome tab at `http://127.0.0.1:8873/draft_exercises.html`, without changing user
or parent review tabs. I clicked and verified all 54 dropdowns open, including eight solutions;
eleven images load; 65 math source elements are present; there are zero math-error elements, console
warnings/errors or page horizontal overflow. The first all-dropdown call timed out after 21 opens;
bounded batches opened the rest and the final live DOM verifies all 54. Actual browser screenshots
of the KKT matrix and expanded derivation, Faith-Shap closed form and solution/stub context, and
native play were captured, saved as genuine JPEG bytes and visually inspected. They are
`browser_kkt.jpg`, `browser_closed_form.jpg` and `browser_native_play.jpg`. The temporary reviewer
tab was closed.

Both plays teach recomputation. The analytic play varies dividend membership/value/order; the native
microscope changes model, row, bank, pair, context and donor. Its single-donor game is labeled
separately and post-study audit reuse is exploratory. No play result is falsely represented as a new
native fit or selection-free confirmatory experiment.

Original ARENA preservation was freshly audited: PASS, zero blockers. The limitations remain
explicit: finite empirical intact-donor replacement, known-model replication, exponential
enumeration, kernel-specific faithfulness, floating-point solver precision and no
population/physical causal claim.

## Publication interface handoff

`independent_recheck.json` supplies the exact `APPROVE_PUBLICATION`, independent role,
after-execution timing, original protocol-freeze hash and execution source-input map required by
`publish_study.validate_publication_review`. `independent_figure_review.json` binds the actual
eleven PNGs, lesson, completed learner, browser results and saved screenshots. Neither record
modifies the old freeze to claim that its original author resolution was independent approval.

The coordinator must copy these new records and actual review files into the intended
repository/learner evidence locations, assemble `learner/visual_review.json` with the explicit
evidence-file records required by `validate_visual_review`, and then call the actual publisher. Its
screenshot and inspection-contact entries must resolve within the learner/fresh evidence roots; our
reviewer-cache paths are not silently asserted to satisfy that relocation contract. The three real
screenshots and individual inspected PNGs are now available to assemble those records. Then run the
normal registered report and integration gates. No outstanding science/source fix or interface
defect requires another native fit; the remaining interface work is concrete evidence assembly and
publication execution.
