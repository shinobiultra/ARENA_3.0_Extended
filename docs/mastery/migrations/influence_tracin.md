# 16.10 full Influence Functions and TracIn lesson master migration

## Implementation plan

- **Current learner failure:** the complete historical lesson is published as 16.9 from a separate readable-lesson builder, while canonical 16.10 is missing.
  Its former number also collides with current Data Shapley; numerical identity must follow methods and paths.
- **Question:** which training images help a particular prediction, and how do infinitesimal upweighting, observed SGD-path accounting, checkpoint gradient
  scores and actual deletion refits differ?
- **Signature result:** preserve all eighteen figures, especially every-row/final-query influence-versus-LOO scatter, per-query ranking/sign failures,
  class-matched deletion loss/accuracy curves, actual-image galleries and exact-path/TracIn controls.
- **Student methods:** logistic losses/gradients, weighted objective/curvature, HVP, checked conjugate-gradient solve, converged weighted refits, signed
  influence, recorded SGD, exact/Taylor path effects, checkpoint TracIn, affine ghost product and class-matched selection.
- **Controls:** clean/corrupted labels, wrong regularizer normalization, unfinished optimization, damping, gradient/pixel similarity, actual LOO, exact path,
  checkpoint/order choices, class-matched random deletions, explicit flattening and broken ghost controls.
- **Play:** learning-rate/query Taylor-error play and actual query/class-matched deletion refit play.
- **Files:** new canonical `master_16_10.py`, `part10_influence_tracin` notebooks/module/tests/media and matching instruction pages; this note; scoped
  archive/parity/proposed-case evidence; focused migration tests. Shared config, registry, guided workbook and navigation belong to root.
- **Validation:** immutable legacy inventory/archive/source/receipt hashes; exact current-lesson reconstruction; actual completed-learner AST/output/count
  provenance; all generated scientific code nonblank/AST parity with bounded test-import adaptation; figure bytes; complete prose retention and own-answer
  completion; schema/import safety; tiny CPU semantic fixtures only; repeat canonical build and independent review.

Preserve every original 16.9 source, notebook, published page, asset, receipt and failed/development bundle physically and byte-for-byte. Original ARENA
chapters 0–4 remain unchanged. No native study/model/data download, scientific re-execution, GPU/cluster work, cleanup or Git mutation is authorized. Historical
carried outputs remain labeled historical; root owns any guided-case adoption.

## Complete canonical lesson and learner changes

The canonical source is `infrastructure/chapters/chapter16_shapley_attribution_baselines/master_16_10.py`.
It generates both `16.10_Influence_Functions_and_TracIn` notebooks, `solutions.py` and the two matching
`10_[16.10]_Influence_Functions_and_TracIn` instruction pages. The ordinary generated master `.ipynb`
is a derivative of this source. The new section directory is `part10_influence_tracin`; all historical
`part9_influence_and_tracin` material remains in place.

The generated exercises have **141 cells, forty code cells, approximately 10,883 words of prose,
eleven exercise groups teaching thirteen core functions, eighteen figures and both original play activities**.
Four canonical numbered parts group the original twelve subsections without dropping their headings or prose.
Each exercise retains immediate semantic checks, help and expected-output text, and a properly fenced solution
from the visible master. Historical coding answers preceded their stubs; canonical answers follow their code
cells as required by the converter. Completing each stub from its own following answer gives the identical
solution AST. No taught method is replaced by a hidden implementation import.

The student methods are `logistic_losses`, `logistic_gradients`, `weighted_objective`, `objective_derivatives`,
`logistic_hvp`, `conjugate_gradient`, `fit_weighted_logistic`, `upweight_influence`, `sgd_trace`,
`trace_loss_reductions`, `tracin_checkpoint_scores`, `ghost_linear_products` and `class_matched_removals`.
The full visible native preparation, checkpoint scoring, final evaluation, per-query correlation, image-gallery
and query-deletion play functions also survive. Exercise metadata is rendered in the converter's standard
quoted YAML format, preserving the original difficulty, importance and time text.

Seventeen additive forecast/reflection pairs cover query-relative harm, inverse curvature and checked solves,
normalized upweighting, exact/Taylor path accounting, checkpoint choice, large-step play, balanced native
budget-8 deletion, native training and path controls, LOO signs, per-query ranking, stationarity/damping,
finite perturbations, group deletion, actual images, ghost products and query-specific cleaning play. Each
asks for a numerical interval/contrast or explicit predicted sign/ranking, confidence, causal reasoning and
prior exposure, followed by observed controls, discrepancy explanation and a changed-condition forecast.
The native deletion forecast appears before native training/ranking preparation and before final evaluation.
No learner forecast, observation or acceptance is fabricated.

## Preserved scientific distinctions and negative findings

All eighteen original figures retain their exact bytes and scientific meanings. The full influence/LOO
scatter and per-query agreement heatmap keep strong ranking agreement together with **54/8192 clean and
34/8192 corrupted wrong-sign pairs**. Finite removals can depart from local derivative magnitudes, even
when rank correlations exceed 0.997. Unfinished optimization, inverse damping and wrong regularizer
normalization remain distinct controls.

The exact single-example path accounting still telescopes to the actual observed query-loss reduction;
first-order and regularizer-omission approximations visibly differ. All three order seeds and all/early/late/last
checkpoint choices remain visible. TracInCP is not relabeled as exact path accounting or an influence estimate
of the same counterfactual. Loss and accuracy remain together in the deletion curves: some checkpoint
policies worsen clean-label loss, and some corrupted-label deletions lower loss while introducing errors.
All 32 class-matched random rankings are retained as a spread, not a population confidence interval.
Actual-image galleries remain fixed by source-row/class rules rather than selected after inspecting scores.

The ghost result remains one affine invocation checked against explicit complete parameter gradients and a
rolled-residual control. It does not become a GhostSuite package benchmark or a guarantee for shared parameters,
repeated tokens, a whole transformer or nonlinear network attribution. The original limitations, failed/development
bundles and recorded implementation corrections remain available; the original protocol records its finite-precision
line-search correction without weakening the 1e-10 stationarity requirement.

The two play activities retain their complete computations: learning-rate/query changes exposing Taylor error,
and class-balanced deletion for a chosen ranking query with predicted versus actual refit effects and all
random controls. Changes made after seeing final outputs remain exploratory. Final queries stay historical
and monitored; migration does not turn them into untouched TEST.

## Archive, receipts and exact output provenance

Before generation, [`published_before_master.tar.gz`](../evidence/influence_tracin_master_migration/published_before_master.tar.gz)
archived **48 publication/source files**: all legacy top-level sources/notebooks/configuration, top-level expected-output
JSON receipts, the old instruction page and eighteen original figure assets. Its reopened member hashes and archive
SHA-256 are in [`before.json`](../evidence/influence_tracin_master_migration/before.json). The same record hashes
**91 preserved files**, including every original historical/development execution bundle. Those bundles remain
physically present at their original paths; they are inventoried rather than duplicated inside the new archive.
No old notebook, page, receipt, helper, failed bundle or asset was overwritten.

All thirteen `learner_execution.json` source inputs, fourteen `fresh_reproduction.json` source inputs, old published
notebook/page hashes and eighteen figure hashes match their current physical originals. Completing the retained
`lesson.md` using its own preceding answers reconstructs all eighty published solution cells exactly. The actual
`completed_learner.ipynb` in `expected_outputs/full_own_answer_v3/execution.tar.gz` matches all forty published code
ASTs exactly, with no execution-harness exclusions. Canonical outputs and execution counts also equal this actual
stored completed learner, after nbformat's ordinary multiline-output normalization.

All generated solution scientific code has identical nonblank lines and identical AST to that publication.
There is one explicitly bounded setup adaptation:

```python
# historical immediate semantic tests
from chapter16_shapley_attribution_baselines.exercises.part9_influence_and_tracin import (
    attribution_methods_tests as tests,
)
# canonical byte-identical copy
from chapter16_shapley_attribution_baselines.exercises.part10_influence_tracin import tests
```

The new `tests.py` is a byte-identical copy of `attribution_methods_tests.py`. Each exercise passes its own
visible function explicitly. Migration fixtures additionally point the test module's reference namespace at
the generated methods when testing coupled routines and regularized normalization. The copied tests' original
legacy fallback imports remain unchanged; they do not hide notebook implementation.

The other 39 code cells need no route substitution. [`output_parity.json`](../evidence/influence_tracin_master_migration/output_parity.json)
records all published/generated code hashes, AST hashes, route-normalized AST hashes, output hashes and historical
counts. The canonical solutions retain **forty historical execution counts and 31 output-bearing cells, with eighteen
embedded PNGs**. Canonical exercises contain all eighteen exact figure payloads, including two figures in each
perturbation/gallery output cell. Repeat canonical generation retained outputs using its strict identical-code check.

**These are retained historical reference results, not a fresh canonical execution.** That label lives in the
master, both notebooks, page and durable sidecar. Original receipt hashes remain unchanged and still certify the
archived publication. Historical fresh reproduction is separately labeled and was not rerun here. Schema checks,
source review, mathematical tests and preserved outputs do not establish new native execution or human acceptance.

## Explicit path and numerical-identity adaptations

Canonical publication number **16.10** identifies Influence Functions and TracIn. The preserved old **16.9** number
belongs to this historical publication. Current canonical **16.9** is Data Shapley in One Training Run. No checkpoint,
metric or experiment identity is transferred merely because numbers match. The introduction's nearby Data Shapley
link is explicitly corrected from old `16.7/part7_data_shapley_in_one_training_run/README.md` to current
`16.9/part9_data_shapley/16.9_Data_Shapley_in_One_Training_Run_exercises.ipynb`.
[`route_adaptations.json`](../evidence/influence_tracin_master_migration/route_adaptations.json) records this exact prose edit.

The original `attribution_data` and independent-reference imports, manifest-relative path, data-order namespace
`arena16.9/v1` and default output directory `/tmp/arena16_9_influence_tracin` remain explicit frozen dependencies.
Changing the source-row hashing namespace would change the cohort, so it is deliberately preserved. This is
scientific input identity, not the canonical lesson ID. Notebook root discovery reaches the same repository from
its new directory. Python-only `__file__` root discovery permits import from an unrelated working directory.

`MAIN` guards protect all data loading, directory creation, plotting, style/thread changes, training, refits,
reference evaluation and output writes on module import. The converter removes those guards from notebook code,
retaining historical computations. Pure protocol constants and provided function definitions remain available for
explicit calls. Import safety is tested with network, temporary-directory creation, plotting and runtime mutations
forbidden. No legacy publisher was invoked.

## Guided-case proposal and remaining integration

[`proposed_case.json`](../evidence/influence_tracin_master_migration/proposed_case.json) is a proposal only; root owns
adoption and shared registry/workbook/navigation/configuration. Its numeric target is **actual mean final unregularized
BCE after a corrupted-label, influence-selected, balanced budget-8 deletion refit minus full-data BCE**, in nats. Selection
uses the 48 ranking queries only; final evaluation averages the fixed 64 final queries. The 128 training rows, twelve
flips, L2=.02 including bias, independent zero-start converged refits and four-per-observed-class removal remain explicit.
The changed condition uses balanced budget sixteen with the same ranking/model/objective/metric. It is a
different group intervention, not a seed replication, but budget sixteen is already measured and displayed in the
historical five-budget curve: this is exposed guided practice, not an unseen transfer. Controls retain clean-label and all 32 matched random rankings,
convergence checks and loss/accuracy together. The frozen manifest/protocol hashes and distinct old/current lesson
identities are recorded separately in `proposed_case_identity.json`. No native experiment was run to adopt this proposal. The exact deletion tensors have not
been located; retained
figures and rounded prose support approximate historical comparison only. An exact numeric machine observation
requires a separately declared and authorized re-execution unless a recoverable exact tensor source is established.
Visual estimates must state their precision and must not be scored as exact targets.

All eighteen figures were inspected in the [contact sheet](../evidence/influence_tracin_master_migration/figure_contact_sheet.png).
The final LOO comparison, deletion curves and affine ghost-product panels were also inspected at full resolution.
This is retained-figure visual QA, not a new browser-rendering or learner-acceptance claim.

## Validation and limits

```bash
UV_CACHE_DIR=/tmp/arena-mastery-uv \
uv run --no-sync --offline python infrastructure/extension/build.py 16.10
UV_CACHE_DIR=/tmp/arena-mastery-uv MPLCONFIGDIR=/tmp/arena-influence-migration/mpl \
uv run --no-sync --offline pytest -q tests/mastery/test_influence_tracin_master_migration.py
```

Focused result before final documentation: **23 passed in 8.17 seconds**. Fourteen checks exercise generated visible
methods against tiny independent CPU/autograd/SciPy fixtures, including the regularizer-normalization counterexample;
none load digits or run the real study. Remaining checks cover every archive/original/source/receipt/asset byte,
actual stored execution ASTs/outputs/counts, exact current-lesson reconstruction, all thirteen own-answer methods,
complete original teaching-line preservation, notebook schema, forecasts/controls/play/negative findings and safe import.
All original ARENA chapters 0–4 remain unchanged by this lane. No heavy model work, native study, download, GPU/cluster
job, cleanup, staging, commit, revert or push occurred. Fresh canonical execution, shared integration and human learner
acceptance remain separate gates.

Proposed chapter description: **Build influence, exact SGD-path accounting, checkpoint TracIn and affine ghost products,
then test query-relative signs and balanced deletion interventions against actual refits and explicit controls.**

The recorded external `native_directory`, `learner_directory` and `fresh_directory` cache paths are currently
absent; [`external_cache_status.json`](../evidence/influence_tracin_master_migration/external_cache_status.json)
records the read-only check. The preserved in-repository execution archives and published outputs remain available.
A live external tensor cache is not assumed by this migration. The proposed guided case explicitly separates
approximate historical curve readings from exact observations requiring separate authorized re-execution. Final focused checks after
receipt-label/source-inventory tightening passed **23 tests in 7.82 seconds**; the master, generated Python module and
focused test source also compile.

Independent review passed all 23 focused checks in 8.35 seconds and confirmed full source/content/output retention. It identified the proposed changed condition
as already exposed
and found no study deletion tensors in any of the fourteen retained execution archives; both precision/exposure boundaries are now explicit. The import-safety
fixture was corrected
to use the actual `ARENA_ATTRIBUTION_OUTPUT_DIRECTORY` key. After that correction, all **23 checks passed in 5.69 seconds**. No scientific source or output
changed during these
review fixes.
