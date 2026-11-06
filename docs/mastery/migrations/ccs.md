# 11.5 full CCS lesson canonical migration

## Preservation plan

Current learner failure: the full published lesson bypasses canonical ARENA authoring and has no durable forecast/reflection route.

Target section claim: a low contrast-consistency loss can identify competing features; semantic orientation, customer accuracy, distractor sensitivity and actual model outputs require separate measurements.

Signature result to preserve: all fifteen published figures, including the exact competing-feature counterexample, every-restart native results and actual-review gallery.

Student-implemented functions: branch centering, original asymmetric CCS loss, paired inference, independent restarts and label-free selection, calibration orientation, binary metrics, paired-difference PCA, padding-aware token gathering, paired opinion change and full-vocabulary output metrics.

Controls/baselines: raw uncentered and shuffled-pair CCS, paired PCA, supervised probe, 32 norm-matched random directions, calibration-only orientation, every restart, native vocabulary winners, exact independent distractor and opinion interventions.

Play cells: distractor-strength sweep and fixed-reporter review editing, including changed-customer and changed-other-reader comparisons.

Files to modify: new master_11_5.py; canonical generated notebooks/module/page/media; this migration note; exact historical archive/parity receipts; focused migration tests. The legacy underscore-stem notebooks and page, sources, reports, failed/development evidence and originals remain physically present. No generic AUTHORING cleanup is authorized.

Tests to run: exact archive/source/figure hashes; complete own-answer reconstruction; all 32 code-cell AST and nonblank-line parity; carried historical outputs/counts; student own-dropdown completion; preservation of original teaching lines; import safety from an unrelated directory; independent tiny CPU numerical fixtures with ambient dtype restoration.

No model loads/downloads, GPU/cluster work, model training or fresh science are authorized in this lane. Historical outputs are retained references, not current execution or learner acceptance. Original ARENA chapters 0–4 remain unchanged.

## Canonical artifacts and fidelity

The canonical source is `infrastructure/chapters/chapter11_representation_geometry/master_11_5.py`. Current configuration names the generated artifacts
`11.5_Contrast-Consistent_Search_{exercises,solutions}.ipynb` and instruction page `05_[11.5]_Contrast-Consistent_Search.{md,py}`. The old underscore-stem notebooks and
`05_[11.5]_Contrast_Consistent_Search.md` remain unchanged and continue matching their publication receipt. `solutions.py` is now generated from the visible canonical master; its
previous bytes are archived.

The generated lesson has **121 cells, 32 code cells, 8,156 prose words and nine exercises** before the lead's guided-workbook integration. It preserves all original teaching prose,
twelve student-implemented functions, immediate tests, help, expected outputs, solutions, controls, both play cells, paper links, exact counterexamples and real-model negative
findings. Canonical headers, numbered summaries and ten specific forecast/reflection pairs are additive. Forecasts precede result execution; reflections ask for observed controls,
discrepancies and a changed-condition forecast. No answer, confidence or learner observation is fabricated.

The visible implementation retains branch centering; original asymmetric confidence-plus-consistency objective; averaged paired inference; batched independent AdamW restarts;
minimum-training-loss selection; calibration-only sign orientation; strict-threshold binary metrics; paired-difference PCA; padding-aware final-token gathering; matched-base
opinion effects; and complete-vocabulary model output metrics. No core method is replaced by a hidden utility import.

`published_before_master.tar.gz` archives exact old notebooks, reference module, lesson, builder, publication receipt and instruction page. `before.json` hashes the archive, its
members, all old section artifacts, thirteen publication source inputs and fifteen figures. The old source inputs and all fourteen source inputs in the separate fresh-reproduction
receipt remain current. Existing expected-output/development archives and failures are retained physically.

Completing the old lesson from its own answer dropdowns reconstructs the complete 65-cell publication exactly. All 32 published solution code cells match the archived completed
learner in `expected_outputs/full_own_answer_v3/execution.tar.gz` by AST, without exclusions. All 32 canonical code cells also preserve both AST and nonblank lines. Only whitespace
normalization differs. `output_parity.json` records old/new source hashes, AST hashes, original output hashes and execution counts. This establishes the basis for carrying forward
all **32 historical execution counts and 30 output-bearing cells**, without running models. A subsequent canonical rebuild retains these through the pipeline's unchanged-code
check.

The fifteen `media/` PNGs preserve exact original instruction-asset bytes; exercises embed the same image bytes. All fifteen were reviewed in a contact sheet. The native review
gallery, whole-vocabulary baseline, controlled opinion effects, exact counterexample and distractor-strength sweep retain their original axes, controls and negative findings. No
figure was regenerated or reinterpreted as new empirical evidence.

Generated Python imports safely from an unrelated directory. `MAIN` guards protect tests, plotting, global style/thread changes, filesystem writes, model imports, checkpoint
loading, training, capture and study execution. Python-only root discovery uses `__file__`; the notebook keeps its original setup. Visible numerical functions remain importable.
The historical callable `run_smoke_test`, `run_gpu_test` and `run_full_experiment` interfaces remain available; none runs during import. Their legacy real-model route still depends
on external historical learner directories and is not invoked in this migration.

## Frozen science and proposed prediction case

Preserve GPT-2 revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`, CPU float32/eager model execution, float64 fitting, training/calibration/development/final cohorts of
**256/64/64/64 distinct base reviews**, three variants, two wrappers and hidden-state indices **4/8/12**. Every independent-opinion review appears with both Morgan opinions. Final
prompt variants comprise 512 contrast pairs but only 64 distinct base-review identities. Every case retains 64 reporters: ten centered CCS, ten raw CCS, ten shuffled-pair CCS,
paired PCA, supervised probe and 32 norm-matched random directions. CCS fitting uses ten seeds, 1,000 AdamW updates, learning rate .01 and weight decay .01. Selection uses
unlabelled final training loss; semantic sign uses the separate labelled calibration split. Native model-output metrics read the prefix before the candidate answer; reporter
activations read after the proposed answer token.

Proposed guided question: for `distracting_reader/customer/layer8`, forecast loss-selected centered CCS minus supervised-probe customer accuracy in percentage points, with the
calibration sign fixed, on all 128 final contrast rows (64 base reviews × two Morgan opinions). Record paired-opinion prediction-flip fraction over the 64 matched base reviews and
retain PCA, raw/shuffled CCS and all random-direction comparisons. The historical references give 64/128 versus 119/128 correct, hence −42.96875 percentage points; their
paired-opinion flip fractions are 64/64 and 1/64. These reference values belong in interpretation, never prefilled forecasts. Loss-selected centered seed 8 has training loss
approximately 5.37e−8 yet chance customer accuracy: low loss alone fails to identify the intended target.

Proposed changed-condition transfer: apply that same customer's fitted reporter, training means and calibrated sign to the `quoted_review` wrapper for the same final base
reviews/opinion pairs. Predict the customer-accuracy change and paired-opinion flip fraction before execution. Keep source wrapper, layer, training data, selection and orientation
fixed. This is a changed scientific condition; a seed change alone is replication. The preserved lesson proposes cross-wrapper frozen transfer as a further investigation. Its
displayed native comparisons fit each wrapper separately, so they **do not measure this frozen cross-wrapper transfer**. New transfer execution remains beyond this migration's
no-model boundary. The lead owns the checkpoint manifest and guided-workbook integration.

## Validation and remaining gates

Generation without execution:

```bash
UV_CACHE_DIR=/tmp/arena-mastery-uv uv run --no-sync --offline python infrastructure/extension/build.py 11.5
```

Focused checks:

```bash
UV_CACHE_DIR=/tmp/arena-mastery-uv MPLCONFIGDIR=/tmp/arena-mastery-ccs/mpl \
uv run --no-sync --offline pytest -q tests/mastery/test_ccs_master_migration.py
```

Result: **20 passed in 4.39 seconds**. Twelve independent CPU-oracle checks exercise the generated visible functions, including NumPy-gradient, hand-worked loss/probability cases,
exact competing features, independent optimizer restarts, rank/SVD and complete-vocabulary fixtures. Eight other checks cover exact historical preservation, both receipts, all
code/output/count parity, stored execution parity, figure bytes, own-dropdown completion, prose/prediction placement and unrelated-directory import safety. Repeating with ambient
PyTorch dtype pre-set to float64 yielded **20 passed in 3.36 seconds**; an explicit assertion verified restoration of float64 after the run.

This is an engineering migration of retained historical evidence, not a fresh canonical full-notebook run, new model replication, new scientific acceptance, human learner
acceptance, readiness promotion or whole-extension completion. External learner/native cache directories named in old receipts are absent; no legacy publisher was run. Frozen
numerical choices were not silently upgraded. No original ARENA material was edited. No model/checkpoint was loaded or downloaded, and no GPU or cluster job was launched.
Independent review and lead integration remain separate gates.

Proposed chapter description: **Implement contrast-consistent reporters, then distinguish low loss, semantic orientation and target identification using exact competing features and frozen-transformer opinion controls.**
