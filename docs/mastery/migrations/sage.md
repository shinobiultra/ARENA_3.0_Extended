# 16.4 full SAGE lesson master migration

## Implementation plan

- **Current learner failure:** the complete, executed 16.12 SAGE lesson bypasses canonical master generation and has a different publication route from syllabus section 16.4.
- **Target section claim:** feature contributions to prediction and predictive performance are different games; correlation, missingness assumptions, population weighting and sampling layers change their interpretation.
- **Signature result:** preserve all eighteen historical figures, including global SAGE versus shuffled-target networks, exact/local-loss agreement, correlation, sampling/bias and held-out recipe-prefix controls.
- **Student-implemented functions:** coalition masks, marginal prediction, loss game, exact Shapley, conditional/marginal Gaussian game, path contributions, sampled SAGE, path means/errors and prefix values. Retain the independent interpretation exercise.
- **Controls/baselines:** XOR singleton/leave-one-out, conditional/marginal and optimal/misleading Gaussian models, clean/shuffled networks, ridge/training mean, exact enumeration, dummy input, alternative backgrounds/targets, mean imputation, inner-sampling bias, random/reversed/LOO prefix rankings.
- **Play cells:** Gaussian correlation/model-weight play and actual recipe/feature/background play.
- **Files to modify:** new `master_16_4.py`; new `part4_sage/` notebooks, Python module, semantic tests and media; matching canonical instruction pages; this note; migration evidence; focused migration tests. Shared config, syllabus, checkpoint and manifests belong to the lead.
- **Tests to run:** immutable legacy archive and all source hashes, receipt identity, own-answer reconstruction, per-cell AST/nonblank/output parity, figure bytes, lesson-line retention, own-answer completion, notebook schema, import safety, tiny CPU semantic oracles, repeat canonical build.

All 16.12 sources, notebooks, instruction assets, receipts and historical failed/development evidence remain physically present and byte-for-byte unchanged. Original ARENA chapters
0–4 remain unchanged. No model load/download, GPU, cluster, scientific execution, cleanup, commit, staging or push is authorized in this lane. Historical output carry is
conditional on exact code parity, apart from an explicitly enumerated import/root route adaptation; it is never a fresh execution claim.

## Canonical artifacts and complete lesson

The canonical source is `infrastructure/chapters/chapter16_shapley_attribution_baselines/master_16_4.py`.
It generates `chapter16_shapley_attribution_baselines/exercises/part4_sage/16.4_SAGE_Global_Importance_{exercises,solutions}.ipynb`,
`solutions.py`, and the canonical markdown/Python instruction pages named `04_[16.4]_SAGE_Global_Importance`.
The converter also creates its ordinary `master_16_4.ipynb` derivative. Shared routing/config was already mapped
to this section; no shared files were edited by this lane.

The generated lesson has **140 cells, 33 code cells, approximately 10,071 words of prose, nine coding exercises,
one additional interpretation exercise, eighteen figures and two play activities**. All original scientific
calculations, training/forward functions, mathematical explanations, help/expected-output dropdowns, independent
checks, negative findings, limitations and readings survive. Tests check every non-coding-answer teaching line,
with only the title, setup/objective heading and numbered-part formatting adapted to the converter. The nine
coding exercises implement `coalition_masks`, `marginal_predict`, `loss_game`, `exact_shapley`, `gaussian_game`,
`path_contributions`, `permutation_sage`, `summarize_paths` and `prefix_values`. Their visible solutions complete
their own stubs to the exact generated solution AST, with immediate semantic checks still called.

Eighteen additive forecast/reflection pairs ask about the nonlinear averaging example, XOR allocation,
conditional correlation, misleading-model Gaussian play, the native global result, training controls, audit
model comparisons, local prediction/loss signs, local/global linearity, additive summaries, sampling/error
coverage, background/target changes, mean imputation, inner bias, hybrid distances, population weighting,
audit prefixes and actual-recipe play. Each forecast asks for a number/range or predicted contrast, confidence,
causal reasoning and prior-exposure status. Each reflection requires the observed controls, discrepancy
explanation and a forecast for a changed scientific condition. Forecasts for the native study and diagnostics
appear before their computation cells. All learner responses remain blank. This reference-rich route is guided
practice; it makes no unseen-mastery claim.

The signature global-SAGE figure retains clean seeds 0/1/2, ridge and shuffled-target networks, including
negative importances. The exact local-loss/global identity remains separate from prediction attribution and
absolute-value summaries. Audit prefix plots retain all fixed models, top/bottom/leave-one-out rankings and
100 random-order controls. The random 10th–90th percentile band is not relabeled as generalization uncertainty.
Neither monotone improvement nor SAGE dominance at every prefix is claimed. The conditional Gaussian example
keeps a known distribution; marginal hybrids on concrete recipes are not presented as conditional or physical
causal effects. The two original play activities change Gaussian correlation/model error and actual recipe,
feature and background size.

## Archive, receipts and exact output carry

Before any generation, [`published_before_master.tar.gz`](../evidence/sage_master_migration/published_before_master.tar.gz)
archived **93 files**: every pre-existing non-bytecode legacy SAGE file (including historical/failed/development
execution bundles and receipts), the old published instruction page, and all eighteen original instruction
assets. [`before.json`](../evidence/sage_master_migration/before.json) records the archive SHA-256 and every
member's SHA-256. The archive was reopened and every member checked. All members also remain at their original
paths with their exact original bytes. No historical publication was overwritten.

All fifteen `source_inputs`, both published-notebook hashes, the old published-page hash and all eighteen
figure hashes match the unchanged `expected_outputs/learner_execution.json`. The retained readable `lesson.md`
reconstructs every one of the eighty published solution cells exactly when completed from its own answers.
The actual `completed_learner.ipynb` in `expected_outputs/full_own_answer_v4/execution.tar.gz` matches the AST of
all thirty-three published code cells exactly: **no execution-harness differences or scientific-code exclusions**
are needed for this comparison.

All canonical solution code has identical nonblank lines and identical AST to the old publication, after
normalizing **one named route adaptation in setup only**:

```python
# historical
from chapter16_shapley_attribution_baselines.exercises.part12_sage_global_importance import (
    sage_methods_tests as tests,
)
# canonical
from chapter16_shapley_attribution_baselines.exercises.part4_sage import tests
```

The canonical `tests.py` is a byte-identical copy of historical `sage_methods_tests.py`; every exercise passes
the generated visible function explicitly to its independent oracle. Its legacy default-function import is
retained unchanged, but those defaults are not used by the lesson or migration tests. Data, source checks,
author-imputer parity, protocol/freeze and background-order imports retain their original fully qualified
legacy identities. This deliberately preserves the manifest-relative `__file__` behavior and stable
`arena16.12/background/...` donor ordering. Renaming those helpers would change a scientific input; this
migration does not do that. The canonical notebook can run from its new directory because root discovery
walks upward to the same repository. The generated Python module uses a Python-only `__file__` root instead.

[`output_parity.json`](../evidence/sage_master_migration/output_parity.json) stores each of the thirty-three
published/generated source hashes, AST hashes, route-normalized AST hashes, historical output hashes and
execution counts. Thirty-two cells need no route normalization. The canonical solutions retain **33 historical
execution counts and 31 output-bearing cells**, including all eighteen embedded PNG outputs. Canonical media
are byte-identical copies of the unchanged original assets. A second canonical build retained outputs through
the converter's strict identical-code check. No warning or evidence gate was weakened to carry them.

**These are retained historical reference results, not a fresh canonical execution.** That label is durable
in the master, generated notebooks/page and sidecar. The old receipt retains all original hashes and still
certifies only its original 16.12 publication. Custom notebook metadata can disappear on regeneration; the
master prose and sidecar remain authoritative. A prior historical fresh-native receipt is distinct from this
migration, and no fresh execution was performed here.

All eighteen PNGs were inspected together in
[`figure_contact_sheet.png`](../evidence/sage_master_migration/figure_contact_sheet.png).
Global importance, mean imputation and audit prefixes were also inspected individually at full resolution.
This is visual QA of retained figures, not a fresh browser rendering or human learner acceptance.

## Import safety and measurement identity

The generated Python module exposes every pure taught method and the provided visible training/study functions.
`MAIN` guards protect data loading, training, diagnostics, plots, filesystem writes, thread/grad/style mutations
and reference downloads on import. Those guards are removed from notebook source by the canonical converter,
so they do not alter the historic notebook calculation. Import from an unrelated directory is tested while
network access, temporary-directory creation, plotting and global thread/grad/style changes are forbidden.

The frozen native study remains CPU float64/float32 as authored, with 758 training rows, 153 explanation rows,
119 recipe-disjoint audit rows, eight inputs, 256 coalitions, 64 primary background rows, three clean network
seeds, three shuffled-target networks and one ridge model. All six networks use 800 Adam updates, two 32-unit
ReLU hidden layers, training-only scaling and original frozen seeds/split identities. All fits are retained;
rankings are frozen from explanation rows before audit loading. Sampling keeps 4,096 direct paths and 32
repeated schedules. The 40,320-permutation, constrained-weighted-regression, NumPy-forward and pinned author
imputer checks remain part of the complete lesson. The migration does not rerun them or promote the monitored
audit cohort to untouched TEST.

The proposed shared guided case is an exact Gaussian controlled contrast using the generated visible
`gaussian_game` and `exact_shapley`: covariance `[[1,.8],[.8,1]]`, truth/model `[1,0]`, noise variance `.25`,
float64, conditional and marginal restriction. Forecast the unused second input's conditional-minus-marginal
SAGE in squared-target units. The transfer changes only model weights to `[1,2]` and asks for the same contrast,
including sign/ranking reasoning. This changes model behavior, not merely a random seed. No numerical answer
is embedded here. The lead owns adoption and checkpoint/config fingerprints, source/reference paths,
workbook generation, manifest/navigation integration and any preserved-directory exception.

## Validation and remaining gates

```bash
UV_CACHE_DIR=/tmp/arena-mastery-uv \
uv run --no-sync --offline python infrastructure/extension/build.py 16.4

UV_CACHE_DIR=/tmp/arena-mastery-uv MPLCONFIGDIR=/tmp/arena-sage-migration/mpl \
uv run --no-sync --offline pytest -q tests/mastery/test_sage_master_migration.py
```

Initial focused result: **18 passed in 4.83 seconds**. Nine parametrized checks run the generated visible
methods against the independent tiny CPU semantic oracles. The remaining checks cover all archived/original
bytes, receipt/source identity, real completed-learner ASTs, all outputs and assets, own-answer completion,
notebook schema, retained teaching lines, controls/plays/forecasts, the explicit test-import adaptation and
unrelated-directory import safety. The second canonical build reported unchanged code and retained outputs.

Engineering migration fidelity and tiny CPU tests are separate from fresh scientific execution, a full
canonical notebook execution, browser rendering, human learner acceptance, readiness promotion and whole-course
completion. No model was loaded/downloaded, no training/study/GPU/cluster operation ran, and no performance
or frozen scientific default was silently changed. All original ARENA chapters 0–4 remain unchanged by this
lane. No files were staged, committed, deleted, reverted or pushed.

Proposed chapter description: **Build exact and sampled SAGE, then test how correlation, missingness,
population weighting and audit controls change what a feature-importance claim means.**

Final repeat-build verification: **18 passed in 4.13 seconds**; the three owned Python sources compile. An independent reviewer also passed all eighteen tests in 5.14 seconds, reopened the actual executed archive, and checked all thirty-three historical outputs/counts plus all eighteen exercise PNG hashes. No code, retained-content, or import-safety blocker was found.
