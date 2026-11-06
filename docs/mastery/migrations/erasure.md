# 11.4 full concept-erasure master migration

## Plan before edits

- **Current learner failure:** the complete published lesson uses a separate publisher instead of the canonical ARENA master pipeline. Replacing it with a short demonstration would lose its scientific controls and visible student methods.
- **Target section claim:** a disabled original reader, zero fitting cross-covariance, failed refitted readers and changed model answers are distinct claims.
- **Signature result:** preserve the actual GPT-2 token gallery, probe-recovery and task/collateral curves, all twelve figures and negative findings.
- **Student methods:** paired moments, support-aware whitening, orthogonal covariance erasure, LEACE, affine application, probe refitting/evaluation, per-example magnitude-matched random edit and full-vocabulary outcome metrics.
- **Controls/baselines:** no intervention, removed probe coefficient, orthogonal covariance, empirical/shrinkage LEACE, sixteen random directions per site, magnitude-matched controls, constant reader and nonlinear recovery.
- **Play cells:** side-variable scale/coupling and continuous native intervention strength.
- **Files:** master 11.4; generated notebooks/module/page; copied media; this note; archived pre-migration sources and parity records; focused mastery migration tests. No shared routing edits and no deletion of legacy files.
- **Tests:** archive hashes, historical-source/AST/nonblank/output/figure parity, own-dropdown completion, import safety and tiny independent CPU method fixtures. Build without `--execute`; no model execution, downloads, GPU, cluster or new study.

Historical receipts are immutable. The learner receipt records an older lesson hash; the fresh-reproduction receipt records thirteen current source hashes. Any retained output is historical reference evidence, never a new execution claim. External caches named in those receipts are absent.

## Generated lesson and preservation

The canonical source is `infrastructure/chapters/chapter11_representation_geometry/master_11_4.py`. The generated exercises and solutions each have **87 cells, 24 code cells and
eight exercise groups covering nine student functions**, with approximately 7,529 words of reference-rich prose. The entire earlier 52-cell lesson survives; added cells provide
canonical chapter structure, seven forecast/reflection pairs and fenced solution dropdowns. Every original explanatory line outside the relocated solution blocks is tested for
preservation.

The actual-token heatmap remains the signature result. The lesson retains all twelve figures: the four-point geometry, erasure constraint/distortion, refitted probes, nonlinear
counterexample, covariance play, native development comparison, final reader recovery, covariance transfer, task/collateral damage, displacement/native KL, actual-token gallery and
native-strength play. The four-point and native results retain their original no-intervention, probe-direction, orthogonal covariance and random/magnitude-matched controls. Both
original play cells remain executable and editable.

No original ARENA chapter 0–4 file was changed by this lane. No legacy source, receipt, report, figure, archive or failed-result evidence was deleted, staged or committed. Old
instruction assets remain byte-identical; twelve exact copies now live under the section's `media/` directory. All twelve were inspected together in a contact sheet; the
actual-token gallery and task/collateral panels were also inspected individually.

## Exact historical provenance

Before generation, **72 files** were archived in
[`../evidence/erasure_master_migration/published_before_master.tar.gz`](../evidence/erasure_master_migration/published_before_master.tar.gz), SHA-256
`1989cd2b663e84056fbdcba4e3959191fc76dae2f6ff108ef514dd54b6e416d8`. This includes both published notebooks, generated Python module, current lesson, every retained section
source/receipt/archive, the registered instruction page and all twelve instruction assets. Every member was reopened and hash-checked.
[`before.json`](../evidence/erasure_master_migration/before.json) records the exact inventory and hashes; all legacy inputs remain physically present and unchanged.

The published notebook and instruction-page hashes match the immutable learner receipt. However, its `source_inputs` records lesson SHA-256
`88f1a2eb6b12e84c8f59b355fab61547e3cc2705417b62fa334aa7f336123daa`, while the current lesson is `6ad24e3664774da4c2e32002ad370b35a20fd59cb25417c39d17e83b2923bf49`. The unchanged
fresh-reproduction receipt records the latter hash and **all thirteen of its named current source hashes match**. The other eleven inputs named by the learner receipt still match.
This bounded lesson-source drift is explicitly documented, not silently repaired. Independently completing the current lesson from its own answers gives every one of the archived
52 published cells exactly, including code and prose. All 24 code ASTs also match the actual `completed_learner.ipynb` stored in
`expected_outputs/full_own_answer_v1/execution.tar.gz`, with no harness code exclusion needed.

[`source_provenance.json`](../evidence/erasure_master_migration/source_provenance.json) records these comparisons and the absence of
`/home/bitwise/.cache/arena_extended/erasure/full-own-answer-v1` and `/home/bitwise/.cache/arena_extended/erasure/fresh-native-v1`. The external author-code cache is also required
by the original optional author parity cell and is not repopulated here. Retained archives provide historical provenance, not the full external model/tensor cache necessary for a
new reproduction.

Every canonical solution cell preserves the archived cell's AST and every nonblank line; only blank-line formatting/final newline normalization differs.
[`output_parity.json`](../evidence/erasure_master_migration/output_parity.json) records the old/new code hashes, AST hashes, output hashes and execution counts for all 24 cells.
This justifies the initial historical carry; subsequent normal builds use the converter's stricter exact-source carry. The generated solution retains all 24 historical execution
counts and **23 output-bearing code cells**, including all twelve original PNG outputs. No historical receipt is rewritten to claim this migration ran a study. The master,
notebooks, page and parity sidecar explicitly label the results as retained historical reference evidence.

For safe importing, Python-only setup uses `__file__` rather than searching the caller's working directory. Runtime plotting, filesystem setup, author parity and native model
execution stay in `MAIN`-guarded generated Python blocks. Immediate student tests remain present in notebooks. The nine taught method functions remain directly implemented in the
master and generated module. Compatibility runner entry points are preserved in Python-only cells; the smoke runner explicitly passes these visible generated functions into the
independent oracles. Importing the generated module performs no model download, device initialization, plotting, thread-setting or output-directory creation.

## Prediction learning and evidence boundaries

Seven specific forecast/reflection pairs cover the affine optimum, probe refitting, nonlinear loss boundary, covariance play, native generalization, final behavior and continuous
native strength. They ask for numerical ranges or rankings, confidence, causal reasoning, observed controls, discrepancy explanations and a forecast under a changed scientific
condition. No learner answer, confidence or observation is fabricated.

Proposed guided workbook case for the integration owner: at frozen site `9/end`, predict recipient correct count out of 288 and unrelated-object correct count out of 384 under no
intervention, empirical LEACE and .01-shrinkage LEACE, comparing empirical effects with all sixteen matched empirical controls and separately examining independently
refitted-reader loss. Transfer changes the covariance objective to the fixed .01 spherical mixture while retaining the frozen cohort and separately fitted reader. This is a changed
scientific condition, not a new random seed. It remains guided practice on a historical, monitored final cohort; it earns no unseen-test claim. Shared
checkpoint/workbook/configuration edits belong to the lead.

## Verification and limits

Executed with `UV_CACHE_DIR=/tmp/arena-mastery-uv uv run --no-sync --offline`:

- `python infrastructure/extension/build.py 11.4` without `--execute`, including a second normal build that retained the previous outputs under exact code-source matching.
- `python -m pytest tests/mastery/test_erasure_master_migration.py -q`: **19 passed**. This verifies archived hashes, unchanged legacy files and both receipt scopes; source/AST/output/count/figure parity; exact old lesson reconstruction; all eight own-dropdown exercise completions; notebook schema; import safety from an unrelated directory; ten small CPU independent method oracles and one predeclared-split check. Tests restore the ambient default tensor dtype.

These checks establish migration correctness and tiny-fixture method behavior. They do not establish a new GPT-2 execution, new author-package parity run, fresh native
reproduction, browser/Streamlit rendering acceptance or human learner acceptance. No model, download, GPU, cluster job or new study ran. Existing scientific limitations remain
explicit: empirical support versus transfer, convex-loss versus threshold accuracy, preserved nonlinear information, poor late-site reader calibration, object damage under matched
random controls, a native object baseline of only 50%, finite templates and two single-token sites.

## Retained legacy files requiring a structural exception

The task explicitly overrides AUTHORING's generic cleanup guidance. These direct children remain unchanged alongside canonical generated artifacts:

`README.md`, `artifacts.lock.yml`, `build_notebooks.py`, `erasure_data.py`, `erasure_methods.py`, `erasure_methods_tests.py`, `erasure_native.py`, `erasure_protocol.py`,
`erasure_reproduction.py`, `erasure_reproduction_tests.py`, `execute_learner.py`, `lesson.md`, `native_protocol.md`, `publish_study.py`, `reference_parity.py`,
`run_development.py`, `verification_report.json`, `verification_report.schema.json`, and the entire `expected_outputs/` tree. `tests.py` is the pre-existing immediate-test
compatibility module and remains unchanged. The exact recursive retained inventory and hashes are in `before.json` under `source_inputs`; old instruction `assets/erasure_*.png`
also remain. No deletion is authorized by this lane.
