# 11.2 full alignment lesson master migration

## Scope and implementation plan

- **Current learner failure:** the rich, published alignment lesson bypasses the canonical ARENA master pipeline. Its executed study and negative examples must survive migration.
- **Target section claim:** held-out coordinate correlation, reproduction of a local native-donor distribution and correct name prediction are separate questions.
- **Signature result:** retain all nine figures, especially the GPT-2 transport-rank KL curves with native-donor, unchanged/no-op, shuffled-correspondence, Procrustes, ridge and matched-norm controls, plus the fixed actual-token failure gallery.
- **Student methods:** `svd_whitener`, `fit_cca`, `paired_column_correlations`, `heldout_correlations`, `fit_procrustes`, `cca_difference_transport`, `fit_ridge_transport`, `distribution_kl`; preserve the sixth interpretation exercise and its prose answer.
- **Controls/baselines:** population-known latent correlations; shuffled correspondence and independent views; five-seed high-dimensional noise; discarded low-variance causal feature; zero-rank transport; unchanged/no-op and actual native donor; ridge and PCA-space Procrustes; sixteen same-norm random draws; seen/unseen-name subsets.
- **Play cells:** independent-noise sample-count/width manipulation; low-variance causal scale/cutoff; real transport rank/layer-pair/final-row distribution inspection.
- **Files to modify:** new `master_11_2.py`, generated exercises/solutions/module/page/media, this note, migration archive/parity records and `tests/mastery/test_alignment_master_migration.py`. The lead owns shared routing, config, manifest and workbook changes.
- **Tests to run:** archive/source/notebook/figure hashes; all twenty generated-code cells against the archived publication and stored execution; original explanatory-line preservation; own-answer completion; notebook schema; import safety from an unrelated directory; nine independent tiny CPU oracles against generated visible methods, with save/restore of PyTorch's ambient default dtype.

All historical originals, prototypes, helpers, instruction assets, receipts and failed/development evidence remain physically present. No AUTHORING cleanup deletions are
authorized. Original ARENA chapters 0–4 remain untouched. This migration performs no model loading, GPU operations, checkpoint downloads, training, cluster jobs or new study
execution. Historical carry-forward is explicitly labeled in the master and durable sidecar; new learner forecasts and reflections remain blank.

## Canonical artifacts and learner changes

The canonical source is `infrastructure/chapters/chapter11_representation_geometry/master_11_2.py`. Its generated notebooks have 79 cells, twenty code cells, six exercises and
about 6,312 words of prose including the guided-workbook link. Five exercises implement all eight methods listed above; the sixth asks the learner to separate representation
similarity, donor reproduction and name correctness. Its original prose answer remains intact. No taught method is replaced by an import from a hidden implementation.

The original introduction, mathematical explanations, help/expected-output dropdowns, all study functions, actual-token examples, limitations, readings and the negative Pythia
development finding survive. Tests check every original non-coding-answer teaching line. The canonical format adds a header image, standard learning-objective blocks, numbered-part
summaries and a bonus boundary. Student stubs retain immediate semantic tests and properly fenced answers generated from the visible master solution.

Eight additive forecast/reflection pairs cover known latent correlations, independent-noise overfit, the small-variance causal feature, full-vocabulary donor KL, fixed signed
coordinate spectra, correct changed-name predictions, seen/unseen-name cohorts and single-row rank play. The donor-KL forecast occurs **before** the study execution cell; its
reflection follows the intervention plot. Prompts ask for a numeric range/ranking, confidence, causal reasoning and exposure status, then observed controls, discrepancies and a
forecast for changed scientific conditions. No learner values or answers are fabricated. The original answer-rich outputs make this guided practice, not an unseen challenge.

The three play activities remain: changing noise sample count/feature widths across five seeds, varying causal-feature scale/SVCCA variance cutoff, and inspecting a real row/layer
pair/transport rank with full-vocabulary probabilities. Rank zero and actual donor remain explicit controls. The nine figures retain their original data and assets. They were
inspected as a contact sheet; noise overfit, real KL curves and the actual-token failure gallery were also viewed individually.

The generated Python module imports safely from an unrelated working directory. `MAIN` guards protect plotting, test invocation, model loading, global thread/style changes,
directory creation, protocol reading and study execution. Pure visible numerical, capture, replacement and study functions remain importable. Notebook source retains the original
calculations because the converter removes the Python guards. Python-only root discovery uses `__file__`; safe data imports remain available to study functions. Historical
`run_smoke_test`, `run_gpu_test` and `run_full_experiment` names remain as Python-only compatibility entry points, never executed on import. Despite its legacy name, `run_gpu_test`
explicitly reproduces this section on CPU only when called. The smoke wrapper invokes the generated visible methods.

## Archive and historical carry-forward provenance

Before canonical generation, the published exercises/solutions, reference module, readable lesson, old builder, publication receipt and markdown instruction page were compressed
into [`published_before_master.tar.gz`](../evidence/alignment_master_migration/published_before_master.tar.gz). [`before.json`](../evidence/alignment_master_migration/before.json)
records the archive and per-file SHA-256 values, source-input inventory, figure hashes and original implementation-cell indices. Every archived member was reopened and checked byte
for byte. All eleven `source_inputs`, both notebook hashes, page hash and nine publication-asset hashes match the unchanged `expected_outputs/learner_execution.json` receipt.

Completing the retained `lesson.md` from its own answers reconstructs all 43 published solution cells exactly, including prose and code. All twenty published code cells also match
the AST of the completed learner in `expected_outputs/own_full_release_v3/execution.tar.gz`. The only excluded difference is the stored execution harness's two setup statements
redirecting `assets_dir` to `/home/bitwise/.cache/arena_extended/alignment/full-own-answer-v3/figures` and creating that directory. No model operation, scientific input, metric or
parameter is excluded.

All twenty generated solution code cells have **identical nonblank lines and identical AST** to the archived published code. Only empty-line and terminal-newline normalization
differ. [`output_parity.json`](../evidence/alignment_master_migration/output_parity.json) records every old/generated code hash, AST hash, historical output hash and execution
count. On this evidence, the original outputs were carried forward: twenty historical execution counts and sixteen output-bearing cells. A subsequent canonical rebuild retained
them through its strict same-code check. All nine new `media/` files are byte-identical copies of the unchanged original instruction assets.

**These are retained historical reference results, not a fresh current GPT-2 execution.** This label is durable in the master, generated notebooks/page and parity sidecar. The old
publication receipt continues to certify its archived publication, with its old notebook/page hashes unchanged. New prompts have no fresh execution or learner-response claim. The
original prototypes, expected-output directory, archives, failed/development evidence and original assets remain physically present.

## Keep the measurement scopes distinct

The pinned main study uses GPT-2 revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`, CPU float32, eager attention and batches of 24. It fits on 72 training prompts, selects
separately by method on 48 development prompts and retains all 96 final prompts across new templates: 66 use training names and 30 include unseen names. Role-reversed name pairs
remain grouped. Layer-input comparisons are 2→6, 4→8 and 6→10. CCA has an input-rank cap of 32; SVCCA retains 99% of training variance; requested transport ranks are 2/4/8/16;
ridge penalty is 0.1; sixteen same-norm random controls remain per layer pair.

The **published main study** records 12,816 outcomes, 12,888 model-forward rows and five additional play forward rows. The separate **historical fresh-CPU reproduction** compares 5,184 rows over 108 conditions using the first and last 24 final rows per condition. That subset reproduction is not the full lesson, and neither execution occurred during this migration.

The headline KL is `KL(native target-layer donor || transported)`, computed by summing over the complete vocabulary for each prompt, then averaging equally over the cohort. It
measures reproduction of an actual local intervention, whose other recipient positions remain unchanged. It is not desired-name accuracy, agreement with a necessarily correct
oracle or unrestricted donor-prompt reproduction. Correct-name plots use the whole-vocabulary winner, and the fixed token gallery retains wrong donors and unsuccessful maps. Small
absolute KL at an early layer must be interpreted against its almost unchanged baseline. Gray random-control points are draw means, not confidence intervals. The small, related
template/name cohorts do not justify independent-sample claims.

## Limits and remaining gates

Canonical generation, source/output parity and tiny CPU oracles establish migration fidelity and engineering behavior. They do not establish a new real-model execution, fresh
canonical full-notebook run, new scientific replication, human learner acceptance, readiness promotion or whole-course completion. The final data remain historical/monitored. No
checkpoint was loaded or downloaded; no training, GPU work or cluster job was launched. Frozen numerical/scientific choices were not silently upgraded for performance.

The canonical authoring/build entry point is now `infrastructure/extension/build.py 11.2`. Old builders/publishers are preserved historical tooling. The lead owns the guided checkpoint, workbook/navigation integration, shared config and exact preserved-directory exception. This lane does not mark the section written or accepted.

Proposed chapter description: **Build CCA, SVCCA and transport maps, then distinguish coordinate similarity, native-donor reproduction and correct-name prediction with explicit controls.**

## Validation record

Canonical generation without execution passed repeatedly:

```bash
UV_CACHE_DIR=/tmp/arena-alignment-migration/uv-cache \
uv run --no-sync --offline python infrastructure/extension/build.py 11.2
```

The final build reports **79 cells (20 code), 6,312 words of prose, six exercises**, and retains historical outputs through its strict same-code check. No `--execute` run was performed.

```bash
UV_CACHE_DIR=/tmp/arena-alignment-migration/uv-cache \
MPLCONFIGDIR=/tmp/arena-alignment-migration/mpl \
uv run --no-sync --offline pytest -q \
  tests/mastery/test_geometry_master_migration.py \
  tests/mastery/test_alignment_master_migration.py
```

Result: **41 passed in 6.43 seconds**, including all 24 geometry and seventeen alignment migration checks. The seventeen alignment tests separately passed in 3.93 seconds after the
final build, and in 2.98 seconds with PyTorch's ambient default pre-set to float64; an explicit post-run assertion verified dtype restoration. Nine independent alignment CPU
oracles exercise the generated visible functions. Other checks cover all source/archive/asset hashes, twenty code/AST/output parity pairs, exact reconstruction from the retained
lesson, every original explanatory line (including the interpretation answer), all eight student stubs, notebook schema, all controls/play/prediction prompts and
unrelated-directory import safety.

The static direct-experiment contract check also passes; this is structural evidence only. The new master, generated Python module and new test file compile successfully. Scoped
whitespace checks pass after removing the converter's extra EOF blank from the owned instruction page. An independent reviewer reran all seventeen focused checks (4.36 seconds) and
identified one documentation wording issue: vocabulary terms are **summed**, then prompt KLs are averaged. That wording is repaired; the metric implementation and scientific code
were unchanged.
