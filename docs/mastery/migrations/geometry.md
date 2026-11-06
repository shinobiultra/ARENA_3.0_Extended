# 11.1 full geometry lesson master migration

The canonical source is now `infrastructure/chapters/chapter11_representation_geometry/master_11_1.py`. It migrates the **entire published 11.1 lesson**, rather than substituting a
short PCA example. The generated exercises and solutions retain 35 code cells, eleven exercise groups covering twenty student functions, all thirteen reference figures, the
complete real-model study, negative findings and play cells. There are 118 generated exercise cells and about 8,646 words of prose before the shared guided-workbook link is added.

## Scope and plan

- **Current learner failure:** 11.1 already had a rich executed learner lesson, but its separate lesson publisher bypassed the canonical ARENA master pipeline. A small replacement would have lost its results and controls.
- **Target section claim:** identifying calendar words, recovering chronological geometry, and causally controlling answers are distinct claims. Student-built numerical and native-hook methods let the learner test each one.
- **Signature result:** retain the full exact-model answer heatmap and the Mistral coordinate/mean-ablation/full-donor comparison, alongside all-donor heatmaps and actual-token examples. Strong supervised decoding coexists with weak pure-coordinate answer transfer.
- **Student methods:** centered PCA; circular coordinates and phase decoding; balanced template centering; affine ridge probe and R²; kNN; chronological-distance statistic and permutation test; trustworthiness and continuity; Procrustes; minimum-norm coordinate edit; donor-subspace replacement; both mean-ablation conventions; native capture, position gathering, semantic-span alignment, residual replacement and restricted/full-vocabulary scoring.
- **Controls:** original/no-op, mean-only, full donor, five-PC donor, orthogonal toy plane, same-rank random donor planes, equal-norm random edits, class-order permutations, independent noise and shuffled labels; all seeds/settings/null draws and scientific split roles remain explicit.
- **Play cells:** exact-model angle/radius play; nonlinear-projection seed/setting gallery; real calendar intervention family/start/offset/shift/radius play; the continuous real angle/radius sweep remains visible.
- **Files:** the new master; its two generated notebooks, generated `solutions.py`, markdown and Python instruction page; thirteen copied `media/` assets; migration snapshot/parity records; `tests/mastery/test_geometry_master_migration.py`; this note. Shared routing/config changes belong to the lead.
- **Verification:** canonical build without `--execute`; source/output/asset parity; notebook-schema and own-answer completion tests; import safety from an unrelated working directory; sixteen tiny CPU mathematical/native-hook oracles; existing reproduction-cache and direct-experiment-contract regression tests.

Original ARENA chapters 0–4 are unchanged by this lane. All historical geometry prototypes, helpers, receipts, archives, failed reproductions and original instruction assets remain physically present. This task explicitly overrides AUTHORING's prototype-deletion guidance. No files were staged, committed, reverted or deleted by this lane.

## Historical evidence and exact provenance

The pre-migration published notebooks, generated reference module, old publisher source, current readable lesson, plotting source, publication receipt and instruction page were
snapshotted **before generation** in [`../evidence/geometry_master_migration/published_before_master.tar.gz`](../evidence/geometry_master_migration/published_before_master.tar.gz).
[`before.json`](../evidence/geometry_master_migration/before.json) records the archive SHA-256, every archived file hash, the current helper/source inventory, all thirteen figure
hashes and the eleven original exercise-cell indices. The archive was reopened and every member checked byte for byte.

The archived notebooks and page match the SHA-256 values in the unchanged historical `expected_outputs/learner_execution.json`. Completing the current `lesson.md` from its own
answers reproduces all 71 cells of the published solution notebook **exactly**, including code and prose. Thus the changed lesson-file hash is not silently treated as the recorded
source hash. Its normalized authored-cell content was checked independently.

The recorded `own_full_release_v3/execution.tar.gz` contains the actual completed learner. The current published solution has the same AST in all 35 code cells after removing one
documented execution-harness addition in setup: the historical executor redirects `assets_dir` to `/home/bitwise/.cache/arena_extended/geometry/full-own-answer-v3/figures` and
creates that directory. No model computation, scientific input or parameter is excluded from this comparison. The current plotting helper has exactly the same AST as the plotting
helper in that execution archive, despite different file bytes. Methods, method tests, data, study and protocol source hashes remain the recorded values.

Every generated solution cell has **identical nonblank lines and identical AST** to its archived published counterpart. Only blank-line normalization and final newlines differ.
[`output_parity.json`](../evidence/geometry_master_migration/output_parity.json) records all 35 old/new code hashes, AST hashes, output hashes and historical execution counts. This
justified carrying the original outputs and counts forward; the generated solutions retain 35 historical counts and 25 output-bearing code cells. The thirteen `media/` PNGs are
byte-identical copies of the publication assets. All figures were inspected as a contact sheet, with the actual-answer gallery, final controls and angle/radius grid also viewed
individually.

**These are retained historical reference results, not a fresh current Mistral execution.** That label lives in the master, both generated notebooks, the instruction page and the
parity sidecar. The old `learner_execution.json` certifies the archived publication it names; its old published-notebook hashes are not updated to impersonate a current execution.
Repeat canonical builds use the converter's strict same-code carry rule after this checked initial migration. The sidecar and master prose remain authoritative when shared
generation discards custom notebook metadata.

The published figures retain calendar-restricted metrics where originally plotted; accompanying prose and tables also retain full-vocabulary scoring. The new numeric forecast specifies the full-vocabulary metric explicitly. The recorded final cohort is historical and already monitored; it does not become untouched TEST through migration or the prediction workbook.

## Student-facing changes

The original introduction and full exposition are preserved, reorganized into the converter's title/header, introduction, learning-objective, six numbered parts and bonus
structure. Explanation and exercise headings are split into valid master cells. Every exercise contains visible solution code in the master, a working student stub, immediate
semantic tests and a properly fenced solution dropdown in the generated exercises. No taught method is replaced by an import from a hidden implementation.

Seven additive forecast/reflection pairs precede template centering, chronological-order nulls, exact-model intervention, real decoding transfer, real causal controls, nonlinear
generalization and continuous angle/radius results. Each asks for a method-specific ranking or numerical range, confidence, causal reasoning, exposure history, observed controls,
discrepancy and a changed-condition prediction. They contain no fabricated learner forecasts, observations or reflections. Existing expected outputs make this guided practice;
students who have seen them cannot claim unseen performance.

The generated Python module exposes the visible numerical and study functions on import. Notebook-only demonstrations, tests, model loading, global thread/style changes and
directory creation are protected by `MAIN` guards in Python. Notebook source retains the original computations because the converter removes those guards. Python-only root
discovery uses `__file__`, allowing import from an unrelated working directory. Existing `run_smoke_test`, `run_gpu_test` and `run_full_experiment` compatibility entry points are
preserved in a Python-only master cell; they never run on import. The smoke entry point uses the generated visible methods. Historical experiment/publisher scripts are preserved as
reference tooling; the canonical authoring/build entry point is now `infrastructure/extension/build.py 11.1`.

## Prediction case supplied to the lead

Forecast **desired full-vocabulary answer accuracy** of `pure_coordinate`, frozen donor shift +2, weekday layer-input 16, five PCA components and training-fitted probe. The cohort
is all 147 weekday final prompts: seven starts × seven offsets × three new templates. The metric is `mean(top_token_id == desired_calendar_token_id)`, with no successful-only
filtering and no replacement by calendar-restricted argmax. Compare author plane plus mean, full donor, mean-only, no-op, same-rank random plane, equal-norm random edit and
permuted calendar-order probes; random conditions retain all 32 draws per kind. The changed scientific condition is the 432-prompt month cohort at its frozen layer-input 10, with
the same +2 shift and metric. Answers belong to historical reference material, never to the pre-commit checkpoint manifest. The lead owns workbook/manifest integration.

## Execution limits and remaining gates

No large model was rerun, downloaded, trained or loaded. No CUDA work, GPU reservation or cluster job was launched. The recorded Mistral-7B-v0.1 revision remains
`27d67f1b5f57dc0953326b2601d68371d40ea8da`; its full historical execution used about 13.7 GiB CUDA memory on the tested RTX 5090 Laptop GPU. Current hardware/software reproduction
is unverified. The complete study remains expensive and uses frozen scientific choices, so performance/version upgrades were not silently folded into the protocol.

Canonical generation and bounded CPU oracles establish migration fidelity and engineering behavior. They are not a new scientific replication, fresh full-notebook execution,
learner acceptance, paper-equivalent reproduction or whole-extension completion. Nonlinear t-SNE remains transductive; reference exposure remains explicit; final data remain
historical/monitored. The existing numerical-reproduction failure evidence is retained.

Proposed chapter description: **Build geometry probes and native interventions to distinguish calendar decoding, circular structure and causal control, with full random and nonlinear-view controls.**

## Validation record

Validation results are recorded below after the targeted run. Shared notebook-schema normalization is owned by the lead: the original converter emitted code-only `execution_count` and `outputs` fields on markdown cells; the extension build now removes only empty such fields and validates the resulting notebooks.

The final bounded command was:

```bash
UV_CACHE_DIR=/tmp/arena-geometry-migration/uv-cache \
MPLCONFIGDIR=/tmp/arena-geometry-migration/mpl \
uv run --no-sync --offline pytest -q \
  tests/mastery/test_geometry_master_migration.py \
  tests/test_geometry_reproduction_cache.py \
  tests/test_course_surface_audit.py::test_native_reproduction_requires_validation_actual_model_and_called_entrypoint
```

Result: **33 passed in 4.98 seconds**. This includes sixteen independent CPU oracles against the **generated** visible functions, all twenty own-answer stubs, line-by-line
preservation of original teaching prose, exact reconstruction of the archived publication from the retained lesson, notebook schema validation and retained-source/output/asset
checks. Repeated `uv run --no-sync --offline python infrastructure/extension/build.py 11.1` generation passed and retained the previous outputs under its strict same-code check. No
`--execute` run was performed. The new master, generated Python module and migration test compile successfully; scoped whitespace checks pass after removing the converter's extra
final blank line from the owned instruction page.

A combined CPU batch later exposed ambient dtype coupling in the preserved native-hook oracle: its tiny model explicitly emits float32, while a different module had changed
PyTorch's process default to float64. The migration test wrapper now saves the ambient default, sets the historical float32 default for each oracle invocation and restores the
previous dtype in `finally`. Historical lesson/method/oracle files remain unchanged. The complete 24-test migration file passed both with the ordinary default (**4.63 seconds**)
and in a process pre-set to float64 (**3.97 seconds**); an explicit post-run assertion verified that float64 was restored.
